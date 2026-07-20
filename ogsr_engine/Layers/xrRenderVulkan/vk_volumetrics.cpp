// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).
//
// Froxel volumetric lighting — see vk_volumetrics.h. Two compute dispatches
// (inject -> integrate) into a 160x90x64 RGBA16F 3D grid; the composite is
// folded into the tonemap. Mirrors the compute/storage discipline of
// vk_clustered + the image-alloc pattern of vk_scene_color.

#include "stdafx.h"
#include "vk_volumetrics.h"
#include "vk_color_space.h"   // ColorSpace::LinearizeRGB — env colours are authored sRGB
#include "vk_buffer.h"            // CVulkanBuffer
#include "vk_command_buffer.h"    // CVulkanCommandManager::FRAMES_IN_FLIGHT
#include "vk_shaders.h"           // g_ShaderManager (SPIR-V loader)
#include "vk_image.h"             // VK::CreateImage / CreateImageView
#include "vk_compute_util.h"      // VK::CreateComputePipeline
#include "vk_shadow.h"            // ShadowMap — sun cascade maps + matrices (occlusion source)
#include "vk_pass_shadow.h"       // SpotShadow_TileOfLight — spot-pool tile per fog light
#include "vk_vsm.h"               // VSM atlas + page table + clipmap UBO (smooth occlusion, no cache tick)
#include "vk_clustered.h"         // Clustered::DeriveGridZ — ONE source of truth for the exp-Z grid
#include "vk_light.h"             // Lights::CollectFrame / GpuLight — P2 local lights in fog
#include "vk_profiler.h"          // VK::Prof zones + NameImage
#include "vk_gpu_particles.h"     // GPUParticles::SplatMedia — #6 GPU smoke → froxel media
#include "../../xr_3da/IGame_Persistent.h"   // g_pGamePersistent->Environment()
#include "../../xr_3da/Environment.h"        // CEnvDescriptorMixer (sun_dir/sun_color/ambient)
#include "../../xr_3da/device.h"             // Device.vCameraPosition
#include <cstring>
#include <cmath>

// r_vol knobs (global scope — a block-scope extern inside namespace VK would
// mangle to VK::ps_r_vol and miss the C-linkage console symbol → LNK2001).
extern int   ps_r_vol;            // master enable (0/1)
extern int   ps_r_vol_debug;      // debug view — also self-activates the path
extern float ps_r_vol_density;    // base extinction / scatter density
extern float ps_r_vol_height;     // height-fog falloff rate (0 = uniform fog)
extern float ps_r_vol_g;          // Henyey-Greenstein anisotropy (~0.6 forward)
extern float ps_r_vol_lights_g;   // local-light forward scatter (torch glare; sep. from sun r_vol_g)
extern float ps_r_torch_vol;      // fog shaft strength for handheld torches only (fixtures keep ×3)
extern float ps_r_vol_intensity;  // in-scatter brightness multiplier
extern float ps_r_vol_amb;        // indoor ambient floor (fraction of sky ambient kept under a roof)
extern float ps_r_vol_ambient;    // fog sky-fill tint scale (decoupled from surface r_ambient_floor)
extern float ps_r_vol_indoor;     // indoor density boost (thicken fog under a roof so small rooms show it)
extern float ps_r_vol_sun;        // sun-beam in-scatter boost (directional shaft pops through the ambient haze)
extern float ps_r_vol_lights;     // P2: local light in-scatter strength in fog
extern float ps_r_vol_noise;      // P3: animated dust/mist amount
extern float ps_r_vol_noise_scale;// P3: noise frequency
extern float ps_r_vol_noise_speed;// P3: dust drift speed
extern float ps_r_vol_soft;       // cascade shadow PCF blur radius (texels) — soft penumbra hides the cache tick
extern int   ps_r_vol_ta;         // temporal accumulation on/off
extern float ps_r_vol_ta_blend;   // EMA history weight
extern int   ps_r_vol_shadow;     // dedicated per-frame fog sun-shadow (continuous) vs cascade/VSM
extern float ps_r_sun_boost;      // same global sun multiplier the receivers use
extern float ps_r_ambient_floor;  // same flat ambient lift the receivers use
extern float ps_r_vol_smoke_inject;  // Stage-1 VMS: inject smoke-particle density into the froxel grid (0 = off)
extern float ps_r_vol_smoke_density;  // Stage-1 VMS: density mass per particle opacity (splat scale)
extern int   ps_r_vol_smoke_debug;    // Stage-1 VMS debug: 1 = isolate injected smoke, 2 = density heatmap
extern float ps_r_vol_smoke_footprint; // Stage-1.1 VMS: max splat footprint (froxel cells; 0 = point)
extern float ps_r_vol_smoke_shadow;       // Stage-2 VMS: smoke self-shadow strength (0 = off)
extern float ps_r_vol_smoke_shadow_step;  // Stage-2 VMS: self-shadow sun-march step (world m)
extern float ps_r_vol_smoke_dist;         // media cutoff (m) — same LOD zone CollectSmokeParticles uses
extern float ps_r_vol_smoke_dist_full;    // full-quality inner radius (m)
extern int   ps_r__detail_radius;         // caps the media cutoff (terrain detail bubble)
extern int   ps_r_light_occ;              // dynamic-light terrain/static occlusion (stops indoor lamps leaking into fog/smoke)
extern int   ps_r_atmo;                   // atmospheric scattering (Rayleigh+Mie aerial perspective) master
extern float ps_r_atmo_rayleigh;          // Rayleigh (blue distance) strength
extern float ps_r_atmo_mie;               // Mie (warm sun halo) strength
extern float ps_r_atmo_mie_g;             // Mie forward anisotropy g

namespace VK { namespace Vol {

namespace {
    constexpr u32 kFramesInFlight = CVulkanCommandManager::FRAMES_IN_FLIGHT;
    constexpr VkFormat kVolFormat = VK_FORMAT_R16G16B16A16_SFLOAT;

    // Stage-1 VMS smoke media grid. Smoke is low-frequency → a COARSE grid (half-res
    // of the fog grid) is plenty: ~9 MB accum + 4.7 MB media vs 76 MB at full res,
    // less atomic contention, cheaper splat. inject samples it trilinearly at the
    // SAME normalized frustum coords, so the resolution mismatch is invisible.
    constexpr u32 kSmokeX = 128, kSmokeY = 72, kSmokeZ = 64;
    constexpr u32 kSmokeCells = kSmokeX * kSmokeY * kSmokeZ;
    constexpr u32 kMaxSmokeParticles = 32768;   // cap per frame (silently clamped; logged)
    constexpr float kSmokeFixed = 4096.0f;       // atomic fixed-point scale (uint accum)

    // Splat push (camera basis + grid params; 112 B). Resolve push (32 B).
    struct SplatPush  { float camPos[4], camDir[4], camRightT[4], camTopT[4], zParams[4], dims[4], params[4]; };
    struct ResolvePush { s32 dims[4]; float params[4]; };

    // Vol UBO (std140 — all vec4/mat4, naturally 16-aligned). MUST match the
    // `Vol` block in vol_inject.comp.glsl / vol_integrate.comp.glsl.
    struct VolUBO {
        float camPos[4];       // xyz world camera pos
        float camDir[4];       // xyz camera forward (unit)
        float camRightT[4];    // xyz right * tan(fovX/2)
        float camTopT[4];      // xyz top   * tan(fovY/2)
        float sun_dir[4];      // xyz sun TRAVEL dir (downward); to-sun = -sun_dir
        float sun_color[4];    // rgb (sun_boost applied)
        float sky_ambient[4];  // rgb flat fill (ambient + floor)
        float sun_vp[16];      // far cascade VP
        float sun_near_vp[16]; // cascade 0 VP
        float sun_c1_vp[16];   // cascade 1 VP
        float gridParams[4];   // x=dimX y=dimY z=dimZ
        float zParams[4];      // x=near y=far z=log2(far/near)
        float fog[4];          // x=baseDensity y=heightBase z=heightFalloff w=HG_g
        float fog2[4];         // x=intensity y=ambient floor z=indoor boost w=sun-beam boost
        float rain_vp[16];     // top-down ortho VP for the sky-visibility (ambient) occlusion map
        // --- Temporal accumulation (r_vol_ta): jitter the froxel sample + reproject
        // the previous frame's scatter volume and EMA-blend → smooth, dense fog.
        float prevViewProj[16];
        float prevCamPos[4];   // xyz prev camera world pos, w = prev near
        float prevCamDir[4];   // xyz prev camera forward, w = prev log2(far/near)
        float temporal[4];     // xyz = froxel jitter (−0.5..0.5), w = history blend (0 = no temporal)
        float fog_shadow_vp[16]; // r_vol_shadow: dedicated per-frame fog sun-shadow VP
        // P2 — local lights in fog (nearest few; matches Lights::GpuLight layout).
        float lightParams[4];  // x = count, y = boost, z = spot-shadowed light idx (-1), w = point-shadowed idx (-1)
        struct { float pos[4]; float color[4]; float dir[4]; } lights[8];
        float noiseParams[4];  // P3: x = amount, y = scale, z = speed, w = time clock
        // Spot shadow POOL: per-tile view·proj (occlude each pooled light's fog
        // cone by ITS OWN tile — headlights and the flashlight all cut at once).
        // The per-light tile rides in lights[i].color[3] bits 2+ (tile+1, <<2).
        float spot_pool_vp[Lights::kMaxShadowSpots][16];
        float smokeParams[4];  // Stage-1 VMS: x = smoke-inject strength (0 = no injected smoke)
        float light_occ[4];    // r_light_occ: x = enable, y = bury bias, z = frag-below band, w = strength
        // Atmospheric scattering (r_atmo): Rayleigh (blue) + Mie (forward halo) →
        // aerial perspective. Appended last (integrate UBO stays a valid prefix).
        float atmo[4];         // x = enable, y = Rayleigh strength, z = Mie strength, w = Mie g
        float atmoR[4];        // rgb = Rayleigh scattering tint (blue-heavy), w unused
    };
    constexpr u32 kVolMaxLights = 8;
    constexpr VkDeviceSize kUboStride = (sizeof(VolUBO) + 255) & ~VkDeviceSize(255);

    bool s_inited = false, s_failed = false, s_ready = false;
    u32  s_generation = 0;

    // 3D volumes.
    VkImage       s_scatterImg = VK_NULL_HANDLE, s_integImg = VK_NULL_HANDLE;
    VmaAllocation s_scatterAlloc = VK_NULL_HANDLE, s_integAlloc = VK_NULL_HANDLE;
    VkImageView   s_scatterView = VK_NULL_HANDLE, s_integView = VK_NULL_HANDLE;
    VkSampler     s_sampler = VK_NULL_HANDLE;   // linear/clamp (trilinear composite)

    // Temporal history (r_vol_ta): last frame's blended scatter, sampled by inject
    // for the reproject + EMA. A dedicated volume + a per-frame copy from s_scatter
    // (simpler than ping-pong — bindings stay fixed, only layouts + the copy move).
    VkImage       s_histImg = VK_NULL_HANDLE;
    VmaAllocation s_histAlloc = VK_NULL_HANDLE;
    VkImageView   s_histView = VK_NULL_HANDLE;
    bool          s_haveHistory = false;   // false until the first frame fills it (alpha 0)
    u32           s_frame = 0;             // jitter sequence index

    // Previous frame's reprojection inputs (cached at the end of Execute).
    Fmatrix s_prevVP;
    Fvector s_prevCamPos{}, s_prevCamDir{};
    float   s_prevNear = 0.1f, s_prevLogFN = 8.0f;
    bool    s_prevValid = false;

    // Radical-inverse Halton — deterministic per-frame sub-froxel jitter (no RNG).
    float Halton(u32 i, u32 b) {
        float f = 1.0f, r = 0.0f;
        while (i > 0) { f /= float(b); r += f * float(i % b); i /= b; }
        return r;
    }

    // Per-slot UBO (host-visible, double-buffered: CPU writes off the GPU timeline).
    CVulkanBuffer s_ubo;
    u8*           s_uboMapped = nullptr;

    // Dummy SSBO — keeps the inject set's VSM page-table binding (8) valid when VSM
    // isn't ready (the shader has the access statically; it just isn't sampled).
    CVulkanBuffer s_dummySSBO;
    VkImageView   s_boundVsmAtlas = VK_NULL_HANDLE;   // per-frame rebind guard
    VkBuffer      s_boundVsmPT = VK_NULL_HANDLE, s_boundVsmUBO = VK_NULL_HANDLE;

    // Inject pipeline (set: 0=UBO 1-3=cascades 4=scatter[storage]).
    VkDescriptorSetLayout s_injSetL = VK_NULL_HANDLE;
    VkPipelineLayout      s_injLayout = VK_NULL_HANDLE;
    VkPipeline            s_injPipe = VK_NULL_HANDLE;
    VkDescriptorSet       s_injSet[kFramesInFlight] = {};

    // Integrate pipeline (set: 0=UBO 1=scatter[storage read] 2=integrated[storage write]).
    VkDescriptorSetLayout s_intSetL = VK_NULL_HANDLE;
    VkPipelineLayout      s_intLayout = VK_NULL_HANDLE;
    VkPipeline            s_intPipe = VK_NULL_HANDLE;
    VkDescriptorSet       s_intSet[kFramesInFlight] = {};

    // Stage-1 VMS smoke media: a resolved RGBA16F volume (rgb=albedo, a=density) that
    // inject samples; fed by a per-particle SPLAT (atomic accumulation SSBO) + RESOLVE.
    VkImage       s_smokeImg = VK_NULL_HANDLE;
    VmaAllocation s_smokeAlloc = VK_NULL_HANDLE;
    VkImageView   s_smokeView = VK_NULL_HANDLE;
    CVulkanBuffer s_smokeAccum;                              // device-local uint[cells*4] (atomic accumulation)
    CVulkanBuffer s_smokeParts[kFramesInFlight];             // host-visible per-particle upload ring
    u8*           s_smokePartsMapped[kFramesInFlight] = {};
    VkDescriptorSetLayout s_splatSetL = VK_NULL_HANDLE, s_resolveSetL = VK_NULL_HANDLE;
    VkPipelineLayout      s_splatLayout = VK_NULL_HANDLE, s_resolveLayout = VK_NULL_HANDLE;
    VkPipeline            s_splatPipe = VK_NULL_HANDLE, s_resolvePipe = VK_NULL_HANDLE;
    VkDescriptorSet       s_splatSet[kFramesInFlight] = {};  // per-slot (particle buffer differs by slot)
    VkDescriptorSet       s_resolveSet = VK_NULL_HANDLE;     // shared (accum + media)

    VkDescriptorPool s_pool = VK_NULL_HANDLE;

    // Defensive cascade/rain-view rebind: if ShadowMap ever recreates a view,
    // re-point the inject sets (rare; the device is idle when that happens).
    // [0]=cascade0 [1]=cascade1 [2]=far [3]=rain (sky-visibility / ambient occ).
    VkImageView s_boundCasc[4] = {};

    GridZParams s_gridZ{ 0.1f, 300.f, 8.0f };   // last Execute's exp-Z params (composite reads it)

    // ---- One-shot 16-byte-aligned image-memory barrier (matches SceneColor) ----
    void ImgBarrier(VkCommandBuffer cmd, VkImage img,
                    VkImageLayout oldL, VkImageLayout newL,
                    VkAccessFlags2 srcA, VkAccessFlags2 dstA,
                    VkPipelineStageFlags2 srcS, VkPipelineStageFlags2 dstS)
    {
        VkImageMemoryBarrier2 b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        b.srcStageMask = srcS; b.srcAccessMask = srcA;
        b.dstStageMask = dstS; b.dstAccessMask = dstA;
        b.oldLayout = oldL; b.newLayout = newL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = img;
        b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VkDependencyInfo di{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &di);
    }

    bool CreateVolume(VkImage& img, VmaAllocation& alloc, VkImageView& view,
                      u32 ex, u32 ey, u32 ez, const char* name)
    {
        VK::ImageDesc d;
        d.type   = VK_IMAGE_TYPE_3D;
        d.format = kVolFormat;
        d.extent = { ex, ey, ez };
        // TRANSFER_SRC/DST: the temporal path vkCmdCopyImage's scatter -> history
        // (Execute), which requires both the usage flags and the TRANSFER layouts.
        // Without these the copy + its layout barriers are invalid (VUID-01212/01213/
        // 06662/06663). integ never copies but the extra flags are harmless.
        d.usage  = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                 | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        d.name   = name;
        if (!VK::CreateImage(d, img, alloc)) return false;
        view = VK::CreateImageView(img, kVolFormat, VK_IMAGE_VIEW_TYPE_3D);
        return view != VK_NULL_HANDLE;
    }

    VkPipeline BuildComputePipe(const char* spv, VkPipelineLayout layout)
    {
        VkShaderModule cs = g_ShaderManager->Load(spv);
        if (!cs) { Msg("![VK Vol] %s load failed", spv); return VK_NULL_HANDLE; }
        return VK::CreateComputePipeline(cs, layout, spv);
    }
}

bool Wanted() { return ps_r_vol != 0 || ps_r_vol_debug != 0; }   // debug self-activates (mirrors r_clustered_debug)
bool Ready()  { return s_ready && !s_failed; }
VkImageView GetIntegratedView() { return s_integView; }
VkImageView GetScatterView()    { return s_scatterView; }
VkSampler   GetSampler()        { return s_sampler; }
u32         Generation()        { return s_generation; }
GridZParams GetGridZ()          { return s_gridZ; }

bool Init()
{
    if (s_inited) return s_ready;
    s_inited = true;

    if (!g_ShaderManager) g_ShaderManager = xr_new<VK::CVulkanSPIRVLoader>();

    // --- 3D volumes (eager: keeps the tonemap binding valid even with r_vol off).
    if (!CreateVolume(s_scatterImg, s_scatterAlloc, s_scatterView, kGridX, kGridY, kGridZ, "Vol.Scatter")) { s_failed = true; return false; }
    if (!CreateVolume(s_integImg,   s_integAlloc,   s_integView,   kGridX, kGridY, kGridZ, "Vol.Integrated")) { s_failed = true; return false; }
    if (!CreateVolume(s_histImg,    s_histAlloc,    s_histView,    kGridX, kGridY, kGridZ, "Vol.History")) { s_failed = true; return false; }
    if (!CreateVolume(s_smokeImg,   s_smokeAlloc,   s_smokeView,   kSmokeX, kSmokeY, kSmokeZ, "Vol.SmokeMedia")) { s_failed = true; return false; }

    // Linear/clamp sampler — trilinear composite (NEAREST z = banding).
    VkSamplerCreateInfo si{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod = 0.0f;
    if (vkCreateSampler(VulkanHW.m_Device, &si, nullptr, &s_sampler) != VK_SUCCESS) { Msg("![VK Vol] sampler failed"); s_failed = true; return false; }

    // Per-slot UBO (host-visible mapped).
    s_ubo.Create(kUboStride * kFramesInFlight, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
    s_uboMapped = static_cast<u8*>(s_ubo.Map());
    if (!s_uboMapped) { Msg("![VK Vol] UBO map failed"); s_failed = true; return false; }

    // Dummy SSBO — keeps the VSM page-table binding valid before VSM is ready.
    s_dummySSBO.Create(64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);

    // Stage-1 smoke: device-local atomic accumulation SSBO (uint[cells*4], cleared +
    // splatted + resolved each frame on the GPU) + a host-visible per-particle upload
    // ring (one buffer per in-flight slot, CPU writes off the GPU timeline).
    s_smokeAccum.Create(VkDeviceSize(kSmokeCells) * 4 * sizeof(u32),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    for (u32 i = 0; i < kFramesInFlight; ++i) {
        s_smokeParts[i].Create(VkDeviceSize(kMaxSmokeParticles) * sizeof(SmokeParticle),
                               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
        s_smokePartsMapped[i] = static_cast<u8*>(s_smokeParts[i].Map());
        if (!s_smokePartsMapped[i]) { Msg("![VK Vol] smoke particle buffer map failed"); s_failed = true; return false; }
    }

    // --- Inject set layout (0=UBO 1-3=cascades 4=scatter storage 5=rain/sky-vis 6=history sampler).
    {
        VkDescriptorSetLayoutBinding b[14]{};
        b[0] = { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,        1, VK_SHADER_STAGE_COMPUTE_BIT };
        b[1] = { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT };
        b[2] = { 2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT };
        b[3] = { 3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT };
        b[4] = { 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT };
        b[5] = { 5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT };
        b[6] = { 6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT };
        b[7] = { 7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT };  // VSM atlas
        b[8] = { 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         1, VK_SHADER_STAGE_COMPUTE_BIT };  // VSM page table
        b[9] = { 9, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,        1, VK_SHADER_STAGE_COMPUTE_BIT };  // VSM clipmap UBO
        b[10] = { 10, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT }; // fog sun-shadow
        b[11] = { 11, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT }; // spot (flashlight) shadow
        b[12] = { 12, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT }; // point (campfire) shadow cube
        b[13] = { 13, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT }; // Stage-1 smoke media
        VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        lci.bindingCount = 14; lci.pBindings = b;
        if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &s_injSetL) != VK_SUCCESS) { Msg("![VK Vol] inject set layout failed"); s_failed = true; return false; }
    }
    // --- Integrate set layout (0=UBO 1=scatter[read] 2=integrated[write]).
    {
        VkDescriptorSetLayoutBinding b[3]{};
        b[0] = { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT };
        b[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT };
        b[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT };
        VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        lci.bindingCount = 3; lci.pBindings = b;
        if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &s_intSetL) != VK_SUCCESS) { Msg("![VK Vol] integrate set layout failed"); s_failed = true; return false; }
    }
    // --- Stage-1 splat set layout (0=particles SSBO[read] 1=accum SSBO[read-modify-write]).
    {
        VkDescriptorSetLayoutBinding b[2]{};
        b[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT };
        b[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT };
        VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        lci.bindingCount = 2; lci.pBindings = b;
        if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &s_splatSetL) != VK_SUCCESS) { Msg("![VK Vol] splat set layout failed"); s_failed = true; return false; }
    }
    // --- Stage-1 resolve set layout (0=accum SSBO[read] 1=media image[write]).
    {
        VkDescriptorSetLayoutBinding b[2]{};
        b[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT };
        b[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT };
        VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        lci.bindingCount = 2; lci.pBindings = b;
        if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &s_resolveSetL) != VK_SUCCESS) { Msg("![VK Vol] resolve set layout failed"); s_failed = true; return false; }
    }

    // Pool: UBO 3*F, sampler 10*F (9 + Stage-1 smoke media on inject), storage image
    // 3*F+1 (+resolve media write), storage buffer 3*F+1 (VSM PT*F + splat particles+accum
    // *F + resolve accum). Sets: 2*F (inject+integrate) + F (per-slot splat) + 1 (resolve).
    VkDescriptorPoolSize ps[4] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         3 * kFramesInFlight },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 10 * kFramesInFlight },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          3 * kFramesInFlight + 1 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         3 * kFramesInFlight + 1 },
    };
    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets = 3 * kFramesInFlight + 1; pci.poolSizeCount = 4; pci.pPoolSizes = ps;
    if (vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &s_pool) != VK_SUCCESS) { Msg("![VK Vol] pool failed"); s_failed = true; return false; }

    // Allocate per-slot sets.
    {
        VkDescriptorSetLayout iL[kFramesInFlight], tL[kFramesInFlight];
        for (u32 i = 0; i < kFramesInFlight; ++i) { iL[i] = s_injSetL; tL[i] = s_intSetL; }
        VkDescriptorSetAllocateInfo dai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        dai.descriptorPool = s_pool;
        dai.descriptorSetCount = kFramesInFlight; dai.pSetLayouts = iL;
        if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, s_injSet) != VK_SUCCESS) { Msg("![VK Vol] inject set alloc failed"); s_failed = true; return false; }
        dai.pSetLayouts = tL;
        if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, s_intSet) != VK_SUCCESS) { Msg("![VK Vol] integrate set alloc failed"); s_failed = true; return false; }
    }
    // Stage-1 smoke sets: per-slot splat (its slot's particle buffer) + one resolve.
    {
        VkDescriptorSetLayout sL[kFramesInFlight];
        for (u32 i = 0; i < kFramesInFlight; ++i) sL[i] = s_splatSetL;
        VkDescriptorSetAllocateInfo dai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        dai.descriptorPool = s_pool; dai.descriptorSetCount = kFramesInFlight; dai.pSetLayouts = sL;
        if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, s_splatSet) != VK_SUCCESS) { Msg("![VK Vol] splat set alloc failed"); s_failed = true; return false; }
        dai.descriptorSetCount = 1; dai.pSetLayouts = &s_resolveSetL;
        if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, &s_resolveSet) != VK_SUCCESS) { Msg("![VK Vol] resolve set alloc failed"); s_failed = true; return false; }
    }

    // Write the stable descriptors (UBO per slot + storage images). Cascade
    // samplers (inject 1-3) are written in Execute (and rebind if they change).
    for (u32 i = 0; i < kFramesInFlight; ++i) {
        VkDescriptorBufferInfo ubi{ s_ubo.GetHandle(), kUboStride * i, sizeof(VolUBO) };
        VkDescriptorImageInfo  scatterI{ VK_NULL_HANDLE, s_scatterView, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo  integI{ VK_NULL_HANDLE, s_integView, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo  histI{ s_sampler, s_histView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        // VSM fallbacks (rebound per-frame to the real atlas/PT/UBO once VSM is ready).
        VkDescriptorImageInfo  vAtlasI{ ShadowMap::GetSampler(), ShadowMap::GetView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorBufferInfo vPtI{ s_dummySSBO.GetHandle(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo vUboI{ s_ubo.GetHandle(), kUboStride * i, 192 };
        VkDescriptorImageInfo  fogShI{ ShadowMap::GetSampler(), ShadowMap::GetFogShadowView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        // Fog beam samples the spot+GRASS map (blades cut the beam); surfaces keep
        // the clean spot map so grass doesn't blanket the ground's light pool.
        VkDescriptorImageInfo  spotShI{ ShadowMap::GetSampler(), ShadowMap::GetSpotBeamView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo  pointShI{ ShadowMap::GetSampler(), ShadowMap::GetPointCubeView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo  smokeI{ s_sampler, s_smokeView, VK_IMAGE_LAYOUT_GENERAL };   // Stage-1 smoke media (sampled from GENERAL)

        VkWriteDescriptorSet w[13]{};
        // inject: UBO(0) + scatter storage(4) + history sampler(6) + VSM atlas(7)/PT(8)/clipmap(9)
        w[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; w[0].dstSet = s_injSet[i]; w[0].dstBinding = 0; w[0].descriptorCount = 1; w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[0].pBufferInfo = &ubi;
        w[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; w[1].dstSet = s_injSet[i]; w[1].dstBinding = 4; w[1].descriptorCount = 1; w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[1].pImageInfo = &scatterI;
        w[5] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; w[5].dstSet = s_injSet[i]; w[5].dstBinding = 6; w[5].descriptorCount = 1; w[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[5].pImageInfo = &histI;
        w[6] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; w[6].dstSet = s_injSet[i]; w[6].dstBinding = 7; w[6].descriptorCount = 1; w[6].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[6].pImageInfo = &vAtlasI;
        w[7] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; w[7].dstSet = s_injSet[i]; w[7].dstBinding = 8; w[7].descriptorCount = 1; w[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[7].pBufferInfo = &vPtI;
        w[8] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; w[8].dstSet = s_injSet[i]; w[8].dstBinding = 9; w[8].descriptorCount = 1; w[8].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[8].pBufferInfo = &vUboI;
        w[9] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; w[9].dstSet = s_injSet[i]; w[9].dstBinding = 10; w[9].descriptorCount = 1; w[9].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[9].pImageInfo = &fogShI;
        w[10] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; w[10].dstSet = s_injSet[i]; w[10].dstBinding = 11; w[10].descriptorCount = 1; w[10].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[10].pImageInfo = &spotShI;
        w[11] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; w[11].dstSet = s_injSet[i]; w[11].dstBinding = 12; w[11].descriptorCount = 1; w[11].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[11].pImageInfo = &pointShI;
        w[12] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; w[12].dstSet = s_injSet[i]; w[12].dstBinding = 13; w[12].descriptorCount = 1; w[12].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[12].pImageInfo = &smokeI;
        // integrate: UBO(0) + scatter read(1) + integrated write(2)
        w[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; w[2].dstSet = s_intSet[i]; w[2].dstBinding = 0; w[2].descriptorCount = 1; w[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[2].pBufferInfo = &ubi;
        w[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; w[3].dstSet = s_intSet[i]; w[3].dstBinding = 1; w[3].descriptorCount = 1; w[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[3].pImageInfo = &scatterI;
        w[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; w[4].dstSet = s_intSet[i]; w[4].dstBinding = 2; w[4].descriptorCount = 1; w[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[4].pImageInfo = &integI;
        vkUpdateDescriptorSets(VulkanHW.m_Device, 13, w, 0, nullptr);
    }

    // Stage-1 smoke descriptor writes: per-slot splat (its particle buffer + the
    // shared accum SSBO) + the resolve set (accum read + media storage write).
    {
        VkDescriptorBufferInfo accumI{ s_smokeAccum.GetHandle(), 0, VK_WHOLE_SIZE };
        for (u32 i = 0; i < kFramesInFlight; ++i) {
            VkDescriptorBufferInfo partI{ s_smokeParts[i].GetHandle(), 0, VK_WHOLE_SIZE };
            VkWriteDescriptorSet w[2]{};
            w[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; w[0].dstSet = s_splatSet[i]; w[0].dstBinding = 0; w[0].descriptorCount = 1; w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[0].pBufferInfo = &partI;
            w[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; w[1].dstSet = s_splatSet[i]; w[1].dstBinding = 1; w[1].descriptorCount = 1; w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[1].pBufferInfo = &accumI;
            vkUpdateDescriptorSets(VulkanHW.m_Device, 2, w, 0, nullptr);
        }
        VkDescriptorImageInfo mediaStoreI{ VK_NULL_HANDLE, s_smokeView, VK_IMAGE_LAYOUT_GENERAL };
        VkWriteDescriptorSet w[2]{};
        w[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; w[0].dstSet = s_resolveSet; w[0].dstBinding = 0; w[0].descriptorCount = 1; w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[0].pBufferInfo = &accumI;
        w[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; w[1].dstSet = s_resolveSet; w[1].dstBinding = 1; w[1].descriptorCount = 1; w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[1].pImageInfo = &mediaStoreI;
        vkUpdateDescriptorSets(VulkanHW.m_Device, 2, w, 0, nullptr);
    }

    // Pipeline layouts + pipelines.
    {
        VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        plci.setLayoutCount = 1; plci.pSetLayouts = &s_injSetL;
        if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_injLayout) != VK_SUCCESS) { Msg("![VK Vol] inject layout failed"); s_failed = true; return false; }
        plci.pSetLayouts = &s_intSetL;
        if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_intLayout) != VK_SUCCESS) { Msg("![VK Vol] integrate layout failed"); s_failed = true; return false; }
    }
    s_injPipe = BuildComputePipe("vol_inject.comp.spv",    s_injLayout);
    s_intPipe = BuildComputePipe("vol_integrate.comp.spv", s_intLayout);
    if (!s_injPipe || !s_intPipe) { s_failed = true; return false; }

    // Stage-1 smoke splat + resolve (push-constant driven, no UBO).
    {
        VkPushConstantRange pcS{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(SplatPush) };
        VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        plci.setLayoutCount = 1; plci.pSetLayouts = &s_splatSetL;
        plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcS;
        if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_splatLayout) != VK_SUCCESS) { Msg("![VK Vol] splat layout failed"); s_failed = true; return false; }
        VkPushConstantRange pcR{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ResolvePush) };
        plci.pSetLayouts = &s_resolveSetL; plci.pPushConstantRanges = &pcR;
        if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_resolveLayout) != VK_SUCCESS) { Msg("![VK Vol] resolve layout failed"); s_failed = true; return false; }
    }
    s_splatPipe   = BuildComputePipe("vol_splat.comp.spv",   s_splatLayout);
    s_resolvePipe = BuildComputePipe("vol_resolve.comp.spv", s_resolveLayout);
    if (!s_splatPipe || !s_resolvePipe) { s_failed = true; return false; }

    // One-time UNDEFINED -> SHADER_READ so the tonemap binding is valid before the
    // first Execute (and stays valid on the menu / with r_vol off — never sampled).
    if (VkCommandBuffer cmd = VulkanHW.BeginSingleTimeCommands()) {
        ImgBarrier(cmd, s_scatterImg, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   0, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                   VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
        ImgBarrier(cmd, s_integImg, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   0, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                   VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
        // History starts SHADER_READ so inject's first sample (alpha 0) is valid.
        ImgBarrier(cmd, s_histImg, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   0, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                   VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
        // Smoke media stays GENERAL for its life (resolve writes it as a storage image,
        // inject samples it from GENERAL); never sampled until the first splat (gated).
        ImgBarrier(cmd, s_smokeImg, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                   0, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                   VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
        VulkanHW.EndSingleTimeCommands(cmd);
    }

    s_ready = true;
    ++s_generation;
    Msg("[VK Vol] init OK — froxel grid %ux%ux%u (RGBA16F x3 = %.1f MB), temporal accumulation, r_vol gates compute",
        kGridX, kGridY, kGridZ, double(kGridX * kGridY * kGridZ * 8 * 3) / (1024.0 * 1024.0));
    return true;
}

void Execute(VkCommandBuffer cmd, const ProjTerms& pt, u32 slot,
             const SmokeParticle* smoke, u32 smokeCount)
{
    if (!s_ready || !Wanted()) return;
    if (slot >= kFramesInFlight) slot = 0;

    // Stage-1 smoke: clamp the upload to capacity; gate the splat + inject sample on
    // the knob (or the debug view, which also drives the path). smokeActive false →
    // no splat/resolve, smokeParams 0 → inject skips it.
    const u32  smokeN      = (smoke && smokeCount) ? ((smokeCount < kMaxSmokeParticles) ? smokeCount : kMaxSmokeParticles) : 0u;
    // #6: GPU-routed smoke has no CPU particles — its media comes from a GPU-side
    // splat of the particle pool (recorded below, same accum SSBO). The GPU alive
    // count is unknown on the CPU, so when the GP pool is live the splat+resolve
    // run every frame (bounded: one early-out dispatch + the coarse-grid resolve).
    const bool gpSplat     = GPUParticles::WantsMediaSplat();
    const bool smokeActive = (ps_r_vol_smoke_inject > 0.0f || ps_r_vol_smoke_debug != 0) && (smokeN > 0u || gpSplat);

    // exp-Z grid — ONE source of truth shared with the clustered cull, so the
    // froxel Z mapping is identical to the inverse the composite uses.
    const Clustered::GridZ gz = Clustered::DeriveGridZ(pt);
    s_gridZ = { gz.nearZ, gz.farZ, gz.logFarNear };

    // ---- Fill this slot's UBO (off the GPU timeline; the slot's fence cleared it).
    VolUBO ub{};
    const Fvector& eye = Device.vCameraPosition;
    ub.camPos[0] = eye.x; ub.camPos[1] = eye.y; ub.camPos[2] = eye.z;
    ub.camDir[0] = pt.dir.x; ub.camDir[1] = pt.dir.y; ub.camDir[2] = pt.dir.z;
    ub.camRightT[0] = pt.right.x * pt.tanX; ub.camRightT[1] = pt.right.y * pt.tanX; ub.camRightT[2] = pt.right.z * pt.tanX;
    ub.camTopT[0]   = pt.top.x   * pt.tanY; ub.camTopT[1]   = pt.top.y   * pt.tanY; ub.camTopT[2]   = pt.top.z   * pt.tanY;

    // Sun + ambient from the SAME env source the receivers use (so the fog colour
    // tracks the lit surfaces). Neutral fallback before a level loads.
    ub.sun_dir[0] = 0.f; ub.sun_dir[1] = -1.f; ub.sun_dir[2] = 0.f;
    ub.sun_color[0] = ub.sun_color[1] = ub.sun_color[2] = 0.6f;
    // Fog ambient tracks the ENV ambient (dark/blue at night) × r_vol_ambient — NOT
    // the surface receiver floor (r_ambient_floor). That flat white floor lifts lit
    // geometry off black, but in the fog it integrates along the sightline into a
    // whitish haze that self-brightens the whole night scene. Scaling the real env
    // ambient instead keeps night air genuinely dark.
    ub.sky_ambient[0] = ub.sky_ambient[1] = ub.sky_ambient[2] = 0.05f;
    if (g_pGamePersistent) {
        if (auto* E = g_pGamePersistent->Environment().CurrentEnv) {
            ub.sun_dir[0] = E->sun_dir.x; ub.sun_dir[1] = E->sun_dir.y; ub.sun_dir[2] = E->sun_dir.z;
            ub.sun_color[0] = E->sun_color.x;
            ub.sun_color[1] = E->sun_color.y;
            ub.sun_color[2] = E->sun_color.z;
            ub.sky_ambient[0] = E->ambient.x;
            ub.sky_ambient[1] = E->ambient.y;
            ub.sky_ambient[2] = E->ambient.z;
        }
    }
    // Decode, THEN scale — the r_sun_boost / r_vol_ambient multiplies used to be fused
    // into the assignments above, which would have applied them on the wrong side of
    // the sRGB curve. Split so the fallback and the real env values share one
    // conversion and the knobs keep meaning "scale the radiance".
    ColorSpace::LinearizeRGB(ub.sun_color);
    ColorSpace::LinearizeRGB(ub.sky_ambient);
    for (int c = 0; c < 3; ++c) {
        ub.sun_color[c]   *= ps_r_sun_boost;
        ub.sky_ambient[c] *= ps_r_vol_ambient;
    }

    memcpy(ub.sun_vp,      &ShadowMap::GetLightVP(),    sizeof(ub.sun_vp));
    memcpy(ub.sun_near_vp, &ShadowMap::GetCascadeVP(0), sizeof(ub.sun_near_vp));
    memcpy(ub.sun_c1_vp,   &ShadowMap::GetCascadeVP(1), sizeof(ub.sun_c1_vp));

    ub.gridParams[0] = float(kGridX); ub.gridParams[1] = float(kGridY); ub.gridParams[2] = float(kGridZ);
    // Occlusion source enum: 2 = dedicated per-frame fog sun-shadow (CONTINUOUS, no
    // cache tick — r_vol_shadow), 1 = VSM atlas (smooth-ish), 0 = cascade (default).
    // r_vol_shadow wins; each falls back to the cascade where it has no coverage.
    ub.gridParams[3] = ps_r_vol_shadow ? 2.0f : (VSM::AtlasReady() ? 1.0f : 0.0f);
    ub.zParams[0] = gz.nearZ; ub.zParams[1] = gz.farZ; ub.zParams[2] = gz.logFarNear; ub.zParams[3] = ps_r_vol_soft;
    // Height fog anchored just below eye level (ground unknown per-level): full
    // density below, exp falloff above. r_vol_height 0 = uniform fog.
    ub.fog[0] = ps_r_vol_density;
    ub.fog[1] = eye.y - 2.0f;
    ub.fog[2] = ps_r_vol_height;
    ub.fog[3] = ps_r_vol_g;
    ub.fog2[0] = ps_r_vol_intensity;
    ub.fog2[1] = ps_r_vol_amb;     // indoor ambient floor (skyVis remap in the shader)
    ub.fog2[2] = ps_r_vol_indoor;  // indoor density boost (×(1+boost) under a roof)
    ub.fog2[3] = ps_r_vol_sun;     // sun-beam in-scatter boost (directional shaft)
    // Sky-visibility map (top-down statics depth) → the inject occludes the
    // unconditional ambient term so interiors don't glow with outdoor haze.
    memcpy(ub.rain_vp, &ShadowMap::GetRainVP(), sizeof(ub.rain_vp));
    memcpy(ub.fog_shadow_vp, &ShadowMap::GetFogShadowVP(), sizeof(ub.fog_shadow_vp));

    // P2 — nearest local lights (flashlight/lamps/campfires) so they glow/cone in the
    // fog. Reuse the same per-frame collection EnvLight/shadows use (nearest-first).
    {
        const auto& FL = Lights::CollectFrame(eye);
        const u32 nL = (FL.count < kVolMaxLights) ? FL.count : kVolMaxLights;
        ub.lightParams[0] = float(nL);
        ub.lightParams[1] = ps_r_vol_lights;
        // z: local-light forward-scatter anisotropy (separate from the sun's
        // r_vol_g) — a gentler peak so a torch shone at the camera doesn't spike
        // into a cold-white searchlight glare. Was unused (-1); harmless if an
        // old shader ignores it (localLights clamps g into hgPhase's valid range).
        ub.lightParams[2] = ps_r_vol_lights_g;
        ub.lightParams[3] = (FL.pointIdx >= 0 && (u32)FL.pointIdx < nL) ? float(FL.pointIdx) : -1.f;
        for (u32 i = 0; i < nL; ++i) {
            memcpy(ub.lights[i].pos,   FL.gpu[i].pos,   sizeof(float) * 4);
            memcpy(ub.lights[i].color, FL.gpu[i].color, sizeof(float) * 4);
            memcpy(ub.lights[i].dir,   FL.gpu[i].dir,   sizeof(float) * 4);
            // Handheld/worn torches (CTorch head-lamp, weapon light) vs fixtures
            // (hanging lamps, searchlights). A torch is a SMALL worn light, not a
            // searchlight — so it does NOT get the ×3 lamp beam boost; instead its
            // fog shaft is scaled by r_torch_vol (premultiplied into the inject's
            // OWN colour copy — surface lighting is a separate UBO, untouched).
            // This is what stops an NPC head-lamp shone at the camera reading as a
            // projector while overhead lamps keep their punchy beam.
            if (FL.flashFlag[i]) {
                ub.lights[i].color[0] *= ps_r_torch_vol;
                ub.lights[i].color[1] *= ps_r_torch_vol;
                ub.lights[i].color[2] *= ps_r_torch_vol;
            }
            // Volumetric-flagged FIXTURES (R4 renders a visible BEAM for these —
            // pole lamps, car headlights): +2 on the spot flag, the inject boosts
            // their fog in-scatter so the shaft actually reads. Torches excluded.
            else if (FL.volFlag[i]) ub.lights[i].color[3] += 2.0f;
            // Pool index (+1, <<2): a light is spot OR point, so the same bits
            // carry the spot tile or the point cube — every pooled light's fog
            // is cut by its own map, not just the single budget winner.
            const int tile = SpotShadow_TileOfLight(FL.src[i]);
            if (tile >= 0) ub.lights[i].color[3] += float((tile + 1) << 2);
            else {
                const int cube = PointShadow_CubeOfLight(FL.src[i]);
                if (cube >= 0) ub.lights[i].color[3] += float((cube + 1) << 2);
            }
        }
    }
    for (u32 t = 0; t < Lights::kMaxShadowSpots; ++t)
        memcpy(ub.spot_pool_vp[t], &ShadowMap::GetSpotTileVP(t), sizeof(ub.spot_pool_vp[t]));

    // P3 — animated dust/mist: amount/scale/speed + a wall-clock for the drift.
    ub.noiseParams[0] = ps_r_vol_noise;
    ub.noiseParams[1] = ps_r_vol_noise_scale;
    ub.noiseParams[2] = ps_r_vol_noise_speed;
    ub.noiseParams[3] = float(Device.dwTimeGlobal) * 0.001f;   // seconds
    // Stage-1 gate: force a non-zero sample strength when debug-isolating with inject 0.
    ub.smokeParams[0] = smokeActive ? ((ps_r_vol_smoke_inject > 0.0f) ? ps_r_vol_smoke_inject : 1.0f) : 0.0f;
    ub.smokeParams[1] = float(ps_r_vol_smoke_debug);   // 0 off / 1 isolate colour / 2 density heatmap
    ub.smokeParams[2] = ps_r_vol_smoke_shadow;         // Stage-2 self-shadow strength
    ub.smokeParams[3] = ps_r_vol_smoke_shadow_step;    // Stage-2 self-shadow march step (m)
    // Dynamic-light terrain occlusion (r_light_occ) — SAME values the forward shaders
    // use (vk_env_light) so the fog/smoke local lights match the surfaces: indoor lamps
    // buried under a roof don't leak out and light the smoke through walls.
    ub.light_occ[0] = ps_r_light_occ ? 1.0f : 0.0f;
    ub.light_occ[1] = 0.001f;   // bury bias (NDC-z)
    ub.light_occ[2] = 0.006f;   // frag-below band
    ub.light_occ[3] = 1.0f;     // strength

    // Atmospheric scattering (r_atmo) — Rayleigh + Mie aerial perspective in the fog.
    ub.atmo[0] = ps_r_atmo ? 1.0f : 0.0f;
    ub.atmo[1] = ps_r_atmo_rayleigh;
    ub.atmo[2] = ps_r_atmo_mie;
    ub.atmo[3] = ps_r_atmo_mie_g;
    // Physically-based Rayleigh tint (β ratio 5.8/13.5/33.1e-6 normalized → blue-heavy).
    ub.atmoR[0] = 0.18f; ub.atmoR[1] = 0.41f; ub.atmoR[2] = 1.0f; ub.atmoR[3] = 0.0f;

    // ---- Temporal accumulation (r_vol_ta): sub-froxel Halton jitter + the prev
    // frame's reprojection inputs. alpha 0 until we have a valid history frame.
    const bool taOn = (ps_r_vol_ta != 0) && s_prevValid && s_haveHistory;
    if (ps_r_vol_ta != 0) {
        ub.temporal[0] = Halton(s_frame, 2) - 0.5f;
        ub.temporal[1] = Halton(s_frame, 3) - 0.5f;
        ub.temporal[2] = Halton(s_frame, 5) - 0.5f;
    }
    ub.temporal[3] = taOn ? ps_r_vol_ta_blend : 0.0f;
    memcpy(ub.prevViewProj, &s_prevVP, sizeof(ub.prevViewProj));
    ub.prevCamPos[0] = s_prevCamPos.x; ub.prevCamPos[1] = s_prevCamPos.y; ub.prevCamPos[2] = s_prevCamPos.z; ub.prevCamPos[3] = s_prevNear;
    ub.prevCamDir[0] = s_prevCamDir.x; ub.prevCamDir[1] = s_prevCamDir.y; ub.prevCamDir[2] = s_prevCamDir.z; ub.prevCamDir[3] = s_prevLogFN;

    memcpy(s_uboMapped + size_t(slot) * kUboStride, &ub, sizeof(ub));

    // ---- Rebind the cascade + rain samplers if ShadowMap recreated them (defensive).
    VkImageView rainV = ShadowMap::GetRainView();
    if (!rainV) rainV = ShadowMap::GetView();   // fallback keeps the descriptor valid
    VkImageView casc[4] = { ShadowMap::GetCascadeView(0), ShadowMap::GetCascadeView(1), ShadowMap::GetView(), rainV };
    if (casc[0] != s_boundCasc[0] || casc[1] != s_boundCasc[1] || casc[2] != s_boundCasc[2] || casc[3] != s_boundCasc[3]) {
        VkSampler ss = ShadowMap::GetSampler();
        const u32 bindOf[4] = { 1, 2, 3, 5 };   // cascade0/1/far at 1-3, rain at 5
        for (u32 i = 0; i < kFramesInFlight; ++i) {
            VkDescriptorImageInfo ci[4] = {
                { ss, casc[0], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
                { ss, casc[1], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
                { ss, casc[2], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
                { ss, casc[3], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
            };
            VkWriteDescriptorSet w[4]{};
            for (u32 k = 0; k < 4; ++k) {
                w[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[k].dstSet = s_injSet[i];
                w[k].dstBinding = bindOf[k]; w[k].descriptorCount = 1;
                w[k].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[k].pImageInfo = &ci[k];
            }
            vkUpdateDescriptorSets(VulkanHW.m_Device, 4, w, 0, nullptr);
        }
        s_boundCasc[0] = casc[0]; s_boundCasc[1] = casc[1]; s_boundCasc[2] = casc[2]; s_boundCasc[3] = casc[3];
    }

    // ---- Rebind the VSM atlas / page table / clipmap UBO when VSM is ready (the
    // SMOOTH occlusion source — no cascade cache tick). Last frame's atlas content is
    // valid here (world-anchored toroidal slots) + this frame's clipmap params.
    if (VSM::AtlasReady()) {
        VkImageView atlas = VSM::GetAtlasView();
        VkBuffer    pt    = VSM::GetPageTableHandle();
        VkBuffer    cubo  = VSM::GetUBOHandle();
        if (atlas && pt && cubo &&
            (atlas != s_boundVsmAtlas || pt != s_boundVsmPT || cubo != s_boundVsmUBO)) {
            VkDescriptorImageInfo  aI{ VSM::GetSampler(), atlas, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkDescriptorBufferInfo pI{ pt, 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo uI{ cubo, 0, VK_WHOLE_SIZE };
            for (u32 i = 0; i < kFramesInFlight; ++i) {
                VkWriteDescriptorSet w[3]{};
                w[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; w[0].dstSet = s_injSet[i]; w[0].dstBinding = 7; w[0].descriptorCount = 1; w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[0].pImageInfo = &aI;
                w[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; w[1].dstSet = s_injSet[i]; w[1].dstBinding = 8; w[1].descriptorCount = 1; w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[1].pBufferInfo = &pI;
                w[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; w[2].dstSet = s_injSet[i]; w[2].dstBinding = 9; w[2].descriptorCount = 1; w[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[2].pBufferInfo = &uI;
                vkUpdateDescriptorSets(VulkanHW.m_Device, 3, w, 0, nullptr);
            }
            s_boundVsmAtlas = atlas; s_boundVsmPT = pt; s_boundVsmUBO = cubo;
        }
    }

    // Make last frame's occluder writes visible to this frame's COMPUTE sample: the
    // cascade (graphics raster) AND the VSM atlas/page-table (graphics + compute).
    {
        VkMemoryBarrier2 mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        mb.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        mb.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_SHADER_READ_BIT;
        mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mb.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        VkDependencyInfo di{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        di.memoryBarrierCount = 1; di.pMemoryBarriers = &mb;
        vkCmdPipelineBarrier2(cmd, &di);
    }

    // ---- Stage-1 VMS: SPLAT smoke particles → accum SSBO → RESOLVE to the media
    // volume the inject below samples. Particles were simulated in CollectVisuals
    // (before any pass), so their data is valid here. Skipped when disabled / no smoke.
    if (smokeActive) {
        const int z = Prof::ZoneBegin(cmd, "VolSmoke");
        if (ps_r_vol_smoke_debug && (s_frame % 120u) == 0u)
            Msg("[VK Vol] smoke inject: %u particles (debug %d, density %.2f)", smokeN, ps_r_vol_smoke_debug, ps_r_vol_smoke_density);
        if (smokeN)
            memcpy(s_smokePartsMapped[slot], smoke, size_t(smokeN) * sizeof(SmokeParticle));

        // accum + media are SHARED (not per-slot); with frames in flight, order the
        // PREVIOUS frame's reads (resolve's accum read, inject's media sample) before
        // this frame's overwrites (the clear + resolve store). One global barrier
        // covers both — it spans submissions on the same queue, and media stays GENERAL
        // (no layout change), so a memory barrier is enough.
        {
            VkMemoryBarrier2 mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            mb.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            mb.dstStageMask  = VK_PIPELINE_STAGE_2_CLEAR_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            mb.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            VkDependencyInfo di{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO }; di.memoryBarrierCount = 1; di.pMemoryBarriers = &mb;
            vkCmdPipelineBarrier2(cmd, &di);
        }

        // Clear the accumulation SSBO, then make the clear visible to the splat.
        vkCmdFillBuffer(cmd, s_smokeAccum.GetHandle(), 0, VK_WHOLE_SIZE, 0);
        {
            VkMemoryBarrier2 mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            mb.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;     mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT; mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            VkDependencyInfo di{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO }; di.memoryBarrierCount = 1; di.pMemoryBarriers = &mb;
            vkCmdPipelineBarrier2(cmd, &di);
        }

        // SPLAT — one thread per particle, atomic-add into the accum SSBO.
        SplatPush sp{};
        sp.camPos[0]    = eye.x;             sp.camPos[1]    = eye.y;             sp.camPos[2]    = eye.z;
        sp.camDir[0]    = pt.dir.x;          sp.camDir[1]    = pt.dir.y;          sp.camDir[2]    = pt.dir.z;
        sp.camRightT[0] = pt.right.x*pt.tanX; sp.camRightT[1] = pt.right.y*pt.tanX; sp.camRightT[2] = pt.right.z*pt.tanX;
        sp.camTopT[0]   = pt.top.x*pt.tanY;   sp.camTopT[1]   = pt.top.y*pt.tanY;   sp.camTopT[2]   = pt.top.z*pt.tanY;
        sp.zParams[0]   = gz.nearZ; sp.zParams[1] = gz.farZ; sp.zParams[2] = gz.logFarNear;
        sp.dims[0] = float(kSmokeX); sp.dims[1] = float(kSmokeY); sp.dims[2] = float(kSmokeZ); sp.dims[3] = kSmokeFixed;
        sp.params[0] = float(smokeN); sp.params[1] = ps_r_vol_smoke_density; sp.params[2] = ps_r_vol_smoke_footprint;
        if (smokeN) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_splatPipe);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_splatLayout, 0, 1, &s_splatSet[slot], 0, nullptr);
            vkCmdPushConstants(cmd, s_splatLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(sp), &sp);
            vkCmdDispatch(cmd, (smokeN + 63u) / 64u, 1, 1);
        }

        // #6: GPU-routed smoke splats itself from the particle pool into the same
        // accum SSBO (no CPU particles behind it any more). Same camera basis /
        // grid push; the spare .w lanes carry the CollectSmokeParticles LOD zone.
        if (gpSplat) {
            const float maxD     = _min(ps_r_vol_smoke_dist, float(_max(ps_r__detail_radius, 1)));
            const float distFull = _min(ps_r_vol_smoke_dist_full, maxD);
            GPUParticles::MediaSplatPush mp{};
            static_assert(sizeof(mp) == sizeof(sp), "media splat push mirrors SplatPush");
            memcpy(&mp, &sp, sizeof(mp));
            mp.camPos[3] = distFull;
            mp.camDir[3] = maxD;
            GPUParticles::SplatMedia(cmd, s_smokeAccum.GetHandle(), mp);
        }

        // accum splat-write → resolve read.
        {
            VkMemoryBarrier2 mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT; mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT; mb.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
            VkDependencyInfo di{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO }; di.memoryBarrierCount = 1; di.pMemoryBarriers = &mb;
            vkCmdPipelineBarrier2(cmd, &di);
        }

        // RESOLVE — accum → RGBA16F media (written as a storage image, kept GENERAL).
        ResolvePush rp{};
        rp.dims[0] = s32(kSmokeX); rp.dims[1] = s32(kSmokeY); rp.dims[2] = s32(kSmokeZ);
        rp.params[0] = 1.0f / kSmokeFixed;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_resolvePipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_resolveLayout, 0, 1, &s_resolveSet, 0, nullptr);
        vkCmdPushConstants(cmd, s_resolveLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(rp), &rp);
        vkCmdDispatch(cmd, (kSmokeX + 3) / 4, (kSmokeY + 3) / 4, (kSmokeZ + 3) / 4);

        // media resolve-write → inject sampled-read (stays GENERAL).
        ImgBarrier(cmd, s_smokeImg, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                   VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
        Prof::ZoneEnd(cmd, z);
    }

    // ---- INJECT: per froxel in-scatter + extinction.
    {
        const int z = Prof::ZoneBegin(cmd, "VolInject");
        // scatter: (prev frame's integrate read / history copy) -> GENERAL for the
        // storage write. WAR vs last frame's copy READ → src stage covers COPY too.
        ImgBarrier(cmd, s_scatterImg, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                   VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_injPipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_injLayout, 0, 1, &s_injSet[slot], 0, nullptr);
        vkCmdDispatch(cmd, (kGridX + 3) / 4, (kGridY + 3) / 4, (kGridZ + 3) / 4);
        Prof::ZoneEnd(cmd, z);
    }

    // ---- INTEGRATE: march Z, accumulate in-scatter + transmittance.
    {
        const int z = Prof::ZoneBegin(cmd, "VolIntegrate");
        // scatter write -> read (same GENERAL layout, memory barrier only).
        ImgBarrier(cmd, s_scatterImg, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                   VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
        // integrated: (prev tonemap FRAGMENT read) -> GENERAL for the storage write (WAR).
        ImgBarrier(cmd, s_integImg, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                   VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_intPipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_intLayout, 0, 1, &s_intSet[slot], 0, nullptr);
        vkCmdDispatch(cmd, (kGridX + 7) / 8, (kGridY + 7) / 8, 1);
        // integrated write -> tonemap FRAGMENT sample.
        ImgBarrier(cmd, s_integImg, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
        Prof::ZoneEnd(cmd, z);
    }

    // ---- Temporal: copy this frame's blended scatter → history for next frame.
    if (ps_r_vol_ta != 0) {
        ImgBarrier(cmd, s_scatterImg, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   VK_ACCESS_2_SHADER_STORAGE_READ_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COPY_BIT);
        ImgBarrier(cmd, s_histImg, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COPY_BIT);
        VkImageCopy cp{};
        cp.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        cp.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        cp.extent = { kGridX, kGridY, kGridZ };
        vkCmdCopyImage(cmd, s_scatterImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       s_histImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);
        ImgBarrier(cmd, s_histImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                   VK_PIPELINE_STAGE_2_COPY_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
        s_haveHistory = true;
    }

    // Leave the LOCAL scatter volume in SHADER_READ so the particle pass (later this
    // frame) can sample it as a per-froxel light probe for smoke (Stage-0 volumetric
    // lighting). Source layout differs by path: temporal on → scatter ended in
    // TRANSFER_SRC (the history copy read); temporal off → GENERAL (integrate's
    // storage read left it there). Next frame's inject re-acquires it via an
    // UNDEFINED→GENERAL barrier, so discarding the contents here is fine.
    if (ps_r_vol_ta != 0) {
        ImgBarrier(cmd, s_scatterImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VK_ACCESS_2_TRANSFER_READ_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                   VK_PIPELINE_STAGE_2_COPY_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
    } else {
        ImgBarrier(cmd, s_scatterImg, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   VK_ACCESS_2_SHADER_STORAGE_READ_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
    }

    // Cache this frame's reprojection inputs for next frame's temporal pass.
    s_prevVP      = Device.mFullTransform;
    s_prevCamPos  = eye;
    s_prevCamDir  = pt.dir;
    s_prevNear    = gz.nearZ;
    s_prevLogFN   = gz.logFarNear;
    s_prevValid   = true;
    ++s_frame;
}

void Destroy()
{
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
    if (s_injPipe)   { vkDestroyPipeline(VulkanHW.m_Device, s_injPipe, nullptr); s_injPipe = VK_NULL_HANDLE; }
    if (s_intPipe)   { vkDestroyPipeline(VulkanHW.m_Device, s_intPipe, nullptr); s_intPipe = VK_NULL_HANDLE; }
    if (s_splatPipe)   { vkDestroyPipeline(VulkanHW.m_Device, s_splatPipe, nullptr); s_splatPipe = VK_NULL_HANDLE; }
    if (s_resolvePipe) { vkDestroyPipeline(VulkanHW.m_Device, s_resolvePipe, nullptr); s_resolvePipe = VK_NULL_HANDLE; }
    if (s_injLayout) { vkDestroyPipelineLayout(VulkanHW.m_Device, s_injLayout, nullptr); s_injLayout = VK_NULL_HANDLE; }
    if (s_intLayout) { vkDestroyPipelineLayout(VulkanHW.m_Device, s_intLayout, nullptr); s_intLayout = VK_NULL_HANDLE; }
    if (s_splatLayout)   { vkDestroyPipelineLayout(VulkanHW.m_Device, s_splatLayout, nullptr); s_splatLayout = VK_NULL_HANDLE; }
    if (s_resolveLayout) { vkDestroyPipelineLayout(VulkanHW.m_Device, s_resolveLayout, nullptr); s_resolveLayout = VK_NULL_HANDLE; }
    if (s_pool)      { vkDestroyDescriptorPool(VulkanHW.m_Device, s_pool, nullptr); s_pool = VK_NULL_HANDLE; }
    if (s_injSetL)   { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_injSetL, nullptr); s_injSetL = VK_NULL_HANDLE; }
    if (s_intSetL)   { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_intSetL, nullptr); s_intSetL = VK_NULL_HANDLE; }
    if (s_splatSetL)   { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_splatSetL, nullptr); s_splatSetL = VK_NULL_HANDLE; }
    if (s_resolveSetL) { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_resolveSetL, nullptr); s_resolveSetL = VK_NULL_HANDLE; }
    if (s_sampler)   { vkDestroySampler(VulkanHW.m_Device, s_sampler, nullptr); s_sampler = VK_NULL_HANDLE; }
    if (s_scatterView) { vkDestroyImageView(VulkanHW.m_Device, s_scatterView, nullptr); s_scatterView = VK_NULL_HANDLE; }
    if (s_integView)   { vkDestroyImageView(VulkanHW.m_Device, s_integView, nullptr); s_integView = VK_NULL_HANDLE; }
    if (s_histView)    { vkDestroyImageView(VulkanHW.m_Device, s_histView, nullptr); s_histView = VK_NULL_HANDLE; }
    if (s_smokeView)   { vkDestroyImageView(VulkanHW.m_Device, s_smokeView, nullptr); s_smokeView = VK_NULL_HANDLE; }
    if (s_scatterImg)  { VK::Vram::DestroyImage(VulkanHW.m_Allocator, s_scatterImg, s_scatterAlloc); s_scatterImg = VK_NULL_HANDLE; s_scatterAlloc = VK_NULL_HANDLE; }
    if (s_integImg)    { VK::Vram::DestroyImage(VulkanHW.m_Allocator, s_integImg, s_integAlloc); s_integImg = VK_NULL_HANDLE; s_integAlloc = VK_NULL_HANDLE; }
    if (s_histImg)     { VK::Vram::DestroyImage(VulkanHW.m_Allocator, s_histImg, s_histAlloc); s_histImg = VK_NULL_HANDLE; s_histAlloc = VK_NULL_HANDLE; }
    if (s_smokeImg)    { VK::Vram::DestroyImage(VulkanHW.m_Allocator, s_smokeImg, s_smokeAlloc); s_smokeImg = VK_NULL_HANDLE; s_smokeAlloc = VK_NULL_HANDLE; }
    s_ubo.Destroy(); s_uboMapped = nullptr;
    s_dummySSBO.Destroy();
    s_smokeAccum.Destroy();
    for (u32 i = 0; i < kFramesInFlight; ++i) { s_smokeParts[i].Destroy(); s_smokePartsMapped[i] = nullptr; }
    for (u32 i = 0; i < kFramesInFlight; ++i) { s_injSet[i] = VK_NULL_HANDLE; s_intSet[i] = VK_NULL_HANDLE; s_splatSet[i] = VK_NULL_HANDLE; }
    s_resolveSet = VK_NULL_HANDLE;
    s_boundCasc[0] = s_boundCasc[1] = s_boundCasc[2] = s_boundCasc[3] = VK_NULL_HANDLE;
    s_boundVsmAtlas = VK_NULL_HANDLE; s_boundVsmPT = VK_NULL_HANDLE; s_boundVsmUBO = VK_NULL_HANDLE;
    s_haveHistory = false; s_prevValid = false; s_frame = 0;
    s_inited = false; s_failed = false; s_ready = false;
}

}}  // namespace VK::Vol
