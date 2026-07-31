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
#include "vk_env_light.h"         // EnvLight::SunDirVisual — sun held steady through a thunderbolt
#include "vk_profiler.h"          // VK::Prof zones + NameImage
#include "vk_barriers.h"          // ImageBarrier — terrain height bake transitions
#include "vk_Visual.h"            // vkFVisual / vkFHierrarhyVisual / m_pWorldMaterial
#include "vk_world_material.h"    // WorldMaterial::isTerrain (terrain height bake filter)
#include "CRender_Vulkan.h"       // RImplementation.Visuals + b_loaded (terrain height bake)
#include "vk_DetailManager.h"     // detail slot grid — source of the grass canopy field
#include "vk_Detail.h"            // VK::CDetail::bv_bb / scale range — canopy height per type
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
extern int   ps_r_vol_ground_debug;   // forensics: paint height-above-baked-terrain instead of light
extern int   ps_r_vol_hillaire;       // V-0: energy-conserving slice integration (0 = legacy A/B)
extern float ps_r_vol_albedo;         // V-0: single-scatter albedo (sigma_s = sigma_t x albedo)
extern float ps_r_vol_mist;           // V-1: ground-mist layer density (0 = single-layer medium)
extern float ps_r_vol_mist_h;         // V-1: ground-mist thickness (m)
extern float ps_r_vol_mist_g;         // V-1: ground-mist phase anisotropy
extern float ps_r_vol_mist_relief;    // V-1: terrain drop (m) the mist fades in over (0 = everywhere)
extern float ps_r_vol_mist_noise;       // V-1: mist height-warp amplitude (billows instead of a flat lid)
extern float ps_r_vol_mist_noise_scale; // V-1: size of those billows (1/m)
extern float ps_r_vol_mist_micro;     // V-1: micro-relief strength from the rain map (0 = off)
extern float ps_r_vol_mist_micro_h;   // V-1: dip depth (m) at which the micro effect saturates
extern int   ps_r_vol_mist_fade;      // V-1b: thin the bank the camera is inside (0 off, 1 immersion, 2 near, 3 both)
extern float ps_r_vol_mist_inside;    // V-1b: what it thins down to, as a multiple of the dust density
extern float ps_r_vol_mist_near;      // V-1b: radius (m) of the near-camera clearing
extern float ps_r_vol_mist_inside_h;  // V-1b: height (m) over which the immersion fades out — NOT the layer thickness
extern float ps_r_vol_canopy;         // grass canopy occlusion of the sun term (0 = off)
extern float ps_r_vol_canopy_h;       // canopy height multiplier (taste knob over the authored model height)
extern float ps_current_detail_scale; // r__detail_scale — the same per-item size multiplier the decompressor applies
extern int   ps_r_vol_weather;        // V-2: drive the medium from the weather descriptor (0 = manual cvars only, 2 = +log)
extern float ps_r_vol_w_clouds;       // V-2: how much full overcast damps the fog's sun term
extern float ps_r_vol_w_sky;          // V-2: how much full overcast lifts the sky term
extern float ps_r_vol_w_fog;          // V-2: MIST density added at fog_density 1 (the veil)
extern float ps_r_vol_w_fog_dust;     // V-2: dust density gain at fog_density 1 (small — dust makes rays, not fog)
extern float ps_r_vol_w_rain;         // V-2: mist density added at full wetness
extern float ps_r_vol_w_wet_force;    // V-2 test hook: <0 = real weather, 0..1 = pretend this wetness
extern float ps_r_vol_w_flat;         // V-2: how far full overcast flattens the phase toward isotropic
extern int   ps_r_vol_noise_scatter;  // V-0: animated noise on scattering, not extinction
extern int   ps_r_vol_ms;             // V-1: multiple-scattering octaves (1 = single scatter)
extern float ps_r_vol_g_back;         // V-1: backward phase lobe eccentricity
extern float ps_r_vol_g_mix;          // V-1: backward lobe weight
extern int   ps_r_vol_term;           // isolate one in-scatter source (diagnostic)
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
extern float ps_r_vol_vsm_bias;           // fog AIR bias vs the VSM static atlas (0 = legacy surface bias)
extern float ps_r_vol_depth_reject;       // froxel depth-rejection slack (m); 0 = off
extern float ps_r_vol_surf_clip;          // depth-rejection front shell (m): also clip fog within X m BEFORE the surface

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
        // TERRAIN HEIGHT FIELD — world→height-map clip + the depth↔world-Y mapping.
        // Anchors the height fog to the GROUND instead of to the camera. Appended
        // last (the integrate UBO stays a valid prefix).
        float terra_vp[16];
        float terra[4];        // x = eyeY, y = zNear, z = zRange, w = valid (0 = fall back to eye-relative)
        float fog3[4];         // V-0: x = albedo, y = noise-on-scatter flag, zw reserved
        float fog4[4];         // V-1: x = MS octaves, y = backward lobe g, z = backward lobe weight, w = term isolation
        float fog5[4];         // V-1: ground-mist layer — x = density at ground, y = 1/thickness, z = HG g, w = hollow bias (m)
        float fog6[4];         // V-1: micro relief from the rain map — x = dip scale (m), y = strength, z = eyeY - zNear, w = zFar - zNear
        float fog7[4];         // V-1b: mist thinning — x = density scale inside (1 = off), y = near radius (m; 0 = off), z = immersion fade on, w unused
        float canopy[4];       // grass canopy — x = extinction strength (0 = off), y = height scale (m per R unit), z = tallest canopy (m, early-out), w unused
        float canopy_uv[4];    // world XZ → canopy uv: u = x*x + y, v = z*z + w
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
    VkImageView   s_boundVsmAtlasD = VK_NULL_HANDLE;  // dyn atlas (near wind-trees/NPC/grass)
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
        VkDescriptorSetLayoutBinding b[20]{};
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
        b[14] = { 14, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT }; // VSM DYNAMIC atlas (near wind-trees/NPC/grass)
        b[15] = { 15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         1, VK_SHADER_STAGE_COMPUTE_BIT }; // VSM dyn page table
        b[16] = { 16, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         1, VK_SHADER_STAGE_COMPUTE_BIT }; // VSM dyn has-caster flags
        b[17] = { 17, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT }; // PREV frame scene depth (depth rejection)
        b[18] = { 18, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT }; // TERRAIN height field (ground anchor for height fog)
        b[19] = { 19, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT }; // GRASS CANOPY field (sun occlusion by the grass medium)
        VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        lci.bindingCount = 20; lci.pBindings = b;
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
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 14 * kFramesInFlight },   // +4: VSM dyn atlas + prev scene depth + terrain height + grass canopy on inject
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          3 * kFramesInFlight + 1 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         5 * kFramesInFlight + 1 },   // +2*F: VSM dyn PT + dynUsed on inject
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

        VkWriteDescriptorSet w[17]{};
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
        // VSM dyn atlas(14)/PT(15)/dynUsed(16) fallbacks — rebound to the real VSM
        // resources per-frame once AtlasReady (same scheme as the static 7/8).
        w[13] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; w[13].dstSet = s_injSet[i]; w[13].dstBinding = 14; w[13].descriptorCount = 1; w[13].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[13].pImageInfo = &vAtlasI;
        w[14] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; w[14].dstSet = s_injSet[i]; w[14].dstBinding = 15; w[14].descriptorCount = 1; w[14].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[14].pBufferInfo = &vPtI;
        w[15] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; w[15].dstSet = s_injSet[i]; w[15].dstBinding = 16; w[15].descriptorCount = 1; w[15].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[15].pBufferInfo = &vPtI;
        // Prev scene depth(17) fallback — re-written EVERY Execute for the current slot.
        w[16] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; w[16].dstSet = s_injSet[i]; w[16].dstBinding = 17; w[16].descriptorCount = 1; w[16].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[16].pImageInfo = &vAtlasI;
        // integrate: UBO(0) + scatter read(1) + integrated write(2)
        w[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; w[2].dstSet = s_intSet[i]; w[2].dstBinding = 0; w[2].descriptorCount = 1; w[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[2].pBufferInfo = &ubi;
        w[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; w[3].dstSet = s_intSet[i]; w[3].dstBinding = 1; w[3].descriptorCount = 1; w[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[3].pImageInfo = &scatterI;
        w[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; w[4].dstSet = s_intSet[i]; w[4].dstBinding = 2; w[4].descriptorCount = 1; w[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[4].pImageInfo = &integI;
        vkUpdateDescriptorSets(VulkanHW.m_Device, 17, w, 0, nullptr);
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
        // The integrate set only carries a PREFIX of the Vol UBO, so its V-0 flag
        // (r_vol_hillaire) rides a push constant rather than forcing the shader to
        // declare every field up to a new tail lane.
        VkPushConstantRange intPC{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float) * 4 };
        plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &intPC;
        if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_intLayout) != VK_SUCCESS) { Msg("![VK Vol] integrate layout failed"); s_failed = true; return false; }
        plci.pushConstantRangeCount = 0; plci.pPushConstantRanges = nullptr;
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

// ─────────────────────────────────────────────────────────────────────────────
// TERRAIN HEIGHT FIELD — the height-fog anchor, baked once per level.
//
// The anchor used to be `eye.y - 2`: the fog layer hung two metres under the
// CAMERA and travelled with it. Valleys never filled, hills never poked through,
// and climbing onto a roof lifted the whole layer with you — the density field had
// no world structure at all, which is exactly why the fog reads "uniformly thick"
// everywhere.
//
// Anchor it to the GROUND instead. The top-down map we already have (the rain map,
// via Surface Field's SF_HeightAt) stores the TOPMOST surface — roofs and canopy
// included — so reusing it would reproduce the rooftop artefact in a new costume.
// So bake our own: terrain visuals ONLY (mat->isTerrain), straight down, once per
// level. Terrain never moves, so this costs nothing per frame.
namespace {

constexpr u32 kTerraSize = 2048;                    // ~0.5 m/texel over a 1 km level
VkPipelineLayout s_terraLayout = VK_NULL_HANDLE;    // mat4 push (world→height-map clip)
VkImage       s_terraImg   = VK_NULL_HANDLE;
VmaAllocation s_terraAlloc = VK_NULL_HANDLE;
VkImageView   s_terraView  = VK_NULL_HANDLE;
Fmatrix       s_terraVP;
float         s_terraEyeY = 0.f, s_terraNear = 1.f, s_terraRange = 1.f;
bool          s_terraOk = false;
size_t        s_terraVisuals = size_t(-1);          // Visuals.size() this bake was made from (level swap → re-bake)
struct StridePipe { u32 stride; VkPipeline pipe; };
xr_vector<StridePipe> s_terraPipes;                 // outlive the recording (see BakeTerrainHeight)

// ---- GRASS CANOPY field (r_vol_canopy) --------------------------------------
// The fog's sun term used to walk straight through a grass field, and the shadow
// atlas cannot fix that: grass only casts within ~48 m of the camera (near band
// dynamic, far band rigid into the static cache), it is a centimetre-scale stipple
// that 3x3 PCF averages into "half lit", and it sways, so what little it does
// contribute shimmers. All three problems come from treating blades as individual
// occluders. A grass field is a MEDIUM: what matters is how much canopy the sunlight
// had to cross to reach this froxel.
// Source = the level's own detail-slot grid (level.details, 2 m slots, already in
// memory and covering the WHOLE map, LOD and draw distance be damned) crossed with
// each detail model's bounding box for the height. Baked once per level into a tiny
// RG8 map: R = canopy height in kCanopyHRange units, G = coverage 0..1. Bilinear
// across slots gives a smooth field for free; nothing is spent per frame.
constexpr float kCanopyHRange = 4.0f;               // metres encoded by R = 1 (grass never approaches this)
VkImage       s_canopyImg   = VK_NULL_HANDLE;
VmaAllocation s_canopyAlloc = VK_NULL_HANDLE;
VkImageView   s_canopyView  = VK_NULL_HANDLE;
u32           s_canopyW = 0, s_canopyH = 0;
float         s_canopyUx = 0.f, s_canopyUz = 0.f;   // world XZ → uv: u = x*Ux + Uw
float         s_canopyUw = 0.f, s_canopyUh = 0.f;
float         s_canopyMaxH = 0.f;                   // tallest canopy on the level (m) — the shader's early-out
bool          s_canopyOk = false;
size_t        s_canopyVisuals = size_t(-1);

// Build the canopy field from the detail-slot grid. CPU only, once per level.
void BakeGrassCanopy()
{
    if (!RImplementation.b_loaded) return;
    if (s_canopyVisuals == RImplementation.Visuals.size()) return;
    s_canopyVisuals = RImplementation.Visuals.size();
    s_canopyOk = false;

    VK::CDetailManager* DM = RImplementation.Details;
    if (!DM || !DM->dtSlots || DM->objects.empty() || DM->dtH.size_x == 0 || DM->dtH.size_z == 0) {
        Msg("[VK Vol] grass canopy: no detail grid on this level — the sun term keeps passing through grass");
        return;
    }
    const u32 W = DM->dtH.size_x, H = DM->dtH.size_z;
    if (u64(W) * u64(H) > 64ull * 1024 * 1024) { Msg("![VK Vol] grass canopy: detail grid %ux%u too large — skipped", W, H); return; }

    // Per-type canopy height. The decompressor scales every instance by
    // randF(min*0.5, max*0.9) * ps_current_detail_scale, so the field wants the MEAN
    // of that range — a canopy is an average, not a tallest-blade envelope.
    float th[dm_max_objects] = {};
    const u32 nObj = _min((u32)DM->objects.size(), (u32)dm_max_objects);
    for (u32 i = 0; i < nObj; ++i) {
        const VK::CDetail* o = DM->objects[i];
        if (!o) continue;
        const float meanScale = 0.5f * (o->m_MinScale * 0.5f + o->m_MaxScale * 0.9f) * ps_current_detail_scale;
        th[i] = _max(0.f, o->bv_bb.max.y) * meanScale;
    }

    xr_vector<u8> px(size_t(W) * H * 2, 0);
    float maxH = 0.f; u32 nCover = 0;
    for (u32 j = 0; j < H; ++j)
    for (u32 i = 0; i < W; ++i) {
        const DetailSlot& d = DM->dtSlots[size_t(j) * W + i];
        float hMax = 0.f, cov = 0.f;
        for (u32 k = 0; k < 4; ++k) {
            const u8 id = d.r_id(k);
            if (id == DetailSlot::ID_Empty || id >= nObj) continue;
            // The 4 palette entries are the CORNER alphas of a bilinear coverage
            // field over the slot (see cache_Decompress: they are interpolated and
            // dithered per candidate position), so their mean IS the slot's mean
            // coverage for that object — density- and scale-independent.
            const DetailPalette& p = d.palette[k];
            const float a = float(p.a0 + p.a1 + p.a2 + p.a3) / (4.f * 15.f);
            if (a <= 0.001f) continue;
            cov += a;
            hMax = _max(hMax, th[id]);
        }
        if (cov > 0.001f) { ++nCover; maxH = _max(maxH, hMax); }
        const size_t o = (size_t(j) * W + i) * 2;
        px[o + 0] = u8(clampr(hMax / kCanopyHRange, 0.f, 1.f) * 255.f + 0.5f);
        px[o + 1] = u8(clampr(cov,                  0.f, 1.f) * 255.f + 0.5f);
    }
    if (!nCover) { Msg("[VK Vol] grass canopy: detail grid %ux%u has no covered slots — nothing to occlude", W, H); return; }

    VK::Vram::Scope _vram_scope("Vol.Canopy");
    if (s_canopyImg != VK_NULL_HANDLE && (W != s_canopyW || H != s_canopyH)) {
        if (s_canopyView) vkDestroyImageView(VulkanHW.m_Device, s_canopyView, nullptr);
        VK::Vram::DestroyImage(VulkanHW.m_Allocator, s_canopyImg, s_canopyAlloc);
        s_canopyImg = VK_NULL_HANDLE; s_canopyView = VK_NULL_HANDLE; s_canopyAlloc = VK_NULL_HANDLE;
    }
    if (s_canopyImg == VK_NULL_HANDLE) {
        VK::ImageDesc d;
        d.type = VK_IMAGE_TYPE_2D; d.format = VK_FORMAT_R8G8_UNORM;
        d.extent = { W, H, 1 };
        d.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        d.name  = "Vol.GrassCanopy";
        if (!VK::CreateImage(d, s_canopyImg, s_canopyAlloc)) { Msg("![VK Vol] grass canopy image failed"); return; }
        s_canopyView = VK::CreateImageView(s_canopyImg, VK_FORMAT_R8G8_UNORM, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT);
        if (!s_canopyView) { Msg("![VK Vol] grass canopy view failed"); return; }
    }

    VK::CVulkanBuffer staging;
    staging.Create(px.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
    if (void* m = staging.Map()) { memcpy(m, px.data(), px.size()); staging.Flush(); }
    else { Msg("![VK Vol] grass canopy staging map failed"); return; }

    VkCommandBuffer cmd = VulkanHW.BeginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) { Msg("![VK Vol] grass canopy: no upload command buffer"); return; }
    VkImageMemoryBarrier b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    b.srcAccessMask = 0; b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = s_canopyImg; b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    VkBufferImageCopy r{};
    r.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    r.imageExtent = { W, H, 1 };
    vkCmdCopyBufferToImage(cmd, staging.GetHandle(), s_canopyImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, 0, nullptr, 1, &b);
    VulkanHW.EndSingleTimeCommands(cmd);

    // World XZ → uv. Texel i holds slot (i - offs_x), whose CENTRE sits at world
    // x = (i - offs_x)*slot + slot/2, so u = (x/slot + offs_x) / W lands exactly on
    // the texel centre — bilinear then interpolates between slot centres, not edges.
    s_canopyW = W; s_canopyH = H;
    s_canopyUx = 1.f / (dm_slot_size * float(W));
    s_canopyUw = float(DM->dtH.offs_x) / float(W);
    s_canopyUz = 1.f / (dm_slot_size * float(H));
    s_canopyUh = float(DM->dtH.offs_z) / float(H);
    s_canopyMaxH = maxH;
    s_canopyOk = true;
    Msg("[VK Vol] grass canopy: %ux%u slots (%.0f m), %u covered (%.1f%%), tallest %.2f m, %u types",
        W, H, dm_slot_size, nCover, 100.f * float(nCover) / float(W * H), maxH, nObj);
}

void CollectTerrain(vkRender_Visual* rv, xr_vector<vkFVisual*>& out, Fbox& box)
{
    if (!rv) return;
    const u32 t = rv->Type;
    if (t == MT_NORMAL || t == MT_PROGRESSIVE) {
        auto* fv = static_cast<vkFVisual*>(rv);
        if (!fv->m_pWorldMaterial || !fv->m_pWorldMaterial->isTerrain) return;
        if (!fv->m_mesh.IsValid() || !fv->m_mesh.p_rm_Vertices || !fv->m_mesh.p_rm_Indices) return;
        out.push_back(fv);
        box.merge(fv->vis.box);
        return;
    }
    if (t == MT_HIERRARHY || t == MT_LOD) {
        auto* hv = dynamic_cast<vkFHierrarhyVisual*>(rv);
        if (!hv) return;
        for (auto* child : hv->children) CollectTerrain(child, out, box);
    }
}

// Depth-only pipeline for the bake, one per vertex stride (position is at offset 0
// in every level layout, so a single attribute covers them all — same trick as the
// sun-shadow caster pass, whose vertex shader we reuse verbatim).
VkPipeline MakeTerraPipe(u32 stride, VkPipelineLayout layout)
{
    VkShaderModule vs = g_ShaderManager->Load("shadow_depth.vert.spv");
    if (!vs) { Msg("![VK Vol] shadow_depth.vert.spv missing — terrain height bake skipped"); return VK_NULL_HANDLE; }

    VkPipelineShaderStageCreateInfo st{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    st.stage = VK_SHADER_STAGE_VERTEX_BIT; st.module = vs; st.pName = "main";

    VkVertexInputBindingDescription vb{ 0, stride, VK_VERTEX_INPUT_RATE_VERTEX };
    VkVertexInputAttributeDescription va{ 0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0 };
    VkPipelineVertexInputStateCreateInfo vi{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &vb;
    vi.vertexAttributeDescriptionCount = 1; vi.pVertexAttributeDescriptions = &va;

    VkPipelineInputAssemblyStateCreateInfo ia{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    vp.viewportCount = 1; vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE;   // X-Ray winding is not trustworthy
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo ds{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_TRUE; ds.depthCompareOp = VK_COMPARE_OP_LESS;
    VkPipelineColorBlendStateCreateInfo cb{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    const VkDynamicState dyn[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dy{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dy.dynamicStateCount = 2; dy.pDynamicStates = dyn;

    const VkFormat dfmt = VK_FORMAT_D32_SFLOAT;
    VkPipelineRenderingCreateInfo rci{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    rci.depthAttachmentFormat = dfmt;

    VkGraphicsPipelineCreateInfo pci{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    pci.pNext = &rci; pci.stageCount = 1; pci.pStages = &st;
    pci.pVertexInputState = &vi; pci.pInputAssemblyState = &ia; pci.pViewportState = &vp;
    pci.pRasterizationState = &rs; pci.pMultisampleState = &ms; pci.pDepthStencilState = &ds;
    pci.pColorBlendState = &cb; pci.pDynamicState = &dy; pci.layout = layout;

    VkPipeline p = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(VulkanHW.m_Device, VK_NULL_HANDLE, 1, &pci, nullptr, &p) != VK_SUCCESS) {
        Msg("![VK Vol] terrain height pipeline (stride %u) failed", stride);
        return VK_NULL_HANDLE;
    }
    return p;
}

void BakeTerrainHeight(VkCommandBuffer cmd)
{
    if (!RImplementation.b_loaded) return;
    if (s_terraVisuals == RImplementation.Visuals.size()) return;   // already baked for this level
    s_terraVisuals = RImplementation.Visuals.size();
    s_terraOk = false;

    VK::Vram::Scope _vram_scope("Vol.TerrainH");

    if (s_terraLayout == VK_NULL_HANDLE) {
        s_terraLayout = VK::MakePipelineLayout({}, sizeof(Fmatrix), VK_SHADER_STAGE_VERTEX_BIT);
        if (s_terraLayout == VK_NULL_HANDLE) { Msg("![VK Vol] terrain height layout failed"); return; }
    }

    xr_vector<vkFVisual*> terra; terra.reserve(1024);
    Fbox box; box.invalidate();
    for (IRenderVisual* iv : RImplementation.Visuals)
        CollectTerrain(static_cast<vkRender_Visual*>(iv), terra, box);
    if (terra.empty()) {
        Msg("[VK Vol] terrain height: no terrain visuals — height fog stays camera-anchored");
        return;
    }

    // Square ortho over the terrain's XZ extent, looking straight down from above it.
    Fvector c; box.getcenter(c);
    Fvector sz; box.getsize(sz);
    const float half = 0.5f * _max(sz.x, sz.z) + 8.0f;      // margin so edge texels are real
    const float top  = box.max.y + 16.0f;
    s_terraNear  = 1.0f;
    s_terraRange = (top - box.min.y) + 32.0f;               // zFar - zNear
    s_terraEyeY  = top;

    Fmatrix view; view.build_camera_dir(Fvector{ c.x, top, c.z }, Fvector{ 0.f, -1.f, 0.f }, Fvector{ 0.f, 0.f, 1.f });
    Fmatrix proj; proj.build_projection_ortho(2.f * half, 2.f * half, s_terraNear, s_terraNear + s_terraRange);
    s_terraVP.mul(proj, view);

    // Image (lazily; survives level swaps — only the content is re-rendered).
    if (s_terraImg == VK_NULL_HANDLE) {
        VK::ImageDesc d;
        d.type = VK_IMAGE_TYPE_2D; d.format = VK_FORMAT_D32_SFLOAT;
        d.extent = { kTerraSize, kTerraSize, 1 };
        d.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        d.name  = "Vol.TerrainHeight";
        if (!VK::CreateImage(d, s_terraImg, s_terraAlloc)) { Msg("![VK Vol] terrain height image failed"); return; }
        s_terraView = VK::CreateImageView(s_terraImg, VK_FORMAT_D32_SFLOAT, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_DEPTH_BIT);
        if (!s_terraView) { Msg("![VK Vol] terrain height view failed"); return; }
    }

    // One pipeline per distinct stride (terrain is normally a single layout).
    // They must OUTLIVE this function: we only RECORD the draws here — the command
    // buffer is submitted later, and a pipeline destroyed before its commands execute
    // is a use-after-free the GPU answers by drawing nothing (the map then stays
    // cleared to 1.0 and every froxel silently falls back to the eye anchor, with the
    // bake still logging a cheerful success). Kept until Destroy(); a re-bake frees
    // the previous level's set first, long after those commands retired.
    for (auto& sp : s_terraPipes) if (sp.pipe) vkDestroyPipeline(VulkanHW.m_Device, sp.pipe, nullptr);
    s_terraPipes.clear();
    auto pipeFor = [&](u32 stride) -> VkPipeline {
        for (auto& sp : s_terraPipes) if (sp.stride == stride) return sp.pipe;
        if (s_terraPipes.size() >= 4) return VK_NULL_HANDLE;   // pathological content — skip the tail
        VkPipeline p = MakeTerraPipe(stride, s_terraLayout);
        s_terraPipes.push_back({ stride, p });
        return p;
    };

    ImageBarrier(cmd, s_terraImg, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);

    VkRenderingAttachmentInfo dAtt{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    dAtt.imageView = s_terraView; dAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    dAtt.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; dAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    dAtt.clearValue.depthStencil = { 1.0f, 0 };
    VkRenderingInfo ri{ VK_STRUCTURE_TYPE_RENDERING_INFO };
    ri.renderArea.extent = { kTerraSize, kTerraSize }; ri.layerCount = 1; ri.pDepthAttachment = &dAtt;
    vkCmdBeginRendering(cmd, &ri);

    // NEGATIVE-height viewport: the same D3D→Vulkan Y flip the scene and shadow
    // passes use, so the stored depth matches the sampling convention on the
    // camera side (uv.y = 1 - uv.y, as in SF_MapUV).
    VkViewport vpr{ 0.f, float(kTerraSize), float(kTerraSize), -float(kTerraSize), 0.f, 1.f };
    vkCmdSetViewport(cmd, 0, 1, &vpr);
    VkRect2D scr{ {0, 0}, { kTerraSize, kTerraSize } };
    vkCmdSetScissor(cmd, 0, 1, &scr);

    u32 drawn = 0;
    VkPipeline bound = VK_NULL_HANDLE;
    for (vkFVisual* fv : terra) {
        VkPipeline p = pipeFor(fv->m_mesh.vStride);
        if (p == VK_NULL_HANDLE) continue;
        if (p != bound) { vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p); bound = p; }
        // Level geometry is already in world space (the caster path pushes the bare
        // light VP for exactly this reason) — no per-object transform.
        vkCmdPushConstants(cmd, s_terraLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Fmatrix), &s_terraVP);
        VkBuffer     vbh = fv->m_mesh.p_rm_Vertices->GetHandle();
        VkDeviceSize off = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &vbh, &off);
        vkCmdBindIndexBuffer(cmd, fv->m_mesh.p_rm_Indices->GetHandle(), 0, fv->m_mesh.iType);
        vkCmdDrawIndexed(cmd, fv->m_mesh.iCount, 1, fv->m_mesh.iBase, int32_t(fv->m_mesh.vBase), 0);
        ++drawn;
    }
    vkCmdEndRendering(cmd);

    ImageBarrier(cmd, s_terraImg, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);

    s_terraOk = (drawn > 0);
    Msg("[VK Vol] terrain height baked: %u/%u terrain meshes, %.0f x %.0f m at %.2f m/texel, Y %.1f..%.1f",
        drawn, (u32)terra.size(), 2.f * half, 2.f * half, (2.f * half) / float(kTerraSize), box.min.y, box.max.y);
}

}  // anonymous namespace

VkImageView GetTerrainHeightView() { return s_terraView; }

void Execute(VkCommandBuffer cmd, const ProjTerms& pt, u32 slot,
             const SmokeParticle* smoke, u32 smokeCount,
             VkImageView sceneDepthPrev)
{
    if (!s_ready || !Wanted()) return;
    if (slot >= kFramesInFlight) slot = 0;

    // Ground anchor for the height fog — one-shot per level, before anything samples it.
    BakeTerrainHeight(cmd);
    // Grass canopy for the sun occlusion — same deal, CPU-side (own command buffer).
    BakeGrassCanopy();

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
            // Held steady through a thunderbolt (VK::EnvLight::SunDirVisual): the bolt
            // hijacks CurrentEnv->sun_dir, and following it here is what shot god rays
            // out of a fake second sun — with the temporal history keeping one per bolt.
            const Fvector& sunD = VK::EnvLight::SunDirVisual();
            ub.sun_dir[0] = sunD.x; ub.sun_dir[1] = sunD.y; ub.sun_dir[2] = sunD.z;
            ub.sun_color[0] = E->sun_color.x;
            ub.sun_color[1] = E->sun_color.y;
            ub.sun_color[2] = E->sun_color.z;
            // SKY IN-SCATTER = HEMISPHERE + the ambient floor.
            //
            // This used to be E->ambient alone, and that is a FLOOR, not the sky: a
            // measured frame gave ambient=(0.020,0.021,0.023) against hemi=(0.444,
            // 0.417,0.389) — the actual hemisphere the surfaces are lit by is ~20x
            // larger and was not in the fog at all. Consequences, all confirmed with
            // r_vol_term: the sky term was invisible (term 2 ≈ black), r_vol_ambient
            // did nothing across its whole range, and the fog existed ONLY where the
            // sun's phase function is strong — i.e. air in shadow scattered almost
            // nothing and read as dead grey, which is the "dirty cigarette smoke"
            // look. Real fog in shade is lit by the sky; now it is here too.
            ub.sky_ambient[0] = E->hemi_color.x + E->ambient.x;
            ub.sky_ambient[1] = E->hemi_color.y + E->ambient.y;
            ub.sky_ambient[2] = E->hemi_color.z + E->ambient.z;
        }
    }
    // ---- V-2: WEATHER (r_vol_weather) -------------------------------------------
    // Until now the froxel medium read exactly three things from the weather — sun
    // direction, sun colour, hemisphere — and its DENSITY was a bare cvar. The
    // forward distance haze, meanwhile, is fully weather-driven (fog_near/fog_far/
    // fog_color from the same descriptor), so changing weather thickened one system
    // and left the other standing: two fogs visibly disagreeing in the same frame.
    // These four fields were sitting in the descriptor unread. Package 1 wires them
    // as SCALARS (no per-froxel work, hence no shader change):
    //   clouds_color.w  overcast diffuses the sun disc into the dome → damp the sun
    //                   term, lift the sky term. Flat light, not just darker light.
    //   fog_density     the weather's own "how foggy" 0..1 → scales the dust layer,
    //                   which is what stops the two systems diverging.
    //   rain/wetness    ground fog after rain — r_vol_mist finally gets a DRIVER
    //                   instead of being typed in by hand.
    // ⚠ WHICH LAYER a weather term lands on decides what the player sees. First cut
    // hung fog_density on the DUST layer and it read as "the sun got brighter": dust
    // density scales sigma_s, dust carries the sharp forward lobe, so a foggy overcast
    // multiplied the SHAFTS by 2.8 while the cloud damping was only 0.78 — net sun
    // term UP, exactly backwards. Fog belongs on the MIST layer (near-isotropic, the
    // veil-maker); dust keeps a small share so the air still gains some body.
    // And overcast is flattened by moving the PHASE toward isotropic, not by dimming:
    // a brightness multiplier is eaten by auto-exposure (the r_vol_sun 12 lesson),
    // a change in the phase function is a change in SHAPE and survives it.
    struct { float sun, sky, dust, mistAdd, flat, fogd; } wx{ 1.f, 1.f, 1.f, 0.f, 1.f, 0.f };
    if (ps_r_vol_weather && g_pGamePersistent) {
        const auto& env = g_pGamePersistent->Environment();
        if (const auto* E = env.CurrentEnv) {
            const float clouds = clampr(E->clouds_color.w, 0.f, 1.f);
            const float fogd   = clampr(E->fog_density,    0.f, 1.f);
            const float rain   = clampr(E->rain_density,   0.f, 1.f);
            // OWN wetness accumulator, same reasoning as the rain pass: mod weather
            // scripts overwrite env.wetness_factor freely, so soak/dry here (~45 s /
            // ~3 min) and only take the engine value when a script drives it HIGHER.
            // Without this the ground fog would pop the instant rain stops.
            static float s_vwet = 0.f;
            const float dt = clampr(Device.fTimeDelta, 0.f, 0.1f);
            if (rain > 0.001f) s_vwet += rain * dt / 45.f;
            else               s_vwet -= dt / 180.f;
            clamp(s_vwet, 0.f, 1.f);
            float wet = _max(s_vwet, clampr(env.wetness_factor, 0.f, 1.f));
            // Test hook: the rain-driven bank is the half of V-2 that a dry level can
            // never show, and waiting for the mod's weather script to bring rain is a
            // poor way to verify a feature. >= 0 pretends this wetness (1 = just
            // rained), so the bank and the hollow bypass can be tuned on demand.
            if (ps_r_vol_w_wet_force >= 0.f) wet = clampr(ps_r_vol_w_wet_force, 0.f, 1.f);

            wx.sun     = _max(0.f, 1.f - ps_r_vol_w_clouds * clouds);
            wx.sky     = 1.f + ps_r_vol_w_sky * clouds;
            wx.dust    = 1.f + ps_r_vol_w_fog_dust * fogd;
            // ⚠ fog_density drives DUST, and by default nothing else. It is not "how
            // much ground fog" — it is where the forward haze STARTS
            // (fog_near = (1-density)*0.85*fog_distance), so its honest counterpart is
            // the thin tall layer, which IS the distance haze. Hanging the ground bank
            // on it shipped soup: a dry, hazy morning at fog_density 0.90 doubled the
            // bank AND lifted it out of the hollows, i.e. invented a fog day out of a
            // parameter that never claimed one. The bank's weather driver is RAIN.
            wx.mistAdd = ps_r_vol_w_fog * fogd + ps_r_vol_w_rain * wet;
            wx.flat    = _max(0.f, 1.f - ps_r_vol_w_flat * clouds);
            wx.fogd    = fogd;

            if (ps_r_vol_weather > 1) {
                static u32 s_cd = 0;
                if (s_cd == 0) { s_cd = 60;
                    // Name + the two hour-sections being blended: a weather cycle is a
                    // table over the day, so "the rain cycle is active" and "it is
                    // raining right now" are different claims, and only this tells them
                    // apart. wfx = a weather EFFECT is running, which BLOCKS set_weather.
                    auto& envr = g_pGamePersistent->Environment();
                    Msg("[VK Vol] weather: '%s'%s [%s -> %s] clouds %.2f fog %.2f rain %.2f wet %.2f -> sun x%.2f sky x%.2f dust x%.2f mist +%.3f phase x%.2f",
                        envr.GetWeather().c_str(), envr.IsWeatherFXPlaying() ? " WFX" : "",
                        envr.Current[0] ? envr.Current[0]->m_identifier.c_str() : "?",
                        envr.Current[1] ? envr.Current[1]->m_identifier.c_str() : "?",
                        clouds, fogd, rain, wet, wx.sun, wx.sky, wx.dust, wx.mistAdd, wx.flat);
                } else --s_cd;
            }
        }
    }
    // Decode, THEN scale — the r_sun_boost / r_vol_ambient multiplies used to be fused
    // into the assignments above, which would have applied them on the wrong side of
    // the sRGB curve. Split so the fallback and the real env values share one
    // conversion and the knobs keep meaning "scale the radiance".
    ColorSpace::LinearizeRGB(ub.sun_color);
    ColorSpace::LinearizeRGB(ub.sky_ambient);
    for (int c = 0; c < 3; ++c) {
        ub.sun_color[c]   *= ps_r_sun_boost   * wx.sun;
        ub.sky_ambient[c] *= ps_r_vol_ambient * wx.sky;
    }
    // Leak forensics (viewed via the tonemap RAW composite, r_vol_debug != 0):
    // 3 = paint froxel sun visibility, 4 = paint the occlusion SOURCE per froxel.
    ub.sun_color[3] = float(ps_r_vol_debug);
    // Front surface-shell clip (r_vol_surf_clip, m) — see depthVis in vol_inject.
    ub.sky_ambient[3] = ps_r_vol_surf_clip;

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
    ub.fog[0] = ps_r_vol_density * wx.dust;   // V-2: weather's own fog_density scales the dust layer
    // LEGACY height anchor — only the fallback now: with the terrain height field
    // baked (terra.w = 1) the shader takes the GROUND under each froxel instead, so
    // the layer stops riding the camera up hills and onto rooftops.
    ub.fog[1] = eye.y - 2.0f;
    ub.fog[2] = ps_r_vol_height;
    ub.fog[3] = ps_r_vol_g * wx.flat;   // V-2: overcast blends the lobe toward isotropic = flat light
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
    // .w = AIR bias vs the VSM static atlas (r_vol_vsm_bias; 0 = legacy surface bias):
    // air has no acne to hide — the 0.6 m surface slack lit fog THROUGH thin geometry.
    ub.atmoR[0] = 0.18f; ub.atmoR[1] = 0.41f; ub.atmoR[2] = 1.0f; ub.atmoR[3] = ps_r_vol_vsm_bias;
    // Terrain height field (baked once per level; 0 = not available → eye-relative).
    memcpy(ub.terra_vp, &s_terraVP, sizeof(ub.terra_vp));
    ub.terra[0] = s_terraEyeY; ub.terra[1] = s_terraNear;
    ub.terra[2] = s_terraRange;
    // w: 0 = no field (eye-relative fallback), 1 = anchored, 2 = anchored + forensics
    // view (r_vol_ground_debug, read with r_vol_debug 1).
    // 3 = micro-relief forensics (r_vol_ground_debug 2): is there small-scale relief in
    // the rain map at all, and does it agree with the baked ground? See the shader.
    // 4 = grass-canopy forensics (r_vol_ground_debug 3): is the baked canopy field real?
    ub.terra[3] = s_terraOk ? (ps_r_vol_ground_debug >= 3 ? 4.0f
                                                          : (ps_r_vol_ground_debug >= 2 ? 3.0f
                                                                                        : (ps_r_vol_ground_debug ? 2.0f : 1.0f)))
                            : 0.0f;
    // V-0: scattering albedo + where the animated noise lands (light vs thickness).
    ub.fog3[0] = ps_r_vol_albedo;
    ub.fog3[1] = ps_r_vol_noise_scatter ? 1.0f : 0.0f;
    // V-1: the mist layer's own shape noise (height warp — see r_vol_mist_noise).
    ub.fog3[2] = ps_r_vol_mist_noise;
    ub.fog3[3] = ps_r_vol_mist_noise_scale;
    // V-1: multiple-scattering octaves + the backward phase lobe.
    ub.fog4[0] = float(ps_r_vol_ms);
    ub.fog4[1] = ps_r_vol_g_back;
    ub.fog4[2] = ps_r_vol_g_mix;
    ub.fog4[3] = float(ps_r_vol_term);   // term isolation (0 = normal)
    // V-1: the SECOND medium layer (ground mist). Summed into the same sigma_s/sigma_t
    // as the dust layer above, but with its own thickness and phase — see r_vol_mist.
    // Density 0 short-circuits the whole thing in the shader (single-layer A/B).
    // V-2: rain/wetness ADDS to the hand-tuned bank rather than replacing it, so a dry
    // level keeps exactly the look that was tuned and a wet one grows ground fog. The
    // hollow gate still applies — fog after rain pools in the low ground, as it should.
    const float mistEff = ps_r_vol_mist + wx.mistAdd;
    ub.fog5[0] = mistEff;
    ub.fog5[1] = 1.0f / _max(ps_r_vol_mist_h, 0.05f);   // metres → falloff rate
    ub.fog5[2] = ps_r_vol_mist_g * wx.flat;
    // 0 = layer everywhere, >0 = hollows only.
    // ⚠ V-2 first tried to let weather out of the hollows by SHRINKING this band
    // (relief * (1-fog_density)). That is wrong and it showed up in game as sharp-edged
    // CLEAR POCKETS inside dense fog: the band is the width of the smoothstep RAMP, so
    // narrowing it to 0.8 m turns the gate into a step — ground a hair below its
    // surroundings gets full fog, ground level with them gets exactly none. The gate
    // reads 4 taps in a cross at ±45 m, and a near-binary function over a 4-tap cross
    // is precisely the rectangular edge that appeared. The bypass now lifts the gate
    // toward 1 instead (fog7.w) and the ramp keeps its width.
    ub.fog5[3] = ps_r_vol_mist_relief;
    // V-1 micro relief: the RAIN map (1024² over ±75 m = 14.6 cm/texel) resolves ruts
    // and dips the level-wide terrain bake (71 cm/texel) averages away. Pass the ortho
    // Z terms so the shader can turn a sampled depth back into a world height.
    ub.fog6[0] = _max(ps_r_vol_mist_micro_h, 0.01f);
    ub.fog6[1] = ps_r_vol_mist_micro;
    ub.fog6[2] = ShadowMap::RainEyeY() - ShadowMap::RainZNear();
    ub.fog6[3] = ShadowMap::RainZFar() - ShadowMap::RainZNear();
    // V-1b: the bank is scenery — see r_vol_mist_fade. The knob is a multiple of the
    // DUST density; the shader wants a plain scale on the mist layer, so the division
    // happens here (once) rather than per froxel. Clamped to 1 = no thinning, which is
    // also what an r_vol_mist_inside above the bank's own density means: "already
    // thinner than the target, leave it alone".
    {
        // Against the EFFECTIVE densities (weather included) — otherwise rain-grown
        // fog would be measured against the hand-typed bank and the target would drift.
        const bool  fadeOn = (ps_r_vol_mist_fade != 0) && mistEff > 1e-6f;
        ub.fog7[0] = fadeOn ? _min(1.0f, ps_r_vol_mist_inside * ub.fog[0] / _max(mistEff, 1e-6f)) : 1.0f;
        ub.fog7[1] = (fadeOn && (ps_r_vol_mist_fade & 2)) ? ps_r_vol_mist_near : 0.0f;
        // Immersion falloff RATE (1/m), 0 = immersion off. Its scale is deliberately
        // independent of the layer thickness — see the shader for what tying them
        // together did to rooftops.
        ub.fog7[2] = (fadeOn && (ps_r_vol_mist_fade & 1)) ? 1.0f / _max(ps_r_vol_mist_inside_h, 0.5f) : 0.0f;
        // V-2: hollow-gate BYPASS, driven by the WEATHER'S SHARE of the bank rather
        // than by fog_density. Rationale: only fog the weather grew has any business
        // ignoring the terrain — the hand-tuned bank is authored as a hollows effect
        // and must stay one. So a dry level keeps its valley pattern exactly, and a
        // wet one climbs out onto the ridges in proportion to how much of the bank the
        // rain actually made. Keying this off fog_density instead lifted the gate
        // across the whole map on a merely HAZY morning.
        ub.fog7[3] = clampr(wx.mistAdd / _max(mistEff, 1e-6f), 0.f, 1.f);
    }
    // GRASS CANOPY (r_vol_canopy). The sun term is attenuated by how much canopy the
    // light had to cross to reach the froxel — a medium, not a stipple of blade
    // shadows. z carries the tallest canopy on the level so the shader can skip the
    // fetch for everything above it (which is most of the grid).
    {
        const bool on = s_canopyOk && ps_r_vol_canopy > 0.001f;
        ub.canopy[0] = on ? ps_r_vol_canopy : 0.0f;
        ub.canopy[1] = kCanopyHRange * ps_r_vol_canopy_h;   // R (0..1) → metres
        ub.canopy[2] = s_canopyMaxH * ps_r_vol_canopy_h;
        ub.canopy[3] = 0.0f;
        ub.canopy_uv[0] = s_canopyUx; ub.canopy_uv[1] = s_canopyUw;
        ub.canopy_uv[2] = s_canopyUz; ub.canopy_uv[3] = s_canopyUh;
    }

    // DEPTH REJECTION (r_vol_depth_reject): occluded froxels get no in-scatter.
    // Needs the prev-frame depth (valid only once a full frame ran → s_prevValid,
    // same gate as the temporal history) — the shader path is off at 0.
    // camPos.w/camDir.w = this projection's _43/_33: viewZ = _43 / (ndcZ - _33).
    const bool depthRej = ps_r_vol_depth_reject > 0.f && sceneDepthPrev != VK_NULL_HANDLE && s_prevValid;
    ub.sun_dir[3] = depthRej ? ps_r_vol_depth_reject : 0.f;
    ub.camPos[3]  = pt.p43;   // same terms DeriveGridZ linearizes the depth buffer with
    ub.camDir[3]  = pt.p33;

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
        VkImageView atlas  = VSM::GetAtlasView();
        VkBuffer    pt     = VSM::GetPageTableHandle();
        VkBuffer    cubo   = VSM::GetUBOHandle();
        VkImageView atlasD = VSM::GetDynAtlasView();
        VkBuffer    ptD    = VSM::GetDynPageTableHandle();
        VkBuffer    used   = VSM::GetDynUsedHandle();
        if (atlas && pt && cubo && atlasD && ptD && used &&
            (atlas != s_boundVsmAtlas || pt != s_boundVsmPT || cubo != s_boundVsmUBO || atlasD != s_boundVsmAtlasD)) {
            VkDescriptorImageInfo  aI{ VSM::GetSampler(), atlas, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkDescriptorBufferInfo pI{ pt, 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo uI{ cubo, 0, VK_WHOLE_SIZE };
            VkDescriptorImageInfo  aDI{ VSM::GetSampler(), atlasD, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkDescriptorBufferInfo pDI{ ptD, 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo usI{ used, 0, VK_WHOLE_SIZE };
            for (u32 i = 0; i < kFramesInFlight; ++i) {
                VkWriteDescriptorSet w[6]{};
                w[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; w[0].dstSet = s_injSet[i]; w[0].dstBinding = 7; w[0].descriptorCount = 1; w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[0].pImageInfo = &aI;
                w[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; w[1].dstSet = s_injSet[i]; w[1].dstBinding = 8; w[1].descriptorCount = 1; w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[1].pBufferInfo = &pI;
                w[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; w[2].dstSet = s_injSet[i]; w[2].dstBinding = 9; w[2].descriptorCount = 1; w[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[2].pBufferInfo = &uI;
                w[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; w[3].dstSet = s_injSet[i]; w[3].dstBinding = 14; w[3].descriptorCount = 1; w[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[3].pImageInfo = &aDI;
                w[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; w[4].dstSet = s_injSet[i]; w[4].dstBinding = 15; w[4].descriptorCount = 1; w[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[4].pBufferInfo = &pDI;
                w[5] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; w[5].dstSet = s_injSet[i]; w[5].dstBinding = 16; w[5].descriptorCount = 1; w[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[5].pBufferInfo = &usI;
                vkUpdateDescriptorSets(VulkanHW.m_Device, 6, w, 0, nullptr);
            }
            s_boundVsmAtlas = atlas; s_boundVsmPT = pt; s_boundVsmUBO = cubo; s_boundVsmAtlasD = atlasD;
        }
    }

    // ---- Prev-frame scene depth for the depth rejection — RE-WRITTEN every Execute
    // for the current slot (the depth attachment can be recreated on resize; a
    // per-slot write each frame sidesteps every stale-view hazard). Falls back to a
    // known-valid map when rejection is off (the shader path is gated by sun_dir.w).
    {
        VkImageView dv = depthRej ? sceneDepthPrev : (ShadowMap::GetRainView() ? ShadowMap::GetRainView() : ShadowMap::GetView());
        VkDescriptorImageInfo dI{ ShadowMap::GetSampler(), dv, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        // Terrain height field — the bake above may have just created it. Falls back
        // to a known-valid depth map; the shader path is gated by terra.w, so the
        // fallback is never sampled, it only keeps the binding legal.
        VkImageView th = s_terraView ? s_terraView : (ShadowMap::GetRainView() ? ShadowMap::GetRainView() : ShadowMap::GetView());
        VkDescriptorImageInfo tI{ ShadowMap::GetSampler(), th, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        // Grass canopy — same deal: gated by canopy.x, the fallback only keeps it legal.
        VkImageView cv = s_canopyView ? s_canopyView : th;
        VkDescriptorImageInfo cI{ ShadowMap::GetSampler(), cv, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet w[3]{};
        w[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w[0].dstSet = s_injSet[slot]; w[0].dstBinding = 17; w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[0].pImageInfo = &dI;
        w[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w[1].dstSet = s_injSet[slot]; w[1].dstBinding = 18; w[1].descriptorCount = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[1].pImageInfo = &tI;
        w[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w[2].dstSet = s_injSet[slot]; w[2].dstBinding = 19; w[2].descriptorCount = 1;
        w[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[2].pImageInfo = &cI;
        vkUpdateDescriptorSets(VulkanHW.m_Device, 3, w, 0, nullptr);
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
        const float intPush[4] = { ps_r_vol_hillaire ? 1.0f : 0.0f, 0.f, 0.f, 0.f };
        vkCmdPushConstants(cmd, s_intLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(intPush), intPush);
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
    for (auto& sp : s_terraPipes) if (sp.pipe) vkDestroyPipeline(VulkanHW.m_Device, sp.pipe, nullptr);
    s_terraPipes.clear();
    if (s_terraLayout) { vkDestroyPipelineLayout(VulkanHW.m_Device, s_terraLayout, nullptr); s_terraLayout = VK_NULL_HANDLE; }
    if (s_terraView) { vkDestroyImageView(VulkanHW.m_Device, s_terraView, nullptr); s_terraView = VK_NULL_HANDLE; }
    if (s_terraImg)  { VK::Vram::DestroyImage(VulkanHW.m_Allocator, s_terraImg, s_terraAlloc); s_terraImg = VK_NULL_HANDLE; s_terraAlloc = VK_NULL_HANDLE; }
    if (s_canopyView) { vkDestroyImageView(VulkanHW.m_Device, s_canopyView, nullptr); s_canopyView = VK_NULL_HANDLE; }
    if (s_canopyImg) { VK::Vram::DestroyImage(VulkanHW.m_Allocator, s_canopyImg, s_canopyAlloc); s_canopyImg = VK_NULL_HANDLE; s_canopyAlloc = VK_NULL_HANDLE; }
    s_canopyOk = false; s_canopyVisuals = size_t(-1);
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
    s_boundVsmAtlas = VK_NULL_HANDLE; s_boundVsmAtlasD = VK_NULL_HANDLE; s_boundVsmPT = VK_NULL_HANDLE; s_boundVsmUBO = VK_NULL_HANDLE;
    s_haveHistory = false; s_prevValid = false; s_frame = 0;
    s_inited = false; s_failed = false; s_ready = false;
}

}}  // namespace VK::Vol
