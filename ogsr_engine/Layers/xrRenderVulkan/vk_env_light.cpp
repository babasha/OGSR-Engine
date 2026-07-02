// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan — shared per-frame environment lighting UBO. See vk_env_light.h.
#include "stdafx.h"
#include "vk_env_light.h"
#include "vk_buffer.h"                     // CVulkanBuffer
#include "vk_command_buffer.h"             // CVulkanCommandManager::FRAMES_IN_FLIGHT
#include "vk_shadow.h"                     // ShadowMap (binding 1 = shadow map, sun_vp)
#include "vk_vsm.h"                        // VSM receivers (bindings 14-16: atlas, page table, clipmap UBO)
#include "vk_clustered.h"                  // Clustered forward (bindings 17-19: lights, grid, indices)
#include "vk_water_sim.h"                  // WaterSim (binding 11 = water depth)
#include "vk_deform.h"                     // Deform (binding 20 = snow deform press field)
#include "vk_pass_skinned.h"               // Skinned_CollectFeet (snow footprint deformation)
#include "vk_texture.h"                    // CVulkanTexture (fallback ambient cube)
#include "vk_pass_sky.h"                   // SkyPass::AcquireAmbientCubes (hemisphere sky ambient)
#include "vk_pass_ssao.h"                  // GTAO result (binding 8, white fallback until ready)
#include "../../xr_3da/IGame_Persistent.h" // g_pGamePersistent->Environment()
#include "../../xr_3da/Environment.h"      // CEnvDescriptorMixer (sun_dir/sun_color/hemi/ambient)
#include "../../xr_3da/device.h"           // Device.vCameraPosition (light collection)

#include <cstring>

extern int   ps_r_ssao_debug;     // r_ssao_debug — draw the raw AO map (vk_console_min.cpp)
extern float ps_r_ssao_strength;  // r_ssao_strength — live AO depth knob
extern float ps_r_sun_boost;      // r_sun_boost — global sun multiplier (was ×1.25 literals in 5 shaders)
extern float ps_r_ambient_floor;  // r_ambient_floor — flat ambient lift (was +0.05 literals in 4 shaders)
extern float ps_r_wet_darken;     // r_wet_darken — wet albedo darkening strength (0..1)
extern float ps_r_wet_refl;       // r_wet_refl — wet sky-reflection strength
extern int   ps_r_wet_debug;      // r_wet_debug — draw the wet mask (negative darken = flag)
extern int   ps_r_pom;            // r_pom — parallax occlusion mapping on/off
extern float ps_r_pom_height;     // r_pom_height — POM march amplitude (UV space)
extern float ps_r_pom_steps;      // r_pom_steps — POM max ray samples
extern float ps_r_pom_far;        // r_pom_far — POM distance fade (m)
extern float ps_r_pom_blur;       // r_pom_blur — POM heightfield extra mip blur
extern float ps_r_pom_normal;     // r_pom_normal — POM normal-perturbation strength
extern float ps_r_pom_shadow;     // r_pom_shadow — POM groove self-shadow strength
extern float ps_r_pom_ao;         // r_pom_ao — POM view-independent contact AO strength
extern int   ps_r_pom_debug;      // r_pom_debug — draw the POM AO×self-shadow mask
extern int   ps_r_rain_enable;    // r_rain — master rain on/off (forces wetness 0 when off)
extern int   ps_r_ao_flat;        // r_ao_flat — debug: neutralize all ambient occlusion
extern float ps_r_pom_ceil;       // r_pom_ceil — POM strength on down-facing surfaces (ceilings)
extern float ps_r_pom_floor;      // r_pom_floor — POM strength on up-facing surfaces (floors)
extern int   ps_r_pom_terrain;    // r_pom_terrain — terrain POM enable (experimental, default off)
extern float ps_r_terrain_normal; // r_terrain_normal — terrain detail normal-mapping strength
extern float ps_r_terrain_ao;     // r_terrain_ao — terrain micro contact AO strength
extern int   ps_r_terrain_debug;  // r_terrain_debug — terrain debug view (0..3)
extern float ps_r_terrain_gloss;  // r_terrain_gloss — terrain dry sun-gloss strength
extern float ps_r_glass_opacity;  // r_glass_opacity — glass opacity ceiling (frag glass branches)
extern int   ps_r_puddle_debug;   // r_puddle_debug — draw the geometric puddle mask
extern int   ps_r_water_sim;      // r_water_sim — water flow sim enable (puddles from the sim)
extern float ps_r_water_murk;     // r_water_murk — volumetric absorption per metre
extern float ps_r_water_refract;  // r_water_refract — bottom refraction strength
extern float ps_r_spec_occ;       // r_spec_occ — bent-normal specular occlusion of wet reflections
extern int   ps_r_clustered;      // r_clustered — clustered forward light culling (vk_clustered)
extern int   ps_r_clustered_debug; // r_clustered_debug — per-cluster light-count heatmap
extern int   ps_r_light_occ;      // r_light_occ — dynamic-light terrain/static occlusion (ground map march)
extern int   ps_r_puddle_sss;     // r_puddle_sss — SSS per-pixel puddles (default source)
extern float ps_r_puddle_level;   // r_puddle_level — water rise level vs micro-height
extern float ps_r_puddle_scale;   // r_puddle_scale — macro puddle-body size (procedural mask freq)
extern int   ps_r_sf;             // r_sf — Surface Field master enable (consumers later)
extern int   ps_r_sf_debug;       // r_sf_debug — Surface Field debug view (0..5)
extern float ps_r_sf_eps;         // r_sf_eps — derive finite-difference epsilon (m)
extern float ps_r_snow;           // r_snow — TARGET snow coverage (Surface Field consumer)
extern float ps_r_snow_rate;      // r_snow_rate — snow accumulate/melt speed (per sec)
extern int   ps_r_snow_deform;        // r_snow_deform — footprint deformation enable
extern float ps_r_snow_deform_depth;  // r_snow_deform_depth — print press depth (m)
extern float ps_r_snow_deform_radius; // r_snow_deform_radius — print radius (m)
extern float ps_r_snow_deform_time;   // r_snow_deform_time — print lifetime (sec, time-decay)
extern int   ps_r_snow_deform_tex;    // r_snow_deform_tex — use the dense deform texture (vk_deform)
extern float ps_r_mud_deform;         // r_mud_deform — mud footprints on soft terrain (same deform texture, no snow needed)
extern float ps_r_mud_depth;          // r_mud_depth — mud print POM carve depth (fraction of the detail height range)
extern int   ps_r_snow_mesh;          // r_snow_mesh — dense snow surface mesh (VHM-style)

namespace VK { namespace EnvLight {

namespace {
    constexpr u32 kFramesInFlight = CVulkanCommandManager::FRAMES_IN_FLIGHT;

    // Per-slot stride: descriptor offsets must honour minUniformBufferOffsetAlignment
    // (64 on NVIDIA, up to 256 elsewhere). sizeof(LightUBO) alone is NOT aligned —
    // binding slot 1 at a misaligned offset is undefined behaviour (in practice the
    // GPU read garbage lights on alternate in-flight frames → scene-wide strobing).
    constexpr VkDeviceSize kSlotStride = (sizeof(LightUBO) + 255) & ~VkDeviceSize(255);

    bool                  s_inited     = false;
    bool                  s_failed     = false;
    VkDescriptorSetLayout s_setLayout  = VK_NULL_HANDLE;
    VkDescriptorPool      s_pool       = VK_NULL_HANDLE;
    VkDescriptorSet       s_set[kFramesInFlight] = {};
    CVulkanBuffer         s_ubo;
    CVulkanBuffer         s_dummyBuf;   // valid SSBO/UBO placeholder for VSM bindings 15/16 before VSM inits
    u8*                   s_mapped     = nullptr;
    VkDescriptorSet       s_current    = VK_NULL_HANDLE;

    // Hemisphere sky ambient: bindings 6/7 hold the two weather sky cubes
    // (owned by SkyPass). EnvLight owns a trilinear cube sampler + a 1×1×6
    // fallback so the set is always complete (SkyPass::Init runs AFTER us). Each
    // slot tracks which cube views it currently holds; Update() rebinds only on
    // change — fence-safe because Begin() already waited the slot's fence.
    VkSampler             s_cubeSampler = VK_NULL_HANDLE;
    CVulkanTexture*       s_fallbackCube = nullptr;
    VkImageView           s_boundCube0[kFramesInFlight] = {};
    VkImageView           s_boundCube1[kFramesInFlight] = {};

    // GTAO (binding 8): white 1×1 fallback until the SSAO pass produced a
    // frame (SSAOPass::Init runs after us, and the prepass may be off).
    CVulkanTexture*       s_fallbackWhite = nullptr;
    VkImageView           s_boundAO[kFramesInFlight] = {};

    // SSIL (binding 15): BLACK 1×1 fallback — black IL → ssilBoost() returns 1.0
    // (no bounce) until the GTAO+IL pass has a frame / r_ssil is on.
    CVulkanTexture*       s_fallbackBlack = nullptr;
    VkImageView           s_boundIL[kFramesInFlight] = {};

    // Spot light cookie (flashlight beam texture, binding 10): loaded once per
    // texture name (Torch config "spot_texture"), per-slot bound-view tracking.
    VkImageView           s_boundCookie[kFramesInFlight] = {};
    xr_map<shared_str, CVulkanTexture*> s_cookieCache;

    // Water depth (sim, binding 11): per-slot bound-view tracking. Swapped from
    // the white fallback to the WaterSim buffer once it exists (lazy, like AO).
    VkImageView           s_boundWater[kFramesInFlight] = {};
    // Water velocity (sim, binding 12): same lazy-bind tracking.
    VkImageView           s_boundFlow[kFramesInFlight]  = {};
    // Ground-height map (binding 13, SSS puddle real-dip placement): lazy-bind.
    VkImageView           s_boundGround[kFramesInFlight] = {};
    // Snow deform press field (binding 20, vk_deform): white fallback until the
    // first Deform dispatch creates it (lazy, like water).
    VkImageView           s_boundDeform[kFramesInFlight] = {};

    // Clustered forward (bindings 17/18/19 = lights/grid/indices SSBOs). Start on
    // the dummy buffer; Update() swaps in the real vk_clustered buffers once that
    // module has inited (lazy, one-shot per slot). False = still on the dummy.
    bool                  s_boundCluster[kFramesInFlight] = {};

    // Load-once cookie lookup: "$game_textures$\<name>.dds". Negative results
    // cached too (null) so a missing texture logs once, not per frame.
    VkImageView GetCookieView(const shared_str& name)
    {
        auto it = s_cookieCache.find(name);
        if (it != s_cookieCache.end()) return it->second ? it->second->GetView() : VK_NULL_HANDLE;
        CVulkanTexture* t = nullptr;
        string_path leaf, full;
        xr_sprintf(leaf, "%s.dds", name.c_str());
        FS.update_path(full, "$game_textures$", leaf);
        if (FS.exist(full)) {
            t = xr_new<CVulkanTexture>();
            if (!t->LoadDDS(full, /*applyBCSwizzle*/ false)) { xr_delete(t); t = nullptr; }
        }
        if (t) Msg("[VK Light] spot cookie loaded: '%s'", name.c_str());
        else   Msg("![VK Light] spot cookie not found: '%s'", name.c_str());
        s_cookieCache.emplace(name, t);
        return t ? t->GetView() : VK_NULL_HANDLE;
    }

    // 2.6 was LDR-era compensation (pre-HDR/auto-exposure) — with the tonemap
    // it flooded terrain so bright the sun shadow under trees washed out
    // ("земля не имеет затенения"). R4's effective ambient scale is ~1.0
    // (hmodel: env_d × hemi, no boost) — and in SHADE ambient is the only
    // light, so any boost here brightens shadowed ground 1:1 vs R4 while the
    // sun hides it elsewhere. 1.1 ≈ R4 shade depth; auto-exposure owns the
    // overall brightness.
    constexpr float kAmbientScale = 1.1f;   // sky-ambient strength knob (tune in-game)
    constexpr float kAmbientLod   = 0.0f;   // sky cubes are BC1/BC3 single-mip → mip 0 (no blur available)
}

VkDescriptorSetLayout GetSetLayout() { return s_setLayout; }
VkDescriptorSet       GetCurrentSet() { return s_current; }

bool Init()
{
    if (s_inited) return !s_failed;
    s_inited = true;

    // The shadow map (binding 1) must exist before we write the descriptor sets.
    ShadowMap::Init();

    // Set layout: binding 0 = UBO, 1 = far sun map, 2 = spot map, 3 = point
    // shadow cube, 4 = sun cascade 0, 5 = sun cascade 1, 6/7 = sky ambient
    // cubes (hemisphere fill), 8 = GTAO, 9 = rain occlusion map (wetness),
    // 10 = spot light cookie, 11 = water depth (sim), 12 = water velocity (sim),
    // 13 = clean ground-height map. Bindings 11-13 belong to the PARKED experimental
    // water-sim (r_water_sim, off by default); the SSS puddle path doesn't use them
    // but they stay bound (harmless) so the sim can be switched on without relayout.
    // All FRAGMENT.
    VkDescriptorSetLayoutBinding b[22]{};
    b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    // Also visible to VS/TES: snow geometric displacement reads sf_params.w (coverage).
    b[0].descriptorCount = 1; b[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
    for (u32 i = 1; i < 14; ++i) {
        b[i].binding = i; b[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    // VSM receivers (Phase 1C): 14 = atlas sampler, 15 = page table SSBO, 16 = clipmap UBO.
    b[14].binding = 14; b[14].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; b[14].descriptorCount = 1; b[14].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    b[15].binding = 15; b[15].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;         b[15].descriptorCount = 1; b[15].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    b[16].binding = 16; b[16].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;         b[16].descriptorCount = 1; b[16].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    // Clustered forward (vk_clustered): 17 = light list SSBO, 18 = cluster grid
    // SSBO (per-cluster count), 19 = light index list SSBO. Read only when
    // cluster_params.w (r_clustered) is set; bound to the dummy buffer otherwise.
    for (u32 i = 17; i < 20; ++i) {
        b[i].binding = i; b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    // Snow deform press field (vk_deform): 20 = press texture. Read by the terrain
    // tessellation eval (broad groove) + fragment (sharp dimple).
    b[20].binding = 20; b[20].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; b[20].descriptorCount = 1;
    // + TESS_CONTROL: the terrain TCS samples the press field to gate subdivision
    // to patches that actually contain prints (mud path — no always-on tess cost).
    b[20].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
    // SSIL (binding 21): one-bounce indirect-light buffer, sampled by the forward
    // receivers (ssilBoost multiplies the ambient term). All FRAGMENT.
    b[21].binding = 21; b[21].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; b[21].descriptorCount = 1;
    b[21].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    // The snow MESH (vk_pass_snow) vertex shader samples the RAIN map (9, base height)
    // + deform field (20) to place + displace its dense grid -> need VERTEX visibility.
    b[9].stageFlags  |= VK_SHADER_STAGE_VERTEX_BIT;
    b[13].stageFlags |= VK_SHADER_STAGE_VERTEX_BIT;
    VkDescriptorSetLayoutCreateInfo slci{};
    slci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    slci.bindingCount = 22; slci.pBindings = b;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &slci, nullptr, &s_setLayout) != VK_SUCCESS) {
        Msg("![VK EnvLight] set layout create failed"); s_failed = true; return false;
    }

    VkDescriptorPoolSize ps[3]{
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         kFramesInFlight * 2 },    // LightUBO + VSM clipmap UBO
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kFramesInFlight * 16 },   // 13 shadow/sky/ao + VSM atlas + deform + SSIL
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         kFramesInFlight * 4 },    // VSM page table + 3 cluster SSBOs
    };
    VkDescriptorPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = kFramesInFlight; pci.poolSizeCount = 3; pci.pPoolSizes = ps;
    if (vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &s_pool) != VK_SUCCESS) {
        Msg("![VK EnvLight] pool create failed"); s_failed = true; return false;
    }

    VkDescriptorSetLayout layouts[kFramesInFlight];
    for (u32 i = 0; i < kFramesInFlight; ++i) layouts[i] = s_setLayout;
    VkDescriptorSetAllocateInfo dai{};
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = s_pool; dai.descriptorSetCount = kFramesInFlight; dai.pSetLayouts = layouts;
    if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, s_set) != VK_SUCCESS) {
        Msg("![VK EnvLight] alloc sets failed"); s_failed = true; return false;
    }

    // Host-visible UBO, one aligned LightUBO region per in-flight slot; each set
    // bound to its slice at an alignment-safe offset.
    s_ubo.Create(kSlotStride * kFramesInFlight,
                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
    s_mapped = static_cast<u8*>(s_ubo.Map());
    if (!s_mapped) { Msg("![VK EnvLight] UBO map failed"); s_failed = true; return false; }

    // Tiny valid SSBO/UBO bound to VSM bindings 15/16 until VSM initialises (lazy on
    // first r_vsm). Receivers gate VSM sampling on shadow_params.w, so it's never read.
    s_dummyBuf.Create(256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

    // Trilinear cube sampler (mips → blurred sky = diffuse irradiance) + a
    // neutral 1×1×6 fallback so bindings 6/7 are valid before SkyPass loads the
    // real weather cubes (SkyPass::Init runs after us).
    {
        VkSamplerCreateInfo sci{};
        sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter    = VK_FILTER_LINEAR; sci.minFilter = VK_FILTER_LINEAR;
        sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.maxAnisotropy = 1.0f; sci.minLod = 0.0f; sci.maxLod = VK_LOD_CLAMP_NONE;
        if (vkCreateSampler(VulkanHW.m_Device, &sci, nullptr, &s_cubeSampler) != VK_SUCCESS) {
            Msg("![VK EnvLight] cube sampler create failed"); s_failed = true; return false;
        }
    }
    s_fallbackCube = xr_new<CVulkanTexture>();
    s_fallbackCube->m_bCubemap = true; s_fallbackCube->m_ArrayLayers = 6;
    s_fallbackCube->Create(1, 1, VK_FORMAT_R8G8B8A8_UNORM, 1);
    {
        u8 face[24];
        for (u32 i = 0; i < 6; ++i) { face[i*4+0]=128; face[i*4+1]=128; face[i*4+2]=128; face[i*4+3]=255; }
        s_fallbackCube->UploadData(face, sizeof(face));
    }
    const VkImageView fbView = s_fallbackCube->GetView();

    // White 1×1 for binding 8 — "no occlusion" until GTAO renders.
    s_fallbackWhite = xr_new<CVulkanTexture>();
    s_fallbackWhite->Create(1, 1, VK_FORMAT_R8G8B8A8_UNORM, 1);
    {
        const u8 white[4] = { 255, 255, 255, 255 };
        s_fallbackWhite->UploadData(white, sizeof(white));
    }
    const VkImageView fbWhite = s_fallbackWhite->GetView();

    // Black 1×1 for binding 15 (SSIL) — "no bounce" until the IL pass renders.
    s_fallbackBlack = xr_new<CVulkanTexture>();
    s_fallbackBlack->Create(1, 1, VK_FORMAT_R8G8B8A8_UNORM, 1);
    {
        const u8 black[4] = { 0, 0, 0, 255 };
        s_fallbackBlack->UploadData(black, sizeof(black));
    }
    const VkImageView fbBlack = s_fallbackBlack->GetView();

    for (u32 i = 0; i < kFramesInFlight; ++i) {
        VkDescriptorBufferInfo bi{ s_ubo.GetHandle(), kSlotStride * i, sizeof(LightUBO) };
        VkDescriptorImageInfo  si[5]{};
        si[0].sampler = ShadowMap::GetSampler(); si[0].imageView = ShadowMap::GetView();           // 1: far sun
        si[1].sampler = ShadowMap::GetSampler(); si[1].imageView = ShadowMap::GetSpotView();       // 2: spot
        si[2].sampler = ShadowMap::GetSampler(); si[2].imageView = ShadowMap::GetPointCubeView();  // 3: point cube
        si[3].sampler = ShadowMap::GetSampler(); si[3].imageView = ShadowMap::GetCascadeView(0);   // 4: sun cascade 0
        si[4].sampler = ShadowMap::GetSampler(); si[4].imageView = ShadowMap::GetCascadeView(1);   // 5: sun cascade 1
        for (auto& s : si) s.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        // Binding 9: rain occlusion map (white border sampler — outside = open sky).
        VkDescriptorImageInfo rainI{ ShadowMap::GetSampler(), ShadowMap::GetRainView(),
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

        // Bindings 6/7: sky ambient cubes — start on the neutral fallback;
        // Update() swaps in the real weather cubes once SkyPass has them.
        VkDescriptorImageInfo cube[2]{};
        for (auto& c : cube) {
            c.sampler = s_cubeSampler; c.imageView = fbView;
            c.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
        s_boundCube0[i] = fbView; s_boundCube1[i] = fbView;

        // Binding 8: GTAO — white fallback; Update() swaps in the real AO view.
        VkDescriptorImageInfo aoI{ s_cubeSampler, fbWhite, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        s_boundAO[i] = fbWhite;

        // Binding 21: SSIL — black fallback (ssilBoost=1); Update() swaps in the real IL view.
        VkDescriptorImageInfo ilI{ s_cubeSampler, fbBlack, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        s_boundIL[i] = fbBlack;

        // Binding 10: spot cookie — white until the flashlight's texture loads
        // (shadow_params.z gates sampling anyway).
        VkDescriptorImageInfo ckI{ s_cubeSampler, fbWhite, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        s_boundCookie[i] = fbWhite;

        // Binding 11: water depth (sim) — white fallback until WaterSim runs;
        // Update() swaps in the real water buffer once it exists.
        VkDescriptorImageInfo wtI{ s_cubeSampler, fbWhite, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        s_boundWater[i] = fbWhite;
        // Binding 12: water velocity — white fallback until WaterSim runs.
        VkDescriptorImageInfo flI{ s_cubeSampler, fbWhite, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        s_boundFlow[i] = fbWhite;
        // Binding 13: clean ground-height map — white fallback until rendered;
        // Update() swaps in ShadowMap::GetGroundView once it exists.
        VkDescriptorImageInfo gdI{ s_cubeSampler, fbWhite, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        s_boundGround[i] = fbWhite;
        // Binding 20: snow deform press field — white fallback until vk_deform runs;
        // Update() swaps in Deform::GetView once it exists (gated by deform_tex.x so
        // the white fallback is never actually sampled before then).
        VkDescriptorImageInfo dfI{ s_cubeSampler, fbWhite, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        s_boundDeform[i] = fbWhite;

        // Clustered forward (bindings 17/18/19): valid SSBO placeholder until the
        // vk_clustered module inits; Update() swaps in the real buffers (gated by
        // cluster_params.w so the dummy is never actually read).
        VkDescriptorBufferInfo clI[3]{
            { s_dummyBuf.GetHandle(), 0, VK_WHOLE_SIZE },
            { s_dummyBuf.GetHandle(), 0, VK_WHOLE_SIZE },
            { s_dummyBuf.GetHandle(), 0, VK_WHOLE_SIZE },
        };

        VkWriteDescriptorSet w[19]{};
        w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[0].dstSet = s_set[i]; w[0].dstBinding = 0; w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[0].pBufferInfo = &bi;

        // Shadow maps (sun/spot/point/cascades) — ShadowMap::Init ran above, views exist.
        u32 count = 1;
        for (u32 m = 0; m < 5; ++m) {
            if (si[m].imageView == VK_NULL_HANDLE || si[m].sampler == VK_NULL_HANDLE) continue;
            w[count].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[count].dstSet = s_set[i]; w[count].dstBinding = 1 + m; w[count].descriptorCount = 1;
            w[count].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[count].pImageInfo = &si[m];
            ++count;
        }
        for (u32 c = 0; c < 2; ++c) {
            w[count].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[count].dstSet = s_set[i]; w[count].dstBinding = 6 + c; w[count].descriptorCount = 1;
            w[count].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[count].pImageInfo = &cube[c];
            ++count;
        }
        w[count].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[count].dstSet = s_set[i]; w[count].dstBinding = 8; w[count].descriptorCount = 1;
        w[count].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[count].pImageInfo = &aoI;
        ++count;
        w[count].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[count].dstSet = s_set[i]; w[count].dstBinding = 21; w[count].descriptorCount = 1;
        w[count].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[count].pImageInfo = &ilI;
        ++count;
        if (rainI.imageView != VK_NULL_HANDLE && rainI.sampler != VK_NULL_HANDLE) {
            w[count].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[count].dstSet = s_set[i]; w[count].dstBinding = 9; w[count].descriptorCount = 1;
            w[count].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[count].pImageInfo = &rainI;
            ++count;
        }
        w[count].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[count].dstSet = s_set[i]; w[count].dstBinding = 10; w[count].descriptorCount = 1;
        w[count].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[count].pImageInfo = &ckI;
        ++count;
        w[count].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[count].dstSet = s_set[i]; w[count].dstBinding = 11; w[count].descriptorCount = 1;
        w[count].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[count].pImageInfo = &wtI;
        ++count;
        w[count].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[count].dstSet = s_set[i]; w[count].dstBinding = 12; w[count].descriptorCount = 1;
        w[count].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[count].pImageInfo = &flI;
        ++count;
        w[count].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[count].dstSet = s_set[i]; w[count].dstBinding = 13; w[count].descriptorCount = 1;
        w[count].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[count].pImageInfo = &gdI;
        ++count;
        for (u32 k = 0; k < 3; ++k) {
            w[count].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[count].dstSet = s_set[i]; w[count].dstBinding = 17 + k; w[count].descriptorCount = 1;
            w[count].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[count].pBufferInfo = &clI[k];
            ++count;
        }
        w[count].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[count].dstSet = s_set[i]; w[count].dstBinding = 20; w[count].descriptorCount = 1;
        w[count].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[count].pImageInfo = &dfI;
        ++count;
        vkUpdateDescriptorSets(VulkanHW.m_Device, count, w, 0, nullptr);
    }

    s_current = s_set[0];
    Msg("[VK EnvLight] init OK (UBO %u bytes x %u slots)", (u32)sizeof(LightUBO), kFramesInFlight);
    return true;
}

void Update(u32 slot)
{
    if (s_failed || !s_mapped) return;
    if (slot >= kFramesInFlight) slot = 0;

    // Neutral fallback when env isn't up yet (e.g. main menu before a level).
    LightUBO ub{};
    ub.sun_dir[0]=0.f; ub.sun_dir[1]=-1.f; ub.sun_dir[2]=0.f;
    ub.sun_color[0]=ub.sun_color[1]=ub.sun_color[2]=0.6f;
    ub.hemi_color[0]=ub.hemi_color[1]=ub.hemi_color[2]=0.45f;
    ub.ambient[0]=ub.ambient[1]=ub.ambient[2]=0.05f;
    // Fog fallback: pushed far away → effectively no fog before a level loads.
    ub.fog_params[0]=0.f; ub.fog_params[1]=1e6f; ub.fog_params[2]=1e6f; ub.fog_params[3]=0.f;
    if (g_pGamePersistent) {
        if (auto* E = g_pGamePersistent->Environment().CurrentEnv) {
            ub.sun_dir[0]=E->sun_dir.x;  ub.sun_dir[1]=E->sun_dir.y;  ub.sun_dir[2]=E->sun_dir.z;
            ub.sun_color[0]=E->sun_color.x; ub.sun_color[1]=E->sun_color.y; ub.sun_color[2]=E->sun_color.z;
            ub.hemi_color[0]=E->hemi_color.x; ub.hemi_color[1]=E->hemi_color.y; ub.hemi_color[2]=E->hemi_color.z; ub.hemi_color[3]=E->hemi_color.w;
            ub.ambient[0]=E->ambient.x; ub.ambient[1]=E->ambient.y; ub.ambient[2]=E->ambient.z;

            // Distance fog (R4 cl_fog_params/cl_fog_color binders + combine_1.ps):
            // fog_near/far are derived in CEnvDescriptorMixer (Environment_misc.cpp:
            // near = (1-density)*0.85*dist, far = 0.99*dist); fog_color is the env
            // colour the sky/horizon use, so the haze tints with time-of-day.
            const float fn = E->fog_near, ff = E->fog_far;
            const float r  = (ff > fn + 1e-3f) ? 1.f / (ff - fn) : 0.f;
            ub.fog_params[0] = -fn * r; ub.fog_params[1] = fn; ub.fog_params[2] = ff; ub.fog_params[3] = r;
            ub.fog_color[0] = E->fog_color.x; ub.fog_color[1] = E->fog_color.y; ub.fog_color[2] = E->fog_color.z;

            // One-time dump of the real env magnitudes — to tune the shader balance.
            static bool s_diag = false;
            if (!s_diag) { s_diag = true;
                Msg("[VK Light] sun_color=(%.3f,%.3f,%.3f) hemi=(%.3f,%.3f,%.3f) ambient=(%.3f,%.3f,%.3f) sun_dir=(%.3f,%.3f,%.3f)",
                    E->sun_color.x,E->sun_color.y,E->sun_color.z, E->hemi_color.x,E->hemi_color.y,E->hemi_color.z,
                    E->ambient.x,E->ambient.y,E->ambient.z, E->sun_dir.x,E->sun_dir.y,E->sun_dir.z);
            }
        }
    }
    // LDR-era brightness hacks, centralized: the sun boost (was a ×1.25 literal
    // in five shaders) and the ambient floor (was +0.05 in four) apply ONCE
    // here, so every receiver consumes FINAL values. Live knobs r_sun_boost /
    // r_ambient_floor; 1.0/0.0 = raw env values. The grass/tree sun pushes get
    // the same boost in their managers (they bypass this UBO); sunshafts
    // kDensity is retuned ÷1.25 to keep shaft brightness unchanged.
    for (int c = 0; c < 3; ++c) {
        ub.sun_color[c] *= ps_r_sun_boost;
        ub.ambient[c]   += ps_r_ambient_floor;
    }

    // Sun light view·proj for the shadow lookup (Pass_SunShadow ran earlier this
    // frame and stored it). Fmatrix is 16 floats row-major → straight copy.
    memcpy(ub.sun_vp, &ShadowMap::GetLightVP(), sizeof(ub.sun_vp));

    // Dynamic point/spot lights (STEP 3): nearest active ones around the camera.
    // Same per-frame result Pass_SunShadow used to render the dynamic shadow maps,
    // so the shadowed indices match the array the shaders iterate.
    const auto& FL = Lights::CollectFrame(Device.vCameraPosition);
    memcpy(ub.lights, FL.gpu, sizeof(ub.lights));   // first kMaxGpuLights → UBO (foliage + non-clustered fallback)
    // counts.x drives the UBO lights[16] loop — clamp it so a >16-light frame
    // never reads past the UBO array (the clustered path reads the SSBO instead).
    ub.counts[0] = float(_min(FL.count, kMaxGpuLights));
    // Clustered forward: upload ALL collected lights to this slot's SSBO so the
    // compute cull (Pass_World) can bin them. r_clustered_debug also activates the
    // machinery (so the heatmap works without also typing r_clustered 1). This
    // also lazily inits the module on the first frame it's switched on.
    const bool clusterActive = (ps_r_clustered || ps_r_clustered_debug);
    if (clusterActive) {
        Clustered::UploadLights(FL, slot);
        // Diag (throttled ~3 s, only with r_clustered_debug): the shadowed picks'
        // range + distance-to-eye — to see why a near light might not bin.
        if (ps_r_clustered_debug) {
            static u32 s_clDiagCd = 0;
            if (s_clDiagCd == 0) {
                s_clDiagCd = 180;
                const Fvector& e = Device.vCameraPosition;
                auto logL = [&](const char* tag, int idx) {
                    if (idx < 0 || idx >= int(FL.count)) { Msg("[VK Clustered] %s: none", tag); return; }
                    const auto& g = FL.gpu[idx];
                    const float dx = g.pos[0]-e.x, dy = g.pos[1]-e.y, dz = g.pos[2]-e.z;
                    Msg("[VK Clustered] %s idx=%d range=%.1f dist=%.2f spotFlag=%.0f", tag, idx, g.pos[3], _sqrt(dx*dx+dy*dy+dz*dz), g.color[3]);
                };
                logL("SPOT(flashlight)", FL.spotIdx);
                logL("POINT(campfire)",  FL.pointIdx);
                Msg("[VK Clustered] collected=%u eye=(%.1f,%.1f,%.1f)", FL.count, e.x, e.y, e.z);
            } else --s_clDiagCd;
        }
    }
    memcpy(ub.spot_vp, &ShadowMap::GetSpotVP(), sizeof(ub.spot_vp));
    ub.shadow_params[0] = float(FL.spotIdx);
    ub.shadow_params[1] = float(FL.pointIdx);
    // Spot cookie (R4 projective light texture — the flashlight beam pattern):
    // load the picked spot's texture once, bind it at binding 10 for this slot,
    // and flag its presence in shadow_params.z (shaders project it with the
    // SAME spot_vp the shadow lookup uses).
    {
        VkImageView cookie = VK_NULL_HANDLE;
        if (FL.spotIdx >= 0 && FL.spotTexture.size())
            cookie = GetCookieView(FL.spotTexture);
        ub.shadow_params[2] = (cookie != VK_NULL_HANDLE) ? 1.f : 0.f;
        VkImageView want = cookie ? cookie
                                  : (s_fallbackWhite ? s_fallbackWhite->GetView() : VK_NULL_HANDLE);
        if (want != VK_NULL_HANDLE && want != s_boundCookie[slot]) {
            VkDescriptorImageInfo ii{ s_cubeSampler, want, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkWriteDescriptorSet wck{};
            wck.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wck.dstSet = s_set[slot]; wck.dstBinding = 10; wck.descriptorCount = 1;
            wck.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; wck.pImageInfo = &ii;
            vkUpdateDescriptorSets(VulkanHW.m_Device, 1, &wck, 0, nullptr);
            s_boundCookie[slot] = want;
        }
    }
    ub.shadow_params[3] = VK::VSM::MaskReady() ? 1.f : 0.f;  // VSM receiver gate (screen-space mask resolved)
    // Sun cascade VPs — recomputed every frame in Pass_SunShadow (earlier this
    // frame), so receivers always sample the freshly rendered cascade maps.
    memcpy(ub.sun_near_vp, &ShadowMap::GetCascadeVP(0), sizeof(ub.sun_near_vp));
    memcpy(ub.sun_c1_vp,   &ShadowMap::GetCascadeVP(1), sizeof(ub.sun_c1_vp));

    // Camera world position — receivers compute fog distance = length(wp - eye).
    ub.eye_pos[0] = Device.vCameraPosition.x;
    ub.eye_pos[1] = Device.vCameraPosition.y;
    ub.eye_pos[2] = Device.vCameraPosition.z;

    // Hemisphere sky ambient (R4 hmodel): fetch the current weather cubes from
    // SkyPass and rebind this slot's bindings 6/7 if they changed (fence-safe —
    // Begin waited this slot's fence). Weight cross-fades the two cubes.
    float skyWeight = 0.f;
    {
        VkImageView v0 = VK_NULL_HANDLE, v1 = VK_NULL_HANDLE, samp = VK_NULL_HANDLE;
        VkSampler   skSamp = VK_NULL_HANDLE;
        float       w = 0.f;
        if (SkyPass::AcquireAmbientCubes(v0, v1, skSamp, w)) {
            skyWeight = w;
            if (v0 && v1 && (v0 != s_boundCube0[slot] || v1 != s_boundCube1[slot])) {
                VkDescriptorImageInfo cube[2]{};
                cube[0].sampler = s_cubeSampler; cube[0].imageView = v0;
                cube[1].sampler = s_cubeSampler; cube[1].imageView = v1;
                cube[0].imageLayout = cube[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                VkWriteDescriptorSet w2[2]{};
                for (u32 c = 0; c < 2; ++c) {
                    w2[c].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    w2[c].dstSet = s_set[slot]; w2[c].dstBinding = 6 + c; w2[c].descriptorCount = 1;
                    w2[c].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w2[c].pImageInfo = &cube[c];
                }
                vkUpdateDescriptorSets(VulkanHW.m_Device, 2, w2, 0, nullptr);
                s_boundCube0[slot] = v0; s_boundCube1[slot] = v1;
                static bool s_skyDiag = false;
                if (!s_skyDiag) { s_skyDiag = true;
                    Msg("[VK SkyAmb] real sky cubes bound (slot %u, weight=%.2f, scale=%.2f, lod=%.1f)",
                        slot, w, kAmbientScale, kAmbientLod);
                }
            }
            (void)samp;
        } else {
            static bool s_skyFail = false;
            if (!s_skyFail) { s_skyFail = true; Msg("![VK SkyAmb] AcquireAmbientCubes returned false — fallback grey ambient"); }
        }
    }
    ub.sky_params[0] = skyWeight;
    ub.sky_params[1] = kAmbientScale;
    ub.sky_params[2] = kAmbientLod;
    // Animation clock for the wet-surface ripples (wrapped to keep float sin()
    // precision; 1000×2π → a re-phase only every ~1.7 h).
    ub.sky_params[3] = fmodf(Device.fTimeGlobal, 6283.185f);

    // GTAO (binding 8): swap the real AO view in when the pass has one, back to
    // the white fallback when it doesn't (prepass off / not rendered yet).
    // Strength 0 keeps the receivers' mix() on 1.0 either way.
    {
        VkImageView ao = SSAOPass::GetResultView();
        VkSampler   aoSamp = SSAOPass::GetSampler();
        if (ao == VK_NULL_HANDLE || aoSamp == VK_NULL_HANDLE) {
            ao = s_fallbackWhite ? s_fallbackWhite->GetView() : VK_NULL_HANDLE;
            aoSamp = s_cubeSampler;
        }
        if (ao != VK_NULL_HANDLE && ao != s_boundAO[slot]) {
            VkDescriptorImageInfo ii{ aoSamp, ao, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = s_set[slot]; w.dstBinding = 8; w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w.pImageInfo = &ii;
            vkUpdateDescriptorSets(VulkanHW.m_Device, 1, &w, 0, nullptr);
            s_boundAO[slot] = ao;
        }
        // Debug: confirm which image binding 8 actually holds (real AO vs the
        // white fallback) — pairs with the [VK SSAO] readback stats.
        if (ps_r_ssao_debug) {
            static u32 s_aoLogCd = 0;
            if (s_aoLogCd == 0) {
                s_aoLogCd = 300;
                const bool real = (SSAOPass::GetResultView() != VK_NULL_HANDLE)
                               && (s_boundAO[slot] == SSAOPass::GetResultView());
                Msg("[VK SSAO] binding8 slot %u: %s (view %p, result %p), strength %.2f",
                    slot, real ? "REAL AO" : "FALLBACK/WHITE",
                    (void*)s_boundAO[slot], (void*)SSAOPass::GetResultView(),
                    SSAOPass::Strength() * ps_r_ssao_strength);
            } else --s_aoLogCd;
        }
    }

    // SSIL (binding 15): swap the real IL view in when r_ssil is on and the pass
    // has a result, back to the BLACK fallback otherwise (ssilBoost → 1.0 = no-op).
    {
        VkImageView il   = SSAOPass::GetILResultView();
        VkSampler   ilSp = SSAOPass::GetSampler();
        if (il == VK_NULL_HANDLE || ilSp == VK_NULL_HANDLE) {
            il = s_fallbackBlack ? s_fallbackBlack->GetView() : VK_NULL_HANDLE;
            ilSp = s_cubeSampler;
        }
        if (il != VK_NULL_HANDLE && il != s_boundIL[slot]) {
            VkDescriptorImageInfo ii{ ilSp, il, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = s_set[slot]; w.dstBinding = 21; w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w.pImageInfo = &ii;
            vkUpdateDescriptorSets(VulkanHW.m_Device, 1, &w, 0, nullptr);
            s_boundIL[slot] = il;
        }
    }

    // Binding 11: water depth (sim). Swap the white fallback for the real water
    // buffer once WaterSim has created it (lazy, one update per slot).
    {
        VkImageView wv = WaterSim::GetStateView();
        VkSampler   ws = WaterSim::GetSampler();
        if (wv != VK_NULL_HANDLE && ws != VK_NULL_HANDLE && wv != s_boundWater[slot]) {
            VkDescriptorImageInfo ii{ ws, wv, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = s_set[slot]; w.dstBinding = 11; w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w.pImageInfo = &ii;
            vkUpdateDescriptorSets(VulkanHW.m_Device, 1, &w, 0, nullptr);
            s_boundWater[slot] = wv;
        }
        VkImageView fv = WaterSim::GetVelView();
        if (fv != VK_NULL_HANDLE && ws != VK_NULL_HANDLE && fv != s_boundFlow[slot]) {
            VkDescriptorImageInfo ii{ ws, fv, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = s_set[slot]; w.dstBinding = 12; w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w.pImageInfo = &ii;
            vkUpdateDescriptorSets(VulkanHW.m_Device, 1, &w, 0, nullptr);
            s_boundFlow[slot] = fv;
        }
        // Binding 13: clean ground-height map (no trees) for SSS real-dip puddle
        // placement. Same depth image / sampler as the rain map; rendered in
        // Pass_SunShadow when puddles/sim are on (else stays the white fallback).
        VkImageView gv = ShadowMap::GetGroundView();
        VkSampler   gs = ShadowMap::GetSampler();
        if (gv != VK_NULL_HANDLE && gs != VK_NULL_HANDLE && gv != s_boundGround[slot]) {
            VkDescriptorImageInfo ii{ gs, gv, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = s_set[slot]; w.dstBinding = 13; w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w.pImageInfo = &ii;
            vkUpdateDescriptorSets(VulkanHW.m_Device, 1, &w, 0, nullptr);
            s_boundGround[slot] = gv;
            { static bool s_gbl = false; if (!s_gbl) { s_gbl = true;
                Msg("[VK Puddle] ground map bound to EnvLight binding 13 (slot %u)", slot); } }
        }
        // Binding 20: snow deform press field (vk_deform). Swap the white fallback for
        // the real field once the first deform dispatch created it (lazy, like water).
        VkImageView dv = Deform::GetView();
        VkSampler   ds = Deform::GetSampler();
        if (dv != VK_NULL_HANDLE && ds != VK_NULL_HANDLE && dv != s_boundDeform[slot]) {
            VkDescriptorImageInfo ii{ ds, dv, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = s_set[slot]; w.dstBinding = 20; w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w.pImageInfo = &ii;
            vkUpdateDescriptorSets(VulkanHW.m_Device, 1, &w, 0, nullptr);
            s_boundDeform[slot] = dv;
        }
    }
    ub.ao_params[0] = Device.dwWidth  ? 1.f / float(Device.dwWidth)  : 0.f;
    ub.ao_params[1] = Device.dwHeight ? 1.f / float(Device.dwHeight) : 0.f;
    ub.ao_params[2] = SSAOPass::Strength() * ps_r_ssao_strength;
    ub.ao_params[3] = (SSAOPass::Strength() > 0.f && ps_r_ssao_debug) ? 1.f : 0.f;

    // Rain wetness: the engine accumulates wetness_factor from rain_density
    // (Environment_misc.cpp) — surfaces stay wet a while after the rain stops.
    // The rain map VP comes from Pass_SunShadow earlier this frame.
    memcpy(ub.rain_vp, &ShadowMap::GetRainVP(), sizeof(ub.rain_vp));
    if (g_pGamePersistent && ps_r_rain_enable) {
        const auto& env = g_pGamePersistent->Environment();
        const float density = env.CurrentEnv ? env.CurrentEnv->rain_density : 0.f;
        ub.rain_params[0] = density;
        // OWN wetness accumulator: the engine's env.wetness_factor is freely
        // overwritten by mod scripts (Gunslinger's weather script bounced it
        // 0.28 → 0.11 mid-rain). Soak ~45 s of full-density rain, dry ~3 min.
        // The engine value still contributes via max() if a script drives it
        // ABOVE ours (e.g. level designer forces wet ground).
        static float s_wet = 0.f;
        const float dt = clampr(Device.fTimeDelta, 0.f, 0.1f);
        if (density > 0.001f) s_wet += density * dt / 45.f;
        else                  s_wet -= dt / 180.f;
        clamp(s_wet, 0.f, 1.f);
        const float wet = _max(s_wet, clampr(env.wetness_factor, 0.f, 1.f));
        // sqrt = perceptual ramp: the wet look shows up in the first minute of
        // rain instead of creeping linearly toward the full soak.
        ub.rain_params[1] = _sqrt(wet);
    } else {
        // r_rain master OFF → force density + wetness to 0 so the wet shading and
        // puddles disappear immediately (the sim buffer freezes but isn't drawn).
        ub.rain_params[0] = 0.f;
        ub.rain_params[1] = 0.f;
    }
    // Debug flag rides as a NEGATIVE darken (z) — shaders early-out on z<0 and
    // draw the wet mask grayscale; no extra UBO slot needed.
    ub.rain_params[2] = ps_r_wet_debug ? -1.f : ps_r_wet_darken;
    ub.rain_params[3] = ps_r_wet_refl;

    // Scene camera for the tonemap SSR puddles: depth→world reconstruction
    // (frustum-ray basis) + world→uv projection for the ray march.
    memcpy(ub.scene_vp, &Device.mFullTransform, sizeof(ub.scene_vp));
    {
        const ProjTerms pt = DeriveProjTerms(Device.mFullTransform);
        ub.cam_dir[0] = pt.dir.x;   ub.cam_dir[1] = pt.dir.y;   ub.cam_dir[2] = pt.dir.z;   ub.cam_dir[3] = pt.p33;
        ub.cam_rightT[0] = pt.right.x * pt.tanX; ub.cam_rightT[1] = pt.right.y * pt.tanX;
        ub.cam_rightT[2] = pt.right.z * pt.tanX; ub.cam_rightT[3] = pt.p43;
        ub.cam_topT[0] = pt.top.x * pt.tanY; ub.cam_topT[1] = pt.top.y * pt.tanY;
        ub.cam_topT[2] = pt.top.z * pt.tanY; ub.cam_topT[3] = 0.f;

        // Clustered forward: the froxel-grid params the fragments use to find
        // their cluster (same near/far + slice formula the compute cull uses).
        // cluster_params.w gates the whole clustered path in the receivers.
        const Clustered::GridZ gz = Clustered::DeriveGridZ(pt);
        const bool clusterOn = (ps_r_clustered || ps_r_clustered_debug) && Clustered::Ready();
        ub.cluster_params[0]  = gz.sliceScale;
        ub.cluster_params[1]  = gz.sliceBias;
        ub.cluster_params[2]  = gz.nearZ;
        // 0 = off, 1 = on, 2 = on + debug heatmap (receivers branch on >0.5 / >1.5).
        ub.cluster_params[3]  = clusterOn ? (ps_r_clustered_debug ? 2.f : 1.f) : 0.f;
        ub.cluster_params2[0] = float(Clustered::kGridX);
        ub.cluster_params2[1] = float(Clustered::kGridY);
        ub.cluster_params2[2] = float(Clustered::kGridZ);
        ub.cluster_params2[3] = float(Clustered::kMaxPerCluster);
    }

    // Dynamic-light terrain/static occlusion (r_light_occ): the forward shaders
    // march the ground-height map (rain_vp) between the fragment and each light.
    // Biases are in the rain ortho's NDC-z (~rainVis uses 0.0015); tune if needed.
    ub.light_occ[0] = ps_r_light_occ ? 1.f : 0.f;
    ub.light_occ[1] = 0.001f;   // bury bias (NDC-z; ~0.35 m the light must be below the surface to count as buried)
    ub.light_occ[2] = 0.006f;   // frag-below band (~2 m): a fragment deeper than this under the surface = fully lit (basement)
    ub.light_occ[3] = 1.0f;     // strength (1 = full cut where occluded)

    // POM params (world lmap/vlit fragment parallax).
    ub.pom_params[0] = ps_r_pom ? ps_r_pom_height : 0.f;
    ub.pom_params[1] = ps_r_pom_steps;
    ub.pom_params[2] = ps_r_pom_far;
    ub.pom_params[3] = ps_r_pom ? 1.f : 0.f;
    ub.pom_params2[0] = ps_r_pom_blur;
    ub.pom_params2[1] = ps_r_pom_normal;
    ub.pom_params2[2] = ps_r_pom_shadow;
    ub.pom_params2[3] = ps_r_pom_ao;
    ub.pom_params3[0] = ps_r_pom_debug ? 1.f : 0.f;
    ub.pom_params3[1] = ps_r_ao_flat ? 1.f : 0.f;
    ub.pom_params3[2] = ps_r_pom_ceil;
    ub.pom_params3[3] = ps_r_pom_floor;
    ub.pom_params4[0] = ps_r_pom_terrain ? 1.f : 0.f;
    ub.pom_params4[1] = ps_r_terrain_normal;        // terrain detail normal-mapping strength
    ub.pom_params4[2] = ps_r_terrain_ao;            // terrain micro contact AO strength
    ub.pom_params4[3] = (float)ps_r_terrain_debug;  // terrain debug view (0..3)
    ub.pom_params5[0] = ps_r_terrain_gloss;         // terrain dry sun-gloss strength
    ub.pom_params5[1] = ps_r_glass_opacity;          // glass opacity ceiling (r_glass_opacity; frag glass branches)
    ub.pom_params5[2] = ps_r_mud_deform;             // mud footprint strength (terrain reads the deform field on soft splats)
    ub.pom_params5[3] = (float)ps_r_puddle_debug;   // puddle debug mode (0 off, 1 coverage, 2 micro/flow)
    ub.pom_params6[0] = (ps_r_water_sim && ps_r_rain_enable) ? 1.f : 0.f;  // sim puddles off when r_rain off
    ub.pom_params6[1] = ps_r_water_murk;                // volumetric absorption /m
    ub.pom_params6[2] = ps_r_water_refract;             // bottom refraction strength
    ub.pom_params6[3] = ps_r_spec_occ;                 // bent-normal spec occlusion of wet reflections
    // SSS puddles (default source). Gate off when r_rain is off so dry weather clears.
    ub.pom_params7[0] = (ps_r_puddle_sss && ps_r_rain_enable) ? 1.f : 0.f;
    ub.pom_params7[1] = ps_r_puddle_level;              // coverage (more/larger puddles)
    ub.pom_params7[2] = ps_r_mud_depth;                 // mud print POM carve depth (fraction of the height range)
    ub.pom_params7[3] = ps_r_puddle_scale;              // puddle-body size (procedural mask freq)
    // Surface Field ("smart heightmap"): metre-scale derive read in-shader (Phase
    // 2.0). Consumers (snow/fog/water) come later; for now drives r_sf_debug.
    ub.sf_params[0] = ps_r_sf ? 1.f : 0.f;              // enable (reserved for consumers)
    ub.sf_params[1] = (float)ps_r_sf_debug;             // debug view 0..5
    ub.sf_params[2] = ps_r_sf_eps;                      // finite-difference epsilon (m)
    // Snow accumulation: r_snow is the TARGET; ease the actual coverage toward it at
    // r_snow_rate/sec so snow "falls and covers" gradually (and melts when lowered).
    {
        static float s_snowAccum = 0.f;
        const float target = (ps_r_snow < 0.f) ? 0.f : (ps_r_snow > 1.f ? 1.f : ps_r_snow);
        const float dt     = (Device.fTimeDelta < 0.1f) ? Device.fTimeDelta : 0.1f;   // clamp hitches
        const float step   = ps_r_snow_rate * dt;
        const float d      = target - s_snowAccum;
        if      (d >  step) s_snowAccum += step;
        else if (d < -step) s_snowAccum -= step;
        else                s_snowAccum  = target;
        ub.sf_params[3] = s_snowAccum;                  // actual coverage (consumer: SC_SnowAmount x this)
    }

    // Snow footprint deformation: a persistent ring of recent foot contacts. Each stamp
    // fades over r_snow_deform_time seconds (TIME-based, so the trail lasts ~a minute,
    // not just the last N metres); re-stepping a print refreshes it (trampled stays
    // trampled). At 256 slots and ~4 contacts/s a full minute fits before recycle.
    // snow_displace.glsl carves these (geometry in the tese, sharp dimple in the frag).
    {
        constexpr int kMaxStamps = 256;
        struct DStamp { float x, z, r, s; };
        static DStamp s_st[kMaxStamps] = {};
        static int    s_cnt = 0, s_head = 0;
        const float snowNow = ub.sf_params[3];
        const bool  on      = ps_r_snow_deform && snowNow > 0.01f;

        // Per-frame TIME decay: every live stamp loses dt/lifetime of its strength, so a
        // print made now is gone in ~lifetime seconds (smooth fade, no pop).
        if (ps_r_snow_deform) {
            const float life = (ps_r_snow_deform_time > 0.1f) ? ps_r_snow_deform_time : 0.1f;
            const float dt   = (Device.fTimeDelta < 0.1f) ? Device.fTimeDelta : 0.1f;
            const float dec  = dt / life;
            for (int i = 0; i < s_cnt; ++i)
                if (s_st[i].s > 0.f) s_st[i].s = (s_st[i].s > dec) ? (s_st[i].s - dec) : 0.f;
        }

        if (on) {
            xr_vector<Fvector> feet;
            // Player: first-person has no body skeleton in the render list, so estimate
            // two feet from the camera XZ + facing (Y is irrelevant - the carve is XZ).
            {
                const Fvector& camp = Device.vCameraPosition;
                Fvector fwd = Device.vCameraDirection; fwd.y = 0.f;
                if (fwd.square_magnitude() > 1e-4f) fwd.normalize(); else fwd.set(0.f, 0.f, 1.f);
                const Fvector perp = { -fwd.z, 0.f, fwd.x };   // right vector (XZ plane)
                feet.push_back({ camp.x - perp.x * 0.13f, camp.y, camp.z - perp.z * 0.13f });
                feet.push_back({ camp.x + perp.x * 0.13f, camp.y, camp.z + perp.z * 0.13f });
            }
            // NPCs (and any 3rd-person body): foot bones, fallback object root.
            xr_vector<Fvector> npc; Skinned_CollectFeet(npc, 24);
            for (const Fvector& f : npc) feet.push_back(f);

            const float rad   = ps_r_snow_deform_radius;
            const float minSp = rad * 0.8f;             // don't duplicate prints closer than this
            for (const Fvector& f : feet) {
                int hit = -1;                            // existing LIVE print at this spot?
                for (int i = 0; i < s_cnt; ++i) {
                    if (s_st[i].s <= 0.f) continue;      // skip faded/dead slots
                    const float dx = s_st[i].x - f.x, dz = s_st[i].z - f.z;
                    if (dx * dx + dz * dz < minSp * minSp) { hit = i; break; }
                }
                if (hit >= 0) { s_st[hit].s = 1.f; continue; }   // refresh, don't duplicate
                s_st[s_head] = { f.x, f.z, rad, 1.f };           // new print (overwrites oldest)
                s_head = (s_head + 1) % kMaxStamps;
                if (s_cnt < kMaxStamps) ++s_cnt;
            }
        }
        ub.deform_count[0] = float(on ? s_cnt : 0);
        ub.deform_count[1] = ps_r_snow_deform_depth;          // press depth (m)
        ub.deform_count[2] = ps_r_snow_deform_depth * 0.45f;  // ridge height (m)
        ub.deform_count[3] = ps_r_snow_deform ? 1.f : 0.f;
        for (int i = 0; i < kMaxStamps; ++i) {
            ub.deform_stamps[i][0] = s_st[i].x;
            ub.deform_stamps[i][1] = s_st[i].z;
            ub.deform_stamps[i][2] = s_st[i].r;
            ub.deform_stamps[i][3] = (i < s_cnt && s_st[i].s > 0.f) ? s_st[i].s : 0.f;
        }

        // Texture path (r_snow_deform_tex): the dense persistent press field. When on
        // and ready, the shaders sample uDeform (binding 20) instead of the stamp loop.
        // Mud footprints (r_mud_deform) ride the same texture, snow or not.
        const bool texOn = ps_r_snow_deform_tex && (ps_r_snow_deform || ps_r_mud_deform > 0.f) && Deform::Ready();
        memcpy(ub.deform_vp, &Deform::GetVP(), sizeof(ub.deform_vp));
        const float dsize = float(Deform::Size() ? Deform::Size() : 1);
        // 0 = off, 1 = texture dent on the TERRAIN, 2 = MESH mode (terrain skips snow
        // geometry; the dense snow mesh owns it — vk_pass_snow).
        ub.deform_tex[0] = texOn ? (ps_r_snow_mesh ? 2.f : 1.f) : 0.f;
        ub.deform_tex[1] = 1.f / dsize;                          // 1/size (uv step)
        ub.deform_tex[2] = ps_r_snow_deform_depth;               // max dent depth (m)
        ub.deform_tex[3] = (2.f * Deform::Half()) / dsize;       // world metres per texel
    }
    // Periodic state log while debugging wetness (pairs with the mask view).
    if (ps_r_wet_debug) {
        static u32 s_wetLogCd = 0;
        if (s_wetLogCd == 0) {
            s_wetLogCd = 300;
            Msg("[VK Rain] wet state: density=%.3f wetness=%.3f darken=%.2f refl=%.2f",
                ub.rain_params[0], ub.rain_params[1], ps_r_wet_darken, ps_r_wet_refl);
        } else --s_wetLogCd;
    }

    memcpy(s_mapped + size_t(slot) * kSlotStride, &ub, sizeof(LightUBO));

    // VSM receiver bindings. As of the temporal-resolve phase, binding 14 is the
    // SCREEN-SPACE sun-shadow mask (resolved each frame in Pass_World) — receivers
    // sample it by screen UV instead of doing the atlas/page-table lookup themselves.
    // Bindings 15/16 (page table / clipmap UBO) stay in the layout but are unused by
    // receivers now (the resolve compute owns them). Mask is in GENERAL layout; the
    // placeholder = white (lit) when not yet resolved (also gated off by shadow_params.w).
    {
        const bool maskOk = VK::VSM::MaskReady();
        VkDescriptorImageInfo  atI{
            maskOk ? VK::VSM::GetMaskSampler() : s_cubeSampler,
            maskOk ? VK::VSM::GetMaskView()    : s_fallbackWhite->GetView(),
            maskOk ? VK_IMAGE_LAYOUT_GENERAL   : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        const VkBuffer ptBuf = VK::VSM::GetPageTableHandle();
        const VkBuffer ubBuf = VK::VSM::GetUBOHandle();
        VkDescriptorBufferInfo ptI{ ptBuf ? ptBuf : s_dummyBuf.GetHandle(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo vuI{ ubBuf ? ubBuf : s_dummyBuf.GetHandle(), 0, VK_WHOLE_SIZE };
        VkWriteDescriptorSet wv[3]{};
        for (u32 k = 0; k < 3; ++k) { wv[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; wv[k].dstSet = s_set[slot]; wv[k].dstBinding = 14 + k; wv[k].descriptorCount = 1; }
        wv[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; wv[0].pImageInfo  = &atI;
        wv[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;         wv[1].pBufferInfo = &ptI;
        wv[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;         wv[2].pBufferInfo = &vuI;
        vkUpdateDescriptorSets(VulkanHW.m_Device, 3, wv, 0, nullptr);
    }

    // Clustered forward SSBOs (bindings 17/18/19): swap the dummy for the real
    // vk_clustered buffers once it inited (lazy, one-shot per slot — handles are
    // stable; the light buffer binds to THIS slot's region). Fence-safe (Begin
    // waited this slot's fence). Receivers gate on cluster_params.w regardless.
    if ((ps_r_clustered || ps_r_clustered_debug) && Clustered::Ready() && !s_boundCluster[slot]) {
        VkDescriptorBufferInfo cb[3] = {
            { Clustered::GetLightsHandle(slot), Clustered::GetLightsOffset(slot), Clustered::GetLightsRange() },
            { Clustered::GetGridHandle(),    0, VK_WHOLE_SIZE },
            { Clustered::GetIndicesHandle(), 0, VK_WHOLE_SIZE },
        };
        VkWriteDescriptorSet wc[3]{};
        for (u32 k = 0; k < 3; ++k) {
            wc[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; wc[k].dstSet = s_set[slot];
            wc[k].dstBinding = 17 + k; wc[k].descriptorCount = 1;
            wc[k].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wc[k].pBufferInfo = &cb[k];
        }
        vkUpdateDescriptorSets(VulkanHW.m_Device, 3, wc, 0, nullptr);
        s_boundCluster[slot] = true;
    }

    s_current = s_set[slot];
}

void Destroy()
{
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
    if (s_fallbackCube) { s_fallbackCube->Destroy(); xr_delete(s_fallbackCube); }
    if (s_fallbackWhite) { s_fallbackWhite->Destroy(); xr_delete(s_fallbackWhite); }
    if (s_fallbackBlack) { s_fallbackBlack->Destroy(); xr_delete(s_fallbackBlack); }
    for (auto& kv : s_cookieCache) {
        if (kv.second) { kv.second->Destroy(); xr_delete(kv.second); }
    }
    s_cookieCache.clear();
    if (s_cubeSampler) { vkDestroySampler(VulkanHW.m_Device, s_cubeSampler, nullptr); s_cubeSampler = VK_NULL_HANDLE; }
    if (s_pool)      { vkDestroyDescriptorPool(VulkanHW.m_Device, s_pool, nullptr); s_pool = VK_NULL_HANDLE; }
    if (s_setLayout) { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_setLayout, nullptr); s_setLayout = VK_NULL_HANDLE; }
    s_ubo.Destroy();
    s_dummyBuf.Destroy();
    s_mapped = nullptr;
    s_current = VK_NULL_HANDLE;
    for (u32 i = 0; i < kFramesInFlight; ++i) { s_set[i] = VK_NULL_HANDLE; s_boundCube0[i] = VK_NULL_HANDLE; s_boundCube1[i] = VK_NULL_HANDLE; s_boundAO[i] = VK_NULL_HANDLE; s_boundIL[i] = VK_NULL_HANDLE; s_boundCookie[i] = VK_NULL_HANDLE; s_boundWater[i] = VK_NULL_HANDLE; s_boundFlow[i] = VK_NULL_HANDLE; s_boundGround[i] = VK_NULL_HANDLE; s_boundDeform[i] = VK_NULL_HANDLE; s_boundCluster[i] = false; }
    s_inited = false; s_failed = false;
}

}}  // namespace VK::EnvLight
