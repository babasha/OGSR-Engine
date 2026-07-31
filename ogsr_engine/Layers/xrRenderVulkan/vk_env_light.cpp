// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan — shared per-frame environment lighting UBO. See vk_env_light.h.
#include "stdafx.h"
#include "vk_env_light.h"
#include "vk_color_space.h"                 // ColorSpace::Linearize* — sRGB→linear on upload (r_linear_color)
#include "vk_buffer.h"                     // CVulkanBuffer
#include "vk_command_buffer.h"             // CVulkanCommandManager::FRAMES_IN_FLIGHT
#include "vk_shadow.h"                     // ShadowMap (binding 1 = shadow map, sun_vp)
#include "vk_pass_shadow.h"                // SpotShadow_TileOfLight — spot-pool tile per light
#include "vk_vsm.h"                        // VSM receivers (bindings 14-16: atlas, page table, clipmap UBO)
#include "vk_clustered.h"                  // Clustered forward (bindings 17-19: lights, grid, indices)
#include "vk_profiler.h"                    // VK::Prof::NameSet — TEMP VUID-hunt instrumentation
#include "vk_water_sim.h"                  // WaterSim (binding 11 = water depth)
#include "vk_deform.h"                     // Deform (binding 20 = snow deform press field)
#include "vk_pass_skinned.h"               // Skinned_CollectFeet (snow footprint deformation)
#include "vk_texture.h"                    // CVulkanTexture (fallback ambient cube)
#include "vk_texture_stream.h"             // TextureStreamer feedback SSBO (binding 30)
#include "vk_pass_sky.h"                   // SkyPass::AcquireAmbientCubes (hemisphere sky ambient)
#include "vk_ibl.h"                        // Sky specular IBL (binding 26 = prefiltered cube)
#include "vk_volumetrics.h"                // integrated froxel volume (binding 27 = sun-beam ground deposit)
#include "vk_terrain_cache.h"              // terrain composite cache (bindings 28/29 + tcache UBO params)
#include "vk_pass_ssao.h"                  // GTAO result (binding 8, white fallback until ready)
#include "vk_pipeline_cache.h"             // PipelineCache::SetFrameSpecMask (Inc 1 world uber-FS variant bits)

extern int ps_r_txstream;                  // texture streaming master toggle — gates the WS_FEEDBACK cadence bit
#include "../../xr_3da/IGame_Persistent.h" // g_pGamePersistent->Environment()
#include "../../xr_3da/IGame_Level.h"      // g_pGameLevel->name() — per-level terrain channel offsets
#include "../../xr_3da/Environment.h"      // CEnvDescriptorMixer (sun_dir/sun_color/hemi/ambient)
#include "../../xr_3da/device.h"           // Device.vCameraPosition (light collection)
#include "vk_pass_context.h"               // FrameContext — scene render extent (ao_params, DLSS mip bias)

#include <cstring>
#include <cmath>    // log2f — DLSS texture mip bias

extern VK::FrameContext g_FrameCtx;   // CRender_Vulkan.cpp — scene render / display extents

extern int   ps_r_ssao_debug;     // r_ssao_debug — draw the raw AO map (vk_console_min.cpp)
extern float ps_r_ssao_strength;  // r_ssao_strength — live AO depth knob
extern float ps_r_dlss_bias;      // r_dlss_bias — scale of the DLSS mip-LOD bias (1 = full NVIDIA log2(render/display), 0 = off)
extern float ps_r_sun_boost;      // r_sun_boost — global sun multiplier (was ×1.25 literals in 5 shaders)
extern float ps_r_sun_beam;       // r_sun_beam — surface beam-gap recovery strength (0 = off)
extern float ps_r_sun_beam_dist;  // r_sun_beam_dist — max recovery distance (m)
extern float ps_r_sun_beam_boost; // r_sun_beam_boost — extra sun in the recovered gap (splash)
extern float ps_r_sun_beam_bias;  // r_sun_beam_bias — atlas self-bias (m along the sun ray)
extern float ps_r_sun_beam_ground;     // r_sun_beam_ground — forward sun-beam GROUND deposit strength (0 = off)
extern float ps_r_sun_beam_ground_thr; // r_sun_beam_ground_thr — in-scatter luminance threshold for the deposit
extern int   ps_r_vol;                 // r_vol — volumetric fog master (the deposit needs the volume)
extern float ps_r_ambient_floor;  // r_ambient_floor — flat ambient lift (was +0.05 literals in 4 shaders)
extern float ps_r_ambient_sky_gate; // r_ambient_sky_gate — gate the flat sky ambient by sky visibility (no indoor leak)
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
extern int   ps_r_shade_debug;    // r_shade_debug — lighting-component isolation views (world shaders)
extern float ps_r_fog_dist;       // r_fog_dist — forward distance-fog range scale (0 = off)
extern float ps_r_pom_ceil;       // r_pom_ceil — POM strength on down-facing surfaces (ceilings)
extern float ps_r_pom_floor;      // r_pom_floor — POM strength on up-facing surfaces (floors)
extern int   ps_r_pom_terrain;    // r_pom_terrain — terrain POM enable (experimental, default off)
extern float ps_r_pom_zoff;       // r_pom_zoff — terrain POM depth offset strength (SSFX, 1 = 0.11 m)
extern float ps_r_terra_blend;    // r_terra_blend — detail-blend depth (0 = GAMMA plain-mask cross-fade)
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
extern float ps_r_puddle_geo;     // r_puddle_geo — puddles follow REAL ground dips (Surface Field concavity)
extern float ps_r_wet_dist;       // r_wet_dist — range (m) wet shading survives to
extern float ps_r_terrain_detail_dist; // r_terrain_detail_dist — range (m) terrain detail normal/AO/gloss survive to
extern float ps_r_bolt_flash;     // r_bolt_flash — lightning lifts the hemisphere light instead of faking a sun
extern float ps_r_bump;           // r_bump — static material normal-map strength
extern int   ps_r_bump_debug;     // r_bump_debug — 1 world normal, 2 gloss
extern float ps_r_gloss_scale;    // r_gloss_scale — material gloss -> IBL roughness
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
extern int   ps_r_spot_grass;         // r_spot_grass — grass casters into the spot beam map
extern float ps_r_spot_grass_shadow;  // r_spot_grass_shadow — grass shadow strength on SURFACES (0..1)
extern float ps_r_flashlight_grass;   // r_flashlight_grass — FULL grass shadow for flashlight tiles (night wow)
extern int   ps_r_point_debug;        // r_point_debug — point-shadow coverage overlay
extern int   ps_r_grass_debug;        // r_grass_debug — grass lighting-component isolation view
extern float ps_r_grass_self_bias;    // r_grass_self_bias — grass dyn-atlas anti-acne slack (m)
extern int   ps_r_ibl;                // r_ibl — sky specular IBL master (prefiltered sky reflections + sun glint)
extern float ps_r_ibl_spec;           // r_ibl_spec — specular IBL strength
extern int   ps_r_ibl_debug;          // r_ibl_debug — show only the specular field
extern int   ps_r_sky_sh_debug;       // r_sky_sh_debug — dump SH coefficients + the sun/ambient magnitudes
extern int   ps_r_sky_proc;           // r_sky_proc — procedural Rayleigh+Mie sky
extern float ps_r_sky_intensity;      // r_sky_intensity
extern float ps_r_sky_turbidity;      // r_sky_turbidity
extern float ps_r_sky_mie_g;          // r_sky_mie_g
extern int   ps_r_sky_sun_from_atmo;  // r_sky_sun_from_atmo — derive sun_color from the model
extern float ps_r_sky_sun_scale;      // r_sky_sun_scale — sun irradiance scale
extern int   ps_r_sky_sh;             // r_sky_sh — diffuse sky irradiance via SH9 (0 = probe-mip fallback)
extern float ps_r_sky_sh_ground;      // r_sky_sh_ground — below-horizon (ground bounce) weight in the projection

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

    // Sky specular IBL (binding 26): grey cube fallback until vk_ibl prefilters the
    // real sky; Update() swaps in IBL::GetSpecView() once ready (gated by ibl_params.x).
    VkImageView           s_boundIBL[kFramesInFlight] = {};
    VkBuffer              s_boundSH[kFramesInFlight] = {};   // sky SH9 SSBO (binding 31), swapped in lazily
    VkImageView           s_boundVol3D[kFramesInFlight] = {};   // binding 27: integrated froxel volume (sun-beam ground deposit)
    VkImageView           s_boundTCacheH[kFramesInFlight] = {}; // binding 28: terrain composite cache height
    VkImageView           s_boundTCacheW[kFramesInFlight] = {}; // binding 29: terrain composite cache weights

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
            // Spot cookie — Colour: it is a projected light-colour pattern that
            // multiplies emitted radiance, so leaving it gamma-encoded would make the
            // lamp's colour the one gamma-space term in an otherwise linear light path.
            if (!t->LoadDDS(full, /*applyBCSwizzle*/ false, TexStreamClass::UI,
                            TexColorSpace::Color)) { xr_delete(t); t = nullptr; }
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

    // ── Sunlight colour from atmospheric extinction (r_sky_sun_from_atmo) ───────
    // CPU mirror of SunTransmittance() in atmosphere.glsl — same constants, same
    // march, so the directional sun agrees with the sky the model draws.
    //
    // This exists because the weather config CANNOT be trusted at the hours that
    // matter most. Measured at a late sunset with the sun still up (elev +1.4°):
    // sun_color = (0.009, 0.004, 0.002) — effectively black. X-Ray zeroes the sun
    // near the horizon because the classic renderer had no twilight model, so dusk
    // arrived with no directional light at all and the world went flat grey.
    // Physics does know the answer: a low sun's beam crosses ~40x more air, Rayleigh
    // strips its blue, and what survives is red and still bright enough to light a
    // landscape. Deriving it here restores exactly that.
    Fvector3 AtmoSunTransmittance(const Fvector3& toSun, float turbidity)
    {
        constexpr float kRp = 6371000.f, kRa = 6471000.f;
        constexpr float kBetaR[3] = { 5.5e-6f, 13.0e-6f, 22.4e-6f };
        constexpr float kBetaM = 21e-6f, kHR = 8000.f, kHM = 1200.f;

        Fvector3 out; out.set(0.f, 0.f, 0.f);
        const Fvector3 ro{ 0.f, kRp + 1.f, 0.f };

        auto raySphere = [&](float r, float& t0, float& t1) -> bool {
            const float b = ro.x * toSun.x + ro.y * toSun.y + ro.z * toSun.z;
            const float c = ro.x * ro.x + ro.y * ro.y + ro.z * ro.z - r * r;
            float d = b * b - c;
            if (d < 0.f) return false;
            d = _sqrt(d); t0 = -b - d; t1 = -b + d;
            return true;
        };

        float a0, a1;
        if (!raySphere(kRa, a0, a1) || a1 <= 0.f) return out;
        // Below the geometric horizon the planet blocks the beam outright.
        float g0, g1;
        if (raySphere(kRp, g0, g1) && g1 > 0.f && g0 > 0.f) return out;

        const int   kSteps = 16;
        const float step = a1 / float(kSteps);
        float t = step * 0.5f, odR = 0.f, odM = 0.f;
        for (int i = 0; i < kSteps; ++i) {
            const float qx = ro.x + toSun.x * t, qy = ro.y + toSun.y * t, qz = ro.z + toSun.z * t;
            const float h = _max(_sqrt(qx * qx + qy * qy + qz * qz) - kRp, 0.f);
            odR += expf(-h / kHR) * step;
            odM += expf(-h / kHM) * step;
            t   += step;
        }
        const float bm = kBetaM * _max(turbidity, 0.f) * 1.1f;
        out.x = expf(-(kBetaR[0] * odR + bm * odM));
        out.y = expf(-(kBetaR[1] * odR + bm * odM));
        out.z = expf(-(kBetaR[2] * odR + bm * odM));
        return out;
    }
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
    VkDescriptorSetLayoutBinding b[32]{};
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
    // Spot BEAM map (binding 22): spot map + grass casters. spotShadowF blends it
    // with the clean spot map so grass shadows surfaces PARTIALLY (r_spot_grass_shadow).
    b[22].binding = 22; b[22].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; b[22].descriptorCount = 1;
    b[22].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    // VSM STATIC atlas (binding 23): GRASS samples it directly at the blade's own
    // world position (detail.frag VSM_GRASS_DIRECT) — the screen-space mask (14)
    // belongs to the surface BEHIND the blade and paints terrain shadows onto the
    // canopy. Pairs with the page table (15) + clipmap UBO (16) already bound.
    b[23].binding = 23; b[23].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; b[23].descriptorCount = 1;
    b[23].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    // + the DYNAMIC atlas (24) and its page table (25): grass-on-grass / NPC-on-grass
    // shadows, sampled with an extra bias so a casting blade doesn't acne on itself.
    b[24].binding = 24; b[24].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; b[24].descriptorCount = 1;
    b[24].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    b[25].binding = 25; b[25].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;         b[25].descriptorCount = 1;
    b[25].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    // Sky specular IBL (binding 26): prefiltered sky cube (roughness mips), sampled
    // by iblSpecular()/sunSpec() in the forward receivers. FRAGMENT.
    b[26].binding = 26; b[26].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; b[26].descriptorCount = 1;
    b[26].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    // Integrated froxel volume (binding 27, sampler3D): the forward receivers probe the
    // shaft in-scatter at their own pixel to deposit a sun-beam "ground splash" (see
    // world_terrain.frag). Written per-slot in Update (Vol::Init runs AFTER EnvLight::Init,
    // so the 3D view is null here; the first Update fills it before any forward draw).
    b[27].binding = 27; b[27].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; b[27].descriptorCount = 1;
    b[27].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    // Terrain COMPOSITE CACHE (bindings 28/29, vk_terrain_cache): baked composite
    // height + blend weights, marched by world_terrain(+_depth).frag when live.
    // Written per-slot in Update (the cache bakes after EnvLight::Init) — white
    // fallback until then.
    b[28].binding = 28; b[28].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; b[28].descriptorCount = 1;
    b[28].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    b[29].binding = 29; b[29].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; b[29].descriptorCount = 1;
    b[29].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    // Texture-streaming GPU feedback (binding 30, vk_texture_stream): the world FS
    // atomicMin the encoded desired LOD of the base diffuse it sampled. Bound to the
    // streamer's feedback SSBO (or the dummy when creation failed — shaders skip on
    // out-of-range streamID pushes, so the dummy is never written).
    b[30].binding = 30; b[30].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; b[30].descriptorCount = 1;
    b[30].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    // Diffuse sky SH9 (binding 31, vk_ibl): 9 irradiance coefficients projected from
    // the world-space sky probe. Read by skyAmbient() in every forward receiver
    // (world / terrain / grass / trees / skinned). Bound to the dummy until the
    // first projection lands; sh_params.x gates the read.
    b[31].binding = 31; b[31].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; b[31].descriptorCount = 1;
    b[31].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    // The snow MESH (vk_pass_snow) vertex shader samples the RAIN map (9, base height)
    // + deform field (20) to place + displace its dense grid -> need VERTEX visibility.
    b[9].stageFlags  |= VK_SHADER_STAGE_VERTEX_BIT;
    b[13].stageFlags |= VK_SHADER_STAGE_VERTEX_BIT;
    VkDescriptorSetLayoutCreateInfo slci{};
    slci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    slci.bindingCount = 32; slci.pBindings = b;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &slci, nullptr, &s_setLayout) != VK_SUCCESS) {
        Msg("![VK EnvLight] set layout create failed"); s_failed = true; return false;
    }

    VkDescriptorPoolSize ps[3]{
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         kFramesInFlight * 2 },    // LightUBO + VSM clipmap UBO
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kFramesInFlight * 24 },   // 13 shadow/sky/ao + VSM mask + deform + SSIL + spot beam + VSM static/dyn atlases (grass) + IBL spec cube + froxel volume
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         kFramesInFlight * 7 },    // VSM static+dyn page tables + 3 cluster SSBOs + tex-stream feedback + sky SH9
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
    for (u32 i = 0; i < kFramesInFlight; ++i) VK::Prof::NameSet(s_set[i], "EnvLight.WorldSet1");   // TEMP diag: VUID hunt

    // Host-visible UBO, one aligned LightUBO region per in-flight slot; each set
    // bound to its slice at an alignment-safe offset.
    s_ubo.Create(kSlotStride * kFramesInFlight,
                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
    s_mapped = static_cast<u8*>(s_ubo.Map());
    if (!s_mapped) { Msg("![VK EnvLight] UBO map failed"); s_failed = true; return false; }

    // Tiny valid SSBO/UBO bound to VSM bindings 15/16 until VSM initialises (lazy on
    // first r_vsm). Receivers gate VSM sampling on shadow_params.w, so it's never read.
    s_dummyBuf.Create(256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);

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

        // Binding 22: spot BEAM map (spot + grass casters) — partial grass
        // shadowing on surfaces (spotShadowF blend, r_spot_grass_shadow).
        VkDescriptorImageInfo beamI{ ShadowMap::GetSampler(), ShadowMap::GetSpotBeamView(),
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };

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
        // Binding 26: sky specular IBL cube — grey cube fallback until vk_ibl has a
        // probe; Update() swaps in IBL::GetSpecView() (gated by ibl_params.x anyway).
        VkDescriptorImageInfo iblI{ s_cubeSampler, fbView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        s_boundIBL[i] = fbView;

        // Clustered forward (bindings 17/18/19): valid SSBO placeholder until the
        // vk_clustered module inits; Update() swaps in the real buffers (gated by
        // cluster_params.w so the dummy is never actually read).
        VkDescriptorBufferInfo clI[3]{
            { s_dummyBuf.GetHandle(), 0, VK_WHOLE_SIZE },
            { s_dummyBuf.GetHandle(), 0, VK_WHOLE_SIZE },
            { s_dummyBuf.GetHandle(), 0, VK_WHOLE_SIZE },
        };

        VkWriteDescriptorSet w[32]{};
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
        if (beamI.imageView != VK_NULL_HANDLE) {
            w[count].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[count].dstSet = s_set[i]; w[count].dstBinding = 22; w[count].descriptorCount = 1;
            w[count].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[count].pImageInfo = &beamI;
            ++count;
        }
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
        w[count].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[count].dstSet = s_set[i]; w[count].dstBinding = 26; w[count].descriptorCount = 1;
        w[count].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[count].pImageInfo = &iblI;
        ++count;
        // VSM receiver bindings 14/15/16 + 23/24/25 — fallback (white image / dummy
        // buffer) so they are DEFINED from Init, before the first EnvLight::Update
        // writes the real VSM data. Without this, a draw that binds this set before the
        // first Update reads an unwritten binding 14 (uVsmMask) → VUID-08114. Update()
        // overwrites all six each frame.
        VkDescriptorImageInfo  vsmFbImg{ s_cubeSampler, fbWhite, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorBufferInfo vsmFbBuf{ s_dummyBuf.GetHandle(), 0, VK_WHOLE_SIZE };
        const struct { u32 binding; VkDescriptorType type; bool img; } vsmFb[6] = {
            { 14, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, true  },
            { 15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         false },
            { 16, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         false },
            { 23, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, true  },
            { 24, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, true  },
            { 25, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         false },
        };
        for (const auto& f : vsmFb) {
            w[count].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[count].dstSet = s_set[i]; w[count].dstBinding = f.binding; w[count].descriptorCount = 1;
            w[count].descriptorType = f.type;
            if (f.img) w[count].pImageInfo = &vsmFbImg; else w[count].pBufferInfo = &vsmFbBuf;
            ++count;
        }
        // Terrain composite cache (28/29) — white fallback until the first bake;
        // Update() swaps in TerrainCache views (gated by tcache_params.x anyway).
        VkDescriptorImageInfo tcI{ s_cubeSampler, fbWhite, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        s_boundTCacheH[i] = fbWhite; s_boundTCacheW[i] = fbWhite;
        for (u32 tb = 28; tb <= 29; ++tb) {
            w[count].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[count].dstSet = s_set[i]; w[count].dstBinding = tb; w[count].descriptorCount = 1;
            w[count].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[count].pImageInfo = &tcI;
            ++count;
        }
        // Texture-streaming GPU feedback SSBO (binding 30). The streamer creates it
        // on this first call; on failure the dummy keeps the binding defined (world
        // FS skips the write for streamID >= kFeedbackSlots, and no real slots are
        // ever handed out when the buffer doesn't exist).
        VkBuffer fbStream = VK::TextureStreamer::Instance().GetFeedbackBuffer();
        VkDescriptorBufferInfo fbStreamI{ fbStream != VK_NULL_HANDLE ? fbStream : s_dummyBuf.GetHandle(), 0, VK_WHOLE_SIZE };
        w[count].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[count].dstSet = s_set[i]; w[count].dstBinding = 30; w[count].descriptorCount = 1;
        w[count].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[count].pBufferInfo = &fbStreamI;
        ++count;
        // Sky SH9 (binding 31): dummy for now — IBL::Init() runs AFTER this block, so
        // the real buffer does not exist yet. Update() swaps it in lazily per slot
        // (same pattern as the IBL cube at binding 26). The binding must be defined
        // here regardless: an unwritten descriptor is undefined behaviour even for a
        // shader that never reads it.
        VkDescriptorBufferInfo shI{ s_dummyBuf.GetHandle(), 0, VK_WHOLE_SIZE };
        w[count].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[count].dstSet = s_set[i]; w[count].dstBinding = 31; w[count].descriptorCount = 1;
        w[count].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[count].pBufferInfo = &shI;
        ++count;
        vkUpdateDescriptorSets(VulkanHW.m_Device, count, w, 0, nullptr);
    }

    // Sky specular IBL prefilter module (owns the RGBA16F reflection cube at
    // binding 26). Self-guards if the compute .spv is missing → binding stays grey.
    IBL::Init();

    s_current = s_set[0];
    Msg("[VK EnvLight] init OK (UBO %u bytes x %u slots)", (u32)sizeof(LightUBO), kFramesInFlight);
    return true;
}

// ---- Per-level terrain channel depth offsets (SSFX ssfx_terrain_offset) ----
// gamedata\config\terrain_details.ltx, [<level name>] offsets = R,G,B,A —
// per-channel height shifts (asphalt sinks below soil etc.), same table as
// GAMMA's ssfx_parallax_setup. Missing file/section = all-zero (SSFX default
// for unlisted levels). Cached per level; re-read on level change.
static float      s_chOff[4]   = { 0.f, 0.f, 0.f, 0.f };
static shared_str s_chOffLevel;
static bool       s_chOffInit  = false;

const float* TerrainChOff()
{
    shared_str lvl = g_pGameLevel ? g_pGameLevel->name() : shared_str("");
    if (s_chOffInit && lvl.equal(s_chOffLevel)) return s_chOff;
    s_chOffInit  = true;
    s_chOffLevel = lvl;
    s_chOff[0] = s_chOff[1] = s_chOff[2] = s_chOff[3] = 0.f;
    if (lvl.size()) {
        string_path fn;
        if (FS.exist(fn, "$game_config$", "terrain_details.ltx")) {
            CInifile ini(fn, TRUE);
            if (ini.section_exist(lvl) && ini.line_exist(lvl, "offsets")) {
                const Fvector4 v = ini.r_fvector4(lvl, "offsets");
                s_chOff[0] = v.x; s_chOff[1] = v.y; s_chOff[2] = v.z; s_chOff[3] = v.w;
            }
        }
        Msg("[VK Terrain] level '%s' channel offsets = (%.3f, %.3f, %.3f, %.3f)",
            lvl.c_str(), s_chOff[0], s_chOff[1], s_chOff[2], s_chOff[3]);
    }
    return s_chOff;
}

// ---- Visual sun direction: LATCHED for the duration of a thunderbolt ----------
// CEffect_Thunderbolt overwrites CurrentEnv->sun_dir with the strike direction while
// a bolt flashes (thunderbolt.cpp) — a 2000s trick that buys dramatic light for free.
// The cost is that every consumer which draws something AT the sun draws a SECOND SUN
// at the bolt's azimuth: the sky disc, the volumetric sun beam with god rays fanning
// out of it. A clap fires several bolts, each at its own azimuth, and the volume's
// temporal history (r_vol_ta, blend 0.92) keeps the previous beams alive for dozens of
// frames — so you end up looking at two or three suns, each shooting light. Real
// lightning is a huge, very distant AREA light: it lifts the whole sky and puts no
// disc in it. So the direction is held to the last real sun here, and the flash is
// spent as sky/ambient light in Update() instead.
const Fvector& SunDirVisual()
{
    static Fvector s_dir{0.f, -1.f, 0.f};
    if (g_pGamePersistent) {
        auto& env = g_pGamePersistent->Environment();
        if (const auto* E = env.CurrentEnv)
            if (!env.IsThunderboltActive())
                s_dir = E->sun_dir;
    }
    return s_dir;
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
            const Fvector& sunD = SunDirVisual();   // held steady through a bolt (no second sun)
            ub.sun_dir[0]=sunD.x;  ub.sun_dir[1]=sunD.y;  ub.sun_dir[2]=sunD.z;
            ub.sun_color[0]=E->sun_color.x; ub.sun_color[1]=E->sun_color.y; ub.sun_color[2]=E->sun_color.z;
            ub.hemi_color[0]=E->hemi_color.x; ub.hemi_color[1]=E->hemi_color.y; ub.hemi_color[2]=E->hemi_color.z; ub.hemi_color[3]=E->hemi_color.w;
            ub.ambient[0]=E->ambient.x; ub.ambient[1]=E->ambient.y; ub.ambient[2]=E->ambient.z;
            // LIGHTNING AS SKY LIGHT (r_bolt_flash). Holding the sun direction above
            // would otherwise cost the flash entirely: the engine spends a bolt through
            // sun_color, and with the sun back at its real place (below the horizon at
            // night) that boost lands on nothing. A strike is an enormous distant area
            // light, so it belongs in the HEMISPHERE term — the whole world brightens
            // for an instant, no disc, no rays. fog_color/sky_color stay boosted by the
            // engine, so the sky and the distance haze flash on their own.
            const Fvector& flash = g_pGamePersistent->Environment().ThunderboltFlash();
            if (ps_r_bolt_flash > 0.f && (flash.x + flash.y + flash.z) > 0.001f) {
                ub.hemi_color[0] += flash.x * ps_r_bolt_flash;
                ub.hemi_color[1] += flash.y * ps_r_bolt_flash;
                ub.hemi_color[2] += flash.z * ps_r_bolt_flash;
            }

            // Distance fog (R4 cl_fog_params/cl_fog_color binders + combine_1.ps):
            // fog_near/far are derived in CEnvDescriptorMixer (Environment_misc.cpp:
            // near = (1-density)*0.85*dist, far = 0.99*dist); fog_color is the env
            // colour the sky/horizon use, so the haze tints with time-of-day.
            // r_fog_dist scales that distance (0 = OFF). This is the THIRD haze in the
            // frame — froxel fog + Rayleigh/Mie atmosphere are the other two — and it
            // was the only one with no control at all: purely distance-based, so it
            // has no height profile and cannot be anchored to the ground. That is why
            // it reads as "fog that follows me" and why no r_vol_*/r_atmo knob ever
            // moved it. Needed as an A/B before judging the froxel layer, and as the
            // seam where the three systems eventually get unified.
            const float fogScale = ps_r_fog_dist;
            const float fn = E->fog_near * fogScale, ff = E->fog_far * fogScale;
            const float r  = (fogScale > 0.f && ff > fn + 1e-3f) ? 1.f / (ff - fn) : 0.f;
            if (r > 0.f) {
                ub.fog_params[0] = -fn * r; ub.fog_params[1] = fn; ub.fog_params[2] = ff; ub.fog_params[3] = r;
            }   // else: keep the "no fog" defaults set above (0 / 1e6 / 1e6 / 0)
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
    // LINEARISE BEFORE THE KNOBS (r_linear_color; no-op in the gamma pipeline).
    // Placed here so it covers BOTH the real env values above and the neutral
    // fallbacks — one conversion point instead of two that can drift apart.
    //
    // Order matters and this order is the physical one: r_sun_boost is a radiance
    // SCALE and r_ambient_floor a radiance OFFSET, so they must act on linear values.
    // Applying them before the decode would put them on the wrong side of a pow(2.4)
    // and quietly change what the knobs mean — a boost of 1.25 in gamma space is not
    // a boost of 1.25 in light.
    //
    // .w riders survive untouched by construction: hemi_color[3] is the "R2
    // correction" factor and ambient[3] the sky gate, neither of which is colour.
    ColorSpace::LinearizeRGB(ub.sun_color);
    ColorSpace::LinearizeRGB(ub.hemi_color);
    ColorSpace::LinearizeRGB(ub.ambient);
    ColorSpace::LinearizeRGB(ub.fog_color);

    // Sun colour from the atmosphere model (r_sky_sun_from_atmo). Applied AFTER the
    // linearisation block on purpose: transmittance is a physical, already-linear
    // quantity, so pushing it through an sRGB decode would be a second, bogus
    // conversion. The boost below still applies — it stays a user knob.
    if (ps_r_sky_proc && ps_r_sky_sun_from_atmo) {
        Fvector3 toSun; toSun.set(-ub.sun_dir[0], -ub.sun_dir[1], -ub.sun_dir[2]);
        const float len = toSun.magnitude();
        if (len > 1e-4f) {
            toSun.div(len);
            const Fvector3 tr = AtmoSunTransmittance(toSun, ps_r_sky_turbidity);
            ub.sun_color[0] = tr.x * ps_r_sky_sun_scale;
            ub.sun_color[1] = tr.y * ps_r_sky_sun_scale;
            ub.sun_color[2] = tr.z * ps_r_sky_sun_scale;
        }
    }

    // LDR-era brightness hacks, centralized: the sun boost (was a ×1.25 literal
    // in five shaders) and the ambient floor (was +0.05 in four) apply ONCE
    // here, so every receiver consumes FINAL values. Live knobs r_sun_boost /
    // r_ambient_floor; 1.0/0.0 = raw env values. The grass/tree sun pushes get
    // the same boost in their managers (they bypass this UBO); sunshafts
    // kDensity is retuned ÷1.25 to keep shaft brightness unchanged.
    //
    // r_ambient_floor is DECODED, r_sun_boost is NOT — and the asymmetry is the point.
    // A multiplier is unitless: ×1.25 means the same thing in either space, so boosting
    // linear radiance directly is right. An ADDITIVE floor is not: the 0.05 was dialled
    // in by eye against the screen, so it is a display-space quantity like every other
    // authored colour in this pipeline. Adding it raw to linear radiance and then
    // encoding for output turns it into 0.05^(1/2.2) ≈ 0.26 on screen — five times the
    // intended lift, and precisely in the black end, which is why it read as "night is
    // too bright" while daylight looked fine (there it drowns in the sun).
    //
    // Decoding it keeps the knob's number meaning exactly what it has always meant to
    // the person turning it, in BOTH pipelines. Same rule as the env colours above:
    // authored by eye ⇒ sRGB ⇒ decode on the way in.
    const float ambFloor = ColorSpace::Active() ? ColorSpace::SrgbToLinear(ps_r_ambient_floor)
                                                : ps_r_ambient_floor;
    for (int c = 0; c < 3; ++c) {
        ub.sun_color[c] *= ps_r_sun_boost;
        ub.ambient[c]   += ambFloor;
    }
    // Sky-visibility gate for the flat ambient (env_common skyAmbientGate): the
    // sky-coloured fill leaks indoors, so houses/basements read as sky-lit. .w = how
    // much covered surfaces (rainVis) lose it. 0 = old ungated look. Surfaces only.
    ub.ambient[3] = _min(_max(ps_r_ambient_sky_gate, 0.f), 1.f);

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
    // Spot shadow POOL: per-light tile assignment (+1, 0 = none) and the
    // per-tile view·proj matrices (Pass_SunShadow rendered them earlier).
    u32 flashTileMask = 0;   // bit t set → pooled tile t's owner is a handheld torch
    for (u32 i = 0; i < kMaxGpuLights; ++i) {
        const int tile = (i < FL.count) ? SpotShadow_TileOfLight(FL.src[i]) : -1;
        ub.spot_assign[i >> 2][i & 3] = float(tile + 1);
        if (tile >= 0 && i < FL.count && FL.flashFlag[i]) flashTileMask |= (1u << u32(tile));
    }
    for (u32 t = 0; t < Lights::kMaxShadowSpots; ++t)
        memcpy(ub.spot_pool_vp[t], &ShadowMap::GetSpotTileVP(t), sizeof(ub.spot_pool_vp[t]));
    // Point shadow POOL: per-light cube-array index (+1, packed 4/vec4).
    for (u32 i = 0; i < kMaxGpuLights; ++i) {
        const int cube = (i < FL.count) ? PointShadow_CubeOfLight(FL.src[i]) : -1;
        ub.point_assign[i >> 2][i & 3] = float(cube + 1);
    }
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
    // The editor used to be pinned to the grey fallback here, because the Sky pass was
    // skipped entirely in -vk_editor and its cubes were assumed to be left unusable. The
    // cost was severe and easy to misread as "the editor has no shaders": skyAmbient()
    // samples this cube ALONG THE SURFACE NORMAL and is the largest term in the lighting
    // sum, so a 1x1 uniform cube hands every surface of every object the same value —
    // flat, no matter how it is oriented. The Sky pass now runs as soon as the host
    // pushes a scene, which loads and transitions the real weather cubes, so take them.
    // The weather's sky spin. It is part of the probe's identity (the probe is built
    // in world space, so a rotated dome is a different probe) and it is also what the
    // raw-cube last-resort path needs to undo. Same source the sky draw reads.
    // ...and the weather tint the dome draw applies. Both come from the same
    // descriptor the sky pass reads, so probe and dome cannot drift apart.
    float skyRot = 0.f;
    float skyTint[3] = { 1.f, 1.f, 1.f };
    if (g_pGamePersistent)
        if (auto* mixEnv = g_pGamePersistent->Environment().CurrentEnv) {
            skyRot     = mixEnv->sky_rotation;
            skyTint[0] = mixEnv->sky_color.x;
            skyTint[1] = mixEnv->sky_color.y;
            skyTint[2] = mixEnv->sky_color.z;
        }
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
            // Sky specular IBL: refresh the prefiltered reflection cube from the SAME
            // weather cubes (no-op unless they/the cross-fade/the spin changed;
            // fence-waited immediate submit). The prefilter also drives the DIFFUSE
            // path now — it unwraps the sky into world space and the SH9 projection
            // rides along in the same submit — so it must run whenever either
            // consumer is on, not just for r_ibl.
            if (ps_r_ibl || ps_r_sky_sh) {
                IBL::SkyDesc sd;
                sd.weight = w; sd.rotation = skyRot; sd.groundBounce = ps_r_sky_sh_ground;
                sd.tint[0] = skyTint[0]; sd.tint[1] = skyTint[1]; sd.tint[2] = skyTint[2];
                // Direction TO the sun, normalised (env sun_dir travels downward).
                Fvector3 ts; ts.set(-ub.sun_dir[0], -ub.sun_dir[1], -ub.sun_dir[2]);
                const float tl = ts.magnitude();
                if (tl > 1e-4f) ts.div(tl); else ts.set(0.f, 1.f, 0.f);
                sd.sunDir[0] = ts.x; sd.sunDir[1] = ts.y; sd.sunDir[2] = ts.z;
                sd.proc      = (ps_r_sky_proc != 0);
                sd.intensity = ps_r_sky_intensity;
                sd.turbidity = ps_r_sky_turbidity;
                sd.mieG      = ps_r_sky_mie_g;
                IBL::Update(v0, v1, s_cubeSampler, sd);
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

    // Sky specular IBL (binding 26): swap the grey fallback for the prefiltered
    // reflection cube once vk_ibl has content (lazy, per slot). ibl_params gates
    // the receivers regardless, so the grey fallback is never actually reflected.
    {
        const bool iblReady = ps_r_ibl && IBL::Ready();
        VkImageView iv = iblReady ? IBL::GetSpecView()
                                  : (s_fallbackCube ? s_fallbackCube->GetView() : VK_NULL_HANDLE);
        VkSampler   is = iblReady ? IBL::GetSampler() : s_cubeSampler;
        if (iv != VK_NULL_HANDLE && iv != s_boundIBL[slot]) {
            VkDescriptorImageInfo ii{ is, iv, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = s_set[slot]; w.dstBinding = 26; w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w.pImageInfo = &ii;
            vkUpdateDescriptorSets(VulkanHW.m_Device, 1, &w, 0, nullptr);
            s_boundIBL[slot] = iv;
        }
        // Fade-in (~1 s) when the probe first becomes ready → hides the reflection
        // "pop" the user saw when the first prefilter completes at load. x = enable×fade.
        static float s_iblFade = 0.f;
        const float dtF = (Device.fTimeDelta < 0.1f) ? Device.fTimeDelta : 0.1f;
        if (iblReady) { s_iblFade += dtF; if (s_iblFade > 1.f) s_iblFade = 1.f; }
        else            s_iblFade = 0.f;
        ub.ibl_params[0] = s_iblFade;                         // enable × fade-in (0..1)
        ub.ibl_params[1] = ps_r_ibl_spec;                    // spec strength
        ub.ibl_params[2] = float(IBL::GetMaxMip());           // max roughness mip
        ub.ibl_params[3] = (iblReady && ps_r_ibl_debug) ? 1.f : 0.f;  // debug field view
    }

    // Diffuse sky irradiance, SH9 (binding 31 + sh_params). Swap the dummy for the
    // real coefficient buffer once vk_ibl has projected it (lazy, per slot).
    {
        const bool shReady = ps_r_sky_sh && IBL::SHReady();
        VkBuffer   shb     = shReady ? IBL::GetSHBuffer() : VK_NULL_HANDLE;
        if (shb == VK_NULL_HANDLE) shb = s_dummyBuf.GetHandle();
        if (shb != VK_NULL_HANDLE && shb != s_boundSH[slot]) {
            VkDescriptorBufferInfo bi{ shb, 0, VK_WHOLE_SIZE };
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = s_set[slot]; w.dstBinding = 31; w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w.pBufferInfo = &bi;
            vkUpdateDescriptorSets(VulkanHW.m_Device, 1, &w, 0, nullptr);
            s_boundSH[slot] = shb;
        }
        // .x doubles as the strength knob AND the gate: 0 makes every receiver fall
        // back to the prefiltered probe's top mip, which is the A/B for r_sky_sh.
        // Its own ~1 s fade, NOT ibl_params.x: that one is gated on r_ibl, so sharing
        // it would silently pin the diffuse to the fallback whenever specular IBL is
        // switched off — exactly the configuration someone testing r_sky_sh alone
        // would be in. The fade itself hides the ambient stepping when the first
        // projection lands mid-load.
        const bool shBound = shReady && (s_boundSH[slot] == IBL::GetSHBuffer());
        static float s_shFade = 0.f;
        const float dtS = (Device.fTimeDelta < 0.1f) ? Device.fTimeDelta : 0.1f;
        if (shBound) { s_shFade += dtS; if (s_shFade > 1.f) s_shFade = 1.f; }
        else           s_shFade = 0.f;
        ub.sh_params[0] = s_shFade;
        ub.sh_params[1] = skyRot;                        // raw-cube fallback needs to undo the spin
        ub.sh_params[2] = float(IBL::GetMaxMip());       // fallback diffuse LOD = roughest probe mip
        ub.sh_params[3] = 0.f;

        // Companion to the [VK SH] dump: the SH azimuth is only meaningful next to
        // the sun's, and the DIRECTIONAL sky term is only meaningful next to the FLAT
        // one it is added to (L.ambient). If flat >> directional, terrain reads as
        // uniformly sky-lit no matter how good the SH is — the fix would then be the
        // balance, not the irradiance. Throttled to ~2 s.
        if (ps_r_sky_sh_debug) {
            static float s_dbgT = 0.f;
            s_dbgT += (Device.fTimeDelta < 0.1f) ? Device.fTimeDelta : 0.1f;
            if (s_dbgT > 2.f) {
                s_dbgT = 0.f;
                const float sunAz = atan2f(-ub.sun_dir[2], -ub.sun_dir[0]) * 57.2957795f;   // to-sun = -sun_dir
                Msg("[VK SH] sun: to-sun=(%.2f,%.2f,%.2f) azimuth=%.1f deg elev=%.1f deg | "
                    "sun_color lum=%.4f | FLAT ambient=(%.4f,%.4f,%.4f) lum=%.4f | sky scale=%.2f | sh_fade=%.2f",
                    -ub.sun_dir[0], -ub.sun_dir[1], -ub.sun_dir[2], sunAz,
                    asinf(_max(_min(-ub.sun_dir[1], 1.f), -1.f)) * 57.2957795f,
                    0.2126f * ub.sun_color[0] + 0.7152f * ub.sun_color[1] + 0.0722f * ub.sun_color[2],
                    ub.ambient[0], ub.ambient[1], ub.ambient[2],
                    0.2126f * ub.ambient[0] + 0.7152f * ub.ambient[1] + 0.0722f * ub.ambient[2],
                    kAmbientScale, ub.sh_params[0]);
            }
        }
    }

    // [PARKED 2026-07-06 — r_sun_beam_ground default 0, so beam2.z stays 0 and the forward
    //  deposit is skipped; binding 27 is still written (cheap, one descriptor) but unused.
    //  Kept as scaffolding. See the PARKED note in vk_console_min.cpp.]
    // Sun-beam GROUND DEPOSIT (r_sun_beam_ground, binding 27): bind the integrated
    // froxel volume so the forward receivers can probe the shaft in-scatter at their
    // own pixel and deposit sun where a visible beam lands (world_terrain.frag). The
    // 3D view is eager (created at Vol::Init) and stable; write it per slot when it
    // first appears / changes. Layout stays SHADER_READ (Vol::Execute leaves it so
    // before the forward pass — same as the tonemap read).
    {
        VkImageView vv = Vol::GetIntegratedView();
        if (vv != VK_NULL_HANDLE && vv != s_boundVol3D[slot]) {
            VkDescriptorImageInfo ii{ Vol::GetSampler(), vv, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = s_set[slot]; w.dstBinding = 27; w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w.pImageInfo = &ii;
            vkUpdateDescriptorSets(VulkanHW.m_Device, 1, &w, 0, nullptr);
            s_boundVol3D[slot] = vv;
        }
        // Terrain composite cache (28/29): swap in the baked views once they exist.
        VkImageView th = TerrainCache::HeightView(), tw = TerrainCache::WeightsView();
        if (th != VK_NULL_HANDLE && tw != VK_NULL_HANDLE
            && (th != s_boundTCacheH[slot] || tw != s_boundTCacheW[slot])) {
            VkDescriptorImageInfo hi{ TerrainCache::Sampler(), th, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkDescriptorImageInfo wi{ TerrainCache::Sampler(), tw, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkWriteDescriptorSet tws[2]{};
            tws[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            tws[0].dstSet = s_set[slot]; tws[0].dstBinding = 28; tws[0].descriptorCount = 1;
            tws[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; tws[0].pImageInfo = &hi;
            tws[1] = tws[0]; tws[1].dstBinding = 29; tws[1].pImageInfo = &wi;
            vkUpdateDescriptorSets(VulkanHW.m_Device, 2, tws, 0, nullptr);
            s_boundTCacheH[slot] = th; s_boundTCacheW[slot] = tw;
        }
        // tcache UBO params: duv->cacheUV transform + live flag. Live only once
        // baked AND the views are bound this slot (first frames keep the white
        // fallback -> flag 0 keeps the shaders on the per-channel path).
        const bool tcLive = TerrainCache::Live() && s_boundTCacheH[slot] == th && th != VK_NULL_HANDLE;
        float txf[4] = { 0,0,0,0 }, tof[2] = { 0,0 };
        if (tcLive) TerrainCache::GetXform(txf, tof);
        ub.tcache_xform[0] = txf[0]; ub.tcache_xform[1] = txf[1];
        ub.tcache_xform[2] = txf[2]; ub.tcache_xform[3] = txf[3];
        ub.tcache_params[0] = tcLive ? 1.f : 0.f;
        // .y = march features of the CURRENT bake (toggling the cvars forces a
        // rebake, so this tracks): 1 = cone-step, 2 = cone + sun horizon.
        ub.tcache_params[1] = (tcLive && TerrainCache::ConeLive())
                                  ? (TerrainCache::HorizonLive() ? 2.f : 1.f) : 0.f;
        ub.tcache_params[2] = tof[0]; ub.tcache_params[3] = tof[1];
        // beam2 = { froxel nearZ, log2(far/near), deposit strength, in-scatter threshold }.
        // Strength 0 unless r_vol is on AND the volume is ready (else the froxel data is
        // stale/garbage). Same exp-Z terms the tonemap composite uses (single source).
        const Vol::GridZParams gz = Vol::GetGridZ();
        const bool volLive = (ps_r_vol != 0) && Vol::Ready();
        ub.beam2[0] = gz.nearZ;
        ub.beam2[1] = gz.logFarNear;
        ub.beam2[2] = volLive ? ps_r_sun_beam_ground : 0.f;
        ub.beam2[3] = ps_r_sun_beam_ground_thr;
    }

    // Sun-beam ground recovery (r_sun_beam): let surfaces re-open the thin sun gaps
    // the temporal screen mask smears shut, so the volumetric shaft and the ground
    // agree on where the sun lands. Only meaningful under VSM (crisp atlas source).
    ub.beam_params[0] = ps_r_sun_beam;        // recovery strength (0 = off)
    ub.beam_params[1] = ps_r_sun_beam_dist;   // max distance (m)
    ub.beam_params[2] = ps_r_sun_beam_boost;  // extra-sun kick in the recovered gap
    ub.beam_params[3] = ps_r_sun_beam_bias;   // atlas self-bias (m)

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
    // 1/screen must be the SCENE RENDER resolution, not the window: gl_FragCoord in
    // the scene passes runs over the render extent, and every screen-space consumer
    // (uVsmMask, uAO, uIL, cluster lookup) maps fragcoord*ao_params.xy → [0,1]. With
    // DLSS render<display upscaling those differ — Device.dwWidth here displaced all
    // shadows/AO ("тени слетают со своих мест"). Fall back to the window before the
    // first CRender::Begin has published the frame extent.
    const u32 sceneW = g_FrameCtx.extent.width  ? g_FrameCtx.extent.width  : Device.dwWidth;
    const u32 sceneH = g_FrameCtx.extent.height ? g_FrameCtx.extent.height : Device.dwHeight;
    ub.ao_params[0] = sceneW ? 1.f / float(sceneW) : 0.f;
    ub.ao_params[1] = sceneH ? 1.f / float(sceneH) : 0.f;
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
    ub.pom_params3[0] = float(ps_r_pom_debug);   // 1 = AO x shadow mask, 2 = self-shadow only
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
    // 2 = SSFX-STRICT shading (see wetness.glsl): passes the MODE through, not a
    // flag, so the shader can tell "our tuned water body" from "the SSFX gloss patch".
    ub.pom_params7[0] = (ps_r_puddle_sss && ps_r_rain_enable) ? (float)ps_r_puddle_sss : 0.f;
    ub.pom_params7[1] = ps_r_puddle_level;              // coverage (more/larger puddles)
    ub.pom_params7[2] = ps_r_mud_depth;                 // mud print POM carve depth (fraction of the height range)
    ub.pom_params7[3] = ps_r_puddle_scale;              // puddle-body size (procedural mask freq)
    // Puddles in real ground dips. Deliberately NOT gated by r_sf: that cvar is a
    // master switch for future Surface Field consumers, and hanging this on it would
    // make the whole feature a silent no-op for anyone with r_sf 0. The curvature
    // only needs the RAIN map, which is rendered whenever it rains or dries — i.e.
    // exactly when puddles exist — so no extra pass and no extra dependency.
    ub.puddle_geo[0] = ps_r_rain_enable ? clampr(ps_r_puddle_geo, 0.f, 1.f) : 0.f;
    ub.puddle_geo[1] = ps_r_wet_dist;   // how far wet shading survives (m) — SSFX fades at 250..200
    ub.puddle_geo[2] = ps_r_terrain_detail_dist;   // how far the terrain detail normal/AO/gloss survive (m)
    ub.bump_params[0] = ps_r_bump;         // static material normal-map strength
    ub.bump_params[1] = (float)ps_r_bump_debug;
    ub.bump_params[2] = ps_r_gloss_scale;  // material gloss -> IBL roughness
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
    // Grass shadow strength on surfaces (spotShadowF blends clean vs beam map).
    // Zero when grass casters are off — the beam map then equals the clean map
    // anyway, but skipping the 9 extra taps is free.
    ub.spot_params[0] = ps_r_spot_grass ? _min(_max(ps_r_spot_grass_shadow, 0.f), 1.f) : 0.f;
    ub.spot_params[1] = float(ps_r_point_debug);   // point-shadow debug overlay
    ub.spot_params[2] = float(ps_r_grass_debug);   // grass component isolation (detail.frag)
    ub.spot_params[3] = ps_r_grass_self_bias;      // grass dyn-atlas anti-acne slack (m)
    // Flashlight tiles paint crisp grass dapples in their ground pool (night wow) —
    // full-strength grass shadow instead of the subtle spot_params.x lamp blend.
    ub.spot_flash[0] = float(flashTileMask);
    ub.spot_flash[1] = ps_r_spot_grass ? _min(_max(ps_r_flashlight_grass, 0.f), 1.f) : 0.f;
    // Texture mip-LOD bias when DLSS renders below display res: log2(render/display)
    // (negative). Without it material textures pick the coarser mip for the low-res
    // raster and the DLSS output stays soft — NVIDIA requires this bias for SR.
    // Material frag shaders add it to their implicit-LOD albedo/detail fetches.
    {
        float texBias = 0.f;
        if (g_FrameCtx.displayExtent.width > g_FrameCtx.extent.width && g_FrameCtx.extent.width > 0)
            texBias = log2f(float(g_FrameCtx.extent.width) / float(g_FrameCtx.displayExtent.width));
        // r_dlss_bias scales the NVIDIA-recommended bias (1 = full). The full bias
        // SHARPENS the alpha mips of alpha-tested foliage → leaf coverage shrinks at
        // the cutoff → black holes onto the dark crown interior at distance (the
        // «чёрные пятна на листве» saga). 0 = off for A/B.
        ub.spot_flash[2] = texBias * _min(_max(ps_r_dlss_bias, 0.f), 1.f);
    }
    ub.spot_flash[3] = 0.f;
    // Terrain DEPTH OFFSET (r_pom_zoff, SSFX port): sink terrain depth into the POM
    // cracks (prepass + color) so GTAO / VSM resolve shade INTO the relief. Only
    // meaningful with terrain POM data; the pipeline variant gates on the same pair.
    ub.zoff_params[0] = ps_r_pom_terrain ? _max(ps_r_pom_zoff, 0.f) : 0.f;
    // .y = r_terra_blend: terrain detail-blend transition depth. 0 = plain mask
    // cross-fade (GAMMA/SSFX ships with height-blending commented out — asphalt
    // fades smoothly into soil); >0 = Mishkinis height blend (0.25 = old sharp).
    ub.zoff_params[1] = ps_r_terra_blend;
    // .z = r_shade_debug: lighting-component isolation views in the world shaders
    // (1 albedo, 2 baked lmap/vertex, 3 hemi-occ, 4 GTAO, 5 ambient sky gate,
    //  6 sky hemisphere, 7 total lighting, 8 flat ambient, 9 sun, 10 wetness).
    ub.zoff_params[2] = float(ps_r_shade_debug);
    ub.zoff_params[3] = 0.f;
    // Per-level SSFX terrain channel offsets (terrain_details.ltx; also consumed
    // by the terrain composite cache bake, which pulls TerrainChOff() directly).
    memcpy(ub.ch_off, TerrainChOff(), sizeof(ub.ch_off));
    // Baked terrain splat mask (mask-less maps): world XZ -> mask UV affine;
    // z == 0 (inactive) keeps the shaders on the material's own mask at vUV.
    TerrainMask::GetParams(ub.tmask_params);
    // Periodic state log while debugging wetness (pairs with the mask view).
    if (ps_r_wet_debug) {
        static u32 s_wetLogCd = 0;
        if (s_wetLogCd == 0) {
            s_wetLogCd = 300;
            Msg("[VK Rain] wet state: density=%.3f wetness=%.3f darken=%.2f refl=%.2f",
                ub.rain_params[0], ub.rain_params[1], ps_r_wet_darken, ps_r_wet_refl);
        } else --s_wetLogCd;
    }

    // Inc 1: fold the frame-global world uber-FS variant bits (WS_FRAME) from THIS
    // frame's UBO so the world + terrain pipelines bake matching spec constants — a
    // plain summer/dry/no-debug frame → all off → the lean variant. Reads the same ub
    // fields the shader's snow/wet/ibl/debug branches test, so the mask can't disagree.
    {
        u8 sm = 0;
        if (ub.sf_params[3]   > 0.f)    sm |= VK::PipelineCache::WS_SNOW;   // eased snow coverage
        if (ub.rain_params[1] > 0.f)    sm |= VK::PipelineCache::WS_WET;    // wetness (already rain-gated)
        if (ub.ibl_params[0]  > 0.004f) sm |= VK::PipelineCache::WS_IBL;    // r_ibl enable×fade (matches shader gate)
        // ⚠ EVERY debug view in the world shaders sits behind SPEC_DEBUG, so a new one
        // is DEAD CODE until its UBO field is listed HERE — the branch compiles out and
        // the cvar looks broken rather than unimplemented. Add the field with the view.
        if (ub.ao_params[3]   > 0.5f || ub.rain_params[2]   < 0.f  || ub.pom_params5[3]   > 0.5f ||
            ub.pom_params3[0] > 0.5f || ub.pom_params3[1]   > 0.5f || ub.cluster_params[3] > 1.5f ||
            ub.sf_params[1]   > 0.5f || ub.pom_params4[3]   > 0.5f || ub.zoff_params[2]   > 0.5f ||
            ub.ibl_params[3]  > 0.5f || ub.bump_params[1]   > 0.5f)   sm |= VK::PipelineCache::WS_DEBUG;
        // Texture-streaming feedback master gate: off ⇒ the txfbReport atomic is
        // DCE'd out of the world FS. The atomic's occluded-fragment cost (it is
        // an FS side effect, which forbids automatic early-Z) is solved by the
        // EARLY_ZTEST shader twin on the no-z-write statics pipelines — see
        // EarlyTwin in vk_pipeline_cache.cpp — so feedback runs EVERY frame.
        if (ps_r_txstream) sm |= VK::PipelineCache::WS_FEEDBACK;
        VK::PipelineCache::SetFrameSpecMask(sm);
    }

    memcpy(s_mapped + size_t(slot) * kSlotStride, &ub, sizeof(LightUBO));

    // VSM receiver bindings. Binding 14 = the SCREEN-SPACE sun-shadow mask (resolved
    // each frame in Pass_World) — SURFACE receivers (in the prepass) sample it by
    // screen UV. Bindings 15/16 (static page table / clipmap UBO) + 23/24/25 (static
    // atlas / dyn atlas / dyn page table) serve the GRASS receiver, which does the
    // full atlas lookup at the blade's own world position instead (the mask belongs
    // to the surface BEHIND a blade — see vsm_sample.glsl VSM_GRASS_DIRECT). Mask is
    // in GENERAL layout; placeholders = white (lit) until resolved/rendered (also
    // gated off by shadow_params.w).
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
        // Bindings 23/24/25: the STATIC + DYNAMIC atlases and the dyn page table —
        // grass receivers sample them at the blade's own world pos (no screen-space
        // parallax; dyn taps use an extra bias so a casting blade doesn't acne on
        // itself). White (= lit) until the atlases have content; receivers
        // additionally gate on shadow_params.w.
        const bool atlasOk = VK::VSM::AtlasReady();
        VkDescriptorImageInfo  asI{
            atlasOk ? VK::VSM::GetSampler()    : s_cubeSampler,
            atlasOk ? VK::VSM::GetAtlasView()  : s_fallbackWhite->GetView(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo  adI{
            atlasOk ? VK::VSM::GetSampler()       : s_cubeSampler,
            atlasOk ? VK::VSM::GetDynAtlasView()  : s_fallbackWhite->GetView(),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        const VkBuffer pdBuf = VK::VSM::GetDynPageTableHandle();
        VkDescriptorBufferInfo pdI{ pdBuf ? pdBuf : s_dummyBuf.GetHandle(), 0, VK_WHOLE_SIZE };
        VkWriteDescriptorSet wv[6]{};
        for (u32 k = 0; k < 6; ++k) { wv[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; wv[k].dstSet = s_set[slot]; wv[k].dstBinding = 14 + k; wv[k].descriptorCount = 1; }
        wv[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; wv[0].pImageInfo  = &atI;
        wv[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;         wv[1].pBufferInfo = &ptI;
        wv[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;         wv[2].pBufferInfo = &vuI;
        wv[3].dstBinding = 23;
        wv[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; wv[3].pImageInfo  = &asI;
        wv[4].dstBinding = 24;
        wv[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; wv[4].pImageInfo  = &adI;
        wv[5].dstBinding = 25;
        wv[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;         wv[5].pBufferInfo = &pdI;
        vkUpdateDescriptorSets(VulkanHW.m_Device, 6, wv, 0, nullptr);
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
    IBL::Destroy();   // EnvLight owns the IBL module's lifecycle (Init'd in Init())
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
    for (u32 i = 0; i < kFramesInFlight; ++i) { s_set[i] = VK_NULL_HANDLE; s_boundCube0[i] = VK_NULL_HANDLE; s_boundCube1[i] = VK_NULL_HANDLE; s_boundAO[i] = VK_NULL_HANDLE; s_boundIL[i] = VK_NULL_HANDLE; s_boundCookie[i] = VK_NULL_HANDLE; s_boundWater[i] = VK_NULL_HANDLE; s_boundFlow[i] = VK_NULL_HANDLE; s_boundGround[i] = VK_NULL_HANDLE; s_boundDeform[i] = VK_NULL_HANDLE; s_boundCluster[i] = false; s_boundIBL[i] = VK_NULL_HANDLE; s_boundSH[i] = VK_NULL_HANDLE; }
    s_inited = false; s_failed = false;
}

}}  // namespace VK::EnvLight
