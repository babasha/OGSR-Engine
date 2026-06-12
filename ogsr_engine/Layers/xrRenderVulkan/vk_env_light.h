// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan — shared per-frame environment lighting UBO.
//
// One small uniform buffer (sun/hemi/ambient from CEnvDescriptorMixer, the same
// source the sky reads) made available as a descriptor set to every forward pass
// that shades with real light: the world pass binds it at set 1, the skinned pass
// at set 2. The set LAYOUT is shared (a descriptor set can bind to any set index
// whose pipeline-layout slot uses the same layout), so all passes read identical
// lighting and the env is sampled once per frame.
//
// Per in-flight slot: its own UBO region + descriptor set, written only after the
// slot's fence proved the GPU finished the previous frame-slot read (same
// discipline as the bone SSBO / sky sets). Groundwork for STEP 2 (shadows): the
// light-space matrix + shadow sampler will join this set.
#pragma once
#include "HW_Vulkan.h"
#include "vk_light.h"   // Lights::GpuLight (dynamic point/spot lights, STEP 3)

namespace VK { namespace EnvLight {

constexpr u32 kMaxGpuLights = Lights::kMaxLights;   // nearest active dynamic lights per frame

// Must match the `Lighting` UBO block in the world / skinned fragment shaders.
struct LightUBO {
    float sun_dir[4];     // xyz = travel dir (downward, .y<0); direction-to-sun = -sun_dir
    float sun_color[4];   // rgb
    float hemi_color[4];  // rgb (w = R2 correction)
    float ambient[4];     // rgb
    float sun_vp[16];     // sun light view·proj (row-major) for shadow lookup (STEP 2)
    float counts[4];      // x = active dynamic light count (as float), yzw unused
    Lights::GpuLight lights[kMaxGpuLights];   // 3 vec4 each (STEP 3)
    float spot_vp[16];    // spot (flashlight) shadow view·proj (STEP 3b)
    float shadow_params[4]; // x = spot-shadowed gpu[] index (-1 none), y = point-shadowed index, z unused
    float sun_near_vp[16];  // sun cascade 0 view·proj (25 m, R4 scheme) — appended
                            // at the END so foliage prefix declarations stay valid
    float sun_c1_vp[16];    // sun cascade 1 view·proj (60 m)
    // Distance fog (R4 combine_1.ps port): the "wet air" / haze that fades the
    // street into fog_color with distance. Appended last (prefix-safe).
    float fog_color[4];     // rgb (env), w unused
    float fog_params[4];    // x=-near*r, y=near, z=far, w=r=1/(far-near); fog = saturate(dist*w + x)
    float eye_pos[4];       // xyz = camera world pos (for dist = length(worldPos - eye)), w unused
    // Hemisphere sky ambient (R4 hmodel.h): the sky cubes are bound at set
    // bindings 6/7; this gives the per-frame cross-fade + tuning knobs.
    float sky_params[4];    // x = cube cross-fade weight, y = ambient scale, z = sample LOD (blur), w unused
    // GTAO (binding 8, vk_pass_ssao): receivers sample the half-res AO at
    // gl_FragCoord × xy and multiply their hemi+ambient terms (R4 combine_1.ps
    // parity). z = strength EXPONENT (pow(ao, z): 0 = off, 1 = raw, 2-3 deeper),
    // w = debug flag (world shaders draw the raw AO map).
    float ao_params[4];     // x = 1/screenW, y = 1/screenH, z = strength exp, w = debug
};
static_assert(sizeof(LightUBO) == 128 + 16 + 48 * kMaxGpuLights + 80 + 64 + 64 + 48 + 16 + 16,
              "LightUBO must match the GLSL Lighting block");

bool                  Init();                 // idempotent; safe to call from multiple pass inits
void                  Destroy();
VkDescriptorSetLayout GetSetLayout();          // the shared set layout (binding 0 = UBO, FRAGMENT)
void                  Update(u32 slot);        // fill slot's UBO from env; remember it as the current set
VkDescriptorSet       GetCurrentSet();         // set chosen by the last Update() this frame

}}  // namespace VK::EnvLight
