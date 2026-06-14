// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan - Particle billboard render pass.
//
// Draws PAPI-simulated particle effects as camera-facing billboards over the
// already-rendered scene (depth-tested, no depth write). Registered after Sky
// so particles composite on top of the world. Owns the shared particle
// pipeline-per-blend-mode, the sprite texture descriptor cache, and a
// per-frame dynamic vertex ring.

#pragma once
#include "vk_core.h"
#include "vk_pass_context.h"

enum EParticleBlendMode : int;

namespace VK {

// Lifecycle (called from CRender::create / device teardown).
bool ParticlePass_Init();
void ParticlePass_Destroy();

// Registered pass entry — iterates g_DynamicVisuals + g_HudVisuals for
// particle visuals. Three phases: world effects (scene projection), HUD
// effects (HUD-FOV projection + near depth range — muzzle flashes), and
// PBM_DISTORT effects (rendered into the distortion buffer for the tonemap).
void Pass_Particles(FrameContext& ctx);

// Resources shared with the particle visual classes.
namespace ParticlePass {
    bool             Ready();
    VkPipelineLayout GetLayout();
    VkPipeline       GetPipeline(EParticleBlendMode mode);     // lazy per-mode
    // Load (or fetch from cache) the sprite texture's descriptor set, keyed by
    // texture name. Returns VK_NULL_HANDLE if the texture can't be loaded.
    VkDescriptorSet  GetTextureSet(const char* texture_name);

    // Heat-haze distortion buffer (PBM_DISTORT effects): full-res RGBA8, neutral
    // 0.5, written by Pass_Particles, consumed by the tonemap composite which
    // offsets the scene UVs by (rg - 0.5) * kDistortAmount. Created lazily the
    // first frame a distort effect is visible; null until then (tonemap skips).
    VkImageView      GetDistortView();
    u32              DistortGeneration();   // bumps on (re)create — rebind trigger

    // Decal pipelines (VK::Wallmarks): generic = MODULATE2X (R4
    // effects_wallmark parity); blood = alpha blend with the SSS
    // effects_wallmark_blood alpha remap (readable on dark clothing).
    VkPipeline       GetWallmarkPipeline();
    VkPipeline       GetBloodWallmarkPipeline();

    // Rain (VK::Pass_Rain): splashes = alpha blend, shape from tex.a, colour
    // from vertex tint (fx_rain rgb is a refraction normal map); drop streaks
    // = procedural shape (the fx_rain alpha reads empty — rain_drop.frag).
    VkPipeline       GetRainPipeline();
    VkPipeline       GetRainDropPipeline();
}

}  // namespace VK
