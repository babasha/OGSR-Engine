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

// Registered pass entry — iterates g_DynamicVisuals for particle visuals.
void Pass_Particles(FrameContext& ctx);

// Resources shared with the particle visual classes.
namespace ParticlePass {
    bool             Ready();
    VkPipelineLayout GetLayout();
    VkPipeline       GetPipeline(EParticleBlendMode mode);     // lazy per-mode
    // Load (or fetch from cache) the sprite texture's descriptor set, keyed by
    // texture name. Returns VK_NULL_HANDLE if the texture can't be loaded.
    VkDescriptorSet  GetTextureSet(const char* texture_name);
}

}  // namespace VK
