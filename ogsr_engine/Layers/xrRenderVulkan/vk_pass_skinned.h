// xrRenderVulkan - Skinned-mesh pass (STEP B sub-step 2).
//
// Draws the skinned leaves (vkSkeletonX_ST/_PM) of dynamic CKinematics visuals
// collected in g_DynamicVisuals. GPU skinning: per-skeleton bone matrices are
// written to a host-visible SSBO, the skinned pipeline (skinned.vert/frag.spv)
// blends 1-4 bones per vertex. Called inside Pass_World (shares color+depth).
#pragma once
#include "vk_core.h"
#include "vk_pass_context.h"

namespace VK
{
    void Pass_Skinned(FrameContext& ctx);
    void Skinned_Destroy();
}
