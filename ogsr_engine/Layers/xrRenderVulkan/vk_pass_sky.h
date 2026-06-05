// xrRenderVulkan - Sky pass.
//
// Phase 6: fills cleared (z = 1.0) pixels with a sky colour. Procedural
// gradient for now; cubemap path lands once CEnvironment is hooked. Runs
// after Pass_World so it can depth-test against the world's z buffer.

#pragma once
#include "vk_core.h"
#include "vk_pass_context.h"

namespace VK {

namespace SkyPass {
    bool Init();
    void Destroy();
}

void Pass_Sky(FrameContext& ctx);

}  // namespace VK
