// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan — sun shadow caster pass (STEP 2). Renders static world depth
// from the sun's POV into the shadow map, then leaves it SHADER_READ for the
// world pass to sample. Registered BEFORE "World" in CRender::Render.
#pragma once
#include "vk_pass_context.h"

namespace VK {
void Pass_SunShadow(FrameContext& ctx);
void SunShadow_Destroy();   // grass-spot caster pipeline teardown (device shutdown)
}
