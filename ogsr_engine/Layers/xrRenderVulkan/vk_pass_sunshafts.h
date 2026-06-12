// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan — volumetric sun shafts (god rays).
//
// Fullscreen pass after Sky: raymarches the view ray against scene depth,
// sampling the sun shadow map per step — air that sees the sun glows with the
// sun colour, occluded air doesn't. Additively blended over the scene. The
// scene depth flips DEPTH_ATTACHMENT → SHADER_READ for the draw and back, so
// later passes (Particles) still depth-test as usual.
#pragma once
#include "vk_core.h"
#include "vk_pass_context.h"

namespace VK
{
    void Pass_SunShafts(FrameContext& ctx);
    void SunShafts_Destroy();
}
