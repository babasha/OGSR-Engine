// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

#include "stdafx.h"
#include "vk_framegraph.h"

namespace VK
{

FrameGraph g_FrameGraph;

void FrameGraph::BeginFrame(VkImage colorImg, VkImage depthImg)
{
    // CRender::Begin leaves the HDR color in COLOR_ATTACHMENT (after the clear's
    // TRANSFER_DST -> COLOR_ATTACHMENT barrier) and the scene depth in
    // DEPTH_ATTACHMENT (the once-per-frame UNDEFINED -> depth transition). Seed
    // both to those states — Seed records only, it emits no barrier. The tracked
    // stage/access mirror the last write Begin performed, so the first consumer's
    // Require() derives a correct src scope.
    m_color.Seed(colorImg, VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    m_depth.Seed(depthImg, VK_IMAGE_ASPECT_DEPTH_BIT,
                 VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                 VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                 VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
}

} // namespace VK
