// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan - Per-frame context shared between render passes.
//
// Plain-data carrier populated once in CRender::Begin() and forwarded to each
// Pass_* function. Pass code reads from this instead of poking globals
// directly, so when the framegraph lands the only thing that has to change is
// who fills the struct.

#pragma once
#include "vk_core.h"
// Fmatrix lives in xrCore/_matrix.h, pulled in transitively via stdafx in
// every TU that includes this header.

namespace VK
{
    struct FrameContext
    {
        VkCommandBuffer cmd          = VK_NULL_HANDLE;
        u32             imageIndex   = 0;
        VkExtent2D      extent       = { 0, 0 };

        // Render targets. Passes render to THESE, never to Swapchain.m_Images[...]
        // directly — so when the framegraph repoints them at an offscreen HDR /
        // G-buffer target, pass code needs no change. Filled in CRender::Begin();
        // for now colorImage/View are this frame's swapchain image + view.
        // Layout invariant: colorImage is in COLOR_ATTACHMENT_OPTIMAL for the
        // entire pass sequence (Begin transitions in once, End transitions to
        // PRESENT once); passes must NOT transition it. depthView stays in
        // DEPTH_ATTACHMENT_OPTIMAL the whole frame likewise.
        VkImage         colorImage   = VK_NULL_HANDLE;
        VkImageView     colorView    = VK_NULL_HANDLE;
        VkImageView     depthView    = VK_NULL_HANDLE;

        const Fmatrix*  viewProj     = nullptr;
        const Fmatrix*  viewProjPrev = nullptr;  // history matrix for motion vectors / TAA — unused in phase 1
        u32             frameIdx     = 0;
        float           dt           = 0.0f;
    };
}
