// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan - Minimal render-pass registry (framegraph seam).
//
// The thin seam between "hardcoded pass order in CRender::Render()" and a full
// framegraph. Passes register an ordered Execute callback; ExecutePasses runs
// them in registration order and inserts a SceneAttachmentBarrier between
// adjacent passes (the single-layout convention keeps the color/depth targets
// in their attachment layout the whole frame — see vk_pass_context.h).
//
// To add a pass (G-buffer, SSAO, bloom, lights...) you RegisterPass it instead
// of editing Render(). Targets travel in FrameContext (colorView/depthView), so
// a pass never needs to know it's drawing to the swapchain vs an offscreen RT.
//
// Deliberately minimal: Execute-only (no Setup phase yet). Setup — declaring the
// resources a pass reads/writes so barriers can be derived per-resource — lands
// with the RT registry, when there are offscreen targets to track.

#pragma once
#include "vk_core.h"
#include "vk_pass_context.h"
#include <functional>

namespace VK
{
    using PassExecuteFn = std::function<void(FrameContext&)>;

    struct RenderPass
    {
        const char*   name;       // debug label
        PassExecuteFn execute;    // records this pass's draws into ctx.cmd
    };

    // Append a pass to the ordered list (idempotent registration is the caller's
    // job — ExecutePasses runs every registered pass each frame, in order).
    void RegisterPass(const char* name, PassExecuteFn fn);

    // Run every registered pass in order, inserting SceneAttachmentBarrier
    // between adjacent passes. Passes may early-out internally (no level, etc.).
    void ExecutePasses(FrameContext& ctx);

    // Drop all registrations (device teardown).
    void ClearPasses();

    // Destroy the GPU pass-timing query pool (device teardown, before vkDestroyDevice).
    void PassTimingDestroy();
}
