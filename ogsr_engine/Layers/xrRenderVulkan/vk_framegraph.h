// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan - FrameGraph: the render graph's frame-wide execute core (v1).
//
// A single place that tracks the current (layout / stage / access) of the scene's
// persistent attachments — the HDR color target and the scene depth — via the
// ImageState primitive (vk_barriers). Any pass Requires the state it needs and a
// barrier is emitted ONLY on a real change; a run of consumers that all need the
// same state coalesces to one transition. Seeded each frame in CRender::Begin at
// the layouts Begin leaves color/depth in.
//
// WHY it exists: the renderer's barriers were hand-placed per pass, and the depth
// target thrashed ATTACHMENT<->SHADER_READ four times per frame (SSAO, VRS, VSM
// mark, VSM resolve each round-tripping it) — serializing the GPU. Routing that
// window through Depth() coalesces it to a single round-trip (r_fg_coalesce).
//
// v1 SCOPE — deliberately minimal, NOT over-built:
//   * Tracks color + depth. The depth-read window (Pass_World) is routed through
//     Depth(); the coalescing is the measured win.
//   * It is NOT a scheduler: no topological sort, no transient-memory aliasing, no
//     async-queue assignment. The inter-pass SceneAttachmentBarrier stays the
//     conservative default until passes declare per-resource read/write sets.
// EXTENSION SURFACE (documented, not built): async compute adds a queue-family to
//   ImageState + cross-queue release/acquire barriers here; aliasing adds a
//   transient-resource table; precise inter-pass barriers replace SceneAttachment-
//   Barrier once passes declare their resource usage. See the framegraph memory.

#pragma once
#include "vk_barriers.h"   // VK::ImageState

namespace VK
{
    class FrameGraph
    {
    public:
        // Seed color + depth at the layouts CRender::Begin leaves them in. Emits no
        // barrier — it only records the known frame-start state. Called every frame.
        void BeginFrame(VkImage colorImg, VkImage depthImg);

        // The tracked scene attachments. Passes route their transitions through
        // these so redundant barriers coalesce.
        ImageState& Color() { return m_color; }
        ImageState& Depth() { return m_depth; }

    private:
        ImageState m_color;
        ImageState m_depth;
    };

    extern FrameGraph g_FrameGraph;
}
