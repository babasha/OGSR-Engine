// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan - World (level static geometry) pass.
//
// Phase 1 of the pass refactor: render-loop body lives here, pipeline
// caching/teardown stays in vk_pipeline_cache. Caller is CRender::Render().

#pragma once
#include "vk_core.h"
#include "vk_pass_context.h"

class vkRender_Visual;

namespace VK
{
    void Pass_World(FrameContext& ctx);

    // Dynamic (spawned) visuals — NPCs, weapons, items, HUD. CRender::add_Visual
    // pushes {visual, world-xform} here every frame during the object traversal;
    // Pass_World drains them (in the same render pass as the level statics) with a
    // per-item MVP; CRender::Calculate() clears the list before the traversal.
    // hemi = sky-visibility factor (0..1) ray-traced at the object's position
    // (CRender::add_Visual). Gates the hemisphere sky ambient so dynamics indoors
    // (basements, rooms) go dark instead of glowing with the full sky — they have
    // no baked lightmap occlusion of their own. 1.0 = fully open sky.
    struct DynVisual { vkRender_Visual* vis; Fmatrix xform; float hemi = 1.0f; };
    extern xr_vector<DynVisual> g_DynamicVisuals;

    // First-person HUD visuals (player hands + active weapon/item). Collected
    // separately from world dynamics because they render with the HUD projection
    // (Device.mFullTransform_hud, narrow FOV, camera at origin) and a near depth
    // range so they draw on top of the world without clipping into it. Populated
    // by CRender::add_Visual when root->renderable_HUD() is set (g_hud->Render_MAIN).
    extern xr_vector<DynVisual> g_HudVisuals;
}
