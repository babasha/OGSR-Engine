// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan - Per-frame draw queue.
//
// Phase 3 splits "scene assembled this DrawItem" from "GPU executed it".
// Visuals call Push() during traversal; Pass_World sorts and Flushes once.
// Flush is the only place that issues Vulkan draw/bind calls — all
// state-tracking (pipeline, VB, IB, material) lives there so we can collapse
// adjacent items that share state.

#pragma once
#include "vk_core.h"
#include "vk_pass_context.h"

class vkRender_Visual;
class CFrustum;

namespace VK {

struct DrawItem
{
    vkRender_Visual* vis     = nullptr;
    Fmatrix          xform   = {};      // world matrix (identity for level statics)
    float            lod     = 0.0f;
    u64              sortKey = 0;       // see makeSortKey() in vk_render_queue.cpp

    // Optional draw-range override. FProgressive picks a LOD-specific slice
    // of its own IB; iCountOverride!=0 tells Flush to use these instead of
    // the visual's full m_mesh.iBase/iCount. firstIndex passed to
    // vkCmdDrawIndexed becomes iBaseOverride (already includes m_mesh.iBase
    // + sw_offsets[lod]).
    u32              iBaseOverride  = 0;
    u32              iCountOverride = 0;

    // Sky-ambient occlusion for dynamic objects (1.0 = open sky / statics).
    // Stamped by Push() from the queue's current submit-hemi; Flush feeds it to
    // the world shader so indoor dynamics don't glow. See CRender::add_Visual.
    float            hemi           = 1.0f;

    // Translucent glass pane (WorldMaterial::isGlass): Push routes it to the
    // LATE glass list, drawn by Pass_WorldGlass after the whole opaque world.
    bool             lateGlass      = false;
};

class RenderQueue
{
public:
    void Push(const DrawItem& item);
    // Raw append for REPLAYING items captured from a prior Push (rain AT-caster
    // cache): skips the hemi stamp and glass routing — the stored item already
    // went through them. Items appended in pre-sorted order need no SortByKey.
    void PushPresorted(const DrawItem& item) { m_Items.push_back(item); }
    void Clear();
    void SortByKey();
    void Flush(FrameContext& ctx);

    // Late translucent (glass) list — accumulated by Push across the statics AND
    // dynamics flushes, drawn once by Pass_WorldGlass after the opaque world
    // (GPU statics/trees/grass/sky), cleared there. ClearGlass() at frame start.
    void FlushGlass(FrameContext& ctx);
    void ClearGlass();
    bool HasGlass() const { return !m_GlassItems.empty(); }
    // Post-FlushGlass view (deduped): the particles pass re-draws these panes
    // into the heat-haze distortion RT = glass refraction (r_glass_refr).
    const xr_vector<DrawItem>& GlassItems() const { return m_GlassItems; }

    // Sky-ambient occlusion stamped onto every subsequently-pushed item (until
    // changed). Set per dynamic object before its Submit; reset to 1.0 for
    // statics. Lets one shared queue carry per-object hemi without touching the
    // many Submit() overrides.
    void SetSubmitHemi(float h) { m_SubmitHemi = h; }

    // Heightmap tessellation is statics-only: dynamic visuals carry a model
    // xform, but the world VS emits MODEL-space vWorldPos, so the TES distance
    // factor (vs the camera) would be garbage for them. Pass_World flips this
    // off for the dynamic-visuals flush. Default on.
    void SetAllowTess(bool b) { m_AllowTess = b; }

    // Depth-only flush: binds PipelineCache depth pipelines (by stride) and
    // pushes mvp = item.xform · vp per item. No materials/colour — position
    // only. Used by the sun/spot shadow casters AND the camera depth prepass;
    // the prepass passes skipAlphaTested=true (a position-only shader can't
    // discard, so punch-out materials would poke opaque holes in the depth).
    // alphaTestedOnly=true skips OPAQUE casters (they're rendered by the
    // GPU-driven compute-cull + indirect path, vk_shadow_gpu) — the CPU queue
    // then draws only the alpha-tested cutout casters the GPU path can't.
    // cull (optional): per-item bounding-sphere vs frustum test — spot tiles /
    // point-cube faces collect their queue by range-SPHERE, but the actual
    // cone/face frustum is a small slice of it; this skips the rest.
    void FlushDepth(VkCommandBuffer cmd, const Fmatrix& vp, bool skipAlphaTested = false,
                    bool alphaTestedOnly = false, bool displaceTerrain = false,
                    const CFrustum* cull = nullptr);

    size_t Size() const { return m_Items.size(); }

    // Read-only view of the queued items (diagnostics / GPU-driven overlap analysis).
    const xr_vector<DrawItem>& Items() const { return m_Items; }

    // GPU-driven world split: drop items whose visual is in the GPU set
    // (inGpuSet(vis) → drawn by WorldGPU::Draw*) AND dedup the rest by visual.
    // The CPU world walk double-submits hierarchy children (parent recursion +
    // own top-level entry) → ~3.7× redundant draws; this keeps one DrawItem per
    // unique non-GPU visual. Call after SortByKey, before the prepass/color flush.
    void DedupExclude(bool (*inGpuSet)(vkRender_Visual*));

private:
    xr_vector<DrawItem> m_Items;
    xr_vector<DrawItem> m_GlassItems;   // late translucent panes (see FlushGlass)
    float               m_SubmitHemi = 1.0f;
    bool                m_AllowTess  = true;
};

extern RenderQueue g_RenderQueue;

// Encode a 64-bit sort key from the bits that change pipeline/material/VB
// binds. High bits change the rarest binds, low bits change cheapest —
// items sharing a prefix collapse into a single bind in Flush.
//
// Layout (MSB → LSB):
//   [59]      wmark — baked level decal (1 bit, sorts LAST: blends over opaque)
//   [58]      tess — heightmap-tessellated material (clusters tess pipelines)
//   [56]      depthTest                (1 bit)
//   [48..55]  stride                   (8 bits)
//   [40..47]  tcOffset                 (8 bits)
//   [16..39]  material descriptor hash (24 bits) — clusters by descriptor set
//   [0..15]   VB pointer hash          (16 bits) — sub-clusters by VB
u64 makeSortKey(u32 stride, u32 tcOffset, bool depthTest, const void* mat, const void* vb, bool wmark = false, bool tess = false);

}  // namespace VK
