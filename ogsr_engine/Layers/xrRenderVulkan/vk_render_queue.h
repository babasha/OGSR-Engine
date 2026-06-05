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
};

class RenderQueue
{
public:
    void Push(const DrawItem& item);
    void Clear();
    void SortByKey();
    void Flush(FrameContext& ctx);

    size_t Size() const { return m_Items.size(); }

private:
    xr_vector<DrawItem> m_Items;
};

extern RenderQueue g_RenderQueue;

// Encode a 64-bit sort key from the bits that change pipeline/material/VB
// binds. High bits change the rarest binds, low bits change cheapest —
// items sharing a prefix collapse into a single bind in Flush.
//
// Layout (MSB → LSB):
//   [56]      depthTest                (1 bit)
//   [48..55]  stride                   (8 bits)
//   [40..47]  tcOffset                 (8 bits)
//   [16..39]  material descriptor hash (24 bits) — clusters by descriptor set
//   [0..15]   VB pointer hash          (16 bits) — sub-clusters by VB
u64 makeSortKey(u32 stride, u32 tcOffset, bool depthTest, const void* mat, const void* vb);

}  // namespace VK
