#include "stdafx.h"
#include "vk_render_queue.h"
#include "vk_pipeline_cache.h"
#include "vk_world_material.h"  // WorldMaterial set bind
#include "vk_Visual.h"
#include "vk_UIPipeline.h"   // g_VkUI_FrameCmd

#include <algorithm>

namespace VK {

RenderQueue g_RenderQueue;

u64 makeSortKey(u32 stride, u32 tcOffset, bool depthTest, const void* mat, const void* vb)
{
    u64 k = 0;
    k |= (u64(depthTest ? 1 : 0)) << 56;
    k |= (u64(stride   & 0xFF))   << 48;
    k |= (u64(tcOffset & 0xFF))   << 40;
    // 24-bit material hash (>> 4 to drop alloc-alignment bits).
    k |= ((u64(uintptr_t(mat)) >> 4) & 0xFFFFFFull) << 16;
    // 16-bit VB hash to sub-cluster within a material.
    k |= ((u64(uintptr_t(vb)) >> 4) & 0xFFFFull);
    return k;
}

void RenderQueue::Push(const DrawItem& item)
{
    m_Items.push_back(item);
}

void RenderQueue::Clear()
{
    m_Items.clear();
}

void RenderQueue::SortByKey()
{
    std::sort(m_Items.begin(), m_Items.end(),
              [](const DrawItem& a, const DrawItem& b) { return a.sortKey < b.sortKey; });
}

void RenderQueue::Flush(FrameContext& ctx)
{
    if (m_Items.empty()) return;
    if (ctx.cmd == VK_NULL_HANDLE) return;

    VkCommandBuffer cmd = ctx.cmd;

    // State tracking: each bind only fires when the next item differs from
    // what's already bound. After a sort by sortKey, runs of identical
    // (pipeline, material, VB, IB) collapse to one bind apiece.
    VkPipeline      lastPipe   = VK_NULL_HANDLE;
    VkDescriptorSet lastMatSet = VK_NULL_HANDLE;
    float           lastAref   = 999.0f;          // sentinel — first push always fires
    float           lastDetailScale = -999.0f;    // sentinel
    VkBuffer        lastVB     = VK_NULL_HANDLE;
    VkBuffer        lastIB     = VK_NULL_HANDLE;
    VkIndexType     lastIType  = VK_INDEX_TYPE_MAX_ENUM;
    Fmatrix         lastXform;                 // per-item MVP tracker (dynamic objects)
    bool            haveXform  = false;

    u32 nDraw = 0, nPipeBind = 0, nMatBind = 0, nTailPush = 0, nVBBind = 0, nIBBind = 0;

    for (const DrawItem& it : m_Items)
    {
        auto* fv = static_cast<vkFVisual*>(it.vis);
        if (!fv || !fv->m_mesh.IsValid()) continue;
        if (!fv->m_mesh.p_rm_Vertices || !fv->m_mesh.p_rm_Indices) continue;

        // Per-item MVP (push-constant offset 0). Dynamic visuals carry their own
        // world matrix in it.xform; level statics submit identity, giving
        // mvp == *viewProj (same as Pass_World's pre-loop push). Push only when
        // the xform changes — identical runs (e.g. all statics) collapse to one.
        if (ctx.viewProj && (!haveXform || 0 != memcmp(&it.xform, &lastXform, sizeof(Fmatrix)))) {
            Fmatrix mvp;
            // Fmatrix::mul(A,B) == B·A; need uploaded = world·viewProj, so A=viewProj, B=xform.
            // (Identity xform — statics — gives viewProj either way, which is why they rendered.)
            mvp.mul(*ctx.viewProj, it.xform);   // = it.xform · viewProj (model->clip)
            vkCmdPushConstants(cmd, PipelineCache::GetLayout(),
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(Fmatrix), &mvp);
            lastXform = it.xform;
            haveXform = true;
        }

        PipelineCache::Key k{};
        k.stride    = fv->m_mesh.vStride;
        k.tcOffset  = fv->m_mesh.tcOffset;
        // Sub-layout drives shader variant: tcOffset==24 → lmap (TC1+lightmap),
        // tcOffset==28 → vert-lit (D3DCOLOR + sun mask). Both share the same
        // pipeline layout and push range.
        const bool lmap = (k.tcOffset == 24);
        k.vs        = lmap ? PipelineCache::WorldLmapVS() : PipelineCache::WorldVlitVS();
        k.fs        = lmap ? PipelineCache::WorldLmapFS() : PipelineCache::WorldVlitFS();
        k.depthTest = true;

        VkPipeline pipe = PipelineCache::Get(k);
        if (pipe == VK_NULL_HANDLE) continue;

        if (pipe != lastPipe) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
            lastPipe = pipe;
            ++nPipeBind;
            // Pipeline change forces VB/IB/material rebind even if the
            // handles haven't changed — the spec treats them as dirty after
            // a new pipeline. Reset the trackers so we re-emit.
            lastVB     = VK_NULL_HANDLE;
            lastIB     = VK_NULL_HANDLE;
            lastMatSet = VK_NULL_HANDLE;
        }

        // Bind material (descriptor set 0) and patch alphaRef when either
        // the visual's material or its aref changes.
        WorldMaterial* mat = fv->m_pWorldMaterial
                              ? fv->m_pWorldMaterial
                              : WorldMaterialCache::GetDefault();
        if (mat && mat->set != VK_NULL_HANDLE && mat->set != lastMatSet) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    PipelineCache::GetLayout(), 0, 1, &mat->set, 0, nullptr);
            lastMatSet = mat->set;
            ++nMatBind;
        }
        // Per-material tail at offset 72: { float alphaRef; float detailScale }.
        // Push 8 bytes whenever either changes. stageFlags must include both
        // VS and FS — detailScale is read by the vertex shader for v_DetailUV,
        // alphaRef by the fragment shader for the discard.
        const float aref        = mat ? mat->alphaRef    : -1.0f;
        const float detailScale = mat ? mat->detailScale :  0.0f;
        if (aref != lastAref || detailScale != lastDetailScale) {
            constexpr u32 kTailOffset = sizeof(Fmatrix) + 2 * sizeof(float);
            float tail[2] = { aref, detailScale };
            vkCmdPushConstants(cmd, PipelineCache::GetLayout(),
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               kTailOffset, sizeof(tail), tail);
            lastAref        = aref;
            lastDetailScale = detailScale;
            ++nTailPush;
        }

        VkBuffer vb = fv->m_mesh.p_rm_Vertices->GetHandle();
        if (vb != lastVB) {
            VkDeviceSize vbOffset = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &vbOffset);
            lastVB = vb;
            ++nVBBind;
        }

        VkBuffer    ib     = fv->m_mesh.p_rm_Indices->GetHandle();
        VkIndexType iType  = fv->m_mesh.iType;
        if (ib != lastIB || iType != lastIType) {
            vkCmdBindIndexBuffer(cmd, ib, 0, iType);
            lastIB    = ib;
            lastIType = iType;
            ++nIBBind;
        }

        const u32 firstIndex = it.iCountOverride ? it.iBaseOverride  : fv->m_mesh.iBase;
        const u32 indexCount = it.iCountOverride ? it.iCountOverride : fv->m_mesh.iCount;
        vkCmdDrawIndexed(cmd, indexCount, 1, firstIndex,
                         (s32)fv->m_mesh.vBase, 0);
        ++nDraw;
    }

    static bool s_diag_done = false;
    if (!s_diag_done) {
        Msg("[VK Queue] Flush: items=%zu draws=%u pipeBinds=%u matBinds=%u tailPush=%u vbBinds=%u ibBinds=%u",
            m_Items.size(), nDraw, nPipeBind, nMatBind, nTailPush, nVBBind, nIBBind);
        s_diag_done = true;
    }
}

}  // namespace VK
