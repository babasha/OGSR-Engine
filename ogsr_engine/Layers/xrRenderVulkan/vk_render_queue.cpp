// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

#include "stdafx.h"
#include "vk_render_queue.h"
#include "vk_pipeline_cache.h"
#include "vk_world_material.h"  // WorldMaterial set bind
#include "vk_env_light.h"       // EnvLight::GetCurrentSet — set 1 (per-frame lighting)
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
    DrawItem it = item;
    it.hemi = m_SubmitHemi;   // stamp the current submit-hemi (1.0 for statics)
    m_Items.push_back(it);
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
    VkPipeline       lastPipe   = VK_NULL_HANDLE;
    VkDescriptorSet  lastMatSet = VK_NULL_HANDLE;
    float            lastAref   = 999.0f;          // sentinel — first push always fires
    float            lastDetailScale = -999.0f;    // sentinel
    float            lastHemi        = -999.0f;    // sentinel (per-object sky-ambient gate)
    VkBuffer         lastVB     = VK_NULL_HANDLE;
    VkBuffer         lastIB     = VK_NULL_HANDLE;
    VkIndexType      lastIType  = VK_INDEX_TYPE_MAX_ENUM;
    Fmatrix          lastXform;                 // per-item MVP tracker (dynamic objects)
    bool             haveXform  = false;
    // Terrain splatting uses a separate pipeline layout (7-binding set). Push
    // constants are range-compatible, but binds/pushes must target the active
    // layout — track it and force re-emits when it flips.
    VkPipelineLayout lastLayout = VK_NULL_HANDLE;

    u32 nDraw = 0, nPipeBind = 0, nMatBind = 0, nTailPush = 0, nVBBind = 0, nIBBind = 0;

    for (const DrawItem& it : m_Items)
    {
        auto* fv = static_cast<vkFVisual*>(it.vis);
        if (!fv || !fv->m_mesh.IsValid()) continue;
        if (!fv->m_mesh.p_rm_Vertices || !fv->m_mesh.p_rm_Indices) continue;

        // Resolve material + pick the render path. Terrain (diffuse under
        // "terrain\") uses a separate pipeline + 7-binding splat set + layout;
        // everything else takes the standard lmap/vlit path. Decide first so
        // all pushes/binds below target the correct (active) layout.
        WorldMaterial* mat = fv->m_pWorldMaterial
                              ? fv->m_pWorldMaterial
                              : WorldMaterialCache::GetDefault();

        VkPipeline       pipe   = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkDescriptorSet  set    = VK_NULL_HANDLE;

        const bool terrain = mat && mat->isTerrain
                          && mat->terrainSet != VK_NULL_HANDLE
                          && PipelineCache::GetTerrainPipeline() != VK_NULL_HANDLE;
        if (terrain) {
            pipe   = PipelineCache::GetTerrainPipeline();
            layout = PipelineCache::GetTerrainLayout();
            set    = mat->terrainSet;
        } else {
            PipelineCache::Key k{};
            k.stride   = fv->m_mesh.vStride;
            k.tcOffset = fv->m_mesh.tcOffset;
            // Sub-layout drives shader variant: tcOffset==24 → lmap (TC1+lightmap),
            // tcOffset==28 → vert-lit (D3DCOLOR + sun mask).
            const bool lmap = (k.tcOffset == 24);
            k.vs        = lmap ? PipelineCache::WorldLmapVS() : PipelineCache::WorldVlitVS();
            k.fs        = lmap ? PipelineCache::WorldLmapFS() : PipelineCache::WorldVlitFS();
            k.depthTest = true;
            pipe   = PipelineCache::Get(k);
            layout = PipelineCache::GetLayout();
            set    = mat ? mat->set : VK_NULL_HANDLE;
        }
        if (pipe == VK_NULL_HANDLE || layout == VK_NULL_HANDLE) continue;

        // Layout flip (terrain ↔ standard): a new layout invalidates bound
        // descriptors and may disturb push constants — force MVP + tail re-push
        // and reset pipeline/set/buffer trackers.
        if (layout != lastLayout) {
            lastLayout      = layout;
            haveXform       = false;
            lastAref        = 999.0f;
            lastDetailScale = -999.0f;
            lastPipe        = VK_NULL_HANDLE;
            lastMatSet      = VK_NULL_HANDLE;
            lastVB          = VK_NULL_HANDLE;
            lastIB          = VK_NULL_HANDLE;

            // set 1 = per-frame env lighting. The world and terrain layouts differ at
            // set 0, so a layout flip disturbs set 1 too — (re)bind it on every flip.
            VkDescriptorSet envSet = EnvLight::GetCurrentSet();
            if (envSet != VK_NULL_HANDLE)
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 1, 1, &envSet, 0, nullptr);
        }

        // Per-item MVP (push-constant offset 0). Dynamic visuals carry their own
        // world matrix in it.xform; level statics submit identity, giving
        // mvp == *viewProj. Push only when the xform changes.
        if (ctx.viewProj && (!haveXform || 0 != memcmp(&it.xform, &lastXform, sizeof(Fmatrix)))) {
            Fmatrix mvp;
            mvp.mul(*ctx.viewProj, it.xform);   // = it.xform · viewProj (model->clip)
            vkCmdPushConstants(cmd, layout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(Fmatrix), &mvp);
            lastXform = it.xform;
            haveXform = true;
        }

        if (pipe != lastPipe) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
            lastPipe = pipe;
            ++nPipeBind;
            lastVB     = VK_NULL_HANDLE;
            lastIB     = VK_NULL_HANDLE;
            lastMatSet = VK_NULL_HANDLE;
        }

        if (set != VK_NULL_HANDLE && set != lastMatSet) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    layout, 0, 1, &set, 0, nullptr);
            lastMatSet = set;
            ++nMatBind;
        }
        // Per-material/-item tail at offset 72: { alphaRef, detailScale, dynHemi }.
        // dynHemi gates the sky ambient per object (1.0 = statics/open sky).
        const float aref        = mat ? mat->alphaRef    : -1.0f;
        const float detailScale = mat ? mat->detailScale :  0.0f;
        if (aref != lastAref || detailScale != lastDetailScale || it.hemi != lastHemi) {
            constexpr u32 kTailOffset = sizeof(Fmatrix) + 2 * sizeof(float);
            float tail[3] = { aref, detailScale, it.hemi };
            vkCmdPushConstants(cmd, layout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               kTailOffset, sizeof(tail), tail);
            lastAref        = aref;
            lastDetailScale = detailScale;
            lastHemi        = it.hemi;
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

void RenderQueue::FlushDepth(VkCommandBuffer cmd, const Fmatrix& lightVP, bool skipAlphaTested)
{
    if (m_Items.empty() || cmd == VK_NULL_HANDLE) return;
    VkPipelineLayout layoutSolid = PipelineCache::GetDepthLayout();
    VkPipelineLayout layoutAT    = PipelineCache::GetDepthATLayout();
    if (layoutSolid == VK_NULL_HANDLE) return;

    // Push block of the AT variant (must match shadow_depth_at.{vert,frag}).
    struct ATPush { Fmatrix mvp; float uvScale[2]; float aref; float _pad; };

    VkPipeline      lastPipe   = VK_NULL_HANDLE;
    VkPipelineLayout lastLayout = VK_NULL_HANDLE;
    VkDescriptorSet lastMatSet = VK_NULL_HANDLE;
    float           lastAref   = -999.f;
    VkBuffer    lastVB   = VK_NULL_HANDLE;
    VkBuffer    lastIB   = VK_NULL_HANDLE;
    VkIndexType lastIType = VK_INDEX_TYPE_MAX_ENUM;
    Fmatrix     lastXform; bool haveXform = false;
    u32 nDraw = 0;

    for (const DrawItem& it : m_Items)
    {
        auto* fv = static_cast<vkFVisual*>(it.vis);
        if (!fv || !fv->m_mesh.IsValid()) continue;
        if (!fv->m_mesh.p_rm_Vertices || !fv->m_mesh.p_rm_Indices) continue;

        // Alpha-tested items go through the AT variant (punch-out silhouette in
        // the depth); without it (shader missing / caller opted out) skip them.
        WorldMaterial* mat = fv->m_pWorldMaterial ? fv->m_pWorldMaterial : WorldMaterialCache::GetDefault();
        const float aref = mat ? mat->alphaRef : -1.f;
        const bool  at   = aref >= 0.f;
        if (at && (skipAlphaTested || layoutAT == VK_NULL_HANDLE
                   || mat->set == VK_NULL_HANDLE)) continue;

        VkPipeline pipe = at ? PipelineCache::GetDepthATPipeline(fv->m_mesh.vStride, fv->m_mesh.tcOffset)
                             : PipelineCache::GetDepthPipeline(fv->m_mesh.vStride);
        if (pipe == VK_NULL_HANDLE) continue;
        VkPipelineLayout layout = at ? layoutAT : layoutSolid;

        if (layout != lastLayout) {   // layout flip disturbs pushes + sets
            lastLayout = layout;
            haveXform  = false;
            lastMatSet = VK_NULL_HANDLE;
            lastAref   = -999.f;
        }
        if (pipe != lastPipe) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
            lastPipe = pipe; lastVB = VK_NULL_HANDLE; lastIB = VK_NULL_HANDLE;
        }

        if (at) {
            if (mat->set != lastMatSet) {
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &mat->set, 0, nullptr);
                lastMatSet = mat->set;
            }
            if (!haveXform || aref != lastAref || 0 != memcmp(&it.xform, &lastXform, sizeof(Fmatrix))) {
                ATPush push{};
                push.mvp.mul(lightVP, it.xform);
                push.uvScale[0] = 1.0f / 1024.0f;   // level statics' SHORT2 quant
                push.uvScale[1] = 1.0f / 1024.0f;
                push.aref       = aref;
                vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   0, sizeof(push), &push);
                lastXform = it.xform; haveXform = true; lastAref = aref;
            }
        } else if (!haveXform || 0 != memcmp(&it.xform, &lastXform, sizeof(Fmatrix))) {
            Fmatrix lmvp; lmvp.mul(lightVP, it.xform);   // = it.xform · lightVP
            vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Fmatrix), &lmvp);
            lastXform = it.xform; haveXform = true;
        }

        VkBuffer vb = fv->m_mesh.p_rm_Vertices->GetHandle();
        if (vb != lastVB) { VkDeviceSize o = 0; vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &o); lastVB = vb; }

        VkBuffer ib = fv->m_mesh.p_rm_Indices->GetHandle();
        VkIndexType it2 = fv->m_mesh.iType;
        if (ib != lastIB || it2 != lastIType) { vkCmdBindIndexBuffer(cmd, ib, 0, it2); lastIB = ib; lastIType = it2; }

        const u32 firstIndex = it.iCountOverride ? it.iBaseOverride  : fv->m_mesh.iBase;
        const u32 indexCount = it.iCountOverride ? it.iCountOverride : fv->m_mesh.iCount;
        vkCmdDrawIndexed(cmd, indexCount, 1, firstIndex, (s32)fv->m_mesh.vBase, 0);
        ++nDraw;
    }

    static bool s_diagD = false;
    if (!s_diagD) { s_diagD = true; Msg("[VK Shadow] FlushDepth: items=%zu draws=%u", m_Items.size(), nDraw); }
}

}  // namespace VK
