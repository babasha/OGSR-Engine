// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

#include "stdafx.h"
#include "vk_pass_world.h"
#include "vk_pipeline_cache.h"
#include "vk_render_queue.h"   // g_RenderQueue (phase 3)
#include "vk_swapchain.h"      // Swapchain.m_Images / m_ImageViews
#include "CRender_Vulkan.h"    // RImplementation, vkRender_Visual ref
#include "vk_Visual.h"         // vkRender_Visual::Submit
#include "vk_pass_skinned.h"   // Pass_Skinned (skinned dynamic leaves)
#include "vk_env_light.h"      // EnvLight::Update — per-frame sun/hemi/ambient UBO (set 1)
#include "vk_command_buffer.h" // CommandManager.GetCurrentFrame() — in-flight slot
#include "vk_barriers.h"       // SceneAttachmentBarrier — prepass → color ordering
#include "vk_pass_ssao.h"      // GTAO from the prepass depth (before the color pass)
#include "vk_TreeManager.h"    // trees join the prepass depth (GTAO occluders + early-Z)

namespace VK {

// Dynamic (spawned) visuals collected by CRender::add_Visual this frame, drained
// below after the level statics. Declared in vk_pass_world.h.
xr_vector<DynVisual> g_DynamicVisuals;
xr_vector<DynVisual> g_HudVisuals;

// Layout is managed centrally now (single-layout convention): the swapchain
// image is in COLOR_ATTACHMENT_OPTIMAL for the whole frame and depth in
// DEPTH_ATTACHMENT_OPTIMAL — see CRender::Begin/End + ExecutePasses. This pass
// only records draws; it never transitions the targets.

// Push constants must match world.{vert,frag}.glsl PushConstants block.
// alphaRef (FS) + detailScale (VS) are the per-material tail at offset 72
// (8 bytes), patched together by RenderQueue::Flush when a new material
// binds. Pushing both at once requires the range stageFlags to include
// both VS and FS — the layout already does (PipelineCache::Init).
struct WorldPush
{
    Fmatrix mvp;
    float   uvScale[2];
    float   alphaRef;       // FS — discard threshold (<0 disables)
    float   detailScale;    // VS — UV scale for detail-texture sampling (.thm detail_scale)
};
static_assert(sizeof(WorldPush) == 80, "WorldPush must match GLSL PushConstants block (80 bytes)");

void Pass_World(FrameContext& ctx)
{
    if (ctx.cmd == VK_NULL_HANDLE)         return;
    if (!RImplementation.b_loaded)         return;        // No level loaded yet
    if (RImplementation.Visuals.empty())   return;
    if (PipelineCache::GetLayout()    == VK_NULL_HANDLE) return;
    if (PipelineCache::WorldLmapVS() == VK_NULL_HANDLE) return;
    if (PipelineCache::WorldVlitVS() == VK_NULL_HANDLE) return;

    VkCommandBuffer cmd = ctx.cmd;

    // (The shared depth pipelines are baked for D32 — if the driver fell back to
    // another depth format, skip the prepass rather than mismatch formats.)
    const bool prepass = PipelineCache::GetDepthLayout() != VK_NULL_HANDLE
                      && Swapchain.m_DepthFormat == VK_FORMAT_D32_SFLOAT;

    // SSAO targets must exist BEFORE EnvLight::Update binds the AO view at
    // binding 8 — a resize re-creates them here, not mid-frame after the bind.
    // Only when the prepass will run: without it Execute never transitions the
    // fresh image out of UNDEFINED, and binding it would be invalid to sample.
    if (prepass)
        SSAOPass::EnsureTargets(ctx.extent);

    // Refresh this frame's env-lighting UBO (sun/hemi/ambient) once, before any
    // draw. RenderQueue::Flush binds the resulting set at set 1; Pass_Skinned
    // reuses the same set (at set 2). Fence-guarded slot → no in-flight write hazard.
    EnvLight::Update(CommandManager.GetCurrentFrame());

    // Collect + sort the static queue up front — the depth prepass and the
    // color pass below both consume it.
    Fmatrix identity;
    identity.identity();
    g_RenderQueue.Clear();
    for (IRenderVisual* iv : RImplementation.Visuals) {
        if (!iv) continue;
        static_cast<vkRender_Visual*>(iv)->Submit(g_RenderQueue, identity, 0.0f);
    }
    g_RenderQueue.SortByKey();

    // --- DEPTH PREPASS: statics into the scene depth (CLEAR). The color pass
    // then LOADs depth and early-Z rejects every occluded pixel BEFORE the
    // (heavy) forward fragment shader runs — kills overdraw cost. Alpha-tested
    // items go through the AT depth variant (same discard threshold as the
    // color pass → identical coverage, no holes).
    if (prepass)
    {
        VkRenderingAttachmentInfo pdAtt{};
        pdAtt.sType                   = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        pdAtt.imageView               = ctx.depthView;
        pdAtt.imageLayout             = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        pdAtt.loadOp                  = VK_ATTACHMENT_LOAD_OP_CLEAR;
        pdAtt.storeOp                 = VK_ATTACHMENT_STORE_OP_STORE;
        pdAtt.clearValue.depthStencil = { 1.0f, 0 };

        VkRenderingInfo pri{};
        pri.sType             = VK_STRUCTURE_TYPE_RENDERING_INFO;
        pri.renderArea.extent = ctx.extent;
        pri.layerCount        = 1;
        pri.pDepthAttachment  = &pdAtt;
        vkCmdBeginRendering(cmd, &pri);

        VkViewport pvp{ 0.f, (float)ctx.extent.height, (float)ctx.extent.width, -(float)ctx.extent.height, 0.f, 1.f };
        vkCmdSetViewport(cmd, 0, 1, &pvp);
        VkRect2D psc{ {}, ctx.extent };
        vkCmdSetScissor(cmd, 0, 1, &psc);
        vkCmdSetDepthBias(cmd, 0.f, 0.f, 0.f);   // depth pipelines have dynamic bias — none here

        g_RenderQueue.FlushDepth(cmd, *ctx.viewProj, false /*include alpha-tested via AT variant*/);

        // Trees into the prepass depth too: GTAO sees trunks/canopies (R4's
        // gbuffer includes trees — most wilderness SSAO comes from them) and
        // the foliage color passes get early-Z. Same VP + alpha-ref as the
        // tree color pass (LEQUAL, no vertex animation) → re-raster matches.
        if (RImplementation.Trees) {
            CFrustum f;
            Fmatrix fullT = *ctx.viewProj;   // CreateFromMatrix takes a non-const ref
            f.CreateFromMatrix(fullT, FRUSTUM_P_LRTB | FRUSTUM_P_FAR);
            RImplementation.Trees->RenderDepth(cmd, *ctx.viewProj, -1, &f);
        }

        // NPCs too (alpha-tested) — GTAO contact darkening under characters +
        // early-Z behind them. Identical skinning math → depths match the
        // color pass exactly (LEQUAL).
        Skinned_RenderDepthPrepass(cmd, *ctx.viewProj);

        vkCmdEndRendering(cmd);
        SceneAttachmentBarrier(cmd);   // order prepass depth writes before the color pass

        // GTAO from the prepass depth (R4 SSAO analog): flip depth to
        // SHADER_READ, render half-res AO + blur, flip back. The color pass
        // below samples the result via EnvLight binding 8 (ambient/hemi only).
        if (SSAOPass::Enabled()) {
            ImageBarrier(cmd, Swapchain.m_DepthImage, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
            SSAOPass::Execute(cmd, ctx.extent);
            ImageBarrier(cmd, Swapchain.m_DepthImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
        }
    }

    // Targets travel in FrameContext; both are already in their attachment layout
    // (Begin set them, ExecutePasses ordered prior passes). No transition here.

    VkRenderingAttachmentInfo cAtt{};
    cAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    cAtt.imageView   = ctx.colorView;
    cAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    cAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;   // preserve background clear
    cAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingAttachmentInfo dAtt{};
    dAtt.sType                       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    dAtt.imageView                   = ctx.depthView;
    dAtt.imageLayout                 = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    // Prepass already laid the depth down — LOAD it so early-Z can reject;
    // without a prepass keep the old CLEAR behaviour.
    dAtt.loadOp                      = prepass ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
    // STORE so Pass_Sky can run depth-test against world's z and only paint
    // cleared pixels (z == 1.0).
    dAtt.storeOp                     = VK_ATTACHMENT_STORE_OP_STORE;
    dAtt.clearValue.depthStencil     = { 1.0f, 0 };

    VkRenderingInfo ri{};
    ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea.extent    = ctx.extent;
    ri.layerCount           = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments    = &cAtt;
    ri.pDepthAttachment     = &dAtt;
    vkCmdBeginRendering(cmd, &ri);

    // X-Ray builds D3D-style projection (Y-up clip space). Vulkan clip-space Y
    // is down → negate viewport height to flip the image. (VK_KHR_maintenance1
    // / Vulkan 1.1+ allows negative height.)
    VkViewport vp{};
    vp.x        = 0.0f;
    vp.y        = (float)ctx.extent.height;
    vp.width    = (float)ctx.extent.width;
    vp.height   = -(float)ctx.extent.height;
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{ {}, ctx.extent };
    vkCmdSetScissor(cmd, 0, 1, &sc);

    // MVP + uvScale are constant across the whole pass: world-space verts
    // (model = identity), and every level vertex format we render uses
    // SHORT2 SSCALED TCs. Push once before the loop. (The tess block at
    // offsets 84..112 is owned by RenderQueue::Flush.) Stage flags must
    // match the layout's push range exactly — GetPushStages, not VS|FS.
    WorldPush push{};
    push.mvp        = *ctx.viewProj;
    push.uvScale[0] = 1.0f / 1024.0f;
    push.uvScale[1] = 1.0f / 1024.0f;
    push.alphaRef    = -1.0f;  // Flush patches FS-tail (alphaRef + detailScale) per material
    push.detailScale = 0.0f;
    vkCmdPushConstants(cmd, PipelineCache::GetLayout(), PipelineCache::GetPushStages(),
                       0, sizeof(WorldPush), &push);

    // Per-pass DIAG: counts visuals by type and stride. The 'unsupported'
    // bucket is what's still rendered as a dummy — once it's zero on a given
    // level we're done porting visual classes.
    static bool s_diag_done = false;
    if (!s_diag_done) {
        u32 cntTotal=0, cntFV=0, cntHier=0, cntProg=0, cntSkel=0, cntTree=0, cntLOD=0, cntPart=0, cntOtherType=0;
        u32 cntNoVB=0, cntStride12=0, cnt32=0, cnt36=0, cnt40=0, cnt44=0, cntOtherStride=0;
        for (IRenderVisual* iv : RImplementation.Visuals) {
            if (!iv) continue;
            cntTotal++;
            auto* rv = static_cast<vkRender_Visual*>(iv);
            switch (rv->Type) {
                case MT_HIERRARHY:           cntHier++; continue;
                case MT_SKELETON_ANIM:
                case MT_SKELETON_RIGID:      cntSkel++;  continue;
                case MT_TREE_ST:
                case MT_TREE_PM:             cntTree++;  continue;
                case MT_LOD:                 cntLOD++;   continue;
                case MT_PARTICLE_EFFECT:
                case MT_PARTICLE_GROUP:      cntPart++;  continue;
            }
            if (rv->Type == MT_NORMAL || rv->Type == MT_PROGRESSIVE) {
                if (rv->Type == MT_NORMAL)      cntFV++;
                else                            cntProg++;
                auto* fv = static_cast<vkFVisual*>(rv);
                if (!fv->m_mesh.p_rm_Vertices) { cntNoVB++; continue; }
                switch (fv->m_mesh.vStride) {
                    case 12: cntStride12++;   break;
                    case 32: cnt32++;         break;
                    case 36: cnt36++;         break;
                    case 40: cnt40++;         break;
                    case 44: cnt44++;         break;
                    default: cntOtherStride++;break;
                }
            } else cntOtherType++;
        }
        Msg("[VK World] DIAG types: total=%u FV=%u Prog=%u Hier=%u | Skel=%u Tree=%u LOD=%u Part=%u other=%u",
            cntTotal, cntFV, cntProg, cntHier, cntSkel, cntTree, cntLOD, cntPart, cntOtherType);
        Msg("[VK World] DIAG mesh:  noVB=%u s12=%u s32=%u s36=%u s40=%u s44=%u sOther=%u",
            cntNoVB, cntStride12, cnt32, cnt36, cnt40, cnt44, cntOtherStride);
        s_diag_done = true;
    }

    // Phase 3: Flush the statics queue collected (and sorted) above, before the
    // prepass. With the prepass depth already in place, early-Z rejects every
    // occluded fragment before the forward shader runs.
    g_RenderQueue.Flush(ctx);

    // Dynamic (spawned) visuals — collected by CRender::add_Visual this frame.
    // Same render pass / depth as the statics; each carries its own world matrix,
    // so Flush pushes a per-item MVP (= xform * viewProj). vkFHierrarhyVisual /
    // CKinematics recurse into children inside Submit. Skinned leaves with no VB
    // (index-only stopgap) self-skip in Submit until the skinned path is ported.
    if (!g_DynamicVisuals.empty()) {
        g_RenderQueue.Clear();
        for (const DynVisual& d : g_DynamicVisuals) {
            if (d.vis) {
                g_RenderQueue.SetSubmitHemi(d.hemi);   // per-object sky-ambient gate
                d.vis->Submit(g_RenderQueue, d.xform, 0.0f);
            }
        }
        g_RenderQueue.SetSubmitHemi(1.0f);             // reset for the next (static) flush
        g_RenderQueue.SortByKey();
        // No tessellation for dynamics: the world VS emits MODEL-space
        // vWorldPos, so the TES camera-distance factor would be garbage
        // under a non-identity model xform.
        g_RenderQueue.SetAllowTess(false);
        g_RenderQueue.Flush(ctx);
        g_RenderQueue.SetAllowTess(true);
    }

    // Skinned dynamic leaves (NPCs / weapons / hands): GPU skinning, own pipeline +
    // bone SSBO. Same render pass (color + depth) as the statics above.
    Pass_Skinned(ctx);

    vkCmdEndRendering(cmd);
    // No exit transition: the image stays in COLOR_ATTACHMENT for the next pass.
    // ExecutePasses inserts the inter-pass barrier; End brings it to PRESENT.
}

}  // namespace VK
