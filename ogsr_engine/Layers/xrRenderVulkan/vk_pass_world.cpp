#include "stdafx.h"
#include "vk_pass_world.h"
#include "vk_pipeline_cache.h"
#include "vk_render_queue.h"   // g_RenderQueue (phase 3)
#include "vk_swapchain.h"      // Swapchain.m_Images / m_ImageViews
#include "CRender_Vulkan.h"    // RImplementation, vkRender_Visual ref
#include "vk_Visual.h"         // vkRender_Visual::Submit
#include "vk_pass_skinned.h"   // Pass_Skinned (skinned dynamic leaves)

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
    dAtt.loadOp                      = VK_ATTACHMENT_LOAD_OP_CLEAR;
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
    // SHORT2 SSCALED TCs. Push once before the loop.
    WorldPush push{};
    push.mvp        = *ctx.viewProj;
    push.uvScale[0] = 1.0f / 1024.0f;
    push.uvScale[1] = 1.0f / 1024.0f;
    push.alphaRef    = -1.0f;  // Flush patches FS-tail (alphaRef + detailScale) per material
    push.detailScale = 0.0f;
    vkCmdPushConstants(cmd, PipelineCache::GetLayout(),
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
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

    // Phase 3: Submit → Sort → Flush. Visuals push DrawItems during
    // traversal (Hier nodes recurse into children); the queue sorts by
    // (depthTest, stride, tcOffset, VB hash) so adjacent items share
    // pipeline/VB binds, then Flush issues the actual Vulkan draws.
    Fmatrix identity;
    identity.identity();

    g_RenderQueue.Clear();
    for (IRenderVisual* iv : RImplementation.Visuals) {
        if (!iv) continue;
        static_cast<vkRender_Visual*>(iv)->Submit(g_RenderQueue, identity, 0.0f);
    }
    g_RenderQueue.SortByKey();
    g_RenderQueue.Flush(ctx);

    // Dynamic (spawned) visuals — collected by CRender::add_Visual this frame.
    // Same render pass / depth as the statics; each carries its own world matrix,
    // so Flush pushes a per-item MVP (= xform * viewProj). vkFHierrarhyVisual /
    // CKinematics recurse into children inside Submit. Skinned leaves with no VB
    // (index-only stopgap) self-skip in Submit until the skinned path is ported.
    if (!g_DynamicVisuals.empty()) {
        g_RenderQueue.Clear();
        for (const DynVisual& d : g_DynamicVisuals) {
            if (d.vis)
                d.vis->Submit(g_RenderQueue, d.xform, 0.0f);
        }
        g_RenderQueue.SortByKey();
        g_RenderQueue.Flush(ctx);
    }

    // Skinned dynamic leaves (NPCs / weapons / hands): GPU skinning, own pipeline +
    // bone SSBO. Same render pass (color + depth) as the statics above.
    Pass_Skinned(ctx);

    vkCmdEndRendering(cmd);
    // No exit transition: the image stays in COLOR_ATTACHMENT for the next pass.
    // ExecutePasses inserts the inter-pass barrier; End brings it to PRESENT.
}

}  // namespace VK
