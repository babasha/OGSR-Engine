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
#include "vk_world_gpu.h"      // VK::WorldGPU (GPU-driven static forward path)
#include <algorithm>
#include "vk_swapchain.h"      // Swapchain.m_Images / m_ImageViews
#include "CRender_Vulkan.h"    // RImplementation, vkRender_Visual ref
#include "vk_Visual.h"         // vkRender_Visual::Submit
#include "vk_pass_skinned.h"   // Pass_Skinned (skinned dynamic leaves)
#include "vk_env_light.h"      // EnvLight::Update — per-frame sun/hemi/ambient UBO (set 1)
#include "vk_command_buffer.h" // CommandManager.GetCurrentFrame() — in-flight slot
#include "vk_barriers.h"       // SceneAttachmentBarrier — prepass → color ordering
#include "vk_pass_ssao.h"      // GTAO from the prepass depth (before the color pass)
#include "vk_TreeManager.h"    // trees join the prepass depth (GTAO occluders + early-Z)
#include "vk_profiler.h"       // VK::Prof sub-zones (Depth/SSAO/Color/Statics/Skinned breakdown)
#include "vk_vrs.h"            // VK::VRS — variable rate shading (depth-driven SRI)
#include "vk_vsm.h"            // VK::VSM — virtual shadow maps page marking (WIP, r_vsm)
#include "vk_clustered.h"      // VK::Clustered — clustered forward light cull (r_clustered)
#include "vk_volumetrics.h"    // VK::Vol — froxel volumetric inject/integrate (r_vol)
#include "vk_pass_particles.h" // VK::CollectSmokeParticles — Stage-1 smoke media inject
#include "vk_DetailManager.h"  // RImplementation.Details — shared Hi-Z pyramid (r_hzb_cull)

extern float ps_r_vol_smoke_inject;   // gate the per-frame smoke collect (0 = skip)
#include "HW_Vulkan.h"         // VulkanHW.m_bVRSSupported
#include "../../xr_3da/device.h" // Device.dwTimeGlobal (cull diag throttle)

// Global scope (not in namespace VK → avoid VK::ps_r_cull mangling), like the rain externs.
extern int ps_r_cull;
extern int ps_r_gpu_world;   // GPU-driven static world forward path (A/B with 0)
extern int ps_r_clustered;   // clustered forward light cull (A/B with 0)
extern int ps_r_clustered_debug; // clustered froxel light-count heatmap (also activates the cull)
extern int ps_r_hzb_cull;    // Hi-Z occlusion cull of the GPU-driven static color pass (A/B with 0)
extern int ps_r_ssao_npc_normals;   // NPC normal G-buffer for GTAO (global scope: block-scope extern in namespace VK would mangle → LNK2001)

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

    // New frame: drop last frame's late-glass list (normally consumed by
    // Pass_WorldGlass; this also covers frames where that pass didn't run).
    g_RenderQueue.ClearGlass();

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

    // VSM (r_vsm): compute this frame's clipmap params + ensure the screen-space mask
    // target BEFORE EnvLight binds it on the receiver set (the actual mark/alloc/render/
    // resolve happen in the prepass below; this only sets up params + the mask image).
    if (VK::VSM::Wanted()) VK::VSM::BeginFrame(Device.vCameraPosition, ctx.extent);

    // Refresh this frame's env-lighting UBO (sun/hemi/ambient) once, before any
    // draw. RenderQueue::Flush binds the resulting set at set 1; Pass_Skinned
    // reuses the same set (at set 2). Fence-guarded slot → no in-flight write hazard.
    EnvLight::Update(CommandManager.GetCurrentFrame());

    // Collect the static queue up front — the depth prepass and the color pass
    // below both consume it. With r_gpu_world the GPU draws the eligible static set
    // (Cull below) and the CPU touches ONLY the pre-built non-GPU static leaves
    // (wmark/tess/no-diffuse) — no per-frame walk over all level visuals, no
    // hierarchy double-submit. This DECOUPLES per-frame static CPU from total
    // object count (the thing that walls detail-heavy levels). Without it (A/B),
    // the old full walk runs.
    const bool gpuWorld = ps_r_gpu_world && WorldGPU::Built();

    // CPU-cost diag: time the static collect vs the whole-frame cpu — shows how
    // much CPU the collection costs (and that gpuWorld decouples it from objects).
    static double s_qpcToMs = []{ LARGE_INTEGER f; QueryPerformanceFrequency(&f); return 1000.0 / double(f.QuadPart); }();
    LARGE_INTEGER t_collA; QueryPerformanceCounter(&t_collA);

    Fmatrix identity;
    identity.identity();
    g_RenderQueue.Clear();
    const bool doCull = (ps_r_cull != 0);
    u32 cullTotal = 0, cullSkipped = 0;
    if (gpuWorld) {
        // Only the small pre-built non-GPU static set (frustum-culled). The GPU set
        // is compute-culled + drawn by WorldGPU::Draw* (no CPU walk for it at all).
        cullTotal = WorldGPU::SubmitCpuMeshes(g_RenderQueue, *ctx.viewProj, doCull);
    } else {
        // FRUSTUM CULLING (old path): walk ALL level visuals, frustum-test, Submit.
        // Lossless (off-frustum geometry produces no pixels). Toggle: r_cull.
        CFrustum camFrustum;
        if (doCull) { Fmatrix vp = *ctx.viewProj; camFrustum.CreateFromMatrix(vp, FRUSTUM_P_LRTB | FRUSTUM_P_FAR); }
        for (IRenderVisual* iv : RImplementation.Visuals) {
            if (!iv) continue;
            auto* rv = static_cast<vkRender_Visual*>(iv);
            ++cullTotal;
            if (doCull) {
                const Fsphere& bs = rv->vis.sphere;
                if (bs.R > 0.f && !camFrustum.testSphere_dirty(bs.P, bs.R)) { ++cullSkipped; continue; }
            }
            rv->Submit(g_RenderQueue, identity, 0.0f);
        }
    }
    g_RenderQueue.SortByKey();
    LARGE_INTEGER t_collB; QueryPerformanceCounter(&t_collB);
    const double cpuCollectMs = double(t_collB.QuadPart - t_collA.QuadPart) * s_qpcToMs;
    { static u32 s_log = 0; if (Device.dwTimeGlobal > s_log + 3000) { s_log = Device.dwTimeGlobal;
        if (gpuWorld) Msg("[VK WorldGPU] CPU set drawn=%u (GPU set %u on GPU)", cullTotal, WorldGPU::SetSize());
        else          Msg("[VK Cull] world statics: %u/%u drawn (%u culled)", cullTotal - cullSkipped, cullTotal, cullSkipped); } }

    // GPU-driven static world: compute-cull the GPU static set (outside any render
    // pass); DrawDepth/DrawColor below consume its indirect buffer in both passes.
    if (gpuWorld)
        WorldGPU::Cull(cmd, *ctx.viewProj);

    // Clustered forward: bin this frame's dynamic lights into the froxel grid
    // (compute, outside the render pass). EnvLight::Update above already uploaded
    // the light list to this slot's SSBO; the world/skinned color fragments read
    // the resulting per-cluster lists. (Init is lazy via EnvLight::Update.)
    if ((ps_r_clustered || ps_r_clustered_debug) && VK::Clustered::Ready()) {
        const VK::ProjTerms cpt = VK::DeriveProjTerms(*ctx.viewProj);
        VK::Clustered::Cull(cmd, cpt, Device.vCameraPosition, ctx.extent,
                            CommandManager.GetCurrentFrame(),
                            VK::Lights::CollectFrame(Device.vCameraPosition).count);
    }

    // Froxel volumetrics (r_vol): inject + integrate the view-frustum volume
    // (compute, OUTSIDE the render pass). Needs NO scene depth — the froxel world
    // pos is analytic — so it runs here, before the prepass. EnvLight::Update +
    // the earlier Pass_SunShadow already produced the sun cascade maps it samples.
    // The composite is folded into the tonemap pass (gated on r_vol).
    if (VK::Vol::Wanted() && VK::Vol::Ready()) {
        const VK::ProjTerms vpt = VK::DeriveProjTerms(*ctx.viewProj);
        // Stage-1 VMS: gather this frame's alpha-smoke particles to inject as media.
        static xr_vector<VK::Vol::SmokeParticle> s_smoke;
        s_smoke.clear();
        if (ps_r_vol_smoke_inject > 0.0f)
            VK::CollectSmokeParticles(s_smoke);
        VK::Vol::Execute(cmd, vpt, CommandManager.GetCurrentFrame(),
                         s_smoke.empty() ? nullptr : s_smoke.data(), (u32)s_smoke.size());
    }

    // Throttled average: static-collect CPU vs the whole-frame cpu — confirms the
    // collect cost (now decoupled from object count when gpuWorld).
    {
        static double s_accColl = 0; static u32 s_accN = 0, s_accLast = 0;
        s_accColl += cpuCollectMs; ++s_accN;
        if (Device.dwTimeGlobal > s_accLast + 3000 && s_accN) {
            Msg("[VK WorldGPU] CPU diag: collect=%.2fms (avg/%u fr, gpuWorld=%d) | frame cpu=%.2fms",
                s_accColl / s_accN, s_accN, gpuWorld ? 1 : 0, Device.fTimeDeltaRealMS);
            s_accColl = 0; s_accN = 0; s_accLast = Device.dwTimeGlobal;
        }
    }

    // --- DEPTH PREPASS: statics into the scene depth (CLEAR). The color pass
    // then LOADs depth and early-Z rejects every occluded pixel BEFORE the
    // (heavy) forward fragment shader runs — kills overdraw cost. Alpha-tested
    // items go through the AT depth variant (same discard threshold as the
    // color pass → identical coverage, no holes).
    if (prepass)
    {
        const int zDepth = VK::Prof::ZoneBegin(cmd, "World/Depth");
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

        g_RenderQueue.FlushDepth(cmd, *ctx.viewProj, false /*include alpha-tested*/, false /*alphaTestedOnly*/, true /*displaceTerrain: snow volume matches color*/);
        if (gpuWorld) WorldGPU::DrawDepth(cmd, *ctx.viewProj, true /*displaceTerrain*/);   // GPU static set into the prepass depth

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
        VK::Prof::ZoneEnd(cmd, zDepth);

        // NPC NORMAL G-buffer for GTAO: render the near-camera skinned casters into
        // the SSAO normal RT, depth-tested against the prepass depth (no depth write).
        // Gives GTAO real per-pixel normals where NPCs are visible → no depth-
        // derivative speckle on characters; statics/trees fall back to depth recon.
        // The RT is always CLEARed (defined for the GTAO sample) even if the draw is
        // gated off. ps_r_ssao_npc_normals (vk_console_min) — A/B toggle.
        if (SSAOPass::Enabled() && SSAOPass::GetNormalView() != VK_NULL_HANDLE)
        {
            const int zNrm = VK::Prof::ZoneBegin(cmd, "World/AOnormal");
            ImageBarrier(cmd, SSAOPass::GetNormalImage(), VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

            VkRenderingAttachmentInfo nAtt{};
            nAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            nAtt.imageView   = SSAOPass::GetNormalView();
            nAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            nAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;       // a=0 everywhere → GTAO fallback
            nAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
            nAtt.clearValue.color = { { 0.f, 0.f, 0.f, 0.f } };

            VkRenderingAttachmentInfo nDepth{};
            nDepth.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            nDepth.imageView   = ctx.depthView;
            nDepth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            nDepth.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;       // test the prepass depth
            nDepth.storeOp     = VK_ATTACHMENT_STORE_OP_DONT_CARE; // pipeline has depthWrite off

            VkRenderingInfo nri{};
            nri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
            nri.renderArea.extent    = ctx.extent;
            nri.layerCount           = 1;
            nri.colorAttachmentCount = 1;
            nri.pColorAttachments    = &nAtt;
            nri.pDepthAttachment     = &nDepth;
            vkCmdBeginRendering(cmd, &nri);

            VkViewport nvp{ 0.f, (float)ctx.extent.height, (float)ctx.extent.width, -(float)ctx.extent.height, 0.f, 1.f };
            vkCmdSetViewport(cmd, 0, 1, &nvp);
            VkRect2D nsc{ {}, ctx.extent };
            vkCmdSetScissor(cmd, 0, 1, &nsc);

            if (ps_r_ssao_npc_normals)
                Skinned_RenderNormalPrepass(cmd, *ctx.viewProj);

            vkCmdEndRendering(cmd);
            ImageBarrier(cmd, SSAOPass::GetNormalImage(), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            VK::Prof::ZoneEnd(cmd, zNrm);
        }

        // GTAO from the prepass depth (R4 SSAO analog): flip depth to
        // SHADER_READ, render half-res AO + blur, flip back. The color pass
        // below samples the result via EnvLight binding 8 (ambient/hemi only).
        const int zSSAO = VK::Prof::ZoneBegin(cmd, "World/SSAO");
        if (SSAOPass::Enabled()) {
            ImageBarrier(cmd, Swapchain.m_DepthImage, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
            SSAOPass::Execute(cmd, ctx.extent);
            ImageBarrier(cmd, Swapchain.m_DepthImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
        }
        VK::Prof::ZoneEnd(cmd, zSSAO);
        // Motion vectors moved OUT of the World pass: the static depth-reconstruction
        // now runs in MotionVec::ExecuteDynamic (registered after LODs), where the
        // depth holds ALL opaque geometry — so trees (rigid → exact) and grass get
        // correct camera motion for free, not the background's. See vk_motionvec.cpp.

        // VRS: build this frame's depth-driven shading-rate image (compute reads the
        // prepass depth → coarser rate with distance). Needs depth SHADER_READ.
        if (VK::VRS::Wanted()) {
            const int zVRS = VK::Prof::ZoneBegin(cmd, "World/VRSbuild");
            ImageBarrier(cmd, Swapchain.m_DepthImage, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
            const VK::ProjTerms pt = VK::DeriveProjTerms(*ctx.viewProj);
            VK::VRS::BuildFromDepth(cmd, CommandManager.GetCurrentFrame(), ctx.depthView,
                                    ctx.extent, pt.p43, pt.p33);
            ImageBarrier(cmd, Swapchain.m_DepthImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
            VK::Prof::ZoneEnd(cmd, zVRS);
        }

        // VSM page marking (WIP, r_vsm): which sun-clipmap pages do visible pixels
        // need? Reads the prepass depth in SHADER_READ. Gated on Wanted() (the cvar,
        // no Init dependency); MarkPages lazily Inits + no-ops if not yet Ready.
        if (VK::VSM::Wanted()) {
            // Scene depth is sampled by the VSM mark/resolve COMPUTE dispatches. The auto
            // ImageBarrier derives SHADER_READ as FRAGMENT-only (vk_barriers DeriveStageAccess),
            // so the compute read is left unordered vs the prepass depth write — a latent sync
            // hole (works today only because the prepass finished long before; breaks under async
            // compute / strict drivers). Use explicit barriers that include COMPUTE (mirrors VSM's
            // own atlasToRead). [#3]
            auto depthToRead = [&]() {
                VkImageMemoryBarrier2 b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
                b.srcStageMask  = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
                b.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                b.dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
                b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                b.oldLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
                b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.image = Swapchain.m_DepthImage;
                b.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
                VkDependencyInfo di{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                vkCmdPipelineBarrier2(cmd, &di);
            };
            auto depthToAttach = [&]() {
                VkImageMemoryBarrier2 b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
                b.srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                b.srcAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                b.dstStageMask  = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
                b.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
                b.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                b.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
                b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.image = Swapchain.m_DepthImage;
                b.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
                VkDependencyInfo di{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
                vkCmdPipelineBarrier2(cmd, &di);
            };

            const int zVSM = VK::Prof::ZoneBegin(cmd, "World/VSMmark");
            depthToRead();
            VK::VSM::MarkPages(cmd, ctx.depthView, ctx.extent, *ctx.viewProj);
            depthToAttach();
            VK::Prof::ZoneEnd(cmd, zVSM);
            // Rasterize the casters into the VSM atlas (own render pass; the scene
            // depth is already restored above).
            const int zVSMr = VK::Prof::ZoneBegin(cmd, "World/VSMrender");
            VK::VSM::RenderAtlas(cmd);
            VK::Prof::ZoneEnd(cmd, zVSMr);
            // Temporal resolve: sample the atlas per screen pixel + reproject history →
            // the screen-space sun-shadow mask the receivers read. Needs the prepass
            // depth in SHADER_READ (same transition pattern as the mark pass above).
            const int zVSMres = VK::Prof::ZoneBegin(cmd, "World/VSMresolve");
            depthToRead();
            VK::VSM::ResolveMask(cmd, ctx.depthView, ctx.extent, *ctx.viewProj);
            depthToAttach();
            VK::Prof::ZoneEnd(cmd, zVSMres);
        }
    }

    // r_hzb_cull (Phase A): build the Hi-Z pyramid from THIS frame's PREPASS depth,
    // then occlusion-cull the GPU static set into a 2nd indirect buffer drawn only
    // by the color pass below — meshes fully behind nearer geometry skip the heavy
    // forward shading. Outside any render pass; depth is DEPTH_ATTACHMENT here
    // (SSAO/VRS/VSM restored it), which BuildHZBForFrame expects. We SHARE the grass
    // pyramid: grass Render runs after Pass_World, so its BuildHZB no-ops (same-frame
    // stamp) and reuses this prepass-depth build. The depth prepass already drew the
    // full frustum set (it is the pyramid source), so this never over-culls.
    bool worldOccluded = false;
    if (prepass && gpuWorld && ps_r_hzb_cull && WorldGPU::OcclusionReady()
        && RImplementation.Details && RImplementation.Details->HZBReady())
    {
        const int zHZB = VK::Prof::ZoneBegin(cmd, "World/HZBcull");
        RImplementation.Details->BuildHZBForFrame(ctx);
        WorldGPU::CullColor(cmd, *ctx.viewProj, Device.vCameraPosition,
                            RImplementation.Details->HZBView(), RImplementation.Details->HZBSampler());
        worldOccluded = true;
        VK::Prof::ZoneEnd(cmd, zHZB);
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

    VkRenderingFragmentShadingRateAttachmentInfoKHR sriAtt{ VK_STRUCTURE_TYPE_RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT_INFO_KHR };

    VkRenderingInfo ri{};
    ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea.extent    = ctx.extent;
    ri.layerCount           = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments    = &cAtt;
    ri.pDepthAttachment     = &dAtt;
    // Attach the shading-rate image so distant tiles coarse-shade (built above).
    if (VK::VRS::Wanted() && VK::VRS::GetView() != VK_NULL_HANDLE) {
        sriAtt.imageView                      = VK::VRS::GetView();
        sriAtt.imageLayout                    = VK_IMAGE_LAYOUT_FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR;
        sriAtt.shadingRateAttachmentTexelSize = VK::VRS::TexelSize();
        ri.pNext = &sriAtt;
    }
    const int zColor = VK::Prof::ZoneBegin(cmd, "World/Color");
    vkCmdBeginRendering(cmd, &ri);
    // World-color pipelines carry the dynamic FSR state → must set the rate (combiner
    // REPLACE) before any draw. With no SRI bound (r_vrs 0) the attachment defaults to
    // 1x1 → no effect.
    if (VulkanHW.m_bVRSSupported) VK::VRS::CmdSetRate(cmd);

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
    const int zStatics = VK::Prof::ZoneBegin(cmd, "World/Statics");
    g_RenderQueue.Flush(ctx);
    // GPU static set (drawn after the CPU flush so DrawColor's self-contained
    // pushes don't disturb Flush's push state). EnvLight set = set 1. worldOccluded
    // → draw the Hi-Z-culled set (CullColor ran above); else the frustum set.
    if (gpuWorld) WorldGPU::DrawColor(cmd, *ctx.viewProj, EnvLight::GetCurrentSet(), worldOccluded);
    VK::Prof::ZoneEnd(cmd, zStatics);

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
    const int zSkin = VK::Prof::ZoneBegin(cmd, "World/Skinned");
    Pass_Skinned(ctx);
    VK::Prof::ZoneEnd(cmd, zSkin);

    vkCmdEndRendering(cmd);
    VK::Prof::ZoneEnd(cmd, zColor);
    // No exit transition: the image stays in COLOR_ATTACHMENT for the next pass.
    // ExecutePasses inserts the inter-pass barrier; End brings it to PRESENT.
}

// Late translucent pass: GLASS panes (WorldMaterial::isGlass) accumulated by
// RenderQueue::Push across the statics/dynamics flushes draw HERE — after the
// whole opaque world (GPU statics, trees, grass, LODs, sky). They blend without
// z-write, so drawing them inside the normal flush let everything rendered
// after them overwrite the blended pixels: level windows looked missing, prop
// panes opaque. Registered between "Shafts" and "Wallmarks" (CRender_Vulkan).
void Pass_WorldGlass(FrameContext& ctx)
{
    if (ctx.cmd == VK_NULL_HANDLE) return;
    if (!g_RenderQueue.HasGlass()) return;
    VkCommandBuffer cmd = ctx.cmd;
    const int z = VK::Prof::ZoneBegin(cmd, "World/Glass");
    // LOAD colour+depth, depth test vs the finished opaque scene, no depth write
    // (the glass pipelines are the wmark variants — no z-write by construction).
    BeginOverlayRendering(cmd, ctx, VK_ATTACHMENT_STORE_OP_DONT_CARE);
    g_RenderQueue.FlushGlass(ctx);
    vkCmdEndRendering(cmd);
    VK::Prof::ZoneEnd(cmd, z);
}

}  // namespace VK
