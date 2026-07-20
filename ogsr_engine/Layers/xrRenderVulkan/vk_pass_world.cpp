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
#include "vk_instance_gpu.h"   // InstanceGPU — GPU-driven instanced host scene (editor)
#include "vk_command_buffer.h" // CommandManager.GetCurrentFrame() — in-flight slot
#include "vk_barriers.h"       // SceneAttachmentBarrier + ImageState — prepass → color ordering
#include "vk_framegraph.h"     // VK::g_FrameGraph — frame-wide depth/color state (depth-thrash coalescing)
#include "vk_pass_ssao.h"      // GTAO from the prepass depth (before the color pass)
#include "vk_TreeManager.h"    // trees join the prepass depth (GTAO occluders + early-Z)
#include "vk_profiler.h"       // VK::Prof sub-zones (Depth/SSAO/Color/Statics/Skinned breakdown)
#include "vk_vrs.h"            // VK::VRS — variable rate shading (depth-driven SRI)
#include "vk_vsm.h"            // VK::VSM — virtual shadow maps page marking (WIP, r_vsm)
#include "vk_clustered.h"      // VK::Clustered — clustered forward light cull (r_clustered)
#include "vk_volumetrics.h"    // VK::Vol — froxel volumetric inject/integrate (r_vol)
#include "vk_terrain_cache.h"  // VK::TerrainCache — composite ground cache bake (r_terra_cache)
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
extern int ps_r_cluster_debug; // cluster-LOD debug overlay: 1 = fill colors, 2 = wireframe (live)
extern int ps_r_fg_coalesce; // framegraph: coalesce SSAO/VRS/VSM depth-read round-trips (A/B with 0)
extern int ps_r_ssao_npc_normals;   // NPC normal G-buffer for GTAO (global scope: block-scope extern in namespace VK would mangle → LNK2001)
extern int ps_r_compose;     // Stage D world composition: total chunk count (home + clones), 0/1 = off
extern int ps_r_vrs;         // VRS level (0/1/2) — here only to gate the FS-invocation stats query
extern int ps_r_vrs_force;   // diag: force pipeline-rate NxN on the world color pass, ignoring the SRI

namespace VK {

// Dynamic (spawned) visuals collected by CRender::add_Visual this frame, drained
// below after the level statics. Declared in vk_pass_world.h.
xr_vector<DynVisual> g_DynamicVisuals;
xr_vector<DynVisual> g_HudVisuals;

// ---------------------------------------------------------------------------
// Stage D world composition (r_compose N): replicate the loaded level's GPU
// static set as N-1 clone chunks tiled on a grid around the home footprint.
// A chunk is {the ONE resident dataset, world offset T}: it culls with
// viewProj·T (frustum planes land in chunk-local space for free) and draws
// with mvp = viewProj·T — zero shader changes, zero extra geometry VRAM.
// vWorldPos in the FS stays in HOME coordinates (fog/VSM/dyn-light sampling
// behaves "as the original tile") — known slice-1 tradeoff.
// Clones are render-only statics: no collision/AI; trees/grass/LODs/CPU
// leaves stay home-only in this slice.
// ---------------------------------------------------------------------------
namespace Compose {
    static xr_vector<Fvector> s_Offsets;    // clone offsets (home excluded)
    static size_t             s_vis = 0;    // Visuals.size() the layout was built from
    static Fvector            s_center{};   // home footprint center (world)
    static float              s_radius = 0.f; // bounding-sphere radius of the footprint

    static void EnsureLayout()
    {
        const size_t nVis = RImplementation.Visuals.size();
        if (!s_Offsets.empty() && s_vis == nVis) return;
        s_Offsets.clear();
        s_vis = nVis;
        s_radius = 0.f;
        if (!nVis) return;
        // Chunk pitch MUST be >= the level's full GEOMETRY envelope (sphere P±R),
        // not just the playable footprint. The first attempt tiled by the 2-98%
        // playable width (~3.2 km) while the actual geometry — the fake backdrop
        // hills + terrain skirt X-Ray adds to hide the map edge — reaches R≈2.3 km
        // from centre. A clone at +3.2 km then had geometry back at 3.2-2.3≈0.9 km,
        // overlapping home (which reaches 2.3 km): the overlap wrote nearer depth
        // over home walls (early-Z ate them) and its backdrop sliced through the
        // scene as a grey plane. Envelope pitch makes tiles ABUT with no overlap.
        // (Copies of one level still seam backdrop-to-backdrop — inherent to
        // cloning; truly seamless needs authored-abutting levels, a later slice.)
        Fvector mn{ 1e9f, 1e9f, 1e9f }, mx{ -1e9f, -1e9f, -1e9f };
        u32 nUsed = 0;
        for (IRenderVisual* iv : RImplementation.Visuals) {
            if (!iv) continue;
            const Fsphere& bs = static_cast<vkRender_Visual*>(iv)->vis.sphere;
            if (bs.R <= 0.f || bs.R > 2000.f) continue;   // skip sky domes / fog volumes
            mn.x = _min(mn.x, bs.P.x - bs.R); mx.x = _max(mx.x, bs.P.x + bs.R);
            mn.y = _min(mn.y, bs.P.y - bs.R); mx.y = _max(mx.y, bs.P.y + bs.R);
            mn.z = _min(mn.z, bs.P.z - bs.R); mx.z = _max(mx.z, bs.P.z + bs.R);
            ++nUsed;
        }
        if (nUsed < 2 || mx.x <= mn.x || mx.z <= mn.z) return;
        // 8 m seam gap so abutting tiles don't z-fight at the shared edge.
        const float pitchX = (mx.x - mn.x) + 8.f;
        const float pitchZ = (mx.z - mn.z) + 8.f;
        s_center.set((mn.x + mx.x) * 0.5f, (mn.y + mx.y) * 0.5f, (mn.z + mx.z) * 0.5f);
        // Bounding sphere of one chunk envelope (for the per-chunk frustum
        // pre-check in the hooks — chunks fully out of view never dispatch).
        const float ex = (mx.x - mn.x) * 0.5f, ey = _max(1.f, (mx.y - mn.y) * 0.5f), ez = (mx.z - mn.z) * 0.5f;
        s_radius = _sqrt(ex * ex + ey * ey + ez * ez);
        // Ring order around home: 8 first-ring neighbours, then the 16 of ring 2
        // → supports up to r_compose 25 (5×5 world).
        static const int ring[24][2] = {
            { 1,0},{-1,0},{0, 1},{0,-1},{ 1, 1},{-1, 1},{ 1,-1},{-1,-1},
            { 2,0},{-2,0},{0, 2},{0,-2},{ 2, 1},{ 2,-1},{-2, 1},{-2,-1},
            { 1,2},{-1,2},{ 1,-2},{-1,-2},{ 2, 2},{-2, 2},{ 2,-2},{-2,-2} };
        for (const auto& rc : ring) {
            Fvector o; o.set(rc[0] * pitchX, 0.f, rc[1] * pitchZ);
            s_Offsets.push_back(o);
        }
        Msg("[VK Compose] layout: envelope %.0fx%.0fm (%u visuals), pitch %.0fx%.0fm, chunk R=%.0fm, %zu clone slots",
            mx.x - mn.x, mx.z - mn.z, nUsed, pitchX, pitchZ, s_radius, s_Offsets.size());
    }

    // Number of clone chunks to draw this frame (0 = composition off).
    static u32 CloneCount()
    {
        if (ps_r_compose <= 1) return 0;
        EnsureLayout();
        return (u32)_min((size_t)(ps_r_compose - 1), s_Offsets.size());
    }

    // viewProj·T and chunk-local camera for clone c.
    static void ChunkView(u32 c, const Fmatrix& viewProj, Fmatrix& outVP, Fvector& outCam)
    {
        Fmatrix T; T.translate(s_Offsets[c]);
        outVP.mul(viewProj, T);                      // same composition as Flush's mvp = viewProj·xform
        outCam.sub(Device.vCameraPosition, s_Offsets[c]);
    }

    // CPU pre-check: is clone c's footprint sphere in the camera frustum at all?
    // Chunks beyond the far plane / behind the camera skip their cull dispatch
    // AND their render pass entirely — with 24 slots and a ~1.5 km far plane
    // only the adjacent visible chunks cost anything (measured: 24 blind culls
    // of the full entry set burned ~10 ms/frame of pure compute).
    static bool ChunkInView(u32 c, CFrustum& f)
    {
        if (s_radius <= 0.f) return true;
        Fvector p; p.add(s_center, s_Offsets[c]);
        return !!f.testSphere_dirty(p, s_radius);
    }
} // namespace Compose

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

    // Terrain composite cache (r_terra_cache): probe the terrain uv affine /
    // re-bake the height+weights clipmap when the camera left the window.
    // MUST run BEFORE EnvLight::Update — the UBO's tcache transform is derived
    // from the BAKED window origin; baking after the UBO write made a re-bake
    // frame sample the NEW image with the OLD transform (one-frame relief pop
    // every few metres of running). Compute, outside any render pass, before
    // the prepass — both terrain FS passes sample the frame-static result.
    VK::TerrainCache::Update(cmd);

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
    // Pool compaction freed the GPU-set meshes' CPU slices — the old full-walk
    // path can no longer draw them, so the GPU path stays forced on. For a true
    // r_gpu_world A/B set r_pool_compact 0 and reload the level.
    const bool gpuWorld = (ps_r_gpu_world || WorldGPU::PoolsCompacted()) && WorldGPU::Built();
    if (!ps_r_gpu_world && gpuWorld) {
        static u32 s_warned = 0;
        if (Device.dwTimeGlobal > s_warned + 5000) { s_warned = Device.dwTimeGlobal;
            Msg("![VK WorldGPU] r_gpu_world 0 ignored: pools compacted (r_pool_compact 0 + reload for A/B)"); }
    }

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

        // NOTE: Stage D clone depth is DELIBERATELY NOT drawn here. Clones are
        // distant backdrop; letting their (6+ km) depth into the scene buffer
        // before SSAO / VSM MarkPages / Hi-Z would pollute those home-only
        // screen-space passes (VSM would mark pages 6 km away and starve the home
        // pages → home surfaces fall back to "fully lit" = bright white patches).
        // The clone depth+color is drawn AFTER those consumers, just before the
        // home color pass — see the Compose block below.

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

        // --- SCENE-DEPTH READ WINDOW (framegraph seam) ---------------------
        // SSAO, VRS, VSM-mark and VSM-resolve all sample THIS prepass depth. Each
        // historically did its own ATTACHMENT->SHADER_READ->ATTACHMENT round-trip
        // (4 round-trips = 8 depth barriers that serialize the GPU). The ImageState
        // tracker coalesces them: with r_fg_coalesce the depth flips to SHADER_READ
        // once, every consumer runs, then it flips back once before the color pass.
        // VSM::RenderAtlas rasters into its OWN atlas depth (never scene depth), so
        // it rides happily inside the read window. r_fg_coalesce 0 = the exact old
        // per-consumer round-trips (A/B fallback). The read barrier now includes
        // COMPUTE (SSAO/VRS/VSM are compute dispatches) — stricter than the old auto
        // ImageBarrier's FRAGMENT-only derivation, closing a latent sync hole [#3].
        // The depth state is the frame-wide g_FrameGraph.Depth() (seeded to
        // DEPTH_ATTACHMENT in CRender::Begin; the prepass just wrote it in that same
        // layout) — routing through the graph so any later pass sees the true state.
        VK::ImageState& depthState = VK::g_FrameGraph.Depth();
        const bool fgCoalesce = (ps_r_fg_coalesce != 0);
        auto depthToRead = [&]() {
            depthState.Require(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        };
        auto depthToAttach = [&]() {
            depthState.Require(cmd, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT);
        };

        // GTAO from the prepass depth (R4 SSAO analog): render half-res AO + blur.
        // The color pass below samples the result via EnvLight binding 8 (ambient/hemi).
        const int zSSAO = VK::Prof::ZoneBegin(cmd, "World/SSAO");
        if (SSAOPass::Enabled()) {
            depthToRead();
            SSAOPass::Execute(cmd, ctx.extent);
            if (!fgCoalesce) depthToAttach();
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
            depthToRead();
            const VK::ProjTerms pt = VK::DeriveProjTerms(*ctx.viewProj);
            VK::VRS::BuildFromDepth(cmd, CommandManager.GetCurrentFrame(), ctx.depthView,
                                    ctx.extent, pt.p43, pt.p33);
            if (!fgCoalesce) depthToAttach();
            VK::Prof::ZoneEnd(cmd, zVRS);
        }

        // VSM page marking (WIP, r_vsm): which sun-clipmap pages do visible pixels
        // need? Reads the prepass depth in SHADER_READ. Gated on Wanted() (the cvar,
        // no Init dependency); MarkPages lazily Inits + no-ops if not yet Ready.
        // NIGHT FREEZE: with the sun below the horizon it lights nothing, so the whole
        // sun-shadow update (mark + atlas raster + resolve, ~1.7 ms) is pure waste. Skip
        // it and keep the last daylit mask — receivers multiply it by sun_color≈0, so
        // there is no visible change. Resumes automatically at dawn (NightFrozen() clears).
        if (VK::VSM::Wanted() && !VK::VSM::NightFrozen()) {
            const int zVSM = VK::Prof::ZoneBegin(cmd, "World/VSMmark");
            depthToRead();
            VK::VSM::MarkPages(cmd, ctx.depthView, ctx.extent, *ctx.viewProj);
            if (!fgCoalesce) depthToAttach();
            VK::Prof::ZoneEnd(cmd, zVSM);
            // Rasterize the casters into the VSM atlas (own render pass, own atlas
            // depth — never touches scene depth, so it stays inside the read window).
            const int zVSMr = VK::Prof::ZoneBegin(cmd, "World/VSMrender");
            VK::VSM::RenderAtlas(cmd);
            VK::Prof::ZoneEnd(cmd, zVSMr);
            // Temporal resolve: sample the atlas per screen pixel + reproject history →
            // the screen-space sun-shadow mask the receivers read. Also reads the
            // prepass depth in SHADER_READ.
            const int zVSMres = VK::Prof::ZoneBegin(cmd, "World/VSMresolve");
            depthToRead();
            VK::VSM::ResolveMask(cmd, ctx.depthView, ctx.extent, *ctx.viewProj);
            if (!fgCoalesce) depthToAttach();
            VK::Prof::ZoneEnd(cmd, zVSMres);
        }

        // Close the depth-read window: if any consumer left scene depth in
        // SHADER_READ (the coalesced path defers the flip-back to here), restore
        // DEPTH_ATTACHMENT once for the Hi-Z build + color pass — its src waits on
        // the union of every reader above. Skipped when depth is already attachment
        // (non-coalesced path restored it per consumer; all-disabled never opened
        // the window) → default r_fg_coalesce 0 emits the SAME barriers as the
        // original hand-placed code, no extras.
        if (depthState.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            depthToAttach();
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
    // Stage D compose: clone chunks' DEPTH — drawn HERE (after SSAO/VSM/Hi-Z have
    // consumed the home-only depth), so clones never pollute those home passes.
    // All visible clones' depth is laid first so the color loop below gets correct
    // inter-clone occlusion (a near clone occludes a far clone). Clones are far, so
    // their depth never overwrites a nearer home wall (LEQUAL) — home is untouched.
    // Zone opened every frame (profiler matches zones by open order).
    {
        const int zCompZ = VK::Prof::ZoneBegin(cmd, "World/ComposeZ");
        const u32 nClones = (gpuWorld && prepass) ? Compose::CloneCount() : 0;
        CFrustum composeFr;
        if (nClones) { Fmatrix fm = *ctx.viewProj; composeFr.CreateFromMatrix(fm, FRUSTUM_P_LRTB | FRUSTUM_P_FAR); }
        u32 nDrawn = 0;
        for (u32 c = 0; c < nClones; ++c) {
            if (!Compose::ChunkInView(c, composeFr)) continue;
            ++nDrawn;
            Fmatrix cvp; Fvector camL;
            Compose::ChunkView(c, *ctx.viewProj, cvp, camL);
            WorldGPU::Cull(cmd, cvp, camL, Device.vCameraDirection);

            VkRenderingAttachmentInfo cdAtt{};
            cdAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            cdAtt.imageView   = ctx.depthView;
            cdAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            cdAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;    // append to the home depth
            cdAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
            VkRenderingInfo cri{};
            cri.sType             = VK_STRUCTURE_TYPE_RENDERING_INFO;
            cri.renderArea.extent = ctx.extent;
            cri.layerCount        = 1;
            cri.pDepthAttachment  = &cdAtt;
            vkCmdBeginRendering(cmd, &cri);
            VkViewport cvpv{ 0.f, (float)ctx.extent.height, (float)ctx.extent.width, -(float)ctx.extent.height, 0.f, 1.f };
            vkCmdSetViewport(cmd, 0, 1, &cvpv);
            VkRect2D csc{ {}, ctx.extent };
            vkCmdSetScissor(cmd, 0, 1, &csc);
            vkCmdSetDepthBias(cmd, 0.f, 0.f, 0.f);
            WorldGPU::DrawDepth(cmd, cvp, true /*displaceTerrain*/);
            vkCmdEndRendering(cmd);
        }
        if (nDrawn) SceneAttachmentBarrier(cmd);   // order clone depth writes before the color pass reads
        if (nClones) {
            static u32 s_lastLog = 0;
            if (Device.dwTimeGlobal > s_lastLog + 3000) { s_lastLog = Device.dwTimeGlobal;
                Msg("[VK Compose] %u/%u clones in view", nDrawn, nClones); }
        }
        VK::Prof::ZoneEnd(cmd, zCompZ);
    }

    // The clone depth loop above left the LAST clone's set in the frustum indirect
    // buffer (cmds1). When the color pass draws the frustum set (no Hi-Z cull this
    // frame), re-cull HOME first or its color would draw the clone's indirect set
    // with the home matrix. With worldOccluded the home color reads the untouched
    // Hi-Z set (cmds2) — no re-cull needed.
    if (prepass && gpuWorld && !worldOccluded && Compose::CloneCount())
        WorldGPU::Cull(cmd, *ctx.viewProj);

    const int zColor = VK::Prof::ZoneBegin(cmd, "World/Color");
    // VRS diag: FS-invocation count for THIS pass ([VK VRS] world-color FS
    // invocations). r_vrs_force 1 = stats only (rate stays 1x1) for a baseline.
    if (VulkanHW.m_bVRSSupported && (ps_r_vrs > 0 || ps_r_vrs_force > 0))
        VK::VRS::StatsBegin(cmd, CommandManager.GetCurrentFrame());
    vkCmdBeginRendering(cmd, &ri);
    // VRS: no vkCmdSetFragmentShadingRateKHR here — world pipelines carry a STATIC
    // {1x1, KEEP, REPLACE} FSR state (SRI attachment wins when bound). A dynamic
    // rate proved unusable: any bind of a non-FSR-dynamic pipeline invalidates it
    // (validation 18-07-2026), which is why r_vrs looked like a no-op for so long.

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
    // Cluster-LOD debug overlay (r_cluster_debug): the DAG cut made visible —
    // clusters as flat colors (1) or wireframe (2), same indirect set as above.
    if (gpuWorld && ps_r_cluster_debug) WorldGPU::DrawDebug(cmd, *ctx.viewProj, ps_r_cluster_debug, worldOccluded);
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
    VK::VRS::StatsEnd(cmd, CommandManager.GetCurrentFrame());   // no-op if StatsBegin didn't run
    VK::Prof::ZoneEnd(cmd, zColor);

    // Inc 1: a few frames after load — once the dry/summer world + terrain uber-FS
    // variants have been lazily built — pre-create their WET/SNOW weather flips so the
    // first rain/snowfall doesn't stall compiling pipelines mid-gameplay. Spread ONE
    // create per frame (cold uber-FS compile is ~300ms — doing all 21 at once froze a
    // whole frame for ~7.8s on the first run).
    {
        static u32  s_prewarmCd   = 8;
        static bool s_prewarmDone = false;
        if (!s_prewarmDone) {
            if (s_prewarmCd) --s_prewarmCd;                                          // let the dry set build first
            else if (!PipelineCache::PrewarmWeatherVariants(1)) s_prewarmDone = true; // 1 pipeline/frame
        }
    }

    // Stage D compose: clone chunks' COLOR. Own cull → LOAD color+depth pass per
    // clone (cull can't run inside a rendering scope; the single indirect buffer
    // is reused sequentially — home already consumed its Hi-Z set above). Opaque
    // draws are z-buffered so ordering vs sky/trees/grass is irrelevant; the sky
    // pass only paints z==1 pixels and the clones' depth landed in the prepass.
    // Frustum-only cull (no Hi-Z: the pyramid slot was consumed by home's
    // CullColor; clone over-draw is acceptable in this slice). Zone opened every
    // frame (profiler matches zones by open order).
    {
        const int zCompC = VK::Prof::ZoneBegin(cmd, "World/ComposeC");
        const u32 nClones = (gpuWorld && prepass) ? Compose::CloneCount() : 0;
        CFrustum composeFr;
        if (nClones) {
            Fmatrix fm = *ctx.viewProj;
            composeFr.CreateFromMatrix(fm, FRUSTUM_P_LRTB | FRUSTUM_P_FAR);
        }
        for (u32 c = 0; c < nClones; ++c) {
            if (!Compose::ChunkInView(c, composeFr)) continue;
            Fmatrix cvp; Fvector camL;
            Compose::ChunkView(c, *ctx.viewProj, cvp, camL);
            WorldGPU::Cull(cmd, cvp, camL, Device.vCameraDirection);

            dAtt.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;     // never re-clear over the home scene
            vkCmdBeginRendering(cmd, &ri);                // same attachments (cAtt LOAD / dAtt LOAD / VRS static state)
            vkCmdSetViewport(cmd, 0, 1, &vp);
            vkCmdSetScissor(cmd, 0, 1, &sc);
            WorldGPU::DrawColor(cmd, cvp, EnvLight::GetCurrentSet(), false /*frustum set*/);
            vkCmdEndRendering(cmd);
        }
        VK::Prof::ZoneEnd(cmd, zCompC);
    }
    // No exit transition: the image stays in COLOR_ATTACHMENT for the next pass.
    // ExecutePasses inserts the inter-pass barrier; End brings it to PRESENT.
}

// PHASE B (editor-on-Vulkan, chunk 2): draw the add_Visual'd model(s) with no level.
// Pass_World owns the dynamics draw normally, but returns early with no level
// (!b_loaded), so nothing add_Visual'd this frame is rendered. Replicate JUST the
// color-pass dynamics block here (rigid RenderQueue + Pass_Skinned), lit by the
// editor-driven environment. Runs into the HDR SceneColor that EditorClear seeded
// (grey + depth=1.0), so the lit model composites on the grey backdrop through the
// normal tonemap. Registered only in -vk_editor (guarded by VKEditor::Active()).
void Pass_EditorDynamics(FrameContext& ctx)
{
    if (ctx.cmd == VK_NULL_HANDLE || !ctx.viewProj)   return;
    if (g_DynamicVisuals.empty() && g_HudVisuals.empty()) return;
    if (PipelineCache::GetLayout() == VK_NULL_HANDLE) return;

    VkCommandBuffer cmd = ctx.cmd;

    // Depth prepass + GTAO. Both live inside Pass_World, which returns on its first line
    // without a level, so the editor had no ambient occlusion at all — nothing darkened
    // contacts, and objects read as pasted onto surfaces rather than resting on them.
    // Everything the two need — extent, depth target, geometry — is already here.
    const bool prepass = PipelineCache::GetDepthLayout() != VK_NULL_HANDLE
                      && Swapchain.m_DepthFormat == VK_FORMAT_D32_SFLOAT;

    // Ahead of EnvLight::Update, which binds the AO view at binding 8: a target created
    // after that bind would be sampled while still UNDEFINED.
    if (prepass)
        SSAOPass::EnsureTargets(ctx.extent);

    // Per-frame sun/hemi/ambient UBO (set 1). Pass_World normally calls this; it is
    // skipped with no level, so drive it here. Runs BEFORE the render pass begins —
    // it may record uploads/barriers, which must be outside dynamic rendering.
    EnvLight::Update(CommandManager.GetCurrentFrame());

    // Volumetric fog (froxel inject + integrate). Its only other call site is inside
    // Pass_World, so with no level the volume was never integrated and held
    // transmittance 0 — which is why the tonemap composite had to be force-disabled
    // in editor. Driving it here restores aerial perspective / distance haze, the
    // single biggest "why doesn't this look like the game" term outdoors. Compute,
    // needs no scene depth (froxel world pos is analytic), so it belongs here,
    // before the prepass and outside any render pass — same placement as Pass_World.
    if (VK::Vol::Wanted() && VK::Vol::Ready()) {
        const VK::ProjTerms vpt = VK::DeriveProjTerms(*ctx.viewProj);
        // No smoke particles in an editor scene — the host pushes static models only.
        VK::Vol::Execute(cmd, vpt, CommandManager.GetCurrentFrame(), nullptr, 0);
    }

    // Pass_World owns this at the top of every frame, and it is unreachable here, so
    // the glass list was never reset in the editor: panes from deleted or replaced
    // host objects kept drawing until some other visual displaced them.
    g_RenderQueue.ClearGlass();

    // Instanced host scene (vk_instance_gpu): compute-cull against the camera into
    // the TGT_COLOR region. MUST precede BeginRendering — it is compute. Also does
    // the lazy (re)build, so ColorReady() below reflects THIS frame.
    InstanceGPU::CullColor(cmd, *ctx.viewProj);
    // Whatever the instanced path owns is skipped here — that CPU walk (a Submit
    // per object plus a sort, every frame) was the dominant cost of this pass, and
    // leaving an owned visual in would draw it twice. Ownership is whole-visual, so
    // the handful carrying late-translucent leaves still comes through and reaches
    // the blended pass intact. Skeletons go to Pass_Skinned, which self-collects.
    const bool instColor = InstanceGPU::ColorReady();

    // Collect once — the same queue feeds the depth prepass and the colour pass.
    g_RenderQueue.Clear();
    if (!g_DynamicVisuals.empty()) {
        for (const DynVisual& d : g_DynamicVisuals) {
            if (d.vis && !(instColor && InstanceGPU::OwnsVisual(d.vis))) {
                g_RenderQueue.SetSubmitHemi(d.hemi);
                d.vis->Submit(g_RenderQueue, d.xform, 0.0f);
            }
        }
        g_RenderQueue.SetSubmitHemi(1.0f);
        g_RenderQueue.SortByKey();
    }

    if (prepass && (g_RenderQueue.Size() || instColor))
    {
        const int zDepth = VK::Prof::ZoneBegin(cmd, "EditorDepth");

        VkRenderingAttachmentInfo pdAtt{};
        pdAtt.sType                   = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        pdAtt.imageView               = ctx.depthView;
        pdAtt.imageLayout             = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        pdAtt.loadOp                  = VK_ATTACHMENT_LOAD_OP_CLEAR;
        pdAtt.storeOp                 = VK_ATTACHMENT_STORE_OP_STORE;   // the colour pass LOADs it
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
        vkCmdSetDepthBias(cmd, 0.f, 0.f, 0.f);

        g_RenderQueue.FlushDepth(cmd, *ctx.viewProj);
        InstanceGPU::DrawColorDepth(cmd, *ctx.viewProj);   // indirect, opaque host instances
        Skinned_RenderDepthPrepass(cmd, *ctx.viewProj);

        vkCmdEndRendering(cmd);
        VK::Prof::ZoneEnd(cmd, zDepth);

        if (SSAOPass::Enabled())
        {
            VK::ImageState& depthState = VK::g_FrameGraph.Depth();
            depthState.Require(cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            SSAOPass::Execute(cmd, ctx.extent);
            depthState.Require(cmd, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT);
        }
    }

    // No own profiler zone — ExecutePasses already wraps this pass in a zone named
    // "EditorDynamics" (a nested same-name zone printed a confusing double entry).
    BeginOverlayRendering(cmd, ctx, VK_ATTACHMENT_STORE_OP_STORE);

    // Constant across the pass: MVP = viewProj (per-item model xform applied by
    // Flush), unit UV scale, no global alpha ref. Same push Pass_World sets.
    WorldPush push{};
    push.mvp         = *ctx.viewProj;
    push.uvScale[0]  = 1.0f / 1024.0f;
    push.uvScale[1]  = 1.0f / 1024.0f;
    push.alphaRef    = -1.0f;
    push.detailScale = 0.0f;
    vkCmdPushConstants(cmd, PipelineCache::GetLayout(), PipelineCache::GetPushStages(),
                       0, sizeof(WorldPush), &push);

    // Rigid dynamics — the queue was collected above, before the depth prepass.
    if (g_RenderQueue.Size()) {
        g_RenderQueue.SetAllowTess(false);   // dynamics use model-space vWorldPos → no tess
        g_RenderQueue.Flush(ctx);
        g_RenderQueue.SetAllowTess(true);
    }

    // Instanced host scene: one indirect draw per material group instead of one
    // per object. Alpha-tested groups ride along — the world FS applies alphaRef
    // from the material tail, so colour needs no opaque/AT split.
    InstanceGPU::DrawColor(cmd, *ctx.viewProj, EnvLight::GetCurrentSet());

    // Skinned dynamics (skeleton devices / NPCs) — self-collects from g_DynamicVisuals.
    Pass_Skinned(ctx);

    vkCmdEndRendering(cmd);
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
