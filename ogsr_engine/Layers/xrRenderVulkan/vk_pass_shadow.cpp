// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan — sun shadow caster pass. See vk_pass_shadow.h.
#include "stdafx.h"
#include "vk_pass_shadow.h"
#include "vk_shadow.h"
#include "vk_render_queue.h"               // RenderQueue (local caster queue)
#include "vk_water_sim.h"                  // WaterSim::Dispatch (shallow-water flow sim)
#include "vk_pipeline_cache.h"             // depth pipelines/layout
#include "vk_barriers.h"                   // ImageBarrier
#include "vk_pass_skinned.h"               // Skinned_UploadBones / Skinned_RenderShadow
#include "vk_light.h"                      // Lights::CollectFrame (shadowed spot/point picks)
#include "vk_TreeManager.h"                // Trees->RenderDepth (leafy crown casters)
#include "CRender_Vulkan.h"                // RImplementation.Visuals / b_loaded
#include "vk_Visual.h"                     // vkRender_Visual::Submit
#include "HW_Vulkan.h"
#include "../../xr_3da/IGame_Persistent.h" // g_pGamePersistent->Environment()
#include "../../xr_3da/Environment.h"      // CEnvDescriptorMixer (sun_dir)
#include "../../xr_3da/device.h"           // Device.vCameraPosition (cache check)

// GLOBAL scope (NOT inside namespace VK — a block-scope extern there would
// mangle as VK::ps_r_rain_enable → LNK2001). r_rain master off skips the map.
extern int ps_r_rain_enable;
extern int ps_r_water_sim;   // gate the ground-height map render (only the sim uses it)

namespace VK {

namespace {
    RenderQueue s_ShadowQueue;            // static caster queue (rebuilt on redraw)
    bool        s_firstUse = true;        // static map starts UNDEFINED, TRANSFER_SRC thereafter
    bool        s_combinedFirst = true;   // combined map starts UNDEFINED, SHADER_READ thereafter
    bool        s_spotFirst     = true;   // spot map: UNDEFINED on first use
    bool        s_pointFirst    = true;   // point cube: UNDEFINED on first use

    // Cached static-caster queues for the shadowed dynamic lights. Rebuilding
    // them means walking ALL level visuals — only do that when the light sphere
    // actually changed (flashlight moved / another fire selected / level load),
    // never per frame.
    RenderQueue s_SpotQueue;
    bool        s_spotQValid = false;
    Fvector     s_spotQPos{};  float s_spotQRange = 0.f;  size_t s_spotQVis = 0;
    bool        s_pointHadSkinned = false;   // cube contains NPC shadows from last render

    // Statics with huge bounding spheres (terrain chunks, giant merges) are
    // receivers, not meaningful blockers, for a hand-held light — and they cost
    // the most raster. Keep fences/props/buildings, drop the monsters.
    constexpr float kSpotCasterMaxR = 30.f;

    // Depth bias to fight self-shadow acne (tunable). Constant + slope-scaled.
    constexpr float kBiasConst = 1.5f;
    constexpr float kBiasSlope = 2.5f;

    // Static-map cache: static casters + the box tracking the camera only go
    // stale when the camera or the sun actually moves. Redraw when the camera
    // drifted > kRedrawDist from the last render, the sun rotated more than
    // ~0.1° (cos threshold), or the visual set changed (level load/unload).
    // Dynamic (skinned) casters are NOT cached — they render every frame into
    // the combined map on top of a copy of this one.
    constexpr float kRedrawDist   = 10.f;
    constexpr float kSunDotRedraw = 0.99999847f;  // cos(0.1°)
    bool    s_cacheValid   = false;
    Fvector s_lastCamPos;
    Fvector s_lastSunDir;
    size_t  s_lastVisCount = 0;

    // Near sun cascades (R4 scheme): the MAPS are re-rendered every frame with
    // the continuous sun (stability comes from the R4 world-anchored texel
    // alignment in ComputeCascadeVP, exactly like xrRenderPC_R4), but each
    // static-caster QUEUE is cached — rebuilding one walks all level visuals.
    // Casters are collected with kCascQueueInflate of slack so a queue stays
    // correct while the camera drifts < kCascQueueDist and the sun creeps.
    RenderQueue s_CascQueue[ShadowMap::kNumSunCascades];
    bool        s_cascQValid[ShadowMap::kNumSunCascades] = {};
    bool        s_cascFirst = true;           // cascade maps start UNDEFINED
    Fvector     s_cascQCamPos[ShadowMap::kNumSunCascades] = {};
    Fvector     s_cascQSunDir[ShadowMap::kNumSunCascades] = {};
    size_t      s_cascQVis[ShadowMap::kNumSunCascades] = {};
    constexpr float kCascQueueDist    = 4.f;
    constexpr float kCascQueueInflate = 8.f;
    constexpr float kCascQueueSunDot  = 0.9999863f;   // cos(0.3°)

    // Sun-direction low-pass: R4 feeds the raw env sun in, but our env mixer
    // showed frame-to-frame wobble earlier in this port — a 1 s exponential
    // filter eats it for free and lags the real motion imperceptibly.
    constexpr float kCascSunTau = 1.0f;
    Fvector s_cascSunDir{};
    bool    s_cascSunInit = false;

    // Transient sun-direction rejection. A lightning flash (thunderbolt.cpp)
    // OVERWRITES CurrentEnv->sun_dir with the bolt direction for ~1 s, so the
    // shadow sun would swing to the bolt and (via the low-pass) slowly rotate
    // back — the "shadows open up and rotate back" bug. The real sun moves
    // <0.01°/frame, so a jump > ~2° in one frame is a transient: hold the last
    // stable direction through it. A jump that PERSISTS (level load / scripted
    // time skip) is accepted after kSunAcceptHold frames so we don't get stuck.
    constexpr float kSunJumpDot    = 0.99939f;   // cos(2°): below → treat as a jump
    constexpr u32   kSunAcceptHold = 90;         // ~1.5 s — longer than a lightning flash
    Fvector s_stableSunDir{};
    bool    s_stableSunInit = false;
    u32     s_sunHoldFrames = 0;

    // Rain occlusion map cache: statics + trees from straight above, redrawn
    // when the camera moved far enough or the level changed. Rendered only
    // while it's raining (or surfaces are still drying) — but at least once,
    // so the sampled image is never in UNDEFINED layout.
    RenderQueue s_RainQueue;
    bool        s_rainFirst  = true;
    bool        s_rainValid  = false;
    Fvector     s_rainCamPos{};
    size_t      s_rainVis    = 0;
    constexpr float kRainRedrawDist = 8.f;
}

void Pass_SunShadow(FrameContext& ctx)
{
    if (ctx.cmd == VK_NULL_HANDLE) return;
    if (!ShadowMap::Init())        return;        // image/sampler not available
    if (PipelineCache::GetDepthLayout() == VK_NULL_HANDLE) return;  // caster shader missing

    VkCommandBuffer cmd = ctx.cmd;

    // Env sun direction (normalized for the cache comparison below).
    Fvector sunDir; sunDir.set(0.f, -1.f, 0.f);
    if (g_pGamePersistent)
        if (auto* E = g_pGamePersistent->Environment().CurrentEnv) sunDir = E->sun_dir;
    if (sunDir.magnitude() < 1e-4f) sunDir.set(0.f, -1.f, 0.f);
    sunDir.normalize();

    // Fresh level: the sun-smoothing statics persist across loads, so a new
    // level's different sun would be HELD as a "transient" and swing into place
    // ~1.5 s later (and the cascade low-pass would lerp from the old level's
    // direction). Detect the load transition and snap the smoothing state so the
    // first frame adopts this level's sun directly.
    {
        const bool nowLoaded = RImplementation.b_loaded && !RImplementation.Visuals.empty();
        static bool s_wasLoaded = false;
        if (nowLoaded && !s_wasLoaded) {
            s_stableSunInit = false;
            s_cascSunInit   = false;
            s_sunHoldFrames = 0;
        }
        s_wasLoaded = nowLoaded;
    }

    // Reject lightning-flash transients (see kSunJumpDot): hold the last stable
    // direction through a brief big jump; accept it only if it persists. Used by
    // BOTH the far map cache check and the cascades below, so neither swings.
    if (!s_stableSunInit) {
        s_stableSunDir = sunDir; s_stableSunInit = true;
    } else if (s_stableSunDir.dotproduct(sunDir) > kSunJumpDot) {
        s_stableSunDir = sunDir; s_sunHoldFrames = 0;     // small delta → real sun motion
    } else if (++s_sunHoldFrames > kSunAcceptHold) {
        s_stableSunDir = sunDir; s_sunHoldFrames = 0;     // persisted → real change (level/skip)
    }
    sunDir = s_stableSunDir;

    const bool   loaded = RImplementation.b_loaded && !RImplementation.Visuals.empty();
    const size_t nVis   = loaded ? RImplementation.Visuals.size() : 0;

    // Bones for the skinned casters below (Pass_Skinned reuses the same upload).
    Skinned_UploadBones();

    const u32 sz = ShadowMap::Size();
    const VkViewport vpShadow{ 0.f, (float)sz, (float)sz, -(float)sz, 0.f, 1.f };
    const VkRect2D   scShadow{ {0,0}, { sz, sz } };

    // ---- STATIC map (cached): redraw only when the camera/sun/level moved. ----
    // While valid we keep the cached lightVP (ComputeLightVP is NOT re-run), so
    // the sampling matrix always matches the cached contents.
    const bool staticValid = s_cacheValid && !s_firstUse && nVis == s_lastVisCount
        && Device.vCameraPosition.distance_to_sqr(s_lastCamPos) < kRedrawDist * kRedrawDist
        && s_lastSunDir.dotproduct(sunDir) > kSunDotRedraw;

    if (!staticValid)
    {
        ShadowMap::ComputeLightVP(sunDir);
        s_cacheValid   = true;
        s_lastCamPos   = Device.vCameraPosition;
        s_lastSunDir   = sunDir;
        s_lastVisCount = nVis;

        // Collect static casters (only when a level is loaded; otherwise we still
        // clear+transition so the map is a valid "fully lit" texture for receivers).
        // Visuals whose bounding sphere misses the light ortho box can't write any
        // shadow texel — cull them here instead of pushing the whole level.
        s_ShadowQueue.Clear();
        if (loaded) {
            Fmatrix identity; identity.identity();
            for (IRenderVisual* iv : RImplementation.Visuals) {
                if (!iv) continue;
                auto* rv = static_cast<vkRender_Visual*>(iv);
                const Fsphere& bs = rv->vis.sphere;
                if (bs.R > 0.f && !ShadowMap::SphereVisible(bs.P, bs.R)) continue;
                rv->Submit(s_ShadowQueue, identity, 0.0f);
            }
            s_ShadowQueue.SortByKey();
        }

        // TRANSFER_SRC (or UNDEFINED on first use) → DEPTH_ATTACHMENT for writing.
        const VkImageLayout oldLayout = s_firstUse ? VK_IMAGE_LAYOUT_UNDEFINED
                                                   : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        s_firstUse = false;
        ImageBarrier(cmd, ShadowMap::GetStaticImage(), oldLayout,
                     VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);

        VkRenderingAttachmentInfo dAtt{};
        dAtt.sType                   = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        dAtt.imageView               = ShadowMap::GetStaticView();
        dAtt.imageLayout             = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        dAtt.loadOp                  = VK_ATTACHMENT_LOAD_OP_CLEAR;
        dAtt.storeOp                 = VK_ATTACHMENT_STORE_OP_STORE;
        dAtt.clearValue.depthStencil = { 1.0f, 0 };

        VkRenderingInfo ri{};
        ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
        ri.renderArea.extent    = { sz, sz };
        ri.layerCount           = 1;
        ri.colorAttachmentCount = 0;
        ri.pDepthAttachment     = &dAtt;
        vkCmdBeginRendering(cmd, &ri);

        // Negative-height viewport (same X-Ray D3D→Vulkan Y-flip as the scene passes,
        // so the stored depth matches the camera-side sampling convention).
        vkCmdSetViewport(cmd, 0, 1, &vpShadow);
        vkCmdSetScissor(cmd, 0, 1, &scShadow);
        vkCmdSetDepthBias(cmd, kBiasConst, 0.0f, kBiasSlope);

        s_ShadowQueue.FlushDepth(cmd, ShadowMap::GetLightVP());

        // Trees (GPU-driven elsewhere — not in the queue): alpha-tested leafy
        // crowns into the sun map, CPU-culled by the light box. Redraw-only cost.
        if (RImplementation.Trees && RImplementation.Trees->IsBuilt())
            RImplementation.Trees->RenderDepth(cmd, ShadowMap::GetLightVP());

        vkCmdEndRendering(cmd);

        // DEPTH_ATTACHMENT → TRANSFER_SRC: the static map is only ever copied from.
        ImageBarrier(cmd, ShadowMap::GetStaticImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    // ---- COMBINED map (sampled): every frame = static copy + dynamic casters. ----
    ImageBarrier(cmd, ShadowMap::GetImage(),
                 s_combinedFirst ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
    s_combinedFirst = false;

    VkImageCopy copy{};
    copy.srcSubresource = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1 };
    copy.dstSubresource = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1 };
    copy.extent         = { sz, sz, 1 };
    vkCmdCopyImage(cmd, ShadowMap::GetStaticImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   ShadowMap::GetImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

    ImageBarrier(cmd, ShadowMap::GetImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);

    {
        VkRenderingAttachmentInfo dAtt{};
        dAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        dAtt.imageView   = ShadowMap::GetView();
        dAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        dAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;     // keep the static copy
        dAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

        VkRenderingInfo ri{};
        ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
        ri.renderArea.extent    = { sz, sz };
        ri.layerCount           = 1;
        ri.colorAttachmentCount = 0;
        ri.pDepthAttachment     = &dAtt;
        vkCmdBeginRendering(cmd, &ri);

        vkCmdSetViewport(cmd, 0, 1, &vpShadow);
        vkCmdSetScissor(cmd, 0, 1, &scShadow);
        vkCmdSetDepthBias(cmd, kBiasConst, 0.0f, kBiasSlope);

        Skinned_RenderShadow(cmd, ShadowMap::GetLightVP());

        vkCmdEndRendering(cmd);
    }

    // DEPTH_ATTACHMENT → SHADER_READ for the world/skinned receivers this frame.
    ImageBarrier(cmd, ShadowMap::GetImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);

    // ---- RAIN occlusion map: top-down statics+trees depth, cached. Receivers
    // multiply wetness by it — geometry overhead (roof/tunnel) keeps a surface
    // dry. Rendered on demand: only while rain_density or wetness_factor is
    // non-zero (plus one initial clear so the bound image has a defined layout).
    {
        float rainNeed = 0.f;
        if (g_pGamePersistent && ps_r_rain_enable) {
            const auto& env = g_pGamePersistent->Environment();
            rainNeed = env.wetness_factor;
            if (env.CurrentEnv) rainNeed = _max(rainNeed, env.CurrentEnv->rain_density);
        }
        const bool wantRain = rainNeed > 0.001f;
        const bool stale = !s_rainValid || nVis != s_rainVis
            || Device.vCameraPosition.distance_to_sqr(s_rainCamPos) > kRainRedrawDist * kRainRedrawDist;

        if (s_rainFirst || (wantRain && stale))
        {
            ShadowMap::ComputeRainVP();

            s_RainQueue.Clear();
            if (loaded && wantRain) {
                Fmatrix identity; identity.identity();
                for (IRenderVisual* iv : RImplementation.Visuals) {
                    if (!iv) continue;
                    auto* rv = static_cast<vkRender_Visual*>(iv);
                    const Fsphere& bs = rv->vis.sphere;
                    if (bs.R > 0.f && !ShadowMap::RainSphereVisible(bs.P, bs.R)) continue;
                    rv->Submit(s_RainQueue, identity, 0.0f);
                }
                s_RainQueue.SortByKey();
            }
            s_rainValid  = wantRain;     // a pure-clear first frame stays "stale" until rain starts
            s_rainCamPos = Device.vCameraPosition;
            s_rainVis    = nVis;

            ImageBarrier(cmd, ShadowMap::GetRainImage(),
                         s_rainFirst ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
            s_rainFirst = false;

            const u32 rsz = ShadowMap::RainSize();
            VkRenderingAttachmentInfo dAtt{};
            dAtt.sType                   = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            dAtt.imageView               = ShadowMap::GetRainView();
            dAtt.imageLayout             = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            dAtt.loadOp                  = VK_ATTACHMENT_LOAD_OP_CLEAR;
            dAtt.storeOp                 = VK_ATTACHMENT_STORE_OP_STORE;
            dAtt.clearValue.depthStencil = { 1.0f, 0 };

            VkRenderingInfo ri{};
            ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
            ri.renderArea.extent    = { rsz, rsz };
            ri.layerCount           = 1;
            ri.colorAttachmentCount = 0;
            ri.pDepthAttachment     = &dAtt;
            vkCmdBeginRendering(cmd, &ri);

            const VkViewport vpR{ 0.f, (float)rsz, (float)rsz, -(float)rsz, 0.f, 1.f };
            const VkRect2D   scR{ {0,0}, { rsz, rsz } };
            vkCmdSetViewport(cmd, 0, 1, &vpR);
            vkCmdSetScissor(cmd, 0, 1, &scR);
            vkCmdSetDepthBias(cmd, kBiasConst, 0.0f, kBiasSlope);

            s_RainQueue.FlushDepth(cmd, ShadowMap::GetRainVP());
            if (wantRain && RImplementation.Trees && RImplementation.Trees->IsBuilt())
                RImplementation.Trees->RenderDepth(cmd, ShadowMap::GetRainVP());

            vkCmdEndRendering(cmd);

            ImageBarrier(cmd, ShadowMap::GetRainImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);

            // Clean GROUND-height map for the water sim (only when the sim runs).
            static bool s_groundFirst = true;
            if (ps_r_water_sim) {
            ImageBarrier(cmd, ShadowMap::GetGroundImage(),
                         s_groundFirst ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
            s_groundFirst = false;
            {
                VkRenderingAttachmentInfo gAtt{};
                gAtt.sType                   = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                gAtt.imageView               = ShadowMap::GetGroundView();
                gAtt.imageLayout             = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
                gAtt.loadOp                  = VK_ATTACHMENT_LOAD_OP_CLEAR;
                gAtt.storeOp                 = VK_ATTACHMENT_STORE_OP_STORE;
                gAtt.clearValue.depthStencil = { 1.0f, 0 };
                VkRenderingInfo gri{};
                gri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
                gri.renderArea.extent    = { rsz, rsz };
                gri.layerCount           = 1;
                gri.colorAttachmentCount = 0;
                gri.pDepthAttachment     = &gAtt;
                vkCmdBeginRendering(cmd, &gri);
                vkCmdSetViewport(cmd, 0, 1, &vpR);
                vkCmdSetScissor(cmd, 0, 1, &scR);
                vkCmdSetDepthBias(cmd, kBiasConst, 0.0f, kBiasSlope);
                s_RainQueue.FlushDepth(cmd, ShadowMap::GetRainVP());   // statics+terrain, NO trees
                vkCmdEndRendering(cmd);
            }
            ImageBarrier(cmd, ShadowMap::GetGroundImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
            } // ps_r_water_sim

            static bool s_diag = false;
            if (!s_diag && wantRain) { s_diag = true;
                Msg("[VK Rain] occlusion map drawn (%u items, need %.2f)", s_RainQueue.Size(), rainNeed); }
        }

        // ---- Water flow sim: advance the shallow-water field on the rain map
        // (top-down ground height = a height field). Drives geometric puddles +
        // the volumetric water render. Runs every frame while rain is enabled
        // (water keeps flowing and drains via evaporation after the rain stops).
        if (ps_r_rain_enable) {
            float rd = 0.f;
            if (g_pGamePersistent) {
                const auto& env = g_pGamePersistent->Environment();
                if (env.CurrentEnv) rd = env.CurrentEnv->rain_density;
            }
            VK::WaterSim::Dispatch(cmd, rd);
        }
    }

    // ---- NEAR cascades (R4 port): both re-rendered EVERY frame with the
    // continuous (low-passed) sun. Stability against the per-frame sun creep
    // comes from the world-anchored texel alignment inside ComputeCascadeVP.
    {
        if (!s_cascSunInit) {
            s_cascSunDir  = sunDir;
            s_cascSunInit = true;
        } else {
            const float k = 1.f - expf(-Device.fTimeDelta / kCascSunTau);
            s_cascSunDir.lerp(s_cascSunDir, sunDir, k);
            if (s_cascSunDir.magnitude() > 1e-4f) s_cascSunDir.normalize();
            else                                  s_cascSunDir = sunDir;
        }

        for (u32 ci = 0; ci < ShadowMap::kNumSunCascades; ++ci)
        {
            // Cascade 1 re-renders every OTHER frame (30 Hz shadow update is
            // invisible at 12–30 m) — halves its raster cost. Its VP is only
            // recomputed when it actually renders, so sampling always matches
            // the cached contents. Cascade 0 stays per-frame.
            if (ci == 1 && !s_cascFirst && (Device.dwFrame & 1))
                continue;

            ShadowMap::ComputeCascadeVP(ci, s_cascSunDir);

            // Cached caster queue: the visuals walk is the expensive part —
            // only redo it when the camera/sun drifted past the slack the
            // queue was collected with (kCascQueueInflate).
            const bool qValid = s_cascQValid[ci] && nVis == s_cascQVis[ci]
                && Device.vCameraPosition.distance_to_sqr(s_cascQCamPos[ci]) < kCascQueueDist * kCascQueueDist
                && s_cascQSunDir[ci].dotproduct(sunDir) > kCascQueueSunDot;
            if (!qValid)
            {
                s_cascQValid[ci]  = true;
                s_cascQCamPos[ci] = Device.vCameraPosition;
                s_cascQSunDir[ci] = sunDir;
                s_cascQVis[ci]    = nVis;
                s_CascQueue[ci].Clear();
                if (loaded) {
                    Fmatrix identity; identity.identity();
                    for (IRenderVisual* iv : RImplementation.Visuals) {
                        if (!iv) continue;
                        auto* rv = static_cast<vkRender_Visual*>(iv);
                        const Fsphere& bs = rv->vis.sphere;
                        if (bs.R > 0.f && !ShadowMap::CascadeSphereVisible(ci, bs.P, bs.R, kCascQueueInflate)) continue;
                        rv->Submit(s_CascQueue[ci], identity, 0.0f);
                    }
                    s_CascQueue[ci].SortByKey();
                }
            }

            ImageBarrier(cmd, ShadowMap::GetCascadeImage(ci),
                         s_cascFirst ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);

            const u32 nsz = ShadowMap::CascadeSize(ci);
            VkRenderingAttachmentInfo dAtt{};
            dAtt.sType                   = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            dAtt.imageView               = ShadowMap::GetCascadeView(ci);
            dAtt.imageLayout             = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            dAtt.loadOp                  = VK_ATTACHMENT_LOAD_OP_CLEAR;
            dAtt.storeOp                 = VK_ATTACHMENT_STORE_OP_STORE;
            dAtt.clearValue.depthStencil = { 1.0f, 0 };

            VkRenderingInfo ri{};
            ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
            ri.renderArea.extent    = { nsz, nsz };
            ri.layerCount           = 1;
            ri.colorAttachmentCount = 0;
            ri.pDepthAttachment     = &dAtt;
            vkCmdBeginRendering(cmd, &ri);

            const VkViewport vpC{ 0.f, (float)nsz, (float)nsz, -(float)nsz, 0.f, 1.f };
            const VkRect2D   scC{ {0,0}, { nsz, nsz } };
            vkCmdSetViewport(cmd, 0, 1, &vpC);
            vkCmdSetScissor(cmd, 0, 1, &scC);
            vkCmdSetDepthBias(cmd, kBiasConst, 0.0f, kBiasSlope);

            s_CascQueue[ci].FlushDepth(cmd, ShadowMap::GetCascadeVP(ci));
            if (RImplementation.Trees && RImplementation.Trees->IsBuilt())
                RImplementation.Trees->RenderDepth(cmd, ShadowMap::GetCascadeVP(ci), (s32)ci);
            Skinned_RenderShadow(cmd, ShadowMap::GetCascadeVP(ci));

            vkCmdEndRendering(cmd);

            ImageBarrier(cmd, ShadowMap::GetCascadeImage(ci), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
        }
        s_cascFirst = false;
    }

    // ===== Dynamic light shadows (STEP 3b): 1 spot + 1 point cube per frame =====
    // EnvLight::Update (Pass_World, later this frame) reads the same CollectFrame
    // result, so the gpu[] indices below match what the receivers see.
    const auto& FL = Lights::CollectFrame(Device.vCameraPosition);
    const bool haveSpot  = FL.spotIdx  >= 0;
    const bool havePoint = FL.pointIdx >= 0;

    // Nothing selected and both maps already initialized → zero cost.
    if (!haveSpot && !havePoint && !s_spotFirst && !s_pointFirst)
        return;

    // Rebuild a cached static-caster queue if the light sphere changed (full
    // visuals walk — the expensive part we must NOT do per frame). Oversized
    // visuals (terrain chunks) are excluded — see kSpotCasterMaxR.
    auto refreshStaticQueue = [&](RenderQueue& q, bool& valid, Fvector& qPos, float& qRange,
                                  size_t& qVis, const Fvector& p, float range) -> bool {
        if (valid && qVis == nVis && qRange == range
            && qPos.distance_to_sqr(p) < 1.0f)   // < 1m drift keeps the cache
            return false;
        q.Clear();
        if (loaded) {
            Fmatrix identity; identity.identity();
            for (IRenderVisual* iv : RImplementation.Visuals) {
                if (!iv) continue;
                auto* rv = static_cast<vkRender_Visual*>(iv);
                const Fsphere& bs = rv->vis.sphere;
                if (bs.R > kSpotCasterMaxR) continue;
                if (bs.R > 0.f) {
                    const float rr = range + bs.R;
                    if (p.distance_to_sqr(bs.P) > rr * rr) continue;
                }
                rv->Submit(q, identity, 0.0f);
            }
            q.SortByKey();
        }
        valid = true; qPos = p; qRange = range; qVis = nVis;
        return true;
    };

    // One depth render into a dynamic shadow target (spot map or a cube face).
    // statics == nullptr → skinned casters only (point cube: NPC shadows are the
    // point; statics there cost terrain×6 faces for near-zero visual gain).
    auto renderDepthTarget = [&](VkImageView view, u32 size, const Fmatrix& vp, RenderQueue* statics,
                                 const Fvector& lightPos, float lightRange, bool drawCasters) {
        VkRenderingAttachmentInfo dAtt{};
        dAtt.sType                   = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        dAtt.imageView               = view;
        dAtt.imageLayout             = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        dAtt.loadOp                  = VK_ATTACHMENT_LOAD_OP_CLEAR;
        dAtt.storeOp                 = VK_ATTACHMENT_STORE_OP_STORE;
        dAtt.clearValue.depthStencil = { 1.0f, 0 };

        VkRenderingInfo ri{};
        ri.sType             = VK_STRUCTURE_TYPE_RENDERING_INFO;
        ri.renderArea.extent = { size, size };
        ri.layerCount        = 1;
        ri.pDepthAttachment  = &dAtt;
        vkCmdBeginRendering(cmd, &ri);

        if (drawCasters) {
            VkViewport vp2{ 0.f, (float)size, (float)size, -(float)size, 0.f, 1.f };
            vkCmdSetViewport(cmd, 0, 1, &vp2);
            VkRect2D sc{ {0,0}, { size, size } };
            vkCmdSetScissor(cmd, 0, 1, &sc);
            vkCmdSetDepthBias(cmd, kBiasConst, 0.0f, kBiasSlope);
            if (statics) statics->FlushDepth(cmd, vp);
            Skinned_RenderShadow(cmd, vp, &lightPos, lightRange);
        }
        vkCmdEndRendering(cmd);
    };

    // --- SPOT (flashlight): re-rendered while selected (it moves with the player);
    // the static-caster queue only rebuilds when the player walked > 1m.
    if (haveSpot || s_spotFirst)
    {
        ImageBarrier(cmd, ShadowMap::GetSpotImage(),
                     s_spotFirst ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
        s_spotFirst = false;

        if (haveSpot) {
            ShadowMap::ComputeSpotVP(FL.spotPos, FL.spotDir, FL.spotRange, FL.spotCone);
            refreshStaticQueue(s_SpotQueue, s_spotQValid, s_spotQPos, s_spotQRange, s_spotQVis,
                               FL.spotPos, FL.spotRange);
        } else {
            s_spotQValid = false;
        }
        renderDepthTarget(ShadowMap::GetSpotView(), ShadowMap::SpotSize(),
                          ShadowMap::GetSpotVP(), &s_SpotQueue, FL.spotPos, FL.spotRange, haveSpot);

        ImageBarrier(cmd, ShadowMap::GetSpotImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    // --- POINT (campfire) cube: SKINNED casters only — the visual point is NPC
    // shadows around the fire; statics here cost terrain×6 faces for near-zero
    // gain. Re-render only while someone is (or just was) inside the radius.
    {
        const bool anySkinned  = havePoint && Skinned_AnyCasterInSphere(FL.pointPos, FL.pointRange);
        const bool renderPoint = havePoint && (anySkinned || s_pointHadSkinned);
        if (renderPoint || s_pointFirst)
        {
            ImageBarrier(cmd, ShadowMap::GetPointImage(),
                         s_pointFirst ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT, 6);
            s_pointFirst = false;

            for (u32 f = 0; f < 6; ++f) {
                const Fmatrix faceVP = ShadowMap::ComputePointFaceVP(FL.pointPos, FL.pointRange, f);
                renderDepthTarget(ShadowMap::GetPointFaceView(f), ShadowMap::PointSize(),
                                  faceVP, nullptr, FL.pointPos, FL.pointRange, renderPoint);
            }

            ImageBarrier(cmd, ShadowMap::GetPointImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT, 6);
        }
        s_pointHadSkinned = anySkinned;
    }
}

}  // namespace VK
