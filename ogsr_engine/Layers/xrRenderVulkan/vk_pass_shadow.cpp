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
#include "vk_deform.h"                     // Deform::Dispatch (snow deform press field)
#include "vk_pipeline_cache.h"             // depth pipelines/layout
#include "vk_barriers.h"                   // ImageBarrier
#include "vk_pass_skinned.h"               // Skinned_UploadBones / Skinned_RenderShadow
#include "vk_light.h"                      // Lights::CollectFrame (shadowed spot/point picks)
#include "vk_TreeManager.h"                // Trees->RenderDepth (leafy crown casters)
#include "vk_shadow_gpu.h"                 // ShadowGPU::Cull/Draw (GPU-driven opaque casters)
#include "vk_vsm.h"                        // VSM::MaskReady (skip the redundant sun cascade/far raster under VSM)
#include "vk_cull.h"                       // VK::ExtractFrustumPlanes (light frustum → cull planes)
#include "vk_profiler.h"                   // VK::Prof::ZoneBegin/ZoneEnd (per-target shadow GPU timing)
#include "CRender_Vulkan.h"                // RImplementation.Visuals / b_loaded
#include "vk_Visual.h"                     // vkRender_Visual::Submit
#include "HW_Vulkan.h"
#include "../../xr_3da/IGame_Persistent.h" // g_pGamePersistent->Environment()
#include "../../xr_3da/Environment.h"      // CEnvDescriptorMixer (sun_dir)
#include "../../xr_3da/device.h"           // Device.vCameraPosition (cache check)

// GLOBAL scope (NOT inside namespace VK — a block-scope extern there would
// mangle as VK::ps_r_rain_enable → LNK2001). r_rain master off skips the map.
extern int ps_r_rain_enable;
extern int   ps_r_snow_deform;        // snow footprint deformation enable
extern int   ps_r_snow_deform_tex;    // use the dense deform texture (vk_deform)
extern int   ps_r_snow_mesh;          // dense snow surface mesh (needs the ground-height map)
extern float ps_r_snow_deform_radius; // print radius (m)
extern float ps_r_snow;               // TARGET snow coverage (gate the deform dispatch)
extern int ps_r_water_sim;   // gate the ground-height map render (sim)
extern int ps_r_puddle_sss;  // gate the ground-height map render (SSS puddle real-dip placement)
extern int ps_r_light_occ;   // dynamic-light occlusion: render the ground-height map always
extern int ps_r_gpu_shadows; // GPU-driven opaque sun shadow casters (A/B with 0)
extern int   ps_r_shadow_lod;      // caster-LOD: distant casters draw coarse geometry (A/B with 0)
extern float ps_r_shadow_lod_dist; // metres from camera beyond which casters go coarse
extern int   ps_r_shadow_casc_cache; // cascade static-map cache (0 = re-raster every frame)
extern float ps_r_shadow_casc_sun;   // sun-rotation degrees that forces a cascade static redraw
extern int   ps_r_vsm;               // VSM on → its mask drives ALL sun receivers, so the cascade/far sun maps are redundant
extern int   ps_r_vol;               // froxel volumetrics: samples the cascade for froxel SUN occlusion → keep it rendered even under VSM
extern int   ps_r_vol_debug;
extern int   ps_r_vol_shadow;        // dedicated per-frame fog sun-shadow (continuous → no cache tick); 0 = cascade/VSM path

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
    RenderQueue s_PointQueue;                // static occluders for the point cube (incl. terrain)
    bool        s_pointQValid = false;
    Fvector     s_pointQPos{};  float s_pointQRange = 0.f;  size_t s_pointQVis = 0;
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

    // Fog sun-shadow (r_vol_shadow): its own cached caster queue (re-collected on
    // camera move); the depth is RE-RENDERED every frame with a fresh VP (continuous).
    RenderQueue s_FogQueue;
    bool        s_fogQValid = false;
    bool        s_fogFirst  = true;            // fog map starts UNDEFINED
    Fvector     s_fogQCamPos = {};
    size_t      s_fogQVis = 0;

    // Cascade STATIC-map cache (mirrors the far map's static/combined split): the
    // statics-only cascade depth is re-rastered ONLY when the camera moved past
    // kCascRedrawDist, the sun rotated past kCascQueueSunDot, or the level changed.
    // While valid the cascade VP is FROZEN (ComputeCascadeVP not re-run) so the
    // sampled matrix matches the cached contents; every frame the combined map is
    // copy(static)+skinned overlay. Standing/aiming/turning → ZERO static raster
    // (camera rotation doesn't move vCameraPosition). This is what kills the
    // ~3 ms/frame Shadow/Casc0 raster the profiler sub-zones pinned down.
    bool        s_cascStaticValid[ShadowMap::kNumSunCascades] = {};
    Fvector     s_cascStaticCamPos[ShadowMap::kNumSunCascades] = {};
    Fvector     s_cascStaticSunDir[ShadowMap::kNumSunCascades] = {};
    size_t      s_cascStaticVis[ShadowMap::kNumSunCascades] = {};
    bool        s_cascStaticFirst = true;   // static cascade images start UNDEFINED
    // Per-cascade redraw threshold (m). Cascade 0 (25 m box, 0.6 cm texels) tracks
    // the player tighter; cascade 1 (60 m, lower res) tolerates more drift. Small
    // enough that the near-field stays well inside the frozen box between redraws;
    // beyond it cascade 1 / the far map cover, so a stale edge never drops shadows.
    constexpr float kCascRedrawDist[ShadowMap::kNumSunCascades] = { 2.0f, 4.0f };

    // Sun-direction low-pass: R4 feeds the raw env sun in, but our env mixer
    // showed frame-to-frame wobble earlier in this port — a 1 s exponential
    // filter eats it for free and lags the real motion imperceptibly.
    constexpr float kCascSunTau = 1.0f;
    // A one-frame change bigger than this (with no bolt corrupting sunDir — see
    // below) is a discontinuity (level/save load, scripted time-skip): the
    // cascade low-pass SNAPS to it instead of slewing, so shadows don't visibly
    // rotate into place over ~1 s after a load.
    constexpr float kCascSnapDot = 0.99939f;     // cos(2°)
    Fvector s_cascSunDir{};
    bool    s_cascSunInit = false;

    // Lightning rejection. A flash (thunderbolt.cpp OnFrame) OVERWRITES
    // CurrentEnv->sun_dir with the bolt direction for the WHOLE strike, so the
    // shadow sun would swing to the bolt and back — the "cascades shift then roll
    // back" bug. We hold the last real direction while a bolt is active (queried
    // directly via CEnvironment::IsThunderboltActive — robust at any framerate /
    // strike length, unlike the old frame-counter hold which released mid-strike
    // on high-refresh displays).
    Fvector s_stableSunDir{};
    bool    s_stableSunInit = false;

    // Rain occlusion map cache: statics + trees from straight above, redrawn
    // when the camera moved far enough or the level changed. Rendered only
    // while it's raining (or surfaces are still drying) — but at least once,
    // so the sampled image is never in UNDEFINED layout.
    RenderQueue s_RainQueue;
    bool        s_rainFirst  = true;
    bool        s_rainValid  = false;
    Fvector     s_rainCamPos{};
    size_t      s_rainVis    = 0;
    // Rain occlusion map redraws on camera move (nVis = total loaded visuals,
    // only changes on stream/spawn — NOT per frame). Profiling showed the redraw
    // (19795 casters + all trees into 1024²) is the rain GPU/CPU SPIKE while
    // walking. 16 m halves that frequency vs the old 8 m; the top-down ±75 m map
    // is tolerant of a stale border.
    constexpr float kRainRedrawDist = 16.f;
    // Skip tiny casters (barrels/crates/debris): the dry patch a sub-1.5 m prop
    // shelters is negligible, but they're most of the 19795-item count → cutting
    // them shrinks the redraw spike. Roofs/cars/buildings/trees (large R) stay.
    constexpr float kRainMinCasterR = 1.5f;
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
        }
        s_wasLoaded = nowLoaded;
    }

    // Hold the last real sun direction while a thunderbolt is flashing (it
    // transiently overwrites CurrentEnv->sun_dir — see s_stableSunDir notes).
    // Used by BOTH the far map cache check and the cascades below, so neither
    // swings during a strike.
    const bool boltActive = g_pGamePersistent && g_pGamePersistent->Environment().IsThunderboltActive();
    if (!s_stableSunInit) {
        s_stableSunDir = sunDir; s_stableSunInit = true;
    } else if (!boltActive) {
        s_stableSunDir = sunDir;                          // no bolt → track the real sun exactly
    }
    sunDir = s_stableSunDir;                              // bolt → hold last real direction

    const bool   loaded = RImplementation.b_loaded && !RImplementation.Visuals.empty();
    const size_t nVis   = loaded ? RImplementation.Visuals.size() : 0;

    // GPU-driven opaque sun casters: compute-cull + indirect draw replaces the
    // per-object CPU FlushDepth of OPAQUE statics in the three sun targets (far
    // map + 2 near cascades). The CPU queues then draw only the alpha-tested
    // cutout casters (atOnly). Falls back to the full CPU path when disabled or
    // when the GPU system has no casters / failed to build (so opaque shadows
    // never silently vanish).
    const bool gpuShadows = ps_r_gpu_shadows && ShadowGPU::Built();
    // caster-LOD distance (0 disables → full detail everywhere). Distant casters
    // (esp. progressive terrain/big meshes) emit their coarse slice in the cull.
    const float shadowLodDist = ps_r_shadow_lod ? ps_r_shadow_lod_dist : 0.0f;

    // VSM owns the SUN shadow once its screen-space mask is ready: every sun receiver
    // samples the mask instead of these cascade/far maps (vk_env_light shadow_params.w).
    // So skip the redundant sun RASTER (caster cull + opaque/tree draws + skinned overlay)
    // while VSM is live — the maps' transitions/clears/copies still run, leaving a valid
    // fully-lit map for the one remaining reader (the HUD weapon). Reclaims the measured
    // double-pay (~Combined+Casc0+Casc1; more during sun motion). NOTE: the rain/ground/
    // water-sim maps and the spot/point dynamic-light shadows (Shadow/Dyn) below are NOT
    // sun shadows → they are unaffected.
    const bool vsmActive = ps_r_vsm && VK::VSM::MaskReady();
    // Froxel volumetrics samples the sun cascade per-froxel for in-air occlusion
    // (god rays through windows). VSM's screen-space mask can't answer arbitrary
    // world/air points, so when r_vol is on we keep the NEAR cascades rendered even
    // under VSM (the far map stays VSM-only). Cached → ~one redraw/sec on motion.
    const bool cascForVol = (ps_r_vol || ps_r_vol_debug);
    const bool cascRaster = !vsmActive || cascForVol;   // render the cascade casters this frame
    // r_vol toggled WHILE under VSM: the near cascades were sitting idle (cleared /
    // never transitioned), so their cached-static state + image layouts are stale.
    // Force a fresh first-frame re-init so the froxel occlusion samples real depth.
    {
        static bool s_prevCascForVol = false;
        if (vsmActive && cascForVol != s_prevCascForVol) {
            s_cascStaticFirst = true; s_cascFirst = true;
            for (u32 ci = 0; ci < ShadowMap::kNumSunCascades; ++ci) {
                s_cascStaticValid[ci] = false; s_cascQValid[ci] = false;
            }
        }
        s_prevCascForVol = cascForVol;
    }

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

    // GPU-zone the far static-map REDRAW. Opened EVERY frame (even when the cache
    // holds → brackets no work → ~0 ms) so the profiler's open-order zone indices
    // never shift; a non-zero reading means the cache missed (camera/sun/level).
    const int zFar = VK::Prof::ZoneBegin(cmd, "Shadow/Far");
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

        // GPU-driven opaque caster cull (compute) — MUST run before BeginRendering.
        if (gpuShadows && !vsmActive) {
            Fvector4 planes[6];
            VK::ExtractFrustumPlanes(ShadowMap::GetLightVP(), planes);
            ShadowGPU::Target tgt = ShadowGPU::TGT_FAR;
            ShadowGPU::Cull(cmd, &tgt, planes, 1, Device.vCameraPosition, shadowLodDist);
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

        // Opaque statics via the GPU indirect path; the CPU queue draws only the
        // alpha-tested cutout casters (atOnly). Without the GPU path, full CPU flush.
        // Skipped under VSM (the map is left cleared/fully-lit — receivers use the mask).
        if (!vsmActive) {
            s_ShadowQueue.FlushDepth(cmd, ShadowMap::GetLightVP(), false, gpuShadows);
            if (gpuShadows)
                ShadowGPU::Draw(cmd, ShadowGPU::TGT_FAR, ShadowMap::GetLightVP());

            // Trees (GPU-driven elsewhere — not in the queue): alpha-tested leafy
            // crowns into the sun map, CPU-culled by the light box. Redraw-only cost.
            if (RImplementation.Trees && RImplementation.Trees->IsBuilt())
                RImplementation.Trees->RenderDepth(cmd, ShadowMap::GetLightVP());
        }

        vkCmdEndRendering(cmd);

        // DEPTH_ATTACHMENT → TRANSFER_SRC: the static map is only ever copied from.
        ImageBarrier(cmd, ShadowMap::GetStaticImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
    }
    VK::Prof::ZoneEnd(cmd, zFar);

    // ---- COMBINED map (sampled): every frame = static copy + dynamic casters. ----
    const int zComb = VK::Prof::ZoneBegin(cmd, "Shadow/Combined");
    ImageBarrier(cmd, ShadowMap::GetImage(),
                 s_combinedFirst ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
    s_combinedFirst = false;

    VkImageCopy copy{};
    copy.srcSubresource = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1 };
    copy.dstSubresource = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1 };
    copy.extent         = { sz, sz, 1 };
    if (!vsmActive)   // under VSM nothing samples the far sun map (mask drives receivers) → skip the 4096^2 copy
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

        if (!vsmActive)                                       // VSM live → no sun skinned overlay (mask drives it)
            Skinned_RenderShadow(cmd, ShadowMap::GetLightVP());

        vkCmdEndRendering(cmd);
    }

    // DEPTH_ATTACHMENT → SHADER_READ for the world/skinned receivers this frame.
    ImageBarrier(cmd, ShadowMap::GetImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
    VK::Prof::ZoneEnd(cmd, zComb);

    // ---- RAIN occlusion map: top-down statics+trees depth, cached. Receivers
    // multiply wetness by it — geometry overhead (roof/tunnel) keeps a surface
    // dry. Rendered on demand: only while rain_density or wetness_factor is
    // non-zero (plus one initial clear so the bound image has a defined layout).
    // Zone also covers the per-frame water-sim dispatch below.
    const int zRain = VK::Prof::ZoneBegin(cmd, "Shadow/Rain");
    {
        float rainNeed = 0.f;
        if (g_pGamePersistent && ps_r_rain_enable) {
            const auto& env = g_pGamePersistent->Environment();
            rainNeed = env.wetness_factor;
            if (env.CurrentEnv) rainNeed = _max(rainNeed, env.CurrentEnv->rain_density);
        }
        const bool wantRain   = rainNeed > 0.001f;
        const bool occWanted  = ps_r_light_occ != 0;   // dynamic-light occlusion samples the RAIN map (binding 9)
        const bool wantGround = ps_r_water_sim != 0;   // ground map (binding 13) is water-sim only
        // The snow deform compute samples the RAIN map (binding 9 — the ground map came
        // out empty) for the snow-mesh base height AND the stamp ground-gate, so force
        // the rain map to render + stay fresh whenever deform is active.
        const bool wantDeform = ps_r_snow_deform && ps_r_snow_deform_tex && ps_r_snow > 0.f;
        const bool wantMap    = wantRain || occWanted || wantGround || wantDeform;
        // Occlusion-only (no rain) tolerates a much larger redraw step: the map covers
        // ±120 m and terrain is static, so redrawing every 48 m (vs 16 m for wetness)
        // cuts the always-on spike frequency 3× — fewer frame hitches that make the
        // sun cascades "tick".
        const float redrawDist = wantRain ? kRainRedrawDist : 48.0f;
        const bool stale = !s_rainValid || nVis != s_rainVis
            || Device.vCameraPosition.distance_to_sqr(s_rainCamPos) > redrawDist * redrawDist;

        if (s_rainFirst || (wantMap && stale))
        {
            ShadowMap::ComputeRainVP();

            s_RainQueue.Clear();
            if (loaded && wantMap) {
                // Occlusion-only (no rain) needs just the big blockers (terrain, walls,
                // floors) — skip small props with a much higher min radius so the
                // always-on redraw isn't the all-statics SPIKE that makes the frame
                // hitch ("sun ticks") every 16 m. Wetness still wants the small
                // occluders, so keep the tight cutoff while it's actually raining.
                const float minCasterR = wantRain ? kRainMinCasterR : 6.0f;
                Fmatrix identity; identity.identity();
                for (IRenderVisual* iv : RImplementation.Visuals) {
                    if (!iv) continue;
                    auto* rv = static_cast<vkRender_Visual*>(iv);
                    const Fsphere& bs = rv->vis.sphere;
                    if (bs.R > 0.f && bs.R < minCasterR) continue;   // skip clutter (spike trim)
                    if (bs.R > 0.f && !ShadowMap::RainSphereVisible(bs.P, bs.R)) continue;
                    rv->Submit(s_RainQueue, identity, 0.0f);
                }
                s_RainQueue.SortByKey();
            }
            // Only mark the cache valid if we actually BUILT the queue (loaded). A
            // first render during level load (loaded=false → empty queue) must NOT
            // stick: otherwise, with nVis already full, the staleness check never
            // re-fires and the map stays empty until the camera moves 16 m — the
            // "occlusion does nothing" bug. Staying invalid re-renders next frame.
            s_rainValid  = wantMap && loaded;
            s_rainCamPos = Device.vCameraPosition;
            s_rainVis    = nVis;

            const u32 rsz = ShadowMap::RainSize();
            const VkViewport vpR{ 0.f, (float)rsz, (float)rsz, -(float)rsz, 0.f, 1.f };
            const VkRect2D   scR{ {0,0}, { rsz, rsz } };

            // RAIN map (binding 9): wetness (statics+trees while raining) AND now the
            // dynamic-light occlusion (r_light_occ) — it samples THIS map (known-good
            // plumbing; the separate ground map came out empty). Trees only when
            // raining (a canopy must not occlude a lamp). Render when either wants it;
            // first frame clears once for a defined layout.
            if (wantRain || occWanted || wantDeform || s_rainFirst) {
                ImageBarrier(cmd, ShadowMap::GetRainImage(),
                             s_rainFirst ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);

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
                vkCmdSetViewport(cmd, 0, 1, &vpR);
                vkCmdSetScissor(cmd, 0, 1, &scR);
                vkCmdSetDepthBias(cmd, kBiasConst, 0.0f, kBiasSlope);
                s_RainQueue.FlushDepth(cmd, ShadowMap::GetRainVP());
                if (wantRain && RImplementation.Trees && RImplementation.Trees->IsBuilt())
                    RImplementation.Trees->RenderDepth(cmd, ShadowMap::GetRainVP());
                vkCmdEndRendering(cmd);
                ImageBarrier(cmd, ShadowMap::GetRainImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
            }
            s_rainFirst = false;

            // Clean GROUND-height map (statics+terrain, NO trees): the dynamic-light
            // occlusion march (r_light_occ) AND the water flow sim sample it. Rendered
            // whenever either wants it (the leak fix needs it every frame), else cleared
            // once for a defined layout. Trees are EXCLUDED — a canopy must not block a lamp.
            static bool s_groundFirst = true;
            if (wantGround || s_groundFirst) {
                ImageBarrier(cmd, ShadowMap::GetGroundImage(),
                             s_groundFirst ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
                s_groundFirst = false;
                { static bool s_gl = false; if (!s_gl && s_RainQueue.Size() > 0) { s_gl = true;
                    Msg("[VK Light] ground-height map rendered (occ=%d sim=%d, queue %u items)",
                        ps_r_light_occ, ps_r_water_sim, s_RainQueue.Size()); } }

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
                ImageBarrier(cmd, ShadowMap::GetGroundImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
            }

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

        // ---- Snow deform texture: stamp foot contacts into the dense persistent
        // player-centred press field (vk_deform). Read by the terrain shaders
        // (r_snow_deform_tex) for sharp, persistent, non-faceted footprints.
        if (ps_r_snow_deform && ps_r_snow_deform_tex && ps_r_snow > 0.f) {
            xr_vector<VK::Deform::Stamp> stamps;
            const Fvector& camp = Device.vCameraPosition;
            Fvector fwd = Device.vCameraDirection; fwd.y = 0.f;
            if (fwd.square_magnitude() > 1e-4f) fwd.normalize(); else fwd.set(0.f, 0.f, 1.f);
            const Fvector perp = { -fwd.z, 0.f, fwd.x };
            const float r = ps_r_snow_deform_radius;
            // Player feet ~at ground (≈1.65 m below the camera) so the compute's terrain
            // ground-gate passes them (it compares the stamp Y to the real terrain height).
            const float footY = camp.y - 1.65f;
            stamps.push_back({ { camp.x - perp.x * 0.13f, footY, camp.z - perp.z * 0.13f }, r, 1.f });
            stamps.push_back({ { camp.x + perp.x * 0.13f, footY, camp.z + perp.z * 0.13f }, r, 1.f });
            // Landed items/props: shallow print (~1/3 depth, smaller). The COMPUTE
            // ground-gates each stamp against the real terrain height, so flying items
            // don't stamp (no camera heuristic) and it's robust on uneven terrain.
            xr_vector<Fvector> props; Skinned_CollectProps(props, 16);
            for (const Fvector& p : props) stamps.push_back({ p, r * 0.6f, 0.34f });
            xr_vector<Fvector> npc; Skinned_CollectFeet(npc, 48);
            for (const Fvector& f : npc) stamps.push_back({ f, r, 1.f });
            VK::Deform::Dispatch(cmd, stamps.data(), (u32)stamps.size());
        }
    }
    VK::Prof::ZoneEnd(cmd, zRain);

    // ---- NEAR cascades (R4 port): static/combined split like the far map. The
    // statics-only depth is CACHED (re-rastered only when the camera/sun/level
    // moved past kCascRedrawDist — see Phase A); every frame the sampled combined
    // map = copy(static)+skinned overlay. Stability vs the sun creep still comes
    // from the world-anchored texel alignment inside ComputeCascadeVP.
    {
        if (!s_cascSunInit) {
            s_cascSunDir  = sunDir;
            s_cascSunInit = true;
        } else if (s_cascSunDir.dotproduct(sunDir) < kCascSnapDot) {
            s_cascSunDir = sunDir;                         // load / time-skip discontinuity → snap, don't slew
        } else {
            const float k = 1.f - expf(-Device.fTimeDelta / kCascSunTau);
            s_cascSunDir.lerp(s_cascSunDir, sunDir, k);
            if (s_cascSunDir.magnitude() > 1e-4f) s_cascSunDir.normalize();
            else                                  s_cascSunDir = sunDir;
        }

        // Phase A: pick the cascades whose STATIC map must be re-rastered this
        // frame (camera/sun/level moved past the cache thresholds), recompute ONLY
        // those VPs (the rest stay FROZEN so sampling matches the cached depth),
        // rebuild their caster queues, and batch-cull them in ONE compute pass
        // (so the depth raster never interleaves with compute).
        bool              redrawStatic[ShadowMap::kNumSunCascades] = {};
        ShadowGPU::Target cullTgts[ShadowMap::kNumSunCascades];
        Fvector4          cullPlanes[ShadowMap::kNumSunCascades * 6];
        u32 nCull = 0;
        // Sun-rotation redraw threshold: a frozen cascade re-rasters once the sun
        // creeps past r_shadow_casc_sun degrees so shadows keep moving with the sun.
        // Measured as distance-of-unit-vectors² (≈ θ²) — far better float precision
        // near 0 than a cos dot-product (cos(0.05°) is below float32 epsilon at 1.0).
        // Cheap amortized: ~one 3 ms redraw per second of sun motion. 0 master-toggle
        // (r_shadow_casc_cache) forces a full per-frame raster (old smooth behaviour).
        const float sunEps   = deg2rad(ps_r_shadow_casc_sun);
        const float sunEpsSq = sunEps * sunEps;
        for (u32 ci = 0; ci < ShadowMap::kNumSunCascades; ++ci)
        {
            redrawStatic[ci] = !ps_r_shadow_casc_cache || s_cascStaticFirst || !s_cascStaticValid[ci] || nVis != s_cascStaticVis[ci]
                || Device.vCameraPosition.distance_to_sqr(s_cascStaticCamPos[ci]) > kCascRedrawDist[ci] * kCascRedrawDist[ci]
                || s_cascStaticSunDir[ci].distance_to_sqr(sunDir) > sunEpsSq;
            if (!redrawStatic[ci])
                continue;   // static cached → VP frozen, no queue/cull/raster; only the combined copy+skinned runs in Phase B

            s_cascStaticValid[ci]  = true;
            s_cascStaticCamPos[ci] = Device.vCameraPosition;
            s_cascStaticSunDir[ci] = sunDir;
            s_cascStaticVis[ci]    = nVis;
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

            if (gpuShadows) {
                VK::ExtractFrustumPlanes(ShadowMap::GetCascadeVP(ci), &cullPlanes[nCull * 6]);
                cullTgts[nCull] = (ShadowGPU::Target)(ShadowGPU::TGT_CASCADE0 + ci);
                ++nCull;
            }
        }
        const int zCull = VK::Prof::ZoneBegin(cmd, "Shadow/CascCull");
        if (gpuShadows && nCull && cascRaster)
            ShadowGPU::Cull(cmd, cullTgts, cullPlanes, nCull, Device.vCameraPosition, shadowLodDist);
        VK::Prof::ZoneEnd(cmd, zCull);

        // Phase B: per cascade — (re-)raster the STATIC map only on a cache miss,
        // then EVERY frame build the sampled COMBINED map = copy(static) + skinned
        // dynamics on top (NPCs move, so they can't be cached). The per-cascade
        // zone is opened EVERY frame regardless (profiler matches by open index).
        for (u32 ci = 0; ci < ShadowMap::kNumSunCascades; ++ci)
        {
            const int zCasc = VK::Prof::ZoneBegin(cmd, ci == 0 ? "Shadow/Casc0" : "Shadow/Casc1");
            const u32 nsz = ShadowMap::CascadeSize(ci);
            const VkViewport vpC{ 0.f, (float)nsz, (float)nsz, -(float)nsz, 0.f, 1.f };
            const VkRect2D   scC{ {0,0}, { nsz, nsz } };

            // --- STATIC map (cached): statics + opaque casters + trees. Only on a miss.
            if (redrawStatic[ci])
            {
                ImageBarrier(cmd, ShadowMap::GetCascadeStaticImage(ci),
                             s_cascStaticFirst ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                             VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);

                VkRenderingAttachmentInfo sAtt{};
                sAtt.sType                   = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
                sAtt.imageView               = ShadowMap::GetCascadeStaticView(ci);
                sAtt.imageLayout             = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
                sAtt.loadOp                  = VK_ATTACHMENT_LOAD_OP_CLEAR;
                sAtt.storeOp                 = VK_ATTACHMENT_STORE_OP_STORE;
                sAtt.clearValue.depthStencil = { 1.0f, 0 };
                VkRenderingInfo sri{};
                sri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
                sri.renderArea.extent    = { nsz, nsz };
                sri.layerCount           = 1;
                sri.colorAttachmentCount = 0;
                sri.pDepthAttachment     = &sAtt;
                vkCmdBeginRendering(cmd, &sri);
                vkCmdSetViewport(cmd, 0, 1, &vpC);
                vkCmdSetScissor(cmd, 0, 1, &scC);
                vkCmdSetDepthBias(cmd, kBiasConst, 0.0f, kBiasSlope);

                if (cascRaster) {   // VSM live → normally cleared/fully-lit, but kept rendered for r_vol froxel occlusion
                    s_CascQueue[ci].FlushDepth(cmd, ShadowMap::GetCascadeVP(ci), false, gpuShadows);
                    if (gpuShadows)
                        ShadowGPU::Draw(cmd, (ShadowGPU::Target)(ShadowGPU::TGT_CASCADE0 + ci), ShadowMap::GetCascadeVP(ci));
                    if (RImplementation.Trees && RImplementation.Trees->IsBuilt())
                        RImplementation.Trees->RenderDepth(cmd, ShadowMap::GetCascadeVP(ci), (s32)ci);
                }

                vkCmdEndRendering(cmd);

                ImageBarrier(cmd, ShadowMap::GetCascadeStaticImage(ci), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                             VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
            }

            // --- COMBINED map (sampled): copy the cached static depth, then overlay
            // the skinned (dynamic) casters at their current positions, every frame.
            ImageBarrier(cmd, ShadowMap::GetCascadeImage(ci),
                         s_cascFirst ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);

            VkImageCopy copy{};
            copy.srcSubresource = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1 };
            copy.dstSubresource = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1 };
            copy.extent         = { nsz, nsz, 1 };
            if (cascRaster)   // copy cached static → sampled combined (skipped only when truly VSM-only)
                vkCmdCopyImage(cmd, ShadowMap::GetCascadeStaticImage(ci), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               ShadowMap::GetCascadeImage(ci), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

            ImageBarrier(cmd, ShadowMap::GetCascadeImage(ci), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);

            VkRenderingAttachmentInfo dAtt{};
            dAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            dAtt.imageView   = ShadowMap::GetCascadeView(ci);
            dAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
            dAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;     // keep the static copy
            dAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
            VkRenderingInfo ri{};
            ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
            ri.renderArea.extent    = { nsz, nsz };
            ri.layerCount           = 1;
            ri.colorAttachmentCount = 0;
            ri.pDepthAttachment     = &dAtt;
            vkCmdBeginRendering(cmd, &ri);
            vkCmdSetViewport(cmd, 0, 1, &vpC);
            vkCmdSetScissor(cmd, 0, 1, &scC);
            vkCmdSetDepthBias(cmd, kBiasConst, 0.0f, kBiasSlope);
            if (cascRaster)                                  // skinned (NPC) overlay → they also cast volumetric shadows
                Skinned_RenderShadow(cmd, ShadowMap::GetCascadeVP(ci));
            vkCmdEndRendering(cmd);

            ImageBarrier(cmd, ShadowMap::GetCascadeImage(ci), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
            VK::Prof::ZoneEnd(cmd, zCasc);
        }
        s_cascStaticFirst = false;
        s_cascFirst = false;
    }

    // ===== TODO REMOVE (dead detour, r_vol_shadow default 0): a dedicated per-frame
    // fog sun-shadow built to fix "trembling shafts" — but the real bug was the
    // temporal reprojection (fixed in vol_inject). The VSM/cascade path + temporal is
    // the live one. Safe to delete this whole block + the vk_shadow GetFogShadow*/
    // ComputeFogShadowVP API + vk_volumetrics binding 10 / sampleFogShadow / mode 2. =====
    // Volumetric fog sun-shadow: dedicated low-res sun depth, re-rendered EVERY frame
    // with a fresh anchor-snapped VP → continuous (no cache tick). Cached caster queue. =====
    if (ps_r_vol && ps_r_vol_shadow && loaded) {
        const float kFogRange = 130.f;   // collect casters within this radius of the camera
        if (!s_fogQValid || nVis != s_fogQVis
            || Device.vCameraPosition.distance_to_sqr(s_fogQCamPos) > 8.f * 8.f) {
            s_fogQValid = true; s_fogQCamPos = Device.vCameraPosition; s_fogQVis = nVis;
            s_FogQueue.Clear();
            Fmatrix identity; identity.identity();
            for (IRenderVisual* iv : RImplementation.Visuals) {
                if (!iv) continue;
                auto* rv = static_cast<vkRender_Visual*>(iv);
                const Fsphere& bs = rv->vis.sphere;
                if (bs.R > 0.f && Device.vCameraPosition.distance_to(bs.P) > kFogRange + bs.R) continue;
                rv->Submit(s_FogQueue, identity, 0.0f);
            }
            s_FogQueue.SortByKey();
        }
        ShadowMap::ComputeFogShadowVP(sunDir);
        const u32 fsz = ShadowMap::FogShadowSize();
        ImageBarrier(cmd, ShadowMap::GetFogShadowImage(),
                     s_fogFirst ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
        VkRenderingAttachmentInfo fAtt{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        fAtt.imageView   = ShadowMap::GetFogShadowView();
        fAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        fAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
        fAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
        fAtt.clearValue.depthStencil = { 1.0f, 0 };
        VkRenderingInfo fri{ VK_STRUCTURE_TYPE_RENDERING_INFO };
        fri.renderArea.extent = { fsz, fsz };
        fri.layerCount        = 1;
        fri.pDepthAttachment  = &fAtt;
        vkCmdBeginRendering(cmd, &fri);
        const VkViewport vpF{ 0.f, (float)fsz, (float)fsz, -(float)fsz, 0.f, 1.f };
        const VkRect2D   scF{ {0,0}, { fsz, fsz } };
        vkCmdSetViewport(cmd, 0, 1, &vpF);
        vkCmdSetScissor(cmd, 0, 1, &scF);
        vkCmdSetDepthBias(cmd, kBiasConst, 0.0f, kBiasSlope);
        // Same split as the far map: alpha-tested casters on the CPU queue, OPAQUE
        // statics via the GPU-driven path (their CPU mesh buffers are freed when
        // r_gpu_shadows is on → a CPU FlushDepth of them dereferences dangling
        // pointers = the crash). FlushDepth's alphaTestedOnly=gpuShadows skips
        // opaque when the GPU path covers them; reuse TGT_FAR's culled opaque set
        // (covers the fog box) drawn with the fresh fog VP.
        s_FogQueue.FlushDepth(cmd, ShadowMap::GetFogShadowVP(), false, gpuShadows);
        if (gpuShadows)
            ShadowGPU::Draw(cmd, ShadowGPU::TGT_FAR, ShadowMap::GetFogShadowVP());
        if (RImplementation.Trees && RImplementation.Trees->IsBuilt())
            RImplementation.Trees->RenderDepth(cmd, ShadowMap::GetFogShadowVP());
        vkCmdEndRendering(cmd);
        ImageBarrier(cmd, ShadowMap::GetFogShadowImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
        s_fogFirst = false;
    }

    // ===== Dynamic light shadows (STEP 3b): 1 spot + 1 point cube per frame =====
    // EnvLight::Update (Pass_World, later this frame) reads the same CollectFrame
    // result, so the gpu[] indices below match what the receivers see.
    const auto& FL = Lights::CollectFrame(Device.vCameraPosition);
    const bool haveSpot  = FL.spotIdx  >= 0;
    const bool havePoint = FL.pointIdx >= 0;

    // Spot (flashlight) + point (campfire) dynamic shadow maps. Zone opened here
    // and closed on BOTH exits below so the per-frame zone count is constant even
    // when the early-out fires (profiler matches zones by open-order index).
    const int zDyn = VK::Prof::ZoneBegin(cmd, "Shadow/Dyn");

    // Nothing selected and both maps already initialized → zero cost.
    if (!haveSpot && !havePoint && !s_spotFirst && !s_pointFirst)
    {
        VK::Prof::ZoneEnd(cmd, zDyn);
        return;
    }

    // Rebuild a cached static-caster queue if the light sphere changed (full
    // visuals walk — the expensive part we must NOT do per frame). Oversized
    // visuals (terrain chunks) are excluded — see kSpotCasterMaxR.
    auto refreshStaticQueue = [&](RenderQueue& q, bool& valid, Fvector& qPos, float& qRange,
                                  size_t& qVis, const Fvector& p, float range, float maxCasterR) -> bool {
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
                if (bs.R > maxCasterR) continue;
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
            // Flashlight: keep the terrain-excluding cap (wide range, terrain is a
            // receiver here, not a blocker) — see kSpotCasterMaxR.
            refreshStaticQueue(s_SpotQueue, s_spotQValid, s_spotQPos, s_spotQRange, s_spotQVis,
                               FL.spotPos, FL.spotRange, kSpotCasterMaxR);
        } else {
            s_spotQValid = false;
        }
        renderDepthTarget(ShadowMap::GetSpotView(), ShadowMap::SpotSize(),
                          ShadowMap::GetSpotVP(), &s_SpotQueue, FL.spotPos, FL.spotRange, haveSpot);

        ImageBarrier(cmd, ShadowMap::GetSpotImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    // --- POINT (campfire / lamp) cube: STATIC occluders + skinned casters. The
    // statics (floors, walls, AND terrain — no size cap here) are what stop the
    // light + the NPC shadow from leaking through the ground onto geometry above
    // (a basement lamp lighting the earth + a fence overhead, with the NPC's
    // shadow cast onto it). Terrain is the blocker for a small-range buried lamp,
    // unlike the wide flashlight where it's a receiver. Re-render while an NPC is
    // (or was) in radius OR when the static set changed (light picked / moved
    // >1 m); otherwise the cached cube keeps occluding for free.
    {
        bool staticChanged = false;
        if (havePoint)
            // Statics only (walls/floors) — NOT terrain. Rendering terrain chunks into
            // 6 cube faces was a per-frame SPIKE that made the sun cascades "tick".
            // Under-terrain (earth) lamps are occluded by the rain-map heightfield
            // (r_light_occ) in the receivers instead; building floors still occlude here.
            staticChanged = refreshStaticQueue(s_PointQueue, s_pointQValid, s_pointQPos, s_pointQRange,
                                               s_pointQVis, FL.pointPos, FL.pointRange, kSpotCasterMaxR);
        else
            s_pointQValid = false;

        const bool anySkinned  = havePoint && Skinned_AnyCasterInSphere(FL.pointPos, FL.pointRange);
        const bool renderPoint = havePoint && (anySkinned || s_pointHadSkinned || staticChanged);
        if (renderPoint || s_pointFirst)
        {
            ImageBarrier(cmd, ShadowMap::GetPointImage(),
                         s_pointFirst ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT, 6);
            s_pointFirst = false;

            for (u32 f = 0; f < 6; ++f) {
                const Fmatrix faceVP = ShadowMap::ComputePointFaceVP(FL.pointPos, FL.pointRange, f);
                renderDepthTarget(ShadowMap::GetPointFaceView(f), ShadowMap::PointSize(),
                                  faceVP, &s_PointQueue, FL.pointPos, FL.pointRange, renderPoint);
            }

            ImageBarrier(cmd, ShadowMap::GetPointImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT, 6);
        }
        s_pointHadSkinned = anySkinned;
    }
    VK::Prof::ZoneEnd(cmd, zDyn);
}

}  // namespace VK
