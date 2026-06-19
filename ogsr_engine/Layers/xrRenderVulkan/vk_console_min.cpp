// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// Vulkan port of xrRender_console.cpp. Adapted from
// Layers/xrRender/xrRender_console.cpp — variables + CMD registrations are
// shared; the few R4-only command classes (CCC_DumpResources, CCC_OCC_Enable,
// CCC_VideoMemoryStats, ...) that reach into RImplementation / HW.stats_manager
// are stripped — Settings menu only needs the value/token registrations.
#include "stdafx.h"

#include "../xrRender/xrRender_console.h"
#include "vk_profiler.h"   // VK::Prof::RequestMark — the `vk_perf` MARK command

// DLSS preset enum values used in default initializers; pull them in directly
// rather than depending on the Vulkan-side DLSS wrapper.
enum {
    NVSDK_NGX_DLSS_Hint_Render_Preset_Default = 0,
    NVSDK_NGX_DLSS_Hint_Render_Preset_F       = 6,
    NVSDK_NGX_DLSS_Hint_Render_Preset_J       = 10,
    NVSDK_NGX_DLSS_Hint_Render_Preset_K       = 11,
};

u32 r2_SmapCascade0Size{2048}, /*r2_SmapCascade1Size{1536},*/ r2_SmapCascade2Size{1024};
constexpr xr_token CascadesSmapSizeToken[]{// {"512x512", 512},
                                      {"1024x1024", 1024},
                                      {"1536x1536", 1536},
                                      {"2048x2048", 2048},
                                      {"2560x2560", 2560},
                                      {"3072x3072", 3072},
                                      {"4096x4096", 4096},
                                      //{"6144x6144", 6144},
                                      //{"8192x8192", 8192},
                                      {}};

u32 r2_SmapLightsSize = 3072;
constexpr xr_token LightsSmapSizeToken[]{//{"1536x1536", 1536},
                                        //{"2048x2048", 2048},
                                        {"2560x2560", 2560},
                                        {"3072x3072", 3072},
                                        {"4096x4096", 4096},
                                        {"6144x6144", 6144},
                                        {"8192x8192", 8192},
                                        {}};

u32 r2_SmapRainSize = 1024;
constexpr xr_token RainSmapSizeToken[]{{"512x512", 512},
                                      {"1024x1024", 1024},
                                      {"1536x1536", 1536},
                                      {"2048x2048", 2048},
                                      {}};

u32 ps_r_pp_aa_mode = DLSS;
constexpr xr_token pp_aa_mode_token[] = {
    {"st_opt_off", NO_AA},
    {"st_opt_dlss", DLSS},
    {"st_opt_fsr2", FSR2},
    {"st_opt_taa", TAA},
    {"st_opt_smaa", SMAA},

    {nullptr, 0},
};

u32 ps_r_dlss_preset = NVSDK_NGX_DLSS_Hint_Render_Preset_F;
constexpr xr_token dlss_mode_token[]{
    {"st_opt_dlss_default", NVSDK_NGX_DLSS_Hint_Render_Preset_Default}, // default behavior, may or may not change after OTA
    {"st_opt_dlss_f", NVSDK_NGX_DLSS_Hint_Render_Preset_F},
    {"st_opt_dlss_j", NVSDK_NGX_DLSS_Hint_Render_Preset_J},
    {"st_opt_dlss_k", NVSDK_NGX_DLSS_Hint_Render_Preset_K},
    {},
};

float ps_r_dlss_3dss_scale_factor{1.0f};

u32 ps_r_sunshafts_mode = SS_SS_OGSE;
constexpr xr_token sunshafts_mode_token[]{{"st_opt_off", SS_OFF},
                                          {"volumetric", SS_VOLUMETRIC},
                                          {"ss_ogse", SS_SS_OGSE},
                                          {"ss_manowar", SS_SS_MANOWAR},
                                          {"combined_ogse", SS_COMBINED_OGSE},
                                          {"combined_manowar", SS_COMBINED_MANOWAR},
                                          {}};

// Sunshafts
u32 ps_r_sun_shafts = 3;

float ps_r_ss_sunshafts_length = 0.9f; // 1.0f;
float ps_r_ss_sunshafts_radius = 2.f; // 1.0f;
float ps_r_prop_ss_radius = 1.56f;
float ps_r_prop_ss_blend = 0.25f; // 0.066f;
float ps_r_prop_ss_sample_step_phase0 = 0.09f;
float ps_r_prop_ss_sample_step_phase1 = 0.07f;

float ps_r_alphatest_threshold{200.f / 255.f};

u32 ps_preset = 2;
constexpr xr_token qpreset_token[] = {{"Minimum", 0}, {"Low", 1}, {"Default", 2}, {"High", 3}, {"Extreme", 4}, {nullptr, 0}};

u32 ps_r_ao_mode = AO_MODE_SSDO;
constexpr xr_token ao_mode_token[] = {{"st_gtao", AO_MODE_GTAO}, {"st_ssdo", AO_MODE_SSDO}, {nullptr, 0}};

u32 ps_r_ao_quality = 0;
constexpr xr_token qssao_token[] = {{"st_opt_off", 0},
                                    {"st_opt_low", 1},
                                    {"st_opt_medium", 2},
                                    {"st_opt_high", 3},
                                    {nullptr, 0}};

// Vulkan GTAO (vk_pass_ssao): live debug view + strength, no restart needed.
int   ps_r_ssao_debug    = 0;     // 1 = draw the raw AO map instead of the scene
// Global GTAO on/off (r_ssao): 0 SKIPS the whole pass (depth flip + Execute) →
// zero GPU cost, binding 8 falls back to white. For A/B perf measurement and a
// real off switch (r2_ssao 0 is treated as "medium", not off).
int   ps_r_ssao_enable   = 1;
// NPC normal G-buffer for GTAO (r_ssao_npc_normals): 1 = NPCs feed real per-pixel
// normals into GTAO (no depth-derivative speckle on characters); 0 = NPCs fall
// back to depth-reconstructed normals like statics. A/B toggle, no restart.
int   ps_r_ssao_npc_normals = 1;
// Strength is an exponent on the AO value (0 = off, 1 = raw GTAO). Default 2:
// our AO input is the depth prepass (statics+trees, no grass/NPCs), and the
// forward path applies AO to a smaller ambient share than R4's deferred
// hemisphere — the deepened curve compensates to a comparable look.
float ps_r_ssao_strength = 2.0f;

// Vulkan SSIL — screen-space indirect lighting (one-bounce SSGI). The "IL +
// COLOR" complement to GTAO's "AO": adds the coloured light bouncing off nearby
// lit surfaces. FOLDED INTO the GTAO horizon march (the occluder that raises the
// horizon also bounces its colour), gathered from the PREVIOUS frame's lit scene
// (history buffer — also the temporal foundation). Requires r_ssao on (shared
// pass). Reach = the GTAO radius.
// Default OFF: the v2 horizon-folded IL still bands a little at half-res without
// temporal accumulation (the planned next step), so it ships opt-in — r_ssil 1
// to A/B. AO is unaffected either way.
int   ps_r_ssil_enable   = 0;     // r_ssil — global on/off (gates the prev-colour taps in the GTAO march)
int   ps_r_ssil_debug    = 0;     // r_ssil_debug — 1 = show ONLY the indirect bounce field
// Strength multiplies the gathered radiance at composite. IL is an occluder-colour
// AVERAGE gated by (1-AO)² (≈0 on open surfaces, only fills real recesses), added
// on top of the forward ambient — keep it subtle so it tints corners without a
// global brightness boost. Raise to taste.
float ps_r_ssil_strength = 0.5f;

// Vulkan motion vectors — screen-space (prevUV − curUV) reconstructed from the
// prepass depth + the previous frame's view-proj. Foundation for DLSS/FSR
// upscaling, frame-gen and the path-tracer denoiser. Phase 1 = camera + static
// world (the dominant motion); self-moving geometry is a later phase. Default
// ON (cheap fullscreen pass, no result consumer yet — harmless until DLSS lands).
int   ps_r_motion_vectors  = 1;     // r_motion_vectors — global MV pass on/off
int   ps_r_mv_debug        = 0;     // r_mv_debug       — false-colour overlay (verify sign/Y-flip/magnitude)
float ps_r_mv_debug_scale  = 30.0f; // r_mv_debug_scale — overlay magnitude (per-frame UV motion is tiny)

// SSFX tree wind (live tuning). Trunk bend defaulted DOWN from the SSFX 0.5 —
// our trees are tall, the H² trunk term made them "sway like sails" at 0.5.
float ps_r_wind_tree_bend    = 0.18f; // r_wind_tree_bend    — trunk sway intensity (SSFX wsetup_trees.z)
float ps_r_wind_tree_anim    = 11.0f; // r_wind_tree_anim    — branch/leaf flutter speed (SSFX wsetup_trees.x)
float ps_r_wind_tree_trunk   = 0.15f; // r_wind_tree_trunk   — trunk anim speed (SSFX wsetup_trees.y)
float ps_r_wind_tree_flutter = 4.0f;  // r_wind_tree_flutter — crown/leaf flutter amplitude (our extra, SSFX has none)
float ps_r_wind_tree_crown   = 4.0f;  // r_wind_tree_crown   — height (m) where leaf flutter fades in (low trunk stays still)
float ps_r_wind_shadow_dist  = 40.0f; // r_wind_shadow_dist  — radius (m) where tree SHADOWS sway (near=per-frame, far=cached); 0 = all static (cheapest)
int   ps_r_vsm_tree_wind     = 0;     // r_vsm_tree_wind — TEST/experimental: apply wind to VSM tree shadow pages (animates only when static pages refresh; static sun = frozen). Default OFF.

// Vulkan lighting normalization knobs (live, no restart). Both used to be
// literals scattered across the scene shaders (LDR-era compensation that
// predates the HDR tonemap): sun ×1.25 in world/terrain/vlit/skinned/grass
// and +0.05 ambient floors. They now apply ONCE on the CPU (vk_env_light UBO
// fill + the grass/tree sun pushes), so shaders consume final values and the
// look can be A/B'd in-game: r_sun_boost 1 + r_ambient_floor 0 = raw env values.
float ps_r_sun_boost     = 1.25f;
float ps_r_ambient_floor = 0.05f;

// Rain wetness knobs (live): darken = how much wet albedo darkens (0 = off,
// 0.4 ≈ wet asphalt), refl = sky-reflection strength on wet surfaces.
// r_wet_debug 1 = world shaders draw the wet mask (wetness × rain-map
// visibility) grayscale — white = soaked, black = dry/covered.
float ps_r_wet_darken = 0.4f;
float ps_r_wet_refl   = 0.6f;
int   ps_r_wet_debug  = 0;
// r_rain_debug 1 = rain streaks drawn SOLID RED (triage: geometry vs texture).
int   ps_r_rain_debug = 0;
// r_rain 0 = master OFF: skips the whole Rain pass (drops/splashes/thunderbolt)
// AND forces wetness/density to 0 (dry surfaces, no rain occlusion map). Does
// NOT change the weather — just suppresses the rain effect for testing.
int   ps_r_rain_enable = 1;

// Global render profiler (vk_profiler). 0 = no [VK Perf] logging, 1 = periodic
// (~5s) GPU/CPU/VRAM log, 2 = + the live ImGui overlay (Phase 2). `vk_perf`
// forces an immediate MARK snapshot regardless of this value.
int   ps_r_profiler = 1;

// Variable Rate Shading (vk_vrs, depth-driven). 0 = off, 1 = mild, 2 = aggressive.
// near/far = distance (m) thresholds: below near stays 1x1, above far goes coarsest.
// (Live — tune without restart. Level 2 pulls them ~closer.)
int   ps_r_vrs      = 0;
float ps_r_vrs_near = 40.0f;
float ps_r_vrs_far  = 75.0f;

// Frustum culling of world statics (default ON). The big win vs R4 — we used to
// submit the entire level every frame. r_cull 0 = brute-force all (A/B).
int   ps_r_cull = 1;

// GPU-driven sun shadow casters (vk_shadow_gpu): opaque statics are compute-culled
// per cascade and drawn via vkCmdDrawIndexedIndirectCount; the CPU queue then draws
// only the alpha-tested cutout casters. Default ON. r_gpu_shadows 0 = old CPU
// FlushDepth-all path (A/B for the SunShadow profiler zone).
int   ps_r_gpu_shadows = 1;

// caster-LOD for GPU-driven sun shadows: casters farther than r_shadow_lod_dist
// metres from the camera draw their COARSE sliding-window slice (progressive
// terrain/big meshes → far fewer triangles in the shadow map). r_shadow_lod 0 =
// full detail everywhere (A/B). Only affects the GPU shadow path.
int   ps_r_shadow_lod      = 1;
float ps_r_shadow_lod_dist = 30.0f;

// Cascade shadow-map STATIC cache (vk_pass_shadow): the statics-only cascade depth
// is re-rastered only when the camera moves past kCascRedrawDist OR the sun rotates
// past r_shadow_casc_sun degrees; otherwise it's reused and only the skinned dynamics
// are overlaid (kills the ~3 ms/frame Shadow/Casc0 raster while standing). 0 = old
// per-frame raster (smoothest sun creep, full cost). r_shadow_casc_sun is the sun
// step that forces a redraw — SMALLER = smoother shadow motion as the sun moves, at
// the cost of more frequent (but cheaply amortized) redraws.
int   ps_r_shadow_casc_cache = 1;
float ps_r_shadow_casc_sun   = 0.05f;

// Virtual Shadow Maps (vk_vsm) — sun directional clipmap with sparse physical pages:
// only pages sampled by visible pixels are rendered, and they're cached across
// frames in world space. Smooth moving-sun shadows (only visible pages re-render)
// + standing ≈ 0. WIP, default OFF; the cascade path above stays the shipped default
// until VSM proves out. r_vsm_debug logs per-frame page mark/alloc counts.
int   ps_r_vsm       = 0;
int   ps_r_vsm_debug = 0;
// VSM receiver depth-compare bias (normalized clipmap Z, range ~2000 m). Larger =
// less acne but more light-leak (small/thin caster shadows fade). Live-tunable.
float ps_r_vsm_bias  = 0.0003f;
// VSM clipmap detail: base extent (m) of clipmap level 0 → finest texel = base/4096.
// 24 m ≈ the old 4096² cascade (5.9 mm), balanced. Smaller = sharper but more pages
// (the 2048-page atlas can overflow on wide vistas → distant pages drop, graceful).
// Live-tunable; meant to back a future "VSM detail Low/Med/High" graphics slider.
float ps_r_vsm_base  = 24.0f;   // finest texel 5.9mm ≈ old cascade; temporal accumulation makes this coarse base look as clean as 12 did (user-verified) → cheap default. Future graphics slider Low/Med/High = 32/24/16.
// VSM temporal accumulation (TAA-for-shadows): sub-texel jitter the clipmap origin each
// frame + EMA-blend a reprojected history → kills the moving-sun "crawling snake" along
// shadow edges. The proper crawl fix (lets base go back to a cheap coarse value). Live.
int   ps_r_vsm_temporal = 1;
// History weight (EMA alpha): higher = smoother/stabler but more ghosting under motion;
// lower = crisper but more residual crawl. 0.9 ≈ ~10-frame convergence. Live-tunable.
float ps_r_vsm_ta_blend = 0.9f;
// Grass casts VSM shadows (near + L0 only; reads the GPU-driven detail instance buffer
// 1 frame stale). DEFAULT OFF: grass blades are thinner than the shadow texel (blobby) and
// the static-caster temporal filter ghosts the near-static grass ("see-through"). Kept,
// gated, for experimentation — see [[vulkan-vsm]]. Live-tunable.
int   ps_r_vsm_grass      = 0;
float ps_r_vsm_grass_dist = 12.0f;   // max grass cast distance from camera (m)
float ps_r_vsm_npc_dist   = 50.0f;   // max NPC shadow-cast distance into the VSM atlas (m); 0 = no cull (NPC shadows tiny past ~50m)
float ps_r_vsm_lod_dist   = 0.0f;    // VSM caster-LOD: opaque casters draw coarse slice past this (m); 0 = off (measured marginal in village, like the cascade; kept for open maps)
int   ps_r_vsm_mark_half  = 1;       // page-mark at half-res (4x fewer threads/atomics); 0 = full-res
// VSM static-atlas CACHE (Phase 2): the static (opaque + tree) atlas is persistent; when
// the sun hasn't moved (> r_vsm_cache_sun) AND the camera is still within the current page
// cell, the whole static atlas pass is SKIPPED and last frame's atlas is reused (the dynamic
// NPC/grass atlas still re-renders). Standing/aiming on a vista: VSMrender ~10ms -> ~0. Also
// page-snaps the static clipmap window (vs texel) so sub-page camera moves don't invalidate.
// DEFAULT OFF (0 = exactly the Step-0 per-frame path) — A/B knob. Live.
int   ps_r_vsm_cache     = 0;
// Sun-rotation tolerance (deg) before the cached static atlas re-renders. Larger = holds the
// cache longer (cheaper) but a bigger one-shot shadow jump when it ticks (temporal resolve is
// meant to smooth it). 0.05 ~ the old cascade-cache threshold. Live-tunable.
float ps_r_vsm_cache_sun = 0.05f;
// Camera-TURN tolerance (deg) before the cached static atlas re-renders. A frozen atlas only
// holds the pages visible at freeze-time, so turning past this reveals un-rendered pages →
// re-render. Smaller = no missing-shadow wedge while turning but more re-render (turning costs
// like moving anyway); larger = holds the freeze through bigger turns but a thin wedge may show
// at the screen edge. Live-tunable. (The proper fix for smooth turning is per-page residency.)
float ps_r_vsm_cache_rot = 8.0f;
// Round-robin refresh period (frames) for the moving-sun case (Phase 1b toroidal cache): each
// frame ~1/N of the resident static pages re-render to track the creeping sun smoothly (no
// jump). Smaller = fresher but costlier; only active while the sun moves (paused sun → 0). Live.
int   ps_r_vsm_cache_refresh = 8;

// GPU-driven world forward pass (vk_world_gpu): static opaque/AT meshes are
// compute-culled + drawn via indirect (1 draw/material group) instead of the
// per-object CPU queue, which also dedups the hierarchy double-submit. Cuts CPU
// draw-call count massively → fps win on CPU-bound / detail-heavy levels.
// r_gpu_world 0 = old CPU path (A/B). Default ON: measured ~2× fps in the village
// (CPU ~halved, World/Statics ~5× lower) + user-verified visually identical.
int   ps_r_gpu_world = 1;

// Clustered forward (Forward+, vk_clustered): a compute pass bins the active
// dynamic lights into a 16x9x24 froxel grid; each fragment iterates only the
// few lights touching its cluster instead of all 16 with zero culling. Raises
// the light cap 16 -> 256 AND makes shaded pixels cheaper (the structural gap
// vs R4 deferred). Default OFF for A/B + safety: r_clustered 0 keeps the exact
// old per-fragment 16-light loop. v1 covers world (lmap/vlit/terrain) + skinned;
// foliage stays on the 16-light path. r_clustered_debug draws a per-cluster
// light-count heatmap (validates the cull).
int   ps_r_clustered       = 0;
int   ps_r_clustered_debug = 0;

// Froxel volumetric lighting (vk_volumetrics, r_vol) — P1. A 3D froxel grid over
// the frustum: compute injects sun in-scatter (Henyey-Greenstein phase × cascade
// sun-shadow) + height/base fog, integrates it front-to-back, and the tonemap
// composites scene*transmittance + in-scatter (HDR, pre-tonemap). = god rays
// through geometry + depth fog (the Metro base look). Default OFF. r_vol_height 0
// = uniform fog; r_vol_debug shows the raw integrated in-scatter pattern.
int   ps_r_vol           = 1;       // ON by default — shipped feature (god-rays + depth fog + indoor haze), ~1.5ms
float ps_r_vol_density   = 0.02f;   // base extinction / scatter density
float ps_r_vol_height    = 0.10f;   // height-fog falloff above eye level (0 = uniform)
float ps_r_vol_g         = 0.80f;   // Henyey-Greenstein anisotropy (forward scatter)
float ps_r_vol_intensity = 3.0f;    // in-scatter brightness multiplier (raised: fog must GLOW more than it dims to read as haze)
float ps_r_vol_amb       = 0.60f;   // indoor ambient floor: fraction of sky ambient kept under a roof (0=pitch-dark interior, 1=no occlusion)
float ps_r_vol_indoor    = 6.0f;    // indoor density boost: fog ×(1+this) under a roof — short interior sightlines need denser air to show
float ps_r_vol_sun       = 3.0f;    // sun-beam in-scatter boost: directional shaft brightness (pops the god-ray through the ambient haze)
float ps_r_vol_lights    = 2.5f;    // P2: local light (flashlight/lamp/campfire) in-scatter in fog — glow/cone strength; 0 = off
float ps_r_vol_smoke     = 1.0f;    // Stage-0: light smoke billboards with the froxel in-scatter (sun shaft/flashlight/campfire catch the smoke); 0 = off (old flat look)
float ps_r_vol_smoke_clamp = 6.0f;  // Stage-0: upper bound on the per-froxel radiance added to smoke (keeps it from blowing to white next to a campfire/sun beam)
float ps_r_vol_smoke_inject  = 1.0f; // Stage-1 VMS: inject smoke-particle density into the froxel grid → smoke becomes real participating media (lit/shadowed like fog); 0 = off
float ps_r_vol_smoke_density = 1.0f; // Stage-1 VMS: density mass per particle opacity (splat scale) — thicker/thinner injected smoke
int   ps_r_vol_smoke_debug   = 0;    // VMS debug: 0 off / 1 smoke colour glow (albedo) / 2 density heatmap (grey) / 3 self-shadow sunVis (blue=shadowed, warm=sun-lit)
float ps_r_vol_smoke_dist_full = 12.0f; // Stage-1 VMS LOD: radius (m) of FULL-quality volumetric smoke (close-up "beautiful" body); past it density fades to billboard by r_vol_smoke_dist.
float ps_r_vol_smoke_dist    = 40.0f; // Stage-1 VMS: max distance (m) smoke injects as volumetric media (tied to the terrain detail bubble — capped at r__detail_radius); beyond it the billboard alone represents it (far froxels = coarse muddy blobs). Fades from dist_full → here.
float ps_r_vol_smoke_footprint = 1.0f; // Stage-1.1 VMS: max smoke splat footprint in froxel cells (0 = point splat / cheapest; 1 = 3³ gaussian blob; 2 = 5³). Soft + denser; cost ~ particles × (2·fp+1)³ — watch the VolSmoke profiler zone.
float ps_r_vol_smoke_shadow  = 1.0f; // Stage-2 VMS: smoke SELF-SHADOW strength (cloud sun-side bright, far/deep side dark — gives it volume). 0 = off (flat lit). Only smoky froxels march toward the sun.
float ps_r_vol_smoke_shadow_step = 0.6f; // Stage-2 VMS: self-shadow sun-march step (world m); 6 steps total. Bigger = longer reach/softer, smaller = tighter.
float ps_r_vol_noise     = 0.55f;   // P3: animated 3D noise on the fog density → drifting dust/mist ("living air"); 0 = off
float ps_r_vol_noise_scale = 0.40f; // P3 noise frequency (world units; higher = finer motes)
float ps_r_vol_noise_speed = 0.10f; // P3 drift speed of the dust
float ps_r_vol_soft      = 2.5f;    // cascade shadow PCF blur radius (texels): soft penumbra in fog so the cache TICK (sun creep through foliage) barely shows; 0.5 = crisp
int   ps_r_vol_ta        = 1;       // temporal accumulation (jitter + reproject prev frame): smooths the froxel grid → clean dense fog
float ps_r_vol_ta_blend  = 0.92f;   // history weight (EMA): higher = smoother but more ghosting on motion
// TODO REMOVE (dead detour): the dedicated per-frame fog sun-shadow. Built to fix
// the "trembling shafts", but the real bug was the temporal reprojection (now fixed),
// so this is NOT needed. Kept OFF as an A/B toggle; safe to delete later along with
// vk_shadow GetFogShadow*/ComputeFogShadowVP, the vk_pass_shadow fog render block,
// vk_volumetrics binding 10 + sampleFogShadow, and gridParams.w mode 2.
int   ps_r_vol_shadow    = 0;       // [deprecated] 0 = cascade/VSM path (the live one)
int   ps_r_vol_debug     = 0;       // view the raw integrated in-scatter

// Dynamic-light terrain/static occlusion (vk_shadow ground-height map + a per-light
// height-march in the forward shaders). Stops un-shadowed point lights from
// lighting through the ground/walls — a basement lamp no longer lights the earth +
// fence overhead. Scales O(1) per light to ANY light count (unlike per-light shadow
// maps). Needs the top-down ground map rendered every frame (cached on camera move).
// Default ON; r_light_occ 0 = off (A/B / if the map redraw is too costly).
int   ps_r_light_occ       = 1;

// World heightmap tessellation (R4 TESS_HM port, live): bump-mapped statics
// displace along the normal by the `<bump>#` alpha height near the camera.
// r_tess 0 routes everything back to the flat pipelines. max = subdivision
// factor at point-blank; near/far = distance band over which the factor (and
// the displacement amplitude) fades to flat; height scales R4's 0.07 m
// amplitude. Consumed by RenderQueue::Flush (vk_render_queue.cpp).
// Default OFF: vertex tessellation inflates on this content (the `#` alpha is
// unsigned parallax/error-height, not a centered displacement map). POM
// (r_pom) is the relief mechanism for now; re-enable r_tess once the "done
// right" roadmap lands (high-pass centered height + tess depth-prepass).
float ps_r_tess        = 0.0f;
float ps_r_tess_max    = 8.0f;
float ps_r_tess_near   = 3.0f;
float ps_r_tess_far    = 18.0f;
float ps_r_tess_height = 1.0f;
// Parallax occlusion mapping (per-pixel brick/cobble relief — the right tool
// for un-authored content, vs vertex tessellation). Reads the same `<bump>#`
// height (alpha) as tess but per-pixel: carves grooves, never inflates. Only
// world lmap/vlit materials with a real `#` height (flat-bump = no-op).
// height = UV-space march amplitude; steps = max ray samples; far = fade dist.
int   ps_r_pom        = 1;
float ps_r_pom_height = 0.02f;
float ps_r_pom_steps  = 24.0f;
float ps_r_pom_far    = 12.0f;
// Extra mip blur on the POM heightfield: removes fine albedo speckle (which
// the diffuse-luminance height would otherwise turn into "spikes" on brick
// faces) while the large-scale mortar grid survives. Visible texture stays
// sharp — only the displacement is smoothed. Raise if bricks look spiky.
float ps_r_pom_blur   = 0.5f;
// Strength of the POM normal perturbation (relief catches sun/dyn/hemi light).
// 0 = flat lighting (offset only), 1 = default, higher = deeper-looking grooves.
float ps_r_pom_normal = 1.0f;
// POM self-shadow strength: contact shadows the relief casts in its own grooves
// toward the sun. 0 = off, 1 = default. Adds 8 height taps on sun-facing pixels.
float ps_r_pom_shadow = 1.0f;
// POM contact AO: view-independent groove darkening on the AMBIENT term, so the
// relief reads with depth even out of direct sun. 0 = off, 1 = default.
float ps_r_pom_ao     = 1.0f;
// r_pom_debug 1 = world shaders draw the POM occlusion mask (AO × self-shadow)
// grayscale — white = lit/open, dark = occluded grooves. Like r_ssao_debug but
// for the per-pixel POM contributions (which the GTAO debug view can't show).
int   ps_r_pom_debug  = 0;
// r_ao_flat 1 = DEBUG: force ALL ambient occlusion to neutral on world lmap/vlit
// — GTAO, POM-AO, the baked lightmap hemi-occlusion AND the dynamic hemi gate.
// Shows the scene with perfectly flat ambient (looks "wrong" by design) so you
// can see how much each occlusion source contributes. The baked lightmap is the
// dominant interior AO and this is the only way to neutralize it at runtime.
int   ps_r_ao_flat    = 0;
// Per-orientation POM strength. Walls/fences (horizontal normal) always full;
// floors (normal up) scale toward r_pom_floor; ceilings (normal down) toward
// r_pom_ceil. Defaults: floor 0.75, ceiling 0.25 (overhead POM reads strong).
float ps_r_pom_ceil   = 0.25f;
float ps_r_pom_floor  = 0.75f;
// Terrain POM is EXPERIMENTAL / off by default: the ground is viewed at grazing
// angles almost always and its base texture is high-contrast, so POM there
// "swims"/mirrors. Walls/fences/floors-of-structures (lmap/vlit) keep POM.
int   ps_r_pom_terrain = 0;
// Terrain DETAIL NORMAL MAPPING strength (independent of POM). Perturbs the
// ground normal from the per-channel <detail>_bump maps (grass/asphalt/earth/
// gravel), blended by the splat mask — feeds sun + dyn lights only (the sharp
// sky cube stays on the flat geometric normal, else up-facing ground mirrors).
// No UV march → no grazing-angle "swim". 0 = off; 1 = full.
float ps_r_terrain_normal = 1.0f;
// Terrain MICRO contact AO: darkens micro-grooves using the detail-normal tilt
// (cavity) AND the detail height (R4 terrain AO = detail diffuse alpha). Cheap,
// no UV march, no swim — deepens the relief the detail normals create. 0 = off.
float ps_r_terrain_ao    = 0.5f;
// Terrain debug view: 0 off, 1 = world normal (Nw*0.5+0.5), 2 = micro-AO,
// 3 = detail height (splat-blended detail alpha).
int   ps_r_terrain_debug = 0;
// Terrain DRY sun gloss: a material-aware specular highlight from the bump .r
// channel (R4 gloss). Asphalt/gravel catch the sun even when dry; grass stays
// matte. Fades out as the ground wets (the wet reflection takes over). 0 = off.
float ps_r_terrain_gloss = 0.5f;
int   ps_r_puddle_debug  = 0;       // puddle/water debug: 0 off, 1 = coverage, 2 = micro-height / sim flow

// SURFACE FIELD (the "smart heightmap"): metre-scale ground height/slope/curvature/
// exposure/canopy derived from the rain+ground maps (shaders/surface_field.glsl).
// r_sf = master enable for future consumers (snow/fog/water); r_sf_debug visualizes
// the field; r_sf_eps tunes the finite-difference scale.
int   ps_r_sf        = 0;       // r_sf — master enable (consumers later)
int   ps_r_sf_debug  = 0;       // r_sf_debug — 0 off,1 height,2 slope,3 curvature,4 sky,5 canopy
float ps_r_sf_eps    = 1.5f;    // r_sf_eps — derive finite-difference epsilon (metres)
float ps_r_snow      = 0.f;     // r_snow — TARGET snow coverage 0..1 (Surface Field consumer; whitens by type x slope x sky)
float ps_r_snow_rate = 0.25f;   // r_snow_rate — snow accumulate/melt speed per second (eases the actual amount toward r_snow)
int   ps_r_snow_deform        = 1;      // r_snow_deform — footprint deformation (carve prints into the snow volume)
float ps_r_snow_deform_depth  = 0.14f;  // r_snow_deform_depth — print press depth (m)
float ps_r_snow_deform_radius = 0.22f;  // r_snow_deform_radius — print radius (m)
float ps_r_snow_deform_time   = 60.f;   // r_snow_deform_time — print lifetime (sec); time-decay so the trail lasts ~a minute
int   ps_r_snow_deform_tex    = 1;      // r_snow_deform_tex — dense persistent deform TEXTURE (vk_deform) vs the stamp loop
int   ps_r_snow_mesh          = 0;      // r_snow_mesh — dense player-centred snow surface mesh (VHM-style; needs deform_tex)
float ps_r_snow_berm          = 0.2f;   // r_snow_berm — displaced-snow BERM height around prints (fraction of dent depth)
float ps_r_snow_ripple        = 0.9f;   // r_snow_ripple — wind-ripple/sastrugi relief strength on the open snow (0 = off)
float ps_r_snow_rough         = 0.5f;   // r_snow_rough — trail/print IMPERFECTION (per-print width/depth + wavy edges)

// SSS PUDDLES (SSFX deffer_terrain_high_flat port): the puddle look. Distinct
// procedural puddle bodies (stand-in for SSFX's per-level artist puddles_mask) that
// GROW with the wetness accumulator and RECEDE as it dries; slope-masked to flat
// ground; reflection / drop-ripples / sun glint applied per pixel. r_puddle_sss =
// master; r_puddle_level = coverage (more/larger puddles); r_puddle_scale = size.
int   ps_r_puddle_sss    = 1;
float ps_r_puddle_level  = 0.5f;    // puddle coverage (higher = more/larger puddles, lower = fewer)
// Macro placement scale (procedural stand-in for SSFX's per-level puddles_mask).
// World frequency of the puddle-body noise: bigger = smaller/tighter puddles,
// smaller = broader pools. ~1.0 ≈ 5-6 m puddles. This is what gives DISTINCT
// puddles instead of a uniform wet sheet on levels without an artist mask.
float ps_r_puddle_scale  = 1.0f;

// =========================================================================
// EXPERIMENTAL / PARKED: WATER FLOW SIMULATION (compute, vk_water_sim).
// Off by default — NOT part of the shipped SSS puddle look. A shallow-water field
// on the rain ortho box: rain feeds it, water flows downhill and pools, evaporation
// drains it; when r_water_sim is on it becomes the puddle source + a volumetric
// water render. Kept for a future revisit (SSS+RDR2 water). All r_water_* below.
// =========================================================================
int   ps_r_water_sim    = 0;   // water flow sim OFF by default (experimental; revisit for SSS+RDR2 water)
float ps_r_water_rain   = 0.4f;
// Leak RATE (exponential drain ∝ water amount). Flat ground settles at depth
// ≈ rainRate/leak (shallow, self-limiting — no flood, no knife-edge); dips fed
// by runoff settle deeper. Higher = drier/shallower everywhere. ~3–8 is sane.
// Exponential leak. With the velocity sim the FLOW drains flats into dips, so the
// leak only needs to dry things over time — keep it LOW so water accumulates and
// stays visible (high leak = water evaporates before it can pool/stream).
// Flat-ground equilibrium depth ≈ rain/leak — keep leak high enough that light
// rain leaves flats only DAMP (sub-threshold), while dips collect runoff into
// small visible puddles. Too low = whole ground floods like a downpour.
float ps_r_water_evap   = 3.0f;
// Downhill ACCELERATION (gravity) for the velocity sim — how fast water runs /
// how strongly streams form (replaces the old relaxation factor). Higher = faster.
float ps_r_water_flow   = 0.5f;
int   ps_r_water_iters  = 1;   // legacy (velocity sim runs 1 step/frame)
// Volumetric water render (tonemap raymarch): r_water_murk = absorption per metre
// (higher = murkier / floor hidden sooner / "deeper" feel); r_water_refract =
// how much the surface ripples bend the view of the bottom.
float ps_r_water_murk   = 1.2f;
float ps_r_water_refract = 0.02f;

// PN-triangle silhouette curvature (R4 TESS_PN). OFF by default: it rounds
// hard-surface props (barrels/crates/walls) by inflating along smoothed
// vertex normals — meant for organic curves, looks wrong on world geometry.
// Was previously read UNINITIALIZED (push offset 112 sat outside the 112-byte
// range and was never pushed → garbage curvature warped tessellated props).
float ps_r_tess_pn     = 0.0f;

u32 ps_r_sun_quality = 0;
constexpr xr_token qsun_quality_token[] = {{"st_opt_low", 0},
                                           {"st_opt_medium", 1},
                                           {"st_opt_high", 2},
                                           {"st_opt_ultra", 3},
                                           {"st_opt_extreme", 4},
                                           {nullptr, 0}};

//	“Off”
//	“DX10.0 style [Standard]”
//	“DX10.1 style [Higher quality]”

// Common
extern int psSkeletonUpdate;
extern float r__dtex_range;

Fvector3 ps_r_taa_jitter{};
Fvector2 ps_r_taa_jitter_full{};
float ps_r_cas{};

int ps_r__LightSleepFrames = 100;

float ps_r__WallmarkTTL = 60.f;
float ps_r__WallmarkSHIFT = 0.0001f;
float ps_r__WallmarkSHIFT_V = 0.0001f;

float ps_r__GLOD_ssa_start = 256.f;
float ps_r__GLOD_ssa_end = 64.f;

float ps_r__LOD = 0.5f;
float ps_r__LOD_k = 1.f;

float ps_r__ssaDISCARD = 3.5f; // RO

int ps_r__tf_Anisotropic{16}, ps_r__tf_Anisotropic_SMAP{1};
float ps_r__tf_Mipbias{-0.5f}, ps_r__tf_Mipbias_SMAP{};

// R2
float ps_r2_ssaLOD_A = 64.f;
float ps_r2_ssaLOD_B = 48.f;

// crookr
int scope_fake_enabled = 0;

float scope_fake_power = 0.66f;
float scope_fake_radius = 0;
float scope_fake_interp = 0.15f;

Fvector4 ps_scope1_params = Fvector4().set(0.2f, 1.f, 1.f, 0.0f); // inner blur, outer blur, brightness, ___
Fvector4 ps_scope2_params = Fvector4().set(0.0066f, 0.66f, 0.25f, 0.25f); // chroma abber, fog attack(aim), fog attack(move), fog max travel
Fvector4 ps_scope3_params = Fvector4().set(1.25f, 4.f, 0.0f, 0.0f); // relative fog radius, fog sharpness, ___, ___

// R2-specific
Flags64 ps_r2_ls_flags = {
    R2FLAG_SUN |
    R2FLAG_EXP_DONT_TEST_UNSHADOWED | 
    R3FLAG_DYN_WET_SURF |
    R3FLAG_VOLUMETRIC_SMOKE |
    R2FLAG_DETAIL_BUMP | 
    R2FLAG_SSFX_HEIGHT_FOG |
    R2FLAG_SSFX_BLOOM |
    R2FLAG_STEEP_PARALLAX | 
    R2FLAG_TONEMAP | 
//    R2FLAG_VOLUMETRIC_LIGHTS |
//    R2FLAG_VOLUMETRIC_LIGHTS_BLUR |
    R2FLAG_EXP_MT_RAIN |
    R2FLAG_EXP_MT_RAIN_DRAW |
    R2FLAG_EXP_MT_SUN |
    R2FLAG_EXP_MT_SUN_DRAW |
    R2FLAG_EXP_MT_PARTICLES |
    R2FLAG_EXP_MT_LIGHTS |
    R2FLAG_EXP_MT_BONES
    // | R2FLAG_LIGHT_NO_DIST_SHADOWS //SIMP: по дефолту пусть будет выключено, чтоб волюметрики не вырубались с расстоянием
    | //R2FLAGEXT_ENABLE_TESSELLATION | 
//    R2FLAGEXT_RAIN_DROPS | 
//    R2FLAGEXT_RAIN_DROPS_CONTROL | 
//    R2FLAGEXT_MASK | 
//    R2FLAGEXT_MASK_CONTROL | 
//    R2FLAGEXT_MT_TEXLOAD  |
    R2FLAGEXT_SSLR |
    R2FLAGEXT_SSFX_INTER_GRASS |
    R2FLAGEXT_FONT_SHADOWS
//    | R2FLAGEXT_SSFX_SHADOWS
//    | R2FLAGEXT_SSFX_SSS
    | R2FLAGEXT_SMAP_LOW_LOD
    | R2FLAGEXT_DISABLE_SMAPVIS
};

BOOL ps_no_scale_on_fade = 0; // Alundaio

float ps_r2_df_parallax_h = 0.02f;
float ps_r2_df_parallax_range = 60.f;
float ps_r2_tonemap_middlegray = 1.f; // r2-only
float ps_r2_tonemap_adaptation = 1.f; // r2-only
float ps_r2_tonemap_low_lum = 0.0001f; // r2-only
float ps_r2_tonemap_amount = 0.7f; // r2-only
float ps_r2_ls_bloom_speed = 100.f; // r2-only
float ps_r2_ls_bloom_threshold = .00001f; // r2-only
float ps_r2_mblur = .0f; // .5f
float ps_r2_ls_depth_scale = 1.00001f; // 1.00001f
float ps_r2_ls_depth_bias = -0.00005f; // SSS19 Edited //-0.0003f; // -0.0001f
float ps_r2_ls_squality = 1.0f; // 1.00f
float ps_r2_sun_tsm_bias = -0.01f; //

float ps_r2_sun_depth_far_scale = 1.00000f; // 1.00001f
float ps_r2_sun_depth_near_scale = 1.0000f; // 1.00001f
float ps_r2_sun_lumscale = 1.0f; // 1.0f
float ps_r2_sun_lumscale_hemi = 1.0f; // 1.0f
float ps_r2_sun_lumscale_amb = 1.0f;

float ps_r2_dhemi_sky_scale = 0.08f; // 1.5f
float ps_r2_dhemi_light_scale = 0.2f;
float ps_r2_dhemi_light_flow = 0.1f;

int ps_r2_dhemi_count = 5; // 5

float ps_lens_flare_sun_blend{};

float ps_r2_lt_smooth = 1.f;
float ps_r2_slight_fade = 1.0f;

Fvector4 ps_ssfx_lut{}; // x - интенсивность, y - номер эффекта
Fvector3 ps_ssfx_shadows{
    1024.f, 1536.f,
    0.0f}; // x - Minimum shadow map resolution. When lights are away from the player the resolution of shadows drop to improve performance ( at the cost of image quality ), y -
           // Maximum shadow map resolution. When lights are closer, the resolution increases to improve the image quality of shadows ( at the cost of performance ).
Fvector3 ps_ssfx_shadow_bias{0.4f, 0.03f, 0.0f};

int ps_ssfx_bloom_use_presets = 0;
Fvector4 ps_ssfx_bloom_1 = {4.f, 4.f, 0.f, 0.5f}; // Threshold, Exposure, -, Sky
Fvector4 ps_ssfx_bloom_2 = {1.7f, 0.7f, 0.5f, 0.5f}; // Blur Radius, Vibrance, Lens, Dirt

BOOL ps_ssfx_pom_refine{FALSE}, ps_ssfx_terrain_pom_refine{FALSE};
Fvector4 ps_ssfx_pom = {16, 12, 0.035f, 0.4f}; // Samples , Range, Height, AO
Fvector4 ps_ssfx_terrain_pom{12, 20, 0.04f, 1.0f}; // Samples, Range, Height, Water Limit
Fvector4 ps_ssfx_terrain_offset{};
Fvector4 ps_ssfx_ssr_1{1.f, 0.f, 0.f, 0.f}, ps_ssfx_ssr_2{1.f, 1.f, 0.2f, 0.015f};

// Screen Space Shaders Stuff
Fvector4 ps_ssfx_wind_grass{9.5f, 1.4f, 1.5f, 0.4f}; // Anim Speed, Turbulence, Push, Wave
Fvector4 ps_ssfx_wind_trees{11.0f, 0.15f, 0.5f, 0.1f}; // Branches Speed, Trunk Speed, Bending, Min Wind Speed

Fvector4 ps_ssfx_florafixes_1{0.1f, 0.2f, 0.2f, 0.3f}; // Specular value when the grass is dry, Specular value when the grass is wet, Specular when trees and bushes are dry, Specular when trees and bushes are wet
Fvector4 ps_ssfx_florafixes_2{2.0f, 1.0f, 0.0f, 0.0f}; // Intensity of the flora SubSurface Scattering, How much sun color is added to the flora SubSurface Scattering (1.0 is 100% sun color)

int ps_ssfx_is_underground{};

int ps_ssfx_gloss_method{1};
Fvector3 ps_ssfx_gloss_minmax{0.6f, 0.9f, 0.0f}; // Minimum value of gloss, Maximum value of gloss, Extra gloss to the weapons HUD elements when raining
Fvector4 ps_ssfx_lightsetup_1{0.35f, 0.5f, 1.0f, 1.0f}; // intensity of specular lighting, Porcentage of the specular color. ( 0 = 0% | 1 = 100% ), Automatic adjustment of gloss based on wetness (0 or 1), Value to control the maximum value of gloss when full wetness is reached. ( 0 = 0% | 1 = 100% )
float ps_ssfx_gloss_factor{}; //Управляется из IGame_Persistent::UpdateRainGloss()

Fvector4 ps_ssfx_wetsurfaces_1_cfg{1.5f, 1.4f, 0.7f, 1.25f}; //ripples_size, ripples_speed, ripples_min_speed, ripples_intensity
Fvector4 ps_ssfx_wetsurfaces_2_cfg{1.2f, 1.5f, 0.2f, 0.7f}; // waterfall_size, waterfall_speed, waterfall_min_speed, waterfall_intensity
Fvector4 ps_ssfx_wetsurfaces_1{}, ps_ssfx_wetsurfaces_2{}; // Управляется из IGame_Persistent::UpdateRainGloss()

Fvector4 ps_ssfx_hud_drops_1_cfg{3.0f, 1.f, 1.f, 50.f}; // Quantity of drops, Refrelction intensity, Refraction intensity, Speed of the drops animation
Fvector4 ps_ssfx_hud_drops_2_cfg{50.f, 50.f, 0.75f, 2.f}; // Drops build up speed, Drying speed, Size of the drops, Raindrops gloss intensity
Fvector4 ps_ssfx_hud_drops_1{}, ps_ssfx_hud_drops_2{}; // Значениями этих векторов управляет IGame_Persistent::UpdateHudRaindrops()

Fvector4 ps_ssfx_blood_decals{0.6f, 0.6f, 0.f, 0.f};

Fvector4 ps_ssfx_rain_1{10.0f, 0.02f, 5.f, 2.f}; // Len, Width, Speed, Quality
Fvector4 ps_ssfx_rain_2{0.4f, 0.5f, 5.0f, 1.0f}; // Alpha, Brigthness, Refraction, Reflection
Fvector4 ps_ssfx_rain_3{0.95f, 0.5f, 0.0f, 0.0f}; // Alpha, Refraction ( Splashes )

Fvector3 ps_ssfx_shadow_cascades{25.f, 60.f, 160.f};
Fvector4 ps_ssfx_grass_shadows = {0.0f, 0.0f, 0.0f, 0.0f}; // X - каскады на которых будут рендериться тени (0 - на первом, 1 - на первом и втором, 2 - на всех трёх), Y - устарело и более не используется, Z - дальность на которой будут рендериться тени от источников света (НЕ СОЛНЦА)
Fvector4 ps_ssfx_grass_interactive{1.f, static_cast<float>(GRASS_SHADER_DATA_COUNT), 2000.f, 1.0f};
Fvector4 ps_ssfx_int_grass_params_1{2.0f, 1.0f, 1.0f, 25.f};
Fvector4 ps_ssfx_int_grass_params_2{1.0f, 5.0f, 1.0f, 1.0f};

BOOL ps_ssfx_terrain_grass_align{TRUE}; // Grass align
float ps_ssfx_terrain_grass_slope{1.0f}; // Grass slope limit

float ps_ssfx_wpn_dof_2 = 0.5f;

float ps_r3_dyn_wet_surf_near = 5.f; // 10.0f
float ps_r3_dyn_wet_surf_far = 200.0f; //при 100 при резкой смене погоды видна граница намокшей земли и сухой когда вертишь камеру, но если выставить в районе 300 - намокание может вообще пропасть.

float ps_r2_rain_drops_intensity = 0.00003f;
float ps_r2_rain_drops_speed = 1.25f;

float ps_r2_visor_refl_intensity = 0.39f;
float ps_r2_visor_refl_radius = 0.4f;

int ps_r__detail_radius = 100;

u32 dm_size = 24;
u32 dm_cache1_line = 12; // dm_size*2/dm_cache1_count
u32 dm_cache_line = 49; // dm_size+1+dm_size
u32 dm_cache_size = 2401; // dm_cache_line*dm_cache_line
float dm_fade = 47.5; // float(2*dm_size)-.5f;
u32 dm_current_size = 24;
u32 dm_current_cache1_line = 12; // dm_current_size*2/dm_cache1_count
u32 dm_current_cache_line = 49; // dm_current_size+1+dm_current_size
u32 dm_current_cache_size = 2401; // dm_current_cache_line*dm_current_cache_line
float dm_current_fade = 47.5; // float(2*dm_current_size)-.5f;

float ps_current_detail_density = 0.6;
float ps_current_detail_scale = 1.f;
float ps_r2_no_details_radius = 0.f;
float ps_r2_no_rain_radius = 0.f;

float ps_r2_gloss_factor = 4.0f;

int ps_pnv_mode = 0;

float ps_pnv_noise = 0.15;
float ps_pnv_scanlines = 0.175;
float ps_pnv_scintillation = 0.999;
float ps_pnv_position = 100.0;
float ps_pnv_radius = 0.5;

float ps_pnv_params_1 = 0; // unused
float ps_pnv_params_2 = 1;
float ps_pnv_params_3 = 0.1f;
float ps_pnv_params_4 = 1;

float ps_pnv_params_1_2 = 2;
float ps_pnv_params_2_2 = 0;
float ps_pnv_params_3_2 = 0;
float ps_pnv_params_4_2 = 1;

// textures
int psTextureLOD = 0;

float ps_r2_img_exposure = 1.0f; // r2-only
float ps_r2_img_gamma = 1.0f; // r2-only
float ps_r2_img_saturation = 1.0f; // r2-only

Fvector ps_r2_img_cg{0.5f, 0.5f, 0.5f};

float ps_r__opt_dist = 750.f;

#include "../../xr_3da/xr_ioconsole.h"
#include "../../xr_3da/xr_ioc_cmd.h"

float ps_particle_update_coeff = 0.3f;

// Geometry optimization from Anomaly
int opt_static_geom = 0;
int opt_shadow_geom = 0;

int r_back_buffer_count{2};

// `ps_r_sunshafts_intensity` is owned by xrEngine (Environment_misc.cpp).
extern float ps_r_sunshafts_intensity;

// xrRender_console.cpp expects these as engine-side externs. In OGSR's R4
// build they're defined in xrRender lib, which we don't link. Provide local
// definitions so console registration resolves.
// psSkeletonUpdate is now owned by OGSR's xrRender/SkeletonCustom.cpp (compiled
// via vk_SkeletonCustom.cpp); defining it here too caused LNK4006. Use the extern.
float r__dtex_range    = 50.f;


//-----------------------------------------------------------------------
class CCC_detail_radius : public CCC_Integer
{
public:
    void apply()
    {
        dm_current_size = iFloor((float)ps_r__detail_radius / 4) * 2;
        dm_current_cache1_line = dm_current_size * 2 / 4; // assuming cache1_count = 4
        dm_current_cache_line = dm_current_size + 1 + dm_current_size;
        dm_current_cache_size = dm_current_cache_line * dm_current_cache_line;
        dm_current_fade = float(2 * dm_current_size) - .5f;
    }

    CCC_detail_radius(LPCSTR N, int* V, int _min = 0, int _max = 999) : CCC_Integer(N, V, _min, _max){};
    virtual void Execute(LPCSTR args)
    {
        CCC_Integer::Execute(args);
        apply();
    }
    virtual void Status(TStatus& S) { CCC_Integer::Status(S); }
};

// Vulkan stub — R4 version pings RImplementation.Details->need_init.
// Vulkan grass system reseeds itself on the next CPU update tick anyway.
class CCC_detail_reset : public CCC_Float
{
public:
    CCC_detail_reset(LPCSTR N, float* V, float _min = 0, float _max = 1) : CCC_Float(N, V, _min, _max){}
};

// Vulkan stub — no model-pool dump (models live differently in vk_loader).
class CCC_ModelPoolStat : public IConsole_Command
{
public:
    CCC_ModelPoolStat(LPCSTR N) : IConsole_Command(N) { bEmptyArgsHandled = TRUE; };
    virtual void Execute(LPCSTR /*args*/) { Msg("[VK] stat_models — not implemented"); }
};

class CCC_Preset : public CCC_Token
{
public:
    CCC_Preset(LPCSTR N, u32* V, const xr_token* T) : CCC_Token(N, V, T){};

    virtual void Execute(LPCSTR args)
    {
        CCC_Token::Execute(args);
        string_path _cfg;
        string_path cmd;

        switch (*value)
        {
        case 0: xr_strcpy(_cfg, "rspec_minimum.ltx"); break;
        case 1: xr_strcpy(_cfg, "rspec_low.ltx"); break;
        case 2: xr_strcpy(_cfg, "rspec_default.ltx"); break;
        case 3: xr_strcpy(_cfg, "rspec_high.ltx"); break;
        case 4: xr_strcpy(_cfg, "rspec_extreme.ltx"); break;
        }
        FS.update_path(_cfg, fsgame::game_configs, _cfg);
        strconcat(sizeof(cmd), cmd, "cfg_load", " ", _cfg);
        Console->Execute(cmd);
    }
};

// Vulkan stub — VMA owns all memory; expose a dump via vmaCalculateStatistics
// in a follow-up pass if needed. R4 version queries D3DPOOL_* counters.
class CCC_VideoMemoryStats : public IConsole_Command
{
public:
    CCC_VideoMemoryStats(LPCSTR N) : IConsole_Command(N) { bEmptyArgsHandled = true; };
    virtual void Execute(LPCSTR /*args*/) { Msg("[VK] video_memory_stats — not implemented"); }
};

// `vk_perf` — force the global profiler to print an immediate [VK Perf] MARK
// snapshot (per-pass GPU/CPU ms + VRAM + rain/wet state) the instant it's typed.
// Type it the moment FPS sits to capture exactly that frame's breakdown.
class CCC_VkPerf : public IConsole_Command
{
public:
    CCC_VkPerf(LPCSTR N) : IConsole_Command(N) { bEmptyArgsHandled = TRUE; }
    virtual void Execute(LPCSTR /*args*/) { VK::Prof::RequestMark(); }
};

// Vulkan stub — R4 dumps Models + Resources. Vulkan models/resources are
// elsewhere; not wired here yet.
class CCC_DumpResources : public IConsole_Command
{
public:
    CCC_DumpResources(LPCSTR N) : IConsole_Command(N) { bEmptyArgsHandled = TRUE; };
    virtual void Execute(LPCSTR /*args*/) { Msg("[VK] dump_resources — not implemented"); }
};

class CCC_SunshaftsIntensity : public CCC_Float
{
public:
    CCC_SunshaftsIntensity(LPCSTR N, float* V, float _min, float _max) : CCC_Float(N, V, _min, _max)
    {
        SetCanSave(FALSE);
    }
};

// Particle export/import + debug dumps live in R4's PSLibrary; nothing in
// our Vulkan slice owns them yet. Keep the command names registered so console
// scripts don't crash when calling them; bodies are no-ops.
class CCC_PART_Export : public IConsole_Command
{
public:
    CCC_PART_Export(LPCSTR N) : IConsole_Command(N) { bEmptyArgsHandled = TRUE; };
    virtual void Execute(LPCSTR /*args*/) { Msg("[VK] particles_export — not implemented"); }
};
class CCC_PART_Import : public IConsole_Command
{
public:
    CCC_PART_Import(LPCSTR N) : IConsole_Command(N) { bEmptyArgsHandled = TRUE; };
    virtual void Execute(LPCSTR /*args*/) { Msg("[VK] particles_import — not implemented"); }
};
class CCC_PART_DumpTextures : public IConsole_Command
{
public:
    CCC_PART_DumpTextures(LPCSTR N) : IConsole_Command(N) { bEmptyArgsHandled = TRUE; };
    virtual void Execute(LPCSTR /*args*/) { Msg("[VK] particles_dump_textures — not implemented"); }
};
class CCC_Dbg_DumpStaticVisual : public IConsole_Command
{
public:
    CCC_Dbg_DumpStaticVisual(LPCSTR N) : IConsole_Command(N) { bEmptyArgsHandled = TRUE; };
    virtual void Execute(LPCSTR /*args*/) { Msg("[VK] dbg_dump_static_at_look — not implemented"); }
};
class CCC_OCC_Enable : public CCC_Bool
{
    BOOL v{TRUE};
public:
    CCC_OCC_Enable(LPCSTR N) : CCC_Bool(N, &v) {}
    // Hardware occlusion query toggle is R4-specific; keep value in `v` so
    // future Vulkan HZB code can read it without re-wiring the console.
};

void xrRender_initconsole()
{
    if (!FS.path_exist(fsgame::game_weathers))
    {
        ps_r2_ls_bloom_threshold = 1.0f;
        ps_r2_gloss_factor = 1.0f;
    }

    CMD3(CCC_Preset, "_preset", &ps_preset, qpreset_token);

    CMD4(CCC_Integer, "rs_skeleton_update", &psSkeletonUpdate, 2, 128);

#ifdef DEBUG
    CMD1(CCC_DumpResources, "dump_resources");
#endif //	 DEBUG

    //CMD4(CCC_Float, "r__dtex_range", &r__dtex_range, 5, 175);

    //CMD4(CCC_Integer, "r_smapvis_sleep_frames", &ps_r__LightSleepFrames, 15, 1000);

#ifdef DEBUG
    CMD4(CCC_Float, "r__wallmark_shift_pp", &ps_r__WallmarkSHIFT, 0.0f, 1.f);
    CMD4(CCC_Float, "r__wallmark_shift_v", &ps_r__WallmarkSHIFT_V, 0.0f, 1.f);
#endif // DEBUG

    CMD1(CCC_ModelPoolStat, "stat_models");

    CMD4(CCC_Float, "r__wallmark_ttl", &ps_r__WallmarkTTL, 1.0f, 10.f * 60.f);

    CMD4(CCC_Float, "r__geometry_lod", &ps_r__LOD, 0.1f, 1.0f);
    CMD4(CCC_Float, "r__lod_k", &ps_r__LOD_k, 0.1f, 10.f);

    CMD4(CCC_detail_radius, "r__detail_radius", &ps_r__detail_radius, 70, 300);
    CMD4(CCC_detail_reset, "r__detail_density", &ps_current_detail_density, 0.2f, 0.9f);
    CMD4(CCC_detail_reset, "r__detail_scale", &ps_current_detail_scale, 0.7f, 1.5f);

    CMD4(CCC_Float, "r2_no_details_radius", &ps_r2_no_details_radius, 0.f, 5.f);
    CMD4(CCC_Float, "r2_no_rain_radius", &ps_r2_no_rain_radius, 0.f, 5.f);

    CMD4(CCC_Integer, "r__tf_aniso", &ps_r__tf_Anisotropic, 1, 16);
    CMD4(CCC_Integer, "r__tf_aniso_smap", &ps_r__tf_Anisotropic_SMAP, 1, 16);
    CMD4(CCC_Float, "r__tf_mipbias", &ps_r__tf_Mipbias, -3.f, +3.f);
    CMD4(CCC_Float, "r__tf_mipbias_smap", &ps_r__tf_Mipbias_SMAP, -3.f, +3.f);

    CMD4(CCC_Float, "r2_ssa_lod_a", &ps_r2_ssaLOD_A, 16, 192);
    CMD4(CCC_Float, "r2_ssa_lod_b", &ps_r2_ssaLOD_B, 32, 128);

    CMD4(CCC_Float, "r__ssa_glod_start", &ps_r__GLOD_ssa_start, 128, 1024);
    CMD4(CCC_Float, "r__ssa_glod_end", &ps_r__GLOD_ssa_end, 16, 256);

    CMD4(CCC_Float, "r2_ssa_discard", &ps_r__ssaDISCARD, 0.5f, 10);

    CMD3(CCC_Mask64, "r2_tonemap", &ps_r2_ls_flags, R2FLAG_TONEMAP);
    // NOTE: the four r2_tonemap_* knobs below are INERT in the Vulkan renderer.
    // The VK tonemap (vk_pass_tonemap.cpp) uses fixed exposure constants
    // (kMiddleGray/kLowLum/kExpMin/kExpMax) and is instantaneous (no temporal
    // eye-adaptation), so middlegray/lowlum/adaptation/amount have no effect.
    // They are kept registered only so the options menu (ui_mm_opt.xml) and DX-R4
    // parity don't error when setting them. The LIVE image controls are
    // r2_img_exposure / r2_img_saturation / r2_img_gamma / r2_img_cg_*.
    CMD4(CCC_Float, "r2_tonemap_middlegray", &ps_r2_tonemap_middlegray, 0.0f, 2.0f);
    CMD4(CCC_Float, "r2_tonemap_adaptation", &ps_r2_tonemap_adaptation, 0.01f, 10.0f);
    CMD4(CCC_Float, "r2_tonemap_lowlum", &ps_r2_tonemap_low_lum, 0.0001f, 1.0f);
    CMD4(CCC_Float, "r2_tonemap_amount", &ps_r2_tonemap_amount, 0.0000f, 1.0f);

    CMD4(CCC_Float, "r2_ls_bloom_threshold", &ps_r2_ls_bloom_threshold, 0.f, 1.f);
    CMD4(CCC_Float, "r2_ls_bloom_speed", &ps_r2_ls_bloom_speed, 0.f, 100.f);

    CMD4(CCC_Float, "r2_ls_squality", &ps_r2_ls_squality, .5f, 3.f);

    //- Mad Max
    CMD4(CCC_Float, "r2_gloss_factor", &ps_r2_gloss_factor, .0f, 10.f);
    //- Mad Max

    // CMD3(CCC_Mask, "r_taa_jitter_enable", &ps_r2_ls_flags, R2FLAG_DBG_TAA_JITTER_ENABLE);

    CMD3(CCC_Mask64, "r2_disable_hom", &ps_r2_ls_flags_ext, R2FLAGEXT_DISABLE_HOM);
    CMD3(CCC_Mask64, "r2_disable_particles", &ps_r2_ls_flags_ext, R2FLAGEXT_DISABLE_PARTICLES);
    CMD3(CCC_Mask64, "r2_disable_dynamic", &ps_r2_ls_flags_ext, R2FLAGEXT_DISABLE_DYNAMIC);
    CMD3(CCC_Mask64, "r2_disable_light", &ps_r2_ls_flags_ext, R2FLAGEXT_DISABLE_LIGHT);
    //CMD3(CCC_Mask64, "r2_disable_smapvis", &ps_r2_ls_flags_ext, R2FLAGEXT_DISABLE_SMAPVIS);
    CMD3(CCC_Mask64, "r2_disable_sectors", &ps_r2_ls_flags_ext, R2FLAGEXT_DISABLE_SECTORS);

    CMD3(CCC_Mask64, "r2_disable_static_normal", &ps_r2_ls_flags_ext, R2FLAGEXT_DISABLE_STATIC_NORMAL);
    CMD3(CCC_Mask64, "r2_disable_static_lod", &ps_r2_ls_flags_ext, R2FLAGEXT_DISABLE_STATIC_LOD);
    CMD3(CCC_Mask64, "r2_disable_static_progressive", &ps_r2_ls_flags_ext, R2FLAGEXT_DISABLE_STATIC_PROGRESSIVE);
    CMD3(CCC_Mask64, "r2_disable_static_tree", &ps_r2_ls_flags_ext, R2FLAGEXT_DISABLE_STATIC_TREE);
    CMD3(CCC_Mask64, "r2_disable_static_tree_progressive", &ps_r2_ls_flags_ext, R2FLAGEXT_DISABLE_STATIC_TREE_PROGRESSIVE);

    // CMD3(CCC_Mask64, "r2_render_on_prefetch", &ps_r2_ls_flags_ext, R2FLAGEXT_RENDER_ON_PREFETCH);

    CMD3(CCC_Mask64, "r2_smap_low_lod", &ps_r2_ls_flags_ext, R2FLAGEXT_SMAP_LOW_LOD);

    CMD3(CCC_Mask64, "r2_rain_drops", &ps_r2_ls_flags_ext, R2FLAGEXT_RAIN_DROPS);
    CMD3(CCC_Mask64, "r2_rain_drops_control", &ps_r2_ls_flags_ext, R2FLAGEXT_RAIN_DROPS_CONTROL);
    CMD4(CCC_Float, "r2_rain_drops_intensity", &ps_r2_rain_drops_intensity, 0.f, 0.0001f);
    CMD4(CCC_Float, "r2_rain_drops_speed", &ps_r2_rain_drops_speed, 0.8f, 5.f);

    CMD3(CCC_Mask64, "r2_mask", &ps_r2_ls_flags_ext, R2FLAGEXT_MASK);
    CMD3(CCC_Mask64, "r2_mask_control", &ps_r2_ls_flags_ext, R2FLAGEXT_MASK_CONTROL);

    CMD3(CCC_Mask64, "r_sslr_enable", &ps_r2_ls_flags_ext, R2FLAGEXT_SSLR);

    CMD3(CCC_Mask64, "r_terrain_parallax_enable", &ps_r2_ls_flags_ext, R2FLAGEXT_TERRAIN_PARALLAX);

    CMD3(CCC_Mask64, "r_mt_texload", &ps_r2_ls_flags_ext, R2FLAGEXT_MT_TEXLOAD);

    // Солнце на этом рендере отключать нельзя, могут появиться куча различных графических багов.
    //CMD3(CCC_Mask64, "r2_sun", &ps_r2_ls_flags, R2FLAG_SUN);
    CMD3(CCC_Mask64, "r2_sun_details", &ps_r2_ls_flags, R2FLAG_SUN_DETAILS);

    CMD3(CCC_Mask64, "r2_light_details", &ps_r2_ls_flags, R2FLAG_LIGHT_DETAILS);

    CMD3(CCC_Mask64, "r2_exp_far_no_shadows", &ps_r2_ls_flags, R2FLAG_LIGHT_NO_DIST_SHADOWS);

    CMD4(CCC_Float, "r2_sun_tsm_bias", &ps_r2_sun_tsm_bias, -0.5, +0.5);

    CMD3(CCC_Token, "r__smap_cascade0_size", &r2_SmapCascade0Size, CascadesSmapSizeToken);
    // CMD3(CCC_Token, "r__smap_cascade1_size", &r2_SmapCascade1Size, CascadesSmapSizeToken);
    CMD3(CCC_Token, "r__smap_cascade2_size", &r2_SmapCascade2Size, CascadesSmapSizeToken);
    CMD3(CCC_Token, "r__smap_lights_size", &r2_SmapLightsSize, LightsSmapSizeToken);
    CMD3(CCC_Token, "r__smap_rain_size", &r2_SmapRainSize, RainSmapSizeToken);

    CMD4(CCC_Float, "r2_sun_depth_far_scale", &ps_r2_sun_depth_far_scale, 0.5, 1.5);
    CMD4(CCC_Float, "r2_sun_depth_near_scale", &ps_r2_sun_depth_near_scale, 0.5, 1.5);
    CMD4(CCC_Float, "r2_sun_lumscale", &ps_r2_sun_lumscale, -1.0, +3.0);
    CMD4(CCC_Float, "r2_sun_lumscale_hemi", &ps_r2_sun_lumscale_hemi, 0.0, +3.0);
    CMD4(CCC_Float, "r2_sun_lumscale_amb", &ps_r2_sun_lumscale_amb, 0.0, +3.0);

    CMD4(CCC_Float, "r2_mblur", &ps_r2_mblur, 0.0f, 1.0f);
    CMD3(CCC_Mask64, "r2_mblur_enable", &ps_r2_ls_flags_ext, R2FLAGEXT_MOTION_BLUR);

    // Shader param stuff
    constexpr Fvector4 tw2_min{-100.f, -100.f, -100.f, -100.f};
    constexpr Fvector4 tw2_max{100.f, 100.f, 100.f, 100.f};

    // CMD4(CCC_Integer, "r2_dhemi_count", &ps_r2_dhemi_count, 4, 25);
    // CMD4(CCC_Float, "r2_dhemi_sky_scale", &ps_r2_dhemi_sky_scale, 0.0f, 100.f);
    // CMD4(CCC_Float, "r2_dhemi_light_scale", &ps_r2_dhemi_light_scale, 0, 100.f);
    // CMD4(CCC_Float, "r2_dhemi_light_flow", &ps_r2_dhemi_light_flow, 0, 1.f);
    // CMD4(CCC_Float, "r2_dhemi_smooth", &ps_r2_lt_smooth, 0.f, 10.f);
    // CMD3(CCC_Mask, "rs_hom_depth_draw", &ps_r2_ls_flags_ext, R2FLAGEXT_HOM_DEPTH_DRAW);

    CMD3(CCC_Mask64, "r2_shadow_cascede_zcul", &ps_r2_ls_flags_ext, R2FLAGEXT_SUN_ZCULLING);

    CMD4(CCC_Float, "r2_ls_depth_scale", &ps_r2_ls_depth_scale, 0.5, 1.5);
    CMD4(CCC_Float, "r2_ls_depth_bias", &ps_r2_ls_depth_bias, -0.5, +0.5);

    CMD4(CCC_Float, "r2_parallax_h", &ps_r2_df_parallax_h, .0f, .5f);
    CMD4(CCC_Float, "r2_parallax_range", &ps_r2_df_parallax_range, 5.0f, 175.0f);

    CMD4(CCC_Float, "r2_slight_fade", &ps_r2_slight_fade, .2f, 2.f);

    CMD3(CCC_Mask64, "r_mt_sun", &ps_r2_ls_flags, R2FLAG_EXP_MT_SUN);
    CMD3(CCC_Mask64, "r_mt_sun_draw", &ps_r2_ls_flags, R2FLAG_EXP_MT_SUN_DRAW);
    CMD3(CCC_Mask64, "r_mt_rain", &ps_r2_ls_flags, R2FLAG_EXP_MT_RAIN);
    CMD3(CCC_Mask64, "r_mt_rain_draw", &ps_r2_ls_flags, R2FLAG_EXP_MT_RAIN_DRAW);
    CMD3(CCC_Mask64, "r_mt_particles", &ps_r2_ls_flags, R2FLAG_EXP_MT_PARTICLES);
    CMD3(CCC_Mask64, "r_mt_lights", &ps_r2_ls_flags, R2FLAG_EXP_MT_LIGHTS);
    CMD3(CCC_Mask64, "r_mt_bones", &ps_r2_ls_flags, R2FLAG_EXP_MT_BONES);

    CMD3(CCC_Mask64, "r2_volumetric_lights", &ps_r2_ls_flags, R2FLAG_VOLUMETRIC_LIGHTS);

    // Sunshafts
    CMD3(CCC_Token, "r_sunshafts_mode", &ps_r_sunshafts_mode, sunshafts_mode_token);
    CMD4(CCC_SunshaftsIntensity, "r_sunshafts_intensity", &ps_r_sunshafts_intensity, 0.0f, 5.0f); // Dbg

    CMD4(CCC_Float, "r_ss_sunshafts_length", &ps_r_ss_sunshafts_length, 0.2f, 1.5f);
    CMD4(CCC_Float, "r_ss_sunshafts_radius", &ps_r_ss_sunshafts_radius, 0.5f, 2.f);

    // CMD4(CCC_Float, "r_SunShafts_SampleStep_Phase1", &ps_r_prop_ss_sample_step_phase0, 0.01f, 0.2f);
    // CMD4(CCC_Float, "r_SunShafts_SampleStep_Phase2", &ps_r_prop_ss_sample_step_phase1, 0.01f, 0.2f);
    CMD4(CCC_Float, "r_SunShafts_Radius", &ps_r_prop_ss_radius, 0.5f, 2.0f);
    CMD4(CCC_Float, "r_SunShafts_Blend", &ps_r_prop_ss_blend, 0.01f, 1.0f);

    CMD3(CCC_Token, "r_ao_mode", &ps_r_ao_mode, ao_mode_token);
    CMD3(CCC_Token, "r2_ssao", &ps_r_ao_quality, qssao_token);
    CMD4(CCC_Integer, "r_ssao", &ps_r_ssao_enable, 0, 1);        // global GTAO on/off (perf A/B + real off)
    CMD4(CCC_Integer, "r_ssao_npc_normals", &ps_r_ssao_npc_normals, 0, 1); // NPC normal G-buffer for GTAO (A/B)
    CMD4(CCC_Integer, "r_ssao_debug", &ps_r_ssao_debug, 0, 3);   // 1=AO map, 2=depth view, 3=normal view
    CMD4(CCC_Float, "r_ssao_strength", &ps_r_ssao_strength, 0.f, 4.f);
    CMD4(CCC_Integer, "r_ssil", &ps_r_ssil_enable, 0, 1);        // SSIL (folded into GTAO) on/off (A/B; needs r_ssao on)
    CMD4(CCC_Integer, "r_ssil_debug", &ps_r_ssil_debug, 0, 1);   // 1 = show ONLY the indirect bounce field
    CMD4(CCC_Float, "r_ssil_strength", &ps_r_ssil_strength, 0.f, 8.f);  // IL intensity multiplier
    CMD4(CCC_Integer, "r_motion_vectors", &ps_r_motion_vectors, 0, 1);  // screen-space MV pass on/off (DLSS/FSR/PT foundation)
    CMD4(CCC_Integer, "r_mv_debug", &ps_r_mv_debug, 0, 1);              // false-colour MV overlay (grey=still, R=+x, G=+y)
    CMD4(CCC_Float, "r_mv_debug_scale", &ps_r_mv_debug_scale, 1.f, 500.f); // MV overlay magnitude scale
    CMD4(CCC_Float, "r_wind_tree_bend", &ps_r_wind_tree_bend, 0.f, 2.f);   // tree trunk sway intensity (0 = rigid)
    CMD4(CCC_Float, "r_wind_tree_anim", &ps_r_wind_tree_anim, 0.f, 40.f);  // tree branch/leaf flutter speed
    CMD4(CCC_Float, "r_wind_tree_trunk", &ps_r_wind_tree_trunk, 0.f, 2.f); // tree trunk anim speed
    CMD4(CCC_Float, "r_wind_tree_flutter", &ps_r_wind_tree_flutter, 0.f, 16.f); // crown/leaf flutter amplitude
    CMD4(CCC_Float, "r_wind_tree_crown", &ps_r_wind_tree_crown, 0.f, 30.f);     // height where leaf flutter fades in
    CMD4(CCC_Float, "r_wind_shadow_dist", &ps_r_wind_shadow_dist, 0.f, 160.f);  // tree shadow wind radius (0 = all static)
    CMD4(CCC_Integer, "r_vsm_tree_wind", &ps_r_vsm_tree_wind, 0, 1);            // TEST: wind in VSM tree shadow pages
    CMD4(CCC_Float, "r_sun_boost", &ps_r_sun_boost, 0.f, 4.f);
    CMD4(CCC_Float, "r_ambient_floor", &ps_r_ambient_floor, 0.f, 0.5f);
    CMD4(CCC_Float, "r_wet_darken", &ps_r_wet_darken, 0.f, 1.f);
    CMD4(CCC_Float, "r_wet_refl", &ps_r_wet_refl, 0.f, 3.f);
    CMD4(CCC_Integer, "r_wet_debug", &ps_r_wet_debug, 0, 1);
    CMD4(CCC_Integer, "r_rain_debug", &ps_r_rain_debug, 0, 1);
    CMD4(CCC_Integer, "r_rain", &ps_r_rain_enable, 0, 1);   // master rain on/off (effect only, not weather)

    // Global render profiler (vk_profiler): r_profiler 0/1/2, vk_perf = MARK dump.
    CMD4(CCC_Integer, "r_profiler", &ps_r_profiler, 0, 2);
    CMD1(CCC_VkPerf,  "vk_perf");

    // Variable Rate Shading (vk_vrs): 0 off / 1 mild / 2 aggressive + distance thresholds.
    CMD4(CCC_Integer, "r_vrs", &ps_r_vrs, 0, 2);
    CMD4(CCC_Float, "r_vrs_near", &ps_r_vrs_near, 0.f, 300.f);
    CMD4(CCC_Float, "r_vrs_far",  &ps_r_vrs_far,  0.f, 500.f);

    // Frustum culling of world statics (A/B with r_cull 0).
    CMD4(CCC_Integer, "r_cull", &ps_r_cull, 0, 1);

    // GPU-driven sun shadow casters (vk_shadow_gpu) — A/B with r_gpu_shadows 0.
    CMD4(CCC_Integer, "r_gpu_shadows", &ps_r_gpu_shadows, 0, 1);

    // caster-LOD for GPU shadows: coarse geometry for distant casters (A/B with r_shadow_lod 0).
    CMD4(CCC_Integer, "r_shadow_lod", &ps_r_shadow_lod, 0, 1);
    CMD4(CCC_Float, "r_shadow_lod_dist", &ps_r_shadow_lod_dist, 5.0f, 200.0f);

    // Cascade static-map cache (A/B with r_shadow_casc_cache 0 = old per-frame raster);
    // r_shadow_casc_sun = sun-rotation degrees that forces a redraw (smaller = smoother).
    CMD4(CCC_Integer, "r_shadow_casc_cache", &ps_r_shadow_casc_cache, 0, 1);
    CMD4(CCC_Float,   "r_shadow_casc_sun",   &ps_r_shadow_casc_sun,   0.005f, 1.0f);

    // Virtual Shadow Maps (WIP, default OFF). r_vsm_base = clipmap detail (m, smaller=sharper).
    CMD4(CCC_Integer, "r_vsm",          &ps_r_vsm,          0, 1);
    CMD4(CCC_Integer, "r_vsm_debug",    &ps_r_vsm_debug,    0, 1);
    CMD4(CCC_Float,   "r_vsm_base",     &ps_r_vsm_base,     8.0f, 64.0f);
    CMD4(CCC_Float,   "r_vsm_bias",     &ps_r_vsm_bias,     0.0f, 0.02f);
    CMD4(CCC_Integer, "r_vsm_temporal", &ps_r_vsm_temporal, 0, 1);
    CMD4(CCC_Float,   "r_vsm_ta_blend", &ps_r_vsm_ta_blend, 0.0f, 0.98f);
    CMD4(CCC_Integer, "r_vsm_grass",      &ps_r_vsm_grass,      0, 1);
    CMD4(CCC_Float,   "r_vsm_grass_dist", &ps_r_vsm_grass_dist, 4.0f, 48.0f);
    CMD4(CCC_Float,   "r_vsm_npc_dist",   &ps_r_vsm_npc_dist,   0.0f, 500.0f);
    CMD4(CCC_Float,   "r_vsm_lod_dist",   &ps_r_vsm_lod_dist,   0.0f, 500.0f);
    CMD4(CCC_Integer, "r_vsm_mark_half",  &ps_r_vsm_mark_half,  0, 1);
    CMD4(CCC_Integer, "r_vsm_cache",      &ps_r_vsm_cache,      0, 1);
    CMD4(CCC_Float,   "r_vsm_cache_sun",  &ps_r_vsm_cache_sun,  0.0f, 5.0f);
    CMD4(CCC_Float,   "r_vsm_cache_rot",  &ps_r_vsm_cache_rot,  0.0f, 90.0f);
    CMD4(CCC_Integer, "r_vsm_cache_refresh", &ps_r_vsm_cache_refresh, 1, 64);

    // GPU-driven world forward pass (vk_world_gpu) — A/B with r_gpu_world 0.
    CMD4(CCC_Integer, "r_gpu_world", &ps_r_gpu_world, 0, 1);

    // Clustered forward / Forward+ (vk_clustered) — A/B with r_clustered 0.
    // r_clustered_debug 1 = per-cluster light-count heatmap on the world.
    CMD4(CCC_Integer, "r_clustered",       &ps_r_clustered,       0, 1);
    CMD4(CCC_Integer, "r_clustered_debug", &ps_r_clustered_debug, 0, 1);

    // Froxel volumetric lighting (vk_volumetrics) — P1: sun god rays + depth fog.
    CMD4(CCC_Integer, "r_vol",           &ps_r_vol,           0, 1);
    CMD4(CCC_Float,   "r_vol_density",   &ps_r_vol_density,   0.0f, 1.0f);
    CMD4(CCC_Float,   "r_vol_height",    &ps_r_vol_height,    0.0f, 2.0f);
    CMD4(CCC_Float,   "r_vol_g",         &ps_r_vol_g,         0.0f, 0.95f);
    CMD4(CCC_Float,   "r_vol_intensity", &ps_r_vol_intensity, 0.0f, 8.0f);
    CMD4(CCC_Float,   "r_vol_amb",       &ps_r_vol_amb,       0.0f, 1.0f);
    CMD4(CCC_Float,   "r_vol_indoor",    &ps_r_vol_indoor,    0.0f, 16.0f);
    CMD4(CCC_Float,   "r_vol_sun",       &ps_r_vol_sun,       0.0f, 16.0f);
    CMD4(CCC_Float,   "r_vol_lights",    &ps_r_vol_lights,    0.0f, 16.0f);
    CMD4(CCC_Float,   "r_vol_smoke",     &ps_r_vol_smoke,     0.0f, 16.0f);
    CMD4(CCC_Float,   "r_vol_smoke_clamp", &ps_r_vol_smoke_clamp, 0.0f, 64.0f);
    CMD4(CCC_Float,   "r_vol_smoke_inject",  &ps_r_vol_smoke_inject,  0.0f, 16.0f);
    CMD4(CCC_Float,   "r_vol_smoke_density", &ps_r_vol_smoke_density, 0.0f, 16.0f);
    CMD4(CCC_Integer, "r_vol_smoke_debug",   &ps_r_vol_smoke_debug,   0, 3);
    CMD4(CCC_Float,   "r_vol_smoke_dist_full", &ps_r_vol_smoke_dist_full, 0.0f, 300.0f);
    CMD4(CCC_Float,   "r_vol_smoke_dist",    &ps_r_vol_smoke_dist,    0.0f, 300.0f);
    CMD4(CCC_Float,   "r_vol_smoke_footprint", &ps_r_vol_smoke_footprint, 0.0f, 3.0f);
    CMD4(CCC_Float,   "r_vol_smoke_shadow",  &ps_r_vol_smoke_shadow,  0.0f, 8.0f);
    CMD4(CCC_Float,   "r_vol_smoke_shadow_step", &ps_r_vol_smoke_shadow_step, 0.05f, 4.0f);
    CMD4(CCC_Float,   "r_vol_noise",       &ps_r_vol_noise,       0.0f, 1.0f);
    CMD4(CCC_Float,   "r_vol_noise_scale", &ps_r_vol_noise_scale, 0.02f, 2.0f);
    CMD4(CCC_Float,   "r_vol_noise_speed", &ps_r_vol_noise_speed, 0.0f, 1.0f);
    CMD4(CCC_Float,   "r_vol_soft",      &ps_r_vol_soft,      0.5f, 8.0f);
    CMD4(CCC_Integer, "r_vol_ta",        &ps_r_vol_ta,        0, 1);
    CMD4(CCC_Float,   "r_vol_ta_blend",  &ps_r_vol_ta_blend,  0.0f, 0.98f);
    CMD4(CCC_Integer, "r_vol_shadow",    &ps_r_vol_shadow,    0, 1);
    CMD4(CCC_Integer, "r_vol_debug",     &ps_r_vol_debug,     0, 1);

    // Dynamic-light terrain/static occlusion (ground-height map + per-light march).
    CMD4(CCC_Integer, "r_light_occ", &ps_r_light_occ, 0, 1);

    // World heightmap tessellation (live, no restart) — see vk_render_queue.cpp.
    CMD4(CCC_Float, "r_tess", &ps_r_tess, 0.f, 1.f);
    CMD4(CCC_Float, "r_tess_max", &ps_r_tess_max, 1.f, 64.f);
    CMD4(CCC_Float, "r_tess_near", &ps_r_tess_near, 0.f, 50.f);
    CMD4(CCC_Float, "r_tess_far", &ps_r_tess_far, 1.f, 100.f);
    CMD4(CCC_Float, "r_tess_height", &ps_r_tess_height, 0.f, 4.f);
    CMD4(CCC_Float, "r_tess_pn", &ps_r_tess_pn, 0.f, 1.f);
    CMD4(CCC_Integer, "r_pom", &ps_r_pom, 0, 1);
    CMD4(CCC_Float, "r_pom_height", &ps_r_pom_height, 0.f, 0.1f);
    CMD4(CCC_Float, "r_pom_steps", &ps_r_pom_steps, 8.f, 64.f);
    CMD4(CCC_Float, "r_pom_far", &ps_r_pom_far, 1.f, 50.f);
    CMD4(CCC_Float, "r_pom_blur", &ps_r_pom_blur, 0.f, 4.f);
    CMD4(CCC_Float, "r_pom_normal", &ps_r_pom_normal, 0.f, 4.f);
    CMD4(CCC_Float, "r_pom_shadow", &ps_r_pom_shadow, 0.f, 3.f);
    CMD4(CCC_Float, "r_pom_ao", &ps_r_pom_ao, 0.f, 3.f);
    CMD4(CCC_Integer, "r_pom_debug", &ps_r_pom_debug, 0, 1);
    CMD4(CCC_Integer, "r_ao_flat", &ps_r_ao_flat, 0, 1);   // debug: kill ALL ambient occlusion (incl. baked lmap)
    CMD4(CCC_Float, "r_pom_ceil", &ps_r_pom_ceil, 0.f, 1.f);     // POM strength on ceilings (down-facing)
    CMD4(CCC_Float, "r_pom_floor", &ps_r_pom_floor, 0.f, 1.f);   // POM strength on floors (up-facing)
    CMD4(CCC_Integer, "r_pom_terrain", &ps_r_pom_terrain, 0, 1); // terrain POM (experimental, default off)
    CMD4(CCC_Float, "r_terrain_normal", &ps_r_terrain_normal, 0.f, 3.f); // terrain detail normal-mapping strength
    CMD4(CCC_Float, "r_terrain_ao", &ps_r_terrain_ao, 0.f, 1.f);         // terrain micro contact AO strength
    CMD4(CCC_Integer, "r_terrain_debug", &ps_r_terrain_debug, 0, 3);     // 0 off,1 normal,2 AO,3 height
    CMD4(CCC_Float, "r_terrain_gloss", &ps_r_terrain_gloss, 0.f, 2.f);   // terrain dry sun-gloss strength
    CMD4(CCC_Integer, "r_puddle_debug", &ps_r_puddle_debug, 0, 2);       // 0 off, 1 coverage, 2 micro-height/flow
    CMD4(CCC_Integer, "r_puddle_sss", &ps_r_puddle_sss, 0, 1);           // SSS puddles (default puddle source)
    CMD4(CCC_Float, "r_puddle_level", &ps_r_puddle_level, 0.f, 1.f);     // puddle coverage (more/larger puddles)
    CMD4(CCC_Float, "r_puddle_scale", &ps_r_puddle_scale, 0.1f, 6.f);    // puddle size (bigger = smaller pools)
    CMD4(CCC_Integer, "r_sf", &ps_r_sf, 0, 1);                  // Surface Field master enable
    CMD4(CCC_Integer, "r_sf_debug", &ps_r_sf_debug, 0, 5);      // 0 off,1 height,2 slope,3 curvature,4 sky,5 canopy
    CMD4(CCC_Float, "r_sf_eps", &ps_r_sf_eps, 0.25f, 8.f);      // derive finite-difference epsilon (m)
    CMD4(CCC_Float, "r_snow", &ps_r_snow, 0.f, 1.f);            // TARGET snow coverage (Surface Field consumer)
    CMD4(CCC_Float, "r_snow_rate", &ps_r_snow_rate, 0.f, 5.f);  // snow accumulate/melt speed (per sec)
    CMD4(CCC_Integer, "r_snow_deform", &ps_r_snow_deform, 0, 1);              // footprint deformation enable
    CMD4(CCC_Float, "r_snow_deform_depth", &ps_r_snow_deform_depth, 0.f, 0.5f);   // print press depth (m)
    CMD4(CCC_Float, "r_snow_deform_radius", &ps_r_snow_deform_radius, 0.05f, 1.f); // print radius (m)
    CMD4(CCC_Float, "r_snow_deform_time", &ps_r_snow_deform_time, 1.f, 300.f);     // print lifetime (sec, time-decay)
    CMD4(CCC_Integer, "r_snow_deform_tex", &ps_r_snow_deform_tex, 0, 1);           // dense deform texture (vk_deform) vs stamp loop
    CMD4(CCC_Integer, "r_snow_mesh", &ps_r_snow_mesh, 0, 2);                       // dense snow surface mesh (1=on, 2=debug magenta)
    CMD4(CCC_Float, "r_snow_berm", &ps_r_snow_berm, 0.f, 2.f);                     // displaced-snow berm height around prints
    CMD4(CCC_Float, "r_snow_ripple", &ps_r_snow_ripple, 0.f, 4.f);                 // wind-ripple relief strength on open snow
    CMD4(CCC_Float, "r_snow_rough", &ps_r_snow_rough, 0.f, 1.f);                   // trail/print imperfection (width/depth/edge noise)
    CMD4(CCC_Integer, "r_water_sim", &ps_r_water_sim, 0, 1);             // water flow sim master enable
    CMD4(CCC_Float, "r_water_rain", &ps_r_water_rain, 0.f, 5.f);         // sim rain input rate (depth/s)
    CMD4(CCC_Float, "r_water_evap", &ps_r_water_evap, 0.f, 20.f);        // sim leak rate (exp drain ∝ amount)
    CMD4(CCC_Float, "r_water_flow", &ps_r_water_flow, 0.f, 4.f);         // sim downhill accel (flow speed / streams)
    CMD4(CCC_Integer, "r_water_iters", &ps_r_water_iters, 1, 6);         // sim relaxation steps per frame
    CMD4(CCC_Float, "r_water_murk", &ps_r_water_murk, 0.f, 8.f);         // volumetric absorption /m (deeper feel)
    CMD4(CCC_Float, "r_water_refract", &ps_r_water_refract, 0.f, 0.2f);  // bottom refraction strength

    CMD3(CCC_Mask64, "r4_enable_tessellation", &ps_r2_ls_flags_ext, R2FLAGEXT_ENABLE_TESSELLATION); // Need restart

    CMD3(CCC_Mask64, "r4_wireframe", &ps_r2_ls_flags_ext, R2FLAGEXT_WIREFRAME); // Need restart
    CMD3(CCC_Mask64, "r2_steep_parallax", &ps_r2_ls_flags, R2FLAG_STEEP_PARALLAX);
    CMD3(CCC_Mask64, "r2_detail_bump", &ps_r2_ls_flags, R2FLAG_DETAIL_BUMP);

    CMD3(CCC_Token, "r2_sun_quality", &ps_r_sun_quality, qsun_quality_token);

    CMD3(CCC_Mask64, "r2_visor_refl", &ps_r2_ls_flags_ext, R2FLAGEXT_VISOR_REFL);
    CMD3(CCC_Mask64, "r2_visor_refl_control", &ps_r2_ls_flags_ext, R2FLAGEXT_VISOR_REFL_CONTROL);
    CMD4(CCC_Float, "r2_visor_refl_intensity", &ps_r2_visor_refl_intensity, 0.f, 1.f);
    CMD4(CCC_Float, "r2_visor_refl_radius", &ps_r2_visor_refl_radius, 0.3f, 0.6f);

    CMD3(CCC_Token, "r_aa_mode", &ps_r_pp_aa_mode, pp_aa_mode_token);
    //CMD3(CCC_Token, "r_aa_dlss_preset", &ps_r_dlss_preset, dlss_mode_token);

    CMD4(CCC_Float, "r_3dss_scale_factor", &ps_r_dlss_3dss_scale_factor, 1.f, 2.5f);

    CMD4(CCC_Integer, "r__no_scale_on_fade", &ps_no_scale_on_fade, 0, 1); // Alundaio

    // r3_fog_reload omitted — R4 references dx103DFluidManager.

    CMD3(CCC_Mask64, "r3_dynamic_wet_surfaces", &ps_r2_ls_flags, R3FLAG_DYN_WET_SURF);
    CMD4(CCC_Float, "r3_dynamic_wet_surfaces_near", &ps_r3_dyn_wet_surf_near, 5, 70);
    CMD4(CCC_Float, "r3_dynamic_wet_surfaces_far", &ps_r3_dyn_wet_surf_far, 30, 279);

    CMD3(CCC_Mask64, "r3_volumetric_smoke", &ps_r2_ls_flags, R3FLAG_VOLUMETRIC_SMOKE);

    CMD1(CCC_VideoMemoryStats, "video_memory_stats");

    CMD4(CCC_Integer, "r_pnv_mode", &ps_pnv_mode, 0, 3);

    CMD4(CCC_Float, "r_pnv_noise", &ps_pnv_noise, 0.f, 1.f);
    CMD4(CCC_Float, "r_pnv_scanlines", &ps_pnv_scanlines, 0.f, 5.f);
    CMD4(CCC_Float, "r_pnv_scintillation", &ps_pnv_scintillation, 0.f, 1.f);

    CMD4(CCC_Float, "r_pnv_position", &ps_pnv_position, 0.f, 100.f);
    CMD4(CCC_Float, "r_pnv_radius", &ps_pnv_radius, 0.f, 1.f);

    // A - теперь не надо, параметры можно менять отдельно
    //CMD4(CCC_Float, "r_pnv_generation", &ps_pnv_params_1, 0.f, 100.f);
    // A - яркость lua_param_nvg_gain_current - по сути сила ПНВ
    CMD4(CCC_Float, "r_pnv_gain_current", &ps_pnv_params_2, 0.1f, 3.f);
    // A - размер виньетки
    CMD4(CCC_Float, "r_pnv_size_vignet", &ps_pnv_params_3, 0.f, 1.f);
    // A - lua_param_nvg_gain_offset
    CMD4(CCC_Float, "r_pnv_gain_offset", &ps_pnv_params_4, 0.5f, 3.f);

    // B - режим виньетки (чисто трубок) 
    CMD4(CCC_Float, "r_pnv_num_tubes", &ps_pnv_params_1_2, 1.f, 4.f); //  1, 1.1, 1.2, 2, 4
    // B - Порог для размывания источника света
    CMD4(CCC_Float, "r_pnv_washout_thresh", &ps_pnv_params_2_2, 0.1f, 0.9f);
    // B - тряска картинки
    CMD4(CCC_Float, "r_pnv_glitch", &ps_pnv_params_3_2, 0.f, 0.9f);
    // B - прозрачность краев виньетки  0, 1, 2, 3
    CMD4(CCC_Float, "r_pnv_alfa_vignete", &ps_pnv_params_4_2, 0.f, 3.f); // 0 - blur, 1 - black, 2 - image overlay

/*
    //CMD4(CCC_Integer, "r__fakescope", &scope_fake_enabled, 0, 1); // crookr for fake scope
    CMD4(CCC_Float, "fake_scope_radius", &scope_fake_radius, 0, 1); // crookr for fake scope
    CMD4(CCC_Float, "fake_scope_power", &scope_fake_power, 0, 1); // crookr for fake scope
    CMD4(CCC_Float, "fake_scope_interp", &scope_fake_interp, 0, 1); // crookr for fake scope

    constexpr Fvector4 tw_min{};
    constexpr Fvector4 tw_max{10.f, 10.f, 10.f, 10.f};

    CMD4(CCC_Vector4, "fake_scope_params_1", &ps_scope1_params, tw_min, tw_max); // crookr for fake scope
    CMD4(CCC_Vector4, "fake_scope_params_2", &ps_scope2_params, tw_min, tw_max); // crookr for fake scope
    CMD4(CCC_Vector4, "fake_scope_params_3", &ps_scope3_params, tw_min, tw_max); // crookr for fake scope
*/

    // Screen Space Shaders
    CMD4(CCC_Vector3, "ssfx_shadows", &ps_ssfx_shadows, Fvector3().set(128, 1536, 0), Fvector3().set(1536, 4096, 0));

    CMD4(CCC_Vector3, "ssfx_shadow_bias", &ps_ssfx_shadow_bias, Fvector3().set(0, 0, 0), Fvector3().set(1.0, 1.0, 1.0));
    CMD4(CCC_Vector4, "ssfx_lut", &ps_ssfx_lut, Fvector4().set(0.0, 0.0, 0.0, 0.0), tw2_max);

    CMD4(CCC_Vector4, "ssfx_wind_grass", &ps_ssfx_wind_grass, (Fvector4{}), (Fvector4{20.0f, 5.0f, 5.0f, 5.0f}));
    CMD4(CCC_Vector4, "ssfx_wind_trees", &ps_ssfx_wind_trees, (Fvector4{}), (Fvector4{20.0f, 5.0f, 5.0f, 1.0f}));

    CMD4(CCC_Vector4, "ssfx_florafixes_1", &ps_ssfx_florafixes_1, (Fvector4{}), (Fvector4{1.0f, 1.0f, 1.0f, 1.0f}));
    CMD4(CCC_Vector4, "ssfx_florafixes_2", &ps_ssfx_florafixes_2, (Fvector4{}), (Fvector4{10.0f, 1.0f, 1.0f, 1.0f}));
    
    CMD4(CCC_Vector4, "ssfx_wetsurfaces_1", &ps_ssfx_wetsurfaces_1_cfg, (Fvector4{0.01f, 0.01f, 0.01f, 0.01f}), (Fvector4{2.0f, 2.0f, 2.0f, 2.0f}));
    CMD4(CCC_Vector4, "ssfx_wetsurfaces_2", &ps_ssfx_wetsurfaces_2_cfg, (Fvector4{0.01f, 0.01f, 0.01f, 0.01f}), (Fvector4{2.0f, 2.0f, 2.0f, 2.0f}));
    
    CMD4(CCC_Integer, "ssfx_is_underground", &ps_ssfx_is_underground, 0, 1);

    CMD4(CCC_Integer, "ssfx_gloss_method", &ps_ssfx_gloss_method, 0, 1);
    CMD4(CCC_Vector3, "ssfx_gloss_minmax", &ps_ssfx_gloss_minmax, (Fvector3{}), (Fvector3{1.0, 1.0, 1.0}));

    CMD4(CCC_Vector4, "ssfx_lightsetup_1", &ps_ssfx_lightsetup_1, (Fvector4{}), (Fvector4{1.0f, 1.0f, 1.0f, 1.0f}));

    CMD4(CCC_Vector4, "ssfx_hud_drops_1", &ps_ssfx_hud_drops_1_cfg, (Fvector4{}), (Fvector4{100.f, 100.f, 100.f, 100.f}));
    CMD4(CCC_Vector4, "ssfx_hud_drops_2", &ps_ssfx_hud_drops_2_cfg, (Fvector4{}), (Fvector4{100.f, 100.f, 100.f, 100.f}));

    CMD4(CCC_Vector4, "ssfx_blood_decals", &ps_ssfx_blood_decals, (Fvector4{}), (Fvector4{5.f, 5.f, 0.f, 0.f}));

    CMD4(CCC_Vector4, "ssfx_rain_1", &ps_ssfx_rain_1, (Fvector4{}), (Fvector4{10.f, 5.f, 5.f, 2.f}));
    CMD4(CCC_Vector4, "ssfx_rain_2", &ps_ssfx_rain_2, (Fvector4{}), (Fvector4{1.f, 10.f, 10.f, 10.f}));
    CMD4(CCC_Vector4, "ssfx_rain_3", &ps_ssfx_rain_3, (Fvector4{}), (Fvector4{1.f, 10.f, 10.f, 10.f}));

    CMD4(CCC_Vector4, "ssfx_grass_shadows", &ps_ssfx_grass_shadows, (Fvector4{}), (Fvector4{2.f, 0.f, 100.f, 0.f}));
    CMD4(CCC_Vector3, "ssfx_shadow_cascades", &ps_ssfx_shadow_cascades, (Fvector3{1.0f, 1.0f, 1.0f}), (Fvector3{300.f, 300.f, 300.f}));
    CMD4(CCC_Float, "ssfx_wpn_dof_2", &ps_ssfx_wpn_dof_2, 0, 1);
    CMD4(CCC_Vector4, "ssfx_grass_interactive", &ps_ssfx_grass_interactive, (Fvector4{}), (Fvector4{1.f, static_cast<float>(GRASS_SHADER_DATA_COUNT), 5000.f, 1.f}));
    CMD4(CCC_Vector4, "ssfx_int_grass_params_1", &ps_ssfx_int_grass_params_1, (Fvector4{}), (Fvector4{5.f, 5.f, 5.f, 60.f}));
    CMD4(CCC_Vector4, "ssfx_int_grass_params_2", &ps_ssfx_int_grass_params_2, (Fvector4{}), (Fvector4{5.f, 20.f, 1.f, 5.f}));

    CMD4(CCC_Integer, "ssfx_terrain_grass_align", &ps_ssfx_terrain_grass_align, FALSE, TRUE);
    CMD4(CCC_Float, "ssfx_terrain_grass_slope", &ps_ssfx_terrain_grass_slope, 0.f, 1.f);

    CMD4(CCC_Vector3, "ssfx_color_grading", &ps_r2_img_cg, (Fvector3{}), (Fvector3{1.f, 1.f, 1.f}));

    CMD3(CCC_Mask64, "ssfx_height_fog", &ps_r2_ls_flags, R2FLAG_SSFX_HEIGHT_FOG);
    CMD3(CCC_Mask64, "ssfx_sky_debanding", &ps_r2_ls_flags, R2FLAG_SSFX_SKY_DEBANDING);
    CMD3(CCC_Mask64, "ssfx_indirect_light", &ps_r2_ls_flags, R2FLAG_SSFX_INDIRECT_LIGHT);
 //   CMD3(CCC_Mask64, "ssfx_bloom", &ps_r2_ls_flags, R2FLAG_SSFX_BLOOM);
    CMD3(CCC_Mask64, "ssfx_use_aces", &ps_r2_ls_flags, R2FLAGEXT_USE_ACES);
    CMD3(CCC_Mask64, "ssfx_shadows_enable", &ps_r2_ls_flags, R2FLAGEXT_SSFX_SHADOWS);
    CMD3(CCC_Mask64, "ssfx_sss_enable", &ps_r2_ls_flags, R2FLAGEXT_SSFX_SSS);

    CMD3(CCC_Mask64, "ssfx_inter_grass", &ps_r2_ls_flags_ext, R2FLAGEXT_SSFX_INTER_GRASS);
    CMD3(CCC_Mask64, "ssfx_inter_branches", &ps_r2_ls_flags_ext, R2FLAGEXT_SSFX_INTER_BRANCHES);
    CMD3(CCC_Mask64, "r_font_shadows", &ps_r2_ls_flags_ext, R2FLAGEXT_FONT_SHADOWS);

    CMD4(CCC_Integer, "ssfx_bloom_use_presets", &ps_ssfx_bloom_use_presets, 0, 1);
    CMD4(CCC_Vector4, "ssfx_bloom_1", &ps_ssfx_bloom_1, (Fvector4{1, 1, 0, 0}), (Fvector4{10, 100, 100, 10}));
    CMD4(CCC_Vector4, "ssfx_bloom_2", &ps_ssfx_bloom_2, (Fvector4{1, 0, 0, 0}), (Fvector4{5, 10, 10, 10}));

    CMD2(CCC_Bool, "ssfx_pom_refine", &ps_ssfx_pom_refine);
    CMD2(CCC_Bool, "ssfx_terrain_pom_refine", &ps_ssfx_terrain_pom_refine);
    CMD4(CCC_Vector4, "ssfx_pom", &ps_ssfx_pom, Fvector4().set(0, 0, 0, 0), Fvector4().set(36, 60, 1, 1));
    CMD4(CCC_Vector4, "ssfx_terrain_pom", &ps_ssfx_terrain_pom, (Fvector4{}), (Fvector4{36, 60, 1, 2}));
    CMD4(CCC_Vector4, "ssfx_terrain_offset", &ps_ssfx_terrain_offset, (Fvector4{-1.f, -1.f, -1.f, -1.f}), (Fvector4{1.f, 1.f, 1.f, 1.f}));
    CMD4(CCC_Vector4, "ssfx_ssr_1", &ps_ssfx_ssr_1, (Fvector4{1.f, 0.f, 0.f, 0.f}), (Fvector4{2.f, 1.f, 1.f, 1.f}));
    CMD4(CCC_Vector4, "ssfx_ssr_2", &ps_ssfx_ssr_2, (Fvector4{}), (Fvector4{2.f, 2.f, 2.f, 2.f}));

    CMD4(CCC_Float, "ssfx_exposure", &ps_r2_img_exposure, 0.5f, 1.5f);
    CMD4(CCC_Float, "ssfx_gamma", &ps_r2_img_gamma, 0.5f, 1.5f);
    CMD4(CCC_Float, "ssfx_saturation", &ps_r2_img_saturation, 0.5f, 1.5f);

#pragma todo("Simp: В общем эта настройка работает, но надо убирать мипмапы у текстур ui. Да и заметного влияния на fps я не вижу.")
    //CMD4(CCC_Integer, "texture_lod", &psTextureLOD, 0, 2);

    CMD1(CCC_PART_Export, "particles_export");
    CMD1(CCC_PART_Import, "particles_import");

    CMD1(CCC_PART_DumpTextures, "particles_dump_textures");

    CMD4(CCC_Float, "particle_update_mod", &ps_particle_update_coeff, 0.04f, 10.f);

    CMD3(CCC_Mask64, "r_lens_flare", &ps_r2_ls_flags_ext, R2FLAGEXT_LENS_FLARE);

    CMD1(CCC_Dbg_DumpStaticVisual, "dbg_dump_static_at_look");

    CMD4(CCC_Float, "r__dyn_opt_dist", &ps_r__opt_dist, 100.0f, 1000.0f);

    CMD4(CCC_Float, "r_aa_cas", &ps_r_cas, 0.0f, 1.0f);

    CMD4(CCC_Float, "r_alphatest_threshold", &ps_r_alphatest_threshold, 0.0f, 1.0f);

    CMD4(CCC_Integer, "exp_optimize_static_geom", &opt_static_geom, 0, 4);
    CMD4(CCC_Integer, "exp_optimize_shadow_geom", &opt_shadow_geom, 0, 1);

    CMD4(CCC_Integer, "r_back_buffer_count", &r_back_buffer_count, 2, 5);

    // shaders_xr_export omitted — bShadersXrExport lives in xrRender lib (not linked here).

    CMD1(CCC_OCC_Enable, "r_occ_enable");
    // r_occ_delay_invisible_{min,max} omitted — delay_invisible_* lives in xrRender lib.
}
