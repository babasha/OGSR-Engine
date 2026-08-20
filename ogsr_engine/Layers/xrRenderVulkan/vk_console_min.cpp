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
#include "vk_texture_stream.h"   // VK::TextureStreamer — video_memory_stats / r_txstream_stats
#include "vk_vram_stats.h"       // VK::Vram::DumpVmaJson — r_vram_dump
#include "vk_water_ripple.h"     // VK::WaterRipple::Splat — the r_wtr_drop probe
#include "vk_UIPipeline.h"       // VulkanUI::TraceUIPass — the `ui_pass_trace` command

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

// Preset K = the DLSS 4 transformer model (best quality, esp. Perf/UltraPerf —
// UltraPerf otherwise DEFAULTS to the old CNN preset F = «мыло»). 0 = driver default.
u32 ps_r_dlss_preset = NVSDK_NGX_DLSS_Hint_Render_Preset_K;
constexpr xr_token dlss_mode_token[]{
    {"st_opt_dlss_default", NVSDK_NGX_DLSS_Hint_Render_Preset_Default}, // default behavior, may or may not change after OTA
    {"st_opt_dlss_f", NVSDK_NGX_DLSS_Hint_Render_Preset_F},
    {"st_opt_dlss_j", NVSDK_NGX_DLSS_Hint_Render_Preset_J},
    {"st_opt_dlss_k", NVSDK_NGX_DLSS_Hint_Render_Preset_K},
    {},
};

float ps_r_dlss_3dss_scale_factor{1.0f};

// ⚠ INERT on the Vulkan path since 2026-07-24 — kept only so the graphics options
// screen and existing user.ltx files still resolve `r_sunshafts_mode`. The pass it
// drove (vk_pass_sunshafts) is deleted: it sampled the FAR sun map, which under VSM
// is left cleared = "lit everywhere", so it drew rays through walls; god rays now
// come out of the froxel medium (r_vol / r_vol_mist), which has actual occlusion
// and a phase function. Nothing reads this value any more.
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
// DEFAULT ON (2026-07-04): temporal accumulation (r_ssil_temporal, below) resolves
// the half-res banding, user-verified clean in-game → shipped on. r_ssil 0 to A/B
// against the no-bounce look. AO is unaffected either way.
int   ps_r_ssil_enable   = 1;     // r_ssil — global on/off (gates the prev-colour taps in the GTAO march)
int   ps_r_ssil_debug    = 0;     // r_ssil_debug — 1 = show ONLY the indirect bounce field
// Strength multiplies the gathered radiance at composite. IL is an occluder-colour
// AVERAGE gated by (1-AO)² (≈0 on open surfaces, only fills real recesses), added
// on top of the forward ambient — keep it subtle so it tints corners without a
// global brightness boost. Raise to taste.
float ps_r_ssil_strength = 0.5f;

// r_ssil_temporal — temporal accumulation for the GTAO pass (the missing piece
// that lets r_ssil ship ON). The horizon-folded IL bands without it because the
// 4-slice gather is directionally quantized AND static per frame (so a plain EMA
// of identical frames is a no-op). With temporal > 0 the gather noise is rotated
// PER FRAME (so each frame's bands differ) and the result is reprojected via the
// motion vectors ([[vk_motionvec]]) and EMA-blended — the per-frame patterns
// average into a smooth fill. AO shares the same march, so it is accumulated too
// (≥ the current spatial-only quality, just temporally stable). The value IS the
// history weight α (0 = OFF → byte-identical to the spatial-only path; ~0.9 =
// strong smoothing). Needs r_motion_vectors on for reprojection (falls back to
// same-pixel EMA when MV is unavailable).
// DEFAULT ON 0.9 (2026-07-04): this is what lets r_ssil ship on (kills the IL bands);
// verified clean, no ghosting. Set 0 to inspect the raw spatial-only IL.
float ps_r_ssil_temporal = 0.9f;

// r_ssao_temporal — the SAME temporal accumulation, but as a first-class control
// for the AO channel (so it works WITHOUT r_ssil). Value = history weight α. The
// gather noise (slice direction + radial phase) is rotated per frame → each frame
// is a DIFFERENT realisation of the AO → MV-reprojected EMA averages them into a
// stable, low-noise result, which also lets the sample count drop (r2_ssao) without
// banding showing. 0 = OFF (byte-identical spatial path); ~0.85 = clean+stable.
// Needs r_motion_vectors on for moving-camera reprojection (else same-pixel EMA —
// fine when still). AO and IL share one jitter+history; the effective α used by the
// blur is max(r_ssao_temporal, r_ssil_temporal).
// DEFAULT ON 0.85 (2026-06-21): measured ~free (no SSAO cost change vs 0) and it
// keeps AO stable + clean at the now-default LOW sample count (see SampleCount()).
// Reprojection rides r_motion_vectors (also default ON); MV off → same-pixel EMA.
float ps_r_ssao_temporal = 0.85f;

// GTAO grazing-angle fade threshold (N·V). The depth-reconstructed normal is
// unreliable at grazing angles, where a flat OPEN surface fails to cancel its own
// horizon → residual self-occlusion (AO<1) that r_ssao_strength's pow amplifies.
// Below this N·V the AO fades back to fully open; head-on surfaces keep full AO
// (real contact shade). 0 = off; higher = fades more of the grazing floor.
// (The related bent-normal striping has its own fix: the geomN deadband in
// env_common.glsl gtaoBentN — so this knob may now afford a lower value.)
float ps_r_ssao_bias = 0.30f;

// Final 8-bit output dither amplitude (in swapchain LSBs), applied at the very
// end of the tonemap. The swapchain is B8G8R8A8_UNORM: slow lighting gradients
// (shaded ambient on flat asphalt, sky-ambient on distant slopes, dusk sky)
// quantize into visible contour stripes no matter how clean the source buffers
// are. TPDF dither turns the steps into imperceptible grain. 0 = off (A/B),
// 1 = textbook ±1 LSB.
float ps_r_dither = 1.0f;

// Vulkan motion vectors — screen-space (prevUV − curUV) reconstructed from the
// prepass depth + the previous frame's view-proj. Foundation for DLSS/FSR
// upscaling, frame-gen and the path-tracer denoiser. Phase 1 = camera + static
// world (the dominant motion); self-moving geometry is a later phase. Default
// ON (cheap fullscreen pass, no result consumer yet — harmless until DLSS lands).
int   ps_r_motion_vectors  = 1;     // r_motion_vectors — global MV pass on/off
u32   ps_r_dlss            = 0;     // r_dlss — DLSS Super Resolution; needs r_motion_vectors + NGX available
u32   ps_r_dlss_quality    = 0;     // r_dlss_quality — 0=DLAA(native),1=Quality,2=Balanced,3=Performance,4=UltraPerf (render<display upscale = the fps win)
float ps_r_dlss_sharp      = 0.3f;  // r_dlss_sharp — CAS sharpen on the DLSS output in the tonemap (0 = off; NGX dropped built-in sharpening)
int   ps_r_dlss_debug      = 0;     // r_dlss_debug — 1=CAS delta heatmap, 2=split (left=no-DLSS bilinear, right=composite), 3=gate flag (green=DLSS output live, red=plain scene)
int   ps_r_dlss_jitter_flip= 2;     // r_dlss_jitter_flip — sign of the jitter REPORTED to DLSS Evaluate: bit0=flip X, bit1=flip Y. DEFAULT 2 (flip Y): the scene rasterizes through a D3D negative-height viewport, so our +jy clip jitter moves the image UP while DLSS pixel space is y-down — user A/B-verified 2026-07-07 (flip 2 = «намного лучше, практически без апскейлинга»)
int   ps_r_dlss_exp        = 1;     // r_dlss_exp — 1 = feed DLSS the real tonemap exposure (1x1 texture; fixes black undershoot blotches on distant foliage), 0 = NGX AutoExposure flag (old)
float ps_r_dlss_bias       = 1.0f;  // r_dlss_bias — scale of the material mip-LOD bias under upscaling (1 = full log2(render/display); sharpens alpha mips → foliage coverage holes; 0 = off A/B)
int   ps_r_dlss_sl         = 0;     // r_dlss_sl — Stage B: route the SR evaluate through Streamline sl.dlss (FG prerequisite); 0 = raw NGX (proven default)
int   ps_r_dlss_sl_flip    = 2;     // r_dlss_sl_flip — SL-reported jitter sign (bit0=X, bit1=Y); SL convention may differ from NGX's verified flip 2
int   ps_r_dlss_sl_mv      = 0;     // r_dlss_sl_mv — 1 = negate mvecScale (SL MV direction A/B)
int   ps_r_dlss_fg         = 0;     // r_dlss_fg — Stage C: DLSS Frame Generation (MFG); needs r_dlss 1 + r_dlss_sl 1 + Reflex (auto). 0 = off
int   ps_r_dlss_fg_mult    = 2;     // r_dlss_fg_mult — frame multiplier 2..6 (2 = 1 interpolated frame; up to 6x if the GPU supports MFG)
int   ps_r_dlss_fg_debug   = 0;     // r_dlss_fg_debug — 1 = periodic [VK SL FG] log of DLSS-G state (status/frames presented/VRAM) + Reflex state (low-latency avail/latency); 2 = every frame

// ⭐r_dlss_avail — the NGX capability probe's answer, published as a console
// entry so the OPTIONS SCREEN can ask it. The renderer already refuses to run
// DLSS on hardware that cannot, but a settings row that toggles and changes
// nothing is exactly the kind of placebo this pass exists to remove: the menu
// hides the whole DLSS group when this is 0 (`data-need="r_dlss_avail"`).
// ⚠Not savable — it describes THIS machine, and a value carried over from
// another one would be a lie.
// ⚠⚠THREE states, not two: -1 = the probe has not run yet. user.ltx is read
// BEFORE the Vulkan device exists, so a two-state flag made every startup print
// "DLSS unavailable on this GPU" while restoring a perfectly good DLSS setting
// on a card that supports it — a false alarm in the log of every single run.
// The menu reads this through GetBool, where -1 counts as "yes"; by the time a
// menu exists the probe has always answered.
int   ps_r_dlss_avail      = -1;

// ⭐r_render_scale — internal resolution as a fraction of the display, for the
// case DLSS cannot cover (no NGX, or the player wants it off). The renderer has
// had a render-extent≠display-extent path since DLSS landed and every pass sizes
// itself from it; the tonemap composites at display res and samples the smaller
// buffers with normalised UVs, so this is a plain bilinear upscale of a cheaper
// frame. 1 = native. Ignored while DLSS is upscaling — that owns the extent.
float ps_r_render_scale    = 1.0f;
int   ps_r_mv_debug        = 0;     // r_mv_debug       — false-colour overlay (verify sign/Y-flip/magnitude)
int   ps_r_mv_trees        = 1;     // r_mv_trees       — tree wind-sway MV overlay on/off (A/B: is the sway MV better than none under DLSS?)
int   ps_r_mv_grass        = 1;     // r_mv_grass       — grass wind-sway MV overlay on/off (same A/B)
float ps_r_mv_debug_scale  = 30.0f; // r_mv_debug_scale — overlay magnitude (per-frame UV motion is tiny)

// SSFX tree wind (live tuning). Trunk bend defaulted DOWN from the SSFX 0.5 —
// our trees are tall, the H² trunk term made them "sway like sails" at 0.5.
float ps_r_wind_tree_bend    = 0.18f; // r_wind_tree_bend    — trunk sway intensity (SSFX wsetup_trees.z)
float ps_r_wind_tree_anim    = 11.0f; // r_wind_tree_anim    — branch/leaf flutter speed (SSFX wsetup_trees.x)
float ps_r_wind_tree_trunk   = 0.15f; // r_wind_tree_trunk   — trunk anim speed (SSFX wsetup_trees.y)
float ps_r_wind_tree_flutter = 4.0f;  // r_wind_tree_flutter — crown/leaf flutter amplitude (our extra, SSFX has none)
float ps_r_wind_tree_crown   = 4.0f;  // r_wind_tree_crown   — height (m) over which leaf flutter fades in from the tree base; 0 = off. (Dead until 2026-07-02 — pushed but never consumed by a shader.) Fixes the "sail": flutter amplitude is ABSOLUTE world metres, so without the gate a 2 m fir's foliage gets the same displacement as a 20 m crown and balloons. Tall crowns (above 4 m) are untouched.
float ps_r_wind_shadow_dist  = 40.0f; // r_wind_shadow_dist  — radius (m) where tree SHADOWS sway (near=per-frame, far=cached); 0 = all static (cheapest)

// Forward draw distance for trees that an FLOD container ALREADY draws as a
// billboard past VK::kImposterMinDist (250 m). Until this existed the tree cull
// had no distance term at all, so out there the full mesh and the imposter both
// rendered — the same tree twice. Only billboard-backed trees are affected; the
// rest keep their mesh at any range. 0 = off (restores the old double-draw).
// Values below kImposterMinDist are floored to it — see TreeFlodCutDist().
float ps_r_tree_dist         = 250.0f; // r_tree_dist

// Hi-Z occlusion cull for the forward tree set — drops trees fully hidden behind
// nearer geometry (hills, buildings). Shares the pyramid Pass_World builds from
// this frame's prepass depth; if that build did not happen, the test self-disables
// rather than sampling a stale (previous-frame) pyramid.
int ps_r_tree_hzb            = 1;      // r_tree_hzb

// Range past which tree.frag stops computing its SUBTLE per-pixel terms: the
// screen-space SSIL bounce, the sky-cube canopy sheen, the dynamic point/spot
// light walk and the shoreline wetness lookup. The sun, its shadow, ambient, AO,
// snow and fog are NEVER dropped — those are the terms you can actually see at
// range, and thinning them would read as trees changing colour as you approach.
//
// WHY THE FRAGMENT SHADER IS THE LEVER: the tree pass is FILL-bound, not
// geometry-bound. That was measured, twice, the hard way — meshlet culling cut
// caster vertices and moved the pass by zero, and the crown hull's whole point is
// "no texture fetch / no discard". So what moves Trees is the cost of a fragment
// that SURVIVED the alpha test, and out at 150 m a crown is a handful of pixels
// each while there are hundreds of crowns — the far band is where the fill goes.
//
// Fades out over the last quarter of the band rather than switching, or the forest
// grows a visible ring that slides as the camera moves. 0 = off (full shading at
// every distance, the pre-2026-08-13 behaviour).
float ps_r_tree_shade_dist   = 100.0f; // r_tree_shade_dist
// r_vsm_tree_wind — near/far WIND HYBRID (UE5 WPO-disable-distance pattern): trees within
// r_vsm_tree_wind_dist cast into the DYNAMIC atlas every frame WITH live wind → their
// shadows sway smoothly (one time slice per frame); farther trees stay rigid in the
// toroidal static cache (imperceptible at distance). Boundary crossings invalidate the
// tree's static pages the same frame (no ghosts). Replaces the old TEST mode that applied
// wind to STATIC pages — staggered round-robin refreshes froze each page at a different
// wind phase = the "jelly" shadows. Costs: near trees re-raster into the dyn atlas per
// frame (bounded by the distance). Default ON (VSM itself is the opt-in Ultra path).
int   ps_r_vsm_tree_wind      = 1;
// 2x2 coarse shading (pipelineFragmentShadingRate) on the near-tree DYNAMIC crown-shadow
// raster — cuts the fill-bound VSM/DynTrees cost that persists UNDER WIND (can't be cached).
// A/B knob: 1 = on (2x2), 0 = per-pixel. Falls back to per-pixel where HW lacks pipeline-rate.
// PARKED at 0: measured A/B (2026-07-04) showed NO gain — the crown raster is geometry/raster-
// bound, not fragment-bound, so coarse shading does nothing. Kept as a knob; infra reused by cones.
int   ps_r_vsm_tree_vrs       = 0;
// DYNAMIC-pass cadence (decoupled, Level 1): the sun/clipmap + the cheap STATIC atlas run
// EVERY frame (wall/terrain/ground shadows track the sun smoothly, no strobe); only the
// EXPENSIVE dynamic pass (NPC + grass + near-tree wind crowns) updates every Nth frame, its
// atlas preserved (world-anchored) between. 0/1 = every frame (off); 2-3 ≈ 30-45 Hz; dyn
// shadows go 1..N-1 frames stale (EMA r_vsm_ta_blend_dyn smooths). Default off.
int   ps_r_vsm_cadence        = 1;
// Camera-motion deadband for the dyn cadence: the frozen dyn atlas is world-anchored → only
// aligned while (nearly) stationary, so the skip is allowed ONLY when the camera moved less
// than this (metres/frame) AND barely turned. Walking closer/farther otherwise scrolls dyn
// pages + batches near-set transitions → the crown/NPC shadows flicker. Moving = full-rate
// dyn (correct, no flicker); standing = the cadence perf saving. 0 = never skip while moving.
float ps_r_vsm_cadence_still  = 0.02f;
// Second half of the cadence alignment gate: the frozen dyn atlas is valid only for the SUN
// angle it was rendered at. The living sun — especially env time-of-day KEYFRAME STEPS (diag
// measured ~18-22° jumps across a few frames) — misaligns it → the "тени пропали на кадр"
// flicker. Force a full-rate dyn update when the sun rotated more than this (degrees) since the
// last dyn update. Small enough to catch steps (normal drift is ~0.03°/3-frames), large enough
// not to fire on smooth motion. Window page-snaps are always caught regardless. 0 = sun gate off.
float ps_r_vsm_cadence_sun    = 0.30f;
// Near-set radius (m) — LATERAL light-space distance from the camera to the tree's shadow
// COLUMN (not trunk distance: a low sun lands a tree's shadow tens of metres down-sun, and
// what must sway is the shadow NEAR THE PLAYER regardless of where its tree stands). Trees
// whose column passes within this radius (and within 3× world distance) cast dynamically.
float ps_r_vsm_tree_wind_dist = 40.0f;
// Impostor crown shadows: replace the near-tree DYNAMIC crown MESH raster with ONE baked
// sun-facing billboard/tree (2 tris vs thousands of crown verts every frame under wind).
// Kills the geometry/raster-bound VSM/DynTrees cost. Trades dappling fidelity for speed —
// eyeball. Default 0 (PENDING quality/perf sign-off). _scale tunes the shadow footprint.
int   ps_r_vsm_tree_impostor       = 0;
float ps_r_vsm_tree_impostor_scale = 1.0f;
// Crown-HULL caster LOD (the "AC Shadows" shadow LOD, 2026-07-11): near trees BEYOND
// r_vsm_tree_hull_dist cast into the dyn atlas from a baked low-poly OPAQUE hull
// (ellipsoid lobes over k-means vertex clusters) instead of the full alpha-tested crown
// mesh; only the closest trees keep the real crown (dappling perfect where the player
// looks). Unlike the rejected FLAT impostor, the hull is a 3D volume — correct on walls
// and at any sun elevation. This is the first lever aimed at the PROVEN alpha-test-fill
// bound of VSM/DynTrees (no texture fetch / no discard / early-Z). Geometry is always
// baked at load (~a few hundred KB) → toggles LIVE for the A/B. Default 0 pending the
// perf/quality sign-off (A/B: stand in a forest, r_profiler 1, flip r_vsm_tree_hull,
// watch VSM/DynTrees + the shadow look; r_vsm_debug 1 logs "hull=N" in the hybrid line).
// 2 = ALSO hull the FAR/STATIC foliage beyond _dist (crowns only, same lateral metric):
// closes the non-monotonic LOD hole (a 60 m tree casting a MORE detailed shadow than a
// 25 m one) and cuts the full-crown re-raster on dirty cache refreshes (sun-move spikes).
// NOTE: flipping 2 on/off converts the static cache via invalidation circles (≤4/frame)
// + the moving-sun round-robin — expect the far shadows to transition over ~10-30 s.
// DEFAULT ON since 12-08-2026, together with r_vsm: this is the whole point of
// making VSM primary. Mode 1 = near DYNAMIC casters beyond _dist swap the alpha-
// tested crown for the opaque hull — exactly the layer measured at 6.34 ms under
// the cascades. Mode 2 additionally hulls the far/static foliage; left off for now
// because flipping it re-converts the static cache over ~10-30 s (see above), so it
// deserves its own A/B rather than riding in on this change.
int   ps_r_vsm_tree_hull      = 1;
float ps_r_vsm_tree_hull_dist = 18.0f;
// Crown VOXELIZATION (baked at load → level reload to change): 0 = PCA ellipsoid LOBES
// only (the original "AC Shadows" clusters). >0 bakes TWO things per crown, UE Nanite-
// foliage style: (a) a modest merged shell for the depth-only SHADOW caster (solid
// silhouette, resolution auto-clamped ~8–20 regardless of this value), and (b) the
// VISUAL voxel CLOUD — individual small colored cubes on the leaf cards (per-voxel
// palette/AO/height tint, size jitter), kHullLods LOD levels with the voxel edge
// DOUBLING per level. The value = level-0 cells across the crown's largest axis
// (64 ≈ 10–20 cm cubes on a full tree — the "Witcher 4 demo" look; higher = finer).
int   ps_r_vsm_tree_hull_vox  = 64;
// r_vsm_tree_vox_cloud — 1 = bake the VISUAL cube cloud even when its viewmode is
// off, which is what the code did unconditionally. The cloud has exactly one
// consumer (r_vsm_tree_hull_debug) and that viewmode needs a level reload anyway,
// so it is now built only when something will read it. Skipping it drops a sweep
// over every cell of every LOD of every species plus a 26-neighbour crowding
// lookup per occupied cell — 777k cubes on pripyat_full, all of them discarded.
// The SHADOW BRICKS out of the same occupancy grid are unaffected: the shipping
// caster path reads those, and they are still baked.
int   ps_r_vsm_tree_vox_cloud = 0;
// LIVE coarseness bias for the voxel viewmode: added to the screen-size-picked LOD level
// (0 = as graded; negative → force finer/smaller; positive → coarser/bigger). No rebake.
int   ps_r_vsm_tree_hull_lod  = 0;
// Target PROJECTED voxel size in PIXELS — the rendered cube edge is CONTINUOUS
// (dist × this angular size, clamped below by the finest baked cell), so voxels grow
// smoothly from tiny near to huge far with no size steps; the baked grids only supply
// positions/density (largest cell ≤ the current cube size). Smaller = finer/more
// instances; live, no rebake.
float ps_r_vsm_tree_hull_vox_px = 4.0f;
// Distance GAIN for the target pixel size: +this many px per 100 m. 0 = constant screen
// size (uniform fine grain everywhere — reads as if "nothing changes" while flying);
// >0 = far crowns become VISIBLY chunkier blocks, a continuous small→huge gradient.
float ps_r_vsm_tree_hull_vox_far = 4.0f;
// SHADOW-only floor on the voxel cube edge (meters). The caster instances every cube
// LEGACY (36-vert-cube caster era, superseded by the brick caster's shadow-view LOD):
// kept registered so the persisted user.ltx line still parses, but the code no longer
// reads it — _stex/_sfloor below replaced it. The 0.35 m floor is exactly what made
// the brick shadows read as solid boxes (0.28+ m cells hold no leaf gaps).
float ps_r_vsm_tree_hull_vox_smin = 0.35f;
// UE-style SHADOW voxel LOD (brick caster): page texels per voxel edge. The caster
// picks the baked grid whose cell ≈ stex × the page texel of the tree's clipmap level
// (texel(L) = r_vsm_base·2^L / 4096, level from the same lateral shadow-column metric
// the residency uses) — the shadow view drives the LOD, NOT the camera (UE
// NaniteClusterCulling LODScale). Near rings → finest bake (real leaf gaps), far rings
// → coarse bricks that stay sub-texel-ish. Smaller = finer/costlier. Live.
float ps_r_vsm_tree_hull_vox_stex = 3.0f;
// Safety floor on the caster voxel edge (m) for weak HW — 0 disables. Live.
float ps_r_vsm_tree_hull_vox_sfloor = 0.05f;
// Hard cap on the caster slice per tree (BRICK count of the picked baked level, ~×14
// fewer units than cubes at 6 verts each); telemetry `slice=` in [VK VoxLOD]. Live.
int ps_r_vsm_tree_hull_vox_cap = 1500;
// STATIC-tier proxy pick: 0 (default) = the merged watertight SHELL (~96 tris/tree —
// the blind-test-verified far representation; crowns there get meshlet per-page
// culling, bricks do NOT, so full brick slices into every dirty page measurably LOST
// to the shell on living-sun cache refreshes: VSM/Static 0.42-0.60 vs 0.13-0.44);
// 1 = bricks in the static tier too (visual parity with the dyn band; costs the
// missing per-page brick cull — the planned stage-2 refinement). Live.
int ps_r_vsm_tree_hull_vox_static = 0;
// Stage-2 per-page brick cull (vsm_vox_cull.comp) — the UE-parity piece: only bricks
// overlapping a shadow page rasterize into it (compacted (tree,page) draws + a
// surviving-brick remap list), instead of the full slice sweeping the VS for every
// page of the tree's column. This is what the crown path already had via the meshlet
// stage 2 — and what makes static-tier bricks (vox_static 1) affordable. Live.
int ps_r_vsm_tree_hull_vox_cull = 1;
// UE WPODisableDistance analog (needs r_vsm_tree_hull 2): past the hull boundary a
// brick's wind amp has ramped to 0 across the crossfade band — the shadow is rigid,
// so the tree is DEMOTED from the per-frame dyn atlas into the cached static tier.
// The dyn near set shrinks to the real-crown ring + the band (the main perf win);
// wind ALU on voxel casters drops to zero (the VS skips ssfxTreeWind at amp 0). Live;
// toggling converts the cache through the usual invalidation circles over a few sec.
int ps_r_vsm_tree_hull_vox_wpo = 1;
// SHADOW crossfade band (voxel caster): fraction of r_vsm_tree_hull_dist just inside the
// boundary where a tree's shadow hands over to true leaves in TWO stages (fade 1 → 0):
// the outer half thins the cubes one-by-one in leaf-coverage order (the UE close-up
// look — "extras removed", survivors sit on the leaf clumps), the inner half adds the
// real alpha-tested crown while the remaining cubes dissolve out. 0.5 default so both
// stages read (at _dist 18 m the band spans 9–18 m of lateral shadow-column distance).
float ps_r_vsm_tree_hull_band = 0.5f;
// Dithered LOD crossfade: within this fraction of the handover distance BEFORE a level
// switch, both levels draw with complementary screen-door masks (fine dissolves out,
// coarse dissolves in — exactly one survives per pixel), so flying the camera never
// pops between voxel sizes. Instance cost doubles only inside the band. 0 = hard cut.
// Same infra the future SHADOW-side voxel LOD will need (depth-only discard works too).
float ps_r_vsm_tree_hull_vox_fade = 0.25f;
// Voxel viewmode ("voxels instead of leaves"): crowns drawn as their baked voxel cloud
// (face-shaded colored cubes; falls back to the shaded hull overlay when _vox = 0).
// 1 = only hull-TIER trees (shows the _dist boundary); 2 = EVERY tree with a baked
// cloud, so you can walk up to a near tree and inspect its voxel crown.
int   ps_r_vsm_tree_hull_debug = 0;
// ============================================================================================
// ⛔ "VSM tree perf" arc — MEASURED NO NET GAIN on dGPU (2026-07-02), KEPT BUT DISABLED, DO NOT RE-CHASE.
// Both cvars below default 0 = OFF (zero cost, picture identical). Code is intentionally retained (not
// deleted) — it is correct and verified, just not a win here. WHY it didn't pay (see memory
// [[vulkan-vsm-meshlet-plan]] for the full A/B): the near-tree VSM cost is REAL VISIBLE shadow coverage
// of big near crowns, not occludable overdraw — meshlet-cull only cut vertices (pass is FILL-bound → 0),
// HZB only culls the ~6% hidden behind static geometry while its reduce pass costs more than it saves
// (clean stationary A/B: HZB on ≈ off, net −0.15ms). The real levers (coarser near-tree pages / fewer
// near trees) were rejected on purpose — near shadows the player examines must stay pixel-perfect.
// → Leave OFF. Possible future value ONLY on iGPU (bandwidth-bound) or dense interiors. Don't retest on dGPU.
// ============================================================================================
// r_vsm_meshlet — per-page MESHLET culling of VSM tree casters (Phase A+B). Dices trees into ~128-tri
// clusters at load; a 2nd bin stage draws only clusters overlapping each atlas page. Correct, picture
// identical. 2026-07-05: VSM/DynTrees is geometry-bound (VRS/HZB no help) → meshlet cull is the one
// caster-tri lever, so DEFAULT ON on the chance it helps dense forest; user measured NEUTRAL in a
// moderate view. NET-0/NET− per the old audit → if VSM/DynTrees ever WORSENS with it, set 0 to revert.
int   ps_r_vsm_meshlet        = 1;
// r_vsm_hzb — shadow-HZB occlusion cull (static + Option A dyn-vs-static-occluder). Reduce maxes prior
// static-page depth; caster bins skip pairs fully behind cached walls/terrain. Correct, picture identical.
// NO NET GAIN (reduce costs > the little hidden it culls). Default 0 = OFF, kept for reference / interiors.
int   ps_r_vsm_hzb            = 0;
// r_vsm_hzb_margin — depth slack for the HZB occluder (stale-sun safety). Only used when r_vsm_hzb=1 (OFF).
float ps_r_vsm_hzb_margin     = 0.002f;

// Vulkan lighting normalization knobs (live, no restart). Both used to be
// literals scattered across the scene shaders (LDR-era compensation that
// predates the HDR tonemap): sun ×1.25 in world/terrain/vlit/skinned/grass
// and +0.05 ambient floors. They now apply ONCE on the CPU (vk_env_light UBO
// fill + the grass/tree sun pushes), so shaders consume final values and the
// look can be A/B'd in-game: r_sun_boost 1 + r_ambient_floor 0 = raw env values.
float ps_r_sun_boost     = 1.25f;
float ps_r_ambient_floor = 0.05f;

// LINEAR COLOUR PIPELINE — 0 = the historical gamma-space pipeline (albedo sampled
// raw from UNORM textures, lighting maths done on gamma-encoded values, tonemap
// output passed through unencoded); 1 = physically correct linear (Colour textures
// loaded as _SRGB so the sampler decodes on fetch, CPU-side light/env colours
// linearised on upload, and the tonemap applies the real sRGB OETF at the end).
//
// NOT live-switchable: the colourspace is baked into each texture's VkFormat at load
// time, so a change only affects textures loaded afterwards and needs a level reload
// to be coherent. Kept as a cvar anyway because it is the only honest A/B for a change
// that touches every lit pixel — and because the mod's art was authored against the
// gamma-space look, so "correct" and "what it used to look like" genuinely differ.
int ps_r_linear_color = 0;

// Gate the FLAT sky ambient (L.ambient.rgb) by sky visibility (rainVis) on static
// world surfaces. The flat sky-coloured fill was added everywhere ungated → houses
// and basements looked "lit by the sky". 0 = old ungated look; 1 = covered surfaces
// get NO flat sky fill (only baked hemi + local lights). Live-tunable to taste.
float ps_r_ambient_sky_gate = 0.7f;

// NIGHT SUN FREEZE: when the sun gives effectively no light (below the horizon),
// its shadows contribute nothing (every sun receiver multiplies by ~0 sun_color),
// so re-rasterizing them is pure waste. This freezes the VSM DYNAMIC atlas (NPC /
// grass / near-tree crowns — the dominant per-frame shadow cost) at night, reusing
// the existing cadence freeze. r_sun_night_freeze 0 = off; r_sun_night_lum = the
// sun_color luminance below which "night" kicks in (0.02 ≈ deep dusk).
int   ps_r_sun_night_freeze = 1;
float ps_r_sun_night_lum    = 0.05f;   // secondary: freeze if sun_color luminance drops below this
float ps_r_sun_night_alt    = 0.02f;   // primary: freeze when to-sun.y (sun altitude) drops below this (below horizon)

// Grass alpha-test cutoff (live). Lower = fatter/denser blades (less "see-through"),
// higher = thinner. Was hard-coded 0.5 in detail.frag → blades too thin, you could
// see the ground through the grass. 0.33 keeps more of each blade body.
float ps_r_grass_aref    = 0.33f;
// Mip-compensated grass alpha ("alpha sharpen"): boosts sampled alpha by ~N per mip
// level so distant grass keeps its coverage instead of dissolving see-through (alpha
// mips average toward 0). 0 = off (old look), higher = denser far grass. Live.
float ps_r_grass_asharp  = 0.35f;

// Glass opacity CEILING (live). The R4 formula takes both the pane's translucency
// AND the env-reflection share from the TEXTURE alpha — but this mod's glass DDS
// carry alpha ≈ 1 (repacked opaque), which reads as "just a texture". The ceiling
// clamps that: 0.55 = clearly-glass look, 1.0 = R4-faithful (pure texture alpha).
float ps_r_glass_opacity = 0.55f;

// Glass refraction ("uneven old pane" wobble): the late-glass panes re-draw into
// the heat-haze distortion RT and the tonemap bends the scene behind them.
// Strength multiplier; 0 = off. Live.
float ps_r_glass_refr    = 0.5f;   // user-tuned default (2026-07-02)

// COMPUTE PRE-SKINNING (r_preskin). Skin every visible skeleton leaf ONCE per
// frame in a compute pass into a shared world-space pool, then let the ~10
// consumer passes (depth prepass, AO normals, colour, sun cascades, point cube,
// spot tiles, VSM pages, glass) draw that pool instead of re-running the 1-4
// bone blend in each of their vertex shaders. Live; leaves that don't fit the
// pool silently fall back to the old path.
//
// STILL UNPROVEN — the first A/B (l01_escape, 2026-07-23) came out net negative
// but measured the WRONG WORKLOAD: of 835 sampled skeletons, 46% had 1-3 bones
// and 84% had no animation playing (doors, lamps, physics crates, dropped guns);
// only ~4.5% were NPC-scale (42-62 bones). 1-weight geometry can never win here
// — its classic shader is already a single matrix fetch — so the pool was mostly
// paying for leaves with nothing to save. Skinned_PreSkin now skips skinMode<2,
// which removes that loss; the real question (an animated NPC crowd) is still
// open. Numbers from that run, for reference: World/Skinned 0.31->0.31 and
// Shadow/Dyn* unchanged, SunShadow -0.22 ms and World/Depth -0.15 ms against
// +0.30 ms for the compute pass.
//
// Re-measure with MANY ANIMATED NPCs on screen before trusting either verdict,
// and watch [VK PreSkin] skipped1W / maxBones to confirm the scene is actually
// NPC-heavy. Note the frame is GPU-bound on statics (World/Statics ~5.3 ms of a
// 17 ms frame), so the whole skinned budget is ~1.5 ms — the ceiling is small.
int   ps_r_preskin       = 1;

// Animated cloud layer in the sky pass (R4 RenderClouds port): two scrolling
// cloud textures composited over the static cubemap sky. r_clouds on/off,
// r_clouds_intensity = additive brightness, r_clouds_speed = UV scroll rate.
int   ps_r_clouds           = 1;
float ps_r_clouds_intensity = 1.0f;
float ps_r_clouds_speed     = 1.0f;

// r_light_debug — dump the dynamic-light registry + collected set to the log every
// ~2 s (diagnose "lamp lights in R4 but not here": absent from registry = game never
// created it; in registry but not collected = cull; collected = shading side).
int   ps_r_light_debug   = 0;

// REAL volumetric light cones (vk_pass_lightcones): every volumetric-flagged
// SPOT light gets an analytic raymarched beam built from its true pos/dir/
// cone/range/colour — replaces the R4 `models\lightplanes` texture-sheet fakes.
int   ps_r_light_cones        = 1;      // master
float ps_r_light_cone_density = 1.2f;   // in-scatter strength (live)
float ps_r_light_cone_len     = 2.5f;   // beam length = light range × this (live)
float ps_r_light_cone_glare   = 2.0f;   // extra flare looking into the beam (live)
float ps_r_light_cone_narrow  = 0.55f;  // visible beam = lit cone × this (bright core look)

// Beam luminance in HDR scene units (2.2 ≈ sun-lit level). The honest media
// model makes beams physically vanish against a daylit background — raise
// this for the R4-style "always visible" look, lower for realism.
float ps_r_light_cone_lum = 2.2f;

// Beams synthesized from `models\lightplanes` fan geometry (car headlights,
// searchlights, halogen lamps — carriers R4 gives NO dynamic light, the beam
// is baked into the model). See vkSynthBeam / SynthCones.
int   ps_r_light_cone_synth = 1;

// Vertical lift (m) for the synthesized beam apex: the fan quads hang a few
// centimetres BELOW the lamp centre, so the cone started just under the
// headlight. Live-tunable.
float ps_r_light_cone_lift = 0.11f;   // user-tuned on the zaz headlight

// Lamp-face radius multiplier for synthesized beams. The frustum base radius
// comes from the fan's own width at the lamp — 1.0 = as authored; raise for
// a fatter "whole headlight glows" look, 0 → point cone.
float ps_r_light_cone_base = 0.2f;   // user-tuned on the zaz headlight

// Synthesized beams are REAL lights: reach = fan length × this (a headlight
// shines tens of metres, not the 5 m fan the artist modelled) — drives both
// the spot light range and the visible beam length (depth still cuts it at
// geometry). power = surface-light intensity (0 = visual beam only).
float ps_r_light_cone_reach = 6.0f;   // user: 4.0 felt short for a headlight
float ps_r_light_cone_power = 1.0f;

// Synth beams draw only a short lamp-face glow: full within ~15 cm of the lamp,
// then exp(-d/this) to fully transparent — the LONG beam shape comes from the
// froxel fog / particles (like the flashlight and sun shafts), not from a
// painted-on milk cone.
float ps_r_light_cone_fade = 0.8f;

// Per-step shadow-tap disc radius (texels) in the visible-beam raymarch — an
// area-light penumbra. The spot map resolves grass blades at mm texels, so a
// hard point tap paints razor "threads" through a grass field; the 24 march
// steps integrate the jittered disc into a soft penumbra (no temporal needed).
float ps_r_light_cone_soft = 6.0f;

// Analytic visible cones for ordinary dynamic torches (CTorch — the player's
// and NPCs' head-lamps). Default 0 = R4 behaviour: a torch is just a lit spot
// whose volumetric shaft comes from the froxel fog (r_vol), NOT a hard searchlight
// cone. The synthesized lamp/headlight beams (from lightplanes geometry) are a
// SEPARATE path and are unaffected by this flag. 1 = also draw the analytic cone
// for torches (old behaviour — NPC head-lamps read as projectors at night).
int   ps_r_light_cone_torch = 0;

// Bright lamp-face glow for a FLASHLIGHT aimed at the camera (R4 lens-flare analog).
// The torch cone is off (not a searchlight), but an NPC pointing its head-lamp AT you
// should show a bright "bulb" — otherwise the beam lights the scene while the lamp
// reads as switched off. Facing-gated (0 when aimed away → your own torch never flares
// in-face). 0 = off. Higher = brighter bulb. (5.0 was too hot — 2.0 reads as a lamp.)
float ps_r_flashlight_glow = 2.0f;

// Grass casters into the SPOT shadow map: blades cut a beam's light pool and
// its visible volumetric cone (headlight/searchlight/flashlight through a
// grass field). Same 1-frame-stale GPU instance buffer the VSM grass uses.
int   ps_r_spot_grass    = 1;

// Grass casters into the SUN cascades (r_sun_grass): swaying blades cast real
// sun shadows — drawn into the per-frame dynamic overlay (with the NPCs and
// wind trees) of BOTH near cascades. dist = instance-cull radius around the
// camera; blades beyond it are subpixel in the map anyway.
int   ps_r_sun_grass      = 1;
float ps_r_sun_grass_dist = 30.f;

// Grass casters into the POINT (campfire) cubes: blades around a near fire cut
// its light into swaying dapples (surfaces + froxel fog). Adds a cadence
// re-render for in-view fires closer than ~40 m — same LOD as the NPC overlay.
int   ps_r_point_grass    = 1;

// GPU per-light grass caster cull + compaction (Phase 1 of "grass × dynamic
// lights"): one compute counting-sort compacts near blades into per-(light,type)
// arena regions so each spot-beam / campfire-cube depth draw is a few-hundred-
// instance indirect instead of a full grass VS pass (×6 for a cube). A/B with 0
// = the old brute-force DrawGrassSpotCasters. See vk_pass_shadow.cpp (GrassCull).
int   ps_r_grass_cull       = 1;
int   ps_r_grass_cull_debug = 0;   // [VK GrassCull] per-call lights/types/cells log

// Spot shadow POOL size: how many spots get their own shadow tile per frame
// (atlas holds 8). Nearest-first, narrow beams keep a 4× distance advantage.
// Tiles are cached — a static lamp renders its tile once — so 8 costs little.
int   ps_r_spot_pool = 8;

// Campfire/brazier glow tuning (VOLUMETRIC-flagged point lights only — fires,
// pole lamps; plain table lamps stay untouched). boost = light intensity ×,
// range = reach ×. Applied consistently to shading, fog AND the shadow cube.
// range default 2 = stock fires light twice as far (user-tuned 2026-07-03).
float ps_r_point_boost = 1.0f;
float ps_r_point_range = 2.0f;

// Point-shadow DEBUG overlay: paint opaque receivers by pooled-point coverage —
// GREEN = lit by a pooled point light, RED = that light's shadow on this pixel.
// Instantly shows whether a campfire casts a light pool and an NPC shadow in it.
int   ps_r_point_debug = 0;

// Grass self-shadow anti-acne slack (m, along the sun ray): grass receivers sample
// the DYN VSM atlas (which contains the blades themselves) at their own position —
// occluders closer than this don't shadow (kills self-acne), further ones do
// (clump-on-clump dapples, NPC shadows). Lower = fuller grass shadows but risk of
// dark sparkle; higher = cleaner but shadows start further from the caster's base.
float ps_r_grass_self_bias = 0.15f;

// Sun-beam ground recovery (r_sun_beam). The volumetric shaft samples the crisp VSM
// ATLAS; surfaces sample the temporally-SMEARED screen mask (vsm_resolve), which
// closes the thin sun gap a crown casts — so the beam "dissolves" on the ground
// instead of landing as a lit pool. Terrain re-samples the crisp atlas at its own
// world pos and takes the max, recovering the gap so the ground lights up in
// agreement with the shaft; boost adds a small extra sun kick in the recovered gap
// for a cinematic splash (bloom then picks up the bright pool). dist caps it to near
// ground (resident pages + where it reads); bias = anti-acne slack (m along sun ray).
// Only active under r_vsm (crisp atlas source). 0 = off (old smeared-mask behaviour).
// ================================ PARKED 2026-07-06 ================================
// SUN-BEAM ground-lighting arc (r_sun_beam* family) — PARKED by user ("что-то не то").
// Goal was: a sunbeam cutting through a tree should "разбиваться о терейн" (land as a lit
// pool) instead of dissolving. Three approaches tried, ALL default 0 now (inert), code
// kept behind the cvars as scaffolding (do NOT delete — see the memory note + world_
// terrain.frag / tonemap.frag / vk_env_light.cpp PARKED comments):
//   (1) r_sun_beam        — crisp-atlas gap RECOVERY on terrain. Dead end: r_terrain_debug
//                           7 proved the ground is genuinely sun-occluded (no atlas gap).
//   (2) r_sun_beam_splash — tonemap in-scatter/surface splash. Read as AIR haze, not ground.
//   (3) r_sun_beam_ground — forward deposit (albedo×sun) driven by a shadow-march overhead.
//                           View-independent dapple, but user still "не то". Parked here.
// r_terrain_debug 5/6/7 (recovery probes) + 8/9 (deposit probes) left wired for a revisit.
// ===================================================================================
// Terrain recovery (r_sun_beam): DEFAULT 0 — proven a no-op in dappled-crown scenes
// (r_terrain_debug 7 showed gap≈0: the ground under a visible shaft is genuinely sun-
// occluded, so the crisp atlas agrees with the smeared mask — nothing to recover).
// Kept behind the cvar for wide gaps where the atlas DOES resolve a lit patch.
float ps_r_sun_beam        = 0.0f;
float ps_r_sun_beam_dist   = 40.0f;
float ps_r_sun_beam_boost  = 1.6f;
float ps_r_sun_beam_bias   = 0.10f;

// God-ray GROUND SPLASH (r_sun_beam_splash, tonemap.frag). The real fix for "луч
// растворяется у земли": the visible shaft is lit air your view ray gathered through
// the crown gap — bright integrated in-scatter that, at the ground, reads as thin
// haze. Where that beam luminance outshines the (shadowed, dark) ground it lands on,
// add the excess back boosted so the contact pops and bloom flares it into a splash.
// strength scales it (0 = off); thr = the in-scatter luminance below which it's just
// ambient fog and stays untouched (raise if general fog brightens; lower for subtler
// beams to splash). Aesthetic, not a relight — the sun does not physically reach there.
float ps_r_sun_beam_splash     = 0.0f;   // superseded by the forward GROUND deposit below (kept for A/B)
float ps_r_sun_beam_splash_thr = 0.12f;

// Sun-beam GROUND DEPOSIT (r_sun_beam_ground, world_terrain.frag — the "for real" fix).
// The visible god-ray is lit air the view ray gathered through the crown gap; the
// integrated froxel in-scatter at the pixel is bright even though the sun is occluded on
// the ground there (proven r_terrain_debug 7). Instead of the tonemap haze-splash (which
// read as air), the FORWARD terrain shader deposits the sun the SHADOW removed — gated by
// that beam, shaped by the ground's own N.L — into the sun lighting bucket, so it gets
// ×albedo and reads as REAL sunlight on the terrain. strength lerps the ground from
// shadowed toward fully sunlit under the shaft (0 = off = A side of the A/B); thr = the
// in-scatter luminance below which it's just ambient fog and stays untouched. Needs r_vol.
float ps_r_sun_beam_ground     = 0.0f;    // PARKED (was 1.5) — user "что-то не то"; kept behind cvar
float ps_r_sun_beam_ground_thr = 0.05f;

// Grass lighting-component debug (detail.frag): paint the grass ONLY with one
// isolated term, greyscale — attributes any pattern "seen through the bush" to
// its real source. 1=applied sun shadow, 2=VSM mask R (full, ground-behind),
// 3=mask B (static-only), 4=channel pick weights, 5=GTAO-behind, 6=SSIL-behind,
// 7=baked slot sun occlusion, 8=baked slot hemi, 9=full ambient, 10=dyn lights,
// 11=shadow source split (grey=static shadow, RED=dyn/grass-atlas shadow).
int   ps_r_grass_debug = 0;

// Point shadow POOL size: how many campfires/lamps get their own shadow cube
// per frame (array holds kMaxShadowPoints). Cubes are 6× a spot tile, so the
// cap is smaller; cached + LOD-gated like spots (static fire = one render).
int   ps_r_point_pool = 4;

// Grass shadow strength on SURFACES (spotShadowF blends the clean spot map
// with the spot+grass beam map). 0 = grass never shadows surfaces (sterile
// pool), 1 = full blanket (dense grass eats the headlight's ground pool);
// mid = translucent dapples — light scatters through grass IRL.
float ps_r_spot_grass_shadow = 0.55f;

// Grass shadow strength for FLASHLIGHT tiles only (handheld/worn torch — CTorch
// head-lamp, weapon light), SEPARATE from the wide-fixture blend above. A torch
// beam raking through a grass field at night should throw CRISP blade shadows on
// the ground (the "wow"), so flashlight tiles default to FULL grass shadow (1.0)
// while lamps/searchlights stay at the subtle r_spot_grass_shadow. Needs r_spot_grass.
float ps_r_flashlight_grass = 1.0f;

// Respect the DO_NO_WAVING detail flag (1) or force SSFX wind on every detail type
// (0, the old behaviour). R4 keeps flagged micro-plants (tiny shoots in asphalt)
// static; without this they stretched under the tree/grass wind.
int   ps_r_grass_nowave  = 1;

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
// r_rain_sun — how strongly the streaks catch the DIRECTIONAL light (sun, moon,
// lightning). Drops were tinted by hemi alone, which is flat-lit rain and, worse,
// rain a thunderbolt cannot touch: the bolt boosts sun/sky/fog colour and swings
// sun_dir to the strike (thunderbolt.cpp), and hemi is the one field it never
// writes. Water cylinders scatter FORWARD, so real rain lights up when you look
// toward the light and goes near-black when you look away — one forward lobe on
// the sun direction buys both the backlit downpour and the free lightning flash.
// 0 = the old flat hemi look.
float ps_r_rain_sun = 0.35f;

// Global render profiler (vk_profiler). 0 = no [VK Perf] logging, 1 = periodic
// (~5s) GPU/CPU/VRAM log, 2 = + the live ImGui overlay (Phase 2). `vk_perf`
// forces an immediate MARK snapshot regardless of this value.
// GPU-driven particles (gpu_particles_roadmap.md). Phase 1 — one hardcoded
// effect (Source+Gravity+KillOld) simulated + drawn entirely on the GPU.
// 0 = off (CPU PAPI path draws everything), 1 = GPU test effect at the camera.
int   ps_r_gpu_particles     = 0;       // r_gpu_particles — master GPU-particles switch
int   ps_r_gpu_particles_max = 1 << 16; // r_gpu_particles_max — particle pool cap (64K)
int   ps_r_gpu_particles_sort = 1;      // r_gpu_particles_sort — Phase 4 depth sort of alpha smoke (back-to-front)
float ps_r_gpu_particles_life_cap = 5.0f;  // r_gpu_particles_life_cap — clamp GPU particle lifetime (s): "immortal"
                                           // flames (KillOld 1000s) would hold their #5 alive budget forever and
                                           // starve every other instance of the program. 0 = off. Applied at
                                           // translate time (cached per program — takes effect on restart/new level).

int   ps_r_profiler = 1;

// Variable Rate Shading (vk_vrs, depth-driven). 0 = off, 1 = mild, 2 = aggressive.
// near/far = distance (m) thresholds: below near stays 1x1, above far goes coarsest.
// (Live — tune without restart. Level 2 pulls them ~closer.)
int   ps_r_vrs      = 0;
int   ps_r_vrs_force = 0;   // diag: force pipeline-rate NxN on the world pass (2 or 4), ignoring the SRI
int   ps_r_vrs_static = 0;  // diag: bake a STATIC 2x2 rate into world pipelines at creation (set BEFORE loading a level)
// Alpha-tested statics color path: depth EQUAL + no write (the prepass depth is
// final) -> early-Z kills occluded AT layers AND transparent texels of the front
// layer before the uber-FS runs. Found 23-07-2026 on Кордон: world-color FS
// invocations hit 29.5M vs a 4.1M-pixel screen (7.2x overshading) staring into
// layered AT (bushes/fences), World/Statics 10.5-11.8 ms. Live A/B.
int   ps_r_at_equal = 1;
// Prepass-covered statics color path: depth write OFF (prepass depth is final).
// THE early-Z fix: the world uber-FS statically contains `discard`, and
// discard + z-write ON disables early-Z entirely — the whole frustum's depth
// complexity was being shaded (gpuStatics 10.8M FS inv vs ~3.5M visible,
// r_fsinv_split, Кордон 23-07-2026). Live A/B.
int   ps_r_z_prepass = 1;
// Screen-size (SSA) cull of plain whole meshes in the GPU world cull: skip a
// mesh once its bounding sphere projects under N pixels of DIAMETER. The
// r_fsinv_split hunt (23-07-2026, Кордон) showed distant small props (<256 tris
// never cluster/LOD) rendering full geometry to the horizon = sub-pixel
// triangles = 8x quad-helper cost (gpuMesh 8.24M FS inv vs 2.26M visible
// samples). Clustered/DAG geometry is exempt. 0 = off. Live.
float ps_r_ssa_px = 2.0f;
// Diag: split the world-color FS-invocation counter 4 ways (CPU statics flush /
// GPU statics / dynamics / skinned) to attribute overshading. Replaces the
// single [VK VRS] invocations line with [VK FSinv] while on. Live.
int   ps_r_fsinv_split = 0;
int   ps_r_uber_variants = 1; // Inc 1: 1 = lean world uber-FS spec variants (POM/SNOW/WET/IBL/DEBUG); 0 = A/B the old monolithic uber (force WS_ALL). Live-switchable.
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
// + standing ≈ 0. r_vsm_debug logs per-frame page mark/alloc counts.
//
// DEFAULT ON since 12-08-2026 — VSM is now the PRIMARY sun-shadow path. The reason
// is measured, not aesthetic: under the classic cascades the per-frame near-tree
// caster layer (Sh/TreeDyn0/1) cost 6.34 ms = 65% of the whole SunShadow budget,
// and the cascade path has NO caster LOD for trees at all. Every tree caster-LOD
// mechanism in this renderer (crown hulls, voxel clouds, impostors, meshlet+HZB
// binning) lives on the VSM side, so the cheap fix and the shadow system are the
// same decision. See [[vulkan-tree-forward-no-lod-double-draw]].
// ⚠ The cascade path is NOT dead and must keep working — see the note at
// `cascRaster` in vk_pass_shadow.cpp: it is the volumetric-fog occluder while the
// VSM atlas warms up, and the froxel shader falls back to it for pages that are
// not resident yet.
int   ps_r_vsm       = 1;
int   ps_r_vsm_debug = 0;
// VSM receiver depth-compare bias (normalized clipmap Z, range ~2000 m). Larger =
// less acne but more light-leak (small/thin caster shadows fade). Live-tunable.
int   ps_r_vsm_grass_static = 1;  // far-grass hybrid: L1/L2 (12..48 m) grass RIGID into the cached static atlas
                                  // (dirty pages only ≈ free standing); L0 (±12 m) stays dynamic with live wind.
                                  // 0 = old behaviour: ALL grass every-frame dynamic, L0..L2.
float ps_r_vsm_bias  = 0.0003f;   // STATIC atlas (terrain in it → needs acne slack; 0.0003 = 0.6 m at the ±1000 m z-range)
                                  // With r_vsm_bias_min > 0 this is only the CAP of the slope-scaled bias.
float ps_r_vsm_bias_dyn = 0.00002f;   // DYNAMIC atlas (casters only, ground never in it → epsilon over D16 quantization
                                      // + write-side raster bias; the shared 0.6 m ate grass shadows below knee height)
float ps_r_vsm_bias_min = 0.00003f;   // STATIC receiver bias, constant part (0.00003 = 6 cm ≈ write raster bias + D16
                                      // quantization): the resolve's NORMAL OFFSET carries the acne band now, so this
                                      // stays small — the flat legacy 0.6 m (and any big slope slack) leaked sun
                                      // through everything thinner than the slack. 0 = legacy constant r_vsm_bias.
float ps_r_vsm_raster_bias  = 0.5f;   // atlas WRITE depth bias, constant (D16 units ≈ 3 cm each). The old 1.5 pushed
                                      // every caster ~5 cm deep — most of a plank wall's thickness gone before any
                                      // receiver got a vote. Live; changing it drops the page cache (crisp A/B).
float ps_r_vsm_raster_slope = 1.5f;   // atlas WRITE depth bias, slope part (caster-side grazing acne; old 2.5).
float ps_r_vol_vsm_bias = 0.00004f;   // fog (vol_inject) AIR bias vs the static atlas (0.00004 = 8 cm): air has no
                                      // acne to hide — needs only write bias + D16 quantization. The initial 0.2 m
                                      // lit a fog shell straight through thin roofs/walls. 0 = legacy (r_vsm_bias).
float ps_r_vol_surf_clip = 0.15f;     // depth-rejection FRONT shell (m): also kill in-scatter within X m BEFORE the
                                      // visible surface. The "lit shell" hugging thin sun-facing walls can't be fixed
                                      // by any receiver bias (atlas write bias + D16 quantization ≈ plank thickness);
                                      // fog centimetres from a wall carries no legit light anyway. 0 = off.
float ps_r_vol_depth_reject = 0.12f;  // froxel depth-rejection slack (m): fog BEHIND the geometry of its own view
                                      // column gets no in-scatter — kills the sunlit-outdoors glow the trilinear
                                      // volume fetch smears onto walls around windows. 0 = off (legacy).
                                      // MUST be well under half a froxel slice (~1 m at 8 m with the scene-far
                                      // grid): the leaking froxel is the wall-STRADDLING one, its centre sits
                                      // only ~0.1-0.5 m behind the surface — a bigger slack filters nothing.
// VSM clipmap detail: base extent (m) of clipmap level 0 → finest texel = base/4096.
// 24 m ≈ the old 4096² cascade (5.9 mm), balanced. Smaller = sharper but more pages
// (the 2048-page atlas can overflow on wide vistas → distant pages drop, graceful).
// Live-tunable; meant to back a future "VSM detail Low/Med/High" graphics slider.
float ps_r_vsm_base  = 24.0f;   // finest texel 5.9mm ≈ old cascade; temporal accumulation makes this coarse base look as clean as 12 did (user-verified) → cheap default. Future graphics slider Low/Med/High = 32/24/16.
// VSM temporal accumulation (TAA-for-shadows): EMA-blend a reprojected history in the
// screen-space resolve → smooths the moving-sun shadow-edge crawl and the staggered
// round-robin page refreshes. (No clipmap jitter — it would mismatch the toroidal
// page cache; the window is page-snapped instead.) Live.
int   ps_r_vsm_temporal = 1;
// History weight (EMA alpha): higher = smoother/stabler but more ghosting under motion;
// lower = crisper but more residual crawl. 0.9 ≈ ~10-frame convergence. Live-tunable.
float ps_r_vsm_ta_blend = 0.9f;
// History weight on pixels the DYNAMIC atlas shadows (wind-swaying crowns, NPCs): their
// casters move every frame, so the full 0.9 EMA drags a ~10-frame smear ("jelly") behind
// the shadow. Low weight = crisp sway; the edge moves anyway, so the texel crawl the EMA
// exists to hide is imperceptible there. Static-shadow pixels keep r_vsm_ta_blend. Live.
float ps_r_vsm_ta_blend_dyn = 0.35f;
// Foliage-shadow "каша" fix (3 knobs on the temporal resolve). The 0.9 EMA above is great
// for STILL scenes but smears high-frequency leaf/bush shadows into mush under CAMERA
// motion (bilinear history reproject blurs a bit more each frame; the distance reject only
// catches disocclusion, not same-surface shadow change). These target that without losing
// the still-scene detail:
// (1) neighbourhood clamp — history bound to the current shadow ±tol (kills the trail, keeps
//     sub-texel jitter averaging). 0 = off (raw EMA). ~0.2-0.3 = crisp, higher = softer.
float ps_r_vsm_ta_clamp = 0.24f;
// (2) motion-adaptive — fade the history weight to a floor by reprojected screen motion:
//     ta_motion = px of motion to reach the floor; ta_motion_floor = weight there. Still
//     camera keeps the full EMA; moving camera stops compounding the blur.
float ps_r_vsm_ta_motion = 6.0f;
float ps_r_vsm_ta_motion_floor = 0.30f;
// (3) DLSS-aware — when r_dlss is on, DLSS ALSO temporally resolves the shadow (baked into
//     the colour it upscales) → scale the VSM history weight down to avoid a double blur.
float ps_r_vsm_ta_blend_dlss = 0.6f;
// (4) NO-PAGE CARRY — weight of the reprojected history on pixels whose clipmap page is not
//     resident this frame (a scroll-in redraw deferred over r_vsm_dirty_budget). Those used
//     to resolve LIT, so a burst of deferrals BLINKED the shadows off for a few frames —
//     visible only while moving, where the motion fade above has already dropped the EMA.
//     Carrying keeps the (slightly stale) shadow until the page wins its redraw; < 1 so a
//     page that never returns decays to lit over ~a second. 0 = the old flash-lit behaviour.
float ps_r_vsm_ta_carry = 0.98f;
// SOFT SHADOWS (stochastic PCSS) — replaces the fixed 3x3 PCF in vsm_resolve with a
// blocker search + a filter disc sized by the blocker distance, so a shadow is HARD at
// the contact point and softens with the caster's height. Both discs are Vogel spirals
// rotated per pixel and per frame: the error lands as noise, which the temporal resolve
// right below already exists to average away. r_vsm_soft = filter tap count (0 = the
// legacy PCF path, byte-identical). See vsm_resolve.comp.glsl for the derivation.
// ⭐Tap counts. Were 12/8 while the cone was 2 deg wide and the disc needed the coverage;
// at the 0.2 deg below the filter disc is ~5 cm and 8/4 already sample it densely. Fewer
// taps on a SMALLER disc is also cheaper twice over: less arithmetic, and the taps land
// in the same texture cache lines instead of scattering across a metre.
int   ps_r_vsm_soft        = 8;
int   ps_r_vsm_soft_search = 4;
// Sun cone HALF-angle in degrees — the ONE knob that sets penumbra width, linearly:
// w = tan(angle) * blockerDistance, capped at tan(angle) * r_vsm_soft_range.
//
// ⚠⚠THIS DEFAULT WAS 2.0 AND THE SHADOWS WERE INVISIBLE. USER-REPORTED 19-08, confirmed
// by A/B in one command (r_vsm_soft 0 → shadows snap back instantly). 2.0 deg is 7.5x the
// real sun (0.265) and puts the cap at a 1.05 m RADIUS — a 2.1 m wide smear. Worse than
// the width alone: 12 taps over a 1 m disc sit ~0.6 m apart, so the estimate is mostly
// noise, and the temporal resolve below (blend 0.9, carry 0.98) grinds that into grey mush.
// The blocker search averages over a disc of the same radius, so a trunk at 2 m and a
// crown at 15 m end up in one number.
//
// 0.2 is a deliberate artistic choice — very slightly SHARPER than the true sun. It caps
// the penumbra at ~5 cm with r_vsm_soft_range 15: contact shadows read as crisp, and the
// softening stays where it belongs (crowns, distant roofs). Raise toward 0.6 for visibly
// cinematic; 0.265 is the physical value. ⛔Do not go back above ~1.0 without re-checking
// the tap count — the two are coupled through disc area.
float ps_r_vsm_soft_angle  = 0.2f;
// Max blocker search distance (m). Caps the widest penumbra (angle * range) AND sizes the
// blocker-search disc, so lowering it both bounds the smear and keeps the blocker estimate
// local. A caster further than this softens no further — a graceful clamp, not a dropout.
// ⚠NOT a pure quality knob despite what rspec_vk_*.ltx does with it: it multiplies into
// penumbra WIDTH, so moving it changes the look, not just the cost.
float ps_r_vsm_soft_range  = 15.0f;
// Neighbourhood clamp used INSTEAD of r_vsm_ta_clamp while soft is on: the 0.24 above was
// tuned against deterministic taps and would pin the history to the stochastic noise.
// ⭐Was 0.55, calibrated for the 2 deg disc's noise. On a 5 cm disc that much slack is pure
// smear — 0.55 on a 0..1 shadow lets the history sit half the range away from the current
// value without being clipped, which reads as a trail, not a penumbra. Tightening THIS is
// what actually restored edge definition for the user — more than narrowing the cone did.
float ps_r_vsm_soft_clamp  = 0.28f;
// Grass casts VSM shadows (near + L0 only; reads the GPU-driven detail CASTER buffer
// 1 frame stale). DEFAULT ON since 2026-07-03: the two blockers that parked it are gone —
// dyn pages now resolve with the low adaptive history weight (r_vsm_ta_blend_dyn kills the
// "see-through" ghosting) and the caster buffer is frustum-independent (blades behind the
// camera still cast). Pages also get SSFX wind now — the shadow sways with the blade.
int   ps_r_vsm_grass      = 1;
float ps_r_vsm_grass_dist = 20.0f;   // max grass cast distance from camera (m)
float ps_r_vsm_npc_dist   = 50.0f;   // max NPC shadow-cast distance into the VSM atlas (m); 0 = no cull (NPC shadows tiny past ~50m)
float ps_r_vsm_lod_dist   = 0.0f;    // VSM caster-LOD: opaque casters draw coarse slice past this (m); 0 = off (measured marginal in village, like the cascade; kept for open maps)
int   ps_r_vsm_mark_half  = 1;       // page-mark at half-res (4x fewer threads/atomics); 0 = full-res
// Resolve skips the 9 dyn-atlas PCF taps on pages no NPC/grass caster binned into (the dyn
// alloc claims a slot for EVERY visible page, so most dyn pages are cleared-empty). Big
// bandwidth save on the resolve (esp. iGPU). 0 = old behavior: sample the dyn atlas on every
// resident page (A/B knob). Verified in-game 2026-07-01 (red-overlay debug r_vsm_debug_dyn:
// NPC shadows intact, flagged pages match the visible shadows) → DEFAULT ON.
int   ps_r_vsm_dyn_gate   = 1;
// Debug visualizer: everything shadowed by the VSM DYNAMIC atlas (NPC/grass casters) is
// tinted RED by the tonemap (mask.B, written raw by the resolve, UNGATED by r_vsm_dyn_gate).
// Answers "are NPC shadows actually there?" under any weather/lighting.
int   ps_r_vsm_debug_dyn  = 0;
// VSM static-atlas CACHE: toroidal per-page residency for the static (opaque + tree) atlas —
// each frame only pages that scrolled into the window (or hit their round-robin refresh while
// the sun moves) re-render; the rest keep their cached depth (the dynamic NPC/grass atlas
// still re-renders every frame). DEFAULT ON (1): the shipping path. Measured 2026-06-19 A/B in
// the same scene with r_vol on: VSMrender 14.2 -> 2.4 ms, gpu_total 23.3 -> 12.5 ms, fps 43 -> 83.
// 0 = the render-all-every-frame baseline (A/B knob). Live.
int   ps_r_vsm_cache     = 1;
// Round-robin refresh period (frames) for the moving-sun case (toroidal cache): each
// frame ~1/N of the resident static pages re-render to track the creeping sun smoothly (no
// jump). Smaller = fresher but costlier; only active while the sun moves (paused sun → 0). Live.
int   ps_r_vsm_cache_refresh = 8;
// THROTTLE (UE5 VirtualShadowMapThrottle port): feed last frame's World/VSMrender GPU ms
// back into a clipmap LOD bias — over budget marks pages 1-2 levels COARSER (each level
// quarters the marked pages = the only lever that cuts alpha-test FILL), under budget
// recovers slowly. Engages only when the scene actually exceeds the budget; 0 = off (A/B).
int   ps_r_vsm_throttle        = 1;
float ps_r_vsm_throttle_budget = 3.0f;   // target VSMrender ms (default clips only the heavy scenes/spikes)
float ps_r_vsm_throttle_max    = 2.0f;   // max bias in clipmap levels (2 = up to 16x fewer pages)
// DIRTY BUDGET (UE5 DeferredInvalidationBudget analog): cap the scrolled-in (wrong-tile)
// static pages re-rendered per frame — the toroidal-scroll eviction bursts behind the
// 7+ ms VSMrender spikes. Excess pages defer (receivers fall back a level for a frame
// or two via the vsm_resolve walk). Inval circles/refresh are NOT budgeted. 0 = off (A/B).
int   ps_r_vsm_dirty_budget    = 128;
// ⭐r_vsm_load_freeze — the sun-shadow update behind the LOAD SCREEN. Measured on a
// pripyat_full load (18-08): the 60 precache frames cost 40 ms of GPU each, of which
// VSM mark+atlas+resolve is 21 -- spent on a world nobody can see, for camera
// directions the precache sweep invents and the player never looks at. Worse, the
// 8-frame prime that exists so the first VISIBLE frames are not full of dark squares
// burns behind that same screen. Freezing the update while the load screen is up and
// re-arming the prime the moment the world appears spends both where they show.
int   ps_r_vsm_load_freeze     = 1;
// RECEIVER MASK (UE5 VSM): vsm_mark records per page an 8×8 bitmask of the 16-texel
// cells visible receivers actually sample; the DYNAMIC tree bins (crown pass + voxel
// bricks) drop (caster, page) pairs whose footprint misses every sampled cell — culls
// instances AND alpha-test fill at sub-page granularity. Static cached pages are never
// masked (a partial page would cache incomplete). 0 = off (A/B); diag = rmaskCulled in
// the [VK Trees] hybrid/vox-cull log lines.
int   ps_r_vsm_rmask           = 1;

// GAZE refresh (vsm_mark + vsm_resid): while the sun moves, static-atlas pages the
// player is actually looking at re-render on a cadence proportional to their
// on-screen footprint (sampled-pixel count from the mark) — big/near shadows glide
// every frame like the dynamic atlas, small/far ones tick on the round-robin where
// the jump is sub-noticeable. Budget = max gaze pages/frame; px = mark samples
// (half-res: screen px / 4) for the every-frame tier, cadence = ceil(px/hits).
int   ps_r_vsm_gaze            = 1;
int   ps_r_vsm_gaze_pages      = 96;
int   ps_r_vsm_gaze_px         = 4096;

// GPU-driven world forward pass (vk_world_gpu): static opaque/AT meshes are
// compute-culled + drawn via indirect (1 draw/material group) instead of the
// per-object CPU queue, which also dedups the hierarchy double-submit. Cuts CPU
// draw-call count massively → fps win on CPU-bound / detail-heavy levels.
// r_gpu_world 0 = old CPU path (A/B). Default ON: measured ~2× fps in the village
// (CPU ~halved, World/Statics ~5× lower) + user-verified visually identical.
int   ps_r_gpu_world = 1;

// Cluster (meshlet) granularity for the GPU world cull (vk_world_gpu): static
// meshes above r_cluster_tris triangles are split at level load into ~128-tri
// clusters (meshoptimizer), so the compute frustum + Hi-Z cull operates on
// cluster spheres instead of whole material fragments — a building wall no
// longer draws entirely because one corner peeks into view. Foundation of the
// cluster-LOD (Nanite-like) system, Phase 1. Takes effect on level (re)load.
int   ps_r_cluster = 1;
int   ps_r_cluster_tris = 256;   // 256 = 2+ meshlets → the mesh can LOD at all
// Phase 2 — cluster DAG LOD: clusters carry a simplification error + their
// birth-group sphere; the cull shader picks the DAG cut whose projected error
// stays under r_cluster_lod PIXELS (live knob: smaller = finer, 1.0 ≈ visually
// lossless; 4.0 with the dither crossfade reads clean and lets trims/frames
// dissolve into facades at gameplay distances — user-tuned on Кордон).
// r_cluster_debug: 1 = clusters as flat colors (UE-style cluster
// view), 2 = colored wireframe over the scene — both show density fall off
// with distance in real time.
float ps_r_cluster_lod = 4.0f;
int   ps_r_cluster_debug = 0;
// Phase 3 — cluster-LOD SHADOW casters. For an ORTHO light the DAG cut is a
// constant world-error budget = target texel size × r_vsm_cluster_lod (in
// texels, live) — camera distance does not participate, so cached shadow
// content (VSM static pages, cascade static maps) never changes as the player
// moves: shadow LOD is invalidation-free by construction.
// r_shadow_cluster: sun far map + cascades draw the cluster cut instead of the
// per-mesh ShadowGPU set. r_vsm_cluster: the VSM static-atlas bin walks cluster
// entries with a per-clipmap-level budget (coarse levels render coarse DAG
// levels — the geometry win). Both fall back to the per-mesh path when off or
// when the cluster set is unavailable.
int   ps_r_shadow_cluster   = 1;
// r_gpu_shadows_at: alpha-tested casters (fences, bush/crown cutouts) draw
// through the SAME cluster shadow cull/indirect path as the opaques — the cull
// stops skipping meta flag bit0 and DrawShadow binds the material's diffuse for
// the discard (mirrors WorldGPU::DrawDepth). With it, the CPU shadow caster
// queues (far/cascade/fog/rain-AT/spot/point statics) stop being built at all.
// 0 = legacy split: GPU opaques + CPU FlushDepth(alphaTestedOnly) cutouts.
int   ps_r_gpu_shadows_at   = 1;
// r_vsm_at: alpha-tested statics cast into the VSM STATIC atlas (they never did
// before — fences/grates had no shadow under r_vsm). Cluster path only: the AT
// groups become per-material buffer combos and draw with a discard page
// pipeline (vsm_page_at) binding the material's diffuse. Flip invalidates the
// toroidal cache (page content changes). A/B live.
int   ps_r_vsm_at           = 1;
int   ps_r_vsm_cluster      = 1;
float ps_r_vsm_cluster_lod  = 1.0f;
// Crossfade band for DAG-cut transitions, as a fraction of the px threshold:
// within the band BOTH levels draw with complementary Bayer screen-door masks
// (prepass + color dither identically), so the swap reads as a short dissolve
// instead of a pop. 0 = hard cut (old behavior). Live.
float ps_r_cluster_fade = 0.25f;
// Phase 2.5 — component merge: SOLID fragments of one building whose AABBs
// touch are merged into a single cluster DAG, so low-poly parts (doors, window
// frames, trims) dissolve into the facade at distance instead of collapsing
// onto themselves (their own DAG has nowhere to dissolve to). Alpha-tested
// meshes stay on the per-mesh path. Takes effect on level (re)load; the DAG
// disk cache is keyed on this flag.
int   ps_r_cluster_merge = 1;

// r_cluster_cache — 1 = use the on-disk cluster DAG cache, 0 = always full-rebake.
// The rebake path is the only one that has ever corrupted memory during a load
// (see the VisualGuard note in rvk_loader.cpp), and with the cache working it is
// almost never taken — which also makes it almost impossible to reproduce. This
// forces it.
int   ps_r_cluster_cache = 1;


// Clustered forward (Forward+, vk_clustered): a compute pass bins the active
// dynamic lights into a 16x9x24 froxel grid; each fragment iterates only the
// few lights touching its cluster instead of all 16 with zero culling. Raises
// the light cap 16 -> 256 AND makes shaded pixels cheaper (the structural gap
// vs R4 deferred). DEFAULT ON (2026-07-04): Release A/B verified identical picture,
// no perf regression, and it unlocks the 256-light cap. r_clustered 0 falls back to
// the exact old per-fragment 16-light loop. v1 covers world (lmap/vlit/terrain) +
// skinned; foliage stays on the 16-light path. r_clustered_debug draws a per-cluster
// light-count heatmap (validates the cull).
int   ps_r_clustered       = 1;
int   ps_r_clustered_debug = 0;

// Framegraph depth-thrash coalescing (vk_barriers ImageState): the prepass depth
// is sampled in a RUN by SSAO -> VRS -> VSM-mark -> VSM-resolve. Each used to do
// its own ATTACHMENT<->SHADER_READ round-trip (4 round-trips = 8 depth barriers
// serializing the GPU). With this ON the depth flips to SHADER_READ once, all
// consumers run, then it flips back once (VSM::RenderAtlas uses its own atlas
// depth, so it rides inside the read window). r_fg_coalesce 0 = per-consumer
// round-trips via the SAME tracker (≈ the old behaviour, marginally stricter/safer
// barriers). DEFAULT ON (2026-07-05): user ran with it in-game, renders correctly.
// Fallback r_fg_coalesce 0 kept as the A/B safety net.
int   ps_r_fg_coalesce     = 1;

// Async compute (vk_async): record compute work into a SECOND command buffer on the
// dedicated compute queue, ordered to graphics via a timeline semaphore, so compute
// overlaps graphics raster. INCREMENT 1 = machinery + INERT probe (empty compute
// submit the graphics frame waits on) to prove the cross-queue path is stable on
// this GPU before real passes move over. Default 0 (dormant = exact single-queue
// path). Needs a dedicated compute family (else no-op). See vk_async.h.
int   ps_r_async           = 0;

// Sky specular IBL (vk_ibl): prefiltered sky-cube reflections (roughness mips) +
// a real sun GGX glint on the forward surfaces (the world sun path was diffuse-
// only). r_ibl_spec scales it, r_ibl_debug = spec field only.
//
// ⛔ DEFAULT OFF SINCE 2026-07-25 — PARKED, NOT SHIPPABLE ON THIS CONTENT. Do not
// flip this back to 1 without first fixing everything below; the machinery is sound,
// the INPUTS are not, and the result reads as a wax coating over the whole world.
//
// The blocker is that a specular term needs ROUGHNESS per material and this content
// has none. Measured, not assumed (all 118 <bump>.dds in the install, gloss = the R
// channel): p50 0.031 / p90 0.094 / p95 0.161 / p99 0.322. X-Ray authored that channel
// for R2's far weaker specular, so it sits in the bottom sixth of [0,1] and carries
// almost no signal. Every consumer therefore INVENTS a roughness, and they disagree:
//
//   statics  world_lmap/vlit_frag_body  0.85 - gloss*scale*0.75  -> in practice 0.78-0.85
//   NPCs     skinned.frag               0.6 flat  (cloth, leather, skin, metal alike)
//   trees    tree.frag                  mix(0.7, 0.5, wetF)
//   terrain  world_terrain.frag         its own
//
// F0 is 0.04 everywhere, so metal does not exist as a material either — the whole
// world is one dielectric with a uniform 4% sheen. User-visible verdict (25-07): a
// fence matte and the fence a metre away glossy, tree trunks glossy, NPCs "как воском
// облитые". Turning r_ibl_spec down does not rescue it: sheen reads as CONTRAST against
// a dark albedo, so it stays visible until the knob hits exactly 0.
//
// Two further defects found while diagnosing, both still unfixed:
//   * env_common.glsl:200 — sunSpec multiplies by ibl_params.x but NOT .y, so
//     r_ibl_spec silently does not scale the SUN highlight. The knob lies about its own
//     scope, which is why dialling it read as "no effect".
//   * skinned.frag:518 — NPCs gate the reflection by pc.hemi, a PER-OBJECT scalar,
//     while statics gate per-pixel by sOcc*skyVis. An NPC reflects sky off the side
//     that is pressed against a wall.
//
// Reviving this needs material CLASSES, which do exist even though roughness maps do
// not: the texture path already encodes them (mtl\ wood\ crete\ ston\ glas\ grnd\ act\)
// and is read at material load. A prefix->{roughness, F0} table feeding ALL FOUR paths
// from one place, plus the same sky-visibility gate everywhere, is the real fix.
// Cost of leaving it off is zero: the prefilter is gated by `if (ps_r_ibl ||
// ps_r_sky_sh)` in vk_env_light.cpp:1001, so nothing is computed for a disabled feature.
int   ps_r_ibl             = 0;
float ps_r_ibl_spec        = 1.0f;
int   ps_r_ibl_debug       = 0;

// Diffuse sky irradiance via SH9 (vk_ibl + sky_sh_project.comp). The sky ambient
// used to be ONE texel of the weather cube fetched along the surface normal at LOD
// 0 — a point sample of a sharp skybox, not an irradiance integral. Three failures
// compounded: it could not be blurred (the cubes ship single-mip BC), so terrain and
// world had to feed it the FLAT geometric normal or the fill turned into a mirror,
// killing all relief; and because X-Ray cubes are authored in the half-cube space
// (horizon at the bottom edge), an up-facing normal always landed on the zenith
// texel — the warm horizon band that carries nearly all the energy at dusk was
// unreachable. Net effect at sunset: a beautifully graded sky over uniformly-lit
// ground, which is exactly what it looked like.
//
// The fix projects the world-space probe onto 9 SH coefficients once per weather
// change (~6k texel fetches, riding the prefilter's existing immediate submit) and
// evaluates them per pixel in a few MADs. Being a real cosine-weighted integral it
// is directional in azimuth for free, and being smooth by construction it finally
// lets the DETAIL normal drive ambient. r_sky_sh 0 = A/B (falls back to the
// prefiltered probe's roughest mip, which is still azimuth-correct).
//
// r_sky_sh_ground scales the below-horizon hemisphere during projection: the sky
// cube's skirt is not sky radiance, it is where the ground is, so taking it at full
// strength would light everything from underneath. ~0.3 reads as a plausible ground
// bounce (it is still the right colour to bounce — the ground is lit by this sky).
int   ps_r_sky_sh          = 1;
float ps_r_sky_sh_ground   = 0.3f;
// r_sky_sh_debug — dump each projection's coefficients: L0 (omnidirectional level),
// L1 (the linear band: how much irradiance VARIES with direction, and toward what),
// and aniso = |L1|/L0. This is the measurement that separates "the maths downstream
// is wrong" from "the sky the probe sees has no direction in it" — if aniso is near
// zero at dusk, no receiver-side fix can make terrain directional, and the missing
// energy is elsewhere (e.g. the sun disk/aureole, which sky.frag adds ADDITIVELY at
// draw time and is therefore absent from the cubemap the probe is built from).
int   ps_r_sky_sh_debug    = 0;

// PROCEDURAL SKY (r_sky_proc) — single-scattering Rayleigh+Mie, atmosphere.glsl.
//
// Measured root cause (2026-07-20): at a late sunset the weather config supplies
// sun_color=(0.009,0.004,0.002), hemi/ambient/sky_color all NEUTRAL GREY, and the sun
// zeroed at elevation +1.4° (X-Ray does this because the classic renderer could not do
// twilight). So dusk had no warm light and no directional light anywhere in the data,
// and the world rendered uniformly grey — faithfully. No receiver-side fix could reach
// it; the light itself was missing.
//
// This derives sky radiance from GEOMETRY (sun elevation) instead of authored colours.
// Sunset then emerges from physics: the low sun's path through air is ~40x longer,
// Rayleigh strips the blue out of the beam (leaving red), and the Mie forward lobe
// wraps a warm aureole around it while the anti-solar sky stays blue — real azimuthal
// structure, which is exactly what the SH probe measured as absent.
//
// The SAME function feeds the dome AND the light probe (ibl_prefilter), so "the sky
// you see" and "the light you get" cannot drift apart. r_sky_proc 0 = legacy cubemap.
int   ps_r_sky_proc        = 0;      // default OFF until verified — big look change
float ps_r_sky_intensity   = 22.0f;  // maps the model's physical units onto game exposure
float ps_r_sky_turbidity   = 1.0f;   // aerosol multiplier: 1 = clean air, >1 = hazy/dusty
float ps_r_sky_mie_g       = 0.76f;  // Mie anisotropy — tightness of the sun's aureole
// Drive the DIRECTIONAL sun colour from the same atmosphere model instead of the
// config's sun_color. This is what actually re-lights dusk: the config says 0.009,
// physics says "warm and low". Needs r_sky_proc.
int   ps_r_sky_sun_from_atmo = 1;
float ps_r_sky_sun_scale     = 4.0f; // sun irradiance scale (transmittance is 0..1)

// VOLUMETRIC CLOUDS (r_clouds_vol) — raymarched 3D medium, clouds.glsl + vk_clouds.
//
// Replaces BOTH cloud sources at once: the ones painted into the weather cubemap and
// the flat scrolling R4 cloud dome. The painted ones need no suppression heuristic
// (which is how this was originally planned) because the procedural sky never samples
// that cube; and the scrolling dome is skipped in the shader when this is on — one
// kind of cloud, never two.
//
// Lit by the SAME atmosphere model that draws the sky, which is the entire point: at
// dusk the reddened low sun lights cloud UNDERSIDES warm while their tops stay cool
// from the blue zenith, and thin edges glow through the forward scattering lobe.
// Authored cloud colours (which measure neutral grey at that hour) cannot do this.
int   ps_r_clouds_vol           = 0;      // default OFF — opt-in, and the most expensive effect here
float ps_r_clouds_coverage      = 0.55f;  // 0 clear .. 1 overcast (also the shader's off switch)
float ps_r_clouds_density       = 1.0f;
float ps_r_clouds_detail        = 0.35f;  // edge erosion strength
float ps_r_clouds_bottom        = 1500.f; // deck base (m)
float ps_r_clouds_top           = 6000.f; // deck top (m) — the band the type gradient spans
float ps_r_clouds_shape_scale   = 0.00008f;
float ps_r_clouds_detail_scale  = 0.0008f;
float ps_r_clouds_weather_scale = 0.000012f;
float ps_r_clouds_wind_dir      = 45.f;   // heading (deg)
float ps_r_clouds_wind_speed    = 12.f;   // m/s
float ps_r_clouds_phase_g       = 0.72f;  // forward lobe (silver lining toward the sun)
float ps_r_clouds_phase_g_back  = 0.35f;  // back lobe (glow with the sun behind you)
float ps_r_clouds_extinction    = 0.08f;
float ps_r_clouds_powder        = 0.7f;   // dark-edge term; 0 = pure Beer (edges wrongly bright)
float ps_r_clouds_sun           = 12.f;
float ps_r_clouds_ambient       = 1.0f;
int   ps_r_clouds_steps         = 48;     // primary march budget — THE perf knob (was 96: unaffordable at fullscreen sky)
float ps_r_clouds_max_dist      = 30000.f;// marched chord cap. 90k made low-elevation steps ~1 km — clouds fell between samples
float ps_r_clouds_cirrus        = 0.5f;   // high 2D ice layer: the parallax that reads as "different heights"
float ps_r_clouds_cirrus_alt    = 9000.f;
float ps_r_clouds_cirrus_scale  = 0.00004f;
// r_clouds_debug — show the marched coverage unlit: 1 = raw alpha, 2 = alpha x20.
// The question a blank-looking sky raises is whether the model produces ANY density,
// and guessing at that has already cost round trips. This answers it directly.
int   ps_r_clouds_debug         = 0;
// r_clouds_weather — drive coverage from the weather's own cloud weight
// (clouds_color.w) instead of the flat cvar. Without it an hour the mod authored as
// solid overcast renders as the same scattered puffs as a clear one, because the
// procedural deck never looked at the weather at all.
int   ps_r_clouds_weather       = 1;

// Hi-Z occlusion cull for the GPU-driven static color pass (vk_world_gpu, Phase A
// of cluster cull). Pass_World builds a depth pyramid from this frame's PREPASS
// depth, then a compute pass frustum+occlusion-tests the static set into a second
// indirect buffer drawn only in the (heavy forward) color pass — meshes fully
// behind nearer geometry skip shading. The depth prepass still draws the full
// frustum set (builds the pyramid), so this never over-culls. Needs r_gpu_world 1.
// No effect if the occlusion pipeline failed to build.
// Default ON since 17-07: the footprint math got its missing focal scale
// (lodParams.w) — the old understated footprint picked too fine a mip and
// false-culled (vanishing walls/terrain tiles, why this stayed situational).
// Measured on Pripyat: color-pass draws 11118 → 3704 (×3 fewer). 0 to A/B.
int   ps_r_hzb_cull = 1;

// GPU-driven LOD imposters (vk_LODManager): compute cull (distance + frustum +
// shared HZB) + vertex-pulling draw replace the CPU per-frame FLOD walk that
// cost 6-9.4 ms on Pripyat. 0 = the original CPU path (instant A/B, no rebuild).
int   ps_r_lods_gpu = 1;

// Froxel volumetric lighting (vk_volumetrics, r_vol) — P1. A 3D froxel grid over
// the frustum: compute injects sun in-scatter (Henyey-Greenstein phase × cascade
// sun-shadow) + height/base fog, integrates it front-to-back, and the tonemap
// composites scene*transmittance + in-scatter (HDR, pre-tonemap). = god rays
// through geometry + depth fog (the Metro base look). Default OFF. r_vol_height 0
// = uniform fog; r_vol_debug shows the raw integrated in-scatter pattern.
// ⚠ The five numbers below were RE-TUNED FROM SCRATCH on 2026-07-24, in-game, after
// the fog finally received the sky hemisphere instead of the ambient floor (see
// r_vol_ambient). Every value tuned before that was compensating for air that could
// not scatter sky light at all, so the old set is not a starting point — it is the
// shape of the bug. Do not "restore" them.
int   ps_r_vol           = 1;       // ON by default — shipped feature (god-rays + depth fog + indoor haze), ~1.5ms
float ps_r_vol_density   = 0.005f;  // DUST layer extinction (V-1 layer 1): thin "volume in the air", NOT a fog bank — see r_vol_mist for that
float ps_r_vol_height    = 0.40f;   // dust height falloff above the BAKED GROUND (0 = uniform)
float ps_r_vol_g         = 0.88f;   // Henyey-Greenstein anisotropy (forward scatter) — user-verified: "makes the rays brighter near the ground"
float ps_r_vol_intensity = 3.0f;    // in-scatter brightness multiplier (raised: fog must GLOW more than it dims to read as haze)
float ps_r_vol_amb       = 0.60f;   // indoor ambient floor: fraction of sky ambient kept under a roof (0=pitch-dark interior, 1=no occlusion)
// Fog AMBIENT (sky-fill) in-scatter tint scale. The fog's ambient term used to
// inherit the SURFACE receiver floor (r_ambient_floor) — a flat white lift that
// keeps lit geometry off pure black. In the fog that neutral floor integrates
// along the sightline into a whitish haze that LIFTS the whole night scene (the
// air self-glows even with the sun down). The fog now tracks the env's real
// ambient (dark/blue at night) scaled by this instead: 1.0 = env ambient as-is,
// lower = darker night air, 0 = no ambient fog (sun beam + local lights only).
// Scale for the fog's SKY in-scatter. Rescaled with the source: the term used to be
// the ambient FLOOR (~0.02 in a measured frame) and now carries the real hemisphere
// (~0.42), i.e. ~20x more light, so the old scale would blow the fog out. Anything
// tuned before that change (a saved 4.0, say) must be re-tuned around this default.
float ps_r_vol_ambient   = 0.15f;
// Indoor density boost, ×(1+this) under a roof, applied to the ambient term only.
// It existed because the fog's sky term was the ~0.02 ambient floor: indoors that
// was nothing at all, so the air needed a x7 crutch to show up. With the real
// hemisphere feeding it the crutch over-fogs every room — re-tuned to 0 in-game.
// Kept as a knob (weather with heavy interior haze may want it back).
float ps_r_vol_indoor    = 0.0f;
float ps_r_vol_sun       = 6.0f;    // sun-beam in-scatter boost: directional shaft brightness (pops the god-ray through the ambient haze)

// Atmospheric scattering (r_atmo): physical Rayleigh (blue distance) + Mie (warm
// sun halo) in-scatter in the froxel fog = aerial perspective. NEEDS r_vol on.
// DEFAULT ON (2026-07-05): verified in-game ("вроде прикольно", nice lightning
// interaction). Sun-relative → dawn/dusk warm. r_atmo_rayleigh/_mie tune it live.
int   ps_r_atmo          = 1;
float ps_r_atmo_rayleigh = 2.0f;    // blue-distance strength
float ps_r_atmo_mie      = 0.6f;    // warm sun-halo strength
float ps_r_atmo_mie_g    = 0.76f;   // Mie forward anisotropy

// Auto-exposure TEMPORAL adaptation (eye adaptation): time constant in seconds for
// the exposure to ease toward the metered target. 0 = instant (old snappy behaviour
// where the image visibly darkens/brightens as you tilt between sky and ground).
float ps_r_exp_adapt     = 1.0f;
// Auto-exposure user knobs (vk_exposure.h reads these; tonemap + bloom share them).
// r_expo: EV-style compensation multiplier on the metered exposure (1 = neutral).
// r_expo_gray: metering target override; 0 = auto (0.58 gamma / 0.18 linear).
// r_expo_min/_max: exposure clamp range (night floor / brightening ceiling).
float ps_r_expo          = 1.0f;
float ps_r_expo_gray     = 0.0f;
float ps_r_expo_min      = 0.80f;
float ps_r_expo_max      = 2.20f;
float ps_r_vol_lights    = 2.5f;    // P2: local light (flashlight/lamp/campfire) in-scatter in fog — glow/cone strength; 0 = off
// Forward-scatter anisotropy for LOCAL lights only (torches/lamps/campfires),
// SEPARATE from the sun's r_vol_g (0.80). The sun's sharp forward peak is the
// god-ray halo; reusing it for local lights made a torch shone AT the camera
// spike into a blinding, HDR-desaturated (cold-white) glare — an NPC head-lamp
// read as a searchlight, not the soft R4 shaft. 0.35 = gentle bias (R4-like).
float ps_r_vol_lights_g  = 0.35f;
// Fog in-scatter multiplier for HANDHELD/WORN torches only (CTorch head-lamp,
// weapon light) — kept SEPARATE from fixtures (hanging lamps, searchlights,
// campfires) which keep the ×3 lamp beam boost. In practice this is the NPC
// head-lamp knob: the actor's own torch (volumetric_for_actor) and weapon lights
// (volumetric_enabled) default OFF, so only AI head-lamps scatter in the fog. A
// group of standing NPCs was flooding the air with light (scene read over-bright),
// so the shaft is dialed well below a fixture's: 0.4 = restrained R4 haze. Raise
// for a punchier beam; 0 = torches light surfaces but cast no fog shaft.
float ps_r_torch_vol     = 0.4f;
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
// r_fog_dist — range scale for the FORWARD distance fog (the legacy R4 haze in the
// world shaders: colour = mix(colour, fog_color, dist·k)). It is a THIRD fog on top
// of the froxel volume and the Rayleigh/Mie atmosphere, it is purely distance-based
// (no height profile — it cannot be anchored to the ground, so it always "follows"
// the camera), and it had no cvar at all: every r_vol_*/r_atmo experiment left it
// untouched. 1 = weather value, >1 pushes the haze back, 0 = off (A/B).
float ps_r_fog_dist      = 1.0f;
// r_vol_ground_debug — forensics for the baked terrain height field: the inject
// paints each froxel's height ABOVE THE GROUND (red 0 m → blue 10 m) instead of
// light, with green = 1 where the value came from the map rather than the
// eye-relative fallback. View it with r_vol_debug 1. Terrain-shaped bands that stay
// put while you walk = the anchor works; a flat wash sliding with the camera = the
// map read is failing.
int   ps_r_vol_ground_debug = 0;
// ── V-0: the froxel fog's radiometric foundation ────────────────────────────────
// r_vol_hillaire — energy-conserving slice integration (Frostbite/Hillaire) instead
// of "slice in-scatter × transmittance at the slice's FRONT face". The old form let
// light born deep in a slice cross it unattenuated, so error grew with density:
// thick fog DARKENED the scene faster than it lit it (the "dirty smoke" look) and
// brightness was non-linear in density. 0 = legacy, for A/B.
int   ps_r_vol_hillaire  = 1;
// r_vol_albedo — single-scatter albedo: sigma_s = sigma_t * albedo. It was hardcoded
// to 1, which welded "how brightly this air glows" to "how much of the world it
// hides" — the reason thin air could not carry visible shafts. 1 = old behaviour;
// lower = more absorbing (smoke/dust). The "god rays in clear weather" hybrid wants
// a THIN, HIGH-albedo, strongly forward-scattering medium.
float ps_r_vol_albedo    = 1.0f;
// r_vol_noise_scatter — put the animated 3D noise on the SCATTERING coefficient
// instead of the extinction (1 = new). On extinction it makes the air's opacity
// flicker, and since the noise field drifts on a wall clock while the temporal pass
// reprojects froxel centres, history never agrees with the current frame → the fog
// hisses. On the scattering side the same wisps read as shape in the light and the
// transmittance the composite multiplies by stays smooth. 0 = legacy.
int   ps_r_vol_noise_scatter = 1;
// ── V-1: shape. Where the fog stops being a flat pall and starts being lit air. ──
// r_vol_ms — multiple-scattering octaves (Wrenninge). Single scatter saturates dense
// fog at the source radiance, so shadowed air stays flat and reads as dirty grey:
// the light that would really bounce several times inside the medium is simply
// missing. Each octave halves the scattering weight and the phase eccentricity and
// softens the shadow term, which is what fills shadowed fog with a soft glow.
// Weights are normalised (shape, not gain). 1 = single scatter (old look).
int   ps_r_vol_ms        = 3;
// r_vol_g_back / r_vol_g_mix — the BACKWARD phase lobe. One HG lobe can only do the
// forward halo, so the sun never dominated the ambient veil at other angles and the
// fog looked directionless. Negative g = backward.
float ps_r_vol_g_back    = -0.30f;
float ps_r_vol_g_mix     = 0.25f;
// ── V-1: TWO MEDIA LAYERS (r_vol_mist*) ────────────────────────────────────────
// The froxel medium used to be ONE exponential layer, which cannot be both things
// the scene needs. Tuned thin enough that clear weather stays clear (the in-game
// answer was r_vol_density 0.005, "very light, just some volume for the scene"),
// no gain will ever make a valley read as a fog bank; tuned thick enough for the
// bank, the whole map turns to soup. So a SECOND, independent layer is summed into
// the same sigma_s/sigma_t: dense, ground-hugging, with its own phase. The grid and
// the integrator never learn there are two — cost is a few ALU in the inject.
//   r_vol_density/_height/_g = layer 1 "DUST": thin, tall, sharp forward lobe → the
//     god rays that should be visible even without fog.
//   r_vol_mist*              = layer 2 "MIST": the actual fog bank.
// 0 = off → byte-identical to the single-layer look, so this is a clean A/B.
float ps_r_vol_mist        = 0.0f;   // mist extinction AT GROUND LEVEL (compare: dust is 0.005; a real bank is 0.1-0.5)
float ps_r_vol_mist_h      = 4.0f;   // mist layer thickness in METRES (e-fold above the baked terrain height)
// Mist phase. Dense water droplets scatter far less directionally than thin dust —
// a fog bank glows as a body instead of throwing a razor shaft, and giving it the
// dust's 0.88 forward lobe is exactly what makes cheap volumetrics look like smoke.
float ps_r_vol_mist_g      = 0.55f;
// r_vol_mist_relief — metres of terrain drop over which the mist fades IN. 0 = the
// layer sits everywhere at a uniform thickness (a weather-wide fog). Above 0 the
// inject compares this froxel's baked ground against a ~32 m neighbourhood and
// keeps the mist only where the terrain sits BELOW its surroundings: milk in the
// hollows and riverbeds, clear air on the ridge. Costs 4 height-map taps, and only
// for froxels the layer actually reaches.
float ps_r_vol_mist_relief = 0.0f;
// r_vol_mist_micro — the SMALL scale, which the level-wide field physically cannot
// reach: at 71 cm/texel a wheel rut or a shell crater is averaged into flat ground.
// The RAIN map is 1024² over ±75 m around the player = 14.6 cm/texel, five times
// finer, and it already gets rebuilt as you walk — so the mist can thicken inside
// ruts, ditches and craters near the camera. Strength = how many times denser at
// full dip depth (0 = off); _micro_h = the dip depth in metres at which it saturates
// (0.25 = ankle-deep). ⚠ The rain map stores the TOP-MOST surface, so the shader
// only trusts it where it agrees with the baked ground — under a roof or a canopy
// this term switches itself off rather than floating mist onto the roof.
// r_vol_mist_micro is an ABSOLUTE extinction — the density of the air at the bottom
// of a fully-resolved dip — NOT a multiplier on r_vol_mist. That independence is the
// whole point and was learned in game: as a multiplier, the only way to get visible
// fog into the wheel ruts was a bank so thick the player could not stand in it. It
// also works with r_vol_mist 0, which is a look in its own right: clear air over the
// field, mist lying only in the tracks and ditches.
// _micro_h is the dip depth (m) at which the effect saturates; a wheel rut measures a
// few centimetres, so 0.10 = "a rut counts fully", 0.5 = "only real ditches count".
float ps_r_vol_mist_micro   = 0.0f;
float ps_r_vol_mist_micro_h = 0.10f;
// r_vol_mist_noise — the mist layer's SHAPE. An exponential falloff has a perfectly
// smooth top surface, and at fog-bank density that surface reads as a wall of grey
// standing in the hollow: the layer looks like a solid filled to a level line rather
// than fog lying in the terrain. This warps the layer's HEIGHT with fbm, so its top
// rises and falls by ±(this many) layer thicknesses. Note r_vol_noise cannot do this
// job — V-0 moved that one onto the scattering coefficient on purpose, where it
// shapes light rather than shape. Defaults ON (the layer itself is off by default,
// so r_vol_mist stays the clean A/B) and drifts slowly, like valley air.
float ps_r_vol_mist_noise       = 0.6f;
float ps_r_vol_mist_noise_scale = 0.05f;   // 1/m — ~20 m billows
// ── V-1b: the bank is SCENERY — dense to look at, liveable to stand in ────────────
// A fog bank that reads well from a hilltop (cloud lying at the foot of the slope) is
// unpleasant to walk into: inside it, optical depth accumulates from zero metres and
// every direction saturates to milk within a few steps. Physically correct, and not
// what the fog is here for. So the density the player SWIMS IN is decoupled from the
// density the player LOOKS AT — the bank stays a picture, and stepping into it costs
// visibility without taking it away.
//   r_vol_mist_fade   0 = off (byte-identical to the plain layer)
//                     1 = IMMERSION: thin the whole bank once the CAMERA is in it
//                         (this is the literal request: "walk in and it eases off")
//                     2 = NEAR: thin only the metres around the camera, so the bank
//                         keeps its body further out and you stand in a soft clearing
//                     3 = both (whichever thins more wins)
//   r_vol_mist_inside = what the mist thins DOWN to, as a multiple of the DUST layer
//                       (r_vol_density). 2 = "twice the general haze"; expressing it
//                       against dust rather than as a fraction keeps the knob's
//                       meaning when the bank density changes. 0 = mist vanishes
//                       entirely inside; a value ≥ mist/dust means no thinning at all.
//   r_vol_mist_near   = radius (m) of the mode-2 clearing.
// ⚠ Applies to the BANK only. The micro mist in the ruts (r_vol_mist_micro) is
// deliberately untouched: it is thin already and it is precisely what you are meant
// to keep seeing at your feet.
int   ps_r_vol_mist_fade   = 1;
float ps_r_vol_mist_inside = 2.0f;
float ps_r_vol_mist_near   = 45.0f;
// r_vol_mist_inside_h — the height over which "I am inside the bank" fades out.
// ⚠ NOT the layer thickness, and tying it to that was a real bug: with a 4 m layer,
// stepping onto a 6 m roof (or jumping off anything raised) took the eye "out" of the
// bank, switched the thinning off, and made the world DENSER a metre higher than it
// was below. Whether the camera shares the low ground with the bank is the hollow
// gate's job; this only has to answer "am I above the whole thing", which is a
// question on the scale of tens of metres — a hilltop, not a porch.
float ps_r_vol_mist_inside_h = 25.0f;
// ── GRASS CANOPY in the sun term (r_vol_canopy) ──────────────────────────────────
// The fog's sun in-scatter used to walk straight through a field of grass, and no
// amount of shadow-map work fixes that: grass only casts within ~48 m of the camera,
// at froxel scale its shadow is a stipple that PCF averages back into "half lit", and
// it sways, so what little survives shimmers. A field of grass is not a crowd of
// blades, it is a MEDIUM — so the sun term is attenuated by how much canopy the light
// had to cross (Beer-Lambert along the slant path to the sun), from a field baked
// once per level out of the level's own detail-slot grid. Works at ANY distance,
// never flickers, costs one texture fetch below the canopy line.
//   r_vol_canopy   = extinction per metre of fully-covered canopy (0 = off, clean A/B)
//   r_vol_canopy_h = taste multiplier on the authored model height (the canopy the
//                    light sees is not the blade tip — 1.0 = as modelled)
// ⚠ Composed with the shadow-atlas occlusion via min(), not multiply: inside 48 m
// both describe the SAME grass, and a product would double-darken right where the
// player stands. Verify the baked field FIRST with r_vol_ground_debug 3 + r_vol_debug 1.
float ps_r_vol_canopy   = 1.2f;
float ps_r_vol_canopy_h = 1.0f;
// ── V-2: WEATHER DRIVES THE MEDIUM (r_vol_weather) ───────────────────────────────
// The froxel fog used to read three things from the weather (sun dir, sun colour,
// hemisphere) and take its DENSITY from a cvar, while the forward distance haze is
// weather-driven end to end. Change the weather and one fog thickened while the
// other stood still — two systems disagreeing in the same frame, and no knob could
// reconcile them because they had no shared input. These fields were already in the
// descriptor, unread:
//   r_vol_w_flat    clouds_color.w → blends the PHASE toward isotropic. This is the
//                   one that actually reads as overcast: dimming is eaten by
//                   auto-exposure (the r_vol_sun 12 lesson), flattening the lobe is a
//                   change in SHAPE — the shafts dissolve into an even glow.
//   r_vol_w_clouds  …plus a modest damp of the sun term, r_vol_w_sky lifts the sky
//                   term to match. ⚠ The weather's own sun_color already dims in
//                   overcast profiles, so both stack with it — keep them modest.
//   r_vol_w_fog_dust  fog_density scales the DUST layer. ⚠ Read this before touching
//                   the two knobs above: fog_density is NOT "how much ground fog", it
//                   is where the forward haze STARTS (fog_near = (1-density)*0.85*
//                   fog_distance). Its honest counterpart is the thin tall layer,
//                   which is itself the distance haze — and matching those two is the
//                   whole reason V-2 exists.
//   r_vol_w_fog     extra MIST from fog_density. DEFAULT 0, on purpose: hanging the
//                   ground bank on that parameter shipped soup. A dry, hazy morning
//                   sits at fog_density 0.90, which doubled the bank AND lifted it out
//                   of the hollows — a fog day invented out of a number that never
//                   claimed one. Raise it only if a weather set really does mean
//                   ground fog by it.
//   r_vol_w_rain    rain/wetness ADD to the bank — THIS is the bank's weather driver,
//                   because ground fog is about moisture, not about sight distance.
//                   Added, not replaced: a dry level keeps the hand-tuned look.
// The hollow gate (r_vol_mist_relief) is relaxed in proportion to the WEATHER'S SHARE
// of the bank: only fog the rain grew ignores the terrain, so a dry level keeps its
// valley pattern exactly and a wet one climbs onto the ridges.
// Transitions are smoothed by an accumulator (~45 s soak / ~3 min dry) — weather
// changes are ramps, not steps. r_vol_weather 2 logs the inputs once a second, which
// is the fast way to see whether THIS weather even carries them.
int   ps_r_vol_weather    = 1;
float ps_r_vol_w_clouds   = 0.5f;
float ps_r_vol_w_sky      = 0.5f;
float ps_r_vol_w_flat     = 0.7f;
float ps_r_vol_w_fog      = 0.0f;
float ps_r_vol_w_fog_dust = 0.5f;
float ps_r_vol_w_rain     = 0.15f;
// r_vol_w_wet_force — test hook. The rain-driven bank is the half of V-2 a dry level
// can never show, and "wait for the weather script to bring rain" is a bad way to
// verify a feature. -1 = use the real weather; 0..1 = pretend that wetness.
float ps_r_vol_w_wet_force = -1.0f;
// r_vol_term — isolate ONE in-scatter source in the froxel fog: 1 sun beam, 2 sky
// ambient, 3 local lights, 4 atmosphere (Rayleigh/Mie), 5 smoke media, 0 = normal.
// The fog is a sum of five sources and tuning it blind means pushing on whichever
// one happens to be quiet: r_vol_sun x4 and two phase-model upgrades all landed on
// the SUN term and were invisible away from the sun, which says something else is
// painting the picture. This answers which, in one look.
int   ps_r_vol_term      = 0;
// r_vol_upsample — reconstruction filter for the froxel volume in the composite.
// The grid is 256×144, i.e. ~7-8 screen pixels per froxel; the old composite took ONE
// trilinear tap with a screen-STATIC dither, so the volume's own resolution showed
// through as blocks (obvious with r_vol_ta 0) and the temporal pass had to hide it.
// 1 = rotated 4-tap tent over ±0.5 froxel (3 extra 3D taps, ~0.05 ms at 1080p); the
// pattern also rotates per frame WHEN an upscaler resolves, turning DLSS into free
// supersampling of the volume instead of a preserver of fixed grain. 0 = old path.
int   ps_r_vol_upsample  = 1;
int   ps_r_vol_ta        = 1;       // temporal accumulation (jitter + reproject prev frame): smooths the froxel grid → clean dense fog
float ps_r_vol_ta_blend  = 0.92f;   // history weight (EMA): higher = smoother but more ghosting on motion
// (r_vol_shadow and its dedicated per-frame fog sun-shadow map were REMOVED
// 12-08-2026, per the TODO that stood here: the "trembling shafts" it was built for
// were the temporal reprojection, fixed in vol_inject.)
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
float ps_r_pom_far    = 30.0f;
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
float ps_r_pom_shadow = 3.0f;
// POM contact AO: view-independent groove darkening on the AMBIENT term, so the
// relief reads with depth even out of direct sun. 0 = off, 1 = default.
float ps_r_pom_ao     = 0.3f;
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
// r_shade_debug — lighting-component isolation in world_lmap/world_vlit/world_terrain:
// 1 albedo, 2 baked lmap/vertex data, 3 hemi-occ scalar, 4 GTAO, 5 ambient sky gate,
// 6 sky hemisphere colour, 7 total lighting (no albedo), 8 flat-ambient term, 9 sun
// term, 10 wetness factor. 0 = off.
int   ps_r_shade_debug = 0;
// Per-orientation POM strength. Walls/fences (horizontal normal) always full;
// floors (normal up) scale toward r_pom_floor; ceilings (normal down) toward
// r_pom_ceil. Defaults: floor 0.75, ceiling 0.25 (overhead POM reads strong).
float ps_r_pom_ceil   = 0.25f;
float ps_r_pom_floor  = 0.75f;
// Terrain POM is EXPERIMENTAL / off by default: the ground is viewed at grazing
// angles almost always and its base texture is high-contrast, so POM there
// "swims"/mirrors. Walls/fences/floors-of-structures (lmap/vlit) keep POM.
int   ps_r_pom_terrain = 0;
// Terrain POM DEPTH OFFSET (SSFX deffer_terrain_high_flat_d.ps port): sink the
// terrain gl_FragDepth into the parallax cracks in the depth prepass AND the
// color pass, so GTAO and the VSM screen resolve see the carved micro-surface —
// AO darkens the pits, sun shadows wrap into them ("cut-in" ground, not painted).
// 1 = SSFX strength (0.11 m max sink along the view ray). Needs r_pom_terrain 1.
float ps_r_pom_zoff = 1.0f;
// TERRAIN COMPOSITE CACHE (vk_terrain_cache): compute-bakes the 4-channel splat
// ground (mask x heights x offsets x height-blend) into a camera-anchored
// height+weights clipmap once per camera-window move; the terrain POM then
// marches ONE texture instead of four (color pass AND depth prepass). Pure
// perf/architecture — the picture should be near-identical to the 4-march path.
// 0 = off (old path), 1 = on when the cache module is alive.
int   ps_r_terra_cache = 1;
// Cone-step marching on the composite cache: the bake also stores a
// conservative "empty cone" ratio per texel (height .g) and the POM march
// leaps guaranteed-free gaps instead of walking 24 fixed layers - crisper
// cracks at grazing angles (no stair-stepping / missed thin walls), usually
// cheaper. 0 = fixed-layer walk on the cache (A/B fallback).
int   ps_r_terra_cone = 1;
// Terrain detail-blend transition depth. 0 = plain mask cross-fade — asphalt
// fades SMOOTHLY into soil, the GAMMA/SSFX look (their HeightBlending() ships
// commented out). >0 = Mishkinis height blend: the RAISED material wins
// per-texel with this transition width — sharp interlocking edges (0.25 = the
// old hard look; the sharpness was OURS, not GAMMA's).
float ps_r_terra_blend = 1.f;
// Sun-horizon channel in the cache bake (needs r_terra_cone's pass): per-texel
// max slope toward the CURRENT sun azimuth -> POM self-shadow becomes a
// geometric one-tap tan(sun) vs tan(horizon) test instead of an 8-tap march
// (exact over 32 texels, re-baked when the sun drifts ~2.5 deg). Strength
// still rides r_pom_shadow. 0 = old march (A/B fallback).
int   ps_r_terra_horizon = 1;
// Terrain DETAIL NORMAL MAPPING strength (independent of POM). Perturbs the
// ground normal from the per-channel <detail>_bump maps (grass/asphalt/earth/
// gravel), blended by the splat mask — feeds sun + dyn lights only (the sharp
// sky cube stays on the flat geometric normal, else up-facing ground mirrors).
// No UV march → no grazing-angle "swim". 0 = off; 1 = full.
float ps_r_terrain_normal = 3.0f;
// Terrain MICRO contact AO: darkens micro-grooves using the detail-normal tilt
// (cavity) AND the detail height (R4 terrain AO = detail diffuse alpha). Cheap,
// no UV march, no swim — deepens the relief the detail normals create. 0 = off.
float ps_r_terrain_ao    = 0.125f;   // SSFX TERRAIN_POM_AO parity
// Terrain debug view: 0 off, 1 = world normal (Nw*0.5+0.5), 2 = micro-AO,
// 3 = detail height (splat-blended detail alpha).
int   ps_r_terrain_debug = 0;
// Terrain DRY sun gloss: a material-aware specular highlight from the bump .r
// channel (R4 gloss). Asphalt/gravel catch the sun even when dry; grass stays
// matte. Fades out as the ground wets (the wet reflection takes over). 0 = off.
float ps_r_terrain_gloss = 2.0f;
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

// MUD footprints: the SAME persistent deform texture (vk_deform), read by the
// terrain shader on SOFT splat channels (earth/grass press, asphalt doesn't) —
// feet compress the POM ground and leave dark wet prints, no snow required.
float ps_r_mud_deform = 1.f;    // r_mud_deform — mud footprint strength (0 = off; shading + normal dimple)
float ps_r_mud_depth  = 0.45f;  // r_mud_depth — POM carve depth in prints (fraction of the detail height range)

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
// r_puddle_geo — put puddles in REAL ground dips. The SSFX recipe places them by
// texture micro-height times a per-level HAND-PAINTED mask, so on a level without
// that mask a metre-scale hollow in the asphalt collects nothing: the water level
// only ever sees centimetre texture relief, and the macro placement is noise that
// knows nothing about the ground. This blends in the Surface Field's concavity
// (SF_Concavity — top-down curvature, sky-exposure gated) so water pools where the
// ground is actually concave. 0 = pure SSFX/noise placement, 1 = follow the dips.
float ps_r_puddle_geo    = 0.75f;
// r_wet_dist — how far (m) wet shading survives. Ours faded over 70->35 m, so the
// ground was wet underfoot and dry thirty steps out: on open terrain that reads as
// a wet patch following the player rather than a rained-on world. SSFX fades its
// wet gloss over 250->200 m (rain_patch_normal.ps). The ripple field keeps its own
// tight fade, so the extra range is one cube tap on already-wet pixels.
float ps_r_wet_dist      = 200.0f;
// r_terrain_detail_dist — how far (m) the terrain detail NORMAL / cavity AO / gloss
// survive. Was a hard 45->25 m fade: past it the ground drops its bump taps, so the
// asphalt loses grain, micro-AO and its specular glint in one step and reads as flat
// matte paint. SSFX samples the bump at any distance (only their POM march stops, at
// 20 m). Costs 4 taps per terrain pixel inside the range — lower it if it shows up.
float ps_r_terrain_detail_dist = 120.0f;
// r_bolt_flash — how much a lightning strike lifts the HEMISPHERE light. The engine
// spends a bolt by boosting sun_color and swinging sun_dir at the strike, which grows
// a second sun (with god rays) wherever the renderer draws the sun — several per clap,
// held on screen by the volume's temporal history. We hold the real sun direction
// instead (SunDirVisual), which would cost the flash entirely, and pay it back here:
// a strike is a huge distant AREA light, so it belongs in the sky term. 0 = no flash.
float ps_r_bolt_flash = 1.0f;
// Material normal + gloss on STATICS (`<bump>.dds`, see bump_common.glsl). This path
// had neither: wall relief came only from the `#` height through POM, and the sky
// specular ran on one invented roughness for the whole world.
// r_bump       — normal-map strength (0 = off, the pre-existing look).
// r_bump_debug — 1 draws the decoded WORLD normal, 2 the gloss. The R4 unpack is
//                .wzy/.x; a mod texture authored as plain RGB would come out with
//                inverted normals, and that is far easier to see as a colour field
//                than as "something is off with the walls".
// r_gloss_scale— how hard the material gloss drives the IBL roughness (R4 called
//                this r2_gloss_factor). ⚠ INERT while r_ibl is 0 (the shipped
//                default): roughness is consumed ONLY inside the SPEC_IBL block, so
//                with no specular term there is nothing for it to modulate. Left
//                wired for whoever revives IBL — see the r_ibl note above.
//
// r_bump itself stays ON and is INDEPENDENT of IBL: bumpNormal perturbs the shading
// normal that sun, hemi and dynamic lights all consume, not just the reflection. That
// half of this feature is verified good (user, 25-07: surface grain on walls reads);
// only the gloss half died with r_ibl.
float ps_r_bump        = 1.0f;
int   ps_r_bump_debug  = 0;
float ps_r_gloss_scale = 1.0f;

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

// =========================================================================
// WATER BODIES (vk_pass_water) — level ponds / rivers / flooded basements.
// A DIFFERENT thing from the r_water_* family above (that one is the parked
// puddle flow sim); hence the separate `r_wtr_*` prefix. Before this pass
// existed, `effects\water` geometry rode the opaque vert-lit path and — with
// zero baked vertex light and zero baked sky access on a water polygon —
// rendered pure BLACK on every level.
// =========================================================================
int   ps_r_wtr        = 1;      // master enable
int   ps_r_wtr_debug  = 0;      // 1 flat magenta (does the pass run), 2 column depth, 3 wave normal, 4 alpha, 5 Fresnel
// Wave SLOPE amplitude (not height — the surface is a per-pixel normal field).
// Scaled at runtime by wind velocity and the weather's own m_fWaterIntensity.
float ps_r_wtr_wave   = 0.26f;
// What a DEAD CALM still leaves of that wave. wind_velocity is zero for most of
// this game's day, and the old implicit floor (0.35) took 0.19 m down to 0.071 —
// measured, in the [VK Water] wind line. Water is never actually glass.
float ps_r_wtr_calm   = 0.75f;
float ps_r_wtr_scale  = 0.22f;  // base wave frequency (1/m) — lower = longer swells
float ps_r_wtr_speed  = 1.0f;   // dispersion speed multiplier
// Extinction per metre of water column (Beer-Lambert). The single biggest
// "which game is this" knob: 0.2 = a clear mountain lake, 0.9 = the Zone's
// green-brown murk where the bottom vanishes at knee depth.
float ps_r_wtr_murk   = 0.9f;
float ps_r_wtr_refl   = 1.0f;   // sky reflection strength
float ps_r_wtr_glint  = 1.0f;   // sun glitter strength
float ps_r_wtr_detail = 60.0f;  // distance (m) over which the short chop fades out (anti-shimmer LOD)
float ps_r_wtr_rough  = 0.045f; // base specular roughness (grows with distance in the shader)
// Murk colour — what the body scatters back. Green-brown by default: the
// standing water of the Zone, not a swimming pool.
Fvector3 ps_r_wtr_color{ 0.085f, 0.135f, 0.105f };
// SHORELINE. Absorption alone leaves the water/terrain seam a hard cut, because
// Fresnel puts a floor under the alpha (at a grazing angle even a millimetre of
// water stays a mirror). r_wtr_shore fades the whole layer out over the last
// centimetres of depth so the edge dissolves; the foam band then hides what is
// left of it — a shore reads as a shore because of the scum line.
float ps_r_wtr_shore  = 0.22f;   // fade-out depth (m)
// SWASH: how far the waterline rides up and down with the passing wave. This is
// what makes the water WASH OVER the sand instead of ending at a painted line.
// A multiple of the wave height, so a calm pond barely moves and a rough one
// visibly climbs the beach.
// SWASH — the run-up. ⚠ These are METRES now, not a gain on the swell height.
// The old meaning multiplied the wave and was added to a DEPTH to widen an alpha
// fade, so 2.2 turned a 19 cm wave into 42 cm of "water" standing on the bank with
// nothing on screen to account for it. The surge is its own function now (front,
// fast up / slow drain, in sets) — see waterSwash in water_common.glsl.
// ⛔ PARKED 05-08 with the whitewater, same call: the shore break is OFF by
// default and the waterline is static again. Everything below still works and is
// two numbers away — r_wtr_surf_h 0.30 brings the arriving crest back, and this
// one the tongue that runs up the sand.
float ps_r_wtr_swash  = 0.0f;    // metres the biggest surges climb above the still line
float ps_r_wtr_swash_t = 6.0f;   // seconds between arriving crests
// SHORE BREAK (water_shore.glsl). Phased by the still water DEPTH, not by the
// wind — which is what makes a crest arrive parallel to the beach in every bay
// without a shoreline being authored anywhere. Iso-depth lines are the shoreline.
//
// ⛔ PARKED 04-08 at the author's call: "пока что решил отключить, может потом
// доделаю". The whitewater never looked right on this content — the run-up tongue
// reads as white sheeting over grass rather than as surf, and the last round left
// it plausible but not good. The MOTION it drives is kept (the waterline still
// arrives and drains, and the wetness still follows it); only the white is off.
// Turn it back on with r_wtr_surf 1 — nothing else needs changing.
float ps_r_wtr_surf     = 0.0f;   // whitewater strength
float ps_r_wtr_surf_h   = 0.0f;   // offshore height of the arriving wave (m) — PARKED
float ps_r_wtr_surf_len = 11.0f;  // metres between crests
// ⚠ Metres of water in the RUN-UP SHEET. A swash tongue is a couple of
// centimetres deep and you see the sand through it; the run-up HEIGHT is how far
// it climbs, not how much water is standing there. Feeding the height into the
// depth fade put a 45 cm slab of opaque water on the bank with a vertical face.
float ps_r_wtr_film     = 0.05f;
// Foam the backwash leaves ON THE GROUND once the sheet has drained off it. Costs
// no extra channel: the wetness map's decay already carries "how long ago", and an
// exponent rescales that clock (see shoreFoam).
//
// ⛔ PARKED 04-08 with r_wtr_surf, same call and same reason. The mechanism is
// sound and cheap; what it paints on this content is not. r_wtr_foam_land 0.55
// brings it back.
float ps_r_wtr_foam_land = 0.0f;
float ps_r_wtr_foam_life = 2.5f;   // seconds it takes to dissolve
float ps_r_wtr_foam   = 0.45f;   // foam strength (0 = off)
float ps_r_wtr_foam_w = 0.45f;   // foam band depth (m)
// MICRO-RIPPLE: short capillary waves, always present regardless of weather.
// The swell dies with the wind; this does not — without it a windless hour
// renders a mirror-flat sheet. Fades out over the first ~25 m (below a pixel
// past that, and pure shimmer if kept).
float ps_r_wtr_micro  = 0.012f;
// How much the weather's wind moves the SWELL. 1 = calm ~0.45x, storm ~2.6x;
// 0 = ignore the wind and always use r_wtr_wave as-is.
float ps_r_wtr_wind   = 1.0f;
// FETCH — how literally to take the SIZE of each body of water. The wave field
// is a function of world position, so without this every puddle, cellar and
// water butt got the same 28-metre open-water swell, complete with run-up and
// a shoreline foam line. A wave needs room: octaves whose wavelength does not
// fit in the surface's own bounding box several times over are dropped, which
// leaves a small pool with nothing but capillary ripple and the interactive sim.
// 1 = take the box at face value; >1 = pretend pools are larger (rougher);
// 0 = off, every pool is open sea again (the old look, for comparison).
float ps_r_wtr_fetch  = 1.0f;
// ⚠ ...AND "the surface's own bounding box" is where that idea leaked. Level
// water is not one mesh per pool: Cordon is ONE water visual, 1952 triangles,
// 79.8 x 148.2 m, so the fetch above came out as 108 m for every drop of water on
// the level — including the two-metre circle inside a well in the newbie village.
// Measured there with r_wtr_audit (16-08): pool 2.9 m2, span 2.0 x 1.9 m, still
// sheet at y=-19.98, and the wetness map holding marks up to -19.55. The drawn
// surface was climbing 37 cm above its own level in 22 cm of water — over the
// stone ring, out into the yard as a floating square, and soaking a ring of dirt
// that dried and re-wetted with every crest.
//
// This gates the fix: a compute pass (water_fetch.comp) measures each column's
// own body of water out of the POOL MASK, which knows where water ends at 12.5 cm,
// and the consumers take the smaller of that and the box. It can only ever say
// SMALLER — anything reaching the search radius is written as "no answer" and
// keeps the box — so open water is untouched by construction.
// 0 = back to the bounding box alone (the A/B).
int   ps_r_wtr_fetch_local = 1;
// SHELTER — the other half of the same problem: wind waves need WIND, and there
// is none under a roof. Queried from the sky-occlusion map, hemispherically (a
// pond under a pipe rack still catches wind from the sides, so a straight-up
// test would stamp dead-calm blobs under every overhead object). Scales the
// swell only: sheltered water keeps its capillary ripple and still takes rings
// from a footstep, which is what standing water indoors actually looks like.
// 0 = a roof does nothing, the old look.
float ps_r_wtr_shelter = 1.0f;
// Murk multiplier for sheltered water. A pond is flushed by rain, inflow and
// wind mixing and stays comparatively clear; a flooded cellar has none of that,
// so the silt stays in suspension and the bottom goes dim within centimetres.
// Multiplies r_wtr_murk, scaled by the same shelter query. 1 = indoor water is
// as clear as a pond (the old look).
float ps_r_wtr_murk_still = 7.0f;

// ---- SPECTRAL WAVES (vk_water_fft) ----------------------------------------
// Tessendorf's FFT ocean: the sea built in the Fourier domain from the Phillips
// spectrum, inverse-transformed to a displacement map on the GPU every frame.
// It exists for ONE thing the nine-octave analytic swell cannot do at any
// setting — the HORIZONTAL displacement that makes a crest sharp. A sum of
// sines displaced along Y has round tops however tall you make it.
int   ps_r_wtr_fft        = 1;
// Largest cascade's tile, metres, and the shrink to the next one. 250 / 0.175
// gives 250 m, 43.8 m and 7.7 m — each carrying only the band between its own
// size and the next one down, so nothing is represented twice and the small
// tile's repeat never shows.
float ps_r_wtr_fft_size   = 250.f;
float ps_r_wtr_fft_ratio  = 0.175f;
// ⚠ A BASE wind speed in m/s, NOT a scale on the weather's. Measured, not
// assumed: wind_velocity in this content is 0 for most of the day, and the
// analytic swell was effectively dead for exactly that reason until it was
// given a floor. The weather modulates around this (0.65x .. 1.55x); it cannot
// zero it. This is the main knob for "how rough is the sea".
// ⭐ 5 m/s, and the number was MEASURED, not picked. The Phillips peak sits at
// lambda = 2*pi*sqrt(2)*V^2/g, so the wind speed is really a knob for WAVE
// LENGTH: at 7 m/s the peak is a 44 m swell, which on a 60 m pond is not a wave
// at all, it is the water level tilting. At 5 m/s the peak lands at ~23 m and
// the 7.7..44 m cascade carries 7x the energy of the long one — waves you can
// see across a pond. Height is then set independently by the amplitude below,
// which is exactly why Phillips has both knobs.
float ps_r_wtr_fft_wind   = 5.f;
// Phillips A, in units of 1e-6. 28 puts the significant wave height at 0.29 m in
// dead calm and 0.75 m in a storm — measured by summing the spectrum over the
// same three cascade grids the shader uses. It has to be this large BECAUSE the
// wind was lowered: energy in this spectrum goes as roughly V^4, so moving the
// peak down to a useful wavelength costs a factor of ~30 in height.
float ps_r_wtr_fft_amp    = 28.f;
float ps_r_wtr_fft_chop   = 1.1f;    // lambda: how far a crest pulls water in sideways
float ps_r_wtr_fft_depth  = 60.f;    // sea depth for the dispersion relation (m)
// The whole field repeats EXACTLY this often: every wave's frequency is snapped
// to a multiple of 2*pi/repeat, so nothing drifts out of phase over a long
// session and the sea at minute 40 is the sea at minute 0.
float ps_r_wtr_fft_repeat = 200.f;
float ps_r_wtr_fft_small  = 0.06f;   // capillary cutoff (m) — kills the sub-centimetre noise
float ps_r_wtr_fft_dir    = 2.f;     // how sharply the spectrum favours the wind direction
float ps_r_wtr_fft_gain   = 1.f;     // how much of the transformed field to use (0 = none)
float ps_r_wtr_fft_foam   = 0.7f;    // whitewater from the Jacobian (folded-over water)
// The body span at which a cascade reaches full strength — the spectral field is
// gated per band by the SAME fit test the analytic octaves use, so a 250 m
// cascade is simply absent from a flooded cellar while the 8 m one still lives
// there. This is what stops an open-sea model from putting surf in a basement.
float ps_r_wtr_fft_fetch  = 60.f;
float ps_r_wtr_fft_slope  = 1.f;     // shading response, independent of the displacement

// ---- INTERACTIVE RIPPLES (vk_water_ripple) --------------------------------
// A 2D wave-equation heightfield on a tile that follows the camera. The half of
// "water" no analytic wave can fake: a sum of sines cannot propagate, reflect,
// interfere — or REACT. Footsteps and rain drops write into this field and the
// waves travel outward on their own.
int   ps_r_wtr_sim        = 1;
float ps_r_wtr_sim_size   = 64.f;    // tile edge (m); 512 texels -> 12.5 cm each
// Velocity retention PER STEP at 60 Hz — the LIFETIME of a ripple. Damping
// multiplies velocity, so an oscillating wave's amplitude decays as damp^30 per
// second: 0.995 leaves 86% after a second, 0.97 only 40%.
//
// This knob spent two days doing a job that was never its own. "A splash in my
// puddle shows up in the far ones" was fought here, by making waves die before
// they could travel — and it worked, at the price of the thing the sim exists
// for: at 0.97 a footstep ring was gone before the eye caught it. The two
// demands are not reconcilable on one dial, because they are not the same
// question. Lifetime is time; a puddle boundary is SPACE.
//
// The boundary now lives where it belongs — r_wtr_sim_lid, which asks the
// collision model which water is roofed by a floor and therefore is not a puddle
// at all. With the wave properly confined, lifetime is free to be generous
// again, which is what this value is: rings that spread, reflect off the rim and
// take a few seconds to fade.
float ps_r_wtr_sim_damp   = 0.995f;
float ps_r_wtr_sim_speed  = 0.35f;   // stiffness; the explicit scheme blows up past 0.5
// POOL BOUNDARIES. The sim is one heightfield over a 64 m tile, so without a mask
// of where the water actually IS, a splash in one puddle propagates across the dry
// floor and surfaces in every other puddle in range. The mask (drawn each frame
// from the water meshes, straight down) turns each body of water into its own
// pool with reflecting rims. 0 = the old unbounded field.
int   ps_r_wtr_sim_pools  = 1;
// LID SOURCE. The mask says where the water IS; what separates puddles is where
// the water can be SEEN, and that is decided by the FLOORS above it. Level water
// is a couple of huge sheets that dive under the buildings, so a cellar's
// "puddles" are one slab surfacing through an uneven floor — connected, and a
// wave crossing between them is correct physics that no mask may forbid.
//
// The rasterized lid was meant to catch this and cannot: after pool compaction
// the statics are drawn only by the GPU-driven paths, and the CPU queue it walks
// held five leftovers ("lid map: 5 static(s)"). 1 = ask the COLLISION MODEL
// instead — one short ray up from the surface per texel, immune to which path
// draws the world. 0 = the old rasterizer.
int   ps_r_wtr_sim_lid      = 1;
// Ray budget per frame. A CDB trace is a couple of microseconds, so this is the
// dial between "the lid lags behind a sprint" and "filling it hitches the frame".
int   ps_r_wtr_sim_lid_rays = 1500;
// SHALLOW WATER. The same sweep that finds the floor above also finds the bottom
// below, which gives the water COLUMN — and a wave's speed is sqrt(g*h). Letting
// the stiffness follow the depth makes a ripple slow down as the bottom rises,
// turn to face the shore and bunch up on it, none of which a constant-speed field
// can do. It also retires shoreAbsorb: a wave stops at the rim because it runs
// out of depth, not because a multiplier was told to stop it.
int   ps_r_wtr_sim_depth     = 1;
// Depth at which the wave runs at the full r_wtr_sim_speed. Not a physical
// constant: the grid's CFL limit caps the absolute speed well below sqrt(g*h) at
// this texel size, so what is reproduced is the SHAPE of the relationship —
// shallow is slower than deep by the right ratio — not the metres per second.
// Set it from the profile of the BOTTOM, not from the mean: the lid reports 74%
// of open water under 20 cm on this content, so a mean of 0.36 m (dragged up by
// a deep minority) still ran three quarters of the pool slowed down. This is
// puddle scale. Everything at or above it runs at full speed; below it the wave
// shoals — so the value is really "where does the rim begin".
float ps_r_wtr_sim_depth_ref = 0.15f;
float ps_r_wtr_sim_bed       = 0.99f;   // per-step damping at zero depth (1 = none)
// SHORE ABSORPTION, applied once per dry side of a texel per step. A rim that
// merely reflects loses nothing: one splash bounces around the pool until the
// whole sheet is standing waves — and since a cellar's "puddles" are one 20 x 13 m
// slab surfacing in several places, that meant jumping in one of them rippled all
// of them. Bleeding energy at the shore keeps the disturbance where it was made.
// 1 = the old reflecting rim.
float ps_r_wtr_sim_shore  = 0.90f;
// REACH: metres of full-strength ripple around the player, fading to nothing at
// twice that. The audit proved the far "puddles" are the SAME body of water as
// the near one — a cellar floor is one 20 x 13 m slab surfacing in patches — so a
// wave reaching them is physically correct and no boundary test can forbid it.
// The eye still expects a splash to stay where it was made, so the disturbance is
// bounded in space. 0 = unbounded (waves cross the whole sheet again).
float ps_r_wtr_sim_reach  = 6.0f;
float ps_r_wtr_sim_slope  = 1.0f;    // how much of the sim reaches the NORMAL
float ps_r_wtr_sim_height = 1.0f;    // ...and the geometric DISPLACEMENT
float ps_r_wtr_step       = 0.075f;  // dip a footstep punches into the surface (m)
// BOW WAVE, as a multiple of the footstep dip. A body wading does not only make
// holes, it displaces water forward — and every source in this sim was a
// symmetric dip, so nothing was ever pushed. Scales with how fast you are moving.
// 0 = off (waves only where the feet land, the old behaviour).
float ps_r_wtr_bow        = 1.2f;
// SHORE WETNESS. Water that touches a bank, a wall or a heap of rubble leaves it
// dark and glossy, and it dries afterwards — until now every wet surface in this
// renderer was wet because of RAIN, so a river ran past bone-dry ground. The
// tile remembers CONTACT (see vk_water_ripple), which is what lets a receding
// waterline leave a wet strip that fades instead of switching off.
float ps_r_wtr_wet      = 1.0f;    // strength (0 = off)
float ps_r_wtr_wet_dry  = 25.0f;   // seconds from soaked to dry
// CAPILLARY RISE: metres of damp above the line the water actually TOUCHED.
//
// ⚠ It used to mean something much larger and much cruder — a flat band added to
// the still waterline everywhere the sheet reached in plan view, which is a
// contour line, not a shore: at 0.35 on a 1:10 beach it painted three and a half
// metres of sand wet that no wave had been near, from the first frame, for ever.
// The run-up strip is now measured (the wave that is drawn against the ground
// that is there), so this went back to being what it says: the few centimetres
// that soak UP a stem, a stone or a bank above the water's own line.
float ps_r_wtr_wet_lift = 0.06f;

// ---- TESSELLATION ---------------------------------------------------------
// Stock water meshes are 4-7 METRES per triangle (see the [VK Water] census),
// so there is nothing to displace until the surface is subdivided. tessMax is
// spent quadratically toward the camera; past tess_far the surface is flat and
// the wave is a normal again.
// CEILING on the subdivision of one edge, not the factor itself — water.tesc
// budgets by metres of surface per segment now, because a factor means something
// quite different on a 2 m edge and on the 200 m ones this game's sheets have.
// 64 is the hardware limit; below ~32 the long swell stops resolving at all.
float ps_r_wtr_tess      = 64.f;
float ps_r_wtr_tess_near = 8.f;
float ps_r_wtr_tess_far  = 70.f;
float ps_r_wtr_disp      = 1.0f;     // metres of displacement per unit wave height
// NEAR-FIELD GRID: quads per side of the dense camera-locked water mesh (0 = off,
// the level's own polygons only). ⚠ This is the piece that makes a wave possible
// at all: stock water is a handful of triangles up to 200 m across, and hardware
// tessellation caps at 64 per edge, i.e. ~3 m per segment — coarser than the waves
// themselves. 128 over the 64 m ripple tile is 50 cm a quad.
int   ps_r_wtr_grid      = 128;

// ---- REFRACTION + CAUSTICS ------------------------------------------------
// Until now the bottom arrived as the destination of the blend: correct, but a
// blend cannot BEND what is behind it, so the riverbed sat dead straight under
// a rippling surface. The pass now takes a half-res copy of the scene and
// samples it itself — which also buys per-channel absorption (red dies first)
// and a place to put the caustics.
float ps_r_wtr_refract   = 0.035f;   // UV offset per unit surface slope
// Caustics: the sunlight web on the bottom. Computed from the DIVERGENCE of the
// same wave field (no texture, no photon pass) — where the surface is concave
// the refracted rays bunch up.
float ps_r_wtr_caustic   = 0.55f;
float ps_r_wtr_caustic_p = 1.6f;     // exponent — higher = thinner, sharper web

// Bent-normal specular occlusion of wet/puddle sky reflections (UE-style). 0 = off,
// 1 = full; fades reflections in crevices / under overhangs where sky can't reach.
float ps_r_spec_occ     = 1.0f;

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

// Texture streamer (vk_texture_stream.{h,cpp}).
//  r_txstream          — 1 enables dynamic per-frame mip promote/demote (opt-in;
//                        the load-time budget cap + memory_priority are always on).
//  r_txstream_budget   — texture VRAM budget in MB (0 = auto: device budget − headroom
//                        − non-texture usage).
//  r_txstream_headroom — MB of device VRAM kept free (never filled by textures) so
//                        render targets / FG buffers / spikes always fit.
// Default ON since 16-07: Stage A verified in-game (Pripyat marathon 15-07) —
// GPU feedback + idle-free swaps + VMA-aware budget + 256px visible floor. On
// light maps the streamer idles by design (everything fits, zero swaps).
int ps_r_txstream          = 1;
int ps_r_txstream_budget   = 0;
int ps_r_txstream_headroom = 512;
//  r_txstream_loadcap  — LOAD TIME, not VRAM: the largest base dimension a
//                        budget-fit texture may arrive at DURING a level load
//                        (0 = off). The existing budget-fit only demotes when the
//                        card is full, so on an 8 GB card a level pulls in every
//                        texture at full res: measured on pripyat_full 17-08,
//                        2458 files = 3558 MB read + 2556 MB uploaded = 8.4 s of a
//                        21 s load. Mip 0 alone is ~3/4 of a chain's bytes, so
//                        arriving coarse and letting the feedback-driven promote
//                        path (StreamStep) sharpen what is actually on screen buys
//                        most of that back. Quality heals in seconds of play.
// r_tex_prefetch — level texture prefetch (VK::TexPrefetch, vk_texture.cpp).
// The loader opens ~2460 .dds one at a time; on pripyat_full that is 2860 of the
// 6565 ms texture phase spent inside FS.r_open on ONE thread. The set is known
// as soon as the level shader table is parsed, so workers open them ahead.
// 0 = old serial behaviour (kept for A/B — the walk itself is unchanged).
// r_tex_prefetch_mb caps the bytes parked ahead of the walk; workers stall at the
// cap and resume as the walk consumes. Floor 64 MB.
// r_upload_wc_copy — 1 = streaming (non-temporal) stores when filling the upload
// staging ring, 0 = plain CRT memcpy. The ring is mapped WRITE-COMBINING, where
// `rep movsb` measured 1.97 GB/s and made the staging copy the single largest item
// left in a level load's texture phase. Kept as a switch because it is a CPU-level
// micro-optimisation whose payoff is hardware-dependent — A/B it, don't assume.
int ps_r_upload_wc_copy = 1;

// r_upload_cached — 1 = allocate the upload staging ring in HOST_CACHED memory
// instead of the usual write-combined one. Read ONCE, when the command manager
// builds the ring, so it only takes effect from user.ltx / a restart.
int ps_r_upload_cached = 0;

// r_upload_copy_threads — extra threads that help fill the staging ring, 0 = the
// caller copies alone. Only pieces of 256 KB and up are split. Measured on
// pripyat_full's 1869 MB of geometry: 2 / 4 / 8 helpers all land within noise of
// each other (432 / 406 / 420 ms) because the copy is bound by its SOURCE, not by
// the writes. Helpers idle on a condition variable, so this costs nothing outside
// a load.
int ps_r_upload_copy_threads = 4;

// r_geom_prefault — diagnostic for the geometry stage: 0 = off, 1 = touch every
// page of the mapped window before copying it, 2 = PrefetchVirtualMemory over the
// window. Both cost the same faults, but they time them apart from the copy.
int ps_r_geom_prefault = 0;

// r_geom_lead — 1 = stage level.geom out of the texture prefetch's whole-file view
// (its pages are being faulted in by sixteen workers), 0 = out of the loader's own
// sliding window, which faults every 4 KB on the loading thread. Here to A/B the
// two in one binary; there is no reason to run with it off.
int ps_r_geom_lead = 1;

// r_vis_warm — 1 = sixteen workers fault the visuals blob in alongside the walk
// that reads it (it is a span of the mapped level file, so every 4 KB otherwise
// costs the loading thread a fault). Interleaved pairs: visual phase -46/-37/-86 ms.
int ps_r_vis_warm = 1;

// r_vis_walk_split — 1 = rdtsc split of that loop (open_chunk / header / Create+Load
// / close). Four counters per visual, so it is off unless asked for.
int ps_r_vis_walk_split = 0;

// r_vis_triage — 1 = the per-visual glass/cabinet/glow probes in LoadTexture. They
// answered "which shader does this pane use" once and have cost a lowercased heap
// string per visual ever since. Off by default; turn it on when a NEW routing
// question needs the map.
int ps_r_vis_triage = 0;

// r_vis_guard — 1 = the between-phases visual integrity sweep (VisualGuard). It
// was written to name the phase that corrupted Visuals[] during the repeat-load
// heap bug; that bug is fixed (the quadtree pool freed an uninitialized slot), so
// what is left is a diagnostic that dereferences 450k scattered visuals SIXTEEN
// times per load — every checkpoint is two passes over 3.6 MB of pointers plus a
// cache miss per object. Off by default; turn it on when Visuals[] is suspect
// again, and the sweep will name the first phase that broke it.
int ps_r_vis_guard = 0;

// r_tex_repack_threads — helpers for the BC3->BC4 lightmap gather (0 = the loading
// thread alone). A level's hemi maps are 4096^2, and the gather is 16 bytes in / 8
// out per block with nothing to synchronise, so it splits cleanly. Uses the
// persistent staging-ring pool, not fresh threads per texture.
// r_tex_prefetch_lmaps_first -- 1 = the prefetch parks the level's lightmaps
// before the 2400 diffuse bases. The walk needs a lightmap in its first
// milliseconds and used to miss all 27 of them (432 MB read twice).
int ps_r_tex_prefetch_lmaps_first = 1;

// r_prewarm_async -- 1 = the WET/SNOW weather pipeline variants are compiled on a
// worker thread instead of one per precache frame, and the precache stops waiting
// for them. Measured on pripyat_full: 28 precache frames become the 12-frame floor
// (~12 ms a frame of load screen spent compiling variants nothing draws until it
// rains). 0 = the old 1/frame pace on the render thread, for the A/B.
int ps_r_prewarm_async = 1;

// r_thm_cache -- 1 = remember what a texture's .thm said. The file is opened and
// parsed TWICE per material: once by the prefetch worker expanding the base into
// its file set, once by the walk creating the material. Keyed by base name + level
// tag (a .thm can resolve level-locally). 0 = the old re-read, for the A/B.
int ps_r_thm_cache = 1;

// r_clpage_map -- 1 = the pinned cluster pages are uploaded straight out of a
// mapping of the page cache file, with the pages faulted in by workers first.
// The old path fseek/fread 336 MB of a 2.2 GB file into a scratch buffer on the
// loading thread and then copied that into staging: two copies and a serial read.
// 0 = the fread path, for the A/B.
int ps_r_clpage_map = 1;

// r_tex_residency_threads — helpers that read a residency plan's .dds files before
// the images are built (0 = the old serial read inside each build). The tree path
// raises 42 materials to a 1024 floor in one batch and measured read 45 of its
// 57 ms: 42 archive entries LZO-decompressed one after another on the thread the
// level is waiting on. The image create + staging copy stay serial.
int ps_r_tex_residency_threads = 8;

int ps_r_tex_repack_threads = 4;

// r_tex_materialize - 1 = a prefetch worker takes a texture all the way (read,
// parse, repack, image, upload, publish into the cache the walk looks in) instead
// of reading the file and parking the bytes. Parking only ever moved the READ off
// the loading thread: of the 451 ms the walk spent on textures, the read was 59 and
// the other 392 (repack, image creation, staging copy) stayed on the one thread the
// level waits for, while sixteen workers sat blocked on the parking budget.
// 0 = the old park-and-hand-over path; 1 = build from the first job; 2 = read and
// park during the geometry stage, then build once the walk starts.
//
// ⛔ DEFAULT 0 — MEASURED LOSS, kept as the instrument that proved it. The walk's
// texture time does fall (538 -> 105 ms), and the load gets LONGER anyway:
//   mode 1: geometry 352 -> 832 ms. Texture uploads mixed into the geometry stage
//           drag the shared ring from 18.5 to 8.4 GB/s; geometry hands back every
//           millisecond the walk saved (3182 vs 2972 ms total).
//   mode 2: the walk keeps its texture time (596-656 vs 538). Sixteen workers each
//           fill the ring single-file (one upload mutex) while the machine is busy,
//           and the ring copy falls to 6.4 GB/s -- see RingCopy in vk_command_buffer.
// The lesson: that 538 ms is not CPU work waiting for cores, it is a page-fault-
// bound copy into one ring that ALREADY uses every core, one copy at a time.
// What is still worth trying is the other split: workers parse/repack/create, the
// walk keeps the upload (its measured ceiling is the 258 ms that is not upload).
int ps_r_tex_materialize = 0;

// r_tex_mat_threads - how many workers may be BUILDING a texture at once. The
// pool is sixteen because reading is what it was sized for; building ends in the
// upload ring, which is one mutex, and a machine with every core busy is a bad
// place to fill it. 0 = no limit.
int ps_r_tex_mat_threads = 0;

// r_tex_repack_scratch — 1 = the gather writes into one buffer per thread, grown to
// the largest image and kept, instead of a fresh xr_malloc per texture. Half a
// gigabyte of destination per load is otherwise faulted in a page at a time and
// handed straight back.
int ps_r_tex_repack_scratch = 0;

// r_tex_repack_verify — 1 = rebuild every gather with the plain per-block copy and
// compare the two buffers. Doubles the repack cost, so it is a proof run, not a
// setting.
int ps_r_tex_repack_verify = 0;


int ps_r_tex_prefetch    = 1;
int ps_r_tex_prefetch_mb = 1024;

int ps_r_txstream_loadcap  = 0;

// r_txstream_plan_live - 1 restores the pre-19-08 load-time budget fit, which asked
// the driver "how full is the card" once per texture. That answer is only right if
// textures are created AFTER everything else the level allocates; it is what the
// serial walk happened to do. Kept as the A/B lever for the arithmetic that replaced
// it (base at load start + level.geom's share + what the plan has already committed).
int ps_r_txstream_plan_live = 0;

// r_vram_small_images - 1 routes sub-1MB images into the 8 MB-block small pools,
// which costs a driver size probe (vkGetDeviceImageMemoryRequirements) in front of
// EVERY non-dedicated image allocation. 0 skips both. The pools exist because small
// long-lived allocations pinned 64 MB blocks (1740 MB of unreleasable slack on
// Pripyat, 18-07); the knob is here to price the probe against that.
int ps_r_vram_small_images = 1;
//  r_txstream_reserve  — MB of device VRAM that must be FREE after a level finishes
//                        loading. If less is free, the streamer trims the largest
//                        world textures (at the load-end idle point) until it is.
//                        Leaves room for Streamline Frame-Gen buffers / spikes so a
//                        heavy map (Pripyat) doesn't crash Streamline at first present.
int ps_r_txstream_reserve  = 1024;

// Cluster-LOD page streaming (vk_cluster_stream, Stage B of the streaming world).
//  r_clpage        — 1 = keep only a budgeted pool of cluster-IB pages resident
//                    (pinned roots/coarse always in VRAM; GPU feedback streams the
//                    rest from the cluster cache file, coarser DAG levels cover
//                    while pages arrive). 0 = eager install-all (pre-Stage-B
//                    behavior/VRAM). Read at level load.
//  r_clpage_budget — resident page-pool budget in MB (pinned pages always fit).
// Default ON since 16-07: verified on Pripyat (resident ~85 vs 326 MB, request
// feedback converges to zero, no holes/flicker; self-loop cut closed the last
// stuck-request tail). 0 = eager install-all remains the parity fallback.
int ps_r_clpage        = 1;
int ps_r_clpage_budget = 160;
// r_cl_audit — heavyweight streaming/draw forensics (the 17-07 white-polygon
// hunt): install-time page payload validation, per-frame GPU slot copy-back
// hash audit, 3s draw-contract validation, and the full indirect-set draw
// audit (cmds+counts readback, CPU cut replica, HZB cross-diff). Costs host
// memory (~35 MB on Pripyat) + a few ms every 3 s. Read at LEVEL LOAD (the
// audit buffers are created then) — set it before loading. Logs: [VK ClPage]
// gpu-audit / [VK ClVB-RT] / [VK CmdAudit] / [VK ClChain].
int ps_r_cl_audit      = 0;

// Pool compaction (Stage B increment (б)): after WorldGPU::Build the vertex/
// index slices of fully-repacked clustered meshes are duplicates of the
// ClusterStream page pools — rebuild the level VB/IB pools with only the
// slices CPU paths still need (plain meshes, alpha-tested casters, terrain,
// trees, CPU leaves) and free the rest. While active, the always-CPU static
// shadow consumers (rain/ground map, spot/point lights) draw the statics via
// the cluster shadow targets, and the CPU fallbacks (r_gpu_world 0,
// r_gpu_shadows 0, r_shadow_cluster 0) are forced to the GPU paths — set
// r_pool_compact 0 (+ level reload) for a true A/B of those. Read at load.
int ps_r_pool_compact  = 1;

// Stage D world composition: total chunk count (home + clones of the loaded
// level tiled on a grid). 0/1 = off. Render-only replication slice — clones
// have no collision/AI; statics only (trees/grass/LODs home-only for now).
int ps_r_compose = 0;

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

    // ⚠⚠THE OLD FILE NAMES WERE THE PROBLEM, not the mechanism. `rspec_*.ltx`
    // ships inside the mod's archives and was written for the DX renderer: the
    // bulk of what it sets (r2_sun_quality, r__smap_cascade0_size,
    // r4_enable_tessellation, r2_volumetric_lights …) is not read by a single
    // line of Layers/xrRenderVulkan, so picking a preset moved a page of
    // settings and changed no pixels — the same placebo the rest of this screen
    // suffered from. A NEW name (rspec_vk_*) cannot be shadowed by an archive
    // and says which renderer it belongs to.
    //
    // ⭐The file is content on purpose: what "High" means is a judgement about
    // this renderer's costs, and it will keep moving. Editing five text files
    // beats editing five arrays and rebuilding.
    virtual void Execute(LPCSTR args)
    {
        CCC_Token::Execute(args);
        string_path leaf;

        switch (*value)
        {
        case 0: xr_strcpy(leaf, "rspec_vk_minimum.ltx"); break;
        case 1: xr_strcpy(leaf, "rspec_vk_low.ltx"); break;
        case 2: xr_strcpy(leaf, "rspec_vk_default.ltx"); break;
        case 3: xr_strcpy(leaf, "rspec_vk_high.ltx"); break;
        case 4: xr_strcpy(leaf, "rspec_vk_extreme.ltx"); break;
        default: return;
        }

        string_path _cfg;
        FS.update_path(_cfg, fsgame::game_configs, leaf);
        // ⚠A warning, not a refusal: `exist` answers from the file registry and
        // the path here is absolute, so a false negative is possible — and
        // refusing on it would break a preset that is actually there. cfg_load
        // is harmless on a missing file (it opens nothing); this line is what
        // explains the silence afterwards.
        if (!FS.exist(_cfg))
            Msg("! [VK preset] '%s' not found by the file registry — if nothing changes, that is why", _cfg);

        string_path cmd;
        strconcat(sizeof(cmd), cmd, "cfg_load", " ", _cfg);
        Msg("~ [VK preset] %s", leaf);
        Console->Execute(cmd);
    }
};

// Vulkan stub — VMA owns all memory; expose a dump via vmaCalculateStatistics
// in a follow-up pass if needed. R4 version queries D3DPOOL_* counters.
class CCC_VideoMemoryStats : public IConsole_Command
{
public:
    CCC_VideoMemoryStats(LPCSTR N) : IConsole_Command(N) { bEmptyArgsHandled = true; };
    virtual void Execute(LPCSTR /*args*/) { VK::TextureStreamer::Instance().DumpStats(); }
};

// `r_vram_dump` — full VMA allocator JSON (every pool/block/allocation with
// names) to _appdata_\vma_stats.json. The scalpel for "who owns the vma-slack".
class CCC_VramDump : public IConsole_Command
{
public:
    CCC_VramDump(LPCSTR N) : IConsole_Command(N) { bEmptyArgsHandled = true; };
    virtual void Execute(LPCSTR /*args*/) { VK::Vram::DumpVmaJson(); }
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

// Screenshot ON DEMAND. The renderer has always been able to capture the
// swapchain (CRender::Screenshot, writes $screenshots$/ss_*.tga) but the only
// trigger was a KEY BIND — useless to a scripted run, which has no keyboard.
// With this, a harness driving the client through XROS_CLIENT_CONSOLE can look
// at what it just built. `vk_shot <name>` writes that exact path instead.
// Throw a stone in the pond from the console: a disturbance a few metres in
// front of the camera. The one-line way to see whether the ripple sim is alive
// without wading into water first.
// r_wtr_audit — read the ripple mask + field back and report the CONNECTED
// bodies of water around the camera. Answers "are these puddles one pool joined
// off-screen?", which staring at the water cannot.
class CCC_WtrAudit : public IConsole_Command
{
public:
    CCC_WtrAudit(LPCSTR N) : IConsole_Command(N) { bEmptyArgsHandled = TRUE; };
    virtual void Execute(LPCSTR) { VK::WaterRipple::RequestAudit(); Msg("[VK Ripple] audit requested"); }
};

class CCC_WtrDrop : public IConsole_Command
{
public:
    CCC_WtrDrop(LPCSTR N) : IConsole_Command(N) { bEmptyArgsHandled = TRUE; };
    virtual void Execute(LPCSTR args)
    {
        float dist = 4.f, str = 0.25f;
        if (args && args[0]) sscanf_s(args, "%f %f", &dist, &str);
        Fvector d = Device.vCameraDirection; d.y = 0.f;
        if (d.square_magnitude() > 1e-4f) d.normalize(); else d.set(0.f, 0.f, 1.f);
        const Fvector& c = Device.vCameraPosition;
        // A deliberate probe: it should land wherever you aim it, so it opts out
        // of the source-height test the same way rain does.
        VK::WaterRipple::Splat(c.x + d.x * dist, c.z + d.z * dist, 1.0f, -str,
                               VK::WaterRipple::kNoHeightTest);
        Msg("[VK Ripple] drop at (%.1f, %.1f) r=1.0 s=%.2f", c.x + d.x * dist, c.z + d.z * dist, str);
    }
};

class CCC_VkUiPassTrace : public IConsole_Command
{
public:
    CCC_VkUiPassTrace(LPCSTR N) : IConsole_Command(N) {}
    virtual void Execute(LPCSTR args)
    {
        u32 lines = 0;
        if (!args || 1 != sscanf(args, "%u", &lines)) { InvalidSyntax(); return; }
        VulkanUI::TraceUIPass(lines);
        Msg("~ [VK UIpass] tracing the next %u transition(s)", lines);
    }
    virtual void Info(TInfo& I) { strcpy_s(I, "<lines> -- traces that many Vulkan UI render-pass transitions"); }
};

class CCC_VkShot : public IConsole_Command
{
public:
    CCC_VkShot(LPCSTR N) : IConsole_Command(N) { bEmptyArgsHandled = TRUE; };
    virtual void Execute(LPCSTR args)
    {
        if (!::Render) { Msg("![VK] vk_shot: no renderer"); return; }
        ::Render->Screenshot(IRender_interface::SM_NORMAL, (args && args[0]) ? args : nullptr);
    }
};

// An integer cvar that is NEVER written to user.ltx. Debug views are set for one
// measurement and forgotten; the client saves the console on exit, so a savable
// one strands the next launch in whatever the last experiment left behind (the
// water pass shipped magenta this way once).
class CCC_IntegerNoSave : public CCC_Integer
{
public:
    CCC_IntegerNoSave(LPCSTR N, int* V, int _min, int _max) : CCC_Integer(N, V, _min, _max)
    {
        SetCanSave(FALSE);
    }
};

// The same for a float, and for a second reason: a cvar that is really a TUNING
// CONSTANT of some scheme (a damping factor, a stability limit) has its true home
// in the source, next to the reasoning for the number. Make it savable and the
// first value ever written to user.ltx outlives every later default silently —
// the machine keeps running the old physics while the code, the comments and the
// notes all describe the new. That cost a day: r_wtr_sim_damp was lowered
// 0.9985 -> 0.97 to stop one footstep ringing a whole 20 x 13 m slab, and the
// only machine it mattered on went on loading 0.9985 from its own config.
class CCC_FloatNoSave : public CCC_Float
{
public:
    CCC_FloatNoSave(LPCSTR N, float* V, float _min, float _max) : CCC_Float(N, V, _min, _max)
    {
        SetCanSave(FALSE);
    }
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

// GPU-particles Phase 3 — `gp_mirror <effect>` translates a real .pe into the
// GPU action program and runs it on the camera-pinned test path; no arg reverts
// to the authored campfire. Forward-declared to keep the GP module headers out.
namespace VK { namespace GPUParticles { bool MirrorEffect(const char* name); void DebugEffect(const char* name); void DumpStats(); } }
class CCC_GP_Mirror : public IConsole_Command
{
public:
    CCC_GP_Mirror(LPCSTR N) : IConsole_Command(N) { bEmptyArgsHandled = TRUE; }
    virtual void Execute(LPCSTR args)
    {
        string_path name;
        xr_strcpy(name, args ? args : "");
        // Trim leading spaces the console may leave before the argument.
        char* p = name;
        while (*p == ' ') ++p;
        VK::GPUParticles::MirrorEffect(p);
    }
};

// `gp_stats` — per-program budget/alive counters to the log.
class CCC_GP_Stats : public IConsole_Command
{
public:
    CCC_GP_Stats(LPCSTR N) : IConsole_Command(N) { bEmptyArgsHandled = TRUE; }
    virtual void Execute(LPCSTR /*args*/) { VK::GPUParticles::DumpStats(); }
};

// `gp_debug <effect>` — dump the .pe def + its GPU translation to the log.
class CCC_GP_Debug : public IConsole_Command
{
public:
    CCC_GP_Debug(LPCSTR N) : IConsole_Command(N) { bEmptyArgsHandled = TRUE; }
    virtual void Execute(LPCSTR args)
    {
        string_path name;
        xr_strcpy(name, args ? args : "");
        char* p = name;
        while (*p == ' ') ++p;
        if (!*p) { Msg("![VK GP] gp_debug: usage 'gp_debug <effect>' (see gp_list)"); return; }
        VK::GPUParticles::DebugEffect(p);
    }
};

// GPU-particles Phase 3 step 2 — `gp_spawn <effect>` places a persistent WORLD
// emitter of the effect at the current camera position (does not follow the
// camera); `gp_spawn_clear` removes all such emitters. Many can coexist.
namespace VK { namespace GPUParticles { bool SpawnEffect(const char* name); void ClearSpawns(); } }
class CCC_GP_Spawn : public IConsole_Command
{
public:
    CCC_GP_Spawn(LPCSTR N) : IConsole_Command(N) { bEmptyArgsHandled = TRUE; }
    virtual void Execute(LPCSTR args)
    {
        string_path name;
        xr_strcpy(name, args ? args : "");
        char* p = name;
        while (*p == ' ') ++p;
        VK::GPUParticles::SpawnEffect(p);
    }
};
class CCC_GP_SpawnClear : public IConsole_Command
{
public:
    CCC_GP_SpawnClear(LPCSTR N) : IConsole_Command(N) { bEmptyArgsHandled = TRUE; }
    virtual void Execute(LPCSTR /*args*/) { VK::GPUParticles::ClearSpawns(); }
};

// `gp_list [substr]` — print loaded .pe effect names (optionally filtered) so
// you can discover real names to feed `gp_mirror`. Library enumeration lives in
// the compat TU (vk_PSLibrary.cpp); call it via its plain extern.
void VK_ParticleEffectFillName(xr_vector<shared_str>& dest);
class CCC_GP_List : public IConsole_Command
{
public:
    CCC_GP_List(LPCSTR N) : IConsole_Command(N) { bEmptyArgsHandled = TRUE; }
    virtual void Execute(LPCSTR args)
    {
        string256 filt;
        xr_strcpy(filt, args ? args : "");
        char* f = filt; while (*f == ' ') ++f;
        _strlwr(f);

        xr_vector<shared_str> names;
        VK_ParticleEffectFillName(names);

        u32 shown = 0;
        const u32 cap = 300;
        for (const shared_str& s : names) {
            if (!s.size()) continue;
            if (f[0]) {
                string256 lower; xr_strcpy(lower, s.c_str()); _strlwr(lower);
                if (!strstr(lower, f)) continue;
            }
            if (shown < cap) Msg("  %s", s.c_str());
            ++shown;
        }
        Msg("[VK GPUParticles] gp_list: %u effect(s)%s%s%s (of %u total)%s",
            shown, f[0] ? " matching '" : "", f[0] ? f : "", f[0] ? "'" : "",
            (u32)names.size(), shown > cap ? " — output capped at 300" : "");
    }
};

// ============================================================ r_aa_mode ====
//
// ⚠⚠THE ROW THAT LIED. `r_aa_mode` was a token over ps_r_pp_aa_mode, offering
// off / DLSS / FSR2 / TAA / SMAA — and NOT ONE LINE of the Vulkan renderer ever
// read that variable. Picking "DLSS" in the options screen changed a number in
// memory, saved it to user.ltx, and left the image untouched. (Two of the four
// techniques it offered do not exist in this renderer at all.)
//
// What does work is a pair: `r_dlss` (master) and `r_dlss_quality`
// (0=DLAA … 4=UltraPerf, i.e. how far below the display the scene renders).
// A player has no business knowing that the second means nothing while the first
// is off, so this command owns both and Status() derives the row back from them
// — which is what keeps the menu honest after a cfg_load or a console poke.
enum
{
    AA_OFF = 0,
    AA_DLAA = 1,       // render == display, DLSS used purely as an antialiaser
    AA_DLSS_Q = 2,
    AA_DLSS_B = 3,
    AA_DLSS_P = 4,
    AA_DLSS_UP = 5,
};
constexpr xr_token aa_mode_token[] = {
    {"st_opt_off", AA_OFF},
    {"dlaa", AA_DLAA},
    {"dlss_quality", AA_DLSS_Q},
    {"dlss_balanced", AA_DLSS_B},
    {"dlss_performance", AA_DLSS_P},
    {"dlss_ultra_perf", AA_DLSS_UP},
    {},
};

class CCC_AAMode : public CCC_Token
{
public:
    CCC_AAMode(LPCSTR N, u32* V, const xr_token* T) : CCC_Token(N, V, T) {}

    void Execute(LPCSTR args) override
    {
        // ⚠Every user.ltx in existence holds one of the OLD names (this install
        // has `r_aa_mode st_opt_dlss`). Rejecting them would spray InvalidSyntax
        // over every startup and, worse, leave the setting at whatever the
        // default was. Map them instead: anything that used to mean "some kind of
        // temporal AA is on" now means DLSS on, at the quality already stored.
        u32 want = u32(-1);
        for (const xr_token* t = aa_mode_token; t->name; ++t)
            if (_stricmp(t->name, args) == 0)
                want = (u32)t->id;

        if (want == u32(-1))
        {
            if (_stricmp(args, "st_opt_dlss") == 0 || _stricmp(args, "st_opt_taa") == 0 || _stricmp(args, "st_opt_smaa") == 0 ||
                _stricmp(args, "st_opt_fsr2") == 0)
                want = (ps_r_dlss_quality <= 4) ? u32(ps_r_dlss_quality + 1) : u32(AA_DLAA);
            else
            {
                InvalidSyntax();
                return;
            }
        }

        Apply(want);
    }

    // ⭐Derived, never stored. The two variables underneath are also plain
    // console entries (r_dlss / r_dlss_quality are how the renderer was A/B'd
    // for months and they stay), so anything that writes them directly must not
    // be able to leave this row showing something else.
    void Status(TStatus& S) override
    {
        u32 v = u32(AA_OFF);
        if (ps_r_dlss)
            v = (ps_r_dlss_quality <= 4) ? u32(ps_r_dlss_quality + 1) : u32(AA_DLAA);
        *value = v;
        for (const xr_token* t = aa_mode_token; t->name; ++t)
            if ((u32)t->id == v)
            {
                xr_strcpy(S, t->name);
                return;
            }
        xr_strcpy(S, "st_opt_off");
    }

private:
    void Apply(u32 v)
    {
        *value = v;
        if (v == AA_OFF)
        {
            ps_r_dlss = 0;
        }
        else
        {
            ps_r_dlss = 1;
            ps_r_dlss_quality = v - 1; // AA_DLAA → 0 (native), … AA_DLSS_UP → 4
            // ⚠DLSS is fed by the motion-vector pass; without it Dlss::Upscaling()
            // stays false and the whole thing silently degrades to "nothing
            // happened". Turning the prerequisite on with the feature is the
            // difference between a setting and a riddle.
            if (!ps_r_motion_vectors)
            {
                ps_r_motion_vectors = 1;
                Msg("~ [VK AA] r_motion_vectors turned on — DLSS needs it");
            }
            if (ps_r_dlss_avail == 0) // 0 = probed and said no; -1 = not probed yet (startup)
                Msg("~ [VK AA] DLSS requested, but NGX reports it unavailable on this GPU/driver — the renderer will keep it off");
        }
    }
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

    // ⚠vk_shot lives OUTSIDE the DEBUG block, and it used to sit inside it next
    // to dump_resources -- the same trap fs_extract fell into. The command
    // exists so a scripted run can look at what it built, and the builds worth
    // looking at are RELEASE ones: that is where the mod's content is, and where
    // a UI port is checked. Registered in DEBUG only, it was missing from every
    // run that needed it and reported itself as "Unknown command".
    CMD1(CCC_VkShot, "vk_shot");
    CMD1(CCC_VkUiPassTrace, "ui_pass_trace");

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
    CMD4(CCC_Integer, "r_ssao_debug", &ps_r_ssao_debug, 0, 4);   // 1=AO map, 2=depth view, 3=normal view, 4=fully-open AO (no horizon → isolates normal/integral banding)
    CMD4(CCC_Float, "r_ssao_strength", &ps_r_ssao_strength, 0.f, 4.f);
    CMD4(CCC_Float, "r_ssao_temporal", &ps_r_ssao_temporal, 0.f, 0.97f);  // AO temporal accumulation α (0=off; jitter+MV-reprojected EMA; needs r_motion_vectors for moving cam)
    CMD4(CCC_Float, "r_ssao_bias", &ps_r_ssao_bias, 0.f, 0.9f);           // grazing-fade N·V threshold (fades AO→open at grazing → kills flat-floor bands at any strength; 0=off)
    CMD4(CCC_Float, "r_dither", &ps_r_dither, 0.f, 2.f);                  // final 8-bit output dither, LSBs (kills gradient contouring/"полосы" on flat surfaces; 0=off A/B)
    CMD4(CCC_Integer, "r_ssil", &ps_r_ssil_enable, 0, 1);        // SSIL (folded into GTAO) on/off (A/B; needs r_ssao on)
    CMD4(CCC_Integer, "r_ssil_debug", &ps_r_ssil_debug, 0, 1);   // 1 = show ONLY the indirect bounce field
    CMD4(CCC_Float, "r_ssil_strength", &ps_r_ssil_strength, 0.f, 8.f);  // IL intensity multiplier
    CMD4(CCC_Float, "r_ssil_temporal", &ps_r_ssil_temporal, 0.f, 0.97f);  // GTAO temporal accumulation α (0=off; lets r_ssil ship clean)
    CMD4(CCC_Integer, "r_motion_vectors", &ps_r_motion_vectors, 0, 1);  // screen-space MV pass on/off (DLSS/FSR/PT foundation)
    // Read-only in spirit: the NGX probe writes it, the options screen reads it.
    // NoSave — it describes THIS GPU and must never travel in a config.
    CMD4(CCC_IntegerNoSave, "r_dlss_avail", &ps_r_dlss_avail, -1, 1);
    // Internal resolution as a fraction of the display, for when DLSS is off or
    // absent. Floor 0.5 = quarter the pixels; below that the bilinear upscale
    // stops being "cheaper" and starts being "broken".
    CMD4(CCC_Float, "r_render_scale", &ps_r_render_scale, 0.5f, 1.0f);
    CMD4(CCC_Integer, "r_dlss", (int*)&ps_r_dlss, 0, 1);                 // DLSS Super Resolution; needs r_motion_vectors + NGX available
    CMD4(CCC_Integer, "r_dlss_quality", (int*)&ps_r_dlss_quality, 0, 4); // 0=DLAA 1=Quality 2=Balanced 3=Performance 4=UltraPerf (render<display upscale)
    CMD4(CCC_Integer, "r_dlss_preset", (int*)&ps_r_dlss_preset, 0, 15);  // DLSS model preset hint: 0=driver default, 6=F (CNN), 10=J, 11=K (transformer, default)
    CMD4(CCC_Float,   "r_dlss_sharp", &ps_r_dlss_sharp, 0.0f, 1.0f);     // CAS sharpen on the DLSS output (tonemap), 0 = off
    CMD4(CCC_Integer, "r_dlss_debug", &ps_r_dlss_debug, 0, 4);           // 1=CAS heatmap, 2=split no-DLSS|composite, 3=gate flag, 4=input sanitizer (NaN/neg/huge in the render-res scene)
    CMD4(CCC_Integer, "r_dlss_jitter_flip", &ps_r_dlss_jitter_flip, 0, 3); // reported-jitter sign A/B: bit0=X, bit1=Y
    CMD4(CCC_Integer, "r_dlss_exp", &ps_r_dlss_exp, 0, 1);               // 1=real exposure texture (default), 0=AutoExposure flag
    CMD4(CCC_Float,   "r_dlss_bias", &ps_r_dlss_bias, 0.0f, 1.0f);       // material mip-bias scale under upscaling (0=off A/B)
    CMD4(CCC_Integer, "r_dlss_sl", &ps_r_dlss_sl, 0, 1);                 // SR via Streamline sl.dlss (Stage B, FG prerequisite)
    CMD4(CCC_Integer, "r_dlss_sl_flip", &ps_r_dlss_sl_flip, 0, 3);       // SL jitter sign A/B (bit0=X, bit1=Y)
    CMD4(CCC_Integer, "r_dlss_sl_mv", &ps_r_dlss_sl_mv, 0, 1);           // SL MV direction A/B (1 = negate)
    CMD4(CCC_Integer, "r_dlss_fg", &ps_r_dlss_fg, 0, 1);                 // DLSS Frame Generation (Stage C, MFG); needs r_dlss + r_dlss_sl
    CMD4(CCC_Integer, "r_dlss_fg_mult", &ps_r_dlss_fg_mult, 2, 6);       // FG frame multiplier 2..6 (2x default, up to 6x MFG)
    CMD4(CCC_Integer, "r_dlss_fg_debug", &ps_r_dlss_fg_debug, 0, 2);     // DLSS-G + Reflex state log (1=periodic, 2=every frame)
    CMD4(CCC_Integer, "r_mv_debug", &ps_r_mv_debug, 0, 1);              // false-colour MV overlay (grey=still, R=+x, G=+y)
    CMD4(CCC_Integer, "r_mv_trees", &ps_r_mv_trees, 0, 1);              // tree wind-sway MV overlay (A/B under DLSS)
    CMD4(CCC_Integer, "r_mv_grass", &ps_r_mv_grass, 0, 1);              // grass wind-sway MV overlay (A/B under DLSS)
    CMD4(CCC_Float, "r_mv_debug_scale", &ps_r_mv_debug_scale, 1.f, 500.f); // MV overlay magnitude scale
    CMD4(CCC_Float, "r_wind_tree_bend", &ps_r_wind_tree_bend, 0.f, 2.f);   // tree trunk sway intensity (0 = rigid)
    CMD4(CCC_Float, "r_wind_tree_anim", &ps_r_wind_tree_anim, 0.f, 40.f);  // tree branch/leaf flutter speed
    CMD4(CCC_Float, "r_wind_tree_trunk", &ps_r_wind_tree_trunk, 0.f, 2.f); // tree trunk anim speed
    CMD4(CCC_Float, "r_wind_tree_flutter", &ps_r_wind_tree_flutter, 0.f, 16.f); // crown/leaf flutter amplitude
    CMD4(CCC_Float, "r_wind_tree_crown", &ps_r_wind_tree_crown, 0.f, 30.f);     // height where leaf flutter fades in
    CMD4(CCC_Float, "r_wind_shadow_dist", &ps_r_wind_shadow_dist, 0.f, 160.f);  // tree shadow wind radius (0 = all static)
    CMD4(CCC_Float, "r_tree_dist", &ps_r_tree_dist, 0.f, 5000.f);               // forward tree mesh range where an FLOD billboard takes over (0 = off, draws both)
    CMD4(CCC_Integer, "r_tree_hzb", &ps_r_tree_hzb, 0, 1);                      // Hi-Z occlusion cull for the forward tree set
    CMD4(CCC_Float, "r_tree_shade_dist", &ps_r_tree_shade_dist, 0.f, 5000.f);   // range past which tree.frag drops SSIL / sky sheen / dynamic lights / shore wetness (0 = off)
    CMD4(CCC_Integer, "r_vsm_tree_wind",      &ps_r_vsm_tree_wind,      0, 1);            // near/far hybrid: near trees sway in the dyn atlas
    CMD4(CCC_Integer, "r_vsm_meshlet",        &ps_r_vsm_meshlet,        0, 1);            // per-page meshlet cull of VSM tree casters (Phase B)
    CMD4(CCC_Integer, "r_vsm_hzb",            &ps_r_vsm_hzb,            0, 1);            // shadow-HZB occlusion cull of VSM casters (kills VSMrender overdraw)
    CMD4(CCC_Float,   "r_vsm_hzb_margin",     &ps_r_vsm_hzb_margin,     0.0f, 0.05f);     // depth slack for the shadow-HZB occluder (stale-sun safety)
    CMD4(CCC_Float,   "r_vsm_tree_wind_dist", &ps_r_vsm_tree_wind_dist, 8.0f, 200.0f);    // near set radius (m)
    CMD4(CCC_Integer, "r_vsm_tree_impostor",       &ps_r_vsm_tree_impostor,       0, 1);         // near crown mesh → baked sun-facing billboard in the dyn atlas
    CMD4(CCC_Float,   "r_vsm_tree_impostor_scale", &ps_r_vsm_tree_impostor_scale, 0.3f, 3.0f);   // billboard size vs crown sphere (shadow footprint)
    CMD4(CCC_Integer, "r_vsm_tree_hull",      &ps_r_vsm_tree_hull,      0, 2);                   // caster LOD: 1 = near trees beyond _dist cast the baked opaque crown hull; 2 = far/static foliage too (monotonic LOD)
    CMD4(CCC_Float,   "r_vsm_tree_hull_dist", &ps_r_vsm_tree_hull_dist, 4.0f, 200.0f);           // crown-mesh tier radius (m); the rest of the near set uses the hull
    CMD4(CCC_Integer, "r_vsm_tree_hull_vox",  &ps_r_vsm_tree_hull_vox,  0, 128);                 // crown voxelization: 0 = lobes only, >0 = level-0 voxel-cloud resolution (cells across the crown); baked → level reload
    CMD4(CCC_Integer, "r_vsm_tree_vox_cloud", &ps_r_vsm_tree_vox_cloud, 0, 1);                   // bake the visual cube cloud even with its viewmode off (the old unconditional bake); baked → level reload
    CMD4(CCC_Integer, "r_vsm_tree_hull_lod",  &ps_r_vsm_tree_hull_lod, -4, 8);                   // LIVE LOD bias for the voxel viewmode (- = finer, + = coarser); no rebake
    CMD4(CCC_Float,   "r_vsm_tree_hull_vox_px", &ps_r_vsm_tree_hull_vox_px, 1.0f, 32.0f);        // target projected voxel size in px (Nanite screen-error LOD cut); live
    CMD4(CCC_Float,   "r_vsm_tree_hull_vox_far", &ps_r_vsm_tree_hull_vox_far, 0.0f, 32.0f);      // +px per 100m: far cubes visibly chunkier (0 = constant screen size); live
    CMD4(CCC_Float,   "r_vsm_tree_hull_vox_smin", &ps_r_vsm_tree_hull_vox_smin, 0.0f, 2.0f);     // LEGACY no-op (cube era) — kept so persisted user.ltx lines parse
    CMD4(CCC_Float,   "r_vsm_tree_hull_vox_stex", &ps_r_vsm_tree_hull_vox_stex, 0.5f, 16.0f);    // SHADOW voxel LOD: page texels per voxel edge (UE shadow-view metric); live
    CMD4(CCC_Float,   "r_vsm_tree_hull_vox_sfloor", &ps_r_vsm_tree_hull_vox_sfloor, 0.0f, 2.0f); // SHADOW-only safety floor on voxel edge (m); live
    CMD4(CCC_Integer, "r_vsm_tree_hull_vox_cap",  &ps_r_vsm_tree_hull_vox_cap, 200, 20000);      // SHADOW-only max bricks per tree slice (perf); live
    CMD4(CCC_Integer, "r_vsm_tree_hull_vox_wpo",  &ps_r_vsm_tree_hull_vox_wpo, 0, 1);            // UE WPO-off: rigid voxel trees → static cache (needs hull 2); live
    CMD4(CCC_Integer, "r_vsm_tree_hull_vox_static", &ps_r_vsm_tree_hull_vox_static, 0, 1);       // static-tier proxy: 0=shell (cheap, per-page-friendly) 1=bricks; live
    CMD4(CCC_Integer, "r_vsm_tree_hull_vox_cull",  &ps_r_vsm_tree_hull_vox_cull, 0, 1);          // stage-2 per-page brick compaction (UE-parity cull); live
    CMD4(CCC_Float,   "r_vsm_tree_hull_band",   &ps_r_vsm_tree_hull_band, 0.02f, 0.9f);          // shadow crossfade band: crown↔voxels dither inside this fraction of _dist; live
    CMD4(CCC_Float,   "r_vsm_tree_hull_vox_fade", &ps_r_vsm_tree_hull_vox_fade, 0.0f, 0.6f);     // dithered voxel-LOD crossfade band, fraction of handover dist (0 = hard cut); live
    CMD4(CCC_Integer, "r_vsm_tree_hull_debug", &ps_r_vsm_tree_hull_debug, 0, 2);                 // 1 = shaded hull over hull-tier trees (_dist boundary); 2 = over EVERY hulled tree (walk-up voxel inspect)
    CMD4(CCC_Float, "r_sun_boost", &ps_r_sun_boost, 0.f, 4.f);
    CMD4(CCC_Integer, "r_linear_color", &ps_r_linear_color, 0, 1);                // linear colour pipeline; LATCHED per process (ColorSpace::Active) — takes effect on next game start
    CMD4(CCC_Integer, "r_sun_night_freeze", &ps_r_sun_night_freeze, 0, 1);        // freeze the sun-shadow (VSM) update when the sun is down (no light → no cost)
    CMD4(CCC_Float,   "r_sun_night_lum",    &ps_r_sun_night_lum,    0.f, 0.5f);   // sun_color luminance threshold for "night" (secondary)
    CMD4(CCC_Float,   "r_sun_night_alt",    &ps_r_sun_night_alt,   -0.2f, 0.5f);  // to-sun.y (sun altitude) below which it's "night" (primary; 0 = horizon)
    CMD4(CCC_Float, "r_grass_aref", &ps_r_grass_aref, 0.05f, 0.9f);
    CMD4(CCC_Float, "r_grass_asharp", &ps_r_grass_asharp, 0.0f, 2.0f);   // mip alpha compensation (far grass density)
    CMD4(CCC_Integer, "r_light_debug", &ps_r_light_debug, 0, 1);         // dump dynamic lights to the log (~2 s)
    CMD4(CCC_Integer, "r_light_cones",        &ps_r_light_cones,        0, 2);          // real volumetric beams for volumetric spots (2 = bounding-sphere debug)
    CMD4(CCC_Float,   "r_light_cone_density", &ps_r_light_cone_density, 0.0f, 5.0f);    // beam in-scatter strength
    CMD4(CCC_Float,   "r_light_cone_len",     &ps_r_light_cone_len,     0.5f, 10.0f);   // beam length = range × this
    CMD4(CCC_Float,   "r_light_cone_glare",   &ps_r_light_cone_glare,   0.0f, 8.0f);    // looking-into-the-beam flare
    CMD4(CCC_Float,   "r_light_cone_narrow",  &ps_r_light_cone_narrow,  0.1f, 1.0f);    // visible beam vs lit cone angle
    CMD4(CCC_Float,   "r_light_cone_lum",     &ps_r_light_cone_lum,     0.1f, 12.0f);   // beam luminance (HDR units; raise for day visibility)
    CMD4(CCC_Integer, "r_light_cone_synth",   &ps_r_light_cone_synth,   0, 1);          // beams synthesized from lightplanes model geometry
    CMD4(CCC_Float,   "r_light_cone_lift",    &ps_r_light_cone_lift,    0.0f, 0.5f);    // vertical apex lift for synthesized beams (m)
    CMD4(CCC_Float,   "r_light_cone_base",    &ps_r_light_cone_base,    0.0f, 4.0f);    // lamp-face radius multiplier (frustum base)
    CMD4(CCC_Float,   "r_light_cone_reach",   &ps_r_light_cone_reach,   1.0f, 10.0f);   // beam/light reach = fan length × this
    CMD4(CCC_Float,   "r_light_cone_power",   &ps_r_light_cone_power,   0.0f, 10.0f);   // synthesized-light surface intensity (0 = beam only)
    CMD4(CCC_Float,   "r_light_cone_fade",    &ps_r_light_cone_fade,    0.05f, 10.0f);  // synth lamp-face glow fade length (m)
    CMD4(CCC_Integer, "r_spot_grass",         &ps_r_spot_grass,         0, 1);          // grass casters into the spot shadow map (beam cutouts)
    CMD4(CCC_Integer, "r_sun_grass",          &ps_r_sun_grass,          0, 1);          // grass casters into the sun cascades (swaying blade shadows)
    CMD4(CCC_Float,   "r_sun_grass_dist",     &ps_r_sun_grass_dist,     5.f, 100.f);    // sun grass-shadow cull radius around the camera (m)
    CMD4(CCC_Integer, "r_point_grass",        &ps_r_point_grass,        0, 1);          // grass casters into the campfire point cubes (dapples)
    CMD4(CCC_Integer, "r_grass_cull",         &ps_r_grass_cull,         0, 1);          // GPU per-light grass cull+compaction (A/B: 0 = brute-force full VS pass)
    CMD4(CCC_Integer, "r_grass_cull_debug",   &ps_r_grass_cull_debug,   0, 1);          // [VK GrassCull] per-call lights/types/cells log
    CMD4(CCC_Float,   "r_light_cone_soft",    &ps_r_light_cone_soft,    0.0f, 32.0f);   // beam shadow penumbra radius (spot-map texels)
    CMD4(CCC_Integer, "r_light_cone_torch",   &ps_r_light_cone_torch,   0, 1);          // analytic cone for ordinary torches (0 = R4: froxel shaft only, no searchlight cone)
    CMD4(CCC_Float,   "r_flashlight_glow",    &ps_r_flashlight_glow,    0.0f, 20.0f);   // bright lamp-face glow for a torch aimed at the camera (0 = off)
    CMD4(CCC_Float,   "r_spot_grass_shadow",  &ps_r_spot_grass_shadow,  0.0f, 1.0f);    // grass shadow strength on surfaces (0 off, 1 full blanket)
    CMD4(CCC_Float,   "r_flashlight_grass",   &ps_r_flashlight_grass,   0.0f, 1.0f);    // grass shadow strength for FLASHLIGHT tiles only (night wow; lamps keep r_spot_grass_shadow)
    CMD4(CCC_Integer, "r_spot_pool",          &ps_r_spot_pool,          1, 8);          // spot shadow tiles per frame (each light = own map)
    CMD4(CCC_Integer, "r_point_pool",         &ps_r_point_pool,         1, 4);          // point shadow cubes per frame (each campfire = own cube)
    CMD4(CCC_Integer, "r_point_debug",        &ps_r_point_debug,        0, 1);          // debug overlay: green=lit by pooled point, red=its shadow
    CMD4(CCC_Integer, "r_grass_debug",        &ps_r_grass_debug,        0, 11);         // grass component isolation (1..11, see decl)
    CMD4(CCC_Float,   "r_grass_self_bias",    &ps_r_grass_self_bias,    0.0f, 1.0f);    // grass-on-grass anti-acne slack (m along sun ray)
    CMD4(CCC_Float,   "r_sun_beam",           &ps_r_sun_beam,           0.0f, 1.0f);    // ground beam-gap recovery strength (0 = off)
    CMD4(CCC_Float,   "r_sun_beam_dist",      &ps_r_sun_beam_dist,      5.0f, 120.0f);  // beam-recovery max distance (m)
    CMD4(CCC_Float,   "r_sun_beam_boost",     &ps_r_sun_beam_boost,     1.0f, 4.0f);    // extra sun in the recovered gap (cinematic splash)
    CMD4(CCC_Float,   "r_sun_beam_bias",      &ps_r_sun_beam_bias,      0.0f, 1.0f);    // beam-recovery atlas self-bias (m along sun ray)
    CMD4(CCC_Float,   "r_sun_beam_splash",     &ps_r_sun_beam_splash,     0.0f, 8.0f);   // god-ray ground-splash strength (0 = off)
    CMD4(CCC_Float,   "r_sun_beam_splash_thr", &ps_r_sun_beam_splash_thr, 0.0f, 2.0f);   // in-scatter luminance threshold before the splash kicks in
    CMD4(CCC_Float,   "r_sun_beam_ground",     &ps_r_sun_beam_ground,     0.0f, 4.0f);   // forward sun-beam GROUND deposit strength (0 = off, A/B)
    CMD4(CCC_Float,   "r_sun_beam_ground_thr", &ps_r_sun_beam_ground_thr, 0.0f, 2.0f);   // in-scatter luminance threshold for the ground deposit
    CMD4(CCC_Float,   "r_point_boost",        &ps_r_point_boost,        0.1f, 8.0f);    // campfire/brazier glow intensity multiplier (volumetric points)
    CMD4(CCC_Float,   "r_point_range",        &ps_r_point_range,    0.5f, 4.0f);    // campfire/brazier light reach multiplier (volumetric points)
    CMD4(CCC_Float,   "r_glass_opacity", &ps_r_glass_opacity, 0.05f, 1.0f); // glass opacity ceiling (1 = texture alpha as in R4)
    CMD4(CCC_Float,   "r_glass_refr",    &ps_r_glass_refr,    0.0f,  3.0f); // glass refraction wobble strength (0 = off)
    CMD4(CCC_Integer, "r_preskin",       &ps_r_preskin,       0, 1);        // compute pre-skinning: skin once per frame, not once per pass
    CMD4(CCC_Integer, "r_clouds",           &ps_r_clouds,           0, 1);         // animated cloud layer on/off
    CMD4(CCC_Float,   "r_clouds_intensity", &ps_r_clouds_intensity, 0.0f, 4.0f);   // cloud additive brightness
    CMD4(CCC_Float,   "r_clouds_speed",     &ps_r_clouds_speed,     0.0f, 8.0f);   // cloud UV scroll speed
    CMD4(CCC_Integer, "r_grass_nowave", &ps_r_grass_nowave, 0, 1);
    CMD4(CCC_Float, "r_ambient_floor", &ps_r_ambient_floor, 0.f, 0.5f);
    CMD4(CCC_Float, "r_ambient_sky_gate", &ps_r_ambient_sky_gate, 0.f, 1.f);   // gate flat sky ambient by sky visibility (0=old leak, 1=dark interiors)
    CMD4(CCC_Float, "r_wet_darken", &ps_r_wet_darken, 0.f, 1.f);
    CMD4(CCC_Float, "r_wet_refl", &ps_r_wet_refl, 0.f, 3.f);
    CMD4(CCC_Integer, "r_wet_debug", &ps_r_wet_debug, 0, 1);
    CMD4(CCC_Integer, "r_rain_debug", &ps_r_rain_debug, 0, 1);
    CMD4(CCC_Integer, "r_rain", &ps_r_rain_enable, 0, 1);   // master rain on/off (effect only, not weather)
    CMD4(CCC_Float, "r_rain_sun", &ps_r_rain_sun, 0.f, 3.f); // streaks catch sun/moon/lightning (0 = flat hemi)

    // GPU-driven particles (gpu_particles_roadmap.md Phase 1). r_gpu_particles 1
    // runs the GPU-resident test effect (emit/simulate/draw) at the camera.
    CMD4(CCC_Integer, "r_gpu_particles", &ps_r_gpu_particles, 0, 1);
    CMD4(CCC_Integer, "r_gpu_particles_max", &ps_r_gpu_particles_max, 1024, 1 << 24);
    CMD4(CCC_Integer, "r_gpu_particles_sort", &ps_r_gpu_particles_sort, 0, 1);
    CMD4(CCC_Float,   "r_gpu_particles_life_cap", &ps_r_gpu_particles_life_cap, 0.0f, 300.0f);
    CMD1(CCC_GP_Mirror, "gp_mirror");       // Phase 3: mirror a real .pe onto the camera emitter
    CMD1(CCC_GP_Spawn,  "gp_spawn");        // Phase 3.2: place a persistent world emitter
    CMD1(CCC_GP_SpawnClear, "gp_spawn_clear"); // Phase 3.2: remove world emitters
    CMD1(CCC_GP_List,   "gp_list");         // Phase 3: list loaded .pe names (optional substr filter)
    CMD1(CCC_GP_Debug,  "gp_debug");        // dump a .pe def + its GPU translation to the log
    CMD1(CCC_GP_Stats,  "gp_stats");        // per-program budget/alive counters

    // Global render profiler (vk_profiler): r_profiler 0/1/2, vk_perf = MARK dump.
    CMD4(CCC_Integer, "r_profiler", &ps_r_profiler, 0, 2);
    CMD1(CCC_VkPerf,  "vk_perf");

    // Variable Rate Shading (vk_vrs): 0 off / 1 mild / 2 aggressive + distance thresholds.
    CMD4(CCC_Integer, "r_vrs", &ps_r_vrs, 0, 2);
    CMD4(CCC_Integer, "r_vrs_force", &ps_r_vrs_force, 0, 4);
    CMD4(CCC_Integer, "r_vrs_static", &ps_r_vrs_static, 0, 1);
    CMD4(CCC_Integer, "r_at_equal", &ps_r_at_equal, 0, 1);   // AT statics: depth EQUAL color path (live A/B)
    CMD4(CCC_Integer, "r_fsinv_split", &ps_r_fsinv_split, 0, 1);   // diag: 4-way FS-invocation attribution in World/Color
    CMD4(CCC_Integer, "r_z_prepass", &ps_r_z_prepass, 0, 1);       // statics color: no z-write -> early-Z with discard (live A/B)
    CMD4(CCC_Float,   "r_ssa_px", &ps_r_ssa_px, 0.f, 16.f);        // SSA cull of plain meshes, projected-diameter px (0 = off, live)
    CMD4(CCC_Integer, "r_uber_variants", &ps_r_uber_variants, 0, 1);   // A/B: 0 = old monolithic world uber-FS
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
    CMD4(CCC_Float,   "r_vsm_bias_dyn", &ps_r_vsm_bias_dyn, 0.0f, 0.02f);
    CMD4(CCC_Float,   "r_vsm_bias_min", &ps_r_vsm_bias_min, 0.0f, 0.02f);   // static receiver bias const part (0 = legacy flat r_vsm_bias)
    CMD4(CCC_Float,   "r_vsm_raster_bias",  &ps_r_vsm_raster_bias,  0.0f, 16.0f);  // atlas write bias const (D16 units)
    CMD4(CCC_Float,   "r_vsm_raster_slope", &ps_r_vsm_raster_slope, 0.0f, 16.0f);  // atlas write bias slope
    CMD4(CCC_Float,   "r_vol_vsm_bias", &ps_r_vol_vsm_bias, 0.0f, 0.02f);   // fog air bias vs static atlas (0 = legacy)
    CMD4(CCC_Float,   "r_vol_depth_reject", &ps_r_vol_depth_reject, 0.0f, 5.0f);   // froxel depth-rejection slack m (0 = off)
    CMD4(CCC_Float,   "r_vol_surf_clip",    &ps_r_vol_surf_clip,    0.0f, 2.0f);   // depth-rejection front shell m (0 = off)
    CMD4(CCC_Integer, "r_vsm_grass_static", &ps_r_vsm_grass_static, 0, 1);   // far-grass static-cache hybrid (0 = all-dynamic L0..L2)
    CMD4(CCC_Integer, "r_vsm_temporal", &ps_r_vsm_temporal, 0, 1);
    CMD4(CCC_Float,   "r_vsm_ta_blend", &ps_r_vsm_ta_blend, 0.0f, 0.98f);
    CMD4(CCC_Float,   "r_vsm_ta_blend_dyn", &ps_r_vsm_ta_blend_dyn, 0.0f, 0.98f);
    CMD4(CCC_Float,   "r_vsm_ta_clamp",        &ps_r_vsm_ta_clamp,        0.0f, 1.0f);   // (1) history clamp tol (0 = off)
    CMD4(CCC_Float,   "r_vsm_ta_motion",       &ps_r_vsm_ta_motion,       0.5f, 64.0f);  // (2) px of reproj motion to reach the floor
    CMD4(CCC_Float,   "r_vsm_ta_motion_floor", &ps_r_vsm_ta_motion_floor, 0.0f, 0.98f);  // (2) history weight at full motion
    CMD4(CCC_Float,   "r_vsm_ta_blend_dlss",   &ps_r_vsm_ta_blend_dlss,   0.0f, 1.0f);   // (3) history-weight scale when r_dlss on
    CMD4(CCC_Float,   "r_vsm_ta_carry",        &ps_r_vsm_ta_carry,        0.0f, 1.0f);   // (4) history carry on non-resident pages (0 = old flash-lit)
    CMD4(CCC_Integer, "r_vsm_soft",        &ps_r_vsm_soft,        0, 32);        // stochastic PCSS filter taps (0 = legacy 3x3 PCF)
    CMD4(CCC_Integer, "r_vsm_soft_search", &ps_r_vsm_soft_search, 2, 16);        // blocker-search taps
    CMD4(CCC_Float,   "r_vsm_soft_angle",  &ps_r_vsm_soft_angle,  0.1f, 8.0f);   // sun cone HALF-angle, degrees (0.265 = physical)
    CMD4(CCC_Float,   "r_vsm_soft_range",  &ps_r_vsm_soft_range,  2.0f, 80.0f);  // max blocker distance m = widest penumbra
    CMD4(CCC_Float,   "r_vsm_soft_clamp",  &ps_r_vsm_soft_clamp,  0.0f, 1.0f);   // temporal clamp while soft is on (noise must pass)
    CMD4(CCC_Integer, "r_vsm_grass",      &ps_r_vsm_grass,      0, 1);
    CMD4(CCC_Float,   "r_vsm_grass_dist", &ps_r_vsm_grass_dist, 4.0f, 48.0f);
    CMD4(CCC_Integer, "r_vsm_tree_vrs",   &ps_r_vsm_tree_vrs,   0, 1);   // 2x2 coarse crown-shadow shading (A/B; parked — no gain)
    CMD4(CCC_Integer, "r_vsm_cadence",    &ps_r_vsm_cadence,    0, 4);   // DYN pass update every Nth frame (0/1 = off; 2-3 ≈ 30-45Hz); static+sun always live
    CMD4(CCC_Float,   "r_vsm_cadence_still", &ps_r_vsm_cadence_still, 0.0f, 1.0f);   // dyn-skip only when cam moved < this (m/frame); moving = full-rate (no flicker)
    CMD4(CCC_Float,   "r_vsm_cadence_sun",   &ps_r_vsm_cadence_sun,   0.0f, 45.0f);  // force dyn update when sun rotated > this (deg) since last update (env-step flicker fix)
    CMD4(CCC_Float,   "r_vsm_npc_dist",   &ps_r_vsm_npc_dist,   0.0f, 500.0f);
    CMD4(CCC_Float,   "r_vsm_lod_dist",   &ps_r_vsm_lod_dist,   0.0f, 500.0f);
    CMD4(CCC_Integer, "r_vsm_mark_half",  &ps_r_vsm_mark_half,  0, 1);
    CMD4(CCC_Integer, "r_vsm_dyn_gate",   &ps_r_vsm_dyn_gate,   0, 1);
    CMD4(CCC_Integer, "r_vsm_debug_dyn",  &ps_r_vsm_debug_dyn,  0, 4);   // red overlay: dyn-atlas (NPC/grass) shadows; 2 = RAW occlusion (no static/orientation gate); 3 = PRESENCE (any dyn depth, no z-test); 4 = SUN VISIBILITY (red = shadow system says sun-lit)
    CMD4(CCC_Integer, "r_vsm_cache",      &ps_r_vsm_cache,      0, 1);
    CMD4(CCC_Integer, "r_vsm_cache_refresh", &ps_r_vsm_cache_refresh, 1, 64);
    CMD4(CCC_Integer, "r_vsm_throttle",        &ps_r_vsm_throttle, 0, 1);                 // cost-feedback LOD bias (A/B)
    CMD4(CCC_Float,   "r_vsm_throttle_budget", &ps_r_vsm_throttle_budget, 0.5f, 20.0f);   // target VSMrender ms
    CMD4(CCC_Float,   "r_vsm_throttle_max",    &ps_r_vsm_throttle_max, 0.0f, 4.0f);       // max bias (clipmap levels)
    CMD4(CCC_Integer, "r_vsm_dirty_budget",    &ps_r_vsm_dirty_budget, 0, 4096);          // wrong-tile pages/frame, 0 = off (A/B)
    CMD4(CCC_Integer, "r_vsm_load_freeze",     &ps_r_vsm_load_freeze,  0, 1);             // freeze the sun-shadow update behind the load screen (A/B)
    CMD4(CCC_Integer, "r_vsm_rmask",           &ps_r_vsm_rmask, 0, 1);                    // receiver-mask sub-page cull (A/B)
    CMD4(CCC_Integer, "r_vsm_gaze",            &ps_r_vsm_gaze, 0, 1);                     // gaze refresh: looked-at pages re-render ∝ footprint (A/B)
    CMD4(CCC_Integer, "r_vsm_gaze_pages",      &ps_r_vsm_gaze_pages, 0, 1024);            // gaze budget, pages/frame
    CMD4(CCC_Integer, "r_vsm_gaze_px",         &ps_r_vsm_gaze_px, 64, 65536);             // mark samples for every-frame tier

    // GPU-driven world forward pass (vk_world_gpu) — A/B with r_gpu_world 0.
    CMD4(CCC_Integer, "r_gpu_world", &ps_r_gpu_world, 0, 1);
    // Cluster-granularity world cull (Phase 1 of the cluster-LOD system) —
    // takes effect on level (re)load. r_cluster_tris = min mesh size to split.
    CMD4(CCC_Integer, "r_cluster",      &ps_r_cluster,      0, 1);
    CMD4(CCC_Integer, "r_cluster_tris", &ps_r_cluster_tris, 128, 65536);
    CMD4(CCC_Integer, "r_cluster_cache", &ps_r_cluster_cache, 0, 1);   // 0 = always rebake the DAG (repro aid)

    // Cluster DAG LOD (Phase 2): px error threshold (live) + debug view (live).
    CMD4(CCC_Float,   "r_cluster_lod",   &ps_r_cluster_lod,   0.05f, 64.0f);
    CMD4(CCC_Integer, "r_cluster_debug", &ps_r_cluster_debug, 0, 4);   // 3 = path view (red plain / green per-mesh / blue component), 4 = LOD health (red = can never coarsen)
    CMD4(CCC_Float,   "r_cluster_fade",  &ps_r_cluster_fade,  0.0f, 1.0f);
    // Phase 2.5: merge touching solid fragments into one DAG (level reload).
    CMD4(CCC_Integer, "r_cluster_merge", &ps_r_cluster_merge, 0, 1);
    // Phase 3: cluster-LOD shadow casters (live gates) + LOD budget in target
    // texels (live; VSM invalidates its static cache on change).
    CMD4(CCC_Integer, "r_shadow_cluster",  &ps_r_shadow_cluster,  0, 1);
    // Alpha-tested casters via the cluster shadow indirect path (live A/B).
    CMD4(CCC_Integer, "r_gpu_shadows_at",  &ps_r_gpu_shadows_at,  0, 1);
    // Alpha-tested statics into the VSM static atlas (live A/B; cache reset on flip).
    CMD4(CCC_Integer, "r_vsm_at",          &ps_r_vsm_at,          0, 1);
    CMD4(CCC_Integer, "r_vsm_cluster",     &ps_r_vsm_cluster,     0, 1);
    CMD4(CCC_Float,   "r_vsm_cluster_lod", &ps_r_vsm_cluster_lod, 0.1f, 16.0f);

    // Clustered forward / Forward+ (vk_clustered) — A/B with r_clustered 0.
    // r_clustered_debug 1 = per-cluster light-count heatmap on the world.
    CMD4(CCC_Integer, "r_clustered",       &ps_r_clustered,       0, 1);
    CMD4(CCC_Integer, "r_clustered_debug", &ps_r_clustered_debug, 0, 1);
    // Framegraph: coalesce the SSAO/VRS/VSM depth-read round-trips into one (A/B with 0).
    CMD4(CCC_Integer, "r_fg_coalesce",     &ps_r_fg_coalesce,     0, 1);
    // Async compute: run compute on a dedicated queue overlapping graphics (vk_async).
    CMD4(CCC_Integer, "r_async",           &ps_r_async,           0, 1);
    // Sky specular IBL (vk_ibl): prefiltered sky reflections + sun GGX glint.
    CMD4(CCC_Integer, "r_ibl",             &ps_r_ibl,             0, 1);
    CMD4(CCC_Float,   "r_ibl_spec",        &ps_r_ibl_spec,        0.f, 4.f);
    CMD4(CCC_Integer, "r_ibl_debug",       &ps_r_ibl_debug,       0, 1);
    // Diffuse sky irradiance via SH9 (0 = A/B against the prefiltered probe mip).
    CMD4(CCC_Integer, "r_sky_sh",          &ps_r_sky_sh,          0, 1);
    CMD4(CCC_Float,   "r_sky_sh_ground",   &ps_r_sky_sh_ground,   0.f, 1.f);
    CMD4(CCC_Integer, "r_sky_sh_debug",    &ps_r_sky_sh_debug,    0, 1);
    // Procedural Rayleigh+Mie sky: feeds the dome AND the light probe from one model.
    CMD4(CCC_Integer, "r_sky_proc",        &ps_r_sky_proc,        0, 1);
    CMD4(CCC_Float,   "r_sky_intensity",   &ps_r_sky_intensity,   0.f, 200.f);
    CMD4(CCC_Float,   "r_sky_turbidity",   &ps_r_sky_turbidity,   0.f, 20.f);
    CMD4(CCC_Float,   "r_sky_mie_g",       &ps_r_sky_mie_g,       0.f, 0.99f);
    CMD4(CCC_Integer, "r_sky_sun_from_atmo", &ps_r_sky_sun_from_atmo, 0, 1);
    CMD4(CCC_Float,   "r_sky_sun_scale",   &ps_r_sky_sun_scale,   0.f, 40.f);
    // Volumetric clouds (raymarched). Needs r_sky_proc for its lighting to make sense.
    CMD4(CCC_Integer, "r_clouds_vol",           &ps_r_clouds_vol,           0, 1);
    CMD4(CCC_Float,   "r_clouds_coverage",      &ps_r_clouds_coverage,      0.f, 1.f);
    CMD4(CCC_Float,   "r_clouds_density",       &ps_r_clouds_density,       0.f, 8.f);
    CMD4(CCC_Float,   "r_clouds_detail",        &ps_r_clouds_detail,        0.f, 1.f);
    CMD4(CCC_Float,   "r_clouds_bottom",        &ps_r_clouds_bottom,        200.f, 12000.f);
    CMD4(CCC_Float,   "r_clouds_top",           &ps_r_clouds_top,           400.f, 20000.f);
    CMD4(CCC_Float,   "r_clouds_shape_scale",   &ps_r_clouds_shape_scale,   0.000001f, 0.01f);
    CMD4(CCC_Float,   "r_clouds_detail_scale",  &ps_r_clouds_detail_scale,  0.000001f, 0.05f);
    CMD4(CCC_Float,   "r_clouds_weather_scale", &ps_r_clouds_weather_scale, 0.0000001f, 0.01f);
    CMD4(CCC_Float,   "r_clouds_wind_dir",      &ps_r_clouds_wind_dir,      0.f, 360.f);
    CMD4(CCC_Float,   "r_clouds_wind_speed",    &ps_r_clouds_wind_speed,    0.f, 200.f);
    CMD4(CCC_Float,   "r_clouds_phase_g",       &ps_r_clouds_phase_g,       0.f, 0.99f);
    CMD4(CCC_Float,   "r_clouds_phase_g_back",  &ps_r_clouds_phase_g_back,  0.f, 0.99f);
    CMD4(CCC_Float,   "r_clouds_extinction",    &ps_r_clouds_extinction,    0.001f, 2.f);
    CMD4(CCC_Float,   "r_clouds_powder",        &ps_r_clouds_powder,        0.f, 1.f);
    CMD4(CCC_Float,   "r_clouds_sun",           &ps_r_clouds_sun,           0.f, 200.f);
    CMD4(CCC_Float,   "r_clouds_ambient",       &ps_r_clouds_ambient,       0.f, 20.f);
    CMD4(CCC_Integer, "r_clouds_steps",         &ps_r_clouds_steps,         16, 256);
    CMD4(CCC_Float,   "r_clouds_max_dist",      &ps_r_clouds_max_dist,      5000.f, 400000.f);
    CMD4(CCC_Float,   "r_clouds_cirrus",        &ps_r_clouds_cirrus,        0.f, 2.f);
    CMD4(CCC_Float,   "r_clouds_cirrus_alt",    &ps_r_clouds_cirrus_alt,    2000.f, 20000.f);
    CMD4(CCC_Float,   "r_clouds_cirrus_scale",  &ps_r_clouds_cirrus_scale,  0.0000001f, 0.01f);
    CMD4(CCC_Integer, "r_clouds_debug",         &ps_r_clouds_debug,         0, 2);
    CMD4(CCC_Integer, "r_clouds_weather",       &ps_r_clouds_weather,       0, 1);

    // Hi-Z occlusion cull of the GPU-driven static color pass (vk_world_gpu).
    CMD4(CCC_Integer, "r_hzb_cull", &ps_r_hzb_cull, 0, 1);

    // GPU-driven LOD imposters (vk_LODManager): 0 = CPU walk (A/B).
    CMD4(CCC_Integer, "r_lods_gpu", &ps_r_lods_gpu, 0, 1);

    // Froxel volumetric lighting (vk_volumetrics) — P1: sun god rays + depth fog.
    CMD4(CCC_Integer, "r_vol",           &ps_r_vol,           0, 1);
    CMD4(CCC_Float,   "r_vol_density",   &ps_r_vol_density,   0.0f, 1.0f);
    CMD4(CCC_Float,   "r_vol_height",    &ps_r_vol_height,    0.0f, 2.0f);
    CMD4(CCC_Float,   "r_vol_g",         &ps_r_vol_g,         0.0f, 0.95f);
    CMD4(CCC_Float,   "r_vol_intensity", &ps_r_vol_intensity, 0.0f, 8.0f);
    // Atmospheric scattering (r_atmo) — Rayleigh/Mie aerial perspective (needs r_vol).
    CMD4(CCC_Integer, "r_atmo",          &ps_r_atmo,          0, 1);
    CMD4(CCC_Float,   "r_atmo_rayleigh", &ps_r_atmo_rayleigh, 0.0f, 8.0f);
    CMD4(CCC_Float,   "r_atmo_mie",      &ps_r_atmo_mie,      0.0f, 8.0f);
    CMD4(CCC_Float,   "r_atmo_mie_g",    &ps_r_atmo_mie_g,    0.0f, 0.95f);
    // Auto-exposure temporal adaptation (eye adaptation) — seconds; 0 = instant.
    CMD4(CCC_Float,   "r_exp_adapt",     &ps_r_exp_adapt,     0.0f, 5.0f);
    // Auto-exposure user knobs (all live; see vk_exposure.h).
    CMD4(CCC_Float,   "r_expo",          &ps_r_expo,          0.1f, 4.0f);
    CMD4(CCC_Float,   "r_expo_gray",     &ps_r_expo_gray,     0.0f, 1.0f);
    CMD4(CCC_Float,   "r_expo_min",      &ps_r_expo_min,      0.1f, 2.0f);
    CMD4(CCC_Float,   "r_expo_max",      &ps_r_expo_max,      0.5f, 8.0f);
    CMD4(CCC_Float,   "r_vol_amb",       &ps_r_vol_amb,       0.0f, 1.0f);
    CMD4(CCC_Float,   "r_vol_ambient",   &ps_r_vol_ambient,   0.0f, 4.0f);   // fog sky-fill tint scale (lower = darker night air; was tied to r_ambient_floor)
    CMD4(CCC_Float,   "r_vol_indoor",    &ps_r_vol_indoor,    0.0f, 16.0f);
    CMD4(CCC_Float,   "r_vol_sun",       &ps_r_vol_sun,       0.0f, 16.0f);
    CMD4(CCC_Float,   "r_vol_lights",    &ps_r_vol_lights,    0.0f, 16.0f);
    CMD4(CCC_Float,   "r_vol_lights_g",  &ps_r_vol_lights_g,  0.0f, 0.95f);   // local-light forward scatter (torch-into-camera glare; sep. from sun r_vol_g)
    CMD4(CCC_Float,   "r_torch_vol",     &ps_r_torch_vol,     0.0f, 8.0f);    // fog shaft strength for handheld torches only (fixtures keep the ×3 lamp boost)
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
    CMD4(CCC_Integer, "r_vol_upsample",  &ps_r_vol_upsample,  0, 1);   // volume reconstruction filter in the composite (0 = old 1-tap)
    CMD4(CCC_Float,   "r_fog_dist",      &ps_r_fog_dist,      0.0f, 8.0f);   // forward distance-fog range scale (0 = off, 1 = weather)
    CMD4(CCC_Integer, "r_vol_ground_debug", &ps_r_vol_ground_debug, 0, 3);   // 1 = height above baked terrain, 2 = micro-relief in the rain map, 3 = baked grass canopy (view with r_vol_debug 1)
    CMD4(CCC_Integer, "r_vol_hillaire",      &ps_r_vol_hillaire,      0, 1);        // energy-conserving slice integration (0 = legacy A/B)
    CMD4(CCC_Float,   "r_vol_albedo",        &ps_r_vol_albedo,        0.0f, 1.0f);  // single-scatter albedo (sigma_s = sigma_t x albedo)
    CMD4(CCC_Integer, "r_vol_noise_scatter", &ps_r_vol_noise_scatter, 0, 1);        // animated noise on scattering, not extinction (0 = legacy)
    CMD4(CCC_Integer, "r_vol_ms",            &ps_r_vol_ms,            1, 4);        // multiple-scattering octaves (1 = single scatter)
    CMD4(CCC_Float,   "r_vol_g_back",        &ps_r_vol_g_back,       -0.9f, 0.0f);  // backward phase lobe g
    CMD4(CCC_Float,   "r_vol_g_mix",         &ps_r_vol_g_mix,         0.0f, 1.0f);  // backward lobe weight (0 = single lobe)
    CMD4(CCC_Float,   "r_vol_mist",          &ps_r_vol_mist,          0.0f, 2.0f);   // V-1 layer 2: ground-mist density at ground level (0 = single layer)
    CMD4(CCC_Float,   "r_vol_mist_h",        &ps_r_vol_mist_h,        0.25f, 64.0f); // ground-mist thickness (m)
    CMD4(CCC_Float,   "r_vol_mist_g",        &ps_r_vol_mist_g,        0.0f, 0.95f);  // ground-mist phase (droplets scatter less directionally than dust)
    CMD4(CCC_Float,   "r_vol_mist_relief",   &ps_r_vol_mist_relief,   0.0f, 40.0f);  // terrain drop (m) the mist fades in over (0 = layer is everywhere)
    CMD4(CCC_Float,   "r_vol_mist_noise",       &ps_r_vol_mist_noise,       0.0f, 2.0f);    // mist height-warp: billowing top instead of a flat lid (0 = smooth exponential)
    CMD4(CCC_Float,   "r_vol_mist_noise_scale", &ps_r_vol_mist_noise_scale, 0.005f, 0.5f);  // size of the billows (1/m; 0.05 = ~20 m)
    CMD4(CCC_Float,   "r_vol_mist_micro",    &ps_r_vol_mist_micro,    0.0f, 4.0f);   // ABSOLUTE density of mist pooling in ruts/craters, independent of r_vol_mist (0 = off)
    CMD4(CCC_Float,   "r_vol_mist_micro_h",  &ps_r_vol_mist_micro_h,  0.02f, 2.0f);  // dip depth (m) at which the micro effect saturates
    CMD4(CCC_Integer, "r_vol_mist_fade",     &ps_r_vol_mist_fade,     0, 3);         // thin the bank you are STANDING IN: 0 off, 1 immersion, 2 near-camera, 3 both
    CMD4(CCC_Float,   "r_vol_mist_inside",   &ps_r_vol_mist_inside,   0.0f, 64.0f);  // what it thins down to, as a MULTIPLE OF r_vol_density (2 = twice the general haze)
    CMD4(CCC_Float,   "r_vol_mist_near",     &ps_r_vol_mist_near,     1.0f, 200.0f); // radius (m) of the mode-2 clearing around the camera
    CMD4(CCC_Float,   "r_vol_mist_inside_h", &ps_r_vol_mist_inside_h, 1.0f, 200.0f); // height (m) the immersion fades out over — a hilltop, NOT a rooftop
    CMD4(CCC_Float,   "r_vol_canopy",        &ps_r_vol_canopy,        0.0f, 8.0f);   // grass canopy occlusion of the fog's sun term (0 = off, grass invisible to the sun as before)
    CMD4(CCC_Float,   "r_vol_canopy_h",      &ps_r_vol_canopy_h,      0.1f, 4.0f);   // canopy height multiplier over the authored detail-model height
    CMD4(CCC_Integer, "r_vol_weather",       &ps_r_vol_weather,       0, 2);         // V-2: weather drives the medium (0 = manual cvars only, 1 = on, 2 = on + log the inputs)
    CMD4(CCC_Float,   "r_vol_w_clouds",      &ps_r_vol_w_clouds,      0.0f, 1.0f);   // overcast damping of the fog's sun term (1 = full overcast kills it)
    CMD4(CCC_Float,   "r_vol_w_sky",         &ps_r_vol_w_sky,         0.0f, 4.0f);   // overcast lift of the sky term
    CMD4(CCC_Float,   "r_vol_w_flat",        &ps_r_vol_w_flat,        0.0f, 1.0f);   // overcast flattens the phase toward isotropic (the knob that actually READS as overcast)
    CMD4(CCC_Float,   "r_vol_w_fog",         &ps_r_vol_w_fog,         0.0f, 2.0f);   // extra MIST from fog_density (0 = off; fog_density means sight distance, not ground fog)
    CMD4(CCC_Float,   "r_vol_w_fog_dust",    &ps_r_vol_w_fog_dust,    0.0f, 8.0f);   // dust density gain at fog_density 1 (small: dust makes rays, not fog)
    CMD4(CCC_Float,   "r_vol_w_rain",        &ps_r_vol_w_rain,        0.0f, 2.0f);   // mist density added at full wetness
    CMD4(CCC_Float,   "r_vol_w_wet_force",   &ps_r_vol_w_wet_force,  -1.0f, 1.0f);   // test hook: -1 = real weather, 0..1 = pretend this wetness (see the rain fog without waiting for rain)
    CMD4(CCC_Integer, "r_vol_term",          &ps_r_vol_term,          0, 5);        // isolate one in-scatter source (1 sun 2 ambient 3 lights 4 atmo 5 smoke)
    CMD4(CCC_Float,   "r_vol_ta_blend",  &ps_r_vol_ta_blend,  0.0f, 0.98f);
    CMD4(CCC_Integer, "r_vol_debug",     &ps_r_vol_debug,     0, 4);   // 1 = raw in-scatter, 3 = froxel sun visibility, 4 = occlusion source (green VSM / red miss->skyVis / blue cascade)

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
    CMD4(CCC_Integer, "r_pom_debug", &ps_r_pom_debug, 0, 2);    // 1 = AO x shadow, 2 = self-shadow only
    CMD4(CCC_Integer, "r_ao_flat", &ps_r_ao_flat, 0, 1);   // debug: kill ALL ambient occlusion (incl. baked lmap)
    CMD4(CCC_Integer, "r_shade_debug", &ps_r_shade_debug, 0, 10);   // debug: isolate lighting components (see ps_r_shade_debug)
    CMD4(CCC_Float, "r_pom_ceil", &ps_r_pom_ceil, 0.f, 1.f);     // POM strength on ceilings (down-facing)
    CMD4(CCC_Float, "r_pom_floor", &ps_r_pom_floor, 0.f, 1.f);   // POM strength on floors (up-facing)
    CMD4(CCC_Integer, "r_pom_terrain", &ps_r_pom_terrain, 0, 1); // terrain POM (experimental, default off)
    CMD4(CCC_Float,   "r_pom_zoff", &ps_r_pom_zoff, 0.f, 3.f);   // terrain POM depth offset (SSFX; 1 = 0.11 m sink)
    CMD4(CCC_Integer, "r_terra_cache", &ps_r_terra_cache, 0, 1); // terrain composite cache (1 march vs 4; perf)
    CMD4(CCC_Integer, "r_terra_cone", &ps_r_terra_cone, 0, 1);   // cone-step march on the cache (vs fixed layers)
    CMD4(CCC_Float, "r_terra_blend", &ps_r_terra_blend, 0.f, 1.f); // detail blend: 0 = GAMMA soft mask, 0.25 = sharp height-blend
    CMD4(CCC_Integer, "r_terra_horizon", &ps_r_terra_horizon, 0, 1); // baked sun-horizon self-shadow (vs 8-tap march)
    CMD4(CCC_Float, "r_terrain_normal", &ps_r_terrain_normal, 0.f, 3.f); // terrain detail normal-mapping strength
    CMD4(CCC_Float, "r_terrain_ao", &ps_r_terrain_ao, 0.f, 1.f);         // terrain micro contact AO strength
    CMD4(CCC_Integer, "r_terrain_debug", &ps_r_terrain_debug, 0, 9);     // ...5 beam directLit,6 beam mask,7 beam gap(red),8 ground-deposit sampled in-scatter,9 deposit gate(red)/inscat(green)
    CMD4(CCC_Float, "r_terrain_gloss", &ps_r_terrain_gloss, 0.f, 2.f);   // terrain dry sun-gloss strength
    CMD4(CCC_Integer, "r_puddle_debug", &ps_r_puddle_debug, 0, 2);       // 0 off, 1 coverage, 2 micro-height/flow
    CMD4(CCC_Integer, "r_puddle_sss", &ps_r_puddle_sss, 0, 2);           // 0 off, 1 our tuned water body, 2 SSFX-strict shading
    CMD4(CCC_Float, "r_puddle_level", &ps_r_puddle_level, 0.f, 1.f);     // puddle coverage (more/larger puddles)
    CMD4(CCC_Float, "r_puddle_scale", &ps_r_puddle_scale, 0.1f, 6.f);    // puddle size (bigger = smaller pools)
    CMD4(CCC_Float, "r_puddle_geo", &ps_r_puddle_geo, 0.f, 1.f);         // puddles follow REAL ground dips (Surface Field concavity)
    CMD4(CCC_Float, "r_wet_dist", &ps_r_wet_dist, 20.f, 400.f);          // range (m) wet shading survives to (SSFX ~200)
    CMD4(CCC_Float, "r_terrain_detail_dist", &ps_r_terrain_detail_dist, 30.f, 400.f); // range (m) terrain detail normal/AO/gloss survive to
    CMD4(CCC_Float, "r_bolt_flash", &ps_r_bolt_flash, 0.f, 4.f);         // lightning lifts the sky/hemi light (0 = no flash)
    CMD4(CCC_Float, "r_bump", &ps_r_bump, 0.f, 2.f);                     // static material normal-map strength (0 = off)
    CMD4(CCC_Integer, "r_bump_debug", &ps_r_bump_debug, 0, 2);           // 1 = decoded world normal, 2 = material gloss
    CMD4(CCC_Float, "r_gloss_scale", &ps_r_gloss_scale, 0.f, 4.f);       // material gloss -> IBL roughness (R4 r2_gloss_factor)
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
    CMD4(CCC_Float, "r_mud_deform", &ps_r_mud_deform, 0.f, 2.f);                   // mud footprints on soft terrain (0 = off)
    CMD4(CCC_Float, "r_mud_depth", &ps_r_mud_depth, 0.f, 1.f);                     // mud print POM carve depth (fraction)
    CMD4(CCC_Integer, "r_water_sim", &ps_r_water_sim, 0, 1);             // water flow sim master enable
    CMD4(CCC_Float, "r_water_rain", &ps_r_water_rain, 0.f, 5.f);         // sim rain input rate (depth/s)
    CMD4(CCC_Float, "r_water_evap", &ps_r_water_evap, 0.f, 20.f);        // sim leak rate (exp drain ∝ amount)
    CMD4(CCC_Float, "r_water_flow", &ps_r_water_flow, 0.f, 4.f);         // sim downhill accel (flow speed / streams)
    CMD4(CCC_Integer, "r_water_iters", &ps_r_water_iters, 1, 6);         // sim relaxation steps per frame
    CMD4(CCC_Float, "r_water_murk", &ps_r_water_murk, 0.f, 8.f);         // volumetric absorption /m (deeper feel)
    CMD4(CCC_Float, "r_water_refract", &ps_r_water_refract, 0.f, 0.2f);  // bottom refraction strength

    // Water BODIES (vk_pass_water) — see the ps_r_wtr_* block for the split
    // between this and the r_water_* puddle sim above.
    CMD4(CCC_Integer, "r_wtr",        &ps_r_wtr,        0, 1);
    // NOT savable: a debug view is set for one measurement and the client writes
    // user.ltx on exit — that is how a magenta-water build "randomly" survives a
    // restart. Same rule as every other look-at-it-once cvar in this engine.
    // ⚠ EXTEND THE RANGE WITH THE MODE. `r_wtr_debug 6` was silently REFUSED for
    // a while because the range still said 0..5, and the screenshot that came
    // back was an ordinary one — a debug view that lies about being on is worse
    // than no debug view.
    CMD4(CCC_IntegerNoSave, "r_wtr_debug", &ps_r_wtr_debug, 0, 9);   // 6 = fetch/shelter, 7 = pool mask, 8 = ripple field, 9 = spectral field
    CMD4(CCC_Float,   "r_wtr_wave",   &ps_r_wtr_wave,   0.f,  1.5f);
    CMD4(CCC_Float,   "r_wtr_calm",   &ps_r_wtr_calm,   0.f,  1.0f);
    CMD4(CCC_Float,   "r_wtr_scale",  &ps_r_wtr_scale,  0.02f, 4.f);
    CMD4(CCC_Float,   "r_wtr_speed",  &ps_r_wtr_speed,  0.f,  4.f);
    CMD4(CCC_Float,   "r_wtr_murk",   &ps_r_wtr_murk,   0.f,  6.f);
    CMD4(CCC_Float,   "r_wtr_refl",   &ps_r_wtr_refl,   0.f,  3.f);
    CMD4(CCC_Float,   "r_wtr_glint",  &ps_r_wtr_glint,  0.f,  8.f);
    CMD4(CCC_Float,   "r_wtr_detail", &ps_r_wtr_detail, 5.f,  400.f);
    CMD4(CCC_Float,   "r_wtr_rough",  &ps_r_wtr_rough,  0.005f, 0.5f);
    CMD4(CCC_Vector3, "r_wtr_color",  &ps_r_wtr_color,  (Fvector3{}), (Fvector3{ 1.f, 1.f, 1.f }));
    CMD4(CCC_Float,   "r_wtr_shore",  &ps_r_wtr_shore,  0.f, 3.f);    // waterline fade-out depth (m)
    CMD4(CCC_Float,   "r_wtr_foam",   &ps_r_wtr_foam,   0.f, 2.f);    // shore foam strength (0 = off)
    CMD4(CCC_Float,   "r_wtr_foam_w", &ps_r_wtr_foam_w, 0.02f, 3.f);  // foam band depth (m)
    CMD4(CCC_Float,   "r_wtr_micro",  &ps_r_wtr_micro,  0.f, 0.3f);   // capillary ripple, weather-independent
    CMD4(CCC_Float,   "r_wtr_wind",   &ps_r_wtr_wind,   0.f, 3.f);    // how hard the weather's wind drives the swell
    // Interactive ripple sim + tessellation.
    CMD4(CCC_Integer, "r_wtr_sim",        &ps_r_wtr_sim,        0, 1);
    CMD4(CCC_Float,   "r_wtr_sim_size",   &ps_r_wtr_sim_size,   16.f, 256.f);
    // NOT savable, all four: these are the constants OF THE SCHEME, not settings.
    // They decide whether a ripple stays in the puddle it was made in, and they get
    // retuned in the source as the sim is understood better — so the source must
    // win. A saved copy silently pins the machine to whatever number was current
    // the first time it quit cleanly, which is exactly how "I walk in one puddle
    // and every puddle ripples" came back after being fixed and measured.
    CMD4(CCC_FloatNoSave, "r_wtr_sim_damp",  &ps_r_wtr_sim_damp,  0.9f, 0.9999f);
    CMD4(CCC_FloatNoSave, "r_wtr_sim_speed", &ps_r_wtr_sim_speed, 0.01f, 0.5f);
    CMD4(CCC_FloatNoSave, "r_wtr_sim_shore", &ps_r_wtr_sim_shore, 0.5f, 1.f);  // 1 = reflecting rim (waves fill the pool)
    CMD4(CCC_FloatNoSave, "r_wtr_sim_reach", &ps_r_wtr_sim_reach, 0.f, 64.f);  // 0 = ripples cross the whole sheet
    // A fallback gate, set once to answer "is the mask doing anything?" — same rule
    // as every other look-at-it-once cvar.
    CMD4(CCC_IntegerNoSave, "r_wtr_sim_pools", &ps_r_wtr_sim_pools, 0, 1);   // 0 = ripples cross between puddles again
    CMD4(CCC_IntegerNoSave, "r_wtr_sim_lid",      &ps_r_wtr_sim_lid,      0, 1);      // 0 = old rasterized lid
    CMD4(CCC_IntegerNoSave, "r_wtr_sim_lid_rays", &ps_r_wtr_sim_lid_rays, 0, 20000);
    CMD4(CCC_IntegerNoSave, "r_wtr_sim_depth",     &ps_r_wtr_sim_depth,     0, 1);
    CMD4(CCC_FloatNoSave,   "r_wtr_sim_depth_ref", &ps_r_wtr_sim_depth_ref, 0.05f, 8.f);
    CMD4(CCC_FloatNoSave,   "r_wtr_sim_bed",       &ps_r_wtr_sim_bed,       0.5f, 1.f);
    CMD4(CCC_Float,   "r_wtr_sim_slope",  &ps_r_wtr_sim_slope,  0.f, 4.f);
    CMD4(CCC_Float,   "r_wtr_sim_height", &ps_r_wtr_sim_height, 0.f, 4.f);
    CMD4(CCC_Float,   "r_wtr_step",       &ps_r_wtr_step,       0.f, 0.5f);
    CMD4(CCC_FloatNoSave, "r_wtr_bow",    &ps_r_wtr_bow,        0.f, 6.f);
    // NEGATIVE = debug view: the world, the grass and the foliage all paint
    // shoreWetness() straight to the screen (red = wet). The only link in this
    // chain that has never been observed is what the function RETURNS.
    // -1 = debug view of the COMBINED wetness, -2 = the STATIC level map alone.
    // -1 both halves, -2 the level map alone, -3 its raw contents, -4 the tile alone.
    CMD4(CCC_FloatNoSave, "r_wtr_wet",      &ps_r_wtr_wet,      -4.f, 2.f);
    CMD4(CCC_FloatNoSave, "r_wtr_wet_dry",  &ps_r_wtr_wet_dry,  0.f, 600.f);
    CMD4(CCC_FloatNoSave, "r_wtr_wet_lift", &ps_r_wtr_wet_lift, 0.f, 0.5f);
    CMD4(CCC_Integer, "r_wtr_grid",       &ps_r_wtr_grid,       0,   384);
    CMD4(CCC_Float,   "r_wtr_tess",       &ps_r_wtr_tess,       0.f, 64.f);
    CMD4(CCC_Float,   "r_wtr_tess_near",  &ps_r_wtr_tess_near,  1.f, 100.f);
    CMD4(CCC_Float,   "r_wtr_tess_far",   &ps_r_wtr_tess_far,   5.f, 400.f);
    CMD4(CCC_Float,   "r_wtr_disp",       &ps_r_wtr_disp,       0.f, 4.f);
    CMD4(CCC_Float,   "r_wtr_swash",     &ps_r_wtr_swash,     0.f, 3.f);   // metres of run-up
    CMD4(CCC_Float,   "r_wtr_swash_t",   &ps_r_wtr_swash_t,   1.f, 30.f);  // seconds between crests
    CMD4(CCC_Float,   "r_wtr_surf",      &ps_r_wtr_surf,      0.f, 3.f);   // whitewater
    CMD4(CCC_Float,   "r_wtr_surf_h",    &ps_r_wtr_surf_h,    0.f, 2.f);   // offshore wave height
    CMD4(CCC_Float,   "r_wtr_surf_len",  &ps_r_wtr_surf_len,  2.f, 60.f);  // metres between crests
    CMD4(CCC_Float,   "r_wtr_film",      &ps_r_wtr_film,      0.01f, 0.5f); // run-up sheet depth
    CMD4(CCC_Float,   "r_wtr_foam_land", &ps_r_wtr_foam_land, 0.f, 2.f);
    CMD4(CCC_Float,   "r_wtr_foam_life", &ps_r_wtr_foam_life, 0.2f, 20.f);
    CMD4(CCC_Float,   "r_wtr_fetch",     &ps_r_wtr_fetch,     0.f, 8.f);   // 0 = every pool is open sea
    // NoSave, like every other gate in this file: a stale 0 in user.ltx would put
    // the well's swell back months from now and look like a fresh regression.
    CMD4(CCC_IntegerNoSave, "r_wtr_fetch_local", &ps_r_wtr_fetch_local, 0, 1);
    CMD4(CCC_Float,   "r_wtr_shelter",   &ps_r_wtr_shelter,   0.f, 1.f);   // 0 = a roof no longer calms the water
    // ---- spectral waves (vk_water_fft) ------------------------------------
    // NOT SAVABLE, the whole family. These are the constants OF A MODEL that is
    // brand new and will be retuned in source as it is understood — the same
    // rule the ripple sim's scheme constants live under, and for the same
    // reason: a value pinned in user.ltx the first time the game quit cleanly
    // silently outranks every later fix, and this project has lost days to
    // exactly that (r_wtr_sim_damp, twice). They graduate to savable when the
    // defaults stop moving.
    CMD4(CCC_IntegerNoSave, "r_wtr_fft",        &ps_r_wtr_fft,        0, 1);
    CMD4(CCC_FloatNoSave,   "r_wtr_fft_size",   &ps_r_wtr_fft_size,   32.f, 2000.f);
    CMD4(CCC_FloatNoSave,   "r_wtr_fft_ratio",  &ps_r_wtr_fft_ratio,  0.05f, 0.6f);
    CMD4(CCC_FloatNoSave,   "r_wtr_fft_wind",   &ps_r_wtr_fft_wind,   0.5f, 40.f);
    CMD4(CCC_FloatNoSave,   "r_wtr_fft_amp",    &ps_r_wtr_fft_amp,    0.f, 100.f);
    CMD4(CCC_FloatNoSave,   "r_wtr_fft_chop",   &ps_r_wtr_fft_chop,   0.f, 3.f);
    CMD4(CCC_FloatNoSave,   "r_wtr_fft_depth",  &ps_r_wtr_fft_depth,  0.5f, 500.f);
    CMD4(CCC_FloatNoSave,   "r_wtr_fft_repeat", &ps_r_wtr_fft_repeat, 10.f, 3600.f);
    CMD4(CCC_FloatNoSave,   "r_wtr_fft_small",  &ps_r_wtr_fft_small,  0.001f, 5.f);
    CMD4(CCC_FloatNoSave,   "r_wtr_fft_dir",    &ps_r_wtr_fft_dir,    0.f, 12.f);
    CMD4(CCC_FloatNoSave,   "r_wtr_fft_gain",   &ps_r_wtr_fft_gain,   0.f, 4.f);
    CMD4(CCC_FloatNoSave,   "r_wtr_fft_foam",   &ps_r_wtr_fft_foam,   0.f, 4.f);
    CMD4(CCC_FloatNoSave,   "r_wtr_fft_fetch",  &ps_r_wtr_fft_fetch,  1.f, 600.f);
    CMD4(CCC_FloatNoSave,   "r_wtr_fft_slope",  &ps_r_wtr_fft_slope,  0.f, 4.f);
    CMD4(CCC_Float,   "r_wtr_murk_still", &ps_r_wtr_murk_still, 1.f, 12.f); // murk multiplier for standing indoor water
    CMD4(CCC_Float,   "r_wtr_refract",   &ps_r_wtr_refract,   0.f, 0.3f);
    CMD4(CCC_Float,   "r_wtr_caustic",   &ps_r_wtr_caustic,   0.f, 4.f);
    CMD4(CCC_Float,   "r_wtr_caustic_p", &ps_r_wtr_caustic_p, 0.2f, 6.f);
    CMD1(CCC_WtrDrop, "r_wtr_drop");
    CMD1(CCC_WtrAudit, "r_wtr_audit");
    CMD4(CCC_Float, "r_spec_occ", &ps_r_spec_occ, 0.f, 1.f);             // bent-normal spec occlusion of wet reflections

    CMD3(CCC_Mask64, "r4_enable_tessellation", &ps_r2_ls_flags_ext, R2FLAGEXT_ENABLE_TESSELLATION); // Need restart

    CMD3(CCC_Mask64, "r4_wireframe", &ps_r2_ls_flags_ext, R2FLAGEXT_WIREFRAME); // Need restart
    CMD3(CCC_Mask64, "r2_steep_parallax", &ps_r2_ls_flags, R2FLAG_STEEP_PARALLAX);
    CMD3(CCC_Mask64, "r2_detail_bump", &ps_r2_ls_flags, R2FLAG_DETAIL_BUMP);

    CMD3(CCC_Token, "r2_sun_quality", &ps_r_sun_quality, qsun_quality_token);

    CMD3(CCC_Mask64, "r2_visor_refl", &ps_r2_ls_flags_ext, R2FLAGEXT_VISOR_REFL);
    CMD3(CCC_Mask64, "r2_visor_refl_control", &ps_r2_ls_flags_ext, R2FLAGEXT_VISOR_REFL_CONTROL);
    CMD4(CCC_Float, "r2_visor_refl_intensity", &ps_r2_visor_refl_intensity, 0.f, 1.f);
    CMD4(CCC_Float, "r2_visor_refl_radius", &ps_r2_visor_refl_radius, 0.3f, 0.6f);

    // ⭐The one antialiasing/upscaling row a player sees. See CCC_AAMode: it owns
    // r_dlss + r_dlss_quality, both of which stay registered below for A/B work.
    CMD3(CCC_AAMode, "r_aa_mode", &ps_r_pp_aa_mode, aa_mode_token);
    // ⚠The captions the player reads are NOT these token ids — the options screen
    // supplies them per row (`data-values` in main_menu.rml), the same way it does
    // for the numeric entries like r_dlss_preset. Token ids stay machine-readable
    // so a console line and a saved config keep meaning what they meant.

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

    // Texture quality slider — reinstated for the Vulkan streamer. Skips the top N
    // mips of WORLD/MODEL diffuse at load (UI/detail/lmap/terrain stay full-res). Each
    // step ≈ −75% of that texture's VRAM. Live: takes effect on the next level load
    // (already-resident textures keep their mips until reloaded / streamed).
    CMD4(CCC_Integer, "texture_lod", &psTextureLOD, 0, 3);

    // Texture streamer knobs (see globals above).
    CMD4(CCC_Integer, "r_txstream",          &ps_r_txstream,          0, 1);
    CMD4(CCC_Integer, "r_txstream_budget",   &ps_r_txstream_budget,   0, 16384);
    CMD4(CCC_Integer, "r_txstream_headroom", &ps_r_txstream_headroom, 0, 4096);
    CMD4(CCC_Integer, "r_txstream_reserve",  &ps_r_txstream_reserve,  0, 8192);
    CMD4(CCC_Integer, "r_txstream_loadcap",  &ps_r_txstream_loadcap,  0, 4096);
    CMD4(CCC_Integer, "r_txstream_plan_live", &ps_r_txstream_plan_live, 0, 1);   // old live-VRAM load-time fit
    CMD4(CCC_Integer, "r_vram_small_images",  &ps_r_vram_small_images,  0, 1);   // small-image pools + their size probe
    CMD4(CCC_Integer, "r_tex_prefetch",      &ps_r_tex_prefetch,      0, 1);       // parallel level-texture prefetch
    CMD4(CCC_Integer, "r_tex_prefetch_mb",   &ps_r_tex_prefetch_mb,   64, 4096);   // bytes parked ahead of the visual walk
    CMD4(CCC_Integer, "r_upload_wc_copy",    &ps_r_upload_wc_copy,    0, 1);      // streaming stores into the write-combined staging ring
    CMD4(CCC_Integer, "r_upload_cached",     &ps_r_upload_cached,     0, 1);      // HOST_CACHED staging ring (needs a restart)
    CMD4(CCC_Integer, "r_upload_copy_threads", &ps_r_upload_copy_threads, 0, 15);  // helper threads for the big staging copies
    CMD4(CCC_Integer, "r_geom_prefault",     &ps_r_geom_prefault,     0, 2);      // time the window faults apart from the geometry copy
    CMD4(CCC_Integer, "r_geom_lead",         &ps_r_geom_lead,         0, 1);      // stage geometry out of the prefetch's already-faulted view
    CMD4(CCC_Integer, "r_vis_warm",          &ps_r_vis_warm,          0, 1);      // prefault the visuals blob on workers before the walk
    CMD4(CCC_Integer, "r_vis_walk_split",    &ps_r_vis_walk_split,    0, 1);      // rdtsc split of the visual walk loop
    CMD4(CCC_Integer, "r_vis_triage",        &ps_r_vis_triage,        0, 1);      // per-visual glass/glow probes in LoadTexture
    CMD4(CCC_Integer, "r_vis_guard",         &ps_r_vis_guard,         0, 1);      // between-phases sweep over Visuals[] (repeat-load corruption hunt)
    CMD4(CCC_Integer, "r_tex_prefetch_lmaps_first", &ps_r_tex_prefetch_lmaps_first, 0, 1);  // lightmaps ahead of the diffuse bases
    CMD4(CCC_Integer, "r_prewarm_async",     &ps_r_prewarm_async,     0, 1);      // weather variants off the render thread
    CMD4(CCC_Integer, "r_thm_cache",         &ps_r_thm_cache,         0, 1);      // remember parsed .thm per base name
    CMD4(CCC_Integer, "r_clpage_map",        &ps_r_clpage_map,        0, 1);      // pinned cluster pages upload from a mapping
    CMD4(CCC_Integer, "r_tex_residency_threads", &ps_r_tex_residency_threads, 0, 16); // helpers that read a residency plan's .dds files
    CMD4(CCC_Integer, "r_tex_repack_threads", &ps_r_tex_repack_threads, 0, 15);   // helpers for the BC3->BC4 lightmap gather
    CMD4(CCC_Integer, "r_tex_materialize",    &ps_r_tex_materialize,    0, 2);    // 1 = workers build textures, 2 = only once the walk starts
    CMD4(CCC_Integer, "r_tex_mat_threads",    &ps_r_tex_mat_threads,    0, 16);   // concurrent texture builders (0 = all)
    CMD4(CCC_Integer, "r_tex_repack_scratch", &ps_r_tex_repack_scratch, 0, 1);    // reuse one gather destination per thread
    CMD4(CCC_Integer, "r_tex_repack_verify",  &ps_r_tex_repack_verify,  0, 1);    // compare the gather against the plain copy



    CMD1(CCC_VideoMemoryStats, "r_txstream_stats");
    CMD1(CCC_VramDump,         "r_vram_dump");

    // Cluster-LOD page streaming (Stage B; read at level load).
    CMD4(CCC_Integer, "r_clpage",        &ps_r_clpage,        0, 1);
    CMD4(CCC_Integer, "r_clpage_budget", &ps_r_clpage_budget, 32, 4096);
    CMD4(CCC_Integer, "r_cl_audit",      &ps_r_cl_audit,      0, 1);
    CMD4(CCC_Integer, "r_pool_compact",  &ps_r_pool_compact,  0, 1);
    CMD4(CCC_Integer, "r_compose",       &ps_r_compose,       0, 25);

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
