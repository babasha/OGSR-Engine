// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).
//
// Virtual Shadow Maps — sun directional clipmap. See vk_vsm.h.
// PHASE 1A: clipmap params + page MARKING compute + a diagnostic visible-page count.

#include "stdafx.h"
#include "vk_vsm.h"
#include "HW_Vulkan.h"
#include "vk_shaders.h"           // g_ShaderManager (vsm_mark.comp.spv)
#include "vk_buffer.h"            // CVulkanBuffer
#include "vk_shadow_gpu.h"        // ShadowGPU caster meta + groups (reused for VSM binning/render)
#include "vk_world_gpu.h"         // WorldGPU cluster-LOD entries (Phase 3: r_vsm_cluster bin/render)
#include "vk_cluster_stream.h"    // Stage B: page residency bits + slot bases (vsm_bin_cluster bindings)
#include "vk_pass_skinned.h"      // Skinned_CollectCasters/GetBoneSet — NPC casters into the atlas
#include "CRender_Vulkan.h"       // RImplementation.Details/Trees — grass + tree casters into the atlas
#include "vk_DetailManager.h"     // CDetailManager VSM grass getters + DetailInstance/GPU_OUTPUT_CAPACITY
#include "vk_TreeManager.h"       // CTreeManager VsmBin/VsmRender — tree casters into the atlas
#include "vk_barriers.h"          // ImageBarrier (atlas layout transitions)
#include "vk_command_buffer.h"    // VK_FRAMES_IN_FLIGHT
#include <unordered_map>          // per-stride page pipelines
#include "vk_profiler.h"          // Prof::NameBuffer
#include "../../xr_3da/device.h"  // Device.dwTimeGlobal
#include "../../xr_3da/IGame_Persistent.h" // g_pGamePersistent->Environment()
#include "../../xr_3da/Environment.h"      // CurrentEnv->sun_dir
#include "vk_dlss.h"              // VK::Dlss::Enabled — DLSS-aware history weight (double-temporal avoidance)
#include "vk_world_material.h"    // WorldMaterial::set/alphaRef + WorldMaterialCache::GetSetLayout (r_vsm_at discard pipeline)

extern int   ps_r_vsm;
extern int   ps_r_vsm_debug;
extern int   ps_r_sun_night_freeze;   // freeze the sun-shadow update when the sun is below the horizon
extern float ps_r_sun_night_lum;      // sun_color luminance below which it's "night" (secondary signal)
extern float ps_r_sun_night_alt;      // to-sun.y below which the sun counts as below the horizon
extern int   ps_r_vsm_hzb;   // shadow-HZB: cull casters fully behind cached occluders (kills VSMrender overdraw)
extern float ps_r_vsm_base;       // clipmap level-0 extent (m) → finest texel = base/4096 (live, settings-bound)
extern float ps_r_vsm_bias;       // receiver depth-compare bias, STATIC atlas (live)
extern float ps_r_vsm_bias_dyn;   // receiver depth-compare bias, DYNAMIC atlas (live; tiny — no ground in it)
extern float ps_r_vsm_bias_min;   // slope-scaled static bias: min slack (live; 0 = legacy constant ps_r_vsm_bias)
extern float ps_r_vsm_raster_bias;   // atlas write-side raster depth bias, constant (D16 units; live)
extern float ps_r_vsm_raster_slope;  // atlas write-side raster depth bias, slope factor (live)
extern int   ps_r_vsm_grass_static;   // far-grass static-cache hybrid (L1/L2 rigid into dirty static pages)
extern int   ps_r_vsm_temporal;   // TAA-for-shadows: clipmap jitter + reprojected history accumulate (live)
extern float ps_r_vsm_ta_blend;       // history weight (EMA alpha) for the temporal resolve (live)
extern float ps_r_vsm_ta_blend_dyn;   // history weight on dyn-atlas-shadowed pixels (anti-"jelly" for wind/NPC shadows, live)
extern float ps_r_vsm_ta_clamp;        // (1) neighbourhood clamp: history bound to current shadow ±tol (anti-ghost, live)
extern float ps_r_vsm_ta_motion;       // (2) reprojected motion (px) at which history weight fades to the floor (live)
extern float ps_r_vsm_ta_motion_floor; // (2) history weight at/after that motion (live)
extern float ps_r_vsm_ta_blend_dlss;   // (3) history-weight scale when r_dlss on (DLSS also resolves the shadow → avoid double blur, live)
extern int   ps_r_vsm_soft;            // SOFT SHADOWS (stochastic PCSS): filter taps, 0 = legacy 3x3 PCF (live)
extern int   ps_r_vsm_soft_search;     // blocker-search taps (live)
extern float ps_r_vsm_soft_angle;      // sun cone half-angle in DEGREES (0.265 = physical; higher = cinematic) (live)
extern float ps_r_vsm_soft_range;      // max blocker search distance (m) — caps the widest penumbra (live)
extern float ps_r_vsm_soft_clamp;      // temporal neighbourhood clamp used while soft is on (wider: it must pass stochastic noise) (live)
extern int   ps_r_vsm_grass;      // cast near grass into the atlas (L0 only, GPU-driven, 1-frame stale) (live)
extern int   ps_r_vsm_tree_wind;       // near/far wind hybrid (near trees re-raster into dyn atlas) (live)
extern float ps_r_vsm_tree_wind_dist;  // near-set light-space radius (m) — the dominant dyn-tree cost knob (live)
extern int   ps_r_vsm_tree_vrs;        // 2x2 coarse shading on dyn crown shadows (A/B)
extern int   ps_r_vsm_cadence;         // update the DYN pass every Nth frame (0/1=every frame; 2-3 = ~30Hz), reuse frozen otherwise (A/B)
extern float ps_r_vsm_cadence_still;   // dyn-skip only when the camera moved < this (m/frame); moving → full-rate dyn (no flicker) (live)
extern float ps_r_vsm_cadence_sun;     // force a dyn update when the sun rotated > this (deg) since last update; catches env-keyframe steps (live)
extern float ps_r_vsm_grass_dist; // max grass cast distance from camera, m (live)
extern int   ps_r_vsm_cache;      // Phase 1b: toroidal per-page cache (1) vs render-all baseline (0) (live)
extern int   ps_r_vsm_cache_refresh; // round-robin refresh period (frames) for the moving sun; smaller = fresher/costlier (live)
extern float ps_r_vsm_lod_dist;      // caster-LOD: distance (m) beyond which opaque casters draw their coarse slice (0 = off) (live)
extern int   ps_r_vsm_mark_half;     // page-mark at half-res (1) = 4x fewer threads/atomics, vs full-res (0) (live)
extern int   ps_r_vsm_dyn_gate;      // resolve skips dyn-atlas taps on pages with no dynamic casters (dynUsed flags); 0 = sample dyn on every resident page (live)
extern int   ps_r_vsm_debug_dyn;     // write dyn-atlas occlusion to mask B; tonemap tints it red (NPC/grass shadow visualizer) (live)
extern int   ps_r_vsm_cluster;       // Phase 3: bin the WorldGPU cluster-LOD entries instead of per-mesh ShadowGPU casters (live; flips invalidate the cache)
extern float ps_r_vsm_cluster_lod;   // Phase 3: LOD error budget in clipmap-level texels (live; changes invalidate the cache)
extern int   ps_r_vsm_at;            // alpha-tested statics into the static atlas (cluster path; live, flips invalidate the cache)
extern int   ps_r_profiler;             // telemetry gate: the throttle/dirty log line also prints in plain perf sessions
extern int   ps_r_vsm_throttle;         // UE5-style cost feedback: over-budget frames mark COARSER clipmap levels (fewer pages = less alpha-test fill) (live, A/B)
extern float ps_r_vsm_throttle_budget;  // World/VSMrender GPU budget (ms) the controller steers to (live)
extern float ps_r_vsm_throttle_max;     // max LOD bias (clipmap levels) the throttle may apply (live)
extern int   ps_r_vsm_dirty_budget;     // max scrolled-in (wrong-tile) static pages re-rendered per frame; extra defer a frame (anti-spike); 0 = unlimited (live, A/B)
extern int   ps_r_vsm_rmask;            // receiver mask (UE5): 8×8 sampled-cell mask per page; dyn tree bins + vox cull sub-page-cull against it (live, A/B)
extern int   ps_r_vsm_gaze;             // gaze refresh: pages the player looks at re-render on a cadence ∝ on-screen footprint — visible shadows glide, peripheral ones tick on round-robin (live, A/B)
extern int   ps_r_vsm_gaze_pages;       // gaze budget: max gaze-refreshed pages per frame (live)
extern int   ps_r_vsm_gaze_px;          // gaze full-rate threshold: mark samples for every-frame refresh; cadence = ceil(this/hits) frames (live)

namespace VK { namespace VSM {

namespace {

// Clipmap geometry — MUST match shaders/vsm_common.glsl.
constexpr u32   kLevels      = 6;
constexpr u32   kVirtualRes  = 4096;
constexpr u32   kPageSize    = 128;
constexpr u32   kPagesAxis   = kVirtualRes / kPageSize;          // 32
constexpr u32   kPagesPerLvl = kPagesAxis * kPagesAxis;          // 1024
constexpr u32   kPageCount   = kLevels * kPagesPerLvl;           // 6144
constexpr u32   kMaxPhys     = 2048;                             // DYNAMIC physical atlas pages (MUST match VSM_MAX_PHYS)
constexpr u32   kMaxPhysS    = kPageCount;                       // STATIC toroidal atlas pages = 6144 (MUST match VSM_MAX_PHYS_S)
constexpr u32   kPagesCap    = 1024;                            // max pages per caster (MUST match VSM_PAGES_CAP)
constexpr u32   kGroupStride = 1024;                            // max VSM draws per group region (MUST match VSM_GROUP_STRIDE)
constexpr u32   kMaxGroups   = 256;                             // ShadowGPU group ceiling (sizes vsmIndirect)
constexpr u32   kMaxCasters  = 8192;                            // caster ceiling (sizes casterPages)
// Phase 3 cluster bin (r_vsm_cluster) at scale: the bin runs over a CPU-static
// CANDIDATE set, not all entries. The ortho LOD cut is camera-independent, so
// "can this entry ever cast" is a static predicate: parentError > errB(0) —
// entries below the finest level's cut never draw (their parent draws instead),
// and that bound also covers streaming leaves (sleaf still requires
// parentError > errB(L)). casterPages is a shared ARENA (atomic cursor, exact
// per-entry reservations — no per-entry cap, so terrain chunks overlapping
// 140+ dirty pages during sun round-robin are never truncated), and indirect
// cmd regions are per BUFFER-COMBO (vb/ib/stride/iType — after VB paging most
// groups collapse onto the cluster pools), not per material group, so the
// caster draw loop stays a handful of vkCmdDrawIndexedIndirectCount.
constexpr u32   kClSlots      = kMaxCasters * kPagesCap;                   // reused casterPages arena (8.4M u32)
constexpr u32   kMaxCombos    = 4096;                                      // distinct (vb,ib,stride,iType) ceiling (sizes s_clCount; 12-bit pack in candIdx). Pripyat: 769 level VB pools → 1469 real combos (plain meshes bind per-pool pairs)
// No cmd-count ceiling: the cluster path gets its OWN indirect buffer sized to
// the candidate count (Pripyat: 574k candidates = 71% of entries — the VSM L0
// error budget is centimetres, so the finest cut legitimately sits at the DAG
// leaves and "leaves + all ancestors" is most of the DAG; ~11.5 MB there).
// The 20-bit entry-id gate (1M) bounds it.
constexpr u32   kAtlasW      = 64;                              // DYNAMIC atlas pages across (MUST match VSM_ATLAS_W)
constexpr u32   kAtlasH      = 32;                              // DYNAMIC atlas pages down  (MUST match VSM_ATLAS_H)
constexpr u32   kAtlasW_S    = 64;                              // STATIC atlas pages across (MUST match VSM_ATLAS_W_S) -> 8192 x 12288
constexpr u32   kAtlasH_S    = 96;                              // STATIC atlas pages down   (MUST match VSM_ATLAS_H_S)
constexpr float kZNear       = -1000.0f;                         // light-space depth range
constexpr float kZFar        =  1000.0f;
constexpr u32   kSkinnedCap  = 256;                              // max atlas pages a skinned leaf bins into (across all clipmap levels; log showed ~155-188 for close NPCs)
constexpr u32   kMaxSkinned  = 256;                              // max skinned leaves rasterized per frame

constexpr u32 N = VK_FRAMES_IN_FLIGHT;

bool s_inited = false;
bool s_dead   = false;

VkPipeline            s_pipe   = VK_NULL_HANDLE;
VkPipelineLayout      s_layout = VK_NULL_HANDLE;
VkDescriptorSetLayout s_setL   = VK_NULL_HANDLE;
VkDescriptorPool      s_pool   = VK_NULL_HANDLE;
VkDescriptorSet       s_set[N] = {};
VkSampler             s_depthSampler = VK_NULL_HANDLE;

CVulkanBuffer* s_ubo[N]   = {};   // host-visible clipmap params (per frame in flight)
void*          s_uboPtr[N] = {};
CVulkanBuffer* s_needed   = nullptr;   // device, per-level page flags (cleared each frame)
CVulkanBuffer* s_counter  = nullptr;   // device, unique-page atomic counter (mark)
CVulkanBuffer* s_rmask    = nullptr;   // device, receiver mask: 2 u32 per virtual page = 8×8 sampled cells
                                       // (written by vsm_mark under r_vsm_rmask; dyn tree bins + vox cull test
                                       // caster rects against it — sub-page cull, the UE5 receiver-mask idea)
CVulkanBuffer* s_pageHits = nullptr;   // device, kPageCount u32: per-page sampled-pixel count (r_vsm_gaze; cleared each frame)
CVulkanBuffer* s_readback = nullptr;   // host, counters copied here for the debug log (8 u32)
u32*           s_readPtr  = nullptr;

// Page tables: virtual page -> physical atlas slot + the per-slot render list. The STATIC
// table/list (s_pageTable/s_pageList) is written by the toroidal RESIDENCY pass (vsm_resid);
// the DYNAMIC table/list (s_dynPageTable/s_dynPageList) is demand-allocated from scratch
// every frame by the alloc pass (vsm_alloc) — NPC/grass casters move.
VkPipeline            s_allocPipe   = VK_NULL_HANDLE;
VkPipelineLayout      s_allocLayout = VK_NULL_HANDLE;
VkDescriptorSetLayout s_allocSetL   = VK_NULL_HANDLE;
VkDescriptorPool      s_allocPool   = VK_NULL_HANDLE;
VkDescriptorSet       s_dynAllocSet = VK_NULL_HANDLE;   // fixed buffers — written once in Init
CVulkanBuffer* s_pageTable = nullptr;  // device, kPageCount u32 (virtual -> STATIC slot / UNMAPPED)
CVulkanBuffer* s_pageList  = nullptr;  // device, kMaxPhysS uvec4 (STATIC slot -> level,px,py)
CVulkanBuffer* s_dynPageTable = nullptr; // device, kPageCount u32 (virtual -> DYNAMIC slot / UNMAPPED)
CVulkanBuffer* s_dynPageList  = nullptr; // device, kMaxPhys uvec4 (DYNAMIC slot -> level,px,py)
CVulkanBuffer* s_dynAllocInfo = nullptr; // device, [0]=count, [1..kLevels]=per-level (dynamic)
CVulkanBuffer* s_dynPageUsed  = nullptr; // device, kMaxPhys u32 (dyn slot -> 1 if any NPC/grass caster binned into it; the resolve skips the dynamic atlas on untouched pages)

// Toroidal STATIC-atlas residency (Phase 1b): persistent slot->tile + a per-frame dirty set.
// One compute pass maps each visible page to its fixed toroidal slot and flags it dirty if the
// slot holds a different tile (camera scrolled) or a round-robin refresh is due (sun moving).
VkPipeline            s_residPipe   = VK_NULL_HANDLE;
VkPipelineLayout      s_residLayout = VK_NULL_HANDLE;
VkDescriptorSetLayout s_residSetL   = VK_NULL_HANDLE;
VkDescriptorPool      s_residPool   = VK_NULL_HANDLE;
VkDescriptorSet       s_residSet    = VK_NULL_HANDLE;   // static (fixed buffers) — written once
CVulkanBuffer* s_physTile  = nullptr;  // device, kMaxPhysS uvec2 (slot -> absTile) — PERSISTENT across frames
CVulkanBuffer* s_slotDirty = nullptr;  // device, kMaxPhysS u32 (1 if dirty this frame; cleared each frame)
CVulkanBuffer* s_dirtyList = nullptr;  // device, kMaxPhysS u32 (compact dirty slots)
CVulkanBuffer* s_drawClear = nullptr;  // device, VkDrawIndirectCommand (clear draw: instanceCount = dirty count)
bool           s_physInit  = false;    // physTile filled to EMPTY (once after create)
CVulkanBuffer* s_residRB   = nullptr;  // host, dirty count readback (diagnostic)
u32*           s_residPtr  = nullptr;

// shadow-HZB (r_vsm_hzb): per-slot MAX of the prior-frame static-atlas depth = occluder for the
// tree/opaque caster bins. s_priorValid (written by residency) gates which slots hold a valid
// cached occluder (physTile matched). Reduce runs in MarkPages before RenderAtlas overwrites.
VkPipeline            s_hzbPipe   = VK_NULL_HANDLE;
VkPipelineLayout      s_hzbLayout = VK_NULL_HANDLE;
VkDescriptorSetLayout s_hzbSetL   = VK_NULL_HANDLE;
VkDescriptorPool      s_hzbPool   = VK_NULL_HANDLE;
VkDescriptorSet       s_hzbSet    = VK_NULL_HANDLE;
CVulkanBuffer* s_pageMax    = nullptr;  // device, kMaxPhysS float (per static slot: max prior depth, 1.0 = no occlusion)
CVulkanBuffer* s_pageMaxBlk = nullptr;  // device, kMaxPhysS × 64 float (8×8 blocks of 16×16 texels — brick-granular occluders)
CVulkanBuffer* s_priorValid = nullptr;  // device, kMaxPhysS u32 (1 = slot's cached depth is this world tile's)
CVulkanBuffer* s_slotDirtyPrev = nullptr;  // device, kMaxPhysS u32 — LAST frame's slotDirty (reduce refreshes only re-rendered slots)
bool           s_hzbWasOn  = false;     // OFF→ON flip detector: reset the persistent occluder maxes (stale-content hazard)

// Clear-dirty graphics pipeline (depth-only: one quad per dirty slot -> depth 1.0).
VkPipeline            s_clearPipe   = VK_NULL_HANDLE;
VkPipelineLayout      s_clearLayout = VK_NULL_HANDLE;
VkDescriptorSetLayout s_clearSetL   = VK_NULL_HANDLE;
VkDescriptorPool      s_clearPool   = VK_NULL_HANDLE;
VkDescriptorSet       s_clearSet    = VK_NULL_HANDLE;
VkShaderModule        s_clearVS     = VK_NULL_HANDLE;

Fvector s_prevSunDir = { 0.f, -1.f, 0.f };   // last frame's sun dir (round-robin enable)
bool    s_sunMoving   = false;                // sun rotated since last frame (drives round-robin refresh)

// Throttle (r_vsm_throttle) — the UE5 VirtualShadowMapThrottle idea: feed last frame's
// measured VSM render cost back into a clipmap LOD bias. Over budget → mark coarser
// (each +1 level quarters the marked pages = real FILL relief, the thing cull/meshlet/
// HZB/D16 never touched); under budget → recover slowly (recovery re-renders finer
// pages, so it must trickle). s_throttleVal is the continuous controller state;
// s_lodBias its integer quantization with a deadband (no boundary flip-flop).
float s_throttleVal = 0.f;
u32   s_lodBias     = 0;

// Throttle/dirty TELEMETRY (r_vsm_debug or r_profiler): 3-second windows so the log can
// attribute "who eats what" — VSMrender cost avg/max + % of frames over budget, % of time
// at each bias level, dirty/wrong page counts and how many redraws the budget deferred.
// Bias TRANSITIONS log unconditionally (rare + the key A/B event).
float s_thCostSum = 0.f, s_thCostMax = 0.f;   // World/VSMrender ms over the window
u32   s_thFrames = 0, s_thOver = 0;           // frames sampled / frames over budget
u32   s_thBiasHist[3] = {};                   // frames spent at bias 0 / 1 / >=2
u64   s_wrongSum = 0, s_dirtySum = 0, s_deferSum = 0, s_gazeSum = 0;
u32   s_wrongMax = 0, s_dirtyMax = 0, s_gazeMax = 0, s_rbFrames = 0;
u32   s_thLastLogMs = 0;

// std430 push for vsm_resid.comp: per-level window page-base packed as ivec4[3] +
// up to 4 invalidation circles (light-space xy, radius; [0].w = L0 page width in m) —
// fed by tree near/far wind-hybrid transitions. 64 + 64 = 128 B (the push limit).
struct ResidPush { s32 pageBase[12]; u32 frame, refreshN, sunMoving, forceDirty; float inval[16]; };

Fmatrix s_sunView;   // this frame's world->light view (BeginFrame -> transition spheres to light XY)

// Caster binning: per-page caster lists (the per-page render's draw input).
VkPipeline            s_binPipe   = VK_NULL_HANDLE;
VkPipelineLayout      s_binLayout = VK_NULL_HANDLE;
VkDescriptorSetLayout s_binSetL   = VK_NULL_HANDLE;
// Phase 3 cluster bin (vsm_bin_cluster.comp): WorldGPU entries + per-level
// ortho LOD cut. Separate layout (9 bindings: + groupBase) + per-frame sets.
VkPipeline            s_binClPipe   = VK_NULL_HANDLE;
VkPipelineLayout      s_binClLayout = VK_NULL_HANDLE;
VkDescriptorSetLayout s_binClSetL   = VK_NULL_HANDLE;
VkDescriptorPool      s_binClPool   = VK_NULL_HANDLE;
VkDescriptorSet       s_binClSet[N] = {};
CVulkanBuffer*        s_clCount     = nullptr;   // per-combo draw counts (kMaxCombos)
// Bin/render path snapshot: MarkPages decides, RenderAtlas follows the SAME
// frame's decision (a live cvar flip between the two would mismatch buffers).
bool  s_clusterFrame     = false;
bool  s_lastClusterMode  = false;   // cache invalidation on live path flips
float s_lastClusterK     = -1.f;    // + on error-budget changes (page content depends on both)
// Cluster candidate table (CPU-static, see RebuildClusterCandidates): the set of
// entries that can EVER cast + their cmd-region routing. Rebuilt lazily when the
// WorldGPU build stamp moves (level reload) or errB(0) changes (k / clipmap base).
// mat != null ⇒ ALPHA-TESTED combo (r_vsm_at): keyed additionally by material so
// the draw can bind its diffuse; sorted AFTER the opaque combos (one pipeline-
// layout switch per pass). tcOffset = the UV byte offset for the AT vertex input.
struct ClCombo { VkBuffer vb; VkBuffer ib; u32 stride; VkIndexType iType; u32 base; u32 count;
                 const WorldMaterial* mat; u32 tcOffset; };
xr_vector<ClCombo> s_clCombos;               // draw loop: one indirect-count draw each
CVulkanBuffer* s_clCandIdx   = nullptr;      // device, u32[candCount]: (combo<<20)|entry
CVulkanBuffer* s_clComboBase = nullptr;      // device, u32[kMaxCombos] cmd-region bases
CVulkanBuffer* s_clIndirect  = nullptr;      // device, candCount VkDrawIndexedIndirectCommand (cluster path's own; per-mesh keeps s_vsmIndirect)
u32   s_clCandCount = 0;
u32   s_clCandStamp = 0;                     // WorldGPU::BuildStamp() the table matches (0 = none)
float s_clCandErrB0 = -1.f;                  // errB(0) it was built for
bool  s_clCandAT    = false;                 // table includes AT combos (r_vsm_at at build time)
bool  s_lastAtMode  = false;                 // cache invalidation on r_vsm_at flips (page content changes)
bool  s_clCandOk    = false;
VkDescriptorPool      s_binPool   = VK_NULL_HANDLE;
VkDescriptorSet       s_binSet[N] = {};   // per-frame (UBO + ShadowGPU meta vary)
CVulkanBuffer* s_casterPages   = nullptr;  // device, kMaxCasters * kPagesCap u32 (c*CAP+i -> page slot)
CVulkanBuffer* s_vsmIndirect   = nullptr;  // device, kMaxGroups * kGroupStride VkDrawIndexedIndirectCommand
CVulkanBuffer* s_vsmGroupCount = nullptr;  // device, kMaxGroups u32 (per-group draw count)
CVulkanBuffer* s_binStats      = nullptr;  // device, [0]=draws [1]=instances [2]=maxPagesPerCaster [3]=groupOverflow
CVulkanBuffer* s_binReadback   = nullptr;  // host
u32*           s_binReadPtr    = nullptr;

u32 s_frame    = 0;
u32 s_lastLog  = 0;
bool s_sunDown = false;   // last BeginFrame: sun below horizon → freeze the whole sun-shadow update
u32 s_curSlot  = 0;   // frame-in-flight slot used by the latest MarkPages (RenderAtlas reuses it)
bool s_dynSkip = false;   // cadence: this frame REUSES the frozen DYNAMIC atlas (NPC/grass/near-trees); static + sun still update
Fvector s_dynPrevCam = { 0.f, 0.f, 0.f };   // camera pos last frame — dyn-skip motion gate (the frozen dyn atlas is only aligned while still)
Fvector s_dynPrevDir = { 0.f, 0.f, 1.f };   // camera dir last frame — turning also scrolls the marked pages
// Cadence staleness diagnostics (r_vsm_debug): the frozen dyn atlas is only valid for the SUN
// angle + clipmap window it was rendered at. A living sun rotates the light-space lattice → the
// window page base drifts and SNAPS a page → the frozen dyn shadows misalign for a frame (the
// suspected "тени пропали на кадр"). Snapshot at each dyn update, compare on skip frames.
u32     s_dynUpdFrame = 0;
Fvector s_dynUpdSun   = { 0.f, -1.f, 0.f };
s32     s_dynUpdBase[8][2] = {};   // per-level window page base at last dyn update (kLevels <= 8)
float   s_cadSunDriftMax = 0.f;    // max sun drift (deg) seen on a skip frame since the last summary log
u32     s_cadHeldMax     = 0;      // max frames the dyn atlas was held frozen since the last summary log
u32     s_cadSnapTally   = 0;      // count of skip frames with a page snap (flicker-risk) since the last summary log
int  s_lastCad    = -1;   // detect r_vsm_cadence change → reset the tally
u32  s_updTally   = 0, s_skipTally = 0;  // frames updated / skipped since last cadence change (diag proof)

s32 s_pageBase[kLevels][2] = {};   // this frame's per-level page-lattice index of the window's first page (BeginFrame -> residency push)

// Physical atlas + the per-page rasterization pipeline. TWO atlases (Phase 2 split):
//  - STATIC: opaque-static + tree casters (world-static → cacheable in Phase 2).
//  - DYNAMIC: skinned(NPC) + grass casters (re-rendered every frame, cleared each frame).
// Both share the per-frame page table / page list / slot layout (same atlas grid), so a
// virtual page maps to the SAME slot/atlas sub-rect in both images. The resolve samples
// both and takes the nearer occluder (min depth) → identical to one combined atlas, but
// the static image can later freeze while the dynamic image keeps redrawing.
VkImage       s_atlasImage   = VK_NULL_HANDLE;   // STATIC atlas
VmaAllocation s_atlasAlloc   = VK_NULL_HANDLE;
VkImageView   s_atlasView    = VK_NULL_HANDLE;
VkSampler     s_atlasSampler = VK_NULL_HANDLE;   // shared by both atlases (NEAREST, clamp-to-white)
bool          s_atlasFirst   = true;
VkImage       s_dynImage     = VK_NULL_HANDLE;   // DYNAMIC atlas (NPC + grass)
VmaAllocation s_dynAlloc     = VK_NULL_HANDLE;
VkImageView   s_dynView      = VK_NULL_HANDLE;
bool          s_dynFirst     = true;
VkPipelineLayout      s_renderLayout = VK_NULL_HANDLE;
VkDescriptorSetLayout s_renderSetL   = VK_NULL_HANDLE;
VkDescriptorPool      s_renderPool   = VK_NULL_HANDLE;
VkDescriptorSet       s_renderSet[N] = {};   // per-frame (UBO varies)
VkShaderModule        s_pageVS       = VK_NULL_HANDLE;
std::unordered_map<u32, VkPipeline> s_pagePipes;   // keyed by vertex stride
// r_vsm_at: alpha-tested page pipeline (vsm_page_at VS+FS, discard by the
// material's diffuse). Layout = {render set, material set} + 16 B push
// (uvScale + alphaRef), created LAZILY at first draw (the material cache's
// set layout may not exist at VSM init time). Pipelines keyed (stride<<8)|tcOffset.
VkPipelineLayout      s_renderATLayout = VK_NULL_HANDLE;
VkShaderModule        s_pageATVS       = VK_NULL_HANDLE;
VkShaderModule        s_pageATFS       = VK_NULL_HANDLE;
bool                  s_pageATFailed   = false;   // one attempt; missing spv → AT combos skip
std::unordered_map<u32, VkPipeline> s_pageATPipes;

// Temporal resolve (TAA-for-shadows): screen-space mask + reprojected history.
VkPipeline            s_resolvePipe   = VK_NULL_HANDLE;
VkPipelineLayout      s_resolveLayout = VK_NULL_HANDLE;
VkDescriptorSetLayout s_resolveSetL   = VK_NULL_HANDLE;
VkDescriptorPool      s_resolvePool   = VK_NULL_HANDLE;
VkDescriptorSet       s_resolveSet[N] = {};   // per-frame (depth/history/output/UBOs vary)
CVulkanBuffer*        s_resolveUbo[N]    = {};
void*                 s_resolveUboPtr[N] = {};
VkImage       s_maskImage[N]   = {};   // RGBA16F screen-space mask (R=lit, G=dist), N-deep ping-pong
VmaAllocation s_maskAlloc[N]   = {};
VkImageView   s_maskView[N]    = {};
bool          s_maskFirst[N]   = {};   // UNDEFINED → GENERAL transition pending (per slot)
VkSampler     s_maskSampler    = VK_NULL_HANDLE;   // LINEAR clamp-to-edge (history + receiver)
VkExtent2D    s_maskExtent     = {};
bool          s_maskValid      = false;   // resolved at least once at the current extent
u32           s_resolveCount   = 0;       // resolves since last (re)create — history valid once >= 1
Fmatrix       s_prevViewProj;             // last frame's scene view·proj (history reproject)
Fvector       s_prevCamPos     = {};      // last frame's camera (history disocclusion check)
Fvector       s_curCamPos      = {};      // this frame's camera (captured in BeginFrame)

constexpr float kRejectTol = 0.05f;   // history distance reject tolerance (fraction of depth)

// Skinned (NPC) casters into the atlas: bin (find resident pages) + skin-then-route render.
VkPipeline            s_skinBinPipe   = VK_NULL_HANDLE;
VkPipelineLayout      s_skinBinLayout = VK_NULL_HANDLE;
VkDescriptorSetLayout s_skinBinSetL   = VK_NULL_HANDLE;
VkDescriptorPool      s_skinBinPool   = VK_NULL_HANDLE;
VkDescriptorSet       s_skinBinSet[N] = {};
VkPipelineLayout      s_skinPageLayout = VK_NULL_HANDLE;
VkDescriptorSetLayout s_skinPageSetL   = VK_NULL_HANDLE;
VkDescriptorPool      s_skinPagePool   = VK_NULL_HANDLE;
VkDescriptorSet       s_skinPageSet[N] = {};
VkShaderModule        s_skinPageVS     = VK_NULL_HANDLE;
std::unordered_map<u32, VkPipeline> s_skinPagePipes;   // keyed by vertex stride
CVulkanBuffer* s_skinMeta[N]      = {};   void* s_skinMetaPtr[N] = {};
CVulkanBuffer* s_skinCasterPages  = nullptr;   // device, kMaxSkinned * kSkinnedCap u32
CVulkanBuffer* s_skinIndirect     = nullptr;   // device, kMaxSkinned VkDrawIndexedIndirectCommand
CVulkanBuffer* s_skinStats        = nullptr;   // device, [0]=draws [1]=instances [2]=maxPages
CVulkanBuffer* s_skinStatsRB      = nullptr;   u32* s_skinStatsPtr = nullptr;
xr_vector<VsmSkinnedCaster> s_skinCasters;     // this frame's leaves (CPU; parallels meta/indirect order)
u32 s_skinCount = 0;

// std430 — matches SkinMeta in vsm_skinned_bin.comp.glsl (32 B).
struct SkinMetaGPU { Fvector sphere_P; float sphere_R; u32 index_count, ib_first, first_vertex, pad; };

// Grass (detail) casters into the atlas — near + L0 only, read from CDetailManager's
// GPU-driven instance buffer (1 frame stale; gated by r_vsm_grass). All grass types share
// one vertex layout → a single page pipeline.
VkPipeline            s_grassBinPipe    = VK_NULL_HANDLE;
VkPipelineLayout      s_grassBinLayout  = VK_NULL_HANDLE;
VkDescriptorSetLayout s_grassBinSetL    = VK_NULL_HANDLE;
VkDescriptorPool      s_grassBinPool    = VK_NULL_HANDLE;
VkDescriptorSet       s_grassBinSet[N]  = {};
VkPipeline            s_grassPagePipe   = VK_NULL_HANDLE;   // lazy (needs the grass mesh vertex stride)
VkPipelineLayout      s_grassPageLayout = VK_NULL_HANDLE;
VkDescriptorSetLayout s_grassPageSetL   = VK_NULL_HANDLE;
VkDescriptorPool      s_grassPagePool   = VK_NULL_HANDLE;
VkDescriptorSet       s_grassPageSet[N] = {};
VkShaderModule        s_grassPageVS     = VK_NULL_HANDLE;
VkShaderModule        s_grassPageFS     = VK_NULL_HANDLE;   // alpha-test (blade cutout)
// STATIC-cache hybrid half (r_vsm_grass_static): far grass (L1/L2 pairs) rendered
// RIGID into DIRTY static-atlas pages — cached until sun motion / window scroll
// dirties them (≈ free while standing), same near/far idea as the tree wind hybrid
// but PAGE-LEVEL based: the L0/L1 boundary is the near/dyn boundary, no per-instance
// flags or invalidation circles needed (receivers read the finest resident level).
VkPipeline            s_grassPagePipeS   = VK_NULL_HANDLE;  // static-atlas variant (_S grid consts, no wind)
VkShaderModule        s_grassPageVSS     = VK_NULL_HANDLE;
VkDescriptorSet       s_grassBinSetS[N]  = {};
VkDescriptorSet       s_grassPageSetS[N] = {};
// Compact (instance, page-slot) PAIR arena: the bin appends one u32 per overlapped
// accepted page (slot(13) << 19 | instLocal(19)) into a per-type section; the page
// pass then draws EXACTLY pairCount[type] instances via indirect — no fixed
// per-instance slot stride, no dead UNMAPPED writes over the full 1.5M-capacity
// buffer, no multi-pass VS re-walk of every instance (the old 9/12-pass scheme).
constexpr u32 kGrassPairCap   = 2097152;   // DYN arena, total pairs (×4 B = 8 MB; was 72 MB of slot strides).
                                           // Sized generously: the per-TYPE section is cap/types, and the
                                           // dominant grass type within r_vsm_grass_dist can hit tens of
                                           // thousands of instances × ~2-4 pages — a 512k cap truncated it
                                           // (stable "some tufts cast no shadow"). stats[1] counts drops.
constexpr u32 kGrassPairCapS  = 1048576;   // STATIC arena (4 MB): dirty pages only — small steady-state,
                                           // bounded by full-window redraws (scroll bursts / r_vsm_cache 0).
constexpr u32 kGrassMaxTypes  = 32;        // indirect/counter array bound (detail types are ≤ ~16)
CVulkanBuffer* s_grassPairs    = nullptr;  // device, kGrassPairCap u32 (per-type sections)
CVulkanBuffer* s_grassPairCnt  = nullptr;  // device, kGrassMaxTypes u32 append counters
CVulkanBuffer* s_grassPairInd  = nullptr;  // device, kGrassMaxTypes VkDrawIndexedIndirectCommand
CVulkanBuffer* s_grassPairsS   = nullptr;  // static-arena mirrors of the three above
CVulkanBuffer* s_grassPairCntS = nullptr;
CVulkanBuffer* s_grassPairIndS = nullptr;
CVulkanBuffer* s_grassStats   = nullptr;   // device [0]=casting instances
CVulkanBuffer* s_grassStatsRB = nullptr;   u32* s_grassStatsPtr = nullptr;
CVulkanBuffer* s_grassPairCntRB = nullptr; u32* s_grassPairCntPtr = nullptr;   // per-type pair counts (diag)
u32 s_grassSection = 0, s_grassTypes = 0;  // captured by the bin for the same-frame render
u32 s_grassPairSection  = 0;               // per-type DYN pair-arena capacity this frame
u32 s_grassPairSectionS = 0;               // per-type STATIC pair-arena capacity this frame

struct GrassBinPush { u32 sectionSize, typeCount, pairSection, mode; float camRange[4]; };   // mode: 0 dyn L0 | 1 static L1..L2 | 2 dyn L0..L2; xyz cam, w dist

// Grass-page raster push: SSFX wind (same values as the colour pass, 1 frame
// stale — the cast shadow sways with the visible blade) + the type's VisibleSSBO
// section base + its pair-arena section base (the VS pulls the transform rows
// from the SSBO by the pair's local index). wind_params.w = per-type DO_NO_WAVING scale.
struct GrassPagePush { Fvector4 wind_params, wsetup_grass, wind_anim; u32 instanceBase, pairBase; };

// std140 — matches VsmParams in vsm_mark.comp.glsl (176 B).
struct VsmParams {
    Fmatrix view;             // world -> sun light space
    float   level[kLevels][4]; // xy = level origin (light XY of texel 0,0), z = extent (m)
    float   zparams[4];        // x = zNear, y = 1/(zFar-zNear)
};

struct MarkPush {
    Fmatrix invViewProj;       // clip -> world
    float   screen[4];         // xy = dims, zw = 1/dims
    u32     markStep;          // 1 = full-res, 2 = half-res
    u32     lodBias;           // throttle: mark N clipmap levels coarser (r_vsm_throttle)
    u32     rmaskOn;           // receiver mask: mark sampled 8×8 cells (r_vsm_rmask)
    u32     gazeOn;            // gaze refresh: count samples per page into pageHits (r_vsm_gaze)
};

// std140 — matches the Resolve UBO in vsm_resolve.comp.glsl (240 B).
struct ResolveParams {
    Fmatrix invViewProj;       // current clip -> world
    Fmatrix prevViewProj;      // world -> previous-frame clip (history reproject)
    float   prevCamPos[4];     // xyz = previous frame camera
    float   curCamPos[4];      // xyz = this frame camera (stored as G for next frame); w = dyn-pixel EMA alpha (r_vsm_ta_blend_dyn)
    float   screen[4];         // xy = dims, zw = 1/dims
    float   params[4];         // x = alpha, y = reject tol, z = historyValid, w = dyn-gate
    float   params2[4];        // x = clamp tol, y = motion ref px, z = motion-floor weight, w = static bias const
    float   params3[4];        // SOFT: x = filter taps (0 = legacy 3x3 PCF), y = search taps, z = tan(sun half-angle), w = max blocker search dist (m)
    float   params4[4];        // x = frame noise phase, yzw = reserved
};

void MemBarrier(VkCommandBuffer cmd, VkAccessFlags src, VkAccessFlags dst,
                VkPipelineStageFlags ss, VkPipelineStageFlags ds)
{
    VkMemoryBarrier b{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    b.srcAccessMask = src; b.dstAccessMask = dst;
    vkCmdPipelineBarrier(cmd, ss, ds, 0, 1, &b, 0, nullptr, 0, nullptr);
}

bool CreatePipeline()
{
    if (!g_ShaderManager) g_ShaderManager = xr_new<VK::CVulkanSPIRVLoader>();
    VkShaderModule cs = g_ShaderManager->Load("vsm_mark.comp.spv");
    if (!cs) { Msg("![VK VSM] vsm_mark.comp.spv load failed"); return false; }

    VkDescriptorSetLayoutBinding b[6]{};
    b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b[1].binding = 1; b[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    b[2].binding = 2; b[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[3].binding = 3; b[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[4].binding = 4; b[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;   // receiver mask (r_vsm_rmask)
    b[5].binding = 5; b[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;   // page hits (r_vsm_gaze)
    for (u32 i = 0; i < 6; ++i) { b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = 6; lci.pBindings = b;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &s_setL) != VK_SUCCESS) return false;

    VkDescriptorPoolSize ps[3] = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, N },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         N },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         N * 4 },
    };
    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets = N; pci.poolSizeCount = 3; pci.pPoolSizes = ps;
    if (vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &s_pool) != VK_SUCCESS) return false;
    VkDescriptorSetLayout layouts[N]; for (u32 i = 0; i < N; ++i) layouts[i] = s_setL;
    VkDescriptorSetAllocateInfo dai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dai.descriptorPool = s_pool; dai.descriptorSetCount = N; dai.pSetLayouts = layouts;
    if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, s_set) != VK_SUCCESS) return false;

    VkSamplerCreateInfo si{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    si.magFilter = si.minFilter = VK_FILTER_NEAREST;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(VulkanHW.m_Device, &si, nullptr, &s_depthSampler) != VK_SUCCESS) return false;

    VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MarkPush) };
    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1; plci.pSetLayouts = &s_setL;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_layout) != VK_SUCCESS) return false;

    VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = cs;
    cpci.stage.pName  = "main";
    cpci.layout       = s_layout;
    if (vkCreateComputePipelines(VulkanHW.m_Device, VK_NULL_HANDLE, 1, &cpci, nullptr, &s_pipe) != VK_SUCCESS) return false;
    return true;
}

// Allocation pipeline: 4 SSBOs (needed, pageTable, pageList, allocInfo). Serves the
// DYNAMIC atlas only (the static atlas is mapped by the residency pass). The set is
// fixed-buffer so it's written once in Init, not per frame.
bool CreateAllocPipeline()
{
    VkShaderModule cs = g_ShaderManager->Load("vsm_alloc.comp.spv");
    if (!cs) { Msg("![VK VSM] vsm_alloc.comp.spv load failed"); return false; }

    VkDescriptorSetLayoutBinding b[4]{};
    for (u32 i = 0; i < 4; ++i) { b[i].binding = i; b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = 4; lci.pBindings = b;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &s_allocSetL) != VK_SUCCESS) return false;

    VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 };
    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets = 1; pci.poolSizeCount = 1; pci.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &s_allocPool) != VK_SUCCESS) return false;
    VkDescriptorSetAllocateInfo dai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dai.descriptorPool = s_allocPool; dai.descriptorSetCount = 1; dai.pSetLayouts = &s_allocSetL;
    if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, &s_dynAllocSet) != VK_SUCCESS) return false;

    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1; plci.pSetLayouts = &s_allocSetL;
    if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_allocLayout) != VK_SUCCESS) return false;

    VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = cs;
    cpci.stage.pName  = "main";
    cpci.layout       = s_allocLayout;
    if (vkCreateComputePipelines(VulkanHW.m_Device, VK_NULL_HANDLE, 1, &cpci, nullptr, &s_allocPipe) != VK_SUCCESS) return false;
    return true;
}

// Binning pipeline: 5 SSBOs (meta, pageTable, count, casters, stats) + 1 UBO. Per-frame
// sets (the UBO + ShadowGPU's meta handle vary), updated in MarkPages.
bool CreateBinPipeline()
{
    VkShaderModule cs = g_ShaderManager->Load("vsm_bin.comp.spv");
    if (!cs) { Msg("![VK VSM] vsm_bin.comp.spv load failed"); return false; }

    VkDescriptorSetLayoutBinding b[8]{};
    const VkDescriptorType types[8] = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,   // meta, ubo
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,   // pageTable, casterPages
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,   // indirect, groupCount
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,   // stats, slotDirty (Phase 1b)
    };
    for (u32 i = 0; i < 8; ++i) { b[i].binding = i; b[i].descriptorType = types[i]; b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = 8; lci.pBindings = b;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &s_binSetL) != VK_SUCCESS) return false;

    VkDescriptorPoolSize ps[2] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, N * 7 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, N },
    };
    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets = N; pci.poolSizeCount = 2; pci.pPoolSizes = ps;
    if (vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &s_binPool) != VK_SUCCESS) return false;
    VkDescriptorSetLayout layouts[N]; for (u32 i = 0; i < N; ++i) layouts[i] = s_binSetL;
    VkDescriptorSetAllocateInfo dai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dai.descriptorPool = s_binPool; dai.descriptorSetCount = N; dai.pSetLayouts = layouts;
    if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, s_binSet) != VK_SUCCESS) return false;

    VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, 2 * sizeof(u32) + 4 * sizeof(float) };   // casterCount, groupStride, camXYZ, lodDist
    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1; plci.pSetLayouts = &s_binSetL; plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_binLayout) != VK_SUCCESS) return false;

    VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = cs;
    cpci.stage.pName  = "main";
    cpci.layout       = s_binLayout;
    if (vkCreateComputePipelines(VulkanHW.m_Device, VK_NULL_HANDLE, 1, &cpci, nullptr, &s_binPipe) != VK_SUCCESS) return false;
    return true;
}

// Phase 3 cluster-bin pipeline (vsm_bin_cluster.comp): same shape as the bin
// above + binding 8 = per-combo cmd-region bases, 11 = the candidate list.
// Per-frame sets (UBO + the WorldGPU buffer handles), updated in MarkPages.
// Best-effort: failure leaves s_binClPipe null → the per-mesh bin keeps running.
bool CreateBinClusterPipeline()
{
    VkShaderModule cs = g_ShaderManager->Load("vsm_bin_cluster.comp.spv");
    if (!cs) { Msg("![VK VSM] vsm_bin_cluster.comp.spv load failed"); return false; }

    // 0-8 as before; Stage B: 9 = cluster-stream bits, 10 = page slot bases;
    // at-scale: 11 = candidate list ((combo<<20)|entry).
    VkDescriptorSetLayoutBinding b[12]{};
    for (u32 i = 0; i < 12; ++i) {
        b[i].binding = i;
        b[i].descriptorType = (i == 1) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = 12; lci.pBindings = b;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &s_binClSetL) != VK_SUCCESS) return false;

    VkDescriptorPoolSize ps[2] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, N * 11 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, N },
    };
    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets = N; pci.poolSizeCount = 2; pci.pPoolSizes = ps;
    if (vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &s_binClPool) != VK_SUCCESS) return false;
    VkDescriptorSetLayout layouts[N]; for (u32 i = 0; i < N; ++i) layouts[i] = s_binClSetL;
    VkDescriptorSetAllocateInfo dai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dai.descriptorPool = s_binClPool; dai.descriptorSetCount = N; dai.pSetLayouts = layouts;
    if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, s_binClSet) != VK_SUCCESS) return false;

    VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, 2 * sizeof(u32) + sizeof(float) };   // candCount, arenaSlots, errK
    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1; plci.pSetLayouts = &s_binClSetL; plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_binClLayout) != VK_SUCCESS) return false;

    VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = cs;
    cpci.stage.pName  = "main";
    cpci.layout       = s_binClLayout;
    if (vkCreateComputePipelines(VulkanHW.m_Device, VK_NULL_HANDLE, 1, &cpci, nullptr, &s_binClPipe) != VK_SUCCESS) return false;
    return true;
}

// Toroidal residency compute: 7 SSBOs (needed, pageTable, pageList, physTile, slotDirty,
// dirtyList, drawClear). Static set (all device buffers fixed) — written once in Init.
bool CreateResidPipeline()
{
    VkShaderModule cs = g_ShaderManager->Load("vsm_resid.comp.spv");
    if (!cs) { Msg("![VK VSM] vsm_resid.comp.spv load failed"); return false; }
    VkDescriptorSetLayoutBinding b[9]{};   // +7 priorValid (shadow-HZB), +8 pageHits (r_vsm_gaze)
    for (u32 i = 0; i < 9; ++i) { b[i].binding = i; b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = 9; lci.pBindings = b;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &s_residSetL) != VK_SUCCESS) return false;
    VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 9 };
    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets = 1; pci.poolSizeCount = 1; pci.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &s_residPool) != VK_SUCCESS) return false;
    VkDescriptorSetAllocateInfo dai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dai.descriptorPool = s_residPool; dai.descriptorSetCount = 1; dai.pSetLayouts = &s_residSetL;
    if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, &s_residSet) != VK_SUCCESS) return false;
    VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ResidPush) };
    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1; plci.pSetLayouts = &s_residSetL; plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_residLayout) != VK_SUCCESS) return false;
    VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = cs; cpci.stage.pName = "main"; cpci.layout = s_residLayout;
    if (vkCreateComputePipelines(VulkanHW.m_Device, VK_NULL_HANDLE, 1, &cpci, nullptr, &s_residPipe) != VK_SUCCESS) return false;
    return true;
}

// shadow-HZB reduce pipeline (r_vsm_hzb): set = {atlas sampler, prevDirty, priorValid, pageMax, pageMaxBlk}.
// Call AFTER CreateRenderResources (needs s_atlasView/s_atlasSampler) and after the resid buffers.
// v1 verdict (2026-07-02, single whole-page max + tree×page test only): net −0.15ms — the full-residency
// reduce cost ate the gain. v2 (2026-07-13, UE-parity rework): INCREMENTAL reduce (only last frame's dirty
// slots — cost tracks the sun's dirty rate, not residency) + 8×8 BLOCK maxes consumed at BRICK granularity
// by vsm_vox_cull — a canopy gap no longer defeats the whole page. Re-chase target: dense forest.
bool CreateHzbReducePipeline()
{
    VkShaderModule cs = g_ShaderManager->Load("vsm_hzb_reduce.comp.spv");
    if (!cs) { Msg("![VK VSM] vsm_hzb_reduce.comp.spv load failed - shadow-HZB disabled"); return false; }
    VkDescriptorSetLayoutBinding b[5]{};
    b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; b[0].descriptorCount = 1; b[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    for (u32 i = 1; i < 5; ++i) { b[i].binding = i; b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = 5; lci.pBindings = b;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &s_hzbSetL) != VK_SUCCESS) return false;
    VkDescriptorPoolSize ps[2] = { { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 }, { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 } };
    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets = 1; pci.poolSizeCount = 2; pci.pPoolSizes = ps;
    if (vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &s_hzbPool) != VK_SUCCESS) return false;
    VkDescriptorSetAllocateInfo dai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dai.descriptorPool = s_hzbPool; dai.descriptorSetCount = 1; dai.pSetLayouts = &s_hzbSetL;
    if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, &s_hzbSet) != VK_SUCCESS) return false;
    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1; plci.pSetLayouts = &s_hzbSetL;
    if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_hzbLayout) != VK_SUCCESS) return false;
    VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = cs; cpci.stage.pName = "main"; cpci.layout = s_hzbLayout;
    if (vkCreateComputePipelines(VulkanHW.m_Device, VK_NULL_HANDLE, 1, &cpci, nullptr, &s_hzbPipe) != VK_SUCCESS) return false;

    VkDescriptorImageInfo ii{ s_atlasSampler, s_atlasView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkDescriptorBufferInfo bufi[4] = {
        { s_slotDirtyPrev->GetHandle(), 0, VK_WHOLE_SIZE },   // binding 1 = LAST frame's dirty set
        { s_priorValid->GetHandle(),    0, VK_WHOLE_SIZE },
        { s_pageMax->GetHandle(),       0, VK_WHOLE_SIZE },
        { s_pageMaxBlk->GetHandle(),    0, VK_WHOLE_SIZE },
    };
    VkWriteDescriptorSet w[5]{};
    w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[0].dstSet = s_hzbSet; w[0].dstBinding = 0; w[0].descriptorCount = 1; w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[0].pImageInfo = &ii;
    for (u32 i = 0; i < 4; ++i) { w[i + 1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i + 1].dstSet = s_hzbSet; w[i + 1].dstBinding = i + 1; w[i + 1].descriptorCount = 1; w[i + 1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i + 1].pBufferInfo = &bufi[i]; }
    vkUpdateDescriptorSets(VulkanHW.m_Device, 5, w, 0, nullptr);
    return true;
}

// Clear-dirty graphics pipeline: depth-only, no vertex input, no fragment — one instanced quad
// per dirty slot writes depth 1.0 into the static atlas (depthCompareOp ALWAYS over loadOp LOAD).
bool CreateClearPipeline()
{
    s_clearVS = g_ShaderManager->Load("vsm_clear.vert.spv");
    if (!s_clearVS) { Msg("![VK VSM] vsm_clear.vert.spv load failed"); return false; }
    VkDescriptorSetLayoutBinding b{}; b.binding = 0; b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; b.descriptorCount = 1; b.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = 1; lci.pBindings = &b;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &s_clearSetL) != VK_SUCCESS) return false;
    VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets = 1; pci.poolSizeCount = 1; pci.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &s_clearPool) != VK_SUCCESS) return false;
    VkDescriptorSetAllocateInfo dai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dai.descriptorPool = s_clearPool; dai.descriptorSetCount = 1; dai.pSetLayouts = &s_clearSetL;
    if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, &s_clearSet) != VK_SUCCESS) return false;
    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1; plci.pSetLayouts = &s_clearSetL;
    if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_clearLayout) != VK_SUCCESS) return false;

    VkPipelineVertexInputStateCreateInfo vi{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    VkPipelineShaderStageCreateInfo stage{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    stage.stage = VK_SHADER_STAGE_VERTEX_BIT; stage.module = s_clearVS; stage.pName = "main";
    VkPipelineInputAssemblyStateCreateInfo ia{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    vp.viewportCount = 1; vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo ds{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_TRUE; ds.depthCompareOp = VK_COMPARE_OP_ALWAYS;
    VkPipelineColorBlendStateCreateInfo cb{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    cb.attachmentCount = 0;
    // DEPTH_BIAS is declared dynamic even though this pipeline biases nothing
    // (depthBiasEnable stays FALSE). Binding a pipeline APPLIES every state it does
    // NOT declare dynamic, so a clear pipeline carrying a static zero bias wipes the
    // vkCmdSetDepthBias that beginAtlas issued three lines earlier — and every caster
    // drawn after it rasterizes into the static atlas with NO write-side bias. That
    // was the terrain shadow-acne banding (25-07): the receiver constant
    // r_vsm_bias_min is sized on the assumption that this write bias exists, so
    // losing it silently disables BOTH halves of the anti-acne scheme at once.
    // Same class as the VRS dynamic-state loss ([[vulkan-vrs-saga]], VUID-07834).
    VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS };
    VkPipelineDynamicStateCreateInfo dynState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dynState.dynamicStateCount = 3; dynState.pDynamicStates = dyn;
    VkPipelineRenderingCreateInfo prci{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    prci.colorAttachmentCount = 0; prci.depthAttachmentFormat = VK_FORMAT_D16_UNORM;   // D16 atlas (see CreateRenderResources)
    VkGraphicsPipelineCreateInfo pi{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    pi.pNext = &prci; pi.stageCount = 1; pi.pStages = &stage;
    pi.pVertexInputState = &vi; pi.pInputAssemblyState = &ia; pi.pViewportState = &vp;
    pi.pRasterizationState = &rs; pi.pMultisampleState = &ms; pi.pDepthStencilState = &ds;
    pi.pColorBlendState = &cb; pi.pDynamicState = &dynState; pi.layout = s_clearLayout;
    if (vkCreateGraphicsPipelines(VulkanHW.m_Device, VK_NULL_HANDLE, 1, &pi, nullptr, &s_clearPipe) != VK_SUCCESS) return false;
    return true;
}

// Atlas image + sampler + the page render descriptor/layout + vsm_page.vert (per-stride
// pipelines are created lazily in GetPagePipeline).
bool CreateRenderResources()
{
    VK::Vram::Scope _vram_scope("VSM");
    VkImageCreateInfo ici{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ici.imageType = VK_IMAGE_TYPE_2D;
    // D16 (not D32): the sun VSM is ORTHOGRAPHIC → depth is LINEAR, so D16 quantizes
    // uniformly (~3 cm over the clipmap range, drowned by the receiver bias) instead
    // of crunching at the near plane like a perspective map would. Halves the atlas
    // fill depth-write bandwidth, the resolve/receiver/VolInject 3x3 PCF read
    // bandwidth, AND the memory (static 402->201 MB + dyn 134->67 = 536->268 MB) —
    // the last removes the "402 MB = weak-HW" block against defaulting r_vsm on.
    // (Spot/point maps stay D32: perspective near-crunch makes D16 acne there.)
    ici.format    = VK_FORMAT_D16_UNORM;
    ici.extent    = { kAtlasW_S * kPageSize, kAtlasH_S * kPageSize, 1 };   // STATIC: 8192 x 12288 (6144 toroidal pages)
    ici.mipLevels = 1; ici.arrayLayers = 1; ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling    = VK_IMAGE_TILING_OPTIMAL;
    ici.usage     = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo aci{}; aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.priority = 1.0f;   // VSM shadow atlas — hot, never evict before streamable textures
    if (VK::Vram::CreateImage(VulkanHW.m_Allocator, &ici, &aci, &s_atlasImage, &s_atlasAlloc, nullptr) != VK_SUCCESS) {
        Msg("![VK VSM] atlas image create failed"); return false;
    }
    Prof::NameImage(s_atlasImage, "VSM.Atlas");
    VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vci.image = s_atlasImage; vci.viewType = VK_IMAGE_VIEW_TYPE_2D; vci.format = VK_FORMAT_D16_UNORM;   // matches the D16 atlas
    vci.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
    if (vkCreateImageView(VulkanHW.m_Device, &vci, nullptr, &s_atlasView) != VK_SUCCESS) return false;

    // DYNAMIC atlas — same format/usage, the smaller 2048-page grid (8192 x 4096).
    ici.extent = { kAtlasW * kPageSize, kAtlasH * kPageSize, 1 };
    if (VK::Vram::CreateImage(VulkanHW.m_Allocator, &ici, &aci, &s_dynImage, &s_dynAlloc, nullptr) != VK_SUCCESS) {
        Msg("![VK VSM] dynamic atlas image create failed"); return false;
    }
    Prof::NameImage(s_dynImage, "VSM.AtlasDyn");
    vci.image = s_dynImage;
    if (vkCreateImageView(VulkanHW.m_Device, &vci, nullptr, &s_dynView) != VK_SUCCESS) return false;

    VkSamplerCreateInfo sci{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    sci.magFilter = sci.minFilter = VK_FILTER_NEAREST;   // receiver does explicit 3x3 PCF (compare-then-average)
    sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    sci.borderColor  = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;   // outside any page → depth 1 → lit
    sci.maxLod = 1.0f;
    if (vkCreateSampler(VulkanHW.m_Device, &sci, nullptr, &s_atlasSampler) != VK_SUCCESS) return false;

    // Render set: pageList(SSBO) + casterPages(SSBO) + VSM UBO, all VERTEX stage.
    VkDescriptorSetLayoutBinding rb[3]{};
    const VkDescriptorType rtypes[3] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER };
    for (u32 i = 0; i < 3; ++i) { rb[i].binding = i; rb[i].descriptorType = rtypes[i]; rb[i].descriptorCount = 1; rb[i].stageFlags = VK_SHADER_STAGE_VERTEX_BIT; }
    VkDescriptorSetLayoutCreateInfo rlci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    rlci.bindingCount = 3; rlci.pBindings = rb;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &rlci, nullptr, &s_renderSetL) != VK_SUCCESS) return false;
    VkDescriptorPoolSize rps[2] = { { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, N * 2 }, { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, N } };
    VkDescriptorPoolCreateInfo rpci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    rpci.maxSets = N; rpci.poolSizeCount = 2; rpci.pPoolSizes = rps;
    if (vkCreateDescriptorPool(VulkanHW.m_Device, &rpci, nullptr, &s_renderPool) != VK_SUCCESS) return false;
    VkDescriptorSetLayout rl[N]; for (u32 i = 0; i < N; ++i) rl[i] = s_renderSetL;
    VkDescriptorSetAllocateInfo rdai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    rdai.descriptorPool = s_renderPool; rdai.descriptorSetCount = N; rdai.pSetLayouts = rl;
    if (vkAllocateDescriptorSets(VulkanHW.m_Device, &rdai, s_renderSet) != VK_SUCCESS) return false;

    VkPipelineLayoutCreateInfo rplci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    rplci.setLayoutCount = 1; rplci.pSetLayouts = &s_renderSetL;
    if (vkCreatePipelineLayout(VulkanHW.m_Device, &rplci, nullptr, &s_renderLayout) != VK_SUCCESS) return false;

    s_pageVS = g_ShaderManager->Load("vsm_page.vert.spv");
    if (!s_pageVS) { Msg("![VK VSM] vsm_page.vert.spv load failed"); return false; }
    return true;
}

// Per-stride depth-only page pipeline (mirrors the cascade depth pipeline, but with
// vsm_page.vert routing + the render layout; gl_ClipDistance enabled via the shader).
VkPipeline GetPagePipeline(u32 stride)
{
    auto it = s_pagePipes.find(stride);
    if (it != s_pagePipes.end()) return it->second;

    VkVertexInputBindingDescription binding{ 0, stride, VK_VERTEX_INPUT_RATE_VERTEX };
    VkVertexInputAttributeDescription attr{ 0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0 };   // pos @ offset 0, all strides
    VkPipelineVertexInputStateCreateInfo vi{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &binding;
    vi.vertexAttributeDescriptionCount = 1; vi.pVertexAttributeDescriptions = &attr;
    VkPipelineShaderStageCreateInfo stage{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    stage.stage = VK_SHADER_STAGE_VERTEX_BIT; stage.module = s_pageVS; stage.pName = "main";
    VkPipelineInputAssemblyStateCreateInfo ia{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    vp.viewportCount = 1; vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;
    rs.depthBiasEnable = VK_TRUE;
    VkPipelineMultisampleStateCreateInfo ms{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo ds{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_TRUE; ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    VkPipelineColorBlendStateCreateInfo cb{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    cb.attachmentCount = 0;
    VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS };
    VkPipelineDynamicStateCreateInfo dynState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dynState.dynamicStateCount = 3; dynState.pDynamicStates = dyn;
    VkPipelineRenderingCreateInfo prci{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    prci.colorAttachmentCount = 0; prci.depthAttachmentFormat = VK_FORMAT_D16_UNORM;   // D16 atlas (see CreateRenderResources)
    VkGraphicsPipelineCreateInfo pi{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    pi.pNext = &prci; pi.stageCount = 1; pi.pStages = &stage;
    pi.pVertexInputState = &vi; pi.pInputAssemblyState = &ia; pi.pViewportState = &vp;
    pi.pRasterizationState = &rs; pi.pMultisampleState = &ms; pi.pDepthStencilState = &ds;
    pi.pColorBlendState = &cb; pi.pDynamicState = &dynState; pi.layout = s_renderLayout;
    VkPipeline h = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(VulkanHW.m_Device, VK_NULL_HANDLE, 1, &pi, nullptr, &h) != VK_SUCCESS)
        Msg("![VK VSM] page pipeline create failed (stride=%u)", stride);
    s_pagePipes.emplace(stride, h);
    return h;
}

// r_vsm_at: 16 B push shared by vsm_page_at.vert (uvScale) and .frag (aref).
struct PageATPush { float uvScale[2]; float aref; float _pad; };

// Lazy AT layout/modules: the material cache's set layout only exists once the
// level's materials are built, which is after VSM init — create on first draw.
bool EnsurePageATResources()
{
    if (s_renderATLayout != VK_NULL_HANDLE) return true;
    if (s_pageATFailed) return false;
    VkDescriptorSetLayout matL = WorldMaterialCache::GetSetLayout();
    if (matL == VK_NULL_HANDLE || s_renderSetL == VK_NULL_HANDLE) return false;   // not ready yet — retry next draw
    s_pageATVS = g_ShaderManager->Load("vsm_page_at.vert.spv");
    s_pageATFS = g_ShaderManager->Load("vsm_page_at.frag.spv");
    if (!s_pageATVS || !s_pageATFS) {
        Msg("![VK VSM] vsm_page_at spv load failed - AT casters stay out of the atlas");
        s_pageATFailed = true;   // definitive (missing file won't appear mid-session)
        return false;
    }
    VkDescriptorSetLayout sets[2] = { s_renderSetL, matL };
    VkPushConstantRange pcr{ VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PageATPush) };
    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 2; plci.pSetLayouts = sets;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_renderATLayout) != VK_SUCCESS) {
        Msg("![VK VSM] AT page layout create failed");
        s_pageATFailed = true;
        return false;
    }
    return true;
}

// Alpha-tested page pipeline: GetPagePipeline + the UV attribute (SHORT2
// SSCALED @ tcOffset, same quantization as every static depth-AT path) + the
// discard fragment stage sampling the material's diffuse (set 1).
VkPipeline GetPageATPipeline(u32 stride, u32 tcOffset)
{
    const u32 key = (stride << 8) | (tcOffset & 0xFFu);
    auto it = s_pageATPipes.find(key);
    if (it != s_pageATPipes.end()) return it->second;
    if (!EnsurePageATResources()) return VK_NULL_HANDLE;

    VkVertexInputBindingDescription binding{ 0, stride, VK_VERTEX_INPUT_RATE_VERTEX };
    VkVertexInputAttributeDescription attrs[2] = {
        { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0 },
        { 1, 0, VK_FORMAT_R16G16_SSCALED, tcOffset },
    };
    VkPipelineVertexInputStateCreateInfo vi{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &binding;
    vi.vertexAttributeDescriptionCount = 2; vi.pVertexAttributeDescriptions = attrs;
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;   stages[0].module = s_pageATVS; stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = s_pageATFS; stages[1].pName = "main";
    VkPipelineInputAssemblyStateCreateInfo ia{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    vp.viewportCount = 1; vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;
    rs.depthBiasEnable = VK_TRUE;
    VkPipelineMultisampleStateCreateInfo ms{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo ds{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_TRUE; ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    VkPipelineColorBlendStateCreateInfo cb{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    cb.attachmentCount = 0;
    VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS };
    VkPipelineDynamicStateCreateInfo dynState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dynState.dynamicStateCount = 3; dynState.pDynamicStates = dyn;
    VkPipelineRenderingCreateInfo prci{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    prci.colorAttachmentCount = 0; prci.depthAttachmentFormat = VK_FORMAT_D16_UNORM;   // D16 atlas
    VkGraphicsPipelineCreateInfo pi{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    pi.pNext = &prci; pi.stageCount = 2; pi.pStages = stages;
    pi.pVertexInputState = &vi; pi.pInputAssemblyState = &ia; pi.pViewportState = &vp;
    pi.pRasterizationState = &rs; pi.pMultisampleState = &ms; pi.pDepthStencilState = &ds;
    pi.pColorBlendState = &cb; pi.pDynamicState = &dynState; pi.layout = s_renderATLayout;
    VkPipeline h = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(VulkanHW.m_Device, VK_NULL_HANDLE, 1, &pi, nullptr, &h) != VK_SUCCESS)
        Msg("![VK VSM] AT page pipeline create failed (stride=%u tc=%u)", stride, tcOffset);
    s_pageATPipes.emplace(key, h);
    return h;
}

// Temporal resolve compute: 7 bindings (depth, atlas, pageTable, clipmap UBO,
// history, output mask, resolve UBO). Per-frame sets (most handles vary).
bool CreateResolvePipeline()
{
    VkShaderModule cs = g_ShaderManager->Load("vsm_resolve.comp.spv");
    if (!cs) { Msg("![VK VSM] vsm_resolve.comp.spv load failed"); return false; }

    VkDescriptorSetLayoutBinding b[10]{};
    const VkDescriptorType types[10] = {
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,  // depth, static atlas
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,          // static pageTable, clipmap UBO
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,           // history, output mask
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,                                                     // resolve UBO
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,                                             // dynamic atlas
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,                                                     // dynamic pageTable
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,                                                     // dynUsed (skip empty dyn pages)
    };
    for (u32 i = 0; i < 10; ++i) { b[i].binding = i; b[i].descriptorType = types[i]; b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = 10; lci.pBindings = b;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &s_resolveSetL) != VK_SUCCESS) return false;

    VkDescriptorPoolSize ps[4] = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, N * 4 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         N * 3 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         N * 2 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          N },
    };
    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets = N; pci.poolSizeCount = 4; pci.pPoolSizes = ps;
    if (vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &s_resolvePool) != VK_SUCCESS) return false;
    VkDescriptorSetLayout layouts[N]; for (u32 i = 0; i < N; ++i) layouts[i] = s_resolveSetL;
    VkDescriptorSetAllocateInfo dai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dai.descriptorPool = s_resolvePool; dai.descriptorSetCount = N; dai.pSetLayouts = layouts;
    if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, s_resolveSet) != VK_SUCCESS) return false;

    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1; plci.pSetLayouts = &s_resolveSetL;
    if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_resolveLayout) != VK_SUCCESS) return false;

    VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = cs;
    cpci.stage.pName  = "main";
    cpci.layout       = s_resolveLayout;
    if (vkCreateComputePipelines(VulkanHW.m_Device, VK_NULL_HANDLE, 1, &cpci, nullptr, &s_resolvePipe) != VK_SUCCESS) return false;

    // History/receiver sampler: LINEAR clamp-to-edge (gentle 1:1 read; edge clamp on reproject).
    VkSamplerCreateInfo si{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(VulkanHW.m_Device, &si, nullptr, &s_maskSampler) != VK_SUCCESS) return false;
    return true;
}

void DestroyMaskTargets()
{
    for (u32 i = 0; i < N; ++i) {
        if (s_maskView[i])  { vkDestroyImageView(VulkanHW.m_Device, s_maskView[i], nullptr); s_maskView[i] = VK_NULL_HANDLE; }
        if (s_maskImage[i]) { VK::Vram::DestroyImage(VulkanHW.m_Allocator, s_maskImage[i], s_maskAlloc[i]); s_maskImage[i] = VK_NULL_HANDLE; s_maskAlloc[i] = VK_NULL_HANDLE; }
        s_maskFirst[i] = true;
    }
    s_maskExtent = {}; s_maskValid = false; s_resolveCount = 0;
}

// Create (or resize) the N-deep screen-space mask. RGBA16F = the only mandatory
// storage format wide enough for (lit, dist) without shaderStorageImageExtendedFormats.
bool EnsureMaskTargets(VkExtent2D screen)
{
    VK::Vram::Scope _vram_scope("VSM");
    if (screen.width == 0 || screen.height == 0) return false;
    if (s_maskImage[0] && screen.width == s_maskExtent.width && screen.height == s_maskExtent.height)
        return true;
    DestroyMaskTargets();
    s_maskExtent = screen;
    for (u32 i = 0; i < N; ++i) {
        VkImageCreateInfo ici{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format    = VK_FORMAT_R16G16B16A16_SFLOAT;
        ici.extent    = { screen.width, screen.height, 1 };
        ici.mipLevels = 1; ici.arrayLayers = 1; ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling    = VK_IMAGE_TILING_OPTIMAL;
        ici.usage     = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VmaAllocationCreateInfo aci{}; aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        aci.priority = 1.0f;   // VSM receiver mask — hot, never evict before streamable textures
        if (VK::Vram::CreateImage(VulkanHW.m_Allocator, &ici, &aci, &s_maskImage[i], &s_maskAlloc[i], nullptr) != VK_SUCCESS) {
            Msg("![VK VSM] mask image %u create failed", i); DestroyMaskTargets(); return false;
        }
        VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        vci.image = s_maskImage[i]; vci.viewType = VK_IMAGE_VIEW_TYPE_2D; vci.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        if (vkCreateImageView(VulkanHW.m_Device, &vci, nullptr, &s_maskView[i]) != VK_SUCCESS) { DestroyMaskTargets(); return false; }
        Prof::NameImage(s_maskImage[i], "VSM.Mask");
    }
    return true;
}

// Skinned-caster bin pipeline: 7 bindings (meta, clipmap UBO, pageTable, casterPages,
// indirect, stats, dynUsed). Per-frame sets (meta + UBO vary).
bool CreateSkinnedBinPipeline()
{
    VkShaderModule cs = g_ShaderManager->Load("vsm_skinned_bin.comp.spv");
    if (!cs) { Msg("![VK VSM] vsm_skinned_bin.comp.spv load failed"); return false; }
    VkDescriptorSetLayoutBinding b[7]{};
    const VkDescriptorType t[7] = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    };
    for (u32 i = 0; i < 7; ++i) { b[i].binding = i; b[i].descriptorType = t[i]; b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = 7; lci.pBindings = b;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &s_skinBinSetL) != VK_SUCCESS) return false;
    VkDescriptorPoolSize ps[2] = { { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, N * 6 }, { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, N } };
    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets = N; pci.poolSizeCount = 2; pci.pPoolSizes = ps;
    if (vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &s_skinBinPool) != VK_SUCCESS) return false;
    VkDescriptorSetLayout ls[N]; for (u32 i = 0; i < N; ++i) ls[i] = s_skinBinSetL;
    VkDescriptorSetAllocateInfo dai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dai.descriptorPool = s_skinBinPool; dai.descriptorSetCount = N; dai.pSetLayouts = ls;
    if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, s_skinBinSet) != VK_SUCCESS) return false;
    VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, 2 * sizeof(u32) };
    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1; plci.pSetLayouts = &s_skinBinSetL; plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_skinBinLayout) != VK_SUCCESS) return false;
    VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = cs; cpci.stage.pName = "main"; cpci.layout = s_skinBinLayout;
    if (vkCreateComputePipelines(VulkanHW.m_Device, VK_NULL_HANDLE, 1, &cpci, nullptr, &s_skinBinPipe) != VK_SUCCESS) return false;
    return true;
}

// Skinned-page pipeline resources are LAZY: the layout needs Skinned's bone-set layout,
// which only exists once the skinned pass has initialised (after the first NPC frame).
bool EnsureSkinnedPageResources()
{
    if (s_skinPageLayout != VK_NULL_HANDLE) return true;
    VkDescriptorSetLayout boneL = VK::Skinned_GetBoneSetLayout();
    if (boneL == VK_NULL_HANDLE) return false;   // skinned not ready yet
    VkDescriptorSetLayoutBinding b[3]{};
    const VkDescriptorType t[3] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER };
    for (u32 i = 0; i < 3; ++i) { b[i].binding = i; b[i].descriptorType = t[i]; b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_VERTEX_BIT; }
    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = 3; lci.pBindings = b;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &s_skinPageSetL) != VK_SUCCESS) return false;
    VkDescriptorPoolSize ps[2] = { { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, N * 2 }, { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, N } };
    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets = N; pci.poolSizeCount = 2; pci.pPoolSizes = ps;
    if (vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &s_skinPagePool) != VK_SUCCESS) return false;
    VkDescriptorSetLayout ls[N]; for (u32 i = 0; i < N; ++i) ls[i] = s_skinPageSetL;
    VkDescriptorSetAllocateInfo dai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dai.descriptorPool = s_skinPagePool; dai.descriptorSetCount = N; dai.pSetLayouts = ls;
    if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, s_skinPageSet) != VK_SUCCESS) return false;
    VkDescriptorSetLayout sets[2] = { boneL, s_skinPageSetL };   // set0 = bones (shared), set1 = page data
    VkPushConstantRange pcr{ VK_SHADER_STAGE_VERTEX_BIT, 0, 4 * sizeof(u32) };   // skinMode, baseBone, boneCount, pad
    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 2; plci.pSetLayouts = sets; plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_skinPageLayout) != VK_SUCCESS) return false;
    return true;
}

VkPipeline GetSkinnedPagePipeline(u32 stride)
{
    auto it = s_skinPagePipes.find(stride);
    if (it != s_skinPagePipes.end()) return it->second;
    VkVertexInputBindingDescription binding{};
    VkVertexInputAttributeDescription attrs[6]{};
    VK::Skinned_BuildVertexInput(stride, binding, attrs);
    VkPipelineVertexInputStateCreateInfo vi{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &binding;
    vi.vertexAttributeDescriptionCount = 6; vi.pVertexAttributeDescriptions = attrs;
    VkPipelineShaderStageCreateInfo stage{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    stage.stage = VK_SHADER_STAGE_VERTEX_BIT; stage.module = s_skinPageVS; stage.pName = "main";
    VkPipelineInputAssemblyStateCreateInfo ia{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    vp.viewportCount = 1; vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;
    rs.depthBiasEnable = VK_TRUE;
    VkPipelineMultisampleStateCreateInfo ms{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo ds{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_TRUE; ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    VkPipelineColorBlendStateCreateInfo cb{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    cb.attachmentCount = 0;
    VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS };
    VkPipelineDynamicStateCreateInfo dynState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dynState.dynamicStateCount = 3; dynState.pDynamicStates = dyn;
    VkPipelineRenderingCreateInfo prci{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    prci.colorAttachmentCount = 0; prci.depthAttachmentFormat = VK_FORMAT_D16_UNORM;   // D16 atlas (see CreateRenderResources)
    VkGraphicsPipelineCreateInfo pi{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    pi.pNext = &prci; pi.stageCount = 1; pi.pStages = &stage;
    pi.pVertexInputState = &vi; pi.pInputAssemblyState = &ia; pi.pViewportState = &vp;
    pi.pRasterizationState = &rs; pi.pMultisampleState = &ms; pi.pDepthStencilState = &ds;
    pi.pColorBlendState = &cb; pi.pDynamicState = &dynState; pi.layout = s_skinPageLayout;
    VkPipeline h = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(VulkanHW.m_Device, VK_NULL_HANDLE, 1, &pi, nullptr, &h) != VK_SUCCESS)
        Msg("![VK VSM] skinned page pipeline create failed (stride=%u)", stride);
    s_skinPagePipes.emplace(stride, h);
    return h;
}

// CPU: collect this frame's visible NPC leaves + upload their bin meta. Parallels the
// meta/indirect order so the render loop's leaf index c matches indirect[c].
void CollectSkinned(u32 cur)
{
    s_skinCasters.clear();
    s_skinCount = 0;
    if (s_skinPageVS == VK_NULL_HANDLE) return;   // no skinned-page shader → no NPC casting
    VK::Skinned_CollectCasters(s_skinCasters);
    u32 n = (u32)s_skinCasters.size();
    if (n > kMaxSkinned) n = kMaxSkinned;
    s_skinCount = n;
    if (!s_skinMetaPtr[cur] || n == 0) return;
    SkinMetaGPU* dst = (SkinMetaGPU*)s_skinMetaPtr[cur];
    for (u32 c = 0; c < n; ++c) {
        const VsmSkinnedCaster& s = s_skinCasters[c];
        dst[c].sphere_P = s.sphere_P; dst[c].sphere_R = s.sphere_R;
        dst[c].index_count = s.index_count; dst[c].ib_first = s.ib_first;
        dst[c].first_vertex = (u32)s.first_vertex; dst[c].pad = 0u;
    }
}

void DispatchSkinnedBin(VkCommandBuffer cmd, u32 cur)
{
    if (s_skinCount == 0 || s_skinBinPipe == VK_NULL_HANDLE) return;
    VkDescriptorBufferInfo bi[7] = {
        { s_skinMeta[cur]->GetHandle(),   0, VK_WHOLE_SIZE },
        { s_ubo[cur]->GetHandle(),        0, VK_WHOLE_SIZE },
        { s_dynPageTable->GetHandle(),    0, VK_WHOLE_SIZE },   // DYNAMIC table (NPC atlas)
        { s_skinCasterPages->GetHandle(), 0, VK_WHOLE_SIZE },
        { s_skinIndirect->GetHandle(),    0, VK_WHOLE_SIZE },
        { s_skinStats->GetHandle(),       0, VK_WHOLE_SIZE },
        { s_dynPageUsed->GetHandle(),     0, VK_WHOLE_SIZE },   // dyn slot -> has-caster flag (resolve skip)
    };
    const VkDescriptorType t[7] = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    };
    VkWriteDescriptorSet w[7]{};
    for (u32 i = 0; i < 7; ++i) { w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = s_skinBinSet[cur]; w[i].dstBinding = i; w[i].descriptorCount = 1; w[i].descriptorType = t[i]; w[i].pBufferInfo = &bi[i]; }
    vkUpdateDescriptorSets(VulkanHW.m_Device, 7, w, 0, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_skinBinPipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_skinBinLayout, 0, 1, &s_skinBinSet[cur], 0, nullptr);
    const u32 push[2] = { s_skinCount, kSkinnedCap };
    vkCmdPushConstants(cmd, s_skinBinLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
    vkCmdDispatch(cmd, (s_skinCount + 63) / 64, 1, 1);
}

// Inside the atlas render pass, after the static draws: rasterize the NPC leaves into
// their bound pages (skin → page route). One indirect draw per leaf (instanceCount =
// the leaf's resident page count, written by the bin). set0 = shared bones, set1 = page data.
void RenderSkinnedCasters(VkCommandBuffer cmd, u32 cur)
{
    if (s_skinCount == 0 || s_skinPageVS == VK_NULL_HANDLE) return;
    if (!EnsureSkinnedPageResources()) return;
    VkDescriptorSet boneSet = VK::Skinned_GetBoneSet();
    if (boneSet == VK_NULL_HANDLE) return;

    VkDescriptorBufferInfo bi[3] = {
        { s_dynPageList->GetHandle(),     0, VK_WHOLE_SIZE },   // DYNAMIC list (NPC atlas)
        { s_skinCasterPages->GetHandle(), 0, VK_WHOLE_SIZE },
        { s_ubo[cur]->GetHandle(),        0, VK_WHOLE_SIZE },
    };
    const VkDescriptorType t[3] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER };
    VkWriteDescriptorSet w[3]{};
    for (u32 i = 0; i < 3; ++i) { w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = s_skinPageSet[cur]; w[i].dstBinding = i; w[i].descriptorCount = 1; w[i].descriptorType = t[i]; w[i].pBufferInfo = &bi[i]; }
    vkUpdateDescriptorSets(VulkanHW.m_Device, 3, w, 0, nullptr);

    VkPipeline lastPipe = VK_NULL_HANDLE;
    for (u32 c = 0; c < s_skinCount; ++c) {
        const VsmSkinnedCaster& sc = s_skinCasters[c];
        VkPipeline pipe = GetSkinnedPagePipeline(sc.stride);
        if (pipe == VK_NULL_HANDLE) continue;
        if (pipe != lastPipe) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
            VkDescriptorSet sets[2] = { boneSet, s_skinPageSet[cur] };
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_skinPageLayout, 0, 2, sets, 0, nullptr);
            lastPipe = pipe;
        }
        const u32 push[4] = { sc.skin_mode, sc.base_bone, sc.bone_count, 0u };
        vkCmdPushConstants(cmd, s_skinPageLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push), push);
        VkDeviceSize z = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &sc.vb, &z);
        vkCmdBindIndexBuffer(cmd, sc.ib, 0, sc.iType);
        vkCmdDrawIndexedIndirect(cmd, s_skinIndirect->GetHandle(),
                                 (VkDeviceSize)c * sizeof(VkDrawIndexedIndirectCommand), 1, sizeof(VkDrawIndexedIndirectCommand));
    }
}

// Grass-caster bin pipeline: 8 bindings (VisibleSSBO, detail indirect, clipmap UBO,
// pageTable, pair arena, stats, dynUsed, pair counters). Per-frame sets (the clipmap UBO varies).
bool CreateGrassBinPipeline()
{
    VkShaderModule cs = g_ShaderManager->Load("vsm_grass_bin.comp.spv");
    if (!cs) { Msg("![VK VSM] vsm_grass_bin.comp.spv load failed"); return false; }
    VkDescriptorSetLayoutBinding b[8]{};
    const VkDescriptorType t[8] = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    };
    for (u32 i = 0; i < 8; ++i) { b[i].binding = i; b[i].descriptorType = t[i]; b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = 8; lci.pBindings = b;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &s_grassBinSetL) != VK_SUCCESS) return false;
    // ×2 sets per frame slot: one for the DYN dispatch, one for the STATIC (hybrid) one.
    VkDescriptorPoolSize ps[2] = { { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, N * 14 }, { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, N * 2 } };
    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets = N * 2; pci.poolSizeCount = 2; pci.pPoolSizes = ps;
    if (vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &s_grassBinPool) != VK_SUCCESS) return false;
    VkDescriptorSetLayout ls[N]; for (u32 i = 0; i < N; ++i) ls[i] = s_grassBinSetL;
    VkDescriptorSetAllocateInfo dai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dai.descriptorPool = s_grassBinPool; dai.descriptorSetCount = N; dai.pSetLayouts = ls;
    if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, s_grassBinSet) != VK_SUCCESS) return false;
    if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, s_grassBinSetS) != VK_SUCCESS) return false;
    VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(GrassBinPush) };
    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1; plci.pSetLayouts = &s_grassBinSetL; plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_grassBinLayout) != VK_SUCCESS) return false;
    VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = cs; cpci.stage.pName = "main"; cpci.layout = s_grassBinLayout;
    if (vkCreateComputePipelines(VulkanHW.m_Device, VK_NULL_HANDLE, 1, &cpci, nullptr, &s_grassBinPipe) != VK_SUCCESS) return false;
    return true;
}

// Grass-page set layout (pair arena, pageList, clipmap UBO, VisibleSSBO) + pipeline
// layout + the VS. The graphics pipeline itself is lazy (needs the grass mesh vertex stride).
bool CreateGrassPageResources()
{
    VkDescriptorSetLayoutBinding b[4]{};
    const VkDescriptorType t[4] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
    for (u32 i = 0; i < 4; ++i) { b[i].binding = i; b[i].descriptorType = t[i]; b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_VERTEX_BIT; }
    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = 4; lci.pBindings = b;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &s_grassPageSetL) != VK_SUCCESS) return false;
    // ×2 sets per frame slot: dyn-atlas draw + static-atlas (hybrid) draw.
    VkDescriptorPoolSize ps[2] = { { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, N * 6 }, { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, N * 2 } };
    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets = N * 2; pci.poolSizeCount = 2; pci.pPoolSizes = ps;
    if (vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &s_grassPagePool) != VK_SUCCESS) return false;
    VkDescriptorSetLayout ls[N]; for (u32 i = 0; i < N; ++i) ls[i] = s_grassPageSetL;
    VkDescriptorSetAllocateInfo dai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dai.descriptorPool = s_grassPagePool; dai.descriptorSetCount = N; dai.pSetLayouts = ls;
    if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, s_grassPageSet) != VK_SUCCESS) return false;
    if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, s_grassPageSetS) != VK_SUCCESS) return false;
    s_grassPageVS  = g_ShaderManager->Load("vsm_grass_page.vert.spv");
    s_grassPageVSS = g_ShaderManager->Load("vsm_grass_page_s.vert.spv");   // static-atlas variant (rigid, _S grid)
    s_grassPageFS  = g_ShaderManager->Load("vsm_grass_page.frag.spv");   // alpha test (blade cutout, not solid quad)
    if (!s_grassPageVS || !s_grassPageFS) Msg("![VK VSM] vsm_grass_page.{vert,frag}.spv missing - grass VSM shadows disabled");
    if (!s_grassPageVSS) Msg("![VK VSM] vsm_grass_page_s.vert.spv missing - STATIC grass shadow hybrid disabled");
    return true;
}

// Lazy: the pipeline LAYOUT needs the detail manager's diffuse set layout (set 1, for the
// alpha test) which exists once the level + grass gfx pipeline are up. Pipeline needs the stride.
// One creator, two variants: DYN atlas VS (wind) and STATIC atlas VS (_S grid, rigid).
VkPipeline EnsureGrassPagePipelineFor(VkPipeline& pipe, VkShaderModule vs, u32 vstride)
{
    if (pipe != VK_NULL_HANDLE) return pipe;
    if (vs == VK_NULL_HANDLE || s_grassPageFS == VK_NULL_HANDLE || vstride == 0) return VK_NULL_HANDLE;
    if (s_grassPageLayout == VK_NULL_HANDLE) {
        CDetailManager* dm = RImplementation.Details;
        VkDescriptorSetLayout diffuseL = dm ? dm->Vsm_GfxSetLayout() : VK_NULL_HANDLE;
        if (diffuseL == VK_NULL_HANDLE) return VK_NULL_HANDLE;
        VkDescriptorSetLayout sets[2] = { s_grassPageSetL, diffuseL };   // set0 = page data (VS), set1 = diffuse (FS) + s_waves (VS)
        VkPushConstantRange pcr{ VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(GrassPagePush) };   // wind + instanceBase
        VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        plci.setLayoutCount = 2; plci.pSetLayouts = sets; plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
        if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_grassPageLayout) != VK_SUCCESS) return VK_NULL_HANDLE;
    }
    // Mesh vertices only — the instance transform is PULLED from the VisibleSSBO by the
    // pair's local index (set 0 binding 3); a compacted pair list can't ride instance-rate
    // attributes (firstInstance can't remap non-contiguous instances).
    VkVertexInputBindingDescription vibd[1] = {
        { 0, vstride, VK_VERTEX_INPUT_RATE_VERTEX },
    };
    VkVertexInputAttributeDescription via[3] = {
        { 0, 0, VK_FORMAT_R32G32B32_SFLOAT,    0  },// aPos (binding 0)
        { 1, 0, VK_FORMAT_R32G32_SFLOAT,       12 },// aUV  (binding 0) — alpha test
        { 2, 0, VK_FORMAT_R32_SFLOAT,          20 },// aHeight (binding 0) — wind stiffness
    };
    VkPipelineVertexInputStateCreateInfo vi{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = vibd;
    vi.vertexAttributeDescriptionCount = 3; vi.pVertexAttributeDescriptions = via;
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;   stages[0].module = vs; stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = s_grassPageFS; stages[1].pName = "main";
    VkPipelineInputAssemblyStateCreateInfo ia{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    vp.viewportCount = 1; vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;
    rs.depthBiasEnable = VK_TRUE;
    VkPipelineMultisampleStateCreateInfo ms{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo ds{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_TRUE; ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    VkPipelineColorBlendStateCreateInfo cb{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    cb.attachmentCount = 0;
    VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS };
    VkPipelineDynamicStateCreateInfo dynState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dynState.dynamicStateCount = 3; dynState.pDynamicStates = dyn;
    VkPipelineRenderingCreateInfo prci{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    prci.colorAttachmentCount = 0; prci.depthAttachmentFormat = VK_FORMAT_D16_UNORM;   // D16 atlas (see CreateRenderResources)
    VkGraphicsPipelineCreateInfo pi{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    pi.pNext = &prci; pi.stageCount = 2; pi.pStages = stages;
    pi.pVertexInputState = &vi; pi.pInputAssemblyState = &ia; pi.pViewportState = &vp;
    pi.pRasterizationState = &rs; pi.pMultisampleState = &ms; pi.pDepthStencilState = &ds;
    pi.pColorBlendState = &cb; pi.pDynamicState = &dynState; pi.layout = s_grassPageLayout;
    if (vkCreateGraphicsPipelines(VulkanHW.m_Device, VK_NULL_HANDLE, 1, &pi, nullptr, &pipe) != VK_SUCCESS)
        Msg("![VK VSM] grass page pipeline create failed (stride=%u)", vstride);
    return pipe;
}
VkPipeline EnsureGrassPagePipeline(u32 vstride)  { return EnsureGrassPagePipelineFor(s_grassPagePipe,  s_grassPageVS,  vstride); }
VkPipeline EnsureGrassPagePipelineS(u32 vstride) { return EnsureGrassPagePipelineFor(s_grassPagePipeS, s_grassPageVSS, vstride); }

// Frame-start clears/templates for the grass pair path (recorded inside MarkPages'
// big fill block, before its transfer→compute barrier — NO barriers of its own):
// zero the pair counters + stats, and write the per-type indirect templates
// (indexCount from the type's mesh; instanceCount is copied from the counters
// after the bin runs).
void GrassBinPrepare(VkCommandBuffer cmd)
{
    if (!ps_r_vsm_grass || s_grassBinPipe == VK_NULL_HANDLE || !s_grassPairCnt) return;
    CDetailManager* dm = RImplementation.Details;
    if (!dm) return;
    const u32 types = _min(dm->Vsm_TypeCount(), kGrassMaxTypes);
    if (types == 0) return;
    vkCmdFillBuffer(cmd, s_grassStats->GetHandle(),   0, VK_WHOLE_SIZE, 0u);
    vkCmdFillBuffer(cmd, s_grassPairCnt->GetHandle(), 0, VK_WHOLE_SIZE, 0u);
    if (s_grassPairCntS) vkCmdFillBuffer(cmd, s_grassPairCntS->GetHandle(), 0, VK_WHOLE_SIZE, 0u);
    u32 tmpl[kGrassMaxTypes * 5]{};
    for (u32 i = 0; i < types; ++i) {
        VkBuffer mvb, mib; u32 ic = 0;
        if (!dm->Vsm_TypeMesh(i, mvb, mib, ic)) ic = 0;
        tmpl[i * 5 + 0] = ic;   // indexCount; instanceCount(+1)/firstIndex/vertexOffset/firstInstance = 0
    }
    vkCmdUpdateBuffer(cmd, s_grassPairInd->GetHandle(), 0, (VkDeviceSize)types * 5 * sizeof(u32), tmpl);
    if (s_grassPairIndS) vkCmdUpdateBuffer(cmd, s_grassPairIndS->GetHandle(), 0, (VkDeviceSize)types * 5 * sizeof(u32), tmpl);
}

// Two dispatches over the same shader (like the tree hybrid): DYN (mode 0, L0 pairs,
// gated by the caller's cadence flag) and STATIC (mode 1, L1/L2 pairs into DIRTY
// static pages — runs EVERY frame: sun round-robin dirties pages even while the dyn
// pass is cadence-frozen). With the hybrid off (r_vsm_grass_static 0) the dyn
// dispatch falls back to mode 2 = L0..L2 every-frame, the pre-hybrid behaviour.
void DispatchGrassBin(VkCommandBuffer cmd, u32 cur, bool dyn)
{
    s_grassSection = 0; s_grassTypes = 0; s_grassPairSection = 0; s_grassPairSectionS = 0;
    if (!ps_r_vsm_grass || s_grassBinPipe == VK_NULL_HANDLE) return;
    CDetailManager* dm = RImplementation.Details;
    if (!dm) return;
    VkBuffer vis = dm->Vsm_VisibleSSBO();
    VkBuffer ind = dm->Vsm_IndirectBuf();
    const u32 types = _min(dm->Vsm_TypeCount(), kGrassMaxTypes);
    const u32 section = dm->Vsm_SectionSize();
    if (vis == VK_NULL_HANDLE || ind == VK_NULL_HANDLE || types == 0 || section == 0) return;
    const bool hybrid = ps_r_vsm_grass_static && s_grassPairsS && s_grassPageVSS != VK_NULL_HANDLE;
    s_grassSection = section; s_grassTypes = types;
    s_grassPairSection  = kGrassPairCap  / types;
    s_grassPairSectionS = hybrid ? (kGrassPairCapS / types) : 0;

    // No fills/barriers here: GrassBinPrepare cleared the counters in the frame-start
    // fill block, and the caller's single residency→bins barrier ordered every
    // producer (prev-frame grass gen + indirect copy, residency's slotDirty — same
    // queue, barrier-chained) before all stage-1 bins. The two dispatches write
    // DISJOINT buffers (stats is atomic) → no barrier between them either.
    GrassBinPush gp{}; gp.sectionSize = section; gp.typeCount = types;
    gp.camRange[0] = Device.vCameraPosition.x; gp.camRange[1] = Device.vCameraPosition.y; gp.camRange[2] = Device.vCameraPosition.z;
    gp.camRange[3] = ps_r_vsm_grass_dist;
    const VkDescriptorType t[8] = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_grassBinPipe);

    if (dyn) {   // DYN dispatch: L0 (hybrid) or L0..L2 (fallback) into the dyn atlas
        VkDescriptorBufferInfo bi[8] = {
            { vis,                       0, VK_WHOLE_SIZE }, { ind,                     0, VK_WHOLE_SIZE },
            { s_ubo[cur]->GetHandle(),   0, VK_WHOLE_SIZE }, { s_dynPageTable->GetHandle(),0, VK_WHOLE_SIZE },  // DYNAMIC table
            { s_grassPairs->GetHandle(), 0, VK_WHOLE_SIZE }, { s_grassStats->GetHandle(),0, VK_WHOLE_SIZE },
            { s_dynPageUsed->GetHandle(),0, VK_WHOLE_SIZE },   // dyn slot -> has-caster flag (resolve skip)
            { s_grassPairCnt->GetHandle(),0, VK_WHOLE_SIZE },  // per-type pair append counters
        };
        VkWriteDescriptorSet w[8]{};
        for (u32 i = 0; i < 8; ++i) { w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = s_grassBinSet[cur]; w[i].dstBinding = i; w[i].descriptorCount = 1; w[i].descriptorType = t[i]; w[i].pBufferInfo = &bi[i]; }
        vkUpdateDescriptorSets(VulkanHW.m_Device, 8, w, 0, nullptr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_grassBinLayout, 0, 1, &s_grassBinSet[cur], 0, nullptr);
        gp.pairSection = s_grassPairSection; gp.mode = hybrid ? 0u : 2u;
        vkCmdPushConstants(cmd, s_grassBinLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(gp), &gp);
        vkCmdDispatch(cmd, (section * types + 63) / 64, 1, 1);
    }

    if (hybrid) {   // STATIC dispatch: L1/L2 into DIRTY static pages (rigid cache half)
        VkDescriptorBufferInfo bi[8] = {
            { vis,                        0, VK_WHOLE_SIZE }, { ind,                     0, VK_WHOLE_SIZE },
            { s_ubo[cur]->GetHandle(),    0, VK_WHOLE_SIZE }, { s_pageTable->GetHandle(),   0, VK_WHOLE_SIZE },  // STATIC table
            { s_grassPairsS->GetHandle(), 0, VK_WHOLE_SIZE }, { s_grassStats->GetHandle(),0, VK_WHOLE_SIZE },
            { s_slotDirty->GetHandle(),   0, VK_WHOLE_SIZE },   // binding 6 = slotDirty in mode 1 (dirty filter)
            { s_grassPairCntS->GetHandle(),0, VK_WHOLE_SIZE },
        };
        VkWriteDescriptorSet w[8]{};
        for (u32 i = 0; i < 8; ++i) { w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = s_grassBinSetS[cur]; w[i].dstBinding = i; w[i].descriptorCount = 1; w[i].descriptorType = t[i]; w[i].pBufferInfo = &bi[i]; }
        vkUpdateDescriptorSets(VulkanHW.m_Device, 8, w, 0, nullptr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_grassBinLayout, 0, 1, &s_grassBinSetS[cur], 0, nullptr);
        gp.pairSection = s_grassPairSectionS; gp.mode = 1u;
        vkCmdPushConstants(cmd, s_grassBinLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(gp), &gp);
        vkCmdDispatch(cmd, (section * types + 63) / 64, 1, 1);
    }
}

// Shared draw loop for both hybrid halves. ONE indirect draw per type: instanceCount =
// the bin's pair count (copied from the counters after the bins) — exactly the
// (instance, page) pairs that exist. The old scheme drew ALL visible instances once
// per slot pass (9, then 12 passes), running the full VS over every blade per pass.
void RenderGrassCastersFor(VkCommandBuffer cmd, u32 cur, VkPipeline pipe, VkDescriptorSet* sets,
                           CVulkanBuffer* pairs, CVulkanBuffer* pairInd, CVulkanBuffer* pageList, u32 pairSection)
{
    CDetailManager* dm = RImplementation.Details;
    VkBuffer vis = dm->Vsm_VisibleSSBO();
    if (vis == VK_NULL_HANDLE || !pairs || !pairInd || pipe == VK_NULL_HANDLE || pairSection == 0) return;

    VkDescriptorBufferInfo bi[4] = {
        { pairs->GetHandle(),      0, VK_WHOLE_SIZE },
        { pageList->GetHandle(),   0, VK_WHOLE_SIZE },   // dyn or static list per variant
        { s_ubo[cur]->GetHandle(), 0, VK_WHOLE_SIZE },
        { vis,                     0, VK_WHOLE_SIZE },   // instance rows, pulled by pair local idx
    };
    const VkDescriptorType t[4] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
    VkWriteDescriptorSet w[4]{};
    for (u32 i = 0; i < 4; ++i) { w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = sets[cur]; w[i].dstBinding = i; w[i].descriptorCount = 1; w[i].descriptorType = t[i]; w[i].pBufferInfo = &bi[i]; }
    vkUpdateDescriptorSets(VulkanHW.m_Device, 4, w, 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_grassPageLayout, 0, 1, &sets[cur], 0, nullptr);
    GrassPagePush push{};
    dm->Vsm_WindPush(push.wind_params, push.wsetup_grass, push.wind_anim);   // static VS ignores the wind fields
    for (u32 i = 0; i < s_grassTypes; ++i) {
        VkBuffer mvb, mib; u32 ic;
        if (!dm->Vsm_TypeMesh(i, mvb, mib, ic)) continue;
        VkDescriptorSet diffuse = dm->Vsm_TypeDiffuseSet(i);   // set 1 = grass diffuse (alpha test) + s_waves (wind)
        if (diffuse == VK_NULL_HANDLE) continue;
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_grassPageLayout, 1, 1, &diffuse, 0, nullptr);
        VkDeviceSize z = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &mvb, &z);
        vkCmdBindIndexBuffer(cmd, mib, 0, VK_INDEX_TYPE_UINT16);
        push.wind_params.w = dm->Vsm_TypeWindScale(i);   // DO_NO_WAVING micro-plants stay static
        push.instanceBase  = i * s_grassSection;
        push.pairBase      = i * pairSection;
        vkCmdPushConstants(cmd, s_grassPageLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push), &push);
        vkCmdDrawIndexedIndirect(cmd, pairInd->GetHandle(), (VkDeviceSize)i * sizeof(VkDrawIndexedIndirectCommand), 1, sizeof(VkDrawIndexedIndirectCommand));
    }
}

// DYN half: L0 pairs into the dynamic atlas, live wind (inside the dyn atlas pass).
void RenderGrassCasters(VkCommandBuffer cmd, u32 cur)
{
    if (!ps_r_vsm_grass || s_grassPageVS == VK_NULL_HANDLE || s_grassTypes == 0 || s_grassSection == 0) return;
    CDetailManager* dm = RImplementation.Details;
    if (!dm) return;
    RenderGrassCastersFor(cmd, cur, EnsureGrassPagePipeline(dm->Vsm_VertexStride()), s_grassPageSet,
                          s_grassPairs, s_grassPairInd, s_dynPageList, s_grassPairSection);
}

// STATIC half: L1/L2 pairs into DIRTY static-atlas pages, rigid (inside the static
// pass, after opaque + far trees — the dirty clear already ran at pass start).
void RenderGrassCastersStatic(VkCommandBuffer cmd, u32 cur)
{
    if (!ps_r_vsm_grass || !ps_r_vsm_grass_static || s_grassPageVSS == VK_NULL_HANDLE || s_grassTypes == 0 || s_grassSection == 0) return;
    CDetailManager* dm = RImplementation.Details;
    if (!dm) return;
    RenderGrassCastersFor(cmd, cur, EnsureGrassPagePipelineS(dm->Vsm_VertexStride()), s_grassPageSetS,
                          s_grassPairsS, s_grassPairIndS, s_pageList, s_grassPairSectionS);
}

} // anonymous namespace

bool Init()
{
    VK::Vram::Scope _vram_scope("VSM");
    if (s_inited) return !s_dead;
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return false;
    s_inited = true;

    if (!CreatePipeline()) { Msg("![VK VSM] compute init failed — VSM disabled"); s_dead = true; return false; }

    for (u32 i = 0; i < N; ++i) {
        s_ubo[i] = xr_new<CVulkanBuffer>();
        s_ubo[i]->Create(sizeof(VsmParams), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
        s_uboPtr[i] = s_ubo[i]->Map();
    }
    s_needed = xr_new<CVulkanBuffer>();
    s_needed->Create((VkDeviceSize)kPageCount * sizeof(u32),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    s_rmask = xr_new<CVulkanBuffer>();   // receiver mask: 8×8 sampled cells per virtual page (48 KB)
    s_rmask->Create((VkDeviceSize)kPageCount * 2 * sizeof(u32),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    s_pageHits = xr_new<CVulkanBuffer>();   // gaze refresh: per-page sampled-pixel counts (24 KB)
    s_pageHits->Create((VkDeviceSize)kPageCount * sizeof(u32),
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                       VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    s_counter = xr_new<CVulkanBuffer>();
    s_counter->Create(sizeof(u32),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    s_readback = xr_new<CVulkanBuffer>();
    s_readback->Create(8 * sizeof(u32), VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
    s_readPtr = (u32*)s_readback->Map();
    if (s_readPtr) memset(s_readPtr, 0, 8 * sizeof(u32));

    // Allocation buffers + pipeline.
    s_pageTable = xr_new<CVulkanBuffer>();
    s_pageTable->Create((VkDeviceSize)kPageCount * sizeof(u32),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    s_pageList = xr_new<CVulkanBuffer>();
    s_pageList->Create((VkDeviceSize)kMaxPhysS * 4 * sizeof(u32),   // uvec4 per STATIC toroidal slot (6144)
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    // Dynamic atlas's own demand-alloc table/list/info (rebuilt every frame).
    s_dynPageTable = xr_new<CVulkanBuffer>();
    s_dynPageTable->Create((VkDeviceSize)kPageCount * sizeof(u32),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                           VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    s_dynPageList = xr_new<CVulkanBuffer>();
    s_dynPageList->Create((VkDeviceSize)kMaxPhys * 4 * sizeof(u32),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    s_dynAllocInfo = xr_new<CVulkanBuffer>();
    s_dynAllocInfo->Create((VkDeviceSize)(1 + kLevels) * sizeof(u32),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                           VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    s_dynPageUsed = xr_new<CVulkanBuffer>();
    s_dynPageUsed->Create((VkDeviceSize)kMaxPhys * sizeof(u32),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);

    if (!CreateAllocPipeline()) { Msg("![VK VSM] alloc pipeline failed - VSM disabled"); s_dead = true; return false; }

    // Toroidal STATIC residency buffers + pipelines (Phase 1b).
    s_physTile = xr_new<CVulkanBuffer>();
    s_physTile->Create((VkDeviceSize)kMaxPhysS * 2 * sizeof(u32),   // uvec2 per slot, persistent
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    s_slotDirty = xr_new<CVulkanBuffer>();
    s_slotDirty->Create((VkDeviceSize)kMaxPhysS * sizeof(u32),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    s_dirtyList = xr_new<CVulkanBuffer>();
    // +1 tail dword = the wrong-tile request counter (UE5's "counter shares the tail" trick:
    // the append guard is < kMaxPhysS so list writes never touch it; the frame-start fill
    // zeroes it, the debug readback copies it out). NOT in drawClear — [2]/[3] there are
    // firstVertex/firstInstance of the live clear draw, a counter would corrupt indexing.
    // +2 tail dwords: [kMaxPhysS] = wrong-tile request counter, [kMaxPhysS+1] = gaze-refresh counter.
    s_dirtyList->Create((VkDeviceSize)(kMaxPhysS + 2) * sizeof(u32),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    // shadow-HZB (r_vsm_hzb): per-slot occluder max + validity (written by residency).
    s_priorValid = xr_new<CVulkanBuffer>();
    s_priorValid->Create((VkDeviceSize)kMaxPhysS * sizeof(u32),
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    s_pageMax = xr_new<CVulkanBuffer>();
    s_pageMax->Create((VkDeviceSize)kMaxPhysS * sizeof(float),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    s_pageMaxBlk = xr_new<CVulkanBuffer>();
    s_pageMaxBlk->Create((VkDeviceSize)kMaxPhysS * 64 * sizeof(float),   // 8×8 blocks per page (~1.6 MB)
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    s_slotDirtyPrev = xr_new<CVulkanBuffer>();
    s_slotDirtyPrev->Create((VkDeviceSize)kMaxPhysS * sizeof(u32),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    s_drawClear = xr_new<CVulkanBuffer>();
    s_drawClear->Create(4 * sizeof(u32),   // VkDrawIndirectCommand
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    s_residRB = xr_new<CVulkanBuffer>();
    s_residRB->Create(4 * sizeof(u32), VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
    s_residPtr = (u32*)s_residRB->Map();
    if (s_residPtr) memset(s_residPtr, 0, 4 * sizeof(u32));
    if (!CreateResidPipeline()) { Msg("![VK VSM] resid pipeline failed - VSM disabled"); s_dead = true; return false; }
    if (!CreateClearPipeline()) { Msg("![VK VSM] clear pipeline failed - VSM disabled"); s_dead = true; return false; }
    {
        VkDescriptorBufferInfo bi[9] = {
            { s_needed->GetHandle(),    0, VK_WHOLE_SIZE }, { s_pageTable->GetHandle(), 0, VK_WHOLE_SIZE },
            { s_pageList->GetHandle(),  0, VK_WHOLE_SIZE }, { s_physTile->GetHandle(),  0, VK_WHOLE_SIZE },
            { s_slotDirty->GetHandle(), 0, VK_WHOLE_SIZE }, { s_dirtyList->GetHandle(), 0, VK_WHOLE_SIZE },
            { s_drawClear->GetHandle(), 0, VK_WHOLE_SIZE }, { s_priorValid->GetHandle(), 0, VK_WHOLE_SIZE },
            { s_pageHits->GetHandle(),  0, VK_WHOLE_SIZE },
        };
        VkWriteDescriptorSet w[9]{};
        for (u32 i = 0; i < 9; ++i) { w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = s_residSet; w[i].dstBinding = i; w[i].descriptorCount = 1; w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pBufferInfo = &bi[i]; }
        VkDescriptorBufferInfo cb{ s_dirtyList->GetHandle(), 0, VK_WHOLE_SIZE };
        VkWriteDescriptorSet cw{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        cw.dstSet = s_clearSet; cw.dstBinding = 0; cw.descriptorCount = 1; cw.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; cw.pBufferInfo = &cb;
        vkUpdateDescriptorSets(VulkanHW.m_Device, 9, w, 0, nullptr);
        vkUpdateDescriptorSets(VulkanHW.m_Device, 1, &cw, 0, nullptr);
    }

    // Binning + draw-build buffers + pipeline.
    s_casterPages = xr_new<CVulkanBuffer>();
    s_casterPages->Create((VkDeviceSize)kMaxCasters * kPagesCap * sizeof(u32),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    s_vsmIndirect = xr_new<CVulkanBuffer>();
    s_vsmIndirect->Create((VkDeviceSize)kMaxGroups * kGroupStride * sizeof(VkDrawIndexedIndirectCommand),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                          VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    s_vsmGroupCount = xr_new<CVulkanBuffer>();
    s_vsmGroupCount->Create((VkDeviceSize)kMaxGroups * sizeof(u32),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    // [4] = casterPages arena cursor, [5] = arena-full drops (cluster bin).
    s_binStats = xr_new<CVulkanBuffer>();
    s_binStats->Create(8 * sizeof(u32),
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    s_binReadback = xr_new<CVulkanBuffer>();
    s_binReadback->Create(8 * sizeof(u32), VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
    s_binReadPtr = (u32*)s_binReadback->Map();
    if (s_binReadPtr) memset(s_binReadPtr, 0, 8 * sizeof(u32));

    // Phase 3 cluster bin: per-combo counts + the cluster-bin pipeline.
    // Non-fatal — without it the per-mesh ShadowGPU bin below keeps running.
    s_clCount = xr_new<CVulkanBuffer>();
    s_clCount->Create((VkDeviceSize)kMaxCombos * sizeof(u32),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    if (!CreateBinClusterPipeline()) Msg("![VK VSM] cluster bin pipeline failed - r_vsm_cluster inert (per-mesh bin active)");

    if (!CreateBinPipeline()) { Msg("![VK VSM] bin pipeline failed - VSM disabled"); s_dead = true; return false; }
    if (!CreateRenderResources()) { Msg("![VK VSM] render resources failed - VSM disabled"); s_dead = true; return false; }
    if (!CreateHzbReducePipeline()) Msg("![VK VSM] shadow-HZB reduce pipeline failed - r_vsm_hzb inert");   // non-fatal

    // Temporal resolve: pipeline + per-frame resolve UBOs (mask images are lazy, screen-sized).
    for (u32 i = 0; i < N; ++i) {
        s_resolveUbo[i] = xr_new<CVulkanBuffer>();
        s_resolveUbo[i]->Create(sizeof(ResolveParams), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
        s_resolveUboPtr[i] = s_resolveUbo[i]->Map();
    }
    if (!CreateResolvePipeline()) { Msg("![VK VSM] resolve pipeline failed - VSM disabled"); s_dead = true; return false; }
    s_prevViewProj.identity();

    // Skinned (NPC) casters: per-frame meta + page/indirect/stats buffers + bin pipeline.
    for (u32 i = 0; i < N; ++i) {
        s_skinMeta[i] = xr_new<CVulkanBuffer>();
        s_skinMeta[i]->Create((VkDeviceSize)kMaxSkinned * sizeof(SkinMetaGPU), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
        s_skinMetaPtr[i] = s_skinMeta[i]->Map();
    }
    s_skinCasterPages = xr_new<CVulkanBuffer>();
    s_skinCasterPages->Create((VkDeviceSize)kMaxSkinned * kSkinnedCap * sizeof(u32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    s_skinIndirect = xr_new<CVulkanBuffer>();
    s_skinIndirect->Create((VkDeviceSize)kMaxSkinned * sizeof(VkDrawIndexedIndirectCommand),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    s_skinStats = xr_new<CVulkanBuffer>();
    s_skinStats->Create(4 * sizeof(u32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    s_skinStatsRB = xr_new<CVulkanBuffer>();
    s_skinStatsRB->Create(4 * sizeof(u32), VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
    s_skinStatsPtr = (u32*)s_skinStatsRB->Map();
    if (s_skinStatsPtr) memset(s_skinStatsPtr, 0, 4 * sizeof(u32));
    if (!CreateSkinnedBinPipeline()) { Msg("![VK VSM] skinned bin pipeline failed - NPC VSM shadows disabled"); }
    s_skinPageVS = g_ShaderManager->Load("vsm_skinned_page.vert.spv");
    if (!s_skinPageVS) Msg("![VK VSM] vsm_skinned_page.vert.spv missing - NPC VSM shadows disabled");

    // Grass (detail) casters: compact (instance, page) pair arena + per-type counters
    // + indirect (instanceCount = pair count) + bin/page pipelines.
    s_grassPairs = xr_new<CVulkanBuffer>();
    s_grassPairs->Create((VkDeviceSize)kGrassPairCap * sizeof(u32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    s_grassPairCnt = xr_new<CVulkanBuffer>();
    s_grassPairCnt->Create((VkDeviceSize)kGrassMaxTypes * sizeof(u32),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    s_grassPairInd = xr_new<CVulkanBuffer>();
    s_grassPairInd->Create((VkDeviceSize)kGrassMaxTypes * sizeof(VkDrawIndexedIndirectCommand),
                           VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    // Static-hybrid mirrors (far grass into dirty static pages).
    s_grassPairsS = xr_new<CVulkanBuffer>();
    s_grassPairsS->Create((VkDeviceSize)kGrassPairCapS * sizeof(u32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    s_grassPairCntS = xr_new<CVulkanBuffer>();
    s_grassPairCntS->Create((VkDeviceSize)kGrassMaxTypes * sizeof(u32),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    s_grassPairIndS = xr_new<CVulkanBuffer>();
    s_grassPairIndS->Create((VkDeviceSize)kGrassMaxTypes * sizeof(VkDrawIndexedIndirectCommand),
                            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    s_grassStats = xr_new<CVulkanBuffer>();
    s_grassStats->Create(8 * sizeof(u32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    s_grassStatsRB = xr_new<CVulkanBuffer>();
    s_grassStatsRB->Create(8 * sizeof(u32), VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
    s_grassStatsPtr = (u32*)s_grassStatsRB->Map();
    if (s_grassStatsPtr) memset(s_grassStatsPtr, 0, 8 * sizeof(u32));
    s_grassPairCntRB = xr_new<CVulkanBuffer>();
    s_grassPairCntRB->Create((VkDeviceSize)kGrassMaxTypes * sizeof(u32), VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
    s_grassPairCntPtr = (u32*)s_grassPairCntRB->Map();
    if (s_grassPairCntPtr) memset(s_grassPairCntPtr, 0, kGrassMaxTypes * sizeof(u32));
    if (!CreateGrassBinPipeline())  { Msg("![VK VSM] grass bin pipeline failed - grass VSM shadows disabled"); }
    if (!CreateGrassPageResources()) { Msg("![VK VSM] grass page resources failed - grass VSM shadows disabled"); }

    // Dynamic alloc descriptor set (all buffers fixed) — written once. Reads needed[],
    // writes the dynamic page table / list / info.
    {
        VkDescriptorBufferInfo bd[4] = {
            { s_needed->GetHandle(),       0, VK_WHOLE_SIZE },
            { s_dynPageTable->GetHandle(), 0, VK_WHOLE_SIZE },
            { s_dynPageList->GetHandle(),  0, VK_WHOLE_SIZE },
            { s_dynAllocInfo->GetHandle(), 0, VK_WHOLE_SIZE },
        };
        VkWriteDescriptorSet w[4]{};
        for (u32 i = 0; i < 4; ++i) { w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = s_dynAllocSet; w[i].dstBinding = i; w[i].descriptorCount = 1; w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pBufferInfo = &bd[i]; }
        vkUpdateDescriptorSets(VulkanHW.m_Device, 4, w, 0, nullptr);
    }

    Prof::NameBuffer(s_needed->GetHandle(),    "VSM.PageNeeded");
    Prof::NameBuffer(s_counter->GetHandle(),   "VSM.UniqueCounter");
    Prof::NameBuffer(s_pageTable->GetHandle(), "VSM.PageTable");
    Prof::NameBuffer(s_pageList->GetHandle(),  "VSM.PageList");

    Msg("[VK VSM] init OK (%u levels, %u virtual, %u pages/axis, %u phys, base %.0fm -> finest texel %.1fmm)",
        kLevels, kVirtualRes, kPagesAxis, kMaxPhys, ps_r_vsm_base, 1000.0f * ps_r_vsm_base / float(kVirtualRes));
    return true;
}

bool Ready()   { return s_inited && !s_dead && s_pipe != VK_NULL_HANDLE; }
bool Wanted()  { return ps_r_vsm != 0; }   // no Init dependency — see vk_pass_world guard
bool Enabled() { return Ready() && ps_r_vsm; }
// Sun below the horizon (set by BeginFrame): skip the whole VSM sun-shadow update.
// Only once the atlas+mask are valid to freeze (else we'd skip the very first build
// and never get a mask to reuse). Receivers keep sampling the last daylit mask ×
// sun_color≈0 → invisible, so freezing is free of visible change.
bool NightFrozen() { return s_sunDown && AtlasReady() && MaskReady(); }

void BeginFrame(const Fvector& camPos, VkExtent2D screen)
{
    VK::Vram::Scope _vram_scope("VSM");
    if (!s_inited) Init();
    if (!Enabled()) return;

    EnsureMaskTargets(screen);   // screen-space mask (create/resize) — before EnvLight binds it
    s_curCamPos = camPos;

    // Write-side raster bias changed (live A/B) → cached static pages hold depth
    // rendered with the OLD bias — drop them so the flip is crisp, not a seconds-long
    // round-robin mix of old and new page content.
    {
        static float s_lastRB = -1.f, s_lastRS = -1.f;
        if (ps_r_vsm_raster_bias != s_lastRB || ps_r_vsm_raster_slope != s_lastRS) {
            if (s_lastRB >= 0.f) InvalidateCache();
            s_lastRB = ps_r_vsm_raster_bias; s_lastRS = ps_r_vsm_raster_slope;
        }
    }

    // ---- THROTTLE (r_vsm_throttle): cost-feedback LOD bias, the UE5 VSM throttle port.
    // Cost source = the profiler's World/VSMrender GPU zone (timestamps are collected
    // every frame regardless of r_profiler logging; 2-3 frames of fence latency is fine
    // for a controller). Asymmetric steps like UE5's (MaxStepUp 0.2 / MaxStepDown -0.01):
    // react fast to an overload spike, recover slowly — each recovery step re-renders the
    // finer level's pages, so trickling it keeps that burst inside the dirty budget.
    {
        // Cost sample — taken even with the throttle OFF so an A/B baseline log carries
        // the exact same telemetry (cost avg/max, over-budget %).
        const VK::Prof::ZoneStat* zs = nullptr;
        const u32 zn = VK::Prof::GetZones(&zs);
        float cost = -1.f;
        for (u32 z = 0; z < zn; ++z)
            if (!strcmp(zs[z].name, "World/VSMrender")) { cost = zs[z].gpuLast; break; }
        if (cost >= 0.f) {
            s_thCostSum += cost; s_thFrames++;
            if (cost > s_thCostMax) s_thCostMax = cost;
            if (cost > ps_r_vsm_throttle_budget) s_thOver++;
        }
        if (ps_r_vsm_throttle && ps_r_vsm_cache) {
            if (cost >= 0.f) {
                float step = (cost - ps_r_vsm_throttle_budget) * 0.15f;   // response gain
                if (step >  0.10f) step =  0.10f;
                if (step < -0.01f) step = -0.01f;
                s_throttleVal += step;
                const float mx = ps_r_vsm_throttle_max < 0.f ? 0.f : ps_r_vsm_throttle_max;
                if (s_throttleVal < 0.f) s_throttleVal = 0.f;
                if (s_throttleVal > mx)  s_throttleVal = mx;
            }
        } else
            s_throttleVal = 0.f;
        // Integer quantization with a deadband: bias UP the moment the controller crosses
        // a boundary (relief now), bias DOWN only after it recedes 0.25 below it — with the
        // -0.01 down-step that's ≥25 calm frames, so the marked level set never flip-flops.
        const u32 prevBias = s_lodBias;
        const u32 q = (u32)s_throttleVal;
        if (q > s_lodBias) s_lodBias = q;
        else if (s_lodBias > 0u && s_throttleVal < float(s_lodBias) - 0.25f) s_lodBias--;
        s_thBiasHist[s_lodBias >= 2u ? 2u : s_lodBias]++;
        if (s_lodBias != prevBias)   // rare + THE key A/B event → always log
            Msg("[VK VSM] throttle: bias %u -> %u (ctl=%.2f, VSMrender=%.2fms, budget=%.2f)",
                prevBias, s_lodBias, s_throttleVal, cost, ps_r_vsm_throttle_budget);
    }

    // ---- CADENCE (r_vsm_cadence), DECOUPLED (Level 1): the sun/clipmap and the cheap
    // STATIC atlas (walls, terrain, ground — where most visible shadows land) update EVERY
    // frame, so those shadows track the moving sun smoothly with NO strobe. Only the
    // EXPENSIVE dynamic pass (NPC + grass + near-tree wind crowns) is throttled to every
    // Nth frame; on skip frames its atlas + page table + dynUsed flags are PRESERVED
    // (world-anchored → still aligned standing/slow) and the resolve reuses them 1..N-1
    // frames stale. 0/1 = every frame (off); 2-3 ≈ 30-45 Hz dyn. (The OLD cadence froze the
    // WHOLE atlas incl. sun/static → the periodic global snap was the visible twitch.)
    const int cad = ps_r_vsm_cadence < 1 ? 1 : ps_r_vsm_cadence;
    // Frame-aligned (Device.dwFrame ticks exactly once/frame) — immune to any extra
    // BeginFrame call. Tally update/skip so the diag log can PROVE the skip ratio.
    s_dynSkip = (cad > 1) && (Device.dwFrame % (u32)cad != 0);
    if (s_dynFirst) s_dynSkip = false;   // never skip until the dyn atlas is initialized once
    // Camera-motion gate: the frozen dyn atlas is world-anchored → aligned only while (nearly)
    // stationary. Moving OR turning scrolls dyn pages in/out of coverage + batches the near-set
    // static↔dyn transitions → crown/NPC shadows FLICKER while walking closer/farther. So force
    // full-rate dyn on any real camera motion this frame; cadence then saves work only when you
    // HOLD STILL (the vista case where the peaks live). r_vsm_cadence_still = m/frame deadband.
    if (s_dynSkip) {
        const float dPos = camPos.distance_to(s_dynPrevCam);
        const float dDir = 1.0f - Device.vCameraDirection.dotproduct(s_dynPrevDir);   // ~θ²/2; 2e-4 ≈ 1.1°/frame
        if (dPos > ps_r_vsm_cadence_still || dDir > 2.0e-4f) s_dynSkip = false;
    }
    s_dynPrevCam = camPos;
    s_dynPrevDir = Device.vCameraDirection;
    if (cad != s_lastCad) { s_lastCad = cad; s_updTally = s_skipTally = 0; }
    // (skip/update tally is counted AFTER the sun/window alignment gate below — it can still
    // flip s_dynSkip to false once the clipmap is known.)
    // No early return — the static side + clipmap update unconditionally below.

    const u32 cur = s_frame % N;
    s_frame++;
    s_curSlot = cur;   // MarkPages / RenderAtlas / receivers all reuse this slot's UBO

    // ---- Sun direction (same env source as Pass_SunShadow). NIGHT test: the sun is
    // below the horizon → it casts NO direct light on the scene (occluded by the
    // earth), so its whole shadow update is wasted. Primary signal is GEOMETRIC
    // (to-sun.y < 0 = below horizon) — robust regardless of the residual night
    // sun_color; luminance is a secondary OR (some env keeps the sun near the
    // horizon but dims its colour). NightFrozen() (below) then skips the update.
    Fvector sunDir; sunDir.set(0.f, -1.f, 0.f);
    float sunLum = 1.f;
    if (g_pGamePersistent)
        if (auto* E = g_pGamePersistent->Environment().CurrentEnv) {
            sunDir = E->sun_dir;
            sunLum = E->sun_color.x * 0.299f + E->sun_color.y * 0.587f + E->sun_color.z * 0.114f;
        }
    {
        Fvector sdn = sunDir; if (sdn.magnitude() > 1e-4f) sdn.normalize(); else sdn.set(0.f, -1.f, 0.f);
        const float toSunY = -sdn.y;   // sun altitude proxy (>0 above horizon, <0 below)
        s_sunDown = ps_r_sun_night_freeze && (toSunY < ps_r_sun_night_alt || sunLum < ps_r_sun_night_lum);
        // Frozen-state diag here (the split-count log lives in RenderAtlas, which is
        // SKIPPED when frozen — so it would never report the freeze). r_vsm_debug 1.
        if (ps_r_vsm_debug) {
            static bool s_prevDown = false;
            static u32  s_lastFreezeLog = 0;
            if (s_sunDown != s_prevDown || Device.dwTimeGlobal - s_lastFreezeLog > 3000) {
                s_lastFreezeLog = Device.dwTimeGlobal;
                Msg("[VK VSM] night-freeze: %s (to-sun.y=%.3f sunLum=%.3f | alt<%.3f lum<%.3f)",
                    (s_sunDown && AtlasReady() && MaskReady()) ? "FROZEN (sun-shadow update skipped)" :
                    (s_sunDown ? "pending (atlas not ready yet)" : "active (sun up)"),
                    toSunY, sunLum, ps_r_sun_night_alt, ps_r_sun_night_lum);
            }
            s_prevDown = s_sunDown;
        }
    }
    const bool sunDark = s_sunDown;

    // ---- Clipmap params: sun view (world-anchored: eye at origin → light XY is
    // camera-INDEPENDENT, so pages are world-anchored and cacheable). Per level the
    // window follows the camera but its origin is snapped to that level's texel grid.
    Fvector sd = sunDir; if (sd.magnitude() < 1e-4f) sd.set(0.f, -1.f, 0.f); sd.normalize();
    Fvector up; up.set(0.f, 1.f, 0.f); if (_abs(sd.y) > 0.99f) up.set(0.f, 0.f, 1.f);
    Fvector eye; eye.set(0.f, 0.f, 0.f);
    Fmatrix view; view.build_camera_dir(eye, sd, up);

    s_sunView = view;   // for MarkPages' transition-sphere light transforms

    Fvector camL; view.transform_tiny(camL, camPos);   // camera in light space

    // The window origin below is snapped to the PAGE grid (128 texels) of a CAMERA-
    // INDEPENDENT, world-anchored lattice (phase 0 = world origin in light space, since
    // the sun view has eye at origin). While the sun is static the page lattice is world-
    // fixed → the same window pages cover the same world tiles every frame, so the toroidal
    // cache stays valid and the camera can roam within a page cell (0.75 m at L0) with zero
    // invalidation. NO sub-texel jitter — it would mismatch the cached pages (r_vsm_temporal
    // only gates the EMA history blend in the resolve). Sun motion since last frame drives
    // the round-robin refresh (paused sun → 0 refresh → full cache).
    s_sunMoving  = (sd.x != s_prevSunDir.x) || (sd.y != s_prevSunDir.y) || (sd.z != s_prevSunDir.z);
    s_prevSunDir = sd;

    VsmParams params{};
    params.view = view;
    for (u32 L = 0; L < kLevels; ++L) {
        const float ext   = ps_r_vsm_base * float(1u << L);
        const float texel = ext / float(kVirtualRes);
        const float pageTexel = texel * float(kPageSize);                // one page in light-space metres
        const float baseX = camL.x - 0.5f * ext;                         // window centred on the camera
        const float baseY = camL.y - 0.5f * ext;
        const s32   pbx = (s32)floorf(baseX / pageTexel);                // page-lattice index (phase 0 = world origin)
        const s32   pby = (s32)floorf(baseY / pageTexel);
        s_pageBase[L][0] = pbx; s_pageBase[L][1] = pby;
        params.level[L][0] = float(pbx) * pageTexel;                     // window origin, page-aligned to the world lattice (toroidal cache)
        params.level[L][1] = float(pby) * pageTexel;
        params.level[L][2] = ext;
        params.level[L][3] = 0.f;
    }
    params.zparams[0] = kZNear;
    params.zparams[1] = 1.0f / (kZFar - kZNear);
    params.zparams[2] = ps_r_vsm_bias;       // STATIC-atlas receiver bias (live, r_vsm_bias) — terrain acne
    params.zparams[3] = ps_r_vsm_bias_dyn;   // DYNAMIC-atlas receiver bias (live, r_vsm_bias_dyn) — casters only,
                                             // no ground in that atlas → tiny epsilon; the shared 0.0003 (=0.6 m
                                             // at the 2000 m z-range) ate every grass shadow below ~knee height

    if (s_uboPtr[cur]) memcpy(s_uboPtr[cur], &params, sizeof(params));

    // ---- Cadence ALIGNMENT GATE. The frozen dyn atlas is valid ONLY for the sun angle +
    // clipmap window it was rendered at. A living sun — especially env-keyframe STEPS (measured
    // up to ~18-22° across a few frames in the diag log) — rotates the light-space lattice → the
    // window page base SNAPS and the frozen dyn shadows misalign for a frame = the shadow flicker
    // the user saw. Fix: force a full-rate dyn update whenever the window snapped OR the sun
    // rotated past r_vsm_cadence_sun since the last dyn update (analogous to the camera-motion
    // gate above). Between snaps the cadence still skips → the perf saving stays.
    if (s_dynSkip) {
        u32 winSnaps = 0;
        for (u32 L = 0; L < kLevels; ++L)
            if (s_pageBase[L][0] != s_dynUpdBase[L][0] || s_pageBase[L][1] != s_dynUpdBase[L][1]) winSnaps++;
        float dp = sunDir.dotproduct(s_dynUpdSun); dp = dp > 1.f ? 1.f : (dp < -1.f ? -1.f : dp);
        const float sunDrift = acosf(dp) * 57.29578f;
        if (winSnaps > 0 || sunDrift > ps_r_vsm_cadence_sun) {
            s_dynSkip = false;   // misaligned → re-render the dyn atlas at the CURRENT sun/window
            if (ps_r_vsm_debug) { if (sunDrift > s_cadSunDriftMax) s_cadSunDriftMax = sunDrift; s_cadSnapTally++; }
        }
    }
    // NIGHT FREEZE (wins over the motion/alignment gates above): the sun is below
    // the horizon → sun_color ≈ 0 → every sun receiver multiplies its shadow term by
    // ~0, so the dyn atlas (NPC/grass/near-tree crowns — the dominant re-raster cost)
    // is invisible no matter how stale/misaligned. Freeze it. Never before the dyn
    // atlas is initialized once (s_dynFirst), and the snapshot below is deliberately
    // NOT taken while frozen so that at dawn the alignment gate re-renders correctly.
    // Belt-and-suspenders: also freeze the dyn atlas (the whole update is skipped at
    // the Pass_World call site via NightFrozen(), but keep this so any other caller is safe).
    if (sunDark && !s_dynFirst) s_dynSkip = true;
    (s_dynSkip ? s_skipTally : s_updTally)++;   // final, post-gate decision
    // Snapshot on (real or forced) dyn-update frames — the sun/window the dyn atlas is now valid for.
    if (!s_dynSkip) {
        s_dynUpdFrame = Device.dwFrame; s_dynUpdSun = sunDir;
        for (u32 L = 0; L < kLevels; ++L) { s_dynUpdBase[L][0] = s_pageBase[L][0]; s_dynUpdBase[L][1] = s_pageBase[L][1]; }
    } else if (ps_r_vsm_debug) {
        const u32 held = Device.dwFrame - s_dynUpdFrame;
        if (held > s_cadHeldMax) s_cadHeldMax = held;
    }
}

VkBuffer    GetPageTableHandle() { return s_pageTable ? s_pageTable->GetHandle() : VK_NULL_HANDLE; }
VkBuffer    GetRMaskHandle()     { return s_rmask ? s_rmask->GetHandle() : VK_NULL_HANDLE; }   // receiver mask (r_vsm_rmask)
VkBuffer    GetUBOHandle()       { return s_ubo[s_curSlot] ? s_ubo[s_curSlot]->GetHandle() : VK_NULL_HANDLE; }

// Cluster-bin candidate table. The ortho LOD cut is camera-independent, so the
// set of entries that can EVER cast is static per (level geometry, errB(0)):
// an entry draws into level L iff (selfError <= errB(L) || sleaf) && parentError
// > errB(L), and errB grows with L, so ANY drawing entry has parentError >
// errB(0). That predicate (minus alpha-tested) is evaluated once on the CPU
// over the WorldGPU host metas; everything below the finest cut — the vast
// majority of DAG clusters on big maps — never reaches the GPU bin at all.
// Candidates are routed to indirect cmd regions by BUFFER-COMBO (vb/ib/stride/
// iType): after VB paging the repacked groups all share the cluster pools, so
// combos stay a handful and the atlas draw loop shrinks accordingly. Region
// capacity = the combo's candidate count → an entry emits ≤ 1 cmd → no overflow
// by construction (the invariant the per-group entryOffset regions used to give).
void DestroyClusterCandidates()
{
    auto del = [](CVulkanBuffer*& b) { if (b) { xr_delete(b); b = nullptr; } };
    del(s_clCandIdx); del(s_clComboBase); del(s_clIndirect);
    s_clCombos.clear(); s_clCandCount = 0; s_clCandOk = false;
}

bool RebuildClusterCandidates(float errB0)
{
    // Live rebuild (k / clipmap-base cvar): in-flight frames may still read the
    // old buffers — drain the queues before freeing them. Rare (cvar flip or
    // level load, where the device is idle anyway), so the hitch is acceptable.
    if (s_clCandIdx || s_clComboBase || s_clIndirect) vkDeviceWaitIdle(VulkanHW.m_Device);
    DestroyClusterCandidates();
    s_clCandStamp = WorldGPU::BuildStamp();
    s_clCandErrB0 = errB0;

    const auto& meta   = WorldGPU::HostMeta();
    const u32   groupN = WorldGPU::GroupCount();
    if (meta.empty() || groupN == 0) return false;
    if (meta.size() > 0x000FFFFFull) {   // entry id must fit 20 bits (combo packs the top 12)
        Msg("![VK VSM] cluster candidates: %zu entries exceed 20-bit ids - per-mesh bin active", meta.size());
        return false;
    }

    // Group -> buffer combo. Opaque combos key on (vb,ib,stride,iType) as
    // before; with r_vsm_at the ALPHA-TESTED groups get combos too, keyed
    // additionally by MATERIAL (the draw binds its diffuse for the discard) and
    // appended AFTER every opaque combo — one pipeline-layout switch per pass.
    const bool atOn = ps_r_vsm_at != 0;
    s_clCandAT = atOn;
    xr_vector<u32> comboOf(groupN, ~0u);
    xr_vector<ClCombo> combos;
    u32 atGroups = 0;
    for (int pass = 0; pass < 2; ++pass)   // 0 = opaque, 1 = AT (keeps AT combos last)
    for (u32 g = 0; g < groupN; ++g) {
        VkBuffer vb, ib; u32 stride; VkIndexType iType; u32 eOff, eCnt; bool at;
        const WorldMaterial* mat = nullptr; u32 tcOffset = 0;
        if (!WorldGPU::GetGroupShadow(g, vb, ib, stride, iType, eOff, eCnt, at, &mat, &tcOffset)) continue;
        if (eCnt == 0 || at != (pass == 1)) continue;
        if (at) {
            if (!atOn || !mat || mat->set == VK_NULL_HANDLE) continue;
            ++atGroups;
        }
        const WorldMaterial* keyMat = at ? mat : nullptr;   // opaque combos stay material-agnostic
        u32 ci = ~0u;
        for (u32 i = 0; i < (u32)combos.size(); ++i)
            if (combos[i].vb == vb && combos[i].ib == ib && combos[i].stride == stride
                && combos[i].iType == iType && combos[i].mat == keyMat) { ci = i; break; }
        if (ci == ~0u) {
            ci = (u32)combos.size();
            combos.push_back({ vb, ib, stride, iType, 0u, 0u, keyMat, at ? tcOffset : 0u });
        }
        comboOf[g] = ci;
    }
    if (combos.empty()) return false;
    if (combos.size() > kMaxCombos) {
        Msg("![VK VSM] cluster candidates: %zu buffer combos > %u - per-mesh bin active", combos.size(), kMaxCombos);
        return false;
    }

    // The candidate walk. flags bit0 = alpha-tested: those entries participate
    // only when their group earned an AT combo above (r_vsm_at); otherwise the
    // legacy behaviour — AT statics don't cast into the static atlas.
    xr_vector<u32> cand; cand.reserve(meta.size() / 8);
    for (u32 e = 0; e < (u32)meta.size(); ++e) {
        const WorldGPU::GpuMeshMeta& m = meta[e];
        if ((m.flags & 1u) && !atOn) continue;
        if (m.parentError <= errB0) continue;
        if (m.group >= groupN || comboOf[m.group] == ~0u) continue;
        cand.push_back((comboOf[m.group] << 20u) | e);
        combos[comboOf[m.group]].count++;
    }
    if (cand.empty()) return false;

    // Per-combo cmd-region bases (prefix over candidate counts).
    xr_vector<u32> comboBase(kMaxCombos, 0u);
    u32 base = 0;
    for (u32 i = 0; i < (u32)combos.size(); ++i) { combos[i].base = base; comboBase[i] = base; base += combos[i].count; }

    VK::Vram::Scope _vram_scope("VSM");
    s_clCandIdx = xr_new<CVulkanBuffer>();
    s_clCandIdx->Create((VkDeviceSize)cand.size() * sizeof(u32),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    s_clCandIdx->Upload(cand.data(), (VkDeviceSize)cand.size() * sizeof(u32));
    s_clComboBase = xr_new<CVulkanBuffer>();
    s_clComboBase->Create((VkDeviceSize)kMaxCombos * sizeof(u32),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    s_clComboBase->Upload(comboBase.data(), kMaxCombos * sizeof(u32));
    // The cluster path's own cmd stream, sized exactly to the candidate count
    // (the shared s_vsmIndirect stays per-mesh-sized; region capacity == the
    // combo's candidate count, and an entry emits at most one cmd).
    s_clIndirect = xr_new<CVulkanBuffer>();
    s_clIndirect->Create((VkDeviceSize)cand.size() * sizeof(VkDrawIndexedIndirectCommand),
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                         VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);

    s_clCombos    = combos;
    s_clCandCount = (u32)cand.size();
    s_clCandOk    = true;
    Msg("[VK VSM] cluster candidates: %u of %u entries, %u combos (errB0=%.3fm, AT %s: %u groups)",
        s_clCandCount, WorldGPU::EntryCount(), (u32)combos.size(), errB0,
        atOn ? "on" : "off", atGroups);
    return true;
}

void MarkPages(VkCommandBuffer cmd, VkImageView sceneDepth, VkExtent2D screen, const Fmatrix& viewProj)
{
    if (!Enabled() || sceneDepth == VK_NULL_HANDLE || screen.width == 0 || screen.height == 0) return;
    // (cadence: static parts below run every frame; the dynamic ones are gated on !s_dynSkip)
    const u32 cur = s_curSlot;

    // Phase 3: pick this frame's opaque-static bin/render path. Snapshotted so
    // RenderAtlas reads the SAME decision (a live cvar flip between the two
    // dispatches would draw one path's indirect data with the other's regions).
    // Page content depends on the path + the error budget → any change
    // invalidates the whole toroidal cache (one full re-render, like level load).
    const float clusterK = _max(0.1f, ps_r_vsm_cluster_lod);
    const float errB0 = ps_r_vsm_base / float(kVirtualRes) * clusterK;   // finest level's error budget (m)
    // Lazy candidate-table rebuild: level reload (build stamp) or errB(0) change
    // (k / clipmap base). Failure leaves s_clCandOk false → per-mesh bin.
    if (ps_r_vsm_cluster && s_binClPipe != VK_NULL_HANDLE
        && WorldGPU::Built() && WorldGPU::MetaBuffer() != VK_NULL_HANDLE
        && (s_clCandStamp != WorldGPU::BuildStamp() || !fsimilar(s_clCandErrB0, errB0)
            || s_clCandAT != (ps_r_vsm_at != 0)))   // r_vsm_at flip → AT combos enter/leave the table
        RebuildClusterCandidates(errB0);
    s_clusterFrame = ps_r_vsm_cluster && s_binClPipe != VK_NULL_HANDLE
        && WorldGPU::Built() && WorldGPU::MetaBuffer() != VK_NULL_HANDLE
        && s_clCandOk;
    {
        static bool s_warnedCl = false;
        if (ps_r_vsm_cluster && !s_clusterFrame && WorldGPU::Built() && !s_warnedCl) {
            s_warnedCl = true;
            Msg("![VK VSM] r_vsm_cluster unavailable (entries=%u candidates=%u pipe=%d) - per-mesh bin active",
                WorldGPU::EntryCount(), s_clCandCount, s_binClPipe != VK_NULL_HANDLE);
        }
    }
    if (s_clusterFrame != s_lastClusterMode || (s_clusterFrame && !fsimilar(clusterK, s_lastClusterK))
        || (s_clusterFrame && s_clCandAT != s_lastAtMode))   // AT casters entered/left → cached pages are stale
        InvalidateCache();   // s_physInit=false → the fill below restarts the toroidal cache
    s_lastClusterMode = s_clusterFrame;
    s_lastClusterK    = clusterK;
    s_lastAtMode      = s_clCandAT;

    // Collect this frame's NPC leaves (CPU) + upload their bin meta to this slot. Dyn → skip
    // on cadence-skip frames (the frozen dyn atlas keeps last update's NPC shadows).
    if (!s_dynSkip) CollectSkinned(cur);

    // ---- Descriptor set for this frame.
    VkDescriptorImageInfo di{ s_depthSampler, sceneDepth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkDescriptorBufferInfo ui{ s_ubo[cur]->GetHandle(), 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo ni{ s_needed->GetHandle(),   0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo ci{ s_counter->GetHandle(),  0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo ri{ s_rmask->GetHandle(),    0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo hi{ s_pageHits->GetHandle(), 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet w[6]{};
    for (u32 i = 0; i < 6; ++i) { w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = s_set[cur]; w[i].dstBinding = i; w[i].descriptorCount = 1; }
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[0].pImageInfo  = &di;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;         w[1].pBufferInfo = &ui;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;         w[2].pBufferInfo = &ni;
    w[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;         w[3].pBufferInfo = &ci;
    w[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;         w[4].pBufferInfo = &ri;
    w[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;         w[5].pBufferInfo = &hi;
    vkUpdateDescriptorSets(VulkanHW.m_Device, 6, w, 0, nullptr);

    // ---- Clear per-frame flags/counters, then MARK, then RESIDENCY (static) + ALLOC (dynamic).
    // physTile is PERSISTENT (the toroidal cache) — filled to EMPTY only once after create.
    // WAR guard: the PREVIOUS frame's resolve (compute) reads several of these buffers
    // (dynPageTable/dynUsed), the page-VS reads the grass pair arena (VERTEX) and its
    // indirect fetch (DRAW_INDIRECT) — nothing else orders those reads against this
    // frame's fills on the same queue — execution-order the transfers after them.
    const int zMark = VK::Prof::ZoneBegin(cmd, "VSM/Mark");
    // (+TRANSFER_READ: the shadow-HZB slotDirty→slotDirtyPrev copy below reads last frame's resid write.)
    MemBarrier(cmd, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                   VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT);
    vkCmdFillBuffer(cmd, s_needed->GetHandle(),          0, VK_WHOLE_SIZE, 0u);
    vkCmdFillBuffer(cmd, s_counter->GetHandle(),         0, VK_WHOLE_SIZE, 0u);
    if (ps_r_vsm_rmask) vkCmdFillBuffer(cmd, s_rmask->GetHandle(), 0, VK_WHOLE_SIZE, 0u);   // receiver mask reset
    if (ps_r_vsm_gaze) vkCmdFillBuffer(cmd, s_pageHits->GetHandle(), 0, VK_WHOLE_SIZE, 0u); // gaze hit counts reset
    if (!s_dynSkip) vkCmdFillBuffer(cmd, s_dynPageTable->GetHandle(), 0, VK_WHOLE_SIZE, 0xFFFFFFFFu);  // UNMAPPED (frozen on dyn-skip)
    if (!s_dynSkip) vkCmdFillBuffer(cmd, s_dynAllocInfo->GetHandle(), 0, VK_WHOLE_SIZE, 0u);
    vkCmdFillBuffer(cmd, s_vsmGroupCount->GetHandle(),   0, VK_WHOLE_SIZE, 0u);
    if (s_clCount) vkCmdFillBuffer(cmd, s_clCount->GetHandle(), 0, VK_WHOLE_SIZE, 0u);   // Phase 3 cluster-group counts
    vkCmdFillBuffer(cmd, s_binStats->GetHandle(),        0, VK_WHOLE_SIZE, 0u);
    vkCmdFillBuffer(cmd, s_skinStats->GetHandle(),       0, VK_WHOLE_SIZE, 0u);
    if (!s_dynSkip) vkCmdFillBuffer(cmd, s_dynPageUsed->GetHandle(), 0, VK_WHOLE_SIZE, 0u); // dyn has-caster flags reset (frozen on dyn-skip)
    // shadow-HZB: keep LAST frame's dirty set (those slots were re-rendered → their cached depth
    // changed → the reduce refreshes exactly them) before resetting the live one. WAR barrier:
    // the fill below overwrites the copy's source — execution-order transfer read before write.
    if (ps_r_vsm_hzb && s_slotDirtyPrev) {
        VkBufferCopy dc{ 0, 0, (VkDeviceSize)kMaxPhysS * sizeof(u32) };
        vkCmdCopyBuffer(cmd, s_slotDirty->GetHandle(), s_slotDirtyPrev->GetHandle(), 1, &dc);
        MemBarrier(cmd, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    }
    vkCmdFillBuffer(cmd, s_slotDirty->GetHandle(),       0, VK_WHOLE_SIZE, 0u);            // dirty set reset
    if (ps_r_vsm_hzb) vkCmdFillBuffer(cmd, s_priorValid->GetHandle(), 0, VK_WHOLE_SIZE, 0u);   // shadow-HZB: priorValid==1 means resident+valid THIS frame
    vkCmdFillBuffer(cmd, s_drawClear->GetHandle(),       0, VK_WHOLE_SIZE, 0u);            // clear draw: instanceCount=0,...
    vkCmdFillBuffer(cmd, s_drawClear->GetHandle(),       0, sizeof(u32),  6u);            // ...vertexCount=6 (the quad)
    vkCmdFillBuffer(cmd, s_dirtyList->GetHandle(), (VkDeviceSize)kMaxPhysS * sizeof(u32), 2 * sizeof(u32), 0u);   // tail = wrong-tile + gaze counters reset
    if (!s_physInit) { vkCmdFillBuffer(cmd, s_physTile->GetHandle(), 0, VK_WHOLE_SIZE, 0xFFFFFFFFu); s_physInit = true; }   // toroidal cache starts empty
    // Grass pair counters/templates + tree bin stats/meshlet counters — batched into
    // THIS fill block so the stage-1 bins later need NO per-bin fills/barriers.
    // UNCONDITIONAL (not gated on the dyn cadence): the STATIC grass hybrid bins/draws
    // every frame — sun round-robin dirties static pages even while the dyn atlas is
    // frozen. Clearing the dyn counters on a skip frame is harmless (nothing reads them).
    GrassBinPrepare(cmd);
    if (RImplementation.Trees && RImplementation.Trees->IsBuilt())
        RImplementation.Trees->VsmBinClears(cmd);
    MemBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    // MARK: which pages do visible pixels need?
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_layout, 0, 1, &s_set[cur], 0, nullptr);
    MarkPush pc{};
    pc.invViewProj.invert_44(viewProj);   // FULL 4x4 inverse — viewProj is projective (Fmatrix::invert is affine-only 4x3 → garbage)
    pc.screen[0] = (float)screen.width; pc.screen[1] = (float)screen.height;
    pc.screen[2] = 1.0f / (float)screen.width; pc.screen[3] = 1.0f / (float)screen.height;
    const u32 markStep = ps_r_vsm_mark_half ? 2u : 1u;
    pc.markStep = markStep;
    pc.lodBias  = s_lodBias;   // throttle: mark N levels coarser (receivers walk to the finest mapped)
    pc.rmaskOn  = ps_r_vsm_rmask ? 1u : 0u;   // receiver mask: record sampled 8×8 cells too
    pc.gazeOn   = (ps_r_vsm_gaze && ps_r_vsm_cache) ? 1u : 0u;   // gaze refresh: count samples per page (pointless with the cache off)
    vkCmdPushConstants(cmd, s_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    const u32 mw = (screen.width  + markStep - 1) / markStep;
    const u32 mh = (screen.height + markStep - 1) / markStep;
    vkCmdDispatch(cmd, (mw + 7) / 8, (mh + 7) / 8, 1);
    VK::Prof::ZoneEnd(cmd, zMark);

    // needed[] write (mark) → read (alloc)
    MemBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    const int zResid = VK::Prof::ZoneBegin(cmd, "VSM/Resid");

    // RESIDENCY (static, toroidal): map each needed page to its fixed slot + flag dirty pages
    // (slot holds a different tile, or a round-robin refresh is due). forceDirty when cache is
    // OFF → mark every visible page dirty = render-all baseline for A/B.
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_residPipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_residLayout, 0, 1, &s_residSet, 0, nullptr);
    ResidPush rp{};
    for (u32 L = 0; L < kLevels; ++L) { rp.pageBase[2 * L] = s_pageBase[L][0]; rp.pageBase[2 * L + 1] = s_pageBase[L][1]; }
    rp.frame = s_frame; rp.refreshN = (u32)ps_r_vsm_cache_refresh;
    rp.sunMoving = s_sunMoving ? 1u : 0u; rp.forceDirty = (ps_r_vsm_cache != 0) ? 0u : 1u;
    // Tree wind-hybrid transitions → invalidation circles: static pages a boundary-crossing
    // tree overlaps re-render THIS frame (its rigid shadow appears/disappears without ghosts).
    rp.inval[3] = ps_r_vsm_base / float(kPagesAxis);   // L0 page width (m); pw(L) = this * 2^L
    rp.inval[7] = (float)(ps_r_vsm_dirty_budget < 0 ? 0 : ps_r_vsm_dirty_budget);   // [1].w = wrong-tile budget/frame (0 = unlimited)
    rp.inval[11] = (ps_r_vsm_gaze && ps_r_vsm_cache) ? (float)_max(0, ps_r_vsm_gaze_pages) : 0.f;   // [2].w = gaze budget (0 = off)
    rp.inval[15] = (float)_max(1, ps_r_vsm_gaze_px);                                                // [3].w = full-rate hits threshold
    // Near-set + static↔dyn transition circles: freeze on dyn-skip so no tree flips
    // static↔dyn while the dyn atlas is frozen (a flip would double- or lose-shadow it).
    if (!s_dynSkip && RImplementation.Trees && RImplementation.Trees->IsBuilt()) {
        RImplementation.Trees->VsmUpdateNearSet(s_sunView);
        Fvector4 sph[4]; const u32 nInv = RImplementation.Trees->VsmPopTransitions(sph, 4);
        for (u32 i = 0; i < nInv; ++i) {
            Fvector l; s_sunView.transform_tiny(l, Fvector{ sph[i].x, sph[i].y, sph[i].z });
            rp.inval[i * 4 + 0] = l.x; rp.inval[i * 4 + 1] = l.y; rp.inval[i * 4 + 2] = sph[i].w;
        }
    }
    vkCmdPushConstants(cmd, s_residLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(rp), &rp);
    vkCmdDispatch(cmd, (kPageCount + 63) / 64, 1, 1);

    // shadow-HZB (r_vsm_hzb): refresh the per-slot occluder maxes (whole-page + 8×8 blocks) from the
    // PRIOR-frame atlas depth. INCREMENTAL: only slots re-rendered last frame (slotDirtyPrev) are
    // re-reduced, evicted slots self-reset to 1.0 inside the shader — cost tracks the dirty rate.
    // Runs here — the atlas still holds prior-frame depth (SHADER_READ, before RenderAtlas overwrites)
    // and residency just wrote priorValid. Consumers: tree bins (pageMax) + vsm_vox_cull (pageMaxBlk).
    {
        const bool hzbNow = ps_r_vsm_hzb && !s_atlasFirst && s_hzbPipe != VK_NULL_HANDLE;
        if (hzbNow) {
            if (!s_hzbWasOn) {   // first enabled frame (or OFF→ON flip): persistent maxes hold garbage/stale → reset
                vkCmdFillBuffer(cmd, s_pageMax->GetHandle(),    0, VK_WHOLE_SIZE, 0x3F800000u);   // 1.0f → no occlusion
                vkCmdFillBuffer(cmd, s_pageMaxBlk->GetHandle(), 0, VK_WHOLE_SIZE, 0x3F800000u);
            }
            MemBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_hzbPipe);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_hzbLayout, 0, 1, &s_hzbSet, 0, nullptr);
            vkCmdDispatch(cmd, kMaxPhysS, 1, 1);
        }
        s_hzbWasOn = hzbNow;
    }

    // ALLOC (dynamic): demand-allocate a slot per needed page into the dynamic table.
    // Skipped on dyn-skip frames — the frozen dyn page table stays valid (world-anchored).
    if (!s_dynSkip) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_allocPipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_allocLayout, 0, 1, &s_dynAllocSet, 0, nullptr);
        vkCmdDispatch(cmd, (kPageCount + 63) / 64, 1, 1);
    }
    VK::Prof::ZoneEnd(cmd, zResid);
    const int zBins = VK::Prof::ZoneBegin(cmd, "VSM/Bins");
    // Per-dispatch sub-zones (r_profiler 2): VSM/Bins is ~8 different dispatches in one
    // interval — attribution for the hzb/vox investigations lives here. Depth-1 zones
    // never sum into gpu_total, and the default (r_profiler 1) zone list is unchanged.
    const bool binSub = ps_r_profiler > 1;

    // ONE barrier for ALL stage-1 bins: residency's pageTable/slotDirty + the dynamic alloc
    // before every bin's read. The five stage-1 bins (opaque static, skinned, grass, tree
    // static, tree dyn) write DISJOINT buffers — the only shared target is dynUsed, and
    // that's atomicOr (commutative) — so nothing orders them against EACH OTHER; the old
    // per-bin barriers (one full compute→compute drain each) serialized the whole VSM/Bins
    // zone for no correctness gain.
    MemBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    // BIN (static): scatter each opaque static caster into the DIRTY static pages it overlaps.
    // Only dirty pages get caster draws (cached pages keep their depth); the residency set the
    // dirty flags. Phase 3 (s_clusterFrame): the casters are the WorldGPU cluster-LOD entries
    // and each clipmap level bins the DAG cut whose world error fits its texel budget —
    // camera-independent, so the cut adds ZERO cache invalidations.
    const VkBuffer metaBuf = VK::ShadowGPU::GetMetaBuffer();
    u32            casterN = VK::ShadowGPU::CasterCount();
    const u32      groupN  = VK::ShadowGPU::GroupCount();
    const int zbWorld = binSub ? VK::Prof::ZoneBegin(cmd, "Bins/World") : -1;
    if (s_clusterFrame) {
        VkDescriptorBufferInfo bi[12] = {
            { WorldGPU::MetaBuffer(),               0, VK_WHOLE_SIZE },
            { s_ubo[cur]->GetHandle(),              0, VK_WHOLE_SIZE },
            { s_pageTable->GetHandle(),             0, VK_WHOLE_SIZE },
            { s_casterPages->GetHandle(),           0, VK_WHOLE_SIZE },
            { s_clIndirect->GetHandle(),            0, VK_WHOLE_SIZE },   // cluster path's own cmd stream
            { s_clCount->GetHandle(),               0, VK_WHOLE_SIZE },   // per-combo cmd counts
            { s_binStats->GetHandle(),              0, VK_WHOLE_SIZE },
            { s_slotDirty->GetHandle(),             0, VK_WHOLE_SIZE },
            { s_clComboBase->GetHandle(),           0, VK_WHOLE_SIZE },   // per-combo cmd-region bases
            { ClusterStream::BitsBuffer(),          0, VK_WHOLE_SIZE },   // Stage B residency
            { ClusterStream::SlotBaseBuffer(),      0, VK_WHOLE_SIZE },
            { s_clCandIdx->GetHandle(),             0, VK_WHOLE_SIZE },   // (combo<<20)|entry
        };
        VkWriteDescriptorSet bw[12]{};
        for (u32 i = 0; i < 12; ++i) {
            bw[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; bw[i].dstSet = s_binClSet[cur]; bw[i].dstBinding = i;
            bw[i].descriptorCount = 1;
            bw[i].descriptorType = (i == 1) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bw[i].pBufferInfo = &bi[i];
        }
        vkUpdateDescriptorSets(VulkanHW.m_Device, 12, bw, 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_binClPipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_binClLayout, 0, 1, &s_binClSet[cur], 0, nullptr);
        struct { u32 candCount; u32 arenaSlots; float errK; } clPush{ s_clCandCount, kClSlots, clusterK };
        vkCmdPushConstants(cmd, s_binClLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(clPush), &clPush);
        vkCmdDispatch(cmd, (clPush.candCount + 63) / 64, 1, 1);
    }
    else if (metaBuf != VK_NULL_HANDLE && casterN > 0 && groupN > 0 && groupN <= kMaxGroups) {
        if (casterN > kMaxCasters) casterN = kMaxCasters;   // clamp to casterPages capacity
        VkDescriptorBufferInfo bi[8] = {
            { metaBuf,                         0, VK_WHOLE_SIZE },
            { s_ubo[cur]->GetHandle(),         0, VK_WHOLE_SIZE },
            { s_pageTable->GetHandle(),        0, VK_WHOLE_SIZE },
            { s_casterPages->GetHandle(),      0, VK_WHOLE_SIZE },
            { s_vsmIndirect->GetHandle(),      0, VK_WHOLE_SIZE },
            { s_vsmGroupCount->GetHandle(),    0, VK_WHOLE_SIZE },
            { s_binStats->GetHandle(),         0, VK_WHOLE_SIZE },
            { s_slotDirty->GetHandle(),        0, VK_WHOLE_SIZE },
        };
        const VkDescriptorType bt[8] = {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        };
        VkWriteDescriptorSet bw[8]{};
        for (u32 i = 0; i < 8; ++i) { bw[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; bw[i].dstSet = s_binSet[cur]; bw[i].dstBinding = i; bw[i].descriptorCount = 1; bw[i].descriptorType = bt[i]; bw[i].pBufferInfo = &bi[i]; }
        vkUpdateDescriptorSets(VulkanHW.m_Device, 8, bw, 0, nullptr);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_binPipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_binLayout, 0, 1, &s_binSet[cur], 0, nullptr);
        struct { u32 casterCount; u32 groupStride; float camX, camY, camZ, lodDist; } binPush{
            casterN, kGroupStride,
            Device.vCameraPosition.x, Device.vCameraPosition.y, Device.vCameraPosition.z,
            ps_r_vsm_lod_dist };
        vkCmdPushConstants(cmd, s_binLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(binPush), &binPush);
        vkCmdDispatch(cmd, (casterN + 63) / 64, 1, 1);
    }
    else if (metaBuf != VK_NULL_HANDLE && groupN > kMaxGroups) {
        static bool s_warned = false;
        if (!s_warned) { s_warned = true; Msg("![VK VSM] ShadowGPU groups %u > kMaxGroups %u - bump kMaxGroups", groupN, kMaxGroups); }
    }
    if (zbWorld >= 0) VK::Prof::ZoneEnd(cmd, zbWorld);

    // Skinned (NPC) casters: bin into the allocated pages (own per-leaf draw stream).
    if (!s_dynSkip && s_skinCount > 0) {
        const int zbSkin = binSub ? VK::Prof::ZoneBegin(cmd, "Bins/Skin") : -1;
        DispatchSkinnedBin(cmd, cur);
        if (zbSkin >= 0) VK::Prof::ZoneEnd(cmd, zbSkin);
    }

    // Grass casters: append (instance, page) pairs — DYN dispatch (L0, cadence-gated)
    // + STATIC dispatch (L1/L2 dirty pages, every frame). (No-op unless r_vsm_grass +
    // a detail manager with grass.)
    {
        const int zbGrass = binSub ? VK::Prof::ZoneBegin(cmd, "Bins/Grass") : -1;
        DispatchGrassBin(cmd, cur, !s_dynSkip);
        if (zbGrass >= 0) VK::Prof::ZoneEnd(cmd, zbGrass);
    }

    // Tree casters, near/far wind hybrid: FAR trees bin into the toroidal STATIC cache
    // (dirty pages only, rigid); NEAR trees (r_vsm_tree_wind_dist) bin into the DYNAMIC
    // atlas (all resident pages, re-rendered each frame WITH wind → smooth sway) and mark
    // dynUsed for the resolve gate. [Phase 1 + Phase 2 wind] Stage-1 only here — the
    // meshlet stage-2 refinement is the ONE genuine cross-dispatch dependency, so it
    // runs after the single barrier below.
    if (RImplementation.Trees && RImplementation.Trees->IsBuilt()) {
        const VkBuffer hzbMax = (ps_r_vsm_hzb && s_pageMax) ? s_pageMax->GetHandle() : VK_NULL_HANDLE;
        const int zbTreeS = binSub ? VK::Prof::ZoneBegin(cmd, "Bins/TreeS") : -1;
        RImplementation.Trees->VsmBin(cmd, s_pageTable->GetHandle(), s_slotDirty->GetHandle(), s_dynPageUsed->GetHandle(), s_ubo[cur]->GetHandle(), s_pageList->GetHandle(), hzbMax);   // static (far) trees — always
        if (zbTreeS >= 0) VK::Prof::ZoneEnd(cmd, zbTreeS);
        if (!s_dynSkip) {   // near wind crowns — throttled with the rest of the dyn pass
            // NB: this zone also absorbs the hull_cmd barrier's drain of every prior bin
            // (DispatchHullCmd records a full compute→compute barrier) — compare A/B, not
            // absolute (the drain is identical on both sides of a flag flip).
            const int zbTreeD = binSub ? VK::Prof::ZoneBegin(cmd, "Bins/TreeD") : -1;
            RImplementation.Trees->VsmBinDyn(cmd, s_dynPageTable->GetHandle(), s_dynPageUsed->GetHandle(), s_ubo[cur]->GetHandle(), s_dynPageList->GetHandle(), hzbMax, s_pageTable->GetHandle());
            if (zbTreeD >= 0) VK::Prof::ZoneEnd(cmd, zbTreeD);
        }
    }

    // Stage-2: one barrier for (a) the tree meshlet refinement reading tree stage-1
    // output and (b) the grass pair-count → indirect instanceCount copies. TRANSFER is in
    // the SRC scope too: the indirect TEMPLATE (vkCmdUpdateBuffer in the fill block) writes
    // the same instanceCount dword the copy below overwrites — transfer-transfer WAW is
    // NOT ordered by the intervening transfer→compute / compute→compute barriers, and a
    // late-landing template zero blanked whole grass types' shadows for a frame.
    {
        MemBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT);
        if (RImplementation.Trees && RImplementation.Trees->IsBuilt()) {
            // shadow-HZB for the stage-2 brick cull: block-granular occluder maxes + the static
            // page table (mode-0 Option A lookup). NULL when off/first-frame → cull skips the test.
            const VkBuffer hzbBlk = (s_hzbWasOn && s_pageMaxBlk) ? s_pageMaxBlk->GetHandle() : VK_NULL_HANDLE;
            RImplementation.Trees->VsmBinMeshlets(cmd, !s_dynSkip, s_pageList->GetHandle(), s_dynPageList->GetHandle(), s_ubo[cur]->GetHandle(),
                                                  hzbBlk, s_pageTable->GetHandle());
        }
        if (s_grassTypes > 0 && s_grassPairCnt && s_grassPairInd) {
            xr_vector<VkBufferCopy> copies(s_grassTypes);
            for (u32 i = 0; i < s_grassTypes; ++i) {
                copies[i].srcOffset = i * sizeof(u32);
                copies[i].dstOffset = i * sizeof(VkDrawIndexedIndirectCommand) + sizeof(u32);   // instanceCount @ +4
                copies[i].size      = sizeof(u32);
            }
            if (!s_dynSkip)
                vkCmdCopyBuffer(cmd, s_grassPairCnt->GetHandle(), s_grassPairInd->GetHandle(), s_grassTypes, copies.data());
            if (s_grassPairSectionS > 0)   // static hybrid ran this frame (cadence-independent)
                vkCmdCopyBuffer(cmd, s_grassPairCntS->GetHandle(), s_grassPairIndS->GetHandle(), s_grassTypes, copies.data());
        }
    }
    VK::Prof::ZoneEnd(cmd, zBins);

    // ---- Diagnostics: copy the GPU counters to the host readbacks (read stale next
    // frames, fine — the buffers were zeroed at Init). The tiny resid readback (dirty +
    // wrong-tile counts, 8 B) also runs under r_profiler so the throttle/dirty telemetry
    // line can attribute costs in ordinary perf sessions; the rest is r_vsm_debug only.
    if (ps_r_vsm_debug || ps_r_profiler > 0) {
        MemBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferCopy rdc{ sizeof(u32), 0, sizeof(u32) };                 // drawClear[1] (static dirty count) -> residRB[0]
        vkCmdCopyBuffer(cmd, s_drawClear->GetHandle(), s_residRB->GetHandle(), 1, &rdc);
        VkBufferCopy rwc{ (VkDeviceSize)kMaxPhysS * sizeof(u32), 2 * sizeof(u32), sizeof(u32) };   // dirtyList tail (wrong-tile requests) -> residRB[2]
        vkCmdCopyBuffer(cmd, s_dirtyList->GetHandle(), s_residRB->GetHandle(), 1, &rwc);
        VkBufferCopy rgz{ (VkDeviceSize)(kMaxPhysS + 1) * sizeof(u32), sizeof(u32), sizeof(u32) }; // gaze-refresh counter -> residRB[1]
        vkCmdCopyBuffer(cmd, s_dirtyList->GetHandle(), s_residRB->GetHandle(), 1, &rgz);
    }
    if (ps_r_vsm_debug || ps_r_profiler > 0) {   // pagesSeen feeds the throttle summary line too
        VkBufferCopy rc{ 0, 0, sizeof(u32) };                            // mark counter -> readback[0]
        vkCmdCopyBuffer(cmd, s_counter->GetHandle(), s_readback->GetHandle(), 1, &rc);
        VkBufferCopy ra{ 0, sizeof(u32), (VkDeviceSize)(1 + kLevels) * sizeof(u32) };   // dynAllocInfo -> readback[1..7]
        vkCmdCopyBuffer(cmd, s_dynAllocInfo->GetHandle(), s_readback->GetHandle(), 1, &ra);
        VkBufferCopy rb{ 0, 0, 8 * sizeof(u32) };                        // binStats -> binReadback[0..7]
        vkCmdCopyBuffer(cmd, s_binStats->GetHandle(), s_binReadback->GetHandle(), 1, &rb);
        VkBufferCopy rsk{ 0, 0, 4 * sizeof(u32) };                       // skinStats -> skinStatsRB[0..3]
        vkCmdCopyBuffer(cmd, s_skinStats->GetHandle(), s_skinStatsRB->GetHandle(), 1, &rsk);
        VkBufferCopy rgr{ 0, 0, 8 * sizeof(u32) };                       // grassStats -> grassStatsRB[0..7]
        vkCmdCopyBuffer(cmd, s_grassStats->GetHandle(), s_grassStatsRB->GetHandle(), 1, &rgr);
        if (s_grassPairCnt && s_grassPairCntRB) {                        // per-type pair counts (diag)
            VkBufferCopy rpc{ 0, 0, (VkDeviceSize)kGrassMaxTypes * sizeof(u32) };
            vkCmdCopyBuffer(cmd, s_grassPairCnt->GetHandle(), s_grassPairCntRB->GetHandle(), 1, &rpc);
        }
    }

    // ---- Throttle/dirty TELEMETRY (r_vsm_debug or r_profiler). Per-frame accumulate from
    // the resid readback (stale by the in-flight depth — fine for windows), then a 3-second
    // summary line: cost (avg/max/over-budget%), time share per bias level, dirty/wrong page
    // rates and the number of redraws the budget actually pushed to later frames.
    if ((ps_r_vsm_debug || ps_r_profiler > 0) && s_residPtr) {
        const u32 dirtyN = s_residPtr[0], wrongN = s_residPtr[2], gazeN = s_residPtr[1];
        s_dirtySum += dirtyN; s_wrongSum += wrongN; s_gazeSum += gazeN; s_rbFrames++;
        if (dirtyN > s_dirtyMax) s_dirtyMax = dirtyN;
        if (wrongN > s_wrongMax) s_wrongMax = wrongN;
        if (gazeN > s_gazeMax) s_gazeMax = gazeN;
        if (ps_r_vsm_dirty_budget > 0 && wrongN > (u32)ps_r_vsm_dirty_budget)
            s_deferSum += wrongN - (u32)ps_r_vsm_dirty_budget;
        if (Device.dwTimeGlobal > s_thLastLogMs + 3000) {
            s_thLastLogMs = Device.dwTimeGlobal;
            const u32 tf = s_thFrames ? s_thFrames : 1;
            const u32 bf = s_thBiasHist[0] + s_thBiasHist[1] + s_thBiasHist[2];
            const u32 bd = bf ? bf : 1;
            const u32 rf = s_rbFrames ? s_rbFrames : 1;
            // pagesSeen = pages visible pixels sampled this frame (the mark counter) — the
            // "how much shadow the eye actually sees" anchor every other count compares to:
            // dirty/gaze are re-RENDERED subsets of it, everything else stays cached.
            Msg("[VK VSM] throttle[%s]: VSMrender avg=%.2f max=%.2f ms, over-budget %u%% (budget=%.1f) | ctl=%.2f bias=%u, time L+0=%u%% L+1=%u%% L+2+=%u%% | pagesSeen=%u | dirty avg=%u max=%u | wrong avg=%u max=%u deferred=%u (budget=%d%s) | gaze avg=%u max=%u (budget=%d)",
                ps_r_vsm_throttle ? "ON" : "OFF",
                s_thCostSum / (float)tf, s_thCostMax, s_thOver * 100u / tf, ps_r_vsm_throttle_budget,
                s_throttleVal, s_lodBias,
                s_thBiasHist[0] * 100u / bd, s_thBiasHist[1] * 100u / bd, s_thBiasHist[2] * 100u / bd,
                s_readPtr ? s_readPtr[0] : 0u,
                (u32)(s_dirtySum / rf), s_dirtyMax, (u32)(s_wrongSum / rf), s_wrongMax, (u32)s_deferSum,
                ps_r_vsm_dirty_budget, ps_r_vsm_dirty_budget ? "" : "=OFF",
                (u32)(s_gazeSum / rf), s_gazeMax, ps_r_vsm_gaze ? ps_r_vsm_gaze_pages : 0);
            s_thCostSum = 0.f; s_thCostMax = 0.f; s_thFrames = 0; s_thOver = 0;
            s_thBiasHist[0] = s_thBiasHist[1] = s_thBiasHist[2] = 0;
            s_wrongSum = s_dirtySum = s_deferSum = 0; s_wrongMax = s_dirtyMax = 0; s_gazeSum = 0; s_gazeMax = 0; s_rbFrames = 0;
        }
    }

    if (ps_r_vsm_debug && s_readPtr && Device.dwTimeGlobal > s_lastLog + 2000) {
        s_lastLog = Device.dwTimeGlobal;
        const u32 dirty = s_residPtr ? s_residPtr[0] : 0u;
        const u32 wrongReq = s_residPtr ? s_residPtr[2] : 0u;   // scrolled-in pages wanting a redraw (pre-budget)
        const int dBudget = ps_r_vsm_dirty_budget;
        Msg("[VK VSM] static: dirty=%u/%u rendered | wrong=%u%s budget=%d | bias=%u(%.2f) | mark=%u | mode=%s sun=%s | dyn demand=%u/%u",
            dirty, kMaxPhysS, wrongReq,
            (dBudget > 0 && wrongReq > (u32)dBudget) ? " [DEFERRING]" : "", dBudget,
            s_lodBias, s_throttleVal, s_readPtr[0], (ps_r_vsm_cache ? "cache" : "render-all"),
            (s_sunMoving ? "moving" : "static"), s_readPtr[1], kMaxPhys);
        if (s_binReadPtr) {
            if (s_clusterFrame)
                Msg("[VK VSM] bin[cluster]: draws=%u instances=%u maxPagesPerEntry=%u arena=%u/%u%s lodCulled=%u (cand=%u of %u, k=%.2f)",
                    s_binReadPtr[0], s_binReadPtr[1], s_binReadPtr[2], s_binReadPtr[4], kClSlots,
                    s_binReadPtr[5] ? " [ARENA FULL!]" : "", s_binReadPtr[3],
                    s_clCandCount, WorldGPU::EntryCount(), s_lastClusterK);
            else
                Msg("[VK VSM] bin: draws=%u instances=%u maxPagesPerCaster=%u/%u groupOverflow=%u (casters=%u)",
                    s_binReadPtr[0], s_binReadPtr[1], s_binReadPtr[2], kPagesCap, s_binReadPtr[3], casterN);
        }
        if (s_skinStatsPtr)
            Msg("[VK VSM] skinned: leaves=%u draws=%u instances=%u maxPages=%u/%u",
                s_skinCount, s_skinStatsPtr[0], s_skinStatsPtr[1], s_skinStatsPtr[2], kSkinnedCap);
        if (s_grassStatsPtr && ps_r_vsm_grass)
            Msg("[VK VSM] grass: casting instances=%u/%u candidates droppedPairs=%u%s pairsDyn=%u pairsStatic=%u (types=%u, sections=%u/%u, near<=%.0fm, hybrid=%d)",
                s_grassStatsPtr[0], s_grassStatsPtr[4], s_grassStatsPtr[1], s_grassStatsPtr[1] ? " [TRUNCATED!]" : "",
                s_grassStatsPtr[2], s_grassStatsPtr[3],
                s_grassTypes, s_grassPairSection, s_grassPairSectionS, ps_r_vsm_grass_dist, ps_r_vsm_grass_static);
            if (s_readPtr)   // dyn page residency by level (allocInfo[1..6] → readback[2..7])
                Msg("[VK VSM] dyn pages by level: L0=%u L1=%u L2=%u L3=%u L4=%u L5=%u",
                    s_readPtr[2], s_readPtr[3], s_readPtr[4], s_readPtr[5], s_readPtr[6], s_readPtr[7]);
            // Per-type audit: pair counts (GPU) + CPU render-side validity. A type with
            // pairs>0 but mesh=0/dif=0 has casters that silently DON'T draw into the
            // atlas — "this bush has a shadow, the identical-looking one next to it
            // (different detail type) doesn't".
            if (s_grassPairCntPtr && RImplementation.Details) {
                CDetailManager* dmD = RImplementation.Details;
                string2048 line; line[0] = 0;
                for (u32 i = 0; i < s_grassTypes; ++i) {
                    VkBuffer mvb, mib; u32 ic = 0;
                    const bool mesh = dmD->Vsm_TypeMesh(i, mvb, mib, ic) && ic > 0;
                    const bool dif  = dmD->Vsm_TypeDiffuseSet(i) != VK_NULL_HANDLE;
                    string64 t;
                    xr_sprintf(t, "%u:%u%s ", i, s_grassPairCntPtr[i], (!mesh || !dif) ? (!mesh ? "[NOMESH]" : "[NODIF]") : "");
                    xr_strcat(line, t);
                }
                Msg("[VK VSM] grass pairs/type: %s", line);
            }
    }
}

VkImageView GetAtlasView() { return s_atlasView; }
VkSampler   GetSampler()   { return s_atlasSampler; }
VkImageView GetDynAtlasView()       { return s_dynView; }
VkBuffer    GetDynPageTableHandle() { return s_dynPageTable ? s_dynPageTable->GetHandle() : VK_NULL_HANDLE; }
VkBuffer    GetDynUsedHandle()      { return s_dynPageUsed ? s_dynPageUsed->GetHandle() : VK_NULL_HANDLE; }
bool        AtlasReady()   { return s_atlasView != VK_NULL_HANDLE && s_dynView != VK_NULL_HANDLE && !s_atlasFirst && !s_dynFirst; }

void RenderAtlas(VkCommandBuffer cmd)
{
    if (!Enabled() || s_atlasView == VK_NULL_HANDLE || s_dynView == VK_NULL_HANDLE) return;
    // (cadence: the static pass always runs; the dynamic block below is gated on !s_dynSkip)
    // Phase 3: the opaque static casters come from whichever path THIS frame's
    // MarkPages binned (s_clusterFrame snapshot) — WorldGPU cluster groups or
    // the per-mesh ShadowGPU groups.
    // The per-mesh static source can be ABSENT while everything else still has
    // casters: ShadowGPU's set is empty when WorldGPU::CompactPools repacked every
    // static, and a host-driven editor scene never builds one at all. Gate only the
    // per-mesh loop below — returning here would also skip the trees, the grass and
    // the whole DYNAMIC atlas (NPC / grass / near-tree crowns), none of which read
    // ShadowGPU, so the frame would silently lose every VSM shadow instead of just
    // the static ones. Condition mirrors the MarkPages bin exactly: that pass
    // decides, and this one must follow the same frame's decision.
    u32 groupN = 0;
    if (s_clusterFrame)
        groupN = (u32)s_clCombos.size();
    else {
        const u32 gn = VK::ShadowGPU::GroupCount();
        if (VK::ShadowGPU::GetMetaBuffer() != VK_NULL_HANDLE && VK::ShadowGPU::CasterCount() > 0 && gn > 0 && gn <= kMaxGroups)
            groupN = gn;
    }
    const u32 cur = s_curSlot;

    // bin's writes (vsmIndirect / casterPages / pageList / grass pairs) + residency (dirtyList /
    // drawClear) + the grass pairCount→indirect TRANSFER copy → render reads (vertex + indirect).
    MemBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT);

    // Begin an atlas depth pass, setting viewport/scissor/bias the page draws expect (per-page
    // routing via gl_Position). loadOp LOAD preserves the toroidal cache; CLEAR for the dynamic
    // atlas (and the static atlas's very first frame).
    auto beginAtlas = [&](VkImageView view, VkAttachmentLoadOp loadOp, u32 w, u32 h) {
        VkRenderingAttachmentInfo dAtt{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        dAtt.imageView = view; dAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        dAtt.loadOp = loadOp; dAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        dAtt.clearValue.depthStencil = { 1.0f, 0 };
        VkRenderingInfo ri{ VK_STRUCTURE_TYPE_RENDERING_INFO };
        ri.renderArea.extent = { w, h }; ri.layerCount = 1; ri.colorAttachmentCount = 0; ri.pDepthAttachment = &dAtt;
        vkCmdBeginRendering(cmd, &ri);
        VkViewport vp{ 0.f, 0.f, (float)w, (float)h, 0.f, 1.f };
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc{ {0, 0}, { w, h } };
        vkCmdSetScissor(cmd, 0, 1, &sc);
        // Write-side raster bias (r_vsm_raster_bias/_slope, D16 units ≈ 3 cm each).
        // Kept SMALL: every unit here is depth the sun can PUNCH THROUGH thin
        // geometry before any receiver even gets a vote (the old 1.5/2.5 ≈ 5+ cm
        // was most of a plank wall's thickness). Receiver acne is handled by the
        // resolve's normal-offset + its own small bias, not by pushing casters deep.
        vkCmdSetDepthBias(cmd, ps_r_vsm_raster_bias, 0.f, ps_r_vsm_raster_slope);
    };
    // Depth attachment → SHADER_READ, visible to COMPUTE (the resolve samples it) as well
    // as FRAGMENT (the auto-derived ImageBarrier only targets FRAGMENT — too narrow now).
    auto atlasToRead = [&](VkImage img) {
        VkImageMemoryBarrier ab{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        ab.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        ab.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        ab.oldLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        ab.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        ab.srcQueueFamilyIndex = ab.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ab.image = img;
        ab.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &ab);
    };

    // ===== STATIC atlas (toroidal cache): clear DIRTY pages, then draw casters into them only. =====
    // loadOp LOAD keeps cached pages; the clear-dirty quad resets just the dirty cells; the bin
    // already filtered each caster's instance list to dirty pages → cached pages are untouched.
    // Sub-zones split World/VSMrender into STATIC (cached, only dirty pages) vs the three
    // per-frame DYNAMIC contributors (NPC / grass / near-trees) — all opened every frame in
    // fixed order so zone indices stay stable (see [[vulkan-profiler-monitoring]]).
    const int zStatic = VK::Prof::ZoneBegin(cmd, "VSM/Static");
    const u32 sw = kAtlasW_S * kPageSize, sh = kAtlasH_S * kPageSize;
    ImageBarrier(cmd, s_atlasImage, s_atlasFirst ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
    const VkAttachmentLoadOp sLoad = s_atlasFirst ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
    s_atlasFirst = false;

    VkDescriptorBufferInfo ri3[3] = {
        { s_pageList->GetHandle(),    0, VK_WHOLE_SIZE },
        { s_casterPages->GetHandle(), 0, VK_WHOLE_SIZE },
        { s_ubo[cur]->GetHandle(),    0, VK_WHOLE_SIZE },
    };
    const VkDescriptorType rt[3] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER };
    VkWriteDescriptorSet rw[3]{};
    for (u32 i = 0; i < 3; ++i) { rw[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; rw[i].dstSet = s_renderSet[cur]; rw[i].dstBinding = i; rw[i].descriptorCount = 1; rw[i].descriptorType = rt[i]; rw[i].pBufferInfo = &ri3[i]; }
    vkUpdateDescriptorSets(VulkanHW.m_Device, 3, rw, 0, nullptr);

    beginAtlas(s_atlasView, sLoad, sw, sh);
    // Clear dirty pages to 1.0 (depth-only instanced quad; instanceCount = dirty count, indirect).
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_clearPipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_clearLayout, 0, 1, &s_clearSet, 0, nullptr);
    vkCmdDrawIndirect(cmd, s_drawClear->GetHandle(), 0, 1, sizeof(VkDrawIndirectCommand));
    // Re-arm the write-side bias after the clear. Belt and braces: the clear pipeline
    // now declares DEPTH_BIAS dynamic so it no longer clobbers it, but this pass mixes
    // pipelines from three files (page/AT here, trees in vk_TreeManager_Render, grass
    // below) and one of them regressing to a static bias would silently reopen the
    // acne. Setting it here costs nothing and makes the guarantee local to the draws.
    vkCmdSetDepthBias(cmd, ps_r_vsm_raster_bias, 0.f, ps_r_vsm_raster_slope);
    // Caster draws (only dirty pages, per the bin filter).
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_renderLayout, 0, 1, &s_renderSet[cur], 0, nullptr);
    VkPipeline lastPipe = VK_NULL_HANDLE;
    if (s_clusterFrame) {
        // Cluster path: one indirect-count draw per BUFFER-COMBO (exact prefix
        // cmd regions over the combo's candidate count, counts in s_clCount).
        // After VB paging the repacked groups all share the cluster pools, so
        // this is a handful of draws even at 5k+ material groups. vsm_page.vert
        // reads only the position, so the cluster IBs draw through the same
        // page pipelines.
        VkBuffer lastVB = VK_NULL_HANDLE, lastIB = VK_NULL_HANDLE;
        VkIndexType lastIType = VK_INDEX_TYPE_MAX_ENUM;
        const WorldMaterial* lastMat = nullptr;
        for (u32 g = 0; g < groupN; ++g) {
            const ClCombo& cb = s_clCombos[g];
            if (cb.count == 0) continue;
            const bool at = cb.mat != nullptr;   // AT combos are sorted after every opaque one
            VkPipeline pipe = at ? GetPageATPipeline(cb.stride, cb.tcOffset) : GetPagePipeline(cb.stride);
            if (pipe == VK_NULL_HANDLE) continue;
            if (at && cb.mat->set == VK_NULL_HANDLE) continue;
            if (pipe != lastPipe) { vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe); lastPipe = pipe; }
            if (at && cb.mat != lastMat) {
                // AT layout has push constants the opaque layout lacks → set 0
                // compatibility breaks at the first AT combo; rebind both sets
                // (0 = page routing, 1 = the material's diffuse) per material.
                VkDescriptorSet sets[2] = { s_renderSet[cur], cb.mat->set };
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_renderATLayout, 0, 2, sets, 0, nullptr);
                PageATPush pcAT{ { 1.0f / 1024.0f, 1.0f / 1024.0f }, cb.mat->alphaRef, 0.f };
                vkCmdPushConstants(cmd, s_renderATLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   0, sizeof(pcAT), &pcAT);
                lastMat = cb.mat;
            }
            if (cb.vb != lastVB) { VkDeviceSize z = 0; vkCmdBindVertexBuffers(cmd, 0, 1, &cb.vb, &z); lastVB = cb.vb; }
            if (cb.ib != lastIB || cb.iType != lastIType) { vkCmdBindIndexBuffer(cmd, cb.ib, 0, cb.iType); lastIB = cb.ib; lastIType = cb.iType; }
            const VkDeviceSize cmdOff = (VkDeviceSize)cb.base * sizeof(VkDrawIndexedIndirectCommand);
            const VkDeviceSize cntOff = (VkDeviceSize)g * sizeof(u32);
            vkCmdDrawIndexedIndirectCount(cmd, s_clIndirect->GetHandle(), cmdOff, s_clCount->GetHandle(), cntOff,
                                          cb.count, sizeof(VkDrawIndexedIndirectCommand));
        }
    }
    else for (u32 g = 0; g < groupN; ++g) {
        VkBuffer vb, ib; u32 stride; VkIndexType iType;
        if (!VK::ShadowGPU::GetGroupBind(g, vb, ib, stride, iType)) continue;
        VkPipeline pipe = GetPagePipeline(stride);
        if (pipe == VK_NULL_HANDLE) continue;
        if (pipe != lastPipe) { vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe); lastPipe = pipe; }
        VkDeviceSize z = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &z);
        vkCmdBindIndexBuffer(cmd, ib, 0, iType);
        const VkDeviceSize cmdOff = (VkDeviceSize)g * kGroupStride * sizeof(VkDrawIndexedIndirectCommand);
        const VkDeviceSize cntOff = (VkDeviceSize)g * sizeof(u32);
        vkCmdDrawIndexedIndirectCount(cmd, s_vsmIndirect->GetHandle(), cmdOff, s_vsmGroupCount->GetHandle(), cntOff,
                                      kGroupStride, sizeof(VkDrawIndexedIndirectCommand));
    }
    // Trees: STATIC cached casters — drawn into the SAME dirty static pages (their bin filtered
    // to slotDirty). Cached pages keep their depth → trees stop re-rasterizing every frame; they
    // refresh only as the moving sun dirties their pages (round-robin). [Phase 1]
    if (RImplementation.Trees && RImplementation.Trees->IsBuilt())
        RImplementation.Trees->VsmRender(cmd, s_pageList->GetHandle(), s_ubo[cur]->GetHandle());
    // FAR grass (static-cache hybrid, r_vsm_grass_static): L1/L2 pairs into the same
    // dirty pages, rigid — cached until sun motion / scroll re-dirties them. Near (L0)
    // grass stays in the dyn pass below with live wind.
    RenderGrassCastersStatic(cmd, cur);
    vkCmdEndRendering(cmd);
    atlasToRead(s_atlasImage);
    VK::Prof::ZoneEnd(cmd, zStatic);

    // ===== DYNAMIC atlas: skinned (NPC) + grass + near-tree wind crowns. Re-rendered +
    // cleared every UPDATE frame; on a cadence dyn-skip frame the whole block is bypassed
    // and s_dynImage keeps its last-update content (stays SHADER_READ) for the resolve. =====
    if (!s_dynSkip) {
        const u32 dw = kAtlasW * kPageSize, dh = kAtlasH * kPageSize;
        ImageBarrier(cmd, s_dynImage, s_dynFirst ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
        s_dynFirst = false;
        beginAtlas(s_dynView, VK_ATTACHMENT_LOAD_OP_CLEAR, dw, dh);
        const int zNPC = VK::Prof::ZoneBegin(cmd, "VSM/DynNPC");
        RenderSkinnedCasters(cmd, cur);   // viewport/scissor/bias already set by beginAtlas
        VK::Prof::ZoneEnd(cmd, zNPC);
        const int zGrass = VK::Prof::ZoneBegin(cmd, "VSM/DynGrass");
        RenderGrassCasters(cmd, cur);
        VK::Prof::ZoneEnd(cmd, zGrass);
        // NEAR trees with live wind (r_vsm_tree_wind hybrid) — swaying crown shadows. Usually
        // the dominant dynamic cost (the main perf knob — VsmUpdateNearSet / r_vsm_tree_wind_dist).
        const int zDynTree = VK::Prof::ZoneBegin(cmd, "VSM/DynTrees");
        if (RImplementation.Trees && RImplementation.Trees->IsBuilt())
            RImplementation.Trees->VsmRenderDyn(cmd, s_dynPageList->GetHandle(), s_ubo[cur]->GetHandle());
        VK::Prof::ZoneEnd(cmd, zDynTree);
        vkCmdEndRendering(cmd);
        atlasToRead(s_dynImage);
    }

    // Throttled counts to give the zone splits context (nearTrees drives VSM/DynTrees,
    // skinCasters drives VSM/DynNPC). r_vsm_debug 1.
    if (ps_r_vsm_debug) {
        static u32 s_lastSplitLog = 0;
        if (Device.dwTimeGlobal - s_lastSplitLog > 2000) {
            s_lastSplitLog = Device.dwTimeGlobal;
            const u32 nearTrees = (RImplementation.Trees && RImplementation.Trees->IsBuilt())
                                ? RImplementation.Trees->VsmNearCount() : 0u;
            Msg("[VK VSM] split counts: nearTrees=%u (VSM/DynTrees) skinCasters=%u (VSM/DynNPC) "
                "wind_dist=%.0f wind=%d treeVRS=%d cadence=%d (upd=%u skip=%u)%s", nearTrees, s_skinCount,
                ps_r_vsm_tree_wind_dist, ps_r_vsm_tree_wind, ps_r_vsm_tree_vrs, ps_r_vsm_cadence,
                s_updTally, s_skipTally, s_sunDown ? " NIGHT-FROZEN (sun down, VSM update skipped)" : "");
            // Cadence gate summary: forcedUpdates = frames where the alignment gate re-rendered
            // the dyn atlas because the window snapped / sun stepped (these WOULD have flickered).
            // heldMax = longest frozen streak between forced/scheduled updates. sunDriftMax = worst
            // sun step caught. After the fix, no visible flicker despite forcedUpdates > 0.
            Msg("[VK VSM] cadence gate: heldMax=%u sunDriftMax=%.3fdeg forcedUpdates=%u (gate re-renders misaligned dyn frames -> no flicker)",
                s_cadHeldMax, s_cadSunDriftMax, s_cadSnapTally);
            s_cadHeldMax = 0; s_cadSunDriftMax = 0.f; s_cadSnapTally = 0;
        }
    }
}

bool        MaskReady()      { return Enabled() && s_maskImage[0] != VK_NULL_HANDLE && s_maskValid; }
VkImageView GetMaskView()    { return s_maskView[s_curSlot]; }
VkSampler   GetMaskSampler() { return s_maskSampler; }

void ResolveMask(VkCommandBuffer cmd, VkImageView sceneDepth, VkExtent2D screen, const Fmatrix& viewProj)
{
    if (!Enabled() || sceneDepth == VK_NULL_HANDLE) return;
    if (s_maskImage[0] == VK_NULL_HANDLE || !AtlasReady()) return;   // need mask targets + a rendered atlas
    const u32 cur  = s_curSlot;
    const u32 prev = (cur + N - 1) % N;
    const bool histOK = ps_r_vsm_temporal && s_resolveCount >= 1;

    // ---- Resolve UBO (matrices + cameras + screen + blend params).
    ResolveParams rp{};
    rp.invViewProj.invert_44(viewProj);
    rp.prevViewProj = s_prevViewProj;
    rp.prevCamPos[0] = s_prevCamPos.x; rp.prevCamPos[1] = s_prevCamPos.y; rp.prevCamPos[2] = s_prevCamPos.z;
    rp.prevCamPos[3] = (float)ps_r_vsm_debug_dyn;   // dyn-debug: 1 = visible-darkening red, 2 = RAW dyn occlusion (mask B)
    rp.curCamPos[0]  = s_curCamPos.x;  rp.curCamPos[1]  = s_curCamPos.y;  rp.curCamPos[2]  = s_curCamPos.z;
    rp.curCamPos[3]  = ps_r_vsm_ta_blend_dyn;   // EMA alpha where the dyn atlas shadows (resolve takes min with params.x)
    rp.screen[0] = (float)screen.width; rp.screen[1] = (float)screen.height;
    rp.screen[2] = 1.0f / (float)screen.width; rp.screen[3] = 1.0f / (float)screen.height;
    // (3) DLSS-AWARE base weight. When r_dlss is on, DLSS ALSO temporally resolves the
    // shadow (it's baked into the scene colour it upscales) → the VSM EMA + DLSS = a
    // double temporal blur. Scale the VSM history weight down so DLSS carries the edge
    // AA and we don't compound the smear. No-op when DLSS is off.
    float histW = ps_r_vsm_ta_blend;
    if (VK::Dlss::Enabled()) histW *= ps_r_vsm_ta_blend_dlss;
    rp.params[0] = histOK ? histW : 0.f;               // EMA alpha (0 = current only)
    rp.params[1] = kRejectTol;
    rp.params[2] = histOK ? 1.f : 0.f;                 // historyValid
    rp.params[3] = ps_r_vsm_dyn_gate ? 1.f : 0.f;      // dyn-gate (skip caster-less dyn pages)
    // (1)+(2) neighbourhood clamp + motion-adaptive fade (vsm_resolve.comp params2).
    rp.params2[0] = ps_r_vsm_ta_clamp;                 // history clamp tol around current
    rp.params2[1] = ps_r_vsm_ta_motion;                // reproj motion (px) to reach the floor
    rp.params2[2] = ps_r_vsm_ta_motion_floor;          // history weight at/after that motion
    rp.params2[3] = ps_r_vsm_bias_min;                 // slope-scaled static bias: min slack (0 = legacy 0.6 m constant)
    // SOFT SHADOWS (r_vsm_soft) — stochastic PCSS in place of the 3x3 PCF. The
    // angle is authored in degrees because that is how the look is reasoned about
    // (0.265 = the sun's true half-angle; larger = the cinematic penumbra).
    const bool softOn = ps_r_vsm_soft >= 1;
    rp.params3[0] = softOn ? (float)ps_r_vsm_soft : 0.f;
    rp.params3[1] = (float)ps_r_vsm_soft_search;
    rp.params3[2] = tanf(deg2rad(ps_r_vsm_soft_angle));
    rp.params3[3] = ps_r_vsm_soft_range;
    // Frame phase for the per-pixel disc rotation. Wrapped at 64: the hash only
    // needs to walk a decorrelated cycle, and an unbounded counter would lose
    // float precision long before the session ends.
    rp.params4[0] = (float)(s_resolveCount & 63u);
    // The neighbourhood clamp was tuned (0.24) against a DETERMINISTIC 3x3 PCF,
    // where any big frame-to-frame deviation was a real shadow change. Stochastic
    // taps deviate by ~1/sqrt(taps) on their own, so that clamp would pin the
    // history to the noise and defeat the accumulation that denoises it.
    if (softOn) rp.params2[0] = ps_r_vsm_soft_clamp;
    if (s_resolveUboPtr[cur]) memcpy(s_resolveUboPtr[cur], &rp, sizeof(rp));

    // ---- First use of each slot since (re)create: UNDEFINED → GENERAL (so both the
    // output slot and the history slot's descriptor layouts are valid).
    for (u32 i = 0; i < N; ++i)
        if (s_maskFirst[i]) { ImageBarrier(cmd, s_maskImage[i], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT); s_maskFirst[i] = false; }

    // Order prev-frame history write (compute) + prev color-pass read (fragment, WAR on
    // this slot) before this resolve's read/write. Single queue → covers prior submissions.
    MemBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    // ---- Descriptor set.
    VkDescriptorImageInfo dDepth{ s_depthSampler, sceneDepth,        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkDescriptorImageInfo dAtlas{ s_atlasSampler, s_atlasView,       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkDescriptorImageInfo dAtlasD{ s_atlasSampler, s_dynView,        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkDescriptorImageInfo dHist { s_maskSampler,  s_maskView[prev],  VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo dOut  { VK_NULL_HANDLE, s_maskView[cur],   VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorBufferInfo bPT{ s_pageTable->GetHandle(),       0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo bPTd{ s_dynPageTable->GetHandle(),   0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo bPU{ s_dynPageUsed->GetHandle(),     0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo bCM{ s_ubo[cur]->GetHandle(),      0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo bRU{ s_resolveUbo[cur]->GetHandle(), 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet w[10]{};
    for (u32 i = 0; i < 10; ++i) { w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = s_resolveSet[cur]; w[i].dstBinding = i; w[i].descriptorCount = 1; }
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[0].pImageInfo  = &dDepth;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[1].pImageInfo  = &dAtlas;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;         w[2].pBufferInfo = &bPT;
    w[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;         w[3].pBufferInfo = &bCM;
    w[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[4].pImageInfo  = &dHist;
    w[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;          w[5].pImageInfo  = &dOut;
    w[6].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;         w[6].pBufferInfo = &bRU;
    w[7].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[7].pImageInfo  = &dAtlasD;
    w[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;         w[8].pBufferInfo = &bPTd;
    w[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;         w[9].pBufferInfo = &bPU;
    vkUpdateDescriptorSets(VulkanHW.m_Device, 10, w, 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_resolvePipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_resolveLayout, 0, 1, &s_resolveSet[cur], 0, nullptr);
    vkCmdDispatch(cmd, (screen.width + 7) / 8, (screen.height + 7) / 8, 1);

    // Mask write (compute) → receiver sampled read (fragment, color pass).
    MemBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    s_prevViewProj = viewProj;
    s_prevCamPos   = s_curCamPos;
    if (s_resolveCount < 0xFFFF) s_resolveCount++;
    s_maskValid = true;
}

void InvalidateCache()
{
    // Level unload: the toroidal page cache is WORLD-anchored, and a different
    // level reuses the same coordinates — a resident page would keep serving the
    // PREVIOUS level's depth (the "light/dark squares on walls after a level
    // change" bug). Dropping s_physInit makes the next frame's MarkPages
    // refill the physical-tile table to EMPTY (vkCmdFillBuffer):
    // every page re-renders against the new level's casters. Cheap: one full
    // atlas re-render on the first frame, exactly like a fresh start.
    s_physInit = false;
}

void Destroy()
{
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
    auto del = [](CVulkanBuffer*& b) { if (b) { xr_delete(b); b = nullptr; } };
    for (u32 i = 0; i < N; ++i) { del(s_ubo[i]); s_uboPtr[i] = nullptr; s_set[i] = VK_NULL_HANDLE; }
    del(s_needed); del(s_counter); del(s_rmask); del(s_pageHits); del(s_readback); s_readPtr = nullptr;
    del(s_pageTable); del(s_pageList);
    del(s_dynPageTable); del(s_dynPageList); del(s_dynAllocInfo); del(s_dynPageUsed);
    del(s_physTile); del(s_slotDirty); del(s_dirtyList); del(s_drawClear); del(s_residRB); s_residPtr = nullptr; s_physInit = false;
    del(s_pageMax); del(s_pageMaxBlk); del(s_priorValid); del(s_slotDirtyPrev); s_hzbWasOn = false;   // shadow-HZB
    del(s_casterPages); del(s_vsmIndirect); del(s_vsmGroupCount); del(s_binStats); del(s_binReadback); s_binReadPtr = nullptr;
    del(s_clCount);
    DestroyClusterCandidates(); s_clCandStamp = 0; s_clCandErrB0 = -1.f; s_clCandAT = false;
    if (s_binClPipe)    { vkDestroyPipeline(VulkanHW.m_Device, s_binClPipe, nullptr); s_binClPipe = VK_NULL_HANDLE; }
    if (s_binClLayout)  { vkDestroyPipelineLayout(VulkanHW.m_Device, s_binClLayout, nullptr); s_binClLayout = VK_NULL_HANDLE; }
    if (s_binClPool)    { vkDestroyDescriptorPool(VulkanHW.m_Device, s_binClPool, nullptr); s_binClPool = VK_NULL_HANDLE; }
    if (s_binClSetL)    { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_binClSetL, nullptr); s_binClSetL = VK_NULL_HANDLE; }
    for (u32 i = 0; i < N; ++i) s_binClSet[i] = VK_NULL_HANDLE;
    s_clusterFrame = false; s_lastClusterMode = false; s_lastClusterK = -1.f; s_lastAtMode = false;
    if (s_pipe)         { vkDestroyPipeline(VulkanHW.m_Device, s_pipe, nullptr); s_pipe = VK_NULL_HANDLE; }
    if (s_layout)       { vkDestroyPipelineLayout(VulkanHW.m_Device, s_layout, nullptr); s_layout = VK_NULL_HANDLE; }
    if (s_pool)         { vkDestroyDescriptorPool(VulkanHW.m_Device, s_pool, nullptr); s_pool = VK_NULL_HANDLE; }
    if (s_setL)         { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_setL, nullptr); s_setL = VK_NULL_HANDLE; }
    if (s_allocPipe)    { vkDestroyPipeline(VulkanHW.m_Device, s_allocPipe, nullptr); s_allocPipe = VK_NULL_HANDLE; }
    if (s_allocLayout)  { vkDestroyPipelineLayout(VulkanHW.m_Device, s_allocLayout, nullptr); s_allocLayout = VK_NULL_HANDLE; }
    if (s_allocPool)    { vkDestroyDescriptorPool(VulkanHW.m_Device, s_allocPool, nullptr); s_allocPool = VK_NULL_HANDLE; }
    if (s_allocSetL)    { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_allocSetL, nullptr); s_allocSetL = VK_NULL_HANDLE; }
    if (s_residPipe)    { vkDestroyPipeline(VulkanHW.m_Device, s_residPipe, nullptr); s_residPipe = VK_NULL_HANDLE; }
    if (s_residLayout)  { vkDestroyPipelineLayout(VulkanHW.m_Device, s_residLayout, nullptr); s_residLayout = VK_NULL_HANDLE; }
    if (s_residPool)    { vkDestroyDescriptorPool(VulkanHW.m_Device, s_residPool, nullptr); s_residPool = VK_NULL_HANDLE; }
    if (s_residSetL)    { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_residSetL, nullptr); s_residSetL = VK_NULL_HANDLE; }
    s_residSet = VK_NULL_HANDLE;
    if (s_hzbPipe)      { vkDestroyPipeline(VulkanHW.m_Device, s_hzbPipe, nullptr); s_hzbPipe = VK_NULL_HANDLE; }
    if (s_hzbLayout)    { vkDestroyPipelineLayout(VulkanHW.m_Device, s_hzbLayout, nullptr); s_hzbLayout = VK_NULL_HANDLE; }
    if (s_hzbPool)      { vkDestroyDescriptorPool(VulkanHW.m_Device, s_hzbPool, nullptr); s_hzbPool = VK_NULL_HANDLE; }
    if (s_hzbSetL)      { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_hzbSetL, nullptr); s_hzbSetL = VK_NULL_HANDLE; }
    s_hzbSet = VK_NULL_HANDLE;
    if (s_clearPipe)    { vkDestroyPipeline(VulkanHW.m_Device, s_clearPipe, nullptr); s_clearPipe = VK_NULL_HANDLE; }
    if (s_clearLayout)  { vkDestroyPipelineLayout(VulkanHW.m_Device, s_clearLayout, nullptr); s_clearLayout = VK_NULL_HANDLE; }
    if (s_clearPool)    { vkDestroyDescriptorPool(VulkanHW.m_Device, s_clearPool, nullptr); s_clearPool = VK_NULL_HANDLE; }
    if (s_clearSetL)    { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_clearSetL, nullptr); s_clearSetL = VK_NULL_HANDLE; }
    s_clearSet = VK_NULL_HANDLE; s_clearVS = VK_NULL_HANDLE;
    if (s_binPipe)      { vkDestroyPipeline(VulkanHW.m_Device, s_binPipe, nullptr); s_binPipe = VK_NULL_HANDLE; }
    if (s_binLayout)    { vkDestroyPipelineLayout(VulkanHW.m_Device, s_binLayout, nullptr); s_binLayout = VK_NULL_HANDLE; }
    if (s_binPool)      { vkDestroyDescriptorPool(VulkanHW.m_Device, s_binPool, nullptr); s_binPool = VK_NULL_HANDLE; }
    if (s_binSetL)      { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_binSetL, nullptr); s_binSetL = VK_NULL_HANDLE; }
    for (auto& kv : s_pagePipes) if (kv.second) vkDestroyPipeline(VulkanHW.m_Device, kv.second, nullptr);
    s_pagePipes.clear();
    for (auto& kv : s_pageATPipes) if (kv.second) vkDestroyPipeline(VulkanHW.m_Device, kv.second, nullptr);
    s_pageATPipes.clear();
    if (s_renderATLayout) { vkDestroyPipelineLayout(VulkanHW.m_Device, s_renderATLayout, nullptr); s_renderATLayout = VK_NULL_HANDLE; }
    s_pageATVS = VK_NULL_HANDLE; s_pageATFS = VK_NULL_HANDLE; s_pageATFailed = false;
    if (s_renderLayout) { vkDestroyPipelineLayout(VulkanHW.m_Device, s_renderLayout, nullptr); s_renderLayout = VK_NULL_HANDLE; }
    if (s_renderPool)   { vkDestroyDescriptorPool(VulkanHW.m_Device, s_renderPool, nullptr); s_renderPool = VK_NULL_HANDLE; }
    if (s_renderSetL)   { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_renderSetL, nullptr); s_renderSetL = VK_NULL_HANDLE; }
    if (s_atlasView)    { vkDestroyImageView(VulkanHW.m_Device, s_atlasView, nullptr); s_atlasView = VK_NULL_HANDLE; }
    if (s_dynView)      { vkDestroyImageView(VulkanHW.m_Device, s_dynView, nullptr); s_dynView = VK_NULL_HANDLE; }
    if (s_atlasSampler) { vkDestroySampler(VulkanHW.m_Device, s_atlasSampler, nullptr); s_atlasSampler = VK_NULL_HANDLE; }
    if (s_atlasImage)   { VK::Vram::DestroyImage(VulkanHW.m_Allocator, s_atlasImage, s_atlasAlloc); s_atlasImage = VK_NULL_HANDLE; s_atlasAlloc = VK_NULL_HANDLE; }
    if (s_dynImage)     { VK::Vram::DestroyImage(VulkanHW.m_Allocator, s_dynImage, s_dynAlloc); s_dynImage = VK_NULL_HANDLE; s_dynAlloc = VK_NULL_HANDLE; }
    s_pageVS = VK_NULL_HANDLE; s_atlasFirst = true; s_dynFirst = true;
    for (u32 i = 0; i < N; ++i) s_renderSet[i] = VK_NULL_HANDLE;
    if (s_depthSampler) { vkDestroySampler(VulkanHW.m_Device, s_depthSampler, nullptr); s_depthSampler = VK_NULL_HANDLE; }
    s_dynAllocSet = VK_NULL_HANDLE;
    for (u32 i = 0; i < N; ++i) s_binSet[i] = VK_NULL_HANDLE;

    // Temporal resolve teardown.
    for (u32 i = 0; i < N; ++i) { del(s_resolveUbo[i]); s_resolveUboPtr[i] = nullptr; s_resolveSet[i] = VK_NULL_HANDLE; }
    DestroyMaskTargets();
    if (s_maskSampler)   { vkDestroySampler(VulkanHW.m_Device, s_maskSampler, nullptr); s_maskSampler = VK_NULL_HANDLE; }
    if (s_resolvePipe)   { vkDestroyPipeline(VulkanHW.m_Device, s_resolvePipe, nullptr); s_resolvePipe = VK_NULL_HANDLE; }
    if (s_resolveLayout) { vkDestroyPipelineLayout(VulkanHW.m_Device, s_resolveLayout, nullptr); s_resolveLayout = VK_NULL_HANDLE; }
    if (s_resolvePool)   { vkDestroyDescriptorPool(VulkanHW.m_Device, s_resolvePool, nullptr); s_resolvePool = VK_NULL_HANDLE; }
    if (s_resolveSetL)   { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_resolveSetL, nullptr); s_resolveSetL = VK_NULL_HANDLE; }

    // Skinned-caster teardown.
    for (u32 i = 0; i < N; ++i) { del(s_skinMeta[i]); s_skinMetaPtr[i] = nullptr; s_skinBinSet[i] = VK_NULL_HANDLE; s_skinPageSet[i] = VK_NULL_HANDLE; }
    del(s_skinCasterPages); del(s_skinIndirect); del(s_skinStats); del(s_skinStatsRB); s_skinStatsPtr = nullptr;
    s_skinCasters.clear(); s_skinCount = 0;
    for (auto& kv : s_skinPagePipes) if (kv.second) vkDestroyPipeline(VulkanHW.m_Device, kv.second, nullptr);
    s_skinPagePipes.clear();
    if (s_skinBinPipe)    { vkDestroyPipeline(VulkanHW.m_Device, s_skinBinPipe, nullptr); s_skinBinPipe = VK_NULL_HANDLE; }
    if (s_skinBinLayout)  { vkDestroyPipelineLayout(VulkanHW.m_Device, s_skinBinLayout, nullptr); s_skinBinLayout = VK_NULL_HANDLE; }
    if (s_skinBinPool)    { vkDestroyDescriptorPool(VulkanHW.m_Device, s_skinBinPool, nullptr); s_skinBinPool = VK_NULL_HANDLE; }
    if (s_skinBinSetL)    { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_skinBinSetL, nullptr); s_skinBinSetL = VK_NULL_HANDLE; }
    if (s_skinPageLayout) { vkDestroyPipelineLayout(VulkanHW.m_Device, s_skinPageLayout, nullptr); s_skinPageLayout = VK_NULL_HANDLE; }
    if (s_skinPagePool)   { vkDestroyDescriptorPool(VulkanHW.m_Device, s_skinPagePool, nullptr); s_skinPagePool = VK_NULL_HANDLE; }
    if (s_skinPageSetL)   { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_skinPageSetL, nullptr); s_skinPageSetL = VK_NULL_HANDLE; }
    s_skinPageVS = VK_NULL_HANDLE;   // module owned by g_ShaderManager

    // Grass-caster teardown.
    del(s_grassPairs); del(s_grassPairCnt); del(s_grassPairInd);
    del(s_grassPairsS); del(s_grassPairCntS); del(s_grassPairIndS);
    del(s_grassStats); del(s_grassStatsRB); s_grassStatsPtr = nullptr;
    del(s_grassPairCntRB); s_grassPairCntPtr = nullptr;
    for (u32 i = 0; i < N; ++i) { s_grassBinSet[i] = VK_NULL_HANDLE; s_grassPageSet[i] = VK_NULL_HANDLE; s_grassBinSetS[i] = VK_NULL_HANDLE; s_grassPageSetS[i] = VK_NULL_HANDLE; }
    s_grassSection = 0; s_grassTypes = 0; s_grassPairSection = 0; s_grassPairSectionS = 0;
    s_grassPageVSS = VK_NULL_HANDLE;   // module owned by g_ShaderManager
    if (s_grassPagePipeS)  { vkDestroyPipeline(VulkanHW.m_Device, s_grassPagePipeS, nullptr); s_grassPagePipeS = VK_NULL_HANDLE; }
    if (s_grassPagePipe)   { vkDestroyPipeline(VulkanHW.m_Device, s_grassPagePipe, nullptr); s_grassPagePipe = VK_NULL_HANDLE; }
    if (s_grassBinPipe)    { vkDestroyPipeline(VulkanHW.m_Device, s_grassBinPipe, nullptr); s_grassBinPipe = VK_NULL_HANDLE; }
    if (s_grassBinLayout)  { vkDestroyPipelineLayout(VulkanHW.m_Device, s_grassBinLayout, nullptr); s_grassBinLayout = VK_NULL_HANDLE; }
    if (s_grassBinPool)    { vkDestroyDescriptorPool(VulkanHW.m_Device, s_grassBinPool, nullptr); s_grassBinPool = VK_NULL_HANDLE; }
    if (s_grassBinSetL)    { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_grassBinSetL, nullptr); s_grassBinSetL = VK_NULL_HANDLE; }
    if (s_grassPageLayout) { vkDestroyPipelineLayout(VulkanHW.m_Device, s_grassPageLayout, nullptr); s_grassPageLayout = VK_NULL_HANDLE; }
    if (s_grassPagePool)   { vkDestroyDescriptorPool(VulkanHW.m_Device, s_grassPagePool, nullptr); s_grassPagePool = VK_NULL_HANDLE; }
    if (s_grassPageSetL)   { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_grassPageSetL, nullptr); s_grassPageSetL = VK_NULL_HANDLE; }
    s_grassPageVS = VK_NULL_HANDLE; s_grassPageFS = VK_NULL_HANDLE;   // modules owned by g_ShaderManager

    s_inited = false; s_dead = false; s_frame = 0;
}

}}  // namespace VK::VSM
