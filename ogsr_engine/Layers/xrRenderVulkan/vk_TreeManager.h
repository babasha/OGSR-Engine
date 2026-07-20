// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan - GPU-driven tree manager.
//
// Trees (MT_TREE_ST / MT_TREE_PM) extracted from Visuals[] at level load,
// uploaded as metadata + transforms SSBOs, then drawn each frame via a
// compute cull → indirect draw pipeline (Session B). Architecture mirrors
// the GPU-driven grass path (CDetailManager); replaces R4's DSGraph
// FloraVertData streaming.
//
// SESSION A SCOPE (this file): extraction + SSBO upload + per-texture
// descriptor sets + indirect/draw-count buffer allocation. NO compute or
// graphics pipeline yet — that's Session B.

#pragma once
#include "vk_core.h"
#include "vk_buffer.h"

// Visual classes live in the global namespace (vk_Visual.h).
class vkRender_Visual;
class vkFTreeVisual;
class vkFHierrarhyVisual;
class CFrustum;   // camera-frustum cull for the depth-prepass path

namespace VK
{

class CVulkanTexture;
struct FrameContext;

// Tree graphics push constants (160 B, VS+FS). Matches tree.vert/frag.
struct TreeGfxPush
{
    Fmatrix  mViewProj;    // 64 B  world → clip
    float    uvScale;      //  4 B  1/2048 (FTreeVisual_quant = 32768/16)
    float    alphaRef;     //  4 B  fragment alpha cutoff
    float    _pad0;        //  4 B  (vec4 below needs 16-byte alignment → offset 80)
    float    _pad1;        //  4 B
    Fvector4 vSunColor;    // 16 B  env sun colour (rgb); w unused
    Fvector4 vHemiColor;   // 16 B  env hemi colour (rgb); w unused
    Fvector4 wind_params;  // 16 B  SSFX (wind_direction, wind_velocity, _, _)
    Fvector4 wsetup_trees; // 16 B  SSFX (branchSpeed, trunkSpeed, bend, minWindSpeed)
    Fvector4 wind_anim;    // 16 B  Environment.wind_anim drift (xyz); w unused
};
static_assert(sizeof(TreeGfxPush) == 160, "TreeGfxPush must be 160 B");

// Tree wind-sway motion-vector push (VS+FS) — pairs with tree_motion.{vert,frag}.
// Projects the cur AND prev wind pose so swaying trees carry their true screen
// motion; the fullscreen MV pass only reconstructs camera motion from depth and
// misses the sway. wsetup/wind_params are frame-stable-ish; wind_anim (drift) is the
// dominant per-frame delta. 240 B (device max push 256).
struct TreeMotionPush
{
    Fmatrix  curVP;             // 0    this frame's view-proj, UNJITTERED (jitter-free MV)
    Fmatrix  prevVP;            // 64   previous frame's view-proj, UNJITTERED
    float    uvScale;           // 128  1/2048
    float    alphaRef;          // 132  leaf cutout (fragment)
    float    jitterX;           // 136  this frame's sub-pixel jitter (D3D-NDC), re-applied to gl_Position
    float    jitterY;           // 140
    Fvector4 wind_params;       // 144  cur
    Fvector4 wsetup_trees;      // 160  cur
    Fvector4 wind_anim;         // 176  cur
    Fvector4 wind_params_prev;  // 192
    Fvector4 wsetup_trees_prev; // 208
    Fvector4 wind_anim_prev;    // 224
};
static_assert(sizeof(TreeMotionPush) == 240, "TreeMotionPush must be 240 B");

// View frustum UBO (binding 1 of the cull set). 6 normalized planes, std140.
// LEGACY: kept only so the cull descriptor still has a valid buffer bound at
// binding 1 — the planes now travel via TreeCullPush (see below) to avoid the
// single-buffered cross-frame race that caused black tree silhouettes at the
// screen edge during rotation.
struct TreeFrustumUBO
{
    Fvector4 planes[6];  // 96 B
};

// Cull compute push constants (116 B). The frustum planes ride in push constants
// (recorded per dispatch) instead of the racy single-buffered UBO. planes FIRST
// (offset 0) keeps the trailing u32s 4-aligned and the total under the 128 B
// guaranteed push limit. Mirrors tree_cull.comp's push_constant block.
struct TreeCullPush
{
    Fvector4 planes[6];   // 96 B  view frustum (normalized; 0=L 1=R 2=B 3=T 4=N 5=F)
    u32      mesh_count;   //  4 B  trees in this group
    u32      _unused;      //  4 B
    u32      mesh_offset;  //  4 B  start index in the metadata buffer
    u32      output_base;  //  4 B  base index into the indirect command buffer
    u32      count_index;  //  4 B  index into the draw-count buffer (= group)
};
static_assert(sizeof(TreeCullPush) == 116, "TreeCullPush must be 116 B");

// ============================================================================
// Per-instance data uploaded to GPU. The vertex shader reads
// transforms[gl_InstanceIndex] (firstInstance encodes the global tree index
// in the indirect draw command). Layout matches a column-major mat4 plus
// padding so std430 alignment is automatic.
// ============================================================================
struct GpuTreeInstance
{
    Fmatrix xform;          // 64 B  Per-instance world transform
    float   c_scale_hemi;   //  4 B
    float   c_bias_hemi;    //  4 B
    u32     _pad0;          //  4 B
    u32     _pad1;          //  4 B
};
static_assert(sizeof(GpuTreeInstance) == 80, "GpuTreeInstance must be 80 B");

// ============================================================================
// Per-mesh metadata SSBO entry. Cull compute reads sphere_P/R for frustum
// test; on hit it writes a VkDrawIndexedIndirectCommand built from
// index_count + ib_first + first_vertex + global tree index.
// ============================================================================
struct GpuTreeMeta
{
    Fvector  sphere_P;      // 12 B  World-space center
    float    sphere_R;      //  4 B  Radius
    u32      index_count;   //  4 B  num_tris * 3
    u32      ib_first;      //  4 B  First index (element offset into IB)
    u32      first_vertex;  //  4 B  Base vertex (element offset into VB)
    u32      _pad;          //  4 B  std430 alignment
};
static_assert(sizeof(GpuTreeMeta) == 32, "GpuTreeMeta must be 32 B");

// ============================================================================
// Meshlet cluster (Phase A — per-page VSM meshlet culling, r_vsm_meshlet).
// One cluster ≈ kMeshletTris triangles of a UNIQUE tree mesh (shared across all
// instances of that species). center/radius are LOCAL (pre-xform) mesh space —
// the bin transforms them per instance. first_index/index_count point into the
// dedicated meshlet index buffer (m_MeshletIndexBuffer), index values still
// relative to the mesh's vBase so vertexOffset = first_vertex works unchanged.
// ============================================================================
struct GpuMeshlet
{
    Fvector center;       // 12 B  local mesh-space sphere center
    float   radius;       //  4 B  local sphere radius
    u32     first_index;  //  4 B  offset into m_MeshletIndexBuffer
    u32     index_count;  //  4 B  tris * 3
    u32     _pad0;        //  4 B
    u32     _pad1;        //  4 B  std430 (vec3+float+4u = 32 B)
};
static_assert(sizeof(GpuMeshlet) == 32, "GpuMeshlet must be 32 B");

// Per-tree meshlet slice into GpuMeshlet[] (instances of one mesh share a slice).
struct GpuTreeMeshletRange
{
    u32 base;    // first meshlet index in m_MeshletBuffer
    u32 count;   // meshlet count
};
static_assert(sizeof(GpuTreeMeshletRange) == 8, "GpuTreeMeshletRange must be 8 B");

// ============================================================================
// Per-tree crown-HULL range (r_vsm_tree_hull) — the AC-Shadows-style shadow-caster
// LOD. Built per UNIQUE mesh in BuildMeshlets (k-means vertex clusters → ellipsoid
// icosphere lobes, local mesh space), instances share the range. ib indices are
// relative to vb_first (vertexOffset of the indirect draw). count 0 = no hull.
// Matches uvec4 in vsm_hull_cmd.comp.
// ============================================================================
struct GpuTreeHullInfo
{
    u32 ib_first;   // first index in m_HullIB
    u32 ib_count;   // index count (0 = hull unavailable → tree keeps its crown mesh)
    u32 vb_first;   // base vertex in m_HullVB (→ indirect vertexOffset)
    u32 _pad;
};
static_assert(sizeof(GpuTreeHullInfo) == 16, "GpuTreeHullInfo must be 16 B");

// ============================================================================
// One crown VOXEL (r_vsm_tree_hull_vox debug viewmode, UE Nanite-foliage style):
// an individual colored cube, NOT a merged shell face. Species-local center +
// baked RGBA8 (rgb = per-voxel foliage tint with AO/height grading; a = size
// jitter seed in the HIGH nibble + leaf coverage 1-15 in the LOW nibble — the
// caster dissolves cubes in coverage order across the crossfade band). Fed to
// the voxel-debug pipeline as a PER-INSTANCE vertex stream (stride 16); the
// cube's 36 corners are generated from gl_VertexIndex.
// ============================================================================
struct GpuTreeVoxel
{
    Fvector p;      // voxel center, species-local mesh space (same space as m_HullVB)
    u32     rgba;   // packed R|G<<8|B<<16|A<<24
};
static_assert(sizeof(GpuTreeVoxel) == 16, "GpuTreeVoxel must be 16 B");

// ============================================================================
// Voxel BRICK — the shadow-caster unit, straight from UE Nanite's voxel bricks
// (RasterizeBricks.usf): 4×4×4 cells of a crown voxel grid packed into one
// 64-bit occupancy mask. The caster draws ONE light-facing quad per brick
// (6 verts) and the FS DDA-raycasts the mask along the sun direction, writing
// conservative depth — per-voxel cost collapses from 36 HW verts to a few
// mask tests per covered pixel. Two masks: `full` = every occupied cell,
// `core` = only cells whose leaf coverage nibble ≥ 6 — the crossfade band
// swaps full→core per brick (staggered by hash) so approaching a tree first
// drops the low-coverage "extras", leaving cubes on the leaf clumps.
// meta: bits 0-3 = average coverage (rank for the band dissolve),
//       bits 4-11 = bake hash (stagger seed). Bit index inside a mask:
//       bit = x | y<<2 | z<<4 (x fastest), matching the FS DDA.
// ============================================================================
struct GpuTreeBrick
{
    Fvector p;        // brick MIN corner, species-local mesh space
    u32     meta;     // avgCov4 | hash8<<4
    u32     fullLo, fullHi;
    u32     coreLo, coreHi;
};
static_assert(sizeof(GpuTreeBrick) == 32, "GpuTreeBrick must be 32 B (std430 mirror)");

// Per-tree, per-LOD slice of the voxel-cloud buffer (CPU-side; debug draws are
// CPU-recorded). `size` is the voxel edge length in mesh-local meters — the
// renderer picks the finest level whose projected size still covers the target
// pixel count (constant screen-size voxels, the Nanite LOD-cut metric).
struct TreeVoxLod
{
    u32   first;    // first voxel (instance) in m_VoxVB
    u32   count;    // voxel count (0 = level unavailable)
    float size;     // voxel edge, mesh-local meters
    u32   _pad;
};

// ============================================================================
// One contiguous (vb, ib, tcOffset, descSetIdx) span in m_TreeMetadataBuffer.
// CullCompute writes its indirect-cmds into m_TreeIndirectBuffer at offset
// `groupIndex * m_MaxGroupMeshCount * sizeof(VkDrawIndexedIndirectCommand)`,
// and DrawTrees reads `vkCmdDrawIndexedIndirectCount` from there. Per-group
// pipeline variant chosen by tcOffset (24 vs 28).
// ============================================================================
struct TreeIndirectGroup
{
    VkBuffer vb;
    VkBuffer ib;
    u32      stride;
    u32      tcOffset;       // 24 or 28 → pipeline variant
    u32      descSetIdx;     // index into m_TexDescSets
    u32      meshOffset;     // start in m_TreeMetadataBuffer
    u32      meshCount;
};

// ============================================================================
// CTreeManager — Session A only (extraction + upload).
// Sessions B/C add: cull compute pipeline + graphics pipeline + render path.
// ============================================================================
class CTreeManager
{
public:
    CTreeManager();
    ~CTreeManager();

    // Walks RImplementation.Visuals[] for MT_TREE_ST/PM, builds groups,
    // uploads metadata + transforms SSBOs, allocates indirect/draw-count
    // buffers, creates per-texture descriptor sets. No-op on subsequent
    // calls or empty levels.
    void Build();
    void Destroy();

    // Session B per-frame entry point — registered as a scene pass. Runs the
    // frustum-cull compute then issues per-group indirect draws. No-op until
    // pipelines are built and at least one tree group exists.
    void Render(VK::FrameContext& ctx);

    // Shadow caster path: depth-only, alpha-tested draws of the trees whose
    // sphere intersects the sun ortho box, CPU-recorded per tree (no compute
    // cull). cascade < 0: the far static map (ShadowMap::SphereVisible, runs
    // only on sun static redraws). cascade >= 0: the per-frame near cascade
    // with that index (ShadowMap::CascadeSphereVisible). frustum != null
    // overrides both culls: camera depth-PREPASS path — trees join the scene
    // depth so GTAO sees them and the foliage color passes get early-Z
    // (same VP + alpha-ref as the LEQUAL color pass → re-raster matches).
    // Caller owns render begin/end, viewport and bias.
    // minDist/maxDist: cull trees by camera distance to split the shadow into a
    // cached STATIC layer (far: minDist=wind_shadow_dist) and a per-frame DYNAMIC
    // layer (near: maxDist=wind_shadow_dist) so near trees sway in the wind without
    // re-rasterizing the whole forest. Defaults = all trees (no split).
    void RenderDepth(VkCommandBuffer cmd, const Fmatrix& lightVP, s32 cascade = -1,
                     const CFrustum* frustum = nullptr,
                     float minDist = 0.0f, float maxDist = 1e9f);

    // Wind-sway motion-vector overlay. Called by VK::MotionVec::ExecuteDynamic
    // INSIDE its already-begun MV render pass (MV target + scene depth bound,
    // negative-height full-depth viewport). Re-draws THIS frame's visible trees
    // (same indirect buffers the forward pass filled), reprojecting the cur+prev
    // wind pose. curVP/prevVP are UNJITTERED (jitter-free MV); jitterNdcX/Y (D3D-NDC,
    // 0 when DLSS off) is re-applied to gl_Position so tree depth still bit-matches the
    // jittered forward draw. No-op until trees are loaded.
    void RenderMotion(const VK::FrameContext& ctx, const Fmatrix& curVP, const Fmatrix& prevVP,
                      float jitterNdcX, float jitterNdcY);

    // VSM caster path. vk_vsm provides the page buffers (its own pageTable/pageList +
    // this frame's clipmap UBO). VsmBin/VsmBinDyn run the per-tree page binning (compute,
    // before the atlas pass); VsmRender/VsmRenderDyn rasterize the trees into the bound
    // atlas pages (call INSIDE vk_vsm's RenderAtlas static/dynamic pass respectively).
    // Near/far wind hybrid: VsmUpdateNearSet refreshes the CPU near set (call BEFORE the
    // residency dispatch, with this frame's world->light sun view — the near metric is the
    // LIGHT-SPACE lateral distance to the tree's shadow column, not trunk distance);
    // VsmPopTransitions hands out boundary-crossing tree spheres for the residency
    // invalidation circles (≤maxOut per call, rest stay queued).
    void VsmUpdateNearSet(const Fmatrix& sunView);
    u32  VsmPopTransitions(Fvector4* out, u32 maxOut);
    // pageList (slot -> level,page) is finalized by the residency/alloc passes BEFORE the
    // bin runs; the meshlet-cull stage 2 (r_vsm_meshlet) needs it, so it's threaded through
    // VsmBin/VsmBinDyn even though the per-tree stage 1 ignores it.
    // pageMax = shadow-HZB per-static-slot occluder max (r_vsm_hzb); occlusion is applied only in the
    // STATIC pass (far trees vs cached walls/terrain). VsmBinDyn binds it for layout parity but never culls.
    void VsmBin(VkCommandBuffer cmd, VkBuffer pageTable, VkBuffer slotDirty, VkBuffer dynUsed, VkBuffer clipmapUBO, VkBuffer pageList, VkBuffer pageMax);
    // staticPageTable (Option A): lets the DYNAMIC near-tree bin look up the STATIC occluder for each
    // world page (virtual page -> static slot -> pageMax) → cull near-tree pages hidden behind walls.
    void VsmBinDyn(VkCommandBuffer cmd, VkBuffer dynPageTable, VkBuffer dynUsed, VkBuffer clipmapUBO, VkBuffer dynPageList, VkBuffer pageMax, VkBuffer staticPageTable);
    // De-serialized bin phasing (see vk_vsm MarkPages): VsmBinClears batches this pass's
    // counter/stat fills into the frame-start fill block (NO barriers of its own);
    // VsmBinMeshlets runs the stage-2 meshlet refinement for both atlases AFTER the
    // caller's single stage-1→stage-2 barrier (stage 2 reads stage-1 casterPages/indirect).
    void VsmBinClears(VkCommandBuffer cmd);
    void VsmBinMeshlets(VkCommandBuffer cmd, bool dyn, VkBuffer pageListStatic, VkBuffer pageListDyn, VkBuffer clipmapUBO,
                        VkBuffer pageMaxBlk = VK_NULL_HANDLE, VkBuffer staticPT = VK_NULL_HANDLE);
    void VsmRender(VkCommandBuffer cmd, VkBuffer pageList, VkBuffer clipmapUBO);
    void VsmRenderDyn(VkCommandBuffer cmd, VkBuffer dynPageList, VkBuffer clipmapUBO);

    bool IsBuilt() const { return m_bBuilt; }
    u32  VsmNearCount() const { return m_VsmNearCount; }   // this frame's near-set size (diag)
    bool IsReady() const;
    u32  GetTotalCount() const { return m_TotalCount; }
    u32  GetGroupCount() const { return (u32)m_Groups.size(); }

private:
    // Recursive helper: descends MT_HIERRARHY / MT_LOD into children
    // until it reaches MT_TREE_ST/PM leaves.
    void ExtractFromVisual(::vkRender_Visual* vis,
                           xr_vector<::vkFTreeVisual*>& outTrees,
                           u32 source = 0);   // 0=top-level 1=hierarchy 2=LOD

    void UploadMetadata(const xr_vector<GpuTreeMeta>& meta);
    void UploadTransforms(const xr_vector<GpuTreeInstance>& xforms);

    // Phase A: build per-mesh meshlet clusters for the VSM meshlet-cull path.
    // GPU-reads back each unique tree mesh's positions + indices (pools carry
    // TRANSFER_SRC), Morton-slices them into GpuMeshlet clusters, and uploads the
    // dedicated meshlet IB + GpuMeshlet[] + per-tree range SSBOs. Best-effort:
    // on any failure the meshlet path just stays unavailable (old path unaffected).
    // meta is index-aligned with trees (holds the real drawn ib_first/index_count/
    // first_vertex, incl. the MT_TREE_PM sw[0] window).
    void BuildMeshlets(const xr_vector<::vkFTreeVisual*>& trees,
                       const xr_vector<GpuTreeMeta>& meta);
    void CreateIndirectBuffers();
    void CreateTextureDescriptors(const xr_vector<VkImageView>& uniqueViews);

    // ----- Session B: pipelines + per-frame -----
    void CreateFrustumUBO();
    void CreateCullPipeline();        // compute cull (tree_cull.comp)
    void CreateXformDescriptor();     // set 0 of the graphics pipeline (transforms SSBO)
    void CreateGfxPipelines();        // tree.vert/frag — tcOffset 24 + 28 variants
    void CreateMotionPipelines();     // tree_motion.vert/frag — lazy MV overlay (24/28)
    void DestroyMotionPipelines();
    void DestroySessionB();

private:
    bool m_bBuilt           = false;
    u32  m_TotalCount       = 0;
    u32  m_MaxGroupMeshCount = 0;

    xr_vector<TreeIndirectGroup> m_Groups;

    // GPU resources — owned by manager, freed in Destroy().
    CVulkanBuffer* m_TreeMetadataBuffer  = nullptr;
    CVulkanBuffer* m_TreeTransformsBuffer = nullptr;
    CVulkanBuffer* m_TreeIndirectBuffer   = nullptr;  // size = numGroups * maxGroupMesh * sizeof(VkDrawIndexedIndirectCommand)
    CVulkanBuffer* m_TreeDrawCountBuffer  = nullptr;  // size = numGroups * sizeof(u32)

    // Per-texture descriptor sets (one COMBINED_IMAGE_SAMPLER each).
    // Layout is created in CreateTextureDescriptors() and reused by Session B's
    // graphics pipeline.
    VkDescriptorSetLayout      m_TexDescLayout = VK_NULL_HANDLE;
    VkDescriptorPool           m_TexDescPool   = VK_NULL_HANDLE;
    VkSampler                  m_TexSampler    = VK_NULL_HANDLE;
    VK::CVulkanTexture*        m_WaveTex       = nullptr;          // SSFX wind flow map (s_waves)
    VkSampler                  m_WaveSampler   = VK_NULL_HANDLE;   // linear/repeat for s_waves
    xr_vector<VkDescriptorSet> m_TexDescSets;

    // ----- Session B GPU resources -----
    // View-frustum UBO (host-visible, rewritten each frame in Render).
    CVulkanBuffer* m_FrustumUBO = nullptr;

    // Cull compute: descriptor set {meta, frustum, indirect, count}.
    VkDescriptorSetLayout m_CullDescLayout    = VK_NULL_HANDLE;
    VkDescriptorPool      m_CullDescPool      = VK_NULL_HANDLE;
    VkDescriptorSet       m_CullDescSet       = VK_NULL_HANDLE;
    VkPipelineLayout      m_CullPipelineLayout = VK_NULL_HANDLE;
    VkPipeline            m_CullPipeline       = VK_NULL_HANDLE;

    // Graphics set 0 = transforms SSBO (set 1 = m_TexDescSets, reused).
    VkDescriptorSetLayout m_XformDescLayout = VK_NULL_HANDLE;
    VkDescriptorPool      m_XformDescPool   = VK_NULL_HANDLE;
    VkDescriptorSet       m_XformDescSet    = VK_NULL_HANDLE;

    // Graphics pipeline — two variants by UV byte offset (24 / 28).
    VkPipelineLayout m_GfxPipelineLayout = VK_NULL_HANDLE;
    VkPipeline       m_GfxPipeline24     = VK_NULL_HANDLE;
    VkPipeline       m_GfxPipeline28     = VK_NULL_HANDLE;

    // Shadow caster: depth-only alpha-tested variants (tree_depth.{vert,frag},
    // same layout) + a CPU copy of the per-mesh metadata for CPU-side culling.
    VkPipeline       m_DepthPipeline24   = VK_NULL_HANDLE;
    VkPipeline       m_DepthPipeline28   = VK_NULL_HANDLE;

    // Wind-sway MV overlay: reprojects the cur+prev wind pose into the MV target.
    // Own layout (set0 = xform+waves, set1 = diffuse; NO env set) + push; lazy-built.
    // LEQUAL no-write, no bias → depth bit-matches the forward tree draw.
    VkPipelineLayout m_MotionPipelineLayout = VK_NULL_HANDLE;
    VkPipeline       m_MotionPipeline24     = VK_NULL_HANDLE;
    VkPipeline       m_MotionPipeline28     = VK_NULL_HANDLE;
    // Wind history for the MV pass (cur = this frame, prev = last frame). Rolled in
    // Render() each frame; wind is built locally there (no persistent gfx-const).
    Fvector4 m_MvWindParams{}, m_MvWsetup{}, m_MvWindAnim{};
    Fvector4 m_MvWindParamsPrev{}, m_MvWsetupPrev{}, m_MvWindAnimPrev{};
    bool     m_MvWindValid = false;
    xr_vector<GpuTreeMeta> m_MetaCPU;

    // ----- Phase A meshlet clusters (r_vsm_meshlet) — built in Build(), consumed by
    // the two-stage VSM bin (Phase B). Device-local, owned; freed in Destroy().
    bool           m_MeshletsReady    = false;
    u32            m_MeshletTotal     = 0;   // GpuMeshlet[] entries
    u32            m_MeshletIndexTotal = 0;  // u16 indices in the dedicated meshlet IB
    CVulkanBuffer* m_MeshletBuffer      = nullptr;   // GpuMeshlet[]        (m_MeshletTotal)
    CVulkanBuffer* m_MeshletIndexBuffer = nullptr;   // u16 dedicated IB    (m_MeshletIndexTotal)
    CVulkanBuffer* m_TreeMeshletRangeBuffer = nullptr;   // GpuTreeMeshletRange[] (m_TotalCount)

    // ----- VSM caster path (lazy; built on first VsmBin once Session B is up) -----
    // Near/far WIND HYBRID (r_vsm_tree_wind): NEAR trees (< r_vsm_tree_wind_dist, CPU set
    // with hysteresis) cast into the DYNAMIC atlas every frame WITH wind → smooth coherent
    // sway; FAR trees stay in the toroidal STATIC cache, rigid. Boundary crossings emit
    // invalidation spheres (VsmPopTransitions) that the residency pass turns into dirty
    // static pages — the rigid shadow is added/removed the same frame (no ghosts).
    void CreateVsmResources();
    void DestroyVsm();
    // Phase B (r_vsm_meshlet): build the stage-2 meshlet-bin pipeline + meshlet page
    // pipelines + command/group buffers. Called from CreateVsmResources once meshlets exist.
    void CreateMeshletVsmResources();
    // Stage 2 dispatch (mode 0 = static / 1 = dynamic). pageList = slot->page for that atlas.
    void DispatchMeshletBin(VkCommandBuffer cmd, u32 mode, VkBuffer pageList, VkBuffer clipmapUBO);
    // Meshlet render for one atlas (mode 0 = static / 1 = dynamic). Reuses set2 (m_Vsm[Dyn]PageSet).
    void DrawMeshlets(VkCommandBuffer cmd, u32 mode, VkDescriptorSet pageSet, bool wind);
    bool IsMeshletMode() const;   // r_vsm_meshlet on + resources ready

    bool                  m_VsmReady      = false;
    VkPipeline            m_VsmBinPipe    = VK_NULL_HANDLE;   // vsm_tree_bin.comp (mode 0 = static+dirty filter, mode 1 = dynamic+dynUsed)
    VkPipelineLayout      m_VsmBinLayout  = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_VsmBinSetL    = VK_NULL_HANDLE;
    VkDescriptorPool      m_VsmDescPool   = VK_NULL_HANDLE;
    VkDescriptorSet       m_VsmBinSet[VK_FRAMES_IN_FLIGHT]    = {};   // static pass
    VkDescriptorSet       m_VsmDynBinSet[VK_FRAMES_IN_FLIGHT] = {};   // dynamic pass
    VkDescriptorSetLayout m_VsmPageSetL   = VK_NULL_HANDLE;   // set 2: pageList + casterPages + clipmap UBO
    VkDescriptorSet       m_VsmPageSet[VK_FRAMES_IN_FLIGHT]    = {};
    VkDescriptorSet       m_VsmDynPageSet[VK_FRAMES_IN_FLIGHT] = {};
    VkPipelineLayout      m_VsmPageLayout = VK_NULL_HANDLE;
    VkPipeline            m_VsmPagePipe24 = VK_NULL_HANDLE;      // STATIC atlas variant (rigid)
    VkPipeline            m_VsmPagePipe28 = VK_NULL_HANDLE;
    VkPipeline            m_VsmPageDynPipe24 = VK_NULL_HANDLE;   // DYNAMIC atlas variant (wind)
    VkPipeline            m_VsmPageDynPipe28 = VK_NULL_HANDLE;
    CVulkanBuffer* m_VsmCasterPages = nullptr;   // shared ARENA, kVsmTreeArena u32 of (treeIdx<<13)|slot (was trees×cap slicing = 396MB on 193k-tree maps)
    CVulkanBuffer* m_VsmIndirect    = nullptr;   // m_TotalCount cmds — STATIC pass (near trees carry instanceCount 0)
    CVulkanBuffer* m_VsmDynIndirect = nullptr;   // m_TotalCount cmds — DYNAMIC pass (far trees carry instanceCount 0)

    // ----- Phase B meshlet-cull path (r_vsm_meshlet). Stage 2 refines the per-tree page
    // list into per-(meshlet,page) draws pooled into each tree's GROUP command section.
    bool                  m_MeshletVsmReady = false;
    VkPipeline            m_MeshletBinPipe   = VK_NULL_HANDLE;   // vsm_tree_meshlet_bin.comp
    VkPipelineLayout      m_MeshletBinLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_MeshletBinSetL   = VK_NULL_HANDLE;
    VkDescriptorPool      m_MeshletDescPool  = VK_NULL_HANDLE;
    VkDescriptorSet       m_MeshletBinSet[VK_FRAMES_IN_FLIGHT]    = {};   // static stage 2
    VkDescriptorSet       m_MeshletDynBinSet[VK_FRAMES_IN_FLIGHT] = {};   // dynamic stage 2
    VkPipeline            m_MeshletPagePipe24    = VK_NULL_HANDLE;   // STATIC atlas meshlet VS
    VkPipeline            m_MeshletPagePipe28    = VK_NULL_HANDLE;
    VkPipeline            m_MeshletPageDynPipe24 = VK_NULL_HANDLE;   // DYNAMIC atlas meshlet VS
    VkPipeline            m_MeshletPageDynPipe28 = VK_NULL_HANDLE;
    CVulkanBuffer* m_VsmMeshletCmd        = nullptr;   // m_TotalCount*kCmdPerTree cmds (STATIC)
    CVulkanBuffer* m_VsmMeshletCmdDyn     = nullptr;   // m_TotalCount*kCmdPerTree cmds (DYNAMIC)
    CVulkanBuffer* m_VsmMeshletGroupCount = nullptr;   // per-group append counter / draw count (STATIC)
    CVulkanBuffer* m_VsmMeshletGroupCountDyn = nullptr;// (DYNAMIC)
    CVulkanBuffer* m_TreeGroupBuffer      = nullptr;   // u32/tree -> group index
    CVulkanBuffer* m_GroupInfoBuffer      = nullptr;   // uvec2/group -> (cmdBase, cmdCap)
    xr_vector<u32> m_MeshletGroupBase;                 // CPU copy for the render (cmd offset per group)
    xr_vector<u32> m_MeshletGroupCap;                  // CPU copy for the render (maxDrawCount per group)
    CVulkanBuffer* m_MeshletStats         = nullptr;   // [0]=commands [1]=overflow (both passes accumulate)
    CVulkanBuffer* m_MeshletStatsRB       = nullptr;
    u32*           m_MeshletStatsPtr      = nullptr;

    CVulkanBuffer* m_VsmStats       = nullptr;
    CVulkanBuffer* m_VsmStatsRB     = nullptr;   // host readback (r_vsm_debug diagnostics)
    u32*           m_VsmStatsPtr    = nullptr;
    // Forward drawn-count readback (r_profiler): the main-view cull writes the
    // per-group instance counts GPU-side only — this mirrors them to the host
    // (a frame or two stale, no fence — diagnostic) so "how many trees do we
    // actually draw" exists as a number. Lazy-created on first profiled frame.
    CVulkanBuffer* m_FwdCountRB     = nullptr;
    u32*           m_FwdCountPtr    = nullptr;
    u32            m_FwdLastLog     = 0;
    // Visible-tree bitset (r_profiler): tree.frag marks trees whose fragments
    // survive alpha+early-Z — ≈ "trees with visible pixels". Set-0 binding 2 of
    // the gfx layout; cleared per frame in Render, snapshot copied to the RB
    // BEFORE the clear (so the readback is last frame's set).
    CVulkanBuffer* m_SeenBits       = nullptr;
    CVulkanBuffer* m_SeenBitsRB     = nullptr;
    u32*           m_SeenBitsPtr    = nullptr;
    // VSM bin diag bitsets (binding 12 of the bin layout): [0..W) distinct trees
    // binned to the DYN atlas per frame, [W..2W) trees ever binned to the cached
    // STATIC atlas (accumulated).
    CVulkanBuffer* m_VsmTreeBits    = nullptr;
    CVulkanBuffer* m_VsmTreeBitsRB  = nullptr;
    u32*           m_VsmTreeBitsPtr = nullptr;
    CVulkanBuffer* m_VsmNearFlags[VK_FRAMES_IN_FLIGHT] = {};   // host-visible u32/tree: bit0 = near (dynamic wind), bit1 = hull tier (r_vsm_tree_hull)
    void*          m_VsmNearPtr[VK_FRAMES_IN_FLIGHT]   = {};   // legacy (device-local flags are not mapped anymore)
    xr_vector<u8>       m_VsmNearCPU;        // current near set (CPU truth, hysteresis)
    xr_vector<u32>      m_VsmNearU32;        // widened upload scratch (shader reads u32 per tree)
    xr_vector<Fvector4> m_VsmPendingInval;   // world spheres of boundary-crossing trees (→ residency)
    u32            m_VsmNearCount   = 0;     // CPU near-set size this frame (diagnostic)
    u32            m_VsmHullCount   = 0;     // hull-tier subset of the near set (diagnostic)
    u32            m_VsmHullFarCount = 0;    // far/static hull tier size (r_vsm_tree_hull 2, diagnostic)
    u32            m_VsmLastLog     = 0;
    u32            m_VsmSlot        = 0;          // frame-in-flight ring for the descriptor sets

    // ----- Impostor crown shadows (r_vsm_tree_impostor) — near-tree DYNAMIC pass only.
    // Bakes a per-UNIQUE-MESH silhouette atlas at load (8 azimuth facets/row, alpha-tested
    // from the crown diffuse), then replaces the near crown MESH raster with ONE sun-facing
    // billboard per tree (2 tris) routed through the same VSM page mechanism. Kills the
    // geometry/raster cost of DynTrees under wind. See tree_vsm_impostor.* / BuildImpostorAtlas.
    void BuildImpostorAtlas(const xr_vector<::vkFTreeVisual*>& trees,
                            const xr_vector<GpuTreeMeta>& meta,
                            const xr_vector<u32>& treeTexIdx);
    void CreateImpostorResources();   // billboard pipeline + sets + indirect (from CreateVsmResources)
    void DestroyImpostor();
    bool IsImpostorMode() const;      // r_vsm_tree_impostor on + atlas + resources ready
    // Fills m_VsmImpostorDynIndirect from m_VsmDynIndirect (indexCount→6, non-indexed);
    // records the copy compute + a compute→compute barrier. Call at the end of VsmBinDyn.
    void DispatchImpostorCmd(VkCommandBuffer cmd);

    // ----- Crown-HULL shadow casters (r_vsm_tree_hull) — near-tree DYNAMIC pass tiering.
    // The AC-Shadows-style caster LOD: near trees WITHIN r_vsm_tree_hull_dist keep the full
    // alpha-tested crown mesh (dappling stays perfect where the player looks); near trees
    // BEYOND it cast from a baked low-poly OPAQUE hull (ellipsoid lobes over k-means vertex
    // clusters, per unique mesh) — no texture fetch / no discard / early-Z, attacking the
    // proven alpha-test-fill bound of VSM/DynTrees. Hull geometry is built unconditionally
    // in BuildMeshlets (shares the mesh readback, ~KBs) so the cvar toggles LIVE; the swap
    // happens per frame in vsm_hull_cmd.comp (crown cmd zeroed → plain AND meshlet paths
    // skip the tree automatically). Far/static cache is untouched.
    void CreateHullResources();       // cmd compute + opaque hull pipeline (from CreateVsmResources)
    void DestroyHull();
    bool IsHullMode() const;          // cvar on + geometry + pipelines ready (excludes impostor mode)
    // Swaps crown→hull indirect cmds for hull-tier trees in BOTH bins (zeroes the crown
    // cmds; static tier only at r_vsm_tree_hull 2). One compute→compute barrier covers
    // the stage-1 writes. Call at the end of VsmBinDyn (both bins recorded by then).
    void DispatchHullCmd(VkCommandBuffer cmd);
    // ONE multi-draw of the tier's hull trees (staticPass: rigid wind + static atlas).
    void DrawHulls(VkCommandBuffer cmd, u32 slot, bool staticPass);
    void DrawVoxCasters(VkCommandBuffer cmd, u32 slot, bool staticPass);
    // Stage-2 per-page BRICK cull (r_vsm_tree_hull_vox_cull, vsm_vox_cull.comp) — the
    // UE-parity piece: only bricks overlapping a page rasterize into it, instead of
    // the whole slice sweeping the VS for every page of the tree's shadow column.
    // Compacts (tree,page) pairs into one draw each + a surviving-brick remap list;
    // DrawVoxCasters then uses vkCmdDrawIndirectCount over the compacted stream.
    // Called from VsmBinMeshlets (after the caller's stage-1→stage-2 barrier, which
    // orders vsm_hull_cmd's writes before this pass's reads).
    // pageMaxBlk/staticPT (shadow-HZB, r_vsm_hzb): per-STATIC-slot 8×8 block occluder
    // maxes + the static page table — bricks fully behind the cached canopy/terrain are
    // culled per (brick,page). NULL → occlusion test off (XY-only compaction as before).
    void DispatchVoxCull(VkCommandBuffer cmd, VkBuffer pageListDyn, VkBuffer pageListStatic, VkBuffer clipmapUBO,
                         VkBuffer pageMaxBlk, VkBuffer staticPT);

    bool           m_HullDataReady   = false;     // geometry baked + uploaded (Build time)
    bool           m_HullReady       = false;     // pipelines/sets built (lazy, CreateVsmResources)
    u32            m_HullLobeTotal   = 0;         // diag: ellipsoid lobes baked
    CVulkanBuffer* m_HullVB          = nullptr;   // tightly packed vec3 positions (stride 12)
    CVulkanBuffer* m_HullIB          = nullptr;   // u16, per-species ranges (relative to vb_first)
    CVulkanBuffer* m_HullInfoBuffer  = nullptr;   // GpuTreeHullInfo[m_TotalCount]
    CVulkanBuffer* m_VsmHullIndirect  = nullptr;  // VkDrawIndexedIndirectCommand/tree (DYN hull tier)
    CVulkanBuffer* m_VsmHullIndirectS = nullptr;  // VkDrawIndexedIndirectCommand/tree (STATIC far tier, mode 2)
    VkDescriptorSetLayout m_HullCmdSetL   = VK_NULL_HANDLE;
    VkDescriptorPool      m_HullCmdPool   = VK_NULL_HANDLE;
    VkDescriptorSet       m_HullCmdSet[VK_FRAMES_IN_FLIGHT] = {};   // ring (nearFlags is per-frame)
    VkPipelineLayout      m_HullCmdLayout = VK_NULL_HANDLE;
    VkPipeline            m_HullCmdPipe   = VK_NULL_HANDLE;
    VkPipeline            m_VsmHullPipe   = VK_NULL_HANDLE;   // opaque depth-only hull raster, DYN atlas (reuses m_VsmPageLayout)
    VkPipeline            m_VsmHullPipeS  = VK_NULL_HANDLE;   // STATIC atlas variant (tree_vsm_hull_s.vert, rigid)
    // Voxel-cloud SHADOW caster (r_vsm_tree_hull_vox > 0 supersedes the merged shell):
    // the shadow casts from the SAME cubes the viewmode shows — camera-distance LOD +
    // continuous density from the per-tree CHOICE ring (CPU-computed in VsmUpdateNearSet),
    // and a dithered CROSSFADE band below r_vsm_tree_hull_dist where the real crown
    // still casts while the cubes dissolve in (approach = shadow morphs to true leaves).
    CVulkanBuffer*        m_VsmVoxIndirect  = nullptr;        // VkDrawIndirectCommand/tree (DYN)
    CVulkanBuffer*        m_VsmVoxIndirectS = nullptr;        // VkDrawIndirectCommand/tree (STATIC, mode 2)
    CVulkanBuffer*        m_VoxChoiceBuf[VK_FRAMES_IN_FLIGHT] = {};   // host-visible ring: uvec4/tree
    void*                 m_VoxChoicePtr[VK_FRAMES_IN_FLIGHT] = {};
    xr_vector<u32>        m_VoxChoiceCPU;                     // per tree ×4: brickFirst, brickCount, cellBits, fade16<<16|amp16 (amp = band wind amplitude, 0 = rigid)
    // STATIC-tier voxel LOD ratchet: cube size at the tree's last page invalidation.
    // Cached static pages never re-render on a choice change alone, so a far tree's
    // shadow keeps the cube size from whenever its pages last drew; when the current
    // cube leaves ±1.5× of this anchor, VsmUpdateNearSet queues an invalidation
    // circle and re-anchors (0 = not tracked / dyn tier).
    xr_vector<float>      m_VoxLodSizeCPU;
    VkDescriptorSetLayout m_VoxCasterSetL   = VK_NULL_HANDLE; // set3: voxel SSBO + choice
    VkDescriptorPool      m_VoxCasterPool   = VK_NULL_HANDLE;
    VkDescriptorSet       m_VoxCasterSet[VK_FRAMES_IN_FLIGHT] = {};
    VkPipelineLayout      m_VoxCasterLayout = VK_NULL_HANDLE; // sets 0-2 = m_VsmPageLayout's, +set3
    VkPipeline            m_VsmVoxPipe  = VK_NULL_HANDLE;     // DYN atlas (tree_vsm_vox.vert)
    VkPipeline            m_VsmVoxPipeS = VK_NULL_HANDLE;     // STATIC atlas (tree_vsm_vox_s.vert)
    u32                   m_VoxCastCount = 0;                 // diag: trees casting voxels this frame
    u32                   m_VoxCastSlice = 0;                 // diag: summed caster slice (bricks; VS cost ∝ survivors × 6)
    u32                   m_VoxBandCount = 0;                 // diag: trees in the crown↔voxel crossfade band
    // Stage-2 brick cull resources (see DispatchVoxCull). The remap list is created
    // whenever bricks exist (set3 binding 2 must stay valid even if the cull pipeline
    // fails and the un-culled path draws).
    static constexpr u32  kVoxCullCmdCap  = 32768;            // compacted (tree,page) draws per tier
    static constexpr u32  kVoxCullListCap = 1u << 20;         // surviving brick indices (shared by both tiers)
    bool                  m_VoxCullReady  = false;
    bool                  m_VoxCullActive = false;            // dispatched this frame → draws use the compacted stream
    VkDescriptorSetLayout m_VoxCullSetL   = VK_NULL_HANDLE;
    VkDescriptorPool      m_VoxCullPool   = VK_NULL_HANDLE;
    VkDescriptorSet       m_VoxCullSet[VK_FRAMES_IN_FLIGHT] = {};
    VkPipelineLayout      m_VoxCullLayout = VK_NULL_HANDLE;
    VkPipeline            m_VoxCullPipe   = VK_NULL_HANDLE;
    CVulkanBuffer*        m_VoxCullCmdD   = nullptr;          // VkDrawIndirectCommand[kVoxCullCmdCap] (dyn)
    CVulkanBuffer*        m_VoxCullCmdS   = nullptr;          // (static)
    CVulkanBuffer*        m_VoxCullList   = nullptr;          // u32[kVoxCullListCap] global brick indices
    CVulkanBuffer*        m_VoxCullStats  = nullptr;          // [0]=cmdD [1]=cmdS [2]=list [3]=overflow [4]=hzbCulled — words 0/1 double as draw counts
    CVulkanBuffer*        m_VoxCullStatsRB = nullptr;         // host readback of the above (r_vsm_debug telemetry)
    u32*                  m_VoxCullStatsPtr = nullptr;
    // Compact dispatch: the cull runs one workgroup per tree WITH a live brick choice
    // (CPU list from VsmUpdateNearSet), not per tree in the level — the fixed overhead
    // of ~2600 mostly-empty workgroups ×2 modes was a measurable slice of VSM/Bins.
    xr_vector<u32>        m_VoxCastListCPU;                                 // tree indices with choice.count > 0 this frame
    CVulkanBuffer*        m_VoxCullTreeList[VK_FRAMES_IN_FLIGHT] = {};      // host-visible ring: the list, uploaded in VsmBin
    void*                 m_VoxCullTreeListPtr[VK_FRAMES_IN_FLIGHT] = {};
    u32                   m_VoxCullTreeN = 0;                               // entries in this frame's slot
    VkPipeline            m_HullDebugPipe = VK_NULL_HANDLE;   // r_vsm_tree_hull_debug: shaded hull/lobe fallback overlay
    xr_vector<GpuTreeHullInfo> m_HullInfoCPU;                 // per-tree hull ranges (shadow shell; debug fallback draws)
    // Crown VOXEL CLOUD (UE Nanite-foliage style, r_vsm_tree_hull_vox): per species,
    // kHullLods levels of individual colored cubes baked from the leaf-card triangles;
    // voxel GRID cell doubles per level (finest = level 0 = maxExt / r_vsm_tree_hull_vox).
    // The RENDERED cube edge is CONTINUOUS — dist × r_vsm_tree_hull_vox_px angular size,
    // so voxels grow smoothly with every meter (constant projected px, no size steps);
    // the baked levels only supply positions/density, picked as the largest cell ≤ the
    // current cube size (grid swaps hidden by the dither crossfade).
    static constexpr u32       kHullLods = 5;
    CVulkanBuffer*             m_VoxVB = nullptr;             // GpuTreeVoxel[], per-instance vertex stream
    CVulkanBuffer*             m_BrickVB = nullptr;           // GpuTreeBrick[] SSBO — the shadow caster's unit
    xr_vector<TreeVoxLod>      m_BrickLodInfoCPU;             // per-tree × kHullLods brick slices (finest→coarsest)
    u32                        m_BrickTotal = 0;              // diag: bricks baked (all species, all LODs)
    VkPipeline                 m_VoxDebugPipe = VK_NULL_HANDLE;   // instanced-cube voxel viewmode
    xr_vector<TreeVoxLod>      m_VoxLodInfoCPU;               // per-tree × kHullLods (finest→coarsest)
    u32                        m_VoxTotal = 0;                // diag: voxels baked (all species, all LODs)
    xr_vector<u8>              m_WindClassCPU;                // per-tree wind class (2=foliage 1=bark 0=rigid) — hull tier is CROWNS-ONLY

    // Baked silhouette atlas (R8 coverage; FACETS cols × m_ImpostorMeshCount rows).
    VkImage        m_ImpostorAtlas      = VK_NULL_HANDLE;
    VmaAllocation  m_ImpostorAtlasAlloc = VK_NULL_HANDLE;
    VkImageView    m_ImpostorAtlasView  = VK_NULL_HANDLE;
    VkSampler      m_ImpostorSampler    = VK_NULL_HANDLE;
    u32            m_ImpostorMeshCount  = 0;   // atlas rows (unique meshes)
    CVulkanBuffer* m_ImpostorSilIdx     = nullptr;   // u32/tree → atlas row
    CVulkanBuffer* m_VsmImpostorDynIndirect = nullptr;   // VkDrawIndirectCommand/tree (dyn near set)

    bool                  m_ImpostorReady    = false;
    VkDescriptorSetLayout m_ImpostorSetL     = VK_NULL_HANDLE;   // set1: meta + silIdx + atlas
    VkDescriptorPool      m_ImpostorPool     = VK_NULL_HANDLE;
    VkDescriptorSet       m_ImpostorSet      = VK_NULL_HANDLE;
    VkPipelineLayout      m_ImpostorLayout   = VK_NULL_HANDLE;
    VkPipeline            m_ImpostorPipe     = VK_NULL_HANDLE;
    // Indirect builder (VkDrawIndexedIndirectCommand → VkDrawIndirectCommand, indexCount→6).
    VkDescriptorSetLayout m_ImpostorCmdSetL  = VK_NULL_HANDLE;
    VkDescriptorSet       m_ImpostorCmdSet   = VK_NULL_HANDLE;
    VkPipelineLayout      m_ImpostorCmdLayout = VK_NULL_HANDLE;
    VkPipeline            m_ImpostorCmdPipe  = VK_NULL_HANDLE;
};

}  // namespace VK
