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
    void VsmRender(VkCommandBuffer cmd, VkBuffer pageList, VkBuffer clipmapUBO);
    void VsmRenderDyn(VkCommandBuffer cmd, VkBuffer dynPageList, VkBuffer clipmapUBO);

    bool IsBuilt() const { return m_bBuilt; }
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
    CVulkanBuffer* m_VsmCasterPages = nullptr;   // m_TotalCount * kVsmTreeCap u32 (shared: near/far slices disjoint)
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
    CVulkanBuffer* m_VsmNearFlags[VK_FRAMES_IN_FLIGHT] = {};   // host-visible u32/tree: 1 = near (dynamic wind) set
    void*          m_VsmNearPtr[VK_FRAMES_IN_FLIGHT]   = {};
    xr_vector<u8>       m_VsmNearCPU;        // current near set (CPU truth, hysteresis)
    xr_vector<Fvector4> m_VsmPendingInval;   // world spheres of boundary-crossing trees (→ residency)
    u32            m_VsmNearCount   = 0;     // CPU near-set size this frame (diagnostic)
    u32            m_VsmLastLog     = 0;
    u32            m_VsmSlot        = 0;          // frame-in-flight ring for the descriptor sets
};

}  // namespace VK
