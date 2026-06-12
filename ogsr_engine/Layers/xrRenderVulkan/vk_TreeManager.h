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

// Tree graphics push constants (72 B, VS+FS). Matches tree.vert/frag.
struct TreeGfxPush
{
    Fmatrix  mViewProj;   // 64 B  world → clip
    float    uvScale;     //  4 B  1/2048 (FTreeVisual_quant = 32768/16)
    float    alphaRef;    //  4 B  fragment alpha cutoff
    float    _pad0;       //  4 B  (vec4 below needs 16-byte alignment → offset 80)
    float    _pad1;       //  4 B
    Fvector4 vSunColor;   // 16 B  env sun colour (rgb); w unused
    Fvector4 vHemiColor;  // 16 B  env hemi colour (rgb); w unused
};
static_assert(sizeof(TreeGfxPush) == 112, "TreeGfxPush must be 112 B");

// View frustum UBO (binding 1 of the cull set). 6 normalized planes, std140.
struct TreeFrustumUBO
{
    Fvector4 planes[6];  // 96 B
};

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
    void RenderDepth(VkCommandBuffer cmd, const Fmatrix& lightVP, s32 cascade = -1,
                     const CFrustum* frustum = nullptr);

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
};

}  // namespace VK
