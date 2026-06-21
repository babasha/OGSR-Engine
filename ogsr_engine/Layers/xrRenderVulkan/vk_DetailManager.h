// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan - GPU-driven grass / detail-objects manager.
//
// Three-session port from monolith. THIS HEADER COVERS SESSION A ONLY:
//   - Asset load (level.details)
//   - Heightmap bake (collision tris → R32_SFLOAT image, ~1 m/texel)
//   - Slot data SSBO upload (DetailSlot → GpuSlotPacked, 32 B each)
//   - Object info SSBO upload (per-detail-type scale/radius/flags)
//
// Sessions B/C add: visible/indirect/atomic SSBOs, gen + trail compute
// pipelines, graphics pipeline, render path, optional bindless multi-draw.
// See `memory/vulkan_grass_port_plan.md` for the full plan.

#pragma once
#include "vk_core.h"
#include "vk_buffer.h"
#include "vk_Detail.h"
#include "../xrRender/DetailFormat.h"

namespace VK
{

class CVulkanTexture;
struct FrameContext;


// ============================================================================
// Constants — pinned to monolith's values; do not tune without porting plan
// reference (`memory/vulkan_grass_port_plan.md`).
// ============================================================================
inline constexpr u32   dm_max_objects     = 64;
inline constexpr float dm_slot_size       = DETAIL_SLOT_SIZE;   // 2.0 m
inline constexpr u32   GPU_MAX_OBJ_TYPES  = 64;
// Output capacity. 1.5M × 64 B = 96 MB — matches monolith default,
// covers max-radius (250m) × max-density (0.2) without overflow.
inline constexpr u32   GPU_OUTPUT_CAPACITY = 1500000u;
inline constexpr u32   MAX_GRASS_INTERACTORS = 4;

// ============================================================================
// Wire-format structs that hit the GPU (Session B will reference them).
// Sized exactly per monolith — keep static_asserts happy.
// ============================================================================

// Per-instance compacted by the gen compute, used as VB binding 1 in draw.
// Channel layout: row0..2 = 3×4 transform, color = (sun, trail, objId, hemi).
struct DetailInstance
{
    Fvector4 row0, row1, row2;
    Fvector4 color;
};
static_assert(sizeof(DetailInstance) == 64, "DetailInstance must be 64 B");

// Per-slot packed payload (one per (size_x*size_z) entry).
struct GpuSlotPacked
{
    float y_base;       // DetailSlot::r_ybase()
    float y_height;     // DetailSlot::r_yheight()
    u32   ids;          // id0:8 | id1:8 | id2:8 | id3:8
    u32   lighting;     // c_dir:16 | c_hemi:16 (UNORM-style 16.16)
    u32   palette0;     // per-obj corner alphas, 4×8 bits
    u32   palette1;
    u32   palette2;
    u32   palette3;
};
static_assert(sizeof(GpuSlotPacked) == 32, "GpuSlotPacked must be 32 B");

// Per-detail-type metadata (mirrors CDetail render params).
struct GpuDetailObjInfo
{
    float minScale;
    float maxScale;
    float bvRadius;
    u32   flags;
};
static_assert(sizeof(GpuDetailObjInfo) == 16, "GpuDetailObjInfo must be 16 B");

// ============================================================================
// Push-constant blocks
// ============================================================================

// Compute generator push (208 B) — viewport / culling / slot range params.
struct DetailGenPushConstants
{
    Fmatrix  viewProj;          // 64
    Fvector4 frustumPlanes[6];  // 96  (left, right, bottom, top, near, far)
    Fvector4 cameraPos;         // 16  (xyz=eye, w=fTimeGlobal)
    Fvector4 fadeParams;        // 16  (fadeStartSq, fadeLimitSq, fadeRangeSq, density)
    int      slotMinSX;
    int      slotMinSZ;
    int      slotCountX;
    int      slotCountZ;
};
static_assert(sizeof(DetailGenPushConstants) == 208, "Gen push must be 208 B");

// Generator UBO (64 B) — params that don't fit in push. Mapped host-visible
// and rewritten each frame (or on level load if unchanging).
struct DetailGenUBO
{
    float hmOriginX, hmOriginZ;
    float hmInvScaleX, hmInvScaleZ;
    u32   totalPositions;
    u32   numObjTypes;
    u32   outputCapacity;
    u32   posPerSlot;
    float dtOffsX, dtOffsZ;
    float dtSizeX, dtSizeZ;
    float slotSize;
    float detailHeight;
    u32   hmWidth, hmHeight;
};
static_assert(sizeof(DetailGenUBO) == 64, "Gen UBO must be 64 B");

// Graphics push (224 B) — SSFX wind / lighting / interactors.
struct DetailGfxPushConstants
{
    Fmatrix  mViewProj;                            // 64
    Fvector4 wind_params;                          // 16  (wind_direction, wind_velocity, treeAmplitude, _)
    Fvector4 wsetup_grass;                         // 16  SSFX (animspeed, turbulence, push, wave)
    Fvector4 wind_anim;                            // 16  Environment.wind_anim (xyz drift) + w = minWindSpeed
    Fvector4 vConsts;                              // 16  (1, 1, sun.y, ambient_floor)
    Fvector4 vInteractors[MAX_GRASS_INTERACTORS];  // 64  xyz=pos, w=radius (0=unused)
    Fvector4 vSunColor;                            // 16  env sun colour (rgb); w unused
    Fvector4 vHemiColor;                           // 16  env hemi colour (rgb); w unused
};
static_assert(sizeof(DetailGfxPushConstants) == 224, "Gfx push must be 224 B");

// HZB build push (32 B) — matches hzb_build.comp.glsl. srcMip/dstMip select
// levels; isFirstPass=1 reads the depth buffer, =0 reads the previous HZB mip.
struct HZBBuildPush
{
    s32 srcW, srcH;     // ivec2 srcSize
    s32 dstW, dstH;     // ivec2 dstSize
    u32 srcMip, dstMip;
    u32 isFirstPass;
    u32 _pad;
};
static_assert(sizeof(HZBBuildPush) == 32, "HZB build push must be 32 B");

// ============================================================================
// CDetailManager — Sessions A + B (B-min: gen compute + forward draw).
// Trail map / character interaction / bindless multi-draw deferred.
// ============================================================================
class CDetailManager
{
public:
    struct SSwingValue
    {
        float rot1 = 0.0f, rot2 = 0.0f;
        float amp1 = 0.0f, amp2 = 0.0f;
        float speed = 0.0f;
        void lerp(const SSwingValue& v1, const SSwingValue& v2, float factor);
    };

    typedef svector<VK::CDetail*, dm_max_objects> DetailVec;

public:
    // ----- Asset data -------------------------------------------------------
    IReader*     dtFS    = nullptr;
    DetailHeader dtH{};
    DetailSlot*  dtSlots = nullptr;        // pointer into VFS-mapped chunk 2
    DetailSlot   DS_empty{};

    DetailVec    objects;

    // ----- Wind animation params (env-driven, used in Session B render) ----
    SSwingValue  swing_desc[2]{};          // [0]=normal, [1]=fast
    SSwingValue  swing_current{};
    float        m_time_rot_1     = 0.0f;
    float        m_time_rot_2     = 0.0f;
    float        m_time_pos       = 0.0f;
    float        m_global_time_old = 0.0f;
    float        fade_distance    = 60.0f;

    // ----- VSM grass-shadow caster access (read-only) -----------------------
    // vk_vsm rasterizes NEAR grass into the virtual shadow atlas from the SAME
    // GPU-driven instance buffer (1 frame stale — grass gen runs after VSM in the
    // frame). The normal grass gen+draw is untouched; this is a read-only side path.
    VkBuffer Vsm_VisibleSSBO()  const;
    VkBuffer Vsm_IndirectBuf()  const;
    u32      Vsm_TypeCount()    const;
    u32      Vsm_SectionSize()  const;     // GPU_OUTPUT_CAPACITY / max(types,1) — VisibleSSBO per-type stride
    u32      Vsm_VertexStride() const;     // grass mesh binding-0 stride (sizeof CDetail::Vertex)
    bool     Vsm_TypeMesh(u32 i, VkBuffer& vb, VkBuffer& ib, u32& indexCount) const;
    VkDescriptorSetLayout Vsm_GfxSetLayout() const;    // diffuse-sampler set layout (for the grass-page alpha test)
    VkDescriptorSet       Vsm_TypeDiffuseSet(u32 i) const;

private:
    // ----- Session B per-frame state ---------------------------------------
    DetailGfxPushConstants m_GfxConstants{};
    struct FrameState
    {
        int  slotMinSX     = 0;
        int  slotMinSZ     = 0;
        int  slotCountX    = 0;
        int  slotCountZ    = 0;
        u32  totalPositions = 0;
        u32  groupCount    = 0;
        u32  posPerSlot    = 25;     // (d_size+1)² for default density 0.6 → d_size=4
        u32  d_size        = 4;
        bool valid         = false;
    } m_Frame;

    // ----- Session A GPU resources -----------------------------------------

    // R32F heightmap baked from static collision triangles. Sentinel
    // value 9000.0f marks texels no triangle covered (compute shader
    // tests `abs(h) < 9000` and falls back to the slot's y_base).
    VkImage         m_HeightmapImage   = VK_NULL_HANDLE;
    VmaAllocation   m_HeightmapAlloc   = VK_NULL_HANDLE;
    VkImageView     m_HeightmapView    = VK_NULL_HANDLE;
    VkSampler       m_HeightmapSampler = VK_NULL_HANDLE;
    u32             m_HeightmapW       = 0;
    u32             m_HeightmapH       = 0;
    float           m_HMOriginX        = 0.0f;
    float           m_HMOriginZ        = 0.0f;
    float           m_HMWorldSizeX     = 0.0f;
    float           m_HMWorldSizeZ     = 0.0f;

    // Packed slot payload — size_x * size_z * 32 B.
    VK::CVulkanBuffer* m_SlotDataSSBO  = nullptr;
    u32                m_TotalSlots    = 0;

    // Per-type metadata — objects.size() * 16 B.
    VK::CVulkanBuffer* m_ObjInfoSSBO   = nullptr;

    // ----- Session B GPU resources -----------------------------------------

    // Compute output: per-instance compacted draw stream. Bound as VB
    // binding 1 (INSTANCE rate) by the graphics draw.
    VK::CVulkanBuffer* m_VisibleSSBO   = nullptr;     // 96 MB
    VK::CVulkanBuffer* m_IndirectCmdBuf= nullptr;     // 64 × 20 B
    VK::CVulkanBuffer* m_AtomicCounters= nullptr;     // 132 × 4 B
    VK::CVulkanBuffer* m_GenUBO        = nullptr;     // 64 B host-visible

    // Compute (generator) pipeline.
    VkPipeline             m_GenPipeline      = VK_NULL_HANDLE;
    VkPipelineLayout       m_GenPipelineLayout= VK_NULL_HANDLE;
    VkDescriptorSetLayout  m_GenDescLayout    = VK_NULL_HANDLE;
    VkDescriptorPool       m_GenDescPool      = VK_NULL_HANDLE;
    VkDescriptorSet        m_GenDescSet       = VK_NULL_HANDLE;

    // 1×1 placeholder textures for HZB / TrailMap bindings (B-min skips
    // both occlusion culling and footprint memory). HZB white = depth 1.0
    // never culls; TrailMap black = no press-down.
    VkImage         m_DummyHZBImage   = VK_NULL_HANDLE;
    VmaAllocation   m_DummyHZBAlloc   = VK_NULL_HANDLE;
    VkImageView     m_DummyHZBView    = VK_NULL_HANDLE;
    VkImage         m_DummyTrailImage = VK_NULL_HANDLE;
    VmaAllocation   m_DummyTrailAlloc = VK_NULL_HANDLE;
    VkImageView     m_DummyTrailView  = VK_NULL_HANDLE;

    // ----- HZB: Hierarchical-Z occlusion pyramid for grass culling ---------
    // Built each frame from the scene depth buffer (after Pass_World) via a
    // max-reduce mip chain; the gen compute samples it (binding 6) to drop
    // instances hidden behind world geometry. mip0 is half the depth res.
    VkImage         m_HZBImage        = VK_NULL_HANDLE;
    VmaAllocation   m_HZBAlloc        = VK_NULL_HANDLE;
    VkImageView     m_HZBView         = VK_NULL_HANDLE;   // full-mip sampled (gen binding 6 + build src)
    xr_vector<VkImageView> m_HZBMipViews;                 // per-mip single-level storage (build dst)
    VkSampler       m_HZBSampler      = VK_NULL_HANDLE;   // nearest, clamp-to-edge, all mips
    VkImageView     m_DepthSampleView = VK_NULL_HANDLE;   // DEPTH-aspect view of swapchain depth (first pass src)
    u32             m_HZBWidth        = 0;                // mip0 dims
    u32             m_HZBHeight       = 0;
    u32             m_HZBMipCount     = 0;
    VkExtent2D      m_HZBDepthExtent  = { 0, 0 };         // depth extent HZB was sized for (resize detect)
    bool            m_bHZBLayoutInit  = false;            // HZB transitioned UNDEFINED→GENERAL once
    u32             m_HZBBuiltFrame   = 0xFFFFFFFFu;      // Device.dwFrame the pyramid was last built (build-once-per-frame guard)

    // HZB build compute pipeline. One descriptor set per mip pass.
    VkPipeline             m_HZBPipeline       = VK_NULL_HANDLE;
    VkPipelineLayout       m_HZBPipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout  m_HZBDescLayout     = VK_NULL_HANDLE;
    VkDescriptorPool       m_HZBDescPool       = VK_NULL_HANDLE;
    xr_vector<VkDescriptorSet> m_HZBDescSets;

    // Graphics (forward draw) pipeline.
    VkPipeline             m_GfxPipeline       = VK_NULL_HANDLE;
    VkPipelineLayout       m_GfxPipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout  m_GfxDescLayout     = VK_NULL_HANDLE;
    VkDescriptorPool       m_GfxDescPool       = VK_NULL_HANDLE;
    // One descriptor set per detail type (single sampler binding = diffuse).
    xr_vector<VkDescriptorSet> m_GfxDescSets;

    // Per-detail-type loaded diffuse textures (one entry per `objects[]`).
    xr_vector<VK::CVulkanTexture*> m_DetailTextures;
    VkSampler                      m_DetailSampler = VK_NULL_HANDLE;
    VK::CVulkanTexture*            m_WaveTex       = nullptr;  // SSFX wind flow map (s_waves, wind_wave.dds)

    bool            m_bCreated         = false;

public:
    CDetailManager();
    virtual ~CDetailManager();

    void Load();          // open level.details, bake heightmap, upload SSBOs
    void Unload();
    bool IsLoaded() const { return m_bCreated; }

    // ----- Session B render entry ------------------------------------------
    // PrepareFrame fills m_GfxConstants + m_Frame from camera/wind/density.
    // Render does the full per-frame pipeline (clear → barrier → trail-skip
    // → gen dispatch → barrier → atomic→indirect copy → barrier → draw).
    // Called from CRender::Render() between Pass_World and Pass_Sky.
    void Render(struct VK::FrameContext& ctx);

    // World-slot lookup. Returns DS_empty for out-of-bounds queries
    // (id0..3 = ID_Empty marker).
    DetailSlot& QueryDB(int sx, int sz);

    // ----- Hi-Z occlusion pyramid sharing (Phase A: WorldGPU::CullColor) -----
    // Pass_World builds the pyramid from THIS frame's PREPASS depth (before its
    // color pass) so the GPU-driven static color pass can occlusion-cull against
    // it. The build is frame-stamped (m_HZBBuiltFrame) → grass's own BuildHZB at
    // the top of Render() then no-ops, so the pyramid is built once per frame
    // regardless of which caller runs first. HZB lives in VK_IMAGE_LAYOUT_GENERAL.
    void        BuildHZBForFrame(struct VK::FrameContext& ctx) { BuildHZB(ctx); }
    bool        HZBReady()   const { return m_HZBImage != VK_NULL_HANDLE && m_HZBView != VK_NULL_HANDLE && m_HZBPipeline != VK_NULL_HANDLE; }
    VkImageView HZBView()    const { return m_HZBView; }
    VkSampler   HZBSampler() const { return m_HZBSampler; }

private:
    // Session A
    void BakeHeightmap();
    void UploadSlotData();
    void UploadObjInfo();

    // Session B
    void CreateGpuBuffers();
    void DestroyGpuBuffers();
    void CreateDummyTextures();
    void DestroyDummyTextures();
    // HZB occlusion pyramid (sized to depth extent; recreated on resize).
    void CreateHZB(u32 depthW, u32 depthH);
    void DestroyHZB();
    void BuildHZB(VK::FrameContext& ctx);   // per-frame: depth → Hi-Z mip chain
    void CreateGpuGenPipeline();
    void DestroyGpuGenPipeline();
    void CreateGfxPipeline();
    void DestroyGfxPipeline();
    void LoadDetailTextures();
    void DestroyDetailTextures();

    void PrepareFrame(const VK::FrameContext& ctx);
    void ExtractFrustumPlanes(const Fmatrix& vp, Fvector4 planes[6]) const;
    void UpdateGenDescriptors();   // rebinds when textures might rotate
};

}  // namespace VK
