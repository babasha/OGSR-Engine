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

extern int   ps_r_vsm;
extern int   ps_r_vsm_debug;
extern int   ps_r_vsm_hzb;   // shadow-HZB: cull casters fully behind cached occluders (kills VSMrender overdraw)
extern float ps_r_vsm_base;       // clipmap level-0 extent (m) → finest texel = base/4096 (live, settings-bound)
extern float ps_r_vsm_bias;       // receiver depth-compare bias (live)
extern int   ps_r_vsm_temporal;   // TAA-for-shadows: clipmap jitter + reprojected history accumulate (live)
extern float ps_r_vsm_ta_blend;       // history weight (EMA alpha) for the temporal resolve (live)
extern float ps_r_vsm_ta_blend_dyn;   // history weight on dyn-atlas-shadowed pixels (anti-"jelly" for wind/NPC shadows, live)
extern int   ps_r_vsm_grass;      // cast near grass into the atlas (L0 only, GPU-driven, 1-frame stale) (live)
extern float ps_r_vsm_grass_dist; // max grass cast distance from camera, m (live)
extern int   ps_r_vsm_cache;      // Phase 1b: toroidal per-page cache (1) vs render-all baseline (0) (live)
extern int   ps_r_vsm_cache_refresh; // round-robin refresh period (frames) for the moving sun; smaller = fresher/costlier (live)
extern float ps_r_vsm_lod_dist;      // caster-LOD: distance (m) beyond which opaque casters draw their coarse slice (0 = off) (live)
extern int   ps_r_vsm_mark_half;     // page-mark at half-res (1) = 4x fewer threads/atomics, vs full-res (0) (live)
extern int   ps_r_vsm_dyn_gate;      // resolve skips dyn-atlas taps on pages with no dynamic casters (dynUsed flags); 0 = sample dyn on every resident page (live)
extern int   ps_r_vsm_debug_dyn;     // write dyn-atlas occlusion to mask B; tonemap tints it red (NPC/grass shadow visualizer) (live)

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
CVulkanBuffer* s_priorValid = nullptr;  // device, kMaxPhysS u32 (1 = slot's cached depth is this world tile's)

// Clear-dirty graphics pipeline (depth-only: one quad per dirty slot -> depth 1.0).
VkPipeline            s_clearPipe   = VK_NULL_HANDLE;
VkPipelineLayout      s_clearLayout = VK_NULL_HANDLE;
VkDescriptorSetLayout s_clearSetL   = VK_NULL_HANDLE;
VkDescriptorPool      s_clearPool   = VK_NULL_HANDLE;
VkDescriptorSet       s_clearSet    = VK_NULL_HANDLE;
VkShaderModule        s_clearVS     = VK_NULL_HANDLE;

Fvector s_prevSunDir = { 0.f, -1.f, 0.f };   // last frame's sun dir (round-robin enable)
bool    s_sunMoving   = false;                // sun rotated since last frame (drives round-robin refresh)

// std430 push for vsm_resid.comp: per-level window page-base packed as ivec4[3] +
// up to 4 invalidation circles (light-space xy, radius; [0].w = L0 page width in m) —
// fed by tree near/far wind-hybrid transitions. 64 + 64 = 128 B (the push limit).
struct ResidPush { s32 pageBase[12]; u32 frame, refreshN, sunMoving, forceDirty; float inval[16]; };

Fmatrix s_sunView;   // this frame's world->light view (BeginFrame -> transition spheres to light XY)

// Caster binning: per-page caster lists (the per-page render's draw input).
VkPipeline            s_binPipe   = VK_NULL_HANDLE;
VkPipelineLayout      s_binLayout = VK_NULL_HANDLE;
VkDescriptorSetLayout s_binSetL   = VK_NULL_HANDLE;
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
u32 s_curSlot  = 0;   // frame-in-flight slot used by the latest MarkPages (RenderAtlas reuses it)

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
CVulkanBuffer* s_grassSlot    = nullptr;   // device, GPU_OUTPUT_CAPACITY u32 (global instance idx -> slot/UNMAPPED)
CVulkanBuffer* s_grassStats   = nullptr;   // device [0]=casting instances
CVulkanBuffer* s_grassStatsRB = nullptr;   u32* s_grassStatsPtr = nullptr;
u32 s_grassSection = 0, s_grassTypes = 0;  // captured by the bin for the same-frame render

struct GrassBinPush { u32 sectionSize, typeCount, pad0, pad1; float camRange[4]; };   // xyz cam, w dist

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
};

// std140 — matches the Resolve UBO in vsm_resolve.comp.glsl (192 B).
struct ResolveParams {
    Fmatrix invViewProj;       // current clip -> world
    Fmatrix prevViewProj;      // world -> previous-frame clip (history reproject)
    float   prevCamPos[4];     // xyz = previous frame camera
    float   curCamPos[4];      // xyz = this frame camera (stored as G for next frame); w = dyn-pixel EMA alpha (r_vsm_ta_blend_dyn)
    float   screen[4];         // xy = dims, zw = 1/dims
    float   params[4];         // x = alpha, y = reject tol, z = historyValid, w = unused
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

    VkDescriptorSetLayoutBinding b[4]{};
    b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b[1].binding = 1; b[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    b[2].binding = 2; b[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[3].binding = 3; b[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    for (u32 i = 0; i < 4; ++i) { b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = 4; lci.pBindings = b;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &s_setL) != VK_SUCCESS) return false;

    VkDescriptorPoolSize ps[3] = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, N },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         N },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         N * 2 },
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

// Toroidal residency compute: 7 SSBOs (needed, pageTable, pageList, physTile, slotDirty,
// dirtyList, drawClear). Static set (all device buffers fixed) — written once in Init.
bool CreateResidPipeline()
{
    VkShaderModule cs = g_ShaderManager->Load("vsm_resid.comp.spv");
    if (!cs) { Msg("![VK VSM] vsm_resid.comp.spv load failed"); return false; }
    VkDescriptorSetLayoutBinding b[8]{};   // +7 priorValid (shadow-HZB)
    for (u32 i = 0; i < 8; ++i) { b[i].binding = i; b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = 8; lci.pBindings = b;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &s_residSetL) != VK_SUCCESS) return false;
    VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8 };
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

// shadow-HZB reduce pipeline (r_vsm_hzb): set = {atlas sampler, slotDirty, priorValid, pageMax}.
// Call AFTER CreateRenderResources (needs s_atlasView/s_atlasSampler) and after the resid buffers.
// ⛔ NO NET PERF GAIN on dGPU (2026-07-02, clean stationary A/B: HZB on ≈ off, net −0.15ms — the reduce
//    costs more than the ~6% hidden near-tree pages it culls). KEPT but DISABLED (r_vsm_hzb=0); the reduce
//    dispatch is gated on the cvar so this is inert by default. Don't re-chase — see memory + cvar block.
bool CreateHzbReducePipeline()
{
    VkShaderModule cs = g_ShaderManager->Load("vsm_hzb_reduce.comp.spv");
    if (!cs) { Msg("![VK VSM] vsm_hzb_reduce.comp.spv load failed - shadow-HZB disabled"); return false; }
    VkDescriptorSetLayoutBinding b[4]{};
    b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; b[0].descriptorCount = 1; b[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    for (u32 i = 1; i < 4; ++i) { b[i].binding = i; b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = 4; lci.pBindings = b;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &s_hzbSetL) != VK_SUCCESS) return false;
    VkDescriptorPoolSize ps[2] = { { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 }, { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 } };
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
    VkDescriptorBufferInfo bufi[3] = {
        { s_slotDirty->GetHandle(),  0, VK_WHOLE_SIZE },
        { s_priorValid->GetHandle(), 0, VK_WHOLE_SIZE },
        { s_pageMax->GetHandle(),    0, VK_WHOLE_SIZE },
    };
    VkWriteDescriptorSet w[4]{};
    w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[0].dstSet = s_hzbSet; w[0].dstBinding = 0; w[0].descriptorCount = 1; w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[0].pImageInfo = &ii;
    for (u32 i = 0; i < 3; ++i) { w[i + 1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i + 1].dstSet = s_hzbSet; w[i + 1].dstBinding = i + 1; w[i + 1].descriptorCount = 1; w[i + 1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i + 1].pBufferInfo = &bufi[i]; }
    vkUpdateDescriptorSets(VulkanHW.m_Device, 4, w, 0, nullptr);
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
    VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dynState.dynamicStateCount = 2; dynState.pDynamicStates = dyn;
    VkPipelineRenderingCreateInfo prci{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    prci.colorAttachmentCount = 0; prci.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;
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
    VkImageCreateInfo ici{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format    = VK_FORMAT_D32_SFLOAT;
    ici.extent    = { kAtlasW_S * kPageSize, kAtlasH_S * kPageSize, 1 };   // STATIC: 8192 x 12288 (6144 toroidal pages)
    ici.mipLevels = 1; ici.arrayLayers = 1; ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling    = VK_IMAGE_TILING_OPTIMAL;
    ici.usage     = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo aci{}; aci.usage = VMA_MEMORY_USAGE_AUTO;
    if (vmaCreateImage(VulkanHW.m_Allocator, &ici, &aci, &s_atlasImage, &s_atlasAlloc, nullptr) != VK_SUCCESS) {
        Msg("![VK VSM] atlas image create failed"); return false;
    }
    Prof::NameImage(s_atlasImage, "VSM.Atlas");
    VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vci.image = s_atlasImage; vci.viewType = VK_IMAGE_VIEW_TYPE_2D; vci.format = VK_FORMAT_D32_SFLOAT;
    vci.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
    if (vkCreateImageView(VulkanHW.m_Device, &vci, nullptr, &s_atlasView) != VK_SUCCESS) return false;

    // DYNAMIC atlas — same format/usage, the smaller 2048-page grid (8192 x 4096).
    ici.extent = { kAtlasW * kPageSize, kAtlasH * kPageSize, 1 };
    if (vmaCreateImage(VulkanHW.m_Allocator, &ici, &aci, &s_dynImage, &s_dynAlloc, nullptr) != VK_SUCCESS) {
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
    prci.colorAttachmentCount = 0; prci.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;
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
        if (s_maskImage[i]) { vmaDestroyImage(VulkanHW.m_Allocator, s_maskImage[i], s_maskAlloc[i]); s_maskImage[i] = VK_NULL_HANDLE; s_maskAlloc[i] = VK_NULL_HANDLE; }
        s_maskFirst[i] = true;
    }
    s_maskExtent = {}; s_maskValid = false; s_resolveCount = 0;
}

// Create (or resize) the N-deep screen-space mask. RGBA16F = the only mandatory
// storage format wide enough for (lit, dist) without shaderStorageImageExtendedFormats.
bool EnsureMaskTargets(VkExtent2D screen)
{
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
        if (vmaCreateImage(VulkanHW.m_Allocator, &ici, &aci, &s_maskImage[i], &s_maskAlloc[i], nullptr) != VK_SUCCESS) {
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
    prci.colorAttachmentCount = 0; prci.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;
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

// Grass-caster bin pipeline: 7 bindings (VisibleSSBO, detail indirect, clipmap UBO,
// pageTable, grassSlot, stats, dynUsed). Per-frame sets (the clipmap UBO varies).
bool CreateGrassBinPipeline()
{
    VkShaderModule cs = g_ShaderManager->Load("vsm_grass_bin.comp.spv");
    if (!cs) { Msg("![VK VSM] vsm_grass_bin.comp.spv load failed"); return false; }
    VkDescriptorSetLayoutBinding b[7]{};
    const VkDescriptorType t[7] = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    };
    for (u32 i = 0; i < 7; ++i) { b[i].binding = i; b[i].descriptorType = t[i]; b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = 7; lci.pBindings = b;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &s_grassBinSetL) != VK_SUCCESS) return false;
    VkDescriptorPoolSize ps[2] = { { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, N * 6 }, { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, N } };
    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets = N; pci.poolSizeCount = 2; pci.pPoolSizes = ps;
    if (vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &s_grassBinPool) != VK_SUCCESS) return false;
    VkDescriptorSetLayout ls[N]; for (u32 i = 0; i < N; ++i) ls[i] = s_grassBinSetL;
    VkDescriptorSetAllocateInfo dai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dai.descriptorPool = s_grassBinPool; dai.descriptorSetCount = N; dai.pSetLayouts = ls;
    if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, s_grassBinSet) != VK_SUCCESS) return false;
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

// Grass-page set layout (grassSlot, pageList, clipmap UBO) + pipeline layout + the VS.
// The graphics pipeline itself is lazy (needs the grass mesh vertex stride).
bool CreateGrassPageResources()
{
    VkDescriptorSetLayoutBinding b[3]{};
    const VkDescriptorType t[3] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER };
    for (u32 i = 0; i < 3; ++i) { b[i].binding = i; b[i].descriptorType = t[i]; b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_VERTEX_BIT; }
    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = 3; lci.pBindings = b;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &s_grassPageSetL) != VK_SUCCESS) return false;
    VkDescriptorPoolSize ps[2] = { { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, N * 2 }, { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, N } };
    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets = N; pci.poolSizeCount = 2; pci.pPoolSizes = ps;
    if (vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &s_grassPagePool) != VK_SUCCESS) return false;
    VkDescriptorSetLayout ls[N]; for (u32 i = 0; i < N; ++i) ls[i] = s_grassPageSetL;
    VkDescriptorSetAllocateInfo dai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dai.descriptorPool = s_grassPagePool; dai.descriptorSetCount = N; dai.pSetLayouts = ls;
    if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, s_grassPageSet) != VK_SUCCESS) return false;
    s_grassPageVS = g_ShaderManager->Load("vsm_grass_page.vert.spv");
    s_grassPageFS = g_ShaderManager->Load("vsm_grass_page.frag.spv");   // alpha test (blade cutout, not solid quad)
    if (!s_grassPageVS || !s_grassPageFS) Msg("![VK VSM] vsm_grass_page.{vert,frag}.spv missing - grass VSM shadows disabled");
    return true;
}

// Lazy: the pipeline LAYOUT needs the detail manager's diffuse set layout (set 1, for the
// alpha test) which exists once the level + grass gfx pipeline are up. Pipeline needs the stride.
VkPipeline EnsureGrassPagePipeline(u32 vstride)
{
    if (s_grassPagePipe != VK_NULL_HANDLE) return s_grassPagePipe;
    if (s_grassPageVS == VK_NULL_HANDLE || s_grassPageFS == VK_NULL_HANDLE || vstride == 0) return VK_NULL_HANDLE;
    if (s_grassPageLayout == VK_NULL_HANDLE) {
        CDetailManager* dm = RImplementation.Details;
        VkDescriptorSetLayout diffuseL = dm ? dm->Vsm_GfxSetLayout() : VK_NULL_HANDLE;
        if (diffuseL == VK_NULL_HANDLE) return VK_NULL_HANDLE;
        VkDescriptorSetLayout sets[2] = { s_grassPageSetL, diffuseL };   // set0 = page data (VS), set1 = diffuse (FS)
        VkPushConstantRange pcr{ VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(u32) };   // instanceBase
        VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        plci.setLayoutCount = 2; plci.pSetLayouts = sets; plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
        if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_grassPageLayout) != VK_SUCCESS) return VK_NULL_HANDLE;
    }
    VkVertexInputBindingDescription vibd[2] = {
        { 0, vstride, VK_VERTEX_INPUT_RATE_VERTEX },
        { 1, (u32)sizeof(DetailInstance), VK_VERTEX_INPUT_RATE_INSTANCE },
    };
    VkVertexInputAttributeDescription via[5] = {
        { 0, 0, VK_FORMAT_R32G32B32_SFLOAT,    0  },// aPos (binding 0)
        { 1, 0, VK_FORMAT_R32G32_SFLOAT,       12 },// aUV  (binding 0) — alpha test
        { 3, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0  },// aInstRow0 (binding 1)
        { 4, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 16 },// aInstRow1
        { 5, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 32 },// aInstRow2
    };
    VkPipelineVertexInputStateCreateInfo vi{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    vi.vertexBindingDescriptionCount = 2; vi.pVertexBindingDescriptions = vibd;
    vi.vertexAttributeDescriptionCount = 5; vi.pVertexAttributeDescriptions = via;
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;   stages[0].module = s_grassPageVS; stages[0].pName = "main";
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
    prci.colorAttachmentCount = 0; prci.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;
    VkGraphicsPipelineCreateInfo pi{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    pi.pNext = &prci; pi.stageCount = 2; pi.pStages = stages;
    pi.pVertexInputState = &vi; pi.pInputAssemblyState = &ia; pi.pViewportState = &vp;
    pi.pRasterizationState = &rs; pi.pMultisampleState = &ms; pi.pDepthStencilState = &ds;
    pi.pColorBlendState = &cb; pi.pDynamicState = &dynState; pi.layout = s_grassPageLayout;
    if (vkCreateGraphicsPipelines(VulkanHW.m_Device, VK_NULL_HANDLE, 1, &pi, nullptr, &s_grassPagePipe) != VK_SUCCESS)
        Msg("![VK VSM] grass page pipeline create failed (stride=%u)", vstride);
    return s_grassPagePipe;
}

void DispatchGrassBin(VkCommandBuffer cmd, u32 cur)
{
    s_grassSection = 0; s_grassTypes = 0;
    if (!ps_r_vsm_grass || s_grassBinPipe == VK_NULL_HANDLE) return;
    CDetailManager* dm = RImplementation.Details;
    if (!dm) return;
    VkBuffer vis = dm->Vsm_VisibleSSBO();
    VkBuffer ind = dm->Vsm_IndirectBuf();
    const u32 types = dm->Vsm_TypeCount();
    const u32 section = dm->Vsm_SectionSize();
    if (vis == VK_NULL_HANDLE || ind == VK_NULL_HANDLE || types == 0 || section == 0) return;
    s_grassSection = section; s_grassTypes = types;

    vkCmdFillBuffer(cmd, s_grassStats->GetHandle(), 0, VK_WHOLE_SIZE, 0u);
    // Last frame's grass gen (compute) + indirect copy (transfer) + this stats clear → bin reads.
    MemBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    VkDescriptorBufferInfo bi[7] = {
        { vis,                       0, VK_WHOLE_SIZE }, { ind,                     0, VK_WHOLE_SIZE },
        { s_ubo[cur]->GetHandle(),   0, VK_WHOLE_SIZE }, { s_dynPageTable->GetHandle(),0, VK_WHOLE_SIZE },  // DYNAMIC table
        { s_grassSlot->GetHandle(),  0, VK_WHOLE_SIZE }, { s_grassStats->GetHandle(),0, VK_WHOLE_SIZE },
        { s_dynPageUsed->GetHandle(),0, VK_WHOLE_SIZE },   // dyn slot -> has-caster flag (resolve skip)
    };
    const VkDescriptorType t[7] = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    };
    VkWriteDescriptorSet w[7]{};
    for (u32 i = 0; i < 7; ++i) { w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = s_grassBinSet[cur]; w[i].dstBinding = i; w[i].descriptorCount = 1; w[i].descriptorType = t[i]; w[i].pBufferInfo = &bi[i]; }
    vkUpdateDescriptorSets(VulkanHW.m_Device, 7, w, 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_grassBinPipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_grassBinLayout, 0, 1, &s_grassBinSet[cur], 0, nullptr);
    GrassBinPush gp{}; gp.sectionSize = section; gp.typeCount = types;
    gp.camRange[0] = Device.vCameraPosition.x; gp.camRange[1] = Device.vCameraPosition.y; gp.camRange[2] = Device.vCameraPosition.z;
    gp.camRange[3] = ps_r_vsm_grass_dist;
    vkCmdPushConstants(cmd, s_grassBinLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(gp), &gp);
    vkCmdDispatch(cmd, (section * types + 63) / 64, 1, 1);
}

void RenderGrassCasters(VkCommandBuffer cmd, u32 cur)
{
    if (!ps_r_vsm_grass || s_grassPageVS == VK_NULL_HANDLE || s_grassTypes == 0 || s_grassSection == 0) return;
    CDetailManager* dm = RImplementation.Details;
    if (!dm) return;
    VkBuffer vis = dm->Vsm_VisibleSSBO();
    VkBuffer ind = dm->Vsm_IndirectBuf();
    if (vis == VK_NULL_HANDLE || ind == VK_NULL_HANDLE) return;
    VkPipeline pipe = EnsureGrassPagePipeline(dm->Vsm_VertexStride());
    if (pipe == VK_NULL_HANDLE) return;

    VkDescriptorBufferInfo bi[3] = {
        { s_grassSlot->GetHandle(), 0, VK_WHOLE_SIZE },
        { s_dynPageList->GetHandle(), 0, VK_WHOLE_SIZE },   // DYNAMIC list (grass atlas)
        { s_ubo[cur]->GetHandle(),  0, VK_WHOLE_SIZE },
    };
    const VkDescriptorType t[3] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER };
    VkWriteDescriptorSet w[3]{};
    for (u32 i = 0; i < 3; ++i) { w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = s_grassPageSet[cur]; w[i].dstBinding = i; w[i].descriptorCount = 1; w[i].descriptorType = t[i]; w[i].pBufferInfo = &bi[i]; }
    vkUpdateDescriptorSets(VulkanHW.m_Device, 3, w, 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_grassPageLayout, 0, 1, &s_grassPageSet[cur], 0, nullptr);
    for (u32 i = 0; i < s_grassTypes; ++i) {
        VkBuffer mvb, mib; u32 ic;
        if (!dm->Vsm_TypeMesh(i, mvb, mib, ic)) continue;
        VkDescriptorSet diffuse = dm->Vsm_TypeDiffuseSet(i);   // set 1 = grass diffuse (alpha test)
        if (diffuse == VK_NULL_HANDLE) continue;
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_grassPageLayout, 1, 1, &diffuse, 0, nullptr);
        VkBuffer vbs[2] = { mvb, vis };
        VkDeviceSize off[2] = { 0, (VkDeviceSize)i * s_grassSection * sizeof(DetailInstance) };
        vkCmdBindVertexBuffers(cmd, 0, 2, vbs, off);
        vkCmdBindIndexBuffer(cmd, mib, 0, VK_INDEX_TYPE_UINT16);
        const u32 base = i * s_grassSection;
        vkCmdPushConstants(cmd, s_grassPageLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(u32), &base);
        vkCmdDrawIndexedIndirect(cmd, ind, (VkDeviceSize)i * sizeof(VkDrawIndexedIndirectCommand), 1, sizeof(VkDrawIndexedIndirectCommand));
    }
}

} // anonymous namespace

bool Init()
{
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
                     VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    s_counter = xr_new<CVulkanBuffer>();
    s_counter->Create(sizeof(u32),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                      VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    s_readback = xr_new<CVulkanBuffer>();
    s_readback->Create(8 * sizeof(u32), VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
    s_readPtr = (u32*)s_readback->Map();
    if (s_readPtr) memset(s_readPtr, 0, 8 * sizeof(u32));

    // Allocation buffers + pipeline.
    s_pageTable = xr_new<CVulkanBuffer>();
    s_pageTable->Create((VkDeviceSize)kPageCount * sizeof(u32),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    s_pageList = xr_new<CVulkanBuffer>();
    s_pageList->Create((VkDeviceSize)kMaxPhysS * 4 * sizeof(u32),   // uvec4 per STATIC toroidal slot (6144)
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    // Dynamic atlas's own demand-alloc table/list/info (rebuilt every frame).
    s_dynPageTable = xr_new<CVulkanBuffer>();
    s_dynPageTable->Create((VkDeviceSize)kPageCount * sizeof(u32),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                           VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    s_dynPageList = xr_new<CVulkanBuffer>();
    s_dynPageList->Create((VkDeviceSize)kMaxPhys * 4 * sizeof(u32),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    s_dynAllocInfo = xr_new<CVulkanBuffer>();
    s_dynAllocInfo->Create((VkDeviceSize)(1 + kLevels) * sizeof(u32),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                           VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    s_dynPageUsed = xr_new<CVulkanBuffer>();
    s_dynPageUsed->Create((VkDeviceSize)kMaxPhys * sizeof(u32),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

    if (!CreateAllocPipeline()) { Msg("![VK VSM] alloc pipeline failed - VSM disabled"); s_dead = true; return false; }

    // Toroidal STATIC residency buffers + pipelines (Phase 1b).
    s_physTile = xr_new<CVulkanBuffer>();
    s_physTile->Create((VkDeviceSize)kMaxPhysS * 2 * sizeof(u32),   // uvec2 per slot, persistent
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    s_slotDirty = xr_new<CVulkanBuffer>();
    s_slotDirty->Create((VkDeviceSize)kMaxPhysS * sizeof(u32),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    s_dirtyList = xr_new<CVulkanBuffer>();
    s_dirtyList->Create((VkDeviceSize)kMaxPhysS * sizeof(u32),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    // shadow-HZB (r_vsm_hzb): per-slot occluder max + validity (written by residency).
    s_priorValid = xr_new<CVulkanBuffer>();
    s_priorValid->Create((VkDeviceSize)kMaxPhysS * sizeof(u32),
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    s_pageMax = xr_new<CVulkanBuffer>();
    s_pageMax->Create((VkDeviceSize)kMaxPhysS * sizeof(float),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    s_drawClear = xr_new<CVulkanBuffer>();
    s_drawClear->Create(4 * sizeof(u32),   // VkDrawIndirectCommand
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    s_residRB = xr_new<CVulkanBuffer>();
    s_residRB->Create(4 * sizeof(u32), VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
    s_residPtr = (u32*)s_residRB->Map();
    if (s_residPtr) memset(s_residPtr, 0, 4 * sizeof(u32));
    if (!CreateResidPipeline()) { Msg("![VK VSM] resid pipeline failed - VSM disabled"); s_dead = true; return false; }
    if (!CreateClearPipeline()) { Msg("![VK VSM] clear pipeline failed - VSM disabled"); s_dead = true; return false; }
    {
        VkDescriptorBufferInfo bi[8] = {
            { s_needed->GetHandle(),    0, VK_WHOLE_SIZE }, { s_pageTable->GetHandle(), 0, VK_WHOLE_SIZE },
            { s_pageList->GetHandle(),  0, VK_WHOLE_SIZE }, { s_physTile->GetHandle(),  0, VK_WHOLE_SIZE },
            { s_slotDirty->GetHandle(), 0, VK_WHOLE_SIZE }, { s_dirtyList->GetHandle(), 0, VK_WHOLE_SIZE },
            { s_drawClear->GetHandle(), 0, VK_WHOLE_SIZE }, { s_priorValid->GetHandle(), 0, VK_WHOLE_SIZE },
        };
        VkWriteDescriptorSet w[8]{};
        for (u32 i = 0; i < 8; ++i) { w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = s_residSet; w[i].dstBinding = i; w[i].descriptorCount = 1; w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pBufferInfo = &bi[i]; }
        VkDescriptorBufferInfo cb{ s_dirtyList->GetHandle(), 0, VK_WHOLE_SIZE };
        VkWriteDescriptorSet cw{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        cw.dstSet = s_clearSet; cw.dstBinding = 0; cw.descriptorCount = 1; cw.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; cw.pBufferInfo = &cb;
        vkUpdateDescriptorSets(VulkanHW.m_Device, 8, w, 0, nullptr);
        vkUpdateDescriptorSets(VulkanHW.m_Device, 1, &cw, 0, nullptr);
    }

    // Binning + draw-build buffers + pipeline.
    s_casterPages = xr_new<CVulkanBuffer>();
    s_casterPages->Create((VkDeviceSize)kMaxCasters * kPagesCap * sizeof(u32),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    s_vsmIndirect = xr_new<CVulkanBuffer>();
    s_vsmIndirect->Create((VkDeviceSize)kMaxGroups * kGroupStride * sizeof(VkDrawIndexedIndirectCommand),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                          VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    s_vsmGroupCount = xr_new<CVulkanBuffer>();
    s_vsmGroupCount->Create((VkDeviceSize)kMaxGroups * sizeof(u32),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    s_binStats = xr_new<CVulkanBuffer>();
    s_binStats->Create(4 * sizeof(u32),
                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    s_binReadback = xr_new<CVulkanBuffer>();
    s_binReadback->Create(4 * sizeof(u32), VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
    s_binReadPtr = (u32*)s_binReadback->Map();
    if (s_binReadPtr) memset(s_binReadPtr, 0, 4 * sizeof(u32));

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
    s_skinCasterPages->Create((VkDeviceSize)kMaxSkinned * kSkinnedCap * sizeof(u32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    s_skinIndirect = xr_new<CVulkanBuffer>();
    s_skinIndirect->Create((VkDeviceSize)kMaxSkinned * sizeof(VkDrawIndexedIndirectCommand),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    s_skinStats = xr_new<CVulkanBuffer>();
    s_skinStats->Create(4 * sizeof(u32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    s_skinStatsRB = xr_new<CVulkanBuffer>();
    s_skinStatsRB->Create(4 * sizeof(u32), VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
    s_skinStatsPtr = (u32*)s_skinStatsRB->Map();
    if (s_skinStatsPtr) memset(s_skinStatsPtr, 0, 4 * sizeof(u32));
    if (!CreateSkinnedBinPipeline()) { Msg("![VK VSM] skinned bin pipeline failed - NPC VSM shadows disabled"); }
    s_skinPageVS = g_ShaderManager->Load("vsm_skinned_page.vert.spv");
    if (!s_skinPageVS) Msg("![VK VSM] vsm_skinned_page.vert.spv missing - NPC VSM shadows disabled");

    // Grass (detail) casters: a slot buffer (one per instance capacity) + bin/page pipelines.
    s_grassSlot = xr_new<CVulkanBuffer>();
    s_grassSlot->Create((VkDeviceSize)GPU_OUTPUT_CAPACITY * sizeof(u32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    s_grassStats = xr_new<CVulkanBuffer>();
    s_grassStats->Create(4 * sizeof(u32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    s_grassStatsRB = xr_new<CVulkanBuffer>();
    s_grassStatsRB->Create(4 * sizeof(u32), VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
    s_grassStatsPtr = (u32*)s_grassStatsRB->Map();
    if (s_grassStatsPtr) memset(s_grassStatsPtr, 0, 4 * sizeof(u32));
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

void BeginFrame(const Fvector& camPos, VkExtent2D screen)
{
    if (!s_inited) Init();
    if (!Enabled()) return;

    EnsureMaskTargets(screen);   // screen-space mask (create/resize) — before EnvLight binds it
    s_curCamPos = camPos;

    const u32 cur = s_frame % N;
    s_frame++;
    s_curSlot = cur;   // MarkPages / RenderAtlas / receivers all reuse this slot's UBO

    // ---- Sun direction (same env source as Pass_SunShadow).
    Fvector sunDir; sunDir.set(0.f, -1.f, 0.f);
    if (g_pGamePersistent)
        if (auto* E = g_pGamePersistent->Environment().CurrentEnv) sunDir = E->sun_dir;

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
    params.zparams[2] = ps_r_vsm_bias;   // receiver bias (live, r_vsm_bias)
    params.zparams[3] = 0.f;

    if (s_uboPtr[cur]) memcpy(s_uboPtr[cur], &params, sizeof(params));
}

VkBuffer    GetPageTableHandle() { return s_pageTable ? s_pageTable->GetHandle() : VK_NULL_HANDLE; }
VkBuffer    GetUBOHandle()       { return s_ubo[s_curSlot] ? s_ubo[s_curSlot]->GetHandle() : VK_NULL_HANDLE; }

void MarkPages(VkCommandBuffer cmd, VkImageView sceneDepth, VkExtent2D screen, const Fmatrix& viewProj)
{
    if (!Enabled() || sceneDepth == VK_NULL_HANDLE || screen.width == 0 || screen.height == 0) return;
    const u32 cur = s_curSlot;

    // Collect this frame's NPC leaves (CPU) + upload their bin meta to this slot.
    CollectSkinned(cur);

    // ---- Descriptor set for this frame.
    VkDescriptorImageInfo di{ s_depthSampler, sceneDepth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkDescriptorBufferInfo ui{ s_ubo[cur]->GetHandle(), 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo ni{ s_needed->GetHandle(),   0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo ci{ s_counter->GetHandle(),  0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet w[4]{};
    for (u32 i = 0; i < 4; ++i) { w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = s_set[cur]; w[i].dstBinding = i; w[i].descriptorCount = 1; }
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[0].pImageInfo  = &di;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;         w[1].pBufferInfo = &ui;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;         w[2].pBufferInfo = &ni;
    w[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;         w[3].pBufferInfo = &ci;
    vkUpdateDescriptorSets(VulkanHW.m_Device, 4, w, 0, nullptr);

    // ---- Clear per-frame flags/counters, then MARK, then RESIDENCY (static) + ALLOC (dynamic).
    // physTile is PERSISTENT (the toroidal cache) — filled to EMPTY only once after create.
    // WAR guard: the PREVIOUS frame's resolve (compute) reads several of these buffers
    // (dynPageTable/dynUsed) and nothing else orders its reads against this frame's fills
    // on the same queue — execution-order the transfers after prior shader reads.
    MemBarrier(cmd, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    vkCmdFillBuffer(cmd, s_needed->GetHandle(),          0, VK_WHOLE_SIZE, 0u);
    vkCmdFillBuffer(cmd, s_counter->GetHandle(),         0, VK_WHOLE_SIZE, 0u);
    vkCmdFillBuffer(cmd, s_dynPageTable->GetHandle(),    0, VK_WHOLE_SIZE, 0xFFFFFFFFu);  // UNMAPPED
    vkCmdFillBuffer(cmd, s_dynAllocInfo->GetHandle(),    0, VK_WHOLE_SIZE, 0u);
    vkCmdFillBuffer(cmd, s_vsmGroupCount->GetHandle(),   0, VK_WHOLE_SIZE, 0u);
    vkCmdFillBuffer(cmd, s_binStats->GetHandle(),        0, VK_WHOLE_SIZE, 0u);
    vkCmdFillBuffer(cmd, s_skinStats->GetHandle(),       0, VK_WHOLE_SIZE, 0u);
    vkCmdFillBuffer(cmd, s_dynPageUsed->GetHandle(),     0, VK_WHOLE_SIZE, 0u);            // dyn has-caster flags reset
    vkCmdFillBuffer(cmd, s_slotDirty->GetHandle(),       0, VK_WHOLE_SIZE, 0u);            // dirty set reset
    if (ps_r_vsm_hzb) vkCmdFillBuffer(cmd, s_priorValid->GetHandle(), 0, VK_WHOLE_SIZE, 0u);   // shadow-HZB: priorValid==1 means resident+valid THIS frame
    vkCmdFillBuffer(cmd, s_drawClear->GetHandle(),       0, VK_WHOLE_SIZE, 0u);            // clear draw: instanceCount=0,...
    vkCmdFillBuffer(cmd, s_drawClear->GetHandle(),       0, sizeof(u32),  6u);            // ...vertexCount=6 (the quad)
    if (!s_physInit) { vkCmdFillBuffer(cmd, s_physTile->GetHandle(), 0, VK_WHOLE_SIZE, 0xFFFFFFFFu); s_physInit = true; }   // toroidal cache starts empty
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
    vkCmdPushConstants(cmd, s_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    const u32 mw = (screen.width  + markStep - 1) / markStep;
    const u32 mh = (screen.height + markStep - 1) / markStep;
    vkCmdDispatch(cmd, (mw + 7) / 8, (mh + 7) / 8, 1);

    // needed[] write (mark) → read (alloc)
    MemBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

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
    if (RImplementation.Trees && RImplementation.Trees->IsBuilt()) {
        RImplementation.Trees->VsmUpdateNearSet(s_sunView);
        Fvector4 sph[4]; const u32 nInv = RImplementation.Trees->VsmPopTransitions(sph, 4);
        for (u32 i = 0; i < nInv; ++i) {
            Fvector l; s_sunView.transform_tiny(l, Fvector{ sph[i].x, sph[i].y, sph[i].z });
            rp.inval[i * 4 + 0] = l.x; rp.inval[i * 4 + 1] = l.y; rp.inval[i * 4 + 2] = sph[i].w;
        }
    }
    vkCmdPushConstants(cmd, s_residLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(rp), &rp);
    vkCmdDispatch(cmd, (kPageCount + 63) / 64, 1, 1);

    // shadow-HZB (r_vsm_hzb): reduce each dirty+valid static slot's PRIOR-frame depth to a per-slot
    // MAX occluder into s_pageMax. Runs here — the atlas still holds prior-frame depth (SHADER_READ,
    // before RenderAtlas overwrites the dirty pages) and residency just wrote slotDirty/priorValid.
    // The tree/opaque caster bins (below) read s_pageMax to cull fully-behind casters (kills overdraw).
    if (ps_r_vsm_hzb && !s_atlasFirst && s_hzbPipe != VK_NULL_HANDLE) {
        vkCmdFillBuffer(cmd, s_pageMax->GetHandle(), 0, VK_WHOLE_SIZE, 0x3F800000u);   // 1.0f → no occlusion
        MemBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_hzbPipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_hzbLayout, 0, 1, &s_hzbSet, 0, nullptr);
        vkCmdDispatch(cmd, kMaxPhysS, 1, 1);
    }

    // ALLOC (dynamic): demand-allocate a slot per needed page into the dynamic table.
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_allocPipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_allocLayout, 0, 1, &s_dynAllocSet, 0, nullptr);
    vkCmdDispatch(cmd, (kPageCount + 63) / 64, 1, 1);

    // BIN (static): scatter each ShadowGPU caster into the DIRTY static pages it overlaps. Only
    // dirty pages get caster draws (cached pages keep their depth); the residency set the dirty
    // flags. The barrier orders residency's pageTable/slotDirty + the dynamic alloc before the read.
    const VkBuffer metaBuf = VK::ShadowGPU::GetMetaBuffer();
    u32            casterN = VK::ShadowGPU::CasterCount();
    const u32      groupN  = VK::ShadowGPU::GroupCount();
    if (metaBuf != VK_NULL_HANDLE && casterN > 0 && groupN > 0 && groupN <= kMaxGroups) {
        if (casterN > kMaxCasters) casterN = kMaxCasters;   // clamp to casterPages capacity
        MemBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,    // residency/alloc write -> bin read
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
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

    // Skinned (NPC) casters: bin into the allocated pages (own per-leaf draw stream).
    // pageTable is resident (alloc above); this barrier also covers the static-bin-skipped path.
    if (s_skinCount > 0) {
        MemBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        DispatchSkinnedBin(cmd, cur);
    }

    // Grass casters: bin NEAR grass (prev-frame GPU instances) into resident L0 pages.
    // (Own barriers inside; no-op unless r_vsm_grass + a detail manager with grass.)
    DispatchGrassBin(cmd, cur);

    // Tree casters, near/far wind hybrid: FAR trees bin into the toroidal STATIC cache
    // (dirty pages only, rigid); NEAR trees (r_vsm_tree_wind_dist) bin into the DYNAMIC
    // atlas (all resident pages, re-rendered each frame WITH wind → smooth sway) and mark
    // dynUsed for the resolve gate. [Phase 1 + Phase 2 wind]
    if (RImplementation.Trees && RImplementation.Trees->IsBuilt()) {
        const VkBuffer hzbMax = (ps_r_vsm_hzb && s_pageMax) ? s_pageMax->GetHandle() : VK_NULL_HANDLE;
        RImplementation.Trees->VsmBin(cmd, s_pageTable->GetHandle(), s_slotDirty->GetHandle(), s_dynPageUsed->GetHandle(), s_ubo[cur]->GetHandle(), s_pageList->GetHandle(), hzbMax);
        RImplementation.Trees->VsmBinDyn(cmd, s_dynPageTable->GetHandle(), s_dynPageUsed->GetHandle(), s_ubo[cur]->GetHandle(), s_dynPageList->GetHandle(), hzbMax, s_pageTable->GetHandle());
    }

    // ---- Diagnostics (r_vsm_debug only): copy the GPU counters to the host readbacks
    // (read stale next frames, fine — the buffers were zeroed at Init).
    if (ps_r_vsm_debug) {
        MemBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferCopy rc{ 0, 0, sizeof(u32) };                            // mark counter -> readback[0]
        vkCmdCopyBuffer(cmd, s_counter->GetHandle(), s_readback->GetHandle(), 1, &rc);
        VkBufferCopy ra{ 0, sizeof(u32), (VkDeviceSize)(1 + kLevels) * sizeof(u32) };   // dynAllocInfo -> readback[1..7]
        vkCmdCopyBuffer(cmd, s_dynAllocInfo->GetHandle(), s_readback->GetHandle(), 1, &ra);
        VkBufferCopy rdc{ sizeof(u32), 0, sizeof(u32) };                 // drawClear[1] (static dirty count) -> residRB[0]
        vkCmdCopyBuffer(cmd, s_drawClear->GetHandle(), s_residRB->GetHandle(), 1, &rdc);
        VkBufferCopy rb{ 0, 0, 4 * sizeof(u32) };                        // binStats -> binReadback[0..3]
        vkCmdCopyBuffer(cmd, s_binStats->GetHandle(), s_binReadback->GetHandle(), 1, &rb);
        VkBufferCopy rsk{ 0, 0, 4 * sizeof(u32) };                       // skinStats -> skinStatsRB[0..3]
        vkCmdCopyBuffer(cmd, s_skinStats->GetHandle(), s_skinStatsRB->GetHandle(), 1, &rsk);
        VkBufferCopy rgr{ 0, 0, 4 * sizeof(u32) };                       // grassStats -> grassStatsRB[0..3]
        vkCmdCopyBuffer(cmd, s_grassStats->GetHandle(), s_grassStatsRB->GetHandle(), 1, &rgr);
    }

    if (ps_r_vsm_debug && s_readPtr && Device.dwTimeGlobal > s_lastLog + 2000) {
        s_lastLog = Device.dwTimeGlobal;
        const u32 dirty = s_residPtr ? s_residPtr[0] : 0u;
        Msg("[VK VSM] static: dirty=%u/%u rendered | mark=%u | mode=%s sun=%s | dyn demand=%u/%u",
            dirty, kMaxPhysS, s_readPtr[0], (ps_r_vsm_cache ? "cache" : "render-all"),
            (s_sunMoving ? "moving" : "static"), s_readPtr[1], kMaxPhys);
        if (s_binReadPtr)
            Msg("[VK VSM] bin: draws=%u instances=%u maxPagesPerCaster=%u/%u groupOverflow=%u (casters=%u)",
                s_binReadPtr[0], s_binReadPtr[1], s_binReadPtr[2], kPagesCap, s_binReadPtr[3], casterN);
        if (s_skinStatsPtr)
            Msg("[VK VSM] skinned: leaves=%u draws=%u instances=%u maxPages=%u/%u",
                s_skinCount, s_skinStatsPtr[0], s_skinStatsPtr[1], s_skinStatsPtr[2], kSkinnedCap);
        if (s_grassStatsPtr && ps_r_vsm_grass)
            Msg("[VK VSM] grass: casting instances=%u (types=%u, near<=%.0fm, L0)",
                s_grassStatsPtr[0], s_grassTypes, ps_r_vsm_grass_dist);
    }
}

VkImageView GetAtlasView() { return s_atlasView; }
VkSampler   GetSampler()   { return s_atlasSampler; }
bool        AtlasReady()   { return s_atlasView != VK_NULL_HANDLE && s_dynView != VK_NULL_HANDLE && !s_atlasFirst && !s_dynFirst; }

void RenderAtlas(VkCommandBuffer cmd)
{
    if (!Enabled() || s_atlasView == VK_NULL_HANDLE || s_dynView == VK_NULL_HANDLE) return;
    if (VK::ShadowGPU::GetMetaBuffer() == VK_NULL_HANDLE) return;
    const u32 groupN = VK::ShadowGPU::GroupCount();
    if (groupN == 0 || groupN > kMaxGroups) return;
    const u32 cur = s_curSlot;

    // bin's writes (vsmIndirect / casterPages / pageList) + residency (dirtyList / drawClear)
    // → render reads (vertex + indirect).
    MemBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT);

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
        vkCmdSetDepthBias(cmd, 1.5f, 0.f, 2.5f);
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
    // Caster draws (only dirty pages, per the bin filter).
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_renderLayout, 0, 1, &s_renderSet[cur], 0, nullptr);
    VkPipeline lastPipe = VK_NULL_HANDLE;
    for (u32 g = 0; g < groupN; ++g) {
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
    vkCmdEndRendering(cmd);
    atlasToRead(s_atlasImage);

    // ===== DYNAMIC atlas: skinned (NPC) + grass — re-rendered + cleared every frame. =====
    const u32 dw = kAtlasW * kPageSize, dh = kAtlasH * kPageSize;
    ImageBarrier(cmd, s_dynImage, s_dynFirst ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
    s_dynFirst = false;
    beginAtlas(s_dynView, VK_ATTACHMENT_LOAD_OP_CLEAR, dw, dh);
    RenderSkinnedCasters(cmd, cur);   // viewport/scissor/bias already set by beginAtlas
    RenderGrassCasters(cmd, cur);
    // NEAR trees with live wind (r_vsm_tree_wind hybrid) — swaying crown shadows.
    if (RImplementation.Trees && RImplementation.Trees->IsBuilt())
        RImplementation.Trees->VsmRenderDyn(cmd, s_dynPageList->GetHandle(), s_ubo[cur]->GetHandle());
    vkCmdEndRendering(cmd);
    atlasToRead(s_dynImage);
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
    rp.prevCamPos[3] = ps_r_vsm_debug_dyn ? 1.f : 0.f;   // dyn-debug: resolve writes dyn occlusion to mask B
    rp.curCamPos[0]  = s_curCamPos.x;  rp.curCamPos[1]  = s_curCamPos.y;  rp.curCamPos[2]  = s_curCamPos.z;
    rp.curCamPos[3]  = ps_r_vsm_ta_blend_dyn;   // EMA alpha where the dyn atlas shadows (resolve takes min with params.x)
    rp.screen[0] = (float)screen.width; rp.screen[1] = (float)screen.height;
    rp.screen[2] = 1.0f / (float)screen.width; rp.screen[3] = 1.0f / (float)screen.height;
    rp.params[0] = histOK ? ps_r_vsm_ta_blend : 0.f;   // EMA alpha (0 = current only)
    rp.params[1] = kRejectTol;
    rp.params[2] = histOK ? 1.f : 0.f;                 // historyValid
    rp.params[3] = ps_r_vsm_dyn_gate ? 1.f : 0.f;      // dyn-gate (skip caster-less dyn pages)
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
    del(s_needed); del(s_counter); del(s_readback); s_readPtr = nullptr;
    del(s_pageTable); del(s_pageList);
    del(s_dynPageTable); del(s_dynPageList); del(s_dynAllocInfo); del(s_dynPageUsed);
    del(s_physTile); del(s_slotDirty); del(s_dirtyList); del(s_drawClear); del(s_residRB); s_residPtr = nullptr; s_physInit = false;
    del(s_pageMax); del(s_priorValid);   // shadow-HZB
    del(s_casterPages); del(s_vsmIndirect); del(s_vsmGroupCount); del(s_binStats); del(s_binReadback); s_binReadPtr = nullptr;
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
    if (s_renderLayout) { vkDestroyPipelineLayout(VulkanHW.m_Device, s_renderLayout, nullptr); s_renderLayout = VK_NULL_HANDLE; }
    if (s_renderPool)   { vkDestroyDescriptorPool(VulkanHW.m_Device, s_renderPool, nullptr); s_renderPool = VK_NULL_HANDLE; }
    if (s_renderSetL)   { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_renderSetL, nullptr); s_renderSetL = VK_NULL_HANDLE; }
    if (s_atlasView)    { vkDestroyImageView(VulkanHW.m_Device, s_atlasView, nullptr); s_atlasView = VK_NULL_HANDLE; }
    if (s_dynView)      { vkDestroyImageView(VulkanHW.m_Device, s_dynView, nullptr); s_dynView = VK_NULL_HANDLE; }
    if (s_atlasSampler) { vkDestroySampler(VulkanHW.m_Device, s_atlasSampler, nullptr); s_atlasSampler = VK_NULL_HANDLE; }
    if (s_atlasImage)   { vmaDestroyImage(VulkanHW.m_Allocator, s_atlasImage, s_atlasAlloc); s_atlasImage = VK_NULL_HANDLE; s_atlasAlloc = VK_NULL_HANDLE; }
    if (s_dynImage)     { vmaDestroyImage(VulkanHW.m_Allocator, s_dynImage, s_dynAlloc); s_dynImage = VK_NULL_HANDLE; s_dynAlloc = VK_NULL_HANDLE; }
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
    del(s_grassSlot); del(s_grassStats); del(s_grassStatsRB); s_grassStatsPtr = nullptr;
    for (u32 i = 0; i < N; ++i) { s_grassBinSet[i] = VK_NULL_HANDLE; s_grassPageSet[i] = VK_NULL_HANDLE; }
    s_grassSection = 0; s_grassTypes = 0;
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
