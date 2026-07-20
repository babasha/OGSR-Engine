// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).
//
// GPU-driven world forward pass — see vk_world_gpu.h. Cull mirrors vk_shadow_gpu;
// the draw mirrors RenderQueue::Flush / FlushDepth.

#include "stdafx.h"
#include "vk_profiler.h"   // TEMP VUID-hunt: VK::Prof::NameSet
#include "vk_world_gpu.h"
#include "vk_cluster_stream.h"  // Stage B: cluster-IB page streaming (pools + residency bits)
#include "CRender_Vulkan.h"     // RImplementation.Visuals
#include "vk_Visual.h"          // vkFVisual, vkFHierrarhyVisual, m_mesh, m_pWorldMaterial
#include "vk_world_material.h"  // WorldMaterial / WorldMaterialCache
#include "vk_render_queue.h"    // VK::RenderQueue (SubmitCpuMeshes)
#include "vk_terrain_cache.h"   // TerrainCache capture (composite ground cache)
#include "vk_buffer.h"          // CVulkanBuffer
#include "vk_pipeline_cache.h"  // PipelineCache pipelines/layouts
#include "vk_env_light.h"       // EnvLight::GetCurrentSet (set 1, terrain snow-depth)
#include "vk_scene_color.h"     // HDR scene target format (cluster debug overlay)
#include "vk_swapchain.h"       // Swapchain.m_DepthFormat (cluster debug overlay)
#include "vk_shaders.h"         // g_ShaderManager
#include "vk_cull.h"            // VK::ExtractFrustumPlanes
#include "vk_command_buffer.h"  // CVulkanCommandManager::FRAMES_IN_FLIGHT (cmd-audit cadence)
#include "../../xr_3da/device.h" // Device.dwTimeGlobal (occlusion-stats throttle)
#include "../../../3rd_party/Src/meshoptimizer/src/meshoptimizer.h"  // cluster split (vk_meshopt.cpp)
#include "../../../3rd_party/Src/meshoptimizer/demo/clusterlod.h"    // clodBuild — Nanite-style DAG (impl in vk_meshopt.cpp)
#include <algorithm>
#include <unordered_map>
#include <atomic>
#include <thread>

extern int ps_r_cluster;        // cluster-granularity cull (Phase 1 of cluster-LOD) — read at Build
extern int ps_r_cluster_tris;   // min triangles before a mesh is split into clusters
extern int ps_r_cluster_merge;  // Phase 2.5: merge touching solid fragments into one component DAG — read at Build
extern float ps_r_cluster_lod;  // DAG-cut px error threshold (Phase 2) — live, read per frame
extern float ps_r_vsm_cluster_lod;   // Phase 3 shadow LOD quality k (error budget in target texels) — live
extern float ps_r_cluster_fade; // crossfade band as a fraction of the threshold (0 = hard cut) — live
extern int ps_r_cluster_debug;  // 1 = cluster fill colors, 2 = wireframe (live)
extern int ps_r_profiler;       // 2 = verbose: enables the throttled occlusion/cpu-cut stats log
extern int ps_r_cl_audit;       // draw-command audit (white-polygon forensics) — read at level load
extern float ps_r_tess;         // global tessellation toggle — decides tessellated-material routing at Build
extern int ps_r_pool_compact;   // Stage B (б): free cluster-repacked slices from the level VB/IB pools — read at load
extern int ps_r_gpu_shadows_at; // AT casters through the cluster shadow cull/draw (live A/B; cull+draw read it in the same record)

namespace VK { namespace WorldGPU {

namespace {

// GpuMeshMeta (one cullable ENTRY — a whole mesh or one LOD-DAG cluster) and
// kErrInf moved to vk_world_gpu.h: vk_cluster_stream shares the layout for the
// page assembly / residency tables. The cull shader picks the DAG cut per
// frame: draw iff projected selfError <= threshold < projected parentError;
// siblings share the exact values their parents test as self → the whole group
// flips atomically (crack-free). Plain meshes: selfError=0 / parentError=INF.
// Cluster ib_first is PAGE-LOCAL since Stage B; the cull adds the page's slot
// base (pageSlotBase SSBO), so eviction/install never touches this meta.

// One indirect-draw batch: a run of meshes sharing material + pipeline key + VB/IB,
// so the draw binds those once and issues a single vkCmdDrawIndexedIndirectCount.
// entryOffset/entryCount = this group's slice of the meta/indirect arrays (exact
// prefix-sum capacity — clusters skew group sizes far too much for the old uniform
// maxGroupMesh regions).
struct Group {
    WorldMaterial* mat;
    bool        terrain;
    u32         stride, tcOffset;
    VkBuffer    vb, ib;
    VkIndexType iType;
    u32         meshCount;
    u32         entryOffset, entryCount;
};

// lodParams: x = viewportH/(2·tan(fovY/2))/thresholdPx (projected-error scale),
// y = min camera distance clamp; z,w spare.
struct CullPush { Fvector4 planes[6]; Fvector4 cameraPos; Fvector4 viewDir; Fvector4 lodParams; u32 numGroups, unused0 /*was maxGroupMesh*/, total, _pad; };
// Hi-Z occlusion cull push (240 B) — matches world_cull_hzb.comp `PC`, same shape
// as the proven detail_generate.comp push (viewProj + planes + cameraPos).
struct CullColorPush { Fmatrix viewProj; Fvector4 planes[6]; Fvector4 cameraPos; Fvector4 viewDir; Fvector4 lodParams; u32 numGroups, unused0 /*was maxGroupMesh*/, total, _pad; };

bool s_built = false;
u32  s_total = 0;          // total cullable ENTRIES (meshes + clusters) = dispatch size
xr_vector<Group> s_groups;
xr_vector<vkRender_Visual*> s_meshSet;   // pointer-sorted, for InSet() (CPU-queue exclusion)
xr_vector<vkFVisual*> s_cpuMeshes;       // non-GPU static leaves (wmark/tess/no-diffuse) — CPU draws these

// Pool-compaction bookkeeping (CompactPools): per GPU-set mesh, its meta run +
// how its entries reference the original pools. Filled during the grouping walk,
// consumed once by CompactPools, cleared on Destroy.
struct CompactRef {
    vkFVisual* fv;
    u32 metaFirst, metaCount;
    u8  clustered;   // entries live in the ClusterStream IB pools
    u8  ib32;        // u32 pool-global component DAG (page indices address the pool VB directly)
    u8  repacked;    // vertices live in the ClusterStream VB pool (original VB slice unused)
};
xr_vector<CompactRef> s_compactRefs;
bool s_poolsCompacted = false;

CVulkanBuffer* s_meta      = nullptr;
xr_vector<GpuMeshMeta> s_metaCPU;   // host copy for the throttled cut simulation (diag)
u32 s_buildStamp = 0;               // bumped each Build (VSM candidate-table freshness)
CVulkanBuffer* s_indirect  = nullptr;   // s_total cmds, per-group prefix-sum regions (frustum set)
CVulkanBuffer* s_count     = nullptr;   // numGroups u32 (frustum set)
CVulkanBuffer* s_groupBase = nullptr;   // numGroups u32 — each group's entryOffset (cmd region base)
// Cluster index data lives in the ClusterStream page pools since Stage B
// (PoolIB16 = u16 per-mesh DAGs rel. vBase, PoolIB32 = u32 pool-global
// component DAGs). Page directory + streaming source, filled by BuildClusters:
xr_vector<ClusterStream::PageRec> s_pageDir;
string_path s_cachePath = "";
u64 s_blobOff16 = 0, s_blobOff32 = 0, s_blobOffVb = 0;

VkDescriptorSetLayout s_setL  = VK_NULL_HANDLE;
VkDescriptorPool      s_pool  = VK_NULL_HANDLE;
VkDescriptorSet       s_set   = VK_NULL_HANDLE;
VkPipelineLayout      s_cullLayout = VK_NULL_HANDLE;
VkPipeline            s_cullPipe   = VK_NULL_HANDLE;

// ---- Phase 3: cluster-LOD shadow casters (sun far map + 2 cascades) ----
// One indirect/count slice per shadow target; the ortho cull writes exact
// prefix-sum regions (groupBase) inside each target's slice. Best-effort like
// the occlusion set — failure leaves s_shadowReady=false and vk_pass_shadow
// falls back to the per-mesh ShadowGPU path.
constexpr u32 kShadowTargets = 5;   // ShadowGPU::TGT_COUNT (far, casc0, casc1) + kTargetRain + kTargetDyn
bool s_shadowReady = false;
CVulkanBuffer* s_indirectSh = nullptr;   // kShadowTargets × s_total cmds
CVulkanBuffer* s_countSh    = nullptr;   // kShadowTargets × numGroups u32
VkDescriptorSetLayout s_setL3  = VK_NULL_HANDLE;
VkDescriptorPool      s_pool3  = VK_NULL_HANDLE;
VkDescriptorSet       s_set3   = VK_NULL_HANDLE;
VkPipelineLayout      s_cullLayout3 = VK_NULL_HANDLE;
VkPipeline            s_cullPipe3   = VK_NULL_HANDLE;
// Matches world_cull_shadow.comp PC (116 B).
struct CullShadowPush { Fvector4 planes[6]; float errBudget; u32 target, numGroups, total, atOn; };

// ---- Phase A: Hi-Z occlusion cull (separate set, drawn only in the color pass) ----
bool s_occlReady = false;
CVulkanBuffer* s_indirect2 = nullptr;  // numGroups * maxGroupMesh cmds (occlusion set)
CVulkanBuffer* s_count2    = nullptr;  // numGroups u32 (occlusion set)
VkDescriptorSetLayout s_setL2  = VK_NULL_HANDLE;
VkDescriptorPool      s_pool2  = VK_NULL_HANDLE;
VkDescriptorSet       s_set2   = VK_NULL_HANDLE;
VkPipelineLayout      s_cullLayout2 = VK_NULL_HANDLE;
VkPipeline            s_cullPipe2   = VK_NULL_HANDLE;

// Occlusion-stats readback (host-visible, mapped): per-frame we copy both count
// sets here [0..nGroups)=frustum, [nGroups..2n)=occlusion; CPU sums them throttled
// to log "culled K of N". Read is a frame or two stale (no fence wait) — fine for a
// diagnostic. Created raw via VMA (HOST_ACCESS_RANDOM = readable; the CVulkanBuffer
// helper only offers SEQUENTIAL_WRITE for storage buffers).
VkBuffer      s_countReadback      = VK_NULL_HANDLE;
VmaAllocation s_countReadbackAlloc = VK_NULL_HANDLE;
u32*          s_countReadbackPtr   = nullptr;
u32           s_nGroups            = 0;   // cached group count (readback stride)

// ---- Draw-command audit (white-polygon hunt 17-07) --------------------------
// Data, addressing and state on the GPU are all PROVEN correct (payload +
// slot-hash + state audits in vk_cluster_stream) yet the hole persists — so
// the bug must be in what the cull EMITS. Every ~3 s the frustum set (cmds +
// counts, exactly what the color pass consumes with r_hzb_cull 0) is copied to
// a host buffer along with a snapshot of the cull inputs of that dispatch
// (camera, lodParams, planes, stream bits, slot bases). A FIF cycle later each
// recorded draw is verified against the host expectation (region overflow,
// group routing, firstIndex/vertexOffset/indexCount, drawable bit), and every
// near-camera entry is coverage-checked: a cut member that SHOULD draw but is
// absent = the missing wall; a draw the cut can't explain = the white polygon.
// Region 1 = frustum set (s_indirect/s_count, depth prepass), region 2 = the
// HZB-occlusion set (s_indirect2/s_count2) — with r_hzb_cull 1 the COLOR pass
// consumes region 2, so a wall present in region 1 but absent in region 2 is
// exactly "depth written, color skipped" = hole + background showing through.
VkBuffer      s_cmdAudit        = VK_NULL_HANDLE;
VmaAllocation s_cmdAuditAlloc   = VK_NULL_HANDLE;
u8*           s_cmdAuditPtr     = nullptr;
bool          s_cmdAuditPending = false;
bool          s_cmdAuditHasOccl = false;   // region 2 captured this cycle
u32           s_cmdAuditFrame   = 0;   // dwFrame the copy was recorded
u32           s_cmdAuditLast    = 0;   // dwTimeGlobal of the last verify
Fvector       s_cmdAuditCam{}, s_cmdAuditDir{};
Fvector4      s_cmdAuditLod{};
Fvector4      s_cmdAuditPlanes[6];
xr_vector<u32> s_cmdAuditBits;       // stream-bits snapshot at record time
xr_vector<u32> s_cmdAuditSlotBase;   // page slot-base snapshot at record time

const char* GroupMatName(u32 g)
{
    return (g < (u32)s_groups.size() && s_groups[g].mat && s_groups[g].mat->name.c_str())
        ? s_groups[g].mat->name.c_str() : "?";
}

// ---- Cluster debug view (r_cluster_debug): the LOD cut made visible, UE-style.
// Redraws the SAME per-group indirect regions the color pass consumed, with a
// tiny pipeline that colors by cluster id (the cull shader stores the entry id
// in firstInstance → gl_InstanceIndex, no extra bindings). mode 1 = flat fill,
// mode 2 = wireframe (LINE polygon) over the lit scene. Lazy-built.
VkPipelineLayout        s_dbgLayout = VK_NULL_HANDLE;
xr_map<u64, VkPipeline> s_dbgPipes;   // key = stride*2 + (line ? 1 : 0)
bool                    s_dbgFailed = false;   // don't retry every frame after a failure
// Debug set 0 = the meta SSBO (mode 4 LOD-health view reads parentError per entry).
VkDescriptorSetLayout   s_dbgSetL = VK_NULL_HANDLE;
VkDescriptorPool        s_dbgPool = VK_NULL_HANDLE;
VkDescriptorSet         s_dbgSet  = VK_NULL_HANDLE;

// DAG-cut projection scale for the cull shaders (recomputed per frame — the px
// threshold r_cluster_lod is a live knob). worldError · x / dist = error/threshold;
// the shader draws the first cut whose value stays <= 1.
Fvector4 LodParams()
{
    const float fovR    = deg2rad(Device.fFOV);
    const float pxScale = float(Device.dwHeight) / (2.f * tanf(fovR * 0.5f));
    // .z = crossfade band (0 = hard cut). Gated on the fade shader variants
    // being present — the cull must not widen the cut if the draw side can't
    // dither (that would double-draw the whole transition band).
    const bool canFade = PipelineCache::WorldLmapFadeVS() != VK_NULL_HANDLE
                      && PipelineCache::GetDepthFadePipeline(32) != VK_NULL_HANDLE;
    // .w = vertical focal scale P11 = 1/tan(fovY/2). The HZB occlusion cull
    // needs it for the sphere's true screen footprint: ndcHeight = 2·r·P11/w.
    // The old footprint (2·r/w, no focal term) UNDERSTATED the size ~1.5× →
    // too fine a mip → the 2×2 block missed part of the footprint → false
    // culls (the reason r_hzb_cull stayed situational).
    Fvector4 p; p.set(pxScale / _max(0.05f, ps_r_cluster_lod), 0.01f,
                      canFade ? _max(0.f, ps_r_cluster_fade) : 0.f,
                      1.f / _max(0.05f, tanf(fovR * 0.5f)));
    return p;
}

void MemBarrier(VkCommandBuffer cmd, VkAccessFlags src, VkAccessFlags dst,
                VkPipelineStageFlags ss, VkPipelineStageFlags ds)
{
    VkMemoryBarrier b{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    b.srcAccessMask = src; b.dstAccessMask = dst;
    vkCmdPipelineBarrier(cmd, ss, ds, 0, 1, &b, 0, nullptr, 0, nullptr);
}

// Recursively partition ALL renderable STATIC world-mesh leaves into:
//   gpuOut — opaque/AT meshes the GPU forward path can draw (have a material set +
//            diffuse, not wmark/tess) → compute-culled + indirect-drawn.
//   cpuOut — everything else still renderable (wmark decals, tessellated, no diffuse
//            / no material) → drawn by the CPU RenderQueue (Flush handles them).
// gpuOut ∪ cpuOut = exactly what the old full CPU walk drew (deduped), so nothing
// is lost. Descends MT_HIERRARHY/MT_LOD (getVisual references; deduped by caller).
// Trees/skeletons/particles are other passes → skipped entirely.
void ExtractMeshes(vkRender_Visual* rv, xr_vector<vkFVisual*>& gpuOut, xr_vector<vkFVisual*>& cpuOut)
{
    if (!rv) return;
    const u32 t = rv->Type;
    if (t == MT_NORMAL || t == MT_PROGRESSIVE) {
        auto* fv = static_cast<vkFVisual*>(rv);
        if (!fv->m_mesh.IsValid() || !fv->m_mesh.p_rm_Vertices || !fv->m_mesh.p_rm_Indices) return;  // not renderable
        WorldMaterial* mat = fv->m_pWorldMaterial;
        // `tessellated` = the material HAS a height texture (every bump-mapped
        // wall!), but the CPU queue only tessellates when r_tess is globally on
        // (RenderQueue::Flush: tessAvail) — otherwise those meshes draw on the
        // very same flat lmap/vlit pipelines the GPU path binds. So when tess
        // is inactive at Build time, claim them for the GPU set: that's most
        // of the building geometry (and the cluster-LOD coverage with it).
        // Flipping r_tess later only takes effect on statics after level reload.
        const bool tessActive = PipelineCache::TessAvailable() && ps_r_tess > 0.5f;
        const bool gpuOk = mat && !mat->isWmark && !(mat->tessellated && tessActive)
                        && (mat->view != VK_NULL_HANDLE || mat->isTerrain);
        if (gpuOk) gpuOut.push_back(fv);
        else       cpuOut.push_back(fv);
        return;
    }
    if (t == MT_HIERRARHY || t == MT_LOD) {
        auto* hv = dynamic_cast<vkFHierrarhyVisual*>(rv);
        if (!hv) return;
        for (auto* child : hv->children)
            ExtractMeshes(child, gpuOut, cpuOut);
    }
}

// ============================================================================
// Cluster split (r_cluster) — Phase 1 of the cluster-LOD system.
// Big static meshes are split at load into ~128-tri meshlets (meshoptimizer)
// so the compute cull works at cluster granularity: a whole building wall no
// longer draws because one corner peeks into the frustum, and the Hi-Z test
// gets small spheres it can actually reject (fragment-sized spheres almost
// never pass). Cluster indices are u16 RELATIVE TO vBase exactly like the
// source IB (level indices are u16-per-slice already), so the draw side needs
// nothing new: same vertex fetch, same first_vertex, just another VkBuffer.
// NOTE: no normal-cone backface culling — every world pipeline rasterizes
// with CULL_MODE_NONE (two-sided), a cone test would drop visible geometry.
// ============================================================================
constexpr u32 kClusterMaxVerts = 64;
constexpr u32 kClusterMaxTris  = 128;   // multiple of 4 (meshopt requirement)

// GPU-readback of a device-local buffer into a host vector — same one-shot
// staging pattern as CTreeManager::ReadbackRange (pools carry TRANSFER_SRC
// specifically for this, see rvk_loader). Returns false on any failure.
bool ReadbackBuffer(VkBuffer src, VkDeviceSize size, xr_vector<u8>& out)
{
    if (src == VK_NULL_HANDLE || size == 0) return false;
    CVulkanBuffer stg;
    stg.Create(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
    VkCommandBuffer cmd = VulkanHW.BeginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) { stg.Destroy(); return false; }
    VkBufferCopy cp{ 0, 0, size };
    vkCmdCopyBuffer(cmd, src, stg.GetHandle(), 1, &cp);
    VulkanHW.EndSingleTimeCommands(cmd);   // waits (one-shot)
    stg.Invalidate();
    void* p = stg.Map();
    if (!p) { stg.Destroy(); return false; }
    out.resize((size_t)size);
    memcpy(out.data(), p, (size_t)size);
    stg.Unmap();
    stg.Destroy();
    return true;
}

// The drawn IB slice = exactly what the CPU path draws: PROGRESSIVE meshes use
// their finest sliding-window slice, not the whole span (the coarse slices
// poke through the fine mesh — same rule as the meta fill / vk_shadow_gpu).
void DrawnSlice(vkFVisual* fv, u32& first, u32& count)
{
    first = fv->m_mesh.iBase; count = fv->m_mesh.iCount;
    if (fv->Type == MT_PROGRESSIVE) {
        auto* pg = static_cast<vkFProgressive*>(fv);
        if (pg->sw_count > 0 && pg->sw_offsets && pg->sw_counts) {
            first = fv->m_mesh.iBase + pg->sw_offsets[0];
            count = pg->sw_counts[0];
        }
    }
}

// Which cluster-meta entries belong to a mesh. Per-mesh DAGs own a contiguous
// run; component DAGs (Phase 2.5) scatter a mesh's entries across clusters
// shared with its neighbors, so it's an explicit index list either way. ib32
// picks the IB the entries reference (u16 rel-vBase vs u32 pool-global).
struct MeshEntries { bool ib32; xr_vector<u32> entries; };

u32 s_statSimplified = 0, s_statStalled = 0;   // DAG diag (reset per BuildClusters)
u32 s_statCompUnits = 0, s_statCompMeshes = 0; // Phase 2.5 diag: component units / meshes merged into them

// ============================================================================
// Cluster DAG disk cache — the clodBuild result depends only on the level
// geometry + build params, so it's baked once to $app_data_root$ and reloaded
// on subsequent loads of the same level (~0.5s parallel bake → ~0). Pattern:
// vk_pipeline_cache. One file per level ("vk_cluster_<crc(levelpath)>.bin"),
// stamped with level.geom size^CRC so edited geometry rebuilds automatically.
// ============================================================================
constexpr u32 kClusterCacheMagic   = 0x4B4C4356u;   // 'VCLK'
constexpr u32 kClusterCacheVersion = 13;            // bump on ANY builder change (v11: Stage B page-ordered blobs; v12: slice 2 vertex repack blob; v13: bake hygiene — self-loop stall entries dropped)

#pragma pack(push, 1)
// v12: pageCount + absolute file offsets of the idx/vertex blobs (the blobs
// are read lazily — the cache file doubles as the page-streaming source).
struct CacheHeader { u32 magic, params, stamp, meshCount, totalMeta, totalIdx16, totalIdx32, pageCount, totalVbBytes; u64 blobOff16, blobOff32, blobOffVb; };
// Stable mesh identity. vbSize/ibSize discriminate the owning pools: vBase/ibFirst are
// offsets INTO a pool, and two different pools can host meshes at equal offsets.
struct CacheMeshKey { u32 vBase, ibFirst, idxCount, vCount, stride, tcOffset, vbSize, ibSize; };
struct CacheMeshRec { CacheMeshKey key; u32 ib32, entryCount; };  // + entryCount inline u32 meta indices follow
#pragma pack(pop)

u32 ClusterParamsHash()
{
    u32 h = kClusterCacheVersion * 2166136261u;
    h = (h ^ (u32)kClusterMaxTris)     * 16777619u;
    h = (h ^ (u32)ps_r_cluster_tris)   * 16777619u;
    h = (h ^ (ps_r_cluster_merge ? 1u : 0u)) * 16777619u;
    return h;
}

CacheMeshKey MakeMeshKey(vkFVisual* fv)
{
    u32 first, count; DrawnSlice(fv, first, count);
    return { fv->m_mesh.vBase, first, count, fv->m_mesh.vCount, fv->m_mesh.vStride, fv->m_mesh.tcOffset,
             (u32)fv->m_mesh.p_rm_Vertices->GetSize(), (u32)fv->m_mesh.p_rm_Indices->GetSize() };
}

void ClusterCachePath(string_path& fn, u32& stamp)
{
    string_path lp{};
    FS.update_path(lp, "$level$", "");
    string64 name;
    xr_sprintf(name, "vk_cluster_%08x.bin", crc32(lp, (u32)xr_strlen(lp)));
    FS.update_path(fn, "$app_data_root$", name);

    stamp = 0;
    if (IReader* r = FS.r_open("$level$", "level.geom")) {
        stamp = (u32)r->length() ^ crc32(r->pointer(), (u32)_min<size_t>(65536, (size_t)r->length()));
        FS.r_close(r);
    }
}

// Cache hit = every eligible mesh has a record with a bit-identical key.
// v11: read with plain stdio (NOT FS.r_open — that pulls the whole 400 MB file
// into RAM; we only parse header/records/pages/meta and leave the page-ordered
// idx blobs ON DISK: the same file is the runtime page-streaming source).
// ib_first in the stored meta is page-local; the page directory + blob offsets
// land in s_pageDir / s_cachePath / s_blobOff*.
bool LoadClusterCache(const xr_vector<vkFVisual*>& eligMeshes,
                      xr_vector<GpuMeshMeta>& outMeta, xr_vector<u16>& outIdx16, xr_vector<u32>& outIdx32,
                      xr_map<vkFVisual*, MeshEntries>& outRanges)
{
    (void)outIdx16; (void)outIdx32;   // blobs stay on disk (v11)
    string_path fn; u32 stamp;
    ClusterCachePath(fn, stamp);
    FILE* f = fopen(fn, "rb");
    if (!f) return false;
    _fseeki64(f, 0, SEEK_END);
    const u64 fileSize = (u64)_ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);
    auto rd = [&](void* dst, size_t bytes) { return fread(dst, 1, bytes, f) == bytes; };

    bool ok = false;
    const char* missWhy = nullptr;   // every non-hit path names itself — misses must be diagnosable from the log
    do {
        CacheHeader h;
        if (!rd(&h, sizeof(h))) { missWhy = "truncated header"; break; }
        if (h.magic != kClusterCacheMagic) { missWhy = "bad magic"; break; }
        if (h.params != ClusterParamsHash()) { missWhy = "params/version changed"; break; }
        if (h.stamp != stamp) { missWhy = "level.geom stamp changed"; break; }
        if (h.meshCount != (u32)eligMeshes.size()) { missWhy = "mesh count differs"; break; }

        // Records carry inline entry lists (variable length) — read sequentially.
        struct Rec { CacheMeshRec r; xr_vector<u32> entries; };
        xr_vector<Rec> recs(h.meshCount);
        bool match = true;
        for (u32 i = 0; i < h.meshCount && match; ++i) {
            if (!rd(&recs[i].r, sizeof(CacheMeshRec))) { match = false; break; }
            const u32 n = recs[i].r.entryCount;
            if (n > h.totalMeta) { match = false; break; }
            recs[i].entries.resize(n);
            if (n && !rd(recs[i].entries.data(), n * sizeof(u32))) { match = false; break; }
            for (u32 e : recs[i].entries)
                if (e >= h.totalMeta) { match = false; break; }
        }
        if (!match) { missWhy = "corrupt records"; break; }
        if (h.pageCount == 0 || h.pageCount > 1u << 20) { missWhy = "bad page table"; break; }
        if (h.blobOff16 + (u64)h.totalIdx16 * sizeof(u16) > fileSize
         || h.blobOff32 + (u64)h.totalIdx32 * sizeof(u32) > fileSize
         || h.blobOffVb + (u64)h.totalVbBytes > fileSize) { missWhy = "truncated blobs"; break; }
        s_pageDir.resize(h.pageCount);
        if (!rd(s_pageDir.data(), h.pageCount * sizeof(ClusterStream::PageRec))) { missWhy = "truncated page table"; break; }

        // Mesh extraction order is pointer-sort — NOT stable across runs — so
        // records match by KEY, not by position. Identical keys are legal: two
        // visuals can reference the SAME geometry slice (same pool, offsets,
        // counts ⇒ same triangles ⇒ same DAG), and one record serves them all —
        // Pripyat-sized levels have thousands of such shares, and treating them
        // as a miss meant an 18s rebake every single load.
        auto keyHash = [](const CacheMeshKey& k) -> u64 {
            u64 hh = 1469598103934665603ull;
            const u8* b = (const u8*)&k;
            for (size_t i = 0; i < sizeof(k); ++i) { hh ^= b[i]; hh *= 1099511628211ull; }
            return hh;
        };
        std::unordered_map<u64, u32> byKey;
        byKey.reserve(h.meshCount);
        for (u32 i = 0; i < h.meshCount && match; ++i) {
            auto ins = byKey.emplace(keyHash(recs[i].r.key), i);
            if (!ins.second && memcmp(&recs[i].r.key, &recs[ins.first->second].r.key, sizeof(CacheMeshKey)) != 0)
                match = false;   // true 64-bit hash collision of DIFFERENT keys — bail, don't mis-serve
        }
        if (!match) { missWhy = "key hash collision"; break; }
        xr_vector<u32> recOf(eligMeshes.size());
        for (u32 i = 0; i < (u32)eligMeshes.size(); ++i) {
            const CacheMeshKey k = MakeMeshKey(eligMeshes[i]);
            auto it = byKey.find(keyHash(k));
            if (it == byKey.end() || memcmp(&k, &recs[it->second].r.key, sizeof(k)) != 0) { match = false; break; }
            recOf[i] = it->second;
        }
        if (!match) { missWhy = "mesh key not in cache"; break; }

        outMeta.resize(h.totalMeta);
        if (!rd(outMeta.data(), (size_t)h.totalMeta * sizeof(GpuMeshMeta))) { missWhy = "truncated meta"; break; }
        for (u32 i = 0; i < (u32)eligMeshes.size(); ++i) {
            Rec& rec = recs[recOf[i]];
            if (rec.r.entryCount)   // COPY entries — a shared-slice record can serve several visuals
                outRanges.emplace(eligMeshes[i], MeshEntries{ rec.r.ib32 != 0, rec.entries });
        }
        xr_strcpy(s_cachePath, fn);
        s_blobOff16 = h.blobOff16;
        s_blobOff32 = h.blobOff32;
        s_blobOffVb = h.blobOffVb;
        ok = true;
    } while (false);

    fclose(f);
    if (!ok) {
        s_pageDir.clear();
        Msg("![VK Cluster] cache MISS (%s) — full rebake", missWhy ? missWhy : "?");
    }
    return ok;
}

// v12 layout: header | records+entry lists | page table | meta | idx16 | idx32
// | vertex blob. The blob offsets are precomputed and stored in the header —
// the cache file is also the RUNTIME page-streaming source, so pages must be
// seekable directly. On success s_cachePath / s_blobOff* point the streamer at
// the fresh file.
void SaveClusterCache(const xr_vector<vkFVisual*>& eligMeshes,
                      const xr_vector<GpuMeshMeta>& meta, const xr_vector<u16>& idx16, const xr_vector<u32>& idx32,
                      const xr_vector<u8>& vbBlob,
                      const xr_map<vkFVisual*, MeshEntries>& ranges,
                      const xr_vector<ClusterStream::PageRec>& pages)
{
    string_path fn; u32 stamp;
    ClusterCachePath(fn, stamp);
    s_cachePath[0] = 0;   // set only on a successful write
    IWriter* w = FS.w_open(fn);
    if (!w) { Msg("![VK Cluster] cache write failed (%s)", fn); return; }

    size_t listBytes = 0;
    for (vkFVisual* fv : eligMeshes) {
        auto it = ranges.find(fv);
        if (it != ranges.end()) listBytes += it->second.entries.size() * sizeof(u32);
    }
    const u64 metaOff = sizeof(CacheHeader) + eligMeshes.size() * sizeof(CacheMeshRec) + listBytes
                      + pages.size() * sizeof(ClusterStream::PageRec);
    const u64 off16 = metaOff + meta.size() * sizeof(GpuMeshMeta);
    const u64 off32 = off16 + idx16.size() * sizeof(u16);
    const u64 offVb = off32 + idx32.size() * sizeof(u32);

    CacheHeader h{ kClusterCacheMagic, ClusterParamsHash(), stamp,
                   (u32)eligMeshes.size(), (u32)meta.size(), (u32)idx16.size(), (u32)idx32.size(),
                   (u32)pages.size(), (u32)vbBlob.size(), off16, off32, offVb };
    w->w(&h, sizeof(h));
    for (vkFVisual* fv : eligMeshes) {
        CacheMeshRec rec{ MakeMeshKey(fv), 0, 0 };
        auto it = ranges.find(fv);
        if (it != ranges.end()) { rec.ib32 = it->second.ib32 ? 1u : 0u; rec.entryCount = (u32)it->second.entries.size(); }
        w->w(&rec, sizeof(rec));
        if (rec.entryCount) w->w(it->second.entries.data(), rec.entryCount * sizeof(u32));
    }
    if (!pages.empty()) w->w(pages.data(), (u32)(pages.size() * sizeof(ClusterStream::PageRec)));
    if (!meta.empty())  w->w(meta.data(),  (u32)(meta.size()  * sizeof(GpuMeshMeta)));
    if (!idx16.empty()) w->w(idx16.data(), (u32)(idx16.size() * sizeof(u16)));
    if (!idx32.empty()) w->w(idx32.data(), (u32)(idx32.size() * sizeof(u32)));
    if (!vbBlob.empty()) w->w(vbBlob.data(), (u32)vbBlob.size());
    FS.w_close(w);
    xr_strcpy(s_cachePath, fn);
    s_blobOff16 = off16;
    s_blobOff32 = off32;
    s_blobOffVb = offVb;
    Msg("[VK Cluster] cache saved: %u meshes, %u pages, %.1f MB (%.1f MB vertex blob)", (u32)eligMeshes.size(), (u32)pages.size() - 1,
        double(offVb + vbBlob.size()) / (1024.0 * 1024.0), double(vbBlob.size()) / (1024.0 * 1024.0));
}

// Split eligible meshes into meshlets with a LOD DAG. Two build paths:
//  - per-mesh (Phase 2): one clodBuild per mesh; u16 indices rel. to vBase.
//  - component (Phase 2.5, r_cluster_merge): solid fragments whose dilated
//    AABBs touch within one VB pool merge into ONE clodBuild — the simplifier
//    sees the building as a single surface, so a low-poly door DISSOLVES into
//    its facade at distance instead of collapsing onto itself. Each cluster's
//    triangles are binned back to their source mesh (by v0) → one entry per
//    (cluster × mesh); entries share the cluster's LOD spheres/errors, so the
//    cluster still flips atomically (crack-free). Cross-mesh triangles can't
//    fit a u16 window over one vBase → component clusters use pool-global u32
//    indices with first_vertex = 0 (the cull/draw need nothing new for that).
// Fills outMeta (per-entry data, group patched later), outIdx16/outIdx32 (the
// shared cluster IBs) and outRanges (mesh → its entry list). Terrain is
// excluded: tiles are already grid-partitioned, and the prepass TES displaces
// them (spheres would lie).
void BuildClusters(const xr_vector<vkFVisual*>& meshes,
                   xr_vector<GpuMeshMeta>& outMeta, xr_vector<u16>& outIdx16, xr_vector<u32>& outIdx32,
                   xr_vector<u8>& outVbBlob,
                   xr_map<vkFVisual*, MeshEntries>& outRanges)
{
    const u32 minTris = (u32)_max(ps_r_cluster_tris, (int)kClusterMaxTris);
    s_statSimplified = 0; s_statStalled = 0; s_statCompUnits = 0; s_statCompMeshes = 0;
    s_pageDir.clear(); s_cachePath[0] = 0; s_blobOff16 = s_blobOff32 = s_blobOffVb = 0;

    // Eligibility (no GPU readback needed) — the exact predicate the cache was
    // saved under; buffer-content guards run only on the build path below.
    // With component merge on, SOLID meshes of ANY size are candidates (a
    // 100-tri door must JOIN the building's DAG to dissolve into it); AT
    // meshes and merge-off solids keep the size filter — alone, a small mesh
    // gains nothing from clustering below ~2 meshlets.
    xr_vector<vkFVisual*> eligMeshes;
    for (vkFVisual* fv : meshes) {
        if (fv->m_pWorldMaterial->isTerrain) continue;
        if (fv->m_mesh.vStride < 12 || fv->m_mesh.vCount == 0) continue;
        u32 first, count; DrawnSlice(fv, first, count);
        if (count < 3) continue;
        const bool mergeCand = ps_r_cluster_merge && fv->m_pWorldMaterial->alphaRef < 0.f;
        if (!mergeCand && count < minTris * 3) continue;
        eligMeshes.push_back(fv);
    }
    if (eligMeshes.empty()) return;

    // Disk cache: on a hit the whole readback + clodBuild is skipped.
    if (LoadClusterCache(eligMeshes, outMeta, outIdx16, outIdx32, outRanges)) {
        Msg("[VK Cluster] DAG loaded from cache: %u meshes, %u entries", (u32)outRanges.size(), (u32)outMeta.size());
        return;
    }

    // Read back each unique VB/IB pool ONCE (host copies), then slice in memory —
    // per-mesh readback would mean thousands of queue round-trips.
    xr_map<VkBuffer, xr_vector<u8>> vbData, ibData;
    for (vkFVisual* fv : eligMeshes) {
        vbData.emplace(fv->m_mesh.p_rm_Vertices->GetHandle(), xr_vector<u8>());
        ibData.emplace(fv->m_mesh.p_rm_Indices->GetHandle(),  xr_vector<u8>());
    }
    for (vkFVisual* fv : eligMeshes) {   // fill (buffer sizes come from the owning CVulkanBuffer)
        auto vit = vbData.find(fv->m_mesh.p_rm_Vertices->GetHandle());
        if (vit != vbData.end() && vit->second.empty())
            if (!ReadbackBuffer(vit->first, fv->m_mesh.p_rm_Vertices->GetSize(), vit->second))
                { Msg("![VK Cluster] VB readback failed — mesh(es) stay unclustered"); vbData.erase(vit); }
        auto iit = ibData.find(fv->m_mesh.p_rm_Indices->GetHandle());
        if (iit != ibData.end() && iit->second.empty())
            if (!ReadbackBuffer(iit->first, fv->m_mesh.p_rm_Indices->GetSize(), iit->second))
                { Msg("![VK Cluster] IB readback failed — mesh(es) stay unclustered"); ibData.erase(iit); }
    }

    // Full guards (buffer contents) → the build list with resolved pointers.
    struct Elig { vkFVisual* fv; const float* pos; const u16* idx; u32 idxCount; };
    xr_vector<Elig> elig;
    for (vkFVisual* fv : eligMeshes) {
        const auto& mesh = fv->m_mesh;
        u32 ibFirst, idxCount; DrawnSlice(fv, ibFirst, idxCount);

        auto vit = vbData.find(mesh.p_rm_Vertices->GetHandle());
        auto iit = ibData.find(mesh.p_rm_Indices->GetHandle());
        if (vit == vbData.end() || iit == ibData.end()) continue;
        if ((size_t)(ibFirst + idxCount) * sizeof(u16) > iit->second.size()) continue;
        if ((size_t)(mesh.vBase + mesh.vCount) * mesh.vStride > vit->second.size()) continue;

        const float* pos = (const float*)(vit->second.data() + (size_t)mesh.vBase * mesh.vStride);
        const u16*   idx = (const u16*)iit->second.data() + ibFirst;

        // Corrupt-data guard: indices must stay inside the mesh's vertex slice.
        bool ok = true;
        for (u32 i = 0; i < idxCount; ++i) if (idx[i] >= mesh.vCount) { ok = false; break; }
        if (!ok) { Msg("![VK Cluster] mesh has out-of-slice indices — skipped"); continue; }

        elig.push_back({ fv, pos, idx, idxCount });
    }
    if (elig.empty()) return;

    // ---- Build units --------------------------------------------------------
    // A unit = one clodBuild call. With r_cluster_merge, solid fragments whose
    // dilated AABBs overlap within one VB pool (same vb/stride/tcOffset — the
    // draw binds one VB per group and the attr layout must be uniform) union
    // into COMPONENT units; oversized components split along their longest
    // axis at a triangle cap (keeps the bake parallel and clod memory bounded).
    // Everything else (AT meshes, isolated solids) stays per-mesh.
    struct Unit { xr_vector<u32> members; bool comp; u32 tris; };
    xr_vector<Unit> units;
    constexpr u32 kNoUnit = 0xFFFFFFFFu;
    xr_vector<u32> unitOf(elig.size(), kNoUnit);

    if (ps_r_cluster_merge) {
        xr_vector<u32> uf(elig.size());
        for (u32 i = 0; i < (u32)uf.size(); ++i) uf[i] = i;
        auto find = [&](u32 x) { while (uf[x] != x) { uf[x] = uf[uf[x]]; x = uf[x]; } return x; };

        xr_map<std::tuple<VkBuffer, u32, u32>, xr_vector<u32>> poolsOf;
        for (u32 i = 0; i < (u32)elig.size(); ++i) {
            vkFVisual* fv = elig[i].fv;
            if (fv->m_pWorldMaterial->alphaRef >= 0.f) continue;   // AT: per-mesh path
            poolsOf[std::make_tuple(fv->m_mesh.p_rm_Vertices->GetHandle(),
                                    (u32)fv->m_mesh.vStride, (u32)fv->m_mesh.tcOffset)].push_back(i);
        }

        constexpr float kDilate = 0.5f;   // fragments of one building adjoin exactly; dilation absorbs authoring slack
        for (auto& kv : poolsOf) {
            const xr_vector<u32>& list = kv.second;
            for (size_t a = 0; a < list.size(); ++a) {
                const Fbox& ba = elig[list[a]].fv->vis.box;
                for (size_t b = a + 1; b < list.size(); ++b) {
                    const u32 ra = find(list[a]), rb = find(list[b]);
                    if (ra == rb) continue;
                    const Fbox& bb = elig[list[b]].fv->vis.box;
                    if (ba.min.x - kDilate <= bb.max.x && bb.min.x - kDilate <= ba.max.x &&
                        ba.min.y - kDilate <= bb.max.y && bb.min.y - kDilate <= ba.max.y &&
                        ba.min.z - kDilate <= bb.max.z && bb.min.z - kDilate <= ba.max.z)
                        uf[rb] = ra;
                }
            }
        }

        xr_map<u32, xr_vector<u32>> comps;
        for (auto& kv : poolsOf)
            for (u32 i : kv.second)
                comps[find(i)].push_back(i);

        // Recursive spatial split: chop every component along its longest axis
        // until BOTH the tri cap and the EXTENT cap hold. The extent cap is
        // what keeps the DAG usable: monotonic containment (Karis) inflates
        // deep-level group spheres toward the unit's bounding sphere, and the
        // parent test needs the camera OUTSIDE that sphere (dist - r). AABB
        // chaining (fence -> shed -> house...) builds map-sized components —
        // measured parent spheres avg 187m / max 1197m on Кордон = the cut
        // froze at full detail everywhere. ~48m keeps a building whole while
        // deep spheres stay escapable a few dozen meters out.
        constexpr u32   kCompMaxTris   = 200000;
        constexpr float kCompMaxExtent = 48.f;
        for (auto& kv : comps) {
            xr_vector<xr_vector<u32>> stack;
            stack.push_back(std::move(kv.second));
            while (!stack.empty()) {
                xr_vector<u32> cur = std::move(stack.back());
                stack.pop_back();
                if (cur.size() < 2) continue;   // singleton → per-mesh path below

                u32 tris = 0;
                Fbox all; all.invalidate();
                for (u32 i : cur) { tris += elig[i].idxCount / 3; all.merge(elig[i].fv->vis.box); }
                Fvector ext; all.getsize(ext);
                const float maxExt = _max(ext.x, _max(ext.y, ext.z));

                if (tris <= kCompMaxTris && maxExt <= kCompMaxExtent) {
                    Unit u{}; u.comp = true; u.tris = tris;
                    u.members = std::move(cur);
                    for (u32 m : u.members) unitOf[m] = (u32)units.size();
                    units.push_back(std::move(u));
                    continue;
                }

                const int axis = (ext.x >= ext.y && ext.x >= ext.z) ? 0 : (ext.y >= ext.z ? 1 : 2);
                std::sort(cur.begin(), cur.end(), [&](u32 a, u32 b) {
                    const Fbox& A = elig[a].fv->vis.box; const Fbox& B = elig[b].fv->vis.box;
                    return (&A.min.x)[axis] + (&A.max.x)[axis] < (&B.min.x)[axis] + (&B.max.x)[axis];
                });
                // Prefer cutting at the largest spatial GAP along the axis — a gap
                // wider than the AABB dilate means nothing is welded across it, so
                // the cut adds no cross-chunk locks and buildings stay whole. Fall
                // back to the tri-count median when the run is contiguous. Both
                // halves are non-empty by construction, so the recursion ends.
                u32 half = 0;
                {
                    float bestGap = 0.75f;   // > kDilate: separable by construction
                    float runMax = -1e30f;
                    for (u32 k = 0; k < (u32)cur.size(); ++k) {
                        const Fbox& B = elig[cur[k]].fv->vis.box;
                        if (k && (&B.min.x)[axis] - runMax > bestGap) { bestGap = (&B.min.x)[axis] - runMax; half = k; }
                        runMax = _max(runMax, (&B.max.x)[axis]);
                    }
                }
                if (half == 0) {
                    u32 acc = 0;
                    for (u32 k = 0; k < (u32)cur.size() - 1 && acc * 2 < tris; ++k) { acc += elig[cur[k]].idxCount / 3; half = k + 1; }
                    if (half == 0) half = 1;
                }
                xr_vector<u32> right(cur.begin() + half, cur.end());
                cur.resize(half);
                stack.push_back(std::move(cur));
                stack.push_back(std::move(right));
            }
        }
        s_statCompUnits = (u32)units.size();
        for (const Unit& u : units) s_statCompMeshes += (u32)u.members.size();
    }

    // Not merged: per-mesh unit if big enough, else PLAIN (stays a whole-mesh
    // entry — but its vertices still pin shared seams below: it never moves).
    for (u32 i = 0; i < (u32)elig.size(); ++i) {
        if (unitOf[i] != kNoUnit) continue;
        if (elig[i].idxCount < minTris * 3) continue;
        Unit u{}; u.comp = false; u.tris = elig[i].idxCount / 3; u.members.push_back(i);
        unitOf[i] = (u32)units.size();
        units.push_back(std::move(u));
    }

    // ---- Attach pass ---------------------------------------------------------
    // The extent split can orphan a small mesh from its host surface (a window
    // frame separated from its wall when the wall's chunk got cut) — and an
    // orphan below the per-mesh size gate becomes a PLAIN entry: full detail
    // forever, unable to dissolve into anything (this was exactly the "frames
    // never dissolve at any distance" report: ~1500 orphans, all parentError
    // INF). Reattach every leftover solid merge candidate to the TOUCHING unit
    // whose bounds grow the least (its wall — not the 500m rail line that also
    // happens to touch), converting single-mesh hosts into components. Hosts
    // must be all-solid and share the orphan's pool (vb/stride/tcOffset).
    if (ps_r_cluster_merge) {
        struct HostInfo { Fbox box; bool solid; VkBuffer vb; u32 stride, tc; };
        xr_vector<HostInfo> hosts(units.size());
        for (u32 u = 0; u < (u32)units.size(); ++u) {
            HostInfo& h = hosts[u];
            h.box.invalidate(); h.solid = true;
            vkFVisual* f0 = elig[units[u].members[0]].fv;
            h.vb = f0->m_mesh.p_rm_Vertices->GetHandle(); h.stride = f0->m_mesh.vStride; h.tc = f0->m_mesh.tcOffset;
            for (u32 m : units[u].members) {
                h.box.merge(elig[m].fv->vis.box);
                if (elig[m].fv->m_pWorldMaterial->alphaRef >= 0.f) h.solid = false;
            }
        }
        constexpr float kAttachDilate = 0.5f;
        auto overlaps = [&](const Fbox& a, const Fbox& b) {
            return a.min.x - kAttachDilate <= b.max.x && b.min.x - kAttachDilate <= a.max.x
                && a.min.y - kAttachDilate <= b.max.y && b.min.y - kAttachDilate <= a.max.y
                && a.min.z - kAttachDilate <= b.max.z && b.min.z - kAttachDilate <= a.max.z;
        };
        u32 attached = 0;
        for (u32 i = 0; i < (u32)elig.size(); ++i) {
            if (unitOf[i] != kNoUnit) continue;
            vkFVisual* fv = elig[i].fv;
            if (fv->m_pWorldMaterial->alphaRef >= 0.f) continue;   // AT never joins components
            const Fbox& bi = fv->vis.box;
            u32 best = kNoUnit; float bestExt = 1e30f;
            for (u32 u = 0; u < (u32)units.size(); ++u) {
                const HostInfo& h = hosts[u];
                if (!h.solid || h.vb != fv->m_mesh.p_rm_Vertices->GetHandle()
                    || h.stride != fv->m_mesh.vStride || h.tc != fv->m_mesh.tcOffset) continue;
                if (!overlaps(bi, h.box)) continue;   // cheap unit-bounds reject
                bool touch = false;
                for (u32 m : units[u].members)
                    if (overlaps(bi, elig[m].fv->vis.box)) { touch = true; break; }
                if (!touch) continue;
                Fbox mb = h.box; mb.merge(bi);
                Fvector me; mb.getsize(me);
                const float ext = _max(me.x, _max(me.y, me.z));
                if (ext < bestExt) { bestExt = ext; best = u; }
            }
            if (best == kNoUnit) continue;   // genuinely isolated → plain
            unitOf[i] = best;
            units[best].members.push_back(i);
            units[best].tris += elig[i].idxCount / 3;
            units[best].comp = units[best].members.size() >= 2;
            hosts[best].box.merge(bi);
            ++attached;
        }
        s_statCompMeshes += attached;
    }
    if (units.empty()) {   // nothing clusterable — save the (empty) result so reloads skip the readback
        s_pageDir.clear();
        s_pageDir.push_back({ ClusterStream::kPagePinned, 0, 0, 0, 0 });   // identity page only
        SaveClusterCache(eligMeshes, outMeta, outIdx16, outIdx32, outVbBlob, outRanges, s_pageDir);
        return;
    }

    // ---- Cross-UNIT seam pins -----------------------------------------------
    // Vertices whose POSITION is shared between two COMPONENT units must not
    // move: those seams are welded surfaces the extent-cap split cut in half.
    // Only component↔component positions pin — pinning against AT/plain meshes
    // (Phase 2 behavior) hard-locked the ring around every window (glass pins
    // frame pins wall...) and froze 16% of component verts → stalled groups →
    // the DAG could never ascend to the levels where prune removes the frames.
    // Drift against unpinned neighbors is bounded by the error metric, exactly
    // like every other collapse (Nanite does no cross-object welding at all).
    constexpr u32 kShared = 0xFFFFFFFFu;
    auto posKey = [](const float* p) -> u64 {
        u64 h = 1469598103934665603ull;
        const u8* b = (const u8*)p;
        for (int i = 0; i < 12; ++i) { h ^= b[i]; h *= 1099511628211ull; }
        return h;
    };
    std::unordered_map<u64, u32> posOwner;
    posOwner.reserve(1u << 20);
    for (u32 m = 0; m < (u32)elig.size(); ++m) {
        if (unitOf[m] == kNoUnit || !units[unitOf[m]].comp) continue;   // only component units pin
        const Elig& e = elig[m];
        const u8* vb = (const u8*)e.pos;
        const u32 stride = e.fv->m_mesh.vStride;
        const u32 owner = unitOf[m];
        for (u32 v = 0; v < e.fv->m_mesh.vCount; ++v) {
            auto ins = posOwner.emplace(posKey((const float*)(vb + (size_t)v * stride)), owner);
            if (!ins.second && ins.first->second != owner) ins.first->second = kShared;
        }
    }

    // ---- clodBuild per unit ---------------------------------------------------
    // The hierarchy, boundary locking (position-welded — lmap-UV seam duplicates
    // count as one vertex), permissive simplification with UV-seam protect bits
    // and monotonic error propagation all come from clusterlod (the reference
    // Nanite-style builder shipped with meshopt). Units are independent, so the
    // build fans out over a thread pool, biggest units first (components dwarf
    // single meshes — the pool drains them while small units backfill).
    xr_vector<u32> buildOrder(units.size());
    for (u32 i = 0; i < (u32)buildOrder.size(); ++i) buildOrder[i] = i;
    std::sort(buildOrder.begin(), buildOrder.end(), [&](u32 a, u32 b) { return units[a].tris > units[b].tris; });

    // Per-thread outputs, merged (with offset fixups) after the join so the
    // final meta/IB layout is identical to a serial build.
    struct ThreadOut {
        xr_vector<GpuMeshMeta> meta;
        xr_vector<u16>         idx16;
        xr_vector<u32>         idx32;
        xr_vector<std::pair<vkFVisual*, MeshEntries>> ranges;
        u32 simplified = 0, stalled = 0;
        // Component diagnostics ("why doesn't the building LOD?"):
        u32 compBuilt = 0, compFlat = 0;          // units built / units whose DAG never left level 0
        u32 l0Groups = 0, l0Stalled = 0;          // level-0 groups: total / terminal (stuck at full detail)
        u64 compVerts = 0, lockVerts = 0, protVerts = 0;   // component verts: total / hard-locked / UV-protected
        u32 flatWorstTris = 0, flatWorstMeshes = 0;        // biggest flat unit (for the log)
        xr_vector<std::pair<u32, Fvector>> flatUnits;      // (tris, world centre) of every flat unit
    };
    const u32 nThreads = _min(8u, _max(1u, (u32)std::thread::hardware_concurrency()));
    xr_vector<ThreadOut> touts(nThreads);
    std::atomic<u32> cursor{ 0 };

    auto worker = [&](u32 t) {
        ThreadOut& out = touts[t];
        xr_vector<unsigned int>  idxIn;
        xr_vector<float>         attrs;    // 7 floats/vert: nx ny nz u0 v0 u1 v1
        xr_vector<unsigned char> vlock;
        xr_vector<float>         cpos;     // component: compact float3 positions
        xr_vector<u32>           meshOfV, globalOf;   // component: compact vert → member idx / pool-global id
        xr_vector<u64>           pkeys;    // component: compact vert → position hash
        xr_vector<std::pair<u32, u32>> bins;          // component emit scratch: (member, tri)
        std::unordered_map<u64, u32> canon;           // component: position hash → first compact vert
        xr_vector<clodBounds> groups;
        const float attrWeights[3] = { 0.5f, 0.5f, 0.5f };   // normals in the error metric; UVs protect-only

        // Attributes are only compared/weighted — absolute scale is irrelevant,
        // so raw SHORT2 UVs and unorm-byte normals go in undecoded-but-consistent.
        auto vertexAttrs = [](const u8* vp, u32 stride, u32 tcOffset, float* a) {
            a[0] = vp[12] / 255.f; a[1] = vp[13] / 255.f; a[2] = vp[14] / 255.f;   // packed normal
            const s16* uv0 = (const s16*)(vp + tcOffset);
            a[3] = (float)uv0[0]; a[4] = (float)uv0[1];
            if (tcOffset == 24 && stride >= 32) {   // lmap layout: 2nd UV @28
                const s16* uv1 = (const s16*)(vp + 28);
                a[5] = (float)uv1[0]; a[6] = (float)uv1[1];
            } else { a[5] = a[6] = 0.f; }
        };

        auto baseConfig = []() {
            clodConfig cfg = clodDefaultConfig(kClusterMaxTris);
            cfg.optimize_bounds = true;            // tight per-cluster cull spheres
            // NO sloppy fallback: vertex-clustering "simplification" is what folds
            // low-poly doors into triangles while under-reporting the error. Groups
            // it would have rescued now stay roots — full detail forever, no LOD,
            // but also no visible breakage. X-Ray content is low-poly; quality wins.
            cfg.simplify_fallback_sloppy = false;
            // PRUNE is what actually dissolves window frames/sills/trims: they are
            // separate SHELLS dropped into wall openings, not welded to the wall —
            // edge collapses can never merge disconnected shells, so those groups
            // stalled at full detail forever (health view: red frames everywhere).
            // Pruning removes a small shell outright once its size fits the error
            // budget, and that size lands in the reported error → the cut swaps to
            // the pruned parent exactly when the frame is ~subpixel.
            cfg.simplify_prune = true;
            return cfg;
        };

        // Shared cluster meta prototype. Parent test = the group this cluster is
        // a MEMBER of (drawn only while that group's simplified error projects
        // over the threshold); self test = the group whose simplification
        // PRODUCED this cluster.
        auto protoMeta = [&groups](const clodGroup& g, const clodCluster& c) {
            GpuMeshMeta e{};
            e.sphere_P.set(c.bounds.center[0], c.bounds.center[1], c.bounds.center[2]);
            e.sphere_R = c.bounds.radius;
            e.lodParent.set(g.simplified.center[0], g.simplified.center[1], g.simplified.center[2], g.simplified.radius);
            e.parentError = _min(g.simplified.error, kErrInf);
            if (c.refined >= 0) {
                const clodBounds& rb = groups[c.refined];
                e.lodSelf.set(rb.center[0], rb.center[1], rb.center[2], rb.radius);
                e.selfError = _min(rb.error, kErrInf);
            } else {
                e.lodSelf.set(c.bounds.center[0], c.bounds.center[1], c.bounds.center[2], c.bounds.radius);
                e.selfError = 0.f;   // original geometry
            }
            return e;
        };

        for (;;) {
            const u32 uIdx = cursor.fetch_add(1);
            if (uIdx >= (u32)buildOrder.size()) break;
            const Unit& un = units[buildOrder[uIdx]];
            groups.clear();

            if (!un.comp) {
                // ---- per-mesh DAG (Phase 2 path, u16 rel-vBase) ----
                const Elig& e = elig[un.members[0]];
                const auto& mesh = e.fv->m_mesh;

                idxIn.resize(e.idxCount);
                for (u32 i = 0; i < e.idxCount; ++i) idxIn[i] = e.idx[i];

                attrs.resize((size_t)mesh.vCount * 7);
                vlock.assign(mesh.vCount, 0);
                const u8* vb = (const u8*)e.pos;
                for (u32 v = 0; v < mesh.vCount; ++v) {
                    const u8* vp = vb + (size_t)v * mesh.vStride;
                    vertexAttrs(vp, mesh.vStride, mesh.tcOffset, &attrs[(size_t)v * 7]);
                    auto sh = posOwner.find(posKey((const float*)vp));   // read-only after the seam pass
                    if (sh != posOwner.end() && sh->second == kShared) vlock[v] = meshopt_SimplifyVertex_Lock;
                }

                clodConfig cfg = baseConfig();
                clodMesh cm{};
                cm.indices                 = idxIn.data();
                cm.index_count             = e.idxCount;
                cm.vertex_count            = mesh.vCount;
                cm.vertex_positions        = e.pos;
                cm.vertex_positions_stride = mesh.vStride;
                cm.vertex_attributes       = attrs.data();
                cm.vertex_attributes_stride = 7 * sizeof(float);
                cm.vertex_lock             = vlock.data();
                cm.attribute_weights       = attrWeights;
                cm.attribute_count         = 3;
                cm.attribute_protect_mask  = (1u << 3) | (1u << 4) | (1u << 5) | (1u << 6);   // both UV sets

                const u32    metaFirst = (u32)out.meta.size();
                const size_t idxFirst  = out.idx16.size();
                u32 sim = 0, stall = 0;
                clodBuild(cfg, cm, [&](clodGroup g, const clodCluster* cs, size_t n) -> int {
                    const int id = (int)groups.size();
                    groups.push_back(g.simplified);
                    if (g.simplified.error >= FLT_MAX) ++stall; else ++sim;
                    for (size_t i = 0; i < n; ++i) {
                        const clodCluster& c = cs[i];
                        GpuMeshMeta e2 = protoMeta(g, c);
                        e2.index_count  = (u32)c.index_count;
                        e2.ib_first     = (u32)out.idx16.size();
                        e2.first_vertex = mesh.vBase;
                        for (size_t j = 0; j < c.index_count; ++j) out.idx16.push_back((u16)c.indices[j]);
                        out.meta.push_back(e2);
                    }
                    return id;
                });
                const u32 emitted = (u32)out.meta.size() - metaFirst;
                if (emitted < 2) {   // degenerate (single cluster) — the plain path is equal & cheaper
                    out.meta.resize(metaFirst);
                    out.idx16.resize(idxFirst);
                    continue;
                }
                out.simplified += sim; out.stalled += stall;
                MeshEntries me{ false, {} };
                me.entries.reserve(emitted);
                for (u32 k = 0; k < emitted; ++k) me.entries.push_back(metaFirst + k);
                out.ranges.emplace_back(e.fv, std::move(me));
                continue;
            }

            // ---- component DAG (Phase 2.5, u32 pool-global) ----
            // Gather a COMPACT vertex/index set: clodBuild runs O(vertex_count)
            // passes per level, so feeding it whole-pool arrays would scale with
            // the pool, not the component. Compact verts remap back to
            // pool-global ids (globalOf) at emit time.
            const u32 nMembers = (u32)un.members.size();
            u32 totalV = 0, totalI = 0;
            for (u32 m : un.members) { totalV += elig[m].fv->m_mesh.vCount; totalI += elig[m].idxCount; }

            cpos.resize((size_t)totalV * 3);
            attrs.resize((size_t)totalV * 7);
            vlock.assign(totalV, 0);
            meshOfV.resize(totalV); globalOf.resize(totalV); pkeys.resize(totalV);
            idxIn.clear(); idxIn.reserve(totalI);

            u32 vCursor = 0;
            for (u32 mi = 0; mi < nMembers; ++mi) {
                const Elig& e = elig[un.members[mi]];
                const auto& mesh = e.fv->m_mesh;
                const u8* vb = (const u8*)e.pos;
                const u32 base = vCursor;
                for (u32 v = 0; v < mesh.vCount; ++v, ++vCursor) {
                    const u8* vp = vb + (size_t)v * mesh.vStride;
                    memcpy(&cpos[(size_t)vCursor * 3], vp, 12);
                    vertexAttrs(vp, mesh.vStride, mesh.tcOffset, &attrs[(size_t)vCursor * 7]);
                    meshOfV[vCursor]  = mi;
                    globalOf[vCursor] = mesh.vBase + v;
                    pkeys[vCursor]    = posKey((const float*)vp);
                }
                for (u32 i = 0; i < e.idxCount; ++i) idxIn.push_back(base + e.idx[i]);
            }

            // Locks & protects. Lock: position shared with another unit (the
            // neighbor won't move). Protect: LMAP-UV discontinuity against the
            // position-canonical vertex WITHIN THE SAME MESH — light seams are
            // the ones that flash visibly when collapsed. Diffuse-UV seams get
            // NO protect (unlike the per-mesh path): low-poly content has a UV
            // seam on nearly every corner, and protecting them froze 40% of
            // component verts → stalled DAGs. Their stretch is subpixel at the
            // distances where a coarse cut is chosen. Cross-mesh discontinuities
            // get no protect either — those are the material boundaries a door
            // must collapse across (the Nanite answer to material boundaries).
            canon.clear();
            canon.reserve(totalV);
            for (u32 v = 0; v < totalV; ++v) canon.emplace(pkeys[v], v);
            for (u32 v = 0; v < totalV; ++v) {
                auto sh = posOwner.find(pkeys[v]);
                if (sh != posOwner.end() && sh->second == kShared) vlock[v] |= meshopt_SimplifyVertex_Lock;
                const u32 c = canon.find(pkeys[v])->second;
                if (c != v && meshOfV[c] == meshOfV[v]) {
                    const float* av = &attrs[(size_t)v * 7];
                    const float* ac = &attrs[(size_t)c * 7];
                    if (av[5] != ac[5] || av[6] != ac[6]) vlock[v] |= meshopt_SimplifyVertex_Protect;   // lmap UV only
                }
            }
            for (u32 v = 0; v < totalV; ++v) {   // diag: how much of the component is frozen
                if (vlock[v] & meshopt_SimplifyVertex_Lock)    ++out.lockVerts;
                if (vlock[v] & meshopt_SimplifyVertex_Protect) ++out.protVerts;
            }
            out.compVerts += totalV;

            clodConfig cfg = baseConfig();
            clodMesh cm{};
            cm.indices                  = idxIn.data();
            cm.index_count              = idxIn.size();
            cm.vertex_count             = totalV;
            cm.vertex_positions         = cpos.data();
            cm.vertex_positions_stride  = 3 * sizeof(float);
            cm.vertex_attributes        = attrs.data();
            cm.vertex_attributes_stride = 7 * sizeof(float);
            cm.vertex_lock              = vlock.data();
            cm.attribute_weights        = attrWeights;
            cm.attribute_count          = 3;
            cm.attribute_protect_mask   = 0;   // protect bits self-computed above (intra-mesh only)

            const u32    metaFirst = (u32)out.meta.size();
            const size_t idxFirst  = out.idx32.size();
            xr_vector<xr_vector<u32>> memberEntries(nMembers);
            u32 clustersEmitted = 0;
            u32 sim = 0, stall = 0;
            int maxDepth = 0; u32 l0 = 0, l0s = 0;   // diag: did the DAG ever leave level 0?
            clodBuild(cfg, cm, [&](clodGroup g, const clodCluster* cs, size_t n) -> int {
                const int id = (int)groups.size();
                groups.push_back(g.simplified);
                if (g.simplified.error >= FLT_MAX) ++stall; else ++sim;
                maxDepth = _max(maxDepth, g.depth);
                if (g.depth == 0) { ++l0; if (g.simplified.error >= FLT_MAX) ++l0s; }
                for (size_t i = 0; i < n; ++i) {
                    const clodCluster& c = cs[i];
                    ++clustersEmitted;
                    const GpuMeshMeta proto = protoMeta(g, c);
                    // Bin the cluster's triangles by their v0's source mesh →
                    // one entry per (cluster × mesh), sharing the cluster's lod
                    // spheres/errors so the cull flips them as one.
                    bins.clear();
                    const size_t triCount = c.index_count / 3;
                    for (size_t t2 = 0; t2 < triCount; ++t2)
                        bins.emplace_back(meshOfV[c.indices[t2 * 3]], (u32)t2);
                    std::stable_sort(bins.begin(), bins.end(),
                                     [](const std::pair<u32, u32>& a, const std::pair<u32, u32>& b) { return a.first < b.first; });
                    for (size_t a2 = 0; a2 < bins.size();) {
                        size_t b2 = a2;
                        while (b2 < bins.size() && bins[b2].first == bins[a2].first) ++b2;
                        GpuMeshMeta e2 = proto;
                        e2.index_count  = (u32)((b2 - a2) * 3);
                        e2.ib_first     = (u32)out.idx32.size();
                        e2.first_vertex = 0;   // pool-global u32 indices
                        for (size_t k = a2; k < b2; ++k) {
                            const u32 t3 = bins[k].second;
                            out.idx32.push_back(globalOf[c.indices[t3 * 3 + 0]]);
                            out.idx32.push_back(globalOf[c.indices[t3 * 3 + 1]]);
                            out.idx32.push_back(globalOf[c.indices[t3 * 3 + 2]]);
                        }
                        memberEntries[bins[a2].first].push_back((u32)out.meta.size());
                        out.meta.push_back(e2);
                        a2 = b2;
                    }
                }
                return id;
            });
            if (clustersEmitted < 2) {   // whole component fits one cluster — the plain path is equal & cheaper
                out.meta.resize(metaFirst);
                out.idx32.resize(idxFirst);
                continue;
            }
            ++out.compBuilt;
            out.l0Groups += l0; out.l0Stalled += l0s;
            if (maxDepth == 0) {   // DAG never simplified anything — the unit stays full-detail forever
                ++out.compFlat;
                if (un.tris > out.flatWorstTris) { out.flatWorstTris = un.tris; out.flatWorstMeshes = nMembers; }
                Fbox fb; fb.invalidate();
                for (u32 m : un.members) fb.merge(elig[m].fv->vis.box);
                Fvector fc; fb.getcenter(fc);
                out.flatUnits.emplace_back(un.tris, fc);
            }
            out.simplified += sim; out.stalled += stall;
            for (u32 mi = 0; mi < nMembers; ++mi)
                if (!memberEntries[mi].empty())
                    out.ranges.emplace_back(elig[un.members[mi]].fv, MeshEntries{ true, std::move(memberEntries[mi]) });
        }
    };

    xr_vector<std::thread> pool;
    for (u32 t = 1; t < nThreads; ++t) pool.emplace_back(worker, t);
    worker(0);
    for (auto& th : pool) th.join();

    // Merge thread outputs: shift each thread's meta/IB offsets — the result is
    // layout-identical to a serial run. Every meta entry belongs to exactly one
    // mesh's entry list, so the list's ib32 flag routes its ib_first fixup.
    for (ThreadOut& out : touts) {
        const u32 metaBase  = (u32)outMeta.size();
        const u32 idx16Base = (u32)outIdx16.size();
        const u32 idx32Base = (u32)outIdx32.size();
        for (auto& r : out.ranges) {
            for (u32& e : r.second.entries) {
                out.meta[e].ib_first += r.second.ib32 ? idx32Base : idx16Base;
                e += metaBase;
            }
            outRanges.emplace(r.first, std::move(r.second));
        }
        outMeta.insert(outMeta.end(), out.meta.begin(), out.meta.end());
        outIdx16.insert(outIdx16.end(), out.idx16.begin(), out.idx16.end());
        outIdx32.insert(outIdx32.end(), out.idx32.begin(), out.idx32.end());
        s_statSimplified += out.simplified; s_statStalled += out.stalled;
    }

    {   // component diag: the answer to "why doesn't this building LOD?"
        u32 built = 0, flat = 0, l0g = 0, l0s = 0, worstT = 0, worstM = 0;
        u64 cv = 0, lv = 0, pv = 0;
        for (const ThreadOut& out : touts) {
            built += out.compBuilt; flat += out.compFlat;
            l0g += out.l0Groups; l0s += out.l0Stalled;
            cv += out.compVerts; lv += out.lockVerts; pv += out.protVerts;
            if (out.flatWorstTris > worstT) { worstT = out.flatWorstTris; worstM = out.flatWorstMeshes; }
        }
        if (built)
            Msg("[VK Cluster] comp diag: %u units (%u FLAT = never simplified; worst flat %u tris/%u meshes) | L0 groups %u, %u stalled (%.0f%%) | verts: %.1f%% locked, %.1f%% protected",
                built, flat, worstT, worstM, l0g, l0s, l0g ? 100.0 * l0s / l0g : 0.0,
                cv ? 100.0 * lv / cv : 0.0, cv ? 100.0 * pv / cv : 0.0);
        xr_vector<std::pair<u32, Fvector>> flats;
        for (const ThreadOut& out : touts) flats.insert(flats.end(), out.flatUnits.begin(), out.flatUnits.end());
        std::sort(flats.begin(), flats.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        for (u32 i = 0; i < _min<u32>(8, (u32)flats.size()); ++i)
            Msg("[VK Cluster]   flat unit %u: %u tris @ (%.0f, %.0f, %.0f)", i, flats[i].first,
                flats[i].second.x, flats[i].second.y, flats[i].second.z);
    }

    // ⭐ Bake hygiene (17-07): drop SELF-LOOP stall entries — clusters that
    // declare THEMSELVES their own parent ((lodSelf,selfError) bitwise equal to
    // (lodParent,parentError); clusterlod stall chains emit them). They can
    // never produce pixels: the hard cut needs sp<=1<pp (impossible with
    // sp==pp) and in the fade band their two dither masks cancel exactly. They
    // only cost meta slots, page bytes, cull invocations and VSM candidates —
    // and their severed child links were what left stall groups with no
    // refiner (the orphan-cohort/white-polygon saga; direct demand now covers
    // those regardless). The runtime self-loop cut in ClusterStream stays as a
    // safety net and should report 0 for caches baked with this filter.
    {
        const u32 n0 = (u32)outMeta.size();
        xr_vector<u8> drop(n0, 0);
        u32 nDrop = 0;
        for (u32 i = 0; i < n0; ++i) {
            const GpuMeshMeta& m = outMeta[i];
            if (m.selfError > 0.f && m.parentError < kErrInf
                && memcmp(&m.lodSelf, &m.lodParent, sizeof(Fvector4)) == 0
                && memcmp(&m.selfError, &m.parentError, sizeof(float)) == 0)
                { drop[i] = 1; ++nDrop; }
        }
        if (nDrop) {
            // Never empty a whole mesh (would silently unclusterize it): if every
            // entry of a mesh is a self-loop — shouldn't happen, its level-0
            // leaves carry distinct values — keep that mesh untouched.
            for (auto& r : outRanges) {
                u32 live = 0;
                for (u32 e : r.second.entries) if (e < n0 && !drop[e]) ++live;
                if (!live)
                    for (u32 e : r.second.entries) if (e < n0 && drop[e]) { drop[e] = 0; --nDrop; }
            }
        }
        if (nDrop) {
            xr_vector<u32> remap(n0, 0xFFFFFFFFu);
            u32 keep = 0;
            for (u32 i = 0; i < n0; ++i)
                if (!drop[i]) { remap[i] = keep; if (keep != i) outMeta[keep] = outMeta[i]; ++keep; }
            outMeta.resize(keep);
            for (auto& r : outRanges) {
                auto& es = r.second.entries;
                size_t w = 0;
                for (u32 e : es) if (e < n0 && remap[e] != 0xFFFFFFFFu) es[w++] = remap[e];
                es.resize(w);
            }
            Msg("[VK Cluster] bake hygiene: dropped %u self-loop stall entries (%u -> %u)", nDrop, n0, keep);
        }
    }

    {   // Stage B: pack the DAG into streaming pages (rewrites ib_first to
        // page-local + fills meta._pad1 with page ids, reorders the blobs).
        // Slice 2: hand the assembler each entry's vertex source (the host VB
        // pool copies read back above) so repackable meshes (stride ==
        // kVbStride) get their vertices repacked into per-page blocks too —
        // outVbBlob becomes the vertex-pool image (cached + uploaded).
        xr_vector<u8>  entryIb32(outMeta.size(), 0);
        xr_vector<u32> entryMesh(outMeta.size(), 0);
        xr_vector<ClusterStream::MeshVbInfo> meshInfo;
        xr_vector<ClusterStream::VbSource>   sources;
        xr_map<std::pair<VkBuffer, u32>, u32> srcOf;   // (pool VB, stride) -> source id
        for (const auto& kv : outRanges) {
            const auto& mesh = kv.first->m_mesh;
            const VkBuffer vbh = mesh.p_rm_Vertices->GetHandle();
            auto vit = vbData.find(vbh);
            u32 srcId = 0; u8 repack = 0;
            if (vit != vbData.end() && !vit->second.empty()) {
                auto ins = srcOf.emplace(std::make_pair(vbh, (u32)mesh.vStride), (u32)sources.size());
                if (ins.second) sources.push_back({ vit->second.data(), (u64)vit->second.size(), (u32)mesh.vStride });
                srcId = ins.first->second;
                repack = mesh.vStride == ClusterStream::kVbStride ? 1 : 0;
            }
            const u32 mid = (u32)meshInfo.size();
            meshInfo.push_back({ srcId, repack });
            for (u32 e : kv.second.entries) {
                if (kv.second.ib32) entryIb32[e] = 1;
                entryMesh[e] = mid;
            }
        }
        ClusterStream::AssemblePages(outMeta, entryIb32, entryMesh, meshInfo, sources,
                                     outIdx16, outIdx32, outVbBlob, s_pageDir);
    }
    SaveClusterCache(eligMeshes, outMeta, outIdx16, outIdx32, outVbBlob, outRanges, s_pageDir);
}


bool CreateCullPipeline()
{
    if (!g_ShaderManager) g_ShaderManager = xr_new<VK::CVulkanSPIRVLoader>();
    VkShaderModule cs = g_ShaderManager->Load("world_cull.comp.spv");
    if (!cs) { Msg("![VK WorldGPU] world_cull.comp.spv load failed"); return false; }

    // 0 meta, 1 cmds, 2 counts, 3 groupBase, + Stage B: 4 stream bits,
    // 5 page slot bases, 6 page requests (feedback), 7 touched-page bitset.
    constexpr u32 kB = 8;
    VkDescriptorSetLayoutBinding b[kB]{};
    for (u32 i = 0; i < kB; ++i) {
        b[i].binding = i; b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = kB; lci.pBindings = b;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &s_setL) != VK_SUCCESS) return false;

    VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kB };
    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets = 1; pci.poolSizeCount = 1; pci.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &s_pool) != VK_SUCCESS) return false;
    VkDescriptorSetAllocateInfo dai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dai.descriptorPool = s_pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &s_setL;
    if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, &s_set) != VK_SUCCESS) return false;
    VK::Prof::NameSet(s_set, "WorldGPU.Set");   // TEMP diag: VUID hunt

    VkDescriptorBufferInfo bi[kB] = {
        { s_meta->GetHandle(),                  0, VK_WHOLE_SIZE },
        { s_indirect->GetHandle(),              0, VK_WHOLE_SIZE },
        { s_count->GetHandle(),                 0, VK_WHOLE_SIZE },
        { s_groupBase->GetHandle(),             0, VK_WHOLE_SIZE },
        { ClusterStream::BitsBuffer(),          0, VK_WHOLE_SIZE },
        { ClusterStream::SlotBaseBuffer(),      0, VK_WHOLE_SIZE },
        { ClusterStream::RequestBuffer(),       0, VK_WHOLE_SIZE },
        { ClusterStream::TouchedBuffer(),       0, VK_WHOLE_SIZE },
    };
    VkWriteDescriptorSet w[kB]{};
    for (u32 i = 0; i < kB; ++i) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = s_set; w[i].dstBinding = i;
        w[i].descriptorCount = 1; w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(VulkanHW.m_Device, kB, w, 0, nullptr);

    VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CullPush) };
    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1; plci.pSetLayouts = &s_setL; plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_cullLayout) != VK_SUCCESS) return false;

    VkComputePipelineCreateInfo cp{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cp.stage.module = cs; cp.stage.pName = "main";
    cp.layout = s_cullLayout;
    if (vkCreateComputePipelines(VulkanHW.m_Device, VK_NULL_HANDLE, 1, &cp, nullptr, &s_cullPipe) != VK_SUCCESS) return false;
    return true;
}

// Shadow cull pipeline (world_cull_shadow.comp): 4 SSBOs — meta (0), the
// per-target shadow indirect (1), per-target counts (2), groupBase (3).
bool CreateShadowCullPipeline()
{
    if (!g_ShaderManager) g_ShaderManager = xr_new<VK::CVulkanSPIRVLoader>();
    VkShaderModule cs = g_ShaderManager->Load("world_cull_shadow.comp.spv");
    if (!cs) { Msg("![VK WorldGPU] world_cull_shadow.comp.spv load failed"); return false; }

    // 0 meta, 1 per-target indirect, 2 per-target counts, 3 groupBase,
    // + Stage B: 4 stream bits, 5 page slot bases (no requests — main view asks).
    constexpr u32 kB3 = 6;
    VkDescriptorSetLayoutBinding b[kB3]{};
    for (u32 i = 0; i < kB3; ++i) {
        b[i].binding = i; b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = kB3; lci.pBindings = b;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &s_setL3) != VK_SUCCESS) return false;

    VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kB3 };
    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets = 1; pci.poolSizeCount = 1; pci.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &s_pool3) != VK_SUCCESS) return false;
    VkDescriptorSetAllocateInfo dai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dai.descriptorPool = s_pool3; dai.descriptorSetCount = 1; dai.pSetLayouts = &s_setL3;
    if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, &s_set3) != VK_SUCCESS) return false;

    VkDescriptorBufferInfo bi[kB3] = {
        { s_meta->GetHandle(),             0, VK_WHOLE_SIZE },
        { s_indirectSh->GetHandle(),       0, VK_WHOLE_SIZE },
        { s_countSh->GetHandle(),          0, VK_WHOLE_SIZE },
        { s_groupBase->GetHandle(),        0, VK_WHOLE_SIZE },
        { ClusterStream::BitsBuffer(),     0, VK_WHOLE_SIZE },
        { ClusterStream::SlotBaseBuffer(), 0, VK_WHOLE_SIZE },
    };
    VkWriteDescriptorSet w[kB3]{};
    for (u32 i = 0; i < kB3; ++i) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = s_set3; w[i].dstBinding = i;
        w[i].descriptorCount = 1; w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(VulkanHW.m_Device, kB3, w, 0, nullptr);

    VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CullShadowPush) };
    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1; plci.pSetLayouts = &s_setL3; plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_cullLayout3) != VK_SUCCESS) return false;

    VkComputePipelineCreateInfo cp{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cp.stage.module = cs; cp.stage.pName = "main";
    cp.layout = s_cullLayout3;
    if (vkCreateComputePipelines(VulkanHW.m_Device, VK_NULL_HANDLE, 1, &cp, nullptr, &s_cullPipe3) != VK_SUCCESS) return false;
    return true;
}

// Hi-Z occlusion cull pipeline (world_cull_hzb.comp). 4 bindings: meta (0),
// cmds2 (1), count2 (2) — written once here; HZB combined-image-sampler (3) —
// (re)written per-frame in CullColor since the depth pyramid view/sampler can
// change on resize. Best-effort: failure leaves s_occlReady=false (the base
// frustum path keeps working; the color pass just won't occlusion-cull).
bool CreateCullColorPipeline()
{
    if (!g_ShaderManager) g_ShaderManager = xr_new<VK::CVulkanSPIRVLoader>();
    VkShaderModule cs = g_ShaderManager->Load("world_cull_hzb.comp.spv");
    if (!cs) { Msg("![VK WorldGPU] world_cull_hzb.comp.spv load failed — occlusion cull disabled"); return false; }

    // 0 meta, 1 cmds2, 2 count2, 3 HZB (per-frame), 4 groupBase,
    // + Stage B: 5 stream bits, 6 page slot bases.
    constexpr u32 kB2 = 7;
    VkDescriptorSetLayoutBinding b[kB2]{};
    for (u32 i = 0; i < kB2; ++i) {
        b[i].binding = i; b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    b[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;   // HZB
    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = kB2; lci.pBindings = b;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &s_setL2) != VK_SUCCESS) return false;

    VkDescriptorPoolSize ps[2] = { { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kB2 - 1 }, { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 } };
    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets = 1; pci.poolSizeCount = 2; pci.pPoolSizes = ps;
    if (vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &s_pool2) != VK_SUCCESS) return false;
    VkDescriptorSetAllocateInfo dai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dai.descriptorPool = s_pool2; dai.descriptorSetCount = 1; dai.pSetLayouts = &s_setL2;
    if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, &s_set2) != VK_SUCCESS) return false;
    VK::Prof::NameSet(s_set2, "WorldGPU.Set2");   // TEMP diag: VUID hunt

    const u32 dstBind[6] = { 0, 1, 2, 4, 5, 6 };   // 3 = HZB, written per-frame
    VkDescriptorBufferInfo bi[6] = {
        { s_meta->GetHandle(),             0, VK_WHOLE_SIZE },
        { s_indirect2->GetHandle(),        0, VK_WHOLE_SIZE },
        { s_count2->GetHandle(),           0, VK_WHOLE_SIZE },
        { s_groupBase->GetHandle(),        0, VK_WHOLE_SIZE },
        { ClusterStream::BitsBuffer(),     0, VK_WHOLE_SIZE },
        { ClusterStream::SlotBaseBuffer(), 0, VK_WHOLE_SIZE },
    };
    VkWriteDescriptorSet w[6]{};
    for (u32 i = 0; i < 6; ++i) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = s_set2;
        w[i].dstBinding = dstBind[i];
        w[i].descriptorCount = 1; w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(VulkanHW.m_Device, 6, w, 0, nullptr);   // binding 3 (HZB) written per-frame

    VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CullColorPush) };
    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1; plci.pSetLayouts = &s_setL2; plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_cullLayout2) != VK_SUCCESS) return false;

    VkComputePipelineCreateInfo cp{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cp.stage.module = cs; cp.stage.pName = "main";
    cp.layout = s_cullLayout2;
    if (vkCreateComputePipelines(VulkanHW.m_Device, VK_NULL_HANDLE, 1, &cp, nullptr, &s_cullPipe2) != VK_SUCCESS) return false;
    return true;
}

} // anonymous namespace

void Build()
{
    VK::Vram::Scope _vram_scope("WorldGPU");
    if (s_built) return;
    s_built = true;   // one attempt; stays "built" (possibly empty) so we don't retry every frame
    ++s_buildStamp;   // consumers holding derived tables (VSM candidates) rebuild on stamp change
    s_compactRefs.clear();
    s_poolsCompacted = false;

    xr_vector<vkFVisual*> meshes; meshes.reserve(16384);
    xr_vector<vkFVisual*> cpu;    cpu.reserve(8192);
    for (IRenderVisual* iv : RImplementation.Visuals)
        ExtractMeshes(static_cast<vkRender_Visual*>(iv), meshes, cpu);
    std::sort(meshes.begin(), meshes.end());
    meshes.erase(std::unique(meshes.begin(), meshes.end()), meshes.end());
    // CPU (non-GPU) static leaves — pre-built + deduped so the per-frame path walks
    // only these (small) instead of all level visuals + hierarchy recursion.
    std::sort(cpu.begin(), cpu.end());
    cpu.erase(std::unique(cpu.begin(), cpu.end()), cpu.end());
    s_cpuMeshes.swap(cpu);

    if (meshes.empty()) { Msg("[VK WorldGPU] no eligible static meshes (cpu-set=%u)", (u32)s_cpuMeshes.size()); return; }

    // Pointer-sorted membership set (meshes is currently sorted by pointer +
    // deduped) for InSet() — the CPU queue excludes these to avoid double-draw.
    s_meshSet.assign(meshes.begin(), meshes.end());

    // Cluster split (r_cluster): build the per-cluster meta pool + shared
    // cluster IBs BEFORE grouping — clustered meshes bind s_clusterIB (u16,
    // per-mesh DAGs) or s_clusterIB32 (u32 pool-global, component DAGs) instead
    // of their pool IB, so the grouping key must see the effective IB.
    xr_vector<GpuMeshMeta> clusterMeta;
    xr_vector<u16>         clusterIdx16;
    xr_vector<u32>         clusterIdx32;
    xr_vector<u8>          clusterVbBlob;
    xr_map<vkFVisual*, MeshEntries> clustered;
    u64 clusterMs = 0;
    if (ps_r_cluster) {
        LARGE_INTEGER f, t0, t1; QueryPerformanceFrequency(&f);
        QueryPerformanceCounter(&t0);
        BuildClusters(meshes, clusterMeta, clusterIdx16, clusterIdx32, clusterVbBlob, clustered);
        QueryPerformanceCounter(&t1);
        clusterMs = (u64)((t1.QuadPart - t0.QuadPart) * 1000 / f.QuadPart);
    }
    // Stage B pools (phase 1): the cluster index data now lives in ClusterStream
    // page pools — pinned pages install here; the rest streams by GPU feedback
    // (or everything, when r_clpage 0 / no cache file = pre-streaming parity).
    // Slice 2: repacked cluster vertices live in the ClusterStream vertex pool.
    if (!ClusterStream::CreatePools(s_pageDir, s_cachePath, s_blobOff16, s_blobOff32, s_blobOffVb,
                                    &clusterIdx16, &clusterIdx32, &clusterVbBlob)) {
        Msg("![VK Cluster] page pools failed — clusters disabled (plain meshes only)");
        clustered.clear(); clusterMeta.clear(); s_pageDir.clear();
        ClusterStream::CreatePools(s_pageDir, nullptr, 0, 0, 0, nullptr, nullptr, nullptr);   // identity-only state
    }
    {   // A referenced pool must exist — otherwise nothing may reference it.
        bool need16 = false, need32 = false;
        for (const auto& kv : clustered) (kv.second.ib32 ? need32 : need16) = true;
        if ((need16 && ClusterStream::PoolIB16() == VK_NULL_HANDLE)
         || (need32 && ClusterStream::PoolIB32() == VK_NULL_HANDLE)) {
            Msg("![VK Cluster] pool missing for referenced index type — clusters disabled");
            clustered.clear();
        }
    }
    auto effectiveIB = [&](vkFVisual* a) -> VkBuffer {
        auto it = clustered.find(a);
        if (it == clustered.end()) return a->m_mesh.p_rm_Indices->GetHandle();
        return it->second.ib32 ? ClusterStream::PoolIB32() : ClusterStream::PoolIB16();
    };
    // Slice 2: a clustered mesh whose entries were vertex-repacked (its first
    // entry's page carries a VB payload — the flag is uniform per mesh by
    // construction) draws from the ClusterStream vertex pool; everything else
    // keeps its original pool VB. Same layout/stride — only the binding moves.
    auto effectiveVB = [&](vkFVisual* a) -> VkBuffer {
        auto it = clustered.find(a);
        if (it != clustered.end() && !it->second.entries.empty()) {
            const u32 pg = clusterMeta[it->second.entries[0]]._pad1;
            if (pg < (u32)s_pageDir.size() && (s_pageDir[pg].flags & ClusterStream::kPageHasVB)
                && ClusterStream::PoolVB() != VK_NULL_HANDLE)
                return ClusterStream::PoolVB();
        }
        return a->m_mesh.p_rm_Vertices->GetHandle();
    };

    // Sort into draw batches: terrain first-class, then by (material, stride,
    // tcOffset, effective vb, effective ib) so a run shares pipeline + descriptor
    // set + buffer binds. Clustered meshes separate naturally (their ib/vb =
    // the ClusterStream pools).
    auto key = [&](vkFVisual* a) {
        return std::make_tuple(a->m_pWorldMaterial->isTerrain ? 0 : 1,
                               (const void*)a->m_pWorldMaterial,
                               a->m_mesh.vStride, a->m_mesh.tcOffset,
                               (const void*)effectiveVB(a),
                               (const void*)effectiveIB(a));
    };
    std::sort(meshes.begin(), meshes.end(), [&](vkFVisual* a, vkFVisual* b) { return key(a) < key(b); });

    const u32 nMeshes = (u32)meshes.size();

    // DIAG (one-shot): SWI LOD-level distribution of the GPU-set progressive meshes.
    // Tells us if there are INTERMEDIATE LODs (sw_count>2) to grade through (smooth,
    // no popping) or only fine+coarse, and the finest->coarsest triangle reduction
    // (= the LOD savings headroom). Drives whether main-view LOD is graduated or binary.
    {
        u32 progN = 0, multiN = 0, swMin = 0xFFFFFFFFu, swMax = 0; u64 swSum = 0;
        u64 fineTris = 0, coarseTris = 0; u32 hist[9] = { 0 };   // buckets sw_count = 1,2,..,8,9+
        for (vkFVisual* fv : meshes) {
            if (fv->Type != MT_PROGRESSIVE) continue;
            auto* pg = static_cast<vkFProgressive*>(fv);
            if (!pg->sw_counts || pg->sw_count == 0) continue;
            ++progN;
            const u32 c = pg->sw_count;
            swMin = _min(swMin, c); swMax = _max(swMax, c); swSum += c;
            hist[_min(c, 9u) - 1]++;
            if (c > 1) { ++multiN; fineTris += pg->sw_counts[0] / 3; coarseTris += pg->sw_counts[c - 1] / 3; }
        }
        if (progN) {
            Msg("[VK WorldGPU] SWI-LOD: %u progressive (%u multi-level), sw_count min/avg/max = %u/%.1f/%u",
                progN, multiN, swMin, double(swSum) / progN, swMax);
            Msg("[VK WorldGPU] SWI-LOD hist[1..9+] = %u %u %u %u %u %u %u %u %u",
                hist[0], hist[1], hist[2], hist[3], hist[4], hist[5], hist[6], hist[7], hist[8]);
            if (multiN) Msg("[VK WorldGPU] SWI-LOD tris: finest avg=%llu coarsest avg=%llu (%.0f%% fewer)",
                (unsigned long long)(fineTris / multiN), (unsigned long long)(coarseTris / multiN),
                100.0 * (1.0 - double(coarseTris) / double(_max((u64)1, fineTris))));
        } else {
            Msg("[VK WorldGPU] SWI-LOD: no progressive meshes in the GPU set");
        }
    }

    // Grouping walk: one pass over the sorted meshes builds the groups AND the
    // meta array in entry order — an entry is either the whole mesh (drawn slice
    // = finest SWI window for PROGRESSIVE, see DrawnSlice: the coarse slices poke
    // through the fine mesh) or, for clustered meshes, each prebuilt cluster.
    xr_vector<GpuMeshMeta> meta;
    meta.reserve(nMeshes + clusterMeta.size());

    auto appendEntries = [&](vkFVisual* fv) {
        // Hard-cut flag: alpha-tested materials draw the prepass through the AT
        // pipeline, which has no dither — their transitions stay instant. Set
        // here (not at cluster build) so the disk cache stays material-agnostic.
        const u32 flags = (fv->m_pWorldMaterial->alphaRef >= 0.f) ? 1u : 0u;
        auto it = clustered.find(fv);
        // Pool-compaction bookkeeping: remember this mesh's meta run + how its
        // entries reference the original pools (CompactPools patches by these).
        CompactRef ref{};
        ref.fv        = fv;
        ref.metaFirst = (u32)meta.size();
        ref.clustered = (it != clustered.end()) ? 1 : 0;
        ref.ib32      = (ref.clustered && it->second.ib32) ? 1 : 0;
        ref.repacked  = (ref.clustered && effectiveVB(fv) == ClusterStream::PoolVB()
                         && ClusterStream::PoolVB() != VK_NULL_HANDLE) ? 1 : 0;
        if (it != clustered.end()) {
            for (u32 ei : it->second.entries) {
                GpuMeshMeta e = clusterMeta[ei];
                e.flags = flags;
                meta.push_back(e);
            }
            ref.metaCount = (u32)meta.size() - ref.metaFirst;
            s_compactRefs.push_back(ref);
            return;
        }
        GpuMeshMeta m{};
        m.sphere_P = fv->vis.sphere.P;
        m.sphere_R = fv->vis.sphere.R;
        DrawnSlice(fv, m.ib_first, m.index_count);
        m.first_vertex = fv->m_mesh.vBase;
        m.group        = 0;   // assigned after grouping
        m.flags        = flags;
        m.lodSelf.set(m.sphere_P.x, m.sphere_P.y, m.sphere_P.z, m.sphere_R);
        m.selfError    = 0.f;        // plain mesh: exact geometry ...
        m.parentError  = kErrInf;    // ... with no coarser parent → always drawn
        meta.push_back(m);
        ref.metaCount = 1;
        s_compactRefs.push_back(ref);
    };

    s_groups.clear();
    Group cur{};
    auto seed = [&](u32 i) {
        vkFVisual* fv = meshes[i];
        cur.mat      = fv->m_pWorldMaterial;
        cur.terrain  = fv->m_pWorldMaterial->isTerrain;
        cur.stride   = fv->m_mesh.vStride;
        cur.tcOffset = fv->m_mesh.tcOffset;
        cur.vb       = effectiveVB(fv);
        cur.ib       = effectiveIB(fv);
        auto itc = clustered.find(fv);
        cur.iType    = itc == clustered.end() ? fv->m_mesh.iType
                     : (itc->second.ib32 ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16);
        cur.meshCount   = 1;
        cur.entryOffset = (u32)meta.size();
        cur.entryCount  = 0;   // finalized when the group closes
        // Composite cache (r_terra_cache): capture the terrain material + mesh
        // for the bake (GPU-driven path bypasses the CPU render queue). The
        // probe needs a CONTIGUOUS vertex run at vBase — that only exists in
        // the ORIGINAL pool VB (cur.vb is the page-repacked ClusterStream pool
        // for clustered terrain, where vBase addresses someone else's pages).
        if (cur.terrain && cur.mat->terrainSet != VK_NULL_HANDLE) {
            TerrainCache::OnTerrainMaterial(cur.mat->terrainSet, cur.mat->detailScale);
            TerrainCache::OnTerrainMesh(fv->m_mesh.p_rm_Vertices->GetHandle(),
                                        fv->m_mesh.vBase, cur.stride, cur.tcOffset);
        }
    };
    seed(0); appendEntries(meshes[0]);
    for (u32 i = 1; i < nMeshes; ++i) {
        vkFVisual* fv = meshes[i];
        const bool same = fv->m_pWorldMaterial == cur.mat
            && fv->m_mesh.vStride == cur.stride && fv->m_mesh.tcOffset == cur.tcOffset
            && effectiveVB(fv) == cur.vb
            && effectiveIB(fv) == cur.ib;
        if (same) cur.meshCount++;
        else { cur.entryCount = (u32)meta.size() - cur.entryOffset; s_groups.push_back(cur); seed(i); }
        appendEntries(fv);
    }
    cur.entryCount = (u32)meta.size() - cur.entryOffset;
    s_groups.push_back(cur);

    s_total = (u32)meta.size();   // dispatch size = all cullable entries
    const u32 nGroups = (u32)s_groups.size();
    for (u32 g = 0; g < nGroups; ++g)
        for (u32 e = s_groups[g].entryOffset; e < s_groups[g].entryOffset + s_groups[g].entryCount; ++e)
            meta[e].group = g;

    {   // Slice-2 VB-BINDING VALIDATOR (bisected bug: occasional white stretched
        // polygons / missing walls). The cull adds the page VB base to first_vertex
        // unconditionally, and the group binds ONE VB. So every entry's page MUST
        // agree with its group's VB: a kPageHasVB page ⟺ the group binds PoolVB
        // (repacked, page-local first_vertex). A mismatch means the entry fetches
        // from the WRONG buffer → garbage vertices. effectiveVB decides per-mesh
        // from entries[0] only; this catches any mesh whose entries disagree.
        const VkBuffer poolVB = ClusterStream::PoolVB();
        u32 nMis = 0, nFv = 0, shown = 0;
        for (u32 g = 0; g < nGroups; ++g) {
            const bool bindsPool = (poolVB != VK_NULL_HANDLE) && (s_groups[g].vb == poolVB);
            for (u32 e = s_groups[g].entryOffset; e < s_groups[g].entryOffset + s_groups[g].entryCount; ++e) {
                const u32 pg = meta[e]._pad1;
                const bool hasVB = pg < (u32)s_pageDir.size()
                                && (s_pageDir[pg].flags & ClusterStream::kPageHasVB) != 0;
                // (A) group VB binding must match the entry's page repack class.
                if (hasVB != bindsPool) {
                    ++nMis;
                    if (shown < 12) { ++shown;
                        Msg("![VK ClVB-MISMATCH] entry %u group %u page %u: pageHasVB=%d groupBindsPool=%d groupMeshes=%u first_vertex=%u mat=%s",
                            e, g, pg, hasVB ? 1 : 0, bindsPool ? 1 : 0,
                            s_groups[g].meshCount, meta[e].first_vertex,
                            (s_groups[g].mat && s_groups[g].mat->name.c_str()) ? s_groups[g].mat->name.c_str() : "?");
                    }
                }
                // (B) a repacked entry (its page carries VB) MUST have first_vertex 0
                //     (page-local; the cull adds the page VB base). Non-zero = the
                //     assembler left an absolute offset → base+abs overshoots.
                if (hasVB && meta[e].first_vertex != 0u) {
                    ++nFv;
                    if (shown < 12) { ++shown;
                        Msg("![VK ClVB-FIRSTVERT] entry %u group %u page %u: kPageHasVB but first_vertex=%u (should be 0)",
                            e, g, pg, meta[e].first_vertex);
                    }
                }
            }
        }
        Msg("[VK ClVB] validator: %u entries, %u groups, PoolVB=%s | mismatches: binding=%u firstVertex=%u",
            s_total, nGroups, poolVB != VK_NULL_HANDLE ? "yes" : "no", nMis, nFv);
        if (nMis || nFv) Msg("![VK ClVB] ^^ THIS is the white-polygon / missing-wall bug");
    }

    {   // LOD-cut data sanity: parentError histogram (meters) + parent sphere radii.
        // An entry can only ever be REPLACED by its parent if parentError is finite
        // and the camera can get beyond the parent sphere (dist - r > 0).
        u32 hInf = 0, h100 = 0, h10 = 0, h1 = 0, h01 = 0, hSm = 0, level0 = 0;
        float rMax = 0.f; double rSum = 0.0;
        for (const GpuMeshMeta& m : meta) {
            if (m.selfError == 0.f) ++level0;
            if      (m.parentError >= kErrInf) ++hInf;
            else if (m.parentError > 100.f)    ++h100;
            else if (m.parentError > 10.f)     ++h10;
            else if (m.parentError > 1.f)      ++h1;
            else if (m.parentError > 0.1f)     ++h01;
            else                               ++hSm;
            if (m.parentError < kErrInf) { rMax = _max(rMax, m.lodParent.w); rSum += m.lodParent.w; }
        }
        const u32 fin = s_total - hInf;
        Msg("[VK Cluster] cut diag: %u entries (%u level-0) | parentError: INF=%u >100m=%u 10-100=%u 1-10=%u 0.1-1=%u <0.1=%u | parent sphere r avg=%.1f max=%.1f",
            s_total, level0, hInf, h100, h10, h1, h01, hSm, fin ? rSum / fin : 0.0, rMax);

        // Cut COMPLEMENTARITY invariant (Nanite: "parents are all dependent"): a
        // child hides exactly when its (lodParent, parentError) projects small,
        // trusting that some entry with a BITWISE-equal (lodSelf, selfError)
        // appears at that same threshold. If no such entry exists (emit bug,
        // cache corruption, future refactor), the child vanishes into a hole.
        // Note: a parent entry existing for ANOTHER mesh of the same cluster is
        // fine (that's how pruned shells dissolve) — the check is per-group, not
        // per-mesh. Runs on both the fresh-build and cache-load paths.
        {
            auto key = [](const Fvector4& s, float err) {
                u64 h = 1469598103934665603ull;
                auto mix = [&h](u32 v) { h = (h ^ v) * 1099511628211ull; };
                mix(*(const u32*)&s.x); mix(*(const u32*)&s.y); mix(*(const u32*)&s.z); mix(*(const u32*)&s.w);
                mix(*(const u32*)&err);
                return h;
            };
            xr_set<u64> selfKeys;
            for (const GpuMeshMeta& m : meta)
                if (m.selfError > 0.f) selfKeys.insert(key(m.lodSelf, m.selfError));
            u32 holes = 0;
            for (const GpuMeshMeta& m : meta)
                if (m.parentError < kErrInf && selfKeys.find(key(m.lodParent, m.parentError)) == selfKeys.end()) ++holes;
            if (holes)
                // Legal source: a group fully pruned to nothing (lone-shell units
                // dissolve into nothing by design) — expect a SMALL, stable count.
                // A jump after a builder/cache change = bitwise mismatch = popping
                // holes at the swap distance; bisect the change.
                Msg("[VK Cluster] cut invariant: %u/%u entries dissolve to nothing (no parent-level entry)", holes, s_total);
        }
    }

    s_metaCPU = meta;   // host copy: the stats tick simulates the cut CPU-side (diag)

    // Stage B state (phase 2): group/liveness tables + residency bits + feedback
    // buffers over the FINAL entry array. The cull shaders bind these — failure
    // means the new cull pipelines can't run, so the whole GPU path steps aside
    // (CPU queue draws everything, like a cull-pipeline failure).
    if (!ClusterStream::BuildState(meta)) {
        Msg("![VK WorldGPU] cluster-stream state failed — GPU world path disabled");
        s_groups.clear();
        return;
    }

    // meta SSBO (device-local)
    s_meta = xr_new<CVulkanBuffer>();
    s_meta->Create(sizeof(GpuMeshMeta) * s_total, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                   VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    s_meta->Upload(meta.data(), sizeof(GpuMeshMeta) * s_total);

    // Per-group cmd-region bases = entryOffset (exact prefix-sum layout: the
    // indirect buffer holds exactly s_total cmds; region g = [base[g], +entryCount)).
    {
        xr_vector<u32> bases(nGroups);
        for (u32 g = 0; g < nGroups; ++g) bases[g] = s_groups[g].entryOffset;
        s_groupBase = xr_new<CVulkanBuffer>();
        s_groupBase->Create(sizeof(u32) * nGroups, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
        s_groupBase->Upload(bases.data(), sizeof(u32) * nGroups);
    }

    // TRANSFER_SRC: the draw-command audit copies cmds+counts back for verification.
    const VkBufferUsageFlags iu = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    s_indirect = xr_new<CVulkanBuffer>();
    s_indirect->Create((VkDeviceSize)s_total * sizeof(VkDrawIndexedIndirectCommand), iu,
                       VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    s_count = xr_new<CVulkanBuffer>();
    s_count->Create((VkDeviceSize)nGroups * sizeof(u32), iu, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);

    if (!CreateCullPipeline()) { Msg("![VK WorldGPU] cull pipeline failed — disabled"); s_groups.clear(); return; }

    // Phase A: a SECOND indirect/count set + Hi-Z cull pipeline for r_hzb_cull.
    // Best-effort — if it fails, the base frustum path is unaffected (s_occlReady
    // stays false → CullColor no-ops, DrawColor uses the frustum set).
    s_indirect2 = xr_new<CVulkanBuffer>();
    s_indirect2->Create((VkDeviceSize)s_total * sizeof(VkDrawIndexedIndirectCommand), iu,
                        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    s_count2 = xr_new<CVulkanBuffer>();
    s_count2->Create((VkDeviceSize)nGroups * sizeof(u32), iu, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    s_occlReady = CreateCullColorPipeline();
    if (!s_occlReady) Msg("![VK WorldGPU] Hi-Z occlusion cull unavailable (r_hzb_cull will be a no-op)");

    // Phase 3: shadow-caster indirect/count slices (far + 2 cascades) + the
    // ortho-cut cull pipeline. Best-effort — failure keeps the per-mesh
    // ShadowGPU fallback (vk_pass_shadow gates on ShadowReady()).
    s_indirectSh = xr_new<CVulkanBuffer>();
    s_indirectSh->Create((VkDeviceSize)kShadowTargets * s_total * sizeof(VkDrawIndexedIndirectCommand), iu,
                         VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    s_countSh = xr_new<CVulkanBuffer>();
    s_countSh->Create((VkDeviceSize)kShadowTargets * nGroups * sizeof(u32), iu, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    {   // zero-init: a draw can hit a target slice before its first cull (live
        // r_shadow_cluster toggle, VSM-active fog reuse) — count 0 draws nothing.
        xr_vector<u32> zeros(kShadowTargets * nGroups, 0u);
        s_countSh->Upload(zeros.data(), zeros.size() * sizeof(u32));
    }
    s_shadowReady = CreateShadowCullPipeline();
    if (!s_shadowReady) Msg("![VK WorldGPU] shadow cluster cull unavailable (r_shadow_cluster will be a no-op)");

    // Host-visible readback for the occlusion-stats log (frustum vs occlusion counts).
    s_nGroups = nGroups;
    if (s_occlReady) {
        VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bci.size = (VkDeviceSize)2 * nGroups * sizeof(u32);
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo ai{};
        if (VK::Vram::CreateBuffer(VulkanHW.m_Allocator, &bci, &aci, &s_countReadback, &s_countReadbackAlloc, &ai) == VK_SUCCESS)
            s_countReadbackPtr = (u32*)ai.pMappedData;
    }

    if (ps_r_cl_audit) {
        // Draw-command audit readback: 2 × (cmds + counts) — frustum + HZB sets. Non-fatal.
        VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bci.size = 2 * ((VkDeviceSize)s_total * sizeof(VkDrawIndexedIndirectCommand) + (VkDeviceSize)nGroups * sizeof(u32));
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo ai{};
        if (VK::Vram::CreateBuffer(VulkanHW.m_Allocator, &bci, &aci, &s_cmdAudit, &s_cmdAuditAlloc, &ai) == VK_SUCCESS && ai.pMappedData)
            s_cmdAuditPtr = (u8*)ai.pMappedData;
        else
            Msg("![VK WorldGPU] cmd-audit buffer alloc failed (%llu MB) — draw audit disabled",
                (unsigned long long)(bci.size >> 20));
    }

    Msg("[VK WorldGPU] built: %u GPU meshes -> %u cull entries, %u groups | cpu-set=%u (non-GPU static leaves) | occlusion=%d",
        nMeshes, s_total, nGroups, (u32)s_cpuMeshes.size(), s_occlReady ? 1 : 0);
    if (!clustered.empty()) {
        u32 mergedMeshes = 0;
        for (const auto& kv : clustered) if (kv.second.ib32) ++mergedMeshes;
        u64 ibBytes16 = 0, ibBytes32 = 0;   // from the page dir (blobs may live on disk)
        for (const auto& pr : s_pageDir) ((pr.flags & ClusterStream::kPageIB32) ? ibBytes32 : ibBytes16) += pr.byteSize;
        Msg("[VK Cluster] %u meshes -> %u entries (IB %.1f MB u16 + %.1f MB u32 in %u pages) in %llu ms | DAG groups: %u simplified, %u stalled",
            (u32)clustered.size(), (u32)clusterMeta.size(),
            double(ibBytes16) / (1024.0 * 1024.0),
            double(ibBytes32) / (1024.0 * 1024.0),
            (u32)_max<size_t>(1, s_pageDir.size()) - 1, (unsigned long long)clusterMs,
            s_statSimplified, s_statStalled);
        if (mergedMeshes)
            Msg("[VK Cluster] merge: %u meshes in component DAGs%s", mergedMeshes,
                s_statCompUnits ? "" : " (from cache)");
        if (s_statCompUnits)
            Msg("[VK Cluster] merge: %u components baked from %u meshes", s_statCompUnits, s_statCompMeshes);
    }
    else if (ps_r_cluster)
        Msg("[VK Cluster] no meshes eligible for cluster split (r_cluster_tris=%d)", ps_r_cluster_tris);
}

u32 SubmitCpuMeshes(VK::RenderQueue& q, const Fmatrix& viewProj, bool doCull)
{
    if (s_cpuMeshes.empty()) return 0;
    Fvector4 planes[6];
    if (doCull) VK::ExtractFrustumPlanes(viewProj, planes);   // same 6-plane test as the GPU cull
    Fmatrix identity; identity.identity();
    u32 n = 0;
    for (vkFVisual* fv : s_cpuMeshes) {
        if (doCull) {
            const Fvector& c = fv->vis.sphere.P; const float r = fv->vis.sphere.R;
            if (r > 0.f) {
                bool outside = false;
                for (int i = 0; i < 6; ++i)
                    if (planes[i].x*c.x + planes[i].y*c.y + planes[i].z*c.z + planes[i].w < -r) { outside = true; break; }
                if (outside) continue;
            }
        }
        fv->Submit(q, identity, 0.0f);   // leaf → one DrawItem (no hierarchy recursion, no dups)
        ++n;
    }
    return n;
}

bool Built() { return s_built && !s_groups.empty() && s_cullPipe != VK_NULL_HANDLE; }

bool InSet(vkRender_Visual* v) { return std::binary_search(s_meshSet.begin(), s_meshSet.end(), v); }
u32  SetSize() { return (u32)s_meshSet.size(); }

// Verify the draw-command copy recorded a FIF cycle ago (see the state block
// above). Runs on the render thread from Cull() — pure CPU.
static void CmdAuditVerify()
{
    if (!s_cmdAuditPending || Device.dwFrame < s_cmdAuditFrame + CVulkanCommandManager::FRAMES_IN_FLIGHT + 1) return;
    s_cmdAuditPending = false;
    s_cmdAuditLast = Device.dwTimeGlobal;
    if (s_metaCPU.size() != s_total || !s_cmdAuditPtr) return;
    const u32 nGroups = (u32)s_groups.size();
    vmaInvalidateAllocation(VulkanHW.m_Allocator, s_cmdAuditAlloc, 0, VK_WHOLE_SIZE);
    const VkDrawIndexedIndirectCommand* cmds = (const VkDrawIndexedIndirectCommand*)s_cmdAuditPtr;
    const u32* counts = (const u32*)(s_cmdAuditPtr + (size_t)s_total * sizeof(VkDrawIndexedIndirectCommand));
    const u32* sb  = s_cmdAuditSlotBase.empty() ? nullptr : s_cmdAuditSlotBase.data();
    const u32 sbN  = (u32)s_cmdAuditSlotBase.size();
    const u32* bits = s_cmdAuditBits.empty() ? nullptr : s_cmdAuditBits.data();

    // ---- Pass 1: every recorded draw vs the host expectation ----------------
    // Region 1 = frustum set (depth prepass); region 2 = HZB set (what the
    // COLOR pass consumes with r_hzb_cull 1), captured the same frame.
    const bool hasOccl = s_cmdAuditHasOccl;
    s_cmdAuditHasOccl = false;
    xr_vector<u32> drawnMask((s_total + 31) / 32, 0u);
    xr_vector<u32> drawnMask2(hasOccl ? (s_total + 31) / 32 : 0, 0u);
    u32 total = 0, total2 = 0, overflow = 0, badId = 0, route = 0, args = 0, notDrawable = 0, shown = 0;
    auto decodeSet = [&](const VkDrawIndexedIndirectCommand* cs, const u32* cn, xr_vector<u32>& mask, u32& tot, const char* tag) {
        for (u32 g = 0; g < nGroups; ++g) {
            const Group& grp = s_groups[g];
            u32 n = cn[g];
            if (n > grp.entryCount) {
                ++overflow;
                Msg("![VK CmdAudit] %s group %u (%s) count %u > region %u — REGION OVERFLOW", tag, g, GroupMatName(g), n, grp.entryCount);
                n = grp.entryCount;
            }
            for (u32 i = 0; i < n; ++i) {
                const VkDrawIndexedIndirectCommand& c = cs[grp.entryOffset + i];
                const u32 e = c.firstInstance & 0xFFFFFu;
                ++tot;
                if (e >= s_total) {
                    ++badId;
                    if (shown++ < 12) Msg("![VK CmdAudit] %s group %u cmd %u: entry id %u out of range", tag, g, i, e);
                    continue;
                }
                mask[e >> 5u] |= 1u << (e & 31u);
                const GpuMeshMeta& m = s_metaCPU[e];
                if (m.group != g) {
                    ++route;
                    if (shown++ < 12) Msg("![VK CmdAudit] %s entry %u drawn in group %u (%s) but meta.group=%u — ROUTING", tag, e, g, GroupMatName(g), m.group);
                }
                const u32 p = m._pad1;
                u32 wantFI = m.ib_first, wantVO = m.first_vertex;
                if (sb && 2 * p + 1 < sbN) { wantFI += sb[2 * p]; wantVO += sb[2 * p + 1]; }
                if (c.indexCount != m.index_count || c.firstIndex != wantFI || c.vertexOffset != (int32_t)wantVO || c.instanceCount != 1u) {
                    ++args;
                    if (shown++ < 12)
                        Msg("![VK CmdAudit] %s entry %u page %u ARGS: idx %u/%u fi %u/%u vo %d/%u inst %u (grp %u %s, sphere %.0f,%.0f,%.0f)",
                            tag, e, p, c.indexCount, m.index_count, c.firstIndex, wantFI, c.vertexOffset, wantVO, c.instanceCount,
                            g, GroupMatName(g), m.sphere_P.x, m.sphere_P.y, m.sphere_P.z);
                }
                if (bits && !((bits[e >> 4u] >> ((e & 15u) * 2u)) & 1u)) {
                    ++notDrawable;
                    if (shown++ < 12) Msg("![VK CmdAudit] %s entry %u page %u drawn but NOT drawable in the snapshot bits", tag, e, p);
                }
            }
        }
    };
    decodeSet(cmds, counts, drawnMask, total, "frustum");
    if (hasOccl) {
        const u8* r2 = s_cmdAuditPtr + (size_t)s_total * sizeof(VkDrawIndexedIndirectCommand) + (size_t)nGroups * sizeof(u32);
        decodeSet((const VkDrawIndexedIndirectCommand*)r2,
                  (const u32*)(r2 + (size_t)s_total * sizeof(VkDrawIndexedIndirectCommand)),
                  drawnMask2, total2, "HZB");
    }

    // ---- Pass 2: near-camera coverage (the shader's cut replayed on the CPU) ----
    // Margins keep float noise out: report a miss only when the entry is CLEARLY
    // inside the cut, an extra only when clearly outside it.
    u32 nearMiss = 0, nearExtra = 0, nearShown = 0, nearDrawn = 0, nearHzbCulled = 0, hzbShown = 0;
    u32 nearSleaf = 0, sleafShown = 0, nearDead = 0, deadShown = 0;
    const float R = 25.f;
    const Fvector cam = s_cmdAuditCam, dir = s_cmdAuditDir;
    const Fvector4 lp = s_cmdAuditLod;
    const float band = lp.z;
    for (u32 e = 0; e < s_total; ++e) {
        const GpuMeshMeta& m = s_metaCPU[e];
        const float dx = m.sphere_P.x - cam.x, dy = m.sphere_P.y - cam.y, dz = m.sphere_P.z - cam.z;
        const float dist = sqrtf(dx * dx + dy * dy + dz * dz) - m.sphere_R;
        if (dist > R) continue;
        bool out = false;
        for (int i = 0; i < 6 && !out; ++i)
            out = s_cmdAuditPlanes[i].x * m.sphere_P.x + s_cmdAuditPlanes[i].y * m.sphere_P.y
                + s_cmdAuditPlanes[i].z * m.sphere_P.z + s_cmdAuditPlanes[i].w < -m.sphere_R;
        if (out) continue;
        const u32 sb2 = bits ? (bits[e >> 4u] >> ((e & 15u) * 2u)) & 3u : 1u;
        const bool drawableB = (sb2 & 1u) != 0, leaf = (sb2 & 2u) != 0;
        const float dsd = _max(lp.y, dir.dotproduct(Fvector{ m.lodSelf.x - cam.x, m.lodSelf.y - cam.y, m.lodSelf.z - cam.z }) - m.lodSelf.w);
        const float dpd = _max(lp.y, dir.dotproduct(Fvector{ m.lodParent.x - cam.x, m.lodParent.y - cam.y, m.lodParent.z - cam.z }) - m.lodParent.w);
        const float sp = m.selfError * lp.x / dsd;
        const float pp = _min(m.parentError, kErrInf) * lp.x / dpd;
        const bool drawn = (drawnMask[e >> 5u] >> (e & 31u)) & 1u;
        if (drawn) ++nearDrawn;
        // ---- HZB cross-diff: in the depth prepass but NOT in the color set =
        // depth written, color skipped → hole with the background showing.
        if (hasOccl && drawn && !((drawnMask2[e >> 5u] >> (e & 31u)) & 1u)) {
            ++nearHzbCulled;
            if (hzbShown++ < 12)
                Msg("![VK CmdAudit] HZB-CULLED near entry %u page %u R=%.1fm bits=%u d=%.1f grp=%u %s sphere=(%.0f,%.0f,%.0f)",
                    e, m._pad1, m.sphere_R, sb2, dist, m.group, GroupMatName(m.group),
                    m.sphere_P.x, m.sphere_P.y, m.sphere_P.z);
        }
        // A near entry drawn AS A STREAMING LEAF = a coarser cluster covering
        // children that never arrived. Persisting while standing still, this is
        // the hole's mechanism — dump WHY its child cohort is broken.
        if (drawn && leaf) {
            ++nearSleaf;
            if (sleafShown++ < 4) {
                Msg("![VK CmdAudit] NEAR-SLEAF entry %u page %u sp=%.2f pp=%.2f d=%.1f grp=%u %s sphere=(%.0f,%.0f,%.0f)",
                    e, m._pad1, sp, pp, dist, m.group, GroupMatName(m.group), m.sphere_P.x, m.sphere_P.y, m.sphere_P.z);
                ClusterStream::DebugEntryChain(e);
            }
        }
        // A near entry the cut NEEDS (parent too coarse) but that can't draw
        // (cohort dead) = the region only a leaf ancestor can cover — the wall.
        if (!drawableB && pp > 1.1f) {
            ++nearDead;
            if (deadShown++ < 4) {
                Msg("![VK CmdAudit] NEAR-DEAD entry %u page %u bits=%u sp=%.2f pp=%.2f d=%.1f grp=%u %s sphere=(%.0f,%.0f,%.0f)",
                    e, m._pad1, sb2, sp, pp, dist, m.group, GroupMatName(m.group), m.sphere_P.x, m.sphere_P.y, m.sphere_P.z);
                ClusterStream::DebugEntryChain(e);
            }
        }
        const bool wantStrict = drawableB && pp > 1.1f && (sp < 0.9f || leaf);
        const bool wantLoose  = drawableB && pp > 0.95f && (sp < 1.05f + band || leaf);
        if (wantStrict && !drawn) {
            ++nearMiss;
            u32 res, slot, vbSlot; ClusterStream::DebugPageInfo(m._pad1, res, slot, vbSlot);
            if (nearShown++ < 12)
                Msg("![VK CmdAudit] NEAR-MISS entry %u page %u (res=%u slot=%d vb=%d) bits=%u sp=%.2f pp=%.2f d=%.1f grp=%u %s sphere=(%.0f,%.0f,%.0f)",
                    e, m._pad1, res, (int)slot, (int)vbSlot, sb2, sp, pp, dist, m.group, GroupMatName(m.group),
                    m.sphere_P.x, m.sphere_P.y, m.sphere_P.z);
        } else if (drawn && !wantLoose) {
            ++nearExtra;
            if (nearShown++ < 12)
                Msg("![VK CmdAudit] NEAR-EXTRA entry %u page %u bits=%u sp=%.2f pp=%.2f d=%.1f grp=%u %s sphere=(%.0f,%.0f,%.0f)",
                    e, m._pad1, sb2, sp, pp, dist, m.group, GroupMatName(m.group),
                    m.sphere_P.x, m.sphere_P.y, m.sphere_P.z);
        }
    }
    Msg("[VK CmdAudit] frustum %u / hzb %u draws (occl=%d) | overflow=%u badId=%u route=%u args=%u notDrawable=%u | near %.0fm: drawn=%u miss=%u extra=%u hzbCulled=%u sleaf=%u dead=%u | cam=(%.0f,%.0f,%.0f)",
        total, total2, hasOccl ? 1 : 0, overflow, badId, route, args, notDrawable, R, nearDrawn, nearMiss, nearExtra, nearHzbCulled, nearSleaf, nearDead, cam.x, cam.y, cam.z);
}

void Cull(VkCommandBuffer cmd, const Fmatrix& viewProj)
{
    Cull(cmd, viewProj, Device.vCameraPosition, Device.vCameraDirection);
}

void Cull(VkCommandBuffer cmd, const Fmatrix& viewProj, const Fvector& camPos, const Fvector& viewDir)
{
    if (!Built()) return;
    const u32 nGroups = (u32)s_groups.size();

    CmdAuditVerify();   // check the copy recorded a FIF cycle ago (throttled)

    Fvector4 planes[6];
    VK::ExtractFrustumPlanes(viewProj, planes);

    // WAR-guard vs the previous frame's indirect reads (TRANSFER src stage: the
    // cmd-audit and count-stats copies also read these buffers).
    MemBarrier(cmd, 0, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
               VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    vkCmdFillBuffer(cmd, s_count->GetHandle(), 0, (VkDeviceSize)nGroups * sizeof(u32), 0u);
    MemBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_cullPipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_cullLayout, 0, 1, &s_set, 0, nullptr);
    CullPush pc{};
    for (int p = 0; p < 6; ++p) pc.planes[p] = planes[p];
    pc.cameraPos.set(camPos.x, camPos.y, camPos.z, 0.f);
    pc.viewDir.set(viewDir.x, viewDir.y, viewDir.z, 0.f);
    pc.lodParams = LodParams();
    pc.numGroups = nGroups; pc.unused0 = 0; pc.total = s_total;
    vkCmdPushConstants(cmd, s_cullLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (s_total + 255) / 256, 1, 1);

    // ---- Draw-command audit: snapshot THIS dispatch's output + inputs -------
    if (s_cmdAuditPtr && !s_cmdAuditPending && Device.dwTimeGlobal > s_cmdAuditLast + 3000) {
        MemBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferCopy cc{ 0, 0, (VkDeviceSize)s_total * sizeof(VkDrawIndexedIndirectCommand) };
        vkCmdCopyBuffer(cmd, s_indirect->GetHandle(), s_cmdAudit, 1, &cc);
        VkBufferCopy cn{ 0, (VkDeviceSize)s_total * sizeof(VkDrawIndexedIndirectCommand), (VkDeviceSize)nGroups * sizeof(u32) };
        vkCmdCopyBuffer(cmd, s_count->GetHandle(), s_cmdAudit, 1, &cn);
        s_cmdAuditCam = camPos; s_cmdAuditDir = viewDir; s_cmdAuditLod = pc.lodParams;
        for (int i = 0; i < 6; ++i) s_cmdAuditPlanes[i] = planes[i];
        const u32* hb = ClusterStream::HostBits();
        if (hb) s_cmdAuditBits.assign(hb, hb + (s_total + 15) / 16);
        else s_cmdAuditBits.clear();
        u32 sbN = 0; const u32* hsb = ClusterStream::HostSlotBase(sbN);
        if (hsb) s_cmdAuditSlotBase.assign(hsb, hsb + sbN);
        else s_cmdAuditSlotBase.clear();
        s_cmdAuditFrame = Device.dwFrame;
        s_cmdAuditPending = true;
    }

    MemBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT);
}

bool OcclusionReady() { return s_occlReady; }

void CullColor(VkCommandBuffer cmd, const Fmatrix& viewProj, const Fvector& cameraPos,
               VkImageView hzbView, VkSampler hzbSampler)
{
    if (!Built() || !s_occlReady) return;
    if (hzbView == VK_NULL_HANDLE || hzbSampler == VK_NULL_HANDLE) return;
    const u32 nGroups = (u32)s_groups.size();

    // Occlusion-stats log (throttled): sum the readback copied a frame or two ago.
    // frustum = sum(s_count) = frustum-visible; occl = sum(s_count2) = also Hi-Z
    // visible; the difference is what occlusion removed from the heavy color pass.
    // Verbose-only (r_profiler 2 / cluster debug view on) — it's 2 lines every 3s.
    if (s_countReadbackPtr && (ps_r_profiler > 1 || ps_r_cluster_debug > 0)) {
        static u32 s_statLast = 0;
        if (Device.dwTimeGlobal > s_statLast + 3000) {
            s_statLast = Device.dwTimeGlobal;
            u32 frust = 0, occl = 0;
            for (u32 g = 0; g < s_nGroups; ++g) { frust += s_countReadbackPtr[g]; occl += s_countReadbackPtr[s_nGroups + g]; }
            Msg("[VK WorldGPU] occlusion: frustum=%u -> visible=%u (culled %u | %u total meshes)",
                frust, occl, frust >= occl ? frust - occl : 0u, s_total);

            // CPU replica of the shader's DAG cut (no frustum) — diverging from
            // the GPU counts pins a bug to the runtime side; matching counts pin
            // it to the DAG data. Also splits what the cut keeps at this camera.
            if (!s_metaCPU.empty()) {
                const Fvector4 lp = LodParams();
                const Fvector cam = Device.vCameraPosition;
                const Fvector dir = Device.vCameraDirection;   // view-Z metric — must match the shaders
                const u32* sbits = ClusterStream::HostBits();  // Stage B residency (must match the GPU test)
                u32 drawn = 0, drawnComp = 0, drawnInf = 0, drawnL0 = 0, drawnLeaf = 0;
                for (u32 ei = 0; ei < (u32)s_metaCPU.size(); ++ei) {
                    const GpuMeshMeta& m = s_metaCPU[ei];
                    bool leaf = false;
                    if (sbits) {
                        const u32 sb = (sbits[ei >> 4u] >> ((ei & 15u) * 2u)) & 3u;
                        if (!(sb & 1u)) continue;   // page/cohort not drawable
                        leaf = (sb & 2u) != 0u;
                    }
                    const float dsd = _max(lp.y, dir.dotproduct(Fvector{ m.lodSelf.x - cam.x, m.lodSelf.y - cam.y, m.lodSelf.z - cam.z }) - m.lodSelf.w);
                    const float dpd = _max(lp.y, dir.dotproduct(Fvector{ m.lodParent.x - cam.x, m.lodParent.y - cam.y, m.lodParent.z - cam.z }) - m.lodParent.w);
                    const float sp = m.selfError * lp.x / dsd;
                    const float pp = _min(m.parentError, kErrInf) * lp.x / dpd;
                    if ((sp <= 1.f || leaf) && pp > 1.f) {
                        ++drawn;
                        if (sp > 1.f) ++drawnLeaf;
                        if (m._pad1 != 0) ++drawnComp;   // paged (clustered) entries — first_vertex went page-local in slice 2
                        if (m.parentError >= kErrInf) ++drawnInf;
                        if (m.selfError == 0.f) ++drawnL0;
                    }
                }
                Msg("[VK Cluster] cpu-cut: drawn=%u (paged=%u, level0=%u, infParent=%u, streamLeaf=%u) lodx=%.1f cam=(%.0f,%.0f,%.0f)",
                    drawn, drawnComp, drawnL0, drawnInf, drawnLeaf, lp.x, cam.x, cam.y, cam.z);
            }
        }
    }

    // Point binding 3 at this frame's HZB (view/sampler can change on resize).
    VkDescriptorImageInfo hi{};
    hi.sampler     = hzbSampler;
    hi.imageView   = hzbView;
    hi.imageLayout = VK_IMAGE_LAYOUT_GENERAL;   // HZB lives in GENERAL (see CreateHZB)
    VkWriteDescriptorSet hw{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    hw.dstSet = s_set2; hw.dstBinding = 3; hw.descriptorCount = 1;
    hw.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; hw.pImageInfo = &hi;
    vkUpdateDescriptorSets(VulkanHW.m_Device, 1, &hw, 0, nullptr);

    // WAR-guard vs the previous frame's indirect reads of cmds2/count2
    // (TRANSFER src stage: the cmd-audit copy also reads them).
    MemBarrier(cmd, 0, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
               VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    vkCmdFillBuffer(cmd, s_count2->GetHandle(), 0, (VkDeviceSize)nGroups * sizeof(u32), 0u);
    MemBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_cullPipe2);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_cullLayout2, 0, 1, &s_set2, 0, nullptr);
    CullColorPush pc{};
    pc.viewProj = viewProj;
    VK::ExtractFrustumPlanes(viewProj, pc.planes);   // same normalized planes as the frustum cull
    pc.cameraPos.set(cameraPos.x, cameraPos.y, cameraPos.z, 0.f);
    pc.viewDir.set(Device.vCameraDirection.x, Device.vCameraDirection.y, Device.vCameraDirection.z, 0.f);
    pc.lodParams = LodParams();
    pc.numGroups = nGroups; pc.unused0 = 0; pc.total = s_total;
    vkCmdPushConstants(cmd, s_cullLayout2, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (s_total + 255) / 256, 1, 1);

    // Snapshot both count sets → host readback (for the throttled stats log above,
    // read next frame). Global compute-write→transfer-read barrier covers s_count
    // (frustum, written earlier) and s_count2 (just dispatched).
    if (s_countReadback) {
        MemBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferCopy cA{ 0, 0, (VkDeviceSize)nGroups * sizeof(u32) };
        vkCmdCopyBuffer(cmd, s_count->GetHandle(),  s_countReadback, 1, &cA);
        VkBufferCopy cB{ 0, (VkDeviceSize)nGroups * sizeof(u32), (VkDeviceSize)nGroups * sizeof(u32) };
        vkCmdCopyBuffer(cmd, s_count2->GetHandle(), s_countReadback, 1, &cB);
    }

    // ---- Draw-command audit: capture the HZB set of the SAME frame Cull()
    // armed (region 2). This is what the color pass actually consumes.
    if (s_cmdAuditPtr && s_cmdAuditPending && s_cmdAuditFrame == Device.dwFrame) {
        MemBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        const VkDeviceSize base2 = (VkDeviceSize)s_total * sizeof(VkDrawIndexedIndirectCommand) + (VkDeviceSize)nGroups * sizeof(u32);
        VkBufferCopy cc{ 0, base2, (VkDeviceSize)s_total * sizeof(VkDrawIndexedIndirectCommand) };
        vkCmdCopyBuffer(cmd, s_indirect2->GetHandle(), s_cmdAudit, 1, &cc);
        VkBufferCopy cn{ 0, base2 + (VkDeviceSize)s_total * sizeof(VkDrawIndexedIndirectCommand), (VkDeviceSize)nGroups * sizeof(u32) };
        vkCmdCopyBuffer(cmd, s_count2->GetHandle(), s_cmdAudit, 1, &cn);
        s_cmdAuditHasOccl = true;
    }

    MemBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT);
}

void DrawDepth(VkCommandBuffer cmd, const Fmatrix& viewProj, bool displaceTerrain)
{
    if (!Built()) return;
    VkPipelineLayout layoutSolid = PipelineCache::GetDepthLayout();
    VkPipelineLayout layoutAT    = PipelineCache::GetDepthATLayout();
    if (layoutSolid == VK_NULL_HANDLE) return;
    const u32 nGroups = (u32)s_groups.size();

    struct ATPush { Fmatrix mvp; float uvScale[2]; float aref; float _pad; };   // matches FlushDepth
    Fmatrix vp = viewProj;

    VkPipeline       lastPipe   = VK_NULL_HANDLE;
    VkPipelineLayout lastLayout = VK_NULL_HANDLE;
    VkDescriptorSet  lastMatSet = VK_NULL_HANDLE;
    VkBuffer    lastVB = VK_NULL_HANDLE, lastIB = VK_NULL_HANDLE;
    VkIndexType lastIType = VK_INDEX_TYPE_MAX_ENUM;

    for (u32 g = 0; g < nGroups; ++g) {
        const Group& grp = s_groups[g];
        WorldMaterial* mat = grp.mat;

        // Terrain in the DEPTH PREPASS: snow-displaced terrain depth pipeline (same
        // world_terrain.vert as color -> matching displacement -> no z-fight). Set 1
        // = EnvLight (sf_params.w). Only in the prepass (displaceTerrain); shadows
        // pass false. Terrain groups sort first (group key isTerrain?0:1).
        if (displaceTerrain && mat->isTerrain && mat->terrainSet != VK_NULL_HANDLE) {
            VkPipeline       tpipe = PipelineCache::GetTerrainDepthPipeline();
            VkPipelineLayout tlay  = PipelineCache::GetTerrainLayout();
            VkDescriptorSet  eset  = EnvLight::GetCurrentSet();
            if (tpipe == VK_NULL_HANDLE || tlay == VK_NULL_HANDLE || eset == VK_NULL_HANDLE) continue;
            if (tlay != lastLayout) { lastLayout = tlay; lastMatSet = VK_NULL_HANDLE; lastPipe = VK_NULL_HANDLE; lastVB = VK_NULL_HANDLE; lastIB = VK_NULL_HANDLE; }
            if (tpipe != lastPipe)  { vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, tpipe); lastPipe = tpipe; lastVB = VK_NULL_HANDLE; lastIB = VK_NULL_HANDLE; }
            if (eset != lastMatSet) { vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, tlay, 1, 1, &eset, 0, nullptr); lastMatSet = eset; }   // set 1 = EnvLight
            // Set 0 = terrain material: the tessellation eval samples uMask (soil
            // softness for the mud carve) — unbound set 0 here was a device-lost.
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, tlay, 0, 1, &mat->terrainSet, 0, nullptr);
            struct TPush { Fmatrix mvp; float uv[2]; float aref; float ds; } tp{}; tp.mvp = vp;
            // uvScale must match the color pass (1/1024): the TES samples uMask at vUV
            // for the mud carve — zero uvScale = different carve = prepass z-fight.
            tp.uv[0] = tp.uv[1] = 1.0f / 1024.0f;
            tp.ds = mat->detailScale;
            vkCmdPushConstants(cmd, tlay, PipelineCache::GetPushStages(), 0, sizeof(TPush), &tp);
            if (grp.vb != lastVB) { VkDeviceSize z = 0; vkCmdBindVertexBuffers(cmd, 0, 1, &grp.vb, &z); lastVB = grp.vb; }
            if (grp.ib != lastIB || grp.iType != lastIType) { vkCmdBindIndexBuffer(cmd, grp.ib, 0, grp.iType); lastIB = grp.ib; lastIType = grp.iType; }
            const VkDeviceSize cmdOffT = (VkDeviceSize)grp.entryOffset * sizeof(VkDrawIndexedIndirectCommand);
            const VkDeviceSize cntOffT = (VkDeviceSize)g * sizeof(u32);
            vkCmdDrawIndexedIndirectCount(cmd, s_indirect->GetHandle(), cmdOffT, s_count->GetHandle(), cntOffT,
                                          grp.entryCount, sizeof(VkDrawIndexedIndirectCommand));
            continue;
        }

        const bool at = mat->alphaRef >= 0.f;
        if (at && (layoutAT == VK_NULL_HANDLE || mat->set == VK_NULL_HANDLE)) continue;  // mirror FlushDepth skip

        // Crossfade (r_cluster_fade): solid statics swap to the dithering depth
        // variant so prepass coverage matches the color pass exactly. AT keeps
        // its own pipeline — AT entries are hard-cut in the cull (meta flag).
        const bool fade = ps_r_cluster_fade > 0.f;
        VkPipeline pipe = at ? PipelineCache::GetDepthATPipeline(grp.stride, grp.tcOffset)
                             : (fade ? PipelineCache::GetDepthFadePipeline(grp.stride) : VK_NULL_HANDLE);
        if (!at && pipe == VK_NULL_HANDLE) pipe = PipelineCache::GetDepthPipeline(grp.stride);
        if (pipe == VK_NULL_HANDLE) continue;
        VkPipelineLayout layout = at ? layoutAT : layoutSolid;

        if (layout != lastLayout) { lastLayout = layout; lastMatSet = VK_NULL_HANDLE; lastPipe = VK_NULL_HANDLE; lastVB = VK_NULL_HANDLE; lastIB = VK_NULL_HANDLE; }
        if (pipe != lastPipe) { vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe); lastPipe = pipe; lastVB = VK_NULL_HANDLE; lastIB = VK_NULL_HANDLE; }

        if (at) {
            if (mat->set != lastMatSet) { vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &mat->set, 0, nullptr); lastMatSet = mat->set; }
            ATPush push{}; push.mvp = vp; push.uvScale[0] = 1.0f/1024.0f; push.uvScale[1] = 1.0f/1024.0f; push.aref = mat->alphaRef;
            vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
        } else {
            vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Fmatrix), &vp);
        }

        if (grp.vb != lastVB) { VkDeviceSize z = 0; vkCmdBindVertexBuffers(cmd, 0, 1, &grp.vb, &z); lastVB = grp.vb; }
        if (grp.ib != lastIB || grp.iType != lastIType) { vkCmdBindIndexBuffer(cmd, grp.ib, 0, grp.iType); lastIB = grp.ib; lastIType = grp.iType; }
        const VkDeviceSize cmdOff = (VkDeviceSize)grp.entryOffset * sizeof(VkDrawIndexedIndirectCommand);
        const VkDeviceSize cntOff = (VkDeviceSize)g * sizeof(u32);
        vkCmdDrawIndexedIndirectCount(cmd, s_indirect->GetHandle(), cmdOff, s_count->GetHandle(), cntOff,
                                      grp.entryCount, sizeof(VkDrawIndexedIndirectCommand));
    }
}

// ---- Phase 3: cluster-LOD shadow casters ----------------------------------

VkBuffer MetaBuffer()      { return s_meta      ? s_meta->GetHandle()      : VK_NULL_HANDLE; }
VkBuffer GroupBaseBuffer() { return s_groupBase ? s_groupBase->GetHandle() : VK_NULL_HANDLE; }
u32      EntryCount()      { return s_total; }
u32      GroupCount()      { return (u32)s_groups.size(); }
const xr_vector<GpuMeshMeta>& HostMeta() { return s_metaCPU; }
u32      BuildStamp()      { return s_buildStamp; }

bool GetGroupShadow(u32 g, VkBuffer& vb, VkBuffer& ib, u32& stride, VkIndexType& iType,
                    u32& entryOffset, u32& entryCount, bool& alphaTested,
                    const WorldMaterial** matOut, u32* tcOffsetOut)
{
    if (g >= s_groups.size()) return false;
    const Group& grp = s_groups[g];
    vb = grp.vb; ib = grp.ib; stride = grp.stride; iType = grp.iType;
    entryOffset = grp.entryOffset; entryCount = grp.entryCount;
    alphaTested = grp.mat && grp.mat->alphaRef >= 0.f;
    if (matOut)      *matOut      = grp.mat;
    if (tcOffsetOut) *tcOffsetOut = grp.tcOffset;
    return true;
}

bool ShadowReady() { return Built() && s_shadowReady; }

// GPU AT casters: live cvar + the AT depth layout must exist. CullShadow and
// DrawShadow both read this inside one frame's command record (console vars
// apply between frames), so the culled set and the drawn groups always agree.
bool ShadowATActive() { return ps_r_gpu_shadows_at && PipelineCache::GetDepthATLayout() != VK_NULL_HANDLE; }

void CullShadow(VkCommandBuffer cmd, const u32* targets, const Fmatrix* viewProjs,
                const float* texelWorld, u32 n)
{
    if (!ShadowReady() || n == 0) return;
    const u32 nGroups = (u32)s_groups.size();
    const float k = _max(0.05f, ps_r_vsm_cluster_lod);

    // WAR-guard vs the previous frame's indirect reads of the shadow slices.
    MemBarrier(cmd, 0, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
               VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    for (u32 i = 0; i < n; ++i)
        vkCmdFillBuffer(cmd, s_countSh->GetHandle(), (VkDeviceSize)targets[i] * nGroups * sizeof(u32),
                        (VkDeviceSize)nGroups * sizeof(u32), 0u);
    MemBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_cullPipe3);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_cullLayout3, 0, 1, &s_set3, 0, nullptr);
    for (u32 i = 0; i < n; ++i) {
        CullShadowPush pc{};
        VK::ExtractFrustumPlanes(viewProjs[i], pc.planes);
        pc.errBudget = texelWorld[i] * k;
        pc.target    = targets[i];
        pc.numGroups = nGroups;
        pc.total     = s_total;
        pc.atOn      = ShadowATActive() ? 1u : 0u;
        vkCmdPushConstants(cmd, s_cullLayout3, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, (s_total + 255) / 256, 1, 1);
    }
    MemBarrier(cmd, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT);
}

void DrawShadow(VkCommandBuffer cmd, u32 target, const Fmatrix& lightVP, bool skipTerrain)
{
    if (!ShadowReady()) return;
    VkPipelineLayout layoutSolid = PipelineCache::GetDepthLayout();
    VkPipelineLayout layoutAT    = PipelineCache::GetDepthATLayout();
    if (layoutSolid == VK_NULL_HANDLE) return;
    const bool atOn = ShadowATActive();
    const u32 nGroups = (u32)s_groups.size();

    // Depth-only, statics are world-space → the same lightVP for every group.
    // AT groups draw with the discard pipeline + their material's diffuse
    // bound (mirrors DrawDepth); with atOn off their regions hold zero
    // commands (the cull skips flags bit0) — skip their binds. Terrain draws
    // through the plain per-stride depth pipeline (no displacement in shadow
    // targets, same as the CPU/ShadowGPU paths).
    struct ATPush { Fmatrix mvp; float uvScale[2]; float aref; float _pad; };   // matches FlushDepth
    Fmatrix vp = lightVP;

    VkPipeline       lastPipe   = VK_NULL_HANDLE;
    VkPipelineLayout lastLayout = VK_NULL_HANDLE;
    VkDescriptorSet  lastMatSet = VK_NULL_HANDLE;
    VkBuffer    lastVB = VK_NULL_HANDLE, lastIB = VK_NULL_HANDLE;
    VkIndexType lastIType = VK_INDEX_TYPE_MAX_ENUM;
    for (u32 g = 0; g < nGroups; ++g) {
        const Group& grp = s_groups[g];
        const bool at = grp.mat && grp.mat->alphaRef >= 0.f;
        if (at && (!atOn || grp.mat->set == VK_NULL_HANDLE)) continue;   // legacy split: AT → CPU cutout path
        if (skipTerrain && grp.terrain) continue;            // spot/point maps exclude terrain
        VkPipeline pipe = at ? PipelineCache::GetDepthATPipeline(grp.stride, grp.tcOffset)
                             : PipelineCache::GetDepthPipeline(grp.stride);
        if (pipe == VK_NULL_HANDLE) continue;
        VkPipelineLayout layout = at ? layoutAT : layoutSolid;
        if (layout != lastLayout) { lastLayout = layout; lastMatSet = VK_NULL_HANDLE; }
        if (pipe != lastPipe) { vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe); lastPipe = pipe; }
        if (at) {
            if (grp.mat->set != lastMatSet) { vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &grp.mat->set, 0, nullptr); lastMatSet = grp.mat->set; }
            ATPush push{}; push.mvp = vp; push.uvScale[0] = 1.0f/1024.0f; push.uvScale[1] = 1.0f/1024.0f; push.aref = grp.mat->alphaRef;
            vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
        } else {
            vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Fmatrix), &vp);
        }
        if (grp.vb != lastVB) { VkDeviceSize z = 0; vkCmdBindVertexBuffers(cmd, 0, 1, &grp.vb, &z); lastVB = grp.vb; }
        if (grp.ib != lastIB || grp.iType != lastIType) { vkCmdBindIndexBuffer(cmd, grp.ib, 0, grp.iType); lastIB = grp.ib; lastIType = grp.iType; }
        const VkDeviceSize cmdOff = ((VkDeviceSize)target * s_total + grp.entryOffset) * sizeof(VkDrawIndexedIndirectCommand);
        const VkDeviceSize cntOff = ((VkDeviceSize)target * nGroups + g) * sizeof(u32);
        vkCmdDrawIndexedIndirectCount(cmd, s_indirectSh->GetHandle(), cmdOff, s_countSh->GetHandle(), cntOff,
                                      grp.entryCount, sizeof(VkDrawIndexedIndirectCommand));
    }
}

void DrawColor(VkCommandBuffer cmd, const Fmatrix& viewProj, VkDescriptorSet envSet, bool useOcclusion)
{
    if (!Built()) return;
    const VkShaderStageFlags kStages = PipelineCache::GetPushStages();
    const u32 nGroups = (u32)s_groups.size();
    // r_hzb_cull: draw the Hi-Z-culled set (CullColor must have run this frame).
    // Falls back to the frustum set if occlusion isn't ready/active.
    const bool occ = useOcclusion && s_occlReady;
    VkBuffer indirectBuf = (occ ? s_indirect2 : s_indirect)->GetHandle();
    VkBuffer countBuf    = (occ ? s_count2    : s_count   )->GetHandle();

    // mvp + uvScale at offset 0 (constant: world-space verts, SHORT2 SSCALED TCs) —
    // re-pushed per layout flip. Per-material tail {aref, detailScale, hemi} at 72.
    // Tess block at 84 = zero (we only bind flat, non-tess pipelines).
    struct MvpUV { Fmatrix mvp; float uvScale[2]; };
    MvpUV base{}; base.mvp = viewProj; base.uvScale[0] = 1.0f/1024.0f; base.uvScale[1] = 1.0f/1024.0f;
    struct TessBlock { float tessMax, tessNear, tessFar; float eyeHeight[4]; float pnScale; };
    const TessBlock tessZero{ 0.f, 0.f, 1.f, {0,0,0,0}, 0.f };
    constexpr u32 kTailOffset = sizeof(Fmatrix) + 2 * sizeof(float);   // 72
    constexpr u32 kTessOffset = sizeof(Fmatrix) + 5 * sizeof(float);   // 84

    VkPipeline       lastPipe   = VK_NULL_HANDLE;
    VkPipelineLayout lastLayout = VK_NULL_HANDLE;
    VkDescriptorSet  lastMatSet = VK_NULL_HANDLE;
    VkBuffer    lastVB = VK_NULL_HANDLE, lastIB = VK_NULL_HANDLE;
    VkIndexType lastIType = VK_INDEX_TYPE_MAX_ENUM;

    for (u32 g = 0; g < nGroups; ++g) {
        const Group& grp = s_groups[g];
        WorldMaterial* mat = grp.mat;

        VkPipeline       pipe;
        VkPipelineLayout layout;
        VkDescriptorSet  set;
        const bool terrain = grp.terrain && mat->terrainSet != VK_NULL_HANDLE
                          && PipelineCache::GetTerrainPipeline() != VK_NULL_HANDLE;
        if (terrain) {
            pipe = PipelineCache::GetTerrainPipeline(); layout = PipelineCache::GetTerrainLayout(); set = mat->terrainSet;
        } else {
            PipelineCache::Key k{};
            k.stride = grp.stride; k.tcOffset = grp.tcOffset;
            const bool lmap = (grp.tcOffset == 24);
            // Crossfade (r_cluster_fade): the fade shader variants decode the
            // cull-packed fades and screen-door the LOD transitions. Separate
            // modules → separate cached pipelines; zero cost when fade is off.
            const bool fade = ps_r_cluster_fade > 0.f && PipelineCache::WorldLmapFadeVS() != VK_NULL_HANDLE;
            if (fade) {
                k.vs = lmap ? PipelineCache::WorldLmapFadeVS() : PipelineCache::WorldVlitFadeVS();
                k.fs = lmap ? PipelineCache::WorldLmapFadeFS() : PipelineCache::WorldVlitFadeFS();
            } else {
                k.vs = lmap ? PipelineCache::WorldLmapVS() : PipelineCache::WorldVlitVS();
                k.fs = lmap ? PipelineCache::WorldLmapFS() : PipelineCache::WorldVlitFS();
            }
            k.depthTest = true; k.wmark = false; k.tess = false;
            // Uber-FS variant (Inc 1): POM bit per-material; frame bits stamped in Get().
            k.specMask = mat->tessellated ? (u8)PipelineCache::WS_POM : (u8)0;
            pipe = PipelineCache::Get(k); layout = PipelineCache::GetLayout(); set = mat->set;
        }
        if (pipe == VK_NULL_HANDLE || layout == VK_NULL_HANDLE) continue;

        if (layout != lastLayout) {
            lastLayout = layout; lastPipe = VK_NULL_HANDLE; lastMatSet = VK_NULL_HANDLE; lastVB = VK_NULL_HANDLE; lastIB = VK_NULL_HANDLE;
            if (envSet != VK_NULL_HANDLE)
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 1, 1, &envSet, 0, nullptr);
            vkCmdPushConstants(cmd, layout, kStages, 0, sizeof(base), &base);
            vkCmdPushConstants(cmd, layout, kStages, kTessOffset, sizeof(tessZero), &tessZero);
        }
        if (pipe != lastPipe) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
            lastPipe = pipe; lastVB = VK_NULL_HANDLE; lastIB = VK_NULL_HANDLE; lastMatSet = VK_NULL_HANDLE;
        }
        if (set != VK_NULL_HANDLE && set != lastMatSet) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &set, 0, nullptr);
            lastMatSet = set;
        }
        float tail[3] = { mat->alphaRef, mat->detailScale, 1.0f };   // hemi=1.0 (statics = open sky)
        vkCmdPushConstants(cmd, layout, kStages, kTailOffset, sizeof(tail), tail);
        // GPU-feedback stream id (offset 116) — the statics color pass is where most
        // world diffuse gets sampled, so this is the main feedback producer.
        constexpr u32 kStreamIDOffset = sizeof(Fmatrix) + 13 * sizeof(float);   // 116
        vkCmdPushConstants(cmd, layout, kStages, kStreamIDOffset, sizeof(u32), &mat->streamID);

        if (grp.vb != lastVB) { VkDeviceSize z = 0; vkCmdBindVertexBuffers(cmd, 0, 1, &grp.vb, &z); lastVB = grp.vb; }
        if (grp.ib != lastIB || grp.iType != lastIType) { vkCmdBindIndexBuffer(cmd, grp.ib, 0, grp.iType); lastIB = grp.ib; lastIType = grp.iType; }
        const VkDeviceSize cmdOff = (VkDeviceSize)grp.entryOffset * sizeof(VkDrawIndexedIndirectCommand);
        const VkDeviceSize cntOff = (VkDeviceSize)g * sizeof(u32);
        vkCmdDrawIndexedIndirectCount(cmd, indirectBuf, cmdOff, countBuf, cntOff,
                                      grp.entryCount, sizeof(VkDrawIndexedIndirectCommand));
    }
}

// Lazily build one debug pipeline per (vertex stride, fill/line). Mirrors the
// tree crown-hull debug overlay (vk_TreeManager_Render): dynamic rendering on
// SceneColor + swapchain depth, LEQUAL test, no depth write, no descriptors —
// just an mvp push in the VS and the cluster id via gl_InstanceIndex.
VkPipeline GetDebugPipeline(u32 stride, bool line)
{
    const u64 key = (u64)stride * 2 + (line ? 1 : 0);
    auto it = s_dbgPipes.find(key);
    if (it != s_dbgPipes.end()) return it->second;
    if (s_dbgFailed) return VK_NULL_HANDLE;

    if (!g_ShaderManager) g_ShaderManager = xr_new<VK::CVulkanSPIRVLoader>();
    VkShaderModule vs = g_ShaderManager->Load("world_cluster_debug.vert.spv");
    VkShaderModule fs = g_ShaderManager->Load("world_cluster_debug.frag.spv");
    if (!vs || !fs) { Msg("![VK Cluster] debug shaders missing — r_cluster_debug disabled"); s_dbgFailed = true; return VK_NULL_HANDLE; }

    if (s_dbgLayout == VK_NULL_HANDLE) {
        // Set 0 = meta SSBO (VS reads parentError for the mode-4 health view).
        VkDescriptorSetLayoutBinding db{};
        db.binding = 0; db.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        db.descriptorCount = 1; db.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        VkDescriptorSetLayoutCreateInfo dlci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        dlci.bindingCount = 1; dlci.pBindings = &db;
        if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &dlci, nullptr, &s_dbgSetL) != VK_SUCCESS) { s_dbgFailed = true; return VK_NULL_HANDLE; }
        VkDescriptorPoolSize dps{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
        VkDescriptorPoolCreateInfo dpci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &dps;
        if (vkCreateDescriptorPool(VulkanHW.m_Device, &dpci, nullptr, &s_dbgPool) != VK_SUCCESS) { s_dbgFailed = true; return VK_NULL_HANDLE; }
        VkDescriptorSetAllocateInfo dsai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        dsai.descriptorPool = s_dbgPool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &s_dbgSetL;
        if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dsai, &s_dbgSet) != VK_SUCCESS) { s_dbgFailed = true; return VK_NULL_HANDLE; }
        VkDescriptorBufferInfo dbi{ s_meta->GetHandle(), 0, VK_WHOLE_SIZE };
        VkWriteDescriptorSet dw{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        dw.dstSet = s_dbgSet; dw.dstBinding = 0; dw.descriptorCount = 1;
        dw.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; dw.pBufferInfo = &dbi;
        vkUpdateDescriptorSets(VulkanHW.m_Device, 1, &dw, 0, nullptr);

        // mvp (64) + tint vec4 (16): tint.a 1 = mode 3 path color, 2 = mode 4 health.
        VkPushConstantRange pcr{ VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(Fmatrix) + sizeof(Fvector4) };
        VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        plci.setLayoutCount = 1; plci.pSetLayouts = &s_dbgSetL;
        plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
        if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_dbgLayout) != VK_SUCCESS) { s_dbgFailed = true; return VK_NULL_HANDLE; }
    }

    VkPipelineShaderStageCreateInfo ss[2]{};
    ss[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; ss[0].stage = VK_SHADER_STAGE_VERTEX_BIT;   ss[0].module = vs; ss[0].pName = "main";
    ss[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; ss[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; ss[1].module = fs; ss[1].pName = "main";
    VkVertexInputBindingDescription vibd{ 0, stride, VK_VERTEX_INPUT_RATE_VERTEX };
    VkVertexInputAttributeDescription via{ 0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0 };   // position only
    VkPipelineVertexInputStateCreateInfo vi{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &vibd;
    vi.vertexAttributeDescriptionCount = 1; vi.pVertexAttributeDescriptions = &via;
    VkPipelineInputAssemblyStateCreateInfo ia{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    vp.viewportCount = 1; vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rs.polygonMode = line ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE; rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo ds{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_FALSE; ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    VkPipelineColorBlendAttachmentState ba{};
    ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    cb.attachmentCount = 1; cb.pAttachments = &ba;
    VkDynamicState dyn[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dynState.dynamicStateCount = 2; dynState.pDynamicStates = dyn;
    VkFormat colorFmt = VK::SceneColor::Format();
    VkPipelineRenderingCreateInfo prci{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    prci.colorAttachmentCount = 1; prci.pColorAttachmentFormats = &colorFmt;
    prci.depthAttachmentFormat = Swapchain.m_DepthFormat;
    VkGraphicsPipelineCreateInfo pi{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    pi.pNext = &prci; pi.stageCount = 2; pi.pStages = ss;
    pi.pVertexInputState = &vi; pi.pInputAssemblyState = &ia; pi.pViewportState = &vp;
    pi.pRasterizationState = &rs; pi.pMultisampleState = &ms; pi.pDepthStencilState = &ds;
    pi.pColorBlendState = &cb; pi.pDynamicState = &dynState; pi.layout = s_dbgLayout;
    // Bound inside the world-color pass (SRI may be attached) — needs the create
    // flag; no dynamic FSR state, so the overlay stays full-rate (static 1x1).
    if (VulkanHW.m_bVRSSupported)
        pi.flags |= VK_PIPELINE_CREATE_RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR;
    VkPipeline pipe = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(VulkanHW.m_Device, PipelineCache::GetCacheObject(), 1, &pi, nullptr, &pipe) != VK_SUCCESS) {
        Msg("![VK Cluster] debug pipeline create failed (stride=%u line=%d)", stride, line ? 1 : 0);
        s_dbgFailed = true; return VK_NULL_HANDLE;
    }
    s_dbgPipes.emplace(key, pipe);
    return pipe;
}

void DrawDebug(VkCommandBuffer cmd, const Fmatrix& viewProj, int mode, bool useOcclusion)
{
    if (!Built() || mode <= 0) return;
    const bool line   = (mode == 2);
    const bool path   = (mode == 3);   // path view: which LOD system draws each mesh
    const bool health = (mode == 4);   // health view: can this entry ever coarsen?
    const bool occ  = useOcclusion && s_occlReady;
    VkBuffer indirectBuf = (occ ? s_indirect2 : s_indirect)->GetHandle();
    VkBuffer countBuf    = (occ ? s_count2    : s_count   )->GetHandle();
    const VkShaderStageFlags kDbgStages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    const VkBuffer ib16 = ClusterStream::PoolIB16();
    const VkBuffer ib32 = ClusterStream::PoolIB32();

    const u32 nGroups = (u32)s_groups.size();
    VkPipeline  lastPipe = VK_NULL_HANDLE;
    VkBuffer    lastVB = VK_NULL_HANDLE, lastIB = VK_NULL_HANDLE;
    VkIndexType lastIType = VK_INDEX_TYPE_MAX_ENUM;
    bool pushed = false;

    for (u32 g = 0; g < nGroups; ++g) {
        const Group& grp = s_groups[g];
        VkPipeline pipe = GetDebugPipeline(grp.stride, line);
        if (pipe == VK_NULL_HANDLE) return;   // shaders/pipeline unavailable
        if (!pushed) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_dbgLayout, 0, 1, &s_dbgSet, 0, nullptr);
            vkCmdPushConstants(cmd, s_dbgLayout, kDbgStages, 0, sizeof(Fmatrix), &viewProj);
            Fvector4 off; off.set(0.f, 0.f, 0.f, health ? 2.f : 0.f);   // a: 0 = hash colors, 2 = health view
            vkCmdPushConstants(cmd, s_dbgLayout, kDbgStages, sizeof(Fmatrix), sizeof(Fvector4), &off);
            pushed = true;
        }
        if (path) {   // red = plain (no LOD), green = per-mesh DAG, blue = component DAG
            Fvector4 tint;
            if      (grp.ib == ib32) tint.set(0.15f, 0.35f, 1.0f, 1.f);
            else if (grp.ib == ib16) tint.set(0.15f, 1.0f, 0.25f, 1.f);
            else                     tint.set(1.0f, 0.2f, 0.15f, 1.f);
            vkCmdPushConstants(cmd, s_dbgLayout, kDbgStages, sizeof(Fmatrix), sizeof(Fvector4), &tint);
        }
        if (pipe != lastPipe) { vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe); lastPipe = pipe; lastVB = VK_NULL_HANDLE; lastIB = VK_NULL_HANDLE; }
        if (grp.vb != lastVB) { VkDeviceSize z = 0; vkCmdBindVertexBuffers(cmd, 0, 1, &grp.vb, &z); lastVB = grp.vb; }
        if (grp.ib != lastIB || grp.iType != lastIType) { vkCmdBindIndexBuffer(cmd, grp.ib, 0, grp.iType); lastIB = grp.ib; lastIType = grp.iType; }
        const VkDeviceSize cmdOff = (VkDeviceSize)grp.entryOffset * sizeof(VkDrawIndexedIndirectCommand);
        const VkDeviceSize cntOff = (VkDeviceSize)g * sizeof(u32);
        vkCmdDrawIndexedIndirectCount(cmd, indirectBuf, cmdOff, countBuf, cntOff,
                                      grp.entryCount, sizeof(VkDrawIndexedIndirectCommand));
    }
}

// ============================================================================
//  Pool compaction (Stage B increment (б)) — see vk_world_gpu.h.
//  Runs once from level_Load, AFTER Build (needs the cluster/repack refs) and
//  BEFORE Trees->Build / ShadowGPU::Build (they snapshot pool handles/offsets
//  from m_mesh — building them after means they see the compacted state).
//  Caller must drain async uploads first (FlushUploadsAndWait).
// ============================================================================
bool PoolsCompacted() { return s_poolsCompacted; }

void CompactPools()
{
    // One-shot: the refs are only valid right after this level's Build.
    xr_vector<CompactRef> refs;
    refs.swap(s_compactRefs);

    if (!ps_r_pool_compact) { Msg("[VK PoolCompact] off (r_pool_compact 0)"); return; }
    if (!Built() || refs.empty()) return;
    // Freed meshes can only be drawn by the cluster shadow targets — without
    // the shadow-cull pipeline the rain/spot/point maps would lose them.
    if (!s_shadowReady) { Msg("![VK PoolCompact] skipped: shadow cluster cull unavailable"); return; }

    VK::Vram::Scope _vram_scope("Geom/Pools");

    // ---- Pool lookup: CVulkanBuffer* -> pool table slot --------------------
    struct Pool {
        CVulkanBuffer* buf = nullptr;
        bool     isVB = false;
        bool     bail = false;      // u32 pool-global page indices point into this VB — can't move it
        xr_vector<std::pair<u64, u64>> keep;   // byte intervals, merged later
        // After merge: prefix-summed new offsets (newOff[i] for keep[i]).
        xr_vector<u64> newOff;
        u64      keepBytes = 0;
        bool     compact = false;   // this buffer actually gets rebuilt
        VkBuffer oldHandle = VK_NULL_HANDLE;
    };
    xr_vector<Pool> pools;
    xr_map<CVulkanBuffer*, u32> poolOf;
    auto addPools = [&](xr_vector<CVulkanBuffer*>& v, bool isVB) {
        for (CVulkanBuffer* b : v) {
            if (!b || !b->IsValid()) continue;
            if (poolOf.find(b) != poolOf.end()) continue;
            poolOf[b] = (u32)pools.size();
            Pool p; p.buf = b; p.isVB = isVB; p.oldHandle = b->GetHandle();
            pools.push_back(p);
        }
    };
    addPools(RImplementation.nVB, true);
    addPools(RImplementation.nIB, false);
    if (pools.empty()) return;

    // ---- Freed predicate ----------------------------------------------------
    // A mesh's pool slices can go iff every GPU path draws it from the
    // ClusterStream pools (clustered + vertex-repacked) and no always-CPU path
    // needs it: alpha-tested casters draw on the CPU cutout path in every
    // shadow target, and the terrain slice feeds the TerrainCache probe copy.
    xr_map<vkFVisual*, const CompactRef*> refOf;
    for (const CompactRef& r : refs) refOf[r.fv] = &r;
    auto isFreed = [&](vkFVisual* fv) -> bool {
        auto it = refOf.find(fv);
        if (it == refOf.end()) return false;
        const CompactRef& r = *it->second;
        if (!r.clustered || !r.repacked) return false;
        WorldMaterial* m = fv->m_pWorldMaterial;
        if (!m || m->alphaRef >= 0.f || m->isTerrain) return false;
        return true;
    };

    // u32 pool-global component pages address their pool VB by ABSOLUTE vertex
    // index baked into the page blobs — that VB must not move. (Repacked pages
    // don't reference the pool at all, so only the raw-u32 combination bails.)
    for (const CompactRef& r : refs)
        if (r.clustered && r.ib32 && !r.repacked && r.fv->m_mesh.p_rm_Vertices) {
            auto it = poolOf.find(r.fv->m_mesh.p_rm_Vertices);
            if (it != poolOf.end()) pools[it->second].bail = true;
        }

    // ---- Collect every pool-backed visual (recurse hierarchies/LODs) -------
    xr_vector<vkFVisual*> all;
    all.reserve(RImplementation.Visuals.size());
    xr_vector<vkRender_Visual*> stack;
    for (IRenderVisual* iv : RImplementation.Visuals)
        if (iv) stack.push_back(static_cast<vkRender_Visual*>(iv));
    while (!stack.empty()) {
        vkRender_Visual* rv = stack.back(); stack.pop_back();
        if (rv->Type == MT_HIERRARHY || rv->Type == MT_LOD) {
            if (auto* hv = dynamic_cast<vkFHierrarhyVisual*>(rv))
                for (auto* c : hv->children) if (c) stack.push_back(c);
            continue;
        }
        auto* fv = dynamic_cast<vkFVisual*>(rv);
        if (!fv || !fv->m_mesh.IsValid()) continue;
        if (poolOf.find(fv->m_mesh.p_rm_Vertices) == poolOf.end()
         && poolOf.find(fv->m_mesh.p_rm_Indices)  == poolOf.end()) continue;   // owns its buffers
        all.push_back(fv);
    }
    std::sort(all.begin(), all.end());
    all.erase(std::unique(all.begin(), all.end()), all.end());

    // ---- Keep intervals + why-kept accounting (VB bytes) --------------------
    u64 accFreed = 0, accAT = 0, accTerrain = 0, accRaw = 0, accPlainGpu = 0, accCpu = 0;
    auto ibSize = [](const VK_Render_Mesh& m) -> u64 { return m.iType == VK_INDEX_TYPE_UINT32 ? 4 : 2; };
    // Progressive SWI windows live inside [iBase, iBase+iCount) by construction,
    // but guard the keep range with the actual window extents anyway.
    auto ibEnd = [&](vkFVisual* fv) -> u64 {
        u64 end = (u64)fv->m_mesh.iBase + fv->m_mesh.iCount;
        if (fv->Type == MT_PROGRESSIVE) {
            auto* pg = static_cast<vkFProgressive*>(fv);
            if (pg->sw_offsets && pg->sw_counts)
                for (u32 j = 0; j < pg->sw_count; ++j)
                    end = _max(end, (u64)fv->m_mesh.iBase + pg->sw_offsets[j] + pg->sw_counts[j]);
        }
        return end;
    };
    for (vkFVisual* fv : all) {
        const VK_Render_Mesh& m = fv->m_mesh;
        const u64 vbBytes = (u64)m.vCount * m.vStride;
        if (isFreed(fv)) { accFreed += vbBytes; continue; }
        auto itV = poolOf.find(m.p_rm_Vertices);
        if (itV != poolOf.end() && m.vCount) {
            Pool& p = pools[itV->second];
            p.keep.emplace_back((u64)m.vBase * m.vStride, ((u64)m.vBase + m.vCount) * m.vStride);
        }
        auto itI = poolOf.find(m.p_rm_Indices);
        if (itI != poolOf.end() && m.iCount) {
            Pool& p = pools[itI->second];
            p.keep.emplace_back((u64)m.iBase * ibSize(m), ibEnd(fv) * ibSize(m));
        }
        // Attribution (diagnostic): why does this mesh keep its VB slice?
        auto itR = refOf.find(fv);
        WorldMaterial* mat = fv->m_pWorldMaterial;
        if      (itR == refOf.end())              accCpu     += vbBytes;   // CPU leaf / tree / non-GPU-set
        else if (mat && mat->isTerrain)           accTerrain += vbBytes;
        else if (mat && mat->alphaRef >= 0.f)     accAT      += vbBytes;
        else if (itR->second->clustered)          accRaw     += vbBytes;   // clustered but not repacked
        else                                      accPlainGpu+= vbBytes;   // plain identity mesh
    }

    // ---- Merge intervals, decide per buffer, build remap tables -------------
    u64 oldVbTotal = 0, newVbTotal = 0, oldIbTotal = 0, newIbTotal = 0;
    u32 nCompact = 0, nBail = 0, nSkip = 0;
    for (Pool& p : pools) {
        std::sort(p.keep.begin(), p.keep.end());
        xr_vector<std::pair<u64, u64>> merged;
        for (const auto& iv : p.keep) {
            if (iv.second <= iv.first) continue;
            if (!merged.empty() && iv.first <= merged.back().second)
                merged.back().second = _max(merged.back().second, iv.second);
            else
                merged.push_back(iv);
        }
        p.keep.swap(merged);
        p.newOff.resize(p.keep.size());
        u64 off = 0;
        for (size_t i = 0; i < p.keep.size(); ++i) {
            p.newOff[i] = off;
            off += p.keep[i].second - p.keep[i].first;
        }
        p.keepBytes = off;
        const u64 sz = p.buf->GetSize();
        (p.isVB ? oldVbTotal : oldIbTotal) += sz;
        // Not worth the churn if nearly everything stays; bailed VBs stay whole.
        p.compact = !p.bail && p.keepBytes + (sz / 32) < sz;   // > ~3% savings
        if (p.bail) ++nBail;
        else if (!p.compact) ++nSkip;
        else ++nCompact;
        (p.isVB ? newVbTotal : newIbTotal) += p.compact ? p.keepBytes : sz;
    }

    // Byte remap inside a compacted pool (interval binary search). Offsets fed
    // to it always come from keep-interval owners, so a miss is a logic bug.
    auto remap = [](const Pool& p, u64 oldByte) -> u64 {
        size_t lo = 0, hi = p.keep.size();
        while (lo + 1 < hi) {
            const size_t mid = (lo + hi) / 2;
            if (p.keep[mid].first <= oldByte) lo = mid; else hi = mid;
        }
        R_ASSERT(!p.keep.empty() && p.keep[lo].first <= oldByte && oldByte <= p.keep[lo].second);
        return p.newOff[lo] + (oldByte - p.keep[lo].first);
    };

    // ---- Rebuild the compacted buffers (one submit, waits on completion) ----
    xr_vector<CVulkanBuffer> fresh(pools.size());
    {
        VkCommandBuffer cmd = VulkanHW.BeginSingleTimeCommands();
        if (cmd == VK_NULL_HANDLE) { Msg("![VK PoolCompact] no upload cmd — skipped"); return; }
        for (size_t i = 0; i < pools.size(); ++i) {
            Pool& p = pools[i];
            if (!p.compact || p.keepBytes == 0) continue;
            const VkBufferUsageFlags usage = (p.isVB
                ? VK_BUFFER_USAGE_VERTEX_BUFFER_BIT : VK_BUFFER_USAGE_INDEX_BUFFER_BIT)
                | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;   // tree meshlet readback + future rebuilds
            fresh[i].Create(p.keepBytes, usage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
            if (!fresh[i].IsValid()) { p.compact = false; ++nSkip; --nCompact; continue; }
            xr_vector<VkBufferCopy> regions(p.keep.size());
            for (size_t k = 0; k < p.keep.size(); ++k)
                regions[k] = { p.keep[k].first, p.newOff[k], p.keep[k].second - p.keep[k].first };
            vkCmdCopyBuffer(cmd, p.buf->GetHandle(), fresh[i].GetHandle(),
                            (u32)regions.size(), regions.data());
        }
        VulkanHW.EndSingleTimeCommands(cmd);   // waits — old buffers are idle after this
    }

    // ---- Swap in place + raw-handle map -------------------------------------
    xr_map<VkBuffer, VkBuffer> handleMap;
    for (size_t i = 0; i < pools.size(); ++i) {
        Pool& p = pools[i];
        if (!p.compact) continue;
        if (p.keepBytes == 0) { p.buf->Destroy(); continue; }   // fully freed pool: no one binds it
        p.buf->AdoptFrom(fresh[i]);
        handleMap[p.oldHandle] = p.buf->GetHandle();
    }

    // ---- Patch visuals (m_mesh offsets; freed ones become empty) ------------
    xr_map<vkFVisual*, s64> vbDelta;   // new vBase - old vBase (raw-u16 cluster meta patch)
    for (vkFVisual* fv : all) {
        VK_Render_Mesh& m = fv->m_mesh;
        if (isFreed(fv)) {
            // Every CPU path gates on IsValid() (vCount>0) / draws iCount — this
            // makes any submit of the mesh a clean no-op. GPU paths draw it from
            // the ClusterStream pools via the meta, which never reads m_mesh.
            m.vCount = 0; m.iCount = 0; m.dwPrimitives = 0;
            continue;
        }
        auto itV = poolOf.find(m.p_rm_Vertices);
        if (itV != poolOf.end() && pools[itV->second].compact && m.vCount) {
            const Pool& p = pools[itV->second];
            const u32 nb = (u32)(remap(p, (u64)m.vBase * m.vStride) / m.vStride);
            if (nb != m.vBase) vbDelta[fv] = (s64)nb - (s64)m.vBase;
            m.vBase = nb;
        }
        auto itI = poolOf.find(m.p_rm_Indices);
        if (itI != poolOf.end() && pools[itI->second].compact && m.iCount) {
            const Pool& p = pools[itI->second];
            m.iBase = (u32)(remap(p, (u64)m.iBase * ibSize(m)) / ibSize(m));
        }
    }

    // ---- Patch the GPU-set state --------------------------------------------
    // Groups snapshot raw pool handles; identity metas hold pool-global offsets;
    // raw-u16 cluster metas hold pool-global first_vertex (their page indices
    // are mesh-local). Repacked metas are page-local — untouched.
    for (Group& g : s_groups) {
        auto it = handleMap.find(g.vb); if (it != handleMap.end()) g.vb = it->second;
        it = handleMap.find(g.ib);      if (it != handleMap.end()) g.ib = it->second;
    }
    for (const CompactRef& r : refs) {
        if (r.metaFirst + r.metaCount > (u32)s_metaCPU.size()) continue;
        if (!r.clustered) {
            GpuMeshMeta& m = s_metaCPU[r.metaFirst];
            m.first_vertex = r.fv->m_mesh.vBase;
            DrawnSlice(r.fv, m.ib_first, m.index_count);
        } else if (!r.repacked && !r.ib32) {
            auto it = vbDelta.find(r.fv);
            if (it == vbDelta.end()) continue;
            for (u32 e = r.metaFirst; e < r.metaFirst + r.metaCount; ++e)
                s_metaCPU[e].first_vertex = (u32)((s64)s_metaCPU[e].first_vertex + it->second);
        }
    }
    if (s_meta) s_meta->Upload(s_metaCPU.data(), sizeof(GpuMeshMeta) * s_metaCPU.size());
    ++s_buildStamp;   // derived tables (VSM candidates) refresh from the patched meta

    // Terrain composite cache captured (original pool VB handle, vBase) during
    // grouping — re-point it at the compacted slice (same rule as seed(): the
    // FIRST terrain capture wins, so re-capture from the first terrain mesh).
    for (const CompactRef& r : refs) {
        if (r.metaFirst >= (u32)s_metaCPU.size()) continue;
        const u32 g = s_metaCPU[r.metaFirst].group;
        if (g >= (u32)s_groups.size()) continue;
        const Group& grp = s_groups[g];
        if (!grp.terrain || !grp.mat || grp.mat->terrainSet == VK_NULL_HANDLE) continue;
        if (r.metaFirst != grp.entryOffset) continue;   // first mesh of the group only
        if (!r.fv->m_mesh.p_rm_Vertices) continue;
        TerrainCache::RecaptureMesh(r.fv->m_mesh.p_rm_Vertices->GetHandle(), r.fv->m_mesh.vBase);
        break;   // OnTerrainMesh is first-capture-wins — mirror that
    }

    s_poolsCompacted = true;
    const double MB = 1.0 / (1024.0 * 1024.0);
    Msg("[VK PoolCompact] VB %.1f -> %.1f MB, IB %.1f -> %.1f MB (%u compacted, %u skipped, %u bailed of %u pools)",
        oldVbTotal * MB, newVbTotal * MB, oldIbTotal * MB, newIbTotal * MB,
        nCompact, nSkip, nBail, (u32)pools.size());
    Msg("[VK PoolCompact] VB freed %.1f MB | kept: cpu/trees %.1f, terrain %.1f, alpha-test %.1f, cluster-raw %.1f, plain %.1f MB",
        accFreed * MB, accCpu * MB, accTerrain * MB, accAT * MB, accRaw * MB, accPlainGpu * MB);
}

void Destroy()
{
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
    if (s_cullPipe)   { vkDestroyPipeline(VulkanHW.m_Device, s_cullPipe, nullptr); s_cullPipe = VK_NULL_HANDLE; }
    if (s_cullLayout) { vkDestroyPipelineLayout(VulkanHW.m_Device, s_cullLayout, nullptr); s_cullLayout = VK_NULL_HANDLE; }
    if (s_pool)       { vkDestroyDescriptorPool(VulkanHW.m_Device, s_pool, nullptr); s_pool = VK_NULL_HANDLE; }
    if (s_setL)       { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_setL, nullptr); s_setL = VK_NULL_HANDLE; }
    for (auto& kv : s_dbgPipes) if (kv.second) vkDestroyPipeline(VulkanHW.m_Device, kv.second, nullptr);
    s_dbgPipes.clear(); s_dbgFailed = false;
    if (s_dbgLayout)   { vkDestroyPipelineLayout(VulkanHW.m_Device, s_dbgLayout, nullptr); s_dbgLayout = VK_NULL_HANDLE; }
    if (s_dbgPool)     { vkDestroyDescriptorPool(VulkanHW.m_Device, s_dbgPool, nullptr); s_dbgPool = VK_NULL_HANDLE; }
    if (s_dbgSetL)     { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_dbgSetL, nullptr); s_dbgSetL = VK_NULL_HANDLE; }
    s_dbgSet = VK_NULL_HANDLE;
    if (s_cullPipe2)   { vkDestroyPipeline(VulkanHW.m_Device, s_cullPipe2, nullptr); s_cullPipe2 = VK_NULL_HANDLE; }
    if (s_cullLayout2) { vkDestroyPipelineLayout(VulkanHW.m_Device, s_cullLayout2, nullptr); s_cullLayout2 = VK_NULL_HANDLE; }
    if (s_pool2)       { vkDestroyDescriptorPool(VulkanHW.m_Device, s_pool2, nullptr); s_pool2 = VK_NULL_HANDLE; }
    if (s_setL2)       { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_setL2, nullptr); s_setL2 = VK_NULL_HANDLE; }
    if (s_cullPipe3)   { vkDestroyPipeline(VulkanHW.m_Device, s_cullPipe3, nullptr); s_cullPipe3 = VK_NULL_HANDLE; }
    if (s_cullLayout3) { vkDestroyPipelineLayout(VulkanHW.m_Device, s_cullLayout3, nullptr); s_cullLayout3 = VK_NULL_HANDLE; }
    if (s_pool3)       { vkDestroyDescriptorPool(VulkanHW.m_Device, s_pool3, nullptr); s_pool3 = VK_NULL_HANDLE; }
    if (s_setL3)       { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_setL3, nullptr); s_setL3 = VK_NULL_HANDLE; }
    auto del = [](CVulkanBuffer*& b) { if (b) { xr_delete(b); b = nullptr; } };
    del(s_meta); del(s_indirect); del(s_count); del(s_indirect2); del(s_count2);
    del(s_indirectSh); del(s_countSh);
    del(s_groupBase);
    ClusterStream::Destroy();   // pools, residency state, IO thread
    s_pageDir.clear(); s_cachePath[0] = 0; s_blobOff16 = s_blobOff32 = s_blobOffVb = 0;
    if (s_countReadback) { VK::Vram::DestroyBuffer(VulkanHW.m_Allocator, s_countReadback, s_countReadbackAlloc);
                           s_countReadback = VK_NULL_HANDLE; s_countReadbackAlloc = VK_NULL_HANDLE; s_countReadbackPtr = nullptr; }
    if (s_cmdAudit) { VK::Vram::DestroyBuffer(VulkanHW.m_Allocator, s_cmdAudit, s_cmdAuditAlloc);
                      s_cmdAudit = VK_NULL_HANDLE; s_cmdAuditAlloc = VK_NULL_HANDLE; s_cmdAuditPtr = nullptr; }
    s_cmdAuditPending = false; s_cmdAuditHasOccl = false; s_cmdAuditFrame = 0; s_cmdAuditLast = 0;
    s_cmdAuditBits.clear(); s_cmdAuditSlotBase.clear();
    s_groups.clear(); s_meshSet.clear(); s_cpuMeshes.clear(); s_metaCPU.clear(); s_total = 0; s_nGroups = 0;
    s_compactRefs.clear(); s_poolsCompacted = false;
    s_set = VK_NULL_HANDLE; s_set2 = VK_NULL_HANDLE; s_set3 = VK_NULL_HANDLE;
    s_occlReady = false; s_shadowReady = false; s_built = false;
}

}} // namespace VK::WorldGPU
