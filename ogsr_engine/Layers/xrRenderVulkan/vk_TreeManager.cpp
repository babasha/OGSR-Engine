// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan - CTreeManager Session A: extract MT_TREE_ST/PM visuals
// from the level, group by (vb,ib,tcOffset,texture), upload metadata +
// transforms SSBOs, allocate indirect/draw-count buffers + per-texture
// descriptor sets. No rendering yet — Session B adds compute cull + draw.

#include "stdafx.h"
#include "vk_TreeManager.h"
#include "vk_Visual.h"
#include "vk_material.h"
#include "vk_world_material.h"   // VK::WorldMaterial — live diffuse for level statics
#include "vk_texture.h"
#include "vk_texture_stream.h"   // SetStreamable — tree-path textures opt OUT of mip streaming
#include "vk_buffer.h"
#include "HW_Vulkan.h"
#include "CRender_Vulkan.h"

#include <algorithm>
#include <cfloat>
#include <map>       // icosphere edge-midpoint dedup (crown-hull bake)

// GLOBAL scope (an extern inside namespace VK would mangle as VK::* → LNK2001).
extern int ps_r_vsm_tree_impostor;   // r_vsm_tree_impostor — parked/default 0; gates the load-time silhouette bake
extern int ps_r_vsm_tree_hull_vox;   // r_vsm_tree_hull_vox — hull shape: 0 = PCA lobes, >0 = voxel-shell resolution (baked)

namespace VK
{

CTreeManager::CTreeManager() {}
CTreeManager::~CTreeManager() { Destroy(); }

// ============================================================================
// Recursive extraction. Mirrors monolith's ExtractMeshesFromVisual but only
// the tree-relevant cases — no AREF static or shadow GBuffer paths here.
// ============================================================================
void CTreeManager::ExtractFromVisual(::vkRender_Visual* vis, xr_vector<::vkFTreeVisual*>& outTrees, u32 source)
{
    if (!vis) return;

    const u32 type = vis->Type;

    if (type == MT_TREE_ST || type == MT_TREE_PM)
    {
        auto* tv = dynamic_cast<vkFTreeVisual*>(vis);
        if (!tv) return;
        if (!tv->m_mesh.p_rm_Vertices || !tv->m_mesh.p_rm_Indices) return;
        if (tv->m_mesh.dwPrimitives == 0) return;
        // Live diffuse for level statics lives in m_pWorldMaterial (the parked
        // CMaterial / m_pMaterial is a stub for the deferred path — null here).
        if (!tv->m_pWorldMaterial || tv->m_pWorldMaterial->view == VK_NULL_HANDLE) return;
        outTrees.push_back(tv);
        return;
    }

    if (type == MT_HIERRARHY)
    {
        auto* hv = dynamic_cast<vkFHierrarhyVisual*>(vis);
        if (!hv) return;
        for (auto* child : hv->children)
            ExtractFromVisual(child, outTrees, 1);
        return;
    }

    // LOD containers (MT_LOD = FLOD, extends FHierrarhyVisual): at close range
    // R4 renders ALL children — the container holds the tree's SEPARATE parts
    // (trunk + leaf/branch visuals), NOT a stack of LOD levels. Taking only
    // children[0] dropped every other part → missing twigs, and floating crowns
    // when children[0] happened to be a leaf cluster (the trunk lives in a later
    // child). Recurse over ALL children, same as MT_HIERRARHY.
    // (Ref: r__dsgraph_build.cpp:557-562 `for (Vis : pV->children) add_leafs_static`.)
    if (type == MT_LOD)
    {
        auto* hv = dynamic_cast<vkFHierrarhyVisual*>(vis);
        if (!hv) return;
        for (auto* child : hv->children)
            ExtractFromVisual(child, outTrees, 2);
        return;
    }
    // MT_NORMAL / MT_PROGRESSIVE / MT_SKELETON_*: skip (not trees).
}

// ============================================================================
// Build — main entry point. Called from CRender::level_Load after Visuals[]
// is populated.
// ============================================================================
void CTreeManager::Build()
{
    VK::Vram::Scope _vram_scope("Trees");
    if (m_bBuilt) return;

    xr_vector<vkFTreeVisual*> trees;
    trees.reserve(2048);
    for (IRenderVisual* iv : RImplementation.Visuals)
        ExtractFromVisual(static_cast<vkRender_Visual*>(iv), trees);

    // Dedup: MT_HIERRARHY/MT_LOD children are getVisual(id) REFERENCES into
    // Visuals[], so the same tree object gets collected both as a top-level
    // entry and via container recursion. Rendering it 2-3× overlaps alpha-tested
    // foliage (z-fight → "moth-eaten"/floating leaf cards). Keep one per object.
    {
        const u32 before = (u32)trees.size();
        std::sort(trees.begin(), trees.end());
        trees.erase(std::unique(trees.begin(), trees.end()), trees.end());
        if (before != (u32)trees.size())
            Msg("[VK Trees] Dedup: %u -> %u unique tree visuals", before, (u32)trees.size());
    }

    if (trees.empty())
    {
        // Diagnostic dump: tally MT_* types across Visuals[] so we can spot
        // when a level uses an unexpected wrapping (e.g. trees baked as
        // MT_NORMAL inside MT_LOD containers we never recurse through).
        u32 typeCount[32]{};
        for (IRenderVisual* iv : RImplementation.Visuals) {
            auto* v = static_cast<vkRender_Visual*>(iv);
            if (!v) continue;
            const u32 t = v->Type;
            if (t < 32) typeCount[t]++;
        }
        Msg("[VK Trees] No MT_TREE_ST/PM visuals — top-level type tally:");
        for (u32 t = 0; t < 32; ++t)
            if (typeCount[t]) Msg("[VK Trees]   type=%u : %u", t, typeCount[t]);
        m_bBuilt = true;
        return;
    }

    // ----- Sort by (tcOffset, diffuse, vb, ib) so consecutive trees share
    // pipeline + descriptor + buffer binds.
    std::sort(trees.begin(), trees.end(),
        [](vkFTreeVisual* a, vkFTreeVisual* b) {
            if (a->m_mesh.tcOffset != b->m_mesh.tcOffset) return a->m_mesh.tcOffset < b->m_mesh.tcOffset;
            if (a->m_pWorldMaterial != b->m_pWorldMaterial)
                return a->m_pWorldMaterial < b->m_pWorldMaterial;
            const VkBuffer av = a->m_mesh.p_rm_Vertices->GetHandle();
            const VkBuffer bv = b->m_mesh.p_rm_Vertices->GetHandle();
            if (av != bv) return av < bv;
            return a->m_mesh.p_rm_Indices->GetHandle() < b->m_mesh.p_rm_Indices->GetHandle();
        });

    // ----- Collect unique diffuse views and assign per-tree desc index.
    xr_vector<VkImageView> uniqueViews;
    uniqueViews.reserve(64);
    xr_vector<u32> treeTexIdx(trees.size(), 0u);
    for (size_t i = 0; i < trees.size(); ++i)
    {
        // This instanced path bakes the RAW VkImageView into its own descriptor
        // sets for the whole level lifetime — a dynamic mip swap would retire and
        // destroy that view under us (white/transparent trunks), and this path's
        // FS writes no streaming feedback, so the streamer would judge these
        // textures "invisible" and demote them while they fill the screen. So:
        // raise anything the load-time cap crushed to a presentable 512px floor
        // (safe: nothing captured our view yet), THEN opt out of streaming for
        // good — the residency is frozen from here on.
        WorldMaterial* wm = trees[i]->m_pWorldMaterial;
        // 1024px floor: this path also carries building walls/vehicles — at the
        // earlier 512 floor large facades still read as mush (frozen forever).
        VK::TextureStreamer::Instance().EnsureMinResidency(wm->tex, 1024);
        VK::TextureStreamer::Instance().SetStreamable(wm->tex, false);
        wm->streamID = 0xFFFFFFFFu;   // stale feedback slot must not be pushed by the world pass

        VkImageView view = wm->view;
        u32 idx = ~0u;
        for (u32 j = 0; j < uniqueViews.size(); ++j)
            if (uniqueViews[j] == view) { idx = j; break; }
        if (idx == ~0u) {
            idx = (u32)uniqueViews.size();
            uniqueViews.push_back(view);
        }
        treeTexIdx[i] = idx;
    }

    // ----- Pack per-instance + per-mesh GPU arrays.
    m_TotalCount = (u32)trees.size();
    xr_vector<GpuTreeMeta>     meta(m_TotalCount);
    xr_vector<GpuTreeInstance> xforms(m_TotalCount);

    for (u32 i = 0; i < m_TotalCount; ++i)
    {
        vkFTreeVisual* t = trees[i];
        GpuTreeMeta& m   = meta[i];
        m.sphere_P     = t->vis.sphere.P;
        m.sphere_R     = t->vis.sphere.R;
        m.index_count  = t->m_mesh.dwPrimitives * 3;
        m.ib_first     = t->m_mesh.iBase;       // element offset into the IB pool
        m.first_vertex = t->m_mesh.vBase;       // element offset into the VB pool
        m._pad         = 0;

        // Progressive trees (MT_TREE_PM): draw the finest sliding window sw[0]
        // (== R4 select_lod_id at closest range), NOT the full container. The
        // container's first ~80% of indices are progressive vsplit/collapse
        // records; drawing them as a flat triangle list yields long stretched
        // degenerate triangles ("streaky branches in the air"). sw[0] is the
        // clean full-detail crown ([iBase+offset, iBase+offset+num_tris*3]).
        if (t->Type == MT_TREE_PM)
        {
            auto* pm = static_cast<vkFTreeVisual_PM*>(t);
            if (pm->sw_count > 0)
            {
                m.index_count = pm->sw_counts[0];
                m.ib_first    = t->m_mesh.iBase + pm->sw_offsets[0];
            }
        }

        GpuTreeInstance& x = xforms[i];
        x.xform        = t->xform;
        x.c_scale_hemi = t->c_scale.hemi;
        x.c_bias_hemi  = t->c_bias.hemi;
        // Wind class by diffuse TEXTURE NAME (tree materials don't expose alphaRef,
        // and this "tree" path also carries non-foliage statics — vehicles, props,
        // walls). Only real foliage gets wind:
        //   2 = foliage (bend + flow-map flutter): under "trees\" and NOT bark/spil
        //   1 = trunk   (gentle bend only): under "trees\" bark/spil (sways with crown)
        //   0 = rigid   (no wind): everything else (veh\, prop\, mtl\, wood\, ...)
        auto hasKw = [](const char* s, std::initializer_list<const char*> kws) {
            if (!s || !s[0]) return false;
            xr_string low = s; std::transform(low.begin(), low.end(), low.begin(), ::tolower);
            for (auto* k : kws) if (low.find(k) != xr_string::npos) return true;
            return false;
        };
        const char* nm = "";
        if (t->m_pWorldMaterial) { const char* n = t->m_pWorldMaterial->name.c_str(); if (n) nm = n; }
        const bool underTrees = (strncmp(nm, "trees\\", 6) == 0) || (strncmp(nm, "trees/", 6) == 0);
        const bool isBark     = hasKw(nm, { "bark", "spil" });   // trunk / cut-stump cross-section
        u32 windClass = 0u;
        if (underTrees) windClass = isBark ? 1u : 2u;
        x._pad0 = windClass;
        x._pad1 = 0;
        if (m_WindClassCPU.size() != m_TotalCount) m_WindClassCPU.assign(m_TotalCount, 0);
        m_WindClassCPU[i] = (u8)windClass;   // hull tier gates on foliage (crowns-only)
    }

    // ----- Group consecutive trees with same (tcOffset, descSet, vb, ib, stride).
    m_Groups.clear();
    m_Groups.reserve(64);
    m_MaxGroupMeshCount = 0;
    {
        TreeIndirectGroup cur{};
        auto seedFrom = [&](u32 i) {
            cur.vb         = trees[i]->m_mesh.p_rm_Vertices->GetHandle();
            cur.ib         = trees[i]->m_mesh.p_rm_Indices->GetHandle();
            cur.stride     = trees[i]->m_mesh.vStride;
            cur.tcOffset   = trees[i]->m_mesh.tcOffset;
            cur.descSetIdx = treeTexIdx[i];
            cur.meshOffset = i;
            cur.meshCount  = 1;
        };
        seedFrom(0);
        for (u32 i = 1; i < m_TotalCount; ++i)
        {
            const VkBuffer vb = trees[i]->m_mesh.p_rm_Vertices->GetHandle();
            const VkBuffer ib = trees[i]->m_mesh.p_rm_Indices->GetHandle();
            const u32 stride  = trees[i]->m_mesh.vStride;
            const u32 tcOff   = trees[i]->m_mesh.tcOffset;
            const u32 desc    = treeTexIdx[i];
            if (vb == cur.vb && ib == cur.ib && stride == cur.stride &&
                tcOff == cur.tcOffset && desc == cur.descSetIdx)
            {
                cur.meshCount++;
            }
            else
            {
                if (cur.meshCount > m_MaxGroupMeshCount) m_MaxGroupMeshCount = cur.meshCount;
                m_Groups.push_back(cur);
                seedFrom(i);
            }
        }
        if (cur.meshCount > m_MaxGroupMeshCount) m_MaxGroupMeshCount = cur.meshCount;
        m_Groups.push_back(cur);
    }


    // ----- Upload + allocate.
    UploadMetadata(meta);
    UploadTransforms(xforms);
    BuildMeshlets(trees, meta);   // Phase A: VSM meshlet-cull clusters (r_vsm_meshlet)
    CreateIndirectBuffers();
    CreateTextureDescriptors(uniqueViews);
    // r_vsm_tree_impostor: bake per-mesh crown silhouettes — ONLY when the (parked,
    // default-0) feature is enabled AT LOAD. Off = nothing baked/allocated at all, so
    // the impostor is provably inert (it never touched bushes anyway — R<3m visuals are
    // excluded from the near/dyn set in VsmUpdateNearSet). Toggling it on now needs the
    // cvar set before the level loads (reload), which is fine for a parked debug path.
    if (ps_r_vsm_tree_impostor)
        BuildImpostorAtlas(trees, meta, treeTexIdx);

    // ----- Session B: compute-cull + graphics pipelines + per-frame resources.
    CreateFrustumUBO();
    CreateCullPipeline();
    CreateXformDescriptor();
    CreateGfxPipelines();

    Msg("[VK Trees] Built: %u instances in %u groups, %u textures, max group %u",
        m_TotalCount, (u32)m_Groups.size(), (u32)uniqueViews.size(), m_MaxGroupMeshCount);

    m_bBuilt = true;
}

// ============================================================================
// Upload helpers — staging via host-visible TRANSFER_SRC + one-shot copy.
// Same pattern as CDetailManager::BakeHeightmap (host stage → cmd copy).
// ============================================================================
static void UploadDeviceLocal(CVulkanBuffer*& dst, const void* data, VkDeviceSize size,
                              VkBufferUsageFlags extraUsage)
{
    dst = xr_new<CVulkanBuffer>();
    // gpuOnly=true: without it the storage-usage default requests host access and VMA
    // places the buffer in BAR/system memory — tree META is read by every thread of
    // the 193k-thread bins, so that means PCIe per warp (the Pripyat Bins/Tree* burn).
    dst->Create(size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | extraUsage,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);

    CVulkanBuffer staging;
    staging.Create(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
    if (void* mapped = staging.Map())
    {
        memcpy(mapped, data, size);
        staging.Flush();
    }

    VkCommandBuffer cmd = VulkanHW.BeginSingleTimeCommands();
    if (cmd != VK_NULL_HANDLE)
    {
        VkBufferCopy cp{ 0, 0, size };
        vkCmdCopyBuffer(cmd, staging.GetHandle(), dst->GetHandle(), 1, &cp);
        VulkanHW.EndSingleTimeCommands(cmd);
    }
    staging.Destroy();
}

void CTreeManager::UploadMetadata(const xr_vector<GpuTreeMeta>& meta)
{
    UploadDeviceLocal(m_TreeMetadataBuffer, meta.data(),
                      meta.size() * sizeof(GpuTreeMeta), 0);
    m_MetaCPU = meta;   // kept for the CPU-culled shadow caster path (RenderDepth)
}

void CTreeManager::UploadTransforms(const xr_vector<GpuTreeInstance>& xforms)
{
    UploadDeviceLocal(m_TreeTransformsBuffer, xforms.data(),
                      xforms.size() * sizeof(GpuTreeInstance), 0);
}

// ============================================================================
// Phase A — meshlet clusters for the VSM per-page meshlet-cull path.
// ⛔ NO NET PERF GAIN on dGPU (2026-07-02) — VSMrender is fill-bound, cutting vertices didn't help.
// KEPT (correct, picture identical) but DISABLED via r_vsm_meshlet=0. Don't re-chase; see the cvar
// block in vk_console_min.cpp + memory [[vulkan-vsm-meshlet-plan]]. Possible future value on iGPU.
// ============================================================================
namespace {

// Triangles per meshlet cluster (tunable — Phase C sweeps 64/128/256).
constexpr u32 kMeshletTris = 128;

// 10-bit Morton (Z-order) — spatial sort so a meshlet's triangles stay compact.
inline u32 Part1By2(u32 n) {
    n &= 0x3ff;
    n = (n | (n << 16)) & 0x030000ff;
    n = (n | (n <<  8)) & 0x0300f00f;
    n = (n | (n <<  4)) & 0x030c30c3;
    n = (n | (n <<  2)) & 0x09249249;
    return n;
}
inline u32 Morton3(u32 x, u32 y, u32 z) { return Part1By2(x) | (Part1By2(y) << 1) | (Part1By2(z) << 2); }

// GPU-readback of a device-local sub-range into a freshly-allocated host vector.
// Pools carry TRANSFER_SRC (rvk_loader). Returns false on any failure.
bool ReadbackRange(VkBuffer src, VkDeviceSize offset, VkDeviceSize size, xr_vector<u8>& out)
{
    if (src == VK_NULL_HANDLE || size == 0) return false;
    CVulkanBuffer stg;
    stg.Create(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
    VkCommandBuffer cmd = VulkanHW.BeginSingleTimeCommands();
    if (cmd == VK_NULL_HANDLE) { stg.Destroy(); return false; }
    VkBufferCopy cp{ offset, 0, size };
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

// ----- Crown-HULL bake helpers (r_vsm_tree_hull) — see BuildMeshlets' hull block. -----

// Unit icosphere, subdiv 1: 42 verts / 80 tris. Deterministic; built once, reused per lobe.
struct IcoSphere { xr_vector<Fvector> v; xr_vector<u16> i; };
static const IcoSphere& UnitIcosphere()
{
    static IcoSphere s;
    if (!s.v.empty()) return s;
    const float t = 1.6180339887f;   // golden ratio
    const float base[12][3] = {
        {-1,t,0},{1,t,0},{-1,-t,0},{1,-t,0},{0,-1,t},{0,1,t},
        {0,-1,-t},{0,1,-t},{t,0,-1},{t,0,1},{-t,0,-1},{-t,0,1} };
    for (auto& b : base) { Fvector p{ b[0], b[1], b[2] }; p.normalize(); s.v.push_back(p); }
    const u16 f[20][3] = {
        {0,11,5},{0,5,1},{0,1,7},{0,7,10},{0,10,11},{1,5,9},{5,11,4},{11,10,2},{10,7,6},{7,1,8},
        {3,9,4},{3,4,2},{3,2,6},{3,6,8},{3,8,9},{4,9,5},{2,4,11},{6,2,10},{8,6,7},{9,8,1} };
    // One midpoint subdivision (edge-midpoint dedup) → 42/80.
    std::map<u32, u16> mid;
    auto midpoint = [&](u16 a, u16 b) -> u16 {
        const u32 key = (u32)(std::min)(a, b) << 16 | (std::max)(a, b);
        auto it = mid.find(key);
        if (it != mid.end()) return it->second;
        Fvector p; p.add(s.v[a], s.v[b]); p.normalize();
        const u16 idx = (u16)s.v.size();
        s.v.push_back(p); mid[key] = idx;
        return idx;
    };
    for (auto& tri : f) {
        const u16 a = tri[0], b = tri[1], c = tri[2];
        const u16 ab = midpoint(a, b), bc = midpoint(b, c), ca = midpoint(c, a);
        const u16 out[4][3] = { {a,ab,ca},{b,bc,ab},{c,ca,bc},{ab,bc,ca} };
        for (auto& o : out) { s.i.push_back(o[0]); s.i.push_back(o[1]); s.i.push_back(o[2]); }
    }
    return s;
}

// Cluster the crown's used vertices (AREA-WEIGHTED k-means, farthest-point init) and emit
// one ORIENTED ellipsoid icosphere lobe per leaf-mass cluster into hullVerts/hullIdx
// (indices RELATIVE to the species' first hull vertex — the indirect draw supplies
// vertexOffset). The lobe union is the merged "AC Shadows"-style crown shadow proxy.
// Smart placement (user feedback 11-07): per-vertex weights = incident triangle area, so
// centroids gravitate to the leaf TUFTS (big leaf cards dominate area; twig strips are
// tiny) — lobes hang ON the foliage. Per-cluster PCA then CLASSIFIES the shape: a long
// thin cluster (σ0 >> σ1, second extent < ~12 cm) is a bare TWIG → dropped entirely (its
// shadow is noise); everything else (tufts = planar/volumetric, thick boughs = fat
// sticks) gets a lobe. Trunks never reach here (bark visuals are excluded from the hull
// tier — they cast their real opaque mesh). Returns the emitted index count.
static u32 BuildHullLobes(const xr_vector<Fvector>& ptsIn, const xr_vector<float>& wtsIn,
                          xr_vector<Fvector>& hullVerts, xr_vector<u16>& hullIdx)
{
    // Load-time bound: k-means is O(k·n·iters) — stride-subsample huge crowns (the lobe
    // statistics converge long before 3k points; per-species, once per level load).
    xr_vector<Fvector> sampledP; xr_vector<float> sampledW;
    const xr_vector<Fvector>* p = &ptsIn;
    const xr_vector<float>*   w = &wtsIn;
    if (ptsIn.size() > 3000) {
        const u32 stride = ((u32)ptsIn.size() + 2999) / 3000;
        sampledP.reserve(3000); sampledW.reserve(3000);
        for (u32 i = 0; i < (u32)ptsIn.size(); i += stride) { sampledP.push_back(ptsIn[i]); sampledW.push_back(wtsIn[i]); }
        p = &sampledP; w = &sampledW;
    }
    const xr_vector<Fvector>& pts = *p;
    const xr_vector<float>&   wts = *w;
    const u32 n = (u32)pts.size();
    if (n < 12) return 0;
    // Lobe budget: a big crown gets ~25-32 lobes, bushes ~10 — the union must READ as the
    // tree's silhouette (branch structure), not 3-4 balloons (user feedback 11-07).
    const u32 k = (std::min)(32u, (std::max)(10u, 6u + (u32)ptsIn.size() / 100u));

    // Farthest-point init (deterministic), then area-weighted Lloyd iterations.
    xr_vector<Fvector> ctr; ctr.reserve(k);
    ctr.push_back(pts[0]);
    xr_vector<float> bestD(n, FLT_MAX);
    for (u32 c = 1; c < k; ++c) {
        u32 far_i = 0; float far_d = -1.f;
        for (u32 i = 0; i < n; ++i) {
            const float d = (std::min)(bestD[i], ctr.back().distance_to_sqr(pts[i]));
            bestD[i] = d;
            if (d > far_d) { far_d = d; far_i = i; }
        }
        ctr.push_back(pts[far_i]);
    }
    xr_vector<u32> assign(n, 0);
    for (u32 iter = 0; iter < 6; ++iter) {
        for (u32 i = 0; i < n; ++i) {
            u32 bi = 0; float bd = FLT_MAX;
            for (u32 c = 0; c < k; ++c) {
                const float d = ctr[c].distance_to_sqr(pts[i]);
                if (d < bd) { bd = d; bi = c; }
            }
            assign[i] = bi;
        }
        xr_vector<Fvector> sum(k, Fvector{ 0, 0, 0 });
        xr_vector<float> cw(k, 0.f);
        for (u32 i = 0; i < n; ++i) {
            const float wi = (std::max)(wts[i], 1e-6f);
            Fvector pw = pts[i]; pw.mul(wi);
            sum[assign[i]].add(pw); cw[assign[i]] += wi;
        }
        for (u32 c = 0; c < k; ++c)
            if (cw[c] > 0.f) { ctr[c].set(sum[c].x / cw[c], sum[c].y / cw[c], sum[c].z / cw[c]); }
    }

    // Per cluster: ORIENTED ellipsoid lobe via PCA (3x3 covariance → Jacobi eigenvectors).
    // A branch/trunk cluster stretches along its own axis → a thin "stick" lobe instead of
    // an axis-aligned balloon; leaf tufts stay roundish. Extents = min(1.9σ, max projection)
    // per principal axis (σ covers the bulk, the projection clamp stops outlier twigs from
    // ballooning the lobe), floored so flat leaf-card clusters still cast.
    const IcoSphere& ico = UnitIcosphere();
    const u32 vbase0 = (u32)hullVerts.size();
    const u32 ibase0 = (u32)hullIdx.size();
    for (u32 c = 0; c < k; ++c) {
        Fvector mean{ 0, 0, 0 };
        u32 cnt = 0; double sumW = 0.0;
        for (u32 i = 0; i < n; ++i) {
            if (assign[i] != c) continue;
            const float wi = (std::max)(wts[i], 1e-6f);
            Fvector pw = pts[i]; pw.mul(wi);
            mean.add(pw); sumW += wi; ++cnt;
        }
        if (cnt < 4 || sumW <= 0.0) continue;
        mean.div((float)sumW);

        // Symmetric AREA-WEIGHTED covariance (leaf mass shapes the lobe, twig verts barely).
        double C[3][3] = {};
        for (u32 i = 0; i < n; ++i) {
            if (assign[i] != c) continue;
            const double wi = (std::max)(wts[i], 1e-6f);
            const double d[3] = { pts[i].x - mean.x, pts[i].y - mean.y, pts[i].z - mean.z };
            for (int a = 0; a < 3; ++a)
                for (int b = a; b < 3; ++b) C[a][b] += wi * d[a] * d[b];
        }
        for (int a = 0; a < 3; ++a)
            for (int b = a; b < 3; ++b) { C[a][b] /= sumW; C[b][a] = C[a][b]; }

        // Cyclic Jacobi eigendecomposition: V columns = principal axes, diag(C) = variances.
        double V[3][3] = { {1,0,0},{0,1,0},{0,0,1} };
        for (int sweep = 0; sweep < 8; ++sweep) {
            for (int p = 0; p < 2; ++p)
            for (int q = p + 1; q < 3; ++q) {
                if (fabs(C[p][q]) < 1e-12) continue;
                const double theta = 0.5 * atan2(2.0 * C[p][q], C[q][q] - C[p][p]);
                const double s = sin(theta), co = cos(theta);
                for (int r = 0; r < 3; ++r) {
                    const double crp = C[r][p], crq = C[r][q];
                    C[r][p] = co * crp - s * crq;
                    C[r][q] = s * crp + co * crq;
                }
                for (int r = 0; r < 3; ++r) {
                    const double crp = C[p][r], crq = C[q][r];
                    C[p][r] = co * crp - s * crq;
                    C[q][r] = s * crp + co * crq;
                }
                for (int r = 0; r < 3; ++r) {
                    const double vrp = V[r][p], vrq = V[r][q];
                    V[r][p] = co * vrp - s * vrq;
                    V[r][q] = s * vrp + co * vrq;
                }
            }
        }
        Fvector axis[3];
        float sigma[3], maxProj[3] = { 0, 0, 0 };
        for (int a = 0; a < 3; ++a) {
            axis[a].set((float)V[0][a], (float)V[1][a], (float)V[2][a]);
            axis[a].normalize_safe();
            sigma[a] = sqrtf((float)(std::max)(C[a][a], 0.0));
        }
        for (u32 i = 0; i < n; ++i) {
            if (assign[i] != c) continue;
            const Fvector d{ pts[i].x - mean.x, pts[i].y - mean.y, pts[i].z - mean.z };
            for (int a = 0; a < 3; ++a)
                maxProj[a] = (std::max)(maxProj[a], fabsf(d.dotproduct(axis[a])));
        }
        float ext[3];
        for (int a = 0; a < 3; ++a)
            ext[a] = (std::max)((std::min)(1.9f * sigma[a], maxProj[a]), 0.08f);

        // Shape classification (extents sorted descending): LINEAR + thin = bare TWIG —
        // no lobe at all (its shadow is noise the merged proxy shouldn't fake). Planar
        // leaf cards / volumetric tufts / thick boughs all pass.
        int ord[3] = { 0, 1, 2 };
        std::sort(ord, ord + 3, [&](int a, int b) { return ext[a] > ext[b]; });
        if (ext[ord[1]] < 0.12f && ext[ord[0]] > 2.5f * ext[ord[1]]) continue;

        const u16 lobeBase = (u16)(hullVerts.size() - vbase0);
        for (const Fvector& u : ico.v) {
            Fvector p = mean;
            p.mad(axis[0], u.x * ext[0]);
            p.mad(axis[1], u.y * ext[1]);
            p.mad(axis[2], u.z * ext[2]);
            hullVerts.push_back(p);
        }
        for (const u16 ii : ico.i) hullIdx.push_back((u16)(lobeBase + ii));
    }
    return (u32)hullIdx.size() - ibase0;
}

// VOXEL-SHELL crown hull (r_vsm_tree_hull_vox > 0) — UE5 ShapePreservation=Voxelize in
// spirit: bin the crown's area-weighted leaf vertices into a 3D occupancy grid, then emit
// a WATERTIGHT OPAQUE surface: only the faces of occupied cells whose neighbour is empty
// (interior cells are fully occluded → pure overdraw in a depth-only shadow, so skipped).
// The exposed-face shell reads as the crown SILHOUETTE from every angle — a truer shape
// than the few PCA balloons — while staying opaque (no alpha-discard, the whole point).
// Indices are RELATIVE to the species' first hull vertex (the indirect draw supplies
// vertexOffset), matching BuildHullLobes' contract. `tris` = flat triangle corners
// (3 per triangle) of the crown foliage. `voxN` = target cells across the crown's
// largest axis. Returns the emitted index count.
//
// Occupancy is by TRIANGLE RASTERIZATION, not vertices: STALKER foliage is mostly
// billboards / low-poly cards (a whole crown = a few big quads = ~8 verts), so marking
// vertex cells collapsed the crown to one box. We sample each leaf triangle at ~cell
// density and mark every cell it crosses → the shell fills along the real cards. The
// AABB is ROBUST (mean ± 3σ) so one stray far vertex can't inflate the grid and crush
// the crown into a corner cell.
// Robust crown AABB: mean ± 3σ per axis, clamped to the true extent. A far outlier
// vertex barely moves the mean but would blow up a raw min/max — clipping keeps the
// voxel grid on the actual crown. Shared by the shell and the voxel-cloud bakes.
static bool RobustTriBounds(const xr_vector<Fvector>& tris, Fvector& lo, Fvector& hi)
{
    const u32 nC = (u32)tris.size();
    if (nC < 6) return false;
    Fvector mean{ 0,0,0 };
    for (u32 i = 0; i < nC; ++i) mean.add(tris[i]);
    mean.div((float)nC);
    Fvector var{ 0,0,0 }, tmin = tris[0], tmax = tris[0];
    for (u32 i = 0; i < nC; ++i) {
        const Fvector d{ tris[i].x - mean.x, tris[i].y - mean.y, tris[i].z - mean.z };
        var.x += d.x * d.x; var.y += d.y * d.y; var.z += d.z * d.z;
        tmin.x = (std::min)(tmin.x, tris[i].x); tmin.y = (std::min)(tmin.y, tris[i].y); tmin.z = (std::min)(tmin.z, tris[i].z);
        tmax.x = (std::max)(tmax.x, tris[i].x); tmax.y = (std::max)(tmax.y, tris[i].y); tmax.z = (std::max)(tmax.z, tris[i].z);
    }
    const Fvector sig{ sqrtf(var.x / nC), sqrtf(var.y / nC), sqrtf(var.z / nC) };
    lo = Fvector{ (std::max)(tmin.x, mean.x - 3.f * sig.x), (std::max)(tmin.y, mean.y - 3.f * sig.y), (std::max)(tmin.z, mean.z - 3.f * sig.z) };
    hi = Fvector{ (std::min)(tmax.x, mean.x + 3.f * sig.x), (std::min)(tmax.y, mean.y + 3.f * sig.y), (std::min)(tmax.z, mean.z + 3.f * sig.z) };
    return true;
}

static u32 BuildHullVoxels(const xr_vector<Fvector>& tris,
                           xr_vector<Fvector>& hullVerts, xr_vector<u16>& hullIdx, u32 voxN)
{
    const u32 nC = (u32)tris.size();          // corner count (= 3 * triangles)
    Fvector lo, hi;
    if (!RobustTriBounds(tris, lo, hi)) return 0;
    const float ext[3] = { hi.x - lo.x, hi.y - lo.y, hi.z - lo.z };
    const float maxExt = (std::max)(ext[0], (std::max)(ext[1], ext[2]));
    if (maxExt < 1e-3f) return 0;
    const float cell = maxExt / float((std::max)(2u, voxN));
    const float inv  = 1.f / cell;
    // Cap at 36/axis: a filled shell's worst-case vertex count (~48·D²) must stay under the
    // u16 index range per species (65536) — and it bounds bake time. voxN normally sets the
    // max-axis dim to ~voxN anyway (cell scales with the crown), so this only clamps extremes.
    auto dim = [&](int a) { return (std::min)(36, (std::max)(1, (int)ceilf(ext[a] * inv) + 1)); };
    const int gx = dim(0), gy = dim(1), gz = dim(2);
    auto cellIdx = [&](int x, int y, int z) { return (z * gy + y) * gx + x; };
    auto clampi = [](int v, int m) { return v < 0 ? 0 : (v >= m ? m - 1 : v); };

    // Rasterize each triangle: barycentric samples at ~cell density mark every cell the
    // leaf card crosses. Samples outside the robust AABB (stray geometry) are dropped.
    xr_vector<u8> occ((size_t)gx * gy * gz, 0);
    auto mark = [&](const Fvector& p) {
        if (p.x < lo.x || p.y < lo.y || p.z < lo.z || p.x > hi.x || p.y > hi.y || p.z > hi.z) return;
        const int x = clampi((int)((p.x - lo.x) * inv), gx);
        const int y = clampi((int)((p.y - lo.y) * inv), gy);
        const int z = clampi((int)((p.z - lo.z) * inv), gz);
        occ[cellIdx(x, y, z)] = 1u;
    };
    for (u32 t = 0; t + 2 < nC; t += 3) {
        const Fvector &a = tris[t], &b = tris[t + 1], &c = tris[t + 2];
        const float eab = a.distance_to(b), eac = a.distance_to(c), ebc = b.distance_to(c);
        const int S = (std::min)(48, (std::max)(1, (int)ceilf((std::max)(eab, (std::max)(eac, ebc)) * inv)));
        for (int u = 0; u <= S; ++u)
        for (int v = 0; v <= S - u; ++v) {
            const float fu = (float)u / (float)S, fv = (float)v / (float)S;
            Fvector p = a;
            p.x += (b.x - a.x) * fu + (c.x - a.x) * fv;
            p.y += (b.y - a.y) * fu + (c.y - a.y) * fv;
            p.z += (b.z - a.z) * fu + (c.z - a.z) * fv;
            mark(p);
        }
    }
    auto occAt = [&](int x, int y, int z) -> bool {
        if (x < 0 || y < 0 || z < 0 || x >= gx || y >= gy || z >= gz) return false;
        return occ[cellIdx(x, y, z)] != 0;
    };

    // Emit exposed faces only. Per face: 4 verts + 2 tris (CULL_MODE_NONE world pipelines →
    // winding doesn't matter for depth). Corner positions in local mesh space.
    const u32 vbase0 = (u32)hullVerts.size();
    const u32 ibase0 = (u32)hullIdx.size();
    // 6 face normals; each face's 4 corner offsets (unit cube corners of the +face).
    static const int nrm[6][3] = { {-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1} };
    static const int corner[6][4][3] = {
        { {0,0,0},{0,1,0},{0,1,1},{0,0,1} },   // -X
        { {1,0,0},{1,0,1},{1,1,1},{1,1,0} },   // +X
        { {0,0,0},{0,0,1},{1,0,1},{1,0,0} },   // -Y
        { {0,1,0},{1,1,0},{1,1,1},{0,1,1} },   // +Y
        { {0,0,0},{1,0,0},{1,1,0},{0,1,0} },   // -Z
        { {0,0,1},{0,1,1},{1,1,1},{1,0,1} },   // +Z
    };
    for (int z = 0; z < gz; ++z)
    for (int y = 0; y < gy; ++y)
    for (int x = 0; x < gx; ++x) {
        if (!occ[cellIdx(x, y, z)]) continue;
        for (int f = 0; f < 6; ++f) {
            if (occAt(x + nrm[f][0], y + nrm[f][1], z + nrm[f][2])) continue;   // interior face → skip
            const u16 q = (u16)(hullVerts.size() - vbase0);   // base of this quad's 4 verts
            for (int cc = 0; cc < 4; ++cc) {
                Fvector p{ lo.x + float(x + corner[f][cc][0]) * cell,
                           lo.y + float(y + corner[f][cc][1]) * cell,
                           lo.z + float(z + corner[f][cc][2]) * cell };
                hullVerts.push_back(p);
            }
            hullIdx.push_back(q); hullIdx.push_back((u16)(q + 1)); hullIdx.push_back((u16)(q + 2));
            hullIdx.push_back(q); hullIdx.push_back((u16)(q + 2)); hullIdx.push_back((u16)(q + 3));
        }
    }
    return (u32)hullIdx.size() - ibase0;
}

// Deterministic per-cell hash → color/size variety (stable across bakes).
static inline u32 VoxHash(int x, int y, int z)
{
    u32 h = (u32)(x * 73856093) ^ (u32)(y * 19349663) ^ (u32)(z * 83492791);
    h ^= h >> 13; h *= 0x85ebca6bu; h ^= h >> 16;
    return h;
}

// ============================================================================
// Crown alpha map — CPU-side alpha channel of the foliage diffuse, sampled by
// the voxel bake so cells are marked by actual LEAVES, not by the leaf CARDS.
// A stalker crown is a handful of big quads whose leaves live entirely in the
// texture alpha; rasterizing the quads alpha-blind marks every cell the card
// crosses and the voxel union reads as one solid blob at any grid resolution.
// With the alpha test the fine grids get real gaps where the card is
// transparent — walking up to a tree, the voxel shadow dapples like foliage
// before the crossfade hands it to the true alpha-tested crown (the same idea
// as UE's per-voxel ray-traced Coverage, Cluster.cpp bVoxelOpacity).
// Decodes the alpha block of BC1/BC2/BC3 only (~all stalker foliage); any
// other format falls back to the alpha-blind bake for that species.
// ============================================================================
// Leaf-coverage nibble 1–15 from the bake sample counters (opaque hits / all card
// samples in the cell). Shared by the visual voxel cloud and the shadow bricks.
static u32 Cov4(u32 hits, u32 total)
{
    return (std::min)(15u, (std::max)(1u, (hits * 15u + total / 2) / (std::max)(total, 1u)));
}

struct CrownAlphaMap
{
    u32 w = 0, h = 0;
    xr_vector<u8> a;
    bool ok() const { return w && h && a.size() == (size_t)w * h; }
    bool opaque(float u, float v, u8 aref) const
    {
        u -= floorf(u); v -= floorf(v);   // wrap (leaf cards tile freely)
        u32 x = (u32)(u * (float)w); if (x >= w) x = w - 1;
        u32 y = (u32)(v * (float)h); if (y >= h) y = h - 1;
        return a[(size_t)y * w + x] >= aref;
    }
};

static bool LoadCrownAlphaMap(const char* texName, CrownAlphaMap& out)
{
    if (!texName || !texName[0]) return false;
    string_path leaf, full;
    xr_strcpy(leaf, texName);
    xr_strcat(leaf, ".dds");
    FS.update_path(full, "$game_textures$", leaf);
    IReader* F = FS.r_open(full);
    if (!F) return false;

    bool okRead = false;
    do {
        u32 magic = 0;
        if (F->elapsed() < 4) break;
        F->r(&magic, 4);
        if (magic != 0x20534444u) break;   // "DDS "
        struct { u32 size, flags, height, width, pitch, depth, mipCount, res1[11];
                 struct { u32 size, flags, fourCC, bitCount, rM, gM, bM, aM; } pf;
                 u32 caps, caps2, caps3, caps4, res2; } H{};
        static_assert(sizeof(H) == 124, "DDS header must be 124 B");
        if (F->elapsed() < (int)sizeof(H)) break;
        F->r(&H, sizeof(H));

        constexpr u32 kDXT1 = 0x31545844u, kDXT3 = 0x33545844u, kDXT5 = 0x35545844u, kDX10 = 0x30315844u;
        u32 fourCC = (H.pf.flags & 0x4u) ? H.pf.fourCC : 0u;
        int mode = 0;                       // 1 = BC1 punch-through, 2 = BC2 explicit, 3 = BC3 interpolated
        if (fourCC == kDX10) {
            struct { u32 fmt, dim, misc, arr, misc2; } h10{};
            if (F->elapsed() < (int)sizeof(h10)) break;
            F->r(&h10, sizeof(h10));
            if      (h10.fmt == 71u || h10.fmt == 72u) mode = 1;   // BC1_UNORM(_SRGB)
            else if (h10.fmt == 74u || h10.fmt == 75u) mode = 2;   // BC2_UNORM(_SRGB)
            else if (h10.fmt == 77u || h10.fmt == 78u) mode = 3;   // BC3_UNORM(_SRGB)
            else break;                                            // BC7 etc. — no cheap alpha decode
        }
        else if (fourCC == kDXT1) mode = 1;
        else if (fourCC == kDXT3) mode = 2;
        else if (fourCC == kDXT5) mode = 3;
        else break;                                                // uncompressed — rare for foliage
        const u32 blockBytes = (mode == 1) ? 8u : 16u;

        // Walk mips down to ≤256/axis — leaf clumps survive the downfilter and the
        // decode stays trivially cheap (a 256² alpha grid per unique texture).
        u32 w = (std::max)(1u, H.width), h = (std::max)(1u, H.height);
        u32 mips = (H.flags & 0x20000u) ? (std::max)(1u, H.mipCount) : 1u;
        while (mips > 1 && (std::max)(w, h) > 256u) {
            F->advance(((w + 3) / 4) * ((h + 3) / 4) * blockBytes);
            w = (std::max)(1u, w >> 1); h = (std::max)(1u, h >> 1);
            --mips;
        }
        if ((std::max)(w, h) > 2048u) break;   // mipless oversize oddity — not worth a 16 MB+ cache entry
        const u32 bw = (w + 3) / 4, bh = (h + 3) / 4;
        const size_t bytes = (size_t)bw * bh * blockBytes;
        if ((size_t)F->elapsed() < bytes) break;
        xr_vector<u8> blocks(bytes);
        F->r(blocks.data(), (int)bytes);

        out.w = w; out.h = h;
        out.a.assign((size_t)w * h, 255);
        for (u32 by = 0; by < bh; ++by)
        for (u32 bx = 0; bx < bw; ++bx) {
            const u8* b = &blocks[((size_t)by * bw + bx) * blockBytes];
            u8 al[16];
            if (mode == 1) {
                const u16 c0 = (u16)(b[0] | (b[1] << 8)), c1 = (u16)(b[2] | (b[3] << 8));
                const u32 idx = (u32)b[4] | ((u32)b[5] << 8) | ((u32)b[6] << 16) | ((u32)b[7] << 24);
                const bool hasA = c0 <= c1;
                for (int t = 0; t < 16; ++t)
                    al[t] = (hasA && ((idx >> (2 * t)) & 3u) == 3u) ? 0 : 255;
            } else if (mode == 2) {
                for (int t = 0; t < 16; ++t) {
                    const u8 nib = (u8)((b[t >> 1] >> ((t & 1) * 4)) & 0xFu);
                    al[t] = (u8)(nib * 17u);
                }
            } else {
                const u8 a0 = b[0], a1 = b[1];
                u64 bits = 0;
                for (int i = 0; i < 6; ++i) bits |= (u64)b[2 + i] << (8 * i);
                u8 lut[8];
                lut[0] = a0; lut[1] = a1;
                if (a0 > a1) for (int i = 1; i < 7; ++i) lut[1 + i] = (u8)(((7 - i) * a0 + i * a1) / 7);
                else {
                    for (int i = 1; i < 5; ++i) lut[1 + i] = (u8)(((5 - i) * a0 + i * a1) / 5);
                    lut[6] = 0; lut[7] = 255;
                }
                for (int t = 0; t < 16; ++t) al[t] = lut[(bits >> (3 * t)) & 7u];
            }
            for (int t = 0; t < 16; ++t) {
                const u32 x = bx * 4 + (t & 3), y = by * 4 + (t >> 2);
                if (x < w && y < h) out.a[(size_t)y * w + x] = al[t];
            }
        }
        okRead = true;
    } while (false);

    FS.r_close(F);
    if (!okRead) { out = CrownAlphaMap{}; return false; }
    return true;
}

// VOXEL CLOUD crown (the UE Nanite-foliage / "Witcher 4 demo" look): individual small
// colored cubes where the leaf cards actually are — NOT a merged watertight shell.
// The shell (BuildHullVoxels above) reads as one solid blob and is only right for the
// depth-only shadow caster; visually a crown is a SPARSE set of cubes with gaps, hue
// variety and darker interior. Per occupied cell we bake:
//   rgb — foliage palette pick by cell hash (greens/olive/rare brown), graded by
//         normalized height in the crown (canopy top lit, underside dark) and by
//         neighbor crowding (a cheap bake-time AO: dense interior → darker);
//   a   — size-jitter seed (the VS scales the cube 0.75–1.15) so the grid reads
//         organic instead of minecraft-regular.
// Cells fully enclosed on all 6 sides are skipped (never visible through the gaps at
// this density in practice — pure instance-count waste). `voxN` = cells across the
// crown's largest axis for THIS level. Returns the emitted voxel count; `cellOut`
// reports the voxel edge (mesh-local meters) for the renderer's screen-size LOD cut.
// `uvs`/`am` (both optional): per-corner UVs + the diffuse alpha map — samples on
// transparent parts of a leaf card don't mark cells, so fine grids grow real gaps
// where there are no leaves (the near-range "cubes at leaf positions" restructure);
// coarse grids stay ~solid since a big cell always holds some opaque texels.
// `outBricks` additionally receives the SHADOW representation of the same grid:
// 4×4×4-cell bricks with 64-bit occupancy masks (full + high-coverage core) — see
// GpuTreeBrick. Interior (fully-enclosed) cells stay IN the masks: they can never be
// the first hit of a ray, and the bits are free.
static u32 BuildVoxelCloud(const xr_vector<Fvector>& tris, const xr_vector<Fvector2>& uvs,
                           const CrownAlphaMap* am,
                           xr_vector<GpuTreeVoxel>& outVox, xr_vector<GpuTreeBrick>& outBricks,
                           u32 voxN, float& cellOut)
{
    cellOut = 0.f;
    const u32 nC = (u32)tris.size();
    Fvector lo, hi;
    if (!RobustTriBounds(tris, lo, hi)) return 0;
    const float ext[3] = { hi.x - lo.x, hi.y - lo.y, hi.z - lo.z };
    const float maxExt = (std::max)(ext[0], (std::max)(ext[1], ext[2]));
    if (maxExt < 1e-3f) return 0;
    const float cell = maxExt / float((std::max)(2u, voxN));
    const float inv  = 1.f / cell;
    // 144/axis cap only guards degenerate extents (voxN itself is clamped ≤128 by the
    // cvar); occupancy grid worst case 144³ = ~3 MB u8, transient.
    auto dim = [&](int a) { return (std::min)(144, (std::max)(1, (int)ceilf(ext[a] * inv) + 1)); };
    const int gx = dim(0), gy = dim(1), gz = dim(2);
    auto cellIdx = [&](int x, int y, int z) { return ((size_t)z * gy + y) * gx + x; };
    auto clampi = [](int v, int m) { return v < 0 ? 0 : (v >= m ? m - 1 : v); };

    // Occupancy by triangle rasterization — same scheme as the shell: barycentric
    // samples at ~cell density mark every cell a leaf card crosses. Per cell we also
    // count ALL card samples vs alpha-PASSING ones: their ratio is the cell's leaf
    // COVERAGE (how much of the card area in this cell is actual foliage), baked into
    // the voxel's low alpha nibble — the near-range sparsification kills cubes in
    // coverage order (card edges and thin wisps first, dense leaf clumps last).
    xr_vector<u8>  occ((size_t)gx * gy * gz, 0);
    xr_vector<u16> hitC((size_t)gx * gy * gz, 0), totC((size_t)gx * gy * gz, 0);
    auto mark = [&](const Fvector& p, bool pass) {
        if (p.x < lo.x || p.y < lo.y || p.z < lo.z || p.x > hi.x || p.y > hi.y || p.z > hi.z) return;
        const int x = clampi((int)((p.x - lo.x) * inv), gx);
        const int y = clampi((int)((p.y - lo.y) * inv), gy);
        const int z = clampi((int)((p.z - lo.z) * inv), gz);
        const size_t c = cellIdx(x, y, z);
        if (totC[c] < 0xFFFFu) ++totC[c];
        if (pass) { occ[c] = 1u; if (hitC[c] < 0xFFFFu) ++hitC[c]; }
    };
    // Alpha threshold matches the crown shadow FS (alphaRef 200/255) with a little
    // slack for the mip downfilter — the voxel gaps line up with the leaf shadow
    // the crossfade hands over to.
    constexpr u8 kARef = 160;
    const bool alphaAware = am && am->ok() && uvs.size() == tris.size();
    for (u32 t = 0; t + 2 < nC; t += 3) {
        const Fvector &a = tris[t], &b = tris[t + 1], &c = tris[t + 2];
        const float eab = a.distance_to(b), eac = a.distance_to(c), ebc = b.distance_to(c);
        // 2× sample density when alpha-testing: pass/fail per cell becomes a vote of
        // ~4 samples instead of ~1, so cell marking tracks the leaf clumps instead of
        // single-texel luck.
        const int S = (std::min)(96, (std::max)(1, (int)ceilf((std::max)(eab, (std::max)(eac, ebc)) * inv * (alphaAware ? 2.f : 1.f))));
        for (int u = 0; u <= S; ++u)
        for (int v = 0; v <= S - u; ++v) {
            const float fu = (float)u / (float)S, fv = (float)v / (float)S;
            bool pass = true;
            if (alphaAware) {
                const Fvector2 &ta = uvs[t], &tb = uvs[t + 1], &tc = uvs[t + 2];
                const float su = ta.x + (tb.x - ta.x) * fu + (tc.x - ta.x) * fv;
                const float sv = ta.y + (tb.y - ta.y) * fu + (tc.y - ta.y) * fv;
                pass = am->opaque(su, sv, kARef);
            }
            Fvector p = a;
            p.x += (b.x - a.x) * fu + (c.x - a.x) * fv;
            p.y += (b.y - a.y) * fu + (c.y - a.y) * fv;
            p.z += (b.z - a.z) * fu + (c.z - a.z) * fv;
            mark(p, pass);
        }
    }
    auto occAt = [&](int x, int y, int z) -> bool {
        if (x < 0 || y < 0 || z < 0 || x >= gx || y >= gy || z >= gz) return false;
        return occ[cellIdx(x, y, z)] != 0;
    };

    // Foliage palette — hue variety like the reference: mid/dark greens dominate,
    // olive and yellow-green sprinkled in, the odd brown (inner branches).
    static const float kPal[6][3] = {
        { 0.30f, 0.42f, 0.16f },   // mid green
        { 0.21f, 0.33f, 0.13f },   // dark green
        { 0.15f, 0.25f, 0.11f },   // deep green
        { 0.38f, 0.44f, 0.15f },   // olive
        { 0.47f, 0.44f, 0.17f },   // yellow-green
        { 0.30f, 0.23f, 0.12f },   // brown
    };
    static const u8 kPick[16] = { 0,0,0,0, 1,1,1, 2,2, 3,3, 4,4, 5, 0,1 };   // weighted

    const u32 base = (u32)outVox.size();
    const float invExtY = 1.f / (std::max)(ext[1], 1e-3f);
    for (int z = 0; z < gz; ++z)
    for (int y = 0; y < gy; ++y)
    for (int x = 0; x < gx; ++x) {
        if (!occ[cellIdx(x, y, z)]) continue;
        if (occAt(x - 1, y, z) && occAt(x + 1, y, z) && occAt(x, y - 1, z) &&
            occAt(x, y + 1, z) && occAt(x, y, z - 1) && occAt(x, y, z + 1)) continue;   // fully enclosed
        // Bake-time AO: 26-neighborhood crowding — interior/dense cells darker.
        int crowd = 0;
        for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
            if ((dx | dy | dz) && occAt(x + dx, y + dy, z + dz)) ++crowd;
        const u32   h    = VoxHash(x, y, z);
        const float* pc  = kPal[kPick[h & 15u]];
        const float tUp  = ((float)y + 0.5f) * cell * invExtY;            // 0 = crown bottom, 1 = top
        const float ao   = 1.f - 0.45f * ((float)crowd / 26.f);
        const float jit  = 0.85f + 0.30f * (float)((h >> 8) & 255u) / 255.f;
        const float br   = (0.70f + 0.40f * tUp) * ao * jit;
        auto ch = [&](float c) { return (u32)(std::min)(255.f, (std::max)(0.f, c * br * 255.f + 0.5f)); };
        // Alpha byte split: HIGH nibble = size-jitter seed, LOW nibble = leaf coverage
        // 1–15 (opaque-sample fraction; 15 everywhere on the alpha-blind fallback) —
        // the caster dissolves cubes in coverage order across the crossfade band.
        const size_t ci   = cellIdx(x, y, z);
        const u32    cov4 = Cov4(hitC[ci], totC[ci]);
        GpuTreeVoxel vx;
        vx.p    = Fvector{ lo.x + ((float)x + 0.5f) * cell,
                           lo.y + ((float)y + 0.5f) * cell,
                           lo.z + ((float)z + 0.5f) * cell };
        vx.rgba = ch(pc[0]) | (ch(pc[1]) << 8) | (ch(pc[2]) << 16) | ((((h >> 16) & 0xF0u) | cov4) << 24);
        outVox.push_back(vx);
    }

    // ---- SHADOW bricks: 4×4×4 cells → one 64-bit mask (+ high-coverage core mask).
    // A brick with an empty core simply vanishes at its band swap point — that IS the
    // "extras removed" stage, no fallback wanted.
    const size_t brickBase = outBricks.size();
    const int bgx = (gx + 3) / 4, bgy = (gy + 3) / 4, bgz = (gz + 3) / 4;
    for (int bz = 0; bz < bgz; ++bz)
    for (int by = 0; by < bgy; ++by)
    for (int bx = 0; bx < bgx; ++bx) {
        GpuTreeBrick bk{};
        u32 covSum = 0, covN = 0;
        for (int lz = 0; lz < 4; ++lz)
        for (int ly = 0; ly < 4; ++ly)
        for (int lx = 0; lx < 4; ++lx) {
            const int x = bx * 4 + lx, y = by * 4 + ly, z = bz * 4 + lz;
            if (x >= gx || y >= gy || z >= gz) continue;
            const size_t ci = cellIdx(x, y, z);
            if (!occ[ci]) continue;
            const u32 bit  = (u32)lx | ((u32)ly << 2) | ((u32)lz << 4);
            const u32 cov4 = Cov4(hitC[ci], totC[ci]);
            (bit < 32u ? bk.fullLo : bk.fullHi) |= 1u << (bit & 31u);
            if (cov4 >= 6u) (bit < 32u ? bk.coreLo : bk.coreHi) |= 1u << (bit & 31u);
            covSum += cov4; ++covN;
        }
        if (!(bk.fullLo | bk.fullHi)) continue;
        bk.p = Fvector{ lo.x + (float)(bx * 4) * cell,
                        lo.y + (float)(by * 4) * cell,
                        lo.z + (float)(bz * 4) * cell };
        const u32 h = VoxHash(bx, by, bz);
        bk.meta = ((covSum + covN / 2) / (std::max)(covN, 1u)) | ((h >> 8) & 0xFF0u);   // avgCov4 | hash8<<4
        outBricks.push_back(bk);
    }
    // Sort the slice by DESCENDING dissolve rank (the same 0.5·cov + 0.5·hash the VS
    // uses, both read from meta) → the band's survivors are always a PREFIX, so
    // vsm_hull_cmd can shrink vertexCount to ~fade·count instead of drawing every
    // brick and letting the VS kill most of them (the band double-cast was the whole
    // remaining dyn-tier brick cost on light scenes).
    std::sort(outBricks.begin() + brickBase, outBricks.end(),
              [](const GpuTreeBrick& a, const GpuTreeBrick& b) {
                  const u32 ra = ((a.meta & 0xFu) * 255u * 8u) / 15u + ((a.meta >> 4) & 0xFFu) * 8u;
                  const u32 rb = ((b.meta & 0xFu) * 255u * 8u) / 15u + ((b.meta >> 4) & 0xFFu) * 8u;
                  return ra > rb;
              });

    cellOut = cell;
    return (u32)outVox.size() - base;
}

}  // namespace

void CTreeManager::BuildMeshlets(const xr_vector<vkFTreeVisual*>& trees,
                                 const xr_vector<GpuTreeMeta>& meta)
{
    if (m_TotalCount == 0 || trees.size() < m_TotalCount || meta.size() < m_TotalCount) return;

    // A unique mesh = one (vb, ib, vBase, vCount, ib_first, index_count) span. Instances
    // of the same species share it, so we meshletize + read back geometry ONCE per span.
    struct MeshKey {
        VkBuffer vb, ib; u32 vBase, vCount, ibFirst, idxCount, stride;
        bool operator==(const MeshKey& o) const {
            return vb == o.vb && ib == o.ib && vBase == o.vBase && vCount == o.vCount &&
                   ibFirst == o.ibFirst && idxCount == o.idxCount && stride == o.stride;
        }
    };
    xr_vector<MeshKey>            keys;
    xr_vector<GpuTreeMeshletRange> uniqRange;   // meshlet slice per unique mesh
    xr_vector<u32>               treeUniq(m_TotalCount, ~0u);   // tree -> unique-mesh index

    // Output accumulators (uploaded once at the end).
    xr_vector<GpuMeshlet> meshlets;   meshlets.reserve(4096);
    xr_vector<u16>        mIndices;   mIndices.reserve(1 << 18);

    // Crown-HULL accumulators (r_vsm_tree_hull) — baked here to reuse the same unique-mesh
    // readback. Always built (a few hundred KB total) so the cvar toggles live at runtime.
    xr_vector<Fvector>         hullVerts;  hullVerts.reserve(1 << 14);
    xr_vector<u16>             hullIdx;    hullIdx.reserve(1 << 15);
    xr_vector<GpuTreeHullInfo> uniqHull;      // hull range per unique mesh (→ shadow caster)
    xr_vector<GpuTreeVoxel>    voxData;    voxData.reserve(1 << 16);   // voxel-cloud viewmode cubes
    xr_vector<TreeVoxLod>      uniqVoxLods;  // kHullLods voxel ranges per unique mesh
    xr_vector<GpuTreeBrick>    brickData;  brickData.reserve(1 << 13); // shadow bricks (4³ masks)
    xr_vector<TreeVoxLod>      uniqBrickLods;
    // Diffuse alpha maps for the alpha-aware voxel bake, one per unique texture
    // (species share textures). The returned pointer is only valid until the next
    // lookup (vector may reallocate) — consumed within the same tree iteration.
    xr_vector<std::pair<xr_string, CrownAlphaMap>> alphaCache;
    u32 alphaSpecies = 0, blindSpecies = 0;   // diag: voxel bakes with/without texture alpha
    auto crownAlpha = [&](const char* nm) -> const CrownAlphaMap* {
        if (!nm || !nm[0]) return nullptr;
        for (auto& e : alphaCache) if (e.first == nm) return e.second.ok() ? &e.second : nullptr;
        alphaCache.emplace_back(nm, CrownAlphaMap{});
        LoadCrownAlphaMap(nm, alphaCache.back().second);
        return alphaCache.back().second.ok() ? &alphaCache.back().second : nullptr;
    };

    for (u32 i = 0; i < m_TotalCount; ++i)
    {
        const vkFTreeVisual* t = trees[i];
        if (!t || !t->m_mesh.p_rm_Vertices || !t->m_mesh.p_rm_Indices) continue;
        MeshKey k{ t->m_mesh.p_rm_Vertices->GetHandle(), t->m_mesh.p_rm_Indices->GetHandle(),
                   t->m_mesh.vBase, t->m_mesh.vCount, meta[i].ib_first, meta[i].index_count,
                   t->m_mesh.vStride };
        if (k.vCount == 0 || k.idxCount < 3 || k.stride < 12) continue;

        // Already meshletized?
        u32 u = ~0u;
        for (u32 j = 0; j < keys.size(); ++j) if (keys[j] == k) { u = j; break; }
        if (u != ~0u) { treeUniq[i] = u; continue; }

        // --- Read back this mesh's positions + indices (relative to vBase). ---
        xr_vector<u8> vbytes, ibbytes;
        if (!ReadbackRange(k.vb, (VkDeviceSize)k.vBase * k.stride, (VkDeviceSize)k.vCount * k.stride, vbytes)) continue;
        if (!ReadbackRange(k.ib, (VkDeviceSize)k.ibFirst * sizeof(u16), (VkDeviceSize)k.idxCount * sizeof(u16), ibbytes)) continue;
        const u16* idx = (const u16*)ibbytes.data();
        auto pos = [&](u32 v) -> Fvector {
            const float* f = (const float*)(vbytes.data() + (size_t)v * k.stride);
            return Fvector{ f[0], f[1], f[2] };
        };
        const u32 triCount = k.idxCount / 3;

        // --- Per-triangle centroids + their AABB (Morton normalization box). ---
        xr_vector<Fvector> cen(triCount);
        Fvector cmin{ FLT_MAX, FLT_MAX, FLT_MAX }, cmax{ -FLT_MAX, -FLT_MAX, -FLT_MAX };
        for (u32 tri = 0; tri < triCount; ++tri) {
            const Fvector a = pos(idx[tri * 3 + 0]), b = pos(idx[tri * 3 + 1]), c = pos(idx[tri * 3 + 2]);
            Fvector ctr{ (a.x + b.x + c.x) / 3.f, (a.y + b.y + c.y) / 3.f, (a.z + b.z + c.z) / 3.f };
            cen[tri] = ctr;
            cmin.min(ctr); cmax.max(ctr);
        }
        const float ex = (std::max)(cmax.x - cmin.x, 1e-4f);
        const float ey = (std::max)(cmax.y - cmin.y, 1e-4f);
        const float ez = (std::max)(cmax.z - cmin.z, 1e-4f);

        // --- Morton-sort triangle order. ---
        xr_vector<u32> order(triCount);
        for (u32 tri = 0; tri < triCount; ++tri) order[tri] = tri;
        xr_vector<u32> code(triCount);
        for (u32 tri = 0; tri < triCount; ++tri) {
            const u32 qx = (u32)(std::min)(1023.f, (std::max)(0.f, (cen[tri].x - cmin.x) / ex * 1023.f));
            const u32 qy = (u32)(std::min)(1023.f, (std::max)(0.f, (cen[tri].y - cmin.y) / ey * 1023.f));
            const u32 qz = (u32)(std::min)(1023.f, (std::max)(0.f, (cen[tri].z - cmin.z) / ez * 1023.f));
            code[tri] = Morton3(qx, qy, qz);
        }
        std::sort(order.begin(), order.end(), [&](u32 a, u32 b) { return code[a] < code[b]; });

        // --- Slice into clusters; local sphere per cluster; emit reordered indices. ---
        const u32 base = (u32)meshlets.size();
        for (u32 s = 0; s < triCount; s += kMeshletTris) {
            const u32 e = (std::min)(triCount, s + kMeshletTris);
            const u32 firstIndex = (u32)mIndices.size();
            Fvector bmin{ FLT_MAX, FLT_MAX, FLT_MAX }, bmax{ -FLT_MAX, -FLT_MAX, -FLT_MAX };
            for (u32 o = s; o < e; ++o) {
                const u32 tri = order[o];
                for (u32 w = 0; w < 3; ++w) {
                    const u16 vi = idx[tri * 3 + w];
                    mIndices.push_back(vi);
                    const Fvector p = pos(vi);
                    bmin.min(p); bmax.max(p);
                }
            }
            Fvector ctr{ (bmin.x + bmax.x) * 0.5f, (bmin.y + bmax.y) * 0.5f, (bmin.z + bmax.z) * 0.5f };
            float r2 = 0.f;
            for (u32 o = s; o < e; ++o) {
                const u32 tri = order[o];
                for (u32 w = 0; w < 3; ++w) r2 = (std::max)(r2, ctr.distance_to_sqr(pos(idx[tri * 3 + w])));
            }
            GpuMeshlet ml{};
            ml.center = ctr; ml.radius = sqrtf(r2);
            ml.first_index = firstIndex; ml.index_count = (e - s) * 3;
            meshlets.push_back(ml);
        }

        // --- Crown hull: FOLIAGE meshes only (windClass 2). Trunks/props never enter the
        // hull tier (crowns-only gate in VsmUpdateNearSet), so baking them would be waste.
        if (i < m_WindClassCPU.size() && m_WindClassCPU[i] == 2)
        {
            // Per-vertex weight = incident triangle area: leaf cards (big quads) dominate,
            // twig strips barely register → lobes centre on the leaf TUFTS.
            xr_vector<float> vertArea(k.vCount, 0.f);
            for (u32 tri = 0; tri < triCount; ++tri) {
                const u16 i0 = idx[tri * 3 + 0], i1 = idx[tri * 3 + 1], i2 = idx[tri * 3 + 2];
                if (i0 >= k.vCount || i1 >= k.vCount || i2 >= k.vCount) continue;
                const Fvector a = pos(i0), b = pos(i1), cc = pos(i2);
                Fvector e1{ b.x - a.x, b.y - a.y, b.z - a.z }, e2{ cc.x - a.x, cc.y - a.y, cc.z - a.z }, cr;
                cr.crossproduct(e1, e2);
                const float third = cr.magnitude() * 0.5f / 3.f;
                vertArea[i0] += third; vertArea[i1] += third; vertArea[i2] += third;
            }
            xr_vector<Fvector> pts; pts.reserve(k.vCount);
            xr_vector<float>   wts; wts.reserve(k.vCount);
            for (u32 v = 0; v < k.vCount; ++v)
                if (vertArea[v] > 0.f) { pts.push_back(pos(v)); wts.push_back(vertArea[v]); }
            // Flat triangle corners for voxel rasterization (fills the crown from leaf
            // cards, not sparse verts). Lobes still use the pts/wts point cloud.
            // UVs ride along (SHORT2 SSCALED @ tcOffset, ×1/2048 — same decode as the
            // tree VS) so the voxel bake can alpha-test against the diffuse.
            const u32  tcOff = t->m_mesh.tcOffset;
            const bool hasUV = tcOff + 2 * sizeof(s16) <= k.stride;
            auto uvAt = [&](u32 v) -> Fvector2 {
                const s16* s = (const s16*)(vbytes.data() + (size_t)v * k.stride + tcOff);
                return Fvector2{ (float)s[0] / 2048.f, (float)s[1] / 2048.f };
            };
            xr_vector<Fvector>  tris;  tris.reserve((size_t)triCount * 3);
            xr_vector<Fvector2> triUV; if (hasUV) triUV.reserve((size_t)triCount * 3);
            for (u32 tri = 0; tri < triCount; ++tri) {
                const u16 i0 = idx[tri * 3 + 0], i1 = idx[tri * 3 + 1], i2 = idx[tri * 3 + 2];
                if (i0 >= k.vCount || i1 >= k.vCount || i2 >= k.vCount) continue;
                tris.push_back(pos(i0)); tris.push_back(pos(i1)); tris.push_back(pos(i2));
                if (hasUV) { triUV.push_back(uvAt(i0)); triUV.push_back(uvAt(i1)); triUV.push_back(uvAt(i2)); }
            }
            const CrownAlphaMap* am = hasUV && t->m_pWorldMaterial
                                    ? crownAlpha(t->m_pWorldMaterial->name.c_str()) : nullptr;
            if (ps_r_vsm_tree_hull_vox > 0) { am ? ++alphaSpecies : ++blindSpecies; }
            // SHADOW hull — one merged watertight shell at a modest resolution (depth-only
            // caster wants a cheap SOLID silhouette, not the fine visual cubes). Kept
            // deliberately close to the pre-voxel default (~12–16 cells across) no matter
            // how fine the visual voxels go. Lobes when the voxel cvar is off.
            GpuTreeHullInfo hs{};
            hs.ib_first = (u32)hullIdx.size();
            hs.vb_first = (u32)hullVerts.size();
            if (ps_r_vsm_tree_hull_vox > 0) {
                const u32 shellN = (std::min)(20u, (std::max)(8u, (u32)ps_r_vsm_tree_hull_vox / 4u));
                hs.ib_count = BuildHullVoxels(tris, hullVerts, hullIdx, shellN);
            } else {
                hs.ib_count = BuildHullLobes(pts, wts, hullVerts, hullIdx);
            }
            uniqHull.push_back(hs);

            // VISUAL voxel cloud (UE Nanite-foliage style) — kHullLods levels of individual
            // colored cubes; voxel edge DOUBLES per level (resolution halves), exactly the
            // Nanite DAG progression. Level 0 = r_vsm_tree_hull_vox cells across the crown.
            for (u32 l = 0; l < CTreeManager::kHullLods; ++l) {
                TreeVoxLod vl{}, bl{};
                vl.first = (u32)voxData.size();
                bl.first = (u32)brickData.size();
                if (ps_r_vsm_tree_hull_vox > 0) {
                    const u32 voxN = (std::max)(4u, (u32)ps_r_vsm_tree_hull_vox >> l);
                    vl.count = BuildVoxelCloud(tris, triUV, am, voxData, brickData, voxN, vl.size);
                    bl.count = (u32)brickData.size() - bl.first;
                    bl.size  = vl.size;
                }
                uniqVoxLods.push_back(vl);
                uniqBrickLods.push_back(bl);
            }
        }
        else {
            uniqHull.push_back(GpuTreeHullInfo{ 0, 0, 0, 0 });   // no hull → tree keeps its real mesh
            for (u32 l = 0; l < CTreeManager::kHullLods; ++l) {
                uniqVoxLods.push_back(TreeVoxLod{});
                uniqBrickLods.push_back(TreeVoxLod{});
            }
        }

        u = (u32)keys.size();
        keys.push_back(k);
        uniqRange.push_back({ base, (u32)meshlets.size() - base });
        treeUniq[i] = u;
    }

    if (meshlets.empty() || mIndices.empty()) { Msg("[VK Trees] Meshlets: nothing built"); return; }

    // Per-tree range (index-aligned with transforms/meta); trees with no mesh get {0,0}.
    xr_vector<GpuTreeMeshletRange> treeRange(m_TotalCount, GpuTreeMeshletRange{ 0, 0 });
    for (u32 i = 0; i < m_TotalCount; ++i)
        if (treeUniq[i] != ~0u) treeRange[i] = uniqRange[treeUniq[i]];

    // Upload: GpuMeshlet[] (SSBO), dedicated meshlet IB (u16), per-tree ranges (SSBO).
    UploadDeviceLocal(m_MeshletBuffer, meshlets.data(), meshlets.size() * sizeof(GpuMeshlet), 0);
    UploadDeviceLocal(m_MeshletIndexBuffer, mIndices.data(), mIndices.size() * sizeof(u16),
                      VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
    UploadDeviceLocal(m_TreeMeshletRangeBuffer, treeRange.data(),
                      treeRange.size() * sizeof(GpuTreeMeshletRange), 0);

    m_MeshletTotal      = (u32)meshlets.size();
    m_MeshletIndexTotal = (u32)mIndices.size();
    m_MeshletsReady     = true;
    Msg("[VK Trees] Meshlets: %u clusters (%u tris/cluster) from %u unique meshes, %u idx (%u KB) — %u instances",
        m_MeshletTotal, kMeshletTris, (u32)keys.size(), m_MeshletIndexTotal,
        (u32)(m_MeshletIndexTotal * sizeof(u16) / 1024), m_TotalCount);

    // --- Crown-HULL upload (r_vsm_tree_hull): tight vec3 VB + u16 IB + per-tree info. ---
    if (!hullVerts.empty() && !hullIdx.empty()) {
        xr_vector<GpuTreeHullInfo> treeHull(m_TotalCount, GpuTreeHullInfo{ 0, 0, 0, 0 });
        for (u32 i = 0; i < m_TotalCount; ++i)
            if (treeUniq[i] != ~0u) treeHull[i] = uniqHull[treeUniq[i]];
        // Per-tree × kHullLods voxel-cloud table for the graded viewmode + the parallel
        // brick table for the shadow caster.
        m_VoxLodInfoCPU.assign((size_t)m_TotalCount * CTreeManager::kHullLods, TreeVoxLod{});
        m_BrickLodInfoCPU.assign((size_t)m_TotalCount * CTreeManager::kHullLods, TreeVoxLod{});
        for (u32 i = 0; i < m_TotalCount; ++i)
            if (treeUniq[i] != ~0u)
                for (u32 l = 0; l < CTreeManager::kHullLods; ++l) {
                    m_VoxLodInfoCPU[(size_t)i * CTreeManager::kHullLods + l] =
                        uniqVoxLods[(size_t)treeUniq[i] * CTreeManager::kHullLods + l];
                    m_BrickLodInfoCPU[(size_t)i * CTreeManager::kHullLods + l] =
                        uniqBrickLods[(size_t)treeUniq[i] * CTreeManager::kHullLods + l];
                }
        UploadDeviceLocal(m_HullVB, hullVerts.data(), hullVerts.size() * sizeof(Fvector),
                          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        UploadDeviceLocal(m_HullIB, hullIdx.data(), hullIdx.size() * sizeof(u16),
                          VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        UploadDeviceLocal(m_HullInfoBuffer, treeHull.data(), treeHull.size() * sizeof(GpuTreeHullInfo), 0);
        m_HullInfoCPU = treeHull;   // shadow-shell ranges (also the debug fallback overlay)
        m_VoxTotal = (u32)voxData.size();
        if (m_VoxTotal)
            UploadDeviceLocal(m_VoxVB, voxData.data(), voxData.size() * sizeof(GpuTreeVoxel),
                              VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
        m_BrickTotal = (u32)brickData.size();
        if (m_BrickTotal)
            UploadDeviceLocal(m_BrickVB, brickData.data(), brickData.size() * sizeof(GpuTreeBrick), 0);
        m_HullLobeTotal = (u32)hullIdx.size() / 3;   // total hull triangles (shape-agnostic: lobes or voxels)
        m_HullDataReady = true;
        Msg("[VK Trees] Crown hulls: %s, %u tris / %u species, %u verts (%u KB); voxel cloud %u cubes + %u shadow bricks x%u LODs (%u+%u KB), alpha-aware %u/%u species — r_vsm_tree_hull ready",
            ps_r_vsm_tree_hull_vox > 0 ? "VOXEL shell" : "PCA lobes",
            m_HullLobeTotal, (u32)keys.size(), (u32)hullVerts.size(),
            (u32)((hullVerts.size() * sizeof(Fvector) + hullIdx.size() * sizeof(u16)) / 1024),
            m_VoxTotal, m_BrickTotal, CTreeManager::kHullLods,
            (u32)(voxData.size() * sizeof(GpuTreeVoxel) / 1024),
            (u32)(brickData.size() * sizeof(GpuTreeBrick) / 1024),
            alphaSpecies, alphaSpecies + blindSpecies);
    }
}

void CTreeManager::CreateIndirectBuffers()
{
    const VkDeviceSize indirectSize =
        (VkDeviceSize)m_Groups.size() * m_MaxGroupMeshCount *
        sizeof(VkDrawIndexedIndirectCommand);
    m_TreeIndirectBuffer = xr_new<CVulkanBuffer>();
    m_TreeIndirectBuffer->Create(indirectSize,
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT  |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

    const VkDeviceSize countSize = (VkDeviceSize)m_Groups.size() * sizeof(u32);
    m_TreeDrawCountBuffer = xr_new<CVulkanBuffer>();
    m_TreeDrawCountBuffer->Create(countSize,
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT  |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT    |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,   // forward drawn-count readback (r_profiler)
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

    Msg("[VK Trees] Indirect buffer: %u KB, draw-count: %u B",
        (u32)(indirectSize / 1024), (u32)countSize);
}

// ============================================================================
// One COMBINED_IMAGE_SAMPLER per unique diffuse texture. Layout is reused by
// Session B's graphics pipeline (set=1).
// ============================================================================
void CTreeManager::CreateTextureDescriptors(const xr_vector<VkImageView>& uniqueViews)
{
    if (uniqueViews.empty()) return;

    // Sampler — REPEAT addressing (UVs are SHORT4 quantized with 16:1 tile).
    {
        VkSamplerCreateInfo sci{};
        sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter    = VK_FILTER_LINEAR;
        sci.minFilter    = VK_FILTER_LINEAR;
        sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.maxLod       = VK_LOD_CLAMP_NONE;
        sci.anisotropyEnable = VK_FALSE;
        vkCreateSampler(VulkanHW.m_Device, &sci, nullptr, &m_TexSampler);
    }

    // Layout: 1 binding, COMBINED_IMAGE_SAMPLER (fragment).
    {
        VkDescriptorSetLayoutBinding b{};
        b.binding         = 0;
        b.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b.descriptorCount = 1;
        b.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo ci{};
        ci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        ci.bindingCount = 1;
        ci.pBindings    = &b;
        vkCreateDescriptorSetLayout(VulkanHW.m_Device, &ci, nullptr, &m_TexDescLayout);
    }

    // Pool.
    const u32 setCount = (u32)uniqueViews.size();
    {
        VkDescriptorPoolSize sz{};
        sz.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        sz.descriptorCount = setCount;

        VkDescriptorPoolCreateInfo ci{};
        ci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        ci.poolSizeCount = 1;
        ci.pPoolSizes    = &sz;
        ci.maxSets       = setCount;
        vkCreateDescriptorPool(VulkanHW.m_Device, &ci, nullptr, &m_TexDescPool);
    }

    // Allocate + write.
    m_TexDescSets.resize(setCount);
    for (u32 i = 0; i < setCount; ++i)
    {
        VkDescriptorSetAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = m_TexDescPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &m_TexDescLayout;
        vkAllocateDescriptorSets(VulkanHW.m_Device, &ai, &m_TexDescSets[i]);

        VkDescriptorImageInfo ii{};
        ii.sampler     = m_TexSampler;
        ii.imageView   = uniqueViews[i];
        ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet w{};
        w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet          = m_TexDescSets[i];
        w.dstBinding      = 0;
        w.descriptorCount = 1;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo      = &ii;
        vkUpdateDescriptorSets(VulkanHW.m_Device, 1, &w, 0, nullptr);
    }
}

// ============================================================================
// Destroy — free all GPU resources. Safe to call on partially-built or empty
// state (Build() may early-return on no trees).
// ============================================================================
void CTreeManager::Destroy()
{
    DestroySessionB();

    auto destroyBuf = [](CVulkanBuffer*& b) {
        if (b) { b->Destroy(); xr_delete(b); }
    };
    destroyBuf(m_TreeMetadataBuffer);
    destroyBuf(m_TreeTransformsBuffer);
    destroyBuf(m_TreeIndirectBuffer);
    destroyBuf(m_TreeDrawCountBuffer);
    destroyBuf(m_FwdCountRB);
    m_FwdCountPtr = nullptr;
    destroyBuf(m_SeenBits);
    destroyBuf(m_SeenBitsRB);
    m_SeenBitsPtr = nullptr;
    destroyBuf(m_MeshletBuffer);           // Phase A meshlet clusters
    destroyBuf(m_MeshletIndexBuffer);
    destroyBuf(m_TreeMeshletRangeBuffer);
    m_MeshletsReady = false;
    m_MeshletTotal = m_MeshletIndexTotal = 0;
    destroyBuf(m_HullVB);                  // crown-hull bake (r_vsm_tree_hull)
    destroyBuf(m_HullIB);
    destroyBuf(m_HullInfoBuffer);
    destroyBuf(m_VoxVB);
    destroyBuf(m_BrickVB);
    m_HullInfoCPU.clear();
    m_VoxLodInfoCPU.clear();
    m_BrickLodInfoCPU.clear();
    m_WindClassCPU.clear();
    m_HullDataReady = false;
    m_HullLobeTotal = 0;
    m_VoxTotal      = 0;
    m_BrickTotal    = 0;

    m_TexDescSets.clear();
    if (m_TexDescPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(VulkanHW.m_Device, m_TexDescPool, nullptr);
        m_TexDescPool = VK_NULL_HANDLE;
    }
    if (m_TexDescLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(VulkanHW.m_Device, m_TexDescLayout, nullptr);
        m_TexDescLayout = VK_NULL_HANDLE;
    }
    if (m_TexSampler != VK_NULL_HANDLE) {
        vkDestroySampler(VulkanHW.m_Device, m_TexSampler, nullptr);
        m_TexSampler = VK_NULL_HANDLE;
    }

    m_Groups.clear();
    m_TotalCount = 0;
    m_MaxGroupMeshCount = 0;
    m_bBuilt = false;
}

}  // namespace VK
