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
#include "vk_buffer.h"
#include "HW_Vulkan.h"
#include "CRender_Vulkan.h"

#include <algorithm>

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
        VkImageView view = trees[i]->m_pWorldMaterial->view;
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
        x._pad0 = x._pad1 = 0;
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
    CreateIndirectBuffers();
    CreateTextureDescriptors(uniqueViews);

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
    dst->Create(size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | extraUsage,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

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
}

void CTreeManager::UploadTransforms(const xr_vector<GpuTreeInstance>& xforms)
{
    UploadDeviceLocal(m_TreeTransformsBuffer, xforms.data(),
                      xforms.size() * sizeof(GpuTreeInstance), 0);
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
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
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
