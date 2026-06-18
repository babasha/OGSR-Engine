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
#include "vk_world_gpu.h"
#include "CRender_Vulkan.h"     // RImplementation.Visuals
#include "vk_Visual.h"          // vkFVisual, vkFHierrarhyVisual, m_mesh, m_pWorldMaterial
#include "vk_world_material.h"  // WorldMaterial / WorldMaterialCache
#include "vk_render_queue.h"    // VK::RenderQueue (SubmitCpuMeshes)
#include "vk_buffer.h"          // CVulkanBuffer
#include "vk_pipeline_cache.h"  // PipelineCache pipelines/layouts
#include "vk_env_light.h"       // EnvLight::GetCurrentSet (set 1, terrain snow-depth)
#include "vk_shaders.h"         // g_ShaderManager
#include "vk_cull.h"            // VK::ExtractFrustumPlanes
#include <algorithm>

namespace VK { namespace WorldGPU {

namespace {

struct GpuMeshMeta {       // std430, 32 B — matches world_cull.comp `Meta`
    Fvector sphere_P; float sphere_R;
    u32 index_count; u32 ib_first; u32 first_vertex; u32 group;
};

// One indirect-draw batch: a run of meshes sharing material + pipeline key + VB/IB,
// so the draw binds those once and issues a single vkCmdDrawIndexedIndirectCount.
struct Group {
    WorldMaterial* mat;
    bool        terrain;
    u32         stride, tcOffset;
    VkBuffer    vb, ib;
    VkIndexType iType;
    u32         meshOffset, meshCount;
};

struct CullPush { Fvector4 planes[6]; u32 numGroups, maxGroupMesh, total, _pad; };

bool s_built = false;
u32  s_total = 0;
u32  s_maxGroupMesh = 0;
xr_vector<Group> s_groups;
xr_vector<vkRender_Visual*> s_meshSet;   // pointer-sorted, for InSet() (CPU-queue exclusion)
xr_vector<vkFVisual*> s_cpuMeshes;       // non-GPU static leaves (wmark/tess/no-diffuse) — CPU draws these

CVulkanBuffer* s_meta     = nullptr;
CVulkanBuffer* s_indirect = nullptr;   // numGroups * maxGroupMesh cmds
CVulkanBuffer* s_count    = nullptr;   // numGroups u32

VkDescriptorSetLayout s_setL  = VK_NULL_HANDLE;
VkDescriptorPool      s_pool  = VK_NULL_HANDLE;
VkDescriptorSet       s_set   = VK_NULL_HANDLE;
VkPipelineLayout      s_cullLayout = VK_NULL_HANDLE;
VkPipeline            s_cullPipe   = VK_NULL_HANDLE;

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
        const bool gpuOk = mat && !mat->isWmark && !mat->tessellated
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

bool CreateCullPipeline()
{
    if (!g_ShaderManager) g_ShaderManager = xr_new<VK::CVulkanSPIRVLoader>();
    VkShaderModule cs = g_ShaderManager->Load("world_cull.comp.spv");
    if (!cs) { Msg("![VK WorldGPU] world_cull.comp.spv load failed"); return false; }

    VkDescriptorSetLayoutBinding b[3]{};
    for (u32 i = 0; i < 3; ++i) {
        b[i].binding = i; b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = 3; lci.pBindings = b;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &s_setL) != VK_SUCCESS) return false;

    VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 };
    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets = 1; pci.poolSizeCount = 1; pci.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &s_pool) != VK_SUCCESS) return false;
    VkDescriptorSetAllocateInfo dai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dai.descriptorPool = s_pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &s_setL;
    if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, &s_set) != VK_SUCCESS) return false;

    VkDescriptorBufferInfo bi[3] = {
        { s_meta->GetHandle(),     0, VK_WHOLE_SIZE },
        { s_indirect->GetHandle(), 0, VK_WHOLE_SIZE },
        { s_count->GetHandle(),    0, VK_WHOLE_SIZE },
    };
    VkWriteDescriptorSet w[3]{};
    for (u32 i = 0; i < 3; ++i) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = s_set; w[i].dstBinding = i;
        w[i].descriptorCount = 1; w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(VulkanHW.m_Device, 3, w, 0, nullptr);

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

} // anonymous namespace

void Build()
{
    if (s_built) return;
    s_built = true;   // one attempt; stays "built" (possibly empty) so we don't retry every frame

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

    // Sort into draw batches: terrain first-class, then by (material, stride,
    // tcOffset, vb, ib) so a run shares pipeline + descriptor set + buffer binds.
    auto key = [](vkFVisual* a) {
        return std::make_tuple(a->m_pWorldMaterial->isTerrain ? 0 : 1,
                               (const void*)a->m_pWorldMaterial,
                               a->m_mesh.vStride, a->m_mesh.tcOffset,
                               (const void*)a->m_mesh.p_rm_Vertices->GetHandle(),
                               (const void*)a->m_mesh.p_rm_Indices->GetHandle());
    };
    std::sort(meshes.begin(), meshes.end(), [&](vkFVisual* a, vkFVisual* b) { return key(a) < key(b); });

    s_total = (u32)meshes.size();
    xr_vector<GpuMeshMeta> meta(s_total);
    for (u32 i = 0; i < s_total; ++i) {
        vkFVisual* fv = meshes[i];
        GpuMeshMeta& m = meta[i];
        m.sphere_P     = fv->vis.sphere.P;
        m.sphere_R     = fv->vis.sphere.R;
        // Draw range = what the CPU path draws. PROGRESSIVE meshes (incl. terrain)
        // must draw their finest sliding-window slice (iBase + sw_offsets[0]/
        // sw_counts[0], = vkFProgressive::Submit), NOT the whole IB span — the full
        // span contains ALL LOD slices, and the coarse ones (different/decimated
        // heights) poke through the fine mesh → a "step" on terrain in the color
        // pass. (Same over-draw fix already applied to vk_shadow_gpu.)
        u32 fullFirst = fv->m_mesh.iBase, fullCount = fv->m_mesh.iCount;
        if (fv->Type == MT_PROGRESSIVE) {
            auto* pg = static_cast<vkFProgressive*>(fv);
            if (pg->sw_count > 0 && pg->sw_offsets && pg->sw_counts) {
                fullFirst = fv->m_mesh.iBase + pg->sw_offsets[0];
                fullCount = pg->sw_counts[0];
            }
        }
        m.index_count  = fullCount;
        m.ib_first     = fullFirst;
        m.first_vertex = fv->m_mesh.vBase;
        m.group        = 0;   // assigned after grouping
    }

    s_groups.clear(); s_maxGroupMesh = 0;
    Group cur{};
    auto seed = [&](u32 i) {
        vkFVisual* fv = meshes[i];
        cur.mat      = fv->m_pWorldMaterial;
        cur.terrain  = fv->m_pWorldMaterial->isTerrain;
        cur.stride   = fv->m_mesh.vStride;
        cur.tcOffset = fv->m_mesh.tcOffset;
        cur.vb       = fv->m_mesh.p_rm_Vertices->GetHandle();
        cur.ib       = fv->m_mesh.p_rm_Indices->GetHandle();
        cur.iType    = fv->m_mesh.iType;
        cur.meshOffset = i; cur.meshCount = 1;
    };
    seed(0);
    for (u32 i = 1; i < s_total; ++i) {
        vkFVisual* fv = meshes[i];
        const bool same = fv->m_pWorldMaterial == cur.mat
            && fv->m_mesh.vStride == cur.stride && fv->m_mesh.tcOffset == cur.tcOffset
            && fv->m_mesh.p_rm_Vertices->GetHandle() == cur.vb
            && fv->m_mesh.p_rm_Indices->GetHandle() == cur.ib;
        if (same) cur.meshCount++;
        else { s_maxGroupMesh = _max(s_maxGroupMesh, cur.meshCount); s_groups.push_back(cur); seed(i); }
    }
    s_maxGroupMesh = _max(s_maxGroupMesh, cur.meshCount);
    s_groups.push_back(cur);

    const u32 nGroups = (u32)s_groups.size();
    for (u32 g = 0; g < nGroups; ++g)
        for (u32 i = s_groups[g].meshOffset; i < s_groups[g].meshOffset + s_groups[g].meshCount; ++i)
            meta[i].group = g;

    // meta SSBO (device-local)
    s_meta = xr_new<CVulkanBuffer>();
    s_meta->Create(sizeof(GpuMeshMeta) * s_total, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                   VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    s_meta->Upload(meta.data(), sizeof(GpuMeshMeta) * s_total);

    const VkBufferUsageFlags iu = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    s_indirect = xr_new<CVulkanBuffer>();
    s_indirect->Create((VkDeviceSize)nGroups * s_maxGroupMesh * sizeof(VkDrawIndexedIndirectCommand), iu,
                       VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    s_count = xr_new<CVulkanBuffer>();
    s_count->Create((VkDeviceSize)nGroups * sizeof(u32), iu, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

    if (!CreateCullPipeline()) { Msg("![VK WorldGPU] cull pipeline failed — disabled"); s_groups.clear(); return; }

    Msg("[VK WorldGPU] built: %u GPU meshes, %u groups, maxGroup=%u | cpu-set=%u (non-GPU static leaves)",
        s_total, nGroups, s_maxGroupMesh, (u32)s_cpuMeshes.size());
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

void Cull(VkCommandBuffer cmd, const Fmatrix& viewProj)
{
    if (!Built()) return;
    const u32 nGroups = (u32)s_groups.size();

    Fvector4 planes[6];
    VK::ExtractFrustumPlanes(viewProj, planes);

    // WAR-guard vs the previous frame's indirect reads.
    MemBarrier(cmd, 0, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
               VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    vkCmdFillBuffer(cmd, s_count->GetHandle(), 0, (VkDeviceSize)nGroups * sizeof(u32), 0u);
    MemBarrier(cmd, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
               VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_cullPipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_cullLayout, 0, 1, &s_set, 0, nullptr);
    CullPush pc{};
    for (int p = 0; p < 6; ++p) pc.planes[p] = planes[p];
    pc.numGroups = nGroups; pc.maxGroupMesh = s_maxGroupMesh; pc.total = s_total;
    vkCmdPushConstants(cmd, s_cullLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (s_total + 255) / 256, 1, 1);

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
        if (displaceTerrain && mat->isTerrain) {
            VkPipeline       tpipe = PipelineCache::GetTerrainDepthPipeline();
            VkPipelineLayout tlay  = PipelineCache::GetTerrainLayout();
            VkDescriptorSet  eset  = EnvLight::GetCurrentSet();
            if (tpipe == VK_NULL_HANDLE || tlay == VK_NULL_HANDLE || eset == VK_NULL_HANDLE) continue;
            if (tlay != lastLayout) { lastLayout = tlay; lastMatSet = VK_NULL_HANDLE; lastPipe = VK_NULL_HANDLE; lastVB = VK_NULL_HANDLE; lastIB = VK_NULL_HANDLE; }
            if (tpipe != lastPipe)  { vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, tpipe); lastPipe = tpipe; lastVB = VK_NULL_HANDLE; lastIB = VK_NULL_HANDLE; }
            if (eset != lastMatSet) { vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, tlay, 1, 1, &eset, 0, nullptr); lastMatSet = eset; }   // set 1 = EnvLight
            struct TPush { Fmatrix mvp; float uv[2]; float aref; float ds; } tp{}; tp.mvp = vp;
            vkCmdPushConstants(cmd, tlay, PipelineCache::GetPushStages(), 0, sizeof(TPush), &tp);
            if (grp.vb != lastVB) { VkDeviceSize z = 0; vkCmdBindVertexBuffers(cmd, 0, 1, &grp.vb, &z); lastVB = grp.vb; }
            if (grp.ib != lastIB || grp.iType != lastIType) { vkCmdBindIndexBuffer(cmd, grp.ib, 0, grp.iType); lastIB = grp.ib; lastIType = grp.iType; }
            const VkDeviceSize cmdOffT = (VkDeviceSize)g * s_maxGroupMesh * sizeof(VkDrawIndexedIndirectCommand);
            const VkDeviceSize cntOffT = (VkDeviceSize)g * sizeof(u32);
            vkCmdDrawIndexedIndirectCount(cmd, s_indirect->GetHandle(), cmdOffT, s_count->GetHandle(), cntOffT,
                                          s_maxGroupMesh, sizeof(VkDrawIndexedIndirectCommand));
            continue;
        }

        const bool at = mat->alphaRef >= 0.f;
        if (at && (layoutAT == VK_NULL_HANDLE || mat->set == VK_NULL_HANDLE)) continue;  // mirror FlushDepth skip

        VkPipeline pipe = at ? PipelineCache::GetDepthATPipeline(grp.stride, grp.tcOffset)
                             : PipelineCache::GetDepthPipeline(grp.stride);
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
        const VkDeviceSize cmdOff = (VkDeviceSize)g * s_maxGroupMesh * sizeof(VkDrawIndexedIndirectCommand);
        const VkDeviceSize cntOff = (VkDeviceSize)g * sizeof(u32);
        vkCmdDrawIndexedIndirectCount(cmd, s_indirect->GetHandle(), cmdOff, s_count->GetHandle(), cntOff,
                                      s_maxGroupMesh, sizeof(VkDrawIndexedIndirectCommand));
    }
}

void DrawColor(VkCommandBuffer cmd, const Fmatrix& viewProj, VkDescriptorSet envSet)
{
    if (!Built()) return;
    const VkShaderStageFlags kStages = PipelineCache::GetPushStages();
    const u32 nGroups = (u32)s_groups.size();

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
            k.vs = lmap ? PipelineCache::WorldLmapVS() : PipelineCache::WorldVlitVS();
            k.fs = lmap ? PipelineCache::WorldLmapFS() : PipelineCache::WorldVlitFS();
            k.depthTest = true; k.wmark = false; k.tess = false;
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

        if (grp.vb != lastVB) { VkDeviceSize z = 0; vkCmdBindVertexBuffers(cmd, 0, 1, &grp.vb, &z); lastVB = grp.vb; }
        if (grp.ib != lastIB || grp.iType != lastIType) { vkCmdBindIndexBuffer(cmd, grp.ib, 0, grp.iType); lastIB = grp.ib; lastIType = grp.iType; }
        const VkDeviceSize cmdOff = (VkDeviceSize)g * s_maxGroupMesh * sizeof(VkDrawIndexedIndirectCommand);
        const VkDeviceSize cntOff = (VkDeviceSize)g * sizeof(u32);
        vkCmdDrawIndexedIndirectCount(cmd, s_indirect->GetHandle(), cmdOff, s_count->GetHandle(), cntOff,
                                      s_maxGroupMesh, sizeof(VkDrawIndexedIndirectCommand));
    }
}

void Destroy()
{
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
    if (s_cullPipe)   { vkDestroyPipeline(VulkanHW.m_Device, s_cullPipe, nullptr); s_cullPipe = VK_NULL_HANDLE; }
    if (s_cullLayout) { vkDestroyPipelineLayout(VulkanHW.m_Device, s_cullLayout, nullptr); s_cullLayout = VK_NULL_HANDLE; }
    if (s_pool)       { vkDestroyDescriptorPool(VulkanHW.m_Device, s_pool, nullptr); s_pool = VK_NULL_HANDLE; }
    if (s_setL)       { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_setL, nullptr); s_setL = VK_NULL_HANDLE; }
    auto del = [](CVulkanBuffer*& b) { if (b) { xr_delete(b); b = nullptr; } };
    del(s_meta); del(s_indirect); del(s_count);
    s_groups.clear(); s_meshSet.clear(); s_cpuMeshes.clear(); s_total = 0; s_maxGroupMesh = 0; s_set = VK_NULL_HANDLE; s_built = false;
}

}} // namespace VK::WorldGPU
