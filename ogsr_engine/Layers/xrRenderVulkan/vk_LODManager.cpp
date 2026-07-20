// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan — LOD imposter manager implementation.
// See vk_LODManager.h. v1: nearest-facet billboard, distance-gated.

#include "stdafx.h"
#include "vk_LODManager.h"
#include "vk_Visual.h"            // vkFLOD
#include "vk_world_material.h"    // fallback white
#include "vk_texture.h"           // CVulkanTexture (level_lods atlas)
#include "vk_buffer.h"
#include "vk_pass_context.h"
#include "vk_swapchain.h"
#include "vk_scene_color.h"       // HDR scene target format
#include "vk_shaders.h"           // g_ShaderManager
#include "vk_pipeline_cache.h"    // PipelineCache::GetCacheObject
#include "vk_command_buffer.h"    // CommandManager
#include "vk_compute_util.h"      // MakePipelineLayout / CreateComputePipeline
#include "vk_cull.h"              // ExtractFrustumPlanes (GPU cull push)
#include "vk_DetailManager.h"     // shared HZB pyramid (grass/world)
#include "vk_profiler.h"          // GPU zones for the cull dispatch
#include "HW_Vulkan.h"
#include "CRender_Vulkan.h"
#include "../../xr_3da/IGame_Persistent.h" // g_pGamePersistent->Environment()
#include "../../xr_3da/Environment.h"      // CEnvDescriptorMixer (sun_color/hemi_color)

// Imposters only show beyond this range so up-close trees keep their full mesh
// (and don't overlay flat billboards on detailed geometry near the player). Far
// enough that the flatness reads as distant haze, not a low-poly pop-in.
static constexpr float kImposterMinDist = 250.0f;

// GPU path (compute cull + vertex pulling) — instant A/B against the CPU walk.
extern int ps_r_lods_gpu;

namespace VK
{

// Draw push block — shared by the CPU and GPU imposter pipelines (128 B, the
// guaranteed push-constant minimum; layout mirrors lod_imposter*.vert/frag PC).
struct LodPush
{
    Fmatrix  mvp;
    Fvector4 tint;
    Fvector4 fog_color;
    Fvector4 fog_params;
    Fvector4 eye_pos;
};

// Env tint for the pre-lit imposter atlas: clamp(hemi + sun*0.5, 0, 1). ≈1 at
// midday (atlas unchanged), warm & dim at sunset, dark at night — tracks the
// near trees without knowing the atlas bake's weather. Neutral fallback if env down.
static void FillLodPush(LodPush& lp, const VK::FrameContext& ctx)
{
    lp.mvp = *ctx.viewProj;
    lp.tint.set(0.9f, 0.9f, 0.9f, 0.0f);
    lp.fog_params.set(0.f, 1e6f, 1e6f, 0.f);   // no fog until env is up
    if (g_pGamePersistent) {
        if (auto* E = g_pGamePersistent->Environment().CurrentEnv) {
            lp.tint.x = clampr(E->hemi_color.x + E->sun_color.x * 0.5f, 0.0f, 1.0f);
            lp.tint.y = clampr(E->hemi_color.y + E->sun_color.y * 0.5f, 0.0f, 1.0f);
            lp.tint.z = clampr(E->hemi_color.z + E->sun_color.z * 0.5f, 0.0f, 1.0f);
            // Distance fog (R4) — same params the world shaders use (vk_env_light).
            const float fn = E->fog_near, ff = E->fog_far;
            const float r  = (ff > fn + 1e-3f) ? 1.f / (ff - fn) : 0.f;
            lp.fog_params.set(-fn * r, fn, ff, r);
            lp.fog_color.set(E->fog_color.x, E->fog_color.y, E->fog_color.z, 0.f);
        }
    }
    lp.eye_pos.set(Device.vCameraPosition.x, Device.vCameraPosition.y, Device.vCameraPosition.z, 0.f);
}

// Cull-dispatch push (208 B) — must match lod_cull.comp PC.
struct LodCullPush
{
    Fmatrix  viewProj;
    Fvector4 planes[6];
    Fvector4 cameraPos;   // xyz eye, w = kImposterMinDist
    Fvector4 hzbParams;   // x = focal scale 1/tan(fovY/2)
    u32 total; u32 hzbOn; u32 _p0; u32 _p1;
};
static_assert(sizeof(LodCullPush) == 208, "LodCullPush must match lod_cull.comp");

CLODManager::CLODManager() {}
CLODManager::~CLODManager() { Destroy(); }

bool CLODManager::IsReady() const
{
    return m_bBuilt && m_Pipeline != VK_NULL_HANDLE && m_DescSet != VK_NULL_HANDLE && !m_Lods.empty();
}

// ============================================================================
void CLODManager::Build()
{
    if (m_bBuilt) return;
    m_bBuilt = true;

    // ----- Collect FLODs with valid billboard facets.
    m_Lods.reserve(2048);
    for (IRenderVisual* iv : RImplementation.Visuals)
    {
        auto* lod = dynamic_cast<vkFLOD*>(static_cast<vkRender_Visual*>(iv));
        if (lod && lod->facetsValid && lod->vis.sphere.R > 0.01f)
            m_Lods.push_back(lod);
    }

    if (m_Lods.empty())
    {
        Msg("[VK LOD] No FLOD imposters on this level");
        return;
    }

    // ----- Triple-buffered dynamic VB: 6 verts (2 tris) per FLOD quad.
    m_MaxVerts = (u32)m_Lods.size() * 6;
    const VkDeviceSize vbSize = (VkDeviceSize)m_MaxVerts * sizeof(LodImposterVertex);
    for (u32 f = 0; f < LOD_FRAMES; ++f)
    {
        m_DynVB[f] = xr_new<CVulkanBuffer>();
        m_DynVB[f]->Create(vbSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
        m_DynVB[f]->Map();   // persistent map — VERTEX buffers aren't auto-mapped (only UBO/SSBO are)
    }

    CreateAtlasDescriptor();
    CreatePipeline();
    BuildGpuPath();

    Msg("[VK LOD] Built: %u FLOD imposters (VB %u KB ×%u)",
        (u32)m_Lods.size(), (u32)(vbSize / 1024), LOD_FRAMES);
}

// ============================================================================
// Atlas descriptor — one COMBINED_IMAGE_SAMPLER pointing at level_lods.
// ============================================================================
void CLODManager::CreateAtlasDescriptor()
{
    // level_lods.dds lives in the level's own dir ($level$), packed in the level
    // .db — WorldMaterialCache only searches $game_textures$, so load it directly.
    VkImageView view = VK_NULL_HANDLE;
    {
        string_path full;
        FS.update_path(full, "$level$", "level_lods.dds");
        if (!FS.exist(full))
            FS.update_path(full, "$game_textures$", "level_lods.dds");

        m_AtlasTex = xr_new<CVulkanTexture>();
        // Imposter atlas holds baked albedo of distant statics — Colour, so the far
        // LODs shade in the same space as the near geometry they hand over to.
        if (FS.exist(full) && m_AtlasTex->LoadDDS(full, /*applyBCSwizzle*/ false,
                                                  TexStreamClass::UI, TexColorSpace::Color))
        {
            view = m_AtlasTex->GetView();
        }
        else
        {
            xr_delete(m_AtlasTex);
            view = WorldMaterialCache::GetDefault()->view;   // white fallback
        }
    }

    VkSamplerCreateInfo sci{};
    sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter    = VK_FILTER_LINEAR;
    sci.minFilter    = VK_FILTER_LINEAR;
    sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;   // atlas — clamp
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxLod       = VK_LOD_CLAMP_NONE;
    vkCreateSampler(VulkanHW.m_Device, &sci, nullptr, &m_Sampler);

    VkDescriptorSetLayoutBinding b{};
    b.binding = 0; b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b.descriptorCount = 1; b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lci.bindingCount = 1; lci.pBindings = &b;
    vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &m_DescLayout);

    VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 };
    VkDescriptorPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = 1; pci.poolSizeCount = 1; pci.pPoolSizes = &ps;
    vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &m_DescPool);

    VkDescriptorSetAllocateInfo dai{};
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = m_DescPool; dai.descriptorSetCount = 1; dai.pSetLayouts = &m_DescLayout;
    vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, &m_DescSet);

    VkDescriptorImageInfo ii{};
    ii.sampler = m_Sampler; ii.imageView = view;
    ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet w{};
    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet = m_DescSet; w.dstBinding = 0; w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w.pImageInfo = &ii;
    vkUpdateDescriptorSets(VulkanHW.m_Device, 1, &w, 0, nullptr);

    Msg("[VK LOD] Atlas 'level_lods' %s", m_AtlasTex ? "loaded from $level$" : "MISSING (white fallback)");
}

// ============================================================================
// Shared graphics-pipeline builder for both imposter draw paths. They differ
// only in the vertex stage: the CPU path streams LodImposterVertex through the
// fixed-function input, the GPU path (vertexInput=false) pulls everything from
// SSBOs by gl_VertexIndex. Fragment stage / state is identical.
static VkPipeline CreateImposterPipeline(const char* vsName, bool vertexInput, VkPipelineLayout layout)
{
    VkShaderModule vs = g_ShaderManager->Load(vsName);
    VkShaderModule fs = g_ShaderManager->Load("lod_imposter.frag.spv");
    if (vs == VK_NULL_HANDLE || fs == VK_NULL_HANDLE) {
        Msg("![VK LOD] %s / lod_imposter.frag.spv load failed", vsName); return VK_NULL_HANDLE;
    }

    VkPipelineShaderStageCreateInfo ss[2]{};
    ss[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ss[0].stage = VK_SHADER_STAGE_VERTEX_BIT;   ss[0].module = vs; ss[0].pName = "main";
    ss[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ss[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; ss[1].module = fs; ss[1].pName = "main";

    VkVertexInputBindingDescription vibd{ 0, sizeof(LodImposterVertex), VK_VERTEX_INPUT_RATE_VERTEX };
    VkVertexInputAttributeDescription via[3]{};
    via[0] = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0  };   // pos
    via[1] = { 1, 0, VK_FORMAT_R32G32_SFLOAT,    12 };   // uv
    via[2] = { 2, 0, VK_FORMAT_R8G8B8A8_UNORM,   20 };   // color
    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    if (vertexInput) {
        vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &vibd;
        vi.vertexAttributeDescriptionCount = 3; vi.pVertexAttributeDescriptions = via;
    }   // else: empty input state — the GPU path pulls verts from SSBOs

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1; vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;      // billboards double-sided
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendAttachmentState ba{};
    ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    ba.blendEnable = VK_FALSE;
    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1; cb.pAttachments = &ba;

    VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynState{};
    dynState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynState.dynamicStateCount = 2; dynState.pDynamicStates = dyn;

    VkFormat colorFmt = VK::SceneColor::Format();
    VkPipelineRenderingCreateInfo prci{};
    prci.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    prci.colorAttachmentCount = 1; prci.pColorAttachmentFormats = &colorFmt;
    prci.depthAttachmentFormat = Swapchain.m_DepthFormat;

    VkGraphicsPipelineCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pi.pNext = &prci; pi.stageCount = 2; pi.pStages = ss;
    pi.pVertexInputState = &vi; pi.pInputAssemblyState = &ia;
    pi.pViewportState = &vp; pi.pRasterizationState = &rs;
    pi.pMultisampleState = &ms; pi.pDepthStencilState = &ds;
    pi.pColorBlendState = &cb; pi.pDynamicState = &dynState;
    pi.layout = layout;
    VkPipeline pipe = VK_NULL_HANDLE;
    if (vkCreateGraphicsPipelines(VulkanHW.m_Device, VK::PipelineCache::GetCacheObject(),
                                  1, &pi, nullptr, &pipe) != VK_SUCCESS) {
        Msg("![VK LOD] pipeline create failed (%s)", vsName); return VK_NULL_HANDLE;
    }
    return pipe;
}

// ============================================================================
void CLODManager::CreatePipeline()
{
    if (!g_ShaderManager) { Msg("![VK LOD] g_ShaderManager null"); return; }

    // Push: mat4 viewProj + vec4 tint + vec4 fog_color + vec4 fog_params +
    // vec4 eye_pos = 128 bytes (the guaranteed push-constant minimum).
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcr.offset = 0; pcr.size = sizeof(LodPush);
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1; plci.pSetLayouts = &m_DescLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &m_PipelineLayout);

    m_Pipeline = CreateImposterPipeline("lod_imposter.vert.spv", true, m_PipelineLayout);
    if (m_Pipeline != VK_NULL_HANDLE)
        Msg("[VK LOD] Pipeline OK");
}

// ============================================================================
// GPU path (r_lods_gpu): static facet SSBO + cull compute + pulling pipeline.
// All buffers are DEVICE_LOCAL (gpuOnly — see the vk_buffer BAR lesson); the
// only CPU work left per frame is a 16 B vkCmdUpdateBuffer and one dispatch.
// ============================================================================
void CLODManager::BuildGpuPath()
{
    if (!g_ShaderManager || m_DescLayout == VK_NULL_HANDLE) return;
    VkDevice dev = VulkanHW.m_Device;
    const u32 N = (u32)m_Lods.size();

    // ----- Pack the static facet data (must match LodEntry in lod_cull.comp).
    struct GpuLodFacet { Fvector4 n; Fvector4 c[4]; Fvector4 vs; };
    struct GpuLodEntry { Fvector4 sphere; GpuLodFacet f[8]; };
    static_assert(sizeof(GpuLodEntry) == 784, "GpuLodEntry must match lod_cull.comp (std430)");

    xr_vector<GpuLodEntry> data(N);
    for (u32 i = 0; i < N; ++i)
    {
        const vkFLOD* lod = m_Lods[i];
        GpuLodEntry& e = data[i];
        e.sphere.set(lod->vis.sphere.P.x, lod->vis.sphere.P.y, lod->vis.sphere.P.z, lod->vis.sphere.R);
        for (u32 s = 0; s < 8; ++s)
        {
            const vkFLOD::LodFacet& F = lod->facets[s];
            e.f[s].n.set(F.N.x, F.N.y, F.N.z, 0.f);
            for (u32 k = 0; k < 4; ++k)
                e.f[s].c[k].set(F.v[k].v.x, F.v[k].v.y, F.v[k].v.z, F.v[k].t.x);
            e.f[s].vs.set(F.v[0].t.y, F.v[1].t.y, F.v[2].t.y, F.v[3].t.y);
        }
    }

    m_GpuLods = xr_new<CVulkanBuffer>();
    m_GpuLods->Create((VkDeviceSize)N * sizeof(GpuLodEntry),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    m_GpuLods->Upload(data.data(), (VkDeviceSize)N * sizeof(GpuLodEntry));

    m_GpuInsts = xr_new<CVulkanBuffer>();
    m_GpuInsts->Create((VkDeviceSize)N * sizeof(u32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                       VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);

    m_GpuIndirect = xr_new<CVulkanBuffer>();
    m_GpuIndirect->Create(sizeof(VkDrawIndirectCommand),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);

    // ----- Descriptor layouts: cull (3 SSBO + HZB sampler), pulling VS (2 SSBO).
    {
        VkDescriptorSetLayoutBinding cb[4]{};
        for (u32 i = 0; i < 3; ++i) {
            cb[i].binding = i; cb[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            cb[i].descriptorCount = 1; cb[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        cb[3].binding = 3; cb[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        cb[3].descriptorCount = 1; cb[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 4; lci.pBindings = cb;
        vkCreateDescriptorSetLayout(dev, &lci, nullptr, &m_CullDescLayout);

        VkDescriptorSetLayoutBinding vb[2]{};
        for (u32 i = 0; i < 2; ++i) {
            vb[i].binding = i; vb[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            vb[i].descriptorCount = 1; vb[i].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        }
        lci.bindingCount = 2; lci.pBindings = vb;
        vkCreateDescriptorSetLayout(dev, &lci, nullptr, &m_VtxDescLayout);
    }

    // ----- Pool + sets. Cull sets are per-frame-in-flight so the HZB binding can
    // be rewritten each frame without touching a set the GPU still reads.
    {
        VkDescriptorPoolSize ps[2] = {
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         3 * LOD_FRAMES + 2 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, LOD_FRAMES },
        };
        VkDescriptorPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets = LOD_FRAMES + 1; pci.poolSizeCount = 2; pci.pPoolSizes = ps;
        vkCreateDescriptorPool(dev, &pci, nullptr, &m_GpuDescPool);

        VkDescriptorSetLayout layouts[LOD_FRAMES];
        for (u32 f = 0; f < LOD_FRAMES; ++f) layouts[f] = m_CullDescLayout;
        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = m_GpuDescPool; dai.descriptorSetCount = LOD_FRAMES; dai.pSetLayouts = layouts;
        vkAllocateDescriptorSets(dev, &dai, m_CullSets);
        dai.descriptorSetCount = 1; dai.pSetLayouts = &m_VtxDescLayout;
        vkAllocateDescriptorSets(dev, &dai, &m_VtxSet);

        // Static buffer bindings (b0..b2 cull, b0..b1 vtx). HZB (b3) is per frame.
        VkDescriptorBufferInfo bLods{ m_GpuLods->GetHandle(),     0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo bInst{ m_GpuInsts->GetHandle(),    0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo bInd { m_GpuIndirect->GetHandle(), 0, VK_WHOLE_SIZE };
        xr_vector<VkWriteDescriptorSet> ws;
        auto add = [&ws](VkDescriptorSet set, u32 binding, const VkDescriptorBufferInfo* bi) {
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = set; w.dstBinding = binding; w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w.pBufferInfo = bi;
            ws.push_back(w);
        };
        for (u32 f = 0; f < LOD_FRAMES; ++f) {
            add(m_CullSets[f], 0, &bLods);
            add(m_CullSets[f], 1, &bInst);
            add(m_CullSets[f], 2, &bInd);
        }
        add(m_VtxSet, 0, &bLods);
        add(m_VtxSet, 1, &bInst);
        vkUpdateDescriptorSets(dev, (u32)ws.size(), ws.data(), 0, nullptr);
    }

    // ----- Pipelines: cull compute + pulling graphics (set 0 = atlas, set 1 = geo).
    VkShaderModule cs = g_ShaderManager->Load("lod_cull.comp.spv");
    if (cs != VK_NULL_HANDLE) {
        m_CullPipelineLayout = MakePipelineLayout({ m_CullDescLayout }, sizeof(LodCullPush));
        m_CullPipeline = CreateComputePipeline(cs, m_CullPipelineLayout, "lod_cull");
    }

    {
        VkPushConstantRange pcr{};
        pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pcr.offset = 0; pcr.size = sizeof(LodPush);
        VkDescriptorSetLayout sets[2] = { m_DescLayout, m_VtxDescLayout };
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 2; plci.pSetLayouts = sets;
        plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
        vkCreatePipelineLayout(dev, &plci, nullptr, &m_PipelineLayoutGPU);
        m_PipelineGPU = CreateImposterPipeline("lod_imposter_gpu.vert.spv", false, m_PipelineLayoutGPU);
    }

    const bool ok = m_CullPipeline != VK_NULL_HANDLE && m_PipelineGPU != VK_NULL_HANDLE;
    Msg("[VK LOD] GPU path %s: %u entries, SSBO %u KB", ok ? "ready" : "FAILED (CPU fallback)",
        N, (u32)((N * sizeof(GpuLodEntry)) / 1024));
}

// ============================================================================
void CLODManager::Render(VK::FrameContext& ctx)
{
    if (!IsReady() || ctx.cmd == VK_NULL_HANDLE || ctx.viewProj == nullptr) return;

    if (ps_r_lods_gpu && m_CullPipeline != VK_NULL_HANDLE && m_PipelineGPU != VK_NULL_HANDLE)
    {
        RenderGpu(ctx);
        return;
    }

    const u32 frame = CommandManager.GetCurrentFrame() % LOD_FRAMES;
    CVulkanBuffer* vb = m_DynVB[frame];
    if (!vb || !vb->IsMapped()) return;

    const Fvector eye = Device.vCameraPosition;
    auto* out = (LodImposterVertex*)vb->m_Mapped;
    u32 nVerts = 0;

    // Build best-facet quads for distant FLODs.
    for (vkFLOD* lod : m_Lods)
    {
        const Fvector& C = lod->vis.sphere.P;
        const float R = lod->vis.sphere.R;
        Fvector d; d.sub(C, eye);
        const float dist = d.magnitude();
        if (dist < kImposterMinDist) continue;     // close → full mesh handles it

        Fvector Ldir = d; Ldir.div(dist > 0.001f ? dist : 1.0f);   // camera→object
        Fvector shift; shift.mul(Ldir, -0.5f * R);                 // pull toward camera

        // Pick the facet whose normal best faces the camera.
        u32 best = 0; float bestDot = -1e9f;
        for (u32 s = 0; s < 8; ++s) {
            const float dt = Ldir.dotproduct(lod->facets[s].N);
            if (dt > bestDot) { bestDot = dt; best = s; }
        }
        const vkFLOD::LodFacet& F = lod->facets[best];

        // Emit quad v0,v1,v2 + v0,v2,v3 (cull NONE → winding irrelevant).
        const int tri[6] = { 0, 1, 2, 0, 2, 3 };
        for (int k = 0; k < 6; ++k) {
            const vkFLOD::LodVertex& sv = F.v[tri[k]];
            LodImposterVertex& dv = out[nVerts++];
            dv.pos.add(sv.v, shift);
            dv.uv = sv.t;
            dv.color = 0xFFFFFFFFu;   // v1: atlas already carries baked colour
        }
    }

    if (nVerts == 0) return;
    vb->Flush();

    // ----- Draw into the live color+depth pass (single-layout convention). ----
    VK::BeginOverlayRendering(ctx.cmd, ctx);   // shared overlay begin — see vk_pass_context.h

    vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_Pipeline);
    vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_PipelineLayout,
                            0, 1, &m_DescSet, 0, nullptr);
    LodPush lp{};
    FillLodPush(lp, ctx);
    vkCmdPushConstants(ctx.cmd, m_PipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(lp), &lp);
    VkBuffer h = vb->GetHandle(); VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(ctx.cmd, 0, 1, &h, &off);
    vkCmdDraw(ctx.cmd, nVerts, 1, 0, 0);

    vkCmdEndRendering(ctx.cmd);

    static bool s_diag = false;
    if (!s_diag) { Msg("[VK LOD] First Render: %u imposter verts (%u quads)", nVerts, nVerts / 6); s_diag = true; }
}

// ============================================================================
// GPU path — the whole per-frame CPU cost is: rewrite one HZB descriptor, a
// 16 B indirect reset, one dispatch and one indirect draw. The FLOD walk,
// facet picks and vertex building all moved to lod_cull.comp / the pulling VS.
// ============================================================================
void CLODManager::RenderGpu(VK::FrameContext& ctx)
{
    VkCommandBuffer cmd = ctx.cmd;
    const u32 N = (u32)m_Lods.size();

    // Hi-Z pyramid: Pass_World (r_hzb_cull) or grass already built it this frame;
    // the guarded call covers configs where neither ran (build-once-per-frame
    // stamp makes it a no-op otherwise). Runs outside a render pass — we are too.
    bool hzbOn = false;
    if (RImplementation.Details && RImplementation.Details->HZBReady())
    {
        RImplementation.Details->BuildHZBForFrame(ctx);
        hzbOn = true;
    }

    // Rewrite the HZB binding of THIS frame slot's cull set (its fence passed, the
    // GPU no longer reads it). Without a pyramid bind the atlas as an inert dummy
    // (never sampled — the shader gates on hzbOn) so validation sees a live view.
    const u32 frame = CommandManager.GetCurrentFrame() % LOD_FRAMES;
    {
        VkDescriptorImageInfo ii{};
        if (hzbOn) {
            ii.sampler     = RImplementation.Details->HZBSampler();
            ii.imageView   = RImplementation.Details->HZBView();
            ii.imageLayout = VK_IMAGE_LAYOUT_GENERAL;   // HZB lives in GENERAL
        } else {
            ii.sampler     = m_Sampler;
            ii.imageView   = m_AtlasTex ? m_AtlasTex->GetView() : WorldMaterialCache::GetDefault()->view;
            ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = m_CullSets[frame]; w.dstBinding = 3; w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w.pImageInfo = &ii;
        vkUpdateDescriptorSets(VulkanHW.m_Device, 1, &w, 0, nullptr);
    }

    const int zc = Prof::ZoneBegin(cmd, "LODs/cull");

    // Reset the indirect draw {vertexCount=0, instanceCount=1}. The barrier also
    // orders the PREVIOUS frame's indirect/VS reads of these single-buffered
    // streams before this frame's transfer + compute writes (WAR, execution-only).
    const VkDrawIndirectCommand rst{ 0, 1, 0, 0 };
    vkCmdUpdateBuffer(cmd, m_GpuIndirect->GetHandle(), 0, sizeof(rst), &rst);
    {
        VkMemoryBarrier mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_CullPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_CullPipelineLayout,
                            0, 1, &m_CullSets[frame], 0, nullptr);

    LodCullPush cp{};
    cp.viewProj = *ctx.viewProj;
    VK::ExtractFrustumPlanes(*ctx.viewProj, cp.planes);
    const Fvector eye = Device.vCameraPosition;
    cp.cameraPos.set(eye.x, eye.y, eye.z, kImposterMinDist);
    // Focal scale for the HZB footprint (same formula as WorldGPU::LodParams .w).
    cp.hzbParams.set(1.f / _max(0.05f, tanf(deg2rad(Device.fFOV) * 0.5f)), 0.f, 0.f, 0.f);
    cp.total = N;
    cp.hzbOn = hzbOn ? 1u : 0u;
    vkCmdPushConstants(cmd, m_CullPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cp), &cp);
    vkCmdDispatch(cmd, (N + 255) / 256, 1, 1);

    // Cull results → indirect arg + VS-pulled instance stream.
    {
        VkMemoryBarrier mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
            0, 1, &mb, 0, nullptr, 0, nullptr);
    }
    Prof::ZoneEnd(cmd, zc);

    // ----- Indirect draw into the live color+depth pass (same as the CPU path).
    VK::BeginOverlayRendering(cmd, ctx);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_PipelineGPU);
    VkDescriptorSet sets[2] = { m_DescSet, m_VtxSet };
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_PipelineLayoutGPU,
                            0, 2, sets, 0, nullptr);
    LodPush lp{};
    FillLodPush(lp, ctx);
    vkCmdPushConstants(cmd, m_PipelineLayoutGPU, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(lp), &lp);
    vkCmdDrawIndirect(cmd, m_GpuIndirect->GetHandle(), 0, 1, sizeof(VkDrawIndirectCommand));

    vkCmdEndRendering(cmd);

    static bool s_diagGpu = false;
    if (!s_diagGpu) { Msg("[VK LOD] GPU path first frame: %u entries, hzb=%u", N, cp.hzbOn); s_diagGpu = true; }
}

// ============================================================================
void CLODManager::Destroy()
{
    if (VulkanHW.m_Device == VK_NULL_HANDLE) { m_bBuilt = false; m_Lods.clear(); return; }
    VkDevice dev = VulkanHW.m_Device;

    if (m_Pipeline)       { vkDestroyPipeline(dev, m_Pipeline, nullptr); m_Pipeline = VK_NULL_HANDLE; }
    if (m_PipelineLayout) { vkDestroyPipelineLayout(dev, m_PipelineLayout, nullptr); m_PipelineLayout = VK_NULL_HANDLE; }

    // GPU path (r_lods_gpu)
    if (m_CullPipeline)       { vkDestroyPipeline(dev, m_CullPipeline, nullptr); m_CullPipeline = VK_NULL_HANDLE; }
    if (m_CullPipelineLayout) { vkDestroyPipelineLayout(dev, m_CullPipelineLayout, nullptr); m_CullPipelineLayout = VK_NULL_HANDLE; }
    if (m_PipelineGPU)        { vkDestroyPipeline(dev, m_PipelineGPU, nullptr); m_PipelineGPU = VK_NULL_HANDLE; }
    if (m_PipelineLayoutGPU)  { vkDestroyPipelineLayout(dev, m_PipelineLayoutGPU, nullptr); m_PipelineLayoutGPU = VK_NULL_HANDLE; }
    if (m_GpuDescPool)        { vkDestroyDescriptorPool(dev, m_GpuDescPool, nullptr); m_GpuDescPool = VK_NULL_HANDLE;
                                for (u32 f = 0; f < LOD_FRAMES; ++f) m_CullSets[f] = VK_NULL_HANDLE;
                                m_VtxSet = VK_NULL_HANDLE; }
    if (m_CullDescLayout)     { vkDestroyDescriptorSetLayout(dev, m_CullDescLayout, nullptr); m_CullDescLayout = VK_NULL_HANDLE; }
    if (m_VtxDescLayout)      { vkDestroyDescriptorSetLayout(dev, m_VtxDescLayout, nullptr); m_VtxDescLayout = VK_NULL_HANDLE; }
    if (m_GpuLods)            { m_GpuLods->Destroy(); xr_delete(m_GpuLods); }
    if (m_GpuInsts)           { m_GpuInsts->Destroy(); xr_delete(m_GpuInsts); }
    if (m_GpuIndirect)        { m_GpuIndirect->Destroy(); xr_delete(m_GpuIndirect); }
    if (m_DescPool)       { vkDestroyDescriptorPool(dev, m_DescPool, nullptr); m_DescPool = VK_NULL_HANDLE; m_DescSet = VK_NULL_HANDLE; }
    if (m_DescLayout)     { vkDestroyDescriptorSetLayout(dev, m_DescLayout, nullptr); m_DescLayout = VK_NULL_HANDLE; }
    if (m_Sampler)        { vkDestroySampler(dev, m_Sampler, nullptr); m_Sampler = VK_NULL_HANDLE; }
    if (m_AtlasTex)       { m_AtlasTex->Destroy(); xr_delete(m_AtlasTex); }
    for (u32 f = 0; f < LOD_FRAMES; ++f)
        if (m_DynVB[f]) { m_DynVB[f]->Destroy(); xr_delete(m_DynVB[f]); }

    m_Lods.clear();
    m_MaxVerts = 0;
    m_bBuilt = false;
}

}  // namespace VK
