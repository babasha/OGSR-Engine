// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan — CTreeManager Session B: GPU-driven tree render.
//
// Per-frame flow (mirrors the monolith CShadowManager g-buffer tree path,
// adapted to the OGSR forward single-pass renderer):
//   1. Fill the view-frustum UBO from ctx.viewProj.
//   2. Clear per-group draw counts; barrier transfer→compute.
//   3. Per group: dispatch tree_cull.comp — frustum-tests each tree sphere and
//      appends a VkDrawIndexedIndirectCommand (firstInstance = global tree idx)
//      into the group's section, bumping the group's draw count.
//   4. Barrier compute→draw-indirect.
//   5. Begin dynamic-rendering pass (color + depth LOAD, single-layout
//      convention — no swapchain transitions here).
//   6. Per group: bind pipeline variant (tcOffset 24/28) + diffuse + VB/IB,
//      vkCmdDrawIndexedIndirectCount over the group's section.
//
// Build() (Session A) already uploaded m_TreeMetadataBuffer / m_TreeTransforms
// and allocated m_TreeIndirectBuffer / m_TreeDrawCountBuffer + per-texture
// descriptor sets. This file only adds the pipelines + the per-frame path.

#include "stdafx.h"
#include "vk_TreeManager.h"
#include "vk_pass_context.h"
#include "vk_swapchain.h"
#include "vk_scene_color.h"       // HDR scene target format
#include "vk_texture.h"
#include "vk_shaders.h"          // g_ShaderManager
#include "vk_pipeline_cache.h"   // PipelineCache::GetCacheObject
#include "vk_env_light.h"        // VK::EnvLight — set 2 (sun_vp + sun shadow map)
#include "vk_shadow.h"           // ShadowMap::SphereVisible — caster culling (RenderDepth)
#include "vk_cull.h"             // VK::ExtractFrustumPlanes (shared with DetailManager)
#include "HW_Vulkan.h"
#include "../../xr_3da/IGame_Persistent.h" // g_pGamePersistent->Environment()
#include "../../xr_3da/Environment.h"      // CEnvDescriptorMixer (sun_color/hemi_color)

// GLOBAL scope (an extern inside namespace VK would mangle as VK::* → LNK2001).
extern float ps_r_sun_boost;   // r_sun_boost — global sun multiplier (see vk_env_light)
extern float ps_r_wind_tree_bend;   // r_wind_tree_bend  — trunk sway intensity
extern float ps_r_wind_tree_anim;   // r_wind_tree_anim  — branch flutter speed
extern float ps_r_wind_tree_trunk;  // r_wind_tree_trunk — trunk anim speed
extern float ps_r_wind_tree_flutter;// r_wind_tree_flutter — crown flutter amplitude
extern float ps_r_wind_tree_crown;  // r_wind_tree_crown — flutter fade-in height

namespace VK
{

bool CTreeManager::IsReady() const
{
    return m_bBuilt
        && m_CullPipeline != VK_NULL_HANDLE
        && (m_GfxPipeline24 != VK_NULL_HANDLE || m_GfxPipeline28 != VK_NULL_HANDLE)
        && m_TreeIndirectBuffer != nullptr
        && m_TreeDrawCountBuffer != nullptr
        && m_XformDescSet != VK_NULL_HANDLE
        && !m_Groups.empty();
}

// ============================================================================
// Frustum UBO — host-visible, rewritten each frame in Render().
// ============================================================================
void CTreeManager::CreateFrustumUBO()
{
    m_FrustumUBO = xr_new<CVulkanBuffer>();
    m_FrustumUBO->Create(sizeof(TreeFrustumUBO),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
}

// ============================================================================
// Cull compute pipeline — descriptor {meta SSBO, frustum UBO, indirect SSBO,
// count SSBO}, 20 B push. Loads tree_cull.comp.spv.
// ============================================================================
void CTreeManager::CreateCullPipeline()
{
    if (!g_ShaderManager) { Msg("![VK Trees] g_ShaderManager null — cull pipeline disabled"); return; }
    if (!m_TreeMetadataBuffer || !m_FrustumUBO || !m_TreeIndirectBuffer || !m_TreeDrawCountBuffer) return;

    // Descriptor layout: 0=meta(SSBO) 1=frustum(UBO) 2=indirect(SSBO) 3=count(SSBO).
    VkDescriptorSetLayoutBinding b[4]{};
    b[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    b[1] = { 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    b[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    b[3] = { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };

    VkDescriptorSetLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lci.bindingCount = 4; lci.pBindings = b;
    vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &m_CullDescLayout);

    VkDescriptorPoolSize ps[2]{};
    ps[0] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 };
    ps[1] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 };
    VkDescriptorPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = 1; pci.poolSizeCount = 2; pci.pPoolSizes = ps;
    vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &m_CullDescPool);

    VkDescriptorSetAllocateInfo dai{};
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = m_CullDescPool; dai.descriptorSetCount = 1;
    dai.pSetLayouts = &m_CullDescLayout;
    vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, &m_CullDescSet);

    VkDescriptorBufferInfo bi[4]{};
    bi[0] = { m_TreeMetadataBuffer->GetHandle(),  0, VK_WHOLE_SIZE };
    bi[1] = { m_FrustumUBO->GetHandle(),          0, VK_WHOLE_SIZE };
    bi[2] = { m_TreeIndirectBuffer->GetHandle(),  0, VK_WHOLE_SIZE };
    bi[3] = { m_TreeDrawCountBuffer->GetHandle(), 0, VK_WHOLE_SIZE };

    VkWriteDescriptorSet w[4]{};
    for (u32 i = 0; i < 4; ++i) {
        w[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet          = m_CullDescSet;
        w[i].dstBinding      = i;
        w[i].descriptorCount = 1;
        w[i].descriptorType  = (i == 1) ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                        : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo     = &bi[i];
    }
    vkUpdateDescriptorSets(VulkanHW.m_Device, 4, w, 0, nullptr);

    // Pipeline layout: 1 set + 20 B push (5 u32).
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0; pcr.size = 5 * sizeof(u32);
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1; plci.pSetLayouts = &m_CullDescLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &m_CullPipelineLayout);

    VkShaderModule cs = g_ShaderManager->Load("tree_cull.comp.spv");
    if (cs == VK_NULL_HANDLE) { Msg("![VK Trees] tree_cull.comp.spv load failed"); return; }

    VkPipelineShaderStageCreateInfo ss{};
    ss.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ss.stage = VK_SHADER_STAGE_COMPUTE_BIT; ss.module = cs; ss.pName = "main";

    VkComputePipelineCreateInfo cpi{};
    cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage = ss; cpi.layout = m_CullPipelineLayout;
    if (vkCreateComputePipelines(VulkanHW.m_Device, VK::PipelineCache::GetCacheObject(),
                                 1, &cpi, nullptr, &m_CullPipeline) != VK_SUCCESS) {
        Msg("![VK Trees] cull pipeline create failed"); m_CullPipeline = VK_NULL_HANDLE; return;
    }
    Msg("[VK Trees] Cull pipeline OK");
}

// ============================================================================
// Graphics set 0 — transforms SSBO (read by tree.vert).
// ============================================================================
void CTreeManager::CreateXformDescriptor()
{
    if (!m_TreeTransformsBuffer) return;

    // SSFX wind flow map (s_waves) — sampled in tree.vert for the crown flutter.
    // Load it + a linear/repeat sampler here so the wind branch has its texture.
    if (m_WaveSampler == VK_NULL_HANDLE) {
        VkSamplerCreateInfo sci{};
        sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter = VK_FILTER_LINEAR; sci.minFilter = VK_FILTER_LINEAR;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.maxLod = VK_LOD_CLAMP_NONE;
        vkCreateSampler(VulkanHW.m_Device, &sci, nullptr, &m_WaveSampler);
    }
    if (m_WaveTex == nullptr) {
        string_path full;
        FS.update_path(full, "$game_textures$", "fx\\wind_wave.dds");
        if (FS.exist(full)) {
            auto* tex = xr_new<VK::CVulkanTexture>();
            if (tex->LoadDDS(full, /*applyBCSwizzle*/ false)) { m_WaveTex = tex; Msg("[VK Tree] SSFX wind flow map loaded"); }
            else { Msg("![VK Tree] LoadDDS failed for fx\\wind_wave.dds — tree flutter flat"); xr_delete(tex); }
        } else {
            Msg("![VK Tree] fx\\wind_wave.dds not found — tree flutter flat");
        }
    }

    // Set 0: binding 0 = transforms SSBO (vertex), binding 1 = s_waves (vertex).
    VkDescriptorSetLayoutBinding b[2]{};
    b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[0].descriptorCount = 1; b[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    b[1].binding = 1; b[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b[1].descriptorCount = 1; b[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    VkDescriptorSetLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lci.bindingCount = 2; lci.pBindings = b;
    vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &m_XformDescLayout);

    VkDescriptorPoolSize ps[2] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 } };
    VkDescriptorPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = 1; pci.poolSizeCount = 2; pci.pPoolSizes = ps;
    vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &m_XformDescPool);

    VkDescriptorSetAllocateInfo dai{};
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = m_XformDescPool; dai.descriptorSetCount = 1;
    dai.pSetLayouts = &m_XformDescLayout;
    vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, &m_XformDescSet);

    VkDescriptorBufferInfo bi{ m_TreeTransformsBuffer->GetHandle(), 0, VK_WHOLE_SIZE };
    // s_waves view — fall back to the SSBO-only case is unsafe (shader samples it),
    // so bind the flow map; if it failed to load the tree still draws (flat wind).
    VkDescriptorImageInfo wi{ m_WaveSampler,
                              m_WaveTex ? m_WaveTex->GetView() : VK_NULL_HANDLE,
                              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkWriteDescriptorSet w[2]{};
    w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[0].dstSet = m_XformDescSet; w[0].dstBinding = 0; w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[0].pBufferInfo = &bi;
    w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[1].dstSet = m_XformDescSet; w[1].dstBinding = 1; w[1].descriptorCount = 1;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[1].pImageInfo = &wi;
    const u32 wc = (wi.imageView != VK_NULL_HANDLE) ? 2u : 1u;   // skip the null-view write
    vkUpdateDescriptorSets(VulkanHW.m_Device, wc, w, 0, nullptr);
}

// ============================================================================
// Graphics pipelines — tree.vert/frag, two variants by UV byte offset.
// Set 0 = transforms (m_XformDescLayout), Set 1 = diffuse (m_TexDescLayout).
// ============================================================================
void CTreeManager::CreateGfxPipelines()
{
    if (!g_ShaderManager) { Msg("![VK Trees] g_ShaderManager null — gfx pipeline disabled"); return; }
    if (m_XformDescLayout == VK_NULL_HANDLE || m_TexDescLayout == VK_NULL_HANDLE) return;

    // set0 = per-tree transforms (SSBO), set1 = per-group diffuse,
    // set2 = shared env lighting (sun_vp + sun shadow map — vk_env_light).
    VkDescriptorSetLayout setLayouts[3] = { m_XformDescLayout, m_TexDescLayout, VK::EnvLight::GetSetLayout() };
    if (setLayouts[2] == VK_NULL_HANDLE) { Msg("![VK Trees] EnvLight layout not ready"); return; }

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcr.offset = 0; pcr.size = sizeof(TreeGfxPush);
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 3; plci.pSetLayouts = setLayouts;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &m_GfxPipelineLayout);

    VkShaderModule vs = g_ShaderManager->Load("tree.vert.spv");
    VkShaderModule fs = g_ShaderManager->Load("tree.frag.spv");
    if (vs == VK_NULL_HANDLE || fs == VK_NULL_HANDLE) {
        Msg("![VK Trees] tree.{vert,frag}.spv load failed"); return;
    }

    auto createVariant = [&](u32 tcOffset, VkPipeline& out) -> bool
    {
        VkPipelineShaderStageCreateInfo ss[2]{};
        ss[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ss[0].stage = VK_SHADER_STAGE_VERTEX_BIT;   ss[0].module = vs; ss[0].pName = "main";
        ss[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ss[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; ss[1].module = fs; ss[1].pName = "main";

        // Vertex input: stride=32, FLOAT3 pos @ 0, SHORT2 SSCALED UV @ tcOffset.
        VkVertexInputBindingDescription vibd{ 0, 32, VK_VERTEX_INPUT_RATE_VERTEX };
        VkVertexInputAttributeDescription via[2]{};
        via[0] = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0 };
        via[1] = { 1, 0, VK_FORMAT_R16G16_SSCALED,   tcOffset };

        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &vibd;
        vi.vertexAttributeDescriptionCount = 2; vi.pVertexAttributeDescriptions = via;

        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vp{};
        vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vp.viewportCount = 1; vp.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode    = VK_CULL_MODE_NONE;   // trees: double-sided leaves
        rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth   = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp  = VK_COMPARE_OP_LESS_OR_EQUAL;

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
        pi.layout = m_GfxPipelineLayout;
        if (vkCreateGraphicsPipelines(VulkanHW.m_Device, VK::PipelineCache::GetCacheObject(),
                                      1, &pi, nullptr, &out) != VK_SUCCESS) {
            Msg("![VK Trees] gfx pipeline (tcOff=%u) create failed", tcOffset);
            out = VK_NULL_HANDLE; return false;
        }
        return true;
    };

    bool ok24 = createVariant(24, m_GfxPipeline24);
    bool ok28 = createVariant(28, m_GfxPipeline28);
    Msg("[VK Trees] Gfx pipeline (stride=32, tcOff 24=%s 28=%s)",
        ok24 ? "ok" : "FAIL", ok28 ? "ok" : "FAIL");

    // Shadow caster variants: tree_depth.{vert,frag} (alpha-tested, depth-only
    // into the sun map's D32), same pipeline layout (push offset 0..72 used).
    VkShaderModule dvs = g_ShaderManager->Load("tree_depth.vert.spv");
    VkShaderModule dfs = g_ShaderManager->Load("tree_depth.frag.spv");
    if (dvs == VK_NULL_HANDLE || dfs == VK_NULL_HANDLE) {
        Msg("![VK Trees] tree_depth.{vert,frag}.spv missing — tree shadow casting disabled");
        return;
    }
    auto createDepthVariant = [&](u32 tcOffset, VkPipeline& out)
    {
        VkPipelineShaderStageCreateInfo ss[2]{};
        ss[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ss[0].stage = VK_SHADER_STAGE_VERTEX_BIT;   ss[0].module = dvs; ss[0].pName = "main";
        ss[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ss[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; ss[1].module = dfs; ss[1].pName = "main";

        VkVertexInputBindingDescription vibd{ 0, 32, VK_VERTEX_INPUT_RATE_VERTEX };
        VkVertexInputAttributeDescription via[2]{};
        via[0] = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0 };
        via[1] = { 1, 0, VK_FORMAT_R16G16_SSCALED,   tcOffset };
        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &vibd;
        vi.vertexAttributeDescriptionCount = 2; vi.pVertexAttributeDescriptions = via;

        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vp{};
        vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vp.viewportCount = 1; vp.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode    = VK_CULL_MODE_NONE;   // double-sided leaves
        rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth   = 1.0f;
        rs.depthBiasEnable = VK_TRUE;         // dynamic — caller sets the sun bias

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp  = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineColorBlendStateCreateInfo cb{};   // no color attachments
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;

        VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS };
        VkPipelineDynamicStateCreateInfo dynState{};
        dynState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynState.dynamicStateCount = 3; dynState.pDynamicStates = dyn;

        VkPipelineRenderingCreateInfo prci{};
        prci.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        prci.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;   // sun shadow map

        VkGraphicsPipelineCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pi.pNext = &prci; pi.stageCount = 2; pi.pStages = ss;
        pi.pVertexInputState = &vi; pi.pInputAssemblyState = &ia;
        pi.pViewportState = &vp; pi.pRasterizationState = &rs;
        pi.pMultisampleState = &ms; pi.pDepthStencilState = &ds;
        pi.pColorBlendState = &cb; pi.pDynamicState = &dynState;
        pi.layout = m_GfxPipelineLayout;
        if (vkCreateGraphicsPipelines(VulkanHW.m_Device, VK::PipelineCache::GetCacheObject(),
                                      1, &pi, nullptr, &out) != VK_SUCCESS) {
            Msg("![VK Trees] depth pipeline (tcOff=%u) create failed", tcOffset);
            out = VK_NULL_HANDLE;
        }
    };
    createDepthVariant(24, m_DepthPipeline24);
    createDepthVariant(28, m_DepthPipeline28);
}

// ============================================================================
// Shadow caster path — depth-only alpha-tested draws of trees inside the sun
// ortho box. CPU-culled per tree (runs only on sun static-map redraws, so the
// metadata walk + per-tree draws are off the per-frame path).
// ============================================================================
void CTreeManager::RenderDepth(VkCommandBuffer cmd, const Fmatrix& lightVP, s32 cascade,
                               const CFrustum* frustum)
{
    if (!m_bBuilt || m_MetaCPU.empty()) return;
    if (m_XformDescSet == VK_NULL_HANDLE || m_GfxPipelineLayout == VK_NULL_HANDLE) return;
    if (m_DepthPipeline24 == VK_NULL_HANDLE && m_DepthPipeline28 == VK_NULL_HANDLE) return;

    // Push: light view·proj + tree UV quant + leaf alpha cutoff (first 72 B of
    // the TreeGfxPush range; sun/hemi tail unused by the depth shaders).
    TreeGfxPush pc{};
    pc.mViewProj = lightVP;
    pc.uvScale   = 1.0f / 2048.0f;
    pc.alphaRef  = 200.0f / 255.0f;
    // Same wind as the forward pass — the shadow caster must bend with the trees,
    // else a swaying trunk self-shadows (black band). See tree_depth.vert.
    pc.wsetup_trees.set(ps_r_wind_tree_anim, ps_r_wind_tree_trunk, ps_r_wind_tree_bend, 0.1f);
    if (g_pGamePersistent) {
        const Fvector3 wa = g_pGamePersistent->Environment().wind_anim;
        pc.wind_anim.set(wa.x, wa.y, wa.z, ps_r_wind_tree_flutter);
        if (auto* E = g_pGamePersistent->Environment().CurrentEnv)
            pc.wind_params.set(E->wind_direction, E->wind_velocity, 0.0f, 0.0f);
    }
    vkCmdPushConstants(cmd, m_GfxPipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(pc), &pc);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_GfxPipelineLayout,
                            0, 1, &m_XformDescSet, 0, nullptr);

    VkPipeline      lastPipe = VK_NULL_HANDLE;
    VkDescriptorSet lastTex  = VK_NULL_HANDLE;
    u32 nDraw = 0;

    for (const TreeIndirectGroup& grp : m_Groups)
    {
        VkPipeline pipe = (grp.tcOffset == 24) ? m_DepthPipeline24 : m_DepthPipeline28;
        if (pipe == VK_NULL_HANDLE) continue;

        bool boundGroup = false;
        for (u32 i = 0; i < grp.meshCount; ++i)
        {
            const GpuTreeMeta& m = m_MetaCPU[grp.meshOffset + i];
            if (frustum) {
                if (!frustum->testSphere_dirty(m.sphere_P, m.sphere_R)) continue;
            } else if (cascade >= 0 ? !ShadowMap::CascadeSphereVisible((u32)cascade, m.sphere_P, m.sphere_R)
                                    : !ShadowMap::SphereVisible(m.sphere_P, m.sphere_R)) continue;

            if (!boundGroup) {
                if (pipe != lastPipe) {
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
                    lastPipe = pipe;
                }
                if (grp.descSetIdx < m_TexDescSets.size() && m_TexDescSets[grp.descSetIdx] != lastTex) {
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_GfxPipelineLayout,
                                            1, 1, &m_TexDescSets[grp.descSetIdx], 0, nullptr);
                    lastTex = m_TexDescSets[grp.descSetIdx];
                }
                VkDeviceSize vbOff = 0;
                vkCmdBindVertexBuffers(cmd, 0, 1, &grp.vb, &vbOff);
                vkCmdBindIndexBuffer(cmd, grp.ib, 0, VK_INDEX_TYPE_UINT16);
                boundGroup = true;
            }
            // firstInstance = global tree index → gl_InstanceIndex picks the xform.
            vkCmdDrawIndexed(cmd, m.index_count, 1, m.ib_first, (s32)m.first_vertex, grp.meshOffset + i);
            ++nDraw;
        }
    }

    static bool s_diag = false;
    if (!s_diag && nDraw) { s_diag = true; Msg("[VK Trees] first shadow render: %u trees in the sun box", nDraw); }
}

// ============================================================================
// Per-frame render.
// ============================================================================
void CTreeManager::Render(VK::FrameContext& ctx)
{
    if (!IsReady()) return;
    if (ctx.cmd == VK_NULL_HANDLE || ctx.viewProj == nullptr) return;

    const VkCommandBuffer cmd = ctx.cmd;
    const Fmatrix& vp = *ctx.viewProj;
    const u32 numGroups = (u32)m_Groups.size();

    // ----- 1) Refresh frustum UBO (Gribb/Hartmann; same extraction as grass). -
    if (m_FrustumUBO && m_FrustumUBO->IsMapped())
    {
        TreeFrustumUBO* fu = (TreeFrustumUBO*)m_FrustumUBO->m_Mapped;
        VK::ExtractFrustumPlanes(vp, fu->planes);   // shared Gribb/Hartmann (vk_cull.h)
        m_FrustumUBO->Flush();
    }

    // ----- 2) Clear per-group draw counts; barrier transfer→compute. ----------
    // Same cross-frame WAR hazard as grass: m_TreeIndirectBuffer and
    // m_TreeDrawCountBuffer are single-buffered while 3 frames are in flight.
    // Order the previous frame's indirect fetches before this frame's
    // fill/compute rewrites (execution dependency is enough for WAR).
    {
        VkMemoryBarrier b{};
        b.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        b.srcAccessMask = 0;
        b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &b, 0, nullptr, 0, nullptr);
    }
    vkCmdFillBuffer(cmd, m_TreeDrawCountBuffer->GetHandle(), 0, VK_WHOLE_SIZE, 0u);
    {
        VkMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &b, 0, nullptr, 0, nullptr);
    }

    // ----- 3) Cull dispatch per group. ----------------------------------------
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_CullPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_CullPipelineLayout,
                            0, 1, &m_CullDescSet, 0, nullptr);
    for (u32 g = 0; g < numGroups; ++g)
    {
        const TreeIndirectGroup& grp = m_Groups[g];
        u32 pushData[5] = { grp.meshCount, 0u, grp.meshOffset, g * m_MaxGroupMeshCount, g };
        vkCmdPushConstants(cmd, m_CullPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pushData), pushData);
        vkCmdDispatch(cmd, (grp.meshCount + 255) / 256, 1, 1);
    }

    // ----- 4) Barrier compute→draw-indirect. ----------------------------------
    {
        VkMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                             0, 1, &b, 0, nullptr, 0, nullptr);
    }

    // ----- 5) Begin dynamic-rendering pass (LOAD color + depth from world). ----
    VK::BeginOverlayRendering(cmd, ctx);   // shared overlay begin — see vk_pass_context.h

    // Push constants + transforms (set 0) once — shared across both variants.
    TreeGfxPush pc{};
    pc.mViewProj = vp;
    // Tree UVs are quantized with FTreeVisual_quant = 32768/16 = 2048
    // (FTreeVisual.cpp tree_data consts.xy), NOT the static-geometry 1/1024.
    pc.uvScale   = 1.0f / 2048.0f;
    pc.alphaRef  = 200.0f / 255.0f;
    // Env lighting (colorize the baked per-tree hemi factor + open-sky sun term so
    // foliage tracks time-of-day like the world). Neutral fallback if env not up.
    pc.vSunColor.set(0.6f, 0.6f, 0.6f, 0.0f);
    pc.vHemiColor.set(0.45f, 0.45f, 0.45f, 0.0f);
    // SSFX trunk wind (screenspace_wind.h): direction/velocity drive the sway,
    // drift from the shared Environment.wind_anim. Defaults match SSFX "10 - Wind"
    // (branches 11.0, trunk 0.15, bend 0.5, min wind speed 0.1).
    pc.wsetup_trees.set(ps_r_wind_tree_anim, ps_r_wind_tree_trunk, ps_r_wind_tree_bend, 0.1f);
    if (g_pGamePersistent) {
        auto& envMgr = g_pGamePersistent->Environment();
        const Fvector3 wa = envMgr.wind_anim;
        pc.wind_anim.set(wa.x, wa.y, wa.z, ps_r_wind_tree_flutter);   // w = crown flutter amplitude
        if (auto* E = envMgr.CurrentEnv) {
            pc.vSunColor.set(E->sun_color.x, E->sun_color.y, E->sun_color.z, 0.0f);
            pc.vHemiColor.set(E->hemi_color.x, E->hemi_color.y, E->hemi_color.z, 0.0f);
            pc.wind_params.set(E->wind_direction, E->wind_velocity, 0.0f, 0.0f);
        }
    }
    pc.wind_params.w = ps_r_wind_tree_crown;   // leaf-flutter fade-in height (gate the low trunk)
    // Same global sun boost as the world (vk_env_light premultiplies it into
    // the LightUBO; the tree sun colour travels via push constants).
    pc.vSunColor.mul(ps_r_sun_boost);
    vkCmdPushConstants(cmd, m_GfxPipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(pc), &pc);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_GfxPipelineLayout,
                            0, 1, &m_XformDescSet, 0, nullptr);
    // set 2 = shared env lighting (sun_vp + sun shadow map) — updated by Pass_World.
    if (VkDescriptorSet envSet = VK::EnvLight::GetCurrentSet())
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_GfxPipelineLayout,
                                2, 1, &envSet, 0, nullptr);

    // ----- 6) Per-group indirect draw. ----------------------------------------
    VkPipeline      lastPipe = VK_NULL_HANDLE;
    VkDescriptorSet lastTex  = VK_NULL_HANDLE;
    for (u32 g = 0; g < numGroups; ++g)
    {
        const TreeIndirectGroup& grp = m_Groups[g];
        VkPipeline pipe = (grp.tcOffset == 24) ? m_GfxPipeline24 : m_GfxPipeline28;
        if (pipe == VK_NULL_HANDLE) continue;

        if (pipe != lastPipe) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
            lastPipe = pipe;
        }
        if (grp.descSetIdx < m_TexDescSets.size()) {
            VkDescriptorSet tex = m_TexDescSets[grp.descSetIdx];
            if (tex != lastTex) {
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_GfxPipelineLayout,
                                        1, 1, &tex, 0, nullptr);
                lastTex = tex;
            }
        }

        VkDeviceSize vbOff = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &grp.vb, &vbOff);
        vkCmdBindIndexBuffer(cmd, grp.ib, 0, VK_INDEX_TYPE_UINT16);

        const VkDeviceSize cmdOff = (VkDeviceSize)g * m_MaxGroupMeshCount * sizeof(VkDrawIndexedIndirectCommand);
        const VkDeviceSize cntOff = (VkDeviceSize)g * sizeof(u32);
        vkCmdDrawIndexedIndirectCount(cmd,
            m_TreeIndirectBuffer->GetHandle(), cmdOff,
            m_TreeDrawCountBuffer->GetHandle(), cntOff,
            m_MaxGroupMeshCount, sizeof(VkDrawIndexedIndirectCommand));
    }

    vkCmdEndRendering(cmd);

    static bool s_diag = false;
    if (!s_diag) {
        Msg("[VK Trees] First Render: %u trees in %u groups (maxGroupMesh=%u)",
            m_TotalCount, numGroups, m_MaxGroupMeshCount);
        s_diag = true;
    }
}

// ============================================================================
// Session B teardown — called from Destroy() before the buffers go.
// ============================================================================
void CTreeManager::DestroySessionB()
{
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
    VkDevice dev = VulkanHW.m_Device;

    if (m_GfxPipeline24)     { vkDestroyPipeline(dev, m_GfxPipeline24, nullptr); m_GfxPipeline24 = VK_NULL_HANDLE; }
    if (m_GfxPipeline28)     { vkDestroyPipeline(dev, m_GfxPipeline28, nullptr); m_GfxPipeline28 = VK_NULL_HANDLE; }
    if (m_DepthPipeline24)   { vkDestroyPipeline(dev, m_DepthPipeline24, nullptr); m_DepthPipeline24 = VK_NULL_HANDLE; }
    if (m_DepthPipeline28)   { vkDestroyPipeline(dev, m_DepthPipeline28, nullptr); m_DepthPipeline28 = VK_NULL_HANDLE; }
    m_MetaCPU.clear();
    if (m_GfxPipelineLayout) { vkDestroyPipelineLayout(dev, m_GfxPipelineLayout, nullptr); m_GfxPipelineLayout = VK_NULL_HANDLE; }

    if (m_XformDescPool)   { vkDestroyDescriptorPool(dev, m_XformDescPool, nullptr); m_XformDescPool = VK_NULL_HANDLE; m_XformDescSet = VK_NULL_HANDLE; }
    if (m_XformDescLayout) { vkDestroyDescriptorSetLayout(dev, m_XformDescLayout, nullptr); m_XformDescLayout = VK_NULL_HANDLE; }
    if (m_WaveTex)     { m_WaveTex->Destroy(); xr_delete(m_WaveTex); m_WaveTex = nullptr; }
    if (m_WaveSampler) { vkDestroySampler(dev, m_WaveSampler, nullptr); m_WaveSampler = VK_NULL_HANDLE; }

    if (m_CullPipeline)       { vkDestroyPipeline(dev, m_CullPipeline, nullptr); m_CullPipeline = VK_NULL_HANDLE; }
    if (m_CullPipelineLayout) { vkDestroyPipelineLayout(dev, m_CullPipelineLayout, nullptr); m_CullPipelineLayout = VK_NULL_HANDLE; }
    if (m_CullDescPool)       { vkDestroyDescriptorPool(dev, m_CullDescPool, nullptr); m_CullDescPool = VK_NULL_HANDLE; m_CullDescSet = VK_NULL_HANDLE; }
    if (m_CullDescLayout)     { vkDestroyDescriptorSetLayout(dev, m_CullDescLayout, nullptr); m_CullDescLayout = VK_NULL_HANDLE; }

    if (m_FrustumUBO) { m_FrustumUBO->Destroy(); xr_delete(m_FrustumUBO); }

    DestroyVsm();
}

// ============================================================================
// VSM caster path — trees into the virtual shadow atlas (static → temporal-safe).
// Mirrors RenderDepth, but each tree is instanced over the atlas pages it overlaps
// (binned by vsm_skinned_bin.comp — GpuTreeMeta shares SkinMeta's 32 B layout) and
// rasterized with tree_vsm_page.{vert,frag} which routes per page + alpha-tests leaves.
// ============================================================================
namespace { constexpr u32 kVsmTreeCap = 512; }   // max atlas pages a tree bins into (big crowns span many)

void CTreeManager::CreateVsmResources()
{
    if (m_VsmReady) return;
    if (m_XformDescLayout == VK_NULL_HANDLE || m_TexDescLayout == VK_NULL_HANDLE) return;   // Session B not up
    if (m_XformDescSet == VK_NULL_HANDLE) return;
    if (!m_TreeMetadataBuffer || !m_TreeTransformsBuffer || m_TotalCount == 0) return;
    if (!g_ShaderManager) return;

    VkShaderModule binCS = g_ShaderManager->Load("vsm_tree_bin.comp.spv");   // skinned-bin + slotDirty cache filter (STATIC atlas)
    VkShaderModule pvs   = g_ShaderManager->Load("tree_vsm_page.vert.spv");
    VkShaderModule pfs   = g_ShaderManager->Load("tree_vsm_page.frag.spv");
    if (!binCS || !pvs || !pfs) { Msg("![VK Trees] VSM caster shaders missing — tree VSM shadows disabled"); return; }

    VkDevice dev = VulkanHW.m_Device;
    const u32 N = VK_FRAMES_IN_FLIGHT;

    // Buffers.
    m_VsmCasterPages = xr_new<CVulkanBuffer>();
    m_VsmCasterPages->Create((VkDeviceSize)m_TotalCount * kVsmTreeCap * sizeof(u32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    m_VsmIndirect = xr_new<CVulkanBuffer>();
    m_VsmIndirect->Create((VkDeviceSize)m_TotalCount * sizeof(VkDrawIndexedIndirectCommand),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    m_VsmStats = xr_new<CVulkanBuffer>();
    m_VsmStats->Create(4 * sizeof(u32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

    // Pool: bin (5 SSBO + 1 UBO) + page (2 SSBO + 1 UBO), ×N.
    VkDescriptorPoolSize ps[2] = { { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, N * 8 }, { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, N * 2 } };   // bin 6 SSBO (+slotDirty) + page 2 SSBO
    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets = N * 2; pci.poolSizeCount = 2; pci.pPoolSizes = ps;
    if (vkCreateDescriptorPool(dev, &pci, nullptr, &m_VsmDescPool) != VK_SUCCESS) return;

    // Bin set layout (6 bindings) + N sets + pipeline.
    {
        VkDescriptorSetLayoutBinding b[7]{};
        const VkDescriptorType t[7] = {
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,   // 6: slotDirty (Phase 2 cache filter)
        };
        for (u32 i = 0; i < 7; ++i) { b[i].binding = i; b[i].descriptorType = t[i]; b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
        VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        lci.bindingCount = 7; lci.pBindings = b;
        if (vkCreateDescriptorSetLayout(dev, &lci, nullptr, &m_VsmBinSetL) != VK_SUCCESS) return;
        VkDescriptorSetLayout ls[VK_FRAMES_IN_FLIGHT]; for (u32 i = 0; i < N; ++i) ls[i] = m_VsmBinSetL;
        VkDescriptorSetAllocateInfo dai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        dai.descriptorPool = m_VsmDescPool; dai.descriptorSetCount = N; dai.pSetLayouts = ls;
        if (vkAllocateDescriptorSets(dev, &dai, m_VsmBinSet) != VK_SUCCESS) return;
        VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, 2 * sizeof(u32) };
        VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        plci.setLayoutCount = 1; plci.pSetLayouts = &m_VsmBinSetL; plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
        if (vkCreatePipelineLayout(dev, &plci, nullptr, &m_VsmBinLayout) != VK_SUCCESS) return;
        VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = binCS; cpci.stage.pName = "main"; cpci.layout = m_VsmBinLayout;
        if (vkCreateComputePipelines(dev, VK::PipelineCache::GetCacheObject(), 1, &cpci, nullptr, &m_VsmBinPipe) != VK_SUCCESS) return;
    }

    // Page set layout (3 bindings: pageList, casterPages, clipmap UBO) + N sets.
    {
        VkDescriptorSetLayoutBinding b[3]{};
        const VkDescriptorType t[3] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER };
        for (u32 i = 0; i < 3; ++i) { b[i].binding = i; b[i].descriptorType = t[i]; b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_VERTEX_BIT; }
        VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        lci.bindingCount = 3; lci.pBindings = b;
        if (vkCreateDescriptorSetLayout(dev, &lci, nullptr, &m_VsmPageSetL) != VK_SUCCESS) return;
        VkDescriptorSetLayout ls[VK_FRAMES_IN_FLIGHT]; for (u32 i = 0; i < N; ++i) ls[i] = m_VsmPageSetL;
        VkDescriptorSetAllocateInfo dai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        dai.descriptorPool = m_VsmDescPool; dai.descriptorSetCount = N; dai.pSetLayouts = ls;
        if (vkAllocateDescriptorSets(dev, &dai, m_VsmPageSet) != VK_SUCCESS) return;
    }

    // Page pipeline layout: set0 = transforms (reuse), set1 = diffuse (reuse), set2 = page data.
    {
        VkDescriptorSetLayout sets[3] = { m_XformDescLayout, m_TexDescLayout, m_VsmPageSetL };
        VkPushConstantRange pcr{ VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4 * sizeof(u32) };   // uvScale, alphaRef, cap, pad
        VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        plci.setLayoutCount = 3; plci.pSetLayouts = sets; plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
        if (vkCreatePipelineLayout(dev, &plci, nullptr, &m_VsmPageLayout) != VK_SUCCESS) return;
    }

    // Page pipelines (tcOffset 24 / 28) — depth-only into the D32 atlas, alpha-test FS.
    auto createPageVariant = [&](u32 tcOffset, VkPipeline& out) {
        VkPipelineShaderStageCreateInfo ss[2]{};
        ss[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; ss[0].stage = VK_SHADER_STAGE_VERTEX_BIT;   ss[0].module = pvs; ss[0].pName = "main";
        ss[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; ss[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; ss[1].module = pfs; ss[1].pName = "main";
        VkVertexInputBindingDescription vibd{ 0, 32, VK_VERTEX_INPUT_RATE_VERTEX };
        VkVertexInputAttributeDescription via[2]{};
        via[0] = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0 };
        via[1] = { 1, 0, VK_FORMAT_R16G16_SSCALED,   tcOffset };
        VkPipelineVertexInputStateCreateInfo vi{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
        vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &vibd;
        vi.vertexAttributeDescriptionCount = 2; vi.pVertexAttributeDescriptions = via;
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
        VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS };
        VkPipelineDynamicStateCreateInfo dynState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
        dynState.dynamicStateCount = 3; dynState.pDynamicStates = dyn;
        VkPipelineRenderingCreateInfo prci{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
        prci.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;
        VkGraphicsPipelineCreateInfo pi{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
        pi.pNext = &prci; pi.stageCount = 2; pi.pStages = ss;
        pi.pVertexInputState = &vi; pi.pInputAssemblyState = &ia; pi.pViewportState = &vp;
        pi.pRasterizationState = &rs; pi.pMultisampleState = &ms; pi.pDepthStencilState = &ds;
        pi.pColorBlendState = &cb; pi.pDynamicState = &dynState; pi.layout = m_VsmPageLayout;
        if (vkCreateGraphicsPipelines(dev, VK::PipelineCache::GetCacheObject(), 1, &pi, nullptr, &out) != VK_SUCCESS) {
            Msg("![VK Trees] VSM page pipeline (tcOff=%u) create failed", tcOffset); out = VK_NULL_HANDLE;
        }
    };
    createPageVariant(24, m_VsmPagePipe24);
    createPageVariant(28, m_VsmPagePipe28);

    m_VsmReady = true;
    Msg("[VK Trees] VSM caster path ready (%u trees, cap %u pages)", m_TotalCount, kVsmTreeCap);
}

void CTreeManager::VsmBin(VkCommandBuffer cmd, VkBuffer pageTable, VkBuffer slotDirty, VkBuffer clipmapUBO)
{
    if (!m_VsmReady) CreateVsmResources();
    if (!m_VsmReady || m_TotalCount == 0 || m_VsmBinPipe == VK_NULL_HANDLE) return;
    if (pageTable == VK_NULL_HANDLE || slotDirty == VK_NULL_HANDLE || clipmapUBO == VK_NULL_HANDLE) return;

    m_VsmSlot = (m_VsmSlot + 1) % VK_FRAMES_IN_FLIGHT;
    const u32 slot = m_VsmSlot;

    vkCmdFillBuffer(cmd, m_VsmStats->GetHandle(), 0, VK_WHOLE_SIZE, 0u);
    { VkMemoryBarrier b{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
      b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr); }

    VkDescriptorBufferInfo bi[7] = {
        { m_TreeMetadataBuffer->GetHandle(), 0, VK_WHOLE_SIZE }, { clipmapUBO,                     0, VK_WHOLE_SIZE },
        { pageTable,                         0, VK_WHOLE_SIZE }, { m_VsmCasterPages->GetHandle(),  0, VK_WHOLE_SIZE },
        { m_VsmIndirect->GetHandle(),        0, VK_WHOLE_SIZE }, { m_VsmStats->GetHandle(),        0, VK_WHOLE_SIZE },
        { slotDirty,                         0, VK_WHOLE_SIZE },   // 6: STATIC-atlas dirty set (cache filter)
    };
    const VkDescriptorType t[7] = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    };
    VkWriteDescriptorSet w[7]{};
    for (u32 i = 0; i < 7; ++i) { w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = m_VsmBinSet[slot]; w[i].dstBinding = i; w[i].descriptorCount = 1; w[i].descriptorType = t[i]; w[i].pBufferInfo = &bi[i]; }
    vkUpdateDescriptorSets(VulkanHW.m_Device, 7, w, 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_VsmBinPipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_VsmBinLayout, 0, 1, &m_VsmBinSet[slot], 0, nullptr);
    const u32 push[2] = { m_TotalCount, kVsmTreeCap };
    vkCmdPushConstants(cmd, m_VsmBinLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
    vkCmdDispatch(cmd, (m_TotalCount + 63) / 64, 1, 1);
}

void CTreeManager::VsmRender(VkCommandBuffer cmd, VkBuffer pageList, VkBuffer clipmapUBO)
{
    if (!m_VsmReady || m_TotalCount == 0 || m_XformDescSet == VK_NULL_HANDLE) return;
    if (m_VsmPagePipe24 == VK_NULL_HANDLE && m_VsmPagePipe28 == VK_NULL_HANDLE) return;
    if (pageList == VK_NULL_HANDLE || clipmapUBO == VK_NULL_HANDLE) return;
    const u32 slot = m_VsmSlot;

    VkDescriptorBufferInfo bi[3] = {
        { pageList,                      0, VK_WHOLE_SIZE },
        { m_VsmCasterPages->GetHandle(), 0, VK_WHOLE_SIZE },
        { clipmapUBO,                    0, VK_WHOLE_SIZE },
    };
    const VkDescriptorType t[3] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER };
    VkWriteDescriptorSet w[3]{};
    for (u32 i = 0; i < 3; ++i) { w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = m_VsmPageSet[slot]; w[i].dstBinding = i; w[i].descriptorCount = 1; w[i].descriptorType = t[i]; w[i].pBufferInfo = &bi[i]; }
    vkUpdateDescriptorSets(VulkanHW.m_Device, 3, w, 0, nullptr);

    struct { float uvScale, alphaRef; u32 cap, pad; } pc{ 1.0f / 2048.0f, 200.0f / 255.0f, kVsmTreeCap, 0u };
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_VsmPageLayout, 0, 1, &m_XformDescSet, 0, nullptr);       // set0 transforms
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_VsmPageLayout, 2, 1, &m_VsmPageSet[slot], 0, nullptr);   // set2 page data
    vkCmdPushConstants(cmd, m_VsmPageLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

    VkPipeline lastPipe = VK_NULL_HANDLE; VkDescriptorSet lastTex = VK_NULL_HANDLE;
    for (const TreeIndirectGroup& grp : m_Groups) {
        VkPipeline pipe = (grp.tcOffset == 24) ? m_VsmPagePipe24 : m_VsmPagePipe28;
        if (pipe == VK_NULL_HANDLE || grp.descSetIdx >= m_TexDescSets.size()) continue;
        if (pipe != lastPipe) { vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe); lastPipe = pipe; lastTex = VK_NULL_HANDLE; }
        VkDescriptorSet tex = m_TexDescSets[grp.descSetIdx];
        if (tex != lastTex) { vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_VsmPageLayout, 1, 1, &tex, 0, nullptr); lastTex = tex; }
        VkDeviceSize vbOff = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &grp.vb, &vbOff);
        vkCmdBindIndexBuffer(cmd, grp.ib, 0, VK_INDEX_TYPE_UINT16);
        // [#7] ONE multi-draw indirect for the whole group — the group's trees are contiguous
        // in m_VsmIndirect (tree-index order), so a single call replaces meshCount individual
        // draws. Cuts ~6288 draw calls/frame to one per group (~tens). Trees with no dirty pages
        // carry instanceCount=0 from the bin → ~free on the GPU. Needs multiDrawIndirect +
        // drawIndirectFirstInstance (both enabled in HW_Vulkan).
        vkCmdDrawIndexedIndirect(cmd, m_VsmIndirect->GetHandle(),
                                 (VkDeviceSize)grp.meshOffset * sizeof(VkDrawIndexedIndirectCommand),
                                 grp.meshCount, sizeof(VkDrawIndexedIndirectCommand));
    }
}

void CTreeManager::DestroyVsm()
{
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
    VkDevice dev = VulkanHW.m_Device;
    if (m_VsmPagePipe24) { vkDestroyPipeline(dev, m_VsmPagePipe24, nullptr); m_VsmPagePipe24 = VK_NULL_HANDLE; }
    if (m_VsmPagePipe28) { vkDestroyPipeline(dev, m_VsmPagePipe28, nullptr); m_VsmPagePipe28 = VK_NULL_HANDLE; }
    if (m_VsmPageLayout) { vkDestroyPipelineLayout(dev, m_VsmPageLayout, nullptr); m_VsmPageLayout = VK_NULL_HANDLE; }
    if (m_VsmBinPipe)    { vkDestroyPipeline(dev, m_VsmBinPipe, nullptr); m_VsmBinPipe = VK_NULL_HANDLE; }
    if (m_VsmBinLayout)  { vkDestroyPipelineLayout(dev, m_VsmBinLayout, nullptr); m_VsmBinLayout = VK_NULL_HANDLE; }
    if (m_VsmDescPool)   { vkDestroyDescriptorPool(dev, m_VsmDescPool, nullptr); m_VsmDescPool = VK_NULL_HANDLE;
                           for (u32 i = 0; i < VK_FRAMES_IN_FLIGHT; ++i) { m_VsmBinSet[i] = VK_NULL_HANDLE; m_VsmPageSet[i] = VK_NULL_HANDLE; } }
    if (m_VsmBinSetL)    { vkDestroyDescriptorSetLayout(dev, m_VsmBinSetL, nullptr); m_VsmBinSetL = VK_NULL_HANDLE; }
    if (m_VsmPageSetL)   { vkDestroyDescriptorSetLayout(dev, m_VsmPageSetL, nullptr); m_VsmPageSetL = VK_NULL_HANDLE; }
    if (m_VsmCasterPages) { m_VsmCasterPages->Destroy(); xr_delete(m_VsmCasterPages); }
    if (m_VsmIndirect)    { m_VsmIndirect->Destroy(); xr_delete(m_VsmIndirect); }
    if (m_VsmStats)       { m_VsmStats->Destroy(); xr_delete(m_VsmStats); }
    m_VsmReady = false; m_VsmSlot = 0;
}

}  // namespace VK
