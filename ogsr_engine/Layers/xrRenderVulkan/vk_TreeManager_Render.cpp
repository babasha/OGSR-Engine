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
extern int   ps_r_vsm_tree_wind;      // r_vsm_tree_wind — near/far hybrid: near trees cast into the DYNAMIC atlas with live wind
extern float ps_r_vsm_tree_wind_dist; // r_vsm_tree_wind_dist — near set radius (m)
extern int   ps_r_vsm_debug;          // r_vsm_debug — hybrid stats log
extern int   ps_r_vsm_meshlet;        // r_vsm_meshlet — per-page meshlet cull of tree casters (Phase B)
extern int   ps_r_vsm_hzb;            // r_vsm_hzb — shadow-HZB occlusion cull of tree casters (static pass)
extern float ps_r_vsm_hzb_margin;     // r_vsm_hzb_margin — occluder depth slack (stale-sun safety)

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

    // Pipeline layout: 1 set + push = frustum planes (6×vec4 = 96 B) + 5 u32 (20 B).
    // The frustum used to live in a single-buffered UBO (binding 1) but that raced
    // the in-flight frames during rotation (the next frame's CPU write clobbered this
    // frame's planes before the GPU cull read them) → the colour cull tested a rotated
    // frustum vs the depth prepass → black tree silhouettes at screen edges. Push
    // constants are recorded per dispatch, so they can't alias across frames.
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcr.offset = 0; pcr.size = sizeof(TreeCullPush);   // 116 B (< 128 B push limit)
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
                               const CFrustum* frustum, float minDist, float maxDist)
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
    pc.wind_params.w = ps_r_wind_tree_crown;   // flutter height gate — must match the forward pass
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
            // Distance split (wind shadows): keep near trees in the per-frame
            // dynamic layer and far trees in the cached static layer.
            if (minDist > 0.0f || maxDist < 1e9f) {
                const float d = Device.vCameraPosition.distance_to(m.sphere_P);
                if (d < minDist || d >= maxDist) continue;
            }
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

    // ----- 1) Extract this frame's frustum planes (Gribb/Hartmann; same as grass).
    // These go into the per-dispatch PUSH below — NOT the shared UBO — so a rotating
    // camera's next-frame planes can't clobber this frame's cull (the old single-
    // buffered UBO did, producing black tree silhouettes at the screen edge).
    TreeCullPush cullPush{};
    VK::ExtractFrustumPlanes(vp, cullPush.planes);

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
        // planes stay as extracted above; only the per-group indices change.
        cullPush.mesh_count  = grp.meshCount;
        cullPush._unused     = 0u;
        cullPush.mesh_offset = grp.meshOffset;
        cullPush.output_base = g * m_MaxGroupMeshCount;
        cullPush.count_index = g;
        vkCmdPushConstants(cmd, m_CullPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(cullPush), &cullPush);
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
// Max atlas pages a tree bins into. 512 was enough for the STATIC path (dirty-filtered:
// a tree touches only a handful of pages per frame) but the DYNAMIC wind pass bins a NEAR
// tree into ALL its resident pages every frame — a 10m crown a few metres away covers
// hundreds of L0 pages alone (L0 page = base/32 ≈ 0.75 m) and overflowed 512 → truncated
// page lists → near tree shadows visibly fell apart / vanished. 1024 covers it (watch
// "maxPages" in the r_vsm_debug hybrid log; the dyn pool ceiling is VSM_MAX_PHYS=2048).
namespace { constexpr u32 kVsmTreeCap = 1024; }

// Phase B: meshlet-cull command budget. Commands are POOLED per group; cap[g] =
// max(meshCount*kCmdPerTree, kGroupFloor). The per-tree term scales big species groups;
// the FLOOR covers small groups of a few large near trees (a single near crown bins
// ~maxPages×fewMeshlets ≈ up to ~430 draws — a 2-tree rare-species group would starve on
// meshCount*k alone → that was the steady-state [TRUNCATED] in the first test). Bases are a
// prefix sum of cap[] (not meshOffset*k). Both static + dynamic buffers share this layout.
namespace { constexpr u32 kCmdPerTree = 128; constexpr u32 kGroupFloor = 8192; }

void CTreeManager::CreateVsmResources()
{
    if (m_VsmReady) return;
    if (m_XformDescLayout == VK_NULL_HANDLE || m_TexDescLayout == VK_NULL_HANDLE) return;   // Session B not up
    if (m_XformDescSet == VK_NULL_HANDLE) return;
    if (!m_TreeMetadataBuffer || !m_TreeTransformsBuffer || m_TotalCount == 0) return;
    if (!g_ShaderManager) return;

    VkShaderModule binCS = g_ShaderManager->Load("vsm_tree_bin.comp.spv");   // near/far hybrid bin (mode 0 static / 1 dynamic)
    VkShaderModule pvs   = g_ShaderManager->Load("tree_vsm_page.vert.spv");        // STATIC atlas (rigid)
    VkShaderModule pvsD  = g_ShaderManager->Load("tree_vsm_page_dyn.vert.spv");    // DYNAMIC atlas (wind)
    VkShaderModule pfs   = g_ShaderManager->Load("tree_vsm_page.frag.spv");
    if (!binCS || !pvs || !pvsD || !pfs) { Msg("![VK Trees] VSM caster shaders missing — tree VSM shadows disabled"); return; }

    VkDevice dev = VulkanHW.m_Device;
    const u32 N = VK_FRAMES_IN_FLIGHT;

    // Buffers.
    m_VsmCasterPages = xr_new<CVulkanBuffer>();
    m_VsmCasterPages->Create((VkDeviceSize)m_TotalCount * kVsmTreeCap * sizeof(u32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    m_VsmIndirect = xr_new<CVulkanBuffer>();
    m_VsmIndirect->Create((VkDeviceSize)m_TotalCount * sizeof(VkDrawIndexedIndirectCommand),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    m_VsmDynIndirect = xr_new<CVulkanBuffer>();
    m_VsmDynIndirect->Create((VkDeviceSize)m_TotalCount * sizeof(VkDrawIndexedIndirectCommand),
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    m_VsmStats = xr_new<CVulkanBuffer>();
    m_VsmStats->Create(8 * sizeof(u32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);   // [4]=shadow-HZB culled
    m_VsmStatsRB = xr_new<CVulkanBuffer>();
    m_VsmStatsRB->Create(8 * sizeof(u32), VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
    m_VsmStatsPtr = (u32*)m_VsmStatsRB->Map();
    if (m_VsmStatsPtr) memset(m_VsmStatsPtr, 0, 8 * sizeof(u32));
    for (u32 i = 0; i < N; ++i) {   // per-frame CPU near-set flags (host-visible ring)
        m_VsmNearFlags[i] = xr_new<CVulkanBuffer>();
        m_VsmNearFlags[i]->Create((VkDeviceSize)m_TotalCount * sizeof(u32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
        m_VsmNearPtr[i] = m_VsmNearFlags[i]->Map();
        if (m_VsmNearPtr[i]) memset(m_VsmNearPtr[i], 0, m_TotalCount * sizeof(u32));
    }

    // Pool: bin (10 SSBO + 1 UBO) ×2N (static + dynamic) + page (2 SSBO + 1 UBO) ×2N.
    VkDescriptorPoolSize ps[2] = { { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, N * 28 }, { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, N * 4 } };
    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets = N * 4; pci.poolSizeCount = 2; pci.pPoolSizes = ps;
    if (vkCreateDescriptorPool(dev, &pci, nullptr, &m_VsmDescPool) != VK_SUCCESS) return;

    // Bin set layout (11 bindings: +7 dynUsed, +8 nearFlags, +9 pageMax, +10 staticPageTable=Option A) + N static + N dyn sets.
    {
        VkDescriptorSetLayoutBinding b[11]{};
        VkDescriptorType t[11];
        for (u32 i = 0; i < 11; ++i) t[i] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        t[1] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        for (u32 i = 0; i < 11; ++i) { b[i].binding = i; b[i].descriptorType = t[i]; b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
        VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        lci.bindingCount = 11; lci.pBindings = b;
        if (vkCreateDescriptorSetLayout(dev, &lci, nullptr, &m_VsmBinSetL) != VK_SUCCESS) return;
        VkDescriptorSetLayout ls[VK_FRAMES_IN_FLIGHT]; for (u32 i = 0; i < N; ++i) ls[i] = m_VsmBinSetL;
        VkDescriptorSetAllocateInfo dai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        dai.descriptorPool = m_VsmDescPool; dai.descriptorSetCount = N; dai.pSetLayouts = ls;
        if (vkAllocateDescriptorSets(dev, &dai, m_VsmBinSet) != VK_SUCCESS) return;
        if (vkAllocateDescriptorSets(dev, &dai, m_VsmDynBinSet) != VK_SUCCESS) return;
        VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, 6 * sizeof(u32) };   // count, cap, mode, hzbOn, margin(float), pad
        VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        plci.setLayoutCount = 1; plci.pSetLayouts = &m_VsmBinSetL; plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
        if (vkCreatePipelineLayout(dev, &plci, nullptr, &m_VsmBinLayout) != VK_SUCCESS) return;
        VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = binCS; cpci.stage.pName = "main"; cpci.layout = m_VsmBinLayout;
        if (vkCreateComputePipelines(dev, VK::PipelineCache::GetCacheObject(), 1, &cpci, nullptr, &m_VsmBinPipe) != VK_SUCCESS) return;
    }

    // Page set layout (3 bindings: pageList, casterPages, clipmap UBO) + N static + N dynamic sets.
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
        if (vkAllocateDescriptorSets(dev, &dai, m_VsmDynPageSet) != VK_SUCCESS) return;
    }

    // Page pipeline layout: set0 = transforms (reuse), set1 = diffuse (reuse), set2 = page data.
    {
        VkDescriptorSetLayout sets[3] = { m_XformDescLayout, m_TexDescLayout, m_VsmPageSetL };
        // 16 B base (uvScale, alphaRef, cap, pad) + 48 B TEST wind (wind_params,
        // wsetup_trees, wind_anim) for r_vsm_tree_wind. See tree_vsm_page.vert.
        VkPushConstantRange pcr{ VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 64 };
        VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        plci.setLayoutCount = 3; plci.pSetLayouts = sets; plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
        if (vkCreatePipelineLayout(dev, &plci, nullptr, &m_VsmPageLayout) != VK_SUCCESS) return;
    }

    // Page pipelines (tcOffset 24 / 28 × static/dynamic VS) — depth-only into the D32 atlas, alpha-test FS.
    auto createPageVariant = [&](VkShaderModule vs, u32 tcOffset, VkPipeline& out) {
        VkPipelineShaderStageCreateInfo ss[2]{};
        ss[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; ss[0].stage = VK_SHADER_STAGE_VERTEX_BIT;   ss[0].module = vs; ss[0].pName = "main";
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
    createPageVariant(pvs,  24, m_VsmPagePipe24);
    createPageVariant(pvs,  28, m_VsmPagePipe28);
    createPageVariant(pvsD, 24, m_VsmPageDynPipe24);
    createPageVariant(pvsD, 28, m_VsmPageDynPipe28);

    m_VsmReady = true;
    Msg("[VK Trees] VSM caster path ready (%u trees, cap %u pages, wind hybrid %s dist %.0fm)",
        m_TotalCount, kVsmTreeCap, ps_r_vsm_tree_wind ? "ON" : "off", ps_r_vsm_tree_wind_dist);

    CreateMeshletVsmResources();   // Phase B: stage-2 meshlet-cull path (r_vsm_meshlet)
}

bool CTreeManager::IsMeshletMode() const
{
    return ps_r_vsm_meshlet && m_MeshletVsmReady && m_MeshletsReady;
}

// ============================================================================
// Phase B — stage-2 meshlet-cull resources: the meshlet-bin compute pipeline, the
// meshlet page pipelines (reuse m_VsmPageLayout / set2), and the command/group buffers.
// Best-effort: any failure leaves m_MeshletVsmReady=false → the old per-tree path runs.
// ⛔ NO NET PERF GAIN on dGPU (2026-07-02) — kept but DISABLED (r_vsm_meshlet=0). Reached only when
//    the cvar is on; inert by default. Don't re-chase — see vk_console_min.cpp cvar block + memory.
// ============================================================================
void CTreeManager::CreateMeshletVsmResources()
{
    if (m_MeshletVsmReady) return;
    if (!m_MeshletsReady || !m_MeshletBuffer || !m_MeshletIndexBuffer || !m_TreeMeshletRangeBuffer) return;
    if (m_VsmPageLayout == VK_NULL_HANDLE || m_TotalCount == 0) return;
    if (!g_ShaderManager) return;

    VkDevice   dev = VulkanHW.m_Device;
    const u32  N   = VK_FRAMES_IN_FLIGHT;

    VkShaderModule binCS = g_ShaderManager->Load("vsm_tree_meshlet_bin.comp.spv");
    VkShaderModule pvs   = g_ShaderManager->Load("tree_vsm_meshlet_page.vert.spv");
    VkShaderModule pvsD  = g_ShaderManager->Load("tree_vsm_meshlet_page_dyn.vert.spv");
    VkShaderModule pfs   = g_ShaderManager->Load("tree_vsm_page.frag.spv");   // reuse the alpha-test FS
    if (!binCS || !pvs || !pvsD || !pfs) { Msg("![VK Trees] meshlet shaders missing — r_vsm_meshlet disabled"); return; }

    // ----- CPU: per-tree group index + per-group (cmdBase, cmdCap). Groups are contiguous
    // tree ranges, so the pooled section [meshOffset*kCmdPerTree, +meshCount*kCmdPerTree) is
    // contiguous and renderable per group with one DrawIndexedIndirectCount.
    const u32 groupN = (u32)m_Groups.size();
    xr_vector<u32>  treeGroup(m_TotalCount, 0u);
    xr_vector<u32>  groupInfo(groupN * 2u, 0u);   // uvec2 (cmdBase, cmdCap) per group
    m_MeshletGroupBase.assign(groupN, 0u);
    m_MeshletGroupCap.assign(groupN, 0u);
    u32 cursor = 0;   // running command-slot cursor -> per-group base (prefix sum of caps)
    for (u32 g = 0; g < groupN; ++g) {
        const TreeIndirectGroup& grp = m_Groups[g];
        u32 cap = grp.meshCount * kCmdPerTree;
        if (cap < kGroupFloor) cap = kGroupFloor;
        groupInfo[g * 2 + 0] = cursor;
        groupInfo[g * 2 + 1] = cap;
        m_MeshletGroupBase[g] = cursor;
        m_MeshletGroupCap[g]  = cap;
        cursor += cap;
        for (u32 i = 0; i < grp.meshCount; ++i)
            if (grp.meshOffset + i < m_TotalCount) treeGroup[grp.meshOffset + i] = g;
    }
    const VkDeviceSize cmdBytes = (VkDeviceSize)cursor * sizeof(VkDrawIndexedIndirectCommand);

    auto mkBuf = [&](CVulkanBuffer*& b, VkDeviceSize sz, VkBufferUsageFlags usage, VmaMemoryUsage mem) {
        b = xr_new<CVulkanBuffer>(); b->Create(sz, usage, mem);
    };
    mkBuf(m_VsmMeshletCmd,        cmdBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    mkBuf(m_VsmMeshletCmdDyn,     cmdBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    mkBuf(m_VsmMeshletGroupCount,    (VkDeviceSize)groupN * sizeof(u32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    mkBuf(m_VsmMeshletGroupCountDyn, (VkDeviceSize)groupN * sizeof(u32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    mkBuf(m_TreeGroupBuffer, (VkDeviceSize)m_TotalCount * sizeof(u32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    mkBuf(m_GroupInfoBuffer, (VkDeviceSize)groupN * 2 * sizeof(u32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    m_TreeGroupBuffer->Upload(treeGroup.data(), treeGroup.size() * sizeof(u32));
    m_GroupInfoBuffer->Upload(groupInfo.data(), groupInfo.size() * sizeof(u32));
    mkBuf(m_MeshletStats, 4 * sizeof(u32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    mkBuf(m_MeshletStatsRB, 4 * sizeof(u32), VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
    m_MeshletStatsPtr = (u32*)m_MeshletStatsRB->Map();
    if (m_MeshletStatsPtr) memset(m_MeshletStatsPtr, 0, 4 * sizeof(u32));

    // ----- Descriptor pool + layout (13 bindings, all SSBO except binding 1 = clipmap UBO).
    VkDescriptorPoolSize ps[2] = { { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, N * 2 * 12 }, { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, N * 2 } };
    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets = N * 2; pci.poolSizeCount = 2; pci.pPoolSizes = ps;
    if (vkCreateDescriptorPool(dev, &pci, nullptr, &m_MeshletDescPool) != VK_SUCCESS) return;

    {
        VkDescriptorSetLayoutBinding b[13]{};
        for (u32 i = 0; i < 13; ++i) { b[i].binding = i; b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
        b[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;   // VsmParams
        VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        lci.bindingCount = 13; lci.pBindings = b;
        if (vkCreateDescriptorSetLayout(dev, &lci, nullptr, &m_MeshletBinSetL) != VK_SUCCESS) return;
        VkDescriptorSetLayout ls[VK_FRAMES_IN_FLIGHT]; for (u32 i = 0; i < N; ++i) ls[i] = m_MeshletBinSetL;
        VkDescriptorSetAllocateInfo dai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        dai.descriptorPool = m_MeshletDescPool; dai.descriptorSetCount = N; dai.pSetLayouts = ls;
        if (vkAllocateDescriptorSets(dev, &dai, m_MeshletBinSet) != VK_SUCCESS) return;
        if (vkAllocateDescriptorSets(dev, &dai, m_MeshletDynBinSet) != VK_SUCCESS) return;
        VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, 4 * sizeof(u32) };   // casterCount, cap, slop(float), pad
        VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        plci.setLayoutCount = 1; plci.pSetLayouts = &m_MeshletBinSetL; plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
        if (vkCreatePipelineLayout(dev, &plci, nullptr, &m_MeshletBinLayout) != VK_SUCCESS) return;
        VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = binCS; cpci.stage.pName = "main"; cpci.layout = m_MeshletBinLayout;
        if (vkCreateComputePipelines(dev, VK::PipelineCache::GetCacheObject(), 1, &cpci, nullptr, &m_MeshletBinPipe) != VK_SUCCESS) return;
    }

    // ----- Meshlet page pipelines (reuse m_VsmPageLayout: set0 xform, set1 tex, set2 page).
    auto createPageVariant = [&](VkShaderModule vs, u32 tcOffset, VkPipeline& out) {
        VkPipelineShaderStageCreateInfo ss[2]{};
        ss[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; ss[0].stage = VK_SHADER_STAGE_VERTEX_BIT;   ss[0].module = vs;  ss[0].pName = "main";
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
            Msg("![VK Trees] meshlet page pipeline (tcOff=%u) create failed", tcOffset); out = VK_NULL_HANDLE;
        }
    };
    createPageVariant(pvs,  24, m_MeshletPagePipe24);
    createPageVariant(pvs,  28, m_MeshletPagePipe28);
    createPageVariant(pvsD, 24, m_MeshletPageDynPipe24);
    createPageVariant(pvsD, 28, m_MeshletPageDynPipe28);

    m_MeshletVsmReady = true;
    Msg("[VK Trees] Meshlet VSM path ready (%u clusters, %u cmd/tree, %u KB cmd x2)",
        m_MeshletTotal, kCmdPerTree, (u32)(cmdBytes / 1024));
}

// Stage 2: refine the per-tree page list into per-(meshlet,page) draws. mode 0 = static
// (m_VsmIndirect / m_VsmMeshletCmd), mode 1 = dynamic (m_VsmDynIndirect / m_VsmMeshletCmdDyn).
// pageList is that atlas's slot->page map. Runs right after the matching stage-1 dispatch.
void CTreeManager::DispatchMeshletBin(VkCommandBuffer cmd, u32 mode, VkBuffer pageList, VkBuffer clipmapUBO)
{
    if (!m_MeshletVsmReady) return;
    const u32 slot = m_VsmSlot;
    const bool  isDyn = (mode == 1u);
    VkDescriptorSet set   = isDyn ? m_MeshletDynBinSet[slot] : m_MeshletBinSet[slot];
    VkBuffer indirect     = isDyn ? m_VsmDynIndirect->GetHandle()          : m_VsmIndirect->GetHandle();
    VkBuffer outCmd       = isDyn ? m_VsmMeshletCmdDyn->GetHandle()        : m_VsmMeshletCmd->GetHandle();
    VkBuffer groupCnt     = isDyn ? m_VsmMeshletGroupCountDyn->GetHandle() : m_VsmMeshletGroupCount->GetHandle();

    // Clear this pass's group append-counter (= draw count). Clear stats once (static pass).
    vkCmdFillBuffer(cmd, groupCnt, 0, VK_WHOLE_SIZE, 0u);
    if (!isDyn) vkCmdFillBuffer(cmd, m_MeshletStats->GetHandle(), 0, VK_WHOLE_SIZE, 0u);
    // Order: stage-1 writes (casterPages/indirect) + this clear → stage-2 reads/writes.
    { VkMemoryBarrier b{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
      b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
      b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr); }

    VkDescriptorBufferInfo bi[13] = {
        { m_TreeMetadataBuffer->GetHandle(),      0, VK_WHOLE_SIZE },   // 0 meta
        { clipmapUBO,                             0, VK_WHOLE_SIZE },   // 1 VsmParams (UBO)
        { m_TreeTransformsBuffer->GetHandle(),    0, VK_WHOLE_SIZE },   // 2 xform
        { m_MeshletBuffer->GetHandle(),           0, VK_WHOLE_SIZE },   // 3 meshlets
        { m_TreeMeshletRangeBuffer->GetHandle(),  0, VK_WHOLE_SIZE },   // 4 tree range
        { m_VsmCasterPages->GetHandle(),          0, VK_WHOLE_SIZE },   // 5 caster pages (stage 1)
        { pageList,                               0, VK_WHOLE_SIZE },   // 6 slot -> page
        { indirect,                               0, VK_WHOLE_SIZE },   // 7 stage-1 indirect (pageCount)
        { outCmd,                                 0, VK_WHOLE_SIZE },   // 8 out commands
        { groupCnt,                               0, VK_WHOLE_SIZE },   // 9 group counter
        { m_TreeGroupBuffer->GetHandle(),         0, VK_WHOLE_SIZE },   // 10 tree -> group
        { m_GroupInfoBuffer->GetHandle(),         0, VK_WHOLE_SIZE },   // 11 group (base,cap)
        { m_MeshletStats->GetHandle(),            0, VK_WHOLE_SIZE },   // 12 stats
    };
    VkDescriptorType t[13]; for (u32 i = 0; i < 13; ++i) t[i] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; t[1] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    VkWriteDescriptorSet w[13]{};
    for (u32 i = 0; i < 13; ++i) { w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = set; w[i].dstBinding = i; w[i].descriptorCount = 1; w[i].descriptorType = t[i]; w[i].pBufferInfo = &bi[i]; }
    vkUpdateDescriptorSets(VulkanHW.m_Device, 13, w, 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_MeshletBinPipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_MeshletBinLayout, 0, 1, &set, 0, nullptr);
    struct { u32 casterCount, cap; float slop; u32 pad; } push{ m_TotalCount, kVsmTreeCap, isDyn ? 0.5f : 0.0f, 0u };
    vkCmdPushConstants(cmd, m_MeshletBinLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd, m_TotalCount, 1, 1);   // one workgroup per tree
}

// Meshlet render for one atlas. Per group: bind meshlet VS + diffuse + group VB + the shared
// meshlet IB, then DrawIndexedIndirectCount over the group's pooled command section.
void CTreeManager::DrawMeshlets(VkCommandBuffer cmd, u32 mode, VkDescriptorSet pageSet, bool wind)
{
    const bool isDyn = (mode == 1u);
    VkPipeline p24 = isDyn ? m_MeshletPageDynPipe24 : m_MeshletPagePipe24;
    VkPipeline p28 = isDyn ? m_MeshletPageDynPipe28 : m_MeshletPagePipe28;
    if (p24 == VK_NULL_HANDLE && p28 == VK_NULL_HANDLE) return;
    VkBuffer cmdBuf   = isDyn ? m_VsmMeshletCmdDyn->GetHandle()        : m_VsmMeshletCmd->GetHandle();
    VkBuffer countBuf = isDyn ? m_VsmMeshletGroupCountDyn->GetHandle() : m_VsmMeshletGroupCount->GetHandle();

    struct VsmPagePush {
        float    uvScale, alphaRef; u32 cap, pad;
        Fvector4 wind_params, wsetup_trees, wind_anim;
    } pc{};
    pc.uvScale = 1.0f / 2048.0f; pc.alphaRef = 200.0f / 255.0f; pc.cap = kVsmTreeCap;
    if (wind) {
        pc.wsetup_trees.set(ps_r_wind_tree_anim, ps_r_wind_tree_trunk, ps_r_wind_tree_bend, 0.1f);
        if (g_pGamePersistent) {
            const Fvector3 wa = g_pGamePersistent->Environment().wind_anim;
            pc.wind_anim.set(wa.x, wa.y, wa.z, ps_r_wind_tree_flutter);
            if (auto* E = g_pGamePersistent->Environment().CurrentEnv)
                pc.wind_params.set(E->wind_direction, E->wind_velocity, 0.0f, 0.0f);
        }
        pc.wind_params.w = ps_r_wind_tree_crown;   // flutter height gate — must match the forward pass
    }
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_VsmPageLayout, 0, 1, &m_XformDescSet, 0, nullptr);   // set0 transforms
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_VsmPageLayout, 2, 1, &pageSet, 0, nullptr);          // set2 page data
    vkCmdPushConstants(cmd, m_VsmPageLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

    VkPipeline lastPipe = VK_NULL_HANDLE; VkDescriptorSet lastTex = VK_NULL_HANDLE;
    for (u32 g = 0; g < (u32)m_Groups.size(); ++g) {
        const TreeIndirectGroup& grp = m_Groups[g];
        VkPipeline pipe = (grp.tcOffset == 24) ? p24 : p28;
        if (pipe == VK_NULL_HANDLE || grp.descSetIdx >= m_TexDescSets.size()) continue;
        if (pipe != lastPipe) { vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe); lastPipe = pipe; lastTex = VK_NULL_HANDLE; }
        VkDescriptorSet tex = m_TexDescSets[grp.descSetIdx];
        if (tex != lastTex) { vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_VsmPageLayout, 1, 1, &tex, 0, nullptr); lastTex = tex; }
        VkDeviceSize vbOff = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &grp.vb, &vbOff);
        vkCmdBindIndexBuffer(cmd, m_MeshletIndexBuffer->GetHandle(), 0, VK_INDEX_TYPE_UINT16);   // shared meshlet IB
        const VkDeviceSize cmdOff = (VkDeviceSize)m_MeshletGroupBase[g] * sizeof(VkDrawIndexedIndirectCommand);
        const VkDeviceSize cntOff = (VkDeviceSize)g * sizeof(u32);
        vkCmdDrawIndexedIndirectCount(cmd, cmdBuf, cmdOff, countBuf, cntOff,
                                      m_MeshletGroupCap[g], sizeof(VkDrawIndexedIndirectCommand));
    }
}

// CPU near/far set for the wind hybrid. THE METRIC IS LIGHT-SPACE, NOT WORLD DISTANCE:
// with a low sun a tree's shadow lands tens of metres DOWN-SUN of its trunk, so "trunk
// near the camera" misses exactly the shadows the player is looking at (a house shaded
// by a tree 60 m up-sun stayed rigid). A tree goes DYNAMIC when its shadow COLUMN passes
// within r_vsm_tree_wind_dist of the camera — i.e. the light-space-XY distance between
// the tree and the camera (both projected along the sun) is under the threshold. A world
// -distance bound (3×dist) caps the up-sun corridor cost. Hysteresis (×1.15 on both)
// stops boundary flapping; every crossing queues an invalidation sphere so the residency
// pass re-renders the tree's STATIC pages the same frame (no ghost doubling). Call
// BEFORE the residency dispatch (vk_vsm) with this frame's sun view.
void CTreeManager::VsmUpdateNearSet(const Fmatrix& sunView)
{
    if (m_TotalCount == 0 || m_MetaCPU.size() < m_TotalCount) return;
    if (m_VsmNearCPU.size() != m_TotalCount) m_VsmNearCPU.assign(m_TotalCount, 0);
    const float nd = ps_r_vsm_tree_wind ? ps_r_vsm_tree_wind_dist : 0.f;
    // World-distance corridor bound: 2× the lateral radius (80 m at default 40). Trees
    // further up-sun still shade the player, but their sway detail no longer reads and
    // every corridor tree re-rasters its crown into the dyn atlas EVERY frame — measured
    // 3× (near 330 trees, instDyn 5-12k, VSMrender to 2-3.5ms) → the main perf knob.
    const float wd = nd * 2.f;
    const float ndIn2  = nd * nd,                    ndOut2 = (nd * 1.15f) * (nd * 1.15f);
    const float wdIn2  = wd * wd,                    wdOut2 = (wd * 1.15f) * (wd * 1.15f);
    const Fvector cam = Device.vCameraPosition;
    Fvector camL; sunView.transform_tiny(camL, cam);  // camera in light space
    u32 nearCount = 0;
    // Small visuals (bushes/shrubs, R < 3 m) never go dynamic: their shadows are small
    // ground blobs where sway is unreadable, but they dominate the near set by COUNT
    // (the manager holds every trees\* visual incl. bushes — user saw "20 trees", the
    // set held 330). They keep casting rigidly from the static cache.
    constexpr float kWindMinRadius = 3.0f;
    for (u32 i = 0; i < m_TotalCount; ++i) {
        const GpuTreeMeta& m = m_MetaCPU[i];
        Fvector tL; sunView.transform_tiny(tL, m.sphere_P);
        const float lx = tL.x - camL.x, ly = tL.y - camL.y;
        const float dxy2 = lx * lx + ly * ly;                 // lateral dist to the shadow column
        const float dw2  = cam.distance_to_sqr(m.sphere_P);   // world dist (cost bound)
        const bool wasNear = m_VsmNearCPU[i] != 0;
        const bool isNear  = (nd > 0.f) && (m.sphere_R >= kWindMinRadius)
                           && (wasNear ? (dxy2 < ndOut2 && dw2 < wdOut2)
                                       : (dxy2 < ndIn2  && dw2 < wdIn2));
        if (isNear) ++nearCount;
        if (isNear == wasNear) continue;
        m_VsmNearCPU[i] = isNear ? 1 : 0;
        // sqrt(2) inflation: the bin covers the sphere's bounding SQUARE in light XY —
        // the invalidation circle must reach its corners.
        m_VsmPendingInval.push_back({ m.sphere_P.x, m.sphere_P.y, m.sphere_P.z, m.sphere_R * 1.4143f });
    }
    m_VsmNearCount = nearCount;
}

u32 CTreeManager::VsmPopTransitions(Fvector4* out, u32 maxOut)
{
    // FIFO: the resid pass consumes ≤4 circles/frame, so a burst of crossings (sprint
    // through a treeline) queues up — LIFO kept serving the NEWEST and starved the
    // oldest, leaving a crossed tree's stale rigid shadow doubled with its dyn one
    // until the queue drained. Oldest-first bounds the ghost to queue-length frames.
    const u32 n = _min(maxOut, (u32)m_VsmPendingInval.size());
    for (u32 i = 0; i < n; ++i) out[i] = m_VsmPendingInval[i];
    m_VsmPendingInval.erase(m_VsmPendingInval.begin(), m_VsmPendingInval.begin() + n);
    return n;
}

void CTreeManager::VsmBin(VkCommandBuffer cmd, VkBuffer pageTable, VkBuffer slotDirty, VkBuffer dynUsed, VkBuffer clipmapUBO, VkBuffer pageList, VkBuffer pageMax)
{
    if (!m_VsmReady) CreateVsmResources();
    if (!m_VsmReady || m_TotalCount == 0 || m_VsmBinPipe == VK_NULL_HANDLE) return;
    if (pageTable == VK_NULL_HANDLE || slotDirty == VK_NULL_HANDLE || dynUsed == VK_NULL_HANDLE || clipmapUBO == VK_NULL_HANDLE) return;
    if (pageMax == VK_NULL_HANDLE) pageMax = m_VsmCasterPages->GetHandle();   // HZB off: bind any valid buffer for layout parity

    m_VsmSlot = (m_VsmSlot + 1) % VK_FRAMES_IN_FLIGHT;
    const u32 slot = m_VsmSlot;

    // Upload this frame's CPU near-set flags (computed by VsmUpdateNearSet) into the ring slot.
    if (u32* flags = (u32*)m_VsmNearPtr[slot]) {
        if (m_VsmNearCPU.size() == m_TotalCount)
            for (u32 i = 0; i < m_TotalCount; ++i) flags[i] = m_VsmNearCPU[i];
        else
            memset(flags, 0, m_TotalCount * sizeof(u32));
    }

    vkCmdFillBuffer(cmd, m_VsmStats->GetHandle(), 0, VK_WHOLE_SIZE, 0u);
    { VkMemoryBarrier b{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
      b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr); }

    // mode 0 = STATIC pass: FAR trees into dirty static pages. dynUsed is bound for layout
    // completeness only (the shader's write is mode-gated).
    VkDescriptorBufferInfo bi[11] = {
        { m_TreeMetadataBuffer->GetHandle(), 0, VK_WHOLE_SIZE }, { clipmapUBO,                     0, VK_WHOLE_SIZE },
        { pageTable,                         0, VK_WHOLE_SIZE }, { m_VsmCasterPages->GetHandle(),  0, VK_WHOLE_SIZE },
        { m_VsmIndirect->GetHandle(),        0, VK_WHOLE_SIZE }, { m_VsmStats->GetHandle(),        0, VK_WHOLE_SIZE },
        { slotDirty,                         0, VK_WHOLE_SIZE },   // 6: STATIC-atlas dirty set (cache filter)
        { dynUsed,                           0, VK_WHOLE_SIZE },   // 7: unused in mode 0
        { m_VsmNearFlags[slot]->GetHandle(), 0, VK_WHOLE_SIZE },   // 8: near set
        { pageMax,                           0, VK_WHOLE_SIZE },   // 9: shadow-HZB occluder max (r_vsm_hzb)
        { pageTable,                         0, VK_WHOLE_SIZE },   // 10: staticPageTable — in the static call pageTable IS static (mode 0 ignores it)
    };
    VkDescriptorType t[11]; for (u32 i = 0; i < 11; ++i) t[i] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; t[1] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    VkWriteDescriptorSet w[11]{};
    for (u32 i = 0; i < 11; ++i) { w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = m_VsmBinSet[slot]; w[i].dstBinding = i; w[i].descriptorCount = 1; w[i].descriptorType = t[i]; w[i].pBufferInfo = &bi[i]; }
    vkUpdateDescriptorSets(VulkanHW.m_Device, 11, w, 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_VsmBinPipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_VsmBinLayout, 0, 1, &m_VsmBinSet[slot], 0, nullptr);
    struct { u32 count, cap, mode, hzbOn; float margin; u32 pad; } push{
        m_TotalCount, kVsmTreeCap, 0u, (u32)(ps_r_vsm_hzb ? 1 : 0), ps_r_vsm_hzb_margin, 0u };   // mode 0 = static + shadow-HZB
    vkCmdPushConstants(cmd, m_VsmBinLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd, (m_TotalCount + 63) / 64, 1, 1);

    // Phase B: refine this pass's per-tree page list into per-(meshlet,page) draws.
    if (IsMeshletMode()) DispatchMeshletBin(cmd, 0u, pageList, clipmapUBO);
}

// mode 1 = DYNAMIC pass: NEAR trees into all resident dyn pages (re-rendered with wind
// every frame) + dynUsed flags for the resolve gate. Call right after VsmBin (shares the
// frame slot + the stats clear/barrier; the two dispatches write disjoint buffers/slices).
void CTreeManager::VsmBinDyn(VkCommandBuffer cmd, VkBuffer dynPageTable, VkBuffer dynUsed, VkBuffer clipmapUBO, VkBuffer dynPageList, VkBuffer pageMax, VkBuffer staticPageTable)
{
    if (!m_VsmReady || m_TotalCount == 0 || m_VsmBinPipe == VK_NULL_HANDLE) return;
    if (dynPageTable == VK_NULL_HANDLE || dynUsed == VK_NULL_HANDLE || clipmapUBO == VK_NULL_HANDLE) return;
    const bool hzb = ps_r_vsm_hzb && pageMax != VK_NULL_HANDLE && staticPageTable != VK_NULL_HANDLE;
    if (pageMax == VK_NULL_HANDLE)         pageMax = m_VsmCasterPages->GetHandle();          // layout parity when HZB off
    if (staticPageTable == VK_NULL_HANDLE) staticPageTable = dynPageTable;                    // layout parity when HZB off
    const u32 slot = m_VsmSlot;

    VkDescriptorBufferInfo bi[11] = {
        { m_TreeMetadataBuffer->GetHandle(), 0, VK_WHOLE_SIZE }, { clipmapUBO,                     0, VK_WHOLE_SIZE },
        { dynPageTable,                      0, VK_WHOLE_SIZE }, { m_VsmCasterPages->GetHandle(),  0, VK_WHOLE_SIZE },
        { m_VsmDynIndirect->GetHandle(),     0, VK_WHOLE_SIZE }, { m_VsmStats->GetHandle(),        0, VK_WHOLE_SIZE },
        { dynUsed,                           0, VK_WHOLE_SIZE },   // 6: slotDirty slot — unused in mode 1, any valid buffer
        { dynUsed,                           0, VK_WHOLE_SIZE },   // 7: dynUsed flags (resolve gate)
        { m_VsmNearFlags[slot]->GetHandle(), 0, VK_WHOLE_SIZE },   // 8: near set
        { pageMax,                           0, VK_WHOLE_SIZE },   // 9: STATIC occluder max (Option A)
        { staticPageTable,                   0, VK_WHOLE_SIZE },   // 10: virtual -> STATIC slot (Option A)
    };
    VkDescriptorType t[11]; for (u32 i = 0; i < 11; ++i) t[i] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; t[1] = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    VkWriteDescriptorSet w[11]{};
    for (u32 i = 0; i < 11; ++i) { w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = m_VsmDynBinSet[slot]; w[i].dstBinding = i; w[i].descriptorCount = 1; w[i].descriptorType = t[i]; w[i].pBufferInfo = &bi[i]; }
    vkUpdateDescriptorSets(VulkanHW.m_Device, 11, w, 0, nullptr);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_VsmBinPipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_VsmBinLayout, 0, 1, &m_VsmDynBinSet[slot], 0, nullptr);
    // Option A: near trees (dyn) cull ONLY pages hidden behind STATIC occluders → visible near shadow unchanged.
    struct { u32 count, cap, mode, hzbOn; float margin; u32 pad; } push{
        m_TotalCount, kVsmTreeCap, 1u, (u32)(hzb ? 1 : 0), ps_r_vsm_hzb_margin, 0u };
    vkCmdPushConstants(cmd, m_VsmBinLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd, (m_TotalCount + 63) / 64, 1, 1);

    // Phase B: refine the DYNAMIC per-tree page list into per-(meshlet,page) draws (+wind slop).
    if (IsMeshletMode()) DispatchMeshletBin(cmd, 1u, dynPageList, clipmapUBO);

    // ---- Hybrid diagnostics (r_vsm_debug): stats readback + 2s log. stats[3] = instances
    // the DYN pass binned (0 while near trees exist = the GPU never saw the near flags).
    if (ps_r_vsm_debug && m_VsmStatsRB) {
        VkMemoryBarrier b{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
        VkBufferCopy rc{ 0, 0, 8 * sizeof(u32) };
        vkCmdCopyBuffer(cmd, m_VsmStats->GetHandle(), m_VsmStatsRB->GetHandle(), 1, &rc);
        if (m_VsmStatsPtr && Device.dwTimeGlobal > m_VsmLastLog + 2000) {
            m_VsmLastLog = Device.dwTimeGlobal;
            if (ps_r_vsm_hzb)
                Msg("[VK Trees] VSM shadow-HZB: culled (tree,page) pairs = %u", m_VsmStatsPtr[4]);
            Msg("[VK Trees] VSM hybrid: near=%u/%u pendingInval=%zu | GPU draws=%u instTotal=%u instDyn=%u maxPages=%u/%u%s",
                m_VsmNearCount, m_TotalCount, m_VsmPendingInval.size(),
                m_VsmStatsPtr[0], m_VsmStatsPtr[1], m_VsmStatsPtr[3],
                m_VsmStatsPtr[2], kVsmTreeCap, (m_VsmStatsPtr[2] >= kVsmTreeCap) ? " [TRUNCATED!]" : "");
        }
        // Meshlet path (r_vsm_meshlet): commands emitted + group overflow (truncation).
        if (IsMeshletMode() && m_MeshletStatsRB && m_MeshletStatsPtr) {
            VkBufferCopy mc{ 0, 0, 4 * sizeof(u32) };
            vkCmdCopyBuffer(cmd, m_MeshletStats->GetHandle(), m_MeshletStatsRB->GetHandle(), 1, &mc);
            if (Device.dwTimeGlobal == m_VsmLastLog)   // logged this tick above
                Msg("[VK Trees] VSM meshlet: cmds=%u overflow=%u (clusters=%u, %u cmd/tree)%s",
                    m_MeshletStatsPtr[0], m_MeshletStatsPtr[1], m_MeshletTotal, kCmdPerTree,
                    m_MeshletStatsPtr[1] ? " [TRUNCATED!]" : "");
        }
    }
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

    // Phase B: meshlet-cull path draws only the clusters that survived stage 2 (STATIC = rigid).
    if (IsMeshletMode()) { DrawMeshlets(cmd, 0u, m_VsmPageSet[slot], /*wind*/false); return; }

    // STATIC pass push: NO wind, ever — the toroidal cache needs frame-invariant page
    // content (per-page wind was the "jelly": staggered refreshes froze each page at a
    // different wind phase). Near trees sway via VsmRenderDyn instead.
    struct VsmPagePush {
        float    uvScale, alphaRef; u32 cap, pad;
        Fvector4 wind_params, wsetup_trees, wind_anim;
    } pc{};
    pc.uvScale = 1.0f / 2048.0f; pc.alphaRef = 200.0f / 255.0f; pc.cap = kVsmTreeCap;
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

// DYNAMIC pass: near trees into the dyn atlas with LIVE wind (one time slice per frame →
// smooth coherent sway, the r_vsm_tree_wind feature done right). Call inside vk_vsm's
// RenderAtlas DYNAMIC pass, after the skinned/grass casters.
void CTreeManager::VsmRenderDyn(VkCommandBuffer cmd, VkBuffer dynPageList, VkBuffer clipmapUBO)
{
    if (!ps_r_vsm_tree_wind) return;   // hybrid off → all trees are in the static pass
    if (!m_VsmReady || m_TotalCount == 0 || m_XformDescSet == VK_NULL_HANDLE) return;
    if (m_VsmPageDynPipe24 == VK_NULL_HANDLE && m_VsmPageDynPipe28 == VK_NULL_HANDLE) return;
    if (dynPageList == VK_NULL_HANDLE || clipmapUBO == VK_NULL_HANDLE) return;
    const u32 slot = m_VsmSlot;

    VkDescriptorBufferInfo bi[3] = {
        { dynPageList,                   0, VK_WHOLE_SIZE },
        { m_VsmCasterPages->GetHandle(), 0, VK_WHOLE_SIZE },
        { clipmapUBO,                    0, VK_WHOLE_SIZE },
    };
    const VkDescriptorType t[3] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER };
    VkWriteDescriptorSet w[3]{};
    for (u32 i = 0; i < 3; ++i) { w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = m_VsmDynPageSet[slot]; w[i].dstBinding = i; w[i].descriptorCount = 1; w[i].descriptorType = t[i]; w[i].pBufferInfo = &bi[i]; }
    vkUpdateDescriptorSets(VulkanHW.m_Device, 3, w, 0, nullptr);

    // Phase B: meshlet-cull path draws only surviving clusters, with LIVE wind (near trees).
    if (IsMeshletMode()) { DrawMeshlets(cmd, 1u, m_VsmDynPageSet[slot], /*wind*/true); return; }

    // Live wind push — same params as the forward tree pass (ssfx_tree_wind).
    struct VsmPagePush {
        float    uvScale, alphaRef; u32 cap, pad;
        Fvector4 wind_params, wsetup_trees, wind_anim;
    } pc{};
    pc.uvScale = 1.0f / 2048.0f; pc.alphaRef = 200.0f / 255.0f; pc.cap = kVsmTreeCap;
    pc.wsetup_trees.set(ps_r_wind_tree_anim, ps_r_wind_tree_trunk, ps_r_wind_tree_bend, 0.1f);
    if (g_pGamePersistent) {
        const Fvector3 wa = g_pGamePersistent->Environment().wind_anim;
        pc.wind_anim.set(wa.x, wa.y, wa.z, ps_r_wind_tree_flutter);
        if (auto* E = g_pGamePersistent->Environment().CurrentEnv)
            pc.wind_params.set(E->wind_direction, E->wind_velocity, 0.0f, 0.0f);
    }
    pc.wind_params.w = ps_r_wind_tree_crown;   // flutter height gate — must match the forward pass
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_VsmPageLayout, 0, 1, &m_XformDescSet, 0, nullptr);          // set0 transforms
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_VsmPageLayout, 2, 1, &m_VsmDynPageSet[slot], 0, nullptr);   // set2 page data (dyn)
    vkCmdPushConstants(cmd, m_VsmPageLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

    VkPipeline lastPipe = VK_NULL_HANDLE; VkDescriptorSet lastTex = VK_NULL_HANDLE;
    for (const TreeIndirectGroup& grp : m_Groups) {
        VkPipeline pipe = (grp.tcOffset == 24) ? m_VsmPageDynPipe24 : m_VsmPageDynPipe28;
        if (pipe == VK_NULL_HANDLE || grp.descSetIdx >= m_TexDescSets.size()) continue;
        if (pipe != lastPipe) { vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe); lastPipe = pipe; lastTex = VK_NULL_HANDLE; }
        VkDescriptorSet tex = m_TexDescSets[grp.descSetIdx];
        if (tex != lastTex) { vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_VsmPageLayout, 1, 1, &tex, 0, nullptr); lastTex = tex; }
        VkDeviceSize vbOff = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &grp.vb, &vbOff);
        vkCmdBindIndexBuffer(cmd, grp.ib, 0, VK_INDEX_TYPE_UINT16);
        // Far trees carry instanceCount=0 in m_VsmDynIndirect → ~free on the GPU.
        vkCmdDrawIndexedIndirect(cmd, m_VsmDynIndirect->GetHandle(),
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
    if (m_VsmPageDynPipe24) { vkDestroyPipeline(dev, m_VsmPageDynPipe24, nullptr); m_VsmPageDynPipe24 = VK_NULL_HANDLE; }
    if (m_VsmPageDynPipe28) { vkDestroyPipeline(dev, m_VsmPageDynPipe28, nullptr); m_VsmPageDynPipe28 = VK_NULL_HANDLE; }
    if (m_VsmPageLayout) { vkDestroyPipelineLayout(dev, m_VsmPageLayout, nullptr); m_VsmPageLayout = VK_NULL_HANDLE; }
    if (m_VsmBinPipe)    { vkDestroyPipeline(dev, m_VsmBinPipe, nullptr); m_VsmBinPipe = VK_NULL_HANDLE; }
    if (m_VsmBinLayout)  { vkDestroyPipelineLayout(dev, m_VsmBinLayout, nullptr); m_VsmBinLayout = VK_NULL_HANDLE; }
    if (m_VsmDescPool)   { vkDestroyDescriptorPool(dev, m_VsmDescPool, nullptr); m_VsmDescPool = VK_NULL_HANDLE;
                           for (u32 i = 0; i < VK_FRAMES_IN_FLIGHT; ++i) { m_VsmBinSet[i] = VK_NULL_HANDLE; m_VsmDynBinSet[i] = VK_NULL_HANDLE; m_VsmPageSet[i] = VK_NULL_HANDLE; m_VsmDynPageSet[i] = VK_NULL_HANDLE; } }
    if (m_VsmBinSetL)    { vkDestroyDescriptorSetLayout(dev, m_VsmBinSetL, nullptr); m_VsmBinSetL = VK_NULL_HANDLE; }
    if (m_VsmPageSetL)   { vkDestroyDescriptorSetLayout(dev, m_VsmPageSetL, nullptr); m_VsmPageSetL = VK_NULL_HANDLE; }
    if (m_VsmCasterPages) { m_VsmCasterPages->Destroy(); xr_delete(m_VsmCasterPages); }
    if (m_VsmIndirect)    { m_VsmIndirect->Destroy(); xr_delete(m_VsmIndirect); }
    if (m_VsmDynIndirect) { m_VsmDynIndirect->Destroy(); xr_delete(m_VsmDynIndirect); }
    if (m_VsmStats)       { m_VsmStats->Destroy(); xr_delete(m_VsmStats); }
    if (m_VsmStatsRB)     { m_VsmStatsRB->Destroy(); xr_delete(m_VsmStatsRB); }
    m_VsmStatsPtr = nullptr;
    for (u32 i = 0; i < VK_FRAMES_IN_FLIGHT; ++i) {
        if (m_VsmNearFlags[i]) { m_VsmNearFlags[i]->Destroy(); xr_delete(m_VsmNearFlags[i]); }
        m_VsmNearPtr[i] = nullptr;
    }
    m_VsmNearCPU.clear(); m_VsmPendingInval.clear();

    // ----- Phase B meshlet-cull resources.
    if (m_MeshletBinPipe)   { vkDestroyPipeline(dev, m_MeshletBinPipe, nullptr); m_MeshletBinPipe = VK_NULL_HANDLE; }
    if (m_MeshletBinLayout) { vkDestroyPipelineLayout(dev, m_MeshletBinLayout, nullptr); m_MeshletBinLayout = VK_NULL_HANDLE; }
    if (m_MeshletPagePipe24)    { vkDestroyPipeline(dev, m_MeshletPagePipe24, nullptr); m_MeshletPagePipe24 = VK_NULL_HANDLE; }
    if (m_MeshletPagePipe28)    { vkDestroyPipeline(dev, m_MeshletPagePipe28, nullptr); m_MeshletPagePipe28 = VK_NULL_HANDLE; }
    if (m_MeshletPageDynPipe24) { vkDestroyPipeline(dev, m_MeshletPageDynPipe24, nullptr); m_MeshletPageDynPipe24 = VK_NULL_HANDLE; }
    if (m_MeshletPageDynPipe28) { vkDestroyPipeline(dev, m_MeshletPageDynPipe28, nullptr); m_MeshletPageDynPipe28 = VK_NULL_HANDLE; }
    if (m_MeshletDescPool)  { vkDestroyDescriptorPool(dev, m_MeshletDescPool, nullptr); m_MeshletDescPool = VK_NULL_HANDLE;
                              for (u32 i = 0; i < VK_FRAMES_IN_FLIGHT; ++i) { m_MeshletBinSet[i] = VK_NULL_HANDLE; m_MeshletDynBinSet[i] = VK_NULL_HANDLE; } }
    if (m_MeshletBinSetL)   { vkDestroyDescriptorSetLayout(dev, m_MeshletBinSetL, nullptr); m_MeshletBinSetL = VK_NULL_HANDLE; }
    auto delBuf = [](CVulkanBuffer*& b) { if (b) { b->Destroy(); xr_delete(b); } };
    delBuf(m_VsmMeshletCmd); delBuf(m_VsmMeshletCmdDyn);
    delBuf(m_VsmMeshletGroupCount); delBuf(m_VsmMeshletGroupCountDyn);
    delBuf(m_TreeGroupBuffer); delBuf(m_GroupInfoBuffer);
    delBuf(m_MeshletStats); delBuf(m_MeshletStatsRB);
    m_MeshletStatsPtr = nullptr;
    m_MeshletGroupBase.clear(); m_MeshletGroupCap.clear();
    m_MeshletVsmReady = false;

    m_VsmReady = false; m_VsmSlot = 0;
}

}  // namespace VK
