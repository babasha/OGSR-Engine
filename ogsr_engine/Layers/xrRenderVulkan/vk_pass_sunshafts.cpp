// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan — volumetric sun shafts (god rays). See vk_pass_sunshafts.h.
#include "stdafx.h"
#include "vk_pass_sunshafts.h"
#include "vk_swapchain.h"               // Swapchain.m_DepthImage/m_DepthView/m_Format
#include "vk_scene_color.h"             // HDR scene target format
#include "vk_shaders.h"                 // g_ShaderManager (SPIRV loader)
#include "vk_pipeline_cache.h"          // PipelineCache::GetCacheObject
#include "vk_env_light.h"               // EnvLight set (sun_vp + sun shadow map)
#include "vk_shadow.h"                  // ShadowMap::GetSampler (reused for depth)
#include "vk_barriers.h"                // ImageBarrier
#include "vk_pass_ssao.h"               // DeriveProjTerms (Device.mProject is identity on this path)
#include "vk_command_buffer.h"          // CommandManager.GetCurrentFrame()
#include "CRender_Vulkan.h"             // RImplementation.b_loaded
#include "HW_Vulkan.h"
#include "../../xr_3da/device.h"        // Device camera basis + mProject
#include "../../xr_3da/IGame_Persistent.h"
#include "../../xr_3da/Environment.h"   // CEnvDescriptorMixer (sun gate)

namespace VK {

namespace {
    constexpr u32 kFramesInFlight = CVulkanCommandManager::FRAMES_IN_FLIGHT;

    bool                  s_inited    = false;
    bool                  s_failed    = false;
    VkDescriptorSetLayout s_setLayout = VK_NULL_HANDLE;   // set 0: binding 0 = scene depth
    VkDescriptorPool      s_pool      = VK_NULL_HANDLE;
    VkDescriptorSet       s_set[kFramesInFlight] = {};
    VkPipelineLayout      s_layout    = VK_NULL_HANDLE;
    VkPipeline            s_pipeline  = VK_NULL_HANDLE;

    // Tuning: overall shaft strength and how far the ray marches.
    constexpr float kDensity   = 0.35f;
    constexpr float kMaxMarch  = 60.f;

    struct ShaftsPush {
        float camPos[4];
        float camDir[4];
        float camRightT[4];   // right * tan(fovX/2)
        float camTopT[4];     // up    * tan(fovY/2)
        float zp[4];          // proj _33, proj _43, density, maxMarch
    };
    static_assert(sizeof(ShaftsPush) == 80, "must match sunshafts.frag PC block");

    bool Init()
    {
        if (s_inited) return !s_failed;
        s_inited = true;

        if (!g_ShaderManager) { s_failed = true; return false; }
        VkShaderModule vs = g_ShaderManager->Load("sunshafts.vert.spv");
        VkShaderModule fs = g_ShaderManager->Load("sunshafts.frag.spv");
        if (vs == VK_NULL_HANDLE || fs == VK_NULL_HANDLE) {
            Msg("![VK Shafts] sunshafts.{vert,frag}.spv missing — sun shafts disabled");
            s_failed = true; return false;
        }
        if (EnvLight::GetSetLayout() == VK_NULL_HANDLE) { s_failed = true; return false; }

        // Set 0: scene depth (combined sampler, FRAGMENT).
        VkDescriptorSetLayoutBinding b{};
        b.binding = 0; b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b.descriptorCount = 1; b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo slci{};
        slci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        slci.bindingCount = 1; slci.pBindings = &b;
        if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &slci, nullptr, &s_setLayout) != VK_SUCCESS) {
            s_failed = true; return false;
        }

        VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kFramesInFlight };
        VkDescriptorPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets = kFramesInFlight; pci.poolSizeCount = 1; pci.pPoolSizes = &ps;
        if (vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &s_pool) != VK_SUCCESS) {
            s_failed = true; return false;
        }
        VkDescriptorSetLayout layouts[kFramesInFlight];
        for (u32 i = 0; i < kFramesInFlight; ++i) layouts[i] = s_setLayout;
        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = s_pool; dai.descriptorSetCount = kFramesInFlight; dai.pSetLayouts = layouts;
        if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, s_set) != VK_SUCCESS) {
            s_failed = true; return false;
        }

        VkDescriptorSetLayout sets[2] = { s_setLayout, EnvLight::GetSetLayout() };
        VkPushConstantRange pcr{ VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ShaftsPush) };
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 2; plci.pSetLayouts = sets;
        plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
        if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_layout) != VK_SUCCESS) {
            s_failed = true; return false;
        }

        // Fullscreen-triangle pipeline: no vertex input, no depth attachment
        // (depth is SAMPLED here), additive blend over the scene colour.
        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;   stages[0].module = vs; stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = fs; stages[1].pName = "main";

        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vp{};
        vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vp.viewportCount = 1; vp.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode    = VK_CULL_MODE_NONE;
        rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth   = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

        VkPipelineColorBlendAttachmentState ba{};
        ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        ba.blendEnable         = VK_TRUE;
        ba.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        ba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        ba.colorBlendOp        = VK_BLEND_OP_ADD;
        ba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        ba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        ba.alphaBlendOp        = VK_BLEND_OP_ADD;
        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1; cb.pAttachments = &ba;

        VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynState{};
        dynState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynState.dynamicStateCount = 2; dynState.pDynamicStates = dyn;

        VkFormat colorFormat = VK::SceneColor::Format();
        VkPipelineRenderingCreateInfo prci{};
        prci.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        prci.colorAttachmentCount    = 1;
        prci.pColorAttachmentFormats = &colorFormat;

        VkGraphicsPipelineCreateInfo pi{};
        pi.sType             = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pi.pNext             = &prci;
        pi.stageCount        = 2;     pi.pStages             = stages;
        pi.pVertexInputState = &vi;   pi.pInputAssemblyState = &ia;
        pi.pViewportState    = &vp;   pi.pRasterizationState = &rs;
        pi.pMultisampleState = &ms;   pi.pDepthStencilState  = &ds;
        pi.pColorBlendState  = &cb;   pi.pDynamicState       = &dynState;
        pi.layout            = s_layout;
        if (vkCreateGraphicsPipelines(VulkanHW.m_Device, PipelineCache::GetCacheObject(), 1, &pi, nullptr, &s_pipeline) != VK_SUCCESS) {
            Msg("![VK Shafts] pipeline create failed"); s_failed = true; return false;
        }

        Msg("[VK Shafts] init OK");
        return true;
    }
}  // anon namespace

void Pass_SunShafts(FrameContext& ctx)
{
    if (ctx.cmd == VK_NULL_HANDLE)  return;
    if (!RImplementation.b_loaded)  return;
    if (!g_pGamePersistent)         return;
    auto* E = g_pGamePersistent->Environment().CurrentEnv;
    if (!E) return;

    // Only when the sun is meaningfully up — at night there's nothing to shaft.
    const float sunLum = 0.299f * E->sun_color.x + 0.587f * E->sun_color.y + 0.114f * E->sun_color.z;
    if (E->sun_dir.y > -0.02f || sunLum < 0.03f) return;

    if (!Init()) return;
    VkDescriptorSet envSet = EnvLight::GetCurrentSet();
    if (envSet == VK_NULL_HANDLE) return;

    VkCommandBuffer cmd = ctx.cmd;

    // Scene depth: attachment → sampled for this draw (restored below).
    ImageBarrier(cmd, Swapchain.m_DepthImage, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);

    // Refresh this slot's depth descriptor (slot is fence-guarded; the view can
    // change on swapchain recreate).
    const u32 slot = CommandManager.GetCurrentFrame() % kFramesInFlight;
    {
        VkDescriptorImageInfo ii{};
        ii.sampler     = ShadowMap::GetSampler();
        ii.imageView   = Swapchain.m_DepthView;
        ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = s_set[slot]; w.dstBinding = 0; w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w.pImageInfo = &ii;
        vkUpdateDescriptorSets(VulkanHW.m_Device, 1, &w, 0, nullptr);
    }

    VkRenderingAttachmentInfo cAtt{};
    cAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    cAtt.imageView   = ctx.colorView;
    cAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    cAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;
    cAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo ri{};
    ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea.extent    = ctx.extent;
    ri.layerCount           = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments    = &cAtt;
    vkCmdBeginRendering(cmd, &ri);

    // Positive viewport — fullscreen triangle handles orientation itself.
    VkViewport vp{ 0.f, 0.f, (float)ctx.extent.width, (float)ctx.extent.height, 0.f, 1.f };
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{ {}, ctx.extent };
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_layout, 0, 1, &s_set[slot], 0, nullptr);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_layout, 1, 1, &envSet, 0, nullptr);

    // Frustum-ray reconstruction params — derived from the matrix that rendered
    // the depth (Device.mProject is NOT maintained on the Vulkan path: it reads
    // as identity, which made zview collapse to 0 → the march length was 0 and
    // the shafts degenerated to a uniform sun-facing glow).
    const ProjTerms pt = DeriveProjTerms(Device.mFullTransform);
    ShaftsPush push{};
    push.camPos[0] = Device.vCameraPosition.x;  push.camPos[1] = Device.vCameraPosition.y;  push.camPos[2] = Device.vCameraPosition.z;
    push.camDir[0] = pt.dir.x;   push.camDir[1] = pt.dir.y;   push.camDir[2] = pt.dir.z;
    push.camRightT[0] = pt.right.x * pt.tanX; push.camRightT[1] = pt.right.y * pt.tanX; push.camRightT[2] = pt.right.z * pt.tanX;
    push.camTopT[0]   = pt.top.x  * pt.tanY;  push.camTopT[1]   = pt.top.y  * pt.tanY;  push.camTopT[2]   = pt.top.z  * pt.tanY;
    push.zp[0] = pt.p33;
    push.zp[1] = pt.p43;
    push.zp[2] = kDensity;
    push.zp[3] = kMaxMarch;
    vkCmdPushConstants(cmd, s_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);

    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRendering(cmd);

    // Restore the frame-wide depth layout invariant for the passes after us.
    ImageBarrier(cmd, Swapchain.m_DepthImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
}

void SunShafts_Destroy()
{
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
    if (s_pipeline)  { vkDestroyPipeline(VulkanHW.m_Device, s_pipeline, nullptr); s_pipeline = VK_NULL_HANDLE; }
    if (s_layout)    { vkDestroyPipelineLayout(VulkanHW.m_Device, s_layout, nullptr); s_layout = VK_NULL_HANDLE; }
    if (s_pool)      { vkDestroyDescriptorPool(VulkanHW.m_Device, s_pool, nullptr); s_pool = VK_NULL_HANDLE; }
    if (s_setLayout) { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_setLayout, nullptr); s_setLayout = VK_NULL_HANDLE; }
    for (u32 i = 0; i < kFramesInFlight; ++i) s_set[i] = VK_NULL_HANDLE;
    s_inited = false; s_failed = false;
}

}  // namespace VK
