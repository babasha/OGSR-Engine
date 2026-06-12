// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan — tonemap / exposure composite pass. See vk_pass_tonemap.h.
#include "stdafx.h"
#include "vk_pass_tonemap.h"
#include "vk_scene_color.h"
#include "vk_pass_bloom.h"         // BloomPass — bright-pass + blur before the composite
#include "vk_swapchain.h"          // Swapchain.m_Images / m_ImageViews / m_Format
#include "vk_shaders.h"            // g_ShaderManager
#include "vk_pipeline_cache.h"     // PipelineCache::GetCacheObject
#include "vk_barriers.h"           // ImageBarrier
#include "HW_Vulkan.h"
#include "../xrRender/xrRender_console.h"  // ps_r2_img_* — R4 color-grading console knobs

namespace VK {

namespace {
    constexpr u32 kMaxImages = 8;

    VkPipeline            s_Pipeline       = VK_NULL_HANDLE;
    VkPipelineLayout      s_PipelineLayout = VK_NULL_HANDLE;
    VkShaderModule        s_VS             = VK_NULL_HANDLE;
    VkShaderModule        s_FS             = VK_NULL_HANDLE;
    VkDescriptorSetLayout s_SetLayout      = VK_NULL_HANDLE;
    VkDescriptorPool      s_Pool           = VK_NULL_HANDLE;
    VkDescriptorSet       s_Set[kMaxImages] = {};
    VkSampler             s_Sampler        = VK_NULL_HANDLE;
    u32                   s_boundGen       = 0;   // SceneColor generation the sets were written for
    u32                   s_boundBloomGen  = 0;   // BloomPass RT generation (binding 1)

    // R4 auto-exposure (bloom_luminance_3.ps): exposure = middlegray/(avgLum +
    // low), measured from the whole-frame average luminance (the HDR target's
    // top mip), clamped. R4 defaults are middlegray 1.0 / low 0.0001 with a
    // def_hdr-scaled luminance; our HDR luminance is unscaled, so middlegray is
    // retuned to land midday near exposure ~1 and the clamp bounds how far a
    // dim/bright scene can drift (keeps night from washing to mid-gray).
    // White point 11.2 = R4 tonemap_srgb.h fWhiteIntensity: a soft filmic
    // highlight rolloff (the glow comes from bloom, not from clipping).
    // Screenshot comparison vs R4 (morning, bright sky + tree shade): the
    // sky-heavy frame average pushed our exposure into the 0.6 floor and the
    // whole foreground went darker/flatter than R4. Raise the target + floor.
    // NOTE: kMiddleGray/kLowLum/kExpMin/kExpMax are DUPLICATED in
    // vk_pass_bloom.cpp (bloom pre-exposes with the same formula) — keep in sync.
    constexpr float kWhitePoint  = 11.2f;    // R4 tonemap_sRGB fWhiteIntensity
    constexpr float kMiddleGray  = 0.58f;    // exposure target (our HDR scale)
    constexpr float kLowLum      = 0.0001f;  // R4 ps_r2_tonemap_low_lum
    constexpr float kExpMin      = 0.80f;    // exposure clamp
    constexpr float kExpMax      = 2.20f;
    constexpr float kExpComp     = 1.0f;     // overall compensation knob
    constexpr float kBloomIntensity = 0.8f;  // bloom add strength (blend_soft analog)

    struct TonemapPush { float p0[4]; float p1[4]; float p2[4]; float p3[4]; };
}

namespace TonemapPass {

bool Init()
{
    if (s_Pipeline) return true;

    if (!g_ShaderManager) g_ShaderManager = xr_new<VK::CVulkanSPIRVLoader>();
    s_VS = g_ShaderManager->Load("tonemap.vert.spv");
    s_FS = g_ShaderManager->Load("tonemap.frag.spv");
    if (s_VS == VK_NULL_HANDLE || s_FS == VK_NULL_HANDLE) {
        Msg("![VK Tonemap] shader load failed");
        return false;
    }

    // Set 0: binding 0 = HDR scene target, binding 1 = blurred bloom (both FS).
    VkDescriptorSetLayoutBinding b[2]{};
    for (u32 i = 0; i < 2; ++i) {
        b[i].binding = i; b[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lci.bindingCount = 2; lci.pBindings = b;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &s_SetLayout) != VK_SUCCESS) {
        Msg("![VK Tonemap] set layout failed"); return false;
    }

    VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxImages * 2 };
    VkDescriptorPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = kMaxImages; pci.poolSizeCount = 1; pci.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &s_Pool) != VK_SUCCESS) {
        Msg("![VK Tonemap] pool failed"); return false;
    }
    VkDescriptorSetLayout layouts[kMaxImages];
    for (u32 i = 0; i < kMaxImages; ++i) layouts[i] = s_SetLayout;
    VkDescriptorSetAllocateInfo dai{};
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = s_Pool; dai.descriptorSetCount = kMaxImages; dai.pSetLayouts = layouts;
    if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, s_Set) != VK_SUCCESS) {
        Msg("![VK Tonemap] alloc sets failed"); return false;
    }

    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_LINEAR; si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.minLod = 0.0f; si.maxLod = VK_LOD_CLAMP_NONE;   // textureLod(maxMip) for the avg-luminance read
    if (vkCreateSampler(VulkanHW.m_Device, &si, nullptr, &s_Sampler) != VK_SUCCESS) {
        Msg("![VK Tonemap] sampler failed"); return false;
    }

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT; pcr.offset = 0; pcr.size = sizeof(TonemapPush);
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1; plci.pSetLayouts = &s_SetLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_PipelineLayout) != VK_SUCCESS) {
        Msg("![VK Tonemap] pipeline layout failed"); return false;
    }

    VkPipelineVertexInputStateCreateInfo vi{}; vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;   stages[0].module = s_VS; stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = s_FS; stages[1].pName = "main";

    VkPipelineInputAssemblyStateCreateInfo ia{}; ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{}; vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1; vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{}; rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{}; ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo ds{}; ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable = VK_FALSE; ds.depthWriteEnable = VK_FALSE;
    VkPipelineColorBlendAttachmentState ba{};
    ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    ba.blendEnable = VK_FALSE;
    VkPipelineColorBlendStateCreateInfo cb{}; cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1; cb.pAttachments = &ba;
    VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynState{}; dynState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynState.dynamicStateCount = 2; dynState.pDynamicStates = dyn;

    VkFormat colorFormat = Swapchain.m_Format;     // writes the actual swapchain (UNORM)
    VkPipelineRenderingCreateInfo prci{};
    prci.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    prci.colorAttachmentCount = 1; prci.pColorAttachmentFormats = &colorFormat;
    prci.depthAttachmentFormat = VK_FORMAT_UNDEFINED;

    VkGraphicsPipelineCreateInfo pi{};
    pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pi.pNext = &prci; pi.stageCount = 2; pi.pStages = stages;
    pi.pVertexInputState = &vi; pi.pInputAssemblyState = &ia; pi.pViewportState = &vp;
    pi.pRasterizationState = &rs; pi.pMultisampleState = &ms; pi.pDepthStencilState = &ds;
    pi.pColorBlendState = &cb; pi.pDynamicState = &dynState; pi.layout = s_PipelineLayout;
    if (vkCreateGraphicsPipelines(VulkanHW.m_Device, PipelineCache::GetCacheObject(), 1, &pi, nullptr, &s_Pipeline) != VK_SUCCESS) {
        Msg("![VK Tonemap] pipeline create failed"); return false;
    }

    Msg("[VK Tonemap] Init OK (auto-exposure mg=%.2f white=%.2f clamp[%.2f,%.2f])",
        kMiddleGray, kWhitePoint, kExpMin, kExpMax);
    return true;
}

void Destroy()
{
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
    if (s_Pipeline)       { vkDestroyPipeline(VulkanHW.m_Device, s_Pipeline, nullptr); s_Pipeline = VK_NULL_HANDLE; }
    if (s_PipelineLayout) { vkDestroyPipelineLayout(VulkanHW.m_Device, s_PipelineLayout, nullptr); s_PipelineLayout = VK_NULL_HANDLE; }
    if (s_Sampler)        { vkDestroySampler(VulkanHW.m_Device, s_Sampler, nullptr); s_Sampler = VK_NULL_HANDLE; }
    if (s_Pool)           { vkDestroyDescriptorPool(VulkanHW.m_Device, s_Pool, nullptr); s_Pool = VK_NULL_HANDLE; }
    if (s_SetLayout)      { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_SetLayout, nullptr); s_SetLayout = VK_NULL_HANDLE; }
    s_VS = s_FS = VK_NULL_HANDLE; s_boundGen = 0; s_boundBloomGen = 0;
}

}  // namespace TonemapPass

void Pass_TonemapComposite(FrameContext& ctx)
{
    if (s_Pipeline == VK_NULL_HANDLE || ctx.cmd == VK_NULL_HANDLE) return;
    const u32 idx = ctx.imageIndex;
    if (SceneColor::GetImage(idx) == VK_NULL_HANDLE) return;

    VkCommandBuffer cmd = ctx.cmd;

    // Downsample the HDR scene to its full mip chain (top mip = whole-frame
    // average luminance for auto-exposure). This also leaves ALL mips — incl.
    // mip0 (the scene) — in SHADER_READ for the composite below.
    SceneColor::GenerateMips(cmd, idx);

    // Bloom chain (bright pass from the scene mips + gaussian blur) — result
    // lands SHADER_READ for binding 1.
    BloomPass::Execute(cmd, idx, ctx.extent, SceneColor::Generation());

    // Rebind the per-image descriptor sets when the HDR target or the bloom RT
    // was (re)created.
    const u32 gen      = SceneColor::Generation();
    const u32 bloomGen = BloomPass::Generation();
    if (gen != s_boundGen || bloomGen != s_boundBloomGen) {
        const u32 n = SceneColor::Count();
        for (u32 i = 0; i < n && i < kMaxImages; ++i) {
            VkDescriptorImageInfo ii[2]{};
            ii[0].sampler = s_Sampler; ii[0].imageView = SceneColor::GetSampleView(i);  // full mip chain
            ii[1].sampler = s_Sampler; ii[1].imageView = BloomPass::GetResultView();
            ii[0].imageLayout = ii[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            if (ii[1].imageView == VK_NULL_HANDLE) ii[1].imageView = SceneColor::GetSampleView(i); // pre-bloom fallback
            VkWriteDescriptorSet w[2]{};
            for (u32 k = 0; k < 2; ++k) {
                w[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[k].dstSet = s_Set[i]; w[k].dstBinding = k; w[k].descriptorCount = 1;
                w[k].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[k].pImageInfo = &ii[k];
            }
            vkUpdateDescriptorSets(VulkanHW.m_Device, 2, w, 0, nullptr);
        }
        s_boundGen = gen;
        s_boundBloomGen = bloomGen;
    }

    // Swapchain image: UNDEFINED (untouched this frame) → COLOR_ATTACHMENT for the composite.
    ImageBarrier(cmd, Swapchain.m_Images[idx],
                 VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    VkRenderingAttachmentInfo cAtt{};
    cAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    cAtt.imageView   = Swapchain.m_ImageViews[idx];
    cAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    cAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_DONT_CARE;   // fullscreen overwrite
    cAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo ri{};
    ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea.extent    = ctx.extent;
    ri.layerCount           = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments    = &cAtt;
    vkCmdBeginRendering(cmd, &ri);

    VkViewport vp{ 0.f, 0.f, (float)ctx.extent.width, (float)ctx.extent.height, 0.f, 1.f };
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{ {0,0}, ctx.extent };
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_Pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_PipelineLayout, 0, 1, &s_Set[idx], 0, nullptr);

    TonemapPush push{};
    push.p0[0] = kWhitePoint;
    push.p0[1] = float(SceneColor::MipLevels() - 1);   // top-mip LOD = whole-frame average
    push.p0[2] = kMiddleGray;
    push.p0[3] = kLowLum;
    push.p1[0] = kExpMin; push.p1[1] = kExpMax; push.p1[2] = kExpComp;
    push.p1[3] = kBloomIntensity;
    // R4 CDL grading (ACES_settings.h): Slope = r2_img_exposure, Power =
    // 2*(1 - r2_img_cg), Saturation = r2_img_saturation, gamma from
    // img_corrections — all live console knobs (ssfx_exposure/_gamma/
    // _saturation/ssfx_color_grading), neutral by default.
    push.p2[0] = ps_r2_img_exposure;
    push.p2[1] = ps_r2_img_saturation;
    push.p2[2] = 1.0f / (ps_r2_img_gamma > 0.05f ? ps_r2_img_gamma : 1.0f);
    push.p3[0] = 2.0f * (1.0f - ps_r2_img_cg.x);
    push.p3[1] = 2.0f * (1.0f - ps_r2_img_cg.y);
    push.p3[2] = 2.0f * (1.0f - ps_r2_img_cg.z);
    vkCmdPushConstants(cmd, s_PipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);

    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRendering(cmd);
    // Swapchain stays COLOR_ATTACHMENT for the UI pass; End brings it to PRESENT.
}

}  // namespace VK
