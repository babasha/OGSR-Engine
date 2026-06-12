// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan — bloom pass. See vk_pass_bloom.h.
#include "stdafx.h"
#include "vk_pass_bloom.h"
#include "vk_scene_color.h"
#include "vk_shaders.h"            // g_ShaderManager
#include "vk_pipeline_cache.h"     // PipelineCache::GetCacheObject
#include "vk_barriers.h"           // ImageBarrier
#include "HW_Vulkan.h"

namespace VK { namespace BloomPass {

namespace {
    constexpr u32 kMaxImages = 8;   // mirror SceneColor

    // R4 knobs: threshold is the post-exposure luminance where glow starts
    // (b_params.x analog), knee softens the cutoff.
    constexpr float kThreshold = 0.85f;
    constexpr float kKnee      = 0.35f;
    // Auto-exposure params — MUST match vk_pass_tonemap.cpp (same formula).
    constexpr float kMiddleGray = 0.58f;
    constexpr float kLowLum     = 0.0001f;
    constexpr float kExpMin     = 0.80f;
    constexpr float kExpMax     = 2.20f;

    bool                  s_inited = false;
    VkPipeline            s_PipeBuild = VK_NULL_HANDLE;
    VkPipeline            s_PipeBlur  = VK_NULL_HANDLE;
    VkPipelineLayout      s_Layout    = VK_NULL_HANDLE;   // shared: 1 sampler set + push
    VkDescriptorSetLayout s_SetLayout = VK_NULL_HANDLE;
    VkDescriptorPool      s_Pool      = VK_NULL_HANDLE;
    VkDescriptorSet       s_SetScene[kMaxImages] = {};    // build: scene sample view per image idx
    VkDescriptorSet       s_SetA = VK_NULL_HANDLE;        // blur H reads A
    VkDescriptorSet       s_SetB = VK_NULL_HANDLE;        // blur V reads B
    VkSampler             s_Sampler = VK_NULL_HANDLE;
    u32                   s_boundSceneGen = 0;

    // Quarter-res ping-pong RTs (A holds the final result).
    VkImage       s_img[2] = {};
    VmaAllocation s_alloc[2] = {};
    VkImageView   s_view[2] = {};
    VkExtent2D    s_extent = {};
    u32           s_generation = 0;
    bool          s_first[2] = { true, true };   // images start UNDEFINED

    struct BuildPush { float p0[4]; float p1[4]; };
    struct BlurPush  { float dir[4]; };
    static_assert(sizeof(BuildPush) >= sizeof(BlurPush), "shared push range sized by BuildPush");

    void DestroyRTs()
    {
        if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
        for (u32 i = 0; i < 2; ++i) {
            if (s_view[i]) { vkDestroyImageView(VulkanHW.m_Device, s_view[i], nullptr); s_view[i] = VK_NULL_HANDLE; }
            if (s_img[i])  { vmaDestroyImage(VulkanHW.m_Allocator, s_img[i], s_alloc[i]); s_img[i] = VK_NULL_HANDLE; s_alloc[i] = VK_NULL_HANDLE; }
            s_first[i] = true;
        }
        s_extent = {};
    }

    bool EnsureRTs(VkExtent2D sceneExtent)
    {
        VkExtent2D want{ sceneExtent.width / 4 ? sceneExtent.width / 4 : 1,
                         sceneExtent.height / 4 ? sceneExtent.height / 4 : 1 };
        if (s_img[0] && want.width == s_extent.width && want.height == s_extent.height)
            return true;
        DestroyRTs();
        s_extent = want;
        for (u32 i = 0; i < 2; ++i) {
            VkImageCreateInfo ici{};
            ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            ici.imageType = VK_IMAGE_TYPE_2D;
            ici.format = VK_FORMAT_R16G16B16A16_SFLOAT;
            ici.extent = { want.width, want.height, 1 };
            ici.mipLevels = 1; ici.arrayLayers = 1;
            ici.samples = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling = VK_IMAGE_TILING_OPTIMAL;
            ici.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            if (vmaCreateImage(VulkanHW.m_Allocator, &ici, &aci, &s_img[i], &s_alloc[i], nullptr) != VK_SUCCESS) {
                Msg("![VK Bloom] RT %u create failed", i); return false;
            }
            VkImageViewCreateInfo vci{};
            vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vci.image = s_img[i];
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format = VK_FORMAT_R16G16B16A16_SFLOAT;
            vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            vci.subresourceRange.levelCount = 1;
            vci.subresourceRange.layerCount = 1;
            if (vkCreateImageView(VulkanHW.m_Device, &vci, nullptr, &s_view[i]) != VK_SUCCESS) {
                Msg("![VK Bloom] view %u create failed", i); return false;
            }
        }
        // Blur sets point at the (new) ping-pong views.
        auto writeSet = [&](VkDescriptorSet set, VkImageView view) {
            VkDescriptorImageInfo ii{ s_Sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = set; w.dstBinding = 0; w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w.pImageInfo = &ii;
            vkUpdateDescriptorSets(VulkanHW.m_Device, 1, &w, 0, nullptr);
        };
        writeSet(s_SetA, s_view[0]);
        writeSet(s_SetB, s_view[1]);
        ++s_generation;
        Msg("[VK Bloom] RTs ready (%ux%u, gen %u)", want.width, want.height, s_generation);
        return true;
    }

    VkPipeline CreatePipe(VkShaderModule vs, VkShaderModule fs)
    {
        VkPipelineVertexInputStateCreateInfo vi{}; vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        VkPipelineShaderStageCreateInfo st[2]{};
        st[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;   st[0].module = vs; st[0].pName = "main";
        st[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; st[1].module = fs; st[1].pName = "main";
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
        VkPipelineColorBlendAttachmentState ba{};
        ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo cb{}; cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1; cb.pAttachments = &ba;
        VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynState{}; dynState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynState.dynamicStateCount = 2; dynState.pDynamicStates = dyn;
        VkFormat fmt = VK_FORMAT_R16G16B16A16_SFLOAT;
        VkPipelineRenderingCreateInfo prci{};
        prci.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        prci.colorAttachmentCount = 1; prci.pColorAttachmentFormats = &fmt;
        VkGraphicsPipelineCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pi.pNext = &prci; pi.stageCount = 2; pi.pStages = st;
        pi.pVertexInputState = &vi; pi.pInputAssemblyState = &ia; pi.pViewportState = &vp;
        pi.pRasterizationState = &rs; pi.pMultisampleState = &ms; pi.pDepthStencilState = &ds;
        pi.pColorBlendState = &cb; pi.pDynamicState = &dynState; pi.layout = s_Layout;
        VkPipeline out = VK_NULL_HANDLE;
        if (vkCreateGraphicsPipelines(VulkanHW.m_Device, PipelineCache::GetCacheObject(), 1, &pi, nullptr, &out) != VK_SUCCESS)
            Msg("![VK Bloom] pipeline create failed");
        return out;
    }

    // One fullscreen pass: render into dst view with the given pipeline/set/push.
    void Draw(VkCommandBuffer cmd, VkImageView dst, VkPipeline pipe, VkDescriptorSet set,
              const void* push, u32 pushSize)
    {
        VkRenderingAttachmentInfo cAtt{};
        cAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        cAtt.imageView = dst;
        cAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        cAtt.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        cAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        VkRenderingInfo ri{};
        ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        ri.renderArea.extent = s_extent;
        ri.layerCount = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments = &cAtt;
        vkCmdBeginRendering(cmd, &ri);
        VkViewport vp{ 0.f, 0.f, (float)s_extent.width, (float)s_extent.height, 0.f, 1.f };
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc{ {0,0}, s_extent };
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_Layout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(cmd, s_Layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, pushSize, push);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRendering(cmd);
    }
}

VkImageView GetResultView() { return s_view[0]; }
u32         Generation()    { return s_generation; }

bool Init()
{
    if (s_inited) return s_PipeBuild != VK_NULL_HANDLE;
    s_inited = true;

    if (!g_ShaderManager) g_ShaderManager = xr_new<VK::CVulkanSPIRVLoader>();
    VkShaderModule vs    = g_ShaderManager->Load("tonemap.vert.spv");   // shared fullscreen triangle
    VkShaderModule build = g_ShaderManager->Load("bloom_build.frag.spv");
    VkShaderModule blur  = g_ShaderManager->Load("bloom_blur.frag.spv");
    if (vs == VK_NULL_HANDLE || build == VK_NULL_HANDLE || blur == VK_NULL_HANDLE) {
        Msg("![VK Bloom] shader load failed"); return false;
    }

    VkDescriptorSetLayoutBinding b{};
    b.binding = 0; b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b.descriptorCount = 1; b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lci.bindingCount = 1; lci.pBindings = &b;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &s_SetLayout) != VK_SUCCESS) {
        Msg("![VK Bloom] set layout failed"); return false;
    }

    const u32 nSets = kMaxImages + 2;
    VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, nSets };
    VkDescriptorPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = nSets; pci.poolSizeCount = 1; pci.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &s_Pool) != VK_SUCCESS) {
        Msg("![VK Bloom] pool failed"); return false;
    }
    {
        VkDescriptorSetLayout layouts[kMaxImages + 2];
        for (u32 i = 0; i < nSets; ++i) layouts[i] = s_SetLayout;
        VkDescriptorSet sets[kMaxImages + 2]{};
        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = s_Pool; dai.descriptorSetCount = nSets; dai.pSetLayouts = layouts;
        if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, sets) != VK_SUCCESS) {
            Msg("![VK Bloom] alloc sets failed"); return false;
        }
        for (u32 i = 0; i < kMaxImages; ++i) s_SetScene[i] = sets[i];
        s_SetA = sets[kMaxImages];
        s_SetB = sets[kMaxImages + 1];
    }

    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_LINEAR; si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.minLod = 0.0f; si.maxLod = VK_LOD_CLAMP_NONE;
    if (vkCreateSampler(VulkanHW.m_Device, &si, nullptr, &s_Sampler) != VK_SUCCESS) {
        Msg("![VK Bloom] sampler failed"); return false;
    }

    VkPushConstantRange pcr{ VK_SHADER_STAGE_FRAGMENT_BIT, 0, (u32)sizeof(BuildPush) };
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1; plci.pSetLayouts = &s_SetLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_Layout) != VK_SUCCESS) {
        Msg("![VK Bloom] pipeline layout failed"); return false;
    }

    s_PipeBuild = CreatePipe(vs, build);
    s_PipeBlur  = CreatePipe(vs, blur);
    if (s_PipeBuild == VK_NULL_HANDLE || s_PipeBlur == VK_NULL_HANDLE) return false;

    Msg("[VK Bloom] Init OK (threshold %.2f knee %.2f, quarter-res 9-tap gaussian)", kThreshold, kKnee);
    return true;
}

void Destroy()
{
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
    DestroyRTs();
    if (s_PipeBuild) { vkDestroyPipeline(VulkanHW.m_Device, s_PipeBuild, nullptr); s_PipeBuild = VK_NULL_HANDLE; }
    if (s_PipeBlur)  { vkDestroyPipeline(VulkanHW.m_Device, s_PipeBlur, nullptr);  s_PipeBlur = VK_NULL_HANDLE; }
    if (s_Layout)    { vkDestroyPipelineLayout(VulkanHW.m_Device, s_Layout, nullptr); s_Layout = VK_NULL_HANDLE; }
    if (s_Sampler)   { vkDestroySampler(VulkanHW.m_Device, s_Sampler, nullptr); s_Sampler = VK_NULL_HANDLE; }
    if (s_Pool)      { vkDestroyDescriptorPool(VulkanHW.m_Device, s_Pool, nullptr); s_Pool = VK_NULL_HANDLE; }
    if (s_SetLayout) { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_SetLayout, nullptr); s_SetLayout = VK_NULL_HANDLE; }
    s_inited = false; s_boundSceneGen = 0; s_generation = 0;
    for (auto& s : s_SetScene) s = VK_NULL_HANDLE;
    s_SetA = s_SetB = VK_NULL_HANDLE;
}

void Execute(VkCommandBuffer cmd, u32 imageIndex, VkExtent2D sceneExtent, u32 sceneGen)
{
    if (s_PipeBuild == VK_NULL_HANDLE) return;
    if (!EnsureRTs(sceneExtent)) return;

    // (Re)bind the scene sample views when SceneColor was recreated.
    if (sceneGen != s_boundSceneGen) {
        for (u32 i = 0; i < SceneColor::Count() && i < kMaxImages; ++i) {
            VkDescriptorImageInfo ii{ s_Sampler, SceneColor::GetSampleView(i), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = s_SetScene[i]; w.dstBinding = 0; w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w.pImageInfo = &ii;
            vkUpdateDescriptorSets(VulkanHW.m_Device, 1, &w, 0, nullptr);
        }
        s_boundSceneGen = sceneGen;
    }

    auto toColor = [&](u32 i) {
        ImageBarrier(cmd, s_img[i],
                     s_first[i] ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        s_first[i] = false;
    };
    auto toRead = [&](u32 i) {
        ImageBarrier(cmd, s_img[i],
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    };

    // Pick the mip whose size is closest to quarter res (mip 2 of the scene).
    const float quarterLod = 2.0f;
    const float topLod     = float(SceneColor::MipLevels() - 1);

    // 1) Bright pass: scene (mips, SHADER_READ after GenerateMips) → A.
    toColor(0);
    BuildPush bp{};
    bp.p0[0] = quarterLod; bp.p0[1] = topLod; bp.p0[2] = kMiddleGray; bp.p0[3] = kLowLum;
    bp.p1[0] = kExpMin;    bp.p1[1] = kExpMax; bp.p1[2] = kThreshold; bp.p1[3] = kKnee;
    Draw(cmd, s_view[0], s_PipeBuild, s_SetScene[imageIndex < kMaxImages ? imageIndex : 0], &bp, sizeof(bp));
    toRead(0);

    // 2) Blur H: A → B.
    toColor(1);
    BlurPush hp{}; hp.dir[0] = 1.f; hp.dir[1] = 0.f;
    Draw(cmd, s_view[1], s_PipeBlur, s_SetA, &hp, sizeof(hp));
    toRead(1);

    // 3) Blur V: B → A (final result in A).
    toColor(0);
    BlurPush vp2{}; vp2.dir[0] = 0.f; vp2.dir[1] = 1.f;
    Draw(cmd, s_view[0], s_PipeBlur, s_SetB, &vp2, sizeof(vp2));
    toRead(0);
}

}}  // namespace VK::BloomPass
