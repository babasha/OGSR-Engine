// xrRenderVulkan - VulkanUI subsystem implementation. See header for adaptation
// notes vs the monolith original.

#include "stdafx.h"
#include "vk_UIPipeline.h"
#include "vk_buffer.h"
#include "vk_texture.h"
#include "vk_shaders.h"
#include "vk_swapchain.h"
#include "vk_pipeline_cache.h"     // VK::PipelineCache::GetCacheObject() — shared disk-backed cache
#include "vk_barriers.h"           // VK::SceneAttachmentBarrier — inter-pass ordering
#include "HW_Vulkan.h"
#include "../../xr_3da/device.h"   // Device.dwWidth / dwHeight

VkCommandBuffer g_VkUI_FrameCmd = VK_NULL_HANDLE;

// Defined in vk_RenderFactory.cpp (global scope) — frees the engine-lifetime UI
// texture cache. Declared here so VulkanUI::Destroy can call it at teardown.
void VK_ClearUITextureCache();

namespace VulkanUI
{
    // Resources --------------------------------------------------------------
    static VK::CVulkanBuffer  s_VertexBuffer;
    static VK::CVulkanTexture s_WhiteTexture;
    VkDescriptorSetLayout    s_DescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool         s_DescriptorPool      = VK_NULL_HANDLE;
    VkPipelineLayout         s_PipelineLayout      = VK_NULL_HANDLE;
    VkPipeline               s_Pipeline            = VK_NULL_HANDLE;   // TRIANGLE_LIST (default UI quads)
    VkPipeline               s_PipelineLineList    = VK_NULL_HANDLE;   // LINE_LIST  (crosshair)
    VkPipeline               s_PipelineLineStrip   = VK_NULL_HANDLE;   // LINE_STRIP (UIWindow borders)
    VkDescriptorSet          s_WhiteTextureSet     = VK_NULL_HANDLE;

    // Frame state ------------------------------------------------------------
    bool   s_bUIPassActive  = false;
    u32    s_UIVertexOffset = 0;
    void*  s_pMappedVB      = nullptr;

    // Deferred command queue --------------------------------------------------
    DeferredUICmd s_DeferredCmds[MAX_DEFERRED_CMDS];
    u32           s_DeferredCmdCount = 0;

    FrameStats s_FrameStats;

    // ----- Helpers ----------------------------------------------------------
    VkBuffer GetVertexBufferHandle() { return s_VertexBuffer.GetHandle(); }
    void     FlushVertexBuffer()     { s_VertexBuffer.Flush(); }

    // Swapchain layout is managed centrally now (single-layout convention): the
    // image is in COLOR_ATTACHMENT for the whole frame, so the UI pass does no
    // layout transition — only an inter-pass ordering barrier on entry.

    // ----- Create / Destroy -------------------------------------------------

    void Create()
    {
        Msg("[Vulkan UI] Creating UI infrastructure...");

        // 1. Vertex buffer (persistent-mapped).
        s_VertexBuffer.Create(VERTEX_BUFFER_SIZE,
                              VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                              VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
        s_pMappedVB = s_VertexBuffer.Map();
        if (!s_pMappedVB) { Msg("![Vulkan UI] Failed to map vertex buffer"); return; }

        // 2. Descriptor set layout.
        VkDescriptorSetLayoutBinding samplerBinding{};
        samplerBinding.binding         = 0;
        samplerBinding.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerBinding.descriptorCount = 1;
        samplerBinding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 1;
        layoutInfo.pBindings    = &samplerBinding;
        VK_CHECK(vkCreateDescriptorSetLayout(VulkanHW.m_Device, &layoutInfo, nullptr, &s_DescriptorSetLayout));

        // 3. Descriptor pool — 256 sets covers menu fonts + textures.
        VkDescriptorPoolSize poolSize{};
        poolSize.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSize.descriptorCount = 256;

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.maxSets       = 256;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes    = &poolSize;
        VK_CHECK(vkCreateDescriptorPool(VulkanHW.m_Device, &poolInfo, nullptr, &s_DescriptorPool));

        // 4. Pipeline layout: 1 descriptor set + push-constant for screen size.
        VkPushConstantRange pushConstant{};
        pushConstant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pushConstant.offset     = 0;
        pushConstant.size       = 8;  // vec2 screenSize

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount         = 1;
        pipelineLayoutInfo.pSetLayouts            = &s_DescriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges    = &pushConstant;
        VK_CHECK(vkCreatePipelineLayout(VulkanHW.m_Device, &pipelineLayoutInfo, nullptr, &s_PipelineLayout));

        // 5. Fallback texture — used when a UI shader's texture file is missing.
        // Solid white is hostile (paints menu bg pure white). Solid transparent
        // hides backgrounds entirely (looks empty). Use a dim Stalker-grey so
        // missing assets at least look like an intentional Zone-mood backdrop.
        u32 stalkerGrey = 0xFF1A2018u;  // R8G8B8A8: ARGB ≈ dark olive-grey
        s_WhiteTexture.CreateFromData(&stalkerGrey, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, 4);

        // 6. Descriptor set for white texture.
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool     = s_DescriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts        = &s_DescriptorSetLayout;
        VK_CHECK(vkAllocateDescriptorSets(VulkanHW.m_Device, &allocInfo, &s_WhiteTextureSet));

        VkDescriptorImageInfo imageInfo{};
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView   = s_WhiteTexture.GetView();
        imageInfo.sampler     = s_WhiteTexture.GetSampler();

        VkWriteDescriptorSet write{};
        write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet          = s_WhiteTextureSet;
        write.dstBinding      = 0;
        write.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.descriptorCount = 1;
        write.pImageInfo      = &imageInfo;
        vkUpdateDescriptorSets(VulkanHW.m_Device, 1, &write, 0, nullptr);

        // 7. Lazily create the SPIR-V loader, then load UI shaders.
        if (!g_ShaderManager) {
            g_ShaderManager = xr_new<VK::CVulkanSPIRVLoader>();
            Msg("[Vulkan UI] Created g_ShaderManager (lazy init)");
        }
        VkShaderModule vertShader = g_ShaderManager->Load("ui.vert.spv");
        VkShaderModule fragShader = g_ShaderManager->Load("ui.frag.spv");
        if (vertShader == VK_NULL_HANDLE || fragShader == VK_NULL_HANDLE) {
            Msg("![Vulkan UI] Failed to load UI shaders (ui.vert.spv / ui.frag.spv)");
            return;
        }

        // 8. Build the graphics pipeline. FVF::TL = vec4 pos + u32 color + vec2 uv = 28 B.
        VkPipelineShaderStageCreateInfo shaderStages[2]{};
        shaderStages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
        shaderStages[0].module = vertShader;
        shaderStages[0].pName  = "main";
        shaderStages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        shaderStages[1].module = fragShader;
        shaderStages[1].pName  = "main";

        VkVertexInputBindingDescription binding{};
        binding.binding   = 0;
        binding.stride    = 28;
        binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkVertexInputAttributeDescription attributes[3]{};
        attributes[0].binding = 0; attributes[0].location = 0; attributes[0].format = VK_FORMAT_R32G32B32A32_SFLOAT; attributes[0].offset = 0;
        attributes[1].binding = 0; attributes[1].location = 1; attributes[1].format = VK_FORMAT_B8G8R8A8_UNORM;     attributes[1].offset = 16;
        attributes[2].binding = 0; attributes[2].location = 2; attributes[2].format = VK_FORMAT_R32G32_SFLOAT;       attributes[2].offset = 20;

        VkPipelineVertexInputStateCreateInfo vertexInput{};
        vertexInput.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInput.vertexBindingDescriptionCount   = 1;
        vertexInput.pVertexBindingDescriptions      = &binding;
        vertexInput.vertexAttributeDescriptionCount = 3;
        vertexInput.pVertexAttributeDescriptions    = attributes;

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.scissorCount  = 1;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth   = 1.0f;
        rasterizer.cullMode    = VK_CULL_MODE_NONE;
        rasterizer.frontFace   = VK_FRONT_FACE_CLOCKWISE;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        // depth disabled for UI

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable         = VK_TRUE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        colorBlendAttachment.colorBlendOp        = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.alphaBlendOp        = VK_BLEND_OP_ADD;

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments    = &colorBlendAttachment;

        VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = 2;
        dynamicState.pDynamicStates    = dynamicStates;

        VkFormat colorFormat = Swapchain.m_Format;
        VkPipelineRenderingCreateInfo renderingInfo{};
        renderingInfo.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        renderingInfo.colorAttachmentCount    = 1;
        renderingInfo.pColorAttachmentFormats = &colorFormat;

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.pNext               = &renderingInfo;
        pipelineInfo.stageCount          = 2;
        pipelineInfo.pStages             = shaderStages;
        pipelineInfo.pVertexInputState   = &vertexInput;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState      = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState   = &multisampling;
        pipelineInfo.pDepthStencilState  = &depthStencil;
        pipelineInfo.pColorBlendState    = &colorBlending;
        pipelineInfo.pDynamicState       = &dynamicState;
        pipelineInfo.layout              = s_PipelineLayout;

        VkPipelineCache pcache = VK::PipelineCache::GetCacheObject();
        VkResult result = vkCreateGraphicsPipelines(VulkanHW.m_Device, pcache, 1, &pipelineInfo, nullptr, &s_Pipeline);
        if (result != VK_SUCCESS) {
            Msg("![Vulkan UI] Failed to create UI pipeline! Error: %d", result);
            return;
        }

        // Line-topology variants — identical state, only inputAssembly.topology differs.
        // UIRender drives ptLineList (HUD crosshair) / ptLineStrip (UIWindow borders);
        // without these they'd render through the triangle-list pipeline and the line
        // endpoints would assemble into stray stretched triangles (the "sky spike").
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
        if (vkCreateGraphicsPipelines(VulkanHW.m_Device, pcache, 1, &pipelineInfo, nullptr, &s_PipelineLineList) != VK_SUCCESS)
            Msg("![Vulkan UI] Failed to create UI line-list pipeline");
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
        if (vkCreateGraphicsPipelines(VulkanHW.m_Device, pcache, 1, &pipelineInfo, nullptr, &s_PipelineLineStrip) != VK_SUCCESS)
            Msg("![Vulkan UI] Failed to create UI line-strip pipeline");

        Msg("[Vulkan UI] UI infrastructure created successfully");
    }

    void Destroy()
    {
        if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
        Msg("[Vulkan UI] Destroying UI infrastructure...");
        vkDeviceWaitIdle(VulkanHW.m_Device);

        // Free the engine-lifetime UI texture cache (UI/map/font/HUD textures +
        // any video staging buffer) before we tear down the descriptor pool and
        // (later) the VMA allocator. Defined at global scope in vk_RenderFactory.cpp.
        ::VK_ClearUITextureCache();

        if (s_Pipeline)            { vkDestroyPipeline(VulkanHW.m_Device, s_Pipeline, nullptr);                   s_Pipeline            = VK_NULL_HANDLE; }
        if (s_PipelineLineList)    { vkDestroyPipeline(VulkanHW.m_Device, s_PipelineLineList, nullptr);           s_PipelineLineList    = VK_NULL_HANDLE; }
        if (s_PipelineLineStrip)   { vkDestroyPipeline(VulkanHW.m_Device, s_PipelineLineStrip, nullptr);          s_PipelineLineStrip   = VK_NULL_HANDLE; }
        if (s_PipelineLayout)      { vkDestroyPipelineLayout(VulkanHW.m_Device, s_PipelineLayout, nullptr);       s_PipelineLayout      = VK_NULL_HANDLE; }
        s_WhiteTexture.Destroy();
        if (s_DescriptorPool)      { vkDestroyDescriptorPool(VulkanHW.m_Device, s_DescriptorPool, nullptr);       s_DescriptorPool      = VK_NULL_HANDLE; }
        if (s_DescriptorSetLayout) { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_DescriptorSetLayout, nullptr); s_DescriptorSetLayout = VK_NULL_HANDLE; }
        if (s_pMappedVB)           { s_VertexBuffer.Unmap(); s_pMappedVB = nullptr; }
        s_VertexBuffer.Destroy();
        Msg("[Vulkan UI] UI infrastructure destroyed");
    }

    // ----- Frame UI pass ----------------------------------------------------

    void BeginUIPassInternal()
    {
        VkCommandBuffer cmd = g_VkUI_FrameCmd;
        if (!cmd || s_bUIPassActive) return;

        u32 imageIndex = Swapchain.m_CurrentImageIndex;
        if (imageIndex >= Swapchain.m_Images.size()) return;

        VkImage     img = Swapchain.m_Images[imageIndex];
        VkImageView view = Swapchain.m_ImageViews[imageIndex];
        if (!img || !view) return;

        // Image is already COLOR_ATTACHMENT (Begin set it; scene passes left it
        // there). Just order any prior scene color writes before the UI draws —
        // no layout change. Covers both the immediate path (opened mid-frame) and
        // the deferred replay in CRender::End.
        VK::SceneAttachmentBarrier(cmd);

        VkRenderingAttachmentInfo colorAttachment{};
        colorAttachment.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colorAttachment.imageView   = view;
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;   // preserve clear colour
        colorAttachment.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

        VkRenderingInfo renderInfo{};
        renderInfo.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
        renderInfo.renderArea.extent    = Swapchain.m_Extent;
        renderInfo.layerCount           = 1;
        renderInfo.colorAttachmentCount = 1;
        renderInfo.pColorAttachments    = &colorAttachment;
        vkCmdBeginRendering(cmd, &renderInfo);

        VkViewport viewport{};
        viewport.width    = (float)Swapchain.m_Extent.width;
        viewport.height   = (float)Swapchain.m_Extent.height;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.extent = Swapchain.m_Extent;
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        s_bUIPassActive = true;
    }

    void EndUIPass()
    {
        if (!s_bUIPassActive) return;
        VkCommandBuffer cmd = g_VkUI_FrameCmd;
        if (!cmd) return;

        vkCmdEndRendering(cmd);

        // No layout transition here: the image stays COLOR_ATTACHMENT. CRender::End
        // owns the single COLOR→PRESENT transition for the whole frame.
        Swapchain.m_bRenderedThisFrame = true;   // vestigial; End no longer branches on it
        s_bUIPassActive  = false;
        s_UIVertexOffset = 0;
    }

    void ReplayDeferredUI()
    {
        if (s_DeferredCmdCount == 0) return;
        VkCommandBuffer cmd = g_VkUI_FrameCmd;
        if (!cmd) { s_DeferredCmdCount = 0; return; }

        s_VertexBuffer.Flush();
        BeginUIPassInternal();
        if (s_Pipeline == VK_NULL_HANDLE) { s_DeferredCmdCount = 0; return; }

        for (u32 i = 0; i < s_DeferredCmdCount; ++i)
        {
            const DeferredUICmd& dcmd = s_DeferredCmds[i];
            switch (dcmd.type)
            {
            case DeferredUICmd::Draw: {
                VkPipeline pipe = dcmd.pipeline ? dcmd.pipeline : s_Pipeline;
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
                float screenSize[2] = { (float)Device.dwWidth, (float)Device.dwHeight };
                vkCmdPushConstants(cmd, s_PipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, 8, screenSize);
                VkBuffer     vbs[]     = { s_VertexBuffer.GetHandle() };
                VkDeviceSize offsets[] = { dcmd.vertexBufferOffset };
                vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offsets);
                VkDescriptorSet texSet = (dcmd.textureSet != VK_NULL_HANDLE) ? dcmd.textureSet : s_WhiteTextureSet;
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_PipelineLayout, 0, 1, &texSet, 0, nullptr);
                vkCmdDraw(cmd, dcmd.vertexCount, 1, 0, 0);
                break;
            }
            case DeferredUICmd::Scissor:
                vkCmdSetScissor(cmd, 0, 1, &dcmd.scissorRect);
                break;
            case DeferredUICmd::ResetScissor: {
                VkRect2D fullScissor{};
                fullScissor.extent = Swapchain.m_Extent;
                vkCmdSetScissor(cmd, 0, 1, &fullScissor);
                break;
            }
            }
        }
        s_DeferredCmdCount = 0;
    }
}  // namespace VulkanUI
