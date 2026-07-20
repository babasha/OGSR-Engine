// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

#include "stdafx.h"
#include "vk_compute_util.h"
#include "vk_pipeline_cache.h"   // VK::PipelineCache::GetCacheObject
#include "HW_Vulkan.h"           // VulkanHW (device)

namespace VK
{

VkPipelineLayout MakePipelineLayout(std::initializer_list<VkDescriptorSetLayout> setLayouts,
                                    u32 pushSize, VkShaderStageFlags pushStages)
{
    VkPushConstantRange pcr{ pushStages, 0, pushSize };

    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = (u32)setLayouts.size();
    plci.pSetLayouts    = setLayouts.begin();   // initializer_list storage is contiguous
    if (pushSize > 0)
    {
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcr;
    }

    VkPipelineLayout layout = VK_NULL_HANDLE;
    const VkResult res = vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &layout);
    if (res != VK_SUCCESS)
    {
        Msg("![VK] pipeline layout create failed (vk=%d)", (int)res);
        return VK_NULL_HANDLE;
    }
    return layout;
}

VkPipeline CreateComputePipeline(VkShaderModule cs, VkPipelineLayout layout, const char* tag)
{
    VkComputePipelineCreateInfo cp{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cp.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cp.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cp.stage.module = cs;
    cp.stage.pName  = "main";
    cp.layout       = layout;

    VkPipeline pipe = VK_NULL_HANDLE;
    const VkResult res = vkCreateComputePipelines(VulkanHW.m_Device, VK::PipelineCache::GetCacheObject(),
                                                  1, &cp, nullptr, &pipe);
    if (res != VK_SUCCESS)
    {
        Msg("![VK] compute pipeline '%s' create failed (vk=%d)", tag ? tag : "?", (int)res);
        return VK_NULL_HANDLE;
    }
    return pipe;
}

} // namespace VK
