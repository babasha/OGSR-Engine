// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

#pragma once

#include "vk_core.h"
#include <initializer_list>

namespace VK
{

// ============================================================================
// Compute pipeline helpers
//
// Collapses the two byte-identical tails every compute pass hand-rolls:
//   * VkPipelineLayoutCreateInfo (+ optional push-constant range) -> vkCreatePipelineLayout
//   * VkComputePipelineCreateInfo (single COMPUTE stage, "main") -> vkCreateComputePipelines
// The pipeline is always built against the shared PipelineCache (some sites used
// VK_NULL_HANDLE before — routing them through the cache is identical output, just
// warm-startable). Both return VK_NULL_HANDLE on failure and log a tagged error.
// ============================================================================

// Create a pipeline layout from a set of descriptor-set layouts and an optional
// push-constant block. pushSize == 0 means no push-constant range.
VkPipelineLayout MakePipelineLayout(std::initializer_list<VkDescriptorSetLayout> setLayouts,
                                    u32 pushSize = 0,
                                    VkShaderStageFlags pushStages = VK_SHADER_STAGE_COMPUTE_BIT);

// Create a compute pipeline from a shader module + layout (entry point "main").
// `tag` names the pipeline in the failure log. Does NOT take ownership of `cs`.
VkPipeline CreateComputePipeline(VkShaderModule cs, VkPipelineLayout layout, const char* tag = "compute");

// Same, but loads the module by SPIR-V name through g_ShaderManager first — the
// load+create pair every pass repeated by hand. `tag` defaults to the shader name.
VkPipeline CreateComputePipeline(const char* spv, VkPipelineLayout layout, const char* tag = nullptr);

} // namespace VK
