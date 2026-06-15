// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

#pragma once
#include "vk_core.h"

// Shared fullscreen-triangle pass helpers. The post-processing passes (bloom,
// ssao, sunshafts, tonemap) all built the SAME graphics-pipeline skeleton — no
// vertex input, triangle list, cull none, no depth attachment, 1 color
// attachment, dynamic viewport+scissor — differing only in shaders, color
// format, blend attachment and pipeline layout. CreatePipeline captures that.
// DrawSimple captures the bloom/ssao draw (1 set, DONT_CARE load, fragment push).
namespace VK { namespace Fullscreen {

// Opaque no-blend attachment writing the given channels (default RGBA).
VkPipelineColorBlendAttachmentState OpaqueAttachment(
    VkColorComponentFlags writeMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT);

// Additive (ONE, ONE / ADD) RGBA attachment — e.g. sun-shafts over the scene.
VkPipelineColorBlendAttachmentState AdditiveAttachment();

// Build a fullscreen-triangle graphics pipeline. `tag` is only used in the
// failure log. Returns VK_NULL_HANDLE on failure (logged).
VkPipeline CreatePipeline(VkShaderModule vs, VkShaderModule fs, VkFormat colorFmt,
                          VkPipelineLayout layout,
                          const VkPipelineColorBlendAttachmentState& blend,
                          const char* tag);

// One fullscreen draw into dstView at `extent`: DONT_CARE color load, positive
// viewport + full scissor, bind `set` at 0, push `pushSize` bytes to FRAGMENT,
// draw the 3-vertex fullscreen triangle. Caller's image must be in
// COLOR_ATTACHMENT_OPTIMAL. (bloom/ssao shape — passes that bind extra sets or
// sample depth keep their own draw.)
void DrawSimple(VkCommandBuffer cmd, VkImageView dstView, VkExtent2D extent,
                VkPipeline pipe, VkPipelineLayout layout, VkDescriptorSet set,
                const void* push, u32 pushSize);

}}  // namespace VK::Fullscreen
