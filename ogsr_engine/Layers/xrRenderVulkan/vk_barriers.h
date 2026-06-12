// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

#pragma once
#include "vk_core.h"

namespace VK
{

// Image barrier with auto-derived stage/access from layouts.
// Uses VkImageMemoryBarrier2 + vkCmdPipelineBarrier2 (Vulkan 1.3).
void ImageBarrier(VkCommandBuffer cmd, VkImage image,
    VkImageLayout oldLayout, VkImageLayout newLayout,
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT,
    u32 layerCount = 1, u32 mipLevels = 1);

// Batch image barriers - same transition for multiple images.
void ImageBarriers(VkCommandBuffer cmd, u32 count, const VkImage* images,
    VkImageLayout oldLayout, VkImageLayout newLayout,
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT, u32 layerCount = 1);

// Buffer barrier - explicit stage/access (no auto-derivation).
void BufferBarrier(VkCommandBuffer cmd, VkBuffer buffer,
    VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
    VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess,
    VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE);

// Memory barrier - global sync point.
void MemoryBarrier(VkCommandBuffer cmd,
    VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
    VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess);

// Inter-pass ordering for the shared color+depth swapchain attachments WITHOUT a
// layout change. With the single-layout convention (image stays COLOR_ATTACHMENT
// the whole frame) two consecutive dynamic-rendering passes that LOAD the same
// color/depth target have no implicit dependency, so this orders pass N's
// attachment writes before pass N+1's attachment access. Replaces the old
// COLOR<->TRANSFER_DST round-trip each pass used to do (the layout thrash).
// A real framegraph will later swap this conservative global barrier for precise
// per-resource ones.
void SceneAttachmentBarrier(VkCommandBuffer cmd);

} // namespace VK
