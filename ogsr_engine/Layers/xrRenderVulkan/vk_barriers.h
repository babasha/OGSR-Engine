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

// ---------------------------------------------------------------------------
// ImageState — per-image layout/stage/access tracker (the framegraph execute seam).
//
// The first brick of the render graph. Instead of every consumer hand-placing a
// flip-to-READ / flip-back-to-ATTACHMENT round-trip, it calls Require(the state
// it needs) and a barrier is emitted ONLY when the tracked state actually has to
// change. A RUN of consumers that all need the SAME state (SSAO, VRS, VSM-mark,
// VSM-resolve all sampling the prepass depth) coalesces to a single transition —
// killing the depth-thrash (4 round-trips -> 1). A later write correctly waits on
// the union of every intervening reader (stage/access are OR-accumulated on the
// read-after-read no-op). Generalizes to every scene resource and, once it carries
// a queue family, to cross-queue ownership for async compute.
// ---------------------------------------------------------------------------
struct ImageState
{
    VkImage               image  = VK_NULL_HANDLE;
    VkImageAspectFlags    aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    VkImageLayout         layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkPipelineStageFlags2 stage  = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    VkAccessFlags2        access = 0;

    // Record the state the image is already known to be in — no barrier emitted.
    void Seed(VkImage img, VkImageAspectFlags asp, VkImageLayout lay,
              VkPipelineStageFlags2 stg, VkAccessFlags2 acc)
    { image = img; aspect = asp; layout = lay; stage = stg; access = acc; }

    // Make the image usable as (lay, stg, acc). Emits vkCmdPipelineBarrier2 only
    // when the layout differs or a write hazard exists; a read-after-read at the
    // same layout no-ops (the coalescing win) but is remembered for the next write.
    void Require(VkCommandBuffer cmd, VkImageLayout lay,
                 VkPipelineStageFlags2 stg, VkAccessFlags2 acc,
                 u32 mipLevels = 1, u32 layerCount = 1);
};

} // namespace VK
