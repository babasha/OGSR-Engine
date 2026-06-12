// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan — HDR scene colour target. See vk_scene_color.h.
#include "stdafx.h"
#include "vk_scene_color.h"

namespace VK { namespace SceneColor {

namespace {
    constexpr u32 kMaxImages = 8;   // swapchain rarely exceeds 3-4; cap for the arrays

    VkFormat      s_format = VK_FORMAT_R16G16B16A16_SFLOAT;
    VkImage       s_image[kMaxImages] = {};
    VmaAllocation s_alloc[kMaxImages] = {};
    VkImageView   s_view[kMaxImages]  = {};       // mip-0 (render attachment)
    VkImageView   s_sampleView[kMaxImages] = {};  // full chain (tonemap sampling)
    u32           s_count  = 0;
    u32           s_mips   = 1;
    VkExtent2D    s_extent = {};
    u32           s_generation = 0;

    u32 CalcMips(u32 w, u32 h)
    {
        u32 m = 1, d = (w > h ? w : h);
        while (d > 1) { d >>= 1; ++m; }
        return m;
    }

    void DestroyAll()
    {
        if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
        for (u32 i = 0; i < s_count; ++i) {
            if (s_sampleView[i]) { vkDestroyImageView(VulkanHW.m_Device, s_sampleView[i], nullptr); s_sampleView[i] = VK_NULL_HANDLE; }
            if (s_view[i])  { vkDestroyImageView(VulkanHW.m_Device, s_view[i], nullptr); s_view[i] = VK_NULL_HANDLE; }
            if (s_image[i]) { vmaDestroyImage(VulkanHW.m_Allocator, s_image[i], s_alloc[i]); s_image[i] = VK_NULL_HANDLE; s_alloc[i] = VK_NULL_HANDLE; }
        }
        s_count = 0;
    }
}

VkFormat    Format()            { return s_format; }
VkImage     GetImage(u32 i)     { return (i < s_count) ? s_image[i] : VK_NULL_HANDLE; }
VkImageView GetView(u32 i)      { return (i < s_count) ? s_view[i]  : VK_NULL_HANDLE; }
VkImageView GetSampleView(u32 i){ return (i < s_count) ? s_sampleView[i] : VK_NULL_HANDLE; }
u32         MipLevels()         { return s_mips; }
u32         Count()             { return s_count; }
u32         Generation()        { return s_generation; }

void EnsureSize(VkExtent2D extent, u32 count)
{
    if (count > kMaxImages) count = kMaxImages;
    if (count == s_count && extent.width == s_extent.width && extent.height == s_extent.height && s_count > 0)
        return;   // already current

    // Resize happens at swapchain recreation, which idles the device; safe to
    // free + recreate here.
    DestroyAll();
    s_extent = extent;
    s_mips   = CalcMips(extent.width, extent.height);   // full chain → top mip = whole-frame average

    for (u32 i = 0; i < count; ++i) {
        VkImageCreateInfo ici{};
        ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType     = VK_IMAGE_TYPE_2D;
        ici.format        = s_format;
        ici.extent        = { extent.width, extent.height, 1 };
        ici.mipLevels     = s_mips;
        ici.arrayLayers   = 1;
        ici.samples       = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
        ici.usage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                            VK_IMAGE_USAGE_TRANSFER_DST_BIT |   // per-frame clear + blit dst (mips)
                            VK_IMAGE_USAGE_TRANSFER_SRC_BIT;    // blit src (mip-gen for avg luminance)
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        if (vmaCreateImage(VulkanHW.m_Allocator, &ici, &aci, &s_image[i], &s_alloc[i], nullptr) != VK_SUCCESS) {
            Msg("![VK SceneColor] image %u create failed", i);
            s_count = i; return;
        }

        // mip-0 view (render attachment).
        VkImageViewCreateInfo vci{};
        vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image    = s_image[i];
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format   = s_format;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.baseMipLevel = 0;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.layerCount = 1;
        if (vkCreateImageView(VulkanHW.m_Device, &vci, nullptr, &s_view[i]) != VK_SUCCESS) {
            Msg("![VK SceneColor] view %u create failed", i);
            s_count = i; return;
        }
        // full-chain view (tonemap sampling: mip0 = scene, top mip = avg luminance).
        vci.subresourceRange.levelCount = s_mips;
        if (vkCreateImageView(VulkanHW.m_Device, &vci, nullptr, &s_sampleView[i]) != VK_SUCCESS) {
            Msg("![VK SceneColor] sample view %u create failed", i);
            s_count = i; return;
        }
    }
    s_count = count;
    ++s_generation;
    Msg("[VK SceneColor] HDR target ready (%ux%u R16F x%u, %u mips, gen %u)", extent.width, extent.height, count, s_mips, s_generation);
}

void GenerateMips(VkCommandBuffer cmd, u32 index)
{
    if (index >= s_count || s_mips <= 1) return;
    VkImage img = s_image[index];

    auto barrier = [&](u32 baseMip, u32 mipCount, VkImageLayout oldL, VkImageLayout newL,
                       VkAccessFlags2 srcA, VkAccessFlags2 dstA,
                       VkPipelineStageFlags2 srcS, VkPipelineStageFlags2 dstS) {
        VkImageMemoryBarrier2 b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        b.srcStageMask = srcS; b.srcAccessMask = srcA;
        b.dstStageMask = dstS; b.dstAccessMask = dstA;
        b.oldLayout = oldL; b.newLayout = newL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = img;
        b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, baseMip, mipCount, 0, 1 };
        VkDependencyInfo di{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &di);
    };

    // mip0: COLOR_ATTACHMENT (scene just rendered) → TRANSFER_SRC.
    barrier(0, 1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_2_BLIT_BIT);

    s32 mw = (s32)s_extent.width, mh = (s32)s_extent.height;
    for (u32 i = 1; i < s_mips; ++i) {
        const s32 nw = mw > 1 ? mw / 2 : 1;
        const s32 nh = mh > 1 ? mh / 2 : 1;
        barrier(i, 1, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                0, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_BLIT_BIT, VK_PIPELINE_STAGE_2_BLIT_BIT);
        VkImageBlit blit{};
        blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, 1 };
        blit.srcOffsets[1]  = { mw, mh, 1 };
        blit.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1 };
        blit.dstOffsets[1]  = { nw, nh, 1 };
        vkCmdBlitImage(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
        barrier(i, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_2_BLIT_BIT, VK_PIPELINE_STAGE_2_BLIT_BIT);
        mw = nw; mh = nh;
    }

    // All mips TRANSFER_SRC → SHADER_READ for the tonemap.
    barrier(0, s_mips, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_2_TRANSFER_READ_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
            VK_PIPELINE_STAGE_2_BLIT_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
}

void Destroy()
{
    DestroyAll();
    s_extent = {};
}

}}  // namespace VK::SceneColor
