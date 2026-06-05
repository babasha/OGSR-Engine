// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
// Licensed under the same terms as X-Ray Engine (see root License.txt)

#pragma once
#include "vk_core.h"
#include <vector>

// Swapchain management
class CVulkanSwapchain
{
public:
    VkSwapchainKHR           m_Swapchain  = VK_NULL_HANDLE;
    std::vector<VkImage>     m_Images;
    std::vector<VkImageView> m_ImageViews;
    VkFormat                 m_Format     = VK_FORMAT_UNDEFINED;
    VkExtent2D               m_Extent     = {};
    u32                      m_ImageCount = 0;

    // Per-swapchain-image "render finished" semaphores (signaled by the frame's
    // queue submit, waited by vkQueuePresentKHR). MUST be indexed by the acquired
    // imageIndex, NOT by frame-in-flight: a binary semaphore signaled by submit is
    // only freed once its present has executed, and an image isn't re-acquired
    // until its present completed — so a per-image semaphore is guaranteed
    // unsignaled before the next submit re-signals it. Indexing by frame-in-flight
    // (acquire can return images in any order) lets a submit re-signal a semaphore
    // a pending present still owns → VUID-vkQueueSubmit-pSignalSemaphores-00067.
    std::vector<VkSemaphore> m_RenderFinished;

    // Depth buffer
    VkImage       m_DepthImage      = VK_NULL_HANDLE;
    VkImageView   m_DepthView       = VK_NULL_HANDLE;
    VkImageView   m_DepthImageView  = VK_NULL_HANDLE;  // Alias for m_DepthView (for compatibility)
    VmaAllocation m_DepthAllocation = VK_NULL_HANDLE;
    VkFormat      m_DepthFormat     = VK_FORMAT_D32_SFLOAT;

    bool m_IsMinimized = false;

    // Whether any pass wrote to the swapchain this frame (set by phase_combine / EndUIPass)
    bool m_bRenderedThisFrame = false;

    // Current frame tracking
    VkImage       m_CurrentImage     = VK_NULL_HANDLE;
    VkImageView   m_CurrentImageView = VK_NULL_HANDLE;
    u32           m_CurrentImageIndex = 0;

public:
    void Create(u32 width, u32 height);
    void Destroy();
    void Recreate(u32 width, u32 height);

    u32 AcquireNextImage(VkSemaphore semaphore, VkFence fence = VK_NULL_HANDLE);
    void Present(VkSemaphore waitSemaphore, u32 imageIndex);

    bool ShouldRender() const { return !m_IsMinimized && m_Swapchain != VK_NULL_HANDLE; }

    // Accessors
    u32 GetWidth() const { return m_Extent.width; }
    u32 GetHeight() const { return m_Extent.height; }
    VkFormat GetFormat() const { return m_Format; }
    VkImage GetCurrentImage() const { return m_CurrentImage; }
    VkImageView GetCurrentImageView() const { return m_CurrentImageView; }

private:
    VkSurfaceFormatKHR ChooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
    VkPresentModeKHR ChoosePresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
    VkExtent2D ChooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities, u32 width, u32 height);

    // Depth buffer
    void CreateDepthResources();
    void DestroyDepthResources();
    VkFormat FindDepthFormat();
    VkFormat FindSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features);
};

// Глобальный экземпляр
extern CVulkanSwapchain Swapchain;
