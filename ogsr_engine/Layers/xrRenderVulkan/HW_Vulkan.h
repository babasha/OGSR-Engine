// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

#pragma once
#include "vk_core.h"
#include "HWCaps_Vulkan.h"

// Аналог CHW из DirectX, но для Vulkan (переименован чтобы избежать конфликта с R4)
class CVulkanHW : public pureAppActivate, public pureAppDeactivate
{
public:
    // Vulkan Core handles
    VkInstance              m_Instance       = VK_NULL_HANDLE;
    VkPhysicalDevice        m_PhysicalDevice = VK_NULL_HANDLE;
    VkDevice                m_Device         = VK_NULL_HANDLE;
    VkSurfaceKHR            m_Surface        = VK_NULL_HANDLE;

    // Queues.
    // Compute/Transfer prefer DEDICATED families (async-capable) and fall back to
    // the graphics family when the GPU exposes none — in that case the handle
    // aliases m_GraphicsQueue and the *Family index equals m_GraphicsFamily, so
    // any future async submit must check `m_ComputeFamily != m_GraphicsFamily`
    // before assuming real overlap. Layer 1 only discovers + creates them; no
    // async submission is wired yet (see HW_Vulkan.cpp FindQueueFamilies).
    VkQueue     m_GraphicsQueue  = VK_NULL_HANDLE;
    VkQueue     m_PresentQueue   = VK_NULL_HANDLE;
    VkQueue     m_ComputeQueue   = VK_NULL_HANDLE;
    VkQueue     m_TransferQueue  = VK_NULL_HANDLE;
    u32         m_GraphicsFamily = UINT32_MAX;
    u32         m_PresentFamily  = UINT32_MAX;
    u32         m_ComputeFamily  = UINT32_MAX;
    u32         m_TransferFamily = UINT32_MAX;

    // Memory allocator
    VmaAllocator m_Allocator = VK_NULL_HANDLE;

    // Command pool for transfer operations
    VkCommandPool m_TransferCommandPool = VK_NULL_HANDLE;

    // Capabilities (аналог Caps из DX11)
    VulkanCaps Caps;

    // Bindless descriptor indexing support (Vulkan 1.2 core)
    bool m_bBindlessSupported = false;

    // Hardware ray tracing support (VK_KHR_ray_query + acceleration_structure)
    bool m_bRayQuerySupported = false;

    // tessellationShader device feature (world heightmap tessellation, R4 TESS_HM)
    bool m_bTessellationSupported = false;

    // VK_NV_device_diagnostic_checkpoints: named GPU progress markers, readable
    // AFTER a device loss — turns a TDR/DEVICE_LOST into "hung between pass X and Y"
    // in the log (see VK::Prof::DumpCheckpoints).
    bool m_bCheckpointsSupported = false;

    // Variable Rate Shading — VK_KHR_fragment_shading_rate (attachment-based):
    // coarse-shade distant/peripheral pixels via a shading-rate image. m_VRSTexelSize
    // is the SRI tile size queried from the device (NVIDIA = 16x16).
    bool       m_bVRSSupported = false;
    VkExtent2D m_VRSTexelSize  = { 16, 16 };
    // pipelineFragmentShadingRate: a per-pipeline coarse rate (no SRI image). Used to
    // 2x2-shade the volumetric light cones (low-freq media) — see vk_pass_lightcones.
    bool       m_bVRSPipelineSupported = false;

    // VK_EXT_memory_budget: real per-heap VRAM budget/usage (VMA queries the OS +
    // driver rather than guessing). Feeds the texture streamer's residency budget
    // and the ResourcesGetMemoryUsage report. See GetVramBudget() below.
    bool       m_bMemoryBudgetSupported   = false;
    // VK_EXT_memory_priority: per-allocation eviction priority (0..1). Streamable
    // world textures get LOW priority so the driver spills THEM to system RAM under
    // pressure before touching render targets / VSM / geometry (HIGH priority).
    bool       m_bMemoryPrioritySupported = false;

    // Window handle (аналог m_hWnd из DX11)
    HWND m_hWnd = nullptr;

    VkDebugUtilsMessengerEXT m_DebugMessenger = VK_NULL_HANDLE;

public:
    CVulkanHW();
    ~CVulkanHW();

    // Основные методы (аналог DX11)
    bool CreateDevice(HWND hWnd);  // Returns true on success, false on error
    void DestroyDevice();
    void Reset(HWND hWnd);

    // App activation/deactivation
    virtual void OnAppActivate() override;
    virtual void OnAppDeactivate() override;

    // Single-time command helpers (for buffer uploads, etc.)
    VkCommandBuffer BeginSingleTimeCommands();
    void EndSingleTimeCommands(VkCommandBuffer cmd);

    // Accessor methods
    VmaAllocator GetAllocator() const { return m_Allocator; }
    VkDevice GetDevice() const { return m_Device; }

    // Real device-local VRAM budget/usage in bytes, summed over all DEVICE_LOCAL
    // heaps. When VK_EXT_memory_budget is active these are the driver's live figures
    // (what the OS + other apps actually left us); otherwise they fall back to the
    // static heap size (budget) and VMA's own bookkeeping (usage). Safe to call every
    // frame — VMA caches the vkGetPhysicalDeviceMemoryProperties2 query internally.
    void GetVramBudget(VkDeviceSize& outUsage, VkDeviceSize& outBudget) const;

private:
    // Вспомогательные методы
    bool FindQueueFamilies();
    bool CreateLogicalDevice();
    bool CreateVMA();
};

// Глобальный экземпляр Vulkan HW (переименован чтобы избежать конфликта с R4)
extern ECORE_API CVulkanHW VulkanHW;
