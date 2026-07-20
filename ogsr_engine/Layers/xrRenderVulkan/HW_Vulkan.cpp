// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

#include "stdafx.h"
#include "HW_Vulkan.h"
#include "vk_sl.h"          // VK::SL — Streamline init/probe (must precede vkCreateInstance)
#include "vk_geometry.h"
#include "vk_lighting.h"
#include "vk_authorship.h"
#include "vk_command_buffer.h"   // CommandManager — single-time helpers delegate to the immediate path
#include "vk_vram_stats.h"       // VK::Vram::DestroySmallPools — before vmaDestroyAllocator
#include <vector>
#include <set>

// Use VK namespace globals
using VK::g_VulkanGeometry;
using VK::g_VulkanLighting;

// Глобальный экземпляр Vulkan HW (переименован чтобы избежать конфликта с R4)
CVulkanHW VulkanHW;

CVulkanHW::CVulkanHW()
{
    // DON'T use Msg() in global constructors - logging system isn't initialized yet!
    // Msg("[Vulkan] CVulkanHW constructor");
}

CVulkanHW::~CVulkanHW()
{
    // Msg("[Vulkan] CVulkanHW destructor");
}

// Поиск queue families (graphics + present)
bool CVulkanHW::FindQueueFamilies()
{
    u32 queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(m_PhysicalDevice, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(m_PhysicalDevice, &queueFamilyCount, queueFamilies.data());

    // Ищем graphics queue и present queue
    m_GraphicsFamily = UINT32_MAX;
    m_PresentFamily = UINT32_MAX;

    for (u32 i = 0; i < queueFamilyCount; i++) {
        // Graphics queue
        if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            m_GraphicsFamily = i;
        }

        // Present queue
        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(m_PhysicalDevice, i, m_Surface, &presentSupport);
        if (presentSupport) {
            m_PresentFamily = i;
        }

        // Если нашли обе - выходим
        if (m_GraphicsFamily != UINT32_MAX && m_PresentFamily != UINT32_MAX) {
            break;
        }
    }

    if (m_GraphicsFamily == UINT32_MAX) {
        Msg("!Failed to find graphics queue family");
        return false;
    }

    if (m_PresentFamily == UINT32_MAX) {
        Msg("!Failed to find present queue family");
        return false;
    }

    Msg("[Vulkan] Graphics queue family: %u", m_GraphicsFamily);
    if (m_GraphicsFamily == m_PresentFamily) {
        Msg("[Vulkan] Present queue family: %u (same as graphics)", m_PresentFamily);
    } else {
        Msg("[Vulkan] Present queue family: %u", m_PresentFamily);
    }

    // --- Dedicated compute family (Layer 1: discovery only, not yet submitted to) ---
    // Prefer a COMPUTE family WITHOUT graphics → a genuinely async-capable queue.
    // Fall back to any compute-capable family (usually == graphics) so the handle
    // is always valid; callers gate real overlap on (m_ComputeFamily != m_GraphicsFamily).
    m_ComputeFamily = UINT32_MAX;
    for (u32 i = 0; i < queueFamilyCount; i++) {
        const VkQueueFlags f = queueFamilies[i].queueFlags;
        if ((f & VK_QUEUE_COMPUTE_BIT) && !(f & VK_QUEUE_GRAPHICS_BIT)) { m_ComputeFamily = i; break; }
    }
    if (m_ComputeFamily == UINT32_MAX) {
        for (u32 i = 0; i < queueFamilyCount; i++)
            if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { m_ComputeFamily = i; break; }
    }
    if (m_ComputeFamily == UINT32_MAX) m_ComputeFamily = m_GraphicsFamily;  // last resort

    // --- Dedicated transfer family ---
    // Prefer a pure DMA family (TRANSFER without graphics/compute) for off-timeline
    // uploads. Note: per spec GRAPHICS/COMPUTE families support transfer implicitly,
    // so the fallback to the graphics family is always valid.
    m_TransferFamily = UINT32_MAX;
    for (u32 i = 0; i < queueFamilyCount; i++) {
        const VkQueueFlags f = queueFamilies[i].queueFlags;
        if ((f & VK_QUEUE_TRANSFER_BIT) && !(f & VK_QUEUE_GRAPHICS_BIT) && !(f & VK_QUEUE_COMPUTE_BIT)) { m_TransferFamily = i; break; }
    }
    if (m_TransferFamily == UINT32_MAX) m_TransferFamily = m_GraphicsFamily;  // graphics implies transfer

    Msg("[Vulkan] Compute queue family: %u%s", m_ComputeFamily,
        (m_ComputeFamily == m_GraphicsFamily) ? " (shared with graphics — no async compute)" : " (dedicated, async-capable)");
    Msg("[Vulkan] Transfer queue family: %u%s", m_TransferFamily,
        (m_TransferFamily == m_GraphicsFamily) ? " (shared with graphics)" : " (dedicated DMA)");

    return true;
}

// Создание logical device
bool CVulkanHW::CreateLogicalDevice()
{
    // Queue create infos
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    // std::set dedups, so families that coincide (e.g. compute == graphics on a
    // single-family GPU) request just one queue; their handles will then alias below.
    std::set<u32> uniqueQueueFamilies = { m_GraphicsFamily, m_PresentFamily, m_ComputeFamily, m_TransferFamily };

    float queuePriority = 1.0f;
    for (u32 queueFamily : uniqueQueueFamilies) {
        VkDeviceQueueCreateInfo queueCreateInfo = {};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    // Vulkan 1.3 features
    VkPhysicalDeviceVulkan13Features features13 = {};
    features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    features13.dynamicRendering = VK_TRUE;
    features13.synchronization2 = VK_TRUE;
    features13.maintenance4 = VK_TRUE;

    VkPhysicalDeviceVulkan12Features features12 = {};
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features12.bufferDeviceAddress = VK_TRUE;
    features12.separateDepthStencilLayouts = VK_TRUE;
    features12.drawIndirectCount = VK_TRUE;
    features12.timelineSemaphore = VK_TRUE;
    features12.pNext = &features13;

    // Query descriptor indexing support for bindless textures
    {
        VkPhysicalDeviceVulkan12Features supported12 = {};
        supported12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        VkPhysicalDeviceFeatures2 supported = {};
        supported.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        supported.pNext = &supported12;
        vkGetPhysicalDeviceFeatures2(m_PhysicalDevice, &supported);

        m_bBindlessSupported =
            supported12.descriptorBindingPartiallyBound &&
            supported12.runtimeDescriptorArray &&
            supported12.shaderSampledImageArrayNonUniformIndexing;

        if (m_bBindlessSupported)
        {
            features12.descriptorBindingPartiallyBound = VK_TRUE;
            features12.runtimeDescriptorArray = VK_TRUE;
            features12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
            Msg("[Vulkan] Bindless descriptor indexing: ENABLED");
        }
        else
        {
            Msg("[Vulkan] Bindless descriptor indexing: NOT SUPPORTED (fallback to per-type loop)");
        }
    }

    // Device features
    VkPhysicalDeviceFeatures2 deviceFeatures = {};
    deviceFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    deviceFeatures.features.samplerAnisotropy = VK_TRUE;
    deviceFeatures.features.fillModeNonSolid = VK_TRUE;
    deviceFeatures.features.wideLines = VK_TRUE;
    deviceFeatures.features.multiDrawIndirect = VK_TRUE;
    deviceFeatures.features.drawIndirectFirstInstance = VK_TRUE;  // trees: firstInstance encodes global tree index
    deviceFeatures.features.shaderClipDistance = VK_TRUE;         // VSM: gl_ClipDistance clips page geometry to its atlas sub-rect
    deviceFeatures.features.imageCubeArray = VK_TRUE;             // point shadow POOL: array of shadow cubes (samplerCubeArray)
    deviceFeatures.features.fragmentStoresAndAtomics = VK_TRUE;   // texture-streaming GPU feedback: world FS atomicMin desired mips
    deviceFeatures.features.pipelineStatisticsQuery = VK_TRUE;    // VRS diag: FS-invocation counter around the world color pass
    deviceFeatures.pNext = &features12;

    // Tessellation (world heightmap displacement, R4 TESS_HM). Universal on
    // desktop GPUs, but query anyway — when absent the renderer keeps the
    // plain triangle pipelines and PipelineCache::TessAvailable() stays false.
    {
        VkPhysicalDeviceFeatures supportedBase{};
        vkGetPhysicalDeviceFeatures(m_PhysicalDevice, &supportedBase);
        m_bTessellationSupported = supportedBase.tessellationShader == VK_TRUE;
        if (m_bTessellationSupported)
            deviceFeatures.features.tessellationShader = VK_TRUE;
        else
            Msg("[Vulkan] tessellationShader not supported — world tessellation disabled");
    }

    // Build final extension list: required + optional NGX extensions if available
    std::vector<const char*> enabledExtensions(g_DeviceExtensions, g_DeviceExtensions + g_DeviceExtensionCount);

    // Check which optional NGX extensions are supported
    {
        u32 extCount = 0;
        vkEnumerateDeviceExtensionProperties(m_PhysicalDevice, nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> availableExts(extCount);
        vkEnumerateDeviceExtensionProperties(m_PhysicalDevice, nullptr, &extCount, availableExts.data());

        u32 ngxFound = 0;
        for (u32 i = 0; i < g_NgxDeviceExtensionCount; i++) {
            bool found = false;
            for (const auto& ext : availableExts) {
                if (strcmp(g_NgxDeviceExtensions[i], ext.extensionName) == 0) {
                    found = true;
                    break;
                }
            }
            if (found) {
                enabledExtensions.push_back(g_NgxDeviceExtensions[i]);
                ngxFound++;
                Msg("[Vulkan] NGX extension enabled: %s", g_NgxDeviceExtensions[i]);
            } else {
                Msg("[Vulkan] NGX extension not available: %s", g_NgxDeviceExtensions[i]);
            }
        }
        g_bNgxExtensionsEnabled = (ngxFound == g_NgxDeviceExtensionCount);
        Msg("[Vulkan] NGX extensions: %u/%u available", ngxFound, g_NgxDeviceExtensionCount);

        // Check optional RT extensions (ray query + acceleration structure)
        u32 rtFound = 0;
        for (u32 i = 0; i < g_RtDeviceExtensionCount; i++) {
            bool found = false;
            for (const auto& ext : availableExts) {
                if (strcmp(g_RtDeviceExtensions[i], ext.extensionName) == 0) {
                    found = true;
                    break;
                }
            }
            if (found) {
                rtFound++;
                Msg("[Vulkan] RT extension available: %s", g_RtDeviceExtensions[i]);
            } else {
                Msg("[Vulkan] RT extension not available: %s", g_RtDeviceExtensions[i]);
            }
        }

        if (rtFound == g_RtDeviceExtensionCount) {
            // All RT extensions present — query feature support
            VkPhysicalDeviceRayQueryFeaturesKHR rqFeatures = {};
            rqFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
            VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures = {};
            asFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
            asFeatures.pNext = &rqFeatures;
            VkPhysicalDeviceFeatures2 rtQuery = {};
            rtQuery.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            rtQuery.pNext = &asFeatures;
            vkGetPhysicalDeviceFeatures2(m_PhysicalDevice, &rtQuery);

            if (rqFeatures.rayQuery && asFeatures.accelerationStructure) {
                for (u32 i = 0; i < g_RtDeviceExtensionCount; i++)
                    enabledExtensions.push_back(g_RtDeviceExtensions[i]);
                m_bRayQuerySupported = true;
                Msg("[Vulkan] Ray Query + Acceleration Structure: ENABLED");
            } else {
                Msg("[Vulkan] RT extensions present but features not supported (rayQuery=%d, accelStruct=%d)",
                    rqFeatures.rayQuery, asFeatures.accelerationStructure);
            }
        } else {
            Msg("[Vulkan] RT extensions: %u/%u available (need all 3 for RTGI)", rtFound, g_RtDeviceExtensionCount);
        }

        // Optional: VK_NV_device_diagnostic_checkpoints — GPU progress markers that
        // survive a device loss. Near-zero cost; enables post-mortem "which pass
        // hung" logging on DEVICE_LOST (the level-transition GPU-hang hunt).
        for (const auto& ext : availableExts) {
            if (strcmp(VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME, ext.extensionName) == 0) {
                enabledExtensions.push_back(VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME);
                m_bCheckpointsSupported = true;
                Msg("[Vulkan] NV diagnostic checkpoints: ENABLED");
                break;
            }
        }
        if (!m_bCheckpointsSupported)
            Msg("[Vulkan] NV diagnostic checkpoints not available");

        // Optional: VK_KHR_fragment_shading_rate (attachment-based VRS) — coarse-shade
        // distant/peripheral pixels via a shading-rate image to cut forward fragment cost.
        {
            bool fsrAvail = false;
            for (const auto& ext : availableExts)
                if (strcmp(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME, ext.extensionName) == 0) { fsrAvail = true; break; }
            if (fsrAvail) {
                VkPhysicalDeviceFragmentShadingRateFeaturesKHR fsrFeat = {};
                fsrFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR;
                VkPhysicalDeviceFeatures2 fsrQuery = {};
                fsrQuery.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
                fsrQuery.pNext = &fsrFeat;
                vkGetPhysicalDeviceFeatures2(m_PhysicalDevice, &fsrQuery);
                if (fsrFeat.attachmentFragmentShadingRate) {
                    enabledExtensions.push_back(VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME);
                    m_bVRSSupported = true;
                    VkPhysicalDeviceFragmentShadingRatePropertiesKHR fsrProps = {};
                    fsrProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_PROPERTIES_KHR;
                    VkPhysicalDeviceProperties2 props2 = {};
                    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
                    props2.pNext = &fsrProps;
                    vkGetPhysicalDeviceProperties2(m_PhysicalDevice, &props2);
                    // Use the coarsest allowed tile (smallest SRI). NVIDIA = 16x16.
                    m_VRSTexelSize = fsrProps.maxFragmentShadingRateAttachmentTexelSize;
                    // Per-pipeline coarse rate (separate feature) — lets us 2x2-shade the
                    // low-freq light cones without an SRI image.
                    m_bVRSPipelineSupported = fsrFeat.pipelineFragmentShadingRate != VK_FALSE;
                    Msg("[Vulkan] VRS (attachment fragment shading rate): ENABLED (tile %ux%u, pipeline-rate %s)",
                        m_VRSTexelSize.width, m_VRSTexelSize.height,
                        m_bVRSPipelineSupported ? "yes" : "no");
                } else {
                    Msg("[Vulkan] VRS extension present but attachmentFragmentShadingRate unsupported");
                }
            } else {
                Msg("[Vulkan] VRS extension not available");
            }
        }

        // Optional: VK_EXT_memory_budget — lets VMA read the driver's LIVE VRAM
        // budget/usage (what the OS + other processes actually left free) instead of
        // guessing from static heap sizes. The texture streamer sizes its residency
        // budget from this; without it we fall back to the heap size. No feature
        // struct — enabling the extension is enough.
        for (const auto& ext : availableExts) {
            if (strcmp(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME, ext.extensionName) == 0) {
                enabledExtensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
                m_bMemoryBudgetSupported = true;
                Msg("[Vulkan] VK_EXT_memory_budget: ENABLED (live VRAM budget)");
                break;
            }
        }
        if (!m_bMemoryBudgetSupported)
            Msg("[Vulkan] VK_EXT_memory_budget not available (static heap-size budget)");

        // Optional: VK_EXT_memory_priority — per-allocation eviction priority. We tag
        // streamable world textures LOW so the driver demotes them to system RAM under
        // VRAM pressure before evicting render targets / VSM / geometry (HIGH). Turns a
        // hard OOM into a graceful slowdown. Needs the feature bit chained below.
        {
            bool prioAvail = false;
            for (const auto& ext : availableExts)
                if (strcmp(VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME, ext.extensionName) == 0) { prioAvail = true; break; }
            if (prioAvail) {
                VkPhysicalDeviceMemoryPriorityFeaturesEXT prioFeat = {};
                prioFeat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT;
                VkPhysicalDeviceFeatures2 prioQuery = {};
                prioQuery.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
                prioQuery.pNext = &prioFeat;
                vkGetPhysicalDeviceFeatures2(m_PhysicalDevice, &prioQuery);
                if (prioFeat.memoryPriority) {
                    enabledExtensions.push_back(VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME);
                    m_bMemoryPrioritySupported = true;
                    Msg("[Vulkan] VK_EXT_memory_priority: ENABLED (fail-soft texture eviction)");
                } else {
                    Msg("[Vulkan] VK_EXT_memory_priority present but feature unsupported");
                }
            } else {
                Msg("[Vulkan] VK_EXT_memory_priority not available");
            }
        }
    }

    // Chain RT feature structs if supported
    VkPhysicalDeviceRayQueryFeaturesKHR enableRayQuery = {};
    enableRayQuery.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    enableRayQuery.rayQuery = VK_TRUE;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR enableAccelStruct = {};
    enableAccelStruct.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    enableAccelStruct.accelerationStructure = VK_TRUE;
    if (m_bRayQuerySupported) {
        enableAccelStruct.pNext = &enableRayQuery;
        features13.pNext = &enableAccelStruct;
    }

    // Chain VRS feature (prepend, preserving whatever is already on features13 — RT or null).
    VkPhysicalDeviceFragmentShadingRateFeaturesKHR enableFsr = {};
    enableFsr.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR;
    enableFsr.attachmentFragmentShadingRate = VK_TRUE;
    enableFsr.pipelineFragmentShadingRate   = m_bVRSPipelineSupported ? VK_TRUE : VK_FALSE;
    if (m_bVRSSupported) {
        enableFsr.pNext = const_cast<void*>(features13.pNext);
        features13.pNext = &enableFsr;
    }

    // Chain memory-priority feature (prepend, preserving the existing pNext chain).
    VkPhysicalDeviceMemoryPriorityFeaturesEXT enableMemPrio = {};
    enableMemPrio.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PRIORITY_FEATURES_EXT;
    enableMemPrio.memoryPriority = VK_TRUE;
    if (m_bMemoryPrioritySupported) {
        enableMemPrio.pNext = const_cast<void*>(features13.pNext);
        features13.pNext = &enableMemPrio;
    }

    // Device create info
    VkDeviceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pNext = &deviceFeatures;
    createInfo.queueCreateInfoCount = static_cast<u32>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.enabledExtensionCount = static_cast<u32>(enabledExtensions.size());
    createInfo.ppEnabledExtensionNames = enabledExtensions.data();

    // Device-level layers are deprecated and ignored by modern loaders — validation
    // is controlled entirely at the instance (see VK_CreateInstance, gated on the
    // -vk_validation flag). Leave these zero so device creation never pulls the
    // (expensive) validation layer regardless of the instance choice.
    createInfo.enabledLayerCount = 0;
    createInfo.ppEnabledLayerNames = nullptr;

    // Создаём device
    VkResult result = vkCreateDevice(m_PhysicalDevice, &createInfo, nullptr, &m_Device);
    if (result != VK_SUCCESS) {
        Msg("!Failed to create logical device. Error: %d", result);
        return false;
    }

    Msg("[Vulkan] Logical device created");

    // Получаем queue handles. When families coincide, vkGetDeviceQueue(fam, 0)
    // legally returns the SAME VkQueue handle, so compute/transfer alias graphics.
    vkGetDeviceQueue(m_Device, m_GraphicsFamily, 0, &m_GraphicsQueue);
    vkGetDeviceQueue(m_Device, m_PresentFamily,  0, &m_PresentQueue);
    vkGetDeviceQueue(m_Device, m_ComputeFamily,  0, &m_ComputeQueue);
    vkGetDeviceQueue(m_Device, m_TransferFamily, 0, &m_TransferQueue);

    Msg("[Vulkan] Queue handles obtained (graphics/present/compute/transfer)");

    return true;
}

// Создание VMA
bool CVulkanHW::CreateVMA()
{
    // Явно указываем Vulkan функции для VMA
    VmaVulkanFunctions vulkanFunctions = {};
    vulkanFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vulkanFunctions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
    vulkanFunctions.vkGetPhysicalDeviceProperties = vkGetPhysicalDeviceProperties;
    vulkanFunctions.vkGetPhysicalDeviceMemoryProperties = vkGetPhysicalDeviceMemoryProperties;
    vulkanFunctions.vkAllocateMemory = vkAllocateMemory;
    vulkanFunctions.vkFreeMemory = vkFreeMemory;
    vulkanFunctions.vkMapMemory = vkMapMemory;
    vulkanFunctions.vkUnmapMemory = vkUnmapMemory;
    vulkanFunctions.vkFlushMappedMemoryRanges = vkFlushMappedMemoryRanges;
    vulkanFunctions.vkInvalidateMappedMemoryRanges = vkInvalidateMappedMemoryRanges;
    vulkanFunctions.vkBindBufferMemory = vkBindBufferMemory;
    vulkanFunctions.vkBindImageMemory = vkBindImageMemory;
    vulkanFunctions.vkGetBufferMemoryRequirements = vkGetBufferMemoryRequirements;
    vulkanFunctions.vkGetImageMemoryRequirements = vkGetImageMemoryRequirements;
    vulkanFunctions.vkCreateBuffer = vkCreateBuffer;
    vulkanFunctions.vkDestroyBuffer = vkDestroyBuffer;
    vulkanFunctions.vkCreateImage = vkCreateImage;
    vulkanFunctions.vkDestroyImage = vkDestroyImage;
    vulkanFunctions.vkCmdCopyBuffer = vkCmdCopyBuffer;

    VmaAllocatorCreateInfo allocatorInfo = {};
    allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    // Let VMA consume the optional streaming extensions we enabled at device
    // creation. BUDGET → vmaGetHeapBudgets returns the driver's live figures;
    // PRIORITY → VmaAllocationCreateInfo::priority is forwarded to the allocation.
    // Both flags are only legal when the matching extension is actually enabled.
    if (m_bMemoryBudgetSupported)   allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
    if (m_bMemoryPrioritySupported) allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT;
    allocatorInfo.physicalDevice = m_PhysicalDevice;
    allocatorInfo.device = m_Device;
    allocatorInfo.instance = m_Instance;
    allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_3;
    allocatorInfo.pVulkanFunctions = &vulkanFunctions;
    // 64 MB heap blocks instead of VMA's 256 MB default. A block returns to the OS
    // only when its LAST allocation dies; with 256 MB granularity the load-time
    // churn (bake scratch, staging, swapped textures) left blocks pinned by a few
    // survivors — measured 1449 MB of "vma-slack" on Pripyat (17-07 23:30 log), the
    // bulk of the overcommit that starved NGX/DLSS. Finer blocks bound each pin to
    // 64 MB and empty out far more often. Allocation-count cost is trivial
    // (~120 blocks for the whole card + dedicated images, limit 4096).
    allocatorInfo.preferredLargeHeapBlockSize = 64ull << 20;

    VkResult result = vmaCreateAllocator(&allocatorInfo, &m_Allocator);

    if (result != VK_SUCCESS) {
        return false;
    }

    return true;
}

// Sum live usage/budget across all DEVICE_LOCAL heaps. With VK_EXT_memory_budget
// active these are the driver's real figures; otherwise VMA reports the static heap
// size as budget and its own allocation total as usage (still useful, just not
// OS-aware). Called by the texture streamer + ResourcesGetMemoryUsage.
void CVulkanHW::GetVramBudget(VkDeviceSize& outUsage, VkDeviceSize& outBudget) const
{
    outUsage = 0;
    outBudget = 0;
    if (m_Allocator == VK_NULL_HANDLE)
        return;

    const VkPhysicalDeviceMemoryProperties* memProps = nullptr;
    vmaGetMemoryProperties(m_Allocator, &memProps);
    if (!memProps)
        return;

    VmaBudget budgets[VK_MAX_MEMORY_HEAPS] = {};
    vmaGetHeapBudgets(m_Allocator, budgets);

    for (u32 h = 0; h < memProps->memoryHeapCount; ++h) {
        if (memProps->memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            outUsage  += budgets[h].usage;
            outBudget += budgets[h].budget;
        }
    }
}

// Helper struct for finding unique video modes
struct _uniq_mode_vulkan
{
    LPCSTR _val;
    _uniq_mode_vulkan(LPCSTR v) : _val(v) {}
    bool operator()(LPCSTR _other) { return !_stricmp(_val, _other); }
};

// Free vid_mode_token list (called on shutdown)
static void free_vid_mode_list_vulkan()
{
    if (vid_mode_token == NULL) return;

    for (int i = 0; vid_mode_token[i].name; i++)
    {
        xr_free(vid_mode_token[i].name);
    }
    xr_free(vid_mode_token);
    vid_mode_token = NULL;
    Msg("[Vulkan] Video mode list freed");
}

// Fill vid_mode_token with available display modes
static void fill_vid_mode_list_vulkan()
{
    if (vid_mode_token != NULL) return;

    xr_vector<LPCSTR> _tmp;
    DEVMODEW dm;
    ZeroMemory(&dm, sizeof(dm));
    dm.dmSize = sizeof(dm);

    // Enumerate all display modes
    for (DWORD iModeNum = 0; EnumDisplaySettingsW(NULL, iModeNum, &dm); iModeNum++)
    {
        // Filter out small resolutions and non-32bit modes
        if (dm.dmPelsWidth < 800)
            continue;
        if (dm.dmBitsPerPel < 32)
            continue;

        string32 str;
        xr_sprintf(str, sizeof(str), "%dx%d", dm.dmPelsWidth, dm.dmPelsHeight);

        // Check if this mode is already in the list
        if (_tmp.end() != std::find_if(_tmp.begin(), _tmp.end(), _uniq_mode_vulkan(str)))
            continue;

        _tmp.push_back(NULL);
        _tmp.back() = xr_strdup(str);
    }

    // Sort by resolution (width first, then height)
    std::sort(_tmp.begin(), _tmp.end(), [](LPCSTR a, LPCSTR b) {
        int wa, ha, wb, hb;
        sscanf(a, "%dx%d", &wa, &ha);
        sscanf(b, "%dx%d", &wb, &hb);
        if (wa != wb) return wa < wb;
        return ha < hb;
    });

    u32 _cnt = _tmp.size() + 1;
    vid_mode_token = xr_alloc<xr_token>(_cnt);
    vid_mode_token[_cnt - 1].id = -1;
    vid_mode_token[_cnt - 1].name = NULL;

    Msg("[Vulkan] Available video modes[%d]:", _tmp.size());
    for (u32 i = 0; i < _tmp.size(); ++i)
    {
        vid_mode_token[i].id = i;
        vid_mode_token[i].name = _tmp[i];
        Msg("  [%d] %s", i, _tmp[i]);
    }
}

// Главный метод создания device (аналог DX11)
// Returns true on success, false on error
bool CVulkanHW::CreateDevice(HWND hWnd)
{
    Msg("=================================================================");
    Msg("[Vulkan] Initializing Vulkan renderer...");
    Msg("=================================================================");

    m_hWnd = hWnd;

    // ========================================================================
    // Step 0: NVIDIA Streamline — MUST init before ANY Vulkan call (the linked
    // sl.interposer proxies vkCreateInstance/Device so SL can hook DLSS/Reflex/FG).
    // Non-fatal: if slInit fails, SL stays off and the raw-NGX DLSS path is used.
    // ========================================================================
    VK::SL::Init();

    // ========================================================================
    // Step 1: Create Vulkan instance
    // ========================================================================
    Msg("[Vulkan] Step 1/11: Creating Vulkan instance...");
    if (!VK_CreateInstance(&m_Instance)) {
        Msg("![Vulkan] FAILED: Step 1 - Failed to create Vulkan instance");
        Msg("![Vulkan] ");
        Msg("![Vulkan] Possible reasons:");
        Msg("![Vulkan] - Vulkan SDK not installed");
        Msg("![Vulkan] - Outdated GPU drivers");
        Msg("![Vulkan] - Vulkan not supported by system");
        Msg("![Vulkan] ");
        Msg("![Vulkan] Solutions:");
        Msg("![Vulkan] 1. Install Vulkan SDK from https://vulkan.lunarg.com/");
        Msg("![Vulkan] 2. Update GPU drivers to latest version");
        Msg("![Vulkan] 3. Use DirectX renderer with -dx11 flag");
        return false;
    }
    Msg("[Vulkan] Step 1/11: SUCCESS - Vulkan instance created");

    // 2. Debug messenger (validation error/warning output)
    VK_SetupDebugMessenger(m_Instance, &m_DebugMessenger);

    // ========================================================================
    // Step 2: Create window surface
    // ========================================================================
    Msg("[Vulkan] Step 2/11: Creating window surface...");
    VkWin32SurfaceCreateInfoKHR surfaceInfo = {};
    surfaceInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
    surfaceInfo.hinstance = GetModuleHandle(nullptr);
    surfaceInfo.hwnd = hWnd;

    VkResult result = vkCreateWin32SurfaceKHR(m_Instance, &surfaceInfo, nullptr, &m_Surface);
    if (result != VK_SUCCESS) {
        Msg("![Vulkan] FAILED: Step 2 - Failed to create window surface (error: %d)", result);
        Msg("![Vulkan] Window handle may be invalid");
        return false;
    }
    Msg("[Vulkan] Step 2/11: SUCCESS - Window surface created");

    // ========================================================================
    // Step 3: Select physical device (GPU)
    // ========================================================================
    Msg("[Vulkan] Step 3/11: Selecting physical device (GPU)...");
    m_PhysicalDevice = VK_SelectPhysicalDevice(m_Instance, &Caps);
    if (m_PhysicalDevice == VK_NULL_HANDLE) {
        Msg("![Vulkan] FAILED: Step 3 - No compatible GPU found");
        Msg("![Vulkan] ");
        Msg("![Vulkan] Possible reasons:");
        Msg("![Vulkan] - No Vulkan-capable GPU detected");
        Msg("![Vulkan] - GPU drivers too old");
        Msg("![Vulkan] - Integrated GPU disabled in BIOS");
        Msg("![Vulkan] ");
        Msg("![Vulkan] Solutions:");
        Msg("![Vulkan] 1. Update GPU drivers to latest version");
        Msg("![Vulkan] 2. Check GPU is enabled in Device Manager");
        Msg("![Vulkan] 3. Use DirectX renderer with -dx11 flag");
        return false;
    }

    // Streamline: probe per-feature support now that the adapter is known.
    VK::SL::OnDeviceReady(m_PhysicalDevice);

    // Display GPU information
    Msg("[Vulkan] Step 3/11: SUCCESS - Physical device selected");
    Msg("[Vulkan] ");
    Msg("[Vulkan] GPU Information:");
    Msg("[Vulkan] ----------------------------------------");
    Msg("[Vulkan] Device Name: %s", Caps.deviceName);
    Msg("[Vulkan] Vulkan API: %d.%d.%d",
        VK_VERSION_MAJOR(Caps.apiVersion),
        VK_VERSION_MINOR(Caps.apiVersion),
        VK_VERSION_PATCH(Caps.apiVersion));
    Msg("[Vulkan] Driver Version: %d.%d.%d",
        VK_VERSION_MAJOR(Caps.driverVersion),
        VK_VERSION_MINOR(Caps.driverVersion),
        VK_VERSION_PATCH(Caps.driverVersion));
    Msg("[Vulkan] ----------------------------------------");
    Msg("[Vulkan] ");

    // ========================================================================
    // Step 4: Check minimum requirements
    // ========================================================================
    Msg("[Vulkan] Step 4/11: Checking GPU capabilities...");
    if (!Caps.CheckMinimumRequirements()) {
        Msg("![Vulkan] FAILED: Step 4 - GPU doesn't meet minimum requirements");
        Msg("![Vulkan] ");
        Msg("![Vulkan] Requirements:");
        Msg("![Vulkan] - Vulkan 1.2 or 1.3");
        Msg("![Vulkan] - VK_KHR_dynamic_rendering extension");
        Msg("![Vulkan] - VK_KHR_synchronization2 extension");
        Msg("![Vulkan] ");
        Msg("![Vulkan] Your GPU:");
        Msg("![Vulkan] - Vulkan API: %d.%d.%d",
            VK_VERSION_MAJOR(Caps.apiVersion),
            VK_VERSION_MINOR(Caps.apiVersion),
            VK_VERSION_PATCH(Caps.apiVersion));
        Msg("![Vulkan] ");
        Msg("![Vulkan] Solutions:");
        Msg("![Vulkan] 1. Update GPU drivers to latest version");
        Msg("![Vulkan] 2. For NVIDIA: Driver 515+ (GTX 1000 series+)");
        Msg("![Vulkan] 3. For AMD: Driver 21.10.1+ (RX 5000 series+)");
        Msg("![Vulkan] 4. For Intel: Driver 30.0.101.1191+ (Arc A-series)");
        Msg("![Vulkan] 5. Use DirectX renderer with -dx11 flag");
        return false;
    }
    Msg("[Vulkan] Step 4/11: SUCCESS - GPU meets all requirements");

    // ========================================================================
    // Step 5: Find queue families
    // ========================================================================
    Msg("[Vulkan] Step 5/11: Finding queue families...");
    if (!FindQueueFamilies()) {
        Msg("![Vulkan] FAILED: Step 5 - Failed to find suitable queue families");
        Msg("![Vulkan] GPU doesn't support required graphics/present queues");
        return false;
    }
    Msg("[Vulkan] Step 5/11: SUCCESS - Queue families found (Graphics: %u, Present: %u)",
        m_GraphicsFamily, m_PresentFamily);

    // ========================================================================
    // Step 6: Create logical device
    // ========================================================================
    Msg("[Vulkan] Step 6/11: Creating logical device...");
    if (!CreateLogicalDevice()) {
        Msg("![Vulkan] FAILED: Step 6 - Failed to create logical device");
        Msg("![Vulkan] This may indicate driver issues or unsupported features");
        return false;
    }
    Msg("[Vulkan] Step 6/11: SUCCESS - Logical device created");

    // Streamline: sl.dlss_g learns the game window (frame-pacing refresh-rate
    // queries) from its vkCreateWin32SurfaceKHR after-hook, but SL plugins only
    // initialise INSIDE vkCreateDevice — the Step-2 surface predates them, so
    // the hook never fired and dlss_g polls a garbage HWND every frame
    // ("Window handle ... is not a valid window" log spam; FG pacing would be
    // blind). Recreate the surface now that the hooks are live; nothing holds
    // the old one yet (the swapchain comes later).
    if (VK::SL::Inited()) {
        vkDestroySurfaceKHR(m_Instance, m_Surface, nullptr);
        m_Surface = VK_NULL_HANDLE;
        VkWin32SurfaceCreateInfoKHR slSurfaceInfo = {};
        slSurfaceInfo.sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR;
        slSurfaceInfo.hinstance = GetModuleHandle(nullptr);
        slSurfaceInfo.hwnd = hWnd;
        result = vkCreateWin32SurfaceKHR(m_Instance, &slSurfaceInfo, nullptr, &m_Surface);
        if (result != VK_SUCCESS) {
            Msg("![Vulkan] FAILED: Streamline surface recreate (error: %d)", result);
            return false;
        }
        Msg("[Vulkan] Streamline: window surface recreated post-device (SL surface hook now live)");
    }

    // ========================================================================
    // Step 7: Create VMA (Vulkan Memory Allocator)
    // ========================================================================
    Msg("[Vulkan] Step 7/11: Creating VMA allocator...");
    if (!CreateVMA()) {
        Msg("![Vulkan] FAILED: Step 7 - Failed to create VMA allocator");
        Msg("![Vulkan] Memory allocation system initialization failed");
        return false;
    }
    Msg("[Vulkan] Step 7/11: SUCCESS - VMA allocator created");

    // ========================================================================
    // Step 8: Create command pool for transfer operations
    // ========================================================================
    Msg("[Vulkan] Step 8/11: Creating transfer command pool...");
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = m_GraphicsFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;  // Short-lived commands

    result = vkCreateCommandPool(m_Device, &poolInfo, nullptr, &m_TransferCommandPool);
    if (result != VK_SUCCESS) {
        Msg("![Vulkan] FAILED: Step 8 - Failed to create transfer command pool (error: %d)", result);
        return false;
    }
    Msg("[Vulkan] Step 8/11: SUCCESS - Transfer command pool created");

    // ========================================================================
    // Step 9 + 10: Geometry + Lighting subsystems — DEFERRED to Stage 2.b.
    // Their .cpp files transitively pull shared xrRender (rvk.h, dsgraph,
    // light.h) which is R4-D3D-specific. Bringing them online needs the
    // framegraph + a CRender that owns dsgraph state.
    // For Stage 2.a (clear-color frame) the swapchain is enough.
    // ========================================================================
#if 0  // STAGE_2B_GEOMETRY_LIGHTING
    Msg("[Vulkan] Step 9/11: Creating geometry system...");
    g_VulkanGeometry = xr_new<VK::CVulkanGeometry>();
    g_VulkanGeometry->Create();
    Msg("[Vulkan] Step 9/11: SUCCESS - Geometry system created");

    Msg("[Vulkan] Step 10/11: Creating lighting manager...");
    g_VulkanLighting = xr_new<VK::CVulkanLighting>();
    g_VulkanLighting->Create();
    Msg("[Vulkan] Step 10/11: SUCCESS - Lighting manager created");
#else
    Msg("[Vulkan] Step 9+10: SKIPPED (Stage 2.a — geometry/lighting deferred)");
#endif

    // ========================================================================
    // Step 11: Fill video mode list for options menu
    // ========================================================================
    Msg("[Vulkan] Step 11/11: Filling video mode list...");
    fill_vid_mode_list_vulkan();
    Msg("[Vulkan] Step 11/11: SUCCESS - Video mode list filled");

    Msg("=================================================================");
    Msg("[Vulkan] Initialization completed successfully!");
    Msg("[Vulkan] All systems ready");
    Msg("=================================================================");

    // Build provenance / fingerprint banner (decoded from vk_authorship).
    ogsr::sig::Register();

    return true;
}

// Уничтожение device
void CVulkanHW::DestroyDevice()
{
    if (m_Device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_Device);
    }

    // Free video mode list
    free_vid_mode_list_vulkan();

    // Stage 2.a: paired with the create-side #if 0 above. Re-enable for 2.b.
#if 0  // STAGE_2B_GEOMETRY_LIGHTING
    if (g_VulkanLighting) { xr_delete(g_VulkanLighting); g_VulkanLighting = nullptr; }
    if (g_VulkanGeometry) { xr_delete(g_VulkanGeometry); g_VulkanGeometry = nullptr; }
#endif

    // Уничтожаем command pool
    if (m_TransferCommandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(m_Device, m_TransferCommandPool, nullptr);
        m_TransferCommandPool = VK_NULL_HANDLE;
        Msg("[Vulkan] Transfer command pool destroyed");
    }

    // Уничтожаем VMA
    if (m_Allocator != VK_NULL_HANDLE) {
        VK::Vram::DestroySmallPools(m_Allocator);   // custom pools must die first
        vmaDestroyAllocator(m_Allocator);
        m_Allocator = VK_NULL_HANDLE;
        Msg("[Vulkan] VMA allocator destroyed");
    }

    // Уничтожаем device
    if (m_Device != VK_NULL_HANDLE) {
        vkDestroyDevice(m_Device, nullptr);
        m_Device = VK_NULL_HANDLE;
        Msg("[Vulkan] Logical device destroyed");
    }

    // Уничтожаем surface
    if (m_Surface != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(m_Instance, m_Surface, nullptr);
        m_Surface = VK_NULL_HANDLE;
        Msg("[Vulkan] Surface destroyed");
    }

    // Destroy debug messenger
    if (m_DebugMessenger != VK_NULL_HANDLE) {
        VK_DestroyDebugMessenger(m_Instance, m_DebugMessenger);
        m_DebugMessenger = VK_NULL_HANDLE;
    }

    // Уничтожаем instance
    if (m_Instance != VK_NULL_HANDLE) {
        VK_DestroyInstance(m_Instance);
        m_Instance = VK_NULL_HANDLE;
    }

    Msg("[Vulkan] Device destruction complete");
}

// Reset device (при resize окна и т.д.)
void CVulkanHW::Reset(HWND hWnd)
{
    Msg("[Vulkan] Reset device");

    // Ждём завершения всех операций
    if (m_Device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_Device);
    }

    // TODO: Пересоздать swapchain и render targets
}

// App activation
void CVulkanHW::OnAppActivate()
{
    Msg("[Vulkan] App activated");
}

// App deactivation
void CVulkanHW::OnAppDeactivate()
{
    Msg("[Vulkan] App deactivated");
}

// ============================================================================
// Single-time command helpers (for buffer uploads, layout transitions, etc.)
// ============================================================================

// These now delegate to CommandManager's immediate (one-shot) path instead of
// hand-rolling a second submit-and-wait. That path is device-lost guarded, waits
// on a fence (not a full vkQueueWaitIdle of the whole graphics queue), and uses a
// graphics-family pool — the old code allocated from m_TransferCommandPool yet
// submitted to m_GraphicsQueue, a queue-family mismatch on GPUs with a dedicated
// transfer queue. All callers are load-time + main-thread, so reusing the single
// shared immediate buffer is not reentrant here.
VkCommandBuffer CVulkanHW::BeginSingleTimeCommands()
{
    return CommandManager.BeginImmediate();
}

void CVulkanHW::EndSingleTimeCommands(VkCommandBuffer commandBuffer)
{
    CommandManager.EndAndSubmitImmediate(commandBuffer);
}
