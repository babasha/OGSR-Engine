// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
// Licensed under the same terms as X-Ray Engine (see root License.txt)

#include "stdafx.h"
#include "vk_core.h"
#include "HWCaps_Vulkan.h"
#include <vector>

// Debug messenger callback (always compiled — validation is essential for Vulkan dev)
static VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData)
{
    const char* severity = "INFO";
    if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        severity = "ERROR";
    } else if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        severity = "WARNING";
    }

    // Log message ID for easy filtering (e.g. VUID-vkCmdDraw-...)
    if (pCallbackData->pMessageIdName)
        Msg("[VK-VAL %s] [%s] %s", severity, pCallbackData->pMessageIdName, pCallbackData->pMessage);
    else
        Msg("[VK-VAL %s] %s", severity, pCallbackData->pMessage);

    // FlushLog() existed in monolith xrCore; OGSR removed it. Validation
    // messages are still queued by Msg() — the engine flushes log on its own
    // schedule (file flushed on shutdown / crash dump).
    return VK_FALSE;
}

bool VK_CreateInstance(VkInstance* outInstance)
{
    VERIFY(outInstance);

    // Application info
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "X-Ray Engine";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "X-Ray";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    // Instance create info
    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = g_InstanceExtensionCount;
    createInfo.ppEnabledExtensionNames = g_InstanceExtensions;

    // Check if validation layer is actually installed on this system
    u32 layerCount = 0;
    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    std::vector<VkLayerProperties> availableLayers(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    bool validationAvailable = false;
    for (const auto& layer : availableLayers) {
        if (strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0) {
            validationAvailable = true;
            break;
        }
    }

    // Validation is OFF by default — the KHRONOS layer realistically costs 2-10x
    // (NOT the "~5%" the old message claimed), and every error also runs through
    // the debug callback → Msg → synchronous log write. Opt in with the
    // `-vk_validation` command-line flag when debugging barriers/descriptors/layouts.
    const bool wantValidation = Core.Params && strstr(Core.Params, "-vk_validation");
    validationAvailable = validationAvailable && wantValidation;

    // Debug messenger for instance creation/destruction messages
    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo = {};
    debugCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    debugCreateInfo.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debugCreateInfo.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debugCreateInfo.pfnUserCallback = DebugCallback;

    if (validationAvailable) {
        createInfo.enabledLayerCount = g_ValidationLayerCount;
        createInfo.ppEnabledLayerNames = g_ValidationLayers;
        createInfo.pNext = &debugCreateInfo;
        Msg("[Vulkan] Validation layers ENABLED (-vk_validation) — expect a large FPS hit");
        Msg("[Vulkan] Barrier/descriptor/layout errors will appear as [VK-VAL] in log");
    } else {
        createInfo.enabledLayerCount = 0;
        createInfo.pNext = nullptr;
        Msg("[Vulkan] Validation layers OFF (pass -vk_validation to enable diagnostics)");
    }

    // Создаём instance
    VkResult result = vkCreateInstance(&createInfo, nullptr, outInstance);

    if (result != VK_SUCCESS) {
        Msg("!Failed to create Vulkan instance. Error code: %d", result);
        return false;
    }

    Msg("[Vulkan] Instance created successfully");
    return true;
}

void VK_DestroyInstance(VkInstance instance)
{
    if (instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
        Msg("[Vulkan] Instance destroyed");
    }
}

VkResult VK_SetupDebugMessenger(VkInstance instance, VkDebugUtilsMessengerEXT* outMessenger)
{
    VERIFY(outMessenger);

    VkDebugUtilsMessengerCreateInfoEXT createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = DebugCallback;

    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
        instance, "vkCreateDebugUtilsMessengerEXT");

    if (func != nullptr) {
        VkResult result = func(instance, &createInfo, nullptr, outMessenger);
        if (result == VK_SUCCESS) {
            Msg("[Vulkan] Debug messenger created (validation errors will appear in log)");
        }
        return result;
    } else {
        Msg("[Vulkan] vkCreateDebugUtilsMessengerEXT not available");
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void VK_DestroyDebugMessenger(VkInstance instance, VkDebugUtilsMessengerEXT messenger)
{
    if (messenger != VK_NULL_HANDLE) {
        auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(
            instance, "vkDestroyDebugUtilsMessengerEXT");

        if (func != nullptr) {
            func(instance, messenger, nullptr);
            Msg("[Vulkan] Debug messenger destroyed");
        }
    }
}

// Проверка поддержки расширений устройства
bool VK_CheckDeviceExtensionSupport(VkPhysicalDevice device)
{
    u32 extensionCount = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

    // Проверяем все требуемые расширения
    for (u32 i = 0; i < g_DeviceExtensionCount; i++) {
        bool found = false;
        for (const auto& extension : availableExtensions) {
            if (strcmp(g_DeviceExtensions[i], extension.extensionName) == 0) {
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }

    return true;
}

// Выбор физического устройства
VkPhysicalDevice VK_SelectPhysicalDevice(VkInstance instance, VulkanCaps* outCaps)
{
    VERIFY(outCaps);

    u32 deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    if (deviceCount == 0) {
        Msg("!Failed to find GPUs with Vulkan support");
        return VK_NULL_HANDLE;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    // Система оценки устройств
    struct DeviceScore {
        VkPhysicalDevice device;
        int score;
        VkPhysicalDeviceProperties props;
        VkPhysicalDeviceVulkan13Features features13;
    };
    std::vector<DeviceScore> scores;

    for (auto device : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(device, &props);

        // Проверка версии API
        if (props.apiVersion < VK_API_VERSION_1_3) {
            continue;
        }

        // Проверка расширений
        if (!VK_CheckDeviceExtensionSupport(device)) {
            continue;
        }

        // Проверка Vulkan 1.3 features
        VkPhysicalDeviceVulkan13Features features13 = {};
        features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;

        VkPhysicalDeviceFeatures2 features2 = {};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = &features13;

        vkGetPhysicalDeviceFeatures2(device, &features2);

        // Проверяем обязательные features
        if (!features13.dynamicRendering || !features13.synchronization2) {
            continue;
        }

        // Подсчет очков
        int score = 0;

        // Discrete GPU +1000
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            score += 1000;
        }

        // Vendor preference
        switch (props.vendorID) {
        case 0x10DE:  // NVIDIA
            score += 100;
            Msg("[Vulkan] Found NVIDIA: %s", props.deviceName);
            break;
        case 0x1002:  // AMD
            score += 90;
            Msg("[Vulkan] Found AMD: %s", props.deviceName);
            break;
        case 0x8086:  // Intel
            score += 50;
            Msg("[Vulkan] Found Intel: %s", props.deviceName);
            break;
        default:
            Msg("[Vulkan] Found GPU: %s (Vendor: 0x%X)", props.deviceName, props.vendorID);
            break;
        }

        // VRAM bonus
        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(device, &memProps);
        VkDeviceSize totalVRAM = 0;
        for (u32 i = 0; i < memProps.memoryHeapCount; i++) {
            if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
                totalVRAM += memProps.memoryHeaps[i].size;
            }
        }
        score += static_cast<int>(totalVRAM / (1024ULL * 1024 * 1024));  // +1 per GB

        scores.push_back({ device, score, props, features13 });
    }

    if (scores.empty()) {
        Msg("!No suitable Vulkan GPU found");
        return VK_NULL_HANDLE;
    }

    // Сортировка по score
    std::sort(scores.begin(), scores.end(),
              [](const auto& a, const auto& b) { return a.score > b.score; });

    // Выбираем лучшее устройство
    DeviceScore& best = scores[0];

    // ========================================================================
    // Query full features for Update()
    // ========================================================================
    VkPhysicalDeviceFeatures baseFeatures;
    vkGetPhysicalDeviceFeatures(best.device, &baseFeatures);

    // ========================================================================
    // Fill capabilities using Update()
    // ========================================================================
    outCaps->Update(best.device, best.props, baseFeatures, best.features13);

    // ========================================================================
    // Calculate VRAM (not done in Update())
    // ========================================================================
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(best.device, &memProps);
    outCaps->totalDeviceMemory = 0;  // Reset before counting
    for (u32 i = 0; i < memProps.memoryHeapCount; i++) {
        if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            outCaps->totalDeviceMemory += memProps.memoryHeaps[i].size;
        }
    }

    Msg("[Vulkan] Selected: %s (score: %d)", best.props.deviceName, best.score);

    // ========================================================================
    // Print full capabilities report
    // ========================================================================
    outCaps->LogInfo();

    return best.device;
}
