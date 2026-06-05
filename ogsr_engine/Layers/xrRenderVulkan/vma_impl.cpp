// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
// Licensed under the same terms as X-Ray Engine (see root License.txt)

// ВАЖНО: VMA требует exceptions, поэтому компилируем этот файл с /EHsc
// Этот файл полностью изолирован и НЕ использует stdafx.h или xrCore.h

// Отключаем warnings
#pragma warning(push)
#pragma warning(disable: 4100) // unreferenced formal parameter
#pragma warning(disable: 4189) // local variable is initialized but not referenced

// Определяем Windows и Vulkan макросы напрямую
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>

// Настройки VMA для минимальных зависимостей
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#define VMA_VULKAN_VERSION 1003000 // Vulkan 1.3

// VMA leak diagnostics (permanent regression guard). On a clean shutdown these
// emit NOTHING; if any VkBuffer/VkImage outlives vmaDestroyAllocator() they are
// each enumerated into the xray log by name, and the leak is logged rather than
// popping a blocking Debug assert dialog (leak-on-exit is harmless — the OS
// reclaims it — but we want it visible, not fatal). `Msg` lives in xrCore
// (statically linked into the exe); forward-declare it so this otherwise-isolated
// TU can call it without pulling xrCore headers.
void __cdecl Msg(const char* format, ...);
#define VMA_LEAK_LOG_FORMAT(format, ...)  do { Msg("![VK-VMA-LEAK] " format, __VA_ARGS__); } while (0)
#define VMA_ASSERT_LEAK(expr)             do { if (!(expr)) Msg("![VK-VMA] leak-assert: %s", #expr); } while (0)

// Определяем VMA implementation только здесь
#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#pragma warning(pop)

// Этот файл служит только для компиляции VMA implementation
// Все функции VMA будут доступны через vk_mem_alloc.h в других файлах
