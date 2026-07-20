// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

#include "stdafx.h"
#include "vk_buffer.h"
#include "HW_Vulkan.h"
#include "vk_command_buffer.h"

namespace VK
{

// Threshold для staging buffer (64 KB)
static constexpr VkDeviceSize STAGING_THRESHOLD = 64 * 1024;

// Constructor
CVulkanBuffer::CVulkanBuffer()
{
}

// Destructor
CVulkanBuffer::~CVulkanBuffer()
{
    Destroy();
}

// Создание буфера
void CVulkanBuffer::Create(VkDeviceSize size, VkBufferUsageFlags usage, VmaMemoryUsage memUsage, bool gpuOnly)
{
    if (m_Buffer != VK_NULL_HANDLE) {
        Msg("![Vulkan] Buffer already created, call Destroy first");
        return;
    }

    if (size == 0) {
        Msg("![Vulkan] Cannot create buffer with size 0");
        return;
    }

    m_Size = size;
    m_Usage = usage;
    m_MemUsage = memUsage;

    // Buffer create info
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    // VMA allocation info
    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = memUsage;

    if (gpuOnly) {
        // GPU-only: DEVICE_LOCAL is REQUIRED and no host access is requested. The
        // default storage-buffer path below forces HOST_ACCESS_SEQUENTIAL_WRITE,
        // which makes VMA pick HOST_VISIBLE memory — the 256MB BAR heap, or worse,
        // SYSTEM RAM once BAR is full: every shader access then rides PCIe (the
        // Pripyat tree-bin 20-30ms pathology). CPU writes go through Upload/Fill.
        allocInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    } else {
    // Для uniform/storage buffers включаем HOST_VISIBLE для persistent mapping
    // Storage buffers (e.g. bone SSBO) also need CPU write access for per-frame updates
    if (usage & (VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
        allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                          VMA_ALLOCATION_CREATE_MAPPED_BIT;
    }

    // Для staging buffers (TRANSFER_SRC) или HOST memory - нужен host access
    if ((usage & VK_BUFFER_USAGE_TRANSFER_SRC_BIT) || memUsage == VMA_MEMORY_USAGE_AUTO_PREFER_HOST) {
        allocInfo.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    }
    }

    // ВАЖНО: Для VERTEX/INDEX буферов добавляем TRANSFER_DST для staging uploads
    if ((usage & (VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT)) &&
        !(usage & VK_BUFFER_USAGE_TRANSFER_DST_BIT)) {
        bufferInfo.usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    }

    // Upload targets are filled by the async uploader on the dedicated TRANSFER
    // queue and read on the GRAPHICS queue. With distinct families that's a
    // cross-family access — use CONCURRENT sharing so the contents survive without
    // explicit queue-ownership-transfer barriers (negligible cost for buffers).
    // Only when a TRANSFER_DST target AND the families actually differ.
    const u32 families[2] = { VulkanHW.m_GraphicsFamily, VulkanHW.m_TransferFamily };
    if ((bufferInfo.usage & VK_BUFFER_USAGE_TRANSFER_DST_BIT) &&
        VulkanHW.m_TransferFamily != VulkanHW.m_GraphicsFamily) {
        bufferInfo.sharingMode           = VK_SHARING_MODE_CONCURRENT;
        bufferInfo.queueFamilyIndexCount = 2;
        bufferInfo.pQueueFamilyIndices   = families;
    }

    VK_CHECK(VK::Vram::CreateBuffer(VulkanHW.m_Allocator, &bufferInfo, &allocInfo,
                             &m_Buffer, &m_Allocation, nullptr));

    // Tag the VMA allocation by usage so any leaked buffer is identifiable in the
    // vmaDestroyAllocator leak dump (see vma_impl.cpp). Negligible cost.
    if (m_Allocation) {
        const char* tag =
            (usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) ? "buf:storage" :
            (usage & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) ? "buf:uniform" :
            (usage & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT)  ? "buf:vertex"  :
            (usage & VK_BUFFER_USAGE_INDEX_BUFFER_BIT)   ? "buf:index"   :
            (usage & VK_BUFFER_USAGE_TRANSFER_SRC_BIT)   ? "buf:staging" : "buf:other";
        vmaSetAllocationName(VulkanHW.m_Allocator, m_Allocation, tag);
    }

    // Если буфер создан с MAPPED_BIT, получаем mapped pointer
    if (allocInfo.flags & VMA_ALLOCATION_CREATE_MAPPED_BIT) {
        VmaAllocationInfo allocInfoResult;
        vmaGetAllocationInfo(VulkanHW.m_Allocator, m_Allocation, &allocInfoResult);
        m_Mapped = allocInfoResult.pMappedData;
    }
}

// Уничтожение буфера
void CVulkanBuffer::Destroy()
{
    if (m_Buffer == VK_NULL_HANDLE) {
        return;
    }

    // Unmap if mapped (skip for persistent-mapped uniform/storage buffers — VMA handles them)
    if (m_Mapped != nullptr && !(m_Usage & (VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT))) {
        Unmap();
    }

    VK::Vram::DestroyBuffer(VulkanHW.m_Allocator, m_Buffer, m_Allocation);

    m_Buffer = VK_NULL_HANDLE;
    m_Allocation = VK_NULL_HANDLE;
    m_Size = 0;
    m_Mapped = nullptr;
}

void CVulkanBuffer::AdoptFrom(CVulkanBuffer& other)
{
    if (this == &other) return;
    Destroy();
    m_Buffer     = other.m_Buffer;
    m_Allocation = other.m_Allocation;
    m_Size       = other.m_Size;
    m_Mapped     = other.m_Mapped;
    m_Usage      = other.m_Usage;
    m_MemUsage   = other.m_MemUsage;
    other.m_Buffer     = VK_NULL_HANDLE;
    other.m_Allocation = VK_NULL_HANDLE;
    other.m_Size       = 0;
    other.m_Mapped     = nullptr;
}

// Upload данных
void CVulkanBuffer::Upload(const void* data, VkDeviceSize size, VkDeviceSize offset)
{
    if (!IsValid()) {
        Msg("![Vulkan] Upload: buffer is not valid (Create() was not called or failed)");
        return;
    }

    if (!data) {
        Msg("![Vulkan] Upload: data is null");
        return;
    }

    if (offset + size > m_Size) {
        Msg("![Vulkan] Upload: offset + size exceeds buffer size");
        return;
    }

    // Проверяем, является ли буфер host-visible (можно ли Map)
    VmaAllocationInfo allocInfo;
    vmaGetAllocationInfo(VulkanHW.m_Allocator, m_Allocation, &allocInfo);

    VkMemoryPropertyFlags memFlags;
    vmaGetMemoryTypeProperties(VulkanHW.m_Allocator, allocInfo.memoryType, &memFlags);

    bool isHostVisible = (memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;

    // Если буфер НЕ host-visible (DEVICE_LOCAL only) - всегда используем staging
    if (!isHostVisible) {
        UploadViaStaging(data, size, offset);
        return;
    }

    // Буфер host-visible - можем мапить напрямую
    // Для небольших данных - прямой memcpy
    if (size < STAGING_THRESHOLD) {
        void* mapped = Map();
        if (!mapped) {
            Msg("![Vulkan] Failed to map buffer for upload");
            // Fallback to staging
            UploadViaStaging(data, size, offset);
            return;
        }

        memcpy((u8*)mapped + offset, data, size);
        Flush();

        // Unmap только если не persistent-mapped uniform buffer
        if (!(m_Usage & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)) {
            Unmap();
        }
    }
    else {
        // Для больших данных - staging buffer (быстрее для transfer)
        UploadViaStaging(data, size, offset);
    }
}

// Upload через staging — теперь асинхронно на выделенной transfer-очереди.
// Раньше тут на КАЖДЫЙ upload создавался+уничтожался staging-буфер и делался
// immediate submit+wait (полный стол). Теперь — общий async-аплоадер
// (персистентный staging-ring + timeline), без per-upload аллокаций и без стола:
// следующий graphics-сабмит ждёт копию через upload-timeline (см. CommandManager).
void CVulkanBuffer::UploadViaStaging(const void* data, VkDeviceSize size, VkDeviceSize offset)
{
    if (!IsValid()) {
        Msg("![Vulkan] UploadViaStaging: destination buffer is not valid");
        return;
    }
    CommandManager.UploadBuffer(m_Buffer, offset, data, size);
}

// Map memory
void* CVulkanBuffer::Map()
{
    if (!IsValid()) {
        Msg("![Vulkan] Map: buffer is not valid (Create() was not called or failed)");
        return nullptr;
    }

    if (m_Mapped != nullptr) {
        // Already mapped (persistent-mapped uniform buffer)
        return m_Mapped;
    }

    void* data = nullptr;
    VkResult result = vmaMapMemory(VulkanHW.m_Allocator, m_Allocation, &data);

    if (result != VK_SUCCESS) {
        Msg("![Vulkan] vmaMapMemory failed: %d", result);
        return nullptr;
    }

    m_Mapped = data;
    return data;
}

// Unmap memory
void CVulkanBuffer::Unmap()
{
    if (!IsValid()) {
        return;
    }

    if (m_Mapped == nullptr) {
        return;
    }

    // Не unmap persistent-mapped uniform/storage buffers
    if (m_Usage & (VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
        return;
    }

    vmaUnmapMemory(VulkanHW.m_Allocator, m_Allocation);
    m_Mapped = nullptr;
}

// Flush (для non-coherent memory)
void CVulkanBuffer::Flush()
{
    if (!IsValid()) {
        return;
    }

    VkResult result = vmaFlushAllocation(VulkanHW.m_Allocator, m_Allocation, 0, VK_WHOLE_SIZE);
    if (result != VK_SUCCESS) {
        Msg("![Vulkan] vmaFlushAllocation failed: %d", result);
    }
}

// Invalidate (для non-coherent memory)
void CVulkanBuffer::Invalidate()
{
    if (!IsValid()) {
        return;
    }

    VkResult result = vmaInvalidateAllocation(VulkanHW.m_Allocator, m_Allocation, 0, VK_WHOLE_SIZE);
    if (result != VK_SUCCESS) {
        Msg("![Vulkan] vmaInvalidateAllocation failed: %d", result);
    }
}

// Get buffer device address
VkDeviceAddress CVulkanBuffer::GetDeviceAddress() const
{
    if (!IsValid()) return 0;
    VkBufferDeviceAddressInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    info.buffer = m_Buffer;
    return vkGetBufferDeviceAddress(VulkanHW.m_Device, &info);
}

} // namespace VK
