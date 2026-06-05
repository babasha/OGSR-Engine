// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
// Licensed under the same terms as X-Ray Engine (see root License.txt)

#include "stdafx.h"
#include "vk_command_buffer.h"
#include "HW_Vulkan.h"

// Глобальный экземпляр
CVulkanCommandManager CommandManager;

// Создание command pools и buffers
void CVulkanCommandManager::Create()
{
    Msg("[Vulkan] Creating command pools and buffers...");

    // Создаём command pool для каждого frame in flight
    VkCommandPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = VulkanHW.m_GraphicsFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    for (u32 i = 0; i < FRAMES_IN_FLIGHT; i++) {
        VK_CHECK_CRITICAL(vkCreateCommandPool(VulkanHW.m_Device, &poolInfo, nullptr, &m_CommandPools[i]));
    }

    // Создаём command buffers
    for (u32 i = 0; i < FRAMES_IN_FLIGHT; i++) {
        VkCommandBufferAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = m_CommandPools[i];
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;

        VK_CHECK_CRITICAL(vkAllocateCommandBuffers(VulkanHW.m_Device, &allocInfo, &m_CommandBuffers[i]));
    }

    // Create dedicated immediate command pool+buffer for one-shot operations
    // (texture uploads, layout transitions) — separate from per-frame render buffers
    VK_CHECK_CRITICAL(vkCreateCommandPool(VulkanHW.m_Device, &poolInfo, nullptr, &m_ImmediatePool));

    {
        VkCommandBufferAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = m_ImmediatePool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;
        VK_CHECK_CRITICAL(vkAllocateCommandBuffers(VulkanHW.m_Device, &allocInfo, &m_ImmediateCmd));
    }

    {
        VkFenceCreateInfo fenceInfo = {};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VK_CHECK_CRITICAL(vkCreateFence(VulkanHW.m_Device, &fenceInfo, nullptr, &m_ImmediateFence));
    }

    // --- Async uploader: transfer-family pool + cmd, timeline semaphore, staging ring ---
    {
        VkCommandPoolCreateInfo upi = {};
        upi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        upi.queueFamilyIndex = VulkanHW.m_TransferFamily;   // dedicated DMA family when present
        upi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VK_CHECK_CRITICAL(vkCreateCommandPool(VulkanHW.m_Device, &upi, nullptr, &m_UploadPool));

        VkCommandBufferAllocateInfo ai = {};
        ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ai.commandPool = m_UploadPool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VK_CHECK_CRITICAL(vkAllocateCommandBuffers(VulkanHW.m_Device, &ai, &m_UploadCmd));

        VkSemaphoreTypeCreateInfo tci = {};
        tci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
        tci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        tci.initialValue = 0;
        VkSemaphoreCreateInfo sci = {};
        sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        sci.pNext = &tci;
        VK_CHECK_CRITICAL(vkCreateSemaphore(VulkanHW.m_Device, &sci, nullptr, &m_UploadTimeline));
        m_UploadValue = 0;

        m_StagingSize = (VkDeviceSize)64 * 1024 * 1024;   // 64 MB host-visible ring
        VkBufferCreateInfo bci = {};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = m_StagingSize;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;       // CPU-written, transfer-read only
        VmaAllocationCreateInfo aci = {};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo si = {};
        VK_CHECK_CRITICAL(vmaCreateBuffer(VulkanHW.m_Allocator, &bci, &aci, &m_StagingBuf, &m_StagingAlloc, &si));
        m_StagingPtr  = (u8*)si.pMappedData;
        m_StagingHead = 0;
        if (m_StagingAlloc) vmaSetAllocationName(VulkanHW.m_Allocator, m_StagingAlloc, "buf:upload-ring");
    }

    Msg("[Vulkan] Command pools and buffers created (3 frames in flight + 1 immediate + async uploader, transfer family %u)",
        VulkanHW.m_TransferFamily);
}

// Уничтожение
void CVulkanCommandManager::Destroy()
{
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;

    // Async uploader teardown (device already idle when Destroy runs).
    if (m_StagingBuf != VK_NULL_HANDLE) {
        vmaDestroyBuffer(VulkanHW.m_Allocator, m_StagingBuf, m_StagingAlloc);
        m_StagingBuf = VK_NULL_HANDLE; m_StagingAlloc = VK_NULL_HANDLE; m_StagingPtr = nullptr;
    }
    if (m_UploadTimeline != VK_NULL_HANDLE) {
        vkDestroySemaphore(VulkanHW.m_Device, m_UploadTimeline, nullptr);
        m_UploadTimeline = VK_NULL_HANDLE;
    }
    if (m_UploadPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(VulkanHW.m_Device, m_UploadPool, nullptr);
        m_UploadPool = VK_NULL_HANDLE; m_UploadCmd = VK_NULL_HANDLE;
    }

    if (m_ImmediateFence != VK_NULL_HANDLE) {
        vkDestroyFence(VulkanHW.m_Device, m_ImmediateFence, nullptr);
        m_ImmediateFence = VK_NULL_HANDLE;
    }

    if (m_ImmediatePool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(VulkanHW.m_Device, m_ImmediatePool, nullptr);
        m_ImmediatePool = VK_NULL_HANDLE;
    }

    for (u32 i = 0; i < FRAMES_IN_FLIGHT; i++) {
        if (m_CommandPools[i] != VK_NULL_HANDLE) {
            vkDestroyCommandPool(VulkanHW.m_Device, m_CommandPools[i], nullptr);
            m_CommandPools[i] = VK_NULL_HANDLE;
        }
    }

    Msg("[Vulkan] Command pools destroyed");
}

// Начало записи команд — returns VK_NULL_HANDLE on error
VkCommandBuffer CVulkanCommandManager::Begin()
{
    if (g_bDeviceLost) return VK_NULL_HANDLE;

    VkCommandBuffer cmd = m_CommandBuffers[m_CurrentFrame];

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VkResult res = vkBeginCommandBuffer(cmd, &beginInfo);
    if (res != VK_SUCCESS) {
        Msg("!Vulkan vkBeginCommandBuffer error: %d", res);
        if (res == VK_ERROR_DEVICE_LOST && !g_bDeviceLost) {
            Msg("!Vulkan DEVICE LOST in Begin()");
            g_bDeviceLost = true;
        }
        return VK_NULL_HANDLE;
    }

    return cmd;
}

// Завершение записи команд — returns false on error
bool CVulkanCommandManager::End(VkCommandBuffer cmd)
{
    if (g_bDeviceLost || cmd == VK_NULL_HANDLE) return false;

    VkResult res = vkEndCommandBuffer(cmd);
    if (res != VK_SUCCESS) {
        Msg("!Vulkan vkEndCommandBuffer error: %d", res);
        if (res == VK_ERROR_DEVICE_LOST && !g_bDeviceLost) {
            Msg("!Vulkan DEVICE LOST in End()");
            g_bDeviceLost = true;
        }
        return false;
    }
    return true;
}

// Submit команд в очередь — returns false on error
bool CVulkanCommandManager::Submit(VkCommandBuffer cmd, VkSemaphore waitSemaphore,
                                   VkSemaphore signalSemaphore, VkFence fence)
{
    if (g_bDeviceLost || cmd == VK_NULL_HANDLE) return false;

    // Build the wait list: imageAvailable (binary) plus — when async uploads are
    // pending — the upload timeline, so the GPU defers draws that consume just-
    // uploaded vertex/index/texture data until the transfer-queue copy completes.
    VkSemaphore          waitSems[2];
    VkPipelineStageFlags waitStages[2];
    u64                  waitVals[2];
    u32 nWait = 0;
    if (waitSemaphore != VK_NULL_HANDLE) {
        waitSems[nWait]   = waitSemaphore;
        waitStages[nWait] = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        waitVals[nWait]   = 0;   // binary — value ignored
        nWait++;
    }
    const bool waitUpload = (m_UploadTimeline != VK_NULL_HANDLE && m_UploadValue > 0);
    if (waitUpload) {
        waitSems[nWait]   = m_UploadTimeline;
        waitStages[nWait] = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                          | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
        waitVals[nWait]   = m_UploadValue;
        nWait++;
    }

    u64 signalVal = 0;  // renderFinished is binary — value ignored

    VkTimelineSemaphoreSubmitInfo tl = {};
    tl.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    tl.waitSemaphoreValueCount   = nWait;
    tl.pWaitSemaphoreValues      = waitVals;
    tl.signalSemaphoreValueCount = signalSemaphore != VK_NULL_HANDLE ? 1 : 0;
    tl.pSignalSemaphoreValues    = &signalVal;

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext = waitUpload ? &tl : nullptr;   // only need timeline values when a timeline is in play
    submitInfo.waitSemaphoreCount = nWait;
    submitInfo.pWaitSemaphores = waitSems;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    submitInfo.signalSemaphoreCount = signalSemaphore != VK_NULL_HANDLE ? 1 : 0;
    submitInfo.pSignalSemaphores = &signalSemaphore;

    VkResult res = vkQueueSubmit(VulkanHW.m_GraphicsQueue, 1, &submitInfo, fence);
    if (res != VK_SUCCESS) {
        Msg("!Vulkan vkQueueSubmit error: %d", res);
        if (res == VK_ERROR_DEVICE_LOST && !g_bDeviceLost) {
            Msg("!Vulkan DEVICE LOST in Submit()");
            g_bDeviceLost = true;
        }
        return false;
    }
    return true;
}

// ============================================================================
// Immediate (one-shot) command buffer — safe to call during rendering.
// Uses a dedicated pool+buffer that never conflicts with per-frame render buffers.
// ============================================================================
VkCommandBuffer CVulkanCommandManager::BeginImmediate()
{
    if (g_bDeviceLost || m_ImmediateCmd == VK_NULL_HANDLE) return VK_NULL_HANDLE;

    // Reset the immediate command buffer before reuse
    vkResetCommandBuffer(m_ImmediateCmd, 0);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    VkResult res = vkBeginCommandBuffer(m_ImmediateCmd, &beginInfo);
    if (res != VK_SUCCESS) {
        Msg("!Vulkan BeginImmediate error: %d", res);
        return VK_NULL_HANDLE;
    }

    return m_ImmediateCmd;
}

void CVulkanCommandManager::EndAndSubmitImmediate(VkCommandBuffer cmd)
{
    if (g_bDeviceLost || cmd == VK_NULL_HANDLE) return;

    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    vkResetFences(VulkanHW.m_Device, 1, &m_ImmediateFence);
    VkResult res = vkQueueSubmit(VulkanHW.m_GraphicsQueue, 1, &submitInfo, m_ImmediateFence);
    if (res != VK_SUCCESS) {
        Msg("!Vulkan EndAndSubmitImmediate: submit error %d", res);
        return;
    }

    vkWaitForFences(VulkanHW.m_Device, 1, &m_ImmediateFence, VK_TRUE, 2000000000ULL);
}

// ============================================================================
// Async buffer uploader — transfer queue + staging ring + timeline semaphore.
// ============================================================================
void CVulkanCommandManager::EnsureUploadCmdOpen()
{
    if (m_UploadOpen) return;

    // m_UploadCmd is a single buffer reused across batches and submitted ASYNC.
    // Before resetting it we must know the GPU finished its previous submission —
    // the timeline reaching m_UploadValue proves that. Effectively free: a whole
    // frame passes between flushes, so the small transfer is long done (and after
    // FlushUploadsAndWait this returns immediately).
    if (m_UploadTimeline != VK_NULL_HANDLE && m_UploadValue > 0) {
        VkSemaphoreWaitInfo wi = {};
        wi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        wi.semaphoreCount = 1;
        wi.pSemaphores    = &m_UploadTimeline;
        wi.pValues        = &m_UploadValue;
        vkWaitSemaphores(VulkanHW.m_Device, &wi, 2000000000ULL);
    }

    vkResetCommandBuffer(m_UploadCmd, 0);
    VkCommandBufferBeginInfo bi = {};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(m_UploadCmd, &bi);
    m_UploadOpen = true;
}

VkDeviceSize CVulkanCommandManager::StageBytes(const void* data, VkDeviceSize size)
{
    if (size > m_StagingSize) return UINT64_MAX;            // caller handles oversize

    // Align the ring offset to 16 bytes. vkCmdCopyBufferToImage requires bufferOffset
    // be a multiple of the texel block size (16 for BC3/BC5/BC7, 8 for BC1, 4 for
    // BGRA8) AND — on a transfer-only queue — a multiple of 4. A 16-aligned base
    // keeps every tightly-packed mip offset (base + k*blockSize) compliant. Buffers
    // don't need it but the ≤15-byte waste is irrelevant.
    VkDeviceSize off = (m_StagingHead + 15) & ~VkDeviceSize(15);
    if (off + size > m_StagingSize) {       // wrap: drain in-flight copies, rewind to base (0, already aligned)
        FlushUploadsAndWait();
        off = 0;
    }
    EnsureUploadCmdOpen();
    memcpy(m_StagingPtr + off, data, (size_t)size);
    vmaFlushAllocation(VulkanHW.m_Allocator, m_StagingAlloc, off, size);  // no-op if HOST_COHERENT
    m_StagingHead = off + size;
    return off;
}

void CVulkanCommandManager::UploadBuffer(VkBuffer dst, VkDeviceSize dstOffset, const void* data, VkDeviceSize size)
{
    if (g_bDeviceLost || dst == VK_NULL_HANDLE || !data || size == 0 || !m_StagingPtr) return;

    // Pathological oversize (> whole ring): one-off temp staging, synchronous on the
    // transfer queue. Never happens for real vertex/index buffers; defensive only.
    if (size > m_StagingSize) {
        Msg("![Vulkan] UploadBuffer: %zu bytes exceeds %zu-byte ring — one-off staging", (size_t)size, (size_t)m_StagingSize);
        VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bci.size = size; bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT; bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo aci = {};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VkBuffer tmp; VmaAllocation tmpAlloc; VmaAllocationInfo ti = {};
        if (vmaCreateBuffer(VulkanHW.m_Allocator, &bci, &aci, &tmp, &tmpAlloc, &ti) != VK_SUCCESS) return;
        memcpy(ti.pMappedData, data, (size_t)size);
        vmaFlushAllocation(VulkanHW.m_Allocator, tmpAlloc, 0, size);
        FlushUploadsAndWait();              // drain the ring batch first (ordering)
        EnsureUploadCmdOpen();
        VkBufferCopy r = { 0, dstOffset, size };
        vkCmdCopyBuffer(m_UploadCmd, tmp, dst, 1, &r);
        FlushUploadsAndWait();              // submit + wait so tmp is safe to free
        vmaDestroyBuffer(VulkanHW.m_Allocator, tmp, tmpAlloc);
        return;
    }

    const VkDeviceSize off = StageBytes(data, size);

    VkBufferCopy region = {};
    region.srcOffset = off;
    region.dstOffset = dstOffset;
    region.size      = size;
    vkCmdCopyBuffer(m_UploadCmd, m_StagingBuf, dst, 1, &region);
}

void CVulkanCommandManager::UploadImage(VkImage dst, const void* data, VkDeviceSize size,
                                        const VkBufferImageCopy* regions, u32 regionCount,
                                        u32 mipLevels, u32 arrayLayers)
{
    if (g_bDeviceLost || dst == VK_NULL_HANDLE || !data || size == 0 || !regions || regionCount == 0 || !m_StagingPtr)
        return;
    if (size > m_StagingSize) {
        Msg("![Vulkan] UploadImage: %zu bytes exceeds %zu-byte staging ring — texture skipped", (size_t)size, (size_t)m_StagingSize);
        return;   // realistically never (textures < 64MB); skip rather than corrupt the ring
    }

    const VkDeviceSize off = StageBytes(data, size);   // opens cmd, rebases below

    // Rebase the caller's copy regions (offsets are relative to `data`) into the ring.
    xr_vector<VkBufferImageCopy> r(regions, regions + regionCount);
    for (auto& reg : r) reg.bufferOffset += off;

    VkImageSubresourceRange range = {};
    range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    range.levelCount = mipLevels;
    range.layerCount = arrayLayers;

    auto barrier = [&](VkImageLayout oldL, VkImageLayout newL,
                       VkPipelineStageFlags2 srcS, VkAccessFlags2 srcA,
                       VkPipelineStageFlags2 dstS, VkAccessFlags2 dstA) {
        VkImageMemoryBarrier2 b = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        b.srcStageMask = srcS; b.srcAccessMask = srcA;
        b.dstStageMask = dstS; b.dstAccessMask = dstA;
        b.oldLayout = oldL; b.newLayout = newL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;   // CONCURRENT image → no ownership transfer
        b.image = dst; b.subresourceRange = range;
        VkDependencyInfo di = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(m_UploadCmd, &di);
    };

    // UNDEFINED → TRANSFER_DST, copy, → SHADER_READ_ONLY. The final transition uses
    // dstStage = NONE: a transfer-only queue can't name FRAGMENT_SHADER. Visibility to
    // the graphics sampler comes from the upload timeline (graphics waits at
    // FRAGMENT_SHADER) — valid because the image is CONCURRENT (no ownership transfer).
    barrier(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_2_NONE, 0,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

    vkCmdCopyBufferToImage(m_UploadCmd, m_StagingBuf, dst,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, regionCount, r.data());

    barrier(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_NONE, 0);
}

void CVulkanCommandManager::FlushUploads()
{
    if (g_bDeviceLost || !m_UploadOpen) return;
    vkEndCommandBuffer(m_UploadCmd);
    m_UploadOpen = false;

    const u64 signalVal = ++m_UploadValue;

    VkTimelineSemaphoreSubmitInfo tl = {};
    tl.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    tl.signalSemaphoreValueCount = 1;
    tl.pSignalSemaphoreValues    = &signalVal;

    VkSubmitInfo si = {};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.pNext = &tl;
    si.commandBufferCount   = 1;
    si.pCommandBuffers      = &m_UploadCmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores    = &m_UploadTimeline;

    VkResult res = vkQueueSubmit(VulkanHW.m_TransferQueue, 1, &si, VK_NULL_HANDLE);
    if (res != VK_SUCCESS) Msg("!Vulkan FlushUploads: transfer submit error %d", res);
}

void CVulkanCommandManager::FlushUploadsAndWait()
{
    FlushUploads();
    if (m_UploadTimeline == VK_NULL_HANDLE || m_UploadValue == 0) return;

    VkSemaphoreWaitInfo wi = {};
    wi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    wi.semaphoreCount = 1;
    wi.pSemaphores    = &m_UploadTimeline;
    wi.pValues        = &m_UploadValue;
    vkWaitSemaphores(VulkanHW.m_Device, &wi, 2000000000ULL);

    m_StagingHead = 0;   // every submitted copy is done — the whole ring is free again
}
