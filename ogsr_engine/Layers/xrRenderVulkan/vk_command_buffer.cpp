// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

#include "stdafx.h"
#include "vk_command_buffer.h"
#include "HW_Vulkan.h"
#include "vk_async.h"          // VK::Async::GetGraphicsWait — async-compute timeline wait
#include "vk_profiler.h"       // VK::Prof::Checkpoint / DumpCheckpoints — GPU-hang post-mortem

// Глобальный экземпляр
CVulkanCommandManager CommandManager;

// Создание command pools и buffers
void CVulkanCommandManager::Create()
{
    VK::Vram::Scope _vram_scope("Staging");
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

    // Second graphics segment per slot (async frame-split) — from the same per-slot
    // pool (RESET_COMMAND_BUFFER_BIT lets each buffer re-begin independently).
    for (u32 i = 0; i < FRAMES_IN_FLIGHT; i++) {
        VkCommandBufferAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = m_CommandPools[i];
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;
        VK_CHECK_CRITICAL(vkAllocateCommandBuffers(VulkanHW.m_Device, &allocInfo, &m_PostBuffers[i]));
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
        VK_CHECK_CRITICAL(VK::Vram::CreateBuffer(VulkanHW.m_Allocator, &bci, &aci, &m_StagingBuf, &m_StagingAlloc, &si));
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
        VK::Vram::DestroyBuffer(VulkanHW.m_Allocator, m_StagingBuf, m_StagingAlloc);
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

// Begin the frame's second graphics segment (async frame-split).
VkCommandBuffer CVulkanCommandManager::BeginSecondSegment()
{
    if (g_bDeviceLost) return VK_NULL_HANDLE;
    VkCommandBuffer cmd = m_PostBuffers[m_CurrentFrame];

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VkResult res = vkBeginCommandBuffer(cmd, &beginInfo);
    if (res != VK_SUCCESS) {
        Msg("!Vulkan BeginSecondSegment error: %d", res);
        if (res == VK_ERROR_DEVICE_LOST && !g_bDeviceLost) { Msg("!Vulkan DEVICE LOST in BeginSecondSegment()"); g_bDeviceLost = true; }
        return VK_NULL_HANDLE;
    }
    return cmd;
}

// Submit команд в очередь — returns false on error
bool CVulkanCommandManager::Submit(VkCommandBuffer cmd, VkSemaphore waitSemaphore,
                                   VkSemaphore signalSemaphore, VkFence fence,
                                   bool waitAsyncCompute)
{
    if (g_bDeviceLost || cmd == VK_NULL_HANDLE) return false;

    // Build the wait list: imageAvailable (binary) plus — when async uploads are
    // pending — the upload timeline, so the GPU defers draws that consume just-
    // uploaded vertex/index/texture data until the transfer-queue copy completes.
    VkSemaphore          waitSems[3];
    VkPipelineStageFlags waitStages[3];
    u64                  waitVals[3];
    u32 nWait = 0;
    if (waitSemaphore != VK_NULL_HANDLE) {
        waitSems[nWait]   = waitSemaphore;
        waitStages[nWait] = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        waitVals[nWait]   = 0;   // binary — value ignored
        nWait++;
    }
    // Snapshot the upload timeline under the upload lock: a worker thread may be
    // mid-FlushUploads (++m_UploadValue then transfer submit). Capturing both under
    // the lock yields a value whose copy has already been submitted, so this
    // graphics frame waits on a consistent, in-flight upload point.
    VkSemaphore uploadSem;
    u64         uploadVal;
    {
        std::lock_guard<std::recursive_mutex> lock(m_UploadMutex);
        uploadSem = m_UploadTimeline;
        uploadVal = m_UploadValue;
    }
    const bool waitUpload = (uploadSem != VK_NULL_HANDLE && uploadVal > 0);
    if (waitUpload) {
        waitSems[nWait]   = uploadSem;
        waitStages[nWait] = VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                          | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;
        waitVals[nWait]   = uploadVal;
        nWait++;
    }

    // Async compute (vk_async): when compute work was submitted to the compute queue
    // this frame, the graphics work must wait on its timeline before consuming any
    // compute output. Broad dst stages (indirect/vertex/compute/fragment) so any
    // future consumer is covered. Mirrors the upload-timeline wait above.
    VkSemaphore asyncSem = VK_NULL_HANDLE;
    u64         asyncVal = 0;
    const bool  waitAsync = waitAsyncCompute && VK::Async::GetGraphicsWait(asyncSem, asyncVal);
    if (waitAsync) {
        waitSems[nWait]   = asyncSem;
        waitStages[nWait] = VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
                          | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        waitVals[nWait]   = asyncVal;
        nWait++;
    }

    u64 signalVal = 0;  // renderFinished is binary — value ignored

    const bool useTimeline = waitUpload || waitAsync;   // any timeline semaphore in the wait list
    VkTimelineSemaphoreSubmitInfo tl = {};
    tl.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    tl.waitSemaphoreValueCount   = nWait;
    tl.pWaitSemaphoreValues      = waitVals;
    tl.signalSemaphoreValueCount = signalSemaphore != VK_NULL_HANDLE ? 1 : 0;
    tl.pSignalSemaphoreValues    = &signalVal;

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext = useTimeline ? &tl : nullptr;   // only need timeline values when a timeline is in play
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
        if (res == VK_ERROR_DEVICE_LOST) {
            VK::Prof::DumpCheckpoints("Submit");
            if (!g_bDeviceLost) {
                Msg("!Vulkan DEVICE LOST in Submit()");
                g_bDeviceLost = true;
            }
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

    VK::Prof::Checkpoint(m_ImmediateCmd, "Immediate");
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
        if (res == VK_ERROR_DEVICE_LOST) VK::Prof::DumpCheckpoints("EndAndSubmitImmediate");
        return;
    }

    // A timeout here means the graphics queue is wedged — the next BeginImmediate
    // would reset this buffer while the GPU may still own it. Log + post-mortem
    // (this is the classic pre-TDR window of the level-transition hang).
    res = vkWaitForFences(VulkanHW.m_Device, 1, &m_ImmediateFence, VK_TRUE, 2000000000ULL);
    if (res != VK_SUCCESS) {
        Msg("!Vulkan EndAndSubmitImmediate: fence wait %s (%d)", res == VK_TIMEOUT ? "TIMEOUT" : "error", res);
        VK::Prof::DumpCheckpoints("ImmediateFenceWait");
    }
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
        VkResult wr = vkWaitSemaphores(VulkanHW.m_Device, &wi, 2000000000ULL);
        if (wr != VK_SUCCESS) {
            // Resetting m_UploadCmd below while the transfer queue still owns it is
            // UB that cascades into a device loss — make the real culprit visible.
            Msg("!Vulkan EnsureUploadCmdOpen: upload timeline wait %s (%d, value %llu)",
                wr == VK_TIMEOUT ? "TIMEOUT" : "error", wr, m_UploadValue);
            VK::Prof::DumpCheckpoints("UploadTimelineWait");
        }
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
    VK::Vram::Scope _vram_scope("Staging");
    std::lock_guard<std::recursive_mutex> lock(m_UploadMutex);   // shared upload cmd/ring — see header
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
        if (VK::Vram::CreateBuffer(VulkanHW.m_Allocator, &bci, &aci, &tmp, &tmpAlloc, &ti) != VK_SUCCESS) return;
        memcpy(ti.pMappedData, data, (size_t)size);
        vmaFlushAllocation(VulkanHW.m_Allocator, tmpAlloc, 0, size);
        FlushUploadsAndWait();              // drain the ring batch first (ordering)
        EnsureUploadCmdOpen();
        VkBufferCopy r = { 0, dstOffset, size };
        vkCmdCopyBuffer(m_UploadCmd, tmp, dst, 1, &r);
        FlushUploadsAndWait();              // submit + wait so tmp is safe to free
        VK::Vram::DestroyBuffer(VulkanHW.m_Allocator, tmp, tmpAlloc);
        return;
    }

    const VkDeviceSize off = StageBytes(data, size);

    VK::Prof::Checkpoint(m_UploadCmd, "Upload:buffer");
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
    VK::Vram::Scope _vram_scope("Staging");
    std::lock_guard<std::recursive_mutex> lock(m_UploadMutex);   // shared upload cmd/ring — see header
    if (g_bDeviceLost || dst == VK_NULL_HANDLE || !data || size == 0 || !regions || regionCount == 0 || !m_StagingPtr)
        return;

    // Oversize (> whole ring): one-off temp staging + synchronous submit, mirroring
    // UploadBuffer's oversize path. This DOES happen — a 4096² UNCOMPRESSED mod
    // texture is 89 MB vs the 64 MB ring; the old "skip" left the VkImage with
    // undefined (black) texels → the terrain "black grass channel" bug.
    VkBuffer      tmpBuf   = VK_NULL_HANDLE;
    VmaAllocation tmpAlloc = VK_NULL_HANDLE;
    VkDeviceSize  off      = 0;
    if (size > m_StagingSize) {
        Msg("![Vulkan] UploadImage: %zu bytes exceeds %zu-byte staging ring — one-off staging", (size_t)size, (size_t)m_StagingSize);
        VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bci.size = size; bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT; bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo aci = {};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo ti = {};
        if (VK::Vram::CreateBuffer(VulkanHW.m_Allocator, &bci, &aci, &tmpBuf, &tmpAlloc, &ti) != VK_SUCCESS) {
            Msg("![Vulkan] UploadImage: one-off staging alloc FAILED — texture skipped");
            return;
        }
        memcpy(ti.pMappedData, data, (size_t)size);
        vmaFlushAllocation(VulkanHW.m_Allocator, tmpAlloc, 0, size);
        FlushUploadsAndWait();              // drain the ring batch first (ordering)
        EnsureUploadCmdOpen();
    } else {
        off = StageBytes(data, size);       // opens cmd, rebases below
    }

    // Rebase the caller's copy regions (offsets are relative to `data`) into the
    // ring (one-off buffer keeps them as-is: off = 0).
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
    // NOTE: this is why this barrier is hand-rolled and NOT VK::ImageBarrier — the
    // latter derives FRAGMENT_SHADER for SHADER_READ, which is illegal on this queue.
    barrier(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_2_NONE, 0,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

    VK::Prof::Checkpoint(m_UploadCmd, tmpBuf ? "Upload:image1off" : "Upload:image");
    vkCmdCopyBufferToImage(m_UploadCmd, tmpBuf ? tmpBuf : m_StagingBuf, dst,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, regionCount, r.data());

    barrier(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_NONE, 0);

    if (tmpBuf) {
        FlushUploadsAndWait();              // submit + wait so the one-off staging is safe to free
        VK::Vram::DestroyBuffer(VulkanHW.m_Allocator, tmpBuf, tmpAlloc);
    }
}

void CVulkanCommandManager::FlushUploads()
{
    std::lock_guard<std::recursive_mutex> lock(m_UploadMutex);   // shared upload cmd/ring — see header
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
    if (res != VK_SUCCESS) {
        // m_UploadValue was already incremented but will never be signaled now —
        // every later graphics frame waiting on it would wedge the queue. Roll the
        // counter back so waits target the last value that actually signals.
        Msg("!Vulkan FlushUploads: transfer submit error %d (rolling timeline back to %llu)", res, m_UploadValue - 1);
        --m_UploadValue;
        if (res == VK_ERROR_DEVICE_LOST) VK::Prof::DumpCheckpoints("FlushUploads");
    }
}

void CVulkanCommandManager::FlushUploadsAndWait()
{
    std::lock_guard<std::recursive_mutex> lock(m_UploadMutex);   // held across submit+wait so the ring reset is atomic
    FlushUploads();
    if (m_UploadTimeline == VK_NULL_HANDLE || m_UploadValue == 0) return;

    VkSemaphoreWaitInfo wi = {};
    wi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    wi.semaphoreCount = 1;
    wi.pSemaphores    = &m_UploadTimeline;
    wi.pValues        = &m_UploadValue;
    VkResult wr = vkWaitSemaphores(VulkanHW.m_Device, &wi, 2000000000ULL);
    if (wr != VK_SUCCESS) {
        // Copies may still be in flight — rewinding the ring now would let new
        // stagings overwrite data the GPU is reading. Keep the head; log loudly.
        Msg("!Vulkan FlushUploadsAndWait: timeline wait %s (%d, value %llu) — ring NOT rewound",
            wr == VK_TIMEOUT ? "TIMEOUT" : "error", wr, m_UploadValue);
        VK::Prof::DumpCheckpoints("FlushUploadsAndWait");
        return;
    }

    m_StagingHead = 0;   // every submitted copy is done — the whole ring is free again
}

