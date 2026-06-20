// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

#pragma once
#include "vk_core.h"
#include <mutex>

// Command buffer management (triple buffering)
class CVulkanCommandManager
{
public:
    static constexpr u32 FRAMES_IN_FLIGHT = VK_FRAMES_IN_FLIGHT;

private:
    VkCommandPool   m_CommandPools[FRAMES_IN_FLIGHT];
    VkCommandBuffer m_CommandBuffers[FRAMES_IN_FLIGHT];
    u32             m_CurrentFrame = 0;

    // Dedicated pool+buffer for one-shot immediate operations (uploads, layout transitions)
    // This is separate from the per-frame render command buffers to avoid conflicts.
    VkCommandPool   m_ImmediatePool = VK_NULL_HANDLE;
    VkCommandBuffer m_ImmediateCmd  = VK_NULL_HANDLE;
    VkFence         m_ImmediateFence = VK_NULL_HANDLE;

    // --- Async buffer uploader (dedicated transfer queue + staging ring + timeline) ---
    // Copies are recorded into m_UploadCmd against a persistent host-mapped staging
    // ring, submitted on VulkanHW.m_TransferQueue signaling m_UploadTimeline. The
    // per-frame graphics submit waits on that timeline (added inside Submit), so the
    // GPU orders uploads before the draws that read them — no CPU stall per upload.
    // Ownership across families is handled by CONCURRENT sharing on upload targets
    // (see CVulkanBuffer::Create), so no explicit queue-ownership-transfer barriers.
    VkCommandPool   m_UploadPool     = VK_NULL_HANDLE;   // on transfer family
    VkCommandBuffer m_UploadCmd      = VK_NULL_HANDLE;
    bool            m_UploadOpen     = false;            // m_UploadCmd has pending copies
    VkSemaphore     m_UploadTimeline = VK_NULL_HANDLE;   // monotonic, signaled per flush
    u64             m_UploadValue    = 0;                // last value submitted / to wait on
    VkBuffer        m_StagingBuf     = VK_NULL_HANDLE;
    VmaAllocation   m_StagingAlloc   = VK_NULL_HANDLE;
    u8*             m_StagingPtr     = nullptr;          // persistently mapped
    VkDeviceSize    m_StagingSize    = 0;
    VkDeviceSize    m_StagingHead    = 0;

    // The whole upload path (m_UploadCmd recording + staging ring + m_UploadValue)
    // is touched from BOTH the main render thread (per-frame FlushUploads/Submit)
    // AND the seqParallel worker thread (spawn-time texture/buffer loads route
    // through UploadImage/UploadBuffer). A VkCommandBuffer may not be recorded by
    // one thread while another ends/resets/submits it — doing so corrupts driver
    // state (observed: c0000005 inside the NV driver at vkCmdPipelineBarrier2 while
    // an NPC streamed in). This serializes every upload op. Recursive because
    // StageBytes (on ring-wrap) and FlushUploadsAndWait re-enter locked methods on
    // the same thread.
    std::recursive_mutex m_UploadMutex;

    void EnsureUploadCmdOpen();
    // Copy `size` bytes into the staging ring (opening the upload cmd); returns the
    // ring byte-offset, or UINT64_MAX if `size` exceeds the whole ring.
    VkDeviceSize StageBytes(const void* data, VkDeviceSize size);

public:
    void Create();
    void Destroy();

    // Per-frame render command buffer (used by render loop only)
    VkCommandBuffer Begin();
    bool End(VkCommandBuffer cmd);
    bool Submit(VkCommandBuffer cmd, VkSemaphore waitSemaphore, VkSemaphore signalSemaphore, VkFence fence);

    // One-shot immediate command buffer (safe to call during rendering)
    VkCommandBuffer BeginImmediate();
    void            EndAndSubmitImmediate(VkCommandBuffer cmd);

    // Async buffer upload: stage `size` bytes from `data` and record a transfer-queue
    // copy into `dst` at `dstOffset`. Non-blocking (stalls only if the staging ring
    // wraps mid-batch). The next graphics submit waits on the upload timeline.
    void UploadBuffer(VkBuffer dst, VkDeviceSize dstOffset, const void* data, VkDeviceSize size);

    // Async image upload: stage `size` bytes (all mips/layers, tightly packed) and
    // record on the transfer queue: UNDEFINED→TRANSFER_DST, copy `regions`, then
    // →SHADER_READ_ONLY_OPTIMAL. `regions[].bufferOffset` are relative to `data`
    // (the uploader rebases them into the staging ring). Leaves the image in
    // SHADER_READ_ONLY_OPTIMAL; the graphics frame's timeline wait (FRAGMENT_SHADER)
    // makes the copy+transition visible to samplers. Image MUST be CONCURRENT.
    void UploadImage(VkImage dst, const void* data, VkDeviceSize size,
                     const VkBufferImageCopy* regions, u32 regionCount,
                     u32 mipLevels, u32 arrayLayers);

    void FlushUploads();         // submit pending copies (async); call once per frame before Submit
    void FlushUploadsAndWait();  // submit + block until complete; use at end of level load

    void NextFrame() { m_CurrentFrame = (m_CurrentFrame + 1) % FRAMES_IN_FLIGHT; }
    u32 GetCurrentFrame() const { return m_CurrentFrame; }

    VkCommandBuffer GetCurrentCommandBuffer() const { return m_CommandBuffers[m_CurrentFrame]; }
    VkCommandPool   GetCurrentPool() const { return m_CommandPools[m_CurrentFrame]; }
    VkCommandPool   GetPool(u32 index) const { return m_CommandPools[index]; }
    void            ResetFrameCounter() { m_CurrentFrame = 0; }
};

// Глобальный экземпляр
extern CVulkanCommandManager CommandManager;
