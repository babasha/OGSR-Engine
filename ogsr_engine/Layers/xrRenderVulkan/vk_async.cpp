// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

#include "stdafx.h"
#include "vk_async.h"
#include "HW_Vulkan.h"

// Global scope (not namespaced), like the other ps_r_* render cvars.
extern int ps_r_async;

namespace VK { namespace Async {

static bool            s_inited  = false;
static bool            s_failed  = false;   // init failed once → never retry (avoid per-frame spam)
static VkCommandPool   s_pool    = VK_NULL_HANDLE;
static VkCommandBuffer s_cmd[VK_FRAMES_IN_FLIGHT] = {};
static VkSemaphore     s_timeline = VK_NULL_HANDLE;
static u64             s_value    = 0;       // monotonic timeline value

// Per-frame handoff state (set in FrameSubmit, read by GetGraphicsWait same frame).
static bool s_pendingThisFrame = false;
static u64  s_lastSignaled     = 0;

static bool EnsureInit()
{
    if (s_inited) return true;
    if (s_failed) return false;

    // No async without a dedicated compute family (m_ComputeFamily aliases graphics
    // otherwise — see HW_Vulkan queue selection). Nothing to overlap; stay dormant.
    if (VulkanHW.m_ComputeFamily == VulkanHW.m_GraphicsFamily) { s_failed = true; return false; }

    VkCommandPoolCreateInfo pci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    pci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = VulkanHW.m_ComputeFamily;
    if (vkCreateCommandPool(VulkanHW.m_Device, &pci, nullptr, &s_pool) != VK_SUCCESS) {
        Msg("![VK Async] compute command pool create failed — disabled"); s_failed = true; return false;
    }

    VkCommandBufferAllocateInfo ai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    ai.commandPool        = s_pool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = VK_FRAMES_IN_FLIGHT;
    if (vkAllocateCommandBuffers(VulkanHW.m_Device, &ai, s_cmd) != VK_SUCCESS) {
        Msg("![VK Async] compute command buffer alloc failed — disabled"); s_failed = true; return false;
    }

    VkSemaphoreTypeCreateInfo tci{ VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
    tci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    tci.initialValue  = 0;
    VkSemaphoreCreateInfo sci{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    sci.pNext = &tci;
    if (vkCreateSemaphore(VulkanHW.m_Device, &sci, nullptr, &s_timeline) != VK_SUCCESS) {
        Msg("![VK Async] timeline semaphore create failed — disabled"); s_failed = true; return false;
    }

    s_inited = true;
    Msg("[VK Async] init OK — compute family %u, timeline ready (INERT probe; r_async)", VulkanHW.m_ComputeFamily);
    return true;
}

bool Available()
{
    if (!ps_r_async) return false;
    return EnsureInit();
}

void FrameSubmit(u32 slot)
{
    s_pendingThisFrame = false;
    if (!Available())                 return;
    if (slot >= VK_FRAMES_IN_FLIGHT)  return;

    // This slot's command buffer last ran in the frame that used the same in-flight
    // slot; that graphics frame waited on this timeline and has since retired (its
    // fence was waited in CRender::Begin), so the reset is safe.
    VkCommandBuffer cmd = s_cmd[slot];
    if (vkResetCommandBuffer(cmd, 0) != VK_SUCCESS) return;

    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) return;
    // INCREMENT 1: intentionally empty. Real compute passes record here next, once
    // this cross-queue path is verified stable in-game.
    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) return;

    const u64 signalVal = ++s_value;
    VkTimelineSemaphoreSubmitInfo tl{ VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO };
    tl.signalSemaphoreValueCount = 1;
    tl.pSignalSemaphoreValues    = &signalVal;

    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.pNext                = &tl;
    si.commandBufferCount   = 1;
    si.pCommandBuffers      = &cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores    = &s_timeline;

    const VkResult r = vkQueueSubmit(VulkanHW.m_ComputeQueue, 1, &si, VK_NULL_HANDLE);
    if (r != VK_SUCCESS) { Msg("![VK Async] compute submit failed: %d", r); return; }

    s_lastSignaled     = signalVal;
    s_pendingThisFrame = true;
}

bool GetGraphicsWait(VkSemaphore& sem, u64& value)
{
    if (!s_pendingThisFrame) return false;
    sem   = s_timeline;
    value = s_lastSignaled;
    return true;
}

void Destroy()
{
    // Drain any in-flight compute submit before tearing down the timeline/pool it
    // references (graphics idle at shutdown covers the dependency, but this is the
    // explicit guarantee against a validation error on the compute queue).
    if (s_inited && VulkanHW.m_ComputeQueue != VK_NULL_HANDLE)
        vkQueueWaitIdle(VulkanHW.m_ComputeQueue);
    if (s_timeline != VK_NULL_HANDLE) { vkDestroySemaphore(VulkanHW.m_Device, s_timeline, nullptr); s_timeline = VK_NULL_HANDLE; }
    if (s_pool     != VK_NULL_HANDLE) { vkDestroyCommandPool(VulkanHW.m_Device, s_pool, nullptr);   s_pool = VK_NULL_HANDLE; }
    for (u32 i = 0; i < VK_FRAMES_IN_FLIGHT; ++i) s_cmd[i] = VK_NULL_HANDLE;
    s_inited = false; s_failed = false; s_value = 0; s_pendingThisFrame = false; s_lastSignaled = 0;
}

}} // namespace VK::Async
