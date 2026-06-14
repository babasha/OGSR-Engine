// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan - VulkanUI namespace: pipeline + vertex buffer + descriptor
// pool for the menu/UI subsystem.
//
// Adapted from the monolith UI path with two notable changes: RCache.m_Cmd
// is replaced by the global g_VkUI_FrameCmd (set by CRender::Begin each
// frame), and RCache.stat.calls/verts are dropped — vkUIRender already does
// its own throttled draw-count logging.

#pragma once
#include "vk_core.h"
#include <vulkan/vulkan.h>

// Current frame's command buffer, set by CRender::Begin and reset by End.
// VulkanUI / vkUIRender use this in immediate mode; if VK_NULL_HANDLE, draw
// commands are buffered into the deferred queue and replayed from CRender::End.
extern VkCommandBuffer g_VkUI_FrameCmd;

namespace VulkanUI
{
    void Create();
    void Destroy();
    void EndUIPass();
    void BeginUIPassInternal();
    void ReplayDeferredUI();

    // Per-frame ring rotation: selects this in-flight slot's vertex-buffer
    // region and resets the write offset. MUST be called from CRender::Begin
    // AFTER the slot's fence was waited — that's the proof the GPU finished
    // reading this region (frames-in-flight discipline, same as the bone SSBO).
    // Resetting the offset anywhere else (it used to happen in EndUIPass) lets
    // the CPU overwrite vertex data the GPU is still reading → in Release the
    // HUD flickered with alien fragments (font glyphs on the minimap etc.);
    // Debug masked it by being too slow to outrun the GPU.
    void OnFrameBegin(u32 frameSlot);

    // Exposed for vkUIRender — write target for PushPoint, drain for FlushPrimitive.
    extern void*           s_pMappedVB;
    extern u32             s_UIVertexOffset;   // slot-relative write offset
    extern u32             s_VBBase;           // byte base of this frame-slot's region
    extern VkDescriptorSet s_WhiteTextureSet;
    extern bool            s_bUIPassActive;
    extern VkPipeline      s_Pipeline;          // TRIANGLE_LIST
    extern VkPipeline      s_PipelineLineList;   // LINE_LIST  (crosshair)
    extern VkPipeline      s_PipelineLineStrip;  // LINE_STRIP (UIWindow borders)
    extern VkPipelineLayout s_PipelineLayout;
    extern VkDescriptorSetLayout s_DescriptorSetLayout;
    extern VkDescriptorPool s_DescriptorPool;

    // PER-SLOT region size (the actual buffer is this × FRAMES_IN_FLIGHT — a
    // frame-fenced ring; see OnFrameBegin). Settings menu draws hundreds of
    // widgets per frame (ui_common-backed panels + every option label). 256KB
    // wraps mid-frame and the wraparound path resets s_UIVertexOffset = 0,
    // overwriting still-pending draws — that produced the "only right side of
    // settings renders" bug. 8MB per slot is wasteful but cheap (host VMA).
    static const VkDeviceSize VERTEX_BUFFER_SIZE = 8 * 1024 * 1024;

    struct DeferredUICmd
    {
        enum Type { Draw, Scissor, ResetScissor };
        Type type;
        u32 vertexBufferOffset;
        u32 vertexCount;
        VkDescriptorSet textureSet;
        VkRect2D scissorRect;
        VkPipeline pipeline;   // topology-specific pipeline (line vs triangle); null = default triangle list

    };
    static const u32 MAX_DEFERRED_CMDS = 4096;
    extern DeferredUICmd s_DeferredCmds[MAX_DEFERRED_CMDS];
    extern u32           s_DeferredCmdCount;

    // Stats — minimal counters; vkUIRender already does its own throttled logging.
    struct FrameStats { u32 droppedCmds = 0; u32 bufferWraps = 0; u32 deferredCmds = 0; u32 deferredDraws = 0; u32 totalVerts = 0; };
    extern FrameStats s_FrameStats;

    // VertexBuffer accessor — vkUIRender::FlushPrimitive needs the VkBuffer handle.
    VkBuffer GetVertexBufferHandle();
    void     FlushVertexBuffer();
}
