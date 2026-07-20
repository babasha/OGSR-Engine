// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan - Async compute foundation (vk_async).
//
// The renderer records the WHOLE frame into one graphics command buffer and
// submits it once to the graphics queue. To overlap compute with graphics we need
// a SECOND command buffer recorded + submitted to the dedicated compute queue, with
// a timeline semaphore ordering the cross-queue handoff. This module owns that
// machinery: a compute-family command pool, one command buffer per frame-in-flight,
// and a monotonic timeline the graphics submit waits on.
//
// INCREMENT 1 (this file's current scope) = the machinery + an INERT probe: each
// frame it records an EMPTY compute command buffer, submits it to the compute queue
// signaling the timeline, and the graphics submit waits on it. Purpose: prove the
// cross-queue submit + timeline are STABLE on the target GPU (no hang / device-lost)
// before any real pass moves onto the compute queue. r_async 0 = the machinery is
// dormant (exact old single-queue path).
//
// INCREMENT 2 (next, once the probe is verified stable): record a real pass into
// GetCmd() instead of leaving it empty — starting with one whose barriers are
// rewritten for the compute queue (the existing culls/SSAO carry graphics-stage
// barriers that are invalid on a compute queue) and whose shared buffers are made
// CONCURRENT (graphics+compute). Then split the graphics frame into segments so the
// compute actually OVERLAPS graphics raster (e.g. SSAO behind the VSM atlas render).

#pragma once
#include "vk_core.h"

namespace VK { namespace Async {

// True when async compute is enabled (r_async) AND the device exposes a compute
// queue family distinct from graphics (else there is nothing to overlap). Lazily
// initializes the pool/timeline on first true.
bool Available();

// Record + submit this frame's async compute work to the compute queue, signaling
// the timeline. v1 = inert (empty command buffer). `slot` = the frame-in-flight
// index (CommandManager::GetCurrentFrame()); its command buffer is safe to reset
// because the graphics frame that used it last has retired (its fence gated reuse).
void FrameSubmit(u32 slot);

// For the graphics submit (CommandManager::Submit): if async work was submitted
// this frame, yields the timeline semaphore + value the graphics queue must wait
// on before consuming any compute output. Mirrors the upload-timeline wait.
bool GetGraphicsWait(VkSemaphore& sem, u64& value);

// Device teardown.
void Destroy();

}} // namespace VK::Async
