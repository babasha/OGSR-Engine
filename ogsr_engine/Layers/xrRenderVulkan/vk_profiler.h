// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).
//
// ============================================================================
//  GLOBAL RENDER PROFILER  (vk_profiler)
// ============================================================================
//  One place that answers "where did the frame go" for the whole Vulkan
//  backend — not per-feature, global. Two layers:
//
//  PHASE 0 — debug-utils labels & object names. The EXT_debug_utils extension
//    is always enabled (g_InstanceExtensions), so we wrap every pass in a named
//    label region and tag key resources. Cost ~0; the payoff is that RenderDoc
//    and Nsight Graphics show OUR passes/resources by name and give pro-grade
//    per-draw GPU timing / flame graphs for free.
//
//  PHASE 1 — in-engine data collection:
//    • GPU timestamp ZONES (one per pass; nested-ready) with a rolling history
//      ring → last / avg / min / max ms per zone.
//    • CPU time per zone (QueryPerformanceCounter), CPU frame time, FPS.
//    • VRAM via VMA (vmaGetHeapBudgets) — used / budget / live allocations.
//    Logged to the engine log every ~5 s as "[VK Perf]" and ON DEMAND via the
//    `vk_perf` console command (a precise MARK snapshot at the instant FPS sits).
//
//  PHASE 2 (vk_profiler_overlay, ImGui) reads the GetZones()/GetMem() getters
//  below to draw the live overlay; nothing here depends on ImGui.
// ============================================================================
#pragma once
#include "stdafx.h"

namespace VK { namespace Prof {

// ---------------------------------------------------------------------------
// Phase 0 — debug-utils (idempotent lazy init; all no-ops if unsupported)
// ---------------------------------------------------------------------------
void Init();   // loads the EXT_debug_utils entry points once (needs VulkanHW set up)
void SetObjectName(VkObjectType type, uint64_t handle, const char* name);
void CmdBeginLabel(VkCommandBuffer cmd, const char* name, float r = 0.6f, float g = 0.6f, float b = 0.9f);
void CmdEndLabel(VkCommandBuffer cmd);

// Typed convenience wrappers for the handles we name most.
void NameImage (VkImage  img, const char* name);
void NameBuffer(VkBuffer buf, const char* name);

// ---------------------------------------------------------------------------
// Phase 1 — per-frame GPU/CPU zones
// ---------------------------------------------------------------------------
constexpr u32 kMaxZones    = 24;   // passes + a little headroom for nesting
constexpr u32 kHistory     = 96;   // samples kept per zone (avg/min/max + graph)

// Called by ExecutePasses. FrameBegin reads back the previous occupant of this
// slot (its fence already proved completion), folds it into the history, then
// resets the slot. ZoneBegin/ZoneEnd bracket one pass (GPU timestamps + CPU QPC
// + a debug label). FrameEnd closes the slot.
void FrameBegin(VkCommandBuffer cmd, u32 frameIndex);
int  ZoneBegin (VkCommandBuffer cmd, const char* name);  // returns a zone id (or -1)
void ZoneEnd   (VkCommandBuffer cmd, int zone);
void FrameEnd  (VkCommandBuffer cmd);

// Once per frame (after FrameBegin's readback): emit the [VK Perf] line if 5 s
// elapsed or a MARK was requested.
void MaybeLog();
void RequestMark();   // `vk_perf` console command → forces the next MaybeLog to print

void Shutdown();

// ---------------------------------------------------------------------------
// Counters — incremented from the backend draw paths (best-effort; see .cpp).
// ---------------------------------------------------------------------------
void CountDraw(u32 instances, u32 tris);
void CountPipelineBind();
void CountDescriptorBind();

// ---------------------------------------------------------------------------
// Read-only getters for the overlay (Phase 2) and any diagnostics.
// ---------------------------------------------------------------------------
struct ZoneStat
{
    char  name[32];
    float gpuLast, gpuAvg, gpuMin, gpuMax;   // ms
    float cpuLast;                           // ms
    float ring[kHistory];                    // gpu ms history (oldest..newest via ringHead)
    u32   ringHead;
};
struct MemSnap   { u64 usedBytes; u64 budgetBytes; u32 allocCount; u32 blockCount; };
struct FrameInfo { float cpuMs; float gpuMs; float fps; u32 draws, instances, tris, pipeBinds, descBinds; };

u32              GetZones(const ZoneStat** outArray);   // returns zone count
MemSnap          GetMem();
FrameInfo        GetFrame();

}} // namespace VK::Prof
