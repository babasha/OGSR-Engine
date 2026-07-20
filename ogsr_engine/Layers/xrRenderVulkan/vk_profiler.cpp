// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).
//
// Global render profiler — see vk_profiler.h for the design overview.

#include "stdafx.h"
#include "vk_profiler.h"
#include "HW_Vulkan.h"           // VulkanHW (device / physdev / allocator / instance)
#include "vk_command_buffer.h"   // VK_FRAMES_IN_FLIGHT

#include "../../xr_3da/device.h"           // Device.fTimeDeltaRealMS / dwTimeGlobal
#include "../../xr_3da/IGame_Persistent.h" // g_pGamePersistent (rain/wet state on MARK)
#include "../../xr_3da/Environment.h"      // CEnvironment::wetness_factor / CurrentEnv
#include "../../xr_3da/stats.h"            // Device.Statistic — main-thread CPU attribution

#include <algorithm>                       // std::sort — top CPU zones in MaybeLog
#include <mutex>                           // checkpoint intern table (workers + main thread)
#include <vector>                          // checkpoint post-mortem query

// r_profiler console var (declared/registered in vk_console_min.cpp): 0 off
// logging, 1 = periodic [VK Perf] log, 2 = + overlay (Phase 2). Default 1.
extern int ps_r_profiler;

namespace VK { namespace Prof {

namespace {

// ---- debug-utils entry points (loaded once; null = unsupported → no-ops) ----
PFN_vkSetDebugUtilsObjectNameEXT  pfnSetName   = nullptr;
PFN_vkCmdBeginDebugUtilsLabelEXT  pfnBeginLbl  = nullptr;
PFN_vkCmdEndDebugUtilsLabelEXT    pfnEndLbl    = nullptr;

// ---- NV diagnostic checkpoints (null = unsupported → no-ops) ----
PFN_vkCmdSetCheckpointNV       pfnSetChk = nullptr;
PFN_vkGetQueueCheckpointDataNV pfnGetChk = nullptr;

// Marker strings must stay valid until the driver is queried after a device
// loss, so checkpoint names are interned into this fixed arena. Uploads record
// from seqParallel workers while the main thread records the frame → the
// lookup is mutex-guarded (a handful of calls per frame; cost is noise).
constexpr u32 kChkMax = 96;
char        s_chkNames[kChkMax][40];
u32         s_chkCount = 0;
std::mutex  s_chkLock;

const char* InternChk(const char* name)
{
    std::lock_guard<std::mutex> lock(s_chkLock);
    for (u32 i = 0; i < s_chkCount; ++i)
        if (0 == strcmp(s_chkNames[i], name)) return s_chkNames[i];
    if (s_chkCount >= kChkMax) return s_chkNames[0];   // arena full: reuse slot 0 (better than nothing)
    xr_strcpy(s_chkNames[s_chkCount], name);
    return s_chkNames[s_chkCount++];
}

bool  s_inited      = false;

// ---- GPU timestamp query pool ----
constexpr u32 kSlots       = VK_FRAMES_IN_FLIGHT;
constexpr u32 kQueriesSlot = kMaxZones * 2;             // begin+end per zone
VkQueryPool   s_pool       = VK_NULL_HANDLE;
float         s_periodNs   = 0.f;                       // timestampPeriod (ns/tick)

// ---- per-slot bookkeeping ----
u32  s_curSlot   = 0;
u32  s_curZoneN  = 0;                 // zones opened this frame
u32  s_zoneDepth = 0;                 // current open-zone nesting depth (0 while between top-level passes)
bool s_slotWritten[kSlots] = {};
u32  s_slotZoneN [kSlots]  = {};

// ---- persistent per-zone stats (index = open order; order is stable/frame) ----
ZoneStat s_zone[kMaxZones]   = {};
u32      s_zoneCount         = 0;     // highest zone count seen
LARGE_INTEGER s_cpuStart[kMaxZones]{};
LARGE_INTEGER s_qpcFreq{};

// ---- counters (reset each frame; snapshot kept for getters) ----
FrameInfo s_frameLast = {};
u32 s_cDraws=0, s_cInst=0, s_cTris=0, s_cPipe=0, s_cDesc=0;

// ---- VRAM ----
MemSnap s_mem = {};
u32     s_deviceLocalHeapMask = 0;

// ---- logging ----
u32  s_lastLogMs = 0;
bool s_markReq   = false;

// ---- CPU phase meters (see CpuPhaseIdx in the header) ----
LARGE_INTEGER s_phaseT0[CPU_PHASE_COUNT]{};
float         s_phaseMs[CPU_PHASE_COUNT]{};

double QpcMs(const LARGE_INTEGER& a, const LARGE_INTEGER& b)
{
    if (s_qpcFreq.QuadPart == 0) return 0.0;
    return double(b.QuadPart - a.QuadPart) * 1000.0 / double(s_qpcFreq.QuadPart);
}

void FoldSample(ZoneStat& z, float gpuMs)
{
    z.gpuLast = gpuMs;
    z.ring[z.ringHead] = gpuMs;
    z.ringHead = (z.ringHead + 1) % kHistory;
    // recompute avg/min/max over the ring (cheap; kHistory is small)
    float sum = 0.f, mn = 1e9f, mx = 0.f; u32 n = 0;
    for (u32 i = 0; i < kHistory; ++i) {
        float v = z.ring[i];
        if (v <= 0.f) continue;
        sum += v; n++;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    z.gpuAvg = n ? sum / n : gpuMs;
    z.gpuMin = n ? mn : gpuMs;
    z.gpuMax = n ? mx : gpuMs;
}

void QueryMemory()
{
    if (!VulkanHW.m_Allocator) return;
    VmaBudget budgets[VK_MAX_MEMORY_HEAPS] = {};
    vmaGetHeapBudgets(VulkanHW.m_Allocator, budgets);
    u64 used = 0, budget = 0; u32 allocs = 0, blocks = 0;
    for (u32 h = 0; h < VK_MAX_MEMORY_HEAPS; ++h) {
        if (!(s_deviceLocalHeapMask & (1u << h))) continue;
        used   += budgets[h].usage;
        budget += budgets[h].budget;
        allocs += budgets[h].statistics.allocationCount;
        blocks += budgets[h].statistics.blockCount;
    }
    s_mem.usedBytes   = used;
    s_mem.budgetBytes = budget;
    s_mem.allocCount  = allocs;
    s_mem.blockCount  = blocks;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Phase 0 — debug-utils
// ---------------------------------------------------------------------------
void Init()
{
    if (s_inited) return;
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;   // not ready yet; retried next frame
    s_inited = true;

    QueryPerformanceFrequency(&s_qpcFreq);

    pfnSetName  = (PFN_vkSetDebugUtilsObjectNameEXT) vkGetDeviceProcAddr(VulkanHW.m_Device, "vkSetDebugUtilsObjectNameEXT");
    pfnBeginLbl = (PFN_vkCmdBeginDebugUtilsLabelEXT) vkGetDeviceProcAddr(VulkanHW.m_Device, "vkCmdBeginDebugUtilsLabelEXT");
    pfnEndLbl   = (PFN_vkCmdEndDebugUtilsLabelEXT)   vkGetDeviceProcAddr(VulkanHW.m_Device, "vkCmdEndDebugUtilsLabelEXT");

    if (VulkanHW.m_bCheckpointsSupported) {
        pfnSetChk = (PFN_vkCmdSetCheckpointNV)       vkGetDeviceProcAddr(VulkanHW.m_Device, "vkCmdSetCheckpointNV");
        pfnGetChk = (PFN_vkGetQueueCheckpointDataNV) vkGetDeviceProcAddr(VulkanHW.m_Device, "vkGetQueueCheckpointDataNV");
        Msg("[VK Prof] NV checkpoints %s", (pfnSetChk && pfnGetChk) ? "armed" : "proc-addr FAILED");
    }

    // device-local heap mask for VRAM accounting
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(VulkanHW.m_PhysicalDevice, &mp);
    for (u32 h = 0; h < mp.memoryHeapCount; ++h)
        if (mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
            s_deviceLocalHeapMask |= (1u << h);

    // timestamp query pool
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(VulkanHW.m_PhysicalDevice, &props);
    s_periodNs = props.limits.timestampPeriod;
    if (s_periodNs > 0.f) {
        VkQueryPoolCreateInfo qci{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
        qci.queryType  = VK_QUERY_TYPE_TIMESTAMP;
        qci.queryCount = kQueriesSlot * kSlots;
        if (vkCreateQueryPool(VulkanHW.m_Device, &qci, nullptr, &s_pool) != VK_SUCCESS)
            s_pool = VK_NULL_HANDLE;
    }
    Msg("[VK Perf] profiler init (timestamps=%s, period=%.2f ns)",
        s_pool ? "on" : "off", s_periodNs);
}

void SetObjectName(VkObjectType type, uint64_t handle, const char* name)
{
    if (!s_inited) Init();   // names are often set at resource-creation (pre-frame)
    if (!pfnSetName || !handle || !name) return;
    VkDebugUtilsObjectNameInfoEXT ni{ VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT };
    ni.objectType   = type;
    ni.objectHandle = handle;
    ni.pObjectName  = name;
    pfnSetName(VulkanHW.m_Device, &ni);
}

void NameImage (VkImage  img, const char* name) { SetObjectName(VK_OBJECT_TYPE_IMAGE,  (uint64_t)img, name); }
void NameBuffer(VkBuffer buf, const char* name) { SetObjectName(VK_OBJECT_TYPE_BUFFER, (uint64_t)buf, name); }
void NameSet   (VkDescriptorSet s, const char* name) { SetObjectName(VK_OBJECT_TYPE_DESCRIPTOR_SET, (uint64_t)s, name); }

void CmdBeginLabel(VkCommandBuffer cmd, const char* name, float r, float g, float b)
{
    if (!pfnBeginLbl) return;
    VkDebugUtilsLabelEXT lbl{ VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT };
    lbl.pLabelName = name;
    lbl.color[0] = r; lbl.color[1] = g; lbl.color[2] = b; lbl.color[3] = 1.f;
    pfnBeginLbl(cmd, &lbl);
}

void CmdEndLabel(VkCommandBuffer cmd)
{
    if (pfnEndLbl) pfnEndLbl(cmd);
}

// Receiver for the pure.h seqFrame hook: fold each member's wall time into a
// "seq:<class>" CPU probe (prints in the 5s [VK CPUprobes] line). Members under
// 0.02 ms are skipped so the 48 probe slots go to the real spenders.
static void SeqFrameProbeCb(const char* typeName, float ms)
{
    if (ms < 0.02f) return;
    const char* p = typeName ? typeName : "?";
    if      (0 == strncmp(p, "class ",  6)) p += 6;
    else if (0 == strncmp(p, "struct ", 7)) p += 7;
    char nm[32];
    _snprintf(nm, sizeof(nm) - 1, "seq:%s", p);
    nm[sizeof(nm) - 1] = 0;
    const int slot = CpuProbeSlot(nm);
    if (slot >= 0) CpuProbeAdd(slot, ms);
}

// ---------------------------------------------------------------------------
// Phase 1 — per-frame zones
// ---------------------------------------------------------------------------
void FrameBegin(VkCommandBuffer cmd, u32 frameIndex)
{
    Init();
    Checkpoint(cmd, "FrameBegin");   // marks even zone-less frames (loading screens)
    s_curSlot  = frameIndex % kSlots;
    s_curZoneN = 0;
    s_zoneDepth = 0;   // defensive: rebalance in case a zone span was left open last frame

    // Ask the engine to gather CStatTimer stats (fills [VK CPU] in MaybeLog) even
    // without rs_stats — the rs_stats overlay needs fonts the VK build never creates.
    // device.cpp reads this next frame; the ~µs QPC cost is noise.
    g_bForceStatGather = (ps_r_profiler > 0);
    // Per-member seqFrame attribution (pure.h hook) → "seq:<class>" CPU probes.
    // Diagnoses EngineTOTAL growing while every engine sub-timer reads 0 (the
    // "fps decays after flying, GPU underloaded" pattern — WHO in seqFrame grew).
    g_seq_profile_hook = (ps_r_profiler > 0) ? &SeqFrameProbeCb : nullptr;

    // snapshot + reset CPU-side counters for getters
    s_frameLast.draws     = s_cDraws;     s_cDraws = 0;
    s_frameLast.instances = s_cInst;      s_cInst  = 0;
    s_frameLast.tris      = s_cTris;      s_cTris  = 0;
    s_frameLast.pipeBinds = s_cPipe;      s_cPipe  = 0;
    s_frameLast.descBinds = s_cDesc;      s_cDesc  = 0;

    // read back the PREVIOUS occupant of this slot (its fence already proved the
    // GPU finished that submission), fold into the per-zone history.
    if (s_pool && s_slotWritten[s_curSlot]) {
        const u32 base = s_curSlot * kQueriesSlot;
        const u32 nz   = s_slotZoneN[s_curSlot];
        u64 q[kQueriesSlot] = {};
        if (nz && vkGetQueryPoolResults(VulkanHW.m_Device, s_pool, base, nz * 2,
                                        sizeof(u64) * nz * 2, q, sizeof(u64),
                                        VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
            for (u32 z = 0; z < nz; ++z) {
                const float ms = float(double(q[z * 2 + 1] - q[z * 2]) * s_periodNs * 1e-6);
                if (ms >= 0.f && ms < 1000.f) FoldSample(s_zone[z], ms);
            }
        }
    }

    // reset this slot's range (must be outside a render pass — we are)
    if (s_pool)
        vkCmdResetQueryPool(cmd, s_pool, s_curSlot * kQueriesSlot, kQueriesSlot);

    QueryMemory();
}

// ---------------------------------------------------------------------------
// NV diagnostic checkpoints
// ---------------------------------------------------------------------------
void Checkpoint(VkCommandBuffer cmd, const char* name)
{
    if (!pfnSetChk || cmd == VK_NULL_HANDLE) return;
    pfnSetChk(cmd, InternChk(name));
}

void DumpCheckpoints(const char* why)
{
    static bool s_dumped = false;   // one dump per device loss is enough (SL floods errors after)
    if (s_dumped) return;
    s_dumped = true;

    if (!pfnGetChk) { Msg("![VK Chk] (%s) NV checkpoints unsupported — no GPU post-mortem", why); return; }

    struct { const char* name; VkQueue q; } queues[] = {
        { "graphics", VulkanHW.m_GraphicsQueue },
        { "transfer", VulkanHW.m_TransferQueue },
        { "compute",  VulkanHW.m_ComputeQueue  },
    };
    Msg("![VK Chk] ===== GPU post-mortem (%s) — hang is between COMPLETED and STARTED =====", why);
    for (u32 qi = 0; qi < 3; ++qi) {
        if (queues[qi].q == VK_NULL_HANDLE) continue;
        bool dup = false;   // families often alias to one VkQueue — dump each handle once
        for (u32 j = 0; j < qi; ++j) if (queues[j].q == queues[qi].q) { dup = true; break; }
        if (dup) continue;

        u32 n = 0;
        pfnGetChk(queues[qi].q, &n, nullptr);
        if (n == 0) { Msg("![VK Chk] %s queue: no checkpoint data", queues[qi].name); continue; }
        std::vector<VkCheckpointDataNV> data(n);
        for (auto& d : data) { d.sType = VK_STRUCTURE_TYPE_CHECKPOINT_DATA_NV; d.pNext = nullptr; }
        pfnGetChk(queues[qi].q, &n, data.data());
        for (u32 i = 0; i < n; ++i) {
            const char* marker = (const char*)data[i].pCheckpointMarker;
            // Only our interned pointers are trustworthy strings.
            const bool ours = marker >= &s_chkNames[0][0] && marker < &s_chkNames[0][0] + sizeof(s_chkNames);
            const char* when = (data[i].stage == VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT) ? "COMPLETED"
                             : (data[i].stage == VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT)    ? "STARTED  "
                                                                                       : "stage?   ";
            if (ours) Msg("![VK Chk] %s queue: last %s = '%s'", queues[qi].name, when, marker);
            else      Msg("![VK Chk] %s queue: last %s = %p (foreign)", queues[qi].name, when, marker);
        }
    }
    Msg("![VK Chk] ===== end post-mortem =====");
}

int ZoneBegin(VkCommandBuffer cmd, const char* name)
{
    Checkpoint(cmd, name);   // GPU progress marker (post-mortem on device loss)

    const int z = (int)s_curZoneN;
    if (z >= (int)kMaxZones) { CmdBeginLabel(cmd, name); return -1; }  // overflow: label only

    // stable name for this slot index
    xr_strcpy(s_zone[z].name, name);
    s_zone[z].depth = s_zoneDepth++;   // 0 = top-level pass; nested children are excluded from gpu_total
    QueryPerformanceCounter(&s_cpuStart[z]);

    if (s_pool)
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, s_pool,
                            s_curSlot * kQueriesSlot + z * 2);
    CmdBeginLabel(cmd, name);

    s_curZoneN++;
    if (s_curZoneN > s_zoneCount) s_zoneCount = s_curZoneN;
    return z;
}

void ZoneEnd(VkCommandBuffer cmd, int zone)
{
    if (zone < 0) { CmdEndLabel(cmd); return; }   // overflow zone (label only; never bumped s_zoneDepth)
    if (s_zoneDepth) s_zoneDepth--;
    CmdEndLabel(cmd);
    if (s_pool)
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, s_pool,
                            s_curSlot * kQueriesSlot + zone * 2 + 1);
    LARGE_INTEGER now; QueryPerformanceCounter(&now);
    s_zone[zone].cpuLast = (float)QpcMs(s_cpuStart[zone], now);
}

// CPU probes (see header): named per-frame wall-time accumulators for main-
// thread code the pass zones can't see. Harvested + reset in FrameEnd.
namespace {
    constexpr u32 kMaxProbes = 48;
    struct Probe { char name[32]; float accum; float last; };
    Probe s_probes[kMaxProbes];
    u32   s_probeCount = 0;
}

void FrameEnd(VkCommandBuffer cmd)
{
    Checkpoint(cmd, "FrameEnd");   // a hang AFTER the last pass (present chain) still shows
    s_slotWritten[s_curSlot] = true;
    s_slotZoneN [s_curSlot]  = s_curZoneN;

    // Harvest this frame's CPU probes (streamers/queue-rebuilds/near-set walks).
    for (u32 i = 0; i < s_probeCount; ++i) {
        s_probes[i].last  = s_probes[i].accum;
        s_probes[i].accum = 0.f;
    }
}

void CpuPhaseBegin(u32 idx)
{
    if (idx < CPU_PHASE_COUNT) QueryPerformanceCounter(&s_phaseT0[idx]);
}

void CpuPhaseEnd(u32 idx)
{
    if (idx >= CPU_PHASE_COUNT) return;
    LARGE_INTEGER now; QueryPerformanceCounter(&now);
    s_phaseMs[idx] = (float)QpcMs(s_phaseT0[idx], now);
}

// ---------------------------------------------------------------------------
// CPU probes API (statics live above FrameEnd, which harvests them).
// ---------------------------------------------------------------------------
int CpuProbeSlot(const char* name)
{
    for (u32 i = 0; i < s_probeCount; ++i)
        if (0 == strcmp(s_probes[i].name, name))
            return (int)i;
    if (s_probeCount >= kMaxProbes) return -1;
    xr_strcpy(s_probes[s_probeCount].name, name);
    s_probes[s_probeCount].accum = s_probes[s_probeCount].last = 0.f;
    return (int)s_probeCount++;
}

void CpuProbeAdd(int slot, float ms)
{
    if (slot >= 0 && slot < (int)s_probeCount)
        s_probes[slot].accum += ms;
}

CpuProbeScope::CpuProbeScope(int s) : slot(s)
{
    LARGE_INTEGER t; QueryPerformanceCounter(&t);
    t0 = (u64)t.QuadPart;
}

CpuProbeScope::~CpuProbeScope()
{
    LARGE_INTEGER a, now;
    a.QuadPart = (LONGLONG)t0;
    QueryPerformanceCounter(&now);
    CpuProbeAdd(slot, (float)QpcMs(a, now));
}

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------
void RequestMark() { s_markReq = true; }

void MaybeLog()
{
    const bool mark = s_markReq;
    if (ps_r_profiler <= 0 && !mark) { s_markReq = false; return; }

    const u32 nowMs = Device.dwTimeGlobal;
    if (!mark && nowMs < s_lastLogMs + 5000) return;
    s_lastLogMs = nowMs;
    s_markReq   = false;

    char line[2048]; int off = 0;
    float gpuTotal = 0.f;
    for (u32 z = 0; z < s_zoneCount; ++z) {
        // gpu_total = wall-clock frame GPU time ≈ sum of TOP-LEVEL passes only.
        // Nested zones (World/*, Shadow/*) are a breakdown of their parent, so
        // adding them would double-count; they're still printed individually.
        if (s_zone[z].depth == 0) gpuTotal += s_zone[z].gpuLast;
        off += _snprintf(line + off, sizeof(line) - off - 1, "%s=%.2f(%.2f/%.2f) ",
                         s_zone[z].name, s_zone[z].gpuLast, s_zone[z].gpuMin, s_zone[z].gpuMax);
        if (off > (int)sizeof(line) - 64) break;
    }
    line[off > 0 ? off : 0] = 0;

    const float cpuMs = Device.fTimeDeltaRealMS;
    const float fps   = cpuMs > 0.01f ? 1000.f / cpuMs : 0.f;
    const double usedMB = double(s_mem.usedBytes)   / (1024.0 * 1024.0);
    const double budMB  = double(s_mem.budgetBytes) / (1024.0 * 1024.0);

    Msg("[VK Perf]%s cpu=%.2fms fps=%.0f gpu_total=%.2fms | %s| VRAM %.0f/%.0f MB (alloc %u/%u blk)",
        mark ? " MARK" : "", cpuMs, fps, gpuTotal, line, usedMB, budMB, s_mem.allocCount, s_mem.blockCount);

    // VRAM attribution (Stage E-1): who holds the memory. Compact line with the
    // 5s dump; full sorted table at r_profiler>=3 or on an explicit mark.
    if (mark || ps_r_profiler >= 3)
        VK::Vram::LogFull();
    else
        VK::Vram::LogCompact();

    // Main-thread CPU attribution: the engine's own subsystem timers (same data as
    // rs_stats, but into the log). On vistas the frame is CPU-bound on ONE core
    // (cpu >> gpu while task-manager shows ~10% total) — this line says WHERE.
    if (Device.Statistic) {
        const CStats& S = *Device.Statistic;
        Msg("[VK CPU] engine=%.2f sched=%.2f updateCL=%.2f render=%.2f/%.2f (calc=%.2f dump=%.2f wait=%.2f) anim=%.2f phys=%.2f ai_think=%.2f ai_vis=%.2f snd=%.2f ray=%.2f | dtGame=%.4fms tf=%.3f",
            S.EngineTOTAL.result, S.Sheduler.result, S.UpdateClient.result,
            S.RenderTOTAL.result, S.RenderTOTAL_Real.result,
            S.RenderCALC.result, S.RenderDUMP.result, S.RenderDUMP_Wait.result,
            S.Animation.result, S.Physics.result, S.AI_Think.result, S.AI_Vis.result,
            S.Sound.result, S.clRAY.result, Device.fTimeDelta * 1000.f, Device.time_factor());
        Msg("[VK CPU objs] crows=%u updated=%u active=%u total=%u rays=%u", S.UpdateClient_crows, S.UpdateClient_updated, S.UpdateClient_active, S.UpdateClient_total,
            S.clRAY.count);
    }

    // CPU phases of the render section: real work (calc/record) vs GPU-blocking
    // waits (beginWait = fence+acquire, submitPresent can block on queue/vsync).
    Msg("[VK CPUphase] beginWait=%.2f calc=%.2f record=%.2f submitPresent=%.2f (sum=%.2f)",
        s_phaseMs[CPU_BEGIN], s_phaseMs[CPU_CALC], s_phaseMs[CPU_RECORD], s_phaseMs[CPU_END],
        s_phaseMs[CPU_BEGIN] + s_phaseMs[CPU_CALC] + s_phaseMs[CPU_RECORD] + s_phaseMs[CPU_END]);

    // Renderer-side CPU: top zones by command-recording/collect time this frame
    // (ZoneBegin..ZoneEnd wall time on the render thread, NOT GPU time).
    {
        u32 order[kMaxZones]; u32 n = 0;
        for (u32 z = 0; z < s_zoneCount; ++z) if (s_zone[z].cpuLast > 0.15f) order[n++] = z;
        std::sort(order, order + n, [](u32 a, u32 b) { return s_zone[a].cpuLast > s_zone[b].cpuLast; });
        if (n > 8) n = 8;
        char cl[512]; int co = 0;
        for (u32 i = 0; i < n; ++i)
            co += _snprintf(cl + co, sizeof(cl) - co - 1, "%s=%.2f ", s_zone[order[i]].name, s_zone[order[i]].cpuLast);
        cl[co > 0 ? co : 0] = 0;
        if (n) Msg("[VK CPUzones] %s", cl);
    }

    // CPU probes — helpers OUTSIDE the pass zones (streamer ticks, caster-queue
    // rebuilds, tree near-set walk, grass update...). Same top-by-ms format.
    {
        u32 order[kMaxProbes]; u32 n = 0;
        for (u32 p = 0; p < s_probeCount; ++p) if (s_probes[p].last > 0.05f) order[n++] = p;
        std::sort(order, order + n, [](u32 a, u32 b) { return s_probes[a].last > s_probes[b].last; });
        if (n > 12) n = 12;
        char cl[512]; int co = 0;
        for (u32 i = 0; i < n; ++i)
            co += _snprintf(cl + co, sizeof(cl) - co - 1, "%s=%.2f ", s_probes[order[i]].name, s_probes[order[i]].last);
        cl[co > 0 ? co : 0] = 0;
        if (n) Msg("[VK CPUprobes] %s", cl);
    }

    // On an explicit MARK, also stamp the weather/wet state so the snapshot is
    // self-describing ("was it raining / had puddles formed when FPS sat?").
    if (mark && g_pGamePersistent) {
        const auto& env = g_pGamePersistent->Environment();
        float density = (env.CurrentEnv ? env.CurrentEnv->rain_density : 0.f);
        Msg("[VK Perf] MARK  rain_density=%.2f wetness=%.2f", density, env.wetness_factor);
    }
}

// ---------------------------------------------------------------------------
// Counters / getters
// ---------------------------------------------------------------------------
void CountDraw(u32 instances, u32 tris) { s_cDraws++; s_cInst += instances; s_cTris += tris; }
void CountPipelineBind()   { s_cPipe++; }
void CountDescriptorBind() { s_cDesc++; }

u32 GetZones(const ZoneStat** outArray) { if (outArray) *outArray = s_zone; return s_zoneCount; }
MemSnap GetMem() { return s_mem; }

FrameInfo GetFrame()
{
    float gpu = 0.f;
    for (u32 z = 0; z < s_zoneCount; ++z) if (s_zone[z].depth == 0) gpu += s_zone[z].gpuLast;   // top-level only (see MaybeLog)
    s_frameLast.gpuMs = gpu;
    s_frameLast.cpuMs = Device.fTimeDeltaRealMS;
    s_frameLast.fps   = s_frameLast.cpuMs > 0.01f ? 1000.f / s_frameLast.cpuMs : 0.f;
    return s_frameLast;
}

void Shutdown()
{
    if (s_pool && VulkanHW.m_Device) {
        vkDestroyQueryPool(VulkanHW.m_Device, s_pool, nullptr);
        s_pool = VK_NULL_HANDLE;
    }
    for (bool& w : s_slotWritten) w = false;
    s_inited = false;
    pfnSetName = nullptr; pfnBeginLbl = nullptr; pfnEndLbl = nullptr;
}

}} // namespace VK::Prof
