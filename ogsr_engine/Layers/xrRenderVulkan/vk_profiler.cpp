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

// r_profiler console var (declared/registered in vk_console_min.cpp): 0 off
// logging, 1 = periodic [VK Perf] log, 2 = + overlay (Phase 2). Default 1.
extern int ps_r_profiler;

namespace VK { namespace Prof {

namespace {

// ---- debug-utils entry points (loaded once; null = unsupported → no-ops) ----
PFN_vkSetDebugUtilsObjectNameEXT  pfnSetName   = nullptr;
PFN_vkCmdBeginDebugUtilsLabelEXT  pfnBeginLbl  = nullptr;
PFN_vkCmdEndDebugUtilsLabelEXT    pfnEndLbl    = nullptr;

bool  s_inited      = false;

// ---- GPU timestamp query pool ----
constexpr u32 kSlots       = VK_FRAMES_IN_FLIGHT;
constexpr u32 kQueriesSlot = kMaxZones * 2;             // begin+end per zone
VkQueryPool   s_pool       = VK_NULL_HANDLE;
float         s_periodNs   = 0.f;                       // timestampPeriod (ns/tick)

// ---- per-slot bookkeeping ----
u32  s_curSlot   = 0;
u32  s_curZoneN  = 0;                 // zones opened this frame
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

// ---------------------------------------------------------------------------
// Phase 1 — per-frame zones
// ---------------------------------------------------------------------------
void FrameBegin(VkCommandBuffer cmd, u32 frameIndex)
{
    Init();
    s_curSlot  = frameIndex % kSlots;
    s_curZoneN = 0;

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

int ZoneBegin(VkCommandBuffer cmd, const char* name)
{
    const int z = (int)s_curZoneN;
    if (z >= (int)kMaxZones) { CmdBeginLabel(cmd, name); return -1; }  // overflow: label only

    // stable name for this slot index
    xr_strcpy(s_zone[z].name, name);
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
    if (zone < 0) { CmdEndLabel(cmd); return; }   // overflow zone (label only)
    CmdEndLabel(cmd);
    if (s_pool)
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, s_pool,
                            s_curSlot * kQueriesSlot + zone * 2 + 1);
    LARGE_INTEGER now; QueryPerformanceCounter(&now);
    s_zone[zone].cpuLast = (float)QpcMs(s_cpuStart[zone], now);
}

void FrameEnd(VkCommandBuffer /*cmd*/)
{
    s_slotWritten[s_curSlot] = true;
    s_slotZoneN [s_curSlot]  = s_curZoneN;
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

    char line[1024]; int off = 0;
    float gpuTotal = 0.f;
    for (u32 z = 0; z < s_zoneCount; ++z) {
        gpuTotal += s_zone[z].gpuLast;
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
    for (u32 z = 0; z < s_zoneCount; ++z) gpu += s_zone[z].gpuLast;
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
