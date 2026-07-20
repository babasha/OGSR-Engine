// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

#include "stdafx.h"
#include "vk_texture_stream.h"
#include "HW_Vulkan.h"
#include "vk_command_buffer.h"
#include "vk_profiler.h"   // VK_CPU_PROBE — per-frame CPU attribution

#include <mutex>
#include <algorithm>
#include <vector>
#include <cmath>

// Console cvars (defined in vk_console_min.cpp).
extern int psTextureLOD;              // manual quality slider 0..3 (WorldDiffuse only)
extern int ps_r_txstream;             // 1 = dynamic per-frame mip streaming (opt-in)
extern int ps_r_txstream_budget;      // texture VRAM budget in MB (0 = auto)
extern int ps_r_txstream_headroom;    // VRAM safety margin left free at load, MB
extern int ps_r_txstream_reserve;     // MB of device VRAM to keep free after load (FG room)

namespace VK {

// ---------------------------------------------------------------------------
// Format helpers (self-contained mirror of CVulkanTexture's block logic — the
// streamer must size mip chains without touching the private texture members).
// ---------------------------------------------------------------------------
namespace {
    std::mutex s_Mutex;   // guards the singleton's map + byte counters

    bool IsCompressed(VkFormat f)
    {
        switch (f) {
            case VK_FORMAT_BC1_RGB_UNORM_BLOCK:  case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
            case VK_FORMAT_BC1_RGBA_UNORM_BLOCK: case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
            case VK_FORMAT_BC2_UNORM_BLOCK:      case VK_FORMAT_BC2_SRGB_BLOCK:
            case VK_FORMAT_BC3_UNORM_BLOCK:      case VK_FORMAT_BC3_SRGB_BLOCK:
            case VK_FORMAT_BC4_UNORM_BLOCK:      case VK_FORMAT_BC4_SNORM_BLOCK:
            case VK_FORMAT_BC5_UNORM_BLOCK:      case VK_FORMAT_BC5_SNORM_BLOCK:
            case VK_FORMAT_BC6H_UFLOAT_BLOCK:    case VK_FORMAT_BC6H_SFLOAT_BLOCK:
            case VK_FORMAT_BC7_UNORM_BLOCK:      case VK_FORMAT_BC7_SRGB_BLOCK:
                return true;
            default: return false;
        }
    }

    u32 BlockSize(VkFormat f)
    {
        switch (f) {
            case VK_FORMAT_BC1_RGB_UNORM_BLOCK:  case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
            case VK_FORMAT_BC1_RGBA_UNORM_BLOCK: case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
            case VK_FORMAT_BC4_UNORM_BLOCK:      case VK_FORMAT_BC4_SNORM_BLOCK:
                return 8;
            default: return 16;   // BC2/3/5/6H/7
        }
    }

    u32 BppUncompressed(VkFormat f)
    {
        if (f == VK_FORMAT_R8_UNORM)   return 1;
        if (f == VK_FORMAT_R8G8_UNORM) return 2;
        return 4;   // RGBA8 / BGRA8
    }

    VkDeviceSize LevelBytes(u32 w, u32 h, VkFormat f)
    {
        if (IsCompressed(f))
            return (VkDeviceSize)((w + 3) / 4) * ((h + 3) / 4) * BlockSize(f);
        return (VkDeviceSize)w * h * BppUncompressed(f);
    }

    // Largest base mip a texture may be demoted to while keeping the base
    // dimension >= floorDim (and at least one mip resident).
    u32 MaxBaseForDims(u32 fullW, u32 fullH, u32 fullMips, u32 floorDim)
    {
        u32 mb = 0, w = fullW, h = fullH;
        for (u32 b = 0; b + 1 < fullMips; ++b) {
            u32 nw = (w > 1) ? (w >> 1) : 1, nh = (h > 1) ? (h >> 1) : 1;
            if ((nw > nh ? nw : nh) < floorDim) break;
            mb = b + 1; w = nw; h = nh;
        }
        return mb;
    }

    // GPU feedback LOD decode — mirror of the shader encode in tex_feedback.glsl:
    // enc = clamp((lod + 16) * 4, 0, 4095), lod RELATIVE to the resident base.
    float DecodeFbLod(u32 enc) { return (float)enc * 0.25f - 16.0f; }

    const char* KlassName(TexStreamClass k)
    {
        switch (k) {
            case TexStreamClass::UI:           return "UI/other";
            case TexStreamClass::WorldDiffuse: return "WorldDiffuse";
            case TexStreamClass::Detail:       return "Detail";
            case TexStreamClass::Lmap:         return "Lmap";
            case TexStreamClass::Terrain:      return "Terrain";
            case TexStreamClass::Bump:         return "Bump";
            default:                            return "?";
        }
    }

    // Reusable free space inside VMA's cached DEVICE_LOCAL blocks: freed
    // sub-allocations whose VkDeviceMemory the driver still reports as used but
    // which VMA hands to new allocations without growing driver usage.
    VkDeviceSize VmaFreeInBlocks()
    {
        if (VulkanHW.m_Allocator == VK_NULL_HANDLE) return 0;
        VmaBudget heaps[VK_MAX_MEMORY_HEAPS]{};
        vmaGetHeapBudgets(VulkanHW.m_Allocator, heaps);
        const VkPhysicalDeviceMemoryProperties* mp = nullptr;
        vmaGetMemoryProperties(VulkanHW.m_Allocator, &mp);
        VkDeviceSize free = 0;
        if (mp)
            for (u32 i = 0; i < mp->memoryHeapCount; ++i)
                if ((mp->memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) &&
                    heaps[i].statistics.blockBytes > heaps[i].statistics.allocationBytes)
                    free += heaps[i].statistics.blockBytes - heaps[i].statistics.allocationBytes;
        return free;
    }
}

VkDeviceSize TextureStreamer::MipChainBytes(u32 w, u32 h, u32 fullMips, u32 base, VkFormat fmt)
{
    VkDeviceSize total = 0;
    u32 cw = w, ch = h;
    for (u32 i = 0; i < fullMips; ++i) {
        if (i >= base) total += LevelBytes(cw, ch, fmt);
        if (cw > 1) cw >>= 1;
        if (ch > 1) ch >>= 1;
    }
    return total;
}

TextureStreamer& TextureStreamer::Instance()
{
    static TextureStreamer s_inst;
    return s_inst;
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------
void TextureStreamer::Register(CVulkanTexture* tex, const char* file, u32 fullW, u32 fullH,
                               u32 fullMips, VkFormat fmt, u32 residentBase,
                               VkDeviceSize residentBytes, TexStreamClass klass)
{
    if (!tex) return;
    std::lock_guard<std::mutex> lk(s_Mutex);

    StreamTexture st;
    st.tex           = tex;
    st.file          = file;
    st.fullW         = fullW;
    st.fullH         = fullH;
    st.fullMips      = fullMips;
    st.format        = fmt;
    st.residentBase  = residentBase;
    st.wantedBase    = residentBase;
    st.residentBytes = residentBytes;
    st.lastUsedFrame = Device.dwFrame;
    st.klass         = klass;
    // Only base diffuse of world/model geometry is eligible for dynamic mip streaming:
    // detail/lmap/terrain/bump are tiled/low-res and their descriptor bindings live in
    // multi-texture sets the rebind hook doesn't cover; UI must stay crisp.
    st.streamable    = (klass == TexStreamClass::WorldDiffuse) && (fullMips > 1) && (file && file[0]);

    // Streamables get a compact GPU-feedback slot (the shaders atomicMin the
    // desired LOD there). Slots only exist once the feedback buffer does — before
    // that (or past capacity) the texture simply stays on the LRU fallback.
    if (st.streamable && m_FbDevice != VK_NULL_HANDLE) {
        if (!m_FreeSlots.empty()) { st.fbSlot = m_FreeSlots.back(); m_FreeSlots.pop_back(); }
        else if (m_NextSlot < kFeedbackSlots) st.fbSlot = m_NextSlot++;
        if (st.fbSlot != 0xFFFFFFFFu) {
            if (m_SlotOwner.size() < kFeedbackSlots) m_SlotOwner.resize(kFeedbackSlots, nullptr);
            m_SlotOwner[st.fbSlot] = tex;
        }
    }

    m_Tex[tex] = st;
    m_TrackedBytes += residentBytes;
    if (residentBase > 0) ++m_LoadCapCount;   // tallied for the end-of-load report
}

void TextureStreamer::Unregister(CVulkanTexture* tex)
{
    if (!tex) return;
    std::lock_guard<std::mutex> lk(s_Mutex);
    auto it = m_Tex.find(tex);
    if (it == m_Tex.end()) return;
    if (it->second.fbSlot != 0xFFFFFFFFu) {
        m_FreeSlots.push_back(it->second.fbSlot);
        if (it->second.fbSlot < m_SlotOwner.size()) m_SlotOwner[it->second.fbSlot] = nullptr;
    }
    if (m_TrackedBytes >= it->second.residentBytes) m_TrackedBytes -= it->second.residentBytes;
    else                                            m_TrackedBytes = 0;
    m_Tex.erase(it);
}

void TextureStreamer::SetStreamable(CVulkanTexture* tex, bool enable)
{
    if (!tex) return;
    std::lock_guard<std::mutex> lk(s_Mutex);
    auto it = m_Tex.find(tex);
    if (it == m_Tex.end()) return;
    StreamTexture& s = it->second;
    if (!enable && s.fbSlot != 0xFFFFFFFFu) {
        m_FreeSlots.push_back(s.fbSlot);
        if (s.fbSlot < m_SlotOwner.size()) m_SlotOwner[s.fbSlot] = nullptr;
        s.fbSlot = 0xFFFFFFFFu;
    }
    s.streamable = enable && (s.fullMips > 1) && s.file.size();
}

u32 TextureStreamer::GetFeedbackSlot(CVulkanTexture* tex) const
{
    if (!tex) return 0xFFFFFFFFu;
    std::lock_guard<std::mutex> lk(s_Mutex);
    auto it = m_Tex.find(tex);
    return (it != m_Tex.end()) ? it->second.fbSlot : 0xFFFFFFFFu;
}

void TextureStreamer::EnsureMinResidency(CVulkanTexture* tex, u32 minDim)
{
    if (!tex) return;
    std::vector<std::pair<StreamTexture*, u32>> plan;
    {
        std::lock_guard<std::mutex> lk(s_Mutex);
        auto it = m_Tex.find(tex);
        if (it == m_Tex.end()) return;
        StreamTexture& s = it->second;
        if (s.file.size() == 0 || s.fullMips <= 1) return;
        const u32 floorB = MaxBaseForDims(s.fullW, s.fullH, s.fullMips, minDim);
        if (s.residentBase <= floorB) return;   // already at or above the floor
        s.wantedBase = floorB;
        plan.emplace_back(&s, floorB);
    }
    ApplyResidencyPlan(plan);
}

// ---------------------------------------------------------------------------
// GPU feedback buffers (UE-VT style "sampler feedback lite")
// ---------------------------------------------------------------------------
bool TextureStreamer::EnsureFeedbackGpu()
{
    VK::Vram::Scope _vram_scope("TexStream");
    if (m_FbDevice != VK_NULL_HANDLE) return true;
    if (m_FbTried) return false;
    m_FbTried = true;
    if (VulkanHW.m_Allocator == VK_NULL_HANDLE) return false;

    constexpr VkDeviceSize kSlotBytes = (VkDeviceSize)kFeedbackSlots * sizeof(u32);

    // Device-local SSBO the fragment shaders atomicMin into. Pure DEVICE_LOCAL —
    // per-fragment atomics over PCIe would be a disaster.
    {
        VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bci.size  = kSlotBytes;
        bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        if (VK::Vram::CreateBuffer(VulkanHW.m_Allocator, &bci, &aci, &m_FbDevice, &m_FbDeviceAlloc, nullptr) != VK_SUCCESS) {
            Msg("![VK-TexStream] feedback device buffer alloc FAILED — GPU feedback off (LRU fallback)");
            m_FbDevice = VK_NULL_HANDLE;
            return false;
        }
    }

    // Host readback ring (one region per frame in flight). HOST_ACCESS_RANDOM →
    // cached memory: the CPU scans all 32 KB every frame, write-combined would crawl.
    {
        VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bci.size  = kSlotBytes * CVulkanCommandManager::FRAMES_IN_FLIGHT;
        bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo info{};
        if (VK::Vram::CreateBuffer(VulkanHW.m_Allocator, &bci, &aci, &m_FbRead, &m_FbReadAlloc, &info) != VK_SUCCESS) {
            Msg("![VK-TexStream] feedback readback buffer alloc FAILED — GPU feedback off");
            VK::Vram::DestroyBuffer(VulkanHW.m_Allocator, m_FbDevice, m_FbDeviceAlloc);
            m_FbDevice = VK_NULL_HANDLE; m_FbDeviceAlloc = VK_NULL_HANDLE;
            m_FbRead = VK_NULL_HANDLE;
            return false;
        }
        m_FbReadPtr = (const u32*)info.pMappedData;
        // Poison the ring: 0xFF... = "no data yet", so the first reads are no-ops.
        if (info.pMappedData) memset(info.pMappedData, 0xFF, (size_t)bci.size);
    }

    // First-use init of the device buffer (shader expects 0xFFFFFFFF = untouched;
    // per-frame refills happen in RecordFeedbackResolve).
    if (VkCommandBuffer icmd = CommandManager.BeginImmediate()) {
        vkCmdFillBuffer(icmd, m_FbDevice, 0, VK_WHOLE_SIZE, 0xFFFFFFFFu);
        CommandManager.EndAndSubmitImmediate(icmd);
    }

    Msg("[VK-TexStream] GPU feedback ON: %u slots, %llu KB device + %u x %llu KB readback",
        kFeedbackSlots, (unsigned long long)(kSlotBytes >> 10),
        CVulkanCommandManager::FRAMES_IN_FLIGHT, (unsigned long long)(kSlotBytes >> 10));
    return true;
}

VkBuffer TextureStreamer::GetFeedbackBuffer()
{
    std::lock_guard<std::mutex> lk(s_Mutex);
    EnsureFeedbackGpu();
    return m_FbDevice;
}

void TextureStreamer::RecordFeedbackResolve(VkCommandBuffer cmd, u32 frameSlot)
{
    if (!ps_r_txstream) return;
    if (cmd == VK_NULL_HANDLE || m_FbDevice == VK_NULL_HANDLE) return;
    if (frameSlot >= CVulkanCommandManager::FRAMES_IN_FLIGHT) return;

    constexpr VkDeviceSize kSlotBytes = (VkDeviceSize)kFeedbackSlots * sizeof(u32);

    auto barrier = [&](VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
                       VkPipelineStageFlags dstStage, VkAccessFlags dstAccess) {
        VkMemoryBarrier mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        mb.srcAccessMask = srcAccess;
        mb.dstAccessMask = dstAccess;
        vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 1, &mb, 0, nullptr, 0, nullptr);
    };

    // Previous frames' fragment atomics (same queue, earlier submits) -> copy out.
    barrier(VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    VkBufferCopy region{ 0, kSlotBytes * frameSlot, kSlotBytes };
    vkCmdCopyBuffer(cmd, m_FbDevice, m_FbRead, 1, &region);
    // Copy done -> refill for THIS frame's fragment writes.
    barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    vkCmdFillBuffer(cmd, m_FbDevice, 0, VK_WHOLE_SIZE, 0xFFFFFFFFu);
    barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
}

// Decode host ring slot `frameSlot` (fence-proven complete — CRender::Begin waits
// the frame fence before Frame() runs) into per-texture wanted mips.
void TextureStreamer::ReadFeedback(u32 frameSlot)
{
    if (m_FbReadPtr == nullptr || frameSlot >= CVulkanCommandManager::FRAMES_IN_FLIGHT) return;
    // No-op on HOST_COHERENT (all desktop NV/AMD), strictly required otherwise.
    vmaInvalidateAllocation(VulkanHW.m_Allocator, m_FbReadAlloc,
                            (VkDeviceSize)frameSlot * kFeedbackSlots * sizeof(u32),
                            (VkDeviceSize)kFeedbackSlots * sizeof(u32));
    const u32* slot = m_FbReadPtr + (size_t)frameSlot * kFeedbackSlots;

    std::lock_guard<std::mutex> lk(s_Mutex);
    const u32 now = Device.dwFrame;
    const u32 hi  = std::min<u32>(m_NextSlot, (u32)m_SlotOwner.size());
    for (u32 i = 0; i < hi; ++i) {
        const u32 enc = slot[i];
        if (enc == 0xFFFFFFFFu) continue;            // not sampled that frame
        CVulkanTexture* tex = m_SlotOwner[i];
        if (!tex) continue;
        auto it = m_Tex.find(tex);
        if (it == m_Tex.end()) continue;
        StreamTexture& s = it->second;
        m_FbAliveFrame = now;
        s.fbSeenFrame  = now;
        // The sample was FRAMES_IN_FLIGHT frames ago; if residency changed since,
        // the relative LOD refers to the OLD base — skip until fresh data lands.
        if (now - s.lastSwapFrame <= CVulkanCommandManager::FRAMES_IN_FLIGHT + 1) continue;
        const float rel   = DecodeFbLod(enc);
        const int   want  = (int)s.residentBase + (int)std::lround(rel);
        const u32   maxB  = MaxBaseForDims(s.fullW, s.fullH, s.fullMips, kMinResidentDim);
        s.fbWantedBase    = (u32)std::clamp(want, 0, (int)maxB);
    }
}

// Free retired residencies once no in-flight frame can still be sampling them.
// `force` (shutdown / device idle) drains everything.
void TextureStreamer::TickRetire(bool force)
{
    if (m_Retired.empty()) return;
    const u32 now = Device.dwFrame;
    size_t w = 0;
    for (size_t r = 0; r < m_Retired.size(); ++r) {
        RetiredImage& e = m_Retired[r];
        if (force || now < e.frame /*counter reset*/ ||
            now - e.frame > CVulkanCommandManager::FRAMES_IN_FLIGHT + 1)
            xr_delete(e.holder);
        else
            m_Retired[w++] = e;
    }
    m_Retired.resize(w);
}

void TextureStreamer::Shutdown()
{
    StopIoThread();
    TickRetire(true);
    std::lock_guard<std::mutex> lk(s_Mutex);
    for (auto& kv : m_Tex) kv.second.ioPending = false;   // dropped with the queues
    if (VulkanHW.m_Allocator != VK_NULL_HANDLE) {
        if (m_FbDevice != VK_NULL_HANDLE) VK::Vram::DestroyBuffer(VulkanHW.m_Allocator, m_FbDevice, m_FbDeviceAlloc);
        if (m_FbRead   != VK_NULL_HANDLE) VK::Vram::DestroyBuffer(VulkanHW.m_Allocator, m_FbRead,   m_FbReadAlloc);
    }
    m_FbDevice = VK_NULL_HANDLE; m_FbDeviceAlloc = VK_NULL_HANDLE;
    m_FbRead   = VK_NULL_HANDLE; m_FbReadAlloc   = VK_NULL_HANDLE;
    m_FbReadPtr = nullptr;
    m_FbTried   = false;
    m_SlotOwner.clear(); m_FreeSlots.clear(); m_NextSlot = 0;
}

// ---------------------------------------------------------------------------
// Load-time residency plan
// ---------------------------------------------------------------------------
u32 TextureStreamer::PlanLoadMipSkip(u32 fullW, u32 fullH, u32 fullMips, VkFormat fmt,
                                     TexStreamClass klass) const
{
    if (fullMips <= 1) return 0;

    // Largest skip that keeps the base mip >= kMinResidentDim (never shrink tiny
    // textures; never leave < 1 mip).
    u32 maxSkip = 0;
    {
        u32 w = fullW, h = fullH;
        for (u32 s = 0; s + 1 < fullMips; ++s) {
            u32 nw = (w > 1) ? (w >> 1) : 1;
            u32 nh = (h > 1) ? (h >> 1) : 1;
            // would dropping mip `s` (making mip s+1 the base) keep base >= floor?
            if (std::max(nw, nh) < kMinResidentDim) break;
            maxSkip = s + 1;
            w = nw; h = nh;
        }
    }
    if (maxSkip == 0) return 0;

    // 1) Manual quality lever — WorldDiffuse only (keep UI/detail/lmap/terrain crisp).
    u32 skip = 0;
    if (klass == TexStreamClass::WorldDiffuse && psTextureLOD > 0)
        skip = std::min<u32>((u32)psTextureLOD, maxSkip);

    // 2) Automatic budget-fit — WorldDiffuse ONLY. It is the one class the dynamic
    // streamer can PROMOTE BACK later; every other class (Lmap/Terrain/Detail/Bump/
    // UI) has no recovery path, so a load-time crush would freeze it blurry for the
    // whole session (Pripyat: the terrain splat mask budget-capped at load = ground
    // permanently in mush). Never permanently crush what cannot heal.
    if (klass == TexStreamClass::WorldDiffuse) {
        VkDeviceSize usage = 0, devBudget = 0;
        VulkanHW.GetVramBudget(usage, devBudget);
        // Credit VMA's cached-but-free block space (level transitions: the previous
        // level's freed textures leave blocks the driver still reports as used —
        // this texture will sub-allocate into them without growing real usage).
        const VkDeviceSize fib = VmaFreeInBlocks();
        usage = (usage > fib) ? (usage - fib) : 0;
        if (devBudget > 0) {
            // During a level load the live usage figure LIES LOW: geometry, VSM pages
            // and Streamline FG buffers all land after most textures. Protect their
            // room too (headroom + reserve), or textures fill the card first and the
            // late allocations crash Streamline (seen: 7948 MB alloc vs 7123 budget).
            VkDeviceSize protect = (VkDeviceSize)std::max(0, ps_r_txstream_headroom) << 20;
            if (m_InLevelLoad)
                protect += (VkDeviceSize)std::max(0, ps_r_txstream_reserve) << 20;
            const VkDeviceSize ceil = (devBudget > protect) ? (devBudget - protect) : 0;
            while (skip < maxSkip &&
                   usage + MipChainBytes(fullW, fullH, fullMips, skip, fmt) > ceil)
                ++skip;
        }
    }
    return skip;
}

// ---------------------------------------------------------------------------
// Level load bracketing
// ---------------------------------------------------------------------------
void TextureStreamer::BeginLevelLoad()
{
    std::lock_guard<std::mutex> lk(s_Mutex);
    m_InLevelLoad  = true;
    m_LoadCapCount = 0;
    VkDeviceSize usage = 0, budget = 0;
    VulkanHW.GetVramBudget(usage, budget);
    Msg("[VK-TexStream] level load begin — VRAM %llu/%llu MB, tracked textures %llu MB (lod=%d)",
        (unsigned long long)(usage >> 20), (unsigned long long)(budget >> 20),
        (unsigned long long)(m_TrackedBytes >> 20), psTextureLOD);
}

void TextureStreamer::EndLevelLoad()
{
    {
        std::lock_guard<std::mutex> lk(s_Mutex);
        if (!m_InLevelLoad) return;   // DeferredLoad(FALSE) + ResourcesDeferredUpload both close; report once
        m_InLevelLoad = false;
        VkDeviceSize usage = 0, budget = 0;
        VulkanHW.GetVramBudget(usage, budget);
        Msg("[VK-TexStream] level load end — VRAM %llu/%llu MB, %zu textures (%llu MB), %u demoted by budget/lod cap",
            (unsigned long long)(usage >> 20), (unsigned long long)(budget >> 20),
            m_Tex.size(), (unsigned long long)(m_TrackedBytes >> 20), m_LoadCapCount);
        {   // Class breakdown at load end — the "who owns the irreducible MB" answer.
            VkDeviceSize byClass[6] = {};
            u32          nClass[6]  = {};
            for (auto& kv : m_Tex) {
                const u32 k = std::min<u32>((u32)kv.second.klass, 5u);
                byClass[k] += kv.second.residentBytes;
                ++nClass[k];
            }
            for (u32 k = 0; k < 6; ++k)
                if (nClass[k])
                    Msg("[VK-TexStream]   class %-12s: %4u tex, %5llu MB", KlassName((TexStreamClass)k),
                        nClass[k], (unsigned long long)(byClass[k] >> 20));
        }
    }
    // Lock released — now trim the resident set to leave the reserve free (its own
    // locking + GPU work). Runs here, at the load-end idle point, NOT per frame.
    EnforceLoadReserve();
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------
void TextureStreamer::GetMemoryUsage(u32& outBytes, u32& outCount) const
{
    std::lock_guard<std::mutex> lk(s_Mutex);
    outBytes = (u32)std::min<VkDeviceSize>(m_TrackedBytes, 0xFFFFFFFFull);
    outCount = (u32)m_Tex.size();
}

void TextureStreamer::DumpStats() const
{
    std::lock_guard<std::mutex> lk(s_Mutex);

    VkDeviceSize usage = 0, budget = 0;
    VulkanHW.GetVramBudget(usage, budget);

    // Snapshot + sort by resident footprint for the "top consumers" list.
    std::vector<const StreamTexture*> list;
    list.reserve(m_Tex.size());
    u32 demoted = 0, streamable = 0;
    for (auto& kv : m_Tex) {
        list.push_back(&kv.second);
        if (kv.second.residentBase > 0) ++demoted;
        if (kv.second.streamable)       ++streamable;
    }
    std::sort(list.begin(), list.end(),
              [](const StreamTexture* a, const StreamTexture* b) { return a->residentBytes > b->residentBytes; });

    Msg("==== [VK-TexStream] residency report ====");
    Msg("  device VRAM: %llu / %llu MB (headroom %d MB, mem_budget=%s, mem_priority=%s)",
        (unsigned long long)(usage >> 20), (unsigned long long)(budget >> 20),
        ps_r_txstream_headroom,
        VulkanHW.m_bMemoryBudgetSupported ? "on" : "off",
        VulkanHW.m_bMemoryPrioritySupported ? "on" : "off");
    Msg("  VMA cached-free in blocks: %llu MB (usable by textures without growing driver usage)",
        (unsigned long long)(VmaFreeInBlocks() >> 20));
    Msg("  tracked textures: %zu (%llu MB), streamable %u, mip-capped %u, texture_lod %d",
        m_Tex.size(), (unsigned long long)(m_TrackedBytes >> 20), streamable, demoted, psTextureLOD);
    {   // Per-class residency breakdown — shows where the irreducible bytes live
        // (only WorldDiffuse is dynamically streamable; the rest is a fixed tax).
        VkDeviceSize byClass[6] = {};
        u32          nClass[6]  = {};
        for (auto& kv : m_Tex) {
            const u32 k = std::min<u32>((u32)kv.second.klass, 5u);
            byClass[k] += kv.second.residentBytes;
            ++nClass[k];
        }
        for (u32 k = 0; k < 6; ++k)
            if (nClass[k])
                Msg("    class %-12s: %4u tex, %5llu MB", KlassName((TexStreamClass)k),
                    nClass[k], (unsigned long long)(byClass[k] >> 20));
    }
    Msg("  dynamic streaming (r_txstream): %s, texture budget %llu MB",
        ps_r_txstream ? "ON" : "off", (unsigned long long)(BudgetBytes() >> 20));
    Msg("  GPU feedback: %s, slots %u/%u, last data %u frames ago, retired-in-flight %zu, oversub bias +%.1f",
        (m_FbDevice != VK_NULL_HANDLE) ? "on" : "OFF",
        (u32)(m_NextSlot - m_FreeSlots.size()), kFeedbackSlots,
        m_FbAliveFrame ? (Device.dwFrame - m_FbAliveFrame) : 0,
        m_Retired.size(), m_OverBias);
    const u32 topN = std::min<u32>(15, (u32)list.size());
    for (u32 i = 0; i < topN; ++i) {
        const StreamTexture* s = list[i];
        Msg("  %5llu MB  %ux%u m%u/%u  %s%s",
            (unsigned long long)(s->residentBytes >> 20),
            s->fullW >> s->residentBase, s->fullH >> s->residentBase,
            s->fullMips - s->residentBase, s->fullMips,
            s->streamable ? "[stream] " : "",
            s->file.c_str());
    }
    Msg("=========================================");
}

// ---------------------------------------------------------------------------
// Budget for the dynamic path
// ---------------------------------------------------------------------------
VkDeviceSize TextureStreamer::BudgetBytes() const
{
    if (ps_r_txstream_budget > 0)
        return (VkDeviceSize)ps_r_txstream_budget << 20;

    VkDeviceSize usage = 0, devBudget = 0;
    VulkanHW.GetVramBudget(usage, devBudget);
    if (devBudget == 0) return 0;

    // VMA keeps the VkDeviceMemory blocks of freed sub-allocations cached for
    // reuse, so the DRIVER usage figure never drops when we demote textures — but
    // that space IS available to new texture allocations (VMA sub-allocates into
    // it without growing driver usage). Without counting it back in, the plain
    // (usage - tracked) "non-texture" estimate is a closed loop that can never
    // promote: demoting N MB lowers tracked AND the derived budget by the same N
    // (proven on Pripyat 15-07: usage pinned at 7575 → tracked-budget stuck at a
    // constant +964 MB no matter how much was demoted).
    const VkDeviceSize freeInBlocks = VmaFreeInBlocks();

    const VkDeviceSize headroom = (VkDeviceSize)std::max(0, ps_r_txstream_headroom) << 20;
    // Non-texture usage actually committed (RTs, VSM, geometry, FG buffers):
    // driver usage minus our textures minus the reusable cached-block space.
    const VkDeviceSize occupied = m_TrackedBytes + freeInBlocks;
    const VkDeviceSize nonTex   = (usage > occupied) ? (usage - occupied) : 0;
    if (devBudget <= headroom + nonTex) return 0;
    return devBudget - headroom - nonTex;
}

// ---------------------------------------------------------------------------
// Per-frame
// ---------------------------------------------------------------------------
void TextureStreamer::Touch(CVulkanTexture* tex)
{
    if (!tex || !ps_r_txstream) return;   // only needed for the dynamic path
    std::lock_guard<std::mutex> lk(s_Mutex);
    auto it = m_Tex.find(tex);
    if (it != m_Tex.end()) it->second.lastUsedFrame = Device.dwFrame;
}

void TextureStreamer::Frame()
{
    VK_CPU_PROBE("TexStream");
    // Retired residencies age out every frame regardless of the streaming mode —
    // EnforceLoadReserve also retires through this ring.
    TickRetire();

    // Apply finished async reads every frame (bounded) — even with r_txstream
    // flipped off mid-session, whatever the worker already read must drain.
    DrainIoCompletions();

    if (!ps_r_txstream) return;

    // Decode the feedback ring slot this frame slot last wrote (fence-complete:
    // CRender::Begin waits the frame fence before calling us). Cheap 32 KB scan.
    ReadFeedback(CommandManager.GetCurrentFrame());

    // Throttle the residency heuristic to a few times a second — swaps apply with
    // a couple frames of IO latency anyway, and batching keeps the plan coherent.
    if (Device.dwFrame - m_LastStreamFrame < 15) return;
    m_LastStreamFrame = Device.dwFrame;
    StreamStep();
}

// Feedback-driven residency (UE-VT recipe): the GPU told us, per texture, the
// sharpest mip it actually wanted last frame. Promote straight to it (one rebuild),
// demote with hysteresis (3 consecutive ticks wanting coarser, and only when the
// budget is getting tight), evict invisibles under pressure. Falls back to the LRU
// heuristic while no feedback arrives (stale .spv without the feedback writes).
void TextureStreamer::StreamStep()
{
    const bool fbAlive = (m_FbDevice != VK_NULL_HANDLE) && m_FbAliveFrame != 0 &&
                         (Device.dwFrame - m_FbAliveFrame) < 300;
    if (!fbAlive) { StreamStepLRU(); return; }

    std::vector<std::pair<StreamTexture*, u32>> plan;   // (texture, newBase)
    {
        std::lock_guard<std::mutex> lk(s_Mutex);

        const VkDeviceSize budget = BudgetBytes();
        if (budget == 0) return;
        const VkDeviceSize slack = budget - budget / 8;   // free-promote ceiling (~87%)

        const u32 bias = (u32)m_OverBias;

        // Manual quality lever still floors the ideal: texture_lod 2 means even a
        // fullscreen texture never promotes past base mip 2.
        const u32 idealBase = (psTextureLOD > 0) ? (u32)psTextureLOD : 0;
        const u32 now       = Device.dwFrame;

        VkDeviceSize tracked = m_TrackedBytes;

        struct Cand { StreamTexture* s; u32 target; s32 severity; };
        std::vector<Cand> rescues;    // visible & below the 256px floor — unconditional
        std::vector<Cand> promotes;   // visible quality promotes — budget-gated
        std::vector<Cand> demotes;

        for (auto& kv : m_Tex) {
            StreamTexture& s = kv.second;
            if (!s.streamable) continue;
            if (s.ioPending) continue;   // a read for this texture is already in flight
            const u32 maxB = MaxBaseForDims(s.fullW, s.fullH, s.fullMips, kMinResidentDim);
            const bool seen = s.fbSeenFrame != 0 && (now - s.fbSeenFrame) < 60;

            if (seen && s.fbWantedBase != 0xFFFFFFFFu) {
                // Visible textures live between the feedback wish (+ oversubscription
                // bias) and the 256px PERCEPTUAL FLOOR: whatever the budget says, an
                // on-screen surface below 256px reads as broken, not as "streaming"
                // (the +6-bias run demoted the whole world to 64px mush — never again).
                const u32 floorB = MaxBaseForDims(s.fullW, s.fullH, s.fullMips, kVisibleFloorDim);
                const u32 want   = std::min(floorB,
                                   std::clamp(std::max(s.fbWantedBase + bias, idealBase), 0u, maxB));
                if (want < s.residentBase) {
                    s.demoteTicks = 0;
                    if (s.residentBase > floorB)   // below 256px on screen → rescue to the floor now;
                        rescues.push_back({ &s, floorB, (s32)(s.residentBase - floorB) });
                    else                           // above it, further sharpening queues on the budget
                        promotes.push_back({ &s, want, (s32)(s.residentBase - want) });
                } else if (want > s.residentBase) {
                    // Sustained-coarser only, and only when memory matters — an
                    // over-sharp resident texture is harmless with a roomy budget.
                    if (++s.demoteTicks >= 3 && tracked > slack)
                        demotes.push_back({ &s, want, (s32)(want - s.residentBase) });
                } else {
                    s.demoteTicks = 0;
                }
            } else if (tracked > budget) {
                // Invisible for a while + over budget → give its VRAM back. Straight
                // to the 64px floor: feedback rescues it within ticks if the player
                // turns around.
                const bool longUnseen = s.fbSeenFrame == 0 || (now - s.fbSeenFrame) > 300;
                if (longUnseen && s.residentBase < maxB)
                    demotes.push_back({ &s, maxB, (s32)(maxB - s.residentBase) });
            }
        }

        // Rescues first — visible surfaces below the 256px floor heal UNCONDITIONALLY
        // (tiny images: a 256px BC chain is ~90 KB; even the whole level's visible set
        // is a few dozen MB of overdraft the driver can absorb).
        std::sort(rescues.begin(), rescues.end(),
                  [](const Cand& a, const Cand& b) { return a.severity > b.severity; });
        u32 nRes = 0;
        for (const Cand& c : rescues) {
            if (nRes >= kMaxRescuesPerTick) break;
            const VkDeviceSize after =
                MipChainBytes(c.s->fullW, c.s->fullH, c.s->fullMips, c.target, c.s->format);
            tracked += (after > c.s->residentBytes) ? (after - c.s->residentBytes) : 0;
            c.s->wantedBase = c.target;
            plan.emplace_back(c.s, c.target);
            ++nRes;
        }

        // Demotes (they fund the quality promotes), biggest relief first. Own cap —
        // a big invisible backlog must never starve the promote side (seen in the
        // 21:19 log: 700 pending demotes ate all 8 slots, 0 promotes for minutes).
        std::sort(demotes.begin(), demotes.end(),
                  [](const Cand& a, const Cand& b) { return a.s->residentBytes > b.s->residentBytes; });
        VkDeviceSize freed = 0;   // bytes this tick's demotes give back
        u32 nDem = 0;
        for (const Cand& c : demotes) {
            if (nDem >= kMaxDemotesPerTick) break;
            const VkDeviceSize after =
                MipChainBytes(c.s->fullW, c.s->fullH, c.s->fullMips, c.target, c.s->format);
            freed   += (c.s->residentBytes - after);
            tracked -= (c.s->residentBytes - after);
            c.s->wantedBase = c.target;
            plan.emplace_back(c.s, c.target);
            ++nDem;
        }

        // Quality promotes: most-starved first (biggest mip gap = blurriest on
        // screen). Allowed under the slack ceiling OR as a net-zero pair (funded by
        // this tick's demotes) — over budget the pool composition still shifts
        // toward what's actually on screen instead of freezing.
        std::sort(promotes.begin(), promotes.end(),
                  [](const Cand& a, const Cand& b) { return a.severity > b.severity; });
        u32 nPro = 0, nStarved = 0;
        for (const Cand& c : promotes) {
            if (nPro >= kMaxPromotesPerTick) break;
            const VkDeviceSize after =
                MipChainBytes(c.s->fullW, c.s->fullH, c.s->fullMips, c.target, c.s->format);
            const VkDeviceSize delta = (after > c.s->residentBytes) ? (after - c.s->residentBytes) : 0;
            const bool fits   = tracked + delta <= slack;
            const bool funded = delta <= freed;
            if (delta > 0 && !fits && !funded) { ++nStarved; continue; }
            if (!fits) freed -= delta;
            tracked += delta;
            c.s->wantedBase = c.target;
            plan.emplace_back(c.s, c.target);
            ++nPro;
        }

        // Oversubscription bias — driven by PROMOTE STARVATION, not tracked>budget:
        // tracked includes ~a GB of non-streamable classes (lmaps/details/bumps) the
        // streamer can never free, so the old "over budget → +bias" comparison
        // ratcheted straight to the cap and flattened the visible set (+6 = 64px
        // world, the 21:44 log). Starvation is the honest signal: quality promotes
        // that can't be funded → coarsen the global wish; NO starvation → recover —
        // unconditionally (an early "else if (nPro==0)" froze bias at +2.9 while
        // trickle-promotes ran, pinning the world at quarter-res with 3 GB of budget
        // free — the 21:54 log). Fast decay when the budget has real room.
        // Cap +3: past that the 256px floor dominates anyway.
        // ...but only when tracked actually exceeds the BUDGET. Starving at the
        // 87% slack ceiling with room below budget is equilibrium, not
        // oversubscription — ratcheting bias there pinned the world at +3.0 with
        // 400 MB of budget unused (23:30 log: want 1 pro / starved 1 forever).
        if (nStarved > 0 && tracked > budget)
            m_OverBias = std::min(m_OverBias + 0.25f, 3.0f);
        else {
            const float decay = (tracked + (VkDeviceSize(512) << 20) < slack) ? 0.5f : 0.125f;
            m_OverBias = std::max(m_OverBias - decay, 0.0f);
        }

        // Tick telemetry — only when something happened (or wanted to and couldn't).
        if (!plan.empty() || nStarved > 0) {
            Msg("[VK-TexStream] tick: %u rescue / %u promote / %u demote (want %zu pro %zu dem, starved %u), tracked %llu MB, budget %llu MB, bias +%.1f",
                nRes, nPro, nDem, promotes.size(), demotes.size(), nStarved,
                (unsigned long long)(tracked >> 20), (unsigned long long)(budget >> 20), m_OverBias);
        }
    }

    EnqueuePlanIo(plan);   // async: worker reads the files, DrainIoCompletions applies
}

// Pre-feedback heuristic (LRU + budget), kept as the fallback while the feedback
// buffer carries no data — e.g. the deployed .spv predates the feedback writes.
void TextureStreamer::StreamStepLRU()
{
    std::vector<std::pair<StreamTexture*, u32>> plan;   // (texture, newBase)

    {
        std::lock_guard<std::mutex> lk(s_Mutex);

        const VkDeviceSize budget = BudgetBytes();
        if (budget == 0) return;

        // Every streamable's ideal base is 0 (full res) modulated by the manual lever.
        const u32 idealBase = (psTextureLOD > 0) ? (u32)psTextureLOD : 0;

        // Gather streamables, LRU order (oldest first) for demotion candidates.
        std::vector<StreamTexture*> pool;
        pool.reserve(m_Tex.size());
        for (auto& kv : m_Tex)
            if (kv.second.streamable && !kv.second.ioPending) pool.push_back(&kv.second);
        std::sort(pool.begin(), pool.end(),
                  [](StreamTexture* a, StreamTexture* b) { return a->lastUsedFrame < b->lastUsedFrame; });

        VkDeviceSize tracked = m_TrackedBytes;

        // Largest base a streamable may be demoted to — keeps the base mip >=
        // kMinResidentDim, matching the load-time plan so residency accounting and the
        // actual built image never disagree.
        auto maxBaseFor = [](const StreamTexture* s) -> u32 {
            u32 mb = 0, w = s->fullW, h = s->fullH;
            for (u32 b = 0; b + 1 < s->fullMips; ++b) {
                u32 nw = (w > 1) ? (w >> 1) : 1, nh = (h > 1) ? (h >> 1) : 1;
                if ((nw > nh ? nw : nh) < kMinResidentDim) break;
                mb = b + 1; w = nw; h = nh;
            }
            return mb;
        };

        // Under pressure → demote least-recently-used first (raise base mip by 1).
        if (tracked > budget) {
            for (StreamTexture* s : pool) {
                if (plan.size() >= kMaxSwapsPerTick) break;
                if (tracked <= budget) break;
                if (s->residentBase >= maxBaseFor(s)) continue;         // already at floor
                u32 nb = s->residentBase + 1;
                VkDeviceSize before = s->residentBytes;
                VkDeviceSize after  = MipChainBytes(s->fullW, s->fullH, s->fullMips, nb, s->format);
                tracked -= (before - after);
                s->wantedBase = nb;
                plan.emplace_back(s, nb);
            }
        }
        // Slack → promote most-recently-used demoted textures toward ideal.
        else {
            const VkDeviceSize slack = budget - budget / 8;   // keep ~12% free before promoting
            for (auto it = pool.rbegin(); it != pool.rend(); ++it) {
                StreamTexture* s = *it;
                if (plan.size() >= kMaxSwapsPerTick) break;
                if (s->residentBase <= idealBase) continue;             // already at ideal
                u32 nb = s->residentBase - 1;
                VkDeviceSize before = s->residentBytes;
                VkDeviceSize after  = MipChainBytes(s->fullW, s->fullH, s->fullMips, nb, s->format);
                if (tracked + (after - before) > slack) continue;       // no room
                tracked += (after - before);
                s->wantedBase = nb;
                plan.emplace_back(s, nb);
            }
        }
    }

    EnqueuePlanIo(plan);   // async: worker reads the files, DrainIoCompletions applies
}

// Execute a batch of (texture -> new base mip) residency changes — WITHOUT any
// device idle (the UE-VT recipe: never in-place, never stall). Runs the GPU work
// OUTSIDE the streamer lock:
//   1) build every new image (stages copies on the transfer queue),
//   2) one FlushUploadsAndWait — blocks THIS thread on the transfer timeline only;
//      the graphics queue never stops,
//   3) swap handles + rebind: the rebind callback allocates FRESH descriptor sets
//      and flips the material's set pointer (this runs in CRender::Begin, before
//      the new frame records — pending frames keep their old sets + old views),
//   4) old images retire into m_Retired; TickRetire frees them once every frame
//      that could sample them has fenced out.
void TextureStreamer::ApplyResidencyPlan(std::vector<std::pair<StreamTexture*, u32>>& plan)
{
    if (plan.empty()) return;

    struct Pending { StreamTexture* st; CVulkanTexture* built; u32 newBase; };
    std::vector<Pending> pend;
    pend.reserve(plan.size());

    for (auto& pr : plan) {
        if (!pr.first || !pr.first->tex) continue;
        if (pr.second == pr.first->residentBase) continue;   // nothing to change
        CVulkanTexture* nt = xr_new<CVulkanTexture>();
        if (pr.first->tex->BuildStreamImage(*nt, pr.second) && nt->IsValid())
            pend.push_back({ pr.first, nt, pr.second });
        else
            xr_delete(nt);   // build failed — keep current residency
    }
    if (pend.empty()) return;

    CommandManager.FlushUploadsAndWait();        // new images uploaded (transfer wait only)

    {
        std::lock_guard<std::mutex> lk(s_Mutex);
        for (auto& p : pend) {
            // The live texture object keeps its identity + registration; only its GPU
            // handles + the sampled view change. Old handles migrate into p.built.
            p.st->tex->SwapContents(*p.built);
            if (m_Rebind) m_Rebind(p.st->tex);

            if (m_TrackedBytes >= p.st->residentBytes) m_TrackedBytes -= p.st->residentBytes;
            p.st->residentBase  = p.newBase;
            p.st->residentBytes = MipChainBytes(p.st->fullW, p.st->fullH, p.st->fullMips, p.newBase, p.st->format);
            p.st->lastSwapFrame = Device.dwFrame;
            p.st->demoteTicks   = 0;
            m_TrackedBytes += p.st->residentBytes;
        }
    }

    // Old images (inside the built holders) stay alive until the fence horizon
    // passes — in-flight frames may still sample their views through the OLD
    // descriptor sets (retired on the material side with the same horizon).
    m_Retired.reserve(m_Retired.size() + pend.size());
    for (auto& p : pend)
        m_Retired.push_back({ Device.dwFrame, p.built });
}

// ---------------------------------------------------------------------------
// Async DDS IO worker (see the header comment). The dynamic per-frame path goes
// through here; the synchronous ApplyResidencyPlan above stays for the load-time
// callers (EnsureMinResidency, EnforceLoadReserve) where blocking is fine.
// ---------------------------------------------------------------------------
void TextureStreamer::EnsureIoThread()
{
    if (m_IoThread.joinable()) return;
    {
        std::lock_guard<std::mutex> lk(m_IoMutex);
        m_IoStop = false;
    }
    m_IoThread = std::thread([this] { IoThreadMain(); });
}

void TextureStreamer::IoThreadMain()
{
    for (;;) {
        IoJob job;
        {
            std::unique_lock<std::mutex> lk(m_IoMutex);
            m_IoCv.wait(lk, [this] { return m_IoStop || !m_IoQueue.empty(); });
            if (m_IoStop) return;
            job = std::move(m_IoQueue.front());
            m_IoQueue.pop_front();
        }

        // FS is safe for concurrent reads: r_open maps a fresh view per call and
        // LZO-decompresses with thread_local workmem (the seqParallel spawn path
        // already reads textures off the render thread). Copy out of the reader so
        // it never crosses threads.
        void*  blob = nullptr;
        size_t size = 0;
        if (IReader* F = FS.r_open(job.file.c_str())) {
            size = (size_t)F->length();
            if (size > 0) {
                blob = xr_malloc(size);
                memcpy(blob, F->pointer(), size);
            }
            FS.r_close(F);
        }
        if (!blob)
            Msg("![VK-TexStream] io worker: failed to read '%s'", job.file.c_str());

        {
            std::lock_guard<std::mutex> lk(m_IoMutex);
            m_IoDone.push_back({ job.tex, job.file, job.target, blob, size });
        }
    }
}

void TextureStreamer::EnqueuePlanIo(const std::vector<std::pair<StreamTexture*, u32>>& plan)
{
    if (plan.empty()) return;
    EnsureIoThread();

    std::vector<IoJob> jobs;
    jobs.reserve(plan.size());
    {
        std::lock_guard<std::mutex> lk(s_Mutex);
        for (auto& pr : plan) {
            if (!pr.first || !pr.first->tex) continue;
            if (pr.second == pr.first->residentBase) continue;
            pr.first->ioPending = true;
            jobs.push_back({ pr.first->tex, pr.first->file, pr.second });
        }
    }
    if (jobs.empty()) return;
    {
        std::lock_guard<std::mutex> lk(m_IoMutex);
        for (auto& j : jobs) m_IoQueue.push_back(std::move(j));
    }
    m_IoCv.notify_one();
}

// Apply finished reads: create the new images + stage their copies (bounded per
// frame), swap handles + rebind, retire the old images. NO CPU wait on the GPU:
// UploadImage records on the transfer queue and the frame's graphics submit waits
// the upload timeline before FRAGMENT_SHADER, so draws sampling the new views are
// ordered after the copies GPU-side.
void TextureStreamer::DrainIoCompletions()
{
    VK_CPU_PROBE("TexStream/drain");
    std::vector<IoDone> batch;
    {
        std::lock_guard<std::mutex> lk(m_IoMutex);
        if (m_IoDone.empty()) return;
        VkDeviceSize bytes = 0;
        while (!m_IoDone.empty() && batch.size() < kMaxApplyPerFrame && bytes < kMaxApplyBytesFrame) {
            bytes += m_IoDone.front().size;
            batch.push_back(std::move(m_IoDone.front()));
            m_IoDone.pop_front();
        }
    }

    struct Pending { StreamTexture* st; CVulkanTexture* built; u32 newBase; };
    std::vector<Pending> pend;
    pend.reserve(batch.size());

    for (IoDone& d : batch) {
        StreamTexture* st = nullptr;
        {
            std::lock_guard<std::mutex> lk(s_Mutex);
            auto it = m_Tex.find(d.tex);
            // The texture may have been destroyed (address possibly reused) or
            // re-targeted by a sync plan while the read was in flight — the file +
            // wantedBase checks turn a stale blob into a harmless drop.
            if (it != m_Tex.end() && it->second.file == d.file) {
                it->second.ioPending = false;
                if (d.blob && it->second.wantedBase == d.target && it->second.residentBase != d.target)
                    st = &it->second;
            }
        }
        if (st) {
            CVulkanTexture* nt = xr_new<CVulkanTexture>();
            if (d.tex->BuildStreamImageFromBlob(*nt, d.target, d.blob, d.size) && nt->IsValid())
                pend.push_back({ st, nt, d.target });
            else
                xr_delete(nt);
        }
        if (d.blob) xr_free(d.blob);
    }
    if (pend.empty()) return;

    {
        std::lock_guard<std::mutex> lk(s_Mutex);
        for (auto& p : pend) {
            p.st->tex->SwapContents(*p.built);
            if (m_Rebind) m_Rebind(p.st->tex);

            if (m_TrackedBytes >= p.st->residentBytes) m_TrackedBytes -= p.st->residentBytes;
            p.st->residentBase  = p.newBase;
            p.st->residentBytes = MipChainBytes(p.st->fullW, p.st->fullH, p.st->fullMips, p.newBase, p.st->format);
            p.st->lastSwapFrame = Device.dwFrame;
            p.st->demoteTicks   = 0;
            m_TrackedBytes += p.st->residentBytes;
        }
    }

    m_Retired.reserve(m_Retired.size() + pend.size());
    for (auto& p : pend)
        m_Retired.push_back({ Device.dwFrame, p.built });
}

void TextureStreamer::FreeDeviceVram(u32 mbNeeded)
{
    VkDeviceSize usage = 0, budget = 0;
    VulkanHW.GetVramBudget(usage, budget);
    if (budget == 0) return;
    const VkDeviceSize need    = (VkDeviceSize)mbNeeded << 20;
    const VkDeviceSize freeRaw = (budget > usage) ? (budget - usage) : 0;
    if (freeRaw >= need) return;
    const VkDeviceSize toFree = (need - freeRaw) + (need - freeRaw) / 4;   // +25% slack

    std::vector<std::pair<StreamTexture*, u32>> plan;
    VkDeviceSize planned = 0;
    {
        std::lock_guard<std::mutex> lk(s_Mutex);

        struct Cand { StreamTexture* s; u32 floorB; VkDeviceSize save; bool unseen; };
        std::vector<Cand> cands;
        const u32 now = Device.dwFrame;
        for (auto& kv : m_Tex) {
            StreamTexture& s = kv.second;
            if (!s.streamable || s.ioPending) continue;
            const u32 fb = MaxBaseForDims(s.fullW, s.fullH, s.fullMips, kVisibleFloorDim);
            if (s.residentBase >= fb) continue;
            const VkDeviceSize after = MipChainBytes(s.fullW, s.fullH, s.fullMips, fb, s.format);
            if (after >= s.residentBytes) continue;
            const bool unseen = s.fbSeenFrame == 0 || (now - s.fbSeenFrame) > 120;
            cands.push_back({ &s, fb, s.residentBytes - after, unseen });
        }
        std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
            if (a.unseen != b.unseen) return a.unseen;   // invisible first
            return a.save > b.save;                      // then biggest relief
        });
        for (const Cand& c : cands) {
            if (planned >= toFree) break;
            c.s->wantedBase = c.floorB;
            plan.emplace_back(c.s, c.floorB);
            planned += c.save;
        }

        // Hold the promote side back while NGX races for the freed memory —
        // max oversubscription bias slews every visible wish coarser; it decays
        // on its own once the ticks calm down.
        m_OverBias = 3.0f;
    }
    Msg("[VK-TexStream] DLSS rescue: demoting %zu textures (~%llu MB planned; raw free %llu of %u MB needed)",
        plan.size(), (unsigned long long)(planned >> 20), (unsigned long long)(freeRaw >> 20), mbNeeded);
    EnqueuePlanIo(plan);
}

void TextureStreamer::StopIoThread()
{
    if (!m_IoThread.joinable()) return;
    {
        std::lock_guard<std::mutex> lk(m_IoMutex);
        m_IoStop = true;
    }
    m_IoCv.notify_all();
    m_IoThread.join();

    // Drop whatever was queued or completed — device teardown; blobs are plain heap.
    std::lock_guard<std::mutex> lk(m_IoMutex);
    m_IoQueue.clear();
    for (auto& d : m_IoDone)
        if (d.blob) xr_free(d.blob);
    m_IoDone.clear();
    m_IoStop = false;
}

// Post-load residency trim (called from EndLevelLoad). Everything is resident and the
// device is idle-able, so we know the REAL total VRAM footprint — including geometry /
// VSM / render targets that loaded AFTER the textures and the room Streamline Frame-Gen
// will want. If free VRAM < r_txstream_reserve, demote the largest world textures
// (biggest-first = fewest rebuilds) by one mip at a time until the reserve is met.
void TextureStreamer::EnforceLoadReserve()
{
    // Multiple rounds against the LIVE budget: the per-mip byte estimate and the
    // driver's (lazily updated) budget figure both drift from reality, so one pass
    // routinely under-frees (seen: "~596 MB demoted" yet only 43 MB of reported
    // usage gone). Re-query after applying each plan and keep going until the
    // reserve is met or nothing demotable is left.
    for (int round = 0; round < 4; ++round)
        if (!EnforceLoadReserveRound(round))
            break;
}

// One trim round; returns true if another round makes sense (still short of the
// reserve AND this round actually demoted something).
bool TextureStreamer::EnforceLoadReserveRound(int round)
{
    VkDeviceSize usage = 0, budget = 0;
    VulkanHW.GetVramBudget(usage, budget);
    if (budget == 0) return false;

    const VkDeviceSize reserve = (VkDeviceSize)std::max(0, ps_r_txstream_reserve) << 20;
    if (reserve == 0) return false;

    std::vector<std::pair<StreamTexture*, u32>> plan;
    {
        std::lock_guard<std::mutex> lk(s_Mutex);

        // Physical free PLUS VMA's cached-but-free block space. The reserve is room
        // for late allocators (Streamline FG) that allocate OUTSIDE VMA, so this
        // used to be measured on raw budget-usage — but demoting textures can't
        // hand FG cached-block space anyway (their freed bytes fall back into the
        // same VMA blocks; measured 15-07: "~596 MB demoted" dropped driver usage
        // by 43 MB), while pool compaction legitimately leaves 1.5+ GB of reusable
        // holes the raw figure can't see. On Pripyat that mismatch demoted ~1470
        // textures at load end which the dynamic streamer immediately promoted
        // back through the very same free-in-blocks credit (BudgetBytes) — pure
        // churn. Count the credit here too so both sides agree on headroom.
        const VkDeviceSize fib     = VmaFreeInBlocks();
        const VkDeviceSize freeNow = ((budget > usage) ? (budget - usage) : 0) + fib;
        if (freeNow >= reserve) {
            Msg("[VK-TexStream] post-load reserve OK — %llu MB free (%llu MB of it in cached blocks) >= %d MB reserve (round %d)",
                (unsigned long long)(freeNow >> 20), (unsigned long long)(fib >> 20),
                ps_r_txstream_reserve, round);
            return false;
        }
        VkDeviceSize need = reserve - freeNow;   // bytes we must free

        // Largest base a streamable may reach. The post-load trim keeps a HIGHER floor
        // than dynamic streaming (256 vs 64): with r_txstream off there is no promote
        // path back, so whatever this trim demotes stays demoted for the whole session —
        // a 64px world at rest reads as "все текстуры мыльные", not as a perf feature.
        constexpr u32 kTrimFloorDim = 256;
        auto maxBaseFor = [](const StreamTexture* s) -> u32 {
            u32 mb = 0, w = s->fullW, h = s->fullH;
            for (u32 b = 0; b + 1 < s->fullMips; ++b) {
                u32 nw = (w > 1) ? (w >> 1) : 1, nh = (h > 1) ? (h >> 1) : 1;
                if ((nw > nh ? nw : nh) < kTrimFloorDim) break;
                mb = b + 1; w = nw; h = nh;
            }
            return mb;
        };

        // Working residency for each streamable; greedily demote the current-largest
        // by one mip until we've freed `need` (or nothing more can be demoted).
        struct Work { StreamTexture* st; u32 base; u32 maxBase; VkDeviceSize bytes; };
        std::vector<Work> work;
        work.reserve(m_Tex.size());
        for (auto& kv : m_Tex) {
            StreamTexture& s = kv.second;
            if (!s.streamable) continue;
            u32 mx = maxBaseFor(&s);
            if (s.residentBase < mx)
                work.push_back({ &s, s.residentBase, mx, s.residentBytes });
        }

        VkDeviceSize freed = 0;
        u32 guard = 0;
        while (freed < need && guard++ < 100000) {
            // pick the current-largest demotable entry
            Work* best = nullptr;
            for (auto& w : work) {
                if (w.base >= w.maxBase) continue;
                if (!best || w.bytes > best->bytes) best = &w;
            }
            if (!best) break;   // nothing left to demote
            const u32 nb = best->base + 1;
            const VkDeviceSize nbBytes =
                MipChainBytes(best->st->fullW, best->st->fullH, best->st->fullMips, nb, best->st->format);
            freed += (best->bytes - nbBytes);
            best->base  = nb;
            best->bytes = nbBytes;
        }

        for (auto& w : work)
            if (w.base != w.st->residentBase)
                plan.emplace_back(w.st, w.base);

        Msg("[VK-TexStream] post-load trim round %d: free %llu MB < %d MB reserve — demoting %zu textures (~%llu MB) ...",
            round, (unsigned long long)(freeNow >> 20), ps_r_txstream_reserve, plan.size(),
            (unsigned long long)(freed >> 20));
    }
    if (plan.empty()) {
        Msg("![VK-TexStream] post-load reserve NOT met and nothing left to demote");
        return false;
    }

    // Process in small chunks: ApplyResidencyPlan builds every new image BEFORE
    // freeing the old ones, so a single huge batch would spike VRAM upward exactly
    // when we're already tight. Chunking bounds the peak to a handful of extra images.
    constexpr size_t kChunk = 24;
    for (size_t i = 0; i < plan.size(); i += kChunk) {
        std::vector<std::pair<StreamTexture*, u32>> chunk(
            plan.begin() + i, plan.begin() + std::min(plan.size(), i + kChunk));
        ApplyResidencyPlan(chunk);
        // Level-load context: the next round (and the next chunk's VRAM headroom)
        // needs the demoted images ACTUALLY freed, not parked in the retire ring.
        // One idle at load end is harmless — gameplay swaps never come through here.
        vkDeviceWaitIdle(VulkanHW.m_Device);
        TickRetire(true);
    }

    VkDeviceSize u2 = 0, b2 = 0;
    VulkanHW.GetVramBudget(u2, b2);
    Msg("[VK-TexStream] post-load trim round %d done — VRAM %llu/%llu MB, %llu MB free",
        round, (unsigned long long)(u2 >> 20), (unsigned long long)(b2 >> 20),
        (unsigned long long)((b2 > u2 ? b2 - u2 : 0) >> 20));
    return true;
}

} // namespace VK
