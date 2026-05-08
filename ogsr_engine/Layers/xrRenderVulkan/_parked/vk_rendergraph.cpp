// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
// Licensed under the same terms as X-Ray Engine (see root License.txt)

#include "stdafx.h"
#include "vk_rendergraph.h"

namespace VK
{

// ---------------------------------------------------------------------------
// UsageToBarrierState — map RGUsage to (layout, stage, access)
// ---------------------------------------------------------------------------
static RGResourceState UsageToBarrierState(RGUsage usage)
{
    RGResourceState s = {};

    switch (usage)
    {
    case RGUsage::COLOR_ATTACHMENT:
        s.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        s.stage  = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        s.access = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        break;

    case RGUsage::DEPTH_ATTACHMENT:
        s.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        s.stage  = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT
                 | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        s.access = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                 | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        break;

    case RGUsage::DEPTH_READ_ONLY:
        s.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        s.stage  = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
        s.access = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
        break;

    case RGUsage::SHADER_READ:
        s.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        s.stage  = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        s.access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        break;

    case RGUsage::SHADER_READ_COMPUTE:
        s.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        s.stage  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        s.access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        break;

    case RGUsage::STORAGE_IMAGE:
        s.layout = VK_IMAGE_LAYOUT_GENERAL;
        s.stage  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        s.access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        break;

    case RGUsage::STORAGE_IMAGE_READ:
        s.layout = VK_IMAGE_LAYOUT_GENERAL;
        s.stage  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        s.access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        break;

    case RGUsage::TRANSFER_SRC:
        s.layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        s.stage  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        s.access = VK_ACCESS_2_TRANSFER_READ_BIT;
        break;

    case RGUsage::TRANSFER_DST:
        s.layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        s.stage  = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        s.access = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        break;

    case RGUsage::PRESENT:
        s.layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        s.stage  = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
        s.access = 0;
        break;

    // Buffers — layout field is unused; only stage+access matter
    case RGUsage::UNIFORM_BUFFER:
        s.layout = VK_IMAGE_LAYOUT_UNDEFINED;
        s.stage  = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT
                 | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        s.access = VK_ACCESS_2_UNIFORM_READ_BIT;
        break;

    case RGUsage::STORAGE_BUFFER_READ:
        s.layout = VK_IMAGE_LAYOUT_UNDEFINED;
        s.stage  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        s.access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        break;

    case RGUsage::STORAGE_BUFFER_WRITE:
        s.layout = VK_IMAGE_LAYOUT_UNDEFINED;
        s.stage  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        s.access = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        break;

    case RGUsage::STORAGE_BUFFER_RW:
        s.layout = VK_IMAGE_LAYOUT_UNDEFINED;
        s.stage  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        s.access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        break;

    case RGUsage::INDIRECT_BUFFER:
        s.layout = VK_IMAGE_LAYOUT_UNDEFINED;
        s.stage  = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
        s.access = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        break;

    case RGUsage::VERTEX_BUFFER:
        s.layout = VK_IMAGE_LAYOUT_UNDEFINED;
        s.stage  = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
        s.access = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
        break;

    default:
        VERIFY(!"CFrameGraph: unknown RGUsage");
        break;
    }

    return s;
}

// ---------------------------------------------------------------------------
// IsReadOnlyLayout — same read-only layout: concurrent reads need no barrier
// ---------------------------------------------------------------------------
static bool IsReadOnlyLayout(VkImageLayout layout)
{
    switch (layout)
    {
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
    case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
        return true;
    default:
        return false;
    }
}

// ---------------------------------------------------------------------------
// IsReadOnlyAccess — true if access mask contains no write bits
// Read-after-read on buffers needs no memory barrier (Vulkan spec §7.1).
// ---------------------------------------------------------------------------
static bool IsReadOnlyAccess(VkAccessFlags2 access)
{
    constexpr VkAccessFlags2 WRITE_BITS =
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
      | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
      | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
      | VK_ACCESS_2_TRANSFER_WRITE_BIT
      | VK_ACCESS_2_HOST_WRITE_BIT
      | VK_ACCESS_2_MEMORY_WRITE_BIT
      | VK_ACCESS_2_SHADER_WRITE_BIT;
    return (access & WRITE_BITS) == 0;
}

// ---------------------------------------------------------------------------
// ImportImage
// ---------------------------------------------------------------------------
RGHandle CFrameGraph::ImportImage(const char* name, VkImage image, VkImageLayout initialLayout,
                                   VkImageAspectFlags aspect, u32 layerCount, u32 mipLevels)
{
    // Re-use existing slot if name matches (same RT re-imported after Reset())
    // IMPORTANT: Do NOT overwrite tracked state (layout/stage/access) —
    // the graph tracks where each resource ended up after previous Execute().
    // Re-importing with hardcoded initialLayout would cause wrong oldLayout in barriers.
    for (u32 i = 0; i < m_ResourceCount; i++)
    {
        if (m_Resources[i].name && xr_strcmp(m_Resources[i].name, name) == 0)
        {
            m_Resources[i].image      = image;
            m_Resources[i].aspect     = aspect;
            m_Resources[i].layerCount = layerCount;
            m_Resources[i].mipLevels  = mipLevels;
            // state.layout / stage / access — PRESERVED from previous Execute()
            return { (u16)i, m_Resources[i].version };
        }
    }

    VERIFY(m_ResourceCount < MAX_RESOURCES);
    u32 idx = m_ResourceCount++;
    RGResource& res     = m_Resources[idx];
    res                 = {};
    res.image           = image;
    res.isBuffer        = false;
    res.aspect          = aspect;
    res.layerCount      = layerCount;
    res.mipLevels       = mipLevels;
    res.version         = 0;
    res.state.layout    = initialLayout;
    res.state.stage     = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
    res.state.access    = 0;
    res.name            = name;
    return { (u16)idx, 0 };
}

// ---------------------------------------------------------------------------
// ImportBuffer
// ---------------------------------------------------------------------------
RGHandle CFrameGraph::ImportBuffer(const char* name, VkBuffer buffer,
                                    VkPipelineStageFlags2 initialStage,
                                    VkAccessFlags2 initialAccess)
{
    for (u32 i = 0; i < m_ResourceCount; i++)
    {
        if (m_Resources[i].name && xr_strcmp(m_Resources[i].name, name) == 0)
        {
            m_Resources[i].buffer       = buffer;
            m_Resources[i].state.stage  = initialStage;
            m_Resources[i].state.access = initialAccess;
            return { (u16)i, m_Resources[i].version };
        }
    }

    VERIFY(m_ResourceCount < MAX_RESOURCES);
    u32 idx = m_ResourceCount++;
    RGResource& res     = m_Resources[idx];
    res                 = {};
    res.buffer          = buffer;
    res.isBuffer        = true;
    res.version         = 0;
    res.state.layout    = VK_IMAGE_LAYOUT_UNDEFINED;
    res.state.stage     = initialStage;
    res.state.access    = initialAccess;
    res.name            = name;
    return { (u16)idx, 0 };
}

// ---------------------------------------------------------------------------
// AddPass
// ---------------------------------------------------------------------------
RGHandle CFrameGraph::AddPass(const char* name, RGPassType type,
                               std::initializer_list<RGPassUse> reads,
                               std::initializer_list<RGPassUse> writes,
                               std::function<void(VkCommandBuffer)> cb)
{
    return AddPassConditional(true, name, type, reads, writes, std::move(cb));
}

RGHandle CFrameGraph::AddPassConditional(bool enabled, const char* name, RGPassType type,
                                          std::initializer_list<RGPassUse> reads,
                                          std::initializer_list<RGPassUse> writes,
                                          std::function<void(VkCommandBuffer)> cb)
{
    VERIFY(m_PassCount < MAX_PASSES);
    u32 idx    = m_PassCount++;
    RGPass& p  = m_Passes[idx];
    p          = {};
    p.name     = name;
    p.type     = type;
    p.enabled  = enabled;
    p.callback = std::move(cb);

    p.readCount = 0;
    for (const RGPassUse& u : reads)
    {
        VERIFY(p.readCount < 8);
        p.reads[p.readCount++] = u;
    }

    p.writeCount = 0;
    for (const RGPassUse& u : writes)
    {
        VERIFY(p.writeCount < 8);
        p.writes[p.writeCount++] = u;
    }

    return { (u16)idx, 0 };
}

// ---------------------------------------------------------------------------
// Compile — no-op for A1; reserved for culling/reordering in A2+
// ---------------------------------------------------------------------------
void CFrameGraph::Compile()
{
}

// ---------------------------------------------------------------------------
// InsertBarriers — emit a single batched vkCmdPipelineBarrier2 before a pass
// ---------------------------------------------------------------------------
void CFrameGraph::InsertBarriers(VkCommandBuffer cmd, RGPass& pass)
{
    // Max 8 reads + 8 writes = 16 barriers per pass
    constexpr u32 MAX_USES = 16;
    VkImageMemoryBarrier2  imgBarriers[MAX_USES] = {};
    VkBufferMemoryBarrier2 bufBarriers[MAX_USES] = {};
    u32 imgCount = 0;
    u32 bufCount = 0;

    auto ProcessUse = [&](const RGPassUse& use)
    {
        if (!use.handle.IsValid())               return;
        if (use.handle.index >= m_ResourceCount) return;
        RGResource& res = m_Resources[use.handle.index];
        if (res.version != use.handle.version)   return;  // stale handle

        const RGResourceState required = UsageToBarrierState(use.usage);

        if (res.isBuffer)
        {
            // Read-after-read on buffers needs no memory barrier (Vulkan §7.1:
            // "If there is no write hazard, a memory dependency is not needed").
            // Only emit a barrier when at least one side has write access.
            const bool readAfterRead = IsReadOnlyAccess(res.state.access)
                                    && IsReadOnlyAccess(required.access);
            const bool same = (res.state.stage  == required.stage
                            && res.state.access == required.access);
            if (!same && !readAfterRead)
            {
                VERIFY(bufCount < MAX_USES);
                VkBufferMemoryBarrier2& b = bufBarriers[bufCount++];
                b                        = {};
                b.sType                  = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
                b.srcStageMask           = res.state.stage;
                b.srcAccessMask          = res.state.access;
                b.dstStageMask           = required.stage;
                b.dstAccessMask          = required.access;
                b.srcQueueFamilyIndex    = VK_QUEUE_FAMILY_IGNORED;
                b.dstQueueFamilyIndex    = VK_QUEUE_FAMILY_IGNORED;
                b.buffer                 = res.buffer;
                b.offset                 = 0;
                b.size                   = VK_WHOLE_SIZE;
            }
        }
        else
        {
            // Read-after-read in the same read-only layout needs no barrier
            const bool readAfterRead = (res.state.layout == required.layout
                                     && IsReadOnlyLayout(required.layout));
            if (!readAfterRead)
            {
                VERIFY(imgCount < MAX_USES);
                VkImageMemoryBarrier2& b          = imgBarriers[imgCount++];
                b                                 = {};
                b.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                b.srcStageMask                    = res.state.stage;
                b.srcAccessMask                   = res.state.access;
                b.dstStageMask                    = required.stage;
                b.dstAccessMask                   = required.access;
                b.oldLayout                       = res.state.layout;
                b.newLayout                       = required.layout;
                b.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
                b.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
                b.image                           = res.image;
                b.subresourceRange.aspectMask     = res.aspect;
                b.subresourceRange.baseMipLevel   = 0;
                b.subresourceRange.levelCount     = res.mipLevels;
                b.subresourceRange.baseArrayLayer = 0;
                b.subresourceRange.layerCount     = res.layerCount;
            }
        }

        // Advance tracked state to the required state
        res.state = required;
    };

    for (u8 i = 0; i < pass.readCount;  i++) ProcessUse(pass.reads[i]);
    for (u8 i = 0; i < pass.writeCount; i++) ProcessUse(pass.writes[i]);

    if (imgCount == 0 && bufCount == 0)
        return;

    VkDependencyInfo dep             = {};
    dep.sType                        = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dep.imageMemoryBarrierCount      = imgCount;
    dep.pImageMemoryBarriers         = imgBarriers;
    dep.bufferMemoryBarrierCount     = bufCount;
    dep.pBufferMemoryBarriers        = bufBarriers;
    vkCmdPipelineBarrier2(cmd, &dep);
}

// ---------------------------------------------------------------------------
// Execute
// ---------------------------------------------------------------------------
void CFrameGraph::Execute(VkCommandBuffer cmd)
{
    for (u32 i = 0; i < m_PassCount; i++)
    {
        RGPass& pass = m_Passes[i];
        if (!pass.enabled)
            continue;

        if (pass.type != RGPassType::EXTERNAL)
            InsertBarriers(cmd, pass);

        if (pass.callback)
            pass.callback(cmd);
    }
}

// ---------------------------------------------------------------------------
// Reset — clear passes each frame, keep resource layout state
// ---------------------------------------------------------------------------
void CFrameGraph::Reset()
{
    for (u32 i = 0; i < m_PassCount; i++)
        m_Passes[i] = RGPass{};  // destructs std::function, releases lambda captures
    m_PassCount = 0;
}

// ---------------------------------------------------------------------------
// FullReset — clear everything (resize / device recreate)
// ---------------------------------------------------------------------------
void CFrameGraph::FullReset()
{
    Reset();
    for (u32 i = 0; i < m_ResourceCount; i++)
        m_Resources[i] = RGResource{};
    m_ResourceCount = 0;
}

// ---------------------------------------------------------------------------
// SetImageLayout — update tracked layout after an external operation
// ---------------------------------------------------------------------------
void CFrameGraph::SetImageLayout(RGHandle h, VkImageLayout layout)
{
    if (!h.IsValid() || h.index >= m_ResourceCount) return;
    RGResource& res = m_Resources[h.index];
    if (res.version != h.version)                   return;
    res.state.layout = layout;
    res.state.stage  = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
    res.state.access = 0;
}

// ---------------------------------------------------------------------------
// SetImageState — update full tracked state (layout + stage + access) after
// an external operation. Use when the src stage/access of writes is known.
// ---------------------------------------------------------------------------
void CFrameGraph::SetImageState(RGHandle h, VkImageLayout layout,
                                 VkPipelineStageFlags2 stage, VkAccessFlags2 access)
{
    if (!h.IsValid() || h.index >= m_ResourceCount) return;
    RGResource& res = m_Resources[h.index];
    if (res.version != h.version)                   return;
    res.state.layout = layout;
    res.state.stage  = stage;
    res.state.access = access;
}

// ---------------------------------------------------------------------------
// SetBufferState — update tracked stage+access of a buffer after external op
// ---------------------------------------------------------------------------
void CFrameGraph::SetBufferState(RGHandle h, VkPipelineStageFlags2 stage, VkAccessFlags2 access)
{
    if (!h.IsValid() || h.index >= m_ResourceCount) return;
    RGResource& res = m_Resources[h.index];
    if (res.version != h.version)                   return;
    res.state.stage  = stage;
    res.state.access = access;
}

// ---------------------------------------------------------------------------
// FindResource
// ---------------------------------------------------------------------------
RGHandle CFrameGraph::FindResource(const char* name) const
{
    for (u32 i = 0; i < m_ResourceCount; i++)
    {
        if (m_Resources[i].name && xr_strcmp(m_Resources[i].name, name) == 0)
            return { (u16)i, m_Resources[i].version };
    }
    return RGHandle::Invalid();
}

// ---------------------------------------------------------------------------
// GetTrackedLayout — query current layout without constructing a full handle
// ---------------------------------------------------------------------------
VkImageLayout CFrameGraph::GetTrackedLayout(const char* name) const
{
    RGHandle h = FindResource(name);
    if (!h.IsValid() || h.index >= m_ResourceCount)
        return VK_IMAGE_LAYOUT_UNDEFINED;
    return m_Resources[h.index].state.layout;
}

} // namespace VK
