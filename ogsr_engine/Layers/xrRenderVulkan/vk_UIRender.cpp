// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan - vkUIRender: real IUIRender implementation.
//
// Adapted from the monolith UI render path. Differences worth knowing about:
// RCache.m_Cmd is replaced by the global g_VkUI_FrameCmd (set by
// CRender::Begin per frame), the RCache.stat.{calls,verts} counters are
// dropped (we throttle our own logs locally), and we go through IVkUIShader
// (vk_ui_shader.h) for descriptor-set lookup.
//
// Vertex format: FVF::TL = vec4 pos + u32 color + vec2 uv = 28 bytes.
// pttTL = 4-component pos + colour + uv (UI default).
// pttLIT — supported too in case the engine asks for lit primitives.

#include "stdafx.h"
#include "vk_core.h"
#include "vk_UIPipeline.h"
#include "vk_ui_shader.h"
#include "vk_swapchain.h"
#include "../../Include/xrRender/UIRender.h"
#include "../xrRender/FVF.h"   // FVF::TL, FVF::LIT
#include <set>

// Tracks the last UI shader passed to SetShader so vkUISequenceVideoItem
// can grab it inside CaptureTexture (R4 equivalent: RCache.get_ActiveTexture(0)).
IVkUIShader* g_pLastVkUIShader = nullptr;

namespace {

class vkUIRender final : public IUIRender
{
    ePrimitiveType m_PrimitiveType = ptNone;
    ePointType     m_PointType     = pttNone;
    u32            m_MaxVerts      = 0;
    u32            m_VertCount     = 0;
    IUIShader*     m_pCurrentShader = nullptr;
    VkDescriptorSet m_CurrentTextureSet = VK_NULL_HANDLE;

    FVF::TL*  m_pWriteTL      = nullptr;
    FVF::TL*  m_pWriteTLStart = nullptr;
    FVF::LIT* m_pWriteLIT     = nullptr;
    FVF::LIT* m_pWriteLITStart= nullptr;

public:
    void CreateUIGeom() override  { /* VulkanUI::Create() done at engine init */ }
    void DestroyUIGeom() override { /* VulkanUI::Destroy() at engine teardown */ }

    void SetShader(IUIShader& shader) override
    {
        m_pCurrentShader = &shader;
        IVkUIShader* vkShader = dynamic_cast<IVkUIShader*>(&shader);
        g_pLastVkUIShader = vkShader;
        if (vkShader) {
            m_CurrentTextureSet = vkShader->GetDescriptorSet();
            if (m_CurrentTextureSet == VK_NULL_HANDLE) {
                m_CurrentTextureSet = VulkanUI::s_WhiteTextureSet;
                static std::set<const void*> warned;
                if (warned.insert(vkShader).second)
                    Msg("![VK-UI] SetShader: descriptor=NULL, using white fallback (texW=%u texH=%u)",
                        vkShader->GetTextureWidth(), vkShader->GetTextureHeight());
            }
        } else {
            m_CurrentTextureSet = VulkanUI::s_WhiteTextureSet;
        }
    }

    void SetAlphaRef(int) override { /* alpha handled by blend state */ }

    void SetScissor(Irect* rect) override
    {
        VkCommandBuffer cmd = g_VkUI_FrameCmd;
        if (!cmd) {
            // Deferred: queue scissor change for replay.
            if (VulkanUI::s_DeferredCmdCount < VulkanUI::MAX_DEFERRED_CMDS) {
                auto& dcmd = VulkanUI::s_DeferredCmds[VulkanUI::s_DeferredCmdCount++];
                if (rect) {
                    dcmd.type = VulkanUI::DeferredUICmd::Scissor;
                    dcmd.scissorRect.offset.x      = rect->x1;
                    dcmd.scissorRect.offset.y      = rect->y1;
                    dcmd.scissorRect.extent.width  = rect->x2 - rect->x1;
                    dcmd.scissorRect.extent.height = rect->y2 - rect->y1;
                } else {
                    dcmd.type = VulkanUI::DeferredUICmd::ResetScissor;
                }
            } else { ++VulkanUI::s_FrameStats.droppedCmds; }
            return;
        }

        if (!VulkanUI::s_bUIPassActive) return;
        if (rect) {
            VkRect2D scissor{};
            scissor.offset.x      = rect->x1;
            scissor.offset.y      = rect->y1;
            scissor.extent.width  = rect->x2 - rect->x1;
            scissor.extent.height = rect->y2 - rect->y1;
            vkCmdSetScissor(cmd, 0, 1, &scissor);
        } else {
            VkRect2D scissor{};
            scissor.extent = Swapchain.m_Extent;
            vkCmdSetScissor(cmd, 0, 1, &scissor);
        }
    }

    void GetActiveTextureResolution(Fvector2& res) override
    {
        if (m_pCurrentShader) {
            if (auto* vkShader = dynamic_cast<IVkUIShader*>(m_pCurrentShader)) {
                res.set(float(vkShader->GetTextureWidth()), float(vkShader->GetTextureHeight()));
                return;
            }
        }
        res.set(1024.0f, 1024.0f);
    }

    void StartPrimitive(u32 maxVerts, ePrimitiveType primType, ePointType pointType) override
    {
        m_PrimitiveType = primType;
        m_PointType     = pointType;
        m_MaxVerts      = maxVerts;
        m_VertCount     = 0;

        if (!VulkanUI::s_pMappedVB) { m_pWriteTL = nullptr; m_pWriteLIT = nullptr; return; }

        const u32 vertexSize    = (pointType == pttTL) ? sizeof(FVF::TL) : sizeof(FVF::LIT);
        const u32 requiredBytes = maxVerts * vertexSize;

        if (VulkanUI::s_UIVertexOffset + requiredBytes > VulkanUI::VERTEX_BUFFER_SIZE) {
            // CORRUPTING wrap — the GPU hasn't yet consumed pending draws,
            // resetting offset overwrites their vertex data. Bumping
            // VERTEX_BUFFER_SIZE is the right fix; log so we know if we hit it.
            static u32 lastWarnFrame = 0;
            if (Device.dwFrame > lastWarnFrame + 60) {
                Msg("![VK-UI] vertex buffer wrap mid-frame (size=%u, need=%u) — UI corruption likely",
                    (u32)VulkanUI::VERTEX_BUFFER_SIZE, requiredBytes);
                lastWarnFrame = Device.dwFrame;
            }
            VulkanUI::s_UIVertexOffset = 0;
            ++VulkanUI::s_FrameStats.bufferWraps;
        }

        // Writes land in THIS frame-slot's ring region (s_VBBase) — see
        // VulkanUI::OnFrameBegin for the frames-in-flight discipline.
        u8* base = static_cast<u8*>(VulkanUI::s_pMappedVB) + VulkanUI::s_VBBase + VulkanUI::s_UIVertexOffset;
        if (pointType == pttTL) {
            m_pWriteTL      = reinterpret_cast<FVF::TL*>(base);
            m_pWriteTLStart = m_pWriteTL;
        } else {
            m_pWriteLIT      = reinterpret_cast<FVF::LIT*>(base);
            m_pWriteLITStart = m_pWriteLIT;
        }
    }

    void PushPoint(float x, float y, float z, u32 C, float u, float v) override
    {
        if (m_PointType == pttTL) {
            if (m_pWriteTL) { m_pWriteTL->set(x, y, z, 1.0f, C, u, v); ++m_pWriteTL; ++m_VertCount; }
        } else if (m_PointType == pttLIT) {
            if (m_pWriteLIT) { m_pWriteLIT->set(x, y, z, C, u, v); ++m_pWriteLIT; ++m_VertCount; }
        }
    }

    void FlushPrimitive() override
    {
        const u32 vertCount = m_VertCount;
        if (vertCount == 0) { m_pWriteTL = nullptr; m_pWriteLIT = nullptr; return; }

        VkCommandBuffer cmd = g_VkUI_FrameCmd;
        const u32 vertexSize = (m_PointType == pttTL) ? sizeof(FVF::TL) : sizeof(FVF::LIT);

        // Select the pipeline matching the requested primitive topology. R4 maps
        // ptLineList/ptLineStrip to D3D line topology; without this the crosshair's
        // line vertices assemble into stray stretched triangles (the "sky spike").
        VkPipeline pipe = VulkanUI::s_Pipeline;
        if (m_PrimitiveType == IUIRender::ptLineList  && VulkanUI::s_PipelineLineList)  pipe = VulkanUI::s_PipelineLineList;
        else if (m_PrimitiveType == IUIRender::ptLineStrip && VulkanUI::s_PipelineLineStrip) pipe = VulkanUI::s_PipelineLineStrip;

        if (!cmd) {
            // Deferred path: vertex data already in mapped buffer; queue draw.
            if (VulkanUI::s_DeferredCmdCount < VulkanUI::MAX_DEFERRED_CMDS) {
                auto& dcmd = VulkanUI::s_DeferredCmds[VulkanUI::s_DeferredCmdCount++];
                dcmd.type               = VulkanUI::DeferredUICmd::Draw;
                dcmd.vertexBufferOffset = VulkanUI::s_VBBase + VulkanUI::s_UIVertexOffset;  // absolute (ring slot)
                dcmd.vertexCount        = vertCount;
                dcmd.textureSet         = m_CurrentTextureSet;
                dcmd.pipeline           = pipe;
            } else { ++VulkanUI::s_FrameStats.droppedCmds; }
            VulkanUI::s_FrameStats.totalVerts += vertCount;
            VulkanUI::s_UIVertexOffset += vertCount * vertexSize;
            m_pWriteTL = nullptr; m_pWriteLIT = nullptr; m_VertCount = 0;
            return;
        }

        // Immediate path.
        VulkanUI::FlushVertexBuffer();
        if (!VulkanUI::s_bUIPassActive) VulkanUI::BeginUIPassInternal();
        if (VulkanUI::s_Pipeline == VK_NULL_HANDLE) return;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        float screenSize[2] = { (float)Device.dwWidth, (float)Device.dwHeight };
        vkCmdPushConstants(cmd, VulkanUI::s_PipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, 8, screenSize);

        VkBuffer     vbs[]     = { VulkanUI::GetVertexBufferHandle() };
        VkDeviceSize offsets[] = { VulkanUI::s_VBBase + VulkanUI::s_UIVertexOffset };
        vkCmdBindVertexBuffers(cmd, 0, 1, vbs, offsets);

        VkDescriptorSet texSet = (m_CurrentTextureSet != VK_NULL_HANDLE) ? m_CurrentTextureSet : VulkanUI::s_WhiteTextureSet;
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, VulkanUI::s_PipelineLayout, 0, 1, &texSet, 0, nullptr);

        vkCmdDraw(cmd, vertCount, 1, 0, 0);

        VulkanUI::s_UIVertexOffset += vertCount * vertexSize;
        m_pWriteTL = nullptr; m_pWriteLIT = nullptr; m_VertCount = 0;
    }

    LPCSTR UpdateShaderName(LPCSTR /*tex_name*/, LPCSTR sh_name) override { return sh_name; }

    void CacheSetXformWorld(const Fmatrix&) override {}
    void CacheSetCullMode(CullMode)         override {}
};

vkUIRender g_UIRenderImpl_VK;
} // namespace

IUIRender* GetUIRenderImpl_VK() { return &g_UIRenderImpl_VK; }
