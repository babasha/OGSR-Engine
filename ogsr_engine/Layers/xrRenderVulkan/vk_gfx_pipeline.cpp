// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

#include "stdafx.h"
#include "vk_gfx_pipeline.h"
#include "vk_pipeline_cache.h"   // VK::PipelineCache::GetCacheObject
#include "HW_Vulkan.h"           // VulkanHW (device)

#include <stdarg.h>

namespace VK
{

GfxPipelineBuilder::GfxPipelineBuilder(VkPipelineLayout layout) : m_layout(layout) {}

VkPipelineShaderStageCreateInfo& GfxPipelineBuilder::AddStage(VkShaderStageFlagBits stage, VkShaderModule m)
{
    VERIFY(m_nStages < kMaxStages);
    VkPipelineShaderStageCreateInfo& s = m_stages[m_nStages++];
    s.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    s.stage  = stage;
    s.module = m;
    s.pName  = "main";
    return s;
}

GfxPipelineBuilder& GfxPipelineBuilder::Vert(VkShaderModule m, const VkSpecializationInfo* spec)
{
    AddStage(VK_SHADER_STAGE_VERTEX_BIT, m).pSpecializationInfo = spec;
    return *this;
}

GfxPipelineBuilder& GfxPipelineBuilder::Frag(VkShaderModule m, const VkSpecializationInfo* spec)
{
    AddStage(VK_SHADER_STAGE_FRAGMENT_BIT, m).pSpecializationInfo = spec;
    return *this;
}

GfxPipelineBuilder& GfxPipelineBuilder::Geom(VkShaderModule m)
{
    AddStage(VK_SHADER_STAGE_GEOMETRY_BIT, m);
    return *this;
}

GfxPipelineBuilder& GfxPipelineBuilder::Tess(VkShaderModule tcs, VkShaderModule tes, u32 patchPoints)
{
    AddStage(VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, tcs);
    AddStage(VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, tes);
    m_tess     = true;
    m_patchPts = patchPoints;
    m_topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
    return *this;
}

GfxPipelineBuilder& GfxPipelineBuilder::Binding(u32 binding, u32 stride, VkVertexInputRate rate)
{
    VERIFY(m_nBindings < kMaxBindings);
    m_bindings[m_nBindings++] = { binding, stride, rate };
    return *this;
}

GfxPipelineBuilder& GfxPipelineBuilder::Bindings(const VkVertexInputBindingDescription* b, u32 count)
{
    VERIFY(m_nBindings + count <= kMaxBindings);
    for (u32 i = 0; i < count; ++i) m_bindings[m_nBindings++] = b[i];
    return *this;
}

GfxPipelineBuilder& GfxPipelineBuilder::Attr(u32 location, u32 binding, VkFormat fmt, u32 offset)
{
    VERIFY(m_nAttribs < kMaxAttribs);
    m_attribs[m_nAttribs++] = { location, binding, fmt, offset };
    return *this;
}

GfxPipelineBuilder& GfxPipelineBuilder::Attrs(const VkVertexInputAttributeDescription* a, u32 count)
{
    VERIFY(m_nAttribs + count <= kMaxAttribs);
    for (u32 i = 0; i < count; ++i) m_attribs[m_nAttribs++] = a[i];
    return *this;
}

GfxPipelineBuilder& GfxPipelineBuilder::Topology(VkPrimitiveTopology t) { m_topology = t; return *this; }

GfxPipelineBuilder& GfxPipelineBuilder::Cull(VkCullModeFlags mode, VkFrontFace front)
{
    m_cull = mode; m_front = front; return *this;
}

GfxPipelineBuilder& GfxPipelineBuilder::Polygon(VkPolygonMode mode, float lineWidth)
{
    m_polygon = mode; m_lineWidth = lineWidth; return *this;
}

GfxPipelineBuilder& GfxPipelineBuilder::DepthBias(float constant, float slope)
{
    m_biasOn = true; m_biasConst = constant; m_biasSlope = slope; return *this;
}

GfxPipelineBuilder& GfxPipelineBuilder::DynamicDepthBias()
{
    m_biasOn = true;
    return Dynamic(VK_DYNAMIC_STATE_DEPTH_BIAS);
}

GfxPipelineBuilder& GfxPipelineBuilder::Depth(bool test, bool write, VkCompareOp op)
{
    m_hasDepth = true; m_depthTest = test; m_depthWrite = write; m_depthOp = op; return *this;
}

GfxPipelineBuilder& GfxPipelineBuilder::Color(VkFormat fmt, VkColorComponentFlags mask)
{
    VERIFY(m_nColor < kMaxColor);
    m_colorFormats[m_nColor] = fmt;
    VkPipelineColorBlendAttachmentState& ba = m_blends[m_nColor];
    ba = {};
    ba.blendEnable    = VK_FALSE;
    ba.colorWriteMask = mask;
    ++m_nColor;
    return *this;
}

GfxPipelineBuilder& GfxPipelineBuilder::BlendAlpha()
{
    VERIFY(m_nColor > 0);
    VkPipelineColorBlendAttachmentState& ba = m_blends[m_nColor - 1];
    ba.blendEnable         = VK_TRUE;
    ba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    ba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    ba.colorBlendOp        = VK_BLEND_OP_ADD;
    ba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    ba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    ba.alphaBlendOp        = VK_BLEND_OP_ADD;
    return *this;
}

GfxPipelineBuilder& GfxPipelineBuilder::BlendAdd()
{
    VERIFY(m_nColor > 0);
    VkPipelineColorBlendAttachmentState& ba = m_blends[m_nColor - 1];
    ba.blendEnable         = VK_TRUE;
    ba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    ba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    ba.colorBlendOp        = VK_BLEND_OP_ADD;
    ba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    ba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    ba.alphaBlendOp        = VK_BLEND_OP_ADD;
    return *this;
}

GfxPipelineBuilder& GfxPipelineBuilder::ColorBlend(VkFormat fmt, const VkPipelineColorBlendAttachmentState& blend)
{
    VERIFY(m_nColor < kMaxColor);
    m_colorFormats[m_nColor] = fmt;
    m_blends[m_nColor]       = blend;
    ++m_nColor;
    return *this;
}

GfxPipelineBuilder& GfxPipelineBuilder::DepthTarget(VkFormat fmt) { m_depthFormat = fmt; return *this; }

GfxPipelineBuilder& GfxPipelineBuilder::Dynamic(VkDynamicState s)
{
    VERIFY(m_nDynamic < kMaxDynamic);
    m_dynamic[m_nDynamic++] = s;
    return *this;
}

GfxPipelineBuilder& GfxPipelineBuilder::RenderingNext(const void* p) { m_renderingNext = p; return *this; }
GfxPipelineBuilder& GfxPipelineBuilder::CreateFlags(VkPipelineCreateFlags f) { m_flags |= f; return *this; }

VkPipeline GfxPipelineBuilder::Build(const char* tagFmt, ...)
{
    VkPipelineVertexInputStateCreateInfo vi{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    vi.vertexBindingDescriptionCount   = m_nBindings;
    vi.pVertexBindingDescriptions      = m_nBindings ? m_bindings : nullptr;
    vi.vertexAttributeDescriptionCount = m_nAttribs;
    vi.pVertexAttributeDescriptions    = m_nAttribs ? m_attribs : nullptr;

    VkPipelineInputAssemblyStateCreateInfo ia{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    ia.topology = m_topology;

    VkPipelineTessellationStateCreateInfo ts{ VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO };
    ts.patchControlPoints = m_patchPts;

    VkPipelineViewportStateCreateInfo vp{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    vp.viewportCount = 1;
    vp.scissorCount  = 1;

    VkPipelineRasterizationStateCreateInfo rs{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rs.polygonMode = m_polygon;
    rs.cullMode    = m_cull;
    rs.frontFace   = m_front;
    rs.lineWidth   = m_lineWidth;
    if (m_biasOn)
    {
        rs.depthBiasEnable         = VK_TRUE;
        rs.depthBiasConstantFactor = m_biasConst;
        rs.depthBiasSlopeFactor    = m_biasSlope;
    }

    VkPipelineMultisampleStateCreateInfo ms{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    ds.depthTestEnable  = m_depthTest  ? VK_TRUE : VK_FALSE;
    ds.depthWriteEnable = m_depthWrite ? VK_TRUE : VK_FALSE;
    ds.depthCompareOp   = m_depthOp;

    VkPipelineColorBlendStateCreateInfo cb{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    cb.attachmentCount = m_nColor;
    cb.pAttachments    = m_nColor ? m_blends : nullptr;

    VkPipelineDynamicStateCreateInfo dynState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dynState.dynamicStateCount = m_nDynamic;
    dynState.pDynamicStates    = m_dynamic;

    VkPipelineRenderingCreateInfo prci{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    prci.pNext                   = m_renderingNext;
    prci.colorAttachmentCount    = m_nColor;
    prci.pColorAttachmentFormats = m_nColor ? m_colorFormats : nullptr;
    prci.depthAttachmentFormat   = m_depthFormat;

    VkGraphicsPipelineCreateInfo pi{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    pi.pNext               = &prci;
    pi.flags               = m_flags;
    pi.stageCount          = m_nStages;
    pi.pStages             = m_stages;
    pi.pVertexInputState   = &vi;
    pi.pInputAssemblyState = &ia;
    pi.pTessellationState  = m_tess ? &ts : nullptr;
    pi.pViewportState      = &vp;
    pi.pRasterizationState = &rs;
    pi.pMultisampleState   = &ms;
    pi.pDepthStencilState  = m_hasDepth ? &ds : nullptr;
    pi.pColorBlendState    = &cb;
    pi.pDynamicState       = &dynState;
    pi.layout              = m_layout;

    VkPipeline handle = VK_NULL_HANDLE;
    const VkResult res = vkCreateGraphicsPipelines(VulkanHW.m_Device, PipelineCache::GetCacheObject(),
                                                   1, &pi, nullptr, &handle);
    if (res != VK_SUCCESS)
    {
        char tag[256];
        va_list args;
        va_start(args, tagFmt);
        vsnprintf(tag, sizeof(tag), tagFmt, args);
        va_end(args);
        Msg("![VK] %s pipeline create failed (vk=%d)", tag, (int)res);
        return VK_NULL_HANDLE;
    }
    return handle;
}

} // namespace VK
