// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

#include "stdafx.h"
#include "vk_pass_particles.h"
#include "vk_Particles.h"
#include "vk_pass_world.h"        // g_DynamicVisuals / DynVisual
#include "vk_shaders.h"
#include "vk_texture.h"
#include "vk_buffer.h"
#include "vk_swapchain.h"
#include "vk_scene_color.h"       // HDR scene target format
#include "vk_command_buffer.h"    // CommandManager.GetCurrentFrame() / FRAMES_IN_FLIGHT
#include "vk_pipeline_cache.h"    // PipelineCache::GetCacheObject()
#include "HW_Vulkan.h"

#include "../xrRender/FVF.h"
#include "../../xr_3da/fmesh.h"   // MT_PARTICLE_EFFECT / MT_PARTICLE_GROUP

#include <unordered_map>
#include <string>

namespace VK {

namespace {
    constexpr u32 kFramesInFlight = CVulkanCommandManager::FRAMES_IN_FLIGHT;

    // Particle vertex = FVF::LIT (24 bytes): vec3 pos, D3DCOLOR, vec2 uv.
    constexpr u32 kVtxStride = sizeof(float) * 3 + sizeof(u32) + sizeof(float) * 2;  // 24

    // Per-frame dynamic vertex ring (host-visible). 4 MB ≈ 174k verts ≈ 29k
    // particles per frame — far above any realistic on-screen count.
    constexpr VkDeviceSize kRingBytes = 4 * 1024 * 1024;
    constexpr u32          kRingVerts = (u32)(kRingBytes / kVtxStride);

    VkShaderModule   s_VS = VK_NULL_HANDLE;
    VkShaderModule   s_FS = VK_NULL_HANDLE;
    VkPipelineLayout s_Layout = VK_NULL_HANDLE;
    VkPipeline       s_Pipelines[PBM_COUNT] = {};   // lazy per blend mode
    VkDescriptorSetLayout s_SetLayout = VK_NULL_HANDLE;
    VkDescriptorPool s_Pool   = VK_NULL_HANDLE;
    VkSampler        s_Sampler = VK_NULL_HANDLE;

    CVulkanBuffer    s_Ring[kFramesInFlight];

    bool             s_Init = false;

    // Sprite texture descriptor cache, keyed by texture name. Many effects share
    // a handful of textures (fire/smoke/sparks), so this bounds the pool.
    struct PTex { CVulkanTexture* tex = nullptr; VkDescriptorSet set = VK_NULL_HANDLE; };
    std::unordered_map<std::string, PTex> s_TexCache;

    void BlendFactors(EParticleBlendMode mode, VkBlendFactor& src, VkBlendFactor& dst)
    {
        switch (mode) {
        case PBM_SET:       src = VK_BLEND_FACTOR_ONE;       dst = VK_BLEND_FACTOR_ZERO;                break;
        case PBM_ADD:       src = VK_BLEND_FACTOR_ONE;       dst = VK_BLEND_FACTOR_ONE;                 break;
        case PBM_MUL:       src = VK_BLEND_FACTOR_DST_COLOR; dst = VK_BLEND_FACTOR_ZERO;                break;
        case PBM_MUL_2X:    src = VK_BLEND_FACTOR_DST_COLOR; dst = VK_BLEND_FACTOR_SRC_COLOR;           break;
        case PBM_ALPHA_ADD: src = VK_BLEND_FACTOR_SRC_ALPHA; dst = VK_BLEND_FACTOR_ONE;                 break;
        case PBM_BLEND:
        default:            src = VK_BLEND_FACTOR_SRC_ALPHA; dst = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA; break;
        }
    }

    VkPipeline BuildPipeline(EParticleBlendMode mode)
    {
        VkVertexInputBindingDescription bind{};
        bind.binding   = 0;
        bind.stride    = kVtxStride;
        bind.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkVertexInputAttributeDescription attrs[3]{};
        attrs[0] = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0  };   // pos
        attrs[1] = { 1, 0, VK_FORMAT_B8G8R8A8_UNORM,   12 };   // D3DCOLOR → logical RGBA
        attrs[2] = { 2, 0, VK_FORMAT_R32G32_SFLOAT,    16 };   // uv

        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount   = 1;
        vi.pVertexBindingDescriptions      = &bind;
        vi.vertexAttributeDescriptionCount = 3;
        vi.pVertexAttributeDescriptions    = attrs;

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = s_VS; stages[0].pName = "main";
        stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = s_FS; stages[1].pName = "main";

        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vp{};
        vp.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vp.viewportCount = 1;
        vp.scissorCount  = 1;

        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode    = VK_CULL_MODE_NONE;     // billboards are double-sided
        rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth   = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable  = VK_TRUE;          // occluded by world geometry
        ds.depthWriteEnable = VK_FALSE;         // transparent — don't write depth
        ds.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkBlendFactor src, dst;
        BlendFactors(mode, src, dst);
        VkPipelineColorBlendAttachmentState ba{};
        ba.blendEnable         = VK_TRUE;
        ba.srcColorBlendFactor = src;
        ba.dstColorBlendFactor = dst;
        ba.colorBlendOp        = VK_BLEND_OP_ADD;
        ba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        ba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        ba.alphaBlendOp        = VK_BLEND_OP_ADD;
        ba.colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                 VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1;
        cb.pAttachments    = &ba;

        VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynState{};
        dynState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynState.dynamicStateCount = 2;
        dynState.pDynamicStates    = dyn;

        VkFormat colorFormat = VK::SceneColor::Format();
        VkPipelineRenderingCreateInfo prci{};
        prci.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        prci.colorAttachmentCount    = 1;
        prci.pColorAttachmentFormats = &colorFormat;
        prci.depthAttachmentFormat   = Swapchain.m_DepthFormat;

        VkGraphicsPipelineCreateInfo pi{};
        pi.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pi.pNext               = &prci;
        pi.stageCount          = 2;
        pi.pStages             = stages;
        pi.pVertexInputState   = &vi;
        pi.pInputAssemblyState = &ia;
        pi.pViewportState      = &vp;
        pi.pRasterizationState = &rs;
        pi.pMultisampleState   = &ms;
        pi.pDepthStencilState  = &ds;
        pi.pColorBlendState    = &cb;
        pi.pDynamicState       = &dynState;
        pi.layout              = s_Layout;

        VkPipeline pipe = VK_NULL_HANDLE;
        if (vkCreateGraphicsPipelines(VulkanHW.m_Device, PipelineCache::GetCacheObject(),
                                      1, &pi, nullptr, &pipe) != VK_SUCCESS) {
            Msg("![VK Particles] pipeline create failed (blend=%d)", (int)mode);
            return VK_NULL_HANDLE;
        }
        return pipe;
    }
}

bool ParticlePass_Init()
{
    if (s_Init) return true;

    if (!g_ShaderManager) g_ShaderManager = xr_new<VK::CVulkanSPIRVLoader>();
    s_VS = g_ShaderManager->Load("particle.vert.spv");
    s_FS = g_ShaderManager->Load("particle.frag.spv");
    if (s_VS == VK_NULL_HANDLE || s_FS == VK_NULL_HANDLE) {
        Msg("![VK Particles] failed to load particle shaders");
        return false;
    }

    // Set 0: binding 0 = sprite (combined image sampler, FS).
    {
        VkDescriptorSetLayoutBinding b{};
        b.binding         = 0;
        b.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b.descriptorCount = 1;
        b.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 1;
        lci.pBindings    = &b;
        if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &s_SetLayout) != VK_SUCCESS) {
            Msg("![VK Particles] CreateDescriptorSetLayout failed");
            return false;
        }
    }

    // Pool — one set per unique sprite texture (~256 generous).
    {
        constexpr u32 kMaxSets = 256;
        VkDescriptorPoolSize ps{};
        ps.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ps.descriptorCount = kMaxSets;
        VkDescriptorPoolCreateInfo pci{};
        pci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets       = kMaxSets;
        pci.poolSizeCount = 1;
        pci.pPoolSizes    = &ps;
        if (vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &s_Pool) != VK_SUCCESS) {
            Msg("![VK Particles] CreateDescriptorPool failed");
            return false;
        }
    }

    {
        VkSamplerCreateInfo si{};
        si.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter    = VK_FILTER_LINEAR;
        si.minFilter    = VK_FILTER_LINEAR;
        si.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.minLod       = 0.0f;
        si.maxLod       = VK_LOD_CLAMP_NONE;
        if (vkCreateSampler(VulkanHW.m_Device, &si, nullptr, &s_Sampler) != VK_SUCCESS) {
            Msg("![VK Particles] CreateSampler failed");
            return false;
        }
    }

    // Pipeline layout: set 0 + push mat4 viewProj (VS).
    {
        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pc.offset     = 0;
        pc.size       = sizeof(Fmatrix);
        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &s_SetLayout;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pc;
        if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_Layout) != VK_SUCCESS) {
            Msg("![VK Particles] CreatePipelineLayout failed");
            return false;
        }
    }

    for (u32 i = 0; i < kFramesInFlight; ++i)
        s_Ring[i].Create(kRingBytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);

    s_Init = true;
    Msg("[VK Particles] Init OK (ring=%u verts/frame x%u, per-blend pipelines lazy)", kRingVerts, kFramesInFlight);
    return true;
}

void ParticlePass_Destroy()
{
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;

    for (auto& kv : s_TexCache) {
        if (kv.second.tex) { kv.second.tex->Destroy(); xr_delete(kv.second.tex); }
    }
    s_TexCache.clear();

    for (u32 i = 0; i < kFramesInFlight; ++i) s_Ring[i].Destroy();

    for (u32 i = 0; i < PBM_COUNT; ++i) {
        if (s_Pipelines[i]) { vkDestroyPipeline(VulkanHW.m_Device, s_Pipelines[i], nullptr); s_Pipelines[i] = VK_NULL_HANDLE; }
    }
    if (s_Layout)    { vkDestroyPipelineLayout(VulkanHW.m_Device, s_Layout, nullptr); s_Layout = VK_NULL_HANDLE; }
    if (s_Pool)      { vkDestroyDescriptorPool(VulkanHW.m_Device, s_Pool, nullptr); s_Pool = VK_NULL_HANDLE; }
    if (s_Sampler)   { vkDestroySampler(VulkanHW.m_Device, s_Sampler, nullptr); s_Sampler = VK_NULL_HANDLE; }
    if (s_SetLayout) { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_SetLayout, nullptr); s_SetLayout = VK_NULL_HANDLE; }
    s_VS = s_FS = VK_NULL_HANDLE;
    s_Init = false;
}

namespace ParticlePass {

bool             Ready()      { return s_Init; }
VkPipelineLayout GetLayout()  { return s_Layout; }

VkPipeline GetPipeline(EParticleBlendMode mode)
{
    if (!s_Init || mode < 0 || mode >= PBM_COUNT) return VK_NULL_HANDLE;
    if (mode == PBM_DISTORT) return VK_NULL_HANDLE;       // unsupported — skip
    if (s_Pipelines[mode] == VK_NULL_HANDLE)
        s_Pipelines[mode] = BuildPipeline(mode);
    return s_Pipelines[mode];
}

VkDescriptorSet GetTextureSet(const char* texture_name)
{
    if (!s_Init || !texture_name || !texture_name[0]) return VK_NULL_HANDLE;

    std::string key(texture_name);
    auto it = s_TexCache.find(key);
    if (it != s_TexCache.end()) return it->second.set;

    // Resolve "$game_textures$\<name>.dds".
    string_path leaf, full;
    xr_sprintf(leaf, "%s.dds", texture_name);
    FS.update_path(full, "$game_textures$", leaf);

    PTex entry;
    if (FS.exist(full)) {
        entry.tex = xr_new<CVulkanTexture>();
        if (!entry.tex->LoadDDS(full, /*applyBCSwizzle*/ false)) {
            xr_delete(entry.tex);
            entry.tex = nullptr;
        }
    }
    if (!entry.tex) {
        Msg("![VK Particles] sprite texture not found: '%s'", texture_name);
        s_TexCache.emplace(std::move(key), entry);   // negative cache (set = null)
        return VK_NULL_HANDLE;
    }

    VkDescriptorSetAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool     = s_Pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts        = &s_SetLayout;
    if (vkAllocateDescriptorSets(VulkanHW.m_Device, &ai, &entry.set) != VK_SUCCESS) {
        Msg("![VK Particles] descriptor pool exhausted for '%s'", texture_name);
        entry.tex->Destroy(); xr_delete(entry.tex);
        entry.tex = nullptr; entry.set = VK_NULL_HANDLE;
        s_TexCache.emplace(std::move(key), entry);
        return VK_NULL_HANDLE;
    }

    VkDescriptorImageInfo ii{};
    ii.sampler     = s_Sampler;
    ii.imageView   = entry.tex->GetView();
    ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkWriteDescriptorSet w{};
    w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet          = entry.set;
    w.dstBinding      = 0;
    w.descriptorCount = 1;
    w.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo      = &ii;
    vkUpdateDescriptorSets(VulkanHW.m_Device, 1, &w, 0, nullptr);

    VkDescriptorSet set = entry.set;
    s_TexCache.emplace(std::move(key), entry);
    return set;
}

}  // namespace ParticlePass

void Pass_Particles(FrameContext& ctx)
{
    if (!s_Init)                   return;
    if (ctx.cmd == VK_NULL_HANDLE) return;
    if (g_DynamicVisuals.empty())  return;

    // Flatten visible particle visuals into leaf effects.
    static xr_vector<vkCParticleEffect*> s_effects;
    s_effects.clear();
    for (const DynVisual& d : g_DynamicVisuals) {
        if (!d.vis) continue;
        const u32 t = d.vis->Type;
        if (t != MT_PARTICLE_EFFECT && t != MT_PARTICLE_GROUP) continue;
        static_cast<vkParticleVisual*>(d.vis)->CollectEffects(s_effects);
    }
    if (s_effects.empty()) return;

    VkCommandBuffer cmd = ctx.cmd;
    const u32 slot = CommandManager.GetCurrentFrame();
    CVulkanBuffer& ring = s_Ring[slot];
    if (!ring.IsValid()) return;

    u8* base = (u8*)ring.Map();
    if (!base) return;

    // Begin rendering: preserve scene colour + world depth, no depth write.
    VkRenderingAttachmentInfo cAtt{};
    cAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    cAtt.imageView   = ctx.colorView;
    cAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    cAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;
    cAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingAttachmentInfo dAtt{};
    dAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    dAtt.imageView   = ctx.depthView;
    dAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    dAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;
    dAtt.storeOp     = VK_ATTACHMENT_STORE_OP_DONT_CARE;

    VkRenderingInfo ri{};
    ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea.extent    = ctx.extent;
    ri.layerCount           = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments    = &cAtt;
    ri.pDepthAttachment     = &dAtt;
    vkCmdBeginRendering(cmd, &ri);

    VkViewport vp{};
    vp.x = 0.0f; vp.y = (float)ctx.extent.height;
    vp.width = (float)ctx.extent.width; vp.height = -(float)ctx.extent.height;
    vp.minDepth = 0.0f; vp.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{ {}, ctx.extent };
    vkCmdSetScissor(cmd, 0, 1, &sc);

    // viewProj is constant for the pass (particle verts are world-space).
    if (ctx.viewProj)
        vkCmdPushConstants(cmd, s_Layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Fmatrix), ctx.viewProj);

    VkBuffer vbuf = ring.GetHandle();
    VkDeviceSize voff = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vbuf, &voff);

    VkPipeline      lastPipe = VK_NULL_HANDLE;
    VkDescriptorSet lastSet  = VK_NULL_HANDLE;
    u32 vtxUsed = 0;
    u32 nDraw = 0;

    for (vkCParticleEffect* e : s_effects) {
        if (!e) continue;
        const EParticleBlendMode bm = e->GetBlendMode();
        VkPipeline pipe = ParticlePass::GetPipeline(bm);
        if (pipe == VK_NULL_HANDLE) continue;          // distortion / build failure

        VkDescriptorSet set = e->ResolveTextureSet();
        if (set == VK_NULL_HANDLE) continue;           // missing texture

        if (vtxUsed >= kRingVerts) break;              // ring full
        const u32 avail = kRingVerts - vtxUsed;
        FVF::LIT* dst = (FVF::LIT*)(base + (size_t)vtxUsed * kVtxStride);
        const u32 vcount = e->BuildVertices(dst, avail);
        if (vcount == 0) continue;

        if (pipe != lastPipe) { vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe); lastPipe = pipe; }
        if (set  != lastSet)  { vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_Layout, 0, 1, &set, 0, nullptr); lastSet = set; }

        vkCmdDraw(cmd, vcount, 1, vtxUsed, 0);
        vtxUsed += vcount;
        ++nDraw;
    }

    vkCmdEndRendering(cmd);
    ring.Flush();
    ring.Unmap();

    static bool s_diag = false;
    if (!s_diag && nDraw) {
        Msg("[VK Particles] first frame drawn: effects=%u draws=%u verts=%u", (u32)s_effects.size(), nDraw, vtxUsed);
        s_diag = true;
    }
}

}  // namespace VK
