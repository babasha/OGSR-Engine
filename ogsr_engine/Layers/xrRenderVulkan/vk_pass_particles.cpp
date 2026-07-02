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
#include "vk_render_queue.h"      // g_RenderQueue.GlassItems() — glass refraction into the distort RT
#include "vk_Visual.h"            // vkFVisual mesh access (glass pane re-draw)
#include "vk_world_material.h"    // WorldMaterial::isEmisAdd — glow halos skip refraction
#include "vk_pass_skinned.h"      // Skinned_RenderGlassDistort — kinematics panes refraction
#include "vk_shaders.h"
#include "vk_texture.h"
#include "vk_buffer.h"
#include "vk_swapchain.h"
#include "vk_scene_color.h"       // HDR scene target format
#include "vk_command_buffer.h"    // CommandManager.GetCurrentFrame() / FRAMES_IN_FLIGHT
#include "vk_pipeline_cache.h"    // PipelineCache::GetCacheObject()
#include "vk_barriers.h"          // ImageBarrier — distort RT layout flips
#include "HW_Vulkan.h"

#include "vk_volumetrics.h"        // VK::Vol — froxel in-scatter probe (Stage-0 smoke lighting) + ProjTerms
#include "../xrRender/FVF.h"
#include "../../xrParticles/psystem.h"  // PAPI::Particle / ParticleManager — Stage-1 smoke media collect
#include "../../xr_3da/fmesh.h"   // MT_PARTICLE_EFFECT / MT_PARTICLE_GROUP
#include "../../xr_3da/device.h"  // Device.mFullTransform_hud2 — HUD-FOV projection

#include <unordered_map>
#include <string>

// Stage-0 volumetric lighting knobs for smoke billboards (global scope — matches
// the C-linkage console symbols the other r_vol_* knobs use). strength 0 = off /
// old look; clamp bounds the per-froxel radiance added so smoke can't blow to white.
extern float ps_r_vol_smoke;
extern float ps_r_vol_smoke_clamp;
// Stage-1 VMS: near-range cutoff for injected smoke media (tied to the terrain detail
// bubble — capped at r__detail_radius so smoke media lives within the grass/terrain
// zone where the froxel grid is fine; far smoke stays billboard-only).
extern float ps_r_vol_smoke_dist;
extern float ps_r_vol_smoke_dist_full;   // inner full-quality radius (volumetric LOD)
extern int   ps_r__detail_radius;
extern float ps_r_glass_refr;            // r_glass_refr — glass refraction (uneven-pane wobble) strength, 0 = off

namespace VK {

namespace {
    constexpr u32 kFramesInFlight = CVulkanCommandManager::FRAMES_IN_FLIGHT;

    // Particle vertex = FVF::LIT (24 bytes): vec3 pos, D3DCOLOR, vec2 uv.
    constexpr u32 kVtxStride = sizeof(float) * 3 + sizeof(u32) + sizeof(float) * 2;  // 24

    // Push block shared by the particle VS+FS (and, harmlessly, by the rain/wallmark
    // passes that reuse this layout — they only push the first 64 bytes). Must match
    // the PushConstants block in particle.vert/frag.glsl.
    struct ParticlePush {
        Fmatrix viewProj;        // 64  VS
        float   camPosNear[4];   // 16  VS — xyz cam pos, w = froxel near Z
        float   camDirLogFN[4];  // 16  VS — xyz cam forward, w = log2(far/near)
        float   volParams[4];    // 16  FS — x = probe strength, yz = 1/extent
    };
    static_assert(sizeof(ParticlePush) == 112, "ParticlePush must match the shader push block");

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

    // Set 1 — the froxel in-scatter volume (Stage-0 smoke lighting). One persistent
    // set pointed at Vol's scatter view (eager-created, so valid from init); rebound
    // if Vol ever regenerates. Bound for every particle draw because particle.frag
    // statically references it (the sample itself is gated on volParams.x).
    // Sentinel "no Vol generation bound yet" — != any real Vol::Generation() so the
    // first EnsureVolSet() always writes the descriptor.
    constexpr u32 kVolGenUnbound = 0xFFFFFFFFu;

    VkDescriptorSetLayout s_VolSetLayout = VK_NULL_HANDLE;
    VkDescriptorSet       s_VolSet       = VK_NULL_HANDLE;
    u32                   s_VolBoundGen  = kVolGenUnbound;

    CVulkanBuffer    s_Ring[kFramesInFlight];

    bool             s_Init = false;

    // Heat-haze distortion target (PBM_DISTORT): full-res RGBA8, neutral
    // (0.5, 0.5). Produced and consumed inside the same command buffer every
    // frame, so a single image is safe across frames-in-flight (same argument
    // as the bloom ping-pong). Created lazily on first distort sighting.
    VkImage          s_DistortImg   = VK_NULL_HANDLE;
    VmaAllocation    s_DistortAlloc = VK_NULL_HANDLE;
    VkImageView      s_DistortView  = VK_NULL_HANDLE;
    VkExtent2D       s_DistortExtent{};
    u32              s_DistortGen   = 0;
    bool             s_DistortFirst = true;        // image still UNDEFINED
    VkPipeline       s_DistortPipe  = VK_NULL_HANDLE;
    // Wallmark variants: generic decals = MODULATE2X + wallmark.frag (R4
    // effects_wallmark.s parity); blood decals = alpha blend + the SSS
    // effects_wallmark_blood port (readable red on dark clothing).
    VkShaderModule   s_WallmarkFS   = VK_NULL_HANDLE;
    VkPipeline       s_WallmarkPipe = VK_NULL_HANDLE;
    VkShaderModule   s_BloodFS      = VK_NULL_HANDLE;
    VkPipeline       s_BloodPipe    = VK_NULL_HANDLE;
    // Rain variants: splashes = shape from tex ALPHA (fx_rain rgb is a
    // refraction normal map); drop streaks = fully PROCEDURAL shape (the
    // fx_rain alpha reads ~0 through our loader) — see rain*.frag.glsl.
    VkShaderModule   s_RainFS       = VK_NULL_HANDLE;
    VkPipeline       s_RainPipe     = VK_NULL_HANDLE;
    VkShaderModule   s_RainDropFS   = VK_NULL_HANDLE;
    VkPipeline       s_RainDropPipe = VK_NULL_HANDLE;
    constexpr VkFormat kDistortFormat = VK_FORMAT_R8G8B8A8_UNORM;

    // ---- Glass refraction (r_glass_refr) ------------------------------------
    // The late-glass panes (RenderQueue::GlassItems) are re-drawn into the same
    // distortion RT with a procedural "uneven old glass" wobble — the tonemap
    // then bends whatever is behind the pane. Two pipelines (world stride-32
    // sub-layouts: base UV at offset 24 = lmap, 28 = vert-lit); push = mvp+strength.
    VkPipeline       s_GlassDistortPipe[2] = {};   // [0]=tcOffset 24, [1]=28
    VkPipelineLayout s_GlassDistortLayout  = VK_NULL_HANDLE;
    VkShaderModule   s_GlassVS = VK_NULL_HANDLE, s_GlassFS = VK_NULL_HANDLE;

    struct GlassDistortPush { Fmatrix mvp; float strength; float pad[3]; };

    VkPipeline BuildGlassDistortPipeline(u32 tcOffset)
    {
        if (s_GlassDistortLayout == VK_NULL_HANDLE) {
            VkPushConstantRange pcr{ VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(GlassDistortPush) };
            VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
            if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_GlassDistortLayout) != VK_SUCCESS)
                return VK_NULL_HANDLE;
        }
        if (!s_GlassVS) s_GlassVS = g_ShaderManager->Load("glass_distort.vert.spv");
        if (!s_GlassFS) s_GlassFS = g_ShaderManager->Load("glass_distort.frag.spv");
        if (!s_GlassVS || !s_GlassFS) return VK_NULL_HANDLE;

        VkVertexInputBindingDescription bind{ 0, 32, VK_VERTEX_INPUT_RATE_VERTEX };
        VkVertexInputAttributeDescription attrs[2] = {
            { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0 },
            { 1, 0, VK_FORMAT_R16G16_SSCALED,   tcOffset },
        };
        VkPipelineVertexInputStateCreateInfo vi{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
        vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &bind;
        vi.vertexAttributeDescriptionCount = 2; vi.pVertexAttributeDescriptions = attrs;

        VkPipelineShaderStageCreateInfo st[2]{};
        st[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;   st[0].module = s_GlassVS; st[0].pName = "main";
        st[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; st[1].module = s_GlassFS; st[1].pName = "main";

        VkPipelineInputAssemblyStateCreateInfo ia{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vp{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
        vp.viewportCount = 1; vp.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
        rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
        ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_FALSE;   // occluded panes must not warp
        ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineColorBlendAttachmentState ba{};
        ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        ba.blendEnable         = VK_TRUE;                       // same as the haze sprites
        ba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        ba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        ba.colorBlendOp        = VK_BLEND_OP_ADD;
        ba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        ba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        ba.alphaBlendOp        = VK_BLEND_OP_ADD;
        VkPipelineColorBlendStateCreateInfo cb{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
        cb.attachmentCount = 1; cb.pAttachments = &ba;

        VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
        dynState.dynamicStateCount = 2; dynState.pDynamicStates = dyn;

        VkFormat colorFormat = kDistortFormat;
        VkPipelineRenderingCreateInfo prci{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
        prci.colorAttachmentCount = 1; prci.pColorAttachmentFormats = &colorFormat;
        prci.depthAttachmentFormat = Swapchain.m_DepthFormat;

        VkGraphicsPipelineCreateInfo pi{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
        pi.pNext = &prci;
        pi.stageCount = 2;                 pi.pStages             = st;
        pi.pVertexInputState = &vi;        pi.pInputAssemblyState = &ia;
        pi.pViewportState = &vp;           pi.pRasterizationState = &rs;
        pi.pMultisampleState = &ms;        pi.pDepthStencilState  = &ds;
        pi.pColorBlendState = &cb;         pi.pDynamicState       = &dynState;
        pi.layout = s_GlassDistortLayout;

        VkPipeline h = VK_NULL_HANDLE;
        if (vkCreateGraphicsPipelines(VulkanHW.m_Device, PipelineCache::GetCacheObject(), 1, &pi, nullptr, &h) != VK_SUCCESS) {
            Msg("![VK Particles] glass-distort pipeline failed (tcOffset=%u)", tcOffset);
            return VK_NULL_HANDLE;
        }
        return h;
    }

    // Re-draw the glass panes into the (already begun) distortion pass.
    void DrawGlassDistort(VkCommandBuffer cmd, const Fmatrix& viewProj)
    {
        if (ps_r_glass_refr <= 0.001f) return;
        const auto& items = g_RenderQueue.GlassItems();
        if (items.empty()) return;

        VkPipeline lastPipe = VK_NULL_HANDLE;
        VkBuffer lastVB = VK_NULL_HANDLE, lastIB = VK_NULL_HANDLE;
        for (const DrawItem& it : items) {
            auto* rv = it.vis;
            if (!rv || (rv->Type != MT_NORMAL && rv->Type != MT_PROGRESSIVE)) continue;
            auto* fv = static_cast<vkFVisual*>(rv);
            if (!fv->m_mesh.p_rm_Vertices || !fv->m_mesh.p_rm_Indices) continue;
            if (fv->m_mesh.vStride != 32) continue;                     // world layouts only
            if (fv->m_pWorldMaterial && (fv->m_pWorldMaterial->isEmisAdd || fv->m_pWorldMaterial->isLitBlend)) continue;   // glow halos / light beams don't refract
            const u32 slot = (fv->m_mesh.tcOffset == 28) ? 1u : 0u;
            if (s_GlassDistortPipe[slot] == VK_NULL_HANDLE)
                s_GlassDistortPipe[slot] = BuildGlassDistortPipeline(fv->m_mesh.tcOffset);
            VkPipeline pipe = s_GlassDistortPipe[slot];
            if (pipe == VK_NULL_HANDLE) continue;

            if (pipe != lastPipe) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
                lastPipe = pipe; lastVB = lastIB = VK_NULL_HANDLE;
            }
            GlassDistortPush pc{};
            pc.mvp.mul(viewProj, it.xform);
            pc.strength = ps_r_glass_refr;
            vkCmdPushConstants(cmd, s_GlassDistortLayout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               0, sizeof(pc), &pc);

            VkBuffer vb = fv->m_mesh.p_rm_Vertices->GetHandle();
            if (vb != lastVB) { VkDeviceSize off = 0; vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &off); lastVB = vb; }
            VkBuffer ib = fv->m_mesh.p_rm_Indices->GetHandle();
            if (ib != lastIB) { vkCmdBindIndexBuffer(cmd, ib, 0, fv->m_mesh.iType); lastIB = ib; }
            const u32 firstIndex = it.iCountOverride ? it.iBaseOverride  : fv->m_mesh.iBase;
            const u32 indexCount = it.iCountOverride ? it.iCountOverride : fv->m_mesh.iCount;
            vkCmdDrawIndexed(cmd, indexCount, 1, firstIndex, (s32)fv->m_mesh.vBase, 0);
        }
    }

    void DestroyDistortRT()
    {
        if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
        if (s_DistortView) { vkDestroyImageView(VulkanHW.m_Device, s_DistortView, nullptr); s_DistortView = VK_NULL_HANDLE; }
        if (s_DistortImg)  { vmaDestroyImage(VulkanHW.m_Allocator, s_DistortImg, s_DistortAlloc); s_DistortImg = VK_NULL_HANDLE; s_DistortAlloc = VK_NULL_HANDLE; }
        s_DistortExtent = {};
        s_DistortFirst  = true;
    }

    bool EnsureDistortRT(VkExtent2D extent)
    {
        if (s_DistortImg && extent.width == s_DistortExtent.width && extent.height == s_DistortExtent.height)
            return true;
        DestroyDistortRT();
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = kDistortFormat;
        ici.extent = { extent.width, extent.height, 1 };
        ici.mipLevels = 1; ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        if (vmaCreateImage(VulkanHW.m_Allocator, &ici, &aci, &s_DistortImg, &s_DistortAlloc, nullptr) != VK_SUCCESS) {
            Msg("![VK Particles] distort RT create failed");
            return false;
        }
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = s_DistortImg;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = kDistortFormat;
        vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vci.subresourceRange.levelCount = 1;
        vci.subresourceRange.layerCount = 1;
        if (vkCreateImageView(VulkanHW.m_Device, &vci, nullptr, &s_DistortView) != VK_SUCCESS) {
            Msg("![VK Particles] distort view create failed");
            DestroyDistortRT();
            return false;
        }
        s_DistortExtent = extent;
        s_DistortFirst  = true;
        ++s_DistortGen;
        Msg("[VK Particles] distort RT ready (%ux%u, gen %u)", extent.width, extent.height, s_DistortGen);
        return true;
    }

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

    // `distortTarget` builds the PBM_DISTORT variant: renders into the RGBA8
    // distortion buffer with standard alpha blending (sprite rg = UV offset
    // around neutral 0.5, alpha = haze mask) instead of the HDR scene target.
    // `fsOverride` swaps the fragment shader (wallmark.frag for decals).
    VkPipeline BuildPipeline(EParticleBlendMode mode, bool distortTarget = false,
                             VkShaderModule fsOverride = VK_NULL_HANDLE)
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
        stages[1].module = fsOverride ? fsOverride : s_FS; stages[1].pName = "main";

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
        if (distortTarget) { src = VK_BLEND_FACTOR_SRC_ALPHA; dst = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA; }
        else               BlendFactors(mode, src, dst);
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

        VkFormat colorFormat = distortTarget ? kDistortFormat : VK::SceneColor::Format();
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

    // (Re)point set 1 at Vol's scatter volume. Vol::Init runs before ParticlePass_Init
    // and creates the volume eagerly, so the view is valid here; this also rebinds if
    // Vol ever regenerates (Generation() bumps). No-op once bound for the current gen.
    void EnsureVolSet()
    {
        if (!Vol::Ready()) return;
        const u32 gen = Vol::Generation();
        if (s_VolBoundGen == gen) return;
        VkImageView view = Vol::GetScatterView();
        VkSampler   samp = Vol::GetSampler();
        if (!view || !samp || s_VolSet == VK_NULL_HANDLE) return;
        VkDescriptorImageInfo ii{};
        ii.sampler     = samp;
        ii.imageView   = view;
        ii.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet w{};
        w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet          = s_VolSet;
        w.dstBinding      = 0;
        w.descriptorCount = 1;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo      = &ii;
        vkUpdateDescriptorSets(VulkanHW.m_Device, 1, &w, 0, nullptr);
        s_VolBoundGen = gen;
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

    // Set 1: binding 0 = froxel scatter volume (combined image sampler, FS) — the
    // Stage-0 light probe. Always part of the layout; bound for every particle draw.
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
        if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &s_VolSetLayout) != VK_SUCCESS) {
            Msg("![VK Particles] vol set layout create failed");
            return false;
        }
    }

    // Pool — one set per unique sprite texture (~256 generous) + 1 for the vol set.
    {
        constexpr u32 kMaxSets = 256 + 1;   // +1 for the persistent vol (set 1) descriptor
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

    // Pipeline layout: set 0 (sprite) + set 1 (froxel volume) + the ParticlePush block
    // (viewProj VS + froxel basis VS + volParams FS). The rain/wallmark passes share
    // this layout but only push the first 64 bytes (viewProj) and never bind set 1 —
    // valid, since their fragment shaders don't statically use set 1 or the FS bytes.
    {
        VkDescriptorSetLayout sets[2] = { s_SetLayout, s_VolSetLayout };
        VkPushConstantRange pc{};
        pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        pc.offset     = 0;
        pc.size       = sizeof(ParticlePush);
        VkPipelineLayoutCreateInfo plci{};
        plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount         = 2;
        plci.pSetLayouts            = sets;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pc;
        if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_Layout) != VK_SUCCESS) {
            Msg("![VK Particles] CreatePipelineLayout failed");
            return false;
        }
    }

    // Allocate the persistent vol (set 1) descriptor and point it at Vol's scatter
    // volume (Vol::Init already ran — eager). Re-pointed by EnsureVolSet on regen.
    {
        VkDescriptorSetAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = s_Pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &s_VolSetLayout;
        if (vkAllocateDescriptorSets(VulkanHW.m_Device, &ai, &s_VolSet) != VK_SUCCESS) {
            Msg("![VK Particles] vol set alloc failed");
            return false;
        }
        EnsureVolSet();
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
    if (s_DistortPipe)  { vkDestroyPipeline(VulkanHW.m_Device, s_DistortPipe, nullptr); s_DistortPipe = VK_NULL_HANDLE; }
    for (u32 i = 0; i < 2; ++i)
        if (s_GlassDistortPipe[i]) { vkDestroyPipeline(VulkanHW.m_Device, s_GlassDistortPipe[i], nullptr); s_GlassDistortPipe[i] = VK_NULL_HANDLE; }
    if (s_GlassDistortLayout) { vkDestroyPipelineLayout(VulkanHW.m_Device, s_GlassDistortLayout, nullptr); s_GlassDistortLayout = VK_NULL_HANDLE; }
    s_GlassVS = s_GlassFS = VK_NULL_HANDLE;
    if (s_WallmarkPipe) { vkDestroyPipeline(VulkanHW.m_Device, s_WallmarkPipe, nullptr); s_WallmarkPipe = VK_NULL_HANDLE; }
    if (s_BloodPipe)    { vkDestroyPipeline(VulkanHW.m_Device, s_BloodPipe, nullptr); s_BloodPipe = VK_NULL_HANDLE; }
    if (s_RainPipe)     { vkDestroyPipeline(VulkanHW.m_Device, s_RainPipe, nullptr); s_RainPipe = VK_NULL_HANDLE; }
    if (s_RainDropPipe) { vkDestroyPipeline(VulkanHW.m_Device, s_RainDropPipe, nullptr); s_RainDropPipe = VK_NULL_HANDLE; }
    s_WallmarkFS = VK_NULL_HANDLE;
    s_BloodFS    = VK_NULL_HANDLE;
    s_RainFS     = VK_NULL_HANDLE;
    s_RainDropFS = VK_NULL_HANDLE;
    DestroyDistortRT();
    if (s_Layout)    { vkDestroyPipelineLayout(VulkanHW.m_Device, s_Layout, nullptr); s_Layout = VK_NULL_HANDLE; }
    if (s_Pool)      { vkDestroyDescriptorPool(VulkanHW.m_Device, s_Pool, nullptr); s_Pool = VK_NULL_HANDLE; }  // frees s_VolSet too
    s_VolSet = VK_NULL_HANDLE; s_VolBoundGen = kVolGenUnbound;
    if (s_Sampler)   { vkDestroySampler(VulkanHW.m_Device, s_Sampler, nullptr); s_Sampler = VK_NULL_HANDLE; }
    if (s_SetLayout) { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_SetLayout, nullptr); s_SetLayout = VK_NULL_HANDLE; }
    if (s_VolSetLayout) { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_VolSetLayout, nullptr); s_VolSetLayout = VK_NULL_HANDLE; }
    s_VS = s_FS = VK_NULL_HANDLE;
    s_Init = false;
}

namespace ParticlePass {

bool             Ready()      { return s_Init; }
VkPipelineLayout GetLayout()  { return s_Layout; }

VkPipeline GetPipeline(EParticleBlendMode mode)
{
    if (!s_Init || mode < 0 || mode >= PBM_COUNT) return VK_NULL_HANDLE;
    if (mode == PBM_DISTORT) return VK_NULL_HANDLE;       // own pass — see Pass_Particles phase 3
    if (s_Pipelines[mode] == VK_NULL_HANDLE)
        s_Pipelines[mode] = BuildPipeline(mode);
    return s_Pipelines[mode];
}

VkImageView GetDistortView()      { return s_DistortView; }
u32         DistortGeneration()   { return s_DistortGen; }

VkPipeline GetWallmarkPipeline()
{
    if (!s_Init) return VK_NULL_HANDLE;
    if (s_WallmarkPipe == VK_NULL_HANDLE) {
        if (s_WallmarkFS == VK_NULL_HANDLE && g_ShaderManager)
            s_WallmarkFS = g_ShaderManager->Load("wallmark.frag.spv");
        if (s_WallmarkFS == VK_NULL_HANDLE)
            return GetPipeline(PBM_BLEND);      // shader missing — plain blend fallback
        // MODULATE2X like R4's effects_wallmark.s — wm_* textures are authored
        // for multiply blending (neutral-grey backgrounds, junk alpha).
        s_WallmarkPipe = BuildPipeline(PBM_MUL_2X, /*distortTarget*/ false, s_WallmarkFS);
    }
    return s_WallmarkPipe;
}

VkPipeline GetRainPipeline()
{
    if (!s_Init) return VK_NULL_HANDLE;
    if (s_RainPipe == VK_NULL_HANDLE) {
        if (s_RainFS == VK_NULL_HANDLE && g_ShaderManager)
            s_RainFS = g_ShaderManager->Load("rain.frag.spv");
        if (s_RainFS == VK_NULL_HANDLE)
            return GetPipeline(PBM_BLEND);      // shader missing — plain blend fallback
        // Alpha blend; shape from tex.a only — the fx_rain rgb is a refraction
        // normal map, not a colour sprite (see rain.frag.glsl).
        s_RainPipe = BuildPipeline(PBM_BLEND, /*distortTarget*/ false, s_RainFS);
    }
    return s_RainPipe;
}

VkPipeline GetRainDropPipeline()
{
    if (!s_Init) return VK_NULL_HANDLE;
    if (s_RainDropPipe == VK_NULL_HANDLE) {
        if (s_RainDropFS == VK_NULL_HANDLE && g_ShaderManager)
            s_RainDropFS = g_ShaderManager->Load("rain_drop.frag.spv");
        if (s_RainDropFS == VK_NULL_HANDLE)
            return GetRainPipeline();           // shader missing — tex-alpha fallback
        // ALPHA_ADD: the SSFX drop is refracted-background + hemi lift — its
        // net effect over the backdrop is a faint ADDITIVE glint, not a
        // painted-over streak (alpha blend read as "ливень"/tracers).
        s_RainDropPipe = BuildPipeline(PBM_ALPHA_ADD, /*distortTarget*/ false, s_RainDropFS);
    }
    return s_RainDropPipe;
}

VkPipeline GetBloodWallmarkPipeline()
{
    if (!s_Init) return VK_NULL_HANDLE;
    if (s_BloodPipe == VK_NULL_HANDLE) {
        if (s_BloodFS == VK_NULL_HANDLE && g_ShaderManager)
            s_BloodFS = g_ShaderManager->Load("wallmark_blood.frag.spv");
        if (s_BloodFS == VK_NULL_HANDLE)
            return GetWallmarkPipeline();       // shader missing — generic decal fallback
        s_BloodPipe = BuildPipeline(PBM_BLEND, /*distortTarget*/ false, s_BloodFS);
    }
    return s_BloodPipe;
}

VkDescriptorSet GetTextureSet(const char* texture_name)
{
    if (!s_Init || !texture_name || !texture_name[0]) return VK_NULL_HANDLE;

    std::string key(texture_name);
    auto it = s_TexCache.find(key);
    if (it != s_TexCache.end()) return it->second.set;

    // Normalize legacy particle texture names (matches the DX texture loader):
    // defs may carry a comma-separated list (base texture = the FIRST entry)
    // and a .tga/.bmp source extension — the cooked asset on disk is .dds
    // ('pfx\pfx_flame_01.tga' → 'pfx\pfx_flame_01.dds').
    string_path nm;
    xr_strcpy(nm, texture_name);
    if (char* comma = strchr(nm, ','))
        *comma = 0;
    if (char* dot = strrchr(nm, '.')) {
        const char* sep1 = strrchr(nm, '\\');
        const char* sep2 = strrchr(nm, '/');
        const char* sep  = (sep1 > sep2) ? sep1 : sep2;
        if (!sep || dot > sep)   // only strip a real extension, not a dotted folder
            *dot = 0;
    }

    // Resolve "$game_textures$\<name>.dds".
    string_path leaf, full;
    xr_sprintf(leaf, "%s.dds", nm);
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

VkImageView GetTextureView(const char* texture_name)
{
    if (!s_Init || !texture_name || !texture_name[0]) return VK_NULL_HANDLE;
    GetTextureSet(texture_name);                      // ensure loaded + cached
    auto it = s_TexCache.find(std::string(texture_name));
    if (it != s_TexCache.end() && it->second.tex)
        return it->second.tex->GetView();
    return VK_NULL_HANDLE;
}

}  // namespace ParticlePass

void CollectSmokeParticles(xr_vector<VK::Vol::SmokeParticle>& out)
{
    // World-phase alpha smoke (PBM_BLEND) only — additive fire/sparks/muzzle are
    // self-emissive (not media), HUD smoke uses a different projection. Same flatten
    // as Pass_Particles. Each live PAPI particle → a media splat sample.
    //
    // Two-zone LOD: the exp-Z froxel grid is fine near the camera but coarse far away
    // (distant smoke = blobby muddy clouds + wasted splat). FULL volumetric quality only
    // inside dist_full (~12 m); from there density fades linearly to 0 by smoke_dist
    // (capped at the terrain detail bubble r__detail_radius); beyond that the billboard
    // alone represents it (still lit by the Stage-0 probe). So the pretty volumetric smoke
    // is the near body, the muddy far blobs never form, and the expensive full-footprint
    // splats concentrate near the camera (far footprints shrink with cell size anyway).
    const Fvector& eye    = Device.vCameraPosition;
    const Fvector& camDir = Device.vCameraDirection;   // forward — drop the behind-camera hemisphere
    const float maxD      = _min(ps_r_vol_smoke_dist, float(_max(ps_r__detail_radius, 1)));
    const float maxDSqr   = maxD * maxD;
    const float distFull  = _min(ps_r_vol_smoke_dist_full, maxD);   // full-quality inner radius
    const float fadeRange = _max(maxD - distFull, 0.001f);

    static xr_vector<vkCParticleEffect*> collect;
    for (const DynVisual& d : g_DynamicVisuals) {
        if (!d.vis) continue;
        const u32 t = d.vis->Type;
        if (t != MT_PARTICLE_EFFECT && t != MT_PARTICLE_GROUP) continue;
        collect.clear();
        static_cast<vkParticleVisual*>(d.vis)->CollectEffects(collect);
        for (vkCParticleEffect* e : collect) {
            if (!e || e->GetBlendMode() != PBM_BLEND || e->GetHudMode()) continue;

            PAPI::Particle* particles = nullptr;
            u32 p_cnt = 0;
            PAPI::ParticleManager()->GetParticles(e->GetHandleEffect(), particles, p_cnt);
            if (!particles || !p_cnt) continue;

            const bool xform = e->m_RT_Flags.is(vkCParticleEffect::flRT_XFORM);
            for (u32 i = 0; i < p_cnt; ++i) {
                const PAPI::Particle& m = particles[i];
                if (m.colorA <= 0.003f) continue;            // invisible → contributes no media

                Fvector wp;
                if (xform) e->m_XFORM.transform_tiny(wp, m.pos);
                else       wp.set(m.pos.x, m.pos.y, m.pos.z);

                // Behind the camera → can never land in a view-frustum froxel (the splat
                // would reject it anyway); cull here so it's not uploaded/splatted at all.
                Fvector rel; rel.sub(wp, eye);
                if (rel.dotproduct(camDir) <= 0.0f) continue;
                const float d2 = eye.distance_to_sqr(wp);
                if (d2 > maxDSqr) continue;                  // outside the detail bubble → billboard only
                float dens = m.colorA;
                const float dd = _sqrt(d2);
                if (dd > distFull) dens *= _max(1.0f - (dd - distFull) / fadeRange, 0.0f);   // LOD fade → billboard
                if (dens <= 0.003f) continue;

                VK::Vol::SmokeParticle sp;
                sp.pos[0] = wp.x; sp.pos[1] = wp.y; sp.pos[2] = wp.z;
                sp.radius = (m.size.x + m.size.y) * 0.25f;   // avg billboard half-extent
                sp.color[0] = m.colorR; sp.color[1] = m.colorG; sp.color[2] = m.colorB; sp.color[3] = dens;
                out.push_back(sp);
            }
        }
    }
}

void Pass_Particles(FrameContext& ctx)
{
    if (!s_Init)                   return;
    if (ctx.cmd == VK_NULL_HANDLE) return;
    // Glass refraction needs the distort phase even with no particle around —
    // and not only for world-path glass (kinematics cabinet panes come via the
    // skinned uploads, invisible to the queue), so gate on the cvar alone.
    const bool glassRefr = ps_r_glass_refr > 0.001f;
    if (g_DynamicVisuals.empty() && g_HudVisuals.empty() && s_DistortImg == VK_NULL_HANDLE && !glassRefr) return;

    // Flatten visible particle visuals into leaf effects, split by phase:
    //   1. world  — scene projection, the bulk (fire/smoke/anomalies);
    //   2. HUD    — muzzle flashes etc. (GetHudMode): WORLD-space positions but
    //      the HUD-FOV projection (Device.mFullTransform_hud2, the R4
    //      CHUDTransformHelper path) + near depth range [0, 0.02] (R4 rmNear)
    //      so they sit on the weapon and never clip into walls;
    //   3. distort — PBM_DISTORT heat haze, rendered into the distortion
    //      buffer the tonemap composite uses to offset scene UVs.
    static xr_vector<vkCParticleEffect*> s_world, s_hud, s_distort, s_collect;
    s_world.clear(); s_hud.clear(); s_distort.clear();
    auto collectList = [&](const xr_vector<DynVisual>& list) {
        for (const DynVisual& d : list) {
            if (!d.vis) continue;
            const u32 t = d.vis->Type;
            if (t != MT_PARTICLE_EFFECT && t != MT_PARTICLE_GROUP) continue;
            s_collect.clear();
            static_cast<vkParticleVisual*>(d.vis)->CollectEffects(s_collect);
            for (vkCParticleEffect* e : s_collect) {
                if (!e) continue;
                if (e->GetBlendMode() == PBM_DISTORT) s_distort.push_back(e);
                else if (e->GetHudMode())             s_hud.push_back(e);
                else                                  s_world.push_back(e);
            }
        }
    };
    collectList(g_DynamicVisuals);
    collectList(g_HudVisuals);
    // Glass refraction also drives the distort phase (panes re-drawn as wobble).
    // Once the distort RT exists it must be refreshed (cleared) EVERY frame —
    // otherwise the tonemap would re-apply last frame's frozen haze.
    const bool runDistort = !s_distort.empty() || s_DistortImg != VK_NULL_HANDLE || glassRefr;
    if (s_world.empty() && s_hud.empty() && !runDistort) return;

    VkCommandBuffer cmd = ctx.cmd;
    const u32 slot = CommandManager.GetCurrentFrame();
    CVulkanBuffer& ring = s_Ring[slot];
    if (!ring.IsValid()) return;

    u8* base = (u8*)ring.Map();
    if (!base) return;

    VkPipeline      lastPipe = VK_NULL_HANDLE;
    VkDescriptorSet lastSet  = VK_NULL_HANDLE;
    u32 vtxUsed = 0;
    u32 nDraw = 0, nHud = 0, nDistort = 0;

    // ---- Stage-0 volumetric smoke lighting setup -----------------------------
    // Build the froxel camera basis the VS uses to map each vertex to its froxel.
    // The probe (drawList's per-effect volParams.x) is enabled only for the WORLD
    // phase, only when r_vol actually ran this frame (scatter is valid + SHADER_READ),
    // and only on alpha-blended smoke — see drawList. The basis is built from the
    // scene camera even for the HUD phase (which pushes strength 0, so it's unused).
    EnsureVolSet();
    const float smokeStrength = (Vol::Ready() && Vol::Wanted()) ? ps_r_vol_smoke : 0.0f;

    ParticlePush push{};
    {
        const Vol::GridZParams gz = Vol::GetGridZ();
        const Fvector& eye = Device.vCameraPosition;
        Fvector dir; dir.set(0.f, 0.f, 1.f);
        if (ctx.viewProj) dir = DeriveProjTerms(*ctx.viewProj).dir;
        push.camPosNear[0]  = eye.x; push.camPosNear[1]  = eye.y; push.camPosNear[2]  = eye.z; push.camPosNear[3]  = gz.nearZ;
        push.camDirLogFN[0] = dir.x; push.camDirLogFN[1] = dir.y; push.camDirLogFN[2] = dir.z; push.camDirLogFN[3] = gz.logFarNear;
        push.volParams[1]   = ctx.extent.width  ? 1.0f / float(ctx.extent.width)  : 0.0f;
        push.volParams[2]   = ctx.extent.height ? 1.0f / float(ctx.extent.height) : 0.0f;
        push.volParams[3]   = ps_r_vol_smoke_clamp;
    }

    // Shared draw loop: builds billboards into the ring at the running offset and
    // issues one draw per effect. `vpMat` is the phase's view-projection; `probe`
    // is the Stage-0 light-probe strength, applied ONLY to alpha-blended smoke
    // (PBM_BLEND) — additive fire/sparks/muzzle are self-emissive and would
    // over-glow, so everything else pushes strength 0. The push carries viewProj,
    // so it's re-issued whenever the strength changes (and always on the first
    // drawable effect). `forcedPipe` overrides the per-blend pipeline (the distort
    // phase renders every effect with one pipeline).
    auto drawList = [&](xr_vector<vkCParticleEffect*>& list, VkPipeline forcedPipe,
                        u32& counter, const Fmatrix& vpMat, float probe) {
        push.viewProj = vpMat;
        float lastStrength = -1.0f;   // != any real strength → push on first drawable effect
        for (vkCParticleEffect* e : list) {
            VkPipeline pipe = forcedPipe ? forcedPipe : ParticlePass::GetPipeline(e->GetBlendMode());
            if (pipe == VK_NULL_HANDLE) continue;          // build failure

            VkDescriptorSet set = e->ResolveTextureSet();
            if (set == VK_NULL_HANDLE) continue;           // missing texture

            if (vtxUsed >= kRingVerts) break;              // ring full
            const u32 avail = kRingVerts - vtxUsed;
            FVF::LIT* dst = (FVF::LIT*)(base + (size_t)vtxUsed * kVtxStride);
            const u32 vcount = e->BuildVertices(dst, avail);
            if (vcount == 0) continue;

            const float strength = (e->GetBlendMode() == PBM_BLEND) ? probe : 0.0f;
            if (strength != lastStrength) {
                push.volParams[0] = strength;
                vkCmdPushConstants(cmd, s_Layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   0, sizeof(ParticlePush), &push);
                lastStrength = strength;
            }

            if (pipe != lastPipe) { vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe); lastPipe = pipe; }
            if (set  != lastSet)  { vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_Layout, 0, 1, &set, 0, nullptr); lastSet = set; }

            vkCmdDraw(cmd, vcount, 1, vtxUsed, 0);
            vtxUsed += vcount;
            ++counter;
        }
    };

    VkBuffer vbuf = ring.GetHandle();
    VkDeviceSize voff = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vbuf, &voff);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_Layout, 1, 1, &s_VolSet, 0, nullptr);

    VkRect2D sc{ {}, ctx.extent };

    // ---- Phases 1+2: scene colour + world depth, no depth write -------------
    if (!s_world.empty() || !s_hud.empty()) {
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
        vkCmdSetScissor(cmd, 0, 1, &sc);

        if (!s_world.empty() && ctx.viewProj) {
            drawList(s_world, VK_NULL_HANDLE, nDraw, *ctx.viewProj, smokeStrength);   // alpha smoke catches the froxel light
        }

        if (!s_hud.empty()) {
            // R4 HUD-mode particles: positions stay world-space, but the
            // projection switches to the HUD FOV (mFullTransform_hud2 = real
            // view × HUD-FOV proj) and the depth range compresses to
            // [0, 0.02] (rmNear) — in front of the world, depth-tested only
            // against the HUD weapon itself.
            vp.minDepth = 0.0f; vp.maxDepth = 0.02f;
            vkCmdSetViewport(cmd, 0, 1, &vp);
            // HUD smoke: no froxel probe (different projection/depth) → strength 0.
            drawList(s_hud, VK_NULL_HANDLE, nHud, Device.mFullTransform_hud2, 0.0f);
        }

        vkCmdEndRendering(cmd);
    }

    // ---- Phase 3: distortion buffer (heat haze) ------------------------------
    if (runDistort && EnsureDistortRT(ctx.extent)) {
        if (s_DistortPipe == VK_NULL_HANDLE)
            s_DistortPipe = BuildPipeline(PBM_BLEND, /*distortTarget*/ true);

        // UNDEFINED on the first use; SHADER_READ (tonemap sampled it) after.
        ImageBarrier(cmd, s_DistortImg,
                     s_DistortFirst ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        s_DistortFirst = false;

        VkRenderingAttachmentInfo cAtt{};
        cAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        cAtt.imageView   = s_DistortView;
        cAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        cAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;       // neutral = no offset
        cAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
        cAtt.clearValue.color = { { 0.5f, 0.5f, 0.5f, 0.0f } };

        // World depth (LOAD, no write): haze behind walls must not bleed through.
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
        vkCmdSetScissor(cmd, 0, 1, &sc);

        if (!s_distort.empty() && ctx.viewProj) {
            // distort writes UV offsets, not colour — no probe → strength 0.
            drawList(s_distort, s_DistortPipe, nDistort, *ctx.viewProj, 0.0f);
        }

        // Glass refraction: re-draw the late-glass panes as procedural wobble
        // (r_glass_refr) — the tonemap bends the scene behind them. Both halves:
        // world-path panes (windows/doors/vehicles) + kinematics (cabinet doors).
        if (ps_r_glass_refr > 0.001f && ctx.viewProj) {
            DrawGlassDistort(cmd, *ctx.viewProj);
            Skinned_RenderGlassDistort(cmd, *ctx.viewProj, ps_r_glass_refr);
        }

        vkCmdEndRendering(cmd);
        ImageBarrier(cmd, s_DistortImg,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    ring.Flush();
    ring.Unmap();

    static bool s_diag = false;
    if (!s_diag && (nDraw + nHud + nDistort)) {
        Msg("[VK Particles] first frame drawn: world=%u hud=%u distort=%u verts=%u", nDraw, nHud, nDistort, vtxUsed);
        s_diag = true;
    }
    static bool s_diagHud = false;
    if (!s_diagHud && nHud) { Msg("[VK Particles] first HUD effect drawn (hud draws=%u)", nHud); s_diagHud = true; }
    static bool s_diagDist = false;
    if (!s_diagDist && nDistort) { Msg("[VK Particles] first DISTORT effect drawn (draws=%u)", nDistort); s_diagDist = true; }
}

}  // namespace VK
