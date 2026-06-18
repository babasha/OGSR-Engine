// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan — tonemap / exposure composite pass. See vk_pass_tonemap.h.
#include "stdafx.h"
#include "vk_pass_tonemap.h"
#include "vk_scene_color.h"
#include "vk_pass_bloom.h"         // BloomPass — bright-pass + blur before the composite
#include "vk_pass_ssao.h"          // SSAOPass — GTAO+IL: IL is MRT target 1 (binding 5) + prev-colour capture
#include "vk_pass_particles.h"     // ParticlePass::GetDistortView — heat-haze offsets (binding 2)
#include "vk_swapchain.h"          // Swapchain.m_Images / m_ImageViews / m_Format (+ depth for SSR puddles)
#include "vk_shaders.h"            // g_ShaderManager
#include "vk_pipeline_cache.h"     // PipelineCache::GetCacheObject
#include "vk_barriers.h"           // ImageBarrier
#include "vk_env_light.h"          // EnvLight set (set 1) — rain map/VP + camera terms for SSR puddles
#include "vk_volumetrics.h"        // VK::Vol — integrated froxel volume (binding 4) + exp-Z params
#include "vk_exposure.h"           // VK::Exposure — shared auto-exposure constants (also used by bloom)
#include "vk_fullscreen.h"         // VK::Fullscreen — shared fullscreen pipeline
#include "HW_Vulkan.h"
#include "../xrRender/xrRender_console.h"  // ps_r2_img_* — R4 color-grading console knobs

// r_vol composite knobs (global scope — C-linkage console symbols).
extern int ps_r_vol;
extern int ps_r_vol_debug;
// SSIL composite knobs (global scope — block-scope extern inside the namespace
// would mangle as VK::* → LNK2001).
extern int   ps_r_ssil_enable;
extern int   ps_r_ssil_debug;
extern float ps_r_ssil_strength;

namespace VK {

namespace {
    constexpr u32 kMaxImages = 8;

    VkPipeline            s_Pipeline       = VK_NULL_HANDLE;
    VkPipelineLayout      s_PipelineLayout = VK_NULL_HANDLE;
    VkShaderModule        s_VS             = VK_NULL_HANDLE;
    VkShaderModule        s_FS             = VK_NULL_HANDLE;
    VkDescriptorSetLayout s_SetLayout      = VK_NULL_HANDLE;
    VkDescriptorPool      s_Pool           = VK_NULL_HANDLE;
    VkDescriptorSet       s_Set[kMaxImages] = {};
    VkSampler             s_Sampler        = VK_NULL_HANDLE;
    u32                   s_boundGen       = 0;   // SceneColor generation the sets were written for
    u32                   s_boundBloomGen  = 0;   // BloomPass RT generation (binding 1)
    u32                   s_boundDistortGen = 0;  // distort RT generation (binding 2)
    VkImageView           s_boundDepth     = VK_NULL_HANDLE;  // scene depth (binding 3, SSR puddles)
    u32                   s_boundVolGen    = 0;   // Vol volume generation (binding 4, volumetrics)
    u32                   s_boundSsilGen   = 0;   // SSIL result generation (binding 5, indirect light)

    // Heat-haze strength: scene UV offset = (distort.rg - 0.5) * kDistortAmount —
    // R2/R4 combine_2.ps: (distort.xy - 127/255) * def_distort, def_distort 0.05.
    constexpr float kDistortAmount = 0.05f;

    // R4 auto-exposure (bloom_luminance_3.ps): exposure = middlegray/(avgLum +
    // low), measured from the whole-frame average luminance (the HDR target's
    // top mip), clamped. R4 defaults are middlegray 1.0 / low 0.0001 with a
    // def_hdr-scaled luminance; our HDR luminance is unscaled, so middlegray is
    // retuned to land midday near exposure ~1 and the clamp bounds how far a
    // dim/bright scene can drift (keeps night from washing to mid-gray).
    // White point 11.2 = R4 tonemap_srgb.h fWhiteIntensity: a soft filmic
    // highlight rolloff (the glow comes from bloom, not from clipping).
    // Screenshot comparison vs R4 (morning, bright sky + tree shade): the
    // sky-heavy frame average pushed our exposure into the 0.6 floor and the
    // whole foreground went darker/flatter than R4. Raise the target + floor.
    // Auto-exposure inputs live in vk_exposure.h (shared with vk_pass_bloom.cpp).
    using Exposure::kMiddleGray;   // exposure target (our HDR scale)
    using Exposure::kLowLum;       // R4 ps_r2_tonemap_low_lum
    using Exposure::kExpMin;       // exposure clamp lo
    using Exposure::kExpMax;       // exposure clamp hi
    constexpr float kWhitePoint  = 11.2f;    // R4 tonemap_sRGB fWhiteIntensity
    constexpr float kExpComp     = 1.0f;     // overall compensation knob
    constexpr float kBloomIntensity = 0.8f;  // bloom add strength (blend_soft analog)

    struct TonemapPush { float p0[4]; float p1[4]; float p2[4]; float p3[4]; float p4[4]; float p5[4]; };
}

namespace TonemapPass {

bool Init()
{
    if (s_Pipeline) return true;

    if (!g_ShaderManager) g_ShaderManager = xr_new<VK::CVulkanSPIRVLoader>();
    s_VS = g_ShaderManager->Load("tonemap.vert.spv");
    s_FS = g_ShaderManager->Load("tonemap.frag.spv");
    if (s_VS == VK_NULL_HANDLE || s_FS == VK_NULL_HANDLE) {
        Msg("![VK Tonemap] shader load failed");
        return false;
    }

    // Set 0: binding 0 = HDR scene, 1 = blurred bloom, 2 = distortion,
    // 3 = scene depth (SSR puddles), 4 = integrated volumetrics (sampler3D),
    // 5 = SSIL indirect light (half-res). All FS.
    VkDescriptorSetLayoutBinding b[6]{};
    for (u32 i = 0; i < 6; ++i) {
        b[i].binding = i; b[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lci.bindingCount = 6; lci.pBindings = b;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &s_SetLayout) != VK_SUCCESS) {
        Msg("![VK Tonemap] set layout failed"); return false;
    }

    VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxImages * 6 };
    VkDescriptorPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = kMaxImages; pci.poolSizeCount = 1; pci.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &s_Pool) != VK_SUCCESS) {
        Msg("![VK Tonemap] pool failed"); return false;
    }
    VkDescriptorSetLayout layouts[kMaxImages];
    for (u32 i = 0; i < kMaxImages; ++i) layouts[i] = s_SetLayout;
    VkDescriptorSetAllocateInfo dai{};
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = s_Pool; dai.descriptorSetCount = kMaxImages; dai.pSetLayouts = layouts;
    if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, s_Set) != VK_SUCCESS) {
        Msg("![VK Tonemap] alloc sets failed"); return false;
    }

    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_LINEAR; si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.minLod = 0.0f; si.maxLod = VK_LOD_CLAMP_NONE;   // textureLod(maxMip) for the avg-luminance read
    if (vkCreateSampler(VulkanHW.m_Device, &si, nullptr, &s_Sampler) != VK_SUCCESS) {
        Msg("![VK Tonemap] sampler failed"); return false;
    }

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT; pcr.offset = 0; pcr.size = sizeof(TonemapPush);
    // Set 1 = the shared EnvLight set: the SSR puddles read rain_vp/rain_params,
    // the camera frustum terms and the rain occlusion map (binding 9) from it.
    VkDescriptorSetLayout sets[2] = { s_SetLayout, EnvLight::GetSetLayout() };
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = (sets[1] != VK_NULL_HANDLE) ? 2u : 1u; plci.pSetLayouts = sets;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_PipelineLayout) != VK_SUCCESS) {
        Msg("![VK Tonemap] pipeline layout failed"); return false;
    }

    // Writes the actual swapchain (UNORM), RGBA, no blend (final composite).
    s_Pipeline = Fullscreen::CreatePipeline(s_VS, s_FS, Swapchain.m_Format, s_PipelineLayout,
                                            Fullscreen::OpaqueAttachment(), "Tonemap");
    if (s_Pipeline == VK_NULL_HANDLE) {
        Msg("![VK Tonemap] pipeline create failed"); return false;
    }

    Msg("[VK Tonemap] Init OK (auto-exposure mg=%.2f white=%.2f clamp[%.2f,%.2f])",
        kMiddleGray, kWhitePoint, kExpMin, kExpMax);
    return true;
}

void Destroy()
{
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
    if (s_Pipeline)       { vkDestroyPipeline(VulkanHW.m_Device, s_Pipeline, nullptr); s_Pipeline = VK_NULL_HANDLE; }
    if (s_PipelineLayout) { vkDestroyPipelineLayout(VulkanHW.m_Device, s_PipelineLayout, nullptr); s_PipelineLayout = VK_NULL_HANDLE; }
    if (s_Sampler)        { vkDestroySampler(VulkanHW.m_Device, s_Sampler, nullptr); s_Sampler = VK_NULL_HANDLE; }
    if (s_Pool)           { vkDestroyDescriptorPool(VulkanHW.m_Device, s_Pool, nullptr); s_Pool = VK_NULL_HANDLE; }
    if (s_SetLayout)      { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_SetLayout, nullptr); s_SetLayout = VK_NULL_HANDLE; }
    s_VS = s_FS = VK_NULL_HANDLE; s_boundGen = 0; s_boundBloomGen = 0;
}

}  // namespace TonemapPass

void Pass_TonemapComposite(FrameContext& ctx)
{
    if (s_Pipeline == VK_NULL_HANDLE || ctx.cmd == VK_NULL_HANDLE) return;
    const u32 idx = ctx.imageIndex;
    if (SceneColor::GetImage(idx) == VK_NULL_HANDLE) return;

    VkCommandBuffer cmd = ctx.cmd;

    // Downsample the HDR scene to its full mip chain (top mip = whole-frame
    // average luminance for auto-exposure). This also leaves ALL mips — incl.
    // mip0 (the scene) — in SHADER_READ for the composite below.
    SceneColor::GenerateMips(cmd, idx);

    // SSIL history: snapshot this frame's composited HDR scene (half-res) so the
    // NEXT frame's GTAO horizon march can gather one-bounce indirect light from
    // it. The IL itself is produced inside the GTAO pass (MRT target 1, folded
    // into the occlusion march) — here we only feed it. No-op when r_ssil is off.
    SSAOPass::CapturePrevColor(cmd, SceneColor::GetImage(idx), ctx.extent);

    // Bloom chain (bright pass from the scene mips + gaussian blur) — result
    // lands SHADER_READ for binding 1.
    BloomPass::Execute(cmd, idx, ctx.extent, SceneColor::Generation());

    // Scene depth → SHADER_READ for the SSR puddle march (binding 3). NOT
    // transitioned back: CRender::Begin re-acquires depth from UNDEFINED each
    // frame (its contents are clear-loaded, never carried over).
    ImageBarrier(cmd, Swapchain.m_DepthImage, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);

    // Rebind the per-image descriptor sets when the HDR target, the bloom RT, the
    // distortion RT, the volume or the SSIL (GTAO MRT) result was (re)created.
    const u32 gen        = SceneColor::Generation();
    const u32 bloomGen   = BloomPass::Generation();
    const u32 distortGen = ParticlePass::DistortGeneration();
    const u32 volGen     = Vol::Generation();
    const u32 ssilGen    = SSAOPass::Generation();   // IL RT rides the GTAO RT generation
    if (gen != s_boundGen || bloomGen != s_boundBloomGen || distortGen != s_boundDistortGen
        || Swapchain.m_DepthView != s_boundDepth || volGen != s_boundVolGen || ssilGen != s_boundSsilGen) {
        const u32 n = SceneColor::Count();
        for (u32 i = 0; i < n && i < kMaxImages; ++i) {
            VkDescriptorImageInfo ii[6]{};
            ii[0].sampler = s_Sampler; ii[0].imageView = SceneColor::GetSampleView(i);  // full mip chain
            ii[1].sampler = s_Sampler; ii[1].imageView = BloomPass::GetResultView();
            ii[2].sampler = s_Sampler; ii[2].imageView = ParticlePass::GetDistortView();
            ii[3].sampler = s_Sampler; ii[3].imageView = Swapchain.m_DepthView;          // SSR puddles
            ii[4].sampler = Vol::GetSampler(); ii[4].imageView = Vol::GetIntegratedView(); // integrated volumetrics (3D)
            ii[5].sampler = SSAOPass::GetSampler(); ii[5].imageView = SSAOPass::GetILResultView(); // indirect light (GTAO MRT target 1)
            ii[0].imageLayout = ii[1].imageLayout = ii[2].imageLayout = ii[3].imageLayout = ii[4].imageLayout = ii[5].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            if (ii[1].imageView == VK_NULL_HANDLE) ii[1].imageView = SceneColor::GetSampleView(i); // pre-bloom fallback
            if (ii[2].imageView == VK_NULL_HANDLE) ii[2].imageView = SceneColor::GetSampleView(i); // no-distort fallback (never sampled: scale 0)
            if (ii[3].imageView == VK_NULL_HANDLE) ii[3].imageView = SceneColor::GetSampleView(i); // no-depth fallback (never sampled: wetness 0)
            if (ii[5].imageView == VK_NULL_HANDLE) { ii[5].sampler = s_Sampler; ii[5].imageView = SceneColor::GetSampleView(i); } // SSIL off/not-ready (never sampled: p5.z 0)
            // Write each valid binding by explicit index — binding 4 (3D volume)
            // exists from Vol::Init (eager); the others can transiently be null.
            VkWriteDescriptorSet w[6]{};
            u32 wc = 0;
            for (u32 k = 0; k < 6; ++k) {
                if (ii[k].imageView == VK_NULL_HANDLE) continue;
                w[wc].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[wc].dstSet = s_Set[i]; w[wc].dstBinding = k; w[wc].descriptorCount = 1;
                w[wc].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[wc].pImageInfo = &ii[k];
                ++wc;
            }
            vkUpdateDescriptorSets(VulkanHW.m_Device, wc, w, 0, nullptr);
        }
        s_boundGen        = gen;
        s_boundBloomGen   = bloomGen;
        s_boundDistortGen = distortGen;
        s_boundDepth      = Swapchain.m_DepthView;
        s_boundVolGen     = volGen;
        s_boundSsilGen    = ssilGen;
    }

    // (Depth was already transitioned to SHADER_READ above, before the SSIL gather.)

    // Swapchain image: UNDEFINED (untouched this frame) → COLOR_ATTACHMENT for the composite.
    ImageBarrier(cmd, Swapchain.m_Images[idx],
                 VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    VkRenderingAttachmentInfo cAtt{};
    cAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    cAtt.imageView   = Swapchain.m_ImageViews[idx];
    cAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    cAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_DONT_CARE;   // fullscreen overwrite
    cAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo ri{};
    ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea.extent    = ctx.extent;
    ri.layerCount           = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments    = &cAtt;
    vkCmdBeginRendering(cmd, &ri);

    VkViewport vp{ 0.f, 0.f, (float)ctx.extent.width, (float)ctx.extent.height, 0.f, 1.f };
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{ {0,0}, ctx.extent };
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_Pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_PipelineLayout, 0, 1, &s_Set[idx], 0, nullptr);
    // Set 1: the shared EnvLight set (rain map + camera terms for SSR puddles).
    if (VkDescriptorSet env = EnvLight::GetCurrentSet())
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_PipelineLayout, 1, 1, &env, 0, nullptr);

    TonemapPush push{};
    push.p0[0] = kWhitePoint;
    push.p0[1] = float(SceneColor::MipLevels() - 1);   // top-mip LOD = whole-frame average
    push.p0[2] = kMiddleGray;
    push.p0[3] = kLowLum;
    push.p1[0] = kExpMin; push.p1[1] = kExpMax; push.p1[2] = kExpComp;
    push.p1[3] = kBloomIntensity;
    // R4 CDL grading (ACES_settings.h): Slope = r2_img_exposure, Power =
    // 2*(1 - r2_img_cg), Saturation = r2_img_saturation, gamma from
    // img_corrections — all live console knobs (ssfx_exposure/_gamma/
    // _saturation/ssfx_color_grading), neutral by default.
    push.p2[0] = ps_r2_img_exposure;
    push.p2[1] = ps_r2_img_saturation;
    push.p2[2] = 1.0f / (ps_r2_img_gamma > 0.05f ? ps_r2_img_gamma : 1.0f);
    // Heat-haze: scale 0 until the distort RT exists (binding 2 then holds the
    // scene-view fallback and the shader skips the sample entirely).
    push.p2[3] = (ParticlePass::GetDistortView() != VK_NULL_HANDLE) ? kDistortAmount : 0.0f;
    push.p3[0] = 2.0f * (1.0f - ps_r2_img_cg.x);
    push.p3[1] = 2.0f * (1.0f - ps_r2_img_cg.y);
    push.p3[2] = 2.0f * (1.0f - ps_r2_img_cg.z);
    // Volumetric composite (r_vol): mode + the exp-Z grid params from the LAST
    // Vol::Execute (the EXACT inverse vol_inject used). 0 off / 1 composite / 2 debug.
    const Vol::GridZParams gz = Vol::GetGridZ();
    const bool volOn = (ps_r_vol != 0 || ps_r_vol_debug != 0) && Vol::Ready();
    push.p4[0] = volOn ? (ps_r_vol_debug ? 2.0f : 1.0f) : 0.0f;
    push.p4[1] = gz.nearZ;
    push.p4[2] = gz.farZ;
    push.p4[3] = gz.logFarNear;
    // SSIL (r_ssil): add the half-res indirect-light buffer (binding 5) as
    // coloured bounce. Enabled only once the gather produced a result (its view
    // is non-null); otherwise binding 5 holds the scene-view fallback and the
    // shader skips it. x = strength, y = debug (show only the bounce), z = enable.
    // SSIL is applied in the forward shaders now; the tonemap only shows the debug
    // view (r_ssil_debug) of the raw IL buffer (binding 5). p5.y = that flag.
    const bool ilReady = ps_r_ssil_enable != 0 && SSAOPass::GetILResultView() != VK_NULL_HANDLE;
    push.p5[0] = 0.0f;
    push.p5[1] = (ilReady && ps_r_ssil_debug) ? 1.0f : 0.0f;
    push.p5[2] = 0.0f;
    vkCmdPushConstants(cmd, s_PipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);

    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRendering(cmd);
    // Swapchain stays COLOR_ATTACHMENT for the UI pass; End brings it to PRESENT.
}

}  // namespace VK
