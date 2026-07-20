// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan — tonemap / exposure composite pass. See vk_pass_tonemap.h.
#include "stdafx.h"
#include "vk_pass_tonemap.h"

// Editor viewport state (CRender_Vulkan.cpp) — 0 objects means an EMPTY viewport, the only
// case where the metered auto-exposure has nothing to average. Declared at file scope on
// purpose: inside `namespace VK` below it would bind to a non-existent VK::VKEditor.
namespace VKEditor { int HostModelCount(); }
#include "vk_scene_color.h"
#include "vk_pass_bloom.h"         // BloomPass — bright-pass + blur before the composite
#include "vk_pass_ssao.h"          // SSAOPass — GTAO+IL: IL is MRT target 1 (binding 5) + prev-colour capture
#include "vk_pass_particles.h"     // ParticlePass::GetDistortView — heat-haze offsets (binding 2)
#include "vk_swapchain.h"          // Swapchain.m_Images / m_ImageViews / m_Format (+ depth for SSR puddles)
#include "vk_shaders.h"            // g_ShaderManager
#include "vk_pipeline_cache.h"     // PipelineCache::GetCacheObject
#include "vk_barriers.h"           // ImageBarrier
#include "vk_profiler.h"           // VK::Prof::NameSet — TEMP VUID-hunt instrumentation
#include "vk_env_light.h"          // EnvLight set (set 1) — rain map/VP + camera terms for SSR puddles
#include "vk_vsm.h"                // VSM::MaskReady — dyn-shadow red debug overlay (set 1 binding 14)
#include "vk_volumetrics.h"        // VK::Vol — integrated froxel volume (binding 4) + exp-Z params
#include "vk_dlss.h"               // VK::Dlss — resolved base colour (binding 6) when upscaling
#include "vk_motionvec.h"          // VK::MotionVec::Enabled — DLSS gate
#include "vk_exposure.h"           // VK::Exposure — shared auto-exposure constants (also used by bloom)
#include "vk_fullscreen.h"         // VK::Fullscreen — shared fullscreen pipeline
#include "vk_buffer.h"             // CVulkanBuffer — 1×1 avg-luminance readback (temporal exposure)
#include "HW_Vulkan.h"
#include "../xrRender/xrRender_console.h"  // ps_r2_img_* — R4 color-grading console knobs
#include "../../xr_3da/device.h"   // Device.fTimeDelta — exposure adaptation rate
#include <cmath>                    // expf
#include <cstring>                  // memcpy (half→float decode)
#include <algorithm>               // std::min/max

// r_vol composite knobs (global scope — C-linkage console symbols).
extern int ps_r_vol;
extern int ps_r_vol_debug;
extern int ps_r_linear_color;   // linear colour pipeline — decides whether we owe the display an OETF
// SSIL composite knobs (global scope — block-scope extern inside the namespace
// would mangle as VK::* → LNK2001).
extern int   ps_r_ssil_enable;
extern int   ps_r_ssil_debug;
extern float ps_r_ssil_strength;
extern float ps_r_dither;   // final 8-bit output dither amplitude (LSBs; 0 = off)
extern int   ps_r_vsm_debug_dyn;   // red overlay of VSM dyn-atlas (NPC/grass) shadows
extern float ps_r_exp_adapt;       // auto-exposure temporal adaptation time constant (s; 0 = instant)
extern float ps_r_dlss_sharp;      // CAS sharpen on the DLSS output (0 = off)
extern int   ps_r_dlss_debug;      // DLSS debug view: 1=CAS heatmap, 2=split, 3=gate flag
extern float ps_r_sun_beam_splash;     // god-ray ground-splash strength (0 = off)
extern float ps_r_sun_beam_splash_thr; // in-scatter luminance threshold before the splash kicks in

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
    bool                  s_boundResolvedUp = false;          // was binding 6 the DLSS output (vs render-res scene)?
    VkImageView           s_boundResolved  = VK_NULL_HANDLE;  // resolved base-colour view (binding 6)

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

    // Does the presentation surface apply the sRGB OETF itself? If it does, the shader
    // must NOT encode again; if it does not (the usual case here — ChooseSurfaceFormat
    // prefers B8G8R8A8_UNORM), the shader owns the encode. See the display-gamma block
    // in Execute for why this is a surface property and not a user setting.
    inline bool IsSrgbFormat(VkFormat f)
    {
        switch (f) {
        case VK_FORMAT_R8G8B8A8_SRGB:
        case VK_FORMAT_B8G8R8A8_SRGB:
        case VK_FORMAT_R8G8B8_SRGB:
        case VK_FORMAT_B8G8R8_SRGB:
        case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
            return true;
        default:
            return false;
        }
    }

    struct TonemapPush { float p0[4]; float p1[4]; float p2[4]; float p3[4]; float p4[4]; float p5[4]; float p6[4]; };

    // ---- Temporal auto-exposure (eye adaptation) ----------------------------------
    // The instantaneous exposure (middlegray / metered avg-luminance) makes the image
    // visibly darken/brighten as the camera tilts between bright sky and dark ground.
    // We read back the HDR target's 1×1 top mip (whole-frame average) with a few-frame
    // lag, compute the target exposure on the CPU, and EASE the applied exposure toward
    // it over ps_r_exp_adapt seconds — the eye-adaptation "breathing" the user reported.
    constexpr u32 kExpFrames = 4;                 // readback ring (≥ frames-in-flight → lag-safe)
    CVulkanBuffer s_expBuf;                        // kExpFrames × RGBA16F texel (8 B each)
    u8*           s_expMapped   = nullptr;
    u32           s_expFrame    = 0;               // ring cursor / frames processed
    float         s_exposure    = 1.0f;            // the smoothed, applied exposure
    bool          s_expValid    = false;           // false until the first valid readback

    // IEEE half (RGBA16F texel) → float, for the CPU-side average-luminance decode.
    inline float HalfToFloat(u16 h)
    {
        u32 sign = u32(h & 0x8000) << 16;
        u32 exp  = (h & 0x7C00) >> 10;
        u32 mant = (h & 0x03FF);
        u32 f;
        if (exp == 0) {
            if (mant == 0) f = sign;
            else { exp = 1; while ((mant & 0x0400) == 0) { mant <<= 1; --exp; } mant &= 0x03FF;
                   f = sign | ((exp + 112) << 23) | (mant << 13); }
        } else if (exp == 0x1F) {
            f = sign | 0x7F800000u | (mant << 13);
        } else {
            f = sign | ((exp + 112) << 23) | (mant << 13);
        }
        float out; std::memcpy(&out, &f, 4); return out;
    }
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

    // Set 0: binding 0 = HDR scene (mip chain, avg-lum), 1 = blurred bloom, 2 = distortion,
    // 3 = scene depth (SSR puddles), 4 = integrated volumetrics (sampler3D),
    // 5 = SSIL indirect light (half-res), 6 = RESOLVED base colour (display res; the DLSS
    // upscaled output when upscaling, else the render-res scene). All FS.
    VkDescriptorSetLayoutBinding b[7]{};
    for (u32 i = 0; i < 7; ++i) {
        b[i].binding = i; b[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lci.bindingCount = 7; lci.pBindings = b;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &s_SetLayout) != VK_SUCCESS) {
        Msg("![VK Tonemap] set layout failed"); return false;
    }

    VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxImages * 7 };
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
    for (u32 i = 0; i < kMaxImages; ++i) VK::Prof::NameSet(s_Set[i], "Tonemap.Set");   // TEMP diag: VUID hunt

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

    // Temporal-exposure readback ring: one RGBA16F texel (8 B) per frame slot.
    s_expBuf.Create(kExpFrames * 8, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
    s_expMapped = static_cast<u8*>(s_expBuf.Map());
    s_expFrame  = 0; s_exposure = 1.0f; s_expValid = false;

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
    s_expBuf.Destroy(); s_expMapped = nullptr; s_expValid = false; s_exposure = 1.0f; s_expFrame = 0;
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

    // ---- Temporal auto-exposure: read the previous (lagged) 1×1 average, ease the
    // applied exposure toward its target over ps_r_exp_adapt seconds, then copy THIS
    // frame's average into the ring for a later frame. Eliminates the sky/ground
    // brightness "breathing" (instant exposure). p5.x = 0 on the first frames → the
    // shader falls back to the instantaneous estimate until a readback is valid.
    {
        const u32 slot = s_expFrame % kExpFrames;
        if (s_expMapped && s_expFrame >= kExpFrames) {
            const u16* h = reinterpret_cast<const u16*>(s_expMapped + size_t(slot) * 8);
            float r = HalfToFloat(h[0]), g = HalfToFloat(h[1]), b = HalfToFloat(h[2]);
            float avgLum = std::max(r * 0.2126f + g * 0.7152f + b * 0.0722f, 1e-4f);
            float target = std::min(std::max(kMiddleGray / (avgLum + kLowLum), kExpMin), kExpMax) * kExpComp;
            float dt  = (Device.fTimeDelta > 0.f && Device.fTimeDelta < 0.25f) ? Device.fTimeDelta : 0.016f;
            float tau = ps_r_exp_adapt;
            if (tau > 0.001f) s_exposure += (target - s_exposure) * (1.0f - expf(-dt / tau));
            else              s_exposure  = target;
            s_expValid = true;
        }
        // The editor used to pin ps_r2_img_gamma = 2.2 and ps_r2_img_cg here. Both are
        // gone: the CDL pin was a no-op (0.5,0.5,0.5 IS the default, and nothing in the
        // engine ever writes that cvar — ssfx_color_grading is a user knob, not level
        // data), and the gamma pin is now handled properly for BOTH game and editor by
        // the display-encode split at the push below.
        // What genuinely IS editor-only stays: an EMPTY viewport gives the meter nothing
        // to average and auto-exposure crushes the frame to black. That premise holds
        // only while the viewport is empty — once the host pushes a real scene there is
        // plenty to meter, and a value tuned for one model on a flat backdrop blows a lit
        // outdoor level out to white. So meter as soon as we can.
        if (Core.Params && strstr(Core.Params, "-vk_editor") && VKEditor::HostModelCount() == 0)
        {
            s_exposure = 2.0f;
            s_expValid = true;
        }
        if (s_expMapped) {
            const u32 topMip = SceneColor::MipLevels() - 1;
            VkImage img = SceneColor::GetImage(idx);
            VkImageMemoryBarrier br{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
            br.srcQueueFamilyIndex = br.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            br.image = img; br.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, topMip, 1, 0, 1 };
            br.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; br.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            br.srcAccessMask = VK_ACCESS_SHADER_READ_BIT; br.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &br);

            VkBufferImageCopy region{};
            region.bufferOffset      = VkDeviceSize(slot) * 8;
            region.imageSubresource  = { VK_IMAGE_ASPECT_COLOR_BIT, topMip, 0, 1 };
            region.imageExtent       = { 1, 1, 1 };
            vkCmdCopyImageToBuffer(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s_expBuf.GetHandle(), 1, &region);

            br.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; br.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            br.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT; br.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &br);
        }
        ++s_expFrame;
    }

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
    // Binding 6 (resolved base colour): the DLSS-upscaled display-res output when DLSS
    // ran this frame, else the render-res scene (== display res when not upscaling).
    // ResolvedThisFrame: the evaluate really wrote the DLSS output this frame — a
    // failed SL evaluate leaves it undefined, and sampling it painted the whole
    // screen blue (the mid-game DLSS-enable OOM crash, 22:51 log).
    const bool        resolvedUp  = VK::Dlss::Enabled() && VK::MotionVec::Enabled() && VK::Dlss::FeatureReady()
                                 && VK::Dlss::ResolvedThisFrame();
    const VkImageView dlssOutView = VK::Dlss::OutputView();
    if (gen != s_boundGen || bloomGen != s_boundBloomGen || distortGen != s_boundDistortGen
        || Swapchain.m_DepthView != s_boundDepth || volGen != s_boundVolGen || ssilGen != s_boundSsilGen
        || resolvedUp != s_boundResolvedUp || (resolvedUp && dlssOutView != s_boundResolved)) {
        const u32 n = SceneColor::Count();
        for (u32 i = 0; i < n && i < kMaxImages; ++i) {
            VkDescriptorImageInfo ii[7]{};
            ii[0].sampler = s_Sampler; ii[0].imageView = SceneColor::GetSampleView(i);  // full mip chain (avg-lum)
            ii[1].sampler = s_Sampler; ii[1].imageView = BloomPass::GetResultView();
            ii[2].sampler = s_Sampler; ii[2].imageView = ParticlePass::GetDistortView();
            ii[3].sampler = s_Sampler; ii[3].imageView = Swapchain.m_DepthView;          // SSR puddles
            ii[4].sampler = Vol::GetSampler(); ii[4].imageView = Vol::GetIntegratedView(); // integrated volumetrics (3D)
            ii[5].sampler = SSAOPass::GetSampler(); ii[5].imageView = SSAOPass::GetILResultView(); // indirect light (GTAO MRT target 1)
            ii[6].sampler = s_Sampler; ii[6].imageView = (resolvedUp && dlssOutView != VK_NULL_HANDLE)
                                                       ? dlssOutView : SceneColor::GetSampleView(i);  // resolved base
            ii[0].imageLayout = ii[1].imageLayout = ii[2].imageLayout = ii[3].imageLayout = ii[4].imageLayout = ii[5].imageLayout = ii[6].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            if (ii[1].imageView == VK_NULL_HANDLE) ii[1].imageView = SceneColor::GetSampleView(i); // pre-bloom fallback
            if (ii[2].imageView == VK_NULL_HANDLE) ii[2].imageView = SceneColor::GetSampleView(i); // no-distort fallback (never sampled: scale 0)
            if (ii[3].imageView == VK_NULL_HANDLE) ii[3].imageView = SceneColor::GetSampleView(i); // no-depth fallback (never sampled: wetness 0)
            if (ii[5].imageView == VK_NULL_HANDLE) { ii[5].sampler = s_Sampler; ii[5].imageView = SceneColor::GetSampleView(i); } // SSIL off/not-ready (never sampled: p5.z 0)
            // Write each valid binding by explicit index — binding 4 (3D volume)
            // exists from Vol::Init (eager); the others can transiently be null.
            VkWriteDescriptorSet w[7]{};
            u32 wc = 0;
            for (u32 k = 0; k < 7; ++k) {
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
        s_boundResolvedUp = resolvedUp;
        s_boundResolved   = resolvedUp ? dlssOutView : VK_NULL_HANDLE;
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

    // Composite renders at DISPLAY resolution (the swapchain image). When DLSS
    // upscales, ctx.extent is the smaller render res — the base colour comes from the
    // display-res DLSS output (binding 6); the render-res effects (bloom/depth/SSIL/
    // vol) are sampled with normalised UVs, so they upsample bilinearly.
    const VkExtent2D outExt = ctx.displayExtent;
    VkRenderingInfo ri{};
    ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea.extent    = outExt;
    ri.layerCount           = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments    = &cAtt;
    vkCmdBeginRendering(cmd, &ri);

    VkViewport vp{ 0.f, 0.f, (float)outExt.width, (float)outExt.height, 0.f, 1.f };
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{ {0,0}, outExt };
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
    // Bloom was forced OFF in the editor so the grid and gizmo lines would stay
    // crisp instead of glowing. That reason no longer holds: both now draw
    // POST-tonemap (EditorOverlay::ExecutePostTonemap), where bloom cannot reach
    // them. The gate only cost the scene its highlight glow and made the viewport
    // read flatter than the game — the opposite of what a WYSIWYG viewport is for.
    push.p1[3] = kBloomIntensity;
    // R4 CDL grading (ACES_settings.h): Slope = r2_img_exposure, Power =
    // 2*(1 - r2_img_cg), Saturation = r2_img_saturation, gamma from
    // img_corrections — all live console knobs (ssfx_exposure/_gamma/
    // _saturation/ssfx_color_grading), neutral by default.
    push.p2[0] = ps_r2_img_exposure;
    push.p2[1] = ps_r2_img_saturation;
    // DISPLAY ENCODE vs ARTISTIC TRIM. These are genuinely two different things, but
    // whether the encode is OWED depends on the pipeline feeding it, and this engine
    // runs TWO of them.
    //
    // GAME PATH — gamma space, encode already paid. Albedo is never linearized on the
    // way in: DXGIFormatToVk (vk_texture.cpp) maps every _SRGB DDS variant to its UNORM
    // VkFormat, and the legacy BC1/2/3 path does the same, so the sampler hands the
    // shader gamma-encoded values and no hardware sRGB decode happens anywhere. The
    // lighting math then runs on those, and the whole look — sun boost, ambient floor,
    // the Reinhard white point, the exposure constants above — is tuned around it. The
    // frame that reaches this push is therefore ALREADY display-referred. Applying an
    // OETF here is a SECOND encode: mids lift by roughly 2x and the picture blows out.
    // That is also why ChooseSurfaceFormat deliberately prefers B8G8R8A8_UNORM over the
    // _SRGB surface (see its comment) — the same double-gamma, one layer lower.
    //
    // EDITOR PATH — genuinely linear, encode owed. The host scene is lit by real linear
    // sky ambient / IBL / GTAO with no gamma-space lightmaps, so at 1.0 it renders dark
    // and does need the 2.2.
    //
    // So the -vk_editor term is not a band-aid: it selects between two color pipelines.
    // Collapsing it into an unconditional 2.2 is what made the GAME too bright.
    // pow(pow(c, 1/display), 1/user) == pow(c, 1/(display*user)) — one exponent, shader
    // unchanged. hwSrgb stays honored: if we ever land on an _SRGB surface the hardware
    // owns the encode and we must not double it.
    // Under r_linear_color the GAME joins the editor on the linear side: albedo arrives
    // sRGB-decoded, light/env colours are linearised on upload, so the frame reaching
    // here is finally scene-referred linear and DOES owe the display an OETF. That is
    // what makes this one exponent correct in both modes rather than a special case:
    //   linear pipeline  -> encode (2.2)
    //   gamma pipeline   -> pass through (1.0), because the encode was never undone
    // and hwSrgb still wins over both, since then the surface hardware encodes for us.
    //
    // Kept as a single pow exponent (not the piecewise sRGB curve) so it still composes
    // with the artistic ssfx_gamma trim: pow(pow(c,1/d),1/u) == pow(c,1/(d*u)). The two
    // differ only in the deepest shadows; swapping in the piecewise curve is an L-3
    // refinement, and it would have to give up that composition.
    const bool  isEditor     = Core.Params && strstr(Core.Params, "-vk_editor");
    const bool  hwSrgb       = IsSrgbFormat(Swapchain.m_Format);
    const bool  wantEncode   = (ps_r_linear_color != 0) || isEditor;
    const float displayGamma = (wantEncode && !hwSrgb) ? 2.2f : 1.0f;
    const float userGamma    = (ps_r2_img_gamma > 0.05f) ? ps_r2_img_gamma : 1.0f;
    push.p2[2] = 1.0f / (displayGamma * userGamma);
    // Heat-haze: scale 0 until the distort RT exists (binding 2 then holds the
    // scene-view fallback and the shader skips the sample entirely).
    push.p2[3] = (ParticlePass::GetDistortView() != VK_NULL_HANDLE) ? kDistortAmount : 0.0f;
    push.p3[0] = 2.0f * (1.0f - ps_r2_img_cg.x);
    push.p3[1] = 2.0f * (1.0f - ps_r2_img_cg.y);
    push.p3[2] = 2.0f * (1.0f - ps_r2_img_cg.z);
    push.p3[3] = ps_r_dither;   // output TPDF dither (LSBs) — de-bands the 8-bit swapchain write
    // Volumetric composite (r_vol): mode + the exp-Z grid params from the LAST
    // Vol::Execute (the EXACT inverse vol_inject used). 0 off / 1 composite / 2 debug.
    const Vol::GridZParams gz = Vol::GetGridZ();
    // This used to carry a `&& !s_editorTM` term: with no level the froxel volume was
    // never integrated (inject/integrate live in Pass_World), so uVolume held
    // transmittance 0 and the composite (scene*T + inscatter) absorbed the whole lit
    // scene to black. That was a band-aid over the missing integrate, and it cost the
    // editor its aerial perspective. Pass_EditorDynamics now drives Vol::Execute, so
    // the composite is fed real data and the term is gone — if fog ever comes back
    // black in the editor, check that Vol::Execute actually ran, don't re-add a gate.
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
    push.p5[0] = s_expValid ? s_exposure : 0.0f;   // CPU-smoothed exposure (0 = shader computes it)
    // Publish the same exposure to DLSS (r_dlss_exp real-exposure path). Next frame's
    // Evaluate (which runs before this pass) reads it — one frame stale is fine, the
    // value eases over seconds.
    VK::Dlss::SetExposureHint(s_expValid ? s_exposure : 1.0f);
    push.p5[1] = (ilReady && ps_r_ssil_debug) ? 1.0f : 0.0f;
    push.p5[2] = 0.0f;
    // r_vsm_debug_dyn: red-tint pixels shadowed by the VSM dynamic atlas (mask.B,
    // set 1 binding 14 — the EnvLight set is already bound; dummy-safe when off).
    push.p5[3] = (ps_r_vsm_debug_dyn && VSM::MaskReady()) ? 1.0f : 0.0f;
    // God-ray GROUND SPLASH (r_sun_beam_splash): boost the in-scatter where a bright
    // shaft lands on darker ground so the beam "разбивается о терейн" instead of
    // dissolving (see tonemap.frag). Needs the volume, so off unless the composite runs.
    push.p6[0] = volOn ? ps_r_sun_beam_splash : 0.0f;   // strength (0 = off)
    push.p6[1] = ps_r_sun_beam_splash_thr;              // in-scatter luminance threshold
    // CAS sharpen on the DLSS-resolved colour (only when DLSS actually resolved this
    // frame — sharpening the plain scene would double up with its native mips).
    push.p6[2] = resolvedUp ? ps_r_dlss_sharp : 0.0f;
    // DLSS debug view (r_dlss_debug): magnitude = mode, SIGN = whether the composite
    // actually reads the DLSS output this frame (the gate the shader can't see).
    push.p6[3] = resolvedUp ? float(ps_r_dlss_debug) : -float(ps_r_dlss_debug);
    // One-shot gate log on every state change — pairs with r_dlss_debug 3.
    {
        static int s_lastResolved = -1;
        if (int(resolvedUp) != s_lastResolved) {
            s_lastResolved = int(resolvedUp);
            Msg("[VK DLSS] tonemap composite: base = %s (sharp %.2f, dlss view %p)",
                resolvedUp ? "DLSS output" : "scene colour", ps_r_dlss_sharp, (void*)dlssOutView);
        }
    }
    vkCmdPushConstants(cmd, s_PipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);

    vkCmdDraw(cmd, 3, 1, 0, 0);
    vkCmdEndRendering(cmd);
    // Swapchain stays COLOR_ATTACHMENT for the UI pass; End brings it to PRESENT.
}

}  // namespace VK
