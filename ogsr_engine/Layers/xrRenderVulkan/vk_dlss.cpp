// xrRenderVulkan — NVIDIA DLSS 4.5 (NGX) integration. See vk_dlss.h.
//
// Increment 1: initialise NGX on the live VK device and probe capabilities. The
// required NGX device extensions (VK_NVX_binary_import, VK_NVX_image_view_handle,
// VK_KHR_push_descriptor) are already enabled at device creation (vk_core.h
// g_NgxDeviceExtensions → HW_Vulkan). Frame-Generation will need MORE extensions
// (optical flow / present-id) — added when the DLSS-G feature lands.
#include "stdafx.h"
#include "vk_dlss.h"
#include "HW_Vulkan.h"
#include "vk_image.h"      // VK::CreateImage2D / CreateImageView
#include "vk_barriers.h"       // ImageBarrier — transition DLSS resources to GENERAL for NGX

#include "nvsdk_ngx.h"
#include "nvsdk_ngx_vk.h"
#include "nvsdk_ngx_helpers.h"        // NVSDK_NGX_Parameter_GetI/GetUI wrappers
#include "nvsdk_ngx_helpers_vk.h"
#include "nvsdk_ngx_defs_dlssg.h"     // NVSDK_NGX_Parameter_FrameGeneration_* (MFG)

extern u32 ps_r_dlss;           // console toggle (vk_console_min.cpp, global scope)
extern u32 ps_r_dlss_quality;   // 0=DLAA 1=Quality 2=Balanced 3=Performance 4=UltraPerf
extern u32 ps_r_dlss_preset;    // model preset hint: 0=driver default, 6=F(CNN), 10=J, 11=K (transformer)
extern int ps_r_dlss_jitter_flip; // sign of the REPORTED jitter offset: bit0=flip X, bit1=flip Y (live A/B)
extern int ps_r_dlss_exp;         // 1 = real exposure via 1x1 texture (default), 0 = NGX AutoExposure flag (old)

namespace VK { namespace Dlss {

namespace {
    // Map r_dlss_quality → NGX perf/quality preset. Quality 0 = DLAA (native res AA).
    NVSDK_NGX_PerfQuality_Value PerfQualityForCvar()
    {
        switch (ps_r_dlss_quality) {
            case 1:  return NVSDK_NGX_PerfQuality_Value_MaxQuality;       // ~0.67x
            case 2:  return NVSDK_NGX_PerfQuality_Value_Balanced;         // ~0.58x
            case 3:  return NVSDK_NGX_PerfQuality_Value_MaxPerf;          // ~0.50x
            case 4:  return NVSDK_NGX_PerfQuality_Value_UltraPerformance; // ~0.33x
            default: return NVSDK_NGX_PerfQuality_Value_DLAA;             // native
        }
    }
}

namespace {
    bool s_inited  = false;
    bool s_srAvail = false;
    bool s_fgAvail = false;
    NVSDK_NGX_Parameter* s_caps = nullptr;

    // App/project identity for NGX telemetry. Any stable GUID string is fine for a
    // custom engine (no NVIDIA-assigned app id); ENGINE_TYPE_CUSTOM + our version.
    const char* kProjectId    = "b7f4e2a1-9c3d-4f6b-8a12-0e5d7c9a3b21";
    const char* kEngineVer     = "1.0";
}

bool Inited()             { return s_inited; }
bool SuperResAvailable()  { return s_inited && s_srAvail; }
bool FrameGenAvailable()  { return s_inited && s_fgAvail; }
bool Enabled()            { return s_inited && s_srAvail && ps_r_dlss != 0; }

bool Init()
{
    if (s_inited) return true;
    if (VulkanHW.m_Instance == VK_NULL_HANDLE || VulkanHW.m_Device == VK_NULL_HANDLE) {
        Msg("![VK DLSS] device not ready — skipping NGX init");
        return false;
    }
    if (!g_bNgxExtensionsEnabled)
        Msg("~[VK DLSS] not all NGX device extensions were enabled — init may fail");

    // NGX looks for the feature snippet DLLs (nvngx_dlss.dll / nvngx_dlssg.dll) next
    // to the exe by default; app data path L"." = the working dir (bin_x64).
    NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_Init_with_ProjectID(
        kProjectId, NVSDK_NGX_ENGINE_TYPE_CUSTOM, kEngineVer, L".",
        VulkanHW.m_Instance, VulkanHW.m_PhysicalDevice, VulkanHW.m_Device,
        vkGetInstanceProcAddr, vkGetDeviceProcAddr, nullptr, NVSDK_NGX_Version_API);
    if (NVSDK_NGX_FAILED(r)) {
        Msg("![VK DLSS] NGX_VULKAN_Init failed: 0x%08X (no RTX GPU / driver / DLLs?)", (unsigned)r);
        return false;
    }

    r = NVSDK_NGX_VULKAN_GetCapabilityParameters(&s_caps);
    if (NVSDK_NGX_FAILED(r) || s_caps == nullptr) {
        Msg("![VK DLSS] GetCapabilityParameters failed: 0x%08X", (unsigned)r);
        NVSDK_NGX_VULKAN_Shutdown1(VulkanHW.m_Device);
        return false;
    }

    int srAvail = 0, srNeedsDrv = 0, srDrvMaj = 0, srDrvMin = 0;
    int fgAvail = 0, fgNeedsDrv = 0, fgDrvMaj = 0, fgDrvMin = 0;
    NVSDK_NGX_Parameter_GetI(s_caps, NVSDK_NGX_Parameter_SuperSampling_Available,               &srAvail);
    NVSDK_NGX_Parameter_GetI(s_caps, NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver,      &srNeedsDrv);
    NVSDK_NGX_Parameter_GetI(s_caps, NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMajor,   &srDrvMaj);
    NVSDK_NGX_Parameter_GetI(s_caps, NVSDK_NGX_Parameter_SuperSampling_MinDriverVersionMinor,   &srDrvMin);
    NVSDK_NGX_Parameter_GetI(s_caps, NVSDK_NGX_Parameter_FrameGeneration_Available,             &fgAvail);
    NVSDK_NGX_Parameter_GetI(s_caps, NVSDK_NGX_Parameter_FrameGeneration_NeedsUpdatedDriver,    &fgNeedsDrv);
    NVSDK_NGX_Parameter_GetI(s_caps, NVSDK_NGX_Parameter_FrameGeneration_MinDriverVersionMajor, &fgDrvMaj);
    NVSDK_NGX_Parameter_GetI(s_caps, NVSDK_NGX_Parameter_FrameGeneration_MinDriverVersionMinor, &fgDrvMin);

    s_srAvail = (srAvail != 0);
    s_fgAvail = (fgAvail != 0);
    s_inited  = true;

    Msg("[VK DLSS] NGX initialised.");
    Msg("[VK DLSS]   Super Resolution : %s%s (min driver %d.%d)",
        s_srAvail ? "AVAILABLE" : "unavailable",
        srNeedsDrv ? " [NEEDS NEWER DRIVER]" : "", srDrvMaj, srDrvMin);
    Msg("[VK DLSS]   Frame Generation : %s%s (min driver %d.%d)",
        s_fgAvail ? "AVAILABLE" : "unavailable",
        fgNeedsDrv ? " [NEEDS NEWER DRIVER]" : "", fgDrvMaj, fgDrvMin);
    return true;
}

// ============================================================================
// Super Resolution (increment 2): feature lifecycle + jitter + output + evaluate.
// ============================================================================
namespace {
    NVSDK_NGX_Parameter* s_params  = nullptr;   // feature create/eval parameter block
    NVSDK_NGX_Handle*    s_feature = nullptr;
    u32 s_fRenderW = 0, s_fRenderH = 0, s_fDispW = 0, s_fDispH = 0;
    u32 s_fPreset  = 0xFFFFFFFFu;   // preset the live feature was built with (r_dlss_preset)
    u32 s_fInputsGen = 0xFFFFFFFFu; // input-target generation the feature was built against
    u32 s_fExpMode = 0xFFFFFFFFu;   // exposure mode the feature was built with (r_dlss_exp)

    // Real-exposure path (r_dlss_exp 1): 1x1 R32F texture cleared to the tonemap's
    // CPU-smoothed exposure each Evaluate — NGX normalizes its internal processing
    // by it (AutoExposure flag dropped from the feature). Prevents the transformer's
    // black undershoot on high-contrast sub-pixel detail (distant crowns vs sky).
    float         s_expHint  = 1.0f;
    VkImage       s_expImg   = VK_NULL_HANDLE;
    VmaAllocation s_expAlloc = VK_NULL_HANDLE;
    VkImageView   s_expView  = VK_NULL_HANDLE;

    // Display-res upscaled output (RGBA16F; DLSS writes via storage, tonemap samples).
    VkImage       s_outImg   = VK_NULL_HANDLE;
    VmaAllocation s_outAlloc = VK_NULL_HANDLE;
    VkImageView   s_outView  = VK_NULL_HANDLE;
    const VkFormat s_outFmt  = VK_FORMAT_R16G16B16A16_SFLOAT;
    u32 s_outW = 0, s_outH = 0;
    bool s_outFirst = true;   // first barrier is from UNDEFINED

    // Halton(2,3) sub-pixel jitter, in render-pixel space [-0.5, 0.5].
    u32   s_jitterPhase  = 0;
    float s_jitterPixX   = 0.0f, s_jitterPixY = 0.0f;
    u32   s_jitterCount  = 16;
    // Render dims captured by NewFrameJitter, so GetProjJitterNDC can convert to NDC
    // even on the first frames (before EnsureFeature has set the feature dims).
    u32   s_jitterW      = 0, s_jitterH = 0;

    float Halton(u32 index, u32 base)
    {
        float f = 1.0f, r = 0.0f;
        for (u32 i = index; i > 0; i /= base) { f /= float(base); r += f * float(i % base); }
        return r;
    }

    VkImageSubresourceRange FullRange(VkImageAspectFlags aspect)
    {
        VkImageSubresourceRange r{};
        r.aspectMask = aspect; r.levelCount = 1; r.layerCount = 1;
        return r;
    }

    NVSDK_NGX_Resource_VK WrapImage(const VK::Dlss::Img& im, u32 w, u32 h, VkImageAspectFlags aspect, bool readWrite)
    {
        return NVSDK_NGX_Create_ImageView_Resource_VK(
            im.view, im.image, FullRange(aspect), im.format, w, h, readWrite);
    }
}

void EnsureOutput(u32 displayW, u32 displayH)
{
    if (!s_inited) return;
    if (s_outImg != VK_NULL_HANDLE && displayW == s_outW && displayH == s_outH) return;
    if (s_outView) { vkDestroyImageView(VulkanHW.m_Device, s_outView, nullptr); s_outView = VK_NULL_HANDLE; }
    if (s_outImg)  { VK::Vram::DestroyImage(VulkanHW.m_Allocator, s_outImg, s_outAlloc); s_outImg = VK_NULL_HANDLE; s_outAlloc = VK_NULL_HANDLE; }
    if (displayW == 0 || displayH == 0) return;

    if (!VK::CreateImage2D(s_outFmt, { displayW, displayH },
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            s_outImg, s_outAlloc, "DLSS.Output"))
        return;
    s_outView = VK::CreateImageView(s_outImg, s_outFmt);
    if (s_outView == VK_NULL_HANDLE) {
        VK::Vram::DestroyImage(VulkanHW.m_Allocator, s_outImg, s_outAlloc); s_outImg = VK_NULL_HANDLE; return;
    }
    s_outW = displayW; s_outH = displayH; s_outFirst = true;
    Msg("[VK DLSS] output target ready (%ux%u RGBA16F)", displayW, displayH);
}

VkImageView OutputView()   { return s_outView; }
VkImage     OutputImage()  { return s_outImg; }
VkFormat    OutputFormat() { return s_outFmt; }
bool        OutputReady()  { return s_outImg != VK_NULL_HANDLE; }

// Under the SL route there is no raw NGX feature object — s_slShadow stands in
// for it (see EnsureFeatureSL). s_resolved = this frame's evaluate actually wrote
// OutputView; without it the tonemap sampled an undefined image after a failed
// SL evaluate ("blue screen").
namespace { bool s_slShadow = false; bool s_resolved = false; }
bool        FeatureReady() { return (s_feature != nullptr || s_slShadow) && s_outImg != VK_NULL_HANDLE; }
bool        ResolvedThisFrame()       { return s_resolved; }
void        SetResolvedThisFrame(bool ok) { s_resolved = ok; }

void NewFrameJitter(u32 renderW, u32 renderH)
{
    s_resolved = false;   // set again only by a successful evaluate this frame
    s_jitterW = renderW; s_jitterH = renderH;
    s_jitterPhase = (s_jitterPhase + 1) % s_jitterCount;
    // Halton is 1-based for a well-distributed sequence; map [0,1) → [-0.5, 0.5].
    s_jitterPixX = Halton(s_jitterPhase + 1, 2) - 0.5f;
    s_jitterPixY = Halton(s_jitterPhase + 1, 3) - 0.5f;
}

void GetJitterPix(float& outX, float& outY)
{
    outX = s_jitterPixX; outY = s_jitterPixY;
}

void GetProjJitterNDC(float& outX, float& outY)
{
    // Pixel offset → NDC: 2*px/dim. Use the render dims captured by NewFrameJitter
    // (the feature dims are still 0 before the first EnsureFeature).
    const float w = s_jitterW ? float(s_jitterW) : 1.0f;
    const float h = s_jitterH ? float(s_jitterH) : 1.0f;
    outX = 2.0f * s_jitterPixX / w;
    outY = 2.0f * s_jitterPixY / h;
}

namespace {
    // Cache the optimal render dims so we only query NGX when display size / quality changes.
    u32 s_reDispW = 0, s_reDispH = 0, s_reQual = 0xFFFFFFFFu;
    u32 s_reRenderW = 0, s_reRenderH = 0;
}

void GetRenderExtent(u32 displayW, u32 displayH, u32& outW, u32& outH)
{
    outW = displayW; outH = displayH;
    if (!s_inited || !s_srAvail || displayW == 0 || displayH == 0) return;
    if (ps_r_dlss_quality == 0) { s_reDispW = displayW; s_reDispH = displayH; s_reQual = 0; s_reRenderW = displayW; s_reRenderH = displayH; return; }

    if (displayW == s_reDispW && displayH == s_reDispH && ps_r_dlss_quality == s_reQual) {
        outW = s_reRenderW; outH = s_reRenderH; return;   // cached
    }

    u32 optW = 0, optH = 0, maxW = 0, maxH = 0, minW = 0, minH = 0; float sharp = 0.0f;
    NVSDK_NGX_Result r = NGX_DLSS_GET_OPTIMAL_SETTINGS(
        s_caps, displayW, displayH, PerfQualityForCvar(),
        &optW, &optH, &maxW, &maxH, &minW, &minH, &sharp);
    if (NVSDK_NGX_FAILED(r) || optW == 0 || optH == 0) {
        Msg("![VK DLSS] GetOptimalSettings failed (0x%08X) — falling back to native", (unsigned)r);
        s_reDispW = displayW; s_reDispH = displayH; s_reQual = ps_r_dlss_quality;
        s_reRenderW = displayW; s_reRenderH = displayH;   // cache the fallback so we don't re-query every frame
        return;
    }
    s_reDispW = displayW; s_reDispH = displayH; s_reQual = ps_r_dlss_quality;
    s_reRenderW = optW; s_reRenderH = optH;
    outW = optW; outH = optH;
    Msg("[VK DLSS] render extent %ux%u -> display %ux%u (quality %u)", optW, optH, displayW, displayH, ps_r_dlss_quality);
}

bool Upscaling()
{
    return s_inited && s_srAvail && ps_r_dlss != 0 && ps_r_dlss_quality != 0;
}

bool EnsureFeature(VkCommandBuffer cmd, u32 renderW, u32 renderH, u32 displayW, u32 displayH, u32 inputsGen)
{
    if (!s_inited || !s_srAvail) return false;
    if (renderW == 0 || renderH == 0 || displayW == 0 || displayH == 0) return false;
    if (s_feature && renderW == s_fRenderW && renderH == s_fRenderH && displayW == s_fDispW && displayH == s_fDispH
        && ps_r_dlss_preset == s_fPreset && inputsGen == s_fInputsGen && u32(ps_r_dlss_exp != 0) == s_fExpMode)
        return true;   // already built for these dims + preset + input targets + exposure mode

    // Rare path (quality/preset/window change, or the input targets were recreated —
    // e.g. the r_dlss off→on toggle resizes SceneColor/MV/depth while the feature
    // idles; reusing it then evaluates against DESTROYED images and hangs the GPU).
    // In-flight frames may still reference the old feature's internals — idle before
    // releasing it. (A render-extent change already idled in CRender::Begin; a
    // preset-only or generation-only change reaches here without it.)
    if (s_feature) vkDeviceWaitIdle(VulkanHW.m_Device);
    ReleaseFeature();

    if (s_params == nullptr) {
        NVSDK_NGX_Result pr = NVSDK_NGX_VULKAN_AllocateParameters(&s_params);
        if (NVSDK_NGX_FAILED(pr) || s_params == nullptr) {
            Msg("![VK DLSS] AllocateParameters failed: 0x%08X", (unsigned)pr); return false;
        }
    }

    // Model preset hint (r_dlss_preset), set for EVERY quality mode before create:
    // 11 = Preset K, the DLSS 4 transformer — markedly sharper reconstruction than the
    // old CNN presets. Matters most at UltraPerf, whose driver DEFAULT is still the
    // CNN preset F. 0 = leave the driver/OTA default.
    {
        const u32 p = ps_r_dlss_preset;
        NVSDK_NGX_Parameter_SetUI(s_params, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA,             p);
        NVSDK_NGX_Parameter_SetUI(s_params, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality,          p);
        NVSDK_NGX_Parameter_SetUI(s_params, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced,         p);
        NVSDK_NGX_Parameter_SetUI(s_params, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance,      p);
        NVSDK_NGX_Parameter_SetUI(s_params, NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance, p);
    }

    NVSDK_NGX_DLSS_Create_Params cp{};
    cp.Feature.InWidth        = renderW;
    cp.Feature.InHeight       = renderH;
    cp.Feature.InTargetWidth  = displayW;
    cp.Feature.InTargetHeight = displayH;
    // Preset MUST match the one GetRenderExtent queried for these render dims
    // (quality 0 → DLAA/native, 1..4 → Quality/Balanced/Perf/UltraPerf upscale).
    cp.Feature.InPerfQualityValue = PerfQualityForCvar();
    // HDR scene colour; MVs are at render res; standard (non-reverse) depth.
    // NO MVJittered (Option A): color/depth are jittered (CRender::Begin), but the MV
    // passes reproject with the UNJITTERED view-proj and re-apply the jitter only to
    // gl_Position — so motion vectors are jitter-free (NVIDIA's canonical input).
    // DLSS applies the sub-pixel offset itself from the InJitterOffset we pass to
    // Evaluate. Exposure: r_dlss_exp 1 (default) supplies the REAL tonemap exposure
    // as a 1x1 texture in Evaluate — NVIDIA's recommended path for raw-HDR input;
    // the AutoExposure flag (r_dlss_exp 0) let NGX guess and the transformer
    // undershot to BLACK on high-contrast sub-pixel foliage.
    cp.InFeatureCreateFlags =
        NVSDK_NGX_DLSS_Feature_Flags_IsHDR |
        NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;
    if (ps_r_dlss_exp == 0)
        cp.InFeatureCreateFlags |= NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;

    NVSDK_NGX_Result r = NGX_VULKAN_CREATE_DLSS_EXT(cmd, 1u, 1u, &s_feature, s_params, &cp);
    if (NVSDK_NGX_FAILED(r) || s_feature == nullptr) {
        Msg("![VK DLSS] CreateFeature failed: 0x%08X", (unsigned)r);
        s_feature = nullptr; return false;
    }
    s_fRenderW = renderW; s_fRenderH = renderH; s_fDispW = displayW; s_fDispH = displayH;
    s_fPreset  = ps_r_dlss_preset;
    s_fInputsGen = inputsGen;
    s_fExpMode = u32(ps_r_dlss_exp != 0);

    // 1x1 R32F exposure texture for the real-exposure path (created once, reused).
    if (s_fExpMode && s_expImg == VK_NULL_HANDLE) {
        if (VK::CreateImage2D(VK_FORMAT_R32_SFLOAT, { 1, 1 },
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                s_expImg, s_expAlloc, "DLSS.Exposure")) {
            s_expView = VK::CreateImageView(s_expImg, VK_FORMAT_R32_SFLOAT);
            if (s_expView == VK_NULL_HANDLE) {
                VK::Vram::DestroyImage(VulkanHW.m_Allocator, s_expImg, s_expAlloc);
                s_expImg = VK_NULL_HANDLE; s_expAlloc = VK_NULL_HANDLE; s_expView = VK_NULL_HANDLE;
            }
        }
        if (s_expImg == VK_NULL_HANDLE)
            Msg("![VK DLSS] exposure texture create failed — falling back to NGX defaults");
    }
    // Jitter phase count scales with the upscale ratio: 8*(display/render)^2 so the
    // sub-pixel sample set covers the larger output grid (DLAA → 8). Capped at 64.
    {
        const float ratio = (renderW > 0 && renderH > 0)
            ? (float(displayW) / float(renderW)) * (float(displayH) / float(renderH)) : 1.0f;
        u32 phases = (u32)(8.0f * ratio + 0.5f);
        s_jitterCount = (phases < 8u) ? 8u : (phases > 64u ? 64u : phases);
    }
    Msg("[VK DLSS] SR feature created: render %ux%u -> display %ux%u (%s), jitter phases %u, preset %u",
        renderW, renderH, displayW, displayH,
        (renderW == displayW) ? "DLAA" : "upscale", s_jitterCount, ps_r_dlss_preset);
    s_slShadow = false;   // raw feature is the owner now
    return true;
}

// SL route: sl.dlss owns the NGX feature — keep only the dim cache + jitter phase
// count here, and make sure no raw feature doubles the VRAM (see vk_dlss.h).
bool EnsureFeatureSL(u32 renderW, u32 renderH, u32 displayW, u32 displayH)
{
    if (!s_inited || !s_srAvail) return false;
    if (renderW == 0 || renderH == 0 || displayW == 0 || displayH == 0) return false;
    if (s_slShadow && renderW == s_fRenderW && renderH == s_fRenderH
        && displayW == s_fDispW && displayH == s_fDispH)
        return true;

    // A raw feature from before r_dlss_sl flipped on is pure duplication under SL.
    // Same idle rule as EnsureFeature: in-flight frames may still reference it.
    if (s_feature) { vkDeviceWaitIdle(VulkanHW.m_Device); ReleaseFeature(); }

    s_fRenderW = renderW; s_fRenderH = renderH; s_fDispW = displayW; s_fDispH = displayH;
    {
        const float ratio = (renderW > 0 && renderH > 0)
            ? (float(displayW) / float(renderW)) * (float(displayH) / float(renderH)) : 1.0f;
        u32 phases = (u32)(8.0f * ratio + 0.5f);
        s_jitterCount = (phases < 8u) ? 8u : (phases > 64u ? 64u : phases);
    }
    s_slShadow = true;
    Msg("[VK DLSS] SL feature shadow: render %ux%u -> display %ux%u (%s), jitter phases %u (NGX feature owned by sl.dlss)",
        renderW, renderH, displayW, displayH,
        (renderW == displayW) ? "DLAA" : "upscale", s_jitterCount);
    return true;
}

void ReleaseFeature()
{
    if (s_feature) { NVSDK_NGX_VULKAN_ReleaseFeature(s_feature); s_feature = nullptr; }
    s_slShadow = false;
    s_fRenderW = s_fRenderH = s_fDispW = s_fDispH = 0;
    s_fPreset  = 0xFFFFFFFFu;
    s_fInputsGen = 0xFFFFFFFFu;
}

void Evaluate(VkCommandBuffer cmd, const VK::Dlss::Img& color, const VK::Dlss::Img& depth,
              const VK::Dlss::Img& mv, u32 renderW, u32 renderH, bool reset)
{
    if (s_feature == nullptr || s_params == nullptr || s_outImg == VK_NULL_HANDLE) return;
    if (color.image == VK_NULL_HANDLE || depth.image == VK_NULL_HANDLE || mv.image == VK_NULL_HANDLE) return;

    // NGX wants its VK resources in GENERAL. Inputs come in SHADER_READ_ONLY (their
    // producing passes left them there); output goes UNDEFINED/whatever → GENERAL.
    ImageBarrier(cmd, color.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    ImageBarrier(cmd, mv.image,    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
    ImageBarrier(cmd, depth.image, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_DEPTH_BIT);
    // The output is fully (over)written by NGX every frame and the caller leaves it in
    // SHADER_READ after sampling it in the tonemap — so discard the previous contents
    // (UNDEFINED old layout) rather than tracking SHADER_READ vs GENERAL across frames.
    ImageBarrier(cmd, s_outImg, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    s_outFirst = false;

    NVSDK_NGX_Resource_VK colorR = WrapImage(color, renderW, renderH, VK_IMAGE_ASPECT_COLOR_BIT, false);
    NVSDK_NGX_Resource_VK depthR = WrapImage(depth, renderW, renderH, VK_IMAGE_ASPECT_DEPTH_BIT, false);
    NVSDK_NGX_Resource_VK mvR    = WrapImage(mv,    renderW, renderH, VK_IMAGE_ASPECT_COLOR_BIT, false);
    VK::Dlss::Img outImg{ s_outView, s_outImg, s_outFmt };
    NVSDK_NGX_Resource_VK outR   = WrapImage(outImg, s_outW, s_outH, VK_IMAGE_ASPECT_COLOR_BIT, true);

    // Real exposure (r_dlss_exp 1): clear the 1x1 texture to the tonemap's smoothed
    // exposure (published via SetExposureHint; one frame stale — it eases over
    // seconds, so that's noise-free) and hand it to NGX in GENERAL.
    NVSDK_NGX_Resource_VK expR{};
    const bool useExpTex = (s_fExpMode == 1u) && s_expImg != VK_NULL_HANDLE;
    if (useExpTex) {
        ImageBarrier(cmd, s_expImg, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkClearColorValue cv{}; cv.float32[0] = (s_expHint > 1e-6f) ? s_expHint : 1.0f;
        VkImageSubresourceRange rr = FullRange(VK_IMAGE_ASPECT_COLOR_BIT);
        vkCmdClearColorImage(cmd, s_expImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &cv, 1, &rr);
        ImageBarrier(cmd, s_expImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
        VK::Dlss::Img expImg{ s_expView, s_expImg, VK_FORMAT_R32_SFLOAT };
        expR = WrapImage(expImg, 1, 1, VK_IMAGE_ASPECT_COLOR_BIT, false);
    }

    NVSDK_NGX_VK_DLSS_Eval_Params ep{};
    ep.Feature.pInColor  = &colorR;
    ep.Feature.pInOutput = &outR;
    ep.Feature.InSharpness = 0.0f;
    ep.pInDepth          = &depthR;
    ep.pInMotionVectors  = &mvR;
    if (useExpTex) ep.pInExposureTexture = &expR;
    // Reported jitter sign (r_dlss_jitter_flip). The matrix jitter adds +jx/+jy in
    // CLIP space, but the scene rasterizes through a NEGATIVE-HEIGHT viewport (D3D
    // convention) while DLSS expects the offset in pixel space (y-down) — so the Y
    // sign we report is convention-dependent. The cvar flips what we TELL Evaluate
    // (bit0=X, bit1=Y) without touching the applied matrix, for a live A/B: the
    // wrong sign misaligns the temporal history by up to 1px/axis every frame and
    // the upscale never converges past bilinear.
    ep.InJitterOffsetX   = (ps_r_dlss_jitter_flip & 1) ? -s_jitterPixX : s_jitterPixX;
    ep.InJitterOffsetY   = (ps_r_dlss_jitter_flip & 2) ? -s_jitterPixY : s_jitterPixY;
    ep.InReset           = reset ? 1 : 0;
    ep.InRenderSubrectDimensions.Width  = renderW;
    ep.InRenderSubrectDimensions.Height = renderH;
    // Our MV is UV-space (prevUV − curUV); DLSS wants render-pixel space → scale by
    // render dims. Sign is the #1 thing to flip if the image ghosts/smears.
    ep.InMVScaleX = float(renderW);
    ep.InMVScaleY = float(renderH);

    NVSDK_NGX_Result r = NGX_VULKAN_EVALUATE_DLSS_EXT(cmd, s_feature, s_params, &ep);
    if (NVSDK_NGX_FAILED(r)) {
        static bool once = false;
        if (!once) { once = true; Msg("![VK DLSS] EvaluateFeature failed: 0x%08X", (unsigned)r); }
    }

    // Restore the inputs for the rest of the frame; output stays GENERAL (caller samples it).
    ImageBarrier(cmd, color.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    ImageBarrier(cmd, mv.image,    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    ImageBarrier(cmd, depth.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
}

void SetExposureHint(float exposure)
{
    s_expHint = exposure;
}

void Shutdown()
{
    if (!s_inited) return;
    ReleaseFeature();
    if (s_outView) { vkDestroyImageView(VulkanHW.m_Device, s_outView, nullptr); s_outView = VK_NULL_HANDLE; }
    if (s_outImg)  { VK::Vram::DestroyImage(VulkanHW.m_Allocator, s_outImg, s_outAlloc); s_outImg = VK_NULL_HANDLE; s_outAlloc = VK_NULL_HANDLE; }
    if (s_expView) { vkDestroyImageView(VulkanHW.m_Device, s_expView, nullptr); s_expView = VK_NULL_HANDLE; }
    if (s_expImg)  { VK::Vram::DestroyImage(VulkanHW.m_Allocator, s_expImg, s_expAlloc); s_expImg = VK_NULL_HANDLE; s_expAlloc = VK_NULL_HANDLE; }
    if (s_params)  { NVSDK_NGX_VULKAN_DestroyParameters(s_params); s_params = nullptr; }
    if (s_caps)    { NVSDK_NGX_VULKAN_DestroyParameters(s_caps); s_caps = nullptr; }
    NVSDK_NGX_VULKAN_Shutdown1(VulkanHW.m_Device);
    s_inited = s_srAvail = s_fgAvail = false;
    Msg("[VK DLSS] NGX shut down");
}

}}  // namespace VK::Dlss
