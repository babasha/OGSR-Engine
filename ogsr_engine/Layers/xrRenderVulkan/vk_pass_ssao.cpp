// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan — GTAO pass. See vk_pass_ssao.h.
#include "stdafx.h"
#include "vk_pass_ssao.h"
#include "vk_swapchain.h"          // Swapchain.m_DepthView (prepass depth)
#include "vk_shaders.h"            // g_ShaderManager
#include "vk_pipeline_cache.h"     // PipelineCache::GetCacheObject
#include "vk_barriers.h"           // ImageBarrier
#include "vk_buffer.h"             // CVulkanBuffer (debug AO readback)
#include "vk_command_buffer.h"     // CommandManager.GetCurrentFrame()
#include "vk_fullscreen.h"         // VK::Fullscreen — shared fullscreen pipeline + draw
#include "vk_motionvec.h"          // VK::MotionVec — temporal reprojection of the IL/AO history
#include "HW_Vulkan.h"
#include "../../xr_3da/device.h"   // Device camera basis + mProject

extern u32 ps_r_ao_quality;        // r2_ssao token (0 off / 1 low / 2 med / 3 high / 4 ultra)
extern int ps_r_ssao_enable;       // r_ssao — global GTAO on/off (GLOBAL scope: block-scope extern inside the namespace would mangle as VK::SSAOPass::* → LNK2001)
extern int ps_r_ssao_debug;        // r_ssao_debug — also enables the readback stats below
extern int   ps_r_ssil_enable;     // r_ssil — fold-in SSIL on/off (gates the prev-colour taps in the horizon march)
extern float ps_r_ssil_strength;   // r_ssil_strength — baked into the IL output (forward receivers apply a fixed ssilBoost)
extern float ps_r_ssil_temporal;   // r_ssil_temporal — GTAO temporal accumulation α (0 = off; per-frame jitter + MV-reprojected EMA)
extern float ps_r_ssao_temporal;   // r_ssao_temporal — same temporal accumulation as a first-class AO control (works without r_ssil)
extern float ps_r_ssao_bias;       // r_ssao_bias — grazing-surface horizon bias (rejects coplanar floor samples → kills flat-ground AO bands)

namespace VK {

ProjTerms DeriveProjTerms(const Fmatrix& M)
{
    ProjTerms t{};
    // w column (M_i4, i=1..3) = view dir scaled — pick the largest component's
    // row for the _33 division so a near-axis-aligned camera stays stable.
    const float w[3] = { M._14, M._24, M._34 };
    const float z[3] = { M._13, M._23, M._33 };
    int bi = 0;
    if (std::fabs(w[1]) > std::fabs(w[bi])) bi = 1;
    if (std::fabs(w[2]) > std::fabs(w[bi])) bi = 2;
    t.p33 = (std::fabs(w[bi]) > 1e-9f) ? z[bi] / w[bi] : 1.f;
    t.p43 = M._43 - M._44 * t.p33;
    const float p11 = std::sqrt(M._11 * M._11 + M._21 * M._21 + M._31 * M._31);
    const float p22 = std::sqrt(M._12 * M._12 + M._22 * M._22 + M._32 * M._32);
    t.tanX = (p11 > 1e-9f) ? 1.f / p11 : 1.f;
    t.tanY = (p22 > 1e-9f) ? 1.f / p22 : 1.f;
    t.dir.set(w[0], w[1], w[2]);          t.dir.normalize_safe();
    t.right.set(M._11, M._21, M._31);     t.right.normalize_safe();
    t.top.set(M._12, M._22, M._32);       t.top.normalize_safe();
    return t;
}

namespace SSAOPass {

namespace {
    constexpr u32 kFramesInFlight = CVulkanCommandManager::FRAMES_IN_FLIGHT;

    // R4 gtao.h: GTAO_RADIUS 4 (falloff range; effective horizon-march radius
    // is half that). Strength multiplies the occlusion in the receivers.
    constexpr float kRadius   = 4.0f;
    constexpr float kStrength = 1.0f;

    bool                  s_inited    = false;
    bool                  s_failed    = false;
    VkDescriptorSetLayout s_setLayout = VK_NULL_HANDLE;   // 0 = depth, 1 = raw AO (blur only)
    VkDescriptorPool      s_pool      = VK_NULL_HANDLE;
    VkDescriptorSet       s_setGtao[kFramesInFlight] = {};
    VkDescriptorSet       s_setBlur[kFramesInFlight] = {};
    VkPipelineLayout      s_layout    = VK_NULL_HANDLE;
    VkPipeline            s_pipeGtao  = VK_NULL_HANDLE;
    VkPipeline            s_pipeBlur  = VK_NULL_HANDLE;
    VkSampler             s_sampNear  = VK_NULL_HANDLE;   // nearest/clamp — depth + raw AO taps
    VkSampler             s_sampLin   = VK_NULL_HANDLE;   // linear/clamp — receivers upsample

    // Half-res ping-pong: [0] = final (blurred), [1] = raw GTAO.
    VkImage       s_img[2]   = {};
    VmaAllocation s_alloc[2] = {};
    VkImageView   s_view[2]  = {};

    // SSIL (folded into the GTAO horizon march, MRT target 1). Same half-res
    // ping-pong as AO ([0] = final/blurred IL, [1] = raw IL). RGBA16F (rgb = HDR
    // indirect radiance). Plus a persistent half-res PREV-frame colour buffer the
    // horizon gather samples (captured from the composited scene each frame).
    constexpr VkFormat kILFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    constexpr float    kILFireClamp = 6.0f;   // per-sample radiance clamp → no single specular pixel dominates
    VkImage       s_ilImg[2]   = {};
    VmaAllocation s_ilAlloc[2] = {};
    VkImageView   s_ilView[2]  = {};
    VkImage       s_prevColor      = VK_NULL_HANDLE;   // half-res linear-HDR history (prev frame's lit scene)
    VmaAllocation s_prevColorAlloc = VK_NULL_HANDLE;
    VkImageView   s_prevColorView  = VK_NULL_HANDLE;
    bool          s_prevColorCleared = false;          // false → first Execute clears it to black (valid SHADER_READ)

    // TEMPORAL (r_ssil_temporal): persistent half-res copies of the PREVIOUS frame's
    // FINAL AO and IL. The blur reprojects them through the motion vectors and EMA-
    // blends — the per-frame-jittered gather averages into a band-free result. AO
    // history is RGBA16F (matches s_img[0]); IL history is kILFormat (s_ilImg[0]).
    VkImage       s_aoHist      = VK_NULL_HANDLE;
    VmaAllocation s_aoHistAlloc = VK_NULL_HANDLE;
    VkImageView   s_aoHistView  = VK_NULL_HANDLE;
    VkImage       s_ilHist      = VK_NULL_HANDLE;
    VmaAllocation s_ilHistAlloc = VK_NULL_HANDLE;
    VkImageView   s_ilHistView  = VK_NULL_HANDLE;
    bool          s_histCleared = false;   // false → first Execute clears both history buffers to a valid SHADER_READ black
    bool          s_histValid   = false;   // a fresh final result was copied in last frame → safe to reproject
    bool          s_temporalWasOn = false; // tracks the cvar edge so off→on starts clean (no stale-history blend)
    u32           s_frame       = 0;       // monotonic frame counter → per-frame jitter phase
    u32           s_mvWarm      = 0;       // frames MV has been enabled+present → only reproject once it has surely rendered (SHADER_READ)

    // FULL-res NPC normal G-buffer (skinned pass writes worldN*0.5+0.5, a=valid;
    // GTAO uses it where a=1, else falls back to depth-derived normals). RGBA8.
    constexpr VkFormat kNormalFormat = VK_FORMAT_R8G8B8A8_UNORM;
    VkImage       s_normImg   = VK_NULL_HANDLE;
    VmaAllocation s_normAlloc  = VK_NULL_HANDLE;
    VkImageView   s_normView   = VK_NULL_HANDLE;
    VkExtent2D    s_normExtent  = {};
    VkExtent2D    s_extent   = {};
    u32           s_generation = 0;
    bool          s_first[2] = { true, true };

    // Debug readback (active while r_ssao_debug != 0): every ~300 frames the
    // final AO image is copied to this host buffer and min/max/avg logged —
    // verifies the GTAO output end-to-end without a GPU capture tool.
    CVulkanBuffer s_dbgBuf;
    u16*          s_dbgMap = nullptr;     // R16_SFLOAT texels (half-float)
    int           s_dbgCountdown = -1;   // frames until the mapped copy is safely written
    u32           s_dbgCooldown  = 0;

    // Decode an IEEE-754 half (one R16_SFLOAT AO texel) onto the legacy 0..255
    // scale so the debug log reads the same as it did under R8_UNORM.
    u8 AoHalfToU8(u16 h)
    {
        const u32 exp = (h >> 10) & 0x1F;
        const u32 man = h & 0x3FF;
        float v;
        if (exp == 0)         v = man * (1.0f / 16777216.0f);             // subnormal ~0
        else if (exp == 0x1F) v = 1.0f;                                   // inf/nan → clamp
        else                  v = ldexpf(1.0f + man / 1024.0f, int(exp) - 15);
        if (h & 0x8000) v = 0.0f;                                         // AO is never negative
        if (v < 0.0f) v = 0.0f; else if (v > 1.0f) v = 1.0f;
        return u8(v * 255.0f + 0.5f);
    }

    struct SSAOPush {
        float camDir[4];
        float camRightT[4];   // xyz = right * tan(fovX/2), w = tan(fovX/2)
        float camTopT[4];     // xyz = top   * tan(fovY/2), w = tan(fovY/2)
        float zp[4];          // proj _33, proj _43, radius, samples/side
        float res[4];         // AO size xy, 1/AO size zw
        float dbg[4];         // x = r_ssao_debug mode (2 = depth, 3 = normal); y = SSIL on; z = SSIL firefly clamp; w = SSIL strength
        float temporal[4];    // x = EMA α (0 = off); y = per-frame jitter phase [0,1); z = MV valid; w = history valid
    };
    static_assert(sizeof(SSAOPush) == 112, "must match ssao.frag / ssao_blur.frag PC blocks");

    u32 SampleCount()
    {
        // r2_ssao token → GTAO_SAMPLE (R4: low 2 / medium 3 / high+ 4). The
        // user's carried-over user.ltx may have it off — the Vulkan renderer
        // defaults to medium then (set r2_ssao to override; st_opt_off is
        // honoured once any other value was chosen at least once... for now
        // 0 == "unset" → medium, since legacy configs predate this pass).
        // Unset (legacy user.ltx carries r2_ssao off = 0) now defaults to LOW = 2
        // samples (was medium = 3): with temporal accumulation default ON
        // (r_ssao_temporal) 2 samples is band-free + measured ~33% cheaper than 4
        // (RTX 5070: SSAO 2.04ms→1.36ms; the real win is the bandwidth-bound iGPU).
        // Explicit r2_ssao st_opt_med/high still forces 3/4 for users who want it.
        const u32 q = ps_r_ao_quality ? ps_r_ao_quality : 1;
        return q <= 1 ? 2 : (q == 2 ? 3 : 4);
    }

    void DestroyRTs()
    {
        if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
        for (u32 i = 0; i < 2; ++i) {
            if (s_view[i]) { vkDestroyImageView(VulkanHW.m_Device, s_view[i], nullptr); s_view[i] = VK_NULL_HANDLE; }
            if (s_img[i])  { vmaDestroyImage(VulkanHW.m_Allocator, s_img[i], s_alloc[i]); s_img[i] = VK_NULL_HANDLE; s_alloc[i] = VK_NULL_HANDLE; }
            s_first[i] = true;
        }
        if (s_normView) { vkDestroyImageView(VulkanHW.m_Device, s_normView, nullptr); s_normView = VK_NULL_HANDLE; }
        if (s_normImg)  { vmaDestroyImage(VulkanHW.m_Allocator, s_normImg, s_normAlloc); s_normImg = VK_NULL_HANDLE; s_normAlloc = VK_NULL_HANDLE; }
        for (u32 i = 0; i < 2; ++i) {
            if (s_ilView[i]) { vkDestroyImageView(VulkanHW.m_Device, s_ilView[i], nullptr); s_ilView[i] = VK_NULL_HANDLE; }
            if (s_ilImg[i])  { vmaDestroyImage(VulkanHW.m_Allocator, s_ilImg[i], s_ilAlloc[i]); s_ilImg[i] = VK_NULL_HANDLE; s_ilAlloc[i] = VK_NULL_HANDLE; }
        }
        if (s_prevColorView) { vkDestroyImageView(VulkanHW.m_Device, s_prevColorView, nullptr); s_prevColorView = VK_NULL_HANDLE; }
        if (s_prevColor)     { vmaDestroyImage(VulkanHW.m_Allocator, s_prevColor, s_prevColorAlloc); s_prevColor = VK_NULL_HANDLE; s_prevColorAlloc = VK_NULL_HANDLE; }
        s_prevColorCleared = false;
        if (s_aoHistView) { vkDestroyImageView(VulkanHW.m_Device, s_aoHistView, nullptr); s_aoHistView = VK_NULL_HANDLE; }
        if (s_aoHist)     { vmaDestroyImage(VulkanHW.m_Allocator, s_aoHist, s_aoHistAlloc); s_aoHist = VK_NULL_HANDLE; s_aoHistAlloc = VK_NULL_HANDLE; }
        if (s_ilHistView) { vkDestroyImageView(VulkanHW.m_Device, s_ilHistView, nullptr); s_ilHistView = VK_NULL_HANDLE; }
        if (s_ilHist)     { vmaDestroyImage(VulkanHW.m_Allocator, s_ilHist, s_ilHistAlloc); s_ilHist = VK_NULL_HANDLE; s_ilHistAlloc = VK_NULL_HANDLE; }
        s_histCleared = false; s_histValid = false;
        s_normExtent = {};
        s_extent = {};
    }

    bool EnsureRTs(VkExtent2D sceneExtent)
    {
        VkExtent2D want{ sceneExtent.width  / 2 ? sceneExtent.width  / 2 : 1,
                         sceneExtent.height / 2 ? sceneExtent.height / 2 : 1 };
        if (s_img[0] && want.width == s_extent.width && want.height == s_extent.height)
            return true;
        DestroyRTs();
        s_extent = want;
        for (u32 i = 0; i < 2; ++i) {
            VkImageCreateInfo ici{};
            ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            ici.imageType = VK_IMAGE_TYPE_2D;
            ici.format = VK_FORMAT_R16G16B16A16_SFLOAT;   // r=AO (R16F kills contouring) + gba=world bent normal
            ici.extent = { want.width, want.height, 1 };
            ici.mipLevels = 1; ici.arrayLayers = 1;
            ici.samples = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling = VK_IMAGE_TILING_OPTIMAL;
            ici.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                      | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;   // debug readback
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            if (vmaCreateImage(VulkanHW.m_Allocator, &ici, &aci, &s_img[i], &s_alloc[i], nullptr) != VK_SUCCESS) {
                Msg("![VK SSAO] RT %u create failed", i); return false;
            }
            VkImageViewCreateInfo vci{};
            vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vci.image = s_img[i];
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format = VK_FORMAT_R16G16B16A16_SFLOAT;
            vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            vci.subresourceRange.levelCount = 1;
            vci.subresourceRange.layerCount = 1;
            if (vkCreateImageView(VulkanHW.m_Device, &vci, nullptr, &s_view[i]) != VK_SUCCESS) {
                Msg("![VK SSAO] view %u create failed", i); return false;
            }
        }
        // SSIL ping-pong (half-res RGBA16F, MRT target 1 alongside AO). [0]=final, [1]=raw.
        for (u32 i = 0; i < 2; ++i) {
            VkImageCreateInfo ici{};
            ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            ici.imageType = VK_IMAGE_TYPE_2D;
            ici.format = kILFormat;
            ici.extent = { want.width, want.height, 1 };
            ici.mipLevels = 1; ici.arrayLayers = 1;
            ici.samples = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling = VK_IMAGE_TILING_OPTIMAL;
            ici.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                      | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;   // [0] copied into the temporal IL history
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            if (vmaCreateImage(VulkanHW.m_Allocator, &ici, &aci, &s_ilImg[i], &s_ilAlloc[i], nullptr) != VK_SUCCESS) {
                Msg("![VK SSAO] IL RT %u create failed", i); return false;
            }
            VkImageViewCreateInfo vci{};
            vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vci.image = s_ilImg[i];
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format = kILFormat;
            vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            vci.subresourceRange.levelCount = 1;
            vci.subresourceRange.layerCount = 1;
            if (vkCreateImageView(VulkanHW.m_Device, &vci, nullptr, &s_ilView[i]) != VK_SUCCESS) {
                Msg("![VK SSAO] IL view %u create failed", i); return false;
            }
        }
        // Persistent half-res PREV-frame colour (linear HDR) — the IL gather source.
        // blit dst (capture) + sampled (gather). Cleared to black on first Execute.
        {
            VkImageCreateInfo ici{};
            ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            ici.imageType = VK_IMAGE_TYPE_2D;
            ici.format = kILFormat;
            ici.extent = { want.width, want.height, 1 };
            ici.mipLevels = 1; ici.arrayLayers = 1;
            ici.samples = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling = VK_IMAGE_TILING_OPTIMAL;
            ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            if (vmaCreateImage(VulkanHW.m_Allocator, &ici, &aci, &s_prevColor, &s_prevColorAlloc, nullptr) != VK_SUCCESS) {
                Msg("![VK SSAO] prevColor create failed"); return false;
            }
            VkImageViewCreateInfo vci{};
            vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vci.image = s_prevColor;
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format = kILFormat;
            vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            vci.subresourceRange.levelCount = 1;
            vci.subresourceRange.layerCount = 1;
            if (vkCreateImageView(VulkanHW.m_Device, &vci, nullptr, &s_prevColorView) != VK_SUCCESS) {
                Msg("![VK SSAO] prevColor view create failed"); return false;
            }
            s_prevColorCleared = false;
        }
        // Temporal history pair (half-res): AO history = RGBA16F (matches s_img[0]),
        // IL history = kILFormat (matches s_ilImg[0]). copy dst (capture the final
        // result) + sampled (reproject next frame). Cleared to black on first Execute.
        {
            struct { VkImage* img; VmaAllocation* alloc; VkImageView* view; VkFormat fmt; } hist[2] = {
                { &s_aoHist, &s_aoHistAlloc, &s_aoHistView, VK_FORMAT_R16G16B16A16_SFLOAT },
                { &s_ilHist, &s_ilHistAlloc, &s_ilHistView, kILFormat },
            };
            for (auto& h : hist) {
                VkImageCreateInfo ici{};
                ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
                ici.imageType = VK_IMAGE_TYPE_2D;
                ici.format = h.fmt;
                ici.extent = { want.width, want.height, 1 };
                ici.mipLevels = 1; ici.arrayLayers = 1;
                ici.samples = VK_SAMPLE_COUNT_1_BIT;
                ici.tiling = VK_IMAGE_TILING_OPTIMAL;
                ici.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
                ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                VmaAllocationCreateInfo aci{};
                aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
                if (vmaCreateImage(VulkanHW.m_Allocator, &ici, &aci, h.img, h.alloc, nullptr) != VK_SUCCESS) {
                    Msg("![VK SSAO] temporal history create failed"); return false;
                }
                VkImageViewCreateInfo vci{};
                vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                vci.image = *h.img;
                vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
                vci.format = h.fmt;
                vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                vci.subresourceRange.levelCount = 1;
                vci.subresourceRange.layerCount = 1;
                if (vkCreateImageView(VulkanHW.m_Device, &vci, nullptr, h.view) != VK_SUCCESS) {
                    Msg("![VK SSAO] temporal history view create failed"); return false;
                }
            }
            s_histCleared = false; s_histValid = false;
        }
        // FULL-res NPC normal G-buffer (skinned pass renders into it, GTAO samples it).
        s_normExtent = sceneExtent;
        {
            VkImageCreateInfo ici{};
            ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            ici.imageType = VK_IMAGE_TYPE_2D;
            ici.format = kNormalFormat;
            ici.extent = { sceneExtent.width, sceneExtent.height, 1 };
            ici.mipLevels = 1; ici.arrayLayers = 1;
            ici.samples = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling = VK_IMAGE_TILING_OPTIMAL;
            ici.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            if (vmaCreateImage(VulkanHW.m_Allocator, &ici, &aci, &s_normImg, &s_normAlloc, nullptr) != VK_SUCCESS) {
                Msg("![VK SSAO] normal RT create failed"); return false;
            }
            VkImageViewCreateInfo vci{};
            vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vci.image = s_normImg;
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format = kNormalFormat;
            vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            vci.subresourceRange.levelCount = 1;
            vci.subresourceRange.layerCount = 1;
            if (vkCreateImageView(VulkanHW.m_Device, &vci, nullptr, &s_normView) != VK_SUCCESS) {
                Msg("![VK SSAO] normal view create failed"); return false;
            }
        }

        // Host buffer for the debug readback (RGBA16F = 4 halfs = 8 bytes per AO texel; R = AO).
        s_dbgBuf.Destroy();
        s_dbgMap = nullptr;
        s_dbgCountdown = -1;
        s_dbgBuf.Create(VkDeviceSize(want.width) * want.height * 4 * sizeof(u16),
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
        s_dbgMap = static_cast<u16*>(s_dbgBuf.Map());

        ++s_generation;
        Msg("[VK SSAO] RTs ready (%ux%u half-res, gen %u, samples %u)",
            want.width, want.height, s_generation, SampleCount());
        return true;
    }

    // MRT: target 0 = AO+bentN (RGBA16F), target 1 = SSIL (RGBA16F). Both opaque,
    // all channels, no blend. gtao + blur share this two-attachment layout.
    VkPipeline CreatePipe(VkShaderModule vs, VkShaderModule fs)
    {
        const VkFormat fmts[2] = { VK_FORMAT_R16G16B16A16_SFLOAT, kILFormat };
        const VkPipelineColorBlendAttachmentState blends[2] = {
            Fullscreen::OpaqueAttachment(), Fullscreen::OpaqueAttachment() };
        return Fullscreen::CreatePipelineMRT(vs, fs, fmts, 2, s_layout, blends, "SSAO");
    }

    // One MRT fullscreen draw into (AO, IL) at the half-res extent.
    void Draw(VkCommandBuffer cmd, VkImageView aoDst, VkImageView ilDst, VkPipeline pipe,
              VkDescriptorSet set, const SSAOPush& push)
    {
        const VkImageView views[2] = { aoDst, ilDst };
        Fullscreen::DrawMRT(cmd, views, 2, s_extent, pipe, s_layout, set, &push, sizeof(push));
    }
}  // anon namespace

// Set once the first Execute actually rendered AO — until then (and whenever
// the prepass is unavailable) Strength() keeps the receivers on plain 1.0.
static bool s_hasResult = false;

bool        Enabled()       { return s_inited && !s_failed && ps_r_ssao_enable != 0; }
VkImageView GetResultView() { return Enabled() ? s_view[0] : VK_NULL_HANDLE; }  // disabled → EnvLight uses white fallback
VkSampler   GetSampler()    { return s_sampLin; }
u32         Generation()    { return s_generation; }
float       Strength()      { return (Enabled() && s_hasResult && s_view[0]) ? kStrength : 0.f; }

// SSIL result (MRT target 1). Folded into the GTAO march, so it requires the
// GTAO pass to be running (r_ssao on) AND r_ssil on. Null otherwise → tonemap
// binds its fallback and skips the IL composite.
VkImageView GetILResultView() { return (Enabled() && s_hasResult && ps_r_ssil_enable != 0 && s_ilView[0]) ? s_ilView[0] : VK_NULL_HANDLE; }

// NPC normal G-buffer: the skinned pass renders into it (after the prepass,
// before Execute); GTAO samples it (binding 2). VK_NULL_HANDLE until EnsureRTs.
VkImage     GetNormalImage()  { return s_normImg; }
VkImageView GetNormalView()   { return s_normView; }
VkFormat    GetNormalFormat() { return kNormalFormat; }
VkExtent2D  GetNormalExtent() { return s_normExtent; }

bool EnsureTargets(VkExtent2D sceneExtent)
{
    if (!Enabled()) return false;
    return EnsureRTs(sceneExtent);
}

bool Init()
{
    if (s_inited) return !s_failed;
    s_inited = true;

    if (!g_ShaderManager) { s_failed = true; return false; }
    VkShaderModule vs   = g_ShaderManager->Load("tonemap.vert.spv");   // shared fullscreen triangle
    VkShaderModule gtao = g_ShaderManager->Load("ssao.frag.spv");
    VkShaderModule blur = g_ShaderManager->Load("ssao_blur.frag.spv");
    if (vs == VK_NULL_HANDLE || gtao == VK_NULL_HANDLE || blur == VK_NULL_HANDLE) {
        Msg("![VK SSAO] ssao.{frag,blur}.spv missing — SSAO disabled");
        s_failed = true; return false;
    }

    // Set: 0 = scene depth, 1 = raw AO (blur only), 2 = NPC normal G-buffer
    // (GTAO only), 3 = prev-frame colour (GTAO+IL gather), 4 = raw IL (blur only),
    // 5 = motion vectors, 6 = AO history, 7 = IL history (blur temporal only).
    // Each pipeline statically uses a subset; the others' stale layout during a
    // draw is legal (all bindings are written each frame).
    constexpr u32 kBindings = 8;
    VkDescriptorSetLayoutBinding b[kBindings]{};
    for (u32 i = 0; i < kBindings; ++i) {
        b[i].binding = i; b[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo slci{};
    slci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    slci.bindingCount = kBindings; slci.pBindings = b;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &slci, nullptr, &s_setLayout) != VK_SUCCESS) {
        s_failed = true; return false;
    }

    const u32 nSets = kFramesInFlight * 2;
    VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, nSets * kBindings };
    VkDescriptorPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = nSets; pci.poolSizeCount = 1; pci.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &s_pool) != VK_SUCCESS) {
        s_failed = true; return false;
    }
    {
        VkDescriptorSetLayout layouts[kFramesInFlight * 2];
        for (u32 i = 0; i < nSets; ++i) layouts[i] = s_setLayout;
        VkDescriptorSet sets[kFramesInFlight * 2]{};
        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = s_pool; dai.descriptorSetCount = nSets; dai.pSetLayouts = layouts;
        if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, sets) != VK_SUCCESS) {
            s_failed = true; return false;
        }
        for (u32 i = 0; i < kFramesInFlight; ++i) {
            s_setGtao[i] = sets[i];
            s_setBlur[i] = sets[kFramesInFlight + i];
        }
    }

    auto makeSampler = [&](VkFilter f, VkSampler& out) {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = f; si.minFilter = f;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxAnisotropy = 1.0f;
        return vkCreateSampler(VulkanHW.m_Device, &si, nullptr, &out) == VK_SUCCESS;
    };
    if (!makeSampler(VK_FILTER_NEAREST, s_sampNear) || !makeSampler(VK_FILTER_LINEAR, s_sampLin)) {
        Msg("![VK SSAO] sampler create failed"); s_failed = true; return false;
    }

    VkPushConstantRange pcr{ VK_SHADER_STAGE_FRAGMENT_BIT, 0, (u32)sizeof(SSAOPush) };
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1; plci.pSetLayouts = &s_setLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_layout) != VK_SUCCESS) {
        s_failed = true; return false;
    }

    s_pipeGtao = CreatePipe(vs, gtao);
    s_pipeBlur = CreatePipe(vs, blur);
    if (s_pipeGtao == VK_NULL_HANDLE || s_pipeBlur == VK_NULL_HANDLE) { s_failed = true; return false; }

    Msg("[VK SSAO] init OK (GTAO half-res, radius %.1f, r2_ssao quality %u)", kRadius, ps_r_ao_quality);
    return true;
}

void Execute(VkCommandBuffer cmd, VkExtent2D sceneExtent)
{
    if (!Enabled()) return;
    if (!EnsureRTs(sceneExtent)) return;

    // Temporal accumulation state (r_ssil_temporal). Reproject the previous frame's
    // FINAL AO/IL through the motion vectors and EMA-blend; the per-frame-jittered
    // gather (ssao.frag) makes each history a different realisation → averaging
    // removes the directional banding. Only reproject with CONTINUOUS history (it was
    // refreshed last frame) so an off→on toggle or a resize starts clean, and only
    // once the MV target has surely rendered (else it could be UNDEFINED layout).
    ++s_frame;
    // Temporal is now driven by EITHER r_ssao_temporal (AO, standalone) OR r_ssil_temporal
    // (legacy SSIL path). AO and IL share one per-frame jitter + one history pair, so the
    // blur EMA uses the stronger of the two α's.
    const float temporalAlpha = std::max(ps_r_ssao_temporal, ps_r_ssil_temporal);
    const bool temporalOn = (temporalAlpha > 0.0f);
    const bool histUsable = temporalOn && s_temporalWasOn && s_histValid;
    if (MotionVec::Enabled() && MotionVec::GetResultView() != VK_NULL_HANDLE) { if (s_mvWarm < 4) ++s_mvWarm; }
    else s_mvWarm = 0;
    const bool  mvReady = histUsable && (s_mvWarm >= 2);
    VkImageView mvBind  = mvReady ? MotionVec::GetResultView() : s_prevColorView;  // placeholder is valid SHADER_READ, never sampled when !mvReady
    float jitterPhase = float(s_frame) * 0.61803399f; jitterPhase -= std::floor(jitterPhase);

    // Refresh this slot's sets (fence-guarded): depth view can change on swapchain
    // recreate, AO/IL/prevColor/history views on RT recreate. 8 bindings × 2 sets/frame.
    const u32 slot = CommandManager.GetCurrentFrame() % kFramesInFlight;
    {
        VkDescriptorImageInfo depthI { s_sampNear, Swapchain.m_DepthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo rawI   { s_sampNear, s_view[1],             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo normI  { s_sampNear, s_normView,            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo prevI  { s_sampLin,  s_prevColorView,       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo ilrawI { s_sampNear, s_ilView[1],           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo mvI    { s_sampLin,  mvBind,                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo aoHistI{ s_sampLin,  s_aoHistView,          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo ilHistI{ s_sampLin,  s_ilHistView,          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        // gtao uses 0,2,3; blur uses 0,1,4,5,6,7 — write all 8 to both sets so none is stale.
        const VkDescriptorImageInfo* src[8] = { &depthI, &rawI, &normI, &prevI, &ilrawI, &mvI, &aoHistI, &ilHistI };
        VkWriteDescriptorSet w[16]{};
        for (u32 i = 0; i < 16; ++i) {
            const u32 bi = i % 8;
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = (i < 8) ? s_setGtao[slot] : s_setBlur[slot];
            w[i].dstBinding = bi; w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[i].pImageInfo = src[bi];
        }
        vkUpdateDescriptorSets(VulkanHW.m_Device, 16, w, 0, nullptr);
    }

    // First use after (re)create: clear the prev-colour history to black so the
    // horizon gather reads valid (zero-IL) data until the first capture lands.
    if (!s_prevColorCleared) {
        ImageBarrier(cmd, s_prevColor, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkClearColorValue black{};
        VkImageSubresourceRange range{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdClearColorImage(cmd, s_prevColor, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);
        ImageBarrier(cmd, s_prevColor, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        s_prevColorCleared = true;
    }

    // First use: clear the temporal history pair to black so they're valid
    // SHADER_READ when the blur set binds them (sampled only when history is valid,
    // but the descriptor references them every frame). s_histValid stays false until
    // a real final result is copied in below.
    if (!s_histCleared) {
        VkClearColorValue black{};
        VkImageSubresourceRange range{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        for (VkImage h : { s_aoHist, s_ilHist }) {
            ImageBarrier(cmd, h, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            vkCmdClearColorImage(cmd, h, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);
            ImageBarrier(cmd, h, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        s_histCleared = true;
    }

    // Transition the AO + IL MRT pair for slot i together (one draw writes both).
    auto toColor = [&](u32 i) {
        const VkImageLayout old = s_first[i] ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        ImageBarrier(cmd, s_img[i],   old, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        ImageBarrier(cmd, s_ilImg[i], old, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        s_first[i] = false;
    };
    auto toRead = [&](u32 i) {
        ImageBarrier(cmd, s_img[i],   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        ImageBarrier(cmd, s_ilImg[i], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    };

    // Everything derived from the matrix that RENDERED the prepass depth —
    // Device.mProject is not maintained on this path (reads as identity).
    const ProjTerms pt = DeriveProjTerms(Device.mFullTransform);
    SSAOPush push{};
    push.camDir[0] = pt.dir.x; push.camDir[1] = pt.dir.y; push.camDir[2] = pt.dir.z;
    push.camDir[3] = ps_r_ssao_bias;   // grazing-surface horizon bias (ssao.frag rejects coplanar samples)
    push.camRightT[0] = pt.right.x * pt.tanX; push.camRightT[1] = pt.right.y * pt.tanX;
    push.camRightT[2] = pt.right.z * pt.tanX; push.camRightT[3] = pt.tanX;
    push.camTopT[0]   = pt.top.x  * pt.tanY;  push.camTopT[1]   = pt.top.y  * pt.tanY;
    push.camTopT[2]   = pt.top.z  * pt.tanY;  push.camTopT[3]   = pt.tanY;
    push.zp[0] = pt.p33;
    push.zp[1] = pt.p43;
    push.zp[2] = kRadius;
    push.zp[3] = float(SampleCount());
    push.res[0] = float(s_extent.width);
    push.res[1] = float(s_extent.height);
    push.res[2] = 1.f / float(s_extent.width);
    push.res[3] = 1.f / float(s_extent.height);
    push.dbg[0] = float(ps_r_ssao_debug);
    push.dbg[1] = ps_r_ssil_enable ? 1.0f : 0.0f;   // SSIL: gather prev-frame colour in the horizon march
    push.dbg[2] = kILFireClamp;
    push.dbg[3] = ps_r_ssil_enable ? ps_r_ssil_strength : 0.0f;   // baked into IL → forward ssilBoost; 0 when off
    push.temporal[0] = temporalOn ? temporalAlpha : 0.0f;          // EMA α (gather: gates jitter; blur: history weight)
    push.temporal[1] = temporalOn ? jitterPhase : 0.0f;            // per-frame slice/radial rotation (0 → spatial path)
    push.temporal[2] = mvReady   ? 1.0f : 0.0f;                    // reproject via MV (else same-pixel EMA)
    push.temporal[3] = histUsable ? 1.0f : 0.0f;                   // blur may sample the history this frame

    // One-time dump of the reconstruction inputs — sanity vs the offline test
    // (expect _33 ≈ 1.0006, _43 ≈ -0.2 for zn 0.2 / zf 350, tan ≈ 0.6-1.1).
    static bool s_pushDiag = false;
    if (!s_pushDiag) { s_pushDiag = true;
        Msg("[VK SSAO] push (derived): _33=%.6f _43=%.6f tanX=%.4f tanY=%.4f dir=(%.2f,%.2f,%.2f)",
            pt.p33, pt.p43, pt.tanX, pt.tanY, pt.dir.x, pt.dir.y, pt.dir.z);
    }

    // 1) GTAO+IL: depth + prev-colour → raw AO [1] + raw IL [1].
    toColor(1);
    Draw(cmd, s_view[1], s_ilView[1], s_pipeGtao, s_setGtao[slot], push);
    toRead(1);

    // 2) Depth-aware 3×3 blur: raw [1] → final [0] (both AO and IL). When temporal
    // is on this draw also EMA-blends the reprojected history bound above.
    toColor(0);
    Draw(cmd, s_view[0], s_ilView[0], s_pipeBlur, s_setBlur[slot], push);
    toRead(0);

    // 3) Capture this frame's FINAL AO + IL as next frame's temporal history. Single-
    // buffered like prevColor: the dst→SHADER_READ barrier orders the next frame's
    // reprojection read after this copy (same queue, submission order). Only while
    // temporal is on — a gap invalidates the history so it isn't blended stale.
    if (temporalOn) {
        auto copyHist = [&](VkImage src, VkImage dst) {
            ImageBarrier(cmd, src, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            ImageBarrier(cmd, dst, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkImageCopy region{};
            region.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            region.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            region.extent = { s_extent.width, s_extent.height, 1 };
            vkCmdCopyImage(cmd, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            ImageBarrier(cmd, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            ImageBarrier(cmd, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        };
        copyHist(s_img[0],   s_aoHist);
        copyHist(s_ilImg[0], s_ilHist);
        s_histValid = true;
    } else {
        s_histValid = false;   // stale frames must not be reprojected after a gap
    }
    s_temporalWasOn = temporalOn;

    s_hasResult = true;

    // Debug readback (r_ssao_debug): periodically copy the final AO to the host
    // buffer; a few frames later (fence-safe by frames-in-flight) log stats.
    if (ps_r_ssao_debug && s_dbgMap) {
        if (s_dbgCountdown > 0 && --s_dbgCountdown == 0) {
            const u32 n = s_extent.width * s_extent.height;
            u32 mn = 255, mx = 0; u64 sum = 0; u32 below200 = 0;
            for (u32 i = 0; i < n; ++i) {
                const u8 v = AoHalfToU8(s_dbgMap[i * 4]);   // R = AO (4 halfs/texel, gba = bent normal)
                mn = std::min(mn, (u32)v); mx = std::max(mx, (u32)v);
                sum += v; below200 += (v < 200);
            }
            Msg("[VK SSAO] readback mode %d %ux%u: min %u max %u avg %.1f, <200: %.1f%%",
                ps_r_ssao_debug, s_extent.width, s_extent.height, mn, mx,
                double(sum) / n, 100.0 * below200 / n);
            for (u32 r = 1; r <= 3; ++r) {   // rows at 25/50/75% height, 5 taps each
                const u32 cy = s_extent.height * r / 4;
                Msg("[VK SSAO]   row %u%%: %u %u %u %u %u", r * 25,
                    AoHalfToU8(s_dbgMap[(cy * s_extent.width + s_extent.width / 6)     * 4]),
                    AoHalfToU8(s_dbgMap[(cy * s_extent.width + s_extent.width / 3)     * 4]),
                    AoHalfToU8(s_dbgMap[(cy * s_extent.width + s_extent.width / 2)     * 4]),
                    AoHalfToU8(s_dbgMap[(cy * s_extent.width + s_extent.width * 2 / 3) * 4]),
                    AoHalfToU8(s_dbgMap[(cy * s_extent.width + s_extent.width * 5 / 6) * 4]));
            }
        }
        if (s_dbgCooldown == 0 && s_dbgCountdown <= 0) {
            ImageBarrier(cmd, s_img[0], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            VkBufferImageCopy bic{};
            bic.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            bic.imageExtent      = { s_extent.width, s_extent.height, 1 };
            vkCmdCopyImageToBuffer(cmd, s_img[0], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   s_dbgBuf.GetHandle(), 1, &bic);
            ImageBarrier(cmd, s_img[0], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            s_dbgCountdown = 5;     // read after the in-flight fences cycled
            s_dbgCooldown  = 300;   // ~one log line per few seconds
        }
        if (s_dbgCooldown) --s_dbgCooldown;
    }
}

void CapturePrevColor(VkCommandBuffer cmd, VkImage srcImage, VkExtent2D srcExtent)
{
    // Snapshot THIS frame's composited HDR scene (half-res, linear) into the
    // history the NEXT frame's GTAO+IL gather reads. No-op unless the GTAO pass
    // runs AND r_ssil is on (else there's no IL to feed). prevColor is SHADER_READ
    // here (cleared on the first Execute, left SHADER_READ by the previous capture).
    if (!Enabled() || ps_r_ssil_enable == 0 || s_prevColor == VK_NULL_HANDLE
        || srcImage == VK_NULL_HANDLE || !s_prevColorCleared)
        return;

    // src mip0 SHADER_READ → TRANSFER_SRC (ImageBarrier touches mip0 only, leaving
    // the rest of the HDR mip chain SHADER_READ for the tonemap that follows).
    ImageBarrier(cmd, srcImage,    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    ImageBarrier(cmd, s_prevColor, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    VkImageBlit blit{};
    blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    blit.srcOffsets[1]  = { int32_t(srcExtent.width), int32_t(srcExtent.height), 1 };
    blit.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    blit.dstOffsets[1]  = { int32_t(s_extent.width), int32_t(s_extent.height), 1 };
    vkCmdBlitImage(cmd, srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   s_prevColor, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);

    // src back to SHADER_READ (tonemap samples it); dst to SHADER_READ — this
    // barrier also synchronizes the next frame's gather read (same queue, in
    // submission order), so the single history buffer needs no ping-pong.
    ImageBarrier(cmd, srcImage,    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    ImageBarrier(cmd, s_prevColor, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void Destroy()
{
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
    DestroyRTs();
    s_dbgBuf.Destroy(); s_dbgMap = nullptr; s_dbgCountdown = -1; s_dbgCooldown = 0;
    if (s_pipeGtao)  { vkDestroyPipeline(VulkanHW.m_Device, s_pipeGtao, nullptr); s_pipeGtao = VK_NULL_HANDLE; }
    if (s_pipeBlur)  { vkDestroyPipeline(VulkanHW.m_Device, s_pipeBlur, nullptr); s_pipeBlur = VK_NULL_HANDLE; }
    if (s_layout)    { vkDestroyPipelineLayout(VulkanHW.m_Device, s_layout, nullptr); s_layout = VK_NULL_HANDLE; }
    if (s_sampNear)  { vkDestroySampler(VulkanHW.m_Device, s_sampNear, nullptr); s_sampNear = VK_NULL_HANDLE; }
    if (s_sampLin)   { vkDestroySampler(VulkanHW.m_Device, s_sampLin, nullptr); s_sampLin = VK_NULL_HANDLE; }
    if (s_pool)      { vkDestroyDescriptorPool(VulkanHW.m_Device, s_pool, nullptr); s_pool = VK_NULL_HANDLE; }
    if (s_setLayout) { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_setLayout, nullptr); s_setLayout = VK_NULL_HANDLE; }
    for (u32 i = 0; i < kFramesInFlight; ++i) { s_setGtao[i] = VK_NULL_HANDLE; s_setBlur[i] = VK_NULL_HANDLE; }
    s_inited = false; s_failed = false; s_generation = 0; s_hasResult = false;
}

}}  // namespace VK::SSAOPass
