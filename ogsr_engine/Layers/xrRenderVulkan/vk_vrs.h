// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).
//
// ============================================================================
//  Variable Rate Shading (vk_vrs) — VK_KHR_fragment_shading_rate, attachment-based
// ============================================================================
//  Coarse-shade pixels via a shading-rate image (SRI): a small R8_UINT image,
//  one texel per ~16×16 screen tile, whose value encodes the fragment size
//  (1×1 / 2×2 / 4×4). The World color pass binds it as a shading-rate attachment
//  and sets combinerOps so the attachment REPLACES the rate → distant/peripheral
//  pixels run the (heavy forward) fragment shader once per 4 (2×2) or 16 (4×4).
//
//  v2 = DEPTH-DRIVEN: a compute pass reads the prepass depth and writes the rate
//  per tile by DISTANCE (near → 1×1, far/sky → 2×2 / 4×4). The SRI is written
//  every frame, so it's ring-buffered (FRAMES_IN_FLIGHT) to avoid in-flight
//  overwrite. Reuses the v1 pipeline/attachment/combiner wiring — only the FILL
//  changed from screen-position to depth. (v1 foveated is gone.)
//
//  Gated by `r_vrs` (0 off / 1 mild / 2 aggressive) AND VulkanHW.m_bVRSSupported
//  AND R8_UINT storage-image support.
// ============================================================================
#pragma once
#include "stdafx.h"

namespace VK { namespace VRS {

void        Init();                          // lazy: proc addr + compute pipeline + storage check
bool        Wanted();                        // VulkanHW.m_bVRSSupported && r_vrs>0 (cheap build gate)
// Build THIS frame's SRI from the scene depth (call with depth in SHADER_READ).
// A = ProjTerms.p43, B = ProjTerms.p33 (viewZ = A/(zndc-B)). Picks the ring slot
// from frameIndex; GetView() then returns that slot.
void        BuildFromDepth(VkCommandBuffer cmd, u32 frameIndex, VkImageView depthView,
                           VkExtent2D screen, float A, float B);
VkImageView GetView();                        // current frame's SRI view (null if not built)
VkExtent2D  TexelSize();                      // SRI tile size (device-queried)
void        CmdSetRate(VkCommandBuffer cmd);  // vkCmdSetFragmentShadingRateKHR (combiner=REPLACE)
void        Destroy();

}} // namespace VK::VRS
