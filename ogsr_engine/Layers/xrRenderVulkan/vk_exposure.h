// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

#pragma once

// Auto-exposure formula inputs (R4 bloom_luminance_3.ps):
//   exposure = clamp(middleGray / (avgLum + lowLum), expMin, expMax)
// These are shared by the tonemap composite (vk_pass_tonemap.cpp) AND the bloom
// pre-exposure (vk_pass_bloom.cpp). They MUST be identical or bloom thresholds
// against a different exposure than the composite applies (halo/banding). This
// header is the single source of truth — previously the four values were
// copy-pasted in both files with a "keep in sync" comment.
namespace VK { namespace Exposure {

constexpr float kMiddleGray = 0.58f;    // exposure target (our unscaled HDR luminance)
constexpr float kLowLum     = 0.0001f;  // R4 ps_r2_tonemap_low_lum floor
constexpr float kExpMin     = 0.80f;    // exposure clamp lo (keeps night from washing to mid-gray)
constexpr float kExpMax     = 2.20f;    // exposure clamp hi

}}  // namespace VK::Exposure
