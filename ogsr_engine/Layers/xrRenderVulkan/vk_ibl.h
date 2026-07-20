// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan — Sky specular IBL (image-based lighting) prefilter.
//
// The two weather sky cubes (env_s0/env_s1) ship BC-compressed and single-mip, so
// the forward shaders could only tap them at LOD 0 — no roughness blur, so glass /
// wet / "reflect the sky" looked like a sharp mirror at every roughness. This module
// resamples the BLENDED sky into an RGBA16F cube with a roughness-mip chain (mip N =
// rougher reflection) that the receivers sample with the split-sum specular IBL term
// (env_common.glsl iblSpecular). The sun's own GGX highlight rides the same F0/roughness.
//
// Prefilter runs on the fence-waited IMMEDIATE queue on demand (only when the weather
// cube view changes or the cross-fade moves) — the sky is low-frequency and changes
// rarely, so there is no per-frame cost. v1 = straight resample + box-blit mips; the
// clean upgrade is per-mip GGX importance-sample convolution.
#pragma once
#include "HW_Vulkan.h"

namespace VK { namespace IBL {

bool        Init();                 // idempotent; safe to call from EnvLight::Init
void        Destroy();
bool        Ready();                // true once a prefiltered cube exists (has content)

// Refresh the prefiltered cube from the current weather cubes if they changed.
// Called from EnvLight::Update with the same views/weight it binds for skyAmbient.
// No-op (cheap early-out) when nothing changed. Uses BeginImmediate (fence-waited).
//
// skyRotation (rad) is the weather's sky_rotation — the SAME spin the dome draw
// applies. It is part of the probe's identity, not a cosmetic extra: without it
// the probe's azimuth drifts away from the visible sky, so it participates in the
// change detection like the cubes and the cross-fade do.
// groundBounce scales the below-horizon hemisphere during the SH projection (the
// sky cube's skirt is not sky radiance — see sky_sh_project.comp.glsl).
void        Update(VkImageView sky0, VkImageView sky1, VkSampler skySampler, float weight,
                   float skyRotation, float groundBounce);

VkImageView GetSpecView();          // prefiltered specular cube (roughness mips); null until Ready
VkSampler   GetSampler();           // trilinear clamp sampler (maxLod = mip count)
u32         GetMaxMip();            // highest roughness mip index (= mip count - 1)

// Diffuse sky irradiance as 9 SH coefficients (std430 vec4[9]), projected from the
// prefiltered world-space probe by sky_sh_project.comp. GPU-only; read by the
// forward receivers via skyAmbient() in env_common.glsl.
VkBuffer    GetSHBuffer();          // null until the first projection has run
bool        SHReady();

}}  // namespace VK::IBL
