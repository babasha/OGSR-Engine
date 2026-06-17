// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan — GTAO screen-space ambient occlusion (R4 gtao.h port).
//
// Runs INSIDE Pass_World, between the depth prepass and the color pass: the
// prepass depth (statics + alpha-tested statics) is the only input, so AO is
// ready before any forward shading samples it. Half-res R8 target: GTAO
// (ssao.frag) → depth-aware 3×3 blur (ssao_blur.frag).
//
// Consumers reach the result through the shared EnvLight set (binding 8) — the
// world/skinned fragments multiply their hemisphere+ambient terms by it, which
// is exactly what R4's combine_1.ps does (hdiffuse *= occ; sun & dynamic
// lights are NOT occluded).
#pragma once
#include "HW_Vulkan.h"

namespace VK {

// Perspective terms + camera basis derived from a row-major view·proj matrix.
// Device.mProject is NOT maintained on the Vulkan path (it reads as identity),
// so any pass that decodes the scene depth buffer (SSAO, sun shafts) must pull
// the terms out of the matrix that actually RENDERED that depth
// (Device.mFullTransform == FrameContext.viewProj). Extraction for M = V·P
// (row-vector, V affine orthonormal): _33 = M_i3/M_i4, _43 = M_43 − M_44·_33,
// 1/tan = |M column|; camera dir/right/top are the w/x/y columns normalized.
struct ProjTerms {
    float   p33, p43;     // depth decode: zview = p43 / (zndc − p33)
    float   tanX, tanY;   // tan(fov/2) horizontal / vertical
    Fvector dir, right, top;   // camera basis (unit)
};
ProjTerms DeriveProjTerms(const Fmatrix& viewProj);

namespace SSAOPass {

bool Init();      // pipelines/sets; call after the SPIRV loader exists (vk_RenderDeviceRender)
void Destroy();

// True when the pass will actually run (init OK + quality > 0). Pass_World
// uses this to decide whether to flip depth to SHADER_READ around Execute.
bool Enabled();

// (Re)create the half-res RTs for this scene extent. Pass_World calls this
// BEFORE EnvLight::Update so the descriptor it binds at binding 8 is the view
// Execute will render into this frame (not one a resize is about to destroy).
bool EnsureTargets(VkExtent2D sceneExtent);

// Record GTAO + blur. Depth must already be SHADER_READ_ONLY; leaves the AO
// result in SHADER_READ_ONLY for the color pass that follows.
void Execute(VkCommandBuffer cmd, VkExtent2D sceneExtent);

VkImageView GetResultView();   // final blurred AO; VK_NULL_HANDLE until first Execute sized the RTs
VkSampler   GetSampler();      // linear/clamp — receivers bilinearly upsample the half-res AO
u32         Generation();      // bumps when the RTs are (re)created — rebind triggers
float       Strength();        // ao_params.z for the Lighting UBO (0 disables in-shader)

// NPC normal G-buffer (full-res RGBA8). The skinned normal pass renders into it
// between the depth prepass and Execute; GTAO samples it (real normals where an
// NPC is visible, depth-derived fallback elsewhere). Null until EnsureTargets.
VkImage     GetNormalImage();
VkImageView GetNormalView();
VkFormat    GetNormalFormat();
VkExtent2D  GetNormalExtent();

}  // namespace SSAOPass
}  // namespace VK
