// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan - Skinned-mesh pass (STEP B sub-step 2).
//
// Draws the skinned leaves (vkSkeletonX_ST/_PM) of dynamic CKinematics visuals
// collected in g_DynamicVisuals. GPU skinning: per-skeleton bone matrices are
// written to a host-visible SSBO, the skinned pipeline (skinned.vert/frag.spv)
// blends 1-4 bones per vertex. Called inside Pass_World (shares color+depth).
#pragma once
#include "vk_core.h"
#include "vk_pass_context.h"

namespace VK
{
    void Pass_Skinned(FrameContext& ctx);
    void Skinned_Destroy();

    // Bone upload phase — computes bones for every dynamic/HUD skeleton this frame
    // and writes them into the SSBO PRE-MULTIPLIED by the object's world matrix
    // (so S*pos in the shaders is world-space). Idempotent per Device.dwFrame;
    // called by Pass_SunShadow (casters need bones before the world pass) and as a
    // fallback by Pass_Skinned.
    void Skinned_UploadBones();

    // Depth-only render of the world (non-HUD) skinned casters into the currently
    // bound shadow attachment. Caller owns render begin/end, viewport and bias.
    // Default cull = the sun ortho box; pass cullPos+cullRange to cull against a
    // light sphere instead (spot/point shadow maps).
    void Skinned_RenderShadow(VkCommandBuffer cmd, const Fmatrix& lightVP,
                              const Fvector* cullPos = nullptr, float cullRange = 0.f);

    // Any world skinned caster inside the sphere this frame? (cheap pre-check —
    // lets the point-cube shadow skip re-rendering when no one is near the light)
    bool Skinned_AnyCasterInSphere(const Fvector& pos, float range);

    // NPCs into the scene depth PREPASS (camera VP) — GTAO occluders + early-Z.
    // Alpha-tested with skinned.frag's threshold so cutout coverage matches the
    // color pass. HUD skeletons never enter. Called from Pass_World's prepass.
    void Skinned_RenderDepthPrepass(VkCommandBuffer cmd, const Fmatrix& viewProj);
}
