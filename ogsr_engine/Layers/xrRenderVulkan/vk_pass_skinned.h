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

    // World-space foot-bone positions of this frame's skinned bodies (player + NPCs),
    // for snow footprint deformation. Valid after Skinned_UploadBones().
    void Skinned_CollectFeet(xr_vector<Fvector>& out, u32 maxFeet = 24);

    // World roots of skeleton-less skinned props/items (thrown/dropped bolts, grenades,
    // debris) — caller gates by ground proximity + prints them SHALLOW.
    void Skinned_CollectProps(xr_vector<Fvector>& out, u32 maxProps = 8);

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

    // NPCs into the SSAO normal G-buffer (camera VP) — feeds GTAO real per-pixel
    // normals where an NPC is visible (kills depth-derivative speckle on NPCs).
    // Depth-tested against the prepass depth (no depth write); alpha-tested like
    // the depth prepass. Caller owns render begin/end + viewport (Pass_World).
    void Skinned_RenderNormalPrepass(VkCommandBuffer cmd, const Fmatrix& viewProj);

    // NPC motion vectors (MV Phase 2a): re-draw world skinned leaves into the
    // motion target, skinning the current + previous pose so each pixel gets its
    // true screen motion. Depth-tested (LEQUAL, no write) against the complete
    // scene depth. Caller owns render begin/end + the negative-height viewport
    // (VK::MotionVec::ExecuteDynamic). prevVP = previous frame's view-proj.
    void Skinned_RenderMotion(VkCommandBuffer cmd, const Fmatrix& curVP, const Fmatrix& prevVP);

    // --- VSM skinned casters (vk_vsm consumes these to render NPC shadows into the
    // virtual shadow atlas). One entry per visible world skinned leaf this frame.
    struct VsmSkinnedCaster {
        Fvector      sphere_P;      // world-space bounding sphere (skeleton bound)
        float        sphere_R;
        u32          index_count, ib_first;
        s32          first_vertex;
        VkBuffer     vb, ib;
        VkIndexType  iType;
        u32          stride;        // vertex stride (36/40/44) → pipeline key + vertex input
        u32          base_bone, bone_count, skin_mode;
    };
    // Collect this frame's world skinned leaves as VSM casters (ensures bones are
    // uploaded first). HUD never casts. Append-only into `out` (not cleared here).
    void Skinned_CollectCasters(xr_vector<VsmSkinnedCaster>& out);

    // The shared world-space bone SSBO set (set 0, binding 0) + its layout — vk_vsm's
    // skinned-page pipeline binds the SAME set so baseBone indexes the same matrices.
    VkDescriptorSet       Skinned_GetBoneSet();
    VkDescriptorSetLayout Skinned_GetBoneSetLayout();

    // Build the vertHW skinned vertex input for `stride` (6 attrs) — vk_vsm reuses it
    // for the skinned-page pipeline so the layout matches the mesh buffers exactly.
    void Skinned_BuildVertexInput(u32 stride, VkVertexInputBindingDescription& binding,
                                  VkVertexInputAttributeDescription attrs[6]);
}
