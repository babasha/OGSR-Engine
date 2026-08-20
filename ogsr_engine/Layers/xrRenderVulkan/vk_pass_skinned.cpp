// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan - Skinned-mesh pass (STEP B sub-step 2). See vk_pass_skinned.h.
#include "stdafx.h"
#include "vk_descriptors.h"     // VK::DescriptorWriter

// CKinematics (children + LL_GetTransform_R/LL_BoneCount) â€” same compat trick as
// the skeleton wrapper TUs. Keep dxRender_Visual mapped (children is xr_vector<dxRender_Visual*>).
#define FBasicVisualH
#include "vk_FBasicVisual.h"            // pulls vk_SkeletonCompat.h (dxRender_Visual->vkRender_Visual, vk_Visual.h)
#include "CRender_Vulkan.h"
#include "../xrRender/SkeletonCustom.h" // CKinematics
#include "../xrRender/SkeletonAnimated.h" // CKinematicsAnimated (blend-count diag)

#include "vk_pass_skinned.h"
#include "vk_pass_ssao.h"              // SSAOPass::GetNormalFormat â€” NPC normal G-buffer target
#include "vk_motionvec.h"             // VK::MotionVec::Format â€” RG16F motion target (skinned MV overlay)
#include "vk_pass_world.h"              // g_DynamicVisuals / DynVisual
#include "vk_swapchain.h"              // Swapchain.m_Format / m_DepthFormat
#include "vk_scene_color.h"            // HDR scene target format
#include "vk_shaders.h"               // g_ShaderManager
#include "vk_buffer.h"                // CVulkanBuffer
#include "vk_world_material.h"        // WorldMaterial / WorldMaterialCache (set 1 = diffuse)
#include "HW_Vulkan.h"                // VulkanHW.m_Device
#include "vk_command_buffer.h"        // CommandManager.GetCurrentFrame() â€” in-flight slot
#include "vk_pipeline_cache.h"        // PipelineCache::GetCacheObject() â€” shared disk-backed cache
#include "vk_compute_util.h"          // VK::MakePipelineLayout / CreateComputePipeline
#include "vk_gfx_pipeline.h"          // GfxPipelineBuilder
#include "vk_env_light.h"             // EnvLight â€” shared per-frame sun/hemi/ambient UBO (set 2)
#include "vk_shadow.h"                // ShadowMap::SphereVisible â€” caster culling
#include "vk_pass_lightcones.h"       // SynthCones::Submit â€” lightplanes-derived beam cones
#include "vk_profiler.h"              // VK_CPU_PROBE â€” pre-skinning CPU cost
#include "../../xr_3da/device.h"      // Device.mFullTransform_hud (HUD projection), dwFrame

#include <unordered_map>

// Console cvar at GLOBAL scope â€” a block-scope extern inside namespace VK would mangle as
// VK::ps_r_vsm_npc_dist -> LNK2001 (same trick as vk_pass_shadow's externs).
extern float ps_r_vsm_npc_dist;   // VSM NPC shadow cull distance (m); 0 = no cull
extern int   ps_r_preskin;        // compute pre-skinning on/off (A/B)

namespace VK {

namespace {
    bool                  s_inited     = false;
    bool                  s_failed     = false;
    VkPipelineLayout      s_layout     = VK_NULL_HANDLE;
    VkDescriptorSetLayout s_setLayout  = VK_NULL_HANDLE;
    VkDescriptorPool      s_pool       = VK_NULL_HANDLE;
    VkDescriptorSet       s_set        = VK_NULL_HANDLE;
    VkShaderModule        s_vs         = VK_NULL_HANDLE;
    VkShaderModule        s_fs         = VK_NULL_HANDLE;
    VkShaderModule        s_shadowVS   = VK_NULL_HANDLE;  // depth-only caster VS (optional)
    VkShaderModule        s_prepassVS  = VK_NULL_HANDLE;  // depth-prepass AT caster VS (optional)
    VkShaderModule        s_prepassFS  = VK_NULL_HANDLE;  // ... + alpha-test FS (matches skinned.frag)
    VkShaderModule        s_normalVS   = VK_NULL_HANDLE;  // NPC normal G-buffer VS (skins normal â†’ world, optional)
    VkShaderModule        s_normalFS   = VK_NULL_HANDLE;  // ... + AT FS writing worldN*0.5+0.5
    VkShaderModule        s_mvVS       = VK_NULL_HANDLE;  // motion-vector VS (skins cur+prev pose, optional)
    VkShaderModule        s_mvFS       = VK_NULL_HANDLE;  // ... â†’ RG16F screen motion
    VkPipelineLayout      s_mvLayout   = VK_NULL_HANDLE;  // set0 = bones + 152B push (cur/prev VP + bases + jitter)
    CVulkanBuffer         s_boneSSBO;
    Fmatrix*              s_boneMapped = nullptr;
    std::unordered_map<u32, VkPipeline> s_pipelines;        // keyed by vertex stride (36/40/44)
    std::unordered_map<u32, VkPipeline> s_shadowPipelines;  // depth-only casters, same key
    std::unordered_map<u32, VkPipeline> s_prepassPipelines; // scene depth-prepass AT casters
    std::unordered_map<u32, VkPipeline> s_normalPipelines;  // NPC normal G-buffer casters, same key
    std::unordered_map<u32, VkPipeline> s_motionPipelines;  // motion-vector casters, same key

    // Per-skeleton bone-slot record for the motion pass: maps a CKinematics to the
    // absolute baseBone it occupied. Rolled each frame â€” s_prevBoneMap then holds
    // LAST frame's slots, whose SSBO regions are still live (FRAMES_IN_FLIGHTâ‰¥2),
    // so the MV vertex shader can skin the previous pose for true animation motion.
    std::unordered_map<CKinematics*, std::pair<u32, u32>> s_curBoneMap;   // K -> {baseBone, boneCount}
    std::unordered_map<CKinematics*, std::pair<u32, u32>> s_prevBoneMap;
    // Same, but for the first-person HUD skeletons (drawn with the HUD-FOV
    // projection at near depth) — the HUD MV overlay needs its own prev poses.
    std::unordered_map<CKinematics*, std::pair<u32, u32>> s_curBoneMapHud;
    std::unordered_map<CKinematics*, std::pair<u32, u32>> s_prevBoneMapHud;

    // Per-frame upload registry: every skeleton whose bones landed in the SSBO
    // this frame. Bones are stored PRE-MULTIPLIED by the object's world matrix,
    // so S*pos in the shaders is world-space and one registry serves both the
    // shadow casters (mvp = lightVP) and the main pass (mvp = viewProj).
    struct SkelUpload { CKinematics* K; Fmatrix xform; u32 baseBone; u16 boneCount; float hemi; };
    xr_vector<SkelUpload> s_uploads;      // world dynamics
    xr_vector<SkelUpload> s_uploadsHud;   // first-person HUD (never casts)
    u32                   s_uploadFrame = u32(-1);

    constexpr u32 kMaxBones       = 16384;   // bone-matrix slots PER frame-in-flight (~1 MB each)
    constexpr u32 kFramesInFlight = CVulkanCommandManager::FRAMES_IN_FLIGHT;

    // ---------------------------------------------------------------------
    // COMPUTE PRE-SKINNING (r_preskin) â€” see Skinned_PreSkin / preskin.comp.
    // ---------------------------------------------------------------------
    VkPipeline            s_psPipe    = VK_NULL_HANDLE;
    VkPipelineLayout      s_psLayout  = VK_NULL_HANDLE;
    VkDescriptorSetLayout s_psSetL    = VK_NULL_HANDLE;
    VkDescriptorPool      s_psPool    = VK_NULL_HANDLE;
    VkDescriptorSet       s_psSet     = VK_NULL_HANDLE;
    CVulkanBuffer         s_psBuf;                        // shared output pool (GPU-only)
    bool                  s_psInited  = false;
    bool                  s_psFailed  = false;

    // Output layout = vertHW_1W (36 B). Chosen so the EXISTING skinned pipelines
    // consume it unchanged (see preskin.comp's header).
    constexpr u32 kPsStride     = 36;
    constexpr u32 kPsBytesSlot  = 12u << 20;              // per frame-in-flight (~349k verts)
    constexpr u32 kPsVertsSlot  = kPsBytesSlot / kPsStride;

    // Must match preskin.comp's push block. The 8-byte device address goes first
    // so the struct needs no padding.
    struct PreSkinPush { VkDeviceAddress src; u32 vertCount, srcStrideDW, dstFirst, skinMode, baseBone, boneCount; };
    static_assert(sizeof(PreSkinPush) == 32, "must match preskin.comp PC block");

    // This frame's pool residency: leaf visual -> first vertex in the pool.
    // Rebuilt from scratch every frame; empty when r_preskin is 0, so every draw
    // site falls back to the classic per-pass skinning with no extra branch.
    std::unordered_map<const void*, u32> s_psMap;
    u32 s_psFrame     = u32(-1);
    u32 s_identityBone = 0;      // bone slot holding IDENTITY for this frame's region
    u32 s_psLeaves = 0, s_psVerts = 0, s_psPeak = 0, s_psOverflow = 0;
    u32 s_psSkipped1W = 0, s_psMaxBones = 0;   // diag: how prop-heavy is this scene?

    // Where a leaf's vertices come from THIS frame. Pre-skinned leaves are drawn
    // as 1-weight geometry against the identity bone (the blend already happened
    // in compute); everything else keeps the real bones and its own vertHW stride.
    struct DrawSrc { VkBuffer vb; s32 firstVertex; u32 stride; u32 skinMode; u32 baseBone; u32 boneCount; };

    static DrawSrc ResolveDrawSrc(const void* leaf, const VK_Render_Mesh* mesh, u16 rmode,
                                  u32 baseBone, u32 boneCount)
    {
        // The frame check is not paranoia: the pool region is picked by the
        // in-flight slot, so a map left over from an earlier frame would point a
        // draw at another slot's vertices. Any frame where the PreSkin pass did
        // not run therefore falls through to the classic path.
        if (s_psFrame == Device.dwFrame && !s_psMap.empty()) {
            auto it = s_psMap.find(leaf);
            if (it != s_psMap.end())
                return { s_psBuf.GetHandle(), (s32)it->second, kPsStride, 1u, s_identityBone, 1u };
        }
        return { mesh->p_rm_Vertices->GetHandle(), (s32)mesh->vBase, mesh->vStride,
                 (rmode <= 2u) ? 1u : (u32(rmode) - 1u), baseBone, boneCount };
    }
    // Set 2 (per-frame sun/hemi/ambient) is the shared EnvLight set â€” see vk_env_light.{h,cpp}.

    // Push block â€” must match skinned.{vert,frag}.glsl PushConstants. hudMode (read
    // by the fragment) flags first-person HUD so it gets sun-direction-independent light.
    struct SkinPush { Fmatrix mvp; u32 skinMode; u32 baseBone; u32 boneCount; float hudMode; float hemi; };

    // Motion-vector push â€” must match motion_vec_skinned.vert. 152 B (device max is
    // 256 here, see vk_pipeline.cpp). curVP/prevVP are UNJITTERED and project the
    // cur/prev poses (jitter-free MV); jitter re-applies the sub-pixel offset to
    // gl_Position only, so depth still bit-matches the jittered forward geometry.
    struct MVSkinPush { Fmatrix curVP; Fmatrix prevVP; u32 skinMode; u32 baseBone; u32 prevBase; u32 boneCount; float jitterX; float jitterY; };
    static_assert(sizeof(MVSkinPush) == 152, "must match motion_vec_skinned.vert PC block");

    // Vertex input for the vertHW_* layouts. loc0 pos FLOAT4; loc1/3/4 packed
    // normal/tangent/binormal (UBYTE4N); loc2 = FLOAT2 (36/40) or FLOAT4 (44);
    // loc5 = 4 bone indices (40 only, dummy @0 elsewhere â€” unused by those modes).
    static void BuildSkinnedVI(u32 stride, VkVertexInputBindingDescription& b,
                               VkVertexInputAttributeDescription attrs[6])
    {
        b = { 0, stride, VK_VERTEX_INPUT_RATE_VERTEX };
        attrs[0] = { 0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0  };
        attrs[1] = { 1, 0, VK_FORMAT_R8G8B8A8_UNORM,      16 };
        attrs[3] = { 3, 0, VK_FORMAT_R8G8B8A8_UNORM,      20 };
        attrs[4] = { 4, 0, VK_FORMAT_R8G8B8A8_UNORM,      24 };
        if (stride == 44) {
            attrs[2] = { 2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 28 };  // tc + 2 bone idx
            attrs[5] = { 5, 0, VK_FORMAT_R8G8B8A8_UNORM,       0 };  // unused (2W/3W)
        } else if (stride == 40) {
            attrs[2] = { 2, 0, VK_FORMAT_R32G32_SFLOAT,  28 };       // tc only
            attrs[5] = { 5, 0, VK_FORMAT_R8G8B8A8_UNORM, 36 };       // 4 bone idx (4W)
        } else { // 36 (1W)
            attrs[2] = { 2, 0, VK_FORMAT_R32G32_SFLOAT,  28 };       // tc only
            attrs[5] = { 5, 0, VK_FORMAT_R8G8B8A8_UNORM, 0  };       // unused (1W)
        }
    }

    VkShaderModule s_glassDistFS = VK_NULL_HANDLE;   // glass_distort_skinned.frag (variant 3)

    // Pipeline variants: 0 = lit opaque, 1 = additive unlit (collimator marks),
    // 2 = GLASS (lit, src-alpha blend, no depth write â€” kinematics furniture/door
    // panes; the opaque path rendered them as solid texture), 3 = GLASS DISTORT
    // (same panes re-drawn into the heat-haze RT = refraction, r_glass_refr).
    static VkPipeline CreatePipeline(u32 stride, u32 variant)
    {
        const bool additive = (variant == 1u);
        const bool glass    = (variant == 2u);
        const bool distort  = (variant == 3u);
        if (distort && s_glassDistFS == VK_NULL_HANDLE) {
            s_glassDistFS = g_ShaderManager->Load("glass_distort_skinned.frag.spv");
            if (s_glassDistFS == VK_NULL_HANDLE) return VK_NULL_HANDLE;
        }
        VkVertexInputBindingDescription   binding{};
        VkVertexInputAttributeDescription attrs[6]{};
        BuildSkinnedVI(stride, binding, attrs);

        // Variant 3 renders into the particle heat-haze RT (RGBA8), not the scene.
        VK::GfxPipelineBuilder b(s_layout);
        b.Vert(s_vs).Frag(distort ? s_glassDistFS : s_fs)
         .Bindings(&binding, 1).Attrs(attrs, 6)
         .Cull(VK_CULL_MODE_NONE)
         // marks/glass don't write depth (R4 zb(true,false))
         .Depth(true, !(additive || glass || distort))
         .Color(distort ? VK_FORMAT_R8G8B8A8_UNORM : VK::SceneColor::Format())
         .DepthTarget(Swapchain.m_DepthFormat);
        if (additive) {
            // Collimator marks (R4 hud_reddotsight: blend(srcalpha, one)) â€”
            // the dot ADDS light over the sight glass.
            b.BlendAdd();
        }
        if (glass || distort) {
            // Translucent pane over whatever is behind (lit output, capped
            // alpha comes from the fragment â€” skinMode bit 32). The distort
            // variant blends the wobble over the haze RT's neutral 0.5.
            b.BlendAlpha();
        }

        // VRS: variants 0/1/2 draw inside the world-color pass with the SRI
        // attached — STATIC {1x1, KEEP, REPLACE} state so distant NPCs coarse-
        // shade with the world (static, not dynamic: see vk_pipeline_cache.cpp —
        // a dynamic rate gets invalidated by unrelated binds). The distort
        // variant renders into the haze RT (no SRI) and stays full-rate.
        VkPipelineFragmentShadingRateStateCreateInfoKHR fsrState{ VK_STRUCTURE_TYPE_PIPELINE_FRAGMENT_SHADING_RATE_STATE_CREATE_INFO_KHR };
        fsrState.fragmentSize   = { 1, 1 };
        fsrState.combinerOps[0] = VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR;
        fsrState.combinerOps[1] = VK_FRAGMENT_SHADING_RATE_COMBINER_OP_REPLACE_KHR;   // SRI attachment wins
        if (VulkanHW.m_bVRSSupported && !distort)
            b.RenderingNext(&fsrState).CreateFlags(VK_PIPELINE_CREATE_RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR);

        VkPipeline h = b.Build("Skinned stride=%u variant=%u", stride, variant);
        if (h == VK_NULL_HANDLE) return VK_NULL_HANDLE;
        Msg("[VK Skinned] pipeline created stride=%u variant=%u", stride, variant);
        return h;
    }

    static VkPipeline GetPipeline(u32 stride, u32 variant = 0u)
    {
        const u32 key = stride | (variant << 16);
        auto it = s_pipelines.find(key);
        if (it != s_pipelines.end()) return it->second;
        VkPipeline p = CreatePipeline(stride, variant);
        s_pipelines.emplace(key, p);
        return p;
    }

    static bool Init()
    {
        if (s_inited) return !s_failed;
        s_inited = true;

        if (!g_ShaderManager) g_ShaderManager = xr_new<VK::CVulkanSPIRVLoader>();
        s_vs = g_ShaderManager->Load("skinned.vert.spv");
        s_fs = g_ShaderManager->Load("skinned.frag.spv");
        if (s_vs == VK_NULL_HANDLE || s_fs == VK_NULL_HANDLE) {
            Msg("![VK Skinned] failed to load skinned.vert/frag.spv");
            s_failed = true; return false;
        }
        // Optional â€” absence just disables NPC shadow casting.
        s_shadowVS = g_ShaderManager->Load("shadow_skinned.vert.spv");
        if (s_shadowVS == VK_NULL_HANDLE)
            Msg("![VK Skinned] shadow_skinned.vert.spv missing â€” skinned shadow casting disabled");
        // Optional â€” absence just keeps NPCs out of the depth prepass (no NPC AO).
        s_prepassVS = g_ShaderManager->Load("shadow_skinned_at.vert.spv");
        s_prepassFS = g_ShaderManager->Load("shadow_skinned_at.frag.spv");
        if (s_prepassVS == VK_NULL_HANDLE || s_prepassFS == VK_NULL_HANDLE)
            Msg("![VK Skinned] shadow_skinned_at.{vert,frag}.spv missing â€” NPCs excluded from the depth prepass");
        // Optional â€” absence just keeps NPCs on depth-derived GTAO normals (no NPC normal G-buffer).
        s_normalVS = g_ShaderManager->Load("shadow_skinned_normal.vert.spv");
        s_normalFS = g_ShaderManager->Load("shadow_skinned_normal.frag.spv");
        if (s_normalVS == VK_NULL_HANDLE || s_normalFS == VK_NULL_HANDLE)
            Msg("![VK Skinned] shadow_skinned_normal.{vert,frag}.spv missing â€” NPC AO normals disabled");
        // Optional â€” absence just keeps NPCs on camera-only motion vectors (no animation MV).
        s_mvVS = g_ShaderManager->Load("motion_vec_skinned.vert.spv");
        s_mvFS = g_ShaderManager->Load("motion_vec_skinned.frag.spv");
        if (s_mvVS == VK_NULL_HANDLE || s_mvFS == VK_NULL_HANDLE)
            Msg("![VK Skinned] motion_vec_skinned.{vert,frag}.spv missing â€” NPC animation motion vectors disabled");

        // Descriptor set layout: 1 storage buffer (bone matrices), VERTEX stage.
        if (!VK::MakeDescriptorSets({ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER }, 1,
                                    s_setLayout, s_pool, &s_set,
                                    VK_SHADER_STAGE_VERTEX_BIT, "Skinned.Bones")) {
            s_failed = true; return false;
        }

        // Set 2 = shared per-frame env lighting (sun/hemi/ambient). Owned by EnvLight;
        // ensure it's initialised so its set layout exists for the pipeline layout.
        EnvLight::Init();
        VkDescriptorSetLayout lightLayout = EnvLight::GetSetLayout();
        if (lightLayout == VK_NULL_HANDLE) { Msg("![VK Skinned] EnvLight set layout not ready"); s_failed = true; return false; }

        // Pipeline layout: push {mvp, skinMode, baseBone} (VERTEX) + set0=bone SSBO +
        // set1=material (base/detail/lmap) so the FS can sample the diffuse texture +
        // set2=per-frame environment lighting (FRAGMENT).
        VkDescriptorSetLayout matLayout = WorldMaterialCache::GetSetLayout();
        if (matLayout == VK_NULL_HANDLE) { Msg("![VK Skinned] material set layout not ready"); s_failed = true; return false; }
        VkDescriptorSetLayout setLayouts[3] = { s_setLayout, matLayout, lightLayout };
        VkPushConstantRange pcr{ VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(SkinPush) };
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 3; plci.pSetLayouts = setLayouts;
        plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
        if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_layout) != VK_SUCCESS) { s_failed = true; return false; }

        // Motion-vector pipeline layout: set 0 (bone SSBO) + set 1 (diffuse, for the
        // alpha-test) + a 144-byte VERTEX push (cur/prev VP + bone bases). No light set
        // â€” the MV fragment only needs the two clip positions + the cutout test.
        if (s_mvVS != VK_NULL_HANDLE && s_mvFS != VK_NULL_HANDLE) {
            VkDescriptorSetLayout mvSets[2] = { s_setLayout, matLayout };
            VkPushConstantRange mvPcr{ VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(MVSkinPush) };
            VkPipelineLayoutCreateInfo mvPlci{};
            mvPlci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            mvPlci.setLayoutCount = 2; mvPlci.pSetLayouts = mvSets;
            mvPlci.pushConstantRangeCount = 1; mvPlci.pPushConstantRanges = &mvPcr;
            if (vkCreatePipelineLayout(VulkanHW.m_Device, &mvPlci, nullptr, &s_mvLayout) != VK_SUCCESS) {
                Msg("![VK Skinned] motion-vector pipeline layout failed â€” NPC animation MV disabled");
                s_mvVS = s_mvFS = VK_NULL_HANDLE;   // disable the MV path, keep the rest alive
            }
        }

        // Bone SSBO (host-visible, persistently mapped â€” Create sets HOST_ACCESS+MAPPED for STORAGE).
        // Sized FRAMES_IN_FLIGHT Ã— kMaxBones: each in-flight frame owns region
        // [slot*kMaxBones, (slot+1)*kMaxBones). Frame S only writes its region after
        // WaitForFence(S) in CRender::Begin proved the GPU finished the previous frame-S
        // submission that read it, so the regions still in flight (S+1/S+2) are never
        // clobbered. baseBone push-constant absolutely-indexes the whole buffer, so the
        // single VK_WHOLE_SIZE descriptor below needs no per-frame rebind.
        s_boneSSBO.Create(VkDeviceSize(kMaxBones) * kFramesInFlight * sizeof(Fmatrix),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
        s_boneMapped = static_cast<Fmatrix*>(s_boneSSBO.Map());
        if (!s_boneMapped) { Msg("![VK Skinned] bone SSBO map failed"); s_failed = true; return false; }

        VK::DescriptorWriter(s_set).StorageBuffer(0, s_boneSSBO.GetHandle()).Flush();

        Msg("[VK Skinned] init OK (SSBO %u bones, push=%u)", kMaxBones, (u32)sizeof(SkinPush));
        return true;
    }

    // Depth-only caster pipeline (sun shadow map): shadow_skinned.vert skins the
    // vertex and writes light-clip depth; no fragment stage, no color. Reuses
    // s_layout (sets 1/2 simply go unbound â€” the shader only touches set 0).
    static VkPipeline CreateShadowPipeline(u32 stride)
    {
        VkVertexInputBindingDescription   binding{};
        VkVertexInputAttributeDescription attrs[6]{};
        BuildSkinnedVI(stride, binding, attrs);

        // VERTEX only, no color attachments — depth-only pass.
        VkPipeline h = VK::GfxPipelineBuilder(s_layout)
            .Vert(s_shadowVS)
            .Bindings(&binding, 1).Attrs(attrs, 6)
            .Cull(VK_CULL_MODE_NONE)
            .DynamicDepthBias()                  // caller sets the same bias as statics
            .Depth(true, true)
            .DepthTarget(VK_FORMAT_D32_SFLOAT)   // matches the shadow map
            .Build("Skinned shadow stride=%u", stride);
        if (h == VK_NULL_HANDLE) return VK_NULL_HANDLE;
        Msg("[VK Skinned] shadow pipeline created stride=%u", stride);
        return h;
    }

    static VkPipeline GetShadowPipeline(u32 stride)
    {
        auto it = s_shadowPipelines.find(stride);
        if (it != s_shadowPipelines.end()) return it->second;
        VkPipeline p = CreateShadowPipeline(stride);
        s_shadowPipelines.emplace(stride, p);
        return p;
    }

    // Depth-PREPASS caster pipeline: skinned skinning + alpha-test fragment
    // (same a<0.25 discard as the color pass â†’ identical coverage). Renders
    // into the scene depth, so no depth bias and the scene depth format.
    static VkPipeline CreatePrepassPipeline(u32 stride)
    {
        VkVertexInputBindingDescription   binding{};
        VkVertexInputAttributeDescription attrs[6]{};
        BuildSkinnedVI(stride, binding, attrs);

        VkPipeline h = VK::GfxPipelineBuilder(s_layout)   // no color attachments
            .Vert(s_prepassVS).Frag(s_prepassFS)
            .Bindings(&binding, 1).Attrs(attrs, 6)
            .Cull(VK_CULL_MODE_NONE)
            .Depth(true, true)
            .DepthTarget(Swapchain.m_DepthFormat)   // scene depth (prepass gates on D32)
            .Build("Skinned prepass stride=%u", stride);
        if (h == VK_NULL_HANDLE) return VK_NULL_HANDLE;
        Msg("[VK Skinned] prepass pipeline created stride=%u", stride);
        return h;
    }

    static VkPipeline GetPrepassPipeline(u32 stride)
    {
        auto it = s_prepassPipelines.find(stride);
        if (it != s_prepassPipelines.end()) return it->second;
        VkPipeline p = CreatePrepassPipeline(stride);
        s_prepassPipelines.emplace(stride, p);
        return p;
    }

    // NPC NORMAL G-buffer pipeline: same skinning as the depth prepass (positions
    // bit-match â†’ LEQUAL passes only on the visible surface), but writes the
    // world-space normal to a color attachment and does NOT write depth (tests
    // against the already-laid prepass depth). Feeds GTAO real NPC normals.
    static VkPipeline CreateNormalPipeline(u32 stride)
    {
        VkVertexInputBindingDescription   binding{};
        VkVertexInputAttributeDescription attrs[6]{};
        BuildSkinnedVI(stride, binding, attrs);

        VkPipeline h = VK::GfxPipelineBuilder(s_layout)
            .Vert(s_normalVS).Frag(s_normalFS)
            .Bindings(&binding, 1).Attrs(attrs, 6)
            .Cull(VK_CULL_MODE_NONE)
            .Depth(true, false)                     // test only â€” the prepass owns the depth
            .Color(SSAOPass::GetNormalFormat())
            .DepthTarget(Swapchain.m_DepthFormat)   // tests the scene prepass depth
            .Build("Skinned normal stride=%u", stride);
        if (h == VK_NULL_HANDLE) return VK_NULL_HANDLE;
        Msg("[VK Skinned] normal G-buffer pipeline created stride=%u", stride);
        return h;
    }

    static VkPipeline GetNormalPipeline(u32 stride)
    {
        auto it = s_normalPipelines.find(stride);
        if (it != s_normalPipelines.end()) return it->second;
        VkPipeline p = CreateNormalPipeline(stride);
        s_normalPipelines.emplace(stride, p);
        return p;
    }

    // MOTION-VECTOR pipeline: skins cur+prev pose (motion_vec_skinned.vert) and
    // writes RG16F screen motion. Tests against the already-laid scene depth
    // (LEQUAL, no write) so only the visible NPC surface overwrites the static
    // camera field. Negative-height viewport (set by the caller) matches the
    // forward/prepass raster so positions/depth bit-match â†’ LEQUAL passes on equal.
    static VkPipeline CreateMotionPipeline(u32 stride)
    {
        VkVertexInputBindingDescription   binding{};
        VkVertexInputAttributeDescription attrs[6]{};
        BuildSkinnedVI(stride, binding, attrs);

        VkPipeline h = VK::GfxPipelineBuilder(s_mvLayout)
            .Vert(s_mvVS).Frag(s_mvFS)
            .Bindings(&binding, 1).Attrs(attrs, 6)
            .Cull(VK_CULL_MODE_NONE)
            .Depth(true, false)   // scene depth already owns the surface
            .Color(VK::MotionVec::Format(), VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT)   // RG16F motion
            .DepthTarget(Swapchain.m_DepthFormat)
            .Build("Skinned motion stride=%u", stride);
        if (h == VK_NULL_HANDLE) return VK_NULL_HANDLE;
        Msg("[VK Skinned] motion-vector pipeline created stride=%u", stride);
        return h;
    }

    static VkPipeline GetMotionPipeline(u32 stride)
    {
        auto it = s_motionPipelines.find(stride);
        if (it != s_motionPipelines.end()) return it->second;
        VkPipeline p = CreateMotionPipeline(stride);
        s_motionPipelines.emplace(stride, p);
        return p;
    }

    // Resolve a CKinematics child to its mesh + render mode (skinned leaves only).
    template <class TChild>
    static bool ResolveSkinnedLeaf(TChild* child, VK_Render_Mesh*& mesh, u16& rmode)
    {
        if (auto* st = dynamic_cast<vkSkeletonX_ST*>(child)) { mesh = &st->m_mesh; rmode = st->RenderMode; }
        else if (auto* pm = dynamic_cast<vkSkeletonX_PM*>(child)) { mesh = &pm->m_mesh; rmode = pm->RenderMode; }
        else return false;
        return mesh->p_rm_Vertices && mesh->p_rm_Indices && mesh->iCount != 0;
    }

    // Draw one list of uploaded skeletons. Bones in the SSBO are already
    // world-space (pre-multiplied at upload), so viewProj is pushed as-is.
    // lastPipe/lastMatSet persist across lists (viewport changes between lists
    // don't disturb pipeline/descriptor bindings â€” both are dynamic / separate).
    static void DrawSkinnedList(VkCommandBuffer cmd, const xr_vector<SkelUpload>& list,
                                const Fmatrix& viewProj, u32& nDraw,
                                VkPipeline& lastPipe, VkDescriptorSet& lastMatSet, VkDescriptorSet lightSet,
                                float hudMode)
    {
        for (const SkelUpload& u : list)
        {
            for (auto* child : u.K->children)
            {
                VK_Render_Mesh* mesh = nullptr;
                u16 rmode = 0;
                if (!ResolveSkinnedLeaf(child, mesh, rmode)) continue;

                // Collimator/red-dot marks: additive unlit pipeline (R4
                // hud_reddotsight). Glass panes (kinematics furniture/doors,
                // OGF "glass" shader / glas\ texture): lit blended, no z-write.
                // Lightplanes leaves are never drawn — they only register their
                // synthesized beam (Pass_LightCones draws the real cone).
                const bool emissive = child->m_bEmissiveAdd;
                const bool glass    = child->m_bModelGlass;
                const bool litblend = child->m_bLitBlend;
                if (litblend)
                {
                    // Register the synthesized beam cone (Pass_LightCones draws
                    // it later this frame). Single-bone fans follow their bone —
                    // hidden bone (torch off: X-Ray zeroes it) = no beam, and a
                    // rotating searchlight carries its cone. HUD skipped: a
                    // camera-apex beam is milk (the real flashlight spot covers it).
                    if (hudMode < 0.5f && child->m_SynthBeams.count)
                    {
                        bool    visible = true;
                        Fmatrix xf      = u.xform;
                        u32     boneId  = u32(-1);
                        if (auto* st = dynamic_cast<vkSkeletonX_ST*>(child)) {
                            if (st->RenderMode == vkSkeletonX_ST::RM_SINGLE) boneId = st->RMS_boneid;
                        } else if (auto* pm = dynamic_cast<vkSkeletonX_PM*>(child)) {
                            if (pm->RenderMode == vkSkeletonX_PM::RM_SINGLE) boneId = pm->RMS_boneid;
                        }
                        if (boneId != u32(-1) && u.K) {
                            visible = !!u.K->LL_GetBoneVisible((u16)boneId);
                            if (visible)
                                xf.mul_43(u.xform, u.K->LL_GetBoneInstance((u16)boneId).mRenderTransform);
                        }
                        if (visible) SynthCones::Submit(child, xf);
                        else         SynthCones::Revoke(child);   // torch off — kill the beam now
                    }
                    // The fake sheets themselves are never drawn — Pass_LightCones
                    // draws the REAL volumetric cone from the synthesized light.
                    continue;
                }
                const DrawSrc ds = ResolveDrawSrc(child, mesh, rmode, u.baseBone, u.boneCount);
                VkPipeline pipe = GetPipeline(ds.stride, emissive ? 1u : (glass ? 2u : 0u));
                if (pipe == VK_NULL_HANDLE) continue;

                if (pipe != lastPipe) {
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_layout, 0, 1, &s_set, 0, nullptr);
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_layout, 2, 1, &lightSet, 0, nullptr);  // set2 = env lighting (per-frame)
                    lastPipe = pipe;
                    lastMatSet = VK_NULL_HANDLE;   // pipeline change disturbs set bindings â€” rebind material
                }

                // Diffuse texture (set 1) from the leaf's WorldMaterial; fall back to default white.
                WorldMaterial* mat = child->m_pWorldMaterial ? child->m_pWorldMaterial : WorldMaterialCache::GetDefault();
                VkDescriptorSet matSet = (mat && mat->set != VK_NULL_HANDLE) ? mat->set : VK_NULL_HANDLE;
                if (matSet == VK_NULL_HANDLE) continue;   // no texture descriptor â†’ can't sample, skip
                if (matSet != lastMatSet) {
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_layout, 1, 1, &matSet, 0, nullptr);
                    lastMatSet = matSet;
                }

                // RenderMode enum -> skinMode: RM_SINGLE(1)/RM_SKINNING_1B(2)->1, 2B(3)->2, 3B(4)->3, 4B(5)->4.
                // Bit 4 (16) = emissive-add flag for the fragment (unlit output);
                // bit 5 (32) = glass (skip the alpha test, cap the blend alpha);
                // the vertex shader masks them off before the skinning switch.
                u32 skinMode = ds.skinMode;
                if (emissive) skinMode |= 16u;
                if (glass)    skinMode |= 32u;
                SkinPush pc{ viewProj, skinMode, ds.baseBone, ds.boneCount, hudMode, u.hemi };
                vkCmdPushConstants(cmd, s_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

                VkDeviceSize vbOff = 0;
                vkCmdBindVertexBuffers(cmd, 0, 1, &ds.vb, &vbOff);
                vkCmdBindIndexBuffer(cmd, mesh->p_rm_Indices->GetHandle(), 0, mesh->iType);
                vkCmdDrawIndexed(cmd, mesh->iCount, 1, mesh->iBase, ds.firstVertex, 0);
                ++nDraw;
            }
        }
    }

    // HUD near-depth viewport (matches R4 rmNear = depth range [0, 0.02]): the HUD
    // weapon/hands occupy the front 2% of depth so they always win the depth test
    // against the world without clipping into it. Keeps the negative-height Y-flip.
    static void SetViewportDepth(VkCommandBuffer cmd, const VkExtent2D& ext, float minD, float maxD)
    {
        VkViewport vp{ 0.0f, (float)ext.height, (float)ext.width, -(float)ext.height, minD, maxD };
        vkCmdSetViewport(cmd, 0, 1, &vp);
    }
}  // anon namespace

// GLASS refraction, kinematics half: re-draw this frame's glass leaves (cabinet
// panes etc.) into the ALREADY-BEGUN heat-haze distortion pass with the variant-3
// pipeline (GPU skinning + procedural wobble). Called by Pass_Particles phase 3;
// s_uploads (bones incl.) is still this frame's data there. strength rides the
// SkinPush hemi slot. HUD list excluded (no world positions, never refracts).
void Skinned_RenderGlassDistort(VkCommandBuffer cmd, const Fmatrix& viewProj, float strength)
{
    if (s_uploads.empty() || s_set == VK_NULL_HANDLE) return;
    VkPipeline lastPipe = VK_NULL_HANDLE;
    for (const SkelUpload& u : s_uploads)
    {
        for (auto* child : u.K->children)
        {
            VK_Render_Mesh* mesh = nullptr; u16 rmode = 0;
            if (!ResolveSkinnedLeaf(child, mesh, rmode)) continue;
            if (!child->m_bModelGlass) continue;
            const DrawSrc ds = ResolveDrawSrc(child, mesh, rmode, u.baseBone, u.boneCount);
            VkPipeline pipe = GetPipeline(ds.stride, 3u);
            if (pipe == VK_NULL_HANDLE) continue;
            if (pipe != lastPipe) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_layout, 0, 1, &s_set, 0, nullptr);   // set0 = bones
                lastPipe = pipe;
            }
            SkinPush pc{ viewProj, ds.skinMode, ds.baseBone, ds.boneCount, 0.0f, strength };
            vkCmdPushConstants(cmd, s_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
            VkDeviceSize vbOff = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &ds.vb, &vbOff);
            vkCmdBindIndexBuffer(cmd, mesh->p_rm_Indices->GetHandle(), 0, mesh->iType);
            vkCmdDrawIndexed(cmd, mesh->iCount, 1, mesh->iBase, ds.firstVertex, 0);
        }
    }
}

void Skinned_UploadBones()
{
    if (!Init()) return;
    if (s_uploadFrame == Device.dwFrame) return;   // once per frame
    s_uploadFrame = Device.dwFrame;
    s_uploads.clear();
    s_uploadsHud.clear();

    // Roll the bone-slot records: this frame's map becomes "previous" for the MV
    // pass (its SSBO regions are still live), and we rebuild "current" below.
    s_prevBoneMap = std::move(s_curBoneMap);
    s_curBoneMap.clear();
    s_prevBoneMapHud = std::move(s_curBoneMapHud);
    s_curBoneMapHud.clear();

    // Pick THIS frame's bone-SSBO region by the fence-guarded in-flight slot, so the
    // CPU never writes matrices a still-in-flight frame's GPU may be reading.
    const u32 slot  = CommandManager.GetCurrentFrame();
    u32 cursor      = slot * kMaxBones;
    const u32 limit = cursor + kMaxBones;

    // Slot 0 of this frame's region is reserved for IDENTITY: pre-skinned leaves
    // are drawn as 1-weight geometry pointing at it, so the consumer shaders'
    // `S * pos` / `mat3(S) * nrm` pass the already-skinned world values through
    // untouched (see Skinned_PreSkin). Costs one matrix per frame.
    s_identityBone = cursor;
    s_boneMapped[cursor].identity();
    ++cursor;

    auto uploadList = [&](const xr_vector<DynVisual>& list, xr_vector<SkelUpload>& out) {
        for (const DynVisual& d : list)
        {
            if (!d.vis) continue;
            CKinematics* K = dynamic_cast<CKinematics*>(d.vis);  // CKinematics : FHierrarhyVisual : vkRender_Visual
            if (!K || K->children.empty()) continue;

            // bForceExact=TRUE: recompute bones every frame (matches R4's render
            // path, r__dsgraph_build.cpp pV->CalculateBones(TRUE)). With FALSE the
            // shared CKinematics::CalculateBones takes its "slow update" early-out
            // (Device.dwTimeGlobal < UCalc_Time + UCalc_Interval) and leaves bones
            // STALE for UCalc_Interval ms â†’ visibly jerky animation. The
            // dwTimeGlobal==UCalc_Time guard inside still prevents double-advance.
            K->CalculateBones(TRUE);
            const u16 bc = K->LL_BoneCount();
            if (bc == 0) continue;
            if (cursor + bc > limit) break;   // this frame's SSBO region full

            // Store boneÂ·xform (modelâ†’WORLD): S*pos in the shaders is then world-
            // space, the main pass pushes plain viewProj, the shadow pass plain
            // lightVP, and normals get the object rotation they previously missed.
            const u32 base = cursor;
            for (u16 i = 0; i < bc; ++i)
                s_boneMapped[cursor + i].mul(d.xform, K->LL_GetTransform_R(i));

            // diag: sample skeletons round-robin -- if a
            // bone lands kilometres from its object xform (never-animated bones,
            // garbage matrices), the mesh "draws" but collapses off-screen.
            // blends: are the PlayCycle'd loops actually attached to this visual?
            {
                static u32 s_boneDiag = 0;
                if (++s_boneDiag % 501 == 1)
                {
                    const Fmatrix& b0 = s_boneMapped[base];
                    const Fmatrix& bl = s_boneMapped[base + bc - 1];
                    u32 nBlends = 0;
                    if (auto* ka = dynamic_cast<CKinematicsAnimated*>(K))
                        for (u32 p = 0; p < 4; ++p)
                            nBlends += ka->LL_PartBlendsCount(p);
                    VisMask vm = K->LL_GetBonesVisible();
                    const Fvector mT0 = K->LL_GetBoneInstance(0).mTransform.c;
                    Msg("[VK Skinned] up-diag: bones=%u blends=%u vis=0x%llx/0x%llx mT0=(%.2f,%.2f,%.2f) xform=(%.1f,%.1f,%.1f) bone0=(%.1f,%.1f,%.1f) boneN=(%.1f,%.1f,%.1f)",
                        bc, nBlends, vm._visimask.flags, vm._visimask_ex.flags, VPUSH(mT0), VPUSH(d.xform.c), VPUSH(b0.c), VPUSH(bl.c));
                }

                // ⭐⭐⭐WHICH SKINNED VISUAL PUTS VERTICES IN THE SKY.
                //
                // Reported from play twice and hunted from the game side for
                // three builds without success: a body standing in a bind pose
                // under a fan of grey spikes tens of metres tall. The game-side
                // sweep never even saw the object -- every candidate the game
                // does own was tested and cleared, including the first-person
                // body -- so it is a visual reaching us without a game-side owner.
                //
                // ⚠A bind pose ALONE cannot make spikes: identity skinning
                // matrices simply draw the mesh where it was authored, which is
                // the T-pose we see. Vertices leave for infinity when the shader
                // reads bone matrices OUTSIDE this skeleton's slice, i.e. when a
                // child's declared RenderMode disagrees with the vertex layout
                // the shader is told to assume (skinMode below). So name the mode
                // per child, for every unanimated skeleton, once each.
                //
                // ⭐Costs nothing in the normal case: an animated skeleton has
                // blends and never reaches this branch.
                {
                    u32 blends_now = 0;
                    if (auto* ka = dynamic_cast<CKinematicsAnimated*>(K))
                        for (u32 p = 0; p < 4; ++p)
                            blends_now += ka->LL_PartBlendsCount(p);

                    // ⚠⚠ONCE-PER-VISUAL WAS THE WRONG CADENCE, AND IT COST A
                    // WRONG DIAGNOSIS.
                    //
                    // The first cut printed each visual once and never again. Its
                    // xform was therefore a FIRST-SIGHT SNAPSHOT, and reading
                    // those stale coordinates next to live creature positions
                    // "showed" the phantom bodies riding on the NPCs. They were
                    // not: when the positions were finally compared at the same
                    // instant, the phantoms sat 80 m away, and a fix built on the
                    // false correlation had to be reverted (see
                    // CAttachmentOwner::renderable_Render).
                    //
                    // ⭐So: re-announce the same visual every two seconds. A
                    // diagnostic whose value goes stale must say WHEN it is
                    // speaking, or it invents relationships that are not there.
                    //
                    // Bone names come along because they identify the MODEL CLASS
                    // without any name string being available in a release build:
                    // `bip01_*` is a body, `wpn_*` a weapon, anything else a prop.
                    if (0 == blends_now)
                    {
                        struct Seen { CKinematics* K; u32 ms; };
                        static xr_vector<Seen> s_seen;
                        const u32 now_ms = Device.dwTimeGlobal;
                        auto it = std::find_if(s_seen.begin(), s_seen.end(), [K](const Seen& s) { return s.K == K; });
                        const bool due = (it == s_seen.end()) || (now_ms - it->ms >= 2000);
                        if (due && s_seen.size() < 64)
                        {
                            if (it == s_seen.end())
                                s_seen.push_back({K, now_ms});
                            else
                                it->ms = now_ms;

                            u32 kids = 0;
                            string512 modes{};
                            for (auto* child : K->children)
                            {
                                VK_Render_Mesh* mesh = nullptr;
                                u16 rmode = 0;
                                if (!ResolveSkinnedLeaf(child, mesh, rmode))
                                    continue;
                                ++kids;
                                const u32 skinMode = (rmode <= 2u) ? 1u : (u32(rmode) - 1u);
                                string32 one;
                                xr_sprintf(one, "%s%u->%u", kids > 1 ? "," : "", u32(rmode), skinMode);
                                xr_strcat(modes, one);
                            }
                            // first and last bone name: enough to tell a body from a prop
                            LPCSTR b_first = (bc > 0) ? K->LL_BoneName(0) : "?";
                            LPCSTR b_last = (bc > 0) ? K->LL_BoneName(u16(bc - 1)) : "?";
                            Msg("![VK Skinned] UNANIMATED visual @%u: bones=%u children=%u bone0='%s' boneN='%s' rmode->skinMode[%s] xform=(%.1f,%.1f,%.1f)", now_ms, bc, kids,
                                b_first, b_last, modes, VPUSH(d.xform.c));
                        }
                    }
                }
            }
            cursor += bc;
            out.push_back({ K, d.xform, base, bc, d.hemi });
        }
    };
    uploadList(g_DynamicVisuals, s_uploads);
    uploadList(g_HudVisuals,     s_uploadsHud);

    // Record this frame's world-skeleton slots so NEXT frame's MV pass can find the
    // previous pose (HUD excluded â€” it never enters the MV pass).
    for (const SkelUpload& u : s_uploads)
        s_curBoneMap[u.K] = { u.baseBone, (u32)u.boneCount };
    // HUD weapon now has its own MV overlay (Skinned_RenderMotionHud), so track
    // its prev poses too — last frame's SSBO region stays live (FRAMES_IN_FLIGHT>=2).
    for (const SkelUpload& u : s_uploadsHud)
        s_curBoneMapHud[u.K] = { u.baseBone, (u32)u.boneCount };
}

// ===========================================================================
//  COMPUTE PRE-SKINNING  (r_preskin)
// ===========================================================================
namespace {

static bool PreSkinInit()
{
    if (s_psInited) return !s_psFailed;
    s_psInited = true;

    VkShaderModule cs = g_ShaderManager->Load("preskin.comp.spv");
    if (cs == VK_NULL_HANDLE) {
        Msg("![VK PreSkin] preskin.comp.spv missing - falling back to per-pass skinning");
        s_psFailed = true; return false;
    }

    // set 0: binding 0 = the SAME bone SSBO the graphics pipelines read,
    //        binding 1 = the shared output pool.
    constexpr auto kSSBO = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    if (!VK::MakeDescriptorSets({ kSSBO, kSSBO }, 1, s_psSetL, s_psPool, &s_psSet,
                                VK_SHADER_STAGE_COMPUTE_BIT, "PreSkin")) {
        s_psFailed = true; return false;
    }

    // gpuOnly=TRUE is mandatory: a STORAGE buffer without it gets
    // HOST_ACCESS_SEQUENTIAL_WRITE, so VMA puts it in the BAR heap or plain
    // system RAM and every read rides PCIe (see vk_buffer.h's warning and the
    // VSM saga). Nothing on the CPU ever touches this pool.
    s_psBuf.Create(VkDeviceSize(kPsBytesSlot) * kFramesInFlight,
                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                   VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, /*gpuOnly*/ true);
    if (!s_psBuf.IsValid()) { Msg("![VK PreSkin] pool alloc failed"); s_psFailed = true; return false; }
    Prof::NameBuffer(s_psBuf.GetHandle(), "PreSkinPool");

    VK::DescriptorWriter(s_psSet)
        .StorageBuffer(0, s_boneSSBO.GetHandle())
        .StorageBuffer(1, s_psBuf.GetHandle())
        .Flush();

    s_psLayout = VK::MakePipelineLayout({ s_psSetL }, sizeof(PreSkinPush));
    if (s_psLayout == VK_NULL_HANDLE) { s_psFailed = true; return false; }

    s_psPipe = VK::CreateComputePipeline(cs, s_psLayout, "PreSkin");
    if (s_psPipe == VK_NULL_HANDLE) { s_psFailed = true; return false; }

    Msg("[VK PreSkin] init OK (pool %u MB/slot x%u = %u verts/slot, stride %u)",
        kPsBytesSlot >> 20, kFramesInFlight, kPsVertsSlot, kPsStride);
    return true;
}

}  // anon namespace

// Skin every visible leaf ONCE into the shared pool. Registered as the "PreSkin"
// pass so it runs before Pass_SunShadow (the earliest consumer) and outside any
// dynamic-rendering scope. Leaves that don't fit keep the classic per-pass
// skinning: s_psMap simply has no entry for them.
void Skinned_PreSkin(VkCommandBuffer cmd)
{
    if (!Init()) return;
    if (s_psFrame == Device.dwFrame) return;   // once per frame
    s_psFrame = Device.dwFrame;

    Skinned_UploadBones();   // CPU bones + the identity slot this pass points at

    s_psMap.clear();
    s_psLeaves = s_psVerts = s_psOverflow = s_psSkipped1W = s_psMaxBones = 0;
    if (!ps_r_preskin) return;                             // A/B: fall back to per-pass skinning
    if (s_uploads.empty() && s_uploadsHud.empty()) return;
    if (!PreSkinInit()) return;

    VK_CPU_PROBE("cpu:PreSkin");

    const u32 slot   = CommandManager.GetCurrentFrame();
    u32       cursor = slot * kPsVertsSlot;
    const u32 limit  = cursor + kPsVertsSlot;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_psPipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_psLayout, 0, 1, &s_psSet, 0, nullptr);

    auto skinList = [&](const xr_vector<SkelUpload>& list) {
        for (const SkelUpload& u : list)
        {
            if (u.boneCount > s_psMaxBones) s_psMaxBones = u.boneCount;
            for (auto* child : u.K->children)
            {
                VK_Render_Mesh* mesh = nullptr;
                u16 rmode = 0;
                if (!ResolveSkinnedLeaf(child, mesh, rmode)) continue;
                // Lightplanes fans are never drawn (Pass_LightCones renders the
                // synthesized cone instead) - skinning them would be pure waste.
                if (child->m_bLitBlend) continue;

                // 1-WEIGHT GEOMETRY CAN NEVER WIN HERE — skip it.
                // RM_SINGLE / RM_SKINNING_1B both map to skinMode 1, whose classic
                // vertex shader is already a SINGLE matrix fetch (`S = bones[idx]`).
                // Pre-skinning would replace that with... a single matrix fetch of
                // the identity, and still pay the compute dispatch + pool write.
                // Measured on l01_escape 2026-07-23: 46% of the 583 visible leaves
                // were 1-3 bone props (doors, lamps, physics crates, dropped guns)
                // and they dominated the pool, which is why the first A/B came out
                // net negative. Only a real multi-bone blend (2W/3W/4W — NPCs) has
                // anything to save.
                const u32 skinMode = (rmode <= 2u) ? 1u : (u32(rmode) - 1u);
                if (skinMode < 2u) { ++s_psSkipped1W; continue; }

                const u32 vc = mesh->vCount;
                if (!vc || !mesh->vAddr) continue;         // not addressable -> classic path
                if (cursor + vc > limit) { ++s_psOverflow; continue; }   // pool full -> classic path

                PreSkinPush pc{};
                pc.src         = mesh->vAddr + VkDeviceSize(mesh->vBase) * mesh->vStride;
                pc.vertCount   = vc;
                pc.srcStrideDW = mesh->vStride / 4u;
                pc.dstFirst    = cursor;
                pc.skinMode    = skinMode;
                pc.baseBone    = u.baseBone;
                pc.boneCount   = (u32)u.boneCount;
                vkCmdPushConstants(cmd, s_psLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                vkCmdDispatch(cmd, (vc + 63u) / 64u, 1, 1);

                s_psMap[(const void*)child] = cursor;
                cursor += vc;
                ++s_psLeaves;
            }
        }
    };
    skinList(s_uploads);
    skinList(s_uploadsHud);

    s_psVerts = cursor - slot * kPsVertsSlot;
    if (s_psVerts > s_psPeak) s_psPeak = s_psVerts;

    if (!s_psLeaves) {
        // Nothing multi-bone on screen. Worth saying out loud: it means every
        // skinned leaf this frame was 1-weight prop geometry, so any measurement
        // taken here says nothing about pre-skinning on an NPC crowd.
        static u32 s_idle = 0;
        if (++s_idle % 601 == 1)
            Msg("[VK PreSkin] idle: 0 multi-bone leaves (skipped1W=%u, maxBones=%u) - prop-only scene",
                s_psSkipped1W, s_psMaxBones);
        return;
    }

    // The pool is read as vertex attributes by every consumer from here on.
    VkMemoryBarrier mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);

    static bool s_first = false;
    if (!s_first) { s_first = true;
        Msg("[VK PreSkin] first dispatch: leaves=%u verts=%u (%u%% of slot) skeletons=%zu+%zu",
            s_psLeaves, s_psVerts, (s_psVerts * 100u) / kPsVertsSlot, s_uploads.size(), s_uploadsHud.size()); }
    static u32 s_pulse = 0;
    if (++s_pulse % 601 == 1)
        Msg("[VK PreSkin] leaves=%u verts=%u peak=%u/%u overflow=%u | skipped1W=%u maxBones=%u",
            s_psLeaves, s_psVerts, s_psPeak, kPsVertsSlot, s_psOverflow, s_psSkipped1W, s_psMaxBones);
}

void Skinned_CollectFeet(xr_vector<Fvector>& out, u32 maxFeet)
{
    out.clear();
    if (!s_inited || !s_boneMapped) return;
    // STALKER human bipeds: feet = bip01_l_foot / bip01_r_foot (fallback l_foot/r_foot).
    // s_boneMapped[baseBone+id] is already model->WORLD, so .c is the world foot pos.
    for (const SkelUpload& u : s_uploads) {
        if (!u.K) continue;
        u16 lf = u.K->LL_BoneID("bip01_l_foot"); if (lf == BI_NONE) lf = u.K->LL_BoneID("l_foot");
        u16 rf = u.K->LL_BoneID("bip01_r_foot"); if (rf == BI_NONE) rf = u.K->LL_BoneID("r_foot");
        // ONLY real foot bones (bipeds). NO object-root fallback: the stamp is XZ-only
        // (ignores height), so stamping a skeleton-less object's root made FLYING items
        // (thrown bolt/grenade) leave a ground trail in mid-air.
        if (lf != BI_NONE && lf < u.boneCount) { out.push_back(s_boneMapped[u.baseBone + lf].c); if (out.size() >= maxFeet) return; }
        if (rf != BI_NONE && rf < u.boneCount) { out.push_back(s_boneMapped[u.baseBone + rf].c); if (out.size() >= maxFeet) return; }
    }
}

// DISCRETE footstep detection: a walking foot alternates SWING (fast in world space)
// and STANCE (planted, ~zero world velocity). Track each foot across frames and emit
// ONE event on the transition into stance away from its previous plant â€” separate
// boot prints per step instead of the dragged trench that per-frame stamping made.
namespace {
    struct FootTrack {
        Fvector lastPos{};      // world pos last frame
        Fvector lastPlant{};    // where this foot last printed
        int     settle  = -1;   // frames until the pending print is emitted (-1 = none)
        bool    planted = false;
        bool    init    = false;
        u32     frame   = 0;    // last seen (prune)
    };
    xr_map<const void*, FootTrack> s_footTracks;
}

void Skinned_CollectFootsteps(xr_vector<Footstep>& out, u32 maxSteps)
{
    out.clear();
    if (!s_inited || !s_boneMapped) return;
    const float dt = (Device.fTimeDelta > 1e-4f) ? Device.fTimeDelta : 1e-4f;
    constexpr float kPlantSpeed  = 0.45f;   // stance foot is ~stationary in world space (m/s)
    constexpr float kLiftSpeed   = 0.90f;   // swing foot moves fast -> re-arm for the next plant
    constexpr float kReplantDist = 0.28f;   // min XZ travel from the previous print (no idle pile-up)

    for (const SkelUpload& u : s_uploads) {
        if (!u.K) continue;
        u16 lf = u.K->LL_BoneID("bip01_l_foot"); if (lf == BI_NONE) lf = u.K->LL_BoneID("l_foot");
        u16 rf = u.K->LL_BoneID("bip01_r_foot"); if (rf == BI_NONE) rf = u.K->LL_BoneID("r_foot");
        u16 lt = u.K->LL_BoneID("bip01_l_toe0"); if (lt == BI_NONE) lt = u.K->LL_BoneID("l_toe");
        u16 rt = u.K->LL_BoneID("bip01_r_toe0"); if (rt == BI_NONE) rt = u.K->LL_BoneID("r_toe");
        Fvector fwd = u.xform.k; fwd.y = 0.f;   // body facing (fallback boot orientation)
        if (fwd.square_magnitude() > 1e-4f) fwd.normalize(); else fwd.set(0.f, 0.f, 1.f);
        const u16 feet[2] = { lf, rf };
        const u16 toes[2] = { lt, rt };
        for (int s = 0; s < 2; ++s) {
            const u16 id = feet[s];
            if (id == BI_NONE || id >= u.boneCount) continue;
            const Fvector pos = s_boneMapped[u.baseBone + id].c;
            FootTrack& t = s_footTracks[(const void*)(((uintptr_t)u.K << 1) | (uintptr_t)s)];
            t.frame = Device.dwFrame;
            if (!t.init) { t.init = true; t.planted = true; t.lastPos = pos; t.lastPlant = pos; continue; }
            const float dx = pos.x - t.lastPos.x, dz = pos.z - t.lastPos.z;
            const float speed = sqrtf(dx * dx + dz * dz) / dt;
            if (speed > kLiftSpeed) {
                t.planted = false; t.settle = -1;        // swinging -> armed, cancel any pending print
            } else if (speed < kPlantSpeed) {
                if (!t.planted) {
                    // Heel-strike detected — but the foot is still decelerating/rotating
                    // down for a few cm. Don't print yet: let it fully SETTLE and stamp
                    // at the rested position a few frames later.
                    const float px = pos.x - t.lastPlant.x, pz = pos.z - t.lastPlant.z;
                    if (px * px + pz * pz > kReplantDist * kReplantDist) t.settle = 3;
                    t.planted = true;
                } else if (t.settle > 0) {
                    --t.settle;
                } else if (t.settle == 0) {
                    t.settle = -1;
                    if (out.size() < maxSteps) {
                        // The foot bone pivot is the ANKLE — above the boot rear, off the
                        // sole centre. Centre the print on the actual BOOT via the toe
                        // bone: sole centre ≈ ankle↔toe midpoint, ankle→toe = this foot's
                        // TRUE facing (feet splay while walking, not the body's k).
                        Fvector p = pos; float fx = fwd.x, fz = fwd.z;
                        const u16 tid = toes[s];
                        const bool haveToe = (tid != BI_NONE && tid < u.boneCount);
                        if (haveToe) {
                            const Fvector toe = s_boneMapped[u.baseBone + tid].c;
                            const float tx = toe.x - pos.x, tz = toe.z - pos.z;
                            const float tl = sqrtf(tx * tx + tz * tz);
                            if (tl > 0.04f) { fx = tx / tl; fz = tz / tl; }
                            p.x = 0.5f * (pos.x + toe.x); p.z = 0.5f * (pos.z + toe.z);
                        } else {
                            p.x += fx * 0.06f; p.z += fz * 0.06f;   // no toe bone: nudge ahead of the ankle
                        }
                        static int s_plantLog = 0;   // one-shot diag: is the toe bone actually found?
                        if (s_plantLog < 6) { ++s_plantLog;
                            Msg("[VK Steps] plant %s toe=%s ankle=(%.2f,%.2f) print=(%.2f,%.2f) dir=(%.2f,%.2f)",
                                s ? "R" : "L", haveToe ? "yes" : "NO", pos.x, pos.z, p.x, p.z, fx, fz);
                        }
                        out.push_back({ p, fx, fz });
                        t.lastPlant = pos;
                    }
                }
            }
            t.lastPos = pos;
        }
    }
    // Prune stale skeletons (despawn / level change) so the map doesn't grow forever.
    if (s_footTracks.size() > 160) {
        for (auto it = s_footTracks.begin(); it != s_footTracks.end(); )
            it = (Device.dwFrame - it->second.frame > 600) ? s_footTracks.erase(it) : ++it;
    }
}

// World PROPS / items (skeleton-less skinned visuals: thrown/dropped bolts, grenades,
// debris). Returns their world root. The caller gates these by ground proximity (so a
// FLYING item doesn't stamp) and prints them SHALLOW (an item sinks less than a boot).
void Skinned_CollectProps(xr_vector<Fvector>& out, u32 maxProps)
{
    out.clear();
    if (!s_inited || !s_boneMapped) return;
    for (const SkelUpload& u : s_uploads) {
        if (!u.K) continue;
        u16 lf = u.K->LL_BoneID("bip01_l_foot"); if (lf == BI_NONE) lf = u.K->LL_BoneID("l_foot");
        u16 rf = u.K->LL_BoneID("bip01_r_foot"); if (rf == BI_NONE) rf = u.K->LL_BoneID("r_foot");
        if (lf != BI_NONE || rf != BI_NONE) continue;   // has feet -> biped, handled by CollectFeet
        out.push_back(u.xform.c);                        // object root (world)
        if (out.size() >= maxProps) return;
    }
}

// Same filter uploadList() uses to decide what this path owns. Kept next to it so the
// rigid sun-caster pass (vk_pass_shadow.cpp) can ask instead of repeating the test —
// change the filter and both follow.
bool Skinned_HandlesVisual(IRenderVisual* v)
{
    if (!v)
        return false;
    CKinematics* K = dynamic_cast<CKinematics*>(v);
    return K && !K->children.empty();
}

void Skinned_RenderShadow(VkCommandBuffer cmd, const Fmatrix& lightVP,
                          const Fvector* cullPos, float cullRange, const CFrustum* frustum)
{
    if (!s_inited || s_failed || s_shadowVS == VK_NULL_HANDLE) return;
    if (s_uploads.empty()) return;   // HUD never casts

    VkPipeline lastPipe = VK_NULL_HANDLE;
    bool boundBones = false;
    u32 nDraw = 0;

    for (const SkelUpload& u : s_uploads)
    {
        // Cull: sun ortho box by default, light sphere for spot/point maps.
        const Fsphere& bs = u.K->vis.sphere;
        if (bs.R > 0.f) {
            Fvector c; u.xform.transform_tiny(c, bs.P);
            if (cullPos) {
                const float rr = cullRange + bs.R;
                if (cullPos->distance_to_sqr(c) > rr * rr) continue;
                // Per-face frustum (point cube): skip the ~4 of 6 faces the NPC
                // isn't in — the sphere test alone drew it into all six.
                if (frustum && !frustum->testSphere_dirty(c, bs.R)) continue;
            } else if (!ShadowMap::SphereVisible(c, bs.R)) continue;
        }

        for (auto* child : u.K->children)
        {
            VK_Render_Mesh* mesh = nullptr;
            u16 rmode = 0;
            if (!ResolveSkinnedLeaf(child, mesh, rmode)) continue;
            if (child->m_bEmissiveAdd || child->m_bModelGlass || child->m_bLitBlend) continue;   // marks/glass don't cast shadows

            const DrawSrc ds = ResolveDrawSrc(child, mesh, rmode, u.baseBone, u.boneCount);
            VkPipeline pipe = GetShadowPipeline(ds.stride);
            if (pipe == VK_NULL_HANDLE) continue;

            if (pipe != lastPipe) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
                lastPipe = pipe;
                boundBones = false;
            }
            if (!boundBones) {
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_layout, 0, 1, &s_set, 0, nullptr);
                boundBones = true;
            }

            SkinPush pc{ lightVP, ds.skinMode, ds.baseBone, ds.boneCount, 0.0f, 1.0f };
            vkCmdPushConstants(cmd, s_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

            VkDeviceSize vbOff = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &ds.vb, &vbOff);
            vkCmdBindIndexBuffer(cmd, mesh->p_rm_Indices->GetHandle(), 0, mesh->iType);
            vkCmdDrawIndexed(cmd, mesh->iCount, 1, mesh->iBase, ds.firstVertex, 0);
            ++nDraw;
        }
    }

    static bool s_diag = false;
    if (!s_diag && nDraw) { s_diag = true; Msg("[VK Skinned] first shadow render: skeletons=%zu draws=%u", s_uploads.size(), nDraw); }
}

// NPCs into the scene depth PREPASS (camera VP): they become GTAO occluders â€”
// contact darkening on the ground under characters, like R4 where skinned
// geometry is in the gbuffer â€” and the world color pass gets early-Z behind
// them. Alpha-tested (a < 0.25, same as skinned.frag) so hair/strap cutouts
// don't punch holes into the early-Z'd background. HUD never enters (s_uploads
// only). Caller owns render begin/end and viewport (Pass_World prepass).
// Shared draw loop for the near-camera skinned casters (depth prepass + normal
// G-buffer differ only by which pipeline they bind, picked via `getPipe`).
static void RenderSkinnedCasters(VkCommandBuffer cmd, const Fmatrix& viewProj, VkPipeline (*getPipe)(u32))
{
    // AO is short-range â€” only NPCs near the camera matter as occluders.
    constexpr float kPrepassRange = 60.f;

    VkPipeline      lastPipe   = VK_NULL_HANDLE;
    VkDescriptorSet lastMatSet = VK_NULL_HANDLE;
    bool boundBones = false;

    for (const SkelUpload& u : s_uploads)
    {
        const Fsphere& bs = u.K->vis.sphere;
        if (bs.R > 0.f) {
            Fvector c; u.xform.transform_tiny(c, bs.P);
            const float rr = kPrepassRange + bs.R;
            if (Device.vCameraPosition.distance_to_sqr(c) > rr * rr) continue;
        }

        for (auto* child : u.K->children)
        {
            VK_Render_Mesh* mesh = nullptr;
            u16 rmode = 0;
            if (!ResolveSkinnedLeaf(child, mesh, rmode)) continue;
            if (child->m_bEmissiveAdd || child->m_bModelGlass || child->m_bLitBlend) continue;   // marks/glass: blended, no depth

            const DrawSrc ds = ResolveDrawSrc(child, mesh, rmode, u.baseBone, u.boneCount);
            VkPipeline pipe = getPipe(ds.stride);
            if (pipe == VK_NULL_HANDLE) continue;

            if (pipe != lastPipe) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
                lastPipe = pipe;
                boundBones = false;
                lastMatSet = VK_NULL_HANDLE;
            }
            if (!boundBones) {
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_layout, 0, 1, &s_set, 0, nullptr);
                boundBones = true;
            }

            // Diffuse (set 1) for the alpha test â€” default white (a=1) never discards.
            WorldMaterial* mat = child->m_pWorldMaterial ? child->m_pWorldMaterial : WorldMaterialCache::GetDefault();
            VkDescriptorSet matSet = (mat && mat->set != VK_NULL_HANDLE) ? mat->set : VK_NULL_HANDLE;
            if (matSet == VK_NULL_HANDLE) continue;
            if (matSet != lastMatSet) {
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_layout, 1, 1, &matSet, 0, nullptr);
                lastMatSet = matSet;
            }

            SkinPush pc{ viewProj, ds.skinMode, ds.baseBone, ds.boneCount, 0.0f, 1.0f };
            vkCmdPushConstants(cmd, s_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

            VkDeviceSize vbOff = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &ds.vb, &vbOff);
            vkCmdBindIndexBuffer(cmd, mesh->p_rm_Indices->GetHandle(), 0, mesh->iType);
            vkCmdDrawIndexed(cmd, mesh->iCount, 1, mesh->iBase, ds.firstVertex, 0);
        }
    }
}

void Skinned_RenderDepthPrepass(VkCommandBuffer cmd, const Fmatrix& viewProj)
{
    if (!Init()) return;
    if (s_prepassVS == VK_NULL_HANDLE || s_prepassFS == VK_NULL_HANDLE) return;
    Skinned_UploadBones();   // idempotent; Pass_SunShadow usually did it already
    if (s_uploads.empty()) return;
    RenderSkinnedCasters(cmd, viewProj, &GetPrepassPipeline);
}

// NPC normal G-buffer: same near-camera casters, writing world-space normals
// into the SSAO normal RT (depth-tested against the prepass depth, no write).
// Caller owns the render begin/end + viewport (Pass_World, after the prepass).
void Skinned_RenderNormalPrepass(VkCommandBuffer cmd, const Fmatrix& viewProj)
{
    if (!Init()) return;
    if (s_normalVS == VK_NULL_HANDLE || s_normalFS == VK_NULL_HANDLE) return;
    Skinned_UploadBones();
    if (s_uploads.empty()) return;
    RenderSkinnedCasters(cmd, viewProj, &GetNormalPipeline);
}

// NPC MOTION VECTORS (MV Phase 2a): re-draw the world skinned leaves into the
// motion target, skinning the CURRENT and PREVIOUS pose so each pixel carries its
// true screen motion (camera + animation + body travel). Depth-tested against the
// complete scene depth (LEQUAL, no write) â†’ only visible NPC pixels overwrite the
// camera/static field. Caller owns render begin/end + the negative-height viewport
// (MotionVec::ExecuteDynamic). HUD excluded. prevVP = last frame's view-proj.
void Skinned_RenderMotion(VkCommandBuffer cmd, const Fmatrix& curVP, const Fmatrix& prevVP,
                          float jitterNdcX, float jitterNdcY)
{
    if (!Init()) return;
    if (s_mvVS == VK_NULL_HANDLE || s_mvFS == VK_NULL_HANDLE || s_mvLayout == VK_NULL_HANDLE) return;
    Skinned_UploadBones();   // idempotent; also rolls the prev-bone map
    if (s_uploads.empty()) return;

    VkPipeline lastPipe = VK_NULL_HANDLE;
    VkDescriptorSet lastMatSet = VK_NULL_HANDLE;
    bool boundBones = false;
    u32 nDraw = 0;

    for (const SkelUpload& u : s_uploads)
    {
        // Previous pose: same skeleton last frame. Its SSBO region is still live
        // (FRAMES_IN_FLIGHTâ‰¥2). New skeletons (not seen last frame, or whose bone
        // count changed) fall back to the current base â†’ camera-only motion.
        u32 prevBase = u.baseBone;
        auto it = s_prevBoneMap.find(u.K);
        if (it != s_prevBoneMap.end() && it->second.second == (u32)u.boneCount)
            prevBase = it->second.first;

        for (auto* child : u.K->children)
        {
            VK_Render_Mesh* mesh = nullptr;
            u16 rmode = 0;
            if (!ResolveSkinnedLeaf(child, mesh, rmode)) continue;
            if (child->m_bEmissiveAdd || child->m_bModelGlass || child->m_bLitBlend) continue;   // marks/glass: no depth, skip

            VkPipeline pipe = GetMotionPipeline(mesh->vStride);
            if (pipe == VK_NULL_HANDLE) continue;
            if (pipe != lastPipe) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
                lastPipe = pipe; boundBones = false; lastMatSet = VK_NULL_HANDLE;
            }
            if (!boundBones) {
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_mvLayout, 0, 1, &s_set, 0, nullptr);
                boundBones = true;
            }

            // Diffuse (set 1) for the alpha-test â€” default white (a=1) never discards.
            WorldMaterial* mat = child->m_pWorldMaterial ? child->m_pWorldMaterial : WorldMaterialCache::GetDefault();
            VkDescriptorSet matSet = (mat && mat->set != VK_NULL_HANDLE) ? mat->set : VK_NULL_HANDLE;
            if (matSet == VK_NULL_HANDLE) continue;
            if (matSet != lastMatSet) {
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_mvLayout, 1, 1, &matSet, 0, nullptr);
                lastMatSet = matSet;
            }

            const u32 skinMode = (rmode <= 2u) ? 1u : (u32(rmode) - 1u);
            MVSkinPush pc{ curVP, prevVP, skinMode, u.baseBone, prevBase, (u32)u.boneCount, jitterNdcX, jitterNdcY };
            vkCmdPushConstants(cmd, s_mvLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);

            VkBuffer vb = mesh->p_rm_Vertices->GetHandle();
            VkDeviceSize vbOff = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &vbOff);
            vkCmdBindIndexBuffer(cmd, mesh->p_rm_Indices->GetHandle(), 0, mesh->iType);
            vkCmdDrawIndexed(cmd, mesh->iCount, 1, mesh->iBase, (s32)mesh->vBase, 0);
            ++nDraw;
        }
    }

    static bool s_diag = false;
    if (!s_diag && nDraw) { s_diag = true; Msg("[VK Skinned] first MV render: skeletons=%zu draws=%u", s_uploads.size(), nDraw); }
}

// First-person HUD weapon MV overlay. Same mechanism as Skinned_RenderMotion, but
// for the HUD list and with the HUD-FOV projection (curHudVP/prevHudVP) at the
// near-depth viewport [0,0.02] the forward HUD pass used — so the weapon's clip
// depth bit-matches the HUD depth already in the buffer and LEQUAL passes. Without
// this the viewmodel (the closest, fastest-moving thing on screen) carries the
// fullscreen pass's bogus camera-reprojection MV and would ghost under DLSS/TAA.
// Caller (MotionVec::ExecuteDynamic) owns render begin/end; we restore the range.
void Skinned_RenderMotionHud(VkCommandBuffer cmd, const VkExtent2D& ext,
                             const Fmatrix& curHudVP, const Fmatrix& prevHudVP,
                             float jitterNdcX, float jitterNdcY)
{
    if (!Init()) return;
    if (s_mvVS == VK_NULL_HANDLE || s_mvFS == VK_NULL_HANDLE || s_mvLayout == VK_NULL_HANDLE) return;
    Skinned_UploadBones();   // idempotent; ensures s_uploadsHud + the prev HUD map
    if (s_uploadsHud.empty()) return;

    SetViewportDepth(cmd, ext, 0.0f, 0.02f);   // match the forward HUD near-depth range

    VkPipeline      lastPipe   = VK_NULL_HANDLE;
    VkDescriptorSet lastMatSet = VK_NULL_HANDLE;
    bool            boundBones = false;
    u32             nDraw      = 0;

    for (const SkelUpload& u : s_uploadsHud)
    {
        // Previous pose: same HUD skeleton last frame (SSBO region still live).
        u32 prevBase = u.baseBone;
        auto it = s_prevBoneMapHud.find(u.K);
        if (it != s_prevBoneMapHud.end() && it->second.second == (u32)u.boneCount)
            prevBase = it->second.first;

        for (auto* child : u.K->children)
        {
            VK_Render_Mesh* mesh = nullptr;
            u16 rmode = 0;
            if (!ResolveSkinnedLeaf(child, mesh, rmode)) continue;
            if (child->m_bEmissiveAdd || child->m_bModelGlass || child->m_bLitBlend) continue;   // marks/glass: skip

            VkPipeline pipe = GetMotionPipeline(mesh->vStride);
            if (pipe == VK_NULL_HANDLE) continue;
            if (pipe != lastPipe) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
                lastPipe = pipe; boundBones = false; lastMatSet = VK_NULL_HANDLE;
            }
            if (!boundBones) {
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_mvLayout, 0, 1, &s_set, 0, nullptr);
                boundBones = true;
            }

            WorldMaterial* mat = child->m_pWorldMaterial ? child->m_pWorldMaterial : WorldMaterialCache::GetDefault();
            VkDescriptorSet matSet = (mat && mat->set != VK_NULL_HANDLE) ? mat->set : VK_NULL_HANDLE;
            if (matSet == VK_NULL_HANDLE) continue;
            if (matSet != lastMatSet) {
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_mvLayout, 1, 1, &matSet, 0, nullptr);
                lastMatSet = matSet;
            }

            const u32 skinMode = (rmode <= 2u) ? 1u : (u32(rmode) - 1u);
            MVSkinPush pc{ curHudVP, prevHudVP, skinMode, u.baseBone, prevBase, (u32)u.boneCount, jitterNdcX, jitterNdcY };
            vkCmdPushConstants(cmd, s_mvLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);

            VkBuffer vb = mesh->p_rm_Vertices->GetHandle();
            VkDeviceSize vbOff = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &vbOff);
            vkCmdBindIndexBuffer(cmd, mesh->p_rm_Indices->GetHandle(), 0, mesh->iType);
            vkCmdDrawIndexed(cmd, mesh->iCount, 1, mesh->iBase, (s32)mesh->vBase, 0);
            ++nDraw;
        }
    }

    SetViewportDepth(cmd, ext, 0.0f, 1.0f);   // restore full-depth range

    static bool s_diagHud = false;
    if (!s_diagHud && nDraw) { s_diagHud = true; Msg("[VK Skinned] first HUD MV render: huds=%zu draws=%u", s_uploadsHud.size(), nDraw); }
}

bool Skinned_AnyCasterInSphere(const Fvector& pos, float range)
{
    for (const SkelUpload& u : s_uploads)
    {
        const Fsphere& bs = u.K->vis.sphere;
        Fvector c; u.xform.transform_tiny(c, bs.P);
        const float rr = range + (bs.R > 0.f ? bs.R : 1.f);
        if (pos.distance_to_sqr(c) <= rr * rr) return true;
    }
    return false;
}

// VSM skinned casters â€” one entry per visible world skinned leaf, for vk_vsm to bin +
// rasterize into the virtual shadow atlas. Coarse skeleton world sphere per leaf (the
// binner over-covers slightly, which is harmless â€” extra pages just clip). HUD excluded.
void Skinned_CollectCasters(xr_vector<VsmSkinnedCaster>& out)
{
    if (!Init()) return;
    Skinned_UploadBones();   // idempotent per Device.dwFrame
    const float npcDist = ps_r_vsm_npc_dist;   // 0 = no cull (old behaviour)
    for (const SkelUpload& u : s_uploads)
    {
        const Fsphere& bs = u.K->vis.sphere;
        Fvector c; u.xform.transform_tiny(c, bs.P);
        const float r = (bs.R > 0.f) ? bs.R : 1.f;
        // Distant NPCs cast a few-texel shadow â†’ skip the whole skeleton: saves the per-leaf
        // skinning + VSM binning AND keeps us under kMaxSkinned so NEAR NPCs are never dropped
        // when a crowd would otherwise overflow the cap. (Collect had NO cull before â†’ it
        // saturated at 256 leaves; log showed only ~40 actually cast.)
        if (npcDist > 0.f) {
            const float rr = npcDist + r;
            if (Device.vCameraPosition.distance_to_sqr(c) > rr * rr) continue;
        }
        for (auto* child : u.K->children)
        {
            VK_Render_Mesh* mesh = nullptr; u16 rmode = 0;
            if (!ResolveSkinnedLeaf(child, mesh, rmode)) continue;
            if (child->m_bEmissiveAdd) continue;   // collimator marks don't cast
            const DrawSrc ds = ResolveDrawSrc(child, mesh, rmode, u.baseBone, u.boneCount);
            VsmSkinnedCaster sc{};
            sc.sphere_P     = c;            sc.sphere_R = r;
            sc.index_count  = mesh->iCount; sc.ib_first = mesh->iBase; sc.first_vertex = ds.firstVertex;
            sc.vb           = ds.vb;
            sc.ib           = mesh->p_rm_Indices->GetHandle();
            sc.iType        = mesh->iType;  sc.stride   = ds.stride;
            sc.base_bone    = ds.baseBone;  sc.bone_count = ds.boneCount;
            sc.skin_mode    = ds.skinMode;
            out.push_back(sc);
        }
    }
}

VkDescriptorSet       Skinned_GetBoneSet()       { return s_set; }
VkDescriptorSetLayout Skinned_GetBoneSetLayout() { return s_setLayout; }

void Skinned_BuildVertexInput(u32 stride, VkVertexInputBindingDescription& binding,
                              VkVertexInputAttributeDescription attrs[6])
{
    BuildSkinnedVI(stride, binding, attrs);
}

void Pass_Skinned(FrameContext& ctx)
{
    if (g_DynamicVisuals.empty() && g_HudVisuals.empty()) return;
    if (ctx.cmd == VK_NULL_HANDLE || !ctx.viewProj) return;
    if (!Init()) return;

    // Normally a no-op â€” Pass_SunShadow already uploaded this frame's bones.
    Skinned_UploadBones();

    VkCommandBuffer cmd = ctx.cmd;
    VkPipeline      lastPipe   = VK_NULL_HANDLE;
    VkDescriptorSet lastMatSet = VK_NULL_HANDLE;
    u32 nDraw = 0, nHud = 0;

    // Per-frame env lighting (set 2). Pass_World already called EnvLight::Update this
    // frame (it runs before this pass), so just grab the current set; fall back to
    // updating here if this pass ever runs standalone.
    // ⚠No command buffer on purpose: this pass records inside dynamic rendering, where
    // the sky-probe prefilter's dispatch/blits would be illegal. The UBO fill is all
    // that is wanted here; the probe keeps last frame's content.
    if (EnvLight::GetCurrentSet() == VK_NULL_HANDLE) EnvLight::Update(CommandManager.GetCurrentFrame());
    VkDescriptorSet lightSet = EnvLight::GetCurrentSet();

    // World dynamics: bones are world-space â†’ push plain viewProj.
    DrawSkinnedList(cmd, s_uploads, *ctx.viewProj, nDraw, lastPipe, lastMatSet, lightSet, 0.0f);

    // First-person HUD: HUD-FOV projection (camera at origin) + near depth range so
    // hands/weapon render on top of the world. Restore the normal range afterwards.
    // hudMode=1 â†’ fragment uses flat, sun-direction-independent lighting (view-space bones).
    if (!s_uploadsHud.empty()) {
        SetViewportDepth(cmd, ctx.extent, 0.0f, 0.02f);
        DrawSkinnedList(cmd, s_uploadsHud, Device.mFullTransform_hud, nHud, lastPipe, lastMatSet, lightSet, 1.0f);
        SetViewportDepth(cmd, ctx.extent, 0.0f, 1.0f);
    }

    static bool s_diag = false;
    if (!s_diag) { s_diag = true; Msg("[VK Skinned] first Pass_Skinned: dynVis=%zu hud=%zu draws=%u hudDraws=%u (slot=%u)",
                                      s_uploads.size(), s_uploadsHud.size(), nDraw, nHud, CommandManager.GetCurrentFrame()); }
    // pulse diag (two-player invisibility hunt): proves per-frame whether the
    // main view collected any dynamics and whether their leaves really drew.
    // Odd modulus on purpose -- even throttles latch with two clients.
    static u32 s_pulse = 0;
    if (++s_pulse % 127 == 1)
        Msg("[VK Skinned] pulse: dynVis=%zu hud=%zu ups=%zu draws=%u", g_DynamicVisuals.size(), g_HudVisuals.size(), s_uploads.size(), nDraw);
}

void Skinned_Destroy()
{
    s_footTracks.clear();   // keys are IKinematics* â€” drop before pointers recycle
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
    for (auto& kv : s_pipelines) if (kv.second) vkDestroyPipeline(VulkanHW.m_Device, kv.second, nullptr);
    s_pipelines.clear();
    s_glassDistFS = VK_NULL_HANDLE;   // module owned by g_ShaderManager
    for (auto& kv : s_shadowPipelines) if (kv.second) vkDestroyPipeline(VulkanHW.m_Device, kv.second, nullptr);
    s_shadowPipelines.clear();
    for (auto& kv : s_prepassPipelines) if (kv.second) vkDestroyPipeline(VulkanHW.m_Device, kv.second, nullptr);
    s_prepassPipelines.clear();
    for (auto& kv : s_normalPipelines) if (kv.second) vkDestroyPipeline(VulkanHW.m_Device, kv.second, nullptr);
    s_normalPipelines.clear();
    for (auto& kv : s_motionPipelines) if (kv.second) vkDestroyPipeline(VulkanHW.m_Device, kv.second, nullptr);
    s_motionPipelines.clear();
    s_shadowVS = VK_NULL_HANDLE;   // module owned by g_ShaderManager
    s_prepassVS = VK_NULL_HANDLE; s_prepassFS = VK_NULL_HANDLE;
    s_mvVS = VK_NULL_HANDLE; s_mvFS = VK_NULL_HANDLE;
    s_curBoneMap.clear(); s_prevBoneMap.clear();
    s_curBoneMapHud.clear(); s_prevBoneMapHud.clear();
    s_uploads.clear(); s_uploadsHud.clear(); s_uploadFrame = u32(-1);

    // Compute pre-skinning. s_psMap MUST be dropped here: its keys are leaf
    // visuals that a level unload frees, and its values index a pool that no
    // longer exists.
    s_psMap.clear(); s_psFrame = u32(-1);
    s_psLeaves = s_psVerts = s_psPeak = s_psOverflow = 0;
    if (s_psPipe)   { vkDestroyPipeline(VulkanHW.m_Device, s_psPipe, nullptr); s_psPipe = VK_NULL_HANDLE; }
    if (s_psLayout) { vkDestroyPipelineLayout(VulkanHW.m_Device, s_psLayout, nullptr); s_psLayout = VK_NULL_HANDLE; }
    if (s_psPool)   { vkDestroyDescriptorPool(VulkanHW.m_Device, s_psPool, nullptr); s_psPool = VK_NULL_HANDLE; }
    if (s_psSetL)   { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_psSetL, nullptr); s_psSetL = VK_NULL_HANDLE; }
    s_psBuf.Destroy();
    s_psSet = VK_NULL_HANDLE;
    s_psInited = false; s_psFailed = false;

    if (s_mvLayout)  { vkDestroyPipelineLayout(VulkanHW.m_Device, s_mvLayout, nullptr); s_mvLayout = VK_NULL_HANDLE; }
    if (s_layout)    { vkDestroyPipelineLayout(VulkanHW.m_Device, s_layout, nullptr); s_layout = VK_NULL_HANDLE; }
    if (s_pool)      { vkDestroyDescriptorPool(VulkanHW.m_Device, s_pool, nullptr); s_pool = VK_NULL_HANDLE; }
    if (s_setLayout) { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_setLayout, nullptr); s_setLayout = VK_NULL_HANDLE; }
    // Set 2 (EnvLight) is owned by vk_env_light.cpp â€” destroyed separately in DevRender::Destroy.
    s_boneSSBO.Destroy();
    s_boneMapped = nullptr;
    s_set = VK_NULL_HANDLE;
    s_inited = false; s_failed = false;
}

}  // namespace VK
