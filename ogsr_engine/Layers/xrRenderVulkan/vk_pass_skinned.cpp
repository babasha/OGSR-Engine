// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan - Skinned-mesh pass (STEP B sub-step 2). See vk_pass_skinned.h.
#include "stdafx.h"

// CKinematics (children + LL_GetTransform_R/LL_BoneCount) â€” same compat trick as
// the skeleton wrapper TUs. Keep dxRender_Visual mapped (children is xr_vector<dxRender_Visual*>).
#define FBasicVisualH
#include "vk_FBasicVisual.h"            // pulls vk_SkeletonCompat.h (dxRender_Visual->vkRender_Visual, vk_Visual.h)
#include "CRender_Vulkan.h"
#include "../xrRender/SkeletonCustom.h" // CKinematics

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
#include "vk_env_light.h"             // EnvLight â€” shared per-frame sun/hemi/ambient UBO (set 2)
#include "vk_shadow.h"                // ShadowMap::SphereVisible â€” caster culling
#include "../../xr_3da/device.h"      // Device.mFullTransform_hud (HUD projection), dwFrame

#include <unordered_map>

// Console cvar at GLOBAL scope â€” a block-scope extern inside namespace VK would mangle as
// VK::ps_r_vsm_npc_dist -> LNK2001 (same trick as vk_pass_shadow's externs).
extern float ps_r_vsm_npc_dist;   // VSM NPC shadow cull distance (m); 0 = no cull

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
    VkPipelineLayout      s_mvLayout   = VK_NULL_HANDLE;  // set0 = bones + 144B push (cur/prev VP + bases)
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
    // Set 2 (per-frame sun/hemi/ambient) is the shared EnvLight set â€” see vk_env_light.{h,cpp}.

    // Push block â€” must match skinned.{vert,frag}.glsl PushConstants. hudMode (read
    // by the fragment) flags first-person HUD so it gets sun-direction-independent light.
    struct SkinPush { Fmatrix mvp; u32 skinMode; u32 baseBone; u32 boneCount; float hudMode; float hemi; };

    // Motion-vector push â€” must match motion_vec_skinned.vert. 144 B (device max is
    // 256 here, see vk_pipeline.cpp). curVP/prevVP project the cur/prev poses.
    struct MVSkinPush { Fmatrix curVP; Fmatrix prevVP; u32 skinMode; u32 curBase; u32 prevBase; u32 boneCount; };
    static_assert(sizeof(MVSkinPush) == 144, "must match motion_vec_skinned.vert PC block");

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

        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount   = 1;
        vi.pVertexBindingDescriptions      = &binding;
        vi.vertexAttributeDescriptionCount = 6;
        vi.pVertexAttributeDescriptions    = attrs;

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;   stages[0].module = s_vs; stages[0].pName = "main";
        stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = distort ? s_glassDistFS : s_fs; stages[1].pName = "main";

        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vp{};
        vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vp.viewportCount = 1; vp.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode    = VK_CULL_MODE_NONE;
        rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth   = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable  = VK_TRUE;
        ds.depthWriteEnable = (additive || glass || distort) ? VK_FALSE : VK_TRUE;   // marks/glass don't write depth (R4 zb(true,false))
        ds.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineColorBlendAttachmentState ba{};
        ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        ba.blendEnable    = VK_FALSE;
        if (additive) {
            // Collimator marks (R4 hud_reddotsight: blend(srcalpha, one)) â€”
            // the dot ADDS light over the sight glass.
            ba.blendEnable         = VK_TRUE;
            ba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            ba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
            ba.colorBlendOp        = VK_BLEND_OP_ADD;
            ba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            ba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            ba.alphaBlendOp        = VK_BLEND_OP_ADD;
        }
        if (glass || distort) {
            // Translucent pane over whatever is behind (lit output, capped
            // alpha comes from the fragment â€” skinMode bit 32). The distort
            // variant blends the wobble over the haze RT's neutral 0.5.
            ba.blendEnable         = VK_TRUE;
            ba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            ba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
            ba.colorBlendOp        = VK_BLEND_OP_ADD;
            ba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            ba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            ba.alphaBlendOp        = VK_BLEND_OP_ADD;
        }
        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1; cb.pAttachments = &ba;

        VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynState{};
        dynState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynState.dynamicStateCount = 2; dynState.pDynamicStates = dyn;

        // Variant 3 renders into the particle heat-haze RT (RGBA8), not the scene.
        VkFormat colorFormat = distort ? VK_FORMAT_R8G8B8A8_UNORM : VK::SceneColor::Format();
        VkPipelineRenderingCreateInfo prci{};
        prci.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        prci.colorAttachmentCount    = 1;
        prci.pColorAttachmentFormats = &colorFormat;
        prci.depthAttachmentFormat   = Swapchain.m_DepthFormat;

        VkGraphicsPipelineCreateInfo pi{};
        pi.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pi.pNext               = &prci;
        pi.stageCount          = 2;          pi.pStages = stages;
        pi.pVertexInputState   = &vi;        pi.pInputAssemblyState = &ia;
        pi.pViewportState      = &vp;        pi.pRasterizationState = &rs;
        pi.pMultisampleState   = &ms;        pi.pDepthStencilState  = &ds;
        pi.pColorBlendState    = &cb;        pi.pDynamicState       = &dynState;
        pi.layout              = s_layout;

        VkPipeline h = VK_NULL_HANDLE;
        VkResult r = vkCreateGraphicsPipelines(VulkanHW.m_Device, PipelineCache::GetCacheObject(), 1, &pi, nullptr, &h);
        if (r != VK_SUCCESS) { Msg("![VK Skinned] pipeline create failed (%d) stride=%u", r, stride); return VK_NULL_HANDLE; }
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
        VkDescriptorSetLayoutBinding b{};
        b.binding = 0; b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b.descriptorCount = 1; b.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        VkDescriptorSetLayoutCreateInfo slci{};
        slci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        slci.bindingCount = 1; slci.pBindings = &b;
        if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &slci, nullptr, &s_setLayout) != VK_SUCCESS) { s_failed = true; return false; }

        VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
        VkDescriptorPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets = 1; pci.poolSizeCount = 1; pci.pPoolSizes = &ps;
        if (vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &s_pool) != VK_SUCCESS) { s_failed = true; return false; }

        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = s_pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &s_setLayout;
        if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, &s_set) != VK_SUCCESS) { s_failed = true; return false; }

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

        VkDescriptorBufferInfo bi{ s_boneSSBO.GetHandle(), 0, VK_WHOLE_SIZE };
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = s_set; w.dstBinding = 0; w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w.pBufferInfo = &bi;
        vkUpdateDescriptorSets(VulkanHW.m_Device, 1, &w, 0, nullptr);

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

        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount   = 1; vi.pVertexBindingDescriptions   = &binding;
        vi.vertexAttributeDescriptionCount = 6; vi.pVertexAttributeDescriptions = attrs;

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_VERTEX_BIT; stage.module = s_shadowVS; stage.pName = "main";

        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vp{};
        vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vp.viewportCount = 1; vp.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType           = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode     = VK_POLYGON_MODE_FILL;
        rs.cullMode        = VK_CULL_MODE_NONE;
        rs.frontFace       = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth       = 1.0f;
        rs.depthBiasEnable = VK_TRUE;       // dynamic â€” caller sets the same bias as statics

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable  = VK_TRUE;
        ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineColorBlendStateCreateInfo cb{};   // no color attachments
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;

        VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS };
        VkPipelineDynamicStateCreateInfo dynState{};
        dynState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynState.dynamicStateCount = 3; dynState.pDynamicStates = dyn;

        VkPipelineRenderingCreateInfo prci{};
        prci.sType                 = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        prci.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;   // matches the shadow map

        VkGraphicsPipelineCreateInfo pi{};
        pi.sType             = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pi.pNext             = &prci;
        pi.stageCount        = 1;     pi.pStages             = &stage;
        pi.pVertexInputState = &vi;   pi.pInputAssemblyState = &ia;
        pi.pViewportState    = &vp;   pi.pRasterizationState = &rs;
        pi.pMultisampleState = &ms;   pi.pDepthStencilState  = &ds;
        pi.pColorBlendState  = &cb;   pi.pDynamicState       = &dynState;
        pi.layout            = s_layout;

        VkPipeline h = VK_NULL_HANDLE;
        VkResult r = vkCreateGraphicsPipelines(VulkanHW.m_Device, PipelineCache::GetCacheObject(), 1, &pi, nullptr, &h);
        if (r != VK_SUCCESS) { Msg("![VK Skinned] shadow pipeline create failed (%d) stride=%u", r, stride); return VK_NULL_HANDLE; }
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

        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount   = 1; vi.pVertexBindingDescriptions   = &binding;
        vi.vertexAttributeDescriptionCount = 6; vi.pVertexAttributeDescriptions = attrs;

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;   stages[0].module = s_prepassVS; stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = s_prepassFS; stages[1].pName = "main";

        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vp{};
        vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vp.viewportCount = 1; vp.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode    = VK_CULL_MODE_NONE;
        rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth   = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable  = VK_TRUE;
        ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineColorBlendStateCreateInfo cb{};   // no color attachments
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;

        VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynState{};
        dynState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynState.dynamicStateCount = 2; dynState.pDynamicStates = dyn;

        VkPipelineRenderingCreateInfo prci{};
        prci.sType                 = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        prci.depthAttachmentFormat = Swapchain.m_DepthFormat;   // scene depth (prepass gates on D32)

        VkGraphicsPipelineCreateInfo pi{};
        pi.sType             = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pi.pNext             = &prci;
        pi.stageCount        = 2;     pi.pStages             = stages;
        pi.pVertexInputState = &vi;   pi.pInputAssemblyState = &ia;
        pi.pViewportState    = &vp;   pi.pRasterizationState = &rs;
        pi.pMultisampleState = &ms;   pi.pDepthStencilState  = &ds;
        pi.pColorBlendState  = &cb;   pi.pDynamicState       = &dynState;
        pi.layout            = s_layout;

        VkPipeline h = VK_NULL_HANDLE;
        VkResult r = vkCreateGraphicsPipelines(VulkanHW.m_Device, PipelineCache::GetCacheObject(), 1, &pi, nullptr, &h);
        if (r != VK_SUCCESS) { Msg("![VK Skinned] prepass pipeline create failed (%d) stride=%u", r, stride); return VK_NULL_HANDLE; }
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

        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount   = 1; vi.pVertexBindingDescriptions   = &binding;
        vi.vertexAttributeDescriptionCount = 6; vi.pVertexAttributeDescriptions = attrs;

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;   stages[0].module = s_normalVS; stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = s_normalFS; stages[1].pName = "main";

        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vp{};
        vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vp.viewportCount = 1; vp.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode    = VK_CULL_MODE_NONE;
        rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth   = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable  = VK_TRUE;
        ds.depthWriteEnable = VK_FALSE;                  // test only â€” the prepass owns the depth
        ds.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT
                           | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        cba.blendEnable    = VK_FALSE;
        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1; cb.pAttachments = &cba;

        VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynState{};
        dynState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynState.dynamicStateCount = 2; dynState.pDynamicStates = dyn;

        const VkFormat normalFmt = SSAOPass::GetNormalFormat();
        VkPipelineRenderingCreateInfo prci{};
        prci.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        prci.colorAttachmentCount    = 1;
        prci.pColorAttachmentFormats = &normalFmt;
        prci.depthAttachmentFormat   = Swapchain.m_DepthFormat;   // tests the scene prepass depth

        VkGraphicsPipelineCreateInfo pi{};
        pi.sType             = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pi.pNext             = &prci;
        pi.stageCount        = 2;     pi.pStages             = stages;
        pi.pVertexInputState = &vi;   pi.pInputAssemblyState = &ia;
        pi.pViewportState    = &vp;   pi.pRasterizationState = &rs;
        pi.pMultisampleState = &ms;   pi.pDepthStencilState  = &ds;
        pi.pColorBlendState  = &cb;   pi.pDynamicState       = &dynState;
        pi.layout            = s_layout;

        VkPipeline h = VK_NULL_HANDLE;
        VkResult r = vkCreateGraphicsPipelines(VulkanHW.m_Device, PipelineCache::GetCacheObject(), 1, &pi, nullptr, &h);
        if (r != VK_SUCCESS) { Msg("![VK Skinned] normal pipeline create failed (%d) stride=%u", r, stride); return VK_NULL_HANDLE; }
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

        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount   = 1; vi.pVertexBindingDescriptions   = &binding;
        vi.vertexAttributeDescriptionCount = 6; vi.pVertexAttributeDescriptions = attrs;

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;   stages[0].module = s_mvVS; stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = s_mvFS; stages[1].pName = "main";

        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vp{};
        vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vp.viewportCount = 1; vp.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode    = VK_CULL_MODE_NONE;
        rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth   = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable  = VK_TRUE;
        ds.depthWriteEnable = VK_FALSE;                  // scene depth already owns the surface
        ds.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT;   // RG16F motion
        cba.blendEnable    = VK_FALSE;
        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1; cb.pAttachments = &cba;

        VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynState{};
        dynState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynState.dynamicStateCount = 2; dynState.pDynamicStates = dyn;

        VkFormat mvFmt = VK::MotionVec::Format();
        VkPipelineRenderingCreateInfo prci{};
        prci.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        prci.colorAttachmentCount    = 1;
        prci.pColorAttachmentFormats = &mvFmt;
        prci.depthAttachmentFormat   = Swapchain.m_DepthFormat;

        VkGraphicsPipelineCreateInfo pi{};
        pi.sType             = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pi.pNext             = &prci;
        pi.stageCount        = 2;     pi.pStages             = stages;
        pi.pVertexInputState = &vi;   pi.pInputAssemblyState = &ia;
        pi.pViewportState    = &vp;   pi.pRasterizationState = &rs;
        pi.pMultisampleState = &ms;   pi.pDepthStencilState  = &ds;
        pi.pColorBlendState  = &cb;   pi.pDynamicState       = &dynState;
        pi.layout            = s_mvLayout;

        VkPipeline h = VK_NULL_HANDLE;
        VkResult r = vkCreateGraphicsPipelines(VulkanHW.m_Device, PipelineCache::GetCacheObject(), 1, &pi, nullptr, &h);
        if (r != VK_SUCCESS) { Msg("![VK Skinned] motion pipeline create failed (%d) stride=%u", r, stride); return VK_NULL_HANDLE; }
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
                // Lightplanes beams: SAME blend states as glass (variant 2),
                // the fragment picks the R4 model_def_lq formula via bit 64.
                const bool emissive = child->m_bEmissiveAdd;
                const bool glass    = child->m_bModelGlass;
                const bool litblend = child->m_bLitBlend;
                VkPipeline pipe = GetPipeline(mesh->vStride, emissive ? 1u : ((glass || litblend) ? 2u : 0u));
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
                u32 skinMode = (rmode <= 2u) ? 1u : (u32(rmode) - 1u);
                if (emissive) skinMode |= 16u;
                if (glass)    skinMode |= 32u;
                if (litblend) skinMode |= 64u;   // lightplanes: R4 model_def_lq formula
                SkinPush pc{ viewProj, skinMode, u.baseBone, (u32)u.boneCount, hudMode, u.hemi };
                vkCmdPushConstants(cmd, s_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

                VkBuffer vb = mesh->p_rm_Vertices->GetHandle();
                VkDeviceSize vbOff = 0;
                vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &vbOff);
                vkCmdBindIndexBuffer(cmd, mesh->p_rm_Indices->GetHandle(), 0, mesh->iType);
                vkCmdDrawIndexed(cmd, mesh->iCount, 1, mesh->iBase, (s32)mesh->vBase, 0);
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
            VkPipeline pipe = GetPipeline(mesh->vStride, 3u);
            if (pipe == VK_NULL_HANDLE) continue;
            if (pipe != lastPipe) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_layout, 0, 1, &s_set, 0, nullptr);   // set0 = bones
                lastPipe = pipe;
            }
            u32 skinMode = (rmode <= 2u) ? 1u : (u32(rmode) - 1u);
            SkinPush pc{ viewProj, skinMode, u.baseBone, (u32)u.boneCount, 0.0f, strength };
            vkCmdPushConstants(cmd, s_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
            VkBuffer vb = mesh->p_rm_Vertices->GetHandle();
            VkDeviceSize vbOff = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &vbOff);
            vkCmdBindIndexBuffer(cmd, mesh->p_rm_Indices->GetHandle(), 0, mesh->iType);
            vkCmdDrawIndexed(cmd, mesh->iCount, 1, mesh->iBase, (s32)mesh->vBase, 0);
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

    // Pick THIS frame's bone-SSBO region by the fence-guarded in-flight slot, so the
    // CPU never writes matrices a still-in-flight frame's GPU may be reading.
    const u32 slot  = CommandManager.GetCurrentFrame();
    u32 cursor      = slot * kMaxBones;
    const u32 limit = cursor + kMaxBones;

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

void Skinned_RenderShadow(VkCommandBuffer cmd, const Fmatrix& lightVP,
                          const Fvector* cullPos, float cullRange)
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
            } else if (!ShadowMap::SphereVisible(c, bs.R)) continue;
        }

        for (auto* child : u.K->children)
        {
            VK_Render_Mesh* mesh = nullptr;
            u16 rmode = 0;
            if (!ResolveSkinnedLeaf(child, mesh, rmode)) continue;
            if (child->m_bEmissiveAdd || child->m_bModelGlass || child->m_bLitBlend) continue;   // marks/glass don't cast shadows

            VkPipeline pipe = GetShadowPipeline(mesh->vStride);
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

            const u32 skinMode = (rmode <= 2u) ? 1u : (u32(rmode) - 1u);
            SkinPush pc{ lightVP, skinMode, u.baseBone, (u32)u.boneCount, 0.0f, 1.0f };
            vkCmdPushConstants(cmd, s_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

            VkBuffer vb = mesh->p_rm_Vertices->GetHandle();
            VkDeviceSize vbOff = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &vbOff);
            vkCmdBindIndexBuffer(cmd, mesh->p_rm_Indices->GetHandle(), 0, mesh->iType);
            vkCmdDrawIndexed(cmd, mesh->iCount, 1, mesh->iBase, (s32)mesh->vBase, 0);
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

            VkPipeline pipe = getPipe(mesh->vStride);
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

            const u32 skinMode = (rmode <= 2u) ? 1u : (u32(rmode) - 1u);
            SkinPush pc{ viewProj, skinMode, u.baseBone, (u32)u.boneCount, 0.0f, 1.0f };
            vkCmdPushConstants(cmd, s_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

            VkBuffer vb = mesh->p_rm_Vertices->GetHandle();
            VkDeviceSize vbOff = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &vbOff);
            vkCmdBindIndexBuffer(cmd, mesh->p_rm_Indices->GetHandle(), 0, mesh->iType);
            vkCmdDrawIndexed(cmd, mesh->iCount, 1, mesh->iBase, (s32)mesh->vBase, 0);
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
void Skinned_RenderMotion(VkCommandBuffer cmd, const Fmatrix& curVP, const Fmatrix& prevVP)
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
            MVSkinPush pc{ curVP, prevVP, skinMode, u.baseBone, prevBase, (u32)u.boneCount };
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
            VsmSkinnedCaster sc{};
            sc.sphere_P     = c;            sc.sphere_R = r;
            sc.index_count  = mesh->iCount; sc.ib_first = mesh->iBase; sc.first_vertex = (s32)mesh->vBase;
            sc.vb           = mesh->p_rm_Vertices->GetHandle();
            sc.ib           = mesh->p_rm_Indices->GetHandle();
            sc.iType        = mesh->iType;  sc.stride   = mesh->vStride;
            sc.base_bone    = u.baseBone;   sc.bone_count = (u32)u.boneCount;
            sc.skin_mode    = (rmode <= 2u) ? 1u : (u32(rmode) - 1u);
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
    s_uploads.clear(); s_uploadsHud.clear(); s_uploadFrame = u32(-1);
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
