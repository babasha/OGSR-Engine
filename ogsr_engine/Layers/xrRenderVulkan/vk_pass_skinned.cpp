// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan - Skinned-mesh pass (STEP B sub-step 2). See vk_pass_skinned.h.
#include "stdafx.h"

// CKinematics (children + LL_GetTransform_R/LL_BoneCount) — same compat trick as
// the skeleton wrapper TUs. Keep dxRender_Visual mapped (children is xr_vector<dxRender_Visual*>).
#define FBasicVisualH
#include "vk_FBasicVisual.h"            // pulls vk_SkeletonCompat.h (dxRender_Visual->vkRender_Visual, vk_Visual.h)
#include "CRender_Vulkan.h"
#include "../xrRender/SkeletonCustom.h" // CKinematics

#include "vk_pass_skinned.h"
#include "vk_pass_world.h"              // g_DynamicVisuals / DynVisual
#include "vk_swapchain.h"              // Swapchain.m_Format / m_DepthFormat
#include "vk_scene_color.h"            // HDR scene target format
#include "vk_shaders.h"               // g_ShaderManager
#include "vk_buffer.h"                // CVulkanBuffer
#include "vk_world_material.h"        // WorldMaterial / WorldMaterialCache (set 1 = diffuse)
#include "HW_Vulkan.h"                // VulkanHW.m_Device
#include "vk_command_buffer.h"        // CommandManager.GetCurrentFrame() — in-flight slot
#include "vk_pipeline_cache.h"        // PipelineCache::GetCacheObject() — shared disk-backed cache
#include "vk_env_light.h"             // EnvLight — shared per-frame sun/hemi/ambient UBO (set 2)
#include "vk_shadow.h"                // ShadowMap::SphereVisible — caster culling
#include "../../xr_3da/device.h"      // Device.mFullTransform_hud (HUD projection), dwFrame

#include <unordered_map>

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
    CVulkanBuffer         s_boneSSBO;
    Fmatrix*              s_boneMapped = nullptr;
    std::unordered_map<u32, VkPipeline> s_pipelines;        // keyed by vertex stride (36/40/44)
    std::unordered_map<u32, VkPipeline> s_shadowPipelines;  // depth-only casters, same key
    std::unordered_map<u32, VkPipeline> s_prepassPipelines; // scene depth-prepass AT casters

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
    // Set 2 (per-frame sun/hemi/ambient) is the shared EnvLight set — see vk_env_light.{h,cpp}.

    // Push block — must match skinned.{vert,frag}.glsl PushConstants. hudMode (read
    // by the fragment) flags first-person HUD so it gets sun-direction-independent light.
    struct SkinPush { Fmatrix mvp; u32 skinMode; u32 baseBone; u32 boneCount; float hudMode; float hemi; };

    // Vertex input for the vertHW_* layouts. loc0 pos FLOAT4; loc1/3/4 packed
    // normal/tangent/binormal (UBYTE4N); loc2 = FLOAT2 (36/40) or FLOAT4 (44);
    // loc5 = 4 bone indices (40 only, dummy @0 elsewhere — unused by those modes).
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

    static VkPipeline CreatePipeline(u32 stride, bool additive)
    {
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
        stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = s_fs; stages[1].pName = "main";

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
        ds.depthWriteEnable = additive ? VK_FALSE : VK_TRUE;   // marks don't write depth (R4 zb(true,false))
        ds.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineColorBlendAttachmentState ba{};
        ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        ba.blendEnable    = VK_FALSE;
        if (additive) {
            // Collimator marks (R4 hud_reddotsight: blend(srcalpha, one)) —
            // the dot ADDS light over the sight glass.
            ba.blendEnable         = VK_TRUE;
            ba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            ba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
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

        VkFormat colorFormat = VK::SceneColor::Format();
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
        Msg("[VK Skinned] pipeline created stride=%u additive=%d", stride, (int)additive);
        return h;
    }

    static VkPipeline GetPipeline(u32 stride, bool additive = false)
    {
        const u32 key = stride | (additive ? 0x10000u : 0u);
        auto it = s_pipelines.find(key);
        if (it != s_pipelines.end()) return it->second;
        VkPipeline p = CreatePipeline(stride, additive);
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
        // Optional — absence just disables NPC shadow casting.
        s_shadowVS = g_ShaderManager->Load("shadow_skinned.vert.spv");
        if (s_shadowVS == VK_NULL_HANDLE)
            Msg("![VK Skinned] shadow_skinned.vert.spv missing — skinned shadow casting disabled");
        // Optional — absence just keeps NPCs out of the depth prepass (no NPC AO).
        s_prepassVS = g_ShaderManager->Load("shadow_skinned_at.vert.spv");
        s_prepassFS = g_ShaderManager->Load("shadow_skinned_at.frag.spv");
        if (s_prepassVS == VK_NULL_HANDLE || s_prepassFS == VK_NULL_HANDLE)
            Msg("![VK Skinned] shadow_skinned_at.{vert,frag}.spv missing — NPCs excluded from the depth prepass");

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

        // Bone SSBO (host-visible, persistently mapped — Create sets HOST_ACCESS+MAPPED for STORAGE).
        // Sized FRAMES_IN_FLIGHT × kMaxBones: each in-flight frame owns region
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
    // s_layout (sets 1/2 simply go unbound — the shader only touches set 0).
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
        rs.depthBiasEnable = VK_TRUE;       // dynamic — caller sets the same bias as statics

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
    // (same a<0.25 discard as the color pass → identical coverage). Renders
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
    // don't disturb pipeline/descriptor bindings — both are dynamic / separate).
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
                // hud_reddotsight) instead of the lit opaque one.
                const bool emissive = child->m_bEmissiveAdd;
                VkPipeline pipe = GetPipeline(mesh->vStride, emissive);
                if (pipe == VK_NULL_HANDLE) continue;

                if (pipe != lastPipe) {
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_layout, 0, 1, &s_set, 0, nullptr);
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_layout, 2, 1, &lightSet, 0, nullptr);  // set2 = env lighting (per-frame)
                    lastPipe = pipe;
                    lastMatSet = VK_NULL_HANDLE;   // pipeline change disturbs set bindings — rebind material
                }

                // Diffuse texture (set 1) from the leaf's WorldMaterial; fall back to default white.
                WorldMaterial* mat = child->m_pWorldMaterial ? child->m_pWorldMaterial : WorldMaterialCache::GetDefault();
                VkDescriptorSet matSet = (mat && mat->set != VK_NULL_HANDLE) ? mat->set : VK_NULL_HANDLE;
                if (matSet == VK_NULL_HANDLE) continue;   // no texture descriptor → can't sample, skip
                if (matSet != lastMatSet) {
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_layout, 1, 1, &matSet, 0, nullptr);
                    lastMatSet = matSet;
                }

                // RenderMode enum -> skinMode: RM_SINGLE(1)/RM_SKINNING_1B(2)->1, 2B(3)->2, 3B(4)->3, 4B(5)->4.
                // Bit 4 (16) = emissive-add flag for the fragment (unlit output);
                // the vertex shader masks it off before the skinning switch.
                u32 skinMode = (rmode <= 2u) ? 1u : (u32(rmode) - 1u);
                if (emissive) skinMode |= 16u;
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

void Skinned_UploadBones()
{
    if (!Init()) return;
    if (s_uploadFrame == Device.dwFrame) return;   // once per frame
    s_uploadFrame = Device.dwFrame;
    s_uploads.clear();
    s_uploadsHud.clear();

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
            // STALE for UCalc_Interval ms → visibly jerky animation. The
            // dwTimeGlobal==UCalc_Time guard inside still prevents double-advance.
            K->CalculateBones(TRUE);
            const u16 bc = K->LL_BoneCount();
            if (bc == 0) continue;
            if (cursor + bc > limit) break;   // this frame's SSBO region full

            // Store bone·xform (model→WORLD): S*pos in the shaders is then world-
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
            if (child->m_bEmissiveAdd) continue;   // collimator marks don't cast shadows

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

// NPCs into the scene depth PREPASS (camera VP): they become GTAO occluders —
// contact darkening on the ground under characters, like R4 where skinned
// geometry is in the gbuffer — and the world color pass gets early-Z behind
// them. Alpha-tested (a < 0.25, same as skinned.frag) so hair/strap cutouts
// don't punch holes into the early-Z'd background. HUD never enters (s_uploads
// only). Caller owns render begin/end and viewport (Pass_World prepass).
void Skinned_RenderDepthPrepass(VkCommandBuffer cmd, const Fmatrix& viewProj)
{
    if (!Init()) return;
    if (s_prepassVS == VK_NULL_HANDLE || s_prepassFS == VK_NULL_HANDLE) return;
    Skinned_UploadBones();   // idempotent; Pass_SunShadow usually did it already
    if (s_uploads.empty()) return;

    // AO is short-range — only NPCs near the camera matter as occluders.
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
            if (child->m_bEmissiveAdd) continue;   // collimator marks: blended, no depth

            VkPipeline pipe = GetPrepassPipeline(mesh->vStride);
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

            // Diffuse (set 1) for the alpha test — default white (a=1) never discards.
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

void Pass_Skinned(FrameContext& ctx)
{
    if (g_DynamicVisuals.empty() && g_HudVisuals.empty()) return;
    if (ctx.cmd == VK_NULL_HANDLE || !ctx.viewProj) return;
    if (!Init()) return;

    // Normally a no-op — Pass_SunShadow already uploaded this frame's bones.
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

    // World dynamics: bones are world-space → push plain viewProj.
    DrawSkinnedList(cmd, s_uploads, *ctx.viewProj, nDraw, lastPipe, lastMatSet, lightSet, 0.0f);

    // First-person HUD: HUD-FOV projection (camera at origin) + near depth range so
    // hands/weapon render on top of the world. Restore the normal range afterwards.
    // hudMode=1 → fragment uses flat, sun-direction-independent lighting (view-space bones).
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
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
    for (auto& kv : s_pipelines) if (kv.second) vkDestroyPipeline(VulkanHW.m_Device, kv.second, nullptr);
    s_pipelines.clear();
    for (auto& kv : s_shadowPipelines) if (kv.second) vkDestroyPipeline(VulkanHW.m_Device, kv.second, nullptr);
    s_shadowPipelines.clear();
    for (auto& kv : s_prepassPipelines) if (kv.second) vkDestroyPipeline(VulkanHW.m_Device, kv.second, nullptr);
    s_prepassPipelines.clear();
    s_shadowVS = VK_NULL_HANDLE;   // module owned by g_ShaderManager
    s_prepassVS = VK_NULL_HANDLE; s_prepassFS = VK_NULL_HANDLE;
    s_uploads.clear(); s_uploadsHud.clear(); s_uploadFrame = u32(-1);
    if (s_layout)    { vkDestroyPipelineLayout(VulkanHW.m_Device, s_layout, nullptr); s_layout = VK_NULL_HANDLE; }
    if (s_pool)      { vkDestroyDescriptorPool(VulkanHW.m_Device, s_pool, nullptr); s_pool = VK_NULL_HANDLE; }
    if (s_setLayout) { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_setLayout, nullptr); s_setLayout = VK_NULL_HANDLE; }
    // Set 2 (EnvLight) is owned by vk_env_light.cpp — destroyed separately in DevRender::Destroy.
    s_boneSSBO.Destroy();
    s_boneMapped = nullptr;
    s_set = VK_NULL_HANDLE;
    s_inited = false; s_failed = false;
}

}  // namespace VK
