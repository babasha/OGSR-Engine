// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

#include "stdafx.h"
#include "vk_pass_sky.h"
#include "vk_swapchain.h"
#include "vk_scene_color.h"                // HDR scene target format
#include "vk_shaders.h"
#include "vk_texture.h"
#include "HW_Vulkan.h"
#include "vk_command_buffer.h"             // CommandManager.GetCurrentFrame() — in-flight slot
#include "vk_pipeline_cache.h"             // PipelineCache::GetCacheObject() — shared disk-backed cache

#include "../../xr_3da/device.h"           // Device.vCameraPosition
#include "../../xr_3da/IGame_Persistent.h" // g_pGamePersistent
#include "../../xr_3da/Environment.h"      // CEnvironment / CEnvDescriptor

#include <unordered_map>
#include <string>

namespace VK {

namespace {
    VkPipeline            s_Pipeline       = VK_NULL_HANDLE;
    VkPipelineLayout      s_PipelineLayout = VK_NULL_HANDLE;
    VkShaderModule        s_VS             = VK_NULL_HANDLE;
    VkShaderModule        s_FS             = VK_NULL_HANDLE;

    constexpr u32 kFramesInFlight = CVulkanCommandManager::FRAMES_IN_FLIGHT;

    VkDescriptorSetLayout s_SetLayout      = VK_NULL_HANDLE;
    VkDescriptorPool      s_Pool           = VK_NULL_HANDLE;
    VkDescriptorSet       s_Set[kFramesInFlight] = {};  // one per in-flight slot (no in-place rewrite)
    VkSampler             s_Sampler        = VK_NULL_HANDLE;

    // Cubemap cache — names are the keys (X-Ray sky_texture_name strings).
    // Lifetime = SkyPass lifetime; freed in Destroy. Lookup is rare so the
    // map cost is negligible.
    std::unordered_map<std::string, CVulkanTexture*> s_CubeCache;

    CVulkanTexture* s_FallbackCube = nullptr;  // 1×1×6 mid-blue

    // What each in-flight slot's descriptor set currently holds. Empty = fallback
    // bound to both bindings. A slot is rewritten only when its own names diverge
    // from CEnv, so the GPU never reads a set mid-rewrite (see EnsureCurrentCubes).
    std::string s_BoundName0[kFramesInFlight];
    std::string s_BoundName1[kFramesInFlight];

    // Push must match shaders/sky.{vert,frag}.glsl exactly. Layout = 4×vec4
    // with manual packing (vec3 + scalar trailing). vec3+float push_constant
    // packing is implementation-defined, so we keep it explicit on both sides.
    struct SkyPush
    {
        float camRightTan[3];  // 12 — vCameraRight * tan(fov/2) * aspect
        float skyRotation;     //  4
        float camUpTan[3];     // 12 — vCameraTop   * tan(fov/2)
        float blendWeight;     //  4
        float camForward[3];   // 12 — vCameraDirection (unit)
        float _pad0;           //  4
        float skyColor[3];     // 12
        float _pad1;           //  4
    };
    static_assert(sizeof(SkyPush) == 64, "SkyPush mismatch with GLSL push block");

    // Resolve sky_texture_name → "$game_textures$\<name>.dds".
    bool ResolveCubePath(const char* name, string_path& out)
    {
        if (!name || !name[0]) return false;
        string_path leaf;
        xr_sprintf(leaf, "%s.dds", name);
        FS.update_path(out, "$game_textures$", leaf);
        return FS.exist(out);
    }

    CVulkanTexture* CreateFallbackCube()
    {
        auto* tex = xr_new<CVulkanTexture>();
        tex->m_bCubemap    = true;
        tex->m_ArrayLayers = 6;
        tex->Create(1, 1, VK_FORMAT_R8G8B8A8_UNORM, 1);
        if (!tex->IsValid()) { xr_delete(tex); return nullptr; }
        u8 face[24];
        for (u32 i = 0; i < 6; ++i) {
            face[i*4 + 0] = 90; face[i*4 + 1] = 130;
            face[i*4 + 2] = 175; face[i*4 + 3] = 255;
        }
        tex->UploadData(face, sizeof(face));
        return tex;
    }

    // Load (or fetch from cache) a cubemap by `sky_texture_name`. Returns
    // fallback on miss / failure. Reads are cached so we never hit disk twice
    // for the same sky.
    CVulkanTexture* LoadOrGet(const char* name)
    {
        if (!name || !name[0]) return s_FallbackCube;
        auto it = s_CubeCache.find(name);
        if (it != s_CubeCache.end()) return it->second;

        string_path full;
        if (!ResolveCubePath(name, full)) {
            Msg("![VK Sky] sky_texture '%s' not found at $game_textures$ — using fallback", name);
            s_CubeCache.emplace(name, s_FallbackCube);  // negative cache
            return s_FallbackCube;
        }

        auto* tex = xr_new<CVulkanTexture>();
        if (!tex->LoadDDSCubemap(full, /*applyBCSwizzle*/ false)) {
            Msg("![VK Sky] LoadDDSCubemap failed: %s", full);
            xr_delete(tex);
            s_CubeCache.emplace(name, s_FallbackCube);
            return s_FallbackCube;
        }

        Msg("[VK Sky] Loaded cubemap: '%s' (%ux%u)", name, tex->GetWidth(), tex->GetHeight());
        s_CubeCache.emplace(name, tex);
        return tex;
    }

    void WriteSet(VkDescriptorSet set, VkImageView v0, VkImageView v1)
    {
        VkDescriptorImageInfo ii[2]{};
        for (int i = 0; i < 2; ++i) {
            ii[i].sampler     = s_Sampler;
            ii[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
        ii[0].imageView = v0;
        ii[1].imageView = v1;

        VkWriteDescriptorSet w[2]{};
        for (int i = 0; i < 2; ++i) {
            w[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet          = set;
            w[i].dstBinding      = i;
            w[i].descriptorCount = 1;
            w[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[i].pImageInfo      = &ii[i];
        }
        vkUpdateDescriptorSets(VulkanHW.m_Device, 2, w, 0, nullptr);
    }

    // Fetch [name0, name1] for the active weather interval. Empty strings
    // when the env subsystem hasn't populated yet.
    void GetCurrentSkyNames(const char** outName0, const char** outName1)
    {
        *outName0 = nullptr;
        *outName1 = nullptr;
        if (!g_pGamePersistent) return;
        auto& env = g_pGamePersistent->Environment();
        if (env.Current[0] && env.Current[0]->sky_texture_name.size() > 0)
            *outName0 = env.Current[0]->sky_texture_name.c_str();
        if (env.Current[1] && env.Current[1]->sky_texture_name.size() > 0)
            *outName1 = env.Current[1]->sky_texture_name.c_str();
    }

    void EnsureCurrentCubes(u32 slot)
    {
        const char* n0 = nullptr;
        const char* n1 = nullptr;
        GetCurrentSkyNames(&n0, &n1);

        // Treat null/empty as "keep current". On the very first call before
        // env is up we leave the fallback bound (s_BoundName*[slot] are empty).
        const std::string newN0 = n0 ? std::string(n0) : s_BoundName0[slot];
        const std::string newN1 = n1 ? std::string(n1) : s_BoundName1[slot];
        if (newN0 == s_BoundName0[slot] && newN1 == s_BoundName1[slot]) return;

        // No vkDeviceWaitIdle: WaitForFence(slot) in CRender::Begin already proved
        // the GPU finished the previous frame that used s_Set[slot], so rewriting
        // just this slot's set is safe. The other slots keep their old cubemap
        // views (textures are cached, never freed at runtime) until each is
        // refreshed on its own turn — a weather change converges over <=
        // FRAMES_IN_FLIGHT frames, imperceptible against the seconds-long blend.
        CVulkanTexture* t0 = n0 ? LoadOrGet(n0) : s_FallbackCube;
        CVulkanTexture* t1 = n1 ? LoadOrGet(n1) : t0;  // fall back to t0 if no second
        if (!t0) t0 = s_FallbackCube;
        if (!t1) t1 = t0;

        WriteSet(s_Set[slot], t0->GetView(), t1->GetView());
        s_BoundName0[slot] = newN0;
        s_BoundName1[slot] = newN1;
    }
}

namespace SkyPass {

bool AcquireAmbientCubes(VkImageView& v0, VkImageView& v1, VkSampler& sampler, float& weight)
{
    // Sky system not up yet (no sampler / fallback) — caller keeps its fallback.
    if (s_Sampler == VK_NULL_HANDLE || !s_FallbackCube) return false;

    const char* n0 = nullptr;
    const char* n1 = nullptr;
    GetCurrentSkyNames(&n0, &n1);

    // Same cached load the sky draw uses — first sight of a sky triggers the
    // synchronous DDS load + mip-gen; cached forever after, so this is cheap.
    CVulkanTexture* t0 = n0 ? LoadOrGet(n0) : s_FallbackCube;
    CVulkanTexture* t1 = n1 ? LoadOrGet(n1) : t0;
    if (!t0) t0 = s_FallbackCube;
    if (!t1) t1 = t0;

    v0      = t0->GetView();
    v1      = t1->GetView();
    sampler = s_Sampler;
    weight  = 0.0f;
    if (g_pGamePersistent)
        if (auto* mix = g_pGamePersistent->Environment().CurrentEnv)
            weight = mix->weight;
    return true;
}

bool Init()
{
    if (s_Pipeline) return true;

    if (!g_ShaderManager) g_ShaderManager = xr_new<VK::CVulkanSPIRVLoader>();
    s_VS = g_ShaderManager->Load("sky.vert.spv");
    s_FS = g_ShaderManager->Load("sky.frag.spv");
    if (s_VS == VK_NULL_HANDLE || s_FS == VK_NULL_HANDLE) {
        Msg("![VK Sky] Failed to load sky shaders");
        return false;
    }

    // Set 0: bindings 0+1 — two combined image samplers (cubemaps), FS only.
    {
        VkDescriptorSetLayoutBinding b[2]{};
        for (int i = 0; i < 2; ++i) {
            b[i].binding         = i;
            b[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b[i].descriptorCount = 1;
            b[i].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        VkDescriptorSetLayoutCreateInfo lci{};
        lci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        lci.bindingCount = 2;
        lci.pBindings    = b;
        if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &s_SetLayout) != VK_SUCCESS) {
            Msg("![VK Sky] CreateDescriptorSetLayout failed");
            return false;
        }
    }

    {
        VkDescriptorPoolSize ps{};
        ps.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        ps.descriptorCount = 2 * kFramesInFlight;  // 2 bindings × FRAMES_IN_FLIGHT sets

        VkDescriptorPoolCreateInfo pci{};
        pci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets       = kFramesInFlight;
        pci.poolSizeCount = 1;
        pci.pPoolSizes    = &ps;
        if (vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &s_Pool) != VK_SUCCESS) {
            Msg("![VK Sky] CreateDescriptorPool failed");
            return false;
        }
    }

    {
        VkDescriptorSetLayout layouts[kFramesInFlight];
        for (u32 i = 0; i < kFramesInFlight; ++i) layouts[i] = s_SetLayout;
        VkDescriptorSetAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = s_Pool;
        ai.descriptorSetCount = kFramesInFlight;
        ai.pSetLayouts        = layouts;
        if (vkAllocateDescriptorSets(VulkanHW.m_Device, &ai, s_Set) != VK_SUCCESS) {
            Msg("![VK Sky] AllocateDescriptorSets failed");
            return false;
        }
    }

    {
        VkSamplerCreateInfo si{};
        si.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter    = VK_FILTER_LINEAR;
        si.minFilter    = VK_FILTER_LINEAR;
        si.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.minLod       = 0.0f;
        si.maxLod       = VK_LOD_CLAMP_NONE;
        if (vkCreateSampler(VulkanHW.m_Device, &si, nullptr, &s_Sampler) != VK_SUCCESS) {
            Msg("![VK Sky] CreateSampler failed");
            return false;
        }
    }

    s_FallbackCube = CreateFallbackCube();
    if (!s_FallbackCube) {
        Msg("![VK Sky] Failed to create fallback cubemap");
        return false;
    }
    for (u32 i = 0; i < kFramesInFlight; ++i)
        WriteSet(s_Set[i], s_FallbackCube->GetView(), s_FallbackCube->GetView());

    // Pipeline layout — set 0 + push range, VS|FS (push struct is shared).
    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pc.offset     = 0;
    pc.size       = sizeof(SkyPush);

    VkPipelineLayoutCreateInfo plci{};
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &s_SetLayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pc;
    if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_PipelineLayout) != VK_SUCCESS) {
        Msg("![VK Sky] CreatePipelineLayout failed");
        return false;
    }

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = s_VS; stages[0].pName = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = s_FS; stages[1].pName = "main";

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.scissorCount  = 1;

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
    ds.depthWriteEnable = VK_FALSE;
    ds.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendAttachmentState ba{};
    ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    ba.blendEnable    = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments    = &ba;

    VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynState{};
    dynState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynState.dynamicStateCount = 2;
    dynState.pDynamicStates    = dyn;

    VkFormat colorFormat = VK::SceneColor::Format();
    VkPipelineRenderingCreateInfo prci{};
    prci.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    prci.colorAttachmentCount    = 1;
    prci.pColorAttachmentFormats = &colorFormat;
    prci.depthAttachmentFormat   = Swapchain.m_DepthFormat;

    VkGraphicsPipelineCreateInfo pi{};
    pi.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pi.pNext               = &prci;
    pi.stageCount          = 2;
    pi.pStages             = stages;
    pi.pVertexInputState   = &vi;
    pi.pInputAssemblyState = &ia;
    pi.pViewportState      = &vp;
    pi.pRasterizationState = &rs;
    pi.pMultisampleState   = &ms;
    pi.pDepthStencilState  = &ds;
    pi.pColorBlendState    = &cb;
    pi.pDynamicState       = &dynState;
    pi.layout              = s_PipelineLayout;

    if (vkCreateGraphicsPipelines(VulkanHW.m_Device, PipelineCache::GetCacheObject(), 1, &pi, nullptr, &s_Pipeline) != VK_SUCCESS) {
        Msg("![VK Sky] CreateGraphicsPipelines failed");
        return false;
    }

    Msg("[VK Sky] Init OK (two-cubemap blend, sky_rotation, sky_color tint)");
    return true;
}

void Destroy()
{
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;

    for (auto& kv : s_CubeCache) {
        // Negative-cache entries point at the fallback — skip them.
        if (kv.second && kv.second != s_FallbackCube) {
            kv.second->Destroy();
            xr_delete(kv.second);
        }
    }
    s_CubeCache.clear();
    for (u32 i = 0; i < kFramesInFlight; ++i) { s_BoundName0[i].clear(); s_BoundName1[i].clear(); }

    if (s_FallbackCube) { s_FallbackCube->Destroy(); xr_delete(s_FallbackCube); }

    if (s_Pipeline)       { vkDestroyPipeline(VulkanHW.m_Device, s_Pipeline, nullptr);             s_Pipeline = VK_NULL_HANDLE; }
    if (s_PipelineLayout) { vkDestroyPipelineLayout(VulkanHW.m_Device, s_PipelineLayout, nullptr); s_PipelineLayout = VK_NULL_HANDLE; }
    if (s_Pool)           { vkDestroyDescriptorPool(VulkanHW.m_Device, s_Pool, nullptr);            s_Pool = VK_NULL_HANDLE; }
    if (s_Sampler)        { vkDestroySampler(VulkanHW.m_Device, s_Sampler, nullptr);                s_Sampler = VK_NULL_HANDLE; }
    if (s_SetLayout)      { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_SetLayout, nullptr);  s_SetLayout = VK_NULL_HANDLE; }
    s_VS = VK_NULL_HANDLE;
    s_FS = VK_NULL_HANDLE;
}

}  // namespace SkyPass

// Layout is managed centrally (single-layout convention): the swapchain image
// stays in COLOR_ATTACHMENT_OPTIMAL and depth in DEPTH_ATTACHMENT_OPTIMAL for
// the whole frame. This pass only records draws; ExecutePasses inserts the
// inter-pass barrier before it. See CRender::Begin/End.

void Pass_Sky(FrameContext& ctx)
{
    if (!s_Pipeline)               return;
    if (ctx.cmd == VK_NULL_HANDLE) return;

    const u32 slot = CommandManager.GetCurrentFrame();  // fence-guarded in-flight slot
    EnsureCurrentCubes(slot);

    VkCommandBuffer cmd = ctx.cmd;

    VkRenderingAttachmentInfo cAtt{};
    cAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    cAtt.imageView   = ctx.colorView;
    cAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    cAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;
    cAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingAttachmentInfo dAtt{};
    dAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    dAtt.imageView   = ctx.depthView;
    dAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    dAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;
    dAtt.storeOp     = VK_ATTACHMENT_STORE_OP_DONT_CARE;

    VkRenderingInfo ri{};
    ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea.extent    = ctx.extent;
    ri.layerCount           = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments    = &cAtt;
    ri.pDepthAttachment     = &dAtt;
    vkCmdBeginRendering(cmd, &ri);

    VkViewport vp{};
    vp.x        = 0.0f;
    vp.y        = (float)ctx.extent.height;
    vp.width    = (float)ctx.extent.width;
    vp.height   = -(float)ctx.extent.height;
    vp.minDepth = 0.0f;
    vp.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{ {}, ctx.extent };
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_Pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            s_PipelineLayout, 0, 1, &s_Set[slot], 0, nullptr);

    // Camera basis → sky direction (position-invariant by construction).
    // tan(fov_y/2) maps clip-space [-1,1] into world-space ray spread.
    // Device.fASPECT in X-Ray is height/width (CameraManager.cpp:335 ->
    // CCameraManager::Update), so the horizontal half-tan is tan/fASPECT,
    // not tan*fASPECT. Getting this wrong was making horizontal FOV ~3×
    // narrower than the viewport, so any camera yaw shifted the cubemap
    // sample much faster than reality — visible as the sky "accelerating"
    // opposite to the camera.
    const float halfFovRad = deg2rad(Device.fFOV) * 0.5f;
    const float tanHalf    = tanf(halfFovRad);
    Fvector rightScaled, upScaled;
    rightScaled.mul(Device.vCameraRight, tanHalf / Device.fASPECT);
    upScaled   .mul(Device.vCameraTop,   tanHalf);

    SkyPush push{};
    push.camRightTan[0] = rightScaled.x;
    push.camRightTan[1] = rightScaled.y;
    push.camRightTan[2] = rightScaled.z;
    push.camUpTan[0]    = upScaled.x;
    push.camUpTan[1]    = upScaled.y;
    push.camUpTan[2]    = upScaled.z;
    push.camForward[0]  = Device.vCameraDirection.x;
    push.camForward[1]  = Device.vCameraDirection.y;
    push.camForward[2]  = Device.vCameraDirection.z;

    // Pull the per-frame env knobs (rotation / tint / weight). Default to
    // a neutral state when env isn't ready yet.
    push.skyRotation   = 0.0f;
    push.blendWeight   = 0.0f;
    push.skyColor[0]   = 1.0f;
    push.skyColor[1]   = 1.0f;
    push.skyColor[2]   = 1.0f;
    if (g_pGamePersistent) {
        if (auto* mix = g_pGamePersistent->Environment().CurrentEnv) {
            push.skyRotation = mix->sky_rotation;
            push.skyColor[0] = mix->sky_color.x;
            push.skyColor[1] = mix->sky_color.y;
            push.skyColor[2] = mix->sky_color.z;
            push.blendWeight = mix->weight;
        }
    }

    vkCmdPushConstants(cmd, s_PipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(SkyPush), &push);

    vkCmdDraw(cmd, 3, 1, 0, 0);

    vkCmdEndRendering(cmd);
    // No exit transition: image stays COLOR_ATTACHMENT; End brings it to PRESENT.
}

}  // namespace VK
