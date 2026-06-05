#include "stdafx.h"
#include "vk_pipeline_cache.h"
#include "vk_swapchain.h"
#include "vk_shaders.h"
#include "vk_world_material.h"  // descriptor set 0 layout
#include "HW_Vulkan.h"

#include <unordered_map>

namespace VK { namespace PipelineCache {

namespace {
    VkPipelineLayout                       s_Layout      = VK_NULL_HANDLE;
    VkPipelineCache                        s_CacheObject = VK_NULL_HANDLE;  // shared, disk-backed
    VkShaderModule                         s_WorldLmapVS = VK_NULL_HANDLE;
    VkShaderModule                         s_WorldLmapFS = VK_NULL_HANDLE;
    VkShaderModule                         s_WorldVlitVS = VK_NULL_HANDLE;
    VkShaderModule                         s_WorldVlitFS = VK_NULL_HANDLE;
    std::unordered_map<Key, VkPipeline>    s_Pipelines;

    constexpr const char* kCacheFile = "vk_pipeline_cache.bin";  // under $app_data_root$

    // Create the shared VkPipelineCache, seeding it from last run's on-disk blob.
    // The blob has a vendor/device/driver-UUID header; vkCreatePipelineCache
    // silently discards it if the GPU or driver changed and starts empty, so
    // loading is always safe across hardware/driver swaps.
    void CreateCacheObject()
    {
        if (s_CacheObject != VK_NULL_HANDLE) return;

        xr_vector<u8> initial;
        string_path fn;
        FS.update_path(fn, "$app_data_root$", kCacheFile);
        if (IReader* r = FS.r_open(fn)) {
            const size_t n = (size_t)r->length();
            if (n) { initial.resize(n); memcpy(initial.data(), r->pointer(), n); }
            FS.r_close(r);
        }

        VkPipelineCacheCreateInfo ci{};
        ci.sType           = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
        ci.initialDataSize = initial.size();
        ci.pInitialData    = initial.empty() ? nullptr : initial.data();

        VkResult res = vkCreatePipelineCache(VulkanHW.m_Device, &ci, nullptr, &s_CacheObject);
        if (res != VK_SUCCESS) {
            Msg("![VK PipelineCache] vkCreatePipelineCache failed (%d); pipelines will build uncached", res);
            s_CacheObject = VK_NULL_HANDLE;
            return;
        }
        Msg("[VK PipelineCache] cache object ready (%zu bytes seeded from disk)", initial.size());
    }

    // Serialize the cache blob back to disk so the next run warm-starts.
    void SaveCacheObject()
    {
        if (s_CacheObject == VK_NULL_HANDLE) return;

        size_t sz = 0;
        if (vkGetPipelineCacheData(VulkanHW.m_Device, s_CacheObject, &sz, nullptr) != VK_SUCCESS || sz == 0)
            return;

        xr_vector<u8> data(sz);
        if (vkGetPipelineCacheData(VulkanHW.m_Device, s_CacheObject, &sz, data.data()) != VK_SUCCESS)
            return;

        string_path fn;
        FS.update_path(fn, "$app_data_root$", kCacheFile);
        if (IWriter* w = FS.w_open(fn)) {
            w->w(data.data(), (u32)sz);
            FS.w_close(w);
            Msg("[VK PipelineCache] saved %zu bytes -> %s", sz, fn);
        } else {
            Msg("![VK PipelineCache] could not open %s for write", fn);
        }
    }

    // Push range, shared by VS+FS (matches WorldPush in vk_pass_world.cpp /
    // PushConstants block in world.{vert,frag}.glsl):
    //   mat4 mvp       — VS, offset 0,  64 bytes
    //   vec2 uvScale   — VS, offset 64,  8 bytes
    //   float alphaRef — FS, offset 72,  4 bytes  (<0 disables aref discard)
    //   float _pad     — pad to 16-byte multiple (offset 76, 4 bytes)
    constexpr u32 kPushSize = 80;
}

VkPipelineLayout GetLayout()      { return s_Layout; }
VkPipelineCache  GetCacheObject() { return s_CacheObject; }
VkShaderModule   WorldLmapVS() { return s_WorldLmapVS; }
VkShaderModule   WorldLmapFS() { return s_WorldLmapFS; }
VkShaderModule   WorldVlitVS() { return s_WorldVlitVS; }
VkShaderModule   WorldVlitFS() { return s_WorldVlitFS; }

bool Init()
{
    if (s_Layout) return true;  // idempotent

    // Bring the shared, disk-backed cache up first so every pipeline produced
    // this run (here + Pass_Sky + Pass_Skinned) feeds and warm-starts from it.
    CreateCacheObject();

    if (!g_ShaderManager) {
        g_ShaderManager = xr_new<VK::CVulkanSPIRVLoader>();
        Msg("[VK PipelineCache] Created g_ShaderManager (lazy init)");
    }

    s_WorldLmapVS = g_ShaderManager->Load("world_lmap.vert.spv");
    s_WorldLmapFS = g_ShaderManager->Load("world_lmap.frag.spv");
    s_WorldVlitVS = g_ShaderManager->Load("world_vlit.vert.spv");
    s_WorldVlitFS = g_ShaderManager->Load("world_vlit.frag.spv");
    if (s_WorldLmapVS == VK_NULL_HANDLE || s_WorldLmapFS == VK_NULL_HANDLE ||
        s_WorldVlitVS == VK_NULL_HANDLE || s_WorldVlitFS == VK_NULL_HANDLE) {
        Msg("![VK PipelineCache] Failed to load world shaders (lmap/vlit, .vert.spv/.frag.spv)");
        return false;
    }

    VkPushConstantRange pc{};
    pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pc.offset     = 0;
    pc.size       = kPushSize;

    VkDescriptorSetLayout setLayouts[1] = { WorldMaterialCache::GetSetLayout() };
    if (setLayouts[0] == VK_NULL_HANDLE) {
        Msg("![VK PipelineCache] WorldMaterialCache layout not initialised — call WorldMaterialCache::Init() first");
        return false;
    }

    VkPipelineLayoutCreateInfo plci{};
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = setLayouts;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pc;
    if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_Layout) != VK_SUCCESS) {
        Msg("![VK PipelineCache] vkCreatePipelineLayout failed");
        return false;
    }

    Msg("[VK PipelineCache] Init OK (layout, push=%u bytes)", kPushSize);
    return true;
}

void Destroy()
{
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;

    // Persist the warmed cache, then drop the object. Done first (while every
    // producer's pipelines are still around) so the blob captures this run's work.
    SaveCacheObject();
    if (s_CacheObject) {
        vkDestroyPipelineCache(VulkanHW.m_Device, s_CacheObject, nullptr);
        s_CacheObject = VK_NULL_HANDLE;
    }

    for (auto& kv : s_Pipelines) {
        if (kv.second) vkDestroyPipeline(VulkanHW.m_Device, kv.second, nullptr);
    }
    s_Pipelines.clear();

    if (s_Layout) {
        vkDestroyPipelineLayout(VulkanHW.m_Device, s_Layout, nullptr);
        s_Layout = VK_NULL_HANDLE;
    }
    // Shader modules are owned by g_ShaderManager — leave them.
    s_WorldLmapVS = VK_NULL_HANDLE;
    s_WorldLmapFS = VK_NULL_HANDLE;
    s_WorldVlitVS = VK_NULL_HANDLE;
    s_WorldVlitFS = VK_NULL_HANDLE;
}

// Build vertex input for X-Ray level static vertex layouts (stride 32).
//
// Common across both sub-layouts (locations 0-3):
//   loc 0 = POSITION   FLOAT3                @ 0
//   loc 1 = TC0        SHORT2 SSCALED        @ tcOffset (24 lmap | 28 vlit)
//   loc 2 = TANGENT    UBYTE4_UNORM (D3DCOLOR) @ 16  — only .a (du) is consumed
//   loc 3 = BINORMAL   UBYTE4_UNORM (D3DCOLOR) @ 20  — only .a (dv) is consumed
//
// Sub-layout-specific (location 4):
//   tcOffset==24 ("lmap"): TC1 = lightmap UV, SHORT2 SSCALED @ 28. Shader
//      treats it as a [-1,1] unit UV (raw * 1/32768 in VS).
//   tcOffset==28 ("vert-lit"): COLOR = D3DCOLOR pre-baked lighting @ 24.
//      .bgr = baked RGB (point lights + bounce); .a = sun mask.
//
// du/dv eliminate UV stripe artifacts at triangle edges (R4 unpack_tc_base
// equivalent). Sub-layout choice drives lit/lmap shader selection at the
// pipeline level — caller picks vs/fs accordingly.
static void BuildVertexInputForStride(u32 stride, u32 tcOffset,
                                      VkVertexInputBindingDescription&    binding,
                                      VkVertexInputAttributeDescription   attrs[5])
{
    binding  = { 0, stride, VK_VERTEX_INPUT_RATE_VERTEX };
    attrs[0] = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT,  0  };
    attrs[1] = { 1, 0, VK_FORMAT_R16G16_SSCALED,    tcOffset };
    attrs[2] = { 2, 0, VK_FORMAT_R8G8B8A8_UNORM,    16 };
    attrs[3] = { 3, 0, VK_FORMAT_R8G8B8A8_UNORM,    20 };
    if (tcOffset == 24) {
        // lmap variant: TC1 lightmap UV
        attrs[4] = { 4, 0, VK_FORMAT_R16G16_SSCALED, 28 };
    } else {
        // vert-lit variant: pre-baked vertex color at offset 24
        attrs[4] = { 4, 0, VK_FORMAT_R8G8B8A8_UNORM, 24 };
    }
}

static VkPipeline CreatePipeline(const Key& k)
{
    VkVertexInputBindingDescription   binding{};
    VkVertexInputAttributeDescription attrs[5]{};
    BuildVertexInputForStride(k.stride, k.tcOffset, binding, attrs);

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount   = 1;
    vi.pVertexBindingDescriptions      = &binding;
    vi.vertexAttributeDescriptionCount = 5;
    vi.pVertexAttributeDescriptions    = attrs;

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = k.vs; stages[0].pName = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = k.fs; stages[1].pName = "main";

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
    rs.cullMode    = VK_CULL_MODE_NONE;     // no winding info yet → don't drop faces
    rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable  = k.depthTest ? VK_TRUE : VK_FALSE;
    ds.depthWriteEnable = k.depthTest ? VK_TRUE : VK_FALSE;
    ds.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;
    // Phase 4 wires up an actual depth attachment; right now no depth target
    // is bound at draw time so these flags only affect future passes.

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

    VkFormat colorFormat = Swapchain.m_Format;
    VkPipelineRenderingCreateInfo prci{};
    prci.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    prci.colorAttachmentCount    = 1;
    prci.pColorAttachmentFormats = &colorFormat;
    if (k.depthTest) {
        prci.depthAttachmentFormat = Swapchain.m_DepthFormat;
        // Stencil aspect lives in the same view for combined formats but the
        // pass doesn't read or write it; leave stencilAttachmentFormat = UNDEFINED.
    }

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
    pi.layout              = s_Layout;

    VkPipeline handle = VK_NULL_HANDLE;
    VkResult r = vkCreateGraphicsPipelines(VulkanHW.m_Device, s_CacheObject, 1, &pi, nullptr, &handle);
    if (r != VK_SUCCESS) {
        Msg("![VK PipelineCache] vkCreateGraphicsPipelines failed (%d) for stride=%u tcOff=%u depth=%d",
            r, k.stride, k.tcOffset, (int)k.depthTest);
        return VK_NULL_HANDLE;
    }
    Msg("[VK PipelineCache] Created pipeline stride=%u tcOff=%u depth=%d", k.stride, k.tcOffset, (int)k.depthTest);
    return handle;
}

VkPipeline Get(const Key& key)
{
    if (!s_Layout) return VK_NULL_HANDLE;

    auto it = s_Pipelines.find(key);
    if (it != s_Pipelines.end()) return it->second;

    VkPipeline p = CreatePipeline(key);
    s_Pipelines.emplace(key, p);
    return p;
}

}}  // namespace VK::PipelineCache
