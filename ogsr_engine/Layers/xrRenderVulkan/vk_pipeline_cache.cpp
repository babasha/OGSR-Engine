// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

#include "stdafx.h"
#include "vk_pipeline_cache.h"
#include "vk_swapchain.h"
#include "vk_scene_color.h"   // HDR scene target format (passes render to it, not the swapchain)
#include "vk_shaders.h"
#include "vk_world_material.h"  // descriptor set 0 layout
#include "vk_env_light.h"       // descriptor set 1 layout (per-frame sun/hemi/ambient)
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

    // World heightmap tessellation (R4 TESS_HM): TCS/TES pair per sub-layout.
    // All four must load (and the device feature be enabled) for tess keys.
    VkShaderModule                         s_WorldLmapTCS = VK_NULL_HANDLE;
    VkShaderModule                         s_WorldLmapTES = VK_NULL_HANDLE;
    VkShaderModule                         s_WorldVlitTCS = VK_NULL_HANDLE;
    VkShaderModule                         s_WorldVlitTES = VK_NULL_HANDLE;

    // Terrain splatting: own layout + single lazily-built pipeline + shaders.
    VkShaderModule                         s_TerrainVS       = VK_NULL_HANDLE;
    VkShaderModule                         s_TerrainFS       = VK_NULL_HANDLE;
    VkPipelineLayout                       s_TerrainLayout   = VK_NULL_HANDLE;
    VkPipeline                             s_TerrainPipeline = VK_NULL_HANDLE;

    // Sun shadow caster: depth-only VS (no FS), own layout (push: mat4 lightMVP),
    // per-stride lazily-built pipelines into the shadow map's D32 format.
    VkShaderModule                         s_DepthVS         = VK_NULL_HANDLE;
    VkPipelineLayout                       s_DepthLayout     = VK_NULL_HANDLE;
    std::unordered_map<u32, VkPipeline>    s_DepthPipelines;

    // Alpha-tested caster variant (foliage silhouettes): VS+FS, material set 0,
    // pipelines keyed by (stride << 8) | tcOffset.
    VkShaderModule                         s_DepthATVS       = VK_NULL_HANDLE;
    VkShaderModule                         s_DepthATFS       = VK_NULL_HANDLE;
    VkPipelineLayout                       s_DepthATLayout   = VK_NULL_HANDLE;
    std::unordered_map<u32, VkPipeline>    s_DepthATPipelines;

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

    // Push range, shared by VS+FS(+TCS/TES) — matches the PushConstants block
    // in world_{lmap,vlit}.{vert,frag,tesc,tese}.glsl:
    //   mat4  mvp         — offset 0,  64 bytes
    //   vec2  uvScale     — offset 64,  8 bytes
    //   float alphaRef    — offset 72   (<0 disables aref discard)
    //   float detailScale — offset 76
    //   float dynHemi     — offset 80
    //   float tessMax     — offset 84   (0 = tessellation off)
    //   float tessNear    — offset 88   (full-factor distance, m)
    //   float tessFar     — offset 92   (factor-1 / flat distance, m)
    //   vec4  eyeHeight   — offset 96   (xyz camera pos, w displacement amplitude)
    //   float pnScale     — offset 112  (PN-triangle curvature; TCS/TES only)
    // The 84..116 tail is owned by RenderQueue::Flush (re-pushed on layout
    // flips); FS/VS shaders only declare the first 84 bytes. The range MUST
    // cover pnScale (116) — the TCS/TES read it, so a 112-byte range left it
    // outside the layout and the shader sampled uninitialized memory.
    constexpr u32 kPushSize = 116;
}

// Defined below; GetTerrainPipeline() (right after Init) needs it forward.
static void BuildVertexInputForStride(u32 stride, u32 tcOffset,
                                      VkVertexInputBindingDescription&  binding,
                                      VkVertexInputAttributeDescription attrs[6]);

VkPipelineLayout GetLayout()      { return s_Layout; }
VkPipelineCache  GetCacheObject() { return s_CacheObject; }
VkShaderModule   WorldLmapVS() { return s_WorldLmapVS; }
VkShaderModule   WorldLmapFS() { return s_WorldLmapFS; }
VkShaderModule   WorldVlitVS() { return s_WorldVlitVS; }
VkShaderModule   WorldVlitFS() { return s_WorldVlitFS; }

bool TessAvailable()
{
    return VulkanHW.m_bTessellationSupported
        && s_WorldLmapTCS != VK_NULL_HANDLE && s_WorldLmapTES != VK_NULL_HANDLE
        && s_WorldVlitTCS != VK_NULL_HANDLE && s_WorldVlitTES != VK_NULL_HANDLE;
}

VkShaderStageFlags GetPushStages()
{
    VkShaderStageFlags f = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    if (VulkanHW.m_bTessellationSupported)
        f |= VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
    return f;
}

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

    // Terrain splat shaders — optional; absence just disables the terrain path.
    s_TerrainVS = g_ShaderManager->Load("world_terrain.vert.spv");
    s_TerrainFS = g_ShaderManager->Load("world_terrain.frag.spv");
    if (s_TerrainVS == VK_NULL_HANDLE || s_TerrainFS == VK_NULL_HANDLE)
        Msg("![VK PipelineCache] terrain splat shaders missing — terrain falls back to single-detail path");

    // World tessellation TCS/TES — optional; absence keeps the flat pipelines.
    if (VulkanHW.m_bTessellationSupported) {
        s_WorldLmapTCS = g_ShaderManager->Load("world_lmap.tesc.spv");
        s_WorldLmapTES = g_ShaderManager->Load("world_lmap.tese.spv");
        s_WorldVlitTCS = g_ShaderManager->Load("world_vlit.tesc.spv");
        s_WorldVlitTES = g_ShaderManager->Load("world_vlit.tese.spv");
        if (!TessAvailable())
            Msg("![VK PipelineCache] world tess shaders missing (world_*.tesc/.tese.spv) — tessellation disabled");
    }

    // Sun shadow caster VS — optional; absence disables shadow casting.
    s_DepthVS = g_ShaderManager->Load("shadow_depth.vert.spv");
    if (s_DepthVS == VK_NULL_HANDLE)
        Msg("![VK PipelineCache] shadow_depth.vert.spv missing — sun shadow casting disabled");
    // Alpha-tested caster pair — optional; absence keeps solid-quad shadows.
    s_DepthATVS = g_ShaderManager->Load("shadow_depth_at.vert.spv");
    s_DepthATFS = g_ShaderManager->Load("shadow_depth_at.frag.spv");
    if (s_DepthATVS == VK_NULL_HANDLE || s_DepthATFS == VK_NULL_HANDLE)
        Msg("![VK PipelineCache] shadow_depth_at.{vert,frag}.spv missing — alpha-tested casters disabled");

    VkPushConstantRange pc{};
    pc.stageFlags = GetPushStages();   // VS|FS (+TCS|TES when tess is supported)
    pc.offset     = 0;
    pc.size       = kPushSize;

    // set 0 = material textures; set 1 = per-frame env lighting (sun/hemi/ambient).
    VkDescriptorSetLayout envLayout = EnvLight::GetSetLayout();
    if (envLayout == VK_NULL_HANDLE) {
        Msg("![VK PipelineCache] EnvLight layout not initialised — call EnvLight::Init() first");
        return false;
    }
    VkDescriptorSetLayout setLayouts[2] = { WorldMaterialCache::GetSetLayout(), envLayout };
    if (setLayouts[0] == VK_NULL_HANDLE) {
        Msg("![VK PipelineCache] WorldMaterialCache layout not initialised — call WorldMaterialCache::Init() first");
        return false;
    }

    VkPipelineLayoutCreateInfo plci{};
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 2;
    plci.pSetLayouts            = setLayouts;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pc;
    if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_Layout) != VK_SUCCESS) {
        Msg("![VK PipelineCache] vkCreatePipelineLayout failed");
        return false;
    }

    // Terrain pipeline layout: set 0 = the 7-binding terrain set, set 1 = env lighting; same push range.
    if (s_TerrainVS != VK_NULL_HANDLE && s_TerrainFS != VK_NULL_HANDLE) {
        VkDescriptorSetLayout tset = WorldMaterialCache::GetTerrainSetLayout();
        if (tset != VK_NULL_HANDLE) {
            VkDescriptorSetLayout tsetLayouts[2] = { tset, envLayout };
            VkPipelineLayoutCreateInfo tplci{};
            tplci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            tplci.setLayoutCount         = 2;
            tplci.pSetLayouts            = tsetLayouts;
            tplci.pushConstantRangeCount = 1;
            tplci.pPushConstantRanges    = &pc;
            if (vkCreatePipelineLayout(VulkanHW.m_Device, &tplci, nullptr, &s_TerrainLayout) != VK_SUCCESS) {
                Msg("![VK PipelineCache] terrain pipeline layout failed — terrain path disabled");
                s_TerrainLayout = VK_NULL_HANDLE;
            }
        }
    }

    // Shadow caster pipeline layout: single push { mat4 lightMVP } (VERTEX), no sets.
    if (s_DepthVS != VK_NULL_HANDLE) {
        VkPushConstantRange dpc{ VK_SHADER_STAGE_VERTEX_BIT, 0, (u32)sizeof(Fmatrix) };
        VkPipelineLayoutCreateInfo dplci{};
        dplci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        dplci.setLayoutCount         = 0;
        dplci.pushConstantRangeCount = 1;
        dplci.pPushConstantRanges    = &dpc;
        if (vkCreatePipelineLayout(VulkanHW.m_Device, &dplci, nullptr, &s_DepthLayout) != VK_SUCCESS) {
            Msg("![VK PipelineCache] shadow depth layout failed — casting disabled");
            s_DepthLayout = VK_NULL_HANDLE;
        }
    }

    // Alpha-tested caster layout: set 0 = material textures (diffuse sampled in
    // FS), push { mat4; vec2 uvScale; float aref; pad } (VERTEX|FRAGMENT).
    if (s_DepthATVS != VK_NULL_HANDLE && s_DepthATFS != VK_NULL_HANDLE) {
        VkPushConstantRange apc{ VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                 (u32)(sizeof(Fmatrix) + 4 * sizeof(float)) };
        VkDescriptorSetLayout matLayout = WorldMaterialCache::GetSetLayout();
        VkPipelineLayoutCreateInfo aplci{};
        aplci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        aplci.setLayoutCount         = 1;
        aplci.pSetLayouts            = &matLayout;
        aplci.pushConstantRangeCount = 1;
        aplci.pPushConstantRanges    = &apc;
        if (vkCreatePipelineLayout(VulkanHW.m_Device, &aplci, nullptr, &s_DepthATLayout) != VK_SUCCESS) {
            Msg("![VK PipelineCache] AT shadow depth layout failed");
            s_DepthATLayout = VK_NULL_HANDLE;
        }
    }

    Msg("[VK PipelineCache] Init OK (layout, push=%u bytes, terrain=%s, shadow=%s)",
        kPushSize, s_TerrainLayout != VK_NULL_HANDLE ? "on" : "off",
        s_DepthLayout != VK_NULL_HANDLE ? "on" : "off");
    return true;
}

VkPipelineLayout GetDepthLayout() { return s_DepthLayout; }

// Depth-only caster pipeline for a given vertex stride. One vertex attribute
// (position FLOAT3 @ 0) + the stride-sized binding; depth write on, no color
// attachment, front-face depth bias to fight self-shadow acne.
VkPipeline GetDepthPipeline(u32 stride)
{
    if (s_DepthLayout == VK_NULL_HANDLE || s_DepthVS == VK_NULL_HANDLE) return VK_NULL_HANDLE;
    auto it = s_DepthPipelines.find(stride);
    if (it != s_DepthPipelines.end()) return it->second;

    VkVertexInputBindingDescription binding{ 0, stride, VK_VERTEX_INPUT_RATE_VERTEX };
    VkVertexInputAttributeDescription attr{ 0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0 };
    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount   = 1; vi.pVertexBindingDescriptions   = &binding;
    vi.vertexAttributeDescriptionCount = 1; vi.pVertexAttributeDescriptions = &attr;

    VkPipelineShaderStageCreateInfo stage{};   // VERTEX only — depth-only pass
    stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage  = VK_SHADER_STAGE_VERTEX_BIT; stage.module = s_DepthVS; stage.pName = "main";

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1; vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType        = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode  = VK_POLYGON_MODE_FILL;
    rs.cullMode     = VK_CULL_MODE_NONE;
    rs.frontFace    = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth    = 1.0f;
    rs.depthBiasEnable         = VK_TRUE;          // constant + slope bias vs acne (dynamic)
    rs.depthBiasConstantFactor = 0.0f;
    rs.depthBiasSlopeFactor    = 0.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable  = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendStateCreateInfo cb{};       // no color attachments
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 0;

    VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS };
    VkPipelineDynamicStateCreateInfo dynState{};
    dynState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynState.dynamicStateCount = 3; dynState.pDynamicStates = dyn;

    VkPipelineRenderingCreateInfo prci{};
    prci.sType                 = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    prci.colorAttachmentCount  = 0;
    prci.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;   // matches the shadow map

    VkGraphicsPipelineCreateInfo pi{};
    pi.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pi.pNext               = &prci;
    pi.stageCount          = 1;        pi.pStages = &stage;
    pi.pVertexInputState   = &vi;      pi.pInputAssemblyState = &ia;
    pi.pViewportState      = &vp;      pi.pRasterizationState = &rs;
    pi.pMultisampleState   = &ms;      pi.pDepthStencilState  = &ds;
    pi.pColorBlendState    = &cb;      pi.pDynamicState       = &dynState;
    pi.layout              = s_DepthLayout;

    VkPipeline h = VK_NULL_HANDLE;
    VkResult r = vkCreateGraphicsPipelines(VulkanHW.m_Device, s_CacheObject, 1, &pi, nullptr, &h);
    if (r != VK_SUCCESS) { Msg("![VK PipelineCache] shadow depth pipeline failed (%d) stride=%u", r, stride); h = VK_NULL_HANDLE; }
    else                 Msg("[VK PipelineCache] shadow depth pipeline stride=%u", stride);
    s_DepthPipelines.emplace(stride, h);
    return h;
}

VkPipelineLayout GetDepthATLayout() { return s_DepthATLayout; }

// Alpha-tested caster pipeline: position + UV attributes, VS+FS (FS discards
// transparent texels), otherwise identical to the solid depth pipeline.
VkPipeline GetDepthATPipeline(u32 stride, u32 tcOffset)
{
    if (s_DepthATLayout == VK_NULL_HANDLE) return VK_NULL_HANDLE;
    const u32 key = (stride << 8) | tcOffset;
    auto it = s_DepthATPipelines.find(key);
    if (it != s_DepthATPipelines.end()) return it->second;

    VkVertexInputBindingDescription binding{ 0, stride, VK_VERTEX_INPUT_RATE_VERTEX };
    VkVertexInputAttributeDescription attrs[2]{};
    attrs[0] = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0 };
    attrs[1] = { 1, 0, VK_FORMAT_R16G16_SSCALED,   tcOffset };
    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount   = 1; vi.pVertexBindingDescriptions   = &binding;
    vi.vertexAttributeDescriptionCount = 2; vi.pVertexAttributeDescriptions = attrs;

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;   stages[0].module = s_DepthATVS; stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = s_DepthATFS; stages[1].pName = "main";

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1; vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType        = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode  = VK_POLYGON_MODE_FILL;
    rs.cullMode     = VK_CULL_MODE_NONE;
    rs.frontFace    = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth    = 1.0f;
    rs.depthBiasEnable = VK_TRUE;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable  = VK_TRUE;
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendStateCreateInfo cb{};       // no color attachments
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;

    VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS };
    VkPipelineDynamicStateCreateInfo dynState{};
    dynState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynState.dynamicStateCount = 3; dynState.pDynamicStates = dyn;

    VkPipelineRenderingCreateInfo prci{};
    prci.sType                 = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    prci.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;

    VkGraphicsPipelineCreateInfo pi{};
    pi.sType             = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pi.pNext             = &prci;
    pi.stageCount        = 2;      pi.pStages             = stages;
    pi.pVertexInputState = &vi;    pi.pInputAssemblyState = &ia;
    pi.pViewportState    = &vp;    pi.pRasterizationState = &rs;
    pi.pMultisampleState = &ms;    pi.pDepthStencilState  = &ds;
    pi.pColorBlendState  = &cb;    pi.pDynamicState       = &dynState;
    pi.layout            = s_DepthATLayout;

    VkPipeline h = VK_NULL_HANDLE;
    VkResult r = vkCreateGraphicsPipelines(VulkanHW.m_Device, s_CacheObject, 1, &pi, nullptr, &h);
    if (r != VK_SUCCESS) { Msg("![VK PipelineCache] AT shadow pipeline failed (%d) stride=%u tcOff=%u", r, stride, tcOffset); h = VK_NULL_HANDLE; }
    else                 Msg("[VK PipelineCache] AT shadow pipeline stride=%u tcOff=%u", stride, tcOffset);
    s_DepthATPipelines.emplace(key, h);
    return h;
}

VkPipelineLayout GetTerrainLayout() { return s_TerrainLayout; }

VkPipeline GetTerrainPipeline()
{
    if (s_TerrainPipeline != VK_NULL_HANDLE) return s_TerrainPipeline;
    if (s_TerrainLayout == VK_NULL_HANDLE)   return VK_NULL_HANDLE;

    // Terrain is always the lmap sub-layout: stride 32, tcOffset 24, depth on.
    VkVertexInputBindingDescription   binding{};
    VkVertexInputAttributeDescription attrs[6]{};
    BuildVertexInputForStride(32, 24, binding, attrs);

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount   = 1;
    vi.pVertexBindingDescriptions      = &binding;
    vi.vertexAttributeDescriptionCount = 6;
    vi.pVertexAttributeDescriptions    = attrs;

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = s_TerrainVS; stages[0].pName = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = s_TerrainFS; stages[1].pName = "main";

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
    ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendAttachmentState ba{};
    ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    ba.blendEnable = VK_FALSE;
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
    pi.layout              = s_TerrainLayout;

    if (vkCreateGraphicsPipelines(VulkanHW.m_Device, s_CacheObject, 1, &pi, nullptr, &s_TerrainPipeline) != VK_SUCCESS) {
        Msg("![VK PipelineCache] terrain pipeline create failed");
        s_TerrainPipeline = VK_NULL_HANDLE;
        return VK_NULL_HANDLE;
    }
    Msg("[VK PipelineCache] Created terrain splat pipeline");
    return s_TerrainPipeline;
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

    if (s_TerrainPipeline) {
        vkDestroyPipeline(VulkanHW.m_Device, s_TerrainPipeline, nullptr);
        s_TerrainPipeline = VK_NULL_HANDLE;
    }
    if (s_TerrainLayout) {
        vkDestroyPipelineLayout(VulkanHW.m_Device, s_TerrainLayout, nullptr);
        s_TerrainLayout = VK_NULL_HANDLE;
    }
    for (auto& kv : s_DepthPipelines) if (kv.second) vkDestroyPipeline(VulkanHW.m_Device, kv.second, nullptr);
    s_DepthPipelines.clear();
    if (s_DepthLayout) {
        vkDestroyPipelineLayout(VulkanHW.m_Device, s_DepthLayout, nullptr);
        s_DepthLayout = VK_NULL_HANDLE;
    }
    s_DepthVS = VK_NULL_HANDLE;   // owned by g_ShaderManager
    for (auto& kv : s_DepthATPipelines) if (kv.second) vkDestroyPipeline(VulkanHW.m_Device, kv.second, nullptr);
    s_DepthATPipelines.clear();
    if (s_DepthATLayout) {
        vkDestroyPipelineLayout(VulkanHW.m_Device, s_DepthATLayout, nullptr);
        s_DepthATLayout = VK_NULL_HANDLE;
    }
    s_DepthATVS = VK_NULL_HANDLE;
    s_DepthATFS = VK_NULL_HANDLE;
    if (s_Layout) {
        vkDestroyPipelineLayout(VulkanHW.m_Device, s_Layout, nullptr);
        s_Layout = VK_NULL_HANDLE;
    }
    // Shader modules are owned by g_ShaderManager — leave them.
    s_TerrainVS = VK_NULL_HANDLE;
    s_TerrainFS = VK_NULL_HANDLE;
    s_WorldLmapVS = VK_NULL_HANDLE;
    s_WorldLmapFS = VK_NULL_HANDLE;
    s_WorldVlitVS = VK_NULL_HANDLE;
    s_WorldVlitFS = VK_NULL_HANDLE;
    s_WorldLmapTCS = VK_NULL_HANDLE;
    s_WorldLmapTES = VK_NULL_HANDLE;
    s_WorldVlitTCS = VK_NULL_HANDLE;
    s_WorldVlitTES = VK_NULL_HANDLE;
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
//
// loc 5 = NORMAL, D3DCOLOR @ 12 (STEP 3 dynamic lights): packed (N*0.5+0.5)
// with D3DCOLOR byte order B,G,R,A → reading as R8G8B8A8 the shader decodes
// N = attr.zyx*2-1 (y is order-independent; if x/z look flipped in-game the
// swizzle here in the shaders is the knob).
static void BuildVertexInputForStride(u32 stride, u32 tcOffset,
                                      VkVertexInputBindingDescription&    binding,
                                      VkVertexInputAttributeDescription   attrs[6])
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
    attrs[5] = { 5, 0, VK_FORMAT_R8G8B8A8_UNORM, 12 };   // packed vertex normal
}

static VkPipeline CreatePipeline(const Key& k)
{
    VkVertexInputBindingDescription   binding{};
    VkVertexInputAttributeDescription attrs[6]{};
    BuildVertexInputForStride(k.stride, k.tcOffset, binding, attrs);

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount   = 1;
    vi.pVertexBindingDescriptions      = &binding;
    vi.vertexAttributeDescriptionCount = 6;
    vi.pVertexAttributeDescriptions    = attrs;

    // Heightmap tessellation (R4 TESS_HM): the tess variant inserts the
    // TCS/TES pair (picked by sub-layout) and assembles patch lists. Guard
    // against keys built while the modules are unavailable.
    const bool tess = k.tess && TessAvailable();
    const VkShaderModule tcs = (k.tcOffset == 24) ? s_WorldLmapTCS : s_WorldVlitTCS;
    const VkShaderModule tes = (k.tcOffset == 24) ? s_WorldLmapTES : s_WorldVlitTES;

    u32 stageCount = 2;
    VkPipelineShaderStageCreateInfo stages[4]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = k.vs; stages[0].pName = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = k.fs; stages[1].pName = "main";
    if (tess) {
        stages[2].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[2].stage  = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
        stages[2].module = tcs; stages[2].pName = "main";
        stages[3].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[3].stage  = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
        stages[3].module = tes; stages[3].pName = "main";
        stageCount = 4;
    }

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = tess ? VK_PRIMITIVE_TOPOLOGY_PATCH_LIST : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineTessellationStateCreateInfo ts{};
    ts.sType              = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO;
    ts.patchControlPoints = 3;

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
    if (k.wmark) {
        // Baked level decals (newspapers/dirt) lie exactly on the surface
        // beneath — pull them towards the camera (LESS_OR_EQUAL: smaller
        // depth = closer → bias NEGATIVE) so they pass the depth test
        // without z-fight flicker.
        rs.depthBiasEnable         = VK_TRUE;
        rs.depthBiasConstantFactor = -2.0f;
        rs.depthBiasSlopeFactor    = -2.0f;
    } else if (tess) {
        // The depth prepass rasterizes these surfaces FLAT as plain triangles;
        // the tessellated color pass re-rasterizes the same planes as many
        // small triangles whose snapped vertices can land ±1 ulp off the
        // prepass depth. A ~1-gradient-step pull toward the camera makes the
        // un-displaced margins win those ties instead of sparkling.
        rs.depthBiasEnable         = VK_TRUE;
        rs.depthBiasConstantFactor = -2.0f;
        rs.depthBiasSlopeFactor    = -1.0f;
    }

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable  = k.depthTest ? VK_TRUE : VK_FALSE;
    ds.depthWriteEnable = (k.depthTest && !k.wmark) ? VK_TRUE : VK_FALSE;   // decals never write depth
    ds.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;
    // Phase 4 wires up an actual depth attachment; right now no depth target
    // is bound at draw time so these flags only affect future passes.

    VkPipelineColorBlendAttachmentState ba{};
    ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    ba.blendEnable    = VK_FALSE;
    if (k.wmark) {
        // Alpha-blend the decal over the lit surface (the texture alpha masks
        // the sheet/stain shape).
        ba.blendEnable         = VK_TRUE;
        ba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        ba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        ba.colorBlendOp        = VK_BLEND_OP_ADD;
        ba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        ba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        ba.alphaBlendOp        = VK_BLEND_OP_ADD;
    }

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
    if (k.depthTest) {
        prci.depthAttachmentFormat = Swapchain.m_DepthFormat;
        // Stencil aspect lives in the same view for combined formats but the
        // pass doesn't read or write it; leave stencilAttachmentFormat = UNDEFINED.
    }

    VkGraphicsPipelineCreateInfo pi{};
    pi.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pi.pNext               = &prci;
    pi.stageCount          = stageCount;
    pi.pStages             = stages;
    pi.pVertexInputState   = &vi;
    pi.pInputAssemblyState = &ia;
    pi.pTessellationState  = tess ? &ts : nullptr;
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
        Msg("![VK PipelineCache] vkCreateGraphicsPipelines failed (%d) for stride=%u tcOff=%u depth=%d tess=%d",
            r, k.stride, k.tcOffset, (int)k.depthTest, (int)tess);
        return VK_NULL_HANDLE;
    }
    Msg("[VK PipelineCache] Created pipeline stride=%u tcOff=%u depth=%d wmark=%d tess=%d",
        k.stride, k.tcOffset, (int)k.depthTest, (int)k.wmark, (int)tess);
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
