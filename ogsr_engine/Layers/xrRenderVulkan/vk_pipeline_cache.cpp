// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

#include "stdafx.h"
#include "vk_pipeline_cache.h"
#include "vk_gfx_pipeline.h"    // GfxPipelineBuilder
#include "vk_swapchain.h"
#include "vk_scene_color.h"   // HDR scene target format (passes render to it, not the swapchain)
#include "vk_shaders.h"
#include "vk_world_material.h"  // descriptor set 0 layout
#include "vk_env_light.h"       // descriptor set 1 layout (per-frame sun/hemi/ambient)
#include "HW_Vulkan.h"

#include <unordered_map>
#include <unordered_set>
#include <thread>          // weather prewarm runs off the render thread
#include <mutex>
#include <utility>

// Terrain POM depth offset (SSFX port): the getters pick the zoff pipeline
// variants (FS depth export) when both are on, so call sites stay unchanged.
extern int   ps_r_pom_terrain;   // r_pom_terrain (vk_console_min.cpp)
extern float ps_r_pom_zoff;      // r_pom_zoff — depth-offset strength (0 = off)
extern int   ps_r_vrs_static;    // r_vrs_static — diag: bake a static 2x2 rate into world pipelines
extern int   ps_r_uber_variants; // r_uber_variants — A/B: 0 forces the old monolithic world uber-FS (WS_ALL)

// r_prewarm_async -- see vk_console_min.cpp. Global scope on purpose: a
// namespace-scope extern mangles differently and silently fails to bind.
extern int ps_r_prewarm_async;

namespace VK { namespace PipelineCache {

namespace { void PrewarmTeardown(); }   // defined with the prewarm state, below

namespace {
    VkPipelineLayout                       s_Layout      = VK_NULL_HANDLE;
    VkPipelineCache                        s_CacheObject = VK_NULL_HANDLE;  // shared, disk-backed
    VkShaderModule                         s_WorldLmapVS = VK_NULL_HANDLE;
    VkShaderModule                         s_WorldLmapFS = VK_NULL_HANDLE;
    VkShaderModule                         s_WorldVlitVS = VK_NULL_HANDLE;
    VkShaderModule                         s_WorldVlitFS = VK_NULL_HANDLE;
    // Cluster-LOD crossfade variants (r_cluster_fade): same shaders + a flat
    // fade varying decoded from gl_InstanceIndex and a Bayer screen-door discard.
    // Used ONLY by the GPU-driven world path — the shared modules above must not
    // change (their interface is also consumed by the tess/wmark pipelines).
    VkShaderModule                         s_WorldLmapInstVS = VK_NULL_HANDLE;   // instanced host scene
    VkShaderModule                         s_WorldVlitInstVS = VK_NULL_HANDLE;
    VkShaderModule                         s_WorldLmapFadeVS = VK_NULL_HANDLE;
    VkShaderModule                         s_WorldLmapFadeFS = VK_NULL_HANDLE;
    VkShaderModule                         s_WorldVlitFadeVS = VK_NULL_HANDLE;
    VkShaderModule                         s_WorldVlitFadeFS = VK_NULL_HANDLE;
    // EARLY_ZTEST twins of the world FS modules (see Init + EarlyTwin).
    VkShaderModule                         s_WorldLmapEarlyFS     = VK_NULL_HANDLE;
    VkShaderModule                         s_WorldVlitEarlyFS     = VK_NULL_HANDLE;
    VkShaderModule                         s_WorldLmapFadeEarlyFS = VK_NULL_HANDLE;
    VkShaderModule                         s_WorldVlitFadeEarlyFS = VK_NULL_HANDLE;
    std::unordered_map<Key, VkPipeline>    s_Pipelines;
    // Frame-global world variant bits (WS_FRAME subset), stamped once per frame by
    // Pass_World via SetFrameSpecMask(). Default = full path until first stamp.
    u8                                     s_FrameSpecMask = WorldSpec::WS_FRAME;

    // World heightmap tessellation (R4 TESS_HM): TCS/TES pair per sub-layout.
    // All four must load (and the device feature be enabled) for tess keys.
    VkShaderModule                         s_WorldLmapTCS = VK_NULL_HANDLE;
    VkShaderModule                         s_WorldLmapTES = VK_NULL_HANDLE;
    VkShaderModule                         s_WorldVlitTCS = VK_NULL_HANDLE;
    VkShaderModule                         s_WorldVlitTES = VK_NULL_HANDLE;
    VkShaderModule                         s_WorldTerrainTCS = VK_NULL_HANDLE;  // snow footprint tessellation
    VkShaderModule                         s_WorldTerrainTES = VK_NULL_HANDLE;

    // Terrain splatting: own layout + single lazily-built pipeline + shaders.
    VkShaderModule                         s_TerrainVS       = VK_NULL_HANDLE;
    VkShaderModule                         s_TerrainFS       = VK_NULL_HANDLE;
    VkPipelineLayout                       s_TerrainLayout   = VK_NULL_HANDLE;
    // Terrain COLOR pipelines keyed by (zoff<<8 | world-variant mask) — the uber-FS
    // spec variants (Inc 1) apply to terrain too. Depth-prepass variants stay single.
    std::unordered_map<u32, VkPipeline>    s_TerrainColorPipelines;
    VkPipeline                             s_TerrainDepthPipeline = VK_NULL_HANDLE;   // snow-displaced depth-prepass variant
    // Terrain POM DEPTH OFFSET variants (r_pom_zoff, SSFX port): color FS that
    // exports the displaced gl_FragDepth + a prepass FS writing the same depth
    // (so GTAO / VSM resolve see the carved surface). Getters switch on the cvar.
    VkShaderModule                         s_TerrainFSZoff        = VK_NULL_HANDLE;
    VkShaderModule                         s_TerrainDepthFS       = VK_NULL_HANDLE;
    VkPipeline                             s_TerrainDepthPipelineZoff = VK_NULL_HANDLE;

    // Sun shadow caster: depth-only VS (no FS), own layout (push: mat4 lightMVP),
    // per-stride lazily-built pipelines into the shadow map's D32 format.
    VkShaderModule                         s_DepthVS         = VK_NULL_HANDLE;
    VkPipelineLayout                       s_DepthLayout     = VK_NULL_HANDLE;
    std::unordered_map<u32, VkPipeline>    s_DepthPipelines;

    // Depth-prepass dither variant for the cluster-LOD crossfade (vk_world_gpu
    // DrawDepth when r_cluster_fade > 0). Same layout as the solid depth path.
    VkShaderModule                         s_DepthFadeVS     = VK_NULL_HANDLE;
    VkShaderModule                         s_DepthFadeFS     = VK_NULL_HANDLE;
    std::unordered_map<u32, VkPipeline>    s_DepthFadePipelines;

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
    // The 84..120 tail is owned by RenderQueue::Flush (re-pushed on layout
    // flips); FS shaders only declare the first 84 bytes. The range MUST
    // cover pnScale (116) — the TCS/TES read it, so a 112-byte range left it
    // outside the layout and the shader sampled uninitialized memory.
    // DYNAMIC visuals overlay 84..120 with the model rows {i,j,c} (9 floats) —
    // the VS rebuilds world-space vWorldPos/vNormal from them (dynHemi < -0.5
    // flags it); dynamics never tessellate, statics never read the overlay.
    constexpr u32 kPushSize = 120;
}

// Defined below; GetTerrainPipeline() (right after Init) needs it forward.
static void BuildVertexInputForStride(u32 stride, u32 tcOffset,
                                      VkVertexInputBindingDescription&  binding,
                                      VkVertexInputAttributeDescription attrs[6]);

// Fill a VkSpecializationInfo (+ its backing storage) for the 5 world uber-FS variant
// bits. The caller owns vals/entries/info so they outlive vkCreateGraphicsPipelines.
// Order MUST match the SpecId decorations in shaders/world_variants.glsl (POM=0 ..
// DEBUG=4, FEEDBACK=5). Passed to the FRAGMENT stage only (VS/TCS/TES declare no
// spec constants).
static void BuildWorldSpecInfo(u8 mask, VkBool32 vals[6],
                               VkSpecializationMapEntry entries[6], VkSpecializationInfo& info)
{
    const u8 bits[6] = { WS_POM, WS_SNOW, WS_WET, WS_IBL, WS_DEBUG, WS_FEEDBACK };
    for (u32 i = 0; i < 6; ++i) {
        vals[i]    = (mask & bits[i]) ? VK_TRUE : VK_FALSE;
        entries[i] = { i, i * (u32)sizeof(VkBool32), sizeof(VkBool32) };
    }
    info = {};
    info.mapEntryCount = 6;
    info.pMapEntries   = entries;
    info.dataSize      = 6 * sizeof(VkBool32);
    info.pData         = vals;
}

void SetFrameSpecMask(u8 mask) { s_FrameSpecMask = mask & WS_FRAME; }
u8   GetFrameSpecMask()        { return s_FrameSpecMask; }

VkPipelineLayout GetLayout()      { return s_Layout; }
VkPipelineCache  GetCacheObject() { return s_CacheObject; }
VkShaderModule   WorldLmapVS() { return s_WorldLmapVS; }
VkShaderModule   WorldLmapFS() { return s_WorldLmapFS; }
VkShaderModule   WorldVlitVS() { return s_WorldVlitVS; }
VkShaderModule   WorldVlitFS() { return s_WorldVlitFS; }
VkShaderModule   WorldLmapInstVS() { return s_WorldLmapInstVS; }
VkShaderModule   WorldVlitInstVS() { return s_WorldVlitInstVS; }
VkShaderModule   WorldLmapFadeVS() { return s_WorldLmapFadeVS; }
VkShaderModule   WorldLmapFadeFS() { return s_WorldLmapFadeFS; }
VkShaderModule   WorldVlitFadeVS() { return s_WorldVlitFadeVS; }
VkShaderModule   WorldVlitFadeFS() { return s_WorldVlitFadeFS; }

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
    s_WorldLmapFadeVS = g_ShaderManager->Load("world_lmap_fade.vert.spv");
    s_WorldLmapFadeFS = g_ShaderManager->Load("world_lmap_fade.frag.spv");
    s_WorldVlitFadeVS = g_ShaderManager->Load("world_vlit_fade.vert.spv");
    s_WorldVlitFadeFS = g_ShaderManager->Load("world_vlit_fade.frag.spv");
    if (!s_WorldLmapFadeVS || !s_WorldLmapFadeFS || !s_WorldVlitFadeVS || !s_WorldVlitFadeFS)
        Msg("![VK PipelineCache] world *_fade shaders missing — cluster crossfade (r_cluster_fade) disabled");
    // EARLY_ZTEST twins (same sources, glslc -DEARLY_ZTEST): forced early depth
    // test so the streaming-feedback atomic (an FS side effect that forbids
    // automatic early-Z) never executes for occluded fragments. Bound ONLY to
    // no-z-write statics pipelines — see EarlyTwin() in CreatePipeline.
    // Optional: missing modules fall back to the normal FS (correct, just slow).
    s_WorldLmapEarlyFS     = g_ShaderManager->Load("world_lmap_earlyz.frag.spv");
    s_WorldVlitEarlyFS     = g_ShaderManager->Load("world_vlit_earlyz.frag.spv");
    s_WorldLmapFadeEarlyFS = g_ShaderManager->Load("world_lmap_fade_earlyz.frag.spv");
    s_WorldVlitFadeEarlyFS = g_ShaderManager->Load("world_vlit_fade_earlyz.frag.spv");
    if (!s_WorldLmapEarlyFS || !s_WorldVlitEarlyFS)
        Msg("![VK PipelineCache] world *_earlyz shaders missing — statics keep the late-Z uber-FS (slow)");

    // Instanced world VS variants (vk_instance_gpu, host/editor scenes). Optional:
    // missing modules just keep the host scene on the CPU RenderQueue.
    s_WorldLmapInstVS = g_ShaderManager->Load("world_lmap_inst.vert.spv");
    s_WorldVlitInstVS = g_ShaderManager->Load("world_vlit_inst.vert.spv");
    if (!s_WorldLmapInstVS || !s_WorldVlitInstVS)
        Msg("![VK PipelineCache] world *_inst shaders missing — instanced host scenes stay on the CPU path");
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
    // POM depth-offset variants (r_pom_zoff) — optional; absence keeps flat depth.
    s_TerrainFSZoff  = g_ShaderManager->Load("world_terrain_zoff.frag.spv");
    s_TerrainDepthFS = g_ShaderManager->Load("world_terrain_depth.frag.spv");
    if (s_TerrainFSZoff == VK_NULL_HANDLE || s_TerrainDepthFS == VK_NULL_HANDLE)
        Msg("![VK PipelineCache] terrain zoff shaders missing (world_terrain_zoff/_depth.frag.spv) — r_pom_zoff disabled");

    // World tessellation TCS/TES — optional; absence keeps the flat pipelines.
    if (VulkanHW.m_bTessellationSupported) {
        s_WorldLmapTCS = g_ShaderManager->Load("world_lmap.tesc.spv");
        s_WorldLmapTES = g_ShaderManager->Load("world_lmap.tese.spv");
        s_WorldVlitTCS = g_ShaderManager->Load("world_vlit.tesc.spv");
        s_WorldVlitTES = g_ShaderManager->Load("world_vlit.tese.spv");
        s_WorldTerrainTCS = g_ShaderManager->Load("world_terrain.tesc.spv");   // snow footprint tess
        s_WorldTerrainTES = g_ShaderManager->Load("world_terrain.tese.spv");
        if (!TessAvailable())
            Msg("![VK PipelineCache] world tess shaders missing (world_*.tesc/.tese.spv) — tessellation disabled");
    }

    // Sun shadow caster VS — optional; absence disables shadow casting.
    s_DepthVS = g_ShaderManager->Load("shadow_depth.vert.spv");
    if (s_DepthVS == VK_NULL_HANDLE)
        Msg("![VK PipelineCache] shadow_depth.vert.spv missing — sun shadow casting disabled");

    s_DepthFadeVS = g_ShaderManager->Load("world_depth_fade.vert.spv");
    s_DepthFadeFS = g_ShaderManager->Load("world_depth_fade.frag.spv");
    if (s_DepthFadeVS == VK_NULL_HANDLE || s_DepthFadeFS == VK_NULL_HANDLE)
        Msg("![VK PipelineCache] world_depth_fade.{vert,frag}.spv missing — cluster crossfade prepass disabled");
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

    // VERTEX only, no color attachments — depth-only pass.
    VkPipeline h = VK::GfxPipelineBuilder(s_DepthLayout)
        .Vert(s_DepthVS)
        .Binding(0, stride)
        .Attr(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0)
        .Cull(VK_CULL_MODE_NONE)
        .DynamicDepthBias()                  // constant + slope bias vs acne (dynamic)
        .Depth(true, true)
        .DepthTarget(VK_FORMAT_D32_SFLOAT)   // matches the shadow map
        .Build("PipelineCache shadow depth stride=%u", stride);
    if (h != VK_NULL_HANDLE) Msg("[VK PipelineCache] shadow depth pipeline stride=%u", stride);
    s_DepthPipelines.emplace(stride, h);
    return h;
}

// Solid depth pipeline + the crossfade dither FS (cluster-LOD transitions).
// Identical state to GetDepthPipeline — same layout, same D32 target — so
// vk_world_gpu::DrawDepth can swap them per frame based on r_cluster_fade.
VkPipeline GetDepthFadePipeline(u32 stride)
{
    if (s_DepthLayout == VK_NULL_HANDLE || s_DepthFadeVS == VK_NULL_HANDLE || s_DepthFadeFS == VK_NULL_HANDLE)
        return VK_NULL_HANDLE;
    auto it = s_DepthFadePipelines.find(stride);
    if (it != s_DepthFadePipelines.end()) return it->second;

    VkPipeline h = VK::GfxPipelineBuilder(s_DepthLayout)
        .Vert(s_DepthFadeVS).Frag(s_DepthFadeFS)
        .Binding(0, stride)
        .Attr(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0)
        .Cull(VK_CULL_MODE_NONE)
        .DynamicDepthBias()
        .Depth(true, true)
        .DepthTarget(VK_FORMAT_D32_SFLOAT)
        .Build("PipelineCache depth fade stride=%u", stride);
    s_DepthFadePipelines.emplace(stride, h);
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

    VkPipeline h = VK::GfxPipelineBuilder(s_DepthATLayout)   // no color attachments
        .Vert(s_DepthATVS).Frag(s_DepthATFS)
        .Binding(0, stride)
        .Attr(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0)
        .Attr(1, 0, VK_FORMAT_R16G16_SSCALED,   tcOffset)
        .Cull(VK_CULL_MODE_NONE)
        .DynamicDepthBias()
        .Depth(true, true)
        .DepthTarget(VK_FORMAT_D32_SFLOAT)
        .Build("PipelineCache AT shadow stride=%u tcOff=%u", stride, tcOffset);
    if (h != VK_NULL_HANDLE) Msg("[VK PipelineCache] AT shadow pipeline stride=%u tcOff=%u", stride, tcOffset);
    s_DepthATPipelines.emplace(key, h);
    return h;
}

VkPipelineLayout GetTerrainLayout() { return s_TerrainLayout; }

// True when the POM depth-offset variants should be used this frame (cvar pair
// on + both variant shaders loaded). The zoff COLOR pipeline must run at full
// shading rate: a VRS-coarsened gl_FragDepth would differ from the full-rate
// prepass depth and fail the LEQUAL equality -> holes. So the zoff variant is
// built WITHOUT the VRS dynamic state.
static bool TerrainZoffActive()
{
    return ps_r_pom_terrain && ps_r_pom_zoff > 0.f
        && s_TerrainFSZoff != VK_NULL_HANDLE && s_TerrainDepthFS != VK_NULL_HANDLE;
}

VkPipeline GetTerrainPipeline()
{
    if (s_TerrainLayout == VK_NULL_HANDLE)   return VK_NULL_HANDLE;
    const bool zoff = TerrainZoffActive();
    // Terrain uber-FS variant mask (Inc 1): frame-global bits + terrain POM (r_pom_terrain).
    const u8  mask = (ps_r_uber_variants == 0) ? (u8)WS_ALL
                     : (u8)((s_FrameSpecMask & WS_FRAME) | (ps_r_pom_terrain ? (u8)WS_POM : 0u));
    const u32 pkey = ((zoff ? 1u : 0u) << 8) | mask;
    if (auto it = s_TerrainColorPipelines.find(pkey); it != s_TerrainColorPipelines.end())
        return it->second;

    // Terrain is always the lmap sub-layout: stride 32, tcOffset 24, depth on.
    VkVertexInputBindingDescription   binding{};
    VkVertexInputAttributeDescription attrs[6]{};
    BuildVertexInputForStride(32, 24, binding, attrs);

    // Snow footprint tessellation: VS -> TCS -> TES -> FS over patch lists when the
    // device supports it + the terrain TCS/TES loaded (the TCS gates tess to snow areas).
    const bool tess = VulkanHW.m_bTessellationSupported && s_WorldTerrainTCS != VK_NULL_HANDLE && s_WorldTerrainTES != VK_NULL_HANDLE;
    // Bake the world-variant spec constants into the terrain FS (storage lives to the create).
    VkBool32                 specVals[6];
    VkSpecializationMapEntry specEntries[6];
    VkSpecializationInfo     specInfo;
    BuildWorldSpecInfo(mask, specVals, specEntries, specInfo);

    VK::GfxPipelineBuilder b(s_TerrainLayout);
    b.Vert(s_TerrainVS);
    if (tess) b.Tess(s_WorldTerrainTCS, s_WorldTerrainTES);
    b.Frag(zoff ? s_TerrainFSZoff : s_TerrainFS, &specInfo)
     .Bindings(&binding, 1).Attrs(attrs, 6)
     .Cull(VK_CULL_MODE_NONE)
     .Depth(true, true)
     .Color(VK::SceneColor::Format())
     .DepthTarget(Swapchain.m_DepthFormat);

    // VRS: STATIC {1x1, KEEP, REPLACE} like the world pipelines (see CreatePipeline —
    // a dynamic rate gets invalidated by unrelated binds). zoff variant: NO VRS
    // (full-rate gl_FragDepth must equal the prepass depth) → no FSR state at all
    // (implicit static 1x1, combiners KEEP/KEEP ignore the SRI).
    VkPipelineFragmentShadingRateStateCreateInfoKHR fsrState{ VK_STRUCTURE_TYPE_PIPELINE_FRAGMENT_SHADING_RATE_STATE_CREATE_INFO_KHR };
    fsrState.fragmentSize   = { 1, 1 };
    fsrState.combinerOps[0] = VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR;
    fsrState.combinerOps[1] = VK_FRAGMENT_SHADING_RATE_COMBINER_OP_REPLACE_KHR;   // SRI attachment wins
    if (ps_r_vrs_static > 0) { fsrState.fragmentSize = { 2, 2 }; fsrState.combinerOps[1] = VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR; }
    if (VulkanHW.m_bVRSSupported && !zoff) b.RenderingNext(&fsrState);
    // Same SRI pass as the world pipelines above → same create flag. The zoff
    // variant keeps its static 1x1 rate (no FSR state) but still needs the
    // flag to be legally bound while the SRI is attached.
    if (VulkanHW.m_bVRSSupported)
        b.CreateFlags(VK_PIPELINE_CREATE_RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR);

    VkPipeline h = b.Build("PipelineCache terrain (zoff=%d mask=0x%02x)", zoff ? 1 : 0, mask);
    if (h == VK_NULL_HANDLE) {
        s_TerrainColorPipelines.emplace(pkey, VK_NULL_HANDLE);
        return VK_NULL_HANDLE;
    }
    Msg("[VK PipelineCache] Created terrain splat pipeline (zoff=%d mask=0x%02x)", zoff ? 1 : 0, mask);
    s_TerrainColorPipelines.emplace(pkey, h);
    return h;
}

// Terrain DEPTH-prepass variant: the SAME world_terrain.vert (so the snow vertex
// displacement is IDENTICAL to the color pass -> the prepass depth matches, no
// z-fight / see-through). Vertex-only (depth-only, no FS), terrain layout (set 1 =
// EnvLight, for sf_params.w in the VS). Depth bias dynamic (the prepass sets 0).
// Used by FlushDepth / WorldGPU::DrawDepth for terrain in the depth prepass only.
VkPipeline GetTerrainDepthPipeline()
{
    const bool  zoff = TerrainZoffActive();   // FS variant: sink depth into the POM cracks
    VkPipeline& slot = zoff ? s_TerrainDepthPipelineZoff : s_TerrainDepthPipeline;
    if (slot != VK_NULL_HANDLE) return slot;
    if (s_TerrainLayout == VK_NULL_HANDLE || s_TerrainVS == VK_NULL_HANDLE) return VK_NULL_HANDLE;

    VkVertexInputBindingDescription   binding{};
    VkVertexInputAttributeDescription attrs[6]{};
    BuildVertexInputForStride(32, 24, binding, attrs);

    // Tessellated (snow footprints) when available - MUST match the color terrain
    // pipeline's displacement so the prepass depth lines up (no z-fight on prints).
    const bool tess = VulkanHW.m_bTessellationSupported && s_WorldTerrainTCS != VK_NULL_HANDLE && s_WorldTerrainTES != VK_NULL_HANDLE;

    VK::GfxPipelineBuilder b(s_TerrainLayout);   // no color attachments
    b.Vert(s_TerrainVS);
    if (tess) b.Tess(s_WorldTerrainTCS, s_WorldTerrainTES);
    if (zoff) b.Frag(s_TerrainDepthFS);   // prepass FS: same POM march as color -> displaced gl_FragDepth
    b.Bindings(&binding, 1).Attrs(attrs, 6)
     .Cull(VK_CULL_MODE_NONE)
     .DynamicDepthBias()                     // prepass sets 0 (match color)
     .Depth(true, true)
     .DepthTarget(Swapchain.m_DepthFormat);  // the scene prepass depth

    slot = b.Build("PipelineCache terrain DEPTH (zoff=%d)", zoff ? 1 : 0);
    if (slot == VK_NULL_HANDLE) return VK_NULL_HANDLE;
    Msg("[VK PipelineCache] Created terrain depth (snow-displaced) pipeline (zoff=%d)", zoff ? 1 : 0);
    return slot;
}

void Destroy()
{
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;

    // The prewarm worker holds no lock on anything we are about to free, but it
    // is still creating pipelines against this device -- join before the teardown.
    PrewarmTeardown();

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

    for (auto& kv : s_TerrainColorPipelines)
        if (kv.second) vkDestroyPipeline(VulkanHW.m_Device, kv.second, nullptr);
    s_TerrainColorPipelines.clear();
    if (s_TerrainDepthPipeline) {
        vkDestroyPipeline(VulkanHW.m_Device, s_TerrainDepthPipeline, nullptr);
        s_TerrainDepthPipeline = VK_NULL_HANDLE;
    }
    if (s_TerrainDepthPipelineZoff) {
        vkDestroyPipeline(VulkanHW.m_Device, s_TerrainDepthPipelineZoff, nullptr);
        s_TerrainDepthPipelineZoff = VK_NULL_HANDLE;
    }
    if (s_TerrainLayout) {
        vkDestroyPipelineLayout(VulkanHW.m_Device, s_TerrainLayout, nullptr);
        s_TerrainLayout = VK_NULL_HANDLE;
    }
    for (auto& kv : s_DepthPipelines) if (kv.second) vkDestroyPipeline(VulkanHW.m_Device, kv.second, nullptr);
    s_DepthPipelines.clear();
    for (auto& kv : s_DepthFadePipelines) if (kv.second) vkDestroyPipeline(VulkanHW.m_Device, kv.second, nullptr);
    s_DepthFadePipelines.clear();
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
    s_TerrainFSZoff  = VK_NULL_HANDLE;
    s_TerrainDepthFS = VK_NULL_HANDLE;
    s_WorldLmapVS = VK_NULL_HANDLE;
    s_WorldLmapFS = VK_NULL_HANDLE;
    s_WorldVlitVS = VK_NULL_HANDLE;
    s_WorldVlitFS = VK_NULL_HANDLE;
    s_WorldLmapTCS = VK_NULL_HANDLE;
    s_WorldLmapTES = VK_NULL_HANDLE;
    s_WorldVlitTCS = VK_NULL_HANDLE;
    s_WorldVlitTES = VK_NULL_HANDLE;
    s_WorldTerrainTCS = VK_NULL_HANDLE;
    s_WorldTerrainTES = VK_NULL_HANDLE;
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

// EARLY_ZTEST twin of a world FS module, or the module itself when no twin is
// loaded. The twin forces the depth test BEFORE the shader, so the streaming-
// feedback atomic (an FS side effect — automatic early-Z is illegal with it)
// only ever runs for visible fragments. ONLY legal without depth writes (early
// tests write depth before the AT discard) — the callers below bind it exactly
// for the atEqual/noZWrite statics variants, which force depthWrite off.
static VkShaderModule EarlyTwin(VkShaderModule fs)
{
    if (fs == s_WorldLmapFS     && s_WorldLmapEarlyFS)     return s_WorldLmapEarlyFS;
    if (fs == s_WorldVlitFS     && s_WorldVlitEarlyFS)     return s_WorldVlitEarlyFS;
    if (fs == s_WorldLmapFadeFS && s_WorldLmapFadeEarlyFS) return s_WorldLmapFadeEarlyFS;
    if (fs == s_WorldVlitFadeFS && s_WorldVlitFadeEarlyFS) return s_WorldVlitFadeEarlyFS;
    return fs;
}

static VkPipeline CreatePipeline(const Key& k)
{
    VkVertexInputBindingDescription   binding{};
    VkVertexInputAttributeDescription attrs[6]{};
    BuildVertexInputForStride(k.stride, k.tcOffset, binding, attrs);

    // Instanced host scene (Key::instanced): a SECOND binding at INSTANCE rate
    // carries the model matrix as four vec4 rows on locations 6..9, matching the
    // INSTANCED block in world_{lmap,vlit}_vert_body.glsl. Stride is sizeof(Fmatrix);
    // the indirect draw's firstInstance selects the row, so one draw covers every
    // instance of a mesh. The arrays must outlive vkCreateGraphicsPipelines, hence
    // the full-size locals rather than a branch-local copy.
    VkVertexInputBindingDescription   bindings[2]{};
    VkVertexInputAttributeDescription attrsInst[10]{};
    if (k.instanced) {
        bindings[0] = binding;
        bindings[1] = { 1, (u32)sizeof(Fmatrix), VK_VERTEX_INPUT_RATE_INSTANCE };
        for (u32 i = 0; i < 6; ++i) attrsInst[i] = attrs[i];
        for (u32 r = 0; r < 4; ++r)
            attrsInst[6 + r] = { 6 + r, 1, VK_FORMAT_R32G32B32A32_SFLOAT, r * 16u };
    }

    // Heightmap tessellation (R4 TESS_HM): the tess variant inserts the
    // TCS/TES pair (picked by sub-layout) and assembles patch lists. Guard
    // against keys built while the modules are unavailable.
    const bool tess = k.tess && TessAvailable();
    const VkShaderModule tcs = (k.tcOffset == 24) ? s_WorldLmapTCS : s_WorldVlitTCS;
    const VkShaderModule tes = (k.tcOffset == 24) ? s_WorldLmapTES : s_WorldVlitTES;

    // No-z-write statics (atEqual/noZWrite): swap in the EARLY_ZTEST twin —
    // matches the depth-state block below, which forces depthWrite off for
    // exactly this condition (early tests + z-write would break AT discard).
    const bool earlyZ = (k.atEqual || k.noZWrite) && !tess && !k.wmark && !k.emis;
    // World uber-FS variants (Inc 1): bake POM/SNOW/WET/IBL/DEBUG/FEEDBACK spec constants
    // so the driver DCEs the unused features (and their VGPRs). Storage lives to the create call.
    VkBool32                 specVals[6];
    VkSpecializationMapEntry specEntries[6];
    VkSpecializationInfo     specInfo;
    BuildWorldSpecInfo(k.specMask, specVals, specEntries, specInfo);

    VK::GfxPipelineBuilder b(s_Layout);
    b.Vert(k.vs).Frag(earlyZ ? EarlyTwin(k.fs) : k.fs, &specInfo);
    if (tess) b.Tess(tcs, tes);
    b.Bindings(k.instanced ? bindings : &binding, k.instanced ? 2u : 1u)
     .Attrs(k.instanced ? attrsInst : attrs, k.instanced ? 10u : 6u)
     .Cull(VK_CULL_MODE_NONE);   // no winding info yet → don't drop faces
    if (k.wmark && !k.emis) {   // emissive billboards float free — no decal bias
        // Baked level decals (newspapers/dirt) lie exactly on the surface
        // beneath — pull them towards the camera (LESS_OR_EQUAL: smaller
        // depth = closer → bias NEGATIVE) so they pass the depth test
        // without z-fight flicker.
        b.DepthBias(-2.0f, -2.0f);
    } else if (tess) {
        // The depth prepass rasterizes these surfaces FLAT as plain triangles;
        // the tessellated color pass re-rasterizes the same planes as many
        // small triangles whose snapped vertices can land ±1 ulp off the
        // prepass depth. A ~1-gradient-step pull toward the camera makes the
        // un-displaced margins win those ties instead of sparkling.
        b.DepthBias(-2.0f, -1.0f);
    }

    bool        depthWrite = k.depthTest && !k.wmark;   // decals never write depth
    VkCompareOp depthOp    = VK_COMPARE_OP_LESS_OR_EQUAL;
    // AT statics (r_at_equal): the prepass depth IS this geometry's final depth,
    // so EQUAL + no-write lets early-Z kill occluded AT layers and the front
    // layer's transparent texels without invoking the FS. Depth equality holds
    // because both VS paths compute gl_Position = mvp * vec4(pos, 1) from the
    // same pushed matrix. Never combined with tess (re-rasterized with bias).
    // Prepass-covered statics (earlyZ = same condition that bound the
    // EARLY_ZTEST FS twin above — the twin is only legal WITHOUT depth writes):
    // the prepass already wrote these items' final depth, so drop the redundant
    // write. AT items additionally test EQUAL (r_at_equal) — the prepass depth
    // IS the front opaque texel, so early-Z also kills the transparent texels
    // and every occluded AT layer. Opaque keep LEQUAL (safe on prepass gaps).
    if (earlyZ) {
        depthWrite = false;
        if (k.atEqual) depthOp = VK_COMPARE_OP_EQUAL;
    }
    // Phase 4 wires up an actual depth attachment; right now no depth target
    // is bound at draw time so these flags only affect future passes.
    b.Depth(k.depthTest, depthWrite, depthOp);

    b.Color(VK::SceneColor::Format());
    if (k.emis) {
        // Emissive-additive (glow billboards / selflight parts): the texture ADDS
        // light over the scene — never darkens, alpha shapes the halo.
        b.BlendAdd();
    }
    else if (k.wmark) {
        // Alpha-blend the decal over the lit surface (the texture alpha masks
        // the sheet/stain shape).
        b.BlendAlpha();
    }
    if (k.depthTest) {
        b.DepthTarget(Swapchain.m_DepthFormat);
        // Stencil aspect lives in the same view for combined formats but the
        // pass doesn't read or write it; leave stencilAttachmentFormat = UNDEFINED.
    }

    // VRS: STATIC per-pipeline FSR state {1x1, KEEP, REPLACE} — the attachment
    // (SRI) replaces the rate when Pass_World attaches one; passes without an SRI
    // shade 1x1. Static, NOT the dynamic state: validation proved a dynamic rate
    // can't survive this renderer's command buffers — any vkCmdBindPipeline of a
    // pipeline WITHOUT the dynamic state (trees, VSM, shadow, debug...) legally
    // INVALIDATES the previously set rate, so every draw after it fell back to
    // 1x1 (VUID-...-pipelineFragmentShadingRate-09238, 18-07-2026).
    // r_vrs_static diag: bake a hard 2x2 (KEEP/KEEP ignores the SRI) instead.
    VkPipelineFragmentShadingRateStateCreateInfoKHR fsrState{ VK_STRUCTURE_TYPE_PIPELINE_FRAGMENT_SHADING_RATE_STATE_CREATE_INFO_KHR };
    if (k.vrsStatic) {
        fsrState.fragmentSize   = { 2, 2 };                                        // diag: hard 2x2
        fsrState.combinerOps[0] = VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR;
        fsrState.combinerOps[1] = VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR;   // ignore attachment
    } else {
        fsrState.fragmentSize   = { 1, 1 };
        fsrState.combinerOps[0] = VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR;   // ignore primitive rate
        fsrState.combinerOps[1] = VK_FRAGMENT_SHADING_RATE_COMBINER_OP_REPLACE_KHR;// SRI attachment wins
    }
    // These pipelines draw inside a dynamic-rendering pass that attaches the
    // shading-rate image (Pass_World). Without the create flag the spec leaves the
    // SRI's effect undefined — NVIDIA silently ignores it (draws stay 1x1).
    if (VulkanHW.m_bVRSSupported)
        b.RenderingNext(&fsrState).CreateFlags(VK_PIPELINE_CREATE_RENDERING_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR);

    VkPipeline handle = b.Build("PipelineCache world stride=%u tcOff=%u depth=%d tess=%d",
                                k.stride, k.tcOffset, (int)k.depthTest, (int)tess);
    if (handle == VK_NULL_HANDLE) return VK_NULL_HANDLE;
    Msg("[VK PipelineCache] Created pipeline stride=%u tcOff=%u depth=%d wmark=%d tess=%d vrs2x2=%d",
        k.stride, k.tcOffset, (int)k.depthTest, (int)k.wmark, (int)tess, (int)k.vrsStatic);
    return handle;
}

VkPipeline Get(const Key& key)
{
    if (!s_Layout) return VK_NULL_HANDLE;

    // VRS diag: stamp the static-2x2 bit from the LIVE cvar so flipping
    // r_vrs_static mid-game switches to (lazily created) 2x2 variants — no
    // level reload needed for the A/B.
    Key k = key;
    k.vrsStatic = VulkanHW.m_bVRSSupported && ps_r_vrs_static > 0;
    // r_uber_variants 0 = A/B the OLD monolithic uber (all features baked on). Else
    // stamp the frame-global bits (WS_FRAME) from the per-frame mask; the caller supplied
    // only the per-material POM bit. Flipping weather/debug (or this cvar) mid-game
    // switches to (lazily created, disk-cached) variants — no level reload needed.
    k.specMask = (ps_r_uber_variants == 0) ? (u8)WS_ALL
                                           : (u8)((key.specMask & WS_POM) | s_FrameSpecMask);

    auto it = s_Pipelines.find(k);
    if (it != s_Pipelines.end()) return it->second;

    VkPipeline p = CreatePipeline(k);
    s_Pipelines.emplace(k, p);
    return p;
}

namespace {
    xr_vector<Key> s_prewarmWorld;       // pending world weather-variant keys
    xr_vector<u8>  s_prewarmTerrainMask; // pending terrain frame-masks (WS_FRAME bits)
    bool           s_prewarmBuilt = false;

    // ---- Async prewarm -------------------------------------------------------
    // These compiles used to run one per frame ON the render thread, and the
    // precache countdown waited for them: measured on pripyat_full, that is 17
    // frames of load screen (~12 ms each) spent compiling variants nothing draws
    // until it rains. Worse, the 1/frame pace was chosen for a COLD shader cache
    // (~300 ms a compile) -- so a first-ever run holds the load screen for the
    // full compile time of forty pipelines.
    //
    // vkCreateGraphicsPipelines is thread-safe and the VkPipelineCache passed to
    // it is internally synchronized (it is not listed as externally synchronized
    // in the spec's threading rules), and CreatePipeline touches no shared state
    // beyond the immutable shader modules + layout. So the worker only CREATES;
    // the map insert stays on the render thread, which is what Get() reads.
    std::thread                              s_prewarmThread;
    std::mutex                               s_prewarmMx;
    xr_vector<std::pair<Key, VkPipeline>>    s_prewarmOut;   // worker -> render thread
    bool                                     s_prewarmAsyncRun = false;

}

// Move whatever the worker finished into the live map. Render thread only.
// A key the worker and a lazy Get() both built ends up with two handles -- the
// one Get() returned is already recorded somewhere, so the DUPLICATE we drop is
// always the worker's.
void PrewarmDrain()
{
    if (!s_prewarmAsyncRun) return;
    xr_vector<std::pair<Key, VkPipeline>> got;
    {
        std::lock_guard<std::mutex> lk(s_prewarmMx);
        if (s_prewarmOut.empty()) return;
        got.swap(s_prewarmOut);
    }
    for (auto& kv : got) {
        if (kv.second == VK_NULL_HANDLE) continue;
        if (s_Pipelines.find(kv.first) != s_Pipelines.end())
            vkDestroyPipeline(VulkanHW.m_Device, kv.second, nullptr);
        else
            s_Pipelines.emplace(kv.first, kv.second);
    }
}

namespace {
// Device teardown: let the worker finish, take what it built into the live map
// (the loop in Destroy frees it from there), then go quiet.
void PrewarmTeardown()
{
    if (s_prewarmThread.joinable()) s_prewarmThread.join();
    PrewarmDrain();
    s_prewarmAsyncRun = false;
}
}   // anonymous namespace

bool PrewarmPending()
{
    // The precache countdown asks this. With the world variants on a worker there
    // is nothing left for those frames to warm: the terrain trio is created in one
    // batch on the first call, and the worker needs no frames at all.
    if (ps_r_prewarm_async) return false;
    return !s_prewarmBuilt || !s_prewarmWorld.empty() || !s_prewarmTerrainMask.empty();
}

bool PrewarmWeatherVariants(u32 budget)
{
    if (!s_Layout) return false;
    // First call: snapshot the work list — the WET/SNOW/WET|SNOW flips of every world
    // pipeline seen so far, plus the terrain weather masks. Nothing is created yet.
    if (!s_prewarmBuilt) {
        s_prewarmBuilt = true;
        const u8 flips[3] = { WS_WET, WS_SNOW, (u8)(WS_WET | WS_SNOW) };
        for (auto& kv : s_Pipelines)
            for (u8 f : flips) {
                Key k = kv.first;
                k.specMask = (u8)(kv.first.specMask | f);
                if (k.specMask != kv.first.specMask) s_prewarmWorld.push_back(k);
            }
        for (u8 f : flips)
            s_prewarmTerrainMask.push_back((u8)((s_FrameSpecMask | f) & WS_FRAME));
        Msg("[VK PipelineCache] prewarm queued %u world + %u terrain weather variants (%u/frame)",
            (u32)s_prewarmWorld.size(), (u32)s_prewarmTerrainMask.size(), budget);
    }
    // Create up to `budget` this call — spread across frames so cold uber-FS compiles
    // (~300ms each) don't freeze a whole frame.
    // Async path: hand the whole world list to one worker and take the terrain
    // trio here (GetTerrainPipeline reads the frame-global mask, which belongs to
    // this thread). Returns false -- the caller has nothing left to spread.
    if (ps_r_prewarm_async) {
        if (!s_prewarmAsyncRun && !s_prewarmWorld.empty()) {
            // The 1/frame path skipped a key that a lazy Get() had built in the
            // meantime; the worker cannot look at the map, so filter HERE instead
            // -- otherwise it compiles ~22 pipelines the drain then throws away,
            // and those compiles are what slows the load screen down.
            xr_vector<Key> work;
            work.reserve(s_prewarmWorld.size());
            std::unordered_set<Key> seen;   // two source keys can flip to the same variant
            for (const Key& k : s_prewarmWorld)
                if (s_Pipelines.find(k) == s_Pipelines.end() && seen.insert(k).second) work.push_back(k);
            s_prewarmWorld.clear();
            Msg("[VK PipelineCache] prewarm async: %u of the queued variants still missing", (u32)work.size());
            s_prewarmAsyncRun = true;
            s_prewarmThread = std::thread([work = std::move(work)]() {
                for (const Key& k : work) {
                    VkPipeline p = CreatePipeline(k);
                    if (p == VK_NULL_HANDLE) continue;
                    std::lock_guard<std::mutex> lk(s_prewarmMx);
                    s_prewarmOut.emplace_back(k, p);
                }
            });
        }
        while (!s_prewarmTerrainMask.empty()) {
            const u8 m = s_prewarmTerrainMask.back(); s_prewarmTerrainMask.pop_back();
            const u8 saved = s_FrameSpecMask;   // GetTerrainPipeline reads the frame mask
            s_FrameSpecMask = m; GetTerrainPipeline(); s_FrameSpecMask = saved;
        }
        return false;
    }

    u32 done = 0;
    while (done < budget && !s_prewarmWorld.empty()) {
        Key k = s_prewarmWorld.back(); s_prewarmWorld.pop_back();
        if (s_Pipelines.find(k) == s_Pipelines.end()) {
            s_Pipelines.emplace(k, CreatePipeline(k));
            ++done;
        }
    }
    while (done < budget && !s_prewarmTerrainMask.empty()) {
        const u8 m = s_prewarmTerrainMask.back(); s_prewarmTerrainMask.pop_back();
        const u8 saved = s_FrameSpecMask;   // GetTerrainPipeline reads the frame mask
        s_FrameSpecMask = m; GetTerrainPipeline(); s_FrameSpecMask = saved;
        ++done;
    }
    return !s_prewarmWorld.empty() || !s_prewarmTerrainMask.empty();
}

}}  // namespace VK::PipelineCache
