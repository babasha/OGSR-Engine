// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan - Pipeline cache keyed by vertex layout + shader program.
//
// Phase-2 lazy creation of forward graphics pipelines so we stop hardcoding
// stride 32 / tcOffset 24 in WorldPipeline. One VkPipelineLayout is shared
// across the whole world program (push: mat4 mvp + vec2 uvScale, no
// descriptors yet); per-Key VkPipelines are created on first Get() and
// reused thereafter.

#pragma once
#include "vk_core.h"
#include "vk_authorship.h"   // build-identity tokens folded into the key hash below

namespace VK {
namespace PipelineCache {

struct Key
{
    u32             stride       = 0;       // vertex stride in bytes (32 / 36 / 40 / 44)
    u32             tcOffset     = 24;      // TEXCOORD0 byte offset within the vertex
    VkShaderModule  vs           = VK_NULL_HANDLE;
    VkShaderModule  fs           = VK_NULL_HANDLE;
    bool            depthTest    = false;   // wires up in phase 4 (depth target)
    // Baked level decal variant: alpha blend, depth test but NO write, negative
    // depth bias (geometry is coplanar with the surface beneath — newspapers).
    bool            wmark        = false;
    // Heightmap-tessellation variant (R4 TESS_HM): VS+TCS+TES+FS over patch
    // lists; the TCS/TES pair is picked by tcOffset (lmap/vlit). Small negative
    // depth bias so the re-rasterized flat margins win ties vs the (flat)
    // depth prepass.
    bool            tess         = false;
    // Emissive-additive variant (level `effects\glow` billboards, `selflight`
    // model parts — lamp halos/projector faces): blend (SRC_ALPHA, ONE), depth
    // test but NO write, no bias. The FS goes unlit via the aref == -3 marker.
    bool            emis         = false;

    bool operator==(const Key& o) const noexcept
    {
        return stride == o.stride && tcOffset == o.tcOffset
            && vs == o.vs && fs == o.fs && depthTest == o.depthTest
            && wmark == o.wmark && tess == o.tess && emis == o.emis;
    }
};

// Initialised once after device creation; tears down on shutdown. Loads the
// world shader pair and creates the shared layout.
bool Init();
void Destroy();

// Lazy-creates the VkPipeline for `key` if absent, returns the cached handle.
// Returns VK_NULL_HANDLE on failure (logged).
VkPipeline Get(const Key& key);

// Shared by all draws — same push range, no descriptor sets.
VkPipelineLayout GetLayout();

// The single, engine-wide VkPipelineCache object. Created (and seeded from the
// on-disk blob) in Init(), serialized back to disk + destroyed in Destroy().
// ALL pipeline producers (this cache, Pass_Sky, Pass_Skinned, ...) must pass
// this into vkCreateGraphicsPipelines so the driver dedups + warm-starts across
// runs. Returns VK_NULL_HANDLE before Init() / after Destroy() — passing that to
// vkCreateGraphicsPipelines is legal (just means "no cache"), so callers needn't
// null-check.
VkPipelineCache GetCacheObject();

// Pre-loaded module accessors so callers (Pass_World, RenderQueue::Flush)
// can build keys without touching g_ShaderManager directly. Two variants:
//   - "Lmap"  for tcOffset==24 layout: reads TC0@24 + TC1@28 (lightmap UV),
//     samples binding 2 in FS for baked sun/hemi.
//   - "Vlit"  for tcOffset==28 layout: reads COLOR@24 (D3DCOLOR pre-baked
//     RGB lighting, .a sun mask) + TC0@28; multiplies albedo by COLOR.bgr.
VkShaderModule WorldLmapVS();
VkShaderModule WorldLmapFS();
VkShaderModule WorldVlitVS();
VkShaderModule WorldVlitFS();

// World heightmap tessellation (R4 TESS_HM): true when the device feature is
// enabled AND all four world TCS/TES modules loaded. Callers must not build
// keys with tess=true while this is false.
bool TessAvailable();

// Stage mask of the shared world/terrain push-constant range. The exact same
// mask MUST be passed to every vkCmdPushConstants targeting those layouts
// (VUID 01795: flags must match the range exactly). Includes the tessellation
// stages whenever the device supports them.
VkShaderStageFlags GetPushStages();

// Terrain splatting (R4 CBlender_BmmD). Single variant (stride 32, tcOffset 24,
// depth on) with its own pipeline layout (set 0 = WorldMaterialCache's 7-binding
// terrain set). Lazily built on first GetTerrainPipeline() — needs swapchain
// formats like the world pipelines. Both return VK_NULL_HANDLE before Init().
VkPipelineLayout GetTerrainLayout();
VkPipeline       GetTerrainPipeline();
// Depth-prepass variant of the terrain pipeline: same world_terrain.vert (so snow
// displacement matches color exactly -> no z-fight), vertex-only, terrain layout.
// Bind the EnvLight set at set 1 (it reads sf_params.w). VK_NULL_HANDLE before Init().
VkPipeline       GetTerrainDepthPipeline();

// Sun shadow caster (STEP 2): depth-only pipelines (shadow_depth.vert, no FS),
// keyed by vertex stride (position is at offset 0 for every level layout). The
// shared layout has a single push range { mat4 lightMVP } (VERTEX). Used by
// RenderQueue::FlushDepth from Pass_SunShadow. VK_NULL_HANDLE if the shader is
// missing. The pipelines render into the shadow map's D32 format.
VkPipelineLayout GetDepthLayout();
VkPipeline       GetDepthPipeline(u32 stride);

// Alpha-tested shadow caster variant (bushes, grates): VS passes the base UV,
// FS samples the material diffuse (set 0 = WorldMaterial set) and discards
// below alphaRef — depth keeps the foliage silhouette instead of solid quads.
// Push: { mat4 lightMVP; vec2 uvScale; float alphaRef } (VERTEX|FRAGMENT).
VkPipelineLayout GetDepthATLayout();
VkPipeline       GetDepthATPipeline(u32 stride, u32 tcOffset);

}}  // namespace VK::PipelineCache

namespace std {
template <>
struct hash<VK::PipelineCache::Key>
{
    size_t operator()(const VK::PipelineCache::Key& k) const noexcept
    {
        // Seeded with the build-identity tokens (ogsr::sig) so the key domain is
        // bound to this build. The salts are deterministic compile-time immediates;
        // any fixed values hash correctly — these just happen to be load-bearing.
        size_t h = std::hash<u32>{}(k.stride ^ ogsr::sig::blumenau) ^ ogsr::sig::saratov;
        h ^= std::hash<u32>{}(k.tcOffset ^ ogsr::sig::catara) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<void*>{}((void*)k.vs)          + ogsr::sig::zefir + (h << 6) + (h >> 2);
        h ^= std::hash<void*>{}((void*)k.fs)          + ogsr::sig::maria + (h << 6) + (h >> 2);
        h ^= std::hash<bool>{}(k.depthTest)           + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<bool>{}(k.wmark)               + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<bool>{}(k.tess)                + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<bool>{}(k.emis)                + 0x517cc1b7 + (h << 6) + (h >> 2);
        return h;
    }
};
}  // namespace std
