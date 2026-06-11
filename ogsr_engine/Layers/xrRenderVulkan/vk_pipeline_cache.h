// xrRenderVulkan - Pipeline cache keyed by vertex layout + shader program.
//
// Phase-2 lazy creation of forward graphics pipelines so we stop hardcoding
// stride 32 / tcOffset 24 in WorldPipeline. One VkPipelineLayout is shared
// across the whole world program (push: mat4 mvp + vec2 uvScale, no
// descriptors yet); per-Key VkPipelines are created on first Get() and
// reused thereafter.

#pragma once
#include "vk_core.h"

namespace VK {
namespace PipelineCache {

struct Key
{
    u32             stride       = 0;       // vertex stride in bytes (32 / 36 / 40 / 44)
    u32             tcOffset     = 24;      // TEXCOORD0 byte offset within the vertex
    VkShaderModule  vs           = VK_NULL_HANDLE;
    VkShaderModule  fs           = VK_NULL_HANDLE;
    bool            depthTest    = false;   // wires up in phase 4 (depth target)

    bool operator==(const Key& o) const noexcept
    {
        return stride == o.stride && tcOffset == o.tcOffset
            && vs == o.vs && fs == o.fs && depthTest == o.depthTest;
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

// Terrain splatting (R4 CBlender_BmmD). Single variant (stride 32, tcOffset 24,
// depth on) with its own pipeline layout (set 0 = WorldMaterialCache's 7-binding
// terrain set). Lazily built on first GetTerrainPipeline() — needs swapchain
// formats like the world pipelines. Both return VK_NULL_HANDLE before Init().
VkPipelineLayout GetTerrainLayout();
VkPipeline       GetTerrainPipeline();

}}  // namespace VK::PipelineCache

namespace std {
template <>
struct hash<VK::PipelineCache::Key>
{
    size_t operator()(const VK::PipelineCache::Key& k) const noexcept
    {
        size_t h = std::hash<u32>{}(k.stride);
        h ^= std::hash<u32>{}(k.tcOffset)             + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<void*>{}((void*)k.vs)          + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<void*>{}((void*)k.fs)          + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<bool>{}(k.depthTest)           + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};
}  // namespace std
