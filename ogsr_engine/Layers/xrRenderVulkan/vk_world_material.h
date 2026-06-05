// xrRenderVulkan - World-pass material cache.
//
// Phase 5: maps a diffuse-texture name to a descriptor set the world pass
// can bind. Deliberately small (one combined image sampler per material) and
// independent of the parked CMaterial monolith infrastructure — that one
// pulls deferred-renderer state we don't want yet. When deferred lands this
// cache merges into it.

#pragma once
#include "vk_core.h"

namespace VK {

class CVulkanTexture;

struct WorldMaterial
{
    // Base diffuse (set 0, binding 0).
    CVulkanTexture* tex      = nullptr;
    VkImageView     view     = VK_NULL_HANDLE;
    VkSampler       sampler  = VK_NULL_HANDLE;

    // R4-style detail texture (set 0, binding 1). Always non-null after
    // GetOrCreate — materials with no .thm-declared detail point at the
    // shared 1×1 grey fallback. detailScale = 0 disables UV scaling, the
    // grey fallback's 0.5 sample makes `2 * base * 0.5 = base` a no-op.
    VkImageView     view_detail  = VK_NULL_HANDLE;
    float           detailScale  = 0.0f;

    // Lightmap (set 0, binding 2). Bound only on lmap-style materials
    // (tcOffset==24 with a 3rd texture slot named "lmap*" in the level
    // shader). Other materials get a 1×1 white fallback so the descriptor
    // is always valid; the vert-lit fragment shader simply doesn't read it.
    VkImageView     view_lmap    = VK_NULL_HANDLE;

    VkDescriptorSet set      = VK_NULL_HANDLE;
    float           alphaRef = -1.0f;
};

namespace WorldMaterialCache {

bool Init();
void Destroy();

// Lazy: loads `<diffuse>.dds` (and `<lmap_name>.dds` if non-null) from
// $game_textures$/$level$ on first call. Returns the cache's default
// (1×1 white) on failure or empty diffuse. `lmap_name` may be null/empty
// for non-lightmapped materials → binding 2 falls back to 1×1 white.
// Cache keys both names together so the same diffuse can pair with
// different lmaps.
WorldMaterial* GetOrCreate(const char* diffuse_name, const char* lmap_name, float alphaRef);

VkDescriptorSetLayout GetSetLayout();
VkSampler             GetSampler();
WorldMaterial*        GetDefault();

}  // namespace WorldMaterialCache
}  // namespace VK
