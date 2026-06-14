// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

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

    // Heightmap tessellation (R4 TESS_HM): set 0, binding 3 = `<bump>#.dds`
    // whose ALPHA is the displacement height, sampled by the tessellation
    // evaluation shader. `tessellated` is the opt-in: true only when the
    // diffuse .thm declares a bump association AND the # texture loads AND
    // the material is plain opaque (no aref/wmark — those must match the
    // flat depth prepass exactly). Others bind a 1×1 zero-alpha fallback.
    bool            tessellated  = false;

    VkDescriptorSet set      = VK_NULL_HANDLE;
    float           alphaRef = -1.0f;

    // --- Terrain splatting (R4 CBlender_BmmD) ---
    // When the diffuse name starts with "terrain\", this material also gets an
    // 11-binding terrain set {base, mask, dt_r..dt_a, lmap, dn_r..dn_a} and is
    // rendered with the terrain pipeline. dn_* are the <detail>_bump tangent
    // normal maps (R4 CBlender_BmmD). `terrainSet` is VK_NULL_HANDLE for
    // non-terrain materials, which keep the plain 3-binding `set` above.
    bool            isTerrain  = false;
    VkDescriptorSet terrainSet = VK_NULL_HANDLE;

    // Baked level decal (newspapers / dirt overlays — level shader effects\
    // wallmark*): geometry coplanar with the surface beneath. Rendered last,
    // alpha-blended, no depth write, negative depth bias (else z-fight).
    bool            isWmark    = false;
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
WorldMaterial* GetOrCreate(const char* diffuse_name, const char* lmap_name, float alphaRef, bool wmark = false);

VkDescriptorSetLayout GetSetLayout();
VkDescriptorSetLayout GetTerrainSetLayout();   // 7-binding terrain splat set
VkSampler             GetSampler();
WorldMaterial*        GetDefault();

}  // namespace WorldMaterialCache
}  // namespace VK
