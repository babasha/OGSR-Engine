// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

#include "stdafx.h"
#include "vk_world_material.h"
#include "vk_texture.h"
#include "HW_Vulkan.h"

#include "../xrRender/ETextureParams.h"   // STextureParams::Load + flDiffuseDetail flag

#include <unordered_map>
#include <string>

namespace VK { namespace WorldMaterialCache {

namespace {
    VkDescriptorSetLayout                              s_SetLayout    = VK_NULL_HANDLE;
    VkDescriptorPool                                   s_Pool         = VK_NULL_HANDLE;
    VkSampler                                          s_Sampler      = VK_NULL_HANDLE;
    std::unordered_map<std::string, WorldMaterial*>    s_Cache;
    WorldMaterial*                                     s_Default      = nullptr;

    // Detail-texture cache (shared by reference across materials).
    std::unordered_map<std::string, CVulkanTexture*>   s_DetailTexCache;
    CVulkanTexture*                                    s_GreyDetail   = nullptr;  // 1×1 0.5-grey fallback

    // Lightmap cache and 1×1 white fallback (vert-lit / non-lightmapped
    // materials still need a valid binding; white preserves albedo).
    std::unordered_map<std::string, CVulkanTexture*>   s_LmapTexCache;
    CVulkanTexture*                                    s_WhiteLmap    = nullptr;

    // Bump-error textures (`<bump>#.dds`, alpha = height) for tessellation
    // displacement, shared by reference. Fallback is 1×1 with ZERO alpha —
    // a TES sampling it produces zero displacement.
    std::unordered_map<std::string, CVulkanTexture*>   s_BumpTexCache;
    CVulkanTexture*                                    s_FlatBump     = nullptr;

    // --- Terrain splatting resources (R4 CBlender_BmmD) ---
    // Separate 7-binding set {base, mask, dt_r, dt_g, dt_b, dt_a, lmap} +
    // its own pool, plus a shared white 1×1 mask fallback (normalized → even
    // blend) and the 4 default channel-detail textures (grass/asphalt/earth/
    // gravel). Detail/mask textures are shared by reference across materials.
    VkDescriptorSetLayout                              s_TerrainSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool                                   s_TerrainPool      = VK_NULL_HANDLE;
    CVulkanTexture*                                    s_WhiteMask        = nullptr;  // 1×1 white
    CVulkanTexture*                                    s_TerrainDetail[4] = {};       // R/G/B/A diffuse details
    CVulkanTexture*                                    s_TerrainNormal[4] = {};       // R/G/B/A <detail>_bump normal maps
    CVulkanTexture*                                    s_FlatNormal       = nullptr;  // 1×1 (0,0,1) tangent normal fallback
    std::unordered_map<std::string, CVulkanTexture*>   s_TerrainDetCache;             // by name (diffuse + normal + mask)

    // Resolved file path: $game_textures$\\<name>.dds
    bool ResolveTexturePath(const char* name, string_path& out)
    {
        if (!name || !name[0]) return false;
        string_path leaf;
        xr_sprintf(leaf, "%s.dds", name);
        FS.update_path(out, "$game_textures$", leaf);
        return FS.exist(out);
    }

    VkDescriptorSet AllocateSet()
    {
        VkDescriptorSetAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = s_Pool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &s_SetLayout;

        VkDescriptorSet set = VK_NULL_HANDLE;
        VkResult r = vkAllocateDescriptorSets(VulkanHW.m_Device, &ai, &set);
        if (r != VK_SUCCESS) {
            Msg("![VK WorldMaterial] vkAllocateDescriptorSets failed (%d)", r);
            return VK_NULL_HANDLE;
        }
        return set;
    }

    void WriteSet(VkDescriptorSet set, VkImageView baseView,
                  VkImageView detailView, VkImageView lmapView, VkImageView bumpxView)
    {
        VkDescriptorImageInfo ii[4]{};
        VkImageView views[4] = { baseView, detailView, lmapView, bumpxView };
        for (int i = 0; i < 4; ++i) {
            ii[i].sampler     = s_Sampler;
            ii[i].imageView   = views[i];
            ii[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }

        VkWriteDescriptorSet w[4]{};
        for (int i = 0; i < 4; ++i) {
            w[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet          = set;
            w[i].dstBinding      = (u32)i;
            w[i].descriptorCount = 1;
            w[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[i].pImageInfo      = &ii[i];
        }
        vkUpdateDescriptorSets(VulkanHW.m_Device, 4, w, 0, nullptr);
    }

    CVulkanTexture* GetOrLoadLmapTex(const char* lmap_name)
    {
        if (!lmap_name || !lmap_name[0]) return s_WhiteLmap;

        std::string key(lmap_name);
        auto it = s_LmapTexCache.find(key);
        if (it != s_LmapTexCache.end()) return it->second;

        // Lightmaps live alongside the level data ($level$/<lmap>.dds).
        string_path leaf;
        xr_sprintf(leaf, "%s.dds", lmap_name);
        string_path full;
        FS.update_path(full, "$level$", leaf);
        if (!FS.exist(full)) {
            FS.update_path(full, "$game_textures$", leaf);
            if (!FS.exist(full)) {
                s_LmapTexCache.emplace(std::move(key), s_WhiteLmap);
                return s_WhiteLmap;
            }
        }

        auto* tex = xr_new<CVulkanTexture>();
        if (!tex->LoadDDS(full, /*applyBCSwizzle*/ false)) {
            xr_delete(tex);
            s_LmapTexCache.emplace(std::move(key), s_WhiteLmap);
            return s_WhiteLmap;
        }
        s_LmapTexCache.emplace(std::move(key), tex);
        return tex;
    }

    // Everything we pull out of a diffuse's `<base>.thm`: the R4 detail
    // descriptor (texture + UV scale) and the bump association (tessellation
    // height comes from `<bump>#.dds`).
    struct THMInfo
    {
        std::string detail_name;
        float       detail_scale = 0.0f;
        bool        has_detail   = false;
        std::string bump_name;
        bool        has_bump     = false;
    };

    // Looks up `<base>.thm` in $game_textures$ (then $level$) and fills `out`.
    // Returns false when the .thm is missing/unreadable — caller falls back
    // to the grey detail / flat bump samplers.
    bool LookupTHM(const char* base_name, THMInfo& out)
    {
        if (!base_name || !base_name[0]) return false;

        string_path file_nm;
        xr_sprintf(file_nm, "%s.thm", base_name);

        string_path full;
        FS.update_path(full, "$game_textures$", file_nm);
        IReader* F = FS.r_open(full);
        if (!F) {
            FS.update_path(full, "$level$", file_nm);
            F = FS.r_open(full);
            if (!F) return false;
        }

        // .thm structure: outer chunk THM_CHUNK_TYPE (0x0813) carries the
        // texture-class id (Image / Terrain / NormalMap). STextureParams::Load
        // walks the TEXTUREPARAM / DETAIL_EXT / MATERIAL / BUMP / etc. chunks.
        if (!F->find_chunk_thm(THM_CHUNK_TYPE, base_name)) {
            FS.r_close(F);
            return false;
        }
        F->r_u32();  // ETType — read for parity, value not needed

        STextureParams tp{};
        tp.Load(*F, base_name);
        FS.r_close(F);

        if (tp.detail_name.size() > 0 &&
            tp.flags.is_any(STextureParams::flDiffuseDetail | STextureParams::flBumpDetail)) {
            out.detail_name  = tp.detail_name.c_str();
            out.detail_scale = tp.detail_scale;
            out.has_detail   = true;
        }

        if (tp.bump_name.size() > 0 &&
            (tp.bump_mode == STextureParams::tbmUse || tp.bump_mode == STextureParams::tbmUseParallax)) {
            out.bump_name = tp.bump_name.c_str();
            out.has_bump  = true;
        }

        return true;
    }

    CVulkanTexture* GetOrLoadDetailTex(const std::string& detail_name)
    {
        if (detail_name.empty()) return s_GreyDetail;

        auto it = s_DetailTexCache.find(detail_name);
        if (it != s_DetailTexCache.end()) return it->second;

        // Detail textures live next to base diffuses under $game_textures$.
        string_path leaf;
        xr_sprintf(leaf, "%s.dds", detail_name.c_str());
        string_path full;
        FS.update_path(full, "$game_textures$", leaf);
        if (!FS.exist(full)) {
            // Cache the miss so we don't keep stat'ing the FS.
            s_DetailTexCache.emplace(detail_name, s_GreyDetail);
            return s_GreyDetail;
        }

        auto* tex = xr_new<CVulkanTexture>();
        if (!tex->LoadDDS(full, /*applyBCSwizzle*/ false)) {
            xr_delete(tex);
            s_DetailTexCache.emplace(detail_name, s_GreyDetail);
            return s_GreyDetail;
        }
        s_DetailTexCache.emplace(detail_name, tex);
        return tex;
    }

    // Generic cached loader for a named .dds in $game_textures$ (then $level$).
    // Returns `fallback` (never null) on miss so descriptors stay valid.
    CVulkanTexture* GetOrLoadGameTex(std::unordered_map<std::string, CVulkanTexture*>& cache,
                                     const char* name, CVulkanTexture* fallback)
    {
        if (!name || !name[0]) return fallback;
        std::string key(name);
        auto it = cache.find(key);
        if (it != cache.end()) return it->second;

        string_path leaf, full;
        xr_sprintf(leaf, "%s.dds", name);
        FS.update_path(full, "$game_textures$", leaf);
        if (!FS.exist(full)) {
            FS.update_path(full, "$level$", leaf);
            if (!FS.exist(full)) { cache.emplace(std::move(key), fallback); return fallback; }
        }
        auto* tex = xr_new<CVulkanTexture>();
        if (!tex->LoadDDS(full, /*applyBCSwizzle*/ false)) {
            xr_delete(tex);
            cache.emplace(std::move(key), fallback);
            return fallback;
        }
        cache.emplace(std::move(key), tex);
        return tex;
    }

    // Write the 11-binding terrain set: base, mask, dt_r..dt_a, lmap, dn_r..dn_a.
    void WriteTerrainSet(VkDescriptorSet set, const VkImageView v[11])
    {
        VkDescriptorImageInfo ii[11]{};
        VkWriteDescriptorSet  w[11]{};
        for (int i = 0; i < 11; ++i) {
            ii[i].sampler     = s_Sampler;
            ii[i].imageView   = v[i];
            ii[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            w[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet          = set;
            w[i].dstBinding      = (u32)i;
            w[i].descriptorCount = 1;
            w[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[i].pImageInfo      = &ii[i];
        }
        vkUpdateDescriptorSets(VulkanHW.m_Device, 11, w, 0, nullptr);
    }

    WorldMaterial* CreateDefaultWhite()
    {
        // 1×1 white as the universal fallback for missing textures.
        const u8 white[4] = { 255, 255, 255, 255 };
        auto* tex = xr_new<CVulkanTexture>();
        tex->CreateFromData(white, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, 4);

        auto* m = xr_new<WorldMaterial>();
        m->tex          = tex;
        m->view         = tex->GetView();
        m->sampler      = s_Sampler;
        m->view_detail  = s_GreyDetail->GetView();   // 0.5 detail → no-op
        m->detailScale  = 0.0f;
        m->view_lmap    = s_WhiteLmap->GetView();    // 1.0 lmap   → no-op
        m->alphaRef     = -1.0f;
        m->set          = AllocateSet();
        if (m->set != VK_NULL_HANDLE)
            WriteSet(m->set, m->view, m->view_detail, m->view_lmap, s_FlatBump->GetView());
        return m;
    }
}

VkDescriptorSetLayout GetSetLayout()        { return s_SetLayout;        }
VkDescriptorSetLayout GetTerrainSetLayout() { return s_TerrainSetLayout; }
VkSampler             GetSampler()          { return s_Sampler;          }
WorldMaterial*        GetDefault()          { return s_Default;          }

bool Init()
{
    if (s_SetLayout) return true;

    // Set 0: { binding 0 = diffuse, binding 1 = detail, binding 2 = lmap,
    // binding 3 = bump# height }, all combined image samplers. Bindings 0-2
    // are fragment-only; binding 3 is sampled by the tessellation evaluation
    // shader (displacement height in its alpha). Four bindings per set
    // requires the pool descriptorCount to be 4× maxSets.
    VkDescriptorSetLayoutBinding b[4]{};
    for (int i = 0; i < 4; ++i) {
        b[i].binding         = (u32)i;
        b[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[i].descriptorCount = 1;
        b[i].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    if (VulkanHW.m_bTessellationSupported)
        b[3].stageFlags |= VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;

    VkDescriptorSetLayoutCreateInfo lci{};
    lci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lci.bindingCount = 4;
    lci.pBindings    = b;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &s_SetLayout) != VK_SUCCESS) {
        Msg("![VK WorldMaterial] CreateDescriptorSetLayout failed");
        return false;
    }

    // Pool sized for ~level worth of unique level shaders (549 in test logs).
    // Round up to 1024 to absorb spawned visual textures too.
    constexpr u32 kMaxSets = 1024;
    VkDescriptorPoolSize ps{};
    ps.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    ps.descriptorCount = kMaxSets * 4;   // 4 image samplers per set

    VkDescriptorPoolCreateInfo pci{};
    pci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets       = kMaxSets;
    pci.poolSizeCount = 1;
    pci.pPoolSizes    = &ps;
    if (vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &s_Pool) != VK_SUCCESS) {
        Msg("![VK WorldMaterial] CreateDescriptorPool failed");
        return false;
    }

    // Shared sampler. samplerAnisotropy is enabled at device creation
    // (HW_Vulkan.cpp), so anisotropic filtering is safe to request here.
    // Without it, terrain at glancing angles trilinear-collapses to coarse
    // mips → smudgy "gouache paint" look on the ground.
    VkSamplerCreateInfo si{};
    si.sType            = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter        = VK_FILTER_LINEAR;
    si.minFilter        = VK_FILTER_LINEAR;
    si.mipmapMode       = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    si.addressModeU     = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.addressModeV     = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.addressModeW     = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.minLod           = 0.0f;
    si.maxLod           = VK_LOD_CLAMP_NONE;
    si.anisotropyEnable = VK_TRUE;
    si.maxAnisotropy    = 16.0f;   // device limit is queried but every modern GPU clamps to ≥16
    if (vkCreateSampler(VulkanHW.m_Device, &si, nullptr, &s_Sampler) != VK_SUCCESS) {
        Msg("![VK WorldMaterial] CreateSampler failed");
        return false;
    }

    // Grey 0.5 detail fallback. With the formula `2 * base * detail`, a
    // material with no .thm-declared detail samples 0.5 → factor 1.0 → no-op.
    {
        const u8 grey[4] = { 128, 128, 128, 255 };
        s_GreyDetail = xr_new<CVulkanTexture>();
        s_GreyDetail->CreateFromData(grey, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, 4);
    }

    // White lmap fallback for vert-lit / non-lightmapped materials. Frag
    // shader does `albedo *= lm`, so 1.0 is a no-op. The vert-lit shader
    // variant doesn't sample binding 2 anyway — the white texture only
    // satisfies the descriptor's validity requirement.
    {
        const u8 white[4] = { 255, 255, 255, 255 };
        s_WhiteLmap = xr_new<CVulkanTexture>();
        s_WhiteLmap->CreateFromData(white, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, 4);
    }

    // Flat bump# fallback for materials with no `<bump>#` height. ALPHA = 255
    // (height = 1 = top surface): POM depth = 1-h = 0 → the parallax march
    // exits immediately → NO parallax on these materials (auto-gate). Tess is
    // unaffected — only real-bump materials enter the tess pipeline, so a TES
    // never samples this fallback (and the planned high-pass makes a constant
    // height a no-op there regardless).
    {
        const u8 flat[4] = { 128, 128, 128, 255 };
        s_FlatBump = xr_new<CVulkanTexture>();
        s_FlatBump->CreateFromData(flat, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, 4);
    }

    s_Default = CreateDefaultWhite();

    // ----- Terrain splatting set layout + pool + default channel details -----
    {
        // 11 combined image samplers, fragment-only: base, mask, dt_r..dt_a,
        // lmap, dn_r..dn_a (the 4 <detail>_bump tangent normal maps).
        VkDescriptorSetLayoutBinding tb[11]{};
        for (int i = 0; i < 11; ++i) {
            tb[i].binding         = (u32)i;
            tb[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            tb[i].descriptorCount = 1;
            tb[i].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        VkDescriptorSetLayoutCreateInfo tlci{};
        tlci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        tlci.bindingCount = 11;
        tlci.pBindings    = tb;
        vkCreateDescriptorSetLayout(VulkanHW.m_Device, &tlci, nullptr, &s_TerrainSetLayout);

        // Terrain materials are few (a handful of ground textures per level);
        // 256 sets is generous.
        constexpr u32 kMaxTerrain = 256;
        VkDescriptorPoolSize tps{};
        tps.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        tps.descriptorCount = kMaxTerrain * 11;
        VkDescriptorPoolCreateInfo tpci{};
        tpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        tpci.maxSets       = kMaxTerrain;
        tpci.poolSizeCount = 1;
        tpci.pPoolSizes    = &tps;
        vkCreateDescriptorPool(VulkanHW.m_Device, &tpci, nullptr, &s_TerrainPool);

        // 1×1 white mask fallback (normalized → even blend; never black).
        const u8 white[4] = { 255, 255, 255, 255 };
        s_WhiteMask = xr_new<CVulkanTexture>();
        s_WhiteMask->CreateFromData(white, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, 4);

        // Flat tangent-normal fallback (0,0,1). R4 decodes a detail-normal as
        // `n = tex.wzy*2-1` (gloss in R; tangent normal packed in A,B,G). For a
        // no-op normal we need A=0.5, B=0.5, G=1.0 → RGBA {0,255,128,128}.
        const u8 flatN[4] = { 0, 255, 128, 128 };
        s_FlatNormal = xr_new<CVulkanTexture>();
        s_FlatNormal->CreateFromData(flatN, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, 4);

        // Default channel-detail textures (CBlender_BmmD defaults). Per-shader
        // overrides live in shaders.xr; these cover the common case.
        static const char* kDet[4] = {
            "detail\\detail_grnd_grass",   // R
            "detail\\detail_grnd_asphalt", // G
            "detail\\detail_grnd_earth",   // B
            "detail\\detail_grnd_yantar",  // A
        };
        for (int i = 0; i < 4; ++i) {
            s_TerrainDetail[i] = GetOrLoadGameTex(s_TerrainDetCache, kDet[i], s_GreyDetail);
            // CBlender_BmmD names the detail-normal as "<detail>_bump". Missing
            // ones fall back to the flat normal (no perturbation) — graceful on
            // modpacks that ship diffuse details without bumps.
            std::string bn = std::string(kDet[i]) + "_bump";
            s_TerrainNormal[i] = GetOrLoadGameTex(s_TerrainDetCache, bn.c_str(), s_FlatNormal);
            Msg("[VK Terrain] detail-normal '%s' -> %s", bn.c_str(),
                (s_TerrainNormal[i] != s_FlatNormal) ? "loaded" : "MISSING (flat)");
        }
    }

    Msg("[VK WorldMaterial] Init OK (pool=%u sets base+detail+lmap+bump#; terrain pool=256 x11 w/ detail-normals; anisotropic 16x)", kMaxSets);
    return true;
}

void Destroy()
{
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;

    for (auto& kv : s_Cache) {
        if (!kv.second) continue;
        if (kv.second == s_Default) continue;  // s_Cache may alias s_Default for failed lookups
        if (kv.second->tex) { kv.second->tex->Destroy(); xr_delete(kv.second->tex); }
        xr_delete(kv.second);
    }
    s_Cache.clear();

    // Detail textures are shared (one entry per unique detail name); free once.
    for (auto& kv : s_DetailTexCache) {
        if (kv.second && kv.second != s_GreyDetail) {
            kv.second->Destroy();
            xr_delete(kv.second);
        }
    }
    s_DetailTexCache.clear();

    if (s_GreyDetail) {
        s_GreyDetail->Destroy();
        xr_delete(s_GreyDetail);
        s_GreyDetail = nullptr;
    }

    for (auto& kv : s_LmapTexCache) {
        if (kv.second && kv.second != s_WhiteLmap) {
            kv.second->Destroy();
            xr_delete(kv.second);
        }
    }
    s_LmapTexCache.clear();

    if (s_WhiteLmap) {
        s_WhiteLmap->Destroy();
        xr_delete(s_WhiteLmap);
        s_WhiteLmap = nullptr;
    }

    // Bump# height textures (tessellation displacement), shared by reference.
    for (auto& kv : s_BumpTexCache) {
        if (kv.second && kv.second != s_FlatBump) {
            kv.second->Destroy();
            xr_delete(kv.second);
        }
    }
    s_BumpTexCache.clear();

    if (s_FlatBump) {
        s_FlatBump->Destroy();
        xr_delete(s_FlatBump);
        s_FlatBump = nullptr;
    }

    // Terrain splat resources. s_TerrainDetail[] alias entries in
    // s_TerrainDetCache (or the grey/white fallbacks) — free the cache once,
    // skipping shared fallbacks.
    for (auto& kv : s_TerrainDetCache) {
        if (kv.second && kv.second != s_GreyDetail && kv.second != s_WhiteMask
            && kv.second != s_FlatNormal) {
            kv.second->Destroy();
            xr_delete(kv.second);
        }
    }
    s_TerrainDetCache.clear();
    for (int i = 0; i < 4; ++i) { s_TerrainDetail[i] = nullptr; s_TerrainNormal[i] = nullptr; }

    if (s_FlatNormal) {
        s_FlatNormal->Destroy();
        xr_delete(s_FlatNormal);
        s_FlatNormal = nullptr;
    }

    if (s_WhiteMask) {
        s_WhiteMask->Destroy();
        xr_delete(s_WhiteMask);
        s_WhiteMask = nullptr;
    }
    if (s_TerrainPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(VulkanHW.m_Device, s_TerrainPool, nullptr);
        s_TerrainPool = VK_NULL_HANDLE;
    }
    if (s_TerrainSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_TerrainSetLayout, nullptr);
        s_TerrainSetLayout = VK_NULL_HANDLE;
    }

    if (s_Default) {
        if (s_Default->tex) { s_Default->tex->Destroy(); xr_delete(s_Default->tex); }
        xr_delete(s_Default);
        s_Default = nullptr;
    }

    if (s_Pool) {
        // Frees all sets allocated from the pool — including default + cached.
        vkDestroyDescriptorPool(VulkanHW.m_Device, s_Pool, nullptr);
        s_Pool = VK_NULL_HANDLE;
    }
    if (s_Sampler) {
        vkDestroySampler(VulkanHW.m_Device, s_Sampler, nullptr);
        s_Sampler = VK_NULL_HANDLE;
    }
    if (s_SetLayout) {
        vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_SetLayout, nullptr);
        s_SetLayout = VK_NULL_HANDLE;
    }
}

WorldMaterial* GetOrCreate(const char* diffuse_name, const char* lmap_name, float alphaRef, bool wmark)
{
    if (!s_SetLayout || !s_Default) return s_Default;
    if (!diffuse_name || !diffuse_name[0]) return s_Default;

    // Same diffuse can pair with different lmaps — key on both.
    std::string key(diffuse_name);
    key.push_back('|');
    if (lmap_name && lmap_name[0]) key.append(lmap_name);

    auto it = s_Cache.find(key);
    if (it != s_Cache.end()) {
        // Materials are shared by texture pair — if ANY user is a wallmark
        // shader, the whole material renders as a decal (textures are
        // decal-dedicated in practice).
        if (wmark && it->second != s_Default) it->second->isWmark = true;
        return it->second;
    }

    string_path full;
    if (!ResolveTexturePath(diffuse_name, full)) {
        s_Cache.emplace(std::move(key), s_Default);
        return s_Default;
    }

    auto* tex = xr_new<CVulkanTexture>();
    if (!tex->LoadDDS(full, /*applyBCSwizzle*/ false)) {
        xr_delete(tex);
        s_Cache.emplace(std::move(key), s_Default);
        return s_Default;
    }

    // .thm lookup: R4 detail descriptor (texture + scale) and the bump
    // association. Materials with no detail association get the grey fallback
    // so the shader's `2 * base * detail` reduces to `base`.
    THMInfo         thm{};
    LookupTHM(diffuse_name, thm);
    CVulkanTexture* detail_tex = thm.has_detail ? GetOrLoadDetailTex(thm.detail_name) : s_GreyDetail;

    // R4 TESS_HM gate: a bump association whose `<bump>#.dds` (alpha = height)
    // actually loads. shaders.xr TessMethod is NO_TESS across stock content,
    // so the .thm bump is the practical per-material opt-in. Alpha-tested and
    // decal materials stay flat — their coverage must match the depth prepass.
    CVulkanTexture* bumpx_tex = s_FlatBump;
    if (!wmark && alphaRef < 0.0f && thm.has_bump) {
        std::string bumpx_name = thm.bump_name + "#";
        bumpx_tex = GetOrLoadGameTex(s_BumpTexCache, bumpx_name.c_str(), s_FlatBump);
        // Tess diag: report every material that DID / DID NOT pick up a height
        // texture, so "why is this wall flat?" is answerable from the log.
        // (.thm has a bump assoc but the `#` height texture may be missing.)
        Msg("[VK Tess] '%s': bump '%s' -> %s ('%s#')", diffuse_name, thm.bump_name.c_str(),
            (bumpx_tex != s_FlatBump) ? "TESSELLATED (height loaded)" : "NO height tex, stays flat",
            thm.bump_name.c_str());
    }

    // Lightmap from the level shader's 3rd texture slot. Vert-lit / non-
    // lightmapped materials pass null here → white fallback (no-op multiply).
    CVulkanTexture* lmap_tex = GetOrLoadLmapTex(lmap_name);

    auto* m = xr_new<WorldMaterial>();
    m->tex          = tex;
    m->view         = tex->GetView();
    m->sampler      = s_Sampler;
    m->view_detail  = detail_tex ? detail_tex->GetView() : s_GreyDetail->GetView();
    m->detailScale  = (detail_tex && detail_tex != s_GreyDetail) ? thm.detail_scale : 0.0f;
    m->view_lmap    = lmap_tex   ? lmap_tex->GetView()   : s_WhiteLmap->GetView();
    m->alphaRef     = alphaRef;
    m->isWmark      = wmark;
    m->name         = diffuse_name;
    m->tessellated  = (bumpx_tex != s_FlatBump);
    m->set          = AllocateSet();
    if (m->set == VK_NULL_HANDLE) {
        Msg("![VK WorldMaterial] Pool exhausted creating '%s' — falling back to default", diffuse_name);
        tex->Destroy();
        xr_delete(tex);
        xr_delete(m);
        s_Cache.emplace(std::move(key), s_Default);
        return s_Default;
    }
    WriteSet(m->set, m->view, m->view_detail, m->view_lmap, bumpx_tex->GetView());

    // ----- Terrain splatting: diffuse under "terrain\" gets the 7-binding set.
    // Mask = "<diffuse>_mask"; details = the 4 channel defaults; detail UV
    // scale reuses the base .thm detail_scale (e.g. terrain_escape = 144).
    // Falls back gracefully: missing mask → white (even blend), so terrain is
    // never worse than the single-detail path.
    const bool is_terrain = (strstr(diffuse_name, "terrain\\") == diffuse_name ||
                             strstr(diffuse_name, "terrain/")  == diffuse_name);
    if (is_terrain && s_TerrainSetLayout != VK_NULL_HANDLE) {
        VkDescriptorSetAllocateInfo ai{};
        ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool     = s_TerrainPool;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts        = &s_TerrainSetLayout;
        VkDescriptorSet tset = VK_NULL_HANDLE;
        if (vkAllocateDescriptorSets(VulkanHW.m_Device, &ai, &tset) == VK_SUCCESS) {
            string_path mask_name;
            xr_sprintf(mask_name, "%s_mask", diffuse_name);
            CVulkanTexture* mask = GetOrLoadGameTex(s_TerrainDetCache, mask_name, s_WhiteMask);

            const VkImageView v[11] = {
                m->view,
                mask ? mask->GetView() : s_WhiteMask->GetView(),
                s_TerrainDetail[0]->GetView(), s_TerrainDetail[1]->GetView(),
                s_TerrainDetail[2]->GetView(), s_TerrainDetail[3]->GetView(),
                m->view_lmap,
                s_TerrainNormal[0]->GetView(), s_TerrainNormal[1]->GetView(),
                s_TerrainNormal[2]->GetView(), s_TerrainNormal[3]->GetView(),
            };
            WriteTerrainSet(tset, v);
            m->isTerrain  = true;
            m->terrainSet = tset;
            // Terrain still needs a sane detail UV scale even when the base .thm
            // had none (single-detail path left it 0 → detailUV collapses).
            if (m->detailScale <= 0.0f) m->detailScale = thm.detail_scale > 0.0f ? thm.detail_scale : 64.0f;
            Msg("[VK Terrain] '%s' splat set: mask=%s scale=%.0f", diffuse_name,
                (mask && mask != s_WhiteMask) ? "REAL" : "white", m->detailScale);
        }
    }

    s_Cache.emplace(std::move(key), m);
    return m;
}

}}  // namespace VK::WorldMaterialCache
