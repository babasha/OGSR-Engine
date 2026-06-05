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
                  VkImageView detailView, VkImageView lmapView)
    {
        VkDescriptorImageInfo ii[3]{};
        VkImageView views[3] = { baseView, detailView, lmapView };
        for (int i = 0; i < 3; ++i) {
            ii[i].sampler     = s_Sampler;
            ii[i].imageView   = views[i];
            ii[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }

        VkWriteDescriptorSet w[3]{};
        for (int i = 0; i < 3; ++i) {
            w[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet          = set;
            w[i].dstBinding      = (u32)i;
            w[i].descriptorCount = 1;
            w[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[i].pImageInfo      = &ii[i];
        }
        vkUpdateDescriptorSets(VulkanHW.m_Device, 3, w, 0, nullptr);
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

    // Looks up `<base>.thm` in $game_textures$ (then $level$) and reads the
    // R4 detail-texture descriptor. Returns false when the material has no
    // detail association — caller must fall back to the grey 1×1 sampler.
    bool LookupDetailFromTHM(const char* base_name, std::string& out_detail_name, float& out_scale)
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

        const bool has_diffuse_detail =
            tp.detail_name.size() > 0 &&
            tp.flags.is_any(STextureParams::flDiffuseDetail | STextureParams::flBumpDetail);
        if (!has_diffuse_detail) return false;

        out_detail_name = tp.detail_name.c_str();
        out_scale       = tp.detail_scale;
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
        if (m->set != VK_NULL_HANDLE) WriteSet(m->set, m->view, m->view_detail, m->view_lmap);
        return m;
    }
}

VkDescriptorSetLayout GetSetLayout() { return s_SetLayout; }
VkSampler             GetSampler()   { return s_Sampler;   }
WorldMaterial*        GetDefault()   { return s_Default;   }

bool Init()
{
    if (s_SetLayout) return true;

    // Set 0: { binding 0 = diffuse, binding 1 = detail, binding 2 = lmap },
    // all combined image samplers, fragment-only. Three bindings per set
    // requires the pool descriptorCount to be 3× maxSets.
    VkDescriptorSetLayoutBinding b[3]{};
    for (int i = 0; i < 3; ++i) {
        b[i].binding         = (u32)i;
        b[i].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[i].descriptorCount = 1;
        b[i].stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
    }

    VkDescriptorSetLayoutCreateInfo lci{};
    lci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lci.bindingCount = 3;
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
    ps.descriptorCount = kMaxSets * 3;   // 3 image samplers per set

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

    s_Default = CreateDefaultWhite();
    Msg("[VK WorldMaterial] Init OK (pool=%u sets, 3 bindings: base+detail+lmap, anisotropic 16x)", kMaxSets);
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

WorldMaterial* GetOrCreate(const char* diffuse_name, const char* lmap_name, float alphaRef)
{
    if (!s_SetLayout || !s_Default) return s_Default;
    if (!diffuse_name || !diffuse_name[0]) return s_Default;

    // Same diffuse can pair with different lmaps — key on both.
    std::string key(diffuse_name);
    key.push_back('|');
    if (lmap_name && lmap_name[0]) key.append(lmap_name);

    auto it = s_Cache.find(key);
    if (it != s_Cache.end()) return it->second;

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

    // R4 detail-texture lookup: read `<base>.thm` for detail_name + scale.
    // Materials with no detail association get the grey fallback so the
    // shader's `2 * base * detail` reduces to `base`.
    std::string     detail_name;
    float           detail_scale = 0.0f;
    CVulkanTexture* detail_tex   = s_GreyDetail;
    if (LookupDetailFromTHM(diffuse_name, detail_name, detail_scale)) {
        detail_tex = GetOrLoadDetailTex(detail_name);
    }

    // Lightmap from the level shader's 3rd texture slot. Vert-lit / non-
    // lightmapped materials pass null here → white fallback (no-op multiply).
    CVulkanTexture* lmap_tex = GetOrLoadLmapTex(lmap_name);

    auto* m = xr_new<WorldMaterial>();
    m->tex          = tex;
    m->view         = tex->GetView();
    m->sampler      = s_Sampler;
    m->view_detail  = detail_tex ? detail_tex->GetView() : s_GreyDetail->GetView();
    m->detailScale  = (detail_tex && detail_tex != s_GreyDetail) ? detail_scale : 0.0f;
    m->view_lmap    = lmap_tex   ? lmap_tex->GetView()   : s_WhiteLmap->GetView();
    m->alphaRef     = alphaRef;
    m->set          = AllocateSet();
    if (m->set == VK_NULL_HANDLE) {
        Msg("![VK WorldMaterial] Pool exhausted creating '%s' — falling back to default", diffuse_name);
        tex->Destroy();
        xr_delete(tex);
        xr_delete(m);
        s_Cache.emplace(std::move(key), s_Default);
        return s_Default;
    }
    WriteSet(m->set, m->view, m->view_detail, m->view_lmap);

    s_Cache.emplace(std::move(key), m);
    return m;
}

}}  // namespace VK::WorldMaterialCache
