// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan — CTreeManager Session B: GPU-driven tree render.
//
// Per-frame flow (mirrors the monolith CShadowManager g-buffer tree path,
// adapted to the OGSR forward single-pass renderer):
//   1. Fill the view-frustum UBO from ctx.viewProj.
//   2. Clear per-group draw counts; barrier transfer→compute.
//   3. Per group: dispatch tree_cull.comp — frustum-tests each tree sphere and
//      appends a VkDrawIndexedIndirectCommand (firstInstance = global tree idx)
//      into the group's section, bumping the group's draw count.
//   4. Barrier compute→draw-indirect.
//   5. Begin dynamic-rendering pass (color + depth LOAD, single-layout
//      convention — no swapchain transitions here).
//   6. Per group: bind pipeline variant (tcOffset 24/28) + diffuse + VB/IB,
//      vkCmdDrawIndexedIndirectCount over the group's section.
//
// Build() (Session A) already uploaded m_TreeMetadataBuffer / m_TreeTransforms
// and allocated m_TreeIndirectBuffer / m_TreeDrawCountBuffer + per-texture
// descriptor sets. This file only adds the pipelines + the per-frame path.

#include "stdafx.h"
#include "vk_TreeManager.h"
#include "vk_color_space.h"   // ColorSpace::LinearizeRGB — env colours are authored sRGB
#include "vk_Visual.h"           // vkFTreeVisual (m_mesh / vis.box) — impostor silhouette bake
#include "vk_pass_context.h"
#include "vk_swapchain.h"
#include "vk_scene_color.h"       // HDR scene target format
#include "vk_motionvec.h"         // VK::MotionVec::Format — RG16F MV target (tree wind-sway MV)
#include "vk_texture.h"
#include "vk_shaders.h"          // g_ShaderManager
#include "vk_pipeline_cache.h"   // PipelineCache::GetCacheObject
#include "vk_compute_util.h"     // VK::MakePipelineLayout / CreateComputePipeline
#include "vk_gfx_pipeline.h"     // GfxPipelineBuilder
#include "vk_descriptors.h"      // VK::DescriptorWriter
#include "vk_vsm.h"              // VSM::GetRMaskHandle — receiver mask for the dyn bins (r_vsm_rmask)
#include "vk_profiler.h"         // VK::Prof::ZoneBegin — Bins/Meshlet + Bins/VoxCull sub-zones (r_profiler 2)
#include "vk_barriers.h"         // VK::ImageBarrier / BufferBarrier — impostor bake + indirect
#include "vk_env_light.h"        // VK::EnvLight — set 2 (sun_vp + sun shadow map)
#include "vk_shadow.h"           // ShadowMap::SphereVisible — caster culling (RenderDepth)
#include "vk_LODManager.h"       // VK::kImposterMinDist — floor for the r_tree_dist cut
#include "vk_DetailManager.h"    // shared Hi-Z pyramid (HZBView/HZBSampler/HZBBuiltThisFrame)
#include "CRender_Vulkan.h"      // RImplementation.Details — owner of that pyramid
#include "vk_cull.h"             // VK::ExtractFrustumPlanes (shared with DetailManager)
#include "HW_Vulkan.h"
#include "vk_vrs.h"            // VK::VRS::CmdSetPipelineRate — 2x2 coarse crown-shadow shading
#include "../../xr_3da/IGame_Persistent.h" // g_pGamePersistent->Environment()
#include <bit>                    // std::popcount — diag tree-bitset readbacks
#include "../../xr_3da/Environment.h"      // CEnvDescriptorMixer (sun_color/hemi_color)

// GLOBAL scope (an extern inside namespace VK would mangle as VK::* → LNK2001).
extern float ps_r_sun_boost;   // r_sun_boost — global sun multiplier (see vk_env_light)
extern float ps_r_wind_tree_bend;   // r_wind_tree_bend  — trunk sway intensity
extern float ps_r_wind_tree_anim;   // r_wind_tree_anim  — branch flutter speed
extern float ps_r_wind_tree_trunk;  // r_wind_tree_trunk — trunk anim speed
extern float ps_r_wind_tree_flutter;// r_wind_tree_flutter — crown flutter amplitude
extern float ps_r_wind_tree_crown;  // r_wind_tree_crown — flutter fade-in height
extern int   ps_r_vsm_tree_wind;      // r_vsm_tree_wind — near/far hybrid: near trees cast into the DYNAMIC atlas with live wind
extern float ps_r_vsm_tree_wind_dist; // r_vsm_tree_wind_dist — near set radius (m)
extern int   ps_r_vsm_tree_vrs;       // r_vsm_tree_vrs — 2x2 coarse shading on the dyn crown-shadow raster (A/B)
extern int   ps_r_vsm_debug;          // r_vsm_debug — hybrid stats log
extern int   ps_r_profiler;           // r_profiler — 2+ enables the Bins/* per-dispatch sub-zones
extern int   ps_r_vsm_meshlet;        // r_vsm_meshlet — per-page meshlet cull of tree casters (Phase B)
extern int   ps_r_vsm_hzb;            // r_vsm_hzb — shadow-HZB occlusion cull of tree casters (static pass)
extern float ps_r_vsm_hzb_margin;     // r_vsm_hzb_margin — occluder depth slack (stale-sun safety)
extern int   ps_r_vsm_rmask;          // r_vsm_rmask — receiver mask: dyn bins sub-page-cull vs sampled 8×8 cells
extern int   ps_r_vsm_tree_impostor;       // r_vsm_tree_impostor — near crown mesh → baked sun-facing billboard in the dyn atlas
extern float ps_r_vsm_tree_impostor_scale; // r_vsm_tree_impostor_scale — billboard size vs crown sphere (shadow footprint tuning)
extern int   ps_r_vsm_tree_hull;           // r_vsm_tree_hull — caster LOD: near trees beyond _dist cast from a baked opaque crown hull
extern float ps_r_vsm_tree_hull_dist;      // r_vsm_tree_hull_dist — crown-mesh tier radius (m, light-space lateral, same metric as the near set)
extern int   ps_r_vsm_tree_hull_debug;     // r_vsm_tree_hull_debug — pink lobe overlay over hull-tier trees (forward view)
extern int   ps_r_vsm_tree_hull_lod;       // r_vsm_tree_hull_lod — LIVE coarseness bias for the graded voxel view
extern float ps_r_vsm_tree_hull_vox_px;    // r_vsm_tree_hull_vox_px — target PROJECTED voxel size (pixels) for the LOD cut
extern float ps_r_vsm_tree_hull_vox_fade;  // r_vsm_tree_hull_vox_fade — dithered LOD crossfade band (fraction of handover dist)
extern float ps_r_vsm_tree_hull_vox_far;   // r_vsm_tree_hull_vox_far — extra target px per 100 m (far cubes read chunkier)
extern float ps_r_vsm_tree_hull_vox_stex;   // r_vsm_tree_hull_vox_stex — SHADOW voxel LOD: page texels per voxel edge (UE shadow-view metric)
extern float ps_r_vsm_tree_hull_vox_sfloor; // r_vsm_tree_hull_vox_sfloor — SHADOW-only safety floor on the voxel edge (m)
extern int   ps_r_vsm_tree_hull_vox_cap;    // r_vsm_tree_hull_vox_cap — SHADOW-only max bricks per tree slice
extern int   ps_r_vsm_tree_hull_vox_wpo;    // r_vsm_tree_hull_vox_wpo — UE WPODisableDistance analog: rigid voxel-tier trees live in the static cache
extern int   ps_r_vsm_tree_hull_vox_static; // r_vsm_tree_hull_vox_static — static-tier proxy: 0 = merged shell (default), 1 = bricks
extern int   ps_r_vsm_tree_hull_vox_cull;   // r_vsm_tree_hull_vox_cull — stage-2 per-page brick compaction (vsm_vox_cull.comp)
extern float ps_r_vsm_base;                 // r_vsm_base — VSM clipmap level-0 extent (m); texel(L) = base·2^L / 4096
extern int   ps_r_vsm_tree_hull_vox;       // r_vsm_tree_hull_vox — voxel bake resolution (0 = shell/lobes only)
extern float ps_r_vsm_tree_hull_band;      // r_vsm_tree_hull_band — crown↔voxel shadow crossfade band (fraction of _dist)
extern float ps_r_tree_dist;               // r_tree_dist — forward draw distance for billboard-backed trees (0 = off)
extern int   ps_r_tree_hzb;                // r_tree_hzb — Hi-Z occlusion cull of the forward tree set
extern float ps_r_tree_shade_dist;         // r_tree_shade_dist — range past which tree.frag drops its subtle shading terms

namespace VK
{

// Single source of truth for the cut distance, shared by the colour cull and the
// camera depth prepass. They MUST agree: if the prepass keeps a tree the colour
// pass drops, that tree writes depth with nothing shading it — black silhouettes
// (the bug the frustum margin in tree_cull.comp already guards against). Reading
// the cvar from two places invites exactly that drift, so both go through here.
float TreeFlodCutDist()
{
    if (ps_r_tree_dist <= 0.0f) return 0.0f;            // cut disabled
    // Never cut CLOSER than where the imposter starts drawing, whatever the cvar
    // says — inside that range the mesh is the ONLY thing representing the tree,
    // so an over-eager r_tree_dist would delete trees outright instead of
    // de-duplicating them. Cutting later than the imposter is merely less saving.
    return _max(ps_r_tree_dist, kImposterMinDist);
}

// Shorthands for the set-layout type lists handed to VK::MakeSetLayout — this
// file declares ten sets and the lists read as tables of bindings.
namespace {
    constexpr VkDescriptorType kSSBO = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    constexpr VkDescriptorType kUBO  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    constexpr VkDescriptorType kTex  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
}

bool CTreeManager::IsReady() const
{
    return m_bBuilt
        && m_CullPipeline != VK_NULL_HANDLE
        && (m_GfxPipeline24 != VK_NULL_HANDLE || m_GfxPipeline28 != VK_NULL_HANDLE)
        && m_TreeIndirectBuffer != nullptr
        && m_TreeDrawCountBuffer != nullptr
        && m_XformDescSet != VK_NULL_HANDLE
        && !m_Groups.empty();
}

// ============================================================================
// Frustum UBO — host-visible, rewritten each frame in Render().
// ============================================================================
void CTreeManager::CreateFrustumUBO()
{
    m_FrustumUBO = xr_new<CVulkanBuffer>();
    m_FrustumUBO->Create(sizeof(TreeFrustumUBO),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
}

// ============================================================================
// Cull compute pipeline — descriptor {meta SSBO, frustum UBO, indirect SSBO,
// count SSBO}, 20 B push. Loads tree_cull.comp.spv.
// ============================================================================
void CTreeManager::CreateCullPipeline()
{
    if (!g_ShaderManager) { Msg("![VK Trees] g_ShaderManager null — cull pipeline disabled"); return; }
    if (!m_TreeMetadataBuffer || !m_FrustumUBO || !m_TreeIndirectBuffer || !m_TreeDrawCountBuffer) return;

    // 0=meta(SSBO) 1=frustum(UBO, legacy) 2=indirect(SSBO) 3=count(SSBO) 4=HZB(tex).
    m_CullDescLayout = VK::MakeSetLayout({ kSSBO, kUBO, kSSBO, kSSBO, kTex },
                                         VK_SHADER_STAGE_COMPUTE_BIT, "Trees.Cull");
    if (m_CullDescLayout == VK_NULL_HANDLE) return;

    m_CullDescPool = VK::MakeDescriptorPool({ kSSBO, kUBO, kSSBO, kSSBO, kTex }, 1, "Trees.Cull");
    if (m_CullDescPool == VK_NULL_HANDLE) return;
    if (!VK::AllocSets(m_CullDescPool, m_CullDescLayout, 1, &m_CullDescSet, "Trees.Cull")) return;

    VK::DescriptorWriter(m_CullDescSet)
        .StorageBuffer(0, m_TreeMetadataBuffer->GetHandle())
        .UniformBuffer(1, m_FrustumUBO->GetHandle())
        .StorageBuffer(2, m_TreeIndirectBuffer->GetHandle())
        .StorageBuffer(3, m_TreeDrawCountBuffer->GetHandle())
        .Flush();
    // Binding 4 (HZB) is written per frame in Render() — the pyramid is created
    // and can be RECREATED on resize by CDetailManager, so a write here would go
    // stale. Until the first write it stays unbound, hence hzb_on gates on the
    // view being non-null rather than on the cvar alone.

    // Pipeline layout: 1 set + push = frustum planes (6×vec4 = 96 B) + 5 u32 (20 B).
    // The frustum used to live in a single-buffered UBO (binding 1) but that raced
    // the in-flight frames during rotation (the next frame's CPU write clobbered this
    // frame's planes before the GPU cull read them) → the colour cull tested a rotated
    // frustum vs the depth prepass → black tree silhouettes at screen edges. Push
    // constants are recorded per dispatch, so they can't alias across frames.
    // TreeCullPush = 208 B: over the 128 B guaranteed minimum, under the 256 B
    // every desktop GPU exposes (this renderer already ships a 240 B graphics
    // push). Check it rather than let vkCreatePipelineLayout fail obscurely.
    if (sizeof(TreeCullPush) > VulkanHW.Caps.maxPushConstantsSize) {
        Msg("![VK Trees] TreeCullPush %u B exceeds device maxPushConstantsSize %u B — cull pipeline disabled",
            (u32)sizeof(TreeCullPush), VulkanHW.Caps.maxPushConstantsSize);
        return;
    }
    m_CullPipelineLayout = VK::MakePipelineLayout({ m_CullDescLayout }, sizeof(TreeCullPush));
    if (m_CullPipelineLayout == VK_NULL_HANDLE) return;

    m_CullPipeline = VK::CreateComputePipeline("tree_cull.comp.spv", m_CullPipelineLayout, "Trees.Cull");
    if (m_CullPipeline == VK_NULL_HANDLE) return;
    Msg("[VK Trees] Cull pipeline OK");
}

// ============================================================================
// Graphics set 0 — transforms SSBO (read by tree.vert).
// ============================================================================
void CTreeManager::CreateXformDescriptor()
{
    VK::Vram::Scope _vram_scope("Trees");
    if (!m_TreeTransformsBuffer) return;

    // SSFX wind flow map (s_waves) — sampled in tree.vert for the crown flutter.
    // Load it + a linear/repeat sampler here so the wind branch has its texture.
    if (m_WaveSampler == VK_NULL_HANDLE) {
        VkSamplerCreateInfo sci{};
        sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter = VK_FILTER_LINEAR; sci.minFilter = VK_FILTER_LINEAR;
        sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        sci.maxLod = VK_LOD_CLAMP_NONE;
        vkCreateSampler(VulkanHW.m_Device, &sci, nullptr, &m_WaveSampler);
    }
    if (m_WaveTex == nullptr) {
        string_path full;
        FS.update_path(full, "$game_textures$", "fx\\wind_wave.dds");
        if (FS.exist(full)) {
            auto* tex = xr_new<VK::CVulkanTexture>();
            if (tex->LoadDDS(full, /*applyBCSwizzle*/ false)) { m_WaveTex = tex; Msg("[VK Tree] SSFX wind flow map loaded"); }
            else { Msg("![VK Tree] LoadDDS failed for fx\\wind_wave.dds — tree flutter flat"); xr_delete(tex); }
        } else {
            Msg("![VK Tree] fx\\wind_wave.dds not found — tree flutter flat");
        }
    }

    // Set 0: binding 0 = transforms SSBO (vertex), binding 1 = s_waves (vertex).
    VkDescriptorSetLayoutBinding b[3]{};
    b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[0].descriptorCount = 1; b[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    b[1].binding = 1; b[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b[1].descriptorCount = 1; b[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    b[2].binding = 2; b[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;   // visible-tree bitset (tree.frag diag)
    b[2].descriptorCount = 1; b[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lci.bindingCount = 3; lci.pBindings = b;
    vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &m_XformDescLayout);

    VkDescriptorPoolSize ps[2] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 } };
    VkDescriptorPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = 1; pci.poolSizeCount = 2; pci.pPoolSizes = ps;
    vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &m_XformDescPool);

    VkDescriptorSetAllocateInfo dai{};
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = m_XformDescPool; dai.descriptorSetCount = 1;
    dai.pSetLayouts = &m_XformDescLayout;
    vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, &m_XformDescSet);

    // Visible-tree bitset (tree.frag marks; Render clears per frame + snapshots
    // last frame's set to the host RB). Created here so the set write below can
    // bind it — the layout requires a valid buffer at binding 2.
    {
        const u32 W = (m_TotalCount + 31u) / 32u;
        xr_vector<u32> zero(W, 0u);
        m_SeenBits = xr_new<CVulkanBuffer>();
        m_SeenBits->Create((VkDeviceSize)W * sizeof(u32),
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                           VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
        m_SeenBits->Upload(zero.data(), (VkDeviceSize)W * sizeof(u32));
        m_SeenBitsRB = xr_new<CVulkanBuffer>();
        m_SeenBitsRB->Create((VkDeviceSize)W * sizeof(u32), VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
        m_SeenBitsPtr = (u32*)m_SeenBitsRB->Map();
        if (m_SeenBitsPtr) memset(m_SeenBitsPtr, 0, (size_t)W * sizeof(u32));
    }

    VK::DescriptorWriter dw(m_XformDescSet);
    dw.StorageBuffer(0, m_TreeTransformsBuffer->GetHandle())
      .StorageBuffer(2, m_SeenBits->GetHandle());
    // s_waves view — falling back to the SSBO-only case is unsafe (the shader
    // samples it), so bind the flow map; if it failed to load the tree still
    // draws (flat wind) and we simply skip the write rather than bind a null view.
    if (m_WaveTex)
        dw.ImageSampler(1, m_WaveTex->GetView(), m_WaveSampler);
    dw.Flush();
}

// ============================================================================
// Graphics pipelines — tree.vert/frag, two variants by UV byte offset.
// Set 0 = transforms (m_XformDescLayout), Set 1 = diffuse (m_TexDescLayout).
// ============================================================================
void CTreeManager::CreateGfxPipelines()
{
    if (!g_ShaderManager) { Msg("![VK Trees] g_ShaderManager null — gfx pipeline disabled"); return; }
    if (m_XformDescLayout == VK_NULL_HANDLE || m_TexDescLayout == VK_NULL_HANDLE) return;

    // set0 = per-tree transforms (SSBO), set1 = per-group diffuse,
    // set2 = shared env lighting (sun_vp + sun shadow map — vk_env_light).
    VkDescriptorSetLayout setLayouts[3] = { m_XformDescLayout, m_TexDescLayout, VK::EnvLight::GetSetLayout() };
    if (setLayouts[2] == VK_NULL_HANDLE) { Msg("![VK Trees] EnvLight layout not ready"); return; }

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcr.offset = 0; pcr.size = sizeof(TreeGfxPush);
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 3; plci.pSetLayouts = setLayouts;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &m_GfxPipelineLayout);

    VkShaderModule vs = g_ShaderManager->Load("tree.vert.spv");
    VkShaderModule fs = g_ShaderManager->Load("tree.frag.spv");
    if (vs == VK_NULL_HANDLE || fs == VK_NULL_HANDLE) {
        Msg("![VK Trees] tree.{vert,frag}.spv load failed"); return;
    }

    // Vertex input for every tree pipeline below: stride=32, FLOAT3 pos @ 0,
    // SHORT2 SSCALED UV @ tcOffset.
    auto createVariant = [&](u32 tcOffset, VkPipeline& out) -> bool
    {
        out = VK::GfxPipelineBuilder(m_GfxPipelineLayout)
            .Vert(vs).Frag(fs)
            .Binding(0, 32)
            .Attr(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0)
            .Attr(1, 0, VK_FORMAT_R16G16_SSCALED,   tcOffset)
            .Cull(VK_CULL_MODE_NONE)          // trees: double-sided leaves
            .Depth(true, true)
            .Color(VK::SceneColor::Format())
            .DepthTarget(Swapchain.m_DepthFormat)
            .Build("Trees gfx (tcOff=%u)", tcOffset);
        return out != VK_NULL_HANDLE;
    };

    bool ok24 = createVariant(24, m_GfxPipeline24);
    bool ok28 = createVariant(28, m_GfxPipeline28);
    Msg("[VK Trees] Gfx pipeline (stride=32, tcOff 24=%s 28=%s)",
        ok24 ? "ok" : "FAIL", ok28 ? "ok" : "FAIL");

    // Shadow caster variants: tree_depth.{vert,frag} (alpha-tested, depth-only
    // into the sun map's D32), same pipeline layout (push offset 0..72 used).
    VkShaderModule dvs = g_ShaderManager->Load("tree_depth.vert.spv");
    VkShaderModule dfs = g_ShaderManager->Load("tree_depth.frag.spv");
    if (dvs == VK_NULL_HANDLE || dfs == VK_NULL_HANDLE) {
        Msg("![VK Trees] tree_depth.{vert,frag}.spv missing — tree shadow casting disabled");
        return;
    }
    auto createDepthVariant = [&](u32 tcOffset, VkPipeline& out)
    {
        out = VK::GfxPipelineBuilder(m_GfxPipelineLayout)
            .Vert(dvs).Frag(dfs)
            .Binding(0, 32)
            .Attr(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0)
            .Attr(1, 0, VK_FORMAT_R16G16_SSCALED,   tcOffset)
            .Cull(VK_CULL_MODE_NONE)          // double-sided leaves
            .DynamicDepthBias()               // caller sets the sun bias
            .Depth(true, true)
            .DepthTarget(VK_FORMAT_D32_SFLOAT)   // sun shadow map, no color attachments
            .Build("Trees depth (tcOff=%u)", tcOffset);
    };
    createDepthVariant(24, m_DepthPipeline24);
    createDepthVariant(28, m_DepthPipeline28);
}

// ============================================================================
// Shadow caster path — depth-only alpha-tested draws of trees inside the sun
// ortho box. CPU-culled per tree (runs only on sun static-map redraws, so the
// metadata walk + per-tree draws are off the per-frame path).
// ============================================================================
void CTreeManager::RenderDepth(VkCommandBuffer cmd, const Fmatrix& lightVP, s32 cascade,
                               const CFrustum* frustum, float minDist, float maxDist, float flodCut)
{
    if (!m_bBuilt || m_MetaCPU.empty()) return;
    if (m_XformDescSet == VK_NULL_HANDLE || m_GfxPipelineLayout == VK_NULL_HANDLE) return;
    if (m_DepthPipeline24 == VK_NULL_HANDLE && m_DepthPipeline28 == VK_NULL_HANDLE) return;

    // Push: light view·proj + tree UV quant + leaf alpha cutoff (first 72 B of
    // the TreeGfxPush range; sun/hemi tail unused by the depth shaders).
    TreeGfxPush pc{};
    pc.mViewProj = lightVP;
    pc.uvScale   = 1.0f / 2048.0f;
    pc.alphaRef  = 200.0f / 255.0f;
    // Same wind as the forward pass — the shadow caster must bend with the trees,
    // else a swaying trunk self-shadows (black band). See tree_depth.vert.
    pc.wsetup_trees.set(ps_r_wind_tree_anim, ps_r_wind_tree_trunk, ps_r_wind_tree_bend, 0.1f);
    if (g_pGamePersistent) {
        const Fvector3 wa = g_pGamePersistent->Environment().wind_anim;
        pc.wind_anim.set(wa.x, wa.y, wa.z, ps_r_wind_tree_flutter);
        if (auto* E = g_pGamePersistent->Environment().CurrentEnv)
            pc.wind_params.set(E->wind_direction, E->wind_velocity, 0.0f, 0.0f);
    }
    pc.wind_params.w = ps_r_wind_tree_crown;   // flutter height gate — must match the forward pass
    vkCmdPushConstants(cmd, m_GfxPipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(pc), &pc);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_GfxPipelineLayout,
                            0, 1, &m_XformDescSet, 0, nullptr);

    VkPipeline      lastPipe = VK_NULL_HANDLE;
    VkDescriptorSet lastTex  = VK_NULL_HANDLE;
    u32 nDraw = 0;

    for (const TreeIndirectGroup& grp : m_Groups)
    {
        VkPipeline pipe = (grp.tcOffset == 24) ? m_DepthPipeline24 : m_DepthPipeline28;
        if (pipe == VK_NULL_HANDLE) continue;

        bool boundGroup = false;
        for (u32 i = 0; i < grp.meshCount; ++i)
        {
            const GpuTreeMeta& m = m_MetaCPU[grp.meshOffset + i];
            // Distance split (wind shadows): keep near trees in the per-frame
            // dynamic layer and far trees in the cached static layer.
            if (minDist > 0.0f || maxDist < 1e9f) {
                const float d = Device.vCameraPosition.distance_to(m.sphere_P);
                if (d < minDist || d >= maxDist) continue;
            }
            // r_tree_dist twin of the colour cull (camera prepass only — callers
            // pass flodCut == 0 everywhere else). CENTRE distance, no radius slack,
            // so this set is strictly SMALLER than what tree_cull.comp keeps
            // (it subtracts sphere_R): colour ⊇ prepass holds by construction, and
            // FP differences between the two code paths can't invert it.
            if (flodCut > 0.0f && (m.flags & TREE_FLOD_BACKED) != 0u) {
                if (Device.vCameraPosition.distance_to(m.sphere_P) > flodCut) continue;
            }
            if (frustum) {
                if (!frustum->testSphere_dirty(m.sphere_P, m.sphere_R)) continue;
            } else if (cascade >= 0 ? !ShadowMap::CascadeSphereVisible((u32)cascade, m.sphere_P, m.sphere_R)
                                    : !ShadowMap::SphereVisible(m.sphere_P, m.sphere_R)) continue;

            if (!boundGroup) {
                if (pipe != lastPipe) {
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
                    lastPipe = pipe;
                }
                if (grp.descSetIdx < m_TexDescSets.size() && m_TexDescSets[grp.descSetIdx] != lastTex) {
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_GfxPipelineLayout,
                                            1, 1, &m_TexDescSets[grp.descSetIdx], 0, nullptr);
                    lastTex = m_TexDescSets[grp.descSetIdx];
                }
                VkDeviceSize vbOff = 0;
                vkCmdBindVertexBuffers(cmd, 0, 1, &grp.vb, &vbOff);
                vkCmdBindIndexBuffer(cmd, grp.ib, 0, VK_INDEX_TYPE_UINT16);
                boundGroup = true;
            }
            // firstInstance = global tree index → gl_InstanceIndex picks the xform.
            vkCmdDrawIndexed(cmd, m.index_count, 1, m.ib_first, (s32)m.first_vertex, grp.meshOffset + i);
            ++nDraw;
        }
    }

    static bool s_diag = false;
    if (!s_diag && nDraw) { s_diag = true; Msg("[VK Trees] first shadow render: %u trees in the sun box", nDraw); }
}

// ============================================================================
// Per-frame render.
// ============================================================================
void CTreeManager::Render(VK::FrameContext& ctx)
{
    VK::Vram::Scope _vram_scope("Trees");
    if (!IsReady()) return;
    if (ctx.cmd == VK_NULL_HANDLE || ctx.viewProj == nullptr) return;

    const VkCommandBuffer cmd = ctx.cmd;
    const Fmatrix& vp = *ctx.viewProj;
    const u32 numGroups = (u32)m_Groups.size();

    // ----- 1) Extract this frame's frustum planes (Gribb/Hartmann; same as grass).
    // These go into the per-dispatch PUSH below — NOT the shared UBO — so a rotating
    // camera's next-frame planes can't clobber this frame's cull (the old single-
    // buffered UBO did, producing black tree silhouettes at the screen edge).
    TreeCullPush cullPush{};
    VK::ExtractFrustumPlanes(vp, cullPush.planes);
    // Distance cut for billboard-backed trees (r_tree_dist). The prepass applies
    // the SAME value via RenderDepth(..., flodCut) but on centre distance, so this
    // one — measured from the sphere's near side — always keeps a superset. See
    // tree_cull.comp.glsl.
    cullPush.cut_dist = TreeFlodCutDist();
    cullPush.eye_x = Device.vCameraPosition.x;
    cullPush.eye_y = Device.vCameraPosition.y;
    cullPush.eye_z = Device.vCameraPosition.z;

    // ----- 1b) Hi-Z occlusion (r_tree_hzb). We deliberately do NOT build the
    // pyramid ourselves: Pass_World builds it from the finished prepass depth
    // (which trees are part of) and runs BEFORE this pass. If it did not build
    // this frame, the only pyramid available is last frame's — culling against a
    // stale camera would pop trees in and out, so we simply skip the test.
    // ⚠ The prepass keeps drawing the full frustum set on purpose: it IS the
    // pyramid's source. See the argument in tree_cull.comp.glsl for why culling
    // only the colour set cannot leave unshaded pixels here.
    cullPush.viewProj  = vp;
    cullPush.hzb_focal = 1.0f / _max(0.05f, tanf(deg2rad(Device.fFOV) * 0.5f));
    cullPush.hzb_on    = 0;
    VkImageView hzbView = VK_NULL_HANDLE;
    if (ps_r_tree_hzb && RImplementation.Details
        && RImplementation.Details->HZBReady() && RImplementation.Details->HZBBuiltThisFrame())
    {
        hzbView = RImplementation.Details->HZBView();
        if (hzbView != VK_NULL_HANDLE) cullPush.hzb_on = 1;
    }
    // The descriptor must ALWAYS point at something valid, even with the test off
    // (an unbound combined-image-sampler is UB the moment the shader is dispatched,
    // regardless of whether the branch reading it is taken).
    if (hzbView == VK_NULL_HANDLE && RImplementation.Details)
        hzbView = RImplementation.Details->DummyHZBView();
    // Key the cache on view + GENERATION: after DestroyHZB+CreateHZB the driver can
    // hand back the same numeric VkImageView, and a handle-only check would then keep
    // a descriptor bound to the destroyed object.
    const u32 hzbGen = RImplementation.Details ? RImplementation.Details->HZBGeneration() : 0u;
    if (hzbView == VK_NULL_HANDLE) {
        cullPush.hzb_on = 0;                       // no pyramid, no dummy — never sample
    } else if (hzbView != m_CullHzbView || hzbGen != m_CullHzbGen) {
        // ⚠ ONLY on change. m_CullDescSet is a SINGLE set while kFramesInFlight
        // frames are in flight, so rewriting it every frame would update a set
        // that an executing command buffer still references (UB — the same trap
        // the VSM cascade rebind documents). The view changes only when
        // CDetailManager recreates the pyramid (resize), and the device is idle
        // across that, so a write here is safe.
        VK::DescriptorWriter(m_CullDescSet)
            .ImageSampler(4, hzbView, RImplementation.Details->HZBSampler(), VK_IMAGE_LAYOUT_GENERAL)
            .Flush();
        m_CullHzbView = hzbView;
        m_CullHzbGen  = hzbGen;
    }

    // ----- 2) Clear per-group draw counts; barrier transfer→compute. ----------
    // Same cross-frame WAR hazard as grass: m_TreeIndirectBuffer and
    // m_TreeDrawCountBuffer are single-buffered while 3 frames are in flight.
    // Order the previous frame's indirect fetches before this frame's
    // fill/compute rewrites (execution dependency is enough for WAR).
    {
        VkMemoryBarrier b{};
        b.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        b.srcAccessMask = 0;
        b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &b, 0, nullptr, 0, nullptr);
    }
    // Visible-tree bitset (r_profiler diag): snapshot LAST frame's marks to the
    // host RB, then clear for this frame's fragments. Last frame's fragment
    // writes are ordered before this copy by the queue + the barrier below.
    const bool treeStats = ps_r_profiler > 0 && m_SeenBits && m_SeenBitsPtr;
    if (treeStats) {
        VkMemoryBarrier fb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, &fb, 0, nullptr, 0, nullptr);
        VkBufferCopy sc{ 0, 0, m_SeenBits->GetSize() };
        vkCmdCopyBuffer(cmd, m_SeenBits->GetHandle(), m_SeenBitsRB->GetHandle(), 1, &sc);
        VkMemoryBarrier tb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                            VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, &tb, 0, nullptr, 0, nullptr);   // copy(read) before the clear(write) below
    }
    vkCmdFillBuffer(cmd, m_TreeDrawCountBuffer->GetHandle(), 0, VK_WHOLE_SIZE, 0u);
    if (treeStats)
        vkCmdFillBuffer(cmd, m_SeenBits->GetHandle(), 0, VK_WHOLE_SIZE, 0u);
    {
        VkMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 1, &b, 0, nullptr, 0, nullptr);
    }

    // ----- 3) Cull dispatch per group. ----------------------------------------
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_CullPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_CullPipelineLayout,
                            0, 1, &m_CullDescSet, 0, nullptr);
    for (u32 g = 0; g < numGroups; ++g)
    {
        const TreeIndirectGroup& grp = m_Groups[g];
        // planes, cut_dist and eye_* stay as set above (frame-constant); only the
        // per-group indices change. This slot used to be `_unused` and was zeroed
        // here every group — it now carries cut_dist, so leave it alone.
        cullPush.mesh_count  = grp.meshCount;
        cullPush.mesh_offset = grp.meshOffset;
        cullPush.output_base = g * m_MaxGroupMeshCount;
        cullPush.count_index = g;
        vkCmdPushConstants(cmd, m_CullPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(cullPush), &cullPush);
        vkCmdDispatch(cmd, (grp.meshCount + 255) / 256, 1, 1);
    }

    // ----- 4) Barrier compute→draw-indirect (+transfer for the count readback).
    {
        VkMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, &b, 0, nullptr, 0, nullptr);
    }

    // ----- 4b) Forward drawn-count readback (r_profiler): mirror this frame's
    // per-group counts to the host (read a frame or two stale — diagnostic) and
    // log the total every 2 s: the "how many trees do we actually draw" number
    // the VSM-epic backlog asked for. Only the main-view cull above writes these
    // counts (the shadow paths draw per-mesh on the CPU), so this is main-view.
    if (ps_r_profiler > 0) {
        if (!m_FwdCountRB) {
            m_FwdCountRB = xr_new<CVulkanBuffer>();
            m_FwdCountRB->Create((VkDeviceSize)numGroups * sizeof(u32),
                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
            m_FwdCountPtr = (u32*)m_FwdCountRB->Map();
            if (m_FwdCountPtr) memset(m_FwdCountPtr, 0, (size_t)numGroups * sizeof(u32));
        }
        if (m_FwdCountPtr) {
            VkBufferCopy rc{ 0, 0, (VkDeviceSize)numGroups * sizeof(u32) };
            vkCmdCopyBuffer(cmd, m_TreeDrawCountBuffer->GetHandle(), m_FwdCountRB->GetHandle(), 1, &rc);
            if (Device.dwTimeGlobal > m_FwdLastLog + 2000) {
                m_FwdLastLog = Device.dwTimeGlobal;
                u64 drawn = 0; u32 nonEmpty = 0, maxGrp = 0;
                for (u32 g = 0; g < numGroups; ++g) {
                    const u32 c = m_FwdCountPtr[g];
                    drawn += c;
                    if (c) { ++nonEmpty; maxGrp = _max(maxGrp, c); }
                }
                // visible = trees whose fragments survived alpha + early-Z last frame
                // (tree.frag bitset) — the honest "on screen" count vs the frustum cut.
                u32 vis = 0;
                if (m_SeenBitsPtr)
                    for (u32 i = 0; i < (m_TotalCount + 31u) / 32u; ++i) vis += (u32)std::popcount(m_SeenBitsPtr[i]);
                // cut= is stamped into the line on purpose: an A/B of r_tree_dist
                // compares two logs, and "which mode was this run in" must be IN
                // the measurement, not remembered alongside it.
                Msg("[VK Trees] forward drawn: %llu of %u instances (visible %u) | %u/%u groups, max group %u | cut=%.0fm flod-backed=%u shade=%.0fm hzb=%d",
                    (unsigned long long)drawn, m_TotalCount, vis, nonEmpty, numGroups, maxGrp,
                    TreeFlodCutDist(), m_FlodBackedCount, ps_r_tree_shade_dist, ps_r_tree_hzb);
            }
        }
    }

    // ----- 5) Begin dynamic-rendering pass (LOAD color + depth from world). ----
    VK::BeginOverlayRendering(cmd, ctx);   // shared overlay begin — see vk_pass_context.h

    // Push constants + transforms (set 0) once — shared across both variants.
    TreeGfxPush pc{};
    pc.mViewProj = vp;
    // Tree UVs are quantized with FTreeVisual_quant = 32768/16 = 2048
    // (FTreeVisual.cpp tree_data consts.xy), NOT the static-geometry 1/1024.
    pc.uvScale   = 1.0f / 2048.0f;
    pc.alphaRef  = 200.0f / 255.0f;
    pc.statsOn   = treeStats ? 1.0f : 0.0f;   // tree.frag: mark the visible-tree bitset
    // r_tree_shade_dist — ONLY the forward colour pass gets this. The depth caster
    // and the hull debug view share this push block but have no shading to drop, and
    // the shadow passes must not thin out by camera distance at all.
    pc.shadeDist = ps_r_tree_shade_dist;
    // Env lighting (colorize the baked per-tree hemi factor + open-sky sun term so
    // foliage tracks time-of-day like the world). Neutral fallback if env not up.
    pc.vSunColor.set(0.6f, 0.6f, 0.6f, 0.0f);
    pc.vHemiColor.set(0.45f, 0.45f, 0.45f, 0.0f);
    // SSFX trunk wind (screenspace_wind.h): direction/velocity drive the sway,
    // drift from the shared Environment.wind_anim. Defaults match SSFX "10 - Wind"
    // (branches 11.0, trunk 0.15, bend 0.5, min wind speed 0.1).
    pc.wsetup_trees.set(ps_r_wind_tree_anim, ps_r_wind_tree_trunk, ps_r_wind_tree_bend, 0.1f);
    if (g_pGamePersistent) {
        auto& envMgr = g_pGamePersistent->Environment();
        const Fvector3 wa = envMgr.wind_anim;
        pc.wind_anim.set(wa.x, wa.y, wa.z, ps_r_wind_tree_flutter);   // w = crown flutter amplitude
        if (auto* E = envMgr.CurrentEnv) {
            pc.vSunColor.set(E->sun_color.x, E->sun_color.y, E->sun_color.z, 0.0f);
            pc.vHemiColor.set(E->hemi_color.x, E->hemi_color.y, E->hemi_color.z, 0.0f);
            pc.wind_params.set(E->wind_direction, E->wind_velocity, 0.0f, 0.0f);
        }
    }
    pc.wind_params.w = ps_r_wind_tree_crown;   // leaf-flutter fade-in height (gate the low trunk)
    // Linearise before the boost (see vk_env_light / vk_DetailManager_Render): trees,
    // like grass, take their env colours by push constant and never see the LightUBO.
    VK::ColorSpace::LinearizeRGB(pc.vSunColor);
    VK::ColorSpace::LinearizeRGB(pc.vHemiColor);
    // Same global sun boost as the world (vk_env_light premultiplies it into
    // the LightUBO; the tree sun colour travels via push constants).
    pc.vSunColor.mul(ps_r_sun_boost);

    // Roll wind history for the MV overlay: this frame's wind → cur, last frame's
    // cur → prev (the sway delta). First frame → prev == cur (zero sway motion).
    m_MvWindParamsPrev = m_MvWindValid ? m_MvWindParams : pc.wind_params;
    m_MvWsetupPrev     = m_MvWindValid ? m_MvWsetup     : pc.wsetup_trees;
    m_MvWindAnimPrev   = m_MvWindValid ? m_MvWindAnim   : pc.wind_anim;
    m_MvWindParams = pc.wind_params;
    m_MvWsetup     = pc.wsetup_trees;
    m_MvWindAnim   = pc.wind_anim;
    m_MvWindValid  = true;

    vkCmdPushConstants(cmd, m_GfxPipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(pc), &pc);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_GfxPipelineLayout,
                            0, 1, &m_XformDescSet, 0, nullptr);
    // set 2 = shared env lighting (sun_vp + sun shadow map) — updated by Pass_World.
    if (VkDescriptorSet envSet = VK::EnvLight::GetCurrentSet())
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_GfxPipelineLayout,
                                2, 1, &envSet, 0, nullptr);

    // ----- 6) Per-group indirect draw. ----------------------------------------
    VkPipeline      lastPipe = VK_NULL_HANDLE;
    VkDescriptorSet lastTex  = VK_NULL_HANDLE;
    for (u32 g = 0; g < numGroups; ++g)
    {
        const TreeIndirectGroup& grp = m_Groups[g];
        VkPipeline pipe = (grp.tcOffset == 24) ? m_GfxPipeline24 : m_GfxPipeline28;
        if (pipe == VK_NULL_HANDLE) continue;

        if (pipe != lastPipe) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
            lastPipe = pipe;
        }
        if (grp.descSetIdx < m_TexDescSets.size()) {
            VkDescriptorSet tex = m_TexDescSets[grp.descSetIdx];
            if (tex != lastTex) {
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_GfxPipelineLayout,
                                        1, 1, &tex, 0, nullptr);
                lastTex = tex;
            }
        }

        VkDeviceSize vbOff = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &grp.vb, &vbOff);
        vkCmdBindIndexBuffer(cmd, grp.ib, 0, VK_INDEX_TYPE_UINT16);

        const VkDeviceSize cmdOff = (VkDeviceSize)g * m_MaxGroupMeshCount * sizeof(VkDrawIndexedIndirectCommand);
        const VkDeviceSize cntOff = (VkDeviceSize)g * sizeof(u32);
        vkCmdDrawIndexedIndirectCount(cmd,
            m_TreeIndirectBuffer->GetHandle(), cmdOff,
            m_TreeDrawCountBuffer->GetHandle(), cntOff,
            m_MaxGroupMeshCount, sizeof(VkDrawIndexedIndirectCommand));
    }

    // ----- 7) Crown VOXEL-cloud viewmode (r_vsm_tree_hull_debug): the UE Nanite-foliage
    // look — every hulled crown drawn as individual colored cubes, instanced from m_VoxVB
    // (per-INSTANCE vertex stream; the cube's 36 corners come from gl_VertexIndex). The
    // LOD level per tree keeps the PROJECTED voxel size roughly constant at
    // r_vsm_tree_hull_vox_px pixels (voxel edge doubles per level) — Nanite's screen-
    // error cut: fine cubes near, big cubes far, smooth density across the treeline.
    // mode 1 = hull-TIER trees only (shows the _dist boundary); mode 2 = EVERY hulled
    // tree, so you can walk up to a near crown and inspect its voxels.
    if (ps_r_vsm_tree_hull_debug && m_MetaCPU.size() == m_TotalCount &&
        m_VsmNearCPU.size() == m_TotalCount)
    {
        const bool    allHulls = ps_r_vsm_tree_hull_debug >= 2;
        const Fvector camPos   = Device.vCameraPosition;
        const int     bias     = ps_r_vsm_tree_hull_lod;   // live coarseness tuning (+/-)
        if (m_VoxDebugPipe != VK_NULL_HANDLE && m_VoxVB && m_VoxTotal &&
            m_VoxLodInfoCPU.size() == (size_t)m_TotalCount * kHullLods)
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_VoxDebugPipe);
            VkDeviceSize vbOff = 0;
            VkBuffer vvb = m_VoxVB->GetHandle();
            vkCmdBindVertexBuffers(cmd, 0, 1, &vvb, &vbOff);
            // World meters one screen pixel spans at 1 m distance. The per-tree target
            // cube edge is T = dist × pxAt1m × px(dist), where px(dist) GROWS with
            // distance (vox_px base + vox_px_far per 100 m) — so voxels read small near
            // and VISIBLY chunky far, a continuous gradient rather than constant screen
            // size. The live LOD bias folds in as a ×2-per-step multiplier.
            const float pxAt1m  = 2.f * tanf(deg2rad(Device.fFOV) * 0.5f) / (float)(std::max)(1u, Device.dwHeight);
            const float biasMul = powf(2.f, (float)bias);
            const float pxBase  = (std::max)(1.f, ps_r_vsm_tree_hull_vox_px);
            const float pxFar   = ps_r_vsm_tree_hull_vox_far;
            const float fadeBand = ps_r_vsm_tree_hull_vox_fade;   // crossfade width, fraction of the handover size (0 = hard cut)
            // Telemetry (~2 s cadence while the viewmode is on): per-level draw counts,
            // instance totals, and one probe tree per doubling distance band — enough to
            // read the size-vs-distance curve straight from the log.
            static u32 s_voxLogT = 0;
            const bool doLog = (Device.dwTimeGlobal - s_voxLogT > 2000);
            if (doLog) s_voxLogT = Device.dwTimeGlobal;
            u32 lvlDraws[kHullLods] = {}; u32 fadePairs = 0, treesDrawn = 0; u64 instTotal = 0;
            struct VoxProbe { float dist, T, size, fade; int lvl; u32 cnt; bool used; };
            VoxProbe probes[9] = {};   // bands: <16, 16-32, ..., >2048 m
            // Per-draw payload → the uvScale/alphaRef/_pad slots of TreeGfxPush (offset 64;
            // the voxel shaders never read the tree UV fields). fade: 0 = solid, >0 = this
            // draw dissolves OUT by that fraction, <0 = dissolves IN (screen-door in the FS).
            struct VoxPush { float fade; float density; u32 idx; float size; };
            float instEff = 0.f;   // telemetry: expected cubes after density thinning
            auto drawLod = [&](const TreeVoxLod& l, u32 tree, float fade, float cubeSize) {
                // Density thinning: as the rendered cube outgrows this grid's cell, cull a
                // matching fraction of instances in the VS — (cell/cube)² keeps the crown's
                // covered area constant, so the cube COUNT is continuous in distance (25%
                // of the fine grid ≈ 100% of the next, ~4x-sparser grid at the handover).
                const float r = l.size / (std::max)(cubeSize, 1e-4f);
                const float density = (std::min)(1.f, r * r);
                const VoxPush vp{ fade, density, tree, cubeSize };
                vkCmdPushConstants(cmd, m_GfxPipelineLayout,
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   (u32)offsetof(TreeGfxPush, uvScale), sizeof(vp), &vp);
                vkCmdDraw(cmd, 36, l.count, 0, l.first);
                instEff += density * (float)l.count;
            };
            for (u32 i = 0; i < m_TotalCount; ++i) {
                if (!allHulls && (m_VsmNearCPU[i] & 6) == 0) continue;   // bit1 = near hull, bit2 = far hull (mode 2)
                const TreeVoxLod* lods = &m_VoxLodInfoCPU[(size_t)i * kHullLods];
                if (lods[0].count == 0) continue;
                const float dist = (std::max)(camPos.distance_to(m_MetaCPU[i].sphere_P), 1.f);
                // CONTINUOUS voxel size: the rendered cube edge grows smoothly with every
                // meter (no size steps at all; near trees clamp to the finest baked cell,
                // the far end keeps growing unbounded — huge cubes for far crowns). The
                // baked grids only supply POSITIONS/density: use the coarsest level whose
                // cell fits under the current cube size (cubes span 1–2 grid cells → no
                // gaps), so a grid swap changes positions but never the cube size — and
                // the dither crossfade hides the position swap itself.
                const float pxT = (pxBase + pxFar * dist * 0.01f) * biasMul;   // target px at this dist
                float T = dist * pxAt1m * pxT;                                 // target cube edge (m)
                // Cap the growth at ~the crown extent (coarsest cell ≈ ext/4): the far end
                // converges to one crown-sized block instead of a comically oversized cube.
                const float sLast = lods[kHullLods - 1].count ? lods[kHullLods - 1].size : lods[0].size;
                T = (std::min)(T, 4.f * sLast);
                int lvl = 0;
                while (lvl + 1 < (int)kHullLods && lods[lvl + 1].size <= T && lods[lvl + 1].count) ++lvl;
                const float cubeSize = (std::max)(T, lods[0].size);
                // DITHERED CROSSFADE to the next-coarser grid, computed in SIZE space
                // (T vs the next cell S) so it stays exact for any px(dist) curve: the
                // band ENDS at T == S — fade hits 1.0 the moment the grids swap.
                float f = 0.f;
                if (fadeBand > 0.001f && lvl + 1 < (int)kHullLods && lods[lvl + 1].count) {
                    const float S = lods[lvl + 1].size;
                    f = (T - S * (1.f - fadeBand)) / (S * fadeBand);
                    f = (std::min)(1.f, (std::max)(0.f, f));
                }
                if (f > 0.004f) {
                    drawLod(lods[lvl],     i,  f, cubeSize);   // fine grid dissolves OUT
                    drawLod(lods[lvl + 1], i, -f, cubeSize);   // coarse grid dissolves IN (same cube size!)
                    ++fadePairs; instTotal += (u64)lods[lvl].count + lods[lvl + 1].count;
                } else {
                    drawLod(lods[lvl], i, 0.f, cubeSize);
                    instTotal += lods[lvl].count;
                }
                ++treesDrawn; ++lvlDraws[lvl];
                if (doLog) {
                    int band = 0; float edge = 16.f;
                    while (band < 8 && dist > edge) { edge *= 2.f; ++band; }
                    if (!probes[band].used) {
                        const float pr = lods[lvl].size / cubeSize;
                        const u32 effN = (u32)((float)lods[lvl].count * (std::min)(1.f, pr * pr) + 0.5f);
                        probes[band] = VoxProbe{ dist, T, cubeSize, f, lvl, effN, true };
                    }
                }
            }
            if (doLog && treesDrawn) {
                Msg("[VK VoxLOD] fov=%.1f h=%u pxAt1m=%.6f px=%.1f+%.2f/100m biasx%.2f fade=%.2f | trees=%u inst=%u eff=%u fading=%u | lvl: %u/%u/%u/%u/%u | SHADOW voxcast=%u band=%u slice=%u",
                    Device.fFOV, Device.dwHeight, pxAt1m, pxBase, pxFar, biasMul, fadeBand,
                    treesDrawn, (u32)instTotal, (u32)instEff, fadePairs,
                    lvlDraws[0], lvlDraws[1], lvlDraws[2], lvlDraws[3], lvlDraws[4],
                    m_VoxCastCount, m_VoxBandCount, m_VoxCastSlice);
                for (int b = 0; b < 9; ++b)
                    if (probes[b].used)
                        Msg("[VK VoxLOD]   probe d=%6.1fm T=%.3f cube=%.3fm (%.1fpx) lvl=%d fade=%.2f n=%u",
                            probes[b].dist, probes[b].T, probes[b].size,
                            probes[b].size / (std::max)(probes[b].dist * pxAt1m, 1e-6f),
                            probes[b].lvl, probes[b].fade, probes[b].cnt);
            }
        }
        else if (m_HullDebugPipe != VK_NULL_HANDLE && m_HullVB && m_HullIB &&
                 m_HullInfoCPU.size() == m_TotalCount)
        {
            // Fallback (r_vsm_tree_hull_vox 0 → no cloud baked): shaded hull/lobe overlay.
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_HullDebugPipe);
            VkDeviceSize vbOff = 0;
            VkBuffer hvb = m_HullVB->GetHandle();
            vkCmdBindVertexBuffers(cmd, 0, 1, &hvb, &vbOff);
            vkCmdBindIndexBuffer(cmd, m_HullIB->GetHandle(), 0, VK_INDEX_TYPE_UINT16);
            for (u32 i = 0; i < m_TotalCount; ++i) {
                if (!allHulls && (m_VsmNearCPU[i] & 6) == 0) continue;
                const GpuTreeHullInfo& h = m_HullInfoCPU[i];
                if (h.ib_count == 0) continue;
                vkCmdDrawIndexed(cmd, h.ib_count, 1, h.ib_first, (s32)h.vb_first, i);
            }
        }
    }

    vkCmdEndRendering(cmd);

    static bool s_diag = false;
    if (!s_diag) {
        Msg("[VK Trees] First Render: %u trees in %u groups (maxGroupMesh=%u)",
            m_TotalCount, numGroups, m_MaxGroupMeshCount);
        s_diag = true;
    }
}

// ============================================================================
// Wind-sway motion-vector overlay (tree_motion.{vert,frag}).
// ============================================================================
// Own pipeline layout: set0 = xform SSBO + s_waves (as forward), set1 = diffuse
// (alpha test), NO env set. MV color (RG16F), depth LEQUAL WITHOUT write, and NO
// depth bias — same as the forward tree draw (which uses no bias) so the cur pose's
// clip depth bit-matches → LEQUAL passes over the tree depth already in the buffer.
void CTreeManager::CreateMotionPipelines()
{
    if (m_MotionPipeline24 != VK_NULL_HANDLE || m_MotionPipeline28 != VK_NULL_HANDLE) return;
    if (!g_ShaderManager) return;
    if (m_XformDescLayout == VK_NULL_HANDLE || m_TexDescLayout == VK_NULL_HANDLE) return;

    VkShaderModule vs = g_ShaderManager->Load("tree_motion.vert.spv");
    VkShaderModule fs = g_ShaderManager->Load("tree_motion.frag.spv");
    if (vs == VK_NULL_HANDLE || fs == VK_NULL_HANDLE) {
        Msg("![VK Trees] tree_motion.{vert,frag}.spv load failed — tree MV disabled");
        return;
    }

    VkDescriptorSetLayout setLayouts[2] = { m_XformDescLayout, m_TexDescLayout };
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcr.offset = 0; pcr.size = sizeof(TreeMotionPush);
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 2; plci.pSetLayouts = setLayouts;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &m_MotionPipelineLayout) != VK_SUCCESS) {
        Msg("![VK Trees] MV pipeline layout create failed"); return;
    }

    auto createVariant = [&](u32 tcOffset, VkPipeline& out)
    {
        out = VK::GfxPipelineBuilder(m_MotionPipelineLayout)
            .Vert(vs).Frag(fs)
            .Binding(0, 32)
            .Attr(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0)
            .Attr(1, 0, VK_FORMAT_R16G16_SSCALED,   tcOffset)
            .Cull(VK_CULL_MODE_NONE)             // double-sided leaves (matches forward)
            .Depth(true, false)                  // scene depth owns the surface
            .Color(VK::MotionVec::Format(), VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT)   // RG16F motion
            .DepthTarget(Swapchain.m_DepthFormat)
            .Build("Trees MV (tcOff=%u)", tcOffset);
    };
    createVariant(24, m_MotionPipeline24);
    createVariant(28, m_MotionPipeline28);
    Msg("[VK Trees] MV overlay pipelines OK (wind-sway motion vectors)");
}

void CTreeManager::DestroyMotionPipelines()
{
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
    VkDevice dev = VulkanHW.m_Device;
    if (m_MotionPipeline24)     { vkDestroyPipeline(dev, m_MotionPipeline24, nullptr); m_MotionPipeline24 = VK_NULL_HANDLE; }
    if (m_MotionPipeline28)     { vkDestroyPipeline(dev, m_MotionPipeline28, nullptr); m_MotionPipeline28 = VK_NULL_HANDLE; }
    if (m_MotionPipelineLayout) { vkDestroyPipelineLayout(dev, m_MotionPipelineLayout, nullptr); m_MotionPipelineLayout = VK_NULL_HANDLE; }
}

// Re-draws THIS frame's visible trees (same indirect buffers the forward pass
// filled) into the MV target, reprojecting the cur AND prev wind pose. Runs INSIDE
// MotionVec::ExecuteDynamic's render pass (MV target + scene depth bound, viewport
// set). Same VP + wind + no bias → cur depth bit-matches → LEQUAL passes. No cull
// re-dispatch: the forward Render already filled the indirect/count buffers.
void CTreeManager::RenderMotion(const VK::FrameContext& ctx, const Fmatrix& curVP, const Fmatrix& prevVP,
                                float jitterNdcX, float jitterNdcY)
{
    if (!m_MvWindValid) return;                     // trees didn't render this frame
    if (ctx.cmd == VK_NULL_HANDLE) return;
    if (m_TreeIndirectBuffer == nullptr || m_TreeDrawCountBuffer == nullptr) return;

    CreateMotionPipelines();                        // lazy — needs xform + tex layouts (built by Build)
    if (m_MotionPipeline24 == VK_NULL_HANDLE && m_MotionPipeline28 == VK_NULL_HANDLE) return;

    const VkCommandBuffer cmd = ctx.cmd;
    const u32 numGroups = (u32)m_Groups.size();
    if (numGroups == 0) return;

    TreeMotionPush pc{};
    pc.curVP             = curVP;
    pc.prevVP            = prevVP;
    pc.uvScale           = 1.0f / 2048.0f;          // matches the forward tree pass
    pc.alphaRef          = 200.0f / 255.0f;
    pc.jitterX           = jitterNdcX;              // re-applied to gl_Position (jitter-free MV)
    pc.jitterY           = jitterNdcY;
    pc.wind_params       = m_MvWindParams;
    pc.wsetup_trees      = m_MvWsetup;
    pc.wind_anim         = m_MvWindAnim;
    pc.wind_params_prev  = m_MvWindParamsPrev;
    pc.wsetup_trees_prev = m_MvWsetupPrev;
    pc.wind_anim_prev    = m_MvWindAnimPrev;
    vkCmdPushConstants(cmd, m_MotionPipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(pc), &pc);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_MotionPipelineLayout,
                            0, 1, &m_XformDescSet, 0, nullptr);

    VkPipeline      lastPipe = VK_NULL_HANDLE;
    VkDescriptorSet lastTex  = VK_NULL_HANDLE;
    for (u32 g = 0; g < numGroups; ++g)
    {
        const TreeIndirectGroup& grp = m_Groups[g];
        VkPipeline pipe = (grp.tcOffset == 24) ? m_MotionPipeline24 : m_MotionPipeline28;
        if (pipe == VK_NULL_HANDLE) continue;
        if (pipe != lastPipe) { vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe); lastPipe = pipe; }

        if (grp.descSetIdx < m_TexDescSets.size()) {
            VkDescriptorSet tex = m_TexDescSets[grp.descSetIdx];
            if (tex != lastTex) {
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_MotionPipelineLayout,
                                        1, 1, &tex, 0, nullptr);
                lastTex = tex;
            }
        }

        VkDeviceSize vbOff = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &grp.vb, &vbOff);
        vkCmdBindIndexBuffer(cmd, grp.ib, 0, VK_INDEX_TYPE_UINT16);

        const VkDeviceSize cmdOff = (VkDeviceSize)g * m_MaxGroupMeshCount * sizeof(VkDrawIndexedIndirectCommand);
        const VkDeviceSize cntOff = (VkDeviceSize)g * sizeof(u32);
        vkCmdDrawIndexedIndirectCount(cmd,
            m_TreeIndirectBuffer->GetHandle(), cmdOff,
            m_TreeDrawCountBuffer->GetHandle(), cntOff,
            m_MaxGroupMeshCount, sizeof(VkDrawIndexedIndirectCommand));
    }

    static bool s_diagMv = false;
    if (!s_diagMv) { s_diagMv = true; Msg("[VK Trees] first wind-sway MV render: groups=%u", numGroups); }
}

// ============================================================================
// Session B teardown — called from Destroy() before the buffers go.
// ============================================================================
void CTreeManager::DestroySessionB()
{
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
    VkDevice dev = VulkanHW.m_Device;

    DestroyMotionPipelines();   // MV overlay — built on the xform/tex layouts, free first
    if (m_GfxPipeline24)     { vkDestroyPipeline(dev, m_GfxPipeline24, nullptr); m_GfxPipeline24 = VK_NULL_HANDLE; }
    if (m_GfxPipeline28)     { vkDestroyPipeline(dev, m_GfxPipeline28, nullptr); m_GfxPipeline28 = VK_NULL_HANDLE; }
    if (m_DepthPipeline24)   { vkDestroyPipeline(dev, m_DepthPipeline24, nullptr); m_DepthPipeline24 = VK_NULL_HANDLE; }
    if (m_DepthPipeline28)   { vkDestroyPipeline(dev, m_DepthPipeline28, nullptr); m_DepthPipeline28 = VK_NULL_HANDLE; }
    m_MetaCPU.clear();
    if (m_GfxPipelineLayout) { vkDestroyPipelineLayout(dev, m_GfxPipelineLayout, nullptr); m_GfxPipelineLayout = VK_NULL_HANDLE; }

    if (m_XformDescPool)   { vkDestroyDescriptorPool(dev, m_XformDescPool, nullptr); m_XformDescPool = VK_NULL_HANDLE; m_XformDescSet = VK_NULL_HANDLE; }
    if (m_XformDescLayout) { vkDestroyDescriptorSetLayout(dev, m_XformDescLayout, nullptr); m_XformDescLayout = VK_NULL_HANDLE; }
    if (m_WaveTex)     { m_WaveTex->Destroy(); xr_delete(m_WaveTex); m_WaveTex = nullptr; }
    if (m_WaveSampler) { vkDestroySampler(dev, m_WaveSampler, nullptr); m_WaveSampler = VK_NULL_HANDLE; }

    if (m_CullPipeline)       { vkDestroyPipeline(dev, m_CullPipeline, nullptr); m_CullPipeline = VK_NULL_HANDLE; }
    if (m_CullPipelineLayout) { vkDestroyPipelineLayout(dev, m_CullPipelineLayout, nullptr); m_CullPipelineLayout = VK_NULL_HANDLE; }
    if (m_CullDescPool)       { vkDestroyDescriptorPool(dev, m_CullDescPool, nullptr); m_CullDescPool = VK_NULL_HANDLE; m_CullDescSet = VK_NULL_HANDLE; }
    if (m_CullDescLayout)     { vkDestroyDescriptorSetLayout(dev, m_CullDescLayout, nullptr); m_CullDescLayout = VK_NULL_HANDLE; }

    if (m_FrustumUBO) { m_FrustumUBO->Destroy(); xr_delete(m_FrustumUBO); }

    DestroyVsm();
}

// ============================================================================
// VSM caster path — trees into the virtual shadow atlas (static → temporal-safe).
// Mirrors RenderDepth, but each tree is instanced over the atlas pages it overlaps
// (binned by vsm_skinned_bin.comp — GpuTreeMeta shares SkinMeta's 32 B layout) and
// rasterized with tree_vsm_page.{vert,frag} which routes per page + alpha-tests leaves.
// ============================================================================
// Max atlas pages a tree bins into. 512 was enough for the STATIC path (dirty-filtered:
// a tree touches only a handful of pages per frame) but the DYNAMIC wind pass bins a NEAR
// tree into ALL its resident pages every frame — a 10m crown a few metres away covers
// hundreds of L0 pages alone (L0 page = base/32 ≈ 0.75 m) and overflowed 512 → truncated
// page lists → near tree shadows visibly fell apart / vanished. 1024 covers it (watch
// "maxPages" in the r_vsm_debug hybrid log; the dyn pool ceiling is VSM_MAX_PHYS=2048).
// 512 (was 1024): the buffer is m_TotalCount × cap × 4B — 792 MB on a 193k-tree map, the
// single biggest VRAM line item, paid for every tree while only near crowns bin many pages
// (observed peak 349). The bin stat line prints [TRUNCATED!] if a caster ever hits the cap.
// casterPages is a shared ARENA of (treeIdx<<13)|slot entries (vsm_tree_bin two-pass:
// count → atomic reserve → write; firstInstance = the tree's run). The old per-tree
// 512-slot slicing cost trees × cap × 4 B = 396 MB on a 193k-tree map — the single
// biggest VRAM line item, big enough to overcommit VRAM and demote tree buffers to
// host memory (25-40 ms Bins/Tree* on Pripyat). A frame really bins ~10-40k pairs
// (near crowns × pages + dirty-page far trees); 1M slots = 4 MB with 25× headroom;
// the [ARENA FULL!] diag in the hybrid log fires if a map ever saturates it.
// kVsmTreeCap remains ONLY as the legacy push value for VS layouts (unused there now).
namespace { constexpr u32 kVsmTreeCap = 512; constexpr u32 kVsmTreeArena = 1u << 20; }

// Impostor crown shadows (r_vsm_tree_impostor): baked silhouette atlas grid = kImpFacets
// azimuth columns × (unique mesh) rows, kImpCell px each. See BuildImpostorAtlas.
namespace { constexpr u32 kImpFacets = 8; constexpr u32 kImpCell = 128; }

// Phase B: meshlet-cull command budget. Commands are POOLED per group; cap[g] =
// max(min(meshCount, kMeshletNearTreesCap)*kCmdPerTree, kGroupFloor). The per-tree term
// scales big species groups; the FLOOR covers small groups of a few large near trees (a
// single near crown bins ~maxPages×fewMeshlets ≈ up to ~430 draws — a 2-tree rare-species
// group would starve on meshCount*k alone → that was the steady-state [TRUNCATED] in the
// first test). Bases are a prefix sum of cap[] (not meshOffset*k). Both static + dynamic
// buffers share this layout.
// kMeshletNearTreesCap: only a fraction of a group's trees emit meshlet commands at once
// (near set + the static-cache warm-up burst), yet cap[g] used to reserve slots for EVERY
// tree of the group — 2×495 MB of VRAM on a 193k-tree map, which alone shoved textures
// into mip-demotion ("всё мыло"). 2048 covers the static-bake burst with margin (128 was
// too tight: overflow=24k [TRUNCATED] during warm-up); overruns show up as "overflow" in
// the r_vsm_debug meshlet stat line and truncate gracefully — bump if it ever ticks.
namespace { constexpr u32 kCmdPerTree = 128; constexpr u32 kGroupFloor = 8192; constexpr u32 kMeshletNearTreesCap = 2048; }

void CTreeManager::CreateVsmResources()
{
    VK::Vram::Scope _vram_scope("Trees");
    if (m_VsmReady) return;
    if (m_XformDescLayout == VK_NULL_HANDLE || m_TexDescLayout == VK_NULL_HANDLE) return;   // Session B not up
    if (m_XformDescSet == VK_NULL_HANDLE) return;
    if (!m_TreeMetadataBuffer || !m_TreeTransformsBuffer || m_TotalCount == 0) return;
    if (!g_ShaderManager) return;

    VkShaderModule binCS = g_ShaderManager->Load("vsm_tree_bin.comp.spv");   // near/far hybrid bin (mode 0 static / 1 dynamic)
    VkShaderModule pvs   = g_ShaderManager->Load("tree_vsm_page.vert.spv");        // STATIC atlas (rigid)
    VkShaderModule pvsD  = g_ShaderManager->Load("tree_vsm_page_dyn.vert.spv");    // DYNAMIC atlas (wind)
    VkShaderModule pfs   = g_ShaderManager->Load("tree_vsm_page.frag.spv");
    if (!binCS || !pvs || !pvsD || !pfs) { Msg("![VK Trees] VSM caster shaders missing — tree VSM shadows disabled"); return; }

    VkDevice dev = VulkanHW.m_Device;
    const u32 N = VK_FRAMES_IN_FLIGHT;

    // Buffers.
    // TRANSFER_DST everywhere below: these are GPU-written append/command buffers, and
    // the ONE-TIME zero-fill in VsmBin's lazy-init branch is load-bearing — a recreated
    // TreeManager (level transition) gets RECYCLED VRAM full of garbage, and a garbage
    // VkDrawIndexedIndirectCommand is a billion-instance draw = GPU hang / DEVICE_LOST.
    // (A fresh process gets OS-zeroed pages, which is why only transitions crashed.)
    // gpuOnly=true everywhere below: these are read/written by EVERY thread of the
    // 193k-thread bin dispatches — the default storage-buffer host-access flag would
    // put them in BAR/system memory (PCIe per warp = the 20-30ms Bins/Tree* zones).
    m_VsmCasterPages = xr_new<CVulkanBuffer>();
    m_VsmCasterPages->Create((VkDeviceSize)kVsmTreeArena * sizeof(u32),
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    m_VsmIndirect = xr_new<CVulkanBuffer>();
    m_VsmIndirect->Create((VkDeviceSize)m_TotalCount * sizeof(VkDrawIndexedIndirectCommand),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    m_VsmDynIndirect = xr_new<CVulkanBuffer>();
    m_VsmDynIndirect->Create((VkDeviceSize)m_TotalCount * sizeof(VkDrawIndexedIndirectCommand),
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    m_VsmStats = xr_new<CVulkanBuffer>();
    m_VsmStats->Create(8 * sizeof(u32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);   // [4]=shadow-HZB culled
    m_VsmStatsRB = xr_new<CVulkanBuffer>();
    m_VsmStatsRB->Create(8 * sizeof(u32), VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
    m_VsmStatsPtr = (u32*)m_VsmStatsRB->Map();
    if (m_VsmStatsPtr) memset(m_VsmStatsPtr, 0, 8 * sizeof(u32));
    // Diag bitsets (binding 12): [0..W) = distinct trees binned to the DYN atlas this
    // frame (cleared per frame in VsmBinClears), [W..2W) = trees that ever entered the
    // cached STATIC atlas (accumulated — static bins touch dirty pages only).
    {
        const u32 W = (m_TotalCount + 31u) / 32u;
        m_VsmTreeBits = xr_new<CVulkanBuffer>();
        m_VsmTreeBits->Create((VkDeviceSize)2 * W * sizeof(u32),
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                              VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
        m_VsmTreeBitsRB = xr_new<CVulkanBuffer>();
        m_VsmTreeBitsRB->Create((VkDeviceSize)2 * W * sizeof(u32), VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
        m_VsmTreeBitsPtr = (u32*)m_VsmTreeBitsRB->Map();
        if (m_VsmTreeBitsPtr) memset(m_VsmTreeBitsPtr, 0, (size_t)2 * W * sizeof(u32));
    }
    // Per-frame CPU near-set flags (ring). DEVICE-local + staged upload — NOT host-visible:
    // the bin dispatches read nearFlags[c] from EVERY one of the 193k threads, and a
    // host-memory buffer turned that into per-warp PCIe round-trips = the 17-30 ms
    // Bins/TreeS + Bins/TreeD zones on Pripyat (invisible on few-tree maps). The CPU
    // side uploads m_VsmNearCPU via the async transfer queue instead (774 KB/frame).
    {
        xr_vector<u32> zero(m_TotalCount, 0u);
        for (u32 i = 0; i < N; ++i) {
            m_VsmNearFlags[i] = xr_new<CVulkanBuffer>();
            m_VsmNearFlags[i]->Create((VkDeviceSize)m_TotalCount * sizeof(u32),
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                      VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
            m_VsmNearFlags[i]->Upload(zero.data(), (VkDeviceSize)m_TotalCount * sizeof(u32));   // all-far until the first near-set update
            m_VsmNearPtr[i] = nullptr;
        }
    }
    // One-shot placement audit: any of these landing outside DEVICE_LOCAL means the
    // bins ride PCIe again — the log line is the tripwire.
    {
        auto place = [](const char* n, CVulkanBuffer* b) {
            if (!b || !b->GetAllocation()) return;
            VkMemoryPropertyFlags f = 0;
            vmaGetAllocationMemoryProperties(VulkanHW.m_Allocator, b->GetAllocation(), &f);
            Msg("[VK Trees] vsm buf %-12s: %s%s", n,
                (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? "DEVICE" : "HOST(!)",
                (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ? "+host-visible" : "");
        };
        place("casterPages", m_VsmCasterPages);
        place("indirect",    m_VsmIndirect);
        place("dynIndirect", m_VsmDynIndirect);
        place("stats",       m_VsmStats);
        place("nearFlags0",  m_VsmNearFlags[0]);
        place("meta",        m_TreeMetadataBuffer);
        place("xforms",      m_TreeTransformsBuffer);
    }

    // Pool: bin (12 SSBO + 1 UBO) ×2N (static + dynamic) + page (2 SSBO + 1 UBO) ×2N.
    VkDescriptorPoolSize ps[2] = { { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, N * 32 }, { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, N * 4 } };
    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets = N * 4; pci.poolSizeCount = 2; pci.pPoolSizes = ps;
    if (vkCreateDescriptorPool(dev, &pci, nullptr, &m_VsmDescPool) != VK_SUCCESS) return;

    // Bin set layout (13 bindings: +7 dynUsed, +8 nearFlags, +9 pageMax, +10 staticPageTable=Option A,
    // +11 receiver mask r_vsm_rmask, +12 diag tree bitsets) + N static + N dyn sets.
    {
        m_VsmBinSetL = VK::MakeSetLayout({ kSSBO, kUBO,  kSSBO, kSSBO, kSSBO, kSSBO, kSSBO,
                                           kSSBO, kSSBO, kSSBO, kSSBO, kSSBO, kSSBO },
                                         VK_SHADER_STAGE_COMPUTE_BIT, "Trees.VsmBin");
        if (m_VsmBinSetL == VK_NULL_HANDLE) return;
        if (!VK::AllocSets(m_VsmDescPool, m_VsmBinSetL, N, m_VsmBinSet,    "Trees.VsmBin.static")) return;
        if (!VK::AllocSets(m_VsmDescPool, m_VsmBinSetL, N, m_VsmDynBinSet, "Trees.VsmBin.dyn")) return;
        // push: count, cap, mode, hzbOn, margin(float), rmaskOn, bitBase
        m_VsmBinLayout = VK::MakePipelineLayout({ m_VsmBinSetL }, 7 * sizeof(u32));
        if (m_VsmBinLayout == VK_NULL_HANDLE) return;
        m_VsmBinPipe = VK::CreateComputePipeline(binCS, m_VsmBinLayout, "Trees.VsmBin");
        if (m_VsmBinPipe == VK_NULL_HANDLE) return;
    }

    // Page set layout (3 bindings: pageList, casterPages, clipmap UBO) + N static + N dynamic sets.
    {
        m_VsmPageSetL = VK::MakeSetLayout({ kSSBO, kSSBO, kUBO },
                                          VK_SHADER_STAGE_VERTEX_BIT, "Trees.VsmPage");
        if (m_VsmPageSetL == VK_NULL_HANDLE) return;
        if (!VK::AllocSets(m_VsmDescPool, m_VsmPageSetL, N, m_VsmPageSet,    "Trees.VsmPage.static")) return;
        if (!VK::AllocSets(m_VsmDescPool, m_VsmPageSetL, N, m_VsmDynPageSet, "Trees.VsmPage.dyn")) return;
    }

    // Page pipeline layout: set0 = transforms (reuse), set1 = diffuse (reuse), set2 = page data.
    // Push: 16 B base (uvScale, alphaRef, cap, pad) + 48 B TEST wind (wind_params,
    // wsetup_trees, wind_anim) for r_vsm_tree_wind. See tree_vsm_page.vert.
    m_VsmPageLayout = VK::MakePipelineLayout({ m_XformDescLayout, m_TexDescLayout, m_VsmPageSetL }, 64,
                                             VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
    if (m_VsmPageLayout == VK_NULL_HANDLE) return;

    // Page pipelines (tcOffset 24 / 28 × static/dynamic VS) — depth-only into the D16 atlas, alpha-test FS.
    auto createPageVariant = [&](VkShaderModule vs, u32 tcOffset, VkPipeline& out, bool wantFsr) {
        VK::GfxPipelineBuilder b(m_VsmPageLayout);
        b.Vert(vs).Frag(pfs)
         .Binding(0, 32)
         .Attr(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0)
         .Attr(1, 0, VK_FORMAT_R16G16_SSCALED,   tcOffset)
         .Cull(VK_CULL_MODE_NONE)
         .DynamicDepthBias()
         .Depth(true, true)
         .DepthTarget(VK_FORMAT_D16_UNORM);   // VSM atlas is D16 (see vk_vsm CreateRenderResources)
        // Dynamic crown pipelines add a FRAGMENT_SHADING_RATE dynamic state so VsmRenderDyn
        // can 2x2-coarse the crown-shadow fill at runtime (r_vsm_tree_vrs). Static pipelines
        // never set a rate.
        if (wantFsr && VulkanHW.m_bVRSPipelineSupported)
            b.Dynamic(VK_DYNAMIC_STATE_FRAGMENT_SHADING_RATE_KHR);
        out = b.Build("Trees VSM page (tcOff=%u)", tcOffset);
    };
    createPageVariant(pvs,  24, m_VsmPagePipe24,    false);
    createPageVariant(pvs,  28, m_VsmPagePipe28,    false);
    createPageVariant(pvsD, 24, m_VsmPageDynPipe24, true);   // dyn crowns → VRS-capable
    createPageVariant(pvsD, 28, m_VsmPageDynPipe28, true);

    m_VsmReady = true;
    Msg("[VK Trees] VSM caster path ready (%u trees, arena %u pairs, wind hybrid %s dist %.0fm)",
        m_TotalCount, kVsmTreeArena, ps_r_vsm_tree_wind ? "ON" : "off", ps_r_vsm_tree_wind_dist);
    if (m_TotalCount > 0x40000u)   // (treeIdx<<13) pack = 18-bit tree ids; trees past 262144 never bin (warn, don't corrupt)
        Msg("![VK Trees] %u trees exceed the 18-bit caster-pack ceiling (262144) - the excess cast no VSM shadow", m_TotalCount);

    CreateImpostorResources();     // r_vsm_tree_impostor: billboard crown shadows (near dyn pass)
    // Phase B meshlet-cull (r_vsm_meshlet): resources are LAZY — the two pooled command
    // buffers alone are ~235 MB ×2 on a 193k-tree map, dead VRAM while the mode is off
    // (its own header says "no net perf gain on dGPU"). A live 0→1 flip creates them in
    // VsmBinMeshlets (with an immediate zero — recycled-VRAM garbage in an indirect
    // command buffer is the billion-instance-draw GPU hang of 2026-07-09).
    if (ps_r_vsm_meshlet) CreateMeshletVsmResources();
    CreateHullResources();         // r_vsm_tree_hull: opaque crown-hull caster LOD (near dyn pass)
}

bool CTreeManager::IsMeshletMode() const
{
    return ps_r_vsm_meshlet && m_MeshletVsmReady && m_MeshletsReady;
}

// ============================================================================
// Phase B — stage-2 meshlet-cull resources: the meshlet-bin compute pipeline, the
// meshlet page pipelines (reuse m_VsmPageLayout / set2), and the command/group buffers.
// Best-effort: any failure leaves m_MeshletVsmReady=false → the old per-tree path runs.
// ⛔ NO NET PERF GAIN on dGPU (2026-07-02) — kept but DISABLED (r_vsm_meshlet=0). Reached only when
//    the cvar is on; inert by default. Don't re-chase — see vk_console_min.cpp cvar block + memory.
// ============================================================================
void CTreeManager::CreateMeshletVsmResources()
{
    VK::Vram::Scope _vram_scope("Trees");
    if (m_MeshletVsmReady) return;
    if (!m_MeshletsReady || !m_MeshletBuffer || !m_MeshletIndexBuffer || !m_TreeMeshletRangeBuffer) return;
    if (m_VsmPageLayout == VK_NULL_HANDLE || m_TotalCount == 0) return;
    if (!g_ShaderManager) return;

    VkDevice   dev = VulkanHW.m_Device;
    const u32  N   = VK_FRAMES_IN_FLIGHT;

    VkShaderModule binCS = g_ShaderManager->Load("vsm_tree_meshlet_bin.comp.spv");
    VkShaderModule pvs   = g_ShaderManager->Load("tree_vsm_meshlet_page.vert.spv");
    VkShaderModule pvsD  = g_ShaderManager->Load("tree_vsm_meshlet_page_dyn.vert.spv");
    VkShaderModule pfs   = g_ShaderManager->Load("tree_vsm_page.frag.spv");   // reuse the alpha-test FS
    if (!binCS || !pvs || !pvsD || !pfs) { Msg("![VK Trees] meshlet shaders missing — r_vsm_meshlet disabled"); return; }

    // ----- CPU: per-tree group index + per-group (cmdBase, cmdCap). Groups are contiguous
    // tree ranges, so the pooled section [meshOffset*kCmdPerTree, +meshCount*kCmdPerTree) is
    // contiguous and renderable per group with one DrawIndexedIndirectCount.
    const u32 groupN = (u32)m_Groups.size();
    xr_vector<u32>  treeGroup(m_TotalCount, 0u);
    xr_vector<u32>  groupInfo(groupN * 2u, 0u);   // uvec2 (cmdBase, cmdCap) per group
    m_MeshletGroupBase.assign(groupN, 0u);
    m_MeshletGroupCap.assign(groupN, 0u);
    u32 cursor = 0;   // running command-slot cursor -> per-group base (prefix sum of caps)
    for (u32 g = 0; g < groupN; ++g) {
        const TreeIndirectGroup& grp = m_Groups[g];
        u32 cap = std::min(grp.meshCount, kMeshletNearTreesCap) * kCmdPerTree;
        if (cap < kGroupFloor) cap = kGroupFloor;
        groupInfo[g * 2 + 0] = cursor;
        groupInfo[g * 2 + 1] = cap;
        m_MeshletGroupBase[g] = cursor;
        m_MeshletGroupCap[g]  = cap;
        cursor += cap;
        for (u32 i = 0; i < grp.meshCount; ++i)
            if (grp.meshOffset + i < m_TotalCount) treeGroup[grp.meshOffset + i] = g;
    }
    const VkDeviceSize cmdBytes = (VkDeviceSize)cursor * sizeof(VkDrawIndexedIndirectCommand);

    auto mkBuf = [&](CVulkanBuffer*& b, VkDeviceSize sz, VkBufferUsageFlags usage, VmaMemoryUsage mem) {
        b = xr_new<CVulkanBuffer>(); b->Create(sz, usage, mem, true);   // gpuOnly: GPU-written command/count buffers, no CPU mapping
    };
    // TRANSFER_DST: zero-filled once in VsmBin's lazy-init (recycled-VRAM garbage command
    // = billion-instance indirect draw = the level-transition GPU hang; see CreateVsmResources).
    mkBuf(m_VsmMeshletCmd,        cmdBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    mkBuf(m_VsmMeshletCmdDyn,     cmdBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    mkBuf(m_VsmMeshletGroupCount,    (VkDeviceSize)groupN * sizeof(u32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    mkBuf(m_VsmMeshletGroupCountDyn, (VkDeviceSize)groupN * sizeof(u32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    mkBuf(m_TreeGroupBuffer, (VkDeviceSize)m_TotalCount * sizeof(u32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    mkBuf(m_GroupInfoBuffer, (VkDeviceSize)groupN * 2 * sizeof(u32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    m_TreeGroupBuffer->Upload(treeGroup.data(), treeGroup.size() * sizeof(u32));
    m_GroupInfoBuffer->Upload(groupInfo.data(), groupInfo.size() * sizeof(u32));
    mkBuf(m_MeshletStats, 4 * sizeof(u32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    mkBuf(m_MeshletStatsRB, 4 * sizeof(u32), VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
    m_MeshletStatsPtr = (u32*)m_MeshletStatsRB->Map();
    if (m_MeshletStatsPtr) memset(m_MeshletStatsPtr, 0, 4 * sizeof(u32));

    // ----- Descriptor pool + layout (13 bindings, all SSBO except binding 1 = clipmap UBO).
    VkDescriptorPoolSize ps[2] = { { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, N * 2 * 12 }, { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, N * 2 } };
    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets = N * 2; pci.poolSizeCount = 2; pci.pPoolSizes = ps;
    if (vkCreateDescriptorPool(dev, &pci, nullptr, &m_MeshletDescPool) != VK_SUCCESS) return;

    {
        m_MeshletBinSetL = VK::MakeSetLayout({ kSSBO, kUBO,  kSSBO, kSSBO, kSSBO, kSSBO, kSSBO,   // 1 = VsmParams
                                               kSSBO, kSSBO, kSSBO, kSSBO, kSSBO, kSSBO },
                                             VK_SHADER_STAGE_COMPUTE_BIT, "Trees.MeshletBin");
        if (m_MeshletBinSetL == VK_NULL_HANDLE) return;
        if (!VK::AllocSets(m_MeshletDescPool, m_MeshletBinSetL, N, m_MeshletBinSet,    "Trees.MeshletBin.static")) return;
        if (!VK::AllocSets(m_MeshletDescPool, m_MeshletBinSetL, N, m_MeshletDynBinSet, "Trees.MeshletBin.dyn")) return;
        // push: casterCount, cap, slop(float), pad
        m_MeshletBinLayout = VK::MakePipelineLayout({ m_MeshletBinSetL }, 4 * sizeof(u32));
        if (m_MeshletBinLayout == VK_NULL_HANDLE) return;
        m_MeshletBinPipe = VK::CreateComputePipeline(binCS, m_MeshletBinLayout, "Trees.MeshletBin");
        if (m_MeshletBinPipe == VK_NULL_HANDLE) return;
    }

    // ----- Meshlet page pipelines (reuse m_VsmPageLayout: set0 xform, set1 tex, set2 page).
    auto createPageVariant = [&](VkShaderModule vs, u32 tcOffset, VkPipeline& out) {
        out = VK::GfxPipelineBuilder(m_VsmPageLayout)
            .Vert(vs).Frag(pfs)
            .Binding(0, 32)
            .Attr(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0)
            .Attr(1, 0, VK_FORMAT_R16G16_SSCALED,   tcOffset)
            .Cull(VK_CULL_MODE_NONE)
            .DynamicDepthBias()
            .Depth(true, true)
            .DepthTarget(VK_FORMAT_D16_UNORM)   // VSM atlas is D16 (see vk_vsm CreateRenderResources)
            .Build("Trees meshlet page (tcOff=%u)", tcOffset);
    };
    createPageVariant(pvs,  24, m_MeshletPagePipe24);
    createPageVariant(pvs,  28, m_MeshletPagePipe28);
    createPageVariant(pvsD, 24, m_MeshletPageDynPipe24);
    createPageVariant(pvsD, 28, m_MeshletPageDynPipe28);

    // Immediate one-off zero of the GPU-written buffers: on the LOAD path VsmBin's
    // lazy-init fill covers them, but a live r_vsm_meshlet 0→1 flip creates them
    // mid-session — recycled-VRAM garbage in an indirect command buffer would be a
    // billion-instance draw (the level-transition GPU hang of 2026-07-09).
    if (VkCommandBuffer zc = VulkanHW.BeginSingleTimeCommands()) {
        auto z = [&](CVulkanBuffer* b) { if (b) vkCmdFillBuffer(zc, b->GetHandle(), 0, VK_WHOLE_SIZE, 0u); };
        z(m_VsmMeshletCmd); z(m_VsmMeshletCmdDyn);
        z(m_VsmMeshletGroupCount); z(m_VsmMeshletGroupCountDyn); z(m_MeshletStats);
        VulkanHW.EndSingleTimeCommands(zc);
    }

    m_MeshletVsmReady = true;
    Msg("[VK Trees] Meshlet VSM path ready (%u clusters, %u cmd/tree, %u KB cmd x2)",
        m_MeshletTotal, kCmdPerTree, (u32)(cmdBytes / 1024));
}

// Stage 2: refine the per-tree page list into per-(meshlet,page) draws. mode 0 = static
// (m_VsmIndirect / m_VsmMeshletCmd), mode 1 = dynamic (m_VsmDynIndirect / m_VsmMeshletCmdDyn).
// pageList is that atlas's slot->page map. Runs right after the matching stage-1 dispatch.
void CTreeManager::DispatchMeshletBin(VkCommandBuffer cmd, u32 mode, VkBuffer pageList, VkBuffer clipmapUBO)
{
    if (!m_MeshletVsmReady) return;
    const u32 slot = m_VsmSlot;
    const bool  isDyn = (mode == 1u);
    VkDescriptorSet set   = isDyn ? m_MeshletDynBinSet[slot] : m_MeshletBinSet[slot];
    VkBuffer indirect     = isDyn ? m_VsmDynIndirect->GetHandle()          : m_VsmIndirect->GetHandle();
    VkBuffer outCmd       = isDyn ? m_VsmMeshletCmdDyn->GetHandle()        : m_VsmMeshletCmd->GetHandle();
    VkBuffer groupCnt     = isDyn ? m_VsmMeshletGroupCountDyn->GetHandle() : m_VsmMeshletGroupCount->GetHandle();

    // No fills/barriers here: the group append-counters + stats were cleared by
    // VsmBinClears in the frame-start fill block, and the caller (vk_vsm MarkPages)
    // issued the single stage-1→stage-2 barrier ordering the tree bins' casterPages/
    // indirect writes before this refinement's reads.
    VK::DescriptorWriter(set)
        .StorageBuffer(0,  m_TreeMetadataBuffer->GetHandle())      // meta
        .UniformBuffer(1,  clipmapUBO)                             // VsmParams
        .StorageBuffer(2,  m_TreeTransformsBuffer->GetHandle())    // xform
        .StorageBuffer(3,  m_MeshletBuffer->GetHandle())           // meshlets
        .StorageBuffer(4,  m_TreeMeshletRangeBuffer->GetHandle())  // tree range
        .StorageBuffer(5,  m_VsmCasterPages->GetHandle())          // caster pages (stage 1)
        .StorageBuffer(6,  pageList)                               // slot -> page
        .StorageBuffer(7,  indirect)                               // stage-1 indirect (pageCount)
        .StorageBuffer(8,  outCmd)                                 // out commands
        .StorageBuffer(9,  groupCnt)                               // group counter
        .StorageBuffer(10, m_TreeGroupBuffer->GetHandle())         // tree -> group
        .StorageBuffer(11, m_GroupInfoBuffer->GetHandle())         // group (base,cap)
        .StorageBuffer(12, m_MeshletStats->GetHandle())            // stats
        .Flush();

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_MeshletBinPipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_MeshletBinLayout, 0, 1, &set, 0, nullptr);
    const u32 castN = _min(m_TotalCount, 0x40000u);   // 18-bit caster-pack ceiling (matches the stage-1 bins)
    struct { u32 casterCount, cap; float slop; u32 pad; } push{ castN, kVsmTreeCap, isDyn ? 0.5f : 0.0f, 0u };
    vkCmdPushConstants(cmd, m_MeshletBinLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd, castN, 1, 1);   // one workgroup per tree
}

// Meshlet render for one atlas. Per group: bind meshlet VS + diffuse + group VB + the shared
// meshlet IB, then DrawIndexedIndirectCount over the group's pooled command section.
void CTreeManager::DrawMeshlets(VkCommandBuffer cmd, u32 mode, VkDescriptorSet pageSet, bool wind)
{
    const bool isDyn = (mode == 1u);
    VkPipeline p24 = isDyn ? m_MeshletPageDynPipe24 : m_MeshletPagePipe24;
    VkPipeline p28 = isDyn ? m_MeshletPageDynPipe28 : m_MeshletPagePipe28;
    if (p24 == VK_NULL_HANDLE && p28 == VK_NULL_HANDLE) return;
    VkBuffer cmdBuf   = isDyn ? m_VsmMeshletCmdDyn->GetHandle()        : m_VsmMeshletCmd->GetHandle();
    VkBuffer countBuf = isDyn ? m_VsmMeshletGroupCountDyn->GetHandle() : m_VsmMeshletGroupCount->GetHandle();

    struct VsmPagePush {
        float    uvScale, alphaRef; u32 cap, pad;
        Fvector4 wind_params, wsetup_trees, wind_anim;
    } pc{};
    pc.uvScale = 1.0f / 2048.0f; pc.alphaRef = 200.0f / 255.0f; pc.cap = kVsmTreeCap;
    if (wind) {
        pc.wsetup_trees.set(ps_r_wind_tree_anim, ps_r_wind_tree_trunk, ps_r_wind_tree_bend, 0.1f);
        if (g_pGamePersistent) {
            const Fvector3 wa = g_pGamePersistent->Environment().wind_anim;
            pc.wind_anim.set(wa.x, wa.y, wa.z, ps_r_wind_tree_flutter);
            if (auto* E = g_pGamePersistent->Environment().CurrentEnv)
                pc.wind_params.set(E->wind_direction, E->wind_velocity, 0.0f, 0.0f);
        }
        pc.wind_params.w = ps_r_wind_tree_crown;   // flutter height gate — must match the forward pass
    }
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_VsmPageLayout, 0, 1, &m_XformDescSet, 0, nullptr);   // set0 transforms
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_VsmPageLayout, 2, 1, &pageSet, 0, nullptr);          // set2 page data
    vkCmdPushConstants(cmd, m_VsmPageLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

    VkPipeline lastPipe = VK_NULL_HANDLE; VkDescriptorSet lastTex = VK_NULL_HANDLE;
    for (u32 g = 0; g < (u32)m_Groups.size(); ++g) {
        const TreeIndirectGroup& grp = m_Groups[g];
        VkPipeline pipe = (grp.tcOffset == 24) ? p24 : p28;
        if (pipe == VK_NULL_HANDLE || grp.descSetIdx >= m_TexDescSets.size()) continue;
        if (pipe != lastPipe) { vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe); lastPipe = pipe; lastTex = VK_NULL_HANDLE; }
        VkDescriptorSet tex = m_TexDescSets[grp.descSetIdx];
        if (tex != lastTex) { vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_VsmPageLayout, 1, 1, &tex, 0, nullptr); lastTex = tex; }
        VkDeviceSize vbOff = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &grp.vb, &vbOff);
        vkCmdBindIndexBuffer(cmd, m_MeshletIndexBuffer->GetHandle(), 0, VK_INDEX_TYPE_UINT16);   // shared meshlet IB
        const VkDeviceSize cmdOff = (VkDeviceSize)m_MeshletGroupBase[g] * sizeof(VkDrawIndexedIndirectCommand);
        const VkDeviceSize cntOff = (VkDeviceSize)g * sizeof(u32);
        vkCmdDrawIndexedIndirectCount(cmd, cmdBuf, cmdOff, countBuf, cntOff,
                                      m_MeshletGroupCap[g], sizeof(VkDrawIndexedIndirectCommand));
    }
}

// CPU near/far set for the wind hybrid. THE METRIC IS LIGHT-SPACE, NOT WORLD DISTANCE:
// with a low sun a tree's shadow lands tens of metres DOWN-SUN of its trunk, so "trunk
// near the camera" misses exactly the shadows the player is looking at (a house shaded
// by a tree 60 m up-sun stayed rigid). A tree goes DYNAMIC when its shadow COLUMN passes
// within r_vsm_tree_wind_dist of the camera — i.e. the light-space-XY distance between
// the tree and the camera (both projected along the sun) is under the threshold. A world
// -distance bound (3×dist) caps the up-sun corridor cost. Hysteresis (×1.15 on both)
// stops boundary flapping; every crossing queues an invalidation sphere so the residency
// pass re-renders the tree's STATIC pages the same frame (no ghost doubling). Call
// BEFORE the residency dispatch (vk_vsm) with this frame's sun view.
void CTreeManager::VsmUpdateNearSet(const Fmatrix& sunView)
{
    if (m_TotalCount == 0 || m_MetaCPU.size() < m_TotalCount) return;
    VK_CPU_PROBE("TreeNearSet");   // O(total trees) walk, every non-skip frame
    if (m_VsmNearCPU.size() != m_TotalCount) m_VsmNearCPU.assign(m_TotalCount, 0);
    const float nd = ps_r_vsm_tree_wind ? ps_r_vsm_tree_wind_dist : 0.f;
    // World-distance corridor bound: 2× the lateral radius (80 m at default 40). Trees
    // further up-sun still shade the player, but their sway detail no longer reads and
    // every corridor tree re-rasters its crown into the dyn atlas EVERY frame — measured
    // 3× (near 330 trees, instDyn 5-12k, VSMrender to 2-3.5ms) → the main perf knob.
    const float wd = nd * 2.f;
    const float ndIn2  = nd * nd,                    ndOut2 = (nd * 1.15f) * (nd * 1.15f);
    const float wdIn2  = wd * wd,                    wdOut2 = (wd * 1.15f) * (wd * 1.15f);
    // Crown-hull tier (r_vsm_tree_hull): near trees beyond hd cast from the baked opaque
    // hull, only the closest ones keep the full alpha-tested crown mesh. Same light-space
    // lateral metric as the near set (the tier decides the detail of the shadow NEAR THE
    // PLAYER). Hysteresis avoids per-frame tier flicker at the boundary; crown↔hull
    // crossings need NO invalidation circles — both tiers live in the per-frame dyn atlas.
    const bool  hullOn  = ps_r_vsm_tree_hull && m_HullDataReady;
    const bool  farHullOn = hullOn && ps_r_vsm_tree_hull >= 2;   // mode 2: far/static foliage too
    const float hd     = hullOn ? ps_r_vsm_tree_hull_dist : 0.f;
    const float hdIn2  = hd * hd,                    hdOut2 = (hd * 1.1f) * (hd * 1.1f);
    const Fvector cam = Device.vCameraPosition;
    Fvector camL; sunView.transform_tiny(camL, cam);  // camera in light space
    u32 nearCount = 0, hullCount = 0, farHullCount = 0;
    // Voxel-cloud caster CHOICE (r_vsm_tree_hull_vox): per tree, the camera-distance LOD
    // (same continuous cube-size/density math as the visual viewmode — shadow and view
    // agree on the cubes) plus the CROSSFADE fade: 1 = fully voxel, (0,1) = the band
    // just inside r_vsm_tree_hull_dist where the real crown still casts and the cubes
    // dissolve in — approaching a tree smoothly morphs its shadow into true leaves.
    const bool voxOn = hullOn && ps_r_vsm_tree_hull_vox > 0 && m_BrickTotal &&
                       m_BrickLodInfoCPU.size() == (size_t)m_TotalCount * kHullLods;
    // UE WPODisableDistance analog (needs mode 2 — the demoted trees land in bit2).
    const bool voxWpo = voxOn && farHullOn && ps_r_vsm_tree_hull_vox_wpo;
    if (m_VoxChoiceCPU.size() != (size_t)m_TotalCount * 4) m_VoxChoiceCPU.assign((size_t)m_TotalCount * 4, 0u);
    if (m_VoxLodSizeCPU.size() != m_TotalCount) m_VoxLodSizeCPU.assign(m_TotalCount, 0.f);
    const float band    = (std::min)(0.9f, (std::max)(0.02f, ps_r_vsm_tree_hull_band));
    const float bandIn  = hd * (1.f - band);   // fade 0 at this lateral dist, 1 at hd
    const float biasMul = powf(2.f, (float)ps_r_vsm_tree_hull_lod);
    const float stex    = ps_r_vsm_tree_hull_vox_stex;
    u32 voxCastCount = 0, voxBandCount = 0, voxCastSlice = 0;
    m_VoxCastListCPU.clear();   // compact vox-cull dispatch list (trees with a live brick choice)
    // Small visuals (bushes/shrubs, R < 3 m) never go dynamic: their shadows are small
    // ground blobs where sway is unreadable, but they dominate the near set by COUNT
    // (the manager holds every trees\* visual incl. bushes — user saw "20 trees", the
    // set held 330). They keep casting rigidly from the static cache.
    constexpr float kWindMinRadius = 3.0f;
    for (u32 i = 0; i < m_TotalCount; ++i) {
        const GpuTreeMeta& m = m_MetaCPU[i];
        Fvector tL; sunView.transform_tiny(tL, m.sphere_P);
        const float lx = tL.x - camL.x, ly = tL.y - camL.y;
        const float dxy2 = lx * lx + ly * ly;                 // lateral dist to the shadow column
        const float dw2  = cam.distance_to_sqr(m.sphere_P);   // world dist (cost bound)
        const bool wasNear    = (m_VsmNearCPU[i] & 1) != 0;
        const bool wasHull    = (m_VsmNearCPU[i] & 2) != 0;
        const bool wasFarHull = (m_VsmNearCPU[i] & 4) != 0;
        // Hull tiers (both use the same lateral shadow-column metric + hysteresis).
        // CROWNS-ONLY (windClass 2): trunks are opaque bark — no alpha test, not the
        // bottleneck, and their exact shadow anchors the silhouette; non-foliage statics
        // (vehicles/props also carried by this path) must never turn into lobes.
        //   bit1 = NEAR hull: dyn near set beyond hd (re-rendered per frame, wind).
        //   bit2 = FAR hull (mode 2): STATIC foliage beyond hd — makes the shadow LOD
        //          MONOTONIC (a 60 m tree no longer casts a MORE detailed shadow than a
        //          25 m one) and cuts the full-crown re-raster on dirty cache refreshes.
        //          Bushes/trees whose shadow column passes NEAR the player stay real.
        const bool crown     = i < m_WindClassCPU.size() && m_WindClassCPU[i] == 2;
        const bool hullSide  = hd > 0.f && (wasHull || wasFarHull ? (dxy2 > hdIn2) : (dxy2 > hdOut2));
        // UE WPODisableDistance: past the hull boundary the brick shadow is RIGID (its
        // wind amp ramps to 0 across the crossfade band, see amp16 below), and a rigid
        // caster buys nothing from the per-frame dyn atlas — demote it straight to the
        // cached STATIC tier (bit2). The dyn near set shrinks to the real-crown ring +
        // the band; hd crossings ride the same invalidation circles as every other
        // static-representation flip. Trees without baked bricks keep the old dyn
        // shell path (the shell has no per-brick wind story).
        const bool voxRigid  = voxWpo && crown && hullSide &&
                               m_BrickLodInfoCPU[(size_t)i * kHullLods].count != 0;
        const bool isNear  = !voxRigid && (nd > 0.f) && (m.sphere_R >= kWindMinRadius)
                           && (wasNear ? (dxy2 < ndOut2 && dw2 < wdOut2)
                                       : (dxy2 < ndIn2  && dw2 < wdIn2));
        const bool isHull    = isNear && crown && hullSide;
        const bool isFarHull = !isNear && farHullOn && crown && hullSide;
        if (isNear) ++nearCount;
        if (isHull) ++hullCount;
        if (isFarHull) ++farHullCount;
        // Per-tree voxel-caster choice — refreshed EVERY frame (camera moves even when
        // the tier flags don't), so it lives before the early-out below.
        if (voxOn) {
            u32* ch = &m_VoxChoiceCPU[(size_t)i * 4];
            ch[0] = ch[1] = ch[2] = ch[3] = 0u;
            // Leaving the far-hull tier drops the LOD ratchet anchor: the bit2 flip
            // below already invalidates the pages, and a stale anchor would fire a
            // duplicate circle on re-entry.
            if (!isFarHull) m_VoxLodSizeCPU[i] = 0.f;
            if (crown && (isNear || isHull || isFarHull)) {
                const float dxy = sqrtf(dxy2);
                float fade = 1.f;
                if (!isHull && !isFarHull)   // near-set band: dissolve in toward hd
                    fade = (dxy - bandIn) / (std::max)(hd - bandIn, 0.01f);
                // BRICK slices (UE-style shadow representation): the caster draws one
                // light-facing quad per 4×4×4-cell brick and DDA-raycasts its 64-bit
                // mask in the FS — the slice count here is BRICKS, ~×14-50 fewer units
                // than cubes, each costing 6 verts.
                const TreeVoxLod* lods = &m_BrickLodInfoCPU[(size_t)i * kHullLods];
                if (fade > 0.004f && lods[0].count) {
                    fade = (std::min)(fade, 1.f);
                    // Voxel LOD from the SHADOW VIEW, not the camera (UE: the same
                    // LOD metric serves color and every shadow view; for a VSM page
                    // the resolution is its clipmap texel). The tree's clipmap level
                    // comes from the same lateral window metric the residency uses
                    // (level L window spans base·2^L centred on the camera), so
                    // T = stex page texels at that level: near rings pick the finest
                    // bake (real leaf gaps close up), far rings coarsen in lockstep
                    // with the pages — the old camera-px curve + 0.35 m floor drove
                    // far trees to the 1-brick level = literal solid AABB shadows.
                    float lvlExt = ps_r_vsm_base;
                    while (lvlExt < 2.f * dxy && lvlExt < 32.f * ps_r_vsm_base) lvlExt *= 2.f;
                    float T = (lvlExt / 4096.f) * stex * biasMul;
                    T = (std::max)(T, ps_r_vsm_tree_hull_vox_sfloor);
                    const float sLast = lods[kHullLods - 1].count ? lods[kHullLods - 1].size : lods[0].size;
                    T = (std::min)(T, 4.f * sLast);
                    int lvl = 0;
                    while (lvl + 1 < (int)kHullLods && lods[lvl + 1].size <= T && lods[lvl + 1].count) ++lvl;
                    // Hard cap on the caster slice (bricks/tree) — bounds the worst-case
                    // draw regardless of species density.
                    const u32 voxCap = (u32)ps_r_vsm_tree_hull_vox_cap;
                    while (lvl + 1 < (int)kHullLods && lods[lvl].count > voxCap && lods[lvl + 1].count) ++lvl;
                    // Rendered voxel = the level's CELL exactly (bricks are a fixed grid;
                    // the continuous cube growth of the viewmode doesn't apply — level
                    // steps in the shadow hide behind PCF/EMA at these distances).
                    const float cube = lods[lvl].size;
                    ch[0] = lods[lvl].first;
                    ch[1] = lods[lvl].count;
                    memcpy(&ch[2], &cube, sizeof(float));
                    // Low halfword = brick wind AMPLITUDE (1−fade): full coherent sway
                    // next to the swaying crown at the band's inner edge, exactly 0 by
                    // the hd handover — so the demoted tree enters the rigid static
                    // cache without a snap, and every fully-voxel/static tree carries
                    // amp 0, letting the VS skip the whole ssfxTreeWind block (UE:
                    // WPO disabled at voxel distances, zero wind ALU).
                    const u32 amp16 = (u32)((1.f - fade) * 65535.f + 0.5f);
                    ch[3] = ((u32)(fade * 65535.f + 0.5f) << 16) | amp16;
                    ++voxCastCount;
                    m_VoxCastListCPU.push_back(i);
                    voxCastSlice += ch[1];
                    if (fade < 0.999f) ++voxBandCount;
                    // STATIC-tier LOD ratchet: far-hull voxels live in CACHED pages —
                    // a choice change alone never redraws them, so approaching a far
                    // tree (or any bush: R<3 m never joins the near set) left its
                    // shadow at the cube size from whenever the page last rendered,
                    // then "snapped" at the tier flip. Re-anchor + queue the same
                    // invalidation circle a tier flip uses whenever the cube leaves
                    // ±1.5× of the last rendered size (ratchet = built-in hysteresis;
                    // ≤4 circles/frame FIFO absorbs bursts). Dyn tier re-renders per
                    // frame and needs none of this.
                    // Only when the static tier actually casts bricks — the default far
                    // SHELL never changes with the cube size, so ratchet circles would
                    // be pure wasted cache invalidation traffic.
                    if (isFarHull && ps_r_vsm_tree_hull_vox_static) {
                        float& anchor = m_VoxLodSizeCPU[i];
                        if (anchor <= 0.f) anchor = cube;
                        else if (cube > anchor * 1.5f || cube * 1.5f < anchor) {
                            anchor = cube;
                            m_VsmPendingInval.push_back({ m.sphere_P.x, m.sphere_P.y, m.sphere_P.z, m.sphere_R * 1.4143f });
                        }
                    }
                }
            }
        }
        const u8 v = u8((isNear ? 1 : 0) | (isHull ? 2 : 0) | (isFarHull ? 4 : 0));
        if (v == m_VsmNearCPU[i]) continue;
        m_VsmNearCPU[i] = v;
        // Static pages must redraw when the tree's STATIC representation changes:
        // near↔far moves (existing) AND far real-mesh↔hull flips (bit 2). Near-set
        // crown↔hull (bit 1) is dyn-only — the dyn atlas re-renders every frame anyway.
        if (isNear == wasNear && isFarHull == wasFarHull) continue;
        // sqrt(2) inflation: the bin covers the sphere's bounding SQUARE in light XY —
        // the invalidation circle must reach its corners.
        m_VsmPendingInval.push_back({ m.sphere_P.x, m.sphere_P.y, m.sphere_P.z, m.sphere_R * 1.4143f });
    }
    m_VsmNearCount = nearCount;
    m_VsmHullCount = hullCount;
    m_VsmHullFarCount = farHullCount;
    m_VoxCastCount = voxCastCount;
    m_VoxBandCount = voxBandCount;
    m_VoxCastSlice = voxCastSlice;

    // Burst guard: a representation flip of half the forest (r_vsm_tree_hull 2 at
    // level load — every far tree's bit2 goes 0→1 on the FIRST update — or a live
    // toggle) queues tens of thousands of circles; drained at ≤4/frame that is an
    // HOUR-long permanent redraw storm (seen 17-07: pendingInval=88k, fps 11 on the
    // live flip). Past a threshold one full atlas invalidation is strictly cheaper:
    // a single clean re-render, then the cache is coherent again.
    if (m_VsmPendingInval.size() > 4096) {
        Msg("[VK Trees] VSM inval burst (%zu circles) -> full atlas invalidation instead",
            m_VsmPendingInval.size());
        m_VsmPendingInval.clear();
        VK::VSM::InvalidateCache();
    }
}

u32 CTreeManager::VsmPopTransitions(Fvector4* out, u32 maxOut)
{
    // FIFO: the resid pass consumes ≤4 circles/frame, so a burst of crossings (sprint
    // through a treeline) queues up — LIFO kept serving the NEWEST and starved the
    // oldest, leaving a crossed tree's stale rigid shadow doubled with its dyn one
    // until the queue drained. Oldest-first bounds the ghost to queue-length frames.
    const u32 n = _min(maxOut, (u32)m_VsmPendingInval.size());
    for (u32 i = 0; i < n; ++i) out[i] = m_VsmPendingInval[i];
    m_VsmPendingInval.erase(m_VsmPendingInval.begin(), m_VsmPendingInval.begin() + n);
    return n;
}

void CTreeManager::VsmBin(VkCommandBuffer cmd, VkBuffer pageTable, VkBuffer slotDirty, VkBuffer dynUsed, VkBuffer clipmapUBO, VkBuffer pageList, VkBuffer pageMax)
{
    if (!m_VsmReady) {
        CreateVsmResources();
        if (m_VsmReady) {
            // FIRST FRAME after lazy create: the frame-start VsmBinClears already ran
            // and skipped (m_VsmReady was false), so every GPU-written buffer still
            // holds its allocation garbage. On a level TRANSITION that garbage is
            // recycled VRAM from the destroyed managers — a junk indirect command
            // (indexCount/instanceCount in the billions) hangs the GPU → the
            // 'VSM/DynTrees' DEVICE_LOST. Zero everything NOW, before this frame's
            // bins/draws consume it. One-off cost, once per level.
            auto fill0 = [&](CVulkanBuffer* b) { if (b) vkCmdFillBuffer(cmd, b->GetHandle(), 0, VK_WHOLE_SIZE, 0u); };
            fill0(m_VsmStats);
            fill0(m_VsmTreeBits);
            fill0(m_VsmCasterPages);
            fill0(m_VsmIndirect);
            fill0(m_VsmDynIndirect);
            fill0(m_VsmHullIndirect);
            fill0(m_VsmHullIndirectS);
            fill0(m_VsmVoxIndirect);
            fill0(m_VsmVoxIndirectS);
            fill0(m_VoxCullCmdD);
            fill0(m_VoxCullCmdS);
            fill0(m_VoxCullList);
            fill0(m_VoxCullStats);
            fill0(m_VsmMeshletCmd);
            fill0(m_VsmMeshletCmdDyn);
            fill0(m_VsmMeshletGroupCount);
            fill0(m_VsmMeshletGroupCountDyn);
            fill0(m_MeshletStats);
            VkMemoryBarrier mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                                VK_ACCESS_TRANSFER_WRITE_BIT,
                                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT };
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                                 0, 1, &mb, 0, nullptr, 0, nullptr);
        }
    }
    if (!m_VsmReady || m_TotalCount == 0 || m_VsmBinPipe == VK_NULL_HANDLE) return;
    if (pageTable == VK_NULL_HANDLE || slotDirty == VK_NULL_HANDLE || dynUsed == VK_NULL_HANDLE || clipmapUBO == VK_NULL_HANDLE) return;
    if (pageMax == VK_NULL_HANDLE) pageMax = m_VsmCasterPages->GetHandle();   // HZB off: bind any valid buffer for layout parity

    m_VsmSlot = (m_VsmSlot + 1) % VK_FRAMES_IN_FLIGHT;
    const u32 slot = m_VsmSlot;

    // Upload this frame's CPU near-set flags (computed by VsmUpdateNearSet) into the ring
    // slot — async staged copy to the DEVICE-local buffer (the graphics submit waits the
    // upload timeline). Empty/mismatched CPU set → keep the slot's previous flags (it was
    // zero-filled at create; sizes only mismatch across a rebuild).
    if (m_VsmNearFlags[slot] && m_VsmNearCPU.size() == m_TotalCount) {
        if (m_VsmNearU32.size() != m_TotalCount) m_VsmNearU32.resize(m_TotalCount);
        for (u32 i = 0; i < m_TotalCount; ++i) m_VsmNearU32[i] = m_VsmNearCPU[i];   // u8 truth -> the shader's u32 flags
        m_VsmNearFlags[slot]->Upload(m_VsmNearU32.data(), (VkDeviceSize)m_TotalCount * sizeof(u32));
    }
    // Voxel-caster choice ring (same cadence: vsm_hull_cmd consumes it this VSM frame).
    if (u32* chp = (u32*)m_VoxChoicePtr[slot]) {
        if (m_VoxChoiceCPU.size() == (size_t)m_TotalCount * 4)
            memcpy(chp, m_VoxChoiceCPU.data(), (size_t)m_TotalCount * 4 * sizeof(u32));
        else
            memset(chp, 0, (size_t)m_TotalCount * 4 * sizeof(u32));
    }
    // Compact vox-cull tree list (same ring cadence): the stage-2 brick cull dispatches
    // one workgroup per LISTED tree, not one per tree in the level.
    m_VoxCullTreeN = 0;
    if (u32* tl = (u32*)m_VoxCullTreeListPtr[slot]) {
        const u32 n = _min((u32)m_VoxCastListCPU.size(), m_TotalCount);
        if (n) memcpy(tl, m_VoxCastListCPU.data(), n * sizeof(u32));
        m_VoxCullTreeN = n;
    }

    // No stats fill/barrier here — VsmBinClears batched it into the frame-start fill
    // block, and the caller's single residency→bins barrier orders it before this read.

    // mode 0 = STATIC pass: FAR trees into dirty static pages. dynUsed is bound for layout
    // completeness only (the shader's write is mode-gated).
    VkBuffer rmask = VK::VSM::GetRMaskHandle();
    if (rmask == VK_NULL_HANDLE) rmask = pageTable;   // layout parity (mode 0 never reads it)
    VK::DescriptorWriter(m_VsmBinSet[slot])
        .StorageBuffer(0,  m_TreeMetadataBuffer->GetHandle())
        .UniformBuffer(1,  clipmapUBO)
        .StorageBuffer(2,  pageTable)
        .StorageBuffer(3,  m_VsmCasterPages->GetHandle())
        .StorageBuffer(4,  m_VsmIndirect->GetHandle())
        .StorageBuffer(5,  m_VsmStats->GetHandle())
        .StorageBuffer(6,  slotDirty)                        // STATIC-atlas dirty set (cache filter)
        .StorageBuffer(7,  dynUsed)                          // unused in mode 0
        .StorageBuffer(8,  m_VsmNearFlags[slot]->GetHandle())// near set
        .StorageBuffer(9,  pageMax)                          // shadow-HZB occluder max (r_vsm_hzb)
        .StorageBuffer(10, pageTable)                        // staticPageTable — here pageTable IS static (mode 0 ignores it)
        .StorageBuffer(11, rmask)                            // receiver mask (mode 1 only — static must not partial-render)
        .StorageBuffer(12, m_VsmTreeBits->GetHandle())       // diag tree bitsets (dyn | static-accum)
        .Flush();

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_VsmBinPipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_VsmBinLayout, 0, 1, &m_VsmBinSet[slot], 0, nullptr);
    const u32 castN = _min(m_TotalCount, 0x40000u);   // 18-bit caster-pack ceiling
    struct { u32 count, arenaSlots, mode, hzbOn; float margin; u32 rmaskOn, bitBase; } push{
        castN, kVsmTreeArena, 0u, (u32)(ps_r_vsm_hzb ? 1 : 0), ps_r_vsm_hzb_margin, 0u,
        (m_TotalCount + 31u) / 32u };   // mode 0 = static + shadow-HZB (rmask off); bitBase = static-accum half
    vkCmdPushConstants(cmd, m_VsmBinLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd, (castN + 63) / 64, 1, 1);
    // Stage-2 meshlet refinement moved to VsmBinMeshlets (after the caller's barrier).
}

// mode 1 = DYNAMIC pass: NEAR trees into all resident dyn pages (re-rendered with wind
// every frame) + dynUsed flags for the resolve gate. Call right after VsmBin (shares the
// frame slot + the stats clear/barrier; the two dispatches write disjoint buffers/slices).
void CTreeManager::VsmBinDyn(VkCommandBuffer cmd, VkBuffer dynPageTable, VkBuffer dynUsed, VkBuffer clipmapUBO, VkBuffer dynPageList, VkBuffer pageMax, VkBuffer staticPageTable)
{
    if (!m_VsmReady || m_TotalCount == 0 || m_VsmBinPipe == VK_NULL_HANDLE) return;
    if (dynPageTable == VK_NULL_HANDLE || dynUsed == VK_NULL_HANDLE || clipmapUBO == VK_NULL_HANDLE) return;
    const bool hzb = ps_r_vsm_hzb && pageMax != VK_NULL_HANDLE && staticPageTable != VK_NULL_HANDLE;
    if (pageMax == VK_NULL_HANDLE)         pageMax = m_VsmCasterPages->GetHandle();          // layout parity when HZB off
    if (staticPageTable == VK_NULL_HANDLE) staticPageTable = dynPageTable;                    // layout parity when HZB off
    const u32 slot = m_VsmSlot;

    VkBuffer rmask = VK::VSM::GetRMaskHandle();
    const bool rmaskOn = ps_r_vsm_rmask && rmask != VK_NULL_HANDLE;
    if (rmask == VK_NULL_HANDLE) rmask = dynPageTable;   // layout parity when off
    VK::DescriptorWriter(m_VsmDynBinSet[slot])
        .StorageBuffer(0,  m_TreeMetadataBuffer->GetHandle())
        .UniformBuffer(1,  clipmapUBO)
        .StorageBuffer(2,  dynPageTable)
        .StorageBuffer(3,  m_VsmCasterPages->GetHandle())
        .StorageBuffer(4,  m_VsmDynIndirect->GetHandle())
        .StorageBuffer(5,  m_VsmStats->GetHandle())
        .StorageBuffer(6,  dynUsed)                          // slotDirty slot — unused in mode 1, any valid buffer
        .StorageBuffer(7,  dynUsed)                          // dynUsed flags (resolve gate)
        .StorageBuffer(8,  m_VsmNearFlags[slot]->GetHandle())// near set
        .StorageBuffer(9,  pageMax)                          // STATIC occluder max (Option A)
        .StorageBuffer(10, staticPageTable)                  // virtual -> STATIC slot (Option A)
        .StorageBuffer(11, rmask)                            // receiver mask (r_vsm_rmask sub-page cull)
        .StorageBuffer(12, m_VsmTreeBits->GetHandle())       // diag tree bitsets (dyn | static-accum)
        .Flush();

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_VsmBinPipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_VsmBinLayout, 0, 1, &m_VsmDynBinSet[slot], 0, nullptr);
    // Option A: near trees (dyn) cull ONLY pages hidden behind STATIC occluders → visible near shadow unchanged.
    const u32 castN = _min(m_TotalCount, 0x40000u);   // 18-bit caster-pack ceiling
    struct { u32 count, arenaSlots, mode, hzbOn; float margin; u32 rmaskOn, bitBase; } push{
        castN, kVsmTreeArena, 1u, (u32)(hzb ? 1 : 0), ps_r_vsm_hzb_margin, (u32)(rmaskOn ? 1 : 0),
        (m_TotalCount + 31u) / 32u };
    vkCmdPushConstants(cmd, m_VsmBinLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd, (castN + 63) / 64, 1, 1);

    // Crown-hull path (r_vsm_tree_hull): swap hull-tier trees' crown cmds for hull cmds and
    // zero the crown ones (the meshlet stage-2, which reads instanceCount, then skips them too —
    // it runs after the caller's stage-1→stage-2 barrier, which orders these writes).
    DispatchHullCmd(cmd);
    // Impostor path (r_vsm_tree_impostor): translate the crown page-bin's indirect into a
    // billboard-quad indirect (indexCount→6). Outside the atlas render pass; the draw-indirect
    // hazard is covered by vk_vsm's global barrier after this call.
    DispatchImpostorCmd(cmd);
    // Stage-2 meshlet refinement + diagnostics moved to VsmBinMeshlets (after the caller's barrier).
}

// Frame-start clears for the tree VSM bins — recorded inside vk_vsm's MarkPages fill
// block (before its transfer→compute barrier), so the bin dispatches themselves need
// no per-bin fills or barriers. Lazy resources may not exist on the very first frame
// (VsmBin creates them) — skip here; VsmBin's lazy-init branch then records a one-off
// zero of EVERY GPU-written buffer. That zero is NOT optional: recycled-VRAM garbage
// in an indirect command buffer = billion-instance draw = GPU hang (the level-transition
// DEVICE_LOST of 2026-07-09/10).
void CTreeManager::VsmBinClears(VkCommandBuffer cmd)
{
    if (!m_VsmReady || m_TotalCount == 0) return;
    if (m_VsmStats) vkCmdFillBuffer(cmd, m_VsmStats->GetHandle(), 0, VK_WHOLE_SIZE, 0u);
    // Diag bitsets: clear the per-frame DYN half only — the static half accumulates
    // (the static bin touches dirty pages only, per-frame counts would be noise).
    if (m_VsmTreeBits) vkCmdFillBuffer(cmd, m_VsmTreeBits->GetHandle(), 0,
                                       (VkDeviceSize)((m_TotalCount + 31u) / 32u) * sizeof(u32), 0u);
    if (m_VoxCullStats) vkCmdFillBuffer(cmd, m_VoxCullStats->GetHandle(), 0, VK_WHOLE_SIZE, 0u);   // per-frame append counters / draw counts
    if (m_MeshletVsmReady) {
        if (m_VsmMeshletGroupCount)    vkCmdFillBuffer(cmd, m_VsmMeshletGroupCount->GetHandle(),    0, VK_WHOLE_SIZE, 0u);
        if (m_VsmMeshletGroupCountDyn) vkCmdFillBuffer(cmd, m_VsmMeshletGroupCountDyn->GetHandle(), 0, VK_WHOLE_SIZE, 0u);
        if (m_MeshletStats)            vkCmdFillBuffer(cmd, m_MeshletStats->GetHandle(),            0, VK_WHOLE_SIZE, 0u);
    }
}

// Stage-2 meshlet refinement for both atlases. Call AFTER a barrier ordering the tree
// stage-1 bins' writes (casterPages / indirect) before compute reads — the refinement
// is the only cross-dispatch dependency in the whole bin group. Also hosts the hybrid
// diagnostics (moved from VsmBinDyn so the meshlet stats it copies are this frame's).
void CTreeManager::VsmBinMeshlets(VkCommandBuffer cmd, bool dyn, VkBuffer pageListStatic, VkBuffer pageListDyn, VkBuffer clipmapUBO,
                                  VkBuffer pageMaxBlk, VkBuffer staticPT)
{
    if (!m_VsmReady || m_TotalCount == 0) return;
    // Lazy resource build for a live r_vsm_meshlet 0→1 flip (the load path skips the
    // ~470 MB of pooled command buffers while the mode is off).
    if (ps_r_vsm_meshlet && !m_MeshletVsmReady) CreateMeshletVsmResources();
    const bool binSub = ps_r_profiler > 1;   // per-dispatch attribution (see vk_vsm's Bins/* zones)
    if (IsMeshletMode()) {
        const int zbMesh = binSub ? VK::Prof::ZoneBegin(cmd, "Bins/Meshlet") : -1;
        DispatchMeshletBin(cmd, 0u, pageListStatic, clipmapUBO);
        if (dyn) DispatchMeshletBin(cmd, 1u, pageListDyn, clipmapUBO);
        if (zbMesh >= 0) VK::Prof::ZoneEnd(cmd, zbMesh);
    }
    // Stage-2 brick cull rides the same barrier slot (reads vsm_hull_cmd's outputs).
    {
        const int zbVox = binSub ? VK::Prof::ZoneBegin(cmd, "Bins/VoxCull") : -1;
        DispatchVoxCull(cmd, pageListDyn, pageListStatic, clipmapUBO, pageMaxBlk, staticPT);
        if (zbVox >= 0) VK::Prof::ZoneEnd(cmd, zbVox);
    }

    // ---- Hybrid diagnostics (r_vsm_debug): stats readback + 2s log. stats[3] = instances
    // the DYN pass binned (0 while near trees exist = the GPU never saw the near flags).
    if ((ps_r_vsm_debug || ps_r_profiler > 0) && m_VsmStatsRB) {   // profiler sessions get the tree caster telemetry too
        VkMemoryBarrier b{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
        VkBufferCopy rc{ 0, 0, 8 * sizeof(u32) };
        vkCmdCopyBuffer(cmd, m_VsmStats->GetHandle(), m_VsmStatsRB->GetHandle(), 1, &rc);
        if (m_VsmTreeBits && m_VsmTreeBitsPtr) {
            VkBufferCopy bc{ 0, 0, m_VsmTreeBits->GetSize() };
            vkCmdCopyBuffer(cmd, m_VsmTreeBits->GetHandle(), m_VsmTreeBitsRB->GetHandle(), 1, &bc);
        }
        if (m_VsmStatsPtr && Device.dwTimeGlobal > m_VsmLastLog + 2000) {
            m_VsmLastLog = Device.dwTimeGlobal;
            if (ps_r_vsm_hzb)
                Msg("[VK Trees] VSM shadow-HZB: culled (tree,page) pairs = %u", m_VsmStatsPtr[4]);
            // instStatic/instDyn: what the STATIC (cached, dirty-pages-only) vs DYNAMIC
            // (near, every-frame wind) passes actually rasterize — the "are we promoting
            // too much to dyn / redrawing too much static" watch numbers. arena = pairs
            // reserved this frame of the shared casterPages arena (drops must stay 0).
            Msg("[VK Trees] VSM hybrid: near=%u/%u hull=%u hullFar=%u pendingInval=%zu | GPU draws=%u instStatic=%u instDyn=%u maxPages=%u | arena=%u/%u drops=%u%s | rmaskCulled=%u",
                m_VsmNearCount, m_TotalCount, IsHullMode() ? m_VsmHullCount : 0u,
                IsHullMode() ? m_VsmHullFarCount : 0u, m_VsmPendingInval.size(),
                m_VsmStatsPtr[0], m_VsmStatsPtr[1] - m_VsmStatsPtr[3], m_VsmStatsPtr[3],
                m_VsmStatsPtr[2],
                m_VsmStatsPtr[6], kVsmTreeArena, m_VsmStatsPtr[7], m_VsmStatsPtr[7] ? " [ARENA FULL!]" : "",
                m_VsmStatsPtr[5]);   // receiver-mask culled (tree,page) pairs, dyn bin (r_vsm_rmask)
            // Distinct-tree attribution (diag bitsets, a frame or two stale): how many
            // trees shadow through the ACTIVE dyn atlas this frame vs how many live in
            // the CACHED static atlas (accumulated since it last fully invalidated).
            if (m_VsmTreeBitsPtr) {
                const u32 W = (m_TotalCount + 31u) / 32u;
                u32 dynN = 0, cachedN = 0;
                for (u32 i = 0; i < W; ++i) { dynN += (u32)std::popcount(m_VsmTreeBitsPtr[i]); cachedN += (u32)std::popcount(m_VsmTreeBitsPtr[W + i]); }
                Msg("[VK Trees] shadow trees: dyn(active)=%u/frame, cached(static atlas)=%u accum of %u",
                    dynN, cachedN, m_TotalCount);
            }
        }
        // Stage-2 brick cull: compacted draw counts + list usage + shadow-HZB culled bricks.
        if (m_VoxCullActive && m_VoxCullStatsRB && m_VoxCullStatsPtr) {
            VkBufferCopy vc{ 0, 0, 8 * sizeof(u32) };
            vkCmdCopyBuffer(cmd, m_VoxCullStats->GetHandle(), m_VoxCullStatsRB->GetHandle(), 1, &vc);
            if (Device.dwTimeGlobal == m_VsmLastLog)   // logged this tick above (stale by a frame — fine)
                Msg("[VK Trees] vox-cull: trees=%u cmdD=%u cmdS=%u list=%u/%u overflow=%u hzbCulled=%u rmaskCulled=%u",
                    m_VoxCullTreeN,
                    m_VoxCullStatsPtr[0], m_VoxCullStatsPtr[1], m_VoxCullStatsPtr[2], kVoxCullListCap,
                    m_VoxCullStatsPtr[3], m_VoxCullStatsPtr[4], m_VoxCullStatsPtr[5]);
        }
        // Meshlet path (r_vsm_meshlet): commands emitted + group overflow (truncation).
        if (IsMeshletMode() && m_MeshletStatsRB && m_MeshletStatsPtr) {
            VkBufferCopy mc{ 0, 0, 4 * sizeof(u32) };
            vkCmdCopyBuffer(cmd, m_MeshletStats->GetHandle(), m_MeshletStatsRB->GetHandle(), 1, &mc);
            if (Device.dwTimeGlobal == m_VsmLastLog)   // logged this tick above
                Msg("[VK Trees] VSM meshlet: cmds=%u overflow=%u (clusters=%u, %u cmd/tree)%s",
                    m_MeshletStatsPtr[0], m_MeshletStatsPtr[1], m_MeshletTotal, kCmdPerTree,
                    m_MeshletStatsPtr[1] ? " [TRUNCATED!]" : "");
        }
    }
}

void CTreeManager::VsmRender(VkCommandBuffer cmd, VkBuffer pageList, VkBuffer clipmapUBO)
{
    if (!m_VsmReady || m_TotalCount == 0 || m_XformDescSet == VK_NULL_HANDLE) return;
    if (m_VsmPagePipe24 == VK_NULL_HANDLE && m_VsmPagePipe28 == VK_NULL_HANDLE) return;
    if (pageList == VK_NULL_HANDLE || clipmapUBO == VK_NULL_HANDLE) return;
    const u32 slot = m_VsmSlot;

    VK::DescriptorWriter(m_VsmPageSet[slot])
        .StorageBuffer(0, pageList)
        .StorageBuffer(1, m_VsmCasterPages->GetHandle())
        .UniformBuffer(2, clipmapUBO)
        .Flush();

    // FAR crown-hull tier (r_vsm_tree_hull 2): distant foliage casts rigid opaque lobes
    // into the dirty static pages (its crown cmds were zeroed by vsm_hull_cmd) — the
    // shadow LOD stays monotonic beyond the near set and dirty refreshes stop paying the
    // full alpha-tested crown raster.
    if (IsHullMode() && ps_r_vsm_tree_hull >= 2) {
        DrawHulls(cmd, slot, /*staticPass*/true);
        DrawVoxCasters(cmd, slot, /*staticPass*/true);   // voxel-cloud far tier (vsm_hull_cmd routed the pages)
    }

    // Phase B: meshlet-cull path draws only the clusters that survived stage 2 (STATIC = rigid).
    if (IsMeshletMode()) { DrawMeshlets(cmd, 0u, m_VsmPageSet[slot], /*wind*/false); return; }

    // STATIC pass push: NO wind, ever — the toroidal cache needs frame-invariant page
    // content (per-page wind was the "jelly": staggered refreshes froze each page at a
    // different wind phase). Near trees sway via VsmRenderDyn instead.
    struct VsmPagePush {
        float    uvScale, alphaRef; u32 cap, pad;
        Fvector4 wind_params, wsetup_trees, wind_anim;
    } pc{};
    pc.uvScale = 1.0f / 2048.0f; pc.alphaRef = 200.0f / 255.0f; pc.cap = kVsmTreeCap;
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_VsmPageLayout, 0, 1, &m_XformDescSet, 0, nullptr);       // set0 transforms
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_VsmPageLayout, 2, 1, &m_VsmPageSet[slot], 0, nullptr);   // set2 page data
    vkCmdPushConstants(cmd, m_VsmPageLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

    VkPipeline lastPipe = VK_NULL_HANDLE; VkDescriptorSet lastTex = VK_NULL_HANDLE;
    for (const TreeIndirectGroup& grp : m_Groups) {
        VkPipeline pipe = (grp.tcOffset == 24) ? m_VsmPagePipe24 : m_VsmPagePipe28;
        if (pipe == VK_NULL_HANDLE || grp.descSetIdx >= m_TexDescSets.size()) continue;
        if (pipe != lastPipe) { vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe); lastPipe = pipe; lastTex = VK_NULL_HANDLE; }
        VkDescriptorSet tex = m_TexDescSets[grp.descSetIdx];
        if (tex != lastTex) { vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_VsmPageLayout, 1, 1, &tex, 0, nullptr); lastTex = tex; }
        VkDeviceSize vbOff = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &grp.vb, &vbOff);
        vkCmdBindIndexBuffer(cmd, grp.ib, 0, VK_INDEX_TYPE_UINT16);
        // [#7] ONE multi-draw indirect for the whole group — the group's trees are contiguous
        // in m_VsmIndirect (tree-index order), so a single call replaces meshCount individual
        // draws. Cuts ~6288 draw calls/frame to one per group (~tens). Trees with no dirty pages
        // carry instanceCount=0 from the bin → ~free on the GPU. Needs multiDrawIndirect +
        // drawIndirectFirstInstance (both enabled in HW_Vulkan).
        vkCmdDrawIndexedIndirect(cmd, m_VsmIndirect->GetHandle(),
                                 (VkDeviceSize)grp.meshOffset * sizeof(VkDrawIndexedIndirectCommand),
                                 grp.meshCount, sizeof(VkDrawIndexedIndirectCommand));
    }
}

// DYNAMIC pass: near trees into the dyn atlas with LIVE wind (one time slice per frame →
// smooth coherent sway, the r_vsm_tree_wind feature done right). Call inside vk_vsm's
// RenderAtlas DYNAMIC pass, after the skinned/grass casters.
void CTreeManager::VsmRenderDyn(VkCommandBuffer cmd, VkBuffer dynPageList, VkBuffer clipmapUBO)
{
    if (!ps_r_vsm_tree_wind) return;   // hybrid off → all trees are in the static pass
    if (!m_VsmReady || m_TotalCount == 0 || m_XformDescSet == VK_NULL_HANDLE) return;
    if (m_VsmPageDynPipe24 == VK_NULL_HANDLE && m_VsmPageDynPipe28 == VK_NULL_HANDLE) return;
    if (dynPageList == VK_NULL_HANDLE || clipmapUBO == VK_NULL_HANDLE) return;
    const u32 slot = m_VsmSlot;

    VK::DescriptorWriter(m_VsmDynPageSet[slot])
        .StorageBuffer(0, dynPageList)
        .StorageBuffer(1, m_VsmCasterPages->GetHandle())
        .UniformBuffer(2, clipmapUBO)
        .Flush();

    // Crown-hull tier (r_vsm_tree_hull): draw the hull-tier trees FIRST (opaque low-poly
    // lobes, one multi-draw, no texture) — their crown cmds were zeroed by vsm_hull_cmd,
    // so the plain/meshlet crown draws below cover only the crown tier. Depth-only pass →
    // draw order irrelevant.
    if (IsHullMode()) {
        DrawHulls(cmd, slot, /*staticPass*/false);
        DrawVoxCasters(cmd, slot, /*staticPass*/false);   // voxel-cloud near tier + crown crossfade band
    }

    // Impostor path (r_vsm_tree_impostor): ONE sun-facing billboard per near tree instead of
    // the crown mesh — 2 tris vs thousands of crown verts every frame. Same page routing (set2);
    // the quad samples the baked silhouette atlas. Supersedes the crown mesh + meshlet paths.
    if (IsImpostorMode()) {
        struct ImpPush {
            Fvector4 sunDir, wind_params, wsetup_trees, wind_anim;
            u32 cap, numMeshes; float alphaRef; u32 facetCount;
        } pc{};
        Fvector sd; sd.set(0.f, -1.f, 0.f);
        if (g_pGamePersistent) if (auto* E = g_pGamePersistent->Environment().CurrentEnv) sd = E->sun_dir;
        pc.sunDir.set(sd.x, sd.y, sd.z, ps_r_vsm_tree_impostor_scale);
        pc.wsetup_trees.set(ps_r_wind_tree_anim, ps_r_wind_tree_trunk, ps_r_wind_tree_bend, 0.1f);
        if (g_pGamePersistent) {
            const Fvector3 wa = g_pGamePersistent->Environment().wind_anim;
            pc.wind_anim.set(wa.x, wa.y, wa.z, ps_r_wind_tree_flutter);
            if (auto* E = g_pGamePersistent->Environment().CurrentEnv)
                pc.wind_params.set(E->wind_direction, E->wind_velocity, 0.0f, 0.0f);
        }
        pc.wind_params.w = ps_r_wind_tree_crown;   // flutter height gate — must match the crown pass
        pc.cap = kVsmTreeCap; pc.numMeshes = m_ImpostorMeshCount; pc.alphaRef = 0.5f; pc.facetCount = kImpFacets;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_ImpostorPipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_ImpostorLayout, 0, 1, &m_XformDescSet, 0, nullptr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_ImpostorLayout, 1, 1, &m_ImpostorSet, 0, nullptr);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_ImpostorLayout, 2, 1, &m_VsmDynPageSet[slot], 0, nullptr);
        vkCmdPushConstants(cmd, m_ImpostorLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
        // ONE multi-draw over all trees: far/idle trees carry instanceCount 0 from the copy → free.
        vkCmdDrawIndirect(cmd, m_VsmImpostorDynIndirect->GetHandle(), 0, m_TotalCount, sizeof(VkDrawIndirectCommand));
        return;
    }

    // Phase B: meshlet-cull path draws only surviving clusters, with LIVE wind (near trees).
    if (IsMeshletMode()) { DrawMeshlets(cmd, 1u, m_VsmDynPageSet[slot], /*wind*/true); return; }

    // Live wind push — same params as the forward tree pass (ssfx_tree_wind).
    struct VsmPagePush {
        float    uvScale, alphaRef; u32 cap, pad;
        Fvector4 wind_params, wsetup_trees, wind_anim;
    } pc{};
    pc.uvScale = 1.0f / 2048.0f; pc.alphaRef = 200.0f / 255.0f; pc.cap = kVsmTreeCap;
    pc.wsetup_trees.set(ps_r_wind_tree_anim, ps_r_wind_tree_trunk, ps_r_wind_tree_bend, 0.1f);
    if (g_pGamePersistent) {
        const Fvector3 wa = g_pGamePersistent->Environment().wind_anim;
        pc.wind_anim.set(wa.x, wa.y, wa.z, ps_r_wind_tree_flutter);
        if (auto* E = g_pGamePersistent->Environment().CurrentEnv)
            pc.wind_params.set(E->wind_direction, E->wind_velocity, 0.0f, 0.0f);
    }
    pc.wind_params.w = ps_r_wind_tree_crown;   // flutter height gate — must match the forward pass
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_VsmPageLayout, 0, 1, &m_XformDescSet, 0, nullptr);          // set0 transforms
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_VsmPageLayout, 2, 1, &m_VsmDynPageSet[slot], 0, nullptr);   // set2 page data (dyn)
    vkCmdPushConstants(cmd, m_VsmPageLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);

    // VRS: 2x2 coarse shading of the crown-shadow fill (A/B via r_vsm_tree_vrs). The dyn
    // pipelines carry a FRAGMENT_SHADING_RATE dynamic state (HW pipeline-rate), so the rate
    // MUST be set before the draws; reset to 1x1 after so nothing downstream inherits it.
    // No-op when HW lacks pipeline-rate VRS. Attacks the fill-bound cost that survives WIND.
    const u32 tvrs = (ps_r_vsm_tree_vrs && VulkanHW.m_bVRSPipelineSupported) ? 2u : 1u;
    VK::VRS::CmdSetPipelineRate(cmd, tvrs, tvrs);

    VkPipeline lastPipe = VK_NULL_HANDLE; VkDescriptorSet lastTex = VK_NULL_HANDLE;
    for (const TreeIndirectGroup& grp : m_Groups) {
        VkPipeline pipe = (grp.tcOffset == 24) ? m_VsmPageDynPipe24 : m_VsmPageDynPipe28;
        if (pipe == VK_NULL_HANDLE || grp.descSetIdx >= m_TexDescSets.size()) continue;
        if (pipe != lastPipe) { vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe); lastPipe = pipe; lastTex = VK_NULL_HANDLE; }
        VkDescriptorSet tex = m_TexDescSets[grp.descSetIdx];
        if (tex != lastTex) { vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_VsmPageLayout, 1, 1, &tex, 0, nullptr); lastTex = tex; }
        VkDeviceSize vbOff = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &grp.vb, &vbOff);
        vkCmdBindIndexBuffer(cmd, grp.ib, 0, VK_INDEX_TYPE_UINT16);
        // Far trees carry instanceCount=0 in m_VsmDynIndirect → ~free on the GPU.
        vkCmdDrawIndexedIndirect(cmd, m_VsmDynIndirect->GetHandle(),
                                 (VkDeviceSize)grp.meshOffset * sizeof(VkDrawIndexedIndirectCommand),
                                 grp.meshCount, sizeof(VkDrawIndexedIndirectCommand));
    }
    VK::VRS::CmdSetPipelineRate(cmd, 1u, 1u);   // reset coarse rate (dyn crowns only)
}

// ============================================================================
// Impostor crown shadows (r_vsm_tree_impostor). Bakes a per-UNIQUE-MESH silhouette
// atlas at load (8 azimuth facets/row, alpha-tested from the crown diffuse), then in
// the DYNAMIC near-tree pass replaces the crown MESH raster with ONE sun-facing
// billboard per tree (2 tris) routed through the same VSM page mechanism — killing
// the geometry/raster cost that survives WIND. See tree_vsm_impostor.* + the plan
// memory [vulkan-vsm-tree-impostor-plan]. Best-effort: any failure leaves the atlas
// unbuilt → IsImpostorMode() false → the crown mesh path runs unchanged.
// ============================================================================
bool CTreeManager::IsImpostorMode() const
{
    return ps_r_vsm_tree_impostor && ps_r_vsm_tree_wind && m_ImpostorReady;
}

void CTreeManager::BuildImpostorAtlas(const xr_vector<vkFTreeVisual*>& trees,
                                      const xr_vector<GpuTreeMeta>& meta,
                                      const xr_vector<u32>& treeTexIdx)
{
    VK::Vram::Scope _vram_scope("Trees");
    if (m_TotalCount == 0 || !g_ShaderManager) return;
    if (m_TexDescSets.empty() || trees.size() != m_TotalCount) return;

    VkDevice dev = VulkanHW.m_Device;

    // ----- Dedup unique meshes by DRAWN window (vb, ib, first_vertex, ib_first, index_count).
    struct MeshKey { VkBuffer vb, ib; u32 fv, ibf, ic, tc, desc; };
    xr_vector<MeshKey> uniq; uniq.reserve(64);
    xr_vector<u32> treeSil(m_TotalCount, 0u);
    for (u32 i = 0; i < m_TotalCount; ++i) {
        vkFTreeVisual* t = trees[i];
        MeshKey k{ t->m_mesh.p_rm_Vertices->GetHandle(), t->m_mesh.p_rm_Indices->GetHandle(),
                   meta[i].first_vertex, meta[i].ib_first, meta[i].index_count,
                   t->m_mesh.tcOffset, treeTexIdx[i] };
        u32 slot = ~0u;
        for (u32 j = 0; j < uniq.size(); ++j) {
            const MeshKey& u = uniq[j];
            if (u.vb == k.vb && u.ib == k.ib && u.fv == k.fv && u.ibf == k.ibf && u.ic == k.ic) { slot = j; break; }
        }
        if (slot == ~0u) { slot = (u32)uniq.size(); uniq.push_back(k); }
        treeSil[i] = slot;
    }
    m_ImpostorMeshCount = (u32)uniq.size();
    if (m_ImpostorMeshCount == 0) return;

    // Per-unique-mesh local framing (box center + world sphere radius from the first tree using it).
    struct MeshFrame { Fvector center; float radius; };
    xr_vector<MeshFrame> frame(m_ImpostorMeshCount, MeshFrame{ Fvector{}, 1.0f });
    xr_vector<u8> haveFrame(m_ImpostorMeshCount, 0);
    for (u32 i = 0; i < m_TotalCount; ++i) {
        u32 s = treeSil[i]; if (haveFrame[s]) continue;
        haveFrame[s] = 1;
        const Fbox& bb = trees[i]->vis.box;   // LOCAL AABB (bake renders local verts)
        Fvector c; c.add(bb.min, bb.max); c.mul(0.5f);
        frame[s].center = c;
        frame[s].radius = (meta[i].sphere_R > 1e-3f) ? meta[i].sphere_R : 1.0f;
    }

    // ----- Silhouette atlas (R8 coverage): FACETS cols × meshCount rows, kImpCell each.
    const u32 W = kImpFacets * kImpCell;
    const u32 H = m_ImpostorMeshCount * kImpCell;
    VkImageCreateInfo ici{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ici.imageType = VK_IMAGE_TYPE_2D; ici.extent = { W, H, 1 };
    ici.mipLevels = 1; ici.arrayLayers = 1; ici.format = VK_FORMAT_R8_UNORM;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.samples = VK_SAMPLE_COUNT_1_BIT; ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo aci{}; aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    aci.priority = 1.0f;   // tree impostor atlas — hot render target, never evict before streamable textures
    if (VK::Vram::CreateImage(VulkanHW.m_Allocator, &ici, &aci, &m_ImpostorAtlas, &m_ImpostorAtlasAlloc, nullptr) != VK_SUCCESS) {
        Msg("![VK Trees] impostor atlas image create failed"); m_ImpostorMeshCount = 0; return;
    }
    VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vci.image = m_ImpostorAtlas; vci.viewType = VK_IMAGE_VIEW_TYPE_2D; vci.format = ici.format;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vkCreateImageView(dev, &vci, nullptr, &m_ImpostorAtlasView);
    VkSamplerCreateInfo sci{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    sci.magFilter = sci.minFilter = VK_FILTER_LINEAR; sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vkCreateSampler(dev, &sci, nullptr, &m_ImpostorSampler);

    // Per-tree silhouette-row SSBO (host-visible, read in the impostor VS).
    m_ImpostorSilIdx = xr_new<CVulkanBuffer>();
    m_ImpostorSilIdx->Create((VkDeviceSize)m_TotalCount * sizeof(u32), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
    if (void* p = m_ImpostorSilIdx->Map()) memcpy(p, treeSil.data(), (size_t)m_TotalCount * sizeof(u32));

    // ----- Transient bake pipeline (tree_bake_silhouette VS/FS, tcOffset 24 + 28).
    VkShaderModule bvs = g_ShaderManager->Load("tree_bake_silhouette.vert.spv");
    VkShaderModule bfs = g_ShaderManager->Load("tree_bake_silhouette.frag.spv");
    if (!bvs || !bfs) { Msg("![VK Trees] impostor bake shaders missing"); m_ImpostorMeshCount = 0; return; }
    struct BakePush { Fmatrix mvp; float uvScale, alphaRef; };
    VkPushConstantRange pcr{ VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(BakePush) };
    VkPipelineLayout bakeLayout = VK_NULL_HANDLE;
    { VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
      plci.setLayoutCount = 1; plci.pSetLayouts = &m_TexDescLayout; plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
      vkCreatePipelineLayout(dev, &plci, nullptr, &bakeLayout); }
    VkPipeline bakePipe24 = VK_NULL_HANDLE, bakePipe28 = VK_NULL_HANDLE;
    auto makeBake = [&](u32 tcOffset, VkPipeline& out) {   // colour-only — no depth state at all
        out = VK::GfxPipelineBuilder(bakeLayout)
            .Vert(bvs).Frag(bfs)
            .Binding(0, 32)
            .Attr(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0)
            .Attr(1, 0, VK_FORMAT_R16G16_SSCALED,   tcOffset)
            .Cull(VK_CULL_MODE_NONE)
            .Color(VK_FORMAT_R8_UNORM, VK_COLOR_COMPONENT_R_BIT)
            .Build("Trees impostor bake (tcOff=%u)", tcOffset);
    };
    makeBake(24, bakePipe24); makeBake(28, bakePipe28);

    // ----- Bake: one-off command buffer, all cells in one dynamic-rendering pass.
    VkCommandBuffer cmd = VulkanHW.BeginSingleTimeCommands();
    VK::ImageBarrier(cmd, m_ImpostorAtlas, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VK::RenderingBuilder(W, H)
        .ColorClear(m_ImpostorAtlasView, VkClearColorValue{ { 0.f, 0.f, 0.f, 0.f } })
        .Begin(cmd);
    for (u32 s = 0; s < m_ImpostorMeshCount; ++s) {
        const MeshKey& k = uniq[s];
        VkPipeline pipe = (k.tc == 24) ? bakePipe24 : bakePipe28;
        if (pipe == VK_NULL_HANDLE || k.desc >= m_TexDescSets.size()) continue;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, bakeLayout, 0, 1, &m_TexDescSets[k.desc], 0, nullptr);
        VkDeviceSize vbOff = 0; vkCmdBindVertexBuffers(cmd, 0, 1, &k.vb, &vbOff);
        vkCmdBindIndexBuffer(cmd, k.ib, 0, VK_INDEX_TYPE_UINT16);
        const Fvector C = frame[s].center; const float R = frame[s].radius;
        for (u32 f = 0; f < kImpFacets; ++f) {
            const float ang = float(f) * (6.28318530718f / float(kImpFacets));
            Fvector dir; dir.set(_cos(ang), 0.f, _sin(ang));
            Fvector eye; eye.mad(C, dir, 2.0f * R);      // eye = C + dir*2R (on the sun-facing side)
            Fvector look; look.invert(dir);              // look toward C along -dir
            Fvector up; up.set(0.f, 1.f, 0.f);
            Fmatrix view; view.build_camera_dir(eye, look, up);
            Fmatrix proj; proj.build_projection_ortho(2.0f * R, 2.0f * R, 0.05f, 4.0f * R);
            BakePush bp{}; bp.mvp.mul(proj, view); bp.uvScale = 1.0f / 2048.0f; bp.alphaRef = 200.0f / 255.0f;
            VkViewport vpc{ float(f * kImpCell), float(s * kImpCell), float(kImpCell), float(kImpCell), 0.f, 1.f };
            VkRect2D scc{ { int(f * kImpCell), int(s * kImpCell) }, { kImpCell, kImpCell } };
            vkCmdSetViewport(cmd, 0, 1, &vpc); vkCmdSetScissor(cmd, 0, 1, &scc);
            vkCmdPushConstants(cmd, bakeLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(bp), &bp);
            vkCmdDrawIndexed(cmd, k.ic, 1, k.ibf, (s32)k.fv, 0);
        }
    }
    vkCmdEndRendering(cmd);
    VK::ImageBarrier(cmd, m_ImpostorAtlas, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    VulkanHW.EndSingleTimeCommands(cmd);

    if (bakePipe24) vkDestroyPipeline(dev, bakePipe24, nullptr);
    if (bakePipe28) vkDestroyPipeline(dev, bakePipe28, nullptr);
    if (bakeLayout) vkDestroyPipelineLayout(dev, bakeLayout, nullptr);

    Msg("[VK Trees] impostor silhouette atlas baked: %u meshes × %u facets (%ux%u R8)", m_ImpostorMeshCount, kImpFacets, W, H);
}

void CTreeManager::CreateImpostorResources()
{
    if (m_ImpostorReady) return;
    if (m_ImpostorAtlasView == VK_NULL_HANDLE || !m_ImpostorSilIdx || m_ImpostorMeshCount == 0) return;
    if (m_XformDescLayout == VK_NULL_HANDLE || m_VsmPageSetL == VK_NULL_HANDLE || !m_TreeMetadataBuffer || !m_VsmDynIndirect) return;
    if (!g_ShaderManager) return;
    VkShaderModule vs = g_ShaderManager->Load("tree_vsm_impostor.vert.spv");
    VkShaderModule fs = g_ShaderManager->Load("tree_vsm_impostor.frag.spv");
    VkShaderModule cs = g_ShaderManager->Load("vsm_impostor_cmd.comp.spv");
    if (!vs || !fs || !cs) { Msg("![VK Trees] impostor render shaders missing — impostor disabled"); return; }
    VkDevice dev = VulkanHW.m_Device;

    m_VsmImpostorDynIndirect = xr_new<CVulkanBuffer>();
    m_VsmImpostorDynIndirect->Create((VkDeviceSize)m_TotalCount * sizeof(VkDrawIndirectCommand),
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

    VkDescriptorPoolSize ps[2] = { { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 }, { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 } };
    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets = 2; pci.poolSizeCount = 2; pci.pPoolSizes = ps;
    if (vkCreateDescriptorPool(dev, &pci, nullptr, &m_ImpostorPool) != VK_SUCCESS) return;

    // set1: meta SSBO (VS) + silIdx SSBO (VS) + silhouette sampler (FS).
    { VkDescriptorSetLayoutBinding b[3]{};
      b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;         b[0].descriptorCount = 1; b[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
      b[1].binding = 1; b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;         b[1].descriptorCount = 1; b[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
      b[2].binding = 2; b[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; b[2].descriptorCount = 1; b[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
      VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO }; lci.bindingCount = 3; lci.pBindings = b;
      if (vkCreateDescriptorSetLayout(dev, &lci, nullptr, &m_ImpostorSetL) != VK_SUCCESS) return;
      VkDescriptorSetAllocateInfo dai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO }; dai.descriptorPool = m_ImpostorPool; dai.descriptorSetCount = 1; dai.pSetLayouts = &m_ImpostorSetL;
      if (vkAllocateDescriptorSets(dev, &dai, &m_ImpostorSet) != VK_SUCCESS) return;
      VK::DescriptorWriter(m_ImpostorSet)
          .StorageBuffer(0, m_TreeMetadataBuffer->GetHandle())
          .StorageBuffer(1, m_ImpostorSilIdx->GetHandle())
          .ImageSampler (2, m_ImpostorAtlasView, m_ImpostorSampler)
          .Flush();
    }

    // Graphics pipeline: {set0 xform, set1 impostor, set2 page}, no vertex input (gl_VertexIndex quad).
    m_ImpostorLayout = VK::MakePipelineLayout({ m_XformDescLayout, m_ImpostorSetL, m_VsmPageSetL }, 80,
                                              VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
    if (m_ImpostorLayout == VK_NULL_HANDLE) return;
    m_ImpostorPipe = VK::GfxPipelineBuilder(m_ImpostorLayout)   // no vertex input — gl_VertexIndex quad
        .Vert(vs).Frag(fs)
        .Cull(VK_CULL_MODE_NONE)
        .DynamicDepthBias()
        .Depth(true, true)
        .DepthTarget(VK_FORMAT_D16_UNORM)
        .Build("Trees impostor");
    if (m_ImpostorPipe == VK_NULL_HANDLE) return;

    // Indirect builder compute: set {0 src (dyn indirect), 1 dst (impostor indirect)}.
    { m_ImpostorCmdSetL = VK::MakeSetLayout({ kSSBO, kSSBO }, VK_SHADER_STAGE_COMPUTE_BIT, "Trees.ImpostorCmd");
      if (m_ImpostorCmdSetL == VK_NULL_HANDLE) return;
      if (!VK::AllocSets(m_ImpostorPool, m_ImpostorCmdSetL, 1, &m_ImpostorCmdSet, "Trees.ImpostorCmd")) return;
      VK::DescriptorWriter(m_ImpostorCmdSet)
          .StorageBuffer(0, m_VsmDynIndirect->GetHandle())
          .StorageBuffer(1, m_VsmImpostorDynIndirect->GetHandle())
          .Flush();
      m_ImpostorCmdLayout = VK::MakePipelineLayout({ m_ImpostorCmdSetL }, sizeof(u32));
      if (m_ImpostorCmdLayout == VK_NULL_HANDLE) return;
      m_ImpostorCmdPipe = VK::CreateComputePipeline(cs, m_ImpostorCmdLayout, "Trees.ImpostorCmd");
      if (m_ImpostorCmdPipe == VK_NULL_HANDLE) return; }

    m_ImpostorReady = true;
    Msg("[VK Trees] impostor crown shadows ready (%u meshes, %u facets, scale %.2f)", m_ImpostorMeshCount, kImpFacets, ps_r_vsm_tree_impostor_scale);
}

// Build m_VsmImpostorDynIndirect from the DYN crown bin's m_VsmDynIndirect (indexCount→6,
// non-indexed). Records a compute→compute barrier on the source; the draw-indirect
// hazard on the DST is covered by vk_vsm's global SHADER_WRITE→INDIRECT barrier after
// VsmBinDyn. Call at the end of VsmBinDyn (outside the atlas render pass).
void CTreeManager::DispatchImpostorCmd(VkCommandBuffer cmd)
{
    if (!IsImpostorMode() || m_ImpostorCmdPipe == VK_NULL_HANDLE) return;
    VK::BufferBarrier(cmd, m_VsmDynIndirect->GetHandle(),
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ImpostorCmdPipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_ImpostorCmdLayout, 0, 1, &m_ImpostorCmdSet, 0, nullptr);
    u32 count = m_TotalCount;
    vkCmdPushConstants(cmd, m_ImpostorCmdLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(u32), &count);
    vkCmdDispatch(cmd, (m_TotalCount + 63) / 64, 1, 1);
}

void CTreeManager::DestroyImpostor()
{
    VkDevice dev = VulkanHW.m_Device;
    if (dev == VK_NULL_HANDLE) return;
    if (m_ImpostorCmdPipe)   { vkDestroyPipeline(dev, m_ImpostorCmdPipe, nullptr); m_ImpostorCmdPipe = VK_NULL_HANDLE; }
    if (m_ImpostorCmdLayout) { vkDestroyPipelineLayout(dev, m_ImpostorCmdLayout, nullptr); m_ImpostorCmdLayout = VK_NULL_HANDLE; }
    if (m_ImpostorCmdSetL)   { vkDestroyDescriptorSetLayout(dev, m_ImpostorCmdSetL, nullptr); m_ImpostorCmdSetL = VK_NULL_HANDLE; }
    if (m_ImpostorPipe)      { vkDestroyPipeline(dev, m_ImpostorPipe, nullptr); m_ImpostorPipe = VK_NULL_HANDLE; }
    if (m_ImpostorLayout)    { vkDestroyPipelineLayout(dev, m_ImpostorLayout, nullptr); m_ImpostorLayout = VK_NULL_HANDLE; }
    if (m_ImpostorSetL)      { vkDestroyDescriptorSetLayout(dev, m_ImpostorSetL, nullptr); m_ImpostorSetL = VK_NULL_HANDLE; }
    if (m_ImpostorPool)      { vkDestroyDescriptorPool(dev, m_ImpostorPool, nullptr); m_ImpostorPool = VK_NULL_HANDLE; m_ImpostorSet = VK_NULL_HANDLE; m_ImpostorCmdSet = VK_NULL_HANDLE; }
    if (m_VsmImpostorDynIndirect) { m_VsmImpostorDynIndirect->Destroy(); xr_delete(m_VsmImpostorDynIndirect); }
    if (m_ImpostorSilIdx)    { m_ImpostorSilIdx->Destroy(); xr_delete(m_ImpostorSilIdx); }
    if (m_ImpostorSampler)   { vkDestroySampler(dev, m_ImpostorSampler, nullptr); m_ImpostorSampler = VK_NULL_HANDLE; }
    if (m_ImpostorAtlasView) { vkDestroyImageView(dev, m_ImpostorAtlasView, nullptr); m_ImpostorAtlasView = VK_NULL_HANDLE; }
    if (m_ImpostorAtlas)     { VK::Vram::DestroyImage(VulkanHW.m_Allocator, m_ImpostorAtlas, m_ImpostorAtlasAlloc); m_ImpostorAtlas = VK_NULL_HANDLE; m_ImpostorAtlasAlloc = VK_NULL_HANDLE; }
    m_ImpostorMeshCount = 0;
    m_ImpostorReady = false;
}

// ============================================================================
// Crown-HULL shadow casters (r_vsm_tree_hull) — the AC-Shadows-style caster LOD.
// Near trees beyond r_vsm_tree_hull_dist swap the alpha-tested crown MESH for a baked
// low-poly OPAQUE hull (ellipsoid lobes, BuildMeshlets bake): no texture fetch, no
// discard, early-Z — the first lever aimed at the PROVEN alpha-test-fill bound of
// VSM/DynTrees (meshlet/HZB/D16/VRS all missed it). The closest trees keep the full
// crown (dappling perfect where the player looks). Geometry always baked → live A/B.
// ============================================================================
bool CTreeManager::IsHullMode() const
{
    return ps_r_vsm_tree_hull && ps_r_vsm_tree_wind && m_HullReady && m_HullDataReady && !IsImpostorMode();
}

void CTreeManager::CreateHullResources()
{
    if (m_HullReady) return;
    if (!m_HullDataReady || !m_HullVB || !m_HullIB || !m_HullInfoBuffer) return;
    if (m_VsmPageLayout == VK_NULL_HANDLE || !m_VsmDynIndirect || m_TotalCount == 0) return;
    if (!g_ShaderManager) return;
    VkShaderModule vs  = g_ShaderManager->Load("tree_vsm_hull.vert.spv");
    VkShaderModule vsS = g_ShaderManager->Load("tree_vsm_hull_s.vert.spv");
    VkShaderModule fs  = g_ShaderManager->Load("tree_vsm_hull.frag.spv");
    VkShaderModule cs  = g_ShaderManager->Load("vsm_hull_cmd.comp.spv");
    if (!vs || !vsS || !fs || !cs) { Msg("![VK Trees] crown-hull shaders missing — r_vsm_tree_hull disabled"); return; }
    const u32 N = VK_FRAMES_IN_FLIGHT;

    // TRANSFER_DST: joins the lazy-init zero fill (recycled-VRAM garbage in an indirect
    // buffer = billion-instance draw = DEVICE_LOST — the level-transition lesson).
    auto makeIndirect = [&](CVulkanBuffer*& b) {
        b = xr_new<CVulkanBuffer>();
        b->Create((VkDeviceSize)m_TotalCount * sizeof(VkDrawIndexedIndirectCommand),
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    };
    makeIndirect(m_VsmHullIndirect);    // DYN hull tier
    makeIndirect(m_VsmHullIndirectS);   // STATIC far tier (r_vsm_tree_hull 2)

    // Voxel-caster buffers: NON-indexed indirect cmds (4 u32/tree) + the per-frame
    // choice ring (host-visible, CPU fills camera-distance LOD/fade/density per tree).
    auto makeIndirect4 = [&](CVulkanBuffer*& b) {
        b = xr_new<CVulkanBuffer>();
        b->Create((VkDeviceSize)m_TotalCount * sizeof(u32) * 4,
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    };
    makeIndirect4(m_VsmVoxIndirect);
    makeIndirect4(m_VsmVoxIndirectS);
    for (u32 i = 0; i < N; ++i) {
        m_VoxChoiceBuf[i] = xr_new<CVulkanBuffer>();
        m_VoxChoiceBuf[i]->Create((VkDeviceSize)m_TotalCount * sizeof(u32) * 4,
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
        m_VoxChoicePtr[i] = m_VoxChoiceBuf[i]->Map();
    }

    // Own pool (never resize the shared VSM pool — the regression-prone class of change).
    // Cmd-swap compute: {0 dyn crown RW, 1 hullInfo, 2 nearFlags (per-frame ring),
    //                    3 dyn hull W, 4 STATIC crown RW, 5 static hull W,
    //                    6 vox choice (ring), 7 dyn vox W, 8 static vox W}.
    {
        if (!VK::MakeDescriptorSets({ kSSBO, kSSBO, kSSBO, kSSBO, kSSBO, kSSBO, kSSBO, kSSBO, kSSBO },
                                    N, m_HullCmdSetL, m_HullCmdPool, m_HullCmdSet,
                                    VK_SHADER_STAGE_COMPUTE_BIT, "Trees.HullCmd"))
            return;
        m_HullCmdLayout = VK::MakePipelineLayout({ m_HullCmdSetL }, 3 * sizeof(u32));
        if (m_HullCmdLayout == VK_NULL_HANDLE) return;
        m_HullCmdPipe = VK::CreateComputePipeline(cs, m_HullCmdLayout, "Trees.HullCmd");
        if (m_HullCmdPipe == VK_NULL_HANDLE) return;
    }

    // Opaque depth-only hull raster ×2 (dyn/static atlas VS variants). Reuses
    // m_VsmPageLayout (set1 diffuse unused by the shader — stays bound from the crown
    // draws, layout-compatible). Tight vec3 VB.
    auto createHullPipe = [&](VkShaderModule hvs, VkPipeline& out) {
        out = VK::GfxPipelineBuilder(m_VsmPageLayout)
            .Vert(hvs).Frag(fs)
            .Binding(0, 12)
            .Attr(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0)
            .Cull(VK_CULL_MODE_NONE)
            .DynamicDepthBias()
            .Depth(true, true)
            .DepthTarget(VK_FORMAT_D16_UNORM)   // VSM atlas format
            .Build("Trees crown-hull");
    };
    createHullPipe(vs,  m_VsmHullPipe);    // dyn atlas (wind push)
    createHullPipe(vsS, m_VsmHullPipeS);   // static atlas (rigid)
    if (m_VsmHullPipe == VK_NULL_HANDLE || m_VsmHullPipeS == VK_NULL_HANDLE) return;

    // ---- Voxel-BRICK caster (r_vsm_tree_hull_vox): set3 = brick SSBO + choice ring,
    // sets 0-2 identical to m_VsmPageLayout's → binding compatibility with the crown
    // draws is preserved. Pipelines mirror the hull pipe but with NO vertex input (one
    // light-facing quad per brick from gl_VertexIndex, bricks from the SSBO) and the
    // DDA-raycast FS (UE RasterizeBricks model). Failure here only disables the brick
    // caster — the shell path above still works.
    if (m_BrickVB && m_BrickTotal) do {
        VkShaderModule vvs  = g_ShaderManager->Load("tree_vsm_vox.vert.spv");
        VkShaderModule vvsS = g_ShaderManager->Load("tree_vsm_vox_s.vert.spv");
        VkShaderModule vfs  = g_ShaderManager->Load("tree_vsm_vox.frag.spv");
        if (!vvs || !vvsS || !vfs) { Msg("![VK Trees] voxel caster shaders missing — shell fallback"); break; }
        // Stage-2 cull outputs first — the remap list is set3 binding 2, so it must
        // exist before the descriptor writes even if the cull pipeline itself fails.
        {
            auto mkBuf = [&](CVulkanBuffer*& b, VkDeviceSize sz, VkBufferUsageFlags extra) {
                b = xr_new<CVulkanBuffer>();
                b->Create(sz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | extra,
                          VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);   // gpuOnly: GPU-written, no CPU mapping
            };
            mkBuf(m_VoxCullCmdD,  (VkDeviceSize)kVoxCullCmdCap * sizeof(u32) * 4, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
            mkBuf(m_VoxCullCmdS,  (VkDeviceSize)kVoxCullCmdCap * sizeof(u32) * 4, VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
            mkBuf(m_VoxCullList,  (VkDeviceSize)kVoxCullListCap * sizeof(u32), 0);
            mkBuf(m_VoxCullStats, (VkDeviceSize)8 * sizeof(u32), VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);   // words 0/1 double as draw counts; [4]=hzbCulled
            m_VoxCullStatsRB = xr_new<CVulkanBuffer>();
            m_VoxCullStatsRB->Create((VkDeviceSize)8 * sizeof(u32), VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
            m_VoxCullStatsPtr = (u32*)m_VoxCullStatsRB->Map();
            if (m_VoxCullStatsPtr) memset(m_VoxCullStatsPtr, 0, 8 * sizeof(u32));
            for (u32 i = 0; i < N; ++i) {   // compact dispatch list ring (host-visible, CPU fills in VsmBin)
                m_VoxCullTreeList[i] = xr_new<CVulkanBuffer>();
                m_VoxCullTreeList[i]->Create((VkDeviceSize)m_TotalCount * sizeof(u32),
                                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
                m_VoxCullTreeListPtr[i] = m_VoxCullTreeList[i]->Map();
            }
        }
        {
            if (!VK::MakeDescriptorSets({ kSSBO, kSSBO, kSSBO }, N,
                                        m_VoxCasterSetL, m_VoxCasterPool, m_VoxCasterSet,
                                        VK_SHADER_STAGE_VERTEX_BIT, "Trees.VoxCaster"))
                break;
            for (u32 i = 0; i < N; ++i)     // handles are fixed → write once
                VK::DescriptorWriter(m_VoxCasterSet[i])
                    .StorageBuffer(0, m_BrickVB->GetHandle())
                    .StorageBuffer(1, m_VoxChoiceBuf[i]->GetHandle())
                    .StorageBuffer(2, m_VoxCullList->GetHandle())
                    .Flush();
            m_VoxCasterLayout = VK::MakePipelineLayout(
                { m_XformDescLayout, m_TexDescLayout, m_VsmPageSetL, m_VoxCasterSetL }, 64,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
            if (m_VoxCasterLayout == VK_NULL_HANDLE) break;
        }
        auto createVoxPipe = [&](VkShaderModule mvs, VkPipeline& out) {
            out = VK::GfxPipelineBuilder(m_VoxCasterLayout)   // no vertex input
                .Vert(mvs).Frag(vfs)
                .Cull(VK_CULL_MODE_NONE)
                .DynamicDepthBias()
                .Depth(true, true)
                .DepthTarget(VK_FORMAT_D16_UNORM)   // VSM atlas format
                .Build("Trees voxel caster");
        };
        createVoxPipe(vvs,  m_VsmVoxPipe);
        createVoxPipe(vvsS, m_VsmVoxPipeS);

        // ---- Stage-2 per-page brick cull (vsm_vox_cull.comp): 12 storage + 1 UBO.
        // Failure only disables the compaction — the un-culled instanced path draws.
        do {
            VkShaderModule cs = g_ShaderManager->Load("vsm_vox_cull.comp.spv");
            if (!cs) { Msg("![VK Trees] vsm_vox_cull.comp.spv missing — un-culled brick path"); break; }
            if (!VK::MakeDescriptorSets({ kSSBO, kSSBO, kSSBO, kSSBO, kSSBO, kSSBO, kSSBO, kSSBO,
                                          kUBO,                                        // 8 = clipmap UBO
                                          kSSBO, kSSBO, kSSBO, kSSBO, kSSBO, kSSBO, kSSBO, kSSBO },
                                        N, m_VoxCullSetL, m_VoxCullPool, m_VoxCullSet,
                                        VK_SHADER_STAGE_COMPUTE_BIT, "Trees.VoxCull"))
                break;
            // push (32 B): count, mode, cmdCap, listCap, slop, hzbOn, margin, rmaskOn
            m_VoxCullLayout = VK::MakePipelineLayout({ m_VoxCullSetL }, 32);
            if (m_VoxCullLayout == VK_NULL_HANDLE) break;
            m_VoxCullPipe = VK::CreateComputePipeline(cs, m_VoxCullLayout, "Trees.VoxCull");
            if (m_VoxCullPipe == VK_NULL_HANDLE) {
                Msg("![VK Trees] vox-cull pipeline create failed — un-culled brick path"); break;
            }
            m_VoxCullReady = true;
        } while (false);
    } while (false);

    // Debug overlay (r_vsm_tree_hull_debug): translucent PINK lobes over hull-tier trees
    // in the forward view — makes the r_vsm_tree_hull_dist boundary visible in-world.
    // Rides the forward tree pass: same layout (TreeGfxPush + set0 xform/waves already
    // bound), scene color + depth formats, alpha blend, LEQUAL test, no depth write.
    if (m_GfxPipelineLayout != VK_NULL_HANDLE) {
        VkShaderModule dvs = g_ShaderManager->Load("tree_hull_debug.vert.spv");
        VkShaderModule dfs = g_ShaderManager->Load("tree_hull_debug.frag.spv");
        if (dvs && dfs) {
            m_HullDebugPipe = VK::GfxPipelineBuilder(m_GfxPipelineLayout)
                .Vert(dvs).Frag(dfs)
                .Binding(0, 12)
                .Attr(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0)
                .Cull(VK_CULL_MODE_NONE)
                .Depth(true, false)
                .Color(VK::SceneColor::Format()).BlendAlpha()
                .DepthTarget(Swapchain.m_DepthFormat)
                .Build("Trees crown-hull debug");
        }

        // Voxel-CLOUD viewmode pipeline (tree_voxel_debug.{vert,frag}) — individual colored
        // cubes, instanced: binding 0 is the GpuTreeVoxel stream at PER-INSTANCE rate
        // (center vec3 + packed color uint), no per-vertex attributes (corners from
        // gl_VertexIndex). OPAQUE with depth-write — cubes are solid geometry replacing the
        // crown visually, and blending unsorted instances would z-fight anyway.
        VkShaderModule vvs = g_ShaderManager->Load("tree_voxel_debug.vert.spv");
        VkShaderModule vfs = g_ShaderManager->Load("tree_voxel_debug.frag.spv");
        if (vvs && vfs) {
            m_VoxDebugPipe = VK::GfxPipelineBuilder(m_GfxPipelineLayout)
                .Vert(vvs).Frag(vfs)
                .Binding(0, (u32)sizeof(GpuTreeVoxel), VK_VERTEX_INPUT_RATE_INSTANCE)
                .Attr(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0)    // aCenter
                .Attr(1, 0, VK_FORMAT_R32_UINT,        12)    // aColor
                .Cull(VK_CULL_MODE_NONE)
                .Depth(true, true)
                .Color(VK::SceneColor::Format())
                .DepthTarget(Swapchain.m_DepthFormat)
                .Build("Trees crown voxel-cloud");
        }
    }

    m_HullReady = true;
    Msg("[VK Trees] crown-hull caster LOD ready (%u hull tris, %u voxels, dist %.0fm, %s)",
        m_HullLobeTotal, m_VoxTotal, ps_r_vsm_tree_hull_dist, ps_r_vsm_tree_hull ? "ON" : "off");
}

// Swap crown→hull cmds for hull-tier trees. Called at the end of VsmBinDyn: the barrier
// covers stage-1-bin-write → this read/rewrite; the later stage-1→stage-2 barrier orders
// the zeroed counts before the meshlet refinement, and vk_vsm's global barrier before
// RenderAtlas covers the DRAW_INDIRECT reads of both buffers.
void CTreeManager::DispatchHullCmd(VkCommandBuffer cmd)
{
    if (!IsHullMode() || m_HullCmdPipe == VK_NULL_HANDLE || !m_VsmHullIndirect || !m_VsmHullIndirectS) return;
    const u32 slot = m_VsmSlot;
    // One global compute→compute barrier: orders BOTH stage-1 tree bins' indirect writes
    // (static + dyn — both already recorded; this runs at the end of VsmBinDyn) before
    // this dispatch's read-and-rewrite of them.
    VkMemoryBarrier mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
                        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, nullptr, 0, nullptr);
    VK::DescriptorWriter(m_HullCmdSet[slot])
        .StorageBuffer(0, m_VsmDynIndirect->GetHandle())
        .StorageBuffer(1, m_HullInfoBuffer->GetHandle())
        .StorageBuffer(2, m_VsmNearFlags[slot]->GetHandle())
        .StorageBuffer(3, m_VsmHullIndirect->GetHandle())
        .StorageBuffer(4, m_VsmIndirect->GetHandle())          // STATIC crown cmds (far tier)
        .StorageBuffer(5, m_VsmHullIndirectS->GetHandle())     // static hull cmds
        .StorageBuffer(6, m_VoxChoiceBuf[slot]->GetHandle())   // voxel choice (ring)
        .StorageBuffer(7, m_VsmVoxIndirect->GetHandle())       // dyn voxel cmds
        .StorageBuffer(8, m_VsmVoxIndirectS->GetHandle())      // static voxel cmds
        .Flush();
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_HullCmdPipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_HullCmdLayout, 0, 1, &m_HullCmdSet[slot], 0, nullptr);
    const bool voxCast = ps_r_vsm_tree_hull_vox > 0 && m_VsmVoxPipe != VK_NULL_HANDLE && m_BrickTotal > 0;
    // voxOn bits: 1 = dyn tier uses bricks, 2 = STATIC tier too (default off — the far
    // shell is per-page-cull-free and measured cheaper on living-sun cache refreshes;
    // bricks lack the meshlet-style stage-2 the crowns have).
    struct { u32 count, farOn, voxOn; } push{ m_TotalCount, (u32)(ps_r_vsm_tree_hull >= 2 ? 1 : 0),
        (u32)(voxCast ? (1u | (ps_r_vsm_tree_hull_vox_static ? 2u : 0u)) : 0u) };
    vkCmdPushConstants(cmd, m_HullCmdLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd, (m_TotalCount + 63) / 64, 1, 1);
}

// Stage-2 per-page brick cull (see vk_TreeManager.h / vsm_vox_cull.comp.glsl). Runs
// once per frame from VsmBinMeshlets — the caller's stage-1→stage-2 barrier already
// orders vsm_hull_cmd's brick-cmd writes before this pass's reads; the cull's own
// writes (compacted cmds + remap list) are ordered before the atlas draws by vk_vsm's
// global pre-RenderAtlas barrier, same as the meshlet stage-2 outputs.
void CTreeManager::DispatchVoxCull(VkCommandBuffer cmd, VkBuffer pageListDyn, VkBuffer pageListStatic, VkBuffer clipmapUBO,
                                   VkBuffer pageMaxBlk, VkBuffer staticPT)
{
    m_VoxCullActive = false;
    if (!m_VoxCullReady || !ps_r_vsm_tree_hull_vox_cull || ps_r_vsm_tree_hull_vox <= 0) return;
    if (!IsHullMode() || !m_BrickTotal || !m_VsmVoxIndirect || !m_VsmVoxIndirectS) return;
    if (pageListDyn == VK_NULL_HANDLE || clipmapUBO == VK_NULL_HANDLE) return;
    // Compact dispatch: no tree carries a live brick choice → every brick cmd is zero,
    // nothing to compact — skip both dispatches (DrawVoxCasters skips its draw too).
    if (m_VoxCullTreeN == 0 || !m_VoxCullTreeList[m_VsmSlot]) return;
    if (pageListStatic == VK_NULL_HANDLE) pageListStatic = pageListDyn;   // layout parity (static cmds empty then anyway)
    // shadow-HZB occlusion (r_vsm_hzb): both buffers or nothing; layout parity when off.
    const bool hzb = pageMaxBlk != VK_NULL_HANDLE && staticPT != VK_NULL_HANDLE;
    if (pageMaxBlk == VK_NULL_HANDLE) pageMaxBlk = m_VoxCullStats->GetHandle();
    if (staticPT == VK_NULL_HANDLE)   staticPT   = m_VoxCullStats->GetHandle();
    const u32 slot = m_VsmSlot;
    VkBuffer rmask = VK::VSM::GetRMaskHandle();
    const bool rmaskOn = ps_r_vsm_rmask && rmask != VK_NULL_HANDLE;
    if (rmask == VK_NULL_HANDLE) rmask = m_VoxCullStats->GetHandle();   // layout parity when off
    VK::DescriptorWriter(m_VoxCullSet[slot])
        .StorageBuffer(0,  m_VsmVoxIndirect->GetHandle())
        .StorageBuffer(1,  m_VsmVoxIndirectS->GetHandle())
        .StorageBuffer(2,  m_VoxChoiceBuf[slot]->GetHandle())
        .StorageBuffer(3,  m_BrickVB->GetHandle())
        .StorageBuffer(4,  m_TreeTransformsBuffer->GetHandle())
        .StorageBuffer(5,  m_VsmCasterPages->GetHandle())
        .StorageBuffer(6,  pageListDyn)
        .StorageBuffer(7,  pageListStatic)
        .UniformBuffer(8,  clipmapUBO)
        .StorageBuffer(9,  m_VoxCullCmdD->GetHandle())
        .StorageBuffer(10, m_VoxCullCmdS->GetHandle())
        .StorageBuffer(11, m_VoxCullList->GetHandle())
        .StorageBuffer(12, m_VoxCullStats->GetHandle())
        .StorageBuffer(13, pageMaxBlk)                            // shadow-HZB block maxes
        .StorageBuffer(14, staticPT)                              // virtual page -> static slot
        .StorageBuffer(15, rmask)                                 // receiver mask (r_vsm_rmask, dyn tier)
        .StorageBuffer(16, m_VoxCullTreeList[slot]->GetHandle())  // compact tree list (this frame's ring slot)
        .Flush();
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_VoxCullPipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_VoxCullLayout, 0, 1, &m_VoxCullSet[slot], 0, nullptr);
    // count = LIST length (one workgroup per listed tree), not the level's tree count.
    struct { u32 count, mode, cmdCap, listCap; float slop; u32 hzbOn; float margin; u32 rmaskOn; } push{
        m_VoxCullTreeN, 0u, kVoxCullCmdCap, kVoxCullListCap, 0.75f, (u32)(hzb ? 1 : 0), ps_r_vsm_hzb_margin, (u32)(rmaskOn ? 1 : 0) };
    vkCmdPushConstants(cmd, m_VoxCullLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(cmd, m_VoxCullTreeN, 1, 1);
    if (ps_r_vsm_tree_hull_vox_static) {   // static tier casts bricks → compact it too (rigid: no wind slop)
        push.mode = 1u; push.slop = 0.f;
        vkCmdPushConstants(cmd, m_VoxCullLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(cmd, m_VoxCullTreeN, 1, 1);
    }
    m_VoxCullActive = true;
}

// ONE multi-draw of the tier's trees over its hull indirect (non-hull trees carry
// instanceCount 0 → free). DYN pass: live-wind push (both tiers sway coherently),
// set2 = dyn page data. STATIC pass (r_vsm_tree_hull 2): zero wind → rigid, cache-
// stable pages, set2 = static page data. set1 (diffuse) is layout-required but unused
// by the hull shaders — bind any valid set.
void CTreeManager::DrawHulls(VkCommandBuffer cmd, u32 slot, bool staticPass)
{
    VkPipeline pipe = staticPass ? m_VsmHullPipeS : m_VsmHullPipe;
    CVulkanBuffer* ind = staticPass ? m_VsmHullIndirectS : m_VsmHullIndirect;
    if (pipe == VK_NULL_HANDLE || !ind || !m_HullVB || !m_HullIB) return;
    struct VsmPagePush {
        float    uvScale, alphaRef; u32 cap, pad;
        Fvector4 wind_params, wsetup_trees, wind_anim;
    } pc{};
    pc.uvScale = 1.0f / 2048.0f; pc.alphaRef = 200.0f / 255.0f; pc.cap = kVsmTreeCap;
    if (!staticPass) {
        pc.wsetup_trees.set(ps_r_wind_tree_anim, ps_r_wind_tree_trunk, ps_r_wind_tree_bend, 0.1f);
        if (g_pGamePersistent) {
            const Fvector3 wa = g_pGamePersistent->Environment().wind_anim;
            pc.wind_anim.set(wa.x, wa.y, wa.z, ps_r_wind_tree_flutter);
            if (auto* E = g_pGamePersistent->Environment().CurrentEnv)
                pc.wind_params.set(E->wind_direction, E->wind_velocity, 0.0f, 0.0f);
        }
        pc.wind_params.w = ps_r_wind_tree_crown;   // flutter height gate — match the crown pass
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_VsmPageLayout, 0, 1, &m_XformDescSet, 0, nullptr);
    if (!m_TexDescSets.empty())
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_VsmPageLayout, 1, 1, &m_TexDescSets[0], 0, nullptr);
    VkDescriptorSet pageSet = staticPass ? m_VsmPageSet[slot] : m_VsmDynPageSet[slot];
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_VsmPageLayout, 2, 1, &pageSet, 0, nullptr);
    vkCmdPushConstants(cmd, m_VsmPageLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
    VkDeviceSize vbOff = 0;
    VkBuffer hvb = m_HullVB->GetHandle();
    vkCmdBindVertexBuffers(cmd, 0, 1, &hvb, &vbOff);
    vkCmdBindIndexBuffer(cmd, m_HullIB->GetHandle(), 0, VK_INDEX_TYPE_UINT16);
    vkCmdDrawIndexedIndirect(cmd, ind->GetHandle(), 0, m_TotalCount, sizeof(VkDrawIndexedIndirectCommand));
}

// Voxel-cloud caster multi-draw — same shape as DrawHulls but NON-indexed (cube corners
// from gl_VertexIndex, voxels/choice from set3). Trees not chosen by vsm_hull_cmd carry
// instanceCount 0 → free. Sets 0-2 are re-bound with the 4-set layout (identical layout
// objects → the crown draws that follow keep their own bindings valid).
void CTreeManager::DrawVoxCasters(VkCommandBuffer cmd, u32 slot, bool staticPass)
{
    if (ps_r_vsm_tree_hull_vox <= 0) return;
    // No tree carries a live brick choice → vsm_hull_cmd wrote every vox cmd with
    // instanceCount 0 this frame — skip the whole multi-draw (binds + N zero-draws).
    if (m_VoxCastCount == 0) return;
    VkPipeline pipe = staticPass ? m_VsmVoxPipeS : m_VsmVoxPipe;
    CVulkanBuffer* ind = staticPass ? m_VsmVoxIndirectS : m_VsmVoxIndirect;
    if (pipe == VK_NULL_HANDLE || !ind || m_VoxCasterLayout == VK_NULL_HANDLE) return;
    // Stage-2 compacted stream (pad=1 → VS fetches bricks through the remap list).
    const bool cull = m_VoxCullActive && m_VoxCullCmdD && m_VoxCullStats;
    struct VsmPagePush {
        float    uvScale, alphaRef; u32 cap, pad;
        Fvector4 wind_params, wsetup_trees, wind_anim;
    } pc{};
    pc.uvScale = 1.0f / 2048.0f; pc.alphaRef = 200.0f / 255.0f; pc.cap = kVsmTreeCap;
    pc.pad = cull ? 1u : 0u;
    if (!staticPass) {
        pc.wsetup_trees.set(ps_r_wind_tree_anim, ps_r_wind_tree_trunk, ps_r_wind_tree_bend, 0.1f);
        if (g_pGamePersistent) {
            const Fvector3 wa = g_pGamePersistent->Environment().wind_anim;
            pc.wind_anim.set(wa.x, wa.y, wa.z, ps_r_wind_tree_flutter);
            if (auto* E = g_pGamePersistent->Environment().CurrentEnv)
                pc.wind_params.set(E->wind_direction, E->wind_velocity, 0.0f, 0.0f);
        }
        pc.wind_params.w = ps_r_wind_tree_crown;
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_VoxCasterLayout, 0, 1, &m_XformDescSet, 0, nullptr);
    if (!m_TexDescSets.empty())
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_VoxCasterLayout, 1, 1, &m_TexDescSets[0], 0, nullptr);
    VkDescriptorSet pageSet = staticPass ? m_VsmPageSet[slot] : m_VsmDynPageSet[slot];
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_VoxCasterLayout, 2, 1, &pageSet, 0, nullptr);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_VoxCasterLayout, 3, 1, &m_VoxCasterSet[slot], 0, nullptr);
    vkCmdPushConstants(cmd, m_VoxCasterLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
    if (cull) {
        CVulkanBuffer* cc = staticPass ? m_VoxCullCmdS : m_VoxCullCmdD;
        vkCmdDrawIndirectCount(cmd, cc->GetHandle(), 0, m_VoxCullStats->GetHandle(),
                               staticPass ? sizeof(u32) : 0, kVoxCullCmdCap, sizeof(u32) * 4);
    } else
        vkCmdDrawIndirect(cmd, ind->GetHandle(), 0, m_TotalCount, sizeof(u32) * 4);
}

void CTreeManager::DestroyHull()
{
    VkDevice dev = VulkanHW.m_Device;
    if (dev == VK_NULL_HANDLE) return;
    if (m_VsmHullPipe)   { vkDestroyPipeline(dev, m_VsmHullPipe, nullptr); m_VsmHullPipe = VK_NULL_HANDLE; }
    if (m_VsmHullPipeS)  { vkDestroyPipeline(dev, m_VsmHullPipeS, nullptr); m_VsmHullPipeS = VK_NULL_HANDLE; }
    if (m_HullDebugPipe) { vkDestroyPipeline(dev, m_HullDebugPipe, nullptr); m_HullDebugPipe = VK_NULL_HANDLE; }
    if (m_VoxDebugPipe)  { vkDestroyPipeline(dev, m_VoxDebugPipe,  nullptr); m_VoxDebugPipe  = VK_NULL_HANDLE; }
    if (m_HullCmdPipe)   { vkDestroyPipeline(dev, m_HullCmdPipe, nullptr); m_HullCmdPipe = VK_NULL_HANDLE; }
    if (m_HullCmdLayout) { vkDestroyPipelineLayout(dev, m_HullCmdLayout, nullptr); m_HullCmdLayout = VK_NULL_HANDLE; }
    if (m_HullCmdSetL)   { vkDestroyDescriptorSetLayout(dev, m_HullCmdSetL, nullptr); m_HullCmdSetL = VK_NULL_HANDLE; }
    if (m_HullCmdPool)   { vkDestroyDescriptorPool(dev, m_HullCmdPool, nullptr); m_HullCmdPool = VK_NULL_HANDLE;
                           for (u32 i = 0; i < VK_FRAMES_IN_FLIGHT; ++i) m_HullCmdSet[i] = VK_NULL_HANDLE; }
    if (m_VsmHullIndirect)  { m_VsmHullIndirect->Destroy(); xr_delete(m_VsmHullIndirect); }
    if (m_VsmHullIndirectS) { m_VsmHullIndirectS->Destroy(); xr_delete(m_VsmHullIndirectS); }
    // ----- Voxel-cloud caster resources.
    if (m_VsmVoxPipe)      { vkDestroyPipeline(dev, m_VsmVoxPipe, nullptr); m_VsmVoxPipe = VK_NULL_HANDLE; }
    if (m_VsmVoxPipeS)     { vkDestroyPipeline(dev, m_VsmVoxPipeS, nullptr); m_VsmVoxPipeS = VK_NULL_HANDLE; }
    if (m_VoxCasterLayout) { vkDestroyPipelineLayout(dev, m_VoxCasterLayout, nullptr); m_VoxCasterLayout = VK_NULL_HANDLE; }
    if (m_VoxCasterPool)   { vkDestroyDescriptorPool(dev, m_VoxCasterPool, nullptr); m_VoxCasterPool = VK_NULL_HANDLE;
                             for (u32 i = 0; i < VK_FRAMES_IN_FLIGHT; ++i) m_VoxCasterSet[i] = VK_NULL_HANDLE; }
    if (m_VoxCasterSetL)   { vkDestroyDescriptorSetLayout(dev, m_VoxCasterSetL, nullptr); m_VoxCasterSetL = VK_NULL_HANDLE; }
    if (m_VsmVoxIndirect)  { m_VsmVoxIndirect->Destroy(); xr_delete(m_VsmVoxIndirect); }
    if (m_VsmVoxIndirectS) { m_VsmVoxIndirectS->Destroy(); xr_delete(m_VsmVoxIndirectS); }
    // ----- Stage-2 brick-cull resources.
    m_VoxCullReady = m_VoxCullActive = false;
    if (m_VoxCullPipe)   { vkDestroyPipeline(dev, m_VoxCullPipe, nullptr); m_VoxCullPipe = VK_NULL_HANDLE; }
    if (m_VoxCullLayout) { vkDestroyPipelineLayout(dev, m_VoxCullLayout, nullptr); m_VoxCullLayout = VK_NULL_HANDLE; }
    if (m_VoxCullPool)   { vkDestroyDescriptorPool(dev, m_VoxCullPool, nullptr); m_VoxCullPool = VK_NULL_HANDLE;
                           for (u32 i = 0; i < VK_FRAMES_IN_FLIGHT; ++i) m_VoxCullSet[i] = VK_NULL_HANDLE; }
    if (m_VoxCullSetL)   { vkDestroyDescriptorSetLayout(dev, m_VoxCullSetL, nullptr); m_VoxCullSetL = VK_NULL_HANDLE; }
    if (m_VoxCullCmdD)   { m_VoxCullCmdD->Destroy();  xr_delete(m_VoxCullCmdD); }
    if (m_VoxCullCmdS)   { m_VoxCullCmdS->Destroy();  xr_delete(m_VoxCullCmdS); }
    if (m_VoxCullList)   { m_VoxCullList->Destroy();  xr_delete(m_VoxCullList); }
    if (m_VoxCullStats)  { m_VoxCullStats->Destroy(); xr_delete(m_VoxCullStats); }
    if (m_VoxCullStatsRB) { m_VoxCullStatsPtr = nullptr; m_VoxCullStatsRB->Destroy(); xr_delete(m_VoxCullStatsRB); }
    for (u32 i = 0; i < VK_FRAMES_IN_FLIGHT; ++i) {
        if (m_VoxCullTreeList[i]) { m_VoxCullTreeList[i]->Destroy(); xr_delete(m_VoxCullTreeList[i]); }
        m_VoxCullTreeListPtr[i] = nullptr;
    }
    m_VoxCullTreeN = 0;
    m_VoxCastListCPU.clear();
    for (u32 i = 0; i < VK_FRAMES_IN_FLIGHT; ++i) {
        if (m_VoxChoiceBuf[i]) { m_VoxChoiceBuf[i]->Destroy(); xr_delete(m_VoxChoiceBuf[i]); }
        m_VoxChoicePtr[i] = nullptr;
    }
    m_VoxChoiceCPU.clear();
    m_VoxLodSizeCPU.clear();
    // m_HullVB/IB/InfoBuffer are Build()-owned (bake data) — freed in CTreeManager::Destroy.
    m_HullReady = false;
}

void CTreeManager::DestroyVsm()
{
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
    VkDevice dev = VulkanHW.m_Device;
    DestroyImpostor();   // impostor pipelines reference m_VsmPageSetL — tear down before it
    DestroyHull();       // hull pipelines reference m_VsmPageLayout — same ordering rule
    if (m_VsmPagePipe24) { vkDestroyPipeline(dev, m_VsmPagePipe24, nullptr); m_VsmPagePipe24 = VK_NULL_HANDLE; }
    if (m_VsmPagePipe28) { vkDestroyPipeline(dev, m_VsmPagePipe28, nullptr); m_VsmPagePipe28 = VK_NULL_HANDLE; }
    if (m_VsmPageDynPipe24) { vkDestroyPipeline(dev, m_VsmPageDynPipe24, nullptr); m_VsmPageDynPipe24 = VK_NULL_HANDLE; }
    if (m_VsmPageDynPipe28) { vkDestroyPipeline(dev, m_VsmPageDynPipe28, nullptr); m_VsmPageDynPipe28 = VK_NULL_HANDLE; }
    if (m_VsmPageLayout) { vkDestroyPipelineLayout(dev, m_VsmPageLayout, nullptr); m_VsmPageLayout = VK_NULL_HANDLE; }
    if (m_VsmBinPipe)    { vkDestroyPipeline(dev, m_VsmBinPipe, nullptr); m_VsmBinPipe = VK_NULL_HANDLE; }
    if (m_VsmBinLayout)  { vkDestroyPipelineLayout(dev, m_VsmBinLayout, nullptr); m_VsmBinLayout = VK_NULL_HANDLE; }
    if (m_VsmDescPool)   { vkDestroyDescriptorPool(dev, m_VsmDescPool, nullptr); m_VsmDescPool = VK_NULL_HANDLE;
                           for (u32 i = 0; i < VK_FRAMES_IN_FLIGHT; ++i) { m_VsmBinSet[i] = VK_NULL_HANDLE; m_VsmDynBinSet[i] = VK_NULL_HANDLE; m_VsmPageSet[i] = VK_NULL_HANDLE; m_VsmDynPageSet[i] = VK_NULL_HANDLE; } }
    if (m_VsmBinSetL)    { vkDestroyDescriptorSetLayout(dev, m_VsmBinSetL, nullptr); m_VsmBinSetL = VK_NULL_HANDLE; }
    if (m_VsmPageSetL)   { vkDestroyDescriptorSetLayout(dev, m_VsmPageSetL, nullptr); m_VsmPageSetL = VK_NULL_HANDLE; }
    if (m_VsmCasterPages) { m_VsmCasterPages->Destroy(); xr_delete(m_VsmCasterPages); }
    if (m_VsmIndirect)    { m_VsmIndirect->Destroy(); xr_delete(m_VsmIndirect); }
    if (m_VsmDynIndirect) { m_VsmDynIndirect->Destroy(); xr_delete(m_VsmDynIndirect); }
    if (m_VsmStats)       { m_VsmStats->Destroy(); xr_delete(m_VsmStats); }
    if (m_VsmStatsRB)     { m_VsmStatsRB->Destroy(); xr_delete(m_VsmStatsRB); }
    m_VsmStatsPtr = nullptr;
    if (m_VsmTreeBits)    { m_VsmTreeBits->Destroy(); xr_delete(m_VsmTreeBits); }
    if (m_VsmTreeBitsRB)  { m_VsmTreeBitsRB->Destroy(); xr_delete(m_VsmTreeBitsRB); }
    m_VsmTreeBitsPtr = nullptr;
    for (u32 i = 0; i < VK_FRAMES_IN_FLIGHT; ++i) {
        if (m_VsmNearFlags[i]) { m_VsmNearFlags[i]->Destroy(); xr_delete(m_VsmNearFlags[i]); }
        m_VsmNearPtr[i] = nullptr;
    }
    m_VsmNearCPU.clear(); m_VsmPendingInval.clear();

    // ----- Phase B meshlet-cull resources.
    if (m_MeshletBinPipe)   { vkDestroyPipeline(dev, m_MeshletBinPipe, nullptr); m_MeshletBinPipe = VK_NULL_HANDLE; }
    if (m_MeshletBinLayout) { vkDestroyPipelineLayout(dev, m_MeshletBinLayout, nullptr); m_MeshletBinLayout = VK_NULL_HANDLE; }
    if (m_MeshletPagePipe24)    { vkDestroyPipeline(dev, m_MeshletPagePipe24, nullptr); m_MeshletPagePipe24 = VK_NULL_HANDLE; }
    if (m_MeshletPagePipe28)    { vkDestroyPipeline(dev, m_MeshletPagePipe28, nullptr); m_MeshletPagePipe28 = VK_NULL_HANDLE; }
    if (m_MeshletPageDynPipe24) { vkDestroyPipeline(dev, m_MeshletPageDynPipe24, nullptr); m_MeshletPageDynPipe24 = VK_NULL_HANDLE; }
    if (m_MeshletPageDynPipe28) { vkDestroyPipeline(dev, m_MeshletPageDynPipe28, nullptr); m_MeshletPageDynPipe28 = VK_NULL_HANDLE; }
    if (m_MeshletDescPool)  { vkDestroyDescriptorPool(dev, m_MeshletDescPool, nullptr); m_MeshletDescPool = VK_NULL_HANDLE;
                              for (u32 i = 0; i < VK_FRAMES_IN_FLIGHT; ++i) { m_MeshletBinSet[i] = VK_NULL_HANDLE; m_MeshletDynBinSet[i] = VK_NULL_HANDLE; } }
    if (m_MeshletBinSetL)   { vkDestroyDescriptorSetLayout(dev, m_MeshletBinSetL, nullptr); m_MeshletBinSetL = VK_NULL_HANDLE; }
    auto delBuf = [](CVulkanBuffer*& b) { if (b) { b->Destroy(); xr_delete(b); } };
    delBuf(m_VsmMeshletCmd); delBuf(m_VsmMeshletCmdDyn);
    delBuf(m_VsmMeshletGroupCount); delBuf(m_VsmMeshletGroupCountDyn);
    delBuf(m_TreeGroupBuffer); delBuf(m_GroupInfoBuffer);
    delBuf(m_MeshletStats); delBuf(m_MeshletStatsRB);
    m_MeshletStatsPtr = nullptr;
    m_MeshletGroupBase.clear(); m_MeshletGroupCap.clear();
    m_MeshletVsmReady = false;

    m_VsmReady = false; m_VsmSlot = 0;
}

}  // namespace VK
