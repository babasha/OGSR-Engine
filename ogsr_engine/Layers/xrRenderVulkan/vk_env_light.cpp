// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan — shared per-frame environment lighting UBO. See vk_env_light.h.
#include "stdafx.h"
#include "vk_env_light.h"
#include "vk_buffer.h"                     // CVulkanBuffer
#include "vk_command_buffer.h"             // CVulkanCommandManager::FRAMES_IN_FLIGHT
#include "vk_shadow.h"                     // ShadowMap (binding 1 = shadow map, sun_vp)
#include "vk_texture.h"                    // CVulkanTexture (fallback ambient cube)
#include "vk_pass_sky.h"                   // SkyPass::AcquireAmbientCubes (hemisphere sky ambient)
#include "vk_pass_ssao.h"                  // GTAO result (binding 8, white fallback until ready)
#include "../../xr_3da/IGame_Persistent.h" // g_pGamePersistent->Environment()
#include "../../xr_3da/Environment.h"      // CEnvDescriptorMixer (sun_dir/sun_color/hemi/ambient)
#include "../../xr_3da/device.h"           // Device.vCameraPosition (light collection)

#include <cstring>

extern int   ps_r_ssao_debug;     // r_ssao_debug — draw the raw AO map (vk_console_min.cpp)
extern float ps_r_ssao_strength;  // r_ssao_strength — live AO depth knob

namespace VK { namespace EnvLight {

namespace {
    constexpr u32 kFramesInFlight = CVulkanCommandManager::FRAMES_IN_FLIGHT;

    // Per-slot stride: descriptor offsets must honour minUniformBufferOffsetAlignment
    // (64 on NVIDIA, up to 256 elsewhere). sizeof(LightUBO) alone is NOT aligned —
    // binding slot 1 at a misaligned offset is undefined behaviour (in practice the
    // GPU read garbage lights on alternate in-flight frames → scene-wide strobing).
    constexpr VkDeviceSize kSlotStride = (sizeof(LightUBO) + 255) & ~VkDeviceSize(255);

    bool                  s_inited     = false;
    bool                  s_failed     = false;
    VkDescriptorSetLayout s_setLayout  = VK_NULL_HANDLE;
    VkDescriptorPool      s_pool       = VK_NULL_HANDLE;
    VkDescriptorSet       s_set[kFramesInFlight] = {};
    CVulkanBuffer         s_ubo;
    u8*                   s_mapped     = nullptr;
    VkDescriptorSet       s_current    = VK_NULL_HANDLE;

    // Hemisphere sky ambient: bindings 6/7 hold the two weather sky cubes
    // (owned by SkyPass). EnvLight owns a trilinear cube sampler + a 1×1×6
    // fallback so the set is always complete (SkyPass::Init runs AFTER us). Each
    // slot tracks which cube views it currently holds; Update() rebinds only on
    // change — fence-safe because Begin() already waited the slot's fence.
    VkSampler             s_cubeSampler = VK_NULL_HANDLE;
    CVulkanTexture*       s_fallbackCube = nullptr;
    VkImageView           s_boundCube0[kFramesInFlight] = {};
    VkImageView           s_boundCube1[kFramesInFlight] = {};

    // GTAO (binding 8): white 1×1 fallback until the SSAO pass produced a
    // frame (SSAOPass::Init runs after us, and the prepass may be off).
    CVulkanTexture*       s_fallbackWhite = nullptr;
    VkImageView           s_boundAO[kFramesInFlight] = {};

    // 2.6 was LDR-era compensation (pre-HDR/auto-exposure) — with the tonemap
    // it flooded terrain so bright the sun shadow under trees washed out
    // ("земля не имеет затенения"). R4's effective ambient scale is ~1.0
    // (hmodel: env_d × hemi, no boost) — and in SHADE ambient is the only
    // light, so any boost here brightens shadowed ground 1:1 vs R4 while the
    // sun hides it elsewhere. 1.1 ≈ R4 shade depth; auto-exposure owns the
    // overall brightness.
    constexpr float kAmbientScale = 1.1f;   // sky-ambient strength knob (tune in-game)
    constexpr float kAmbientLod   = 0.0f;   // sky cubes are BC1/BC3 single-mip → mip 0 (no blur available)
}

VkDescriptorSetLayout GetSetLayout() { return s_setLayout; }
VkDescriptorSet       GetCurrentSet() { return s_current; }

bool Init()
{
    if (s_inited) return !s_failed;
    s_inited = true;

    // The shadow map (binding 1) must exist before we write the descriptor sets.
    ShadowMap::Init();

    // Set layout: binding 0 = UBO, 1 = far sun map, 2 = spot map, 3 = point
    // shadow cube, 4 = sun cascade 0, 5 = sun cascade 1, 6/7 = sky ambient
    // cubes (hemisphere fill), 8 = GTAO. All FRAGMENT, combined samplers.
    VkDescriptorSetLayoutBinding b[9]{};
    b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    b[0].descriptorCount = 1; b[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    for (u32 i = 1; i < 9; ++i) {
        b[i].binding = i; b[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo slci{};
    slci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    slci.bindingCount = 9; slci.pBindings = b;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &slci, nullptr, &s_setLayout) != VK_SUCCESS) {
        Msg("![VK EnvLight] set layout create failed"); s_failed = true; return false;
    }

    VkDescriptorPoolSize ps[2]{
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         kFramesInFlight },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kFramesInFlight * 8 },
    };
    VkDescriptorPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = kFramesInFlight; pci.poolSizeCount = 2; pci.pPoolSizes = ps;
    if (vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &s_pool) != VK_SUCCESS) {
        Msg("![VK EnvLight] pool create failed"); s_failed = true; return false;
    }

    VkDescriptorSetLayout layouts[kFramesInFlight];
    for (u32 i = 0; i < kFramesInFlight; ++i) layouts[i] = s_setLayout;
    VkDescriptorSetAllocateInfo dai{};
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = s_pool; dai.descriptorSetCount = kFramesInFlight; dai.pSetLayouts = layouts;
    if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, s_set) != VK_SUCCESS) {
        Msg("![VK EnvLight] alloc sets failed"); s_failed = true; return false;
    }

    // Host-visible UBO, one aligned LightUBO region per in-flight slot; each set
    // bound to its slice at an alignment-safe offset.
    s_ubo.Create(kSlotStride * kFramesInFlight,
                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
    s_mapped = static_cast<u8*>(s_ubo.Map());
    if (!s_mapped) { Msg("![VK EnvLight] UBO map failed"); s_failed = true; return false; }

    // Trilinear cube sampler (mips → blurred sky = diffuse irradiance) + a
    // neutral 1×1×6 fallback so bindings 6/7 are valid before SkyPass loads the
    // real weather cubes (SkyPass::Init runs after us).
    {
        VkSamplerCreateInfo sci{};
        sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sci.magFilter    = VK_FILTER_LINEAR; sci.minFilter = VK_FILTER_LINEAR;
        sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sci.maxAnisotropy = 1.0f; sci.minLod = 0.0f; sci.maxLod = VK_LOD_CLAMP_NONE;
        if (vkCreateSampler(VulkanHW.m_Device, &sci, nullptr, &s_cubeSampler) != VK_SUCCESS) {
            Msg("![VK EnvLight] cube sampler create failed"); s_failed = true; return false;
        }
    }
    s_fallbackCube = xr_new<CVulkanTexture>();
    s_fallbackCube->m_bCubemap = true; s_fallbackCube->m_ArrayLayers = 6;
    s_fallbackCube->Create(1, 1, VK_FORMAT_R8G8B8A8_UNORM, 1);
    {
        u8 face[24];
        for (u32 i = 0; i < 6; ++i) { face[i*4+0]=128; face[i*4+1]=128; face[i*4+2]=128; face[i*4+3]=255; }
        s_fallbackCube->UploadData(face, sizeof(face));
    }
    const VkImageView fbView = s_fallbackCube->GetView();

    // White 1×1 for binding 8 — "no occlusion" until GTAO renders.
    s_fallbackWhite = xr_new<CVulkanTexture>();
    s_fallbackWhite->Create(1, 1, VK_FORMAT_R8G8B8A8_UNORM, 1);
    {
        const u8 white[4] = { 255, 255, 255, 255 };
        s_fallbackWhite->UploadData(white, sizeof(white));
    }
    const VkImageView fbWhite = s_fallbackWhite->GetView();

    for (u32 i = 0; i < kFramesInFlight; ++i) {
        VkDescriptorBufferInfo bi{ s_ubo.GetHandle(), kSlotStride * i, sizeof(LightUBO) };
        VkDescriptorImageInfo  si[5]{};
        si[0].sampler = ShadowMap::GetSampler(); si[0].imageView = ShadowMap::GetView();           // 1: far sun
        si[1].sampler = ShadowMap::GetSampler(); si[1].imageView = ShadowMap::GetSpotView();       // 2: spot
        si[2].sampler = ShadowMap::GetSampler(); si[2].imageView = ShadowMap::GetPointCubeView();  // 3: point cube
        si[3].sampler = ShadowMap::GetSampler(); si[3].imageView = ShadowMap::GetCascadeView(0);   // 4: sun cascade 0
        si[4].sampler = ShadowMap::GetSampler(); si[4].imageView = ShadowMap::GetCascadeView(1);   // 5: sun cascade 1
        for (auto& s : si) s.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        // Bindings 6/7: sky ambient cubes — start on the neutral fallback;
        // Update() swaps in the real weather cubes once SkyPass has them.
        VkDescriptorImageInfo cube[2]{};
        for (auto& c : cube) {
            c.sampler = s_cubeSampler; c.imageView = fbView;
            c.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
        s_boundCube0[i] = fbView; s_boundCube1[i] = fbView;

        // Binding 8: GTAO — white fallback; Update() swaps in the real AO view.
        VkDescriptorImageInfo aoI{ s_cubeSampler, fbWhite, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        s_boundAO[i] = fbWhite;

        VkWriteDescriptorSet w[9]{};
        w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[0].dstSet = s_set[i]; w[0].dstBinding = 0; w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; w[0].pBufferInfo = &bi;

        // Shadow maps (sun/spot/point/cascades) — ShadowMap::Init ran above, views exist.
        u32 count = 1;
        for (u32 m = 0; m < 5; ++m) {
            if (si[m].imageView == VK_NULL_HANDLE || si[m].sampler == VK_NULL_HANDLE) continue;
            w[count].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[count].dstSet = s_set[i]; w[count].dstBinding = 1 + m; w[count].descriptorCount = 1;
            w[count].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[count].pImageInfo = &si[m];
            ++count;
        }
        for (u32 c = 0; c < 2; ++c) {
            w[count].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[count].dstSet = s_set[i]; w[count].dstBinding = 6 + c; w[count].descriptorCount = 1;
            w[count].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[count].pImageInfo = &cube[c];
            ++count;
        }
        w[count].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[count].dstSet = s_set[i]; w[count].dstBinding = 8; w[count].descriptorCount = 1;
        w[count].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[count].pImageInfo = &aoI;
        ++count;
        vkUpdateDescriptorSets(VulkanHW.m_Device, count, w, 0, nullptr);
    }

    s_current = s_set[0];
    Msg("[VK EnvLight] init OK (UBO %u bytes x %u slots)", (u32)sizeof(LightUBO), kFramesInFlight);
    return true;
}

void Update(u32 slot)
{
    if (s_failed || !s_mapped) return;
    if (slot >= kFramesInFlight) slot = 0;

    // Neutral fallback when env isn't up yet (e.g. main menu before a level).
    LightUBO ub{};
    ub.sun_dir[0]=0.f; ub.sun_dir[1]=-1.f; ub.sun_dir[2]=0.f;
    ub.sun_color[0]=ub.sun_color[1]=ub.sun_color[2]=0.6f;
    ub.hemi_color[0]=ub.hemi_color[1]=ub.hemi_color[2]=0.45f;
    ub.ambient[0]=ub.ambient[1]=ub.ambient[2]=0.05f;
    // Fog fallback: pushed far away → effectively no fog before a level loads.
    ub.fog_params[0]=0.f; ub.fog_params[1]=1e6f; ub.fog_params[2]=1e6f; ub.fog_params[3]=0.f;
    if (g_pGamePersistent) {
        if (auto* E = g_pGamePersistent->Environment().CurrentEnv) {
            ub.sun_dir[0]=E->sun_dir.x;  ub.sun_dir[1]=E->sun_dir.y;  ub.sun_dir[2]=E->sun_dir.z;
            ub.sun_color[0]=E->sun_color.x; ub.sun_color[1]=E->sun_color.y; ub.sun_color[2]=E->sun_color.z;
            ub.hemi_color[0]=E->hemi_color.x; ub.hemi_color[1]=E->hemi_color.y; ub.hemi_color[2]=E->hemi_color.z; ub.hemi_color[3]=E->hemi_color.w;
            ub.ambient[0]=E->ambient.x; ub.ambient[1]=E->ambient.y; ub.ambient[2]=E->ambient.z;

            // Distance fog (R4 cl_fog_params/cl_fog_color binders + combine_1.ps):
            // fog_near/far are derived in CEnvDescriptorMixer (Environment_misc.cpp:
            // near = (1-density)*0.85*dist, far = 0.99*dist); fog_color is the env
            // colour the sky/horizon use, so the haze tints with time-of-day.
            const float fn = E->fog_near, ff = E->fog_far;
            const float r  = (ff > fn + 1e-3f) ? 1.f / (ff - fn) : 0.f;
            ub.fog_params[0] = -fn * r; ub.fog_params[1] = fn; ub.fog_params[2] = ff; ub.fog_params[3] = r;
            ub.fog_color[0] = E->fog_color.x; ub.fog_color[1] = E->fog_color.y; ub.fog_color[2] = E->fog_color.z;

            // One-time dump of the real env magnitudes — to tune the shader balance.
            static bool s_diag = false;
            if (!s_diag) { s_diag = true;
                Msg("[VK Light] sun_color=(%.3f,%.3f,%.3f) hemi=(%.3f,%.3f,%.3f) ambient=(%.3f,%.3f,%.3f) sun_dir=(%.3f,%.3f,%.3f)",
                    E->sun_color.x,E->sun_color.y,E->sun_color.z, E->hemi_color.x,E->hemi_color.y,E->hemi_color.z,
                    E->ambient.x,E->ambient.y,E->ambient.z, E->sun_dir.x,E->sun_dir.y,E->sun_dir.z);
            }
        }
    }
    // Sun light view·proj for the shadow lookup (Pass_SunShadow ran earlier this
    // frame and stored it). Fmatrix is 16 floats row-major → straight copy.
    memcpy(ub.sun_vp, &ShadowMap::GetLightVP(), sizeof(ub.sun_vp));

    // Dynamic point/spot lights (STEP 3): nearest active ones around the camera.
    // Same per-frame result Pass_SunShadow used to render the dynamic shadow maps,
    // so the shadowed indices match the array the shaders iterate.
    const auto& FL = Lights::CollectFrame(Device.vCameraPosition);
    memcpy(ub.lights, FL.gpu, sizeof(ub.lights));
    ub.counts[0] = float(FL.count);
    memcpy(ub.spot_vp, &ShadowMap::GetSpotVP(), sizeof(ub.spot_vp));
    ub.shadow_params[0] = float(FL.spotIdx);
    ub.shadow_params[1] = float(FL.pointIdx);
    ub.shadow_params[2] = 0.f;
    ub.shadow_params[3] = 0.f;
    // Sun cascade VPs — recomputed every frame in Pass_SunShadow (earlier this
    // frame), so receivers always sample the freshly rendered cascade maps.
    memcpy(ub.sun_near_vp, &ShadowMap::GetCascadeVP(0), sizeof(ub.sun_near_vp));
    memcpy(ub.sun_c1_vp,   &ShadowMap::GetCascadeVP(1), sizeof(ub.sun_c1_vp));

    // Camera world position — receivers compute fog distance = length(wp - eye).
    ub.eye_pos[0] = Device.vCameraPosition.x;
    ub.eye_pos[1] = Device.vCameraPosition.y;
    ub.eye_pos[2] = Device.vCameraPosition.z;

    // Hemisphere sky ambient (R4 hmodel): fetch the current weather cubes from
    // SkyPass and rebind this slot's bindings 6/7 if they changed (fence-safe —
    // Begin waited this slot's fence). Weight cross-fades the two cubes.
    float skyWeight = 0.f;
    {
        VkImageView v0 = VK_NULL_HANDLE, v1 = VK_NULL_HANDLE, samp = VK_NULL_HANDLE;
        VkSampler   skSamp = VK_NULL_HANDLE;
        float       w = 0.f;
        if (SkyPass::AcquireAmbientCubes(v0, v1, skSamp, w)) {
            skyWeight = w;
            if (v0 && v1 && (v0 != s_boundCube0[slot] || v1 != s_boundCube1[slot])) {
                VkDescriptorImageInfo cube[2]{};
                cube[0].sampler = s_cubeSampler; cube[0].imageView = v0;
                cube[1].sampler = s_cubeSampler; cube[1].imageView = v1;
                cube[0].imageLayout = cube[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                VkWriteDescriptorSet w2[2]{};
                for (u32 c = 0; c < 2; ++c) {
                    w2[c].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    w2[c].dstSet = s_set[slot]; w2[c].dstBinding = 6 + c; w2[c].descriptorCount = 1;
                    w2[c].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w2[c].pImageInfo = &cube[c];
                }
                vkUpdateDescriptorSets(VulkanHW.m_Device, 2, w2, 0, nullptr);
                s_boundCube0[slot] = v0; s_boundCube1[slot] = v1;
                static bool s_skyDiag = false;
                if (!s_skyDiag) { s_skyDiag = true;
                    Msg("[VK SkyAmb] real sky cubes bound (slot %u, weight=%.2f, scale=%.2f, lod=%.1f)",
                        slot, w, kAmbientScale, kAmbientLod);
                }
            }
            (void)samp;
        } else {
            static bool s_skyFail = false;
            if (!s_skyFail) { s_skyFail = true; Msg("![VK SkyAmb] AcquireAmbientCubes returned false — fallback grey ambient"); }
        }
    }
    ub.sky_params[0] = skyWeight;
    ub.sky_params[1] = kAmbientScale;
    ub.sky_params[2] = kAmbientLod;
    ub.sky_params[3] = 0.f;

    // GTAO (binding 8): swap the real AO view in when the pass has one, back to
    // the white fallback when it doesn't (prepass off / not rendered yet).
    // Strength 0 keeps the receivers' mix() on 1.0 either way.
    {
        VkImageView ao = SSAOPass::GetResultView();
        VkSampler   aoSamp = SSAOPass::GetSampler();
        if (ao == VK_NULL_HANDLE || aoSamp == VK_NULL_HANDLE) {
            ao = s_fallbackWhite ? s_fallbackWhite->GetView() : VK_NULL_HANDLE;
            aoSamp = s_cubeSampler;
        }
        if (ao != VK_NULL_HANDLE && ao != s_boundAO[slot]) {
            VkDescriptorImageInfo ii{ aoSamp, ao, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
            VkWriteDescriptorSet w{};
            w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w.dstSet = s_set[slot]; w.dstBinding = 8; w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w.pImageInfo = &ii;
            vkUpdateDescriptorSets(VulkanHW.m_Device, 1, &w, 0, nullptr);
            s_boundAO[slot] = ao;
        }
        // Debug: confirm which image binding 8 actually holds (real AO vs the
        // white fallback) — pairs with the [VK SSAO] readback stats.
        if (ps_r_ssao_debug) {
            static u32 s_aoLogCd = 0;
            if (s_aoLogCd == 0) {
                s_aoLogCd = 300;
                const bool real = (SSAOPass::GetResultView() != VK_NULL_HANDLE)
                               && (s_boundAO[slot] == SSAOPass::GetResultView());
                Msg("[VK SSAO] binding8 slot %u: %s (view %p, result %p), strength %.2f",
                    slot, real ? "REAL AO" : "FALLBACK/WHITE",
                    (void*)s_boundAO[slot], (void*)SSAOPass::GetResultView(),
                    SSAOPass::Strength() * ps_r_ssao_strength);
            } else --s_aoLogCd;
        }
    }
    ub.ao_params[0] = Device.dwWidth  ? 1.f / float(Device.dwWidth)  : 0.f;
    ub.ao_params[1] = Device.dwHeight ? 1.f / float(Device.dwHeight) : 0.f;
    ub.ao_params[2] = SSAOPass::Strength() * ps_r_ssao_strength;
    ub.ao_params[3] = (SSAOPass::Strength() > 0.f && ps_r_ssao_debug) ? 1.f : 0.f;

    memcpy(s_mapped + size_t(slot) * kSlotStride, &ub, sizeof(LightUBO));
    s_current = s_set[slot];
}

void Destroy()
{
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
    if (s_fallbackCube) { s_fallbackCube->Destroy(); xr_delete(s_fallbackCube); }
    if (s_fallbackWhite) { s_fallbackWhite->Destroy(); xr_delete(s_fallbackWhite); }
    if (s_cubeSampler) { vkDestroySampler(VulkanHW.m_Device, s_cubeSampler, nullptr); s_cubeSampler = VK_NULL_HANDLE; }
    if (s_pool)      { vkDestroyDescriptorPool(VulkanHW.m_Device, s_pool, nullptr); s_pool = VK_NULL_HANDLE; }
    if (s_setLayout) { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_setLayout, nullptr); s_setLayout = VK_NULL_HANDLE; }
    s_ubo.Destroy();
    s_mapped = nullptr;
    s_current = VK_NULL_HANDLE;
    for (u32 i = 0; i < kFramesInFlight; ++i) { s_set[i] = VK_NULL_HANDLE; s_boundCube0[i] = VK_NULL_HANDLE; s_boundCube1[i] = VK_NULL_HANDLE; s_boundAO[i] = VK_NULL_HANDLE; }
    s_inited = false; s_failed = false;
}

}}  // namespace VK::EnvLight
