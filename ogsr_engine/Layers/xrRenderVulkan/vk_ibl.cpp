// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).
//
// Sky specular IBL prefilter — see vk_ibl.h.

#include "stdafx.h"
#include "vk_ibl.h"
#include "vk_image.h"            // VK::CreateImage / CreateImageView
#include "vk_compute_util.h"     // VK::MakePipelineLayout / CreateComputePipeline
#include "vk_command_buffer.h"   // CommandManager (BeginImmediate / EndAndSubmitImmediate)
#include "vk_shaders.h"          // g_ShaderManager (SPIR-V loader)
#include "vk_buffer.h"           // CVulkanBuffer (SH coefficient SSBO)
#include <cmath>

extern int ps_r_sky_sh_debug;   // r_sky_sh_debug — dump the projected SH coefficients (L0/L1/aniso)

namespace VK { namespace IBL {

namespace {
    constexpr u32 kFace = 128;    // cube face size (px) of the prefiltered cube
    constexpr u32 kMips = 6;      // roughness mip chain: 128,64,32,16,8,4

    bool s_inited = false, s_failed = false, s_ready = false, s_hasContent = false;

    VkImage       s_img     = VK_NULL_HANDLE;
    VmaAllocation s_alloc   = VK_NULL_HANDLE;
    VkImageView   s_sampleV = VK_NULL_HANDLE;   // CUBE, all mips — read by the forward shaders
    VkImageView   s_storeV  = VK_NULL_HANDLE;   // 2D_ARRAY, mip 0, 6 layers — compute store target
    VkSampler     s_sampler = VK_NULL_HANDLE;

    VkDescriptorSetLayout s_setL   = VK_NULL_HANDLE;
    VkDescriptorPool      s_pool   = VK_NULL_HANDLE;
    VkDescriptorSet       s_set    = VK_NULL_HANDLE;
    VkPipelineLayout      s_layout = VK_NULL_HANDLE;
    VkPipeline            s_pipe   = VK_NULL_HANDLE;

    // ── Diffuse SH9 projection (sky_sh_project.comp) ─────────────────────────
    // The probe cube above is world-space, so projecting it needs no remap — walk
    // it, weight by solid angle, out come 9 coefficients. Source mip is chosen so
    // the walk is a few thousand texels: we are collapsing the sphere to 9 numbers,
    // so sampling it at full 128² would be pure waste.
    constexpr u32 kSHMip     = 2;                    // 128 >> 2 = 32² per face
    constexpr u32 kSHFace    = kFace >> kSHMip;
    constexpr VkDeviceSize kSHBytes = 9 * 4 * sizeof(float);   // vec4[9], std430

    CVulkanBuffer         s_shBuf;
    CVulkanBuffer         s_shRead;    // host-visible mirror for the r_sky_sh_debug dump
    bool                  s_shReady = false;
    VkDescriptorSetLayout s_shSetL  = VK_NULL_HANDLE;
    VkDescriptorPool      s_shPool  = VK_NULL_HANDLE;
    VkDescriptorSet       s_shSet   = VK_NULL_HANDLE;
    VkPipelineLayout      s_shLayout = VK_NULL_HANDLE;
    VkPipeline            s_shPipe   = VK_NULL_HANDLE;

    // Change detection: only re-prefilter when the weather cubes, the cross-fade or
    // the sky spin move. Rotation belongs here because the probe is world-space —
    // a rotated dome is a genuinely different probe, not the same one viewed anew.
    VkImageView s_last0 = VK_NULL_HANDLE, s_last1 = VK_NULL_HANDLE;
    float       s_lastW = -1.f, s_lastRot = -1e9f, s_lastGround = -1.f;
    float       s_lastTint[3] = { -1.f, -1.f, -1.f };
    float       s_lastSun[3]  = { -9.f, -9.f, -9.f };
    int         s_lastProc    = -1;
    float       s_lastAtmo[3] = { -1.f, -1.f, -1.f };   // intensity / turbidity / mieG

    // p[0..3]   = weight / face size / sky rotation (prefilter) or mip / face / ground
    //             (SH projection)
    // p[4..6]   = sky_color tint
    // p[8..10]  = direction TO the sun
    // p[12..15] = procedural sky: enable / intensity / turbidity / Mie g
    // One struct for both pipelines so the push range matches.
    struct Push { float p[16]; };

    // Barrier a mip range (all 6 layers) between layouts.
    void BarrierMips(VkCommandBuffer cmd, u32 baseMip, u32 mipCount,
                     VkImageLayout oldL, VkImageLayout newL,
                     VkAccessFlags srcA, VkAccessFlags dstA,
                     VkPipelineStageFlags srcS, VkPipelineStageFlags dstS)
    {
        VkImageMemoryBarrier b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        b.oldLayout = oldL; b.newLayout = newL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = s_img;
        b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, baseMip, mipCount, 0, 6 };
        b.srcAccessMask = srcA; b.dstAccessMask = dstA;
        vkCmdPipelineBarrier(cmd, srcS, dstS, 0, 0, nullptr, 0, nullptr, 1, &b);
    }
}

bool Init()
{
    if (s_inited) return s_ready;
    s_inited = true;

    if (!g_ShaderManager) g_ShaderManager = xr_new<VK::CVulkanSPIRVLoader>();
    VkShaderModule cs = g_ShaderManager->Load("ibl_prefilter.comp.spv");
    if (!cs) { Msg("![VK IBL] ibl_prefilter.comp.spv load failed — disabled"); s_failed = true; return false; }

    // RGBA16F cube: STORAGE (compute mip0) | SAMPLED (shaders) | TRANSFER src/dst (blit mips).
    VK::ImageDesc d;
    d.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    d.extent = { kFace, kFace, 1 };
    d.mips   = kMips;
    d.layers = 6;
    d.flags  = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    d.usage  = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
             | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    d.name   = "IBL.Cube";
    if (!VK::CreateImage(d, s_img, s_alloc)) { s_failed = true; return false; }

    // Sampled CUBE view (all mips) — the roughness reflection the receivers read.
    s_sampleV = VK::CreateImageView(s_img, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_VIEW_TYPE_CUBE,
                                    VK_IMAGE_ASPECT_COLOR_BIT, 0, kMips, 0, 6);
    if (s_sampleV == VK_NULL_HANDLE) { s_failed = true; return false; }
    // Storage 2D_ARRAY view (mip 0, 6 layers) — the compute writes cube faces as an array.
    s_storeV = VK::CreateImageView(s_img, VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_VIEW_TYPE_2D_ARRAY,
                                   VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 6);
    if (s_storeV == VK_NULL_HANDLE) { s_failed = true; return false; }

    // Trilinear clamp sampler (mip = roughness).
    VkSamplerCreateInfo sci{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.maxAnisotropy = 1.f; sci.minLod = 0.f; sci.maxLod = float(kMips);
    if (vkCreateSampler(VulkanHW.m_Device, &sci, nullptr, &s_sampler) != VK_SUCCESS) {
        Msg("![VK IBL] sampler create failed"); s_failed = true; return false;
    }

    // Compute set: 0/1 = sky cubes (in), 2 = spec cube mip 0 storage (out).
    VkDescriptorSetLayoutBinding b[3]{};
    b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; b[0].descriptorCount = 1; b[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    b[1].binding = 1; b[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; b[1].descriptorCount = 1; b[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    b[2].binding = 2; b[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;          b[2].descriptorCount = 1; b[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = 3; lci.pBindings = b;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &s_setL) != VK_SUCCESS) { Msg("![VK IBL] set layout failed"); s_failed = true; return false; }

    VkDescriptorPoolSize ps[2]{
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1 },
    };
    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets = 1; pci.poolSizeCount = 2; pci.pPoolSizes = ps;
    if (vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &s_pool) != VK_SUCCESS) { Msg("![VK IBL] pool failed"); s_failed = true; return false; }

    VkDescriptorSetAllocateInfo dai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dai.descriptorPool = s_pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &s_setL;
    if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, &s_set) != VK_SUCCESS) { Msg("![VK IBL] alloc set failed"); s_failed = true; return false; }

    // Storage output (binding 2) is stable — write it once; Update rewrites 0/1.
    {
        VkDescriptorImageInfo oi{ VK_NULL_HANDLE, s_storeV, VK_IMAGE_LAYOUT_GENERAL };
        VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w.dstSet = s_set; w.dstBinding = 2; w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w.pImageInfo = &oi;
        vkUpdateDescriptorSets(VulkanHW.m_Device, 1, &w, 0, nullptr);
    }

    s_layout = VK::MakePipelineLayout({ s_setL }, sizeof(Push));
    if (s_layout == VK_NULL_HANDLE) { s_failed = true; return false; }

    s_pipe = VK::CreateComputePipeline(cs, s_layout, "IBL.Prefilter");
    if (s_pipe == VK_NULL_HANDLE) { s_failed = true; return false; }

    // ── SH9 diffuse projection ───────────────────────────────────────────────
    // Set up separately and NON-FATALLY: if this half fails to build, specular IBL
    // still works and the receivers fall back to sampling the probe's top mip, so
    // there is no reason to take the whole module down with it.
    do {
        VkShaderModule shcs = g_ShaderManager->Load("sky_sh_project.comp.spv");
        if (!shcs) { Msg("![VK IBL] sky_sh_project.comp.spv load failed — diffuse SH disabled"); break; }

        // GPU-only: written by this compute, read by the forward fragment shaders.
        // (Without gpuOnly every storage buffer lands in HOST_VISIBLE memory and the
        //  shader reads it over PCIe — the vk_buffer.h warning.)
        s_shBuf.Create(kSHBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
        if (s_shBuf.GetHandle() == VK_NULL_HANDLE) { Msg("![VK IBL] SH buffer create failed"); break; }
        // Host-visible mirror, so r_sky_sh_debug can print what was actually
        // projected. The projection runs on the fence-waited immediate queue, so a
        // copy issued in the same submit is readable the moment Update returns —
        // no extra sync, and nothing is read unless the cvar is on.
        s_shRead.Create(kSHBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);

        VkDescriptorSetLayoutBinding sb[2]{};
        sb[0].binding = 0; sb[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; sb[0].descriptorCount = 1; sb[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        sb[1].binding = 1; sb[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;         sb[1].descriptorCount = 1; sb[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayoutCreateInfo sl{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        sl.bindingCount = 2; sl.pBindings = sb;
        if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &sl, nullptr, &s_shSetL) != VK_SUCCESS) break;

        VkDescriptorPoolSize sps[2]{
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         1 },
        };
        VkDescriptorPoolCreateInfo spci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        spci.maxSets = 1; spci.poolSizeCount = 2; spci.pPoolSizes = sps;
        if (vkCreateDescriptorPool(VulkanHW.m_Device, &spci, nullptr, &s_shPool) != VK_SUCCESS) break;

        VkDescriptorSetAllocateInfo sdai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        sdai.descriptorPool = s_shPool; sdai.descriptorSetCount = 1; sdai.pSetLayouts = &s_shSetL;
        if (vkAllocateDescriptorSets(VulkanHW.m_Device, &sdai, &s_shSet) != VK_SUCCESS) break;

        // Both operands are stable for the module's lifetime — write the set once.
        VkDescriptorImageInfo  pi{ s_sampler, s_sampleV, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorBufferInfo bi{ s_shBuf.GetHandle(), 0, kSHBytes };
        VkWriteDescriptorSet   sw[2]{};
        sw[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; sw[0].dstSet = s_shSet; sw[0].dstBinding = 0;
        sw[0].descriptorCount = 1; sw[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; sw[0].pImageInfo = &pi;
        sw[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; sw[1].dstSet = s_shSet; sw[1].dstBinding = 1;
        sw[1].descriptorCount = 1; sw[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; sw[1].pBufferInfo = &bi;
        vkUpdateDescriptorSets(VulkanHW.m_Device, 2, sw, 0, nullptr);

        s_shLayout = VK::MakePipelineLayout({ s_shSetL }, sizeof(Push));
        if (s_shLayout == VK_NULL_HANDLE) break;
        s_shPipe = VK::CreateComputePipeline(shcs, s_shLayout, "IBL.SHProject");
    } while (false);

    s_ready = true;
    Msg("[VK IBL] init OK — %ux%u RGBA16F spec cube, %u roughness mips, diffuse SH9 %s (mip %u, %u²)",
        kFace, kFace, kMips, s_shPipe ? "on" : "OFF", kSHMip, kSHFace);
    return true;
}

bool Ready() { return s_ready && !s_failed && s_hasContent; }

void Update(VkImageView sky0, VkImageView sky1, VkSampler skySampler, const SkyDesc& sky)
{
    if (!s_ready || s_failed) return;
    // The procedural sky needs no cube at all — it IS the source. Only the legacy
    // path is blocked on the weather textures being up.
    if (!sky.proc && (sky0 == VK_NULL_HANDLE || skySampler == VK_NULL_HANDLE)) return;
    if (sky0 == VK_NULL_HANDLE || skySampler == VK_NULL_HANDLE) return;   // descriptors still need something bound
    if (sky1 == VK_NULL_HANDLE) sky1 = sky0;

    const float  weight       = sky.weight;
    const float  skyRotation  = sky.rotation;
    const float  groundBounce = sky.groundBounce;
    const float* skyTint      = sky.tint;

    // Rotation moves the probe just as much as a cube swap does, so it gates the
    // refresh too. The threshold is ~0.6° — below that the SH9 result is visually
    // identical and re-projecting would just burn an immediate submit every frame
    // on maps whose sky_rotation creeps continuously.
    const bool changed = (sky0 != s_last0) || (sky1 != s_last1)
                       || (std::fabs(weight - s_lastW) > 0.02f)
                       || (std::fabs(skyRotation - s_lastRot) > 0.01f)
                       || (std::fabs(groundBounce - s_lastGround) > 0.01f)
                       // The tint is time-of-day: it is what makes the probe track dusk,
                       // so it must gate the refresh. 0.02 keeps re-projection to roughly
                       // the same cadence the cross-fade already causes.
                       || (std::fabs(skyTint[0] - s_lastTint[0]) > 0.02f)
                       || (std::fabs(skyTint[1] - s_lastTint[1]) > 0.02f)
                       || (std::fabs(skyTint[2] - s_lastTint[2]) > 0.02f)
                       // Procedural sky: the SUN is the model's only real input, so the
                       // probe is stale as soon as it moves. 0.02 on the direction vector
                       // is ~1.1 deg — the prefilter is a fence-waited submit and the
                       // atmosphere march is not free, so re-projecting on every
                       // arc-minute would be a visible hitch for no visible gain.
                       || (sky.proc ? 1 : 0) != s_lastProc
                       || (sky.proc && (std::fabs(sky.sunDir[0] - s_lastSun[0]) > 0.02f
                                     || std::fabs(sky.sunDir[1] - s_lastSun[1]) > 0.02f
                                     || std::fabs(sky.sunDir[2] - s_lastSun[2]) > 0.02f
                                     || std::fabs(sky.intensity - s_lastAtmo[0]) > 0.01f
                                     || std::fabs(sky.turbidity - s_lastAtmo[1]) > 0.01f
                                     || std::fabs(sky.mieG      - s_lastAtmo[2]) > 0.01f))
                       || !s_hasContent;
    if (!changed) return;

    VkCommandBuffer cmd = CommandManager.BeginImmediate();
    if (cmd == VK_NULL_HANDLE) { Msg("![VK IBL] BeginImmediate failed"); return; }

    // Sky cubes (bindings 0/1) for this refresh.
    VkDescriptorImageInfo si[2]{
        { skySampler, sky0, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
        { skySampler, sky1, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },
    };
    VkWriteDescriptorSet w[2]{};
    for (u32 i = 0; i < 2; ++i) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = s_set;
        w[i].dstBinding = i; w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[i].pImageInfo = &si[i];
    }
    vkUpdateDescriptorSets(VulkanHW.m_Device, 2, w, 0, nullptr);

    // mip 0 → GENERAL (we overwrite; UNDEFINED discards the old content).
    BarrierMips(cmd, 0, 1, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                0, VK_ACCESS_SHADER_WRITE_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_layout, 0, 1, &s_set, 0, nullptr);
    Push pc{}; pc.p[0] = weight; pc.p[1] = float(kFace); pc.p[2] = skyRotation;
    pc.p[4] = skyTint[0]; pc.p[5] = skyTint[1]; pc.p[6] = skyTint[2];
    pc.p[8] = sky.sunDir[0]; pc.p[9] = sky.sunDir[1]; pc.p[10] = sky.sunDir[2];
    pc.p[12] = sky.proc ? 1.f : 0.f;
    pc.p[13] = sky.intensity; pc.p[14] = sky.turbidity; pc.p[15] = sky.mieG;
    vkCmdPushConstants(cmd, s_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, (kFace + 7) / 8, (kFace + 7) / 8, 6);

    // mip 0: compute WRITE → transfer READ (blit source). mips 1.. : UNDEFINED → TRANSFER_DST.
    BarrierMips(cmd, 0, 1, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    BarrierMips(cmd, 1, kMips - 1, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                0, VK_ACCESS_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    // Box-blit chain: mip m = downsample of mip m-1 (roughness grows with mip).
    for (u32 m = 1; m < kMips; ++m) {
        const int sw = int(kFace >> (m - 1)), sh = sw;
        const int dw = int(kFace >> m),       dh = dw;
        VkImageBlit blit{};
        blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, m - 1, 0, 6 };
        blit.srcOffsets[1]  = { sw, sh, 1 };
        blit.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, m, 0, 6 };
        blit.dstOffsets[1]  = { dw, dh, 1 };
        vkCmdBlitImage(cmd, s_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            s_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
        // this mip becomes the next blit's source
        BarrierMips(cmd, m, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    }

    // All mips are TRANSFER_SRC now → SHADER_READ, for the forward samplers AND for
    // the SH projection below, which reads this same cube as a sampler.
    BarrierMips(cmd, 0, kMips, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    // Diffuse SH9: collapse the freshly-built world-space probe into 9 coefficients.
    // One workgroup, ~6k texel fetches — rides along in the same immediate submit
    // that already paid for the prefilter, so the diffuse upgrade costs no extra
    // sync and no per-frame work at all.
    if (s_shPipe != VK_NULL_HANDLE) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_shPipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_shLayout, 0, 1, &s_shSet, 0, nullptr);
        Push sp{}; sp.p[0] = float(kSHMip); sp.p[1] = float(kSHFace); sp.p[2] = groundBounce;
        vkCmdPushConstants(cmd, s_shLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(sp), &sp);
        vkCmdDispatch(cmd, 1, 1, 1);

        VkBufferMemoryBarrier bb{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
        bb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        bb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        bb.srcQueueFamilyIndex = bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bb.buffer = s_shBuf.GetHandle(); bb.offset = 0; bb.size = kSHBytes;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 1, &bb, 0, nullptr);

        if (ps_r_sky_sh_debug && s_shRead.GetHandle() != VK_NULL_HANDLE) {
            VkBufferCopy rc{ 0, 0, kSHBytes };
            vkCmdCopyBuffer(cmd, s_shBuf.GetHandle(), s_shRead.GetHandle(), 1, &rc);
        }
    }

    CommandManager.EndAndSubmitImmediate(cmd);   // fence-waited

    s_last0 = sky0; s_last1 = sky1; s_lastW = weight;
    s_lastRot = skyRotation; s_lastGround = groundBounce;
    s_lastTint[0] = skyTint[0]; s_lastTint[1] = skyTint[1]; s_lastTint[2] = skyTint[2];
    s_lastSun[0] = sky.sunDir[0]; s_lastSun[1] = sky.sunDir[1]; s_lastSun[2] = sky.sunDir[2];
    s_lastProc = sky.proc ? 1 : 0;
    s_lastAtmo[0] = sky.intensity; s_lastAtmo[1] = sky.turbidity; s_lastAtmo[2] = sky.mieG;
    if (s_shPipe != VK_NULL_HANDLE) s_shReady = true;
    if (!s_hasContent) {
        s_hasContent = true;
        Msg("[VK IBL] first prefilter done (weight=%.2f, skyRot=%.3f rad, SH9 %s)",
            weight, skyRotation, s_shReady ? "projected" : "off");
    }

    // r_sky_sh_debug: report what the projection actually produced. The question a
    // flat-looking dusk raises is not "did SH run" but "does the sky the probe sees
    // carry any azimuthal energy at all" — and that is exactly the L0 vs L1 ratio.
    //   L0  = the omnidirectional average (how bright the sky is overall),
    //   L1  = the linear band; its length is how much the irradiance VARIES with
    //         direction, and its direction is where the sky is brightest.
    // aniso = |L1|/L0 near 0 means a uniform sky dome: no amount of correct maths
    // downstream can make terrain directional, because the source has no direction.
    if (ps_r_sky_sh_debug && s_shRead.GetHandle() != VK_NULL_HANDLE) {
        if (const float* c = static_cast<const float*>(s_shRead.Map())) {
            auto lum = [](const float* v) { return 0.2126f * v[0] + 0.7152f * v[1] + 0.0722f * v[2]; };
            const float l0 = lum(c + 0);
            // Basis order (sky_sh_project.comp): 1 = y, 2 = z, 3 = x.
            const float dx = lum(c + 12), dy = lum(c + 4), dz = lum(c + 8);
            const float len = sqrtf(dx * dx + dy * dy + dz * dz);
            const float aniso = (l0 > 1e-6f) ? (len / l0) : 0.f;
            Msg("[VK SH] L0=(%.4f,%.4f,%.4f) lum=%.4f | L1 dir=(%.2f,%.2f,%.2f) len=%.4f | aniso=%.3f "
                "| azimuth=%.1f deg | skyRot=%.1f deg | ground=%.2f",
                c[0], c[1], c[2], l0,
                (len > 1e-6f) ? dx / len : 0.f, (len > 1e-6f) ? dy / len : 0.f, (len > 1e-6f) ? dz / len : 0.f,
                len, aniso, atan2f(dz, dx) * 57.2957795f, skyRotation * 57.2957795f, groundBounce);
            Msg("[VK SH]   tint sky_color=(%.3f,%.3f,%.3f) x1.7  (probe = raw cube x tint)",
                skyTint[0], skyTint[1], skyTint[2]);
            s_shRead.Unmap();
        }
    }
}

VkImageView GetSpecView() { return s_hasContent ? s_sampleV : VK_NULL_HANDLE; }
VkSampler   GetSampler()  { return s_sampler; }
u32         GetMaxMip()   { return kMips - 1; }
VkBuffer    GetSHBuffer() { return s_shBuf.GetHandle(); }
bool        SHReady()     { return s_shReady; }

void Destroy()
{
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
    if (s_shPipe)   { vkDestroyPipeline(VulkanHW.m_Device, s_shPipe, nullptr); s_shPipe = VK_NULL_HANDLE; }
    if (s_shLayout) { vkDestroyPipelineLayout(VulkanHW.m_Device, s_shLayout, nullptr); s_shLayout = VK_NULL_HANDLE; }
    if (s_shPool)   { vkDestroyDescriptorPool(VulkanHW.m_Device, s_shPool, nullptr); s_shPool = VK_NULL_HANDLE; }
    if (s_shSetL)   { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_shSetL, nullptr); s_shSetL = VK_NULL_HANDLE; }
    s_shBuf.Destroy();
    s_shRead.Destroy();
    s_shSet = VK_NULL_HANDLE; s_shReady = false;
    if (s_pipe)    { vkDestroyPipeline(VulkanHW.m_Device, s_pipe, nullptr); s_pipe = VK_NULL_HANDLE; }
    if (s_layout)  { vkDestroyPipelineLayout(VulkanHW.m_Device, s_layout, nullptr); s_layout = VK_NULL_HANDLE; }
    if (s_pool)    { vkDestroyDescriptorPool(VulkanHW.m_Device, s_pool, nullptr); s_pool = VK_NULL_HANDLE; }
    if (s_setL)    { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_setL, nullptr); s_setL = VK_NULL_HANDLE; }
    if (s_sampler) { vkDestroySampler(VulkanHW.m_Device, s_sampler, nullptr); s_sampler = VK_NULL_HANDLE; }
    if (s_sampleV) { vkDestroyImageView(VulkanHW.m_Device, s_sampleV, nullptr); s_sampleV = VK_NULL_HANDLE; }
    if (s_storeV)  { vkDestroyImageView(VulkanHW.m_Device, s_storeV, nullptr); s_storeV = VK_NULL_HANDLE; }
    if (s_img)     { VK::Vram::DestroyImage(VulkanHW.m_Allocator, s_img, s_alloc); s_img = VK_NULL_HANDLE; s_alloc = VK_NULL_HANDLE; }
    s_set = VK_NULL_HANDLE;
    s_last0 = s_last1 = VK_NULL_HANDLE; s_lastW = -1.f; s_lastRot = -1e9f; s_lastGround = -1.f;
    s_inited = s_failed = s_ready = s_hasContent = false;
}

}}  // namespace VK::IBL
