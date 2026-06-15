// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).
//
// Variable Rate Shading — depth-driven SRI built by compute. See vk_vrs.h.

#include "stdafx.h"
#include "vk_vrs.h"
#include "HW_Vulkan.h"
#include "vk_shaders.h"    // g_ShaderManager (vrs_build.comp.spv)
#include "vk_profiler.h"   // Prof::NameImage

extern int   ps_r_vrs;        // 0 off / 1 mild / 2 aggressive
extern float ps_r_vrs_near;   // metres: below this stays 1x1 (level 1 threshold)
extern float ps_r_vrs_far;    // metres: above this goes coarsest (level 1 threshold)

namespace VK { namespace VRS {

namespace {

constexpr u32 N = VK_FRAMES_IN_FLIGHT;

PFN_vkCmdSetFragmentShadingRateKHR pfnSetRate = nullptr;
bool s_inited = false;
bool s_dead   = false;   // unsupported / init failed → permanent no-op

// compute pipeline
VkPipeline            s_pipe   = VK_NULL_HANDLE;
VkPipelineLayout      s_layout = VK_NULL_HANDLE;
VkDescriptorSetLayout s_setL   = VK_NULL_HANDLE;
VkDescriptorPool      s_pool   = VK_NULL_HANDLE;
VkDescriptorSet       s_set[N] = {};
VkSampler             s_depthSampler = VK_NULL_HANDLE;

// per-frame shading-rate images (written by compute, read as FSR attachment)
VkImage       s_image[N] = {};
VmaAllocation s_alloc[N] = {};
VkImageView   s_view[N]  = {};
VkExtent2D    s_screen   = { 0, 0 };
VkExtent2D    s_tiles    = { 0, 0 };
int           s_cur      = -1;

struct Push {
    float   invScreen[2];
    float   nearFar[2];
    float   proj[2];     // A (p43), B (p33)
    int32_t tiles[2];
    int32_t texel[2];
    int32_t level;
};

void FreeImages()
{
    for (u32 i = 0; i < N; ++i) {
        if (s_view[i])  { vkDestroyImageView(VulkanHW.m_Device, s_view[i], nullptr);  s_view[i]  = VK_NULL_HANDLE; }
        if (s_image[i]) { vmaDestroyImage(VulkanHW.m_Allocator, s_image[i], s_alloc[i]); s_image[i] = VK_NULL_HANDLE; s_alloc[i] = VK_NULL_HANDLE; }
    }
    s_screen = { 0, 0 }; s_tiles = { 0, 0 }; s_cur = -1;
}

bool CreatePipeline()
{
    if (!g_ShaderManager) g_ShaderManager = xr_new<VK::CVulkanSPIRVLoader>();
    VkShaderModule cs = g_ShaderManager->Load("vrs_build.comp.spv");
    if (!cs) { Msg("![VK VRS] vrs_build.comp.spv load failed"); return false; }

    VkDescriptorSetLayoutBinding b[2]{};
    b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; b[0].descriptorCount = 1; b[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    b[1].binding = 1; b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;          b[1].descriptorCount = 1; b[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = 2; lci.pBindings = b;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &s_setL) != VK_SUCCESS) return false;

    VkDescriptorPoolSize ps[2] = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, N },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          N },
    };
    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets = N; pci.poolSizeCount = 2; pci.pPoolSizes = ps;
    if (vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &s_pool) != VK_SUCCESS) return false;
    VkDescriptorSetLayout layouts[N]; for (u32 i = 0; i < N; ++i) layouts[i] = s_setL;
    VkDescriptorSetAllocateInfo dai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dai.descriptorPool = s_pool; dai.descriptorSetCount = N; dai.pSetLayouts = layouts;
    if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, s_set) != VK_SUCCESS) return false;

    VkSamplerCreateInfo si{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    si.magFilter = si.minFilter = VK_FILTER_NEAREST;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (vkCreateSampler(VulkanHW.m_Device, &si, nullptr, &s_depthSampler) != VK_SUCCESS) return false;

    VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push) };
    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1; plci.pSetLayouts = &s_setL; plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_layout) != VK_SUCCESS) return false;

    VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = cs;
    cpci.stage.pName  = "main";
    cpci.layout       = s_layout;
    if (vkCreateComputePipelines(VulkanHW.m_Device, VK_NULL_HANDLE, 1, &cpci, nullptr, &s_pipe) != VK_SUCCESS) return false;
    return true;
}

void EnsureSize(VkExtent2D screen)
{
    if (s_image[0] && screen.width == s_screen.width && screen.height == s_screen.height) return;
    FreeImages();

    const VkExtent2D tex = VulkanHW.m_VRSTexelSize;
    const u32 tw = (screen.width  + tex.width  - 1) / tex.width;
    const u32 th = (screen.height + tex.height - 1) / tex.height;
    if (tw == 0 || th == 0) return;

    for (u32 i = 0; i < N; ++i) {
        VkImageCreateInfo ici{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        ici.imageType   = VK_IMAGE_TYPE_2D;
        ici.format      = VK_FORMAT_R8_UINT;
        ici.extent      = { tw, th, 1 };
        ici.mipLevels   = 1; ici.arrayLayers = 1;
        ici.samples     = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling      = VK_IMAGE_TILING_OPTIMAL;
        ici.usage       = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VmaAllocationCreateInfo aci{}; aci.usage = VMA_MEMORY_USAGE_AUTO;
        if (vmaCreateImage(VulkanHW.m_Allocator, &ici, &aci, &s_image[i], &s_alloc[i], nullptr) != VK_SUCCESS) {
            Msg("![VK VRS] SRI image %u create failed", i); FreeImages(); return;
        }
        Prof::NameImage(s_image[i], "VRS.ShadingRateImage");
        VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        vci.image = s_image[i]; vci.viewType = VK_IMAGE_VIEW_TYPE_2D; vci.format = VK_FORMAT_R8_UINT;
        vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        if (vkCreateImageView(VulkanHW.m_Device, &vci, nullptr, &s_view[i]) != VK_SUCCESS) {
            Msg("![VK VRS] SRI view %u failed", i); FreeImages(); return;
        }
    }
    s_screen = screen; s_tiles = { tw, th };
    Msg("[VK VRS] depth-driven SRI %ux%u tiles x%u (tile %ux%u)", tw, th, N, tex.width, tex.height);
}

} // anonymous namespace

void Init()
{
    if (s_inited) return;
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
    s_inited = true;

    if (!VulkanHW.m_bVRSSupported) { s_dead = true; return; }

    pfnSetRate = (PFN_vkCmdSetFragmentShadingRateKHR)
                 vkGetDeviceProcAddr(VulkanHW.m_Device, "vkCmdSetFragmentShadingRateKHR");

    // R8_UINT must be storage-image capable (compute writes the SRI directly).
    VkFormatProperties fp{};
    vkGetPhysicalDeviceFormatProperties(VulkanHW.m_PhysicalDevice, VK_FORMAT_R8_UINT, &fp);
    if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) {
        Msg("![VK VRS] R8_UINT storage images unsupported — VRS disabled"); s_dead = true; return;
    }
    if (!pfnSetRate || !CreatePipeline()) {
        Msg("![VK VRS] compute init failed — VRS disabled"); s_dead = true; return;
    }
    Msg("[VK VRS] depth-driven init OK");
}

bool Wanted() { return VulkanHW.m_bVRSSupported && ps_r_vrs > 0; }

void BuildFromDepth(VkCommandBuffer cmd, u32 frameIndex, VkImageView depthView, VkExtent2D screen, float A, float B)
{
    Init();
    if (s_dead || s_pipe == VK_NULL_HANDLE) return;
    EnsureSize(screen);
    if (!s_image[0]) return;

    const u32 cur = frameIndex % N;
    s_cur = (int)cur;

    // descriptor: depth (read) + this slot's SRI (write)
    VkDescriptorImageInfo di{ s_depthSampler, depthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkDescriptorImageInfo si{ VK_NULL_HANDLE, s_view[cur], VK_IMAGE_LAYOUT_GENERAL };
    VkWriteDescriptorSet w[2]{};
    w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[0].dstSet = s_set[cur]; w[0].dstBinding = 0;
    w[0].descriptorCount = 1; w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[0].pImageInfo = &di;
    w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[1].dstSet = s_set[cur]; w[1].dstBinding = 1;
    w[1].descriptorCount = 1; w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[1].pImageInfo = &si;
    vkUpdateDescriptorSets(VulkanHW.m_Device, 2, w, 0, nullptr);

    auto imgBarrier = [&](VkImageLayout oldL, VkImageLayout newL, VkAccessFlags srcA, VkAccessFlags dstA,
                          VkPipelineStageFlags srcS, VkPipelineStageFlags dstS) {
        VkImageMemoryBarrier ib{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        ib.oldLayout = oldL; ib.newLayout = newL;
        ib.srcQueueFamilyIndex = ib.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        ib.image = s_image[cur]; ib.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        ib.srcAccessMask = srcA; ib.dstAccessMask = dstA;
        vkCmdPipelineBarrier(cmd, srcS, dstS, 0, 0, nullptr, 0, nullptr, 1, &ib);
    };

    // UNDEFINED (discard prev) → GENERAL for the compute store.
    imgBarrier(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
               0, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_layout, 0, 1, &s_set[cur], 0, nullptr);

    const VkExtent2D tex = VulkanHW.m_VRSTexelSize;
    const int level = ps_r_vrs;
    Push pc{};
    pc.invScreen[0] = 1.0f / float(screen.width);
    pc.invScreen[1] = 1.0f / float(screen.height);
    // distance thresholds (metres, live via r_vrs_near/r_vrs_far). Level 1 uses
    // them as-is; level 2 (aggressive) pulls them ~closer for more coarse area.
    const float kn = ps_r_vrs_near, kf = ps_r_vrs_far;
    pc.nearFar[0] = (level >= 2) ? _max(2.0f,            kn - 10.0f) : kn;
    pc.nearFar[1] = (level >= 2) ? _max(pc.nearFar[0] + 5.0f, kf - 25.0f) : kf;
    pc.proj[0] = A; pc.proj[1] = B;
    pc.tiles[0] = (int)s_tiles.width; pc.tiles[1] = (int)s_tiles.height;
    pc.texel[0] = (int)tex.width;     pc.texel[1] = (int)tex.height;
    pc.level = level;
    vkCmdPushConstants(cmd, s_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Push), &pc);

    vkCmdDispatch(cmd, (s_tiles.width + 7) / 8, (s_tiles.height + 7) / 8, 1);

    // GENERAL (compute write) → FSR attachment (read by the world color pass).
    imgBarrier(VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR,
               VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_FRAGMENT_SHADING_RATE_ATTACHMENT_READ_BIT_KHR,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR);
}

VkImageView GetView()   { return (s_cur >= 0 && s_image[s_cur]) ? s_view[s_cur] : VK_NULL_HANDLE; }
VkExtent2D  TexelSize() { return VulkanHW.m_VRSTexelSize; }

void CmdSetRate(VkCommandBuffer cmd)
{
    if (!pfnSetRate) return;
    const VkExtent2D rate = { 1, 1 };
    const VkFragmentShadingRateCombinerOpKHR ops[2] = {
        VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR,
        VK_FRAGMENT_SHADING_RATE_COMBINER_OP_REPLACE_KHR,
    };
    pfnSetRate(cmd, &rate, ops);
}

void Destroy()
{
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
    FreeImages();
    if (s_pipe)          { vkDestroyPipeline(VulkanHW.m_Device, s_pipe, nullptr); s_pipe = VK_NULL_HANDLE; }
    if (s_layout)        { vkDestroyPipelineLayout(VulkanHW.m_Device, s_layout, nullptr); s_layout = VK_NULL_HANDLE; }
    if (s_pool)          { vkDestroyDescriptorPool(VulkanHW.m_Device, s_pool, nullptr); s_pool = VK_NULL_HANDLE; }
    if (s_setL)          { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_setL, nullptr); s_setL = VK_NULL_HANDLE; }
    if (s_depthSampler)  { vkDestroySampler(VulkanHW.m_Device, s_depthSampler, nullptr); s_depthSampler = VK_NULL_HANDLE; }
    for (u32 i = 0; i < N; ++i) s_set[i] = VK_NULL_HANDLE;
    s_inited = false; s_dead = false; pfnSetRate = nullptr;
}

}} // namespace VK::VRS
