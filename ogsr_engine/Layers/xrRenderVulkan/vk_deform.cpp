// xrRenderVulkan - Snow (and mud) DEFORMATION texture (compute).
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).
//
// A PERSISTENT, player-centred top-down compression field. Unlike the rain box
// (±75 m @ 1024² = 14.6 cm/texel = too coarse for footprints), this has its OWN
// dense ortho box (±kHalf @ kSize) so prints are sharp. Feet stamp depressions; the
// field is reprojected as the camera moves (texel-snapped -> translation) and decays
// slowly so prints last ~r_snow_deform_time seconds. Read by snow_displace.glsl
// (EnvLight binding 20). Structure mirrors VK::WaterSim (persistent ping-pong +
// per-frame reprojection); see deform_stamp.comp.glsl.

#include "stdafx.h"
#include "vk_deform.h"
#include "vk_shaders.h"         // g_ShaderManager
#include "vk_pipeline_cache.h"  // shared pipeline cache object
#include "vk_shadow.h"          // rain map view/sampler/VP + RainEyeY (terrain-height gate)
#include "vk_command_buffer.h"  // CommandManager.GetCurrentFrame() (per-slot stamp UBO)

extern float ps_r_snow_deform_time;   // r_snow_deform_time — print lifetime (sec)
extern float ps_r_snow_berm;          // r_snow_berm — displaced-snow berm height (fraction of dent depth)
extern float ps_r_snow_rough;         // r_snow_rough — per-print/edge imperfection strength

namespace VK { namespace Deform {

namespace {
    bool          s_inited = false, s_failed = false, s_first = true;
    constexpr u32   kSize = 2048;     // 2048² R16F (~8 MB each) — sharp prints
    constexpr float kHalf = 28.f;     // ±28 m around the camera (~2.7 cm/texel — enough for boot TREAD bars)
    constexpr float kEyeUp = 100.f;
    constexpr float kZNear = 1.f;
    constexpr float kZFar  = 350.f;
    Fmatrix       s_vp, s_prevVP;

    struct Buf { VkImage img = VK_NULL_HANDLE; VmaAllocation alloc = VK_NULL_HANDLE; VkImageView view = VK_NULL_HANDLE; };
    Buf s_state, s_scratch;

    VkSampler             s_sampler    = VK_NULL_HANDLE;
    VkDescriptorSetLayout s_setLayout  = VK_NULL_HANDLE;
    VkDescriptorPool      s_pool       = VK_NULL_HANDLE;
    VkDescriptorSet       s_set        = VK_NULL_HANDLE;
    VkPipelineLayout      s_pipeLayout = VK_NULL_HANDLE;
    VkPipeline            s_pipe       = VK_NULL_HANDLE;

    // Stamps live in a BUFFER (not push constants) so there's no per-frame count cap —
    // many NPC feet + items all stamp the same frame. Per-in-flight-slot region (dynamic
    // UBO offset) so the CPU never overwrites a region the GPU is still reading.
    constexpr u32 kFramesInFlight = CVulkanCommandManager::FRAMES_IN_FLIGHT;
    constexpr u32 kMaxStamps = 64;
    struct StampsUBO {
        float sPos[kMaxStamps][4];    // xyz = world pos, w = radius (m)
        float sPar[kMaxStamps][4];    // x = press strength (1 foot .. ~0.34 item)
    };
    constexpr VkDeviceSize kStampStride = (sizeof(StampsUBO) + 255) & ~VkDeviceSize(255);
    VkBuffer       s_stampBuf    = VK_NULL_HANDLE;
    VmaAllocation  s_stampAlloc  = VK_NULL_HANDLE;
    u8*            s_stampMapped = nullptr;

    struct PushConstants {
        Fmatrix curInvVP;
        Fmatrix prevVP;
        Fmatrix rainVP;               // world -> rain ndc (terrain-height gate)
        float   p0[4];                // N, decay, hasPrev, stampCount
        float   p1[4];                // xy = movement dir (world XZ), z = berm max (frac), w = ground gate (m)
        float   p2[4];                // x = rain eyeY, y = rain zRange, z/w unused
    };

    float s_prevCamX = 0.f, s_prevCamZ = 0.f;
    bool  s_haveCam  = false;

    void barrier(VkCommandBuffer cmd, VkImage img, VkImageLayout oldL, VkImageLayout newL,
                 VkPipelineStageFlags srcS, VkPipelineStageFlags dstS, VkAccessFlags srcA, VkAccessFlags dstA)
    {
        VkImageMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout = oldL; b.newLayout = newL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = img;
        b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        b.srcAccessMask = srcA; b.dstAccessMask = dstA;
        vkCmdPipelineBarrier(cmd, srcS, dstS, 0, 0, nullptr, 0, nullptr, 1, &b);
    }

    bool createImage(VkImageUsageFlags usage, Buf& out)
    {
        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = VK_FORMAT_R16_SFLOAT;
        ici.extent = { kSize, kSize, 1 };
        ici.mipLevels = 1; ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = usage;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        if (vmaCreateImage(VulkanHW.m_Allocator, &ici, &aci, &out.img, &out.alloc, nullptr) != VK_SUCCESS)
            return false;
        VkImageViewCreateInfo vci{};
        vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vci.image = out.img; vci.viewType = VK_IMAGE_VIEW_TYPE_2D; vci.format = VK_FORMAT_R16_SFLOAT;
        vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        return vkCreateImageView(VulkanHW.m_Device, &vci, nullptr, &out.view) == VK_SUCCESS;
    }

    void destroyBuf(Buf& b)
    {
        if (b.view) { vkDestroyImageView(VulkanHW.m_Device, b.view, nullptr); b.view = VK_NULL_HANDLE; }
        if (b.img)  { vmaDestroyImage(VulkanHW.m_Allocator, b.img, b.alloc); b.img = VK_NULL_HANDLE; }
    }

    // compute write (GENERAL) -> copy scratch to state -> back to working layouts.
    void copyBack(VkCommandBuffer cmd)
    {
        barrier(cmd, s_scratch.img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        barrier(cmd, s_state.img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        VkImageCopy cp{};
        cp.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        cp.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        cp.extent = { kSize, kSize, 1 };
        vkCmdCopyImage(cmd, s_scratch.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s_state.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);
        barrier(cmd, s_state.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                    | VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT
                    | VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT     // TCS press-gates the mud tess
                    | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,                  // snow mesh VS samples it
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        barrier(cmd, s_scratch.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT);
    }

    void computeVP()
    {
        const Fvector dir{ 0.f, -1.f, 0.f };
        const Fvector up { 0.f,  0.f, 1.f };
        Fvector eye; eye.set(Device.vCameraPosition.x, Device.vCameraPosition.y + kEyeUp, Device.vCameraPosition.z);
        Fmatrix view; view.build_camera_dir(eye, dir, up);
        const float texel = (2.f * kHalf) / float(kSize);   // texel snap -> stable reprojection
        view.c.x = floorf(view.c.x / texel) * texel;
        view.c.y = floorf(view.c.y / texel) * texel;
        Fmatrix proj; proj.build_projection_ortho(2.f * kHalf, 2.f * kHalf, kZNear, kZFar);
        s_vp.mul(proj, view);
    }
}

bool Ready() { return s_inited && !s_failed; }
VkImageView GetView()    { return s_state.view; }
VkSampler   GetSampler() { return s_sampler; }
const Fmatrix& GetVP()   { return s_vp; }
u32         Size()       { return kSize; }
float       Half()       { return kHalf; }

bool Init()
{
    if (s_inited) return true;
    if (s_failed) return false;
    if (ShadowMap::GetRainView() == VK_NULL_HANDLE) return false;   // need the rain map for the ground gate (retry next frame)
    s_vp.identity(); s_prevVP.identity();

    const VkImageUsageFlags stateUse = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    const VkImageUsageFlags scrUse   = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if (!createImage(stateUse, s_state) || !createImage(scrUse, s_scratch)) {
        Msg("![VK Deform] image create failed"); s_failed = true; return false;
    }

    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod = 0.25f;
    if (vkCreateSampler(VulkanHW.m_Device, &si, nullptr, &s_sampler) != VK_SUCCESS) {
        Msg("![VK Deform] sampler create failed"); s_failed = true; return false;
    }

    if (!g_ShaderManager) { Msg("![VK Deform] g_ShaderManager null"); s_failed = true; return false; }

    // Per-slot stamp UBO (host-visible, persistently mapped).
    {
        VkBufferCreateInfo bci{};
        bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bci.size = kStampStride * kFramesInFlight;
        bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo info{};
        if (vmaCreateBuffer(VulkanHW.m_Allocator, &bci, &aci, &s_stampBuf, &s_stampAlloc, &info) != VK_SUCCESS) {
            Msg("![VK Deform] stamp buffer create failed"); s_failed = true; return false;
        }
        s_stampMapped = (u8*)info.pMappedData;
    }

    VkDescriptorSetLayoutBinding b[4]{};
    b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; b[0].descriptorCount = 1; b[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    b[1].binding = 1; b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;          b[1].descriptorCount = 1; b[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    b[2].binding = 2; b[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC; b[2].descriptorCount = 1; b[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    b[3].binding = 3; b[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; b[3].descriptorCount = 1; b[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo slci{};
    slci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    slci.bindingCount = 4; slci.pBindings = b;
    vkCreateDescriptorSetLayout(VulkanHW.m_Device, &slci, nullptr, &s_setLayout);

    VkDescriptorPoolSize ps[3]{};
    ps[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; ps[0].descriptorCount = 2;   // prev + rain
    ps[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;          ps[1].descriptorCount = 1;
    ps[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC; ps[2].descriptorCount = 1;
    VkDescriptorPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = 1; pci.poolSizeCount = 3; pci.pPoolSizes = ps;
    vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &s_pool);

    VkDescriptorSetAllocateInfo dai{};
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = s_pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &s_setLayout;
    if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, &s_set) != VK_SUCCESS) {
        Msg("![VK Deform] descriptor alloc failed"); s_failed = true; return false;
    }

    VkDescriptorImageInfo ii[2] = {
        { s_sampler, s_state.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL },                 // 0 prev (sampled)
        { VK_NULL_HANDLE, s_scratch.view, VK_IMAGE_LAYOUT_GENERAL },                           // 1 out (storage)
    };
    VkDescriptorImageInfo rainI{ ShadowMap::GetSampler(), ShadowMap::GetRainView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkDescriptorBufferInfo sbI{ s_stampBuf, 0, sizeof(StampsUBO) };
    VkWriteDescriptorSet w[4]{};
    for (int i = 0; i < 2; ++i) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = s_set; w[i].dstBinding = (u32)i; w[i].descriptorCount = 1;
        w[i].descriptorType = (i == 0) ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w[i].pImageInfo = &ii[i];
    }
    w[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[2].dstSet = s_set; w[2].dstBinding = 2; w[2].descriptorCount = 1;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC; w[2].pBufferInfo = &sbI;
    w[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[3].dstSet = s_set; w[3].dstBinding = 3; w[3].descriptorCount = 1;
    w[3].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[3].pImageInfo = &rainI;
    vkUpdateDescriptorSets(VulkanHW.m_Device, 4, w, 0, nullptr);

    VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants) };
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1; plci.pSetLayouts = &s_setLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_pipeLayout);

    VkShaderModule cs = g_ShaderManager->Load("deform_stamp.comp.spv");
    if (cs == VK_NULL_HANDLE) { Msg("![VK Deform] deform_stamp.comp.spv load failed"); s_failed = true; return false; }
    VkComputePipelineCreateInfo cpi{};
    cpi.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpi.stage.module = cs; cpi.stage.pName = "main";
    cpi.layout = s_pipeLayout;
    if (vkCreateComputePipelines(VulkanHW.m_Device, VK::PipelineCache::GetCacheObject(), 1, &cpi, nullptr, &s_pipe) != VK_SUCCESS) {
        Msg("![VK Deform] compute pipeline create failed"); s_failed = true; return false;
    }

    s_inited = true; s_first = true;
    Msg("[VK Deform] init OK (%ux%u R16F, +-%.0f m, %.1f cm/texel)", kSize, kSize, kHalf, 200.f * kHalf / float(kSize));
    return true;
}

void Dispatch(VkCommandBuffer cmd, const Stamp* stamps, u32 count)
{
    if (!s_inited && !Init()) return;

    computeVP();

    if (s_first) {
        barrier(cmd, s_state.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, VK_ACCESS_SHADER_READ_BIT);
        barrier(cmd, s_scratch.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, VK_ACCESS_SHADER_WRITE_BIT);
    }

    Fmatrix curInv; curInv.invert(s_vp);
    const float life = (ps_r_snow_deform_time > 0.1f) ? ps_r_snow_deform_time : 0.1f;
    const float dt   = (Device.fTimeDelta < 0.1f) ? Device.fTimeDelta : 0.1f;

    const u32 n = (count < kMaxStamps) ? count : kMaxStamps;

    // Fill this in-flight slot's stamp region (per-slot -> no CPU/GPU overwrite race).
    const u32 slot = CommandManager.GetCurrentFrame() % kFramesInFlight;
    StampsUBO* su = reinterpret_cast<StampsUBO*>(s_stampMapped + slot * kStampStride);
    for (u32 i = 0; i < n; ++i) {
        su->sPos[i][0] = stamps[i].pos.x;
        su->sPos[i][1] = stamps[i].pos.y;     // world Y for the ground gate
        su->sPos[i][2] = stamps[i].pos.z;
        su->sPos[i][3] = stamps[i].radius;
        su->sPar[i][0] = stamps[i].pressDepth;
        su->sPar[i][1] = stamps[i].dirX;      // boot facing (0,0 = round item contact)
        su->sPar[i][2] = stamps[i].dirZ;
        su->sPar[i][3] = 0.f;
    }

    PushConstants pc{};
    pc.curInvVP = curInv;
    pc.prevVP   = s_prevVP;
    pc.rainVP   = ShadowMap::GetRainVP();
    pc.p0[0] = float(kSize);
    pc.p0[1] = dt / life;                  // decay this frame
    pc.p0[2] = s_first ? 0.f : 1.f;        // hasPrev
    pc.p0[3] = float(n);

    // Movement direction (world XZ) from the camera delta -> the berm is plowed forward.
    float mdx = 0.f, mdz = 0.f;
    {
        const float cx = Device.vCameraPosition.x, cz = Device.vCameraPosition.z;
        if (s_haveCam) {
            const float dx = cx - s_prevCamX, dz = cz - s_prevCamZ;
            const float len = sqrtf(dx * dx + dz * dz);
            if (len > 0.012f) { mdx = dx / len; mdz = dz / len; }   // ignore tiny jitter / standing still
        }
        s_prevCamX = cx; s_prevCamZ = cz; s_haveCam = true;
    }
    pc.p1[0] = mdx; pc.p1[1] = mdz;
    pc.p1[2] = ps_r_snow_berm;   // berm max (× dent depth in the mesh) — front ridge ~1.5×, sides ~0.4×
    pc.p1[3] = 0.45f;            // ground gate (m): contact must be within this of the terrain to stamp
    pc.p2[0] = ShadowMap::RainEyeY();
    pc.p2[1] = kZFar - kZNear;   // rain ortho zRange (349)
    pc.p2[2] = ps_r_snow_rough;  // trail/print imperfection
    pc.p2[3] = 0.f;

    const u32 dynOffset = slot * (u32)kStampStride;
    const u32 groups = (kSize + 7u) / 8u;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_pipeLayout, 0, 1, &s_set, 1, &dynOffset);
    vkCmdPushConstants(cmd, s_pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, groups, groups, 1);

    copyBack(cmd);

    s_prevVP = s_vp;
    s_first  = false;
}

void Destroy()
{
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
    if (s_pipe)       { vkDestroyPipeline(VulkanHW.m_Device, s_pipe, nullptr); s_pipe = VK_NULL_HANDLE; }
    if (s_pipeLayout) { vkDestroyPipelineLayout(VulkanHW.m_Device, s_pipeLayout, nullptr); s_pipeLayout = VK_NULL_HANDLE; }
    if (s_pool)       { vkDestroyDescriptorPool(VulkanHW.m_Device, s_pool, nullptr); s_pool = VK_NULL_HANDLE; }
    if (s_setLayout)  { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_setLayout, nullptr); s_setLayout = VK_NULL_HANDLE; }
    if (s_sampler)    { vkDestroySampler(VulkanHW.m_Device, s_sampler, nullptr); s_sampler = VK_NULL_HANDLE; }
    if (s_stampBuf)   { vmaDestroyBuffer(VulkanHW.m_Allocator, s_stampBuf, s_stampAlloc); s_stampBuf = VK_NULL_HANDLE; s_stampMapped = nullptr; }
    destroyBuf(s_state); destroyBuf(s_scratch);
    s_set = VK_NULL_HANDLE;
    s_inited = false; s_failed = false; s_first = true;
}

}}  // namespace VK::Deform
