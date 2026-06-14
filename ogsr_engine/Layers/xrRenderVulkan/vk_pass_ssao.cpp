// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan — GTAO pass. See vk_pass_ssao.h.
#include "stdafx.h"
#include "vk_pass_ssao.h"
#include "vk_swapchain.h"          // Swapchain.m_DepthView (prepass depth)
#include "vk_shaders.h"            // g_ShaderManager
#include "vk_pipeline_cache.h"     // PipelineCache::GetCacheObject
#include "vk_barriers.h"           // ImageBarrier
#include "vk_buffer.h"             // CVulkanBuffer (debug AO readback)
#include "vk_command_buffer.h"     // CommandManager.GetCurrentFrame()
#include "HW_Vulkan.h"
#include "../../xr_3da/device.h"   // Device camera basis + mProject

extern u32 ps_r_ao_quality;        // r2_ssao token (0 off / 1 low / 2 med / 3 high / 4 ultra)
extern int ps_r_ssao_enable;       // r_ssao — global GTAO on/off (GLOBAL scope: block-scope extern inside the namespace would mangle as VK::SSAOPass::* → LNK2001)
extern int ps_r_ssao_debug;        // r_ssao_debug — also enables the readback stats below

namespace VK {

ProjTerms DeriveProjTerms(const Fmatrix& M)
{
    ProjTerms t{};
    // w column (M_i4, i=1..3) = view dir scaled — pick the largest component's
    // row for the _33 division so a near-axis-aligned camera stays stable.
    const float w[3] = { M._14, M._24, M._34 };
    const float z[3] = { M._13, M._23, M._33 };
    int bi = 0;
    if (std::fabs(w[1]) > std::fabs(w[bi])) bi = 1;
    if (std::fabs(w[2]) > std::fabs(w[bi])) bi = 2;
    t.p33 = (std::fabs(w[bi]) > 1e-9f) ? z[bi] / w[bi] : 1.f;
    t.p43 = M._43 - M._44 * t.p33;
    const float p11 = std::sqrt(M._11 * M._11 + M._21 * M._21 + M._31 * M._31);
    const float p22 = std::sqrt(M._12 * M._12 + M._22 * M._22 + M._32 * M._32);
    t.tanX = (p11 > 1e-9f) ? 1.f / p11 : 1.f;
    t.tanY = (p22 > 1e-9f) ? 1.f / p22 : 1.f;
    t.dir.set(w[0], w[1], w[2]);          t.dir.normalize_safe();
    t.right.set(M._11, M._21, M._31);     t.right.normalize_safe();
    t.top.set(M._12, M._22, M._32);       t.top.normalize_safe();
    return t;
}

namespace SSAOPass {

namespace {
    constexpr u32 kFramesInFlight = CVulkanCommandManager::FRAMES_IN_FLIGHT;

    // R4 gtao.h: GTAO_RADIUS 4 (falloff range; effective horizon-march radius
    // is half that). Strength multiplies the occlusion in the receivers.
    constexpr float kRadius   = 4.0f;
    constexpr float kStrength = 1.0f;

    bool                  s_inited    = false;
    bool                  s_failed    = false;
    VkDescriptorSetLayout s_setLayout = VK_NULL_HANDLE;   // 0 = depth, 1 = raw AO (blur only)
    VkDescriptorPool      s_pool      = VK_NULL_HANDLE;
    VkDescriptorSet       s_setGtao[kFramesInFlight] = {};
    VkDescriptorSet       s_setBlur[kFramesInFlight] = {};
    VkPipelineLayout      s_layout    = VK_NULL_HANDLE;
    VkPipeline            s_pipeGtao  = VK_NULL_HANDLE;
    VkPipeline            s_pipeBlur  = VK_NULL_HANDLE;
    VkSampler             s_sampNear  = VK_NULL_HANDLE;   // nearest/clamp — depth + raw AO taps
    VkSampler             s_sampLin   = VK_NULL_HANDLE;   // linear/clamp — receivers upsample

    // Half-res ping-pong: [0] = final (blurred), [1] = raw GTAO.
    VkImage       s_img[2]   = {};
    VmaAllocation s_alloc[2] = {};
    VkImageView   s_view[2]  = {};
    VkExtent2D    s_extent   = {};
    u32           s_generation = 0;
    bool          s_first[2] = { true, true };

    // Debug readback (active while r_ssao_debug != 0): every ~300 frames the
    // final AO image is copied to this host buffer and min/max/avg logged —
    // verifies the GTAO output end-to-end without a GPU capture tool.
    CVulkanBuffer s_dbgBuf;
    u8*           s_dbgMap = nullptr;
    int           s_dbgCountdown = -1;   // frames until the mapped copy is safely written
    u32           s_dbgCooldown  = 0;

    struct SSAOPush {
        float camDir[4];
        float camRightT[4];   // xyz = right * tan(fovX/2), w = tan(fovX/2)
        float camTopT[4];     // xyz = top   * tan(fovY/2), w = tan(fovY/2)
        float zp[4];          // proj _33, proj _43, radius, samples/side
        float res[4];         // AO size xy, 1/AO size zw
        float dbg[4];         // x = r_ssao_debug mode (2 = depth view, 3 = normal view)
    };
    static_assert(sizeof(SSAOPush) == 96, "must match ssao.frag / ssao_blur.frag PC blocks");

    u32 SampleCount()
    {
        // r2_ssao token → GTAO_SAMPLE (R4: low 2 / medium 3 / high+ 4). The
        // user's carried-over user.ltx may have it off — the Vulkan renderer
        // defaults to medium then (set r2_ssao to override; st_opt_off is
        // honoured once any other value was chosen at least once... for now
        // 0 == "unset" → medium, since legacy configs predate this pass).
        const u32 q = ps_r_ao_quality ? ps_r_ao_quality : 2;
        return q <= 1 ? 2 : (q == 2 ? 3 : 4);
    }

    void DestroyRTs()
    {
        if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
        for (u32 i = 0; i < 2; ++i) {
            if (s_view[i]) { vkDestroyImageView(VulkanHW.m_Device, s_view[i], nullptr); s_view[i] = VK_NULL_HANDLE; }
            if (s_img[i])  { vmaDestroyImage(VulkanHW.m_Allocator, s_img[i], s_alloc[i]); s_img[i] = VK_NULL_HANDLE; s_alloc[i] = VK_NULL_HANDLE; }
            s_first[i] = true;
        }
        s_extent = {};
    }

    bool EnsureRTs(VkExtent2D sceneExtent)
    {
        VkExtent2D want{ sceneExtent.width  / 2 ? sceneExtent.width  / 2 : 1,
                         sceneExtent.height / 2 ? sceneExtent.height / 2 : 1 };
        if (s_img[0] && want.width == s_extent.width && want.height == s_extent.height)
            return true;
        DestroyRTs();
        s_extent = want;
        for (u32 i = 0; i < 2; ++i) {
            VkImageCreateInfo ici{};
            ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            ici.imageType = VK_IMAGE_TYPE_2D;
            ici.format = VK_FORMAT_R8_UNORM;
            ici.extent = { want.width, want.height, 1 };
            ici.mipLevels = 1; ici.arrayLayers = 1;
            ici.samples = VK_SAMPLE_COUNT_1_BIT;
            ici.tiling = VK_IMAGE_TILING_OPTIMAL;
            ici.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                      | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;   // debug readback
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            VmaAllocationCreateInfo aci{};
            aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
            if (vmaCreateImage(VulkanHW.m_Allocator, &ici, &aci, &s_img[i], &s_alloc[i], nullptr) != VK_SUCCESS) {
                Msg("![VK SSAO] RT %u create failed", i); return false;
            }
            VkImageViewCreateInfo vci{};
            vci.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            vci.image = s_img[i];
            vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vci.format = VK_FORMAT_R8_UNORM;
            vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            vci.subresourceRange.levelCount = 1;
            vci.subresourceRange.layerCount = 1;
            if (vkCreateImageView(VulkanHW.m_Device, &vci, nullptr, &s_view[i]) != VK_SUCCESS) {
                Msg("![VK SSAO] view %u create failed", i); return false;
            }
        }
        // Host buffer for the debug readback (one byte per AO texel).
        s_dbgBuf.Destroy();
        s_dbgMap = nullptr;
        s_dbgCountdown = -1;
        s_dbgBuf.Create(VkDeviceSize(want.width) * want.height,
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
        s_dbgMap = static_cast<u8*>(s_dbgBuf.Map());

        ++s_generation;
        Msg("[VK SSAO] RTs ready (%ux%u half-res, gen %u, samples %u)",
            want.width, want.height, s_generation, SampleCount());
        return true;
    }

    VkPipeline CreatePipe(VkShaderModule vs, VkShaderModule fs)
    {
        VkPipelineVertexInputStateCreateInfo vi{}; vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        VkPipelineShaderStageCreateInfo st[2]{};
        st[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;   st[0].module = vs; st[0].pName = "main";
        st[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; st[1].module = fs; st[1].pName = "main";
        VkPipelineInputAssemblyStateCreateInfo ia{}; ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vp{}; vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vp.viewportCount = 1; vp.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{}; rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE;
        rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;
        VkPipelineMultisampleStateCreateInfo ms{}; ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{}; ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        VkPipelineColorBlendAttachmentState ba{};
        ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT;
        VkPipelineColorBlendStateCreateInfo cb{}; cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1; cb.pAttachments = &ba;
        VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynState{}; dynState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynState.dynamicStateCount = 2; dynState.pDynamicStates = dyn;
        VkFormat fmt = VK_FORMAT_R8_UNORM;
        VkPipelineRenderingCreateInfo prci{};
        prci.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        prci.colorAttachmentCount = 1; prci.pColorAttachmentFormats = &fmt;
        VkGraphicsPipelineCreateInfo pi{};
        pi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pi.pNext = &prci; pi.stageCount = 2; pi.pStages = st;
        pi.pVertexInputState = &vi; pi.pInputAssemblyState = &ia; pi.pViewportState = &vp;
        pi.pRasterizationState = &rs; pi.pMultisampleState = &ms; pi.pDepthStencilState = &ds;
        pi.pColorBlendState = &cb; pi.pDynamicState = &dynState; pi.layout = s_layout;
        VkPipeline out = VK_NULL_HANDLE;
        if (vkCreateGraphicsPipelines(VulkanHW.m_Device, PipelineCache::GetCacheObject(), 1, &pi, nullptr, &out) != VK_SUCCESS)
            Msg("![VK SSAO] pipeline create failed");
        return out;
    }

    // One fullscreen draw into dst at the half-res extent.
    void Draw(VkCommandBuffer cmd, VkImageView dst, VkPipeline pipe, VkDescriptorSet set,
              const SSAOPush& push)
    {
        VkRenderingAttachmentInfo cAtt{};
        cAtt.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        cAtt.imageView = dst;
        cAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        cAtt.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        cAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        VkRenderingInfo ri{};
        ri.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        ri.renderArea.extent = s_extent;
        ri.layerCount = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments = &cAtt;
        vkCmdBeginRendering(cmd, &ri);
        VkViewport vp{ 0.f, 0.f, (float)s_extent.width, (float)s_extent.height, 0.f, 1.f };
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc{ {0,0}, s_extent };
        vkCmdSetScissor(cmd, 0, 1, &sc);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_layout, 0, 1, &set, 0, nullptr);
        vkCmdPushConstants(cmd, s_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
        vkCmdDraw(cmd, 3, 1, 0, 0);
        vkCmdEndRendering(cmd);
    }
}  // anon namespace

// Set once the first Execute actually rendered AO — until then (and whenever
// the prepass is unavailable) Strength() keeps the receivers on plain 1.0.
static bool s_hasResult = false;

bool        Enabled()       { return s_inited && !s_failed && ps_r_ssao_enable != 0; }
VkImageView GetResultView() { return Enabled() ? s_view[0] : VK_NULL_HANDLE; }  // disabled → EnvLight uses white fallback
VkSampler   GetSampler()    { return s_sampLin; }
u32         Generation()    { return s_generation; }
float       Strength()      { return (Enabled() && s_hasResult && s_view[0]) ? kStrength : 0.f; }

bool EnsureTargets(VkExtent2D sceneExtent)
{
    if (!Enabled()) return false;
    return EnsureRTs(sceneExtent);
}

bool Init()
{
    if (s_inited) return !s_failed;
    s_inited = true;

    if (!g_ShaderManager) { s_failed = true; return false; }
    VkShaderModule vs   = g_ShaderManager->Load("tonemap.vert.spv");   // shared fullscreen triangle
    VkShaderModule gtao = g_ShaderManager->Load("ssao.frag.spv");
    VkShaderModule blur = g_ShaderManager->Load("ssao_blur.frag.spv");
    if (vs == VK_NULL_HANDLE || gtao == VK_NULL_HANDLE || blur == VK_NULL_HANDLE) {
        Msg("![VK SSAO] ssao.{frag,blur}.spv missing — SSAO disabled");
        s_failed = true; return false;
    }

    // Set: 0 = scene depth, 1 = raw AO (only the blur pipeline reads it; the
    // GTAO pipeline doesn't statically use binding 1, so its stale layout
    // during the GTAO draw is legal).
    VkDescriptorSetLayoutBinding b[2]{};
    for (u32 i = 0; i < 2; ++i) {
        b[i].binding = i; b[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo slci{};
    slci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    slci.bindingCount = 2; slci.pBindings = b;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &slci, nullptr, &s_setLayout) != VK_SUCCESS) {
        s_failed = true; return false;
    }

    const u32 nSets = kFramesInFlight * 2;
    VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, nSets * 2 };
    VkDescriptorPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = nSets; pci.poolSizeCount = 1; pci.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &s_pool) != VK_SUCCESS) {
        s_failed = true; return false;
    }
    {
        VkDescriptorSetLayout layouts[kFramesInFlight * 2];
        for (u32 i = 0; i < nSets; ++i) layouts[i] = s_setLayout;
        VkDescriptorSet sets[kFramesInFlight * 2]{};
        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = s_pool; dai.descriptorSetCount = nSets; dai.pSetLayouts = layouts;
        if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, sets) != VK_SUCCESS) {
            s_failed = true; return false;
        }
        for (u32 i = 0; i < kFramesInFlight; ++i) {
            s_setGtao[i] = sets[i];
            s_setBlur[i] = sets[kFramesInFlight + i];
        }
    }

    auto makeSampler = [&](VkFilter f, VkSampler& out) {
        VkSamplerCreateInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        si.magFilter = f; si.minFilter = f;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxAnisotropy = 1.0f;
        return vkCreateSampler(VulkanHW.m_Device, &si, nullptr, &out) == VK_SUCCESS;
    };
    if (!makeSampler(VK_FILTER_NEAREST, s_sampNear) || !makeSampler(VK_FILTER_LINEAR, s_sampLin)) {
        Msg("![VK SSAO] sampler create failed"); s_failed = true; return false;
    }

    VkPushConstantRange pcr{ VK_SHADER_STAGE_FRAGMENT_BIT, 0, (u32)sizeof(SSAOPush) };
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1; plci.pSetLayouts = &s_setLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_layout) != VK_SUCCESS) {
        s_failed = true; return false;
    }

    s_pipeGtao = CreatePipe(vs, gtao);
    s_pipeBlur = CreatePipe(vs, blur);
    if (s_pipeGtao == VK_NULL_HANDLE || s_pipeBlur == VK_NULL_HANDLE) { s_failed = true; return false; }

    Msg("[VK SSAO] init OK (GTAO half-res, radius %.1f, r2_ssao quality %u)", kRadius, ps_r_ao_quality);
    return true;
}

void Execute(VkCommandBuffer cmd, VkExtent2D sceneExtent)
{
    if (!Enabled()) return;
    if (!EnsureRTs(sceneExtent)) return;

    // Refresh this slot's sets (fence-guarded): depth view can change on
    // swapchain recreate, AO views on RT recreate. 4 trivial writes per frame.
    const u32 slot = CommandManager.GetCurrentFrame() % kFramesInFlight;
    {
        VkDescriptorImageInfo depthI{ s_sampNear, Swapchain.m_DepthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkDescriptorImageInfo rawI  { s_sampNear, s_view[1],             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet w[4]{};
        VkDescriptorSet dst[4] = { s_setGtao[slot], s_setGtao[slot], s_setBlur[slot], s_setBlur[slot] };
        const VkDescriptorImageInfo* ii[4] = { &depthI, &rawI, &depthI, &rawI };
        for (u32 i = 0; i < 4; ++i) {
            w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = dst[i]; w[i].dstBinding = i & 1; w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[i].pImageInfo = ii[i];
        }
        vkUpdateDescriptorSets(VulkanHW.m_Device, 4, w, 0, nullptr);
    }

    auto toColor = [&](u32 i) {
        ImageBarrier(cmd, s_img[i],
                     s_first[i] ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        s_first[i] = false;
    };
    auto toRead = [&](u32 i) {
        ImageBarrier(cmd, s_img[i],
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    };

    // Everything derived from the matrix that RENDERED the prepass depth —
    // Device.mProject is not maintained on this path (reads as identity).
    const ProjTerms pt = DeriveProjTerms(Device.mFullTransform);
    SSAOPush push{};
    push.camDir[0] = pt.dir.x; push.camDir[1] = pt.dir.y; push.camDir[2] = pt.dir.z;
    push.camRightT[0] = pt.right.x * pt.tanX; push.camRightT[1] = pt.right.y * pt.tanX;
    push.camRightT[2] = pt.right.z * pt.tanX; push.camRightT[3] = pt.tanX;
    push.camTopT[0]   = pt.top.x  * pt.tanY;  push.camTopT[1]   = pt.top.y  * pt.tanY;
    push.camTopT[2]   = pt.top.z  * pt.tanY;  push.camTopT[3]   = pt.tanY;
    push.zp[0] = pt.p33;
    push.zp[1] = pt.p43;
    push.zp[2] = kRadius;
    push.zp[3] = float(SampleCount());
    push.res[0] = float(s_extent.width);
    push.res[1] = float(s_extent.height);
    push.res[2] = 1.f / float(s_extent.width);
    push.res[3] = 1.f / float(s_extent.height);
    push.dbg[0] = float(ps_r_ssao_debug);

    // One-time dump of the reconstruction inputs — sanity vs the offline test
    // (expect _33 ≈ 1.0006, _43 ≈ -0.2 for zn 0.2 / zf 350, tan ≈ 0.6-1.1).
    static bool s_pushDiag = false;
    if (!s_pushDiag) { s_pushDiag = true;
        Msg("[VK SSAO] push (derived): _33=%.6f _43=%.6f tanX=%.4f tanY=%.4f dir=(%.2f,%.2f,%.2f)",
            pt.p33, pt.p43, pt.tanX, pt.tanY, pt.dir.x, pt.dir.y, pt.dir.z);
    }

    // 1) GTAO: depth → raw [1].
    toColor(1);
    Draw(cmd, s_view[1], s_pipeGtao, s_setGtao[slot], push);
    toRead(1);

    // 2) Depth-aware 3×3 blur: raw [1] → final [0].
    toColor(0);
    Draw(cmd, s_view[0], s_pipeBlur, s_setBlur[slot], push);
    toRead(0);

    s_hasResult = true;

    // Debug readback (r_ssao_debug): periodically copy the final AO to the host
    // buffer; a few frames later (fence-safe by frames-in-flight) log stats.
    if (ps_r_ssao_debug && s_dbgMap) {
        if (s_dbgCountdown > 0 && --s_dbgCountdown == 0) {
            const u32 n = s_extent.width * s_extent.height;
            u32 mn = 255, mx = 0; u64 sum = 0; u32 below200 = 0;
            for (u32 i = 0; i < n; ++i) {
                const u8 v = s_dbgMap[i];
                mn = std::min(mn, (u32)v); mx = std::max(mx, (u32)v);
                sum += v; below200 += (v < 200);
            }
            Msg("[VK SSAO] readback mode %d %ux%u: min %u max %u avg %.1f, <200: %.1f%%",
                ps_r_ssao_debug, s_extent.width, s_extent.height, mn, mx,
                double(sum) / n, 100.0 * below200 / n);
            for (u32 r = 1; r <= 3; ++r) {   // rows at 25/50/75% height, 5 taps each
                const u32 cy = s_extent.height * r / 4;
                Msg("[VK SSAO]   row %u%%: %u %u %u %u %u", r * 25,
                    s_dbgMap[cy * s_extent.width + s_extent.width / 6],
                    s_dbgMap[cy * s_extent.width + s_extent.width / 3],
                    s_dbgMap[cy * s_extent.width + s_extent.width / 2],
                    s_dbgMap[cy * s_extent.width + s_extent.width * 2 / 3],
                    s_dbgMap[cy * s_extent.width + s_extent.width * 5 / 6]);
            }
        }
        if (s_dbgCooldown == 0 && s_dbgCountdown <= 0) {
            ImageBarrier(cmd, s_img[0], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            VkBufferImageCopy bic{};
            bic.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            bic.imageExtent      = { s_extent.width, s_extent.height, 1 };
            vkCmdCopyImageToBuffer(cmd, s_img[0], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   s_dbgBuf.GetHandle(), 1, &bic);
            ImageBarrier(cmd, s_img[0], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            s_dbgCountdown = 5;     // read after the in-flight fences cycled
            s_dbgCooldown  = 300;   // ~one log line per few seconds
        }
        if (s_dbgCooldown) --s_dbgCooldown;
    }
}

void Destroy()
{
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
    DestroyRTs();
    s_dbgBuf.Destroy(); s_dbgMap = nullptr; s_dbgCountdown = -1; s_dbgCooldown = 0;
    if (s_pipeGtao)  { vkDestroyPipeline(VulkanHW.m_Device, s_pipeGtao, nullptr); s_pipeGtao = VK_NULL_HANDLE; }
    if (s_pipeBlur)  { vkDestroyPipeline(VulkanHW.m_Device, s_pipeBlur, nullptr); s_pipeBlur = VK_NULL_HANDLE; }
    if (s_layout)    { vkDestroyPipelineLayout(VulkanHW.m_Device, s_layout, nullptr); s_layout = VK_NULL_HANDLE; }
    if (s_sampNear)  { vkDestroySampler(VulkanHW.m_Device, s_sampNear, nullptr); s_sampNear = VK_NULL_HANDLE; }
    if (s_sampLin)   { vkDestroySampler(VulkanHW.m_Device, s_sampLin, nullptr); s_sampLin = VK_NULL_HANDLE; }
    if (s_pool)      { vkDestroyDescriptorPool(VulkanHW.m_Device, s_pool, nullptr); s_pool = VK_NULL_HANDLE; }
    if (s_setLayout) { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_setLayout, nullptr); s_setLayout = VK_NULL_HANDLE; }
    for (u32 i = 0; i < kFramesInFlight; ++i) { s_setGtao[i] = VK_NULL_HANDLE; s_setBlur[i] = VK_NULL_HANDLE; }
    s_inited = false; s_failed = false; s_generation = 0; s_hasResult = false;
}

}}  // namespace VK::SSAOPass
