// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan — screen-space motion vectors. See vk_motionvec.h.
#include "stdafx.h"
#include "vk_motionvec.h"
#include "vk_pass_ssao.h"          // VK::DeriveProjTerms / VK::ProjTerms (shared depth→world basis)
#include "vk_pass_context.h"       // VK::FrameContext (dynamic MV pass)
#include "vk_pass_skinned.h"       // VK::Skinned_RenderMotion (NPC animation MV)
#include "vk_swapchain.h"          // Swapchain.m_DepthView / m_Format
#include "vk_shaders.h"            // g_ShaderManager
#include "vk_barriers.h"           // ImageBarrier
#include "vk_command_buffer.h"     // CommandManager.GetCurrentFrame()
#include "vk_fullscreen.h"         // VK::Fullscreen — shared fullscreen pipeline + draw
#include "HW_Vulkan.h"
#include "../../xr_3da/device.h"   // Device.mFullTransform + vCameraPosition

extern int   ps_r_motion_vectors;  // r_motion_vectors — global MV pass on/off
extern int   ps_r_mv_debug;        // r_mv_debug       — false-colour overlay on/off
extern float ps_r_mv_debug_scale;  // r_mv_debug_scale — overlay magnitude scale

namespace VK { namespace MotionVec {

namespace {
    constexpr u32      kFramesInFlight = CVulkanCommandManager::FRAMES_IN_FLIGHT;
    constexpr VkFormat kFormat = VK_FORMAT_R16G16_SFLOAT;

    bool                  s_inited  = false;
    bool                  s_failed  = false;

    VkDescriptorSetLayout s_setLayout = VK_NULL_HANDLE;   // binding 0 = sampler2D (depth for MV / MV for debug)
    VkDescriptorPool      s_pool      = VK_NULL_HANDLE;
    VkDescriptorSet       s_setMV [kFramesInFlight] = {}; // binds scene depth
    VkDescriptorSet       s_setDbg[kFramesInFlight] = {}; // binds the MV target
    VkSampler             s_samp      = VK_NULL_HANDLE;    // nearest/clamp

    VkPipelineLayout      s_mvLayout  = VK_NULL_HANDLE;    // set0 + 128B push (matrices)
    VkPipelineLayout      s_dbgLayout = VK_NULL_HANDLE;    // set0 + 16B push (scale)
    VkPipeline            s_pipeMV    = VK_NULL_HANDLE;
    VkPipeline            s_pipeDbg   = VK_NULL_HANDLE;

    // The RG16F motion target (full-res, one image — written + consumed within a frame).
    VkImage       s_img   = VK_NULL_HANDLE;
    VmaAllocation s_alloc = VK_NULL_HANDLE;
    VkImageView   s_view  = VK_NULL_HANDLE;
    VkExtent2D    s_extent = {};
    u32           s_generation = 0;
    bool          s_first  = true;     // first transition is from UNDEFINED

    // View-proj history. Both MV passes this frame (static fullscreen + dynamic
    // overlay) must use the SAME previous-frame matrix, so we roll it ONCE per
    // Device.dwFrame (the static Execute used to cache at its tail, which the later
    // dynamic pass would then read as the CURRENT frame — a one-frame bug).
    Fmatrix s_prevVP;        // previous frame's view-proj
    Fmatrix s_curVP;         // this frame's view-proj (becomes prev next frame)
    bool    s_vpValid = false;
    u32     s_vpFrame = u32(-1);

    void EnsurePrevVP()
    {
        if (s_vpFrame == Device.dwFrame) return;   // already rolled this frame
        s_prevVP  = s_vpValid ? s_curVP : Device.mFullTransform;   // first frame → zero motion
        s_curVP   = Device.mFullTransform;
        s_vpValid = true;
        s_vpFrame = Device.dwFrame;
    }

    struct MVPush {
        float camDir[4];     // xyz fwd, w eye.x
        float camRightT[4];  // xyz right*tanX, w eye.y
        float camTopT[4];    // xyz top*tanY, w eye.z
        float zp[4];         // _33, _43, 1/w, 1/h
        float prevVP[16];    // row-major Fmatrix copied verbatim (see motion_vec.frag)
    };
    static_assert(sizeof(MVPush) == 128, "must match motion_vec.frag PC block (Vulkan-guaranteed 128B push)");

    struct DbgPush { float p[4]; };   // x = display scale

    void DestroyTarget()
    {
        if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
        if (s_view)  { vkDestroyImageView(VulkanHW.m_Device, s_view, nullptr); s_view = VK_NULL_HANDLE; }
        if (s_img)   { vmaDestroyImage(VulkanHW.m_Allocator, s_img, s_alloc); s_img = VK_NULL_HANDLE; s_alloc = VK_NULL_HANDLE; }
        s_extent = {};
        s_first  = true;
    }
}

bool        Enabled()       { return s_inited && !s_failed && ps_r_motion_vectors != 0 && s_img != VK_NULL_HANDLE; }
VkImageView GetResultView() { return Enabled() ? s_view : VK_NULL_HANDLE; }
VkSampler   GetSampler()    { return s_samp; }
VkFormat    Format()        { return kFormat; }
u32         Generation()    { return s_generation; }

void EnsureSize(VkExtent2D extent)
{
    if (!s_inited || s_failed) return;
    if (s_img && extent.width == s_extent.width && extent.height == s_extent.height) return;
    DestroyTarget();
    if (extent.width == 0 || extent.height == 0) return;
    s_extent = extent;

    VkImageCreateInfo ici{};
    ici.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType     = VK_IMAGE_TYPE_2D;
    ici.format        = kFormat;
    ici.extent        = { extent.width, extent.height, 1 };
    ici.mipLevels     = 1; ici.arrayLayers = 1;
    ici.samples       = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling        = VK_IMAGE_TILING_OPTIMAL;
    ici.usage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (vmaCreateImage(VulkanHW.m_Allocator, &ici, &aci, &s_img, &s_alloc, nullptr) != VK_SUCCESS) {
        Msg("![VK MotionVec] image create failed"); s_img = VK_NULL_HANDLE; return;
    }
    VkImageViewCreateInfo vci{};
    vci.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vci.image    = s_img;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format   = kFormat;
    vci.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    vci.subresourceRange.levelCount = 1;
    vci.subresourceRange.layerCount = 1;
    if (vkCreateImageView(VulkanHW.m_Device, &vci, nullptr, &s_view) != VK_SUCCESS) {
        Msg("![VK MotionVec] view create failed"); DestroyTarget(); return;
    }
    ++s_generation;
    s_vpValid = false;   // history is stale across a resize
    Msg("[VK MotionVec] target ready (%ux%u RG16F, gen %u)", extent.width, extent.height, s_generation);
}

bool Init()
{
    if (s_inited) return !s_failed;
    s_inited = true;

    if (!g_ShaderManager) { s_failed = true; return false; }
    VkShaderModule vs  = g_ShaderManager->Load("tonemap.vert.spv");        // shared fullscreen triangle
    VkShaderModule fs  = g_ShaderManager->Load("motion_vec.frag.spv");
    VkShaderModule fsD = g_ShaderManager->Load("motion_vec_debug.frag.spv");
    if (vs == VK_NULL_HANDLE || fs == VK_NULL_HANDLE || fsD == VK_NULL_HANDLE) {
        Msg("![VK MotionVec] motion_vec.{frag,debug}.spv missing — MV disabled");
        s_failed = true; return false;
    }

    // One binding (sampler2D), reused: MV pass binds depth, debug binds the MV target.
    VkDescriptorSetLayoutBinding b{};
    b.binding = 0; b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b.descriptorCount = 1; b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo slci{};
    slci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    slci.bindingCount = 1; slci.pBindings = &b;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &slci, nullptr, &s_setLayout) != VK_SUCCESS) {
        s_failed = true; return false;
    }

    const u32 nSets = kFramesInFlight * 2;
    VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, nSets };
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
            s_setMV[i]  = sets[i];
            s_setDbg[i] = sets[kFramesInFlight + i];
        }
    }

    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_NEAREST; si.minFilter = VK_FILTER_NEAREST;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxAnisotropy = 1.0f;
    if (vkCreateSampler(VulkanHW.m_Device, &si, nullptr, &s_samp) != VK_SUCCESS) {
        Msg("![VK MotionVec] sampler create failed"); s_failed = true; return false;
    }

    auto makeLayout = [&](u32 pushSize, VkPipelineLayout& out) {
        VkPushConstantRange pcr{ VK_SHADER_STAGE_FRAGMENT_BIT, 0, pushSize };
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1; plci.pSetLayouts = &s_setLayout;
        plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
        return vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &out) == VK_SUCCESS;
    };
    if (!makeLayout((u32)sizeof(MVPush), s_mvLayout) || !makeLayout((u32)sizeof(DbgPush), s_dbgLayout)) {
        s_failed = true; return false;
    }

    s_pipeMV  = Fullscreen::CreatePipeline(vs, fs,  kFormat,           s_mvLayout,  Fullscreen::OpaqueAttachment(), "MotionVec");
    s_pipeDbg = Fullscreen::CreatePipeline(vs, fsD, Swapchain.m_Format, s_dbgLayout, Fullscreen::OpaqueAttachment(), "MotionVecDebug");
    if (s_pipeMV == VK_NULL_HANDLE || s_pipeDbg == VK_NULL_HANDLE) { s_failed = true; return false; }

    Msg("[VK MotionVec] init OK (RG16F, depth reconstruction, r_motion_vectors %d)", ps_r_motion_vectors);
    return true;
}

void Execute(VkCommandBuffer cmd, VkExtent2D extent)
{
    if (!s_inited || s_failed || ps_r_motion_vectors == 0) return;
    EnsureSize(extent);
    if (s_img == VK_NULL_HANDLE) return;

    EnsurePrevVP();

    const u32 slot = CommandManager.GetCurrentFrame() % kFramesInFlight;

    // (Re)bind the scene depth (its view can change on swapchain recreate).
    {
        VkDescriptorImageInfo depthI{ s_samp, Swapchain.m_DepthView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w.dstSet = s_setMV[slot]; w.dstBinding = 0; w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w.pImageInfo = &depthI;
        vkUpdateDescriptorSets(VulkanHW.m_Device, 1, &w, 0, nullptr);
    }

    // Reconstruction basis from the matrix that RENDERED the depth (Device.mProject
    // is identity on this path — derive from mFullTransform, like the GTAO pass).
    const ProjTerms pt = DeriveProjTerms(Device.mFullTransform);
    const Fvector eye  = Device.vCameraPosition;
    MVPush push{};
    push.camDir[0]    = pt.dir.x;          push.camDir[1]    = pt.dir.y;          push.camDir[2]    = pt.dir.z;          push.camDir[3]    = eye.x;
    push.camRightT[0] = pt.right.x * pt.tanX; push.camRightT[1] = pt.right.y * pt.tanX; push.camRightT[2] = pt.right.z * pt.tanX; push.camRightT[3] = eye.y;
    push.camTopT[0]   = pt.top.x  * pt.tanY;  push.camTopT[1]   = pt.top.y  * pt.tanY;  push.camTopT[2]   = pt.top.z  * pt.tanY;  push.camTopT[3]   = eye.z;
    push.zp[0] = pt.p33;
    push.zp[1] = pt.p43;
    push.zp[2] = (extent.width  > 0) ? 1.0f / float(extent.width)  : 0.0f;
    push.zp[3] = (extent.height > 0) ? 1.0f / float(extent.height) : 0.0f;
    std::memcpy(push.prevVP, &s_prevVP, sizeof(push.prevVP));

    const VkImageLayout oldL = s_first ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    ImageBarrier(cmd, s_img, oldL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    s_first = false;
    Fullscreen::DrawSimple(cmd, s_view, extent, s_pipeMV, s_mvLayout, s_setMV[slot], &push, sizeof(push));
    ImageBarrier(cmd, s_img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void ExecuteDynamic(VkCommandBuffer cmd, const FrameContext& ctx)
{
    // The WHOLE motion pass, run after all opaque geometry (registered after LODs):
    //   1) fullscreen depth reconstruction → camera motion for EVERYTHING now in the
    //      depth (statics + TREES (rigid → exact) + GRASS (camera part) + NPCs) + sky.
    //   2) skinned NPC overlay → replaces the camera-only NPC motion with the true
    //      animated motion (camera + skeletal pose + body travel).
    // Doing (1) here (not in the World pass) is what gives trees/grass correct motion
    // instead of the background's — at World time they aren't in the depth yet.
    if (!Enabled() || s_img == VK_NULL_HANDLE || cmd == VK_NULL_HANDLE || ctx.depthView == VK_NULL_HANDLE) return;
    EnsurePrevVP();

    // 1) Camera/static/tree/grass MV from the complete opaque depth (needs SHADER_READ).
    ImageBarrier(cmd, Swapchain.m_DepthImage, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
    Execute(cmd, ctx.extent);   // fullscreen reconstruction (owns its MV-target barriers)
    ImageBarrier(cmd, Swapchain.m_DepthImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);

    // 2) Skinned NPC animation overlay — depth-test (LEQUAL, no write) against the
    //    same complete depth so only visible NPC pixels overwrite the camera field.
    ImageBarrier(cmd, s_img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    VkRenderingAttachmentInfo cAtt{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    cAtt.imageView   = s_view;
    cAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    cAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;    // keep the static camera field
    cAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingAttachmentInfo dAtt{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    dAtt.imageView   = ctx.depthView;
    dAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;   // complete opaque depth (post-LODs)
    dAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;
    dAtt.storeOp     = VK_ATTACHMENT_STORE_OP_DONT_CARE;           // test only, no write

    VkRenderingInfo ri{ VK_STRUCTURE_TYPE_RENDERING_INFO };
    ri.renderArea.extent    = s_extent;
    ri.layerCount           = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments    = &cAtt;
    ri.pDepthAttachment     = &dAtt;
    vkCmdBeginRendering(cmd, &ri);

    // Negative-height viewport — MUST match the forward/prepass raster so the NPC
    // lands on the same pixels and its depth bit-matches (LEQUAL passes on equal).
    VkViewport vp{ 0.0f, (float)s_extent.height, (float)s_extent.width, -(float)s_extent.height, 0.0f, 1.0f };
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{ {}, s_extent };
    vkCmdSetScissor(cmd, 0, 1, &sc);

    const Fmatrix curVP = ctx.viewProj ? *ctx.viewProj : Device.mFullTransform;
    Skinned_RenderMotion(cmd, curVP, s_prevVP);
    // (Trees / grass dynamics MV → Phase 2b, drawn here too.)

    vkCmdEndRendering(cmd);
    ImageBarrier(cmd, s_img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void DrawDebugOverlay(VkCommandBuffer cmd, VkImageView dstView, VkExtent2D extent)
{
    if (!s_inited || s_failed || ps_r_mv_debug == 0 || s_img == VK_NULL_HANDLE || dstView == VK_NULL_HANDLE) return;

    const u32 slot = CommandManager.GetCurrentFrame() % kFramesInFlight;
    {
        VkDescriptorImageInfo mvI{ s_samp, s_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w.dstSet = s_setDbg[slot]; w.dstBinding = 0; w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w.pImageInfo = &mvI;
        vkUpdateDescriptorSets(VulkanHW.m_Device, 1, &w, 0, nullptr);
    }
    DbgPush dp{};
    dp.p[0] = (ps_r_mv_debug_scale > 0.0f) ? ps_r_mv_debug_scale : 30.0f;
    Fullscreen::DrawSimple(cmd, dstView, extent, s_pipeDbg, s_dbgLayout, s_setDbg[slot], &dp, sizeof(dp));
}

void Destroy()
{
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
    DestroyTarget();
    if (s_pipeMV)    { vkDestroyPipeline(VulkanHW.m_Device, s_pipeMV, nullptr); s_pipeMV = VK_NULL_HANDLE; }
    if (s_pipeDbg)   { vkDestroyPipeline(VulkanHW.m_Device, s_pipeDbg, nullptr); s_pipeDbg = VK_NULL_HANDLE; }
    if (s_mvLayout)  { vkDestroyPipelineLayout(VulkanHW.m_Device, s_mvLayout, nullptr); s_mvLayout = VK_NULL_HANDLE; }
    if (s_dbgLayout) { vkDestroyPipelineLayout(VulkanHW.m_Device, s_dbgLayout, nullptr); s_dbgLayout = VK_NULL_HANDLE; }
    if (s_samp)      { vkDestroySampler(VulkanHW.m_Device, s_samp, nullptr); s_samp = VK_NULL_HANDLE; }
    if (s_pool)      { vkDestroyDescriptorPool(VulkanHW.m_Device, s_pool, nullptr); s_pool = VK_NULL_HANDLE; }
    if (s_setLayout) { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_setLayout, nullptr); s_setLayout = VK_NULL_HANDLE; }
    for (u32 i = 0; i < kFramesInFlight; ++i) { s_setMV[i] = VK_NULL_HANDLE; s_setDbg[i] = VK_NULL_HANDLE; }
    s_inited = false; s_failed = false; s_generation = 0; s_vpValid = false; s_vpFrame = u32(-1);
}

}}  // namespace VK::MotionVec
