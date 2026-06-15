// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).
//
// Minimal Dear ImGui Vulkan backend + profiler overlay. See vk_imgui.h.

#include "stdafx.h"
#include "vk_imgui.h"
#include "vk_profiler.h"            // Prof::GetZones / GetMem / GetFrame
#include "HW_Vulkan.h"             // VulkanHW (device/allocator + single-time cmds)
#include "vk_swapchain.h"          // Swapchain.m_Format
#include "vk_shaders.h"            // g_ShaderManager (.spv loader)
#include "vk_command_buffer.h"     // CommandManager.GetCurrentFrame()

#include "../../../3rd_party/Src/imgui/imgui.h"

extern int ps_r_profiler;

namespace VK { namespace ImGuiVK {

namespace {

constexpr u32 kSlots = VK_FRAMES_IN_FLIGHT;

bool s_inited = false;
bool s_failed = false;

VkPipeline            s_pipeline   = VK_NULL_HANDLE;
VkPipelineLayout      s_layout     = VK_NULL_HANDLE;
VkDescriptorSetLayout s_setLayout  = VK_NULL_HANDLE;
VkDescriptorPool      s_pool       = VK_NULL_HANDLE;
VkDescriptorSet       s_fontSet    = VK_NULL_HANDLE;
VkSampler             s_sampler    = VK_NULL_HANDLE;

VkImage       s_fontImage = VK_NULL_HANDLE;
VmaAllocation s_fontAlloc = VK_NULL_HANDLE;
VkImageView   s_fontView  = VK_NULL_HANDLE;

struct FrameBuf {
    VkBuffer      vb = VK_NULL_HANDLE, ib = VK_NULL_HANDLE;
    VmaAllocation vbA = VK_NULL_HANDLE, ibA = VK_NULL_HANDLE;
    void*         vbMap = nullptr; void* ibMap = nullptr;
    VkDeviceSize  vbSize = 0, ibSize = 0;
};
FrameBuf s_frame[kSlots];

ImGuiContext* s_ctx = nullptr;

struct PushC { float scale[2]; float translate[2]; };

// ---- font atlas upload -----------------------------------------------------
bool CreateFontTexture()
{
    ImGuiIO& io = ImGui::GetIO();
    unsigned char* pixels = nullptr; int w = 0, h = 0;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &w, &h);
    const VkDeviceSize bytes = (VkDeviceSize)w * h * 4;

    // device-local image
    VkImageCreateInfo ici{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_R8G8B8A8_UNORM;
    ici.extent = { (u32)w, (u32)h, 1 };
    ici.mipLevels = 1; ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VmaAllocationCreateInfo iaci{}; iaci.usage = VMA_MEMORY_USAGE_AUTO;
    if (vmaCreateImage(VulkanHW.m_Allocator, &ici, &iaci, &s_fontImage, &s_fontAlloc, nullptr) != VK_SUCCESS) {
        Msg("![VK ImGui] font image create failed"); return false;
    }
    Prof::NameImage(s_fontImage, "ImGui.FontAtlas");

    // staging
    VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size = bytes; bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo baci{};
    baci.usage = VMA_MEMORY_USAGE_AUTO;
    baci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VkBuffer staging = VK_NULL_HANDLE; VmaAllocation stagingA = VK_NULL_HANDLE; VmaAllocationInfo sInfo{};
    if (vmaCreateBuffer(VulkanHW.m_Allocator, &bci, &baci, &staging, &stagingA, &sInfo) != VK_SUCCESS) {
        Msg("![VK ImGui] font staging failed"); return false;
    }
    memcpy(sInfo.pMappedData, pixels, (size_t)bytes);
    vmaFlushAllocation(VulkanHW.m_Allocator, stagingA, 0, bytes);

    // copy + layout transitions on a one-shot command buffer
    VkCommandBuffer cmd = VulkanHW.BeginSingleTimeCommands();
    auto barrier = [&](VkImageLayout oldL, VkImageLayout newL,
                       VkAccessFlags srcA, VkAccessFlags dstA,
                       VkPipelineStageFlags srcS, VkPipelineStageFlags dstS) {
        VkImageMemoryBarrier b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        b.oldLayout = oldL; b.newLayout = newL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = s_fontImage;
        b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        b.srcAccessMask = srcA; b.dstAccessMask = dstA;
        vkCmdPipelineBarrier(cmd, srcS, dstS, 0, 0, nullptr, 0, nullptr, 1, &b);
    };
    barrier(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            0, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
    VkBufferImageCopy region{};
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent = { (u32)w, (u32)h, 1 };
    vkCmdCopyBufferToImage(cmd, staging, s_fontImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    barrier(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
    VulkanHW.EndSingleTimeCommands(cmd);
    vmaDestroyBuffer(VulkanHW.m_Allocator, staging, stagingA);

    VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vci.image = s_fontImage; vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = VK_FORMAT_R8G8B8A8_UNORM;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    if (vkCreateImageView(VulkanHW.m_Device, &vci, nullptr, &s_fontView) != VK_SUCCESS) {
        Msg("![VK ImGui] font view failed"); return false;
    }
    io.Fonts->SetTexID((ImTextureID)(intptr_t)s_fontImage);   // unused (single texture)
    return true;
}

bool CreatePipeline()
{
    if (!g_ShaderManager) g_ShaderManager = xr_new<VK::CVulkanSPIRVLoader>();
    VkShaderModule vs = g_ShaderManager->Load("imgui.vert.spv");
    VkShaderModule fs = g_ShaderManager->Load("imgui.frag.spv");
    if (!vs || !fs) { Msg("![VK ImGui] shader load failed"); return false; }

    // set 0: combined image sampler (font), fragment
    VkDescriptorSetLayoutBinding b{};
    b.binding = 0; b.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b.descriptorCount = 1; b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = 1; lci.pBindings = &b;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &s_setLayout) != VK_SUCCESS) return false;

    VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 };
    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets = 1; pci.poolSizeCount = 1; pci.pPoolSizes = &ps;
    if (vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &s_pool) != VK_SUCCESS) return false;
    VkDescriptorSetAllocateInfo dai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dai.descriptorPool = s_pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &s_setLayout;
    if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, &s_fontSet) != VK_SUCCESS) return false;

    VkSamplerCreateInfo si{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.minLod = -1000.f; si.maxLod = 1000.f; si.maxAnisotropy = 1.0f;
    if (vkCreateSampler(VulkanHW.m_Device, &si, nullptr, &s_sampler) != VK_SUCCESS) return false;

    VkDescriptorImageInfo ii{ s_sampler, s_fontView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    w.dstSet = s_fontSet; w.dstBinding = 0; w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w.pImageInfo = &ii;
    vkUpdateDescriptorSets(VulkanHW.m_Device, 1, &w, 0, nullptr);

    VkPushConstantRange pcr{ VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushC) };
    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1; plci.pSetLayouts = &s_setLayout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_layout) != VK_SUCCESS) return false;

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT; stages[0].module = vs; stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = fs; stages[1].pName = "main";

    VkVertexInputBindingDescription bind{ 0, sizeof(ImDrawVert), VK_VERTEX_INPUT_RATE_VERTEX };
    VkVertexInputAttributeDescription attr[3]{
        { 0, 0, VK_FORMAT_R32G32_SFLOAT,  (u32)IM_OFFSETOF(ImDrawVert, pos) },
        { 1, 0, VK_FORMAT_R32G32_SFLOAT,  (u32)IM_OFFSETOF(ImDrawVert, uv)  },
        { 2, 0, VK_FORMAT_R8G8B8A8_UNORM, (u32)IM_OFFSETOF(ImDrawVert, col) },
    };
    VkPipelineVertexInputStateCreateInfo vi{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &bind;
    vi.vertexAttributeDescriptionCount = 3; vi.pVertexAttributeDescriptions = attr;

    VkPipelineInputAssemblyStateCreateInfo ia{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    vp.viewportCount = 1; vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState cb{};
    cb.blendEnable = VK_TRUE;
    cb.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    cb.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cb.colorBlendOp = VK_BLEND_OP_ADD;
    cb.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    cb.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cb.alphaBlendOp = VK_BLEND_OP_ADD;
    cb.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo bs{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    bs.attachmentCount = 1; bs.pAttachments = &cb;

    VkPipelineDepthStencilStateCreateInfo ds{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    ds.depthTestEnable = VK_FALSE; ds.depthWriteEnable = VK_FALSE;

    VkDynamicState dyn[2]{ VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynci{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dynci.dynamicStateCount = 2; dynci.pDynamicStates = dyn;

    VkFormat colFmt = Swapchain.m_Format;
    VkPipelineRenderingCreateInfo prci{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    prci.colorAttachmentCount = 1; prci.pColorAttachmentFormats = &colFmt;

    VkGraphicsPipelineCreateInfo gp{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    gp.pNext = &prci;
    gp.stageCount = 2; gp.pStages = stages;
    gp.pVertexInputState = &vi; gp.pInputAssemblyState = &ia;
    gp.pViewportState = &vp; gp.pRasterizationState = &rs;
    gp.pMultisampleState = &ms; gp.pColorBlendState = &bs;
    gp.pDepthStencilState = &ds; gp.pDynamicState = &dynci;
    gp.layout = s_layout;
    if (vkCreateGraphicsPipelines(VulkanHW.m_Device, VK_NULL_HANDLE, 1, &gp, nullptr, &s_pipeline) != VK_SUCCESS) {
        Msg("![VK ImGui] pipeline create failed"); return false;
    }
    return true;
}

bool Init()
{
    if (s_inited) return true;
    if (s_failed) return false;
    if (VulkanHW.m_Device == VK_NULL_HANDLE || Swapchain.m_Format == VK_FORMAT_UNDEFINED) return false;

    IMGUI_CHECKVERSION();
    s_ctx = ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;            // no imgui.ini on disk
    io.LogFilename = nullptr;
    io.BackendRendererName = "vk_imgui";
    ImGui::StyleColorsDark();
    ImGui::GetStyle().WindowRounding = 4.0f;
    ImGui::GetStyle().Alpha = 0.92f;

    if (!CreateFontTexture() || !CreatePipeline()) {
        Msg("![VK ImGui] init FAILED — overlay disabled");
        s_failed = true;
        return false;
    }
    s_inited = true;
    Msg("[VK ImGui] overlay ready");
    return true;
}

// grow/create this slot's vertex+index buffers to fit
void EnsureBuffers(FrameBuf& f, VkDeviceSize vbNeed, VkDeviceSize ibNeed)
{
    auto make = [&](VkBuffer& buf, VmaAllocation& alloc, void*& map, VkDeviceSize& cap,
                    VkDeviceSize need, VkBufferUsageFlags usage) {
        if (cap >= need && buf) return;
        if (buf) vmaDestroyBuffer(VulkanHW.m_Allocator, buf, alloc);
        VkDeviceSize sz = need + (need / 2) + 4096;
        VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bci.size = sz; bci.usage = usage;
        VmaAllocationCreateInfo aci{};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo info{};
        if (vmaCreateBuffer(VulkanHW.m_Allocator, &bci, &aci, &buf, &alloc, &info) != VK_SUCCESS) {
            buf = VK_NULL_HANDLE; cap = 0; map = nullptr; return;
        }
        map = info.pMappedData; cap = sz;
    };
    make(f.vb, f.vbA, f.vbMap, f.vbSize, vbNeed, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    make(f.ib, f.ibA, f.ibMap, f.ibSize, ibNeed, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
}

// ---- the profiler window ---------------------------------------------------
void BuildProfilerWindow()
{
    const Prof::ZoneStat* zones = nullptr;
    const u32 nz = Prof::GetZones(&zones);
    Prof::MemSnap   mem = Prof::GetMem();
    Prof::FrameInfo fr  = Prof::GetFrame();

    ImGui::SetNextWindowPos(ImVec2(12, 12), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(420, 0), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("VK Profiler", nullptr, ImGuiWindowFlags_NoNav)) { ImGui::End(); return; }

    ImGui::Text("FPS %.0f   CPU %.2f ms   GPU %.2f ms", fr.fps, fr.cpuMs, fr.gpuMs);

    // frame-time history graph (kept locally; the zone rings are GPU-only)
    static float s_frameHist[120] = {};
    static int   s_fhHead = 0;
    s_frameHist[s_fhHead] = fr.cpuMs;
    s_fhHead = (s_fhHead + 1) % IM_ARRAYSIZE(s_frameHist);
    ImGui::PlotLines("##frame", s_frameHist, IM_ARRAYSIZE(s_frameHist), s_fhHead,
                     "CPU frame ms", 0.0f, 33.3f, ImVec2(-1, 48));

    ImGui::Separator();
    ImGui::TextDisabled("Pass GPU ms (last / min / max)");

    // scale bars to the slowest pass so the breakdown reads at a glance
    float maxMs = 0.1f;
    for (u32 z = 0; z < nz; ++z) if (zones[z].gpuMax > maxMs) maxMs = zones[z].gpuMax;

    for (u32 z = 0; z < nz; ++z) {
        const Prof::ZoneStat& s = zones[z];
        char ov[48];
        _snprintf(ov, sizeof(ov), "%.2f  (%.2f/%.2f)", s.gpuLast, s.gpuMin, s.gpuMax);
        ImGui::ProgressBar(s.gpuLast / maxMs, ImVec2(-150, 0), ov);
        ImGui::SameLine(); ImGui::Text("%s", s.name);
    }

    ImGui::Separator();
    const double usedMB = double(mem.usedBytes) / (1024.0 * 1024.0);
    const double budMB  = double(mem.budgetBytes) / (1024.0 * 1024.0);
    char vov[48];
    _snprintf(vov, sizeof(vov), "%.0f / %.0f MB", usedMB, budMB);
    ImGui::ProgressBar(budMB > 0 ? float(usedMB / budMB) : 0.f, ImVec2(-1, 0), vov);
    ImGui::Text("VRAM allocs %u in %u blocks", mem.allocCount, mem.blockCount);
    ImGui::TextDisabled("draws %u  inst %u  tris %u  pipe %u  desc %u",
                        fr.draws, fr.instances, fr.tris, fr.pipeBinds, fr.descBinds);
    ImGui::End();
}

} // anonymous namespace

void DrawOverlay(VkCommandBuffer cmd, VkImageView swapchainView, VkExtent2D extent, float dt)
{
    if (ps_r_profiler < 2) return;
    if (!Init()) return;
    if (extent.width == 0 || extent.height == 0) return;

    ImGui::SetCurrentContext(s_ctx);
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2((float)extent.width, (float)extent.height);
    io.DeltaTime   = dt > 1e-5f ? dt : 1.0f / 60.0f;

    ImGui::NewFrame();
    BuildProfilerWindow();
    ImGui::Render();
    ImDrawData* dd = ImGui::GetDrawData();
    if (!dd || dd->TotalVtxCount == 0) return;

    const u32 slot = CommandManager.GetCurrentFrame() % kSlots;
    FrameBuf& f = s_frame[slot];
    const VkDeviceSize vbNeed = (VkDeviceSize)dd->TotalVtxCount * sizeof(ImDrawVert);
    const VkDeviceSize ibNeed = (VkDeviceSize)dd->TotalIdxCount * sizeof(ImDrawIdx);
    EnsureBuffers(f, vbNeed, ibNeed);
    if (!f.vb || !f.ib) return;

    // upload all command lists into the slot's buffers
    {
        ImDrawVert* vtx = (ImDrawVert*)f.vbMap;
        ImDrawIdx*  idx = (ImDrawIdx*)f.ibMap;
        for (int n = 0; n < dd->CmdListsCount; ++n) {
            const ImDrawList* cl = dd->CmdLists[n];
            memcpy(vtx, cl->VtxBuffer.Data, cl->VtxBuffer.Size * sizeof(ImDrawVert));
            memcpy(idx, cl->IdxBuffer.Data, cl->IdxBuffer.Size * sizeof(ImDrawIdx));
            vtx += cl->VtxBuffer.Size;
            idx += cl->IdxBuffer.Size;
        }
        vmaFlushAllocation(VulkanHW.m_Allocator, f.vbA, 0, vbNeed);
        vmaFlushAllocation(VulkanHW.m_Allocator, f.ibA, 0, ibNeed);
    }

    Prof::CmdBeginLabel(cmd, "ImGui Overlay", 0.9f, 0.7f, 0.2f);

    VkRenderingAttachmentInfo cAtt{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    cAtt.imageView = swapchainView;
    cAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    cAtt.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;     // draw on top of the final frame
    cAtt.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingInfo ri{ VK_STRUCTURE_TYPE_RENDERING_INFO };
    ri.renderArea.extent = extent; ri.layerCount = 1;
    ri.colorAttachmentCount = 1; ri.pColorAttachments = &cAtt;
    vkCmdBeginRendering(cmd, &ri);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_layout, 0, 1, &s_fontSet, 0, nullptr);

    VkViewport vp{ 0.f, 0.f, (float)extent.width, (float)extent.height, 0.f, 1.f };
    vkCmdSetViewport(cmd, 0, 1, &vp);

    PushC pc;
    pc.scale[0] = 2.0f / dd->DisplaySize.x;
    pc.scale[1] = 2.0f / dd->DisplaySize.y;
    pc.translate[0] = -1.0f - dd->DisplayPos.x * pc.scale[0];
    pc.translate[1] = -1.0f - dd->DisplayPos.y * pc.scale[1];
    vkCmdPushConstants(cmd, s_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushC), &pc);

    VkDeviceSize voff = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &f.vb, &voff);
    vkCmdBindIndexBuffer(cmd, f.ib, 0, sizeof(ImDrawIdx) == 2 ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);

    const ImVec2 clipOff = dd->DisplayPos;
    int globalVtx = 0, globalIdx = 0;
    for (int n = 0; n < dd->CmdListsCount; ++n) {
        const ImDrawList* cl = dd->CmdLists[n];
        for (int c = 0; c < cl->CmdBuffer.Size; ++c) {
            const ImDrawCmd* pcmd = &cl->CmdBuffer[c];
            float x0 = pcmd->ClipRect.x - clipOff.x, y0 = pcmd->ClipRect.y - clipOff.y;
            float x1 = pcmd->ClipRect.z - clipOff.x, y1 = pcmd->ClipRect.w - clipOff.y;
            if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0;
            if (x1 > extent.width)  x1 = (float)extent.width;
            if (y1 > extent.height) y1 = (float)extent.height;
            if (x1 <= x0 || y1 <= y0) continue;
            VkRect2D sc{ { (int32_t)x0, (int32_t)y0 }, { (u32)(x1 - x0), (u32)(y1 - y0) } };
            vkCmdSetScissor(cmd, 0, 1, &sc);
            vkCmdDrawIndexed(cmd, pcmd->ElemCount, 1,
                             pcmd->IdxOffset + globalIdx, pcmd->VtxOffset + globalVtx, 0);
        }
        globalVtx += cl->VtxBuffer.Size;
        globalIdx += cl->IdxBuffer.Size;
    }

    vkCmdEndRendering(cmd);
    Prof::CmdEndLabel(cmd);
}

void Shutdown()
{
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
    for (FrameBuf& f : s_frame) {
        if (f.vb) vmaDestroyBuffer(VulkanHW.m_Allocator, f.vb, f.vbA);
        if (f.ib) vmaDestroyBuffer(VulkanHW.m_Allocator, f.ib, f.ibA);
        f = FrameBuf{};
    }
    if (s_pipeline)  { vkDestroyPipeline(VulkanHW.m_Device, s_pipeline, nullptr); s_pipeline = VK_NULL_HANDLE; }
    if (s_layout)    { vkDestroyPipelineLayout(VulkanHW.m_Device, s_layout, nullptr); s_layout = VK_NULL_HANDLE; }
    if (s_pool)      { vkDestroyDescriptorPool(VulkanHW.m_Device, s_pool, nullptr); s_pool = VK_NULL_HANDLE; }
    if (s_setLayout) { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_setLayout, nullptr); s_setLayout = VK_NULL_HANDLE; }
    if (s_sampler)   { vkDestroySampler(VulkanHW.m_Device, s_sampler, nullptr); s_sampler = VK_NULL_HANDLE; }
    if (s_fontView)  { vkDestroyImageView(VulkanHW.m_Device, s_fontView, nullptr); s_fontView = VK_NULL_HANDLE; }
    if (s_fontImage) { vmaDestroyImage(VulkanHW.m_Allocator, s_fontImage, s_fontAlloc); s_fontImage = VK_NULL_HANDLE; }
    if (s_ctx)       { ImGui::DestroyContext(s_ctx); s_ctx = nullptr; }
    s_inited = false;
}

}} // namespace VK::ImGuiVK
