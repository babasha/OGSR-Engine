// xrRenderVulkan - Skinned-mesh pass (STEP B sub-step 2). See vk_pass_skinned.h.
#include "stdafx.h"

// CKinematics (children + LL_GetTransform_R/LL_BoneCount) — same compat trick as
// the skeleton wrapper TUs. Keep dxRender_Visual mapped (children is xr_vector<dxRender_Visual*>).
#define FBasicVisualH
#include "vk_FBasicVisual.h"            // pulls vk_SkeletonCompat.h (dxRender_Visual->vkRender_Visual, vk_Visual.h)
#include "CRender_Vulkan.h"
#include "../xrRender/SkeletonCustom.h" // CKinematics

#include "vk_pass_skinned.h"
#include "vk_pass_world.h"              // g_DynamicVisuals / DynVisual
#include "vk_swapchain.h"              // Swapchain.m_Format / m_DepthFormat
#include "vk_shaders.h"               // g_ShaderManager
#include "vk_buffer.h"                // CVulkanBuffer
#include "vk_world_material.h"        // WorldMaterial / WorldMaterialCache (set 1 = diffuse)
#include "HW_Vulkan.h"                // VulkanHW.m_Device
#include "vk_command_buffer.h"        // CommandManager.GetCurrentFrame() — in-flight slot
#include "vk_pipeline_cache.h"        // PipelineCache::GetCacheObject() — shared disk-backed cache
#include "../../xr_3da/device.h"      // Device.mFullTransform_hud (HUD projection)

#include <unordered_map>

namespace VK {

namespace {
    bool                  s_inited     = false;
    bool                  s_failed     = false;
    VkPipelineLayout      s_layout     = VK_NULL_HANDLE;
    VkDescriptorSetLayout s_setLayout  = VK_NULL_HANDLE;
    VkDescriptorPool      s_pool       = VK_NULL_HANDLE;
    VkDescriptorSet       s_set        = VK_NULL_HANDLE;
    VkShaderModule        s_vs         = VK_NULL_HANDLE;
    VkShaderModule        s_fs         = VK_NULL_HANDLE;
    CVulkanBuffer         s_boneSSBO;
    Fmatrix*              s_boneMapped = nullptr;
    std::unordered_map<u32, VkPipeline> s_pipelines;   // keyed by vertex stride (36/40/44)

    constexpr u32 kMaxBones       = 16384;   // bone-matrix slots PER frame-in-flight (~1 MB each)
    constexpr u32 kFramesInFlight = CVulkanCommandManager::FRAMES_IN_FLIGHT;

    // Push block — must match skinned.vert.glsl PushConstants (VERTEX stage).
    struct SkinPush { Fmatrix mvp; u32 skinMode; u32 baseBone; u32 boneCount; };

    // Vertex input for the vertHW_* layouts. loc0 pos FLOAT4; loc1/3/4 packed
    // normal/tangent/binormal (UBYTE4N); loc2 = FLOAT2 (36/40) or FLOAT4 (44);
    // loc5 = 4 bone indices (40 only, dummy @0 elsewhere — unused by those modes).
    static void BuildSkinnedVI(u32 stride, VkVertexInputBindingDescription& b,
                               VkVertexInputAttributeDescription attrs[6])
    {
        b = { 0, stride, VK_VERTEX_INPUT_RATE_VERTEX };
        attrs[0] = { 0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0  };
        attrs[1] = { 1, 0, VK_FORMAT_R8G8B8A8_UNORM,      16 };
        attrs[3] = { 3, 0, VK_FORMAT_R8G8B8A8_UNORM,      20 };
        attrs[4] = { 4, 0, VK_FORMAT_R8G8B8A8_UNORM,      24 };
        if (stride == 44) {
            attrs[2] = { 2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 28 };  // tc + 2 bone idx
            attrs[5] = { 5, 0, VK_FORMAT_R8G8B8A8_UNORM,       0 };  // unused (2W/3W)
        } else if (stride == 40) {
            attrs[2] = { 2, 0, VK_FORMAT_R32G32_SFLOAT,  28 };       // tc only
            attrs[5] = { 5, 0, VK_FORMAT_R8G8B8A8_UNORM, 36 };       // 4 bone idx (4W)
        } else { // 36 (1W)
            attrs[2] = { 2, 0, VK_FORMAT_R32G32_SFLOAT,  28 };       // tc only
            attrs[5] = { 5, 0, VK_FORMAT_R8G8B8A8_UNORM, 0  };       // unused (1W)
        }
    }

    static VkPipeline CreatePipeline(u32 stride)
    {
        VkVertexInputBindingDescription   binding{};
        VkVertexInputAttributeDescription attrs[6]{};
        BuildSkinnedVI(stride, binding, attrs);

        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount   = 1;
        vi.pVertexBindingDescriptions      = &binding;
        vi.vertexAttributeDescriptionCount = 6;
        vi.pVertexAttributeDescriptions    = attrs;

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;   stages[0].module = s_vs; stages[0].pName = "main";
        stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = s_fs; stages[1].pName = "main";

        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vp{};
        vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vp.viewportCount = 1; vp.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode    = VK_CULL_MODE_NONE;
        rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth   = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{};
        ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds.depthTestEnable  = VK_TRUE;
        ds.depthWriteEnable = VK_TRUE;
        ds.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineColorBlendAttachmentState ba{};
        ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        ba.blendEnable    = VK_FALSE;
        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1; cb.pAttachments = &ba;

        VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynState{};
        dynState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynState.dynamicStateCount = 2; dynState.pDynamicStates = dyn;

        VkFormat colorFormat = Swapchain.m_Format;
        VkPipelineRenderingCreateInfo prci{};
        prci.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
        prci.colorAttachmentCount    = 1;
        prci.pColorAttachmentFormats = &colorFormat;
        prci.depthAttachmentFormat   = Swapchain.m_DepthFormat;

        VkGraphicsPipelineCreateInfo pi{};
        pi.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pi.pNext               = &prci;
        pi.stageCount          = 2;          pi.pStages = stages;
        pi.pVertexInputState   = &vi;        pi.pInputAssemblyState = &ia;
        pi.pViewportState      = &vp;        pi.pRasterizationState = &rs;
        pi.pMultisampleState   = &ms;        pi.pDepthStencilState  = &ds;
        pi.pColorBlendState    = &cb;        pi.pDynamicState       = &dynState;
        pi.layout              = s_layout;

        VkPipeline h = VK_NULL_HANDLE;
        VkResult r = vkCreateGraphicsPipelines(VulkanHW.m_Device, PipelineCache::GetCacheObject(), 1, &pi, nullptr, &h);
        if (r != VK_SUCCESS) { Msg("![VK Skinned] pipeline create failed (%d) stride=%u", r, stride); return VK_NULL_HANDLE; }
        Msg("[VK Skinned] pipeline created stride=%u", stride);
        return h;
    }

    static VkPipeline GetPipeline(u32 stride)
    {
        auto it = s_pipelines.find(stride);
        if (it != s_pipelines.end()) return it->second;
        VkPipeline p = CreatePipeline(stride);
        s_pipelines.emplace(stride, p);
        return p;
    }

    static bool Init()
    {
        if (s_inited) return !s_failed;
        s_inited = true;

        if (!g_ShaderManager) g_ShaderManager = xr_new<VK::CVulkanSPIRVLoader>();
        s_vs = g_ShaderManager->Load("skinned.vert.spv");
        s_fs = g_ShaderManager->Load("skinned.frag.spv");
        if (s_vs == VK_NULL_HANDLE || s_fs == VK_NULL_HANDLE) {
            Msg("![VK Skinned] failed to load skinned.vert/frag.spv");
            s_failed = true; return false;
        }

        // Descriptor set layout: 1 storage buffer (bone matrices), VERTEX stage.
        VkDescriptorSetLayoutBinding b{};
        b.binding = 0; b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b.descriptorCount = 1; b.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        VkDescriptorSetLayoutCreateInfo slci{};
        slci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        slci.bindingCount = 1; slci.pBindings = &b;
        if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &slci, nullptr, &s_setLayout) != VK_SUCCESS) { s_failed = true; return false; }

        VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
        VkDescriptorPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets = 1; pci.poolSizeCount = 1; pci.pPoolSizes = &ps;
        if (vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &s_pool) != VK_SUCCESS) { s_failed = true; return false; }

        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = s_pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &s_setLayout;
        if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, &s_set) != VK_SUCCESS) { s_failed = true; return false; }

        // Pipeline layout: push {mvp, skinMode, baseBone} (VERTEX) + set0=bone SSBO +
        // set1=material (base/detail/lmap) so the FS can sample the diffuse texture.
        VkDescriptorSetLayout matLayout = WorldMaterialCache::GetSetLayout();
        if (matLayout == VK_NULL_HANDLE) { Msg("![VK Skinned] material set layout not ready"); s_failed = true; return false; }
        VkDescriptorSetLayout setLayouts[2] = { s_setLayout, matLayout };
        VkPushConstantRange pcr{ VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(SkinPush) };
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 2; plci.pSetLayouts = setLayouts;
        plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
        if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_layout) != VK_SUCCESS) { s_failed = true; return false; }

        // Bone SSBO (host-visible, persistently mapped — Create sets HOST_ACCESS+MAPPED for STORAGE).
        // Sized FRAMES_IN_FLIGHT × kMaxBones: each in-flight frame owns region
        // [slot*kMaxBones, (slot+1)*kMaxBones). Frame S only writes its region after
        // WaitForFence(S) in CRender::Begin proved the GPU finished the previous frame-S
        // submission that read it, so the regions still in flight (S+1/S+2) are never
        // clobbered. baseBone push-constant absolutely-indexes the whole buffer, so the
        // single VK_WHOLE_SIZE descriptor below needs no per-frame rebind.
        s_boneSSBO.Create(VkDeviceSize(kMaxBones) * kFramesInFlight * sizeof(Fmatrix),
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
        s_boneMapped = static_cast<Fmatrix*>(s_boneSSBO.Map());
        if (!s_boneMapped) { Msg("![VK Skinned] bone SSBO map failed"); s_failed = true; return false; }

        VkDescriptorBufferInfo bi{ s_boneSSBO.GetHandle(), 0, VK_WHOLE_SIZE };
        VkWriteDescriptorSet w{};
        w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w.dstSet = s_set; w.dstBinding = 0; w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w.pBufferInfo = &bi;
        vkUpdateDescriptorSets(VulkanHW.m_Device, 1, &w, 0, nullptr);

        Msg("[VK Skinned] init OK (SSBO %u bones, push=%u)", kMaxBones, (u32)sizeof(SkinPush));
        return true;
    }

    // Draw one list of skinned CKinematics with a given model->clip viewProj.
    // Shared bone-SSBO cursor (caller threads it across lists so write regions
    // don't collide); lastPipe/lastMatSet persist (viewport changes between lists
    // don't disturb pipeline/descriptor bindings — both are dynamic / separate).
    static void DrawSkinnedList(VkCommandBuffer cmd, const xr_vector<DynVisual>& list,
                                const Fmatrix& viewProj, u32& cursor, u32 boneLimit, u32& nDraw,
                                VkPipeline& lastPipe, VkDescriptorSet& lastMatSet)
    {
        for (const DynVisual& d : list)
        {
            if (!d.vis) continue;
            CKinematics* K = dynamic_cast<CKinematics*>(d.vis);  // CKinematics : FHierrarhyVisual : vkRender_Visual
            if (!K || K->children.empty()) continue;

            // bForceExact=TRUE: recompute bones every frame (matches R4's render
            // path, r__dsgraph_build.cpp pV->CalculateBones(TRUE)). With FALSE the
            // shared CKinematics::CalculateBones takes its "slow update" early-out
            // (Device.dwTimeGlobal < UCalc_Time + UCalc_Interval) and leaves bones
            // STALE for UCalc_Interval ms → visibly jerky animation. The
            // dwTimeGlobal==UCalc_Time guard inside still prevents double-advance.
            K->CalculateBones(TRUE);
            const u16 bc = K->LL_BoneCount();
            if (bc == 0) continue;
            if (cursor + bc > boneLimit) break;   // this frame's SSBO region full

            const u32 baseBone = cursor;
            for (u16 i = 0; i < bc; ++i)
                s_boneMapped[cursor + i] = K->LL_GetTransform_R(i);
            cursor += bc;

            Fmatrix mvp;
            // Fmatrix::mul(A,B) == B·A (standard). The GLSL `mvp*pos` needs the uploaded
            // matrix to be world·viewProj (row-major), so A=viewProj, B=xform.
            mvp.mul(viewProj, d.xform);   // = d.xform · viewProj (model->clip)

            for (auto* child : K->children)
            {
                VK_Render_Mesh* mesh = nullptr;
                u16 rmode = 0;
                if (auto* st = dynamic_cast<vkSkeletonX_ST*>(child)) { mesh = &st->m_mesh; rmode = st->RenderMode; }
                else if (auto* pm = dynamic_cast<vkSkeletonX_PM*>(child)) { mesh = &pm->m_mesh; rmode = pm->RenderMode; }
                else continue;

                if (!mesh->p_rm_Vertices || !mesh->p_rm_Indices || mesh->iCount == 0) continue;

                VkPipeline pipe = GetPipeline(mesh->vStride);
                if (pipe == VK_NULL_HANDLE) continue;

                if (pipe != lastPipe) {
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_layout, 0, 1, &s_set, 0, nullptr);
                    lastPipe = pipe;
                    lastMatSet = VK_NULL_HANDLE;   // pipeline change disturbs set bindings — rebind material
                }

                // Diffuse texture (set 1) from the leaf's WorldMaterial; fall back to default white.
                WorldMaterial* mat = child->m_pWorldMaterial ? child->m_pWorldMaterial : WorldMaterialCache::GetDefault();
                VkDescriptorSet matSet = (mat && mat->set != VK_NULL_HANDLE) ? mat->set : VK_NULL_HANDLE;
                if (matSet == VK_NULL_HANDLE) continue;   // no texture descriptor → can't sample, skip
                if (matSet != lastMatSet) {
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_layout, 1, 1, &matSet, 0, nullptr);
                    lastMatSet = matSet;
                }

                // RenderMode enum -> skinMode: RM_SINGLE(1)/RM_SKINNING_1B(2)->1, 2B(3)->2, 3B(4)->3, 4B(5)->4
                const u32 skinMode = (rmode <= 2u) ? 1u : (u32(rmode) - 1u);
                SkinPush pc{ mvp, skinMode, baseBone, (u32)bc };
                vkCmdPushConstants(cmd, s_layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);

                VkBuffer vb = mesh->p_rm_Vertices->GetHandle();
                VkDeviceSize vbOff = 0;
                vkCmdBindVertexBuffers(cmd, 0, 1, &vb, &vbOff);
                vkCmdBindIndexBuffer(cmd, mesh->p_rm_Indices->GetHandle(), 0, mesh->iType);
                vkCmdDrawIndexed(cmd, mesh->iCount, 1, mesh->iBase, (s32)mesh->vBase, 0);
                ++nDraw;
            }
        }
    }

    // HUD near-depth viewport (matches R4 rmNear = depth range [0, 0.02]): the HUD
    // weapon/hands occupy the front 2% of depth so they always win the depth test
    // against the world without clipping into it. Keeps the negative-height Y-flip.
    static void SetViewportDepth(VkCommandBuffer cmd, const VkExtent2D& ext, float minD, float maxD)
    {
        VkViewport vp{ 0.0f, (float)ext.height, (float)ext.width, -(float)ext.height, minD, maxD };
        vkCmdSetViewport(cmd, 0, 1, &vp);
    }
}  // anon namespace

void Pass_Skinned(FrameContext& ctx)
{
    if (g_DynamicVisuals.empty() && g_HudVisuals.empty()) return;
    if (ctx.cmd == VK_NULL_HANDLE || !ctx.viewProj) return;
    if (!Init()) return;

    VkCommandBuffer cmd = ctx.cmd;
    VkPipeline      lastPipe   = VK_NULL_HANDLE;
    VkDescriptorSet lastMatSet = VK_NULL_HANDLE;

    // Pick THIS frame's bone-SSBO region by the fence-guarded in-flight slot, so the
    // CPU never writes matrices a still-in-flight frame's GPU may be reading.
    const u32 slot      = CommandManager.GetCurrentFrame();   // 0..FRAMES_IN_FLIGHT-1, stable across the frame
    const u32 boneBase  = slot * kMaxBones;
    const u32 boneLimit = boneBase + kMaxBones;
    u32 cursor = boneBase;   // next free bone slot in this frame's region (shared across world + HUD)
    u32 nDraw = 0, nHud = 0;

    // World dynamics: world viewProj, normal depth range (viewport already set by Pass_World).
    DrawSkinnedList(cmd, g_DynamicVisuals, *ctx.viewProj, cursor, boneLimit, nDraw, lastPipe, lastMatSet);

    // First-person HUD: HUD-FOV projection (camera at origin) + near depth range so
    // hands/weapon render on top of the world. Restore the normal range afterwards.
    if (!g_HudVisuals.empty()) {
        SetViewportDepth(cmd, ctx.extent, 0.0f, 0.02f);
        DrawSkinnedList(cmd, g_HudVisuals, Device.mFullTransform_hud, cursor, boneLimit, nHud, lastPipe, lastMatSet);
        SetViewportDepth(cmd, ctx.extent, 0.0f, 1.0f);
    }

    static bool s_diag = false;
    if (!s_diag) { s_diag = true; Msg("[VK Skinned] first Pass_Skinned: dynVis=%zu hud=%zu draws=%u hudDraws=%u bonesUsed=%u/%u (slot=%u)",
                                      g_DynamicVisuals.size(), g_HudVisuals.size(), nDraw, nHud, cursor - boneBase, kMaxBones, slot); }
}

void Skinned_Destroy()
{
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
    for (auto& kv : s_pipelines) if (kv.second) vkDestroyPipeline(VulkanHW.m_Device, kv.second, nullptr);
    s_pipelines.clear();
    if (s_layout)    { vkDestroyPipelineLayout(VulkanHW.m_Device, s_layout, nullptr); s_layout = VK_NULL_HANDLE; }
    if (s_pool)      { vkDestroyDescriptorPool(VulkanHW.m_Device, s_pool, nullptr); s_pool = VK_NULL_HANDLE; }
    if (s_setLayout) { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_setLayout, nullptr); s_setLayout = VK_NULL_HANDLE; }
    s_boneSSBO.Destroy();
    s_boneMapped = nullptr;
    s_set = VK_NULL_HANDLE;
    s_inited = false; s_failed = false;
}

}  // namespace VK
