// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan — CDetailManager Session B render path.
//
// Per-frame flow:
//   1. PrepareFrame() — fill m_GfxConstants + m_Frame from camera/wind/density.
//   2. Clear VisibleSSBO + AtomicCounters; pre-fill IndirectCmdBuf with the
//      static {indexCount, firstIndex, vertexOffset, firstInstance} per type.
//   3. Barrier transfer→compute.
//   4. Bind gen compute pipeline; push 208 B; dispatch (groupCount, 1, 1).
//   5. Barrier compute→transfer; per-type vkCmdCopyBuffer of atomic count
//      into the corresponding indirect cmd's instanceCount field.
//   6. Barrier transfer→{vertex_input | indirect_draw_read}.
//   7. Begin dynamic-rendering pass (color = swapchain image LOAD,
//      depth = swapchain depth LOAD).
//   8. Per type: bind descriptor (diffuse), bind VBs (binding 0 = obj VB,
//      binding 1 = VisibleSSBO at section offset), bind IB,
//      vkCmdDrawIndexedIndirect(indirect, i*20, 1, 20).
//   9. End rendering; transition color back to TRANSFER_DST so Pass_Sky and
//      UI can still read the layout we left behind.

#include "stdafx.h"
#include "vk_DetailManager.h"
#include "vk_pass_context.h"
#include "vk_swapchain.h"
#include "vk_env_light.h"               // VK::EnvLight::GetCurrentSet — set 1 (shadow lookup)
#include "HW_Vulkan.h"

#include "../../xr_3da/IGame_Persistent.h"
#include "../../xr_3da/IGame_Level.h"           // g_pGameLevel, CurrentEntity, Objects
#include "../../xr_3da/xr_object.h"             // CObject Position/Radius/getVisible
#include "../../xr_3da/xr_object_list.h"        // o_count / o_get_by_iterator
#include "../../xr_3da/Environment.h"

// Console-tunable knobs from xrRender shared console. Owned by
// vk_console_min.cpp; we just read them per frame.
extern int   ps_r__detail_radius;       // metres, default 100, range 70..300
extern float ps_current_detail_density; // 0..1, default 0.6 (lower = denser)
extern float ps_current_detail_scale;   // r__detail_scale, 0.7..1.5 — per-item size multiplier
extern float ps_r_sun_boost;            // r_sun_boost — global sun multiplier (see vk_env_light)

namespace VK
{

// ----- Frustum plane extraction (D3D-style row-major mat4) ------------------
void CDetailManager::ExtractFrustumPlanes(const Fmatrix& m, Fvector4 planes[6]) const
{
    planes[0].set(m._14 + m._11, m._24 + m._21, m._34 + m._31, m._44 + m._41); // left
    planes[1].set(m._14 - m._11, m._24 - m._21, m._34 - m._31, m._44 - m._41); // right
    planes[2].set(m._14 + m._12, m._24 + m._22, m._34 + m._32, m._44 + m._42); // bottom
    planes[3].set(m._14 - m._12, m._24 - m._22, m._34 - m._32, m._44 - m._42); // top
    planes[4].set(m._14 + m._13, m._24 + m._23, m._34 + m._33, m._44 + m._43); // near
    planes[5].set(m._14 - m._13, m._24 - m._23, m._34 - m._33, m._44 - m._43); // far
    for (int i = 0; i < 6; ++i) {
        const float L = _sqrt(planes[i].x * planes[i].x + planes[i].y * planes[i].y + planes[i].z * planes[i].z);
        if (L > 0.0001f) {
            const float inv = 1.0f / L;
            planes[i].x *= inv; planes[i].y *= inv; planes[i].z *= inv; planes[i].w *= inv;
        }
    }
}

// ----- PrepareFrame ---------------------------------------------------------
void CDetailManager::PrepareFrame(const VK::FrameContext& ctx)
{
    m_Frame.valid = false;
    if (!m_bCreated || !m_GenPipeline || !m_GfxPipeline) return;
    if (objects.empty()) return;
    if (!ctx.viewProj) return;

    // Camera position from device transform (world inverse-transformed origin).
    // Approx: read it from Device.vCameraPosition for now — engine sets that
    // global before Render(). (TODO: thread it through FrameContext explicitly.)
    const Fvector eye = Device.vCameraPosition;

    // Density / d_size: read from `r__detail_density` console var. Lower
    // density = bigger d_size = more grass per slot. Clamped 2..10 so the
    // shader's posPerSlot stays in a sane range.
    const float density = _max(ps_current_detail_density, 0.1f);
    m_Frame.d_size      = u32(_max(2.0f, _min(10.0f, ceilf(dm_slot_size / density))));
    m_Frame.posPerSlot  = (m_Frame.d_size + 1) * (m_Frame.d_size + 1);

    // Slot range driven by `r__detail_radius` cvar (metres). Default 100m.
    // `r__detail_radius 0` disables grass entirely (totalPositions=0).
    const float gen_radius = float(_max(ps_r__detail_radius, 0));
    if (gen_radius < 1.0f) {
        m_Frame.totalPositions = 0;
        m_Frame.valid          = false;
        return;
    }
    const int gen_half  = int(floorf(gen_radius / dm_slot_size));
    const int slot_eye_x = int(floorf(eye.x / dm_slot_size + 0.5f));
    const int slot_eye_z = int(floorf(eye.z / dm_slot_size + 0.5f));
    m_Frame.slotMinSX  = slot_eye_x - gen_half;
    m_Frame.slotMinSZ  = slot_eye_z - gen_half;
    m_Frame.slotCountX = gen_half * 2 + 1;
    m_Frame.slotCountZ = gen_half * 2 + 1;

    m_Frame.totalPositions = u32(m_Frame.slotCountX) * u32(m_Frame.slotCountZ) * m_Frame.posPerSlot;
    m_Frame.groupCount     = (m_Frame.totalPositions + 255) / 256;

    // Refresh GenUBO — written every frame because slot range / camera change.
    if (m_GenUBO && m_GenUBO->IsMapped()) {
        DetailGenUBO* u = (DetailGenUBO*)m_GenUBO->m_Mapped;
        u->hmOriginX     = m_HMOriginX;
        u->hmOriginZ     = m_HMOriginZ;
        u->hmInvScaleX   = (m_HMWorldSizeX > 0.0f) ? (1.0f / m_HMWorldSizeX) : 0.0f;
        u->hmInvScaleZ   = (m_HMWorldSizeZ > 0.0f) ? (1.0f / m_HMWorldSizeZ) : 0.0f;
        u->totalPositions= m_Frame.totalPositions;
        u->numObjTypes   = u32(objects.size());
        u->outputCapacity= GPU_OUTPUT_CAPACITY;
        u->posPerSlot    = m_Frame.posPerSlot;
        u->dtOffsX       = float(dtH.offs_x);
        u->dtOffsZ       = float(dtH.offs_z);
        u->dtSizeX       = float(dtH.size_x);
        u->dtSizeZ       = float(dtH.size_z);
        u->slotSize      = dm_slot_size;
        // The gen shader multiplies every item's scale by this — the original
        // engine does `Item.scale *= ps_current_detail_scale` in Decompress
        // (DetailManager_Decompress.cpp:189). It was hardcoded 1.0 here, so the
        // user's r__detail_scale (1.2 in user.ltx) was ignored and ALL grass /
        // bushes / flowers rendered ~17% smaller than R4 — "не пышно".
        u->detailHeight  = ps_current_detail_scale;
        u->hmWidth       = m_HeightmapW;
        u->hmHeight      = m_HeightmapH;
        m_GenUBO->Flush();
    }

    // Graphics push constants
    m_GfxConstants.mViewProj = *ctx.viewProj;
    m_time_pos += ctx.dt;

    // Env-driven wind:
    //   - direction taken from `Environment().CurrentEnv->wind_direction` (radians),
    //   - amplitude/speed lerped between `swing_desc[0]` (normal) and
    //     `swing_desc[1]` (fast) by `wind_velocity`-derived strength factor.
    // Engine internal `wind_velocity` is NOT a real m/s — it sits in tens-to-
    // hundreds, scaled `*0.001` everywhere it's used (Environment.cpp:490).
    // Multiplying that into the shader's displacement amplitude (which is in
    // metres) yanked grass 10+ m per wave → "rubber-band stretching" look.
    Fvector wind_dir{};      wind_dir.set(0.7f, 0.0f, 0.7f);   // fallback SE
    Fvector sun_dir{};       sun_dir.set(0.0f, 1.0f, 0.0f);
    float   wind_strength = 0.0f;                              // 0..1 lerp factor
    // Env lighting for the grass (colorize the baked sun/hemi scalars to match the
    // world ground). Neutral fallback when env isn't up yet.
    m_GfxConstants.vSunColor.set(0.6f, 0.6f, 0.6f, 0.0f);
    m_GfxConstants.vHemiColor.set(0.45f, 0.45f, 0.45f, 0.0f);
    if (g_pGamePersistent) {
        auto& env = g_pGamePersistent->Environment();
        if (env.CurrentEnv) {
            const float a = env.CurrentEnv->wind_direction;
            wind_dir.set(_cos(a), 0.0f, _sin(a));
            // Map `wind_velocity` (engine units, typically 0..1000) into a
            // 0..1 lerp factor. *0.001 matches Environment.cpp's own scale,
            // then clamp.
            wind_strength = clampr(env.CurrentEnv->wind_velocity * 0.001f, 0.0f, 1.0f);
            sun_dir       = env.CurrentEnv->sun_dir;
            m_GfxConstants.vSunColor.set(env.CurrentEnv->sun_color.x, env.CurrentEnv->sun_color.y, env.CurrentEnv->sun_color.z, 0.0f);
            m_GfxConstants.vHemiColor.set(env.CurrentEnv->hemi_color.x, env.CurrentEnv->hemi_color.y, env.CurrentEnv->hemi_color.z, 0.0f);
        }
    }
    // The same global sun boost the world receives — vk_env_light premultiplies
    // it into the LightUBO; the grass sun colour travels via push constants.
    m_GfxConstants.vSunColor.mul(ps_r_sun_boost);
    swing_current.lerp(swing_desc[0], swing_desc[1], wind_strength);

    m_GfxConstants.vWave.set(0.5f, 0.5f, swing_current.speed, m_time_pos);
    m_GfxConstants.vWind.set(wind_dir.x, 0.0f, wind_dir.z, swing_current.amp1);
    m_GfxConstants.vConsts.set(1.0f, 1.0f, sun_dir.y, 0.2f);   // sun.y feeds shader hemi calc

    // Character interaction: vInteractors[0] = player, [1..3] = nearest 3 NPCs
    // within 15 m. Empty slots have radius=0 so the shader skips them. Mirrors
    // monolith vk_DetailManager_Render.cpp:99-155.
    m_GfxConstants.vInteractors[0].set(eye.x, eye.y, eye.z, 1.2f);
    for (u32 i = 1; i < MAX_GRASS_INTERACTORS; ++i)
        m_GfxConstants.vInteractors[i].set(0.0f, 0.0f, 0.0f, 0.0f);

    if (g_pGameLevel) {
        constexpr float kMaxDistSq = 15.0f * 15.0f;
        CObject* player = g_pGameLevel->CurrentEntity();

        struct Slot { float distSq; CObject* obj; } closest[3] = {
            { kMaxDistSq, nullptr }, { kMaxDistSq, nullptr }, { kMaxDistSq, nullptr }
        };

        const u32 oc = g_pGameLevel->Objects.o_count();
        for (u32 i = 0; i < oc; ++i) {
            CObject* O = g_pGameLevel->Objects.o_get_by_iterator(i);
            if (!O || O == player || !O->getVisible()) continue;
            const Fvector p = O->Position();
            const float dx = p.x - eye.x, dz = p.z - eye.z;
            const float d2 = dx * dx + dz * dz;
            if (d2 >= closest[2].distSq) continue;
            // Insertion-sort into the 3-slot ladder.
            if (d2 < closest[0].distSq) {
                closest[2] = closest[1]; closest[1] = closest[0];
                closest[0] = { d2, O };
            } else if (d2 < closest[1].distSq) {
                closest[2] = closest[1];
                closest[1] = { d2, O };
            } else {
                closest[2] = { d2, O };
            }
        }

        for (int c = 0; c < 3; ++c) {
            if (!closest[c].obj) continue;
            Fvector p = closest[c].obj->Position();
            float r = closest[c].obj->Radius();
            r = _min(_max(r, 0.5f), 2.0f);
            m_GfxConstants.vInteractors[1 + c].set(p.x, p.y, p.z, r);
        }
    }

    m_Frame.valid = true;
}

// ----- Render ---------------------------------------------------------------
// ============================================================================
// BuildHZB — max-reduce the scene depth buffer into the Hi-Z mip chain.
//
// Runs at the top of Render(): depth already holds Pass_World statics/dynamics,
// so grass occludes against this frame's world. Transitions depth to a sampled
// layout for the build and restores DEPTH_ATTACHMENT_OPTIMAL before the grass
// draw / Pass_Sky. No-op (safe fallback to the white dummy) when the pipeline
// failed to build.
// ============================================================================
void CDetailManager::BuildHZB(VK::FrameContext& ctx)
{
    if (m_HZBPipeline == VK_NULL_HANDLE || m_HZBImage == VK_NULL_HANDLE) return;
    if (Swapchain.m_DepthImage == VK_NULL_HANDLE) return;
    if (m_HZBDescSets.size() != m_HZBMipCount) return;

    const VkCommandBuffer cmd = ctx.cmd;
    auto mipDim = [](u32 base, u32 lvl) { const u32 v = base >> lvl; return v ? v : 1u; };

    // 1) depth DEPTH_ATTACHMENT_OPTIMAL → SHADER_READ_ONLY_OPTIMAL (compute sample).
    {
        VkImageMemoryBarrier b{};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout           = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        b.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = Swapchain.m_DepthImage;
        b.subresourceRange    = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
        b.srcAccessMask       = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        b.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &b);
    }

    // 2) Per-mip max-reduce dispatch. Global compute→compute barrier between
    //    passes serializes each mip's writes before the next mip (and the gen
    //    dispatch) reads them.
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_HZBPipeline);
    for (u32 m = 0; m < m_HZBMipCount; ++m) {
        const u32 dstW = mipDim(m_HZBWidth, m);
        const u32 dstH = mipDim(m_HZBHeight, m);

        HZBBuildPush pc{};
        if (m == 0) {
            pc.srcW = s32(m_HZBDepthExtent.width);
            pc.srcH = s32(m_HZBDepthExtent.height);
            pc.srcMip = 0u;
            pc.isFirstPass = 1u;
        } else {
            pc.srcW = s32(mipDim(m_HZBWidth, m - 1));
            pc.srcH = s32(mipDim(m_HZBHeight, m - 1));
            pc.srcMip = m - 1u;
            pc.isFirstPass = 0u;
        }
        pc.dstW = s32(dstW);
        pc.dstH = s32(dstH);
        pc.dstMip = m;

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_HZBPipelineLayout,
                                0, 1, &m_HZBDescSets[m], 0, nullptr);
        vkCmdPushConstants(cmd, m_HZBPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, (dstW + 7u) / 8u, (dstH + 7u) / 8u, 1u);

        VkMemoryBarrier mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &mb, 0, nullptr, 0, nullptr);
    }

    // 3) depth back → DEPTH_ATTACHMENT_OPTIMAL for the grass draw + Pass_Sky.
    {
        VkImageMemoryBarrier b{};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        b.newLayout           = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = Swapchain.m_DepthImage;
        b.subresourceRange    = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
        b.srcAccessMask       = VK_ACCESS_SHADER_READ_BIT;
        b.dstAccessMask       = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            0, 0, nullptr, 0, nullptr, 1, &b);
    }
}

void CDetailManager::Render(VK::FrameContext& ctx)
{
    PrepareFrame(ctx);
    if (!m_Frame.valid) return;
    if (ctx.cmd == VK_NULL_HANDLE) return;
    if (m_VisibleSSBO == nullptr || m_IndirectCmdBuf == nullptr || m_AtomicCounters == nullptr) return;

    const VkCommandBuffer cmd = ctx.cmd;

    // Frames in flight (VK_FRAMES_IN_FLIGHT = 3) share ONE set of grass GPU
    // buffers (VisibleSSBO / AtomicCounters / IndirectCmdBuf / HZB mips). The
    // previous frame's GPU work may still be reading them (instance vertex
    // fetch, indirect fetch, gen-compute HZB reads) when this frame starts
    // overwriting — a write-after-read hazard that shows up as flickering
    // grass patches while the camera moves (the slot window shifts, so the
    // overwritten data no longer matches what frame N-1 meant to draw).
    // WAR needs only an execution dependency: order all prior-frame reads
    // before this frame's transfer/compute writes.
    {
        VkMemoryBarrier b{};
        b.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        b.srcAccessMask = 0;
        b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT |
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &b, 0, nullptr, 0, nullptr);
    }

    // Build the Hi-Z occlusion pyramid from this frame's depth before the gen
    // compute samples it (binding 6). Restores depth to DEPTH_ATTACHMENT_OPTIMAL.
    BuildHZB(ctx);
    const u32 nObj = u32(objects.size());
    const u32 sectionSize = GPU_OUTPUT_CAPACITY / _max(nObj, 1u);

    // -------------------------------------------------------------------
    // 1) Reset atomic counters only. We DON'T clear VisibleSSBO — the gen
    // shader's atomic-append path only writes the first `count` slots per
    // type, and the indirect draw's instanceCount = `count` so old data
    // beyond `count` is never sampled. Skipping the 96 MB fill saves a
    // huge chunk of memory bandwidth per frame.
    //
    // IndirectCmdBuf was pre-filled in CreateGpuBuffers; only instanceCount
    // (offset 4 within each 20 B record) gets patched per frame by the
    // atomic→indirect copy below.
    // -------------------------------------------------------------------
    vkCmdFillBuffer(cmd, m_AtomicCounters->GetHandle(), 0, VK_WHOLE_SIZE, 0u);

    // Barrier transfer-write → compute-read/write.
    {
        VkMemoryBarrier b{};
        b.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &b, 0, nullptr, 0, nullptr);
    }

    // -------------------------------------------------------------------
    // 2) Gen compute dispatch.
    // -------------------------------------------------------------------
    DetailGenPushConstants gpc{};
    gpc.viewProj   = *ctx.viewProj;
    ExtractFrustumPlanes(gpc.viewProj, gpc.frustumPlanes);
    gpc.cameraPos.set(Device.vCameraPosition.x, Device.vCameraPosition.y, Device.vCameraPosition.z, m_time_pos);
    // Fade params: start² | limit² | range² | density. Limit = current
    // gen radius; start = 60% of limit so grass fades over the last ~40m.
    {
        const float limit = float(_max(ps_r__detail_radius, 1));
        const float start = limit * 0.6f;
        gpc.fadeParams.set(start * start, limit * limit,
                           (limit * limit - start * start),
                           ps_current_detail_density);
    }
    gpc.slotMinSX  = m_Frame.slotMinSX;
    gpc.slotMinSZ  = m_Frame.slotMinSZ;
    gpc.slotCountX = m_Frame.slotCountX;
    gpc.slotCountZ = m_Frame.slotCountZ;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_GenPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_GenPipelineLayout,
                            0, 1, &m_GenDescSet, 0, nullptr);
    vkCmdPushConstants(cmd, m_GenPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(gpc), &gpc);
    vkCmdDispatch(cmd, m_Frame.groupCount, 1, 1);

    // Barrier compute-write → transfer-read (atomics → indirect copy).
    {
        VkMemoryBarrier b{};
        b.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 1, &b, 0, nullptr, 0, nullptr);
    }

    // -------------------------------------------------------------------
    // 3) Atomic-count → indirect.instanceCount per type.
    //    instanceCount field is at byte offset 4 within the 20-byte cmd.
    // -------------------------------------------------------------------
    if (nObj > 0) {
        xr_vector<VkBufferCopy> copies(nObj);
        for (u32 i = 0; i < nObj; ++i) {
            copies[i].srcOffset = i * sizeof(u32);
            copies[i].dstOffset = i * sizeof(VkDrawIndexedIndirectCommand) + 4;
            copies[i].size      = sizeof(u32);
        }
        vkCmdCopyBuffer(cmd, m_AtomicCounters->GetHandle(),
                        m_IndirectCmdBuf->GetHandle(),
                        nObj, copies.data());
    }

    // Barrier: VisibleSSBO ↦ vertex-input read; IndirectCmdBuf ↦ indirect read.
    {
        VkMemoryBarrier b{};
        b.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
            0, 1, &b, 0, nullptr, 0, nullptr);
    }

    // -------------------------------------------------------------------
    // 4) Begin dynamic-rendering pass (load color + depth from world pass).
    //    Single-layout convention: color is already COLOR_ATTACHMENT and depth
    //    DEPTH_ATTACHMENT for the whole frame; ExecutePasses inserted the
    //    inter-pass barrier before us. No swapchain layout transition here.
    //    (The compute→vertex/indirect barrier above is our own and stays.)
    // -------------------------------------------------------------------
    VkRenderingAttachmentInfo cAtt{};
    cAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    cAtt.imageView   = ctx.colorView;
    cAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    cAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;
    cAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingAttachmentInfo dAtt{};
    dAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    dAtt.imageView   = ctx.depthView;
    dAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    dAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;     // keep world's depth
    dAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;   // keep for sky

    VkRenderingInfo ri{};
    ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea.extent    = ctx.extent;
    ri.layerCount           = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments    = &cAtt;
    ri.pDepthAttachment     = &dAtt;
    vkCmdBeginRendering(cmd, &ri);

    // Negative-height viewport (X-Ray builds D3D-style projection).
    VkViewport vp{};
    vp.x = 0.0f; vp.y = float(ctx.extent.height);
    vp.width = float(ctx.extent.width); vp.height = -float(ctx.extent.height);
    vp.minDepth = 0.0f; vp.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{ {}, ctx.extent };
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_GfxPipeline);
    vkCmdPushConstants(cmd, m_GfxPipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0, sizeof(m_GfxConstants), &m_GfxConstants);

    // set 1 = shared env lighting (sun_vp + sun shadow map) — updated by Pass_World.
    if (VkDescriptorSet envSet = VK::EnvLight::GetCurrentSet())
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                m_GfxPipelineLayout, 1, 1, &envSet, 0, nullptr);

    // Per-type: bind diffuse, bind VBs, bind IB, draw indirect.
    for (u32 i = 0; i < nObj; ++i) {
        const VK::CDetail* obj = objects[i];
        if (!obj || !obj->m_VertexBuffer || !obj->m_IndexBuffer) continue;

        if (i < m_GfxDescSets.size() && m_GfxDescSets[i] != VK_NULL_HANDLE) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    m_GfxPipelineLayout, 0, 1, &m_GfxDescSets[i], 0, nullptr);
        }

        VkBuffer vbs[2] = { obj->m_VertexBuffer->GetHandle(), m_VisibleSSBO->GetHandle() };
        VkDeviceSize off[2] = { 0, u64(i) * u64(sectionSize) * sizeof(DetailInstance) };
        vkCmdBindVertexBuffers(cmd, 0, 2, vbs, off);
        vkCmdBindIndexBuffer(cmd, obj->m_IndexBuffer->GetHandle(), 0, VK_INDEX_TYPE_UINT16);

        vkCmdDrawIndexedIndirect(cmd,
            m_IndirectCmdBuf->GetHandle(),
            i * sizeof(VkDrawIndexedIndirectCommand),
            1, sizeof(VkDrawIndexedIndirectCommand));
    }

    vkCmdEndRendering(cmd);
    // No exit transition: image stays COLOR_ATTACHMENT for Pass_Sky / UI.

    // Diagnostic — once.
    static bool s_diag = false;
    if (!s_diag) {
        Msg("[VK Grass] First Render: types=%u sectionSize=%u totalPositions=%u groupCount=%u",
            nObj, sectionSize, m_Frame.totalPositions, m_Frame.groupCount);
        Msg("[VK Grass] Tunables: r__detail_radius=%d  r__detail_density=%.2f  posPerSlot=%u",
            ps_r__detail_radius, ps_current_detail_density, m_Frame.posPerSlot);
        s_diag = true;
    }
}

}  // namespace VK
