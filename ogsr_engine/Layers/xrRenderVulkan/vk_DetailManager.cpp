// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan - CDetailManager Session A: load + heightmap bake +
// SSBO upload. See vk_DetailManager.h header notes.

#include "stdafx.h"
#include "vk_DetailManager.h"
#include "HW_Vulkan.h"
#include "vk_image.h"      // VK::CreateImage / CreateImage2D / CreateImageView
#include "vk_compute_util.h" // VK::MakePipelineLayout / CreateComputePipeline
#include "vk_swapchain.h"               // for color/depth formats in CreateGfxPipeline
#include "vk_scene_color.h"             // HDR scene target format
#include "vk_motionvec.h"               // VK::MotionVec::Format — RG16F MV target (grass MV overlay)
#include "vk_pipeline_cache.h"          // VK::PipelineCache::GetCacheObject() — shared disk-backed cache
#include "vk_env_light.h"               // VK::EnvLight — set 1 (sun_vp + sun shadow map)
#include "vk_shaders.h"                 // g_ShaderManager
#include "vk_texture.h"                 // CVulkanTexture (LoadDDS)
#include "vk_pass_context.h"            // FrameContext
#include "CRender_Vulkan.h"             // RImplementation (for sampler reuse if needed)

#include "../../xr_3da/IGame_Persistent.h"
#include "../../xr_3da/IGame_Level.h"
#include "../../xrCDB/xrCDB.h"           // CDB::TRI, CDB::MODEL

namespace VK
{

// Warn loudly if a push range exceeds this GPU's maxPushConstantsSize. The grass
// push blocks are 208 B; Vulkan only *guarantees* 128 B, so vkCreatePipelineLayout
// would fail on a minimal-limit GPU. Desktop GPUs offer 256, so it works here.
static void DM_CheckPushSize(const char* tag, u32 size)
{
    if (size > VulkanHW.Caps.maxPushConstantsSize)
        Msg("![VK Grass] %s push %u B exceeds device maxPushConstantsSize %u B — pipeline layout creation will FAIL on this GPU",
            tag, size, VulkanHW.Caps.maxPushConstantsSize);
}

// ============================================================================
// SSwingValue::lerp
// ============================================================================
void CDetailManager::SSwingValue::lerp(const SSwingValue& v1, const SSwingValue& v2, float factor)
{
    const float inv = 1.0f - factor;
    rot1  = v1.rot1  * inv + v2.rot1  * factor;
    rot2  = v1.rot2  * inv + v2.rot2  * factor;
    amp1  = v1.amp1  * inv + v2.amp1  * factor;
    amp2  = v1.amp2  * inv + v2.amp2  * factor;
    speed = v1.speed * inv + v2.speed * factor;
}

// ============================================================================
// Construction / destruction
// ============================================================================
CDetailManager::CDetailManager()
{
    // Defaults match monolith so Session B picks up sensible values before
    // the env subsystem is wired.
    swing_desc[0] = { 0.0f,     0.0f,     0.1f, 0.2f, 1.0f };  // normal
    swing_desc[1] = { PI_MUL_2, PI_MUL_2, 0.3f, 0.5f, 3.0f };  // fast
    swing_current = swing_desc[0];
}

CDetailManager::~CDetailManager()
{
    Unload();
}

// ============================================================================
// QueryDB — world-slot lookup with empty-slot fallback.
// ============================================================================
DetailSlot& CDetailManager::QueryDB(int sx, int sz)
{
    const int db_x = sx + dtH.offs_x;
    const int db_z = sz + dtH.offs_z;
    if (db_x >= 0 && db_x < int(dtH.size_x) && db_z >= 0 && db_z < int(dtH.size_z) && dtSlots) {
        return dtSlots[u32(db_z) * dtH.size_x + u32(db_x)];
    }
    DS_empty.w_id(0, DetailSlot::ID_Empty);
    DS_empty.w_id(1, DetailSlot::ID_Empty);
    DS_empty.w_id(2, DetailSlot::ID_Empty);
    DS_empty.w_id(3, DetailSlot::ID_Empty);
    return DS_empty;
}

// ============================================================================
// Load — full Session A pipeline:
//   1. Open level.details, parse header + objects + slots
//   2. BakeHeightmap from collision tris
//   3. UploadSlotData (per-slot 32 B SSBO)
//   4. UploadObjInfo (per-type 16 B SSBO)
// ============================================================================
void CDetailManager::Load()
{
    VK::Vram::Scope _vram_scope("Grass");
    if (m_bCreated) return;

    if (!FS.exist("$level$", "level.details")) {
        Msg("[VK Grass] level.details not found — grass disabled");
        return;
    }

    string_path fn;
    FS.update_path(fn, "$level$", "level.details");
    dtFS = FS.r_open(fn);
    if (!dtFS) {
        Msg("![VK Grass] level.details exists but failed to open");
        return;
    }

    // Chunk 0: header
    dtFS->r_chunk_safe(0, &dtH, sizeof(dtH));
    Msg("[VK Grass] DetailHeader: ver=%u objects=%u offs=(%d,%d) size=(%u,%u)",
        dtH.version, dtH.object_count, dtH.offs_x, dtH.offs_z, dtH.size_x, dtH.size_z);

    if (dtH.version != DETAIL_VERSION) {
        Msg("![VK Grass] Unsupported level.details version %u (expected %u)", dtH.version, DETAIL_VERSION);
        FS.r_close(dtFS); dtFS = nullptr;
        return;
    }
    if (dtH.object_count == 0 || dtH.object_count > dm_max_objects) {
        Msg("![VK Grass] object_count=%u out of range (max %u)", dtH.object_count, dm_max_objects);
        FS.r_close(dtFS); dtFS = nullptr;
        return;
    }

    // Chunk 1: object models
    {
        IReader* objs_chunk = dtFS->open_chunk(1);
        if (!objs_chunk) {
            Msg("![VK Grass] objects chunk (1) not found");
            FS.r_close(dtFS); dtFS = nullptr;
            return;
        }
        // svector has fixed capacity (dm_max_objects=64), no reserve().
        for (u32 id = 0; id < dtH.object_count; ++id) {
            auto* dt = xr_new<VK::CDetail>();
            IReader* sub = objs_chunk->open_chunk(id);
            if (sub) {
                dt->Load(sub);
                sub->close();
            } else {
                Msg("![VK Grass] sub-chunk %u missing in objects", id);
            }
            objects.push_back(dt);
        }
        objs_chunk->close();
    }

    // Chunk 2: slot grid (zero-copy view into VFS-mapped data)
    {
        IReader* slots_chunk = dtFS->open_chunk(2);
        if (!slots_chunk) {
            Msg("![VK Grass] slots chunk (2) not found");
            FS.r_close(dtFS); dtFS = nullptr;
            return;
        }
        dtSlots = (DetailSlot*)slots_chunk->pointer();
        slots_chunk->close();
    }
    Memory.mem_fill(&DS_empty, 0, sizeof(DS_empty));
    DS_empty.w_id(0, DetailSlot::ID_Empty);
    DS_empty.w_id(1, DetailSlot::ID_Empty);
    DS_empty.w_id(2, DetailSlot::ID_Empty);
    DS_empty.w_id(3, DetailSlot::ID_Empty);

    // GPU-side prep
    BakeHeightmap();
    UploadSlotData();
    UploadObjInfo();

    // Session B: render-side resources
    CreateGpuBuffers();
    CreateDummyTextures();
    CreateHZB(Swapchain.m_Extent.width, Swapchain.m_Extent.height);  // before gen pipeline → binding 6 picks real HZB
    LoadDetailTextures();
    CreateGpuGenPipeline();
    CreateGfxPipeline();

    m_bCreated = true;
    Msg("[VK Grass] Loaded: %u objects, %ux%u slots, heightmap %ux%u",
        (u32)objects.size(), dtH.size_x, dtH.size_z, m_HeightmapW, m_HeightmapH);
}

// ============================================================================
// Unload
// ============================================================================
void CDetailManager::Unload()
{
    for (auto* obj : objects) {
        if (!obj) continue;
        obj->Unload();
        xr_delete(obj);
    }
    objects.clear();

    if (dtFS) { FS.r_close(dtFS); dtFS = nullptr; }
    dtSlots = nullptr;

    DestroyGfxPipeline();
    DestroyGpuGenPipeline();
    DestroyHZB();
    DestroyDetailTextures();
    DestroyDummyTextures();
    DestroyGpuBuffers();

    if (m_SlotDataSSBO) { m_SlotDataSSBO->Destroy(); xr_delete(m_SlotDataSSBO); }
    if (m_ObjInfoSSBO)  { m_ObjInfoSSBO->Destroy();  xr_delete(m_ObjInfoSSBO);  }
    m_TotalSlots = 0;

    if (VulkanHW.m_Device != VK_NULL_HANDLE) {
        if (m_HeightmapSampler) { vkDestroySampler   (VulkanHW.m_Device, m_HeightmapSampler, nullptr); m_HeightmapSampler = VK_NULL_HANDLE; }
        if (m_HeightmapView)    { vkDestroyImageView (VulkanHW.m_Device, m_HeightmapView,    nullptr); m_HeightmapView    = VK_NULL_HANDLE; }
        if (m_HeightmapImage)   {
            VK::Vram::DestroyImage(VulkanHW.m_Allocator, m_HeightmapImage, m_HeightmapAlloc);
            m_HeightmapImage = VK_NULL_HANDLE;
            m_HeightmapAlloc = VK_NULL_HANDLE;
        }
    }
    m_HeightmapW = m_HeightmapH = 0;

    m_bCreated = false;
    Msg("[VK Grass] Unloaded");
}

// ============================================================================
// BakeHeightmap — rasterize static collision tris into an R32F image where
// each texel holds the MIN ground Y at that XZ. Sentinel 9000.0f marks
// texels no triangle covered (compute shader tests abs(h)<9000).
//
// Tris with normal.y < 0.3 (>73° slopes — walls, ceilings) are skipped:
// grass shouldn't grow on cliffs. Holes are filled with a 5×5 neighbour
// average pass to avoid black spots.
// ============================================================================
void CDetailManager::BakeHeightmap()
{
    if (!g_pGameLevel) {
        Msg("![VK Grass] g_pGameLevel null — heightmap baking skipped");
        return;
    }
    CDB::MODEL* model = g_pGameLevel->ObjectSpace.GetStaticModel();
    if (!model) {
        Msg("![VK Grass] static collision model unavailable — heightmap baking skipped");
        return;
    }

    Fvector*  verts = g_pGameLevel->ObjectSpace.GetStaticVerts();
    CDB::TRI* tris  = g_pGameLevel->ObjectSpace.GetStaticTris();
    const u32 triCount = model->get_tris_count();

    // World bounds derived from the detail header — same coordinate frame
    // the gen compute uses (no extra transform required at sample time).
    m_HMOriginX    = -float(dtH.offs_x) * dm_slot_size;
    m_HMOriginZ    = -float(dtH.offs_z) * dm_slot_size;
    m_HMWorldSizeX = float(dtH.size_x) * dm_slot_size;
    m_HMWorldSizeZ = float(dtH.size_z) * dm_slot_size;

    m_HeightmapW = _min(u32(ceilf(m_HMWorldSizeX)), 2048u);
    m_HeightmapH = _min(u32(ceilf(m_HMWorldSizeZ)), 2048u);
    if (m_HeightmapW == 0 || m_HeightmapH == 0) {
        Msg("![VK Grass] degenerate heightmap dims (%ux%u) — abort", m_HeightmapW, m_HeightmapH);
        return;
    }

    const float texelSizeX = m_HMWorldSizeX / float(m_HeightmapW);
    const float texelSizeZ = m_HMWorldSizeZ / float(m_HeightmapH);

    xr_vector<float> heightData(m_HeightmapW * m_HeightmapH, 10000.0f);

    Msg("[VK Grass] Baking heightmap %ux%u from %u tris (%.0fx%.0fm)…",
        m_HeightmapW, m_HeightmapH, triCount, m_HMWorldSizeX, m_HMWorldSizeZ);

    u32 rasterized = 0;
    for (u32 t = 0; t < triCount; ++t) {
        CDB::TRI& T = tris[t];
        Fvector v0 = verts[T.verts[0]];
        Fvector v1 = verts[T.verts[1]];
        Fvector v2 = verts[T.verts[2]];

        Fvector n; n.mknormal(v0, v1, v2);
        if (n.y < 0.3f) continue;   // skip walls/ceilings

        const float px0 = (v0.x - m_HMOriginX) / texelSizeX, pz0 = (v0.z - m_HMOriginZ) / texelSizeZ;
        const float px1 = (v1.x - m_HMOriginX) / texelSizeX, pz1 = (v1.z - m_HMOriginZ) / texelSizeZ;
        const float px2 = (v2.x - m_HMOriginX) / texelSizeX, pz2 = (v2.z - m_HMOriginZ) / texelSizeZ;

        const int minPX = _max(0,                       (int)floorf(_min(_min(px0, px1), px2)));
        const int maxPX = _min((int)m_HeightmapW - 1,   (int)ceilf (_max(_max(px0, px1), px2)));
        const int minPZ = _max(0,                       (int)floorf(_min(_min(pz0, pz1), pz2)));
        const int maxPZ = _min((int)m_HeightmapH - 1,   (int)ceilf (_max(_max(pz0, pz1), pz2)));

        const float dx10 = px1 - px0, dz10 = pz1 - pz0;
        const float dx20 = px2 - px0, dz20 = pz2 - pz0;
        const float denom = dx10 * dz20 - dx20 * dz10;
        if (_abs(denom) < 1e-6f) continue;       // degenerate
        const float invDenom = 1.0f / denom;

        for (int pz = minPZ; pz <= maxPZ; ++pz) {
            for (int px = minPX; px <= maxPX; ++px) {
                const float qx = float(px) + 0.5f - px0;
                const float qz = float(pz) + 0.5f - pz0;
                const float u  = (qx * dz20 - qz * dx20) * invDenom;
                const float v  = (dx10 * qz - dz10 * qx) * invDenom;
                if (u < -0.01f || v < -0.01f || (u + v) > 1.01f) continue;
                const float y = v0.y + u * (v1.y - v0.y) + v * (v2.y - v0.y);
                const u32 idx = u32(pz) * m_HeightmapW + u32(px);

                // Authored-range filter: UNDERGROUND floors (tunnels, bunkers)
                // face up too, so the plain MIN picked the basement floor for
                // every surface texel above one — the gen's interior guard
                // (terrainY < y_base-1) then rejected ALL surface grass there
                // = the "field 300×300 without grass" holes. Accept a candidate
                // only inside the slot's authored detail Y range (±2 m). Slots
                // with no details keep the raw MIN (their texels only feed
                // border gathers; empty slots never spawn grass anyway).
                const float wx = m_HMOriginX + (float(px) + 0.5f) * texelSizeX;
                const float wz = m_HMOriginZ + (float(pz) + 0.5f) * texelSizeZ;
                DetailSlot& ds = QueryDB(int(floorf(wx / dm_slot_size)), int(floorf(wz / dm_slot_size)));
                const bool slotHasDetails =
                    ds.r_id(0) != DetailSlot::ID_Empty || ds.r_id(1) != DetailSlot::ID_Empty ||
                    ds.r_id(2) != DetailSlot::ID_Empty || ds.r_id(3) != DetailSlot::ID_Empty;
                if (slotHasDetails) {
                    const float yb = ds.r_ybase();
                    if (y < yb - 2.0f || y > yb + ds.r_yheight() + 2.0f) continue;
                }

                if (y < heightData[idx]) heightData[idx] = y;
            }
        }
        ++rasterized;
    }

    // Hole fill: any sentinel-marked texel takes a 5×5 neighbour mean.
    for (u32 z = 0; z < m_HeightmapH; ++z) {
        for (u32 x = 0; x < m_HeightmapW; ++x) {
            const u32 idx = z * m_HeightmapW + x;
            if (heightData[idx] < 9999.0f) continue;
            float sum = 0; int count = 0;
            for (int dz = -2; dz <= 2; ++dz) {
                for (int dx = -2; dx <= 2; ++dx) {
                    const int nx = int(x) + dx, nz = int(z) + dz;
                    if (nx < 0 || nx >= int(m_HeightmapW) || nz < 0 || nz >= int(m_HeightmapH)) continue;
                    const float h = heightData[u32(nz) * m_HeightmapW + u32(nx)];
                    if (h < 9999.0f) { sum += h; ++count; }
                }
            }
            // 9000.0f is the GPU sentinel — preserve when truly isolated.
            heightData[idx] = (count > 0) ? (sum / float(count)) : 9000.0f;
        }
    }

    Msg("[VK Grass] Rasterized %u tris, uploading R32F image…", rasterized);

    // Allocate device-local R32F image via VMA.
    if (!VK::CreateImage2D(VK_FORMAT_R32_SFLOAT, { m_HeightmapW, m_HeightmapH },
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            m_HeightmapImage, m_HeightmapAlloc, "Grass.Heightmap")) {
        return;
    }
    m_HeightmapView = VK::CreateImageView(m_HeightmapImage, VK_FORMAT_R32_SFLOAT);

    VkSamplerCreateInfo sci{};
    sci.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter    = VK_FILTER_LINEAR;
    sci.minFilter    = VK_FILTER_LINEAR;
    sci.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    vkCreateSampler(VulkanHW.m_Device, &sci, nullptr, &m_HeightmapSampler);

    // Stage upload via host buffer + one-shot copy command.
    const u32 dataSize = m_HeightmapW * m_HeightmapH * sizeof(float);
    VK::CVulkanBuffer staging;
    staging.Create(dataSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
    if (void* mapped = staging.Map()) {
        memcpy(mapped, heightData.data(), dataSize);
        staging.Flush();
    }

    VkCommandBuffer cmd = VulkanHW.BeginSingleTimeCommands();
    if (cmd != VK_NULL_HANDLE) {
        VkImageMemoryBarrier b{};
        b.sType                = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.srcAccessMask        = 0;
        b.dstAccessMask        = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.oldLayout            = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout            = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.srcQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
        b.image                = m_HeightmapImage;
        b.subresourceRange     = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &b);

        VkBufferImageCopy r{};
        r.bufferOffset      = 0;
        r.imageSubresource  = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        r.imageExtent       = { m_HeightmapW, m_HeightmapH, 1 };
        vkCmdCopyBufferToImage(cmd, staging.GetHandle(), m_HeightmapImage,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);

        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        b.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &b);

        VulkanHW.EndSingleTimeCommands(cmd);
    }
    staging.Destroy();

    Msg("[VK Grass] Heightmap %ux%u baked (%.1fMB), origin=(%.1f, %.1f)",
        m_HeightmapW, m_HeightmapH, dataSize / (1024.0f * 1024.0f), m_HMOriginX, m_HMOriginZ);
}

// ============================================================================
// UploadSlotData — pack DetailSlot bitfield array into GpuSlotPacked SSBO.
// 32 B/slot. Lighting fields keep R4 quantisation (4-bit→16-bit fixed),
// palette alphas widen 4-bit→8-bit (a*255/15). Compute shader bilerps.
// ============================================================================
void CDetailManager::UploadSlotData()
{
    if (!dtSlots) return;

    m_TotalSlots = dtH.size_x * dtH.size_z;
    const u32 dataSize = m_TotalSlots * u32(sizeof(GpuSlotPacked));

    // Anomaly/GAMMA level.details bakes carry BUGGED full-black slots: c_hemi==0
    // blobs under tree crowns (SSFX's own deffer_grass.vs clamps with the comment
    // "Some spots are bugged (Full black)"). Heal instead of just clamping: rebuild
    // each zero slot from the healthy 7x7 neighbourhood mean. Small crown blobs
    // fill in with plausible shade; large legitimately dark areas (building
    // interiors) keep zero neighbours and stay dark. The SSFX 0.05 floor is
    // applied on top at pack time below.
    xr_vector<u8> hemiHealed(m_TotalSlots);
    for (u32 i = 0; i < m_TotalSlots; ++i) hemiHealed[i] = u8(dtSlots[i].c_hemi);
    u32 healedCnt = 0;
    {
        const int W = int(dtH.size_x), H = int(dtH.size_z);
        for (int z = 0; z < H; ++z)
            for (int x = 0; x < W; ++x) {
                const u32 i = u32(z) * u32(W) + u32(x);
                if (dtSlots[i].c_hemi != 0) continue;
                u32 sum = 0, cnt = 0;
                for (int dz = -3; dz <= 3; ++dz)
                    for (int dx = -3; dx <= 3; ++dx) {
                        const int nx = x + dx, nz = z + dz;
                        if (nx < 0 || nx >= W || nz < 0 || nz >= H) continue;
                        const u32 v = dtSlots[u32(nz) * u32(W) + u32(nx)].c_hemi;
                        if (v > 0) { sum += v; ++cnt; }
                    }
                if (cnt >= 3) { hemiHealed[i] = u8((sum + cnt / 2) / cnt); ++healedCnt; }
            }
    }
    if (healedCnt)
        Msg("[VK Grass] healed %u bugged hemi==0 slots (of %u) from neighbours", healedCnt, m_TotalSlots);

    xr_vector<GpuSlotPacked> packed(m_TotalSlots);
    for (u32 i = 0; i < m_TotalSlots; ++i) {
        DetailSlot& ds = dtSlots[i];
        GpuSlotPacked& p = packed[i];
        p.y_base   = ds.r_ybase();
        p.y_height = ds.r_yheight();
        p.ids      = u32(ds.id0)
                  | (u32(ds.id1) << 8)
                  | (u32(ds.id2) << 16)
                  | (u32(ds.id3) << 24);
        // hemi: healed value + the SSFX 0.05 floor (their deffer_grass.vs clamp).
        const float hemiF = std::max(float(hemiHealed[i]) / 15.0f, 0.05f);
        p.lighting = u32(ds.r_qclr(ds.c_dir,  15) * 65535.0f)
                  | (u32(hemiF * 65535.0f) << 16);

        auto packPal = [](const DetailPalette& pal) -> u32 {
            const u32 a0 = u32(float(pal.a0) / 15.0f * 255.0f);
            const u32 a1 = u32(float(pal.a1) / 15.0f * 255.0f);
            const u32 a2 = u32(float(pal.a2) / 15.0f * 255.0f);
            const u32 a3 = u32(float(pal.a3) / 15.0f * 255.0f);
            return a0 | (a1 << 8) | (a2 << 16) | (a3 << 24);
        };
        p.palette0 = packPal(ds.palette[0]);
        p.palette1 = packPal(ds.palette[1]);
        p.palette2 = packPal(ds.palette[2]);
        p.palette3 = packPal(ds.palette[3]);
    }

    m_SlotDataSSBO = xr_new<VK::CVulkanBuffer>();
    m_SlotDataSSBO->Create(dataSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    m_SlotDataSSBO->Upload(packed.data(), dataSize);

    Msg("[VK Grass] Slot SSBO uploaded: %u slots (%.1f KB)", m_TotalSlots, dataSize / 1024.0f);
}

// ============================================================================
// UploadObjInfo — per-detail-type metadata for compute frustum cull / scale.
// ============================================================================
void CDetailManager::UploadObjInfo()
{
    if (objects.empty()) return;

    const u32 count = u32(objects.size());
    const u32 dataSize = count * u32(sizeof(GpuDetailObjInfo));

    xr_vector<GpuDetailObjInfo> infos(count);
    for (u32 i = 0; i < count; ++i) {
        const VK::CDetail* obj = objects[i];
        if (obj) {
            infos[i].minScale = obj->m_MinScale;
            infos[i].maxScale = obj->m_MaxScale;
            infos[i].bvRadius = obj->bv_sphere.R;
            infos[i].flags    = obj->m_Flags;
        } else {
            infos[i] = { 1.0f, 1.0f, 1.0f, 0u };
        }
    }

    m_ObjInfoSSBO = xr_new<VK::CVulkanBuffer>();
    m_ObjInfoSSBO->Create(dataSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);
    m_ObjInfoSSBO->Upload(infos.data(), dataSize);

    Msg("[VK Grass] ObjInfo SSBO uploaded: %u types (%.1f KB)", count, dataSize / 1024.0f);
}

// ============================================================================
// Session B: GPU buffers — VisibleSSBO (compute output → instance VB),
// IndirectCmdBuf, AtomicCounters, GenUBO.
// ============================================================================
void CDetailManager::CreateGpuBuffers()
{
    m_OutputCapacity  = GPU_OUTPUT_CAPACITY;   // per-level manager: start small, grow to demand
    m_PendingCapacity = 0;
    m_VisibleSSBO = xr_new<VK::CVulkanBuffer>();
    m_VisibleSSBO->Create(
        u64(m_OutputCapacity) * sizeof(DetailInstance),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT  |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);

    m_IndirectCmdBuf = xr_new<VK::CVulkanBuffer>();
    m_IndirectCmdBuf->Create(
        GPU_MAX_OBJ_TYPES * sizeof(VkDrawIndexedIndirectCommand),
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT  |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);

    m_AtomicCounters = xr_new<VK::CVulkanBuffer>();
    m_AtomicCounters->Create(
        (GPU_MAX_OBJ_TYPES * 2 + 4) * sizeof(u32),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT   |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);

    // Host mirror of counters[0..127] (used [0..63] + dropped [64..127] per
    // type) — read a few frames stale, logged throttled from Render, and
    // drives the auto-grow. Overflow used to be SILENT and showed up as whole
    // fields without grass (per-type section exhausted).
    m_OverflowRB = xr_new<VK::CVulkanBuffer>();
    m_OverflowRB->Create(2u * GPU_MAX_OBJ_TYPES * sizeof(u32),
                         VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
    m_OverflowPtr = (u32*)m_OverflowRB->Map();
    if (m_OverflowPtr) memset(m_OverflowPtr, 0, 2u * GPU_MAX_OBJ_TYPES * sizeof(u32));

    // GenUBO: host-visible so PrepareFrame can rewrite without a staging copy.
    m_GenUBO = xr_new<VK::CVulkanBuffer>();
    m_GenUBO->Create(
        sizeof(DetailGenUBO),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_HOST);

    // Shadow-CASTER set: second (smaller) instance stream from the caster gen
    // dispatch — distance-only cull, so blades outside the camera frustum still
    // cast into the spot/cascade/cube/VSM shadow maps.
    m_CasterSSBO = xr_new<VK::CVulkanBuffer>();
    m_CasterSSBO->Create(
        u64(GPU_CASTER_CAPACITY) * sizeof(DetailInstance),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT  |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);

    m_CasterIndirectBuf = xr_new<VK::CVulkanBuffer>();
    m_CasterIndirectBuf->Create(
        GPU_MAX_OBJ_TYPES * sizeof(VkDrawIndexedIndirectCommand),
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT  |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);

    m_CasterAtomic = xr_new<VK::CVulkanBuffer>();
    m_CasterAtomic->Create(
        (GPU_MAX_OBJ_TYPES * 2 + 4) * sizeof(u32),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT   |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, true);

    Msg("[VK Grass] GPU buffers: visible=%.1fMB (auto-grow, cap %.0fMB) indirect=%.1fKB atomic=%.1fKB ubo=64B sectionSize=%u",
        u64(m_OutputCapacity) * sizeof(DetailInstance) / (1024.0f * 1024.0f),
        u64(GPU_OUTPUT_CAP_MAX) * sizeof(DetailInstance) / (1024.0f * 1024.0f),
        GPU_MAX_OBJ_TYPES * sizeof(VkDrawIndexedIndirectCommand) / 1024.0f,
        (GPU_MAX_OBJ_TYPES * 2 + 4) * sizeof(u32) / 1024.0f,
        m_OutputCapacity / _max(u32(objects.size()), 1u));

    // One-shot fill of IndirectCmdBuf: indexCount/firstIndex/vertexOffset/
    // firstInstance never change during the level. instanceCount is patched
    // each frame by the atomic→indirect copy (see Render's Phase 7). Doing
    // the static fill once at load avoids ~21 × 20 B vkCmdUpdateBuffer per
    // frame plus the implicit transfer barrier it triggers.
    {
        const u32 nObj = u32(objects.size());
        xr_vector<VkDrawIndexedIndirectCommand> cmds(GPU_MAX_OBJ_TYPES);
        for (u32 i = 0; i < GPU_MAX_OBJ_TYPES; ++i) {
            cmds[i].indexCount    = (i < nObj && objects[i]) ? objects[i]->m_IndexCount : 0;
            cmds[i].instanceCount = 0;
            cmds[i].firstIndex    = 0;
            cmds[i].vertexOffset  = 0;
            cmds[i].firstInstance = 0;
        }

        // Stage upload via host buffer + one-shot copy command (vkCmdUpdateBuffer
        // also works but is capped at 64 KB; this is safer for the full 1.3 KB).
        const VkDeviceSize sz = GPU_MAX_OBJ_TYPES * sizeof(VkDrawIndexedIndirectCommand);
        VK::CVulkanBuffer staging;
        staging.Create(sz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
        if (void* m = staging.Map()) { memcpy(m, cmds.data(), sz); staging.Flush(); }
        VkCommandBuffer cmd = VulkanHW.BeginSingleTimeCommands();
        if (cmd != VK_NULL_HANDLE) {
            VkBufferCopy r{ 0, 0, sz };
            vkCmdCopyBuffer(cmd, staging.GetHandle(), m_IndirectCmdBuf->GetHandle(), 1, &r);
            if (m_CasterIndirectBuf)   // same static fields; instanceCount patched per frame
                vkCmdCopyBuffer(cmd, staging.GetHandle(), m_CasterIndirectBuf->GetHandle(), 1, &r);
            VulkanHW.EndSingleTimeCommands(cmd);
        }
        staging.Destroy();
    }
}

void CDetailManager::DestroyGpuBuffers()
{
    if (m_VisibleSSBO)    { m_VisibleSSBO->Destroy();    xr_delete(m_VisibleSSBO); }
    if (m_IndirectCmdBuf) { m_IndirectCmdBuf->Destroy(); xr_delete(m_IndirectCmdBuf); }
    if (m_AtomicCounters) { m_AtomicCounters->Destroy(); xr_delete(m_AtomicCounters); }
    if (m_OverflowRB)     { m_OverflowRB->Destroy();     xr_delete(m_OverflowRB); m_OverflowPtr = nullptr; }
    if (m_GenUBO)         { m_GenUBO->Destroy();         xr_delete(m_GenUBO); }
    if (m_CasterSSBO)        { m_CasterSSBO->Destroy();        xr_delete(m_CasterSSBO); }
    if (m_CasterIndirectBuf) { m_CasterIndirectBuf->Destroy(); xr_delete(m_CasterIndirectBuf); }
    if (m_CasterAtomic)      { m_CasterAtomic->Destroy();      xr_delete(m_CasterAtomic); }
}

// ============================================================================
// Helper: 1×1 image with given pixel value in TRANSFER_DST → SHADER_READ_ONLY.
// ============================================================================
namespace {
bool Create1x1Image(VkFormat fmt, const void* pixel, u32 pixelSize,
                    VkImage& outImage, VmaAllocation& outAlloc, VkImageView& outView)
{
    if (!VK::CreateImage2D(fmt, { 1, 1 },
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            outImage, outAlloc))
        return false;

    outView = VK::CreateImageView(outImage, fmt);

    VK::CVulkanBuffer staging;
    staging.Create(pixelSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
    if (void* m = staging.Map()) { memcpy(m, pixel, pixelSize); staging.Flush(); }

    VkCommandBuffer cmd = VulkanHW.BeginSingleTimeCommands();
    if (cmd != VK_NULL_HANDLE) {
        VkImageMemoryBarrier b{};
        b.sType                = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout            = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout            = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.srcQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex  = VK_QUEUE_FAMILY_IGNORED;
        b.image                = outImage;
        b.subresourceRange     = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        b.dstAccessMask        = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &b);

        VkBufferImageCopy r{};
        r.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        r.imageExtent      = { 1, 1, 1 };
        vkCmdCopyBufferToImage(cmd, staging.GetHandle(), outImage,
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);

        b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        b.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        b.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &b);
        VulkanHW.EndSingleTimeCommands(cmd);
    }
    staging.Destroy();
    return true;
}
}  // namespace

void CDetailManager::CreateDummyTextures()
{
    // HZB: white = depth 1.0 → instanceDepth > 1.0 never → no occlusion cull.
    const u32 white = 0x3F800000u;   // 1.0f bits, but we read .r as float → fine for R32F too
    const float whiteF = 1.0f;
    Create1x1Image(VK_FORMAT_R32_SFLOAT, &whiteF, sizeof(float),
                   m_DummyHZBImage, m_DummyHZBAlloc, m_DummyHZBView);

    // Trail: black = no press-down.
    const float blackF = 0.0f;
    Create1x1Image(VK_FORMAT_R32_SFLOAT, &blackF, sizeof(float),
                   m_DummyTrailImage, m_DummyTrailAlloc, m_DummyTrailView);

    (void)white;  // silence unused
}

void CDetailManager::DestroyDummyTextures()
{
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
    if (m_DummyHZBView)   { vkDestroyImageView(VulkanHW.m_Device, m_DummyHZBView,   nullptr); m_DummyHZBView   = VK_NULL_HANDLE; }
    if (m_DummyHZBImage)  { VK::Vram::DestroyImage(VulkanHW.m_Allocator, m_DummyHZBImage,  m_DummyHZBAlloc); m_DummyHZBImage = VK_NULL_HANDLE; }
    if (m_DummyTrailView) { vkDestroyImageView(VulkanHW.m_Device, m_DummyTrailView, nullptr); m_DummyTrailView = VK_NULL_HANDLE; }
    if (m_DummyTrailImage){ VK::Vram::DestroyImage(VulkanHW.m_Allocator, m_DummyTrailImage,m_DummyTrailAlloc); m_DummyTrailImage = VK_NULL_HANDLE; }
}

// ============================================================================
// HZB — Hierarchical-Z occlusion pyramid.
//
// Built each frame inside Render() from the scene depth buffer (which already
// holds Pass_World statics + dynamics). A max-reduce compute pass produces a
// mip chain; the gen compute samples it (binding 6) and drops instances that
// are farther than the farthest surface in their screen region — i.e. hidden
// behind world geometry. mip0 is half the depth resolution.
//
// Layout: the HZB image stays in GENERAL for its whole lifetime (storage write
// each pass + sampled read by both later passes and the gen shader). The depth
// buffer is transitioned DEPTH_ATTACHMENT_OPTIMAL→SHADER_READ_ONLY and back
// around the build (see BuildHZB) — a deliberate, localized exception to the
// single-layout convention, owned entirely by the grass pass.
// ============================================================================
namespace {
inline u32 MipDim(u32 base, u32 level) { const u32 v = base >> level; return v ? v : 1u; }
}  // namespace

void CDetailManager::CreateHZB(u32 depthW, u32 depthH)
{
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
    if (depthW == 0 || depthH == 0) return;
    if (Swapchain.m_DepthImage == VK_NULL_HANDLE) {
        Msg("![VK Grass] HZB: no depth image — occlusion culling disabled");
        return;
    }

    // mip0 = half depth res; full mip chain down to 1×1.
    m_HZBWidth    = _max(1u, depthW / 2u);
    m_HZBHeight   = _max(1u, depthH / 2u);
    m_HZBMipCount = 1u;
    for (u32 d = _max(m_HZBWidth, m_HZBHeight); d > 1u; d >>= 1) ++m_HZBMipCount;
    m_HZBDepthExtent = { depthW, depthH };

    // ----- HZB image (R32F, mip chain, STORAGE+SAMPLED) --------------------
    VK::ImageDesc hzb;
    hzb.format = VK_FORMAT_R32_SFLOAT;
    hzb.extent = { m_HZBWidth, m_HZBHeight, 1 };
    hzb.mips   = m_HZBMipCount;
    hzb.usage  = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    hzb.name   = "Grass.HZB";
    if (!VK::CreateImage(hzb, m_HZBImage, m_HZBAlloc)) return;

    // Full-mip sampled view (gen binding 6 + build source for HZB passes).
    m_HZBView = VK::CreateImageView(m_HZBImage, VK_FORMAT_R32_SFLOAT, VK_IMAGE_VIEW_TYPE_2D,
                                    VK_IMAGE_ASPECT_COLOR_BIT, 0, m_HZBMipCount);

    // Per-mip single-level storage views (build destination).
    m_HZBMipViews.resize(m_HZBMipCount, VK_NULL_HANDLE);
    for (u32 m = 0; m < m_HZBMipCount; ++m)
        m_HZBMipViews[m] = VK::CreateImageView(m_HZBImage, VK_FORMAT_R32_SFLOAT, VK_IMAGE_VIEW_TYPE_2D,
                                               VK_IMAGE_ASPECT_COLOR_BIT, m, 1);

    // DEPTH-aspect view of the swapchain depth (first build pass source).
    // Explicit DEPTH-only aspect so combined D32_S8 formats stay samplable.
    m_DepthSampleView = VK::CreateImageView(Swapchain.m_DepthImage, Swapchain.m_DepthFormat,
                                            VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1);

    // Sampler — nearest + clamp; the build shader picks exact texels itself,
    // the gen shader reads explicit mips. CLAMP avoids wrap at screen edges.
    VkSamplerCreateInfo sci{};
    sci.sType         = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter     = VK_FILTER_NEAREST;
    sci.minFilter     = VK_FILTER_NEAREST;
    sci.mipmapMode    = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW  = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.minLod        = 0.0f;
    sci.maxLod        = float(m_HZBMipCount);
    vkCreateSampler(VulkanHW.m_Device, &sci, nullptr, &m_HZBSampler);

    // ----- One-time UNDEFINED → GENERAL for all mips -----------------------
    if (VkCommandBuffer cmd = VulkanHW.BeginSingleTimeCommands()) {
        VkImageMemoryBarrier b{};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = m_HZBImage;
        b.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, 0, m_HZBMipCount, 0, 1 };
        b.dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &b);
        VulkanHW.EndSingleTimeCommands(cmd);
        m_bHZBLayoutInit = true;
    }

    // ----- Build compute pipeline + per-mip descriptor sets ----------------
    if (!g_ShaderManager) { Msg("![VK Grass] HZB: g_ShaderManager null — build disabled"); return; }

    VkDescriptorSetLayoutBinding bnd[2]{};
    bnd[0].binding = 0; bnd[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bnd[0].descriptorCount = 1; bnd[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bnd[1].binding = 1; bnd[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bnd[1].descriptorCount = 1; bnd[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lci.bindingCount = 2; lci.pBindings = bnd;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &m_HZBDescLayout) != VK_SUCCESS) {
        Msg("![VK Grass] HZB DSL create failed"); return;
    }

    VkDescriptorPoolSize ps[2]{};
    ps[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; ps[0].descriptorCount = m_HZBMipCount;
    ps[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;          ps[1].descriptorCount = m_HZBMipCount;
    VkDescriptorPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = m_HZBMipCount; pci.poolSizeCount = 2; pci.pPoolSizes = ps;
    vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &m_HZBDescPool);

    m_HZBDescSets.resize(m_HZBMipCount, VK_NULL_HANDLE);
    xr_vector<VkDescriptorSetLayout> layouts(m_HZBMipCount, m_HZBDescLayout);
    VkDescriptorSetAllocateInfo dai{};
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = m_HZBDescPool; dai.descriptorSetCount = m_HZBMipCount;
    dai.pSetLayouts = layouts.data();
    vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, m_HZBDescSets.data());

    m_HZBPipelineLayout = VK::MakePipelineLayout({ m_HZBDescLayout }, sizeof(HZBBuildPush));

    VkShaderModule cs = g_ShaderManager->Load("hzb_build.comp.spv");
    if (cs == VK_NULL_HANDLE) { Msg("![VK Grass] hzb_build.comp.spv load failed"); return; }

    m_HZBPipeline = VK::CreateComputePipeline(cs, m_HZBPipelineLayout, "Grass.HZBBuild");
    if (m_HZBPipeline == VK_NULL_HANDLE) return;

    // Write the descriptor sets once — all views are stable for the HZB's life.
    // set[0]: src = depth (SHADER_READ_ONLY at sample time), dst = mip0.
    // set[i]: src = HZB full view (GENERAL),               dst = mip i.
    xr_vector<VkDescriptorImageInfo> srcInfo(m_HZBMipCount);
    xr_vector<VkDescriptorImageInfo> dstInfo(m_HZBMipCount);
    xr_vector<VkWriteDescriptorSet>  writes(m_HZBMipCount * 2);
    for (u32 m = 0; m < m_HZBMipCount; ++m) {
        srcInfo[m].sampler     = m_HZBSampler;
        srcInfo[m].imageView   = (m == 0) ? m_DepthSampleView : m_HZBView;
        srcInfo[m].imageLayout = (m == 0) ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                          : VK_IMAGE_LAYOUT_GENERAL;
        dstInfo[m].imageView   = m_HZBMipViews[m];
        dstInfo[m].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet& ws = writes[m * 2 + 0];
        ws.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        ws.dstSet = m_HZBDescSets[m]; ws.dstBinding = 0; ws.descriptorCount = 1;
        ws.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; ws.pImageInfo = &srcInfo[m];

        VkWriteDescriptorSet& wd = writes[m * 2 + 1];
        wd.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wd.dstSet = m_HZBDescSets[m]; wd.dstBinding = 1; wd.descriptorCount = 1;
        wd.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; wd.pImageInfo = &dstInfo[m];
    }
    vkUpdateDescriptorSets(VulkanHW.m_Device, (u32)writes.size(), writes.data(), 0, nullptr);

    Msg("[VK Grass] HZB ready: %ux%u, %u mips (depth %ux%u)",
        m_HZBWidth, m_HZBHeight, m_HZBMipCount, depthW, depthH);
}

void CDetailManager::DestroyHZB()
{
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
    vkDeviceWaitIdle(VulkanHW.m_Device);   // HZB views/sets may be in flight

    if (m_HZBPipeline)       { vkDestroyPipeline(VulkanHW.m_Device, m_HZBPipeline, nullptr); m_HZBPipeline = VK_NULL_HANDLE; }
    if (m_HZBPipelineLayout) { vkDestroyPipelineLayout(VulkanHW.m_Device, m_HZBPipelineLayout, nullptr); m_HZBPipelineLayout = VK_NULL_HANDLE; }
    if (m_HZBDescPool)       { vkDestroyDescriptorPool(VulkanHW.m_Device, m_HZBDescPool, nullptr); m_HZBDescPool = VK_NULL_HANDLE; }
    m_HZBDescSets.clear();
    if (m_HZBDescLayout)     { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, m_HZBDescLayout, nullptr); m_HZBDescLayout = VK_NULL_HANDLE; }

    if (m_HZBSampler)        { vkDestroySampler(VulkanHW.m_Device, m_HZBSampler, nullptr); m_HZBSampler = VK_NULL_HANDLE; }
    if (m_DepthSampleView)   { vkDestroyImageView(VulkanHW.m_Device, m_DepthSampleView, nullptr); m_DepthSampleView = VK_NULL_HANDLE; }
    for (VkImageView v : m_HZBMipViews) if (v) vkDestroyImageView(VulkanHW.m_Device, v, nullptr);
    m_HZBMipViews.clear();
    if (m_HZBView)           { vkDestroyImageView(VulkanHW.m_Device, m_HZBView, nullptr); m_HZBView = VK_NULL_HANDLE; }
    if (m_HZBImage)          { VK::Vram::DestroyImage(VulkanHW.m_Allocator, m_HZBImage, m_HZBAlloc); m_HZBImage = VK_NULL_HANDLE; }

    m_HZBWidth = m_HZBHeight = m_HZBMipCount = 0;
    m_HZBDepthExtent = { 0, 0 };
    m_bHZBLayoutInit = false;
}

// ============================================================================
// Per-detail-type DDS load + descriptor allocation. Each detail object's
// diffuse texture sits in $level$ or $game_textures$. Sampler is shared.
// ============================================================================
void CDetailManager::LoadDetailTextures()
{
    // Sampler — anisotropic linear, repeat (atlas-style detail textures often
    // tile within their own UV space; addressing mode REPEAT is monolith default).
    VkSamplerCreateInfo sci{};
    sci.sType            = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sci.magFilter        = VK_FILTER_LINEAR;
    sci.minFilter        = VK_FILTER_LINEAR;
    sci.mipmapMode       = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sci.addressModeU     = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeV     = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.addressModeW     = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.minLod           = 0.0f;
    sci.maxLod           = VK_LOD_CLAMP_NONE;
    sci.anisotropyEnable = VK_TRUE;
    sci.maxAnisotropy    = 16.0f;
    vkCreateSampler(VulkanHW.m_Device, &sci, nullptr, &m_DetailSampler);

    m_DetailTextures.resize(objects.size(), nullptr);
    for (size_t i = 0; i < objects.size(); ++i) {
        const VK::CDetail* obj = objects[i];
        if (!obj || obj->m_TextureName.size() == 0) continue;

        string_path leaf;
        xr_sprintf(leaf, "%s.dds", obj->m_TextureName.c_str());
        string_path full;
        FS.update_path(full, "$level$", leaf);
        if (!FS.exist(full)) {
            FS.update_path(full, "$game_textures$", leaf);
            if (!FS.exist(full)) {
                Msg("![VK Grass] detail texture not found: %s.dds — using fallback", obj->m_TextureName.c_str());
                continue;
            }
        }

        auto* tex = xr_new<VK::CVulkanTexture>();
        // Grass blade albedo — Colour. The stream class is passed explicitly only
        // because the colourspace argument follows it; it keeps the historical
        // TexStreamClass::UI default (preserved as-is, not endorsed — changing grass
        // residency is a separate question from colourspace).
        if (!tex->LoadDDS(full, /*applyBCSwizzle*/ false, VK::TexStreamClass::UI,
                          VK::TexColorSpace::Color)) {
            Msg("![VK Grass] LoadDDS failed for %s", full);
            xr_delete(tex);
            continue;
        }
        m_DetailTextures[i] = tex;
    }

    // SSFX wind flow map (s_waves). Sampled in the grass vertex shader to drive
    // the flow-map wind. fx\wind_wave.dds ships with the SSFX "10 - Wind" module.
    {
        string_path full;
        FS.update_path(full, "$game_textures$", "fx\\wind_wave.dds");
        if (FS.exist(full)) {
            auto* tex = xr_new<VK::CVulkanTexture>();
            if (tex->LoadDDS(full, /*applyBCSwizzle*/ false)) {
                m_WaveTex = tex;
                Msg("[VK Grass] SSFX wind flow map loaded: fx\\wind_wave.dds");
            } else {
                Msg("![VK Grass] LoadDDS failed for fx\\wind_wave.dds — grass wind will be flat");
                xr_delete(tex);
            }
        } else {
            Msg("![VK Grass] fx\\wind_wave.dds not found — grass wind will be flat (deploy the SSFX wind texture)");
        }
    }

    Msg("[VK Grass] Detail textures: %u loaded", (u32)m_DetailTextures.size());
}

void CDetailManager::DestroyDetailTextures()
{
    for (auto* t : m_DetailTextures) {
        if (t) { t->Destroy(); xr_delete(t); }
    }
    m_DetailTextures.clear();
    if (m_WaveTex) { m_WaveTex->Destroy(); xr_delete(m_WaveTex); m_WaveTex = nullptr; }

    if (m_DetailSampler && VulkanHW.m_Device != VK_NULL_HANDLE) {
        vkDestroySampler(VulkanHW.m_Device, m_DetailSampler, nullptr);
        m_DetailSampler = VK_NULL_HANDLE;
    }
}

// ============================================================================
// Compute (generator) pipeline. 9-binding descriptor set, 208 B push range.
// Loads detail_generate.comp.spv from the global shader manager.
// ============================================================================
void CDetailManager::CreateGpuGenPipeline()
{
    if (!g_ShaderManager) {
        Msg("![VK Grass] g_ShaderManager null — gen pipeline disabled");
        return;
    }

    // 9 bindings — stored in a compact array for layout creation + write.
    // 0: heightmap   sampler       (R32F + sampler)
    // 1: SlotData    storage RO
    // 2: ObjInfo     storage RO
    // 3: Visible     storage RW
    // 4: Atomic      storage RW
    // 5: Indirect    storage RW    (gen doesn't write, but layout matches)
    // 6: HZB         sampler       (1×1 white placeholder, R32F)
    // 7: GenUBO      uniform
    // 8: TrailMap    sampler       (1×1 black placeholder, R32F)
    VkDescriptorSetLayoutBinding b[9]{};
    auto fill = [&](u32 i, VkDescriptorType t) {
        b[i].binding         = i;
        b[i].descriptorType  = t;
        b[i].descriptorCount = 1;
        b[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    };
    fill(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    fill(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    fill(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    fill(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    fill(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    fill(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    fill(6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    fill(7, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    fill(8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    VkDescriptorSetLayoutCreateInfo lci{};
    lci.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lci.bindingCount = 9;
    lci.pBindings    = b;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &m_GenDescLayout) != VK_SUCCESS) {
        Msg("![VK Grass] gen DSL create failed"); return;
    }

    // ×2: the VISIBLE set + the shadow-CASTER set (same layout, caster
    // SSBO/atomics/indirect swapped in at bindings 3/4/5).
    VkDescriptorPoolSize ps[3]{};
    ps[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; ps[0].descriptorCount = 6;
    ps[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;         ps[1].descriptorCount = 10;
    ps[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;         ps[2].descriptorCount = 2;
    VkDescriptorPoolCreateInfo pci{};
    pci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets       = 2;
    pci.poolSizeCount = 3;
    pci.pPoolSizes    = ps;
    vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &m_GenDescPool);

    VkDescriptorSetAllocateInfo dai{};
    dai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool     = m_GenDescPool;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts        = &m_GenDescLayout;
    vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, &m_GenDescSet);
    vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, &m_CasterDescSet);

    // Pipeline layout: 1 set + 208 B push.
    DM_CheckPushSize("gen", sizeof(DetailGenPushConstants));
    m_GenPipelineLayout = VK::MakePipelineLayout({ m_GenDescLayout }, sizeof(DetailGenPushConstants));

    VkShaderModule cs = g_ShaderManager->Load("detail_generate.comp.spv");
    if (cs == VK_NULL_HANDLE) {
        Msg("![VK Grass] detail_generate.comp.spv load failed"); return;
    }

    m_GenPipeline = VK::CreateComputePipeline(cs, m_GenPipelineLayout, "Grass.DetailGen");
    if (m_GenPipeline == VK_NULL_HANDLE) return;

    UpdateGenDescriptors();
    Msg("[VK Grass] Gen pipeline OK (1 set, 9 bindings, 208 B push)");
}

void CDetailManager::DestroyGpuGenPipeline()
{
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
    if (m_GenPipeline)       { vkDestroyPipeline(VulkanHW.m_Device, m_GenPipeline, nullptr); m_GenPipeline = VK_NULL_HANDLE; }
    if (m_GenPipelineLayout) { vkDestroyPipelineLayout(VulkanHW.m_Device, m_GenPipelineLayout, nullptr); m_GenPipelineLayout = VK_NULL_HANDLE; }
    if (m_GenDescPool)       { vkDestroyDescriptorPool(VulkanHW.m_Device, m_GenDescPool, nullptr); m_GenDescPool = VK_NULL_HANDLE; m_GenDescSet = VK_NULL_HANDLE; m_CasterDescSet = VK_NULL_HANDLE; }
    if (m_GenDescLayout)     { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, m_GenDescLayout, nullptr); m_GenDescLayout = VK_NULL_HANDLE; }
}

void CDetailManager::UpdateGenDescriptors()
{
    if (m_GenDescSet == VK_NULL_HANDLE) return;
    if (!m_HeightmapView || !m_DetailSampler || !m_SlotDataSSBO || !m_ObjInfoSSBO ||
        !m_VisibleSSBO || !m_AtomicCounters || !m_IndirectCmdBuf || !m_GenUBO ||
        !m_DummyHZBView || !m_DummyTrailView) return;

    VkDescriptorImageInfo hm{};   hm.sampler = m_HeightmapSampler;  hm.imageView = m_HeightmapView;  hm.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    // Binding 6 = real Hi-Z pyramid (GENERAL, nearest) when built; else 1×1
    // white dummy (depth 1.0 → never culls). HZB stays GENERAL all frame.
    VkDescriptorImageInfo hzb{};
    if (m_HZBView != VK_NULL_HANDLE) {
        hzb.sampler = m_HZBSampler;   hzb.imageView = m_HZBView;       hzb.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    } else {
        hzb.sampler = m_DetailSampler; hzb.imageView = m_DummyHZBView; hzb.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
    VkDescriptorImageInfo trl{};  trl.sampler= m_DetailSampler;     trl.imageView= m_DummyTrailView; trl.imageLayout= VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorBufferInfo bi[6]{};
    bi[0] = { m_SlotDataSSBO->GetHandle(),  0, VK_WHOLE_SIZE };
    bi[1] = { m_ObjInfoSSBO->GetHandle(),   0, VK_WHOLE_SIZE };
    bi[2] = { m_VisibleSSBO->GetHandle(),   0, VK_WHOLE_SIZE };
    bi[3] = { m_AtomicCounters->GetHandle(),0, VK_WHOLE_SIZE };
    bi[4] = { m_IndirectCmdBuf->GetHandle(),0, VK_WHOLE_SIZE };
    bi[5] = { m_GenUBO->GetHandle(),        0, VK_WHOLE_SIZE };
    // Caster set variant: output/atomics/indirect swapped for the caster buffers.
    VkDescriptorBufferInfo ci[3]{};
    const bool haveCaster = m_CasterDescSet != VK_NULL_HANDLE
        && m_CasterSSBO && m_CasterAtomic && m_CasterIndirectBuf;
    if (haveCaster) {
        ci[0] = { m_CasterSSBO->GetHandle(),        0, VK_WHOLE_SIZE };
        ci[1] = { m_CasterAtomic->GetHandle(),      0, VK_WHOLE_SIZE };
        ci[2] = { m_CasterIndirectBuf->GetHandle(), 0, VK_WHOLE_SIZE };
    }

    VkWriteDescriptorSet w[18]{};
    u32 nW = 0;
    auto setBuf = [&](VkDescriptorSet ds, u32 binding, VkDescriptorType t, const VkDescriptorBufferInfo* info) {
        auto& x = w[nW++];
        x.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        x.dstSet = ds; x.dstBinding = binding; x.descriptorCount = 1;
        x.descriptorType = t; x.pBufferInfo = info;
    };
    auto setImg = [&](VkDescriptorSet ds, u32 binding, const VkDescriptorImageInfo* info) {
        auto& x = w[nW++];
        x.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        x.dstSet = ds; x.dstBinding = binding; x.descriptorCount = 1;
        x.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; x.pImageInfo = info;
    };
    auto writeSet = [&](VkDescriptorSet ds, bool caster) {
        setImg(ds, 0, &hm);
        setBuf(ds, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &bi[0]);
        setBuf(ds, 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, &bi[1]);
        setBuf(ds, 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, caster ? &ci[0] : &bi[2]);
        setBuf(ds, 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, caster ? &ci[1] : &bi[3]);
        setBuf(ds, 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, caster ? &ci[2] : &bi[4]);
        setImg(ds, 6, &hzb);
        setBuf(ds, 7, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, &bi[5]);
        setImg(ds, 8, &trl);
    };
    writeSet(m_GenDescSet, false);
    if (haveCaster) writeSet(m_CasterDescSet, true);

    vkUpdateDescriptorSets(VulkanHW.m_Device, nW, w, 0, nullptr);
}

// ============================================================================
// Graphics pipeline. Vertex layout: binding 0 = per-vertex (24 B = 3 attrs),
// binding 1 = per-instance (64 B = 4 vec4 attrs, INSTANCE rate). Push 208 B.
// 1 descriptor set with 1 sampler (diffuse).
// ============================================================================
void CDetailManager::CreateGfxPipeline()
{
    if (!g_ShaderManager) { Msg("![VK Grass] g_ShaderManager null — gfx pipeline disabled"); return; }

    // Descriptor layout: binding 0 = per-type diffuse (fragment), binding 1 =
    // SSFX wind flow map s_waves (vertex — sampled to drive the wind).
    VkDescriptorSetLayoutBinding b[2]{};
    b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b[0].descriptorCount = 1; b[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    b[1].binding = 1; b[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    b[1].descriptorCount = 1; b[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    VkDescriptorSetLayoutCreateInfo lci{};
    lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    lci.bindingCount = 2; lci.pBindings = b;
    vkCreateDescriptorSetLayout(VulkanHW.m_Device, &lci, nullptr, &m_GfxDescLayout);

    // Pool: one set per object type, 2 image samplers each (diffuse + flow map).
    const u32 nSets = _max(u32(objects.size()), 1u);
    VkDescriptorPoolSize ps{};
    ps.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; ps.descriptorCount = nSets * 2;
    VkDescriptorPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.maxSets = nSets; pci.poolSizeCount = 1; pci.pPoolSizes = &ps;
    vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &m_GfxDescPool);

    m_GfxDescSets.resize(objects.size(), VK_NULL_HANDLE);
    for (size_t i = 0; i < objects.size(); ++i) {
        VkDescriptorSetAllocateInfo dai{};
        dai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool     = m_GfxDescPool;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts        = &m_GfxDescLayout;
        vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, &m_GfxDescSets[i]);

        // binding 0 = per-type diffuse (or dummy white); binding 1 = SSFX flow map
        // (shared; dummy white when absent → flat wind, never crashes).
        VkDescriptorImageInfo ii[2]{};
        ii[0].sampler     = m_DetailSampler;
        ii[0].imageView   = (i < m_DetailTextures.size() && m_DetailTextures[i])
                            ? m_DetailTextures[i]->GetView()
                            : m_DummyHZBView;
        ii[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        ii[1].sampler     = m_DetailSampler;
        ii[1].imageView   = m_WaveTex ? m_WaveTex->GetView() : m_DummyHZBView;
        ii[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet w[2]{};
        for (u32 k = 0; k < 2; ++k) {
            w[k].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[k].dstSet          = m_GfxDescSets[i];
            w[k].dstBinding      = k;
            w[k].descriptorCount = 1;
            w[k].descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            w[k].pImageInfo      = &ii[k];
        }
        vkUpdateDescriptorSets(VulkanHW.m_Device, 2, w, 0, nullptr);
    }

    // Pipeline layout: set0 = per-type diffuse, set1 = shared env lighting
    // (sun_vp + sun shadow map — vk_env_light), + push (VS + FS).
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcr.size = sizeof(DetailGfxPushConstants);
    DM_CheckPushSize("gfx", pcr.size);
    VkDescriptorSetLayout gfxSetLayouts[2] = { m_GfxDescLayout, VK::EnvLight::GetSetLayout() };
    VkPipelineLayoutCreateInfo plci{};
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 2;
    plci.pSetLayouts            = gfxSetLayouts;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (gfxSetLayouts[1] == VK_NULL_HANDLE) { Msg("![VK Grass] EnvLight layout not ready"); return; }
    vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &m_GfxPipelineLayout);

    // Shaders
    VkShaderModule vs = g_ShaderManager->Load("detail.vert.spv");
    VkShaderModule fs = g_ShaderManager->Load("detail.frag.spv");
    if (vs == VK_NULL_HANDLE || fs == VK_NULL_HANDLE) {
        Msg("![VK Grass] detail.{vert,frag}.spv load failed"); return;
    }

    // Vertex bindings: binding 0 per-vertex (24 B), binding 1 per-instance (64 B).
    VkVertexInputBindingDescription vibd[2]{};
    vibd[0] = { 0, sizeof(VK::CDetail::Vertex), VK_VERTEX_INPUT_RATE_VERTEX   };
    vibd[1] = { 1, sizeof(DetailInstance),       VK_VERTEX_INPUT_RATE_INSTANCE };

    // Attribute layout matches detail_vs.glsl locations 0..6.
    VkVertexInputAttributeDescription via[7]{};
    via[0] = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0  };                      // aPos
    via[1] = { 1, 0, VK_FORMAT_R32G32_SFLOAT,    12 };                      // aUV
    via[2] = { 2, 0, VK_FORMAT_R32_SFLOAT,       20 };                      // aHeight
    via[3] = { 3, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0  };                   // aInstRow0
    via[4] = { 4, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 16 };                   // aInstRow1
    via[5] = { 5, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 32 };                   // aInstRow2
    via[6] = { 6, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 48 };                   // aInstColor

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount   = 2;
    vi.pVertexBindingDescriptions      = vibd;
    vi.vertexAttributeDescriptionCount = 7;
    vi.pVertexAttributeDescriptions    = via;

    VkPipelineShaderStageCreateInfo ss[2]{};
    ss[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ss[0].stage = VK_SHADER_STAGE_VERTEX_BIT; ss[0].module = vs; ss[0].pName = "main";
    ss[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ss[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; ss[1].module = fs; ss[1].pName = "main";

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1; vp.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode             = VK_POLYGON_MODE_FILL;
    rs.cullMode                = VK_CULL_MODE_NONE;            // grass is double-sided
    rs.frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth               = 1.0f;
    rs.depthBiasEnable         = VK_TRUE;
    rs.depthBiasConstantFactor = -2.0f;
    rs.depthBiasSlopeFactor    = -1.0f;

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
    cb.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments    = &ba;

    VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynState{};
    dynState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynState.dynamicStateCount = 2;
    dynState.pDynamicStates    = dyn;

    VkFormat colorFmt = VK::SceneColor::Format();
    VkPipelineRenderingCreateInfo prci{};
    prci.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    prci.colorAttachmentCount    = 1;
    prci.pColorAttachmentFormats = &colorFmt;
    prci.depthAttachmentFormat   = Swapchain.m_DepthFormat;

    VkGraphicsPipelineCreateInfo pi{};
    pi.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pi.pNext               = &prci;
    pi.stageCount          = 2;
    pi.pStages             = ss;
    pi.pVertexInputState   = &vi;
    pi.pInputAssemblyState = &ia;
    pi.pViewportState      = &vp;
    pi.pRasterizationState = &rs;
    pi.pMultisampleState   = &ms;
    pi.pDepthStencilState  = &ds;
    pi.pColorBlendState    = &cb;
    pi.pDynamicState       = &dynState;
    pi.layout              = m_GfxPipelineLayout;
    if (vkCreateGraphicsPipelines(VulkanHW.m_Device, VK::PipelineCache::GetCacheObject(), 1, &pi, nullptr, &m_GfxPipeline) != VK_SUCCESS) {
        Msg("![VK Grass] gfx pipeline create failed"); return;
    }

    Msg("[VK Grass] Gfx pipeline OK (1 set, 7 attrs, 208 B push)");
}

void CDetailManager::DestroyGfxPipeline()
{
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
    DestroyMotionPipeline();   // built on top of m_GfxDescLayout — free it first
    if (m_GfxPipeline)       { vkDestroyPipeline(VulkanHW.m_Device, m_GfxPipeline, nullptr); m_GfxPipeline = VK_NULL_HANDLE; }
    if (m_GfxPipelineLayout) { vkDestroyPipelineLayout(VulkanHW.m_Device, m_GfxPipelineLayout, nullptr); m_GfxPipelineLayout = VK_NULL_HANDLE; }
    if (m_GfxDescPool)       { vkDestroyDescriptorPool(VulkanHW.m_Device, m_GfxDescPool, nullptr); m_GfxDescPool = VK_NULL_HANDLE; m_GfxDescSets.clear(); }
    if (m_GfxDescLayout)     { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, m_GfxDescLayout, nullptr); m_GfxDescLayout = VK_NULL_HANDLE; }
}

// Grass wind-sway MV overlay pipeline. Reuses the gfx descriptor layout (set0 =
// diffuse @0 + s_waves @1) and vertex input, but outputs RG16F motion, tests depth
// LEQUAL WITHOUT writing, and carries the SAME depth bias as the forward grass draw
// so its clip depth bit-matches what that pass already stored (LEQUAL passes on
// equal). Push = cur/prev VP + cur/prev wind (detail_motion.vert reprojects both).
void CDetailManager::CreateMotionPipeline()
{
    if (m_MotionPipeline != VK_NULL_HANDLE) return;               // already built
    if (!g_ShaderManager || m_GfxDescLayout == VK_NULL_HANDLE) return;

    VkShaderModule vs = g_ShaderManager->Load("detail_motion.vert.spv");
    VkShaderModule fs = g_ShaderManager->Load("detail_motion.frag.spv");
    if (vs == VK_NULL_HANDLE || fs == VK_NULL_HANDLE) {
        Msg("![VK Grass] detail_motion.{vert,frag}.spv load failed — grass MV disabled");
        return;
    }

    // Layout: set0 = gfx descriptor layout (diffuse + s_waves), push = MV (VS only).
    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pcr.size       = sizeof(DetailMotionPushConstants);
    VkPipelineLayoutCreateInfo plci{};
    plci.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &m_GfxDescLayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &m_MotionPipelineLayout) != VK_SUCCESS) {
        Msg("![VK Grass] MV pipeline layout create failed"); return;
    }

    // Same vertex input as the forward gfx pipeline.
    VkVertexInputBindingDescription vibd[2]{};
    vibd[0] = { 0, sizeof(VK::CDetail::Vertex), VK_VERTEX_INPUT_RATE_VERTEX   };
    vibd[1] = { 1, sizeof(DetailInstance),       VK_VERTEX_INPUT_RATE_INSTANCE };
    VkVertexInputAttributeDescription via[7]{};
    via[0] = { 0, 0, VK_FORMAT_R32G32B32_SFLOAT,    0  };
    via[1] = { 1, 0, VK_FORMAT_R32G32_SFLOAT,       12 };
    via[2] = { 2, 0, VK_FORMAT_R32_SFLOAT,          20 };
    via[3] = { 3, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 0  };
    via[4] = { 4, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 16 };
    via[5] = { 5, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 32 };
    via[6] = { 6, 1, VK_FORMAT_R32G32B32A32_SFLOAT, 48 };
    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount   = 2; vi.pVertexBindingDescriptions   = vibd;
    vi.vertexAttributeDescriptionCount = 7; vi.pVertexAttributeDescriptions = via;

    VkPipelineShaderStageCreateInfo ss[2]{};
    ss[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ss[0].stage = VK_SHADER_STAGE_VERTEX_BIT;   ss[0].module = vs; ss[0].pName = "main";
    ss[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ss[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; ss[1].module = fs; ss[1].pName = "main";

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1; vp.scissorCount = 1;

    // Cull NONE + the SAME depth bias as the forward grass draw so depth bit-matches.
    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode             = VK_POLYGON_MODE_FILL;
    rs.cullMode                = VK_CULL_MODE_NONE;
    rs.frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth               = 1.0f;
    rs.depthBiasEnable         = VK_TRUE;
    rs.depthBiasConstantFactor = -2.0f;
    rs.depthBiasSlopeFactor    = -1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo ds{};
    ds.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    ds.depthTestEnable  = VK_TRUE;
    ds.depthWriteEnable = VK_FALSE;                  // scene depth already owns the surface
    ds.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendAttachmentState ba{};
    ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT;   // RG16F motion
    ba.blendEnable    = VK_FALSE;
    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1; cb.pAttachments = &ba;

    VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dynState{};
    dynState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynState.dynamicStateCount = 2; dynState.pDynamicStates = dyn;

    VkFormat mvFmt = VK::MotionVec::Format();
    VkPipelineRenderingCreateInfo prci{};
    prci.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    prci.colorAttachmentCount    = 1;
    prci.pColorAttachmentFormats = &mvFmt;
    prci.depthAttachmentFormat   = Swapchain.m_DepthFormat;

    VkGraphicsPipelineCreateInfo pi{};
    pi.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pi.pNext               = &prci;
    pi.stageCount          = 2;     pi.pStages             = ss;
    pi.pVertexInputState   = &vi;   pi.pInputAssemblyState = &ia;
    pi.pViewportState      = &vp;   pi.pRasterizationState = &rs;
    pi.pMultisampleState   = &ms;   pi.pDepthStencilState  = &ds;
    pi.pColorBlendState    = &cb;   pi.pDynamicState       = &dynState;
    pi.layout              = m_MotionPipelineLayout;
    if (vkCreateGraphicsPipelines(VulkanHW.m_Device, VK::PipelineCache::GetCacheObject(), 1, &pi, nullptr, &m_MotionPipeline) != VK_SUCCESS) {
        Msg("![VK Grass] MV pipeline create failed");
        vkDestroyPipelineLayout(VulkanHW.m_Device, m_MotionPipelineLayout, nullptr);
        m_MotionPipelineLayout = VK_NULL_HANDLE;
        return;
    }
    Msg("[VK Grass] MV overlay pipeline OK (wind-sway motion vectors)");
}

void CDetailManager::DestroyMotionPipeline()
{
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
    if (m_MotionPipeline)       { vkDestroyPipeline(VulkanHW.m_Device, m_MotionPipeline, nullptr); m_MotionPipeline = VK_NULL_HANDLE; }
    if (m_MotionPipelineLayout) { vkDestroyPipelineLayout(VulkanHW.m_Device, m_MotionPipelineLayout, nullptr); m_MotionPipelineLayout = VK_NULL_HANDLE; }
}

}  // namespace VK
