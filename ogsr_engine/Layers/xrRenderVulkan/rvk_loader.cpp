// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan — level loading for OGSR.
//
// Stripped from monolith's rvk_loader.cpp down to MVP scope: shaders, geometry
// buffers, sliding-window LOD items, and visuals (the four chunks that have
// to land for the engine to consider the level loaded). Subsystems we have
// not ported yet — sectors/portals, lights, wallmarks, HOM, ShadowManager,
// RTGI, 3D fluid, details — are left as no-ops; their entry points exist so
// the engine's call order is preserved, they just do nothing.

#include "stdafx.h"
#include "CRender_Vulkan.h"
#include "vk_Visual.h"
#include "vk_ModelPool.h"
#include "vk_buffer.h"
#include "vk_buffer_pool.h"
#include "vk_d3d_compat.h"
#include "vk_shader.h"          // g_VulkanShaderManager
#include "vk_DetailManager.h"   // VK::CDetailManager (grass)
#include "vk_TreeManager.h"     // VK::CTreeManager (trees)
#include "vk_LODManager.h"      // VK::CLODManager (LOD imposters)
#include "vk_shadow_gpu.h"      // VK::ShadowGPU (GPU-driven sun shadow casters)
#include "vk_world_gpu.h"       // VK::WorldGPU (GPU-driven world forward pass)
#include "vk_vsm.h"             // VK::VSM::InvalidateCache (world-anchored page cache vs level change)
#include "vk_world_material.h"  // VK::WorldMaterialCache::SetLevelTag (per-level lightmap namespacing)
#include "HW_Vulkan.h"          // VulkanHW (vkDeviceWaitIdle in level_Unload)
#include "vk_command_buffer.h"  // CommandManager.FlushUploadsAndWait (drain async uploads)

#include "../../xr_3da/x_ray.h"             // pApp
#include "../../xr_3da/xrLevel.h"           // fsL_SHADERS / fsL_VB / fsL_IB / fsL_VISUALS / fsL_SWIS
#include "../../xr_3da/IGame_Persistent.h"  // g_pGamePersistent
#include "../../xr_3da/IGame_Level.h"       // g_pGameLevel
#include "../../xr_3da/fmesh.h"             // ogf_header, OGF_HEADER, MT_*
#include "../../xrCore/stream_reader.h"

// OGSR has no dedicated-server build path here; geometry loading runs
// unconditionally. (Monolith gates these on `g_dedicated_server`.)

// ============================================================================
// Level Loading
// ============================================================================
void CRender::level_Load(IReader* fs)
{
    R_ASSERT(0 != g_pGameLevel);
    R_ASSERT(!b_loaded);

    Msg("[Vulkan] CRender::level_Load() started");
    pApp->LoadBegin();

    // Namespace the material cache's lightmap keys by THIS level ($level$ is
    // already mounted here): lmap names repeat across levels, and the cache
    // survives level changes — without the tag the new level binds the previous
    // level's lightmaps ("baked" light/dark patches that ignore the sun).
    {
        string_path lp;
        FS.update_path(lp, "$level$", "");
        VK::WorldMaterialCache::SetLevelTag(lp);
    }

    IReader* chunk;

    // ----- Shaders -----------------------------------------------------------
    g_pGamePersistent->LoadTitle("");
    {
        chunk = fs->open_chunk(fsL_SHADERS);
        R_ASSERT2(chunk, "Level doesn't built correctly - no shaders chunk");

        u32 count = chunk->r_u32();
        Msg("[Vulkan] Loading %u level shaders", count);
        Shaders.resize(count, nullptr);

        for (u32 i = 0; i < count; i++)
        {
            LPCSTR n = LPCSTR(chunk->pointer());
            chunk->skip_stringZ();

            if (!g_VulkanShaderManager) continue;

            if (0 == n[0]) {
                Shaders[i] = g_VulkanShaderManager->GetDefaultShader();
                continue;
            }

            string512 n_sh, n_tlist;
            xr_strcpy(n_sh, n);
            n_tlist[0] = 0;
            if (LPSTR delim = strchr(n_sh, '/')) {
                *delim = 0;
                xr_strcpy(n_tlist, delim + 1);
            }

            Shaders[i] = g_VulkanShaderManager->CreateShader(n_sh, n_tlist);
            if (!Shaders[i])
                Shaders[i] = g_VulkanShaderManager->GetDefaultShader();
        }
        chunk->close();

        if (!g_VulkanShaderManager)
            Msg("![Vulkan] g_VulkanShaderManager not initialized — visuals will use nullptr shaders");
        Msg("[Vulkan] Loaded %u shaders", (u32)Shaders.size());
    }

    // ----- Geometry buffers + LOD sliding-window items ----------------------
    g_pGamePersistent->LoadTitle("");
    if (!VK::g_BufferPool)
        VK::g_BufferPool = xr_new<VK::CBufferPool>();
    {
        CStreamReader* geom = FS.rs_open("$level$", "level.geom");
        R_ASSERT2(geom, "level.geom not found");
        LoadBuffers(geom, FALSE);
        LoadSWIs(geom);
        FS.r_close(geom);

        geom = FS.rs_open("$level$", "level.geomx");
        R_ASSERT2(geom, "level.geomx not found");
        LoadBuffers(geom, TRUE);
        FS.r_close(geom);
    }

    g_pGamePersistent->LoadTitle("");
    {
        chunk = fs->open_chunk(fsL_VISUALS);
        R_ASSERT2(chunk, "Level has no visuals");
        LoadVisuals(chunk);
        chunk->close();
    }

    // ----- Subsystem hooks (currently no-op stubs) --------------------------
    g_pGamePersistent->LoadTitle("");
    LoadSectors(fs);
    LoadLights(fs);
    Load3DFluid();

    // ----- Grass / detail objects ------------------------------------------
    // Session A scope: open level.details, bake heightmap from collision tris,
    // upload slot+obj SSBOs. No rendering yet — Sessions B/C add the compute
    // generator + render pipeline. Safe to skip on levels with no level.details.
    if (!Details) Details = xr_new<VK::CDetailManager>();
    Details->Load();

    // ----- Trees ------------------------------------------------------------
    // GPU-driven indirect path: walks Visuals[] for MT_TREE_ST/PM, builds
    // metadata + transforms SSBOs. Session A only — Session B adds compute
    // cull + draw. Must run AFTER LoadVisuals (Visuals[] populated).
    if (!Trees) Trees = xr_new<VK::CTreeManager>();
    Trees->Build();

    // ----- GPU-driven sun shadow casters -----------------------------------
    // Extract opaque static casters (shared VB/IB pools, world-space) into a
    // meta SSBO + indirect/count buffers for per-cascade compute cull. Must run
    // AFTER LoadVisuals (Visuals[] populated). Alpha-tested casters stay on the
    // CPU FlushDepth path. See vk_pass_shadow / vk_shadow_gpu.
    VK::ShadowGPU::Build();

    // ----- GPU-driven world forward pass -----------------------------------
    // Extract opaque/AT static world meshes grouped by material for compute-cull
    // + indirect draw (moves per-object CPU submission off the CPU so detail-rich
    // levels stop hitting the draw-call wall). See vk_pass_world / vk_world_gpu.
    VK::WorldGPU::Build();

    // ----- LOD imposters ----------------------------------------------------
    // FLOD billboard facets (MT_LOD) for distant foliage density. Must run after
    // LoadVisuals so the vkFLOD objects exist.
    if (!LODs) LODs = xr_new<VK::CLODManager>();
    LODs->Build();

    pApp->LoadEnd();

    // Drain all async vertex/index uploads staged during load before the first
    // frame renders — one wait instead of a stall per buffer, and frees the ring.
    CommandManager.FlushUploadsAndWait();

    b_loaded = TRUE;
    Msg("[Vulkan] level_Load() complete: %u visuals", (u32)Visuals.size());
}

// ============================================================================
// Level Unloading
// ============================================================================
void CRender::level_Unload()
{
    if (0 == g_pGameLevel) return;
    if (!b_loaded) return;

    Msg("[Vulkan] CRender::level_Unload()");

    // Wait for in-flight GPU work — VMA can't free buffers the GPU is reading.
    if (VulkanHW.m_Device != VK_NULL_HANDLE)
        vkDeviceWaitIdle(VulkanHW.m_Device);

    // Visuals
    for (IRenderVisual* iv : Visuals)
    {
        if (!iv) continue;
        vkRender_Visual* v = static_cast<vkRender_Visual*>(iv);
        v->Release();
        xr_delete(v);
    }
    Visuals.clear();

    // Sliding window items
    for (FSlideWindowItem& swi : SWIs)
        if (swi.sw) { xr_free(swi.sw); swi.sw = nullptr; }
    SWIs.clear();

    Shaders.clear();

    auto release_bufs = [](xr_vector<VK::CVulkanBuffer*>& v)
    {
        for (VK::CVulkanBuffer* b : v)
            if (b) { b->Destroy(); xr_delete(b); }
        v.clear();
    };
    release_bufs(nVB); release_bufs(xVB);
    release_bufs(nIB); release_bufs(xIB);
    nVB_Strides.clear();
    xVB_Strides.clear();

    if (Models) Models->ClearPool(true);

    if (Details) {
        Details->Unload();
        xr_delete(Details);
    }

    if (Trees) {
        Trees->Destroy();
        xr_delete(Trees);
    }

    // GPU-driven shadow casters borrow the shared VB/IB pool handles — release
    // before those pools (below). Device is already idle (vkDeviceWaitIdle above).
    VK::ShadowGPU::Destroy();
    VK::WorldGPU::Destroy();

    // VSM's toroidal page cache is WORLD-anchored — the next level reuses the
    // same coordinates, so resident pages would keep serving THIS level's depth
    // (stale light/dark page squares on the new level's walls). Mark it empty;
    // the first frame re-renders the atlas from the new casters.
    VK::VSM::InvalidateCache();

    if (LODs) {
        LODs->Destroy();
        xr_delete(LODs);
    }

    if (VK::g_BufferPool) {
        xr_delete(VK::g_BufferPool);
        VK::g_BufferPool = nullptr;
    }

    b_loaded = FALSE;
    Msg("[Vulkan] level_Unload() complete");
}

// ============================================================================
// LoadBuffers — VB/IB pools from level.geom / level.geomx.
// ============================================================================
void CRender::LoadBuffers(CStreamReader* base_fs, BOOL _alternative)
{
    R_ASSERT2(base_fs, "Could not load geometry - file not found");

    xr_vector<VK::CVulkanBuffer*>& _VB = _alternative ? xVB : nVB;
    xr_vector<VK::CVulkanBuffer*>& _IB = _alternative ? xIB : nIB;
    xr_vector<u32>& _Strides           = _alternative ? xVB_Strides : nVB_Strides;

    Msg("[Vulkan] Loading %s geometry buffers", _alternative ? "extended" : "normal");

    // ----- Vertex buffers ---------------------------------------------------
    {
        CStreamReader* fs = base_fs->open_chunk(fsL_VB);
        R_ASSERT2(fs, "fsL_VB chunk not found");

        const u32 count = fs->r_u32();
        _VB.resize(count);
        _Strides.resize(count);

        const u32 declBytes = (MAXD3DDECLLENGTH + 1) * sizeof(D3DVERTEXELEMENT9);
        D3DVERTEXELEMENT9* dcl = (D3DVERTEXELEMENT9*)_alloca(declBytes);

        for (u32 i = 0; i < count; i++)
        {
            fs->r(dcl, declBytes);
            fs->advance(-(int)declBytes);

            const u32 dcl_len = VK_GetDeclLength(dcl) + 1;
            fs->advance(dcl_len * sizeof(D3DVERTEXELEMENT9));

            const u32 vCount = fs->r_u32();
            const u32 vSize  = VK_GetDeclVertexSize(dcl, 0);
            _Strides[i] = vSize;

            // TEXCOORD0 offset — buffer pool uses it to keep stride-32 lightmapped
            // (tcOff=24) and vertex-lit (tcOff=28) layouts on separate pipelines.
            u32 tcOffset = 24;
            for (u32 e = 0; e + 1 < dcl_len; e++) {
                if (dcl[e].Usage == D3DDECLUSAGE_TEXCOORD && dcl[e].UsageIndex == 0) {
                    tcOffset = dcl[e].Offset;
                    break;
                }
            }

            BYTE* pData = xr_alloc<BYTE>(vCount * vSize);
            fs->r(pData, vCount * vSize);

            VkBufferUsageFlags vbUsage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                                         VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT;   // CTreeManager meshlet readback (positions)
            _VB[i] = xr_new<VK::CVulkanBuffer>();
            _VB[i]->Create(vCount * vSize, vbUsage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
            _VB[i]->Upload(pData, vCount * vSize);

            // Only the normal pool publishes to g_BufferPool — extended (geomx)
            // shares VB slot ids and would overwrite the registrations.
            if (VK::g_BufferPool && !_alternative)
                VK::g_BufferPool->RegisterVertexBuffer(i, _VB[i], vSize, tcOffset);

            xr_free(pData);
        }
        fs->close();
    }

    // ----- Index buffers ----------------------------------------------------
    {
        CStreamReader* fs = base_fs->open_chunk(fsL_IB);
        R_ASSERT2(fs, "fsL_IB chunk not found");

        const u32 count = fs->r_u32();
        _IB.resize(count);

        for (u32 i = 0; i < count; i++)
        {
            const u32 iCount = fs->r_u32();
            const u32 iSize  = iCount * sizeof(u16);

            BYTE* pData = xr_alloc<BYTE>(iSize);
            fs->r(pData, iSize);

            VkBufferUsageFlags ibUsage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                         VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT;   // CTreeManager meshlet readback (indices)
            _IB[i] = xr_new<VK::CVulkanBuffer>();
            _IB[i]->Create(iSize, ibUsage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
            _IB[i]->Upload(pData, iSize);

            if (VK::g_BufferPool && !_alternative)
                VK::g_BufferPool->RegisterIndexBuffer(i, _IB[i], VK_INDEX_TYPE_UINT16);

            xr_free(pData);
        }
        fs->close();
    }
}

// ============================================================================
// LoadVisuals — fsL_VISUALS chunk; one vkRender_Visual per OGF entry.
// ============================================================================
void CRender::LoadVisuals(IReader* fs)
{
    Msg("[Vulkan] Loading visuals...");

    IReader* chunk = nullptr;
    u32 index = 0;
    ogf_header H;

    while ((chunk = fs->open_chunk(index)) != nullptr)
    {
        chunk->r_chunk_safe(OGF_HEADER, &H, sizeof(H));

        vkRender_Visual* V = vkVisual_Create(H.type);
        if (V) {
            V->Load(nullptr, chunk, 0);
            Visuals.push_back(V);
        } else {
            Msg("![Vulkan] Visual #%u: unknown type %u — using dummy", index, H.type);
            Visuals.push_back(vkVisual_CreateDummy());
        }

        chunk->close();
        index++;
    }

    Msg("[Vulkan] Loaded %u visuals", index);
}

// ============================================================================
// LoadSWIs — sliding-window items for progressive (LOD) meshes.
// ============================================================================
void CRender::LoadSWIs(CStreamReader* base_fs)
{
    if (!base_fs->find_chunk(fsL_SWIS))
        return;

    CStreamReader* fs = base_fs->open_chunk(fsL_SWIS);
    const u32 item_count = fs->r_u32();
    Msg("[Vulkan] Loading %u sliding window items", item_count);

    for (FSlideWindowItem& swi : SWIs)
        if (swi.sw) { xr_free(swi.sw); swi.sw = nullptr; }
    SWIs.clear();

    SWIs.resize(item_count);
    for (u32 c = 0; c < item_count; c++)
    {
        FSlideWindowItem& swi = SWIs[c];
        swi.reserved[0] = fs->r_u32();
        swi.reserved[1] = fs->r_u32();
        swi.reserved[2] = fs->r_u32();
        swi.reserved[3] = fs->r_u32();
        swi.count       = fs->r_u32();
        swi.sw          = xr_alloc<FSlideWindow>(swi.count);
        fs->r(swi.sw, sizeof(FSlideWindow) * swi.count);
    }

    fs->close();
}

// Resolve a pooled sliding-window item (OGF_SWICONTAINER references this by id).
// Used by vkFTreeVisual_PM / vkFProgressive for per-LOD index ranges.
FSlideWindowItem* CRender::getSWI(int id)
{
    return &SWIs.at(id);
}

// ============================================================================
// Stubs for subsystems we have not ported yet. Engine call order is preserved
// so when the real impls land they just replace the body.
// ============================================================================
void CRender::LoadSectors(IReader* /*fs*/) {}
void CRender::LoadLights (IReader* /*fs*/) {}
void CRender::Load3DFluid()                {}
