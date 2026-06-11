// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
// Licensed under the same terms as X-Ray Engine (see root License.txt)

#include "stdafx.h"
#include "vk_Visual.h"
#include "CRender_Vulkan.h"  // RImplementation (Shaders, model_*)
#include "vk_buffer_pool.h"
#include "vk_shader.h"
#include "vk_material.h"
#include "vk_d3d_compat.h"   // VK_GetFVFVertexSize (FVF parsing)
#include "vk_UIPipeline.h"   // g_VkUI_FrameCmd
#include "vk_render_queue.h" // RenderQueue / DrawItem (phase-3 submission)
#include "vk_world_material.h"  // WorldMaterialCache (phase-5)

// ============================================================================
// VK_Render_Mesh implementation
// ============================================================================
VK_Render_Mesh::~VK_Render_Mesh()
{
    Destroy();
}

void VK_Render_Mesh::Destroy()
{
    // Only delete buffers if we own them (not a shared Copy)
    if (bOwnsBuffers)
    {
        if (p_rm_Vertices)
        {
            xr_delete(p_rm_Vertices);
        }

        if (p_rm_Indices)
        {
            xr_delete(p_rm_Indices);
        }

        if (m_fast)
        {
            xr_delete(m_fast);
        }
    }

    p_rm_Vertices = nullptr;
    p_rm_Indices = nullptr;
    m_fast = nullptr;
    vBase = vCount = vStride = 0;
    tcOffset = 24;
    iBase = iCount = 0;
    dwPrimitives = 0;
}

// ============================================================================
// vkRender_Visual implementation
// ============================================================================
vkRender_Visual::vkRender_Visual()
{
    Type = 0;
    skinning = -1;
}

vkRender_Visual::~vkRender_Visual()
{
}

void vkRender_Visual::Load(const char* name, IReader* data, u32 flags)
{
    dbg_name = name;

    // Load header (required)
    LoadHeader(data);

    // Load texture/shader info (optional)
    LoadTexture(data);
}

void vkRender_Visual::Release()
{
    // Base class has nothing to release
}

void vkRender_Visual::Copy(vkRender_Visual* from)
{
    Type = from->Type;
    vis = from->vis;
    skinning = from->skinning;
    dbg_name = from->dbg_name;
    m_pMaterial = from->m_pMaterial;
    m_fAlphaRef = from->m_fAlphaRef;
    // Pool instances are built via Copy(); without this the instance's diffuse
    // descriptor stays null and it renders white (the base had it from LoadTexture).
    // (Same latent class as the Update_Callback ctor bug.) WorldMaterial* is a shared
    // cache handle — safe to share.
    m_pWorldMaterial = from->m_pWorldMaterial;
}

void vkRender_Visual::Render(float LOD)
{
    // Base class does nothing
}

void vkRender_Visual::LoadHeader(IReader* data)
{
    // Read OGF header
    ogf_header H;
    if (data->find_chunk(OGF_HEADER))
    {
        data->r(&H, sizeof(H));

        Type = H.type;
        shader_id = H.shader_id;  // Phase 2.34: Store shader ID for rendering

        // Set visibility data (copy from OGF header structs to engine structs)
        vis.box.set(H.bb.min, H.bb.max);
        vis.sphere.set(H.bs.c, H.bs.r);
    }
    else
    {
        Msg("![Vulkan] Missing OGF_HEADER in %s", dbg_name.c_str());

        // Initialize to safe defaults
        Type = 0;
        shader_id = 0;  // Will use default shader (index 0) or fallback in Render()
        vis.box.set(Fvector().set(0,0,0), Fvector().set(0,0,0));
        vis.sphere.set(Fvector().set(0,0,0), 0.0f);
    }
}

void vkRender_Visual::LoadTexture(IReader* data)
{
    // Per-model diffuse from the OGF_TEXTURE chunk (dynamic models: NPCs, weapons,
    // items). Level statics have no OGF_TEXTURE — they resolve via shader_id below.
    string256 ogf_diffuse;
    ogf_diffuse[0] = 0;

    // Read texture chunk (optional - present in standalone OGF models)
    if (data->find_chunk(OGF_TEXTURE))
    {
        string256 texture_name;
        string256 shader_name;

        data->r_stringZ(texture_name, sizeof(texture_name));
        data->r_stringZ(shader_name, sizeof(shader_name));

        xr_strcpy(ogf_diffuse, texture_name);

        // Create material from texture name (loads DDS, creates descriptor set)
        if (g_MaterialManager && texture_name[0])
        {
            m_pMaterial = g_MaterialManager->CreateMaterial(texture_name);
        }

        // Detect alpha-ref shader from OGF_TEXTURE shader name
        if (shader_name[0])
        {
            xr_string sn_lower = shader_name;
            std::transform(sn_lower.begin(), sn_lower.end(), sn_lower.begin(), ::tolower);
            if (sn_lower.find("aref") != xr_string::npos ||
                sn_lower.find("alpha") != xr_string::npos ||
                sn_lower.find("trans") != xr_string::npos)
            {
                m_fAlphaRef = 200.0f / 255.0f;  // DX11 def_aref uses oAREF=200
            }
        }
    }

    // Fallback: if no OGF_TEXTURE chunk (level geometry), use shader_id from header
    if (!m_pMaterial && RImplementation.Shaders.size() > 0)
    {
        if (shader_id < (u16)RImplementation.Shaders.size())
        {
            VK::CVulkanShader* pShader = RImplementation.Shaders[shader_id];
            if (pShader)
            {
                m_pMaterial = pShader->GetMaterial();
                if (pShader->m_bAlphaRef)
                    m_fAlphaRef = 200.0f / 255.0f;  // DX11 def_aref uses oAREF=200
            }
        }
    }

    // Phase 5: resolve the world-pass material from the same shader entry.
    // CMaterial above is parked-monolith infrastructure (g_MaterialManager is
    // a stub) — WorldMaterialCache is the live one we sample from. Lightmap
    // (3rd entry of the level shader's texture list) goes alongside diffuse
    // — null on vert-lit / unlit materials, which fall back to white.
    const char* diffuse_name = nullptr;
    const char* lmap_name    = nullptr;
    if (shader_id < (u16)RImplementation.Shaders.size()) {
        VK::CVulkanShader* pShader = RImplementation.Shaders[shader_id];
        if (pShader) {
            if (pShader->m_TexDiffuse.size() > 0) diffuse_name = pShader->m_TexDiffuse.c_str();
            if (pShader->m_TexLmap.size()    > 0) lmap_name    = pShader->m_TexLmap.c_str();
        }
    }
    // Dynamic models (NPCs/weapons/items) carry their diffuse in OGF_TEXTURE, not in
    // the LEVEL shader table — their shader_id indexes that table arbitrarily. The
    // OGF's own texture is authoritative, so prefer it whenever present; level statics
    // (no OGF_TEXTURE) keep the shader-table diffuse.
    if (ogf_diffuse[0]) diffuse_name = ogf_diffuse;
    m_pWorldMaterial = VK::WorldMaterialCache::GetOrCreate(diffuse_name, lmap_name, m_fAlphaRef);
}

// ============================================================================
// vkFVisual implementation - Standard triangle mesh
// ============================================================================
vkFVisual::vkFVisual()
{
    Type = MT_NORMAL;
}

vkFVisual::~vkFVisual()
{
    Release();
}

void vkFVisual::Load(const char* name, IReader* data, u32 flags)
{
    // Load base class data
    vkRender_Visual::Load(name, data, flags);

    // Load geometry (pass flags for VLOAD_NOVERTICES support)
    LoadGeometry(data, flags);

    // Load fast-path (optional)
    LoadFastPath(data);
}

void vkFVisual::Release()
{
    m_mesh.Destroy();
    vkRender_Visual::Release();
}

void vkFVisual::Copy(vkRender_Visual* from)
{
    vkRender_Visual::Copy(from);

    vkFVisual* src = dynamic_cast<vkFVisual*>(from);
    if (!src) return;

    // Share buffer references (not deep copy)
    // In X-Ray, meshes typically share buffers and only differ by shader
    // Mark as non-owning so Destroy() won't double-free the shared buffers
    m_mesh.p_rm_Vertices = src->m_mesh.p_rm_Vertices;
    m_mesh.vBase = src->m_mesh.vBase;
    m_mesh.vCount = src->m_mesh.vCount;
    m_mesh.vStride = src->m_mesh.vStride;
    m_mesh.tcOffset = src->m_mesh.tcOffset;

    m_mesh.p_rm_Indices = src->m_mesh.p_rm_Indices;
    m_mesh.iBase = src->m_mesh.iBase;
    m_mesh.iCount = src->m_mesh.iCount;

    m_mesh.bOwnsBuffers = false;  // Don't free shared buffers on destroy
    m_mesh.iType = src->m_mesh.iType;

    m_mesh.dwPrimitives = src->m_mesh.dwPrimitives;

    // Copy fast-path reference
    m_mesh.m_fast = src->m_mesh.m_fast;
}

void vkFVisual::Render(float /*LOD*/)
{
    // Phase 3: Render is no longer the world-pass entry point — Pass_World
    // calls Submit() instead, which queues a DrawItem; the queue's Flush()
    // does the actual binds + draw. Kept as an empty hook so direct callers
    // (HUD, particles) don't break, and to satisfy IRenderVisual.
}

void vkFVisual::Submit(VK::RenderQueue& q, const Fmatrix& xform, float LOD)
{
    if (!m_mesh.IsValid())                            return;
    if (!m_mesh.p_rm_Vertices || !m_mesh.p_rm_Indices) return;
    // Stride-12 / position-only layouts have no SHORT2 TC at tcOffset; pick
    // them up later with a position-only shader.
    if (m_mesh.vStride < m_mesh.tcOffset + 4)         return;

    VK::DrawItem it{};
    it.vis     = this;
    it.xform   = xform;
    it.lod     = LOD;
    it.sortKey = VK::makeSortKey(m_mesh.vStride, m_mesh.tcOffset,
                                 /*depthTest*/ true,
                                 m_pWorldMaterial,
                                 m_mesh.p_rm_Vertices);
    q.Push(it);
}


// ============================================================================
// ConvertFVFToLevelFormat - Convert FVF vertex data to stride-32 level format
// ============================================================================
// Level-static stride-32 layout (matches pipeline expectation):
//   FLOAT3  position   @ offset 0   (12 bytes)
//   D3DCOLOR normal    @ offset 12  (4 bytes)
//   D3DCOLOR tangent   @ offset 16  (4 bytes)  -- neutral 0x80808080
//   D3DCOLOR binormal  @ offset 20  (4 bytes)  -- neutral 0x80808080
//   SHORT2  tc0        @ offset 24  (4 bytes)  -- UV * 1024
//   SHORT2  tc1        @ offset 28  (4 bytes)  -- lightmap, zero
// Total: 32 bytes
//
static void ConvertFVFToLevelFormat(const u8* src, u8* dst, u32 vertCount, u32 fvf, u32 fvfStride)
{
    // Compute byte offsets of FVF components
    u32 posOffset = 0;
    u32 posSize = 0;
    switch (fvf & D3DFVF_POSITION_MASK) {
    case D3DFVF_XYZ:    posSize = 12; break;
    case D3DFVF_XYZRHW: posSize = 16; break;
    case D3DFVF_XYZB1:  posSize = 16; break;
    case D3DFVF_XYZB2:  posSize = 20; break;
    case D3DFVF_XYZB3:  posSize = 24; break;
    case D3DFVF_XYZB4:  posSize = 28; break;
    case D3DFVF_XYZB5:  posSize = 32; break;
    case D3DFVF_XYZW:   posSize = 16; break;
    default:            posSize = 12; break;
    }

    u32 off = posSize;
    u32 normalOffset = 0;
    bool hasNormal = false;
    if (fvf & D3DFVF_NORMAL)  { normalOffset = off; hasNormal = true; off += 12; }
    if (fvf & D3DFVF_PSIZE)   { off += 4; }
    if (fvf & D3DFVF_DIFFUSE) { off += 4; }
    if (fvf & D3DFVF_SPECULAR){ off += 4; }
    u32 texCount = (fvf & D3DFVF_TEXCOUNT_MASK) >> D3DFVF_TEXCOUNT_SHIFT;
    u32 tc0Offset = off;
    bool hasTC = (texCount > 0);

    const u32 LEVEL_STRIDE = 32;

    // X-Ray skinned/boned meshes carry a non-D3DFVF vertex code (e.g. 0x12071980,
    // stride 76) that we parse here as if it were a plain FVF. The derived
    // component offsets then point PAST the real vertex, so reading normal/uv
    // would over-read `src` and AV. Until the skinned-vertex path is ported
    // (vkSkeletonX_ST::_Load_hw), clamp every source read to within [0,fvfStride):
    // position is always the first 12 bytes (correct for boned verts too), and
    // normal/uv fall back to neutral when they don't fit. No crash; the mesh
    // loads static (no skinning) instead.
    const bool posOk    = (posOffset + 12 <= fvfStride);
    const bool normalOk = hasNormal && (normalOffset + 12 <= fvfStride);
    const bool tcOk     = hasTC && (tc0Offset + 8 <= fvfStride);

    for (u32 i = 0; i < vertCount; i++)
    {
        const u8* sv = src + i * fvfStride;
        u8* dv = dst + i * LEVEL_STRIDE;

        // Position: copy FLOAT3 (always first 12 bytes)
        if (posOk)
            CopyMemory(dv, sv + posOffset, 12);
        else
            ZeroMemory(dv, 12);

        // Normal: pack FLOAT3 -> D3DCOLOR
        if (normalOk) {
            const float* N = (const float*)(sv + normalOffset);
            u32 packed = color_rgba(vk_q_N(N[0]), vk_q_N(N[1]), vk_q_N(N[2]), 0);
            CopyMemory(dv + 12, &packed, 4);
        } else {
            u32 upNormal = color_rgba(vk_q_N(0.f), vk_q_N(1.f), vk_q_N(0.f), 0);
            CopyMemory(dv + 12, &upNormal, 4);
        }

        // Tangent + Binormal: neutral values
        u32 neutral = 0x80808080u;
        CopyMemory(dv + 16, &neutral, 4);
        CopyMemory(dv + 20, &neutral, 4);

        // TC0: convert FLOAT2 -> SHORT2 (multiply by 1024)
        if (tcOk) {
            const float* uv = (const float*)(sv + tc0Offset);
            s16 su = (s16)clampr(iFloor(uv[0] * 1024.f + 0.5f), -32768, 32767);
            s16 sv16 = (s16)clampr(iFloor(uv[1] * 1024.f + 0.5f), -32768, 32767);
            CopyMemory(dv + 24, &su, 2);
            CopyMemory(dv + 26, &sv16, 2);
        } else {
            ZeroMemory(dv + 24, 4);
        }

        // TC1 (lightmap): zero
        ZeroMemory(dv + 28, 4);
    }
}

void vkFVisual::LoadGeometry(IReader* data, u32 flags)
{
    BOOL bNoVertices = (flags & VLOAD_NOVERTICES) != 0;

    // Try OGF_GCONTAINER first (shared buffers - most common for level geometry)
    if (data->find_chunk(OGF_GCONTAINER))
    {
        // Container format:
        // u32 vb_id, u32 vb_offset, u32 vb_count
        // u32 ib_id, u32 ib_offset, u32 ib_count

        u32 vb_id = data->r_u32();
        u32 vb_offset = data->r_u32();
        u32 vb_count = data->r_u32();

        u32 ib_id = data->r_u32();
        u32 ib_offset = data->r_u32();
        u32 ib_count = data->r_u32();

        // OGF_GCONTAINER references shared VBs/IBs from CRender::nVB/nIB
        // (level.geom buffer pool). Visuals never own these — they're freed
        // once by `release_bufs(nVB,xVB,nIB,xIB)` in level_Unload. Without
        // this flag, ~vkFVisual would also try to free them → vkDestroyBuffer
        // on an invalid handle and an AV in level_Unload (caught by VK
        // validation as VUID-vkDestroyBuffer-buffer-parameter).
        m_mesh.bOwnsBuffers = false;

        // Always load index buffer info
        m_mesh.iBase = ib_offset;
        m_mesh.iCount = ib_count;
        m_mesh.dwPrimitives = ib_count / 3;

        // Get IB from pool
        if (VK::g_BufferPool)
        {
            m_mesh.p_rm_Indices = VK::g_BufferPool->GetIndexBuffer(ib_id);
            if (m_mesh.p_rm_Indices)
                m_mesh.iType = VK::g_BufferPool->GetIndexType(ib_id);
        }

        // Only load VB if not skipping vertices
        if (!bNoVertices)
        {
            m_mesh.vBase = vb_offset;
            m_mesh.vCount = vb_count;

            if (VK::g_BufferPool)
            {
                m_mesh.p_rm_Vertices = VK::g_BufferPool->GetVertexBuffer(vb_id);
                if (m_mesh.p_rm_Vertices)
                {
                    m_mesh.vStride = VK::g_BufferPool->GetVertexStride(vb_id);
                    m_mesh.tcOffset = VK::g_BufferPool->GetTexCoordOffset(vb_id);
                }
            }
        }

        if (!m_mesh.p_rm_Indices)
        {
            Msg("! [Vulkan] Visual '%s' geometry: IB[%u] not found in pool",
                dbg_name.c_str(), ib_id);
        }
        if (!bNoVertices && !m_mesh.p_rm_Vertices)
        {
            Msg("! [Vulkan] Visual '%s' geometry: VB[%u] not found in pool",
                dbg_name.c_str(), vb_id);
        }

        return;
    }

    // Try inline vertices (OGF_VERTICES + OGF_INDICES) - standalone models (weapons, etc.)
    if (!bNoVertices && data->find_chunk(OGF_VERTICES))
    {
        u32 vert_format = data->r_u32();
        u32 vert_count = data->r_u32();

        // Compute FVF stride from format flags
        u32 fvf_stride = VK_GetFVFVertexSize(vert_format);
        if (fvf_stride == 0) fvf_stride = 32;

        Msg("[FVF] LoadGeometry '%s': FVF=0x%X stride=%u verts=%u",
            dbg_name.c_str(), vert_format, fvf_stride, vert_count);

        // A real static FVF must have a position component and <= 8 texcoord sets.
        // X-Ray skinned meshes store a non-FVF vertex declaration here (e.g.
        // 0x12071980 -> posMask=0, texCount=9) whose mis-derived stride (76) would
        // make the read below over-run the OGF chunk and AV. Until the skinned
        // vertex path (vkSkeletonX_ST::_Load_hw, from the monolith) is ported, skip
        // the inline VB for such meshes (they load index-only — invisible, no crash).
        const u32 posMask = vert_format & D3DFVF_POSITION_MASK;
        const u32 texCnt = (vert_format & D3DFVF_TEXCOUNT_MASK) >> D3DFVF_TEXCOUNT_SHIFT;
        const bool looksStaticFVF = (posMask != 0) && (texCnt <= 8);

        if (!looksStaticFVF)
        {
            Msg("! [VK] '%s': skinned/non-FVF verts (fmt=0x%X stride=%u) — skipping inline VB (skinned render not ported)",
                dbg_name.c_str(), vert_format, fvf_stride);
        }
        else
        {
            // Read raw FVF vertex data
            xr_vector<u8> raw_data(vert_count * fvf_stride);
            data->r(raw_data.data(), raw_data.size());

            // Convert FVF data to level-static stride-32 format
            u32 level_stride = 32;
            xr_vector<u8> converted(vert_count * level_stride);
            ConvertFVFToLevelFormat(raw_data.data(), converted.data(), vert_count, vert_format, fvf_stride);

            m_mesh.vStride = level_stride;
            m_mesh.vCount = vert_count;
            m_mesh.vBase = 0;

            // Create and upload converted vertex buffer
            u32 data_size = vert_count * level_stride;
            m_mesh.p_rm_Vertices = xr_new<VK::CVulkanBuffer>();
            m_mesh.p_rm_Vertices->Create(
                data_size,
                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
            );
            m_mesh.p_rm_Vertices->Upload(converted.data(), data_size);
        }
    }

    if (data->find_chunk(OGF_INDICES))
    {
        u32 idx_count = data->r_u32();
        m_mesh.iCount = idx_count;
        m_mesh.iBase = 0;
        m_mesh.dwPrimitives = idx_count / 3;
        m_mesh.iType = VK_INDEX_TYPE_UINT16;

        // Read index data
        u32 data_size = idx_count * sizeof(u16);

        // Create index buffer
        m_mesh.p_rm_Indices = xr_new<VK::CVulkanBuffer>();
        m_mesh.p_rm_Indices->Create(
            data_size,
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
        );

        // Read and upload index data
        xr_vector<u16> idata(idx_count);
        data->r(idata.data(), data_size);
        m_mesh.p_rm_Indices->Upload(idata.data(), data_size);
    }
}

void vkFVisual::LoadFastPath(IReader* data)
{
    // Fast-path is optional simplified geometry for shadow maps
    if (!data->find_chunk(OGF_FASTPATH))
        return;

    // TODO: Load fast-path geometry
    // Similar to LoadGeometry but into m_mesh.m_fast

}

// ============================================================================
// vkFProgressive implementation - LOD mesh with sliding-window index ranges
// ============================================================================
vkFProgressive::vkFProgressive()
{
    Type = MT_PROGRESSIVE;
}

vkFProgressive::~vkFProgressive()
{
    Release();
}

void vkFProgressive::Load(const char* name, IReader* data, u32 flags)
{
    // Base loads OGF_HEADER, OGF_TEXTURE, OGF_GCONTAINER (VB/IB), then we
    // append the LOD slide-window table.
    vkFVisual::Load(name, data, flags);
    LoadSlidingWindow(data);
}

void vkFProgressive::Release()
{
    xr_free(sw_offsets);
    xr_free(sw_counts);
    sw_offsets = nullptr;
    sw_counts  = nullptr;
    sw_count   = 0;
    last_lod   = 0;
    vkFVisual::Release();
}

void vkFProgressive::Copy(vkRender_Visual* from)
{
    vkFVisual::Copy(from);

    vkFProgressive* src = dynamic_cast<vkFProgressive*>(from);
    if (!src) return;

    sw_count = src->sw_count;
    if (sw_count > 0)
    {
        sw_offsets = xr_alloc<u32>(sw_count);
        sw_counts  = xr_alloc<u32>(sw_count);
        CopyMemory(sw_offsets, src->sw_offsets, sw_count * sizeof(u32));
        CopyMemory(sw_counts,  src->sw_counts,  sw_count * sizeof(u32));
    }
    last_lod = 0;
}

void vkFProgressive::Submit(VK::RenderQueue& q, const Fmatrix& xform, float LOD)
{
    if (!m_mesh.IsValid())                            return;
    if (!m_mesh.p_rm_Vertices || !m_mesh.p_rm_Indices) return;
    if (m_mesh.vStride < m_mesh.tcOffset + 4)         return;
    if (sw_count == 0)                                return;

    // LOD index 0 = highest detail (full mesh). Real distance-based LOD pick
    // happens once we wire up per-visual world matrix + camera position;
    // sticking to highest detail keeps terrain visible at every range.
    const u32 lod_idx = 0;

    VK::DrawItem it{};
    it.vis            = this;
    it.xform          = xform;
    it.lod            = LOD;
    it.iBaseOverride  = m_mesh.iBase + sw_offsets[lod_idx];
    it.iCountOverride = sw_counts[lod_idx];
    it.sortKey        = VK::makeSortKey(m_mesh.vStride, m_mesh.tcOffset,
                                        /*depthTest*/ true,
                                        m_pWorldMaterial,
                                        m_mesh.p_rm_Vertices);
    q.Push(it);

    last_lod = lod_idx;
}

void vkFProgressive::LoadSlidingWindow(IReader* data)
{
    // OGF_SWIDATA layout (mirrors monolith / R4):
    //   u32 reserved[4]
    //   u32 count
    //   { u32 offset; u16 num_tris; u16 num_verts; } × count
    if (data->find_chunk(OGF_SWIDATA))
    {
        data->r_u32(); data->r_u32();
        data->r_u32(); data->r_u32();

        sw_count = data->r_u32();
        if (sw_count > 0)
        {
            sw_offsets = xr_alloc<u32>(sw_count);
            sw_counts  = xr_alloc<u32>(sw_count);
            for (u32 i = 0; i < sw_count; ++i)
            {
                sw_offsets[i] = data->r_u32();
                sw_counts[i]  = u32(data->r_u16()) * 3;  // num_tris → num_indices
                data->r_u16();                            // num_verts (unused)
            }
        }
    }

    // No SWI chunk (or zero-count) → degrade to single full-mesh "LOD".
    if (sw_count == 0)
    {
        sw_count   = 1;
        sw_offsets = xr_alloc<u32>(1);
        sw_counts  = xr_alloc<u32>(1);
        sw_offsets[0] = 0;
        sw_counts[0]  = m_mesh.iCount;
    }
}

// ============================================================================
// vkFHierrarhyVisual implementation - Composite visual
// ============================================================================
vkFHierrarhyVisual::vkFHierrarhyVisual()
{
    Type = MT_HIERRARHY;
    bDontDelete = FALSE;
}

vkFHierrarhyVisual::~vkFHierrarhyVisual()
{
    Release();
}

void vkFHierrarhyVisual::Load(const char* name, IReader* data, u32 flags)
{
    vkRender_Visual::Load(name, data, flags);

    // Try link-based children first (references to pre-loaded visuals)
    if (data->find_chunk(OGF_CHILDREN_L))
    {
        u32 count = data->r_u32();
        children.resize(count);

        for (u32 i = 0; i < count; ++i)
        {
            u32 id = data->r_u32();
            children[i] = static_cast<vkRender_Visual*>(RImplementation.getVisual(id));

            if (!children[i]) {
                Msg("![Vulkan] Hierarchy '%s': Failed to load child %u (visual ID %u)",
                    dbg_name.c_str(), i, id);
                Msg("![Vulkan] Child will be skipped during rendering");
            }
        }

        bDontDelete = TRUE;

        return;
    }

    // Try stream-based children (inline child definitions)
    if (data->find_chunk(OGF_CHILDREN))
    {
        IReader* chunk = data->open_chunk(OGF_CHILDREN);
        if (chunk)
        {
            u32 count = 0;
            IReader* child_chunk;

            while ((child_chunk = chunk->open_chunk(count)) != nullptr)
            {
                // Create child name: "parent_name:N"
                string_path child_name;
                xr_sprintf(child_name, "%s:%u", name, count + 1);

                // Create child visual
                IRenderVisual* child = RImplementation.model_CreateChild(child_name, child_chunk);
                if (child)
                {
                    children.push_back(static_cast<vkRender_Visual*>(child));
                }

                child_chunk->close();
                ++count;
            }

            chunk->close();
            bDontDelete = FALSE;

        }
    }
}

void vkFHierrarhyVisual::Release()
{
    if (!bDontDelete)
    {
        // Delete owned children
        for (auto& child : children)
        {
            if (child)
            {
                IRenderVisual* p = child;
                RImplementation.model_Delete(p, TRUE);
            }
        }
    }

    children.clear();

    vkRender_Visual::Release();
}

void vkFHierrarhyVisual::Copy(vkRender_Visual* from)
{
    vkRender_Visual::Copy(from);

    vkFHierrarhyVisual* src = dynamic_cast<vkFHierrarhyVisual*>(from);
    if (!src) return;

    // Deep copy children
    children.clear();
    children.reserve(src->children.size());

    for (u32 i = 0; i < src->children.size(); i++)
    {
        auto* src_child = src->children[i];
        IRenderVisual* child_copy = RImplementation.model_Duplicate(src_child);
        children.push_back(static_cast<vkRender_Visual*>(child_copy));
    }

    bDontDelete = FALSE;
}

void vkFHierrarhyVisual::Render(float LOD)
{
    // Phase 3: world-pass goes through Submit(); see vkFVisual::Render.
}

void vkFHierrarhyVisual::Submit(VK::RenderQueue& q, const Fmatrix& xform, float LOD)
{
    for (auto& child : children) {
        if (child) child->Submit(q, xform, LOD);
    }
}

void vkFHierrarhyVisual::Spawn()
{
    // Spawn all children
    for (auto& child : children)
    {
        if (child)
        {
            child->Spawn();
        }
    }
}

void vkFHierrarhyVisual::Depart()
{
    // Depart all children
    for (auto& child : children)
    {
        if (child)
        {
            child->Depart();
        }
    }
}

// ============================================================================
// vkFTreeVisual — base. Geometry loads through vkFVisual; we only add the
// OGF_TREEDEF2 chunk read (xform + per-instance lighting modulation).
// Render/Submit are no-ops: VK::CTreeManager extracts every tree at level
// load and draws all of them via GPU-driven indirect from a single SSBO.
// ============================================================================
vkFTreeVisual::vkFTreeVisual()
{
    Type = MT_TREE_ST;
    xform.identity();
    c_scale.rgb.set(1.f, 1.f, 1.f);
    c_scale.hemi = 1.f;
    c_scale.sun  = 1.f;
    c_bias.rgb.set(0.f, 0.f, 0.f);
    c_bias.hemi  = 0.f;
    c_bias.sun   = 0.f;
}

vkFTreeVisual::~vkFTreeVisual() {}

void vkFTreeVisual::Load(const char* name, IReader* data, u32 flags)
{
    vkFVisual::Load(name, data, flags);
    LoadTreeDef(data);

    // Trees always need alpha test for foliage cutout (matches DX11 oAREF=200)
    m_fAlphaRef = 200.f / 255.f;
}

void vkFTreeVisual::Render(float /*LOD*/) {}
void vkFTreeVisual::Submit(VK::RenderQueue& /*q*/, const Fmatrix& /*xform*/, float /*LOD*/) {}

void vkFTreeVisual::LoadTreeDef(IReader* data)
{
    if (!data->find_chunk(OGF_TREEDEF2))
    {
        Msg("! [vkFTreeVisual] OGF_TREEDEF2 missing in '%s'", dbg_name.c_str());
        return;
    }

    data->r(&xform, sizeof(xform));
    data->r(&c_scale, sizeof(c_scale));
    data->r(&c_bias,  sizeof(c_bias));

    // R4 lighting tweak: hemi halved and multiplied by ps_r__Tree_SBC * 4/3
    // (≈ 1.5 * 1.3333 = 2.0). Net effect on hemi is unchanged but kept verbatim
    // to match R4 numerics if SBC ever varies.
    constexpr float ps_r__Tree_SBC = 1.5f;
    constexpr float s = ps_r__Tree_SBC * 1.3333f;
    c_scale.hemi *= 0.5f * s;
    c_bias.hemi  *= 0.5f * s;
}

// ----------------------------------------------------------------------------
vkFTreeVisual_ST::vkFTreeVisual_ST()  { Type = MT_TREE_ST; }
vkFTreeVisual_ST::~vkFTreeVisual_ST() {}

// ----------------------------------------------------------------------------
vkFTreeVisual_PM::vkFTreeVisual_PM() { Type = MT_TREE_PM; }
vkFTreeVisual_PM::~vkFTreeVisual_PM() { Release(); }

void vkFTreeVisual_PM::Load(const char* name, IReader* data, u32 flags)
{
    vkFTreeVisual::Load(name, data, flags);
    LoadSlidingWindow(data);
}

void vkFTreeVisual_PM::Release()
{
    xr_free(sw_offsets);
    xr_free(sw_counts);
    sw_offsets = nullptr;
    sw_counts  = nullptr;
    sw_count   = 0;
    vkFTreeVisual::Release();
}

void vkFTreeVisual_PM::LoadSlidingWindow(IReader* data)
{
    // Trees store their sliding window POOLED via OGF_SWICONTAINER (an id into
    // CRender::SWIs), exactly like R4 FTreeVisual_PM — NOT the inline OGF_SWIDATA
    // that vkFProgressive uses. sw[0] is the max-quality LOD (R4 select_lod_id:
    // lod=1.0 → lod_id=0). TreeManager draws sw[0]'s range; without this the
    // full container index range includes collapsed PM data → stray "floating
    // branch" triangles.
    if (data->find_chunk(OGF_SWICONTAINER))
    {
        const u32 id = data->r_u32();
        FSlideWindowItem* it = RImplementation.getSWI((int)id);
        if (it && it->sw && it->count > 0)
        {
            sw_count   = it->count;
            sw_offsets = xr_alloc<u32>(sw_count);
            sw_counts  = xr_alloc<u32>(sw_count);
            for (u32 i = 0; i < sw_count; ++i)
            {
                sw_offsets[i] = it->sw[i].offset;
                sw_counts[i]  = (u32)it->sw[i].num_tris * 3;
            }
        }
        return;
    }

    // Fallback: inline OGF_SWIDATA (same layout as vkFProgressive).
    if (!data->find_chunk(OGF_SWIDATA)) return;

    data->r_u32(); data->r_u32();
    data->r_u32(); data->r_u32();

    sw_count = data->r_u32();
    if (sw_count == 0) return;

    sw_offsets = xr_alloc<u32>(sw_count);
    sw_counts  = xr_alloc<u32>(sw_count);
    for (u32 i = 0; i < sw_count; ++i)
    {
        sw_offsets[i] = data->r_u32();
        sw_counts[i]  = (u32)data->r_u16() * 3;  // num_tris * 3
        data->r_u16();                           // skip num_verts
    }
}

// ============================================================================
// vkFLOD — MT_LOD container with billboard imposter facets (OGF_LODDEF2).
// Inherits the hierarchy (children = full-detail meshes the tree manager already
// renders); adds the 8 pre-rendered billboard quads for the imposter pass.
// Mirrors R4 FLOD::Load (facet vertices are world-space; normal = averaged,
// inverted face normal used to pick the best facet per camera direction).
// ============================================================================
vkFLOD::vkFLOD()  { Type = MT_LOD; }
vkFLOD::~vkFLOD() {}

void vkFLOD::Load(const char* name, IReader* data, u32 flags)
{
    vkFHierrarhyVisual::Load(name, data, flags);   // header (vis.sphere) + children
    Type = MT_LOD;

    if (!data->find_chunk(OGF_LODDEF2))
        return;

    for (auto& facet : facets)
    {
        data->r(facet.v, sizeof(facet.v));         // 4 × LodVertex, raw

        Fvector N, T;
        N.set(0, 0, 0);
        T.mknormal(facet.v[0].v, facet.v[1].v, facet.v[2].v); N.add(T);
        T.mknormal(facet.v[1].v, facet.v[2].v, facet.v[3].v); N.add(T);
        T.mknormal(facet.v[2].v, facet.v[3].v, facet.v[0].v); N.add(T);
        T.mknormal(facet.v[3].v, facet.v[0].v, facet.v[1].v); N.add(T);
        N.div(4.f);
        facet.N.normalize(N);
        facet.N.invert();
    }
    facetsValid = true;
}

// ============================================================================
// MVP factory: only static meshes (MT_NORMAL) and composite hierarchies
// (MT_HIERRARHY) build real visuals. Everything else returns a dummy that
// reports its type but draws nothing — the loader needs SOMETHING valid for
// every Visuals[] slot, but an unsupported type silently no-ops at render
// time instead of crashing.
// ============================================================================
namespace {
class vkVisualDummy final : public vkRender_Visual
{
public:
    void Render(float /*LOD*/) override {}
};
} // namespace

// ============================================================================
// Skinned-mesh vertex load (STEP B sub-step 1) — ported from the monolith.
// vertBoned* = raw OGF boned vertex layouts; converted to vertHW_* (vk_Visual.h)
// and uploaded as a Vulkan VB. Bone deform / skinned draw is sub-step 2.
// ============================================================================
struct vertBoned1W { Fvector P; Fvector3 N; Fvector3 T; Fvector3 B; float u, v; u32 matrix; };
#pragma pack(push, 2)
struct vertBoned2W { u16 matrix0; u16 matrix1; Fvector P; Fvector N; Fvector T; Fvector B; float w; float u, v; };
struct vertBoned3W { u16 m[3]; Fvector P; Fvector N; Fvector T; Fvector B; float w[2]; float u, v; };
struct vertBoned4W { u16 m[4]; Fvector P; Fvector N; Fvector T; Fvector B; float w[3]; float u, v; };
#pragma pack(pop)

static void vkUploadConvertedVertices(VK_Render_Mesh& mesh, void* dst, u32 vStride, u32 vertCount)
{
    mesh.p_rm_Vertices = xr_new<VK::CVulkanBuffer>();
    mesh.p_rm_Vertices->Create(vertCount * vStride,
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    mesh.p_rm_Vertices->Upload(dst, vertCount * vStride);
    mesh.vStride = vStride;
}

// Convert vertBoned* -> vertHW_* and upload. renderMode uses the vkSkeletonX_ST enum
// (identical in vkSkeletonX_PM). Bone index is stored RAW (0..BoneCount-1) — the
// skinned shader indexes the bones[] SSBO directly. (Earlier it used the legacy DX
// "*3" matrix-row stride, but that overflowed the u8 index channels in vertHW_* for
// bones >85 (idx*3>255), garbling whole regions — e.g. a sail-like spike at the
// shoulder. Direct indexing supports up to 255 bones, covering all real skeletons.)
static void vkLoadSkinnedVertices(VK_Render_Mesh& mesh, u16 renderMode, void* _verts_, u32 dwVertCount, const char* diagTag)
{
    using RM = vkSkeletonX_ST;
    switch (renderMode)
    {
    case RM::RM_SINGLE:
    case RM::RM_SKINNING_1B:
    {
        u32 vStride = sizeof(vertHW_1W);
        vertHW_1W* dst = xr_alloc<vertHW_1W>(dwVertCount);
        vertBoned1W* src = (vertBoned1W*)_verts_;
        for (u32 i = 0; i < dwVertCount; i++) {
            Fvector2 uv; uv.set(src->u, src->v);
            dst[i].set(src->P, src->N, src->T, src->B, uv, src->matrix);
            src++;
        }
        vkUploadConvertedVertices(mesh, dst, vStride, dwVertCount);
        xr_free(dst);
    } break;
    case RM::RM_SKINNING_2B:
    {
        u32 vStride = sizeof(vertHW_2W);
        vertHW_2W* dst = xr_alloc<vertHW_2W>(dwVertCount);
        vertBoned2W* src = (vertBoned2W*)_verts_;
        for (u32 i = 0; i < dwVertCount; i++) {
            Fvector2 uv; uv.set(src->u, src->v);
            dst[i].set(src->P, src->N, src->T, src->B, uv, src->matrix0, src->matrix1, src->w);
            src++;
        }
        vkUploadConvertedVertices(mesh, dst, vStride, dwVertCount);
        xr_free(dst);
    } break;
    case RM::RM_SKINNING_3B:
    {
        u32 vStride = sizeof(vertHW_3W);
        vertHW_3W* dst = xr_alloc<vertHW_3W>(dwVertCount);
        vertBoned3W* src = (vertBoned3W*)_verts_;
        for (u32 i = 0; i < dwVertCount; i++) {
            Fvector2 uv; uv.set(src->u, src->v);
            dst[i].set(src->P, src->N, src->T, src->B, uv, src->m[0], src->m[1], src->m[2], src->w[0], src->w[1]);
            src++;
        }
        vkUploadConvertedVertices(mesh, dst, vStride, dwVertCount);
        xr_free(dst);
    } break;
    case RM::RM_SKINNING_4B:
    {
        u32 vStride = sizeof(vertHW_4W);
        vertHW_4W* dst = xr_alloc<vertHW_4W>(dwVertCount);
        vertBoned4W* src = (vertBoned4W*)_verts_;
        for (u32 i = 0; i < dwVertCount; i++) {
            Fvector2 uv; uv.set(src->u, src->v);
            dst[i].set(src->P, src->N, src->T, src->B, uv, src->m[0], src->m[1], src->m[2], src->m[3], src->w[0], src->w[1], src->w[2]);
            src++;
        }
        vkUploadConvertedVertices(mesh, dst, vStride, dwVertCount);
        xr_free(dst);
    } break;
    default:
        Msg("! [%s] vkLoadSkinnedVertices: unknown RenderMode %u", diagTag ? diagTag : "SKL", (u32)renderMode);
        break;
    }
}

// Shared bone-analysis: scans the OGF boned vertices to pick RenderMode + bone count.
template <class TLeaf>
static void vk_skinned_analyse(TLeaf* leaf, u32 dwVertType, void* _verts_, u32 dwVertCount)
{
    xr_vector<u16> bids;
    u16 sw_bones_cnt = 0;
    switch (dwVertType)
    {
    case OGF_VERTEXFORMAT_FVF_1L:
    case 1: {
        vertBoned1W* pVO = (vertBoned1W*)_verts_;
        for (u32 it = 0; it < dwVertCount; ++it) {
            u16 mid = (u16)pVO[it].matrix;
            if (bids.end() == std::find(bids.begin(), bids.end(), mid)) bids.push_back(mid);
            sw_bones_cnt = _max(sw_bones_cnt, mid);
        }
        if (1 == bids.size()) { leaf->RenderMode = TLeaf::RM_SINGLE; leaf->RMS_boneid = *bids.begin(); }
        else { leaf->RenderMode = TLeaf::RM_SKINNING_1B; leaf->RMS_bonecount = sw_bones_cnt + 1; leaf->BonesUsed = (u16)bids.size(); }
    } break;
    case OGF_VERTEXFORMAT_FVF_2L:
    case 2: {
        vertBoned2W* pVO = (vertBoned2W*)_verts_;
        for (u32 it = 0; it < dwVertCount; ++it) {
            sw_bones_cnt = _max(sw_bones_cnt, pVO[it].matrix0);
            sw_bones_cnt = _max(sw_bones_cnt, pVO[it].matrix1);
            if (bids.end() == std::find(bids.begin(), bids.end(), pVO[it].matrix0)) bids.push_back(pVO[it].matrix0);
            if (bids.end() == std::find(bids.begin(), bids.end(), pVO[it].matrix1)) bids.push_back(pVO[it].matrix1);
        }
        leaf->RenderMode = TLeaf::RM_SKINNING_2B; leaf->RMS_bonecount = sw_bones_cnt + 1; leaf->BonesUsed = (u16)bids.size();
    } break;
    case OGF_VERTEXFORMAT_FVF_3L:
    case 3: {
        vertBoned3W* pVO = (vertBoned3W*)_verts_;
        for (u32 it = 0; it < dwVertCount; ++it)
            for (int k = 0; k < 3; k++) {
                sw_bones_cnt = _max(sw_bones_cnt, pVO[it].m[k]);
                if (bids.end() == std::find(bids.begin(), bids.end(), pVO[it].m[k])) bids.push_back(pVO[it].m[k]);
            }
        leaf->RenderMode = TLeaf::RM_SKINNING_3B; leaf->RMS_bonecount = sw_bones_cnt + 1; leaf->BonesUsed = (u16)bids.size();
    } break;
    case OGF_VERTEXFORMAT_FVF_4L:
    case 4: {
        vertBoned4W* pVO = (vertBoned4W*)_verts_;
        for (u32 it = 0; it < dwVertCount; ++it)
            for (int k = 0; k < 4; k++) {
                sw_bones_cnt = _max(sw_bones_cnt, pVO[it].m[k]);
                if (bids.end() == std::find(bids.begin(), bids.end(), pVO[it].m[k])) bids.push_back(pVO[it].m[k]);
            }
        leaf->RenderMode = TLeaf::RM_SKINNING_4B; leaf->RMS_bonecount = sw_bones_cnt + 1; leaf->BonesUsed = (u16)bids.size();
    } break;
    default:
        Msg("![Vulkan] vk_skinned_analyse: unknown vertex type 0x%X", dwVertType);
        leaf->RenderMode = TLeaf::RM_SINGLE; leaf->RMS_boneid = 0;
        break;
    }

    // DIAG (logs only on a new high-water mark): the largest bone index used by any
    // skinned leaf. A value > 85 confirms the legacy idx*3 u8 packing would have
    // overflowed (255/3) → the shoulder-spike artifact. Now stored RAW, safe to 255.
    static u16 s_maxBoneIdx = 0;
    if (sw_bones_cnt > s_maxBoneIdx) {
        s_maxBoneIdx = sw_bones_cnt;
        Msg("[VK Skinned] DIAG max bone index = %u (old idx*3 u8 overflow threshold = 85)", (u32)s_maxBoneIdx);
    }
}

void vkSkeletonX_ST::Load(const char* name, IReader* data, u32 flags)
{
    R_ASSERT(data->find_chunk(OGF_VERTICES));
    u32 dwVertType  = data->r_u32();
    u32 dwVertCount = data->r_u32();
    void* _verts_   = data->pointer();

    vk_skinned_analyse(this, dwVertType, _verts_, dwVertCount);

    data->seek(0);
    vkFVisual::Load(name, data, flags | VLOAD_NOVERTICES);   // indices only

    m_mesh.vBase = 0;
    m_mesh.vCount = dwVertCount;
    _Load_hw_VK(_verts_, dwVertType, dwVertCount);
}
void vkSkeletonX_ST::_Load_hw_VK(void* verts, u32 /*vertType*/, u32 vertCount) { vkLoadSkinnedVertices(m_mesh, RenderMode, verts, vertCount, "SKL-ST"); }
void vkSkeletonX_ST::Copy(vkRender_Visual* from)
{
    vkFVisual::Copy(from);
    if (auto* s = dynamic_cast<vkSkeletonX_ST*>(from)) { RenderMode = s->RenderMode; BonesUsed = s->BonesUsed; RMS_bonecount = s->RMS_bonecount; }
}

void vkSkeletonX_PM::Load(const char* name, IReader* data, u32 flags)
{
    R_ASSERT(data->find_chunk(OGF_VERTICES));
    u32 dwVertType  = data->r_u32();
    u32 dwVertCount = data->r_u32();
    void* _verts_   = data->pointer();

    vk_skinned_analyse(this, dwVertType, _verts_, dwVertCount);

    data->seek(0);
    vkFProgressive::Load(name, data, flags | VLOAD_NOVERTICES);   // indices + sliding window only

    m_mesh.vBase = 0;
    m_mesh.vCount = dwVertCount;
    _Load_hw_VK(_verts_, dwVertType, dwVertCount);
}
void vkSkeletonX_PM::_Load_hw_VK(void* verts, u32 /*vertType*/, u32 vertCount) { vkLoadSkinnedVertices(m_mesh, RenderMode, verts, vertCount, "SKL-PM"); }
void vkSkeletonX_PM::Copy(vkRender_Visual* from)
{
    vkFProgressive::Copy(from);
    if (auto* s = dynamic_cast<vkSkeletonX_PM*>(from)) { RenderMode = s->RenderMode; BonesUsed = s->BonesUsed; RMS_bonecount = s->RMS_bonecount; }
}

vkRender_Visual* vkVisual_CreateDummy()
{
    return xr_new<vkVisualDummy>();
}

// Real-skeleton factories live in the wrapper TUs: vkCreateKinematics() in
// vk_SkeletonCustom.cpp (CKinematics), vkCreateKinematicsAnimated() in
// vk_SkeletonAnimated.cpp (CKinematicsAnimated). These compile OGSR's own
// skeleton classes under the dxRender_Visual->vkRender_Visual mapping.
extern vkRender_Visual* vkCreateKinematics();
extern vkRender_Visual* vkCreateKinematicsAnimated();
extern vkRender_Visual* vkCreateKinematicsStub();

vkRender_Visual* vkVisual_Create(u32 type)
{
    switch (type)
    {
    case MT_NORMAL:
        return xr_new<vkFVisual>();

    case MT_HIERRARHY:
        return xr_new<vkFHierrarhyVisual>();

    case MT_PROGRESSIVE:
        // Level terrain / ground / any LOD-able static mesh. Geometry is
        // loaded the same way as MT_NORMAL; OGF_SWIDATA picks per-LOD IB
        // slices. Without this case, all progressive geometry routed to a
        // dummy and the world's ground was missing — visible as gaps where
        // you could see "under" the level.
        return xr_new<vkFProgressive>();

    case MT_TREE_ST:
        return xr_new<vkFTreeVisual_ST>();
    case MT_TREE_PM:
        return xr_new<vkFTreeVisual_PM>();

    case MT_LOD:
        // LOD container — OGF_CHILDREN_L (full-detail meshes, recursed by the
        // tree manager) PLUS OGF_LODDEF2 billboard facets (vkFLOD), drawn as
        // distant imposters by the LOD imposter manager.
        return xr_new<vkFLOD>();

    case MT_SKELETON_ANIM:
        // Real animated skeleton (CKinematicsAnimated). Children are the
        // vkSkeletonX_ST/_PM leaves below; CKinematics::Load child-handling is
        // patched (vk_SkeletonCustom_shared.inl) to link them.
        return vkCreateKinematicsAnimated();
    case MT_SKELETON_RIGID:
        // Real rigid skeleton (CKinematics).
        return vkCreateKinematics();

    // GeomDef leaves: placeholder vkSkeletonX_ST/PM (no rendering). Real
    // skinned mesh classes land in Session 2-3.
    case MT_SKELETON_GEOMDEF_PM:
        return xr_new<vkSkeletonX_PM>();
    case MT_SKELETON_GEOMDEF_ST:
        return xr_new<vkSkeletonX_ST>();

    default:
        // MT_PARTICLE_* / MT_LOD — valid OGF types, visual classes not yet
        // ported.
        Msg("[VK-Visual] type=%u not supported yet — using dummy", type);
        return vkVisual_CreateDummy();
    }
}
