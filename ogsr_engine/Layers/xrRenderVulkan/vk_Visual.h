// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

#pragma once

#include "stdafx.h"
#include "../../xr_3da/vis_common.h"
#include "../../Include/xrRender/RenderVisual.h"
#include "vk_buffer.h"

// Flag: skip vertex loading in base class (skinned meshes create their own VB)
#ifndef VLOAD_NOVERTICES
#define VLOAD_NOVERTICES (1<<0)
#endif

// ============================================================================
// Helper: quantize float normal component [-1,+1] to u8 [0,255]
// Matches DX11's q_N() in FSkinned.cpp
// ============================================================================
inline u8 vk_q_N(float v)
{
    int _v = clampr(iFloor((v + 1.f) * 127.5f), 0, 255);
    return u8(_v);
}

// ============================================================================
// Hardware vertex formats for skinned meshes
// These match DX11's vertHW_1W/2W/3W/4W from FSkinned.cpp
// vertBoned* (raw OGF data) is converted to these formats for GPU consumption
// ============================================================================

// 1-weight skinned vertex (36 bytes)
struct vertHW_1W
{
    float _P[4];    // Position (xyz) + 1.0 (w)
    u32   _N_I;     // Normal packed (RGB) + bone_index (A)
    u32   _T;       // Tangent packed (RGB)
    u32   _B;       // Binormal packed (RGB)
    float _tc[2];   // TexCoord UV

    void set(Fvector3& P, Fvector3 N, Fvector3 T, Fvector3 B, Fvector2& tc, int index)
    {
        N.normalize_safe();
        T.normalize_safe();
        B.normalize_safe();
        _P[0] = P.x; _P[1] = P.y; _P[2] = P.z; _P[3] = 1.f;
        _N_I = color_rgba(vk_q_N(N.x), vk_q_N(N.y), vk_q_N(N.z), u8(index));
        _T   = color_rgba(vk_q_N(T.x), vk_q_N(T.y), vk_q_N(T.z), 0);
        _B   = color_rgba(vk_q_N(B.x), vk_q_N(B.y), vk_q_N(B.z), 0);
        _tc[0] = tc.x; _tc[1] = tc.y;
    }
};  // sizeof = 36
static_assert(sizeof(vertHW_1W) == 36, "vertHW_1W must be 36 bytes");

// 2-weight skinned vertex (44 bytes)
struct vertHW_2W
{
    float _P[4];      // Position (xyz) + 1.0 (w)
    u32   _N_w;       // Normal packed (RGB) + weight0 (A)
    u32   _T;         // Tangent packed (RGB)
    u32   _B;         // Binormal packed (RGB)
    float _tc_i[4];   // tc.xy + bone_index0 (as float bits) + bone_index1 (as float bits)

    void set(Fvector3& P, Fvector3 N, Fvector3 T, Fvector3 B, Fvector2& tc,
             int index0, int index1, float w)
    {
        N.normalize_safe();
        T.normalize_safe();
        B.normalize_safe();
        _P[0] = P.x; _P[1] = P.y; _P[2] = P.z; _P[3] = 1.f;
        _N_w  = color_rgba(vk_q_N(N.x), vk_q_N(N.y), vk_q_N(N.z), u8(clampr(iFloor(w * 255.f + .5f), 0, 255)));
        _T    = color_rgba(vk_q_N(T.x), vk_q_N(T.y), vk_q_N(T.z), 0);
        _B    = color_rgba(vk_q_N(B.x), vk_q_N(B.y), vk_q_N(B.z), 0);
        _tc_i[0] = tc.x; _tc_i[1] = tc.y;
        // Store bone indices as actual float values for Vulkan
        // (DX11 used D3DDECLTYPE_SHORT2 which auto-converts s16->float,
        //  but Vulkan R32G32B32A32_SFLOAT reads raw float bits)
        _tc_i[2] = float(index0);
        _tc_i[3] = float(index1);
    }
};  // sizeof = 44
static_assert(sizeof(vertHW_2W) == 44, "vertHW_2W must be 44 bytes");

// 3-weight skinned vertex (44 bytes)
struct vertHW_3W
{
    float _P[4];      // Position (xyz) + 1.0 (w)
    u32   _N_w;       // Normal packed (RGB) + weight0 (A)
    u32   _T_w;       // Tangent packed (RGB) + weight1 (A)
    u32   _B_i;       // Binormal packed (RGB) + bone_index2 (A)
    float _tc_i[4];   // tc.xy + bone_index0 + bone_index1

    void set(Fvector3& P, Fvector3 N, Fvector3 T, Fvector3 B, Fvector2& tc,
             int index0, int index1, int index2, float w0, float w1)
    {
        N.normalize_safe();
        T.normalize_safe();
        B.normalize_safe();
        _P[0] = P.x; _P[1] = P.y; _P[2] = P.z; _P[3] = 1.f;
        _N_w  = color_rgba(vk_q_N(N.x), vk_q_N(N.y), vk_q_N(N.z), u8(clampr(iFloor(w0 * 255.f + .5f), 0, 255)));
        _T_w  = color_rgba(vk_q_N(T.x), vk_q_N(T.y), vk_q_N(T.z), u8(clampr(iFloor(w1 * 255.f + .5f), 0, 255)));
        _B_i  = color_rgba(vk_q_N(B.x), vk_q_N(B.y), vk_q_N(B.z), u8(index2));
        _tc_i[0] = tc.x; _tc_i[1] = tc.y;
        // Store bone indices as actual float values for Vulkan
        _tc_i[2] = float(index0);
        _tc_i[3] = float(index1);
    }
};  // sizeof = 44
static_assert(sizeof(vertHW_3W) == 44, "vertHW_3W must be 44 bytes");

// 4-weight skinned vertex (40 bytes)
struct vertHW_4W
{
    float _P[4];    // Position (xyz) + 1.0 (w)
    u32   _N_w;     // Normal packed (RGB) + weight0 (A)
    u32   _T_w;     // Tangent packed (RGB) + weight1 (A)
    u32   _B_w;     // Binormal packed (RGB) + weight2 (A)
    float _tc[2];   // TexCoord UV
    u32   _i;       // 4 bone indices packed as RGBA

    void set(Fvector3& P, Fvector3 N, Fvector3 T, Fvector3 B, Fvector2& tc,
             int index0, int index1, int index2, int index3,
             float w0, float w1, float w2)
    {
        N.normalize_safe();
        T.normalize_safe();
        B.normalize_safe();
        _P[0] = P.x; _P[1] = P.y; _P[2] = P.z; _P[3] = 1.f;
        _N_w = color_rgba(vk_q_N(N.x), vk_q_N(N.y), vk_q_N(N.z), u8(clampr(iFloor(w0 * 255.f + .5f), 0, 255)));
        _T_w = color_rgba(vk_q_N(T.x), vk_q_N(T.y), vk_q_N(T.z), u8(clampr(iFloor(w1 * 255.f + .5f), 0, 255)));
        _B_w = color_rgba(vk_q_N(B.x), vk_q_N(B.y), vk_q_N(B.z), u8(clampr(iFloor(w2 * 255.f + .5f), 0, 255)));
        _tc[0] = tc.x; _tc[1] = tc.y;
        _i = color_rgba(u8(index0), u8(index1), u8(index2), u8(index3));
    }
};  // sizeof = 40
static_assert(sizeof(vertHW_4W) == 40, "vertHW_4W must be 40 bytes");

// Forward declarations
namespace VK {
    class CVulkanBuffer;
    class CMaterial;
    class RenderQueue;
    struct WorldMaterial;
}

// ============================================================================
// Vulkan Mesh Data - replaces IRender_Mesh from DX11
// Contains VkBuffer handles instead of D3D buffers
// ============================================================================
struct VK_Render_Mesh
{
    // Vertex buffer
    VK::CVulkanBuffer*  p_rm_Vertices = nullptr;
    u32                 vBase = 0;      // First vertex offset
    u32                 vCount = 0;     // Vertex count
    u32                 vStride = 0;    // Bytes per vertex
    u32                 tcOffset = 24;  // TEXCOORD0 byte offset (24=lmap, 28=vert-lit)

    // Index buffer
    VK::CVulkanBuffer*  p_rm_Indices = nullptr;
    u32                 iBase = 0;      // First index offset
    u32                 iCount = 0;     // Index count
    VkIndexType         iType = VK_INDEX_TYPE_UINT16;

    // Primitive info
    u32                 dwPrimitives = 0;

    // Fast-path geometry (for shadow maps)
    VK_Render_Mesh*     m_fast = nullptr;

    // Ownership flag: false when buffers are shared via Copy() (don't delete them)
    bool                bOwnsBuffers = true;

    VK_Render_Mesh() = default;
    ~VK_Render_Mesh();

    void Destroy();
    bool IsValid() const { return p_rm_Vertices != nullptr && vCount > 0; }
};

// ============================================================================
// vkRender_Visual - Base visual class for Vulkan renderer
// Equivalent to dxRender_Visual from DX11 renderer
// ============================================================================
class vkRender_Visual : public IRenderVisual
{
public:
    // Type from OGF header
    u32                 Type = 0;

    // Shader ID (index into RImplementation.Shaders array)
    // Phase 2.34: Shader-Material Binding
    u16                 shader_id = 0;

    // Visibility data (bounding box/sphere for culling)
    vis_data            vis;

    // Skinning quality (-1 = no skinning)
    s32                 skinning = -1;

    // Material (diffuse texture + descriptor set)
    VK::CMaterial*      m_pMaterial = nullptr;

    // Phase-5 world-pass material — resolved from shader_id during LoadTexture.
    // Always non-null after Load (defaults to white) so Submit doesn't have
    // to null-check.
    VK::WorldMaterial*  m_pWorldMaterial = nullptr;

    // Alpha-test threshold: -1.0 = disabled (solid), 0.5 = enabled (foliage/aref)
    float               m_fAlphaRef = -1.0f;

    // Debug name
    shared_str          dbg_name;

    // Debug ID (used by FHierrarhyVisual shared code)
    u32                 dbg_id = 0;

protected:
    // Render-flag — `setRZFlag/getRZFlag` from xrRender SkeletonCustom.h:76-77.
    // Default true, set to false to skip rendering this child visual.
    bool                _renderFlag = true;

public:
    inline bool         getRZFlag() const                { return _renderFlag; }
    inline void         setRZFlag(const bool f)          { _renderFlag = f; }

    vkRender_Visual();
    virtual ~vkRender_Visual();

    // Overload matching xrRender's dxRender_Visual::Render signature. The
    // Vulkan path doesn't use cmd_list / use_fast_geo (we have Submit instead),
    // so this just delegates to the LOD-only Render(). Skeleton wrappers call
    // this overload via the macro-mapped dxRender_Visual.
    virtual void Render(class CBackend& cmd_list, float lod, bool use_fast_geo) { Render(lod); }

    // ========================================================================
    // IRenderVisual interface
    // ========================================================================
    virtual vis_data&   _BCL getVisData() override { return vis; }
    virtual u32         getType() override { return Type; }
    virtual shared_str  getDebugName() const override { return dbg_name; }
    virtual shared_str  getDebugInfo() const override { return dbg_name; }

    // ========================================================================
    // Core methods - must be implemented by derived classes
    // ========================================================================

    // Load visual data from OGF file
    virtual void Load(const char* name, IReader* data, u32 flags);

    // Release GPU resources
    virtual void Release();

    // Copy from another visual (for duplication)
    virtual void Copy(vkRender_Visual* from);

    // Render the visual with specified LOD. After phase 3 this is a no-op
    // for the world pass — Pass_World traverses via Submit() instead.
    // Other paths (HUD, particles) may still call Render().
    virtual void Render(float LOD);

    // Phase-3 submission: push DrawItem(s) into the render queue. Pass_World
    // calls Submit on top-level Visuals; composite (Hier) visuals recurse
    // into children. Leaf vkFVisual::Submit pushes one DrawItem and does no
    // Vulkan work. Default (base class) is a no-op for skinned/particle
    // stubs and other inert types.
    virtual void Submit(VK::RenderQueue& q, const Fmatrix& xform, float LOD) {}

    // Called when instance is activated from pool
    virtual void Spawn() {}

    // Called when instance is returned to pool
    virtual void Depart() {}

    // Set debug ID (used by shared FHierrarhyVisual)
    virtual void setID(u32 id) { dbg_id = id; }

    // HeatVision support (stub)
    virtual void MarkAsHot(bool is_hot) {}

protected:
    // Load common data from OGF header
    void LoadHeader(IReader* data);

    // Load shader/texture info
    void LoadTexture(IReader* data);
};

// ============================================================================
// vkFVisual - Standard triangle mesh for Vulkan
// Equivalent to Fvisual from DX11 renderer
// ============================================================================
class vkFVisual : public vkRender_Visual
{
public:
    // Mesh data (vertex/index buffers)
    VK_Render_Mesh      m_mesh;

public:
    vkFVisual();
    virtual ~vkFVisual();

    // ========================================================================
    // vkRender_Visual overrides
    // ========================================================================
    virtual void Load(const char* name, IReader* data, u32 flags) override;
    virtual void Release() override;
    virtual void Copy(vkRender_Visual* from) override;
    virtual void Render(float LOD) override;
    virtual void Submit(VK::RenderQueue& q, const Fmatrix& xform, float LOD) override;

protected:
    // Load geometry from OGF_GCONTAINER chunk
    // flags: 0 = normal, VLOAD_NOVERTICES = skip vertex loading (indices only)
    void LoadGeometry(IReader* data, u32 flags = 0);

    // Load fast-path geometry from OGF_FASTPATH chunk
    void LoadFastPath(IReader* data);
};

// ============================================================================
// vkFProgressive - LOD mesh with sliding-window index ranges. Mirrors OGSR
// R4's FProgressive (Layers/xrRender/FProgressive.h) and the monolith
// Vulkan port. Used for level terrain/ground and any other progressive-mesh
// static geometry. Geometry/material loading is identical to vkFVisual; the
// added OGF_SWIDATA chunk picks a (offset,count) IB slice per LOD level.
// ============================================================================
class vkFProgressive : public vkFVisual
{
public:
    u32  sw_count   = 0;       // Number of LOD levels in OGF_SWIDATA
    u32* sw_offsets = nullptr; // Per-LOD: index offset into m_mesh IB span
    u32* sw_counts  = nullptr; // Per-LOD: index count (num_tris * 3)
    u32  last_lod   = 0;

public:
    vkFProgressive();
    virtual ~vkFProgressive();

    virtual void Load(const char* name, IReader* data, u32 flags) override;
    virtual void Release() override;
    virtual void Copy(vkRender_Visual* from) override;
    virtual void Submit(VK::RenderQueue& q, const Fmatrix& xform, float LOD) override;

protected:
    void LoadSlidingWindow(IReader* data);
};

// ============================================================================
// vkFTreeVisual - Tree visual (MT_TREE_ST / MT_TREE_PM). Extends vkFVisual
// for geometry loading. Render() / Submit() are no-ops — actual draw goes
// through VK::CTreeManager (extracted at level load, GPU-driven indirect).
// Stores xform + c_scale.hemi/c_bias.hemi from OGF_TREEDEF2.
// ============================================================================
class vkFTreeVisual : public vkFVisual
{
public:
    // OGF_TREEDEF2 chunk. xform is the per-instance world transform; the
    // hemi scale/bias modulate per-instance lighting (matched to R4 math:
    // halved + multiplied by ps_r__Tree_SBC*1.3333 ≈ 2.0).
    Fmatrix xform;
    struct TreeColor
    {
        Fvector3 rgb;       // Currently unused (R4 reads but does not feed)
        float    hemi;
        float    sun;       // Currently unused
    } c_scale, c_bias;

public:
    vkFTreeVisual();
    virtual ~vkFTreeVisual();

    virtual void Load(const char* name, IReader* data, u32 flags) override;
    virtual void Render(float LOD) override;
    virtual void Submit(VK::RenderQueue& q, const Fmatrix& xform, float LOD) override;

protected:
    void LoadTreeDef(IReader* data);
};

class vkFTreeVisual_ST : public vkFTreeVisual
{
public:
    vkFTreeVisual_ST();
    virtual ~vkFTreeVisual_ST();
};

// Progressive tree — adds OGF_SWIDATA sliding window for LOD selection.
// TreeManager (Session C) reads sw_offsets/sw_counts to pick per-instance LOD.
class vkFTreeVisual_PM : public vkFTreeVisual
{
public:
    u32  sw_count   = 0;
    u32* sw_offsets = nullptr;  // index offset into IB span per LOD
    u32* sw_counts  = nullptr;  // index count (num_tris * 3) per LOD

    vkFTreeVisual_PM();
    virtual ~vkFTreeVisual_PM();

    virtual void Load(const char* name, IReader* data, u32 flags) override;
    virtual void Release() override;

protected:
    void LoadSlidingWindow(IReader* data);
};

// ============================================================================
// vkSkeletonX_ST / vkSkeletonX_PM — skinned-mesh leaves (children of CKinematics).
// Ported from the monolith. STEP B sub-step 1: Load reads the OGF boned vertices,
// analyses bones -> RenderMode, loads indices via the base Load(VLOAD_NOVERTICES),
// and converts vertBoned* -> vertHW_* into a Vulkan VB (_Load_hw_VK). Submit is a
// NO-OP for now: the vertHW VB can't be drawn by the static world pipeline (wrong
// vertex layout + bone-local positions) — the skinned pipeline/deform is sub-step 2.
// ============================================================================
class CKinematics;

class vkSkeletonX_ST : public vkFVisual
{
public:
    enum { RM_SKINNING_SOFT, RM_SINGLE, RM_SKINNING_1B, RM_SKINNING_2B, RM_SKINNING_3B, RM_SKINNING_4B };

    u16          RenderMode = RM_SINGLE;
    u16          BonesUsed  = 0;
    CKinematics* Parent     = nullptr;
    u16          child_idx  = 0;
    union { u32 RMS_boneid; u32 RMS_bonecount; };   // RM_SINGLE: bone id | skinning: max bone id+1

    vkSkeletonX_ST() : RMS_bonecount(0) {}

    void AfterLoad(CKinematics* p, u16 idx) { Parent = p; child_idx = idx; }
    void SetParent(CKinematics* p)          { Parent = p; }

    virtual void Load(const char* name, IReader* data, u32 flags) override;
    virtual void Copy(vkRender_Visual* from) override;
    // Sub-step 1 stopgap: skinned drawing not wired yet — don't push a vertHW VB
    // through the stride-32 world pipeline (garbage). Sub-step 2 replaces this.
    virtual void Submit(VK::RenderQueue& /*q*/, const Fmatrix& /*xform*/, float /*LOD*/) override {}

    void _Load_hw_VK(void* verts, u32 vertType, u32 vertCount);
};

class vkSkeletonX_PM : public vkFProgressive
{
public:
    enum { RM_SKINNING_SOFT, RM_SINGLE, RM_SKINNING_1B, RM_SKINNING_2B, RM_SKINNING_3B, RM_SKINNING_4B };

    u16          RenderMode = RM_SINGLE;
    u16          BonesUsed  = 0;
    CKinematics* Parent     = nullptr;
    u16          child_idx  = 0;
    union { u32 RMS_boneid; u32 RMS_bonecount; };

    vkSkeletonX_PM() : RMS_bonecount(0) {}

    void AfterLoad(CKinematics* p, u16 idx) { Parent = p; child_idx = idx; }
    void SetParent(CKinematics* p)          { Parent = p; }

    virtual void Load(const char* name, IReader* data, u32 flags) override;
    virtual void Copy(vkRender_Visual* from) override;
    virtual void Submit(VK::RenderQueue& /*q*/, const Fmatrix& /*xform*/, float /*LOD*/) override {}

    void _Load_hw_VK(void* verts, u32 vertType, u32 vertCount);
};

// ============================================================================
// vkFHierrarhyVisual - Composite visual with children. Mirrors OGSR R4's
// FHierrarhyVisual (Layers/xrRender/FHierrarhyVisual.h): `children` is a
// public field, no getter. Hierarchy traversal walks the field directly.
// ============================================================================
class vkFHierrarhyVisual : public vkRender_Visual
{
public:
    xr_vector<vkRender_Visual*> children;
    BOOL                        bDontDelete = FALSE;

public:
    vkFHierrarhyVisual();
    virtual ~vkFHierrarhyVisual();

    // ========================================================================
    // vkRender_Visual overrides
    // ========================================================================
    virtual void Load(const char* name, IReader* data, u32 flags) override;
    virtual void Release() override;
    virtual void Copy(vkRender_Visual* from) override;
    virtual void Render(float LOD) override;
    virtual void Submit(VK::RenderQueue& q, const Fmatrix& xform, float LOD) override;
    virtual void Spawn() override;
    virtual void Depart() override;
};

// ============================================================================
// vkFLOD - LOD imposter container (MT_LOD). Extends the hierarchy (so the
// tree manager still recurses its full-detail children) and additionally loads
// the OGF_LODDEF2 billboard facets — 8 pre-rendered camera-direction quads into
// the level_lods atlas. The LOD imposter manager draws the best-facing facet of
// each FLOD for distant foliage density (mirrors R4 r_dsgraph_render_lods).
// ============================================================================
class vkFLOD : public vkFHierrarhyVisual
{
public:
    // Must match R4 FLOD::_vertex byte layout (read raw from OGF_LODDEF2).
    struct LodVertex
    {
        Fvector  v;            // world-space position
        Fvector2 t;            // UV into level_lods atlas
        u32      c_rgb_hemi;   // rgb + hemi
        u8       c_sun;        // sun factor
    };
    struct LodFacet
    {
        LodVertex v[4];        // quad corners
        Fvector   N;           // facet normal (computed at load)
    };

    LodFacet facets[8];
    bool     facetsValid = false;

    vkFLOD();
    virtual ~vkFLOD();
    virtual void Load(const char* name, IReader* data, u32 flags) override;
};

// ============================================================================
// Model type and OGF chunk IDs are defined in fmesh.h
// Include it for MT_*, OGF_* enums and ogf_header struct
// ============================================================================
#include "../../xr_3da/fmesh.h"

// ============================================================================
// MVP factories — only MT_NORMAL (vkFVisual) and MT_HIERRARHY (vkFHierrarhyVisual)
// build real visuals; everything else returns a non-rendering dummy so the
// loader survives unknown types instead of nullptr-dereffing.
// Skinned / progressive / particle / tree / LOD visual types from monolith
// are deferred until those subsystems are ported.
// ============================================================================
vkRender_Visual* vkVisual_Create(u32 type);
vkRender_Visual* vkVisual_CreateDummy();
