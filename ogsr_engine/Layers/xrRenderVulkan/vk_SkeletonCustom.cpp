// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

//---------------------------------------------------------------------------
// Vulkan wrapper for SkeletonCustom.cpp
// Compiles OGSR's own xrRender/SkeletonCustom.cpp against the Vulkan backend
// via vk_SkeletonCompat.h type mappings. CRender_Vulkan.h is required because
// SkeletonCustom.cpp drives RImplementation (model_CreateChild / Models /
// model_Duplicate / append_SkeletonWallmark).
//---------------------------------------------------------------------------
#include "stdafx.h"

// FBasicVisualH MUST be defined before anything that transitively pulls
// FBasicVisual.h (CRender_Vulkan.h does, via the dsgraph chain).
#define FBasicVisualH
#include "vk_FBasicVisual.h"

// Full CRender (Vulkan) — SkeletonCustom.cpp calls RImplementation.* and needs
// the complete type, not just the forward decl from vk_SkeletonCompat.h.
#include "CRender_Vulkan.h"
// vkModelPool must be complete: SkeletonCustom.cpp reads RImplementation.Models->
// userdata_override_ini / bone_override_ini (CRender_Vulkan.h only forward-declares it).
#include "vk_ModelPool.h"

// Patched copy of xrRender/SkeletonCustom.cpp — only the CSkeletonX-child
// handling is adapted for the vkFVisual-based vkSkeletonX_* leaves (see the .inl
// header). Everything else is OGSR's own bone/animation logic verbatim.
#include "vk_SkeletonCustom_shared.inl"

// ============================================================================
// Skeleton wallmark bridges (blood decals on NPCs). This TU sees the FULL
// compat context (CKinematics, CSkeletonWallmark, ref_shader), so everything
// that touches those types lives here; vk_wallmarks.cpp talks through the
// opaque VK_SkelWM_* functions below.
//
// ref_shader carrier: CKinematics::AddWallmark demands a ref_shader, but the
// VK path has no shader system — so each decal TEXTURE NAME gets a permanent
// FAKE Shader object used purely as an identity handle. Its ref_count is
// floored at 1 by the registry, so resptr inc/dec is safe and xr_delete in
// _dec() never fires (resptrcode_shader::destroy is inline _set(nullptr) —
// no link dependency on the R4 resource manager).
// ============================================================================
namespace {
    xr_map<shared_str, Shader*> s_skelWMHandleByTex;
    xr_map<Shader*, shared_str> s_skelWMTexByHandle;
    xrCriticalSection           s_skelWMLock;
    // Wallmarks appended THIS frame by CalculateWallmarks (consumed and cleared
    // by VK::Wallmarks::Render the same frame — the intrusive_ptr keeps them
    // alive across the hand-off, mirroring R4's per-frame skeleton_items).
    xr_vector<intrusive_ptr<CSkeletonWallmark>> s_skelWMFrame;
}

static ref_shader vkSkelWM_Handle(const char* tex)
{
    s_skelWMLock.Enter();
    Shader*& h = s_skelWMHandleByTex[shared_str(tex)];
    if (!h) {
        // Can't CONSTRUCT a Shader (its virtual dtor lives in R4's Shader.cpp,
        // not compiled here) — allocate zeroed raw storage and use the pointer
        // purely as an identity handle. resptr inc/dec only touches ref_count.
        h = (Shader*)xr_malloc(sizeof(Shader));
        memset(h, 0, sizeof(Shader));
        h->ref_count = 1;                       // registry floor — never reaches 0
        s_skelWMTexByHandle[h] = shared_str(tex);
    }
    ref_shader R;
    R._set(h);
    s_skelWMLock.Leave();
    return R;
}

extern const char* VK_WallmarkArray_Pick(IWallMarkArray* A);   // vk_RenderFactory.cpp

void CRender::add_SkeletonWallmark(Fmatrix* xf, IKinematics* obj, IWallMarkArray* pArray,
                                   Fvector& start, Fvector& dir, float size)
{
    if (!xf || !obj || !pArray) return;
    // R4 optimization cheat: no skeleton wallmarks beyond 50 m from the viewer.
    if (xf->c.distance_to_sqr(Device.vCameraPosition) > _sqr(50.f)) return;
    const char* tex = VK_WallmarkArray_Pick(pArray);
    if (!tex) return;
    CKinematics* K = smart_cast<CKinematics*>(obj);
    if (!K) return;

    // A bullet mark ON A BODY should read as a wound, not a concrete chip: the
    // game routes the mtl-pair CollideMarks here (the mod resolves NPC armor to
    // ground/concrete pairs — wm_bullet_ground etc). Substitute any non-blood
    // texture with a random blood splat and let it spread wider than the
    // entry hole would (blood soaks into clothing). ONLY for living things —
    // physics destructibles (crates etc.) are skeleton objects too and must
    // keep their material marks (humanoid models live under actors\, mutants
    // under monsters\).
    const char* mdl = K->dbg_name.c_str();
    if (!strstr(tex, "blood") && mdl && (strstr(mdl, "actors") || strstr(mdl, "monsters")))
    {
        static const char* kBlood[] = {
            "wm\\wm_blood_1", "wm\\wm_blood_1_1", "wm\\wm_blood_1_2", "wm\\wm_blood_1_3",
        };
        tex  = kBlood[::Random.randI(4)];
        size *= ::Random.randF(1.6f, 2.6f);
    }

    K->AddWallmark(xf, start, dir, vkSkelWM_Handle(tex), size);
}

void CRender::append_SkeletonWallmark(const intrusive_ptr<CSkeletonWallmark>& wm)
{
    if (!wm) return;
    s_skelWMLock.Enter();
    s_skelWMFrame.push_back(wm);
    s_skelWMLock.Leave();
}

// Per-visual wallmark tick — called from CRender::add_Visual for every visible
// dynamic visual (CalculateWallmarks self-guards per frame; ages TTL and
// appends visible marks via append_SkeletonWallmark above).
void VK_SkelWM_Calculate(IRenderVisual* V, bool hud)
{
    if (CKinematics* K = smart_cast<CKinematics*>(V))
        K->CalculateWallmarks(hud);
}

// Render-side iteration (vk_wallmarks.cpp). Same-thread with the collection
// (Calculate runs before the passes), so reads are unguarded.
u32 VK_SkelWM_Count() { return (u32)s_skelWMFrame.size(); }
const char* VK_SkelWM_Texture(u32 i)
{
    Shader* h = s_skelWMFrame[i]->Shader()._get();
    const auto it = s_skelWMTexByHandle.find(h);
    return (it != s_skelWMTexByHandle.end()) ? it->second.c_str() : nullptr;
}
u32 VK_SkelWM_VCount(u32 i) { return s_skelWMFrame[i]->VCount(); }
void VK_SkelWM_Fill(u32 i, FVF::LIT*& V)
{
    auto& wm = s_skelWMFrame[i];
    if (wm->Parent())
        wm->Parent()->RenderWallmark(wm, V);
}
void VK_SkelWM_EndFrame()
{
    s_skelWMLock.Enter();
    s_skelWMFrame.clear();
    s_skelWMLock.Leave();
}

#undef dxRender_Visual
#undef FBasicVisualH

// Factory for MT_SKELETON_RIGID — real rigid skeleton. Declared extern in
// vk_Visual.cpp. CKinematics derives FHierrarhyVisual : vkRender_Visual, so the
// upcast is valid.
vkRender_Visual* vkCreateKinematics() { return xr_new<CKinematics>(); }
