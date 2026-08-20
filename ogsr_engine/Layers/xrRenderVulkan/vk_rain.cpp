// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan — rain + thunderbolt. See vk_rain.h.
//
// vkRainRender::Calculate is a line-for-line port of dxRainRender::Calculate
// (drop physics / wrap / renew / Hit). Render builds the drop streak quads and
// the splash detail-model triangles into a per-frame host ring; the lightning
// model + gradients (dxThunderboltRender port) append to the same ring. All
// draws go through the particle pass's pipelines (PBM_BLEND for rain,
// PBM_ADD for lightning) inside the "Rain" pass, after Particles.

#include "stdafx.h"
#include <thread>                  // Rain.h: max_desired_items uses hardware_concurrency
#include <algorithm>               // std::clamp
#include "vk_rain.h"
#include "vk_Particles.h"          // EParticleBlendMode
#include "vk_pass_particles.h"     // ParticlePass::GetPipeline/GetLayout/GetTextureSet
#include "vk_pass_ssao.h"          // DeriveProjTerms (camera basis — Device.vCameraRight unverified on VK)
#include "vk_buffer.h"
#include "vk_command_buffer.h"
#include "vk_R_Backend.h"          // CBackend (pass-through reference only)
#include "HW_Vulkan.h"

#include "../xrRender/FVF.h"
#include "../../xrCDB/Frustum.h"
#include "../../xr_3da/device.h"
#include "../../xr_3da/IGame_Level.h"
#include "../../xr_3da/IGame_Persistent.h"
#include "../../xr_3da/Environment.h"
#include "../../xr_3da/Rain.h"
#include "../../xr_3da/thunderbolt.h"

// Console knobs (vk_console_min.cpp — same vars the DX renderers use).
extern Fvector4 ps_ssfx_rain_1;        // x = drop length, y = drop width, z = speed mult
extern float    ps_r2_no_details_radius;
extern float    ps_r2_no_rain_radius;  // splash-free radius around the camera
extern int      ps_r_rain_debug;       // r_rain_debug — solid red streaks (geometry vs texture triage)
extern int      ps_r_rain_enable;      // r_rain — master rain on/off
extern float    ps_r_rain_sun;         // r_rain_sun — forward-scatter lobe on the sun/moon/lightning direction

namespace {

// ---------------------------------------------------------------------------
// Detail model (.dm) — CPU-side mirror of CDetail::Load (DetailModel.cpp)
// minus the shader/VB/IB creation. Keeps the texture NAME for the particle
// texture cache and the raw verts/indices for per-frame transfer.
// ---------------------------------------------------------------------------
struct vkDM
{
    struct VertIn { Fvector P; float u, v; };

    xr_vector<VertIn> verts;
    xr_vector<u16>    indices;
    shared_str        texture;     // second string in the file (fnT)
    Fsphere           bv_sphere{};
    Fbox              bv_bb{};

    bool Load(IReader* F)
    {
        string256 fnS, fnT;
        F->r_stringZ(fnS, sizeof(fnS));   // shader name — unused (blend mode is fixed per consumer)
        F->r_stringZ(fnT, sizeof(fnT));
        texture = fnT;

        F->r_u32();      // flags
        F->r_float();    // min scale
        F->r_float();    // max scale

        const u32 nV = F->r_u32();
        const u32 nI = F->r_u32();
        if (!nV || !nI || (nI % 3))
            return false;

        verts.resize(nV);
        F->r(verts.data(), nV * sizeof(VertIn));
        indices.resize(nI);
        F->r(indices.data(), nI * sizeof(u16));

        bv_bb.invalidate();
        for (const auto& v : verts)
            bv_bb.modify(const_cast<Fvector&>(v.P));
        bv_bb.getsphere(bv_sphere.P, bv_sphere.R);
        return true;
    }

    // Transform + index-expand straight into the ring (non-indexed triangle
    // list — counts are tiny, splash model is a handful of tris).
    void Emit(const Fmatrix& xform, u32 C, float dv, FVF::LIT* out) const
    {
        for (size_t i = 0; i < indices.size(); ++i)
        {
            const VertIn& src = verts[indices[i]];
            Fvector P;
            xform.transform_tiny(P, src.P);
            out[i].set(P, C, src.u, src.v + dv);
        }
    }
    u32 EmitCount() const { return (u32)indices.size(); }
};

vkDM* LoadDM(const char* fsRoot, const char* name)
{
    IReader* F = FS.r_open(fsRoot, name);
    if (!F) {
        Msg("![VK Rain] can't open detail model '%s'", name);
        return nullptr;
    }
    vkDM* dm = xr_new<vkDM>();
    const bool ok = dm->Load(F);
    FS.r_close(F);
    if (!ok) { xr_delete(dm); Msg("![VK Rain] bad detail model '%s'", name); return nullptr; }
    return dm;
}

// ---------------------------------------------------------------------------
// Shared per-frame vertex ring + draw chunks. Filled by vkRainRender /
// vkThunderboltRender while Pass_Rain runs, drawn once at the end.
// ---------------------------------------------------------------------------
constexpr u32 kFramesInFlight = CVulkanCommandManager::FRAMES_IN_FLIGHT;
constexpr u32 kVtxStride      = sizeof(FVF::LIT);                  // 24
constexpr VkDeviceSize kRingBytes = 2 * 1024 * 1024;               // ~87k verts/frame
constexpr u32 kRingVerts = (u32)(kRingBytes / kVtxStride);

// kind: 0 = generic blend-mode pipeline (thunderbolt), 1 = rain splash
// (tex-alpha shape), 2 = rain drop streak (procedural shape).
struct DrawChunk { shared_str tex; EParticleBlendMode mode; u32 first; u32 count; u8 kind; };

VK::CVulkanBuffer s_Ring[kFramesInFlight];
bool              s_RingInit = false;
u8*               s_RingBase = nullptr;   // mapped while Pass_Rain runs
u32               s_VtxUsed  = 0;
xr_vector<DrawChunk> s_Chunks;
const VK::FrameContext* s_Ctx = nullptr;

FVF::LIT* RingAlloc(u32 count, u32& first)
{
    if (!s_RingBase || s_VtxUsed + count > kRingVerts)
        return nullptr;
    first = s_VtxUsed;
    s_VtxUsed += count;
    return (FVF::LIT*)(s_RingBase + (size_t)first * kVtxStride);
}

void Commit(const shared_str& tex, EParticleBlendMode mode, u32 first, u32 count, u8 kind = 0)
{
    if (count)
        s_Chunks.push_back({ tex, mode, first, count, kind });
}

// CEffect_Rain::Render / CEffect_Thunderbolt::Render take a CBackend& they
// only forward; nothing on this path reads it. Raw storage — no ctor/dtor.
CBackend& DummyBackend()
{
    alignas(CBackend) static char storage[sizeof(CBackend)]{};
    return *reinterpret_cast<CBackend*>(storage);
}

}  // namespace

// ===========================================================================
// vkRainRender — dxRainRender port
// ===========================================================================
class vkRainRender final : public IRainRender
{
public:
    vkRainRender()
    {
        m_DM = LoadDM(fsgame::game_meshes, "dm\\rain.dm");   // splash model ("капля")
        m_DropBounds.P.set(0, 0, 0);
        m_DropBounds.R = 0.f;
        if (m_DM)
            m_DropBounds = m_DM->bv_sphere;
    }
    ~vkRainRender() override
    {
        if (m_DM && m_OwnsDM)
            xr_delete(m_DM);
    }

    void Copy(IRainRender& _in) override
    {
        // FactoryPtr copy semantics: share the (immutable) splash model.
        vkRainRender* src = static_cast<vkRainRender*>(&_in);
        if (m_DM && m_OwnsDM)
            xr_delete(m_DM);
        m_DM = src->m_DM;
        m_OwnsDM = false;
        m_DropBounds = src->m_DropBounds;
    }

    const Fsphere& GetDropBounds() const override { return m_DropBounds; }

    // ----- simulation (dxRainRender::Calculate, verbatim minus stats) -------
    void Calculate(CEffect_Rain& owner) override
    {
        const float factor = g_pGamePersistent->Environment().CurrentEnv->rain_density;
        if (factor < EPS_L)
            return;

        const float _drop_speed = ps_ssfx_rain_1.z;
        const size_t desired_items = iFloor(min_desired_items + (factor * (max_desired_items - min_desired_items)));

        constexpr float b_radius_wrap_sqr = _sqr(source_radius * 1.5f);

        owner.items.reserve(desired_items);
        while (owner.items.size() < desired_items)
        {
            auto& one = owner.items.emplace_back();
            owner.Born(one, source_radius, _drop_speed);
        }

        // build source plane
        Fplane src_plane;
        constexpr Fvector norm{0.f, -1.f, 0.f};
        Fvector upper;
        upper.set(Device.vCameraPosition.x, Device.vCameraPosition.y + source_offset, Device.vCameraPosition.z);
        src_plane.build(upper, norm);

        const Fvector& vEye = Device.vCameraPosition;
        u32 total_cnt = (u32)owner.items.size();

        for (auto& one : owner.items)
        {
            if (one.dwTime_Hit < Device.dwTimeGlobal)
                owner.Hit(one.Phit);

            if (one.dwTime_Life < Device.dwTimeGlobal)
            {
                if (total_cnt <= desired_items)
                    owner.Born(one, source_radius, _drop_speed);
                else
                    total_cnt--;
            }

            const float dt = Device.fTimeDelta;
            one.P.mad(one.D, one.fSpeed * dt);

            Fvector wdir;
            wdir.set(one.P.x - vEye.x, 0, one.P.z - vEye.z);
            float wlen = wdir.square_magnitude();
            if (wlen > b_radius_wrap_sqr)
            {
                wlen = _sqrt(wlen);

                if ((one.P.y - vEye.y) < sink_offset)
                {
                    one.invalidate();   // need born
                }
                else
                {
                    Fvector inv_dir;
                    inv_dir.invert(one.D);
                    wdir.div(wlen);
                    one.P.mad(one.P, wdir, -(wlen + source_radius));

                    if (Fvector src_p; src_plane.intersectRayPoint(one.P, inv_dir, src_p))
                    {
                        const float dist_sqr = one.P.distance_to_sqr(src_p);
                        float height = max_distance;

                        if (owner.RayPick(src_p, one.D, height, collide::rqtBoth))
                        {
                            if (_sqr(height) <= dist_sqr)
                                one.invalidate();                                       // need born
                            else
                                owner.RenewItem(one, height - _sqrt(dist_sqr), TRUE);   // fly to point
                        }
                        else
                        {
                            owner.RenewItem(one, max_distance - _sqrt(dist_sqr), FALSE);
                        }
                    }
                    else
                    {
                        one.invalidate();   // need born
                    }
                }
            }
        }
    }

    // ----- visualization (dxRainRender::Render over the ring) ---------------
    void Render(CBackend& /*unused*/, CEffect_Rain& owner) override
    {
        if (!s_Ctx || !s_RingBase)
            return;

        const float factor = g_pGamePersistent->Environment().CurrentEnv->rain_density;
        if (factor < EPS_L || owner.items.empty())
            return;

        const float _drop_len   = ps_ssfx_rain_1.x;
        const float _drop_width = ps_ssfx_rain_1.y;

        const float factor_visual = factor / 2.f + .5f;
        // ssfx_rain_1.y is tuned for the SSFX refraction shader (0.02 m — the
        // refraction makes sub-pixel streaks visible). Our plain alpha quads
        // need ≥ ~1.5 px on screen or they vanish: widen with distance.
        const float pixAngle = (Device.dwHeight > 0)
            ? deg2rad(Device.fFOV) / float(Device.dwHeight) : 0.0012f;
        const float minWidthPerMeter = pixAngle * 0.9f;   // ~1 px: thin streaks, not "tracers"
        // Colours. SSFX drop output = refracted background + hemi×brightness at
        // alpha 0.4 — the NET effect over the backdrop is a faint hemi-tinted
        // glint. The drop pipeline is ADDITIVE (PBM_ALPHA_ADD), so the vertex
        // colour is ONLY that lift: hemi × brightness; no fog paint-over (that
        // alpha-blend version read as a downpour of tracers vs R4's drizzle).
        const auto* E = g_pGamePersistent->Environment().CurrentEnv;
        // DIRECTIONAL term (r_rain_sun). Hemi alone is flat-lit rain, and it is also
        // rain a thunderbolt cannot reach: the bolt boosts sun/sky/fog colour and
        // points sun_dir at the strike (thunderbolt.cpp:314-325), and hemi is the one
        // field it never writes — so the whole downpour ignored every flash.
        // A falling drop is a water cylinder: it scatters FORWARD, so a downpour
        // lights up when the light is BEHIND it (you look toward the sun/moon/bolt)
        // and goes near-black when the light is behind YOU. That is one lobe on the
        // sun direction, and it delivers the lightning flash through the same path —
        // the bolt swings sun_dir at the camera's part of the sky, so a strike ahead
        // of the player rips through the rain while one behind him only glows.
        // A floor is kept (0.12) so drops never fully vanish side-on.
        Fvector sunD = E->sun_dir;
        sunD.normalize_safe();
        const float fwd = -Device.vCameraDirection.dotproduct(sunD);   // +1 = looking INTO the light
        const float g = _max(0.f, fwd);
        const float lobe = ps_r_rain_sun * (0.12f + 0.88f * g * g * g);
        u32 u_drop_color = color_rgba_f(_min(E->hemi_color.x * 0.6f + E->sun_color.x * lobe, 1.f),
                                        _min(E->hemi_color.y * 0.6f + E->sun_color.y * lobe, 1.f),
                                        _min(E->hemi_color.z * 0.6f + E->sun_color.z * lobe, 1.f),
                                        factor_visual * 0.32f);
        // Splashes stay alpha-blended (their texture alpha works): fog+hemi mix.
        const u32 u_splash_color = color_rgba_f(_min(E->fog_color.x + E->hemi_color.x * 0.4f, 1.f),
                                                _min(E->fog_color.y + E->hemi_color.y * 0.4f, 1.f),
                                                _min(E->fog_color.z + E->hemi_color.z * 0.4f, 1.f),
                                                factor_visual * 0.5f);
        if (ps_r_rain_debug)
            u_drop_color = color_rgba(255, 32, 32, 255);   // alpha 1.0 → solid-quad debug path

        CFrustum view;
        Fmatrix vp = *s_Ctx->viewProj;
        view.CreateFromMatrix(vp, FRUSTUM_P_LRTB | FRUSTUM_P_FAR);

        const Fvector& vEye = Device.vCameraPosition;

        // --- drop streaks: camera-facing quads, 6 verts each ---
        u32 first = 0;
        FVF::LIT* out = RingAlloc((u32)owner.items.size() * 6, first);
        if (!out)
            return;
        u32 vCount = 0;

        for (auto& one : owner.items)
        {
            Fvector& pos_head = one.P;
            Fvector pos_trail;
            pos_trail.mad(pos_head, one.D, -_drop_len * factor_visual);

            // Culling
            Fvector sC, lineD;
            float sR;
            sC.sub(pos_head, pos_trail);
            lineD.normalize(sC);
            sC.mul(.5f);
            sR = sC.magnitude();
            sC.add(pos_trail);

            if (!view.testSphere_dirty(sC, sR))
                continue;

            constexpr Fvector2 UV[2][4]{{{0, 1}, {0, 0}, {1, 1}, {1, 0}}, {{1, 0}, {1, 1}, {0, 0}, {0, 1}}};

            Fvector P0, P1, P2, P3, lineTop, camDir;
            camDir.sub(sC, vEye);
            const float dist = camDir.magnitude();
            camDir.div(_max(dist, 0.01f));
            lineTop.crossproduct(camDir, lineD);
            const float w = _max(_drop_width, dist * minWidthPerMeter);
            const u32 s = one.uv_set;
            P0.mad(pos_trail, lineTop, -w);
            P1.mad(pos_trail, lineTop, w);
            P2.mad(pos_head, lineTop, -w);
            P3.mad(pos_head, lineTop, w);
            // quad (P0,P1,P2,P3) → two tris matching the QuadIB winding
            out[vCount + 0].set(P0, u_drop_color, UV[s][0].x, UV[s][0].y);
            out[vCount + 1].set(P1, u_drop_color, UV[s][1].x, UV[s][1].y);
            out[vCount + 2].set(P2, u_drop_color, UV[s][2].x, UV[s][2].y);
            out[vCount + 3].set(P2, u_drop_color, UV[s][2].x, UV[s][2].y);
            out[vCount + 4].set(P1, u_drop_color, UV[s][1].x, UV[s][1].y);
            out[vCount + 5].set(P3, u_drop_color, UV[s][3].x, UV[s][3].y);
            vCount += 6;
        }
        s_VtxUsed -= (u32)owner.items.size() * 6 - vCount;   // return the unused tail
        Commit("fx\\fx_rain", PBM_BLEND, first, vCount, /*kind*/ 2);

        // --- splash particles (detail model per active particle) ---
        if (!m_DM || !owner.particle_active)
            return;

        const float dt = Device.fTimeDelta;
        const u32 dmVerts = m_DM->EmitCount();
        Fmatrix mXform, mScale;
        u32 splashFirst = 0, splashCount = 0;

        CEffect_Rain::Particle* P = owner.particle_active;
        while (P)
        {
            CEffect_Rain::Particle* next = P->next;

            P->time -= dt;
            if (P->time < 0)
            {
                owner.p_free(P);
                P = next;
                continue;
            }

            if (view.testSphere_dirty(P->bounds.P, P->bounds.R)
                && (fis_zero(ps_r2_no_details_radius) || Device.vCameraPosition.distance_to(P->bounds.P) > ps_r2_no_rain_radius))
            {
                u32 f = 0;
                FVF::LIT* dst = RingAlloc(dmVerts, f);
                if (dst)
                {
                    if (!splashCount)
                        splashFirst = f;
                    const float scale = P->time / particles_time;
                    mScale.scale(scale, scale, scale);
                    mXform.mul_43(P->mXForm, mScale);
                    m_DM->Emit(mXform, u_splash_color, 0.f, dst);
                    splashCount += dmVerts;
                }
            }
            P = next;
        }
        Commit(m_DM->texture, PBM_BLEND, splashFirst, splashCount, /*kind*/ 1);

        static bool s_diag = false;
        if (!s_diag && vCount) {
            s_diag = true;
            Msg("[VK Rain] first frame drawn: drops=%u splashes=%u (factor %.2f)", vCount / 6, splashCount ? splashCount / dmVerts : 0, factor);
        }
    }

private:
    vkDM*   m_DM = nullptr;
    bool    m_OwnsDM = true;
    Fsphere m_DropBounds;
};

// ===========================================================================
// Thunderbolt — dxThunderboltRender / dxThunderboltDescRender / dxFlareRender
// ===========================================================================
class vkFlareRender final : public IFlareRender
{
public:
    shared_str m_Texture;

    void Copy(IFlareRender& _in) override { m_Texture = static_cast<vkFlareRender*>(&_in)->m_Texture; }
    void CreateShader(LPCSTR /*sh*/, LPCSTR tex) override { m_Texture = tex; }
    void DestroyShader() override { m_Texture = nullptr; }
};

class vkThunderboltDescRender final : public IThunderboltDescRender
{
public:
    vkDM* l_model = nullptr;

    ~vkThunderboltDescRender() override { DestroyModel(); }

    void Copy(IThunderboltDescRender& _in) override { l_model = static_cast<vkThunderboltDescRender*>(&_in)->l_model; }

    void CreateModel(LPCSTR m_name) override
    {
        DestroyModel();
        l_model = LoadDM(fsgame::game_meshes, m_name);
    }
    void DestroyModel() override { xr_delete(l_model); }
};

class vkThunderboltRender final : public IThunderboltRender
{
public:
    void Copy(IThunderboltRender&) override {}

    void Render(CBackend& /*unused*/, CEffect_Thunderbolt& owner) override
    {
        if (!s_Ctx || !s_RingBase || !owner.current)
            return;

        // Camera basis: Device.vCameraRight/Top are not maintained on the VK
        // path — derive from the matrix that renders the frame.
        const VK::ProjTerms pt = VK::DeriveProjTerms(*s_Ctx->viewProj);

        // --- lightning model (UV v-anim like dxThunderboltRender) ---
        float dv = owner.lightning_phase * 0.5f;
        dv = (owner.lightning_phase > 0.5f) ? Random.randI(2) * 0.5f : dv;

        const vkThunderboltDescRender* desc = static_cast<vkThunderboltDescRender*>(&*owner.current->m_pRender);
        if (desc && desc->l_model)
        {
            vkDM* dm = desc->l_model;
            u32 first = 0;
            FVF::LIT* out = RingAlloc(dm->EmitCount(), first);
            if (out)
            {
                dm->Emit(owner.current_xform, 0xffffffff, dv, out);
                Commit(dm->texture, PBM_ADD, first, dm->EmitCount());
            }
        }

        // --- gradients: two camera-facing quads (top at the bolt origin,
        //     center at the flash point), additive like the DX sun shader ---
        const SThunderboltDesc::SFlare* gradients[2] = { owner.current->m_GradientTop, owner.current->m_GradientCenter };
        const Fvector centers[2] = { owner.current_xform.c, owner.lightning_center };
        for (int g = 0; g < 2; ++g)
        {
            const auto* G = gradients[g];
            if (!G)
                continue;
            // R4 uses the TOP gradient's opacity for both quads — kept as-is.
            const u32 c_val = std::clamp(iFloor(owner.current->m_GradientTop->fOpacity * owner.lightning_phase * 255.f), 0, 255);
            const u32 c = color_rgba(c_val, c_val, c_val, c_val);

            const vkFlareRender* flare = static_cast<vkFlareRender*>(&*G->m_pFlare);
            if (!flare || !flare->m_Texture.size())
                continue;

            Fvector vecSx, vecSy;
            vecSx.mul(pt.right, G->fRadius.x * owner.lightning_size);
            vecSy.mul(pt.top, -G->fRadius.y * owner.lightning_size);

            const Fvector& C0 = centers[g];
            Fvector q[4];
            q[0].set(C0.x + vecSx.x - vecSy.x, C0.y + vecSx.y - vecSy.y, C0.z + vecSx.z - vecSy.z);
            q[1].set(C0.x + vecSx.x + vecSy.x, C0.y + vecSx.y + vecSy.y, C0.z + vecSx.z + vecSy.z);
            q[2].set(C0.x - vecSx.x - vecSy.x, C0.y - vecSx.y - vecSy.y, C0.z - vecSx.z - vecSy.z);
            q[3].set(C0.x - vecSx.x + vecSy.x, C0.y - vecSx.y + vecSy.y, C0.z - vecSx.z + vecSy.z);

            constexpr Fvector2 uv[4]{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
            u32 first = 0;
            FVF::LIT* out = RingAlloc(6, first);
            if (!out)
                continue;
            out[0].set(q[0], c, uv[0].x, uv[0].y);
            out[1].set(q[1], c, uv[1].x, uv[1].y);
            out[2].set(q[2], c, uv[2].x, uv[2].y);
            out[3].set(q[2], c, uv[2].x, uv[2].y);
            out[4].set(q[1], c, uv[1].x, uv[1].y);
            out[5].set(q[3], c, uv[3].x, uv[3].y);
            Commit(flare->m_Texture, PBM_ADD, first, 6);
        }

        static bool s_diag = false;
        if (!s_diag) { s_diag = true; Msg("[VK Rain] first thunderbolt drawn"); }
    }
};

// ===========================================================================
// Pass
// ===========================================================================
namespace VK {

void Pass_Rain(FrameContext& ctx)
{
    if (!ps_r_rain_enable) return;   // master off (r_rain 0): no drops/splashes/thunderbolt
    if (ctx.cmd == VK_NULL_HANDLE || !ctx.viewProj || !ParticlePass::Ready()
        || !g_pGameLevel || !g_pGamePersistent)
        return;

    CEnvironment& env = g_pGamePersistent->Environment();
    if (!env.eff_Rain || !env.eff_Thunderbolt)
        return;

    // Cheap out: nothing falling, nothing flashing → skip the ring map.
    const bool raining = env.CurrentEnv && env.CurrentEnv->rain_density > EPS_L;

    if (!s_RingInit) {
        for (u32 i = 0; i < kFramesInFlight; ++i)
            s_Ring[i].Create(kRingBytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
        s_RingInit = true;
    }
    CVulkanBuffer& ring = s_Ring[CommandManager.GetCurrentFrame()];
    if (!ring.IsValid())
        return;

    s_Ctx = &ctx;
    s_RingBase = (u8*)ring.Map();
    s_VtxUsed = 0;
    s_Chunks.clear();

    if (s_RingBase)
    {
        // Simulation tick + vertex build. RenderLast's sync path equivalent:
        // Calculate (drop physics, ray-picks, splash spawn) then Render.
        if (raining)
        {
            env.eff_Rain->Calculate();
            env.eff_Rain->Render(DummyBackend());
        }
        env.eff_Thunderbolt->Render(DummyBackend());   // gated on stWorking inside
    }

    if (!s_Chunks.empty())
    {
        VkPipelineLayout layout = ParticlePass::GetLayout();

        // Colour LOAD + depth LOAD (test, no write) — the shared overlay contract.
        VK::BeginOverlayRendering(ctx.cmd, ctx, VK_ATTACHMENT_STORE_OP_DONT_CARE);

        vkCmdPushConstants(ctx.cmd, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Fmatrix), ctx.viewProj);
        VkBuffer vbuf = ring.GetHandle();
        VkDeviceSize voff = 0;
        vkCmdBindVertexBuffers(ctx.cmd, 0, 1, &vbuf, &voff);

        VkPipeline lastPipe = VK_NULL_HANDLE;
        for (const DrawChunk& ch : s_Chunks)
        {
            VkDescriptorSet set = ParticlePass::GetTextureSet(ch.tex.c_str());
            if (set == VK_NULL_HANDLE) {
                // Once per name: a silently skipped chunk looks exactly like
                // "drops don't render" — make the cause visible in the log.
                static xr_vector<shared_str> s_warned;
                if (std::find(s_warned.begin(), s_warned.end(), ch.tex) == s_warned.end()) {
                    s_warned.push_back(ch.tex);
                    Msg("![VK Rain] texture set NULL for '%s' — chunk skipped", ch.tex.c_str());
                }
                continue;
            }
            VkPipeline want = (ch.kind == 2) ? ParticlePass::GetRainDropPipeline()
                            : (ch.kind == 1) ? ParticlePass::GetRainPipeline()
                                             : ParticlePass::GetPipeline(ch.mode);
            if (want == VK_NULL_HANDLE)
                continue;
            if (want != lastPipe) { vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, want); lastPipe = want; }
            vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &set, 0, nullptr);
            vkCmdDraw(ctx.cmd, ch.count, 1, ch.first, 0);
        }

        vkCmdEndRendering(ctx.cmd);
    }

    if (s_RingBase)
    {
        ring.Flush();
        ring.Unmap();
        s_RingBase = nullptr;
    }
    s_Ctx = nullptr;

    // Periodic state line while raining (~every 10 s) — enough to diagnose
    // "no drops / no wetness" reports from the log alone, no debug flags.
    if (raining) {
        static u32 s_stateCd = 0;
        if (s_stateCd == 0) {
            s_stateCd = 600;
            Msg("[VK Rain] state: density=%.2f envWet=%.3f chunks=%u verts=%u",
                env.CurrentEnv->rain_density, env.wetness_factor, (u32)s_Chunks.size(), s_VtxUsed);
        } else --s_stateCd;
    }
}

void RainPass_Destroy()
{
    if (s_RingInit)
    {
        for (u32 i = 0; i < kFramesInFlight; ++i)
            s_Ring[i].Destroy();
        s_RingInit = false;
    }
    s_Chunks.clear();
}

}  // namespace VK

// ===========================================================================
// Factory creators
// ===========================================================================
IRainRender*            VK_CreateRainRender()            { return xr_new<vkRainRender>(); }
IThunderboltRender*     VK_CreateThunderboltRender()     { return xr_new<vkThunderboltRender>(); }
IThunderboltDescRender* VK_CreateThunderboltDescRender() { return xr_new<vkThunderboltDescRender>(); }
IFlareRender*           VK_CreateFlareRender()           { return xr_new<vkFlareRender>(); }
