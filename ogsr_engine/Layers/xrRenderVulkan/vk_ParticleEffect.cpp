// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan - vkCParticleEffect: PAPI-driven single particle effect.
//
// Dedicated Vulkan particle visual (NOT the R4 PS::CParticleEffect). Simulation
// (OnFrame / Play / Stop / UpdateParent / Compile) is ported verbatim from the
// R4 ParticleEffect.cpp; rendering builds camera-facing FVF::LIT billboards that
// the particle pass (vk_pass_particles.cpp) draws. PAPI handles + action loading
// go through ParticleManager() exactly as in R4.

#include "stdafx.h"

// ParticleEffectDef.h -> Shader.h -> tss_def.h needs the xrD3DDefs Vulkan-branch
// D3D type stubs; the skeleton compat preamble (vk_FBasicVisual.h + CRender_Vulkan.h
// + vk_ModelPool.h) establishes them. Same setup as vk_ParticleEffectDef.cpp.
#define FBasicVisualH
#include "vk_FBasicVisual.h"
#include "CRender_Vulkan.h"
#include "vk_ModelPool.h"

#include "vk_Particles.h"
#include "vk_pass_particles.h"           // ParticlePass::GetTextureSet / Ready

#include "../../xrParticles/psystem.h"
#include "../xrRender/ParticleEffectDef.h"
#include "../xrRender/FVF.h"
#include "../../xr_3da/device.h"         // Device.vCameraTop / vCameraRight / vCameraDirection

using namespace PAPI;
using namespace PS;

// PS library lookups (vk_PSLibrary.cpp).
extern PS::CPEDef* VK_FindPED(const char* name);
extern PS::CPGDef* VK_FindPGD(const char* name);

// ============================================================================
// Birth / death callbacks (same as R4 OnEffectParticleBirth/Dead).
// ============================================================================
static void vk_OnEffectParticleBirth(void* owner, u32, PAPI::Particle& m, u32)
{
    vkCParticleEffect* PE = static_cast<vkCParticleEffect*>(owner);
    if (!PE) return;
    const CPEDef* PED = PE->GetDefinition();
    if (PED)
    {
        if (PED->m_Flags.is(CPEDef::dfRandomFrame))
            m.frame = (u16)iFloor(Random.randI(PED->m_Frame.m_iFrameCount) * 255.f);
        if (PED->m_Flags.is(CPEDef::dfAnimated) && PED->m_Flags.is(CPEDef::dfRandomPlayback) && Random.randI(2))
            m.flags.set(Particle::ANIMATE_CCW, TRUE);
    }
}

static void vk_OnEffectParticleDead(void*, u32, PAPI::Particle&, u32) {}

// ============================================================================
// Billboard generation (ported from R4 ParticleRenderStream + FillSprite,
// emitting 6 verts / triangle-list per particle).
// ============================================================================
namespace {

IC void FillSprite6(FVF::LIT*& pv, const Fvector& T, const Fvector& R,
                    const Fvector& pos, const Fvector2& lt, const Fvector2& rb,
                    float r1, float r2, u32 clr, float sina, float cosa)
{
    Fvector Vr, Vt;
    Vr.x = T.x * r1 * sina + R.x * r1 * cosa;
    Vr.y = T.y * r1 * sina + R.y * r1 * cosa;
    Vr.z = T.z * r1 * sina + R.z * r1 * cosa;
    Vt.x = T.x * r2 * cosa - R.x * r2 * sina;
    Vt.y = T.y * r2 * cosa - R.y * r2 * sina;
    Vt.z = T.z * r2 * cosa - R.z * r2 * sina;

    Fvector a, b, c, d;
    a.sub(Vt, Vr);
    b.add(Vt, Vr);
    c.invert(a);
    d.invert(b);

    // Quad corners: d(lt.x,rb.y) a(lt.x,lt.y) c(rb.x,rb.y) b(rb.x,lt.y)
    pv->set(d.x + pos.x, d.y + pos.y, d.z + pos.z, clr, lt.x, rb.y); pv++;
    pv->set(a.x + pos.x, a.y + pos.y, a.z + pos.z, clr, lt.x, lt.y); pv++;
    pv->set(c.x + pos.x, c.y + pos.y, c.z + pos.z, clr, rb.x, rb.y); pv++;
    pv->set(a.x + pos.x, a.y + pos.y, a.z + pos.z, clr, lt.x, lt.y); pv++;
    pv->set(b.x + pos.x, b.y + pos.y, b.z + pos.z, clr, rb.x, lt.y); pv++;
    pv->set(c.x + pos.x, c.y + pos.y, c.z + pos.z, clr, rb.x, rb.y); pv++;
}

IC void FillSprite6(FVF::LIT*& pv, const Fvector& pos, const Fvector& dir,
                    const Fvector2& lt, const Fvector2& rb,
                    float r1, float r2, u32 clr, float sina, float cosa)
{
    const Fvector& T = dir;
    Fvector R;
    R.crossproduct(T, Device.vCameraDirection).normalize_safe();
    FillSprite6(pv, T, R, pos, lt, rb, r1, r2, clr, sina, cosa);
}

}  // namespace

// ============================================================================
// vkCParticleEffect
// ============================================================================
vkCParticleEffect::vkCParticleEffect()
{
    Type = MT_PARTICLE_EFFECT;
    m_RT_Flags.zero();
    m_XFORM.identity();
    m_InitialPosition.set(0, 0, 0);

    m_HandleEffect = ParticleManager()->CreateEffect(1);
    VERIFY(m_HandleEffect >= 0);
    m_HandleActionList = ParticleManager()->CreateActionList();
    VERIFY(m_HandleActionList >= 0);

    vis.box.set(Fvector{ -0.1f, -0.1f, -0.1f }, Fvector{ 0.1f, 0.1f, 0.1f });
    vis.box.getsphere(vis.sphere.P, vis.sphere.R);
}

vkCParticleEffect::~vkCParticleEffect()
{
    ParticleManager()->DestroyEffect(m_HandleEffect);
    ParticleManager()->DestroyActionList(m_HandleActionList);
    m_HandleEffect = m_HandleActionList = -1;
}

EParticleBlendMode vkCParticleEffect::DetermineBlendMode() const
{
    if (!m_Def || !m_Def->m_ShaderName.size())
        return PBM_ADD;

    string256 lower;
    xr_strcpy(lower, sizeof(lower), m_Def->m_ShaderName.c_str());
    _strlwr(lower);

    if (strstr(lower, "distort"))                                             return PBM_DISTORT;
    if (strstr(lower, "alpha_add") || strstr(lower, "alpha-add") || strstr(lower, "aadd")) return PBM_ALPHA_ADD;
    if (strstr(lower, "add"))                                                 return PBM_ADD;
    if (strstr(lower, "mul_2x") || strstr(lower, "mul2x"))                    return PBM_MUL_2X;
    if (strstr(lower, "mul"))                                                 return PBM_MUL;
    if (strstr(lower, "set"))                                                 return PBM_SET;
    if (strstr(lower, "blend"))                                               return PBM_BLEND;
    return PBM_ADD;
}

BOOL vkCParticleEffect::Compile(PS::CPEDef* def)
{
    m_Def = def;
    if (!m_Def) return FALSE;

    m_BlendMode = DetermineBlendMode();

    IReader F(m_Def->m_Actions.pointer(), m_Def->m_Actions.size());
    ParticleManager()->LoadActions(m_HandleActionList, F, m_Def->m_copFormat);
    ParticleManager()->SetMaxParticles(m_HandleEffect, m_Def->m_MaxParticles);
    ParticleManager()->SetCallback(m_HandleEffect, vk_OnEffectParticleBirth, vk_OnEffectParticleDead, this, 0);

    if (m_Def->m_Flags.is(CPEDef::dfTimeLimit))
        m_fElapsedLimit = m_Def->m_fTimeLimit;
    return TRUE;
}

void vkCParticleEffect::Play()
{
    if (m_Def && m_Def->m_Flags.is(CPEDef::dfTimeLimit))
        m_fElapsedLimit = m_Def->m_fTimeLimit;
    m_RT_Flags.set(flRT_DefferedStop, FALSE);
    m_RT_Flags.set(flRT_Playing, TRUE);
    ParticleManager()->PlayEffect(m_HandleEffect, m_HandleActionList);
}

void vkCParticleEffect::Stop(BOOL bDefferedStop)
{
    ParticleManager()->StopEffect(m_HandleEffect, m_HandleActionList, bDefferedStop);
    if (bDefferedStop)
        m_RT_Flags.set(flRT_DefferedStop, TRUE);
    else
        m_RT_Flags.set(flRT_Playing, FALSE);
}

void vkCParticleEffect::UpdateParent(const Fmatrix& m, const Fvector& velocity, BOOL bXFORM)
{
    m_RT_Flags.set(flRT_XFORM, bXFORM);
    if (bXFORM)
        m_XFORM.set(m);
    else
    {
        m_InitialPosition = m.c;
        ParticleManager()->Transform(m_HandleActionList, m, velocity);
    }
}

void vkCParticleEffect::OnFrame(u32 frame_dt)
{
    if (m_Def && m_RT_Flags.is(flRT_Playing))
    {
        m_MemDT += frame_dt;

        int StepCount = 0;
        const u32 uDT_STEP = m_Def->GetUStep();
        const float fDT_STEP = m_Def->GetFStep();
        if (uDT_STEP && m_MemDT >= (s32)uDT_STEP)
        {
            StepCount = m_MemDT / uDT_STEP;
            m_MemDT = m_MemDT % uDT_STEP;
            clamp(StepCount, 0, 3);
        }

        for (; StepCount; StepCount--)
        {
            if (m_Def->m_Flags.is(CPEDef::dfTimeLimit))
            {
                if (!m_RT_Flags.is(flRT_DefferedStop))
                {
                    m_fElapsedLimit -= fDT_STEP;
                    if (m_fElapsedLimit < 0.f)
                    {
                        m_fElapsedLimit = m_Def->m_fTimeLimit;
                        Stop(true);
                        break;
                    }
                }
            }
            ParticleManager()->Update(m_HandleEffect, m_HandleActionList, fDT_STEP);

            Particle* particles;
            u32 p_cnt;
            ParticleManager()->GetParticles(m_HandleEffect, particles, p_cnt);

            if (m_Def->m_Flags.is(CPEDef::dfFramed | CPEDef::dfAnimated))
                m_Def->ExecuteAnimate(particles, p_cnt, fDT_STEP);
            if (m_Def->m_Flags.is(CPEDef::dfCollision))
                m_Def->ExecuteCollision(particles, p_cnt, fDT_STEP, nullptr);

            if (p_cnt)
            {
                vis.box.invalidate();
                float p_size = 0.f;
                for (u32 i = 0; i < p_cnt; i++)
                {
                    Particle& m = particles[i];
                    vis.box.modify((Fvector&)m.pos);
                    if (m.size.x > p_size) p_size = m.size.x;
                    if (m.size.y > p_size) p_size = m.size.y;
                    if (m.size.z > p_size) p_size = m.size.z;
                }
                vis.box.grow(p_size);
                vis.box.getsphere(vis.sphere.P, vis.sphere.R);
            }
            if (m_RT_Flags.is(flRT_DefferedStop) && (0 == p_cnt))
            {
                m_RT_Flags.set(flRT_Playing | flRT_DefferedStop, FALSE);
                break;
            }
        }
    }
    else
    {
        vis.box.set(m_InitialPosition, m_InitialPosition);
        vis.box.grow(EPS_L);
        vis.box.getsphere(vis.sphere.P, vis.sphere.R);
    }
}

u32   vkCParticleEffect::ParticlesCount() { return ParticleManager()->GetParticlesCount(m_HandleEffect); }

float vkCParticleEffect::GetTimeLimit()
{
    if (!m_Def) return -1.f;
    return m_Def->m_Flags.is(CPEDef::dfTimeLimit) ? m_Def->m_fTimeLimit : -1.f;
}

const shared_str vkCParticleEffect::Name()
{
    if (!m_Def) return shared_str{};
    return m_Def->m_Name;
}

void vkCParticleEffect::CollectEffects(xr_vector<vkCParticleEffect*>& out)
{
    if (m_Def && m_Def->m_Flags.is(CPEDef::dfSprite))
        out.push_back(this);
}

VkDescriptorSet vkCParticleEffect::ResolveTextureSet()
{
    if (!m_TextureResolved && VK::ParticlePass::Ready())
    {
        m_TextureResolved = true;
        if (m_Def && m_Def->m_TextureName.size())
            m_TextureSet = VK::ParticlePass::GetTextureSet(m_Def->m_TextureName.c_str());
    }
    return m_TextureSet;
}

u32 vkCParticleEffect::BuildVertices(FVF::LIT* dst, u32 maxVerts)
{
    if (!dst || !m_Def) return 0;

    Particle* particles;
    u32 p_cnt;
    ParticleManager()->GetParticles(m_HandleEffect, particles, p_cnt);
    if (!p_cnt || !particles) return 0;

    if (p_cnt > maxVerts / 6) p_cnt = maxVerts / 6;
    if (!p_cnt) return 0;

    FVF::LIT* pv = dst;

    float sina = 0.f, cosa = 0.f;
    // Sentinel that no real rot.x equals → first particle always computes
    // sin/cos. R4 uses exactly 0xFFFFFFFF here ("some particles won't play"
    // otherwise — see ParticleRenderStream).
    float angle = float(0xFFFFFFFF);

    for (u32 i = 0; i < p_cnt; i++)
    {
        Particle& m = particles[i];

        Fvector2 lt, rb;
        lt.set(0.f, 0.f);
        rb.set(1.f, 1.f);

        if (angle != m.rot.x)
        {
            angle = m.rot.x;
            sina = _sin(angle);
            cosa = _cos(angle);
        }

        if (m_Def->m_Flags.is(CPEDef::dfFramed))
            m_Def->m_Frame.CalculateTC(iFloor(float(m.frame) / 255.f), lt, rb);

        float r_x = m.size.x * 0.5f;
        float r_y = m.size.y * 0.5f;
        float speed = 0.f;
        bool  speed_calculated = false;

        if (m_Def->m_Flags.is(CPEDef::dfVelocityScale))
        {
            speed = m.vel.magnitude();
            speed_calculated = true;
            r_x += speed * m_Def->m_VelocityScale.x;
            r_y += speed * m_Def->m_VelocityScale.y;
        }

        const u32 clr = color_rgba_f(m.colorR, m.colorG, m.colorB, m.colorA);

        if (m_Def->m_Flags.is(CPEDef::dfAlignToPath))
        {
            if (!speed_calculated)
                speed = m.vel.magnitude();

            if ((speed < EPS_S) && m_Def->m_Flags.is(CPEDef::dfWorldAlign))
            {
                Fmatrix M;
                M.setXYZ(m_Def->m_APDefaultRotation);
                if (m_RT_Flags.is(flRT_XFORM))
                {
                    Fvector p;
                    m_XFORM.transform_tiny(p, m.pos);
                    M.mulA_43(m_XFORM);
                    FillSprite6(pv, M.k, M.i, p, lt, rb, r_x, r_y, clr, sina, cosa);
                }
                else
                    FillSprite6(pv, M.k, M.i, m.pos, lt, rb, r_x, r_y, clr, sina, cosa);
            }
            else if ((speed >= EPS_S) && m_Def->m_Flags.is(CPEDef::dfFaceAlign))
            {
                Fmatrix M;
                M.identity();
                M.k.div(m.vel, speed);
                M.j.set(0, 1, 0);
                if (_abs(M.j.dotproduct(M.k)) > .99f)
                    M.j.set(0, 0, 1);
                M.i.crossproduct(M.j, M.k); M.i.normalize();
                M.j.crossproduct(M.k, M.i); M.j.normalize();
                if (m_RT_Flags.is(flRT_XFORM))
                {
                    Fvector p;
                    m_XFORM.transform_tiny(p, m.pos);
                    M.mulA_43(m_XFORM);
                    FillSprite6(pv, M.j, M.i, p, lt, rb, r_x, r_y, clr, sina, cosa);
                }
                else
                    FillSprite6(pv, M.j, M.i, m.pos, lt, rb, r_x, r_y, clr, sina, cosa);
            }
            else
            {
                Fvector dir;
                if (speed >= EPS_S)
                    dir.div(m.vel, speed);
                else
                    dir.setHP(-m_Def->m_APDefaultRotation.y, -m_Def->m_APDefaultRotation.x);
                if (m_RT_Flags.is(flRT_XFORM))
                {
                    Fvector p, dd;
                    m_XFORM.transform_tiny(p, m.pos);
                    m_XFORM.transform_dir(dd, dir);
                    FillSprite6(pv, p, dd, lt, rb, r_x, r_y, clr, sina, cosa);
                }
                else
                    FillSprite6(pv, m.pos, dir, lt, rb, r_x, r_y, clr, sina, cosa);
            }
        }
        else
        {
            if (m_RT_Flags.is(flRT_XFORM))
            {
                Fvector p;
                m_XFORM.transform_tiny(p, m.pos);
                FillSprite6(pv, Device.vCameraTop, Device.vCameraRight, p, lt, rb, r_x, r_y, clr, sina, cosa);
            }
            else
                FillSprite6(pv, Device.vCameraTop, Device.vCameraRight, m.pos, lt, rb, r_x, r_y, clr, sina, cosa);
        }
    }

    return p_cnt * 6;
}

// ============================================================================
// Factories
// ============================================================================
vkCParticleEffect* vkCreateParticleEffect(const char* ped_name)
{
    PS::CPEDef* ped = VK_FindPED(ped_name);
    if (!ped) return nullptr;
    auto* e = xr_new<vkCParticleEffect>();
    e->Compile(ped);
    return e;
}

vkParticleVisual* vkCreateParticle(const char* name)
{
    if (PS::CPEDef* ped = VK_FindPED(name))
    {
        auto* e = xr_new<vkCParticleEffect>();
        e->Compile(ped);
        return e;
    }
    if (PS::CPGDef* pgd = VK_FindPGD(name))
    {
        auto* g = xr_new<vkCParticleGroup>();
        g->Compile(pgd);
        return g;
    }
    return nullptr;
}

#undef FBasicVisualH
