// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan - vkCParticleGroup: timed collection of child particle effects.
//
// Dedicated Vulkan particle group (NOT the R4 PS::CParticleGroup). Holds one
// child vkCParticleEffect per CPGDef::SEffect and plays/stops them on their
// [Time0,Time1] windows — the core of campfires, explosions, etc. The advanced
// on-play / on-birth / on-dead related/free child spawning is NOT ported (rare),
// so those SEffect flags are ignored.
//
// The compat header (vk_FBasicVisual.h) maps dxRender_Visual -> vkRender_Visual
// so ParticleGroup.h (CPGDef lives there, next to the unused CParticleGroup)
// compiles.

#include "stdafx.h"

#define FBasicVisualH
#include "vk_FBasicVisual.h"
#include "CRender_Vulkan.h"
#include "vk_ModelPool.h"

#include "vk_Particles.h"
#include "../../xrParticles/psystem.h"     // PAPI:: — MUST precede ParticleGroup.h
#include "../xrRender/ParticleGroup.h"     // PS::CPGDef

using namespace PS;

vkCParticleGroup::vkCParticleGroup()
{
    Type = MT_PARTICLE_GROUP;
    m_RT_Flags.zero();
    m_InitialPosition.set(0, 0, 0);
    vis.box.set(Fvector{ -0.1f, -0.1f, -0.1f }, Fvector{ 0.1f, 0.1f, 0.1f });
    vis.box.getsphere(vis.sphere.P, vis.sphere.R);
}

vkCParticleGroup::~vkCParticleGroup()
{
    for (auto* e : m_Items)
        xr_delete(e);
    m_Items.clear();
}

BOOL vkCParticleGroup::Compile(PS::CPGDef* def)
{
    m_Def = def;
    for (auto* e : m_Items)
        xr_delete(e);
    m_Items.clear();

    if (m_Def)
    {
        m_Items.resize(m_Def->m_Effects.size(), nullptr);
        for (size_t i = 0; i < m_Def->m_Effects.size(); ++i)
            m_Items[i] = vkCreateParticleEffect(m_Def->m_Effects[i]->m_EffectName.c_str());
    }
    return TRUE;
}

void vkCParticleGroup::CollectEffects(xr_vector<vkCParticleEffect*>& out)
{
    for (auto* e : m_Items)
        if (e)
            e->CollectEffects(out);
}

void vkCParticleGroup::Play()
{
    m_CurrentTime = 0.f;
    m_RT_Flags.set(flRT_DefferedStop, FALSE);
    m_RT_Flags.set(flRT_Playing, TRUE);
}

void vkCParticleGroup::Stop(BOOL bDefferedStop)
{
    if (bDefferedStop)
        m_RT_Flags.set(flRT_DefferedStop, TRUE);
    else
        m_RT_Flags.set(flRT_Playing, FALSE);

    for (auto* e : m_Items)
        if (e)
            e->Stop(bDefferedStop);
}

void vkCParticleGroup::UpdateParent(const Fmatrix& m, const Fvector& velocity, BOOL bXFORM)
{
    m_InitialPosition = m.c;
    for (auto* e : m_Items)
        if (e)
            e->UpdateParent(m, velocity, bXFORM);
}

void vkCParticleGroup::OnFrame(u32 u_dt)
{
    if (m_Def && m_RT_Flags.is(flRT_Playing))
    {
        const float ct = m_CurrentTime;
        const float f_dt = float(u_dt) / 1000.f;

        for (size_t i = 0; i < m_Def->m_Effects.size(); ++i)
        {
            const CPGDef::SEffect* e = m_Def->m_Effects[i];
            if (!e->m_Flags.is(CPGDef::SEffect::flEnabled)) continue;
            vkCParticleEffect* I = (i < m_Items.size()) ? m_Items[i] : nullptr;
            if (!I) continue;

            if (I->IsPlaying())
            {
                if ((ct <= e->m_Time1) && (ct + f_dt >= e->m_Time1))
                    I->Stop(e->m_Flags.is(CPGDef::SEffect::flDefferedStop));
            }
            else
            {
                if (!m_RT_Flags.is(flRT_DefferedStop))
                    if ((ct <= e->m_Time0) && (ct + f_dt >= e->m_Time0))
                        I->Play();
            }
        }

        m_CurrentTime += f_dt;
        if ((m_CurrentTime > m_Def->m_fTimeLimit) && (m_Def->m_fTimeLimit > 0.f))
            if (!m_RT_Flags.is(flRT_DefferedStop))
                Stop(true);

        bool bPlaying = false;
        Fbox box;
        box.invalidate();
        for (auto* I : m_Items)
        {
            if (!I) continue;
            I->OnFrame(u_dt);
            if (I->IsPlaying())
            {
                bPlaying = true;
                if (I->getVisData().box.is_valid())
                    box.merge(I->getVisData().box);
            }
        }

        if (m_RT_Flags.is(flRT_DefferedStop) && !bPlaying)
            m_RT_Flags.set(flRT_Playing | flRT_DefferedStop, FALSE);

        if (box.is_valid())
        {
            vis.box.set(box);
            vis.box.getsphere(vis.sphere.P, vis.sphere.R);
        }
    }
    else
    {
        vis.box.set(m_InitialPosition, m_InitialPosition);
        vis.box.grow(EPS_L);
        vis.box.getsphere(vis.sphere.P, vis.sphere.R);
    }
}

u32 vkCParticleGroup::ParticlesCount()
{
    u32 c = 0;
    for (auto* e : m_Items)
        if (e)
            c += e->ParticlesCount();
    return c;
}

float vkCParticleGroup::GetTimeLimit()
{
    return m_Def ? m_Def->m_fTimeLimit : -1.f;
}

const shared_str vkCParticleGroup::Name()
{
    if (!m_Def) return shared_str{};
    return m_Def->m_Name;
}

void vkCParticleGroup::SetHudMode(BOOL b)
{
    for (auto* e : m_Items)
        if (e)
            e->SetHudMode(b);
}

BOOL vkCParticleGroup::GetHudMode()
{
    return (!m_Items.empty() && m_Items[0]) ? m_Items[0]->GetHudMode() : FALSE;
}

#undef FBasicVisualH
