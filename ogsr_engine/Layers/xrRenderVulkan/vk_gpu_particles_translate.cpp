// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// GPU-Driven Particles — Phase 3: real .pe -> GPU action-program translator.
//
// Walks a CPEDef's loaded PAPI action list and serialises it into the same
// VK::GPUParticles::Program struct the GPU interpreter already runs (Phase 2a).
// The struct layouts were designed for exactly this — domains copy verbatim
// (pDomain field storage matches gp_generate) and our GP_* action ids equal
// PActionEnum, so an action's `type` is its GPU opcode directly.
//
// Kept in a separate TU so the heavy ParticleEffectDef.h / Shader.h include
// chain stays out of the clean GPU module (vk_gpu_particles.cpp). The output
// Program is a POD declared in vk_gpu_particles.h with no PAPI dependency.
#include "stdafx.h"

// ParticleEffectDef.h -> Shader.h -> tss_def.h needs the xrD3DDefs Vulkan-branch
// D3D type stubs; the skeleton compat preamble establishes them (same setup as
// vk_ParticleEffect.cpp / vk_ParticleEffectDef.cpp).
#define FBasicVisualH
#include "vk_FBasicVisual.h"
#include "CRender_Vulkan.h"
#include "vk_ModelPool.h"

#include "../../xrParticles/psystem.h"
#include "../../xrParticles/particle_manager.h"
#include "../../xrParticles/particle_actions_collection.h"
#include "../xrRender/ParticleEffectDef.h"

#include "vk_gpu_particles.h"

using namespace PAPI;

// PS library lookup (vk_PSLibrary.cpp), same extern vk_ParticleEffect.cpp uses.
extern PS::CPEDef* VK_FindPED(const char* name);

namespace VK { namespace GPUParticles {

namespace {
    // pDomain field storage matches gp_generate's expectations byte-for-byte
    // (see particle_core.cpp pDomain ctor/Generate vs gp_common.glsl), so a
    // plain field copy is a faithful translation for every domain type.
    void FillGenDomain(GenDomain& g, const pDomain& d)
    {
        g = {};
        g.dtype = (u32)d.type;
        g.p1[0] = d.p1.x; g.p1[1] = d.p1.y; g.p1[2] = d.p1.z;
        g.p2[0] = d.p2.x; g.p2[1] = d.p2.y; g.p2[2] = d.p2.z;
        g.u[0]  = d.u.x;  g.u[1]  = d.u.y;  g.u[2]  = d.u.z;
        g.v[0]  = d.v.x;  g.v[1]  = d.v.y;  g.v[2]  = d.v.z;
        g.radii[0] = d.radius1;    g.radii[1] = d.radius2;
        g.radii[2] = d.radius1Sqr; g.radii[3] = d.radius2Sqr;
    }

    // Actions the gp_simulate interpreter understands (Phase 2a Tier-A). An
    // action outside this set still gets serialised (preserving list order/count)
    // but no-ops in the shader's default case — counted + logged here so the
    // mismatch with the CPU path is honest, not silent.
    bool IsInterpreted(u32 t)
    {
        switch (t) {
        case PADampingID: case PAGravityID: case PAKillOldID: case PAMoveID:
        case PASpeedLimitID: case PATargetColorID: case PATargetSizeID:
        case PATargetVelocityID:
            return true;
        default:
            return false;
        }
    }
}

// Translate a named .pe into the GPU Program. Returns false (caller keeps its
// current program) if the effect is missing, empty, or has no Source action.
bool TranslateEffect(const char* pedName, Program& out, float& outEmitRate, TexDesc& outTex)
{
    outTex = {};
    PS::CPEDef* def = VK_FindPED(pedName);
    if (!def) { Msg("![VK GP] gp_mirror: effect '%s' not found", pedName); return false; }
    if (!def->m_Actions.size()) { Msg("![VK GP] gp_mirror: '%s' has no actions", pedName); return false; }

    // Sprite texture + atlas frame metadata (mirrors the CPU billboard path:
    // dfFramed → CalculateTC sub-rects; dfAnimated advances frames over life).
    if (def->m_TextureName.size())
        xr_strcpy(outTex.name, sizeof(outTex.name), def->m_TextureName.c_str());
    if (def->m_Flags.is(PS::CPEDef::dfFramed))      outTex.flags |= 1u;
    if (def->m_Flags.is(PS::CPEDef::dfAnimated))    outTex.flags |= 2u;
    if (def->m_Flags.is(PS::CPEDef::dfRandomFrame)) outTex.flags |= 4u;
    outTex.frameDimX   = (u32)_max(def->m_Frame.m_iFrameDimX, 1);
    outTex.frameCount  = (u32)_max(def->m_Frame.m_iFrameCount, 1);
    outTex.frameW      = def->m_Frame.m_fTexSize.x;
    outTex.frameH      = def->m_Frame.m_fTexSize.y;
    outTex.frameSpeed  = def->m_Frame.m_fSpeed;

    auto* mgr = static_cast<CParticleManager*>(ParticleManager());
    const int al = mgr->CreateActionList();
    if (al < 0) { Msg("![VK GP] gp_mirror: CreateActionList failed"); return false; }

    {
        IReader F(def->m_Actions.pointer(), def->m_Actions.size());
        mgr->LoadActions(al, F, def->m_copFormat);
    }
    ParticleActions* list = mgr->GetActionListPtr(al);
    if (!list || list->empty()) {
        mgr->DestroyActionList(al);
        Msg("![VK GP] gp_mirror: '%s' produced no runtime actions", pedName);
        return false;
    }

    out = {};

    // Particle lifetime (sc.w / vel_life.w, used by time-scaled actions) comes
    // from KillOld's age_limit — resolve it before the Source fills its scalars.
    float lifetime = 100.0f;
    bool  haveKill = false, haveMove = false;
    for (PAVecIt it = list->begin(); it != list->end(); ++it) {
        if ((*it)->type == PAKillOldID) {
            PAKillOld* k = static_cast<PAKillOld*>(*it);
            if (!k->kill_less_than) { lifetime = k->age_limit; haveKill = true; }
        }
    }

    bool haveSource = false;
    u32  n = 0, skipped = 0, overflow = 0;

    auto push = [&](const ActionRec& a) {
        if (n < kMaxActions) out.actions[n++] = a;
        else ++overflow;
    };

    for (PAVecIt it = list->begin(); it != list->end(); ++it) {
        ParticleAction* pa = *it;
        switch (pa->type) {

        case PASourceID: {
            PASource* s = static_cast<PASource*>(pa);
            EmitDesc& e = out.emit;
            // Local-space domains: emit samples around the origin, the dispatch
            // offsets by the spawn position (camera, for this mirror path).
            FillGenDomain(e.posDom,   s->positionL);
            FillGenDomain(e.velDom,   s->velocityL);
            FillGenDomain(e.sizeDom,  s->size);
            FillGenDomain(e.colorDom, s->color);
            FillGenDomain(e.rotDom,   s->rot);
            e.sc[0] = s->alpha;
            e.sc[1] = s->age;
            e.sc[2] = s->age_sigma;
            e.sc[3] = lifetime;
            e.parentVel[0] = s->parent_vel.x;
            e.parentVel[1] = s->parent_vel.y;
            e.parentVel[2] = s->parent_vel.z;
            outEmitRate  = (s->particle_rate > 0.f) ? s->particle_rate : 1.0f;
            out.emitRate = outEmitRate;
            haveSource = true;
            break;
        }

        case PAGravityID: {
            PAGravity* g = static_cast<PAGravity*>(pa);
            ActionRec a{}; a.atype = PAGravityID;
            a.a[0] = g->directionL.x; a.a[1] = g->directionL.y; a.a[2] = g->directionL.z;
            push(a);
            break;
        }

        case PAMoveID: {
            ActionRec a{}; a.atype = PAMoveID; push(a);
            haveMove = true;
            break;
        }

        case PADampingID: {
            PADamping* d = static_cast<PADamping*>(pa);
            ActionRec a{}; a.atype = PADampingID;
            a.a[0] = d->damping.x; a.a[1] = d->damping.y; a.a[2] = d->damping.z;
            a.b[0] = d->vlowSqr;   a.b[1] = d->vhighSqr;
            push(a);
            break;
        }

        case PASpeedLimitID: {
            PASpeedLimit* sl = static_cast<PASpeedLimit*>(pa);
            ActionRec a{}; a.atype = PASpeedLimitID;
            a.a[0] = sl->min_speed; a.a[1] = sl->max_speed;
            push(a);
            break;
        }

        case PATargetColorID: {
            PATargetColor* t = static_cast<PATargetColor*>(pa);
            ActionRec a{}; a.atype = PATargetColorID;
            a.a[0] = t->color.x; a.a[1] = t->color.y; a.a[2] = t->color.z; a.a[3] = t->alpha;
            a.b[0] = t->scale;   a.b[1] = t->timeFrom; a.b[2] = t->timeTo;
            push(a);
            break;
        }

        case PATargetSizeID: {
            PATargetSize* t = static_cast<PATargetSize*>(pa);
            ActionRec a{}; a.atype = PATargetSizeID;
            a.a[0] = t->size.x;  a.a[1] = t->size.y;     // GPU size is 2D
            a.b[0] = t->scale.x; a.b[1] = t->scale.y;
            push(a);
            break;
        }

        case PATargetVelocityID: {
            PATargetVelocity* t = static_cast<PATargetVelocity*>(pa);
            ActionRec a{}; a.atype = PATargetVelocityID;
            a.a[0] = t->velocityL.x; a.a[1] = t->velocityL.y; a.a[2] = t->velocityL.z;
            a.b[0] = t->scale;
            push(a);
            break;
        }

        case PAKillOldID: {
            PAKillOld* k = static_cast<PAKillOld*>(pa);
            ActionRec a{}; a.atype = PAKillOldID;
            a.a[0] = k->age_limit; a.a[1] = k->kill_less_than ? 1.0f : 0.0f;
            push(a);
            break;
        }

        default:
            // Preserve order/count; the shader no-ops unknown opcodes.
            { ActionRec a{}; a.atype = (u32)pa->type; push(a); }
            if (!IsInterpreted((u32)pa->type)) ++skipped;
            break;
        }
    }

    out.actionCount = n;
    mgr->DestroyActionList(al);

    if (!haveSource) {
        Msg("![VK GP] gp_mirror: '%s' has no Source action — cannot emit", pedName);
        return false;
    }
    if (overflow)
        Msg("~[VK GP] gp_mirror: '%s' has >%u actions — %u dropped", pedName, kMaxActions, overflow);
    if (!haveMove)
        Msg("~[VK GP] gp_mirror: '%s' has no Move action — particles won't integrate", pedName);
    if (!haveKill)
        Msg("~[VK GP] gp_mirror: '%s' has no KillOld — assuming %.0fs lifetime", pedName, lifetime);

    Msg("[VK GPUParticles] mirrored '%s': %u actions (%u not interpreted), emitRate %.0f/s, life %.1fs",
        pedName, n, skipped, out.emitRate, lifetime);
    return true;
}

}}  // namespace VK::GPUParticles

#undef FBasicVisualH
