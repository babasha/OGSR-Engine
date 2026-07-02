// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// GPU-Driven Particles — Phase 3 step 2 piece #3: real-emitter feed.
//
// Walks the live .pe particle visuals each frame and appends one EmitterSample
// per playing smoke effect that translates to a GPU program. This lets level
// smoke (campfires, anomalies, ...) drive the GPU particle system automatically
// — no manual gp_spawn. Kept in its own TU so the PS/visual includes stay out
// of the clean GPU module (vk_gpu_particles.cpp).
//
// Runs on the render thread inside DispatchComputeAndDraw — the same
// g_DynamicVisuals walk CollectSmokeParticles already does, so no extra locking.

#include "stdafx.h"
#include "vk_Particles.h"
#include "vk_pass_world.h"            // g_DynamicVisuals / DynVisual
#include "vk_gpu_particles.h"         // EmitterSample / Enabled / ResolveProgram / ProgramRate
#include "../../xr_3da/fmesh.h"       // MT_PARTICLE_EFFECT / MT_PARTICLE_GROUP

void VK_GP_CollectWorldEmitters(xr_vector<VK::GPUParticles::EmitterSample>& out, float dt)
{
    using namespace VK::GPUParticles;
    if (!Enabled() || dt <= 0.0f) return;

    static xr_vector<vkCParticleEffect*> leaves;
    for (const VK::DynVisual& d : VK::g_DynamicVisuals)
    {
        if (!d.vis) continue;
        const u32 t = d.vis->Type;
        if (t != MT_PARTICLE_EFFECT && t != MT_PARTICLE_GROUP) continue;

        leaves.clear();
        static_cast<vkParticleVisual*>(d.vis)->CollectEffects(leaves);
        for (vkCParticleEffect* e : leaves)
        {
            if (!e || !e->IsPlaying()) continue;
            if (!e->GpuClaimed()) continue;          // not GPU-routed smoke (or failed → CPU)

            const float rate = ProgramRate(e->m_GpuProgram);
            if (rate <= 0.0f) continue;

            e->m_GpuAccum += rate * dt;
            u32 cnt = (u32)e->m_GpuAccum;
            e->m_GpuAccum -= (float)cnt;
            if (cnt == 0) continue;

            EmitterSample es;
            es.program = (u32)e->m_GpuProgram;
            es.count   = cnt;
            es.pos     = e->m_RT_Flags.is(vkCParticleEffect::flRT_XFORM) ? e->m_XFORM.c
                                                                         : e->m_InitialPosition;
            es.vel.set(0, 0, 0);
            out.push_back(es);
        }
    }
}
