// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan — dynamic light registry. See vk_light.h.
#include "stdafx.h"
#include "vk_light.h"
#include "../../xr_3da/device.h"   // Device.dwFrame (per-frame idempotence)

#include <algorithm>

extern int ps_r_light_debug;   // r_light_debug — log the collected light set every ~2 s

namespace VK {

namespace {
    // All vkLight instances ever created and not yet destroyed (active or not).
    // Game code holds them via ref_light (refcounted) — the destructor
    // unregisters. Count is tens, not thousands; O(n) scans are fine.
    xr_vector<vkLight*> s_registry;

    // Lights farther than (range + this) from the eye can't visibly contribute.
    constexpr float kCullDistance = 120.f;

    // Point lights only get the shadow budget when reasonably close/big — the
    // torch's tiny companion omni or far lamps shouldn't steal it from a fire.
    constexpr float kPointShadowMaxDist = 30.f;
    constexpr float kPointShadowMinRange = 3.f;
}

vkLight::vkLight()  { s_registry.push_back(this); }
vkLight::~vkLight()
{
    auto it = std::find(s_registry.begin(), s_registry.end(), this);
    if (it != s_registry.end()) s_registry.erase(it);
}

namespace Lights {

const FrameLights& CollectFrame(const Fvector& eye)
{
    static FrameLights s_frame;
    static u32 s_frameTag = u32(-1);
    if (s_frameTag == Device.dwFrame) return s_frame;   // already built this frame
    s_frameTag = Device.dwFrame;
    s_frame = FrameLights{};

    struct Cand { vkLight* l; float d2; };
    static xr_vector<Cand> cands;   // scratch — render is single-threaded here
    cands.clear();

    for (vkLight* l : s_registry)
    {
        if (!l->active) continue;
        if (l->type != IRender_Light::POINT && l->type != IRender_Light::SPOT
            && l->type != IRender_Light::OMNIPART) continue;
        if (l->range < 0.05f) continue;
        const float cull = l->range + kCullDistance;
        const float d2 = eye.distance_to_sqr(l->pos);
        if (d2 > cull * cull) continue;
        cands.push_back({ l, d2 });
    }

    // NEAREST-FIRST ordering is REQUIRED: the UBO (foliage + the non-clustered
    // fallback) reads only the first kMaxLights, so those must be the closest
    // lights — otherwise a near light past slot 16 in registry order (e.g. the
    // flashlight) silently drops from the UBO path. The clustered SSBO reads all
    // of them; the order is harmless there. So sort by distance ALWAYS, capping
    // at kMaxClusterLights (partial_sort when there are more than the cap).
    auto byDist = [](const Cand& a, const Cand& b) { return a.d2 < b.d2; };
    if (cands.size() > kMaxClusterLights) {
        std::partial_sort(cands.begin(), cands.begin() + kMaxClusterLights, cands.end(), byDist);
        cands.resize(kMaxClusterLights);
    } else {
        std::sort(cands.begin(), cands.end(), byDist);
    }

    // Fill the GPU array + pick the shadow-casting budget (1 spot + 1 point):
    // prefer lights the game flagged shadow=true (flashlight), then nearest.
    int   spotBest = -1, pointBest = -1;
    bool  pointBestFlag = false;
    float spotBestD2 = 1e30f, pointBestD2 = 1e30f;

    for (const Cand& c : cands)
    {
        const vkLight* l = c.l;
        const u32 i = s_frame.count++;
        s_frame.volFlag[i]   = l->volumetric ? 1 : 0;
        s_frame.synthFlag[i] = l->synthBeam  ? 1 : 0;
        GpuLight& g = s_frame.gpu[i];
        g.pos[0] = l->pos.x; g.pos[1] = l->pos.y; g.pos[2] = l->pos.z; g.pos[3] = l->range;
        g.color[0] = l->color.r; g.color[1] = l->color.g; g.color[2] = l->color.b;
        g.color[3] = (l->type == IRender_Light::SPOT) ? 1.f : 0.f;
        Fvector d = l->dir;
        if (d.magnitude() < 1e-5f) d.set(0.f, -1.f, 0.f);
        d.normalize();
        g.dir[0] = d.x; g.dir[1] = d.y; g.dir[2] = d.z;
        g.dir[3] = cosf(l->cone * 0.5f);

        if (l->type == IRender_Light::SPOT)
        {
            // Spots get the budget ONLY when the game flagged them shadow=true
            // (flashlight). No fallback — a random lamp shouldn't cost a map.
            if (!l->shadow) continue;
            // Narrow beams (headlights/searchlights/synth cones, <60°) are the
            // shadows the player actually SEES — 4× distance advantage over
            // wide 120° utility lamps, so walking away from a car doesn't flip
            // the map to a downward pole lamp whose shadow barely reads.
            const float eff = c.d2 * (l->cone < deg2rad(60.f) ? 0.25f : 1.f);
            if (eff < spotBestD2) { spotBest = int(i); spotBestD2 = eff;
                          s_frame.spotPos = l->pos; s_frame.spotDir = d;
                          s_frame.spotRange = l->range; s_frame.spotCone = l->cone;
                          s_frame.spotTexture = l->texture; }
        }
        else
        {
            if (l->range < kPointShadowMinRange) continue;
            if (c.d2 > kPointShadowMaxDist * kPointShadowMaxDist) continue;
            const bool better = (l->shadow && !pointBestFlag)
                             || (l->shadow == pointBestFlag && c.d2 < pointBestD2);
            if (better) { pointBest = int(i); pointBestFlag = l->shadow; pointBestD2 = c.d2;
                          s_frame.pointPos = l->pos; s_frame.pointRange = l->range; }
        }
    }
    s_frame.spotIdx  = spotBest;
    s_frame.pointIdx = pointBest;

    // r_light_debug 1: dump the whole registry + the collected set every ~2 s.
    // Diagnoses "a lamp in R4 doesn't light here" — if the light is absent from
    // the REGISTRY the game never created/activated it (game-side); if it's in
    // the registry but not collected, look at the cull; if collected, shading-side.
    if (ps_r_light_debug) {
        static u32 s_lastLog = 0;
        if (Device.dwTimeGlobal - s_lastLog > 2000) {
            s_lastLog = Device.dwTimeGlobal;
            Msg("[VK Light] registry=%zu collected=%u (eye %.0f,%.0f,%.0f) spotShadow=%d pointShadow=%d",
                s_registry.size(), s_frame.count, eye.x, eye.y, eye.z, spotBest, pointBest);
            if (spotBest >= 0)
                Msg("[VK Light] spot pick: pos=(%.0f,%.0f,%.0f) dir=(%.2f,%.2f,%.2f) range=%.1f cone=%.0f tex=%s",
                    s_frame.spotPos.x, s_frame.spotPos.y, s_frame.spotPos.z,
                    s_frame.spotDir.x, s_frame.spotDir.y, s_frame.spotDir.z,
                    s_frame.spotRange, rad2deg(s_frame.spotCone),
                    s_frame.spotTexture.c_str() ? s_frame.spotTexture.c_str() : "-");
            u32 li = 0;
            for (vkLight* l : s_registry) {
                const float d = eye.distance_to(l->pos);
                Msg("[VK Light]  #%u %s%s%s type=%u pos=(%.0f,%.0f,%.0f) d=%.0f range=%.1f cone=%.0f deg dir=(%.2f,%.2f,%.2f) rgb=(%.2f,%.2f,%.2f) tex=%s",
                    li++, l->active ? "ON " : "off", l->shadow ? "+sh" : "   ", l->volumetric ? "+v" : "  ", l->type,
                    l->pos.x, l->pos.y, l->pos.z, d, l->range, rad2deg(l->cone),
                    l->dir.x, l->dir.y, l->dir.z,
                    l->color.r, l->color.g, l->color.b, l->texture.c_str() ? l->texture.c_str() : "-");
            }
        }
    }
    return s_frame;
}

}  // namespace Lights
}  // namespace VK
