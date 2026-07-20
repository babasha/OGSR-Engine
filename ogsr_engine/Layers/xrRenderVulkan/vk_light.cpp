// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan — dynamic light registry. See vk_light.h.
#include "stdafx.h"
#include "vk_light.h"
#include "vk_color_space.h"        // ColorSpace::LinearizeRGB — authored light colours are sRGB
#include "../../xr_3da/device.h"   // Device.dwFrame (per-frame idempotence)

#include <algorithm>

extern int   ps_r_light_debug; // r_light_debug — log the collected light set every ~2 s
extern int   ps_r_spot_pool;   // r_spot_pool — active spot-shadow tiles (1..kMaxShadowSpots)
extern int   ps_r_point_pool;  // r_point_pool — active point-shadow cubes (1..kMaxShadowPoints)
extern float ps_r_point_boost; // r_point_boost — campfire glow intensity × (volumetric points)
extern float ps_r_point_range; // r_point_range — campfire light reach × (volumetric points)

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

    // Fill the GPU array + pick the shadow budget: a POOL of up to
    // kMaxShadowSpots spots (each gets its own atlas tile) + 1 point cube.
    // Spots compete only when the game flagged them shadow=true (flashlight,
    // synth beams) — a random lamp shouldn't cost a map.
    struct SpotCand { int gi; float eff; };
    SpotCand spotCands[kMaxClusterLights];
    u32      nSpotCands = 0;
    SpotCand pointCands[kMaxClusterLights];   // {gi, d2} — shadow-flagged points
    u32      nPointCands = 0;

    // POINT range deadband (hysteresis). Campfires animate l->range by ±~0.5 m
    // every frame (flicker). But gpu[].pos[3] is the ONE range shared by shading,
    // fog AND the shadow cube — the cube's far plane + the receiver's depth
    // linearization both key off it. That per-frame jitter forced the cube's
    // cached STATIC faces to re-raster EVERY frame (the static/dynamic split never
    // cached — [VK PointPool] static=1 every frame). Freeze the range inside a 1 m
    // band per light: the visible flicker is colour/intensity (l->color, untouched),
    // not the exact reach, so a frozen radius is imperceptible and the static cube
    // now caches. Keyed by light identity across frames; slot reclaimed by age.
    struct RangeHold { const void* owner = nullptr; float range = 0.f; u32 seen = 0; };
    static RangeHold s_pointRangeHold[kMaxClusterLights];
    const u32 rhFrame = Device.dwFrame;
    auto stablePointRange = [&](const void* owner, float raw) -> float {
        RangeHold* free_ = nullptr; RangeHold* oldest = &s_pointRangeHold[0];
        for (auto& h : s_pointRangeHold) {
            if (h.owner == owner) {
                if (_abs(raw - h.range) > 1.0f) h.range = raw;   // snap only on a real reach change
                h.seen = rhFrame; return h.range;
            }
            if (!h.owner && !free_) free_ = &h;
            if (h.seen < oldest->seen) oldest = &h;
        }
        RangeHold* slot = free_ ? free_ : oldest;               // claim free, else evict LRU
        slot->owner = owner; slot->range = raw; slot->seen = rhFrame;
        return raw;
    };

    for (const Cand& c : cands)
    {
        vkLight* l = c.l;
        const u32 i = s_frame.count++;
        s_frame.volFlag[i]   = l->volumetric ? 1 : 0;
        s_frame.synthFlag[i] = l->synthBeam  ? 1 : 0;
        s_frame.flashFlag[i] = l->flashlight ? 1 : 0;
        s_frame.src[i]       = l;
        GpuLight& g = s_frame.gpu[i];
        // Campfire/brazier tuning: volumetric-flagged POINT lights get the glow +
        // reach multipliers (r_point_boost / r_point_range). Stored into gpu[]
        // so shading, fog AND the shadow cube (which reads gpu[].pos[3]) agree.
        const bool  fire     = (l->type == IRender_Light::POINT) && l->volumetric;
        const float rangeMul = fire ? ps_r_point_range : 1.f;
        const float boost    = fire ? ps_r_point_boost : 1.f;
        // Deadband the POINT reach so a flickering campfire's shadow cube can cache
        // its statics (see s_pointRangeHold). Spots keep their exact range.
        float range3 = l->range * rangeMul;
        if (l->type == IRender_Light::POINT) range3 = stablePointRange(l, range3);
        g.pos[0] = l->pos.x; g.pos[1] = l->pos.y; g.pos[2] = l->pos.z; g.pos[3] = range3;
        // Light colours are authored by eye in item configs / ALife spawn data
        // (torch color_r2, lamp colour, headlights…), i.e. sRGB. Decode here — this is
        // the single collection point feeding BOTH the ≤16-light UBO copy and the
        // clustered SSBO, so one conversion covers every dynamic light in the frame.
        // boost is applied AFTER the decode: like r_sun_boost it is a radiance scale.
        g.color[0] = l->color.r; g.color[1] = l->color.g; g.color[2] = l->color.b;
        ColorSpace::LinearizeRGB(g.color);
        g.color[0] *= boost; g.color[1] *= boost; g.color[2] *= boost;
        g.color[3] = (l->type == IRender_Light::SPOT) ? 1.f : 0.f;   // flag, never colour
        Fvector d = l->dir;
        if (d.magnitude() < 1e-5f) d.set(0.f, -1.f, 0.f);
        d.normalize();
        g.dir[0] = d.x; g.dir[1] = d.y; g.dir[2] = d.z;
        g.dir[3] = cosf(l->cone * 0.5f);

        if (l->type == IRender_Light::SPOT)
        {
            if (!l->shadow) continue;
            // Narrow beams (headlights/searchlights/synth cones, <60°) are the
            // shadows the player actually SEES — 4× distance advantage over
            // wide 120° utility lamps for the pool ordering.
            const float eff = c.d2 * (l->cone < deg2rad(60.f) ? 0.25f : 1.f);
            spotCands[nSpotCands++] = { int(i), eff };
        }
        else
        {
            // Points compete for a pool cube only when the game flagged them
            // shadow=true (campfires) — like spots, no fallback for random lamps.
            if (!l->shadow) continue;
            if (l->range < kPointShadowMinRange) continue;
            if (c.d2 > kPointShadowMaxDist * kPointShadowMaxDist) continue;
            pointCands[nPointCands++] = { int(i), c.d2 };
        }
    }
    // Pool = the best kMaxShadowSpots candidates (r_spot_pool can shrink it).
    std::sort(spotCands, spotCands + nSpotCands,
              [](const SpotCand& a, const SpotCand& b) { return a.eff < b.eff; });
    const u32 poolCap = u32(std::clamp(ps_r_spot_pool, 1, int(kMaxShadowSpots)));
    s_frame.poolCount = (nSpotCands < poolCap) ? nSpotCands : poolCap;
    for (u32 k = 0; k < s_frame.poolCount; ++k)
        s_frame.poolGi[k] = spotCands[k].gi;
    // Cookie pick: the pooled spot with a projection texture (flashlight beam
    // pattern); plain poolGi[0] otherwise. Receivers gate the cookie on this.
    s_frame.spotIdx = (s_frame.poolCount > 0) ? s_frame.poolGi[0] : -1;
    for (u32 k = 0; k < s_frame.poolCount; ++k) {
        const vkLight* l = s_frame.src[s_frame.poolGi[k]];
        if (l->texture.size()) { s_frame.spotIdx = s_frame.poolGi[k]; break; }
    }
    if (s_frame.spotIdx >= 0) {
        const vkLight* l = s_frame.src[s_frame.spotIdx];
        s_frame.spotPos = l->pos; s_frame.spotDir = l->dir;
        if (s_frame.spotDir.magnitude() < 1e-5f) s_frame.spotDir.set(0.f, -1.f, 0.f);
        s_frame.spotDir.normalize();
        s_frame.spotRange = l->range; s_frame.spotCone = l->cone;
        s_frame.spotTexture = l->texture;
    }
    // Point pool = nearest kMaxShadowPoints shadow-flagged points.
    std::sort(pointCands, pointCands + nPointCands,
              [](const SpotCand& a, const SpotCand& b) { return a.eff < b.eff; });
    const u32 poolCapP = u32(std::clamp(ps_r_point_pool, 1, int(kMaxShadowPoints)));
    s_frame.poolCountP = (nPointCands < poolCapP) ? nPointCands : poolCapP;
    for (u32 k = 0; k < s_frame.poolCountP; ++k)
        s_frame.poolGiP[k] = pointCands[k].gi;
    // Legacy single-pick alias = the nearest pooled point (fog/receiver fallbacks).
    if (s_frame.poolCountP > 0) {
        s_frame.pointIdx = s_frame.poolGiP[0];
        const vkLight* l = s_frame.src[s_frame.pointIdx];
        s_frame.pointPos = l->pos; s_frame.pointRange = l->range;
    } else {
        s_frame.pointIdx = -1;
    }

    // r_light_debug 1: dump the whole registry + the collected set every ~2 s.
    // Diagnoses "a lamp in R4 doesn't light here" — if the light is absent from
    // the REGISTRY the game never created/activated it (game-side); if it's in
    // the registry but not collected, look at the cull; if collected, shading-side.
    if (ps_r_light_debug) {
        static u32 s_lastLog = 0;
        if (Device.dwTimeGlobal - s_lastLog > 2000) {
            s_lastLog = Device.dwTimeGlobal;
            Msg("[VK Light] registry=%zu collected=%u (eye %.0f,%.0f,%.0f) spotPool=%u cookie=%d pointPool=%u",
                s_registry.size(), s_frame.count, eye.x, eye.y, eye.z, s_frame.poolCount, s_frame.spotIdx, s_frame.poolCountP);
            if (s_frame.spotIdx >= 0)
                Msg("[VK Light] cookie pick: pos=(%.0f,%.0f,%.0f) dir=(%.2f,%.2f,%.2f) range=%.1f cone=%.0f tex=%s",
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
