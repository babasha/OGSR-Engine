#version 450
#extension GL_GOOGLE_include_directive : require
// xrRenderVulkan — VSM temporal resolve (TAA-for-shadows). One thread per screen
// pixel: reconstruct world from the prepass depth, sample the VSM atlas (the same
// world->light->page->atlas mapping the receiver used to do directly), then blend
// against a reprojected history with an EMA. The clipmap origin is sub-texel JITTERED
// each frame (vk_vsm BeginFrame) so the per-frame texel quantization of the shadow
// edge decorrelates; the EMA averages it → the "crawling snake" along moving-sun
// shadow edges resolves to a smooth, stable soft edge. Output is a screen-space mask
// (R = lit factor, G = distance to this frame's camera, for history disocclusion
// rejection). Receivers then just sample R by screen UV (vsm_sample.glsl). See vk_vsm.cpp.
#include "vsm_common.glsl"

layout(local_size_x = 8, local_size_y = 8) in;

layout(set = 0, binding = 0) uniform sampler2D uDepth;    // current scene prepass depth
layout(set = 0, binding = 1) uniform sampler2D uAtlas;    // STATIC VSM atlas (opaque + trees)
layout(set = 0, binding = 7) uniform sampler2D uAtlasDyn; // DYNAMIC VSM atlas (NPC + grass)
layout(set = 0, binding = 2) readonly buffer VsmPageTable    { uint vsmPageTable[]; };     // STATIC virtual->slot
layout(set = 0, binding = 8) readonly buffer VsmPageTableDyn { uint vsmPageTableDyn[]; };  // DYNAMIC virtual->slot
layout(set = 0, binding = 9) readonly buffer VsmDynUsed      { uint vsmDynUsed[]; };       // dyn slot -> 1 if any NPC/grass caster was binned into it
layout(set = 0, binding = 3) uniform VsmClipmap {
    mat4 view;                 // world -> sun light space
    vec4 level[VSM_LEVELS];    // xy = level origin (light XY of texel 0,0), z = extent (m)
    vec4 zparams;              // x = zNear, y = 1/(zFar-zNear), z = depth-compare bias
} vsmC;
layout(set = 0, binding = 4) uniform sampler2D uHistory; // previous frame's mask (RG = shadow, dist)
layout(set = 0, binding = 5, rgba16f) uniform writeonly image2D uOut;
layout(set = 0, binding = 6) uniform Resolve {
    mat4 invViewProj;    // current clip -> world
    mat4 prevViewProj;   // world -> previous-frame clip (history reproject)
    vec4 prevCamPos;     // xyz = previous frame camera (for history distance check); w = dyn-debug (r_vsm_debug_dyn: write dyn occlusion to mask B)
    vec4 curCamPos;      // xyz = this frame camera (stored as G for next frame); w = history weight on dyn-shadowed pixels (r_vsm_ta_blend_dyn)
    vec4 screen;         // xy = pixel dims, zw = 1/dims
    vec4 params;         // x = history weight (alpha), y = reject tolerance, z = historyValid, w = dyn-gate (1 = skip dyn pages with no casters, r_vsm_dyn_gate)
    vec4 params2;        // x = clamp tol (neighbourhood clamp, r_vsm_ta_clamp), y = motion ref px (r_vsm_ta_motion), z = motion-floor weight (r_vsm_ta_motion_floor), w = unused
} R;

// VSM atlas sample (3x3 PCF) — mirrors the page mapping vsm_page.vert rasterized with.
// Returns lit factor 1 = lit .. 0 = shadowed; out of clipmap / unmapped page → lit.
// dynOcc (out) = fraction of taps occluded by the DYNAMIC atlas alone — only computed
// under r_vsm_debug_dyn (prevCamPos.w), UNGATED by dynUsed: it shows the atlas truth,
// so the tonemap's red overlay reveals NPC/grass shadows even if the gate drops them.
// dynHit (out) = fraction of taps the DYNAMIC atlas shadows (always computed): these
// casters MOVE every frame (wind-swaying crowns, NPCs), so main() cuts the EMA history
// weight there — the full weight drags a many-frame smear ("jelly") behind them.
// statLit (out) = visibility against the STATIC atlas ALONE (trees/buildings, no
// grass/NPC/wind-tree dyn casters) — written to mask B. HISTORY: this was the
// mask-based grass receiver path (grass read B to avoid dyn blade-frequency stripes
// projecting onto the canopy); grass now samples the atlases DIRECTLY at its own
// world pos (vsm_sample VSM_GRASS_DIRECT), so B survives only as a diagnostic
// (r_grass_debug 3 comparison view) — nearly free: same taps, one extra compare.
float sampleVSM(vec3 wp, out float dynOcc, out float dynHit, out float statLit)
{
    dynOcc = 0.0;
    dynHit = 0.0;
    statLit = 1.0;
    vec3 lp = (vsmC.view * vec4(wp, 1.0)).xyz;
    vec2 luv; ivec2 page;
    int  L = vsmSelect(lp.xy, vsmC.level, luv, page);
    if (L < 0) return 1.0;

    // Walk COARSER until the STATIC page is mapped. The static table is fully resident
    // for every marked page, so a miss at the finest containing level means the page
    // was marked coarser (throttle LOD bias, r_vsm_throttle) or its scroll-in redraw
    // was deferred (r_vsm_dirty_budget) — sample the next coarser level instead of
    // falling out "lit". One 4-byte read per extra step; the un-throttled steady state
    // exits on the first iteration (identical to the old single lookup).
    uint slotS = VSM_UNMAPPED;
    for (; L < VSM_LEVELS; ++L) {
        vec2 t = (lp.xy - vsmC.level[L].xy) / vsmC.level[L].z;
        page   = clamp(ivec2(floor(t * float(VSM_PAGES_AXIS))), ivec2(0), ivec2(VSM_PAGES_AXIS - 1));
        slotS  = vsmPageTable[vsmPageIndex(L, page)];
        if (slotS < uint(VSM_MAX_PHYS_S)) { luv = t; break; }
    }
    if (L >= VSM_LEVELS) return 1.0;   // mapped nowhere → lit (old policy)

    int  idx   = vsmPageIndex(L, page);
    uint slotD = vsmPageTableDyn[idx];   // DYNAMIC atlas slot (demand-allocated; 2048 grid)
    bool hasS  = slotS < uint(VSM_MAX_PHYS_S);
    bool resD  = slotD < uint(VSM_MAX_PHYS);
    // The dyn alloc claims a slot for EVERY visible page; only pages some NPC/grass caster
    // actually binned into hold real depth (the rest are cleared-empty) → skip those 9 taps.
    // Gated by r_vsm_dyn_gate (params.w) for live A/B while the flag chain is being verified.
    bool hasD  = resD && (R.params.w < 0.5 || vsmDynUsed[slotD] != 0u);
    bool dbgD  = resD && (R.prevCamPos.w > 0.5);   // debug: always fetch the dyn atlas
    if (!hasS && !hasD && !dbgD) return 1.0;       // page resident in neither atlas → lit

    vec2  pageLocal = luv * float(VSM_PAGES_AXIS) - vec2(page);   // [0,1) within the page
    vec2  baseS = vec2(float(slotS % uint(VSM_ATLAS_W_S)), float(slotS / uint(VSM_ATLAS_W_S)));
    vec2  baseD = vec2(float(slotD % uint(VSM_ATLAS_W)),   float(slotD / uint(VSM_ATLAS_W)));
    float zHere = (lp.z - vsmC.zparams.x) * vsmC.zparams.y;
    float bias  = vsmC.zparams.z;   // STATIC-atlas receiver bias (terrain self-shadow acne)
    float biasD = vsmC.zparams.w;   // DYNAMIC-atlas receiver bias (tiny: no ground in that atlas)
    const vec2 dimS = vec2(float(VSM_ATLAS_W_S), float(VSM_ATLAS_H_S));   // static atlas (6144)
    const vec2 dimD = vec2(float(VSM_ATLAS_W),   float(VSM_ATLAS_H));     // dynamic atlas (2048)

    const float tp    = 1.0 / float(VSM_PAGE_SIZE);   // one page texel, page-local units
    const float inset = 0.5 * tp;
    float lit = 0.0, litS = 0.0;
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        vec2 pl = clamp(pageLocal + vec2(float(dx), float(dy)) * tp, vec2(inset), vec2(1.0 - inset));
        // Each atlas gets its OWN receiver bias. The static atlas holds the ground
        // itself, so its bias (zparams.z) must absorb terrain self-shadow acne — but at
        // a 2000 m clipmap z-range 0.0003 normalized = 0.6 m along the sun ray, which
        // also REJECTED every dyn occluder closer than 0.6 m: grass blade parts below
        // ~half a metre cast nothing (high sun shrinks the gap = H/sin(elev) further) —
        // tuft shadows went missing/cut while tall trees were unaffected. The DYNAMIC
        // atlas holds ONLY casters (grass/NPC/near-crowns), NEVER the receiving ground,
        // so it cannot acne against it → a tiny epsilon (zparams.w, r_vsm_bias_dyn:
        // D16 quantization + the write-side raster bias) keeps low blades shadowing.
        float occS = 1.0;
        if (hasS) occS = min(occS, texture(uAtlas, (baseS + pl) / dimS).r);
        bool sh = (zHere - bias > occS);
        if (hasD || dbgD) {
            float d = texture(uAtlasDyn, (baseD + pl) / dimD).r;
            if (hasD && (zHere - biasD > d)) {
                sh = true;
                dynHit += 1.0 / 9.0;
            }
            // Debug: mode 1 counts only the VISIBLE darkening the dyn atlas causes — taps
            // the static atlas leaves lit but the dyn depth shadows. (Raw dyn occlusion
            // painted whole shadow COLUMNS through houses/crowns — technically correct
            // atlas content, but already dark in the real image = pure confusion.)
            // Mode 2 (r_vsm_debug_dyn 2) = RAW: any dyn occlusion, ungated — for auditing
            // whether a caster made it into the atlas at all (grass pair coverage).
            // Mode 3 (r_vsm_debug_dyn 3) = PRESENCE: any caster DEPTH at this receiver's
            // texel, ignoring the depth comparison — splits "fragments never rasterized
            // into the page" (no red in 3) from "depth written but the test rejects it"
            // (red in 3, none in 2).
            if (dbgD) {
                if (R.prevCamPos.w > 2.5) { if (d < 0.999) dynOcc += 1.0 / 9.0; }
                else if ((R.prevCamPos.w > 1.5 || zHere - bias <= occS) && (zHere - biasD > d)) dynOcc += 1.0 / 9.0;
            }
        }
        lit  += sh ? 0.0 : 1.0;
        litS += (zHere - bias > occS) ? 0.0 : 1.0;   // static atlas alone (grass receivers)
    }
    statLit = litS * (1.0 / 9.0);
    return lit * (1.0 / 9.0);
}

// Reconstruct world from the prepass depth (D3D NDC, y-up — matches vsm_mark.comp / ssao.frag).
vec3 reconWorld(vec2 uv)
{
    float zndc = texture(uDepth, uv).r;
    vec4  clip = vec4(uv.x * 2.0 - 1.0, 1.0 - 2.0 * uv.y, zndc, 1.0);
    vec4  w    = R.invViewProj * clip;
    return w.xyz / w.w;
}

void main()
{
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    if (px.x >= int(R.screen.x) || px.y >= int(R.screen.y)) return;

    vec2  uv   = (vec2(px) + 0.5) * R.screen.zw;
    float zndc = texture(uDepth, uv).r;
    if (zndc >= 0.99999) { imageStore(uOut, px, vec4(1.0, 1e6, 1.0, 0.0)); return; }   // sky → lit (B too)

    vec4 clip  = vec4(uv.x * 2.0 - 1.0, 1.0 - 2.0 * uv.y, zndc, 1.0);
    vec4 world = R.invViewProj * clip;
    vec3 wp    = world.xyz / world.w;

    float dynOcc, dynHit, statLit;
    float cur  = sampleVSM(wp, dynOcc, dynHit, statLit);

    // Debug refinement: a surface FACING AWAY from the sun gets no direct light — a dyn
    // shadow there changes nothing on screen (e.g. a ceiling under a roof hole crossed by
    // a distant NPC's shadow column). Reconstruct the geometric normal from depth and
    // drop the red on backfacing/grazing pixels.
    if (dynOcc > 0.0 && R.prevCamPos.w < 1.5) {   // mode 2 = raw: keep backfacing/grazing red too
        vec3 wpX = reconWorld(uv + vec2(R.screen.z, 0.0));
        vec3 wpY = reconWorld(uv + vec2(0.0, R.screen.w));
        vec3 n   = cross(wpX - wp, wpY - wp);
        if (dot(n, R.curCamPos.xyz - wp) < 0.0) n = -n;   // orient towards the camera
        // Light travel direction (world): row 2 of the world->light view (light-space +Z).
        vec3 sunTravel = normalize(vec3(vsmC.view[0].z, vsmC.view[1].z, vsmC.view[2].z));
        if (dot(normalize(n), -sunTravel) < 0.05) dynOcc = 0.0;
    }
    float dist = length(wp - R.curCamPos.xyz);   // stored for next frame's reject test

    float outShadow = cur;
    if (R.params.z > 0.5) {                       // history valid (not first frame / no resize)
        vec4 pc = R.prevViewProj * vec4(wp, 1.0);
        if (pc.w > 0.0) {
            vec2 puv = (pc.xy / pc.w) * vec2(0.5, -0.5) + 0.5;   // prev-frame screen UV (same y-flip)
            if (all(greaterThanEqual(puv, vec2(0.0))) && all(lessThanEqual(puv, vec2(1.0)))) {
                vec4  hist = texture(uHistory, puv);             // (shadowPrev, distFromPrevCam, dbg, dynHitPrev)
                float expectPrev = length(wp - R.prevCamPos.xyz);
                // Same static surface last frame → stored distance ≈ expected. Reject
                // (use current only) on disocclusion so silhouettes don't ghost.
                if (abs(hist.r) <= 1.0001 && abs(hist.g - expectPrev) <= R.params.y * expectPrev + 0.05) {
                    // Dyn-atlas casters move every frame → their shadow edge is somewhere
                    // ELSE each frame; the full EMA (tuned to average the static edge's
                    // texel quantization) smears them into a trail. Where the dyn atlas
                    // shadows this pixel now — or did last frame (hist.a covers the
                    // trailing edge) — drop to the dyn history weight (r_vsm_ta_blend_dyn).
                    float a = (dynHit > 0.0 || hist.a > 0.0) ? min(R.params.x, R.curCamPos.w) : R.params.x;

                    // (2) MOTION-ADAPTIVE weight. The distance reject above only catches
                    // DISocclusion (surface changed) — but high-frequency foliage shadows on
                    // the SAME ground (same depth) pass it, and under camera motion the
                    // bilinear history fetch at `puv` blurs a bit more each frame → the full
                    // EMA accumulates that blur into "каша" (only while moving). Fade the
                    // weight toward a floor by the reprojected screen motion (px): still
                    // camera keeps the full EMA (clean coarse shadow + sub-texel detail),
                    // moving camera stops compounding the blur.
                    float motionPx = length((puv - uv) * R.screen.xy);
                    float mfade = clamp(motionPx / max(R.params2.y, 1e-3), 0.0, 1.0);
                    a = mix(a, min(a, R.params2.z), mfade);

                    // (1) NEIGHBOURHOOD CLAMP (value space). Bound the history sample to the
                    // current shadow ±tol before the blend so a stale value can't drag a
                    // many-frame trail (the classic TAA anti-ghost, done per-pixel without
                    // the extra sampleVSM taps a spatial min/max would cost): sub-texel
                    // jitter within tol still averages (keeps the AA), a big deviation
                    // (moved foliage edge) is clamped out (kills the smear).
                    float hClamped = clamp(hist.r, cur - R.params2.x, cur + R.params2.x);
                    outShadow = mix(cur, hClamped, a);
                }
            }
        }
    }
    // B = STATIC-only visibility — diagnostic channel (r_grass_debug 3; grass shading
    // now samples the atlases directly, see sampleVSM's statLit note). Raw, no EMA.
    // Under r_vsm_debug_dyn the debug dynOcc takes the channel over (the red overlay).
    // A = this frame's dynHit (next frame's trailing-edge EMA cut).
    float bOut = (R.prevCamPos.w > 0.5) ? dynOcc : statLit;
    imageStore(uOut, px, vec4(outShadow, dist, bOut, dynHit));
}
