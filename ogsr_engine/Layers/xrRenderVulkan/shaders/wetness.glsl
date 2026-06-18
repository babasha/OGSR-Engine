// xrRenderVulkan - shared wet-surface shading (rain darken + puddle sky reflection
// + sun glint). Was three drifting copies of applyWetness across world_lmap /
// world_vlit (identical) and world_terrain (divergent). Decomposed here:
//   applyWetnessCore   - the reflection math, identical in all three (takes the
//                        already-computed wetK + puddle coverage pud)
//   puddleCoverage     - lmap/vlit procedural puddle placement
//   applyWetness       - lmap/vlit entry point (down-facing kill, pud internal)
//   applyWetnessTerrain- terrain entry point (no down-facing kill; pud passed in
//                        from the caller's per-pixel sssPuddle)
// #include AFTER light_ubo.glsl, wet_common.glsl, env_common.glsl (rainVis) and
// flow_sim_sample.glsl (simWaterSoft/simFlow).
#ifndef WETNESS_GLSL
#define WETNESS_GLSL

// lmap/vlit puddle coverage: procedural blobs on near-flat up-facing surfaces.
// SSS placement (puddlesMaskProc) is the default; flow sim is the parked alt.
// `wet` is rain_params.y AFTER the rainVis multiply; `upness` = clamp(N.y,0,1).
float puddleCoverage(vec3 wp, vec3 N, float wet, float upness)
{
    float slope = clamp((1.0 - max(abs(N.x), abs(N.z)) - 0.9) * 13.0, 0.0, 1.0);
    float cov   = clamp(wet * L.pom_params7.y * 1.5, 0.0, 1.0);   // grows/recedes; x1.5 = distinct, not fields
    float pud   = (L.pom_params7.x > 0.5) ? puddlesMaskProc(wp.xz, cov, L.pom_params7.w) * slope
                : (L.pom_params6.x > 0.5) ? smoothstep(0.04, 0.12, simWaterSoft(wp)) * upness
                : 0.0;
    return clamp(pud, 0.0, 1.0);
}

// Shared wet reflection core: caller supplies wetK (wetness factor) and pud
// (puddle coverage, already in [0,1]); this darkens albedo (inout) and returns
// the additive puddle/glint reflection. Math is byte-identical to all three
// pre-refactor copies.
vec3 applyWetnessCore(inout vec3 albedo, vec3 wp, vec3 N, float wetK, float pud, float sunMask)
{
    // DARKEN: FULL in deep puddles (pud^2 -> body fills AFTER the shine), ~NONE open.
    albedo *= 1.0 - L.rain_params.z * wetK * mix(0.05, 1.0, pud * pud);

    vec3  toEye    = L.eye_pos.xyz - wp;
    float dist     = length(toEye);
    float reflFade = smoothstep(70.0, 35.0, dist);
    if (wetK * reflFade < 0.004) return vec3(0.0);

    float t = L.sky_params.w;
    // Ripples ONLY in puddles (still ground off-puddle).
    float ripFade = smoothstep(18.0, 8.0, dist) * clamp(L.rain_params.x * 1.5 + 0.1, 0.0, 1.0)
                  * smoothstep(0.05, 0.35, pud);
    vec2  vel   = (L.pom_params6.x > 0.5) ? simFlow(wp) : vec2(0.0);
    float velMS = length(vel) * 150.0;
    vec2  scrl  = (velMS > 0.01) ? normalize(vel) * (t * velMS * 0.25) : vec2(0.0);
    // Puddle = flat water mirror (flatten normal toward up).
    vec3  Nbase = mix(N, vec3(0.0, 1.0, 0.0), clamp(pud * pud, 0.0, 1.0));
    vec3  Nr = Nbase;
    float crest = 0.0;
    if (ripFade > 0.01) {
        float rainAmp = 0.40 + 0.40 * clamp(L.rain_params.x, 0.0, 1.0);
        vec2 rip = rainRipples(wp.xz - scrl, t * 0.7) * (rainAmp * ripFade);
        if (velMS > 0.1)
            rip += rainRipples(wp.xz * 1.6 - scrl * 1.6, t * 0.9) * (clamp(velMS * 0.12, 0.0, 0.5) * ripFade);
        Nr = normalize(vec3(Nbase.x + rip.x, Nbase.y, Nbase.z + rip.y));
        crest = clamp(length(rip) * 2.5, 0.0, 1.0);
    }
    vec3 V = normalize(toEye);
    vec3 R = reflect(-V, Nr);
    float fres = pow(1.0 - clamp(dot(V, Nr), 0.0, 1.0), 3.0);
    float xf = clamp(L.sky_params.x, 0.0, 1.0);
    // Sharp sky in puddles (LOD0); blurred off-puddle (where reflK ~ 0 anyway).
    float reflLod = mix(5.0, 0.0, pud);
    vec3 sky = textureLod(uSky0, R, reflLod).rgb;
    if (xf > 0.01) sky = mix(sky, textureLod(uSky1, R, reflLod).rgb, xf);
    // Water BODY fills later (pud^2) than the SHINE (reflection ~ pud) - "shine first".
    albedo = mix(albedo, albedo * vec3(0.34, 0.40, 0.46) * (1.0 - 0.25 * pud), pud * pud);
    float puddleK = wetK * pud * (0.45 + 0.55 * fres) * reflFade * clamp(L.rain_params.w, 0.0, 2.0);
    // SUN GLINT (SSFX specular_phong) - ripples shatter it to sparkles.
    vec3  Ld    = normalize(-L.sun_dir.xyz);
    vec3  Hh    = normalize(Ld + V);
    float glint = pow(max(dot(Nr, Hh), 0.0), 220.0) * pud * sunMask;
    float foam = smoothstep(3.0, 6.0, velMS) * pud * ripFade * 0.25;
    // RING WAVE crests - bright leading edge of each ripple (visible drop waves).
    vec3 crestCol = (L.sun_color.rgb + L.ambient.rgb) * (crest * pud * 0.07);
    return sky * puddleK + L.sun_color.rgb * (glint * 3.0) + vec3(foam) + crestCol;
}

// world_lmap / world_vlit: down-facing kill in wetK, pud from puddleCoverage.
vec3 applyWetness(inout vec3 albedo, vec3 wp, vec3 N, float sunMask)
{
    float wet = L.rain_params.y;
    if (wet < 0.005) return vec3(0.0);
    wet *= rainVis(wp);
    float upness = clamp(N.y, 0.0, 1.0);
    // Kill wetness on DOWN-facing surfaces (ceilings/overhang undersides).
    float wetK = wet * mix(0.35, 1.0, upness) * smoothstep(-0.15, 0.05, N.y);
    float pud  = puddleCoverage(wp, N, wet, upness);
    return applyWetnessCore(albedo, wp, N, wetK, pud, sunMask);
}

// world_terrain: pud arrives from the caller's per-pixel sssPuddle. Terrain's
// wetK has NO down-facing smoothstep and pud is gated by upness here - matches
// the pre-refactor terrain applyWetness exactly.
vec3 applyWetnessTerrain(inout vec3 albedo, vec3 wp, vec3 N, float pudIn, float sunMask)
{
    float wet = L.rain_params.y;
    if (wet < 0.005) return vec3(0.0);
    wet *= rainVis(wp);
    float upness = clamp(N.y, 0.0, 1.0);
    float wetK = wet * mix(0.35, 1.0, upness);
    float pud  = clamp(pudIn, 0.0, 1.0) * upness;
    return applyWetnessCore(albedo, wp, N, wetK, pud, sunMask);
}

#endif // WETNESS_GLSL
