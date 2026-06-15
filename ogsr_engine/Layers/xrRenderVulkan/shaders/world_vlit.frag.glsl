#version 450
#extension GL_GOOGLE_include_directive : require
#include "wet_common.glsl"   // vHash/vNoise/puddlesMaskProc/rippleLayer/rainRipples

// World pass - vert-lit variant. Final colour = albedo - pre-baked
// vertex lighting + small ambient floor.
//
// vBakedColor is what the level compiler computed offline (point-light
// contributions + bounce). It does NOT include direct sun - that needs
// runtime sun direction, deferred until env subsystem lands. As an
// approximation we add a constant ambient term and a sun-direction-
// independent fake-sun term gated by the per-vertex sun mask.
//
// Binding 2 (lmap) is bound for descriptor-layout compatibility but
// not sampled here - vert-lit materials don't have a lightmap.

layout(set = 0, binding = 0) uniform sampler2D uTexDiffuse;
layout(set = 0, binding = 1) uniform sampler2D uTexDetail;
layout(set = 0, binding = 2) uniform sampler2D uTexLmap;  // unused (white fallback)
layout(set = 0, binding = 3) uniform sampler2D uTexBumpX; // .a = height (POM); flat=1 → no parallax

// Per-frame environment lighting (set 1) - see vk_env_light.{h,cpp}.
struct DynLight {
    vec4 pos;     // xyz = world position, w = range
    vec4 color;   // rgb = colour,         w = 1 spot / 0 point
    vec4 dir;     // xyz = spot direction, w = cos(cone/2)
};
layout(set = 1, binding = 0) uniform Lighting {
    vec4 sun_dir;
    vec4 sun_color;
    vec4 hemi_color;
    vec4 ambient;
    mat4 sun_vp;      // sun light view-proj (shadow lookup)
    vec4 counts;      // x = dynamic light count
    DynLight lights[16];
    mat4 spot_vp;        // spot (flashlight) shadow view-proj
    vec4 shadow_params;  // x = spot-shadowed light index (-1 none), y = point-shadowed index
    mat4 sun_near_vp;    // sun cascade 0 view-proj (25 m, per-frame, R4 scheme)
    mat4 sun_c1_vp;      // sun cascade 1 view-proj (60 m, per-frame)
    vec4 fog_color;      // rgb haze colour (env)
    vec4 fog_params;     // x=-near*r, y=near, z=far, w=r; fog = saturate(dist*w + x)
    vec4 eye_pos;        // xyz camera world pos
    vec4 sky_params;     // x=cube cross-fade weight, y=ambient scale, z=sample LOD
    vec4 ao_params;      // x=1/screenW, y=1/screenH, z=AO strength (0=off)
    mat4 rain_vp;        // straight-down ortho VP for the rain occlusion map
    vec4 rain_params;    // x=rain density, y=wetness, z=darken, w=reflection scale
    mat4 scene_vp;       // (SSR puddles — declared for layout match, unused here)
    vec4 cam_dir;
    vec4 cam_rightT;
    vec4 cam_topT;
    vec4 pom_params;     // x=POM amplitude (UV), y=max steps, z=fade dist (m), w=on
    vec4 pom_params2;    // x=blur, y=normal, z=self-shadow, w=contact AO
    vec4 pom_params3;    // x=debug view, y=ao_flat, z=ceil strength, w=floor strength
    vec4 pom_params4;    // x=terrain POM enable, y=detail-normal, z=micro-AO, w=debug (terrain only)
    vec4 pom_params5;    // x=terrain gloss, y=geo-puddle radius (0=off), z=geo-puddle depth scale, w=puddle debug
    vec4 pom_params6;    // x=water-sim enable (puddles from the flow sim)
    vec4 pom_params7;    // SSS puddles: x=enable, y=level (coverage), z=micro, w=macro scale
} L;
layout(set = 1, binding = 1) uniform sampler2D uShadow;
layout(set = 1, binding = 2) uniform sampler2D uSpotShadow;
layout(set = 1, binding = 3) uniform samplerCube uPointShadow;
layout(set = 1, binding = 4) uniform sampler2D uShadowNear;    // sun cascade 0 (~0.61 cm texels)
layout(set = 1, binding = 5) uniform sampler2D uShadowC1;      // sun cascade 1 (~1.46 cm texels)
layout(set = 1, binding = 6) uniform samplerCube uSky0;        // sky ambient cube 0 (weather A)
layout(set = 1, binding = 7) uniform samplerCube uSky1;        // sky ambient cube 1 (weather B)
layout(set = 1, binding = 8) uniform sampler2D uAO;            // GTAO (half-res)
layout(set = 1, binding = 9) uniform sampler2D uRainMap;       // top-down rain occlusion (wetness mask)
layout(set = 1, binding = 11) uniform sampler2D uWater;        // water depth (flow sim, metres)
layout(set = 1, binding = 12) uniform sampler2D uFlow;         // water velocity (flow sim, uv/sec)
layout(set = 1, binding = 10) uniform sampler2D uSpotCookie;   // flashlight beam texture (cookie)

// GTAO visibility - see world_lmap.frag (occludes hemi+ambient only).
float gtaoVis()
{
    float ao = textureLod(uAO, gl_FragCoord.xy * L.ao_params.xy, 0.0).r;
    return pow(clamp(ao, 0.0, 1.0), L.ao_params.z);   // strength = exponent (0 = off)
}

// Colored AO - see world_lmap.frag (R4 compute_colored_ao port).
vec3 coloredAO(float ao, vec3 albedo)
{
    vec3 a =  2.0404 * albedo - 0.3324;
    vec3 b = -4.7951 * albedo + 0.6417;
    vec3 c =  2.7552 * albedo + 0.6903;
    return max(vec3(ao), ((ao * a + b) * ao + c) * ao);
}

// Hemisphere sky ambient (R4 hmodel.h) - see world_lmap.frag.
vec3 skyAmbient(vec3 N)
{
    float lod = L.sky_params.z;
    float xf = clamp(L.sky_params.x, 0.0, 1.0);   // weather cross-fade — usually 0/1
    vec3 a = textureLod(uSky0, N, lod).rgb;
    return (xf > 0.01) ? mix(a, textureLod(uSky1, N, lod).rgb, xf) : a;  // 2nd cube only in transition
}

// Spot/point shadow + dynamic lights - same model as world_lmap.frag.
float spotShadowF(vec3 wp)
{
    vec4 c = L.spot_vp * vec4(wp, 1.0);
    if (c.w <= 0.0) return 1.0;
    vec3 ndc = c.xyz / c.w;
    vec2 uv = ndc.xy * 0.5 + 0.5;
    uv.y = 1.0 - uv.y;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z > 1.0) return 1.0;
    float ref   = ndc.z - 0.002;
    vec2  texel = 1.0 / vec2(textureSize(uSpotShadow, 0));
    float sum = 0.0;
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x)
            sum += (ref <= texture(uSpotShadow, uv + vec2(x, y) * texel).r) ? 1.0 : 0.0;
    return sum * (1.0 / 9.0);
}

float pointShadowF(vec3 wp, vec3 lp, float range)
{
    vec3 d = wp - lp;
    float z = max(max(abs(d.x), abs(d.y)), abs(d.z));
    const float n = 0.1;
    float refD = range * (z - n) / (max(z, n) * max(range - n, 1e-3));
    return (refD - 0.01 <= texture(uPointShadow, d).r) ? 1.0 : 0.0;
}

vec3 dynLights(vec3 wp, vec3 N)
{
    vec3 acc = vec3(0.0);
    int n = int(L.counts.x + 0.5);
    int sIdx = int(L.shadow_params.x);
    int pIdx = int(L.shadow_params.y);
    for (int i = 0; i < n; ++i) {
        vec3  dv = L.lights[i].pos.xyz - wp;
        float r  = L.lights[i].pos.w;
        float d2 = dot(dv, dv);
        if (d2 >= r * r) continue;
        float d   = sqrt(max(d2, 1e-6));
        vec3  ld  = dv / d;
        float att = 1.0 - d / r;
        att *= att;
        if (L.lights[i].color.w > 0.5)
            att *= clamp((dot(-ld, L.lights[i].dir.xyz) - L.lights[i].dir.w)
                         / max(1.0 - L.lights[i].dir.w, 1e-3), 0.0, 1.0);
        vec3 tint = L.lights[i].color.rgb;
        if (i == sIdx) {
            att *= spotShadowF(wp);
            // Flashlight cookie (R4 projective light texture): the beam pattern
            // projected through the SAME spot_vp the shadow lookup uses.
            if (L.shadow_params.z > 0.5) {
                vec4 cc = L.spot_vp * vec4(wp, 1.0);
                if (cc.w > 0.0) {
                    vec2 cuv = (cc.xy / cc.w) * 0.5 + 0.5;
                    cuv.y = 1.0 - cuv.y;
                    tint *= textureLod(uSpotCookie, clamp(cuv, 0.0, 1.0), 0.0).rgb;
                }
            }
        }
        else if (i == pIdx) att *= pointShadowF(wp, L.lights[i].pos.xyz, r);
        acc += tint * (att * max(dot(N, ld), 0.0));
    }
    return acc;
}

// Bilinear-weighted near-cascade PCF tap (textureGather) - see world_lmap.frag.
float cascTap(sampler2D smap, vec2 uv, float ref)
{
    vec2 sz = vec2(textureSize(smap, 0));
    vec2 t  = uv * sz - 0.5;
    vec2 f  = fract(t);
    vec4 d  = textureGather(smap, (floor(t) + 1.0) / sz, 0);
    vec4 c  = step(vec4(ref), d);
    return mix(mix(c.w, c.z, f.x), mix(c.x, c.y, f.x), f.y);
}

float cascSample(sampler2D smap, mat4 vp, vec3 wp, float bias_)
{
    vec3 n = (vp * vec4(wp, 1.0)).xyz;
    vec2 uv = n.xy * 0.5 + 0.5;
    uv.y = 1.0 - uv.y;
    if (uv.x < 0.01 || uv.x > 0.99 || uv.y < 0.01 || uv.y > 0.99
        || n.z <= 0.0 || n.z >= 1.0)
        return -1.0;
    float ref = n.z - bias_;
    vec2  tx  = 1.0 / vec2(textureSize(smap, 0));
    return 0.25 * (cascTap(smap, uv + vec2(-0.5, -0.5) * tx, ref)
                 + cascTap(smap, uv + vec2( 0.5, -0.5) * tx, ref)
                 + cascTap(smap, uv + vec2(-0.5,  0.5) * tx, ref)
                 + cascTap(smap, uv + vec2( 0.5,  0.5) * tx, ref));
}

// Rain visibility + wet shading - see world_lmap.frag.
float rainVis(vec3 wp)
{
    vec3 n = (L.rain_vp * vec4(wp, 1.0)).xyz;
    vec2 uv = n.xy * 0.5 + 0.5;
    uv.y = 1.0 - uv.y;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || n.z <= 0.0 || n.z >= 1.0)
        return 1.0;
    // WIDE blur — soften the BLOCKY tree-canopy occlusion ("wetness in squares").
    float ref = n.z - 0.0015;
    vec2  px  = 1.0 / vec2(textureSize(uRainMap, 0));
    float s = cascTap(uRainMap, uv, ref) * 2.0
            + cascTap(uRainMap, uv + vec2( 3.0, 0.0) * px, ref)
            + cascTap(uRainMap, uv + vec2(-3.0, 0.0) * px, ref)
            + cascTap(uRainMap, uv + vec2(0.0,  3.0) * px, ref)
            + cascTap(uRainMap, uv + vec2(0.0, -3.0) * px, ref)
            + cascTap(uRainMap, uv + vec2( 2.0,  2.0) * px, ref)
            + cascTap(uRainMap, uv + vec2(-2.0,  2.0) * px, ref)
            + cascTap(uRainMap, uv + vec2( 2.0, -2.0) * px, ref)
            + cascTap(uRainMap, uv + vec2(-2.0, -2.0) * px, ref);
    return s * (1.0 / 10.0);
}

// (Old geometric-dip placement geoPuddle + procedural-sine puddleMask removed —
//  replaced by the SSS procedural placement. See git history / wet_common.glsl.)

// Water DEPTH (metres) from the flow sim, sampled via rain_vp. 0 where dry.
float simWater(vec3 wp)
{
    vec4 c = L.rain_vp * vec4(wp, 1.0);
    if (c.w <= 0.0) return 0.0;
    vec2 uv = c.xy * 0.5 + 0.5; uv.y = 1.0 - uv.y;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return 0.0;
    return textureLod(uWater, uv, 0.0).r;
}

// Blurred water depth for puddle placement (spreads crease/seam line-pooling
// into smooth area puddles) — see world_terrain.frag.
float simWaterSoft(vec3 wp)
{
    vec4 c = L.rain_vp * vec4(wp, 1.0);
    if (c.w <= 0.0) return 0.0;
    vec2 uv = c.xy * 0.5 + 0.5; uv.y = 1.0 - uv.y;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return 0.0;
    vec2 px = 5.0 / vec2(textureSize(uWater, 0));
    return textureLod(uWater, uv, 0.0).r * 0.4
         + (textureLod(uWater, uv + vec2(px.x, 0.0), 0.0).r
          + textureLod(uWater, uv - vec2(px.x, 0.0), 0.0).r
          + textureLod(uWater, uv + vec2(0.0, px.y), 0.0).r
          + textureLod(uWater, uv - vec2(0.0, px.y), 0.0).r) * 0.15;
}

// Water VELOCITY (uv/sec) from the flow sim — drives the moving-water surface.
vec2 simFlow(vec3 wp)
{
    vec4 c = L.rain_vp * vec4(wp, 1.0);
    if (c.w <= 0.0) return vec2(0.0);
    vec2 uv = c.xy * 0.5 + 0.5; uv.y = 1.0 - uv.y;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return vec2(0.0);
    return vec2(textureLod(uFlow, uv, 0.0).r, -textureLod(uFlow, uv, 0.0).g);  // uv-vel -> world XZ (Z axis is flipped)
}

// Ground height (metres, relative) from the rain map ortho depth — flow debug.
float groundHm(vec3 wp)
{
    vec4 c = L.rain_vp * vec4(wp, 1.0);
    if (c.w <= 0.0) return 0.0;
    vec2 uv = c.xy * 0.5 + 0.5; uv.y = 1.0 - uv.y;
    return -textureLod(uRainMap, uv, 0.0).r * 349.0;
}

// Water debug colour. 1 = DEPTH ramp, 2 = FLOW direction (downhill gradient).
vec3 waterDebugColor(vec3 wp, int mode)
{
    float d = simWater(wp);
    if (mode == 2) {
        float e = 0.6;
        float sx1 = groundHm(wp + vec3( e,0,0)) + simWater(wp + vec3( e,0,0));
        float sx0 = groundHm(wp + vec3(-e,0,0)) + simWater(wp + vec3(-e,0,0));
        float sz1 = groundHm(wp + vec3(0,0, e)) + simWater(wp + vec3(0,0, e));
        float sz0 = groundHm(wp + vec3(0,0,-e)) + simWater(wp + vec3(0,0,-e));
        vec2 flow = -vec2(sx1 - sx0, sz1 - sz0);
        float sp  = clamp(length(flow) * 2.0, 0.0, 1.0);
        vec2 dir  = (length(flow) > 1e-5) ? normalize(flow) : vec2(0.0);
        vec3 c = vec3(dir * 0.5 + 0.5, 0.3) * sp;
        return (d > 0.005) ? c : c * 0.15;
    }
    float t = clamp(d / 0.6, 0.0, 1.0);
    vec3 c = vec3(0.0, t * 0.55, t) + vec3(smoothstep(0.75, 1.0, t));
    return (d < 0.005) ? vec3(0.02) : c;
}

// Travelling surface waves ALONG the water flow — see world_terrain.frag.
vec2 flowWaves(vec3 wp, float t)
{
    float d = simWater(wp);
    if (d < 0.003) return vec2(0.0);
    float e = 0.6;
    float sx1 = groundHm(wp + vec3( e,0,0)) + simWater(wp + vec3( e,0,0));
    float sx0 = groundHm(wp + vec3(-e,0,0)) + simWater(wp + vec3(-e,0,0));
    float sz1 = groundHm(wp + vec3(0,0, e)) + simWater(wp + vec3(0,0, e));
    float sz0 = groundHm(wp + vec3(0,0,-e)) + simWater(wp + vec3(0,0,-e));
    vec2 flow = -vec2(sx1 - sx0, sz1 - sz0);
    float spd = length(flow);
    if (spd < 1e-4) return vec2(0.0);
    vec2 dir = flow / spd;
    float along = dot(wp.xz, dir);
    float w = sin(along * 7.0  - t * (2.0 + spd * 30.0))
            + 0.5 * sin(along * 16.0 - t * (3.5 + spd * 50.0) + 1.3);
    float amp = clamp(spd * 6.0, 0.0, 1.0) * clamp(d * 8.0, 0.0, 1.0);
    return dir * (w * amp * 0.5);
}

// rippleLayer/rainRipples + vHash/vNoise/puddlesMaskProc → wet_common.glsl (shared).

vec3 applyWetness(inout vec3 albedo, vec3 wp, vec3 N, float sunMask)
{
    float wet = L.rain_params.y;
    if (wet < 0.005)
        return vec3(0.0);
    wet *= rainVis(wp);
    float upness = clamp(N.y, 0.0, 1.0);
    // Kill wetness on DOWN-facing surfaces (ceilings/overhang undersides).
    float wetK = wet * mix(0.35, 1.0, upness) * smoothstep(-0.15, 0.05, N.y);

    // Puddle coverage: procedural blobs on near-flat up-facing surfaces.
    float slope = clamp((1.0 - max(abs(N.x), abs(N.z)) - 0.9) * 13.0, 0.0, 1.0);
    float cov   = clamp(wet * L.pom_params7.y * 1.5, 0.0, 1.0);   // grows/recedes; ×1.5 = distinct, not fields
    float pud   = (L.pom_params7.x > 0.5) ? puddlesMaskProc(wp.xz, cov, L.pom_params7.w) * slope
                : (L.pom_params6.x > 0.5) ? smoothstep(0.04, 0.12, simWaterSoft(wp)) * upness
                : 0.0;
    pud = clamp(pud, 0.0, 1.0);

    // DARKEN: FULL in deep puddles (pud² → body fills AFTER the shine), ~NONE open.
    albedo *= 1.0 - L.rain_params.z * wetK * mix(0.05, 1.0, pud * pud);

    vec3  toEye    = L.eye_pos.xyz - wp;
    float dist     = length(toEye);
    float reflFade = smoothstep(70.0, 35.0, dist);
    if (wetK * reflFade < 0.004) return vec3(0.0);

    float t = L.sky_params.w;
    float ripFade = smoothstep(18.0, 8.0, dist) * clamp(L.rain_params.x * 1.5 + 0.1, 0.0, 1.0)
                  * smoothstep(0.05, 0.35, pud);
    vec2  vel   = (L.pom_params6.x > 0.5) ? simFlow(wp) : vec2(0.0);
    float velMS = length(vel) * 150.0;
    vec2  scrl  = (velMS > 0.01) ? normalize(vel) * (t * velMS * 0.25) : vec2(0.0);
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
    float reflLod = mix(5.0, 0.0, pud);
    vec3 sky = textureLod(uSky0, R, reflLod).rgb;
    if (xf > 0.01) sky = mix(sky, textureLod(uSky1, R, reflLod).rgb, xf);
    // Water BODY fills later (pud²) than the SHINE (reflection ∝ pud) — "shine first".
    albedo = mix(albedo, albedo * vec3(0.34, 0.40, 0.46) * (1.0 - 0.25 * pud), pud * pud);
    float puddleK = wetK * pud * (0.45 + 0.55 * fres) * reflFade * clamp(L.rain_params.w, 0.0, 2.0);
    // SUN GLINT (SSFX specular_phong) — ripples shatter it to sparkles.
    vec3  Ld    = normalize(-L.sun_dir.xyz);
    vec3  Hh    = normalize(Ld + V);
    float glint = pow(max(dot(Nr, Hh), 0.0), 220.0) * pud * sunMask;
    float foam = smoothstep(3.0, 6.0, velMS) * pud * ripFade * 0.25;
    // RING WAVE crests — bright leading edge of each ripple (visible drop waves).
    vec3 crestCol = (L.sun_color.rgb + L.ambient.rgb) * (crest * pud * 0.07);
    return sky * puddleK + L.sun_color.rgb * (glint * 3.0) + vec3(foam) + crestCol;
}

// NEAR cascade first (leaf-shaped dapples, smooth motion) - see world_lmap.frag.
float sunShadow(vec3 worldPos)
{
    float s = cascSample(uShadowNear, L.sun_near_vp, worldPos, 0.0004);
    if (s >= 0.0) return s;
    s = cascSample(uShadowC1, L.sun_c1_vp, worldPos, 0.0006);
    if (s >= 0.0) return s;

    vec4 c = L.sun_vp * vec4(worldPos, 1.0);
    if (c.w <= 0.0) return 1.0;
    vec3 ndc = c.xyz / c.w;
    vec2 uv = ndc.xy * 0.5 + 0.5;
    uv.y = 1.0 - uv.y;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z > 1.0) return 1.0;
    float ref   = ndc.z - 0.0015;
    // 4 spread taps (was 3-3) - see world_lmap.frag.
    vec2  texel = 1.0 / vec2(textureSize(uShadow, 0));
    float sum = 0.0;
    sum += (ref <= texture(uShadow, uv + vec2(-0.75, -0.75) * texel).r) ? 1.0 : 0.0;
    sum += (ref <= texture(uShadow, uv + vec2( 0.75, -0.75) * texel).r) ? 1.0 : 0.0;
    sum += (ref <= texture(uShadow, uv + vec2(-0.75,  0.75) * texel).r) ? 1.0 : 0.0;
    sum += (ref <= texture(uShadow, uv + vec2( 0.75,  0.75) * texel).r) ? 1.0 : 0.0;
    return sum * 0.25;
}

layout(push_constant) uniform PushConstants {
    mat4  mvp;
    vec2  uvScale;
    float alphaRef;
    float detailScale;
    float dynHemi;       // sky-ambient gate: 1.0 statics, ray-traced 0..1 for dynamics
} pc;

layout(location = 0) in  vec2  vUV;
layout(location = 1) in  vec2  vDetailUV;
layout(location = 2) in  vec3  vBakedColor;
layout(location = 3) in  float vSunMask;
layout(location = 4) in  vec3  vWorldPos;
layout(location = 5) in  vec3  vNormal;
layout(location = 0) out vec4  outColor;

// POM heightfield from DIFFUSE LUMINANCE (no authored heightmaps on this
// content; the `#` alpha is low-contrast) — see world_lmap.frag.
// Height from the `#` alpha, high-passed - see world_lmap.frag (albedo height
// made dark/light bricks into false cliffs → spikes; `#` is geometry-based).
float pomDepth(vec2 uv, float lod, float baseline)
{
    float h = textureLod(uTexBumpX, uv, lod).a;
    return clamp(0.5 - (h - baseline) * 4.0, 0.0, 1.0);
}

// Parallax occlusion (relief) mapping - see world_lmap.frag for full comments.
vec2 parallaxUV(vec2 uv, vec3 N, vec3 wp, out vec3 outN, out float outShadow, out float outAO)
{
    outN = N;
    outShadow = 1.0;
    outAO = 1.0;
    float amp = L.pom_params.x;
    if (amp <= 0.0) return uv;
    if (textureLod(uTexBumpX, vec2(0.5), 8.0).a > 0.985) return uv;   // flat material → no POM
    float dist = length(L.eye_pos.xyz - wp);
    float fade = 1.0 - smoothstep(L.pom_params.z * 0.5, L.pom_params.z, dist);   // full to far*0.5, gone by far
    amp *= fade;
    // Per-orientation strength (floor→.w, ceiling→.z) - see world_lmap.frag.
    float orient = (N.y >= 0.0)
        ? mix(1.0, L.pom_params3.w, clamp( N.y, 0.0, 1.0))
        : mix(1.0, L.pom_params3.z, clamp(-N.y, 0.0, 1.0));
    amp *= orient;
    if (amp <= 1e-5) return uv;

    vec2 tsz  = vec2(textureSize(uTexBumpX, 0));
    vec2 ddx  = dFdx(uv) * tsz, ddy = dFdy(uv) * tsz;
    float lod = max(0.5 * log2(max(dot(ddx, ddx), dot(ddy, ddy))), 0.0) + L.pom_params2.x;
    float baseline = textureLod(uTexBumpX, uv, lod + 3.0).a;

    vec3 dp1 = dFdx(wp),  dp2 = dFdy(wp);
    vec2 du1 = dFdx(uv),  du2 = dFdy(uv);
    vec3 dp2p = cross(dp2, N), dp1p = cross(N, dp1);
    vec3 T = dp2p * du1.x + dp1p * du2.x;
    vec3 B = dp2p * du1.y + dp1p * du2.y;
    float inv = inversesqrt(max(dot(T, T), dot(B, B)));
    T *= inv; B *= inv;

    vec3 V   = normalize(L.eye_pos.xyz - wp);
    vec3 Vts = vec3(dot(V, T), dot(V, B), dot(V, N));
    vec2 Pmax = (Vts.xy / max(abs(Vts.z), 0.3)) * amp;

    // FEWER steps as POM fades with distance (rides the same `fade` that shrinks
    // amp → invisible cut, halves the mid-distance march). Near stays full.
    int steps = int(clamp(mix(L.pom_params.y, 12.0, abs(Vts.z)) * mix(0.5, 1.0, fade), 8.0, 64.0));
    float layerH = 1.0 / float(steps);
    vec2 dUV = Pmax * layerH;

    float curD = 0.0;
    vec2  curUV = uv;
    float curH = pomDepth(curUV, lod, baseline);
    for (int i = 0; i < 64; ++i) {
        if (i >= steps || curD >= curH) break;
        curUV -= dUV;
        curD  += layerH;
        curH   = pomDepth(curUV, lod, baseline);
    }
    vec2 sUV = dUV; float sD = layerH;
    for (int j = 0; j < 6; ++j) {
        sUV *= 0.5; sD *= 0.5;
        if (curD < pomDepth(curUV, lod, baseline)) { curUV -= sUV; curD += sD; }
        else                                       { curUV += sUV; curD -= sD; }
    }
    // Perturbed normal from the height gradient — see world_lmap.frag.
    float tU = exp2(lod) / tsz.x, tV = exp2(lod) / tsz.y;
    float hu = textureLod(uTexBumpX, curUV + vec2(tU, 0.0), lod).a
             - textureLod(uTexBumpX, curUV - vec2(tU, 0.0), lod).a;
    float hv = textureLod(uTexBumpX, curUV + vec2(0.0, tV), lod).a
             - textureLod(uTexBumpX, curUV - vec2(0.0, tV), lod).a;
    float ns = L.pom_params2.y * 12.0 * fade * orient;
    vec3 nTS = normalize(vec3(-hu * ns, -hv * ns, 1.0));
    outN = normalize(T * nTS.x + B * nTS.y + N * nTS.z);

    // Self-shadow toward the sun - see world_lmap.frag.
    if (L.pom_params2.z > 0.0) {
        vec3 Ld  = normalize(-L.sun_dir.xyz);
        vec3 Lts = vec3(dot(Ld, T), dot(Ld, B), dot(Ld, N));
        vec2 lxy = Lts.xy;
        if (Lts.z > 0.02 && dot(lxy, lxy) > 1e-6) {
            // Horizon self-shadow toward the sun - see world_lmap.frag.
            vec2  sdir  = normalize(lxy);
            float reach = (exp2(lod) / min(tsz.x, tsz.y)) * 6.0;
            float h0    = textureLod(uTexBumpX, curUV, lod).a;
            float occ   = 0.0;
            for (int s = 1; s <= 8; ++s) {
                float hs = textureLod(uTexBumpX, curUV + sdir * reach * (float(s) * 0.125), lod).a;
                occ = max(occ, hs - h0);
            }
            outShadow = clamp(1.0 - occ * L.pom_params2.z * 20.0 * (1.0 - Lts.z) * orient, 0.0, 1.0);
        }
    }
    // POM-AO (view-independent contact occlusion) - see world_lmap.frag.
    if (L.pom_params2.w > 0.0) {
        float aoReach = (exp2(lod) / min(tsz.x, tsz.y)) * 4.0;
        float h0 = textureLod(uTexBumpX, curUV, lod).a;
        float aoSum =
              max(0.0, textureLod(uTexBumpX, curUV + vec2( aoReach, 0.0), lod).a - h0)
            + max(0.0, textureLod(uTexBumpX, curUV + vec2(-aoReach, 0.0), lod).a - h0)
            + max(0.0, textureLod(uTexBumpX, curUV + vec2(0.0,  aoReach), lod).a - h0)
            + max(0.0, textureLod(uTexBumpX, curUV + vec2(0.0, -aoReach), lod).a - h0);
        outAO = clamp(1.0 - (aoSum * 0.25) * L.pom_params2.w * 6.0, 0.35, 1.0);
        outAO = mix(1.0, outAO, orient);   // ceilings: dial contact AO down too
    }
    return curUV;
}

void main()
{
    vec3 pomN; float pomShadow, pomAO;
    vec2 pUV = parallaxUV(vUV, normalize(vNormal), vWorldPos, pomN, pomShadow, pomAO);
    vec2 pDetailUV = vDetailUV + (pUV - vUV) * pc.detailScale;

    vec4 base   = texture(uTexDiffuse, pUV);
    if (pc.alphaRef >= 0.0 && base.a < pc.alphaRef) discard;

    // r_ssao_debug 1: show the raw AO map - see world_lmap.frag.
    if (L.ao_params.w > 0.5) {
        outColor = vec4(vec3(textureLod(uAO, gl_FragCoord.xy * L.ao_params.xy, 0.0).r), base.a);
        return;
    }

    // r_wet_debug 1: rain-map visibility - see world_lmap.frag.
    if (L.rain_params.z < 0.0) {
        outColor = vec4(vec3(rainVis(vWorldPos)), base.a);
        return;
    }

    // r_puddle_debug: sim on → depth/flow; else the SSS puddle coverage (grayscale).
    int pdbg = int(L.pom_params5.w + 0.5);
    if (pdbg > 0) {
        outColor = (L.pom_params6.x > 0.5)
            ? vec4(waterDebugColor(vWorldPos, pdbg), base.a)
            : vec4(vec3(puddlesMaskProc(vWorldPos.xz, clamp(L.rain_params.y * L.pom_params7.y * 1.5, 0.0, 1.0), L.pom_params7.w)), base.a);
        return;
    }

    // r_pom_debug 1: POM occlusion mask (contact AO × sun self-shadow).
    if (L.pom_params3.x > 0.5) {
        outColor = vec4(vec3(pomAO * pomShadow), base.a);
        return;
    }

    vec3 detail = texture(uTexDetail, pDetailUV).rgb;
    vec3 albedo = 2.0 * base.rgb * detail;

    // Lighting: baked vertex color (point lights + bounce, already coloured) +
    // env sun (gated by the baked sun mask) + flat env hemi/ambient sky term.
    // vlit has no per-texel hemi occlusion (no lightmap), so the sky term is a
    // crude flat hemi*0.5; vBakedColor carries the local point-light detail.
    // Small floor avoids pitch-black. Matches world_lmap's env-driven model.
    // DYNAMIC R4-style sun: per-pixel N-L - shadow map (the baked per-vertex
    // vSunMask was frozen at the bake's sun angle - dawn/dusk never showed).
    vec3  geomN   = normalize(vNormal);   // flat — for the sky fill (sharp cube → perturbed = mirror)
    vec3  Nw      = pomN;   // POM-perturbed normal → sun + dyn lights catch the relief
    float sunMask = max(dot(Nw, normalize(-L.sun_dir.xyz)), 0.0);
    if (sunMask > 0.005)
        sunMask *= sunShadow(vWorldPos) * pomShadow;   // cascade × POM groove self-shadow

    // vlit has NO lightmap occlusion, so the dynamic sky fill was applied
    // unoccluded - vertex-lit interior props (tables, mattresses) glowed in a
    // dark basement. Gate it by BOTH: pc.dynHemi (ray-traced sky visibility for
    // DYNAMIC objects) AND the baked vertex brightness (occlusion for STATIC
    // vlit geometry - dark bake = enclosed). A small floor keeps unlit-but-open
    // outdoor vlit from going black if its bake carries no hemi.
    float bakeOcc = clamp(dot(vBakedColor, vec3(0.299, 0.587, 0.114)) * 2.5, 0.15, 1.0);
    // sun_color/ambient arrive final from vk_env_light (r_sun_boost /
    // r_ambient_floor). vBakedColor -1.5 is a vlit-specific baked-light scale.
    vec3  occ      = coloredAO(gtaoVis(), albedo) * pomAO;   // GTAO × fine POM contact AO
    float dynHemiL = pc.dynHemi;
    if (L.pom_params3.y > 0.5) { occ = vec3(1.0); dynHemiL = 1.0; bakeOcc = 1.0; }   // r_ao_flat debug
    vec3 lighting = vBakedColor * 1.5
                  + L.sun_color.rgb  * sunMask
                  + skyAmbient(geomN) * (L.sky_params.y * 0.5 * dynHemiL * bakeOcc) * occ
                  + L.ambient.rgb * occ
                  + dynLights(vWorldPos, Nw);

    // Rain wetness - see world_lmap.frag.
    vec3 wetRefl = applyWetness(albedo, vWorldPos, Nw, sunMask);

    // Distance fog (R4) - see world_lmap.frag.
    vec3 col = albedo * lighting + wetRefl;
    float fog = clamp(length(vWorldPos - L.eye_pos.xyz) * L.fog_params.w + L.fog_params.x, 0.0, 1.0);
    col = mix(col, L.fog_color.rgb, fog);

    outColor = vec4(col, base.a);
}
