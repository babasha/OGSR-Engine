#version 450

// World pass — vert-lit variant. Final colour = albedo × pre-baked
// vertex lighting + small ambient floor.
//
// vBakedColor is what the level compiler computed offline (point-light
// contributions + bounce). It does NOT include direct sun — that needs
// runtime sun direction, deferred until env subsystem lands. As an
// approximation we add a constant ambient term and a sun-direction-
// independent fake-sun term gated by the per-vertex sun mask.
//
// Binding 2 (lmap) is bound for descriptor-layout compatibility but
// not sampled here — vert-lit materials don't have a lightmap.

layout(set = 0, binding = 0) uniform sampler2D uTexDiffuse;
layout(set = 0, binding = 1) uniform sampler2D uTexDetail;
layout(set = 0, binding = 2) uniform sampler2D uTexLmap;  // unused (white fallback)

// Per-frame environment lighting (set 1) — see vk_env_light.{h,cpp}.
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
    mat4 sun_vp;      // sun light view·proj (shadow lookup)
    vec4 counts;      // x = dynamic light count
    DynLight lights[16];
    mat4 spot_vp;        // spot (flashlight) shadow view·proj
    vec4 shadow_params;  // x = spot-shadowed light index (-1 none), y = point-shadowed index
    mat4 sun_near_vp;    // sun cascade 0 view·proj (25 m, per-frame, R4 scheme)
    mat4 sun_c1_vp;      // sun cascade 1 view·proj (60 m, per-frame)
    vec4 fog_color;      // rgb haze colour (env)
    vec4 fog_params;     // x=-near*r, y=near, z=far, w=r; fog = saturate(dist*w + x)
    vec4 eye_pos;        // xyz camera world pos
    vec4 sky_params;     // x=cube cross-fade weight, y=ambient scale, z=sample LOD
    vec4 ao_params;      // x=1/screenW, y=1/screenH, z=AO strength (0=off)
} L;
layout(set = 1, binding = 1) uniform sampler2D uShadow;
layout(set = 1, binding = 2) uniform sampler2D uSpotShadow;
layout(set = 1, binding = 3) uniform samplerCube uPointShadow;
layout(set = 1, binding = 4) uniform sampler2D uShadowNear;    // sun cascade 0 (~0.61 cm texels)
layout(set = 1, binding = 5) uniform sampler2D uShadowC1;      // sun cascade 1 (~1.46 cm texels)
layout(set = 1, binding = 6) uniform samplerCube uSky0;        // sky ambient cube 0 (weather A)
layout(set = 1, binding = 7) uniform samplerCube uSky1;        // sky ambient cube 1 (weather B)
layout(set = 1, binding = 8) uniform sampler2D uAO;            // GTAO (half-res)

// GTAO visibility — see world_lmap.frag (occludes hemi+ambient only).
float gtaoVis()
{
    float ao = textureLod(uAO, gl_FragCoord.xy * L.ao_params.xy, 0.0).r;
    return pow(clamp(ao, 0.0, 1.0), L.ao_params.z);   // strength = exponent (0 = off)
}

// Colored AO — see world_lmap.frag (R4 compute_colored_ao port).
vec3 coloredAO(float ao, vec3 albedo)
{
    vec3 a =  2.0404 * albedo - 0.3324;
    vec3 b = -4.7951 * albedo + 0.6417;
    vec3 c =  2.7552 * albedo + 0.6903;
    return max(vec3(ao), ((ao * a + b) * ao + c) * ao);
}

// Hemisphere sky ambient (R4 hmodel.h) — see world_lmap.frag.
vec3 skyAmbient(vec3 N)
{
    float lod = L.sky_params.z;
    return mix(textureLod(uSky0, N, lod).rgb, textureLod(uSky1, N, lod).rgb,
               clamp(L.sky_params.x, 0.0, 1.0));
}

// Spot/point shadow + dynamic lights — same model as world_lmap.frag.
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
        if (i == sIdx)      att *= spotShadowF(wp);
        else if (i == pIdx) att *= pointShadowF(wp, L.lights[i].pos.xyz, r);
        acc += L.lights[i].color.rgb * (att * max(dot(N, ld), 0.0));
    }
    return acc;
}

// Bilinear-weighted near-cascade PCF tap (textureGather) — see world_lmap.frag.
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

// NEAR cascade first (leaf-shaped dapples, smooth motion) — see world_lmap.frag.
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
    // 4 spread taps (was 3×3) — see world_lmap.frag.
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

void main()
{
    vec4 base   = texture(uTexDiffuse, vUV);
    if (pc.alphaRef >= 0.0 && base.a < pc.alphaRef) discard;

    // r_ssao_debug 1: show the raw AO map — see world_lmap.frag.
    if (L.ao_params.w > 0.5) {
        outColor = vec4(vec3(textureLod(uAO, gl_FragCoord.xy * L.ao_params.xy, 0.0).r), base.a);
        return;
    }

    vec3 detail = texture(uTexDetail, vDetailUV).rgb;
    vec3 albedo = 2.0 * base.rgb * detail;

    // Lighting: baked vertex color (point lights + bounce, already coloured) +
    // env sun (gated by the baked sun mask) + flat env hemi/ambient sky term.
    // vlit has no per-texel hemi occlusion (no lightmap), so the sky term is a
    // crude flat hemi*0.5; vBakedColor carries the local point-light detail.
    // Small floor avoids pitch-black. Matches world_lmap's env-driven model.
    // DYNAMIC R4-style sun: per-pixel N·L × shadow map (the baked per-vertex
    // vSunMask was frozen at the bake's sun angle — dawn/dusk never showed).
    vec3  Nw      = normalize(vNormal);
    float sunMask = max(dot(Nw, normalize(-L.sun_dir.xyz)), 0.0);
    if (sunMask > 0.005)
        sunMask *= sunShadow(vWorldPos);

    // vlit has NO lightmap occlusion, so the dynamic sky fill was applied
    // unoccluded → vertex-lit interior props (tables, mattresses) glowed in a
    // dark basement. Gate it by BOTH: pc.dynHemi (ray-traced sky visibility for
    // DYNAMIC objects) AND the baked vertex brightness (occlusion for STATIC
    // vlit geometry — dark bake = enclosed). A small floor keeps unlit-but-open
    // outdoor vlit from going black if its bake carries no hemi.
    float bakeOcc = clamp(dot(vBakedColor, vec3(0.299, 0.587, 0.114)) * 2.5, 0.15, 1.0);
    // Sun ×1.25 — match world_lmap (R4 reads a touch brighter in direct sun).
    vec3 occ = coloredAO(gtaoVis(), albedo);
    vec3 lighting = vBakedColor * 1.5
                  + L.sun_color.rgb  * (sunMask * 1.25)
                  + skyAmbient(Nw) * (L.sky_params.y * 0.5 * pc.dynHemi * bakeOcc) * occ
                  + (L.ambient.rgb + 0.05) * occ
                  + dynLights(vWorldPos, Nw);

    // Distance fog (R4) — see world_lmap.frag.
    vec3 col = albedo * lighting;
    float fog = clamp(length(vWorldPos - L.eye_pos.xyz) * L.fog_params.w + L.fog_params.x, 0.0, 1.0);
    col = mix(col, L.fog_color.rgb, fog);

    outColor = vec4(col, base.a);
}
