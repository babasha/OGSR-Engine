#version 450
#extension GL_GOOGLE_include_directive : require
#include "vsm_sample.glsl"         // vsmSunShadow (set 1 b14)
#include "light_ubo.glsl"          // DynLight + Lighting UBO (set 1 b0) + samplers (set 1 b1..13)
#include "surface_class.glsl"      // SC_* classification + snow (grass)

// xrRenderVulkan - detail (grass) fragment shader. set 0/binding 0 = per-type
// diffuse (alpha-cutoff 0.5, punch-out). set 1 = the shared EnvLight, now via
// light_ubo.glsl (was a hand-rolled partial copy). Sun gets a single shadow tap;
// grass overdraw noise hides the hard edge.
layout(set = 0, binding = 0) uniform sampler2D uDiffuse;

// Rain visibility - is this blade open to the sky? Single tap (grass overdraw is noisy).
float rainVisGrass(vec3 wp)
{
    vec3 n = (L.rain_vp * vec4(wp, 1.0)).xyz;
    vec2 uv = n.xy * 0.5 + 0.5; uv.y = 1.0 - uv.y;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || n.z <= 0.0 || n.z >= 1.0)
        return 1.0;
    return (n.z - 0.0015 <= texture(uRainMap, uv).r) ? 1.0 : 0.0;
}

// GTAO at this pixel - grass isn't in the prepass, so this is the AO of the GROUND
// behind the blade (grass in a dark corner sits in the same ambient as the dirt).
float gtaoVis()
{
    float ao = textureLod(uAO, gl_FragCoord.xy * L.ao_params.xy, 0.0).r;
    return pow(clamp(ao, 0.0, 1.0), L.ao_params.z);
}

// Colored AO - see world_lmap.frag (R4 tints occlusion toward the blade's albedo).
vec3 coloredAO(float ao, vec3 albedo)
{
    vec3 a =  2.0404 * albedo - 0.3324;
    vec3 b = -4.7951 * albedo + 0.6417;
    vec3 c =  2.7552 * albedo + 0.6903;
    return max(vec3(ao), ((ao * a + b) * ao + c) * ao);
}

// SSIL ambient boost - see env_common.glsl (SSFX hdiffuse *= IL). 0/no-op where
// there's no bounce or r_ssil is off; uIL is binding 21 (light_ubo.glsl).
vec3 ssilBoost()
{
    vec3 il = textureLod(uIL, gl_FragCoord.xy * L.ao_params.xy, 0.0).rgb;
    return vec3(1.0) + il / (1.0 + il);
}

// Hemisphere sky ambient - grass samples straight up (no real normal), same light
// as the terrain beneath it.
vec3 skyAmbientUp()
{
    float lod = L.sky_params.z;
    const vec3 up = vec3(0.0, 1.0, 0.0);
    return mix(textureLod(uSky0, up, lod).rgb, textureLod(uSky1, up, lod).rgb,
               clamp(L.sky_params.x, 0.0, 1.0));
}

// Foliage variant: billboards have no meaningful normal -> attenuation-only.
vec3 dynLightsFoliage(vec3 wp)
{
    vec3 acc = vec3(0.0);
    int n = int(L.counts.x + 0.5);
    for (int i = 0; i < n; ++i) {
        vec3  dv = L.lights[i].pos.xyz - wp;
        float r  = L.lights[i].pos.w;
        float d2 = dot(dv, dv);
        if (d2 >= r * r) continue;
        float d   = sqrt(max(d2, 1e-6));
        float att = 1.0 - d / r;
        att *= att;
        if (L.lights[i].color.w > 0.5)
            att *= clamp((dot(-dv / d, L.lights[i].dir.xyz) - L.lights[i].dir.w)
                         / max(1.0 - L.lights[i].dir.w, 1e-3), 0.0, 1.0);
        acc += L.lights[i].color.rgb * (att * 0.7);
    }
    return acc;
}

layout(push_constant) uniform DetailConstants {
    mat4 mViewProj;
    vec4 wind_params;       // vertex-only (kept for push layout parity)
    vec4 wsetup_grass;      // vertex-only
    vec4 wind_anim;         // vertex-only
    vec4 vConsts;
    vec4 vInteractors[4];
    vec4 vSunColor;
    vec4 vHemiColor;
} pc;

layout(location = 0) in vec2  vUV;
layout(location = 1) in vec4  vColor;
layout(location = 2) in float vHeight;
layout(location = 3) in vec3  vSunLit;
layout(location = 4) in vec3  vWPos;
layout(location = 0) out vec4 outColor;

// 1-tap sun shadow - see world_lmap.frag.
float sunShadow1(vec3 worldPos)
{
    vec4 c = L.sun_vp * vec4(worldPos, 1.0);
    if (c.w <= 0.0) return 1.0;
    vec3 ndc = c.xyz / c.w;
    vec2 uv = ndc.xy * 0.5 + 0.5;
    uv.y = 1.0 - uv.y;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z > 1.0) return 1.0;
    return (ndc.z - 0.0015 <= texture(uShadow, uv).r) ? 1.0 : 0.0;
}

void main()
{
    vec4 diff = texture(uDiffuse, vUV);
    // Mip-fade compensation (r_grass_asharp, pc.vConsts.y), DISTANCE-driven: alpha
    // MIPs average the blade against its transparent background, so far grass
    // dissolves see-through. NOT lod-driven — textureQueryLod is high on thin
    // blades even right at the camera (minification is geometric, not distance),
    // which striped/fringed near grass in-game. Distance ramp: untouched inside
    // 15 m, full boost (1 + 4·asharp) by 60 m.
    diff.a *= 1.0 + smoothstep(15.0, 60.0, length(vWPos - L.eye_pos.xyz)) * 4.0 * pc.vConsts.y;
    // Alpha-test cutoff is live-tunable via r_grass_aref (pc.vConsts.x). Lower =
    // fatter/denser blades (kills the "see-through" look); fall back to 0.5 if
    // the push slot is ever zero.
    float aref = pc.vConsts.x > 0.001 ? pc.vConsts.x : 0.5;
    if (diff.a < aref) discard;

    // r_ssao_debug 1: grass shows the raw AO map too.
    if (L.ao_params.w > 0.5) {
        outColor = vec4(vec3(textureLod(uAO, gl_FragCoord.xy * L.ao_params.xy, 0.0).r), 1.0);
        return;
    }
    // r_sf_debug 5: surface classification - grass (cyan).
    if (int(L.sf_params.y + 0.5) == 5) {
        outColor = vec4(SC_DebugColor(SC_Refine(SC_GRASS, vec3(0.0, 1.0, 0.0))), 1.0);
        return;
    }

    // SNOW cover on the grass (Surface Field consumer, r_snow), gated by sky
    // exposure so grass under cover stays green.
    float snow = SC_SnowAmount(SC_Refine(SC_GRASS, vec3(0.0, 1.0, 0.0)), vec3(0.0, 1.0, 0.0), rainVisGrass(vWPos)) * clamp(L.sf_params.w, 0.0, 1.0);
    diff.rgb = mix(diff.rgb, vec3(0.92, 0.94, 0.98), snow);

    vec3 sunPart = vSunLit;
    if (dot(sunPart, sunPart) > 0.0) {
        // VSM screen-space mask (r_vsm) vs the 1-tap cascade - gated by shadow_params.w.
        float sunSh = (L.shadow_params.w > 0.5) ? vsmSunShadow(gl_FragCoord.xy * L.ao_params.xy)
                                                 : sunShadow1(vWPos);
        sunPart *= sunSh;
    }

    // Ambient = sky-cube light x baked per-slot hemi occlusion (vColor.r). L.ambient
    // = real env ambient (carries r_ambient_floor). GTAO gates it (sun/dyn untouched).
    vec3 ambient = (skyAmbientUp() * (vColor.r * L.sky_params.y) + L.ambient.rgb)
                 * coloredAO(gtaoVis(), diff.rgb) * ssilBoost();   // + SSIL bounce (ambient only)

    vec3 col = diff.rgb * (ambient + sunPart + dynLightsFoliage(vWPos));

    // Wet grass: blades open to the rain darken slightly (+ a touch cooler).
    float wet = clamp(L.rain_params.y, 0.0, 1.0) * max(L.rain_params.z, 0.0);
    if (wet > 0.005) {
        float wv = wet * rainVisGrass(vWPos);
        col *= 1.0 - 0.5 * wv;
        col = mix(col, col * vec3(0.85, 0.92, 1.0), 0.5 * wv);
    }

    // Distance fog (R4).
    float fog = clamp(length(vWPos - L.eye_pos.xyz) * L.fog_params.w + L.fog_params.x, 0.0, 1.0);
    col = mix(col, L.fog_color.rgb, fog);

    outColor = vec4(col, 1.0);
}
