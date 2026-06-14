#version 450
// xrRenderVulkan — detail (grass) fragment shader.
// set=0/binding=0 = per-type diffuse texture. Hard alpha-cutoff at 0.5 because
// grass billboards are punch-out, not blended.
// set=1 = shared per-frame env lighting (vk_env_light): sun_vp + the sun shadow
// map. The sun part of the vertex lighting (vSunLit) is attenuated by a single
// shadow tap — foliage noise hides the hard edge, and 9-tap PCF on the grass
// overdraw would be wasted.

layout(set = 0, binding = 0) uniform sampler2D uDiffuse;

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
    // Padding to reach the fog block at the end of the shared LightUBO — grass
    // doesn't sample the cascade maps, it just needs fog_color/params/eye.
    mat4 _pad_spot_vp;
    vec4 _pad_shadow_params;
    mat4 _pad_sun_near_vp;
    mat4 _pad_sun_c1_vp;
    vec4 fog_color;      // rgb haze colour (env)
    vec4 fog_params;     // x=-near*r, y=near, z=far, w=r; fog = saturate(dist*w + x)
    vec4 eye_pos;        // xyz camera world pos
    vec4 sky_params;     // x=cube cross-fade weight, y=ambient scale, z=sample LOD
    vec4 ao_params;      // x=1/screenW, y=1/screenH, z=AO strength exponent (0=off)
} L;
layout(set = 1, binding = 1) uniform sampler2D uShadow;
layout(set = 1, binding = 6) uniform samplerCube uSky0;   // sky ambient cube (weather A)
layout(set = 1, binding = 7) uniform samplerCube uSky1;   // sky ambient cube (weather B)
layout(set = 1, binding = 8) uniform sampler2D uAO;       // GTAO (half-res)

// GTAO at this pixel — grass isn't in the prepass depth, so this is the AO of
// the GROUND behind the blade: grass in a dark corner sits in the same ambient
// as the dirt it grows from (full strength, matches world_terrain).
float gtaoVis()
{
    float ao = textureLod(uAO, gl_FragCoord.xy * L.ao_params.xy, 0.0).r;
    return pow(clamp(ao, 0.0, 1.0), L.ao_params.z);
}

// Colored AO — see world_lmap.frag. R4 applies this to grass too (deffer_grass
// writes the gbuffer, combine_1.ps tints occlusion by the BLADE's albedo), so
// occluded grass shades toward its own green instead of grey.
vec3 coloredAO(float ao, vec3 albedo)
{
    vec3 a =  2.0404 * albedo - 0.3324;
    vec3 b = -4.7951 * albedo + 0.6417;
    vec3 c =  2.7552 * albedo + 0.6903;
    return max(vec3(ao), ((ao * a + b) * ao + c) * ao);
}

// Hemisphere sky ambient — the SAME light the world ground uses (world_lmap
// skyAmbient). Grass has no real normal: sample straight up, like the terrain
// beneath it, so blades sit in the same light as the ground they grow from.
vec3 skyAmbientUp()
{
    float lod = L.sky_params.z;
    const vec3 up = vec3(0.0, 1.0, 0.0);
    return mix(textureLod(uSky0, up, lod).rgb, textureLod(uSky1, up, lod).rgb,
               clamp(L.sky_params.x, 0.0, 1.0));
}

// Foliage variant: billboards have no meaningful normal → attenuation-only
// (×0.7 stands in for the average N·L of randomly-oriented blades).
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
    vec4 vWave;
    vec4 vWind;
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

// 1-tap sun shadow (same projection convention as world_lmap.frag).
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
    if (diff.a < 0.5) discard;

    // r_ssao_debug 1: grass shows the raw AO map too — see world_lmap.frag.
    if (L.ao_params.w > 0.5) {
        outColor = vec4(vec3(textureLod(uAO, gl_FragCoord.xy * L.ao_params.xy, 0.0).r), 1.0);
        return;
    }

    vec3 sunPart = vSunLit;
    if (dot(sunPart, sunPart) > 0.0)
        sunPart *= sunShadow1(vWPos);

    // Ambient = sky-cube light × baked per-slot hemi occlusion (vColor.r) —
    // matches world_lmap's skyAmbient(N)*(hemiOcc*sky_params.y) on the ground.
    // L.ambient replaces the old literal +0.05 floor: grass now gets the real
    // env ambient (which carries r_ambient_floor) like every other receiver.
    // GTAO gates it like every other receiver (sun/dyn lights untouched).
    vec3 ambient = (skyAmbientUp() * (vColor.r * L.sky_params.y) + L.ambient.rgb)
                 * coloredAO(gtaoVis(), diff.rgb);

    vec3 col = diff.rgb * (ambient + sunPart + dynLightsFoliage(vWPos));

    // Distance fog (R4) — see world_lmap.frag.
    float fog = clamp(length(vWPos - L.eye_pos.xyz) * L.fog_params.w + L.fog_params.x, 0.0, 1.0);
    col = mix(col, L.fog_color.rgb, fog);

    outColor = vec4(col, 1.0);
}
