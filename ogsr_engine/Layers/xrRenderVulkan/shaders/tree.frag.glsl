#version 450
// xrRenderVulkan — tree forward fragment shader (Session B).
// set 1/binding 0 = per-group diffuse texture. Trees are alpha-tested
// (punch-out leaves), not blended; discard below the push-constant cutoff.
// set 2 = shared per-frame env lighting (vk_env_light): sun_vp + the sun shadow
// map. The sun part of the vertex lighting (vSunLit) gets a single shadow tap —
// foliage noise hides the hard edge.

layout(set = 1, binding = 0) uniform sampler2D uDiffuse;

struct DynLight {
    vec4 pos;     // xyz = world position, w = range
    vec4 color;   // rgb = colour,         w = 1 spot / 0 point
    vec4 dir;     // xyz = spot direction, w = cos(cone/2)
};
layout(set = 2, binding = 0) uniform Lighting {
    vec4 sun_dir;
    vec4 sun_color;
    vec4 hemi_color;
    vec4 ambient;
    mat4 sun_vp;      // sun light view·proj (shadow lookup)
    vec4 counts;      // x = dynamic light count
    DynLight lights[16];
    // Padding to reach the tail of the shared LightUBO.
    mat4 _pad_spot_vp;
    vec4 _pad_shadow_params;
    mat4 sun_near_vp;    // sun cascade 0 view·proj (25 m, per-frame)
    mat4 sun_c1_vp;      // sun cascade 1 view·proj (60 m, per-frame)
    vec4 fog_color;      // rgb haze colour (env)
    vec4 fog_params;     // x=-near*r, y=near, z=far, w=r; fog = saturate(dist*w + x)
    vec4 eye_pos;        // xyz camera world pos
    vec4 sky_params;     // x=cube cross-fade weight, y=ambient scale, z=sample LOD
    vec4 ao_params;      // x=1/screenW, y=1/screenH, z=AO strength exponent (0=off)
} L;
layout(set = 2, binding = 1) uniform sampler2D uShadow;
layout(set = 2, binding = 4) uniform sampler2D uShadowNear;  // sun cascade 0 (~0.61 cm texels)
layout(set = 2, binding = 5) uniform sampler2D uShadowC1;    // sun cascade 1 (~1.46 cm texels)
layout(set = 2, binding = 6) uniform samplerCube uSky0;   // sky ambient cube (weather A)
layout(set = 2, binding = 7) uniform samplerCube uSky1;   // sky ambient cube (weather B)
layout(set = 2, binding = 8) uniform sampler2D uAO;       // GTAO (half-res)

// GTAO — trees ARE in the prepass depth (RenderDepth), so crowns/trunks get
// their own AO (denser crown interior reads darker). HALF the strength
// exponent like skinned: alpha-tested leaf depth is noisy at half-res.
float gtaoVis()
{
    float ao = textureLod(uAO, gl_FragCoord.xy * L.ao_params.xy, 0.0).r;
    return pow(clamp(ao, 0.0, 1.0), L.ao_params.z * 0.5);
}

// Colored AO — see world_lmap.frag (R4 tints foliage occlusion by its albedo).
vec3 coloredAO(float ao, vec3 albedo)
{
    vec3 a =  2.0404 * albedo - 0.3324;
    vec3 b = -4.7951 * albedo + 0.6417;
    vec3 c =  2.7552 * albedo + 0.6903;
    return max(vec3(ao), ((ao * a + b) * ao + c) * ao);
}

// Bilinear-gather cascade tap — same as world_lmap.frag's cascTap.
float cascTap(sampler2D smap, vec2 uv, float ref)
{
    vec2 sz = vec2(textureSize(smap, 0));
    vec2 t  = uv * sz - 0.5;
    vec2 f  = fract(t);
    vec4 d  = textureGather(smap, (floor(t) + 1.0) / sz, 0);
    vec4 c  = step(vec4(ref), d);
    return mix(mix(c.w, c.z, f.x), mix(c.x, c.y, f.x), f.y);
}

// One cascade lookup; -1.0 when outside (1% UV inset). Single tap — foliage.
float cascSample1(sampler2D smap, mat4 vp, vec3 wp, float bias_)
{
    vec3 n = (vp * vec4(wp, 1.0)).xyz;          // ortho → already NDC (w == 1)
    vec2 uv = n.xy * 0.5 + 0.5;
    uv.y = 1.0 - uv.y;
    if (uv.x < 0.01 || uv.x > 0.99 || uv.y < 0.01 || uv.y > 0.99
        || n.z <= 0.0 || n.z >= 1.0)
        return -1.0;
    return cascTap(smap, uv, n.z - bias_);
}

// Hemisphere sky ambient — same source as the world ground (world_lmap). Tree
// crowns sample straight up (no per-leaf normal), like the grass.
vec3 skyAmbientUp()
{
    float lod = L.sky_params.z;
    const vec3 up = vec3(0.0, 1.0, 0.0);
    return mix(textureLod(uSky0, up, lod).rgb, textureLod(uSky1, up, lod).rgb,
               clamp(L.sky_params.x, 0.0, 1.0));
}

// Foliage variant: leaves have no per-pixel normal here → attenuation-only.
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

layout(push_constant) uniform PC {
    mat4  mViewProj;
    float uvScale;
    float alphaRef;
} pc;

layout(location = 0) in vec2 vUV;
layout(location = 1) in vec3 vLight;
layout(location = 2) in vec3 vSunLit;
layout(location = 3) in vec3 vWPos;

layout(location = 0) out vec4 outColor;

// 1-tap sun shadow (same projection convention as world_lmap.frag).
// Cascades first (the far map's 15.6 cm texels + its big depth bias swallow the
// crown-on-trunk self-shadow — the crown sits only metres above the trunk, well
// inside the far bias; cascade 0's 0.6 cm texels resolve it like R4 does).
float sunShadow1(vec3 worldPos)
{
    float s = cascSample1(uShadowNear, L.sun_near_vp, worldPos, 0.0004);
    if (s >= 0.0) return s;
    s = cascSample1(uShadowC1, L.sun_c1_vp, worldPos, 0.0006);
    if (s >= 0.0) return s;

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
    if (diff.a < pc.alphaRef)
        discard;

    // r_ssao_debug 1: trees show the raw AO map too — see world_lmap.frag.
    if (L.ao_params.w > 0.5) {
        outColor = vec4(vec3(textureLod(uAO, gl_FragCoord.xy * L.ao_params.xy, 0.0).r), 1.0);
        return;
    }

    vec3 sunPart = vSunLit;
    if (dot(sunPart, sunPart) > 0.0)
        sunPart *= sunShadow1(vWPos);

    // Ambient = sky-cube light × per-tree openness (vLight.x). ×0.75 keeps the
    // old grass:tree ambient ratio (grass 2.0 vs tree 1.5 in the flat model).
    // GTAO gates it (softened — see gtaoVis); sun/dyn lights untouched.
    vec3 ambient = (skyAmbientUp() * (vLight.x * L.sky_params.y * 0.75) + 0.05)
                 * coloredAO(gtaoVis(), diff.rgb);

    vec3 col = diff.rgb * (ambient + sunPart + dynLightsFoliage(vWPos));

    // Distance fog (R4) — see world_lmap.frag.
    float fog = clamp(length(vWPos - L.eye_pos.xyz) * L.fog_params.w + L.fog_params.x, 0.0, 1.0);
    col = mix(col, L.fog_color.rgb, fog);

    outColor = vec4(col, 1.0);
}
