#version 450
#extension GL_GOOGLE_include_directive : require
#define VSM_SET 2                  // EnvLight set is bound at set 2 for trees
#define ENV_SET 2                  // EnvLight UBO + samplers live at set 2
#include "vsm_sample.glsl"         // vsmSunShadow (set 2 b14)
#include "light_ubo.glsl"          // DynLight + Lighting UBO (set 2 b0) + samplers (set 2 b1..13)
#include "foliage_shadow.glsl"     // foliageLightShadow — foliage receives spot/point pool shadows
#include "surface_class.glsl"      // SC_* classification + snow (foliage)

// xrRenderVulkan - tree forward fragment shader. set 1/binding 0 = per-group
// diffuse; trees are alpha-tested (punch-out leaves). set 2 = the shared per-frame
// EnvLight, now via light_ubo.glsl (was a hand-rolled partial copy). The sun part
// of the vertex lighting (vSunLit) gets a single shadow tap; foliage noise hides
// the hard edge.
layout(set = 1, binding = 0) uniform sampler2D uDiffuse;

// GTAO - trees ARE in the prepass depth, so crowns/trunks get their own AO. HALF
// the strength exponent (alpha-tested leaf depth is noisy at half-res).
float gtaoVis()
{
    float ao = textureLod(uAO, gl_FragCoord.xy * L.ao_params.xy, 0.0).r;
    return pow(clamp(ao, 0.0, 1.0), L.ao_params.z * 0.5);
}

// Colored AO - see world_lmap.frag (R4 tints foliage occlusion by its albedo).
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

// Bilinear-gather cascade tap - same as world_lmap's cascTap.
float cascTap(sampler2D smap, vec2 uv, float ref)
{
    vec2 sz = vec2(textureSize(smap, 0));
    vec2 t  = uv * sz - 0.5;
    vec2 f  = fract(t);
    vec4 d  = textureGather(smap, (floor(t) + 1.0) / sz, 0);
    vec4 c  = step(vec4(ref), d);
    return mix(mix(c.w, c.z, f.x), mix(c.x, c.y, f.x), f.y);
}

// One cascade lookup; -1.0 when outside (1% UV inset). Single tap - foliage.
float cascSample1(sampler2D smap, mat4 vp, vec3 wp, float bias_)
{
    vec3 n = (vp * vec4(wp, 1.0)).xyz;          // ortho -> already NDC (w == 1)
    vec2 uv = n.xy * 0.5 + 0.5;
    uv.y = 1.0 - uv.y;
    if (uv.x < 0.01 || uv.x > 0.99 || uv.y < 0.01 || uv.y > 0.99
        || n.z <= 0.0 || n.z >= 1.0)
        return -1.0;
    return cascTap(smap, uv, n.z - bias_);
}

// Sky ambient (crowns sample straight up — no per-leaf normal) now comes from the
// shared header, so the canopy gets the SH9 irradiance the world does instead of a
// private point-sampled copy.
#include "sky_ambient.glsl"

// Sky specular sheen for the canopy (r_ibl). Leaves have no normal, so a real
// reflection is meaningless — instead give the crown a soft view-dependent sky
// sheen (up-reflected, high roughness = blurry), gated by per-tree sky openness
// (vLight.x) and boosted when wet. Karis EnvBRDFApprox (local copy of env_common).
vec3 EnvBRDFApprox(vec3 F0, float roughness, float NoV)
{
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572,  0.022);
    const vec4 c1 = vec4( 1.0,  0.0425,  1.040, -0.040);
    vec4  r    = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NoV)) * r.x + r.y;
    vec2  ab   = vec2(-1.04, 1.04) * a004 + r.zw;
    return F0 * ab.x + ab.y;
}
vec3 canopySkySheen(vec3 wp, float openness)
{
    if (L.ibl_params.x < 0.004) return vec3(0.0);
    const vec3 up = vec3(0.0, 1.0, 0.0);
    vec3  V   = normalize(L.eye_pos.xyz - wp);
    float NoV = clamp(dot(up, V), 0.0, 1.0);
    vec3  R   = reflect(-V, up);
    float wetF  = clamp(L.rain_params.y, 0.0, 1.0);
    float rough = mix(0.7, 0.5, wetF);
    vec3  pref  = textureLod(uSkySpec, R, rough * L.ibl_params.z).rgb;
    // Subtle when dry, glistens when wet; × canopy openness so inner/shaded leaves stay matte.
    return pref * EnvBRDFApprox(vec3(0.04), rough, NoV)
         * (L.ibl_params.x * L.ibl_params.y * openness * (0.12 + 0.88 * wetF));
}

// Foliage variant: leaves have no per-pixel normal -> attenuation-only.
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
        // Narrow beams: windowed falloff (far half still lights) — light_shade.glsl.
        float att;
        if (L.lights[i].color.w > 0.5 && L.lights[i].dir.w > 0.87) {
            att = 1.0 - (d2 / (r * r));
            att *= att;
        } else {
            att = 1.0 - d / r;
            att *= att;
        }
        if (L.lights[i].color.w > 0.5) {
            // Narrow beams: full inside the cone + spill to 2x the angle — see
            // light_shade.glsl (axis-peaked ramp left beam-lit foliage dark).
            float ca = dot(-dv / d, L.lights[i].dir.xyz);
            float ci = L.lights[i].dir.w;
            if (ci > 0.87) {
                float co = 2.0 * ci * ci - 1.0;
                att *= clamp((ca - co) / max(ci - co, 1e-3), 0.0, 1.0);
            } else
                att *= clamp((ca - ci) / max(1.0 - ci, 1e-3), 0.0, 1.0);
        }
        // Dynamic shadow (spot tile / point cube) — see detail.frag.
        att *= foliageLightShadow(i, wp);
        acc += L.lights[i].color.rgb * (att * 0.7);
    }
    return acc;
}

layout(push_constant) uniform PC {
    mat4  mViewProj;
    float uvScale;
    float alphaRef;
    float statsOn;    // 1 = mark the visible-tree bitset (r_profiler diagnostics)
} pc;

layout(location = 0) in vec2 vUV;
layout(location = 1) in vec3 vLight;
layout(location = 2) in vec3 vSunLit;
layout(location = 3) in vec3 vWPos;
layout(location = 4) flat in uint vTreeIdx;
layout(location = 0) out vec4 outColor;

// Visible-tree bitset (set 0 shares the tree gfx layout with the VS transforms).
// A fragment that survives the alpha test marks its tree — with early-Z on the
// prepass depth that's ≈ "this tree has visible pixels". Guard-read first so the
// atomic fires ~once per tree, not per fragment.
layout(set = 0, binding = 2, std430) buffer SeenBits { uint seenBits[]; };

// 1-tap sun shadow (cascades first, then the far map) - see world_lmap.frag.
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
    vec4 diff = texture(uDiffuse, vUV, L.spot_flash.z);   // DLSS mip bias (0 native)
    // Alpha stays UNBIASED: the sharper (biased) alpha mip shrinks the cutout
    // coverage at the alphaRef threshold → black holes onto the dark crown
    // interior at distance («чёрные пятна на листве» under DLSS upscaling).
    if (L.spot_flash.z != 0.0) diff.a = texture(uDiffuse, vUV).a;
    if (diff.a < pc.alphaRef)
        discard;

    if (pc.statsOn > 0.5) {
        uint w = vTreeIdx >> 5u, m = 1u << (vTreeIdx & 31u);
        if ((seenBits[w] & m) == 0u) atomicOr(seenBits[w], m);
    }

    // r_ssao_debug 1: trees show the raw AO map too.
    if (L.ao_params.w > 0.5) {
        outColor = vec4(vec3(textureLod(uAO, gl_FragCoord.xy * L.ao_params.xy, 0.0).r), 1.0);
        return;
    }
    // r_sf_debug 5: surface classification - foliage (yellow).
    if (int(L.sf_params.y + 0.5) == 5) {
        outColor = vec4(SC_DebugColor(SC_Refine(SC_FOLIAGE, vec3(0.0, 1.0, 0.0))), 1.0);
        return;
    }

    // SNOW frosting on the crown (Surface Field consumer, r_snow). Foliage has no
    // per-leaf normal -> treat as up; the canopy is sky-exposed.
    float snow = SC_SnowAmount(SC_Refine(SC_FOLIAGE, vec3(0.0, 1.0, 0.0)), vec3(0.0, 1.0, 0.0), 1.0) * clamp(L.sf_params.w, 0.0, 1.0);
    diff.rgb = mix(diff.rgb, vec3(0.90, 0.93, 0.97), snow);

    vec3 sunPart = vSunLit;
    if (dot(sunPart, sunPart) > 0.0) {
        // VSM screen-space mask (r_vsm) vs the 1-tap cascade - gated by shadow_params.w.
        float sunSh = (L.shadow_params.w > 0.5) ? vsmSunShadow(gl_FragCoord.xy * L.ao_params.xy)
                                                 : sunShadow1(vWPos);
        sunPart *= sunSh;
    }

    // Ambient = sky-cube light x per-tree openness (vLight.x). x0.75 keeps the old
    // grass:tree ambient ratio. L.ambient = real env ambient (carries r_ambient_floor).
    vec3 ambient = (skyAmbientUp() * (vLight.x * L.sky_params.y * 0.75) + L.ambient.rgb)
                 * coloredAO(gtaoVis(), diff.rgb) * ssilBoost();   // + SSIL bounce (ambient only)

    vec3 sheen = canopySkySheen(vWPos, vLight.x);   // sky specular sheen (r_ibl)
    vec3 col = diff.rgb * (ambient + sunPart + dynLightsFoliage(vWPos)) + sheen;

    // Distance fog (R4).
    float fog = clamp(length(vWPos - L.eye_pos.xyz) * L.fog_params.w + L.fog_params.x, 0.0, 1.0);
    col = mix(col, L.fog_color.rgb, fog);

    if (L.ibl_params.w > 0.5) { outColor = vec4(sheen, 1.0); return; }   // r_ibl_debug
    vec4 ptdbg = foliagePointDebug(vWPos);
    if (ptdbg.a > 0.5) { outColor = vec4(ptdbg.rgb, 1.0); return; }
    outColor = vec4(col, 1.0);
}
