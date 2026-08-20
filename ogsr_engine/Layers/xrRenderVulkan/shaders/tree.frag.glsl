#version 450
#extension GL_GOOGLE_include_directive : require
#define VSM_SET 2                  // EnvLight set is bound at set 2 for trees
#define ENV_SET 2                  // EnvLight UBO + samplers live at set 2
#include "vsm_sample.glsl"         // vsmSunShadow (set 2 b14)
#include "light_ubo.glsl"          // DynLight + Lighting UBO (set 2 b0) + samplers (set 2 b1..13)
#include "foliage_shadow.glsl"     // foliageLightShadow — foliage receives spot/point pool shadows
#include "surface_class.glsl"      // SC_* classification + snow (foliage)
#include "shore_wet.glsl"
#include "ao_common.glsl"       // coloredAO (shared with env_common.glsl)          // shoreWetness — trunks and snags lying in water
#include "common_math.glsl"     // EnvBRDFApprox — pure, shared with world/grass
#include "ao_sampled.glsl"      // gtaoVisK / ssilBoost — after light_ubo.glsl (uAO/uIL/L)
#include "shadow_math.glsl"     // cascTap — set-agnostic bilinear PCF tap

// xrRenderVulkan - tree forward fragment shader. set 1/binding 0 = per-group
// diffuse; trees are alpha-tested (punch-out leaves). set 2 = the shared per-frame
// EnvLight, now via light_ubo.glsl (was a hand-rolled partial copy). The sun part
// of the vertex lighting (vSunLit) gets a single shadow tap; foliage noise hides
// the hard edge.
layout(set = 1, binding = 0) uniform sampler2D uDiffuse;

// GTAO - trees ARE in the prepass depth, so crowns/trunks get their own AO. HALF
// the strength exponent (alpha-tested leaf depth is noisy at half-res) — that is
// the gtaoVisK(0.5) at the shading site below. gtaoVisK/ssilBoost live in
// ao_sampled.glsl, cascTap in shadow_math.glsl (both included above).

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
// (vLight.x) and boosted when wet. Karis EnvBRDFApprox → common_math.glsl.
// ⚠wetF is passed IN, not looked up here. This function used to call shoreWet(wp)
// itself while main() called it again eight lines later — two independent lookups
// of the same answer at the same position, and shoreWet is up to five texture taps
// (one level-water + four hand-filtered tile taps). On a fill-bound pass that is
// the cheapest kind of waste there is: pure duplication, no picture attached.
vec3 canopySkySheen(vec3 wp, float openness, float wetF)
{
    if (L.ibl_params.x < 0.004) return vec3(0.0);
    const vec3 up = vec3(0.0, 1.0, 0.0);
    vec3  V   = normalize(L.eye_pos.xyz - wp);
    float NoV = clamp(dot(up, V), 0.0, 1.0);
    vec3  R   = reflect(-V, up);
    float rough = mix(0.7, 0.5, wetF);
    vec3  pref  = textureLod(uSkySpec, R, rough * L.ibl_params.z).rgb;
    // Subtle when dry, glistens when wet; × canopy openness so inner/shaded leaves stay matte.
    return pref * EnvBRDFApprox(vec3(0.04), rough, NoV)
         * (L.ibl_params.x * L.ibl_params.y * openness * (0.12 + 0.88 * wetF));
}

// dynLightsFoliage → foliage_shadow.glsl (shared with the other foliage pass).

// Prefix of VK::TreeGfxPush / tree_gfx_push.glsl — offsets must match that block.
layout(push_constant) uniform PC {
    mat4  mViewProj;
    float uvScale;
    float alphaRef;
    float statsOn;    // 1 = mark the visible-tree bitset (r_profiler diagnostics)
    float shadeDist;  // r_tree_shade_dist — metres past which the subtle terms are dropped (0 = never)
} pc;

// DISTANCE TIER for the subtle per-pixel terms — see r_tree_shade_dist in
// vk_console_min.cpp for why the FRAGMENT shader is the lever on this pass.
// Returns 1 near, fading to 0 over the last quarter of the band. The fade matters:
// a hard radius draws a ring through the forest that slides with the camera,
// whereas a fade only ever removes a term that was already almost gone.
float shadeTier(float dist)
{
    if (pc.shadeDist <= 0.0) return 1.0;                       // r_tree_shade_dist 0 = tier disabled
    return 1.0 - smoothstep(pc.shadeDist * 0.75, pc.shadeDist, dist);
}

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

    // ---- Distance tier ----
    // camDist is needed by the fog below anyway, so the tier costs one compare.
    // `near` is a UNIFORM branch across a distant crown (every fragment of one tree
    // is at roughly one distance), so the skipped work is genuinely skipped rather
    // than executed under a mask by half the warp.
    float camDist = distance(vWPos, L.eye_pos.xyz);
    float tier    = shadeTier(camDist);
    bool  near_   = tier > 0.0031;   // below ~1/320 the term cannot move an 8-bit channel

    // Soaked is DARK — see detail.frag. Wet bark is markedly darker than dry, and
    // for a trunk lying in the water that boundary IS the effect. ONE lookup now,
    // shared with the sheen below; the debug view forces it at any range so
    // r_wtr_wet still shows the map rather than the tier.
    float sw = (near_ || shoreWetDebug()) ? shoreWet(vWPos) : 0.0;
    if (shoreWetDebug()) { outColor = vec4(sw, sw * 0.3, 0.0, 1.0); return; }   // red = wet
    sw *= tier;

    // Ambient = sky-cube light x per-tree openness (vLight.x). x0.75 keeps the old
    // grass:tree ambient ratio. L.ambient = real env ambient (carries r_ambient_floor).
    // SSIL is a screen-space BOUNCE, i.e. a multiplier that trends to 1 — so the
    // tier fades it toward 1, not toward 0.
    vec3 ssil    = near_ ? mix(vec3(1.0), ssilBoost(), tier) : vec3(1.0);
    vec3 ambient = (skyAmbientUp() * (vLight.x * L.sky_params.y * 0.75) + L.ambient.rgb)
                 * coloredAO(gtaoVisK(0.5), diff.rgb) * ssil;

    // Rain OR standing water — a fallen trunk lying in a river is soaked where it lies.
    float wetF  = max(clamp(L.rain_params.y, 0.0, 1.0), sw);
    vec3  sheen = near_ ? canopySkySheen(vWPos, vLight.x, wetF) * tier : vec3(0.0);   // sky specular sheen (r_ibl)
    vec3  dynl  = near_ ? dynLightsFoliage(vWPos) * tier : vec3(0.0);                 // point/spot walk + 1-tap shadows
    vec3  col   = diff.rgb * (ambient + sunPart + dynl) + sheen;
    col *= 1.0 - 0.65 * sw;

    // Distance fog (R4).
    float fog = clamp(camDist * L.fog_params.w + L.fog_params.x, 0.0, 1.0);
    col = mix(col, L.fog_color.rgb, fog);

    if (L.ibl_params.w > 0.5) { outColor = vec4(sheen, 1.0); return; }   // r_ibl_debug
    vec4 ptdbg = foliagePointDebug(vWPos);
    if (ptdbg.a > 0.5) { outColor = vec4(ptdbg.rgb, 1.0); return; }
    outColor = vec4(col, 1.0);
}
