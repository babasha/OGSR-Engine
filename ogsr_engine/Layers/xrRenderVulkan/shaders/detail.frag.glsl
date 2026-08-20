#version 450
#extension GL_GOOGLE_include_directive : require
#define VSM_GRASS_DIRECT           // grass samples the VSM static atlas at its OWN 3D pos (b15/16/23)
#include "vsm_sample.glsl"         // vsmSunShadow (set 1 b14) + vsmSunShadowGrassDirect
#include "light_ubo.glsl"          // DynLight + Lighting UBO (set 1 b0) + samplers (set 1 b1..13)
#include "foliage_shadow.glsl"     // foliageLightShadow — grass receives spot/point pool shadows
#include "surface_class.glsl"      // SC_* classification + snow (grass)
#include "shore_wet.glsl"
#include "ao_common.glsl"       // coloredAO (shared with env_common.glsl)          // shoreWetness — reeds standing in water
#include "common_math.glsl"     // EnvBRDFApprox — pure, shared with world/tree
#include "ao_sampled.glsl"      // gtaoVis / ssilBoost — after light_ubo.glsl (uAO/uIL/L)

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
// gtaoVis / ssilBoost come from ao_sampled.glsl (they read light_ubo's uAO/uIL/L).

// Sky ambient (grass samples straight up — no real normal, same light as the
// terrain beneath it) now comes from the shared header, so grass gets the SH9
// irradiance the world does instead of a private point-sampled copy.
#include "sky_ambient.glsl"

// Sky specular sheen for grass (r_ibl). Blades have no normal → up-reflected blurry
// sky, gated by baked openness + WET (dry grass ≈ matte; the payoff is wet grass
// glistening). Karis EnvBRDFApprox → common_math.glsl. `openness` = vColor.r baked hemi.
vec3 grassSkySheen(vec3 wp, float openness)
{
    if (L.ibl_params.x < 0.004) return vec3(0.0);
    // Rain OR standing water. Reeds grow out of a river and were bone dry to the
    // waterline, because every wet surface in this renderer was wet from the sky.
    float wetF = max(clamp(L.rain_params.y, 0.0, 1.0) * rainVisGrass(wp), shoreWet(wp));
    if (wetF < 0.02 && openness < 0.01) return vec3(0.0);
    const vec3 up = vec3(0.0, 1.0, 0.0);
    vec3  V   = normalize(L.eye_pos.xyz - wp);
    float NoV = clamp(dot(up, V), 0.0, 1.0);
    vec3  R   = reflect(-V, up);
    float rough = mix(0.75, 0.55, wetF);
    vec3  pref  = textureLod(uSkySpec, R, rough * L.ibl_params.z).rgb;
    // Very subtle dry, glistens wet — grass is near-ground and matte.
    return pref * EnvBRDFApprox(vec3(0.04), rough, NoV)
         * (L.ibl_params.x * L.ibl_params.y * openness * (0.05 + 0.95 * wetF));
}

// dynLightsFoliage → foliage_shadow.glsl (shared with the other foliage pass).

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

// r_grass_debug 1..11 (L.spot_params.z): paint the GRASS ONLY with one isolated
// lighting component, greyscale — the rest of the scene renders normally, so a
// pattern "seen through the bush" can be attributed to its actual source. This
// tool closed the 2026-07-04 "shadows through the grass" saga (root cause: the
// screen-space VSM mask belongs to the surface BEHIND the blade) — keep it.
//  1 = sun shadow factor the blade APPLIES (VSM direct atlas @ vWPos / cascade tap)
//  2 = VSM mask R  (FULL shadow, resolved at the surface BEHIND the blade)
//  3 = VSM mask B  (STATIC-only channel — the abandoned mask-based grass path)
//  4 = mask parallax: R = how far behind the blade the mask's surface sits, B = blade height
//  5 = GTAO of the ground behind (ambient gate)
//  6 = SSIL bounce of the ground behind (ambient boost)
//  7 = BAKED per-slot sun occlusion (vSunLit: level "lightmap" on grass)
//  8 = BAKED per-slot hemi occlusion (vColor.r: level "lightmap", ambient side)
//  9 = full ambient term (sky*hemi + floor, x AO x SSIL)
// 10 = dynamic lights on grass (incl. their pool SHADOWS)
// 11 = shadow SOURCE split: white=lit, grey=static shadow (trees/buildings),
//      RED=dyn-atlas shadow (grass-on-grass / NPC) — reacts to r_grass_self_bias live
vec3 grassDebugColor(int gdbg, vec3 albedo)
{
    vec2  suv = gl_FragCoord.xy * L.ao_params.xy;
    vec4  m   = texture(uVsmMask, suv);
    float par = smoothstep(0.15, 0.6, m.g - length(vWPos - L.eye_pos.xyz));
    if (gdbg == 1) return vec3((L.shadow_params.w > 0.5) ? vsmSunShadowGrassDirect(vWPos, L.spot_params.w) : sunShadow1(vWPos));
    if (gdbg == 2) return vec3(m.r);
    if (gdbg == 3) return vec3(m.b);
    if (gdbg == 4) return vec3(par, 0.0, clamp(vHeight, 0.0, 1.0));
    if (gdbg == 5) return vec3(gtaoVis());
    if (gdbg == 6) return ssilBoost() - vec3(1.0);
    if (gdbg == 7) return vSunLit / max(max(pc.vSunColor.r, max(pc.vSunColor.g, pc.vSunColor.b)), 1e-4);
    if (gdbg == 8) return vec3(vColor.r);
    if (gdbg == 9) return (skyAmbientUp() * (vColor.r * L.sky_params.y) + L.ambient.rgb)
                          * coloredAO(gtaoVis(), albedo) * ssilBoost();
    if (gdbg == 10) return dynLightsFoliage(vWPos);
    if (gdbg == 11) {
        float dynHit = 0.0;
        float lit = (L.shadow_params.w > 0.5) ? vsmSunShadowGrassDirectDbg(vWPos, L.spot_params.w, dynHit)
                                              : sunShadow1(vWPos);
        return mix(vec3(mix(0.25, 1.0, lit)), vec3(1.0, 0.0, 0.0), dynHit);
    }
    return vec3(1.0, 0.0, 1.0);   // unknown mode
}

void main()
{
    vec4 diff = texture(uDiffuse, vUV, L.spot_flash.z);   // DLSS mip bias (0 native)
    // Alpha UNBIASED — the biased alpha mip shrinks blade coverage at the cutoff
    // (same black-holes physics as the tree crowns under DLSS upscaling).
    if (L.spot_flash.z != 0.0) diff.a = texture(uDiffuse, vUV).a;
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

    // r_grass_debug: lighting-component isolation views (see grassDebugColor).
    int gdbg = int(L.spot_params.z + 0.5);
    if (gdbg > 0) {
        outColor = vec4(grassDebugColor(gdbg, diff.rgb), 1.0);
        return;
    }

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
        // VSM (r_vsm) vs the 1-tap cascade - gated by shadow_params.w. Grass does NOT
        // read the screen-space mask (it belongs to the surface BEHIND the blade —
        // terrain shadows 20 m away painted themselves onto the bush): it samples
        // BOTH VSM atlases directly at the blade's own 3D position (static = trees/
        // buildings; dyn = grass/NPC with the r_grass_self_bias anti-acne slack).
        // Zero parallax by construction — same principle as the cascade path.
        float sunSh = (L.shadow_params.w > 0.5)
            ? vsmSunShadowGrassDirect(vWPos, L.spot_params.w)
            : sunShadow1(vWPos);
        sunPart *= sunSh;
    }

    // Ambient = sky-cube light x baked per-slot hemi occlusion (vColor.r). L.ambient
    // = real env ambient (carries r_ambient_floor). GTAO gates it (sun/dyn untouched).
    // GTAO/SSIL are SCREEN-SPACE, resolved on the opaque surface BEHIND the blade
    // (grass skips the prepass). Right next to that surface this is the correct
    // ambient (dark-corner case the gtaoVis comment describes); when the background
    // sits METERS behind the blade, its occlusion band (fence-base corner AO, a
    // building's shadowed wall) painted itself onto the lit foreground tuft — the
    // "shadow seen through the grass". Same parallax principle as the direct VSM
    // sun sampling above: fade both toward neutral by how far behind the blade the
    // mask's recorded surface sits (mask G = resolved surface view distance; VSM
    // sessions only — without the mask the distance source doesn't exist).
    float aoPar = 0.0;
    if (L.shadow_params.w > 0.5) {
        float mg = texture(uVsmMask, gl_FragCoord.xy * L.ao_params.xy).g;
        aoPar = smoothstep(0.15, 0.6, mg - length(vWPos - L.eye_pos.xyz));
    }
    vec3 ambient = (skyAmbientUp() * (vColor.r * L.sky_params.y) + L.ambient.rgb)
                 * coloredAO(mix(gtaoVis(), 1.0, aoPar), diff.rgb)
                 * mix(ssilBoost(), vec3(1.0), aoPar);   // + SSIL bounce (ambient only, same parallax fade)

    vec3 col = diff.rgb * (ambient + sunPart + dynLightsFoliage(vWPos));

    // Wet grass: blades open to the rain darken slightly (+ a touch cooler).
    float wet = clamp(L.rain_params.y, 0.0, 1.0) * max(L.rain_params.z, 0.0);
    if (wet > 0.005) {
        float wv = wet * rainVisGrass(vWPos);
        col *= 1.0 - 0.5 * wv;
        col = mix(col, col * vec3(0.85, 0.92, 1.0), 0.5 * wv);
    }

    // Sky specular sheen (r_ibl) — additive, after the wet darken (reflected sky, not absorbed).
    vec3 sheen = grassSkySheen(vWPos, vColor.r);
    col += sheen;

    // SOAKED IS DARK. The sheen alone does not read as wet — it is a grazing-angle
    // highlight, and a reed standing in a river is read by TONE first. Applied
    // after the lighting so it darkens the lit result, which is what water in the
    // blade does: it fills the surface pores and stops them scattering back.
    float sw = shoreWet(vWPos);
    if (shoreWetDebug()) { outColor = vec4(sw, sw * 0.3, 0.0, 1.0); return; }   // red = wet
    // Wet vegetation is markedly darker than dry — more so than soil, because a
    // film of water on a blade kills the diffuse scatter that makes it look pale.
    col *= 1.0 - 0.65 * sw;
    // ...and then the foam the surge left ON it. Grass at a waterline is where the
    // scum line actually catches, so leaving this to the terrain alone would put the
    // foam under the very blades it hangs off. Lightening the LIT colour rather than
    // an albedo, because that is where this shader's wetness already works.
    float sf = shoreFoam(vWPos);
    if (sf > 0.003) col = mix(col, vec3(0.80, 0.81, 0.78) * (0.35 + 0.65 * vColor.r), sf * 0.8);

    // Distance fog (R4).
    float fog = clamp(length(vWPos - L.eye_pos.xyz) * L.fog_params.w + L.fog_params.x, 0.0, 1.0);
    col = mix(col, L.fog_color.rgb, fog);

    if (L.ibl_params.w > 0.5) { outColor = vec4(sheen, 1.0); return; }   // r_ibl_debug
    vec4 ptdbg = foliagePointDebug(vWPos);
    if (ptdbg.a > 0.5) { outColor = vec4(ptdbg.rgb, 1.0); return; }
    outColor = vec4(col, 1.0);
}
