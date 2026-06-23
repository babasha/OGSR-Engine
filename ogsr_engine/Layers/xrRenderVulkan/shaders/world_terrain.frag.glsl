#version 450
#extension GL_GOOGLE_include_directive : require
#include "light_ubo.glsl"       // DynLight + Lighting UBO (set 1 b0) + set-1 samplers (b1..13)
#include "wet_common.glsl"       // vHash/vNoise/puddlesMaskProc/rippleLayer/rainRipples
#include "vsm_sample.glsl"       // vsmSunShadow (set 1 b14)
#include "cluster_lights.glsl"   // clustered forward (set 1 b17..19, r_clustered)
#include "shadow_common.glsl"    // spotShadowF/pointShadowF/cascTap/cascSample/sunShadow
#include "env_common.glsl"       // gtaoVis/gtaoBentN/coloredAO/skyAmbient/rainVis
#include "light_shade.glsl"      // lightTerrainOcc/shadeDynLight/dynLights
#include "flow_sim_sample.glsl"  // simWater*/simFlow/groundHm/waterDebugColor/flowWaves
#include "wetness.glsl"          // applyWetnessTerrain + applyWetnessCore
#include "surface_field.glsl"    // SF_* (smart heightmap; r_sf_debug viz)
#include "surface_class.glsl"    // SC_* surface classification (r_sf_debug 5)
#include "snow_displace.glsl"    // SnowFootprint (fragment footprint dimple)

// World pass - TERRAIN splatting fragment shader (R4 deffer_terrain_high /
// CBlender_BmmD): albedo = 2*base*detail, detail blended by the RGBA splat mask.
// Detail channels: R grass, G asphalt, B earth, A gravel. Adds per-channel detail
// normal maps, micro contact-AO, dry sun-gloss, SSS puddles. Shared lighting/
// shadow/wet helpers live in the includes above; terrain-specific POM, detail
// normals, puddle placement and main() stay here.

layout(set = 0, binding = 0)  uniform sampler2D uBase;   // terrain diffuse
layout(set = 0, binding = 1)  uniform sampler2D uMask;   // RGBA splat weights
layout(set = 0, binding = 2)  uniform sampler2D uDtR;    // grass   detail
layout(set = 0, binding = 3)  uniform sampler2D uDtG;    // asphalt detail
layout(set = 0, binding = 4)  uniform sampler2D uDtB;    // earth   detail
layout(set = 0, binding = 5)  uniform sampler2D uDtA;    // gravel/yantar detail
layout(set = 0, binding = 6)  uniform sampler2D uLmap;
layout(set = 0, binding = 7)  uniform sampler2D uDnR;    // grass   normal (R4: n = tex.wzy*2-1, gloss = tex.r)
layout(set = 0, binding = 8)  uniform sampler2D uDnG;    // asphalt normal
layout(set = 0, binding = 9)  uniform sampler2D uDnB;    // earth   normal
layout(set = 0, binding = 10) uniform sampler2D uDnA;    // gravel  normal

layout(push_constant) uniform PushConstants {
    mat4  mvp;
    vec2  uvScale;
    float alphaRef;
    float detailScale;
} pc;

layout(location = 0) in  vec2 vUV;
layout(location = 1) in  vec2 vDetailUV;
layout(location = 2) in  vec2 vLmapUV;
layout(location = 3) in  vec3 vWorldPos;
layout(location = 4) in  vec3 vNormal;
layout(location = 0) out vec4 outColor;

// ---- Terrain POM. No `#` height map, so relief comes from the HIGH-PASSED
// luminance of the BASE albedo (macro cracks / pebbles / twigs). High-pass
// removes the large tone patches so they don't become false cliffs. Same march +
// normal + self-shadow + contact AO as the wall POM. Off by default (swims at
// grazing) - detail-normal mapping is the primary terrain relief source.
float baseLuma(vec2 uv, float lod) { return dot(textureLod(uBase, uv, lod).rgb, vec3(0.299, 0.587, 0.114)); }

float pomDepth(vec2 uv, float lod, float baseline)
{
    return clamp(0.5 - (baseLuma(uv, lod) - baseline) * 4.0, 0.0, 1.0);
}

vec2 parallaxUV(vec2 uv, vec3 N, vec3 wp, out vec3 outN, out float outShadow, out float outAO)
{
    outN = N; outShadow = 1.0; outAO = 1.0;
    if (L.pom_params4.x < 0.5) return uv;   // terrain POM disabled (r_pom_terrain 0) -> flat ground
    float amp = L.pom_params.x;
    if (amp <= 0.0) return uv;
    float dist = length(L.eye_pos.xyz - wp);
    float fade = 1.0 - smoothstep(L.pom_params.z * 0.5, L.pom_params.z, dist);
    amp *= fade;
    float orient = (N.y >= 0.0)
        ? mix(1.0, L.pom_params3.w, clamp( N.y, 0.0, 1.0))
        : mix(1.0, L.pom_params3.z, clamp(-N.y, 0.0, 1.0));
    amp *= orient;
    if (amp <= 1e-5) return uv;

    vec2  tsz = vec2(textureSize(uBase, 0));
    vec2  ddx = dFdx(uv) * tsz, ddy = dFdy(uv) * tsz;
    float lod = max(0.5 * log2(max(dot(ddx, ddx), dot(ddy, ddy))), 0.0) + L.pom_params2.x;
    float baseline = baseLuma(uv, lod + 3.0);

    vec3 dp1 = dFdx(wp), dp2 = dFdy(wp);
    vec2 du1 = dFdx(uv), du2 = dFdy(uv);
    vec3 dp2p = cross(dp2, N), dp1p = cross(N, dp1);
    vec3 T = dp2p * du1.x + dp1p * du2.x;
    vec3 B = dp2p * du1.y + dp1p * du2.y;
    float inv = inversesqrt(max(dot(T, T), dot(B, B)));
    T *= inv; B *= inv;

    vec3 V   = normalize(L.eye_pos.xyz - wp);
    vec3 Vts = vec3(dot(V, T), dot(V, B), dot(V, N));
    // Terrain is viewed at GRAZING angles almost always (it's the floor). Cap the
    // offset (floor 0.55) and FADE it out at grazing - kills the liquid look.
    vec2 Pmax = (Vts.xy / max(abs(Vts.z), 0.55)) * amp * smoothstep(0.12, 0.45, abs(Vts.z));

    int steps = int(clamp(mix(L.pom_params.y, 12.0, abs(Vts.z)) * mix(0.5, 1.0, fade), 8.0, 64.0));
    float layerH = 1.0 / float(steps);
    vec2 dUV = Pmax * layerH;

    float curD = 0.0;
    vec2  curUV = uv;
    float curH = pomDepth(curUV, lod, baseline);
    for (int i = 0; i < 64; ++i) {
        if (i >= steps || curD >= curH) break;
        curUV -= dUV; curD += layerH;
        curH = pomDepth(curUV, lod, baseline);
    }
    vec2 sUV = dUV; float sD = layerH;
    for (int j = 0; j < 6; ++j) {
        sUV *= 0.5; sD *= 0.5;
        if (curD < pomDepth(curUV, lod, baseline)) { curUV -= sUV; curD += sD; }
        else                                       { curUV += sUV; curD -= sD; }
    }

    // Normal from base-luma gradient. Gentler gain (x3 vs x12 on walls) + tilt
    // clamp - the base-luma gradient is high-contrast and would chrome-mirror.
    float tU = exp2(lod) / tsz.x, tV = exp2(lod) / tsz.y;
    float hu = baseLuma(curUV + vec2(tU, 0.0), lod) - baseLuma(curUV - vec2(tU, 0.0), lod);
    float hv = baseLuma(curUV + vec2(0.0, tV), lod) - baseLuma(curUV - vec2(0.0, tV), lod);
    float ns = L.pom_params2.y * 3.0 * fade * orient;
    vec3 nTS = normalize(vec3(clamp(-hu * ns, -0.6, 0.6), clamp(-hv * ns, -0.6, 0.6), 1.0));
    outN = normalize(T * nTS.x + B * nTS.y + N * nTS.z);

    if (L.pom_params2.z > 0.0) {
        vec3 Ld  = normalize(-L.sun_dir.xyz);
        vec3 Lts = vec3(dot(Ld, T), dot(Ld, B), dot(Ld, N));
        vec2 lxy = Lts.xy;
        if (Lts.z > 0.02 && dot(lxy, lxy) > 1e-6) {
            vec2  sdir  = normalize(lxy);
            float reach = (exp2(lod) / min(tsz.x, tsz.y)) * 6.0;
            float h0    = baseLuma(curUV, lod);
            float occ   = 0.0;
            for (int s = 1; s <= 8; ++s)
                occ = max(occ, baseLuma(curUV + sdir * reach * (float(s) * 0.125), lod) - h0);
            outShadow = clamp(1.0 - occ * L.pom_params2.z * 20.0 * (1.0 - Lts.z) * orient, 0.0, 1.0);
        }
    }
    if (L.pom_params2.w > 0.0) {
        float aoReach = (exp2(lod) / min(tsz.x, tsz.y)) * 4.0;
        float h0 = baseLuma(curUV, lod);
        float aoSum =
              max(0.0, baseLuma(curUV + vec2( aoReach, 0.0), lod) - h0)
            + max(0.0, baseLuma(curUV + vec2(-aoReach, 0.0), lod) - h0)
            + max(0.0, baseLuma(curUV + vec2(0.0,  aoReach), lod) - h0)
            + max(0.0, baseLuma(curUV + vec2(0.0, -aoReach), lod) - h0);
        outAO = clamp(1.0 - (aoSum * 0.25) * L.pom_params2.w * 6.0, 0.35, 1.0);
        outAO = mix(1.0, outAO, orient);
    }
    return curUV;
}

// ---- Detail NORMAL MAPPING (R4 CBlender_BmmD s_dn_*). The primary ground-relief
// mechanism on terrain. Blends the 4 per-channel tangent normals by the splat
// mask, then rotates into world space via a screen-space cotangent frame (terrain
// has no per-vertex tangents). Returns geomN unchanged when off.
vec3 detailNormal(vec3 geomN, vec2 duv, vec4 mask, float strength, out float outCav, out float outGloss)
{
    outCav = 0.0; outGloss = 0.0;
    if (strength <= 0.0) return geomN;
    // One sample per channel; decode normal (R4: n = tex.wzy*2-1) AND gloss
    // (R4: gloss = tex.r), both blended by the splat mask.
    vec4 sR = texture(uDnR, duv), sG = texture(uDnG, duv);
    vec4 sB = texture(uDnB, duv), sA = texture(uDnA, duv);
    vec3 n = (sR.wzy * 2.0 - 1.0) * mask.r
           + (sG.wzy * 2.0 - 1.0) * mask.g
           + (sB.wzy * 2.0 - 1.0) * mask.b
           + (sA.wzy * 2.0 - 1.0) * mask.a;
    outGloss = sR.x * mask.r + sG.x * mask.g + sB.x * mask.b + sA.x * mask.a;
    n.xy *= strength;
    n = normalize(n);
    // Cavity: how far the micro-normal tilts off the surface (drives micro AO).
    outCav = clamp(1.0 - n.z, 0.0, 1.0);

    vec3 dp1 = dFdx(vWorldPos), dp2 = dFdy(vWorldPos);
    vec2 du1 = dFdx(duv),       du2 = dFdy(duv);
    vec3 dp2p = cross(dp2, geomN), dp1p = cross(geomN, dp1);
    vec3 T = dp2p * du1.x + dp1p * du2.x;
    vec3 B = dp2p * du1.y + dp1p * du2.y;
    float inv = inversesqrt(max(dot(T, T), dot(B, B)));
    T *= inv; B *= inv;
    return normalize(T * n.x + B * n.y + geomN * n.z);
}

// SSS-faithful per-pixel puddle placement (SSFX deffer_terrain_high_flat recipe):
// coverage rises with the wetness accumulator (grow/recede), gated to near-flat
// up-facing ground. NO top-down height map for placement (SSFX/R4 don't use one).
float sssPuddle(vec3 N, vec3 wp)
{
    float wet = clamp(L.rain_params.y, 0.0, 1.0);
    if (wet <= 0.0) return 0.0;
    float slope = clamp((1.0 - max(abs(N.x), abs(N.z)) - 0.9) * 13.0, 0.0, 1.0);
    if (slope <= 0.0) return 0.0;
    float cov = clamp(wet * L.pom_params7.y * 1.5, 0.0, 1.0);   // x1.5 = distinct, not fields
    return puddlesMaskProc(wp.xz, cov, L.pom_params7.w) * slope;
}

void main()
{
    vec3 pomN; float pomShadow, pomAO;
    vec2 pUV = parallaxUV(vUV, normalize(vNormal), vWorldPos, pomN, pomShadow, pomAO);
    vec2 pDelta = pUV - vUV;
    vec2 pDetailUV = vDetailUV + pDelta * pc.detailScale;

    // PUDDLE REFRACTION (SSFX N_refra): where a puddle covers this pixel, bend the
    // bottom (base+detail) UV by the water-surface ripple. Gate tight (near + raining).
    if (L.pom_params7.x > 0.5 && L.rain_params.y > 0.01 && L.rain_params.z >= 0.0) {
        vec3  gN = normalize(vNormal);
        float sl = clamp((1.0 - max(abs(gN.x), abs(gN.z)) - 0.9) * 13.0, 0.0, 1.0);
        float pe = (sl > 0.0)
                 ? puddlesMaskProc(vWorldPos.xz, clamp(L.rain_params.y, 0.0, 1.0) * L.pom_params7.y, L.pom_params7.w) * sl
                 : 0.0;
        float rf = smoothstep(18.0, 8.0, distance(L.eye_pos.xyz, vWorldPos)) * smoothstep(0.2, 0.6, pe);
        if (rf > 0.01) {
            vec2 ro = rainRipples(vWorldPos.xz, L.sky_params.w * 0.7) * (0.006 * rf);
            pUV       += ro;
            pDetailUV += ro * pc.detailScale;
        }
    }

    vec4 base = texture(uBase, pUV);
    if (pc.alphaRef >= 0.0 && base.a < pc.alphaRef) discard;

    // r_ssao_debug 1: show the raw AO map.
    if (L.ao_params.w > 0.5) {
        outColor = vec4(vec3(textureLod(uAO, gl_FragCoord.xy * L.ao_params.xy, 0.0).r), base.a);
        return;
    }
    // r_wet_debug 1: rain-map visibility.
    if (L.rain_params.z < 0.0) {
        outColor = vec4(vec3(rainVis(vWorldPos)), base.a);
        return;
    }
    // r_pom_debug 1: POM occlusion mask (contact AO x self-shadow).
    if (L.pom_params3.x > 0.5) {
        outColor = vec4(vec3(pomAO * pomShadow), base.a);
        return;
    }
    // r_clustered_debug: per-cluster dynamic-light-count heatmap.
    if (L.cluster_params.w > 1.5) {
        int cnt = clusterLightCount(vWorldPos, L.eye_pos.xyz, L.cam_dir.xyz, gl_FragCoord.xy, L.ao_params.xy, L.cluster_params, L.cluster_params2);
        outColor = vec4(clusterHeat(cnt), base.a);
        return;
    }

    // r_sf_debug: Surface Field viz (1 height, 2 slope, 3 curvature, 4 sky, 5 canopy).
    int sfdbg = int(L.sf_params.y + 0.5);
    if (sfdbg > 0) {
        vec3 nn = normalize(vNormal);
        outColor = vec4((sfdbg == 5) ? SC_DebugColor(SC_Refine(SC_GROUND, nn))   // terrain = ground
                                      : SF_DebugColor(vWorldPos, nn, sfdbg), base.a);
        return;
    }

    // Splat mask (at the parallax-offset UV) - normalize so the 4 weights sum to
    // 1. Empty/missing mask -> even blend; all-zero -> pure grass (never black).
    vec4 mask = texture(uMask, pUV);
    float wsum = dot(mask, vec4(1.0));
    mask = (wsum > 1e-4) ? (mask / wsum) : vec4(1.0, 0.0, 0.0, 0.0);

    vec4 dR = texture(uDtR, pDetailUV);
    vec4 dG = texture(uDtG, pDetailUV);
    vec4 dB = texture(uDtB, pDetailUV);
    vec4 dA = texture(uDtA, pDetailUV);
    vec3 detail = dR.rgb * mask.r + dG.rgb * mask.g + dB.rgb * mask.b + dA.rgb * mask.a;
    // Detail height (R4 terrain AO source = detail diffuse alpha, splat-blended).
    float detH = dR.a * mask.r + dG.a * mask.g + dB.a * mask.b + dA.a * mask.a;

    vec3 albedo = 2.0 * base.rgb * detail;

    // Lightmap (hemi/AO) + DYNAMIC R4-style sun (per-pixel N.L x shadow map).
    vec4  lm      = texture(uLmap, vLmapUV);
    float hemiOcc = dot(lm.rgb, vec3(1.0 / 3.0));
    vec3  geomN   = normalize(vNormal);   // flat - for the sky fill (cube is sharp -> perturbed = mirror)
    vec3  Nw      = pomN;                  // POM-perturbed - sun + dyn lights catch the relief
    // Detail normal mapping: the primary ground-relief source. Faded to 0 by
    // distance so far terrain skips the 4 taps + AO + gloss entirely.
    float dnStr  = L.pom_params4.y * smoothstep(45.0, 25.0, distance(L.eye_pos.xyz, vWorldPos));
    float cav    = 0.0;
    float glossT = 0.0;
    if (dnStr > 0.0)
        Nw = detailNormal(geomN, pDetailUV, mask, dnStr, cav, glossT);

    // Micro contact AO: darken grooves. Occlusion = max of the normal cavity
    // (1-n.z) and the detail height pit (1-h^2). Applied to albedo (R4 base*=1-AO).
    float aoStr   = L.pom_params4.z;
    float occlT   = max(cav, 1.0 - detH * detH);
    float microAO = 1.0 - aoStr * clamp(occlT, 0.0, 1.0);
    albedo *= microAO;

    // PUDDLE COVERAGE (0..1). SSS procedural placement default; flow sim is parked.
    float microH = min(detH, 1.0 - cav);
    float pud = (L.pom_params7.x > 0.5) ? sssPuddle(geomN, vWorldPos)
              : (L.pom_params6.x > 0.5) ? smoothstep(0.04, 0.12, simWaterSoft(vWorldPos))
              : 0.0;

    // r_puddle_debug: 1 = coverage; 2 = per-pixel detail micro-height.
    int pdbg = int(L.pom_params5.w + 0.5);
    if (pdbg > 0) {
        outColor = (L.pom_params6.x > 0.5) ? vec4(waterDebugColor(vWorldPos, pdbg), base.a)
                                           : vec4(vec3(pdbg >= 2 ? microH : pud), base.a);
        return;
    }

    // r_terrain_debug: 1 world normal, 2 micro-AO, 3 detail height.
    int tdbg = int(L.pom_params4.w + 0.5);
    if (tdbg == 1) { outColor = vec4(Nw * 0.5 + 0.5, base.a); return; }
    if (tdbg == 2) { outColor = vec4(vec3(microAO),  base.a); return; }
    if (tdbg == 3) { outColor = vec4(vec3(detH),     base.a); return; }

    // SNOW (Surface Field consumer, r_snow): whiten by surface type x slope x sky
    // exposure - flat up-facing ground accumulates, steep/under-cover none.
    float snowBase = SC_SnowAmount(SC_Refine(SC_GROUND, geomN), geomN, SF_SkyExposure(vWorldPos));
    // Organic growth: vary the coverage ONSET per world-location (multi-scale noise) so
    // partial snow fills in irregular PATCHES, not per-terrain-triangle. At full coverage
    // (sf_params.w -> 1) every threshold is met -> uniform snow.
    float covThr = (vNoise(vWorldPos.xz * 0.13) * 0.55 + vNoise(vWorldPos.xz * 0.43) * 0.30
                  + vNoise(vWorldPos.xz * 1.30) * 0.15) * 0.75;
    float snow = snowBase * smoothstep(covThr, covThr + 0.22, clamp(L.sf_params.w, 0.0, 1.0));
    albedo = mix(albedo, vec3(0.90, 0.93, 0.97), snow);
    Nw = normalize(mix(Nw, geomN, snow));   // snow smooths the microrelief -> less "texturey"
    if (L.deform_tex.x < 1.5)               // MESH mode: the snow mesh draws the dents on top
        SnowFootprint(albedo, Nw, vWorldPos, snow);   // pressed-snow prints at recent foot contacts

    float sunMask = max(dot(Nw, normalize(-L.sun_dir.xyz)), 0.0);
    if (sunMask > 0.005) {
        float sunSh = (L.shadow_params.w > 0.5) ? vsmSunShadow(gl_FragCoord.xy * L.ao_params.xy)
                                                : sunShadow(vWorldPos);
        sunMask *= sunSh * pomShadow;
    }
    // Hemisphere sky fill (R4 hmodel). sun_color/ambient arrive final from vk_env_light.
    vec3  occ      = coloredAO(gtaoVis(), albedo) * pomAO * ssilBoost();   // GTAO x POM AO x SSIL bounce (ambient only)
    float hemiOccL = hemiOcc;
    if (L.pom_params3.y > 0.5) { occ = vec3(1.0); hemiOccL = 1.0; }   // r_ao_flat debug
    vec3 lighting = skyAmbient(gtaoBentN(geomN)) * (hemiOccL * L.sky_params.y) * occ
                  + L.sun_color.rgb  * sunMask
                  + L.ambient.rgb * occ
                  + dynLights(vWorldPos, Nw);

    // Dry sun gloss (R4 gloss in bump .r): material-aware Blinn specular -
    // asphalt/gravel glint, grass matte. Fades out as the ground wets.
    vec3  drySpec  = vec3(0.0);
    float glossStr = L.pom_params5.x;
    if (glossStr > 0.0 && glossT > 0.0 && sunMask > 0.005) {
        vec3  Ld   = normalize(-L.sun_dir.xyz);
        vec3  V    = normalize(L.eye_pos.xyz - vWorldPos);
        vec3  H    = normalize(Ld + V);
        float g    = clamp(glossT, 0.0, 1.0);
        float shin = mix(16.0, 160.0, g);             // glossier -> tighter highlight
        float s    = pow(max(dot(Nw, H), 0.0), shin) * g;
        float dry  = (1.0 - clamp(L.rain_params.y, 0.0, 1.0)) * (1.0 - snow);   // snow also kills the dry gloss
        drySpec    = L.sun_color.rgb * (s * glossStr * sunMask * dry);
    }

    // Rain wetness (terrain: pud passed in; asphalt reflectivity term was unused).
    vec3 wetRefl = applyWetnessTerrain(albedo, vWorldPos, Nw, pud, sunMask, gtaoBentN(geomN), gtaoVisRaw());

    // Distance fog (R4).
    vec3 col = albedo * lighting + drySpec + wetRefl;
    float fog = clamp(length(vWorldPos - L.eye_pos.xyz) * L.fog_params.w + L.fog_params.x, 0.0, 1.0);
    col = mix(col, L.fog_color.rgb, fog);

    outColor = vec4(col, base.a);
}
