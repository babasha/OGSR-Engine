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
#include "wetness.glsl"          // applyWetness (lmap/vlit) + applyWetnessCore
#include "surface_field.glsl"    // SF_* (smart heightmap; r_sf_debug viz)
#include "surface_class.glsl"    // SC_* surface classification (r_sf_debug 5)

// World pass - vert-lit variant. Final colour = albedo x (baked vertex lighting
// + dynamic R4-style sun + env hemi sky fill gated by bake occlusion + ambient).
// vBakedColor = offline point-light + bounce (NO direct sun). Shared helpers live
// in the includes above; only the vlit-specific POM and main() stay here.

layout(set = 0, binding = 0) uniform sampler2D uTexDiffuse;
layout(set = 0, binding = 1) uniform sampler2D uTexDetail;
layout(set = 0, binding = 2) uniform sampler2D uTexLmap;   // unused (white fallback)
layout(set = 0, binding = 3) uniform sampler2D uTexBumpX;  // .a = height (POM); flat=1 -> no parallax

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
layout(location = 3) in  float vSunMask;     // (legacy per-vertex sun mask; sun is now dynamic)
layout(location = 4) in  vec3  vWorldPos;
layout(location = 5) in  vec3  vNormal;
layout(location = 0) out vec4  outColor;

// POM heightfield from the `#` alpha, high-passed - see world_lmap.frag.
float pomDepth(vec2 uv, float lod, float baseline)
{
    float h = textureLod(uTexBumpX, uv, lod).a;
    return clamp(0.5 - (h - baseline) * 4.0, 0.0, 1.0);
}

// Parallax occlusion (relief) mapping - identical to world_lmap.frag.
vec2 parallaxUV(vec2 uv, vec3 N, vec3 wp, out vec3 outN, out float outShadow, out float outAO)
{
    outN = N;
    outShadow = 1.0;
    outAO = 1.0;
    float amp = L.pom_params.x;
    if (amp <= 0.0) return uv;
    if (textureLod(uTexBumpX, vec2(0.5), 8.0).a > 0.985) return uv;   // flat material -> no POM
    float dist = length(L.eye_pos.xyz - wp);
    float fade = 1.0 - smoothstep(L.pom_params.z * 0.5, L.pom_params.z, dist);
    amp *= fade;
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

    // r_pom_height = WORLD metres → this material's UV units via the UV→world
    // Jacobian; stretched cliff/rock UVs no longer explode into metre-deep
    // "spike" shreds (see world_lmap.frag for the full note).
    float uvArea = abs(du1.x * du2.y - du1.y * du2.x);
    float wPerUV = sqrt(length(cross(dp1, dp2)) / max(uvArea, 1e-12));
    amp = min(amp / max(wPerUV, 1e-3), 0.035);

    vec3 V   = normalize(L.eye_pos.xyz - wp);
    vec3 Vts = vec3(dot(V, T), dot(V, B), dot(V, N));
    vec2 Pmax = (Vts.xy / max(abs(Vts.z), 0.3)) * amp;

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
    float tU = exp2(lod) / tsz.x, tV = exp2(lod) / tsz.y;
    float hu = textureLod(uTexBumpX, curUV + vec2(tU, 0.0), lod).a
             - textureLod(uTexBumpX, curUV - vec2(tU, 0.0), lod).a;
    float hv = textureLod(uTexBumpX, curUV + vec2(0.0, tV), lod).a
             - textureLod(uTexBumpX, curUV - vec2(0.0, tV), lod).a;
    float ns = L.pom_params2.y * 12.0 * fade * orient;
    vec3 nTS = normalize(vec3(-hu * ns, -hv * ns, 1.0));
    outN = normalize(T * nTS.x + B * nTS.y + N * nTS.z);

    if (L.pom_params2.z > 0.0) {
        vec3 Ld  = normalize(-L.sun_dir.xyz);
        vec3 Lts = vec3(dot(Ld, T), dot(Ld, B), dot(Ld, N));
        vec2 lxy = Lts.xy;
        if (Lts.z > 0.02 && dot(lxy, lxy) > 1e-6) {
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
    if (L.pom_params2.w > 0.0) {
        float aoReach = (exp2(lod) / min(tsz.x, tsz.y)) * 4.0;
        float h0 = textureLod(uTexBumpX, curUV, lod).a;
        float aoSum =
              max(0.0, textureLod(uTexBumpX, curUV + vec2( aoReach, 0.0), lod).a - h0)
            + max(0.0, textureLod(uTexBumpX, curUV + vec2(-aoReach, 0.0), lod).a - h0)
            + max(0.0, textureLod(uTexBumpX, curUV + vec2(0.0,  aoReach), lod).a - h0)
            + max(0.0, textureLod(uTexBumpX, curUV + vec2(0.0, -aoReach), lod).a - h0);
        outAO = clamp(1.0 - (aoSum * 0.25) * L.pom_params2.w * 6.0, 0.35, 1.0);
        outAO = mix(1.0, outAO, orient);
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

    // EMISSIVE-ADDITIVE (aref == -3 ONLY: glow halos / selflight) — unlit
    // additive + angle fade, see world_lmap.frag. Lightplanes = aref -4, below.
    if (pc.alphaRef < -2.5 && pc.alphaRef > -3.5) {
        vec3  Vv   = normalize(L.eye_pos.xyz - vWorldPos);
        float face = abs(dot(normalize(vNormal), Vv));
        float w    = face * face;
        outColor = vec4(base.rgb * 2.0 * w, base.a * w);
        return;
    }

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
    // r_puddle_debug: sim on -> depth/flow; else the SSS puddle coverage (grayscale).
    int pdbg = int(L.pom_params5.w + 0.5);
    if (pdbg > 0) {
        outColor = (L.pom_params6.x > 0.5)
            ? vec4(waterDebugColor(vWorldPos, pdbg), base.a)
            : vec4(vec3(puddlesMaskProc(vWorldPos.xz, clamp(L.rain_params.y * L.pom_params7.y * 1.5, 0.0, 1.0), L.pom_params7.w)), base.a);
        return;
    }
    // r_pom_debug 1: POM occlusion mask (contact AO x sun self-shadow).
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
        outColor = vec4((sfdbg == 5) ? SC_DebugColor(SC_Refine(SC_STATIC, nn))   // vlit = prop static
                                      : SF_DebugColor(vWorldPos, nn, sfdbg), base.a);
        return;
    }

    vec3 detail = texture(uTexDetail, pDetailUV).rgb;
    vec3 albedo = 2.0 * base.rgb * detail;

    // Lighting: baked vertex color (point lights + bounce) + dynamic R4-style sun
    // (per-pixel N.L x shadow map) + env hemi sky fill + ambient floor.
    vec3  geomN   = normalize(vNormal);   // flat - for the sky fill (sharp cube -> perturbed = mirror)
    vec3  Nw      = pomN;   // POM-perturbed normal -> sun + dyn lights catch the relief
    // SNOW (Surface Field consumer, r_snow): whiten by surface type x slope x sky
    // exposure - flat up-facing props accumulate, vertical/under-cover none.
    float snow = SC_SnowAmount(SC_Refine(SC_STATIC, geomN), geomN, SF_SkyExposure(vWorldPos)) * clamp(L.sf_params.w, 0.0, 1.0);
    albedo = mix(albedo, vec3(0.90, 0.93, 0.97), snow);
    Nw = normalize(mix(Nw, geomN, snow));   // snow smooths the microrelief -> less "texturey"

    float sunMask = max(dot(Nw, normalize(-L.sun_dir.xyz)), 0.0);
    if (sunMask > 0.005) {
        float sunSh = (L.shadow_params.w > 0.5) ? vsmSunShadow(gl_FragCoord.xy * L.ao_params.xy)
                                                : sunShadow(vWorldPos);
        sunMask *= sunSh * pomShadow;
    }

    // vlit has NO lightmap occlusion, so gate the sky fill by BOTH pc.dynHemi
    // (ray-traced sky visibility for dynamics) AND the baked vertex brightness
    // (occlusion for static vlit geometry). Small floor keeps open outdoor vlit lit.
    float bakeOcc = clamp(dot(vBakedColor, vec3(0.299, 0.587, 0.114)) * 2.5, 0.15, 1.0);
    vec3  occ      = coloredAO(gtaoVis(), albedo) * pomAO * ssilBoost();   // GTAO x POM AO x SSIL bounce (ambient only)
    // dynHemi < -0.5 = dynamic visual (sign = "model xform follows" flag for the VS);
    // real ray-traced sky visibility is -dynHemi-1.
    float dynHemiL = (pc.dynHemi < -0.5) ? (-pc.dynHemi - 1.0) : pc.dynHemi;
    if (L.pom_params3.y > 0.5) { occ = vec3(1.0); dynHemiL = 1.0; bakeOcc = 1.0; }   // r_ao_flat debug
    vec3 lighting = vBakedColor * 1.5
                  + L.sun_color.rgb  * sunMask
                  + skyAmbient(gtaoBentN(geomN)) * (L.sky_params.y * 0.5 * dynHemiL * bakeOcc) * occ
                  + L.ambient.rgb * occ
                  + dynLights(vWorldPos, Nw);

    // Rain wetness - see world_lmap.frag.
    vec3 wetRefl = applyWetness(albedo, vWorldPos, Nw, sunMask, gtaoBentN(geomN), gtaoVisRaw());

    // Distance fog (R4).
    vec3 col = albedo * lighting + wetRefl;
    float fog = clamp(length(vWorldPos - L.eye_pos.xyz) * L.fog_params.w + L.fog_params.x, 0.0, 1.0);
    col = mix(col, L.fog_color.rgb, fog);

    // GLASS (alphaRef == -2, late blended pass) — R4 model_env_lq formula:
    // lerp(env reflection, texture, texture.a), blend alpha = texture alpha × fog².
    // See world_lmap.frag for the full note.
    float outA = base.a;
    if (pc.alphaRef < -3.5) {
        // LIGHTPLANES light beams (R4 model_def_lq): light·base·2 with the simple
        // model light, NOT the full `lighting` — see world_lmap.frag for why.
        vec3 lq = L.ambient.rgb + L.hemi_color.rgb * max(geomN.y, 0.0)
                + L.sun_color.rgb * max(dot(geomN, normalize(-L.sun_dir.xyz)), 0.0);
        col  = mix(base.rgb * 2.0 * lq, L.fog_color.rgb, fog);
        outA = base.a * (1.0 - fog) * (1.0 - fog);
    }
    else if (pc.alphaRef < -1.5) {
        // GLASS: R4 base + fresnel + sun glint — see world_lmap.frag for the note.
        float aG   = min(base.a, L.pom_params5.y);
        vec3  V    = normalize(vWorldPos - L.eye_pos.xyz);
        vec3  Rv   = reflect(V, geomN);
        float NoV  = clamp(dot(geomN, -V), 0.0, 1.0);
        float fres = 0.04 + 0.96 * pow(1.0 - NoV, 5.0);
        float xf   = clamp(L.sky_params.x, 0.0, 1.0);
        vec3  env  = textureLod(uSky0, Rv, 1.0).rgb;
        if (xf > 0.01) env = mix(env, textureLod(uSky1, Rv, 1.0).rgb, xf);
        vec3 glint   = L.sun_color.rgb * (pow(max(dot(Rv, normalize(-L.sun_dir.xyz)), 0.0), 128.0) * 2.0 * sunMask);
        // Reflection modulated by the LOCAL lighting; fresnel coverage weighted by
        // reflection brightness (no dark "tint" film indoors) — see world_lmap.frag.
        vec3 reflCol = env * lighting * (0.5 + fres) + glint;
        float fCov = fres * clamp(dot(reflCol, vec3(0.6)), 0.0, 1.0);
        float aOut = clamp(aG + fCov * (1.0 - aG), 1e-4, 1.0);
        vec3  gcol = (albedo * lighting * aG + reflCol * fCov * (1.0 - aG)) / aOut;
        col  = mix(gcol, L.fog_color.rgb, fog);
        outA = aOut * (1.0 - fog) * (1.0 - fog);
    }
    outColor = vec4(col, outA);
}
