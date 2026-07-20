#version 450
#extension GL_GOOGLE_include_directive : require
#include "light_ubo.glsl"       // DynLight + Lighting UBO (set 1 b0) + set-1 samplers (b1..13)
#include "world_variants.glsl"   // SPEC_POM/SNOW/WET/IBL/DEBUG variant spec constants (Inc 1)
#include "wet_common.glsl"       // vHash/vNoise/puddlesMaskProc/rippleLayer/rainRipples
#define VSM_GRASS_DIRECT         // also pull in vsmSunShadowGrassDirect (crisp atlas at own world pos) for beam-gap recovery
#include "vsm_sample.glsl"       // vsmSunShadow (set 1 b14) + vsmSunShadowGrassDirect (b15/16/23/24/25)
#include "cluster_lights.glsl"   // clustered forward (set 1 b17..19, r_clustered)
#include "shadow_common.glsl"    // spotShadowF/pointShadowF/cascTap/cascSample/sunShadow
#include "env_common.glsl"       // gtaoVis/gtaoBentN/coloredAO/skyAmbient/rainVis
#include "light_shade.glsl"      // lightTerrainOcc/shadeDynLight/dynLights
#include "flow_sim_sample.glsl"  // simWater*/simFlow/groundHm/waterDebugColor/flowWaves
#include "wetness.glsl"          // applyWetnessTerrain + applyWetnessCore
#include "surface_field.glsl"    // SF_* (smart heightmap; r_sf_debug viz)
#include "surface_class.glsl"    // SC_* surface classification (r_sf_debug 5)
#include "snow_displace.glsl"    // SnowFootprint (fragment footprint dimple)
#include "froxel.glsl"           // Froxel_SliceFromViewZ — probe the shaft in-scatter for the sun-beam ground deposit

// Integrated froxel volume (rgb = in-scatter, a = transmittance) on the shared EnvLight
// set. The sun-beam GROUND DEPOSIT (r_sun_beam_ground) probes it at this pixel to find
// where a visible shaft lands and deposits sun on the terrain there. Bound to a valid
// (eager) 3D view every frame; L.beam2.z = 0 disables the read.
layout(set = ENV_SET, binding = 27) uniform sampler3D uVolume;

// [PARKED 2026-07-06 — r_sun_beam_ground default 0, inert. Kept as scaffolding; see the
//  PARKED note in vk_console_min.cpp. Do not delete.]
// SUN-BEAM ground deposit signal — VIEW-INDEPENDENT. The volumetric in-scatter is
// view-dependent (HG phase + view-ray integration) → deposits flickered with the camera.
// Instead march a few short steps from the ground point UP toward the SUN and sample the
// crisp sun SHADOW (VSM atlas). The ground itself is occluded (r_terrain_debug 7), but a
// point a metre or two up toward the sun is INSIDE the beam column where a crown gap lets
// light through → lit. Under a thin gap the sun is found close overhead (strong pool);
// under a thick crown it stays shadowed for metres (dark) → this is exactly dappled sun
// through the canopy, and it does NOT depend on where the camera looks. Weight nearer
// hits more (the beam reaches closer to the ground). Cost-gated by distance (heavy atlas
// taps) and VSM-only (needs the crisp atlas the fog uses).
float shaftOverhead(vec3 wp)
{
    if (L.shadow_params.w <= 0.5) return 0.0;                     // VSM only
    if (distance(L.eye_pos.xyz, wp) > L.beam_params.y) return 0.0; // reuse r_sun_beam_dist as the cost gate
    vec3  toSun = normalize(-L.sun_dir.xyz);
    float best  = 0.0;
    for (int i = 1; i <= 4; ++i) {
        float h = float(i) * 0.9;                                 // 0.9 .. 3.6 m up toward the sun
        float v = vsmSunShadowGrassDirect(wp + toSun * h, L.beam_params.w);   // 1 lit .. 0 shadowed
        best = max(best, v * (1.0 - h * 0.22));                   // nearer sun-hit = stronger beam reaching down
    }
    return best;
}

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
layout(set = 0, binding = 11) uniform sampler2D uDhR;    // grass   height (SSFX-style, .r = elevation)
layout(set = 0, binding = 12) uniform sampler2D uDhG;    // asphalt height
layout(set = 0, binding = 13) uniform sampler2D uDhB;    // earth   height
layout(set = 0, binding = 14) uniform sampler2D uDhA;    // gravel  height

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

// ---- Terrain POM (SSFX-style REAL heightfield parallax) - the marches, mud
// bias and the SSFX depth offset live in terrain_pom.glsl, SHARED with the
// depth-prepass FS (world_terrain_depth.frag) so both passes write the exact
// same displaced gl_FragDepth. Compiled with -DZOFF -> the variant that exports
// the displaced depth (world_terrain_zoff.frag.spv).
#include "terrain_pom.glsl"

#ifdef ZOFF
layout(depth_greater) out float gl_FragDepth;   // sink-only offset keeps early-Z
#endif

// ---- Detail NORMAL MAPPING (R4 CBlender_BmmD s_dn_*). The primary ground-relief
// mechanism on terrain. Blends the 4 per-channel tangent normals by the splat
// mask, then rotates into world space via a screen-space cotangent frame (terrain
// has no per-vertex tangents). Returns geomN unchanged when off.
vec3 detailNormal(vec3 geomN, vec2 uvs[4], vec4 mask, float strength, out float outCav, out float outGloss)
{
    outCav = 0.0; outGloss = 0.0;
    if (strength <= 0.0) return geomN;
    // One sample per channel AT THAT CHANNEL'S OWN parallaxed uv; decode normal
    // (R4: n = tex.wzy*2-1) AND gloss (R4: gloss = tex.r), blended by the mask.
    vec4 sR = texture(uDnR, uvs[0], L.spot_flash.z), sG = texture(uDnG, uvs[1], L.spot_flash.z);   // DLSS mip bias (0 native)
    vec4 sB = texture(uDnB, uvs[2], L.spot_flash.z), sA = texture(uDnA, uvs[3], L.spot_flash.z);
    vec3 n = (sR.wzy * 2.0 - 1.0) * mask.r
           + (sG.wzy * 2.0 - 1.0) * mask.g
           + (sB.wzy * 2.0 - 1.0) * mask.b
           + (sA.wzy * 2.0 - 1.0) * mask.a;
    outGloss = sR.x * mask.r + sG.x * mask.g + sB.x * mask.b + sA.x * mask.a;
    n.xy *= strength;
    n = normalize(n);
    // Cavity: how far the micro-normal tilts off the surface (drives micro AO).
    outCav = clamp(1.0 - n.z, 0.0, 1.0);

    // Frame from the UNparallaxed detail uv (stable derivatives; the frame is the
    // uv axes' world orientation - unaffected by the small per-channel offsets).
    vec3 dp1 = dFdx(vWorldPos), dp2 = dFdy(vWorldPos);
    vec2 du1 = dFdx(vDetailUV), du2 = dFdy(vDetailUV);
    vec3 dp2p = cross(dp2, geomN), dp1p = cross(geomN, dp1);
    vec3 T = dp2p * du1.x + dp1p * du2.x;
    vec3 B = dp2p * du1.y + dp1p * du2.y;
    float inv = inversesqrt(max(dot(T, T), dot(B, B)));
    T *= inv; B *= inv;
    return normalize(T * n.x + B * n.y + geomN * n.z);
}

// SSFX-faithful puddle placement (deffer_terrain_high_flat_d.ps "Puddles slope
// mask and height"): the water RISES UP THE POM HEIGHTFIELD — `puddle = water
// level - texel height` — so rain floods the cracks and potholes FIRST and only
// spills over the raised plates in a downpour. detH = the mask-blended parallax
// hit height (incl. the per-channel CH_OFF sinks: asphalt/mud sit low -> trap
// water, exactly like SSFX ssfx_terrain_offset). The procedural macro mask
// keeps large-scale variation (SSFX modulates by s_puddles_mask the same way)
// so the whole yard doesn't flood uniformly.
float sssPuddle(vec3 N, vec3 wp, float detH)
{
    float wet = clamp(L.rain_params.y, 0.0, 1.0);
    if (wet <= 0.0) return 0.0;
    float slope = clamp((1.0 - max(abs(N.x), abs(N.z)) - 0.9) * 13.0, 0.0, 1.0);
    if (slope <= 0.0) return 0.0;
    // SSFX: height_str = saturate(wetness * PLANE * limit - PixelHeight); x4 ramp.
    // r_puddle_level 0.5 default maps to the SSFX water limit 1.0.
    float level   = wet * POM_PLANE * clamp(L.pom_params7.y * 2.0, 0.0, 2.0);
    float puddles = clamp((level - detH) * 4.0, 0.0, 1.0);
    if (puddles <= 0.0) return 0.0;
    // Macro body variation: soft (0.35 floor), not the old binary placement —
    // the heightfield decides WHERE inside a body the water actually sits.
    float macro = 0.35 + 0.65 * puddlesMaskProc(wp.xz, clamp(wet * 1.4, 0.0, 1.0), L.pom_params7.w);
    return puddles * macro * slope;
}

void main()
{
    vec3 geomN = normalize(vNormal);
    // Splat mask (macro weights). Sample BEFORE parallax - the offset is tiny in
    // macro UV. Normalize so the 4 weights sum to 1; empty/missing -> even blend,
    // all-zero -> pure grass (never black).
    vec4 mask = texture(uMask, terrainMaskUV(vUV, vWorldPos));
    float wsum = dot(mask, vec4(1.0));
    mask = (wsum > 1e-4) ? (mask / wsum) : vec4(1.0, 0.0, 0.0, 0.0);

    // ---- MUD footprints (r_mud_deform): read the persistent press field (vk_deform,
    // stamped once per foot PLANT — boot-shaped, treaded) on SOFT splat channels and
    // sink it into the POM heightfield below. Soil softness varies per material
    // (earth deep, gravel shallow, asphalt none) and rain SOFTENS it (deeper prints
    // on wet ground). Wet asphalt still SHOWS prints (muddy boots track grime) but
    // never carves. Snow owns the prints once it covers the ground.
    float mudPress, mudCarve, mudGrime;      // signed press / print carve / asphalt grime
    float mudWater = 0.0;   // water gathered in the dent (fed into pud below)
    float mudWet   = clamp(L.rain_params.y, 0.0, 1.0);
    terrainMudSetup(mask, vWorldPos, mudPress, mudCarve, mudGrime);   // sets g_mudBias (shared w/ depth prepass)

    // Real-heightfield terrain POM in DETAIL space (the volumetric ground). Base/
    // mask stay at vUV (macro is low-freq -> not parallaxed -> stable at grazing).
    // 4 INDEPENDENT per-channel marches (SSFX) -> each material keeps full relief.
    float pomShadow, pomAO, detH;
    vec2 pUVs[4];
    vec4 pomHits;
    terrainPOM4(vDetailUV, mask, geomN, vWorldPos, pUVs, pomHits, pomShadow, pomAO, detH, true);
#ifdef ZOFF
    // SSFX depth offset (r_pom_zoff): sink the fragment depth into the POM cracks.
    // MUST equal what world_terrain_depth.frag wrote in the prepass (same inputs,
    // same math) - the LEQUAL depth test keeps the pixel only on equality.
    gl_FragDepth = max(gl_FragCoord.z, terrainZoffDepth(pc.mvp, vWorldPos, geomN, detH, gl_FragCoord.z));
#endif
    vec2 pUV = vUV;

    // PUDDLE REFRACTION (SSFX N_refra): where a puddle covers this pixel, bend the
    // bottom (base+detail) UV by the water-surface ripple. Gate tight (near + raining).
    if (L.pom_params7.x > 0.5 && L.rain_params.y > 0.01 && L.rain_params.z >= 0.0) {
        float pe = sssPuddle(normalize(vNormal), vWorldPos, detH);   // heightfield-filled water
        float rf = smoothstep(18.0, 8.0, distance(L.eye_pos.xyz, vWorldPos)) * smoothstep(0.2, 0.6, pe);
        if (rf > 0.01) {
            vec2 ro = rainRipples(vWorldPos.xz, L.sky_params.w * 0.7) * (0.006 * rf);
            pUV += ro;
            vec2 rod = ro * pc.detailScale;
            pUVs[0] += rod; pUVs[1] += rod; pUVs[2] += rod; pUVs[3] += rod;
        }
    }

    vec4 base = texture(uBase, pUV, L.spot_flash.z);   // DLSS mip bias (0 native)
    if (pc.alphaRef >= 0.0 && base.a < pc.alphaRef) discard;

    // r_ssao_debug 1: show the raw AO map.
    if (SPEC_DEBUG && L.ao_params.w > 0.5) {
        outColor = vec4(vec3(textureLod(uAO, gl_FragCoord.xy * L.ao_params.xy, 0.0).r), base.a);
        return;
    }
    // r_wet_debug 1: rain-map visibility.
    if (SPEC_DEBUG && L.rain_params.z < 0.0) {
        outColor = vec4(vec3(rainVis(vWorldPos)), base.a);
        return;
    }
    // r_pom_debug 1: POM occlusion mask (contact AO x self-shadow);
    // 2: SELF-SHADOW only (horizon-map vs march A/B without the AO noise).
    if (SPEC_DEBUG && L.pom_params3.x > 0.5) {
        outColor = vec4(vec3(L.pom_params3.x > 1.5 ? pomShadow : pomAO * pomShadow), base.a);
        return;
    }
    // r_clustered_debug: per-cluster dynamic-light-count heatmap.
    if (SPEC_DEBUG && L.cluster_params.w > 1.5) {
        int cnt = clusterLightCount(vWorldPos, L.eye_pos.xyz, L.cam_dir.xyz, gl_FragCoord.xy, L.ao_params.xy, L.cluster_params, L.cluster_params2);
        outColor = vec4(clusterHeat(cnt), base.a);
        return;
    }

    // r_sf_debug: Surface Field viz (1 height, 2 slope, 3 curvature, 4 sky, 5 canopy).
    int sfdbg = SPEC_DEBUG ? int(L.sf_params.y + 0.5) : 0;   // SPEC_DEBUG off -> all sf-debug branches DCE
    if (sfdbg > 0) {
        vec3 nn = normalize(vNormal);
        outColor = vec4((sfdbg == 5) ? SC_DebugColor(SC_Refine(SC_GROUND, nn))   // terrain = ground
                                      : SF_DebugColor(vWorldPos, nn, sfdbg), base.a);
        return;
    }

    vec4 dR = texture(uDtR, pUVs[0], L.spot_flash.z);   // DLSS mip bias (0 native)
    vec4 dG = texture(uDtG, pUVs[1], L.spot_flash.z);   // each channel at its OWN parallaxed uv
    vec4 dB = texture(uDtB, pUVs[2], L.spot_flash.z);
    vec4 dA = texture(uDtA, pUVs[3], L.spot_flash.z);

    // HEIGHT-BASED detail blend (Mishkinis "advanced terrain texture splatting"):
    // bias each splat weight by that channel's REAL detail height, so the RAISED
    // material wins per-texel -> sharp interlocking transitions (gravel poking
    // through grass) instead of a soft linear cross-fade. Channels the mask doesn't
    // carry are gated out; degenerate -> falls back to the plain mask. `bw` then
    // drives BOTH the diffuse and the detail normal so the relief follows the same
    // boundaries. (Missing height maps = flat 0.5 -> gracefully ~linear.)
    vec4 chH = pomHits;   // per-channel hit heights straight from the 4 marches
    vec4 bw = mask;
    if (g_cacheOn) {
        bw = g_cacheBW;   // composite cache: weights baked (same r_terra_blend rule) at bake time
    } else if (L.zoff_params.y > 0.005) {
        // r_terra_blend (zoff_params.y): 0 = plain mask cross-fade — the GAMMA/
        // SSFX look (their HeightBlending() ships commented out, asphalt fades
        // smoothly into soil); >0 = Mishkinis height blend with this transition
        // width (0.25 = the old sharp interlocking edges).
        float blendDepth = L.zoff_params.y;
        vec4  wh   = mask + chH;
        float maxw = max(max(wh.x, wh.y), max(wh.z, wh.w));
        vec4  b    = max(wh - (maxw - blendDepth), 0.0) * step(vec4(1e-4), mask);
        float bsum = dot(b, vec4(1.0));
        if (bsum > 1e-5) bw = b / bsum;
    }
    vec3 detail = dR.rgb * bw.r + dG.rgb * bw.g + dB.rgb * bw.b + dA.rgb * bw.a;
    // detH (REAL detail height at the parallax hit) comes from terrainPOM.

    vec3 albedo = 2.0 * base.rgb * detail;

    // Lightmap (hemi/AO) + DYNAMIC R4-style sun (per-pixel N.L x shadow map).
    vec4  lm      = texture(uLmap, vLmapUV);
    float hemiOcc = dot(lm.rgb, vec3(1.0 / 3.0));
    vec3  Nw      = geomN;                 // base normal; detailNormal perturbs it near so sun/dyn catch the relief
    // Detail normal mapping: the primary ground-relief source. Faded to 0 by
    // distance so far terrain skips the 4 taps + AO + gloss entirely.
    float dnStr  = L.pom_params4.y * smoothstep(45.0, 25.0, distance(L.eye_pos.xyz, vWorldPos));
    float cav    = 0.0;
    float glossT = 0.0;
    if (dnStr > 0.0)
        Nw = detailNormal(geomN, pUVs, bw, dnStr, cav, glossT);

    // Micro contact AO: darken grooves. Occlusion = max of the normal cavity
    // (1-n.z) and the detail height pit (1-h^2). Applied to albedo (R4 base*=1-AO).
    float aoStr   = L.pom_params4.z;
    float occlT   = max(cav, 1.0 - detH * detH);
    float microAO = 1.0 - aoStr * clamp(occlT, 0.0, 1.0);
    albedo *= microAO;

    // Mud print shading. The soil is PRESSED IN, not painted over: the print reads
    // through geometry (POM carve + tessellation), the gradient-normal dimple and
    // micro-AO; albedo only gets a subtle packed-moisture darkening (more when wet).
    // The berm is the same soil pushed up — normals/geometry carry it, no tint.
    // Wet ASPHALT is the exception: tracked-on grime IS a dark film on top.
    if ((mudCarve > 0.0 || mudGrime > 0.0) && abs(mudPress) > 0.004) {
        float ew  = L.deform_tex.w;              // world metres per deform texel
        float pXp = SnowDeformPress(vWorldPos + vec3(ew, 0.0, 0.0));
        float pXm = SnowDeformPress(vWorldPos - vec3(ew, 0.0, 0.0));
        float pZp = SnowDeformPress(vWorldPos + vec3(0.0, 0.0, ew));
        float pZm = SnowDeformPress(vWorldPos - vec3(0.0, 0.0, ew));
        vec2 slope = vec2(pXp - pXm, pZp - pZm) * (0.05 * min(mudCarve, 1.2) / (2.0 * ew));
        Nw = normalize(Nw + vec3(slope.x, 0.0, slope.y));
        float dent  = clamp(mudPress, 0.0, 1.0);
        float packD = dent * min(mudCarve, 1.2) * (0.18 + 0.42 * mudWet);   // packed moisture only
        albedo *= 1.0 - min(packD, 0.55);
        glossT  = max(glossT, dent * min(mudCarve, 1.2) * (0.15 + 0.55 * mudWet));   // wet compaction sheen
        albedo  = mix(albedo, albedo * vec3(0.55, 0.52, 0.50), min(dent * mudGrime, 0.6));   // asphalt grime
        // GRADUAL water fill: the field decays 1 -> 0, so the local MAX press is an
        // AGE proxy — a fresh print (max≈1) is dry, water seeps in as it ages and
        // drains away before the print itself fades. Deepest parts fill first.
        float maxP = max(max(max(pXp, pXm), max(pZp, pZm)), mudPress);
        float age  = (1.0 - smoothstep(0.60, 0.88, maxP)) * smoothstep(0.10, 0.28, maxP);
        mudWater   = age * smoothstep(0.35, 0.75, mudPress / max(maxP, 1e-3))
                   * (0.2 + 0.8 * mudWet) * min(mudCarve, 1.0);
    }

    // PUDDLE COVERAGE (0..1). SSS procedural placement default; flow sim is parked.
    float microH = min(detH, 1.0 - cav);
    float pud = (L.pom_params7.x > 0.5) ? sssPuddle(geomN, vWorldPos, detH)
              : (L.pom_params6.x > 0.5) ? smoothstep(0.04, 0.12, simWaterSoft(vWorldPos))
              : 0.0;
    // Water gathered in footprint dents (computed above with the print's AGE — it
    // seeps in gradually, not on the first frame): full wet shading, reflections +
    // ripples. applyWetnessTerrain gates by global wetness, so dry weather keeps
    // prints as damp mud only.
    pud = max(pud, mudWater);

    // r_puddle_debug: 1 = coverage; 2 = per-pixel detail micro-height.
    int pdbg = SPEC_DEBUG ? int(L.pom_params5.w + 0.5) : 0;   // SPEC_DEBUG off -> puddle-debug branches DCE
    if (pdbg > 0) {
        outColor = (L.pom_params6.x > 0.5) ? vec4(waterDebugColor(vWorldPos, pdbg), base.a)
                                           : vec4(vec3(pdbg >= 2 ? microH : pud), base.a);
        return;
    }

    // r_terrain_debug: 1 world normal, 2 micro-AO, 3 detail height, 4 mud field.
    int tdbg = SPEC_DEBUG ? int(L.pom_params4.w + 0.5) : 0;   // SPEC_DEBUG off -> all terrain-debug branches DCE
    if (tdbg == 1) { outColor = vec4(Nw * 0.5 + 0.5, base.a); return; }
    if (tdbg == 2) { outColor = vec4(vec3(microAO),  base.a); return; }
    if (tdbg == 3) { outColor = vec4(vec3(detH),     base.a); return; }
    // 8/9 = SUN-BEAM GROUND DEPOSIT probe (r_sun_beam_ground): 8 = the raw froxel
    // in-scatter THIS pixel samples from uVolume (×8, greyscale). BLACK under a visible
    // shaft ⇒ the volume read is broken (binding 27 / layout / froxel-w), NOT a threshold
    // issue. If it shows the shaft ⇒ lower r_sun_beam_ground_thr. 9 = the deposit gate in
    // RED over the raw in-scatter in GREEN (red where the shaft is depositing).
    if (tdbg == 8 || tdbg == 9) {
        float beam = shaftOverhead(vWorldPos);   // shaft in-scatter found overhead (toward the sun)
        if (tdbg == 8) { outColor = vec4(vec3(beam * 8.0), base.a); return; }
        float gt = smoothstep(L.beam2.w, L.beam2.w * 2.0, beam);
        outColor = vec4(gt, beam * 4.0, 0.0, base.a);
        return;
    }
    // 5..7 = SUN-BEAM RECOVERY probe (r_sun_beam): 5 = crisp atlas directLit at THIS
    // world pos (what the shaft samples), 6 = the smeared screen mask surfaces use,
    // 7 = the recovered gap painted RED (directLit lit where the mask is dark = where
    // the ground re-lights under a beam). All-black in 7 under a visible shaft ⇒ the
    // atlas doesn't resolve the gap on the ground at this page resolution.
    if (tdbg >= 5 && L.shadow_params.w > 0.5) {
        float m  = vsmSunShadow(gl_FragCoord.xy * L.ao_params.xy);
        float dc = distance(L.eye_pos.xyz, vWorldPos);
        float dl = (dc < L.beam_params.y) ? vsmSunShadowGrassDirect(vWorldPos, L.beam_params.w) : m;
        if (tdbg == 5) { outColor = vec4(vec3(dl), base.a); return; }
        if (tdbg == 6) { outColor = vec4(vec3(m),  base.a); return; }
        float gap = clamp(dl - m, 0.0, 1.0);
        outColor = vec4(mix(vec3(m * 0.5), vec3(1.0, 0.0, 0.0), gap), base.a);
        return;
    }
    if (tdbg == 4) {   // deform-field overlay at TRUE world position (bypasses POM —
                       // if this sits under the boot but the visible dent doesn't,
                       // the shift is the POM parallax, not the stamp position)
        float p = SnowDeformOn() ? SnowDeformPress(vWorldPos) : 0.0;
        vec3  c = mix(vec3(0.35), (p >= 0.0) ? vec3(1.0, 0.1, 0.1) : vec3(0.1, 0.3, 1.0),
                      clamp(abs(p) * 1.5, 0.0, 1.0));
        outColor = vec4(c, base.a);
        return;
    }

    // SNOW (Surface Field consumer, r_snow): whiten by surface type x slope x sky
    // exposure - flat up-facing ground accumulates, steep/under-cover none.
    // SF_SkyExposure() is ~9 rainVis gathers — skip when there's no snow this frame
    // (summer: sf_params.w == 0). snow ends 0 either way (smoothstep floor below), so
    // covThr/mix/footprint stay byte-identical; only the gathers are saved.
    float snowBase = 0.0;
    if (SPEC_SNOW && L.sf_params.w > 0.0)
        snowBase = SC_SnowAmount(SC_Refine(SC_GROUND, geomN), geomN, SF_SkyExposure(vWorldPos));
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

    float rawNL   = max(dot(Nw, normalize(-L.sun_dir.xyz)), 0.0);   // unshadowed sun N.L
    float sunMask = rawNL;
    if (sunMask > 0.005) {
        float sunSh = (L.shadow_params.w > 0.5) ? vsmSunShadow(gl_FragCoord.xy * L.ao_params.xy)
                                                : sunShadow(vWorldPos);
        // BEAM-GAP RECOVERY (r_sun_beam, VSM only): the screen mask above is temporally
        // smeared (vsm_resolve EMA), so a thin sun gap through a crown — which the
        // volumetric shaft shows crisply — gets closed here and the ground stays dark
        // under a visible beam. Re-sample the crisp atlas at THIS world pos (same source
        // the shaft uses) and take the max, re-opening the gap so the ground lights up in
        // agreement with the shaft. Skip already-lit ground (no gap to recover) and far
        // ground (non-resident pages -> the direct sample defaults to lit). beamKick adds
        // a small extra sun in the recovered gap so the landing reads as a bright splash.
        float beamKick = 1.0;
        if (L.shadow_params.w > 0.5 && L.beam_params.x > 0.0 && sunSh < 0.98) {
            float d    = distance(L.eye_pos.xyz, vWorldPos);
            float fade = (1.0 - smoothstep(L.beam_params.y * 0.7, L.beam_params.y, d)) * L.beam_params.x;
            if (fade > 0.001) {
                float directLit = vsmSunShadowGrassDirect(vWorldPos, L.beam_params.w);
                float gap = clamp(directLit - sunSh, 0.0, 1.0);   // the gap the temporal mask swallowed
                sunSh    = mix(sunSh, max(sunSh, directLit), fade);
                beamKick = 1.0 + gap * fade * (L.beam_params.z - 1.0);
            }
        }
        sunMask *= sunSh * pomShadow * beamKick;
    }

    // SUN-BEAM GROUND DEPOSIT (r_sun_beam_ground, L.beam2). The visible god-ray is lit
    // air your view ray gathered through the crown gap → the integrated in-scatter at
    // THIS pixel is bright, even though the sun is geometrically occluded on the ground
    // here (proven r_terrain_debug 7: no atlas gap). So don't fake a glow — deposit the
    // sun the SHADOW removed, gated by the beam and shaped by the surface's own N.L, so
    // the terrain lights exactly like the sunlit ground beside it, just switched on by
    // the shaft. Folded into the sun bucket below → gets ×albedo = real sun on the
    // ground, not haze. Aesthetic (no direct light physically reaches this point).
    float beamGround = 0.0;   // PARKED: L.beam2.z (r_sun_beam_ground) defaults 0 → block skipped
    if (L.beam2.z > 0.0 && rawNL > 0.005) {
        float beam = shaftOverhead(vWorldPos);   // brightest shaft in-scatter above this point
        float gate = smoothstep(L.beam2.w, L.beam2.w * 2.0, beam);
        beamGround = clamp(gate * L.beam2.z, 0.0, 1.0) * max(rawNL - sunMask, 0.0);
    }

    // Hemisphere sky fill (R4 hmodel). sun_color/ambient arrive final from vk_env_light.
    vec3  occ      = coloredAO(gtaoVis(), albedo) * pomAO * ssilBoost();   // GTAO x POM AO x SSIL bounce (ambient only)
    float hemiOccL = hemiOcc;
    if (SPEC_DEBUG && L.pom_params3.y > 0.5) { occ = vec3(1.0); hemiOccL = 1.0; }   // r_ao_flat debug
    // Ambient takes the DETAIL normal (Nw), not the flat geometric one. It used to
    // take geomN for a concrete reason — skyAmbient was a point sample of a sharp
    // single-mip cube, so perturbing its direction per pixel turned the sky fill
    // into a mirror. SH9 irradiance is smooth by construction, so that constraint
    // is gone, and terrain relief finally modulates the ambient instead of only
    // the sun. This is what makes slopes read as shaped at dusk, when the sun is
    // below the horizon and ambient is nearly the only light there is.
    vec3 lighting = skyAmbient(gtaoBentN(Nw)) * (hemiOccL * L.sky_params.y) * occ
                  + L.sun_color.rgb  * (sunMask + beamGround)   // + sun-beam ground deposit
                  + L.ambient.rgb * occ * skyAmbientGate(vWorldPos)   // flat sky fill gated by sky visibility (no indoor leak)
                  + dynLights(vWorldPos, Nw);

    // r_shade_debug (zoff_params.z) — same view numbers as world_lmap.frag.
    int ldbg = SPEC_DEBUG ? int(L.zoff_params.z + 0.5) : 0;   // SPEC_DEBUG off -> shade-debug branches DCE
    if (ldbg > 0) {
        vec3 dbg =
            (ldbg == 1) ? albedo :
            (ldbg == 2) ? lm.rgb :
            (ldbg == 3) ? vec3(hemiOccL) :
            (ldbg == 4) ? vec3(gtaoVis()) :
            (ldbg == 5) ? vec3(skyAmbientGate(vWorldPos)) :
            (ldbg == 6) ? skyAmbient(gtaoBentN(geomN)) :
            (ldbg == 7) ? lighting :
            (ldbg == 8) ? L.ambient.rgb * occ * skyAmbientGate(vWorldPos) :
            (ldbg == 9) ? L.sun_color.rgb * sunMask :
                          vec3(clamp(L.rain_params.y, 0.0, 1.0) * rainVis(vWorldPos));
        outColor = vec4(dbg, base.a);
        return;
    }

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
    vec3 wetRefl = vec3(0.0);
    if (SPEC_WET)
        wetRefl = applyWetnessTerrain(albedo, vWorldPos, Nw, pud, sunMask, gtaoBentN(geomN), gtaoVisRaw());

    // Sky specular IBL (r_ibl): grazing sky reflection on the ground. SKY ONLY —
    // terrain's own drySpec already owns the sun highlight (no double glint).
    // Ground is MATTE when dry (dirt/grass ≈ no gloss); a constant reflection on
    // near-flat ground reads as "lacquer/varnish" at grazing angles. So keep it very
    // rough + scale the strength by WETNESS — the sky sheen appears only as the ground
    // gets wet (physically right), and dry terrain gets just a faint soft sky tint.
    // Wet gloss gated by rainVis (ground under sheds/canopy stays matte) and toned
    // down so wet ground reads as a soft sheen, not a mirror.
    // r_ibl master gate hoisted up: iblSpecular self-early-outs at ibl_params.x, but
    // Vv/wetF's rainVis gathers/roughI ran regardless. Skip when IBL is off (default)
    // — output identical (iblSpecular returned 0 below the same threshold).
    vec3 specIBL = vec3(0.0);
    if (SPEC_IBL && L.ibl_params.x > 0.004) {
        vec3  Vv      = normalize(L.eye_pos.xyz - vWorldPos);
        float wetF    = clamp(L.rain_params.y, 0.0, 1.0) * rainVis(vWorldPos);
        float roughI  = mix(0.9, 0.5, wetF);
        specIBL       = iblSpecular(Nw, Vv, roughI, vec3(0.04)) * gtaoVisRaw() * (0.08 + 0.45 * wetF);
    }

    // Distance fog (R4).
    vec3 col = albedo * lighting + drySpec + wetRefl + specIBL;
    float fog = clamp(length(vWorldPos - L.eye_pos.xyz) * L.fog_params.w + L.fog_params.x, 0.0, 1.0);
    col = mix(col, L.fog_color.rgb, fog);

    if (SPEC_DEBUG && L.ibl_params.w > 0.5) { outColor = vec4(specIBL, base.a); return; }   // r_ibl_debug
    if (SPEC_DEBUG) {
        vec4 ptdbg = pointDebugOverlay(vWorldPos);
        if (ptdbg.a > 0.5) { outColor = vec4(ptdbg.rgb, base.a); return; }
    }

    outColor = vec4(col, base.a);
}
