#version 450
#extension GL_GOOGLE_include_directive : require
#define VSM_SET 2                 // env set is bound at set 2 for skinned
#include "vsm_sample.glsl"        // vsmSunShadow — screen-space VSM mask (set 2, binding 14)
#define CLUSTER_SET 2             // env set is bound at set 2 for skinned
#include "cluster_lights.glsl"    // clustered forward (set 2 bindings 17-19, r_clustered)
// skinned.frag - forward skinned shading with the diffuse texture.
// The leaf's WorldMaterial descriptor set (base/detail/lmap) is bound as set 1;
// we sample the base diffuse (binding 0) and light it with the REAL environment
// sun + hemi + ambient (set 2), so dynamics track time-of-day like the sky does.
// vertHW UVs are raw floats (no SHORT2 scale), so v_uv is used directly.

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec3 v_nrm;
layout(location = 2) in vec3 v_wpos;   // world-space position (shadow lookup)

layout(location = 0) out vec4 o_color;

layout(set = 1, binding = 0) uniform sampler2D uTexDiffuse;

// Shared with skinned.vert; the fragment only reads hudMode (offset 76).
layout(push_constant) uniform PC {
    mat4  mvp;
    uint  skinMode;
    uint  baseBone;
    uint  boneCount;
    float hudMode;    // 1 = first-person HUD (hands/weapon)
    float hemi;       // sky-ambient gate (ray-traced sky visibility, 0..1)
} pc;

// Per-frame environment lighting (CEnvDescriptorMixer, filled in Pass_Skinned).
// sun_dir.xyz is X-Ray's direction the light TRAVELS (points downward, .y<0);
// the direction toward the sun for N-L is therefore -sun_dir. .w fields unused.
struct DynLight {
    vec4 pos;     // xyz = world position, w = range
    vec4 color;   // rgb = colour,         w = 1 spot / 0 point
    vec4 dir;     // xyz = spot direction, w = cos(cone/2)
};
layout(set = 2, binding = 0) uniform Lighting {
    vec4 sun_dir;     // xyz = travel dir (downward)
    vec4 sun_color;   // rgb
    vec4 hemi_color;  // rgb (w = R2 correction, unused here)
    vec4 ambient;     // rgb
    mat4 sun_vp;      // sun light view-proj (shadow lookup)
    vec4 counts;      // x = dynamic light count
    DynLight lights[16];
    mat4 spot_vp;        // spot (flashlight) shadow view-proj
    vec4 shadow_params;  // x = spot-shadowed light index (-1 none), y = point-shadowed index
    mat4 sun_near_vp;    // sun cascade 0 view-proj (25 m, per-frame, R4 scheme)
    mat4 sun_c1_vp;      // sun cascade 1 view-proj (60 m, per-frame)
    vec4 fog_color;      // rgb haze colour (env)
    vec4 fog_params;     // x=-near*r, y=near, z=far, w=r; fog = saturate(dist*w + x)
    vec4 eye_pos;        // xyz camera world pos
    vec4 sky_params;     // x=cube cross-fade weight, y=ambient scale, z=sample LOD
    vec4 ao_params;      // x=1/screenW, y=1/screenH, z=AO strength (0=off)
    // Tail of the shared LightUBO — declared so the clustered fields at the end
    // are reachable (cam_dir for the view-space depth, cluster_params for the
    // froxel lookup). Everything between is padding skinned doesn't sample.
    mat4 _pad_rain_vp;
    vec4 _pad_rain_params;
    mat4 _pad_scene_vp;
    vec4 cam_dir;          // xyz camera forward (unit) — clustered z-view
    vec4 _pad_cam_rightT;
    vec4 _pad_cam_topT;
    vec4 _pad_pom_params;
    vec4 _pad_pom_params2;
    vec4 _pad_pom_params3;
    vec4 _pad_pom_params4;
    vec4 pom_params5;   // y = glass opacity ceiling (r_glass_opacity); rest unused here
    vec4 _pad_pom_params6;
    vec4 _pad_pom_params7;
    vec4 cluster_params;   // x=sliceScale, y=sliceBias, z=near, w=enable (0/1/2 debug)
    vec4 cluster_params2;  // x=gridX, y=gridY, z=gridZ, w=maxLightsPerCluster
} L;
layout(set = 2, binding = 2) uniform sampler2D uSpotShadow;
layout(set = 2, binding = 3) uniform samplerCube uPointShadow;
layout(set = 2, binding = 4) uniform sampler2D uShadowNear;    // sun cascade 0 (~0.61 cm texels)
layout(set = 2, binding = 5) uniform sampler2D uShadowC1;      // sun cascade 1 (~1.46 cm texels)
layout(set = 2, binding = 6) uniform samplerCube uSky0;        // sky ambient cube 0 (weather A)
layout(set = 2, binding = 7) uniform samplerCube uSky1;        // sky ambient cube 1 (weather B)
layout(set = 2, binding = 10) uniform sampler2D uSpotCookie;   // flashlight beam texture (cookie)
layout(set = 2, binding = 8) uniform sampler2D uAO;            // GTAO (half-res)
layout(set = 2, binding = 21) uniform sampler2D uIL;          // SSIL one-bounce indirect light (half-res, r_ssil)

// SSIL ambient boost - see light_ubo.glsl (SSFX hdiffuse *= IL). Multiplies the
// ambient where AO does; 0/no-op where there's no bounce or r_ssil is off.
vec3 ssilBoost()
{
    vec3 il = textureLod(uIL, gl_FragCoord.xy * L.ao_params.xy, 0.0).rgb;
    return vec3(1.0) + il / (1.0 + il);
}

// GTAO visibility - see world_lmap.frag (occludes hemi+ambient only). NPCs are
// IN the prepass depth (Skinned_RenderDepthPrepass), so this is real per-pixel
// AO of the character itself + its contact with the surroundings.
float gtaoVis()
{
    float ao = textureLod(uAO, gl_FragCoord.xy * L.ao_params.xy, 0.0).r;
    // HALF the strength exponent vs the world receivers: AO normals come from
    // half-res depth derivatives, which are noisy on small curvy skinned
    // geometry (limbs, folds) - full strength reads as dark "dirt" speckle on
    // characters. They still CAST full-strength AO onto the world.
    return pow(clamp(ao, 0.0, 1.0), L.ao_params.z * 0.5);
}

// Colored AO - see world_lmap.frag (R4 compute_colored_ao port).
vec3 coloredAO(float ao, vec3 albedo)
{
    vec3 a =  2.0404 * albedo - 0.3324;
    vec3 b = -4.7951 * albedo + 0.6417;
    vec3 c =  2.7552 * albedo + 0.6903;
    return max(vec3(ao), ((ao * a + b) * ao + c) * ao);
}

// Hemisphere sky ambient (R4 hmodel.h) - see world_lmap.frag.
vec3 skyAmbient(vec3 N)
{
    float lod = L.sky_params.z;
    return mix(textureLod(uSky0, N, lod).rgb, textureLod(uSky1, N, lod).rgb,
               clamp(L.sky_params.x, 0.0, 1.0));
}

// Spot/point shadow + dynamic lights - same model as world_lmap.frag.
// LINEAR-depth compare with a world epsilon (see shadow_common.glsl — a
// constant NDC bias leaked light through fences near the spot's far plane).
float spotLinZ(float zndc, float f)
{
    const float n = 0.5;   // ComputeSpotVP near plane
    return n * f / max(f - zndc * (f - n), 1e-4);
}
float spotShadowF(vec3 wp, float range)
{
    vec4 c = L.spot_vp * vec4(wp, 1.0);
    if (c.w <= 0.0) return 1.0;
    vec3 ndc = c.xyz / c.w;
    vec2 uv = ndc.xy * 0.5 + 0.5;
    uv.y = 1.0 - uv.y;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z > 1.0) return 1.0;
    float f    = max(range, 1.0);
    float zRef = spotLinZ(ndc.z, f) - 0.08;
    vec2  texel = 1.0 / vec2(textureSize(uSpotShadow, 0));
    float sum = 0.0;
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x)
            sum += (zRef <= spotLinZ(texture(uSpotShadow, uv + vec2(x, y) * texel).r, f)) ? 1.0 : 0.0;
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

// Shade ONE dynamic light — same model as world_lmap.frag (passed-in fields so
// it serves both the UBO array fallback and the clustered SSBO). gi = global idx.
vec3 shadeDynLight(vec4 lpos, vec4 lcol, vec4 ldir, vec3 wp, vec3 N, int gi, int sIdx, int pIdx)
{
    vec3  dv = lpos.xyz - wp;
    float r  = lpos.w;
    float d2 = dot(dv, dv);
    if (d2 >= r * r) return vec3(0.0);
    float d   = sqrt(max(d2, 1e-6));
    vec3  ld  = dv / d;
    // Narrow beams: windowed falloff (far half of the beam still lights) —
    // see light_shade.glsl.
    float att;
    if (lcol.w > 0.5 && ldir.w > 0.87) {
        att = 1.0 - (d2 / (r * r));
        att *= att;
    } else {
        att = 1.0 - d / r;
        att *= att;
    }
    if (lcol.w > 0.5) {
        // Narrow beams: full inside the cone + spill to 2x the angle — see
        // light_shade.glsl (the axis-peaked ramp left beam-lit ground/NPCs dark).
        float ca = dot(-ld, ldir.xyz);
        if (ldir.w > 0.87) {
            float co = 2.0 * ldir.w * ldir.w - 1.0;
            att *= clamp((ca - co) / max(ldir.w - co, 1e-3), 0.0, 1.0);
        } else
            att *= clamp((ca - ldir.w) / max(1.0 - ldir.w, 1e-3), 0.0, 1.0);
    }
    vec3 tint = lcol.rgb;
    if (gi == sIdx) {
        att *= spotShadowF(wp, r);
        if (L.shadow_params.z > 0.5) {
            vec4 cc = L.spot_vp * vec4(wp, 1.0);
            if (cc.w > 0.0) {
                vec2 cuv = (cc.xy / cc.w) * 0.5 + 0.5;
                cuv.y = 1.0 - cuv.y;
                tint *= textureLod(uSpotCookie, clamp(cuv, 0.0, 1.0), 0.0).rgb;
            }
        }
    }
    else if (gi == pIdx) att *= pointShadowF(wp, lpos.xyz, r);
    float ndl = dot(N, ld);
    // Narrow-beam wrap diffuse — same as light_shade.glsl (grazing headlight
    // beams painted no light pool at plain Lambert).
    if (lcol.w > 0.5 && ldir.w > 0.87) ndl = (ndl + 0.4) * (1.0 / 1.4);
    return tint * (att * max(ndl, 0.0));
}

// Clustered (r_clustered): only this froxel's lights from the SSBO; fallback:
// the old per-fragment loop over the UBO's 16. See world_lmap.frag.
vec3 dynLights(vec3 wp, vec3 N)
{
    int sIdx = int(L.shadow_params.x);
    int pIdx = int(L.shadow_params.y);
    vec3 acc = vec3(0.0);
    if (L.cluster_params.w > 0.5) {
        int ci  = clusterOfFragment(wp, L.eye_pos.xyz, L.cam_dir.xyz, gl_FragCoord.xy, L.ao_params.xy, L.cluster_params, L.cluster_params2);
        int mxp = int(L.cluster_params2.w);
        int cnt = int(clusterGrid[ci]);
        for (int k = 0; k < cnt; ++k) {
            int gi = int(clusterIndex[ci * mxp + k]);
            acc += shadeDynLight(clusterLights[gi].pos, clusterLights[gi].color, clusterLights[gi].dir, wp, N, gi, sIdx, pIdx);
        }
    } else {
        int n = int(L.counts.x + 0.5);
        for (int i = 0; i < n; ++i)
            acc += shadeDynLight(L.lights[i].pos, L.lights[i].color, L.lights[i].dir, wp, N, i, sIdx, pIdx);
    }
    return acc;
}
layout(set = 2, binding = 1) uniform sampler2D uShadow;   // sun shadow map (D32, manual compare)

// Bilinear-weighted near-cascade PCF tap (textureGather) - see world_lmap.frag.
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

// Sun shadow lookup - same convention as the world shaders (world_lmap.frag).
// NEAR cascade first: leaf-shaped dapples on NPCs standing under trees.
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
    vec2  texel = 1.0 / vec2(textureSize(uShadow, 0));
    float sum = 0.0;
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x)
            sum += (ref <= texture(uShadow, uv + vec2(x, y) * texel).r) ? 1.0 : 0.0;
    return sum * (1.0 / 9.0);
}

void main()
{
    vec4 base = texture(uTexDiffuse, v_uv);

    // Collimator/red-dot sight marks (R4 hud_reddotsight: additive, unlit):
    // output the texture as-is — the additive pipeline ADDS it over the sight
    // glass; lighting/fog/alpha-test don't apply. ×2 keeps the mark vivid
    // through the HDR exposure+tonemap.
    if ((pc.skinMode & 16u) != 0u) {
        // Angle fade for light-beam planes (weapon torches, lamp selflight) — an
        // edge-on plane vanishes instead of showing as a hard line; collimator
        // marks are face-on when aiming, so they keep full brightness. The HUD's
        // v_wpos is camera-relative → the view vector is just -v_wpos there.
        vec3  Vv   = (pc.hudMode > 0.5 && dot(v_wpos, v_wpos) < 9.0)
                   ? -normalize(v_wpos)
                   : normalize(L.eye_pos.xyz - v_wpos);
        float face = abs(dot(normalize(v_nrm), Vv));
        float w    = face * face;
        o_color = vec4(base.rgb * 2.0 * w, base.a * w);
        return;
    }

    // GLASS pane (skinMode bit 32 — kinematics furniture/door panes, blended
    // pipeline): keep the lit shading but skip the cutout test (semi-transparent
    // glass texels would all be discarded) and cap the blend alpha at the end.
    bool isGlass = (pc.skinMode & 32u) != 0u;
    if (!isGlass && base.a < 0.25)   // alpha-tested skinned parts (straps, hair, foliage)
        discard;

    // r_ssao_debug 1: NPCs draw the raw AO map too (never the HUD hands).
    if (L.ao_params.w > 0.5 && pc.hudMode < 0.5) {
        o_color = vec4(vec3(textureLod(uAO, gl_FragCoord.xy * L.ao_params.xy, 0.0).r), 1.0);
        return;
    }

    vec3  N     = normalize(v_nrm);
    vec3  toSun = normalize(-L.sun_dir.xyz);          // direction toward the sun
    float ndl   = max(dot(N, toSun), 0.0);            // directional (sun) term
    float hemiF = 0.6 + 0.4 * N.y;                    // up-facing surfaces see more sky

    // Sun shadow: only worth the PCF taps when the sun term is non-zero, and never
    // for the HUD (its bones are view-relative - v_wpos is not a world position).
    // The NPC is its OWN caster in the map - sample offset along the normal so
    // thin limbs don't self-shadow (acne would eat the whole sun term).
    if (ndl > 0.0 && pc.hudMode < 0.5) {
        // VSM screen-space mask (r_vsm) vs cascade. NOTE: the VSM atlas has no skinned
        // casters yet → NPC self-shadow returns with that work; static-cast shadows
        // (building on NPC) are correct now and match the VSM-shadowed world.
        float sunSh = (L.shadow_params.w > 0.5) ? vsmSunShadow(gl_FragCoord.xy * L.ao_params.xy)
                                                : sunShadow(v_wpos + N * 0.05);
        ndl *= sunSh;
    }

    // Full env sun colour - the world shaders use it untinted (dynamic R4-style
    // N-L sun), so NPCs must match or they read "standing in the dark" next to
    // a warm-lit wall. (The old 60%-desaturate hack predates env-lit statics.)
    // sun_color/ambient arrive final from vk_env_light (r_sun_boost /
    // r_ambient_floor) - same values as the world shaders.
    vec3 sun = L.sun_color.rgb;

    // Hemisphere sky fill (R4 hmodel) for the body; the HUD keeps the flat hemi
    // term because its bones are view-relative (N would rotate with the camera,
    // so a sky-cube sample in N is meaningless there).
    // GTAO gates ambient/hemi like the world shaders - never for the HUD
    // (hands would inherit the AO of whatever wall is behind them on screen).
    vec3 occ = (pc.hudMode < 0.5) ? coloredAO(gtaoVis(), base.rgb) * ssilBoost() : vec3(1.0);   // HUD path = vec3(1) → no IL on hands
    vec3 skyFill = (pc.hudMode < 0.5) ? skyAmbient(N) * (L.sky_params.y * 0.7 * pc.hemi)
                                      : vec3(0.0);   // HUD: light fully replaced below
    vec3 light = L.ambient.rgb * occ
               + skyFill * occ
               + sun * ndl;

    // Dynamic point/spot lights - skip for the HUD (v_wpos is not world-space
    // there; the flat hudLight floor below keeps hands readable instead).
    if (pc.hudMode < 0.5)
        light += dynLights(v_wpos, N);

    // First-person HUD (hands/weapon): the bones live in a view-relative space, so
    // dotting their normals against the world-space sun gives an unreliable (often
    // near-zero) N-L - hands look too dark. Floor the HUD with a flat, sun-direction-
    // independent term so they're consistently lit regardless of camera/sun angle.
    // The SUN part is gated by the shadow sampled at the CAMERA position: standing
    // in a light gap - warm sun-lit hands (the R4 look where hands carry orange
    // morning light), under a crown / indoors - hands drop to ambient. Whole-hand
    // modulation, not per-pixel dapples (HUD has no world positions), but it sells
    // "light through the foliage" where it matters most.
    if (pc.hudMode > 0.5) {
        // Per-pixel dapples: the HUD renders "camera at origin" (X-Ray shifts
        // the hud space by -camera to keep float precision), so v_wpos is the
        // WORLD-ORIENTED position RELATIVE to the camera - the true world
        // point is simply v_wpos + eye. Detect which convention the data is in
        // (near origin - camera-relative; near eye - already world) and
        // reconstruct; the shadow map then paints real light shafts through
        // the foliage onto the hands, like R4.
        vec3 hudWp;
        if (dot(v_wpos, v_wpos) < 9.0)                       // |v_wpos| < 3 - camera-relative
            hudWp = v_wpos + L.eye_pos.xyz;
        else if (distance(v_wpos, L.eye_pos.xyz) < 3.0)      // already world
            hudWp = v_wpos;
        else
            hudWp = L.eye_pos.xyz + vec3(0.0, 0.1, 0.0);     // unknown - camera point
        float hudSun = sunShadow(hudWp);
        // REPLACE the body-path light (it adds the sun UNSHADOWED for the HUD -
        // max() with it was erasing the dapples). ndl is approximate on HUD
        // normals, so the sun term keeps a floor instead of full N-L shaping.
        // 0.44 = old 0.55 - 1.25 - sun_color now arrives pre-boosted
        // (r_sun_boost), so the tuned HUD level is unchanged.
        light = L.ambient.rgb
              + L.hemi_color.rgb * hemiF * 1.4
              + sun * (0.44 + 0.44 * ndl) * hudSun;
    }

    vec3 col = base.rgb * light;
    float outA = 1.0;

    // GLASS pane (kinematics furniture/doors/vehicle windows, NPC glasses) —
    // R4 model_env_lq.ps: colour = light × lerp(ENV REFLECTION, texture, a),
    // blend alpha = the texture's own alpha. Clean glass ≈ invisible + sheen.
    // (Lit-blend lightplanes leaves are never drawn — Pass_LightCones replaces
    // them with real volumetric beams; DrawSkinnedList skips the draw.)
    if (isGlass) {
        // GLASS: R4 base + fresnel + sun glint — see world_lmap.frag for the note.
        float aG   = min(base.a, L.pom_params5.y);
        vec3  V    = normalize(v_wpos - L.eye_pos.xyz);
        vec3  Rv   = reflect(V, N);
        float NoV  = clamp(dot(N, -V), 0.0, 1.0);
        float fres = 0.04 + 0.96 * pow(1.0 - NoV, 5.0);
        float xf   = clamp(L.sky_params.x, 0.0, 1.0);
        vec3  env  = textureLod(uSky0, Rv, 1.0).rgb;
        if (xf > 0.01) env = mix(env, textureLod(uSky1, Rv, 1.0).rgb, xf);
        vec3 glint   = sun * (pow(max(dot(Rv, toSun), 0.0), 128.0) * 2.0 * clamp(ndl * 3.0, 0.0, 1.0));
        // Reflection modulated by the LOCAL lighting; fresnel coverage weighted by
        // reflection brightness (no dark "tint" film indoors) — see world_lmap.frag.
        vec3 reflCol = env * light * (0.5 + fres) + glint;
        float fCov = fres * clamp(dot(reflCol, vec3(0.6)), 0.0, 1.0);
        float aOut = clamp(aG + fCov * (1.0 - aG), 1e-4, 1.0);
        col  = (base.rgb * light * aG + reflCol * fCov * (1.0 - aG)) / aOut;
        outA = aOut;
    }

    // Distance fog (R4) - see world_lmap.frag. NEVER on the HUD: its v_wpos is
    // view-relative, not a world position, so the distance would be garbage.
    if (pc.hudMode < 0.5) {
        float fog = clamp(length(v_wpos - L.eye_pos.xyz) * L.fog_params.w + L.fog_params.x, 0.0, 1.0);
        col = mix(col, L.fog_color.rgb, fog);
        if (isGlass) outA *= (1.0 - fog) * (1.0 - fog);   // R4: alpha fades with fog²
    }

    o_color = vec4(col, outA);
}
