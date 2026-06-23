#version 450
// xrRenderVulkan — detail (grass) vertex shader.
// Per-vertex (binding 0): pos + uv + height
// Per-instance (binding 1): 3 transform rows + color (sun, trail, objId, hemi)
//
// B-min note: vInteractors[i].w == 0 disables the interactor slot, so the
// loop in ApplyInteraction is a no-op. Trail & character interaction kick
// in when sessions B+/C feed real values.

layout(location = 0) in vec3  aPos;
layout(location = 1) in vec2  aUV;
layout(location = 2) in float aHeight;
layout(location = 3) in vec4  aInstRow0;
layout(location = 4) in vec4  aInstRow1;
layout(location = 5) in vec4  aInstRow2;
layout(location = 6) in vec4  aInstColor;   // (sun, trail, objId, hemi)

layout(push_constant) uniform DetailConstants {
    mat4 mViewProj;
    vec4 wind_params;       // (wind_direction, wind_velocity, treeAmplitude, _)
    vec4 wsetup_grass;      // (animspeed, turbulence, push, wave) — SSFX grass tunables
    vec4 wind_anim;         // (drift.x, drift.y, drift.z, minWindSpeed) — Environment.wind_anim
    vec4 vConsts;           // (s_x, s_y, sun.y, ambient_floor)
    vec4 vInteractors[4];   // xyz=pos, w=radius (0 = unused)
    vec4 vSunColor;         // env sun colour (rgb)
    vec4 vHemiColor;        // env hemi colour (rgb)
} pc;

#include "ssfx_wind.glsl"   // SSFX flow-map wind (s_waves @ set0 binding1)

layout(location = 0) out vec2  vUV;
layout(location = 1) out vec4  vColor;    // hemi + floor (shadow-independent part)
layout(location = 2) out float vHeight;
layout(location = 3) out vec3  vSunLit;   // sun part — attenuated by the shadow map in frag
layout(location = 4) out vec3  vWPos;     // world-space position (shadow lookup)

vec3 ApplyInteraction(vec3 pos, float h)
{
    if (h < 0.01) return pos;
    for (int i = 0; i < 4; ++i) {
        float r = pc.vInteractors[i].w;
        if (r < 0.01) continue;
        vec2 d = pos.xz - pc.vInteractors[i].xz;
        float dist = length(d);
        if (dist < r && dist > 0.01) {
            float t = 1.0 - dist / r;
            float s = t * t * h * 0.7;
            pos.xz += (d / dist) * s;
            pos.y  -= s * 0.25;
        }
    }
    return pos;
}

void main()
{
    // Reconstruct 3×4 instance transform (row-major rows in a column-major
    // mat4x3: 3 column-vectors + a translation column).
    mat4x3 inst = mat4x3(
        aInstRow0.xyz, aInstRow1.xyz, aInstRow2.xyz,
        vec3(aInstRow0.w, aInstRow1.w, aInstRow2.w));

    vec3 worldPos = inst * vec4(aPos, 1.0);
    // SSFX flow-map wind (replaces the old sine ApplyWind). Sample at the base
    // world position, drift by Environment.wind_anim.
    WindSetup W = ssfx_wind_setup(pc.wind_params, pc.wsetup_grass, pc.wind_anim.w);
    // wind_params.w = per-detail-type wind scale, pushed each draw: 0 for models
    // flagged DO_NO_WAVING (the tiny "asphalt" plants R4 keeps static), 1 for
    // normal grass. Without this the SSFX wind stretched those micro-meshes.
    worldPos     += ssfx_wind_grass(worldPos, aHeight, W, pc.wind_anim.xy) * pc.wind_params.w;
    worldPos      = ApplyInteraction(worldPos, aHeight);

    // Trail press-down: gen-shader wrote trail intensity into aInstColor.g.
    // Currently always 0 in B-min (TrailMap dummy), so this is a no-op.
    float trail = aInstColor.g;
    if (trail > 0.01 && aHeight > 0.01) {
        float pd = trail * aHeight * 0.6;
        worldPos.y  -= pd;
        worldPos.xz += normalize(aPos.xz + vec2(0.001)) * trail * aHeight * 0.1;
    }

    gl_Position = pc.mViewProj * vec4(worldPos, 1.0);
    vUV         = aUV;

    // Pass the baked per-slot scalars to the fragment: the hemi-occlusion goes
    // in vColor.r — the fragment lights it with the SAME sky-cube ambient the
    // world ground uses (world_lmap skyAmbient), so grass no longer reads dark
    // against the terrain it grows on (R4 lights grass through the same hmodel
    // as the ground — deferred gbuffer). Sun part still travels separately and
    // is shadowed in the fragment.
    float sun  = aInstColor.r;
    float hemi = aInstColor.a;
    vColor     = vec4(hemi, 0.0, 0.0, 1.0);
    vSunLit    = pc.vSunColor.rgb * sun;   // sun colour arrives pre-boosted (r_sun_boost)
    vWPos      = worldPos;
    vHeight    = aHeight;
}
