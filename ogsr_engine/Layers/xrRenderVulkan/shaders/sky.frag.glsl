#version 450
#extension GL_GOOGLE_include_directive : require

// Two-cubemap blended sky, mirroring R4 dxEnvironmentRender::RenderSky, plus an
// animated cloud layer that mirrors R4 dxEnvironmentRender::RenderClouds.
//
// R4 renders a half-cube (`hbox_verts` in dxEnvironmentRender.cpp): top is a
// regular unit cube, the lower hemisphere is squashed flat at y_vis=-0.01,
// and the side face's mid-ring uses tc.y=-1 while the top uses tc.y=1. Net
// effect: the cubemap's V axis gets stretched across the visible upper
// hemisphere, the horizon line lands at tc.y ≈ -0.98 (bottom edge of the
// cubemap), and the lower hemisphere is essentially invisible. X-Ray sky
// cubemaps are authored against this mapping — the sky gradient occupies
// almost the full V range, and the horizon line is at the very bottom.
//
// To get the R4 look from a fullscreen-triangle pass, we remap the per-pixel
// world direction to the same tc the half-cube geometry would have produced
// for that view ray, then sample the cubemap with that.
//
// CLOUDS: R4 draws a SEPARATE pass (RenderClouds) — a flattened cloud dome
// (unit hemisphere scaled (10, 0.4, 10), rotated by sky_rotation) whose UVs
// scroll with global time (clouds.vs): tc = p.xz*CLOUD_TILE + wind*t*CLOUD_SPEED.
// Two cloud textures at different tile/speed are summed and tinted by
// clouds_color (clouds.ps). We reproduce that here per-pixel: project the view
// ray onto the same dome (inverse of the (10,0.4,10) scale), scroll two 2D
// cloud textures by fTimeGlobal, sum + tint, and additively composite over the
// sky. This is what makes the clouds MOVE — the cubemap sky itself is static.

layout(set = 0, binding = 0) uniform samplerCube uSky0;
layout(set = 0, binding = 1) uniform samplerCube uSky1;
layout(set = 0, binding = 2) uniform sampler2D   uClouds0;  // scrolling cloud layer 0
layout(set = 0, binding = 3) uniform sampler2D   uClouds1;  // scrolling cloud layer 1

layout(push_constant) uniform PushConstants {
    vec4 camRightTan_rot;   // .w = skyRotation
    vec4 camUpTan_weight;   // .w = blendWeight
    vec4 camForward_pad;    // .w = unused
    vec4 skyColor_pad;      // .w = unused
    vec4 sunDir_pad;        // xyz = sun TRAVEL dir (to-sun = -sunDir)
    vec4 sunColor_pad;      // xyz = sun colour (env, time-of-day)
    vec4 cloudsColor;       // rgb = clouds_color tint, w = clouds_color.w (intensity / weight)
    vec4 cloudParams;       // x = scroll time (fTimeGlobal/10 * speed), y = enable 0/1, z = intensity, w = unused
} pc;

layout(location = 0) in  vec3 vWorldDir;
layout(location = 0) out vec4 outColor;

// R4 cloudconfig.h — two layers, different tile/speed, fixed wind headings.
const float CLOUD_TILE0  = 0.7;
const float CLOUD_SPEED0 = 0.1;   // 2 * 0.05
const float CLOUD_TILE1  = 2.8;
const float CLOUD_SPEED1 = 0.05;  // 2 * 0.025
// R4 RenderClouds: wd0 heading = 45°, wd1 heading = 67.5° (fixed, not weather wind).
const vec2  WIND0 = vec2(0.70710678, 0.70710678);  // (sin45, cos45)
const vec2  WIND1 = vec2(0.92387953, 0.38268343);  // (sin67.5, cos67.5)

// The half-cube mapping + sky_rotation now live in sky_halfcube.glsl — SHARED
// with ibl_prefilter.comp, which must reproduce this exact mapping to build a
// world-space light probe (see the header for why).
#include "sky_halfcube.glsl"

void main()
{
    vec3 dir = normalize(vWorldDir);

    // R4 rotates the skybox geometry by sky_rotation; sampling at R(-θ)*d
    // lands on the same texel that geometry would expose to the view ray.
    float skyRot = pc.camRightTan_rot.w;
    vec3 dirRot = SkyUnrotate(dir, skyRot);

    vec3 sampleDir = SampleDirHalfCube(dirRot);

    vec4 c0 = texture(uSky0, sampleDir);
    vec4 c1 = texture(uSky1, sampleDir);
    vec3 col = mix(c0.rgb, c1.rgb, clamp(pc.camUpTan_weight.w, 0.0, 1.0));

    // skybox.vs in R4 pre-scales the per-vertex tint by 1.7 ("pre-scale by
    // tonemap"). We do the same here so brightness matches R4 output.
    vec3 tint = pc.skyColor_pad.xyz * 1.7;

    // Sun disk + aureole — additive over the sky, gated above the horizon. The
    // bright HDR core blooms (bloom pass) into a glow; the wide power terms give
    // the soft atmospheric halo around it. sunColor is the time-of-day env colour
    // (warm at dawn/dusk). Compared in UNROTATED world space (skyRot is cube-only).
    vec3  toSun = normalize(-pc.sunDir_pad.xyz);
    float cosA  = dot(dir, toSun);
    float above = smoothstep(-0.08, 0.02, toSun.y);              // fade out below horizon
    float disk  = smoothstep(0.9993, 0.9997, cosA);             // sharp core (~1° radius)
    float halo  = pow(max(cosA, 0.0), 350.0) * 0.35
                + pow(max(cosA, 0.0),  40.0) * 0.05;            // aureole
    vec3  sun   = pc.sunColor_pad.xyz * ((disk * 12.0 + halo) * above);

    vec3 outc = col * tint + sun;

    // --- Animated clouds (R4 RenderClouds port) -----------------------------
    // Gate: r_clouds cvar (cloudParams.y) AND clouds_color.w > 0 (R4 skips the
    // pass when the weather's cloud weight is ~0 — e.g. clear night).
    if (pc.cloudParams.y > 0.5 && pc.cloudsColor.w > 0.001 && dirRot.y > 0.0) {
        // Project the view ray onto R4's flattened cloud dome. The dome is a
        // unit hemisphere scaled (10, 0.4, 10); clouds.vs tiles by the OBJECT-
        // space position p.xz and fades by pow(p.y, 25). Undo the scale to get
        // p from the world ray: p = normalize(S^-1 * dirRot), S^-1 = (0.1,2.5,0.1).
        vec3 p = normalize(vec3(dirRot.x * 0.1, dirRot.y * 2.5, dirRot.z * 0.1));

        float t = pc.cloudParams.x;                              // fTimeGlobal/10 * speed
        vec2 tc0 = p.xz * CLOUD_TILE0 + WIND0 * t * CLOUD_SPEED0;
        vec2 tc1 = p.xz * CLOUD_TILE1 + WIND1 * t * CLOUD_SPEED1;

        vec4 s0 = texture(uClouds0, tc0);
        vec4 s1 = texture(uClouds1, tc1);

        // R4 clouds.ps: mix = I.color * (s0 + s1). clouds.s blends the result
        // with (srcalpha, invsrcalpha) — a normal alpha blend, so the cloud
        // SHAPE lives in the texture ALPHA, not the colour. I.color.w =
        // clouds_color.w * pow(p.y, 25) → fade to the horizon. Reproduce that
        // alpha composite over the sky (additive-of-rgb was wrong → invisible
        // clouds; the visible ones were baked into the day cubemap).
        float fade   = pow(clamp(p.y, 0.0, 1.0), 25.0);
        vec3  cloudRGB = pc.cloudsColor.rgb * (s0.rgb + s1.rgb);
        float cloudA   = clamp(pc.cloudsColor.w * fade * (s0.a + s1.a)
                             * pc.cloudParams.z, 0.0, 1.0);
        outc = mix(outc, cloudRGB, cloudA);
    }

    outColor = vec4(outc, 1.0);
}
