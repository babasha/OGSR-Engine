#version 450
// Tonemap / auto-exposure / bloom / color-grading composite — the final step
// mapping the HDR scene to the display. Ports the R4 (Enhanced Shaders) chain:
//   1. auto-exposure (bloom_luminance_3.ps): exposure = middlegray/(avgLum+low),
//      avgLum from the HDR target's top mip, clamped.
//   2. bloom add (blend_soft, common_functions.h): R4 inverse-tonemaps the LDR,
//      adds the blurred bloom, re-tonemaps — mathematically "add bloom to the
//      linear HDR before tonemapping", which is exactly what we do here.
//   3. CDL color grading in log space (ACES_Color_Grading.h + ACES_settings.h):
//      Slope = r2_img_exposure, Power = 2*(1 - r2_img_cg), Sat = r2_img_saturation,
//      applied on ACEScc-encoded values (USE_LOG_GRADING). Neutral by default.
//      (R4's Contrast_Reduction 0.7 × Contrast_Boost 1.42857 cancel exactly — skipped.)
//   4. Reinhard x/(x+1) normalized by white = 11.2 (tonemap_srgb.h) — soft
//      filmic highlights instead of a hard clip; bloom supplies the glow.
//   5. gamma (img_corrections.h): pow(c, 1/r2_img_gamma).

layout(set = 0, binding = 0) uniform sampler2D uHDR;     // scene, full mip chain
layout(set = 0, binding = 1) uniform sampler2D uBloom;   // blurred bright-pass (quarter res)
layout(set = 0, binding = 2) uniform sampler2D uDistort; // particle heat-haze offsets (rg, neutral 0.5)
layout(set = 0, binding = 3) uniform sampler2D uDepth;   // scene depth (SSR puddles)

// Shared per-frame environment set (same UBO/set the world shaders read at
// set 1) — the SSR puddles need the rain mask/VP, the wetness factor and the
// camera frustum terms. Layout MUST match vk_env_light.h LightUBO.
struct DynLight {
    vec4 pos;
    vec4 color;
    vec4 dir;
};
layout(set = 1, binding = 0) uniform Lighting {
    vec4 sun_dir;
    vec4 sun_color;
    vec4 hemi_color;
    vec4 ambient;
    mat4 sun_vp;
    vec4 counts;
    DynLight lights[16];
    mat4 spot_vp;
    vec4 shadow_params;
    mat4 sun_near_vp;
    mat4 sun_c1_vp;
    vec4 fog_color;
    vec4 fog_params;
    vec4 eye_pos;
    vec4 sky_params;     // w = ripple clock
    vec4 ao_params;
    mat4 rain_vp;        // straight-down ortho VP for the rain occlusion map
    vec4 rain_params;    // x=rain density, y=wetness, z=darken (<0 = debug), w=refl scale
    mat4 scene_vp;       // this frame's view-proj (world - clip)
    vec4 cam_dir;        // xyz = camera forward, w = proj _33
    vec4 cam_rightT;     // xyz = right x tan(fovX/2), w = proj _43
    vec4 cam_topT;       // xyz = top x tan(fovY/2)
    vec4 pom_params;     // (declared for layout — unused here)
    vec4 pom_params2;
    vec4 pom_params3;
    vec4 pom_params4;
    vec4 pom_params5;
    vec4 pom_params6;    // x=water-sim enable, y=murk extinction (/m), z=refraction scale
} L;
layout(set = 1, binding = 9)  uniform sampler2D uRainMap; // top-down rain occlusion
layout(set = 1, binding = 11) uniform sampler2D uWater;   // water depth (flow sim, metres)

layout(push_constant) uniform PC {
    vec4 p0;   // x=whitePoint, y=topMipLOD, z=middleGray, w=lowLum
    vec4 p1;   // x=expMin, y=expMax, z=expComp, w=bloomIntensity
    vec4 p2;   // x=cdlSlope, y=cdlSaturation, z=invGamma, w=distortAmount (0 = off)
    vec4 p3;   // xyz=cdlPower (2*(1-cg)), w=unused
} pc;

layout(location = 0) out vec4 outColor;

const vec3 LUM = vec3(0.2126, 0.7152, 0.0722);

// Procedural-noise domain salts for the rain-ripple hash. These are fixed
// build-provenance constants (mirror of ogsr::sig in vk_authorship.h) — the same
// numbers a classic value-noise hash would use, just named after and bound to
// this build's identity. Retuning them only reshuffles the ripple pattern.
const vec2  SALT_ZEFIR   = vec2(127.1, 311.7);
const vec2  SALT_CATARA  = vec2(269.5, 183.3);
const float SALT_SARATOV = 43758.5453;

// ---- SSR puddle helpers (mirror the world shaders' procedural masks) -----
float puddleMask(vec2 p)
{
    float n = sin(p.x * 0.71 + sin(p.y * 0.53) * 1.7)
            * sin(p.y * 0.67 + sin(p.x * 0.49) * 1.7);
    return smoothstep(0.15, 0.65, n * 0.5 + 0.5);
}

vec2 rippleLayer(vec2 p, float t)
{
    vec2 cell = floor(p);
    vec2 f = p - cell;
    float h1 = fract(sin(dot(cell, SALT_ZEFIR))  * SALT_SARATOV);
    float h2 = fract(sin(dot(cell, SALT_CATARA)) * SALT_SARATOV);
    vec2  cc = vec2(0.3) + 0.4 * vec2(h1, h2);
    float ph = fract(t + h1);
    float d  = length(f - cc);
    float ring = sin(clamp((d - ph * 0.5) * 30.0, -3.1416, 3.1416));
    float fade = (1.0 - ph) * smoothstep(0.5, 0.25, d);
    return (d > 1e-4 ? (f - cc) / d : vec2(0.0)) * (ring * fade);
}

vec2 rainRipples(vec2 p, float t)
{
    return rippleLayer(p * 2.2,                     t * 1.05)
         + rippleLayer(p * 1.34 + vec2(0.50, 0.25), t * 1.31)
         + rippleLayer(p * 1.91 + vec2(0.31, 0.50), t * 1.58);
}

// Water depth (metres) from the flow sim, sampled at the column via rain_vp.
float waterDepthAt(vec3 wp)
{
    vec3 n = (L.rain_vp * vec4(wp, 1.0)).xyz;
    vec2 uvr = n.xy * 0.5 + 0.5; uvr.y = 1.0 - uvr.y;
    if (uvr.x < 0.0 || uvr.x > 1.0 || uvr.y < 0.0 || uvr.y > 1.0) return 0.0;
    // Blurred (5-tap) — spreads the sim's crease/seam line-pooling into smooth
    // area puddles instead of thin lines along mesh folds.
    vec2 px = 5.0 / vec2(textureSize(uWater, 0));
    return textureLod(uWater, uvr, 0.0).r * 0.4
         + (textureLod(uWater, uvr + vec2(px.x, 0.0), 0.0).r
          + textureLod(uWater, uvr - vec2(px.x, 0.0), 0.0).r
          + textureLod(uWater, uvr + vec2(0.0, px.y), 0.0).r
          + textureLod(uWater, uvr - vec2(0.0, px.y), 0.0).r) * 0.15;
}

// Binary rain-map visibility (single tap — only gates the puddle mirror).
float rainVisB(vec3 wp)
{
    vec3 n = (L.rain_vp * vec4(wp, 1.0)).xyz;
    vec2 uvr = n.xy * 0.5 + 0.5;
    uvr.y = 1.0 - uvr.y;
    if (uvr.x < 0.0 || uvr.x > 1.0 || uvr.y < 0.0 || uvr.y > 1.0 || n.z <= 0.0 || n.z >= 1.0)
        return 1.0;
    return (n.z - 0.0015 <= textureLod(uRainMap, uvr, 0.0).r) ? 1.0 : 0.0;
}

void main()
{
    // 1. Whole-frame average luminance from the top mip → R4 auto-exposure.
    vec3  avg    = textureLod(uHDR, vec2(0.5), pc.p0.y).rgb;
    float avgLum = max(dot(avg, LUM), 1e-4);
    float exposure = clamp(pc.p0.z / (avgLum + pc.p0.w), pc.p1.x, pc.p1.y) * pc.p1.z;

    // Particle heat haze (R2/R4 combine_2.ps): offset the scene UV by the
    // distortion buffer (rg around neutral 0.5). At zero offset the linear
    // sample at the pixel centre equals the old texelFetch exactly.
    vec2 uv = gl_FragCoord.xy / vec2(textureSize(uHDR, 0));
    vec2 sceneUV = uv;
    if (pc.p2.w > 0.0)
        sceneUV += (texture(uDistort, uv).rg - 0.5) * pc.p2.w;
    vec3 c = textureLod(uHDR, sceneUV, 0.0).rgb * exposure;

    // ---- Water: volumetric depth (murk + refraction) + SSR surface mirror ----
    // The flow sim gives a per-column water DEPTH. The deeper the water the more
    // it absorbs/scatters (Beer-Lambert) → a murky bottom that reads as DEEP
    // ("knee-deep, can't see the floor"), the bottom refracted by the surface
    // ripples; the SSR march then mirrors the scene on the surface. Falls back
    // to the procedural puddle mask when the sim is off. (rain_params.z<0 = the
    // wet-debug flag → skip.)
    float wet   = L.rain_params.y;
    bool  simOn = L.pom_params6.x > 0.5;
    if ((wet > 0.01 || simOn) && L.rain_params.z >= 0.0) {
        float zndc = textureLod(uDepth, uv, 0.0).r;
        if (zndc < 0.9999) {
            // Depth → world via the frustum-ray basis (SSAO scheme, verified).
            float zview = clamp(L.cam_rightT.w / (zndc - L.cam_dir.w), 0.0, 10000.0);
            vec2 ndc = vec2(uv.x * 2.0 - 1.0, 1.0 - 2.0 * uv.y);
            vec3 ray = L.cam_dir.xyz + L.cam_rightT.xyz * ndc.x + L.cam_topT.xyz * ndc.y;
            vec3 wp  = L.eye_pos.xyz + ray * zview;

            vec3 Ng = normalize(cross(dFdx(wp), dFdy(wp)));
            // Floor puddles are BELOW the camera (we look DOWN at them). The
            // derivative normal's sign is ambiguous (don't flip it up — that put
            // water on ceilings), so gate by the VIEW ray: only surfaces we look
            // down at get water; ceilings (we look up) never do.
            float viewDownY = normalize(wp - L.eye_pos.xyz).y;   // < 0 = looking down
            float upface = smoothstep(0.72, 0.85, abs(Ng.y)) * step(viewDownY, 0.0);

            float wd  = simOn ? waterDepthAt(wp) : 0.0;
            float pud = simOn ? smoothstep(0.04, 0.12, wd) : puddleMask(wp.xz * 0.8);

            float ripFade = smoothstep(18.0, 8.0, zview)
                          * clamp(L.rain_params.x * 1.5 + 0.1, 0.0, 1.0);
            vec2 rip = rainRipples(wp.xz, L.sky_params.w * 0.6) * (0.3 * ripFade);
            vec3 Nw  = normalize(vec3(rip.x, 1.0, rip.y));
            vec3 V   = normalize(L.eye_pos.xyz - wp);

            // Volumetric absorption: murk the bottom by the water the view ray
            // crosses (depth / vertical view component). Deep water → can't see
            // the floor; the bottom is refracted by the surface ripples.
            if (simOn && wd > 0.04 && upface > 0.0) {
                float cosV    = max(V.y, 0.06);
                float pathLen = min(wd / cosV, 8.0);
                float murk    = 1.0 - exp(-pathLen * max(L.pom_params6.y, 0.01));
                vec2  refr    = rip * (L.pom_params6.z * (0.5 + wd));
                vec3  bottom  = textureLod(uHDR, sceneUV + refr, 0.0).rgb * exposure;
                vec3  murkCol = vec3(0.05, 0.10, 0.09) * (0.3 + 0.7 * avgLum * exposure);
                c = mix(c, mix(bottom, murkCol, murk), upface);
            }

            // SSR surface mirror — buildings/trees reflect in the water surface.
            float k0 = (simOn ? max(wet, 0.5) : wet) * pud * upface;
            if (k0 > 0.02) {
                k0 *= rainVisB(wp);
                vec3 R  = reflect(-V, Nw);
                float fres = pow(1.0 - clamp(dot(V, Nw), 0.0, 1.0), 2.0);
                float k = k0 * (0.25 + 0.75 * fres) * clamp(L.rain_params.w, 0.0, 1.5);
                // SSR is the rain perf hog (a depth march per water pixel — cost
                // scales with puddle coverage). Cull it to near the camera; far
                // water keeps the painted sky reflection from the world pass.
                if (k > 0.02 && R.y > 0.01 && zview < 35.0) {
                    vec3  p   = wp + R * 0.15;
                    float stp = 0.3;
                    vec2  hitUV = vec2(-1.0);
                    for (int i = 0; i < 16; ++i) {
                        p   += R * stp;
                        stp *= 1.22;
                        vec4 cp = L.scene_vp * vec4(p, 1.0);
                        if (cp.w <= 0.0) break;
                        vec3 pn = cp.xyz / cp.w;
                        vec2 puv = pn.xy * 0.5 + 0.5;
                        puv.y = 1.0 - puv.y;
                        if (puv.x < 0.0 || puv.x > 1.0 || puv.y < 0.0 || puv.y > 1.0 || pn.z >= 1.0)
                            break;
                        float d = textureLod(uDepth, puv, 0.0).r;
                        if (d < pn.z - 0.0003) {
                            // Thickness test in view depth: don't let rays
                            // passing far BEHIND a thin silhouette "hit" it.
                            float zHit = L.cam_rightT.w / (d    - L.cam_dir.w);
                            float zRay = L.cam_rightT.w / (pn.z - L.cam_dir.w);
                            if (zRay - zHit < 4.0) hitUV = puv;
                            break;
                        }
                    }
                    if (hitUV.x >= 0.0) {
                        vec3 refl = textureLod(uHDR, hitUV, 0.0).rgb * exposure;
                        vec2 ef = min(hitUV, 1.0 - hitUV);
                        float edge = clamp(min(ef.x, ef.y) * 8.0, 0.0, 1.0);
                        c = mix(c, refl, k * edge);
                    }
                }
            }
        }
    }

    // 2. Bloom (built exposure-scaled at quarter res, gaussian-blurred) added in
    //    linear HDR — the blend_soft equivalence.
    c += texture(uBloom, uv).rgb * pc.p1.w;

    // 3. CDL color grading in LINEAR space. R4's user build runs the USE_ACES
    //    path where the saved mod grading (e.g. Gunslinger's cg 0.515-green +
    //    saturation 1.3 in user.ltx) reads subtle; grading those same numbers
    //    in ACEScc LOG space (the non-ACES fallback) multiplies their leverage
    //    ~4× and tinted the whole world green. Linear CDL matches the ACES-path
    //    magnitude. Neutral at default console vars.
    c *= pc.p2.x;                                        // Slope (ssfx_exposure)
    c = pow(max(c, vec3(0.0)), pc.p3.xyz);               // Power (2*(1 - ssfx_color_grading))
    float luma = dot(c, LUM);
    c = luma + pc.p2.y * (c - luma);                     // Saturation (ssfx_saturation)
    c = max(c, vec3(0.0));

    // 4. Reinhard normalized to white = 11.2 (R4 tonemap_sRGB).
    float W = pc.p0.x;
    c = (c / (c + 1.0)) / (W / (W + 1.0));

    // 5. Gamma (img_corrections).
    c = pow(max(c, vec3(0.0)), vec3(pc.p2.z));

    outColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}
