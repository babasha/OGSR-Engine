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

layout(push_constant) uniform PC {
    vec4 p0;   // x=whitePoint, y=topMipLOD, z=middleGray, w=lowLum
    vec4 p1;   // x=expMin, y=expMax, z=expComp, w=bloomIntensity
    vec4 p2;   // x=cdlSlope, y=cdlSaturation, z=invGamma, w=unused
    vec4 p3;   // xyz=cdlPower (2*(1-cg)), w=unused
} pc;

layout(location = 0) out vec4 outColor;

const vec3 LUM = vec3(0.2126, 0.7152, 0.0722);

void main()
{
    // 1. Whole-frame average luminance from the top mip → R4 auto-exposure.
    vec3  avg    = textureLod(uHDR, vec2(0.5), pc.p0.y).rgb;
    float avgLum = max(dot(avg, LUM), 1e-4);
    float exposure = clamp(pc.p0.z / (avgLum + pc.p0.w), pc.p1.x, pc.p1.y) * pc.p1.z;

    vec3 c = texelFetch(uHDR, ivec2(gl_FragCoord.xy), 0).rgb * exposure;

    // 2. Bloom (built exposure-scaled at quarter res, gaussian-blurred) added in
    //    linear HDR — the blend_soft equivalence.
    vec2 uv = gl_FragCoord.xy / vec2(textureSize(uHDR, 0));
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
