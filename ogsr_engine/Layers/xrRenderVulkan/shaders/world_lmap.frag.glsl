#version 450

// World pass — lmap variant. Final colour = albedo × baked lightmap.
//
// Albedo path is identical to the unlit variant: R4-style detail
// modulation `2 * base * detail`. Lightmap modulates this by the
// pre-baked sun + hemi colour the level compiler stored. Sun mask
// (lm.a) is currently unused — proper sun integration needs runtime
// sun direction & colour from the env subsystem.
//
// Without env wiring, sampling the lightmap RGB and multiplying gets
// the bulk of the visual win: baked shadows, bounce, ambient gradients.

layout(set = 0, binding = 0) uniform sampler2D uTexDiffuse;
layout(set = 0, binding = 1) uniform sampler2D uTexDetail;
layout(set = 0, binding = 2) uniform sampler2D uTexLmap;

layout(push_constant) uniform PushConstants {
    mat4  mvp;
    vec2  uvScale;
    float alphaRef;
    float detailScale;
} pc;

layout(location = 0) in  vec2 vUV;
layout(location = 1) in  vec2 vDetailUV;
layout(location = 2) in  vec2 vLmapUV;
layout(location = 0) out vec4 outColor;

void main()
{
    vec4 base   = texture(uTexDiffuse, vUV);
    if (pc.alphaRef >= 0.0 && base.a < pc.alphaRef) discard;

    vec3 detail = texture(uTexDetail, vDetailUV).rgb;
    vec3 albedo = 2.0 * base.rgb * detail;

    // R4 lightmap convention:
    //   lm.rgb = directional/colored hemi (sky bounce baked per-texel)
    //   lm.a   = sun mask (1 = full sun, 0 = shadowed)
    //
    // R4 forward (paraphrasing deffer_base_lmh + accumulator):
    //   final = albedo * (hemi*sky_color + sun_mask*sun_color*NdotL + ambient)
    //
    // Without env subsystem we use fixed daylight-ish constants and skip
    // NdotL (no normal in shader yet). hemi multiplier > 1 because raw lm
    // values are ~0.2–0.6 in typical maps; multiplied by sky tint they
    // approach ~1 in fully open sky, dim correctly in occluded areas.
    // Ambient floor keeps deep shadow visible instead of pitch-black.
    vec4 lm = texture(uTexLmap, vLmapUV);

    const vec3 sunColour = vec3(1.0, 0.95, 0.82);
    const vec3 ambient   = vec3(0.32, 0.34, 0.40);
    vec3 lighting = lm.rgb * 1.8 + sunColour * lm.a * 0.7 + ambient;

    outColor = vec4(albedo * lighting, base.a);
}
