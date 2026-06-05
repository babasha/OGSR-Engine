#version 450

// World pass — vert-lit variant. Final colour = albedo × pre-baked
// vertex lighting + small ambient floor.
//
// vBakedColor is what the level compiler computed offline (point-light
// contributions + bounce). It does NOT include direct sun — that needs
// runtime sun direction, deferred until env subsystem lands. As an
// approximation we add a constant ambient term and a sun-direction-
// independent fake-sun term gated by the per-vertex sun mask.
//
// Binding 2 (lmap) is bound for descriptor-layout compatibility but
// not sampled here — vert-lit materials don't have a lightmap.

layout(set = 0, binding = 0) uniform sampler2D uTexDiffuse;
layout(set = 0, binding = 1) uniform sampler2D uTexDetail;
layout(set = 0, binding = 2) uniform sampler2D uTexLmap;  // unused (white fallback)

layout(push_constant) uniform PushConstants {
    mat4  mvp;
    vec2  uvScale;
    float alphaRef;
    float detailScale;
} pc;

layout(location = 0) in  vec2  vUV;
layout(location = 1) in  vec2  vDetailUV;
layout(location = 2) in  vec3  vBakedColor;
layout(location = 3) in  float vSunMask;
layout(location = 0) out vec4  outColor;

void main()
{
    vec4 base   = texture(uTexDiffuse, vUV);
    if (pc.alphaRef >= 0.0 && base.a < pc.alphaRef) discard;

    vec3 detail = texture(uTexDetail, vDetailUV).rgb;
    vec3 albedo = 2.0 * base.rgb * detail;

    // Lighting: baked vertex color + ambient floor + fake sun.
    // vBakedColor is point-lights + bounce only (sun is in vSunMask),
    // typical values 0.05–0.4. Without env wiring we add a daylight-tinted
    // ambient and a sun-mask-driven contribution so daytime areas don't
    // look uniformly dim. Replace with `c0 = baked + hemi*sky + sun*NdotL`
    // when env values are plumbed.
    const vec3 sunColour = vec3(1.0, 0.95, 0.82);
    const vec3 ambient   = vec3(0.32, 0.34, 0.40);
    vec3 lighting = vBakedColor * 1.5 + sunColour * vSunMask * 0.7 + ambient;

    outColor = vec4(albedo * lighting, base.a);
}
