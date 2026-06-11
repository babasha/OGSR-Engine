#version 450

// World pass — TERRAIN splatting fragment shader.
//
// Texture splatting (R4 deffer_terrain_high / CBlender_BmmD):
//   base  = s_base(uv)                       — terrain diffuse (e.g. terrain_escape)
//   mask  = s_mask(uv); mask /= dot(mask,1)  — RGBA splat weights, normalized
//   det   = dt_r*mask.r + dt_g*mask.g + dt_b*mask.b + dt_a*mask.a   — at detail UV
//   albedo = 2 * base * det                  — R4 "2*base*detail" convention
// Detail channels (CBlender_BmmD defaults): R grass, G asphalt, B earth, A gravel.
// Lightmap modulation identical to world_lmap. POM / puddles / detail-normals
// from the SSS shader are intentionally omitted (MVP — colour splatting only).

layout(set = 0, binding = 0) uniform sampler2D uBase;
layout(set = 0, binding = 1) uniform sampler2D uMask;
layout(set = 0, binding = 2) uniform sampler2D uDtR;   // grass
layout(set = 0, binding = 3) uniform sampler2D uDtG;   // asphalt
layout(set = 0, binding = 4) uniform sampler2D uDtB;   // earth
layout(set = 0, binding = 5) uniform sampler2D uDtA;   // gravel/yantar
layout(set = 0, binding = 6) uniform sampler2D uLmap;

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
    vec4 base = texture(uBase, vUV);
    if (pc.alphaRef >= 0.0 && base.a < pc.alphaRef) discard;

    // Splat mask — normalize so the 4 weights sum to 1. Empty/missing mask
    // (1×1 white fallback → sum 4) degrades to an even blend; an all-zero
    // mask falls back to pure grass so terrain never goes black.
    vec4 mask = texture(uMask, vUV);
    float wsum = dot(mask, vec4(1.0));
    mask = (wsum > 1e-4) ? (mask / wsum) : vec4(1.0, 0.0, 0.0, 0.0);

    vec3 dR = texture(uDtR, vDetailUV).rgb;
    vec3 dG = texture(uDtG, vDetailUV).rgb;
    vec3 dB = texture(uDtB, vDetailUV).rgb;
    vec3 dA = texture(uDtA, vDetailUV).rgb;
    vec3 detail = dR * mask.r + dG * mask.g + dB * mask.b + dA * mask.a;

    vec3 albedo = 2.0 * base.rgb * detail;

    // Lightmap (same convention as world_lmap.frag).
    vec4 lm = texture(uLmap, vLmapUV);
    const vec3 sunColour = vec3(1.0, 0.95, 0.82);
    const vec3 ambient   = vec3(0.32, 0.34, 0.40);
    vec3 lighting = lm.rgb * 1.8 + sunColour * lm.a * 0.7 + ambient;

    outColor = vec4(albedo * lighting, base.a);
}
