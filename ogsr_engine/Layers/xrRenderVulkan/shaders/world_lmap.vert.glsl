#version 450

// World pass — lmap variant. For X-Ray level statics with a baked
// lightmap (`tcOffset == 24` sub-layout). TC1 at offset 28 is the
// per-mesh-region lightmap UV (SHORT2, scale 1/32768 → unit range).
// The lightmap RGBA texture (set 0, binding 2) carries baked
// hemi/sun/bounce colour from the level compiler.

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec2 inUV_short;     // base UV  (SHORT2 SSCALED)
layout(location = 2) in vec4 inTangent;      // .a = du (sub-pixel U fraction)
layout(location = 3) in vec4 inBinormal;     // .a = dv (sub-pixel V fraction)
layout(location = 4) in vec2 inLmapUV_short; // lightmap UV (SHORT2 SSCALED)
layout(location = 5) in vec4 inNormal;       // D3DCOLOR @ 12 (BGRA in memory)

layout(push_constant) uniform PushConstants {
    mat4  mvp;
    vec2  uvScale;
    float alphaRef;
    float detailScale;
} pc;

layout(location = 0) out vec2 vUV;
layout(location = 1) out vec2 vDetailUV;
layout(location = 2) out vec2 vLmapUV;
layout(location = 3) out vec3 vWorldPos;   // level statics are world-space (identity model)
layout(location = 4) out vec3 vNormal;     // world-space normal (dynamic lights)

void main()
{
    gl_Position = pc.mvp * vec4(inPos, 1.0);
    vWorldPos   = inPos;
    // D3DCOLOR memory order is BGRA → real (x,y,z) = .bgr (same swizzle the
    // vlit baked colour uses); unpack from [0,1] to [-1,1].
    vNormal     = inNormal.bgr * 2.0 - 1.0;

    // Sub-pixel UV: 8 extra bits of fractional du/dv from packed tangent /
    // binormal alphas (raw SHORT2 + frac, then scale).
    vec2 uv     = inUV_short + vec2(inTangent.a, inBinormal.a);
    vUV         = uv * pc.uvScale;
    vDetailUV   = vUV * pc.detailScale;

    // Lightmap UV scale is 1/32768 (range ±1) — different from base UV's
    // 1/1024. Hard-coded; lightmap UV layout doesn't vary by stride.
    vLmapUV     = inLmapUV_short * (1.0 / 32768.0);
}
