#version 450
// xrRenderVulkan — NORMAL G-buffer caster for SKINNED dynamics (NPCs).
// Same GPU skinning as shadow_skinned_at.vert (bit-identical positions → the
// LEQUAL depth test against the prepass depth passes only on the NPC's own
// visible surface), but ALSO skins the vertex normal and outputs it in
// WORLD space. Feeds GTAO real per-pixel normals instead of the noisy
// depth-derivative ones, killing the "dirt"/speckle on NPCs. The fragment
// alpha-tests (a < 0.25) so cutout coverage matches the depth prepass.

layout(location = 0) in vec4 a_Position;     // FLOAT4 xyz=pos
layout(location = 1) in vec4 a_Normal;       // u8x4 unorm: xyz=normal, a=boneIdx(1W)/w0(2W+)
layout(location = 2) in vec4 a_TexCoordExt;  // FLOAT2 (36/40) or FLOAT4 (44): zw=idx (44)
layout(location = 3) in vec4 a_Tangent;      // u8x4 unorm: a=w1 (3W/4W)
layout(location = 4) in vec4 a_Binormal;     // u8x4 unorm: a=idx2 (3W) / w2 (4W)
layout(location = 5) in vec4 a_BoneIndices;  // u8x4 unorm: 4 idx (4W only)

layout(location = 0) out vec2 vUV;
layout(location = 1) out vec3 vNormal;       // world-space

layout(push_constant) uniform PC {
    mat4  mvp;       // camera view·proj (bones are already world-space)
    uint  skinMode;  // 1=1W,2=2W,3=3W,4=4W
    uint  baseBone;
    uint  boneCount;
    float hudMode;   // unused here
} pc;

layout(std430, set = 0, binding = 0) readonly buffer Bones { mat4 bones[]; };

uint dN(float a) { return uint(round(a * 255.0)); }
uint dF(float v) { return uint(round(abs(v)));    }
uint clampB(uint i) { return (pc.boneCount == 0u) ? 0u : min(i, pc.boneCount - 1u); }

void main()
{
    vec3 pos = a_Position.xyz;
    uint bb = pc.baseBone;

    mat4 S;
    if (pc.skinMode == 1u) {
        S = bones[bb + clampB(dN(a_Normal.a))];
    } else if (pc.skinMode == 2u) {
        float w0 = a_Normal.a;
        S = bones[bb + clampB(dF(a_TexCoordExt.z))] * (1.0 - w0) + bones[bb + clampB(dF(a_TexCoordExt.w))] * w0;
    } else if (pc.skinMode == 3u) {
        float w0 = a_Normal.a, w1 = a_Tangent.a;
        S = bones[bb + clampB(dF(a_TexCoordExt.z))] * w0
          + bones[bb + clampB(dF(a_TexCoordExt.w))] * w1
          + bones[bb + clampB(dN(a_Binormal.a))]    * (1.0 - w0 - w1);
    } else {
        float w0 = a_Normal.a, w1 = a_Tangent.a, w2 = a_Binormal.a;
        S = bones[bb + clampB(dN(a_BoneIndices.r))] * w0
          + bones[bb + clampB(dN(a_BoneIndices.g))] * w1
          + bones[bb + clampB(dN(a_BoneIndices.b))] * w2
          + bones[bb + clampB(dN(a_BoneIndices.a))] * (1.0 - w0 - w1 - w2);
    }

    vec3 nrm = a_Normal.xyz * 2.0 - 1.0;     // object-space unit normal (same decode as skinned.vert)
    vNormal  = mat3(S) * nrm;                // bone-skinned → world space (normalized in the FS)
    vUV      = a_TexCoordExt.xy;
    gl_Position = pc.mvp * (S * vec4(pos, 1.0));
}
