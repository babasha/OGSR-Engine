#version 450
// ============================================================================
// skinned.vert — FORWARD GPU-skinning vertex shader for OGSR Vulkan (Step B/2).
// Bone math + vertHW attribute layout ported from the monolith's
// gbuffer_skinned.vert (monolith is deferred; this is the forward adaptation:
// no motion vectors, single MVP push, flat output for skinned.frag).
//
// vertHW formats (vk_Visual.h), per skinMode:
//   1W (stride 36): P FLOAT4 | N_I u8x4 (a=boneIdx) | T u8x4 | B u8x4 | tc FLOAT2
//   2W (stride 44): P | N(a=w0) | T | B | tc_i FLOAT4 (zw = idx0,idx1 as float)
//   3W (stride 44): P | N(a=w0) | T(a=w1) | B(a=idx2) | tc_i FLOAT4 (zw = idx0,idx1)
//   4W (stride 40): P | N(a=w0) | T(a=w1) | B(a=w2) | tc FLOAT2 | indices u8x4 (rgba=idx)
// Bone index stored RAW (0..BoneCount-1) -> direct index into bones[].
// bones[] holds the Fmatrix render transforms written row-major; GLSL reads them
// column-major, which transposes, so `bones[i] * v` == X-Ray's `v * Mbone`.
// Bones are PRE-MULTIPLIED by the object's world matrix at upload, so S*pos is
// WORLD-space: pc.mvp is the plain viewProj, v_wpos feeds the shadow lookup,
// and normals carry the object rotation.
// ============================================================================

layout(location = 0) in vec4 a_Position;     // FLOAT4 xyz=pos
layout(location = 1) in vec4 a_Normal;       // u8x4 unorm: xyz=normal, a=boneIdx(1W)/w0(2W+)
layout(location = 2) in vec4 a_TexCoordExt;  // FLOAT2 (36/40) or FLOAT4 (44): xy=uv, zw=idx (44)
layout(location = 3) in vec4 a_Tangent;      // u8x4 unorm: a=w1 (3W/4W)
layout(location = 4) in vec4 a_Binormal;     // u8x4 unorm: a=idx2*3 (3W) / w2 (4W)
layout(location = 5) in vec4 a_BoneIndices;  // u8x4 unorm: 4 idx*3 (4W only)

layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec3 v_nrm;
layout(location = 2) out vec3 v_wpos;   // world-space position (shadow lookup)

layout(push_constant) uniform PC {
    mat4  mvp;       // offset 0:  world->clip (bones are pre-multiplied to world)
    uint  skinMode;  // offset 64: 1=1W,2=2W,3=3W,4=4W
    uint  baseBone;  // offset 68: this skeleton's first bone slot in bones[]
    uint  boneCount; // offset 72: this skeleton's bone count (for index clamp)
    float hudMode;   // offset 76: 1 = first-person HUD (read by the fragment shader)
} pc;

// All skeletons' bone matrices concatenated; this skeleton's start = pc.baseBone.
layout(std430, set = 0, binding = 0) readonly buffer Bones { mat4 bones[]; };

// Bone index decode. Indices are now stored RAW (0..BoneCount-1) — direct SSBO
// index, no legacy "*3" matrix-row stride (which overflowed the u8 channels for
// bones >85). dN: from u8-normalized alpha; dF: from a float field.
uint dN(float a) { return uint(round(a * 255.0)); }
uint dF(float v) { return uint(round(abs(v)));    }

// Clamp a decoded local bone index to [0, boneCount). An out-of-range index would
// read an unwritten (zero) SSBO slot -> S becomes a zero matrix -> w=0 -> the vertex
// shoots to infinity (thin stray spike). Clamping pins it to a valid bone instead.
uint clampB(uint i) { return (pc.boneCount == 0u) ? 0u : min(i, pc.boneCount - 1u); }

void main()
{
    vec3 pos = a_Position.xyz;
    vec3 nrm = a_Normal.xyz * 2.0 - 1.0;
    uint bb = pc.baseBone;

    mat4 S;
    if (pc.skinMode == 1u) {
        S = bones[bb + clampB(dN(a_Normal.a))];
    } else if (pc.skinMode == 2u) {
        // R4 (FSkinned vertHW_2W::get_pos_bones): lerp(boneA, boneB, w) =
        // boneA*(1-w) + boneB*w, with w stored in N.a and A=matrix0, B=matrix1.
        // So matrix0 gets (1-w), matrix1 gets w — NOT the other way round.
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

    vec4 sp = S * vec4(pos, 1.0);    // world-space (bones pre-multiplied)
    gl_Position = pc.mvp * sp;
    v_uv   = a_TexCoordExt.xy;
    v_nrm  = normalize(mat3(S) * nrm);
    v_wpos = sp.xyz;
}
