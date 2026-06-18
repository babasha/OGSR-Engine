#version 450
// xrRenderVulkan — skinned NPC motion vectors (MV Phase 2a).
//
// Re-draws the animated skinned leaves into the motion-vector target, OVER the
// camera/static field the fullscreen pass laid down, so each NPC pixel carries
// its TRUE motion (camera + skeletal animation + the body moving through the
// world), not just camera reprojection.
//
// Skins the vertex TWICE: with THIS frame's bones (curBase) and the PREVIOUS
// frame's bones (prevBase) — both regions live simultaneously in the shared bone
// SSBO (kMaxBones × FRAMES_IN_FLIGHT; last frame's slot isn't overwritten this
// frame). Bones are pre-multiplied to world space (see skinned.vert), so S*pos is
// world; we project cur with curVP and prev with prevVP. The fragment turns the
// two clip positions into screen-space motion. Skinning decode is identical to
// skinned.vert (keep them in sync).

layout(location = 0) in vec4 a_Position;
layout(location = 1) in vec4 a_Normal;       // a = boneIdx(1W) / w0(2W+)
layout(location = 2) in vec4 a_TexCoordExt;  // zw = idx0,idx1 (44)
layout(location = 3) in vec4 a_Tangent;      // a = w1 (3W/4W)
layout(location = 4) in vec4 a_Binormal;     // a = idx2(3W) / w2(4W)
layout(location = 5) in vec4 a_BoneIndices;  // rgba = 4 idx (4W)

layout(location = 0) out vec4 vCurClip;
layout(location = 1) out vec4 vPrevClip;
layout(location = 2) out vec2 vUV;       // for the fragment alpha-test (cutout holes)

layout(push_constant) uniform PC {
    mat4 curVP;     // 0   this frame's view-proj
    mat4 prevVP;    // 64  previous frame's view-proj
    uint skinMode;  // 128 1=1W,2=2W,3=3W,4=4W
    uint curBase;   // 132 this frame's first bone slot
    uint prevBase;  // 136 previous frame's first bone slot (== curBase if the skeleton is new)
    uint boneCount; // 140 bone count (index clamp)
} pc;

layout(std430, set = 0, binding = 0) readonly buffer Bones { mat4 bones[]; };

uint dN(float a) { return uint(round(a * 255.0)); }
uint dF(float v) { return uint(round(abs(v)));    }
uint clampB(uint i) { return (pc.boneCount == 0u) ? 0u : min(i, pc.boneCount - 1u); }

// World-space skinning matrix for this vertex, rooted at bone-slot base `bb`.
mat4 skinMat(uint bb)
{
    uint mode = pc.skinMode & 15u;
    if (mode == 1u) {
        return bones[bb + clampB(dN(a_Normal.a))];
    } else if (mode == 2u) {
        float w0 = a_Normal.a;
        return bones[bb + clampB(dF(a_TexCoordExt.z))] * (1.0 - w0)
             + bones[bb + clampB(dF(a_TexCoordExt.w))] * w0;
    } else if (mode == 3u) {
        float w0 = a_Normal.a, w1 = a_Tangent.a;
        return bones[bb + clampB(dF(a_TexCoordExt.z))] * w0
             + bones[bb + clampB(dF(a_TexCoordExt.w))] * w1
             + bones[bb + clampB(dN(a_Binormal.a))]    * (1.0 - w0 - w1);
    } else {
        float w0 = a_Normal.a, w1 = a_Tangent.a, w2 = a_Binormal.a;
        return bones[bb + clampB(dN(a_BoneIndices.r))] * w0
             + bones[bb + clampB(dN(a_BoneIndices.g))] * w1
             + bones[bb + clampB(dN(a_BoneIndices.b))] * w2
             + bones[bb + clampB(dN(a_BoneIndices.a))] * (1.0 - w0 - w1 - w2);
    }
}

void main()
{
    vec3 pos = a_Position.xyz;
    vec4 wpCur  = skinMat(pc.curBase)  * vec4(pos, 1.0);   // world pos, this frame's pose
    vec4 wpPrev = skinMat(pc.prevBase) * vec4(pos, 1.0);   // world pos, previous pose
    vCurClip  = pc.curVP  * wpCur;
    vPrevClip = pc.prevVP * wpPrev;
    vUV         = a_TexCoordExt.xy;
    gl_Position = vCurClip;   // rasterize at the current position (matches the forward/prepass depth)
}
