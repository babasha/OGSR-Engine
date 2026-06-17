#version 450
#extension GL_GOOGLE_include_directive : require
// xrRenderVulkan — VSM skinned-caster page rasterization (depth-only). Combines the
// GPU skinning of shadow_skinned.vert (bones pre-multiplied by the object world matrix
// → S*pos is world space) with the page routing of vsm_page.vert (each instance = one
// allocated atlas page the leaf overlaps: gl_InstanceIndex → casterPages[] → slot →
// pageList[] → page ortho + atlas sub-rect, gl_ClipDistance clips to the page). So NPCs
// cast into the same virtual shadow atlas the statics do. See vk_vsm.cpp.
#include "vsm_common.glsl"

// vertHW skinned attributes — MUST match BuildSkinnedVI (skinned.vert / shadow_skinned.vert).
layout(location = 0) in vec4 a_Position;
layout(location = 1) in vec4 a_Normal;
layout(location = 2) in vec4 a_TexCoordExt;
layout(location = 3) in vec4 a_Tangent;
layout(location = 4) in vec4 a_Binormal;
layout(location = 5) in vec4 a_BoneIndices;

layout(std430, set = 0, binding = 0) readonly buffer Bones { mat4 bones[]; };   // shared skinned bone SSBO (world-space)

layout(set = 1, binding = 0) readonly buffer PageList    { uvec4 pageList[]; };   // slot -> (level, px, py, _)
layout(set = 1, binding = 1) readonly buffer CasterPages { uint  casterPages[]; };// instance -> slot
layout(set = 1, binding = 2) uniform VsmParams {
    mat4 view;
    vec4 level[VSM_LEVELS];
    vec4 zparams;
} vsm;

layout(push_constant) uniform PC { uint skinMode; uint baseBone; uint boneCount; uint pad; } pc;

out gl_PerVertex { vec4 gl_Position; float gl_ClipDistance[4]; };

uint dN(float a) { return uint(round(a * 255.0)); }
uint dF(float v) { return uint(round(abs(v)));    }
uint clampB(uint i) { return (pc.boneCount == 0u) ? 0u : min(i, pc.boneCount - 1u); }

void main()
{
    // --- Skin → world position (same math as shadow_skinned.vert).
    vec3 pos = a_Position.xyz;
    uint bb  = pc.baseBone;
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
    vec3 wp = (S * vec4(pos, 1.0)).xyz;

    // --- Route to the atlas page (mirror vsm_page.vert).
    uint slot = casterPages[gl_InstanceIndex];
    if (slot >= uint(VSM_MAX_PHYS)) {            // defensive: cull degenerate instance
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        gl_ClipDistance[0] = gl_ClipDistance[1] = gl_ClipDistance[2] = gl_ClipDistance[3] = -1.0;
        return;
    }
    uvec4 pg   = pageList[slot];
    int   L    = int(pg.x);
    ivec2 page = ivec2(pg.yz);

    vec3  lp     = (vsm.view * vec4(wp, 1.0)).xyz;
    vec2  origin = vsm.level[L].xy;
    float pw     = vsm.level[L].z / float(VSM_PAGES_AXIS);
    vec2  pmin   = origin + vec2(page) * pw;
    vec2  pmax   = pmin + vec2(pw);
    vec2  nxy    = (lp.xy - pmin) / pw * 2.0 - 1.0;
    float nz     = (lp.z - vsm.zparams.x) * vsm.zparams.y;

    gl_ClipDistance[0] = lp.x - pmin.x;
    gl_ClipDistance[1] = pmax.x - lp.x;
    gl_ClipDistance[2] = lp.y - pmin.y;
    gl_ClipDistance[3] = pmax.y - lp.y;

    uint  ax = slot % uint(VSM_ATLAS_W);
    uint  ay = slot / uint(VSM_ATLAS_W);
    float halfX = 1.0 / float(VSM_ATLAS_W);
    float halfY = 1.0 / float(VSM_ATLAS_H);
    float cx = (float(ax) + 0.5) * 2.0 * halfX - 1.0;
    float cy = (float(ay) + 0.5) * 2.0 * halfY - 1.0;
    gl_Position = vec4(cx + nxy.x * halfX, cy + nxy.y * halfY, nz, 1.0);
}
