#version 450
// xrRenderVulkan — tree shadow caster (depth-only, alpha-tested in frag).
// Same instancing as tree.vert, but pc.mViewProj is the LIGHT view·proj. Applies
// the SAME SSFX wind displacement as tree.vert (world-space, VP-independent) so
// the shadow tracks the swaying geometry — a rigid shadow on a bent trunk shows
// a black "untextured" self-shadow band.

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aUV;    // SHORT2 SSCALED

struct TreeInstance {
    mat4  xform;
    float c_scale_hemi;
    float c_bias_hemi;
    uint  _p0;          // wind class (2=foliage, 1=trunk, 0=rigid)
    uint  _p1;
};
layout(set = 0, binding = 0, std430) readonly buffer XformBuf {
    TreeInstance inst[];
};

// Full TreeGfxPush layout (must match vk_TreeManager.h) — the depth pass reuses
// it so the wind fields line up with the forward push.
layout(push_constant) uniform PC {
    mat4  mViewProj;    // light view·proj
    float uvScale;      // 1/2048
    float alphaRef;
    float _pad0;
    float _pad1;
    vec4  vSunColor;
    vec4  vHemiColor;
    vec4  wind_params;  // (wind_direction, wind_velocity, _, _)
    vec4  wsetup_trees; // (branchSpeed, trunkSpeed, bend, minWindSpeed)
    vec4  wind_anim;    // (drift.xyz, flutterAmp)
} pc;

#include "ssfx_tree_wind.glsl"

layout(location = 0) out vec2 vUV;

void main()
{
    TreeInstance t = inst[gl_InstanceIndex];
    vec4 worldPos = t.xform * vec4(aPos, 1.0);

    // Same wind as the forward pass (world-space displacement).
    float baseY = t.xform[3].y;
    float H     = worldPos.y - baseY;
    float r     = -pc.wind_params.x + 1.57079;
    vec2  wdir  = vec2(cos(r), sin(r));
    float spd   = max(pc.wsetup_trees.w, clamp(pc.wind_params.y * 0.001, 0.0, 1.0));
    float tc_y  = aUV.y * pc.uvScale;
    worldPos.xyz += ssfxTreeWind(t._p0, worldPos.xyz, H, tc_y, wdir, spd, baseY,
                                 pc.wind_anim.xyz, pc.wsetup_trees.x, pc.wsetup_trees.y,
                                 pc.wsetup_trees.z, pc.wind_anim.w, pc.wind_params.w);

    gl_Position = pc.mViewProj * worldPos;
    vUV = aUV * pc.uvScale;
}
