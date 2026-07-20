#version 450
#extension GL_GOOGLE_include_directive : require
// xrRenderVulkan — crown-HULL debug overlay VS (r_vsm_tree_hull_debug). Draws the
// baked hull lobes of HULL-TIER trees translucently over the forward scene so the
// r_vsm_tree_hull_dist boundary is visible in-world: pink lobes = the tree casts
// its shadow from the hull, unmarked trees = full crown mesh. Rides the forward
// tree pass: same pipeline layout (TreeGfxPush already pushed, set0 xform+waves
// already bound); one direct draw per hull tree with firstInstance = tree index.
// Same ssfxTreeWind as the VSM hull caster (tc_y 0.35) so the lobes track the sway.

layout(location = 0) in vec3 aPos;   // hull VB — tight vec3
layout(location = 0) out vec3 vWorld;

struct TreeInstance { mat4 xform; float c_scale_hemi; float c_bias_hemi; uint _p0; uint _p1; };
layout(set = 0, binding = 0, std430) readonly buffer XformBuf { TreeInstance inst[]; };

layout(push_constant) uniform PC {
    mat4  mViewProj;
    float uvScale; float alphaRef; float _pad0; float _pad1;
    vec4  vSunColor;
    vec4  vHemiColor;
    vec4  wind_params;
    vec4  wsetup_trees;
    vec4  wind_anim;
} pc;

#include "ssfx_tree_wind.glsl"

void main()
{
    TreeInstance t = inst[gl_InstanceIndex];   // firstInstance = tree index
    vec4 worldPos = t.xform * vec4(aPos, 1.0);
    float baseY = t.xform[3].y;
    float H     = worldPos.y - baseY;
    float r     = -pc.wind_params.x + 1.57079;
    vec2  wdir  = vec2(cos(r), sin(r));
    float spd   = max(pc.wsetup_trees.w, clamp(pc.wind_params.y * 0.001, 0.0, 1.0));
    worldPos.xyz += ssfxTreeWind(t._p0, worldPos.xyz, H, 0.35, wdir, spd, baseY,
                                 pc.wind_anim.xyz, pc.wsetup_trees.x, pc.wsetup_trees.y,
                                 pc.wsetup_trees.z, pc.wind_anim.w, pc.wind_params.w);
    vWorld      = worldPos.xyz;   // face-shading normal from screen-space derivatives (FS)
    gl_Position = pc.mViewProj * worldPos;
}
