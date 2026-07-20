#version 450
// xrRenderVulkan — tree wind-sway motion vectors. Pairs with tree_motion.frag.
// Re-draws the visible trees into the MV target, reprojecting the CURRENT and the
// PREVIOUS wind-displaced pose (SSFX trunk sway + crown flutter) so swaying trees
// carry their true screen motion — the fullscreen depth-reconstruction MV only sees
// camera motion and misses the sway (trees ghost on wind under DLSS/TAA). The wind
// decode is IDENTICAL to tree.vert (keep in sync); the current pose must bit-match
// tree.vert so depth LEQUAL passes over the tree depth the forward pass wrote.

layout(location = 0) in vec3 aPos;   // local-space position
layout(location = 1) in vec2 aUV;    // SHORT2 SSCALED raw int16 → float

struct TreeInstance {
    mat4  xform;        // 64 B per-instance world transform
    float c_scale_hemi;
    float c_bias_hemi;
    uint  _p0;          // wind class (2=foliage, 1=trunk, 0=rigid)
    uint  _p1;
};
layout(set = 0, binding = 0, std430) readonly buffer XformBuf { TreeInstance inst[]; };

layout(push_constant) uniform PC {
    mat4  curVP;              // 0    this frame's view-proj, UNJITTERED (jitter-free MV)
    mat4  prevVP;             // 64   previous frame's view-proj, UNJITTERED
    float uvScale;            // 128  1/2048 tree UV quant
    float alphaRef;           // 132  (fragment)
    float jitterX;            // 136  this frame's sub-pixel jitter (D3D-NDC), re-applied to gl_Position
    float jitterY;            // 140
    vec4  wind_params;        // 144  cur (wind_direction, wind_velocity, _, crownH)
    vec4  wsetup_trees;       // 160  cur (branchSpeed, trunkSpeed, bend, minWindSpeed)
    vec4  wind_anim;          // 176  cur drift (xyz) + w = flutter amplitude
    vec4  wind_params_prev;   // 192  previous frame's wind params
    vec4  wsetup_trees_prev;  // 208  previous frame's wsetup
    vec4  wind_anim_prev;     // 224  previous frame's drift
} pc;

#include "ssfx_tree_wind.glsl"   // SSFX tree wind (s_waves @ set0 binding1)

layout(location = 0) out vec4 vCurClip;
layout(location = 1) out vec4 vPrevClip;
layout(location = 2) out vec2 vUV;

// World-space wind displacement — mirrors tree.vert exactly (phase = baseY).
vec3 windDisp(vec3 wp, float H, float tc_y, uint cls, float baseY,
              vec4 wparams, vec4 wsetup, vec4 wanim)
{
    float r    = -wparams.x + 1.57079;
    vec2  wdir = vec2(cos(r), sin(r));
    float spd  = max(wsetup.w, clamp(wparams.y * 0.001, 0.0, 1.0));
    return ssfxTreeWind(cls, wp, H, tc_y, wdir, spd, baseY,
                        wanim.xyz, wsetup.x, wsetup.y, wsetup.z, wanim.w, wparams.w);
}

void main()
{
    TreeInstance t = inst[gl_InstanceIndex];

    vec3  base  = (t.xform * vec4(aPos, 1.0)).xyz;   // world pos BEFORE wind
    float baseY = t.xform[3].y;                       // per-tree phase
    float H     = base.y - baseY;
    float tc_y  = aUV.y * pc.uvScale;
    uint  cls   = t._p0;

    // Current pose — exactly tree.vert's displacement → depth bit-matches.
    vec3 curWorld  = base + windDisp(base, H, tc_y, cls, baseY,
                                     pc.wind_params, pc.wsetup_trees, pc.wind_anim);
    // Previous pose — last frame's wind params + drift (the sway delta).
    vec3 prevWorld = base + windDisp(base, H, tc_y, cls, baseY,
                                     pc.wind_params_prev, pc.wsetup_trees_prev, pc.wind_anim_prev);

    vCurClip    = pc.curVP  * vec4(curWorld,  1.0);   // unjittered → clean MV
    vPrevClip   = pc.prevVP * vec4(prevWorld, 1.0);
    // Rasterize jittered (depth bit-match with the forward tree pass); MV stays jitter-free.
    gl_Position = vCurClip;
    gl_Position.xy += vec2(pc.jitterX, pc.jitterY) * gl_Position.w;
    vUV         = aUV * pc.uvScale;
}
