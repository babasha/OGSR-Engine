#version 450
#extension GL_GOOGLE_include_directive : require
// xrRenderVulkan — VSM tree-caster page rasterization (depth-only, alpha-tested in frag).
// Trees are STATIC instanced geometry (transforms SSBO, level-load) → temporal-friendly,
// no ghosting. Mirrors tree_depth.vert's transform, but each instance is one atlas PAGE:
// gl_InstanceIndex → casterPages[] (from vsm_skinned_bin, reused) → slot; the tree index
// = gl_InstanceIndex / cap picks the per-tree xform. Then route to the page (ortho + atlas
// sub-rect + clip), same as vsm_page.vert. See vk_TreeManager_Render.cpp.
#include "vsm_common.glsl"
#include "ssfx_tree_wind.glsl"   // TEST (r_vsm_tree_wind): same wind as the forward tree; s_waves @ set0 b1

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aUV;    // SHORT2 SSCALED

struct TreeInstance { mat4 xform; float c_scale_hemi; float c_bias_hemi; uint _p0; uint _p1; };
layout(set = 0, binding = 0, std430) readonly buffer XformBuf { TreeInstance inst[]; };

layout(set = 2, binding = 0) readonly buffer PageList    { uvec4 pageList[]; };
layout(set = 2, binding = 1) readonly buffer CasterPages { uint  casterPages[]; };
layout(set = 2, binding = 2) uniform VsmParams {
    mat4 view;
    vec4 level[VSM_LEVELS];
    vec4 zparams;
} vsm;

layout(push_constant) uniform PC {
    float uvScale; float alphaRef; uint cap; uint pad;
    vec4  wind_params;   // TEST wind (r_vsm_tree_wind): all 0 when off → no displacement
    vec4  wsetup_trees;
    vec4  wind_anim;
} pc;

layout(location = 0) out vec2 vUV;
out gl_PerVertex { vec4 gl_Position; float gl_ClipDistance[4]; };

void main()
{
    vUV = aUV * pc.uvScale;
    uint slot = casterPages[gl_InstanceIndex];
    if (slot >= uint(VSM_MAX_PHYS_S)) {   // trees now live in the STATIC (toroidal-cached) atlas
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        gl_ClipDistance[0] = gl_ClipDistance[1] = gl_ClipDistance[2] = gl_ClipDistance[3] = -1.0;
        return;
    }
    uint treeIdx = gl_InstanceIndex / pc.cap;
    mat4 X = inst[treeIdx].xform;
    vec3 wp = (X * vec4(aPos, 1.0)).xyz;
    // TEST wind (r_vsm_tree_wind): same world-space displacement as the forward
    // tree (ssfx_tree_wind). Off → wsetup/anim are 0 → ssfxTreeWind returns 0.
    {
        float baseY = X[3].y;
        float H     = wp.y - baseY;
        float r     = -pc.wind_params.x + 1.57079;
        vec2  wdir  = vec2(cos(r), sin(r));
        float spd   = max(pc.wsetup_trees.w, clamp(pc.wind_params.y * 0.001, 0.0, 1.0));
        wp += ssfxTreeWind(inst[treeIdx]._p0, wp, H, vUV.y, wdir, spd, baseY,
                           pc.wind_anim.xyz, pc.wsetup_trees.x, pc.wsetup_trees.y,
                           pc.wsetup_trees.z, pc.wind_anim.w);
    }

    uvec4 pg   = pageList[slot];
    int   L    = int(pg.x);
    ivec2 page = ivec2(pg.yz);
    vec3  lp   = (vsm.view * vec4(wp, 1.0)).xyz;
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

    uint  ax = slot % uint(VSM_ATLAS_W_S);
    uint  ay = slot / uint(VSM_ATLAS_W_S);
    float hX = 1.0 / float(VSM_ATLAS_W_S);
    float hY = 1.0 / float(VSM_ATLAS_H_S);
    float cx = (float(ax) + 0.5) * 2.0 * hX - 1.0;
    float cy = (float(ay) + 0.5) * 2.0 * hY - 1.0;
    gl_Position = vec4(cx + nxy.x * hX, cy + nxy.y * hY, nz, 1.0);
}
