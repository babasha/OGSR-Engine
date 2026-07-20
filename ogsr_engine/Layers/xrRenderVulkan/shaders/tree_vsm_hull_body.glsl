// xrRenderVulkan — VSM crown-HULL caster VS BODY (r_vsm_tree_hull). Shared by two
// wrappers that pick the target atlas via TV_* macros (same pattern as
// tree_vsm_page_body.glsl):
//   tree_vsm_hull.vert.glsl    DYNAMIC atlas — near hull tier, live wind push
//   tree_vsm_hull_s.vert.glsl  STATIC atlas  — far hull tier (r_vsm_tree_hull 2),
//                              zero wind push → rigid (cached toroidal pages)
// The AC-Shadows-style shadow LOD: crowns beyond r_vsm_tree_hull_dist cast from baked
// opaque ellipsoid lobes instead of the alpha-tested crown mesh — no texture fetch, no
// discard, early-Z, ~100× fewer verts. Page routing is IDENTICAL to
// tree_vsm_page_body.glsl (gl_InstanceIndex → casterPages → slot; tree = index/cap);
// the hull rides the same casterPages slices the crown bin filled (vsm_hull_cmd.comp
// copies instanceCount/firstInstance verbatim). tc_y is a constant (no UVs) → uniform
// lobe motion, flow-map still varies per lobe.
#include "vsm_common.glsl"
#include "ssfx_tree_wind.glsl"

layout(location = 0) in vec3 aPos;

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
    vec4  wind_params;
    vec4  wsetup_trees;
    vec4  wind_anim;
} pc;

out gl_PerVertex { vec4 gl_Position; float gl_ClipDistance[4]; };

void main()
{
    uint entry = casterPages[gl_InstanceIndex];   // arena entry = (treeIdx<<13)|slot (vsm_tree_bin)
    uint slot  = entry & 0x1FFFu;
    if (slot >= uint(TV_MAX_PHYS)) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        gl_ClipDistance[0] = gl_ClipDistance[1] = gl_ClipDistance[2] = gl_ClipDistance[3] = -1.0;
        return;
    }
    uint treeIdx = entry >> 13u;
    mat4 X = inst[treeIdx].xform;
    vec3 wp = (X * vec4(aPos, 1.0)).xyz;
    {
        float baseY = X[3].y;
        float H     = wp.y - baseY;
        float r     = -pc.wind_params.x + 1.57079;
        vec2  wdir  = vec2(cos(r), sin(r));
        float spd   = max(pc.wsetup_trees.w, clamp(pc.wind_params.y * 0.001, 0.0, 1.0));
        wp += ssfxTreeWind(inst[treeIdx]._p0, wp, H, 0.35, wdir, spd, baseY,
                           pc.wind_anim.xyz, pc.wsetup_trees.x, pc.wsetup_trees.y,
                           pc.wsetup_trees.z, pc.wind_anim.w, pc.wind_params.w);
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

    uint  ax = slot % uint(TV_ATLAS_W);
    uint  ay = slot / uint(TV_ATLAS_W);
    float hX = 1.0 / float(TV_ATLAS_W);
    float hY = 1.0 / float(TV_ATLAS_H);
    float cx = (float(ax) + 0.5) * 2.0 * hX - 1.0;
    float cy = (float(ay) + 0.5) * 2.0 * hY - 1.0;
    gl_Position = vec4(cx + nxy.x * hX, cy + nxy.y * hY, nz, 1.0);
}
