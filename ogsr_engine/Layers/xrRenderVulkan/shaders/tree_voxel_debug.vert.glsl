#version 450
#extension GL_GOOGLE_include_directive : require
// xrRenderVulkan — crown VOXEL-cloud viewmode VS (r_vsm_tree_hull_debug with
// r_vsm_tree_hull_vox > 0). The UE Nanite-foliage / "Witcher 4 demo" look: each
// occupied crown cell is an INDIVIDUAL small colored cube, drawn instanced —
// vkCmdDraw(36, voxelCount, 0, firstVoxel) with the voxel stream bound at
// per-INSTANCE rate (center + packed color), the cube's 36 corners generated
// from gl_VertexIndex. The owning tree's index arrives per-draw in pc.treeIdx
// (the spare TreeGfxPush pad), the level's voxel edge in pc.voxSize.
// Wind is applied ONCE at the voxel center so the whole cube sways rigidly,
// same ssfxTreeWind as the crown mesh (tc_y 0.35) — the cloud tracks the sway.

layout(location = 0) in vec3 aCenter;   // per-instance: voxel center, species-local
layout(location = 1) in uint aColor;    // per-instance: R|G<<8|B<<16 | (jitter4|coverage4)<<24

layout(location = 0) out vec3 vWorld;
layout(location = 1) out vec3 vColor;

struct TreeInstance { mat4 xform; float c_scale_hemi; float c_bias_hemi; uint _p0; uint _p1; };
layout(set = 0, binding = 0, std430) readonly buffer XformBuf { TreeInstance inst[]; };

layout(push_constant) uniform PC {
    mat4  mViewProj;
    float fade; float density; uint treeIdx; float voxSize;   // repurposed uvScale/alphaRef/_pad slots
    vec4  vSunColor;
    vec4  vHemiColor;
    vec4  wind_params;
    vec4  wsetup_trees;
    vec4  wind_anim;
} pc;

#include "ssfx_tree_wind.glsl"

const vec3 kCorner[8] = vec3[8](
    vec3(-0.5, -0.5, -0.5), vec3(0.5, -0.5, -0.5), vec3(0.5, 0.5, -0.5), vec3(-0.5, 0.5, -0.5),
    vec3(-0.5, -0.5,  0.5), vec3(0.5, -0.5,  0.5), vec3(0.5, 0.5,  0.5), vec3(-0.5, 0.5,  0.5));
const int kIdx[36] = int[36](
    0, 2, 1,  0, 3, 2,    // -Z
    4, 5, 6,  4, 6, 7,    // +Z
    0, 1, 5,  0, 5, 4,    // -Y
    3, 6, 2,  3, 7, 6,    // +Y
    0, 4, 7,  0, 7, 3,    // -X
    1, 2, 6,  1, 6, 5);   // +X

void main()
{
    // CONTINUOUS density thinning (UE grows voxels per DAG level with ~4x fewer per
    // level; we do it smoothly): as the rendered cube outgrows this grid's cell, a
    // matching fraction of cubes is culled — pc.density = (cell/cube)², so the crown's
    // covered area stays constant (n·T² = const) and the cube COUNT is continuous in
    // distance, converging to 1-2 giant blocks for far trees instead of a pile of
    // overlapping cubes. The kill decision hashes the STABLE voxel index (golden-ratio
    // low-discrepancy), so cubes vanish one-by-one, never reshuffling.
    if (pc.density < 1.0) {
        float h = fract(float(gl_InstanceIndex) * 0.6180339887);
        if (h >= pc.density) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); vWorld = vec3(0.0); vColor = vec3(0.0); return; }
    }
    TreeInstance t = inst[pc.treeIdx];
    vec4 c4 = t.xform * vec4(aCenter, 1.0);
    float baseY = t.xform[3].y;
    float H     = c4.y - baseY;
    float r     = -pc.wind_params.x + 1.57079;
    vec2  wdir  = vec2(cos(r), sin(r));
    float spd   = max(pc.wsetup_trees.w, clamp(pc.wind_params.y * 0.001, 0.0, 1.0));
    vec3 center = c4.xyz + ssfxTreeWind(t._p0, c4.xyz, H, 0.35, wdir, spd, baseY,
                                        pc.wind_anim.xyz, pc.wsetup_trees.x, pc.wsetup_trees.y,
                                        pc.wsetup_trees.z, pc.wind_anim.w, pc.wind_params.w);
    // Instance scale (uniform in tree xforms) scales the voxel edge; per-voxel size
    // jitter (baked hash HIGH nibble — LOW nibble is leaf coverage for the caster's
    // ordered sparsification, 0.75–1.2) breaks the minecraft-regular grid reading.
    float sc   = length(t.xform[0].xyz);
    float jit  = 0.75 + 0.45 * float((aColor >> 28) & 0xFu) / 15.0;
    vec3 corner = center + kCorner[kIdx[gl_VertexIndex]] * (pc.voxSize * sc * jit);
    vWorld      = corner;   // face-shading normal from screen-space derivatives (FS)
    vColor      = unpackUnorm4x8(aColor).rgb;
    gl_Position = pc.mViewProj * vec4(corner, 1.0);
}
