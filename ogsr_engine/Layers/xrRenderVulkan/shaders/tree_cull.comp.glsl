#version 450
#extension GL_GOOGLE_include_directive : require
// xrRenderVulkan — tree frustum-cull compute (Session B).
//
// One dispatch per TreeIndirectGroup. Each invocation tests one tree's
// bounding sphere against the view frustum; survivors atomically append a
// VkDrawIndexedIndirectCommand into the group's section of the indirect
// buffer and bump the group's draw-count. DrawTrees then issues one
// vkCmdDrawIndexedIndirectCount per group.
//
// Layout mirrors VK::GpuTreeMeta (32 B, vk_TreeManager.h). The C++ side
// packs sphere_P(vec3)+sphere_R(float) into 16 B, then 4×u32.

#include "hzb_test.glsl"   // HZB_SphereOccluded — shared with world_cull_hzb / lod_cull

layout(local_size_x = 256) in;

// binding 0: per-tree metadata (read-only). std430 packs vec3+float into 16 B.
struct TreeMeta {
    vec3 sphere_P;     // world-space center
    float sphere_R;    // radius
    uint index_count;  // num_tris * 3
    uint ib_first;     // first index (element offset into IB pool)
    uint first_vertex; // base vertex (element offset into VB pool)
    uint flags;        // bit 0 = TREE_FLOD_BACKED (vk_TreeManager.h)
};
#define TREE_FLOD_BACKED 1u
layout(set = 0, binding = 0, std430) readonly buffer MetaBuf {
    TreeMeta meshes[];
};

// binding 1 was a single-buffered frustum UBO — REMOVED. Written by the CPU each
// frame into shared mapped memory, it raced the in-flight frames: while rotating,
// the next frame's CPU write clobbered this frame's planes before the GPU cull read
// them, so the colour cull tested a DIFFERENT (rotated) frustum than the depth
// prepass → edge trees landed in the prepass depth but were culled from colour →
// black tree silhouettes at the screen edge (only on the iGPU, by frame pacing).
// The frustum now travels in PUSH CONSTANTS (recorded per dispatch → no aliasing).
// The descriptor still declares binding 1 (unused) to avoid touching the layout.

// binding 2: VkDrawIndexedIndirectCommand array (20 B each), write-only.
struct DrawCommand {
    uint indexCount;
    uint instanceCount;
    uint firstIndex;
    int  vertexOffset;
    uint firstInstance;
};
layout(set = 0, binding = 2, std430) writeonly buffer IndirectBuf {
    DrawCommand commands[];
} drawCmds;

// binding 3: per-group visible draw count, read-write (atomic append cursor).
layout(set = 0, binding = 3, std430) buffer CountBuf {
    uint counts[];
} drawCount;

layout(push_constant) uniform PC {
    vec4 planes[6];     // view frustum (6 normalized planes) — per-frame, race-free
    uint  mesh_count;   // trees in this group
    float cut_dist;     // r_tree_dist: drop FLOD-backed trees past this (<=0 = off)
    uint mesh_offset;   // start index in metadata buffer for this group
    uint output_base;   // base index into indirect command buffer (group * maxGroupMesh)
    uint count_index;   // index into draw-count buffer (= group)
    // 128 B EXACTLY = guaranteed push limit. eye_* are SCALARS on purpose: a vec3
    // here aligns to 16 B and would overflow. Mirrors VK::TreeCullPush field for
    // field (vk_TreeManager.h) — keep both in step.
    float eye_x;        // camera position — reference for the distance cut
    float eye_y;
    float eye_z;
    mat4  viewProj;     // @128 (mat4 is 16-B aligned; 128 is) — sphere -> clip for the HZB
    uint  hzb_on;       // 1 = run the Hi-Z occlusion test
    float hzb_focal;    // 1/tan(fovY/2) — true screen footprint of the sphere
    uint  _pad0;
    uint  _pad1;
} pc;

// Hi-Z pyramid (MAX depth, GENERAL layout), shared with grass/world. Bound to a
// 1x1 dummy when occlusion is off, so the descriptor is always valid.
layout(set = 0, binding = 4) uniform sampler2D u_HZB;

bool FrustumTest(vec3 center, float radius)
{
    // This GPU colour cull MUST be a SUPERSET of the depth-prepass cull, otherwise a
    // tree at the frustum boundary can land in the prepass depth (occluding the world
    // → black clear) yet be culled from the colour pass → black tree silhouettes at
    // the screen edge while rotating. The boundary decision diverges between GPUs at
    // FP precision (the 780M iGPU showed it; the dGPU did not). Two changes keep us a
    // superset of the prepass (vk_pass_world.cpp builds CFrustum with LRTB|FAR):
    //   1. SKIP the near plane (index 4) — the prepass has no near plane.
    //   2. Add a metre of slack — the CPU CFrustum and this GPU extraction are
    //      different code paths, so absorb any plane-equation difference.
    // Planes are normalized (vk_cull.h), so dot+w is metres → the margin is metres.
    // Plane order: 0=L 1=R 2=B 3=T 4=N 5=F.
    const float margin = 1.0;
    for (int i = 0; i < 6; ++i)
    {
        if (i == 4) continue;   // near plane — excluded to match the prepass set
        if (dot(pc.planes[i].xyz, center) + pc.planes[i].w < -radius - margin)
            return false;
    }
    return true;
}

void main()
{
    uint local = gl_GlobalInvocationID.x;
    if (local >= pc.mesh_count)
        return;

    TreeMeta m = meshes[pc.mesh_offset + local];
    if (m.sphere_R <= 0.0)
        return;

    // Distance cut (r_tree_dist) — ONLY for trees CLODManager replaces with a
    // billboard past kImposterMinDist. Without this the full mesh AND the imposter
    // both render out there; trees lacking the flag have no stand-in and are kept
    // at any range. Measured from the sphere's NEAR side (dist - R), which keeps
    // strictly more trees than the prepass twin in RenderDepth (it compares centre
    // distance) — that direction is mandatory: colour must be a SUPERSET of the
    // prepass or edge trees land in depth with nothing shading them (black
    // silhouettes, the same failure the frustum margin below guards against).
    if (pc.cut_dist > 0.0 && (m.flags & TREE_FLOD_BACKED) != 0u) {
        vec3 eye = vec3(pc.eye_x, pc.eye_y, pc.eye_z);
        if (distance(m.sphere_P, eye) - m.sphere_R > pc.cut_dist)
            return;
    }

    if (!FrustumTest(m.sphere_P, m.sphere_R))
        return;

    // ---- Hi-Z occlusion (r_tree_hzb): drop trees fully behind nearer geometry --
    // Safe against the black-silhouette failure mode WITHOUT a matching test in
    // the depth prepass, and the reason is worth stating: the pyramid is a MAX
    // reduction of THIS frame's finished prepass depth, which the trees are part
    // of. A tree visible in the prepass has its own depth inside its footprint, so
    // hzbDepth >= its near face and the test can never cull it. A tree culled here
    // therefore contributed NOTHING to the depth buffer — nothing is left unshaded.
    // (Same argument as world_cull_hzb.comp; alpha-tested gaps only ever raise the
    // max toward the background, i.e. they err toward keeping.)
    // The maths lives in hzb_test.glsl, shared with world_cull_hzb / lod_cull.
    // It used to be copied into each of them, and BOTH fixes it carries (the
    // focal-scale footprint, then the mip clamp + texel-space block) had to be
    // re-applied by hand per copy — which is exactly how this shader spent a
    // while false-culling against a mip that did not exist.
    if (pc.hzb_on != 0u
        && HZB_SphereOccluded(u_HZB, pc.viewProj, m.sphere_P, m.sphere_R,
                              vec3(pc.eye_x, pc.eye_y, pc.eye_z), pc.hzb_focal))
        return;   // fully behind the farthest surface in its footprint

    uint out_idx = atomicAdd(drawCount.counts[pc.count_index], 1);
    if (out_idx >= pc.mesh_count)   // never overruns the group section
    {
        atomicAdd(drawCount.counts[pc.count_index], -1);
        return;
    }

    uint ci = pc.output_base + out_idx;
    drawCmds.commands[ci].indexCount    = m.index_count;
    drawCmds.commands[ci].instanceCount = 1;
    drawCmds.commands[ci].firstIndex    = m.ib_first;
    drawCmds.commands[ci].vertexOffset  = int(m.first_vertex);
    drawCmds.commands[ci].firstInstance = pc.mesh_offset + local; // → gl_InstanceIndex in tree.vert
}
