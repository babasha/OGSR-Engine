#version 450
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

layout(local_size_x = 256) in;

// binding 0: per-tree metadata (read-only). std430 packs vec3+float into 16 B.
struct TreeMeta {
    vec3 sphere_P;     // world-space center
    float sphere_R;    // radius
    uint index_count;  // num_tris * 3
    uint ib_first;     // first index (element offset into IB pool)
    uint first_vertex; // base vertex (element offset into VB pool)
    uint _pad;
};
layout(set = 0, binding = 0, std430) readonly buffer MetaBuf {
    TreeMeta meshes[];
};

// binding 1: view frustum (6 normalized planes), refreshed each frame.
layout(set = 0, binding = 1, std140) uniform FrustumBlock {
    vec4 planes[6];    // (nx, ny, nz, d)
} frust;

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
    uint mesh_count;    // trees in this group
    uint _unused;       // (cascade slot in shadow path; unused here)
    uint mesh_offset;   // start index in metadata buffer for this group
    uint output_base;   // base index into indirect command buffer (group * maxGroupMesh)
    uint count_index;   // index into draw-count buffer (= group)
} pc;

bool FrustumTest(vec3 center, float radius)
{
    for (int i = 0; i < 6; ++i)
    {
        if (dot(frust.planes[i].xyz, center) + frust.planes[i].w < -radius)
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
    if (!FrustumTest(m.sphere_P, m.sphere_R))
        return;

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
