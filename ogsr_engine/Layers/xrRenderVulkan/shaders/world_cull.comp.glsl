#version 450
// GPU-driven world forward culling. One invocation per static mesh: sphere vs the
// camera's 6 frustum planes → atomically append a VkDrawIndexedIndirectCommand
// into the mesh's material group region (region = group*maxGroupMesh + slot).
// Mirrors world geometry being world-space (no transform): the draw needs only
// indexCount/firstIndex/vertexOffset. Single dispatch routes all groups via the
// per-mesh group index (no per-group dispatch).
layout(local_size_x = 256) in;

struct Meta { vec4 sphere; uint indexCount; uint ibFirst; uint firstVertex; uint group; };
struct Cmd  { uint indexCount; uint instanceCount; uint firstIndex; int vertexOffset; uint firstInstance; };

layout(set = 0, binding = 0) readonly buffer Metas { Meta metas[]; };
layout(set = 0, binding = 1)          buffer Cmds  { Cmd  cmds[];  };
layout(set = 0, binding = 2)          buffer Count { uint counts[]; };

layout(push_constant) uniform PC {
    vec4 planes[6];
    uint numGroups;
    uint maxGroupMesh;   // region stride: each group's cmd region holds maxGroupMesh slots
    uint total;          // number of meshes
    uint _pad;
} pc;

void main()
{
    uint l = gl_GlobalInvocationID.x;
    if (l >= pc.total) return;

    Meta m = metas[l];
    vec3 c = m.sphere.xyz;
    float r = m.sphere.w;
    for (int i = 0; i < 6; ++i)
        if (dot(pc.planes[i].xyz, c) + pc.planes[i].w < -r) return;   // outside this plane

    uint g    = m.group;
    uint o    = atomicAdd(counts[g], 1u);
    uint base = g * pc.maxGroupMesh;
    cmds[base + o].indexCount    = m.indexCount;
    cmds[base + o].instanceCount = 1u;
    cmds[base + o].firstIndex    = m.ibFirst;
    cmds[base + o].vertexOffset  = int(m.firstVertex);
    cmds[base + o].firstInstance = 0u;
}
