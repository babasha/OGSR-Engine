#version 450
#extension GL_GOOGLE_include_directive : require

// Phase 1 build: fill the VkDrawIndirectCommand from the alive count.
// Run by a single thread after gp_sim, with a barrier before the indirect read.

#include "gp_common.glsl"

layout(local_size_x = 1) in;

void main()
{
    // VkDrawIndirectCommand: vertexCount, instanceCount, firstVertex, firstInstance.
    drawCmd[0] = 6u;            // 2 triangles per billboard
    drawCmd[1] = counters[1];   // one instance per alive particle
    drawCmd[2] = 0u;
    drawCmd[3] = 0u;
}
