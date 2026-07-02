#version 450
#extension GL_GOOGLE_include_directive : require

// Per-frame: clear the alive counter before gp_sim rebuilds aliveList.
// freeCount (counters[0]) is persistent and must NOT be touched here.

#include "gp_common.glsl"

layout(local_size_x = 1) in;

void main()
{
    counters[1] = 0u;
}
