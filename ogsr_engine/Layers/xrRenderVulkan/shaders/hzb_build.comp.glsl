#version 450
// xrRenderVulkan - Hierarchical Z-Buffer build compute shader
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
// SPDX-License-Identifier: MIT
//
// Builds a mip chain for the HZB (Hierarchical Z-Buffer) used by the grass
// generation compute shader for occlusion culling.
// Each mip level contains the MAX depth of the 2x2 region from the source.
//
// Pass 0 (isFirstPass=1): reads from the scene depth buffer, writes HZB mip 0.
// Pass N (isFirstPass=0): reads HZB mip N-1 (selected via srcMip), writes mip N.
//
// Dispatched once per mip level with appropriate push constants. MAX reduction
// is conservative: an instance is occluded only when it is farther than the
// farthest surface in its region, so culling never drops visible grass.

layout(local_size_x = 8, local_size_y = 8) in;

// ============================================================================
// Push constants
// ============================================================================
layout(push_constant) uniform HZBBuildConstants
{
    ivec2 srcSize;      // Source mip dimensions
    ivec2 dstSize;      // Destination mip dimensions
    uint  srcMip;       // Source mip level (depth=0 on first pass)
    uint  dstMip;       // Destination mip level
    uint  isFirstPass;  // 1 = reading from depth buffer, 0 = reading from HZB
    uint  _pad;
} pc;

// ============================================================================
// Bindings
// ============================================================================
// Source: either the depth buffer (first pass) or the previous HZB mip.
layout(set = 0, binding = 0) uniform sampler2D uSrcDepth;

// Destination: current HZB mip level (storage image).
layout(set = 0, binding = 1, r32f) writeonly uniform image2D uDstMip;

// ============================================================================
// Main
// ============================================================================
void main()
{
    ivec2 dstCoord = ivec2(gl_GlobalInvocationID.xy);
    if (dstCoord.x >= pc.dstSize.x || dstCoord.y >= pc.dstSize.y)
        return;

    // Map dst pixel to its 2x2 region in the source mip.
    vec2 srcUV = (vec2(dstCoord) + 0.5) / vec2(pc.dstSize);

    vec2 texelSize = 1.0 / vec2(pc.srcSize);
    vec2 srcCenter = srcUV * vec2(pc.srcSize);

    // Read the 4 covered texels at the source mip level.
    float d00 = textureLod(uSrcDepth, (floor(srcCenter - 0.5) + 0.5)            * texelSize, float(pc.srcMip)).r;
    float d10 = textureLod(uSrcDepth, (floor(srcCenter - 0.5) + vec2(1.5, 0.5)) * texelSize, float(pc.srcMip)).r;
    float d01 = textureLod(uSrcDepth, (floor(srcCenter - 0.5) + vec2(0.5, 1.5)) * texelSize, float(pc.srcMip)).r;
    float d11 = textureLod(uSrcDepth, (floor(srcCenter - 0.5) + vec2(1.5, 1.5)) * texelSize, float(pc.srcMip)).r;

    // Conservative occlusion: keep the FARTHEST depth so an instance is culled
    // only when it is behind everything in the region.
    float maxDepth = max(max(d00, d10), max(d01, d11));

    imageStore(uDstMip, dstCoord, vec4(maxDepth, 0.0, 0.0, 0.0));
}
