#version 450
#extension GL_GOOGLE_include_directive : require
// xrRenderVulkan - Sky specular IBL: resample the blended weather sky cubes into
// mip 0 of an RGBA16F "specular" cube. The roughness-mip chain is then produced
// by a box blit chain (vk_ibl.cpp) - mip N = rougher reflection. v1 is a straight
// resample + box mips (fast, seam-tolerant for a low-freq sky); the clean upgrade
// is a per-mip GGX importance-sample convolution (see the note in vk_ibl.cpp).
//
// The output cube is a TRUE WORLD-SPACE probe: each output texel's world
// direction is resolved through the SAME half-cube remap + sky_rotation the dome
// draw uses (sky_halfcube.glsl), so the probe and the visible sky agree texel for
// texel. That makes this cube the one place the sky is "unwrapped", and every
// consumer downstream — the specular reflection AND the diffuse SH projection —
// gets to index it with a plain world vector.
//
// This used to sample RAW, on the theory that "the half-cube remap is a draw-only
// thing; the light probe is raw". It is not draw-only: it is the mapping the
// cubemap is AUTHORED in. Sampling raw put the probe on a different sky than the
// one on screen — an up-facing normal read the cool zenith while the horizon band
// holding the entire sunset was unreachable, and sky_rotation drifted the probe's
// azimuth away from the dome's.
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#include "sky_halfcube.glsl"

layout(set = 0, binding = 0) uniform samplerCube uSky0;   // weather A
layout(set = 0, binding = 1) uniform samplerCube uSky1;   // weather B
// 2D-array storage view over the cube's 6 layers (mip 0). Writing cube faces as
// an array is the universally-supported path — a CUBE storage view is not.
layout(set = 0, binding = 2, rgba16f) uniform writeonly image2DArray uOut;  // spec cube mip 0 (6 layers)

layout(push_constant) uniform Push {
    vec4 p;   // x = cross-fade weight (A->B), y = face size (px), z = sky_rotation (rad), w unused
} pc;

// Cube face (Vulkan layer order: 0=+X 1=-X 2=+Y 3=-Y 4=+Z 5=-Z) + in-face uv -> dir.
vec3 faceDir(int face, vec2 uv)
{
    vec2 c = uv * 2.0 - 1.0;   // [-1,1]
    if (face == 0) return normalize(vec3( 1.0, -c.y, -c.x));
    if (face == 1) return normalize(vec3(-1.0, -c.y,  c.x));
    if (face == 2) return normalize(vec3( c.x,  1.0,  c.y));
    if (face == 3) return normalize(vec3( c.x, -1.0, -c.y));
    if (face == 4) return normalize(vec3( c.x, -c.y,  1.0));
    return               normalize(vec3(-c.x, -c.y, -1.0));   // 5 = -Z
}

void main()
{
    ivec3 gid  = ivec3(gl_GlobalInvocationID);
    int   size = int(pc.p.y);
    if (gid.x >= size || gid.y >= size) return;

    vec2 uv  = (vec2(gid.xy) + 0.5) / float(size);
    vec3 dir = faceDir(gid.z, uv);                       // WORLD direction of this texel

    // ...resolved to the cube texel the dome would show in that direction.
    vec3 sdir = SkySampleDir(dir, pc.p.z);

    float xf = clamp(pc.p.x, 0.0, 1.0);
    vec3  a  = textureLod(uSky0, sdir, 0.0).rgb;
    vec3  col = (xf > 0.01) ? mix(a, textureLod(uSky1, sdir, 0.0).rgb, xf) : a;

    imageStore(uOut, gid, vec4(col, 1.0));
}
