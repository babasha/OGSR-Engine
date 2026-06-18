#version 450
// xrRenderVulkan — froxel volumetric INTEGRATION (vk_volumetrics, P1).
//
// One thread per (x,y) froxel column. March front-to-back over Z, accumulating
// in-scatter and transmittance (Beer-Lambert). Each froxel stores the running
// accumulation up to its slice so the composite (folded into the tonemap) can
// sample at the scene pixel's depth: scene = scene*transmittance + inscatter.
//
//   T          = product of exp(-extinction * sliceLen)  (camera → this slice)
//   accum.rgb += sliceInScatter * T_before * sliceLen
//
// The slice world thickness comes from the SAME exp-Z mapping the inject used
// (near, log2(far/near)) so geometry depth ↔ froxel Z stays consistent.

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform Vol {
    vec4 camPos;
    vec4 camDir;
    vec4 camRightT;
    vec4 camTopT;
    vec4 sun_dir;
    vec4 sun_color;
    vec4 sky_ambient;
    mat4 sun_vp;
    mat4 sun_near_vp;
    mat4 sun_c1_vp;
    vec4 gridParams;   // x=dimX y=dimY z=dimZ
    vec4 zParams;      // x=near y=far z=log2(far/near)
    vec4 fog;
    vec4 fog2;
    mat4 rain_vp;      // (layout match with vol_inject; unused here)
    mat4 prevViewProj; // (layout match; unused here)
    vec4 prevCamPos;   // (layout match; unused here)
    vec4 prevCamDir;   // (layout match; unused here)
    vec4 temporal;     // (layout match; unused here)
    mat4 fog_shadow_vp; // (layout match; unused here)
} V;

layout(set = 0, binding = 1, rgba16f) uniform readonly  image3D uScatter;     // rgb=in-scatter, a=extinction
layout(set = 0, binding = 2, rgba16f) uniform writeonly image3D uIntegrated;  // rgb=accum, a=transmittance

void main()
{
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    int dimX = int(V.gridParams.x), dimY = int(V.gridParams.y), dimZ = int(V.gridParams.z);
    if (px.x >= dimX || px.y >= dimY) return;

    float nearZ = V.zParams.x, logFN = V.zParams.z;

    vec3  accum = vec3(0.0);
    float T     = 1.0;
    float invDimZ = 1.0 / float(dimZ);

    for (int z = 0; z < dimZ; ++z) {
        float zPrev = nearZ * exp2(logFN * float(z)       * invDimZ);
        float zNext = nearZ * exp2(logFN * float(z + 1)   * invDimZ);
        float sliceLen = zNext - zPrev;

        vec4  s   = imageLoad(uScatter, ivec3(px, z));
        float ext = max(s.a, 0.0);

        // In-scatter contributed by this slice, attenuated by transmittance to
        // its front face; then advance the transmittance through the slice.
        accum += s.rgb * T * sliceLen;
        T     *= exp(-ext * sliceLen);

        imageStore(uIntegrated, ivec3(px, z), vec4(accum, T));
    }
}
