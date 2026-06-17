#version 450
#extension GL_GOOGLE_include_directive : require
// xrRenderVulkan — VSM page rasterization (depth-only). Each instance of a caster
// is ONE page: gl_InstanceIndex -> casterPages[] -> physical slot -> pageList[] ->
// (level, page). World vertex -> light space -> page-local NDC, placed into the
// page's atlas sub-rect, and gl_ClipDistance clips to the page rect so geometry
// outside it doesn't bleed into neighbouring pages. Depth = global clipmap Z (so
// every page stores depth in the same space the receiver compares against).
#include "vsm_common.glsl"

layout(location = 0) in vec3 aPos;   // world-space position (statics are world-baked, offset 0 / FLOAT3)

layout(set = 0, binding = 0) readonly buffer PageList    { uvec4 pageList[]; };   // slot -> (level, px, py, _)
layout(set = 0, binding = 1) readonly buffer CasterPages { uint  casterPages[]; };// instance -> physical slot
layout(set = 0, binding = 2) uniform VsmParams {
    mat4 view;                 // world -> sun light space
    vec4 level[VSM_LEVELS];     // xy = level origin (light XY), z = extent (m)
    vec4 zparams;              // x = zNear, y = 1/(zFar-zNear)
} vsm;

out gl_PerVertex { vec4 gl_Position; float gl_ClipDistance[4]; };

void main()
{
    uint slot = casterPages[gl_InstanceIndex];
    if (slot >= uint(VSM_MAX_PHYS_S)) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }  // defensive: cull

    uvec4 pg    = pageList[slot];
    int   L     = int(pg.x);
    ivec2 page  = ivec2(pg.yz);

    vec3  lp = (vsm.view * vec4(aPos, 1.0)).xyz;     // light space

    vec2  origin = vsm.level[L].xy;
    float pw     = vsm.level[L].z / float(VSM_PAGES_AXIS);   // page world size
    vec2  pmin   = origin + vec2(page) * pw;
    vec2  pmax   = pmin + vec2(pw);

    vec2  nxy = (lp.xy - pmin) / pw * 2.0 - 1.0;     // page-local NDC [-1,1]
    float nz  = (lp.z - vsm.zparams.x) * vsm.zparams.y;  // global clipmap depth [0,1]

    // Clip to the page rect (no bleed into adjacent atlas sub-rects).
    gl_ClipDistance[0] = lp.x - pmin.x;
    gl_ClipDistance[1] = pmax.x - lp.x;
    gl_ClipDistance[2] = lp.y - pmin.y;
    gl_ClipDistance[3] = pmax.y - lp.y;

    // Place into the physical page's atlas sub-rect (slot -> grid cell). STATIC atlas grid.
    uint  ax = slot % uint(VSM_ATLAS_W_S);
    uint  ay = slot / uint(VSM_ATLAS_W_S);
    float halfX = 1.0 / float(VSM_ATLAS_W_S);
    float halfY = 1.0 / float(VSM_ATLAS_H_S);
    float cx = (float(ax) + 0.5) * 2.0 * halfX - 1.0;
    float cy = (float(ay) + 0.5) * 2.0 * halfY - 1.0;
    gl_Position = vec4(cx + nxy.x * halfX, cy + nxy.y * halfY, nz, 1.0);
}
