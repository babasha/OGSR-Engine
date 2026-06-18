#version 450
#extension GL_GOOGLE_include_directive : require
// xrRenderVulkan — VSM grass-caster page rasterization (depth-only). Reuses the detail
// grass vertex inputs (binding 0 = mesh pos, binding 1 = per-instance transform rows,
// INSTANCE rate) and routes each instance into its bound atlas page (gl_InstanceIndex →
// grassSlot[] from vsm_grass_bin → pageList → page ortho + atlas sub-rect + clip). Wind
// is intentionally skipped — a static contact shadow is plenty for near grass. See vk_vsm.cpp.
#include "vsm_common.glsl"

layout(location = 0) in vec3 aPos;        // binding 0: grass mesh vertex (pos @0)
layout(location = 1) in vec2 aUV;         // binding 0: uv @12 (for the alpha-test cutout)
layout(location = 3) in vec4 aInstRow0;   // binding 1: instance transform rows (rate INSTANCE)
layout(location = 4) in vec4 aInstRow1;
layout(location = 5) in vec4 aInstRow2;

layout(location = 0) out vec2 vUV;

layout(set = 0, binding = 0) readonly buffer GrassSlot { uint grassSlot[]; };
layout(set = 0, binding = 1) readonly buffer PageList  { uvec4 pageList[]; };
layout(set = 0, binding = 2) uniform VsmParams {
    mat4 view;
    vec4 level[VSM_LEVELS];
    vec4 zparams;
} vsm;

layout(push_constant) uniform PC { uint instanceBase; } pc;   // type * sectionSize

out gl_PerVertex { vec4 gl_Position; float gl_ClipDistance[4]; };

void main()
{
    vUV = aUV;   // for the fragment alpha test (blade cutout, not the solid quad)
    uint slot = grassSlot[pc.instanceBase + gl_InstanceIndex];
    if (slot >= uint(VSM_MAX_PHYS)) {            // far / not-resident → cull
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        gl_ClipDistance[0] = gl_ClipDistance[1] = gl_ClipDistance[2] = gl_ClipDistance[3] = -1.0;
        return;
    }
    mat4x3 m = mat4x3(aInstRow0.xyz, aInstRow1.xyz, aInstRow2.xyz, vec3(aInstRow0.w, aInstRow1.w, aInstRow2.w));
    vec3 wp = m * vec4(aPos, 1.0);

    uvec4 pg   = pageList[slot];
    int   L    = int(pg.x);
    ivec2 page = ivec2(pg.yz);
    vec3  lp   = (vsm.view * vec4(wp, 1.0)).xyz;
    vec2  origin = vsm.level[L].xy;
    float pw     = vsm.level[L].z / float(VSM_PAGES_AXIS);
    vec2  pmin   = origin + vec2(page) * pw;
    vec2  pmax   = pmin + vec2(pw);
    vec2  nxy    = (lp.xy - pmin) / pw * 2.0 - 1.0;
    float nz     = (lp.z - vsm.zparams.x) * vsm.zparams.y;

    gl_ClipDistance[0] = lp.x - pmin.x;
    gl_ClipDistance[1] = pmax.x - lp.x;
    gl_ClipDistance[2] = lp.y - pmin.y;
    gl_ClipDistance[3] = pmax.y - lp.y;

    uint  ax = slot % uint(VSM_ATLAS_W);
    uint  ay = slot / uint(VSM_ATLAS_W);
    float hX = 1.0 / float(VSM_ATLAS_W);
    float hY = 1.0 / float(VSM_ATLAS_H);
    float cx = (float(ax) + 0.5) * 2.0 * hX - 1.0;
    float cy = (float(ay) + 0.5) * 2.0 * hY - 1.0;
    gl_Position = vec4(cx + nxy.x * hX, cy + nxy.y * hY, nz, 1.0);
}
