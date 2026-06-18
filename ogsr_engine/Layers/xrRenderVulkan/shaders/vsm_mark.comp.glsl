#version 450
#extension GL_GOOGLE_include_directive : require
// xrRenderVulkan — VSM page marking. One thread per scene pixel: reconstruct the
// world position from the prepass depth, map it into the sun clipmap, and mark the
// page it would sample as NEEDED. A unique-page counter (atomicOr-first-wins) gives
// a cheap "how many pages are visible this frame" for diagnostics. See vk_vsm.cpp.
#include "vsm_common.glsl"

layout(local_size_x = 8, local_size_y = 8) in;

layout(set = 0, binding = 0) uniform sampler2D uDepth;   // scene prepass depth (statics+AT)

layout(set = 0, binding = 1) uniform VsmParams {
    mat4 view;                 // world -> sun light space
    vec4 level[VSM_LEVELS];    // xy = level origin (light XY of texel 0,0), z = extent (m)
    vec4 zparams;              // x = light-space zNear, y = 1/(zFar-zNear)
} vsm;

layout(set = 0, binding = 2) buffer Needed  { uint needed[]; };   // per-level page flags
layout(set = 0, binding = 3) buffer Counter { uint uniquePages; }; // first-mark counter

layout(push_constant) uniform Push {
    mat4 invViewProj;   // clip -> world (inverse of the scene view*proj)
    vec4 screen;        // xy = pixel dims, zw = 1/dims
    uint markStep;      // 1 = full-res, 2 = half-res (4x fewer threads + atomicOr)
} pc;

void main()
{
    // Half-res (markStep=2): one thread per 2x2 block samples one pixel. Adjacent pixels almost
    // always map to the SAME clipmap page (128 virtual texels) so 1-of-4 sampling rarely misses a
    // page; the page-boundary cases are covered by neighbouring blocks.
    ivec2 px = ivec2(gl_GlobalInvocationID.xy) * int(pc.markStep);
    if (px.x >= int(pc.screen.x) || px.y >= int(pc.screen.y)) return;

    vec2  uv   = (vec2(px) + 0.5) * pc.screen.zw;
    float zndc = texture(uDepth, uv).r;
    if (zndc >= 0.99999) return;   // sky / cleared — no receiver here

    // Reconstruct world position (D3D ndc, y-up — matches ssao.frag).
    vec4 clip  = vec4(uv.x * 2.0 - 1.0, 1.0 - 2.0 * uv.y, zndc, 1.0);
    vec4 world = pc.invViewProj * clip;
    world.xyz /= world.w;

    // World -> sun light space, then pick the finest clipmap level/page.
    vec3 lp = (vsm.view * vec4(world.xyz, 1.0)).xyz;
    vec2 luv; ivec2 page;
    int  L = vsmSelect(lp.xy, vsm.level, luv, page);
    if (L < 0) return;

    uint idx  = uint(vsmPageIndex(L, page));
    uint prev = atomicOr(needed[idx], 1u);
    if (prev == 0u) atomicAdd(uniquePages, 1u);
}
