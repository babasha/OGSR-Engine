#version 450
#extension GL_GOOGLE_include_directive : require
// ⛔ NO NET PERF GAIN on dGPU (2026-07-02) — shadow-HZB kept but DISABLED (r_vsm_hzb=0, dispatch gated
//    off by default). Correct, picture identical, just not a win here. See vk_console_min.cpp + memory.
// xrRenderVulkan — shadow-HZB reduce (r_vsm_hzb). One workgroup per STATIC-atlas physical slot;
// reduces that slot's 128x128 PRIOR-FRAME depth to a single MAX (the farthest nearest-to-light
// surface in the page). The tree/opaque caster bins then cull a (caster,page) pair whose NEAREST
// light-depth is farther than this max → the caster is fully behind the cached occluders (walls/
// terrain) and can't win the depth test → skip its raster (kills alpha-test overdraw = the VSMrender
// cost). CONSERVATIVE (MAX-reduce + only culls the strictly-behind) → shadow depth identical.
//
// Only DIRTY slots (being re-rendered this frame) with VALID prior depth (priorValid: the slot already
// held THIS world tile last frame — physTile matched, so its cached depth is the right occluder; a
// scrolled-in slot holds a different tile → invalid → left at 1.0 = no occlusion). Non-dirty/invalid
// slots keep the buffer's pre-cleared 1.0. Reads the atlas at MarkPages time = prior-frame content
// (SHADER_READ), before RenderAtlas overwrites the dirty pages. See vk_vsm.cpp DispatchHzbReduce.
#include "vsm_common.glsl"

layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0) uniform sampler2D atlas;                       // STATIC atlas depth (prior frame)
layout(set = 0, binding = 1) readonly buffer SlotDirty  { uint slotDirty[];  };  // (unused now; kept for layout parity)
layout(set = 0, binding = 2) readonly buffer PriorValid { uint priorValid[]; };
layout(set = 0, binding = 3) writeonly buffer PageMax   { float pageMax[];   };

shared float s_max[256];

void main()
{
    uint slot = gl_WorkGroupID.x;
    if (slot >= uint(VSM_MAX_PHYS_S)) return;
    // Every RESIDENT slot with valid cached depth (priorValid==1, cleared each frame → resident+!wrong)
    // produces an occluder max — not just dirty ones. Option A: near trees (DYNAMIC atlas) query the
    // STATIC occluder at their world page, which is often a cache-HIT (non-dirty) static page. Non-
    // resident/invalid slots keep the pre-cleared 1.0 (never occludes) → correct for the toroidal cache.
    if (priorValid[slot] == 0u) return;

    uint ax   = slot % uint(VSM_ATLAS_W_S);
    uint ay   = slot / uint(VSM_ATLAS_W_S);
    ivec2 org = ivec2(int(ax) * VSM_PAGE_SIZE, int(ay) * VSM_PAGE_SIZE);

    // 16x16 threads over a 128x128 page → each thread maxes an 8x8 block.
    float m = 0.0;
    ivec2 t0 = org + ivec2(int(gl_LocalInvocationID.x) * 8, int(gl_LocalInvocationID.y) * 8);
    for (int dy = 0; dy < 8; ++dy)
        for (int dx = 0; dx < 8; ++dx)
            m = max(m, texelFetch(atlas, t0 + ivec2(dx, dy), 0).r);

    uint li = gl_LocalInvocationID.y * 16u + gl_LocalInvocationID.x;
    s_max[li] = m;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1u) {
        if (li < s) s_max[li] = max(s_max[li], s_max[li + s]);
        barrier();
    }
    if (li == 0u) pageMax[slot] = s_max[0];
}
