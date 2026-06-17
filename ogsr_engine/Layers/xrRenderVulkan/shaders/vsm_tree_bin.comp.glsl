#version 450
#extension GL_GOOGLE_include_directive : require
// xrRenderVulkan — VSM TREE-caster binning into the STATIC (toroidal-cached) atlas.
// Same as vsm_skinned_bin.comp (one thread per tree: find the allocated clipmap pages
// its world sphere overlaps, emit ONE indexed-indirect draw, instanceCount = page count),
// but adds the Phase-2 cache filter: only DIRTY static pages get a draw (binding 6
// slotDirty), exactly like vsm_bin.comp for opaque statics. Cached pages keep their depth
// → trees stop re-rasterizing every frame; they refresh only as the sun drift dirties
// their pages (round-robin). See vk_vsm.cpp / vk_TreeManager_Render.cpp. [Phase 1]
#include "vsm_common.glsl"

layout(local_size_x = 64) in;

struct TreeMeta { vec3 sphere_P; float sphere_R; uint index_count, ib_first, first_vertex, pad; };
layout(set = 0, binding = 0) readonly buffer Meta { TreeMeta meta[]; };
layout(set = 0, binding = 1) uniform VsmParams {
    mat4 view;
    vec4 level[VSM_LEVELS];   // xy = level origin (light XY), z = extent (m)
    vec4 zparams;
} vsm;
layout(set = 0, binding = 2) readonly buffer PageTable  { uint pageTable[]; };   // virtual page -> STATIC slot / UNMAPPED
layout(set = 0, binding = 3) buffer CasterPages { uint casterPages[]; };         // c*CAP + i -> slot
layout(set = 0, binding = 4) buffer Indirect    { uint indirect[]; };            // 5 u32 per tree (VkDrawIndexedIndirectCommand)
layout(set = 0, binding = 5) buffer Stats       { uint stats[]; };               // [0]=draws [1]=instances [2]=maxPages
layout(set = 0, binding = 6) readonly buffer SlotDirty { uint slotDirty[]; };     // slot -> 1 if dirty this frame (Phase 2 cache)

layout(push_constant) uniform Push { uint casterCount; uint cap; } pc;

void main()
{
    uint c = gl_GlobalInvocationID.x;
    if (c >= pc.casterCount) return;

    TreeMeta m  = meta[c];
    vec2     lp = (vsm.view * vec4(m.sphere_P, 1.0)).xy;
    float    R  = m.sphere_R;

    uint cnt = 0u;
    for (int L = 0; L < VSM_LEVELS; ++L) {
        vec2  origin = vsm.level[L].xy;
        float pw     = vsm.level[L].z / float(VSM_PAGES_AXIS);
        vec2  lo = (lp - vec2(R) - origin) / pw;
        vec2  hi = (lp + vec2(R) - origin) / pw;
        if (hi.x < 0.0 || hi.y < 0.0 || lo.x >= float(VSM_PAGES_AXIS) || lo.y >= float(VSM_PAGES_AXIS))
            continue;
        ivec2 p0 = clamp(ivec2(floor(lo)), ivec2(0), ivec2(VSM_PAGES_AXIS - 1));
        ivec2 p1 = clamp(ivec2(floor(hi)), ivec2(0), ivec2(VSM_PAGES_AXIS - 1));
        for (int py = p0.y; py <= p1.y; ++py)
        for (int px = p0.x; px <= p1.x; ++px) {
            uint slot = pageTable[vsmPageIndex(L, ivec2(px, py))];
            if (slot == VSM_UNMAPPED) continue;
            if (slotDirty[slot] == 0u) continue;   // page is cached this frame -> don't re-draw the tree into it
            if (cnt < pc.cap) casterPages[c * pc.cap + cnt] = slot;   // thread owns this slice -> no atomic
            ++cnt;
        }
    }
    atomicMax(stats[2], cnt);

    uint inst = min(cnt, pc.cap);
    uint base = c * 5u;                          // VkDrawIndexedIndirectCommand = 5 u32
    indirect[base + 0] = m.index_count;          // indexCount
    indirect[base + 1] = inst;                   // instanceCount
    indirect[base + 2] = m.ib_first;             // firstIndex
    indirect[base + 3] = m.first_vertex;         // vertexOffset (>=0 -> uint bits == int)
    indirect[base + 4] = c * pc.cap;             // firstInstance -> casterPages base for this tree
    if (inst > 0u) { atomicAdd(stats[0], 1u); atomicAdd(stats[1], inst); }
}
