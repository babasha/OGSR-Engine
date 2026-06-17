#version 450
#extension GL_GOOGLE_include_directive : require
// xrRenderVulkan — VSM grass-caster binning (NEAR + L0 only). One thread per grass
// instance slot in the GPU-driven detail buffer (VisibleSSBO, previous frame — grass
// gen runs AFTER the VSM atlas in the frame, so casting uses 1-frame-stale instances;
// invisible since grass is near-static). For each VALID, NEAR instance whose L0 clipmap
// page is resident, store that page's atlas slot; else VSM_UNMAPPED (the grass-page VS
// culls those). Only level 0 — near grass + near receivers both live in the finest level.
#include "vsm_common.glsl"

layout(local_size_x = 64) in;

struct DetailInstance { vec4 row0, row1, row2, color; };   // 64 B (matches DetailInstance; translation = row*.w)
layout(set = 0, binding = 0) readonly buffer Visible  { DetailInstance inst[]; };
layout(set = 0, binding = 1) readonly buffer Indirect { uint indirect[]; };       // 5 u32 / type, instanceCount @ +1
layout(set = 0, binding = 2) uniform VsmParams { mat4 view; vec4 level[VSM_LEVELS]; vec4 zparams; } vsm;
layout(set = 0, binding = 3) readonly buffer PageTable { uint pageTable[]; };
layout(set = 0, binding = 4) buffer GrassSlot { uint grassSlot[]; };               // global instance idx -> slot / UNMAPPED
layout(set = 0, binding = 5) buffer Stats     { uint stats[]; };                   // [0]=casting instances

layout(push_constant) uniform Push {
    uint sectionSize;   // per-type stride in VisibleSSBO (instances)
    uint typeCount;
    uint pad0, pad1;
    vec4 camRange;      // xyz = camera world pos, w = max cast distance (near cull)
} pc;

void main()
{
    uint g = gl_GlobalInvocationID.x;
    if (g >= pc.sectionSize * pc.typeCount) return;

    uint type  = g / pc.sectionSize;
    uint local = g - type * pc.sectionSize;
    uint count = indirect[type * 5u + 1u];          // stale instanceCount for this type
    if (local >= count) { grassSlot[g] = VSM_UNMAPPED; return; }

    DetailInstance di = inst[g];
    vec3 center = vec3(di.row0.w, di.row1.w, di.row2.w);
    if (distance(center, pc.camRange.xyz) > pc.camRange.w) { grassSlot[g] = VSM_UNMAPPED; return; }

    // L0 only.
    vec2 lp = (vsm.view * vec4(center, 1.0)).xy;
    vec2 t  = (lp - vsm.level[0].xy) / vsm.level[0].z;
    if (any(lessThan(t, vec2(0.0))) || any(greaterThanEqual(t, vec2(1.0)))) { grassSlot[g] = VSM_UNMAPPED; return; }
    ivec2 page = ivec2(floor(t * float(VSM_PAGES_AXIS)));
    uint slot = pageTable[vsmPageIndex(0, page)];
    grassSlot[g] = slot;                            // UNMAPPED → VS culls
    if (slot != VSM_UNMAPPED) atomicAdd(stats[0], 1u);
}
