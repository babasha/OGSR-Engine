#version 450
#extension GL_GOOGLE_include_directive : require
// xrRenderVulkan — VSM TREE-caster binning, near/far WIND HYBRID (r_vsm_tree_wind).
// One thread per tree: find the allocated clipmap pages its world sphere overlaps and
// emit ONE indexed-indirect draw (instanceCount = page count). TWO dispatches per frame
// over the same shader, selected by push.mode:
//   mode 0  STATIC atlas: FAR trees only (nearFlags[c]==0), DIRTY pages only (binding 6
//           slotDirty — the toroidal cache filter), rigid raster. [Phase 1]
//   mode 1  DYNAMIC atlas: NEAR trees only (nearFlags[c]!=0), ALL resident pages (the
//           dyn atlas re-renders every frame), marks dynUsed (binding 7) so the resolve's
//           r_vsm_dyn_gate keeps their taps. Rendered with live wind → smooth sway.
// nearFlags (binding 8) is CPU-computed with hysteresis (single source of truth — the
// CPU also emits residency invalidation circles when a tree crosses the boundary).
// See vk_vsm.cpp / vk_TreeManager_Render.cpp.
#include "vsm_common.glsl"

layout(local_size_x = 64) in;

struct TreeMeta { vec3 sphere_P; float sphere_R; uint index_count, ib_first, first_vertex, pad; };
layout(set = 0, binding = 0) readonly buffer Meta { TreeMeta meta[]; };
layout(set = 0, binding = 1) uniform VsmParams {
    mat4 view;
    vec4 level[VSM_LEVELS];   // xy = level origin (light XY), z = extent (m)
    vec4 zparams;
} vsm;
layout(set = 0, binding = 2) readonly buffer PageTable  { uint pageTable[]; };   // virtual page -> slot / UNMAPPED (static or dyn table per mode)
layout(set = 0, binding = 3) buffer CasterPages { uint casterPages[]; };         // c*CAP + i -> slot (shared: near/far tree slices are disjoint)
layout(set = 0, binding = 4) buffer Indirect    { uint indirect[]; };            // 5 u32 per tree (per-mode buffer)
layout(set = 0, binding = 5) buffer Stats       { uint stats[]; };               // [0]=draws [1]=instances [2]=maxPages
layout(set = 0, binding = 6) readonly buffer SlotDirty { uint slotDirty[]; };     // STATIC dirty set (mode 0 only)
layout(set = 0, binding = 7) buffer DynUsed     { uint dynUsed[]; };             // dyn slot -> has-caster flag (mode 1 only; resolve gate)
layout(set = 0, binding = 8) readonly buffer NearFlags { uint nearFlags[]; };     // tree -> 1 if in the NEAR (dynamic wind) set
layout(set = 0, binding = 9) readonly buffer PageMax   { float pageMax[]; };       // shadow-HZB: per STATIC slot, max prior depth (1.0 = no occlusion)
layout(set = 0, binding = 10) readonly buffer StaticPT { uint staticPT[]; };       // virtual page -> STATIC slot (Option A: dyn near-trees query static occluders)

layout(push_constant) uniform Push { uint casterCount; uint cap; uint mode; uint hzbOn; float margin; uint pad; } pc;

void main()
{
    uint c = gl_GlobalInvocationID.x;
    if (c >= pc.casterCount) return;

    TreeMeta m  = meta[c];
    vec3     lc = (vsm.view * vec4(m.sphere_P, 1.0)).xyz;
    vec2     lp = lc.xy;
    float    R  = m.sphere_R;
    // shadow-HZB: the tree's nearest-to-light normalized depth (sphere front point). If this is
    // farther than a page's cached MAX depth, the whole tree is behind the occluders in that page.
    float    nearestD = (lc.z - R - vsm.zparams.x) * vsm.zparams.y;
    // Near/far split: each tree casts into exactly ONE atlas per frame.
    bool pass = (pc.mode == 1u) ? (nearFlags[c] != 0u) : (nearFlags[c] == 0u);

    uint cnt = 0u;
    if (pass)
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
            uint vpi  = vsmPageIndex(L, ivec2(px, py));
            uint slot = pageTable[vpi];
            if (slot == VSM_UNMAPPED) continue;
            if (pc.mode == 0u && slotDirty[slot] == 0u) continue;   // static cache: dirty pages only
            // shadow-HZB: skip a page whose cached STATIC occluders are all NEARER to the light than the
            // tree's front → the tree can't win the depth test there → no raster (kills alpha-test overdraw).
            // mode 0 (static/far trees): occluder slot == this slot. mode 1 (dyn/near trees, Option A): the
            // occluder is the STATIC page for the same world page (walls/terrain), looked up via staticPT.
            // Only culls what's HIDDEN behind static geometry → the visible near shadow is untouched.
            if (pc.hzbOn != 0u) {
                uint sslot = (pc.mode == 0u) ? slot : staticPT[vpi];
                if (sslot != VSM_UNMAPPED && nearestD > pageMax[sslot] + pc.margin) { atomicAdd(stats[4], 1u); continue; }
            }
            if (cnt < pc.cap) {
                casterPages[c * pc.cap + cnt] = slot;   // thread owns this slice -> no atomic
                if (pc.mode == 1u) atomicOr(dynUsed[slot], 1u);   // dyn page gets real caster depth
            }
            ++cnt;
        }
    }
    atomicMax(stats[2], cnt);

    // ALWAYS write the command (inst=0 for skipped trees) — stale counts must not draw.
    uint inst = min(cnt, pc.cap);
    uint base = c * 5u;                          // VkDrawIndexedIndirectCommand = 5 u32
    indirect[base + 0] = m.index_count;          // indexCount
    indirect[base + 1] = inst;                   // instanceCount
    indirect[base + 2] = m.ib_first;             // firstIndex
    indirect[base + 3] = m.first_vertex;         // vertexOffset (>=0 -> uint bits == int)
    indirect[base + 4] = c * pc.cap;             // firstInstance -> casterPages base for this tree
    if (inst > 0u) { atomicAdd(stats[0], 1u); atomicAdd(stats[1], inst); }
    if (pc.mode == 1u && inst > 0u) atomicAdd(stats[3], inst);   // dyn-pass instances (hybrid diagnostic)
}
