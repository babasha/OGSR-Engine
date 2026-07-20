#version 450
// GPU-driven world forward culling WITH Hi-Z occlusion (Phase A of cluster cull).
// Same per-mesh frustum cull as world_cull.comp, plus an HZB occlusion test that
// drops meshes fully hidden behind nearer scene geometry — so the heavy forward
// COLOR pass skips them. The DEPTH PREPASS still draws the full frustum set (it
// builds the Hi-Z), so the pyramid is complete and this never over-culls.
//
// Output goes to a SEPARATE indirect/count region (cmds2/counts2) consumed only
// by DrawColor when r_hzb_cull is on; the depth prepass keeps the frustum set.
//
// Frustum planes + viewProj + cameraPos all travel in the push, mirroring the
// proven grass path in detail_generate.comp (same push shape, same HZB sample
// math: Y-flip, mip select by screen footprint, conservative 4-corner MAX, near
// face). Planes are the CPU-extracted, normalized set (VK::ExtractFrustumPlanes).
layout(local_size_x = 256) in;

// Meta = one cullable entry (whole mesh OR one LOD-DAG cluster) — see world_cull.comp.
struct Meta {
    vec4 sphere;        // cull sphere (frustum/HZB)
    vec4 lodSelf;       // birth-group sphere (self LOD test)
    vec4 lodParent;     // parent-group sphere (parent LOD test)
    uint indexCount; uint ibFirst; uint firstVertex; uint group;
    float selfError; float parentError; uint flags; uint _p1;   // flags bit0 = hard cut (alpha-tested material)
};
struct Cmd  { uint indexCount; uint instanceCount; uint firstIndex; int vertexOffset; uint firstInstance; };

layout(set = 0, binding = 0) readonly buffer Metas { Meta metas[]; };
layout(set = 0, binding = 1)          buffer Cmds  { Cmd  cmds[];  };
layout(set = 0, binding = 2)          buffer Count { uint counts[]; };
layout(set = 0, binding = 3) uniform sampler2D u_HZB;   // Hi-Z pyramid (MAX depth, GENERAL layout)
layout(set = 0, binding = 4) readonly buffer Bases { uint groupBase[]; };   // per-group cmd-region base
// Stage B page streaming — MUST mirror world_cull.comp (prepass and color sets
// have to pick the same cut). Requests/touched are emitted by world_cull only.
layout(set = 0, binding = 5) readonly buffer SBits { uint streamBits[]; };
layout(set = 0, binding = 6) readonly buffer SBase { uint pageSlotBase[]; };

layout(push_constant) uniform PC {
    mat4 viewProj;          // X-Ray viewProj uploaded so viewProj * vec4(p,1) = clip
    vec4 frustumPlanes[6];  // CPU-extracted, normalized (VK::ExtractFrustumPlanes)
    vec4 cameraPos;         // .xyz world camera position (near-face pull + LOD projection)
    vec4 viewDir;           // .xyz normalized camera forward (view-Z LOD projection)
    vec4 lodParams;         // x = pxScale/thresholdPx, y = min dist clamp, z = fade band, w = 1/tan(fovY/2)
    uint numGroups;
    uint unused0;           // was maxGroupMesh (uniform regions) — kept for layout stability
    uint total;             // number of cullable entries (meshes + r_cluster clusters)
    uint _pad;
} pc;

// Projected error / threshold — MUST match world_cull.comp exactly: the prepass
// (frustum set) and this color set have to pick the same DAG cut, or the color
// pass z-fights the prepass depth. View-Z (not Euclidean distance) — see the
// rationale in world_cull.comp.
float projErr(vec4 s, float e)
{
    float d = max(pc.lodParams.y, dot(pc.viewDir.xyz, s.xyz - pc.cameraPos.xyz) - s.w);
    return e * pc.lodParams.x / d;
}

void main()
{
    uint l = gl_GlobalInvocationID.x;
    if (l >= pc.total) return;

    Meta m = metas[l];
    vec3  c = m.sphere.xyz;
    float r = m.sphere.w;

    // ---- Frustum cull (6 planes vs sphere; same test as world_cull.comp) ----
    for (int i = 0; i < 6; ++i)
        if (dot(pc.frustumPlanes[i].xyz, c) + pc.frustumPlanes[i].w < -r) return;   // outside

    // ---- Stage B residency (identical to world_cull.comp) ----
    uint sb = (streamBits[l >> 4u] >> ((l & 15u) * 2u)) & 3u;
    if ((sb & 1u) == 0u) return;
    bool sleaf = (sb & 2u) != 0u;

    // ---- LOD DAG cut (identical math to world_cull.comp incl. crossfade) ----
    float sp = projErr(m.lodSelf,   m.selfError);
    float pp = projErr(m.lodParent, m.parentError);
    float band = pc.lodParams.z;
    uint fA = 63u, fB = 0u;
    if (band <= 0.0 || (m.flags & 1u) != 0u) {
        if ((sp > 1.0 && !sleaf) || pp <= 1.0) return;
    } else {
        if (pp <= 1.0 || (sp > 1.0 + band && !sleaf)) return;
        fA = uint(clamp((pp - 1.0) / band, 0.0, 1.0) * 63.0 + 0.5);
        fB = sleaf ? 0u : uint(clamp((sp - 1.0) / band, 0.0, 1.0) * 63.0 + 0.5);
        if (fA == 0u) return;
    }

    // ---- Hi-Z occlusion cull (conservative) ----
    // Screen position + footprint come from the sphere CENTRE; the compared depth is
    // the sphere's NEAR face (closest point to the camera). detail_generate pulls the
    // sample UV toward the camera by r too, which is fine for tiny grass — but world/
    // terrain bounding spheres are tens of metres, so pulling the UV would offset the
    // sample to a DIFFERENT screen location than the object → false culls (whole
    // terrain tiles / window frames vanishing, view-dependently). Sample at the CENTRE.
    //
    // MID-TRANSITION entries (crossfade) are EXEMPT: the prepass (frustum set)
    // draws the parent+children pair with complementary dither masks, and if the
    // color set occlusion-culls only ONE half of that pair (their spheres differ),
    // the other half's discarded pixels stay black → view-dependent dark/flickering
    // facades. Transitions live for fractions of a second — skipping their
    // occlusion test costs nothing.
    vec4  clipC   = pc.viewProj * vec4(c, 1.0);
    vec3  toCam   = pc.cameraPos.xyz - c;
    float camDist = length(toCam);
    if (fA == 63u && fB == 0u && clipC.w > 0.0 && camDist > 1e-3) {
        vec2 ndc = clipC.xy / clipC.w;
        // Y flip: the colour/depth viewport uses negative height → HZB Y is inverted.
        vec2 uv  = vec2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        if (uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0) {
            // Near-face depth = sphere's closest point to the camera (most conservative
            // depth) — projected only for its z; the UV stays centred on the object.
            vec4 clipN = pc.viewProj * vec4(c + (toCam / camDist) * r, 1.0);
            if (clipN.w > 0.0) {
                float instanceDepth = clipN.z / clipN.w;  // NDC z ∈ [0,1] = depth-buffer space

                // Mip where the sphere covers ~1 HZB texel. TRUE screen footprint
                // needs the projection's focal scale (lodParams.w = 1/tan(fovY/2)):
                // pixel diameter = r · P11 · H / w. The old 2·r/w (no focal term)
                // understated it ~1.5× → too fine a mip → the 2×2 block missed
                // part of the footprint → FALSE CULLS (vanishing walls/tiles —
                // why r_hzb_cull stayed situational). The footprint in TEXELS is
                // axis-symmetric (x: P11/aspect over aspect-more texels), so the
                // vertical axis serves both.
                ivec2 hzbSize      = textureSize(u_HZB, 0);
                float screenTexels = r * pc.lodParams.w * float(hzbSize.y) / clipC.w;
                float mipLevel     = ceil(log2(max(1.0, screenTexels)));

                // GRID-ALIGNED 2x2 block max: with the footprint <= 1 texel at this
                // mip it straddles at most 2x2 texels — fetching the block that
                // CONTAINS the footprint is guaranteed-conservative. (The previous
                // circle-corner taps could all land on a near wall and miss the
                // opening a support post was visible through → false culls.)
                int   mi = int(mipLevel);
                vec2  ts = vec2(textureSize(u_HZB, mi));
                // UV half-extent, focal-corrected; the vertical scale is the
                // larger axis in UV (x is P11/aspect) — conservative for both.
                float uvRadius = 0.5 * r * pc.lodParams.w / clipC.w;
                ivec2 tmax = ivec2(ts) - 1;
                ivec2 t0 = clamp(ivec2(floor((uv - vec2(uvRadius)) * ts)), ivec2(0), tmax);
                ivec2 t1 = clamp(ivec2(floor((uv + vec2(uvRadius)) * ts)), ivec2(0), tmax);
                float d0 = texelFetch(u_HZB, ivec2(t0.x, t0.y), mi).r;
                float d1 = texelFetch(u_HZB, ivec2(t1.x, t0.y), mi).r;
                float d2 = texelFetch(u_HZB, ivec2(t0.x, t1.y), mi).r;
                float d3 = texelFetch(u_HZB, ivec2(t1.x, t1.y), mi).r;
                float hzbDepth = max(max(d0, d1), max(d2, d3));

                if (instanceDepth > hzbDepth && hzbDepth > 0.0)
                    return;   // fully behind the farthest surface in its footprint
            }
        }
    }

    // ---- Visible: append the indirect draw command to the entry's group region ----
    uint g    = m.group;
    uint o    = atomicAdd(counts[g], 1u);
    uint base = groupBase[g];
    cmds[base + o].indexCount    = m.indexCount;
    cmds[base + o].instanceCount = 1u;
    cmds[base + o].firstIndex    = pageSlotBase[2u * m._p1] + m.ibFirst;   // page-local -> pool offset
    cmds[base + o].vertexOffset  = int(pageSlotBase[2u * m._p1 + 1u] + m.firstVertex);   // + page VB base (slice 2)
    cmds[base + o].firstInstance = l | (fA << 20) | (fB << 26);   // entry id (20b) + crossfade fades
}
