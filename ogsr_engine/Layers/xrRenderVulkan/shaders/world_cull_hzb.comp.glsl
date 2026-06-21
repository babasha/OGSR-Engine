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

struct Meta { vec4 sphere; uint indexCount; uint ibFirst; uint firstVertex; uint group; };
struct Cmd  { uint indexCount; uint instanceCount; uint firstIndex; int vertexOffset; uint firstInstance; };

layout(set = 0, binding = 0) readonly buffer Metas { Meta metas[]; };
layout(set = 0, binding = 1)          buffer Cmds  { Cmd  cmds[];  };
layout(set = 0, binding = 2)          buffer Count { uint counts[]; };
layout(set = 0, binding = 3) uniform sampler2D u_HZB;   // Hi-Z pyramid (MAX depth, GENERAL layout)

layout(push_constant) uniform PC {
    mat4 viewProj;          // X-Ray viewProj uploaded so viewProj * vec4(p,1) = clip
    vec4 frustumPlanes[6];  // CPU-extracted, normalized (VK::ExtractFrustumPlanes)
    vec4 cameraPos;         // .xyz world camera position (near-face pull)
    uint numGroups;
    uint maxGroupMesh;      // region stride: each group's cmd region holds maxGroupMesh slots
    uint total;             // number of meshes
    uint _pad;
} pc;

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

    // ---- Hi-Z occlusion cull (conservative) ----
    // Screen position + footprint come from the sphere CENTRE; the compared depth is
    // the sphere's NEAR face (closest point to the camera). detail_generate pulls the
    // sample UV toward the camera by r too, which is fine for tiny grass — but world/
    // terrain bounding spheres are tens of metres, so pulling the UV would offset the
    // sample to a DIFFERENT screen location than the object → false culls (whole
    // terrain tiles / window frames vanishing, view-dependently). Sample at the CENTRE.
    vec4  clipC   = pc.viewProj * vec4(c, 1.0);
    vec3  toCam   = pc.cameraPos.xyz - c;
    float camDist = length(toCam);
    if (clipC.w > 0.0 && camDist > 1e-3) {
        vec2 ndc = clipC.xy / clipC.w;
        // Y flip: the colour/depth viewport uses negative height → HZB Y is inverted.
        vec2 uv  = vec2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        if (uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0) {
            // Near-face depth = sphere's closest point to the camera (most conservative
            // depth) — projected only for its z; the UV stays centred on the object.
            vec4 clipN = pc.viewProj * vec4(c + (toCam / camDist) * r, 1.0);
            if (clipN.w > 0.0) {
                float instanceDepth = clipN.z / clipN.w;  // NDC z ∈ [0,1] = depth-buffer space

                // Mip where the sphere covers ~1 HZB texel (footprint from the centre).
                float projDiameter = 2.0 * r / clipC.w;
                ivec2 hzbSize      = textureSize(u_HZB, 0);
                float screenTexels = projDiameter * 0.5 * float(hzbSize.x);
                float mipLevel     = ceil(log2(max(1.0, screenTexels)));

                // 4-corner MAX (NEAREST sampler) — a single centre tap misses up to 3
                // of the 2x2 texels the footprint straddles → under-estimates the max
                // and flickers visible meshes out. Corner max is conservative.
                float uvRadius = projDiameter * 0.25;     // NDC→UV halving
                float d0 = textureLod(u_HZB, uv + vec2(-uvRadius, -uvRadius), mipLevel).r;
                float d1 = textureLod(u_HZB, uv + vec2( uvRadius, -uvRadius), mipLevel).r;
                float d2 = textureLod(u_HZB, uv + vec2(-uvRadius,  uvRadius), mipLevel).r;
                float d3 = textureLod(u_HZB, uv + vec2( uvRadius,  uvRadius), mipLevel).r;
                float hzbDepth = max(max(d0, d1), max(d2, d3));

                if (instanceDepth > hzbDepth && hzbDepth > 0.0)
                    return;   // fully behind the farthest surface in its footprint
            }
        }
    }

    // ---- Visible: append the indirect draw command to the mesh's group region ----
    uint g    = m.group;
    uint o    = atomicAdd(counts[g], 1u);
    uint base = g * pc.maxGroupMesh;
    cmds[base + o].indexCount    = m.indexCount;
    cmds[base + o].instanceCount = 1u;
    cmds[base + o].firstIndex    = m.ibFirst;
    cmds[base + o].vertexOffset  = int(m.firstVertex);
    cmds[base + o].firstInstance = 0u;
}
