#version 450
#extension GL_GOOGLE_include_directive : require
#include "froxel.glsl"   // Froxel_SliceFromViewZ — shared exp-Z mapping

// xrRenderVulkan — Stage-1 Volumetric Media System: SMOKE DENSITY SPLAT.
//
// One thread per smoke particle. Maps the particle's world centre to its froxel
// in the (coarse) smoke grid — the EXACT inverse of vol_inject's froxel->world
// basis map, so the splat lands in the same cell vol_inject will read — and
// atomic-adds fixed-point {density, density*R, density*G, density*B} into the
// accumulation SSBO. vol_resolve then normalises it into an RGBA16F media volume
// that vol_inject samples as extra extinction + albedo-tinted in-scatter, making
// smoke a true participating medium (lit by sun/lights/phase/shadows like fog).
//
// v1 = POINT splat (one froxel per particle); the coarse grid + resolve's trilinear
// read + temporal accumulation soften it. A radius-driven footprint is a follow-up
// (posRadius.w is carried for it but unused here).

layout(local_size_x = 64) in;

struct SmokeParticle { vec4 posRadius; vec4 color; };  // xyz pos / w radius ; rgb albedo / a density

layout(set = 0, binding = 0) readonly buffer Particles { SmokeParticle parts[]; };
layout(set = 0, binding = 1)          buffer Accum     { uint accum[]; };  // [cell*4 + {0:d,1:R,2:G,3:B}]

layout(push_constant) uniform Push {
    vec4 camPos;     // xyz world camera pos
    vec4 camDir;     // xyz camera forward (unit)
    vec4 camRightT;  // xyz right * tan(fovX/2)  (same scaled basis vol_inject uses)
    vec4 camTopT;    // xyz top   * tan(fovY/2)
    vec4 zParams;    // x=near y=far z=log2(far/near)
    vec4 dims;       // xyz = smoke grid dims, w = fixed-point scale
    vec4 params;     // x = particle count, y = density scale, z = max footprint (cells; 0 = point splat)
} pc;

// Atomic-add a weighted density+colour contribution into one accum cell (bounds-checked).
void depositCell(ivec3 c, ivec3 dim, float dens, vec3 col)
{
    if (any(lessThan(c, ivec3(0))) || any(greaterThanEqual(c, dim))) return;
    uint cell = (uint(c.z) * uint(dim.y) + uint(c.y)) * uint(dim.x) + uint(c.x);
    float sc = pc.dims.w;
    atomicAdd(accum[cell * 4u + 0u], uint(dens * sc));
    atomicAdd(accum[cell * 4u + 1u], uint(dens * col.r * sc));
    atomicAdd(accum[cell * 4u + 2u], uint(dens * col.g * sc));
    atomicAdd(accum[cell * 4u + 3u], uint(dens * col.b * sc));
}

void main()
{
    uint pid = gl_GlobalInvocationID.x;
    if (pid >= uint(pc.params.x)) return;

    SmokeParticle p = parts[pid];
    vec3  rel   = p.posRadius.xyz - pc.camPos.xyz;
    float nearZ = pc.zParams.x;

    float viewZ = dot(rel, pc.camDir.xyz);
    if (viewZ <= nearZ || viewZ >= pc.zParams.y) return;        // behind near / past far

    // Inverse of inject's ray basis (ray = camDir + camRightT*ndcx + camTopT*ndcy,
    // world = camPos + ray*viewZ). camDir/right/top are mutually orthogonal, so
    // ndc = dot(rel, basisT) / (viewZ * |basisT|^2).
    float ndcx = dot(rel, pc.camRightT.xyz) / (viewZ * dot(pc.camRightT.xyz, pc.camRightT.xyz));
    float ndcy = dot(rel, pc.camTopT.xyz)   / (viewZ * dot(pc.camTopT.xyz,   pc.camTopT.xyz));
    if (abs(ndcx) > 1.0 || abs(ndcy) > 1.0) return;             // outside the frustum sides

    vec2  uv = vec2(ndcx * 0.5 + 0.5, (1.0 - ndcy) * 0.5);      // matches inject's ndc->uv
    float w  = Froxel_SliceFromViewZ(viewZ, nearZ, pc.zParams.z);

    ivec3 dim  = ivec3(pc.dims.xyz);
    float dens = max(p.color.a, 0.0) * pc.params.y;             // density mass (opacity × scale)
    vec3  col  = max(p.color.rgb, vec3(0.0));                   // albedo

    vec3  fc = vec3(uv.x * float(dim.x), uv.y * float(dim.y), w * float(dim.z));  // continuous froxel coord
    ivec3 ci = ivec3(floor(fc));

    // FOOTPRINT (Stage 1.1): spread the particle over a froxel neighbourhood sized by
    // its world radius vs the LOCAL cell size, CAPPED at params.z cells so the cost
    // stays bounded (cost ~ particles × (2·fr+1)³). 0 → point splat (cheapest). Gaussian
    // falloff → soft, non-grainy blobs; not normalized, so a bigger particle deposits
    // more total mass = denser & larger clouds.
    float cellWorld = 2.0 * length(pc.camRightT.xyz) * viewZ / float(dim.x);  // ~lateral cell size (world)
    int   fr = min(int(p.posRadius.w / max(cellWorld, 1e-3)), int(pc.params.z));

    if (fr <= 0) {
        depositCell(clamp(ci, ivec3(0), dim - 1), dim, dens, col);
    } else {
        float sig2 = max(float(fr) * float(fr) * 0.5, 0.25);
        for (int dz = -fr; dz <= fr; ++dz)
        for (int dy = -fr; dy <= fr; ++dy)
        for (int dx = -fr; dx <= fr; ++dx) {
            ivec3 c   = ci + ivec3(dx, dy, dz);
            vec3  off = (vec3(c) + 0.5) - fc;
            float wgt = exp(-dot(off, off) / sig2);
            if (wgt > 0.02) depositCell(c, dim, dens * wgt, col);
        }
    }
}
