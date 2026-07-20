#version 450
#extension GL_GOOGLE_include_directive : require
// xrRenderVulkan — VSM tree IMPOSTOR page VS (r_vsm_tree_impostor). Replaces the
// near-tree crown MESH raster in the DYNAMIC atlas with ONE sun-facing billboard
// per tree: 2 tris + 4 verts sway in the wind instead of thousands of crown verts
// + hundreds of alpha-test tris, EVERY frame. Same page routing as
// tree_vsm_page_dyn (each instance = one atlas page); the quad is built from the
// tree's world bounding sphere and alpha-tested against a baked per-species
// silhouette atlas (see CTreeManager::BuildImpostorAtlas). DYNAMIC atlas only.
#define TV_MAX_PHYS VSM_MAX_PHYS
#define TV_ATLAS_W  VSM_ATLAS_W
#define TV_ATLAS_H  VSM_ATLAS_H
#include "vsm_common.glsl"
#include "ssfx_tree_wind.glsl"   // set0 binding1 s_waves; card sways like a crown

struct TreeInstance { mat4 xform; float c_scale_hemi; float c_bias_hemi; uint _p0; uint _p1; };
layout(set = 0, binding = 0, std430) readonly buffer XformBuf { TreeInstance inst[]; };

struct TreeMeta { vec3 sphere_P; float sphere_R; uint index_count, ib_first, first_vertex, pad; };
layout(set = 1, binding = 0, std430) readonly buffer MetaBuf { TreeMeta meta[]; };
layout(set = 1, binding = 1, std430) readonly buffer SilBuf  { uint silIdx[]; };   // per-tree → silhouette atlas ROW

layout(set = 2, binding = 0) readonly buffer PageList    { uvec4 pageList[]; };
layout(set = 2, binding = 1) readonly buffer CasterPages { uint  casterPages[]; };
layout(set = 2, binding = 2) uniform VsmParams { mat4 view; vec4 level[VSM_LEVELS]; vec4 zparams; } vsm;

layout(push_constant) uniform PC {
    vec4  sunDir;        // xyz = world sun_dir (light travel direction); w = card scale
    vec4  wind_params;   // (wind_direction, wind_velocity, _, crownH)
    vec4  wsetup_trees;  // (branchSpeed, trunkSpeed, bend, minWindSpeed)
    vec4  wind_anim;     // (drift.xyz, flutterAmp)
    uint  cap; uint numMeshes; float alphaRef; uint facetCount;
} pc;

layout(location = 0) out vec2 vUV;
out gl_PerVertex { vec4 gl_Position; float gl_ClipDistance[4]; };

void main()
{
    uint entry = casterPages[gl_InstanceIndex];   // arena entry = (treeIdx<<13)|slot (vsm_tree_bin)
    uint slot  = entry & 0x1FFFu;
    if (slot >= uint(TV_MAX_PHYS)) {   // this instance's page didn't resolve → cull
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        gl_ClipDistance[0] = gl_ClipDistance[1] = gl_ClipDistance[2] = gl_ClipDistance[3] = -1.0;
        return;
    }
    uint treeIdx = entry >> 13u;
    TreeMeta m = meta[treeIdx];
    mat4 X = inst[treeIdx].xform;

    // Quad corner (6 verts = 2 tris) → local [-1,1]^2 and [0,1]^2 UV.
    const vec2 QC[6] = vec2[6]( vec2(-1.0,-1.0), vec2(1.0,-1.0), vec2(1.0,1.0),
                                vec2(-1.0,-1.0), vec2(1.0, 1.0), vec2(-1.0,1.0) );
    vec2 q  = QC[gl_VertexIndex];
    vec2 lu = q * 0.5 + 0.5;

    // Card basis: VERTICAL azimuthal billboard (up = world Y, rotate only in azimuth to
    // face the sun) — matches the HORIZONTALLY-baked silhouette + keeps the trunk on the
    // ground and the crown up top so the shadow lands where the tree is (a sun-⊥ tilt would
    // displace the whole blob down-sun as the sun lowers). toSun = -sun_dir (light travel).
    vec3 toSun = normalize(-pc.sunDir.xyz);
    vec3 up    = vec3(0.0, 1.0, 0.0);
    vec3 sunH  = vec3(toSun.x, 0.0, toSun.z);
    float hl   = length(sunH);
    sunH       = (hl > 1e-3) ? sunH / hl : vec3(0.0, 0.0, 1.0);   // sun near-vertical → arbitrary azimuth
    vec3 right = normalize(cross(up, sunH));                       // horizontal, ⊥ sun azimuth

    float S  = m.sphere_R * pc.sunDir.w;
    vec3  wp = m.sphere_P + (q.x * S) * right + (q.y * S) * up;

    // NO WIND on the impostor: a swaying flat card slides its whole silhouette (crown-sway
    // amplitude is metres) → the ground shadow smears badly under any motion. A static
    // billboard shadow is far more stable and the sway is barely readable on the ground.
    // (Wind stays on the full-mesh crown pass for the very-near trees if a crown tier is added.)

    // Facet select: model-space azimuth of the (horizontal) sun dir (removes per-instance yaw).
    vec3  xAxis = normalize(X[0].xyz);
    vec3  zAxis = normalize(X[2].xyz);
    float az    = atan(dot(sunH, zAxis), dot(sunH, xAxis));   // [-pi, pi]
    float fc    = float(pc.facetCount);
    float fi    = mod(floor(az / (6.2831853 / fc) + 0.5), fc);
    float row   = float(silIdx[treeIdx]);
    vUV = vec2((fi + lu.x) / fc, (row + lu.y) / float(pc.numMeshes));

    // Route into the atlas page (identical to tree_vsm_page_body).
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

    uint  ax = slot % uint(TV_ATLAS_W);
    uint  ay = slot / uint(TV_ATLAS_W);
    float hX = 1.0 / float(TV_ATLAS_W);
    float hY = 1.0 / float(TV_ATLAS_H);
    float cx = (float(ax) + 0.5) * 2.0 * hX - 1.0;
    float cy = (float(ay) + 0.5) * 2.0 * hY - 1.0;
    gl_Position = vec4(cx + nxy.x * hX, cy + nxy.y * hY, nz, 1.0);
}
