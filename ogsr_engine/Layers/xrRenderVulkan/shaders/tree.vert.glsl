#version 450
// xrRenderVulkan — tree forward vertex shader (Session B).
//
// GPU-driven indirect draw: the cull compute wrote firstInstance = global
// tree index into each draw command, so gl_InstanceIndex selects this tree's
// world transform from the transforms SSBO (set 0). Vertex format matches the
// level VB pool: FLOAT3 position @ 0, SHORT2 SSCALED UV @ tcOffset (24 or 28,
// picked by the pipeline variant).
//
// Matrix convention: raw X-Ray (row-major) Fmatrices loaded into column-major
// GLSL mat4 transpose, so `M * v` == X-Ray `v * F`. Chaining
// `mViewProj * (xform * pos)` composes local→world→clip correctly (same as the
// monolith gbuffer_tree_indirect.vert).

layout(location = 0) in vec3 aPos;   // local-space position
layout(location = 1) in vec2 aUV;    // SHORT2 SSCALED raw int16 → float

// set 0: per-tree instance data (mirrors VK::GpuTreeInstance, 80 B).
struct TreeInstance {
    mat4  xform;        // 64 B per-instance world transform
    float c_scale_hemi; //  4 B
    float c_bias_hemi;  //  4 B
    uint  _p0;          //  4 B
    uint  _p1;          //  4 B
};
layout(set = 0, binding = 0, std430) readonly buffer XformBuf {
    TreeInstance inst[];
};

layout(push_constant) uniform PC {
    mat4  mViewProj;    // world → clip
    float uvScale;      // 1/2048 — tree UV quant (FTreeVisual_quant = 32768/16)
    float alphaRef;     // fragment alpha cutoff
    float _pad0;
    float _pad1;
    vec4  vSunColor;    // env sun colour (rgb)
    vec4  vHemiColor;   // env hemi colour (rgb)
} pc;

layout(location = 0) out vec2 vUV;
layout(location = 1) out vec3 vLight;    // hemi + floor (shadow-independent part)
layout(location = 2) out vec3 vSunLit;   // sun part — attenuated by the shadow map in frag
layout(location = 3) out vec3 vWPos;     // world-space position (shadow lookup)

void main()
{
    TreeInstance t = inst[gl_InstanceIndex];

    vec4 worldPos = t.xform * vec4(aPos, 1.0);
    gl_Position   = pc.mViewProj * worldPos;

    vUV = aUV * pc.uvScale;
    // c_scale.hemi defaults to ~1.0 (lit), c_bias.hemi ~0 → per-tree openness factor.
    // The openness scalar goes to the fragment in vLight.x — it lights the
    // crown with the SAME sky-cube ambient the world ground uses (see
    // tree.frag), so trees/bushes track the upgraded world lighting instead of
    // the old flat hemi (which read dark next to the sky-lit terrain).
    // The sun part travels separately — the fragment shadows it with the sun map.
    float hemiFac = clamp(t.c_scale_hemi + t.c_bias_hemi, 0.3, 1.5);
    vLight  = vec3(hemiFac, 0.0, 0.0);
    // Sun ×1.1 (was 0.6): foliage was catching HALF the sun the rest of the
    // scene gets (world/terrain/grass run ×1.25) — at golden hour the orange
    // sun visibly "paints" R4's bushes while ours stayed grey. R4 lights
    // foliage through the same deferred sun as everything else.
    vSunLit = pc.vSunColor.rgb  * hemiFac * 1.1;
    vWPos   = worldPos.xyz;
}
