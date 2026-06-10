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
} pc;

layout(location = 0) out vec2  vUV;
layout(location = 1) out float vLight;   // per-instance hemi modulation

void main()
{
    TreeInstance t = inst[gl_InstanceIndex];

    vec4 worldPos = t.xform * vec4(aPos, 1.0);
    gl_Position   = pc.mViewProj * worldPos;

    vUV    = aUV * pc.uvScale;
    // c_scale.hemi defaults to ~1.0 (lit), c_bias.hemi ~0. Floor keeps shadowed
    // trees from going fully black until real hemi lighting lands.
    vLight = clamp(t.c_scale_hemi + t.c_bias_hemi, 0.45, 1.5);
}
