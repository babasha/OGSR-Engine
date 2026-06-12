#version 450
// xrRenderVulkan — tree shadow caster (depth-only, alpha-tested in frag).
// Same instancing as tree.vert (transforms SSBO indexed by gl_InstanceIndex;
// the caster path passes firstInstance = global tree index per draw), but the
// push mViewProj is the LIGHT view·proj and only the UV survives to the frag.

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec2 aUV;    // SHORT2 SSCALED

struct TreeInstance {
    mat4  xform;
    float c_scale_hemi;
    float c_bias_hemi;
    uint  _p0;
    uint  _p1;
};
layout(set = 0, binding = 0, std430) readonly buffer XformBuf {
    TreeInstance inst[];
};

layout(push_constant) uniform PC {
    mat4  mViewProj;    // light view·proj
    float uvScale;      // 1/2048
    float alphaRef;
} pc;

layout(location = 0) out vec2 vUV;

void main()
{
    TreeInstance t = inst[gl_InstanceIndex];
    gl_Position = pc.mViewProj * (t.xform * vec4(aPos, 1.0));
    vUV = aUV * pc.uvScale;
}
