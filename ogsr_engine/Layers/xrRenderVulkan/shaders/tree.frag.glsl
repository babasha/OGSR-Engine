#version 450
// xrRenderVulkan — tree forward fragment shader (Session B).
// set 1/binding 0 = per-group diffuse texture. Trees are alpha-tested
// (punch-out leaves), not blended; discard below the push-constant cutoff.

layout(set = 1, binding = 0) uniform sampler2D uDiffuse;

layout(push_constant) uniform PC {
    mat4  mViewProj;
    float uvScale;
    float alphaRef;
} pc;

layout(location = 0) in vec2  vUV;
layout(location = 1) in float vLight;

layout(location = 0) out vec4 outColor;

void main()
{
    vec4 diff = texture(uDiffuse, vUV);
    if (diff.a < pc.alphaRef)
        discard;
    outColor = vec4(diff.rgb * vLight, 1.0);
}
