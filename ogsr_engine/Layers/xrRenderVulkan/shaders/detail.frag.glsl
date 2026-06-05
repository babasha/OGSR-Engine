#version 450
// xrRenderVulkan — detail (grass) fragment shader.
// set=0/binding=0 = per-type diffuse texture. Hard alpha-cutoff at 0.5 because
// grass billboards are punch-out, not blended.

layout(set = 0, binding = 0) uniform sampler2D uDiffuse;

layout(push_constant) uniform DetailConstants {
    mat4 mViewProj;
    vec4 vWave;
    vec4 vWind;
    vec4 vConsts;
    vec4 vInteractors[4];
} pc;

layout(location = 0) in vec2  vUV;
layout(location = 1) in vec4  vColor;
layout(location = 2) in float vHeight;

layout(location = 0) out vec4 outColor;

void main()
{
    vec4 diff = texture(uDiffuse, vUV);
    if (diff.a < 0.5) discard;
    outColor = vec4(diff.rgb * vColor.rgb, 1.0);
}
