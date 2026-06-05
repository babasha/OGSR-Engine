#version 450
// xrRenderVulkan — LOD imposter fragment shader.
// Sample the level_lods atlas, alpha-test the billboard cutout. The atlas image
// already carries baked tree colour/lighting, so the per-vertex tint is just a
// pass-through (white in v1).

layout(set = 0, binding = 0) uniform sampler2D uAtlas;

layout(location = 0) in vec2 vUV;
layout(location = 1) in vec4 vColor;

layout(location = 0) out vec4 outColor;

void main()
{
    vec4 d = texture(uAtlas, vUV);
    if (d.a < 0.5)
        discard;
    outColor = vec4(d.rgb * vColor.rgb, 1.0);
}
