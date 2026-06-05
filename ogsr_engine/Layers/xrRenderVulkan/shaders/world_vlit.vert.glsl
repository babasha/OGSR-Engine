#version 450

// World pass — vert-lit variant. For X-Ray level statics where the
// level compiler chose to bake lighting per-vertex into a D3DCOLOR
// at offset 24 instead of writing a per-mesh lightmap (`tcOffset == 28`
// sub-layout). COLOR.bgr = baked RGB lighting (point lights + bounce);
// COLOR.a = sun mask (per-vertex sun-direction occlusion).

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec2 inUV_short;   // base UV  @ offset 28
layout(location = 2) in vec4 inTangent;    // .a = du
layout(location = 3) in vec4 inBinormal;   // .a = dv
layout(location = 4) in vec4 inColor;      // D3DCOLOR @ offset 24 (BGRA in memory)

layout(push_constant) uniform PushConstants {
    mat4  mvp;
    vec2  uvScale;
    float alphaRef;
    float detailScale;
} pc;

layout(location = 0) out vec2 vUV;
layout(location = 1) out vec2 vDetailUV;
layout(location = 2) out vec3 vBakedColor;   // RGB lighting (BGR→RGB swizzle)
layout(location = 3) out float vSunMask;

void main()
{
    gl_Position = pc.mvp * vec4(inPos, 1.0);

    vec2 uv     = inUV_short + vec2(inTangent.a, inBinormal.a);
    vUV         = uv * pc.uvScale;
    vDetailUV   = vUV * pc.detailScale;

    // D3DCOLOR memory order is BGRA; reading as R8G8B8A8_UNORM gives the
    // bytes in that order in .rgba → swizzle .bgr to recover real RGB.
    vBakedColor = inColor.bgr;
    vSunMask    = inColor.a;
}
