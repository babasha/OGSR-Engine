#version 450

// World pass — TERRAIN splatting variant. Same vertex layout as world_lmap
// (stride 32, tcOffset 24: base UV @24, lightmap UV @28). Outputs base UV
// (also used to sample the splat mask), detail UV (base * detailScale), and
// lightmap UV. Fragment shader blends 4 detail textures by the mask's RGBA
// weights — mirrors R4 CBlender_BmmD / deffer_terrain_high.

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec2 inUV_short;     // base UV (SHORT2 SSCALED)
layout(location = 2) in vec4 inTangent;      // .a = du (sub-pixel U fraction)
layout(location = 3) in vec4 inBinormal;     // .a = dv (sub-pixel V fraction)
layout(location = 4) in vec2 inLmapUV_short; // lightmap UV (SHORT2 SSCALED)

layout(push_constant) uniform PushConstants {
    mat4  mvp;
    vec2  uvScale;
    float alphaRef;
    float detailScale;
} pc;

layout(location = 0) out vec2 vUV;       // base + mask UV
layout(location = 1) out vec2 vDetailUV; // detail UV (base * detailScale)
layout(location = 2) out vec2 vLmapUV;

void main()
{
    gl_Position = pc.mvp * vec4(inPos, 1.0);

    vec2 uv   = inUV_short + vec2(inTangent.a, inBinormal.a);
    vUV       = uv * pc.uvScale;
    vDetailUV = vUV * pc.detailScale;
    vLmapUV   = inLmapUV_short * (1.0 / 32768.0);
}
