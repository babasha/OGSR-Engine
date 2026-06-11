#version 450

// Particle billboard vertex shader (OGSR Vulkan).
// Vertices are FVF::LIT {vec3 pos; D3DCOLOR color; vec2 uv} built CPU-side in
// world space (PAPI particle positions). Push constant carries the combined
// view-projection (raw X-Ray row-major Fmatrix → GLSL reads it transposed, so
// `viewProj * v` matches X-Ray's `v * M`, same convention as world.vert).

layout(location = 0) in vec3 inPosition;   // world-space position
layout(location = 1) in vec4 inColor;      // D3DCOLOR via B8G8R8A8_UNORM → logical RGBA
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec2 fragUV;

layout(push_constant) uniform PushConstants {
    mat4 viewProj;
} pc;

void main()
{
    gl_Position = pc.viewProj * vec4(inPosition, 1.0);
    fragColor   = inColor;
    fragUV      = inUV;
}
