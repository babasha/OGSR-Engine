#version 450

// Particle billboard fragment shader (OGSR Vulkan).
// Samples the particle sprite and modulates by the per-vertex PAPI colour.
// Blending (additive / alpha / multiply) is configured per-effect on the
// pipeline; this shader just produces tex*color. OGSR's forward path is LDR
// (renders straight to the swapchain), so — unlike the monolith HDR port —
// there is NO sRGB→linear pow(2.2) step here (matches the world shaders).

layout(location = 0) in vec4 fragColor;
layout(location = 1) in vec2 fragUV;

layout(binding = 0) uniform sampler2D texSampler;

layout(location = 0) out vec4 outColor;

void main()
{
    vec4 texColor = texture(texSampler, fragUV);
    outColor = texColor * fragColor;

    // Drop fully-transparent texels (cheap fill saving; black additive texels
    // contribute nothing anyway).
    if (outColor.a < 0.0039)
        discard;
}
