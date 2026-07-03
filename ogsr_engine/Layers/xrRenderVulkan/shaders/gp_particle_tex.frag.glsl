#version 450
#extension GL_EXT_nonuniform_qualifier : require

// GPU particle billboard fragment shader — textured (bindless) variant.
// Samples the effect's sprite texture from a bindless array indexed per
// particle by its program slot (fragLayer). Untextured particles (fragLayer < 0,
// e.g. the campfire test program) fall back to the procedural soft circle.

layout(set = 1, binding = 0) uniform sampler2D uTextures[];

layout(location = 0) in vec4      fragColor;
layout(location = 1) in vec2      fragUV;
layout(location = 2) flat in int  fragLayer;

layout(location = 0) out vec4 outColor;

void main()
{
    if (fragLayer < 0) {
        // Procedural soft circle (no sprite texture for this program).
        vec2  uv_c  = fragUV - 0.5;
        float alpha = 1.0 - smoothstep(0.3, 0.5, length(uv_c));
        if (alpha < 0.01) discard;
        outColor = vec4(fragColor.rgb, fragColor.a * alpha);
        return;
    }

    vec4 tex = texture(uTextures[nonuniformEXT(fragLayer)], fragUV);
    vec4 c   = tex * fragColor;          // modulate sprite by per-particle colour/alpha
    if (c.a < 0.003) discard;
    outColor = c;
}
