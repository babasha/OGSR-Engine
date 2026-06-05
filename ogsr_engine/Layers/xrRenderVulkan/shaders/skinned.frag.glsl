#version 450
// skinned.frag — forward skinned shading with the diffuse texture.
// The leaf's WorldMaterial descriptor set (base/detail/lmap) is bound as set 1;
// we sample the base diffuse (binding 0) and apply a simple hemi+directional light.
// vertHW UVs are raw floats (no SHORT2 scale), so v_uv is used directly.

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec3 v_nrm;

layout(location = 0) out vec4 o_color;

layout(set = 1, binding = 0) uniform sampler2D uTexDiffuse;

void main()
{
    vec4 base = texture(uTexDiffuse, v_uv);
    if (base.a < 0.25)            // alpha-tested skinned parts (straps, hair, foliage)
        discard;

    vec3 N = normalize(v_nrm);
    float ndl = clamp(dot(N, normalize(vec3(0.3, 1.0, 0.4))), 0.0, 1.0);
    float l = ndl * 0.6 + 0.5;    // directional + ambient

    o_color = vec4(base.rgb * l, 1.0);
}
