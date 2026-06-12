#version 450
// xrRenderVulkan — volumetric sun shafts (god rays).
//
// Raymarch the view ray from the camera to the scene depth, sampling the SUN
// SHADOW MAP at each step: air that "sees" the sun glows with the sun colour,
// air in shadow doesn't → real light shafts cut by trees / window frames /
// buildings (better than R2's screen-space radial blur — works with the sun
// off-screen and respects actual occluders via the shadow map).
//
// World reconstruction is matrix-inverse-free: the camera frustum ray basis
// (dir + right·tanX·ndc.x + top·tanY·ndc.y) scaled by view-space depth
// recovered from the D3D projection terms (zview = _43 / (zndc - _33)).

layout(location = 0) in  vec2 vUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D uDepth;   // scene depth (D32)

// Shared per-frame env lighting — prefix of the full Lighting block.
layout(set = 1, binding = 0) uniform Lighting {
    vec4 sun_dir;     // xyz = travel dir (downward)
    vec4 sun_color;   // rgb
    vec4 hemi_color;
    vec4 ambient;
    mat4 sun_vp;      // sun light view·proj (shadow lookup)
} L;
layout(set = 1, binding = 1) uniform sampler2D uShadow;   // sun shadow map

layout(push_constant) uniform PC {
    vec4 camPos;      // xyz = camera position
    vec4 camDir;      // xyz = camera forward (unit)
    vec4 camRightT;   // xyz = camera right * tan(fovX/2)
    vec4 camTopT;     // xyz = camera up    * tan(fovY/2)
    vec4 zp;          // x = proj _33, y = proj _43, z = density, w = max march distance
} pc;

// 1-tap sun visibility at a world point (same projection as the world shaders).
float sunVis(vec3 p)
{
    vec4 c = L.sun_vp * vec4(p, 1.0);
    if (c.w <= 0.0) return 1.0;
    vec3 ndc = c.xyz / c.w;
    vec2 uv = ndc.xy * 0.5 + 0.5;
    uv.y = 1.0 - uv.y;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z > 1.0) return 1.0;
    return (ndc.z - 0.0015 <= texture(uShadow, uv).r) ? 1.0 : 0.0;
}

void main()
{
    // Scene depth → view-space distance → world position (frustum-ray basis).
    float zndc  = texture(uDepth, vUV).r;
    float zview = clamp(pc.zp.y / (zndc - pc.zp.x), 0.0, 10000.0);
    vec2  ndcXY = vec2(vUV.x * 2.0 - 1.0, 1.0 - 2.0 * vUV.y);   // D3D ndc (y up)
    vec3  ray   = pc.camDir.xyz + pc.camRightT.xyz * ndcXY.x + pc.camTopT.xyz * ndcXY.y;
    vec3  ro    = pc.camPos.xyz;
    vec3  wp    = ro + ray * zview;

    vec3  rd   = wp - ro;
    float dist = length(rd);
    rd /= max(dist, 1e-4);
    float tmax = min(dist, pc.zp.w);

    // March with per-pixel dither (interleaved gradient noise) to hide banding.
    const int STEPS  = 12;
    float stepLen = tmax / float(STEPS);
    float dith    = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
    float t   = stepLen * (0.5 + dith);
    float acc = 0.0;
    for (int i = 0; i < STEPS; ++i, t += stepLen)
        acc += sunVis(ro + rd * t);
    // Normalize by step count and by how much of the full march length this ray
    // actually covered (short rays accumulate proportionally less glow).
    acc *= (1.0 / float(STEPS)) * (tmax / pc.zp.w);

    // Forward-scattering phase: shafts are strongest looking toward the sun.
    float ph  = pow(max(dot(rd, normalize(-L.sun_dir.xyz)), 0.0), 6.0);
    vec3  col = L.sun_color.rgb * (acc * pc.zp.z * (0.15 + 0.85 * ph));

    outColor = vec4(col, 0.0);   // additive blend (ONE, ONE)
}
