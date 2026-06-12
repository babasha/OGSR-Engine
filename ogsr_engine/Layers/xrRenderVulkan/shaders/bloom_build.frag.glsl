#version 450
// Bloom bright-pass (R4 bloom_build.ps port). Samples the HDR scene at quarter
// resolution (the mip chain is already built for auto-exposure), applies the
// SAME auto-exposure the tonemap will use (R4's tonemap_high multiplies the
// s_tonemap scale at build time), then keeps only what exceeds the threshold —
// that's the energy that will glow. Soft knee so the cutoff doesn't shimmer.

layout(set = 0, binding = 0) uniform sampler2D uHDR;   // scene, full mip chain

layout(push_constant) uniform PC {
    vec4 p0;   // x=quarterLod, y=topMipLOD, z=middleGray, w=lowLum
    vec4 p1;   // x=expMin, y=expMax, z=threshold, w=knee
} pc;

layout(location = 0) out vec4 outColor;

const vec3 LUM = vec3(0.2126, 0.7152, 0.0722);

void main()
{
    vec2 sz = vec2(textureSize(uHDR, int(pc.p0.x + 0.5)));
    vec2 uv = gl_FragCoord.xy / sz;

    // Same R4 auto-exposure the tonemap applies (top mip = frame average).
    vec3  avg    = textureLod(uHDR, vec2(0.5), pc.p0.y).rgb;
    float avgLum = max(dot(avg, LUM), 1e-4);
    float exposure = clamp(pc.p0.z / (avgLum + pc.p0.w), pc.p1.x, pc.p1.y);

    vec3 c = textureLod(uHDR, uv, pc.p0.x).rgb * exposure;

    // Soft-knee bright pass: zero below (threshold-knee), quadratic ramp
    // through the knee, linear above. Keeps the bloom source stable.
    float lum  = dot(c, LUM);
    float thr  = pc.p1.z;
    float knee = pc.p1.w;
    float soft = clamp(lum - thr + knee, 0.0, 2.0 * knee);
    soft = soft * soft / (4.0 * knee + 1e-4);
    float contrib = max(soft, lum - thr) / max(lum, 1e-4);

    outColor = vec4(c * contrib, 1.0);
}
