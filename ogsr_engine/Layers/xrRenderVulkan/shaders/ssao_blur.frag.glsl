#version 450
// xrRenderVulkan — depth-aware 3×3 blur for the half-res GTAO output.
//
// GTAO is computed with per-pixel spatial noise (4-phase offsets + IGN
// directions) — without a blur the raw output is visibly dithered. The 3×3
// kernel weights taps by relative view-depth difference so AO never bleeds
// across silhouette edges (a corner's darkness must not leak onto the wall
// 10 m behind it).

layout(location = 0) out vec4 outAO;   // r = AO, gba = world bent normal *0.5+0.5

layout(set = 0, binding = 0) uniform sampler2D uDepth;   // full-res scene depth
layout(set = 0, binding = 1) uniform sampler2D uAO;      // half-res raw GTAO

layout(push_constant) uniform PC {
    // FULL SSAOPush layout — the C++ side pushes the same 96-byte block for
    // both the GTAO and blur pipelines, so the offsets here MUST match
    // ssao.frag (declaring only {zp, res} made zp alias camDir and res alias
    // camRightT → uv exploded past [0,1] → the whole output collapsed to one
    // clamped edge texel; that was the "AO map is uniformly flat" bug).
    vec4 camDir;     // unused here
    vec4 camRightT;  // unused here
    vec4 camTopT;    // unused here
    vec4 zp;         // x = proj _33, y = proj _43
    vec4 res;        // xy = AO target size, zw = 1 / AO target size
    vec4 dbg;        // unused here
} pc;

float viewDepth(vec2 uv)
{
    float zndc = texture(uDepth, uv).r;
    return clamp(pc.zp.y / (zndc - pc.zp.x), 0.0, 10000.0);
}

void main()
{
    vec2 uv = gl_FragCoord.xy * pc.res.zw;
    float z0 = viewDepth(uv);

    float sum = 0.0;
    float wsum = 0.0;
    vec3  bentSum = vec3(0.0);   // accumulate the world bent normal (decoded to [-1,1])
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x) {
            vec2 o  = vec2(x, y) * pc.res.zw;
            float zi = viewDepth(uv + o);
            // Relative depth gate: ~5% depth difference ≈ weight 0.45.
            float w  = exp(-abs(zi - z0) * 16.0 / max(z0, 0.1));
            vec4 t = texture(uAO, uv + o);
            sum  += t.r * w;
            bentSum += (t.gba * 2.0 - 1.0) * w;
            wsum += w;
        }

    // ±half-LSB IGN dither: the R8 target has 256 levels, and a LONG smooth
    // gradient (e.g. a straight fence's AO falloff on the ground) quantizes
    // into visible contour stripes — especially after the receivers' strength
    // pow. The dither breaks the contours into sub-pixel noise the bilinear
    // upsample averages away.
    // IGN dither constants, named to this build's provenance (mirror of
    // ogsr::sig) — identical values, just author-bound.
    const float IGN_MARIA    = 52.9829189;
    const vec2  IGN_BLUMENAU = vec2(0.06711056, 0.00583715);
    float dith = fract(IGN_MARIA * fract(dot(gl_FragCoord.xy, IGN_BLUMENAU)));
    float ao = sum / max(wsum, 1e-4) + (dith - 0.5) / 255.0;
    vec3  bentN = (dot(bentSum, bentSum) > 1e-6) ? normalize(bentSum) : vec3(0.0);
    outAO = vec4(ao, bentN * 0.5 + 0.5);
}
