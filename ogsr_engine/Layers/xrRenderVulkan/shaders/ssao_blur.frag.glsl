#version 450
// xrRenderVulkan — depth-aware 3×3 blur for the half-res GTAO output.
//
// GTAO is computed with per-pixel spatial noise (4-phase offsets + IGN
// directions) — without a blur the raw output is visibly dithered. The 3×3
// kernel weights taps by relative view-depth difference so AO never bleeds
// across silhouette edges (a corner's darkness must not leak onto the wall
// 10 m behind it).

layout(location = 0) out vec4 outAO;   // r = AO, gba = world bent normal *0.5+0.5
layout(location = 1) out vec4 outIL;   // rgb = blurred SSIL indirect light

layout(set = 0, binding = 0) uniform sampler2D uDepth;   // full-res scene depth
layout(set = 0, binding = 1) uniform sampler2D uAO;      // half-res raw GTAO (AO + bent normal)
layout(set = 0, binding = 4) uniform sampler2D uILraw;   // half-res raw SSIL (from the GTAO horizon gather)
layout(set = 0, binding = 5) uniform sampler2D uMV;      // full-res motion vectors (prevUV − curUV, UV space) — temporal reprojection
layout(set = 0, binding = 6) uniform sampler2D uAOhist;  // half-res PREV-frame final AO (temporal history)
layout(set = 0, binding = 7) uniform sampler2D uILhist;  // half-res PREV-frame final IL (temporal history)

layout(push_constant) uniform PC {
    // FULL SSAOPush layout — the C++ side pushes the same 112-byte block for
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
    vec4 temporal;   // x = EMA α (0 = off); y = jitter phase (unused here); z = MV valid; w = history valid
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

    // AO + bent normal: tight 3×3 (preserves contact-shadow detail).
    float sum = 0.0;
    float wsum = 0.0;
    vec3  bentSum = vec3(0.0);   // accumulate the world bent normal (decoded to [-1,1])
    float aoMin = 1.0, aoMax = 0.0;   // current-frame neighbourhood bounds (temporal anti-ghost clamp)
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
            aoMin = min(aoMin, t.r); aoMax = max(aoMax, t.r);
        }

    // SSIL: wider 5×5 depth-aware blur. The horizon gather (4 slices, half-res)
    // is far noisier/more banded than AO, so a broader kernel resolves the
    // directional structure ("stripes") into a smooth indirect fill.
    vec3  ilSum = vec3(0.0);
    float ilW   = 0.0;
    vec3  ilMin = vec3(1e9), ilMax = vec3(0.0);   // IL neighbourhood bounds (temporal anti-ghost clamp)
    for (int y = -2; y <= 2; ++y)
        for (int x = -2; x <= 2; ++x) {
            vec2 o  = vec2(x, y) * pc.res.zw;
            float w = exp(-abs(viewDepth(uv + o) - z0) * 16.0 / max(z0, 0.1));
            vec3 il = texture(uILraw, uv + o).rgb;
            ilSum += il * w;
            ilW   += w;
            ilMin = min(ilMin, il); ilMax = max(ilMax, il);
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
    vec3  ilOut = ilSum / max(ilW, 1e-4);

    // ── Temporal accumulation (r_ssil_temporal) ──────────────────────────────
    // Reproject the previous frame's final AO/IL through the motion vectors and
    // EMA-blend. The gather is jittered per frame (ssao.frag), so the history is a
    // DIFFERENT realisation of the same signal → averaging removes the directional
    // banding the spatial blur alone leaves. α=0 (off) or no valid history → pure
    // spatial result, identical to before. Neighbourhood-clamping the history to
    // this frame's local min/max bounds ghosting on motion/disocclusion; an off-
    // screen reprojection falls back to the current frame.
    float alpha = pc.temporal.x;
    if (alpha > 0.0 && pc.temporal.w > 0.5) {
        vec2 mv     = (pc.temporal.z > 0.5) ? texture(uMV, uv).rg : vec2(0.0);
        vec2 prevUV = uv + mv;
        if (all(greaterThanEqual(prevUV, vec2(0.0))) && all(lessThanEqual(prevUV, vec2(1.0)))) {
            // RELAXED neighbourhood clamp for AO: a TIGHT min/max clamp re-introduces
            // the very low-frequency structure temporal is meant to average out (it
            // drags the smooth history back into the current jittered frame's narrow
            // local range every frame). Add a slack margin (~1× the local range,
            // floored) so the EMA can average broad structure while still bounding
            // gross ghosts on disocclusion/motion.
            float aoRange = max(aoMax - aoMin, 0.05);
            float aoH = clamp(texture(uAOhist, prevUV).r, aoMin - aoRange, aoMax + aoRange);
            ao = mix(ao, aoH, alpha);
            vec3 ilH = clamp(texture(uILhist, prevUV).rgb, ilMin, ilMax);
            ilOut = mix(ilOut, ilH, alpha);
        }
    }

    outAO = vec4(ao, bentN * 0.5 + 0.5);
    outIL = vec4(ilOut, 1.0);
}
