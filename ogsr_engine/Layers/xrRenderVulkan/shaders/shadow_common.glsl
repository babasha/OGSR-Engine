// xrRenderVulkan - shared sun/spot/point shadow sampling. Was copy-pasted into
// world_lmap / world_terrain / world_vlit. #include AFTER light_ubo.glsl (reads
// L + the set-1 shadow samplers). cascTap is also used by rainVis (env_common),
// so include shadow_common BEFORE env_common.
#ifndef SHADOW_COMMON_GLSL
#define SHADOW_COMMON_GLSL

// Spot shadow: project by spot_vp, 3x3 PCF manual compare (flashlight quality).
// Compared in LINEAR depth with a WORLD-space epsilon: a constant NDC bias on
// the perspective spot projection is worth centimetres near the lamp but
// METRES near the far plane — light leaked straight through fences standing a
// few metres past a parked headlight.
float spotLinZ(float zndc, float f)
{
    const float n = 0.5;   // ComputeSpotVP near plane
    return n * f / max(f - zndc * (f - n), 1e-4);
}
float spotShadowF(vec3 wp, float range)
{
    vec4 c = L.spot_vp * vec4(wp, 1.0);
    if (c.w <= 0.0) return 1.0;
    vec3 ndc = c.xyz / c.w;
    vec2 uv = ndc.xy * 0.5 + 0.5;
    uv.y = 1.0 - uv.y;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z > 1.0) return 1.0;
    float f    = max(range, 1.0);
    float zRef = spotLinZ(ndc.z, f) - 0.08;   // 8 cm world bias, range-independent
    vec2  texel = 1.0 / vec2(textureSize(uSpotShadow, 0));
    float sum = 0.0;
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x)
            sum += (zRef <= spotLinZ(texture(uSpotShadow, uv + vec2(x, y) * texel).r, f)) ? 1.0 : 0.0;
    float vis = sum * (1.0 / 9.0);
    // Grass shadows surfaces PARTIALLY: a second tap set against the spot+grass
    // beam map (b22), blended by L.spot_params.x. The beam map is a superset of
    // the clean one, so its visibility is <= clean — the mix darkens the pool
    // with translucent grass dapples instead of the binary blanket (which ate
    // the headlight's ground pool at full strength) or nothing (sterile pool).
    float k = L.spot_params.x;
    if (k > 0.001) {
        float sumG = 0.0;
        for (int y = -1; y <= 1; ++y)
            for (int x = -1; x <= 1; ++x)
                sumG += (zRef <= spotLinZ(texture(uSpotShadowGrass, uv + vec2(x, y) * texel).r, f)) ? 1.0 : 0.0;
        vis = mix(vis, sumG * (1.0 / 9.0), k);
    }
    return vis;
}

// Point (cube) shadow: 1-tap, compare D3D-style perspective depth along the major
// axis of the lookup vector (faces rendered at 90 deg with near 0.1).
float pointShadowF(vec3 wp, vec3 lp, float range)
{
    vec3 d = wp - lp;
    float z = max(max(abs(d.x), abs(d.y)), abs(d.z));
    const float n = 0.1;
    float refD = range * (z - n) / (max(z, n) * max(range - n, 1e-3));
    return (refD - 0.01 <= texture(uPointShadow, d).r) ? 1.0 : 0.0;
}

// Bilinear-weighted PCF tap on a manual-compare map: textureGather fetches the
// 2x2 quad, each texel is COMPARED, then the binary results blend with the
// bilinear weights -> smooth gradient with no texel stair-stepping.
float cascTap(sampler2D smap, vec2 uv, float ref)
{
    vec2 sz = vec2(textureSize(smap, 0));
    vec2 t  = uv * sz - 0.5;
    vec2 f  = fract(t);
    vec4 d  = textureGather(smap, (floor(t) + 1.0) / sz, 0);  // w=(0,0) z=(1,0) x=(0,1) y=(1,1)
    vec4 c  = step(vec4(ref), d);                             // 1 = lit
    return mix(mix(c.w, c.z, f.x), mix(c.x, c.y, f.x), f.y);
}

// One sun-cascade lookup: 2x2 spread of bilinear gather taps ~ a 3x3 smooth
// kernel. Returns the lit factor, or -1.0 when worldPos falls outside this
// cascade (1% UV inset: border bilinear taps would mix unrendered texels).
float cascSample(sampler2D smap, mat4 vp, vec3 wp, float bias_)
{
    vec3 n = (vp * vec4(wp, 1.0)).xyz;          // ortho -> already NDC (w == 1)
    vec2 uv = n.xy * 0.5 + 0.5;
    uv.y = 1.0 - uv.y;
    if (uv.x < 0.01 || uv.x > 0.99 || uv.y < 0.01 || uv.y > 0.99
        || n.z <= 0.0 || n.z >= 1.0)
        return -1.0;
    float ref = n.z - bias_;
    vec2  tx  = 1.0 / vec2(textureSize(smap, 0));
    return 0.25 * (cascTap(smap, uv + vec2(-0.5, -0.5) * tx, ref)
                 + cascTap(smap, uv + vec2( 0.5, -0.5) * tx, ref)
                 + cascTap(smap, uv + vec2(-0.5,  0.5) * tx, ref)
                 + cascTap(smap, uv + vec2( 0.5,  0.5) * tx, ref));
}

// Sun shadow, R4 cascade scheme: cascade 0 (25 m) -> cascade 1 (60 m) -> the
// cached far map (320 m). Returns lit factor 1=lit .. 0=fully shadowed.
// Out-of-map -> lit.
float sunShadow(vec3 worldPos)
{
    float s = cascSample(uShadowNear, L.sun_near_vp, worldPos, 0.0004);
    if (s >= 0.0) return s;
    s = cascSample(uShadowC1, L.sun_c1_vp, worldPos, 0.0006);
    if (s >= 0.0) return s;

    vec4 c = L.sun_vp * vec4(worldPos, 1.0);
    if (c.w <= 0.0) return 1.0;
    vec3 ndc = c.xyz / c.w;
    vec2 uv = ndc.xy * 0.5 + 0.5;
    uv.y = 1.0 - uv.y;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z > 1.0) return 1.0;
    float ref   = ndc.z - 0.0015;
    // 4 spread taps (was 3x3): world pixels dominate the frame.
    vec2  texel = 1.0 / vec2(textureSize(uShadow, 0));
    float sum = 0.0;
    sum += (ref <= texture(uShadow, uv + vec2(-0.75, -0.75) * texel).r) ? 1.0 : 0.0;
    sum += (ref <= texture(uShadow, uv + vec2( 0.75, -0.75) * texel).r) ? 1.0 : 0.0;
    sum += (ref <= texture(uShadow, uv + vec2(-0.75,  0.75) * texel).r) ? 1.0 : 0.0;
    sum += (ref <= texture(uShadow, uv + vec2( 0.75,  0.75) * texel).r) ? 1.0 : 0.0;
    return sum * 0.25;
}

#endif // SHADOW_COMMON_GLSL
