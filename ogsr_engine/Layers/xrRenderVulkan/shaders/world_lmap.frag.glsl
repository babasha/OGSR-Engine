#version 450

// World pass — lmap variant. Final colour = albedo × baked lightmap.
//
// Albedo path is identical to the unlit variant: R4-style detail
// modulation `2 * base * detail`. Lightmap modulates this by the
// pre-baked sun + hemi colour the level compiler stored. Sun mask
// (lm.a) is currently unused — proper sun integration needs runtime
// sun direction & colour from the env subsystem.
//
// Without env wiring, sampling the lightmap RGB and multiplying gets
// the bulk of the visual win: baked shadows, bounce, ambient gradients.

layout(set = 0, binding = 0) uniform sampler2D uTexDiffuse;
layout(set = 0, binding = 1) uniform sampler2D uTexDetail;
layout(set = 0, binding = 2) uniform sampler2D uTexLmap;

// Per-frame environment lighting (set 1) — same UBO the skinned pass reads, so
// statics track time-of-day and match the NPCs. See vk_env_light.{h,cpp}.
struct DynLight {
    vec4 pos;     // xyz = world position, w = range
    vec4 color;   // rgb = colour,         w = 1 spot / 0 point
    vec4 dir;     // xyz = spot direction, w = cos(cone/2)
};
layout(set = 1, binding = 0) uniform Lighting {
    vec4 sun_dir;     // xyz = travel dir (downward); unused here (sun is baked via lm.a)
    vec4 sun_color;   // rgb
    vec4 hemi_color;  // rgb
    vec4 ambient;     // rgb
    mat4 sun_vp;      // sun light view·proj (shadow lookup)
    vec4 counts;      // x = dynamic light count
    DynLight lights[16];
    mat4 spot_vp;        // spot (flashlight) shadow view·proj
    vec4 shadow_params;  // x = spot-shadowed light index (-1 none), y = point-shadowed index
    mat4 sun_near_vp;    // sun cascade 0 view·proj (25 m, per-frame, R4 scheme)
    mat4 sun_c1_vp;      // sun cascade 1 view·proj (60 m, per-frame)
    vec4 fog_color;      // rgb haze colour (env)
    vec4 fog_params;     // x=-near*r, y=near, z=far, w=r; fog = saturate(dist*w + x)
    vec4 eye_pos;        // xyz camera world pos
    vec4 sky_params;     // x=cube cross-fade weight, y=ambient scale, z=sample LOD
    vec4 ao_params;      // x=1/screenW, y=1/screenH, z=AO strength (0=off)
} L;
layout(set = 1, binding = 1) uniform sampler2D uShadow;        // far sun map (320 m, cached)
layout(set = 1, binding = 2) uniform sampler2D uSpotShadow;    // spot (flashlight) shadow map
layout(set = 1, binding = 3) uniform samplerCube uPointShadow; // point (campfire) shadow cube
layout(set = 1, binding = 4) uniform sampler2D uShadowNear;    // sun cascade 0 (~0.61 cm texels)
layout(set = 1, binding = 5) uniform sampler2D uShadowC1;      // sun cascade 1 (~1.46 cm texels)
layout(set = 1, binding = 6) uniform samplerCube uSky0;        // sky ambient cube 0 (weather A)
layout(set = 1, binding = 7) uniform samplerCube uSky1;        // sky ambient cube 1 (weather B)
layout(set = 1, binding = 8) uniform sampler2D uAO;            // GTAO (half-res, bilinear upsample)

// GTAO visibility at this pixel (R4 combine_1.ps: occludes hemi+ambient only —
// never the sun or dynamic lights). Strength is an EXPONENT: 0 = off (→1.0),
// 1 = raw GTAO, 2-3 deepens corners without clipping (pow keeps 1 at 1).
float gtaoVis()
{
    float ao = textureLod(uAO, gl_FragCoord.xy * L.ao_params.xy, 0.0).r;
    return pow(clamp(ao, 0.0, 1.0), L.ao_params.z);
}

// Colored AO (R4 common_functions.h / Activision SIGGRAPH'16): mid-range
// occlusion bends toward the albedo colour — occluded light arrives as
// albedo-tinted bounce instead of going straight to grey. Full open (ao=1)
// and full black (ao=0) are unchanged.
vec3 coloredAO(float ao, vec3 albedo)
{
    vec3 a =  2.0404 * albedo - 0.3324;
    vec3 b = -4.7951 * albedo + 0.6417;
    vec3 c =  2.7552 * albedo + 0.6903;
    return max(vec3(ao), ((ao * a + b) * ao + c) * ao);
}

// Hemisphere sky ambient (R4 hmodel.h): the actual sky colour in the surface's
// world-normal direction, sampled at a blurred high mip (≈ diffuse irradiance),
// cross-fading the two weather cubes. This is the fill light that lights the
// whole street — surfaces the sun never reaches still see the sky.
vec3 skyAmbient(vec3 N)
{
    float lod = L.sky_params.z;
    vec3 a = textureLod(uSky0, N, lod).rgb;
    vec3 b = textureLod(uSky1, N, lod).rgb;
    return mix(a, b, clamp(L.sky_params.x, 0.0, 1.0));
}

// Spot shadow: project by spot_vp, 3×3 PCF manual compare (flashlight quality).
float spotShadowF(vec3 wp)
{
    vec4 c = L.spot_vp * vec4(wp, 1.0);
    if (c.w <= 0.0) return 1.0;
    vec3 ndc = c.xyz / c.w;
    vec2 uv = ndc.xy * 0.5 + 0.5;
    uv.y = 1.0 - uv.y;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z > 1.0) return 1.0;
    float ref   = ndc.z - 0.002;
    vec2  texel = 1.0 / vec2(textureSize(uSpotShadow, 0));
    float sum = 0.0;
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x)
            sum += (ref <= texture(uSpotShadow, uv + vec2(x, y) * texel).r) ? 1.0 : 0.0;
    return sum * (1.0 / 9.0);
}

// Point (cube) shadow: 1-tap, compare D3D-style perspective depth along the
// major axis of the lookup vector (faces rendered at 90° with near 0.1).
float pointShadowF(vec3 wp, vec3 lp, float range)
{
    vec3 d = wp - lp;
    float z = max(max(abs(d.x), abs(d.y)), abs(d.z));
    const float n = 0.1;
    float refD = range * (z - n) / (max(z, n) * max(range - n, 1e-3));
    return (refD - 0.01 <= texture(uPointShadow, d).r) ? 1.0 : 0.0;
}

// Dynamic point/spot light accumulation (STEP 3): linear-squared falloff,
// N·L diffuse, smooth spot cone. Range check first — most pixels exit early.
// The two shadow-budget lights additionally sample their shadow maps.
vec3 dynLights(vec3 wp, vec3 N)
{
    vec3 acc = vec3(0.0);
    int n = int(L.counts.x + 0.5);
    int sIdx = int(L.shadow_params.x);
    int pIdx = int(L.shadow_params.y);
    for (int i = 0; i < n; ++i) {
        vec3  dv = L.lights[i].pos.xyz - wp;
        float r  = L.lights[i].pos.w;
        float d2 = dot(dv, dv);
        if (d2 >= r * r) continue;
        float d   = sqrt(max(d2, 1e-6));
        vec3  ld  = dv / d;
        float att = 1.0 - d / r;
        att *= att;
        if (L.lights[i].color.w > 0.5)   // spot cone
            att *= clamp((dot(-ld, L.lights[i].dir.xyz) - L.lights[i].dir.w)
                         / max(1.0 - L.lights[i].dir.w, 1e-3), 0.0, 1.0);
        if (i == sIdx)      att *= spotShadowF(wp);
        else if (i == pIdx) att *= pointShadowF(wp, L.lights[i].pos.xyz, r);
        acc += L.lights[i].color.rgb * (att * max(dot(N, ld), 0.0));
    }
    return acc;
}

// Bilinear-weighted PCF tap (R4-style filtering on a manual-compare map):
// textureGather fetches the 2×2 quad, each texel is COMPARED, then the binary
// results blend with the bilinear weights → smooth gradient with no texel
// stair-stepping. (Comparing AFTER filtering, like texture().r does, is wrong —
// a filtered depth is a depth of nothing.)
float cascTap(sampler2D smap, vec2 uv, float ref)
{
    vec2 sz = vec2(textureSize(smap, 0));
    vec2 t  = uv * sz - 0.5;
    vec2 f  = fract(t);
    vec4 d  = textureGather(smap, (floor(t) + 1.0) / sz, 0);  // w=(0,0) z=(1,0) x=(0,1) y=(1,1)
    vec4 c  = step(vec4(ref), d);                             // 1 = lit
    return mix(mix(c.w, c.z, f.x), mix(c.x, c.y, f.x), f.y);
}

// One sun-cascade lookup: 2×2 spread of bilinear gather taps ≈ a 3×3 smooth
// kernel (~2 texels of penumbra — R4-class softness that also hides the tiny
// residual re-rasterization flicker). Returns the lit factor, or -1.0 when
// worldPos falls outside this cascade (1% UV inset: border bilinear taps would
// mix texels the cascade never rendered casters into).
float cascSample(sampler2D smap, mat4 vp, vec3 wp, float bias_)
{
    vec3 n = (vp * vec4(wp, 1.0)).xyz;          // ortho → already NDC (w == 1)
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

// Sun shadow lookup, R4 cascade scheme: cascade 0 (25 m, ~0.61 cm texels) →
// cascade 1 (60 m, ~1.46 cm) → the cached far map (320 m). Both near cascades
// re-render every frame with the continuous sun; stability comes from the
// world-anchored texel alignment on the CPU side (vk_shadow.cpp).
// Returns lit factor 1=lit … 0=fully shadowed. Out-of-map → lit.
float sunShadow(vec3 worldPos)
{
    float s = cascSample(uShadowNear, L.sun_near_vp, worldPos, 0.0004);
    if (s >= 0.0) return s;
    s = cascSample(uShadowC1, L.sun_c1_vp, worldPos, 0.0006);
    if (s >= 0.0) return s;

    vec4 c = L.sun_vp * vec4(worldPos, 1.0);    // row-major → GLSL transpose = X-Ray pos·M
    if (c.w <= 0.0) return 1.0;
    vec3 ndc = c.xyz / c.w;                      // xy [-1,1], z [0,1] (D3D ortho)
    vec2 uv = ndc.xy * 0.5 + 0.5;
    uv.y = 1.0 - uv.y;                           // shadow rendered with negative-height viewport
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z > 1.0) return 1.0;
    float ref   = ndc.z - 0.0015;                // depth bias (acne)
    // 4 spread taps (was 3×3): world pixels dominate the frame — half the taps,
    // visually close (sun shadows on statics mostly duplicate the bake anyway).
    vec2  texel = 1.0 / vec2(textureSize(uShadow, 0));
    float sum = 0.0;
    sum += (ref <= texture(uShadow, uv + vec2(-0.75, -0.75) * texel).r) ? 1.0 : 0.0;
    sum += (ref <= texture(uShadow, uv + vec2( 0.75, -0.75) * texel).r) ? 1.0 : 0.0;
    sum += (ref <= texture(uShadow, uv + vec2(-0.75,  0.75) * texel).r) ? 1.0 : 0.0;
    sum += (ref <= texture(uShadow, uv + vec2( 0.75,  0.75) * texel).r) ? 1.0 : 0.0;
    return sum * 0.25;
}

layout(push_constant) uniform PushConstants {
    mat4  mvp;
    vec2  uvScale;
    float alphaRef;
    float detailScale;
    float dynHemi;       // sky-ambient gate: 1.0 statics, ray-traced 0..1 for dynamics
} pc;

layout(location = 0) in  vec2 vUV;
layout(location = 1) in  vec2 vDetailUV;
layout(location = 2) in  vec2 vLmapUV;
layout(location = 3) in  vec3 vWorldPos;
layout(location = 4) in  vec3 vNormal;
layout(location = 0) out vec4 outColor;

void main()
{
    vec4 base   = texture(uTexDiffuse, vUV);
    if (pc.alphaRef >= 0.0 && base.a < pc.alphaRef) discard;

    // r_ssao_debug 1: show the raw AO map (corners/contacts should read dark).
    if (L.ao_params.w > 0.5) {
        outColor = vec4(vec3(textureLod(uAO, gl_FragCoord.xy * L.ao_params.xy, 0.0).r), base.a);
        return;
    }

    vec3 detail = texture(uTexDetail, vDetailUV).rgb;
    vec3 albedo = 2.0 * base.rgb * detail;

    // Lightmap keeps HEMI/AO duty only (lm.rgb → occlusion scalar). The SUN is
    // fully dynamic, R4-style: per-pixel N·L against the live sun direction ×
    // the shadow map. (The baked lm.a mask was R1-style — frozen at the bake's
    // sun angle, so a dawn/dusk sun never lit the walls facing it.)
    // hemiOcc ×2 stands in for the missing exposure/tonemap; small floor avoids
    // pitch-black where lm has no data.
    vec4  lm      = texture(uTexLmap, vLmapUV);
    float hemiOcc = dot(lm.rgb, vec3(1.0 / 3.0));
    vec3  Nw      = normalize(vNormal);
    float sunMask = max(dot(Nw, normalize(-L.sun_dir.xyz)), 0.0);
    if (sunMask > 0.005)
        sunMask *= sunShadow(vWorldPos);   // skip the PCF on back-facing pixels

    // Hemisphere sky fill (R4 hmodel) replaces the flat hemi term: real sky
    // colour per normal × lightmap occlusion. Sun ×1.25 — R4's tonemapped
    // pipeline reads a touch brighter in direct sun.
    vec3 occ = coloredAO(gtaoVis(), albedo);
    vec3 lighting = skyAmbient(Nw) * (hemiOcc * L.sky_params.y * pc.dynHemi) * occ
                  + L.sun_color.rgb  * (sunMask * 1.25)
                  + (L.ambient.rgb + 0.05) * occ
                  + dynLights(vWorldPos, Nw);

    // Distance fog (R4): fade to the env haze colour with view distance —
    // the "wet air" that sinks the far street into a light haze.
    vec3 col = albedo * lighting;
    float fog = clamp(length(vWorldPos - L.eye_pos.xyz) * L.fog_params.w + L.fog_params.x, 0.0, 1.0);
    col = mix(col, L.fog_color.rgb, fog);

    outColor = vec4(col, base.a);
}
