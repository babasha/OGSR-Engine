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
layout(set = 0, binding = 3) uniform sampler2D uTexBumpX;   // .a = height (POM); flat=1 → no parallax

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
    mat4 rain_vp;        // straight-down ortho VP for the rain occlusion map
    vec4 rain_params;    // x=rain density, y=wetness, z=darken, w=reflection scale
    mat4 scene_vp;       // (SSR puddles — declared for layout match, unused here)
    vec4 cam_dir;        //  "
    vec4 cam_rightT;     //  "
    vec4 cam_topT;       //  "
    vec4 pom_params;     // x=POM amplitude (UV), y=max steps, z=fade dist (m), w=on
    vec4 pom_params2;    // x=blur, y=normal, z=self-shadow, w=contact AO
    vec4 pom_params3;    // x=debug view (draw POM AO×self-shadow grayscale)
    vec4 pom_params4;    // x=terrain POM enable, y=detail-normal, z=micro-AO, w=debug (terrain only)
    vec4 pom_params5;    // x=terrain gloss, y=geo-puddle radius (0=off), z=geo-puddle depth scale, w=puddle debug
    vec4 pom_params6;    // x=water-sim enable (puddles from the flow sim)
} L;
layout(set = 1, binding = 1) uniform sampler2D uShadow;        // far sun map (320 m, cached)
layout(set = 1, binding = 2) uniform sampler2D uSpotShadow;    // spot (flashlight) shadow map
layout(set = 1, binding = 3) uniform samplerCube uPointShadow; // point (campfire) shadow cube
layout(set = 1, binding = 4) uniform sampler2D uShadowNear;    // sun cascade 0 (~0.61 cm texels)
layout(set = 1, binding = 5) uniform sampler2D uShadowC1;      // sun cascade 1 (~1.46 cm texels)
layout(set = 1, binding = 6) uniform samplerCube uSky0;        // sky ambient cube 0 (weather A)
layout(set = 1, binding = 7) uniform samplerCube uSky1;        // sky ambient cube 1 (weather B)
layout(set = 1, binding = 8) uniform sampler2D uAO;            // GTAO (half-res, bilinear upsample)
layout(set = 1, binding = 9) uniform sampler2D uRainMap;       // top-down rain occlusion (wetness mask)
layout(set = 1, binding = 10) uniform sampler2D uSpotCookie;   // flashlight beam texture (cookie)
layout(set = 1, binding = 11) uniform sampler2D uWater;        // water depth (flow sim, metres)
layout(set = 1, binding = 12) uniform sampler2D uFlow;         // water velocity (flow sim, uv/sec)

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
    float xf = clamp(L.sky_params.x, 0.0, 1.0);   // weather cross-fade — usually 0/1
    vec3 a = textureLod(uSky0, N, lod).rgb;
    return (xf > 0.01) ? mix(a, textureLod(uSky1, N, lod).rgb, xf) : a;  // 2nd cube only in transition
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
        vec3 tint = L.lights[i].color.rgb;
        if (i == sIdx) {
            att *= spotShadowF(wp);
            // Flashlight cookie (R4 projective light texture): the beam pattern
            // projected through the SAME spot_vp the shadow lookup uses.
            if (L.shadow_params.z > 0.5) {
                vec4 cc = L.spot_vp * vec4(wp, 1.0);
                if (cc.w > 0.0) {
                    vec2 cuv = (cc.xy / cc.w) * 0.5 + 0.5;
                    cuv.y = 1.0 - cuv.y;
                    tint *= textureLod(uSpotCookie, clamp(cuv, 0.0, 1.0), 0.0).rgb;
                }
            }
        }
        else if (i == pIdx) att *= pointShadowF(wp, L.lights[i].pos.xyz, r);
        acc += tint * (att * max(dot(N, ld), 0.0));
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

// Rain visibility: 1 = open to the sky (gets rained on), 0 = covered (roof,
// tunnel — stays dry). Bilinear-weighted compare (cascTap) on the straight-down
// rain map; outside the map = open sky (border depth 1.0 → lit).
float rainVis(vec3 wp)
{
    vec3 n = (L.rain_vp * vec4(wp, 1.0)).xyz;   // ortho → already NDC
    vec2 uv = n.xy * 0.5 + 0.5;
    uv.y = 1.0 - uv.y;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || n.z <= 0.0 || n.z >= 1.0)
        return 1.0;
    return cascTap(uRainMap, uv, n.z - 0.0015);
}

// Procedural puddle patches: low-frequency blobs over world XZ. A UNIFORM
// reflective film reads as candle wax — real rain collects in patches (the
// reflective puddles) while the rest of the ground just darkens.
float puddleMask(vec2 p)
{
    float n = sin(p.x * 0.71 + sin(p.y * 0.53) * 1.7)
            * sin(p.y * 0.67 + sin(p.x * 0.49) * 1.7);
    return smoothstep(0.15, 0.65, n * 0.5 + 0.5);
}

// GEOMETRIC puddles: water collects in REAL terrain depressions instead of the
// procedural blobs above. The rain occlusion map is a top-down ortho DEPTH of
// the scene (terrain + statics are rendered into it) = a height field — smaller
// depth = higher ground (the eye is above). Sampling a ring of neighbours gives
// the local rim/average; a pixel deeper than that sits in a dip and holds water.
// Reuses uRainMap + rain_vp, so no extra render target. Tree-occluded spots read
// a false rim but are dry there (rainVis↓), so the puddle is suppressed anyway.
float geoPuddle(vec3 wp)
{
    vec4 c = L.rain_vp * vec4(wp, 1.0);
    if (c.w <= 0.0) return 0.0;
    vec2 uv = c.xy * 0.5 + 0.5; uv.y = 1.0 - uv.y;
    if (uv.x < 0.03 || uv.x > 0.97 || uv.y < 0.03 || uv.y > 0.97) return 0.0;
    float myD = c.z;
    vec2  px  = 1.0 / vec2(textureSize(uRainMap, 0));
    float R   = L.pom_params5.y;                      // basin scale (outer ring radius, texels)
    vec2  o   = R * px, ii = (R * 0.5) * px;
    // Local "water level" = blurred ground height over a WIDE neighbourhood (two
    // rings → smoother). Smaller depth = higher ground; this average is the local
    // rim. A pixel BELOW the level is underwater → water fills valleys and the
    // low side of slopes while ridges/peaks (above the level) stay dry.
    float lvl =
        ( textureLod(uRainMap, uv + vec2( o.x, 0.0), 0.0).r
        + textureLod(uRainMap, uv + vec2(-o.x, 0.0), 0.0).r
        + textureLod(uRainMap, uv + vec2(0.0,  o.y), 0.0).r
        + textureLod(uRainMap, uv + vec2(0.0, -o.y), 0.0).r
        + textureLod(uRainMap, uv + vec2( o.x,  o.y), 0.0).r
        + textureLod(uRainMap, uv + vec2(-o.x,  o.y), 0.0).r
        + textureLod(uRainMap, uv + vec2( o.x, -o.y), 0.0).r
        + textureLod(uRainMap, uv + vec2(-o.x, -o.y), 0.0).r ) * (0.5 / 8.0)
      + ( textureLod(uRainMap, uv + vec2( ii.x, 0.0), 0.0).r
        + textureLod(uRainMap, uv + vec2(-ii.x, 0.0), 0.0).r
        + textureLod(uRainMap, uv + vec2(0.0,  ii.y), 0.0).r
        + textureLod(uRainMap, uv + vec2(0.0, -ii.y), 0.0).r ) * (0.5 / 4.0);
    // Below the local level → fills with water. Scale maps the ortho-depth delta
    // (1 m ≈ 1/349) to 0..1; the deadzone (0.08) kills flat-ground speckle.
    return smoothstep(0.08, 1.0, (myD - lvl) * L.pom_params5.z);
}

// Water DEPTH (metres) from the flow sim, sampled via rain_vp (sim grid aligns
// to the rain map). 0 where dry. See vk_water_sim.
float simWater(vec3 wp)
{
    vec4 c = L.rain_vp * vec4(wp, 1.0);
    if (c.w <= 0.0) return 0.0;
    vec2 uv = c.xy * 0.5 + 0.5; uv.y = 1.0 - uv.y;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return 0.0;
    return textureLod(uWater, uv, 0.0).r;
}

// Blurred water depth for puddle placement (spreads crease/seam line-pooling
// into smooth area puddles) — see world_terrain.frag.
float simWaterSoft(vec3 wp)
{
    vec4 c = L.rain_vp * vec4(wp, 1.0);
    if (c.w <= 0.0) return 0.0;
    vec2 uv = c.xy * 0.5 + 0.5; uv.y = 1.0 - uv.y;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return 0.0;
    vec2 px = 5.0 / vec2(textureSize(uWater, 0));
    return textureLod(uWater, uv, 0.0).r * 0.4
         + (textureLod(uWater, uv + vec2(px.x, 0.0), 0.0).r
          + textureLod(uWater, uv - vec2(px.x, 0.0), 0.0).r
          + textureLod(uWater, uv + vec2(0.0, px.y), 0.0).r
          + textureLod(uWater, uv - vec2(0.0, px.y), 0.0).r) * 0.15;
}

// Water VELOCITY (uv/sec) from the flow sim — drives the moving-water surface.
vec2 simFlow(vec3 wp)
{
    vec4 c = L.rain_vp * vec4(wp, 1.0);
    if (c.w <= 0.0) return vec2(0.0);
    vec2 uv = c.xy * 0.5 + 0.5; uv.y = 1.0 - uv.y;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return vec2(0.0);
    return vec2(textureLod(uFlow, uv, 0.0).r, -textureLod(uFlow, uv, 0.0).g);  // uv-vel -> world XZ (Z axis is flipped)
}

// Ground height (metres, relative) from the rain map ortho depth — flow debug.
float groundHm(vec3 wp)
{
    vec4 c = L.rain_vp * vec4(wp, 1.0);
    if (c.w <= 0.0) return 0.0;
    vec2 uv = c.xy * 0.5 + 0.5; uv.y = 1.0 - uv.y;
    return -textureLod(uRainMap, uv, 0.0).r * 349.0;
}

// Water debug colour. 1 = DEPTH ramp (dry≈black, shallow blue, deep cyan→white).
// 2 = FLOW direction (downhill = −gradient of the surface; dir→RG, speed→bright).
vec3 waterDebugColor(vec3 wp, int mode)
{
    float d = simWater(wp);
    if (mode == 2) {
        float e = 0.6;
        float sx1 = groundHm(wp + vec3( e,0,0)) + simWater(wp + vec3( e,0,0));
        float sx0 = groundHm(wp + vec3(-e,0,0)) + simWater(wp + vec3(-e,0,0));
        float sz1 = groundHm(wp + vec3(0,0, e)) + simWater(wp + vec3(0,0, e));
        float sz0 = groundHm(wp + vec3(0,0,-e)) + simWater(wp + vec3(0,0,-e));
        vec2 flow = -vec2(sx1 - sx0, sz1 - sz0);
        float sp  = clamp(length(flow) * 2.0, 0.0, 1.0);
        vec2 dir  = (length(flow) > 1e-5) ? normalize(flow) : vec2(0.0);
        vec3 c = vec3(dir * 0.5 + 0.5, 0.3) * sp;
        return (d > 0.005) ? c : c * 0.15;
    }
    float t = clamp(d / 0.6, 0.0, 1.0);
    vec3 c = vec3(0.0, t * 0.55, t) + vec3(smoothstep(0.75, 1.0, t));
    return (d < 0.005) ? vec3(0.02) : c;
}

// Travelling surface waves ALONG the water flow — see world_terrain.frag.
vec2 flowWaves(vec3 wp, float t)
{
    float d = simWater(wp);
    if (d < 0.003) return vec2(0.0);
    float e = 0.6;
    float sx1 = groundHm(wp + vec3( e,0,0)) + simWater(wp + vec3( e,0,0));
    float sx0 = groundHm(wp + vec3(-e,0,0)) + simWater(wp + vec3(-e,0,0));
    float sz1 = groundHm(wp + vec3(0,0, e)) + simWater(wp + vec3(0,0, e));
    float sz0 = groundHm(wp + vec3(0,0,-e)) + simWater(wp + vec3(0,0,-e));
    vec2 flow = -vec2(sx1 - sx0, sz1 - sz0);
    float spd = length(flow);
    if (spd < 1e-4) return vec2(0.0);
    vec2 dir = flow / spd;
    float along = dot(wp.xz, dir);
    float w = sin(along * 7.0  - t * (2.0 + spd * 30.0))
            + 0.5 * sin(along * 16.0 - t * (3.5 + spd * 50.0) + 1.3);
    float amp = clamp(spd * 6.0, 0.0, 1.0) * clamp(d * 8.0, 0.0, 1.0);
    return dir * (w * amp * 0.5);
}

// Lagarde ring ripples (SSFX screenspace_common_ripples.h scheme, procedural):
// each grid cell spawns one expanding, fading ring per cycle; three jittered
// layers hide the lattice. Returns an xz normal perturbation.
vec2 rippleLayer(vec2 p, float t)
{
    vec2 cell = floor(p);
    vec2 f = p - cell;
    // procedural-noise salts — build provenance (mirror of ogsr::sig); same
    // numbers a classic value-noise hash uses, just named/bound to this build.
    const vec2  SALT_ZEFIR   = vec2(127.1, 311.7);
    const vec2  SALT_CATARA  = vec2(269.5, 183.3);
    const float SALT_SARATOV = 43758.5453;
    float h1 = fract(sin(dot(cell, SALT_ZEFIR))  * SALT_SARATOV);
    float h2 = fract(sin(dot(cell, SALT_CATARA)) * SALT_SARATOV);
    vec2  c  = vec2(0.3) + 0.4 * vec2(h1, h2);
    float ph = fract(t + h1);                          // drop lifecycle 0..1
    float d  = length(f - c);
    float ring = sin(clamp((d - ph * 0.5) * 30.0, -3.1416, 3.1416));
    float fade = (1.0 - ph) * smoothstep(0.5, 0.25, d);
    return (d > 1e-4 ? (f - c) / d : vec2(0.0)) * (ring * fade);
}

vec2 rainRipples(vec2 p, float t)
{
    return rippleLayer(p * 2.2,                     t * 1.05)
         + rippleLayer(p * 1.34 + vec2(0.50, 0.25), t * 1.31)
         + rippleLayer(p * 1.91 + vec2(0.31, 0.50), t * 1.58);
}

// Wet-surface contribution: darkens the albedo in place and returns the sky
// reflection to ADD after the diffuse light is applied (pre-fog). Up-facing
// surfaces soak fully; walls pick up a weaker sheen.
vec3 applyWetness(inout vec3 albedo, vec3 wp, vec3 N)
{
    float wet = L.rain_params.y;
    if (wet < 0.005)
        return vec3(0.0);
    wet *= rainVis(wp);
    float upness = clamp(N.y, 0.0, 1.0);
    // Kill wetness on DOWN-facing surfaces (ceilings, undersides of overhangs /
    // doorframe tops) — rain can't land there. Stops "drops on the ceiling" from
    // thin roofs self-passing the rain occlusion test. Walls keep most of it.
    float wetK = wet * mix(0.35, 1.0, upness) * smoothstep(-0.15, 0.05, N.y);
    albedo *= 1.0 - L.rain_params.z * wetK;                       // мокрое темнее

    // PERF: everything below (ripples + sky-cube reflection + puddle sample) is
    // the expensive per-pixel wet work that dominated the World pass. Fade it by
    // distance (far wet ground is sub-pixel / fogged) and EARLY-OUT when it's
    // negligible — the bulk of the visible ground then costs only the darken above.
    vec3  toEye    = L.eye_pos.xyz - wp;
    float dist     = length(toEye);
    float reflFade = smoothstep(70.0, 35.0, dist);
    if (wetK * reflFade < 0.004) return vec3(0.0);

    // Ring ripples advected DOWNSTREAM by the flow velocity (moving water) + chop.
    float t = L.sky_params.w;
    float ripFade = smoothstep(18.0, 8.0, dist) * clamp(L.rain_params.x * 1.5 + 0.1, 0.0, 1.0);
    vec2  vel   = (L.pom_params6.x > 0.5) ? simFlow(wp) : vec2(0.0);
    float velMS = length(vel) * 150.0;
    vec2  scrl  = (velMS > 0.01) ? normalize(vel) * (t * velMS * 0.25) : vec2(0.0);
    vec3  Nr = N;
    if (ripFade > 0.01) {
        vec2 rip = rainRipples(wp.xz - scrl, t * 0.6) * (0.4 * ripFade);
        if (velMS > 0.1)
            rip += rainRipples(wp.xz * 1.6 - scrl * 1.6, t * 0.9) * (clamp(velMS * 0.12, 0.0, 0.5) * ripFade);
        Nr = normalize(vec3(N.x + rip.x, N.y, N.z + rip.y));
    }

    vec3 V = normalize(toEye);
    vec3 R = reflect(-V, Nr);
    float fres = pow(1.0 - clamp(dot(V, Nr), 0.0, 1.0), 3.0);
    float xf = clamp(L.sky_params.x, 0.0, 1.0);   // weather cross-fade — usually 0/1
    vec3 sky = textureLod(uSky0, R, 0.0).rgb;
    if (xf > 0.01) sky = mix(sky, textureLod(uSky1, R, 0.0).rgb, xf);  // 2nd cube only in transition
    float pud = ((L.pom_params6.x > 0.5) ? smoothstep(0.04, 0.12, simWaterSoft(wp))
                 : (L.pom_params5.y > 0.0) ? geoPuddle(wp) : puddleMask(wp.xz * 0.8)) * upness;
    // Deep water = clear water BODY: cool dark tint + stronger sharp reflection.
    albedo = mix(albedo, albedo * vec3(0.45, 0.55, 0.62), clamp(pud, 0.0, 1.0));
    float reflK = wetK * (0.1 + 1.0 * pud) * (0.04 + 0.96 * fres) * reflFade;
    float foam = smoothstep(3.0, 6.0, velMS) * clamp(pud, 0.0, 1.0) * ripFade * 0.25;
    return sky * (reflK * L.rain_params.w) + vec3(foam);
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

// Height from the `#` alpha (GEOMETRY-based, R4's displacement source — NOT
// albedo, which on multi-tone brickwork makes dark-vs-light bricks into false
// cliffs → spike fields). HIGH-PASSED against a blurred baseline: removes the
// low-frequency bias that would slide the whole wall, keeps only the detail
// (mortar grooves), centered at 0.5 so it carves both ways. gain deepens it.
float pomDepth(vec2 uv, float lod, float baseline)
{
    float h = textureLod(uTexBumpX, uv, lod).a;
    return clamp(0.5 - (h - baseline) * 4.0, 0.0, 1.0);
}

// Parallax occlusion mapping (relief mapping): steep linear march + binary
// refine, in tangent space, returns the parallax-offset base UV. Tangent frame
// derived per-pixel from screen-space derivatives (Schüler) — no per-vertex
// tangent, works through the tess pipeline. Gated to materials with a real `#`
// (flat fallback alpha=1 at every mip → skipped, so smooth surfaces are flat).
vec2 parallaxUV(vec2 uv, vec3 N, vec3 wp, out vec3 outN, out float outShadow, out float outAO)
{
    outN = N;          // default: unperturbed geometric normal (early-outs below)
    outShadow = 1.0;   // default: no self-shadow
    outAO = 1.0;       // default: no contact AO
    float amp = L.pom_params.x;
    if (amp <= 0.0) return uv;
    if (textureLod(uTexBumpX, vec2(0.5), 8.0).a > 0.985) return uv;   // flat material → no POM
    float dist = length(L.eye_pos.xyz - wp);
    // Well-defined fade (edge0 < edge1): full POM to far*0.5, gone by far.
    float fade = 1.0 - smoothstep(L.pom_params.z * 0.5, L.pom_params.z, dist);
    amp *= fade;
    // Per-orientation strength: walls/fences (N.y≈0) full; floors (N.y→+1) →
    // r_pom_floor; ceilings (N.y→-1) → r_pom_ceil. Scales the whole POM below.
    float orient = (N.y >= 0.0)
        ? mix(1.0, L.pom_params3.w, clamp( N.y, 0.0, 1.0))   // floor (up-facing)
        : mix(1.0, L.pom_params3.z, clamp(-N.y, 0.0, 1.0));  // ceiling (down-facing)
    amp *= orient;
    if (amp <= 1e-5) return uv;

    // Marching LOD = screen-correct (anti-shimmer) + r_pom_blur bias. The `#`
    // mortar detail lives only in LOW mips, so a heavy blur flattens the relief
    // within ~1 m; keep the bias small (r_pom_blur) so detail survives across
    // the fade range. Computed once before the loop (derivatives need uniform flow).
    vec2 tsz  = vec2(textureSize(uTexBumpX, 0));
    vec2 ddx  = dFdx(uv) * tsz, ddy = dFdy(uv) * tsz;
    float lod = max(0.5 * log2(max(dot(ddx, ddx), dot(ddy, ddy))), 0.0) + L.pom_params2.x;
    // Low-frequency baseline for the high-pass — sampled ONCE (it's smooth over
    // the small march offset). +3 mips below the marching LOD.
    float baseline = textureLod(uTexBumpX, uv, lod + 3.0).a;

    // Cotangent frame from world-pos + uv derivatives (maps tangent → world).
    vec3 dp1 = dFdx(wp),  dp2 = dFdy(wp);
    vec2 du1 = dFdx(uv),  du2 = dFdy(uv);
    vec3 dp2p = cross(dp2, N), dp1p = cross(N, dp1);
    vec3 T = dp2p * du1.x + dp1p * du2.x;
    vec3 B = dp2p * du1.y + dp1p * du2.y;
    float inv = inversesqrt(max(dot(T, T), dot(B, B)));
    T *= inv; B *= inv;

    // View dir (surface→eye) in tangent space; /z makes grazing angles dig more.
    vec3 V   = normalize(L.eye_pos.xyz - wp);
    vec3 Vts = vec3(dot(V, T), dot(V, B), dot(V, N));
    vec2 Pmax = (Vts.xy / max(abs(Vts.z), 0.3)) * amp;   // total UV shift at full depth

    // More steps at grazing angle (where parallax is strongest).
    int steps = int(clamp(mix(L.pom_params.y, 12.0, abs(Vts.z)), 12.0, 64.0));
    float layerH = 1.0 / float(steps);
    vec2 dUV = Pmax * layerH;

    // Steep linear march to the first layer below the surface.
    float curD = 0.0;
    vec2  curUV = uv;
    float curH = pomDepth(curUV, lod, baseline);
    for (int i = 0; i < 64; ++i) {
        if (i >= steps || curD >= curH) break;
        curUV -= dUV;
        curD  += layerH;
        curH   = pomDepth(curUV, lod, baseline);
    }
    // Binary-search refine (relief mapping): converge precisely on the
    // intersection — this is what removes the up-close stair-step shimmer.
    vec2 sUV = dUV; float sD = layerH;
    for (int j = 0; j < 6; ++j) {
        sUV *= 0.5; sD *= 0.5;
        if (curD < pomDepth(curUV, lod, baseline)) { curUV -= sUV; curD += sD; }
        else                                       { curUV += sUV; curD -= sD; }
    }

    // Perturbed normal from the height gradient at the hit point — so the
    // relief actually CATCHES LIGHT (sun/dyn/hemi). The `#` DC cancels in the
    // central difference, so raw alpha is fine. Strength = r_pom_normal, faded
    // with distance like the displacement. (The renderer has no other normal
    // mapping for world geometry — this is what makes POM read as 3D.)
    float tU = exp2(lod) / tsz.x, tV = exp2(lod) / tsz.y;
    float hu = textureLod(uTexBumpX, curUV + vec2(tU, 0.0), lod).a
             - textureLod(uTexBumpX, curUV - vec2(tU, 0.0), lod).a;
    float hv = textureLod(uTexBumpX, curUV + vec2(0.0, tV), lod).a
             - textureLod(uTexBumpX, curUV - vec2(0.0, tV), lod).a;
    float ns = L.pom_params2.y * 12.0 * fade * orient;
    vec3 nTS = normalize(vec3(-hu * ns, -hv * ns, 1.0));
    outN = normalize(T * nTS.x + B * nTS.y + N * nTS.z);

    // Self-shadow: march from the hit point toward the SUN through the height
    // field; where the relief rises above the light ray, the groove is in
    // contact shadow. Soft (deepest penetration). Strength = r_pom_shadow.
    if (L.pom_params2.z > 0.0) {
        vec3 Ld  = normalize(-L.sun_dir.xyz);   // to-sun (sun_dir travels downward)
        vec3 Lts = vec3(dot(Ld, T), dot(Ld, B), dot(Ld, N));
        vec2 lxy = Lts.xy;
        if (Lts.z > 0.02 && dot(lxy, lxy) > 1e-6) {
            // Horizon self-shadow: scan ~6 texels toward the sun's tangent
            // projection; a TALLER neighbour blocks the sun. Reach is a fixed
            // texel count, NOT the tiny displacement amp (that march was far too
            // short to reach the next mortar ridge → looked like nothing).
            // Stronger at grazing sun (longer shadows).
            vec2  sdir  = normalize(lxy);
            float reach = (exp2(lod) / min(tsz.x, tsz.y)) * 6.0;
            float h0    = textureLod(uTexBumpX, curUV, lod).a;
            float occ   = 0.0;
            for (int s = 1; s <= 8; ++s) {
                float hs = textureLod(uTexBumpX, curUV + sdir * reach * (float(s) * 0.125), lod).a;
                occ = max(occ, hs - h0);
            }
            outShadow = clamp(1.0 - occ * L.pom_params2.z * 20.0 * (1.0 - Lts.z) * orient, 0.0, 1.0);
        }
    }

    // POM-AO: view-INDEPENDENT contact occlusion from the heightfield — a pixel
    // surrounded by taller neighbours (a groove) is occluded. Darkens the
    // AMBIENT/hemi (not the sun) so the relief reads even out of direct light.
    // Strength = r_pom_ao; floored so grooves never go black.
    if (L.pom_params2.w > 0.0) {
        float aoReach = (exp2(lod) / min(tsz.x, tsz.y)) * 4.0;
        float h0 = textureLod(uTexBumpX, curUV, lod).a;
        float aoSum =
              max(0.0, textureLod(uTexBumpX, curUV + vec2( aoReach, 0.0), lod).a - h0)
            + max(0.0, textureLod(uTexBumpX, curUV + vec2(-aoReach, 0.0), lod).a - h0)
            + max(0.0, textureLod(uTexBumpX, curUV + vec2(0.0,  aoReach), lod).a - h0)
            + max(0.0, textureLod(uTexBumpX, curUV + vec2(0.0, -aoReach), lod).a - h0);
        outAO = clamp(1.0 - (aoSum * 0.25) * L.pom_params2.w * 6.0, 0.35, 1.0);
        outAO = mix(1.0, outAO, orient);   // ceilings: dial contact AO down too
    }
    return curUV;
}

void main()
{
    // POM: offset the base UV before sampling; shift the detail UV by the same
    // world-space delta. Lightmap UV is NOT parallaxed (low-freq baked light).
    vec3 pomN; float pomShadow, pomAO;
    vec2 pUV = parallaxUV(vUV, normalize(vNormal), vWorldPos, pomN, pomShadow, pomAO);
    vec2 pDetailUV = vDetailUV + (pUV - vUV) * pc.detailScale;

    vec4 base   = texture(uTexDiffuse, pUV);
    if (pc.alphaRef >= 0.0 && base.a < pc.alphaRef) discard;

    // r_ssao_debug 1: show the raw AO map (corners/contacts should read dark).
    if (L.ao_params.w > 0.5) {
        outColor = vec4(vec3(textureLod(uAO, gl_FragCoord.xy * L.ao_params.xy, 0.0).r), base.a);
        return;
    }

    // r_wet_debug 1 (darken arrives negative): rain-map visibility — white =
    // open sky (gets rained on), black = covered (roof/tunnel). The wetness
    // NUMBER is in the "[VK Rain] wet state" log line.
    if (L.rain_params.z < 0.0) {
        outColor = vec4(vec3(rainVis(vWorldPos)), base.a);
        return;
    }

    // r_puddle_debug: 1 = water depth (colour ramp), 2 = flow direction; sim off
    // → static geometric mask (grayscale).
    int pdbg = int(L.pom_params5.w + 0.5);
    if (pdbg > 0) {
        outColor = (L.pom_params6.x > 0.5)
            ? vec4(waterDebugColor(vWorldPos, pdbg), base.a)
            : vec4(vec3(geoPuddle(vWorldPos)), base.a);
        return;
    }

    // r_pom_debug 1: POM occlusion mask (contact AO × sun self-shadow) — white =
    // lit/open, dark = occluded grooves. (Set r_pom_ao 0 to see only the
    // self-shadow, or r_pom_shadow 0 to see only the contact AO.)
    if (L.pom_params3.x > 0.5) {
        outColor = vec4(vec3(pomAO * pomShadow), base.a);
        return;
    }

    vec3 detail = texture(uTexDetail, pDetailUV).rgb;
    vec3 albedo = 2.0 * base.rgb * detail;

    // Lightmap keeps HEMI/AO duty only (lm.rgb → occlusion scalar). The SUN is
    // fully dynamic, R4-style: per-pixel N·L against the live sun direction ×
    // the shadow map. (The baked lm.a mask was R1-style — frozen at the bake's
    // sun angle, so a dawn/dusk sun never lit the walls facing it.)
    // sun_color/ambient arrive FINAL from vk_env_light (r_sun_boost
    // premultiplied, r_ambient_floor added) — no brightness literals here.
    vec4  lm      = texture(uTexLmap, vLmapUV);
    float hemiOcc = dot(lm.rgb, vec3(1.0 / 3.0));
    vec3  geomN   = normalize(vNormal);   // flat — for the sky fill (sharp cube → perturbed = mirror)
    vec3  Nw      = pomN;   // POM-perturbed normal → sun + dyn lights catch the relief
    float sunMask = max(dot(Nw, normalize(-L.sun_dir.xyz)), 0.0);
    if (sunMask > 0.005)
        sunMask *= sunShadow(vWorldPos) * pomShadow;   // cascade shadow × POM groove self-shadow

    // Hemisphere sky fill (R4 hmodel) replaces the flat hemi term: real sky
    // colour per normal × lightmap occlusion.
    vec3  occ      = coloredAO(gtaoVis(), albedo) * pomAO;   // GTAO × fine POM contact AO
    float hemiOccL = hemiOcc;
    float dynHemiL = pc.dynHemi;
    if (L.pom_params3.y > 0.5) { occ = vec3(1.0); hemiOccL = 1.0; dynHemiL = 1.0; }   // r_ao_flat debug
    vec3 lighting = skyAmbient(geomN) * (hemiOccL * L.sky_params.y * dynHemiL) * occ
                  + L.sun_color.rgb  * sunMask
                  + L.ambient.rgb * occ
                  + dynLights(vWorldPos, Nw);

    // Rain wetness: darken + sky reflection where the rain map says open sky.
    vec3 wetRefl = applyWetness(albedo, vWorldPos, Nw);

    // Distance fog (R4): fade to the env haze colour with view distance —
    // the "wet air" that sinks the far street into a light haze.
    vec3 col = albedo * lighting + wetRefl;
    float fog = clamp(length(vWorldPos - L.eye_pos.xyz) * L.fog_params.w + L.fog_params.x, 0.0, 1.0);
    col = mix(col, L.fog_color.rgb, fog);

    outColor = vec4(col, base.a);
}
