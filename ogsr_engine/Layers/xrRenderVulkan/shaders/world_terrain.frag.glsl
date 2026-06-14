#version 450

// World pass - TERRAIN splatting fragment shader.
//
// Texture splatting (R4 deffer_terrain_high / CBlender_BmmD):
//   base  = s_base(uv)                       - terrain diffuse (e.g. terrain_escape)
//   mask  = s_mask(uv); mask /= dot(mask,1)  - RGBA splat weights, normalized
//   det   = dt_r*mask.r + dt_g*mask.g + dt_b*mask.b + dt_a*mask.a   - at detail UV
//   albedo = 2 * base * det                  - R4 "2*base*detail" convention
// Detail channels (CBlender_BmmD defaults): R grass, G asphalt, B earth, A gravel.
// Lightmap modulation identical to world_lmap. POM / puddles / detail-normals
// from the SSS shader are intentionally omitted (MVP - colour splatting only).

layout(set = 0, binding = 0) uniform sampler2D uBase;
layout(set = 0, binding = 1) uniform sampler2D uMask;
layout(set = 0, binding = 2) uniform sampler2D uDtR;   // grass
layout(set = 0, binding = 3) uniform sampler2D uDtG;   // asphalt
layout(set = 0, binding = 4) uniform sampler2D uDtB;   // earth
layout(set = 0, binding = 5) uniform sampler2D uDtA;   // gravel/yantar
layout(set = 0, binding = 6) uniform sampler2D uLmap;
// Per-channel detail-normal maps (<detail>_bump). R4 CBlender_BmmD: tangent
// normal packed as n = tex.wzy*2-1 (gloss in R). Blended by the splat mask,
// feeds SUN + DYN only — the sharp sky cube stays on the flat geometric normal.
layout(set = 0, binding = 7)  uniform sampler2D uDnR;   // grass   normal
layout(set = 0, binding = 8)  uniform sampler2D uDnG;   // asphalt normal
layout(set = 0, binding = 9)  uniform sampler2D uDnB;   // earth   normal
layout(set = 0, binding = 10) uniform sampler2D uDnA;   // gravel  normal

// Per-frame environment lighting (set 1) - see vk_env_light.{h,cpp}.
struct DynLight {
    vec4 pos;     // xyz = world position, w = range
    vec4 color;   // rgb = colour,         w = 1 spot / 0 point
    vec4 dir;     // xyz = spot direction, w = cos(cone/2)
};
layout(set = 1, binding = 0) uniform Lighting {
    vec4 sun_dir;
    vec4 sun_color;
    vec4 hemi_color;
    vec4 ambient;
    mat4 sun_vp;      // sun light view-proj (shadow lookup)
    vec4 counts;      // x = dynamic light count
    DynLight lights[16];
    mat4 spot_vp;        // spot (flashlight) shadow view-proj
    vec4 shadow_params;  // x = spot-shadowed light index (-1 none), y = point-shadowed index
    mat4 sun_near_vp;    // sun cascade 0 view-proj (25 m, per-frame, R4 scheme)
    mat4 sun_c1_vp;      // sun cascade 1 view-proj (60 m, per-frame)
    vec4 fog_color;      // rgb haze colour (env)
    vec4 fog_params;     // x=-near*r, y=near, z=far, w=r; fog = saturate(dist*w + x)
    vec4 eye_pos;        // xyz camera world pos
    vec4 sky_params;     // x=cube cross-fade weight, y=ambient scale, z=sample LOD
    vec4 ao_params;      // x=1/screenW, y=1/screenH, z=AO strength (0=off)
    mat4 rain_vp;        // straight-down ortho VP for the rain occlusion map
    vec4 rain_params;    // x=rain density, y=wetness, z=darken, w=reflection scale
    mat4 scene_vp;       // (SSR puddles — declared for layout match, unused here)
    vec4 cam_dir;
    vec4 cam_rightT;
    vec4 cam_topT;
    vec4 pom_params;     // x=POM amplitude (UV), y=max steps, z=fade dist (m), w=on
    vec4 pom_params2;    // x=blur, y=normal, z=self-shadow, w=contact AO
    vec4 pom_params3;    // x=debug, y=ao_flat, z=ceil strength, w=floor strength
    vec4 pom_params4;    // x=terrain POM enable, y=detail-normal strength, z=micro-AO strength, w=debug view
    vec4 pom_params5;    // x=terrain gloss, y=geo-puddle radius, z=geo-puddle depth, w=puddle debug
    vec4 pom_params6;    // x=water-sim enable (puddles + depth come from the flow sim), y=murk, z=refract
    vec4 pom_params7;    // SSS per-pixel puddles: x=enable, y=water level, z=micro-height contrast
} L;
layout(set = 1, binding = 1) uniform sampler2D uShadow;
layout(set = 1, binding = 2) uniform sampler2D uSpotShadow;
layout(set = 1, binding = 3) uniform samplerCube uPointShadow;
layout(set = 1, binding = 4) uniform sampler2D uShadowNear;    // sun cascade 0 (~0.61 cm texels)
layout(set = 1, binding = 5) uniform sampler2D uShadowC1;      // sun cascade 1 (~1.46 cm texels)
layout(set = 1, binding = 6) uniform samplerCube uSky0;        // sky ambient cube 0 (weather A)
layout(set = 1, binding = 7) uniform samplerCube uSky1;        // sky ambient cube 1 (weather B)
layout(set = 1, binding = 8) uniform sampler2D uAO;            // GTAO (half-res)
layout(set = 1, binding = 9) uniform sampler2D uRainMap;       // top-down rain occlusion (wetness mask)
layout(set = 1, binding = 10) uniform sampler2D uSpotCookie;   // flashlight beam texture (cookie)
layout(set = 1, binding = 11) uniform sampler2D uWater;        // water depth (flow sim, metres)
layout(set = 1, binding = 12) uniform sampler2D uFlow;         // water velocity (flow sim, uv/sec)

// GTAO visibility - see world_lmap.frag (occludes hemi+ambient only).
float gtaoVis()
{
    float ao = textureLod(uAO, gl_FragCoord.xy * L.ao_params.xy, 0.0).r;
    return pow(clamp(ao, 0.0, 1.0), L.ao_params.z);   // strength = exponent (0 = off)
}

// Colored AO - see world_lmap.frag (R4 compute_colored_ao port).
vec3 coloredAO(float ao, vec3 albedo)
{
    vec3 a =  2.0404 * albedo - 0.3324;
    vec3 b = -4.7951 * albedo + 0.6417;
    vec3 c =  2.7552 * albedo + 0.6903;
    return max(vec3(ao), ((ao * a + b) * ao + c) * ao);
}

// Hemisphere sky ambient (R4 hmodel.h) - see world_lmap.frag.
vec3 skyAmbient(vec3 N)
{
    float lod = L.sky_params.z;
    float xf = clamp(L.sky_params.x, 0.0, 1.0);   // weather cross-fade — usually 0/1
    vec3 a = textureLod(uSky0, N, lod).rgb;
    return (xf > 0.01) ? mix(a, textureLod(uSky1, N, lod).rgb, xf) : a;  // 2nd cube only in transition
}

// Spot/point shadow + dynamic lights - same model as world_lmap.frag.
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

float pointShadowF(vec3 wp, vec3 lp, float range)
{
    vec3 d = wp - lp;
    float z = max(max(abs(d.x), abs(d.y)), abs(d.z));
    const float n = 0.1;
    float refD = range * (z - n) / (max(z, n) * max(range - n, 1e-3));
    return (refD - 0.01 <= texture(uPointShadow, d).r) ? 1.0 : 0.0;
}

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
        if (L.lights[i].color.w > 0.5)
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

// Bilinear-weighted near-cascade PCF tap (textureGather) - see world_lmap.frag.
float cascTap(sampler2D smap, vec2 uv, float ref)
{
    vec2 sz = vec2(textureSize(smap, 0));
    vec2 t  = uv * sz - 0.5;
    vec2 f  = fract(t);
    vec4 d  = textureGather(smap, (floor(t) + 1.0) / sz, 0);
    vec4 c  = step(vec4(ref), d);
    return mix(mix(c.w, c.z, f.x), mix(c.x, c.y, f.x), f.y);
}

float cascSample(sampler2D smap, mat4 vp, vec3 wp, float bias_)
{
    vec3 n = (vp * vec4(wp, 1.0)).xyz;
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

// Rain visibility + wet shading - see world_lmap.frag. Terrain extra: the
// reflectivity scales with the ASPHALT splat weight (wet road mirrors the sky
// harder than wet dirt/grass - R4 gives terrain material-dependent gloss too).
float rainVis(vec3 wp)
{
    vec3 n = (L.rain_vp * vec4(wp, 1.0)).xyz;
    vec2 uv = n.xy * 0.5 + 0.5;
    uv.y = 1.0 - uv.y;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || n.z <= 0.0 || n.z >= 1.0)
        return 1.0;
    return cascTap(uRainMap, uv, n.z - 0.0015);
}

// Procedural puddle patches - see world_lmap.frag (uniform film reads waxy).
float puddleMask(vec2 p)
{
    float n = sin(p.x * 0.71 + sin(p.y * 0.53) * 1.7)
            * sin(p.y * 0.67 + sin(p.x * 0.49) * 1.7);
    return smoothstep(0.15, 0.65, n * 0.5 + 0.5);
}

// GEOMETRIC puddles - see world_lmap.frag. The rain occlusion map is a top-down
// ortho DEPTH of the scene (terrain is in it) = a height field (smaller depth =
// higher ground, eye is above). A pixel deeper than its neighbourhood average
// sits in a real dip -> holds water. Reuses uRainMap + rain_vp, no extra buffer.
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
    // rim. A pixel BELOW the level is underwater → that's how water fills valleys
    // and the low side of slopes while ridges/peaks (above the level) stay dry.
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

// SSS per-pixel puddles (SSFX deffer_terrain_high_flat.ps). Water is a rising
// LEVEL vs the per-pixel detail MICRO-HEIGHT: grooves of the ground TEXTURE (low
// microH) fill first, raised bumps stay dry — so puddles follow the texture relief
// (wheel ruts / nooks hold water), no coarse 1024² grid, no stripes, no compute
// sim (this is why SSFX road ruts pool while ours didn't). Slope-masked to near-
// horizontal ground (water can't cling to walls or steep dirt). microH is the
// splat-blended detail alpha, fortified with the detail-normal cavity so it still
// has relief on content whose alpha is flat (alpha=1). Returns 0..1 coverage.
float sssPuddle(float microH, vec3 N)
{
    float wet = clamp(L.rain_params.y, 0.0, 1.0);
    // Water plane rises with wetness; a pixel below it (microH < level) fills.
    float level = wet * L.pom_params7.y;                       // r_puddle_level
    float h   = clamp(level - microH * L.pom_params7.z, 0.0, 1.0);  // r_puddle_micro
    float pud = clamp(h * 4.0, 0.0, 1.0);                      // SSS hardness ×4
    // Slope mask (SSFX: 1 - max(|N.x|,|N.z|), then saturate(s-0.9)*13) — only the
    // nearly-flat ground holds standing water; gentle/steep slopes drain.
    float slope = clamp((1.0 - max(abs(N.x), abs(N.z)) - 0.9) * 13.0, 0.0, 1.0);
    return pud * slope;
}

// Water DEPTH (metres) from the flow sim, sampled at the pixel via rain_vp (the
// sim grid is aligned to the rain map). 0 where dry. See vk_water_sim.
float simWater(vec3 wp)
{
    vec4 c = L.rain_vp * vec4(wp, 1.0);
    if (c.w <= 0.0) return 0.0;
    vec2 uv = c.xy * 0.5 + 0.5; uv.y = 1.0 - uv.y;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) return 0.0;
    return textureLod(uWater, uv, 0.0).r;
}

// Blurred water depth for PUDDLE placement. The sim/ground-height map catches
// mesh folds/seams as 1-2 texel lines at 1024² → raw water reads as thin lines
// along the creases. A 5-tap blur spreads that into smooth AREA puddles.
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

// Ground height (metres, relative) from the rain map ortho depth — for the flow
// debug (water surface = groundHm + depth).
float groundHm(vec3 wp)
{
    vec4 c = L.rain_vp * vec4(wp, 1.0);
    if (c.w <= 0.0) return 0.0;
    vec2 uv = c.xy * 0.5 + 0.5; uv.y = 1.0 - uv.y;
    return -textureLod(uRainMap, uv, 0.0).r * 349.0;
}

// Water debug colour. mode 1 = DEPTH ramp (dry≈black, shallow blue, deep cyan→
// white). mode 2 = FLOW direction (downhill = −gradient of the water surface;
// direction → RG, speed → brightness) — shows where water runs and collects.
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

// Travelling surface waves ALONG the water flow (downhill). Where water moves
// (slopes / streams) the surface shows ripples scrolling downstream, scaled by
// flow speed; still pools (speed≈0) get ~none — the rain rings own those.
// Returns an xz normal perturbation. Needs water present (a thin film).
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

// Lagarde ring ripples - see world_lmap.frag.
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
    float ph = fract(t + h1);
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

vec3 applyWetness(inout vec3 albedo, vec3 wp, vec3 N, float asphalt, float pudIn)
{
    float wet = L.rain_params.y;
    if (wet < 0.005)
        return vec3(0.0);
    wet *= rainVis(wp);
    float upness = clamp(N.y, 0.0, 1.0);
    float wetK = wet * mix(0.35, 1.0, upness);
    albedo *= 1.0 - L.rain_params.z * wetK;

    // PERF: distance-fade + early-out the expensive reflection (see world_lmap).
    // Terrain is the biggest screen area, so this gate matters most.
    vec3  toEye    = L.eye_pos.xyz - wp;
    float dist     = length(toEye);
    float reflFade = smoothstep(70.0, 35.0, dist);
    if (wetK * reflFade < 0.004) return vec3(0.0);

    float t = L.sky_params.w;
    float pud = clamp(pudIn, 0.0, 1.0) * upness;
    float ripFade = smoothstep(18.0, 8.0, dist) * clamp(L.rain_params.x * 1.5 + 0.1, 0.0, 1.0);
    // FLOW: water velocity (uv/sec → m/s over the ±75 m box) advects the ripple
    // field DOWNSTREAM so the surface visibly moves (no stripes), + extra chop on
    // fast water. velMS also drives foam below. (SSS source = no flow → vel 0.)
    vec2  vel   = (L.pom_params6.x > 0.5) ? simFlow(wp) : vec2(0.0);
    float velMS = length(vel) * 150.0;
    vec2  scrl  = (velMS > 0.01) ? normalize(vel) * (t * velMS * 0.25) : vec2(0.0);
    // PUDDLE = FLAT WATER MIRROR (SSFX: Ne = lerp(Ne, up, pud²)). Inside a puddle
    // the reflecting surface is the water plane, not the bumpy ground — flatten the
    // normal toward world-up so it mirrors what's ABOVE (sky/scene), the RDR2 look,
    // instead of grazing-reflecting the dirt. Ripples then perturb this flat plane.
    vec3  Nbase = mix(N, vec3(0.0, 1.0, 0.0), clamp(pud * pud, 0.0, 1.0));
    vec3  Nr = Nbase;
    if (ripFade > 0.01) {
        vec2 rip = rainRipples(wp.xz - scrl, t * 0.6) * (0.4 * ripFade);
        if (velMS > 0.1)
            rip += rainRipples(wp.xz * 1.6 - scrl * 1.6, t * 0.9) * (clamp(velMS * 0.12, 0.0, 0.5) * ripFade);
        Nr = normalize(vec3(Nbase.x + rip.x, Nbase.y, Nbase.z + rip.y));
    }
    vec3 V = normalize(toEye);
    vec3 R = reflect(-V, Nr);
    float fres = pow(1.0 - clamp(dot(V, Nr), 0.0, 1.0), 3.0);
    float xf = clamp(L.sky_params.x, 0.0, 1.0);
    vec3 sky = textureLod(uSky0, R, 0.0).rgb;
    if (xf > 0.01) sky = mix(sky, textureLod(uSky1, R, 0.0).rgb, xf);
    // Asphalt: a wet road films over more evenly; dirt/grass: reflection lives
    // almost only in the puddle patches. Overall gloss stays SMALL (R4: 0.2).
    float matRefl = mix(0.6, 1.4, clamp(asphalt, 0.0, 1.0));
    float sheenFloor = mix(0.05, 0.2, clamp(asphalt, 0.0, 1.0));
    // Deep water = a clear water BODY: cool dark tint on the bottom + a stronger,
    // sharper reflection so it reads as WATER, not just wet ground (RDR2-ish).
    albedo = mix(albedo, albedo * vec3(0.45, 0.55, 0.62), clamp(pud, 0.0, 1.0));
    // RDR2/SSFX: standing water has a real BASE reflectivity (~0.2 + fresnel), not
    // just a grazing glint — so even looking straight down the puddle mirrors the
    // sky/scene. Wet-but-not-puddled ground keeps the thin fresnel-only sheen.
    float baseRefl = mix(0.04, 0.25, pud);
    float reflK = wetK * matRefl * (sheenFloor + (1.1 - sheenFloor) * pud) * (baseRefl + (1.0 - baseRefl) * fres) * reflFade;
    // Foam: whiten fast-moving water (rapids / where streams run quick).
    float foam = smoothstep(3.0, 6.0, velMS) * clamp(pud, 0.0, 1.0) * ripFade * 0.25;
    return sky * (reflK * L.rain_params.w) + vec3(foam);
}

// NEAR cascade first (leaf-shaped dapples, smooth motion) - see world_lmap.frag.
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
    // 4 spread taps (was 3-3) - see world_lmap.frag.
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
} pc;

layout(location = 0) in  vec2 vUV;
layout(location = 1) in  vec2 vDetailUV;
layout(location = 2) in  vec2 vLmapUV;
layout(location = 3) in  vec3 vWorldPos;
layout(location = 4) in  vec3 vNormal;
layout(location = 0) out vec4 outColor;

// ---- POM for terrain. The terrain set has NO `#` height map, so the relief
// comes from the HIGH-PASSED luminance of the BASE albedo (where the macro
// cracks / pebbles / twigs live). High-pass removes the large tone patches so
// they don't become false cliffs; same relief march + normal + self-shadow +
// contact AO + orientation weight as the wall POM (terrain = floor → r_pom_floor).
float baseLuma(vec2 uv, float lod) { return dot(textureLod(uBase, uv, lod).rgb, vec3(0.299, 0.587, 0.114)); }

float pomDepth(vec2 uv, float lod, float baseline)
{
    return clamp(0.5 - (baseLuma(uv, lod) - baseline) * 4.0, 0.0, 1.0);
}

vec2 parallaxUV(vec2 uv, vec3 N, vec3 wp, out vec3 outN, out float outShadow, out float outAO)
{
    outN = N; outShadow = 1.0; outAO = 1.0;
    if (L.pom_params4.x < 0.5) return uv;   // terrain POM disabled (r_pom_terrain 0) → flat ground
    float amp = L.pom_params.x;
    if (amp <= 0.0) return uv;
    float dist = length(L.eye_pos.xyz - wp);
    float fade = 1.0 - smoothstep(L.pom_params.z * 0.5, L.pom_params.z, dist);
    amp *= fade;
    float orient = (N.y >= 0.0)
        ? mix(1.0, L.pom_params3.w, clamp( N.y, 0.0, 1.0))
        : mix(1.0, L.pom_params3.z, clamp(-N.y, 0.0, 1.0));
    amp *= orient;
    if (amp <= 1e-5) return uv;

    vec2  tsz = vec2(textureSize(uBase, 0));
    vec2  ddx = dFdx(uv) * tsz, ddy = dFdy(uv) * tsz;
    float lod = max(0.5 * log2(max(dot(ddx, ddx), dot(ddy, ddy))), 0.0) + L.pom_params2.x;
    float baseline = baseLuma(uv, lod + 3.0);

    vec3 dp1 = dFdx(wp), dp2 = dFdy(wp);
    vec2 du1 = dFdx(uv), du2 = dFdy(uv);
    vec3 dp2p = cross(dp2, N), dp1p = cross(N, dp1);
    vec3 T = dp2p * du1.x + dp1p * du2.x;
    vec3 B = dp2p * du1.y + dp1p * du2.y;
    float inv = inversesqrt(max(dot(T, T), dot(B, B)));
    T *= inv; B *= inv;

    vec3 V   = normalize(L.eye_pos.xyz - wp);
    vec3 Vts = vec3(dot(V, T), dot(V, B), dot(V, N));
    // Terrain is viewed at GRAZING angles almost always (it's the floor), where
    // the parallax offset blows up (Vts.z → 0) and the texture "swims"/mirrors
    // as the camera moves. Cap the offset (clamp floor 0.55) and FADE it out at
    // grazing (smoothstep on |Vts.z|) — kills the liquid look, keeps relief at
    // more head-on angles. Normal/self-shadow/AO are unaffected (relief stays).
    vec2 Pmax = (Vts.xy / max(abs(Vts.z), 0.55)) * amp * smoothstep(0.12, 0.45, abs(Vts.z));

    int steps = int(clamp(mix(L.pom_params.y, 12.0, abs(Vts.z)), 12.0, 64.0));
    float layerH = 1.0 / float(steps);
    vec2 dUV = Pmax * layerH;

    float curD = 0.0;
    vec2  curUV = uv;
    float curH = pomDepth(curUV, lod, baseline);
    for (int i = 0; i < 64; ++i) {
        if (i >= steps || curD >= curH) break;
        curUV -= dUV; curD += layerH;
        curH = pomDepth(curUV, lod, baseline);
    }
    vec2 sUV = dUV; float sD = layerH;
    for (int j = 0; j < 6; ++j) {
        sUV *= 0.5; sD *= 0.5;
        if (curD < pomDepth(curUV, lod, baseline)) { curUV -= sUV; curD += sD; }
        else                                       { curUV += sUV; curD -= sD; }
    }

    // Normal from base-luma gradient.
    float tU = exp2(lod) / tsz.x, tV = exp2(lod) / tsz.y;
    float hu = baseLuma(curUV + vec2(tU, 0.0), lod) - baseLuma(curUV - vec2(tU, 0.0), lod);
    float hv = baseLuma(curUV + vec2(0.0, tV), lod) - baseLuma(curUV - vec2(0.0, tV), lod);
    // Base-luma gradient is MUCH higher-contrast than the wall `#` map, so a
    // gentler gain + a tilt clamp — otherwise the ground normal swings wildly
    // and the sharp sky-cube sampling turns the terrain into a chrome mirror
    // ("liquid Terminator"). ×3 (vs ×12 on walls) + ±0.6 tilt cap.
    float ns = L.pom_params2.y * 3.0 * fade * orient;
    vec3 nTS = normalize(vec3(clamp(-hu * ns, -0.6, 0.6), clamp(-hv * ns, -0.6, 0.6), 1.0));
    outN = normalize(T * nTS.x + B * nTS.y + N * nTS.z);

    // Self-shadow toward the sun (horizon scan).
    if (L.pom_params2.z > 0.0) {
        vec3 Ld  = normalize(-L.sun_dir.xyz);
        vec3 Lts = vec3(dot(Ld, T), dot(Ld, B), dot(Ld, N));
        vec2 lxy = Lts.xy;
        if (Lts.z > 0.02 && dot(lxy, lxy) > 1e-6) {
            vec2  sdir  = normalize(lxy);
            float reach = (exp2(lod) / min(tsz.x, tsz.y)) * 6.0;
            float h0    = baseLuma(curUV, lod);
            float occ   = 0.0;
            for (int s = 1; s <= 8; ++s)
                occ = max(occ, baseLuma(curUV + sdir * reach * (float(s) * 0.125), lod) - h0);
            outShadow = clamp(1.0 - occ * L.pom_params2.z * 20.0 * (1.0 - Lts.z) * orient, 0.0, 1.0);
        }
    }
    // Contact AO (view-independent).
    if (L.pom_params2.w > 0.0) {
        float aoReach = (exp2(lod) / min(tsz.x, tsz.y)) * 4.0;
        float h0 = baseLuma(curUV, lod);
        float aoSum =
              max(0.0, baseLuma(curUV + vec2( aoReach, 0.0), lod) - h0)
            + max(0.0, baseLuma(curUV + vec2(-aoReach, 0.0), lod) - h0)
            + max(0.0, baseLuma(curUV + vec2(0.0,  aoReach), lod) - h0)
            + max(0.0, baseLuma(curUV + vec2(0.0, -aoReach), lod) - h0);
        outAO = clamp(1.0 - (aoSum * 0.25) * L.pom_params2.w * 6.0, 0.35, 1.0);
        outAO = mix(1.0, outAO, orient);
    }
    return curUV;
}

// ---- Detail NORMAL MAPPING (R4 CBlender_BmmD s_dn_*). The real ground-relief
// mechanism on terrain (POM swims at grazing angles, so it stays off). Blends
// the 4 per-channel tangent normals by the splat mask, then rotates into world
// space via a screen-space cotangent frame (terrain has no per-vertex tangents).
// `strength` scales the tangent slope (xy). Returns geomN unchanged when off.
vec3 detailNormal(vec3 geomN, vec2 duv, vec4 mask, float strength, out float outCav, out float outGloss)
{
    outCav = 0.0; outGloss = 0.0;
    if (strength <= 0.0) return geomN;
    // One sample per channel; decode normal (R4: n = tex.wzy*2-1) AND gloss
    // (R4: gloss = tex.r), both blended by the splat mask. Grass low gloss,
    // asphalt high → material-aware specular without extra texture reads.
    vec4 sR = texture(uDnR, duv), sG = texture(uDnG, duv);
    vec4 sB = texture(uDnB, duv), sA = texture(uDnA, duv);
    vec3 n = (sR.wzy * 2.0 - 1.0) * mask.r
           + (sG.wzy * 2.0 - 1.0) * mask.g
           + (sB.wzy * 2.0 - 1.0) * mask.b
           + (sA.wzy * 2.0 - 1.0) * mask.a;
    outGloss = sR.x * mask.r + sG.x * mask.g + sB.x * mask.b + sA.x * mask.a;
    n.xy *= strength;
    n = normalize(n);

    // Cavity term: how far the micro-normal tilts off the surface (tangent
    // z < 1 in grooves). Drives the micro contact AO below — always meaningful
    // because the normals loaded (independent of whether detail alpha = height).
    outCav = clamp(1.0 - n.z, 0.0, 1.0);

    // Cotangent frame (Schüler) from world-pos + detail-UV screen derivatives.
    vec3 dp1 = dFdx(vWorldPos), dp2 = dFdy(vWorldPos);
    vec2 du1 = dFdx(duv),       du2 = dFdy(duv);
    vec3 dp2p = cross(dp2, geomN), dp1p = cross(geomN, dp1);
    vec3 T = dp2p * du1.x + dp1p * du2.x;
    vec3 B = dp2p * du1.y + dp1p * du2.y;
    float inv = inversesqrt(max(dot(T, T), dot(B, B)));
    T *= inv; B *= inv;
    return normalize(T * n.x + B * n.y + geomN * n.z);
}

void main()
{
    vec3 pomN; float pomShadow, pomAO;
    vec2 pUV = parallaxUV(vUV, normalize(vNormal), vWorldPos, pomN, pomShadow, pomAO);
    vec2 pDelta = pUV - vUV;
    vec2 pDetailUV = vDetailUV + pDelta * pc.detailScale;

    vec4 base = texture(uBase, pUV);
    if (pc.alphaRef >= 0.0 && base.a < pc.alphaRef) discard;

    // r_ssao_debug 1: show the raw AO map - see world_lmap.frag.
    if (L.ao_params.w > 0.5) {
        outColor = vec4(vec3(textureLod(uAO, gl_FragCoord.xy * L.ao_params.xy, 0.0).r), base.a);
        return;
    }

    // r_wet_debug 1: rain-map visibility - see world_lmap.frag.
    if (L.rain_params.z < 0.0) {
        outColor = vec4(vec3(rainVis(vWorldPos)), base.a);
        return;
    }

    // r_puddle_debug moved below (the SSS source needs the detail micro-height,
    // computed after the splat) — see the pud block.

    // r_pom_debug 1: POM occlusion mask (contact AO × self-shadow).
    if (L.pom_params3.x > 0.5) {
        outColor = vec4(vec3(pomAO * pomShadow), base.a);
        return;
    }

    // Splat mask (at the parallax-offset UV) - normalize so the 4 weights sum
    // to 1. Empty/missing mask (1-1 white fallback - sum 4) degrades to an even
    // blend; an all-zero mask falls back to pure grass so terrain never goes black.
    vec4 mask = texture(uMask, pUV);
    float wsum = dot(mask, vec4(1.0));
    mask = (wsum > 1e-4) ? (mask / wsum) : vec4(1.0, 0.0, 0.0, 0.0);

    vec4 dR = texture(uDtR, pDetailUV);
    vec4 dG = texture(uDtG, pDetailUV);
    vec4 dB = texture(uDtB, pDetailUV);
    vec4 dA = texture(uDtA, pDetailUV);
    vec3 detail = dR.rgb * mask.r + dG.rgb * mask.g + dB.rgb * mask.b + dA.rgb * mask.a;
    // Detail height (R4 terrain AO source = detail diffuse alpha, splat-blended).
    // On content that lacks a real height (alpha = 1) this collapses to 1 and the
    // height term of the micro-AO does nothing — the normal cavity still works.
    float detH = dR.a * mask.r + dG.a * mask.g + dB.a * mask.b + dA.a * mask.a;

    vec3 albedo = 2.0 * base.rgb * detail;

    // Lightmap (hemi/AO) + DYNAMIC R4-style sun (per-pixel N-L - shadow map) -
    // same model as world_lmap.frag.
    vec4  lm      = texture(uLmap, vLmapUV);
    float hemiOcc = dot(lm.rgb, vec3(1.0 / 3.0));
    vec3  geomN   = normalize(vNormal);   // flat — for the sky fill (cube is sharp → perturbed = mirror)
    vec3  Nw      = pomN;                  // POM-perturbed — sun + dyn lights catch the relief
    // Detail normal mapping: the primary ground-relief source. Overrides the
    // POM normal (which is off by default) when r_terrain_normal > 0; the sky
    // fill below keeps using geomN so up-facing ground never turns to mirror.
    // LOD: the 4 detail-normal taps + AO + gloss only matter up close — fade the
    // strength to 0 by distance so far terrain skips them entirely (detailNormal
    // early-outs at strength 0). Big World-pass saving over the visible ground.
    float dnStr  = L.pom_params4.y * smoothstep(45.0, 25.0, distance(L.eye_pos.xyz, vWorldPos));
    float cav    = 0.0;
    float glossT = 0.0;
    if (dnStr > 0.0)
        Nw = detailNormal(geomN, pDetailUV, mask, dnStr, cav, glossT);

    // Micro contact AO: darken grooves. Occlusion = max of the normal cavity
    // (1-n.z) and the detail height pit (1-h²) — whichever the data provides.
    // Cheap, no march, no swim. Applied to albedo (R4 base*=1-AO) so it shades
    // both sun and ambient like real self-occlusion.
    float aoStr   = L.pom_params4.z;
    float occlT   = max(cav, 1.0 - detH * detH);
    float microAO = 1.0 - aoStr * clamp(occlT, 0.0, 1.0);
    albedo *= microAO;

    // PUDDLE COVERAGE (0..1) — choose the source. SSS per-pixel (default) rises a
    // water plane vs the detail micro-height so the ground TEXTURE relief fills;
    // micro-height = detail alpha fortified by the detail-normal cavity (so it
    // still has relief on flat-alpha content). Fallbacks: flow sim → geo dip →
    // procedural sine. Slope/upness masking happens inside sssPuddle / applyWetness.
    float microH = min(detH, 1.0 - cav);
    float pud;
    if (L.pom_params7.x > 0.5)       pud = sssPuddle(microH, geomN);
    else if (L.pom_params6.x > 0.5)  pud = smoothstep(0.04, 0.12, simWaterSoft(vWorldPos));
    else if (L.pom_params5.y > 0.0)  pud = geoPuddle(vWorldPos);
    else                             pud = puddleMask(vWorldPos.xz * 0.8);

    // r_puddle_debug: grayscale puddle coverage (SSS / geo) or sim depth/flow.
    int pdbg = int(L.pom_params5.w + 0.5);
    if (pdbg > 0) {
        outColor = (L.pom_params7.x > 0.5)      ? vec4(vec3(pud), base.a)
                 : (L.pom_params6.x > 0.5)      ? vec4(waterDebugColor(vWorldPos, pdbg), base.a)
                 :                                 vec4(vec3(geoPuddle(vWorldPos)), base.a);
        return;
    }

    // r_terrain_debug: 1 world normal, 2 micro-AO, 3 detail height.
    int tdbg = int(L.pom_params4.w + 0.5);
    if (tdbg == 1) { outColor = vec4(Nw * 0.5 + 0.5, base.a); return; }
    if (tdbg == 2) { outColor = vec4(vec3(microAO),  base.a); return; }
    if (tdbg == 3) { outColor = vec4(vec3(detH),     base.a); return; }

    float sunMask = max(dot(Nw, normalize(-L.sun_dir.xyz)), 0.0);
    if (sunMask > 0.005)
        sunMask *= sunShadow(vWorldPos) * pomShadow;
    // Hemisphere sky fill (R4 hmodel) - see world_lmap.frag. sun_color/ambient
    // arrive final from vk_env_light (r_sun_boost / r_ambient_floor).
    vec3  occ      = coloredAO(gtaoVis(), albedo) * pomAO;
    float hemiOccL = hemiOcc;
    if (L.pom_params3.y > 0.5) { occ = vec3(1.0); hemiOccL = 1.0; }   // r_ao_flat debug
    vec3 lighting = skyAmbient(geomN) * (hemiOccL * L.sky_params.y) * occ
                  + L.sun_color.rgb  * sunMask
                  + L.ambient.rgb * occ
                  + dynLights(vWorldPos, Nw);

    // Dry sun gloss (R4 gloss in bump .r): a material-aware Blinn specular —
    // asphalt/gravel glint, grass stays matte. Additive (specular is ~albedo-
    // independent), respects the sun shadow (sunMask already folds it in), and
    // FADES OUT as the ground wets so it doesn't double up with the wet
    // reflection. Held subtle to avoid grazing-angle shimmer.
    vec3  drySpec  = vec3(0.0);
    float glossStr = L.pom_params5.x;
    if (glossStr > 0.0 && glossT > 0.0 && sunMask > 0.005) {
        vec3  Ld   = normalize(-L.sun_dir.xyz);
        vec3  V    = normalize(L.eye_pos.xyz - vWorldPos);
        vec3  H    = normalize(Ld + V);
        float g    = clamp(glossT, 0.0, 1.0);
        float shin = mix(16.0, 160.0, g);             // glossier → tighter highlight
        float s    = pow(max(dot(Nw, H), 0.0), shin) * g;
        float dry  = 1.0 - clamp(L.rain_params.y, 0.0, 1.0);
        drySpec    = L.sun_color.rgb * (s * glossStr * sunMask * dry);
    }

    // Rain wetness (asphalt reflects harder) - see applyWetness above.
    vec3 wetRefl = applyWetness(albedo, vWorldPos, Nw, mask.g, pud);

    // Distance fog (R4) - see world_lmap.frag.
    vec3 col = albedo * lighting + drySpec + wetRefl;
    float fog = clamp(length(vWorldPos - L.eye_pos.xyz) * L.fog_params.w + L.fog_params.x, 0.0, 1.0);
    col = mix(col, L.fog_color.rgb, fog);

    outColor = vec4(col, base.a);
}
