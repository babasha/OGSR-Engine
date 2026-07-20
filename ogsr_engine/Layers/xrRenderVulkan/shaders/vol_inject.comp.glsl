#version 450
#extension GL_GOOGLE_include_directive : require
#include "vsm_common.glsl"   // VSM clipmap math (vsmSelect/vsmPageIndex) for the smooth atlas occlusion
#include "froxel.glsl"        // exp-Z slice <-> view-Z mapping (shared with tonemap + particle probe)
// xrRenderVulkan — froxel volumetric INJECTION (vk_volumetrics, P1).
//
// One thread per froxel. Reconstruct the froxel's world position from the camera
// basis (DeriveProjTerms scheme — camDir + right*tanX*ndc.x + top*tanY*ndc.y,
// scaled by the exponential view-Z), evaluate height/base fog density, then the
// sun in-scatter = Henyey-Greenstein phase * cascade sun-shadow * sun colour
// (+ a flat sky-ambient fill), all * density. Output: rgb = in-scatter,
// a = extinction (= density). vol_integrate then marches Z over this.
//
// Occlusion = the sun CASCADE maps (cascSample/cascTap ported from
// world_lmap.frag) — the r_vsm-OFF path + the permanent fallback. VSM-atlas
// sampling is the P4 upgrade.

layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;

struct VolLight { vec4 pos; vec4 color; vec4 dir; };  // pos.w=range, color.w=1 spot/0 point, dir.w=cos(cone/2)

layout(set = 0, binding = 0) uniform Vol {
    vec4 camPos;       // xyz world camera pos
    vec4 camDir;       // xyz camera forward (unit)
    vec4 camRightT;    // xyz right * tan(fovX/2)
    vec4 camTopT;      // xyz top   * tan(fovY/2)
    vec4 sun_dir;      // xyz sun TRAVEL dir (down); to-sun = -sun_dir
    vec4 sun_color;    // rgb
    vec4 sky_ambient;  // rgb flat fill
    mat4 sun_vp;       // far cascade VP
    mat4 sun_near_vp;  // cascade 0 VP
    mat4 sun_c1_vp;    // cascade 1 VP
    vec4 gridParams;   // x=dimX y=dimY z=dimZ
    vec4 zParams;      // x=near y=far z=log2(far/near)
    vec4 fog;          // x=baseDensity y=heightBase z=heightFalloff w=HG_g
    vec4 fog2;         // x=intensity y=ambient floor z=indoor density boost w=sun-beam boost
    mat4 rain_vp;      // top-down ortho VP for the sky-visibility (ambient) occlusion
    mat4 prevViewProj; // temporal reprojection (prev frame world→clip)
    vec4 prevCamPos;   // xyz prev cam pos, w = prev near
    vec4 prevCamDir;   // xyz prev cam forward, w = prev log2(far/near)
    vec4 temporal;     // xyz = froxel jitter (−0.5..0.5), w = history blend (0 = off)
    mat4 fog_shadow_vp; // r_vol_shadow: dedicated per-frame fog sun-shadow VP
    vec4 lightParams;  // P2: x = count, y = boost, z = unused (-1), w = point-shadowed idx (-1)
    VolLight lights[8];
    vec4 noiseParams;  // P3: x = amount, y = scale, z = speed, w = time
    // Spot shadow POOL: per-tile VP. A light's tile rides in color.w bits 2+
    // (tile+1, <<2) — every pooled spot's fog cone is cut by its own tile.
    mat4 spot_pool_vp[8];
    vec4 smokeParams;  // Stage-1 VMS: x = smoke-inject strength (0 = no injected smoke)
    vec4 light_occ;    // r_light_occ: x = enable, y = bury bias, z = frag-below band, w = strength
    // Atmospheric scattering (r_atmo): physical Rayleigh (blue) + Mie (forward halo)
    // in-scatter of the sun → aerial perspective. Appended last (integrate prefix-safe).
    vec4 atmo;         // x = enable, y = Rayleigh strength, z = Mie strength, w = Mie g
    vec4 atmoR;        // rgb = Rayleigh scattering tint (blue-heavy), w unused
} V;

layout(set = 0, binding = 1) uniform sampler2D uShadowNear; // cascade 0
layout(set = 0, binding = 2) uniform sampler2D uShadowC1;   // cascade 1
layout(set = 0, binding = 3) uniform sampler2D uShadow;     // far cached map
layout(set = 0, binding = 4, rgba16f) uniform writeonly image3D uScatter;
layout(set = 0, binding = 5) uniform sampler2D uRainMap;    // top-down statics depth (sky visibility)
layout(set = 0, binding = 6) uniform sampler3D uHistory;    // prev frame's blended scatter (temporal)
// VSM static atlas occlusion (smooth, no cache tick) — gated by gridParams.w.
layout(set = 0, binding = 7) uniform sampler2D uVsmAtlas;   // static VSM atlas (opaque + trees)
layout(set = 0, binding = 8) readonly buffer VsmPageTable { uint vsmPageTable[]; }; // virtual page → atlas slot
layout(set = 0, binding = 9) uniform VsmClipmap {
    mat4 view;                 // world → sun light space
    vec4 level[VSM_LEVELS];    // xy = level origin (light XY of texel 0,0), z = extent (m)
    vec4 zparams;              // x = zNear, y = 1/(zFar-zNear), z = depth bias
} vsmC;
layout(set = 0, binding = 10) uniform sampler2D uFogShadow;  // dedicated per-frame fog sun-shadow
layout(set = 0, binding = 11) uniform sampler2D   uSpotShadow;  // spot (flashlight) shadow — occlude the fog cone
layout(set = 0, binding = 12) uniform samplerCubeArray uPointShadow; // point shadow cube POOL
layout(set = 0, binding = 13) uniform sampler3D   uSmokeMedia;  // Stage-1 VMS: splatted+resolved smoke (rgb=albedo, a=density)

const float PI = 3.14159265;

// --- P3: cheap animated 3D value noise (drifting dust / "living air") -----------
float hash13(vec3 p)
{
    p = fract(p * 0.1031);
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}
float vnoise3(vec3 x)
{
    vec3 i = floor(x), f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash13(i + vec3(0,0,0)), hash13(i + vec3(1,0,0)), f.x),
                   mix(hash13(i + vec3(0,1,0)), hash13(i + vec3(1,1,0)), f.x), f.y),
               mix(mix(hash13(i + vec3(0,0,1)), hash13(i + vec3(1,0,1)), f.x),
                   mix(hash13(i + vec3(0,1,1)), hash13(i + vec3(1,1,1)), f.x), f.y), f.z);
}
float fbm3(vec3 p) { return 0.62 * vnoise3(p) + 0.38 * vnoise3(p * 2.3 + 11.7); }

// --- Sun shadow, R4 cascade scheme (ported from world_lmap.frag) -------------
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
    vec3 n = (vp * vec4(wp, 1.0)).xyz;          // ortho → already NDC
    vec2 uv = n.xy * 0.5 + 0.5;
    uv.y = 1.0 - uv.y;
    if (uv.x < 0.01 || uv.x > 0.99 || uv.y < 0.01 || uv.y > 0.99 || n.z <= 0.0 || n.z >= 1.0)
        return -1.0;
    float ref = n.z - bias_;
    // WIDE 3x3 PCF (spacing = r_vol_soft texels): a SOFT penumbra. The fog doesn't
    // need crisp shadows — a soft shadow makes the foliage dapple a gentle gradient
    // instead of sharp speckles, so the cascade cache TICK (sun creep) barely shows.
    vec2  tx  = (1.0 / vec2(textureSize(smap, 0))) * max(V.zParams.w, 0.5);
    float s = 0.0;
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx)
        s += cascTap(smap, uv + vec2(float(dx), float(dy)) * tx, ref);
    return s * (1.0 / 9.0);
}
// VSM static-atlas occlusion (port of vsm_resolve sampleVSM, static atlas only):
// world → light → clipmap level → page → atlas slot → 3x3 PCF depth compare. The
// VSM updates SMOOTHLY per frame (no cascade cache TICK) → stable shafts. Returns
// lit 1..0, or -1.0 when this point has NO resident VSM page → caller falls back to
// the cascade (covers off-screen gaps in the view-dependent VSM).
float sampleVSMStatic(vec3 wp)
{
    vec3 lp = (vsmC.view * vec4(wp, 1.0)).xyz;
    vec2 luv; ivec2 page;
    int  L = vsmSelect(lp.xy, vsmC.level, luv, page);
    if (L < 0) return -1.0;                       // outside the clipmap → cascade fallback
    // Coarser-level fallback (mirrors vsm_resolve): the throttle LOD bias marks pages
    // coarser and the dirty budget can unmap one for a frame — walk up before giving
    // the froxel to the cascade path. Steady state exits on the first iteration.
    uint slot = VSM_UNMAPPED;
    for (; L < VSM_LEVELS; ++L) {
        vec2 t = (lp.xy - vsmC.level[L].xy) / vsmC.level[L].z;
        page   = clamp(ivec2(floor(t * float(VSM_PAGES_AXIS))), ivec2(0), ivec2(VSM_PAGES_AXIS - 1));
        slot   = vsmPageTable[vsmPageIndex(L, page)];
        if (slot < uint(VSM_MAX_PHYS_S)) { luv = t; break; }
    }
    if (L >= VSM_LEVELS) return -1.0;              // page not resident → cascade fallback

    vec2  pageLocal = luv * float(VSM_PAGES_AXIS) - vec2(page);
    vec2  base  = vec2(float(slot % uint(VSM_ATLAS_W_S)), float(slot / uint(VSM_ATLAS_W_S)));
    float zHere = (lp.z - vsmC.zparams.x) * vsmC.zparams.y;
    float bias  = vsmC.zparams.z;
    const vec2  dim   = vec2(float(VSM_ATLAS_W_S), float(VSM_ATLAS_H_S));
    const float tp    = 1.0 / float(VSM_PAGE_SIZE);
    const float inset = 0.5 * tp;
    float lit = 0.0;
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        vec2 pl = clamp(pageLocal + vec2(float(dx), float(dy)) * tp, vec2(inset), vec2(1.0 - inset));
        float occ = texture(uVsmAtlas, (base + pl) / dim).r;
        lit += (zHere - bias > occ) ? 0.0 : 1.0;
    }
    return lit * (1.0 / 9.0);
}

// Dedicated per-frame fog sun-shadow (r_vol_shadow): re-rendered EVERY frame with a
// fresh anchor-snapped VP → CONTINUOUS occlusion (no cache tick). Soft 3x3 PCF (the
// r_vol_soft radius). Ortho. Returns lit 1..0, or -1.0 outside the box → cascade.
float sampleFogShadow(vec3 wp)
{
    vec3 n = (V.fog_shadow_vp * vec4(wp, 1.0)).xyz;   // ortho → already NDC
    vec2 uv = n.xy * 0.5 + 0.5;
    uv.y = 1.0 - uv.y;
    if (uv.x < 0.01 || uv.x > 0.99 || uv.y < 0.01 || uv.y > 0.99 || n.z <= 0.0 || n.z >= 1.0)
        return -1.0;
    float ref = n.z - 0.0006;
    vec2  tx  = (1.0 / vec2(textureSize(uFogShadow, 0))) * max(V.zParams.w, 0.5);
    float s = 0.0;
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx)
        s += cascTap(uFogShadow, uv + vec2(float(dx), float(dy)) * tx, ref);
    return s * (1.0 / 9.0);
}

float sunShadow(vec3 worldPos)
{
    // Occlusion source (gridParams.w): 2 = dedicated per-frame fog shadow (continuous,
    // no tick), 1 = VSM atlas, 0 = cascade. Each falls through to the cascade where it
    // has no coverage (outside the box / no VSM page / r_vsm off).
    float mode = V.gridParams.w;
    if (mode > 1.5) {
        float f = sampleFogShadow(worldPos);
        if (f >= 0.0) return f;
    } else if (mode > 0.5) {
        // VSM STATIC ATLAS is the occluder. Under VSM the near cascades are NOT
        // rendered for the fog (vk_pass_shadow cascForVol → cascRaster false), so their
        // combined maps hold stale/frozen depth — DON'T fall through to them. A froxel
        // with no resident VSM page is treated as LIT (the atlas covers the visible
        // frustum; misses are rare air points over non-visible ground). This makes the
        // fog occlusion independent of cascade state and reclaims the ~2.2ms double-pay.
        float v = sampleVSMStatic(worldPos);
        return (v >= 0.0) ? v : 1.0;
    }
    // Smaller bias than the surface receivers: a froxel is in AIR (no self-shadow
    // acne to hide), so a tight bias just reduces light leaking THROUGH walls.
    float s = cascSample(uShadowNear, V.sun_near_vp, worldPos, 0.00015);
    if (s >= 0.0) return s;
    s = cascSample(uShadowC1, V.sun_c1_vp, worldPos, 0.00025);
    if (s >= 0.0) return s;

    vec4 c = V.sun_vp * vec4(worldPos, 1.0);
    if (c.w <= 0.0) return 1.0;
    vec3 ndc = c.xyz / c.w;
    vec2 uv = ndc.xy * 0.5 + 0.5;
    uv.y = 1.0 - uv.y;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z > 1.0) return 1.0;
    float ref = ndc.z - 0.0008;
    return cascTap(uShadow, uv, ref);
}

// Sky visibility (top-down statics depth): 1 = open to the sky, 0 = under a roof.
// Occludes the unconditional ambient term so the air inside a building doesn't glow
// with outdoor haze (the sun beam through the window still lights via the cascade).
float skyVis(vec3 wp)
{
    vec3 n = (V.rain_vp * vec4(wp, 1.0)).xyz;   // ortho → already NDC
    vec2 uv = n.xy * 0.5 + 0.5;
    uv.y = 1.0 - uv.y;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || n.z <= 0.0 || n.z >= 1.0)
        return 1.0;                              // outside the map = open sky
    float ref = n.z - 0.002;
    vec2  px  = 1.0 / vec2(textureSize(uRainMap, 0));
    return 0.25 * (cascTap(uRainMap, uv + vec2(-0.5, -0.5) * px, ref)
                 + cascTap(uRainMap, uv + vec2( 0.5, -0.5) * px, ref)
                 + cascTap(uRainMap, uv + vec2(-0.5,  0.5) * px, ref)
                 + cascTap(uRainMap, uv + vec2( 0.5,  0.5) * px, ref));
}

// Henyey-Greenstein phase: peaks at cosT=+1 for g>0 (forward scatter = the bright
// halo when looking toward the sun).
float hgPhase(float cosT, float g)
{
    float g2 = g * g;
    float d  = 1.0 + g2 - 2.0 * g * cosT;
    return (1.0 - g2) / (4.0 * PI * pow(max(d, 1e-4), 1.5));
}

// Spot shadow POOL — 3x3 PCF in the light's own atlas tile (4x2 of 1024², see
// shadow_common.glsl), LINEAR-depth compare with a world epsilon. Occludes the
// fog cone of EVERY pooled spot so beams stop at walls / grass cuts them.
float spotLinZ(float zndc, float f)
{
    const float n = 0.5;   // ComputeSpotVPFor near plane
    return n * f / max(f - zndc * (f - n), 1e-4);
}
float spotShadowF(vec3 wp, float range, int tile)
{
    vec4 c = V.spot_pool_vp[tile] * vec4(wp, 1.0);
    if (c.w <= 0.0) return 1.0;
    vec3 ndc = c.xyz / c.w;
    vec2 uv = ndc.xy * 0.5 + 0.5; uv.y = 1.0 - uv.y;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || ndc.z > 1.0) return 1.0;
    float f    = max(range, 1.0);
    float zRef = spotLinZ(ndc.z, f) - 0.08;
    const vec2 kTileScale = vec2(0.25, 0.5);
    vec2  texel = 1.0 / vec2(textureSize(uSpotShadow, 0));
    vec2  tBase = vec2(float(tile & 3), float(tile >> 2)) * kTileScale;
    vec2  tMin  = tBase + texel * 1.5;
    vec2  tMax  = tBase + kTileScale - texel * 1.5;
    vec2  auv   = tBase + uv * kTileScale;
    float sum = 0.0;
    for (int y = -1; y <= 1; ++y)
    for (int x = -1; x <= 1; ++x)
        sum += (zRef <= spotLinZ(texture(uSpotShadow, clamp(auv + vec2(x, y) * texel, tMin, tMax)).r, f)) ? 1.0 : 0.0;
    return sum * (1.0 / 9.0);
}
// Point POOL (cube array) shadow — 1 tap. `cube` = the light's cube-array index.
float pointShadowF(vec3 wp, vec3 lp, float range, int cube)
{
    // LINEAR-depth compare, world epsilon (same fix as the spot pool) — the old
    // 0.01 NDC bias grew to metres at the range edge and ate far NPC shadows.
    vec3 d = wp - lp;
    float z = max(max(abs(d.x), abs(d.y)), abs(d.z));
    const float n = 0.1;                 // kPointNear
    float f = max(range, 1.0);
    float zMap = texture(uPointShadow, vec4(d, float(cube))).r;
    zMap = n * f / max(f - zMap * (f - n), 1e-4);
    return (z - 0.08 <= zMap) ? 1.0 : 0.0;
}

// Terrain/static occlusion for an UN-shadowed lamp (r_light_occ) — ported verbatim
// from light_shade.glsl so the fog/smoke matches the surfaces: if the light is BURIED
// below the top-down surface at its XZ (a lamp/campfire under a roof or in a basement),
// a froxel at/above that surface is only reachable THROUGH the geometry → occlude. This
// is what stops an indoor caster from lighting the smoke/fog OUTSIDE the house.
float lightTerrainOcc(vec3 wp, vec3 lpos)
{
    if (V.light_occ.x < 0.5) return 1.0;
    vec4 lc = V.rain_vp * vec4(lpos, 1.0);
    if (lc.w <= 0.0) return 1.0;
    vec2 luv = lc.xy * 0.5 + 0.5; luv.y = 1.0 - luv.y;
    if (any(lessThan(luv, vec2(0.0))) || any(greaterThan(luv, vec2(1.0)))) return 1.0;
    if (lc.z <= texture(uRainMap, luv).r + V.light_occ.y) return 1.0;   // light at/above the surface → no occlusion
    vec4 fc = V.rain_vp * vec4(wp, 1.0);
    if (fc.w <= 0.0) return 1.0;
    vec2 fuv = fc.xy * 0.5 + 0.5; fuv.y = 1.0 - fuv.y;
    if (any(lessThan(fuv, vec2(0.0))) || any(greaterThan(fuv, vec2(1.0)))) return 1.0;
    float fragBelow = fc.z - texture(uRainMap, fuv).r;                  // >0 = froxel under the surface
    return 1.0 - V.light_occ.w * (1.0 - smoothstep(0.0, V.light_occ.z, max(fragBelow, 0.0)));
}

// P2 — local lights (flashlight CONE, lamp/campfire/anomaly HALO) scattering in the
// fog. Distance falloff × HG phase × (spot) cone gate × occlusion (budget spot/point
// shadow, else the r_light_occ heightfield so indoor lamps don't leak through walls).
vec3 localLights(vec3 world, vec3 viewDir)
{
    int n = int(V.lightParams.x);
    vec3 acc = vec3(0.0);
    for (int i = 0; i < n; ++i) {
        vec3  toL   = V.lights[i].pos.xyz - world;
        float dist  = length(toL);
        float range = V.lights[i].pos.w;
        if (range <= 0.0 || dist >= range) continue;
        vec3  Ld    = toL / max(dist, 1e-3);
        // Narrow beams get the windowed falloff (far half of the beam still
        // glows in the fog) — matches the surface shaders (light_shade.glsl).
        float atten;
        if (V.lights[i].dir.w > 0.87 && (int(V.lights[i].color.w + 0.5) & 1) == 1) {
            atten = 1.0 - (dist * dist) / (range * range);
            atten *= atten;
        } else {
            atten = 1.0 - dist / range;
            atten *= atten;                               // smooth falloff
        }
        // color.w: bit0 = spot, bit1 = VOLUMETRIC-flagged lamp (R4 shows a beam
        // for these — pole lamps / headlights); boost their in-scatter so the
        // shaft reads even in light haze. bits 2+ = pool index+1 (spot tile OR
        // point cube — a light is one or the other, so the bits never collide).
        int   lw    = int(V.lights[i].color.w + 0.5);
        bool  isSpot = (lw & 1) == 1;
        if (isSpot) {                                     // spot cone
            float cosCone = V.lights[i].dir.w;
            float d = dot(-Ld, V.lights[i].dir.xyz);
            if (d < cosCone) continue;
            atten *= smoothstep(cosCone, mix(cosCone, 1.0, 0.5), d);
        }
        if ((lw & 2) == 2) atten *= 3.0;                  // volumetric lamp beam boost
        int pool = (lw >> 2) - 1;                         // pool index (bits 2+, idx+1)
        if (isSpot) {
            if (pool >= 0) atten *= spotShadowF(world, range, pool);
        } else if (pool >= 0) {
            atten *= pointShadowF(world, V.lights[i].pos.xyz, range, pool);   // pooled campfire cube
        } else {
            // Un-shadowed OMNI lamp → no leak through roof/floor (matches
            // light_shade.glsl): the top-down heightfield buries basement lamps.
            atten *= lightTerrainOcc(world, V.lights[i].pos.xyz);
        }
        // Local lights use their OWN anisotropy (lightParams.z), gentler than the
        // sun's sharp forward peak (V.fog.w) — a torch shone AT the camera would
        // otherwise spike into a blinding, HDR-desaturated (cold-white) glare.
        // clamp keeps hgPhase valid if the value ever arrives out of range.
        float gL = clamp(V.lightParams.z, 0.0, 0.95);
        float ph = hgPhase(dot(viewDir, Ld), gL);          // scatter toward the camera
        acc += V.lights[i].color.rgb * (atten * ph);
    }
    return acc * V.lightParams.y;                          // r_vol_lights boost
}

// Stage-2: smoke media density at an arbitrary WORLD point (inverse of the splat/inject
// froxel basis — same math as vol_splat). 0 outside the grid frustum. Used to march
// optical depth toward the sun for smoke self-shadow.
float smokeDensityAt(vec3 wpos)
{
    vec3  rel = wpos - V.camPos.xyz;
    float vz  = dot(rel, V.camDir.xyz);
    if (vz <= V.zParams.x || vz >= V.zParams.y) return 0.0;
    float ndcx = dot(rel, V.camRightT.xyz) / (vz * dot(V.camRightT.xyz, V.camRightT.xyz));
    float ndcy = dot(rel, V.camTopT.xyz)   / (vz * dot(V.camTopT.xyz,   V.camTopT.xyz));
    if (abs(ndcx) > 1.0 || abs(ndcy) > 1.0) return 0.0;
    vec2  smuv = vec2(ndcx * 0.5 + 0.5, (1.0 - ndcy) * 0.5);
    float smw  = Froxel_SliceFromViewZ(vz, V.zParams.x, V.zParams.z);
    return texture(uSmokeMedia, vec3(smuv, smw)).a;
}

void main()
{
    ivec3 id = ivec3(gl_GlobalInvocationID);
    int dimX = int(V.gridParams.x), dimY = int(V.gridParams.y), dimZ = int(V.gridParams.z);
    if (id.x >= dimX || id.y >= dimY || id.z >= dimZ) return;

    float nearZ = V.zParams.x, logFN = V.zParams.z;

    // Froxel sample world pos with the sub-froxel JITTER (temporal.xyz, Halton per
    // frame) → supersamples the shadow/density. cur (below) uses THIS.
    vec2 uv  = (vec2(id.xy) + 0.5 + V.temporal.xy) / vec2(dimX, dimY);
    vec2 ndc = vec2(uv.x * 2.0 - 1.0, 1.0 - 2.0 * uv.y);
    float viewZ = Froxel_ViewZFromSlice((float(id.z) + 0.5 + V.temporal.z) / float(dimZ), nearZ, logFN);
    vec3 ray   = V.camDir.xyz + V.camRightT.xyz * ndc.x + V.camTopT.xyz * ndc.y;
    vec3 world = V.camPos.xyz + ray * viewZ;

    // UNJITTERED froxel CENTRE — used ONLY for the history reprojection. The history
    // is indexed at integer froxel centres, so reprojecting the JITTERED world made
    // the lookup wobble by the jitter every frame → it never settled = "trembling".
    // Reprojecting the stable centre lets the jittered samples accumulate cleanly.
    vec2 uvC  = (vec2(id.xy) + 0.5) / vec2(dimX, dimY);
    vec2 ndcC = vec2(uvC.x * 2.0 - 1.0, 1.0 - 2.0 * uvC.y);
    float viewZC = Froxel_ViewZFromSlice((float(id.z) + 0.5) / float(dimZ), nearZ, logFN);
    vec3 worldCenter = V.camPos.xyz + (V.camDir.xyz + V.camRightT.xyz * ndcC.x + V.camTopT.xyz * ndcC.y) * viewZC;

    // Sky visibility (0 under a roof, 1 open) drives BOTH the indoor density boost
    // and the ambient occlusion — sampled once.
    float skyv = skyVis(world);

    // Base extinction: base × exponential height falloff above the anchor.
    // This drives the TRANSMITTANCE (scene dimming) — kept at base so small rooms
    // aren't over-darkened (over-darkening is why the fog read as "just darker").
    float extinction = V.fog.x * exp(-max(0.0, world.y - V.fog.y) * V.fog.z);
    // P3 — animated dust/mist: mottle the DENSITY (both the extinction below AND the
    // scatter inherit it) with slowly-drifting 3D noise → visible wisps/pockets in the
    // lit fog, the "living air" feel. Applied to density (not scatter-only) so it
    // survives the Z-integration + temporal averaging that washed the subtle version out.
    if (V.noiseParams.x > 0.001) {
        float t  = V.noiseParams.z * V.noiseParams.w;
        vec3  np = world * V.noiseParams.y + vec3(t, t * 0.4, t * 0.8);
        extinction *= max(1.0 + V.noiseParams.x * (fbm3(np) - 0.5) * 2.0, 0.0);
    }
    // Indoor density boost (fog2.z, ×(1+indoor) under a roof) makes the thin diffuse
    // AMBIENT haze visible across a short interior sightline. It is applied ONLY to
    // the ambient term: the SUN BEAM and LOCAL LIGHTS are strong DIRECTIONAL sources,
    // and boosting THEM ×(1+indoor) indoors accumulated over the Z-march into a WHITE
    // BLOWOUT (the campfire column / a god-ray through a doorway). Those use the BASE
    // extinction so bright sources stay bounded; the ambient haze keeps the interior
    // atmosphere. NOTE outdoors skyv=1 → boost=0 → ambDens==extinction → this is
    // byte-identical to the old outdoor look (only interiors change).
    float ambDens = extinction * (1.0 + V.fog2.z * (1.0 - skyv));

    // Sun in-scatter (cascade-occluded → the beam stops at walls/roof, passing
    // only through openings like windows) + sky ambient (occluded by sky visibility
    // → interiors don't glow uniformly, so the sun beam stands out).
    vec3  viewDir = normalize(ray);
    vec3  toSun   = normalize(-V.sun_dir.xyz);
    float phase   = hgPhase(dot(viewDir, toSun), V.fog.w);
    float occ     = sunShadow(world);
    // Ambient: FULL outdoors (skyv 1), but only a FLOOR fraction (fog2.y) under a
    // roof — interiors keep visible haze (real rooms aren't pitch black) while the
    // sun beam from a window still stands out against the dimmer indoor fill.
    float ambK    = V.fog2.y + (1.0 - V.fog2.y) * skyv;
    vec3  ambient = V.sky_ambient.rgb * ambK;
    // The SUN term gets its own boost (fog2.w) so the directional shaft pops THROUGH
    // the ambient haze. Transmittance (a) uses the base extinction.
    vec3  sunScatter = V.sun_color.rgb * (phase * occ * V.fog2.w);
    // Sun beam + local lights at BASE extinction (no indoor ×boost → no blowout);
    // the ambient haze at the indoor-boosted density (the visible interior glow).
    // Local lights carry their OWN boost (r_vol_lights) so lamps can be dialed.
    vec3  localL     = localLights(world, viewDir);
    vec3  inscatter  = sunScatter * V.fog2.x * extinction
                     + ambient    * V.fog2.x * ambDens
                     + localL * extinction;

    // Atmospheric scattering (r_atmo): Rayleigh (blue, wavelength-dependent) + Mie
    // (forward halo toward the sun) in-scatter of the SUN's light along the sightline.
    // Distant surfaces gain a BLUE veil away from the sun and a WARM halo toward it
    // (aerial perspective) — and since sun_color is the time-of-day colour, dawn/dusk
    // read warm automatically. Sun-shadowed (stops at terrain), rides the base air
    // density so it accumulates with distance through the Z-integration.
    if (V.atmo.x > 0.0) {
        float cosT = dot(viewDir, toSun);
        float pR   = 0.05968310 * (1.0 + cosT * cosT);   // Rayleigh phase 3/(16π)(1+cos²)
        float pM   = hgPhase(cosT, V.atmo.w);            // Mie forward halo
        vec3  atmoScatter = V.sun_color.rgb * occ
                          * (V.atmoR.rgb * (pR * V.atmo.y) + vec3(pM * V.atmo.z));
        inscatter += atmoScatter * V.fog2.x * extinction;
    }

    // ---- Stage-1 VMS: smoke as participating media. Sample the splatted+resolved
    // smoke media at this froxel (same normalized frustum coords as the fog, so it
    // lines up). Smoke adds its OWN extinction and scatters the SAME light as fog
    // (sun beam + ambient + local lights), tinted by the particles' own albedo —
    // so smoke catches god-rays / the flashlight cone and self-occludes via the
    // extinction the integrate pass marches. v1: scatter uses the lit terms without
    // the indoor boost (bright sources stay bounded, same reasoning as the sun beam).
    if (V.smokeParams.x > 0.0) {
        vec3  smUVW = vec3(uv, (float(id.z) + 0.5 + V.temporal.z) / float(dimZ));
        vec4  sm    = texture(uSmokeMedia, smUVW);
        float smokeDens = sm.a * V.smokeParams.x;

        // Stage-2 SELF-SHADOW: march the smoke media from here toward the sun and
        // accumulate optical depth → the cloud's sun side stays bright, its far/deep
        // side darkens (gives the cloud real volume/form). Only smoky froxels march
        // (cheap); gated on r_vol_smoke_shadow (smokeParams.z), step = smokeParams.w.
        float sunVis = 1.0;
        if (V.smokeParams.z > 0.0 && sm.a > 1e-5) {
            float mstep = V.smokeParams.w;
            // Seed with THIS froxel's own density so even a thin / single-cell cloud
            // self-attenuates the sun (a smoky cell dims the light passing through it) —
            // the march on top adds the directional gradient (far/deep side darker) when
            // the cloud is thick enough to span several cells toward the sun.
            float tau = sm.a;
            vec3  sp  = world;
            for (int s = 0; s < 6; ++s) { sp += toSun * mstep; tau += smokeDensityAt(sp); }
            sunVis = exp(-tau * mstep * V.smokeParams.z);
        }

        if (V.smokeParams.y > 0.5) {
            // DEBUG (r_vol_smoke_debug): isolate the smoke in the composite. 1 = colour
            // (albedo glow), 2 = density heatmap (grey), 3 = SELF-SHADOW sunVis (blue =
            // shadowed/deep side, warm = sun-lit side) → shows the Stage-2 gradient
            // directly. Empty froxels stay 0 → scene shows through clean.
            float dd      = sm.a;
            float present = (dd > 1e-4) ? 1.0 : 0.0;
            vec3  col;
            if      (V.smokeParams.y > 2.5) col = mix(vec3(0.0, 0.06, 0.5), vec3(1.0, 0.25, 0.0), clamp((1.0 - sunVis) * 4.0, 0.0, 1.0)) * (5.0 * present);  // shadow heatmap: blue=lit, red=self-shadowed (×4 so faint shows)
            else if (V.smokeParams.y > 1.5) col = vec3(dd) * 25.0;
            else                            col = (sm.rgb + 0.15) * dd * 25.0;
            inscatter  = col;
            extinction = dd;
        } else if (smokeDens > 1e-5) {
            // Sun term attenuated by the smoke toward the sun; ambient + local lights full.
            vec3 lit = (sunScatter * sunVis + ambient) * V.fog2.x + localL;
            inscatter  += sm.rgb * lit * smokeDens;
            extinction += smokeDens;
        }
    }

    vec4 cur = vec4(inscatter, extinction);

    // ---- Temporal accumulation: reproject this froxel's world pos into the PREV
    // frame's volume and EMA-blend (alpha = temporal.w). Converges the jittered
    // samples into a clean, dense volume; rejected (off-screen / behind cam) → cur.
    float alpha = V.temporal.w;
    if (alpha > 0.0) {
        vec4 pc = V.prevViewProj * vec4(worldCenter, 1.0);
        if (pc.w > 0.0) {
            vec2 puv = (pc.xy / pc.w) * 0.5 + 0.5;
            puv.y = 1.0 - puv.y;
            float pvz = dot(worldCenter - V.prevCamPos.xyz, V.prevCamDir.xyz);   // prev view-space depth
            float pw  = Froxel_SliceFromViewZ(max(pvz, 1e-3), V.prevCamPos.w, V.prevCamDir.w);
            if (all(greaterThanEqual(vec3(puv, pw), vec3(0.0))) &&
                all(lessThanEqual   (vec3(puv, pw), vec3(1.0)))) {
                // Full EMA blend (no rejection): the moving-sun shadow crawl through
                // foliage is exactly the noise temporal accumulation is meant to
                // average into a stable shaft. Rejecting it (drop history on change)
                // re-exposed the crawl = "trembling". Off-screen → reprojection bounds
                // above already fall back to `cur`, so disocclusion can't ghost badly.
                vec4 hist = textureLod(uHistory, vec3(puv, pw), 0.0);
                cur = mix(cur, hist, alpha);
            }
        }
    }

    imageStore(uScatter, id, cur);
}
