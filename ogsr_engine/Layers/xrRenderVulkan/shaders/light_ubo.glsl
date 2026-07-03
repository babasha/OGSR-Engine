// xrRenderVulkan - shared per-frame lighting descriptor set (set 1).
//
// The DynLight struct, the big `Lighting` UBO (set 1, binding 0) and the set-1
// sampler bindings (1..13) were copy-pasted verbatim into world_lmap /
// world_terrain / world_vlit and their comments drifted out of sync. This is the
// SINGLE SOURCE OF TRUTH: change the UBO layout here once. Layout must match the
// std140 fill in vk_env_light.{h,cpp}. #include AFTER `#version`, and BEFORE any
// helper that reads `L` (shadow_common / env_common / light_shade / wetness).
#ifndef LIGHT_UBO_GLSL
#define LIGHT_UBO_GLSL

// The EnvLight descriptor set binds at set 1 (world / grass) and set 2 (skinned /
// tree) - the SAME shared layout (vk_env_light.h). Define ENV_SET before #include
// to pick the index (default 1). Lets tree/detail reuse this single UBO source.
#ifndef ENV_SET
#define ENV_SET 1
#endif

struct DynLight {
    vec4 pos;     // xyz = world position, w = range
    vec4 color;   // rgb = colour,         w = 1 spot / 0 point
    vec4 dir;     // xyz = spot direction, w = cos(cone/2)
};

layout(set = ENV_SET, binding = 0) uniform Lighting {
    vec4 sun_dir;        // xyz = travel dir (downward)
    vec4 sun_color;      // rgb (final: r_sun_boost premultiplied)
    vec4 hemi_color;     // rgb
    vec4 ambient;        // rgb (final: r_ambient_floor added)
    mat4 sun_vp;         // sun light view-proj (far cached map lookup)
    vec4 counts;         // x = dynamic light count
    DynLight lights[16];
    mat4 spot_vp;        // spot (flashlight) shadow view-proj
    vec4 shadow_params;  // x=spot-shadowed light idx (-1 none), y=point-shadowed idx, z=cookie on, w=VSM gate
    mat4 sun_near_vp;    // sun cascade 0 view-proj (25 m, per-frame, R4 scheme)
    mat4 sun_c1_vp;      // sun cascade 1 view-proj (60 m, per-frame)
    vec4 fog_color;      // rgb haze colour (env)
    vec4 fog_params;     // x=-near*r, y=near, z=far, w=r; fog = saturate(dist*w + x)
    vec4 eye_pos;        // xyz camera world pos
    vec4 sky_params;     // x=cube cross-fade weight, y=ambient scale, z=sample LOD, w=time
    vec4 ao_params;      // x=1/screenW, y=1/screenH, z=AO strength (0=off), w=ssao debug
    mat4 rain_vp;        // straight-down ortho VP for the rain occlusion map
    vec4 rain_params;    // x=rain density, y=wetness, z=darken (<0 = wet debug), w=reflection scale
    mat4 scene_vp;       // (SSR puddles - declared for layout match, unused in fragment)
    vec4 cam_dir;
    vec4 cam_rightT;
    vec4 cam_topT;
    vec4 pom_params;     // x=POM amplitude (UV), y=max steps, z=fade dist (m), w=on
    vec4 pom_params2;    // x=blur, y=normal, z=self-shadow, w=contact AO
    vec4 pom_params3;    // x=debug view, y=ao_flat, z=ceil strength, w=floor strength
    vec4 pom_params4;    // x=terrain POM enable, y=detail-normal, z=micro-AO, w=terrain debug
    vec4 pom_params5;    // x=terrain gloss, y=glass opacity, z=mud footprint strength (r_mud_deform), w=puddle debug
    vec4 pom_params6;    // x=water-sim enable, y=murk, z=refract
    vec4 pom_params7;    // SSS puddles: x=enable, y=level (coverage), z=mud print POM carve depth (r_mud_depth), w=macro scale
    vec4 cluster_params; // x=sliceScale, y=sliceBias, z=near, w=enable (0 off, 1 on, 2 debug)
    vec4 cluster_params2;// x=gridX, y=gridY, z=gridZ, w=maxLightsPerCluster
    vec4 light_occ;      // x=enable, y=bury bias, z=march bias, w=strength (terrain/static light occlusion)
    vec4 sf_params;      // Surface Field: x=enable (r_sf), y=debug view (r_sf_debug 0..5), z=derive eps m (r_sf_eps), w=snow amount (r_snow)
    // Snow footprint deformation: a ring of recent foot contacts (vk_env_light fills
    // from Skinned_CollectFeet). snow_displace.glsl (VS) carves the snow at these.
    vec4 deform_count;       // x=active count, y=press depth (m), z=ridge height (m), w=enable
    vec4 deform_stamps[256]; // xy=world XZ, z=radius (m), w=strength 0..1 (time-decayed, ~1 min trail)
    // Texture-based deform (r_snow_deform_tex): a dense persistent press field in a
    // player-centred ortho box (vk_deform), sampled instead of the stamp loop.
    mat4 deform_vp;          // world -> deform-texture NDC (straight-down ortho)
    vec4 deform_tex;         // x=enable, y=1/size, z=max depth (m), w=world metres per texel
} L;

layout(set = ENV_SET, binding = 1)  uniform sampler2D   uShadow;       // far sun map (cached)
layout(set = ENV_SET, binding = 2)  uniform sampler2D   uSpotShadow;   // spot (flashlight) shadow map
layout(set = ENV_SET, binding = 3)  uniform samplerCube uPointShadow;  // point (campfire) shadow cube
layout(set = ENV_SET, binding = 4)  uniform sampler2D   uShadowNear;   // sun cascade 0 (~0.61 cm texels)
layout(set = ENV_SET, binding = 5)  uniform sampler2D   uShadowC1;     // sun cascade 1 (~1.46 cm texels)
layout(set = ENV_SET, binding = 6)  uniform samplerCube uSky0;         // sky ambient cube 0 (weather A)
layout(set = ENV_SET, binding = 7)  uniform samplerCube uSky1;         // sky ambient cube 1 (weather B)
layout(set = ENV_SET, binding = 8)  uniform sampler2D   uAO;           // GTAO (half-res, bilinear upsample)
layout(set = ENV_SET, binding = 9)  uniform sampler2D   uRainMap;      // top-down rain occlusion (wetness mask)
layout(set = ENV_SET, binding = 10) uniform sampler2D   uSpotCookie;   // flashlight beam texture (cookie)
layout(set = ENV_SET, binding = 11) uniform sampler2D   uWater;        // water depth (flow sim, metres)
layout(set = ENV_SET, binding = 12) uniform sampler2D   uFlow;         // water velocity (flow sim, uv/sec)
layout(set = ENV_SET, binding = 13) uniform sampler2D   uGround;       // top-down ground-height (statics+terrain, no trees)
layout(set = ENV_SET, binding = 20) uniform sampler2D   uDeform;       // snow deform press field (vk_deform; r_snow_deform_tex)
// SSIL one-bounce indirect light (half-res, r_ssil; black until ready). Declared
// here (harmless in VS/TES — just a binding) but consumed only in fragment stages
// via ssilBoost() in env_common.glsl / the local copies (which use gl_FragCoord,
// so the FUNCTION cannot live in this stage-shared header).
layout(set = ENV_SET, binding = 21) uniform sampler2D   uIL;

#endif // LIGHT_UBO_GLSL
