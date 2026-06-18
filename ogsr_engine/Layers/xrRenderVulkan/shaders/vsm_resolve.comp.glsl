#version 450
#extension GL_GOOGLE_include_directive : require
// xrRenderVulkan — VSM temporal resolve (TAA-for-shadows). One thread per screen
// pixel: reconstruct world from the prepass depth, sample the VSM atlas (the same
// world->light->page->atlas mapping the receiver used to do directly), then blend
// against a reprojected history with an EMA. The clipmap origin is sub-texel JITTERED
// each frame (vk_vsm BeginFrame) so the per-frame texel quantization of the shadow
// edge decorrelates; the EMA averages it → the "crawling snake" along moving-sun
// shadow edges resolves to a smooth, stable soft edge. Output is a screen-space mask
// (R = lit factor, G = distance to this frame's camera, for history disocclusion
// rejection). Receivers then just sample R by screen UV (vsm_sample.glsl). See vk_vsm.cpp.
#include "vsm_common.glsl"

layout(local_size_x = 8, local_size_y = 8) in;

layout(set = 0, binding = 0) uniform sampler2D uDepth;    // current scene prepass depth
layout(set = 0, binding = 1) uniform sampler2D uAtlas;    // STATIC VSM atlas (opaque + trees)
layout(set = 0, binding = 7) uniform sampler2D uAtlasDyn; // DYNAMIC VSM atlas (NPC + grass)
layout(set = 0, binding = 2) readonly buffer VsmPageTable    { uint vsmPageTable[]; };     // STATIC virtual->slot
layout(set = 0, binding = 8) readonly buffer VsmPageTableDyn { uint vsmPageTableDyn[]; };  // DYNAMIC virtual->slot
layout(set = 0, binding = 3) uniform VsmClipmap {
    mat4 view;                 // world -> sun light space
    vec4 level[VSM_LEVELS];    // xy = level origin (light XY of texel 0,0), z = extent (m)
    vec4 zparams;              // x = zNear, y = 1/(zFar-zNear), z = depth-compare bias
} vsmC;
layout(set = 0, binding = 4) uniform sampler2D uHistory; // previous frame's mask (RG = shadow, dist)
layout(set = 0, binding = 5, rgba16f) uniform writeonly image2D uOut;
layout(set = 0, binding = 6) uniform Resolve {
    mat4 invViewProj;    // current clip -> world
    mat4 prevViewProj;   // world -> previous-frame clip (history reproject)
    vec4 prevCamPos;     // xyz = previous frame camera (for history distance check)
    vec4 curCamPos;      // xyz = this frame camera (stored as G for next frame)
    vec4 screen;         // xy = pixel dims, zw = 1/dims
    vec4 params;         // x = history weight (alpha), y = reject tolerance, z = historyValid, w = unused
} R;

// VSM atlas sample (3x3 PCF) — mirrors the page mapping vsm_page.vert rasterized with.
// Returns lit factor 1 = lit .. 0 = shadowed; out of clipmap / unmapped page → lit.
float sampleVSM(vec3 wp)
{
    vec3 lp = (vsmC.view * vec4(wp, 1.0)).xyz;
    vec2 luv; ivec2 page;
    int  L = vsmSelect(lp.xy, vsmC.level, luv, page);
    if (L < 0) return 1.0;

    int  idx   = vsmPageIndex(L, page);
    uint slotS = vsmPageTable[idx];      // STATIC atlas slot (toroidal-cached; 6144 grid)
    uint slotD = vsmPageTableDyn[idx];   // DYNAMIC atlas slot (demand-allocated; 2048 grid)
    bool hasS  = slotS < uint(VSM_MAX_PHYS_S);
    bool hasD  = slotD < uint(VSM_MAX_PHYS);
    if (!hasS && !hasD) return 1.0;      // page resident in neither atlas → lit

    vec2  pageLocal = luv * float(VSM_PAGES_AXIS) - vec2(page);   // [0,1) within the page
    vec2  baseS = vec2(float(slotS % uint(VSM_ATLAS_W_S)), float(slotS / uint(VSM_ATLAS_W_S)));
    vec2  baseD = vec2(float(slotD % uint(VSM_ATLAS_W)),   float(slotD / uint(VSM_ATLAS_W)));
    float zHere = (lp.z - vsmC.zparams.x) * vsmC.zparams.y;
    float bias  = vsmC.zparams.z;
    const vec2 dimS = vec2(float(VSM_ATLAS_W_S), float(VSM_ATLAS_H_S));   // static atlas (6144)
    const vec2 dimD = vec2(float(VSM_ATLAS_W),   float(VSM_ATLAS_H));     // dynamic atlas (2048)

    const float tp    = 1.0 / float(VSM_PAGE_SIZE);   // one page texel, page-local units
    const float inset = 0.5 * tp;
    float lit = 0.0;
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        vec2 pl = clamp(pageLocal + vec2(float(dx), float(dy)) * tp, vec2(inset), vec2(1.0 - inset));
        // Nearer occluder across both atlases (each looked up via its own slot + grid). An
        // empty page in either reads its CLEAR / clamp-to-white = 1.0 = no occluder.
        float occ = 1.0;
        if (hasS) occ = min(occ, texture(uAtlas,    (baseS + pl) / dimS).r);
        if (hasD) occ = min(occ, texture(uAtlasDyn, (baseD + pl) / dimD).r);
        lit += (zHere - bias > occ) ? 0.0 : 1.0;
    }
    return lit * (1.0 / 9.0);
}

void main()
{
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    if (px.x >= int(R.screen.x) || px.y >= int(R.screen.y)) return;

    vec2  uv   = (vec2(px) + 0.5) * R.screen.zw;
    float zndc = texture(uDepth, uv).r;
    if (zndc >= 0.99999) { imageStore(uOut, px, vec4(1.0, 1e6, 0.0, 0.0)); return; }   // sky → lit

    // Reconstruct world (D3D NDC, y-up — matches vsm_mark.comp / ssao.frag).
    vec4 clip  = vec4(uv.x * 2.0 - 1.0, 1.0 - 2.0 * uv.y, zndc, 1.0);
    vec4 world = R.invViewProj * clip;
    vec3 wp    = world.xyz / world.w;

    float cur  = sampleVSM(wp);
    float dist = length(wp - R.curCamPos.xyz);   // stored for next frame's reject test

    float outShadow = cur;
    if (R.params.z > 0.5) {                       // history valid (not first frame / no resize)
        vec4 pc = R.prevViewProj * vec4(wp, 1.0);
        if (pc.w > 0.0) {
            vec2 puv = (pc.xy / pc.w) * vec2(0.5, -0.5) + 0.5;   // prev-frame screen UV (same y-flip)
            if (all(greaterThanEqual(puv, vec2(0.0))) && all(lessThanEqual(puv, vec2(1.0)))) {
                vec2  hist = texture(uHistory, puv).rg;          // (shadowPrev, distFromPrevCam)
                float expectPrev = length(wp - R.prevCamPos.xyz);
                // Same static surface last frame → stored distance ≈ expected. Reject
                // (use current only) on disocclusion so silhouettes don't ghost.
                if (abs(hist.r) <= 1.0001 && abs(hist.g - expectPrev) <= R.params.y * expectPrev + 0.05)
                    outShadow = mix(cur, hist.r, R.params.x);
            }
        }
    }
    imageStore(uOut, px, vec4(outShadow, dist, 0.0, 0.0));
}
