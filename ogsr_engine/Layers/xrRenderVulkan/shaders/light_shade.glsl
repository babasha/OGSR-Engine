// xrRenderVulkan - shared dynamic-light shading (point/spot, clustered or UBO
// fallback) + unshadowed-lamp terrain occlusion. Was copy-pasted into world_lmap
// / world_terrain / world_vlit. #include AFTER light_ubo.glsl, shadow_common.glsl
// (spot/pointShadowF) AND cluster_lights.glsl (clusterOfFragment + the SSBOs).
#ifndef LIGHT_SHADE_GLSL
#define LIGHT_SHADE_GLSL

// Terrain/static occlusion for a dynamic light (r_light_occ): if the light is
// BURIED (below the ground surface at its own XZ), march the ground-height map
// between fragment and light. A fragment also below the surface (same buried
// pocket) stays lit; one at/above is only reachable THROUGH the ground -> occlude.
float lightTerrainOcc(vec3 wp, vec3 lpos)
{
    if (L.light_occ.x < 0.5) return 1.0;
    vec4 lc = L.rain_vp * vec4(lpos, 1.0);
    if (lc.w <= 0.0) return 1.0;
    vec2 luv = lc.xy * 0.5 + 0.5; luv.y = 1.0 - luv.y;
    if (any(lessThan(luv, vec2(0.0))) || any(greaterThan(luv, vec2(1.0)))) return 1.0;
    if (lc.z <= texture(uRainMap, luv).r + L.light_occ.y) return 1.0;   // light at/above the surface -> no occlusion
    vec4 fc = L.rain_vp * vec4(wp, 1.0);
    if (fc.w <= 0.0) return 1.0;
    vec2 fuv = fc.xy * 0.5 + 0.5; fuv.y = 1.0 - fuv.y;
    if (any(lessThan(fuv, vec2(0.0))) || any(greaterThan(fuv, vec2(1.0)))) return 1.0;
    float fragBelow = fc.z - texture(uRainMap, fuv).r;                  // >0 = fragment under the surface
    return 1.0 - L.light_occ.w * (1.0 - smoothstep(0.0, L.light_occ.z, max(fragBelow, 0.0)));
}

// Shade ONE dynamic point/spot light: linear-squared falloff, N.L diffuse, smooth
// spot cone, + the two shadow-budget lights' shadow maps & cookie. Fields are
// passed in so this serves BOTH the UBO array (fallback) and the clustered SSBO.
// gi = the light's GLOBAL index (for the shadow picks).
vec3 shadeDynLight(vec4 lpos, vec4 lcol, vec4 ldir, vec3 wp, vec3 N, int gi, int sIdx, int pIdx)
{
    vec3  dv = lpos.xyz - wp;
    float r  = lpos.w;
    float d2 = dot(dv, dv);
    if (d2 >= r * r) return vec3(0.0);
    float d   = sqrt(max(d2, 1e-6));
    vec3  ld  = dv / d;
    float att = 1.0 - d / r;
    att *= att;
    if (lcol.w > 0.5)   // spot cone
        att *= clamp((dot(-ld, ldir.xyz) - ldir.w) / max(1.0 - ldir.w, 1e-3), 0.0, 1.0);
    vec3 tint = lcol.rgb;
    if (gi == sIdx) {
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
    else if (gi == pIdx) att *= pointShadowF(wp, lpos.xyz, r);
    else att *= lightTerrainOcc(wp, lpos.xyz);   // UNshadowed lamps: heightfield terrain occlusion
    return tint * (att * max(dot(N, ld), 0.0));
}

// Dynamic light accumulation. Clustered (r_clustered): iterate only the froxel's
// lights from the SSBO. Fallback: the per-fragment loop over the UBO's 16.
vec3 dynLights(vec3 wp, vec3 N)
{
    int sIdx = int(L.shadow_params.x);
    int pIdx = int(L.shadow_params.y);
    vec3 acc = vec3(0.0);
    if (L.cluster_params.w > 0.5) {
        int ci  = clusterOfFragment(wp, L.eye_pos.xyz, L.cam_dir.xyz, gl_FragCoord.xy, L.ao_params.xy, L.cluster_params, L.cluster_params2);
        int mxp = int(L.cluster_params2.w);
        int cnt = int(clusterGrid[ci]);
        for (int k = 0; k < cnt; ++k) {
            int gi = int(clusterIndex[ci * mxp + k]);
            acc += shadeDynLight(clusterLights[gi].pos, clusterLights[gi].color, clusterLights[gi].dir, wp, N, gi, sIdx, pIdx);
        }
    } else {
        int n = int(L.counts.x + 0.5);
        for (int i = 0; i < n; ++i)
            acc += shadeDynLight(L.lights[i].pos, L.lights[i].color, L.lights[i].dir, wp, N, i, sIdx, pIdx);
    }
    return acc;
}

#endif // LIGHT_SHADE_GLSL
