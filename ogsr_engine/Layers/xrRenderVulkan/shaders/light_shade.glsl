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
    // Distance falloff. NARROW beams (headlights/searchlights): windowed
    // (1-(d/r)^2)^2 — (1-d/r)^2 is basically dead past 60% range, so a 35 m
    // headlight painted nothing beyond ~20 m. Wide lamps keep the classic curve.
    float att;
    if (lcol.w > 0.5 && ldir.w > 0.87) {
        att = 1.0 - (d2 / (r * r));
        att *= att;
    } else {
        att = 1.0 - d / r;
        att *= att;
    }
    if (lcol.w > 0.5) {   // spot cone
        float ca = dot(-ld, ldir.xyz);
        // NARROW beams (headlights/searchlights, cos(half) > 0.87): full strength
        // anywhere INSIDE the cone + a spill fade out to 2x the cone angle (real
        // fixtures spill well past the bright core). The default ramp peaks only
        // ON the axis, so the ground pool a headlight visibly paints sat at the
        // cone edge at ~0 and terrain/grass read as "not reacting" to the beam.
        if (ldir.w > 0.87) {
            float co = 2.0 * ldir.w * ldir.w - 1.0;   // cos(2*half)
            att *= clamp((ca - co) / max(ldir.w - co, 1e-3), 0.0, 1.0);
        } else
            att *= clamp((ca - ldir.w) / max(1.0 - ldir.w, 1e-3), 0.0, 1.0);
    }
    vec3 tint = lcol.rgb;
    if (lcol.w > 0.5) {
        // Spot shadow POOL: every pooled spot samples its own atlas tile —
        // several headlights/searchlights + the flashlight all shadow at once.
        int tile = spotTileOf(gi);
        if (tile >= 0) att *= spotShadowF(wp, r, tile);
        // Flashlight cookie (R4 projective light texture): the beam pattern
        // projected through the cookie light's tile VP (L.spot_vp).
        if (gi == sIdx && L.shadow_params.z > 0.5) {
            vec4 cc = L.spot_vp * vec4(wp, 1.0);
            if (cc.w > 0.0) {
                vec2 cuv = (cc.xy / cc.w) * 0.5 + 0.5;
                cuv.y = 1.0 - cuv.y;
                tint *= textureLod(uSpotCookie, clamp(cuv, 0.0, 1.0), 0.0).rgb;
            }
        }
    }
    else {
        // Point shadow POOL: a pooled point samples its own cube; an UNshadowed
        // omni lamp instead gets heightfield terrain occlusion (stops basement
        // lamps lighting through the ground). SPOTS are exempt from the latter:
        // the top-down map sees any fixture above them and calls the light
        // "buried" → aimed projector beams were silently killed.
        int cube = pointCubeOf(gi);
        if (cube >= 0) att *= pointShadowF(wp, lpos.xyz, r, cube);
        else           att *= lightTerrainOcc(wp, lpos.xyz);
    }
    float ndl = dot(N, ld);
    // NARROW beams (headlights/searchlights, cone < ~30°: cos(half) > 0.87):
    // wrap the diffuse — a near-horizontal beam grazes the ground at N·L ≈ 0.05
    // and painted no light pool where it visibly lands. Wide spots (flashlight
    // 70°, pole lamps 120°) keep plain Lambert.
    if (lcol.w > 0.5 && ldir.w > 0.87) ndl = (ndl + 0.4) * (1.0 / 1.4);
    return tint * (att * max(ndl, 0.0));
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
