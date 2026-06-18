// xrRenderVulkan - shared environment terms: GTAO visibility/bent-normal, colored
// AO, hemisphere sky ambient, and rain-map visibility. Was copy-pasted into
// world_lmap / world_terrain / world_vlit. #include AFTER light_ubo.glsl AND
// shadow_common.glsl (rainVis uses cascTap).
#ifndef ENV_COMMON_GLSL
#define ENV_COMMON_GLSL

// GTAO visibility (R4 combine_1.ps: occludes hemi+ambient only, never sun/dyn).
// Strength is an EXPONENT: 0 = off (->1.0), 1 = raw, 2-3 deepens corners.
float gtaoVis()
{
    float ao = textureLod(uAO, gl_FragCoord.xy * L.ao_params.xy, 0.0).r;
    return pow(clamp(ao, 0.0, 1.0), L.ao_params.z);
}

// GTAO bent normal (gba of the AO RT, world-space): the average UNOCCLUDED
// direction. Sky fill sampled along it pulls ambient from where the hemisphere
// is open. Falls back to the geometric normal when AO is off or degenerate.
vec3 gtaoBentN(vec3 fallbackN)
{
    if (L.ao_params.z <= 0.0) return fallbackN;
    vec3 b = textureLod(uAO, gl_FragCoord.xy * L.ao_params.xy, 0.0).gba * 2.0 - 1.0;
    return (dot(b, b) > 0.25) ? normalize(b) : fallbackN;
}

// SSIL — one-bounce indirect-light boost for the AMBIENT term (SSFX combine_1.ps:
// `hdiffuse *= IL factor`). Multiplies the SAME hemisphere/flat ambient that AO
// darkens — NEVER the sun or dynamic lights. uIL holds the occluder-colour bounce
// (r_ssil_strength baked in by the GTAO pass); 0 where there's no bounce / r_ssil
// off → returns vec3(1) (no-op). `il/(1+il)` = SSFX soft compression → ×[1,2).
vec3 ssilBoost()
{
    vec3 il = textureLod(uIL, gl_FragCoord.xy * L.ao_params.xy, 0.0).rgb;
    return vec3(1.0) + il / (1.0 + il);
}

// Colored AO (R4 / Activision SIGGRAPH'16): mid-range occlusion bends toward the
// albedo colour. Full open (ao=1) and full black (ao=0) are unchanged.
vec3 coloredAO(float ao, vec3 albedo)
{
    vec3 a =  2.0404 * albedo - 0.3324;
    vec3 b = -4.7951 * albedo + 0.6417;
    vec3 c =  2.7552 * albedo + 0.6903;
    return max(vec3(ao), ((ao * a + b) * ao + c) * ao);
}

// Hemisphere sky ambient (R4 hmodel.h): the sky colour in the surface normal
// direction, at a blurred high mip (~ diffuse irradiance), cross-fading the two
// weather cubes. The fill light that lights surfaces the sun never reaches.
vec3 skyAmbient(vec3 N)
{
    float lod = L.sky_params.z;
    float xf = clamp(L.sky_params.x, 0.0, 1.0);   // weather cross-fade - usually 0/1
    vec3 a = textureLod(uSky0, N, lod).rgb;
    return (xf > 0.01) ? mix(a, textureLod(uSky1, N, lod).rgb, xf) : a;  // 2nd cube only in transition
}

// Rain visibility: 1 = open to the sky (gets rained on), 0 = covered (roof,
// tunnel). Bilinear-weighted compare (cascTap) on the straight-down rain map;
// outside the map = open sky (border depth 1.0 -> lit). WIDE blur softens the
// BLOCKY tree-canopy occlusion ("wetness in squares").
float rainVis(vec3 wp)
{
    vec3 n = (L.rain_vp * vec4(wp, 1.0)).xyz;   // ortho -> already NDC
    vec2 uv = n.xy * 0.5 + 0.5;
    uv.y = 1.0 - uv.y;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || n.z <= 0.0 || n.z >= 1.0)
        return 1.0;
    float ref = n.z - 0.0015;
    vec2  px  = 1.0 / vec2(textureSize(uRainMap, 0));
    float s = cascTap(uRainMap, uv, ref) * 2.0
            + cascTap(uRainMap, uv + vec2( 3.0, 0.0) * px, ref)
            + cascTap(uRainMap, uv + vec2(-3.0, 0.0) * px, ref)
            + cascTap(uRainMap, uv + vec2(0.0,  3.0) * px, ref)
            + cascTap(uRainMap, uv + vec2(0.0, -3.0) * px, ref)
            + cascTap(uRainMap, uv + vec2( 2.0,  2.0) * px, ref)
            + cascTap(uRainMap, uv + vec2(-2.0,  2.0) * px, ref)
            + cascTap(uRainMap, uv + vec2( 2.0, -2.0) * px, ref)
            + cascTap(uRainMap, uv + vec2(-2.0, -2.0) * px, ref);
    return s * (1.0 / 10.0);
}

#endif // ENV_COMMON_GLSL
