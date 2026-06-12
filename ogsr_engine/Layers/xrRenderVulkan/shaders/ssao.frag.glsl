#version 450
// xrRenderVulkan — GTAO (port of R4's gtao.h, "GTAO shader by Doenitz").
//
// Runs at HALF resolution between the depth prepass and the world color pass,
// so the only input is the scene depth (statics + alpha-tested statics).
// R4 reads view-space position+normal from the gbuffer; we reconstruct both
// from depth: position via the frustum-ray basis (camDir + right·tanX·ndc.x +
// top·tanY·ndc.y, zview = _43/(zndc−_33) — same scheme as sunshafts.frag),
// normal via 4-neighbour position differences (pick the neighbour closer in
// depth on each axis so geometry edges don't smear the normal).
//
// The GTAO math itself is space-agnostic (any orthonormal frame): we feed it
// camera-RELATIVE world positions; the original's view-space screen axes map
// to the camera right/top unit vectors.
//
// Output: R8 visibility (1 = open, 0 = fully occluded). Pairs with
// ssao_blur.frag (depth-aware 3×3) before receivers sample it.

layout(location = 0) out float outAO;

layout(set = 0, binding = 0) uniform sampler2D uDepth;   // full-res scene depth (D32, prepass)

layout(push_constant) uniform PC {
    vec4 camDir;     // xyz = camera forward (unit)
    vec4 camRightT;  // xyz = right * tan(fovX/2), w = tan(fovX/2)
    vec4 camTopT;    // xyz = top   * tan(fovY/2), w = tan(fovY/2)
    vec4 zp;         // x = proj _33, y = proj _43, z = world radius (m), w = samples/side
    vec4 res;        // xy = AO target size, zw = 1 / AO target size
    vec4 dbg;        // x = r_ssao_debug mode (2 = write linear depth, 3 = write normal.y)
} pc;

const float PI = 3.14159265;
const int   SLICES = 4;

// Fast acos approx (Drobot) — same one R4 uses.
float fast_acos(float v)
{
    v = clamp(v, -1.0, 1.0);
    float res = -0.156583 * abs(v) + (PI * 0.5);
    res *= sqrt(1.0 - abs(v));
    return (v >= 0.0) ? res : PI - res;
}

// Camera-relative world position + view depth from the scene depth at uv.
vec4 fetchPos(vec2 uv)
{
    float zndc  = texture(uDepth, uv).r;
    float zview = clamp(pc.zp.y / (zndc - pc.zp.x), 0.0, 10000.0);
    vec2  ndc   = vec2(uv.x * 2.0 - 1.0, 1.0 - 2.0 * uv.y);   // D3D ndc (y up)
    vec3  ray   = pc.camDir.xyz + pc.camRightT.xyz * ndc.x + pc.camTopT.xyz * ndc.y;
    return vec4(ray * zview, zview);
}

void main()
{
    vec2 uv = gl_FragCoord.xy * pc.res.zw;

    float zndc = texture(uDepth, uv).r;

    // r_ssao_debug 2: dump what the pass actually READS — linear view depth on
    // a 100 m ramp (black = near, white = far/cleared). Bypasses the early-out
    // so a cleared (all-1.0) depth shows as solid white here.
    if (pc.dbg.x > 1.5 && pc.dbg.x < 2.5) {
        float zv = clamp(pc.zp.y / (zndc - pc.zp.x), 0.0, 10000.0);
        outAO = clamp(zv * 0.01, 0.0, 1.0);
        return;
    }

    if (zndc >= 0.9999) { outAO = 1.0; return; }   // sky / no prepass coverage

    vec4 C    = fetchPos(uv);
    vec3 cPos = C.xyz;
    vec3 viewV = -normalize(cPos);

    // Normal from depth: per axis pick the neighbour with the smaller view-depth
    // step — the other side of a silhouette edge would bend the normal.
    vec4 R = fetchPos(uv + vec2(pc.res.z, 0.0));
    vec4 L = fetchPos(uv - vec2(pc.res.z, 0.0));
    vec4 U = fetchPos(uv + vec2(0.0, pc.res.w));
    vec4 D = fetchPos(uv - vec2(0.0, pc.res.w));
    vec3 ddx = (abs(R.w - C.w) < abs(C.w - L.w)) ? (R.xyz - cPos) : (cPos - L.xyz);
    vec3 ddy = (abs(U.w - C.w) < abs(C.w - D.w)) ? (U.xyz - cPos) : (cPos - D.xyz);
    vec3 N = normalize(cross(ddy, ddx));
    if (dot(N, viewV) < 0.0) N = -N;               // face the camera regardless of winding

    // r_ssao_debug 3: reconstructed world normal "up-ness" — flat ground ≈ 1,
    // walls ≈ 0.5, noise here means the depth-derivative normals are broken.
    if (pc.dbg.x > 2.5) { outAO = N.y * 0.5 + 0.5; return; }

    // Screen axes as world directions — the original's view-space (x,y).
    vec3 rightU = normalize(pc.camRightT.xyz);
    vec3 topU   = normalize(pc.camTopT.xyz);

    const float radius  = pc.zp.z;
    const int   nSample = int(pc.zp.w + 0.5);

    // Pixels per world unit at this depth → sampling radius in texels.
    float proj_scale    = pc.res.y / (2.0 * pc.camTopT.w);
    float screen_radius = (radius * 0.5 * proj_scale) / C.w;

    // Spatial noise. R4 uses a 4-PHASE diagonal pattern for the radial offset
    // ((y-x)&3 → only 4 discrete sample-ring phases): on large smooth
    // occlusions the rings survive the blur as visible BANDS. Continuous IGN
    // (second channel, +5.588238 decorrelation shift) turns the rings into
    // per-pixel noise the 3×3 blur resolves into a clean gradient.
    ivec2 ip = ivec2(gl_FragCoord.xy);
    float noiseOffset    = fract(52.9829189 * fract(dot(vec2(ip) + 5.588238, vec2(0.06711056, 0.00583715))));
    float noiseDirection = fract(52.9829189 * fract(dot(vec2(ip), vec2(0.06711056, 0.00583715))));

    float falloff_mul   = 2.0 / (radius * radius);
    vec2  screen_res_mul = (1.0 / float(nSample)) * pc.res.zw;
    float pi_by_slices  = PI / float(SLICES);

    float visibility = 0.0;

    for (int slice = 0; slice < SLICES; ++slice)
    {
        float phi  = (float(slice) + noiseDirection) * pi_by_slices;
        vec2 omega = vec2(cos(phi), sin(phi));
        vec3 directionV = rightU * omega.x + topU * omega.y;

        vec3 orthoDirectionV = directionV - dot(directionV, viewV) * viewV;
        vec3 axisV       = cross(directionV, viewV);
        vec3 projNormalV = N - axisV * dot(N, axisV);
        float projNormalLength = length(projNormalV);

        float sgnN  = sign(dot(orthoDirectionV, projNormalV));
        float cosN  = clamp(dot(projNormalV, viewV) / projNormalLength, 0.0, 1.0);
        float n     = sgnN * fast_acos(cosN);
        float sinN2 = 2.0 * sin(n);

        for (int side = 0; side < 2; ++side)
        {
            float sideSign = -1.0 + 2.0 * float(side);
            float cHorizonCos = -1.0;

            for (int samp = 0; samp < nSample; ++samp)
            {
                // Min offset: R4 uses 4+sample FULL-res texels; we run at half
                // res, so 2+sample keeps the same world-space footprint — the
                // nearest taps are what catch tight contact occlusion.
                vec2 s = max(vec2(screen_radius * (float(samp) + noiseOffset)),
                             vec2(2.0 + float(samp))) * screen_res_mul;
                vec2 sTexCoord = uv + sideSign * s * vec2(omega.x, -omega.y);
                vec3 sPos = fetchPos(sTexCoord).xyz;
                vec3 sHorizonV = sPos - cPos;
                float falloff = clamp(dot(sHorizonV, sHorizonV) * falloff_mul, 0.0, 1.0);
                float H = dot(normalize(sHorizonV), viewV);
                cHorizonCos = (H > cHorizonCos) ? mix(H, cHorizonCos, falloff) : cHorizonCos;
            }

            float h = n + clamp(sideSign * fast_acos(cHorizonCos) - n, -PI * 0.5, PI * 0.5);
            visibility += projNormalLength * (cosN + h * sinN2 - cos(2.0 * h - n)) * 0.25;
        }
    }

    outAO = clamp(visibility / float(SLICES), 0.0, 1.0);
}
