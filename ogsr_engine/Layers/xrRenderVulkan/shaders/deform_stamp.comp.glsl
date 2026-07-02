#version 450
// xrRenderVulkan - SNOW deform texture: stamp / decay / reproject (compute).
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
// SPDX-License-Identifier: MIT
//
// A PERSISTENT, player-centred top-down compression field. SIGNED R16F: + = pressed
// DOWN (foot dent), − = pushed UP (the displaced-snow BERM around it). Each frame:
// reproject the previous field (box follows the camera), decay toward 0, then stamp this
// frame's contacts (feet + landed items). Stamps come from a BUFFER (no push-constant
// count cap, so many NPCs all print). Each stamp is GROUND-GATED against the actual
// terrain height (rain map) so a lifted foot / airborne item doesn't stamp. A
// DIRECTIONAL berm (bigger ahead of movement) models snow plowed forward. Read by
// snow_displace.glsl + the snow mesh.

layout(local_size_x = 8, local_size_y = 8) in;

layout(push_constant) uniform PC {
    mat4 curInvVP;    // cell ndc -> world (current)
    mat4 prevVP;      // world -> previous ndc (reproject)
    mat4 rainVP;      // world -> rain ndc (terrain-height gate)
    vec4 p0;          // x=N, y=decay (dt/life), z=hasPrev, w=stampCount
    vec4 p1;          // xy=movement dir (world XZ), z=berm max (frac), w=ground gate (m)
    vec4 p2;          // x=rain eyeY, y=rain zRange, z/w unused
} pc;

layout(set = 0, binding = 0) uniform sampler2D uPrev;                 // previous press field
layout(set = 0, binding = 1, r16f) writeonly uniform image2D uOut;    // new press field (scratch)
layout(set = 0, binding = 2) uniform Stamps {
    vec4 sPos[64];   // xyz = world pos, w = radius (m)
    vec4 sPar[64];   // x = press strength (1 foot .. ~0.34 item), yz = facing (boot dir; 0,0 = round)
};
layout(set = 0, binding = 3) uniform sampler2D uRain;                 // terrain height (rain ortho depth)

vec2 cellWorldXZ(vec2 uv) {
    vec2 ndc = vec2(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);
    vec4 w   = pc.curInvVP * vec4(ndc, 0.0, 1.0);
    return w.xz / w.w;
}
vec2 toPrevUV(vec2 wxz) {
    vec4 c = pc.prevVP * vec4(wxz.x, 0.0, wxz.y, 1.0);
    vec2 p = c.xy / c.w * 0.5 + 0.5; p.y = 1.0 - p.y;
    return p;
}
// Value noise for per-print / per-edge irregularity (so the trail isn't a perfect repeat).
float cHash(vec2 p) { p = fract(p * vec2(127.1, 311.7)); p += dot(p, p + 34.23); return fract(p.x * p.y); }
float cNoise(vec2 p) {
    vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    float a = cHash(i), b = cHash(i + vec2(1,0)), c = cHash(i + vec2(0,1)), d = cHash(i + vec2(1,1));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// Actual terrain world-Y at a world XZ (rain ortho: eyeY - zNear - d*zRange).
float terrainY(vec2 xz) {
    vec4 c  = pc.rainVP * vec4(xz.x, 0.0, xz.y, 1.0);
    vec2 uv = c.xy * 0.5 + 0.5; uv.y = 1.0 - uv.y;
    float d = textureLod(uRain, clamp(uv, vec2(0.0), vec2(1.0)), 0.0).r;
    return pc.p2.x - 1.0 - d * pc.p2.y;
}

void main() {
    int N = int(pc.p0.x + 0.5);
    ivec2 cell = ivec2(gl_GlobalInvocationID.xy);
    if (cell.x >= N || cell.y >= N) return;
    vec2 uv  = (vec2(cell) + 0.5) / float(N);
    vec2 wxz = cellWorldXZ(uv);

    // Reproject previous press (box translated with the camera), decay toward 0.
    float prev = 0.0;
    if (pc.p0.z > 0.5) {
        vec2 p = toPrevUV(wxz);
        if (all(greaterThanEqual(p, vec2(0.0))) && all(lessThanEqual(p, vec2(1.0))))
            prev = textureLod(uPrev, p, 0.0).r;
    }
    const float dec = pc.p0.y;
    prev = (prev > 0.0) ? max(prev - dec, 0.0) : min(prev + dec, 0.0);

    float dent = max(prev, 0.0);     // + dent (deepest wins)
    float berm = min(prev, 0.0);     // − berm (highest pile wins)
    vec2  md   = pc.p1.xy;
    float bermMax = pc.p1.z;
    float gate    = pc.p1.w;
    bool  moving  = dot(md, md) > 1e-4;
    float rough = clamp(pc.p2.z, 0.0, 1.0);                      // r_snow_rough: trail imperfection
    int sc = int(pc.p0.w + 0.5);
    for (int i = 0; i < sc; ++i) {
        vec4  sp = sPos[i];
        float r  = max(sp.w, 0.05);
        // Per-PRINT variation (noise keyed by the stamp position): the whole print is a
        // bit wider/narrower + deeper/shallower + its berm higher/lower -> no ctrl-c look.
        float wv     = cNoise(sp.xz * 0.8);
        float dv     = cNoise(sp.xz * 0.8 + vec2(19.3, 7.1));
        float rEff   = r * mix(1.0, 0.60 + 0.80 * wv, rough);
        float pressV = mix(1.0, 0.55 + 0.85 * dv, rough);
        vec2  o = wxz - sp.xz;
        float d = length(o);
        if (d > rEff * 2.0) continue;                            // out of this stamp's reach
        if (abs(sp.y - terrainY(sp.xz)) > gate) continue;        // GROUND GATE: skip airborne/lifted
        float press = sPar[i].x * pressV;
        // Per-TEXEL rim wobble (2 octaves -> coarse waviness + fine ragged edge) plus a
        // directional bias so one side is rougher: imperfect walls + the odd "collapse".
        float wob = (cNoise(wxz * 2.6) - 0.5) * 0.7
                  + (cNoise(wxz * 6.5) - 0.5) * 0.35
                  + (cNoise(wxz * 1.3 + vec2(3.7, 9.1)) - 0.5) * 0.5;

        vec2 fdir = sPar[i].yz;
        if (dot(fdir, fdir) > 0.25) {
            // ---- BOOT print (RDR2-style): an oriented rounded-rect SOLE in the foot
            // frame with a TREAD (transverse lug bars), a deeper HEEL block and a
            // shallow ARCH between heel and forefoot. The rounded-rect SDF also drives
            // the displaced-soil berm right against the sole edge.
            vec2  fwd  = normalize(fdir);
            vec2  sideV = vec2(-fwd.y, fwd.x);
            float u = dot(o, fwd), v = dot(o, sideV);
            float halfL = rEff * 0.72, halfW = rEff * 0.30;      // ~30x13 cm boot at r=0.22
            float rc = halfW * 0.7;                              // rounded toe/heel corners
            vec2  q  = abs(vec2(u, v)) - vec2(halfL, halfW) + rc;
            float sd = length(max(q, vec2(0.0))) + min(max(q.x, q.y), 0.0) - rc
                     + wob * rough * 0.035;                      // ragged sole edge
            if (sd < 0.0) {
                float shape = 1.0 - smoothstep(-0.018, 0.0, sd); // sharp-ish wall, soft floor
                // Tread: lug bars across the sole (deep groove between raised soil strips)
                float bars = 0.78 + 0.22 * cos(u * 6.2831853 / 0.075);
                // Heel deeper (weight lands there), arch barely touches.
                float arch = mix(0.35, 1.0, smoothstep(0.06 * halfL, 0.30 * halfL, abs(u + 0.12 * halfL)));
                float heel = 1.0 + 0.15 * smoothstep(-0.35 * halfL, -0.75 * halfL, u);
                dent = max(dent, press * shape * bars * arch * heel);
            } else {
                // Berm: soil squeezed out around the sole; bigger ahead of movement.
                float dirW  = moving ? dot(o / max(d, 1e-4), md) : 0.0;
                float scale = mix(0.45, 1.4, clamp(dirW * 0.5 + 0.5, 0.0, 1.0)) * mix(1.0, 0.55 + 0.9 * wv, rough);
                float reach = rEff * 0.38 * (1.0 + 0.5 * max(dirW, 0.0));
                if (sd < reach) {
                    float hump = sin(clamp(sd / reach, 0.0, 1.0) * 3.14159265);
                    berm = min(berm, -hump * bermMax * scale * press);
                }
            }
        } else {
            // ---- ROUND contact (dropped items / legacy): cosine-bell basin + ring berm.
            float nd = d / rEff + wob * rough;
            nd = max(nd, 0.0);
            if (nd < 1.0) {
                float bell = 0.5 + 0.5 * cos(3.14159265 * nd);   // 1 centre .. 0 rim
                dent = max(dent, press * bell);
            } else {
                float dirW  = moving ? dot(o / max(d, 1e-4), md) : 0.0;
                float scale = mix(0.40, 1.5, clamp(dirW * 0.5 + 0.5, 0.0, 1.0)) * mix(1.0, 0.55 + 0.9 * wv, rough);
                float rOutN = 1.22 + 0.55 * max(dirW, 0.0);      // berm reach (normalised)
                if (nd < rOutN) {
                    float t    = (nd - 1.0) / max(rOutN - 1.0, 1e-4);
                    float hump = sin(t * 3.14159265);
                    berm = min(berm, -hump * bermMax * scale * press);
                }
            }
        }
    }
    float press = (dent > 0.0) ? dent : berm;
    imageStore(uOut, cell, vec4(press, 0.0, 0.0, 0.0));
}
