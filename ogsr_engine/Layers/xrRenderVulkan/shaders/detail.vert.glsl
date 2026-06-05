#version 450
// xrRenderVulkan — detail (grass) vertex shader.
// Per-vertex (binding 0): pos + uv + height
// Per-instance (binding 1): 3 transform rows + color (sun, trail, objId, hemi)
//
// B-min note: vInteractors[i].w == 0 disables the interactor slot, so the
// loop in ApplyInteraction is a no-op. Trail & character interaction kick
// in when sessions B+/C feed real values.

layout(location = 0) in vec3  aPos;
layout(location = 1) in vec2  aUV;
layout(location = 2) in float aHeight;
layout(location = 3) in vec4  aInstRow0;
layout(location = 4) in vec4  aInstRow1;
layout(location = 5) in vec4  aInstRow2;
layout(location = 6) in vec4  aInstColor;   // (sun, trail, objId, hemi)

layout(push_constant) uniform DetailConstants {
    mat4 mViewProj;
    vec4 vWave;             // (freq_x, freq_z, speed, time)
    vec4 vWind;             // (dir.x, 0, dir.z, amplitude)
    vec4 vConsts;           // (s_x, s_y, sun.y, ambient_floor)
    vec4 vInteractors[4];   // xyz=pos, w=radius (0 = unused)
} pc;

layout(location = 0) out vec2  vUV;
layout(location = 1) out vec4  vColor;
layout(location = 2) out float vHeight;

vec3 ApplyWind(vec3 pos, float h)
{
    float wf = h * h;
    float p1 = pos.x * pc.vWave.x + pos.z * pc.vWave.y + pc.vWave.w * pc.vWave.z;
    float p2 = pos.x * pc.vWave.y * 0.7 + pos.z * pc.vWave.x * 1.3 - pc.vWave.w * pc.vWave.z * 0.5;
    float w  = sin(p1) * 0.6 + sin(p2) * 0.4;
    vec3 dir = normalize(vec3(pc.vWind.x, 0.0, pc.vWind.z));
    return pos + dir * w * pc.vWind.w * wf;
}

vec3 ApplyInteraction(vec3 pos, float h)
{
    if (h < 0.01) return pos;
    for (int i = 0; i < 4; ++i) {
        float r = pc.vInteractors[i].w;
        if (r < 0.01) continue;
        vec2 d = pos.xz - pc.vInteractors[i].xz;
        float dist = length(d);
        if (dist < r && dist > 0.01) {
            float t = 1.0 - dist / r;
            float s = t * t * h * 0.7;
            pos.xz += (d / dist) * s;
            pos.y  -= s * 0.25;
        }
    }
    return pos;
}

void main()
{
    // Reconstruct 3×4 instance transform (row-major rows in a column-major
    // mat4x3: 3 column-vectors + a translation column).
    mat4x3 inst = mat4x3(
        aInstRow0.xyz, aInstRow1.xyz, aInstRow2.xyz,
        vec3(aInstRow0.w, aInstRow1.w, aInstRow2.w));

    vec3 worldPos = inst * vec4(aPos, 1.0);
    worldPos      = ApplyWind(worldPos, aHeight);
    worldPos      = ApplyInteraction(worldPos, aHeight);

    // Trail press-down: gen-shader wrote trail intensity into aInstColor.g.
    // Currently always 0 in B-min (TrailMap dummy), so this is a no-op.
    float trail = aInstColor.g;
    if (trail > 0.01 && aHeight > 0.01) {
        float pd = trail * aHeight * 0.6;
        worldPos.y  -= pd;
        worldPos.xz += normalize(aPos.xz + vec2(0.001)) * trail * aHeight * 0.1;
    }

    gl_Position = pc.mViewProj * vec4(worldPos, 1.0);
    vUV         = aUV;

    float sun  = aInstColor.r;
    float hemi = aInstColor.a;
    float L    = max(hemi + sun, pc.vConsts.w);   // ambient_floor floor (0.2)
    L          = clamp(L, 0.0, 2.0);
    vColor     = vec4(L, L, L, 1.0);
    vHeight    = aHeight;
}
