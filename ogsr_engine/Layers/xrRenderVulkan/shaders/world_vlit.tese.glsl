#version 450

// World pass — tessellation evaluation, vert-lit variant. Same PN+HM
// displacement + crack-avoidance scheme as world_lmap.tese (see comments
// there); only the interpolated interface differs.

layout(triangles, fractional_odd_spacing, ccw) in;

layout(set = 0, binding = 3) uniform sampler2D uTexBumpX;   // .a = height

layout(location = 0) in vec2  tUV[];
layout(location = 1) in vec2  tDetailUV[];
layout(location = 2) in vec3  tBakedColor[];
layout(location = 3) in float tSunMask[];
layout(location = 4) in vec3  tWorldPos[];
layout(location = 5) in vec3  tNormal[];
layout(location = 7) in float tBakedHemi[];

layout(location = 8)  patch in vec3 pnB210;
layout(location = 9)  patch in vec3 pnB120;
layout(location = 10) patch in vec3 pnB021;
layout(location = 11) patch in vec3 pnB012;
layout(location = 12) patch in vec3 pnB102;
layout(location = 13) patch in vec3 pnB201;
layout(location = 14) patch in vec3 pnB111;
layout(location = 15) patch in vec3 pnN110;
layout(location = 16) patch in vec3 pnN011;
layout(location = 17) patch in vec3 pnN101;

layout(location = 0) out vec2  vUV;
layout(location = 1) out vec2  vDetailUV;
layout(location = 2) out vec3  vBakedColor;
layout(location = 3) out float vSunMask;
layout(location = 4) out vec3  vWorldPos;
layout(location = 5) out vec3  vNormal;
layout(location = 7) out float vBakedHemi;

layout(push_constant) uniform PushConstants {
    mat4  mvp;
    vec2  uvScale;
    float alphaRef;
    float detailScale;
    float dynHemi;
    float tessMax;     // max subdivision at point-blank range (0 = off)
    float tessNear;    // full-factor distance (m)
    float tessFar;     // factor reaches 1 / displacement fades out (m)
    vec4  eyeHeight;   // xyz = camera world pos, w = displacement amplitude (m)
    float pnScale;     // PN-triangle curvature amount (0 = off, 1 = full R4)
} pc;

void main()
{
    vec3 uvw = gl_TessCoord;
    float edge0 = min(uvw.x, min(uvw.y, uvw.z));   // 0 ⇔ on the patch border

    // R4 redistribution: interior verts within 10% barycentric of an edge
    // snap exactly ONTO the edge (fK solves for the smallest component = 0).
    if (edge0 != 0.0 && ((1.0 / 3.0) - edge0) > 0.01) {
        float fK = (edge0 < 0.1) ? (1.0 / 3.0) / ((1.0 / 3.0) - edge0) : 1.0;
        uvw = mix(vec3(1.0 / 3.0), uvw, fK);
    }

    vUV         = tUV[0]         * uvw.x + tUV[1]         * uvw.y + tUV[2]         * uvw.z;
    vDetailUV   = tDetailUV[0]   * uvw.x + tDetailUV[1]   * uvw.y + tDetailUV[2]   * uvw.z;
    vBakedColor = tBakedColor[0] * uvw.x + tBakedColor[1] * uvw.y + tBakedColor[2] * uvw.z;
    vSunMask    = tSunMask[0]    * uvw.x + tSunMask[1]    * uvw.y + tSunMask[2]    * uvw.z;
    vBakedHemi  = tBakedHemi[0]  * uvw.x + tBakedHemi[1]  * uvw.y + tBakedHemi[2]  * uvw.z;
    vec3 wpLinear = tWorldPos[0] * uvw.x + tWorldPos[1] * uvw.y + tWorldPos[2] * uvw.z;
    vec3 N      = normalize(tNormal[0] * uvw.x + tNormal[1] * uvw.y + tNormal[2] * uvw.z);
    vNormal     = N;   // shading normal stays linear (R4 parity)

    float d    = distance(wpLinear, pc.eyeHeight.xyz);
    float fade = clamp((pc.tessFar - d) / max(pc.tessFar - pc.tessNear, 0.01), 0.0, 1.0);

    vec3 wp    = wpLinear;
    vec3 Ndisp = N;   // displacement direction (PN quadratic normal when PN is on)

    // R4 ComputePatchVertex: cubic Bezier position + quadratic normal.
    if (pc.pnScale > 0.0) {
        float x = uvw.x, y = uvw.y, z = uvw.z;
        vec3 nN0 = normalize(tNormal[0]);
        vec3 nN1 = normalize(tNormal[1]);
        vec3 nN2 = normalize(tNormal[2]);
        vec3 pnPos = tWorldPos[0] * (x * x * x) + tWorldPos[1] * (y * y * y) + tWorldPos[2] * (z * z * z)
                   + pnB210 * (3.0 * x * x * y) + pnB120 * (3.0 * x * y * y)
                   + pnB021 * (3.0 * y * y * z) + pnB012 * (3.0 * y * z * z)
                   + pnB102 * (3.0 * x * z * z) + pnB201 * (3.0 * x * x * z)
                   + pnB111 * (6.0 * x * y * z);
        vec3 pnN = nN0 * (x * x) + nN1 * (y * y) + nN2 * (z * z)
                 + pnN110 * (x * y) + pnN011 * (y * z) + pnN101 * (x * z);
        float k = pc.pnScale * fade;
        wp    = mix(wpLinear, pnPos, k);
        Ndisp = normalize(mix(N, pnN, k));
    }

    if (edge0 != 0.0 && pc.tessMax > 0.0) {
        float h = textureLod(uTexBumpX, vUV, 0.0).a;
        wp += Ndisp * (h * pc.eyeHeight.w * fade);
    }

    if (edge0 == 0.0) wp = wpLinear;   // border verts pinned flat (R4 anti-crack)

    vWorldPos   = wp;
    gl_Position = pc.mvp * vec4(wp, 1.0);
}
