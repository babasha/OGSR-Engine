#version 450
#extension GL_GOOGLE_include_directive : require
#include "light_ubo.glsl"      // L (eye_pos, snow, deform stamps)
#include "snow_displace.glsl"  // SnowFootprintCarve

// World pass - TERRAIN tessellation evaluation. The control points already carry
// the COARSE snow VOLUME (the VS displaced them by SnowDisplace); here we subdivide
// and add the FINE footprint detail (depression + ridge) the coarse mesh can't show.
// Crack avoidance (R4): border verts stay pinned to the linear surface, and interior
// verts within 10% barycentric of an edge snap onto it, so neighbouring patches meet
// without holes. The depth-prepass terrain pipeline runs this exact module too (same
// SPIR-V, no FS) -> prepass depth matches the color geometry -> no z-fight.

layout(triangles, fractional_odd_spacing, ccw) in;

layout(location = 0) in vec2 tUV[];
layout(location = 1) in vec2 tDetailUV[];
layout(location = 2) in vec2 tLmapUV[];
layout(location = 3) in vec3 tWorldPos[];
layout(location = 4) in vec3 tNormal[];

layout(location = 0) out vec2 vUV;
layout(location = 1) out vec2 vDetailUV;
layout(location = 2) out vec2 vLmapUV;
layout(location = 3) out vec3 vWorldPos;
layout(location = 4) out vec3 vNormal;

layout(push_constant) uniform PushConstants {
    mat4  mvp;
    vec2  uvScale;
    float alphaRef;
    float detailScale;
} pc;

const float SNOW_TESS_FAR = 32.0;

void main() {
    vec3 uvw = gl_TessCoord;
    float edge0 = min(uvw.x, min(uvw.y, uvw.z));
    // R4 redistribution: interior verts within 10% barycentric of an edge snap onto it.
    if (edge0 != 0.0 && ((1.0 / 3.0) - edge0) > 0.01) {
        float fK = (edge0 < 0.1) ? (1.0 / 3.0) / ((1.0 / 3.0) - edge0) : 1.0;
        uvw = mix(vec3(1.0 / 3.0), uvw, fK);
    }

    vUV       = tUV[0]       * uvw.x + tUV[1]       * uvw.y + tUV[2]       * uvw.z;
    vDetailUV = tDetailUV[0] * uvw.x + tDetailUV[1] * uvw.y + tDetailUV[2] * uvw.z;
    vLmapUV   = tLmapUV[0]   * uvw.x + tLmapUV[1]   * uvw.y + tLmapUV[2]   * uvw.z;
    vec3 wp   = tWorldPos[0] * uvw.x + tWorldPos[1] * uvw.y + tWorldPos[2] * uvw.z;
    vec3 N    = normalize(tNormal[0] * uvw.x + tNormal[1] * uvw.y + tNormal[2] * uvw.z);
    vNormal   = N;

    // Footprint carve (fine detail on top of the coarse snow surface). Faded by
    // distance. Carve ALL verts INCLUDING patch borders: the carve is a pure function
    // of world XZ and neighbouring patches generate identical shared-edge verts (same
    // endpoints, same symmetric edge factor, same interpolated normal) -> identical
    // carve there, so it stays crack-free. Pinning borders flat while the interior
    // dipped was what produced the sliver SPIKES where a print straddled a patch edge.
    // MESH mode (deform_tex.x>=2): the dense snow mesh owns the dents -> terrain flat.
    if (L.deform_tex.x < 1.5) {
        float d    = distance(wp, L.eye_pos.xyz);
        float fade = clamp((SNOW_TESS_FAR - d) / max(SNOW_TESS_FAR - 1.0, 0.01), 0.0, 1.0);
        wp += N * (SnowFootprintCarve(wp.xz) * fade);
    }

    vWorldPos   = wp;
    gl_Position = pc.mvp * vec4(wp, 1.0);
}
