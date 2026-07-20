
// World pass - vert-lit variant. For X-Ray level statics where lighting is baked
// per-vertex into a D3DCOLOR at offset 24 (tcOffset == 28 sub-layout). COLOR.bgr =
// baked RGB lighting (point lights + bounce); COLOR.a = sun mask.
//
// NOTE: snow VOLUME is NOT applied here (see world_lmap.vert) - vertex displacement
// detaches mixed-normal static geometry. Props get snow via the fragment only.

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec2 inUV_short;   // base UV  @ offset 28
layout(location = 2) in vec4 inTangent;    // .a = du
layout(location = 3) in vec4 inBinormal;   // .a = dv
layout(location = 4) in vec4 inColor;      // D3DCOLOR @ offset 24 (BGRA in memory)
layout(location = 5) in vec4 inNormal;     // D3DCOLOR @ offset 12 (BGRA in memory)

layout(push_constant) uniform PushConstants {
    mat4  mvp;
    vec2  uvScale;
    float alphaRef;
    float detailScale;
    // dynHemi < -0.5 = DYNAMIC visual; model matrix rows follow (see world_lmap.vert).
    float dynHemi;
    float dmi0, dmi1, dmi2;   // model row i (X basis, carries uniform scale)
    float dmj0, dmj1, dmj2;   // model row j (Y basis)
    float dmc0, dmc1, dmc2;   // model translation
} pc;

layout(location = 0) out vec2 vUV;
layout(location = 1) out vec2 vDetailUV;
layout(location = 2) out vec3 vBakedColor;   // RGB lighting (BGR->RGB swizzle)
layout(location = 3) out float vSunMask;
layout(location = 4) out vec3 vWorldPos;     // statics are world-space (identity model)
layout(location = 5) out vec3 vNormal;       // world-space normal (dynamic lights)

#ifdef CLUSTER_FADE
layout(location = 6) flat out uint vFadeBits;   // cull-packed LOD fades (via firstInstance -> gl_InstanceIndex)
#endif

#ifdef INSTANCED
// See world_lmap_vert_body.glsl for the full note: instance-rate model matrix on
// binding 1, pc.mvp is the plain viewProj. (Input and output locations are separate
// namespaces, so these do not collide with the CLUSTER_FADE output above.)
layout(location = 6) in vec4 aXfI;   // model row i
layout(location = 7) in vec4 aXfJ;   // model row j
layout(location = 8) in vec4 aXfK;   // model row k
layout(location = 9) in vec4 aXfC;   // model translation
#endif

void main()
{
#ifdef CLUSTER_FADE
    vFadeBits = uint(gl_InstanceIndex);
#endif
    vec3 pos = inPos;
    vec3 nrm = inNormal.bgr * 2.0 - 1.0;   // D3DCOLOR BGRA -> xyz
    // Dynamic props: model -> world for the lighting-space outputs (fog/sun/dyn
    // lights/wetness read vWorldPos/vNormal). See world_lmap.vert for the full note.
#ifdef INSTANCED
    pos = pos.x * aXfI.xyz + pos.y * aXfJ.xyz + pos.z * aXfK.xyz + aXfC.xyz;
    nrm = normalize(nrm.x * aXfI.xyz + nrm.y * aXfJ.xyz + nrm.z * aXfK.xyz);
    gl_Position = pc.mvp * vec4(pos, 1.0);   // pc.mvp == viewProj
#else
    gl_Position = pc.mvp * vec4(inPos, 1.0);
    if (pc.dynHemi < -0.5) {
        vec3 mi = vec3(pc.dmi0, pc.dmi1, pc.dmi2);
        vec3 mj = vec3(pc.dmj0, pc.dmj1, pc.dmj2);
        vec3 mk = cross(mi, mj) / max(length(mi), 1e-6);
        pos = pos.x * mi + pos.y * mj + pos.z * mk + vec3(pc.dmc0, pc.dmc1, pc.dmc2);
        nrm = normalize(nrm.x * mi + nrm.y * mj + nrm.z * mk);
    }
#endif
    vWorldPos   = pos;
    vNormal     = nrm;

    vec2 uv     = inUV_short + vec2(inTangent.a, inBinormal.a);
    vUV         = uv * pc.uvScale;
    vDetailUV   = vUV * pc.detailScale;

    // D3DCOLOR memory order is BGRA; swizzle .bgr to recover real RGB.
    vBakedColor = inColor.bgr;
    vSunMask    = inColor.a;
}
