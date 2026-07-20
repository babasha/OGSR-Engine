
// World pass - lmap variant. For X-Ray level statics with a baked lightmap
// (tcOffset == 24 sub-layout). TC1 at offset 28 is the per-mesh-region lightmap UV
// (SHORT2, scale 1/32768 -> unit range). The lightmap RGBA (set 0, binding 2)
// carries baked hemi/sun/bounce colour from the level compiler.
//
// NOTE: snow VOLUME is NOT applied here. Vertex displacement on a building (mixed
// normals: roof up, walls sideways) lifts the roof off the walls -> it detaches /
// levitates. Snow volume on statics needs a separate snow-shell layer; roofs get
// snow via the fragment (whiten + normal-smoothing) only.

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec2 inUV_short;     // base UV  (SHORT2 SSCALED)
layout(location = 2) in vec4 inTangent;      // .a = du (sub-pixel U fraction)
layout(location = 3) in vec4 inBinormal;     // .a = dv (sub-pixel V fraction)
layout(location = 4) in vec2 inLmapUV_short; // lightmap UV (SHORT2 SSCALED)
layout(location = 5) in vec4 inNormal;       // D3DCOLOR @ 12 (BGRA in memory)

layout(push_constant) uniform PushConstants {
    mat4  mvp;
    vec2  uvScale;
    float alphaRef;
    float detailScale;
    // dynHemi < -0.5 = DYNAMIC visual (spawned prop: urn/bed/item). Real hemi is
    // -dynHemi-1 (frag decodes); the model matrix rows follow (pushed by Flush into
    // the tess region — dynamics never tessellate). Statics stay on the plain path.
    float dynHemi;
    float dmi0, dmi1, dmi2;   // model row i (X basis, carries uniform scale)
    float dmj0, dmj1, dmj2;   // model row j (Y basis)
    float dmc0, dmc1, dmc2;   // model translation
} pc;

layout(location = 0) out vec2 vUV;
layout(location = 1) out vec2 vDetailUV;
layout(location = 2) out vec2 vLmapUV;
layout(location = 3) out vec3 vWorldPos;   // level statics are world-space (identity model)
layout(location = 4) out vec3 vNormal;     // world-space normal (dynamic lights)

#ifdef CLUSTER_FADE
layout(location = 5) flat out uint vFadeBits;   // cull-packed LOD fades (via firstInstance -> gl_InstanceIndex)
#endif

#ifdef INSTANCED
// Host-driven instanced scene (vk_instance_gpu): the model matrix arrives as four
// INSTANCE-rate vertex attributes (binding 1) instead of push constants, so ONE
// indirect draw covers every instance of a mesh. The X-Ray Fmatrix is row-major,
// so its rows map straight onto i/j/k/c and the transform stays `v * M`.
// pc.mvp is the plain viewProj on this path — the model transform happens here.
layout(location = 6) in vec4 aXfI;   // model row i (X basis)
layout(location = 7) in vec4 aXfJ;   // model row j (Y basis)
layout(location = 8) in vec4 aXfK;   // model row k (Z basis)
layout(location = 9) in vec4 aXfC;   // model translation (row c)
#endif

void main()
{
#ifdef CLUSTER_FADE
    vFadeBits = uint(gl_InstanceIndex);
#endif
    vec3 pos = inPos;
    // D3DCOLOR memory order is BGRA -> real (x,y,z) = .bgr; unpack [0,1] -> [-1,1].
    vec3 nrm = inNormal.bgr * 2.0 - 1.0;
    // Dynamic props reuse this static pipeline but carry a real model matrix —
    // without this transform every world-space consumer (fog distance, cascade sun
    // shadow, dynamic lights, wetness/snow) read MODEL-space coords: props got the
    // fog of the LEVEL ORIGIN (washed-out "glow" that ignores the surroundings) and
    // rotated objects were sun-lit from the wrong side (unrotated normals).
#ifdef INSTANCED
    // Unconditional: every draw on this pipeline is instanced. All FOUR rows are
    // real here, where the push-constant path below only carried i/j and had to
    // rebuild k as i×j — so non-uniform scale and mirroring survive intact.
    pos = pos.x * aXfI.xyz + pos.y * aXfJ.xyz + pos.z * aXfK.xyz + aXfC.xyz;
    nrm = normalize(nrm.x * aXfI.xyz + nrm.y * aXfJ.xyz + nrm.z * aXfK.xyz);
    gl_Position = pc.mvp * vec4(pos, 1.0);   // pc.mvp == viewProj
#else
    gl_Position = pc.mvp * vec4(inPos, 1.0);
    if (pc.dynHemi < -0.5) {
        vec3 mi = vec3(pc.dmi0, pc.dmi1, pc.dmi2);
        vec3 mj = vec3(pc.dmj0, pc.dmj1, pc.dmj2);
        vec3 mk = cross(mi, mj) / max(length(mi), 1e-6);   // det=+1 rotation ⇒ k = i×j (scale folded)
        pos = pos.x * mi + pos.y * mj + pos.z * mk + vec3(pc.dmc0, pc.dmc1, pc.dmc2);
        nrm = normalize(nrm.x * mi + nrm.y * mj + nrm.z * mk);
    }
#endif
    vWorldPos   = pos;
    vNormal     = nrm;

    // Sub-pixel UV: 8 extra bits of fractional du/dv from packed tangent/binormal alphas.
    vec2 uv     = inUV_short + vec2(inTangent.a, inBinormal.a);
    vUV         = uv * pc.uvScale;
    vDetailUV   = vUV * pc.detailScale;

    // Lightmap UV scale is 1/32768 (range +-1) - different from base UV's 1/1024.
    vLmapUV     = inLmapUV_short * (1.0 / 32768.0);
}
