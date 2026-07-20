#version 450
// xrRenderVulkan — crown VOXEL-cloud viewmode FS. Flat face-shading from screen-space
// derivatives of the world position (no per-vertex normals), so every cube face reads
// as a distinct lit facet. The BAKED per-voxel color (palette + height grade + AO)
// carries the hue variety; sun/hemi env colors from the shared TreeGfxPush modulate it
// so the cloud tracks time-of-day exposure like the real foliage. Opaque, depth-write.
layout(location = 0) in vec3 vWorld;
layout(location = 1) in vec3 vColor;
layout(location = 0) out vec4 oColor;

layout(push_constant) uniform PC {
    mat4  mViewProj;
    float fade; float density; uint treeIdx; float voxSize;   // fade: >0 dissolve OUT, <0 dissolve IN
    vec4  vSunColor;
    vec4  vHemiColor;
    vec4  wind_params;
    vec4  wsetup_trees;
    vec4  wind_anim;
} pc;

void main()
{
    // Dithered LOD crossfade (screen-door): the fine and coarse levels of a transitioning
    // tree draw with COMPLEMENTARY masks of the same per-pixel noise — fade > 0 keeps a
    // pixel where noise >= fade (dissolving out), fade < 0 keeps noise < |fade| (dissolving
    // in), so exactly one level survives per pixel and coverage never drops mid-fade.
    if (pc.fade != 0.0) {
        float ign = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
        if (pc.fade > 0.0) { if (ign <  pc.fade) discard; }
        else               { if (ign >= -pc.fade) discard; }
    }
    vec3 N = normalize(cross(dFdx(vWorld), dFdy(vWorld)));
    // Approximate high sun; two-sided so derivative-flipped faces still shade.
    vec3 L = normalize(vec3(0.35, 0.9, 0.2));
    float nl = abs(dot(N, L));
    float up = 0.5 + 0.5 * N.y;   // sky term by facing: tops bright, undersides dark
    vec3 c = vColor * (pc.vSunColor.rgb * nl + pc.vHemiColor.rgb * (0.35 + 0.85 * up));
    oColor = vec4(c, 1.0);
}
