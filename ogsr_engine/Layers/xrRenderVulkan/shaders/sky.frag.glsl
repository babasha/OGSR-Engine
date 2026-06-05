#version 450

// Two-cubemap blended sky, mirroring R4 dxEnvironmentRender::RenderSky.
//
// R4 renders a half-cube (`hbox_verts` in dxEnvironmentRender.cpp): top is a
// regular unit cube, the lower hemisphere is squashed flat at y_vis=-0.01,
// and the side face's mid-ring uses tc.y=-1 while the top uses tc.y=1. Net
// effect: the cubemap's V axis gets stretched across the visible upper
// hemisphere, the horizon line lands at tc.y ≈ -0.98 (bottom edge of the
// cubemap), and the lower hemisphere is essentially invisible. X-Ray sky
// cubemaps are authored against this mapping — the sky gradient occupies
// almost the full V range, and the horizon line is at the very bottom.
//
// To get the R4 look from a fullscreen-triangle pass, we remap the per-pixel
// world direction to the same tc the half-cube geometry would have produced
// for that view ray, then sample the cubemap with that.

layout(set = 0, binding = 0) uniform samplerCube uSky0;
layout(set = 0, binding = 1) uniform samplerCube uSky1;

layout(push_constant) uniform PushConstants {
    vec4 camRightTan_rot;   // .w = skyRotation
    vec4 camUpTan_weight;   // .w = blendWeight
    vec4 camForward_pad;
    vec4 skyColor_pad;
} pc;

layout(location = 0) in  vec3 vWorldDir;
layout(location = 0) out vec4 outColor;

// Map a normalized view direction to the cubemap sample direction the R4
// half-cube geometry would produce.
vec3 SampleDirHalfCube(vec3 d)
{
    float ax = abs(d.x);
    float az = abs(d.z);
    float m  = max(ax, az);

    // Top cap: d.y dominates. Pass through; textureCube picks +Y face.
    if (d.y > m) {
        return d;
    }

    // Side face — y_vis at the wall is d.y / m, valid in [-0.01, 1].
    // tc.y = lerp(-1, 1, (y_vis + 0.01) / 1.01) = 2*y_vis/1.01 + (0.02/1.01) - 1.
    float yAtSide = d.y / m;
    if (yAtSide >= -0.01) {
        float ty = (yAtSide + 0.01) * (2.0 / 1.01) - 1.0;
        return vec3(d.x / m, ty, d.z / m);
    }

    // Skirt / bottom cap: tc.y ≈ -1. Original geometry samples the bottom
    // edge of the side face (X-Ray sky cubemaps put the horizon line there).
    // We must keep the side face dominant — landing on the -Y face instead
    // collapses to whatever uniform/empty content is on that face and
    // visually freezes the sky. Set tc.y just above -1 so |x|=1 or |z|=1
    // still wins the dominance test.
    return vec3(d.x / m, -0.9999, d.z / m);
}

void main()
{
    vec3 dir = normalize(vWorldDir);

    // R4 rotates the skybox geometry by sky_rotation; sampling at R(-θ)*d
    // lands on the same texel that geometry would expose to the view ray.
    float skyRot = pc.camRightTan_rot.w;
    float c = cos(skyRot);
    float s = sin(skyRot);
    dir = vec3(c * dir.x - s * dir.z,
               dir.y,
               s * dir.x + c * dir.z);

    vec3 sampleDir = SampleDirHalfCube(dir);

    vec4 c0 = texture(uSky0, sampleDir);
    vec4 c1 = texture(uSky1, sampleDir);
    vec3 col = mix(c0.rgb, c1.rgb, clamp(pc.camUpTan_weight.w, 0.0, 1.0));

    // skybox.vs in R4 pre-scales the per-vertex tint by 1.7 ("pre-scale by
    // tonemap"). We do the same here so brightness matches R4 output.
    vec3 tint = pc.skyColor_pad.xyz * 1.7;

    outColor = vec4(col * tint, 1.0);
}
