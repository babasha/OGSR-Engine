#version 450

// GPU particle billboard vertex shader.
// Reads particle data from SSBO via aliveList -> pool, expands to a
// camera-facing quad (6 verts = 2 triangles, TRIANGLE_LIST). Computes the
// atlas frame UV (dfFramed effects) and passes the bindless texture slot to
// the fragment shader.

struct GPUParticle {
    vec4  pos_age;
    vec4  vel_life;
    uint  colorRGBA;
    float rot;
    vec2  size;
    uint  defId;
    uint  flags;
};

// Per-program texture/atlas metadata (must match VK::GPUParticles::TexInfo).
struct TexInfo {
    int   layer;        // bindless texture slot (= program), or -1 untextured
    uint  flags;        // bit0 framed, bit1 animated, bit2 random-frame
    vec2  frameSize;    // UV size of one atlas frame
    uint  frameDimX;    // atlas columns
    uint  frameCount;
    float frameSpeed;
    float _pad;
};

layout(set = 0, binding = 0) buffer PoolBuf     { GPUParticle pool[]; };
layout(set = 0, binding = 2) buffer AliveBuf    { uint aliveList[]; };
layout(set = 0, binding = 7) buffer TexInfoBuf  { TexInfo texInfo[]; };

layout(push_constant) uniform PC {
    mat4  viewProj;
    vec4  camRight;
    vec4  camUp;
    vec4  params;
};

layout(location = 0) out vec4      fragColor;
layout(location = 1) out vec2      fragUV;
layout(location = 2) flat out int  fragLayer;

const vec2 kQuadUV[6] = vec2[](
    vec2(0.0, 1.0), vec2(0.0, 0.0), vec2(1.0, 1.0),
    vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0)
);

const vec2 kQuadCorner[6] = vec2[](
    vec2(-1.0, -1.0), vec2(-1.0, 1.0), vec2( 1.0, -1.0),
    vec2(-1.0,  1.0), vec2( 1.0, 1.0), vec2( 1.0, -1.0)
);

vec4 unpackBGRA8(uint c)
{
    float b = float( c        & 0xFFu) / 255.0;
    float g = float((c >> 8)  & 0xFFu) / 255.0;
    float r = float((c >> 16) & 0xFFu) / 255.0;
    float a = float((c >> 24) & 0xFFu) / 255.0;
    return vec4(r, g, b, a);
}

void main()
{
    uint particleIdx = aliveList[gl_InstanceIndex];
    GPUParticle p = pool[particleIdx];

    if (p.pos_age.w < 0.0) {
        gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
        fragLayer = -1;
        return;
    }

    float cosa = cos(p.rot);
    float sina = sin(p.rot);
    vec2  corner = kQuadCorner[gl_VertexIndex];

    vec3 R = camRight.xyz * (p.size.x * corner.x * cosa - p.size.y * corner.y * sina)
           + camUp.xyz    * (p.size.x * corner.x * sina + p.size.y * corner.y * cosa);

    vec3 worldPos = p.pos_age.xyz + R;
    gl_Position   = viewProj * vec4(worldPos, 1.0);

    // ---- Texture / atlas-frame UV -----------------------------------------
    TexInfo ti = texInfo[p.defId];
    vec2 baseUV = kQuadUV[gl_VertexIndex];
    vec2 uv = baseUV;
    if ((ti.flags & 1u) != 0u && ti.frameCount > 1u) {       // dfFramed: pick a frame
        float ageN = (p.vel_life.w > 0.0) ? clamp(p.pos_age.w / p.vel_life.w, 0.0, 0.9999) : 0.0;
        uint fi = uint(ageN * float(ti.frameCount));
        if (fi >= ti.frameCount) fi = ti.frameCount - 1u;
        uint dim = max(ti.frameDimX, 1u);
        vec2 lt = vec2(float(fi % dim), float(fi / dim)) * ti.frameSize;
        uv = lt + baseUV * ti.frameSize;
    }

    fragUV    = uv;
    fragColor = unpackBGRA8(p.colorRGBA);
    fragLayer = ti.layer;
}
