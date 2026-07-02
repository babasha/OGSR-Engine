// Shared structs/bindings for GPU-driven particles — Phase 1 (proof of life).
// Must match VK::GPUParticles::Particle (64 B, std430).
//
// One hardcoded effect (Source + Gravity + KillOld) lives entirely on the GPU:
//   gp_init  — one-time: mark whole pool dead, fill the free-list, set counters.
//   gp_reset — per-frame: aliveCount = 0.
//   gp_emit  — pop free slots (atomic), spawn hardcoded smoke/dust particles.
//   gp_sim   — integrate; KillOld pushes the slot back to the free-list;
//              survivors append themselves to aliveList.
//   gp_build — write the indirect draw (instanceCount = aliveCount).
//
// Phase 2+ (action-program interpreter, spawn requests, sort) was intentionally
// cut — see gpu_particles_roadmap.md.

struct GPUParticle {
    vec4  pos_age;      // xyz = position, w = age (age < 0 → dead)
    vec4  vel_life;     // xyz = velocity, w = lifetime (seconds)
    uint  colorRGBA;    // B8G8R8A8 (matches gp_particle.vert unpackBGRA8)
    float rot;
    vec2  size;
    uint  defId;        // reserved (Phase 3) — 0 for the Phase-1 test effect
    uint  flags;        // reserved
};

// ----- Phase 2: action-program interpreter --------------------------------
// CPU parses a .pe (or, for now, an authored effect) into a GPU program:
//   * EmitDesc  — the Source action resolved into domains the emit shader
//                 samples (position/velocity/size/color/rotation + scalars).
//   * ActionRec[] — the simulate-time actions (Gravity/Move/Damping/…),
//                 interpreted per-particle in gp_simulate, in list order.
// Layout must match VK::GPUParticles::{GenDomain,ActionRec,EmitDesc,Program}.

// One pDomain resolved for GPU sampling (mirrors PAPI::pDomain essentials).
struct GenDomain {
    uint  dtype;            // PDomainEnum (PDPoint=0 … PDRectangle=10)
    uint  _gp0, _gp1, _gp2;
    vec4  p1;               // point / box-min / sphere-centre / line-start …
    vec4  p2;               // box-max / line-delta / cyl axis vector …
    vec4  du;               // 'u' orthonormal basis (cyl/cone/disc/rect)
    vec4  dv;               // 'v' orthonormal basis
    vec4  radii;            // radius1, radius2, radius1Sqr, radius2Sqr
};

// One simulate-time action. `atype` = PActionEnum; a/b/c hold its params.
struct ActionRec {
    uint  atype;
    uint  _ap0, _ap1, _ap2;
    vec4  a;
    vec4  b;
    vec4  c;
};

// Source action resolved for the emit shader.
struct EmitDesc {
    GenDomain posDom;
    GenDomain velDom;
    GenDomain sizeDom;
    GenDomain colorDom;
    GenDomain rotDom;
    vec4  sc;              // x=alpha, y=age, z=age_sigma, w=lifetime
    vec4  parentVel;       // xyz = inherited emitter velocity
};

// PActionEnum ids we interpret (see psystem.h). Unlisted actions are no-ops.
const uint GP_DAMPING        = 4u;
const uint GP_GRAVITY        = 8u;
const uint GP_KILLOLD        = 10u;
const uint GP_MOVE           = 12u;
const uint GP_SPEEDLIMIT     = 22u;
const uint GP_TARGETCOLOR    = 23u;
const uint GP_TARGETSIZE     = 24u;
const uint GP_TARGETVELOCITY = 27u;

// One full effect program (fixed size so programs can live in an array — Phase 3
// step 1, per-defId registry). Must match VK::GPUParticles::Program (2576 B).
#define GP_MAX_ACTIONS 32
struct Program {
    uint      actionCount;
    float     emitRate;     // particles/sec (CPU also uses this; reserved here)
    uint      progFlags;
    uint      _prog_pad;
    EmitDesc  emit;
    ActionRec actions[GP_MAX_ACTIONS];
};

// One emit request (Phase 3 step 2). CPU appends one per active emitter per
// frame; gp_emit walks them so a single dispatch services many emitters, each
// with its own program + world position. Owns emit gids [firstSlot,firstSlot+count).
struct SpawnRequest {       // 48 B std430
    vec4  pos;              // xyz = world spawn position
    vec4  vel;              // xyz = inherited parent velocity
    uint  program;         // program index (becomes particle.defId)
    uint  count;           // particles this request spawns this frame
    uint  firstSlot;       // base emit-gid this request owns
    uint  seed;            // RNG seed base for this request
};

// SSBOs — same binding layout for every dispatch + the draw.
//   counters[0] = freeCount  (top of the free-list stack)
//   counters[1] = aliveCount (this frame, reset by gp_reset)
layout(set = 0, binding = 0) buffer PoolBuf     { GPUParticle pool[]; };
layout(set = 0, binding = 1) buffer CountersBuf { uint counters[]; };
layout(set = 0, binding = 2) buffer AliveBuf    { uint aliveList[]; };
layout(set = 0, binding = 3) buffer IndirectBuf { uint drawCmd[]; };
layout(set = 0, binding = 4) buffer FreeBuf     { uint freeList[]; };
// Per-defId program registry: each live particle picks programs[p.defId];
// gp_emit spawns from the program named by its spawn request.
layout(set = 0, binding = 5) buffer ProgramBuf  { Program programs[]; };
// Per-frame spawn requests (one per active emitter) — written by the CPU.
layout(set = 0, binding = 6) buffer SpawnReqBuf { SpawnRequest spawnReqs[]; };

// Push constants (compute) — must match VK::GPUParticles::ComputePush.
struct GPPush {
    vec4  spawnPos;       // xyz = emitter pos, w = emitCount (as float-bits uint)
    vec4  dt_gravity;     // x = dt, y = gravity magnitude (m/s^2)
    vec4  camPos;         // xyz = camera position (reserved)
    vec4  camForward;     // xyz = camera forward (reserved)
    uint  maxParticles;
    uint  frameSeed;
    uint  activeProgram;  // reserved (emit now reads program per spawn request)
    uint  numRequests;    // active spawn requests this frame
};
layout(push_constant) uniform PC { GPPush pc; };

uint gp_emitCount() { return floatBitsToUint(pc.spawnPos.w); }

// Simple integer hash RNG.
uint gp_hash(uint x)
{
    x = (x ^ 61u) ^ (x >> 16u);
    x = x + (x << 3u);
    x = x ^ (x >> 4u);
    x = x * 0x27d4eb2du;
    x = x ^ (x >> 15u);
    return x;
}
float gp_frand(uint x) { return float(gp_hash(x)) / 4294967296.0; }

// ----- RNG stream (advances the seed; mirrors PAPI drand48 call order) -----
float gp_rand1(inout uint s)  { float r = gp_frand(s); s += 1u; return r; }
vec3  gp_rand3(inout uint s)  { vec3 r = vec3(gp_frand(s), gp_frand(s + 1u), gp_frand(s + 2u)); s += 3u; return r; }

#define GP_PI    3.14159265359
#define GP_2PI   6.28318530718
#define GP_MAXF  1.0e16

// Port of PAPI::pDomain::Generate — a uniform random point in the domain.
vec3 gp_generate(GenDomain d, inout uint s)
{
    switch (d.dtype) {
    case 0u:  // PDPoint
        return d.p1.xyz;
    case 1u:  // PDLine: p1 + p2*t   (p2 is the start→end delta)
        return d.p1.xyz + d.p2.xyz * gp_rand1(s);
    case 4u: {// PDBox: p1=min, p2=max
        vec3 t = gp_rand3(s);
        return d.p1.xyz + (d.p2.xyz - d.p1.xyz) * t;
    }
    case 5u: {// PDSphere shell [radius2..radius1]
        vec3 p = normalize(gp_rand3(s) - vec3(0.5) + vec3(1e-6));
        float r1 = d.radii.x, r2 = d.radii.y;
        float r = (r1 == r2) ? r1 : (r2 + gp_rand1(s) * (r1 - r2));
        return d.p1.xyz + p * r;
    }
    case 6u:   // PDCylinder
    case 7u: { // PDCone (apex at p1)
        float dist  = gp_rand1(s);
        float theta = gp_rand1(s) * GP_2PI;
        float r     = d.radii.y + gp_rand1(s) * (d.radii.x - d.radii.y);
        float x = r * cos(theta);
        float y = r * sin(theta);
        if (d.dtype == 7u) { x *= dist; y *= dist; }
        return d.p1.xyz + d.p2.xyz * dist + d.du.xyz * x + d.dv.xyz * y;
    }
    case 9u: { // PDDisc
        float theta = gp_rand1(s) * GP_2PI;
        float r     = d.radii.y + gp_rand1(s) * (d.radii.x - d.radii.y);
        return d.p1.xyz + d.du.xyz * (r * cos(theta)) + d.dv.xyz * (r * sin(theta));
    }
    case 10u:  // PDRectangle
        return d.p1.xyz + d.du.xyz * gp_rand1(s) + d.dv.xyz * gp_rand1(s);
    case 2u: { // PDTriangle
        float r1 = gp_rand1(s), r2 = gp_rand1(s);
        if (r1 + r2 < 1.0) return d.p1.xyz + d.du.xyz * r1 + d.dv.xyz * r2;
        return d.p1.xyz + d.du.xyz * (1.0 - r1) + d.dv.xyz * (1.0 - r2);
    }
    case 8u:   // PDBlob — approximate the gaussian with a box of width 2*sigma
        return d.p1.xyz + (gp_rand3(s) - vec3(0.5)) * (d.radii.x * 2.0);
    default:   // PDPlane (3) and anything else → the anchor point
        return d.p1.xyz;
    }
}

// Particle colour is stored packed B8G8R8A8 (matches gp_particle.vert).
vec4 gp_unpackBGRA(uint c)
{
    return vec4(float((c >> 16) & 0xFFu),
                float((c >> 8)  & 0xFFu),
                float( c        & 0xFFu),
                float((c >> 24) & 0xFFu)) / 255.0;
}
uint gp_packBGRA(vec4 col)
{
    col = clamp(col, 0.0, 1.0) * 255.0 + 0.5;
    return (uint(col.b)) | (uint(col.g) << 8) | (uint(col.r) << 16) | (uint(col.a) << 24);
}
