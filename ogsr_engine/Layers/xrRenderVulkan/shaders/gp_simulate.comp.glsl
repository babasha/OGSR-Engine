#version 450
#extension GL_GOOGLE_include_directive : require

// Phase 2 simulate: one thread per pool slot. Each live particle runs the
// effect's ACTION PROGRAM in list order — a GPU port of the PAPI per-action
// Execute() bodies (Tier A). All Tier-A actions are per-particle and order-
// independent across particles, so this particle-major loop matches PAPI's
// action-major loop exactly.
//
// Dead particles (KillOld) push their slot back onto the free-list (atomic
// stack push); survivors append their index to aliveList for the indirect draw.

#include "gp_common.glsl"

layout(local_size_x = 64) in;

#define STEP_DEFAULT 0.033

void main()
{
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.maxParticles) return;

    GPUParticle p = pool[idx];
    if (p.pos_age.w < 0.0) return;   // dead slot — skip

    float dt    = pc.dt_gravity.x;
    float life  = p.vel_life.w;      // tm_max for time-scaled actions

    vec3  pos   = p.pos_age.xyz;
    float age   = p.pos_age.w;
    vec3  vel   = p.vel_life.xyz;
    vec2  size  = p.size;
    vec4  col   = gp_unpackBGRA(p.colorRGBA);  // rgba 0..1
    bool  dead  = false;

    uint pi = p.defId;                   // per-particle program (registry index)
    uint n  = programs[pi].actionCount;
    for (uint i = 0u; i < n; ++i) {
        ActionRec act = programs[pi].actions[i];
        switch (act.atype) {

        case GP_GRAVITY:        // vel += direction * dt
            vel += act.a.xyz * dt;
            break;

        case GP_MOVE:           // age += dt; pos += vel * dt
            age += dt;
            pos += vel * dt;
            break;

        case GP_DAMPING: {      // vel *= 1 - (1-damping)*dt, if speed in band
            float vSqr = dot(vel, vel);
            if (vSqr >= act.b.x && vSqr <= act.b.y) {
                vec3 scale = vec3(1.0) - (vec3(1.0) - act.a.xyz) * dt;
                vel *= scale;
            }
            break;
        }

        case GP_SPEEDLIMIT: {   // clamp |vel| into [min,max]
            float minS = act.a.x, maxS = act.a.y;
            float sSqr = dot(vel, vel);
            if (sSqr < minS * minS && sSqr > 0.0)      vel *= (minS / sqrt(sSqr));
            else if (sSqr > maxS * maxS)               vel *= (maxS / sqrt(sSqr));
            break;
        }

        case GP_TARGETVELOCITY: // vel += (target - vel) * scale*dt
            vel += (act.a.xyz - vel) * (act.b.x * dt);
            break;

        case GP_TARGETSIZE: {   // size += (target - size) * scale*dt  (per axis)
            vec2 dif = (act.a.xy - size) * (act.b.xy * dt);
            size += dif;
            break;
        }

        case GP_TARGETCOLOR: {  // ease colour/alpha toward target within [from,to]*life
            float tmFrom = act.b.y * life;
            float tmTo   = act.b.z * life;
            if (age >= tmFrom && age <= tmTo) {
                float COEFF    = STEP_DEFAULT / max(dt, 1e-6);
                float scaleFac = act.b.x * STEP_DEFAULT;
                vec4  target   = vec4(act.a.xyz, act.a.w);
                vec4  cn       = mix(col, target, scaleFac);
                col -= (col - cn) / COEFF;
            }
            break;
        }

        case GP_KILLOLD: {      // kill if !((age < limit) ^ kill_less_than)
            bool less = act.a.y != 0.0;
            bool young = age < act.a.x;
            if (!(young != less)) dead = true;   // (young ^ less) == false → kill
            break;
        }

        default:
            break;              // unsupported action → no-op (CPU-fallback later)
        }
    }

    if (dead) {
        p.pos_age.w = -1.0;
        pool[idx]   = p;
        uint fi = atomicAdd(counters[0], 1u);    // push slot back to free-list
        freeList[fi] = idx;
        return;
    }

    p.pos_age   = vec4(pos, age);
    p.vel_life  = vec4(vel, life);
    p.size      = size;
    p.colorRGBA = gp_packBGRA(col);
    pool[idx]   = p;

    uint ai = atomicAdd(counters[1], 1u);        // append to aliveList
    aliveList[ai] = idx;
}
