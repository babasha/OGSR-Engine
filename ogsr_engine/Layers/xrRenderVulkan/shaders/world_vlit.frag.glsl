#version 450
// Thin wrapper — the actual shader lives in world_vlit_frag_body.glsl (shared with the
// _fade variant, which #defines CLUSTER_FADE for the LOD crossfade dither).
#extension GL_GOOGLE_include_directive : require
#include "world_vlit_frag_body.glsl"
