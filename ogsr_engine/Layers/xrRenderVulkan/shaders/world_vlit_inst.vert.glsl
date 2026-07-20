#version 450
// Thin wrapper — instanced variant of world_vlit.vert. See world_lmap_inst.vert.
#extension GL_GOOGLE_include_directive : require
#define INSTANCED
#include "world_vlit_vert_body.glsl"
