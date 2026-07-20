// xrRenderVulkan - world uber-shader variant specialization constants (Inc 1).
//
// The world FS (world_lmap/vlit/terrain) was a monolithic uber-shader whose VGPR
// budget was sized for the worst path (POM march + snow + wet + IBL + debug), so
// even a flat, dry, summer pixel paid the register pressure of features it skipped
// → low occupancy → the GPU couldn't hide texture latency.
//
// These booleans are Vulkan specialization constants: the CPU (vk_pipeline_cache
// CreatePipeline) bakes concrete values per (material, frame) at pipeline-creation
// time, and the driver dead-code-eliminates the false branches INCLUDING their
// registers. Defaults are all-true so an unspecialized pipeline behaves exactly like
// the old monolithic shader (safe fallback).
//
// Bit layout MUST match WorldSpec* in vk_pipeline_cache.h:
//   bit0 POM   bit1 SNOW   bit2 WET   bit3 IBL   bit4 DEBUG
#ifndef WORLD_VARIANTS_GLSL
#define WORLD_VARIANTS_GLSL

layout(constant_id = 0) const bool SPEC_POM   = true;   // material has a real `#` height (mat->tessellated)
layout(constant_id = 1) const bool SPEC_SNOW  = true;   // snow accumulation active this frame (sf_params.w > 0)
layout(constant_id = 2) const bool SPEC_WET   = true;   // rain/wetness active this frame (rain_params.y > 0)
layout(constant_id = 3) const bool SPEC_IBL   = true;   // sky specular IBL active this frame (r_ibl)
layout(constant_id = 4) const bool SPEC_DEBUG = true;   // any r_*_debug view active (0 in normal play)

#endif
