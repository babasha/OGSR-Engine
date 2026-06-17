// xrRenderVulkan — VSM receiver lookup. As of the temporal-resolve phase, the sun
// shadow is no longer sampled from the atlas per receiver: a screen-space compute
// pass (vsm_resolve.comp) does the world->light->page->atlas lookup ONCE per pixel,
// temporally accumulates it, and writes a screen-space mask. Receivers just sample
// that mask by screen UV — one tap, and trivially shared across every receiver shader.
// Bound on set 1 binding 14 (EnvLight); gated by L.shadow_params.w (receiver decides).
#ifndef VSM_SAMPLE_GLSL
#define VSM_SAMPLE_GLSL

// The mask lives on the shared EnvLight descriptor set — bound at set 1 for world/grass
// shaders, set 2 for skinned/tree. Define VSM_SET before #include to pick the index.
#ifndef VSM_SET
#define VSM_SET 1
#endif
layout(set = VSM_SET, binding = 14) uniform sampler2D uVsmMask;   // R = sun lit factor (temporally resolved)

// screenUV = gl_FragCoord.xy * (1/screenDims). Returns lit factor 1 = lit .. 0 = shadowed.
float vsmSunShadow(vec2 screenUV)
{
    return texture(uVsmMask, screenUV).r;
}

#endif
