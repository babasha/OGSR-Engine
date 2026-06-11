// xrRenderVulkan - compat wrapper: compiles OGSR's shared ParticleEffectActions.cpp
// (PAPI action execution for particle defs) against the Vulkan backend.
#include "stdafx.h"

#define FBasicVisualH
#include "vk_FBasicVisual.h"
#include "CRender_Vulkan.h"
#include "vk_ModelPool.h"
#include "../../xrParticles/psystem.h"

#include "../xrRender/ParticleEffectActions.cpp"

#undef dxRender_Visual
#undef FBasicVisualH
