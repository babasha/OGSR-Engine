// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

//---------------------------------------------------------------------------
// Vulkan wrapper for SkeletonCustom.cpp
// Compiles OGSR's own xrRender/SkeletonCustom.cpp against the Vulkan backend
// via vk_SkeletonCompat.h type mappings. CRender_Vulkan.h is required because
// SkeletonCustom.cpp drives RImplementation (model_CreateChild / Models /
// model_Duplicate / append_SkeletonWallmark).
//---------------------------------------------------------------------------
#include "stdafx.h"

// FBasicVisualH MUST be defined before anything that transitively pulls
// FBasicVisual.h (CRender_Vulkan.h does, via the dsgraph chain).
#define FBasicVisualH
#include "vk_FBasicVisual.h"

// Full CRender (Vulkan) — SkeletonCustom.cpp calls RImplementation.* and needs
// the complete type, not just the forward decl from vk_SkeletonCompat.h.
#include "CRender_Vulkan.h"
// vkModelPool must be complete: SkeletonCustom.cpp reads RImplementation.Models->
// userdata_override_ini / bone_override_ini (CRender_Vulkan.h only forward-declares it).
#include "vk_ModelPool.h"

// Patched copy of xrRender/SkeletonCustom.cpp — only the CSkeletonX-child
// handling is adapted for the vkFVisual-based vkSkeletonX_* leaves (see the .inl
// header). Everything else is OGSR's own bone/animation logic verbatim.
#include "vk_SkeletonCustom_shared.inl"

#undef dxRender_Visual
#undef FBasicVisualH

// Factory for MT_SKELETON_RIGID — real rigid skeleton. Declared extern in
// vk_Visual.cpp. CKinematics derives FHierrarhyVisual : vkRender_Visual, so the
// upcast is valid.
vkRender_Visual* vkCreateKinematics() { return xr_new<CKinematics>(); }
