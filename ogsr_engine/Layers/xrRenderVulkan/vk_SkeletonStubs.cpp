// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
// Licensed under the same terms as X-Ray Engine (see root License.txt)

#include "stdafx.h"
#include "vk_SkeletonCompat.h"

#define FBasicVisualH  // Prevent FBasicVisual.h from being included
#include "../xrRender/SkeletonX.h"
#include "../xrRender/SkeletonCustom.h"
#undef FBasicVisualH

// Implementation of stub symbols required by Skeleton classes

// HW is now provided by ../xrRender/HW.h
// CHW_Stub HW;  // Removed: conflicts with xrRender/HW.h

// Console variables (defined in vk_console.cpp)
extern int ps_r1_SoftwareSkinning;
extern float ps_r__WallmarkTTL;

// Software skinning stubs (not used in Vulkan, but Skeleton code references them)
VertexSoftwareStub* _VertexStream = nullptr;
void* _VS = nullptr;

// NOTE: CSkeletonX::has_visible_bones is now provided by OGSR's own
// xrRender/SkeletonX.cpp (compiled via vk_SkeletonX.cpp). Defining it here too
// caused LNK4006 (duplicate symbol). Removed — the real implementation wins.
