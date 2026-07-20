// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

#pragma once

#include "../../xr_3da/stdafx.h"

// Render backend identifier (matches RENDER define style of other layers)
#define R_R1   1
#define R_R2   2
#define R_R3   3
#define R_R4   4
#define R_VK   5
#define RENDER R_VK

#define XRRENDER_VULKAN_EXPORTS

// d3d9types.h — pulled in for D3DFVF_*, D3DDECLTYPE_*, etc. constants.
// shared Layers/xrRender/FVF.h and our vk_d3d_compat.h reference them as
// integer values to identify vertex layouts. No D3D9 DLL is loaded.
#include <d3d9types.h>
// dxgiformat.h — DXGI_FORMAT enum used by shared CRT / sh_rt.h header for
// render-target format identification. Tiny header, no DLL dependency.
#include <dxgiformat.h>

// Vulkan API + VMA — pulled into the PCH so vk_core.h (copied from monolith,
// it assumes stdafx provides them) and downstream files just `#include "vk_core.h"`.
#ifndef VK_USE_PLATFORM_WIN32_KHR
#  define VK_USE_PLATFORM_WIN32_KHR  // also set in vulkan_renderer.props
#endif
#include <vulkan/vulkan.h>

#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#define VMA_VULKAN_VERSION           1003000
#include <vk_mem_alloc.h>

// VRAM attribution wrappers (VK::Vram::*) — in the PCH so every converted
// vmaCreate/Destroy(Buffer|Image) call site sees the declarations.
#include "vk_vram_stats.h"
