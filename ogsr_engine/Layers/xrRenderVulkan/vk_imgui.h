// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).
//
// ============================================================================
//  Minimal Dear ImGui Vulkan backend (vk_imgui)
// ============================================================================
//  ImGui core is already vendored + built (3rd_party/Src/imgui, imgui.lib); the
//  repo only ships the DX11 backend (xrRenderDX10/imgui_impl_dx11). This is a
//  small, self-contained Vulkan backend in our own style (dynamic rendering,
//  VMA, the .spv loader) — no upstream imgui_impl_vulkan dependency.
//
//  v1 = DISPLAY-ONLY overlay (no mouse/keyboard routing yet): it renders the
//  global profiler window (vk_profiler getters) on top of the final frame,
//  gated by the `r_profiler 2` console var. Input wiring + the r_* knob panel
//  are a later step.
//
//  Lifecycle is lazy: the first DrawOverlay() builds the context, font atlas and
//  pipeline. Call DrawOverlay() once per frame from CRender::End AFTER the UI
//  pass (the swapchain image is COLOR_ATTACHMENT there); Shutdown() on teardown.
// ============================================================================
#pragma once
#include "stdafx.h"

namespace VK { namespace ImGuiVK {

// Build + record the profiler overlay into the current swapchain image. Safe to
// call every frame; no-ops if r_profiler < 2 or init fails. `dt` = real frame
// seconds (Device.fTimeDeltaReal).
void DrawOverlay(VkCommandBuffer cmd, VkImageView swapchainView, VkExtent2D extent, float dt);

void Shutdown();

}} // namespace VK::ImGuiVK
