// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan — screen-space motion vectors (foundation for DLSS/FSR
// upscaling, frame-generation and the path-tracer denoiser).
//
// Phase 1: a single fullscreen pass reconstructs camera/static motion from the
// prepass depth + the previous frame's view-projection — no per-geometry shader
// changes. Self-moving geometry (skinned/trees/grass) is a later phase that
// overwrites this field where dynamics draw.
//
// Owns one full-res RG16F target (prevUV − curUV, in [0,1] UV space). Modelled
// on VK::SSAOPass (depth reconstruction) + VK::SceneColor (target lifecycle).
#pragma once
#include "vk_core.h"

namespace VK { struct FrameContext; }

namespace VK { namespace MotionVec {

bool        Init();                                   // load shaders, pipelines, sets (after swapchain ready)
void        Destroy();
void        EnsureSize(VkExtent2D extent);            // (re)create the RG16F target on size change
bool        Enabled();                                // r_motion_vectors != 0 && inited && target ready

// Reconstruct camera/static screen-space motion from the prepass depth. Caller
// must have the scene depth in SHADER_READ_ONLY. Leaves the MV target in
// SHADER_READ_ONLY (so the debug overlay / future DLSS can sample it).
void        Execute(VkCommandBuffer cmd, VkExtent2D extent);

// Dynamic overlay (Phase 2: self-moving geometry — skinned NPCs now, trees/grass
// later). Re-draws dynamics into the MV target with depth-test against the
// COMPLETE scene depth, so call it AFTER all opaque passes. Reads ctx.depthView +
// ctx.viewProj. No-op unless Enabled() and the static Execute ran this frame.
void        ExecuteDynamic(VkCommandBuffer cmd, const FrameContext& ctx);

// r_mv_debug: overwrite dstView (the swapchain, COLOR_ATTACHMENT) with a
// false-colour view of the motion target. No-op when r_mv_debug == 0.
void        DrawDebugOverlay(VkCommandBuffer cmd, VkImageView dstView, VkExtent2D extent);

VkImageView GetResultView();   // RG16F motion, SHADER_READ after Execute (null until ready)
VkSampler   GetSampler();
VkFormat    Format();
u32         Generation();      // bumped on every (re)create

}}  // namespace VK::MotionVec
