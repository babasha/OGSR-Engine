// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).
//
// ============================================================================
//  GPU-DRIVEN WORLD FORWARD PASS (vk_world_gpu)
// ============================================================================
//  The forward World pass is CPU-bound: every frame the CPU walks all level
//  visuals, frustum-tests each, Submit()s the survivors, sorts, and issues one
//  draw per mesh with full state tracking. That cost scales with OBJECT COUNT —
//  so detail-rich, interior-heavy mod levels hit a CPU draw-call wall and the
//  fps collapses even though the GPU is idle.
//
//  This moves the static-world submission to the GPU: extract the opaque static
//  meshes once at level load (shared VB/IB pools, world-space → no per-instance
//  transform), grouped by (material, pipeline key, vb, ib); per frame a compute
//  shader frustum-culls them into per-group VkDrawIndexedIndirectCommand lists,
//  and the draw loop issues ONE vkCmdDrawIndexedIndirectCount per group. CPU
//  cost becomes ~O(material groups) instead of O(meshes) → object count no
//  longer drives the CPU, so modders get headroom for dense interiors.
//
//  Mirrors vk_shadow_gpu (same cull/indirect machinery) but the draw is the
//  rich forward path: per-group material set + env set + mvp/tail push, deriving
//  the pipeline exactly like RenderQueue::Flush.
//
//  v1: OPAQUE + alpha-tested STATIC level meshes on the standard lmap/vlit +
//  terrain pipelines. Tessellated + wmark materials and DYNAMIC visuals stay on
//  the CPU RenderQueue (added on top). Phase 2 adds HZB occlusion culling.
// ============================================================================
#pragma once
#include "HW_Vulkan.h"
#include "vk_pass_context.h"   // VK::FrameContext

class vkRender_Visual;
namespace VK { class RenderQueue; }

namespace VK { namespace WorldGPU {

void Build();    // extract static world meshes at level load (idempotent)
bool Built();
void Destroy();

// True if this visual is in the GPU-driven set (drawn by Draw*; the CPU queue
// must exclude it to avoid double-draw). Cheap binary search; safe before Build.
bool InSet(vkRender_Visual* v);

// Count of meshes in the GPU set (diagnostics).
u32  SetSize();

// Per-frame CPU path: Submit ONLY the pre-built non-GPU static leaves (wmark/tess/
// no-diffuse), frustum-culled (same 6-plane test as the GPU cull). Replaces the
// O(all-visuals) walk → the CPU static cost no longer scales with object count.
// Returns the number submitted (visible). The GPU set is drawn by Draw*, not here.
u32  SubmitCpuMeshes(RenderQueue& q, const Fmatrix& viewProj, bool doCull);

// Compute-cull the static set against the camera frustum (planes from viewProj)
// → per-group indirect/count regions. MUST run OUTSIDE a dynamic-rendering scope.
void Cull(VkCommandBuffer cmd, const Fmatrix& viewProj);

// Depth prepass: indirect depth-only draw of the culled statics (solid + AT
// variants), reusing the shared depth pipelines. INSIDE the prepass BeginRendering.
void DrawDepth(VkCommandBuffer cmd, const Fmatrix& viewProj, bool displaceTerrain = false);

// Color pass: indirect forward draw of the culled statics. Binds per group the
// pipeline (derived like Flush) + material set + env set, pushes mvp(=viewProj)
// + per-material tail. INSIDE the color BeginRendering. `envSet` = EnvLight set.
void DrawColor(VkCommandBuffer cmd, const Fmatrix& viewProj, VkDescriptorSet envSet);

}} // namespace VK::WorldGPU
