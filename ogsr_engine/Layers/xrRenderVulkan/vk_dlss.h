// xrRenderVulkan — NVIDIA DLSS 4.5 integration (NGX).
//
// Increment 1: NGX init on the live VK device + capability probe. Logs what this
// GPU/driver actually supports (DLSS Super Resolution + DLSS-G Multi-Frame-Gen) so
// the rest of the integration (feature create → jitter → evaluate → present) can
// gate on real availability. Forward-friendly module (like VK::MotionVec) — does
// NOT revive the parked deferred CRenderTarget DLSS scaffold.
#pragma once
#include "vk_core.h"

namespace VK { namespace Dlss {

bool Init();       // NGX init + capability probe (call after the VK device is up). Idempotent.
void Shutdown();   // release capability params + NGX (call at renderer teardown).

bool Inited();             // NGX initialised OK on this device
bool SuperResAvailable();  // DLSS Super Resolution usable on this GPU/driver
bool FrameGenAvailable();  // DLSS-G / Multi-Frame Generation usable on this GPU/driver

// ----- Super Resolution (increment 2) ---------------------------------------
bool        Enabled();     // r_dlss != 0 && SuperResAvailable() (gate for the eval pass)

// Render<display upscale (increment 2c). Fills the recommended RENDER resolution
// for the current r_dlss_quality preset at this display size — NGX optimal
// settings, cached per (display,quality). quality 0 (DLAA) or query failure →
// returns display dims (native, render==display). CRender::Begin sizes all scene
// targets to this and DLSS upscales back to display.
void        GetRenderExtent(u32 displayW, u32 displayH, u32& outW, u32& outH);
bool        Upscaling();   // true when the current quality preset renders below display (render<display)

// A small input bundle so Evaluate stays decoupled from SceneColor/MotionVec/etc.
struct Img {
    VkImageView view  = VK_NULL_HANDLE;
    VkImage     image = VK_NULL_HANDLE;
    VkFormat    format = VK_FORMAT_UNDEFINED;
};

// Display-res upscaled output target (RGBA16F storage). Created on demand / resize.
void        EnsureOutput(u32 displayW, u32 displayH);
VkImageView OutputView();
VkImage     OutputImage();
VkFormat    OutputFormat();
bool        OutputReady();
bool        FeatureReady();   // the SR feature is built (Evaluate ran this frame → tonemap samples OutputView)

// Per-frame sub-pixel jitter (Halton 2,3). Call NewFrameJitter ONCE per frame in
// CRender::Begin (after Device.mFullTransform is final), then post-multiply the
// COMBINED view-proj by a clip translation T (T._41=jx, T._42=jy from
// GetProjJitterNDC) so clip.x += jx*clip.w — the naive proj._31/_32 trick only
// works on a SEPARATE proj matrix, not our combined mFullTransform. Evaluate is
// told the SAME offset in pixel space (s_jitterPix) to un-jitter the result.
void        NewFrameJitter(u32 renderW, u32 renderH);
void        GetProjJitterNDC(float& outX, float& outY);   // NDC offset for the combined view-proj
void        GetJitterPix(float& outX, float& outY);       // raw render-pixel jitter (SL path reports it too)

// (Re)create the DLSS Super Resolution feature for these dims. Records feature
// init into `cmd` on first build. render==display => DLAA (no upscale). Returns
// true when the feature is ready to Evaluate. `inputsGen` = generation of the
// input render targets (SceneColor) — the feature is REBUILT when it changes:
// NGX caches internal state against the input images, so evaluating a feature
// built before the targets were destroyed/recreated (e.g. the r_dlss off→on
// toggle resizes them while the feature idles) page-faults the GPU.
bool        EnsureFeature(VkCommandBuffer cmd, u32 renderW, u32 renderH, u32 displayW, u32 displayH, u32 inputsGen);
void        ReleaseFeature();

// SL path (r_dlss_sl): the NGX SR feature lives INSIDE sl.dlss — building the raw
// one too made VRAM hold TWO tensor sets, so a mid-game enable on a saturated card
// failed sl.dlss's own create (NGX 0xbad00002 OOM) and Streamline then threw out
// of its vkQueuePresentKHR hook. This variant keeps only the bookkeeping the
// render loop needs (dim cache + jitter phase count) and RELEASES any raw feature.
bool        EnsureFeatureSL(u32 renderW, u32 renderH, u32 displayW, u32 displayH);

// Did this frame's evaluate actually resolve into OutputView? Cleared each
// NewFrameJitter. The tonemap must not sample the DLSS output otherwise — a
// failed SL evaluate leaves it undefined (the "blue screen").
bool        ResolvedThisFrame();
void        SetResolvedThisFrame(bool ok);

// Run DLSS: color(render) + depth + mv(render) + jitter → OutputView (display).
// `reset` = 1 discards temporal history (level load / camera cut). Transitions the
// inputs + output to GENERAL for NGX; leaves output in GENERAL (caller samples it).
void        Evaluate(VkCommandBuffer cmd, const Img& color, const Img& depth, const Img& mv,
                     u32 renderW, u32 renderH, bool reset);

// Real scene exposure for DLSS (r_dlss_exp 1, default): the tonemap publishes its
// CPU-smoothed exposure here each frame; Evaluate feeds it to NGX as a 1x1
// exposure texture instead of the AutoExposure flag. Bounded internal range =
// the fix for black undershoot blotches on high-contrast sub-pixel foliage
// (R4's DLSS never sees them because it upscales the TONEMAPPED LDR image).
void        SetExposureHint(float exposure);

}}  // namespace VK::Dlss
