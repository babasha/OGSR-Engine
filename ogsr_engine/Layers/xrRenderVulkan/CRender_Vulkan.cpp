#include "stdafx.h"
#include "CRender_Vulkan.h"
#include "HW_Vulkan.h"
#include "vk_swapchain.h"
#include "vk_sync.h"
#include "vk_command_buffer.h"
#include "vk_barriers.h"
#include "vk_UIPipeline.h"
#include "vk_stub.h"

CRender RImplementation;

namespace {
// Frame state bridging Begin → Clear → End across IRender_interface methods,
// which were shaped around a D3D11 imm-context. Becomes part of the
// framegraph once we have one.
struct FrameInFlight
{
    VkCommandBuffer cmd          = VK_NULL_HANDLE;
    u32             imageIndex   = 0;
    bool            valid        = false;  // false ⇒ swapchain acquire failed; skip submit
};
FrameInFlight g_FrameInFlight{};

// Pre-frame clear color. UI panels render their own backdrops over this.
constexpr VkClearColorValue kBackgroundClear{ {0.0f, 0.0f, 0.0f, 1.0f} };
} // namespace

CRender::CRender()  = default;
CRender::~CRender() = default;

// ----- Loading / Unloading --------------------------------------------------
void CRender::create()       { VK_STUB_ONCE("CRender"); }
void CRender::destroy()      { VK_STUB_ONCE("CRender"); }
void CRender::reset_begin()  { VK_STUB_ONCE("CRender"); }
void CRender::reset_end()    { VK_STUB_ONCE("CRender"); }
void CRender::level_Load(IReader*)  { VK_STUB_ONCE("CRender"); }
void CRender::level_Unload()        { VK_STUB_ONCE("CRender"); }

// ----- Shader compile -------------------------------------------------------
HRESULT CRender::shader_compile(LPCSTR /*name*/, DWORD const* /*pSrcData*/,
                                       UINT /*SrcDataLen*/, LPCSTR /*pFunctionName*/,
                                       LPCSTR /*pTarget*/, DWORD /*Flags*/,
                                       void*& /*result*/)
{
    VK_STUB_ONCE("CRender");
    return E_NOTIMPL;
}

// ----- Information ----------------------------------------------------------
LPCSTR CRender::getShaderPath() { return "vk\\"; }
IRenderVisual* CRender::getVisual(int) { return nullptr; }
u32 CRender::getVisualCount() { return 0; }

// IRender_Target stub. Engine + xrGame UI code ask for it during init to read
// resolution and silently apply post-process settings before any frame runs.
// Width/height reflect Device dimensions; set_* land when post-FX ports over.
namespace {
struct vkRenderTarget_Stub final : IRender_Target
{
    void set_blur(float)               override {}
    void set_gray(float)               override {}
    void set_duality_h(float)          override {}
    void set_duality_v(float)          override {}
    void set_noise(float)              override {}
    void set_noise_scale(float)        override {}
    void set_noise_fps(float)          override {}
    void set_color_base(u32)           override {}
    void set_color_gray(u32)           override {}
    void set_color_add(const Fvector&) override {}
    u32  get_width(CBackend&)          override { return Device.dwWidth; }
    u32  get_height(CBackend&)         override { return Device.dwHeight; }
    void set_cm_imfluence(float)       override {}
    void set_cm_interpolate(float)     override {}
    void set_cm_textures(const shared_str&, const shared_str&) override {}
};
vkRenderTarget_Stub g_RenderTarget_Stub;
} // namespace

IRender_Target* CRender::getTarget()
{
    VK_STUB_ONCE("CRender");
    return &g_RenderTarget_Stub;
}

// ----- Visuals & wallmarks --------------------------------------------------
void CRender::add_Visual(u32, IRenderable*, IRenderVisual*, Fmatrix&) {}

void CRender::add_StaticWallmark(const wm_shader&, const Fvector&, float,
                                        CDB::TRI*, Fvector*) {}

void CRender::add_StaticWallmark(IWallMarkArray*, const Fvector&, float,
                                        CDB::TRI*, Fvector*) {}

void CRender::add_SkeletonWallmark(Fmatrix*, IKinematics*, IWallMarkArray*,
                                          Fvector&, Fvector&, float) {}

void CRender::clear_static_wallmarks() {}

IRender_ObjectSpecific* CRender::ros_create(IRenderable*) { return nullptr; }
void CRender::ros_destroy(IRender_ObjectSpecific*&) {}

// ----- Lighting -------------------------------------------------------------
IRender_Light* CRender::light_create() { return nullptr; }

// ----- Particles ------------------------------------------------------------
void CRender::ParticleEffectFillName(xr_vector<shared_str>&) {}
void CRender::ParticleGroupFillName(xr_vector<shared_str>&) {}
float CRender::GetParticlesTimeLimit(LPCSTR) { return 0.0f; }

// ----- Models ---------------------------------------------------------------
IRenderVisual* CRender::model_CreateParticles(LPCSTR, BOOL) { return nullptr; }
IRenderVisual* CRender::model_Create(LPCSTR, IReader*)      { return nullptr; }
IRenderVisual* CRender::model_CreateChild(LPCSTR, IReader*) { return nullptr; }
IRenderVisual* CRender::model_Duplicate(IRenderVisual*)     { return nullptr; }
void CRender::model_Delete(IRenderVisual*&, BOOL) {}
void CRender::model_Logging(BOOL) {}
void CRender::models_Prefetch() {}
void CRender::models_Clear(BOOL) {}
void CRender::models_savePrefetch() {}
void CRender::models_begin_prefetch1(bool) {}

// ----- Frame ----------------------------------------------------------------
void CRender::Calculate()        { VK_STUB_ONCE("CRender"); }
void CRender::Render()           { VK_STUB_ONCE("CRender"); }
void CRender::AfterWorldRender() { VK_STUB_ONCE("CRender"); }
void CRender::AfterUIRender()    { VK_STUB_ONCE("CRender"); }
void CRender::Screenshot(ScreenshotMode mode, LPCSTR name) { ScreenshotImpl(mode, name); }
void CRender::ScreenshotImpl(ScreenshotMode, LPCSTR)       {}

// ----- Render mode ----------------------------------------------------------
void CRender::rmNear(CBackend&)   {}
void CRender::rmFar(CBackend&)    {}
void CRender::rmNormal(CBackend&) {}

// ----- Stats ----------------------------------------------------------------
u32 CRender::memory_usage()     { return 0; }
u32 CRender::GetCacheStatPolys() { return 0; }

// ----- Backbuffer -----------------------------------------------------------
// Begin/Clear/End drive a single per-frame command buffer so we can confirm
// Vulkan device + swapchain + sync are alive end-to-end. The framegraph will
// own the frame once it lands.
void CRender::Begin()
{
    VK_STUB_ONCE("CRender");
    if (!Swapchain.ShouldRender() || g_bDeviceLost) {
        g_FrameInFlight.valid = false;
        return;
    }

    const u32 frame = CommandManager.GetCurrentFrame();
    if (!Sync.WaitForFence(frame)) { g_FrameInFlight.valid = false; return; }
    if (!Sync.ResetFence(frame))   { g_FrameInFlight.valid = false; return; }

    FrameSync& sync = Sync.GetCurrentFrame(frame);
    g_FrameInFlight.imageIndex = Swapchain.AcquireNextImage(sync.imageAvailable);
    if (g_FrameInFlight.imageIndex == UINT32_MAX) {
        g_FrameInFlight.valid = false;
        return;
    }

    g_FrameInFlight.cmd   = CommandManager.Begin();
    g_FrameInFlight.valid = (g_FrameInFlight.cmd != VK_NULL_HANDLE);
    if (!g_FrameInFlight.valid) return;

    // Expose this frame's command buffer to vkUIRender / VulkanUI so they
    // hit the immediate path instead of buffering into deferred queue.
    g_VkUI_FrameCmd = g_FrameInFlight.cmd;

    // OGSR calls Begin → Calculate → Render → End — no Clear in between, so
    // we clear at frame start. The combine pass will do this for us once the
    // framegraph owns the swapchain image.
    VkImage image = Swapchain.m_Images[g_FrameInFlight.imageIndex];
    VK::ImageBarrier(g_FrameInFlight.cmd, image,
                     VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    VkImageSubresourceRange range{};
    range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    range.levelCount = 1;
    range.layerCount = 1;
    vkCmdClearColorImage(g_FrameInFlight.cmd, image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         &kBackgroundClear, 1, &range);
}

void CRender::Clear()
{
    VK_STUB_ONCE("CRender");
    // Begin already cleared the swapchain image; engine occasionally calls
    // Clear separately for sub-target reset, no-op there.
}

void CRender::End()
{
    VK_STUB_ONCE("CRender");
    if (!g_FrameInFlight.valid) return;

    // Replay any UI draws queued via the deferred path before we close out the
    // frame; finish the dynamic-rendering UI pass so the image lands in
    // PRESENT_SRC for vkQueuePresent.
    VulkanUI::ReplayDeferredUI();
    VulkanUI::EndUIPass();

    VkImage image = Swapchain.m_Images[g_FrameInFlight.imageIndex];

    // EndUIPass already transitioned to PRESENT_SRC if it ran. Otherwise the
    // image is still in TRANSFER_DST from the cornflower clear — bring it
    // home now.
    if (!Swapchain.m_bRenderedThisFrame) {
        VK::ImageBarrier(g_FrameInFlight.cmd, image,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    }
    Swapchain.m_bRenderedThisFrame = false;

    if (!CommandManager.End(g_FrameInFlight.cmd)) return;

    const u32  frame = CommandManager.GetCurrentFrame();
    FrameSync& sync  = Sync.GetCurrentFrame(frame);
    CommandManager.Submit(g_FrameInFlight.cmd,
                          sync.imageAvailable,
                          sync.renderFinished,
                          sync.inFlightFence);
    Swapchain.Present(sync.renderFinished, g_FrameInFlight.imageIndex);
    CommandManager.NextFrame();
    g_FrameInFlight.valid = false;
    g_VkUI_FrameCmd       = VK_NULL_HANDLE;
}

void CRender::ClearTarget() {}

// ----- Cached transforms ----------------------------------------------------
void CRender::SetCacheXform(Fmatrix&, Fmatrix&)    {}
void CRender::SetCacheXformOld(Fmatrix&, Fmatrix&) {}

// ----- Backend --------------------------------------------------------------
// Engine code (xrGame UI, GameFont, CameraManager) calls get_imm_command_list
// purely to forward the returned ref into IRender_Target::get_width/get_height.
// Our Target stub ignores the parameter, so a typed reinterpret over a static
// buffer is safe — nobody dereferences the returned ref. UB the moment somebody
// does, in which case the AV stack will name the offending caller. Real
// CBackend lands when the framegraph drives the frame.
namespace {
alignas(16) char g_DummyBackendStorage[2048];  // sized to outsize current CBackend layout
} // namespace

CBackend& CRender::get_imm_command_list()
{
    VK_STUB_ONCE("CRender");
    return *reinterpret_cast<CBackend*>(g_DummyBackendStorage);
}

void CRender::OnCameraUpdated(bool) {}

// ----- pureFrame ------------------------------------------------------------
void CRender::OnFrame() {}
