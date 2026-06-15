// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

#include "stdafx.h"
#include "vk_RenderDeviceRender.h"
#include "HW_Vulkan.h"
#include "vk_swapchain.h"
#include "vk_sync.h"
#include "vk_command_buffer.h"
#include "vk_UIPipeline.h"
#include "vk_pipeline_cache.h"
#include "vk_world_material.h"
#include "vk_env_light.h"      // VK::EnvLight — shared per-frame sun/hemi/ambient UBO (set 1/2)
#include "vk_shadow.h"         // VK::ShadowMap — sun shadow map (created with EnvLight)
#include "vk_pass_sky.h"
#include "vk_scene_color.h"    // VK::SceneColor — HDR scene target
#include "vk_pass_tonemap.h"   // VK::TonemapPass — HDR → swapchain composite
#include "vk_pass_bloom.h"     // VK::BloomPass — bright-pass + blur for the composite
#include "vk_pass_ssao.h"      // VK::SSAOPass — GTAO (depth prepass → EnvLight binding 8)
#include "vk_pass_registry.h"  // VK::PassTimingDestroy() — GPU timing query pool teardown
#include "vk_imgui.h"          // VK::ImGuiVK::Shutdown() — profiler overlay teardown
#include "vk_vrs.h"            // VK::VRS::Destroy() — shading-rate image teardown
#include "vk_pass_sunshafts.h" // VK::SunShafts_Destroy()
#include "vk_pass_skinned.h"   // VK::Skinned_Destroy() — frees the bone SSBO at teardown
#include "vk_pass_particles.h" // VK::ParticlePass_Init/Destroy — billboard particle pass
#include "vk_wallmarks.h"      // VK::Wallmarks::Destroy — static decals teardown
#include "vk_rain.h"           // VK::RainPass_Destroy — weather effects teardown
#include "vk_water_sim.h"      // VK::WaterSim::Destroy — water flow sim teardown
#include "vk_shader.h"   // g_VulkanShaderManager (level-shader table)
#include "vk_shaders.h"  // g_ShaderManager (SPIRV module loader/cache)
#include "vk_ModelPool.h"
#include "CRender_Vulkan.h"
#include "vk_stub.h"

bool g_bDeviceLost = false;

vkRenderDeviceRender::vkRenderDeviceRender()  = default;
vkRenderDeviceRender::~vkRenderDeviceRender() = default;

void vkRenderDeviceRender::Copy(IRenderDeviceRender&)            { VK_STUB_ONCE("DevRender"); }
void vkRenderDeviceRender::OnDeviceDestroy(BOOL)                 { VK_STUB_ONCE("DevRender"); }
void vkRenderDeviceRender::SetupStates()                         { VK_STUB_ONCE("DevRender"); }
void vkRenderDeviceRender::OnDeviceCreate()                      { VK_STUB_ONCE("DevRender"); }

// ----- Hardware lifecycle (the active wiring) -------------------------------

void vkRenderDeviceRender::Create(HWND hWnd, u32& dwWidth, u32& dwHeight,
                                  float& fWidth_2, float& fHeight_2)
{
    Msg("[VK] DevRender::Create — hWnd=%p, requested %ux%u", hWnd, dwWidth, dwHeight);
    m_hWnd = hWnd;

    // Engine doesn't set dwWidth/dwHeight before calling us — R4 fills them
    // from its DXGI swapchain. Mirror that: take the configured video-mode
    // (i.e. the resolution the user sees) regardless of the tiny default
    // window. Resize the window itself too, otherwise UI ends up squished.
    extern u32 psCurrentVidMode[2];

    // Fullscreen handling. The engine's rs_fullscreen toggle is commented out
    // engine-wide; the DX11/R4 path just runs as a BORDERLESS window covering
    // the whole monitor (CHW::UpdateWindowProps — style 0, sized to the desktop
    // resolution). The Vulkan path previously only resized a *bordered* window,
    // so it was stuck "in a little window" and behaved badly with screen
    // recorders (no proper fullscreen presentation surface). Mirror R4:
    // borderless fullscreen by default, a normal bordered window only with
    // -draw_borders (same launch param R4 honors).
    const bool bDrawBorders = Core.Params && strstr(Core.Params, "-draw_borders");

    if (hWnd && !bDrawBorders) {
        // Cover the monitor the window currently lives on (multi-monitor safe).
        HMONITOR mon = MonitorFromWindow(hWnd, MONITOR_DEFAULTTOPRIMARY);
        MONITORINFO mi{ sizeof(MONITORINFO) };
        if (GetMonitorInfo(mon, &mi)) {
            const LONG mw = mi.rcMonitor.right  - mi.rcMonitor.left;
            const LONG mh = mi.rcMonitor.bottom - mi.rcMonitor.top;

            dwWidth  = (u32)mw;
            dwHeight = (u32)mh;
            // Keep the engine's notion of the screen size in sync, exactly like
            // R4's CHW ctor forces psCurrentVidMode to the desktop resolution —
            // UI layout & aspect ratio read from here.
            psCurrentVidMode[0] = dwWidth;
            psCurrentVidMode[1] = dwHeight;

            SetWindowLongPtr(hWnd, GWL_STYLE, WS_VISIBLE | WS_POPUP);
            SetWindowPos(hWnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top,
                         mw, mh, SWP_FRAMECHANGED | SWP_SHOWWINDOW);
            Msg("[VK] DevRender::Create — borderless fullscreen %ux%u", dwWidth, dwHeight);
        }
    }

    if (dwWidth == 0 || dwHeight == 0) {
        dwWidth  = psCurrentVidMode[0];
        dwHeight = psCurrentVidMode[1];
        Msg("[VK] DevRender::Create — engine passed 0×0, using cfg %ux%u", dwWidth, dwHeight);

        if (hWnd && bDrawBorders) {
            RECT rc{ 0, 0, (LONG)dwWidth, (LONG)dwHeight };
            const DWORD style   = (DWORD)GetWindowLongPtr(hWnd, GWL_STYLE);
            const DWORD exStyle = (DWORD)GetWindowLongPtr(hWnd, GWL_EXSTYLE);
            AdjustWindowRectEx(&rc, style, FALSE, exStyle);
            SetWindowPos(hWnd, nullptr, 0, 0,
                         rc.right - rc.left, rc.bottom - rc.top,
                         SWP_NOMOVE | SWP_NOZORDER);
        }
    }

    if (!VulkanHW.CreateDevice(hWnd)) {
        FATAL("VK: VulkanHW.CreateDevice failed — see log for instance/device errors");
    }

    Sync.Create();
    CommandManager.Create();
    Swapchain.Create(dwWidth, dwHeight);

    // Engine reads back the actual framebuffer dimensions in case driver clamped them.
    dwWidth   = Swapchain.GetWidth();
    dwHeight  = Swapchain.GetHeight();
    fWidth_2  = float(dwWidth)  * 0.5f;
    fHeight_2 = float(dwHeight) * 0.5f;

    // UI pipeline + vertex buffer + white-texture descriptor.
    VulkanUI::Create();

    // World forward pipeline cache (needs swapchain format) + model pool.
    // OGSR engine doesn't call CRender::create() through this build path, so
    // we own the CRender-side subsystems here, where Create is actually
    // invoked. WorldMaterialCache must come first — PipelineCache builds
    // pipelines whose layout includes the material descriptor set layout.
    VK::WorldMaterialCache::Init();
    VK::EnvLight::Init();          // before PipelineCache — its set layout joins the world pipeline layout (set 1)
    VK::PipelineCache::Init();
    VK::SkyPass::Init();
    VK::TonemapPass::Init();       // HDR → swapchain composite (after SkyPass — shares the SPIRV loader)
    VK::BloomPass::Init();         // bright-pass + blur feeding the tonemap composite
    VK::SSAOPass::Init();          // GTAO from the depth prepass (EnvLight binding 8)
    VK::ParticlePass_Init();

    // Load the particle-definition library (particles.xr / .pe-.pg) so
    // model_CreateParticles can resolve effect/group names. Mirrors R4's
    // CRender::create() → PSLibrary.OnCreate() (which this build path skips).
    { extern void vkParticles_OnCreate(); vkParticles_OnCreate(); }

    // Shader manager owns the level-shader table that maps shader_id →
    // (shader name, diffuse texture name). Without it the loader leaves all
    // Shaders[i] nullptr and every visual ends up with the default white
    // material. Cheap to spin up — no GPU work, just the hashmap + a default
    // entry.
    if (!g_VulkanShaderManager) {
        g_VulkanShaderManager = xr_new<VK::CVulkanShaderManager>();
        g_VulkanShaderManager->Create();
    }
    if (!RImplementation.Models)
        RImplementation.Models = xr_new<vkModelPool>();

    m_bInitialized = true;
    Msg("[VK] DevRender::Create OK — swapchain %ux%u, %u images",
        Swapchain.GetWidth(), Swapchain.GetHeight(), Swapchain.m_ImageCount);
}

void vkRenderDeviceRender::Destroy()
{
    Msg("[VK] DevRender::Destroy");
    if (!m_bInitialized) return;

    // Teardown is logged step-by-step: log.cpp flushes every line, so if a future
    // change ever hangs/crashes here the last line pinpoints the failing step.
    // Low-noise (one block per exit); kept as a permanent guard.
    if (VulkanHW.m_Device != VK_NULL_HANDLE) {
        Msg("[VK] DevRender::Destroy: vkDeviceWaitIdle ...");
        VkResult wi = vkDeviceWaitIdle(VulkanHW.m_Device);
        Msg("[VK] DevRender::Destroy: vkDeviceWaitIdle done (res=%d)", wi);
    }

    if (RImplementation.Models) { xr_delete(RImplementation.Models); RImplementation.Models = nullptr; }
    Msg("[VK] DevRender::Destroy: Models freed");
    if (g_VulkanShaderManager) {
        g_VulkanShaderManager->Destroy();
        xr_delete(g_VulkanShaderManager);
        g_VulkanShaderManager = nullptr;
    }
    Msg("[VK] DevRender::Destroy: ShaderManager freed");
    { extern void vkParticles_OnDestroy(); vkParticles_OnDestroy(); }
    VK::Wallmarks::Destroy();           Msg("[VK] DevRender::Destroy: Wallmarks done");
    VK::RainPass_Destroy();             Msg("[VK] DevRender::Destroy: RainPass done");
    VK::WaterSim::Destroy();            Msg("[VK] DevRender::Destroy: WaterSim done");
    VK::ParticlePass_Destroy();         Msg("[VK] DevRender::Destroy: ParticlePass done");
    VK::SkyPass::Destroy();             Msg("[VK] DevRender::Destroy: SkyPass done");
    VK::SSAOPass::Destroy();            Msg("[VK] DevRender::Destroy: SSAOPass done");
    VK::BloomPass::Destroy();           Msg("[VK] DevRender::Destroy: BloomPass done");
    VK::TonemapPass::Destroy();         Msg("[VK] DevRender::Destroy: TonemapPass done");
    VK::SceneColor::Destroy();          Msg("[VK] DevRender::Destroy: SceneColor done");
    VK::SunShafts_Destroy();            Msg("[VK] DevRender::Destroy: SunShafts done");
    VK::Skinned_Destroy();              Msg("[VK] DevRender::Destroy: SkinnedPass done");
    VK::EnvLight::Destroy();            Msg("[VK] DevRender::Destroy: EnvLight done");
    VK::ShadowMap::Destroy();          Msg("[VK] DevRender::Destroy: ShadowMap done");
    VK::VRS::Destroy();                 Msg("[VK] DevRender::Destroy: VRS done");
    VK::ImGuiVK::Shutdown();            Msg("[VK] DevRender::Destroy: ImGui overlay done");
    VK::PassTimingDestroy();            Msg("[VK] DevRender::Destroy: PassTiming done");
    VK::PipelineCache::Destroy();       Msg("[VK] DevRender::Destroy: PipelineCache done");
    VK::WorldMaterialCache::Destroy();  Msg("[VK] DevRender::Destroy: WorldMaterial done");
    VulkanUI::Destroy();                Msg("[VK] DevRender::Destroy: VulkanUI done");

    // Free the SPIRV module cache LAST among shader consumers: SkyPass / Skinned /
    // PipelineCache / VulkanUI / DetailManager only null their own handles (the
    // modules are owned here), so their VkShaderModules outlive them until now.
    // Must run before VulkanHW.DestroyDevice() — otherwise vkDestroyDevice trips
    // VUID-vkDestroyDevice-device-05137 (undestroyed child shader modules).
    if (g_ShaderManager) {
        g_ShaderManager->DestroyAll();
        xr_delete(g_ShaderManager);
        g_ShaderManager = nullptr;
    }
    Msg("[VK] DevRender::Destroy: SPIRV shader modules freed");

    Swapchain.Destroy();                Msg("[VK] DevRender::Destroy: Swapchain done");
    CommandManager.Destroy();           Msg("[VK] DevRender::Destroy: CommandManager done");
    Sync.Destroy();                     Msg("[VK] DevRender::Destroy: Sync done");
    VulkanHW.DestroyDevice();           Msg("[VK] DevRender::Destroy: device destroyed");

    m_bInitialized = false;
    m_hWnd         = nullptr;
}

void vkRenderDeviceRender::Reset(HWND hWnd, u32& dwWidth, u32& dwHeight,
                                 float& /*fWidth_2*/, float& /*fHeight_2*/)
{
    Msg("[VK] DevRender::Reset — %ux%u", dwWidth, dwHeight);
    if (VulkanHW.m_Device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(VulkanHW.m_Device);
    }
    Swapchain.Recreate(dwWidth, dwHeight);
    dwWidth  = Swapchain.GetWidth();
    dwHeight = Swapchain.GetHeight();
}

// ----- Resource stubs (filled in when texture/asset wiring lands) -----------

void vkRenderDeviceRender::DeferredLoad(BOOL)                    { VK_STUB_ONCE("DevRender"); }
void vkRenderDeviceRender::ResourcesDeferredUpload()             { VK_STUB_ONCE("DevRender"); }
void vkRenderDeviceRender::ResourcesDumpMemoryUsage()            { VK_STUB_ONCE("DevRender"); }
void vkRenderDeviceRender::ResourcesPrefetchCreateTexture(LPCSTR){ VK_STUB_ONCE("DevRender"); }

void vkRenderDeviceRender::ResourcesGetMemoryUsage(u32& m_base, u32& c_base,
                                                   u32& m_lmaps, u32& c_lmaps)
{
    m_base = c_base = m_lmaps = c_lmaps = 0;
}

IRenderDeviceRender::DeviceState vkRenderDeviceRender::GetDeviceState()
{
    return g_bDeviceLost ? dsLost : dsOK;
}

void vkRenderDeviceRender::OnAssetsChanged()                     { VK_STUB_ONCE("DevRender"); }
IResourceManager* vkRenderDeviceRender::GetResourceManager() const { return nullptr; }
