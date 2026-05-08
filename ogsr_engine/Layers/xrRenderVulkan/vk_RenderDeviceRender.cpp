#include "stdafx.h"
#include "vk_RenderDeviceRender.h"
#include "HW_Vulkan.h"
#include "vk_swapchain.h"
#include "vk_sync.h"
#include "vk_command_buffer.h"
#include "vk_UIPipeline.h"
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
    if (dwWidth == 0 || dwHeight == 0) {
        dwWidth  = psCurrentVidMode[0];
        dwHeight = psCurrentVidMode[1];
        Msg("[VK] DevRender::Create — engine passed 0×0, using cfg %ux%u", dwWidth, dwHeight);

        if (hWnd) {
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

    m_bInitialized = true;
    Msg("[VK] DevRender::Create OK — swapchain %ux%u, %u images",
        Swapchain.GetWidth(), Swapchain.GetHeight(), Swapchain.m_ImageCount);
}

void vkRenderDeviceRender::Destroy()
{
    Msg("[VK] DevRender::Destroy");
    if (!m_bInitialized) return;

    if (VulkanHW.m_Device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(VulkanHW.m_Device);
    }

    VulkanUI::Destroy();
    Swapchain.Destroy();
    CommandManager.Destroy();
    Sync.Destroy();
    VulkanHW.DestroyDevice();

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
