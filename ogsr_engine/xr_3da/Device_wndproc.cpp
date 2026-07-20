#include "stdafx.h"

bool CRenderDevice::on_message(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam, LRESULT& result)
{
    switch (uMsg)
    {
    case WM_SYSKEYDOWN: {
        return true;
    }
    case WM_ACTIVATE: {
        OnWM_Activate(wParam, lParam);
        return (false);
    }
    case WM_SETCURSOR: {
        // Embedded in the SDK: show a normal arrow over the viewport. We must set it
        // AND consume the message — returning false would let DefWindowProc forward
        // WM_SETCURSOR up to the parent (the SDK), whose ImGui handler hides the OS
        // cursor (it draws a software cursor on its DX9 backbuffer, which our Vulkan
        // child occludes → an invisible pointer). The game path is unchanged.
        extern bool g_ed_embedded;
        if (g_ed_embedded)
        {
            if (LOWORD(lParam) == HTCLIENT)
            {
                SetCursor(LoadCursor(nullptr, IDC_ARROW));
                result = TRUE;
                return (true);
            }
            return (false); // non-client (borders) — default handling
        }
        result = 1;
        return (true);
    }
    case WM_SYSCOMMAND: {
        // Prevent moving/sizing and power loss in fullscreen mode
        switch (wParam)
        {
        case SC_MOVE:
        case SC_SIZE:
        case SC_MAXIMIZE:
        case SC_MONITORPOWER: result = 1; return (true);
        }
        return (false);
    }
    case WM_CLOSE: {
        result = 0;
        return (true);
    }
    }

    return (false);
}
//-----------------------------------------------------------------------------
// Name: WndProc()
// Desc: Static msg handler which passes messages to the application class.
//-----------------------------------------------------------------------------
LRESULT CALLBACK WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    LRESULT result;
    if (Device.on_message(hWnd, uMsg, wParam, lParam, result))
        return (result);

    return (DefWindowProc(hWnd, uMsg, wParam, lParam));
}
