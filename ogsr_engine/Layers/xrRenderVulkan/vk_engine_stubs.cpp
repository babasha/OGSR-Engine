// xrRenderVulkan - Engine-side symbols that the R4 renderer normally provides
// but the Vulkan launcher must also satisfy because xrEngine and xrGame
// reference them unconditionally.
//
// All globals get safe defaults (mostly zero). All functions are no-ops. When
// the Vulkan code grows real implementations of these subsystems, the
// matching definitions move to their proper home and these stubs disappear.

#include "stdafx.h"

// ---------------------------------------------------------------------------
// luabind global allocator hook.
//
// In R4 this lives in Layers/xrRender/ResourceManager_Scripting.cpp:509 and
// xrEngine reaches it through the dxRenderDeviceRender path. We don't link
// xrRender here, so the allocator stays nullptr and the very first luabind::open
// (called from CScriptEngine::init via ai() lazy-init) crashes inside
// luabind::call_allocator with a null function-pointer execute.
//
// We don't include <luabind/luabind_memory.h> here because that header pulls
// LUABIND_API symbols only available after <luabind/luabind.h>'s preamble.
// Forward-declare the two extern globals with the matching signatures instead.
// ---------------------------------------------------------------------------
namespace luabind {
    typedef void* memory_allocation_function_parameter;
    typedef void* (__cdecl* memory_allocation_function_pointer)
        (memory_allocation_function_parameter, void const*, size_t);
    extern memory_allocation_function_pointer    allocator;
    extern memory_allocation_function_parameter  allocator_parameter;
}

static void* __cdecl vk_luabind_allocator(luabind::memory_allocation_function_parameter,
                                          const void* pointer, size_t const size)
{
    if (!size) {
        void* p = const_cast<void*>(pointer);
        xr_free(p);
        return nullptr;
    }
    if (!pointer)
        return Memory.mem_alloc(size);
    return Memory.mem_realloc(const_cast<void*>(pointer), size);
}

// Static-init binding so the allocator is wired before any code path that
// could reach luabind::open (object_factory().init() → ai() → script_engine).
namespace {
struct LuabindAllocatorInstaller
{
    LuabindAllocatorInstaller()
    {
        luabind::allocator           = &vk_luabind_allocator;
        luabind::allocator_parameter = nullptr;
    }
};
LuabindAllocatorInstaller g_LuabindAllocatorInstaller;
} // namespace


// SSFX shader-tunable globals + ps_lens_flare_sun_blend now defined in
// vk_console_min.cpp as part of the full xrRender_console port. Their
// previous stub definitions here would conflict at link time.

// ---------------------------------------------------------------------------
// xrGame ↔ renderer hand-off flag (R4 toggles in r4_R_render.cpp).
// ---------------------------------------------------------------------------
bool RESET_SECTORS_HACK{};

// ---------------------------------------------------------------------------
// xrGame's Level_network.cpp calls this to flush HW occlusion queries on
// object removal. R4 wires it to RImplementation.HWOCC; we have no occlusion
// path yet, so the call is a no-op.
// ---------------------------------------------------------------------------
void Cleanup_R_occlusion() {}

// ---------------------------------------------------------------------------
// D3D11 debug-layer message dump. Called unconditionally from
// xrCore/xrDebugNew.cpp on crash and from device.cpp::on_idle. Vulkan has its
// own validation pipe (debug_utils messenger), so this stays empty here.
// ---------------------------------------------------------------------------
void LogD3D11DebugMessages() {}

// ---------------------------------------------------------------------------
// ImGui DX11 backend hooks called unconditionally from xr_3da/device.cpp.
// Forward-declared signature only — we don't pull in imgui.h or DX headers.
// These get replaced with imgui_impl_vulkan once the Vulkan backend is wired.
// ---------------------------------------------------------------------------
struct ImDrawData;
void ImGui_ImplDX11_NewFrame() {}
void ImGui_ImplDX11_RenderDrawData(ImDrawData* /*draw_data*/) {}
