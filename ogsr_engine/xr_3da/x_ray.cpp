//-----------------------------------------------------------------------------
// File: x_ray.cpp
//
// Programmers:
//	Oles		- Oles Shishkovtsov
//	AlexMX		- Alexander Maksimchuk
//-----------------------------------------------------------------------------
#include "stdafx.h"
#include "igame_level.h"
#include "igame_persistent.h"
#include "xr_input.h"
#include "xr_ioconsole.h"
#include "x_ray.h"
#include "std_classes.h"
#include "LightAnimLibrary.h"
#include "../xrcdb/ispatial.h"
#include "ILoadingScreen.h"
#include "DiscordRPC.hpp"
#include "Render.h"
#include "splash.h"
#include "../../3rd_party/Src/mimalloc/mimalloc/include/mimalloc.h" // -mi_guard (guarded heap sampling)

#define CORE_FEATURE_SET(feature, section) Core.Features.set(xrCore::Feature::feature, READ_IF_EXISTS(pSettings, r_bool, section, #feature, false))

ENGINE_API CApplication* pApp{};
ENGINE_API bool IS_OGSR_GA{};
ENGINE_API CInifile* pGameIni{};
int max_load_stage{};

bool use_reshade{};
extern bool init_reshade();
extern void unregister_reshade();

// startup point
void InitEngine()
{
    Engine.Initialize();
    Device.Initialize();
}

void InitSettings()
{
    string_path fname;
    FS.update_path(fname, "$game_config$", "system.ltx");
    pSettings = xr_new<CInifile>(fname, TRUE);
    CHECK_OR_EXIT(!pSettings->sections().empty(), make_string("Cannot find file %s.\nReinstalling application may fix this problem.", fname));

    FS.update_path(fname, "$game_config$", "game.ltx");
    pGameIni = xr_new<CInifile>(fname, TRUE);
    CHECK_OR_EXIT(!pGameIni->sections().empty(), make_string("Cannot find file %s.\nReinstalling application may fix this problem.", fname));

    IS_OGSR_GA = strstr(READ_IF_EXISTS(pSettings, r_string, "mod_ver", "mod_ver", "nullptr"), "OGSR");

    // load custom shader params declarations

    // Simp: сюда добавлены параметры для актуальных шейдеров, в конфигах их держать не практично (потерялся параметр - и шейдер будет работать не правильно.)
    shader_exports.set_custom_params("shader_param_grayscale", {});

    shader_exports.set_custom_params("pnv_color_old", {});
    shader_exports.set_custom_params("pnv_params_old", {});

    shader_exports.set_custom_params("pnv_color", {});

    shader_exports.set_custom_params("heat_vision_steps", {});
    shader_exports.set_custom_params("heat_vision_blurring", {});
    shader_exports.set_custom_params("heat_fade_distance", {});

    shader_exports.set_custom_params("breath_size", {});
    shader_exports.set_custom_params("breath_idx", {});

    shader_exports.set_custom_params("gasmask_inertia", {});
    shader_exports.set_custom_params("device_inertia", {});

    shader_exports.set_custom_params("mark_number", {});
    shader_exports.set_custom_params("mark_color", {});

    shader_exports.set_custom_params("s3ds_param_1", {});
    shader_exports.set_custom_params("s3ds_param_2", {});
    shader_exports.set_custom_params("s3ds_param_3", {});
    shader_exports.set_custom_params("s3ds_param_4", {});
    //

    if (pSettings->section_exist("shader_params_export"))
    {
        Msg("[shader_params_export] section found!!!");

        int tb_count = pSettings->line_count("shader_params_export");
        for (int tb_idx = 0; tb_idx < tb_count; tb_idx++)
        {
            LPCSTR N, V;
            if (pSettings->r_line("shader_params_export", tb_idx, &N, &V))
            {
                if (strstr(N, "_save"))
                    continue;

                shader_exports.set_custom_params(N, Fvector4{});
            }
        }
    }

    FS.init_gamedata_unused();
}
void InitConsole()
{
    Console = xr_new<CConsole>();
    Console->Initialize();

    strcpy_s(Console->ConfigFile, "user.ltx");
    if (strstr(Core.Params, "-ltx "))
    {
        string64 c_name;
        sscanf(strstr(Core.Params, "-ltx ") + 5, "%[^ ] ", c_name);
        strcpy_s(Console->ConfigFile, c_name);
    }

    CORE_FEATURE_SET(colorize_ammo, "dragdrop");
    CORE_FEATURE_SET(colorize_untradable, "dragdrop");
    CORE_FEATURE_SET(equipped_untradable, "dragdrop");
    CORE_FEATURE_SET(select_mode_1342, "dragdrop");
    CORE_FEATURE_SET(highlight_equipped, "dragdrop");
    CORE_FEATURE_SET(af_radiation_immunity_mod, "features");
    CORE_FEATURE_SET(condition_jump_weight_mod, "features");
    CORE_FEATURE_SET(forcibly_equivalent_slots, "features");
    CORE_FEATURE_SET(slots_extend_menu, "features");
    CORE_FEATURE_SET(dynamic_sun_movement, "features");
    CORE_FEATURE_SET(wpn_bobbing, "features");
    CORE_FEATURE_SET(show_inv_item_condition, "features");
    CORE_FEATURE_SET(remove_alt_keybinding, "features");
    CORE_FEATURE_SET(binoc_firing, "features");
    CORE_FEATURE_SET(cop_style_scope_texture, "features");    
    CORE_FEATURE_SET(stop_anim_playing, "features");
    CORE_FEATURE_SET(corpses_collision, "features");
    CORE_FEATURE_SET(more_hide_weapon, "features");
    CORE_FEATURE_SET(keep_inprogress_tasks_only, "features");
    CORE_FEATURE_SET(show_dialog_numbers, "features");
    CORE_FEATURE_SET(objects_radioactive, "features");
    CORE_FEATURE_SET(af_zero_condition, "features");
    CORE_FEATURE_SET(af_satiety, "features");
    CORE_FEATURE_SET(af_psy_health, "features");
    CORE_FEATURE_SET(outfit_af, "features");
    CORE_FEATURE_SET(gd_master_only, "features");
    CORE_FEATURE_SET(scope_textures_autoresize, "features");
    CORE_FEATURE_SET(ogse_new_slots, "features");
    CORE_FEATURE_SET(ogse_wpn_zoom_system, "features");
    CORE_FEATURE_SET(wpn_cost_include_addons, "features");
    CORE_FEATURE_SET(hard_ammo_reload, "features");
    CORE_FEATURE_SET(engine_ammo_repacker, "features");
    CORE_FEATURE_SET(ruck_flag_preferred, "features");
    CORE_FEATURE_SET(old_outfit_slot_style, "features");
    CORE_FEATURE_SET(npc_simplified_shooting, "features");
    CORE_FEATURE_SET(use_trade_deficit_factor, "features");
    CORE_FEATURE_SET(show_objectives_ondemand, "features");
    CORE_FEATURE_SET(pickup_check_overlaped, "features");
    CORE_FEATURE_SET(actor_thirst, "features");
    CORE_FEATURE_SET(autoreload_wpn, "features");
    CORE_FEATURE_SET(no_progress_bar_animation, "features");
    CORE_FEATURE_SET(disable_dialog_break, "features");
    CORE_FEATURE_SET(busy_actor_restrictions, "features");
}

// Defined in the editor-host facade below (A2.3).
extern bool g_ed_embedded;

// Embedded in a host UI (the SDK): input must be NON-exclusive, otherwise DirectInput
// takes the mouse away from the host's own menus/panels and hides the cursor.
void InitInput() { pInput = xr_new<CInput>(!g_ed_embedded); }
void destroyInput() { xr_delete(pInput); }

void InitSound1() { CSound_manager_interface::_create(0); }

void InitSound2() { CSound_manager_interface::_create(1); }
void destroySound() { CSound_manager_interface::_destroy(); }

void destroySettings()
{
    xr_delete(pSettings);
    xr_delete(pGameIni);
}
void destroyConsole()
{
    //Console->Destroy();
    xr_delete(Console);
}
void destroyEngine()
{
    Device.Destroy();
    Engine.Destroy();
}

void execUserScript()
{
    Console->Execute("unbindall");

    if (FS.exist("$app_data_root$", Console->ConfigFile))
    {
        Console->ExecuteScript(Console->ConfigFile);
    }
    else
    {
        string_path default_full_name;

        FS.update_path(default_full_name, "$game_config$", "rspec_default.ltx");

        Console->ExecuteScript(default_full_name);
    }
}

void Startup()
{
    InitSound1();
    execUserScript();
    InitSound2();

    // ...command line for auto start
    {
        LPCSTR pStartup = strstr(Core.Params, "-start ");
        if (pStartup)
            Console->Execute(pStartup + 1);
    }
    {
        LPCSTR pStartup = strstr(Core.Params, "-load ");
        if (pStartup)
            Console->Execute(pStartup + 1);
    }

    // Initialize APP

    Device.Create();
    LALib.OnCreate();
    pApp = xr_new<CApplication>();
    g_pGamePersistent = (IGame_Persistent*)NEW_INSTANCE(CLSID_GAME_PERSISTANT);
    g_SpatialSpace = xr_new<ISpatial_DB>();
    g_SpatialSpacePhysic = xr_new<ISpatial_DB>();

    Discord.Init();

	// Reshade
#pragma todo("Simp: нужен вообще этот решейд???")
    use_reshade = init_reshade();
    if (use_reshade)
        Msg("--[ReShade]: Loaded compatibility addon");
    else
        Msg("!![ReShade]: ReShade not installed or version too old - didn't load compatibility addon");

    // Main cycle
    Memory.mem_usage();

    Device.Run();

	// Reshade
    if (use_reshade)
        unregister_reshade();

    // Destroy APP
    xr_delete(g_SpatialSpacePhysic);
    xr_delete(g_SpatialSpace);
    xr_delete(g_pGamePersistent);
    xr_delete(pApp);
    Engine.Event.Dump();

    // Destroying
    destroyInput();

    destroySettings();

    LALib.OnDestroy();

    destroyConsole();

    destroySound();

    destroyEngine();
}

constexpr auto dwStickyKeysStructSize = sizeof(STICKYKEYS);
constexpr auto dwFilterKeysStructSize = sizeof(FILTERKEYS);
constexpr auto dwToggleKeysStructSize = sizeof(TOGGLEKEYS);

struct damn_keys_filter
{
    BOOL bScreenSaverState;

    // Sticky & Filter & Toggle keys

    STICKYKEYS StickyKeysStruct;
    FILTERKEYS FilterKeysStruct;
    TOGGLEKEYS ToggleKeysStruct;

    DWORD dwStickyKeysFlags;
    DWORD dwFilterKeysFlags;
    DWORD dwToggleKeysFlags;

    damn_keys_filter()
    {
        // Screen saver stuff

        bScreenSaverState = FALSE;

        // Saveing current state
        SystemParametersInfo(SPI_GETSCREENSAVEACTIVE, 0, (PVOID)&bScreenSaverState, 0);

        if (bScreenSaverState)
            // Disable screensaver
            SystemParametersInfo(SPI_SETSCREENSAVEACTIVE, FALSE, NULL, 0);

        dwStickyKeysFlags = 0;
        dwFilterKeysFlags = 0;
        dwToggleKeysFlags = 0;

        ZeroMemory(&StickyKeysStruct, dwStickyKeysStructSize);
        ZeroMemory(&FilterKeysStruct, dwFilterKeysStructSize);
        ZeroMemory(&ToggleKeysStruct, dwToggleKeysStructSize);

        StickyKeysStruct.cbSize = dwStickyKeysStructSize;
        FilterKeysStruct.cbSize = dwFilterKeysStructSize;
        ToggleKeysStruct.cbSize = dwToggleKeysStructSize;

        // Saving current state
        SystemParametersInfo(SPI_GETSTICKYKEYS, dwStickyKeysStructSize, (PVOID)&StickyKeysStruct, 0);
        SystemParametersInfo(SPI_GETFILTERKEYS, dwFilterKeysStructSize, (PVOID)&FilterKeysStruct, 0);
        SystemParametersInfo(SPI_GETTOGGLEKEYS, dwToggleKeysStructSize, (PVOID)&ToggleKeysStruct, 0);

        if (StickyKeysStruct.dwFlags & SKF_AVAILABLE)
        {
            // Disable StickyKeys feature
            dwStickyKeysFlags = StickyKeysStruct.dwFlags;
            StickyKeysStruct.dwFlags = 0;
            SystemParametersInfo(SPI_SETSTICKYKEYS, dwStickyKeysStructSize, (PVOID)&StickyKeysStruct, 0);
        }

        if (FilterKeysStruct.dwFlags & FKF_AVAILABLE)
        {
            // Disable FilterKeys feature
            dwFilterKeysFlags = FilterKeysStruct.dwFlags;
            FilterKeysStruct.dwFlags = 0;
            SystemParametersInfo(SPI_SETFILTERKEYS, dwFilterKeysStructSize, (PVOID)&FilterKeysStruct, 0);
        }

        if (ToggleKeysStruct.dwFlags & TKF_AVAILABLE)
        {
            // Disable FilterKeys feature
            dwToggleKeysFlags = ToggleKeysStruct.dwFlags;
            ToggleKeysStruct.dwFlags = 0;
            SystemParametersInfo(SPI_SETTOGGLEKEYS, dwToggleKeysStructSize, (PVOID)&ToggleKeysStruct, 0);
        }
    }

    ~damn_keys_filter()
    {
        if (bScreenSaverState)
            // Restoring screen saver
            SystemParametersInfo(SPI_SETSCREENSAVEACTIVE, TRUE, NULL, 0);

        if (dwStickyKeysFlags)
        {
            // Restore StickyKeys feature
            StickyKeysStruct.dwFlags = dwStickyKeysFlags;
            SystemParametersInfo(SPI_SETSTICKYKEYS, dwStickyKeysStructSize, (PVOID)&StickyKeysStruct, 0);
        }

        if (dwFilterKeysFlags)
        {
            // Restore FilterKeys feature
            FilterKeysStruct.dwFlags = dwFilterKeysFlags;
            SystemParametersInfo(SPI_SETFILTERKEYS, dwFilterKeysStructSize, (PVOID)&FilterKeysStruct, 0);
        }

        if (dwToggleKeysFlags)
        {
            // Restore FilterKeys feature
            ToggleKeysStruct.dwFlags = dwToggleKeysFlags;
            SystemParametersInfo(SPI_SETTOGGLEKEYS, dwToggleKeysStructSize, (PVOID)&ToggleKeysStruct, 0);
        }
    }
};

int APIENTRY WinMain_impl(HINSTANCE hInstance, HINSTANCE hPrevInstance, char* lpCmdLine, int nCmdShow)
{
    HANDLE hCheckPresenceMutex = INVALID_HANDLE_VALUE;
    if (!strstr(lpCmdLine, "-multi_instances"))
    { // Check for another instance
        constexpr const char* STALKER_PRESENCE_MUTEX = "STALKER-SoC";
        hCheckPresenceMutex = OpenMutex(READ_CONTROL, FALSE, STALKER_PRESENCE_MUTEX);
        if (hCheckPresenceMutex == nullptr)
        {
            // New mutex
            hCheckPresenceMutex = CreateMutex(nullptr, FALSE, STALKER_PRESENCE_MUTEX);
            if (hCheckPresenceMutex == nullptr)
            {
                // Shit happens
                return 0;
            }
        }
        else
        {
            // Already running
            CloseHandle(hCheckPresenceMutex);
            return 0;
        }
    }

    Debug._initialize();

    // SetThreadAffinityMask		(GetCurrentThread(),1);

    // Title window

    DisableProcessWindowsGhosting();

    ShowSplash(hInstance);

    LPCSTR fsgame_ltx_name = "-fsltx ";
    string_path fsgame = "";
    if (strstr(lpCmdLine, fsgame_ltx_name))
    {
        int sz = xr_strlen(fsgame_ltx_name);
        sscanf(strstr(lpCmdLine, fsgame_ltx_name) + sz, "%[^ ] ", fsgame);
    }

    Core._initialize("xray", NULL, TRUE, fsgame[0] ? fsgame : NULL);

    // Heap-corruption hunt: mimalloc is built with MI_GUARDED (1/4000 allocations
    // get a trailing guard page — an overrun AVs at the corrupting WRITE, not at
    // some later allocation). -mi_guard N tightens sampling to 1/N (default 32).
    if (const char* mg = strstr(lpCmdLine, "-mi_guard"))
    {
        int rate = atoi(mg + sizeof("-mi_guard") - 1);
        if (rate <= 0)
            rate = 32;
        mi_option_set(mi_option_guarded_sample_rate, rate);
        Msg("* [mi_guard] guarded objects: 1/%d allocations get a guard page", rate);
    }

    // Tiled-CDB soak harness (see xrCDB_stress.cpp) — runs headless and exits
    if (strstr(lpCmdLine, "-cdb_stress"))
    {
        extern void run_cdb_tiled_stress(const char* cmd);
        run_cdb_tiled_stress(lpCmdLine);
        Core._destroy();
        return 0;
    }

    InitSettings();

    // Adjust player & computer name for Asian
    if (pSettings->line_exist("string_table", "no_native_input"))
    {
        strcpy_s(Core.UserName, "Player");
        strcpy_s(Core.CompName, "Computer");
    }

    {
        damn_keys_filter filter;
        (void)filter;

        InitEngine();
        InitInput();
        InitConsole();

        Engine.External.Initialize();
        Console->Execute("stat_memory");
        Startup();
        Core._destroy();

        if (!strstr(lpCmdLine, "-multi_instances")) // Delete application presence mutex
            CloseHandle(hCheckPresenceMutex);
    }
    // here damn_keys_filter class instanse will be destroyed

    return 0;
}

int APIENTRY WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, char* lpCmdLine, int nCmdShow)
{
    gModulesLoaded = true;

    WinMain_impl(hInstance, hPrevInstance, lpCmdLine, nCmdShow);

    ExitFromWinMain = true;

    return 0;
}

// ============================================================================
// Editor-host facade (architecture a2) — see ed_facade.h. Boots the engine up to
// the point of rendering but does NOT run Device.Run()'s message loop, so an external
// host (the SDK) drives frames via Ed_RenderFrame(). This is the SAME sequence as
// WinMain_impl + Startup(), minus the mutex/splash niceties and the message loop.
// Additive and DLL-only in practice (built with EdDllBuild=true); the .exe game path
// through WinMain is completely untouched.
// ============================================================================
#include "ed_facade.h"

static bool s_edBooted = false;

// Anchor symbol whose address lives in THIS module — used to resolve the DLL's own
// HMODULE via GetModuleHandleEx(FROM_ADDRESS).
static void ed_module_anchor() {}

// A2.3 — embedded-viewport state. Non-static: the VK renderer reads it to skip its
// window restyle/reposition, and Xr_input to pick a top-level coop window.
// (declared extern above, near InitInput)
bool g_ed_embedded = false;
static HWND s_edChildWnd = nullptr;

extern LRESULT CALLBACK WndProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

// Create the render surface as a child of the host's panel. We make our OWN child
// window rather than rendering straight into the host HWND: the engine needs its
// WndProc on the render window, and the host keeps its panel's proc untouched.
static HWND ed_create_child_window(HWND parent, int w, int h)
{
    static bool s_classReg = false;
    const char* wndclass = "_XRAY_ED_VIEWPORT";
    HINSTANCE hInst = (HINSTANCE)GetModuleHandle(nullptr);
    if (!s_classReg)
    {
        WNDCLASS wc = {CS_OWNDC, WndProc, 0, 0, hInst, nullptr, LoadCursor(nullptr, IDC_ARROW), (HBRUSH)GetStockObject(BLACK_BRUSH), nullptr, wndclass};
        RegisterClass(&wc);
        s_classReg = true;
    }
    return CreateWindowEx(0, wndclass, "", WS_CHILD | WS_VISIBLE | WS_CLIPSIBLINGS | WS_CLIPCHILDREN, 0, 0, w, h, parent, nullptr, hInst, nullptr);
}

int Ed_Init(const char* extra_params) { return Ed_InitEx(nullptr, 0, 0, extra_params); }

int Ed_InitEx(void* parent_hwnd, int width, int height, const char* extra_params)
{
    if (s_edBooted)
        return 1;

    gModulesLoaded = true;
    Debug._initialize();

    // Resolve THIS DLL's own directory (…\bin_x64\) and hand it to Core as the
    // ApplicationPath, so the FS finds fsgame.ltx at fs_root (= its parent, the engine
    // folder) no matter which host process loaded us. The host exe's path is irrelevant.
    string_path dllDir = {0};
    {
        HMODULE hSelf = nullptr;
        GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCSTR>(&ed_module_anchor), &hSelf);
        GetModuleFileNameA(hSelf, dllDir, sizeof(dllDir));
        if (char* sl = strrchr(dllDir, '\\'))
            *(sl + 1) = 0; // strip filename, keep trailing '\\'
    }
    // Build the engine's OWN command line: argv[0] = the engine module, then the
    // facade params. The host's real command line is deliberately NOT inherited — the
    // SDK's flags belong to the SDK, and some (e.g. `-nocache`) are consumed during FS
    // init and would break the engine's archive mounting.
    string_path edParams = {0};
    strconcat(sizeof(edParams), edParams, "\"", dllDir, "xrEngine_VK.dll\" ", extra_params ? extra_params : "");

    Core._initialize("xray", nullptr, TRUE, nullptr, dllDir[0] ? dllDir : nullptr, edParams);
    Msg("[Ed] facade boot — Params=[%s]", Core.Params);

    InitSettings();

    // A2.3: hand Device a substitute HWND BEFORE InitEngine() (which runs
    // Device.Initialize(), and that only creates its own window `if (m_hWnd == nullptr)`).
    if (parent_hwnd && width > 0 && height > 0)
    {
        s_edChildWnd = ed_create_child_window((HWND)parent_hwnd, width, height);
        if (!s_edChildWnd)
        {
            Msg("!![Ed] failed to create child render window in parent %p (err %d)", parent_hwnd, GetLastError());
            return 2;
        }
        g_ed_embedded = true;
        Device.m_hWnd = s_edChildWnd;
        gGameWindow = s_edChildWnd;
        Msg("[Ed] embedded viewport: child hwnd=%p in parent=%p, %dx%d", s_edChildWnd, parent_hwnd, width, height);
    }

    InitEngine();
    InitInput();
    InitConsole();
    Engine.External.Initialize(); // AttachRender + AttachGame (object factory) — MUST precede Device.Create / NEW_INSTANCE

    // ---- Startup() body, minus Device.Run() ----
    InitSound1();
    execUserScript();
    InitSound2();

    Device.Create();
    LALib.OnCreate();
    pApp = xr_new<CApplication>();
    g_pGamePersistent = (IGame_Persistent*)NEW_INSTANCE(CLSID_GAME_PERSISTANT);
    g_SpatialSpace = xr_new<ISpatial_DB>();
    g_SpatialSpacePhysic = xr_new<ISpatial_DB>();

    // A WS_CHILD window never receives WM_ACTIVATE, so OnWM_Activate never runs and
    // b_is_Active stays FALSE — and on_idle() gates ALL rendering on it (device.cpp:315),
    // which is exactly a black viewport. An embedded editor viewport should keep drawing
    // regardless of the host's focus, so latch it active here.
    if (g_ed_embedded)
        Device.b_is_Active = TRUE;

    // ---- Device.Run() pre-loop setup (everything BEFORE message_loop) ----
    Device.dwTimeGlobal = 0;
    Device.seqAppStart.Process(rp_AppStart);
    ::Render->ClearTarget();
    if (!g_ed_embedded)
        Device.ShowMainWindow(); // embedded: the child is already WS_VISIBLE, and
                                 // SetForegroundWindow on the host's window is rude

    s_edBooted = true;
    Msg("[Ed] facade booted OK — external frame driving ready");
    return 0;
}

void Ed_RenderFrame()
{
    if (!s_edBooted)
        return;

    // Pump the engine window's messages (Ed_Init created the window on THIS thread),
    // then render exactly one frame — the body of the engine's own message_loop.
    // When embedded, restrict the pump to OUR child window: the host has its own
    // message loop and must keep receiving (and pre-processing, e.g. through its ImGui
    // Win32 hook) everything addressed to it.
    MSG msg;
    HWND pumpWnd = g_ed_embedded ? s_edChildWnd : nullptr;
    while (PeekMessage(&msg, pumpWnd, 0, 0, PM_REMOVE))
    {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    Device.on_idle();
}

// ---- A2.4: host-driven camera + scene. Implemented in the VK renderer's editor
// viewport (Layers/xrRenderVulkan/CRender_Vulkan.cpp); the monolith links it in.
namespace VKEditor
{
void HostSetCamera(const Fvector& pos, const Fvector& dir, const Fvector& up, float fov_deg, float zn, float zf);
int HostAddModel(const char* name, const Fmatrix& xf);
int HostAddModelFile(const char* ogf_path, const char* logical_name, const Fmatrix& xf);
void SetTextureRoot(const char* dir); // Layers/xrRenderVulkan/vk_world_material.cpp
void HostSetModelXform(int id, const Fmatrix& xf);
void HostClearScene();
bool HostScreenRay(int x, int y, Fvector& out_start, Fvector& out_dir);
void HostSetSelection(const int* ids, int count);
} // namespace VKEditor

void Ed_SetCamera(const float* pos3, const float* dir3, const float* up3, float fov_deg, float znear, float zfar)
{
    if (!s_edBooted || !pos3 || !dir3 || !up3)
        return;

    Fvector p, d, u;
    p.set(pos3[0], pos3[1], pos3[2]);
    d.set(dir3[0], dir3[1], dir3[2]);
    u.set(up3[0], up3[1], up3[2]);
    VKEditor::HostSetCamera(p, d, u, fov_deg, znear, zfar);
}

void Ed_GetCamera(float* pos3, float* dir3, float* up3, float* fov_deg)
{
    if (!s_edBooted)
        return;

    // Straight off the Device: whoever set the camera this frame (host or the built-in
    // orbit camera), this is the basis the frame was actually rendered with.
    if (pos3)
    {
        pos3[0] = Device.vCameraPosition.x;
        pos3[1] = Device.vCameraPosition.y;
        pos3[2] = Device.vCameraPosition.z;
    }
    if (dir3)
    {
        dir3[0] = Device.vCameraDirection.x;
        dir3[1] = Device.vCameraDirection.y;
        dir3[2] = Device.vCameraDirection.z;
    }
    if (up3)
    {
        up3[0] = Device.vCameraTop.x;
        up3[1] = Device.vCameraTop.y;
        up3[2] = Device.vCameraTop.z;
    }
    if (fov_deg)
        *fov_deg = Device.fFOV;
}

int Ed_AddModel(const char* visual_name, const float* xform16)
{
    if (!s_edBooted)
        return -1;

    Fmatrix xf;
    if (xform16)
        CopyMemory(&xf, xform16, sizeof(float) * 16);
    else
        xf.identity();

    return VKEditor::HostAddModel(visual_name, xf);
}

void Ed_SetTextureRoot(const char* dir)
{
    if (s_edBooted)
        VKEditor::SetTextureRoot(dir);
}

int Ed_AddModelFile(const char* ogf_path, const char* logical_name, const float* xform16)
{
    if (!s_edBooted)
        return -1;

    Fmatrix xf;
    if (xform16)
        CopyMemory(&xf, xform16, sizeof(float) * 16);
    else
        xf.identity();

    return VKEditor::HostAddModelFile(ogf_path, logical_name, xf);
}

void Ed_SetModelXform(int id, const float* xform16)
{
    if (!s_edBooted || !xform16)
        return;

    Fmatrix xf;
    CopyMemory(&xf, xform16, sizeof(float) * 16);
    VKEditor::HostSetModelXform(id, xf);
}

void Ed_ClearScene()
{
    if (s_edBooted)
        VKEditor::HostClearScene();
}

int Ed_ScreenRay(int x, int y, float* out_start3, float* out_dir3)
{
    if (!s_edBooted || !out_start3 || !out_dir3)
        return 0;

    Fvector s, d;
    if (!VKEditor::HostScreenRay(x, y, s, d))
        return 0;

    out_start3[0] = s.x; out_start3[1] = s.y; out_start3[2] = s.z;
    out_dir3[0] = d.x; out_dir3[1] = d.y; out_dir3[2] = d.z;
    return 1;
}

void Ed_SetSelection(const int* ids, int count)
{
    if (s_edBooted)
        VKEditor::HostSetSelection(ids, count);
}

// Defined in the VK renderer's ImGui overlay (Layers/xrRenderVulkan/vk_imgui.cpp).
namespace VK { namespace ImGuiVK {
void PushEditorLog(const char* text, bool isError);
void PushEditorStats(float fps, float rfps, int verts, int tris, int dips, int lights, int totalLights);
void SetGizmo(int op, int mode, const float* snap, const Fmatrix& xform);
int GizmoResult(Fmatrix& out_xform, Fmatrix& out_delta);
bool GizmoIsUsing();
bool GizmoWantsMouse();
} }

void Ed_SetGizmo(int op, int mode, const float* snap, const float* xform16)
{
    if (!s_edBooted || !xform16)
        return;

    Fmatrix xf;
    CopyMemory(&xf, xform16, sizeof(float) * 16);
    VK::ImGuiVK::SetGizmo(op, mode, snap, xf);
}

int Ed_GizmoResult(float* out_xform16, float* out_delta16)
{
    if (!s_edBooted || !out_xform16 || !out_delta16)
        return 0;

    Fmatrix xf, d;
    const int changed = VK::ImGuiVK::GizmoResult(xf, d);
    CopyMemory(out_xform16, &xf, sizeof(float) * 16);
    CopyMemory(out_delta16, &d, sizeof(float) * 16);
    return changed;
}

int Ed_GizmoIsUsing() { return (s_edBooted && VK::ImGuiVK::GizmoIsUsing()) ? 1 : 0; }
int Ed_GizmoWantsMouse() { return (s_edBooted && VK::ImGuiVK::GizmoWantsMouse()) ? 1 : 0; }

void Ed_PushLog(const char* text, int isError)
{
    if (s_edBooted)
        VK::ImGuiVK::PushEditorLog(text, isError != 0);
}

void Ed_SetStats(float fps, float rfps, int verts, int tris, int dips, int lights, int total_lights)
{
    if (s_edBooted)
        VK::ImGuiVK::PushEditorStats(fps, rfps, verts, tris, dips, lights, total_lights);
}

void Ed_Resize(int width, int height)
{
    if (!s_edBooted || !g_ed_embedded || width <= 0 || height <= 0)
        return;

    SetWindowPos(s_edChildWnd, nullptr, 0, 0, width, height, SWP_NOZORDER | SWP_NOACTIVATE | SWP_NOMOVE);

    // Device::Reset feeds dwWidth/dwHeight straight into Swapchain.Recreate — the
    // renderer does NOT re-read the window here (only Create does), so publish the new
    // size ourselves, exactly like the resolution-change path does.
    RECT cr{};
    GetClientRect(s_edChildWnd, &cr);
    extern u32 psCurrentVidMode[2];
    Device.dwWidth = u32(cr.right - cr.left);
    Device.dwHeight = u32(cr.bottom - cr.top);
    Device.fWidth_2 = float(Device.dwWidth) * 0.5f;
    Device.fHeight_2 = float(Device.dwHeight) * 0.5f;
    psCurrentVidMode[0] = Device.dwWidth;
    psCurrentVidMode[1] = Device.dwHeight;

    Device.Reset(false);
    Msg("[Ed] resized embedded viewport to %ux%u", Device.dwWidth, Device.dwHeight);
}

void Ed_Shutdown()
{
    if (!s_edBooted)
        return;
    s_edBooted = false;

    // Mirrors Startup()'s post-loop teardown.
    xr_delete(g_SpatialSpacePhysic);
    xr_delete(g_SpatialSpace);
    xr_delete(g_pGamePersistent);
    xr_delete(pApp);
    Engine.Event.Dump();
    destroyInput();
    destroySettings();
    LALib.OnDestroy();
    destroyConsole();
    destroySound();
    destroyEngine();
    Core._destroy();

    // The engine's atexit guard R_ASSERTs on ExitFromWinMain to catch abnormal exits.
    // When hosted, the engine's WinMain never runs (we boot via Ed_Init), so the guard
    // would false-fire "Unexpected application exit" on the host's clean shutdown. We
    // reached Ed_Shutdown normally — mark the exit as expected.
    ExitFromWinMain = true;

    if (s_edChildWnd)
    {
        DestroyWindow(s_edChildWnd);
        s_edChildWnd = nullptr;
        Device.m_hWnd = nullptr;
        g_ed_embedded = false;
    }
}

CApplication::CApplication() : loadingScreen(nullptr)
{
    ll_dwReference = 0;

    // events
    eQuit = Engine.Event.Handler_Attach("KERNEL:quit", this);
    eStart = Engine.Event.Handler_Attach("KERNEL:start", this);
    eDisconnect = Engine.Event.Handler_Attach("KERNEL:disconnect", this);

    // levels
    Level_Current = 0;
    Level_Scan();

    // Register us
    Device.seqFrame.Add(this, REG_PRIORITY_HIGH + 1000);

    Console->Show();
}

CApplication::~CApplication()
{
    Console->Hide();

    Device.seqFrame.Remove(this);

    // events
    Engine.Event.Handler_Detach(eDisconnect, this);
    Engine.Event.Handler_Detach(eStart, this);
    Engine.Event.Handler_Detach(eQuit, this);
}

void CApplication::OnEvent(EVENT E, u64 P1, u64 P2)
{
    if (E == eQuit)
    {
        PostQuitMessage(0);

        for (auto& Level : Levels)
        {
            xr_free(Level.folder);
        }
    }
    else if (E == eStart)
    {
        LPSTR op_server = LPSTR(P1);
        LPSTR op_client = LPSTR(P2);
        R_ASSERT(0 == g_pGameLevel);
        R_ASSERT(g_pGamePersistent);

        // An exception escaping level creation would be swallowed by the pureFrame
        // guard, leaving a half-initialized g_pGameLevel and a disconnect/reconnect
        // livelock (seen 16-07: frozen game, 1200+ per-frame Disconnects). Unrecoverable
        // either way — die loudly with the cause instead.
        try
        {
            Console->Execute("main_menu off");
            Console->Hide();

            g_pGamePersistent->PreStart(op_server);

            g_pGameLevel = (IGame_Level*)NEW_INSTANCE(CLSID_GAME_LEVEL);

            pApp->LoadBegin();
            g_pGamePersistent->Start(op_server);
            g_pGameLevel->net_Start(op_server, op_client);
            pApp->LoadEnd();
        }
        catch (const std::exception& e)
        {
            FATAL("level start threw [%s: %s] — state unrecoverable", typeid(e).name(), e.what());
        }
        catch (...)
        {
            FATAL("level start threw [non-std; lua top: %s] — state unrecoverable", g_lua_error_peek ? g_lua_error_peek() : "<no hook>");
        }
        xr_free(op_server);
        xr_free(op_client);
    }
    else if (E == eDisconnect)
    {
        if (g_pGameLevel)
        {
            Console->Execute("main_menu off");
            Console->Hide();

            // Same as eStart: a throw mid-teardown (e.g. a Lua error raised while level
            // members release luabind refs after the script-engine restart) leaves a
            // dangling half-destroyed level → per-frame Disconnect livelock. Die loudly.
            try
            {
                g_pGameLevel->net_Stop();
                xr_delete(g_pGameLevel);
            }
            catch (const std::exception& e)
            {
                FATAL("level unload threw [%s: %s] — state unrecoverable", typeid(e).name(), e.what());
            }
            catch (...)
            {
                FATAL("level unload threw [non-std; lua top: %s] — state unrecoverable", g_lua_error_peek ? g_lua_error_peek() : "<no hook>");
            }

            Console->Show();

            if ((FALSE == Engine.Event.Peek("KERNEL:quit")) && (FALSE == Engine.Event.Peek("KERNEL:start")))
            {
                Console->Execute("main_menu on");
            }
        }

        g_pGamePersistent->Disconnect();
    }
}

static CTimer phase_timer;
extern ENGINE_API BOOL g_appLoaded = FALSE;

const char* (*g_lua_error_peek)() = nullptr;

// Per-member seqFrame profiling (see pure.h). The VK profiler installs the
// hook; device.cpp arms the live cb around seqFrame.Process only.
seq_profile_cb g_seq_profile_cb   = nullptr;
seq_profile_cb g_seq_profile_hook = nullptr;

void CApplication::LoadBegin(bool quick)
{
    ll_dwReference++;
    if (1 == ll_dwReference)
    {
        g_appLoaded = FALSE;

        phase_timer.Start();

        load_stage = 0;
        if (quick)
            max_load_stage = 4; // при быстром сохранении меньше фаз загрузки. 4 вроде б
        else 
            max_load_stage = 16; // 17; //KRodin: пересчитал кол-во стадий, у нас их 15 при создании НИ + 1 на автопаузу
    }
}

void CApplication::LoadEnd()
{
    ll_dwReference--;
    if (0 == ll_dwReference)
    {
        Msg("* phase time: %d ms", phase_timer.GetElapsed_ms());
        Msg("* phase cmem: %d K", Memory.mem_usage() / 1024);
        Console->Execute("stat_memory");
        g_appLoaded = TRUE;
    }
}

void CApplication::SetLoadingScreen(ILoadingScreen* newScreen)
{
    if (loadingScreen)
    {
        Log("! Trying to create new loading screen, but there is already one..");
        xr_delete(newScreen);
        return;
    }

    loadingScreen = newScreen;
}

void CApplication::DestroyLoadingScreen() { xr_delete(loadingScreen); }

void CApplication::LoadDraw() const
{
    if (g_appLoaded)
        return;
    Device.dwFrame += 1;

    if (!Device.Begin())
        return;

    load_draw_internal();

    Device.End();
}

void CApplication::LoadForceFinish() { loadingScreen->ForceFinish(); }

void CApplication::SetLoadStageTitle(pcstr _ls_title) { loadingScreen->SetStageTitle(_ls_title); }

void CApplication::LoadTitleInt() { loadingScreen->SetStageTip(); }

void CApplication::LoadStage()
{
    VERIFY(ll_dwReference);

    Msg("* phase time: %d ms", phase_timer.GetElapsed_ms());
    phase_timer.Start();
    Msg("* phase cmem: %d K", Memory.mem_usage() / 1024);


    LoadDraw();

    ++load_stage;
    // Msg("--LoadStage is [%d]", load_stage);
}

// Sequential
void CApplication::OnFrame()
{
    ZoneScoped;

    Engine.Event.OnFrame();

    {
        static u32 last_frame{0};

        if (Device.dwFrame > last_frame)
        {
            g_SpatialSpace->update(true);
            g_SpatialSpacePhysic->update(false);

            last_frame = Device.dwFrame + 30;
        }
    }

    if (g_pGameLevel)
        g_pGameLevel->SoundEvent_Dispatch();
}

void CApplication::Level_Append(LPCSTR folder)
{
    string_path N1, N2, N3, N4;
    strconcat(sizeof(N1), N1, folder, "level");
    strconcat(sizeof(N2), N2, folder, "level.ltx");
    strconcat(sizeof(N3), N3, folder, "level.geom");
    strconcat(sizeof(N4), N4, folder, "level.cform");
    if (FS.exist("$game_levels$", N1) && FS.exist("$game_levels$", N2) && FS.exist("$game_levels$", N3) && FS.exist("$game_levels$", N4))
    {
        Levels.emplace_back(sLevelInfo{xr_strdup(folder)});
    }
}

void CApplication::Level_Scan()
{
    for (auto& lvl : Levels)
    {
        xr_free(lvl.folder);
    }
    Levels.clear();

    xr_vector<char*>* folder = FS.file_list_open("$game_levels$", FS_ListFolders | FS_RootOnly);
    R_ASSERT(folder && folder->size());

    for (auto& i : *folder)
        Level_Append(i);

    FS.file_list_close(folder);

#ifdef DEBUG
    folder = FS.file_list_open("$game_levels$", "$debug$\\", FS_ListFolders | FS_RootOnly);
    if (folder)
    {
        string_path tmp_path;
        for (u32 i = 0; i < folder->size(); i++)
        {
            strconcat(sizeof(tmp_path), tmp_path, "$debug$\\", (*folder)[i]);
            Level_Append(tmp_path);
        }

        FS.file_list_close(folder);
    }
#endif
}

// Taken from OpenXray/xray-16 and refactored
void generate_logo_path(string_path& path, pcstr level_name, int num = -1)
{
    strconcat(sizeof(path), path, "intro\\intro_", level_name);

   // const auto len = xr_strlen(path);
  //  if (path[len - 1] == '\\')
  //      path[len - 1] = 0;

    if (num < 0)
        return;

    string16 buff;
    xr_strcat(path, sizeof(path), "_");
    xr_strcat(path, sizeof(path), _itoa(num + 1, buff, 10));
}

// Taken from OpenXray/xray-16 and refactored
// Return true if logo exists
// Always sets the path even if logo doesn't exist
bool validate_logo_path(string_path& path, pcstr level_name, int num = -1)
{
    generate_logo_path(path, level_name, num);
    string_path temp;
    return FS.exist(temp, "$game_textures$", path, ".dds") || FS.exist(temp, "$level$", path, ".dds");
}

void CApplication::Level_Set(u32 L)
{
    if (L >= Levels.size())
        return;

    Level_Current = L;
    FS.get_path("$level$")->_set(Levels[L].folder);

    std::string temp = Levels[L].folder;
    temp.pop_back();
    const char* level_name = temp.c_str();

    static string_path path;
    path[0] = 0;
    
    int count = 0;
    while (true)
    {
        if (validate_logo_path(path, level_name, count))
            count++;
        else
            break;
    }

    if (count)
    {
        const int curr = ::Random.randI(count);
        generate_logo_path(path, level_name, curr);
    }
    else if (!validate_logo_path(path, level_name))
    {
        if (!validate_logo_path(path, "no_start_picture"))
            path[0] = 0;
    }

    if (path[0])
        loadingScreen->SetLevelLogo(path);

    loadingScreen->SetLevelText(level_name);
}

int CApplication::Level_ID(const char* name, const char* ver, const bool bSet)
{
    int result = -1;
    bool arch_res = false;
/* //Вроде б нам это не нужно. что-то от мп
    auto it = FS.m_archives.begin();
    auto it_e = FS.m_archives.end();
    for (; it != it_e; ++it)
    {
        CLocatorAPI::archive& A = *it;
        if (A.hSrcFile == NULL)
        {
            LPCSTR ln = A.header->r_string("header", "level_name");
            LPCSTR lv = A.header->r_string("header", "level_ver");
            if (0 == _stricmp(ln, name) && 0 == _stricmp(lv, ver))
            {
                FS.LoadArchive(A);
                arch_res = true;
            }
        }
    }

    if (arch_res)
        Level_Scan();
*/
    string256 buffer;
    strconcat(sizeof(buffer), buffer, name, "\\");
    for (u32 I = 0; I < Levels.size(); ++I)
    {
        if (0 == _stricmp(buffer, Levels[I].folder))
        {
            result = int(I);
            break;
        }
    }

    if (bSet && result != -1)
        Level_Set(result);

    if (arch_res)
        g_pGamePersistent->OnAssetsChanged();

    return result;
}

extern void render_reshade_effects();

void CApplication::load_draw_internal() const
{
    if (use_reshade)
        render_reshade_effects();

    loadingScreen->Update(load_stage, max_load_stage);
}

bool CApplication::CheckCsCopMode()
{
#pragma todo("Simp: на будущее, активировать эту функцию когда будет надо!")
    if constexpr (true)
        return false;

    // Определять, что мы запускаем ЧН/ЗП будем по названию папки конфигов. Решение так себе, но ничего лучше пока не придумал.
    static const char* cfg_path{FS.get_path("$game_config$")->m_Path};
    static const char* cfg_dir_name{cfg_path + (strlen(cfg_path) - (strlen("configs") + 1))};
    static const bool res{!!strstr(cfg_dir_name, "configs")};

    static bool first_call{true};
    if (first_call)
    {
        Msg("~~[%s] Configs dir name: [%s]", __FUNCTION__, cfg_dir_name);
        first_call = false;
    }

    return res;
}

#pragma todo("Simp: нужно ли это? сомневаюсь.")
// Always request high performance GPU
extern "C" {
// https://docs.nvidia.com/gameworks/content/technologies/desktop/optimus.htm
_declspec(dllexport) u32 NvOptimusEnablement = 0x00000001; // NVIDIA Optimus

// https://gpuopen.com/amdpowerxpressrequesthighperformance/
_declspec(dllexport) u32 AmdPowerXpressRequestHighPerformance = 0x00000001; // PowerXpress or Hybrid Graphics
}