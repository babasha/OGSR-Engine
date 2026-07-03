// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

#include "stdafx.h"
#include "CRender_Vulkan.h"
#include "HW_Vulkan.h"
#include "vk_swapchain.h"
#include "vk_sync.h"
#include "vk_command_buffer.h"
#include "vk_barriers.h"
#include "vk_UIPipeline.h"
#include "vk_pass_world.h"
#include "vk_pass_sky.h"
#include "vk_pass_snow.h"
#include "vk_scene_color.h"     // HDR scene target (passes render here, then tonemap)
#include "vk_motionvec.h"       // screen-space motion vectors + r_mv_debug overlay
#include "vk_pass_tonemap.h"    // Pass_TonemapComposite — HDR → swapchain
#include "vk_pass_registry.h"   // VK::RegisterPass / ExecutePasses — framegraph seam
#include "vk_profiler.h"        // VK::Prof — GPU/CPU zones, debug labels, VRAM
#include "vk_imgui.h"           // VK::ImGuiVK — profiler overlay (r_profiler 2)
#include "vk_DetailManager.h"   // VK::CDetailManager — grass render entry
#include "vk_TreeManager.h"     // VK::CTreeManager — tree render entry
#include "vk_LODManager.h"      // VK::CLODManager — LOD imposter render entry
#include "vk_stub.h"
#include "vk_ModelPool.h"
#include "vk_Visual.h"  // vkRender_Visual base for the particle stub
#include "../../Include/xrRender/ParticleCustom.h"
#include "vk_Particles.h"        // vkCreateParticle (real particle visuals)
#include "vk_pass_particles.h"   // VK::Pass_Particles registration
#include "vk_gpu_particles.h"    // VK::GPUParticles — GPU-driven particles (Phase 0)
#include "vk_wallmarks.h"        // VK::Wallmarks — bullet holes / decals on level geometry
#include "vk_rain.h"             // VK::Pass_Rain — rain drops/splashes + thunderbolt
#include "vk_pass_shadow.h"      // VK::Pass_SunShadow (sun shadow caster, before World)
#include "vk_pass_sunshafts.h"   // VK::Pass_SunShafts (volumetric god rays, after Sky)
#include "vk_pass_lightcones.h"  // VK::Pass_LightCones (per-light volumetric beams)
#include "vk_light.h"            // VK::vkLight (dynamic point/spot lights, STEP 3)
#include "../../xrCDB/ISpatial.h"      // g_SpatialSpace, ISpatial, STYPE_RENDERABLE (dynamic collection)
#include "../../xrCDB/Frustum.h"       // CFrustum
#include "../../xr_3da/IRenderable.h"  // IRenderable::renderable_Render
#include "../../xr_3da/IGame_Level.h"  // g_pGameLevel (skip dynamic collection in the menu)
#include "../../xr_3da/CustomHUD.h"    // g_hud->Render_MAIN (collect first-person HUD visuals)
#include "../../xr_3da/PS_instance.h"  // CPS_Instance::PerformFrame (tick particle simulation)
#include <mutex>                       // per-object hemi cache guard (add_Visual)

CRender RImplementation;

// --- Render-pass registry (framegraph seam, declared in vk_pass_registry.h) ---
namespace VK
{
    static xr_vector<RenderPass> s_Passes;

    void RegisterPass(const char* name, PassExecuteFn fn) { s_Passes.push_back({ name, std::move(fn) }); }
    void ClearPasses() { s_Passes.clear(); }

    // --- GPU/CPU pass timing now lives in the global profiler (vk_profiler.*):
    // named timestamp zones + CPU time + min/avg/max history + VRAM + debug
    // labels (so Nsight/RenderDoc show our passes by name). ExecutePasses just
    // brackets each pass in a zone.
    void ExecutePasses(FrameContext& ctx)
    {
        Prof::FrameBegin(ctx.cmd, CommandManager.GetCurrentFrame());
        Prof::MaybeLog();   // emits [VK Perf] every ~5s or on a `vk_perf` MARK

        for (size_t i = 0; i < s_Passes.size(); ++i)
        {
            // Order pass i's color/depth writes after pass i-1's (no layout change).
            // First pass needs no barrier: CRender::Begin's TRANSFER_DST→COLOR
            // transition already ordered the clear before any attachment access.
            if (i != 0) SceneAttachmentBarrier(ctx.cmd);
            const int z = Prof::ZoneBegin(ctx.cmd, s_Passes[i].name);
            s_Passes[i].execute(ctx);
            Prof::ZoneEnd(ctx.cmd, z);
        }
        // NOTE: FrameEnd is NOT called here — the Tonemap composite runs after
        // ExecutePasses (CRender::Render) and is wrapped in its own zone there,
        // so FrameEnd closes the frame after Tonemap. (Tonemap hosts the SSR
        // puddle march — a key rain cost — so it must be profiled too.)
    }

    void PassTimingDestroy() { Prof::Shutdown(); }
}

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

// Per-frame context handed to Pass_* functions. Begin() fills it; Render()
// forwards to Pass_World. Lives next to FrameInFlight for now — both will
// merge once the framegraph owns the frame.
VK::FrameContext g_FrameCtx{};

// Pre-frame clear color. UI panels render their own backdrops over this.
constexpr VkClearColorValue kBackgroundClear{ {0.0f, 0.0f, 0.0f, 1.0f} };

// Set by CRender::Screenshot, consumed (and cleared) by CRender::End once the
// frame has finished rendering but before vkQueuePresent.
xr_string g_PendingScreenshotPath;
} // namespace

// Allocator-managed staging buffer kept alive across End() submit so the GPU
// can finish the copy before we map. Cleared on the next screenshot.
struct PendingShot {
    VkBuffer       buf{};
    VmaAllocation  alloc{};
    VmaAllocationInfo info{};
    u32            width{}, height{};
    xr_string      path;
};
static PendingShot g_pendingShot{};

static void ScreenshotRecord(VkCommandBuffer cmd, VkImage src, VkExtent2D extent, const xr_string& path)
{
    if (g_pendingShot.buf) {
        // Previous shot wasn't drained yet — skip this one.
        return;
    }
    const u32 width  = extent.width;
    const u32 height = extent.height;
    const VkDeviceSize size = (VkDeviceSize)width * height * 4;  // BGRA8

    VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size  = size;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    if (vmaCreateBuffer(VulkanHW.m_Allocator, &bci, &aci,
                        &g_pendingShot.buf, &g_pendingShot.alloc, &g_pendingShot.info) != VK_SUCCESS) {
        Msg("![VK Screenshot] staging buffer alloc failed");
        g_pendingShot = {};
        return;
    }

    auto barrier = [&](VkImageLayout oldL, VkImageLayout newL,
                       VkPipelineStageFlags2 srcStg, VkAccessFlags2 srcAcc,
                       VkPipelineStageFlags2 dstStg, VkAccessFlags2 dstAcc) {
        VkImageMemoryBarrier2 b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        b.srcStageMask = srcStg; b.srcAccessMask = srcAcc;
        b.dstStageMask = dstStg; b.dstAccessMask = dstAcc;
        b.oldLayout = oldL; b.newLayout = newL;
        b.image = src;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.layerCount = 1;
        VkDependencyInfo di{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &di);
    };

    // Called from End() AFTER the image has been transitioned to PRESENT_SRC
    // (post-EndUIPass / post-tail-transition). Move PRESENT_SRC → TRANSFER_SRC
    // for readback, then back so vkQueuePresent still sees PRESENT_SRC.
    barrier(VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COPY_BIT,         VK_ACCESS_2_TRANSFER_READ_BIT);

    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = { width, height, 1 };
    vkCmdCopyImageToBuffer(cmd, src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g_pendingShot.buf, 1, &region);

    barrier(VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            VK_PIPELINE_STAGE_2_COPY_BIT,         VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT);

    g_pendingShot.width  = width;
    g_pendingShot.height = height;
    g_pendingShot.path   = path;
}

// Drained next frame after the GPU has definitely flushed the CopyImageToBuffer.
static void ScreenshotDrain()
{
    if (!g_pendingShot.buf) return;

    const u8* mapped = (const u8*)g_pendingShot.info.pMappedData;
    if (mapped) {
        FILE* f = nullptr;
        fopen_s(&f, g_pendingShot.path.c_str(), "wb");
        if (f) {
            u8 hdr[18]{};
            hdr[2]  = 2;                  // uncompressed true-color
            hdr[12] = (u8)(g_pendingShot.width  & 0xff); hdr[13] = (u8)((g_pendingShot.width  >> 8) & 0xff);
            hdr[14] = (u8)(g_pendingShot.height & 0xff); hdr[15] = (u8)((g_pendingShot.height >> 8) & 0xff);
            hdr[16] = 24;                 // bits per pixel — TGA bottom-up
            fwrite(hdr, 1, sizeof(hdr), f);

            xr_vector<u8> row(g_pendingShot.width * 3);
            for (int y = (int)g_pendingShot.height - 1; y >= 0; --y) {
                const u8* src_row = mapped + (size_t)y * g_pendingShot.width * 4;
                for (u32 x = 0; x < g_pendingShot.width; ++x) {
                    row[x * 3 + 0] = src_row[x * 4 + 0]; // B
                    row[x * 3 + 1] = src_row[x * 4 + 1]; // G
                    row[x * 3 + 2] = src_row[x * 4 + 2]; // R
                }
                fwrite(row.data(), 1, row.size(), f);
            }
            fclose(f);
            Msg("[VK Screenshot] saved %s (%ux%u)", g_pendingShot.path.c_str(), g_pendingShot.width, g_pendingShot.height);
        } else {
            Msg("![VK Screenshot] fopen failed: %s", g_pendingShot.path.c_str());
        }
    }
    vmaDestroyBuffer(VulkanHW.m_Allocator, g_pendingShot.buf, g_pendingShot.alloc);
    g_pendingShot = {};
}

CRender::CRender()  = default;
CRender::~CRender() = default;

// ----- Loading / Unloading --------------------------------------------------
// level_Load / level_Unload bodies live in rvk_loader.cpp.
//
// NOTE: OGSR engine doesn't call CRender::create()/destroy() through any path
// our build wires up (R4's flow goes through dxRenderDeviceRender). So these
// stay no-ops here and the actual lifecycle (Models, WorldPipeline) is owned
// by vkRenderDeviceRender::Create()/Destroy(), which IS called.
void CRender::create()       { VK_STUB_ONCE("CRender"); }
void CRender::destroy()      { VK_STUB_ONCE("CRender"); }

void CRender::reset_begin()  { VK_STUB_ONCE("CRender"); }
void CRender::reset_end()    { VK_STUB_ONCE("CRender"); }

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
// Resolves OGF_CHILDREN_L links (child IDs into the level's Visuals[] array)
// during hierarchy/LOD load. Without this, every CHILDREN_L lookup silently
// returns null — composite visuals end up empty (e.g. tree LOD containers
// lose their high-quality mesh children, so trees vanish from the level).
IRenderVisual* CRender::getVisual(int id)
{
    if (id >= 0 && id < (int)Visuals.size())
        return Visuals[id];
    return nullptr;
}
u32 CRender::getVisualCount() { return (u32)Visuals.size(); }

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
namespace {
// Per-object sky-visibility (hemi) — amortized ray tracing, mirrors the engine's
// CROS_impl (LightTrack.cpp). Dynamic objects have no baked lightmap occlusion,
// so without this they get the full hemisphere sky ambient even in a basement
// and glow. We trace a few rays over the upper hemisphere against the STATIC
// level: hemi = fraction that reach open sky. Cached per renderable, re-traced
// every kRetraceFrames or on move, smoothed.
struct HemiEntry { float target = 1.0f; float smooth = 1.0f; u32 frame = 0; Fvector pos{ 1e9f, 1e9f, 1e9f }; };
xr_unordered_map<const void*, HemiEntry> s_hemiCache;
std::mutex s_hemiMtx;

const Fvector kHemiDirs[9] = {
    { 0.f, 1.f, 0.f },
    { 0.707f, 0.707f, 0.f }, { -0.707f, 0.707f, 0.f },
    { 0.f, 0.707f, 0.707f }, { 0.f, 0.707f, -0.707f },
    { 0.5f, 0.707f, 0.5f },  { -0.5f, 0.707f, 0.5f },
    { 0.5f, 0.707f, -0.5f }, { -0.5f, 0.707f, -0.5f },
};
constexpr u32   kHemiRetraceFrames = 20;
constexpr float kHemiRange         = 40.f;   // open sky beyond this

float ComputeObjectHemi(const void* key, const Fvector& center)
{
    if (!g_pGameLevel) return 1.0f;
    std::lock_guard<std::mutex> lk(s_hemiMtx);
    if (s_hemiCache.size() > 4096) s_hemiCache.clear();   // bound churn (e.g. particle visuals)
    HemiEntry& e = s_hemiCache[key];
    const bool moved = center.distance_to_sqr(e.pos) > 0.25f;   // > 0.5 m
    if (e.frame == 0 || moved || (Device.dwFrame - e.frame) > kHemiRetraceFrames) {
        u32 miss = 0;
        for (const Fvector& d : kHemiDirs)
            if (!g_pGameLevel->ObjectSpace.RayTest(center, d, kHemiRange, collide::rqtStatic, nullptr, nullptr))
                ++miss;
        e.target = float(miss) / 9.0f;
        e.frame  = Device.dwFrame ? Device.dwFrame : 1;
        e.pos    = center;
    }
    const float k = 1.f - expf(-Device.fTimeDelta * 4.0f);   // smooth (eye-adapt-like)
    e.smooth += (e.target - e.smooth) * k;
    return e.smooth;
}
}  // namespace

// Dynamic (spawned) visual registration. Called per-frame from the object
// traversal kicked off in CRender::Calculate(). Collect {visual, world-xform,
// hemi}; Pass_World drains the list. (Was a no-op stub — dynamic objects never drew.)
void CRender::add_Visual(u32, IRenderable* root, IRenderVisual* V, Fmatrix& m)
{
    if (!V) return;
    auto* rv = static_cast<vkRender_Visual*>(V);
    // HUD visuals (player hands + active item) are flagged via renderable_HUD()
    // by g_hud->Render_MAIN; they render in a separate near-depth/HUD-FOV pass.
    const bool isHud = (root && root->renderable_HUD());
    auto& list = isHud ? VK::g_HudVisuals : VK::g_DynamicVisuals;

    // Sky-visibility for the ambient gate (world dynamics only; HUD is hand-lit).
    float hemi = 1.0f;
    if (!isHud && root) {
        Fvector center; m.transform_tiny(center, rv->vis.sphere.P);
        hemi = ComputeObjectHemi(root, center);
    }
    list.push_back({ rv, m, hemi });

    // Skeleton wallmarks (blood on NPCs): age + append this skeleton's visible
    // marks for the Wallmarks pass (self-guarded per frame; bridge defined in
    // vk_SkeletonCustom.cpp — needs the full CKinematics type).
    extern void VK_SkelWM_Calculate(IRenderVisual* V, bool hud);
    VK_SkelWM_Calculate(V, isHud);
}

// Wallmark bridges (vk_RenderFactory.cpp) — pull the decal texture name out of
// the renderer-side IWallMarkArray / IUIShader implementations.
extern const char* VK_WallmarkArray_Pick(IWallMarkArray* A);
extern const char* VK_UIShaderTexName(IUIShader* S);

void CRender::add_StaticWallmark(const wm_shader& S, const Fvector& P, float s,
                                        CDB::TRI* T, Fvector* V)
{
    if (const char* tex = VK_UIShaderTexName(S.operator->()))
        VK::Wallmarks::AddStatic(tex, P, s, T, V);
}

void CRender::add_StaticWallmark(IWallMarkArray* pArray, const Fvector& P, float s,
                                        CDB::TRI* T, Fvector* V)
{
    if (const char* tex = VK_WallmarkArray_Pick(pArray))
        VK::Wallmarks::AddStatic(tex, P, s, T, V);
}

// add_SkeletonWallmark (blood on NPCs) is defined in vk_SkeletonCustom.cpp —
// it needs the full CKinematics/CSkeletonWallmark compat context.

void CRender::clear_static_wallmarks() { VK::Wallmarks::Clear(); }

// Minimal IRender_ObjectSpecific so AI memory (CVisualMemoryManager) and rain
// queries can read neutral luminocity without nulldereffing.
namespace {
struct vkROS_Stub final : public IRender_ObjectSpecific
{
    float m_hemi_cube[6]{ 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f };
    float        get_luminocity() override              { return 0.5f; }
    float        get_luminocity_hemi() override         { return 0.5f; }
    const float* get_luminocity_hemi_cube() override    { return m_hemi_cube; }
};
} // namespace
IRender_ObjectSpecific* CRender::ros_create(IRenderable*)            { return xr_new<vkROS_Stub>(); }
void                    CRender::ros_destroy(IRender_ObjectSpecific*& p) { xr_delete(p); }

// ----- Lighting -------------------------------------------------------------
// Real dynamic lights (STEP 3): vkLight registers itself in the global registry;
// EnvLight::Update collects the nearest active ones into the shared Lighting UBO
// each frame and the forward shaders accumulate them. See vk_light.{h,cpp}.
IRender_Light* CRender::light_create() { return xr_new<VK::vkLight>(); }

// ----- Particles ------------------------------------------------------------
// Backed by the Vulkan PS library (vk_PSLibrary.cpp).
extern void  VK_ParticleEffectFillName(xr_vector<shared_str>&);
extern void  VK_ParticleGroupFillName(xr_vector<shared_str>&);
extern float VK_GetParticlesTimeLimit(const char* name);

void CRender::ParticleEffectFillName(xr_vector<shared_str>& s) { VK_ParticleEffectFillName(s); }
void CRender::ParticleGroupFillName(xr_vector<shared_str>& s)  { VK_ParticleGroupFillName(s); }
float CRender::GetParticlesTimeLimit(LPCSTR name)             { return VK_GetParticlesTimeLimit(name); }

// Inert particle visual: satisfies both IRenderVisual (via vkRender_Visual)
// and IParticleCustom so xrGame's `smart_cast<IParticleCustom*>(visual)` lands
// on a non-null object. Reports zero lifetime / not playing / no particles —
// the engine ticks it once, sees no work, and moves on. Real particle classes
// (vk_ParticleEffect / vk_ParticleGroup) replace this when they port over.
namespace {
struct vkParticleVisual_Stub final : public vkParticleVisual
{
    shared_str  m_name;

    vkParticleVisual_Stub() { Type = MT_PARTICLE_EFFECT; }

    // vkParticleVisual — nothing to draw (dcast_ParticleCustom comes from the base).
    void  CollectEffects(xr_vector<vkCParticleEffect*>&) override {}

    // IParticleCustom — all no-op / safe defaults.
    void  OnDeviceCreate()                 override {}
    void  OnDeviceDestroy()                override {}
    void  UpdateParent(const Fmatrix&, const Fvector&, BOOL) override {}
    void  OnFrame(u32)                     override {}
    void  Play()                           override {}
    void  Stop(BOOL)                       override {}
    BOOL  IsPlaying()                      override { return FALSE; }
    BOOL  IsDeferredStopped()              override { return TRUE;  }
    u32   ParticlesCount()                 override { return 0; }
    // Tiny positive value so CParticlesObject::Init takes the time-limited
    // branch (m_iLifeTime = 1 ms) instead of the looped one. Returning 0
    // here previously fataled when auto-remove fire-effects (BM16 etc.)
    // tried to spawn through us — looped + auto-remove is illegal.
    float GetTimeLimit()                   override { return 0.001f; }
    const shared_str Name()                override { return m_name; }
    void  SetHudMode(BOOL)                 override {}
    BOOL  GetHudMode()                     override { return FALSE; }
};
} // namespace

// ----- Models — delegates to vkModelPool (created in CRender::create) -------
IRenderVisual* CRender::model_CreateParticles(LPCSTR name, BOOL /*bNoPool*/)
{
    // Resolve against the loaded PS library → dedicated vk particle visual
    // (effect or group), simulated via PAPI and drawn by Pass_Particles.
    if (vkParticleVisual* p = vkCreateParticle(name))
        return p;

    // Unknown name → inert stub so spawn paths never null-deref.
    auto* s   = xr_new<vkParticleVisual_Stub>();
    s->m_name = name ? name : "";
    return s;
}

IRenderVisual* CRender::model_Create(LPCSTR name, IReader* data)
{
    IRenderVisual* v = Models ? Models->Create(name, data, true) : nullptr;
    if (!v)
        Msg("![CRender::model_Create] '%s' returned nullptr (Models=%p)", name ? name : "<null>", (void*)Models);
    return v;
}

IRenderVisual* CRender::model_CreateChild(LPCSTR name, IReader* data)
{
    return Models ? Models->CreateChild(name, data) : nullptr;
}

IRenderVisual* CRender::model_Duplicate(IRenderVisual* V)
{
    if (!Models || !V) return nullptr;
    return Models->Instance_Duplicate(static_cast<vkRender_Visual*>(V));
}

void CRender::model_Delete(IRenderVisual*& V, BOOL bDiscard)
{
    if (!Models || !V) { V = nullptr; return; }
    auto* p = static_cast<vkRender_Visual*>(V);
    Models->Delete(p, bDiscard);
    V = p;  // pool nulls the pointer; mirror back to caller
}

void CRender::model_Logging(BOOL) {}
void CRender::models_Prefetch() {}
void CRender::models_Clear(BOOL b_complete) { if (Models) Models->ClearPool(b_complete); }
void CRender::models_savePrefetch() {}
void CRender::models_begin_prefetch1(bool) {}

// append_SkeletonWallmark is defined in vk_SkeletonCustom.cpp (needs the
// complete CSkeletonWallmark type) — it queues the mark for VK::Wallmarks.

// ----- Frame ----------------------------------------------------------------
void CRender::Calculate()
{
    // Dynamic-visual collection. R4 does this inside Calculate via the dsgraph
    // (portal + HOM culling); here it's a minimal frustum query with no culling.
    // Clear the per-frame list, then drive each visible renderable's
    // renderable_Render(), which calls back into add_Visual({visual, xform}).
    VK::g_DynamicVisuals.clear();
    VK::g_HudVisuals.clear();
    // Only collect dynamics while a level is loaded+rendering. CMainMenu::OnRender
    // also calls Calculate(): there g_pGameLevel is null and g_SpatialSpace is an
    // empty tree, so q_frustum would walk a null root and AV.
    if (!g_pGameLevel || !g_SpatialSpace)
        return;

    CFrustum frustum;
    Fmatrix fullT = Device.mFullTransform;   // CreateFromMatrix takes a non-const ref
    frustum.CreateFromMatrix(fullT, FRUSTUM_P_LRTB | FRUSTUM_P_FAR);
    // Publish the camera frustum on the interface: CKinematics::CalculateWallmarks
    // (skeleton blood decals) and engine code (IGame_Persistent grass benders)
    // read ::Render->ViewBase — it was never set on the VK path before.
    ViewBase = frustum;

    static xr_vector<ISpatial*> lstRenderables;
    lstRenderables.clear();
    g_SpatialSpace->q_frustum(lstRenderables, 0, STYPE_RENDERABLE, frustum);

    for (ISpatial* sp : lstRenderables)
    {
        if (!sp) continue;
        if (IRenderable* r = sp->dcast_Renderable())
            r->renderable_Render(0u, r);
    }

    // Tick particle simulation for visible particle instances. R4 does this in
    // CRender::calculate_particles_async (PerformFrame → CParticlesObject::DoWork
    // → the visual's OnFrame, which steps PAPI). Without it on-screen particles
    // never advance: CParticlesObject::shedule_Update only updates particles that
    // were NOT rendered last frame. DoWork is idempotent per Device.dwFrame, so
    // this never double-ticks against shedule_Update.
    {
        static xr_vector<ISpatial*> lstParticles;
        lstParticles.clear();
        g_SpatialSpace->q_frustum(lstParticles, 0, STYPE_RENDERABLE | STYPE_PARTICLE, frustum);
        for (ISpatial* sp : lstParticles)
            if (CPS_Instance* ps = smart_cast<CPS_Instance*>(sp))
                ps->PerformFrame();
    }

    // First-person HUD (player hands + active item). R4 does this at the tail of
    // its dsgraph collection (r4_R_render.cpp); Render_MAIN sets renderable_HUD()
    // on the root and calls add_Visual → routed into g_HudVisuals (drawn by the
    // near-depth HUD branch of Pass_Skinned).
    if (g_hud)
        g_hud->Render_MAIN(0u);
}
void CRender::Render()
{
    VK_STUB_ONCE("CRender");

    // Register the scene passes once, in draw order. Adding a pass later (lights,
    // SSAO, bloom) is a RegisterPass call, not an edit here. ExecutePasses owns
    // the inter-pass barriers.
    //   World → Grass (depth-tests world geometry, writes own depth) → Sky (fills
    //   uncleared z==1 pixels behind the world). Grass skips if Details didn't
    //   load (no level.details) or compute is disabled.
    static bool s_registered = false;
    if (!s_registered)
    {
        // Sun shadow caster FIRST: renders static world depth from the sun POV into
        // the shadow map (own depth target), leaves it SHADER_READ for the receivers.
        VK::RegisterPass("SunShadow", [](VK::FrameContext& c) { VK::Pass_SunShadow(c); });
        VK::RegisterPass("World", [](VK::FrameContext& c) { VK::Pass_World(c); });
        // Dense snow surface (VHM-style): drawn on top of the world, owns the near snow
        // geometry + footprint dents when r_snow_mesh is on. Writes depth (trees/grass
        // test against it). No-op unless r_snow_mesh + deform texture are active.
        VK::RegisterPass("SnowMesh", [](VK::FrameContext& c) { VK::Pass_SnowMesh(c); });
        VK::RegisterPass("Trees", [](VK::FrameContext& c) {
            if (RImplementation.Trees && RImplementation.Trees->IsReady())
                RImplementation.Trees->Render(c);
        });
        VK::RegisterPass("Grass", [](VK::FrameContext& c) {
            if (RImplementation.Details && RImplementation.Details->IsLoaded())
                RImplementation.Details->Render(c);
        });
        VK::RegisterPass("LODs", [](VK::FrameContext& c) {
            if (RImplementation.LODs && RImplementation.LODs->IsReady())
                RImplementation.LODs->Render(c);
        });
        // Dynamic motion vectors: now that the full opaque depth exists, overlay
        // self-moving geometry (skinned NPCs) onto the camera/static MV field.
        VK::RegisterPass("MotionVecDyn", [](VK::FrameContext& c) { VK::MotionVec::ExecuteDynamic(c.cmd, c); });
        VK::RegisterPass("Sky",   [](VK::FrameContext& c) { VK::Pass_Sky(c); });
        // Volumetric sun shafts (god rays): fullscreen raymarch vs the sun shadow
        // map, additive over the lit scene. After Sky so rays glow against it too.
        VK::RegisterPass("Shafts", [](VK::FrameContext& c) { VK::Pass_SunShafts(c); });
        // Per-light volumetric cones: real raymarched beams for the volumetric
        // spots (headlights/searchlights/pole lamps) — replaces the R4
        // lightplanes texture-sheet fakes. Additive over the lit scene.
        VK::RegisterPass("LightCones", [](VK::FrameContext& c) { VK::Pass_LightCones(c); });
        // Translucent GLASS panes — late flush AFTER the whole opaque world + sky
        // (they blend without z-write; anything drawn after them behind the pane
        // would overwrite the blended pixels). Before wallmarks/particles.
        VK::RegisterPass("Glass", [](VK::FrameContext& c) { VK::Pass_WorldGlass(c); });
        // Particles last: camera-facing billboards composited over the scene
        // (depth-tested vs world, additive/alpha blend, no depth write).
        // Wallmarks (bullet holes / scorch decals) over the lit world, before
        // particles so impact smoke composites on top of the fresh hole.
        VK::RegisterPass("Wallmarks", [](VK::FrameContext& c) { VK::Wallmarks::Render(c); });
        VK::RegisterPass("Particles", [](VK::FrameContext& c) { VK::Pass_Particles(c); });
        // GPU-driven particles (Phase 1): one hardcoded effect simulated +
        // drawn entirely on the GPU when r_gpu_particles=1 (off by default).
        VK::RegisterPass("GPUParticles", [](VK::FrameContext& c) { VK::GPUParticles::DispatchComputeAndDraw(c); });
        // Weather: rain drop streaks + ground splashes + thunderbolt, drawn
        // over everything (alpha/additive, depth-tested). Also ticks the rain
        // simulation (CEffect_Rain::Calculate) — the engine's RenderLast path
        // is never called on the VK build, so the pass owns it.
        VK::RegisterPass("Rain", [](VK::FrameContext& c) { VK::Pass_Rain(c); });
        s_registered = true;
    }

    VK::ExecutePasses(g_FrameCtx);

    // Resolve the HDR scene target to the swapchain: exposure + Reinhard-white
    // tonemap (R4 combine_tonemap). After this the swapchain holds the final LDR
    // image in COLOR_ATTACHMENT; the UI pass (End) draws on top, un-tonemapped.
    // Profiled as its own zone (hosts the SSR puddle march — a rain cost), then
    // FrameEnd closes the profiler frame.
    {
        const int z = VK::Prof::ZoneBegin(g_FrameCtx.cmd, "Tonemap");
        VK::Pass_TonemapComposite(g_FrameCtx);
        VK::Prof::ZoneEnd(g_FrameCtx.cmd, z);
    }

    // r_mv_debug: paint the motion-vector field over the final image (the
    // swapchain is COLOR_ATTACHMENT here; the UI pass in End draws on top). No-op
    // unless r_mv_debug is set. Verifies the MV reconstruction before any DLSS.
    VK::MotionVec::DrawDebugOverlay(g_FrameCtx.cmd, Swapchain.m_ImageViews[g_FrameCtx.imageIndex], g_FrameCtx.extent);
    VK::Prof::FrameEnd(g_FrameCtx.cmd);
}
void CRender::AfterWorldRender() { VK_STUB_ONCE("CRender"); }
void CRender::AfterUIRender()    { VK_STUB_ONCE("CRender"); }
void CRender::Screenshot(ScreenshotMode mode, LPCSTR name) { ScreenshotImpl(mode, name); }
void CRender::ScreenshotImpl(ScreenshotMode /*mode*/, LPCSTR name)
{
    string_path fname;
    if (name && name[0]) {
        xr_strcpy(fname, name);
    } else {
        // Timestamped under $screenshots$/ss_egorb_<timestamp>.tga
        SYSTEMTIME t; GetLocalTime(&t);
        string_path leaf;
        xr_sprintf(leaf, "ss_%s_%02d-%02d-%02d_%02d-%02d-%02d.tga",
                   Core.UserName, t.wDay, t.wMonth, t.wYear % 100, t.wHour, t.wMinute, t.wSecond);
        FS.update_path(fname, "$screenshots$", leaf);
    }
    g_PendingScreenshotPath = fname;
    Msg("[VK Screenshot] queued: %s", fname);
}

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
    // Rotate the UI vertex-buffer ring to this slot's region (fence above
    // proved the GPU finished reading it) and reset the write offset.
    VulkanUI::OnFrameBegin(frame);

    // HDR scene target: all scene passes render into this floating-point image
    // (so highlights exceed 1.0); the Tonemap pass maps it to the swapchain.
    // One per swapchain image index, sized to the swapchain.
    VK::SceneColor::EnsureSize(Swapchain.m_Extent, (u32)Swapchain.m_Images.size());
    VK::MotionVec::EnsureSize(Swapchain.m_Extent);   // RG16F motion target tracks the render resolution
    VkImage     hdrImage = VK::SceneColor::GetImage(g_FrameInFlight.imageIndex);
    VkImageView hdrView  = VK::SceneColor::GetView(g_FrameInFlight.imageIndex);

    // Populate the per-frame context handed to Pass_*. colorView/Image point at
    // the HDR target (scene passes are oblivious — Pass_TonemapComposite reads
    // it and writes the actual swapchain image in Render()).
    g_FrameCtx.cmd          = g_FrameInFlight.cmd;
    g_FrameCtx.imageIndex   = g_FrameInFlight.imageIndex;
    g_FrameCtx.extent       = Swapchain.m_Extent;
    g_FrameCtx.colorImage   = hdrImage;
    g_FrameCtx.colorView    = hdrView;
    g_FrameCtx.depthView    = Swapchain.m_DepthView;
    g_FrameCtx.viewProj     = &Device.mFullTransform;
    g_FrameCtx.viewProjPrev = nullptr;  // history matrix lands with motion vectors / TAA
    g_FrameCtx.frameIdx     = Device.dwFrame;
    g_FrameCtx.dt           = Device.fTimeDelta;

    // OGSR calls Begin → Calculate → Render → End — no Clear in between, so we
    // clear at frame start. Clear the HDR SCENE target (not the swapchain — that
    // gets fully overwritten by the Tonemap pass). Same single-layout idea: the
    // HDR image stays COLOR_ATTACHMENT for every scene pass; Pass_TonemapComposite
    // later flips it to SHADER_READ. The swapchain image is left UNDEFINED here
    // and transitioned by the tonemap pass.
    VK::ImageBarrier(g_FrameInFlight.cmd, hdrImage,
                     VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    VkImageSubresourceRange range{};
    range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    range.levelCount = 1;
    range.layerCount = 1;
    vkCmdClearColorImage(g_FrameInFlight.cmd, hdrImage,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         &kBackgroundClear, 1, &range);

    // TRANSFER_WRITE (clear) → COLOR_ATTACHMENT, ordering the clear before the
    // first pass's color load so ExecutePasses skips the barrier before pass 0.
    VK::ImageBarrier(g_FrameInFlight.cmd, hdrImage,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    // Depth: UNDEFINED → DEPTH_ATTACHMENT_OPTIMAL, ONCE per frame, here in Begin()
    // — not inside Pass_World. Pass_World can early-out (no level / pipelines not
    // ready) while Pass_Sky / Details still run and use the depth attachment with
    // loadOp=LOAD; if the transition lived in Pass_World those passes would begin
    // rendering against an UNDEFINED depth layout (VUID-vkCmdBeginRendering-09588).
    // UNDEFINED as oldLayout is fine — Pass_World clear-loads depth each frame and
    // nothing samples the previous contents. The renderer is depth-ONLY: every
    // depth attachment uses DEPTH_ATTACHMENT_OPTIMAL and no stencil aspect, so the
    // barrier must match (adding a stencil aspect to a depth-only layout would trip
    // VUID-VkImageMemoryBarrier2-aspectMask-08703). FindDepthFormat prefers the
    // stencil-less D32_SFLOAT to keep this consistent.
    {
        VkImageMemoryBarrier2 b{};
        b.sType         = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask  = VK_PIPELINE_STAGE_2_NONE;
        b.srcAccessMask = 0;
        b.dstStageMask  = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                          VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        b.dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        b.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout     = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        b.image         = Swapchain.m_DepthImage;
        b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        b.subresourceRange.levelCount = 1;
        b.subresourceRange.layerCount = 1;
        VkDependencyInfo di{};
        di.sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        di.imageMemoryBarrierCount = 1;
        di.pImageMemoryBarriers    = &b;
        vkCmdPipelineBarrier2(g_FrameInFlight.cmd, &di);
    }
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

    // Profiler overlay (r_profiler 2) — drawn on top of the finished frame while
    // the swapchain image is still COLOR_ATTACHMENT, before the PRESENT barrier.
    VK::ImGuiVK::DrawOverlay(g_FrameInFlight.cmd,
                             Swapchain.m_ImageViews[g_FrameInFlight.imageIndex],
                             Swapchain.m_Extent, Device.fTimeDeltaReal);

    VkImage image = Swapchain.m_Images[g_FrameInFlight.imageIndex];

    // Single-layout convention: the image has been in COLOR_ATTACHMENT since
    // Begin (scene passes + UI no longer transition it). Bring it home to
    // PRESENT exactly once. ScreenshotRecord below round-trips PRESENT↔TRANSFER_SRC
    // on its own, so it still works from here.
    VK::ImageBarrier(g_FrameInFlight.cmd, image,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    Swapchain.m_bRenderedThisFrame = false;

    // F12 / `screenshot` console cmd → record CopyImageToBuffer into this
    // frame's cmd buffer.
    const bool recordScreenshot = !g_PendingScreenshotPath.empty();
    if (recordScreenshot) {
        ScreenshotRecord(g_FrameInFlight.cmd, image, Swapchain.m_Extent, g_PendingScreenshotPath);
        g_PendingScreenshotPath.clear();
    }

    if (!CommandManager.End(g_FrameInFlight.cmd)) return;

    // Submit any async uploads staged this frame on the transfer queue (signals the
    // upload timeline). The graphics Submit below adds a wait on that timeline, so
    // draws consuming just-uploaded data are GPU-ordered after the copy completes.
    CommandManager.FlushUploads();

    const u32  frame = CommandManager.GetCurrentFrame();
    FrameSync& sync  = Sync.GetCurrentFrame(frame);
    // renderFinished is per swapchain IMAGE (indexed by imageIndex), not per
    // frame-in-flight — see CVulkanSwapchain::m_RenderFinished. imageAvailable +
    // inFlightFence stay per-frame-in-flight (the fence gates their reuse).
    VkSemaphore renderFinished = Swapchain.m_RenderFinished[g_FrameInFlight.imageIndex];
    CommandManager.Submit(g_FrameInFlight.cmd,
                          sync.imageAvailable,
                          renderFinished,
                          sync.inFlightFence);
    Swapchain.Present(renderFinished, g_FrameInFlight.imageIndex);

    // Block briefly so the staging buffer has the copy finished before we map.
    // Screenshots are rare (manual F12), one-frame stall is acceptable.
    if (recordScreenshot) {
        vkQueueWaitIdle(VulkanHW.m_GraphicsQueue);
        ScreenshotDrain();
    }

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
