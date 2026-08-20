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
#include "vk_texture_stream.h"   // VK::TextureStreamer::Frame — per-frame streaming tick
#include "vk_vram_stats.h"       // VK::Vram::DumpVmaJson — DLSS-rescue post-mortem
#include "vk_cluster_stream.h"   // VK::ClusterStream — cluster-IB page streaming (Stage B)
#include "vk_world_material.h"   // WorldMaterialCache::FrameTick — retired-set recycling
#include "vk_UIPipeline.h"
#include "vk_pass_world.h"
#include "vk_render_queue.h"   // g_RenderQueue (its glass/water lists are reset in Calculate)
#include "vk_pass_sky.h"
#include "vk_pass_snow.h"
#include "vk_scene_color.h"     // HDR scene target (passes render here, then tonemap)
#include "vk_shaders.h"         // g_ShaderManager (SPIRV loader) — editor overlay lines (Spike 2)
#include "vk_gfx_pipeline.h"    // VK::GfxPipelineBuilder
#include "vk_pass_context.h"    // VK::BeginOverlayRendering — editor overlay pass (Spike 2)
#include "vk_motionvec.h"       // screen-space motion vectors + r_mv_debug overlay
#include "vk_dlss.h"            // VK::Dlss — DLSS Super Resolution (DLAA) upscale before tonemap
#include "vk_sl.h"              // VK::SL — Streamline (Stage B: SR evaluate via sl.dlss, r_dlss_sl)

extern u32 ps_r_dlss;           // r_dlss — master toggle (latched OFF here on a failed evaluate)
extern int ps_r_dlss_sl;        // r_dlss_sl — route the SR evaluate through Streamline (FG prerequisite)
extern int ps_r_dlss_fg;        // r_dlss_fg — Stage C: DLSS Frame Generation (MFG)
extern int ps_r_dlss_fg_mult;   // r_dlss_fg_mult — frame multiplier 2..6
extern float ps_r_render_scale; // r_render_scale — internal resolution / display resolution when DLSS is not upscaling
#include "vk_pass_tonemap.h"    // Pass_TonemapComposite — HDR → swapchain
#include "vk_pass_registry.h"   // VK::RegisterPass / ExecutePasses — framegraph seam
#include "vk_pipeline_cache.h"  // PrewarmPending — the precache countdown asks whether the warm-up still needs frames
#include "vk_framegraph.h"      // VK::g_FrameGraph — frame-wide resource-state tracker
#include "vk_async.h"           // VK::Async — async compute (cross-queue) foundation
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
#include "vk_volumetrics.h"      // VK::Vol::SnapshotShadows (async fog shadow source)
#include "vk_pass_skinned.h"     // VK::Skinned_PreSkin (compute pre-skinning, before every consumer)
#include "vk_pass_lightcones.h"  // VK::Pass_LightCones (per-light volumetric beams)
#include "vk_pass_water.h"       // VK::Pass_Water (level water bodies)
#include "vk_light.h"            // VK::vkLight (dynamic point/spot lights, STEP 3)
#include "vk_instance_gpu.h"     // VK::InstanceGPU (host scene → GPU-driven instanced casters)
#include "../../xrCDB/ISpatial.h"      // g_SpatialSpace, ISpatial, STYPE_RENDERABLE (dynamic collection)
#include "../../xrCDB/Frustum.h"       // CFrustum
#include "../../xr_3da/IRenderable.h"  // IRenderable::renderable_Render
#include "../../xr_3da/IGame_Level.h"  // g_pGameLevel (skip dynamic collection in the menu)
#include "../../xr_3da/CustomHUD.h"    // g_hud->Render_MAIN (collect first-person HUD visuals)
#include "../../xr_3da/PS_instance.h"  // CPS_Instance::PerformFrame (tick particle simulation)
#include "../../xr_3da/IGame_Persistent.h" // g_pGamePersistent (VKEditor readiness gate — same as Pass_Sky)
#include <mutex>                       // per-object hemi cache guard (add_Visual)
#include <cstdlib>                     // std::getenv — XROS_AUTO_SHOT unattended capture
#include "../../xr_3da/XR_IOConsole.h"  // Console->Execute — XROS_AUTO_SHOT_CMDS sweep

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
}   // anonymous namespace

// Per-frame context handed to Pass_* functions. Begin() fills it; Render()
// forwards to Pass_World. EXTERNAL linkage on purpose: vk_env_light.cpp reads
// the scene render / display extents (ao_params screen scale + DLSS mip bias).
VK::FrameContext g_FrameCtx{};

namespace {

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

    if (VK::Vram::CreateBuffer(VulkanHW.m_Allocator, &bci, &aci,
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

    // The round-trip lands back on PRESENT_SRC, so the tracked LAYOUT is still
    // right — but the stage/access it went through are not the ones the registry
    // recorded at End(). Tell it, for the same reason the tonemap does: a tracker
    // is only worth having while it is never quietly behind.
    VK::g_FrameGraph.Track(src, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED)
        .Seed(src, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
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
    VK::Vram::DestroyBuffer(VulkanHW.m_Allocator, g_pendingShot.buf, g_pendingShot.alloc);
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

// ---------------------------------------------------------------------------
// SPIKE 2 (editor-on-Vulkan): minimal world-space line overlay — the primitive
// editors use for grid / gizmos / selection boxes. CDUInterface / IDebugRender
// are no-ops on VK; this proves an "EditorOverlay" framegraph pass (RegisterPass)
// drawing depth-tested 3D lines on top of the scene is feasible. Lazy-inits its
// own line pipeline + host vertex buffer. Additive/experimental; ROADMAP §11.
// ---------------------------------------------------------------------------
namespace EditorOverlay {

struct LineVertex { Fvector pos; u32 color; };   // 16 B; color = R8G8B8A8_UNORM (byte0=R)

static VkPipeline       s_Pipe      = VK_NULL_HANDLE;
static VkPipeline       s_PipeSwap  = VK_NULL_HANDLE;  // swapchain-format variant (post-tonemap editor overlay)
static bool             s_editorBackdrop = false;      // chunk 2: OFF — the HDR EditorClear grey (0.044→85) is the backdrop now,
                                                       // so the lit model (drawn into HDR, depth-tested) survives tonemap. A flat
                                                       // post-tonemap fill would overwrite it. Re-enable only for the no-model view.
static VkPipelineLayout s_Layout    = VK_NULL_HANDLE;
static VkBuffer         s_VB        = VK_NULL_HANDLE;
static VmaAllocation    s_VBAlloc   = VK_NULL_HANDLE;
static void*            s_VBMap     = nullptr;
static bool             s_initDone  = false;
static constexpr u32    MAX_VERTS   = 200000;
static xr_vector<LineVertex> s_Lines;

static inline u32 rgba(u8 r, u8 g, u8 b, u8 a) { return u32(r) | (u32(g) << 8) | (u32(b) << 16) | (u32(a) << 24); }

void AddLine(const Fvector& a, const Fvector& b, u32 col)
{
    if (s_Lines.size() + 2 > MAX_VERTS) return;
    s_Lines.push_back({ a, col });
    s_Lines.push_back({ b, col });
}

void AddBoxAA(const Fvector& c, float h, u32 col)
{
    Fvector p[8];
    for (int i = 0; i < 8; ++i)
        p[i].set(c.x + ((i & 1) ? h : -h), c.y + ((i & 2) ? h : -h), c.z + ((i & 4) ? h : -h));
    static const int e[12][2] = { {0,1},{2,3},{4,5},{6,7}, {0,2},{1,3},{4,6},{5,7}, {0,4},{1,5},{2,6},{3,7} };
    for (auto& pr : e) AddLine(p[pr[0]], p[pr[1]], col);
}

// Tight wireframe AABB from explicit min/max corners — the original SDK LevelEditor
// obtusely hugs an object's real bounding box (CEditableObject::GetBox), not a
// sphere-derived cube. 12 edges.
void AddBox(const Fvector& mn, const Fvector& mx, u32 col)
{
    Fvector p[8];
    for (int i = 0; i < 8; ++i)
        p[i].set((i & 1) ? mx.x : mn.x, (i & 2) ? mx.y : mn.y, (i & 4) ? mx.z : mn.z);
    static const int e[12][2] = { {0,1},{2,3},{4,5},{6,7}, {0,2},{1,3},{4,6},{5,7}, {0,4},{1,5},{2,6},{3,7} };
    for (auto& pr : e) AddLine(p[pr[0]], p[pr[1]], col);
}

// Small RGB world-axis tripod at the origin (X=red, Y=green, Z=blue) — the SDK's
// coordinate marker. Length in metres.
void AddAxis(const Fvector& o, float len)
{
    Fvector x = o, y = o, z = o;
    x.x += len; y.y += len; z.z += len;
    AddLine(o, x, rgba(220, 60, 60, 255));
    AddLine(o, y, rgba(60, 210, 60, 255));
    AddLine(o, z, rgba(70, 110, 230, 255));
}

static void EnsureInit()
{
    if (s_initDone) return;
    if (!g_ShaderManager || VulkanHW.m_Device == VK_NULL_HANDLE) return;
    s_initDone = true;   // one-shot: on failure we log and stay a no-op

    VkShaderModule vs = g_ShaderManager->Load("editor_line.vert.spv");
    VkShaderModule fs = g_ShaderManager->Load("editor_line.frag.spv");
    if (vs == VK_NULL_HANDLE || fs == VK_NULL_HANDLE) { Msg("![VK EditorOverlay] editor_line.{vert,frag}.spv load failed"); return; }

    VkPushConstantRange pcr{}; pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT; pcr.size = sizeof(Fmatrix);
    VkPipelineLayoutCreateInfo plci{}; plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_Layout);

    s_Pipe = VK::GfxPipelineBuilder(s_Layout)
        .Vert(vs).Frag(fs)
        .Binding(0, sizeof(LineVertex))
        .Attr(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0)
        .Attr(1, 0, VK_FORMAT_R8G8B8A8_UNORM,  12)
        .Topology(VK_PRIMITIVE_TOPOLOGY_LINE_LIST)
        .Depth(true, false)
        .Color(VK::SceneColor::Format()).BlendAlpha()
        .DepthTarget(Swapchain.m_DepthFormat)
        .Build("EditorOverlay lines");
    if (s_Pipe == VK_NULL_HANDLE)
        return;

    VkBufferCreateInfo bci{}; bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = VkDeviceSize(MAX_VERTS) * sizeof(LineVertex);
    bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT; bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo aci{}; aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo info{};
    if (VK::Vram::CreateBuffer(VulkanHW.m_Allocator, &bci, &aci, &s_VB, &s_VBAlloc, &info) != VK_SUCCESS) {
        Msg("![VK EditorOverlay] vertex buffer alloc failed"); return;
    }
    s_VBMap = info.pMappedData;
    Msg("[VK EditorOverlay] init OK (LINE_LIST pipeline + %u-vertex host buffer)", MAX_VERTS);
}

void Execute(VK::FrameContext& ctx)
{
    EnsureInit();
    if (s_Pipe == VK_NULL_HANDLE || s_VBMap == nullptr || s_Lines.empty()) { s_Lines.clear(); return; }

    u32 n = (u32)s_Lines.size(); if (n > MAX_VERTS) n = MAX_VERTS;
    memcpy(s_VBMap, s_Lines.data(), size_t(n) * sizeof(LineVertex));

    VK::BeginOverlayRendering(ctx.cmd, ctx, VK_ATTACHMENT_STORE_OP_DONT_CARE);   // test depth, don't write
    vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_Pipe);
    Fmatrix mvp = ctx.viewProj ? *ctx.viewProj : Fidentity;
    vkCmdPushConstants(ctx.cmd, s_Layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Fmatrix), &mvp);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(ctx.cmd, 0, 1, &s_VB, &off);
    vkCmdDraw(ctx.cmd, n, 1, 0, 0);
    vkCmdEndRendering(ctx.cmd);

    s_Lines.clear();
}

// Build the swapchain-format variant of the line pipeline (no depth): used to draw
// the editor grid/gizmos as a POST-TONEMAP overlay on the final LDR image, so the
// bright lines stay crisp (no HDR bloom, no tone-curve crush) — the DCC approach.
static void EnsureInitSwap()
{
    if (s_PipeSwap != VK_NULL_HANDLE) return;
    EnsureInit();  // shares s_Layout + s_VB
    if (s_Layout == VK_NULL_HANDLE || !g_ShaderManager) return;

    VkShaderModule vs = g_ShaderManager->Load("editor_line.vert.spv");
    VkShaderModule fs = g_ShaderManager->Load("editor_line.frag.spv");
    if (vs == VK_NULL_HANDLE || fs == VK_NULL_HANDLE) return;

    // Same as s_Pipe but no depth at all, into the final LDR swapchain image.
    s_PipeSwap = VK::GfxPipelineBuilder(s_Layout)
        .Vert(vs).Frag(fs)
        .Binding(0, sizeof(LineVertex))
        .Attr(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0)
        .Attr(1, 0, VK_FORMAT_R8G8B8A8_UNORM,  12)
        .Topology(VK_PRIMITIVE_TOPOLOGY_LINE_LIST)
        .Color(Swapchain.m_Format).BlendAlpha()   // B8G8R8A8_UNORM
        .Build("EditorOverlay lines (swapchain)");
    if (s_PipeSwap != VK_NULL_HANDLE)
        Msg("[VK EditorOverlay] post-tonemap (swapchain) pipeline OK");
}

// Draw the accumulated lines onto the final tonemapped swapchain image. Crisp +
// bright: direct LDR colors, no bloom, no tone curve. Consumes s_Lines.
void ExecutePostTonemap(VkCommandBuffer cmd, VkImageView colorView, VkExtent2D extent, const Fmatrix& mvp)
{
    EnsureInitSwap();
    if (s_PipeSwap == VK_NULL_HANDLE || s_VBMap == nullptr || s_Lines.empty()) { s_Lines.clear(); return; }

    u32 n = (u32)s_Lines.size(); if (n > MAX_VERTS) n = MAX_VERTS;
    memcpy(s_VBMap, s_Lines.data(), size_t(n) * sizeof(LineVertex));

    VK::RenderingBuilder(extent).Color(colorView).BeginFlipped(cmd);

    // Flat grey backdrop, direct LDR — matches the original SDK editor's
    // scene_clear_color 0x555555. Done here (post-tonemap) because the HDR clear
    // is crushed unpredictably by the tone curve. (Model support: switch to a
    // depth-tested fill so the lit object isn't overwritten — Phase B chunk 2.)
    if (s_editorBackdrop) {
        VkClearAttachment ca{}; ca.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; ca.colorAttachment = 0;
        ca.clearValue.color = { { 85.f / 255.f, 85.f / 255.f, 85.f / 255.f, 1.f } };
        VkClearRect cr{}; cr.rect.extent = extent; cr.layerCount = 1;
        vkCmdClearAttachments(cmd, 1, &ca, 1, &cr);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_PipeSwap);
    Fmatrix m = mvp;
    vkCmdPushConstants(cmd, s_Layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Fmatrix), &m);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &s_VB, &off);
    vkCmdDraw(cmd, n, 1, 0, 0);
    vkCmdEndRendering(cmd);
    s_Lines.clear();
}
} // namespace EditorOverlay

// ---------------------------------------------------------------------------
// SPIKE 1 (editor-on-Vulkan): draw a single .ogf with an orbit camera and NO
// level loaded — the "render a model outside the game" probe. Gated by
// -vk_spike [model_name]. Additive/experimental; see EDITOR_ON_VULKAN_ROADMAP.md
// in the SDK repo. Delete once the real editor host lands.
// ---------------------------------------------------------------------------
namespace Spike {
static bool           s_checked   = false;
static bool           s_enabled   = false;
static string_path    s_model     = { 0 };
static IRenderVisual* s_visual    = nullptr;
static bool           s_triedLoad = false;
static int            s_frame     = 0;
static bool           s_shot      = false;

bool Enabled()
{
    if (!s_checked) {
        s_checked = true;
        const char* p = Core.Params ? strstr(Core.Params, "-vk_spike") : nullptr;
        if (p) {
            s_enabled = true;
            p += xr_strlen("-vk_spike");
            while (*p == ' ') ++p;
            if (*p && *p != '-') {
                int i = 0;
                while (*p && *p != ' ' && i < int(sizeof(s_model)) - 1) s_model[i++] = *p++;
                s_model[i] = 0;
            } else {
                // Placeholder — override with `-vk_spike <path\to\model>` (no .ogf ext).
                xr_strcpy(s_model, "dynamics\\device\\device_torch\\device_torch");
            }
            Msg("[VK][spike] enabled, model='%s'", s_model);
        }
    }
    return s_enabled;
}

void Frame()
{
    if (!s_triedLoad) {
        s_triedLoad = true;
        s_visual = RImplementation.model_Create(s_model, nullptr);
        if (!s_visual)
            Msg("![VK][spike] model_Create failed for '%s'", s_model);
        else
            Msg("[VK][spike] model loaded OK: '%s'", s_model);
    }
    if (!s_visual) return;

    // The main menu draws an opaque fullscreen background over the world pass, so
    // nothing shows there — the box is only visible once a level is loaded.
    if (!g_pGameLevel) return;

    // Drop the box 3 m in front of the player's camera (no camera hijack): it
    // renders on top of the real scene via the normal dynamic path, proving
    // add_Visual + VK rasterization of a model outside the game's own use.
    Fvector pos; pos.mad(Device.vCameraPosition, Device.vCameraDirection, 3.0f);
    Fmatrix world; world.identity(); world.c.set(pos);
    RImplementation.add_Visual(0u, nullptr, s_visual, world);

    // SPIKE 2: editor overlay — yellow selection box around the model + a green
    // ground grid under it, drawn via the EditorOverlay line pass. Proves the
    // grid/gizmo/selection primitives an editor needs are feasible on Vulkan.
    {
        auto* rv = static_cast<vkRender_Visual*>(s_visual);
        const float r  = (rv && rv->vis.sphere.R > 0.1f) ? rv->vis.sphere.R : 0.5f;
        EditorOverlay::AddBoxAA(pos, r, EditorOverlay::rgba(255, 220, 40, 255));
        const float gy = pos.y - r;
        const int   GN = 12;
        const float GS = 0.5f;
        const u32   gc = EditorOverlay::rgba(70, 200, 100, 170);
        for (int i = -GN; i <= GN; ++i)
        {
            Fvector a, b;
            a.set(pos.x + i * GS, gy, pos.z - GN * GS); b.set(pos.x + i * GS, gy, pos.z + GN * GS);
            EditorOverlay::AddLine(a, b, gc);
            a.set(pos.x - GN * GS, gy, pos.z + i * GS); b.set(pos.x + GN * GS, gy, pos.z + i * GS);
            EditorOverlay::AddLine(a, b, gc);
        }
    }

    // One-shot screenshot ~2s after the level is up (swapchain capture in End(),
    // so it's independent of window focus/occlusion). Writes ss_*.tga.
    if (!s_shot && ++s_frame >= 120) {
        s_shot = true;
        RImplementation.Screenshot(IRender_interface::SM_NORMAL, nullptr);
        Msg("[VK][spike] in-level screenshot queued");
    }
}
} // namespace Spike

// ---------------------------------------------------------------------------
// PHASE B — CHUNK 1 (editor-on-Vulkan): a no-menu, no-level editor viewport.
// -vk_editor boots straight into a clean orbit-camera view of ONE model on a
// grid + sky, with NO game menu — the seed of the editor host. A custom
// pureFrame/pureRender sink drives the frame instead of the menu:
//   OnFrame()  (in FrameMove, BEFORE OnCameraUpdated): orbit camera from the
//     mouse → Device.mView/mProject/camera vectors. OnCameraUpdated then builds
//     mFullTransform + SetCacheXform from exactly those.
//   OnRender() (in seqRender, between Begin/End): Calculate (clears the dynamic
//     list; no-ops with no level) → inject the model via add_Visual + build the
//     grid via EditorOverlay → Render (sky + model + overlay).
// The sink self-registers (VKEditor::Tick from CRender::Begin) once
// g_pGamePersistent exists; the menu is suppressed game-side by reading
// -vk_editor. Additive/experimental; see EDITOR_ON_VULKAN_ROADMAP.md §13.
// ---------------------------------------------------------------------------
namespace VKEditor {

static bool        s_checked = false;
static bool        s_enabled = false;
static string_path s_model   = { 0 };

bool Enabled()
{
    if (!s_checked) {
        s_checked = true;
        const char* p = Core.Params ? strstr(Core.Params, "-vk_editor") : nullptr;
        if (p) {
            s_enabled = true;
            p += xr_strlen("-vk_editor");
            while (*p == ' ') ++p;
            if (*p && *p != '-') {
                int i = 0;
                while (*p && *p != ' ' && i < int(sizeof(s_model)) - 1) s_model[i++] = *p++;
                s_model[i] = 0;
            } else {
                // Valid rigid device that exists in Gunslinger's gamedata.db_base_meshes
                // (verified via the .db extractor). The old device_torch path did not.
                xr_strcpy(s_model, "dynamics\\devices\\dev_pda\\dev_pda");
            }
            Msg("[VK][editor] enabled, model='%s'", s_model);
        }
    }
    return s_enabled;
}

// ---- A2.4: host-driven scene ------------------------------------------------
// When the SDK hosts us it owns the camera and the object list; the built-in orbit
// camera + demo model are only the fallback for standalone `-vk_editor`. The host
// calls these from its frame thread, i.e. the same thread as Ed_RenderFrame.
static bool s_hostCam = false;
static Fvector s_camPos, s_camDir, s_camUp;
static float s_camFov = 67.f, s_camZN = 0.1f, s_camZF = 1000.f;

struct HostModel
{
    IRenderVisual* visual;
    Fmatrix xform;
};
static xr_vector<HostModel> s_hostModels;

// Handles the host wants highlighted. Indices into s_hostModels; the host re-sends the
// whole set on every selection change, so we never have to track its scene.
static xr_vector<int> s_hostSelection;

void HostSetCamera(const Fvector& pos, const Fvector& dir, const Fvector& up, float fov_deg, float zn, float zf)
{
    s_hostCam = true;
    s_camPos = pos;
    s_camDir = dir;
    s_camDir.normalize_safe();
    s_camUp = up;
    s_camUp.normalize_safe();
    s_camFov = fov_deg;
    s_camZN = zn;
    s_camZF = zf;
}

int HostAddModel(const char* name, const Fmatrix& xf)
{
    if (!name || !name[0])
        return -1;

    IRenderVisual* v = RImplementation.model_Create(name, nullptr);
    if (!v)
    {
        Msg("![VK][editor] host model_Create failed: '%s'", name);
        return -1;
    }

    s_hostModels.push_back({v, xf});
    Msg("[VK][editor] host model added [%d]: '%s'", int(s_hostModels.size()) - 1, name);
    return int(s_hostModels.size()) - 1;
}

// ---- A2.5: static scene geometry from a HOST-EXPORTED .ogf -------------------
// Ed_AddModel's by-name lookup cannot serve an editor scene: our gamedata holds only
// dynamic visuals and BAKED per-level geometry, never the individual static props a
// level source is built from (0 of Cordon's 434 visuals resolve). So the host exports
// each source object to a real .ogf and hands us the absolute path; we load it through
// the ordinary ModelPool, which keys on `logical_name` — so a level's thousands of
// instances cost only as many file loads as it has distinct visuals.
static xr_vector<shared_str> s_hostLoaded; // logical names whose base model is in the pool

int HostAddModelFile(const char* ogf_path, const char* logical_name, const Fmatrix& xf)
{
    if (!ogf_path || !ogf_path[0] || !logical_name || !logical_name[0])
        return -1;

    bool haveBase = false;
    for (const shared_str& s : s_hostLoaded)
        if (0 == xr_strcmp(s.c_str(), logical_name))
        {
            haveBase = true;
            break;
        }

    IRenderVisual* v = nullptr;
    if (haveBase)
    {
        // Base model already registered — Create duplicates it and never touches the FS.
        v = RImplementation.model_Create(logical_name, nullptr);
    }
    else
    {
        // Feed the reader in explicitly: the path is the host's, outside our gamedata,
        // so it is not in the VFS registry and a by-name lookup would miss it.
        IReader* data = FS.r_open(ogf_path);
        if (!data)
        {
            Msg("![VK][editor] host .ogf not readable: '%s'", ogf_path);
            return -1;
        }
        v = RImplementation.model_Create(logical_name, data);
        FS.r_close(data);
        if (v)
            s_hostLoaded.push_back(shared_str(logical_name));
    }

    if (!v)
    {
        Msg("![VK][editor] host model load failed: '%s' (%s)", logical_name, ogf_path);
        return -1;
    }

    s_hostModels.push_back({v, xf});
    // Adding an object is a STRUCTURAL change to the instanced caster set (new
    // meta entry, possibly a new draw group). Moving one is not — see below.
    VK::InstanceGPU::Invalidate();
    return int(s_hostModels.size()) - 1;
}

int HostModelCount() { return int(s_hostModels.size()); }

void HostSetModelXform(int id, const Fmatrix& xf)
{
    if (id >= 0 && id < int(s_hostModels.size()))
    {
        s_hostModels[id].xform = xf;
        // The whole point of the instanced path: a moved object re-uploads the
        // transform SSBO and nothing else — no re-bake, no realloc, no device
        // wait. Cull bounds are stored in MODEL space precisely so this holds.
        VK::InstanceGPU::TouchTransforms();
    }
}

void HostClearScene()
{
    for (auto& m : s_hostModels)
        RImplementation.model_Delete(m.visual, TRUE);
    s_hostModels.clear();

    // Deleting with discard can drop the registered base models too, so forget what we
    // cached: the next push re-reads the files rather than duplicating a freed base.
    s_hostLoaded.clear();

    // Handles are array indices, so everything the host selected is now meaningless.
    s_hostSelection.clear();

    // The instanced caster set points at meshes we just deleted — it MUST be
    // rebuilt before anything culls or draws it again.
    VK::InstanceGPU::Invalidate();
}

// ---- picking: give the host a ray, let it do the hit test --------------------
// We only unproject. The host's scene knows mesh-level geometry, object classes and
// selection semantics that we never see, so it owns the actual pick; feeding it a ray
// built from OUR matrices is what keeps the pick consistent with the picture.
bool HostScreenRay(int x, int y, Fvector& out_start, Fvector& out_dir)
{
    if (!Device.dwWidth || !Device.dwHeight)
        return false;

    // Client pixels -> NDC. Y flips: window coords run down, clip space runs up.
    const float nx = (2.f * float(x) / float(Device.dwWidth)) - 1.f;
    const float ny = 1.f - (2.f * float(y) / float(Device.dwHeight));

    // Unproject that pixel on the near and far plane. mInvFullTransform is rebuilt every
    // frame in CRenderDevice::OnCameraUpdated from the matrices the frame was drawn with.
    // Clip Z runs 0..1 here, and the renderer draws with a negative-height viewport, so
    // clip space stays Y-up and the flip above is the only one needed.
    Fvector4 src, pNear, pFar;
    src.set(nx, ny, 0.f, 1.f);
    Device.mInvFullTransform.transform(pNear, src);
    src.set(nx, ny, 1.f, 1.f);
    Device.mInvFullTransform.transform(pFar, src);
    if (fis_zero(pNear.w) || fis_zero(pFar.w))
        return false;

    const Fvector a = {pNear.x / pNear.w, pNear.y / pNear.w, pNear.z / pNear.w};
    const Fvector b = {pFar.x / pFar.w, pFar.y / pFar.w, pFar.z / pFar.w};

    out_start = a;
    out_dir.sub(b, a);
    if (fis_zero(out_dir.magnitude()))
        return false;

    out_dir.normalize();
    return true;
}

void HostSetSelection(const int* ids, int count)
{
    s_hostSelection.clear();
    if (!ids || count <= 0)
        return;

    s_hostSelection.reserve(count);
    for (int i = 0; i < count; ++i)
        if (ids[i] >= 0 && ids[i] < int(s_hostModels.size()))
            s_hostSelection.push_back(ids[i]);
}

struct CEditorViewport : public pureFrame, public pureRender
{
    // Orbit camera around `target`: yaw about Y, pitch = elevation (+ = above,
    // looking down), dist = radius. Default is a 3/4 view from ABOVE the grid.
    float   yaw    = PI_DIV_4;
    float   pitch  = 0.55f;
    float   dist   = 6.0f;
    Fvector target = { 0.f, 0.f, 0.f };

    // Polled Win32 input (no WndProc hook — same approach as the Spike 3 ImGui).
    bool  lDrag = false, rDrag = false;
    POINT lastPt = { 0, 0 };

    IRenderVisual* visual  = nullptr;
    int            loadTry = 0;
    bool           framed  = false;
    int            frame   = 0;
    bool           shot    = false;

    bool dragValid = false;

    void PollInput()
    {
        POINT pt;
        if (!GetCursorPos(&pt)) return;
        const bool lb = (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;
        const bool rb = (GetAsyncKeyState(VK_RBUTTON) & 0x8000) != 0;

        if ((lb || rb) && !(lDrag || rDrag)) {
            lastPt = pt; // drag start — seed the delta
            // Only start orbiting if the drag BEGAN over our viewport window AND not over
            // an editor ImGui panel (the Log window) — otherwise dragging on the host's
            // toolbars/panels or the Log would move the camera. Once a drag starts valid,
            // it keeps control even if the cursor wanders out (natural).
            dragValid = !VK::ImGuiVK::EditorWantsMouse();
            if (dragValid && Device.m_hWnd) {
                POINT c = pt; ScreenToClient(Device.m_hWnd, &c);
                RECT rc; GetClientRect(Device.m_hWnd, &rc);
                dragValid = (c.x >= 0 && c.y >= 0 && c.x < rc.right && c.y < rc.bottom);
            }
        }
        if (!dragValid) { lDrag = lb; rDrag = rb; return; }

        const int dx = pt.x - lastPt.x;
        const int dy = pt.y - lastPt.y;
        lastPt = pt;

        if (lb) { yaw -= dx * 0.006f; pitch -= dy * 0.006f; } // LMB drag → orbit
        if (rb) { dist *= (1.f + dy * 0.005f); }              // RMB drag up/down → dolly
        lDrag = lb; rDrag = rb;

        clamp(pitch, -1.45f, 1.45f);
        clamp(dist, 0.3f, 200.f);
    }

    void UpdateCamera()
    {
        // Host-driven (SDK): adopt its camera verbatim. Aspect still comes from OUR
        // swapchain — the host's viewport panel has its own size, not the host window's.
        if (s_hostCam)
        {
            Fvector right;
            right.crossproduct(s_camUp, s_camDir);
            right.normalize_safe();
            Fvector up;
            up.crossproduct(s_camDir, right);
            up.normalize_safe();

            Device.vCameraPosition = s_camPos;
            Device.vCameraDirection = s_camDir;
            Device.vCameraTop = up;
            Device.vCameraRight = right;

            Device.fFOV = s_camFov;
            Device.fASPECT = Device.dwHeight ? float(Device.dwWidth) / float(Device.dwHeight) : (16.f / 9.f);
            Device.mView.build_camera_dir(Device.vCameraPosition, Device.vCameraDirection, Device.vCameraTop);
            Device.mProject.build_projection(deg2rad(Device.fFOV), Device.fASPECT, s_camZN, s_camZF);
            return;
        }

        // Camera sits on a sphere around `target`; look back at it.
        Fvector off;
        off.set(_cos(pitch) * _sin(yaw), _sin(pitch), _cos(pitch) * _cos(yaw));
        Fvector pos; pos.mad(target, off, dist);            // pos = target + off*dist
        Fvector dir; dir.sub(target, pos); dir.normalize(); // cam → target

        Fvector up; up.set(0.f, 1.f, 0.f);
        Fvector right; right.crossproduct(up, dir); right.normalize();
        up.crossproduct(dir, right); up.normalize();

        Device.vCameraPosition  = pos;
        Device.vCameraDirection = dir;
        Device.vCameraTop       = up;
        Device.vCameraRight     = right;

        Device.fFOV    = 67.f;
        Device.fASPECT = Device.dwHeight ? float(Device.dwWidth) / float(Device.dwHeight) : (16.f / 9.f);
        Device.mView.build_camera_dir(Device.vCameraPosition, Device.vCameraDirection, Device.vCameraTop);
        Device.mProject.build_projection(deg2rad(Device.fFOV), Device.fASPECT, 0.1f, 1000.f);
        // mFullTransform / SetCacheXform are rebuilt by CRenderDevice::OnCameraUpdated,
        // which runs right after FrameMove using the matrices we just set.
    }

    // Keep a normal arrow over the viewport. Embedded, several parties fight over the
    // OS cursor (the host's ImGui, DirectInput acquires on focus, the game's own cursor
    // suppression) and any of them can blank it on a click/focus change. Rather than
    // chase each one, re-assert the cursor every frame while the pointer is over our
    // window — self-correcting: whoever hid it, we bring it back next frame.
    void EnforceCursor()
    {
        extern bool g_ed_embedded;
        if (!g_ed_embedded || !Device.m_hWnd) return;

        POINT p;
        if (!GetCursorPos(&p)) return;
        POINT c = p; ScreenToClient(Device.m_hWnd, &c);
        RECT rc; GetClientRect(Device.m_hWnd, &rc);
        if (c.x < 0 || c.y < 0 || c.x >= rc.right || c.y >= rc.bottom)
            return; // pointer is over the host's UI — leave its cursor alone

        SetCursor(LoadCursor(nullptr, IDC_ARROW));
        CURSORINFO ci{ sizeof(ci) };
        if (GetCursorInfo(&ci) && !(ci.flags & CURSOR_SHOWING))
            while (ShowCursor(TRUE) < 0) // bring the display count back to >= 0
                ;
    }

    // pureFrame — set the editor camera BEFORE OnCameraUpdated consumes it.
    void OnFrame() override
    {
        if (!s_hostCam) // the host owns the camera (and the mouse) when embedded
            PollInput();
        UpdateCamera();
        EnforceCursor();
    }

    // Grid + world-origin axis, drawn in BOTH the host-driven and the demo path.
    void DrawEditorGizmos()
    {
        // World-origin axis tripod (X/Y/Z = R/G/B), 0.5 m — the SDK's coordinate marker.
        EditorOverlay::AddAxis(Fvector{ 0.f, 0.f, 0.f }, 0.5f);

        // Ground grid — only for an EMPTY viewport. It is drawn POST-TONEMAP with no depth
        // test (that is what keeps it crisp), so over a loaded level it paints across the
        // whole picture instead of lying on the ground. Once the host has pushed a scene,
        // the level itself is the ground reference and the grid is just noise.
        if (!s_hostModels.empty())
            return;

        // Matches the original SDK editor (D3DUtils UpdateGrid): 1 m cells, minor lines
        // m_ColorGrid 0x909090, major lines every SUB cells m_ColorGridTh 0xb4b4b4.
        const int GN = 25; const float GS = 1.0f; const int SUB = 5;
        const u32 gc  = EditorOverlay::rgba(144, 144, 144, 255); // minor (0x909090)
        const u32 gcM = EditorOverlay::rgba(180, 180, 180, 255); // major (0xb4b4b4)
        for (int i = -GN; i <= GN; ++i) {
            const u32 col = (i % SUB == 0) ? gcM : gc;
            Fvector a, b;
            a.set(i * GS, 0.f, -GN * GS); b.set(i * GS, 0.f, GN * GS); EditorOverlay::AddLine(a, b, col);
            a.set(-GN * GS, 0.f, i * GS); b.set(GN * GS, 0.f, i * GS); EditorOverlay::AddLine(a, b, col);
        }
    }

    // Wireframe box around every object the host has selected. Drawn post-tonemap with no
    // depth test, so a selection stays visible even behind geometry.
    //
    // The box is the visual's MODEL-space AABB pushed through the object's transform, i.e.
    // an ORIENTED box — the same thing the SDK draws (set_xform_world + DrawSelectionBox).
    // Collapsing it to a world AABB instead would visibly balloon around anything rotated.
    void DrawSelection()
    {
        const u32 col = EditorOverlay::rgba(255, 220, 40, 255); // SDK selection yellow

        for (int id : s_hostSelection)
        {
            if (id < 0 || id >= int(s_hostModels.size()))
                continue;

            const HostModel& m = s_hostModels[id];
            auto* rv = static_cast<vkRender_Visual*>(m.visual);
            if (!rv || !rv->vis.box.is_valid())
                continue;

            // Corner i takes its x/y/z from min or max per bit 0/1/2.
            const Fbox& b = rv->vis.box;
            Fvector c[8];
            for (int i = 0; i < 8; ++i)
            {
                Fvector p;
                p.set((i & 1) ? b.max.x : b.min.x, (i & 2) ? b.max.y : b.min.y, (i & 4) ? b.max.z : b.min.z);
                m.xform.transform_tiny(c[i], p);
            }

            static const int edges[12][2] = {{0, 1}, {1, 3}, {3, 2}, {2, 0},  // z = min face
                                             {4, 5}, {5, 7}, {7, 6}, {6, 4},  // z = max face
                                             {0, 4}, {1, 5}, {2, 6}, {3, 7}}; // verticals
            for (const auto& e : edges)
                EditorOverlay::AddLine(c[e[0]], c[e[1]], col);
        }
    }

    // pureRender — drive the render (replaces the menu as the frame's render sink).
    void OnRender() override
    {
        // ---- Host-driven scene (SDK): draw ITS objects, skip the demo model ----
        if (!s_hostModels.empty()) {
            RImplementation.Calculate();
            for (auto& m : s_hostModels)
                RImplementation.add_Visual(0u, nullptr, m.visual, m.xform);
            DrawEditorGizmos();
            DrawSelection();
            RImplementation.Render();
            return;
        }

        if (!visual && loadTry < 240 && (loadTry++ % 30) == 0) { // lazy load + retry
            visual = RImplementation.model_Create(s_model, nullptr);
            Msg(visual ? "[VK][editor] model loaded: '%s'" : "![VK][editor] model_Create failed: '%s'", s_model);
        }

        // Frame the loaded model once: aim the orbit at the CENTRE of its real
        // bounding box (a flat/offset-pivot model isn't centred on its sphere), and
        // pull back to fit the box diagonal — like the SDK's "focus on selection".
        if (visual && !framed) {
            framed = true;
            auto* rv = static_cast<vkRender_Visual*>(visual);
            if (rv && rv->vis.box.is_valid()) {
                Fvector bc; rv->vis.box.getcenter(bc);
                Fvector bs; rv->vis.box.getsize(bs);
                const float diag = bs.magnitude();          // full box diagonal
                if (diag > 0.01f) { target = bc; dist = diag * 1.6f; }
            } else if (rv && rv->vis.sphere.R > 0.01f) {
                target = rv->vis.sphere.P; dist = rv->vis.sphere.R * 3.f;
            }
        }

        RImplementation.Calculate(); // clears the dynamic list; no-ops without a level

        if (visual) {
            Fmatrix world; world.identity();
            RImplementation.add_Visual(0u, nullptr, visual, world);

            // Selection: a TIGHT wireframe AABB hugging the model's real bounding box
            // (world==identity), like the SDK LevelEditor draws around a selected
            // object — not a loose sphere-cube. SDK selection tint = warm yellow.
            auto* rv = static_cast<vkRender_Visual*>(visual);
            if (rv && rv->vis.box.is_valid())
                EditorOverlay::AddBox(rv->vis.box.min, rv->vis.box.max, EditorOverlay::rgba(255, 220, 40, 255));
        }

        DrawEditorGizmos();

        RImplementation.Render();

        if (!shot && ++frame >= 180) { // one-shot screenshot ~2-3s in (warmed up), for verification
            shot = true;
            RImplementation.Screenshot(IRender_interface::SM_NORMAL, nullptr);
            Msg("[VK][editor] viewport screenshot queued");
        }
    }
};

static CEditorViewport* s_vp     = nullptr;
static bool             s_active = false;

void Activate()
{
    if (s_active) return;
    s_active = true;
    s_vp = xr_new<CEditorViewport>();
    Device.seqFrame.Add(s_vp);     // camera update, inside FrameMove
    Device.seqRender.Add(s_vp, 5); // render sink, same priority slot the menu used
    Msg("[VK][editor] viewport sink registered (frame+render)");
}

bool Active() { return s_active; }

// Clear scene color + depth for the no-level viewport. The normal path relies on
// Pass_World's CLEAR loadOp to establish the targets; with no level Pass_World
// returns early, so Sky/overlay would LOAD an uninitialised target (black frame,
// garbage depth → the grid's depth test fails). Runs FIRST, only in editor mode.
void ClearTargets(VK::FrameContext& ctx)
{
    if (!s_active || ctx.cmd == VK_NULL_HANDLE || ctx.colorView == VK_NULL_HANDLE) return;

    // 0.044 → tonemaps to the SDK grey 0x555555 (85,85,85) once fog is off (see
    // the vk_pass_tonemap editor gate). A null depthView leaves the scope
    // depth-less, exactly as before.
    VK::RenderingBuilder(ctx.extent)
        .ColorClear(ctx.colorView, VkClearColorValue{ { 0.044f, 0.044f, 0.044f, 1.0f } })
        .Depth(ctx.depthView, VK_ATTACHMENT_LOAD_OP_CLEAR)
        .Begin(ctx.cmd);
    vkCmdEndRendering(ctx.cmd);
}

// Called every frame from CRender::Begin. Self-registers the sink once the game
// persistent (environment/sky owner) exists — Pass_Sky null-guards CurrentEnv,
// so activating a hair early only costs a dark sky for a frame or two.
void Tick()
{
    if (!Enabled() || s_active) return;
    if (!g_pGamePersistent)     return;
    Activate();
}

} // namespace VKEditor

// ----- Frame ----------------------------------------------------------------
void CRender::Calculate()
{
    VK::Prof::CpuPhaseScope _cpu(VK::Prof::CPU_CALC);   // [VK CPUphase] scene collect
    // Dynamic-visual collection. R4 does this inside Calculate via the dsgraph
    // (portal + HOM culling); here it's a minimal frustum query with no culling.
    // Clear the per-frame list, then drive each visible renderable's
    // renderable_Render(), which calls back into add_Visual({visual, xform}).
    VK::g_DynamicVisuals.clear();
    VK::g_HudVisuals.clear();

    // The late-glass and water lists live ACROSS the frame that builds them (the
    // particles pass re-draws glass into the distortion RT), so somebody has to
    // drop them at the START of the next one. That used to be Pass_World -- which
    // returns early when no level is loaded, while Pass_WorldGlass has no such
    // guard. Result: after a level unload the main menu flushed a list of RAW
    // pointers to deleted visuals, every frame, and the fault was swallowed by
    // CMainMenu's catch(...) (xrGame is built without /EH). Calculate() runs on
    // every frame with or without a level, so the lifetime is now owned here.
    VK::g_RenderQueue.ClearGlass();
    VK::g_RenderQueue.ClearWater();

    // SPIKE 1 (editor-on-Vulkan): with -vk_spike, inject one .ogf. In-level it is
    // placed in front of the camera; then normal collection runs so the world
    // renders too. At the menu the existing !g_pGameLevel guard below returns.
    if (Spike::Enabled()) Spike::Frame();

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
    VK::Prof::CpuPhaseScope _cpu(VK::Prof::CPU_RECORD);   // [VK CPUphase] pass recording

    // Register the scene passes once, in draw order. Adding a pass later (lights,
    // SSAO, bloom) is a RegisterPass call, not an edit here. ExecutePasses owns
    // the inter-pass barriers.
    //   World → Grass (depth-tests world geometry, writes own depth) → Sky (fills
    //   uncleared z==1 pixels behind the world). Grass skips if Details didn't
    //   load (no level.details) or compute is disabled.
    static bool s_registered = false;
    if (!s_registered)
    {
        // PHASE B (editor-on-Vulkan): with -vk_editor there is no level, so Pass_World
        // (the normal owner of the color+depth CLEAR) returns early. Clear here FIRST so
        // Sky + the editor overlay draw onto a defined target with depth=1.0. No-ops in
        // game mode (guarded by VKEditor::Active()).
        VK::RegisterPass("EditorClear", [](VK::FrameContext& c) { VKEditor::ClearTargets(c); });
        // COMPUTE PRE-SKINNING (r_preskin) before any consumer: skins every
        // visible skeleton leaf ONCE into a shared world-space pool, so the
        // shadow/prepass/colour/VSM passes below draw pre-transformed geometry
        // instead of re-running the bone blend in each of their vertex shaders.
        // Must sit outside a render pass (it dispatches compute) — hence its own
        // registered pass rather than a call inside Pass_SunShadow.
        VK::RegisterPass("PreSkin", [](VK::FrameContext& c) { VK::Skinned_PreSkin(c.cmd); });
        // Sun shadow caster FIRST: renders static world depth from the sun POV into
        // the shadow map (own depth target), leaves it SHADER_READ for the receivers.
        VK::RegisterPass("SunShadow", [](VK::FrameContext& c) {
            VK::Pass_SunShadow(c);
            // Async compute (r_async): the fog inject records onto the COMPUTE queue,
            // where sampling these cascades would race the graphics queue rewriting
            // them next frame. Take the downscaled snapshot the inject reads instead,
            // here — the one point where the cascades are complete and still in
            // SHADER_READ. No-op with r_async off.
            VK::Vol::SnapshotShadows(c.cmd);
        });
        VK::RegisterPass("World", [](VK::FrameContext& c) { VK::Pass_World(c); });
        // PHASE B (editor-on-Vulkan, chunk 2): with -vk_editor Pass_World returns early
        // (no level) so add_Visual'd models aren't drawn. Draw them here — into the same
        // HDR SceneColor + depth EditorClear seeded — so the lit model composites on the
        // grey backdrop through tonemap. No-op in game mode (Pass_World already drew them).
        VK::RegisterPass("EditorDynamics", [](VK::FrameContext& c) { if (VKEditor::Active()) VK::Pass_EditorDynamics(c); });
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
        // An EMPTY editor viewport uses a flat grey clear, not the game sky, so the SDK's
        // backdrop shows. Once the host has pushed a scene the sky belongs there: the
        // editor's whole point is to look like the game, and the sky pass also prepares
        // the weather cubes that the hemisphere ambient samples — without it every
        // surface gets the same flat grey constant regardless of its normal.
        // Same "empty viewport vs real scene" split as the grid and auto-exposure.
        VK::RegisterPass("Sky",   [](VK::FrameContext& c) { if (!VKEditor::Active() || VKEditor::HostModelCount() > 0) VK::Pass_Sky(c); });
        // SPIKE 2 (editor-on-Vulkan): editor overlay — depth-tested world-space
        // lines (grid / gizmo / selection box) drawn on top of the opaque scene.
        // In -vk_editor the grid is drawn POST-TONEMAP (crisp, no bloom) — see the
        // ExecutePostTonemap call after Tonemap; skip the HDR pass so the queued lines
        // survive. The Spike (-vk_spike, in-level) still uses this HDR overlay path.
        VK::RegisterPass("EditorOverlay", [](VK::FrameContext& c) { if (!VKEditor::Active()) EditorOverlay::Execute(c); });
        // (Sun shafts used to be registered here — a fullscreen additive raymarch of
        // the FAR sun map. Removed 2026-07-24: under VSM that map is deliberately left
        // cleared = "fully lit", so the pass painted warm rays THROUGH walls and had to
        // be gated off entirely; and god rays are now the froxel volumetrics' job, where
        // they come out of a real medium with a phase function instead of a screen
        // effect layered on top. See [[vulkan-vsm-light-leak-fix]] / the fog V-1 layers.)
        // Per-light volumetric cones: real raymarched beams for the volumetric
        // spots (headlights/searchlights/pole lamps) — replaces the R4
        // lightplanes texture-sheet fakes. Additive over the lit scene.
        // WATER bodies — after the whole opaque world AND the sky (both are the
        // "bottom" the surface blends over: a pond reflects the sky and shows the
        // riverbed through itself), before glass/particles/rain, which composite
        // on top of it. See vk_pass_water.h.
        VK::RegisterPass("Water", [](VK::FrameContext& c) { VK::Pass_Water(c); });
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

    // A frame that failed Begin (fence timeout / device lost / no swapchain image)
    // returned early WITHOUT populating g_FrameCtx — it still points at the PREVIOUS
    // frame's already-submitted command buffer, so recording the scene into it faults
    // in the driver. Skip this frame's render entirely (End() guards the same way at
    // its top). This turns any transient stall — e.g. the DLSS-G present pacer briefly
    // skipping a present — into a dropped frame instead of a hard crash.
    if (!g_FrameInFlight.valid) return;

    VK::ExecutePasses(g_FrameCtx);

    // Async frame-split (r_async): the scene is now fully drawn into the HDR target
    // (no swapchain touched yet). Submit it as its OWN graphics batch and continue
    // tonemap+UI in a second segment, so a compute pass can overlap the scene batch
    // (wired in increment 2b). The scene batch carries NO imageAvailable/renderFinished/
    // fence (the post segment does) and does NOT wait the async timeline (compute is
    // meant to run CONCURRENTLY with it). Uploads are flushed first so the scene draws
    // are GPU-ordered after their data. r_async 0 = one batch (the else path below).
    if (VK::Async::Available()) {
        CommandManager.FlushUploads();
        CommandManager.End(g_FrameCtx.cmd);
        CommandManager.Submit(g_FrameCtx.cmd, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE,
                              /*waitAsyncCompute=*/false);
        VkCommandBuffer post = CommandManager.BeginSecondSegment();
        if (post != VK_NULL_HANDLE) {
            g_FrameInFlight.cmd = post;   // End() submits this segment (imageAvailable/renderFinished/fence)
            g_FrameCtx.cmd      = post;   // tonemap + MV-debug record here
            g_VkUI_FrameCmd     = post;   // deferred UI (End) records here too
        }
    }

    // Resolve the HDR scene target to the swapchain: exposure + Reinhard-white
    // tonemap (R4 combine_tonemap). After this the swapchain holds the final LDR
    // image in COLOR_ATTACHMENT; the UI pass (End) draws on top, un-tonemapped.
    // Profiled as its own zone (hosts the SSR puddle march — a rain cost), then
    // FrameEnd closes the profiler frame.
    // DLSS Super Resolution (increment 2c): temporal-upscale the render-res HDR scene
    // to display resolution BEFORE tonemap, using color + depth + motion vectors (all
    // at render res). The upscaled DlssOutput (display res) is left in SHADER_READ and
    // the tonemap samples it as its base colour (resolved-colour binding); SceneColor
    // stays at render res and still feeds the avg-luminance mip chain + bloom. DLAA
    // (r_dlss_quality 0) takes the same path with render==display. Gated r_dlss
    // (default 0) → the normal path is untouched.
    if (VK::Dlss::Enabled() && VK::MotionVec::Enabled())
    {
        const u32   idx  = g_FrameCtx.imageIndex;
        const VkExtent2D rext = g_FrameCtx.extent;        // scene render res
        const VkExtent2D dext = g_FrameCtx.displayExtent; // upscale target
        VkCommandBuffer  cmd = g_FrameCtx.cmd;
        VkImage     scImg  = VK::SceneColor::GetImage(idx);
        VkImageView scView = VK::SceneColor::GetView(idx);
        VkImageView mvView = VK::MotionVec::GetResultView();
        VkImage     mvImg  = VK::MotionVec::GetResultImage();
        // SL route: sl.dlss owns the NGX feature — the raw EnsureFeature would build
        // a SECOND tensor set (the mid-game-enable OOM crash, 22:51 log).
        const bool useSL = ps_r_dlss_sl && VK::SL::SuperResAvailable();
        // Streamline can only be brought up BEFORE the Vulkan device exists, so it is
        // decided at startup from user.ltx (HW_Vulkan.cpp Step 0). Flipping r_dlss_sl
        // mid-run therefore silently does nothing — say so instead of pretending.
        if (ps_r_dlss_sl && !VK::SL::Inited()) {
            static bool s_saidNoSL = false;
            if (!s_saidNoSL) {
                s_saidNoSL = true;
                Msg("![VK DLSS] r_dlss_sl 1 but Streamline was NOT initialised this run — it must init "
                    "before the Vulkan device is created. Put `r_dlss_sl 1` in user.ltx (or pass "
                    "-force_sl) and RESTART; raw NGX stays active meanwhile.");
            }
        }
        if (scImg != VK_NULL_HANDLE && mvImg != VK_NULL_HANDLE &&
            (useSL ? VK::Dlss::EnsureFeatureSL(rext.width, rext.height, dext.width, dext.height)
                   : VK::Dlss::EnsureFeature(cmd, rext.width, rext.height, dext.width, dext.height,
                                             VK::SceneColor::Generation())))
        {
            VK::Dlss::EnsureOutput(dext.width, dext.height);
            const int zd = VK::Prof::ZoneBegin(cmd, "DLSS");

            // Discard the temporal history on a discontinuity the motion vectors
            // cannot describe (eval-chain gap / level load / teleport). Called ONCE
            // per evaluated frame — it rolls the detector's own camera+frame state.
            const bool histReset = VK::Dlss::TakeHistoryReset();

            // INPUT LAYOUTS. The two routes want different ones, so pick here and
            // restore symmetrically after the eval:
            //   raw NGX  — GENERAL (what NGX requires; go straight there from the
            //              attachment layout, 2 transitions instead of 4);
            //   sl.dlss  — SHADER_READ, because EvaluateSR TAGS its resources with
            //              that layout (vk_sl.cpp MakeRes) and SL trusts the tag.
            // MV already sits in SHADER_READ from its producing pass.
            const VkImageLayout inLayout = useSL ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                                 : VK_IMAGE_LAYOUT_GENERAL;
            VK::ImageBarrier(cmd, scImg, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, inLayout);
            VK::ImageBarrier(cmd, Swapchain.m_DepthImage, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                         inLayout, VK_IMAGE_ASPECT_DEPTH_BIT);
            if (!useSL)
                VK::ImageBarrier(cmd, mvImg, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);

            // Stage B (r_dlss_sl): route the SR evaluate through Streamline's sl.dlss
            // instead of raw NGX — same inputs/output/barriers, prerequisite for FG.
            bool evalOk = true;
            static u32 s_slRescueFrame = 0;   // frame the VRAM rescue started (0 = none)
            // While a rescue is in progress, probe sl.dlss only every 20th frame —
            // hammering its failed NGX creation every frame is what made Streamline
            // throw out of its vkQueuePresentKHR hook before (the 22:51 crash).
            const bool slProbe = (s_slRescueFrame == 0) ||
                                 ((Device.dwFrame - s_slRescueFrame) % 20u == 0u);
            if (useSL) {
                // Stage C: when Frame Generation is on, (re)configure DLSS-G for this
                // frame BEFORE the SR eval — the eval then tags depth+mvec so they
                // survive to present, and the interposer's pacer generates frames.
                if (VK::SL::FrameGenWanted()) {
                    const u32 mult = (ps_r_dlss_fg_mult < 2) ? 2u :
                                     (ps_r_dlss_fg_mult > 6) ? 6u : (u32)ps_r_dlss_fg_mult;
                    VK::SL::ConfigureFG(mult - 1, rext.width, rext.height, dext.width, dext.height);
                    VK::SL::DebugTick();   // r_dlss_fg_debug: verify FG generating + Reflex live
                }
                // Output → GENERAL (fully overwritten; raw path does this inside Evaluate).
                // Explicit stage/access for the same reason as the raw path: sl.dlss ends
                // up in the same NGX snippet, which CLEARS this image as well as computing
                // into it, and the layout-derived compute-only pair does not cover that.
                VK::ImageBarrier(cmd, VK::Dlss::OutputImage(), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                 VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0,
                                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                                 VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                                 VK_ACCESS_2_TRANSFER_WRITE_BIT);
                float jpx = 0.f, jpy = 0.f;
                VK::Dlss::GetJitterPix(jpx, jpy);
                VK::SL::SrImg c{ scImg, scView, VK::SceneColor::Format(), rext.width, rext.height };
                VK::SL::SrImg d{ Swapchain.m_DepthImage, Swapchain.m_DepthView, Swapchain.m_DepthFormat, rext.width, rext.height };
                VK::SL::SrImg m{ mvImg, mvView, VK::MotionVec::Format(), rext.width, rext.height };
                VK::SL::SrImg o{ VK::Dlss::OutputImage(), VK::Dlss::OutputView(), VK::Dlss::OutputFormat(), dext.width, dext.height };
                evalOk = slProbe && VK::SL::EvaluateSR(cmd, c, d, m, o, VK::MotionVec::CurVP(), VK::MotionVec::PrevVP(),
                                                      jpx, jpy, histReset);
            } else {
                VK::Dlss::Img c{ scView, scImg, VK::SceneColor::Format() };
                VK::Dlss::Img d{ Swapchain.m_DepthView, Swapchain.m_DepthImage, Swapchain.m_DepthFormat };
                VK::Dlss::Img m{ mvView, mvImg, VK::MotionVec::Format() };
                VK::Dlss::Evaluate(cmd, c, d, m, rext.width, rext.height, histReset);
            }
            // Both routes leave the inputs in `inLayout` and the output in GENERAL.

            // Gate the tonemap on the ACTUAL resolve — a failed SL evaluate leaves the
            // output undefined (was: solid blue screen), so until it succeeds the
            // composite keeps sampling native scene colour.
            VK::Dlss::SetResolvedThisFrame(evalOk);
            if (!evalOk && useSL) {
                if (s_slRescueFrame == 0) {
                    // sl.dlss could not create its NGX feature — 0xbad0000d = out of
                    // GPU memory (the card is saturated by the streamers, and NGX
                    // allocates OUTSIDE VMA so cached-block holes don't help). Free
                    // REAL memory: hard-demote streamed textures (their big images
                    // are dedicated allocations now → the retire ring returns them
                    // to the OS in a few frames) and keep probing before giving up.
                    Msg("![VK DLSS] evaluate failed — freeing VRAM for NGX (texture demote) and retrying...");
                    VK::TextureStreamer::Instance().FreeDeviceVram(768);
                    s_slRescueFrame = Device.dwFrame ? Device.dwFrame : 1;
                } else if (slProbe && Device.dwFrame - s_slRescueFrame > 240) {
                    Msg("![VK DLSS] still failing after the VRAM rescue — DLSS OFF "
                        "(enable it before loading a level, or lower r_txstream_budget)");
                    // Post-mortem: full allocator JSON so the GB-scale vma-slack that
                    // starved NGX can be attributed pool-by-pool (also: r_vram_dump).
                    VK::Vram::DumpVmaJson();
                    ps_r_dlss = 0;
                    s_slRescueFrame = 0;
                }
            } else if (evalOk) {
                s_slRescueFrame = 0;
            }

            // DlssOutput (display res) → SHADER_READ so the tonemap samples it as the
            // base colour. SceneColor → COLOR_ATTACHMENT again so GenerateMips (avg-lum
            // + bloom source) runs on the render-res scene. Depth → DEPTH_ATTACHMENT
            // for the tonemap's own DEPTH→SHADER_READ barrier. MV back to SHADER_READ
            // for the r_mv_debug overlay. Each undoes exactly the `inLayout` hop above.
            VkImage outImg = VK::Dlss::OutputImage();
            // ⚠Explicit src, mirroring the pre-evaluate barrier: the LAST write to this
            // image is not the compute the GENERAL layout implies — sync validation names
            // it outright, `vkCmdClearColorImage[DLSS::nv.ngx.dlss.Evaluate]`. NGX clears
            // the output inside its own evaluate, so a src of COMPUTE/STORAGE_WRITE alone
            // leaves this transition racing that clear (151 hits in one run, 16-08).
            VK::ImageBarrier(cmd, outImg, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                             VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            VK::ImageBarrier(cmd, scImg,  inLayout, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            VK::ImageBarrier(cmd, Swapchain.m_DepthImage, inLayout,
                         VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
            if (!useSL)
                VK::ImageBarrier(cmd, mvImg, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            VK::Prof::ZoneEnd(cmd, zd);
        }
    }

    {
        const int z = VK::Prof::ZoneBegin(g_FrameCtx.cmd, "Tonemap");
        VK::Pass_TonemapComposite(g_FrameCtx);
        VK::Prof::ZoneEnd(g_FrameCtx.cmd, z);
    }

    // PHASE B (editor-on-Vulkan): draw the editor grid/gizmos POST-TONEMAP on the
    // final LDR swapchain — crisp + bright direct colors, immune to bloom and the
    // tone curve (the scene itself still went fully through HDR). Consumes the lines
    // the CEditorViewport sink queued this frame via EditorOverlay::AddLine.
    if (VKEditor::Active())
        EditorOverlay::ExecutePostTonemap(g_FrameCtx.cmd, Swapchain.m_ImageViews[g_FrameCtx.imageIndex],
                                          g_FrameCtx.displayExtent, Device.mFullTransform);

    // DLSS-G hudless: the swapchain now holds the final tonemapped scene with NO UI
    // (the UI pass runs later in End()). Snapshot + tag it so Frame Generation keeps
    // the HUD crisp on generated frames instead of warping it. FG-only (no-op else).
    // Captured here, before the r_mv_debug overlay, so hudless stays a clean frame.
    VK::SL::TagHudless(g_FrameCtx.cmd, Swapchain.m_Images[g_FrameCtx.imageIndex],
                       g_FrameCtx.displayExtent.width, g_FrameCtx.displayExtent.height, Swapchain.m_Format);

    // r_mv_debug: paint the motion-vector field over the final image (the
    // swapchain is COLOR_ATTACHMENT here; the UI pass in End draws on top). No-op
    // unless r_mv_debug is set. Verifies the MV reconstruction before any DLSS.
    VK::MotionVec::DrawDebugOverlay(g_FrameCtx.cmd, Swapchain.m_ImageViews[g_FrameCtx.imageIndex], g_FrameCtx.displayExtent);
    VK::Prof::FrameEnd(g_FrameCtx.cmd);

    // ⚠⚠From here on the UI may open its own pass; BEFORE this point it must not,
    // and that is not tidiness -- it is the 18-08 crash.
    //
    // 📏The order in a menu-over-a-level frame, from the ui_pass_trace log:
    //   seqRender pri 1..3 (the cursor) draws  -> UI pass OPENS
    //   seqRender pri 5    CMainMenu::OnRender -> Render->Render() lands INSIDE it
    // and the scene's first acts are vkCmdResetQueryPool and image barriers, which
    // are all illegal inside a rendering instance. The validation layer says so
    // six ways; the driver then faults in vkCmdEndRendering with no pass active,
    // 100% of the time, on ESC from a loaded level.
    // ⭐The UI has a queue for exactly this ("deferred", replayed by End on top of
    // the finished image) -- it just was not being used, because the immediate
    // path only checked whether a command buffer existed, never whether the frame
    // was at a point where drawing was legal.
    // ⚠Raised at the END of the scene rather than at Begin: the flag is what makes
    // the cursor of an early seqRender member queue instead of fault.
    VulkanUI::s_SceneDone = true;
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
    // PHASE B — editor viewport: lazily register the no-menu sink once ready.
    VKEditor::Tick();
    VK::Prof::CpuPhaseScope _cpu(VK::Prof::CPU_BEGIN);   // [VK CPUphase] fence wait + acquire
    // ⚠⚠A frame that fails here must ALSO disown the UI command buffer, and that
    // was the hole: g_VkUI_FrameCmd is only cleared at the bottom of End(), which
    // an invalid frame never reaches -- so it kept pointing at the PREVIOUS
    // frame's already-submitted buffer. The UI does not skip drawing when the
    // scene does (CMainMenu::OnRender and CHUDManager::RenderUI run either way),
    // so it took the IMMEDIATE path (cmd non-null) and recorded vkCmdBeginRendering
    // into a buffer the GPU already owns. s_bUIPassActive then stuck true, so the
    // NEXT real frame refused to open a pass and recorded every UI draw outside
    // one -- vkCmdEndRendering with no pass active, and the driver faults.
    //
    // 📏Found 18-08 as a 100% crash on ESC from a loaded level with the new RmlUi
    // menu (CRender::End -> VulkanUI::EndUIPass, ACCESS_VIOLATION inside the
    // driver). The validation layer names both halves in one frame:
    // "vkCmdBeginRendering(): ... inside an active render pass instance" at the
    // top, then eleven "vkCmdDraw(): must be issued inside an active render pass".
    // ⭐With the cmd cleared, UI draws take the deferred path -- they queue into
    // the mapped ring and get dropped by ReplayDeferredUI, which is the right
    // answer for a frame nobody is going to present.
    // ⚠The same comment three screens down (in Render()) describes this exact
    // hazard for the SCENE and was written before the UI shared the buffer.
    if (!Swapchain.ShouldRender() || g_bDeviceLost) {
        g_FrameInFlight.valid = false;
        g_VkUI_FrameCmd = VK_NULL_HANDLE;
        return;
    }

    const u32 frame = CommandManager.GetCurrentFrame();
    if (!Sync.WaitForFence(frame)) { g_FrameInFlight.valid = false; g_VkUI_FrameCmd = VK_NULL_HANDLE; return; }
    if (!Sync.ResetFence(frame))   { g_FrameInFlight.valid = false; g_VkUI_FrameCmd = VK_NULL_HANDLE; return; }

    // Texture streamer tick — BEFORE this frame's command buffer opens: the frame
    // fence just proved slot N-3 done, so (a) the feedback readback slot is safe to
    // decode, (b) descriptor sets/images retired 4+ frames ago recycle, and (c) any
    // mip promote/demote can flip material sets with nothing recording. Swaps wait
    // the TRANSFER timeline only — the graphics queue never stalls (no device idle).
    VK::TextureStreamer::Instance().Frame();
    VK::WorldMaterialCache::FrameTick();
    // Cluster-page streaming tick: decode the fence-proven feedback slot, drain
    // IO completions into staged installs, schedule new page reads (Stage B).
    VK::ClusterStream::Frame();

    FrameSync& sync = Sync.GetCurrentFrame(frame);
    g_FrameInFlight.imageIndex = Swapchain.AcquireNextImage(sync.imageAvailable);
    if (g_FrameInFlight.imageIndex == UINT32_MAX) {
        g_FrameInFlight.valid = false;
        g_VkUI_FrameCmd = VK_NULL_HANDLE; // see the note above the fence waits
        return;
    }

    g_FrameInFlight.cmd   = CommandManager.Begin();
    g_FrameInFlight.valid = (g_FrameInFlight.cmd != VK_NULL_HANDLE);
    if (!g_FrameInFlight.valid) { g_VkUI_FrameCmd = VK_NULL_HANDLE; return; }

    // Texture-streaming GPU feedback resolve: copy LAST frame's per-texture
    // desired-LOD buffer into this slot's host region, then refill it with
    // 0xFFFFFFFF for this frame's world-pass atomics. First thing in the frame
    // cmd — the barrier orders against prior submits on the same queue.
    VK::TextureStreamer::Instance().RecordFeedbackResolve(g_FrameInFlight.cmd, frame);

    // Cluster-page streaming frame ops: request/touched readback + reset, staged
    // page-data copies into the pools, dirty residency-bits/slot-base uploads.
    // MUST precede every world/VSM cull dispatch this frame (they consume the
    // bits) — first thing in the frame cmd, right after the texture resolve.
    VK::ClusterStream::RecordFrameOps(g_FrameInFlight.cmd, frame);

    // Streamline frame start: acquire this frame's SL token (shared by the DLSS SR
    // eval + FG) and, when Frame Generation is on, run Reflex sleep + the sim-start
    // latency marker. No-op when SL is off. Placed on the presenting path so the SL
    // token count tracks the interposer's present count.
    VK::SL::FrameBegin();

    // Expose this frame's command buffer to vkUIRender / VulkanUI so they
    // hit the immediate path instead of buffering into deferred queue.
    g_VkUI_FrameCmd = g_FrameInFlight.cmd;
    // Rotate the UI vertex-buffer ring to this slot's region (fence above
    // proved the GPU finished reading it) and reset the write offset.
    VulkanUI::OnFrameBegin(frame);

    // DLSS render<display (increment 2c): all scene targets render at renderExtent
    // (< display when upscaling); End() upscales the HDR result to the swapchain via
    // DLSS. When not upscaling, renderExtent == the swapchain extent → identical to
    // before. NGX gives the recommended render dims for the r_dlss_quality preset.
    VkExtent2D renderExtent = Swapchain.m_Extent;
    const char* extentOwner = "native";
    if (VK::Dlss::Upscaling() && VK::MotionVec::Enabled()) {
        VK::Dlss::GetRenderExtent(Swapchain.m_Extent.width, Swapchain.m_Extent.height,
                                  renderExtent.width, renderExtent.height);
        extentOwner = "DLSS quality";
    }
    else if (ps_r_render_scale < 0.999f) {
        // ⭐r_render_scale — the same render<display path DLSS drives, offered as
        // a plain knob for the machines DLSS cannot serve (and for anyone who
        // wants it off). No new machinery: every scene pass already sizes itself
        // from renderExtent, and the tonemap composites at displayExtent while
        // sampling the smaller buffers with normalised UVs.
        // ⚠Even dimensions — half-res passes (bloom, SSAO) derive their extent by
        // halving this one, and an odd width there costs a column of pixels.
        const float s = _max(0.5f, ps_r_render_scale);
        renderExtent.width  = _max(64u, u32(float(Swapchain.m_Extent.width)  * s + 0.5f)) & ~1u;
        renderExtent.height = _max(64u, u32(float(Swapchain.m_Extent.height) * s + 0.5f)) & ~1u;
        extentOwner = "r_render_scale";
    }

    // A render-resolution change resizes SHARED scene targets — the scene depth is a
    // single image across frames-in-flight — so wait the GPU idle before recreating.
    // Only fires on a quality / window change (rare); steady-state play is a no-op.
    static VkExtent2D s_lastRenderExtent = { 0, 0 };
    if (renderExtent.width != s_lastRenderExtent.width || renderExtent.height != s_lastRenderExtent.height) {
        vkDeviceWaitIdle(VulkanHW.m_Device);
        s_lastRenderExtent = renderExtent;
        // ⭐One line, and it is the ONLY evidence a settings change had an effect
        // on the resolution the scene is actually drawn at. DLSS logs its own
        // extent inside GetRenderExtent; r_render_scale had nothing, so a run
        // could not tell "the slider moved" from "the slider did something".
        Msg("[VK] render extent %ux%u -> display %ux%u, set by %s",
            renderExtent.width, renderExtent.height,
            Swapchain.m_Extent.width, Swapchain.m_Extent.height, extentOwner);
    }
    Swapchain.ResizeDepth(renderExtent);   // scene depth follows the render resolution

    // HDR scene target: all scene passes render into this floating-point image
    // (so highlights exceed 1.0); the Tonemap pass maps it to the swapchain.
    // One per swapchain image index, sized to the render resolution.
    VK::SceneColor::EnsureSize(renderExtent, (u32)Swapchain.m_Images.size());
    VK::MotionVec::EnsureSize(renderExtent);   // RG16F motion target tracks the render resolution
    VkImage     hdrImage = VK::SceneColor::GetImage(g_FrameInFlight.imageIndex);
    VkImageView hdrView  = VK::SceneColor::GetView(g_FrameInFlight.imageIndex);

    // Populate the per-frame context handed to Pass_*. colorView/Image point at
    // the HDR target (scene passes are oblivious — Pass_TonemapComposite reads
    // it and writes the actual swapchain image in Render()).
    g_FrameCtx.cmd          = g_FrameInFlight.cmd;
    g_FrameCtx.imageIndex   = g_FrameInFlight.imageIndex;
    g_FrameCtx.extent       = renderExtent;        // scene passes render at render res
    g_FrameCtx.displayExtent= Swapchain.m_Extent;  // tonemap composites / present at display res
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

    // Render graph: seed the frame-wide resource-state tracker at the layouts this
    // frame's color/depth are now in (COLOR_ATTACHMENT / DEPTH_ATTACHMENT). Passes
    // route their transitions through g_FrameGraph so redundant barriers coalesce
    // (the depth-read window in Pass_World is the first consumer). See vk_framegraph.
    VK::g_FrameGraph.BeginFrame(hdrImage, Swapchain.m_DepthImage);

    // DLSS sub-pixel JITTER (increment 2b.2). DLAA/DLSS is temporal super-sampling:
    // each frame nudges the projection by a Halton(2,3) sub-pixel offset so the scene
    // is sampled at N sub-pixel sites over N frames — that IS the AA mechanism (without
    // it DLSS is a ~2.4ms no-op). We jitter the COMBINED view-proj here, AFTER
    // OnCameraUpdated made it final and BEFORE any scene pass reads it (passes read
    // g_FrameCtx.viewProj = &Device.mFullTransform live during Render). GOTCHA: the
    // naive proj._31/_32 trick only works on a separate proj matrix (w=vz); for the
    // combined matrix we post-multiply by a clip translation T (T._41=jx, T._42=jy) so
    // clip.x += jx*clip.w. mulA_44(T) computes M := M*T. The HUD weapon uses its own
    // FOV projection (mFullTransform_hud) — jitter it too so the closest/highest-detail
    // geometry AAs consistently. MV history rolls AFTER this (in MotionVecDyn), so every
    // MV carries the jitter → paired with the MVJittered feature flag (Option B).
    // Gated to exactly when the DLSS eval will run (see CRender::End). Option A
    // (jitter-free MV): capture the UNJITTERED view-proj for the MV passes FIRST, then
    // jitter the raster matrix. The MV passes reproject with the unjittered matrix and
    // re-apply `jx/jy` only to gl_Position — so color/depth are super-sampled by the
    // jitter while motion vectors stay clean (MVJittered flag dropped in vk_dlss.cpp).
    float jx = 0.0f, jy = 0.0f;
    if (VK::Dlss::Enabled() && VK::MotionVec::Enabled())
    {
        VK::Dlss::NewFrameJitter(renderExtent.width, renderExtent.height);   // jitter in RENDER pixels
        VK::Dlss::GetProjJitterNDC(jx, jy);
    }
    VK::MotionVec::SetFrameVP(Device.mFullTransform, Device.mFullTransform_hud, jx, jy);
    if (jx != 0.0f || jy != 0.0f)
    {
        Fmatrix Tj; Tj.identity(); Tj._41 = jx; Tj._42 = jy;
        Device.mFullTransform.mulA_44(Tj);        // M := M * T  → clip.x += jx*clip.w
        Device.mFullTransform_hud.mulA_44(Tj);    // same sub-pixel shift for the HUD FOV
    }

    // Async compute (r_async): latch this frame's in-flight slot and the graphics
    // timeline point the compute batch orders against. The work itself is recorded
    // later, by whichever pass owns it (Pass_World records the fog inject), and the
    // graphics Submit() in End() adds the wait on its timeline.
    VK::Async::FrameBegin(frame);
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
    VK::Prof::CpuPhaseScope _cpu(VK::Prof::CPU_END);   // [VK CPUphase] submit + present
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

    // Bring the image home to PRESENT exactly once. This used to assert a fixed
    // COLOR_ATTACHMENT source ("in COLOR_ATTACHMENT since Begin") — which is only
    // true on frames where the tonemap or the UI actually touched it; a frame that
    // drew neither transitioned FROM a layout the image was never in. Asking the
    // registry instead makes the source whatever it really is, and correctly emits
    // NOTHING when the image is already in PRESENT_SRC (untouched frame).
    // ScreenshotRecord below round-trips PRESENT↔TRANSFER_SRC on its own.
    VK::g_FrameGraph.Track(image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED)
        .Require(g_FrameInFlight.cmd, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                 VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0);
    Swapchain.m_bRenderedThisFrame = false;

    // UNATTENDED CAPTURE (XROS_AUTO_SHOT). A scripted run has no keyboard, so
    // the only way it could ever look at what it built was the console channel —
    // and when that channel is unavailable the run is blind. This is the same
    // idea as XROS_CLIENT_CONSOLE, one level lower and with no dependencies:
    //   XROS_AUTO_SHOT=<dir\prefix>    write <prefix>NNN.tga
    //   XROS_AUTO_SHOT_AT=120,240,…    at these frame numbers (default 120)
    //   XROS_AUTO_SHOT_CMDS=a|b|c      console command run 1 frame BEFORE each
    //                                  capture, so one run can sweep debug views
    // Frames are counted from the first frame WITH A LEVEL (b_loaded), so the
    // capture lands at the same point in the scene regardless of how long the
    // load screen or the intro took on this machine.
    {
        static bool  s_autoInit = false;
        static char  s_autoPrefix[512] = {};
        static char  s_autoCmds[1024]  = {};
        // 32, not 8: an A/B sweep needs many alternations so scene drift averages
        // out between the paired samples — 8 points is barely three pairs.
        static u32   s_autoAt[32] = {};
        static u32   s_autoN      = 0;
        static u32   s_autoDone   = 0;
        static u32   s_autoFrame  = 0;
        static bool  s_autoArmed  = false;   // command for shot N already issued
        if (!s_autoInit) {
            s_autoInit = true;
            if (const char* p = std::getenv("XROS_AUTO_SHOT")) {
                xr_strcpy(s_autoPrefix, p);
                if (const char* c = std::getenv("XROS_AUTO_SHOT_CMDS")) xr_strcpy(s_autoCmds, c);
                const char* at = std::getenv("XROS_AUTO_SHOT_AT");
                if (at && at[0]) {
                    u32 v = 0; bool any = false;
                    for (const char* c = at; ; ++c) {
                        if (*c >= '0' && *c <= '9') { v = v * 10 + u32(*c - '0'); any = true; }
                        else { if (any && s_autoN < 32) s_autoAt[s_autoN++] = v; v = 0; any = false; if (!*c) break; }
                    }
                }
                if (!s_autoN) { s_autoAt[0] = 120; s_autoN = 1; }
                Msg("[VK AutoShot] armed: prefix='%s', %u capture point(s), cmds='%s'",
                    s_autoPrefix, s_autoN, s_autoCmds);
            }
        }
        // A loaded save parks on the level's "press any key" screen, which does
        // Device.Pause(bTimer=TRUE). That freezes the GLOBAL CLOCK — which is
        // also why the XROS_CLIENT_CONSOLE channel goes deaf there (it polls on
        // dwTimeGlobal) — while frames keep rendering the backdrop. Injected key
        // events do not reach the game's DirectInput, so an unattended run can
        // never get past it. Do exactly what CGamePersistent::OnKeyboardPress
        // does, ~300 frames in so the scripts have settled (starting the level
        // INSTANTLY, via keypress_on_start off, makes the mod's scripts throw).
        if (s_autoPrefix[0] && b_loaded && load_screen_renderer.b_registered) {
            static u32 s_autoWait = 0;
            if (++s_autoWait > 300) {
                Msg("[VK AutoShot] dismissing the start autopause (unattended run)");
                Device.Pause(FALSE, TRUE, TRUE, "AUTOSHOT_START");
                load_screen_renderer.stop();
            }
        }
        // Count only frames the game actually runs: while it is paused nothing
        // in the scene advances, so counted frames would photograph a freeze.
        if (s_autoPrefix[0] && b_loaded && !Device.Paused()) {
            ++s_autoFrame;
            if (s_autoDone < s_autoN) {
                // One frame early: run this capture's console command so the
                // cvar is already in effect for the frame that gets captured.
                if (!s_autoArmed && s_autoFrame + 1 >= s_autoAt[s_autoDone]) {
                    s_autoArmed = true;
                    if (s_autoCmds[0] && Console) {
                        // pick the s_autoDone-th '|'-separated command
                        const char* c = s_autoCmds; u32 idx = 0;
                        while (idx < s_autoDone && *c) { if (*c == '|') ++idx; ++c; }
                        if (*c) {
                            string256 one; u32 k = 0;
                            while (*c && *c != '|' && k < sizeof(one) - 1) one[k++] = *c++;
                            one[k] = 0;
                            if (one[0]) { Msg("[VK AutoShot] cmd: %s", one); Console->Execute(one); }
                        }
                    }
                }
                if (s_autoFrame >= s_autoAt[s_autoDone] && g_PendingScreenshotPath.empty()) {
                    string_path fn;
                    xr_sprintf(fn, "%s%03u.tga", s_autoPrefix, s_autoDone);
                    g_PendingScreenshotPath = fn;
                    Msg("[VK AutoShot] level-frame %u -> %s", s_autoFrame, fn);
                    ++s_autoDone;
                    s_autoArmed = false;
                }
            }
        }
    }

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
    // Reflex/DLSS-G PCL markers: bracket the sim end + the render-submit + the
    // present so the pacer knows the frame boundaries (no-op unless FG is active).
    VK::SL::MarkSimEnd();
    VK::SL::MarkRenderSubmit(true);
    CommandManager.Submit(g_FrameInFlight.cmd,
                          sync.imageAvailable,
                          renderFinished,
                          sync.inFlightFence);
    VK::SL::MarkRenderSubmit(false);
    VK::SL::MarkPresent(true);
    Swapchain.Present(renderFinished, g_FrameInFlight.imageIndex);
    VK::SL::MarkPresent(false);

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

// The only per-frame warm-up a level load owes the renderer: the weather-variant
// pipeline prewarm, deliberately spread one compile per frame because a cold uber-FS
// takes ~300 ms. While its queue holds keys the precache frames are earning their
// cost; once it is dry they are 40 ms of GPU each for a screen nobody sees.
bool CRender::PrecacheWarmupPending() { return VK::PipelineCache::PrewarmPending(); }

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
