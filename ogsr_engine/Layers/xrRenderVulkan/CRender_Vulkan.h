// xrRenderVulkan - Vulkan renderer for OGSR-Engine
// Copyright (c) 2026 Egor Babushkin (https://github.com/babasha)
//
// CRender_Vulkan — top-level renderer (the IRender_interface impl) that the
// engine talks to via the global ::Render pointer. Mirrors R4's class CRender.
//
// Most overrides are no-op stubs that get fleshed out as each subsystem ports
// over. Frame work and resource management belong to the framegraph; this
// class stays a thin adapter on top.

#pragma once
#include "vk_core.h"
#include "../../xr_3da/Render.h"
#include "../../xr_3da/pure.h"

// Naming follows OGSR R4 convention: class CRender + global RImplementation.
// The filename keeps the _Vulkan suffix so VK sources are obviously distinct
// from R4. Not marked `final` so monolith-derived render structures can
// extend it as they get grafted in.
class CRender : public IRender_interface, public pureFrame
{
public:
    CRender();
    ~CRender() override;

    // ----- IRender_interface : Loading / Unloading --------------------------
    void create() override;
    void destroy() override;
    void reset_begin() override;
    void reset_end() override;
    void level_Load(IReader*) override;
    void level_Unload() override;

    // ----- IRender_interface : Shader compile -------------------------------
    HRESULT shader_compile(LPCSTR name, DWORD const* pSrcData, UINT SrcDataLen,
                           LPCSTR pFunctionName, LPCSTR pTarget, DWORD Flags,
                           void*& result) override;

    // ----- IRender_interface : Information ----------------------------------
    LPCSTR getShaderPath() override;
    IRenderVisual* getVisual(int id) override;
    u32 getVisualCount() override;
    IRender_Target* getTarget() override;

    // ----- IRender_interface : Visuals & wallmarks --------------------------
    void add_Visual(u32 context_id, IRenderable* root,
                    IRenderVisual* V, Fmatrix& m) override;
    void add_StaticWallmark(const wm_shader& S, const Fvector& P, float s,
                            CDB::TRI* T, Fvector* V) override;
    void add_StaticWallmark(IWallMarkArray* pArray, const Fvector& P, float s,
                            CDB::TRI* T, Fvector* V) override;
    void add_SkeletonWallmark(Fmatrix* xf, IKinematics* obj,
                              IWallMarkArray* pArray, Fvector& start,
                              Fvector& dir, float size) override;
    void clear_static_wallmarks() override;

    IRender_ObjectSpecific* ros_create(IRenderable* parent) override;
    void ros_destroy(IRender_ObjectSpecific*&) override;

    // ----- IRender_interface : Lighting -------------------------------------
    IRender_Light* light_create() override;

    // ----- IRender_interface : Particles ------------------------------------
    void ParticleEffectFillName(xr_vector<shared_str>& s) override;
    void ParticleGroupFillName(xr_vector<shared_str>& s) override;
    float GetParticlesTimeLimit(LPCSTR name) override;

    // ----- IRender_interface : Models ---------------------------------------
    IRenderVisual* model_CreateParticles(LPCSTR name, BOOL bNoPool) override;
    IRenderVisual* model_Create(LPCSTR name, IReader* data) override;
    IRenderVisual* model_CreateChild(LPCSTR name, IReader* data) override;
    IRenderVisual* model_Duplicate(IRenderVisual* V) override;
    void model_Delete(IRenderVisual*& V, BOOL bDiscard) override;
    void model_Logging(BOOL bEnable) override;
    void models_Prefetch() override;
    void models_Clear(BOOL b_complete) override;
    void models_savePrefetch() override;
    void models_begin_prefetch1(bool val) override;

    // ----- IRender_interface : Frame ----------------------------------------
    void Calculate() override;
    void Render() override;
    void AfterWorldRender() override;
    void AfterUIRender() override;
    void Screenshot(ScreenshotMode mode, LPCSTR name) override;

    // ----- IRender_interface : Render mode ----------------------------------
    void rmNear(CBackend& cmd_list) override;
    void rmFar(CBackend& cmd_list) override;
    void rmNormal(CBackend& cmd_list) override;

    // ----- IRender_interface : Stats ----------------------------------------
    u32 memory_usage() override;
    u32 GetCacheStatPolys() override;

    // ----- IRender_interface : Backbuffer -----------------------------------
    void Begin() override;
    void Clear() override;
    void End() override;
    void ClearTarget() override;

    // ----- IRender_interface : Cached transforms ----------------------------
    void SetCacheXform(Fmatrix& mView, Fmatrix& mProject) override;
    void SetCacheXformOld(Fmatrix& mView, Fmatrix& mProject) override;

    // ----- IRender_interface : Backend & camera -----------------------------
    CBackend& get_imm_command_list() override;
    void OnCameraUpdated(bool from_actor) override;

    // ----- pureFrame --------------------------------------------------------
    void OnFrame() override;

protected:
    void ScreenshotImpl(ScreenshotMode mode, LPCSTR name) override;
};

extern CRender RImplementation;
