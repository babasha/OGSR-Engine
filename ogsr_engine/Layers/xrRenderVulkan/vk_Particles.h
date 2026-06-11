// xrRenderVulkan - Vulkan-native particle visuals (effect + group).
//
// Dedicated classes built directly on PAPI (ParticleManager()) — they do NOT
// reuse the R4 PS::CParticleEffect/CParticleGroup visuals (too render-coupled).
// Simulation logic (OnFrame/Play/Stop/UpdateParent/Compile) is ported verbatim
// from the R4 sources; rendering is our own camera-facing billboard pass
// (see vk_pass_particles.cpp). The shared definitions PS::CPEDef / PS::CPGDef
// are loaded by vk_ParticleEffectDef.cpp + vk_PSLibrary.cpp.
//
// Deliberately light-weight header: only forward-declares the PS definition
// types so the render pass can include it without pulling the R4 PS headers.

#pragma once

#include "vk_core.h"
#include "vk_Visual.h"
#include "../../Include/xrRender/ParticleCustom.h"

namespace FVF { struct LIT; }
namespace PS  { class CPEDef; class CPGDef; }
namespace VK  { class CVulkanBuffer; }

// Particle blend modes (matches R4 CBlender_Particle oBlend / monolith port).
// Fixed underlying type so vk_pass_particles.h can opaquely forward-declare it.
enum EParticleBlendMode : int
{
    PBM_SET       = 0,  // ONE, ZERO            (opaque)
    PBM_BLEND     = 1,  // SRC_ALPHA, 1-SRC_A   (standard alpha)
    PBM_ADD       = 2,  // ONE, ONE             (additive — fire/sparks/muzzle)
    PBM_MUL       = 3,  // DST_COLOR, ZERO      (multiply)
    PBM_MUL_2X    = 4,  // DST_COLOR, SRC_COLOR (multiply 2x)
    PBM_ALPHA_ADD = 5,  // SRC_ALPHA, ONE       (alpha-weighted additive)
    PBM_DISTORT   = 6,  // distortion — skipped (no distortion buffer)
    PBM_COUNT     = 7,
};

class vkCParticleEffect;

// ----------------------------------------------------------------------------
// Common base for particle visuals. Added to g_DynamicVisuals by add_Visual;
// the particle pass filters by Type and flattens groups to leaf effects.
// ----------------------------------------------------------------------------
class vkParticleVisual : public vkRender_Visual, public IParticleCustom
{
public:
    IParticleCustom* dcast_ParticleCustom() override { return this; }

    // Append the renderable leaf effects (self for an effect, children for a
    // group) into `out`. The pass draws each leaf with the right blend pipeline.
    virtual void CollectEffects(xr_vector<vkCParticleEffect*>& out) = 0;
};

// ----------------------------------------------------------------------------
// Single particle effect — one PAPI effect + action list, sprite billboards.
// ----------------------------------------------------------------------------
class vkCParticleEffect final : public vkParticleVisual
{
public:
    PS::CPEDef* m_Def = nullptr;

    int  m_HandleEffect     = -1;
    int  m_HandleActionList = -1;

    EParticleBlendMode m_BlendMode = PBM_BLEND;

    enum
    {
        flRT_Playing      = (1 << 0),
        flRT_DefferedStop = (1 << 1),
        flRT_XFORM        = (1 << 2),
        flRT_HUDmode      = (1 << 3),
    };
    Flags8  m_RT_Flags;

    float   m_fElapsedLimit = 0.f;
    s32     m_MemDT         = 0;
    Fvector m_InitialPosition{};
    Fmatrix m_XFORM;

    // Cached texture descriptor set (shared, owned by the particle pass cache).
    VkDescriptorSet m_TextureSet = VK_NULL_HANDLE;
    bool            m_TextureResolved = false;

public:
    vkCParticleEffect();
    virtual ~vkCParticleEffect();

    BOOL Compile(PS::CPEDef* def);

    PS::CPEDef* GetDefinition() const { return m_Def; }
    int         GetHandleEffect() const { return m_HandleEffect; }
    EParticleBlendMode GetBlendMode() const { return m_BlendMode; }

    // Resolve (lazily) and return the shared texture descriptor set.
    VkDescriptorSet ResolveTextureSet();

    // Build camera-facing billboards into `dst` (FVF::LIT). Returns vertex count
    // written (6 per particle), clamped to `maxVerts`.
    u32 BuildVertices(FVF::LIT* dst, u32 maxVerts);

    // vkParticleVisual
    void CollectEffects(xr_vector<vkCParticleEffect*>& out) override;

    // IParticleCustom
    void OnDeviceCreate()  override {}
    void OnDeviceDestroy() override {}
    void UpdateParent(const Fmatrix& m, const Fvector& velocity, BOOL bXFORM) override;
    void OnFrame(u32 dt)   override;
    void Play()            override;
    void Stop(BOOL bDefferedStop = TRUE) override;
    BOOL IsPlaying()       override { return m_RT_Flags.is(flRT_Playing); }
    BOOL IsDeferredStopped() override { return m_RT_Flags.is(flRT_DefferedStop); }
    u32  ParticlesCount()  override;
    float GetTimeLimit()   override;
    const shared_str Name() override;
    void SetHudMode(BOOL b) override { m_RT_Flags.set(flRT_HUDmode, b); }
    BOOL GetHudMode()      override { return m_RT_Flags.is(flRT_HUDmode); }

private:
    EParticleBlendMode DetermineBlendMode() const;
};

// ----------------------------------------------------------------------------
// Particle group — N timed child effects (campfires, explosions, ...).
// Time-windowed playback of children; the advanced on-birth/on-dead/on-play
// related/free child spawning is not ported (rare), so flOnPlayChild etc. are
// ignored — the core timed child effects cover the common cases.
// ----------------------------------------------------------------------------
class vkCParticleGroup final : public vkParticleVisual
{
public:
    PS::CPGDef* m_Def = nullptr;

    enum
    {
        flRT_Playing      = (1 << 0),
        flRT_DefferedStop = (1 << 1),
    };
    Flags8  m_RT_Flags;
    float   m_CurrentTime = 0.f;
    Fvector m_InitialPosition{};

    xr_vector<vkCParticleEffect*> m_Items;   // one child effect per CPGDef::SEffect

public:
    vkCParticleGroup();
    virtual ~vkCParticleGroup();

    BOOL Compile(PS::CPGDef* def);

    // vkParticleVisual
    void CollectEffects(xr_vector<vkCParticleEffect*>& out) override;

    // IParticleCustom
    void OnDeviceCreate()  override {}
    void OnDeviceDestroy() override {}
    void UpdateParent(const Fmatrix& m, const Fvector& velocity, BOOL bXFORM) override;
    void OnFrame(u32 dt)   override;
    void Play()            override;
    void Stop(BOOL bDefferedStop = TRUE) override;
    BOOL IsPlaying()       override { return m_RT_Flags.is(flRT_Playing); }
    BOOL IsDeferredStopped() override { return m_RT_Flags.is(flRT_DefferedStop); }
    u32  ParticlesCount()  override;
    float GetTimeLimit()   override;
    const shared_str Name() override;
    void SetHudMode(BOOL b) override;
    BOOL GetHudMode()      override;
};

// ----------------------------------------------------------------------------
// Factories (resolve against the loaded PS library — vk_PSLibrary.cpp).
// ----------------------------------------------------------------------------
vkCParticleEffect* vkCreateParticleEffect(const char* ped_name);   // PED → effect
vkParticleVisual*  vkCreateParticle(const char* name);             // PED or PGD; null if unknown
