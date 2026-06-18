// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan — dynamic light (STEP 3, forward point/spot lights).
//
// vkLight is the real IRender_Light the game drives (campfires, flashlight,
// muzzle flashes, anomalies, lamps — all spawn through CRender::light_create
// and immediately call the setters). Every instance lives in a global registry;
// each frame EnvLight::Update calls Lights::Collect to pick the nearest active
// lights and uploads them in the shared Lighting UBO, where the forward
// fragment shaders (world / skinned / grass / trees) accumulate them.
//
// Not implemented (v1): per-light shadows, volumetric, flares, light textures.
#pragma once
#include "HW_Vulkan.h"
#include "../../xr_3da/Render.h"   // IRender_Light

namespace VK {

class vkLight final : public IRender_Light
{
public:
    u32     type   = POINT;       // LT enum (DIRECT treated as inactive, OMNIPART as POINT)
    bool    active = false;
    bool    hud    = false;
    bool    shadow = false;       // game asked for a shadow-casting light (flashlight etc.)
    Fvector pos{ 0.f, 0.f, 0.f };
    Fvector dir{ 0.f, 0.f, 1.f };
    float   range  = 8.f;
    float   cone   = deg2rad(120.f);   // full apex angle (spot)
    Fcolor  color{};
    shared_str texture;                // spot projection texture (flashlight cookie)

    vkLight();
    ~vkLight() override;

    void  set_type(LT t)                    override { type = u32(t); }
    void  set_active(bool b)                override { active = b; }
    bool  get_active()                      override { return active; }
    void  set_shadow(bool b)                override { shadow = b; }
    bool  get_shadow()                      override { return shadow; }
    void  set_volumetric(bool)              override {}
    bool  get_volumetric()                  override { return false; }
    void  set_volumetric_intensity(float)   override {}
    void  set_volumetric_distance(float)    override {}
    void  set_flare(bool)                   override {}
    bool  get_flare()                       override { return false; }
    void  set_position(const Fvector& P)    override { pos = P; }
    void  set_rotation(const Fvector& D, const Fvector&) override { if (D.magnitude() > 1e-5f) { dir = D; dir.normalize(); } }
    void  set_cone(float angle)             override { cone = angle; }
    void  set_range(float r)                override { range = r; }
    float get_range() const                 override { return range; }
    void  set_virtual_size(float)           override {}
    void  set_texture(LPCSTR n)             override { texture = (n && n[0]) ? n : nullptr; }
    void  set_color(const Fcolor& c)        override { color = c; }
    void  set_color(float r, float g, float b) override { color.set(r, g, b, 1.f); }
    Fcolor get_color() const                override { return color; }
    void  set_hud_mode(bool b)              override { hud = b; }
    bool  get_hud_mode()                    override { return hud; }
    void  set_moveable(bool)                override {}
    bool  get_moveable()                    override { return true; }
};

namespace Lights {

constexpr u32 kMaxLights = 16;          // UBO array size — foliage + the non-clustered fallback path
constexpr u32 kMaxClusterLights = 256;  // SSBO array size — the clustered forward path (vk_clustered)

// GPU-side light record — must match the Lighting UBO `lights[]` entry layout.
struct GpuLight {
    float pos[4];     // xyz = world position, w = range
    float color[4];   // rgb = colour,         w = 1 spot / 0 point
    float dir[4];     // xyz = spot direction, w = cos(cone/2)
};

// Per-frame light set + the two shadow-casting picks (budget: 1 spot + 1 point).
// `gpu` holds up to kMaxClusterLights (clustered path uploads them all to the
// SSBO); the UBO + foliage read only the first kMaxLights. spotIdx/pointIdx are
// indices into this same array, so they're valid for both paths.
struct FrameLights {
    GpuLight gpu[kMaxClusterLights];
    u32      count    = 0;
    int      spotIdx  = -1;   // gpu[] index of the spot-shadowed light (flashlight), -1 = none
    int      pointIdx = -1;   // gpu[] index of the point-shadowed light (campfire),  -1 = none
    // Source params of the shadowed picks (for the shadow-pass cameras):
    Fvector  spotPos{};  Fvector spotDir{ 0.f, 0.f, 1.f }; float spotRange = 0.f; float spotCone = 0.f;
    Fvector  pointPos{}; float pointRange = 0.f;
    shared_str spotTexture;   // the picked spot's cookie (flashlight beam pattern)
};

// Collect the nearest active lights around `eye` and pick the shadowed ones.
// Idempotent per Device.dwFrame — Pass_SunShadow calls it first (to render the
// dynamic shadow maps), EnvLight::Update reuses the same result for the UBO.
const FrameLights& CollectFrame(const Fvector& eye);

}  // namespace Lights
}  // namespace VK
