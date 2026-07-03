// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// xrRenderVulkan — per-light volumetric cones. See vk_pass_lightcones.h.
#include "stdafx.h"
#include "vk_pass_lightcones.h"
#include "vk_swapchain.h"               // Swapchain.m_DepthImage/m_DepthView
#include "vk_scene_color.h"             // HDR scene target format
#include "vk_shaders.h"                 // g_ShaderManager (SPIRV loader)
#include "vk_shadow.h"                  // ShadowMap::GetSampler (reused for depth)
#include "vk_barriers.h"                // ImageBarrier
#include "vk_pass_ssao.h"               // DeriveProjTerms (Device.mProject is identity on this path)
#include "vk_command_buffer.h"          // CommandManager.GetCurrentFrame()
#include "vk_fullscreen.h"              // VK::Fullscreen — shared fullscreen pipeline
#include "vk_light.h"                   // Lights::CollectFrame — this frame's light set
#include "vk_Visual.h"                  // vkSynthBeam (lightplanes-derived cones)
#include "CRender_Vulkan.h"             // RImplementation.b_loaded
#include "HW_Vulkan.h"
#include "../../xr_3da/device.h"        // Device camera basis + mFullTransform
#include <algorithm>                    // std::clamp
#include <cmath>                        // std::sqrt
#include <unordered_map>                // SynthCones::s_lights (beam-backing spot lights)
#include <cstring>                      // std::memcpy (spotVP push)

extern int   ps_r_light_cones;          // master switch (default 1)
extern float ps_r_light_cone_density;   // in-scatter strength
extern float ps_r_light_cone_len;       // beam length = light range × this
extern float ps_r_light_cone_glare;     // looking-into-the-beam flare boost
extern float ps_r_light_cone_narrow;    // visible-beam angle as a share of the lit cone
extern int   ps_r_light_cone_flipy;     // diagnostic: mirror the ray basis vertically
extern float ps_r_light_cone_lum;       // beam luminance in HDR units (day visibility)
extern int   ps_r_light_cone_synth;     // beams synthesized from lightplanes geometry
extern float ps_r_light_cone_lift;      // vertical apex lift for synthesized beams (m)
extern float ps_r_light_cone_base;      // lamp-face radius multiplier for synthesized beams
extern float ps_r_light_cone_reach;     // beam/light reach = fan length × this
extern float ps_r_light_cone_power;     // synthesized-light surface intensity (0 = beam only)
extern float ps_r_light_cone_fade;      // synth beams: exp fade length after the lamp-face glow (m)
extern float ps_r_light_cone_soft;      // per-step shadow-tap disc radius, texels (beam penumbra; 0 = hard)

namespace VK {

// ── SynthCones: this frame's lightplanes carriers (see vk_pass_lightcones.h) ──
namespace SynthCones {
namespace {
    struct Entry { const vkRender_Visual* vis; Fmatrix xform; };
    xr_vector<Entry> s_entries;
    u32              s_frameTag = u32(-1);
    constexpr u32    kMaxEntries = 16;

    // PERSISTENT beam store, keyed per (visual, beam index). Each record backs
    // the visible cone AND a REAL spot light (surfaces lit, spot-shadow budget,
    // fog glow). Persistence is the point: submission comes from the carrier's
    // DRAW, which is view-frustum-culled — turning away from the car culled the
    // wreck, the sweep killed the light, and the fence it lit went dark while
    // in plain view. So records live until an EXPLICIT signal: Revoke (torch
    // bone hidden while the carrier IS drawn) or level unload. A carrier that
    // is merely off-screen keeps shining at its last transform.
    struct LightRec {
        vkLight* L = nullptr;
        bool     shown = false;   // beam alive (carrier existed, bone visible)
        Fvector  apex{}, dir{};
        float    len = 0.f, tanH = 0.f, apexR = 0.f;
        Fvector  rgb{};
    };
    std::unordered_map<u64, LightRec> s_lights;
}
void Submit(const vkRender_Visual* vis, const Fmatrix& xform)
{
    if (!vis || !vis->m_SynthBeams.count) return;
    if (s_frameTag != Device.dwFrame) { s_frameTag = Device.dwFrame; s_entries.clear(); }
    if (s_entries.size() >= kMaxEntries) return;
    for (const Entry& e : s_entries)
        if (e.vis == vis) return;   // hierarchy double-submit (~3.7×) — one per visual
    s_entries.push_back({ vis, xform });
}

// The carrier IS drawn but the beam's bone is hidden (torch switched off):
// kill its beams immediately — this is the explicit "off" signal persistence
// relies on (a merely view-culled carrier sends nothing and keeps shining).
void Revoke(const vkRender_Visual* vis)
{
    const u64 lowKey = (u64)(uintptr_t)vis;
    for (auto& kv : s_lights)
        if ((kv.first & 0x0FFFFFFFFFFFFFFFull) == lowKey) {
            kv.second.shown = false;
            if (kv.second.L) kv.second.L->active = false;
        }
}
}  // namespace SynthCones

namespace {
    constexpr u32 kFramesInFlight = CVulkanCommandManager::FRAMES_IN_FLIGHT;
    constexpr u32 kMaxCones       = 12;     // per frame, nearest-first (FL order)
    constexpr float kMaxConeDist  = 120.f;  // cones farther than this don't read

    bool                  s_inited    = false;
    bool                  s_failed    = false;
    VkDescriptorSetLayout s_setLayout = VK_NULL_HANDLE;   // set 0: binding 0 = scene depth
    VkDescriptorPool      s_pool      = VK_NULL_HANDLE;
    VkDescriptorSet       s_set[kFramesInFlight] = {};
    VkPipelineLayout      s_layout    = VK_NULL_HANDLE;
    VkPipeline            s_pipeline  = VK_NULL_HANDLE;

    struct ConePush {
        float camPos[4];      // xyz cam, w = proj _33
        float camDir[4];      // xyz forward, w = proj _43
        float camRightT[4];   // xyz right·tanX, w = density
        float camTopT[4];     // xyz up·tanY, w = glare boost
        float lightPos[4];    // xyz apex, w = beam length
        float lightDir[4];    // xyz axis, w = tan(half angle)
        float lightCol[4];    // rgb colour × fade, w = index (+100 = debug)
        float beamPrm[4];     // x = apex radius (frustum); y = 1 when this beam owns the spot map
        float spotVP[16];     // spot pick's view·proj (raw Fmatrix) — per-step beam occlusion
    };
    static_assert(sizeof(ConePush) == 192, "must match light_cone.frag PC block (device max 256)");

    bool Init()
    {
        if (s_inited) return !s_failed;
        s_inited = true;

        if (!g_ShaderManager) { s_failed = true; return false; }
        VkShaderModule vs = g_ShaderManager->Load("sunshafts.vert.spv");   // shared fullscreen triangle
        VkShaderModule fs = g_ShaderManager->Load("light_cone.frag.spv");
        if (vs == VK_NULL_HANDLE || fs == VK_NULL_HANDLE) {
            Msg("![VK Cones] light_cone.frag.spv missing — volumetric light cones disabled");
            s_failed = true; return false;
        }

        // Set 0: binding 0 = scene depth, binding 1 = spot shadow map (per-step
        // beam occlusion for the spot-budget pick). Both combined samplers.
        VkDescriptorSetLayoutBinding b[2]{};
        b[0].binding = 0; b[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b[0].descriptorCount = 1; b[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        b[1] = b[0]; b[1].binding = 1;
        VkDescriptorSetLayoutCreateInfo slci{};
        slci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        slci.bindingCount = 2; slci.pBindings = b;
        if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &slci, nullptr, &s_setLayout) != VK_SUCCESS) {
            s_failed = true; return false;
        }

        VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2 * kFramesInFlight };
        VkDescriptorPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        pci.maxSets = kFramesInFlight; pci.poolSizeCount = 1; pci.pPoolSizes = &ps;
        if (vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &s_pool) != VK_SUCCESS) {
            s_failed = true; return false;
        }
        VkDescriptorSetLayout layouts[kFramesInFlight];
        for (u32 i = 0; i < kFramesInFlight; ++i) layouts[i] = s_setLayout;
        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = s_pool; dai.descriptorSetCount = kFramesInFlight; dai.pSetLayouts = layouts;
        if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, s_set) != VK_SUCCESS) {
            s_failed = true; return false;
        }

        VkPushConstantRange pcr{ VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ConePush) };
        VkPipelineLayoutCreateInfo plci{};
        plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plci.setLayoutCount = 1; plci.pSetLayouts = &s_setLayout;
        plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
        if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_layout) != VK_SUCCESS) {
            s_failed = true; return false;
        }

        // Participating-media blend: out = src.rgb + dst·src.a (src.a carries the
        // beam transmittance T) — in-scatter PLUS extinction of the background.
        VkPipelineColorBlendAttachmentState blend{};
        blend.blendEnable         = VK_TRUE;
        blend.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        blend.dstColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        blend.colorBlendOp        = VK_BLEND_OP_ADD;
        blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blend.alphaBlendOp        = VK_BLEND_OP_ADD;
        blend.colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                    VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        s_pipeline = Fullscreen::CreatePipeline(vs, fs, VK::SceneColor::Format(), s_layout,
                                                blend, "Cones");
        if (s_pipeline == VK_NULL_HANDLE) { s_failed = true; return false; }

        Msg("[VK Cones] init OK");
        return true;
    }
}  // anon namespace

void Pass_LightCones(FrameContext& ctx)
{
    if (ctx.cmd == VK_NULL_HANDLE)  return;
    if (!RImplementation.b_loaded)
    {
        // Level unload: the persistent beam store must not survive into the
        // next level (world-anchored cache lesson — see level-transition bugs).
        if (!SynthCones::s_lights.empty()) {
            for (auto& kv : SynthCones::s_lights)
                if (kv.second.L) xr_delete(kv.second.L);
            SynthCones::s_lights.clear();
        }
        return;
    }
    if (ps_r_light_cones < 1)       return;

    // Gather this frame's volumetric SPOTS (idempotent — Pass_SunShadow /
    // EnvLight already collected the same set this frame).
    const Fvector eye = Device.vCameraPosition;
    const auto& FL = Lights::CollectFrame(eye);
    struct Cone { Fvector pos, dir; float len, tanH, apexR, fadeLen; Fvector3 rgb; float fade; bool spotShadow; };
    Cone cones[kMaxCones];
    u32  nCones = 0;
    for (u32 i = 0; i < FL.count && nCones < kMaxCones; ++i) {
        if (!FL.volFlag[i]) continue;
        if (FL.synthFlag[i]) continue;   // our beam lights: SynthCones draws their cone itself
        const auto& gl = FL.gpu[i];
        if (gl.color[3] < 0.5f) continue;                       // spots only (w: 1 spot / 0 point)
        Fvector pos{ gl.pos[0], gl.pos[1], gl.pos[2] };
        const float dist = pos.distance_to(eye);
        if (dist > kMaxConeDist) continue;
        // Fade the cone out when the camera is basically AT the apex (own
        // flashlight / walking into the lamp) — a from-the-eye cone just fills
        // the screen with milk.
        const float fade = std::clamp((dist - 0.5f) / 1.0f, 0.0f, 1.0f);
        if (fade < 0.02f) continue;
        const float cosH = std::clamp(gl.dir[3], 0.05f, 0.999f);
        Cone& c = cones[nCones++];
        c.pos  = pos;
        c.dir  = Fvector{ gl.dir[0], gl.dir[1], gl.dir[2] };
        if (c.dir.magnitude() < 1e-5f) { --nCones; continue; }
        c.dir.normalize();
        c.len  = std::clamp(gl.pos[3] * ps_r_light_cone_len, 0.5f, 60.f);
        // The VISIBLE beam is narrower than the lit cone (bright core inside the
        // wide light pool, like a real headlight). Most game lights sit at the
        // 120° default, which as a media volume reads "blob", not "beam".
        c.tanH = std::sqrt(std::max(1.f - cosH * cosH, 0.f)) / cosH * ps_r_light_cone_narrow;
        c.apexR = 0.08f;    // real lights: small bulb face
        c.fadeLen = 0.f;    // classic axial falloff (game lights keep their look)
        c.spotShadow = (FL.spotIdx == (int)i);   // this cone owns the spot map (e.g. flashlight)
        // Beam luminance premultiplied here (the shader has no free push slot):
        // ~2.2 ≈ sun-lit scene level in our HDR units; raise for day visibility.
        const float lum = fade * ps_r_light_cone_lum;
        c.rgb  = Fvector3{ gl.color[0] * lum, gl.color[1] * lum, gl.color[2] * lum };
        c.fade = fade;
    }

    // Beams synthesized from lightplanes geometry (R4 bakes headlight/searchlight
    // fans into the MODEL with no dynamic light — the game gives us nothing to
    // build a cone from, so the leaf's own fan defines it; see vkSynthBeam).
    if (ps_r_light_cone_synth) {
        // 1) Refresh the persistent store from this frame's carrier submissions
        // (only carriers actually DRAWN submit — off-screen ones keep their
        // last record; see the LightRec comment).
        if (SynthCones::s_frameTag == Device.dwFrame)
        for (const SynthCones::Entry& e : SynthCones::s_entries) {
            const vkSynthBeams& S = e.vis->m_SynthBeams;
            for (u32 bi = 0; bi < S.count; ++bi) {
                const vkSynthBeam& sb = S.b[bi];
                Fvector apex; e.xform.transform_tiny(apex, sb.apex);
                apex.y += ps_r_light_cone_lift;   // fan quads hang just below the lamp
                Fvector dir;  e.xform.transform_dir(dir, sb.dir);
                if (dir.magnitude() < 1e-5f) continue;
                dir.normalize();
                // A real volumetric spot at the same lamp (e.g. a live car)
                // would double the beam — the real light wins.
                bool dup = false;
                for (u32 i = 0; i < FL.count && !dup; ++i) {
                    if (!FL.volFlag[i] || FL.synthFlag[i] || FL.gpu[i].color[3] < 0.5f) continue;
                    const Fvector lp{ FL.gpu[i].pos[0], FL.gpu[i].pos[1], FL.gpu[i].pos[2] };
                    dup = lp.distance_to_sqr(apex) < 9.f;
                }
                if (dup) continue;

                // Headlights/searchlights shine tens of metres, not the short
                // fan the artist modelled — reach drives both the light range
                // and the visible beam (per-pixel depth cuts it at geometry).
                const float reach = std::clamp(sb.len * ps_r_light_cone_reach, 0.5f, 100.f);

                const u64 key = (u64)(uintptr_t)e.vis | ((u64)bi << 60);
                auto& rec  = SynthCones::s_lights[key];
                rec.shown  = true;
                rec.apex   = apex;
                rec.dir    = dir;
                rec.len    = reach;
                rec.tanH   = sb.tanH;
                rec.apexR  = sb.apexR;
                rec.rgb    = S.rgb;
                // Back the beam with a REAL spot light: surfaces lit, spot
                // shadow-map competition (shadow=true — the flashlight still
                // wins while ON: it sits at the eye = always nearest), fog glow.
                if (ps_r_light_cone_power > 0.f) {
                    if (!rec.L) { rec.L = xr_new<vkLight>(); rec.L->type = IRender_Light::SPOT; }
                    vkLight* L = rec.L;
                    L->active     = true;
                    L->shadow     = true;
                    // The LONG beam look now comes from the froxel fog (like the
                    // flashlight/sun shafts): volumetric=true feeds the vol-inject
                    // in-scatter boost; synthBeam keeps the FL cone path off it.
                    L->volumetric = true;
                    L->synthBeam  = true;
                    L->pos        = apex;
                    L->dir        = dir;
                    L->range      = reach;
                    L->cone       = 2.f * atanf(std::max(sb.tanH, 0.05f));
                    L->color.set(rec.rgb.x * ps_r_light_cone_power,
                                 rec.rgb.y * ps_r_light_cone_power,
                                 rec.rgb.z * ps_r_light_cone_power, 1.f);
                } else if (rec.L) rec.L->active = false;
            }
        }

        // 2) Draw the visible cones from the persistent store — the beam of an
        // off-screen carrier still crosses the view and still lights the scene.
        for (auto& kv : SynthCones::s_lights) {
            const SynthCones::LightRec& R = kv.second;
            if (!R.shown) continue;
            if (nCones >= kMaxCones) break;
            const float dist = R.apex.distance_to(eye);
            if (dist > kMaxConeDist) continue;
            const float fade = std::clamp((dist - 0.5f) / 1.0f, 0.0f, 1.0f);
            if (fade < 0.02f) continue;
            Cone& c = cones[nCones++];
            c.pos  = R.apex;
            c.dir  = R.dir;
            c.len  = R.len;
            c.tanH = R.tanH;   // the fan IS the visible beam — no narrow factor
            // Lamp-face radius from the fan geometry × live multiplier:
            // the whole headlight glows, not a point.
            c.apexR = R.apexR * ps_r_light_cone_base;
            // Synth beams: only a short lamp-face glow (~15 cm + exp fade) — the
            // LONG beam shape comes from the fog (vol boost), like the flashlight.
            c.fadeLen = ps_r_light_cone_fade;
            // Our backing light may have WON the spot-shadow budget — then the
            // visible beam is cut per-step by its map.
            c.spotShadow = FL.spotIdx >= 0
                        && FL.spotPos.distance_to_sqr(R.apex) < 1.f
                        && FL.spotDir.dotproduct(R.dir) > 0.8f;
            const float lum = fade * ps_r_light_cone_lum;
            c.rgb  = Fvector3{ R.rgb.x * lum, R.rgb.y * lum, R.rgb.z * lum };
            c.fade = fade;
        }
    }
    // Frustum-ray terms — needed by the draws below AND the triage self-test.
    const ProjTerms pt = DeriveProjTerms(Device.mFullTransform);

    // Triage log (cheap, throttled): are cones being picked at all, and with
    // what params? "0 vol-spots" at a lit lamp = the light isn't volumetric/
    // spot/collected (game side); params present = shading/visibility side.
    // The CENTRE-RAY self-test runs the shader's bounding-sphere intersection
    // on the CPU for the screen-centre ray: "hit" printed while aiming at a
    // lamp but nothing on screen = the GPU side (vUV/ray/depth binding);
    // "miss" while aiming straight at it = the ray basis itself is wrong.
    static u32 s_lastLog = 0;
    if (Device.dwTimeGlobal - s_lastLog > 3000) {
        s_lastLog = Device.dwTimeGlobal;
        const Fvector camDir = Device.vCameraDirection;
        Msg("[VK Cones] basis: pt.dir=(%.2f,%.2f,%.2f) camDir=(%.2f,%.2f,%.2f) dot=%.3f tan=(%.2f,%.2f) p33=%.4f p43=%.4f",
            pt.dir.x, pt.dir.y, pt.dir.z, camDir.x, camDir.y, camDir.z,
            pt.dir.dotproduct(camDir), pt.tanX, pt.tanY, pt.p33, pt.p43);
        if (nCones == 0)
            Msg("[VK Cones] 0 vol-spots this frame (of %u collected lights)", FL.count);
        for (u32 i = 0; i < nCones; ++i) {
            const Cone& c = cones[i];
            // Same sphere the shader intersects, against the centre ray (pt.dir).
            const float baseR = c.len * c.tanH;
            Fvector C = c.dir; C.mul(c.len * 0.5f); C.add(c.pos);
            const float R  = std::sqrt(c.len * c.len * 0.25f + baseR * baseR) * 1.02f;
            Fvector oc = eye; oc.sub(C);
            const float b  = oc.dotproduct(pt.dir);
            const float cc2 = oc.dotproduct(oc) - R * R;
            const float h  = b * b - cc2;
            if (h > 0.f)
                Msg("[VK Cones] #%u pos=(%.1f,%.1f,%.1f) dist=%.1f len=%.1f tanH=%.2f rgb=(%.2f,%.2f,%.2f) centre-ray HIT t=[%.1f..%.1f]",
                    i, c.pos.x, c.pos.y, c.pos.z, c.pos.distance_to(eye), c.len, c.tanH, c.rgb.x, c.rgb.y, c.rgb.z,
                    std::max(-b - std::sqrt(h), 0.f), -b + std::sqrt(h));
            else
                Msg("[VK Cones] #%u pos=(%.1f,%.1f,%.1f) dist=%.1f len=%.1f tanH=%.2f rgb=(%.2f,%.2f,%.2f) centre-ray miss",
                    i, c.pos.x, c.pos.y, c.pos.z, c.pos.distance_to(eye), c.len, c.tanH, c.rgb.x, c.rgb.y, c.rgb.z);
        }
    }
    if (nCones == 0) return;
    if (!Init())     return;

    VkCommandBuffer cmd = ctx.cmd;

    // Scene depth: attachment → sampled for these draws (restored below).
    ImageBarrier(cmd, Swapchain.m_DepthImage, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);

    const u32 slot = CommandManager.GetCurrentFrame() % kFramesInFlight;
    {
        VkDescriptorImageInfo ii[2]{};
        ii[0].sampler     = ShadowMap::GetSampler();
        ii[0].imageView   = Swapchain.m_DepthView;
        ii[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        ii[1] = ii[0];
        ii[1].imageView   = ShadowMap::GetSpotBeamView();   // spot + GRASS casters (SHADER_READ after Pass_SunShadow)
        VkWriteDescriptorSet w[2]{};
        for (u32 k = 0; k < 2; ++k) {
            w[k].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[k].dstSet = s_set[slot]; w[k].dstBinding = k; w[k].descriptorCount = 1;
            w[k].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[k].pImageInfo = &ii[k];
        }
        vkUpdateDescriptorSets(VulkanHW.m_Device, 2, w, 0, nullptr);
    }

    VkRenderingAttachmentInfo cAtt{};
    cAtt.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    cAtt.imageView   = ctx.colorView;
    cAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    cAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;
    cAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo ri{};
    ri.sType                = VK_STRUCTURE_TYPE_RENDERING_INFO;
    ri.renderArea.extent    = ctx.extent;
    ri.layerCount           = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments    = &cAtt;
    vkCmdBeginRendering(cmd, &ri);

    VkViewport vp{ 0.f, 0.f, (float)ctx.extent.width, (float)ctx.extent.height, 0.f, 1.f };
    vkCmdSetViewport(cmd, 0, 1, &vp);
    VkRect2D sc{ {}, ctx.extent };
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_layout, 0, 1, &s_set[slot], 0, nullptr);

    ConePush push{};
    push.camPos[0] = eye.x; push.camPos[1] = eye.y; push.camPos[2] = eye.z; push.camPos[3] = pt.p33;
    push.camDir[0] = pt.dir.x; push.camDir[1] = pt.dir.y; push.camDir[2] = pt.dir.z; push.camDir[3] = pt.p43;
    push.camRightT[0] = pt.right.x * pt.tanX; push.camRightT[1] = pt.right.y * pt.tanX; push.camRightT[2] = pt.right.z * pt.tanX;
    push.camRightT[3] = ps_r_light_cone_density;
    // flipy diagnostic: negating the top vector mirrors the per-pixel rays
    // vertically — equivalent to flipping the shader's ndc.y sign, no SPIRV edit.
    const float topSign = ps_r_light_cone_flipy ? -pt.tanY : pt.tanY;
    push.camTopT[0] = pt.top.x * topSign; push.camTopT[1] = pt.top.y * topSign; push.camTopT[2] = pt.top.z * topSign;
    push.camTopT[3] = ps_r_light_cone_glare;

    for (u32 i = 0; i < nCones; ++i) {
        const Cone& c = cones[i];
        push.lightPos[0] = c.pos.x; push.lightPos[1] = c.pos.y; push.lightPos[2] = c.pos.z; push.lightPos[3] = c.len;
        push.lightDir[0] = c.dir.x; push.lightDir[1] = c.dir.y; push.lightDir[2] = c.dir.z; push.lightDir[3] = c.tanH;
        push.lightCol[0] = c.rgb.x; push.lightCol[1] = c.rgb.y; push.lightCol[2] = c.rgb.z;
        // w: normally the shadow-tap soft radius in texels (r_light_cone_soft —
        // area-light penumbra so grass/crown cutouts don't paint razor threads).
        // r_light_cones 2 replaces it with the debug cone index + 100 (the shader
        // treats w >= 99.5 as debug mode, keyed vs the [VK Cones] log).
        push.lightCol[3] = ps_r_light_cones >= 2 ? (float)i + 100.f : ps_r_light_cone_soft;
        push.beamPrm[0] = c.apexR;
        push.beamPrm[1] = c.spotShadow ? 1.f : 0.f;
        push.beamPrm[2] = FL.spotRange;   // spot far plane for the linear-depth compare
        push.beamPrm[3] = c.fadeLen;      // >0: lamp-face glow with exp fade (synth beams)
        std::memcpy(push.spotVP, &ShadowMap::GetSpotVP(), sizeof(push.spotVP));
        vkCmdPushConstants(cmd, s_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(push), &push);
        vkCmdDraw(cmd, 3, 1, 0, 0);
    }
    vkCmdEndRendering(cmd);

    // Restore the frame-wide depth layout invariant for the passes after us.
    ImageBarrier(cmd, Swapchain.m_DepthImage, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
}

void LightCones_Destroy()
{
    // Beam-backing spot lights (destructor unregisters from the light registry).
    for (auto& kv : SynthCones::s_lights)
        if (kv.second.L) xr_delete(kv.second.L);
    SynthCones::s_lights.clear();

    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
    if (s_pipeline)  { vkDestroyPipeline(VulkanHW.m_Device, s_pipeline, nullptr); s_pipeline = VK_NULL_HANDLE; }
    if (s_layout)    { vkDestroyPipelineLayout(VulkanHW.m_Device, s_layout, nullptr); s_layout = VK_NULL_HANDLE; }
    if (s_pool)      { vkDestroyDescriptorPool(VulkanHW.m_Device, s_pool, nullptr); s_pool = VK_NULL_HANDLE; }
    if (s_setLayout) { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_setLayout, nullptr); s_setLayout = VK_NULL_HANDLE; }
    for (u32 i = 0; i < kFramesInFlight; ++i) s_set[i] = VK_NULL_HANDLE;
    s_inited = false; s_failed = false;
}

}  // namespace VK
