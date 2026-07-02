// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

// GPU-Driven Particles — Phase 1 (proof of life). One hardcoded effect
// (Source + Gravity + KillOld) simulated and drawn entirely on the GPU with a
// resident pool + free-list. See gpu_particles_roadmap.md.
#include "stdafx.h"
#include "vk_gpu_particles.h"
#include "vk_buffer.h"
#include "vk_shaders.h"
#include "vk_profiler.h"
#include "vk_command_buffer.h"
#include "vk_barriers.h"
#include "vk_pipeline_cache.h"
#include "vk_swapchain.h"
#include "vk_scene_color.h"
#include "vk_pass_particles.h"   // ParticlePass::GetTextureView — bindless sprite views
#include "HW_Vulkan.h"
#include "vk_core.h"
#include "../../xr_3da/device.h"

extern int ps_r_gpu_particles;
extern int ps_r_gpu_particles_max;

// World-emitter feed: walks the live .pe particle visuals and appends one
// EmitterSample per playing smoke effect. Defined in vk_gpu_particles_feed.cpp
// (which carries the heavy PS/visual includes the clean module avoids).
namespace VK { namespace GPUParticles { struct EmitterSample; } }
extern void VK_GP_CollectWorldEmitters(xr_vector<VK::GPUParticles::EmitterSample>& out, float dt);

namespace VK { namespace GPUParticles {

namespace {
    constexpr u32 kFramesInFlight     = CVulkanCommandManager::FRAMES_IN_FLIGHT;
    constexpr u32 kDefaultMaxParticles = 1u << 16;   // 64K — plenty for the Phase-1 test
    constexpr u32 kNumBindings         = 8u;
    constexpr u32 kMaxEmitPerFrame     = 1024u;      // safety cap on a frame's total spawn burst
    constexpr u32 kMaxEmitPerEmitter   = 256u;       // per-emitter burst cap
    constexpr u32 kMaxPrograms         = 64u;        // per-defId program registry capacity
    constexpr u32 kMaxEmitters         = 256u;       // active emitters (incl. camera) per frame

    // ---- State -------------------------------------------------------------
    bool s_inited     = false;
    bool s_failed     = false;
    bool s_poolInited = false;        // gp_init ran once (lazy, first dispatch)
    u32  s_maxP       = kDefaultMaxParticles;

    // SSBOs.
    CVulkanBuffer* s_pool      = nullptr;   // GPUParticle[maxP]
    CVulkanBuffer* s_counters  = nullptr;   // [0]=freeCount, [1]=aliveCount
    CVulkanBuffer* s_aliveList = nullptr;   // uint[maxP]
    CVulkanBuffer* s_indirect  = nullptr;   // VkDrawIndirectCommand
    CVulkanBuffer* s_freeList  = nullptr;   // uint[maxP]
    CVulkanBuffer* s_program   = nullptr;   // Program[kMaxPrograms]
    CVulkanBuffer* s_spawnReq[kFramesInFlight] = {};  // SpawnRequest[kMaxEmitters], per-frame
    CVulkanBuffer* s_texInfo   = nullptr;   // TexInfo[kMaxPrograms] (per-program atlas meta)

    // ---- Bindless sprite textures (Phase 3 #5) -----------------------------
    // One COMBINED_IMAGE_SAMPLER array (slot = program index), written
    // append-only as effects register — a freshly written slot is never in use
    // by an in-flight frame, so no UPDATE_AFTER_BIND is needed. Requires
    // descriptor indexing; without it the draw stays on the procedural circle.
    bool                  s_useTextures = false;
    VkDescriptorSetLayout s_texSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool      s_texPool      = VK_NULL_HANDLE;
    VkDescriptorSet       s_texSet       = VK_NULL_HANDLE;
    VkSampler             s_texSampler   = VK_NULL_HANDLE;

    // ---- Per-defId program registry (Phase 3 step 1) -----------------------
    // s_program holds Program[kMaxPrograms]. Slot 0 = authored campfire;
    // gp_mirror/gp_spawn register real .pe effects into later slots. Each
    // particle stores its slot in defId so gp_simulate runs the right program.
    shared_str s_progName[kMaxPrograms];    // [0] = "<campfire>"
    float      s_progRate[kMaxPrograms] = {};
    float      s_progLife[kMaxPrograms] = {};   // particle lifetime (for pool-budget gating)
    u32        s_progCount     = 0;

    // Auto-routing (Phase 3 #3) won't move an effect onto the shared GPU pool if
    // its steady-state budget (rate × life) exceeds this — area fog / persistent
    // anomaly fields (e.g. fog_ground_base = 1024/s × 180s ≈ 184k) would saturate
    // the 64K pool and starve every other effect. They stay on the CPU path.
    constexpr float kRouteBudget = 8192.0f;

    // ---- Emitters (Phase 3 step 2) -----------------------------------------
    // Each emitter feeds one spawn request per frame. [0] is the camera-pinned
    // emitter (set by gp_mirror); the rest are persistent world emitters placed
    // by gp_spawn. Each carries its own fractional emit-rate accumulator.
    struct Emitter {
        u32      program      = 0;
        bool     followCamera = false;
        bool     enabled      = true; // camera emitter starts OFF (test only)
        Fvector  pos          = {};   // world position when !followCamera
        float    accum        = 0.0f; // fractional emit-rate carry
    };
    xr_vector<Emitter> s_emitters;     // [0] = camera emitter (off unless gp_mirror)

    // Pipelines.
    VkPipeline s_pipeInit  = VK_NULL_HANDLE;
    VkPipeline s_pipeReset = VK_NULL_HANDLE;
    VkPipeline s_pipeEmit  = VK_NULL_HANDLE;
    VkPipeline s_pipeSim   = VK_NULL_HANDLE;
    VkPipeline s_pipeBuild = VK_NULL_HANDLE;
    VkPipeline s_pipeDraw  = VK_NULL_HANDLE;

    // Layouts.
    VkPipelineLayout s_compLayout = VK_NULL_HANDLE;
    VkPipelineLayout s_drawLayout = VK_NULL_HANDLE;

    // Descriptors.
    VkDescriptorSetLayout s_setLayout = VK_NULL_HANDLE;
    VkDescriptorPool      s_poolDesc  = VK_NULL_HANDLE;
    VkDescriptorSet       s_set[kFramesInFlight] = {};

    // ---- Helpers -----------------------------------------------------------
    VkShaderModule LoadShader(const char* name)
    {
        if (!g_ShaderManager) return VK_NULL_HANDLE;
        VkShaderModule m = g_ShaderManager->Load(name);
        if (m == VK_NULL_HANDLE) Msg("![VK GP] shader '%s' not found", name);
        return m;
    }

    VkPipeline CreateComputePipeline(VkShaderModule mod, VkPipelineLayout layout)
    {
        VkComputePipelineCreateInfo cp{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        cp.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cp.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
        cp.stage.module = mod;
        cp.stage.pName  = "main";
        cp.layout       = layout;
        VkPipeline pipe = VK_NULL_HANDLE;
        if (vkCreateComputePipelines(VulkanHW.m_Device, VK::PipelineCache::GetCacheObject(),
                                     1, &cp, nullptr, &pipe) != VK_SUCCESS)
            return VK_NULL_HANDLE;
        return pipe;
    }

    // Shader-write → shader-read/write barrier between compute stages.
    void MemBarrier(VkCommandBuffer cmd)
    {
        VkMemoryBarrier mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }

    void DestroyPipelines()
    {
        auto destroy = [](VkPipeline& p) { if (p) { vkDestroyPipeline(VulkanHW.m_Device, p, nullptr); p = VK_NULL_HANDLE; } };
        destroy(s_pipeInit);
        destroy(s_pipeReset);
        destroy(s_pipeEmit);
        destroy(s_pipeSim);
        destroy(s_pipeBuild);
        destroy(s_pipeDraw);
        if (s_compLayout) { vkDestroyPipelineLayout(VulkanHW.m_Device, s_compLayout, nullptr); s_compLayout = VK_NULL_HANDLE; }
        if (s_drawLayout) { vkDestroyPipelineLayout(VulkanHW.m_Device, s_drawLayout, nullptr); s_drawLayout = VK_NULL_HANDLE; }
        if (s_poolDesc)   { vkDestroyDescriptorPool(VulkanHW.m_Device, s_poolDesc, nullptr);   s_poolDesc   = VK_NULL_HANDLE; }
        if (s_setLayout)  { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_setLayout, nullptr); s_setLayout = VK_NULL_HANDLE; }
        for (u32 i = 0; i < kFramesInFlight; ++i) s_set[i] = VK_NULL_HANDLE;
        if (s_texPool)      { vkDestroyDescriptorPool(VulkanHW.m_Device, s_texPool, nullptr);      s_texPool      = VK_NULL_HANDLE; }
        if (s_texSetLayout) { vkDestroyDescriptorSetLayout(VulkanHW.m_Device, s_texSetLayout, nullptr); s_texSetLayout = VK_NULL_HANDLE; }
        if (s_texSampler)   { vkDestroySampler(VulkanHW.m_Device, s_texSampler, nullptr);          s_texSampler   = VK_NULL_HANDLE; }
        s_texSet = VK_NULL_HANDLE;
    }

    void DestroyBuffers()
    {
        auto del = [](CVulkanBuffer*& b) { if (b) { b->Destroy(); xr_delete(b); b = nullptr; } };
        del(s_pool);
        del(s_counters);
        del(s_aliveList);
        del(s_indirect);
        del(s_freeList);
        del(s_program);
        del(s_texInfo);
        for (u32 i = 0; i < kFramesInFlight; ++i) del(s_spawnReq[i]);
    }

    // ---- Authored test effect (campfire smoke) -----------------------------
    // Phase 2 proof: a real action program (not hardcoded sim math) drives the
    // GPU effect. The future .pe translator fills this same Program struct.
    void FillDomainLine(GenDomain& d, float ax, float ay, float az, float bx, float by, float bz)
    {
        d = {};
        d.dtype = 1;                              // PDLine: p1 + p2*t, p2 = delta
        d.p1[0] = ax; d.p1[1] = ay; d.p1[2] = az;
        d.p2[0] = bx - ax; d.p2[1] = by - ay; d.p2[2] = bz - az;
    }
    void FillDomainBox(GenDomain& d, float minx, float miny, float minz, float maxx, float maxy, float maxz)
    {
        d = {};
        d.dtype = 4;                              // PDBox: p1=min, p2=max
        d.p1[0] = minx; d.p1[1] = miny; d.p1[2] = minz;
        d.p2[0] = maxx; d.p2[1] = maxy; d.p2[2] = maxz;
    }
    void FillDomainSphere(GenDomain& d, float cx, float cy, float cz, float r1, float r2)
    {
        d = {};
        d.dtype = 5;                              // PDSphere shell [r2..r1]
        d.p1[0] = cx; d.p1[1] = cy; d.p1[2] = cz;
        d.radii[0] = r1; d.radii[1] = r2;
        d.radii[2] = r1 * r1; d.radii[3] = r2 * r2;
    }

    ActionRec MakeAction(u32 type)
    {
        ActionRec a{};
        a.atype = type;
        return a;
    }

    void BuildTestProgram(Program& prog)
    {
        prog = {};
        prog.emitRate = 80.0f;                    // particles / second

        EmitDesc& e = prog.emit;
        FillDomainSphere(e.posDom, 0.f, 0.f, 0.f, 0.25f, 0.f);   // base puff
        FillDomainBox   (e.velDom, -0.15f, 0.6f, -0.15f, 0.15f, 1.4f, 0.15f); // rises
        FillDomainLine  (e.sizeDom, 0.18f, 0.18f, 0.f, 0.30f, 0.30f, 0.f);     // 0.18..0.30
        FillDomainBox   (e.colorDom, 0.45f, 0.45f, 0.45f, 0.65f, 0.65f, 0.65f);// grey
        FillDomainLine  (e.rotDom, 0.f, 0.f, 0.f, 6.2831853f, 0.f, 0.f);       // 0..2pi
        e.sc[0] = 0.85f;  // alpha
        e.sc[1] = 0.0f;   // initial age
        e.sc[2] = 0.0f;   // age_sigma
        e.sc[3] = 4.5f;   // lifetime

        u32 n = 0;
        // Gravity — gentle upward buoyancy.
        { ActionRec a = MakeAction(/*GP_GRAVITY*/8); a.a[1] = 0.35f; prog.actions[n++] = a; }
        // Damping — bleed ~30%/s of velocity (whole speed band).
        { ActionRec a = MakeAction(/*GP_DAMPING*/4);
          a.a[0] = a.a[1] = a.a[2] = 0.70f;       // damping
          a.b[0] = 0.0f; a.b[1] = 1.0e16f;        // vlowSqr..vhighSqr
          prog.actions[n++] = a; }
        // TargetSize — grow toward 0.8.
        { ActionRec a = MakeAction(/*GP_TARGETSIZE*/24);
          a.a[0] = a.a[1] = 0.80f;                // target
          a.b[0] = a.b[1] = 0.50f;                // scale/axis
          prog.actions[n++] = a; }
        // TargetColor — fade to dark + transparent over the whole life.
        { ActionRec a = MakeAction(/*GP_TARGETCOLOR*/23);
          a.a[0] = a.a[1] = a.a[2] = 0.20f; a.a[3] = 0.0f;   // colour.rgb, alpha
          a.b[0] = 0.50f; a.b[1] = 0.0f; a.b[2] = 1.0f;       // scale, timeFrom, timeTo
          prog.actions[n++] = a; }
        // Move — integrate position + age.
        { prog.actions[n++] = MakeAction(/*GP_MOVE*/12); }
        // KillOld — kill at lifetime.
        { ActionRec a = MakeAction(/*GP_KILLOLD*/10); a.a[0] = 4.5f; a.a[1] = 0.0f; prog.actions[n++] = a; }

        prog.actionCount = n;
    }

    // Load a program's sprite texture into its bindless slot (append-only) and
    // upload its TexInfo (atlas frame metadata). idx == program registry slot.
    void FillProgramTexture(u32 idx, const TexDesc& td)
    {
        TexInfo ti{};
        ti.layer        = -1;                               // untextured → procedural circle
        ti.flags        = td.flags;
        ti.frameSize[0] = (td.flags & 1u) ? td.frameW : 1.0f;
        ti.frameSize[1] = (td.flags & 1u) ? td.frameH : 1.0f;
        ti.frameDimX    = td.frameDimX  ? td.frameDimX  : 1u;
        ti.frameCount   = td.frameCount ? td.frameCount : 1u;
        ti.frameSpeed   = td.frameSpeed;

        if (s_useTextures && td.name[0]) {
            VkImageView view = VK::ParticlePass::GetTextureView(td.name);
            if (view != VK_NULL_HANDLE) {
                VkDescriptorImageInfo ii{ s_texSampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
                VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
                w.dstSet          = s_texSet;
                w.dstBinding      = 0;
                w.dstArrayElement = idx;
                w.descriptorCount = 1;
                w.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                w.pImageInfo      = &ii;
                vkUpdateDescriptorSets(VulkanHW.m_Device, 1, &w, 0, nullptr);
                ti.layer = (s32)idx;
            } else {
                Msg("~[VK GP] program #%u texture '%s' not loaded — procedural fallback", idx, td.name);
            }
        }

        if (s_texInfo)
            s_texInfo->Upload(&ti, sizeof(ti), VkDeviceSize(idx) * sizeof(TexInfo));
    }
}

// ==========================================================================
// Init
// ==========================================================================
bool Init()
{
    if (s_inited) return !s_failed;
    s_inited = true;

    if (VulkanHW.m_Device == VK_NULL_HANDLE) { s_failed = true; return false; }

    s_maxP = ps_r_gpu_particles_max > 0 ? (u32)ps_r_gpu_particles_max : kDefaultMaxParticles;
    const VkDeviceSize poolSize  = VkDeviceSize(s_maxP) * sizeof(Particle);
    const VkDeviceSize idxSize   = VkDeviceSize(s_maxP) * sizeof(u32);

    // ---- Buffers -----------------------------------------------------------
    s_pool = xr_new<CVulkanBuffer>();
    s_pool->Create(poolSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    if (!s_pool->IsValid()) { Msg("![VK GP] pool alloc failed"); s_failed = true; return false; }
    Prof::NameBuffer(s_pool->GetHandle(), "GP_Pool");

    s_counters = xr_new<CVulkanBuffer>();
    s_counters->Create(64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    if (!s_counters->IsValid()) { Msg("![VK GP] counters alloc failed"); s_failed = true; return false; }
    Prof::NameBuffer(s_counters->GetHandle(), "GP_Counters");

    s_aliveList = xr_new<CVulkanBuffer>();
    s_aliveList->Create(idxSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    if (!s_aliveList->IsValid()) { Msg("![VK GP] aliveList alloc failed"); s_failed = true; return false; }
    Prof::NameBuffer(s_aliveList->GetHandle(), "GP_AliveList");

    s_indirect = xr_new<CVulkanBuffer>();
    s_indirect->Create(64,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    if (!s_indirect->IsValid()) { Msg("![VK GP] indirect alloc failed"); s_failed = true; return false; }
    Prof::NameBuffer(s_indirect->GetHandle(), "GP_Indirect");

    s_freeList = xr_new<CVulkanBuffer>();
    s_freeList->Create(idxSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    if (!s_freeList->IsValid()) { Msg("![VK GP] freeList alloc failed"); s_failed = true; return false; }
    Prof::NameBuffer(s_freeList->GetHandle(), "GP_FreeList");

    // Program registry: Program[kMaxPrograms] — host-visible, slots written
    // on demand (slot 0 = campfire now, real .pe via gp_mirror).
    s_program = xr_new<CVulkanBuffer>();
    s_program->Create(VkDeviceSize(kMaxPrograms) * sizeof(Program),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
    if (!s_program->IsValid()) { Msg("![VK GP] program alloc failed"); s_failed = true; return false; }
    Prof::NameBuffer(s_program->GetHandle(), "GP_Program");
    {
        Program prog;
        BuildTestProgram(prog);
        s_program->Upload(&prog, sizeof(prog), 0);   // slot 0
        s_progName[0]   = "<campfire>";
        s_progRate[0]   = prog.emitRate;
        s_progCount     = 1;
        Msg("[VK GPUParticles] program #0 (campfire): %u actions, emitRate %.0f/s, life %.1fs (registry cap %u)",
            prog.actionCount, prog.emitRate, prog.emit.sc[3], kMaxPrograms);
    }

    // Camera-pinned emitter [0] — OFF by default (it would dump smoke on the
    // player and hide everything). `gp_mirror <effect>` arms it for testing;
    // real world emitters (Phase 3 #3) drive themselves regardless.
    s_emitters.clear();
    { Emitter cam; cam.program = 0; cam.followCamera = true; cam.enabled = false; s_emitters.push_back(cam); }

    // Per-frame spawn-request buffers (host-visible, one per frame in flight so
    // the CPU never overwrites a buffer the GPU is still reading).
    for (u32 i = 0; i < kFramesInFlight; ++i) {
        s_spawnReq[i] = xr_new<CVulkanBuffer>();
        s_spawnReq[i]->Create(VkDeviceSize(kMaxEmitters) * sizeof(SpawnRequest),
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
        if (!s_spawnReq[i]->IsValid()) { Msg("![VK GP] spawnReq alloc failed"); s_failed = true; return false; }
        Prof::NameBuffer(s_spawnReq[i]->GetHandle(), "GP_SpawnReq");
    }

    // Per-program TexInfo (atlas frame metadata, read by the billboard VS).
    s_texInfo = xr_new<CVulkanBuffer>();
    s_texInfo->Create(VkDeviceSize(kMaxPrograms) * sizeof(TexInfo),
                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
    if (!s_texInfo->IsValid()) { Msg("![VK GP] texInfo alloc failed"); s_failed = true; return false; }
    Prof::NameBuffer(s_texInfo->GetHandle(), "GP_TexInfo");

    // ---- Bindless sprite texture array (only if descriptor indexing exists) -
    s_useTextures = VulkanHW.m_bBindlessSupported;
    if (s_useTextures) {
        VkSamplerCreateInfo si{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        si.magFilter = si.minFilter = VK_FILTER_LINEAR;
        si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        si.maxLod = VK_LOD_CLAMP_NONE;
        if (vkCreateSampler(VulkanHW.m_Device, &si, nullptr, &s_texSampler) != VK_SUCCESS)
        { Msg("![VK GP] tex sampler failed"); s_useTextures = false; }
    }
    if (s_useTextures) {
        VkDescriptorSetLayoutBinding b{};
        b.binding         = 0;
        b.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        b.descriptorCount = kMaxPrograms;
        b.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorBindingFlags bf = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;
        VkDescriptorSetLayoutBindingFlagsCreateInfo bfci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO };
        bfci.bindingCount  = 1;
        bfci.pBindingFlags = &bf;

        VkDescriptorSetLayoutCreateInfo slci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        slci.pNext        = &bfci;
        slci.bindingCount = 1;
        slci.pBindings    = &b;
        if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &slci, nullptr, &s_texSetLayout) != VK_SUCCESS)
        { Msg("![VK GP] tex set layout failed"); s_useTextures = false; }

        if (s_useTextures) {
            VkDescriptorPoolSize tps{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kMaxPrograms };
            VkDescriptorPoolCreateInfo tpci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            tpci.maxSets       = 1;
            tpci.poolSizeCount = 1;
            tpci.pPoolSizes    = &tps;
            if (vkCreateDescriptorPool(VulkanHW.m_Device, &tpci, nullptr, &s_texPool) != VK_SUCCESS)
            { Msg("![VK GP] tex pool failed"); s_useTextures = false; }
        }
        if (s_useTextures) {
            VkDescriptorSetAllocateInfo dai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            dai.descriptorPool     = s_texPool;
            dai.descriptorSetCount = 1;
            dai.pSetLayouts        = &s_texSetLayout;
            if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, &s_texSet) != VK_SUCCESS)
            { Msg("![VK GP] tex set alloc failed"); s_useTextures = false; }
        }
    }
    Msg("[VK GPUParticles] sprite textures: %s", s_useTextures ? "BINDLESS" : "procedural (no descriptor indexing)");

    // Campfire (slot 0) is untextured → procedural circle.
    { TexDesc td{}; FillProgramTexture(0, td); }

    // ---- Descriptor set layout (5 SSBO bindings) ---------------------------
    VkDescriptorSetLayoutBinding bindings[kNumBindings]{};
    for (u32 i = 0; i < kNumBindings; ++i) {
        bindings[i].binding         = i;
        bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    // pool(0) + aliveList(2) + texInfo(7) are also read by the billboard VS.
    bindings[0].stageFlags |= VK_SHADER_STAGE_VERTEX_BIT;
    bindings[2].stageFlags |= VK_SHADER_STAGE_VERTEX_BIT;
    bindings[7].stageFlags |= VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo slci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    slci.bindingCount = kNumBindings;
    slci.pBindings    = bindings;
    if (vkCreateDescriptorSetLayout(VulkanHW.m_Device, &slci, nullptr, &s_setLayout) != VK_SUCCESS)
    { Msg("![VK GP] set layout failed"); s_failed = true; return false; }

    VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kFramesInFlight * kNumBindings };
    VkDescriptorPoolCreateInfo pci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets       = kFramesInFlight;
    pci.poolSizeCount = 1;
    pci.pPoolSizes    = &ps;
    if (vkCreateDescriptorPool(VulkanHW.m_Device, &pci, nullptr, &s_poolDesc) != VK_SUCCESS)
    { Msg("![VK GP] desc pool failed"); s_failed = true; return false; }

    {
        VkDescriptorSetLayout layouts[kFramesInFlight];
        for (u32 i = 0; i < kFramesInFlight; ++i) layouts[i] = s_setLayout;
        VkDescriptorSetAllocateInfo dai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        dai.descriptorPool     = s_poolDesc;
        dai.descriptorSetCount = kFramesInFlight;
        dai.pSetLayouts        = layouts;
        if (vkAllocateDescriptorSets(VulkanHW.m_Device, &dai, s_set) != VK_SUCCESS)
        { Msg("![VK GP] set alloc failed"); s_failed = true; return false; }
    }

    // ---- Pipeline layouts --------------------------------------------------
    {
        VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ComputePush) };
        VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        plci.setLayoutCount         = 1;
        plci.pSetLayouts            = &s_setLayout;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcr;
        if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_compLayout) != VK_SUCCESS)
        { Msg("![VK GP] comp layout failed"); s_failed = true; return false; }
    }
    {
        VkPushConstantRange pcr{ VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(DrawPush) };
        VkDescriptorSetLayout sets[2] = { s_setLayout, s_texSetLayout };
        VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        plci.setLayoutCount         = s_useTextures ? 2u : 1u;  // set1 = bindless sprites
        plci.pSetLayouts            = sets;
        plci.pushConstantRangeCount = 1;
        plci.pPushConstantRanges    = &pcr;
        if (vkCreatePipelineLayout(VulkanHW.m_Device, &plci, nullptr, &s_drawLayout) != VK_SUCCESS)
        { Msg("![VK GP] draw layout failed"); s_failed = true; return false; }
    }

    // ---- Shaders -----------------------------------------------------------
    VkShaderModule modInit  = LoadShader("gp_init.comp.spv");
    VkShaderModule modReset = LoadShader("gp_reset.comp.spv");
    VkShaderModule modEmit  = LoadShader("gp_emit.comp.spv");
    VkShaderModule modSim   = LoadShader("gp_simulate.comp.spv");
    VkShaderModule modBuild = LoadShader("gp_build.comp.spv");
    VkShaderModule modVS    = LoadShader("gp_particle.vert.spv");
    VkShaderModule modFS    = LoadShader(s_useTextures ? "gp_particle_tex.frag.spv" : "gp_particle.frag.spv");
    if (!modInit || !modReset || !modEmit || !modSim || !modBuild || !modVS || !modFS)
    { s_failed = true; return false; }

    s_pipeInit  = CreateComputePipeline(modInit,  s_compLayout);
    s_pipeReset = CreateComputePipeline(modReset, s_compLayout);
    s_pipeEmit  = CreateComputePipeline(modEmit,  s_compLayout);
    s_pipeSim   = CreateComputePipeline(modSim,   s_compLayout);
    s_pipeBuild = CreateComputePipeline(modBuild, s_compLayout);
    if (!s_pipeInit || !s_pipeReset || !s_pipeEmit || !s_pipeSim || !s_pipeBuild)
    { Msg("![VK GP] compute pipeline creation failed"); s_failed = true; return false; }

    // ---- Graphics pipeline (additive-free alpha billboards) ----------------
    {
        VkPipelineShaderStageCreateInfo ss[2]{};
        ss[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ss[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;   ss[0].module = modVS; ss[0].pName = "main";
        ss[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ss[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT; ss[1].module = modFS; ss[1].pName = "main";

        VkPipelineVertexInputStateCreateInfo vi{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };

        VkPipelineInputAssemblyStateCreateInfo ia{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vp{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
        vp.viewportCount = 1; vp.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rs{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode    = VK_CULL_MODE_NONE;
        rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rs.lineWidth   = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
        ds.depthTestEnable  = VK_TRUE;
        ds.depthWriteEnable = VK_FALSE;
        ds.depthCompareOp   = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineColorBlendAttachmentState ba{};
        ba.blendEnable         = VK_TRUE;
        ba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        ba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        ba.colorBlendOp        = VK_BLEND_OP_ADD;
        ba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        ba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        ba.alphaBlendOp        = VK_BLEND_OP_ADD;
        ba.colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                 VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

        VkPipelineColorBlendStateCreateInfo cb{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
        cb.attachmentCount = 1;
        cb.pAttachments    = &ba;

        VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynState{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
        dynState.dynamicStateCount = 2;
        dynState.pDynamicStates    = dyn;

        VkFormat colorFormat = VK::SceneColor::Format();
        VkPipelineRenderingCreateInfo prci{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
        prci.colorAttachmentCount    = 1;
        prci.pColorAttachmentFormats = &colorFormat;
        prci.depthAttachmentFormat   = Swapchain.m_DepthFormat;

        VkGraphicsPipelineCreateInfo pi{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
        pi.pNext               = &prci;
        pi.stageCount          = 2;
        pi.pStages             = ss;
        pi.pVertexInputState   = &vi;
        pi.pInputAssemblyState = &ia;
        pi.pViewportState      = &vp;
        pi.pRasterizationState = &rs;
        pi.pMultisampleState   = &ms;
        pi.pDepthStencilState  = &ds;
        pi.pColorBlendState    = &cb;
        pi.pDynamicState       = &dynState;
        pi.layout              = s_drawLayout;

        if (vkCreateGraphicsPipelines(VulkanHW.m_Device, VK::PipelineCache::GetCacheObject(),
                                      1, &pi, nullptr, &s_pipeDraw) != VK_SUCCESS)
        { Msg("![VK GP] draw pipeline failed"); s_failed = true; return false; }
    }

    // ---- Descriptor writes -------------------------------------------------
    {
        for (u32 i = 0; i < kFramesInFlight; ++i) {
            // binding 6 (spawn requests) is per-frame; the rest are shared.
            VkDescriptorBufferInfo bi[kNumBindings] = {
                { s_pool->GetHandle(),       0, VK_WHOLE_SIZE },
                { s_counters->GetHandle(),   0, VK_WHOLE_SIZE },
                { s_aliveList->GetHandle(),  0, VK_WHOLE_SIZE },
                { s_indirect->GetHandle(),   0, VK_WHOLE_SIZE },
                { s_freeList->GetHandle(),   0, VK_WHOLE_SIZE },
                { s_program->GetHandle(),    0, VK_WHOLE_SIZE },
                { s_spawnReq[i]->GetHandle(),0, VK_WHOLE_SIZE },
                { s_texInfo->GetHandle(),    0, VK_WHOLE_SIZE },
            };
            VkWriteDescriptorSet w[kNumBindings]{};
            for (u32 b = 0; b < kNumBindings; ++b) {
                w[b].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[b].dstSet          = s_set[i];
                w[b].dstBinding      = b;
                w[b].descriptorCount = 1;
                w[b].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                w[b].pBufferInfo     = &bi[b];
            }
            vkUpdateDescriptorSets(VulkanHW.m_Device, kNumBindings, w, 0, nullptr);
        }
    }

    s_poolInited = false;   // gp_init runs lazily on the first dispatch
    Msg("[VK GPUParticles] init OK — %u particles (%llu MB pool), r_gpu_particles %d",
        s_maxP, (unsigned long long)(poolSize >> 20), ps_r_gpu_particles);
    return true;
}

// ==========================================================================
// Program registry — resolve an effect name to a slot, translating on demand.
// Returns the slot index, or -1 on failure. Empty/null name → slot 0 (campfire).
// Public (the world-emitter feed in vk_gpu_particles_feed.cpp calls it).
// ==========================================================================
int ResolveProgram(const char* name)
{
    if (!s_inited || s_failed || !s_program) return -1;
    if (!name || !name[0]) return 0;             // campfire

    // Already registered? Reuse the slot (no re-translate/upload).
    for (u32 i = 1; i < s_progCount; ++i)
        if (s_progName[i].size() && 0 == strcmp(s_progName[i].c_str(), name))
            return (int)i;

    if (s_progCount >= kMaxPrograms) {
        Msg("![VK GP] registry full (%u programs) — can't add '%s'", kMaxPrograms, name);
        return -1;
    }

    Program prog;
    float   rate = 0.0f;
    TexDesc tex{};
    if (!TranslateEffect(name, prog, rate, tex))  // logs its own failure reason
        return -1;

    const u32 idx = s_progCount++;
    s_progName[idx] = name;
    s_progRate[idx] = rate;
    s_progLife[idx] = prog.emit.sc[3];            // lifetime (sc.w) for budget gating
    s_program->Upload(&prog, sizeof(prog), VkDeviceSize(idx) * sizeof(Program));
    FillProgramTexture(idx, tex);                 // bindless slot + atlas TexInfo
    Msg("[VK GPUParticles] registered '%s' as program #%u (emitRate %.0f/s, life %.1fs, tex '%s')",
        name, idx, rate, s_progLife[idx], tex.name[0] ? tex.name : "<none>");
    return (int)idx;
}

bool  Enabled()              { return s_inited && !s_failed && ps_r_gpu_particles != 0; }
float ProgramRate(int slot)  { return (slot >= 0 && (u32)slot < s_progCount) ? s_progRate[slot] : 0.0f; }

// Safe to auto-route onto the shared GPU pool? (rate × life within budget.)
bool ProgramRoutable(int slot)
{
    if (slot < 0 || (u32)slot >= s_progCount) return false;
    return s_progRate[slot] * s_progLife[slot] <= kRouteBudget;
}

// ==========================================================================
// MirrorEffect — retarget the camera-pinned emitter (console gp_mirror)
// ==========================================================================
bool MirrorEffect(const char* name)
{
    if (!s_inited || s_failed || !s_program || s_emitters.empty()) {
        Msg("![VK GP] gp_mirror: GPU particles not initialised — set r_gpu_particles 1 first");
        return false;
    }

    // No arg → turn the camera test emitter OFF (don't dump smoke on the player).
    if (!name || !name[0]) {
        s_emitters[0].enabled = false;
        s_emitters[0].accum   = 0.0f;
        Msg("[VK GPUParticles] gp_mirror: camera emitter OFF (no effect arg)");
        return true;
    }

    const int slot = ResolveProgram(name);
    if (slot < 0) return false;

    s_emitters[0].program = (u32)slot;
    s_emitters[0].enabled = true;
    s_emitters[0].accum   = 0.0f;
    Msg("[VK GPUParticles] gp_mirror: camera emitter -> program #%d (%s), ON",
        slot, name);
    return true;
}

// ==========================================================================
// SpawnEffect / ClearSpawns — persistent world emitters (console gp_spawn)
// ==========================================================================
bool SpawnEffect(const char* name)
{
    if (!s_inited || s_failed || !s_program) {
        Msg("![VK GP] gp_spawn: GPU particles not initialised — set r_gpu_particles 1 first");
        return false;
    }
    if (!name || !name[0]) {
        Msg("![VK GP] gp_spawn: usage 'gp_spawn <effect>' (see gp_list)");
        return false;
    }
    if (s_emitters.size() >= kMaxEmitters) {
        Msg("![VK GP] gp_spawn: emitter cap (%u) reached — gp_spawn_clear first", kMaxEmitters);
        return false;
    }

    const int slot = ResolveProgram(name);
    if (slot < 0) return false;

    Emitter em;
    em.program      = (u32)slot;
    em.followCamera = false;
    em.pos          = Device.vCameraPosition;
    em.pos.y       -= 0.5f;
    s_emitters.push_back(em);
    Msg("[VK GPUParticles] gp_spawn: '%s' (program #%d) at (%.1f %.1f %.1f) — %u world emitter(s)",
        name, slot, em.pos.x, em.pos.y, em.pos.z, (u32)s_emitters.size() - 1);
    return true;
}

void ClearSpawns()
{
    if (s_emitters.size() > 1) s_emitters.resize(1);   // keep the camera emitter
    Msg("[VK GPUParticles] gp_spawn_clear: world emitters removed");
}

// ==========================================================================
// DispatchComputeAndDraw — init(once) → reset → emit → sim → build → draw
// ==========================================================================
void DispatchComputeAndDraw(FrameContext& ctx)
{
    if (!s_inited || s_failed || ps_r_gpu_particles == 0) return;
    if (ctx.cmd == VK_NULL_HANDLE) return;

    const u32 slot = CommandManager.GetCurrentFrame() % kFramesInFlight;
    VkCommandBuffer cmd = ctx.cmd;
    const u32 maxP = s_maxP;

    auto bindCompute = [&](VkPipeline pipe) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, s_compLayout, 0, 1, &s_set[slot], 0, nullptr);
    };

    // ---- Gather emitter samples (camera + gp_spawn + real world .pe) ------
    const float dt = Device.fTimeDelta;
    static xr_vector<EmitterSample> samples;
    samples.clear();
    Fvector camBelow = Device.vCameraPosition; camBelow.y -= 0.5f;

    // (a) camera-pinned + gp_spawn world emitters (persistent accum in Emitter).
    for (Emitter& em : s_emitters) {
        if (!em.enabled) continue;
        em.accum += s_progRate[em.program] * dt;
        u32 cnt = (u32)em.accum;
        em.accum -= (float)cnt;                  // keep the fraction
        if (cnt == 0) continue;
        EmitterSample es;
        es.program = em.program;
        es.count   = cnt;
        es.pos     = em.followCamera ? camBelow : em.pos;
        es.vel.set(0, 0, 0);
        samples.push_back(es);
    }

    // (b) real .pe emitters playing in the world (pull-based, render thread —
    //     same g_DynamicVisuals walk CollectSmokeParticles already does).
    VK_GP_CollectWorldEmitters(samples, dt);

    // ---- Convert samples → dense spawn-request gid ranges -----------------
    // Each request claims [firstSlot, firstSlot+count); the emit shader walks
    // them to spawn the right program at the right world position.
    SpawnRequest reqs[kMaxEmitters];
    u32 numReq = 0, emitCount = 0;
    for (const EmitterSample& es : samples) {
        if (emitCount >= kMaxEmitPerFrame || numReq >= kMaxEmitters) break;
        u32 cnt = es.count;
        if (cnt > kMaxEmitPerEmitter) cnt = kMaxEmitPerEmitter;
        if (emitCount + cnt > kMaxEmitPerFrame) cnt = kMaxEmitPerFrame - emitCount;
        if (cnt == 0) continue;

        SpawnRequest& sr = reqs[numReq];
        sr.pos[0] = es.pos.x; sr.pos[1] = es.pos.y; sr.pos[2] = es.pos.z; sr.pos[3] = 0.0f;
        sr.vel[0] = es.vel.x; sr.vel[1] = es.vel.y; sr.vel[2] = es.vel.z; sr.vel[3] = 0.0f;
        sr.program   = es.program;
        sr.count     = cnt;
        sr.firstSlot = emitCount;
        sr.seed      = Device.dwFrame * 2654435761u + (numReq + 1u) * 2246822519u;
        emitCount += cnt;
        ++numReq;
    }
    if (numReq > 0)
        s_spawnReq[slot]->Upload(reqs, VkDeviceSize(numReq) * sizeof(SpawnRequest), 0);

    // Shared compute push.
    ComputePush push{};
    push.spawnPos[0] = camBelow.x;               // reserved (emit reads request pos)
    push.spawnPos[1] = camBelow.y;
    push.spawnPos[2] = camBelow.z;
    memcpy(&push.spawnPos[3], &emitCount, 4);
    push.dt_gravity[0] = dt;
    push.dt_gravity[1] = 9.8f;
    push.maxParticles  = maxP;
    push.frameSeed     = Device.dwFrame;
    push.activeProgram = 0;                       // reserved
    push.numRequests   = numReq;

    // ---- One-time pool init ------------------------------------------------
    if (!s_poolInited) {
        const int z = Prof::ZoneBegin(cmd, "GP::Init");
        bindCompute(s_pipeInit);
        vkCmdPushConstants(cmd, s_compLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(cmd, (maxP + 63) / 64, 1, 1);
        MemBarrier(cmd);
        Prof::ZoneEnd(cmd, z);
        s_poolInited = true;
    }

    // ---- Reset alive counter ----------------------------------------------
    {
        const int z = Prof::ZoneBegin(cmd, "GP::Reset");
        bindCompute(s_pipeReset);
        vkCmdPushConstants(cmd, s_compLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(cmd, 1, 1, 1);
        Prof::ZoneEnd(cmd, z);
    }

    // ---- Emit (free-list pop) ---------------------------------------------
    if (emitCount > 0) {
        const int z = Prof::ZoneBegin(cmd, "GP::Emit");
        bindCompute(s_pipeEmit);
        vkCmdPushConstants(cmd, s_compLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(cmd, (emitCount + 63) / 64, 1, 1);
        Prof::ZoneEnd(cmd, z);
    }

    MemBarrier(cmd);   // emit/reset writes → sim reads

    // ---- Simulate ----------------------------------------------------------
    {
        const int z = Prof::ZoneBegin(cmd, "GP::Sim");
        bindCompute(s_pipeSim);
        vkCmdPushConstants(cmd, s_compLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(cmd, (maxP + 63) / 64, 1, 1);
        Prof::ZoneEnd(cmd, z);
    }

    MemBarrier(cmd);   // sim writes aliveCount → build reads

    // ---- Build indirect ----------------------------------------------------
    {
        const int z = Prof::ZoneBegin(cmd, "GP::Build");
        bindCompute(s_pipeBuild);
        vkCmdPushConstants(cmd, s_compLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(cmd, 1, 1, 1);

        VkMemoryBarrier mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                             0, 1, &mb, 0, nullptr, 0, nullptr);
        Prof::ZoneEnd(cmd, z);
    }

    // ---- Draw billboards ---------------------------------------------------
    {
        const int z = Prof::ZoneBegin(cmd, "GP::Draw");

        VkViewport vp{ 0.0f, (float)ctx.extent.height, (float)ctx.extent.width,
                       -(float)ctx.extent.height, 0.0f, 1.0f };
        vkCmdSetViewport(cmd, 0, 1, &vp);
        VkRect2D sc{ {}, ctx.extent };
        vkCmdSetScissor(cmd, 0, 1, &sc);

        VkRenderingAttachmentInfo cAtt{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        cAtt.imageView   = ctx.colorView;
        cAtt.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        cAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;
        cAtt.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;

        VkRenderingAttachmentInfo dAtt{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        dAtt.imageView   = ctx.depthView;
        dAtt.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        dAtt.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;
        dAtt.storeOp     = VK_ATTACHMENT_STORE_OP_DONT_CARE;

        VkRenderingInfo ri{ VK_STRUCTURE_TYPE_RENDERING_INFO };
        ri.renderArea.extent    = ctx.extent;
        ri.layerCount           = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments    = &cAtt;
        ri.pDepthAttachment     = &dAtt;

        vkCmdBeginRendering(cmd, &ri);

        DrawPush dpush{};
        static_assert(sizeof(Fmatrix) == 64, "Fmatrix 64 B");
        memcpy(dpush.viewProj, &Device.mFullTransform, 64);
        dpush.camRight[0] = Device.vCameraRight.x;
        dpush.camRight[1] = Device.vCameraRight.y;
        dpush.camRight[2] = Device.vCameraRight.z;
        dpush.camUp[0]    = Device.vCameraTop.x;
        dpush.camUp[1]    = Device.vCameraTop.y;
        dpush.camUp[2]    = Device.vCameraTop.z;

        vkCmdPushConstants(cmd, s_drawLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(dpush), &dpush);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_pipeDraw);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_drawLayout, 0, 1, &s_set[slot], 0, nullptr);
        if (s_useTextures && s_texSet)
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, s_drawLayout, 1, 1, &s_texSet, 0, nullptr);
        vkCmdDrawIndirect(cmd, s_indirect->GetHandle(), 0, 1, sizeof(VkDrawIndirectCommand));

        vkCmdEndRendering(cmd);
        Prof::ZoneEnd(cmd, z);
    }
}

// ==========================================================================
// Destroy
// ==========================================================================
void Destroy()
{
    if (VulkanHW.m_Device == VK_NULL_HANDLE) return;
    DestroyPipelines();
    DestroyBuffers();
    s_inited     = false;
    s_failed     = false;
    s_poolInited = false;
    for (u32 i = 0; i < kMaxPrograms; ++i) { s_progName[i] = shared_str(); s_progRate[i] = 0.0f; }
    s_progCount = 0;
    s_emitters.clear();
    Msg("[VK GPUParticles] destroyed");
}

}}  // namespace VK::GPUParticles
