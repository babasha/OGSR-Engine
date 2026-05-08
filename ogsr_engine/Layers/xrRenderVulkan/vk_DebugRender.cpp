// xrRenderVulkan - Stub IDebugRender. Debug primitives are no-ops until the
// world rendering pipeline lands.

#include "stdafx.h"
#include "vk_core.h"
#include "../../Include/xrRender/DebugRender.h"

namespace {
class vkDebugRender final : public IDebugRender
{
public:
    void Render() override {}

    void add_lines(Fvector const* /*vertices*/, u32 const& /*vertex_count*/,
                   u16 const* /*pairs*/, u32 const& /*pair_count*/,
                   u32 const& /*color*/, bool /*hud_mode*/) override {}

    void OnFrameEnd() override {}
    void SetShader(const debug_shader&) override {}
    void CacheSetXformWorld(const Fmatrix&) override {}
    void CacheSetCullMode(CullMode) override {}

    void SetDebugShader(dbgShaderHandle) override {}
    void DestroyDebugShader(dbgShaderHandle) override {}

    void dbg_DrawTRI(Fmatrix&, Fvector&, Fvector&, Fvector&, u32) override {}
};

vkDebugRender g_DebugRenderImpl_VK;
} // namespace

IDebugRender* GetDebugRenderImpl_VK() { return &g_DebugRenderImpl_VK; }
