// xrRenderVulkan — LOD imposter manager (MT_LOD / FLOD billboards).
//
// Collects every vkFLOD at level load and, each frame, draws the single
// best-facing billboard facet of each (distant) FLOD into the level_lods atlas.
// This restores the far-forest silhouette density R4 gets from
// r_dsgraph_render_lods(). v1: nearest-facet only (no 2-facet cross-fade),
// distance-gated so up-close trees still use their full mesh.

#pragma once
#include "vk_core.h"

class vkFLOD;

namespace VK
{
class CVulkanBuffer;
class CVulkanTexture;
struct FrameContext;

// Per-vertex imposter quad data (host-built each frame). 24 B.
struct LodImposterVertex
{
    Fvector  pos;     // world-space position (facet vertex + camera-ward shift)
    Fvector2 uv;      // UV into level_lods atlas
    u32      color;   // RGBA8 tint (hemi lighting; white for v1)
};
static_assert(sizeof(LodImposterVertex) == 24, "LodImposterVertex must be 24 B");

inline constexpr u32 LOD_FRAMES = VK_FRAMES_IN_FLIGHT;   // == CommandManager FRAMES_IN_FLIGHT

class CLODManager
{
public:
    CLODManager();
    ~CLODManager();

    void Build();                       // collect FLODs + atlas + pipeline
    void Destroy();
    void Render(VK::FrameContext& ctx); // per-frame best-facet billboard draw
    bool IsReady() const;

private:
    void CreatePipeline();
    void CreateAtlasDescriptor();

private:
    bool m_bBuilt = false;
    xr_vector<vkFLOD*> m_Lods;

    // Triple-buffered host-visible vertex stream (6 verts per FLOD quad).
    CVulkanBuffer* m_DynVB[LOD_FRAMES] = {};
    u32            m_MaxVerts = 0;

    // Atlas (level_lods) descriptor. The atlas lives in $level$, not
    // $game_textures$, so it's loaded directly (not via WorldMaterialCache).
    CVulkanTexture*       m_AtlasTex   = nullptr;
    VkDescriptorSetLayout m_DescLayout = VK_NULL_HANDLE;
    VkDescriptorPool      m_DescPool   = VK_NULL_HANDLE;
    VkDescriptorSet       m_DescSet    = VK_NULL_HANDLE;
    VkSampler             m_Sampler    = VK_NULL_HANDLE;

    VkPipelineLayout m_PipelineLayout = VK_NULL_HANDLE;
    VkPipeline       m_Pipeline       = VK_NULL_HANDLE;
};

}  // namespace VK
