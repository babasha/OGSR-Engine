// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
// Licensed under the same terms as X-Ray Engine (see root License.txt)

#pragma once
#include "../../Include/xrRender/UIShader.h"
#include <vulkan/vulkan.h>

/**
 * Extended UI Shader interface for Vulkan
 * Provides access to Vulkan-specific resources (descriptor sets, textures)
 *
 * Video methods exist so vkUISequenceVideoItem can forward CTheoraSurface
 * playback control to whichever shader currently owns the .ogm-backed
 * texture. Default impls are no-ops — only Theora-loaded shaders override.
 */
class IVkUIShader : public IUIShader
{
public:
    virtual VkDescriptorSet GetDescriptorSet() = 0;
    virtual u32 GetTextureWidth() const = 0;
    virtual u32 GetTextureHeight() const = 0;

    virtual bool HasVideo() const                { return false; }
    virtual void VideoPlay(BOOL /*looped*/, u32 /*time*/) {}
    virtual void VideoSync(u32 /*time*/)         {}
    virtual void VideoStop()                     {}
    virtual BOOL VideoIsPlaying()                { return FALSE; }
};

// Set by vkUIRender::SetShader; consumed by vkUISequenceVideoItem::CaptureTexture.
// Mirrors R4's RCache.get_ActiveTexture(0) — the video item asks "which UI
// shader was just bound for me to attach my Theora playback control to?".
extern IVkUIShader* g_pLastVkUIShader;
