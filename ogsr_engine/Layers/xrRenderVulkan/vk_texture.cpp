// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
// Licensed under the same terms as X-Ray Engine (see root License.txt)

#include "stdafx.h"
#include "vk_texture.h"
#include "vk_buffer.h"
#include "vk_command_buffer.h"
#include "HW_Vulkan.h"

// DDS definitions
const u32 DDS_MAGIC = 0x20534444; // "DDS "

struct DDS_PIXELFORMAT {
    u32 dwSize;
    u32 dwFlags;
    u32 dwFourCC;
    u32 dwRGBBitCount;
    u32 dwRBitMask;
    u32 dwGBitMask;
    u32 dwBBitMask;
    u32 dwABitMask;
};

struct DDS_HEADER {
    u32 dwSize;
    u32 dwFlags;
    u32 dwHeight;
    u32 dwWidth;
    u32 dwPitchOrLinearSize;
    u32 dwDepth;
    u32 dwMipMapCount;
    u32 dwReserved1[11];
    DDS_PIXELFORMAT ddspf;
    u32 dwCaps;
    u32 dwCaps2;
    u32 dwCaps3;
    u32 dwCaps4;
    u32 dwReserved2;
};

// FourCC codes
const u32 FOURCC_DXT1 = 0x31545844; // "DXT1"
const u32 FOURCC_DXT3 = 0x33545844; // "DXT3"
const u32 FOURCC_DXT5 = 0x35545844; // "DXT5"

// Flags
const u32 DDPF_ALPHAPIXELS = 0x1;
const u32 DDPF_ALPHA       = 0x2;
const u32 DDPF_FOURCC      = 0x4;
const u32 DDPF_RGB         = 0x40;
const u32 DDPF_LUMINANCE   = 0x20000;

namespace VK
{

// Constructor
CVulkanTexture::CVulkanTexture()
{
}

// Destructor
CVulkanTexture::~CVulkanTexture()
{
    Destroy();
}

// Создание пустой текстуры
void CVulkanTexture::Create(u32 width, u32 height, VkFormat format, u32 mipLevels,
                            VkImageUsageFlags usage)
{
    if (m_Image != VK_NULL_HANDLE) {
        Msg("![Vulkan] Texture already created, call Destroy first");
        return;
    }

    if (width == 0 || height == 0) {
        Msg("![Vulkan] Cannot create texture with size 0");
        return;
    }

    m_Width = width;
    m_Height = height;
    m_Format = format;
    m_MipLevels = mipLevels > 0 ? mipLevels : 1;

    // Image create info
    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = m_MipLevels;
    imageInfo.arrayLayers = m_ArrayLayers;
    imageInfo.format = format;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.flags = m_bCubemap ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;

    // Upload-target images are filled on the dedicated TRANSFER queue and sampled
    // on GRAPHICS. With distinct families that's cross-family access — CONCURRENT
    // sharing lets the contents (and the transfer-side layout transition) survive
    // without queue-ownership-transfer barriers, which the transfer queue can't
    // express for the final SHADER_READ transition anyway (no FRAGMENT_SHADER
    // stage). Sampled textures aren't DCC-compressed, so the cost is negligible.
    // Only for TRANSFER_DST images when the families actually differ.
    const u32 imgFamilies[2] = { VulkanHW.m_GraphicsFamily, VulkanHW.m_TransferFamily };
    if ((usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) &&
        VulkanHW.m_TransferFamily != VulkanHW.m_GraphicsFamily) {
        imageInfo.sharingMode           = VK_SHARING_MODE_CONCURRENT;
        imageInfo.queueFamilyIndexCount = 2;
        imageInfo.pQueueFamilyIndices   = imgFamilies;
    }

    // VMA allocation info - prefer device local memory
    VmaAllocationCreateInfo allocInfo = {};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    VK_CHECK(vmaCreateImage(VulkanHW.m_Allocator, &imageInfo, &allocInfo,
                            &m_Image, &m_Allocation, nullptr));

    m_CurrentLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    // Создаём view и sampler
    CreateImageView();
    CreateSampler();

    // Msg("[Vulkan] Texture created: %dx%d, format=%d, mips=%d", width, height, format, m_MipLevels);
}

// Создание текстуры из данных
void CVulkanTexture::CreateFromData(const void* data, u32 width, u32 height, VkFormat format,
                                    VkDeviceSize dataSize)
{
    if (!data) {
        Msg("![Vulkan] CreateFromData: data is null");
        return;
    }

    // Рассчитываем размер данных если не указан
    if (dataSize == 0) {
        if (IsCompressedFormat(format)) {
            // Для compressed форматов нужен явный размер
            Msg("![Vulkan] CreateFromData: dataSize required for compressed format");
            return;
        }
        // Для RGBA8 - 4 bytes per pixel
        dataSize = width * height * 4;
    }

    // Создаём текстуру
    Create(width, height, format, 1);
    if (m_Image == VK_NULL_HANDLE) {
        return;
    }
    if (m_Allocation) {  // leak-dump name: procedural textures have no file name
        char nm[32]; xr_sprintf(nm, "proc:%ux%u", width, height);
        vmaSetAllocationName(VulkanHW.m_Allocator, m_Allocation, nm);
    }

    // Upload данных
    UploadData(data, dataSize);
}

// Upload данных через staging buffer
void CVulkanTexture::UploadData(const void* data, VkDeviceSize size)
{
    // Prepare copy regions for mipmaps (and array layers for cubemaps).
    // DDS cubemap layout: for each face, all mipmaps sequentially. bufferOffsets are
    // relative to `data`; the async uploader rebases them into its staging ring.
    xr_vector<VkBufferImageCopy> regions;
    VkDeviceSize offset = 0;

    for (u32 layer = 0; layer < m_ArrayLayers; layer++) {
        u32 currentWidth = m_Width;
        u32 currentHeight = m_Height;

        for (u32 i = 0; i < m_MipLevels; i++) {
            VkBufferImageCopy region = {};
            region.bufferOffset = offset;
            region.bufferRowLength = 0;   // Tightly packed
            region.bufferImageHeight = 0; // Tightly packed
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.mipLevel = i;
            region.imageSubresource.baseArrayLayer = layer;
            region.imageSubresource.layerCount = 1;
            region.imageOffset = {0, 0, 0};
            region.imageExtent = {currentWidth, currentHeight, 1};

            regions.push_back(region);

            // Calculate size of current mip
            VkDeviceSize currentSize = 0;
            if (IsCompressedFormat(m_Format)) {
                u32 blockSize = GetBlockSize(m_Format);
                u32 blocksX = (currentWidth + 3) / 4;
                u32 blocksY = (currentHeight + 3) / 4;
                currentSize = blocksX * blocksY * blockSize;
            } else {
                // Bytes per pixel depends on format
                u32 bpp = 4; // Default: RGBA8/BGRA8
                if (m_Format == VK_FORMAT_R8_UNORM)
                    bpp = 1;
                else if (m_Format == VK_FORMAT_R8G8_UNORM)
                    bpp = 2;
                currentSize = currentWidth * currentHeight * bpp;
            }

            offset += currentSize;

            // Next mip dimensions
            if (currentWidth > 1) currentWidth /= 2;
            if (currentHeight > 1) currentHeight /= 2;
        }
    }

    // Async upload on the dedicated transfer queue — no per-texture command pool, no
    // vkQueueWaitIdle, no per-upload staging buffer. Leaves the image in
    // SHADER_READ_ONLY_OPTIMAL; the graphics frame's upload-timeline wait
    // (FRAGMENT_SHADER) makes the copy+transition visible to samplers. The image is
    // created CONCURRENT{graphics,transfer} so no queue-ownership transfer is needed.
    CommandManager.UploadImage(m_Image, data, size, regions.data(), (u32)regions.size(),
                               m_MipLevels, m_ArrayLayers);
    m_CurrentLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

// Transition image layout
void CVulkanTexture::TransitionLayout(VkCommandBuffer cmd, VkImageLayout oldLayout, VkImageLayout newLayout)
{
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = m_Image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = m_MipLevels;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = m_ArrayLayers;

    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else {
        Msg("![Vulkan] Unsupported layout transition: %d -> %d", oldLayout, newLayout);
        return;
    }

    vkCmdPipelineBarrier(cmd, sourceStage, destinationStage, 0,
                         0, nullptr, 0, nullptr, 1, &barrier);

    m_CurrentLayout = newLayout;
}

// Transition layout с выделенным immediate command buffer
// Uses dedicated cmd buffer to avoid corrupting the render frame's command buffer.
void CVulkanTexture::TransitionLayoutImmediate(VkImageLayout oldLayout, VkImageLayout newLayout)
{
    VkCommandBuffer cmd = CommandManager.BeginImmediate();
    if (cmd == VK_NULL_HANDLE) {
        Msg("![Vulkan] TransitionLayoutImmediate: failed to begin immediate cmd");
        return;
    }
    TransitionLayout(cmd, oldLayout, newLayout);
    CommandManager.EndAndSubmitImmediate(cmd);
}

// Создание ImageView
void CVulkanTexture::CreateImageView()
{
    VkImageViewCreateInfo viewInfo = {};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = m_Image;
    viewInfo.viewType = m_bCubemap ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = m_Format;
    if (m_bAlphaSwizzle) {
        // Alpha-only texture (fonts): R channel → alpha, RGB = white
        viewInfo.components.r = VK_COMPONENT_SWIZZLE_ONE;
        viewInfo.components.g = VK_COMPONENT_SWIZZLE_ONE;
        viewInfo.components.b = VK_COMPONENT_SWIZZLE_ONE;
        viewInfo.components.a = VK_COMPONENT_SWIZZLE_R;
    } else if (m_bBCSwizzle) {
        // BC/DXT textures: swap R<->B channels
        // DirectX DXT textures use BGRA order, Vulkan BC uses RGBA
        viewInfo.components.r = VK_COMPONENT_SWIZZLE_B;
        viewInfo.components.g = VK_COMPONENT_SWIZZLE_G;
        viewInfo.components.b = VK_COMPONENT_SWIZZLE_R;
        viewInfo.components.a = VK_COMPONENT_SWIZZLE_A;
    } else {
        viewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        viewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    }
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = m_MipLevels;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = m_ArrayLayers;

    VK_CHECK(vkCreateImageView(VulkanHW.m_Device, &viewInfo, nullptr, &m_ImageView));
}

// Создание Sampler
// Using CLAMP_TO_EDGE to match R3/R4 DX behavior (shader:dx10sampler("smp_base"):clamp())
void CVulkanTexture::CreateSampler()
{
    VkSamplerCreateInfo samplerInfo = {};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    if (m_bCubemap) {
        // Cubemaps must use CLAMP_TO_EDGE to avoid seam artifacts
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    } else {
        // Use REPEAT for game textures (walls, terrain tile beyond UV [0,1])
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    }
    samplerInfo.anisotropyEnable = VK_TRUE;
    samplerInfo.maxAnisotropy = 16.0f;  // Max anisotropy
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = static_cast<float>(m_MipLevels);

    VK_CHECK(vkCreateSampler(VulkanHW.m_Device, &samplerInfo, nullptr, &m_Sampler));
}

// Уничтожение текстуры
void CVulkanTexture::Destroy()
{
    if (m_Sampler != VK_NULL_HANDLE) {
        vkDestroySampler(VulkanHW.m_Device, m_Sampler, nullptr);
        m_Sampler = VK_NULL_HANDLE;
    }

    if (m_ImageView != VK_NULL_HANDLE) {
        vkDestroyImageView(VulkanHW.m_Device, m_ImageView, nullptr);
        m_ImageView = VK_NULL_HANDLE;
    }

    if (m_Image != VK_NULL_HANDLE) {
        vmaDestroyImage(VulkanHW.m_Allocator, m_Image, m_Allocation);
        m_Image = VK_NULL_HANDLE;
        m_Allocation = VK_NULL_HANDLE;
    }

    m_Width = 0;
    m_Height = 0;
    m_MipLevels = 1;
    m_CurrentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    m_bCubemap = false;
    m_ArrayLayers = 1;
}

// Загрузка DDS
bool CVulkanTexture::LoadDDS(const char* filename, bool applyBCSwizzle)
{
    IReader* F = FS.r_open(filename);
    if (!F) {
        Msg("![Vulkan] Failed to open texture: %s", filename);
        return false;
    }

    // Check magic
    u32 magic = 0;
    F->r(&magic, 4);
    if (magic != DDS_MAGIC) {
        Msg("![Vulkan] Invalid DDS magic in %s", filename);
        FS.r_close(F);
        return false;
    }

    // Read header
    DDS_HEADER header;
    F->r(&header, sizeof(DDS_HEADER));

    // Determine format
    VkFormat format = VK_FORMAT_UNDEFINED;
    
    if (header.ddspf.dwFlags & DDPF_FOURCC) {
        switch (header.ddspf.dwFourCC) {
            case FOURCC_DXT1:
                // DXT1 always has 1-bit punch-through alpha in X-Ray engine.
                // Many DDS files omit DDPF_ALPHAPIXELS flag but still use alpha.
                // D3D11 always treats DXT1 as having alpha, so we do the same.
                format = VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
                break;
            case FOURCC_DXT3:
                format = VK_FORMAT_BC2_UNORM_BLOCK;
                break;
            case FOURCC_DXT5:
                format = VK_FORMAT_BC3_UNORM_BLOCK;
                break;
            default:
                Msg("![Vulkan] Unsupported FourCC: %X in %s", header.ddspf.dwFourCC, filename);
                FS.r_close(F);
                return false;
        }
        // X-Ray UI atlases ship with BGR-ordered BC endpoints (yellow indicators
        // come out blue without R↔B swap). Level statics are stock BC1/3 with
        // RGB endpoints — `applyBCSwizzle=false` keeps them correct.
        m_bBCSwizzle = applyBCSwizzle;
    } else if (header.ddspf.dwFlags & DDPF_RGB) {
        if (header.ddspf.dwRGBBitCount == 32) {
            // Choose format based on channel masks. D3DFMT_A8R8G8B8 (BGRA-in-
            // memory) uses R=0x00FF0000; D3DFMT_A8B8G8R8 (RGBA-in-memory) uses
            // R=0x000000FF. X-Ray ships UI atlases in the latter convention
            // (caught when ui_common indicators rendered with R↔B swapped).
            if (header.ddspf.dwRBitMask == 0x000000FF) {
                format = VK_FORMAT_R8G8B8A8_UNORM;
            } else {
                format = VK_FORMAT_B8G8R8A8_UNORM;
            }
        } else {
            Msg("![Vulkan] Unsupported RGB bit count: %d in %s", header.ddspf.dwRGBBitCount, filename);
            FS.r_close(F);
            return false;
        }
    } else if (header.ddspf.dwFlags & DDPF_ALPHA) {
        // Alpha-only format (A8) — used by font textures
        // Store as R8, swizzle R→A in image view
        format = VK_FORMAT_R8_UNORM;
        m_bAlphaSwizzle = true;
    } else if (header.ddspf.dwFlags & DDPF_LUMINANCE) {
        // Luminance format (L8 or L8A8)
        if (header.ddspf.dwRGBBitCount == 8) {
            format = VK_FORMAT_R8_UNORM;
        } else if (header.ddspf.dwRGBBitCount == 16) {
            format = VK_FORMAT_R8G8_UNORM;
        } else {
            Msg("![Vulkan] Unsupported luminance bit count: %d in %s", header.ddspf.dwRGBBitCount, filename);
            FS.r_close(F);
            return false;
        }
    } else {
        Msg("![Vulkan] Unsupported DDS format flags: %X in %s", header.ddspf.dwFlags, filename);
        FS.r_close(F);
        return false;
    }

    u32 width = header.dwWidth;
    u32 height = header.dwHeight;
    u32 mipLevels = (header.dwFlags & 0x20000) ? header.dwMipMapCount : 1; // DDSD_MIPMAPCOUNT
    if (mipLevels == 0) mipLevels = 1;

    // Debug: log format for magnifier texture specifically
    if (strstr(filename, "magnifier")) {
        Msg("[Vulkan Texture] MAGNIFIER DETAILED:");
        Msg("  File: %s", filename);
        Msg("  Size: %dx%d, mips=%d", width, height, mipLevels);
        Msg("  Header size: %d", header.dwSize);
        Msg("  Flags: 0x%X", header.dwFlags);
        Msg("  PitchOrLinearSize: %d", header.dwPitchOrLinearSize);
        Msg("  PixelFormat size: %d", header.ddspf.dwSize);
        Msg("  PixelFormat flags: 0x%X", header.ddspf.dwFlags);
        Msg("  FourCC: 0x%X ('%c%c%c%c')", header.ddspf.dwFourCC,
            (char)(header.ddspf.dwFourCC & 0xFF),
            (char)((header.ddspf.dwFourCC >> 8) & 0xFF),
            (char)((header.ddspf.dwFourCC >> 16) & 0xFF),
            (char)((header.ddspf.dwFourCC >> 24) & 0xFF));
        Msg("  RGBBitCount: %d", header.ddspf.dwRGBBitCount);
        Msg("  RMask: 0x%X, GMask: 0x%X, BMask: 0x%X, AMask: 0x%X",
            header.ddspf.dwRBitMask, header.ddspf.dwGBitMask,
            header.ddspf.dwBBitMask, header.ddspf.dwABitMask);
        Msg("  Caps: 0x%X, Caps2: 0x%X", header.dwCaps, header.dwCaps2);
        Msg("  VkFormat: %d", format);
    }

    // Create texture
    Create(width, height, format, mipLevels);
    // Tag the VMA allocation with the source file so any leaked image is
    // identifiable by name in the vmaDestroyAllocator leak dump (see vma_impl.cpp).
    if (m_Allocation) vmaSetAllocationName(VulkanHW.m_Allocator, m_Allocation, filename);

    // Read remaining data
    VkDeviceSize dataSize = F->length() - F->tell();
    
    // Allocate temp buffer
    void* data = xr_malloc(dataSize);
    F->r(data, dataSize);
    FS.r_close(F);

    // Upload
    UploadData(data, dataSize);

    xr_free(data);

    // Msg("[Vulkan] Loaded DDS: %s (%dx%d, mips=%d)", filename, width, height, mipLevels);
    return true;
}

// Загрузка DDS cubemap (6 faces)
bool CVulkanTexture::LoadDDSCubemap(const char* filename, bool applyBCSwizzle)
{
    IReader* F = FS.r_open(filename);
    if (!F) {
        Msg("![Vulkan] Failed to open cubemap texture: %s", filename);
        return false;
    }

    // Check magic
    u32 magic = 0;
    F->r(&magic, 4);
    if (magic != DDS_MAGIC) {
        Msg("![Vulkan] Invalid DDS magic in cubemap %s", filename);
        FS.r_close(F);
        return false;
    }

    // Read header
    DDS_HEADER header;
    F->r(&header, sizeof(DDS_HEADER));

    // Check for cubemap flag
    const u32 DDSCAPS2_CUBEMAP = 0x200;
    if (!(header.dwCaps2 & DDSCAPS2_CUBEMAP)) {
        Msg("![Vulkan] DDS file is not a cubemap: %s (caps2=0x%X)", filename, header.dwCaps2);
        FS.r_close(F);
        return false;
    }

    // Determine format
    // Use UNORM (not SRGB) because swapchain is UNORM - no sRGB conversion in pipeline.
    // This matches D3D11/R4 behavior where textures stay in gamma space throughout.
    VkFormat format = VK_FORMAT_UNDEFINED;

    if (header.ddspf.dwFlags & DDPF_FOURCC) {
        switch (header.ddspf.dwFourCC) {
            case FOURCC_DXT1:
                format = VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
                break;
            case FOURCC_DXT3:
                format = VK_FORMAT_BC2_UNORM_BLOCK;
                break;
            case FOURCC_DXT5:
                format = VK_FORMAT_BC3_UNORM_BLOCK;
                break;
            default:
                Msg("![Vulkan] Unsupported cubemap FourCC: %X in %s", header.ddspf.dwFourCC, filename);
                FS.r_close(F);
                return false;
        }
        // Same R↔B swizzle policy as 2D BC: caller decides.
        m_bBCSwizzle = applyBCSwizzle;
    } else if (header.ddspf.dwFlags & DDPF_RGB) {
        if (header.ddspf.dwRGBBitCount == 32) {
            format = VK_FORMAT_B8G8R8A8_UNORM;
        } else {
            Msg("![Vulkan] Unsupported cubemap RGB bit count: %d in %s", header.ddspf.dwRGBBitCount, filename);
            FS.r_close(F);
            return false;
        }
    } else {
        Msg("![Vulkan] Unsupported cubemap DDS format flags: %X in %s", header.ddspf.dwFlags, filename);
        FS.r_close(F);
        return false;
    }

    u32 width = header.dwWidth;
    u32 height = header.dwHeight;
    u32 mipLevels = (header.dwFlags & 0x20000) ? header.dwMipMapCount : 1;
    if (mipLevels == 0) mipLevels = 1;

    // Setup cubemap flags before Create()
    m_bCubemap = true;
    m_ArrayLayers = 6;

    Create(width, height, format, mipLevels);
    if (m_Image == VK_NULL_HANDLE) {
        FS.r_close(F);
        return false;
    }
    if (m_Allocation) vmaSetAllocationName(VulkanHW.m_Allocator, m_Allocation, filename);  // leak-dump name

    // Read remaining data (all 6 faces with mipmaps)
    VkDeviceSize dataSize = F->length() - F->tell();
    void* data = xr_malloc(dataSize);
    F->r(data, dataSize);
    FS.r_close(F);

    // Upload all faces
    UploadData(data, dataSize);
    xr_free(data);

    Msg("[Vulkan] Loaded cubemap DDS: %s (%dx%d, mips=%d, format=%d, caps2=0x%X, dataSize=%llu)",
        filename, width, height, mipLevels, (int)format, header.dwCaps2, (unsigned long long)dataSize);
    return true;
}

// Рассчет количества mip levels
u32 CVulkanTexture::CalculateMipLevels(u32 width, u32 height)
{
    u32 mipLevels = 1;
    u32 maxDim = (width > height) ? width : height;
    while (maxDim > 1) {
        maxDim >>= 1;
        mipLevels++;
    }
    return mipLevels;
}

// Проверка compressed формата
bool CVulkanTexture::IsCompressedFormat(VkFormat format)
{
    switch (format) {
        case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
        case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
        case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
        case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
        case VK_FORMAT_BC2_UNORM_BLOCK:
        case VK_FORMAT_BC2_SRGB_BLOCK:
        case VK_FORMAT_BC3_UNORM_BLOCK:
        case VK_FORMAT_BC3_SRGB_BLOCK:
        case VK_FORMAT_BC4_UNORM_BLOCK:
        case VK_FORMAT_BC4_SNORM_BLOCK:
        case VK_FORMAT_BC5_UNORM_BLOCK:
        case VK_FORMAT_BC5_SNORM_BLOCK:
        case VK_FORMAT_BC6H_UFLOAT_BLOCK:
        case VK_FORMAT_BC6H_SFLOAT_BLOCK:
        case VK_FORMAT_BC7_UNORM_BLOCK:
        case VK_FORMAT_BC7_SRGB_BLOCK:
            return true;
        default:
            return false;
    }
}

// Размер блока для compressed формата
u32 CVulkanTexture::GetBlockSize(VkFormat format)
{
    switch (format) {
        case VK_FORMAT_BC1_RGB_UNORM_BLOCK:
        case VK_FORMAT_BC1_RGB_SRGB_BLOCK:
        case VK_FORMAT_BC1_RGBA_UNORM_BLOCK:
        case VK_FORMAT_BC1_RGBA_SRGB_BLOCK:
        case VK_FORMAT_BC4_UNORM_BLOCK:
        case VK_FORMAT_BC4_SNORM_BLOCK:
            return 8;  // 8 bytes per 4x4 block
        case VK_FORMAT_BC2_UNORM_BLOCK:
        case VK_FORMAT_BC2_SRGB_BLOCK:
        case VK_FORMAT_BC3_UNORM_BLOCK:
        case VK_FORMAT_BC3_SRGB_BLOCK:
        case VK_FORMAT_BC5_UNORM_BLOCK:
        case VK_FORMAT_BC5_SNORM_BLOCK:
        case VK_FORMAT_BC6H_UFLOAT_BLOCK:
        case VK_FORMAT_BC6H_SFLOAT_BLOCK:
        case VK_FORMAT_BC7_UNORM_BLOCK:
        case VK_FORMAT_BC7_SRGB_BLOCK:
            return 16; // 16 bytes per 4x4 block
        default:
            return 0;
    }
}

//------------------------------------------------------------------------------
// CUserTextureRegistry - $user$ texture system (Phase 0.3)
//------------------------------------------------------------------------------

void CUserTextureRegistry::Register(const char* name, CRT* rt)
{
    if (!name || !rt) {
        Msg("![Vulkan] Cannot register null user texture");
        return;
    }

    // Проверяем что имя начинается с $user$
    if (strncmp(name, "$user$", 6) != 0) {
        Msg("![Vulkan] User texture name must start with $user$: %s", name);
        return;
    }

    shared_str key = name;

    // Проверяем дубликаты
    auto it = m_UserTextures.find(key);
    if (it != m_UserTextures.end()) {
        Msg("~[Vulkan] User texture %s already registered, replacing", name);
    }

    m_UserTextures[key] = rt;
    Msg("[Vulkan] User texture registered: %s", name);
}

CRT* CUserTextureRegistry::Get(const char* name)
{
    if (!name) {
        return nullptr;
    }

    shared_str key = name;
    auto it = m_UserTextures.find(key);
    if (it != m_UserTextures.end()) {
        return it->second;
    }

    // Msg("~[Vulkan] User texture not found: %s", name);
    return nullptr;
}

void CUserTextureRegistry::Unregister(const char* name)
{
    if (!name) {
        return;
    }

    shared_str key = name;
    auto it = m_UserTextures.find(key);
    if (it != m_UserTextures.end()) {
        m_UserTextures.erase(it);
        Msg("[Vulkan] User texture unregistered: %s", name);
    }
}

void CUserTextureRegistry::Clear()
{
    m_UserTextures.clear();
    Msg("[Vulkan] All user textures cleared");
}

// Глобальный instance
CUserTextureRegistry g_UserTextureRegistry;

} // namespace VK
