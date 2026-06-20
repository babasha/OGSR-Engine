// xrRenderVulkan - Vulkan renderer for X-Ray Engine
// Copyright (c) 2024-2026 Egor Babushkin (https://github.com/babasha)
//
// Original work, "declared otherwise" per the root LICENSE.md. Non-commercial
// use only (per the X-Ray Engine license); redistribution in source or binary
// form must keep this notice and credit the author in-game (credits or splash).

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
const u32 FOURCC_DX10 = 0x30315844; // "DX10" — extended header (DDS_HEADER_DXT10) follows

// Present immediately after DDS_HEADER when ddspf.dwFourCC == "DX10". Modern
// compressors (BC7/BC6H, and BC1-5 re-saved by texconv) emit this instead of the
// legacy FourCC, carrying an explicit DXGI format.
struct DDS_HEADER_DXT10 {
    u32 dxgiFormat;
    u32 resourceDimension;
    u32 miscFlag;
    u32 arraySize;
    u32 miscFlags2;
};

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

    // VK_CHECK is non-fatal (logs only). If the allocation failed, m_Image is
    // VK_NULL_HANDLE — bail before CreateImageView/CreateSampler, which would
    // otherwise build a view over a null image (invalid usage) and leak a sampler.
    if (m_Image == VK_NULL_HANDLE) {
        Msg("![Vulkan] Texture image allocation failed: %ux%u fmt=%d", width, height, (int)format);
        return;
    }

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

// Map a DXGI_FORMAT (from a DX10 extended DDS header) to the VkFormat we load it
// as. The engine runs an UNORM pipeline (UNORM swapchain, no gamma hardware), so
// sRGB DXGI variants are loaded as their UNORM equivalents — same convention the
// legacy BC1/2/3 path already uses. Returns VK_FORMAT_UNDEFINED for formats we
// don't handle so the caller can bail with a clear message.
static VkFormat DXGIFormatToVk(u32 dxgi)
{
    switch (dxgi) {
        case DXGI_FORMAT_R8G8B8A8_UNORM:
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB: return VK_FORMAT_R8G8B8A8_UNORM;
        case DXGI_FORMAT_B8G8R8A8_UNORM:
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB: return VK_FORMAT_B8G8R8A8_UNORM;
        case DXGI_FORMAT_R8_UNORM:            return VK_FORMAT_R8_UNORM;
        case DXGI_FORMAT_R8G8_UNORM:          return VK_FORMAT_R8G8_UNORM;
        case DXGI_FORMAT_BC1_UNORM:
        case DXGI_FORMAT_BC1_UNORM_SRGB:      return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
        case DXGI_FORMAT_BC2_UNORM:
        case DXGI_FORMAT_BC2_UNORM_SRGB:      return VK_FORMAT_BC2_UNORM_BLOCK;
        case DXGI_FORMAT_BC3_UNORM:
        case DXGI_FORMAT_BC3_UNORM_SRGB:      return VK_FORMAT_BC3_UNORM_BLOCK;
        case DXGI_FORMAT_BC4_UNORM:           return VK_FORMAT_BC4_UNORM_BLOCK;
        case DXGI_FORMAT_BC4_SNORM:           return VK_FORMAT_BC4_SNORM_BLOCK;
        case DXGI_FORMAT_BC5_UNORM:           return VK_FORMAT_BC5_UNORM_BLOCK;
        case DXGI_FORMAT_BC5_SNORM:           return VK_FORMAT_BC5_SNORM_BLOCK;
        case DXGI_FORMAT_BC6H_UF16:           return VK_FORMAT_BC6H_UFLOAT_BLOCK;
        case DXGI_FORMAT_BC6H_SF16:           return VK_FORMAT_BC6H_SFLOAT_BLOCK;
        case DXGI_FORMAT_BC7_UNORM:
        case DXGI_FORMAT_BC7_UNORM_SRGB:      return VK_FORMAT_BC7_UNORM_BLOCK;
        default:                              return VK_FORMAT_UNDEFINED;
    }
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
    bool expand24to32 = false;   // set for 24-bit RGB → upconverted to 32-bit on load

    if (header.ddspf.dwFlags & DDPF_FOURCC) {
        if (header.ddspf.dwFourCC == FOURCC_DX10) {
            // Extended DX10 header carries an explicit DXGI format (BC7/BC6H etc.).
            // Reading it also advances past the 20 extra bytes so the pixel data
            // that follows is at the correct file offset.
            DDS_HEADER_DXT10 h10{};
            F->r(&h10, sizeof(h10));
            format = DXGIFormatToVk(h10.dxgiFormat);
            if (format == VK_FORMAT_UNDEFINED) {
                Msg("![Vulkan] Unsupported DXGI format %u (DX10 header) in %s", h10.dxgiFormat, filename);
                FS.r_close(F);
                return false;
            }
            // The legacy R↔B swap only applies to BGR-ordered DXT UI atlases;
            // DX10 textures encode their true channel order in dxgiFormat, so swap
            // only when the caller asked AND the format is a (BC) compressed one.
            m_bBCSwizzle = applyBCSwizzle && IsCompressedFormat(format);
        } else {
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
        }
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
        } else if (header.ddspf.dwRGBBitCount == 24) {
            // Vulkan optimal-tiling sampled images don't support 3-byte R8G8B8, so
            // we upconvert to 32-bit (opaque alpha) on load. The blue mask tells us
            // the in-memory byte order (B first → D3DFMT_R8G8B8 / BGR, the common
            // case), so we tag the matching 32-bit format and copy bytes straight
            // through in the expansion pass below — no channel shuffle needed.
            format = (header.ddspf.dwBBitMask == 0x000000FF)
                       ? VK_FORMAT_B8G8R8A8_UNORM
                       : VK_FORMAT_R8G8B8A8_UNORM;
            expand24to32 = true;
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

    // 24-bit RGB → 32-bit RGBA expansion. Walk the mip chain exactly as
    // UploadData does (halve per level, floor at 1) so the rebuilt buffer is
    // tightly packed at 4 bpp and the copy regions line up.
    if (expand24to32) {
        u32 dstSize = 0;
        for (u32 w = width, h = height, i = 0; i < mipLevels; ++i) {
            dstSize += w * h * 4;
            if (w > 1) w >>= 1;
            if (h > 1) h >>= 1;
        }
        u8* dst = (u8*)xr_malloc(dstSize);
        const u8* src = (const u8*)data;
        u8* d = dst;
        for (u32 w = width, h = height, i = 0; i < mipLevels; ++i) {
            const u32 px = w * h;
            for (u32 p = 0; p < px; ++p) {
                d[0] = src[0]; d[1] = src[1]; d[2] = src[2]; d[3] = 0xFF;
                d += 4; src += 3;
            }
            if (w > 1) w >>= 1;
            if (h > 1) h >>= 1;
        }
        xr_free(data);
        data     = dst;
        dataSize = dstSize;
    }

    // Guard against a DDS whose header over-claims its contents (truncated file
    // or a wrong dwMipMapCount — seen on some level-packed textures reached via
    // the $level$ fallback). UploadData walks the full mip chain to build copy
    // regions; if those reference more bytes than we actually read, the driver's
    // vkCmdCopyBufferToImage reads past the staging buffer → GPU/driver crash
    // (was: hard crash mid-game when such an NPC/level texture streamed in).
    // Mirror UploadData's exact size walk; on shortfall, fail the load so the
    // caller falls back to the white default instead of feeding the driver.
    {
        VkDeviceSize expected = 0;
        for (u32 w = width, h = height, i = 0; i < mipLevels; ++i) {
            if (IsCompressedFormat(format))
                expected += (VkDeviceSize)((w + 3) / 4) * ((h + 3) / 4) * GetBlockSize(format);
            else {
                const u32 bpp = (format == VK_FORMAT_R8_UNORM) ? 1u : (format == VK_FORMAT_R8G8_UNORM) ? 2u : 4u;
                expected += (VkDeviceSize)w * h * bpp;
            }
            if (w > 1) w >>= 1;
            if (h > 1) h >>= 1;
        }
        if (expected > dataSize) {
            Msg("![Vulkan] DDS truncated/corrupt: '%s' needs %llu bytes (%ux%u mips=%u) but only %llu present — skipping (white default)",
                filename, (unsigned long long)expected, width, height, mipLevels, (unsigned long long)dataSize);
            xr_free(data);
            return false;
        }
    }

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
    u32 srcMips = (header.dwFlags & 0x20000) ? header.dwMipMapCount : 1;
    if (srcMips == 0) srcMips = 1;

    // Sky cubemaps ship single-mip + uncompressed (BGRA8). The hemisphere sky
    // ambient (vk_env_light + world shaders) samples a blurred HIGH mip as a
    // diffuse-irradiance approximation (R4 hmodel.h CUBE_MIPS), so build the
    // full chain at load. Compressed cubes can't be linear-blit-filtered — keep
    // those single-mip (rare; ambient just samples sharper there).
    const bool genMips   = !IsCompressedFormat(format) && width > 1 && srcMips == 1;
    const u32  totalMips = genMips ? CalculateMipLevels(width, height) : srcMips;

    // Setup cubemap flags before Create()
    m_bCubemap = true;
    m_ArrayLayers = 6;

    VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    if (genMips) usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;   // blit source for mip-gen

    Create(width, height, format, totalMips, usage);
    if (m_Image == VK_NULL_HANDLE) {
        FS.r_close(F);
        return false;
    }
    if (m_Allocation) vmaSetAllocationName(VulkanHW.m_Allocator, m_Allocation, filename);  // leak-dump name

    // Read remaining data (6 faces; mip0 only when srcMips==1).
    VkDeviceSize dataSize = F->length() - F->tell();
    void* data = xr_malloc(dataSize);
    F->r(data, dataSize);
    FS.r_close(F);

    if (genMips) GenerateMipsCube(data, dataSize);   // synchronous staging copy + blit chain
    else         UploadData(data, dataSize);
    xr_free(data);

    Msg("[Vulkan] Loaded cubemap DDS: %s (%dx%d, mips=%d, format=%d, caps2=0x%X, dataSize=%llu)",
        filename, width, height, totalMips, (int)format, header.dwCaps2, (unsigned long long)dataSize);
    return true;
}

// Build the full mip chain for a cubemap from mip-0 face data — synchronous.
// The image is created with the full mip count + TRANSFER_SRC; here we copy the
// 6 mip-0 faces from a staging buffer then vkCmdBlitImage each level into the
// next (all 6 layers per blit). One immediate cmd buffer, fence-waited, so it's
// safe to call mid-frame (sky cube loads happen at weather boundaries, rare).
void CVulkanTexture::GenerateMipsCube(const void* mip0Data, VkDeviceSize mip0Size)
{
    CVulkanBuffer staging;
    staging.Create(mip0Size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_AUTO_PREFER_HOST);
    void* p = staging.Map();
    if (!p) { Msg("![Vulkan] GenerateMipsCube: staging map failed"); staging.Destroy(); return; }
    memcpy(p, mip0Data, mip0Size);

    const VkDeviceSize faceSize = mip0Size / 6;   // tightly-packed mip0, 6 faces

    VkCommandBuffer cmd = CommandManager.BeginImmediate();
    if (cmd == VK_NULL_HANDLE) { staging.Destroy(); return; }

    auto barrier = [&](u32 baseMip, u32 mipCount, VkImageLayout oldL, VkImageLayout newL,
                       VkAccessFlags srcA, VkAccessFlags dstA,
                       VkPipelineStageFlags srcS, VkPipelineStageFlags dstS) {
        VkImageMemoryBarrier b{};
        b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.oldLayout           = oldL; b.newLayout = newL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image               = m_Image;
        b.subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, baseMip, mipCount, 0, 6 };
        b.srcAccessMask       = srcA; b.dstAccessMask = dstA;
        vkCmdPipelineBarrier(cmd, srcS, dstS, 0, 0, nullptr, 0, nullptr, 1, &b);
    };

    // Whole image UNDEFINED → TRANSFER_DST, then copy the 6 mip-0 faces.
    barrier(0, m_MipLevels, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            0, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    VkBufferImageCopy regions[6]{};
    for (u32 f = 0; f < 6; ++f) {
        regions[f].bufferOffset      = f * faceSize;
        regions[f].imageSubresource  = { VK_IMAGE_ASPECT_COLOR_BIT, 0, f, 1 };
        regions[f].imageExtent       = { m_Width, m_Height, 1 };
    }
    vkCmdCopyBufferToImage(cmd, staging.GetHandle(), m_Image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 6, regions);

    // mip0 → TRANSFER_SRC (blit source for mip1).
    barrier(0, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

    s32 mw = (s32)m_Width, mh = (s32)m_Height;
    for (u32 i = 1; i < m_MipLevels; ++i) {
        const s32 nw = mw > 1 ? mw / 2 : 1;
        const s32 nh = mh > 1 ? mh / 2 : 1;
        VkImageBlit blit{};
        blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, 6 };
        blit.srcOffsets[1]  = { mw, mh, 1 };
        blit.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 6 };
        blit.dstOffsets[1]  = { nw, nh, 1 };
        vkCmdBlitImage(cmd, m_Image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       m_Image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
        // This level becomes the next blit's source.
        barrier(i, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        mw = nw; mh = nh;
    }

    // All mips are TRANSFER_SRC now → SHADER_READ for the samplers.
    barrier(0, m_MipLevels, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    CommandManager.EndAndSubmitImmediate(cmd);   // fence-waited
    staging.Destroy();
    m_CurrentLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
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
