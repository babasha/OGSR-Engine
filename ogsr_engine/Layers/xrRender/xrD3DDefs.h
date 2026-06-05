#ifndef xrD3DDefs_included
#define xrD3DDefs_included
#pragma once

#if defined(USE_DX11) || defined(USE_DX10)

#include "../xrRenderDX10/DXCommonTypes.h"

#elif defined(XRRENDER_VULKAN_EXPORTS)

// Vulkan stubs for D3D types — needed for shared scene-graph headers
// (FBasicVisual, sh_atomic, sh_texture, sh_rt, ...) that store these
// as opaque pointer members. Vulkan paths never dereference them; the
// real Vulkan-side equivalents live in vk_* parallel files.
//
// All pointer types are `void*` so that `T*` parses and pointer arithmetic
// / null comparisons work. Sizeof matches a native pointer.

typedef void* ID3DBlob;
typedef void* ID3DInclude;
typedef void* ID3DQuery;

typedef void* ID3DVertexShader;
typedef void* ID3DPixelShader;
typedef void* ID3DGeometryShader;
typedef void* ID3D11HullShader;
typedef void* ID3D11DomainShader;
typedef void* ID3D11ComputeShader;

typedef void* ID3DTexture2D;
typedef void* ID3DTexture3D;
typedef void* ID3DBaseTexture;
typedef void* ID3DRenderTargetView;
typedef void* ID3DDepthStencilView;
typedef void* ID3DShaderResourceView;
typedef void* ID3D11UnorderedAccessView;

typedef void* ID3DVertexBuffer;
typedef void* ID3DIndexBuffer;
typedef void* ID3DInputLayout;

typedef void* ID3DState;

// Forward-declare D3D9 leftover used by `tss_def.h::record()` so member
// declarations parse. Method body lives in DX-only TUs we don't compile.
struct IDirect3DStateBlock9;

// `dx10State` is a real class in xrRenderDX10/StateManager. Forward-declared
// so `tss_def.h::UpdateState(dx10State&)` parses — body never linked.
class dx10State;

// Stub structs for desc types passed by reference / used in arrays. The
// Vulkan path never reads these; sizeof must be valid for compilation.
struct D3D_TEXTURE2D_DESC
{
    u32 Width;
    u32 Height;
    u32 MipLevels;
    u32 ArraySize;
    u32 Format;
    u32 SampleCount;
    u32 SampleQuality;
    u32 Usage;
    u32 BindFlags;
    u32 CPUAccessFlags;
    u32 MiscFlags;
};

struct D3D_RASTERIZER_DESC      { u32 _opaque[16]; };
struct D3D_DEPTH_STENCIL_DESC   { u32 _opaque[16]; };
struct D3D_BLEND_DESC           { u32 _opaque[20]; };
struct D3D_SAMPLER_DESC         { u32 _opaque[16]; };
struct D3D_INPUT_ELEMENT_DESC   { u32 _opaque[8];  };

#ifndef D3D_COMMONSHADER_SAMPLER_SLOT_COUNT
#  define D3D_COMMONSHADER_SAMPLER_SLOT_COUNT 16
#endif

#define DX10_ONLY(expr) do {} while (0)

#endif // backend selection

#endif // xrD3DDefs_included
