// xrRenderVulkan - Sun cascade structures (stub)
//
// `rvk.h` declares `xr_vector<VK::SunCascade> m_sun_cascades` member, which
// requires the full type definition. The actual cascade rendering still
// lives in `_parked/vk_lighting.cpp` and isn't compiled today; this stub
// just satisfies the type system so rvk.h parses.

#pragma once

namespace VK
{
    struct SunCascade
    {
        Fmatrix xform{};
        u32     size = 0;
    };
}
