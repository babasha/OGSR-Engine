#pragma once

// Forward decl is enough — `xr_vector<light*>` only needs `light` to be a
// type. Consumers that actually deref a `light*` must include "light.h"
// themselves (and its R4-specific transitives like `light_smapvis.h`,
// `xr_area.h::RayPickAsync`).
class light;
class CBackend;

class light_Package
{
public:
    xr_vector<light*> v_point;
    xr_vector<light*> v_spot;
    xr_vector<light*> v_shadowed;

public:
    void clear();
    void sort();
    void vis_prepare(CBackend& cmd_list) const;
    void vis_update() const;
};
