// xrRenderVulkan — Clustered forward lighting: shared cluster-grid math (constants +
// pure functions, NO resource bindings). Included by cluster_assign.comp (builds the
// per-cluster light lists) and the world fragment shaders (read their cluster's list).
// Both MUST agree on the grid → one header. See vk_clustered.{h,cpp}.
//
// The view frustum is diced into CLUSTERS_X * CLUSTERS_Y screen tiles * CLUSTERS_Z depth
// slices. Depth is sliced EXPONENTIALLY (log) so near slices are thin and far slices fat
// — matches how light density falls off with distance. A fragment finds its cluster from
// screen position + linear view depth (= distance along the camera forward), which the
// world shaders already have (eye_pos + cam_dir in the Lighting UBO) — no projection math.
#ifndef VK_CLUSTER_GLSL
#define VK_CLUSTER_GLSL

const int  CLUSTERS_X = 16;
const int  CLUSTERS_Y = 9;
const int  CLUSTERS_Z = 24;
const int  CLUSTER_COUNT = CLUSTERS_X * CLUSTERS_Y * CLUSTERS_Z;   // 3456
const int  CLUSTER_MAX_LIGHTS = 64;                               // per-cluster cap (overflow drops far/dim ones)

// Linear view depth (metres in front of the camera) -> exponential slice index.
// near/far are the cluster grid's depth range (NOT the scene znear/zfar necessarily).
int clusterSlice(float viewZ, float nearZ, float farZ) {
    viewZ = max(viewZ, nearZ);
    float s = log(viewZ / nearZ) / log(farZ / nearZ);   // [0,1]
    return clamp(int(s * float(CLUSTERS_Z)), 0, CLUSTERS_Z - 1);
}

// The viewZ at the near/far planes of a slice (for the assign pass's cluster bounds).
float clusterSliceNear(int slice, float nearZ, float farZ) {
    return nearZ * pow(farZ / nearZ, float(slice) / float(CLUSTERS_Z));
}

// Flat cluster index from tile (x,y) + depth slice.
int clusterIndex(ivec2 tile, int slice) {
    return slice * (CLUSTERS_X * CLUSTERS_Y) + tile.y * CLUSTERS_X + tile.x;
}

// Fragment -> cluster. screenUV in [0,1] (gl_FragCoord.xy / screenDims); viewZ = linear
// view depth = dot(worldPos - eyePos, camForward).
int clusterFromFragment(vec2 screenUV, float viewZ, float nearZ, float farZ) {
    ivec2 tile = clamp(ivec2(screenUV * vec2(float(CLUSTERS_X), float(CLUSTERS_Y))),
                       ivec2(0), ivec2(CLUSTERS_X - 1, CLUSTERS_Y - 1));
    return clusterIndex(tile, clusterSlice(viewZ, nearZ, farZ));
}

#endif
