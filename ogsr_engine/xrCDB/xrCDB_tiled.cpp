#include "stdafx.h"

#include "xrCDB_tiled.h"

#include <cstdio>
#include <algorithm>
#include <future>
#include <thread>
#include <chrono>

using namespace CDB;
using namespace Opcode;

// ============================================================================
// Cache file: header | verts | tris | directory | per-tile node blobs.
// Node links use the same disk encoding as the monolithic CDB cache
// (index<<1 relative to the TILE-local node array; leaf payloads are
// (globalPrim<<1)|1 and stay verbatim — they are position-independent).
// ============================================================================
namespace
{
constexpr u32 kTiledCacheMagic = 0x54444358u; // 'XCDT'
constexpr u32 kTiledCacheVersion = 1;
constexpr float kTileSize = 256.f;
constexpr u32 kMaxTiles = 65536;
constexpr u32 kEvictProtectTicks = 300; // ~5s at 60fps: recently-touched tiles are hot even outside the bubble

#pragma pack(push, 4)
struct TiledCacheHeader
{
    u32 magic, version;
    u64 key; // caller-provided validity stamp
    u64 verts_count, tris_count;
    float origin_x, origin_z, tile_size;
    u32 nx, nz;
};
struct TiledDirEntry
{
    u64 offset;
    u32 nb_nodes;
    u32 _pad;
    Fvector bb_min, bb_max;
};
#pragma pack(pop)

inline size_t encode_link(size_t v, const AABBNoLeafNode* base)
{
    if (v & 1)
        return v;
    return (size_t)((const AABBNoLeafNode*)v - base) << 1;
}
inline size_t decode_link(size_t v, const AABBNoLeafNode* base)
{
    if (v & 1)
        return v;
    return (size_t)(base + (v >> 1));
}

bool write_all(FILE* f, const void* data, size_t bytes)
{
    const u8* p = (const u8*)data;
    while (bytes)
    {
        const size_t chunk = std::min<size_t>(bytes, 64u << 20);
        if (fwrite(p, 1, chunk, f) != chunk)
            return false;
        p += chunk;
        bytes -= chunk;
    }
    return true;
}

bool read_all(FILE* f, void* data, size_t bytes)
{
    u8* p = (u8*)data;
    while (bytes)
    {
        const size_t chunk = std::min<size_t>(bytes, 64u << 20);
        if (fread(p, 1, chunk, f) != chunk)
            return false;
        p += chunk;
        bytes -= chunk;
    }
    return true;
}

// Build one tile's OPCODE tree over the given GLOBAL tri ids and leave the
// stolen node array (leaf prims rewritten to global ids) in the tile.
bool build_tile_from_ids(TILE_GRID& G, TILE& t, const xr_vector<u32>& ids)
{
    const size_t n0 = ids.size();
    if (!n0)
    {
        t.file_nodes = 0;
        return true;
    }

    // OPCODE needs >=2 tris for a real tree; a 1-tri tile duplicates it (two
    // leaves, same global id — identical to a boundary-dup, deduped the same)
    const size_t n = std::max<size_t>(n0, 2);
    xr_vector<TRI> local(n);
    Fvector bb_min{type_max<float>, type_max<float>, type_max<float>};
    Fvector bb_max{type_min<float>, type_min<float>, type_min<float>};
    for (size_t i = 0; i < n; ++i)
    {
        const TRI& src = G.tris[ids[std::min(i, n0 - 1)]];
        local[i] = src;
        for (u32 k = 0; k < 3; ++k)
        {
            const Fvector& v = G.verts[src.verts[k]];
            bb_min.min(v);
            bb_max.max(v);
        }
    }

    MeshInterface mif;
    mif.SetNbTriangles((udword)n);
    mif.SetNbVertices((udword)G.verts_count);
    mif.SetPointers(reinterpret_cast<const IceMaths::IndexedTriangle*>(local.data()), reinterpret_cast<const IceMaths::Point*>(G.verts));
    mif.SetStrides(sizeof(TRI));

    OPCODECREATE OPCC;
    OPCC.mIMesh = &mif;
    OPCC.mNoLeaf = true;
    OPCC.mQuantized = false;

    Opcode::Model tree;
    if (!tree.Build(OPCC))
        return false;
    if (tree.IsQuantized() || tree.HasLeafNodes() || tree.HasSingleNode() || !tree.GetTree())
        return false;

    AABBNoLeafTree* T = (AABBNoLeafTree*)tree.GetTree();
    const u32 nb = T->GetNbNodes();
    AABBNoLeafNode* nodes = (AABBNoLeafNode*)T->GetData();
    if (!nodes || !nb)
        return false;

    // leaf prims: local -> global
    auto remap = [&](size_t& v) {
        if (v & 1)
            v = ((size_t)ids[std::min<size_t>(v >> 1, n0 - 1)] << 1) | 1;
    };
    for (u32 i = 0; i < nb; ++i)
    {
        remap(nodes[i].mPosData);
        remap(nodes[i].mNegData);
    }

    T->SetData(nullptr, 0); // steal ownership; ~Model deletes null

    t.nodes = nodes;
    t.nb_nodes = nb;
    t.file_nodes = nb;
    t.bb_min = bb_min;
    t.bb_max = bb_max;
    t.state.store(TILE::S_READY, std::memory_order_release);
    G.resident_bytes.fetch_add((size_t)nb * sizeof(AABBNoLeafNode), std::memory_order_relaxed);
    return true;
}

bool save_tiled(TILE_GRID& G, const char* path, u64 key)
{
    static_assert(sizeof(AABBNoLeafNode) == 40, "node layout changed — bump kTiledCacheVersion and re-check encode_link");

    string_path tmp;
    xr_sprintf(tmp, "%s.tmp", path);
    FILE* f = fopen(tmp, "wb");
    if (!f)
        return false;

    const u32 nTiles = G.nx * G.nz;
    TiledCacheHeader h{kTiledCacheMagic, kTiledCacheVersion, key, (u64)G.verts_count, (u64)G.tris_count, G.origin_x, G.origin_z, G.tile_size, G.nx, G.nz};
    xr_vector<TiledDirEntry> dir(nTiles);

    bool ok = write_all(f, &h, sizeof(h)) && write_all(f, G.verts, G.verts_count * sizeof(Fvector)) && write_all(f, G.tris, G.tris_count * sizeof(TRI));
    const s64 dirPos = ok ? _ftelli64(f) : 0;
    ok = ok && write_all(f, dir.data(), nTiles * sizeof(TiledDirEntry)); // placeholder

    constexpr u32 kBatch = 65536;
    xr_vector<AABBNoLeafNode> scratch(kBatch);
    for (u32 i = 0; ok && i < nTiles; ++i)
    {
        TILE& t = G.tiles[i];
        dir[i].nb_nodes = t.file_nodes;
        dir[i].bb_min = t.bb_min;
        dir[i].bb_max = t.bb_max;
        if (!t.file_nodes)
            continue;
        dir[i].offset = (u64)_ftelli64(f);
        for (u32 j = 0; ok && j < t.nb_nodes; j += kBatch)
        {
            const u32 c = std::min(kBatch, t.nb_nodes - j);
            std::memcpy(scratch.data(), t.nodes + j, c * sizeof(AABBNoLeafNode));
            for (u32 k = 0; k < c; ++k)
            {
                scratch[k].mPosData = encode_link(scratch[k].mPosData, t.nodes);
                scratch[k].mNegData = encode_link(scratch[k].mNegData, t.nodes);
            }
            ok = write_all(f, scratch.data(), c * sizeof(AABBNoLeafNode));
        }
    }

    if (ok)
    {
        ok = 0 == _fseeki64(f, dirPos, SEEK_SET) && write_all(f, dir.data(), nTiles * sizeof(TiledDirEntry));
        // runtime tile offsets come straight from what we just wrote
        for (u32 i = 0; ok && i < nTiles; ++i)
            G.tiles[i].file_offset = dir[i].offset;
    }

    fclose(f);
    if (!ok)
    {
        remove(tmp);
        return false;
    }
    remove(path); // rename() refuses to overwrite
    if (0 != rename(tmp, path))
    {
        remove(tmp);
        return false;
    }
    return true;
}

// nodes blob -> live pointers, with bounds sanity (a corrupt cache must fail
// the load, not stab random memory during the first query)
bool fixup_nodes(AABBNoLeafNode* nodes, u32 nb, size_t tris_count)
{
    for (u32 i = 0; i < nb; ++i)
    {
        auto fix = [&](size_t& v) -> bool {
            if (v & 1)
                return (v >> 1) < tris_count;
            if ((v >> 1) >= nb)
                return false;
            v = decode_link(v, nodes);
            return true;
        };
        if (!fix(nodes[i].mPosData) || !fix(nodes[i].mNegData))
            return false;
    }
    return true;
}
} // namespace

// ============================================================================
// TILE_GRID
// ============================================================================

TILE_GRID::~TILE_GRID()
{
    while (prefetch_inflight.load(std::memory_order_acquire))
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (file)
        fclose(file);
    if (tiles)
        for (u32 i = 0, n = nx * nz; i < n; ++i)
            delete[] tiles[i].nodes;
}

bool TILE_GRID::load_tile(TILE& t)
{
    if (!file)
        return false;

    CTimer timer;
    timer.Start();

    AABBNoLeafNode* nodes = new AABBNoLeafNode[t.file_nodes];
    bool ok;
    {
        std::lock_guard<std::mutex> io(io_lock);
        ok = 0 == _fseeki64(file, (s64)t.file_offset, SEEK_SET) && read_all(file, nodes, (size_t)t.file_nodes * sizeof(AABBNoLeafNode));
    }
    ok = ok && fixup_nodes(nodes, t.file_nodes, tris_count);
    if (!ok)
    {
        delete[] nodes;
        return false;
    }

    t.nodes = nodes;
    t.nb_nodes = t.file_nodes;
    t.state.store(TILE::S_READY, std::memory_order_release);
    resident_bytes.fetch_add((size_t)t.nb_nodes * sizeof(AABBNoLeafNode), std::memory_order_relaxed);
    touch(t);

    const u32 n = stat_loads.fetch_add(1, std::memory_order_relaxed) + 1;
    if ((n & (n - 1)) == 0)
        Msg("[CDB tiles] load #%u: %u nodes (%.1f MB) in %u ms | resident %.1f MB", n, t.nb_nodes, t.nb_nodes * sizeof(AABBNoLeafNode) / (1024.f * 1024.f),
            timer.GetElapsed_ms(), resident_bytes.load(std::memory_order_relaxed) / (1024.f * 1024.f));
    return true;
}

bool TILE_GRID::rebuild_tile(TILE& t)
{
    // Corrupt/unreadable blob: the source of truth (verts+tris) is resident,
    // so the tile tree can always be rebuilt in memory. Slow (full tri scan +
    // OPCODE build) but correctness-preserving.
    const size_t idx = (size_t)(&t - tiles.get());
    const u32 ix = (u32)(idx % nx), iz = (u32)(idx / nx);
    const float x0 = origin_x + ix * tile_size, x1 = x0 + tile_size;
    const float z0 = origin_z + iz * tile_size, z1 = z0 + tile_size;

    xr_vector<u32> ids;
    ids.reserve(4096);
    for (u32 i = 0; i < (u32)tris_count; ++i)
    {
        const TRI& T = tris[i];
        float tx0 = type_max<float>, tx1 = type_min<float>, tz0 = type_max<float>, tz1 = type_min<float>;
        for (u32 k = 0; k < 3; ++k)
        {
            const Fvector& v = verts[T.verts[k]];
            tx0 = std::min(tx0, v.x);
            tx1 = std::max(tx1, v.x);
            tz0 = std::min(tz0, v.z);
            tz1 = std::max(tz1, v.z);
        }
        if (tx1 >= x0 && tx0 < x1 && tz1 >= z0 && tz0 < z1)
            ids.push_back(i);
    }

    if (ids.empty())
        return false; // file_nodes>0 promised geometry — directory is corrupt too

    const u32 keep_file_nodes = t.file_nodes; // rebuilt node count may differ from the blob's
    if (!build_tile_from_ids(*this, t, ids))
        return false;
    t.file_nodes = keep_file_nodes;
    stat_rebuilds.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool TILE_GRID::make_ready(TILE& t, bool sync)
{
    if (!t.file_nodes)
        return false;

    std::unique_lock<std::shared_mutex> ul(t.lock);
    if (t.state.load(std::memory_order_relaxed) == TILE::S_READY)
        return true;

    if (sync)
        stat_sync_loads.fetch_add(1, std::memory_order_relaxed);

    if (load_tile(t))
        return true;

    Msg("![CDB tiles] tile load FAILED (offset %llu, %u nodes) — rebuilding from resident tris", t.file_offset, t.file_nodes);
    if (rebuild_tile(t))
        return true;

    Msg("![CDB tiles] tile REBUILD failed — collision hole possible!");
    return false;
}

void TILE_GRID::update_streaming(const Fvector& focus, size_t budget_bytes, float bubble_radius, BOOL verbose)
{
    const u32 now = tick.fetch_add(1, std::memory_order_relaxed) + 1;
    if (!file)
        return; // no streaming source: fully resident, nothing to evict or prefetch

    const int bx0 = std::max(cell_x(focus.x - bubble_radius), 0), bx1 = std::min(cell_x(focus.x + bubble_radius), (int)nx - 1);
    const int bz0 = std::max(cell_z(focus.z - bubble_radius), 0), bz1 = std::min(cell_z(focus.z + bubble_radius), (int)nz - 1);

    // prefetch the bubble (one async task in flight; the sync-load path in the
    // queries stays as the correctness fallback)
    if (!prefetch_inflight.load(std::memory_order_relaxed))
    {
        xr_vector<u32> want;
        for (int iz = bz0; iz <= bz1; ++iz)
            for (int ix = bx0; ix <= bx1; ++ix)
            {
                TILE& t = cell(ix, iz);
                if (t.file_nodes && t.state.load(std::memory_order_relaxed) != TILE::S_READY)
                    want.push_back((u32)(iz * nx + ix));
            }
        if (!want.empty() && !prefetch_inflight.exchange(true))
        {
            TTAPI->submit_detach([this, want] {
                for (const u32 idx : want)
                    make_ready(tiles[idx], false /*prefetch*/);
                prefetch_inflight.store(false, std::memory_order_release);
            });
        }
    }

    // evict LRU outside the bubble while over budget
    if (resident_bytes.load(std::memory_order_relaxed) > budget_bytes)
    {
        struct cand
        {
            u32 idx, use;
        };
        xr_vector<cand> cands;
        for (u32 iz = 0; iz < nz; ++iz)
            for (u32 ix = 0; ix < nx; ++ix)
            {
                if ((int)ix >= bx0 && (int)ix <= bx1 && (int)iz >= bz0 && (int)iz <= bz1)
                    continue; // bubble is pinned
                TILE& t = cell(ix, iz);
                if (t.state.load(std::memory_order_relaxed) != TILE::S_READY)
                    continue;
                const u32 use = t.last_use.load(std::memory_order_relaxed);
                if (now - use < kEvictProtectTicks)
                    continue; // recently touched (distant NPC/sound) — hot
                cands.push_back({iz * nx + ix, use});
            }
        std::sort(cands.begin(), cands.end(), [](const cand& a, const cand& b) { return a.use < b.use; });
        for (const cand& c : cands)
        {
            if (resident_bytes.load(std::memory_order_relaxed) <= budget_bytes)
                break;
            TILE& t = tiles[c.idx];
            if (!t.lock.try_lock()) // someone is traversing/loading — skip
                continue;
            if (t.state.load(std::memory_order_relaxed) == TILE::S_READY)
            {
                resident_bytes.fetch_sub((size_t)t.nb_nodes * sizeof(AABBNoLeafNode), std::memory_order_relaxed);
                t.state.store(TILE::S_EMPTY, std::memory_order_release);
                delete[] t.nodes;
                t.nodes = nullptr;
                t.nb_nodes = 0;
                stat_evictions.fetch_add(1, std::memory_order_relaxed);
            }
            t.lock.unlock();
        }
    }

    if (verbose && 0 == (now % 300))
        Msg("[CDB tiles] resident %.1f MB / budget %.1f MB | loads %u (sync %u) evictions %u rebuilds %u", resident_bytes.load() / (1024.f * 1024.f),
            budget_bytes / (1024.f * 1024.f), stat_loads.load(), stat_sync_loads.load(), stat_evictions.load(), stat_rebuilds.load());
}

// ============================================================================
// MODEL: tiled build / cache load
// ============================================================================

bool MODEL::build_tiled(const Fvector* V, size_t Vcnt, const TRI* T, size_t Tcnt, const Fbox& aabb, build_callback* bc, void* bcp, const char* cache_path, u64 key)
{
    R_ASSERT(S_INIT == status);
    if (Vcnt < 4 || Tcnt < 2)
        return false;

    // resident copies + material callback — identical to the monolithic path
    verts_count = Vcnt;
    verts = xr_alloc<Fvector>(verts_count);
    std::memcpy(verts, V, verts_count * sizeof(Fvector));
    tris_count = Tcnt;
    tris = xr_alloc<TRI>(tris_count);
    std::memcpy(tris, T, tris_count * sizeof(TRI));
    if (bc)
        bc(verts, Vcnt, tris, Tcnt, bcp);

    auto cleanup = [&] {
        xr_free(verts);
        verts_count = 0;
        xr_free(tris);
        tris_count = 0;
    };

    TILE_GRID* G = xr_new<TILE_GRID>();
    G->verts = verts;
    G->tris = tris;
    G->verts_count = verts_count;
    G->tris_count = tris_count;
    G->tile_size = kTileSize;
    G->inv_tile = 1.f / kTileSize;
    G->origin_x = aabb.x1 - 1.f;
    G->origin_z = aabb.z1 - 1.f;
    G->nx = std::max(1, (int)std::ceil((aabb.x2 + 1.f - G->origin_x) / kTileSize));
    G->nz = std::max(1, (int)std::ceil((aabb.z2 + 1.f - G->origin_z) / kTileSize));
    const u32 nTiles = G->nx * G->nz;
    if (nTiles > kMaxTiles)
    {
        Msg("![CDB tiles] insane grid %ux%u — falling back to monolith", G->nx, G->nz);
        xr_delete(G);
        cleanup();
        return false;
    }
    G->tiles = std::make_unique<TILE[]>(nTiles);

    // ---- bin tris to tiles (flat prefix layout; a boundary tri lands in every tile it overlaps)
    CTimer timer;
    timer.Start();
    xr_vector<u32> counts(nTiles + 1, 0);
    auto tri_cells = [&](u32 i, int& ix0, int& ix1, int& iz0, int& iz1) {
        const TRI& t = tris[i];
        float x0 = type_max<float>, x1 = type_min<float>, z0 = type_max<float>, z1 = type_min<float>;
        for (u32 k = 0; k < 3; ++k)
        {
            const Fvector& v = verts[t.verts[k]];
            x0 = std::min(x0, v.x);
            x1 = std::max(x1, v.x);
            z0 = std::min(z0, v.z);
            z1 = std::max(z1, v.z);
        }
        ix0 = std::clamp(G->cell_x(x0), 0, (int)G->nx - 1);
        ix1 = std::clamp(G->cell_x(x1), 0, (int)G->nx - 1);
        iz0 = std::clamp(G->cell_z(z0), 0, (int)G->nz - 1);
        iz1 = std::clamp(G->cell_z(z1), 0, (int)G->nz - 1);
    };
    for (u32 i = 0; i < (u32)tris_count; ++i)
    {
        int ix0, ix1, iz0, iz1;
        tri_cells(i, ix0, ix1, iz0, iz1);
        for (int iz = iz0; iz <= iz1; ++iz)
            for (int ix = ix0; ix <= ix1; ++ix)
                ++counts[iz * G->nx + ix + 1];
    }
    for (u32 c = 1; c <= nTiles; ++c)
        counts[c] += counts[c - 1];
    const size_t totalRefs = counts[nTiles];
    xr_vector<u32> refs(totalRefs);
    {
        xr_vector<u32> cursor(counts.begin(), counts.end() - 1);
        for (u32 i = 0; i < (u32)tris_count; ++i)
        {
            int ix0, ix1, iz0, iz1;
            tri_cells(i, ix0, ix1, iz0, iz1);
            for (int iz = iz0; iz <= iz1; ++iz)
                for (int ix = ix0; ix <= ix1; ++ix)
                    refs[cursor[iz * G->nx + ix]++] = i;
        }
    }
    const u32 binMs = timer.GetElapsed_ms();

    // ---- per-tile OPCODE builds, parallel (no mutable globals in the builder)
    timer.Start();
    std::atomic<bool> failed{false};
    xr_vector<std::future<void>> jobs;
    for (u32 c = 0; c < nTiles; ++c)
    {
        if (counts[c + 1] == counts[c])
            continue;
        auto job = [G, &refs, &counts, &failed, c] {
            if (failed.load(std::memory_order_relaxed))
                return;
            xr_vector<u32> ids(refs.begin() + counts[c], refs.begin() + counts[c + 1]);
            if (!build_tile_from_ids(*G, G->tiles[c], ids))
                failed.store(true, std::memory_order_relaxed);
        };
        if (TTAPI)
            jobs.emplace_back(TTAPI->submit(job));
        else
            job();
    }
    for (auto& j : jobs)
        j.wait();
    const u32 buildMs = timer.GetElapsed_ms();

    if (failed.load())
    {
        Msg("![CDB tiles] tile build FAILED — falling back to monolith");
        xr_delete(G);
        cleanup();
        return false;
    }

    // ---- save + reopen as streaming source
    timer.Start();
    if (save_tiled(*G, cache_path, key))
    {
        G->file = fopen(cache_path, "rb");
        Msg("[CDB tiles] baked %ux%u tiles (%u tris, dup %.2f%%): bin %u ms, build %u ms, save %u ms | tree %.1f MB", G->nx, G->nz, (u32)tris_count,
            totalRefs ? 100.f * float(totalRefs - tris_count) / float(tris_count) : 0.f, binMs, buildMs, timer.GetElapsed_ms(),
            G->resident_bytes.load() / (1024.f * 1024.f));
    }
    else
        Msg("![CDB tiles] baked, but save FAILED '%s' — running fully resident (no eviction)", cache_path);

    tiles_ = G;
    status = S_READY;
    return true;
}

bool MODEL::cache_load_tiled(const char* cache_path, u64 key)
{
    R_ASSERT(S_INIT == status);

    FILE* f = fopen(cache_path, "rb");
    if (!f)
        return false;

    TILE_GRID* G = nullptr;
    bool ok = false;
    do
    {
        TiledCacheHeader h;
        if (!read_all(f, &h, sizeof(h)))
            break;
        if (h.magic != kTiledCacheMagic || h.version != kTiledCacheVersion || h.key != key)
            break;
        if (!h.verts_count || !h.tris_count || !h.nx || !h.nz || h.nx * h.nz > kMaxTiles || h.tile_size < 16.f)
            break;

        verts_count = (size_t)h.verts_count;
        verts = xr_alloc<Fvector>(verts_count);
        tris_count = (size_t)h.tris_count;
        tris = xr_alloc<TRI>(tris_count);
        if (!read_all(f, verts, verts_count * sizeof(Fvector)) || !read_all(f, tris, tris_count * sizeof(TRI)))
            break;

        const u32 nTiles = h.nx * h.nz;
        xr_vector<TiledDirEntry> dir(nTiles);
        if (!read_all(f, dir.data(), nTiles * sizeof(TiledDirEntry)))
            break;

        G = xr_new<TILE_GRID>();
        G->verts = verts;
        G->tris = tris;
        G->verts_count = verts_count;
        G->tris_count = tris_count;
        G->origin_x = h.origin_x;
        G->origin_z = h.origin_z;
        G->tile_size = h.tile_size;
        G->inv_tile = 1.f / h.tile_size;
        G->nx = h.nx;
        G->nz = h.nz;
        G->tiles = std::make_unique<TILE[]>(nTiles);
        bool sane = true;
        for (u32 i = 0; i < nTiles && sane; ++i)
        {
            TILE& t = G->tiles[i];
            t.file_offset = dir[i].offset;
            t.file_nodes = dir[i].nb_nodes;
            t.bb_min = dir[i].bb_min;
            t.bb_max = dir[i].bb_max;
            if (t.file_nodes && t.file_offset < sizeof(h))
                sane = false;
        }
        if (!sane)
            break;

        G->file = f;
        f = nullptr; // grid owns the handle now
        tiles_ = G;
        G = nullptr;
        status = S_READY;
        ok = true;
    } while (false);

    if (f)
        fclose(f);
    if (!ok)
    {
        xr_delete(G);
        xr_free(verts);
        verts_count = 0;
        xr_free(tris);
        tris_count = 0;
        status = S_INIT;
    }
    return ok;
}

void MODEL::update_streaming(const Fvector& focus, size_t budget_mb, float bubble_radius, BOOL verbose)
{
    if (tiles_)
        tiles_->update_streaming(focus, budget_mb << 20, bubble_radius, verbose);
}
