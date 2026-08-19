# Why gmpas exists

## Why

MPAS is unstructured: a variable-resolution spherical Voronoi tessellation,
with scalars at cell centres and normal velocity on cell edges. That is why
ordinary lat-lon tooling does not apply, and why the usual workarounds each
cost something:

| approach | conserves? | edge variables? |
|---|---|---|
| `convert_mpas` → lat-lon | **no** — barycentric sampling | via remap |
| cell polygons in a `.gpkg` | n/a | **no** — centres only |

gmpas keeps the good idea behind the `.gpkg` approach — mesh geometry is static
for a whole simulation, so build it once — and generalises it. The cache is a
directory of memory-mapped `.npy` arrays (cell polygons, edge segments, and
unit-sphere coordinates for the KD-tree) so edge-based fields work too, and no
GIS dependency is needed. Not a single `.npz`: that's a zip, and zips can't be
memory-mapped, which matters once a mesh reaches tens of millions of cells.

## How it compares

The two closest points of comparison are both real, maintained tools:
[`uxarray`](https://github.com/UXARRAY/uxarray), the general-purpose
unstructured-grid analysis library, and
[MPAS-Viewer](https://github.com/jhbravo/mpasviewer), a peer-reviewed,
MPAS-specific viewer ([Mendez & Temimi 2026, SoftwareX][mpasviewer-paper]).
Rather than assert gmpas is faster, here is what a real MPAS mesh + output
file actually measures, methodology included, so the numbers can be checked
rather than taken on faith. The script that produced them is at
[`docs/benchmarks/compare_gmpas_uxarray_mpasviewer.py`](benchmarks/compare_gmpas_uxarray_mpasviewer.py).

[mpasviewer-paper]: https://www.sciencedirect.com/science/article/pii/S2352711025004637

**Methodology.** Each tool ran in its own fresh interpreter process, not in
one script importing all three -- whichever imports matplotlib/cartopy first
would pay that shared cost, making the other two look artificially fast.
*cold* is the first plot a fresh process ever produces (import + mesh/topology
load + render, saved to an actual PNG); *warm* is a second plot immediately
after, in the same process, so whatever a tool caches in-memory is warm.
Hardware: 16-core i7-13620H, 15 GB RAM. Versions: gmpas 0.4.5, uxarray
2026.8.0, xarray 2026.7.0, cartopy 0.25.0, mpasviewer commit `dbdd17d`
(2026-06-17).

**Small regional mesh** -- 8,228 cells (`maritime.region.nc` +
`diag.2019-09-01_00.00.00.nc`):

| tool | import | mesh/topology load | frame 1 (cold) | frame 2 (warm) |
|---|---|---|---|---|
| gmpas (`auto` → poly at this size) | 0.43s | 0.05s | 0.94s | 0.45s |
| mpasviewer | 0.87s | 0.65s | 0.83s | 0.48s |
| uxarray (`polygons`, datashader) | 1.43s | 0.40s | 6.86s | 2.21s |

At this size all three are in the same ballpark for a single frame -- gmpas's
polygon path and mpasviewer's are close enough that either could win a given
run, and uxarray's first call pays a one-time interactive-backend
registration cost (below). The differences that matter show up at scale, not
here.

**Large mesh** -- 413,788 cells, ~5 GB file (`history.2012-02-25_12.00.00.nc`,
a real MPAS-A history output with 55 vertical levels):

| tool | mesh/topology load | frame 1 (cold) | frame 2 (warm) |
|---|---|---|---|
| gmpas (`auto` → raster; no disk cache yet) | 0.84s | 1.82s | 1.07s |
| gmpas (`auto` → raster; disk cache hit) | 0.01s | 1.47s | 1.12s |
| gmpas, forced `method="poly"` (not the default here) | 0.84s | 49.9s | -- |
| mpasviewer (always polygons, no caching) | 0.60s | 7.39s | 6.85s |
| uxarray (`polygons`, datashader) | 0.86s | 31.3s | 3.24s |

This is the real story, and it's worth being precise about what each number
means:

- **gmpas's actual behaviour at this size wins clearly**: `method="auto"`
  switches to rasterizing above 150k cells (`RASTER_THRESHOLD` in
  `raster.py`), so a real `gmpas plot`/`gmpas view` call here costs
  1.1-1.8s, against mpasviewer's 6.9-7.4s (no way to avoid drawing every
  polygon) and uxarray's steady-state 3.2s.
- **uxarray's 31s cold number is dominated by a one-time cost, not
  rendering**: the first `.plot.polygons(backend="matplotlib")` call in a
  process registers holoviews' matplotlib extension, which alone took ~7s
  even on the 8,228-cell mesh. That's a real cost a fresh notebook kernel
  pays once -- but it's also not uxarray's idiomatic path: its default is an
  interactive bokeh plot, not a static PNG, and asking it for a matplotlib
  PNG is swimming somewhat against its grain. Once warm, its datashader
  rasterizing (3.2s) is a fair, if slower, comparison to gmpas's raster path.
- **Transparently, gmpas's own polygon path is not the fast one here**:
  forced away from its own default at this size, it takes 49.9s -- *slower*
  than mpasviewer's polygon rendering (7.4s). That's expected and not
  concerning: gmpas's `auto` threshold exists specifically so nobody hits
  this path on a mesh this size, and the number is included here rather than
  omitted because the point of a comparison like this is to be checkable, not
  flattering. gmpas's real advantage isn't "a faster polygon renderer" --
  it's recognizing when to stop drawing polygons at all.
- **Only gmpas persists anything across a process restart.** mpasviewer and
  uxarray both rebuild their internal representation from scratch every time
  a fresh interpreter starts (a new Jupyter kernel, a new script run); gmpas's
  mesh geometry cache lives on disk (`~/.cache/gmpas/mesh` by default) and
  turns a second process's mesh load into a memory-map, not a rebuild. The
  saving here (~0.8s) is modest at 413k cells; it grows with mesh size, since
  the build cost that's being skipped is the O(nCells) part.

## Why it is fast

1. **Cached geometry.** Building cell polygons from `verticesOnCell` is
   vectorized (no Python loop) and cached to disk. The saving persists across
   sessions and grows with mesh size -- see the "disk cache hit" row above.
2. **Rasterizing instead of drawing polygons.** Polygon rendering costs
   O(nCells) and stalls on large meshes (49.9s at 413k cells, above). An MPAS
   mesh *is* the Voronoi tessellation of its cell centres, so the cell
   containing a point is exactly the nearest cell centre -- one KD-tree
   query, no clipping, no interpolation. That makes rendering O(pixels),
   independent of mesh size, and it's why the `auto`-selected raster path
   above stays under 2s on the same mesh.

`method="auto"` uses polygons below ~150k cells and rasterizes above.
