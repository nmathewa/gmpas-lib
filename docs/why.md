# Why gmpas exists

## Why

MPAS is unstructured: a variable-resolution spherical Voronoi tessellation,
with scalars at cell centres and normal velocity on cell edges. That is why
ordinary lat-lon tooling does not apply, and why the usual workarounds each
cost something:

| approach | fast? | conserves? | edge variables? |
|---|---|---|---|
| `uxarray` | rebuilds grid topology on every call | n/a | yes |
| `convert_mpas` → lat-lon | yes | **no** — barycentric sampling | via remap |
| cell polygons in a `.gpkg` | yes | n/a | **no** — centres only |

gmpas keeps the good idea behind the `.gpkg` approach — mesh geometry is static
for a whole simulation, so build it once — and generalises it. The cache is an
`.npz` holding cell polygons, edge segments and unit-sphere coordinates, so
edge-based fields work too, and no GIS dependency is needed.

## Why it is fast

1. **Cached geometry.** Building cell polygons from `verticesOnCell` is
   vectorized (no Python loop) and then cached to `.npz`. On a small regional
   mesh that is ~63 ms to build and ~1 ms to reload; the saving grows with mesh
   size, and it persists across sessions.
2. **Rasterizing instead of drawing polygons.** Polygon rendering costs
   O(nCells) and stalls on million-cell meshes. An MPAS mesh *is* the Voronoi
   tessellation of its cell centres, so the cell containing a point is exactly
   the nearest cell centre — one KD-tree query, no clipping, no interpolation.
   That makes rendering O(pixels), independent of mesh size.

`method="auto"` uses polygons below ~150k cells and rasterizes above.
