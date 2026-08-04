# Differences from the MCP server

### Two behaviour fixes

Both change output, and neither has been ported back to the MCP server yet.

**The out-of-mesh mask now uses the mesh's own sphere radius.** It converts a
cell's radius in metres to an angle, and the server divides by a hardcoded
6 371 000 m. On a reduced-radius ("small planet", X-factor) run — a real MPAS
configuration — that understates every cell's angular size by the radius
ratio. On a 1/120-radius mesh it blanks *every* pixel: the figure comes back
empty. `MpasMesh` now carries `sphere_radius`, and `rasterize` divides by that.

**The geometry cache can no longer return a different mesh's polygons.** The
server keys its cache on `path | size | mtime`. netCDF4's padded layout means
two meshes with different cell counts can occupy identical byte sizes (a
1-cell and a 3-cell file here are both 18436 bytes), and `st_mtime` truncates
to whole seconds — so regenerating a mesh in place could silently reuse the
previous geometry. The key is now
`path | size | st_mtime_ns | nCells | nEdges`, with the counts read from the
netCDF header (no data I/O). Costs a roughly constant ~2 ms per load.

A unit-sphere mesh (`sphere_radius = 1`, straight from JIGSAW/MPAS-Tools) is
still assumed to be Earth-sized, matching MPAS's own default — but that is now
a stated assumption in the code, not a buried constant. Nothing in such a file
distinguishes an Earth mesh from a small-planet one.

### Packaging

The server writes into its own repo (`data/`, `research/plots/`,
`research/mesh_cache/`). A library cannot: it is installed somewhere the user
never looks. So:

- the cache moved to `~/.cache/gmpas/mesh`, overridable by `GMPAS_CACHE_DIR`
- `save_figure(fig, path)` writes where the caller says, instead of into a
  fixed `research/plots/`
- relative paths resolve against `GMPAS_DATA_DIR` then the working directory,
  instead of a project-local `data/`
- `mesh.cell_width_km` replaces the hexagon-width formula that was repeated at
  three call sites (identical numerics)
- an unknown extent name or render method now raises instead of silently
  falling back, so a typo cannot quietly produce the wrong map

### Known, unfixed

- `is_global` is true as soon as any cell straddles the antimeridian, so a
  regional Pacific domain gets the whole-sphere extent and plots the entire
  globe. Documented by a test; not changed.
- The edge-normal wind reconstruction carries a systematic factor-of-two low
  bias, not just noise: for uniform flow across normals evenly spread over
  [0, π) the unweighted average returns exactly half the true speed. Direction
  is right. Pinned by a test; prefer `uReconstructZonal` /
  `umeridional_*` when the diagnostics file has them.
