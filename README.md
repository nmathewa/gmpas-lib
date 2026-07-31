# gmpas

Fast plotting of MPAS output **on its own native mesh** — no regridding, so
variable resolution is preserved exactly as the model carries it.

This is the installable-library form of the
[gmpas MCP server](../gmpas): the same geometry, caching and rendering code,
packaged so it can be `pip install`ed and imported from anywhere.

```python
import gmpas

ds = gmpas.open_mpas("diag.2019-09-01_00.00.00.nc", mesh="maritime.region.nc")

ds.mpas.plot("mslp")        # cell field, filled Voronoi polygons
ds.mpas.plot("u")           # edge field, drawn on the cell faces themselves
ds.mpas.plot("mslp", time=3)
ds.mpas.plot_mesh()         # where the mesh actually refines, in km
ds.mpas.quiver("uzonal_850hPa", "umeridional_850hPa", background="mslp")
```

`mesh=` may be omitted when the file carries its own mesh information, or when
a mesh file with a matching cell count sits beside it.

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

## Install anywhere

Nothing here needs an MCP server, Claude Desktop, or any of the surrounding
tooling — it is an ordinary Python package. Copy the directory (or clone it)
onto the target machine and install it.

The scientific stack is much easier from conda-forge than from pip:

```bash
conda create -n gmpas -c conda-forge python=3.12 numpy scipy xarray netcdf4 matplotlib cartopy pillow -y
conda activate gmpas
pip install -e ".[dev]" --no-deps
```

Pure pip works too, if wheels are available for your platform:

```bash
pip install -e ".[dev]"
```

Extras: `plot` pulls in matplotlib and cartopy, `test` pulls in pytest, `dev`
is both plus ruff. The core install has no plotting dependency at all —
geometry, caching and rasterizing are usable headless, and `plot.py` imports
matplotlib lazily so `import gmpas` never pays for it.

Check it landed:

```bash
gmpas --version && pytest -q
```

## Command line

```bash
gmpas info    history.2012-02-25_12.00.00.nc
gmpas plot    history.2012-02-25_12.00.00.nc precipw -o pw.png
gmpas view    /path/to/run/
```

`info` summarises the mesh and lists variables grouped by the element they
live on. `plot` renders one field. `view` opens the interactive browser.

Every command takes a file, a directory, or a glob. A directory or glob is
read as one time series across files, which is how MPAS actually writes
output — one `history.YYYY-MM-DD_HH.MM.SS.nc` per interval:

```bash
gmpas plot '/path/to/run/history.*.nc' precipw --all-steps -o 'frames/pw_{step:04d}.png' -j 8
```

`--all-steps` needs `{step}` in the output pattern; `-j` sets the worker count
(default: every core). Useful flags on `plot`: `-t/--time`, `-l/--level`,
`--cmap`, `--extent`, `--symmetric` for anomalies, `--method poly|raster|auto`,
`--style paper|poster|notebook|mesh`.

## In a notebook

```python
import gmpas

ds = gmpas.open_mpas("history.2012-02-25_12.00.00.nc")
ds.mpas                       # <MPAS 413,788 cells / 1,243,651 edges — ...>

ds.mpas.plot("precipw")       # returns (fig, ax); renders inline
ds.mpas.plot("theta", level=20)
ds.mpas.plot("u")             # edge field, drawn on the cell faces
ds.mpas.plot_mesh()           # where the mesh refines, in km
ds.mpas.quiver("uReconstructZonal", "uReconstructMeridional",
               background="precipw")
```

Pass `mesh=` when the file carries no mesh of its own (a `diag.*.nc` usually
does not) and no mesh file sits beside it.

The pieces are usable directly when you want the array rather than the figure:

```python
mesh = ds.mpas.mesh
mesh.n_cells, mesh.extent, mesh.is_global, mesh.coverage

values = gmpas.select(ds["precipw"], time=0)
img, lon, lat = gmpas.rasterize(mesh, values)      # regular lat-lon array
gmpas.save_figure(fig, "out.png")
```

For many files as one time axis:

```python
from gmpas.series import Series

run = Series("/path/to/run/")            # or a glob, or a list of paths
len(run), run.n_files, run.labels[:3]
values = run.values("precipw", step=7)   # opens only the file that step needs
```

For time means, composites and anomalies across a run, use
`xarray.open_mfdataset` with dask instead — that is what it is good at, and
`Series` deliberately does not try to replace it.

## On a cluster

The viewer serves on `127.0.0.1`, so forward the port rather than exporting a
display — this is the case where ncview's X11 forwarding hurts most:

```bash
ssh -L 8765:localhost:8765 user@cluster
```

then on the node, inside your interactive job:

```bash
gmpas view /scratch/run/ --no-browser --port 8765
```

and open `http://localhost:8765` in your own browser.

Two environment variables matter on a shared machine:

- `GMPAS_CACHE_DIR` — where cached mesh geometry goes. Defaults to
  `~/.cache/gmpas/mesh`; point it at scratch if your home quota is small.
  The cache is memory-mapped, so several processes reading the same mesh share
  one copy through the page cache.
- `GMPAS_DATA_DIR` — tried first when resolving relative paths.

Batch rendering scales with `-j`, but two things limit it: worker startup
(cheap under `fork` on Linux, expensive under `spawn` on macOS), and the
KD-tree query inside each worker still requesting every core, which
oversubscribes when many workers run at once. Measured on 12 steps: 10.2 s at
`-j 1`, 5.7 s at `-j 4`, 6.2 s at `-j 10`. Until the query's worker count is
plumbed through, `-j` around half your cores is the sweet spot.

## Configuration

Two environment variables, both optional:

- `GMPAS_CACHE_DIR` — where cached `.npz` mesh geometry goes.
  Defaults to `~/.cache/gmpas/mesh`. Safe to delete; it rebuilds.
- `GMPAS_DATA_DIR` — tried first when resolving a relative path, before the
  working directory.

## Tests

```bash
pytest
```

Tests build synthetic MPAS mesh files in `tmp_path`, so no model output is
needed and the real cache directory is never touched. They pin down the parts
a reader cannot eyeball: the ragged 1-based `verticesOnCell` fill, the
antimeridian unwrap, the `sphere_radius = 1` redimensionalisation, cache
identity, and the factor-of-two bias in the edge-normal wind reconstruction.
Rendering tests skip themselves when the `plot` extra is absent.

## Layout

- `src/gmpas/mesh.py` — `MpasMesh`, geometry build, `.npz` cache, wind reconstruction
- `src/gmpas/raster.py` — KD-tree Voronoi rasterizer
- `src/gmpas/plot.py` — cell / edge / vector / mesh-structure rendering
- `src/gmpas/data.py` — opening output, pairing with a mesh, time/level selection
- `src/gmpas/style.py` — `Style` presets, colormaps, named extents, `save_figure`
- `src/gmpas/accessor.py` — the `ds.mpas` xarray accessor
- `src/gmpas/paths.py` — cache and data directory resolution

## Differences from the MCP server

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

## Status

Plotting is implemented. Conservative remapping (`convert`) is next: the same
KD-tree gives area weights by supersampling a target cell and counting which
source cells the samples land in, which — unlike barycentric sampling —
actually conserves cell integrals.
