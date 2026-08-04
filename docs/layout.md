# Layout

- `src/gmpas/mesh.py` — `MpasMesh`, geometry build, `.npz` cache, wind reconstruction
- `src/gmpas/raster.py` — KD-tree Voronoi rasterizer
- `src/gmpas/plot.py` — cell / edge / vector / mesh-structure rendering
- `src/gmpas/data.py` — opening output, pairing with a mesh, time/level selection
- `src/gmpas/style.py` — `Style` presets, colormaps, named extents, `save_figure`
- `src/gmpas/accessor.py` — the `ds.mpas` xarray accessor
- `src/gmpas/paths.py` — cache and data directory resolution
- `src/gmpas/prep/` — preprocessing: the `gmpas prep` commands and their layout
