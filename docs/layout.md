# Layout

- `src/gmpas/mesh.py` — `MpasMesh`, geometry build, on-disk `.npy` cache, wind reconstruction
- `src/gmpas/raster.py` — KD-tree Voronoi rasterizer
- `src/gmpas/plot.py` — cell / edge / vector / mesh-structure rendering
- `src/gmpas/data.py` — opening output, pairing with a mesh, time/level selection
- `src/gmpas/style.py` — `Style` presets, figure colormaps, named extents, `save_figure`
- `src/gmpas/viewer.py` — the browser viewer for MPAS output, and the shared page
- `src/gmpas/generic.py` — the same viewer for any regular lat-lon file (`--generic`)
- `src/gmpas/colour.py` — palettes and colour options, for both viewers
- `src/gmpas/palettes/` — the cmocean, Ferret and GrADS colour tables, and the fast map's encoder
- `src/gmpas/layers.py` — `--generic` composite maps: the layer stack and its options
- `src/gmpas/accessor.py` — the `ds.mpas` xarray accessor
- `src/gmpas/paths.py` — cache and data directory resolution
- `src/gmpas/prep/` — preprocessing: the `gmpas prep` commands and their layout
