# gmpas

A graphical MPAS (Model for Prediction Across Scales) viewer supports Native Voronoi mesh grid (and UGRID). Additionally any netCDF with lat-lon grid. 

**[Try the viewer in your browser](https://nmathewa.github.io/gmpas-lib/)** 

```bash
pip install "gmpas[plot]"
gmpas view /path/to/run/
```

| install | what it brings |
| --- | --- |
| `pip install gmpas` | the core: mesh geometry, caching, remap weights; no viewer |
| `pip install "gmpas[plot]"` | gmpas viewer  |
| `pip install "gmpas[plot,chat]"` | Built in chat model and LLM API support |
| in a conda environment, or from source | see [Installation](https://github.com/nmathewa/gmpas-lib#installation) in the README |

<video width="100%" autoplay loop muted playsinline controls>
  <source src="_static/gmpas-3.75km-colours.webm" type="video/webm">
  <source src="_static/gmpas-3.75km-colours.mp4" type="video/mp4">
</video>

*`gmpas view` on a 3.75 km global MPAS mesh (DYAMOND2, 41.9 million cells).*

```python
# Scripting

import gmpas

ds = gmpas.open_mpas("diag.2019-09-01_00.00.00.nc", mesh="maritime.region.nc")
ds.mpas.plot("mslp")        # cell field, filled Voronoi polygons
ds.mpas.plot_mesh()         # where the mesh actually refines, in km
```

---

## Comparion with similar tools


[`uxarray`](https://github.com/UXARRAY/uxarray), the general-purpose
unstructured-grid analysis library, and
[MPAS-Viewer](https://github.com/jhbravo/mpasviewer), ([Mendez & Temimi 2026, SoftwareX][mpasviewer-paper]).

[mpasviewer-paper]: https://www.sciencedirect.com/science/article/pii/S2352711025004637

Hardware: 16-core i7-13620H, 15 GB RAM. Versions: gmpas 0.4.5, uxarray
2026.8.0, xarray 2026.7.0, cartopy 0.25.0, mpasviewer commit `dbdd17d`
(2026-06-17).

**Small regional mesh** -- 8,228 cells 

| Tool | Import | mesh/topology load | frame 1 (cold) | frame 2 (warm) |
|---|---|---|---|---|
| gmpas (`auto` → poly at this size) | 0.43s | 0.05s | 0.94s | 0.45s |
| mpasviewer | 0.87s | 0.65s | 0.83s | 0.48s |
| uxarray (`polygons`, datashader) | 1.43s | 0.40s | 6.86s | 2.21s |

> frame 1 is the first `.plot()` call, frame 2 is the second. gmpas's "mesh/topology load" is the one-time cost of building and caching the mesh geometry; mpasviewer and uxarray rebuild it every time.

**Large mesh** -- 413,788 cells, ~5 GB file 

MPAS-A output with 55 vertical levels:

| tool | mesh/topology load | frame 1 (cold) | frame 2 (warm) |
|---|---|---|---|
| gmpas (`auto` → raster; no disk cache yet) | 0.84s | 1.82s | 1.07s |
| gmpas (`auto` → raster; disk cache hit) | 0.01s | 1.47s | 1.12s |
| gmpas, forced `method="poly"` (not the default here) | 0.84s | 49.9s | -- |
| mpasviewer (always polygons, no caching) | 0.60s | 7.39s | 6.85s |
| uxarray (`polygons`, datashader) | 0.86s | 31.3s | 3.24s |

This is the real story, and it's worth being precise about what each number
means:


### Why it is fast

1. **Cached geometry.** Building cell polygons from `verticesOnCell` is
   vectorized (no Python loop) and cached to disk. The saving persists across
   sessions and grows with mesh size.
2. **Rasterizing instead of drawing polygons.** Polygon rendering costs
   O(nCells) and stalls on large meshes (49.9s at 413k cells, above). An MPAS
   mesh is a Voronoi tessellation of its cell centres, so the cell
   containing a point is exactly the nearest cell centre  That makes rendering O(pixels)

```{toctree}
:hidden:

Overview <self>
netCDF viewer <viewer>
```

```{toctree}
:maxdepth: 2
:hidden:
:caption: MPAS mesh tools (experimental)

preprocessing
REMAPPING
```

```{toctree}
:hidden:

Scripting and chat <chat>
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Development

api
layout
testing
```
