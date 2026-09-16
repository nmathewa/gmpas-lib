# Configuration

Environment variables, all optional:

- `GMPAS_CACHE_DIR` — where cached mesh geometry (a directory of `.npy` arrays) goes.
  Defaults to `~/.cache/gmpas/mesh`. Safe to delete; it rebuilds.
- `GMPAS_DATA_DIR` — tried first when resolving a relative path, before the
  working directory.

## Memory budgets

Both are sized in **bytes, not entries**. One field is a couple of megabytes on
a small regional mesh and 320 MB on a 3.75 km global one, so a cache bounded by
a count of entries is a cache whose real footprint swings by a factor of a
hundred with the mesh.

- `GMPAS_VALUES_CACHE_MB` — field values held in memory per open series.
  Defaults to 512.
- `GMPAS_VIEW_CACHE_MB` — pixel-to-cell indices and coastline overlays held per
  viewer. Defaults to 256. An entry costs `nx * ny * 9` bytes, so it scales with
  the browser window rather than with the mesh.
- `GMPAS_SERIES_WORKERS` — how many files a point series opens at once.
  Defaults to 8, or the core count if that is smaller; `1` reads serially.
  Worth raising on a compute node with a parallel filesystem, where the cost is
  the round trip to the metadata server rather than any CPU: simulating 5 ms
  per open, 1000 files took 5.2 s serially, 0.7 s with 8 workers and 0.3 s with
  16. Worth leaving alone on a shared login node.
- `GMPAS_SERIES_POOL_IDLE` — seconds those workers stay alive after a series
  finishes, so the next point clicked does not pay to start them again.
  Defaults to 120; `0` shuts them down after every read. Starting eight takes
  about 490 ms, which was the bulk of every click before they were kept; a
  second click on a warm pool is ~45 ms. The price is memory — eight idle
  readers are roughly 500 MB — so set this to `0` under a hard memory cap, and
  accept the half second.

## Seeing where the time goes

- `GMPAS_TIMING` — `1` reports startup-scale stages to stderr, `2` adds
  per-frame stages and peak-memory deltas. Every run ends with a roll-up
  totalling each stage, which is the useful thing to paste into an issue.
- `GMPAS_TIMING_FILE` — write those lines to a file instead of stderr. Worth
  setting for `plot --all-steps -j N`, where N workers otherwise interleave
  their output.

```
$ GMPAS_TIMING=1 gmpas info run/
gmpas.timing  mesh.discover      0.013s  scanned=13 opened=12
gmpas.timing  series.scan        0.012s  files=13 opened=12
gmpas.timing  mesh.tree_build    0.006s  cells=300
```

`opened=` is the count that matters on a parallel filesystem: each one is a
network round trip, and both of those stages grow with the number of files in
the run directory rather than with the size of the mesh.
