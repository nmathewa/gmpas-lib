# netCDF viewer

```bash
gmpas view /path/to/run/
gmpas view reanalysis.nc    # or any netcdf file(s)
gmpas info history.2012-02-25_12.00.00.nc
gmpas plot history.2012-02-25_12.00.00.nc precipw -o pw.png
```

| command |  |
|---|---|
| `gmpas view` | GMPAS interactive viewer |
| `gmpas info` | Mesh summary |
| `gmpas plot` | Save the plot directly as png |

**gmpas view flags**

| flag (`view`) |  |
|---|---|
| `-m/--mesh` | mesh file, if not in the data's folder |
| `-p/--port`, `--host` | where the server listens (default is `127.0.0.1`) |
| `--width`, `--height` | raster size in pixels |
| `--script FILE` | open with a script applied |
| `--hfun hfun.py` | distance function |

```bash
gmpas plot '/path/to/run/history.*.nc' precipw --all-steps -o 'frames/pw_{step:04d}.png' -j 8
```

**gmpas plot flags**

| flag (`plot`) |  |
|---|---|
| `-t/--time`, `-l/--level` | step and level |
| `--cmap`, `--extent`, `--symmetric` | colours and region |
| `--method poly\|raster\|auto` | how cells are drawn |
| `--all-steps`, `-j` | every step, in parallel (`{step}` in `-o`) |

## Environment variables

| variable | default | note |
|---|---|---|
| `GMPAS_CACHE_DIR` | `~/.cache/gmpas/mesh` | geometry cache |
| `GMPAS_VALUES_CACHE_MB` | 512 | memory for fields that already read |
| `GMPAS_VIEW_CACHE_MB` | 256 | memory for viewer cache |
| `GMPAS_SERIES_WORKERS` | 8 | parallel workers for time series |
