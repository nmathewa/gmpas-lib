# Command line

```bash
gmpas info       history.2012-02-25_12.00.00.nc
gmpas plot       history.2012-02-25_12.00.00.nc precipw -o pw.png
gmpas view       /path/to/run/
gmpas scrip      history.2012-02-25_12.00.00.nc -o src.scrip.nc
gmpas target     -o dst.scrip.nc
gmpas prep view  mesh.nc
gmpas prep hfun  hfun.py
gmpas prep generate hfun.py -o mesh/
```

`info` summarises the mesh and lists variables grouped by the element they
live on. `plot` renders one field. `view` opens the interactive browser.
`scrip` and `target` prepare the two grid files a conservative remapper needs
— see [Conservative remapping](./remapping.md).

Everything above except `prep` is **postprocessing**: it opens a run and
renders, remaps or exports it. `prep` is the other end of the pipeline, for
work that happens before there is any output — see
[Preprocessing](./preprocessing.md).

Running `gmpas` with no arguments prints the whole list with examples.

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
