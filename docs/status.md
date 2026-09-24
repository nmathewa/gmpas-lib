# Status

As of 0.5.7, gmpas covers the pipeline from both ends.

**Postprocessing** — plotting, the interactive viewer, and conservative
remapping — is implemented. `gmpas remap` writes the grid files ESMF needs,
runs it, applies the result and checks that it conserved, all from the
command line; gmpas does not compute the weights itself. On a machine with
`srun` or `mpirun`/`mpiexec` available and an MPI-capable ESMF build, weight
generation uses `-j` ranks instead of one. See [REMAPPING.md](./REMAPPING.md).

gmpas does not install ESMF (or NCO) itself, on purpose — see
[issue 34](https://github.com/nmathewa/gmpas-lib/issues/34). On an HPC site,
load the site's own build (`module load esmf`) rather than a conda-forge copy
in gmpas's own environment; the two compete on `PATH`/`LD_LIBRARY_PATH`
rather than help.

**The viewer** draws a run as a fast raster and, for `--generic` files, as a
composite of layers, a Hovmoller diagram, or any of the xarray plot kinds.
Both viewers offer the same colours: matplotlib, cmocean, the Ferret palettes
and the GrADS tables, with discrete bands, out-of-range and missing colours,
reverse and a power scale, and a colorbar drawn from the very colours the
image was drawn with. Clicking the map opens that point's value and, on
request, its time series through every file of the run.

**There is a live demo** at <https://nmathewa.github.io/gmpas-lib/> — the
viewer's own page on a real MPAS run, rendering in the browser with no server
behind it, so the interface can be tried before anything is installed. See
[demo/README](demo/README.md) for what it leaves out.

**When `--generic` cannot work a file out** it does not guess and draw. The
viewer starts, nothing is drawn, and a panel asks which variable is x and y
and which dimensions are time and level; the answer is checked the same way
a detected one is, and remembered for that file. The panel is also shown,
without blocking, when the file loaded but the reading involved a guess —
an axis identified on a bare CF `axis` attribute, or a second stacking axis
pinned at 0 whose data the map cannot otherwise reach.

**Preprocessing** covers `prep view`, `prep hfun`, `prep generate`,
`prep scale`, `prep relocate` and `prep create-region`: looking at a mesh
after it exists, looking at a distance function before any mesh exists,
running JIGSAW to get from the second to the first, rescaling a regional
mesh around a tangent point, repositioning a refined region without
resizing it, and cropping a global mesh down to a regional subset. A run
leaves everything `mkgrid` reads.

## Not implemented

- **Bundling JIGSAW or mkgrid.** Both are external executables gmpas shells
  out to, named by `$JIGSAWDIR` and `$MKGRIDFILE`. `gmpas prep generate` runs
  the whole chain through to `grid.nc`, but only if you have built them
  ([issue 30](https://github.com/nmathewa/gmpas-lib/issues/30)).
- **Comparing a generated mesh against the `hfun.py` that asked for it** —
  the two are one click apart in the viewer, but nothing yet differences them.
- **Confirming the MPI launcher and the ESMF build it runs actually match**
  on real HPC hardware. The `esmf.mk`-based check only rules out a build with
  no MPI at all (`mpiuni`); a real-MPI build launched by a *different* MPI
  implementation than it was linked against fails the same uncoordinated-rank
  way and cannot be detected from outside
  ([issue 34](https://github.com/nmathewa/gmpas-lib/issues/34)).

## Known rough edges

- Browsing `prep hfun` is slower than it should be on some machines
  ([issue 25](https://github.com/nmathewa/gmpas-lib/issues/25)). The
  distance function is re-evaluated over every pixel of every new view,
  and there is a lot of headroom to exploit — it is a function, so the
  sampling is entirely ours to choose.
