# Status

As of 0.4.5, gmpas covers the pipeline from both ends.

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
