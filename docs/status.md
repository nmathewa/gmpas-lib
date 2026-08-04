# Status

As of 0.4.0, gmpas covers the pipeline from both ends.

**Postprocessing** — plotting, the interactive viewer, and conservative
remapping — is implemented. gmpas does not compute remapping weights itself;
it writes the grid files ESMF or TempestRemap need, applies the result and
checks that it conserved. See [REMAPPING.md](./REMAPPING.md).

**Preprocessing** covers `prep view`, `prep hfun` and `prep generate`: looking
at a mesh after it exists, looking at a distance function before any mesh
exists, and running JIGSAW to get from the second to the first. A run leaves
everything `mkgrid` reads.

## Not implemented

- **`mkgrid`**, the last leg from JIGSAW's output to an MPAS `grid.nc`
  ([issue 15](https://github.com/nmathewa/gmpas-lib/issues/15)). It needs MPI
  and PnetCDF, so it cannot simply be vendored. Everything feeding it is
  produced by `gmpas prep generate`, and the command prints the exact
  `mkgrid` line to run, with `nominalMinDc` already converted to metres.
- **Applying weights from the command line**
  ([issue 2](https://github.com/nmathewa/gmpas-lib/issues/2)). `ncremap -m`
  does it today.
- **Comparing a generated mesh against the `hfun.py` that asked for it** —
  the two are one click apart in the viewer, but nothing yet differences them.

## Known rough edges

- Browsing `prep hfun` is slower than it should be on some machines
  ([issue 25](https://github.com/nmathewa/gmpas-lib/issues/25)). The
  distance function is re-evaluated over every pixel of every new view,
  and there is a lot of headroom to exploit — it is a function, so the
  sampling is entirely ours to choose.
