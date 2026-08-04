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

- **Bundling JIGSAW or mkgrid.** Both are external executables gmpas shells
  out to, named by `$JIGSAWDIR` and `$MKGRIDFILE`. `gmpas prep generate` runs
  the whole chain through to `grid.nc`, but only if you have built them
  ([issue 30](https://github.com/nmathewa/gmpas-lib/issues/30)).
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
