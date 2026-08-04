# Status

Plotting is implemented. Conservative remapping (`convert`) is next: the same
KD-tree gives area weights by supersampling a target cell and counting which
source cells the samples land in, which — unlike barycentric sampling —
actually conserves cell integrals.

Preprocessing now covers `prep view`, `prep hfun` and `prep generate` — looking
at a mesh after it exists, looking at one before it exists, and running JIGSAW
to get from the second to the first.

What is still missing is `mkgrid`, the last leg from JIGSAW's output to an
MPAS `grid.nc` ([issue 15](https://github.com/nmathewa/gmpas-lib/issues/15)).
It needs MPI and PnetCDF, so it cannot simply be vendored; everything feeding
it is now produced by `gmpas prep generate`.
