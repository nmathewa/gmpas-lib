# In a notebook

```python
import gmpas

ds = gmpas.open_mpas("history.2012-02-25_12.00.00.nc")
ds.mpas                       # <MPAS 413,788 cells / 1,243,651 edges — ...>

ds.mpas.plot("precipw")       # returns (fig, ax); renders inline
ds.mpas.plot("theta", level=20)
ds.mpas.plot("u")             # edge field, drawn on the cell faces
ds.mpas.plot_mesh()           # where the mesh refines, in km
ds.mpas.quiver("uReconstructZonal", "uReconstructMeridional",
               background="precipw")
```

Pass `mesh=` when the file carries no mesh of its own (a `diag.*.nc` usually
does not) and no mesh file sits beside it.

The pieces are usable directly when you want the array rather than the figure:

```python
mesh = ds.mpas.mesh
mesh.n_cells, mesh.extent, mesh.is_global, mesh.coverage

values = gmpas.select(ds["precipw"], time=0)
img, lon, lat = gmpas.rasterize(mesh, values)      # regular lat-lon array
gmpas.save_figure(fig, "out.png")
```

For many files as one time axis:

```python
from gmpas.series import Series

run = Series("/path/to/run/")            # or a glob, or a list of paths
len(run), run.n_files, run.labels[:3]
values = run.values("precipw", step=7)   # opens only the file that step needs
```

For time means, composites and anomalies across a run, use
`xarray.open_mfdataset` with dask instead — that is what it is good at, and
`Series` deliberately does not try to replace it.
