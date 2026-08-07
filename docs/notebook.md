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
`Series` deliberately does not try to replace it:

```python
import glob
import xarray as xr
import gmpas

# MPAS's own file naming sorts chronologically as plain text (zero-padded,
# big-endian), so a lexicographic sort is a valid Time order here -- no
# coordinate values to combine on, so combine="nested" rather than "by_coords"
files = sorted(glob.glob("/path/to/run/history.*.nc"))
ds = xr.open_mfdataset(files, combine="nested", concat_dim="Time")

mean = ds["precipw"].mean("Time").compute()   # or an anomaly, a composite, ...

mesh = gmpas.MpasMesh.load("/path/to/run/mesh.nc")
fig, ax = gmpas.cell_field(mesh, mean.values)
```

`open_mfdataset` isn't routed through `open_mpas`, so there is no `.mpas`
accessor here — attach the mesh and call `gmpas.cell_field`/`edge_field`
directly, as above.
