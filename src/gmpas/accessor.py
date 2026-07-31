"""An xarray accessor for MPAS output, so plotting is one line.

xarray gives you `ds.t2m.plot()` because it knows the field's coordinates. MPAS
output has no such luxury: a `diag.*.nc` is a bare (Time, nCells) array with the
geometry living in a separate mesh file. This accessor supplies the missing
half, then gets out of the way::

    import gmpas
    ds = gmpas.open_mpas("diag.2019-09-01_00.00.00.nc", mesh="maritime.region.nc")

    ds.mpas.plot("mslp")              # cell field, native Voronoi mesh
    ds.mpas.plot("u")                 # edge field -- drawn on the faces
    ds.mpas.plot("mslp", time=3)
    ds.mpas.plot_mesh()               # where the mesh refines

The mesh is loaded once per dataset and cached on disk between sessions, so the
second plot -- and every plot in every later session -- skips the build.
"""

from __future__ import annotations

import xarray as xr

from . import data as _data
from . import plot as _plot
from .mesh import MpasMesh, has_mesh
from .paths import resolve_path

MESH_ATTR = "gmpas_mesh_path"


def open_mpas(data_path: str, mesh: str = "", **kwargs) -> xr.Dataset:
    """Open MPAS output with its mesh attached, ready for `.mpas.plot(...)`.

    `mesh` may be omitted when the file carries its own mesh information, or
    when a mesh file with a matching cell count sits in the same directory.
    """
    kwargs.setdefault("decode_timedelta", False)
    path = resolve_path(data_path)
    ds = xr.open_dataset(path, **kwargs)

    if mesh:
        ds.attrs[MESH_ATTR] = str(resolve_path(mesh))
    elif has_mesh(ds):
        ds.attrs[MESH_ATTR] = str(path)
    else:
        found = _data.find_mesh_beside(path, int(ds.sizes.get("nCells", -1)))
        if found is not None:
            ds.attrs[MESH_ATTR] = str(found)
    return ds


@xr.register_dataset_accessor("mpas")
class MpasAccessor:
    """`ds.mpas` -- native-mesh plotting for an MPAS dataset."""

    def __init__(self, ds: xr.Dataset):
        self._ds = ds
        self._mesh: MpasMesh | None = None

    # -- mesh ------------------------------------------------------------

    @property
    def mesh(self) -> MpasMesh:
        """The mesh backing this dataset, loaded once and cached."""
        if self._mesh is None:
            path = self._ds.attrs.get(MESH_ATTR)
            if not path:
                raise ValueError(
                    "No mesh attached to this dataset. Open it with "
                    "gmpas.open_mpas(path, mesh=...), or call "
                    "ds.mpas.use_mesh('mesh.nc')."
                )
            self._mesh = MpasMesh.load(path)
        return self._mesh

    def use_mesh(self, mesh_path: str) -> "MpasAccessor":
        """Attach (or replace) the mesh for this dataset."""
        self._ds.attrs[MESH_ATTR] = str(resolve_path(mesh_path))
        self._mesh = None
        return self

    # -- introspection ---------------------------------------------------

    @property
    def variables(self) -> dict[str, list[str]]:
        """Plottable variables grouped by mesh element."""
        return _data.plottable(self._ds)

    def __repr__(self) -> str:
        try:
            m = self.mesh
            head = f"<MPAS {m.n_cells:,} cells / {m.n_edges:,} edges — {m.path.name}>"
        except ValueError:
            head = "<MPAS dataset (no mesh attached)>"
        groups = self.variables
        counts = ", ".join(f"{len(v)} on {k}" for k, v in groups.items() if v)
        return f"{head}\n  {counts}"

    # -- plotting --------------------------------------------------------

    def plot(self, variable: str, time: int = 0, level: int = 0, **kwargs):
        """Plot a variable on the native mesh, dispatching on where it lives.

        Cell fields fill the Voronoi polygons; edge fields are drawn on the cell
        faces they belong to. Returns (fig, ax) like a matplotlib call.
        """
        if variable not in self._ds:
            raise KeyError(
                f"{variable!r} not in dataset. Available: {self.variables}"
            )
        da = self._ds[variable]
        dim = _data.spatial_dim(da)
        values = _data.select(da, time=time, level=level)
        kwargs.setdefault("label", _data.field_label(da))
        kwargs.setdefault("title", variable)

        if dim == "nCells":
            return _plot.cell_field(self.mesh, values, **kwargs)
        if dim == "nEdges":
            return _plot.edge_field(self.mesh, values, **kwargs)
        raise NotImplementedError(
            f"{variable!r} lives on {dim}; only cell and edge fields are "
            f"plotted directly. Vertex fields need the dual triangle mesh."
        )

    def quiver(self, u: str, v: str, time: int = 0, level: int = 0,
               background: str = "", **kwargs):
        """Quiver cell-centred vector components, optionally over a filled field."""
        uu = _data.select(self._ds[u], time=time, level=level)
        vv = _data.select(self._ds[v], time=time, level=level)
        bg = _data.select(self._ds[background], time=time, level=level) \
            if background else None
        return _plot.vectors(self.mesh, uu, vv, background=bg, **kwargs)

    def plot_mesh(self, **kwargs):
        """Show the mesh itself, cells coloured by width in km."""
        return _plot.mesh_structure(self.mesh, **kwargs)
