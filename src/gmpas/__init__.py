"""gmpas — fast native-mesh plotting for MPAS output.

MPAS is unstructured: a variable-resolution spherical Voronoi tessellation with
scalars at cell centres and normal velocity on cell edges. gmpas draws it on
that mesh directly, with no regridding, so refinement survives into the figure
exactly as the model carries it.

    import gmpas

    ds = gmpas.open_mpas("diag.2019-09-01_00.00.00.nc", mesh="maritime.region.nc")
    ds.mpas.plot("mslp")        # cell field on the native Voronoi mesh
    ds.mpas.plot("u")           # edge field, drawn on the cell faces
    ds.mpas.plot_mesh()         # where the mesh refines

Depends on numpy, scipy, xarray and netCDF4; matplotlib and cartopy are only
needed to actually draw, and are imported lazily. It does not use uxarray
(which rebuilds grid topology every fresh process -- see
docs/benchmarks/ for the comparison) or geopandas (the on-disk geometry cache replaces a .gpkg and also
carries edge geometry).
"""

from __future__ import annotations

import importlib
import importlib.util
import sys

# Every public name is loaded on first use (PEP 562), not here. Importing the
# package is what every process that touches it pays -- the reader workers
# included, which need netCDF4 and nothing else -- and loading the modules
# behind these names brings in xarray and pandas (#129).
_LAZY = {
    "accessor": ("MpasAccessor", "open_mpas"),
    "data": ("field_label", "open_data", "plottable", "select", "spatial_dim"),
    "mesh": ("MpasMesh", "has_mesh", "reconstruct_cell_winds"),
    "paths": ("cache_dir", "resolve_path"),
    "plot": ("cell_field", "edge_field", "mesh_structure", "vectors"),
    "raster": ("RASTER_THRESHOLD", "rasterize", "should_raster", "target_grid"),
    "style": ("CMAPS", "EXTENTS", "Style", "resolve_extent", "save_figure"),
}
_HOME = {name: module for module, names in _LAZY.items() for name in names}


def __getattr__(name: str):
    if name in _HOME:
        value = getattr(importlib.import_module(f".{_HOME[name]}", __name__), name)
        globals()[name] = value             # once; later lookups skip this
        return value
    # a submodule not imported yet: `gmpas.series` after a bare `import gmpas`
    if importlib.util.find_spec(f"{__name__}.{name}") is not None:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))


class _AfterXarray:
    """Registers `ds.mpas` as soon as xarray is imported, if it was not yet.

    The accessor has to be registered with xarray, so it cannot be until
    xarray is loaded -- and `import gmpas` no longer loads it. A finder on
    `sys.meta_path` sees the xarray import go by, lets it run, and registers
    the accessor right after, so `ds.mpas` works whichever of the two a
    script imports first. It removes itself on first use.
    """

    def find_spec(self, name, path=None, target=None):
        if name != "xarray":
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.loader is None:
            return spec
        load = spec.loader.exec_module

        def exec_module(module):
            load(module)
            importlib.import_module(".accessor", __name__)

        spec.loader.exec_module = exec_module
        return spec


if "xarray" in sys.modules:
    importlib.import_module(".accessor", __name__)
else:
    sys.meta_path.insert(0, _AfterXarray())

# kept in step with pyproject.toml by test_version.py -- `gmpas --version` and
# the built wheel disagreeing is the kind of thing only noticed after a release
__version__ = "0.9.0"

__all__ = [
    # entry points
    "open_mpas", "open_data", "MpasMesh", "MpasAccessor",
    # rendering
    "cell_field", "edge_field", "vectors", "mesh_structure", "save_figure",
    # geometry / sampling
    "rasterize", "target_grid", "should_raster", "RASTER_THRESHOLD",
    "reconstruct_cell_winds",
    # data access
    "select", "spatial_dim", "field_label", "plottable", "has_mesh",
    # configuration
    "Style", "EXTENTS", "CMAPS", "resolve_extent",
    "cache_dir", "resolve_path",
    "__version__",
]
