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
(which rebuilds grid topology on every call) or geopandas (the .npz geometry
cache replaces a .gpkg and also carries edge geometry).
"""

from __future__ import annotations

from .accessor import MpasAccessor, open_mpas
from .data import field_label, open_data, plottable, select, spatial_dim
from .mesh import MpasMesh, has_mesh, reconstruct_cell_winds
from .paths import cache_dir, data_dir, resolve_path
from .plot import cell_field, edge_field, mesh_structure, vectors
from .raster import RASTER_THRESHOLD, rasterize, should_raster, target_grid
from .style import CMAPS, EXTENTS, Style, resolve_extent, save_figure

# kept in step with pyproject.toml by test_version.py -- `gmpas --version` and
# the built wheel disagreeing is the kind of thing only noticed after a release
__version__ = "0.4.1"

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
    "cache_dir", "data_dir", "resolve_path",
    "__version__",
]
