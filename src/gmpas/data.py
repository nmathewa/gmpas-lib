"""Opening MPAS output: pairing data with a mesh, selecting time and level."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from .mesh import MpasMesh, has_mesh
from .paths import resolve_path

#: MPAS spatial dimensions, in the order tools care about
SPATIAL_DIMS = ("nCells", "nEdges", "nVertices")

#: dimensions to slice away before plotting, with the kwarg that selects them
INDEX_DIMS = ("Time", "nVertLevels", "nVertLevelsP1", "nSoilLevels",
              "nIsoLevelsT", "nIsoLevelsZ")


def open_data(data_path: str | Path,
              mesh_path: str | Path = "") -> tuple[xr.Dataset, MpasMesh]:
    """Open a data file alongside its mesh.

    MPAS output files usually carry no mesh information -- a `diag.*.nc` has
    nCells but no `verticesOnCell`. When mesh_path is empty the data file is
    tried first, then any mesh-bearing file sitting next to it.
    """
    dpath = resolve_path(data_path)
    if not dpath.exists():
        raise FileNotFoundError(f"No such data file: {dpath}")
    ds = xr.open_dataset(dpath, decode_timedelta=False, engine="netcdf4")

    if mesh_path:
        return ds, MpasMesh.load(resolve_path(mesh_path))
    if has_mesh(ds):
        return ds, MpasMesh.load(dpath)

    found = find_mesh_beside(dpath, int(ds.sizes.get("nCells", -1)))
    if found is None:
        raise KeyError(
            f"{dpath.name} carries no mesh information and no mesh file was found "
            f"beside it. Pass mesh_path explicitly (the init/static/grid file, or "
            f"whichever file has verticesOnCell)."
        )
    return ds, MpasMesh.load(found)


def find_mesh_beside(dpath: Path, n_cells: int) -> Path | None:
    """Look for a mesh file in the same directory with a matching cell count.

    Probes each candidate with netCDF4 directly rather than xarray: this only
    needs two cheap header facts -- does it carry `verticesOnCell`, does its
    `nCells` match -- and `xr.open_dataset` would run full CF decoding
    (attrs, masking, coordinate indexes) over every variable in every
    candidate file just to answer them. A run directory can hold many files,
    so that cost is paid once per file scanned, not once total; matches
    `Series._scan`'s netCDF4-over-xarray choice for the same reason.
    """
    import netCDF4

    for cand in sorted(dpath.parent.glob("*.nc")):
        if cand == dpath:
            continue
        try:
            with netCDF4.Dataset(cand) as nc:
                dim = nc.dimensions.get("nCells")
                if has_mesh(nc) and dim is not None and len(dim) == n_cells:
                    return cand
        except Exception:
            continue
    return None


def spatial_dim(da: xr.DataArray) -> str:
    """Which MPAS mesh element this field lives on."""
    for d in SPATIAL_DIMS:
        if d in da.dims:
            return d
    raise ValueError(
        f"{da.name!r} has dims {da.dims} — none of them is an MPAS spatial "
        f"dimension ({', '.join(SPATIAL_DIMS)}), so it cannot be drawn on the mesh."
    )


def select(da: xr.DataArray, time: int = 0, level: int = 0) -> np.ndarray:
    """Reduce a field to one value per mesh element."""
    for dim in INDEX_DIMS:
        if dim in da.dims:
            idx = time if dim == "Time" else level
            n = da.sizes[dim]
            if not -n <= idx < n:
                raise IndexError(f"{dim}={idx} out of range for {da.name!r} (size {n})")
            da = da.isel({dim: idx})
    return np.asarray(da.squeeze().values, dtype=np.float64)


def field_label(da: xr.DataArray) -> str:
    """Colorbar label: long_name and units when the file provides them."""
    units = da.attrs.get("units", "")
    name = da.attrs.get("long_name", da.name)
    return f"{name} [{units}]" if units else str(name)


def plottable(ds: xr.Dataset) -> dict[str, list[str]]:
    """Group a dataset's variables by the mesh element they live on."""
    out: dict[str, list[str]] = {d: [] for d in SPATIAL_DIMS}
    for name, var in ds.data_vars.items():
        for d in SPATIAL_DIMS:
            if d in var.dims:
                out[d].append(str(name))
                break
    return out
