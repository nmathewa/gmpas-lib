"""Opening MPAS output: pairing data with a mesh, selecting time and level."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from . import netcdf, timing
from .mesh import MpasMesh, has_mesh
from .paths import resolve_path

#: MPAS spatial dimensions, in the order tools care about
SPATIAL_DIMS = ("nCells", "nEdges", "nVertices")


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
    with netcdf.LOCK:                     # see netcdf.LOCK: HDF5 is not thread-safe
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

    with timing.step("mesh.discover") as t:
        # AppleDouble sidecars are `.nc` by name only -- see series.is_sidecar.
        # Skipping them here is not just tidiness: on a parallel filesystem
        # every candidate opened is a metadata round trip, and a Mac-copied
        # run directory doubles the number of them.
        from .series import is_sidecar
        candidates = sorted(f for f in dpath.parent.glob("*.nc")
                            if not is_sidecar(f))
        opened = 0
        try:
            for cand in candidates:
                if cand == dpath:
                    continue
                try:
                    opened += 1
                    # netcdf.LOCK per candidate rather than around the whole
                    # scan: this runs while a Series may be scanning the same
                    # directory on another thread, and holding it for every
                    # file would stall that for the entire probe.
                    with netcdf.LOCK, netCDF4.Dataset(cand) as nc:
                        dim = nc.dimensions.get("nCells")
                        if has_mesh(nc) and dim is not None and len(dim) == n_cells:
                            return cand
                except Exception:
                    continue
            return None
        finally:
            t.note(scanned=len(candidates), opened=opened)


def spatial_dim(da: xr.DataArray) -> str:
    """Which MPAS mesh element this field lives on."""
    for d in SPATIAL_DIMS:
        if d in da.dims:
            return d
    raise ValueError(
        f"{da.name!r} has dims {da.dims} — none of them is an MPAS spatial "
        f"dimension ({', '.join(SPATIAL_DIMS)}), so it cannot be drawn on the mesh."
    )


def level_dims(da: xr.DataArray) -> list[str]:
    """The stacking axes of a field: whatever is left once Time and the mesh go.

    Defined by exclusion rather than by a list of known names. The vertical
    dimension is whatever the person who wrote the diagnostic called it --
    nVertLevels and nSoilLevels from the model core, nIsoLevelsT/nIsoLevelsZ
    from MPAS's own isobaric diagnostics, or a custom nIsoLevels from a build
    that writes its own -- and a name list silently mis-plots every convention
    it has not been told about.
    """
    return [str(d) for d in da.dims if d != "Time" and d not in SPATIAL_DIMS]


def select(da: xr.DataArray, time: int = 0, level: int = 0,
           sel: dict[str, int] | None = None) -> np.ndarray:
    """Reduce a field to one value per mesh element.

    `level` indexes the field's stacking axis. Some fields have more than one
    -- `o3clim(nCells, nOznLevels, nMonths)` is ozone by level *and* by month
    -- and one index cannot mean both: taking `level` as March as well as the
    third level draws a plausible map of the wrong thing. So every axis but
    one must be pinned through `sel`, a {dim: index} mapping, and which axis
    `level` refers to stops being a guess. `remap.remappable` declines the
    same ambiguity rather than resolving it, for the same reason.

    `sel` also names an axis outright -- `sel={"nIsoLevels": 3}` -- which is
    how to reach a specific axis of a field whose dimensions this code has
    never heard of.
    """
    sel = dict(sel or {})
    unknown = [d for d in sel if d not in da.dims]
    if unknown:
        raise KeyError(
            f"{da.name!r} has dims {da.dims}, so sel={{{', '.join(unknown)}}} "
            f"selects nothing."
        )

    free = [d for d in level_dims(da) if d not in sel]
    if len(free) > 1:
        raise ValueError(
            f"{da.name!r} has {len(free)} stacking axes ({', '.join(free)}), so "
            f"level={level} is ambiguous -- it would index every one of them. "
            f"Pin all but the one you want to vary, e.g. sel={{{free[0]!r}: 0}}."
        )

    picks = {"Time": time, **sel, **{d: level for d in free}}
    for dim, idx in picks.items():
        if dim in da.dims:
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
