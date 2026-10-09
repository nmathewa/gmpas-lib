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

#: names a time dimension has carried in files seen in practice, used only
#: when the dimension carries no coordinate whose attributes say what it is.
#: `Time` is MPAS's own; `time`, `valid_time` and `xtime` are what ncrcat,
#: CDO and an xarray round-trip produce. `t` is deliberately NOT here: it is
#: a plausible variable-specific axis, and the name fallback exists for bare
#: record dimensions where there is nothing else to go on.
TIME_NAMES = {"time", "times", "valid_time", "xtime"}


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


def detect_kind(paths, mesh_path: str = "") -> tuple[str, str]:
    """Whether `paths` is MPAS output or a regular lat/lon grid, and why.

    Returns ("mpas" | "generic" | "ask", one line for the terminal). Reads the
    first file that opens, header and 1D coordinates only.

    - "mpas": the file has an `nCells` dimension. That alone decides it, mesh
      or no mesh: an MPAS diag file without its mesh must reach the MPAS
      reader's own "pass -m mesh.nc" message, never be drawn as something else.
    - "generic": no `nCells`, and the --generic reader finds a 1D latitude and
      a 1D longitude on separate dimensions -- the same `_find_axis` it opens
      with, so the two cannot disagree.
    - "ask": neither. The --generic viewer opens in its needs-setup state and
      the page asks which dimension is which.
    """
    import netCDF4

    from .series import expand

    files = expand(paths)
    first, n_cells, inside = None, None, False
    for f in files:
        try:
            with netcdf.LOCK, netCDF4.Dataset(f) as nc:
                dim = nc.dimensions.get("nCells")
                n_cells = len(dim) if dim is not None else None
                inside = has_mesh(nc)
            first = f
            break
        except Exception:
            continue
    if first is None:
        # the reader that opens these next says which files and why
        return "ask", "no file could be read to tell what it is"

    if n_cells is not None:
        what = f"MPAS output · {n_cells:,} cells"
        if inside:
            return "mpas", what
        if mesh_path:
            return "mpas", f"{what} · mesh {Path(mesh_path).name}"
        found = find_mesh_beside(first, n_cells)
        if found is not None:
            return "mpas", f"{what} · mesh {found.name} (beside it)"
        return "mpas", f"{what} · no mesh found beside it"

    from .generic import _find_axis

    try:
        with netcdf.LOCK, xr.open_dataset(first, decode_times=False,
                                          engine="netcdf4") as ds:
            lat, lat_dim = _find_axis(ds, "lat")
            lon, lon_dim = _find_axis(ds, "lon")
            sizes = (ds.sizes[lat_dim], ds.sizes[lon_dim])
    except (ValueError, KeyError) as exc:
        return "ask", f"no nCells and no regular lat/lon grid ({exc})"
    if lat_dim == lon_dim:
        return "ask", (f"no nCells, and {lat!r}/{lon!r} share the dimension "
                       f"{lat_dim!r} -- points, not a grid")
    return "generic", f"regular lat/lon grid · {sizes[0]} x {sizes[1]} ({lat}, {lon})"


def spatial_dim(da: xr.DataArray) -> str:
    """Which MPAS mesh element this field lives on."""
    for d in SPATIAL_DIMS:
        if d in da.dims:
            return d
    raise ValueError(
        f"{da.name!r} has dims {da.dims} — none of them is an MPAS spatial "
        f"dimension ({', '.join(SPATIAL_DIMS)}), so it cannot be drawn on the mesh."
    )


def time_axis(da: xr.DataArray) -> str | None:
    """The field's time dimension, or None when it has one.

    The first of its dimensions `is_time_dim` accepts. At most one is ever
    recognised: a field with two time-like axes is not something this has
    seen, and picking between them by anything other than position would be
    a guess dressed as detection.
    """
    for d in da.dims:
        if is_time_dim(da, str(d)):
            return str(d)
    return None


def level_dims(da: xr.DataArray) -> list[str]:
    """The stacking axes of a field: whatever is left once time and the mesh go.

    Defined by exclusion rather than by a list of known names. The vertical
    dimension is whatever the person who wrote the diagnostic called it --
    nVertLevels and nSoilLevels from the model core, nIsoLevelsT/nIsoLevelsZ
    from MPAS's own isobaric diagnostics, or a custom nIsoLevels from a build
    that writes its own -- and a name list silently mis-plots every convention
    it has not been told about.
    """
    return [str(d) for d in da.dims
            if d not in SPATIAL_DIMS and not is_time_dim(da, d)]


def is_time_dim(da: xr.DataArray, dim: str) -> bool:
    """Whether `dim` is a field's time axis, by CF evidence, then by name.

    The attributes of the dimension's own coordinate variable are the
    evidence: `standard_name: time`, `axis: T`, a `units` of the
    `<unit> since <date>` shape, a datetime dtype. That is how the --generic
    path already recognises time (generic._is_time), and it reads the file
    rather than trusting a spelling.

    The name is the fallback, for the bare record dimension MPAS history
    output usually is -- `Time` with no coordinate at all. The exact
    spelling `Time` is accepted whatever else, because that is the model
    core's own convention; anything else must be lower-case in TIME_NAMES.
    Case matters there: `TIME` is also plausible as a vertical axis spelled
    in caps, and a name-only guess at that spelling is exactly how issue 123
    misread `time` -- a time axis -- as a level, pinning the real vertical
    axis out of reach.
    """
    if dim == "Time":
        return True
    if dim not in da.dims:
        return is_time_name(dim)
    # The dimension's own coordinate is the evidence, and a DataArray carries
    # it on .coords whether attached to a dataset or not. Without one -- the
    # bare record dimension MPAS history output usually is -- only the name
    # can speak.
    if dim not in da.coords:
        return is_time_name(dim)
    coord = da.coords[dim]
    return is_time_coord(dim, coord.attrs, coord.dtype, coord.encoding)


def is_time_name(dim: str) -> bool:
    """Whether a dimension with no coordinate is time: by name alone.

    The exact spelling `Time`, or one of TIME_NAMES in lower case. Shared by
    the MPAS reader and `--generic`, so a file reads the same in both.
    """
    return dim == "Time" or dim in TIME_NAMES


def is_time_coord(dim: str, attrs, dtype=None, encoding=None) -> bool:
    """Whether a dimension whose coordinate carries `attrs` is time.

    The CF evidence `is_time_dim` describes, then the name. `encoding` is
    where xarray moves `units` once it has decoded the values -- a noleap or
    360_day axis decodes to cftime objects, not datetime64, and its
    `<unit> since <date>` would otherwise be missed. `attrs` may be any
    mapping, so a raw netCDF4 variable's attributes do as well as xarray's.
    """
    if dim == "Time":
        return True
    if str(attrs.get("standard_name", "")).lower() == "time":
        return True
    if str(attrs.get("axis", "")).upper() == "T":
        return True
    if str(attrs.get("_CoordinateAxisType", "")).lower() == "time":
        return True
    if " since " in str(attrs.get("units", "")) \
            or " since " in str((encoding or {}).get("units", "")):
        return True
    if dtype is not None and np.issubdtype(dtype, np.datetime64):
        return True
    return is_time_name(dim)


def select(da: xr.DataArray, time: int = 0, level: int = 0,
           sel: dict[str, int] | None = None) -> np.ndarray:
    """Reduce a field to one value per mesh element.

    `time` indexes the field's time axis -- found by `is_time_dim`, which
    reads CF attributes and so accepts `time` as well as MPAS's own `Time`.

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

    # The time axis is indexed by `time`, not `sel`: naming it there would be
    # a second way to say the same thing, and the two disagreeing is a silent
    # wrong frame.
    t_axis = time_axis(da)
    if t_axis is not None:
        sel.pop(t_axis, None)

    free = [d for d in level_dims(da) if d not in sel]
    if len(free) > 1:
        raise ValueError(
            f"{da.name!r} has {len(free)} stacking axes ({', '.join(free)}), so "
            f"level={level} is ambiguous -- it would index every one of them. "
            f"Pin all but the one you want to vary, e.g. sel={{{free[0]!r}: 0}}."
        )

    picks = {**({t_axis: time} if t_axis is not None else {}),
             **sel, **{d: level for d in free}}
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
