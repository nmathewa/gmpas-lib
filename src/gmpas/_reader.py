"""What a reader process runs: open a netCDF file, read a little, answer.

The point series and the step scan hand their per-file work to worker
processes (`series.ReaderPool`). A worker unpickles its function by module
name, so whatever that module imports, every worker imports too. This one
needs numpy and netCDF4 and nothing else: not xarray or pandas, which cost a
worker about 200 ms and 35 MB each to load for nothing (#129).

`series` re-exports these names; this module must not import any other part
of gmpas.
"""

from __future__ import annotations

import time

import numpy as np

#: MPAS spatial dimensions, in the order tools care about
SPATIAL_DIMS = ("nCells", "nEdges", "nVertices")

#: names a time dimension has carried in files seen in practice, used only
#: when the dimension carries no coordinate whose attributes say what it is.
#: `Time` is MPAS's own; `time`, `valid_time` and `xtime` are what ncrcat,
#: CDO and an xarray round-trip produce. `t` is deliberately NOT here: it is
#: a plausible variable-specific axis, and the name fallback exists for bare
#: record dimensions where there is nothing else to go on.
TIME_NAMES = {"time", "times", "valid_time", "xtime"}


def _warm_worker(_):
    """Make a worker do its imports before a real read needs them."""
    import netCDF4  # noqa: F401

    time.sleep(0.05)      # stay busy, so the next warm task starts a new worker
    return True


def _read_point(job) -> list:
    """See `_read_one_file`; the pool calls this."""
    return _read_one_file(job)


def _read_one_file(job) -> list:
    """One file's worth of a point series, in a worker process.

    Deliberately a module-level function taking plain data: it is pickled to
    the worker, so it must not close over a Series, a mesh or anything else
    that would drag the package -- and the point of the worker is to do
    nothing but open, read one hyperslab, and answer.
    """
    import netCDF4

    path, var, cell, level, pins, wanted = job
    try:
        with netCDF4.Dataset(path) as nc:
            return _point_in(nc, path, var, cell, level, pins, wanted)
    except OSError as exc:
        # One unreadable file in a run of three thousand -- a write cut short,
        # a truncated copy -- must cost its own steps and no more. The map
        # already draws every other step of such a run; a series that threw
        # the whole answer away for one bad file was the odd one out.
        return [(index, float("nan")) for index, _ in wanted] + [("error", str(exc))]


def _point_in(nc, name: str, var: str, cell: int, level: int, pins: dict,
              wanted) -> list:
    """(position, value) for each wanted step of one already-open file.

    A masked point comes back NaN, never 0.0. netCDF4 masks fill values for
    us and `np.asarray` of a masked constant hands back the raw 0.0 sitting
    under the mask -- which is a plausible temperature anomaly, a plausible
    flux and a plausible wind component, so it would be believed. Everything
    else in this package decodes a fill to NaN; so does this.
    """
    if var not in nc.variables:
        raise KeyError(f"{var!r} not in {name}")
    v = nc.variables[var]
    t_axis = _nc_time_axis(v)
    picks = _cell_index(v, cell, level, pins, name, t_axis=t_axis)
    out = []
    for index, local in wanted:
        take = tuple(local if d == t_axis else picks[d] for d in v.dimensions)
        raw = np.ma.asarray(v[take]).astype(np.float64)
        out.append((index, float(np.ma.filled(raw, np.nan).reshape(-1)[0])))
    return out


def _cell_index(v, cell: int, level: int, pins: dict, name: str = "",
                t_axis: str | None = None) -> dict:
    """Which index each of a variable's dimensions takes for one cell.

    The mesh dimension takes `cell`; a stacking axis takes `level`, or
    whatever `sel` pins it to. Named by dimension rather than by position
    because a diagnostic writes its levels wherever it likes -- the
    `nIsoLevels` convention this package already follows elsewhere.

    `t_axis` names the variable's time dimension, from `_nc_time_axis`;
    passing it here rather than re-detecting per call keeps the pick and
    the hyperslab in `_point_in` reading the same axis. None means the
    variable has none.

    Every index is checked here rather than left to netCDF, whose own
    complaint is "index exceeds dimension bounds" and names neither the
    variable, the dimension, the size nor the file.
    """
    where = f" in {name}" if name else ""
    sizes = dict(zip(v.dimensions, v.shape, strict=True))
    unknown = [d for d in pins if d not in sizes]
    if unknown:
        raise KeyError(f"{unknown} is not a dimension of {v.name!r}"
                       f"{where}; it has {list(v.dimensions)}")
    picks = {}
    stack = [d for d in v.dimensions
             if d != t_axis and d not in SPATIAL_DIMS]
    for dim in v.dimensions:
        if dim == t_axis:
            continue
        if dim in SPATIAL_DIMS:
            idx = cell
        elif dim in pins:
            idx = pins[dim]
        else:
            # the slider drives the first stacking axis; the rest sit at 0,
            # which is what `Viewer._pins` shows the page
            idx = level if dim == stack[0] else 0
        if not 0 <= idx < sizes[dim]:
            what = "cell" if dim in SPATIAL_DIMS else dim
            raise IndexError(f"{what} {idx} is outside {dim}={sizes[dim]} "
                             f"for {v.name!r}{where}")
        picks[dim] = idx
    return picks


def _nc_time_axis(v) -> str | None:
    """Which of a raw netCDF4 variable's dimensions is its time axis, or None.

    The raw read paths (`_point_in`, the scan) see undecoded netCDF4, where
    attributes are read straight off the variables. Same evidence as
    `data.is_time_dim`, minus what needs decoding (a datetime dtype): name,
    CF attributes, and a `<unit> since <date>` units string. The exact
    spelling `Time` is MPAS's own convention and is accepted as it always
    was; anything else must be lower-case in `data.TIME_NAMES`.

    The variable's own group's variables are consulted, so a dimension
    carrying a coordinate can be judged by its attributes and not just its
    name.
    """
    def judge(d: str, coord) -> bool:
        if d == "Time":
            return True
        if coord is None:
            return d in TIME_NAMES
        attrs = {a: getattr(coord, a) for a in coord.ncattrs()
                 if not a.startswith("_")}
        if str(attrs.get("standard_name", "")).lower() == "time":
            return True
        if str(attrs.get("axis", "")).upper() == "T":
            return True
        if str(attrs.get("_CoordinateAxisType", "")).lower() == "time":
            return True
        if " since " in str(attrs.get("units", "")):
            return True
        return d in TIME_NAMES

    lookup = getattr(v, "group", None)
    variables = lookup().variables if callable(lookup) else {}
    for d in v.dimensions:
        if judge(d, variables.get(d)):
            return d
    return None


def _count_steps(path: str) -> tuple[str, int]:
    """How many timesteps one file holds. Runs in a worker process.

    The whole of what the scan needs from a file, and a pure function of its
    path -- nothing to keep consistent between workers, which is what makes
    the scan an easier thing to parallelise than a point series.
    """
    import netCDF4

    try:
        with netCDF4.Dataset(path) as nc:
            # The time dimension is whatever this file names it -- `Time`
            # from the model core, or `time`/`valid_time` from ncrcat/CDO.
            # Judge it the way `_point_in` judges it, so the count and the
            # reads agree on which axis the steps walk.
            t_axis = _count_time_axis(nc)
            dim = nc.dimensions.get(t_axis) if t_axis else None
            return path, len(dim) if dim is not None else 1
    except Exception:
        return path, 1                    # unreadable: leave it as one step


def _count_time_axis(nc) -> str | None:
    """A raw file's time dimension, judged like `_nc_time_axis`.

    One dimension length has to be read here -- cheap as it is -- so the
    evidence is kept to what the header carries: names and attributes, no
    coordinate values.
    """
    # An exact `Time` dimension is the model core's own and wins outright.
    if "Time" in nc.dimensions:
        return "Time"
    for d, _ in nc.dimensions.items():
        if d not in TIME_NAMES:
            continue
        coord = nc.variables.get(d)
        if coord is None:
            return d
        attrs = {a: getattr(coord, a) for a in coord.ncattrs()
                 if not a.startswith("_")}
        if (str(attrs.get("standard_name", "")).lower() == "time"
                or str(attrs.get("axis", "")).upper() == "T"
                or str(attrs.get("_CoordinateAxisType", "")).lower() == "time"
                or " since " in str(attrs.get("units", ""))):
            return d
        return d                        # name says time, no contrary evidence
    return None
