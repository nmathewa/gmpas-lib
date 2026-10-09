"""Diagnostics computed on the native MPAS mesh, without regridding.

A variable-resolution mesh has no columns: a 1-degree box holds a thousand
3 km cells in the refined region and a handful of 30 km cells outside it.
So every average here is weighted by cell area (`areaCell`) -- a plain mean
would let the many small cells outvote the few large ones -- and a cell
belongs to a bin by where its centre lies, never split. Bins with no cell are
NaN; nothing is interpolated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import jobs as _jobs
from . import timing

#: km per degree of latitude on the 6371.229 km MPAS sphere
KM_PER_DEG = 2 * math.pi * 6371.229 / 360.0


@dataclass(frozen=True)
class HovPlan:
    """Which cells a Hovmöller reads, and how they fold into longitude bins."""
    cells: np.ndarray        # band cells, ascending index (read order)
    weights: object          # scipy.sparse (bins x cells), cell area per membership
    lon_bounds: np.ndarray   # bin boundaries, degrees east, increasing
    counts: np.ndarray       # cells per bin
    band: tuple[float, float]
    width: float             # bin width, degrees

    @property
    def lon(self) -> np.ndarray:
        """Bin centres."""
        return 0.5 * (self.lon_bounds[:-1] + self.lon_bounds[1:])


def _covering_arc(lon: np.ndarray) -> tuple[float, float]:
    """The shortest longitude range holding every point, in degrees east.

    The arc starts just past the widest gap between neighbouring points, so a
    regional mesh across the antimeridian is one range, not the whole circle.
    """
    s = np.sort(np.mod(lon, 360.0))
    gaps = np.diff(np.r_[s, s[0] + 360.0])
    i = int(np.argmax(gaps))
    start = s[(i + 1) % s.size]
    span = 360.0 - float(gaps[i])
    return float(start), float(start) + span


def hovmoller_plan(mesh, band, lons=None, width: float | None = None) -> HovPlan:
    """Resolve a Hovmöller request against the mesh, or refuse it by name.

    `band` is (lat0, lat1); a cell belongs if its centre lies in it, inclusive.
    `lons` is (lon0, lon1) in degrees east, lon1 > lon0, at most 360 wide; by
    default the shortest range covering the band's cells. `width` is the bin
    width in degrees; by default the widest cell in the band, so no bin is
    empty for lack of resolution and none is narrower than the data.
    """
    import scipy.sparse as sp

    lat0, lat1 = (float(v) for v in band)
    if not lat1 >= lat0:
        raise ValueError(f"latitude band {lat0:g}..{lat1:g} must not decrease")
    lat = np.asarray(mesh.lat_cell, dtype=np.float64)
    cells = np.nonzero((lat >= lat0) & (lat <= lat1))[0]
    if not cells.size:
        raise ValueError(f"no cell centre between latitude {lat0:g} and {lat1:g}")
    lon = np.asarray(mesh.lon_cell, dtype=np.float64)[cells]

    if lons is None:
        lon0, lon1 = _covering_arc(lon)
    else:
        lon0, lon1 = (float(v) for v in lons)
        if not 0.0 < lon1 - lon0 <= 360.0:
            raise ValueError(f"longitude range {lon0:g}..{lon1:g} must increase "
                             f"by at most 360 degrees")
    # every centre into [lon0, lon0 + 360): one frame, whatever the mesh used
    lon_u = lon0 + np.mod(lon - lon0, 360.0)
    keep = lon_u <= lon1
    cells, lon_u = cells[keep], lon_u[keep]
    if not cells.size:
        raise ValueError(f"no cell centre in longitude {lon0:g}..{lon1:g} "
                         f"inside latitude {lat0:g}..{lat1:g}")

    if width is None:
        # a cell's east-west extent in degrees grows toward the poles
        km = np.asarray(mesh.cell_width_km, dtype=np.float64)[cells]
        coslat = np.maximum(np.cos(np.radians(lat[cells])), 1e-6)
        width = float(np.max(km / (KM_PER_DEG * coslat)))
    width = float(width)
    if not width > 0:
        raise ValueError(f"bin width {width:g} must be positive")
    n = max(1, math.ceil((lon1 - lon0) / width - 1e-9))
    lon_bounds = lon0 + width * np.arange(n + 1)
    b = np.minimum(((lon_u - lon0) // width).astype(np.int64), n - 1)

    area = np.asarray(mesh.area_cell, dtype=np.float64)[cells]
    order = np.argsort(cells)               # read order: ascending cell index
    cells, b, area = cells[order], b[order], area[order]
    weights = sp.csr_matrix((area, (b, np.arange(cells.size))), shape=(n, cells.size))
    return HovPlan(cells=cells, weights=weights, lon_bounds=lon_bounds,
                   counts=np.bincount(b, minlength=n), band=(lat0, lat1), width=width)


def hovmoller_row(values: np.ndarray, plan: HovPlan) -> np.ndarray:
    """One step: the area-weighted mean of the band's cells in each bin.

    `values` are the band's cells in `plan.cells` order. NaN cells are skipped
    and the weights renormalised over the valid ones; a bin with none is NaN.
    """
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v)
    num = plan.weights @ np.where(ok, v, 0.0)
    den = plan.weights @ ok.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def hovmoller(series, var: str, level: int = 0, band=(-15.0, 15.0), lons=None,
              steps=None, width: float | None = None, sel=None,
              progress=None, cancel=None):
    """`var` averaged over a latitude band on the native mesh, as (time, lon).

    Reads one step at a time through `Series.values` (cached, the same read
    the map uses) and keeps only the band's cells. `progress(done, total)` is
    called after each step; `cancel`, an Event, stops at the next one.
    """
    import xarray as xr

    plan = hovmoller_plan(series.mesh, band, lons, width)
    n = len(series)
    a, b = (0, n - 1) if steps is None else (int(steps[0]), int(steps[1]))
    if not 0 <= a <= b < n:
        raise ValueError(f"steps {a}..{b} are outside 0..{n - 1}")
    total = b - a + 1
    out = np.full((total, plan.lon.size), np.nan)

    with timing.step("native.hovmoller", steps=total, cells=int(plan.cells.size),
                     bins=int(plan.lon.size)):
        for k, step in enumerate(range(a, b + 1)):
            if cancel is not None and cancel.is_set():
                raise _jobs.Cancelled()
            field = series.values(var, step=step, level=level, sel=sel)
            out[k] = hovmoller_row(np.asarray(field)[plan.cells], plan)
            if progress is not None:
                progress(k + 1, total)

    times = series.times[a:b + 1] if series.dated else None
    if times is not None and all(t is not None for t in times):
        tname, tvals = "time", np.array(times, dtype="datetime64[s]")
    else:
        tname, tvals = "step", np.arange(a, b + 1)
    attrs = {k: v for k, v in series.dataarray(var, a).attrs.items()
             if k in ("units", "long_name", "standard_name")}
    lat0, lat1 = plan.band
    return xr.DataArray(
        out, dims=(tname, "lon"),
        coords={tname: tvals, "lon": plan.lon, "cells": ("lon", plan.counts)},
        name=var,
        attrs={**attrs,
               "band": f"latitude {lat0:g} to {lat1:g} ({plan.cells.size} cells)",
               "longitude_range": f"{plan.lon_bounds[0]:g} to {plan.lon_bounds[-1]:g}",
               "bin_width": f"{plan.width:.4g} degrees",
               "weighting": "cell area (areaCell), NaN skipped; a cell belongs "
                            "to the bin its centre lies in",
               "empty_bins": int((plan.counts == 0).sum()),
               "level": int(level)})


# -- vertical cross-section along a great circle ---------------------------

#: most samples one section may take; at 1/4 of the smallest cell this is a
#: ~20,000 km path at 3.75 km, so a request past it is not a real section
SECTION_MAX_SAMPLES = 2_000_000


@dataclass(frozen=True)
class Section:
    """The cells a great-circle path crosses, in order, and where."""
    cells: np.ndarray        # cell index per segment; -1 where the path leaves the mesh
    x0: np.ndarray           # km along the path where each segment starts
    x1: np.ndarray           # ... and ends
    lon: np.ndarray          # segment midpoint, degrees east
    lat: np.ndarray
    length: float            # km, the whole path


def _unit(lon, lat) -> np.ndarray:
    lo, la = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def _great_circle(p0, p1, n: int):
    """`n` evenly spaced points from p0 to p1 along the great circle, as unit
    vectors, with their angular distance from p0 in radians."""
    a, b = _unit(*p0), _unit(*p1)
    omega = float(np.arccos(np.clip(a @ b, -1.0, 1.0)))
    t = np.linspace(0.0, 1.0, n)
    if omega < 1e-12:
        return np.repeat(a[None], n, 0), np.zeros(n), 0.0
    if abs(omega - math.pi) < 1e-9:
        raise ValueError("the two points are antipodal: no single great circle joins them")
    s = math.sin(omega)
    pts = (np.sin((1 - t) * omega)[:, None] * a + np.sin(t * omega)[:, None] * b) / s
    return pts, t * omega, omega


def _locate(mesh, pts) -> np.ndarray:
    """Cell containing each unit-vector point, -1 off the mesh.

    On a Voronoi mesh the nearest centre is the containing cell, exactly. Past
    the edge of a regional mesh the nearest centre is a boundary cell, so a
    point more than two cell widths from it is off the mesh; a global mesh
    has no off.
    """
    dist, idx = mesh.tree().query(pts, workers=-1)
    if mesh.is_global:
        return idx
    km = dist * mesh.sphere_radius / 1000.0          # chord, ~ arc at cell scale
    width = np.asarray(mesh.cell_width_km, dtype=np.float64)[idx]
    return np.where(km <= 2.0 * width, idx, -1)


def _point_at(p0, p1, omega: float, t: np.ndarray) -> np.ndarray:
    a, b = _unit(*p0), _unit(*p1)
    s = math.sin(omega)
    return (np.sin((1 - t) * omega)[:, None] * a + np.sin(t * omega)[:, None] * b) / s


def _refine(mesh, p0, p1, omega, t, ids, eps: float):
    """Bisect every change between samples down to `eps` (fraction of the path),
    all at once: finds where each boundary really is, and any cell a sample
    stepped over -- a path clipping a cell's corner for less than the spacing.

    Returns the cell sequence and the boundary positions between them.
    """
    i = np.flatnonzero(ids[1:] != ids[:-1])
    tl, tr, cl, cr = t[i], t[i + 1], ids[i], ids[i + 1]
    while True:
        open_ = (tr - tl) > eps
        if not open_.any():
            break
        tm = 0.5 * (tl + tr)
        cm = np.where(open_, _locate(mesh, _point_at(p0, p1, omega, tm)), cl)
        left, right = cm == cl, cm == cr
        new = open_ & ~left & ~right              # a third cell between: split
        tl = np.where(open_ & left, tm, tl)
        tr = np.where(open_ & right, tm, tr)
        if new.any():
            tl, tr, cl, cr = (np.r_[np.where(new, tl, tl), tm[new]],
                              np.r_[np.where(new, tm, tr), tr[new]],
                              np.r_[cl, cm[new]],
                              np.r_[np.where(new, cm, cr), cr[new]])
            order = np.argsort(tl, kind="stable")
            tl, tr, cl, cr = tl[order], tr[order], cl[order], cr[order]
    seq = np.r_[ids[0], cr]
    return seq, 0.5 * (tl + tr)


def section_cells(mesh, p0, p1) -> Section:
    """The ordered cells a great-circle path from p0 to p1 crosses.

    A coarse pass finds the smallest cell along the path; the path is then
    sampled at a quarter of that width, each sample mapped to its cell, and
    consecutive repeats merged. A geodesic crosses a convex cell once, so no
    cell is skipped or repeated; a segment's ends sit halfway between the
    last sample in one cell and the first in the next.
    """
    radius_km = mesh.sphere_radius / 1000.0
    _, _, omega = _great_circle(p0, p1, 2)
    length = omega * radius_km
    if length <= 0:
        raise ValueError("the two section points are the same place")
    coarse_pts, _, _ = _great_circle(p0, p1, 2048)
    coarse = _locate(mesh, coarse_pts)
    inside = coarse[coarse >= 0]
    if not inside.size:
        raise ValueError("the section path does not cross the mesh")
    smallest = float(np.min(np.asarray(mesh.cell_width_km)[np.unique(inside)]))
    n = int(math.ceil(length / (smallest / 4.0))) + 1
    if n > SECTION_MAX_SAMPLES:
        raise ValueError(f"a {length:,.0f} km section at {smallest:.2f} km cells needs "
                         f"{n:,} samples, over the {SECTION_MAX_SAMPLES:,} allowed; "
                         f"shorten the path")
    pts, ang, _ = _great_circle(p0, p1, max(n, 2))
    ids = _locate(mesh, pts)
    # boundaries to ~1 m: exact for any purpose a plot has
    cells, at = _refine(mesh, p0, p1, omega, ang / omega, ids, eps=1e-3 / length)
    edges = np.r_[0.0, at * length, length]
    x0, x1 = edges[:-1], edges[1:]
    # segment midpoints back to lon/lat
    frac = (0.5 * (x0 + x1)) / length
    a, b = _unit(*p0), _unit(*p1)
    s = math.sin(omega)
    m = (np.sin((1 - frac) * omega)[:, None] * a + np.sin(frac * omega)[:, None] * b) / s
    lat = np.degrees(np.arcsin(np.clip(m[:, 2], -1, 1)))
    lon = np.degrees(np.arctan2(m[:, 1], m[:, 0]))
    return Section(cells=cells, x0=x0, x1=x1, lon=lon, lat=lat, length=length)


def _vertical_axis(series, var: str, step: int, levdim: str, n: int, cells):
    """The vertical coordinate of a section, as the file states it.

    Returns (name, units, centres (n,), bounds (n+1,) or (n+1, ncells), down)
    where `down` means the axis increases downward (pressure).
    """
    with series._lock:
        src = series._dataset(series.steps[step][0])
        # a 1-D variable on the level dimension, e.g. t_iso_levels on nIsoLevelsT
        for name, v in src.variables.items():
            if v.dims == (levdim,) and name != levdim and np.issubdtype(v.dtype, np.number):
                units = str(v.attrs.get("units", ""))
                values = np.asarray(v.values, dtype=np.float64)
                if units.lower() in ("pa", "pascal", "pascals"):
                    values, units = values / 100.0, "hPa"
                bounds = _bounds(values)
                down = units.lower() in ("hpa", "mb", "mbar", "millibar")
                return name, units, values, bounds, down
        if levdim == "nVertLevels" and "zgrid" in src.variables \
                and src["zgrid"].dims[-1] == "nVertLevelsP1":
            z = src["zgrid"]
            z = z.isel({d: 0 for d in z.dims if d not in ("nCells", "nVertLevelsP1")})
            zc = np.asarray(z.transpose("nVertLevelsP1", "nCells").values[:, cells],
                            dtype=np.float64)                      # (n+1, ncells)
            centres = 0.5 * (zc[:-1] + zc[1:]).mean(axis=1)
            return "height", "m", centres, zc, False
    idx = np.arange(n, dtype=np.float64)
    name = "model level" if levdim.startswith("nVertLevels") else f"{levdim} index"
    return name, "index", idx, _bounds(idx), False


def _bounds(c: np.ndarray) -> np.ndarray:
    """Cell boundaries for centres `c`: midway between, half a step past the ends."""
    c = np.asarray(c, dtype=np.float64)
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    mid = 0.5 * (c[1:] + c[:-1])
    return np.r_[c[0] - (mid[0] - c[0]), mid, c[-1] + (c[-1] - mid[-1])]


def cross_section(series, var: str, step: int, p0, p1, sel=None):
    """`var` on the cells a great-circle path crosses, as (level, distance).

    Native in both directions: each column is one cell, as wide as the path
    inside it; each row is one level of the file. Nothing is interpolated.
    """
    import xarray as xr

    from . import data as _data

    da = series.dataarray(var, step)
    where = _data.spatial_dim(da)
    if where != "nCells":
        raise ValueError(f"{var!r} lives on {where}; a section takes cell fields only")
    levels = _data.level_dims(da)
    if not levels:
        raise ValueError(f"{var!r} has no vertical axis, so it has no section")
    levdim = levels[0]
    n = int(da.sizes[levdim])
    sec = section_cells(series.mesh, p0, p1)
    on = sec.cells >= 0
    block = np.full((n, sec.cells.size), np.nan)
    with timing.step("native.section", cells=int(on.sum()), levels=n):
        for k in range(n):
            field = np.asarray(series.values(var, step=step, level=k, sel=sel))
            block[k, on] = field[sec.cells[on]]
    name, units, centres, bounds, down = _vertical_axis(
        series, var, step, levdim, n, np.where(on, sec.cells, 0))
    if bounds.ndim == 2:                    # per-column heights: off-mesh columns blank
        bounds = np.where(on[None, :], bounds, np.nan)
        lo, hi = ((("level", "distance"), bounds[:-1]), (("level", "distance"), bounds[1:]))
    else:
        lo, hi = ("level", bounds[:-1]), ("level", bounds[1:])
    attrs = {k: v for k, v in da.attrs.items()
             if k in ("units", "long_name", "standard_name")}
    return xr.DataArray(
        block, dims=("level", "distance"),
        coords={"level": ("level", centres, {"long_name": name, "units": units}),
                "distance": ("distance", 0.5 * (sec.x0 + sec.x1), {"units": "km"}),
                "x0": ("distance", sec.x0), "x1": ("distance", sec.x1),
                "cell": ("distance", sec.cells),
                "lon": ("distance", sec.lon), "lat": ("distance", sec.lat),
                "level_lo": lo, "level_hi": hi},
        name=var,
        attrs={**attrs,
               "pressure_down": int(down),
               "path": f"({p0[0]:g}, {p0[1]:g}) to ({p1[0]:g}, {p1[1]:g}), "
                       f"{sec.length:,.0f} km great circle",
               "method": "native: one column per crossed cell, as wide as the path "
                         "inside it; one row per file level; nothing interpolated",
               "time_step": int(step)})


def draw_section(ax, da, cmap=None, vmin=None, vmax=None):
    """Each crossed cell as a column exactly as wide as the path inside it."""
    lo = np.asarray(da["level_lo"].values, dtype=np.float64)
    hi = np.asarray(da["level_hi"].values, dtype=np.float64)
    bounds = np.concatenate([lo[:1], hi], axis=0)       # (n+1,) or (n+1, columns)
    x0, x1 = np.asarray(da["x0"].values), np.asarray(da["x1"].values)
    v = np.asarray(da.values)                      # (level, column)
    nl, nc = v.shape
    # two x positions per column, so neighbouring columns share no corner and
    # per-column level bounds (zgrid) stay per column; the zero-width quad
    # between them is masked
    X = np.empty(2 * nc)
    X[0::2], X[1::2] = x0, x1
    if bounds.ndim == 1:
        Y = np.repeat(bounds[:, None], 2 * nc, axis=1)
    else:
        Y = np.repeat(bounds, 2, axis=1)
    Xg = np.repeat(X[None, :], nl + 1, axis=0)
    C = np.full((nl, 2 * nc - 1), np.nan)
    C[:, 0::2] = v
    mesh = ax.pcolormesh(Xg, Y, np.ma.masked_invalid(C), cmap=cmap, vmin=vmin,
                         vmax=vmax, shading="flat")
    if int(da.attrs.get("pressure_down", 0)):
        ax.invert_yaxis()
    lvl = da["level"]
    ax.set_ylabel(f"{lvl.attrs.get('long_name', 'level')} [{lvl.attrs.get('units', '')}]")
    ax.set_xlabel("distance along the path [km]")
    return mesh
