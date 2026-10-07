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
