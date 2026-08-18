"""A plain, self-describing netCDF file on its own regular lat/lon grid.

`gmpas view --generic` is for output that already lives on a regular
lat/lon grid -- reanalysis, satellite products, anything CF-conventional --
as opposed to `gmpas view`'s MPAS path, which samples an *unstructured*
mesh onto pixels via a KD-tree (`viewer.ViewIndex`, ~200 ms per view box).

A regular grid needs none of that: the data already sits on the grid, so
the "view box" is just an index range into arrays that are already there.
No resampling either -- the returned slice is at the file's own native
resolution, not regridded onto some other pixel count.

`GenericViewer` implements the same public surface `viewer.Viewer` does
(`describe`, `frame`, `overlay`, `probe`, `figure`, `gif`, `netcdf`), so
`viewer._handler()` and the existing browser page serve it completely
unmodified -- this file is the only new code, nothing in viewer.py or
dashboard.py changes.
"""

from __future__ import annotations

import io
import threading
from pathlib import Path

import numpy as np

from . import data as _data
from .viewer import CMAPS, _overlay, _png, ramp

_LAT_NAMES = ("lat", "latitude")
_LON_NAMES = ("lon", "longitude")


def _find_coord(ds, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in ds.variables and ds[name].dims == (name,):
            return name
    return None


def _find_latlon(ds) -> tuple[str, str]:
    """A 1D dimension coordinate for each axis, or a clear error.

    Deliberately narrow: only the CF-conventional plain names, and only if
    each is a true 1D dimension coordinate (`lat(lat)`, not `lat(y, x)`).
    A curvilinear or unstructured file fails here, at startup, rather than
    with a confusing shape error mid-request.
    """
    lat, lon = _find_coord(ds, _LAT_NAMES), _find_coord(ds, _LON_NAMES)
    if lat is None or lon is None:
        raise ValueError(
            f"no 1D lat/lon dimension coordinate found (looked for "
            f"{_LAT_NAMES} and {_LON_NAMES}). --generic needs a plain "
            f"regular lat/lon grid; a curvilinear (2D lat/lon) or "
            f"unstructured file needs a real mesh instead (drop --generic)."
        )
    return lat, lon


def _find_time(ds, lat: str, lon: str) -> str | None:
    for name, da in ds.data_vars.items():
        for dim in da.dims:
            if dim not in (lat, lon) and dim in ds.variables \
                    and str(ds[dim].dtype).startswith(("datetime", "<M8")):
                return dim
    return None


def _nearest_idx(coords: np.ndarray, query: float) -> int:
    """Index of the ascending 1D `coords` entry closest to `query`.

    `searchsorted` alone gives the insertion point -- the first entry >=
    query -- which always rounds toward the higher neighbour, not whichever
    is actually closer; this compares both.
    """
    idx = int(np.clip(np.searchsorted(coords, query), 1, coords.size - 1))
    left, right = coords[idx - 1], coords[idx]
    return idx - 1 if (query - left) <= (right - query) else idx


class _SeriesShim:
    """Stand-in for the `viewer.series.*` accesses `_handler()` makes
    directly (export filenames, /api/status) -- GenericViewer has no real
    Series, and doesn't need one: there's exactly one file, already open."""

    def __init__(self, labels: list[str]):
        self.labels = labels
        self.scanning = False

    def __len__(self) -> int:
        return len(self.labels)


class GenericViewer:
    """Same public surface as `viewer.Viewer`, backed by one plain xarray
    Dataset on a regular grid instead of an MPAS mesh + Series."""

    def __init__(self, path):
        import xarray as xr

        self.path = Path(path)
        self.ds = xr.open_dataset(self.path, decode_timedelta=False, engine="netcdf4")
        self.lat_name, self.lon_name = _find_latlon(self.ds)
        self.time_name = _find_time(self.ds, self.lat_name, self.lon_name)

        lat = np.asarray(self.ds[self.lat_name].values, dtype=np.float64)
        self.lon = np.asarray(self.ds[self.lon_name].values, dtype=np.float64)
        # canonical ascending order, so index-range slicing in frame()/probe()
        # doesn't have to special-case a north-to-south file
        self._lat_flip = lat.size > 1 and lat[0] > lat[-1]
        self.lat = lat[::-1] if self._lat_flip else lat

        # NOT --width/--height: the browser stretches whatever frame() returns
        # to fill a box whose aspect is nx/ny (see the frontend's #wrap img,
        # no object-fit), and every extent it ever requests has lon-span /
        # lat-span == nx/ny by construction (boxOf()). Since frame() returns
        # a slice at native resolution rather than resampling to nx/ny, the
        # only way that stays undistorted is if nx/ny *is* the grid's own
        # index-count aspect -- which equals its degree aspect exactly when
        # the grid's lon/lat spacing is uniform (the common case; a grid
        # with meaningfully different dlon/dlat will show mild stretch,
        # a deliberate v1 tradeoff for not resampling at all).
        self.nx = self.lon.size
        self.ny = self.lat.size
        self.steps = int(self.ds.sizes[self.time_name]) if self.time_name else 1
        self.labels = ([str(v) for v in self.ds[self.time_name].values]
                        if self.time_name else ["static"])
        self.home = (float(self.lon.min()), float(self.lon.max()),
                     float(self.lat.min()), float(self.lat.max()))
        self.series = _SeriesShim(self.labels)

        # netCDF4/HDF5 is not safe for concurrent reads from multiple threads
        # (see the identical note in series.py) -- this is one Dataset shared
        # across every request thread, so every materialising read is
        # serialized the same way Series's are.
        self._lock = threading.Lock()

    # -- variables -------------------------------------------------------

    def _spatial_vars(self) -> list[str]:
        return [name for name, da in self.ds.data_vars.items()
                if self.lat_name in da.dims and self.lon_name in da.dims]

    def describe(self) -> dict:
        variables = []
        for name, da in self.ds.data_vars.items():
            spatial = self.lat_name in da.dims and self.lon_name in da.dims
            variables.append({
                "name": name,
                "label": _data.field_label(da),
                "static": not (self.time_name and self.time_name in da.dims),
                # vertical levels aren't handled yet -- always level 0
                "levels": 1,
                "spatial": spatial,   # extra field; the existing frontend ignores it
            })
        variables.sort(key=lambda v: (not v["spatial"], v["static"], v["name"]))
        return {
            "file": self.path.name,
            "mesh": self.path.name,        # not read by the frontend; kept for the contract
            "cells": int(self.nx * self.ny),
            "regional": not (self.home[1] - self.home[0] >= 359
                              and self.home[3] - self.home[2] >= 179),
            "coverage": 100.0,
            "files": 1,
            "steps": self.steps,
            "labels": self.labels,
            "scanning": False,
            "home": list(self.home),
            "nx": self.nx,
            "ny": self.ny,
            "cmaps": CMAPS,
            "ramps": {name: ramp(name) for name in CMAPS},
            "variables": variables,
        }

    # -- frames ----------------------------------------------------------

    def _slice(self, var: str, time: int, extent) -> np.ndarray:
        """The view box, sliced directly out of the native array -- no
        resampling, no KD-tree: a regular grid already *is* the raster."""
        with self._lock:
            da = self.ds[var]
            if self.time_name and self.time_name in da.dims:
                da = da.isel({self.time_name: time})
            arr = np.asarray(da.transpose(self.lat_name, self.lon_name).values,
                             dtype=np.float64)
        if self._lat_flip:
            arr = arr[::-1, :]

        lon_min, lon_max, lat_min, lat_max = extent
        # side="right" on the upper bound: "left" (the default) would exclude
        # the grid's own last row/column whenever the bound exactly matches
        # it, which the "home" extent -- built from self.lon/lat.min()/max()
        # -- always does. Silently dropping the domain's own edge every time
        # is worse than the asymmetry of two different sides here.
        j0 = np.searchsorted(self.lon, lon_min)
        j1 = np.searchsorted(self.lon, lon_max, side="right")
        i0 = np.searchsorted(self.lat, lat_min)
        i1 = np.searchsorted(self.lat, lat_max, side="right")
        j0, j1 = int(np.clip(j0, 0, self.lon.size)), int(np.clip(j1, 0, self.lon.size))
        i0, i1 = int(np.clip(i0, 0, self.lat.size)), int(np.clip(i1, 0, self.lat.size))
        img = arr[i0:i1, j0:j1]
        return img if img.size else np.full((1, 1), np.nan)

    def _line_png(self, var: str, nx: int, ny: int) -> bytes:
        """A non-spatial variable (no lat/lon dims) plotted as a plain
        matplotlib line -- there's no raster to slice, so just draw it."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with self._lock:
            da = self.ds[var]
            x = da[self.time_name].values if self.time_name in da.dims else None
            y = np.asarray(da.values).ravel()

        fig, ax = plt.subplots(figsize=(max(nx, 200) / 100, max(ny, 150) / 100), dpi=100)
        if x is not None:
            ax.plot(x, y)
            ax.set_xlabel(self.time_name)
        else:
            ax.plot(y)
        ax.set_ylabel(_data.field_label(da))
        ax.set_title(var, fontsize=10)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        return buf.getvalue()

    def frame(self, var, time, level, extent, cmap, vmin, vmax,
              nx=None, ny=None, compress=1):
        nx, ny = nx or self.nx, ny or self.ny
        if var not in self._spatial_vars():
            # no colour range for a line plot; 0..1 is an unused placeholder
            return self._line_png(var, nx, ny), 0.0, 1.0

        img = self._slice(var, time, extent)
        if vmin is not None and vmax is not None:
            lo, hi = vmin, vmax
        else:
            finite = img[np.isfinite(img)]
            lo = vmin if vmin is not None else (
                float(np.percentile(finite, 2)) if finite.size else 0.0)
            hi = vmax if vmax is not None else (
                float(np.percentile(finite, 98)) if finite.size else 1.0)
        if hi <= lo:
            hi = lo + 1.0
        return _png(img, cmap, lo, hi, compress), lo, hi

    def overlay(self, extent, nx=None, ny=None) -> bytes:
        return _overlay(extent, nx or self.nx, ny or self.ny)

    def probe(self, lon, lat, var, time, level):
        j = _nearest_idx(self.lon, lon)
        i = _nearest_idx(self.lat, lat)
        img = self._slice(var, time, self.home)
        value = float(img[i, j]) if img.shape == (self.lat.size, self.lon.size) else float("nan")
        return {"cell": i * self.lon.size + j,
                "lon": round(float(self.lon[j]), 4),
                "lat": round(float(self.lat[i]), 4),
                "value": value}

    # -- export ------------------------------------------------------------
    # Not implemented yet: figure()/netcdf() pull real-field attrs and a
    # publication layout that the MPAS path already has via `plot.cell_field`;
    # wiring that up for a plain grid is follow-up work, not this pass.

    def figure(self, *a, **k):
        raise NotImplementedError("figure export isn't implemented yet for --generic")

    def gif(self, *a, **k):
        raise NotImplementedError("GIF export isn't implemented yet for --generic")

    def netcdf(self, *a, **k):
        raise NotImplementedError("netCDF export isn't implemented yet for --generic")
