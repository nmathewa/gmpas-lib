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

# How each axis is recognised, strongest evidence first. These are the CF
# conventions' own markers, the same ones cf_xarray keys on -- `standard_name`,
# `units`, the `axis` attribute, Unidata's `_CoordinateAxisType` -- with the
# variable's name as the weakest tiebreak rather than the only test. A name
# list alone misses `nav_lat` or a `lat` spelled some other way, and trusts a
# `lat` that is actually metres.

#: normalised by `_norm_units`: lower-case, no spaces or underscores
_LAT_UNITS = {"degreesnorth", "degreenorth", "degreesn", "degreen"}
_LON_UNITS = {"degreeseast", "degreeeast", "degreese", "degreee"}
_PRESSURE_UNITS = {"pa", "hpa", "kpa", "mbar", "millibar", "millibars", "mb", "bar"}

_LAT_NAMES = {"lat", "latitude", "lats", "nav_lat", "xlat", "glat"}
_LON_NAMES = {"lon", "longitude", "lons", "long", "nav_lon", "xlong", "xlon", "glon"}
_TIME_NAMES = {"time", "t", "times", "valid_time", "xtime"}
_VERTICAL_NAMES = {"plev", "lev", "level", "levels", "isobaric", "pressure",
                   "height", "depth", "altitude", "z", "nisolevels"}
_VERTICAL_STANDARD_NAMES = {"air_pressure", "altitude", "height", "depth",
                            "model_level_number", "geopotential_height"}

#: axes that are regular in *their own* space but not on a lat/lon map:
#: rotated-pole and projected grids. Drawing one as degrees is a wrong map, so
#: these are never taken as lat/lon, however they are otherwise marked.
_NOT_GEOGRAPHIC = {"grid_latitude", "grid_longitude",
                   "projection_x_coordinate", "projection_y_coordinate"}


def _norm_units(var) -> str:
    return str(var.attrs.get("units", "")).lower().replace("_", "").replace(" ", "")


def _evidence(var, name: str, axis: str) -> int:
    """How strongly `var` claims to be the "lat" or "lon" axis; 0 is not at all.

    Scored rather than first-match so a file carrying both a named `lat` and a
    properly attributed `nav_lat` resolves to the attributed one.
    """
    std = str(var.attrs.get("standard_name", "")).lower()
    if std in _NOT_GEOGRAPHIC:
        return 0
    units = _norm_units(var)
    want = {"lat": ("latitude", _LAT_UNITS, "Y", "lat", _LAT_NAMES),
            "lon": ("longitude", _LON_UNITS, "X", "lon", _LON_NAMES)}[axis]
    std_name, unit_set, cf_axis, cat, names = want
    score = 0
    if std == std_name:
        score += 8
    if units in unit_set:
        score += 4
    if str(var.attrs.get("_CoordinateAxisType", "")).lower() == cat:
        score += 4
    # `axis` alone also marks projected metres; only count it when the units
    # don't say otherwise
    if str(var.attrs.get("axis", "")).upper() == cf_axis and units in (
            "", "degrees", "degree", *unit_set):
        score += 2
    if name.lower() in names:
        score += 1
    return score


def _find_axis(ds, axis: str) -> tuple[str, str]:
    """(coordinate variable, the dimension it indexes) for "lat" or "lon".

    The two usually share a name -- `lat(lat)` -- but need not: `nav_lat(y)`
    is a 1D latitude over a dimension called `y`, and slicing has to use `y`.
    """
    scored = sorted(((_evidence(v, str(n), axis), str(n)) for n, v in ds.variables.items()),
                    reverse=True)
    best = [(sc, n) for sc, n in scored if sc > 0]
    if not best:
        raise ValueError(
            f"no {axis} coordinate found. --generic looks for CF "
            f"standard_name/units/axis attributes, or a name like "
            f"{sorted(_LAT_NAMES if axis == 'lat' else _LON_NAMES)}. A projected "
            f"or unstructured file needs a real mesh instead (drop --generic)."
        )
    one_d = [n for _, n in best if ds[n].ndim == 1]
    if not one_d:
        n = best[0][1]
        raise ValueError(
            f"{n!r} is the {axis} coordinate but has dims {ds[n].dims}: this is a "
            f"curvilinear grid (2D lat/lon, e.g. WRF's XLAT/XLONG). --generic "
            f"needs a regular grid, where {axis} is 1D; regrid it first, or drop "
            f"--generic."
        )
    # among equally strong 1D candidates, prefer a true dimension coordinate
    top = max(_evidence(ds[n], n, axis) for n in one_d)
    ties = [n for n in one_d if _evidence(ds[n], n, axis) == top]
    name = next((n for n in ties if ds[n].dims == (n,)), ties[0])
    dim = str(ds[name].dims[0])

    values = np.asarray(ds[name].values, dtype=np.float64)
    step = np.diff(values)
    if values.size > 1 and not (np.all(step > 0) or np.all(step < 0)):
        raise ValueError(
            f"{name!r} is not monotonic, so it cannot be sliced as a regular "
            f"grid axis. --generic needs a sorted 1D {axis}."
        )
    if axis == "lat" and np.nanmax(np.abs(values)) > 90.001:
        raise ValueError(
            f"{name!r} looks like latitude but runs to {np.nanmax(np.abs(values)):g}; "
            f"degrees cannot exceed 90. Is it a projected coordinate in metres?"
        )
    return name, dim


def _is_time(ds, dim: str) -> bool:
    if dim not in ds.variables:                 # a bare dimension, no coordinate
        return dim.lower() in _TIME_NAMES
    var = ds[dim]
    return (np.issubdtype(var.dtype, np.datetime64)
            or var.dtype == object and dim.lower() in _TIME_NAMES     # cftime
            or str(var.attrs.get("standard_name", "")).lower() == "time"
            or str(var.attrs.get("axis", "")).upper() == "T"
            or str(var.attrs.get("_CoordinateAxisType", "")).lower() == "time"
            or " since " in str(var.attrs.get("units", ""))
            or dim.lower() in _TIME_NAMES)


def _is_vertical(ds, dim: str) -> bool:
    if dim not in ds.variables:
        return dim.lower() in _VERTICAL_NAMES
    var = ds[dim]
    return (str(var.attrs.get("positive", "")).lower() in ("up", "down")
            or str(var.attrs.get("axis", "")).upper() == "Z"
            or str(var.attrs.get("_CoordinateAxisType", "")).lower()
            in ("pressure", "height", "geopotentialheight")
            or str(var.attrs.get("standard_name", "")).lower() in _VERTICAL_STANDARD_NAMES
            or _norm_units(var) in _PRESSURE_UNITS
            or dim.lower() in _VERTICAL_NAMES)


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
        from . import netcdf
        with netcdf.LOCK:                 # see netcdf.LOCK: HDF5 is not thread-safe
            self.ds = xr.open_dataset(self.path, decode_timedelta=False,
                                      engine="netcdf4")
        self.lat_name, self.lat_dim = _find_axis(self.ds, "lat")
        self.lon_name, self.lon_dim = _find_axis(self.ds, "lon")
        if self.lat_dim == self.lon_dim:
            raise ValueError(
                f"{self.lat_name!r} and {self.lon_name!r} share the dimension "
                f"{self.lat_dim!r}: that is a list of points (a station file or an "
                f"unstructured mesh), not a grid. --generic needs lat and lon on "
                f"separate axes."
            )
        dims = {str(d) for da in self.ds.data_vars.values() for d in da.dims}
        dims -= {self.lat_dim, self.lon_dim}
        self.time_name = next((d for d in sorted(dims) if _is_time(self.ds, d)), None)
        self._vertical = {d for d in dims if _is_vertical(self.ds, d)}

        lat = np.asarray(self.ds[self.lat_name].values, dtype=np.float64)
        self.lon = np.asarray(self.ds[self.lon_name].values, dtype=np.float64)
        # canonical ascending order, so index-range slicing in frame()/probe()
        # doesn't have to special-case a north-to-south file
        self._lat_flip = lat.size > 1 and lat[0] > lat[-1]
        self.lat = lat[::-1] if self._lat_flip else lat
        # a descending longitude is rare but legal; the same treatment
        self._lon_flip = self.lon.size > 1 and self.lon[0] > self.lon[-1]
        if self._lon_flip:
            self.lon = self.lon[::-1]

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
        self.labels = self._time_labels()
        self.home = (float(self.lon.min()), float(self.lon.max()),
                     float(self.lat.min()), float(self.lat.max()))
        self.series = _SeriesShim(self.labels)

        # netCDF4/HDF5 is not safe for concurrent reads from multiple threads
        # (see the identical note in series.py) -- this is one Dataset shared
        # across every request thread, so every materialising read is
        # serialized the same way Series's are.
        self._lock = threading.Lock()

    # -- variables -------------------------------------------------------

    def _time_labels(self) -> list[str]:
        """Dates when the file decodes them; otherwise say what the step is.

        A time axis without CF units -- or with no coordinate variable at all
        -- is still a time axis to slice along, it just has nothing to print.
        """
        if not self.time_name:
            return ["static"]
        if self.time_name not in self.ds.variables:
            return [f"step {i}" for i in range(self.steps)]
        var = self.ds[self.time_name]
        units = str(var.attrs.get("units", ""))
        if np.issubdtype(var.dtype, np.datetime64) or var.dtype == object or not units:
            return [str(v) for v in var.values]
        return [f"{v} {units}" for v in var.values]

    def _spatial_vars(self) -> list[str]:
        return [name for name, da in self.ds.data_vars.items()
                if self.lat_dim in da.dims and self.lon_dim in da.dims]

    def _stack_dims(self, da) -> list[str]:
        """Every axis of `da` that is not lat, lon or time, vertical first.

        The browser has one level slider. A vertical axis is what it is for,
        so one is put first when the file marks it; anything else -- an
        ensemble member, a band -- follows and is held at 0.
        """
        rest = [str(d) for d in da.dims
                if d not in (self.lat_dim, self.lon_dim, self.time_name)]
        return sorted(rest, key=lambda d: d not in self._vertical)

    def describe(self) -> dict:
        variables = []
        for name, da in self.ds.data_vars.items():
            spatial = self.lat_dim in da.dims and self.lon_dim in da.dims
            stack = self._stack_dims(da)
            variables.append({
                "name": name,
                "label": _data.field_label(da),
                "static": not (self.time_name and self.time_name in da.dims),
                # the slider drives the first stacking axis, the rest are
                # held at 0 -- the same contract as viewer.Viewer.describe
                "levels": int(da.sizes[stack[0]]) if spatial and stack else 1,
                "dim": stack[0] if spatial and stack else "",
                "pinned": stack[1:] if spatial else [],
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

    def _slice(self, var: str, time: int, level: int, extent) -> np.ndarray:
        """The view box, sliced directly out of the native array -- no
        resampling, no KD-tree: a regular grid already *is* the raster.

        Every axis but lat and lon is indexed away first: time by `time`, the
        slider's axis by `level`, anything further at 0. Leaving one behind
        is what made every frame of a file with a pressure axis fail.
        """
        with self._lock:
            da = self.ds[var]
            stack = self._stack_dims(da)
            picks = {self.time_name: time} if self.time_name in da.dims else {}
            picks.update({d: (level if k == 0 else 0) for k, d in enumerate(stack)})
            for dim, idx in picks.items():
                n = da.sizes[dim]
                if not -n <= idx < n:
                    raise IndexError(f"{dim}={idx} out of range for {var!r} (size {n})")
            arr = np.asarray(da.isel(picks).transpose(self.lat_dim, self.lon_dim).values,
                             dtype=np.float64)
        if self._lat_flip:
            arr = arr[::-1, :]
        if self._lon_flip:
            arr = arr[:, ::-1]

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

    def _plain_png(self, var: str, nx: int, ny: int) -> bytes:
        """A variable with no lat/lon to map, drawn the way xarray would.

        `DataArray.plot()` picks the kind from the dimensions: one is a line
        against its coordinate, two a pcolormesh with both as axes, more a
        histogram of the values. That is the right default for whatever a
        gridded file keeps beside its maps -- a time series, a (time, plev)
        profile. The previous `ravel()` into a single line drew a 2D profile
        as a meaningless sawtooth.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with self._lock:
            da = self.ds[var].load()

        fig, ax = plt.subplots(figsize=(max(nx, 200) / 100, max(ny, 150) / 100), dpi=100)
        if da.ndim == 0:
            ax.text(0.5, 0.5, f"{float(da.values):g}", ha="center", va="center",
                    fontsize=16, transform=ax.transAxes)
            ax.set_axis_off()
        else:
            da.plot(ax=ax)
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
            return self._plain_png(var, nx, ny), 0.0, 1.0

        img = self._slice(var, time, level, extent)
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
        img = self._slice(var, time, level, self.home)
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
