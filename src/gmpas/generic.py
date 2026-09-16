"""Plain, self-describing netCDF files on one regular lat/lon grid.

`gmpas view --generic` is for output that already lives on a regular
lat/lon grid -- reanalysis, satellite products, anything CF-conventional --
as opposed to `gmpas view`'s MPAS path, which samples an *unstructured*
mesh onto pixels via a KD-tree (`viewer.ViewIndex`, ~200 ms per view box).

A regular grid needs none of that: every pixel's value is one index lookup
per axis, so a frame is a subset read plus a gather.

Like `gmpas view`, it takes a file, a list, a glob or a directory, and
presents them as one time axis (`series.expand` decides the order). And
beside the fast map it draws what `xarray.DataArray.plot` would -- a line,
a filled contour, a histogram -- with real axes, for the variables and the
views a map raster cannot show.

`GenericViewer` implements the same public surface `viewer.Viewer` does
(`describe`, `frame`, `overlay`, `probe`, `figure`, `gif`, `netcdf`), plus
`plot`, which `viewer._handler()` serves only to a viewer that has one.
"""

from __future__ import annotations

import io
import sys
import threading
import time as _time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from . import colour as _colour
from . import data as _data
from . import jobs as _jobs
from . import layers as _layers
from . import netcdf, palettes, timing
from .raster import target_grid
from .series import LRU_SIZE, expand, label_of
from .viewer import _overlay, _png

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


#: how the map is drawn, and the xarray.DataArray.plot kinds beside it
KIND_LABELS = {
    "map": "map (fast raster)",
    "auto": "auto (as xarray)",
    "pcolormesh": "pcolormesh",
    "contourf": "filled contour",
    "contour": "contour",
    "imshow": "imshow",
    "line": "line",
    "step": "step",
    "hist": "histogram",
    "series": "line: time series at point",
    "hovmoller": "Hovmöller (time × lon)",
    "profile": "line: profile at point",
    "layers": "layers (composite)",
}
_GRID_KINDS = ("pcolormesh", "contourf", "contour", "imshow")

#: What each plot kind actually uses, so the page can show, hide or disable a
#: control from one table instead of each control deciding for itself. Written
#: here rather than in the page because the answers come from what `plot`,
#: `frame`, `gif` and `netcdf` below really do with each kind.
#:
#: colour  the colormap picker and colour range
#: options the fast map's colour options (bands, extremes): its encoder only
#: pan     pan, zoom, reset view and the graticule: a map on a geographic axis
#: frames  cached palette frames, which is what the top bar's play button plays
#: probe   the clicked point, which the plot is taken at
#: gif     an animated export
#: data    netCDF export
#: data is False almost everywhere here: `netcdf` below exports the Hovmöller
#: only, where the numbers behind the picture are not in the input files
_KIND_MAP = {"colour": True, "options": True, "pan": True, "frames": True,
             "probe": True, "gif": True, "data": False}
_KIND_FIGURE = {**_KIND_MAP, "pan": False, "frames": False}
_KIND_LINE = {**_KIND_FIGURE, "colour": False, "options": False}
KIND_CAPS = {
    "map": _KIND_MAP,
    "auto": _KIND_FIGURE,
    **{k: _KIND_FIGURE for k in _GRID_KINDS},
    "line": _KIND_LINE,
    "step": _KIND_LINE,
    "hist": {**_KIND_LINE, "probe": False},
    "series": _KIND_LINE,
    "profile": _KIND_LINE,
    # its own panel drives it: colour comes from the picker and the range, and
    # the band options the encoder applies are not part of a matplotlib figure
    "hovmoller": {**_KIND_FIGURE, "options": False, "probe": False, "gif": False,
                  "data": True},
    # every layer carries its own colours, so the shared picker means nothing
    "layers": {**_KIND_FIGURE, "colour": False, "options": False, "probe": False},
}

#: a non-map variable is read whole to plot it; past this it is refused
PLOT_READ_BYTES = 256 * 1024 * 1024

#: the fast map's colour options and their checking live in `gmpas.colour`,
#: the one place both viewers get their colours from; these names stay because
#: the page handler duck-types on `clean_colour` and callers import them
COLOUR_OPTIONS = _colour.OPTIONS
clean_colour = _colour.clean

#: a figure-per-frame GIF renders each step through matplotlib (~0.3 s each)
GIF_FIGURE_FRAMES = 1000

#: one Hovmöller read (a chunk of steps of one file) decodes at most this much
HOV_READ_BYTES = 64 * 1024 * 1024

#: how a Hovmöller is drawn; "auto" picks by size, see GenericViewer._hov_method
HOV_METHODS = ("auto", "contourf", "pcolormesh", "contour")

#: Drawing limits, from rendering a year of hourly 0.25-degree columns (8760 x
#: 1440 = 12.6M cells). A smooth field: imshow 2.9 s / 0.8 GB, filled contour
#: 4.1 s / 0.3 GB, pcolormesh 11.9 s / 1.2 GB. A noisy one -- what hourly
#: precipitation is -- filled contour 28.7 s / 2.1 GB, contour lines 24.4 s /
#: 1.3 GB, imshow 4.8 s. At 2.9M noisy cells filled contour is 4.8 s / 0.5 GB.
#: So "auto" contours only small results and draws the rest with imshow, and an
#: explicit contour or pcolormesh past HOV_RENDER_CELLS is refused, not left to
#: tie up a login node for half a minute. Display only: the values are the same.
HOV_CONTOUR_CELLS = 1_000_000
HOV_RENDER_CELLS = 3_000_000


#: kept as a name: `hovmoller` raises it when a job is superseded, and it is
#: what `jobs.Jobs` catches to end a cancelled read quietly
HovmollerCancelled = _jobs.Cancelled


def _hov_columns(lon_file: np.ndarray, cyclic: bool, lon0: float, lon1: float):
    """Which columns a longitude range covers, as contiguous file slices.

    Returns (pieces, x). `pieces` are (file_slice, reversed) in ascending-x
    order -- one slice, or two when the range wraps the 0/360 seam, because a
    contiguous read is far cheaper than the integer-sequence read an unsorted
    index list falls back to. `x` is the longitude of every selected column,
    strictly increasing and unwrapped past the seam, so a range from 330 to
    30 reads as 330..390 rather than folding back on itself.

    No interpolation: the columns are the grid's own. On a grid that repeats
    its seam column (0 and 360), a full turn drops the duplicate.
    """
    n = lon_file.size
    descending = n > 1 and lon_file[0] > lon_file[-1]
    asc = lon_file[::-1] if descending else lon_file
    asc = np.asarray(asc, dtype=np.float64)

    def piece(k0: int, k1: int):                       # ascending indices, inclusive
        if descending:
            return slice(n - 1 - k1, n - k0), True
        return slice(k0, k1 + 1), False

    if lon1 <= lon0:
        raise ValueError(f"longitude range {lon0:g}..{lon1:g} must increase")
    if n == 1:
        return [piece(0, 0)], asc.copy()
    step = float(np.median(np.diff(asc)))
    usable = n - 1 if cyclic and asc[-1] - asc[0] >= 360.0 - 0.5 * step else n

    if not cyclic:
        k = np.nonzero((asc >= lon0) & (asc <= lon1))[0]
        if not k.size:
            raise ValueError(f"no grid column between longitude {lon0:g} and {lon1:g}")
        return [piece(int(k[0]), int(k[-1]))], asc[k[0]:k[-1] + 1].copy()

    if lon1 - lon0 >= 360.0 - 0.5 * step:              # the whole circle
        shift = 360.0 * np.floor((lon0 - asc[0]) / 360.0)
        return [piece(0, usable - 1)], asc[:usable] + shift

    lo = asc[0] + np.mod(lon0 - asc[0], 360.0)          # lon0 in the grid's own turn
    hi = lo + (lon1 - lon0)
    shift = lon0 - lo                                   # a multiple of 360
    first = np.nonzero((asc[:usable] >= lo) & (asc[:usable] <= hi))[0]
    second = np.nonzero(asc[:usable] + 360.0 <= hi)[0]
    pieces, xs = [], []
    if first.size:
        pieces.append(piece(int(first[0]), int(first[-1])))
        xs.append(asc[first[0]:first[-1] + 1])
    if second.size:
        pieces.append(piece(int(second[0]), int(second[-1])))
        xs.append(asc[second[0]:second[-1] + 1] + 360.0)
    if not pieces:
        raise ValueError(f"no grid column between longitude {lon0:g} and {lon1:g}")
    return pieces, np.concatenate(xs) + shift


def _fmt_times(values, units: str = "") -> list[str]:
    """Labels for one file's time coordinate, as the slider should print them."""
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.datetime64):
        secs = values.astype("datetime64[s]")
        unit = "m" if np.all(secs.astype(np.int64) % 60 == 0) else "s"
        return [np.datetime_as_string(v, unit=unit).replace("T", " ") for v in secs]
    if values.dtype == object:                                   # cftime
        return [v.strftime("%Y-%m-%d %H:%M") if hasattr(v, "strftime") else str(v)
                for v in values]
    if units:
        return [f"{v} {units}" for v in values]
    return [str(v) for v in values]


def _nearest_along(coords: np.ndarray, targets: np.ndarray, cyclic: bool):
    """(index, inside) of the nearest `coords` entry for every target.

    `coords` ascending. A target further than half a cell beyond either end is
    outside the grid -- it must draw as nothing, not as the edge row smeared
    outwards. On a `cyclic` longitude axis nothing is outside: 370 is 10.
    """
    n = coords.size
    if n == 1:
        return np.zeros(targets.size, int), np.ones(targets.size, bool)
    if cyclic:
        lo = coords[0]
        t = lo + np.mod(targets - lo, 360.0)
        ext = np.append(coords, lo + 360.0)          # the seam, as one more cell
        idx = np.clip(np.searchsorted(ext, t), 1, n)
        left, right = ext[idx - 1], ext[idx]
        near = np.where(t - left <= right - t, idx - 1, idx) % n
        return near, np.ones(targets.size, bool)
    idx = np.clip(np.searchsorted(coords, targets), 1, n - 1)
    left, right = coords[idx - 1], coords[idx]
    near = np.where(targets - left <= right - targets, idx - 1, idx)
    half_lo, half_hi = (coords[1] - coords[0]) / 2, (coords[-1] - coords[-2]) / 2
    inside = (targets >= coords[0] - half_lo) & (targets <= coords[-1] + half_hi)
    return near, inside


def _padded(coords: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Indices of `coords` inside [lo, hi], widened by one cell on each side.

    `coords` in file order, ascending or not. A box between two cells takes
    the cells around its centre.
    """
    inside = np.nonzero((coords >= lo) & (coords <= hi))[0]
    if inside.size == 0:
        inside = np.array([int(np.argmin(np.abs(coords - 0.5 * (lo + hi))))])
    first = max(int(inside.min()) - 1, 0)
    last = min(int(inside.max()) + 1, coords.size - 1)
    return np.arange(first, last + 1)


class GenericViewer:
    """Same public surface as `viewer.Viewer`, backed by plain netCDF files on
    one regular grid instead of an MPAS mesh + Series.

    Also stands in as its own `series`: the handler reads `series.labels`,
    `series.scanning` and `len(series)`, and here the time axis *is* the
    viewer's -- there is no separate object for it to belong to.
    """

    def __init__(self, paths, background_scan: bool = False):
        import xarray as xr  # noqa: F401  (the dependency, stated where it is used)

        self.files = expand(paths)
        self._open: OrderedDict[Path, object] = OrderedDict()
        # process-wide and reentrant: see netcdf.LOCK. A dashboard's other
        # sources, and this viewer's own background scan, enter HDF5 too.
        self._lock = netcdf.LOCK

        with self._lock:
            self.ds = self._open_first()
        self.path = self.files[0]

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
        lon = np.asarray(self.ds[self.lon_name].values, dtype=np.float64)
        self._lat_file, self._lon_file = lat, lon
        # canonical ascending order; file order is recovered at read time
        self._lat_flip = lat.size > 1 and lat[0] > lat[-1]
        self._lon_flip = lon.size > 1 and lon[0] > lon[-1]
        self.lat = lat[::-1] if self._lat_flip else lat
        self.lon = lon[::-1] if self._lon_flip else lon
        step = float(np.median(np.diff(self.lon))) if self.lon.size > 1 else 360.0
        # covers the whole circle (with or without a repeated seam column)
        self.cyclic = self.lon.size > 1 and (self.lon[-1] - self.lon[0] + step
                                             >= 360.0 - 0.5 * step)

        # The browser sizes its box off nx/ny and asks for frames at that
        # density; frames are then sampled onto exactly the pixels asked for,
        # so the box and the data agree whatever the grid's spacing.
        self.nx = self.lon.size
        self.ny = self.lat.size
        self.home = (float(self.lon.min()), float(self.lon.max()),
                     float(self.lat.min()), float(self.lat.max()))

        self._counts = {self.files[0]: self._count(self.ds)}
        self._labels_of = {self.files[0]: self._labels_in(self.ds)}
        self._steps, self.labels = self._axis()
        self.scanning = False
        self.series = self
        palettes.register()           # cmo.*, ferret.*, grads.* for this viewer's picker
        # Hovmöller results and the jobs reading them; see hovmoller_progress
        self._hov_jobs = _jobs.Jobs(name="gmpas-hovmoller")

        if background_scan and len(self.files) > 1:
            self.scanning = True
            threading.Thread(target=self._scan, daemon=True).start()
        elif len(self.files) > 1:
            self._scan()

    # -- files and the time axis ----------------------------------------

    def _open_first(self):
        """The first file that opens, reporting (not hiding) any that do not.

        Caller must hold `self._lock`. The same policy as `Series._open_first`,
        for the same reason: a half-written newest file must not end the run.
        """
        problems = []
        while self.files:
            try:
                ds = self._dataset(self.files[0], check=False)
            except Exception as exc:
                problems.append(f"  {self.files.pop(0).name}: {exc}")
                continue
            if problems:
                print(f"gmpas: skipped {len(problems)} unreadable file(s):\n"
                      + "\n".join(problems), file=sys.stderr)
            return ds
        raise OSError("no readable file among those matched:\n" + "\n".join(problems))

    def _dataset(self, path: Path, check: bool = True):
        """Caller must hold `self._lock`. Every file but the first is checked
        against the first one's grid before any of it is used: two files that
        merely share variable names would otherwise draw one on the other's
        coordinates without complaint."""
        import xarray as xr

        if path in self._open:
            self._open.move_to_end(path)
            return self._open[path]
        ds = xr.open_dataset(path, decode_timedelta=False, engine="netcdf4")
        if check:
            problem = self._grid_problem(ds.sizes, ds[self.lat_name].values
                                         if self.lat_name in ds.variables else None,
                                         ds[self.lon_name].values
                                         if self.lon_name in ds.variables else None)
            if problem:
                ds.close()
                raise ValueError(f"{path.name}: {problem}")
        self._open[path] = ds
        while len(self._open) > LRU_SIZE:
            _, old = self._open.popitem(last=False)
            old.close()
        return ds

    def _grid_problem(self, sizes, lat, lon) -> str:
        if lat is None or lon is None:
            return f"no {self.lat_name}/{self.lon_name} coordinate, unlike {self.path.name}"
        if (sizes.get(self.lat_dim) != self._lat_file.size
                or sizes.get(self.lon_dim) != self._lon_file.size):
            return (f"grid is {sizes.get(self.lat_dim)}x{sizes.get(self.lon_dim)}, "
                    f"but {self.path.name} is {self._lat_file.size}x{self._lon_file.size}")
        if not (np.allclose(np.asarray(lat, float), self._lat_file, atol=1e-4)
                and np.allclose(np.asarray(lon, float), self._lon_file, atol=1e-4)):
            return f"same shape as {self.path.name}, but different coordinates"
        return ""

    def _count(self, ds) -> int:
        return int(ds.sizes[self.time_name]) if self.time_name in ds.sizes else 1

    def _labels_in(self, ds) -> list[str] | None:
        """Real time labels for one open file; None to fall back to its name."""
        if self.time_name is None or self.time_name not in ds.sizes:
            return None
        if self.time_name not in ds.variables:
            return None
        var = ds[self.time_name]
        return _fmt_times(var.values, str(var.attrs.get("units", ""))
                          if not np.issubdtype(var.dtype, np.datetime64) else "")

    def _axis(self):
        steps, labels = [], []
        for path in self.files:
            n = self._counts.get(path, 1)
            known = self._labels_of.get(path)
            base = label_of(path)
            for i in range(n):
                steps.append((path, i))
                if known and i < len(known):
                    labels.append(known[i])
                elif self.time_name and self.time_name not in self.ds.variables \
                        and len(self.files) == 1:
                    labels.append(f"step {i}")
                else:
                    labels.append(base if n == 1 else f"{base} +{i}")
        return steps, labels

    def _scan(self) -> None:
        """Count steps, read time labels and check the grid of every file.

        netCDF4 directly, like `Series._scan`: a dimension length and one 1D
        coordinate need none of xarray's decoding. The lock is taken per file
        so a frame request waits for one file's read, not the whole scan. A
        file on a different grid is dropped from the axis and named on stderr.
        """
        import cftime
        import netCDF4

        counts, labels, dropped = dict(self._counts), dict(self._labels_of), []
        for path in self.files:
            if path in counts:
                continue
            try:
                with self._lock, netCDF4.Dataset(path) as nc:
                    nc.set_auto_mask(False)
                    sizes = {k: len(v) for k, v in nc.dimensions.items()}
                    problem = self._grid_problem(
                        sizes,
                        nc[self.lat_name][:] if self.lat_name in nc.variables else None,
                        nc[self.lon_name][:] if self.lon_name in nc.variables else None)
                    if problem:
                        dropped.append(f"  {path.name}: {problem}")
                        continue
                    counts[path] = sizes.get(self.time_name, 1) if self.time_name else 1
                    if self.time_name in nc.variables and self.time_name in sizes:
                        tv = nc[self.time_name]
                        units = getattr(tv, "units", "")
                        if " since " in units:
                            dates = cftime.num2date(tv[:], units,
                                                    getattr(tv, "calendar", "standard"))
                            labels[path] = _fmt_times(np.asarray(dates, dtype=object))
                        else:
                            labels[path] = _fmt_times(tv[:], units)
            except Exception as exc:
                dropped.append(f"  {path.name}: {exc}")
        if dropped:
            print(f"gmpas: left {len(dropped)} file(s) out of the time axis:\n"
                  + "\n".join(dropped), file=sys.stderr)
            keep = set(counts)
            self.files = [f for f in self.files if f in keep]
        self._counts, self._labels_of = counts, labels
        self._steps, self.labels = self._axis()
        self.scanning = False

    def __len__(self) -> int:
        return len(self._steps)

    @property
    def steps(self) -> int:
        """How many timesteps, across every file -- a count, as it always was."""
        return len(self._steps)

    @property
    def title(self) -> str:
        first = self.files[0].name
        return first if len(self.files) == 1 else f"{first}  +{len(self.files) - 1} more"

    def stop_jobs(self, timeout: float = 30.0) -> None:
        """Cancel every Hovmöller job and wait for it to let go of its files.

        A job reads on its own thread; left running past its viewer -- a
        server shutting down, a test finishing -- it keeps entering HDF5 while
        whatever runs next may be writing a file without the lock.
        """
        self._hov_jobs.stop(timeout)

    def close(self) -> None:
        self.stop_jobs()
        with self._lock:
            for ds in self._open.values():
                ds.close()
            self._open.clear()

    def _read(self, var: str, step: int):
        """One step of `var`, lazy. Caller must hold `self._lock`."""
        if not -len(self._steps) <= step < len(self._steps):
            raise IndexError(f"time={step} out of range ({len(self._steps)} steps)")
        path, local = self._steps[step]
        ds = self._dataset(path, check=path != self.files[0])
        if var not in ds:
            raise KeyError(f"{var!r} not in {path.name}")
        da = ds[var]
        if self.time_name in da.dims:
            da = da.isel({self.time_name: local})
        return da

    # -- variables -------------------------------------------------------

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

    def _pick_levels(self, da, level: int):
        """Index the stacking axes away: the slider's by `level`, the rest at 0."""
        picks = {d: (level if k == 0 else 0) for k, d in enumerate(self._stack_dims(da))}
        for dim, idx in picks.items():
            n = da.sizes[dim]
            if not -n <= idx < n:
                raise IndexError(f"{dim}={idx} out of range for {da.name!r} (size {n})")
        return da.isel(picks) if picks else da

    def kinds(self, name: str) -> list[str]:
        """What `plot` can draw for this variable, first entry the default."""
        da = self.ds[name]
        if self.lat_dim in da.dims and self.lon_dim in da.dims:
            out = ["map", *_GRID_KINDS, "hist"]
            if self.time_name in da.dims or len(self.files) > 1:
                out.append("series")
                out.append("hovmoller")
            if self._stack_dims(da):
                out.append("profile")
            out.append("layers")
            return out
        nd = da.ndim
        if nd == 0:
            return ["auto"]
        if nd == 1:
            return ["auto", "line", "step", "hist"]
        if nd == 2:
            return ["auto", *_GRID_KINDS, "line", "hist"]
        return ["auto", "hist"]

    def describe(self) -> dict:
        variables = []
        for name, da in self.ds.data_vars.items():
            spatial = self.lat_dim in da.dims and self.lon_dim in da.dims
            stack = self._stack_dims(da)
            variables.append({
                "name": name,
                "label": _data.field_label(da),
                "static": not (self.time_name and self.time_name in da.dims)
                          and len(self.files) == 1,
                # the slider drives the first stacking axis, the rest are
                # held at 0 -- the same contract as viewer.Viewer.describe
                "levels": int(da.sizes[stack[0]]) if spatial and stack else 1,
                "dim": stack[0] if spatial and stack else "",
                "pinned": stack[1:] if spatial else [],
                "spatial": spatial,
                "kinds": self.kinds(name),
            })
        variables.sort(key=lambda v: (not v["spatial"], v["static"], v["name"]))
        return {
            "file": self.title,
            "mesh": self.path.name,        # not read by the frontend; kept for the contract
            "cells": int(self.nx * self.ny),
            "regional": not self.cyclic,
            "coverage": 100.0,
            "files": len(self.files),
            "steps": len(self._steps),
            "labels": self.labels,
            "scanning": self.scanning,
            "home": list(self.home),
            "nx": self.nx,
            "ny": self.ny,
            **_colour.describe(),
            "kind_labels": KIND_LABELS,
            "kind_caps": KIND_CAPS,
            "layer_schema": _layers.schema(),
            "variables": variables,
        }

    # -- map frames ------------------------------------------------------

    def _gather(self, var, step, level, rows, cols) -> np.ndarray:
        """Values at ascending-order (rows x cols) grid indices.

        Reads only the bounding window of what is asked for, in the file's
        own order, then gathers -- a zoomed view of a global grid does not
        read the globe.
        """
        frow = (self.lat.size - 1 - rows) if self._lat_flip else rows
        fcol = (self.lon.size - 1 - cols) if self._lon_flip else cols
        r0, r1 = int(frow.min()), int(frow.max()) + 1
        c0, c1 = int(fcol.min()), int(fcol.max()) + 1
        with self._lock:
            da = self._pick_levels(self._read(var, step), level)
            window = da.isel({self.lat_dim: slice(r0, r1), self.lon_dim: slice(c0, c1)})
            arr = np.asarray(window.transpose(self.lat_dim, self.lon_dim).values,
                             dtype=np.float64)
        return arr[np.ix_(frow - r0, fcol - c0)]

    @staticmethod
    def clean_colour(colour) -> dict:
        """The fast map's colour options, checked (see `clean_colour`)."""
        return clean_colour(colour)

    def _raster(self, var, step, level, extent, nx, ny) -> np.ndarray:
        return self._raster_masked(var, step, level, extent, nx, ny)[0]

    def _raster_masked(self, var, step, level, extent, nx, ny):
        """`_raster`, and which pixels fall on the grid at all -- so a colour for
        missing cells paints NaN data but never the area beyond a regional grid."""
        """The field sampled onto exactly the (ny, nx) pixels of `extent`.

        Row 0 is the southernmost row, as `_png` expects. Pixels beyond a
        regional grid are NaN, so they draw transparent. Earlier this returned
        the grid's own cells inside the box instead, and the browser stretched
        that over the whole box it had asked for -- which is 1.4x the window,
        so at the default global view the data sat ~30 degrees off the
        coastlines and every map started out wrong.
        """
        lon_t, lat_t = target_grid(tuple(extent), nx, ny)
        cols, col_in = _nearest_along(self.lon, lon_t, self.cyclic)
        rows, row_in = _nearest_along(self.lat, lat_t, False)
        img = np.full((ny, nx), np.nan)
        on_grid = row_in[:, None] & col_in[None, :]
        if not (row_in.any() and col_in.any()):
            return img, on_grid
        vals = self._gather(var, step, level, rows[row_in], cols[col_in])
        img[np.ix_(row_in, col_in)] = vals
        return img, on_grid

    def _slice(self, var: str, time: int, level: int, extent) -> np.ndarray:
        """The grid's own cells inside `extent`, ascending, at native resolution."""
        lon_min, lon_max, lat_min, lat_max = extent
        j = np.nonzero((self.lon >= lon_min) & (self.lon <= lon_max))[0]
        i = np.nonzero((self.lat >= lat_min) & (self.lat <= lat_max))[0]
        if not (i.size and j.size):
            return np.full((1, 1), np.nan)
        return self._gather(var, time, level, i, j)

    def frame(self, var, time, level, extent, cmap, vmin, vmax,
              nx=None, ny=None, compress=1, colour=None, meta=None):
        """A map frame. With no `colour` options this is `viewer._png`, byte for
        byte; with them, `palettes.encode`, and `meta["colorbar"]` describes
        the bar the page should draw beside it."""
        nx, ny = nx or self.nx, ny or self.ny
        if var not in self._spatial_vars():
            # no colour range for a plain plot; 0..1 is an unused placeholder
            return self.plot(var, time, level, "auto", extent, nx, ny), 0.0, 1.0

        if not clean_colour(colour):
            # the plain path does not need the mask, so it does not build one
            img = self._raster(var, time, level, extent, nx, ny)
            lo, hi = self._range(img, vmin, vmax)
            return _png(img, cmap, lo, hi, compress), lo, hi

        img, on_grid = self._raster_masked(var, time, level, extent, nx, ny)
        lo, hi = self._range(img, vmin, vmax)
        return _colour.frame_png(img, cmap, lo, hi, compress, colour,
                                 outside=~on_grid, meta=meta), lo, hi

    @staticmethod
    def _range(img, vmin, vmax) -> tuple[float, float]:
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
        return lo, hi

    def overlay(self, extent, nx=None, ny=None) -> bytes:
        return _overlay(extent, nx or self.nx, ny or self.ny)

    def _point(self, lon: float, lat: float) -> tuple[int, int]:
        """Ascending (row, col) of the cell nearest a point, or an error off-grid."""
        cols, col_in = _nearest_along(self.lon, np.array([float(lon)]), self.cyclic)
        rows, row_in = _nearest_along(self.lat, np.array([float(lat)]), False)
        if not (row_in[0] and col_in[0]):
            raise ValueError(f"({lat:g}, {lon:g}) is outside the grid")
        return int(rows[0]), int(cols[0])

    def probe(self, lon, lat, var, time, level):
        try:
            i, j = self._point(lon, lat)
        except ValueError:
            return {"cell": -1, "lon": float(lon), "lat": float(lat), "value": float("nan")}
        value = float(self._gather(var, time, level, np.array([i]), np.array([j]))[0, 0])
        return {"cell": i * self.lon.size + j,
                "lon": round(float(self.lon[j]), 4),
                "lat": round(float(self.lat[i]), 4),
                "value": value}

    def series_at_point(self, lon, lat, var, level=0, blocking=False) -> dict:
        """The clicked grid point's value at every step, as the page wants it.

        The same shape the MPAS viewer answers with, through the same runner:
        a count while it reads, the numbers when it is done.
        """
        if var not in self._spatial_vars():
            raise ValueError(f"{var!r} is not a map variable, so it has no "
                             f"series at a point")
        i, j = self._point(lon, lat)              # raises off-grid, with the point
        key = ("series", var, i, j, int(level))

        def work(progress, cancel, publish=None):
            return self._series_at(var, lon, lat, int(level),
                                   progress=progress, cancel=cancel)

        total = len(self.files)
        found = (self._hov_jobs.result(key, total, work) if blocking
                 else self._hov_jobs.peek(key))
        if found is None:
            return {"state": "running", **self._hov_jobs.progress(key, total, work)}
        values = np.asarray(found.values, dtype=float)
        axis = found.coords[found.dims[0]].values
        return {"state": "done", "cell": i * self.lon.size + j,
                "lon": round(float(self.lon[j]), 4),
                "lat": round(float(self.lat[i]), 4),
                "label": _data.field_label(self.ds[var]),
                "labels": [str(np.datetime_as_string(t, unit="m"))
                           if np.issubdtype(np.asarray(t).dtype, np.datetime64)
                           else str(t) for t in axis],
                "values": [None if not np.isfinite(v) else float(v) for v in values]}

    # -- Hovmöller: time x longitude, averaged over a latitude band ---------

    def _hov_spec(self, var: str, level: int, band, lons=None, steps=None) -> dict:
        """Resolve a Hovmöller request to grid indices, or refuse it by name.

        Everything downstream -- the reads, the job key, the size estimate --
        works from these resolved indices, never from the floats asked for, so
        two bands that select the same rows are the same Hovmöller.
        """
        if self.scanning:
            raise ValueError(
                "the time axis is still being counted across the files; a "
                "Hovmöller over steps whose numbering is about to change would "
                "mix them up -- try again when the scan finishes"
            )
        if var not in self._spatial_vars():
            raise ValueError(f"{var!r} is not on the map, so it has no Hovmöller")
        lat0, lat1 = (float(v) for v in band)
        if not lat1 >= lat0:
            raise ValueError(f"latitude band {lat0:g}..{lat1:g} must not decrease")
        rows = np.nonzero((self._lat_file >= lat0) & (self._lat_file <= lat1))[0]
        if not rows.size:
            raise ValueError(f"no grid row between latitude {lat0:g} and {lat1:g}")
        if lons is None:
            lons = (float(self.lon.min()), float(self.lon.min()) + 360.0) if self.cyclic \
                else (float(self.lon.min()), float(self.lon.max()))
        pieces, x = _hov_columns(self._lon_file, self.cyclic,
                                 float(lons[0]), float(lons[1]))

        n = len(self._steps)
        a, b = (0, n - 1) if steps is None else (int(steps[0]), int(steps[1]))
        if not 0 <= a <= b < n:
            raise ValueError(f"steps {a}..{b} are outside 0..{n - 1}")
        snapshot = tuple(self._steps[a:b + 1])     # a plain list slice: atomic

        estimate = len(snapshot) * x.size * 8
        if estimate > self._hov_jobs.cache.budget:
            raise ValueError(
                f"{len(snapshot)} steps x {x.size} longitudes is "
                f"~{estimate / 2**20:.0f} MB, over the "
                f"{self._hov_jobs.cache.budget // 2**20} MB "
                f"a result may take; narrow the step or longitude range "
                f"(GMPAS_VIEW_CACHE_MB raises the limit)"
            )
        stack = self._stack_dims(self.ds[var])
        if stack and not 0 <= int(level) < int(self.ds[var].sizes[stack[0]]):
            raise ValueError(f"{stack[0]}={level} is out of range for {var!r}")
        r0, r1 = int(rows[0]), int(rows[-1])
        key = (var, int(level), r0, r1,
               tuple((sl.start, sl.stop, rev) for sl, rev in pieces),
               tuple((str(path), local) for path, local in snapshot))
        return {"var": var, "level": int(level), "rows": slice(r0, r1 + 1),
                "lat": np.asarray(self._lat_file[r0:r1 + 1], dtype=np.float64),
                "pieces": pieces, "x": x, "steps": snapshot, "first": a, "key": key}

    def _hov_chunk_steps(self, da, n_rows: int, n_cols: int) -> int:
        """How many steps one read may take.

        Bounded by what a read decodes, not only by the band it keeps: ERA5
        chunks a whole global field per step, so a 10-degree band still
        decompresses the globe. Values count as 8 bytes -- packed int16
        decodes to floating point.
        """
        per_step = n_rows * n_cols * 8
        chunks = da.encoding.get("chunksizes")
        if chunks and len(chunks) == da.ndim:
            sizes = dict(zip(da.dims, chunks, strict=False))
            touched = 8
            for dim, want in ((self.lat_dim, n_rows), (self.lon_dim, n_cols)):
                size = int(sizes.get(dim, want) or want)
                touched *= max(want, size)
            per_step = max(per_step, touched)
        return max(1, HOV_READ_BYTES // max(per_step, 1))

    def hovmoller(self, var: str, level: int = 0, band=(-15.0, 15.0), lons=None,
                  steps=None, progress=None, cancel=None):
        """`var` averaged over a latitude band, as (time, lon), across every file.

        The band mean weights each row by cos(latitude) -- the area a row's
        cells cover on the sphere -- and skips NaNs, renormalising the weights
        over the cells that are valid; a longitude with none is NaN. There is
        no interpolation: longitude is the grid's own columns, and the rows
        averaged are those whose centres lie inside the band, inclusive.

        Reads go file by file in chunks of steps, each open-and-read under the
        netCDF lock and reduced outside it, so frames keep being served while a
        long run is read. `progress(done, total)` is called after each chunk;
        `cancel`, an Event, stops the run at the next one.
        """
        import xarray as xr

        spec = self._hov_spec(var, level, band, lons, steps)
        total = len(spec["steps"])
        w = np.cos(np.deg2rad(spec["lat"]))[None, :, None]
        out = np.full((total, spec["x"].size), np.nan)
        times: list = []
        dated = self.time_name is not None

        # consecutive steps of one file read together
        runs: list[tuple[Path, list[int]]] = []
        for path, local in spec["steps"]:
            if runs and runs[-1][0] == path and runs[-1][1][-1] == local - 1:
                runs[-1][1].append(local)
            else:
                runs.append((path, [local]))

        done = 0
        with timing.step("generic.hovmoller", steps=total, files=len(runs),
                         cols=int(spec["x"].size)):
            for path, locals_ in runs:
                start = 0
                while start < len(locals_):
                    if cancel is not None and cancel.is_set():
                        raise HovmollerCancelled()
                    with self._lock:
                        ds = self._dataset(path, check=path != self.files[0])
                        if var not in ds:
                            raise KeyError(f"{var!r} not in {path.name}")
                        da = ds[var]
                        count = self._hov_chunk_steps(da, w.shape[1], spec["x"].size)
                        take = locals_[start:start + count]
                        has_time = self.time_name in da.dims
                        if has_time:
                            da = da.isel({self.time_name: slice(take[0], take[-1] + 1)})
                        da = self._pick_levels(da, spec["level"])
                        da = da.isel({self.lat_dim: spec["rows"]})
                        parts = []
                        for sl, rev in spec["pieces"]:
                            part = da.isel({self.lon_dim: sl})
                            order = ((self.time_name,) if has_time else ()) + \
                                (self.lat_dim, self.lon_dim)
                            arr = np.asarray(part.transpose(*order).values,
                                             dtype=np.float64)
                            parts.append(arr[..., ::-1] if rev else arr)
                        if dated:
                            tvar = ds.variables.get(self.time_name)
                            if has_time and tvar is not None and np.issubdtype(
                                    tvar.dtype, np.datetime64):
                                times.extend(tvar.values[take[0]:take[-1] + 1])
                            else:
                                dated = False
                    _time.sleep(0)                       # let a frame request in
                    block = np.concatenate(parts, axis=-1)
                    if not has_time:
                        block = block[None]
                    valid = np.isfinite(block)
                    num = np.where(valid, block * w, 0.0).sum(axis=1)
                    den = (valid * w).sum(axis=1)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        out[done:done + len(take)] = np.where(den > 0, num / den, np.nan)
                    done += len(take)
                    start += len(take)
                    if progress is not None:
                        progress(done, total)

        if dated and len(times) == total:
            tname, tvals = self.time_name, np.asarray(times)
        else:
            tname, tvals = "step", np.arange(spec["first"], spec["first"] + total)
        src = self.ds[var]
        lat = spec["lat"]
        return xr.DataArray(
            out, dims=(tname, "lon"), coords={tname: tvals, "lon": spec["x"]}, name=var,
            attrs={**{k: v for k, v in src.attrs.items()
                      if k in ("units", "long_name", "standard_name")},
                   "band": f"latitude {min(lat[0], lat[-1]):g} to {max(lat[0], lat[-1]):g} "
                           f"({lat.size} rows)",
                   "longitude_range": f"{spec['x'][0]:g} to {spec['x'][-1]:g}",
                   "weighting": "cos(latitude), NaN skipped",
                   "level": self._level_text(var, spec["level"]) or "none"})

    # -- Hovmöller jobs: long runs read in the background --------------------

    def hovmoller_progress(self, var: str, level: int, hov: dict | None, extent) -> dict:
        """Where a Hovmöller stands, starting it if nothing has: done, running or
        error. Never blocks; the page polls this through HTTP 202."""
        spec = self._hov_spec(var, level, *self._hov_args(hov, extent))
        key = spec["key"]
        return self._hov_jobs.progress(key, len(spec["steps"]),
                                       self._hov_work(var, level, hov, extent))

    def _hov_args(self, hov: dict | None, extent):
        hov = hov or {}
        band = hov.get("band") or (extent[2], extent[3])
        lons = hov.get("lons") or (extent[0], extent[1])
        return band, lons, hov.get("steps")

    def _hov_work(self, var, level, hov, extent):
        """The read itself, as the runner wants it: progress in, result out."""
        band, lons, steps = self._hov_args(hov, extent)

        def work(progress, cancel, publish=None):
            return self.hovmoller(var, level, band, lons, steps,
                                  progress=progress, cancel=cancel)
        return work

    def _hov_result(self, var: str, level: int, hov: dict | None, extent):
        """The Hovmöller, waiting for its job if one is reading it. For figures
        and exports, which must return the finished thing."""
        spec = self._hov_spec(var, level, *self._hov_args(hov, extent))
        return self._hov_jobs.result(spec["key"], len(spec["steps"]),
                                     self._hov_work(var, level, hov, extent))

    # -- plots, the way xarray.DataArray.plot draws them -----------------

    def _file_index(self, i: int, j: int) -> dict:
        fi = (self.lat.size - 1 - i) if self._lat_flip else i
        fj = (self.lon.size - 1 - j) if self._lon_flip else j
        return {self.lat_dim: fi, self.lon_dim: fj}

    def _series_at(self, var: str, lon: float, lat: float, level: int = 0,
                   progress=None, cancel=None):
        """`var` at one point, through every step of every file, as a DataArray.

        One read per file rather than per step, each under the lock only for
        that file, so a long series does not freeze the rest of the viewer.
        The read is a hyperslab at one grid point, so what grows here is the
        answer and nothing else.

        `progress(done, total)` is called per file and `cancel` is checked
        there, so this can run under `jobs.Jobs` for a long run.
        """
        import xarray as xr

        i, j = self._point(lon, lat)
        where = self._file_index(i, j)
        values, times = [], []
        for done, path in enumerate(self.files):
            if cancel is not None and cancel.is_set():
                raise _jobs.Cancelled()
            with self._lock:
                ds = self._dataset(path, check=path != self.files[0])
                if var not in ds:
                    raise KeyError(f"{var!r} not in {path.name}")
                da = self._pick_levels(ds[var], level).isel(where)
                values.append(np.atleast_1d(np.asarray(da.values, dtype=np.float64)))
                if self.time_name in da.dims and self.time_name in ds.variables:
                    times.append(np.atleast_1d(ds[self.time_name].values))
            if progress is not None:
                progress(done + 1, len(self.files))
            _time.sleep(0)                       # let a frame request in
        y = np.concatenate(values)
        dated = len(times) == len(self.files) and all(
            np.issubdtype(t.dtype, np.datetime64) for t in times)
        x = np.concatenate(times) if dated else np.arange(y.size)
        xname = self.time_name if dated else "step"
        return xr.DataArray(y, dims=(xname,), coords={xname: x}, name=var,
                            attrs=dict(self.ds[var].attrs))

    def _profile_at(self, var: str, time: int, lon: float, lat: float):
        """`var` along its level axis at one point and step, as a DataArray."""
        i, j = self._point(lon, lat)
        with self._lock:
            da = self._read(var, time).isel(self._file_index(i, j))
            stack = self._stack_dims(da)
            if not stack:
                raise ValueError(f"{var!r} has no level axis to draw a profile along")
            da = da.isel({d: 0 for d in stack[1:]}).load()
        return da

    def _whole(self, var: str):
        """A non-map variable in full -- across every file if it runs in time."""
        import xarray as xr

        first = self.ds[var]
        across = self.time_name in first.dims and len(self.files) > 1
        n_bytes = first.nbytes * (len(self.files) if across else 1)
        if n_bytes > PLOT_READ_BYTES:
            raise ValueError(
                f"{var!r} is ~{n_bytes / 2**20:.0f} MB across the files; plotting "
                f"reads it whole, and the limit is {PLOT_READ_BYTES // 2**20} MB."
            )
        if not across:
            with self._lock:
                return first.load()
        parts = []
        for path in self.files:
            with self._lock:
                parts.append(self._dataset(path, check=path != self.files[0])[var].load())
        return xr.concat(parts, dim=self.time_name)

    def _cropped(self, var: str, time: int, level: int, extent):
        """One step on the map, cut to `extent` in file order, coordinates kept
        -- so xarray labels the axes and titles the scalar coordinates itself."""
        lon_min, lon_max, lat_min, lat_max = extent
        # One cell beyond each edge of the box, so a fill reaches the frame
        # instead of stopping at the last centre inside it -- and so a view
        # zoomed into a coarse grid still holds the 2x2 a contour needs, rather
        # than one cell or none.
        lat_idx = _padded(self._lat_file, lat_min, lat_max)
        if lon_max - lon_min >= 360.0 or not self.cyclic:
            lon_idx = _padded(self._lon_file, lon_min, lon_max)
        else:                                            # a box across the seam
            lo = np.mod(lon_min - self.lon[0], 360.0) + self.lon[0]
            hi = lo + (lon_max - lon_min)
            wrapped = np.where(self._lon_file < lo, self._lon_file + 360.0, self._lon_file)
            lon_idx = np.nonzero((wrapped >= lo) & (wrapped <= hi))[0]
            if lon_idx.size < 2:
                lon_idx = _padded(wrapped, lo, hi)
        if not (lat_idx.size and lon_idx.size):
            raise ValueError("the view box holds no grid cells; zoom out or reset the view")
        with self._lock:
            da = self._pick_levels(self._read(var, time), level)
            da = da.isel({self.lat_dim: slice(int(lat_idx.min()), int(lat_idx.max()) + 1)})
            if lon_idx.size == self._lon_file.size or np.all(np.diff(lon_idx) == 1):
                span = slice(int(lon_idx.min()), int(lon_idx.max()) + 1)
                da = da.isel({self.lon_dim: span})
            else:                                        # wraps: take both pieces
                seam = int(np.nonzero(np.diff(lon_idx) != 1)[0][0]) + 1
                order = np.concatenate([lon_idx[seam:], lon_idx[:seam]])
                da = da.isel({self.lon_dim: order})
                # 350, 358, 0, 2 -> 350, 358, 360, 362: pcolormesh and contour
                # need a monotonic axis, and the map frame follows it past 180
                vals = np.asarray(self._lon_file[order], dtype=float)
                vals[1:] += 360.0 * np.cumsum(np.diff(vals) < 0)
                da = da.assign_coords({self.lon_name: (self.lon_dim, vals)})
            return da.load()

    def _draw(self, fig, var, time, level, kind, extent, cmap, vmin, vmax, lon, lat,
              hov=None, layers=None, colour=None):
        """Put one plot on `fig`. Shared by the live plot, figures and GIF frames."""
        import matplotlib.pyplot as plt  # noqa: F401  (backend already chosen)

        if kind not in self.kinds(var):
            raise ValueError(f"{kind!r} is not a plot of {var!r}; one of {self.kinds(var)}")
        if kind == "layers":
            stack = self.layer_stack(layers, var)
            return _layers.draw(self, fig, stack, time, level, extent)
        if kind == "hovmoller":
            return self._draw_hovmoller(fig, var, time, level, extent, cmap, vmin, vmax,
                                        hov)

        spatial = var in self._spatial_vars()
        if spatial and kind in (*_GRID_KINDS, "map"):
            da = self._cropped(var, time, level, extent)
            # The fast map exports as imshow when the grid is evenly spaced --
            # the same cells pcolormesh would draw, ~25x faster at ERA5's
            # million cells -- and as pcolormesh when it is not, where imshow
            # would put every cell at an even spacing it does not have.
            method = kind if kind != "map" else (
                "imshow" if self._even(da) else "pcolormesh")
            opts = dict(cmap=cmap or None, vmin=vmin, vmax=vmax)
            colours = clean_colour(colour)
            if colours:
                # the fast map's colours, in a figure: the same range rule as
                # the map (_range) and the same colormap and norm (palettes.scale)
                lo, hi = self._range(np.asarray(da.values, float), vmin, vmax)
                levels = None
                if colours.get("bands") and method in ("contour", "contourf"):
                    # contours band by their levels, not by a norm
                    levels = list(palettes.band_edges(lo, hi, colours["bands"]))
                    colours = {**colours, "bands": None}
                cm, norm, _ = palettes.scale({**colours, "cmap": cmap or "viridis"}, lo, hi)
                opts = dict(cmap=cm, norm=norm, extend=colours.get("extend") or "neither")
                if levels is not None:
                    opts["levels"] = levels
            if method != "contour":
                # under the map, not beside it: a 2:1 map leaves a vertical
                # colorbar twice the map's height and the plot squeezed
                opts["cbar_kwargs"] = {"orientation": "horizontal", "shrink": 0.8,
                                       "aspect": 45, "pad": 0.07,
                                       "label": _data.field_label(da)}
            try:
                import cartopy.crs as ccrs
            except ImportError:                          # plain lat/lon axes
                ax = fig.add_subplot()
                getattr(da.plot, method)(ax=ax, x=self.lon_name, y=self.lat_name, **opts)
                ax.set_title(self._title(var, time, level))
                return ax
            from .plot import _frame

            lon_vals = np.asarray(da[self.lon_name].values, dtype=float)
            lat_vals = np.asarray(da[self.lat_name].values, dtype=float)
            box = (float(lon_vals.min()), float(lon_vals.max()),
                   max(-90.0, float(lat_vals.min())), min(90.0, float(lat_vals.max())))
            # the same framing as the MPAS figures: a box past +180 is drawn
            # centred on itself rather than split down both edges of the map
            central, src, framed = _frame(box)
            ax = fig.add_subplot(projection=ccrs.PlateCarree(central_longitude=central))
            # Data expressed in the map's own frame. Handing cartopy degrees
            # east of Greenwich for a map centred elsewhere makes it reproject
            # every cell -- 10x slower for imshow and contourf on a global grid
            # -- to land them exactly where this shift puts them anyway.
            shifted = da.assign_coords({self.lon_name: da[self.lon_name] - central})
            getattr(shifted.plot, method)(ax=ax, x=self.lon_name, y=self.lat_name,
                                          transform=src, **opts)
            ax.coastlines(linewidth=0.6)
            gl = ax.gridlines(draw_labels=True, linewidth=0.3, linestyle="--", alpha=0.5)
            gl.top_labels = gl.right_labels = False
            gl.geo_labels = False            # see plot._basemap: the y=inf title bug
            # An explicit extent, not autoscale: pcolormesh infers cell edges half
            # a cell past each pole, and a map boundary beyond +-90 is not a valid
            # polygon to cartopy's gridliner.
            cell = abs(self.lon[1] - self.lon[0]) if self.lon.size > 1 else 1.0
            if self.cyclic and box[1] - box[0] >= 359.0 - 2 * cell:
                ax.set_global()
            else:
                ax.set_extent(framed, crs=src)
            ax.set_title(self._title(var, time, level))
            return ax

        ax = fig.add_subplot()
        if spatial and kind == "hist":
            self._cropped(var, time, level, extent).plot.hist(ax=ax)
            ax.set_title(self._title(var, time, level))
        elif kind == "series":
            pt = (lon, lat) if lon is not None else self._centre(extent)
            self._series_at(var, *pt, level=level).plot.line(ax=ax)
            ax.set_title(f"{var} at {pt[1]:.2f}°, {pt[0]:.2f}°")
        elif kind == "profile":
            pt = (lon, lat) if lon is not None else self._centre(extent)
            da = self._profile_at(var, time, *pt)
            dim = str(da.dims[0])
            down = (dim in self._vertical and dim in self.ds.variables and (
                str(self.ds[dim].attrs.get("positive", "")).lower() == "down"
                or _norm_units(self.ds[dim]) in _PRESSURE_UNITS))
            da.plot.line(ax=ax, y=dim, yincrease=not down)
            ax.set_title(f"{var} at {pt[1]:.2f}°, {pt[0]:.2f}° · "
                         f"{self.labels[time]}")
        else:
            da = self._whole(var)
            if da.ndim == 0:
                ax.text(0.5, 0.5, f"{var} = {float(da.values):g}", ha="center",
                        va="center", fontsize=16, transform=ax.transAxes)
                ax.set_axis_off()
            elif kind == "auto":
                da.plot(ax=ax)
            else:
                opts = {"cmap": cmap or None, "vmin": vmin, "vmax": vmax} \
                    if kind in _GRID_KINDS else {}
                getattr(da.plot, kind)(ax=ax, **opts)
        return ax

    def _hov_method(self, da, method: str) -> str:
        """The drawing a Hovmöller gets. "auto": filled contours while that is
        quick, then imshow on evenly spaced axes, else pcolormesh."""
        cells = int(da.size)
        auto = method in (None, "", "auto")
        if auto:
            if cells <= HOV_CONTOUR_CELLS:
                return "contourf"
            if self._hov_even(da):
                return "imshow"
            method = "pcolormesh"         # uneven time: imshow would misplace rows
        if method not in HOV_METHODS:
            raise ValueError(f"Hovmöller method {method!r} is not one of {HOV_METHODS}")
        if method != "imshow" and cells > HOV_RENDER_CELLS:
            hint = "narrow the step or longitude range" if auto else \
                "use auto, or narrow the step or longitude range"
            raise ValueError(
                f"{cells:,} cells is over the {HOV_RENDER_CELLS:,} a {method} Hovmöller "
                f"draws in reasonable time; {hint}"
            )
        return method

    @staticmethod
    def _hov_even(da) -> bool:
        for dim in da.dims:
            v = np.asarray(da[dim].values)
            v = v.astype("datetime64[s]").astype(np.int64) if np.issubdtype(
                v.dtype, np.datetime64) else v.astype(float)
            d = np.diff(v)
            if d.size and not np.allclose(d, d[0], rtol=1e-6):
                return False
        return True

    def _draw_hovmoller(self, fig, var, time, level, extent, cmap, vmin, vmax, hov):
        hov = hov or {}
        da = self._hov_result(var, level, hov, extent)
        method = self._hov_method(da, hov.get("method", "auto"))
        down = hov.get("ydir", "down") != "up"
        tdim = da.dims[0]
        ax = fig.add_subplot()
        opts = dict(ax=ax, x="lon", y=tdim, yincrease=not down)
        if method == "contour":
            da.plot.contour(cmap=cmap or None, vmin=vmin, vmax=vmax, **opts)
        else:
            kw = dict(cmap=cmap or None, vmin=vmin, vmax=vmax,
                      cbar_kwargs={"label": _data.field_label(self.ds[var]),
                                   "shrink": 0.9})
            if method == "imshow":
                # imshow on datetimes wants numbers; the extent carries the dates
                self._hov_imshow(ax, da, down, kw)
            else:
                getattr(da.plot, method)(**opts, **kw)
        level_text = self._level_text(var, level)
        parts = [var] + ([level_text] if level_text else []) + [
            f"mean {da.attrs['band'].split(' (')[0]}",
            f"lon {da.lon.values[0]:g}..{da.lon.values[-1]:g}"]
        ax.set_title(" · ".join(parts), fontsize=10)
        ax.set_xlabel("longitude [degrees east]")
        ax._gmpas_hov_time = (tdim, np.asarray(da[tdim].values))
        return ax

    def _hov_imshow(self, ax, da, down: bool, kw: dict):
        import matplotlib.dates as mdates

        tdim = da.dims[0]
        t = np.asarray(da[tdim].values)
        dated = np.issubdtype(t.dtype, np.datetime64)
        tnum = mdates.date2num(t) if dated else t.astype(float)
        lon = np.asarray(da.lon.values, float)
        half_t = 0.5 * (tnum[1] - tnum[0]) if tnum.size > 1 else 0.5
        half_x = 0.5 * (lon[1] - lon[0]) if lon.size > 1 else 0.5
        cbar = kw.pop("cbar_kwargs")
        im = ax.imshow(da.values, aspect="auto", interpolation="nearest",
                       origin="lower", cmap=kw["cmap"], vmin=kw["vmin"], vmax=kw["vmax"],
                       extent=(lon[0] - half_x, lon[-1] + half_x,
                               tnum[0] - half_t, tnum[-1] + half_t))
        if dated:
            ax.yaxis_date()
        if down:
            ax.invert_yaxis()
        ax.set_ylabel(tdim)
        ax.figure.colorbar(im, ax=ax, **cbar)

    @staticmethod
    def _hov_axis(fig, ax) -> dict:
        """Where each step's row sits in the rendered image, top-down pixels,
        with the axes' horizontal span -- the page's current-step marker."""
        import matplotlib.dates as mdates

        tdim, t = ax._gmpas_hov_time
        dated = np.issubdtype(t.dtype, np.datetime64)
        ynum = mdates.date2num(t) if dated else t.astype(float)
        height = fig.bbox.height
        pts = ax.transData.transform(np.column_stack([np.zeros_like(ynum), ynum]))
        box = ax.get_window_extent()
        return {"y": [int(round(height - y)) for y in pts[:, 1]],
                "box": [int(round(box.x0)), int(round(box.x1)),
                        int(round(fig.bbox.width)), int(round(height))]}

    def _title(self, var: str, time: int, level: int) -> str:
        """`t · 2024-01-01 06:00 · pressure_level 500 hPa`.

        xarray's own title lists every scalar coordinate and truncates the line
        -- `pressure_level = 1e+03...` -- which says less than this does.
        """
        parts = [var, self.labels[time]]
        level_text = self._level_text(var, level)
        if level_text:
            parts.append(level_text)
        return " · ".join(parts)

    def _level_text(self, var: str, level: int) -> str:
        """`pressure_level 500 hPa`, or "" for a variable with no level axis."""
        stack = self._stack_dims(self.ds[var])
        if not stack:
            return ""
        dim = stack[0]
        if dim in self.ds.variables and self.ds[dim].ndim == 1:
            coord = self.ds[dim]
            value = coord.values[level]
            units = coord.attrs.get("units", "")
            numeric = np.issubdtype(coord.dtype, np.number)
            shown = f"{value:g}" if numeric else str(value)
            return f"{dim} {shown} {units}".strip()
        return f"{dim} {level}"

    def _even(self, da) -> bool:
        """Whether both map axes of `da` are evenly spaced, as imshow assumes."""
        for name in (self.lon_name, self.lat_name):
            step = np.diff(np.asarray(da[name].values, dtype=float))
            if step.size and not np.allclose(step, step[0], rtol=1e-3, atol=1e-6):
                return False
        return True

    def layer_stack(self, stack, var: str) -> _layers.Stack:
        """A checked layer stack -- or, given none, the field over coastlines."""
        if isinstance(stack, _layers.Stack):
            return stack
        if stack is None or stack == "":
            stack = _layers.default_stack(var)
        spatial = self._spatial_vars()
        counts = {}
        for name in spatial:
            dims = self._stack_dims(self.ds[name])
            counts[name] = int(self.ds[name].sizes[dims[0]]) if dims else 1
        return _layers.clean(stack, spatial, counts)

    def _centre(self, extent) -> tuple[float, float]:
        return (0.5 * (extent[0] + extent[1]), 0.5 * (extent[2] + extent[3]))

    def _render(self, var, time, level, kind, extent, figsize, dpi,
                cmap=None, vmin=None, vmax=None, lon=None, lat=None, hov=None,
                meta=None, layers=None, colour=None) -> bytes:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=figsize, dpi=dpi, layout="constrained")
        try:
            ax = self._draw(fig, var, time, level, kind, extent, cmap, vmin, vmax,
                            lon, lat, hov, layers, colour)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi)
            if meta is not None and kind == "hovmoller":
                meta.update(self._hov_axis(fig, ax))
            return buf.getvalue()
        finally:
            plt.close(fig)

    def plot(self, var, time, level, kind, extent, width=900, height=560,
             cmap=None, vmin=None, vmax=None, lon=None, lat=None, hov=None,
             meta=None, layers=None, colour=None) -> bytes:
        """The live plot pane: `kind` drawn at the browser's pixel size.

        `hov` holds a Hovmöller's band, longitudes, steps, method and time
        direction; `meta`, a dict, receives where its steps landed in the image
        so the page can mark the current one without asking again. `layers` is
        the composite's stack (JSON text or a dict) when `kind` is "layers";
        see `gmpas.layers`.
        """
        width, height = int(np.clip(width, 200, 4000)), int(np.clip(height, 150, 3000))
        return self._render(var, time, level, kind, extent,
                            (width / 100, height / 100), 100, cmap, vmin, vmax, lon, lat,
                            hov, meta, layers, colour)

    # -- export ------------------------------------------------------------

    def figure(self, var, time, level, extent, cmap, vmin, vmax, style="paper",
               kind="map", lon=None, lat=None, hov=None, layers=None,
               colour=None) -> bytes:
        """A publication-shaped figure of what is on screen.

        Sized by the same `Style` presets as the MPAS path. The fast map
        exports as a pcolormesh on coastlines with labelled gridlines; every
        other kind exports as itself.
        """
        from .style import Style

        st = Style.preset(style)
        if var not in self._spatial_vars() and kind == "map":
            kind = "auto"
        return self._render(var, time, level, kind, extent, st.figsize, st.dpi,
                            cmap, vmin, vmax, lon, lat, hov, None, layers, colour)

    def gif(self, var, level, extent, cmap, vmin, vmax, nx=None, ny=None, fps=8,
            kind="map", lon=None, lat=None, hov=None, layers=None,
            colour=None) -> bytes:
        """Every timestep as one animated GIF, drawn the way the screen is.

        The fast map re-containers its own palette frames, as `Viewer.gif`
        does. A contour or a profile renders a figure per step instead, with
        its colour range (or its x limits) fixed from the whole run first, so
        the frames are comparable rather than each rescaled to itself.
        """
        from PIL import Image

        n = len(self._steps)
        spatial = var in self._spatial_vars()
        if kind not in self.kinds(var):
            raise ValueError(f"{kind!r} is not a plot of {var!r}; one of {self.kinds(var)}")
        if n < 2:
            raise ValueError("a GIF steps through time, and there is only one step")
        if not spatial or kind in ("series", "hist", "hovmoller"):
            raise ValueError(
                f"a GIF steps through time, and a {KIND_LABELS.get(kind, kind)} of "
                f"{var!r} is not drawn per step -- export a figure instead"
            )
        frames = []
        if kind == "map":
            nx, ny = nx or self.nx, ny or self.ny
            if vmin is None or vmax is None:
                vmin, vmax = self._range(self._raster(var, 0, level, extent, nx, ny),
                                         vmin, vmax)
            for step in range(n):
                png, _, _ = self.frame(var, step, level, extent, cmap, vmin, vmax,
                                       nx, ny, compress=1, colour=colour)
                frames.append(Image.open(io.BytesIO(png)).convert("P"))
            transparency = {"transparency": 255}
        else:
            if n > GIF_FIGURE_FRAMES:
                raise ValueError(
                    f"{n} steps would be {n} matplotlib figures (~{n * 0.3 / 60:.0f} "
                    f"min); a {KIND_LABELS[kind]} GIF is limited to "
                    f"{GIF_FIGURE_FRAMES}. Open fewer files, or use the map kind."
                )
            if kind == "profile":
                pt = (lon, lat) if lon is not None else self._centre(extent)
                prof = [self._profile_at(var, s, *pt) for s in range(n)]
                lo = min(float(np.nanmin(p.values)) for p in prof)
                hi = max(float(np.nanmax(p.values)) for p in prof)
            elif kind == "layers":
                layers = _layers.freeze_ranges(self, self.layer_stack(layers, var),
                                               level, extent)
            elif vmin is None or vmax is None:
                vmin, vmax = self._range(np.asarray(
                    self._cropped(var, 0, level, extent).values, float), vmin, vmax)
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            from .style import Style

            st = Style.preset("notebook")
            for step in range(n):
                fig = plt.figure(figsize=st.figsize, dpi=80, layout="constrained")
                try:
                    ax = self._draw(fig, var, step, level, kind, extent, cmap,
                                    vmin, vmax, lon, lat, hov, layers, colour)
                    if kind == "profile":
                        pad = 0.02 * (hi - lo or 1.0)
                        ax.set_xlim(lo - pad, hi + pad)
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=80)
                finally:
                    plt.close(fig)
                frames.append(Image.open(io.BytesIO(buf.getvalue()))
                              .convert("RGB").convert("P", palette=Image.ADAPTIVE))
            transparency = {}

        buf = io.BytesIO()
        frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:],
                       duration=max(20, int(1000 / max(fps, 1))), loop=0,
                       disposal=2, **transparency)
        return buf.getvalue()

    def netcdf(self, var, time, level, extent, nx=None, ny=None, kind=None,
               lon=None, lat=None, hov=None, layers=None, colour=None) -> bytes:
        """The numbers behind a Hovmöller, as netCDF. Other kinds: not yet."""
        if kind != "hovmoller":
            raise NotImplementedError(
                "netCDF export for --generic is implemented for the Hovmöller only")
        da = self._hov_result(var, level, hov, extent)
        ds = da.to_dataset()
        ds.attrs.update(source=", ".join(f.name for f in self.files[:20]) +
                        (" ..." if len(self.files) > 20 else ""),
                        history="gmpas --generic Hovmoller export")
        # writing netCDF4 bytes is HDF5 too, and a Hovmöller job may be reading
        # on another thread: the same process-wide lock every read takes
        with self._lock:
            return bytes(ds.to_netcdf())
