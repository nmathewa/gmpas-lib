"""A JIGSAW mesh distance function, before any mesh exists.

In the JIGSAW workflow for MPAS-Atmosphere, a variable-resolution mesh is
defined entirely by a small Python file, `hfun.py`, which says how large a cell
should be at every point on the sphere. Everything after it is mechanical:
`create_hfun.py` samples it onto a lat-lon grid and writes `HFUN.msh`, JIGSAW
turns that into generating points, and `mkgrid` turns those into `grid.nc`.

So the design decisions all live in that one file, and until now the only way
to see what they produced was to run the whole chain and look at the finished
mesh. This module reads `hfun.py` on its own, which is enough to draw the
resolution and to check the guidelines it has to satisfy.

The contract, from the mini-tutorial (Duda, MPAS/WRF Users Workshop 2026):

    hfun_min            module-level float, the minimum grid distance in km
    get_hfun(lon, lat)  lon/lat in RADIANS, returns grid distance in KM

with one property that shapes everything here -- `get_hfun` is called *once*,
with whole arrays, and so is allowed to do expensive setup. A real one may
interpolate a raster dataset. Nothing in this module may call it per pixel.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..paths import resolve_path

#: MPAS-Atmosphere's assumed Earth radius (km), as the tutorial scripts use it
R_EARTH_KM = 6371.229

#: km per degree of great circle, the constant `create_hfun.py` derives
DEG_TO_KM = 2.0 * np.pi * R_EARTH_KM / 360.0

#: "Limit gradients (km of cell size) / (km of distance) to a few percent!
#: Experience suggests a value of 0.03 is generally safe."
GRADIENT_GUIDELINE = 0.03

#: cap on the analysis grid, so a very fine `hfun_min` cannot ask for an array
#: that does not fit in memory. Only bites below hfun_min ~ 6 km.
MAX_ANALYSIS_POINTS = 20_000_000


class HfunError(ValueError):
    """An `hfun.py` that does not meet the contract."""


@dataclass
class Hfun:
    """A loaded `hfun.py`, validated far enough to be worth drawing."""

    path: Path
    hfun_min: float
    _get_hfun: object

    @classmethod
    def load(cls, path: str | Path) -> "Hfun":
        p = resolve_path(path)
        if p.is_dir():                       # `gmpas prep hfun somedir/`
            p = p / "hfun.py"
        if not p.exists():
            raise FileNotFoundError(f"No such hfun file: {p}")

        module = _import_isolated(p)

        if not hasattr(module, "get_hfun"):
            raise HfunError(
                f"{p.name} defines no get_hfun(lon, lat). A JIGSAW hfun file "
                f"must define get_hfun and the module-level float hfun_min."
            )
        if not hasattr(module, "hfun_min"):
            raise HfunError(
                f"{p.name} defines no hfun_min. It is the minimum grid "
                f"distance in km, and create_hfun.py sizes its lat-lon grid "
                f"from it."
            )
        try:
            hfun_min = float(module.hfun_min)
        except (TypeError, ValueError):
            raise HfunError(
                f"{p.name}: hfun_min is {module.hfun_min!r}, not a number"
            ) from None
        if not np.isfinite(hfun_min) or hfun_min <= 0.0:
            raise HfunError(
                f"{p.name}: hfun_min is {hfun_min}, which is not a positive "
                f"grid distance in km"
            )

        self = cls(path=p, hfun_min=hfun_min, _get_hfun=module.get_hfun)
        self._probe()
        return self

    # -- sampling --------------------------------------------------------

    def _probe(self) -> None:
        """Call it once on four points, so a broken file fails here.

        Better a message naming the file than a traceback out of the middle of
        a render, or a blank page with the error in a JSON body.
        """
        lon = np.radians(np.array([[0.0, 90.0], [180.0, 270.0]]))
        lat = np.radians(np.array([[0.0, 45.0], [-45.0, 10.0]]))
        try:
            out = self.sample_radians(lon, lat)
        except Exception as exc:
            raise HfunError(
                f"{self.path.name}: get_hfun raised {type(exc).__name__}: {exc}"
            ) from exc
        if not np.all(np.isfinite(out)):
            raise HfunError(
                f"{self.path.name}: get_hfun returned non-finite grid distances"
            )
        if np.any(out <= 0.0):
            raise HfunError(
                f"{self.path.name}: get_hfun returned a grid distance <= 0"
            )

    def sample_radians(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Grid distance (km) at each point, in ONE call to `get_hfun`.

        The tutorial's own `get_hfun` flattens its inputs and returns a 1-d
        array whatever shape it was handed -- `create_hfun.py` never notices
        because it flattens the result anyway. Reshaping here means the rest of
        this package can treat the field as the 2-d thing it is.
        """
        lon = np.asarray(lon, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        out = np.asarray(self._get_hfun(lon, lat), dtype=np.float64)
        if out.size != lon.size:
            raise HfunError(
                f"{self.path.name}: get_hfun returned {out.size} values for "
                f"{lon.size} points"
            )
        return out.reshape(lon.shape)

    def sample_degrees(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        return self.sample_radians(np.radians(lon), np.radians(lat))


def _import_isolated(path: Path):
    """Import an `hfun.py` by path, without letting it collide with ours.

    Its own directory goes on `sys.path` for the duration: a real hfun file may
    import a sibling of its own, exactly as `create_hfun.py` imports `hfun`.
    The module is given a private name and is not left in `sys.modules`, so
    loading a second hfun file in the same process gets the second file.
    """
    name = "_gmpas_user_hfun"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise HfunError(f"{path} is not importable as a Python module")

    module = importlib.util.module_from_spec(spec)
    parent = str(path.parent.resolve())
    added = parent not in sys.path
    if added:
        sys.path.insert(0, parent)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise HfunError(
            f"{path.name} failed to import: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        if added:
            try:
                sys.path.remove(parent)
            except ValueError:
                pass
    return module


# ------------------------------------------------------------- diagnostics


def analysis_grid(hfun_min: float) -> tuple[np.ndarray, np.ndarray]:
    """The lat-lon grid `create_hfun.py` would build for this `hfun_min`.

    Reproduced rather than approximated, including the `meshgrid(lats, lons)`
    argument order, so that the field analysed here is the same field JIGSAW
    is going to be handed. Coarsened only if it would be enormous, and the
    caller is told when that happens.
    """
    nlat = int(180.0 * DEG_TO_KM / hfun_min) + 1
    while nlat * 2 * nlat > MAX_ANALYSIS_POINTS:
        nlat //= 2
    lats = np.linspace(-0.5 * np.pi, 0.5 * np.pi, num=nlat, endpoint=True)
    lons = np.linspace(-np.pi, np.pi, num=2 * nlat, endpoint=True)
    return lons, lats


@dataclass
class Diagnosis:
    """What can be said about a mesh before generating it."""

    h_min: float                 # km, smallest grid distance the function asks for
    h_max: float                 # km
    max_gradient: float          # km of cell size per km of arc distance
    at_lon: float                # degrees, where that maximum is
    at_lat: float
    spacing_km: float            # analysis grid spacing at the equator
    nlon: int
    nlat: int
    coarsened: bool              # analysis grid reduced from create_hfun.py's

    @property
    def within_guideline(self) -> bool:
        return self.max_gradient <= GRADIENT_GUIDELINE


def diagnose(hfun: Hfun) -> Diagnosis:
    """Measure the resolution gradient the way `mesh_quality.py` does.

    That script reads a finished `grid.nc` and reports

        nominalDx = r_earth * nominalMinDc * (1 / meshDensity) ** 0.25
        gradient  = |nominalDx[c0] - nominalDx[c1]| / dcEdge / r_earth

    which, since `meshDensity` is `(h_min / h)**4` sampled at cell centres and
    `nominalMinDc` is `h_min`, is exactly the difference in grid distance
    between neighbouring cells divided by the distance between them. Its
    continuous limit is |grad h| with respect to arc length, and that is what
    is computed here -- so the number is comparable both to the 0.03 guideline
    and to what `mesh_quality.py` will report once the mesh exists.

    The two polar rows are excluded. Every longitude at a pole is the same
    point, so the zonal derivative there is 0/0 rather than a real gradient.
    """
    lons, lats = analysis_grid(hfun.hfun_min)
    latgrid, longrid = np.meshgrid(lats, lons)      # (nlon, nlat), as create_hfun
    h = hfun.sample_radians(longrid, latgrid)

    dlon = float(lons[1] - lons[0])
    dlat = float(lats[1] - lats[0])

    # metres of arc per radian, along each axis; the zonal one shrinks with
    # cos(lat), which is what makes a degree of longitude worth less near a pole
    dh_dy = np.gradient(h, axis=1) / (dlat * R_EARTH_KM)
    dh_dlon = np.gradient(h, axis=0) / dlon

    # Longitude is periodic, so the ends of the array are not a boundary and
    # `np.gradient`'s one-sided difference there is simply wrong -- it is first
    # order where the interior is second, and the error shows up as a spurious
    # maximum sitting exactly on the antimeridian. The grid runs -pi to +pi
    # inclusive, so its first and last columns are the same meridian and the
    # true neighbours of column 0 are columns 1 and -2.
    dh_dlon[0] = (h[1] - h[-2]) / (2.0 * dlon)
    dh_dlon[-1] = dh_dlon[0]

    dh_dx = dh_dlon / (R_EARTH_KM * np.cos(latgrid))

    grad = np.hypot(dh_dx, dh_dy)[:, 1:-1]          # drop the polar rows
    flat = int(np.argmax(grad))
    i, j = np.unravel_index(flat, grad.shape)

    nlat = lats.size
    return Diagnosis(
        h_min=float(h.min()),
        h_max=float(h.max()),
        max_gradient=float(grad[i, j]),
        at_lon=float(np.degrees(lons[i])),
        at_lat=float(np.degrees(lats[j + 1])),
        spacing_km=float(dlat * R_EARTH_KM),
        nlon=int(lons.size),
        nlat=int(nlat),
        coarsened=nlat < int(180.0 * DEG_TO_KM / hfun.hfun_min) + 1,
    )
