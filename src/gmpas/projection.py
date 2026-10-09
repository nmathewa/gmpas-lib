"""Map projections for the fast map and figures: choosing one, and drawing the
native raster in one.

Two separate things live here.

`auto_projection` picks a projection from a lon/lat box -- the rule of
Projection Wizard (Savric, Jenny & Jenny 2016) and Snyder (1987), reduced to
the projections gmpas offers. It is only ever used when asked for
(`projection: "auto"`); no default changes.

`projected_points` gives the unit vectors of every pixel centre of a view in a
projection, so the same bounded KD-tree query that draws today's lon/lat
raster draws a projected one -- still the containing Voronoi cell, nothing
interpolated. Each pixel's inverse is checked by projecting it forward again:
pyproj's EqualEarth inverse returns finite but wrong coordinates outside the
map outline, so a pixel whose round trip misses by more than half a pixel is
off the map, not a cell.
"""

from __future__ import annotations

import math

import numpy as np

#: what `make_crs` can build; the same names as `layers.PROJECTIONS`
NAMES = ("PlateCarree", "Robinson", "Mollweide", "EqualEarth", "Orthographic",
         "NorthPolarStereo", "SouthPolarStereo", "LambertConformal", "Mercator")


def _box(extent) -> tuple[float, float, float, float, float]:
    """(lon0, dlon, lat0, lat1, lonc) for a box whose longitudes may wrap."""
    lon0, lon1, lat0, lat1 = (float(v) for v in extent)
    lat0, lat1 = max(-90.0, min(lat0, lat1)), min(90.0, max(lat0, lat1))
    dlon = (lon1 - lon0) % 360.0
    if dlon == 0.0 and lon1 != lon0:      # exactly a full circle, e.g. -180..180
        dlon = 360.0
    lonc = ((lon0 + 0.5 * dlon + 180.0) % 360.0) - 180.0
    return lon0, dlon, lat0, lat1, lonc


def box_coverage(extent) -> float:
    """Fraction of the sphere a lon/lat box covers."""
    _lon0, dlon, lat0, lat1, _ = _box(extent)
    return (dlon / 360.0) * (math.sin(math.radians(lat1))
                             - math.sin(math.radians(lat0))) / 2.0


def auto_projection(extent, coverage: float | None = None) -> tuple[str, dict]:
    """A projection for a lon/lat box: (name, params for `make_crs`).

    `coverage`, the fraction of the sphere the data covers (an MPAS mesh's
    `coverage`), stands in for the box's own when given.

    - at least 2/3 of the sphere: PlateCarree, the map as it is today;
    - a box reaching a pole, or wholly poleward of 60 degrees: polar
      stereographic on that side;
    - 1/6 to 2/3: orthographic, centred on the box;
    - smaller: Mercator near the equator (centre within 15, box within 30),
      Lambert conformal at mid latitudes (standard parallels at 1/6 and 5/6
      of the latitude span), polar stereographic from 70 degrees; a tall
      tropical box that fits none of these stays PlateCarree.
    """
    _lon0, _dlon, lat0, lat1, lonc = _box(extent)
    f = box_coverage(extent) if coverage is None else float(coverage)
    latc = 0.5 * (lat0 + lat1)

    if f >= 2.0 / 3.0:
        return "PlateCarree", {}
    if lat1 >= 90.0 - 1e-9 or lat0 >= 60.0:
        return "NorthPolarStereo", {"central_longitude": lonc}
    if lat0 <= -90.0 + 1e-9 or lat1 <= -60.0:
        return "SouthPolarStereo", {"central_longitude": lonc}
    if f >= 1.0 / 6.0:
        return "Orthographic", {"central_longitude": lonc, "central_latitude": latc}
    if abs(latc) >= 70.0:
        return ("NorthPolarStereo" if latc > 0 else "SouthPolarStereo",
                {"central_longitude": lonc})
    if abs(latc) <= 15.0:
        if lat0 >= -30.0 and lat1 <= 30.0:
            return "Mercator", {"central_longitude": lonc}
        return "PlateCarree", {}
    span = lat1 - lat0
    sp = (lat0 + span / 6.0, lat0 + 5.0 * span / 6.0)
    # both parallels on the centre's side of the equator: a cone cut across
    # it is degenerate, and cartopy then has no valid area to draw in
    sp = tuple(max(v, 1.0) for v in sp) if latc > 0 else tuple(min(v, -1.0) for v in sp)
    return "LambertConformal", {
        "central_longitude": lonc, "central_latitude": latc, "standard_parallels": sp}


def make_crs(name: str, params: dict | None = None):
    """The cartopy CRS for a projection name and its parameters."""
    import cartopy.crs as ccrs

    if name not in NAMES:
        raise ValueError(f"unknown projection {name!r}; one of {', '.join(NAMES)}")
    p = dict(params or {})
    lon0 = float(p.get("central_longitude") or 0.0)
    if name == "PlateCarree":
        return ccrs.PlateCarree(central_longitude=lon0)
    if name == "Orthographic":
        return ccrs.Orthographic(central_longitude=lon0,
                                 central_latitude=float(p.get("central_latitude") or 0.0))
    if name == "LambertConformal":
        lat0 = float(p.get("central_latitude") or 40.0)
        sp = p.get("standard_parallels")
        if sp is None:                      # as layers._projection chooses them
            sp = (-33.0, -45.0) if lat0 < 0 else (33.0, 45.0)
        sp = tuple(float(v) for v in sp)
        if sp[0] * sp[1] <= 0:
            raise ValueError(f"Lambert conformal standard parallels {sp} must be on one "
                             f"side of the equator (and not on it)")
        # cartopy cuts the map off at `cutoff` (default -30), away from the
        # cone's pole; a southern cone needs the mirror, or everything north
        # of 30S -- the whole map -- falls outside its domain
        return ccrs.LambertConformal(central_longitude=lon0, central_latitude=lat0,
                                     standard_parallels=sp,
                                     cutoff=30.0 if sp[0] < 0 else -30.0)
    return getattr(ccrs, name)(central_longitude=lon0)


def projected_extent(crs, extent, samples: int = 64):
    """The projected (x0, x1, y0, y1) that holds a lon/lat box.

    Built from points along the box's edges and through it, so a box whose
    corners fall off the map (a hemisphere in orthographic) still gets the
    part that is on it.
    """
    import cartopy.crs as ccrs

    lon0, dlon, lat0, lat1, _ = _box(extent)
    lon = lon0 + dlon * np.linspace(0.0, 1.0, samples)
    lat = np.linspace(lat0, lat1, samples)
    lo, la = np.meshgrid(lon, lat)
    xy = crs.transform_points(ccrs.PlateCarree(), lo.ravel(), la.ravel())[:, :2]
    xy = xy[np.isfinite(xy).all(axis=1)]
    if not xy.size:
        raise ValueError("no part of the box is on this projection")
    return (float(xy[:, 0].min()), float(xy[:, 0].max()),
            float(xy[:, 1].min()), float(xy[:, 1].max()))


def _ortho_inverse(crs, x, y):
    """Closed-form inverse orthographic on the CRS's sphere: (lon, lat, on_disc)."""
    p = crs.proj4_params
    a = float(p.get("a", 6378137.0))
    lam0, phi0 = math.radians(p.get("lon_0", 0.0)), math.radians(p.get("lat_0", 0.0))
    rho = np.hypot(x, y)
    on = rho <= a
    c = np.arcsin(np.clip(rho / a, 0.0, 1.0))
    sin_c, cos_c = np.sin(c), np.cos(c)
    with np.errstate(invalid="ignore", divide="ignore"):
        lat = np.arcsin(np.clip(cos_c * math.sin(phi0)
                                + np.where(rho > 0, y * sin_c * math.cos(phi0) / rho, 0.0),
                                -1.0, 1.0))
        lon = lam0 + np.arctan2(x * sin_c,
                                rho * cos_c * math.cos(phi0) - y * sin_c * math.sin(phi0))
    return np.degrees(lon), np.degrees(lat), on


def projected_points(crs, xy_extent, nx: int, ny: int):
    """Unit vectors for every pixel centre of a projected view, and which are on
    the map: ((ny*nx, 3) array, (ny*nx,) bool). Row 0 is the bottom row, as in
    `raster.grid_points`."""
    import cartopy.crs as ccrs

    x0, x1, y0, y1 = (float(v) for v in xy_extent)
    dx, dy = (x1 - x0) / nx, (y1 - y0) / ny
    xs = x0 + dx * (np.arange(nx) + 0.5)
    ys = y0 + dy * (np.arange(ny) + 0.5)
    X, Y = np.meshgrid(xs, ys)
    X, Y = X.ravel(), Y.ravel()

    if crs.proj4_params.get("proj") == "ortho":
        lon, lat, on = _ortho_inverse(crs, X, Y)
    else:
        ll = ccrs.PlateCarree().transform_points(crs, X, Y)
        lon, lat = ll[:, 0], ll[:, 1]
        on = np.isfinite(lon) & np.isfinite(lat)
        # the forward round trip: an inverse can be finite and still wrong
        back = crs.transform_points(ccrs.PlateCarree(), np.where(on, lon, 0.0),
                                    np.where(on, lat, 0.0))[:, :2]
        miss = np.hypot(back[:, 0] - X, back[:, 1] - Y)
        on &= np.isfinite(miss) & (miss <= 0.5 * max(abs(dx), abs(dy)))

    lon_r = np.radians(np.where(on, lon, 0.0))
    lat_r = np.radians(np.where(on, lat, 0.0))
    cl = np.cos(lat_r)
    pts = np.stack([cl * np.cos(lon_r), cl * np.sin(lon_r), np.sin(lat_r)], axis=1)
    return pts, on
