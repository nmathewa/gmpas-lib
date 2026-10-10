"""Projections on the lat/lon (generic) viewer's fast map.

Each pixel of a projected view is inverse-projected to lon/lat and takes the
grid cell it falls in -- the nearest-along-each-axis rule the lon/lat raster
already uses -- so a projected frame holds the file's own values, nothing
interpolated. Pixels off the map, or beyond a regional grid, are blank.
"""

from __future__ import annotations

import io
import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest
import xarray as xr

from gmpas import projection as P
from gmpas.generic import GenericViewer, _nearest_along

pytest.importorskip("cartopy")


def _grid(path, lat, lon, flip_lat=False):
    """A field whose value names its own cell: row * 1000 + col (ascending)."""
    rows, cols = np.meshgrid(np.arange(lat.size), np.arange(lon.size), indexing="ij")
    v = (rows * 1000 + cols).astype("f8")
    if flip_lat:
        lat, v = lat[::-1], v[::-1]
    xr.Dataset({"cell": (("lat", "lon"), v), "two": (("lat", "lon"), 2 * v)},
               coords={"lat": lat, "lon": lon}).to_netcdf(path)
    return GenericViewer(path, strict=True)


@pytest.fixture
def glob(tmp_path):
    return _grid(tmp_path / "g.nc", np.linspace(-87.5, 87.5, 36),
                 np.arange(0.0, 360.0, 5.0), flip_lat=True)


@pytest.fixture
def regional(tmp_path):
    return _grid(tmp_path / "r.nc", np.linspace(-10, 10, 21), np.linspace(140, 160, 21))


def _expected(gv, crs, extent, nx, ny):
    lon, lat, on = P.projected_lonlat(crs, extent, nx, ny)
    c, cin = _nearest_along(gv.lon, np.where(on, lon, 0), gv.cyclic)
    r, rin = _nearest_along(gv.lat, np.where(on, lat, 0), False)
    want = np.where(on & cin & rin, r * 1000.0 + c, np.nan)
    return want.reshape(ny, nx)


@pytest.mark.parametrize("name, params", [
    ("Orthographic", {"central_longitude": 178.0, "central_latitude": 30.0}),
    ("Robinson", {}),
    ("NorthPolarStereo", {}),
])
def test_a_projected_frame_holds_the_cell_under_each_pixel(glob, name, params):
    proj = (name, params)
    crs = P.make_crs(*proj)
    extent = P.projected_extent(crs, (0, 360, -60 if "Polar" not in name else 0, 90))
    img, on = glob._raster_projected("cell", 0, 0, extent, 90, 70, proj)
    want = _expected(glob, crs, extent, 90, 70)
    assert np.array_equal(np.isnan(img), np.isnan(want))
    assert np.array_equal(img[~np.isnan(img)], want[~np.isnan(want)])
    assert np.array_equal(on, ~np.isnan(want))


def test_the_seam_of_a_global_grid_leaves_no_gap(glob):
    proj = ("Orthographic", {"central_longitude": 180.0})
    a = P.make_crs(*proj).proj4_params["a"]
    img, on = glob._raster_projected("cell", 0, 0, (-a, a, -a, a), 80, 80, proj)
    lon, lat, disc = P.projected_lonlat(P.make_crs(*proj), (-a, a, -a, a), 80, 80)
    assert np.array_equal(on.ravel(), disc)          # every pixel on the globe has a cell
    assert np.isfinite(img.ravel()[disc]).all()


def test_beyond_a_regional_grid_is_blank(regional):
    h = regional.projected_home("Orthographic")
    proj = ("Orthographic", {"central_longitude": h["plon"], "central_latitude": h["plat"]})
    x0, x1, y0, y1 = h["extent"]
    wide = (3 * x0, 3 * x1, 3 * y0, 3 * y1)
    img, on = regional._raster_projected("cell", 0, 0, wide, 60, 60, proj)
    assert on.any() and not on.all()
    assert np.isnan(img[~on]).all() and np.isfinite(img[on]).all()


def test_a_derived_field_draws_projected(glob):
    proj = ("Robinson", {})
    ext = P.projected_extent(P.make_crs(*proj), (0, 360, -80, 80))
    a, _ = glob._raster_projected("cell", 0, 0, ext, 40, 30, proj)
    b, _ = glob._raster_projected("two - cell", 0, 0, ext, 40, 30, proj)
    assert np.array_equal(a, b, equal_nan=True)


def test_a_missing_colour_never_paints_off_the_map(glob):
    from PIL import Image

    proj = ("Orthographic", {})
    a = P.make_crs(*proj).proj4_params["a"]
    png, _, _ = glob.frame("cell", 0, 0, (-a, a, -a, a), "viridis", None, None, 50, 50,
                           colour={"missing_color": "#ff0000"}, proj=proj)
    rgba = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))
    assert rgba[0, 0, 3] == 0 and rgba[25, 25, 3] == 255


def test_without_a_projection_frames_are_exactly_as_before(glob):
    a = glob.frame("cell", 0, 0, glob.home, "viridis", None, None, 60, 40)
    b = glob.frame("cell", 0, 0, glob.home, "viridis", None, None, 60, 40, proj=None)
    assert a[0] == b[0]


def test_the_home_view_of_a_global_grid_is_the_whole_globe(glob):
    h = glob.projected_home("Orthographic")
    a = P.make_crs("Orthographic", {}).proj4_params["a"]
    x0, x1, y0, y1 = h["extent"]
    assert x1 - x0 == pytest.approx(2 * a, rel=0.02)
    assert glob.projected_home("auto")["proj"] == "PlateCarree"


def test_auto_picks_a_projection_for_a_regional_grid(regional):
    assert regional.projected_home("auto")["proj"] != "PlateCarree"


def test_the_routes_draw_a_projected_lat_lon_frame(glob):
    from PIL import Image

    from gmpas.viewer import PAGE, _handler, bind

    srv = bind(_handler(glob, PAGE), 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}/"
    try:
        meta = json.load(urllib.request.urlopen(base + "api/meta"))
        assert "Orthographic" in meta["projections"]
        h = json.load(urllib.request.urlopen(base + "api/projview?proj=Orthographic"))
        q = (f"extent={','.join(map(str, h['extent']))}&nx=64&ny=48"
             f"&proj=Orthographic&plon={h['plon']}&plat={h['plat']}")
        frame = urllib.request.urlopen(f"{base}api/frame?var=cell&time=0&level=0&{q}")
        assert Image.open(io.BytesIO(frame.read())).size == (64, 48)
        over = urllib.request.urlopen(f"{base}api/overlay?{q}")
        assert Image.open(io.BytesIO(over.read())).size == (64, 48)
        u = json.load(urllib.request.urlopen(
            base + f"api/unproject?proj=Orthographic&plon={h['plon']}"
                   f"&plat={h['plat']}&x=0&y=0"))
        assert u["on"] and u["lon"] == pytest.approx(h["plon"])
    finally:
        srv.shutdown()
