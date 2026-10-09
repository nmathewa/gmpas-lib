"""Projections: choosing one from a box, and drawing the native raster in one.

The projected raster must still be exact cells -- the same bounded KD-tree
query on different pixel centres -- and its coastlines must land on the same
pixels the raster puts a place on.
"""

from __future__ import annotations

import io
import json
import threading
import urllib.request

import numpy as np
import pytest

from gmpas import projection as P

# ------------------------------------------------------------- the auto rule


@pytest.mark.parametrize("extent, coverage, name", [
    ((-180, 180, -90, 90), None, "PlateCarree"),            # the globe
    ((0, 360, -60, 60), None, "PlateCarree"),               # f = 0.87
    ((-180, 180, 60, 90), None, "NorthPolarStereo"),        # cap reaching the pole
    ((-30, 30, 62, 80), None, "NorthPolarStereo"),          # wholly poleward of 60
    ((100, 160, -85, -65), None, "SouthPolarStereo"),
    ((0, 120, -40, 40), None, "Orthographic"),              # 1/6 <= f < 2/3
    ((140, 155, -10, 5), None, "Mercator"),                 # tropical regional
    ((140, 155, -10, 35), None, "PlateCarree"),             # tall tropical box: neither
    ((-10, 30, 35, 60), None, "LambertConformal"),          # Europe
    ((170, -160, 30, 50), None, "LambertConformal"),        # across the antimeridian
    ((-20, 20, 72, 78), None, "NorthPolarStereo"),          # regional, centre >= 70
    ((140, 155, -10, 5), 0.9, "PlateCarree"),               # a mesh's coverage wins
])
def test_the_auto_rule(extent, coverage, name):
    assert P.auto_projection(extent, coverage)[0] == name


def test_lambert_parallels_sit_at_a_sixth_and_five_sixths_of_the_span():
    name, params = P.auto_projection((-10, 30, 30, 60))
    assert name == "LambertConformal"
    assert params["standard_parallels"] == pytest.approx((35.0, 55.0))
    assert params["central_latitude"] == pytest.approx(45.0)


def test_lambert_parallels_never_straddle_the_equator():
    """A box from 10S to 45N is mid-latitude by its centre (17.5N), but its
    1/6 parallel falls south of the equator: a degenerate cone."""
    name, params = P.auto_projection((100, 130, -10, 45))
    assert name == "LambertConformal"
    assert min(params["standard_parallels"]) > 0
    with pytest.raises(ValueError, match="one side of the equator"):
        P.make_crs("LambertConformal", {"standard_parallels": (-20, 10)})


def test_an_antimeridian_box_is_centred_on_its_own_middle():
    _name, params = P.auto_projection((170, -170, 30, 50))
    assert params["central_longitude"] == pytest.approx(-180.0) or \
        params["central_longitude"] == pytest.approx(180.0)


def test_coverage_of_a_box():
    assert P.box_coverage((-180, 180, -90, 90)) == pytest.approx(1.0)
    assert P.box_coverage((0, 90, 0, 90)) == pytest.approx(0.125)


# ------------------------------------------------------- projected pixels


def test_the_closed_form_orthographic_inverse_matches_pyproj():
    import cartopy.crs as ccrs

    crs = P.make_crs("Orthographic", {"central_longitude": 30, "central_latitude": 40})
    rng = np.random.default_rng(0)
    x, y = rng.uniform(-5e6, 5e6, 500), rng.uniform(-5e6, 5e6, 500)
    lon, lat, on = P._ortho_inverse(crs, x, y)
    ref = ccrs.PlateCarree().transform_points(crs, x, y)
    assert (on == np.isfinite(ref[:, 1])).all()        # the same disc
    lon, lat, ref = lon[on], lat[on], ref[on]
    assert np.allclose(lat, ref[:, 1], atol=1e-7)
    assert np.allclose(((lon - ref[:, 0] + 180) % 360) - 180, 0.0, atol=1e-7)


def test_a_pixel_off_the_map_outline_is_not_a_cell():
    """pyproj's EqualEarth inverse returns finite, wrong coordinates outside
    the outline: (17e6, 8.3e6) m comes back as (-66.7, 83.0), which projects
    forward 21,000 km away. The round trip has to catch it."""
    crs = P.make_crs("EqualEarth")
    _pts, on = P.projected_points(crs, (16.9e6, 17.1e6, 8.2e6, 8.4e6), 2, 2)
    assert not on.any()
    _pts, on = P.projected_points(crs, (-1e6, 1e6, -1e6, 1e6), 4, 4)
    assert on.all()


def test_points_outside_the_orthographic_disc_are_off_the_map():
    crs = P.make_crs("Orthographic")
    a = crs.proj4_params["a"]
    _pts, on = P.projected_points(crs, (-1.2 * a, 1.2 * a, -1.2 * a, 1.2 * a), 60, 60)
    frac = on.mean()
    assert frac == pytest.approx(np.pi / (4 * 1.2 ** 2), abs=0.03)   # the disc's area


# --------------------------------------------------------- the viewer path


@pytest.fixture
def viewer(tmp_path):
    """A global-ish mesh with a field that is 1 east of 10E and 0 west of it."""
    import xarray as xr

    from conftest import write_mesh
    from gmpas.viewer import Viewer

    lon = np.arange(-180, 180, 4.0) + 2.0
    lat = np.arange(-88, 90, 4.0)
    lo, la = np.meshgrid(lon, lat)
    path = tmp_path / "history.2012-01-01_00.00.00.nc"
    # areas as a 4-degree Voronoi mesh has them, so the off-mesh distance
    # test (twice the radius of a cell's area) holds between centres
    km = 4.0 * 111.2e3
    areas = np.maximum(km * km * np.cos(np.radians(la.ravel())), 1e9)
    centres = list(zip(lo.ravel(), la.ravel(), strict=True))
    write_mesh(path, centres, radius_deg=2.0, areas=areas)
    with xr.open_dataset(path) as ds:
        full = ds.load()
    east = (lo.ravel() > 10.0).astype("f8")
    full["side"] = (("Time", "nCells"), east[None, :])
    full.to_netcdf(path, mode="w")
    v = Viewer(tmp_path, nx=80, ny=60)
    yield v
    v.close()


def test_without_a_projection_frames_are_exactly_as_before(viewer):
    """The new keyword must not change a byte of today's frames."""
    a = viewer.frame("side", 0, 0, viewer.home, "viridis", None, None)
    b = viewer.frame("side", 0, 0, viewer.home, "viridis", None, None, proj=None)
    assert a == b


def test_a_projected_frame_holds_exact_cells(viewer):
    """Every on-map pixel is the containing cell's own value: 0 or 1, never a
    blend, and on the correct side of 10E."""
    import cartopy.crs as ccrs

    proj = ("Orthographic", {"central_longitude": 10.0, "central_latitude": 20.0})
    crs = P.make_crs(*proj)
    a = crs.proj4_params["a"]
    extent = (-a, a, -a, a)
    view = viewer.view(extent, 100, 100, proj)
    img = view.frame(viewer.values("side", 0, 0))
    vals = img[np.isfinite(img)]
    assert set(np.unique(vals)) <= {0.0, 1.0}
    # a pixel well east and one well west of the line, both on the disc
    for lon, lat, want in ((40.0, 20.0, 1.0), (-20.0, 20.0, 0.0)):
        x, y = crs.transform_point(lon, lat, ccrs.PlateCarree())
        col = int((x - extent[0]) / (extent[1] - extent[0]) * 100)
        row = int((y - extent[2]) / (extent[3] - extent[2]) * 100)
        assert img[row, col] == want


def test_a_southern_lambert_overlay_draws_its_coasts():
    """cartopy cuts a Lambert map off at 30S by default -- fine for a northern
    cone, but a southern one then had no domain left and drew no coasts."""
    from PIL import Image

    from gmpas.viewer import _overlay

    crs = P.make_crs("LambertConformal", {"central_longitude": 145.0,
                                          "central_latitude": -20.0,
                                          "standard_parallels": (-30.0, -10.0)})
    ext = P.projected_extent(crs, (110, 180, -45, 0))        # Australia
    alpha = np.asarray(Image.open(io.BytesIO(_overlay(ext, 300, 200, crs=crs)))
                       .convert("RGBA"))[..., 3]
    assert (alpha > 0).mean() > 0.005


@pytest.mark.parametrize("proj", [
    ("Orthographic", {"central_longitude": 10.0, "central_latitude": 20.0}),
    ("NorthPolarStereo", {"central_longitude": 0.0}),
])
def test_coastlines_land_on_the_pixel_the_raster_puts_a_place_on(proj):
    """The overlay's axis maps a place to the same pixel the raster does."""
    import cartopy.crs as ccrs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from gmpas.viewer import _projected_axes

    crs = P.make_crs(*proj)
    extent = P.projected_extent(crs, (-60, 80, 30, 80))
    nx, ny = 300, 200
    fig = plt.figure(figsize=(nx / 100, ny / 100), dpi=100)
    try:
        ax = _projected_axes(fig, extent, crs)
        for lon, lat in ((10.0, 50.0), (-30.0, 60.0), (60.0, 40.0)):
            x, y = crs.transform_point(lon, lat, ccrs.PlateCarree())
            px, py = ax.transData.transform((x, y))
            # the raster's pixel for the same place, in display coordinates
            col = (x - extent[0]) / (extent[1] - extent[0]) * nx
            row = (y - extent[2]) / (extent[3] - extent[2]) * ny
            assert px == pytest.approx(col, abs=0.5)
            assert py == pytest.approx(row, abs=0.5)
    finally:
        plt.close(fig)


def test_the_api_draws_a_projected_frame_and_overlay(viewer):
    from PIL import Image

    from gmpas.viewer import PAGE, _handler, bind

    crs = P.make_crs("Orthographic", {"central_longitude": 10.0})
    a = crs.proj4_params["a"]
    srv = bind(_handler(viewer, PAGE), 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        q = f"extent={-a},{a},{-a},{a}&nx=120&ny=120&proj=Orthographic&plon=10"
        frame = urllib.request.urlopen(f"{base}/api/frame?var=side&time=0&level=0&{q}")
        assert Image.open(io.BytesIO(frame.read())).size == (120, 120)
        over = urllib.request.urlopen(f"{base}/api/overlay?{q}")
        assert Image.open(io.BytesIO(over.read())).size == (120, 120)
        try:
            bogus = q.replace("Orthographic", "Bogus")
            urllib.request.urlopen(f"{base}/api/frame?var=side&{bogus}")
            raise AssertionError("an unknown projection was accepted")
        except urllib.error.HTTPError as exc:
            assert "unknown projection" in json.loads(exc.read())["error"]
    finally:
        srv.shutdown()


def test_the_lat_lon_viewer_refuses_a_projection(tmp_path):
    import xarray as xr

    from gmpas.generic import GenericViewer
    from gmpas.viewer import _proj_params

    xr.Dataset({"v": (("lat", "lon"), np.zeros((4, 5)))},
               coords={"lat": np.linspace(-9, 9, 4), "lon": np.linspace(0, 20, 5)}
               ).to_netcdf(tmp_path / "f.nc")
    gv = GenericViewer(tmp_path / "f.nc", strict=True)
    with pytest.raises(ValueError, match="MPAS viewer only"):
        _proj_params({"proj": "Orthographic"}, gv)


# --------------------------------------------------------- layers: "auto"


def test_auto_is_an_explicit_choice_and_not_the_default():
    from gmpas import layers as L

    spec = L.FIGURE_OPTIONS["projection"]
    assert "auto" in spec["choices"] and spec["default"] == "PlateCarree"


def test_auto_resolves_to_the_rule_for_the_view():
    from gmpas import layers as L

    out = L._resolve_auto({"projection": "auto", "central_longitude": None,
                           "central_latitude": None}, (-10, 30, 30, 60))
    assert out["projection"] == "LambertConformal"
    assert out["standard_parallels"] == pytest.approx((35.0, 55.0))
    kept = L._resolve_auto({"projection": "auto", "central_longitude": 5.0,
                            "central_latitude": None}, (-10, 30, 30, 60))
    assert kept["central_longitude"] == 5.0           # the stack's own centre wins
