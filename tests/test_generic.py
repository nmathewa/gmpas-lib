"""`gmpas view --generic`: a plain netCDF file on a regular lat/lon grid."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from gmpas.generic import GenericViewer

LAT = np.linspace(-89.5, 89.5, 20)
LON = np.linspace(0.5, 359.5, 40)
TIMES = pd.date_range("2024-01-01", periods=3, freq="6h")


def _open(tmp_path, ds, name="f.nc"):
    """Open strictly: an unreadable grid raises, as it does for `gmpas plot`
    and the library. The viewer's own lenient path is `_open_asking`."""
    path = tmp_path / name
    ds.to_netcdf(path)
    return GenericViewer(path, strict=True)


def _open_asking(tmp_path, ds, name="f.nc"):
    """Open the way `gmpas view --generic` does: a grid it cannot work out
    leaves the viewer up and asking, rather than killing the process."""
    path = tmp_path / name
    ds.to_netcdf(path)
    return GenericViewer(path)


def _row(gv, var):
    return next(r for r in gv.describe()["variables"] if r["name"] == var)


# ------------------------------------------------------------ finding lat/lon


def test_the_conventional_names_are_found(tmp_path):
    gv = _open(tmp_path, xr.Dataset({"v": (("latitude", "longitude"), np.zeros((20, 40)))},
                                    coords={"latitude": LAT, "longitude": LON}))
    assert (gv.lat_name, gv.lon_name) == ("latitude", "longitude")


def test_cf_attributes_find_axes_whatever_they_are_called(tmp_path):
    ds = xr.Dataset({"v": (("a", "b"), np.zeros((20, 40)))}, coords={
        "a": ("a", LAT, {"units": "degree_N"}),
        "b": ("b", LON, {"standard_name": "longitude"}),
    })
    gv = _open(tmp_path, ds)
    assert (gv.lat_name, gv.lon_name) == ("a", "b")


def test_a_1d_coordinate_over_a_differently_named_dimension(tmp_path):
    """nav_lat(y): slicing has to use `y`, not the coordinate's own name."""
    vals = np.arange(20 * 40, dtype="f4").reshape(20, 40)
    ds = xr.Dataset({"v": (("y", "x"), vals),
                     "nav_lat": (("y",), LAT), "nav_lon": (("x",), LON)})
    gv = _open(tmp_path, ds)

    assert (gv.lat_name, gv.lat_dim, gv.lon_dim) == ("nav_lat", "y", "x")
    assert gv._slice("v", 0, 0, gv.home) == pytest.approx(vals)


def test_attributes_outrank_a_bare_name(tmp_path):
    ds = xr.Dataset({"v": (("y", "x"), np.zeros((20, 40)))}, coords={
        "lat": ("q", np.arange(7.0)),                           # named lat, is not
        "y": ("y", LAT, {"standard_name": "latitude"}),
        "x": ("x", LON, {"units": "degrees_east"}),
    })
    assert _open(tmp_path, ds).lat_name == "y"


@pytest.mark.parametrize("std, units", [
    ("projection_y_coordinate", "m"),
    ("grid_latitude", "degrees"),                               # rotated pole
])
def test_projected_and_rotated_axes_are_not_taken_as_lat_lon(tmp_path, std, units):
    xstd = std.replace("y_", "x_").replace("latitude", "longitude")
    ds = xr.Dataset({"v": (("y", "x"), np.zeros((20, 40)))}, coords={
        "y": ("y", LAT, {"standard_name": std, "units": units, "axis": "Y"}),
        "x": ("x", LON, {"standard_name": xstd, "units": units, "axis": "X"}),
    })
    with pytest.raises(ValueError, match="no lat coordinate"):
        _open(tmp_path, ds)


def test_a_curvilinear_grid_is_refused_by_name(tmp_path):
    ds = xr.Dataset({
        "v": (("sn", "we"), np.zeros((20, 40))),
        "XLAT": (("sn", "we"), np.repeat(LAT[:, None], 40, 1), {"units": "degree_north"}),
        "XLONG": (("sn", "we"), np.repeat(LON[None, :], 20, 0), {"units": "degree_east"}),
    })
    with pytest.raises(ValueError, match="'XLAT'.*curvilinear"):
        _open(tmp_path, ds)


def test_a_strong_2d_coordinate_beats_a_weak_1d_one(tmp_path):
    """NEMO's shape: correct 2D nav_lat/nav_lon beside bare index axes that
    carry nothing but `axis: Y`/`axis: X`.

    This asserts the *extent*, not the message. Filtering to 1D candidates
    before ranking drew this global grid as an 11-degree box off West Africa
    -- a silently wrong map, which is worse than any refusal.
    """
    ds = xr.Dataset({
        "thetao": (("y", "x"), np.zeros((20, 40))),
        "nav_lat": (("y", "x"), np.repeat(LAT[:, None], 40, 1),
                    {"standard_name": "latitude", "units": "degrees_north"}),
        "nav_lon": (("y", "x"), np.repeat(LON[None, :], 20, 0),
                    {"standard_name": "longitude", "units": "degrees_east"}),
    }, coords={"y": ("y", np.arange(20.0), {"axis": "Y"}),
               "x": ("x", np.arange(40.0), {"axis": "X"})})
    with pytest.raises(ValueError, match="'nav_lat'.*curvilinear"):
        _open(tmp_path, ds)


def test_a_rotated_pole_without_standard_name_is_not_drawn_as_degrees(tmp_path):
    """CORDEX with an incomplete attribute set: `rlat` has `axis: Y` and
    `units: degrees` but no `standard_name`, so `_NOT_GEOGRAPHIC` cannot veto
    it -- while the real 2D lat/lon sit beside it, correctly attributed.

    Central Europe came out over the Gulf of Guinea, about 45 degrees of
    latitude wrong. Asserting on numbers because that is what was wrong.
    """
    rlat = np.linspace(-11.0, 11.0, 20)          # rotated coordinates
    rlon = np.linspace(-14.0, 14.0, 40)
    true_lat = np.repeat(np.linspace(32.6, 57.4, 20)[:, None], 40, 1)
    true_lon = np.repeat(np.linspace(-4.55, 24.55, 40)[None, :], 20, 0)
    ds = xr.Dataset({
        "tas": (("rlat", "rlon"), np.zeros((20, 40))),
        "lat": (("rlat", "rlon"), true_lat,
                {"standard_name": "latitude", "units": "degrees_north"}),
        "lon": (("rlat", "rlon"), true_lon,
                {"standard_name": "longitude", "units": "degrees_east"}),
    }, coords={"rlat": ("rlat", rlat, {"units": "degrees", "axis": "Y"}),
               "rlon": ("rlon", rlon, {"units": "degrees", "axis": "X"})})
    with pytest.raises(ValueError) as caught:
        gv = _open(tmp_path, ds)
        raise AssertionError(
            f"drew a map at {gv.home}; the file covers 32.6..57.4 N")
    assert "curvilinear" in str(caught.value)


def test_a_refusal_names_the_file_it_read(tmp_path):
    """One of three thousand files is files[0]; the message has to say which."""
    ds = xr.Dataset({"v": (("a", "b"), np.zeros((4, 5)))},
                    coords={"a": np.arange(4.0), "b": np.arange(5.0)})
    with pytest.raises(ValueError, match="odd_run.nc"):
        _open(tmp_path, ds, name="odd_run.nc")


def test_points_sharing_one_dimension_are_not_a_grid(tmp_path):
    ds = xr.Dataset({"v": (("n",), np.zeros(10))},
                    coords={"lat": ("n", np.linspace(-9, 9, 10)),
                            "lon": ("n", np.linspace(0, 9, 10))})
    with pytest.raises(ValueError, match="share the dimension"):
        _open(tmp_path, ds)


def test_a_latitude_past_90_is_caught(tmp_path):
    ds = xr.Dataset({"v": (("lat", "lon"), np.zeros((20, 40)))},
                    coords={"lat": LAT * 1000, "lon": LON})
    with pytest.raises(ValueError, match="cannot exceed 90"):
        _open(tmp_path, ds)


# ------------------------------------------------------------------- time


@pytest.mark.parametrize("coords, labels", [
    ({"time": TIMES}, None),
    ({}, ["step 0", "step 1", "step 2"]),
    ({"time": [0, 6, 12]}, ["0", "6", "12"]),
    ({"time": ("time", [0, 6, 12], {"units": "hours"})},
     ["0 hours", "6 hours", "12 hours"]),
])
def test_time_is_found_with_or_without_a_decodable_coordinate(tmp_path, coords, labels):
    vals = np.arange(3 * 20 * 40, dtype="f4").reshape(3, 20, 40)
    ds = xr.Dataset({"v": (("time", "lat", "lon"), vals)},
                    coords={"lat": LAT, "lon": LON, **coords})
    gv = _open(tmp_path, ds)

    assert gv.time_name == "time" and gv.steps == 3
    if labels:
        assert gv.labels == labels
    assert gv._slice("v", 2, 0, gv.home) == pytest.approx(vals[2])


# --------------------------------------------------------- extra axes


def test_a_pressure_axis_is_the_level_slider(tmp_path):
    vals = np.arange(3 * 5 * 20 * 40, dtype="f4").reshape(3, 5, 20, 40)
    ds = xr.Dataset({"t": (("time", "plev", "lat", "lon"), vals)}, coords={
        "lat": LAT, "lon": LON, "time": TIMES,
        "plev": ("plev", [1000.0, 850, 700, 500, 250], {"units": "hPa"}),
    })
    gv = _open(tmp_path, ds)

    row = _row(gv, "t")
    assert (row["levels"], row["dim"], row["pinned"]) == (5, "plev", [])
    assert gv._slice("t", 1, 3, gv.home) == pytest.approx(vals[1, 3])
    png, _, _ = gv.frame("t", 1, 3, gv.home, "viridis", None, None, 40, 20)
    assert png[:4] == b"\x89PNG"


def test_a_marked_vertical_axis_takes_the_slider_over_an_earlier_one(tmp_path):
    vals = np.arange(3 * 4 * 20 * 40, dtype="f4").reshape(3, 4, 20, 40)
    ds = xr.Dataset({"t": (("member", "lev", "lat", "lon"), vals)}, coords={
        "lat": LAT, "lon": LON, "lev": ("lev", np.arange(4.0), {"positive": "down"}),
    })
    gv = _open(tmp_path, ds)

    row = _row(gv, "t")
    assert (row["dim"], row["pinned"]) == ("lev", ["member"])
    assert gv._slice("t", 0, 2, gv.home) == pytest.approx(vals[0, 2])   # member held at 0


def test_an_unmarked_extra_axis_still_slices(tmp_path):
    vals = np.arange(3 * 20 * 40, dtype="f4").reshape(3, 20, 40)
    ds = xr.Dataset({"t": (("member", "lat", "lon"), vals)},
                    coords={"lat": LAT, "lon": LON})
    gv = _open(tmp_path, ds)

    assert _row(gv, "t")["levels"] == 3
    assert gv._slice("t", 0, 2, gv.home) == pytest.approx(vals[2])


def test_the_probe_reads_the_chosen_level(tmp_path):
    vals = (np.arange(5)[:, None, None] * 10.0 + np.zeros((5, 20, 40))).astype("f4")
    ds = xr.Dataset({"t": (("plev", "lat", "lon"), vals)},
                    coords={"lat": LAT, "lon": LON, "plev": [1000.0, 850, 700, 500, 250]})
    gv = _open(tmp_path, ds)
    assert gv.probe(10.0, 45.0, "t", 0, 3)["value"] == pytest.approx(30.0)


def test_descending_axes_come_back_ascending(tmp_path):
    vals = np.arange(20 * 40, dtype="f4").reshape(20, 40)
    ds = xr.Dataset({"v": (("lat", "lon"), vals)},
                    coords={"lat": LAT[::-1], "lon": LON[::-1]})
    gv = _open(tmp_path, ds)
    assert gv._slice("v", 0, 0, gv.home) == pytest.approx(vals[::-1, ::-1])


# ------------------------------------------------- non-spatial variables


@pytest.mark.parametrize("dims, shape", [
    (("time",), (3,)),                  # xarray: line
    (("time", "plev"), (3, 5)),         # xarray: pcolormesh
    (("time", "plev", "band"), (3, 5, 2)),  # xarray: histogram
    ((), ()),                           # a single number
])
def test_a_variable_off_the_map_is_drawn_the_way_xarray_would(tmp_path, dims, shape):
    import matplotlib
    matplotlib.use("Agg")

    ds = xr.Dataset({
        "map": (("lat", "lon"), np.zeros((20, 40))),
        "side": (dims, np.ones(shape, "f4")),
    }, coords={"lat": LAT, "lon": LON, "time": TIMES})
    gv = _open(tmp_path, ds)

    png, _, _ = gv.frame("side", 0, 0, gv.home, "viridis", None, None, 300, 200)
    assert png[:4] == b"\x89PNG"


# ------------------------------------------------------------ many files


def _write_months(tmp_path, months=("01", "02", "03"), steps=2):
    """ERA5-style monthly files, each value encoding (month, step, level)."""
    lat, lon = np.arange(90, -91, -10.0), np.arange(0, 360, 10.0)
    plev = np.array([1000.0, 500.0])
    for m, month in enumerate(months):
        vals = (100 * m + 10 * np.arange(steps)[:, None, None, None]
                + np.arange(2)[None, :, None, None]
                + np.zeros((steps, 2, lat.size, lon.size))).astype("f4")
        xr.Dataset(
            {"t": (("valid_time", "pressure_level", "latitude", "longitude"), vals,
                   {"units": "K"}),
             "gmean": (("valid_time",), vals.mean(axis=(1, 2, 3)))},
            coords={"valid_time": pd.date_range(f"2024-{month}-01", periods=steps,
                                                freq="12h"),
                    "pressure_level": ("pressure_level", plev, {"units": "hPa"}),
                    "latitude": ("latitude", lat, {"units": "degrees_north"}),
                    "longitude": ("longitude", lon, {"units": "degrees_east"})},
        ).to_netcdf(tmp_path / f"era5_2024-{month}.nc")
    return tmp_path


@pytest.mark.parametrize("how", ["directory", "glob", "list"])
def test_many_files_become_one_time_axis(tmp_path, how):
    d = _write_months(tmp_path)
    arg = {"directory": d, "glob": str(d / "era5_*.nc"),
           "list": [str(d / f"era5_2024-{m}.nc") for m in ("03", "01", "02")]}[how]
    gv = GenericViewer(arg)

    assert [f.name for f in gv.files] == [f"era5_2024-0{m}.nc" for m in (1, 2, 3)]
    assert gv.steps == 6 and len(gv) == 6
    assert gv.labels[:3] == ["2024-01-01 00:00", "2024-01-01 12:00", "2024-02-01 00:00"]
    # step 3 is February's second step; level 1 is 500 hPa
    assert gv._slice("t", 3, 1, gv.home) == pytest.approx(np.full((19, 36), 111.0))


def test_a_file_on_another_grid_is_left_out_by_name(tmp_path, capsys):
    d = _write_months(tmp_path)
    xr.Dataset({"t": (("latitude", "longitude"), np.zeros((5, 5)))},
               coords={"latitude": np.linspace(-9, 9, 5),
                       "longitude": np.linspace(0, 9, 5)}).to_netcdf(d / "era5_2024-04.nc")
    gv = GenericViewer(d)

    assert gv.steps == 6
    assert "era5_2024-04.nc" not in [f.name for f in gv.files]
    assert "era5_2024-04.nc" in capsys.readouterr().err


def test_the_background_scan_starts_small_and_completes(tmp_path):
    import time

    gv = GenericViewer(_write_months(tmp_path), background_scan=True)
    deadline = time.time() + 10
    while gv.scanning and time.time() < deadline:
        time.sleep(0.02)
    assert not gv.scanning and gv.steps == 6


def test_files_without_a_time_axis_are_one_step_each(tmp_path):
    for day in ("2024-01-01_00.00.00", "2024-01-02_00.00.00"):
        xr.Dataset({"v": (("lat", "lon"), np.zeros((20, 40)))},
                   coords={"lat": LAT, "lon": LON}).to_netcdf(tmp_path / f"snap.{day}.nc")
    gv = GenericViewer(tmp_path)
    assert gv.steps == 2 and gv.labels == ["2024-01-01 00:00", "2024-01-02 00:00"]


def _write_changeover(tmp_path, later="valid_time"):
    """Four files of two steps: the first two step along `time`, the last two
    along `later` -- ERA5/CDS's `valid_time`, or a noleap `record` axis that
    only its units say is time."""
    truth = []
    for k, (tname, day, base) in enumerate(
            [("time", "2022-12-31", 290), ("time", "2023-01-01", 290),
             (later, "2023-01-02", 340), (later, "2023-01-03", 340)]):
        vals = np.stack([np.full((20, 40), base + 10.0 * s) for s in range(2)])
        truth += [base, base + 10]
        t = pd.date_range(day, periods=2, freq="12h")
        if tname == "record":
            t = ("record", [0.0, 0.5], {"units": f"days since {day}", "calendar": "noleap"})
        xr.Dataset({"t2m": ((tname, "lat", "lon"), vals)},
                   coords={tname: t, "lat": LAT, "lon": LON}).to_netcdf(tmp_path / f"era_{k}.nc")
    return truth


@pytest.mark.parametrize("later", ["valid_time", "record"])
def test_files_that_name_their_time_axis_differently_keep_every_step(tmp_path, capsys, later):
    truth = _write_changeover(tmp_path, later)
    gv = GenericViewer(str(tmp_path / "era_*.nc"), strict=True)

    assert gv.steps == 8
    assert [gv.probe(10.0, 0.0, "t2m", i, 0)["value"] for i in range(8)] == truth
    assert gv.labels[4:6] == ["2023-01-02 00:00", "2023-01-02 12:00"]
    assert gv._series_values("t2m", 10.0, 0.0).tolist() == truth
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("extra, why", [
    ({}, "no time dimension"),
    # a time axis nothing on the grid steps along is not the file's
    ({"valid_time": pd.date_range("2023-01-05", periods=3)}, "no time dimension"),
])
def test_a_file_with_no_time_axis_in_a_timed_glob_is_left_out_by_name(tmp_path, capsys,
                                                                       extra, why):
    _write_changeover(tmp_path)
    xr.Dataset({"t2m": (("lat", "lon"), np.zeros((20, 40)))},
               coords={"lat": LAT, "lon": LON, **extra}).to_netcdf(tmp_path / "era_9.nc")
    gv = GenericViewer(str(tmp_path / "era_*.nc"), strict=True)

    assert gv.steps == 8
    assert "era_9.nc" not in [f.name for f in gv.files]
    assert f"era_9.nc: {why}" in capsys.readouterr().err


def test_a_time_axis_that_cannot_be_renamed_is_left_out_by_name(tmp_path, capsys):
    """`valid_time` steps, but a `time` variable is already there: counting the
    file and then failing every read of it is the worst of both."""
    _write_changeover(tmp_path)
    with xr.open_dataset(tmp_path / "era_3.nc") as ds:
        ds = ds.load().assign(time=((), 0.0))
    ds.to_netcdf(tmp_path / "era_3.nc")
    gv = GenericViewer(str(tmp_path / "era_*.nc"), strict=True)

    assert gv.steps == 6
    assert "era_3.nc: steps along 'valid_time', but also has a variable named 'time'" \
        in capsys.readouterr().err
def test_the_derive_box_works_on_a_generic_file(tmp_path):
    """--generic offered the derive box and evaluated none of it: every
    expression, `q - q` included, fell through to a plain plot of a variable
    named "q - q" and failed. Numbers, and the level, are the point here."""
    q = (np.arange(3)[:, None, None, None] * 10 + np.arange(2)[None, :, None, None]
         + np.zeros((3, 2, 20, 40))).astype("f4") / 1000
    gv = _open(tmp_path, xr.Dataset(
        {"q": (("valid_time", "pressure_level", "lat", "lon"), q)},
        coords={"valid_time": TIMES, "lat": LAT, "lon": LON,
                "pressure_level": ("pressure_level", [1000.0, 500.0], {"units": "hPa"})}))
    at = lambda e: gv.probe(10.0, 0.0, e, 2, 1)["value"]           # noqa: E731
    assert at("q") == pytest.approx(0.021)
    assert at("Q") == at("q")
    assert at("Q * 1000") == pytest.approx(21.0)
    assert at("q - q") == 0.0
    assert at("diff(q)") == pytest.approx(0.010)
    png, lo, hi = gv.frame("q * 1000", 2, 1, gv.home, "viridis", None, None)
    assert png[:4] == b"\x89PNG" and lo == pytest.approx(21.0)


# ------------------------------------------------------ where the map lands


def _outset(view, f=1.4):                       # viewer.py's outset(): the browser's box
    cx, cy = (view[0] + view[1]) / 2, (view[2] + view[3]) / 2
    w, h = (view[1] - view[0]) * f / 2, (view[3] - view[2]) * f / 2
    return (cx - w, cx + w, cy - h, cy + h)


def _stripe_viewer(tmp_path, lon0, lon1, lat0, lat1):
    lat, lon = np.arange(90, -90.5, -0.5), np.arange(0, 360, 0.5)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    mark = ((lo >= lon0) & (lo < lon1) & (la >= lat0) & (la < lat1)).astype("f4")
    xr.Dataset({"m": (("latitude", "longitude"), mark)},
               coords={"latitude": lat, "longitude": lon}).to_netcdf(tmp_path / "m.nc")
    return GenericViewer(tmp_path / "m.nc")


def _drawn_box(gv, box, nx, ny):
    img = gv._raster("m", 0, 0, box, nx, ny)            # row 0 south, as _png expects
    rows, cols = np.nonzero(img > 0.5)
    lon = lambda c: box[0] + (c + 0.5) / nx * (box[1] - box[0])   # noqa: E731
    lat = lambda r: box[2] + (r + 0.5) / ny * (box[3] - box[2])   # noqa: E731
    return lon(cols.min()), lon(cols.max()), lat(rows.min()), lat(rows.max())


def test_the_global_view_draws_data_where_it_is(tmp_path):
    """The browser asks for 1.4x its view; the frame must fill that whole box.

    Returning only the grid's own cells let the browser stretch them over the
    padded box, and the default view showed data ~30 degrees off its coastline.
    """
    gv = _stripe_viewer(tmp_path, 100, 110, 0, 10)
    box = _outset(gv.home)
    lo0, lo1, la0, la1 = _drawn_box(gv, box, round(gv.nx * 1.4), round(gv.ny * 1.4))
    assert (lo0, lo1, la0, la1) == pytest.approx((100, 110, 0, 10), abs=0.6)


def test_a_box_across_greenwich_wraps_a_global_grid(tmp_path):
    gv = _stripe_viewer(tmp_path, 350, 360, 40, 50)
    assert gv.cyclic
    lo0, lo1, _, _ = _drawn_box(gv, (-30, 30, 20, 70), 240, 200)
    assert (lo0, lo1) == pytest.approx((-10, 0), abs=0.6)


def test_beyond_a_regional_grid_is_transparent(tmp_path):
    xr.Dataset({"v": (("lat", "lon"), np.ones((31, 61)))},
               coords={"lat": np.linspace(30, 60, 31), "lon": np.linspace(-20, 40, 61)}
               ).to_netcdf(tmp_path / "r.nc")
    gv = GenericViewer(tmp_path / "r.nc")
    img = gv._raster("v", 0, 0, (-50, 70, 0, 90), 120, 90)
    assert not gv.cyclic
    assert np.isnan(img[:, :25]).all() and np.isnan(img[:25, :]).all()    # west, south
    assert np.isfinite(img[40, 40:90]).all()                                # the domain


# -------------------------------------------------- plots, figures, GIFs


@pytest.fixture
def months(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    return GenericViewer(_write_months(tmp_path))


def test_every_offered_kind_draws(months):
    for var in ("t", "gmean"):
        for kind in months.kinds(var):
            png = months.plot(var, 3, 1, kind, months.home, 400, 300, "viridis",
                              lon=20.0, lat=0.0)
            assert png[:4] == b"\x89PNG", (var, kind)


def test_a_kind_a_variable_does_not_offer_is_refused(months):
    assert "line" not in months.kinds("t")
    with pytest.raises(ValueError, match="not a plot of"):
        months.plot("t", 0, 0, "line", months.home)


def test_the_time_series_at_a_point_runs_through_every_file(months):
    s = months._series_at("t", 20.0, 0.0, level=1)
    assert s.values == pytest.approx([1, 11, 101, 111, 201, 211])
    assert np.issubdtype(s[s.dims[0]].dtype, np.datetime64)


def test_the_profile_at_a_point_is_along_the_level_axis(months):
    p = months._profile_at("t", 3, 20.0, 0.0)
    assert p.dims == ("pressure_level",)
    assert p.values == pytest.approx([110, 111])


def test_a_non_map_variable_running_in_time_is_read_across_files(months):
    assert months._whole("gmean").values == pytest.approx([0.5, 10.5, 100.5, 110.5,
                                                          200.5, 210.5])


def test_a_figure_is_sized_by_the_style_preset(months):
    from PIL import Image

    for style, size in (("paper", (1300, 780)), ("notebook", (900, 500))):
        for kind in ("map", "contourf"):
            png = months.figure("t", 0, 0, months.home, "viridis", None, None,
                                style=style, kind=kind)
            assert Image.open(io.BytesIO(png)).size == size


@pytest.mark.parametrize("kind", ["map", "contourf", "profile"])
def test_a_gif_has_a_frame_per_step(months, kind):
    from PIL import Image

    gif = months.gif("t", 1, months.home, "viridis", 0.0, 220.0, 72, 38, fps=4,
                     kind=kind, lon=20.0, lat=0.0)
    assert Image.open(io.BytesIO(gif)).n_frames == 6


@pytest.mark.parametrize("var, kind", [("t", "series"), ("t", "hist"), ("gmean", "auto")])
def test_a_gif_of_something_not_drawn_per_step_is_refused(months, var, kind):
    with pytest.raises(ValueError, match="GIF steps through time"):
        months.gif(var, 0, months.home, "viridis", None, None, 72, 38, kind=kind)


def test_the_plot_route_and_exports_are_served(months):
    import json
    import threading
    import urllib.request

    from gmpas.viewer import PAGE, _handler, bind

    srv = bind(_handler(months, PAGE), 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        meta = json.loads(urllib.request.urlopen(f"{base}/api/meta").read())
        row = next(v for v in meta["variables"] if v["name"] == "t")
        assert row["kinds"][0] == "map" and meta["files"] == 3 and meta["steps"] == 6

        q = "var=t&time=2&level=1&extent=0,360,-90,90&lon=20&lat=0"
        for path in (f"/api/plot?{q}&kind=contourf&w=400&h=300",
                     f"/api/export/figure?{q}&kind=profile&style=notebook"):
            assert urllib.request.urlopen(base + path).read()[:4] == b"\x89PNG"
        gif = urllib.request.urlopen(
            f"{base}/api/export/gif?{q}&kind=map&nx=72&ny=38").read()
        assert gif[:3] == b"GIF"
    finally:
        srv.shutdown()


# ------------------------------------------- saying which dimension is which


def _ioapi(tmp_path):
    """The IOAPI/CMAQ shape: a time axis gmpas cannot recognise, and a real
    vertical axis behind it. Values are 100*TSTEP + LAY, so a wrong reading
    is visible in the numbers."""
    block = np.array([[[[100 * t + k] * 40] * 20 for k in range(3)] for t in range(6)],
                     dtype="f8")
    ds = xr.Dataset({"O3": (("TSTEP", "LAY", "lat", "lon"), block)},
                    coords={"lat": ("lat", LAT, {"units": "degrees_north"}),
                            "lon": ("lon", LON, {"units": "degrees_east"})})
    return _open_asking(tmp_path, ds, name="cmaq.nc")


def test_a_grid_it_cannot_work_out_leaves_the_viewer_up_to_ask(tmp_path):
    """The whole point: the process used to exit before the server started,
    so the user was left guessing about a file they could not see."""
    ds = xr.Dataset({
        "thetao": (("y", "x"), np.zeros((20, 40))),
        "nav_lat": (("y", "x"), np.repeat(LAT[:, None], 40, 1),
                    {"standard_name": "latitude", "units": "degrees_north"}),
        "nav_lon": (("y", "x"), np.repeat(LON[None, :], 20, 0),
                    {"standard_name": "longitude", "units": "degrees_east"}),
    }, coords={"y": ("y", np.arange(20.0), {"axis": "Y"}),
               "x": ("x", np.arange(40.0), {"axis": "X"})})
    gv = _open_asking(tmp_path, ds)
    assert gv.needs_setup and "curvilinear" in gv.setup_problem
    s = gv.setup()
    assert s["needed"] and set(s["options"]) == {"x", "y", "time", "level"}
    assert s["dims"] == ["x", "y"]
    with pytest.raises(ValueError):                 # strict callers still raise
        _open(tmp_path, ds, name="strict.nc")


def test_choosing_the_dimensions_makes_the_hidden_axis_reachable(tmp_path):
    """TSTEP drove the level slider and LAY could not be reached at all."""
    gv = _ioapi(tmp_path)
    before = _row(gv, "O3")
    assert gv.steps == 1 and before["static"] and before["dim"] == "TSTEP"
    assert before["pinned"] == ["LAY"]              # 2 of 3 layers unreachable

    gv.configure({"time": "TSTEP", "level": "LAY"})

    after = _row(gv, "O3")
    assert gv.steps == 6 and not after["static"]
    assert after["dim"] == "LAY" and after["levels"] == 3 and after["pinned"] == []
    # and the numbers move on both axes now, which is the actual fix
    at = lambda t, k: float(np.asarray(
        gv._read("O3", t).isel(LAY=k).values).ravel()[0])
    assert (at(0, 0), at(3, 0), at(0, 2)) == (0.0, 300.0, 2.0)


def test_a_mapping_that_cannot_work_leaves_the_old_one_alone(tmp_path):
    """A rejected form must not leave the viewer worse than before it opened."""
    gv = _ioapi(tmp_path)
    home, steps = gv.home, gv.steps
    with pytest.raises(ValueError, match="not a variable"):
        gv.configure({"y": "no_such_thing"})
    assert (gv.home, gv.steps) == (home, steps) and not gv.needs_setup
    with pytest.raises(ValueError, match="not a time dimension"):
        gv.configure({"time": "nope"})
    assert (gv.home, gv.steps) == (home, steps)


def test_a_hand_picked_axis_is_checked_like_a_detected_one(tmp_path):
    """Saying "I know what I am doing" cannot make a grid regular."""
    ds = xr.Dataset({"v": (("a", "b"), np.zeros((6, 5)))},
                    coords={"a": ("a", np.array([0.0, 5, 1, 9, 2, 7])),
                            "b": ("b", np.arange(5.0))})
    gv = _open_asking(tmp_path, ds)
    with pytest.raises(ValueError, match="not monotonic"):
        gv.configure({"y": "a", "x": "b"})
    ds2 = xr.Dataset({"v": (("a", "b"), np.zeros((6, 5)))},
                     coords={"a": ("a", np.linspace(0, 4e5, 6)),
                             "b": ("b", np.arange(5.0))})
    gv2 = _open_asking(tmp_path, ds2, name="metres.nc")
    with pytest.raises(ValueError, match="cannot exceed 90"):
        gv2.configure({"y": "a", "x": "b"})


def test_a_guess_is_reported_and_a_reading_is_not(tmp_path):
    """The panel must not nag about files it understood: a warning that fires
    on everything is a warning nobody reads."""
    gv = _ioapi(tmp_path)
    doubts = gv.setup()["doubts"]
    assert any("no time axis was recognised" in d and "'TSTEP'" in d for d in doubts)
    # and it says the consequence that actually costs the user data
    assert any("'LAY' is pinned at 0" in d and "cannot be reached" in d
               for d in doubts)
    # it names the dimension the slider really drives, not the first alphabetically
    assert all("'LAY' drives" not in d for d in doubts)

    clean = _open_asking(tmp_path, xr.Dataset(
        {"t2m": (("time", "plev", "lat", "lon"), np.zeros((3, 4, 20, 40)))},
        coords={"time": TIMES, "lat": LAT, "lon": LON,
                "plev": ("plev", np.array([1000.0, 850, 500, 200]),
                         {"units": "hPa", "positive": "down"})}), name="clean.nc")
    assert clean.setup()["doubts"] == []


@pytest.mark.parametrize("extra, attrs", [
    ("sigma", {}),                              # described by whoever wrote it
    ("ensemble", {}),
    ("band", {"units": "nm"}),
])
def test_an_unrecognised_axis_with_its_own_values_is_left_alone(tmp_path, extra,
                                                                attrs):
    """gmpas does not know what `sigma` or `band` are, and does not need to:
    they carry coordinate values, so someone described them on purpose. Only
    a *bare* dimension -- no values at all, the shape of a record axis -- has
    actually been mistaken for a level here, so only that one is remarked on."""
    ds = xr.Dataset({"v": ((extra, "lat", "lon"), np.zeros((4, 20, 40)))},
                    coords={extra: (extra, np.arange(4.0), attrs),
                            "lat": ("lat", LAT), "lon": ("lon", LON)})
    gv = _open_asking(tmp_path, ds)
    assert not gv.needs_setup and gv.setup()["doubts"] == []


def test_a_plain_lat_lon_is_not_second_guessed(tmp_path):
    """`lat(lat)` with no attributes scores 1, but someone called it latitude
    and almost always meant it. A warning that fires on every ordinary file
    is a warning nobody reads."""
    ds = xr.Dataset({"v": (("lat", "lon"), np.zeros((20, 40)))},
                    coords={"lat": ("lat", LAT), "lon": ("lon", LON)})
    gv = _open_asking(tmp_path, ds)
    assert not gv.needs_setup and gv.setup()["doubts"] == []


def test_an_axis_attribute_on_an_unknown_name_is_flagged(tmp_path):
    """The rotated-pole shape: nothing but `axis: Y` points at `rlat`, and a
    projected axis in metres looks identical. It draws -- but it says so."""
    ds = xr.Dataset({"tas": (("rlat", "rlon"), np.zeros((20, 40)))},
                    coords={"rlat": ("rlat", np.linspace(-11, 11, 20),
                                     {"axis": "Y"}),
                            "rlon": ("rlon", np.linspace(-14, 14, 40),
                                     {"axis": "X"})})
    gv = _open_asking(tmp_path, ds)
    assert not gv.needs_setup
    assert any("rlat" in d and "rotated or projected" in d
               for d in gv.setup()["doubts"])


def test_the_dims_route_reads_and_writes(tmp_path):
    import json
    import threading
    import urllib.error
    import urllib.request

    from gmpas.viewer import PAGE, _handler, bind

    gv = _ioapi(tmp_path)
    srv = bind(_handler(gv, PAGE), 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        got = json.loads(urllib.request.urlopen(f"{base}/api/dims").read())
        assert got["current"]["time"] == "none" and "TSTEP" in got["dims"]

        body = json.dumps({"time": "TSTEP", "level": "LAY"})
        out = json.loads(urllib.request.urlopen(
            f"{base}/api/dims?map={urllib.parse.quote(body)}").read())
        assert out["state"] == "configured" and out["current"]["time"] == "TSTEP"
        meta = json.loads(urllib.request.urlopen(f"{base}/api/meta").read())
        assert meta["steps"] == 6

        for bad, why in ((json.dumps({"nope": "TSTEP"}), "unknown role"),
                         ("{not json", "not valid JSON"),
                         (json.dumps(["TSTEP"]), "expected an object")):
            with pytest.raises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(f"{base}/api/dims?map={urllib.parse.quote(bad)}")
            assert why in caught.value.read().decode()
    finally:
        srv.shutdown()


def test_nothing_is_drawn_while_the_grid_is_unknown(tmp_path):
    """A 500 naming the problem, not a picture of the wrong thing."""
    import json
    import threading
    import urllib.error
    import urllib.request

    from gmpas.viewer import PAGE, _handler, bind

    ds = xr.Dataset({"v": (("a", "b"), np.zeros((4, 5)))},
                    coords={"a": np.arange(4.0), "b": np.arange(5.0)})
    gv = _open_asking(tmp_path, ds)
    assert gv.needs_setup
    srv = bind(_handler(gv, PAGE), 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        # meta and dims still answer -- the page needs them to draw the panel
        assert json.loads(urllib.request.urlopen(f"{base}/api/meta").read())["setup"]
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(
                f"{base}/api/frame?var=v&time=0&level=0"
                f"&extent=0,4,0,3&cmap=viridis")
        assert "has not been worked out" in caught.value.read().decode()
    finally:
        srv.shutdown()


def test_the_mpas_viewer_has_no_plot_route(tmp_path):
    """/api/plot is served only to a viewer with plot(); the MPAS one has none."""
    from gmpas.viewer import Viewer

    assert not hasattr(Viewer, "plot")
