"""`gmpas view --generic`: a plain netCDF file on a regular lat/lon grid."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from gmpas.generic import GenericViewer

LAT = np.linspace(-89.5, 89.5, 20)
LON = np.linspace(0.5, 359.5, 40)
TIMES = pd.date_range("2024-01-01", periods=3, freq="6h")


def _open(tmp_path, ds, name="f.nc"):
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
