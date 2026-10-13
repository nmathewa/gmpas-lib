"""An expression of fields on a lat/lon file is one field everywhere.

The fast map, a click, a question, a layer of a figure and an exported figure
all read `t - 273.15` the same way: from its operands in float64. They used to
part company -- the map and the click evaluated it, the rest looked for a
variable of that name and failed.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from gmpas import script

EXTENT = (100.0, 130.0, -20.0, 10.0)


@pytest.fixture(scope="module")
def gv(tmp_path_factory):
    import matplotlib
    matplotlib.use("Agg")
    from gmpas.generic import GenericViewer

    lat = np.arange(20.0, -30.1, -2.5)                  # north first, as ERA5 is
    lon = np.arange(90.0, 140.1, 2.5)
    lev = np.array([1000.0, 850.0, 500.0])
    rng = np.random.default_rng(1)
    shape = (3, lev.size, lat.size, lon.size)
    t = (280 + 20 * rng.random(shape)).astype("f4")
    t[:, :, 2, 3] = np.nan                              # a missing cell
    ds = xr.Dataset(
        {"t": (("valid_time", "pressure_level", "latitude", "longitude"), t,
               {"units": "K", "long_name": "Temperature"}),
         "u": (("valid_time", "pressure_level", "latitude", "longitude"),
               rng.normal(size=shape).astype("f4"), {"units": "m s-1"}),
         "v": (("valid_time", "pressure_level", "latitude", "longitude"),
               rng.normal(size=shape).astype("f4"), {"units": "m s-1"})},
        coords={"valid_time": pd.date_range("2024-01-01", periods=3, freq="6h"),
                "pressure_level": ("pressure_level", lev, {"units": "hPa"}),
                "latitude": lat, "longitude": lon})
    path = tmp_path_factory.mktemp("expr") / "levels.nc"
    ds.to_netcdf(path)
    viewer = GenericViewer(path)
    yield viewer
    viewer.close()


def test_a_crop_of_an_expression_is_its_operands_in_float64(gv):
    plain = gv._cropped("t", 1, 1, EXTENT)
    shifted = gv._cropped("t - 273.15", 1, 1, EXTENT)
    assert shifted.dtype == np.float64 and shifted.dims == plain.dims
    assert shifted.name == "t - 273.15" and shifted.attrs == {}
    np.testing.assert_array_equal(shifted.values, plain.values.astype("f8") - 273.15)
    speed = gv._cropped("hypot(u, v)", 2, 0, EXTENT)
    np.testing.assert_array_equal(
        speed.values, np.hypot(gv._cropped("u", 2, 0, EXTENT).values.astype("f8"),
                               gv._cropped("v", 2, 0, EXTENT).values.astype("f8")))


def test_a_click_a_question_and_a_figure_agree_on_one_cell(gv):
    clicked = gv.probe(110.0, 0.0, "t - 273.15", 1, 1)
    crop = gv._cropped("t - 273.15", 1, 1, EXTENT)
    assert clicked["value"] == float(crop.sel(longitude=clicked["lon"],
                                              latitude=clicked["lat"]))
    found = gv.extreme("t - 273.15", 1, 1, EXTENT, "max")
    plain = gv.extreme("t", 1, 1, EXTENT, "max")
    assert found["value"] == plain["value"] - 273.15
    assert (found["lon"], found["lat"]) == (plain["lon"], plain["lat"])
    # the missing cell is passed over, and a name in another case is the field
    assert np.isfinite(gv.extreme("T", 1, 1, EXTENT, "min")["value"])
    with pytest.raises(KeyError, match="'nope' is not a variable"):
        gv.extreme("t - nope", 1, 1, EXTENT, "max")


def test_a_layer_may_be_an_expression_on_a_lat_lon_file(gv):
    meta = gv.describe()
    assert meta["layer_expressions"] is True
    state = script.resolve(script.parse(
        "shade(t.isel(level=1))\n"
        "contour((t - 273.15).isel(level=1), levels=[20], labels=True)\n"
        "vectors(u, v)"), meta)
    kinds = [(x["kind"], x.get("var"), x["level"]) for x in state["layers"]["layers"]]
    assert kinds[:2] == [("pcolormesh", "t", 1), ("contour", "t - 273.15", 1)]
    png = gv.plot("t", 1, 1, "layers", EXTENT, 640, 420, layers=state["layers"])
    assert png[:4] == b"\x89PNG"
    # one draw call of an expression is a figure of that one layer
    alone = script.resolve(script.parse("contourf(hypot(u, v).isel(level=2))"), meta)
    assert alone["kind"] == "layers"
    assert alone["layers"]["layers"][0] == {
        "kind": "contourf", "var": "hypot(u, v)", "visible": True, "opacity": 1,
        "level": 2, "options": {}}
    assert gv.plot("u", 0, 0, "layers", EXTENT, 640, 420, layers=alone["layers"])
    with pytest.raises(ValueError, match="level 7 is out of range for 't - 273.15'"):
        gv.layer_stack({"figure": {}, "layers": [
            {"kind": "contour", "var": "t - 273.15", "level": 7}]}, "t")
    with pytest.raises(ValueError, match="var='t - nope' is not a map variable"):
        gv.layer_stack({"figure": {}, "layers": [{"kind": "contour", "var": "t - nope"}]},
                       "t")


def test_an_expression_exports_as_a_figure_of_its_one_layer(gv):
    assert gv.kinds("t - 273.15") == ["map", "layers"]
    png = gv.figure("t - 273.15", 0, 1, EXTENT, "RdBu_r", 5, 30)
    assert png[:4] == b"\x89PNG"
    assert gv.figure("t", 0, 1, EXTENT, "viridis", None, None)[:4] == b"\x89PNG"


# -------------------------------------------------------------- over HTTP

@pytest.fixture(scope="module")
def served(gv):
    from gmpas.viewer import PAGE, _handler, bind

    srv = bind(_handler(gv, PAGE), 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/"
    srv.shutdown()


def _get(base, route, **query):
    url = base + route + "?" + urllib.parse.urlencode(query)
    try:
        with urllib.request.urlopen(url) as reply:
            return reply.status, json.loads(reply.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_a_missing_value_is_null_not_nan(served, gv):
    lat, lon = float(gv._lat_file[2]), float(gv._lon_file[3])      # the missing cell
    status, got = _get(served, "api/probe", lon=lon, lat=lat, var="t", time=0, level=0)
    assert status == 200 and got["value"] is None
    status, got = _get(served, "api/probe", lon=110, lat=0, var="t", time=0, level=0)
    assert status == 200 and isinstance(got["value"], float)


def test_a_question_about_an_expression_is_answered(served):
    status, got = _get(served, "api/extreme", var="t - 273.15", time=1, level=1,
                       extent="100,130,-20,10", what="max")
    assert status == 200 and 0 < got["value"] < 30


def test_a_stop_request_stops_and_starts_nothing(served, gv):
    before = len(gv._series_jobs.cache._items) if hasattr(gv, "_series_jobs") else None
    for var in ("t", "t - 273.15"):
        status, got = _get(served, "api/series", lon=110, lat=0, var=var, level=0, stop=1)
        assert (status, got) == (200, {"state": "stopped", "was_running": False})
    if before is not None:
        assert len(gv._series_jobs.cache._items) == before
