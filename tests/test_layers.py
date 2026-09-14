"""Composite maps for --generic: layer stacks, validated and drawn."""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from gmpas import layers as L
from gmpas.generic import GenericViewer

LAT = np.arange(90, -90.5, -5.0)
LON = np.arange(0, 360, 5.0)
PLEV = np.array([1000.0, 850.0, 500.0])


@pytest.fixture
def gv(tmp_path):
    """Two ERA5-style files; every value encodes (file, step, level) so a read
    can be checked, and u/v/z exist for vector and contour layers."""
    import matplotlib
    matplotlib.use("Agg")

    la, lo = np.meshgrid(LAT, LON, indexing="ij")
    for m, month in enumerate(("01", "02")):
        base = np.zeros((2, PLEV.size, LAT.size, LON.size), "f4")
        code = (100 * m + 10 * np.arange(2)[:, None, None, None]
                + np.arange(PLEV.size)[None, :, None, None]) + base
        wave = np.cos(np.deg2rad(4 * lo))[None, None] + base
        ds = xr.Dataset(
            {"t": (("valid_time", "pressure_level", "latitude", "longitude"),
                   code + np.cos(np.deg2rad(la))[None, None], {"units": "K"}),
             "z": (("valid_time", "pressure_level", "latitude", "longitude"),
                   5000 + 100 * wave, {"units": "m"}),
             "u": (("valid_time", "pressure_level", "latitude", "longitude"),
                   10 + wave, {"units": "m s-1"}),
             "v": (("valid_time", "pressure_level", "latitude", "longitude"),
                   2 * wave, {"units": "m s-1"}),
             "gmean": (("valid_time",), np.arange(2.0))},
            coords={"valid_time": pd.date_range(f"2024-{month}-01", periods=2, freq="12h"),
                    "pressure_level": ("pressure_level", PLEV,
                                       {"units": "hPa", "positive": "down"}),
                    "latitude": ("latitude", LAT, {"units": "degrees_north"}),
                    "longitude": ("longitude", LON, {"units": "degrees_east"})})
        ds.to_netcdf(tmp_path / f"era5_2024-{month}.nc")
    return GenericViewer(tmp_path)


def _png(data: bytes):
    from PIL import Image

    assert data[:4] == b"\x89PNG"
    return Image.open(io.BytesIO(data))


# ---------------------------------------------------------------- the offer


def test_layers_are_offered_for_map_variables_only(gv):
    assert "layers" in gv.kinds("t")
    assert "layers" not in gv.kinds("gmean")
    assert gv.kinds("t")[0] == "map"                   # opt-in: never the default


def test_the_page_gets_the_schema(gv):
    meta = gv.describe()
    assert set(meta["layer_schema"]["kinds"]) == set(L.LAYER_KINDS)
    assert "projection" in meta["layer_schema"]["figure"]
    json.dumps(meta)                                    # all of it serialises


def test_no_stack_draws_the_field_over_coastlines(gv):
    _png(gv.plot("t", 0, 0, "layers", gv.home, 400, 300))


# ------------------------------------------------------------ validation

SPATIAL = ["t", "u", "v", "z"]
COUNTS = {name: 3 for name in SPATIAL}


@pytest.mark.parametrize("stack, message", [
    ({"layers": [{"kind": "exec"}]}, "kind 'exec'"),
    ({"layers": [{"kind": "contourf", "var": "t", "options": {"transform": 1}}]},
     "unknown option"),
    ({"layers": [{"kind": "contourf", "var": "t", "options": {"cmap": "__import__"}}]},
     "not a matplotlib colormap"),
    ({"layers": [{"kind": "coastlines", "options": {"color": "red; x"}}]}, "not a colour"),
    ({"layers": [{"kind": "contour", "var": "t", "options": {"label_fmt": "%s%n"}}]},
     "printf number format"),
    ({"layers": [{"kind": "contourf", "var": "t", "options": {"hatches": "abc"}}]},
     "hatch patterns"),
    ({"layers": [{"kind": "contourf", "var": "t", "options": {"levels": "5, 3"}}]},
     "must increase"),
    ({"layers": [{"kind": "contourf", "var": "t", "options": {"levels": 1}}]},
     "count of levels"),
    ({"layers": [{"kind": "contourf", "var": "__class__"}]}, "not a map variable"),
    ({"layers": [{"kind": "quiver", "u": "u"}]}, "v=None"),
    ({"layers": [{"kind": "contourf", "var": "t", "level": 3}]}, "out of range"),
    ({"layers": [{"kind": "coastlines", "opacity": 2}]}, "outside"),
    ({"layers": [{"kind": "coastlines", "__proto__": 1}]}, "unknown key"),
    ({"layers": [{"kind": "coastlines"}] * (L.MAX_LAYERS + 1)}, "at most"),
    ({"figure": {"projection": "Foo"}, "layers": []}, "projection"),
    ({"layers": [], "extra": 1}, "unknown stack key"),
    ("{not json", "not valid JSON"),
])
def test_a_bad_stack_is_refused_by_name(stack, message):
    with pytest.raises(ValueError, match=message):
        L.clean(stack, SPATIAL, COUNTS)


def test_options_are_coerced_to_their_types():
    stack = L.clean(json.dumps({"layers": [
        {"kind": "contourf", "var": "t", "opacity": "0.5", "level": "2",
         "options": {"levels": "0, 10, 20", "vmin": "1", "colorbar": "false",
                     "hatches": "//, , .."}},
        {"kind": "contour", "var": "z", "options": {"levels": "12", "colors": "red, blue"}},
    ]}), SPATIAL, COUNTS)
    fill, lines = stack["layers"]
    assert fill["opacity"] == 0.5 and fill["level"] == 2
    assert fill["options"] == {"levels": [0.0, 10.0, 20.0], "vmin": 1.0, "colorbar": False,
                               "hatches": ["//", None, ".."]}
    assert lines["options"] == {"levels": 12, "colors": ["red", "blue"]}
    assert isinstance(stack, L.Stack)


def test_empty_options_fall_back_to_the_defaults():
    stack = L.clean({"layers": [{"kind": "coastlines", "options": {"color": ""}}]},
                    SPATIAL, COUNTS)
    assert stack["layers"][0]["options"] == {}


# ------------------------------------------------------------- the data


def test_a_layer_follows_the_slider_unless_pinned(gv):
    stack = gv.layer_stack({"layers": [
        {"kind": "contourf", "var": "t"},
        {"kind": "contourf", "var": "t", "level": 2}]}, "t")
    follow, pinned = stack["layers"]
    # step 3 is February's second step (code 110); the slider sits at level 1
    got_follow = L.layer_data(gv, follow, 3, 1, gv.home)
    got_pinned = L.layer_data(gv, pinned, 3, 1, gv.home)
    assert np.nanmin(got_follow.values) == pytest.approx(111, abs=1)
    assert np.nanmin(got_pinned.values) == pytest.approx(112, abs=1)


def test_a_crop_across_the_seam_is_moved_into_the_map_frame(gv):
    """260..400 unwrapped must become -100..40: streamplot regrids over the data's
    own x range, and drew nothing at all on a map framed at -100..40."""
    da = gv._cropped("u", 0, 0, (-100, 40, 0, 60))
    x = np.asarray(L._in_frame(gv, da, 0.0)[gv.lon_name].values)
    assert x.min() == pytest.approx(-100, abs=5) and x.max() == pytest.approx(40, abs=5)
    assert np.all(np.diff(x) > 0)


def test_a_global_grid_is_closed_at_the_seam(gv):
    da = gv._cropped("t", 0, 0, (0, 360, -90, 90))
    closed = L._in_frame(gv, da, 180.0)
    assert closed.sizes[gv.lon_dim] == da.sizes[gv.lon_dim] + 1
    assert closed.isel({gv.lon_dim: -1}).values == pytest.approx(
        closed.isel({gv.lon_dim: 0}).values)


# ----------------------------------------------------------- drawing


def test_every_layer_kind_draws(gv, monkeypatch):
    # Natural Earth fetches are network I/O; features are drawn from what the
    # test environment has, and a missing one is its own test below
    monkeypatch.setattr(L, "_ensure_natural_earth", lambda *a: None)
    for kind, spec in L.LAYER_KINDS.items():
        if spec["group"] == "feature" and kind not in ("coastlines", "gridlines"):
            continue
        layer = {"kind": kind}
        if "var" in spec["needs"]:
            layer["var"] = "t"
        if "u" in spec["needs"]:
            layer.update(u="u", v="v")
        stack = {"layers": [layer, {"kind": "coastlines"}]}
        _png(gv.plot("t", 1, 1, "layers", (-60, 60, -40, 70), 500, 350, layers=stack))


@pytest.mark.parametrize("projection", L.PROJECTIONS)
def test_every_projection_draws_with_its_title_inside_the_figure(gv, projection):
    import matplotlib.pyplot as plt

    stack = gv.layer_stack({"figure": {"projection": projection, "title": "T"},
                            "layers": [{"kind": "contourf", "var": "t"},
                                       {"kind": "coastlines"}]}, "t")
    fig = plt.figure(figsize=(8, 6), dpi=60, layout="constrained")
    try:
        L.draw(gv, fig, stack, 0, 0, (-30, 50, 20, 70))
        fig.canvas.draw()
        box = fig._suptitle.get_window_extent(fig.canvas.get_renderer())
        assert box.y1 <= fig.bbox.height + 0.5
    finally:
        plt.close(fig)


def test_gridlines_draw_above_a_filled_layer(gv):
    """Gridliner pins itself to zorder 2, under any filled layer, whatever
    zorder it is given -- so it is set afterwards."""
    import matplotlib.pyplot as plt

    stack = gv.layer_stack({"layers": [{"kind": "contourf", "var": "t"},
                                       {"kind": "gridlines"}]}, "t")
    fig = plt.figure()
    try:
        ax = L.draw(gv, fig, stack, 0, 0, gv.home)
        grid = next(a for a in ax.artists if type(a).__name__ == "Gridliner")
        fills = [c.get_zorder() for c in ax.collections if hasattr(c, "levels")]
        assert grid.get_zorder() > max(fills)
    finally:
        plt.close(fig)


def test_hidden_layers_are_not_drawn(gv):
    import matplotlib.pyplot as plt

    stack = gv.layer_stack({"layers": [{"kind": "contour", "var": "z", "visible": False},
                                       {"kind": "coastlines"}]}, "t")
    fig = plt.figure()
    try:
        ax = L.draw(gv, fig, stack, 0, 0, gv.home)
        assert not [c for c in ax.collections if hasattr(c, "levels")]
    finally:
        plt.close(fig)


def test_a_missing_natural_earth_layer_is_a_message(gv, monkeypatch):
    from cartopy.io import shapereader

    def offline(**_):
        raise OSError("no network")

    monkeypatch.setattr(shapereader, "natural_earth", offline)
    with pytest.raises(ValueError, match="borders|admin_0.*not available"):
        gv.plot("t", 0, 0, "layers", gv.home, 300, 200,
                layers={"layers": [{"kind": "borders"}]})


# ------------------------------------------------------ figures and GIFs


def test_a_figure_exports_the_stack_at_the_style_size(gv):
    png = gv.figure("t", 0, 0, gv.home, None, None, None, "notebook", kind="layers",
                    layers={"layers": [{"kind": "contourf", "var": "t"},
                                       {"kind": "barbs", "u": "u", "v": "v"}]})
    assert _png(png).size == (900, 500)


def test_a_gif_freezes_colour_ranges_and_contour_values(gv):
    import matplotlib.pyplot as plt

    stack = gv.layer_stack({"layers": [
        {"kind": "contourf", "var": "t", "options": {"levels": 6}},
        {"kind": "contour", "var": "z"},
        {"kind": "contourf", "var": "t", "options": {"levels": "0, 50, 300"}}]}, "t")
    frozen = L.freeze_ranges(gv, stack, 0, gv.home)
    assert "vmin" in frozen["layers"][0]["options"]
    assert "vmin" not in frozen["layers"][2]["options"]        # explicit levels win
    levels = []
    for step in (0, 3):
        fig = plt.figure()
        ax = L.draw(gv, fig, frozen, step, 0, gv.home)
        levels.append([list(c.levels) for c in ax.collections if hasattr(c, "levels")])
        plt.close(fig)
    assert levels[0] == levels[1]

    from PIL import Image
    gif = gv.gif("t", 0, gv.home, None, None, None, kind="layers", layers=stack)
    assert Image.open(io.BytesIO(gif)).n_frames == gv.steps


def test_layers_and_the_hovmoller_are_offered_side_by_side(gv, monkeypatch):
    """Both extend the same plot calls; each must get its own argument."""
    assert {"layers", "hovmoller"} <= set(gv.kinds("t"))
    assert gv.hovmoller("t", 0, band=(-10.0, 10.0)).dims[0] == gv.time_name
    # a GIF frame that lost the stack would quietly draw the default one instead
    drawn = []
    draw = L.draw
    monkeypatch.setattr(L, "draw", lambda gv_, fig, stack, *a, **k: (
        drawn.append([layer["kind"] for layer in stack["layers"]]),
        draw(gv_, fig, stack, *a, **k))[1])
    gv.gif("t", 0, gv.home, None, None, None, kind="layers",
           layers={"layers": [{"kind": "barbs", "u": "u", "v": "v"}]})
    assert drawn and all(kinds == ["barbs"] for kinds in drawn)


def test_the_stack_travels_over_http(gv):
    import threading
    import urllib.error
    import urllib.parse
    import urllib.request

    from gmpas.viewer import PAGE, _handler, bind

    srv = bind(_handler(gv, PAGE), 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    stack = json.dumps({"layers": [{"kind": "contourf", "var": "t", "level": 2},
                                   {"kind": "quiver", "u": "u", "v": "v"}]})
    q = urllib.parse.urlencode({"var": "t", "time": 1, "level": 0, "kind": "layers",
                                "extent": "0,360,-90,90", "layers": stack})
    try:
        _png(urllib.request.urlopen(f"{base}/api/plot?{q}&w=400&h=300").read())
        _png(urllib.request.urlopen(f"{base}/api/export/figure?{q}&style=notebook").read())
        bad = q.replace("quiver", "exec")
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(f"{base}/api/plot?{bad}")
        assert "kind 'exec'" in json.loads(err.value.read())["error"]
    finally:
        srv.shutdown()


def test_the_mpas_viewer_is_untouched(tmp_path):
    from gmpas.viewer import Viewer

    assert not hasattr(Viewer, "plot") and not hasattr(Viewer, "layer_stack")


@pytest.mark.parametrize("projection", L.PROJECTIONS)
@pytest.mark.parametrize("extent", [(0.5, 358.5, -90, 90), (100, 180, -60, -10),
                                    (-60, 40, -20, 60), (0, 359, -90, -60),
                                    (0, 359, 60, 90), (10, 12, 45, 46)])
def test_every_projection_survives_every_view(gv, projection, extent):
    """A reload starts from the global view with the saved projection; a conic
    map there, or on the far pole, had corners at inf and failed to draw."""
    stack = {"figure": {"projection": projection},
             "layers": [{"kind": "contourf", "var": "t"}, {"kind": "coastlines"}]}
    _png(gv.plot("t", 0, 0, "layers", extent, 320, 240, layers=stack))
