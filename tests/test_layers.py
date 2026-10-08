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
     "not a known colormap"),
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


# ------------------------------------------------------------ grid values


def _printed(ax):
    """(lon, lat, text) of every grid value on the map, and any note. Labels sit
    in the map's own longitude frame, so its central longitude is added back."""
    central = ax.projection.proj4_params.get("lon_0", 0.0)
    values = [(t.get_position()[0] + central, t.get_position()[1], t.get_text())
              for t in ax.texts if t.get_transform() is not ax.transAxes]
    notes = [t.get_text() for t in ax.texts if t.get_transform() is ax.transAxes]
    return values, notes


def _draw_values(gv, extent, **options):
    import matplotlib.pyplot as plt

    stack = gv.layer_stack({"layers": [{"kind": "gridvalues", "var": "t", "level": 1,
                                        "options": options}]}, "t")
    fig = plt.figure()
    try:
        return _printed(L.draw(gv, fig, stack, 1, 0, extent))
    finally:
        plt.close(fig)


def test_grid_values_are_the_files_own_numbers(gv, tmp_path):
    """Display only: each label is the value at that cell, never interpolated."""
    with xr.open_dataset(tmp_path / "era5_2024-01.nc") as ds:
        truth = ds["t"].isel(valid_time=1, pressure_level=1).load()
    printed, notes = _draw_values(gv, (20.0, 80.0, -30.0, 30.0), stride=2, fmt="%.6g")
    assert printed and not notes
    for lon, lat, text in printed:
        cell = truth.sel(longitude=lon % 360, latitude=lat)        # exact, not nearest
        assert text == f"{float(cell):.6g}"


def test_grid_values_stop_at_the_cap_and_say_why(gv):
    printed, notes = _draw_values(gv, gv.home, stride=1, max_labels=10)
    assert printed == []
    assert "past the limit of 10" in notes[0]


def test_grid_values_work_across_the_seam(gv):
    """A view from 340 to 20 degrees takes cells from both ends of the file."""
    printed, _ = _draw_values(gv, (340.0, 380.0, -20.0, 20.0), stride=1)
    lons = {round(lon % 360) for lon, _, _ in printed}
    assert {340, 355, 0, 20} <= lons and not lons & {180}


def test_grid_value_labels_stay_on_their_cells_as_the_map_pans(gv):
    a, _ = _draw_values(gv, (0.0, 60.0, -30.0, 30.0), stride=3)
    b, _ = _draw_values(gv, (5.0, 65.0, -30.0, 30.0), stride=3)
    cells = lambda printed: {(round(x % 360), round(y)) for x, y, _ in printed  # noqa: E731
                             if 10 <= x % 360 <= 50}
    assert cells(a) == cells(b)


def test_a_grid_value_format_is_checked(gv):
    with pytest.raises(ValueError, match="printf number format"):
        gv.layer_stack({"layers": [{"kind": "gridvalues", "var": "t",
                                    "options": {"fmt": "%s"}}]}, "t")


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


# ------------------------------------------------------------ presets

PRESET = {"figure": {"title": "winds"},
          "layers": [{"kind": "contourf", "var": "t", "level": 1,
                      "options": {"cmap": "RdBu_r", "interval": 2}},
                     {"kind": "quiver", "u": "u", "v": "v"},
                     {"kind": "coastlines"}]}


def test_a_preset_applies_unchanged_to_a_file_with_the_same_variables(gv):
    assert gv.bind_preset(json.dumps(PRESET)) == PRESET


def test_a_preset_is_renamed_only_where_it_must_be(gv):
    upper = json.loads(json.dumps(PRESET).replace('"t"', '"T"'))     # T finds t
    got = gv.bind_preset(upper, {"u": "v", "v": "u"})                 # a mapping wins
    assert [got["layers"][0]["var"], got["layers"][1]["u"], got["layers"][1]["v"]] == \
        ["t", "v", "u"]
    assert got["layers"][0]["options"] == PRESET["layers"][0]["options"]
    assert got["figure"] == PRESET["figure"]


def test_a_preset_names_every_variable_the_file_lacks(gv):
    from gmpas.layers import MissingVariables

    alien = json.loads(json.dumps(PRESET))
    alien["layers"][0]["var"], alien["layers"][1]["u"] = "z500", "uwnd"
    with pytest.raises(MissingVariables) as err:
        gv.bind_preset(alien)
    assert err.value.missing == ["z500", "uwnd"]
    assert set(err.value.choices) >= {"t", "u", "v"}


def test_a_preset_goes_through_the_same_validation_as_a_stack(gv):
    bad = json.loads(json.dumps(PRESET))
    bad["layers"][0]["options"]["alpha"] = 0.5
    with pytest.raises(ValueError, match="unknown option"):
        gv.bind_preset(bad)
    bad = json.loads(json.dumps(PRESET))
    bad["layers"][0]["level"] = 99
    with pytest.raises(ValueError, match="level 99 is out of range"):
        gv.bind_preset(bad)


def test_a_preset_is_bound_over_http(gv):
    import threading
    import urllib.error
    import urllib.parse
    import urllib.request

    from gmpas.viewer import PAGE, _handler, bind

    srv = bind(_handler(gv, PAGE), 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}/api/layers/bind?"
    try:
        q = urllib.parse.urlencode({"stack": json.dumps(PRESET)})
        assert json.loads(urllib.request.urlopen(base + q).read())["stack"] == PRESET
        alien = json.dumps(PRESET).replace('"t"', '"z500"')
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(base + urllib.parse.urlencode({"stack": alien}))
        body = json.loads(err.value.read())
        assert err.value.code == 400 and body["missing"] == ["z500"]
        q = urllib.parse.urlencode({"stack": alien, "map": json.dumps({"z500": "t"})})
        assert json.loads(urllib.request.urlopen(base + q).read())["stack"] == PRESET
    finally:
        srv.shutdown()


def test_the_mpas_viewer_is_untouched(tmp_path):
    from gmpas.viewer import Viewer

    # it plots a Hovmöller now (#111), but layers are still --generic only
    assert not hasattr(Viewer, "layer_stack")


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


def test_every_plot_kind_declares_what_its_controls_can_do(gv):
    """The page shows, hides and disables from this table alone, so a kind
    missing from it would silently get the fast map's controls."""
    from gmpas.generic import KIND_CAPS, KIND_LABELS

    assert set(KIND_CAPS) == set(KIND_LABELS)
    keys = {"colour", "options", "pan", "frames", "probe", "gif", "data"}
    assert all(set(caps) == keys for caps in KIND_CAPS.values())
    assert gv.describe()["kind_caps"] == KIND_CAPS
    # only the fast map has palette frames to pan over and play
    assert [k for k, c in KIND_CAPS.items() if c["frames"]] == ["map"]
    assert [k for k, c in KIND_CAPS.items() if c["pan"]] == ["map"]
    # netCDF holds the numbers behind a picture the input files do not have
    assert [k for k, c in KIND_CAPS.items() if c["data"]] == ["hovmoller"]
    for kind in ("layers", "hovmoller", "series"):
        assert not KIND_CAPS[kind]["options"], kind


@pytest.mark.parametrize("stride, points, expect", [
    (0, 24, (10, 19)),           # default: ~STREAM_POINTS along the long axis
    (2, 24, (19, 37)),           # an explicit stride wins
    (1, 24, (37, 73)),           # 1 is the full grid (73: the seam column is closed)
])
def test_streamlines_are_integrated_on_a_strided_field(gv, monkeypatch, stride, points,
                                                        expect):
    """The full 361x720 grid made one streamplot take ~11 s (#125); the input is
    now strided like quiver and barbs. Pins the shape actually handed over."""
    import matplotlib.pyplot as plt
    from cartopy.mpl.geoaxes import GeoAxes

    monkeypatch.setattr(L, "STREAM_POINTS", points)
    seen = []
    real = GeoAxes.streamplot

    def spy(self, x, y, u, v, **kw):
        seen.append(np.shape(u))
        return real(self, x, y, u, v, **kw)

    monkeypatch.setattr(GeoAxes, "streamplot", spy)
    stack = gv.layer_stack({"layers": [{"kind": "streamplot", "u": "u", "v": "v",
                                        "options": {"stride": stride}}]}, "u")
    fig = plt.figure(figsize=(6, 4), dpi=50)
    try:
        L.draw(gv, fig, stack, 0, 0, gv.home)
    finally:
        plt.close(fig)
    assert seen == [expect]
