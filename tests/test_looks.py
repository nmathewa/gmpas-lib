"""Layer colour options and the GrADS/Ferret looks (#92, #93, #94, #96, #112)."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from gmpas import layers as L
from gmpas.generic import GenericViewer

LAT = np.arange(60, 19.5, -2.0)
LON = np.arange(0, 60, 2.0)

SPATIAL = ["t", "u", "v"]
COUNTS = {"t": 1, "u": 1, "v": 1}


@pytest.fixture
def gv(tmp_path):
    import matplotlib
    matplotlib.use("Agg")

    la, lo = np.meshgrid(LAT, LON, indexing="ij")
    t = (240 + (60 - la) + np.zeros((2, *la.shape))).astype("f8")   # 240 north .. 280 south
    t[:, 0, 0] = np.nan
    xr.Dataset({"t": (("time", "latitude", "longitude"), t, {"units": "K"}),
                "u": (("time", "latitude", "longitude"), np.full_like(t, 10.0)),
                "v": (("time", "latitude", "longitude"), np.full_like(t, 5.0))},
               coords={"time": pd.date_range("2024-01-01", periods=2),
                       "latitude": ("latitude", LAT, {"units": "degrees_north"}),
                       "longitude": ("longitude", LON, {"units": "degrees_east"})}
               ).to_netcdf(tmp_path / "f.nc")
    viewer = GenericViewer(tmp_path / "f.nc")
    yield viewer
    viewer.close()


def _draw(gv, stack):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 6), dpi=60, layout="constrained")
    ax = L.draw(gv, fig, gv.layer_stack(stack, "t"), 0, 0, (0, 58, 20, 60))
    fig.canvas.draw()
    return fig, ax


def _colorbars(fig):
    return [a._colorbar for a in fig.axes if hasattr(a, "_colorbar")]


# ------------------------------------------------------------- validation


@pytest.mark.parametrize("options, message", [
    ({"bands": 10, "norm": "log"}, "bands and norm=log"),
    ({"gamma": 0.5}, "gamma applies to norm=power"),
    ({"center": 260, "norm": "power"}, "center and norm=power"),
    ({"bands": 1}, "outside"),
    ({"cmap": "ferret.nonexistent"}, "not a known colormap"),
    ({"over_color": "not-a-colour"}, "not a colour"),
    ({"colorbar_format": "%s"}, "printf number format"),
])
def test_contradictory_or_bad_colour_options_are_refused(options, message):
    with pytest.raises(ValueError, match=message):
        L.clean({"layers": [{"kind": "pcolormesh", "var": "t", "options": options}]},
                SPATIAL, COUNTS)


def test_bands_and_missing_colour_are_pixel_layer_options_only():
    with pytest.raises(ValueError, match="unknown option"):
        L.clean({"layers": [{"kind": "contourf", "var": "t", "options": {"bands": 5}}]},
                SPATIAL, COUNTS)


@pytest.mark.parametrize("name", ["cmo.thermal", "ferret.rnb2", "grads.rainbow", "viridis"])
def test_every_palette_family_is_accepted(name):
    if name.startswith("cmo."):
        pytest.importorskip("cmocean")
    L.clean({"layers": [{"kind": "imshow", "var": "t", "options": {"cmap": name}}]},
            SPATIAL, COUNTS)


# --------------------------------------------------------------- colours


def test_bands_colour_each_step_and_keep_the_maximum_inside(gv):
    fig, ax = _draw(gv, {"layers": [{"kind": "pcolormesh", "var": "t", "options": {
        "cmap": "grads.rainbow", "bands": 4, "vmin": 240, "vmax": 280}}]})
    mesh = next(c for c in ax.collections if hasattr(c, "get_array")
                and type(c.norm).__name__ == "BoundaryNorm")
    edges = mesh.norm.boundaries
    assert edges.size == 5 and edges[0] == 240 and edges[-1] > 280
    assert mesh.norm(280.0) == mesh.norm(275.0)             # the maximum is in band 4


def test_out_of_range_and_missing_colours_reach_the_artist(gv):
    fig, ax = _draw(gv, {"layers": [{"kind": "pcolormesh", "var": "t", "options": {
        "vmin": 250, "vmax": 270, "under_color": "black", "over_color": "white",
        "missing_color": "red", "extend": "both"}}]})
    mesh = next(c for c in ax.collections
                if hasattr(c, "cmap") and c.get_array() is not None)
    assert tuple(mesh.cmap.get_under()) == (0, 0, 0, 1)
    assert tuple(mesh.cmap.get_over()) == (1, 1, 1, 1)
    assert tuple(mesh.cmap.get_bad()) == (1, 0, 0, 1)
    assert _colorbars(fig)[0].extend == "both"


def test_reverse_and_power_scaling(gv):
    fig, ax = _draw(gv, {"layers": [{"kind": "imshow", "var": "t", "options": {
        "cmap": "viridis", "reverse": True, "norm": "power", "gamma": 0.5,
        "vmin": 240, "vmax": 280}}]})
    image = ax.get_images()[0]
    assert type(image.norm).__name__ == "PowerNorm" and image.norm.gamma == 0.5
    from matplotlib import colormaps
    assert np.allclose(image.cmap(0.0), colormaps["viridis"](1.0))


def test_contourf_keeps_custom_extremes_through_its_levels(gv):
    fig, ax = _draw(gv, {"layers": [{"kind": "contourf", "var": "t", "options": {
        "levels": "250, 260, 270", "under_color": "black", "over_color": "white",
        "extend": "both"}}]})
    cs = next(c for c in ax.collections if getattr(c, "filled", False))
    assert tuple(cs.cmap.get_under()) == (0, 0, 0, 1)
    assert tuple(cs.cmap.get_over()) == (1, 1, 1, 1)


def test_colorbar_format_and_tick_count(gv):
    fig, _ = _draw(gv, {"layers": [{"kind": "pcolormesh", "var": "t", "options": {
        "colorbar_format": "%.1f", "colorbar_ticks": 3}}]})
    cb = _colorbars(fig)[0]
    ticks = cb.ax.get_xticklabels() or cb.ax.get_yticklabels()
    labels = [t.get_text().replace("\u2212", "-") for t in ticks if t.get_text()]
    assert 1 <= len(labels) <= 4                            # MaxNLocator(3): up to 4 ticks
    assert all(re.fullmatch(r"-?\d+\.\d", label) for label in labels), labels


# ----------------------------------------------------------------- looks


def test_a_look_supplies_the_palette_only_when_none_is_given(gv):
    fig, ax = _draw(gv, {"figure": {"look": "grads"}, "layers": [
        {"kind": "pcolormesh", "var": "t"},
        {"kind": "contourf", "var": "t", "options": {"cmap": "viridis"}}]})
    maps = [c.cmap.name for c in ax.collections if hasattr(c, "cmap")
            and c.get_array() is not None]
    assert any(name.startswith("grads.rainbow") for name in maps)
    assert any(name.startswith("viridis") for name in maps)


def test_the_grads_look_draws_contour_lines_in_its_rainbow(gv):
    _, ax = _draw(gv, {"figure": {"look": "grads"},
                       "layers": [{"kind": "contour", "var": "t",
                                   "options": {"levels": 5}}]})
    lines = next(c for c in ax.collections if getattr(c, "filled", True) is False)
    assert lines.cmap.name.startswith("grads.rainbow")


def test_grads_colour_key_is_boxed_horizontal_labelled_between_boxes(gv):
    fig, _ = _draw(gv, {"figure": {"look": "grads"}, "layers": [
        {"kind": "pcolormesh", "var": "t",
         "options": {"bands": 5, "vmin": 240, "vmax": 280, "extend": "both"}}]})
    cb = _colorbars(fig)[0]
    assert cb.orientation == "horizontal" and cb.drawedges and not cb.extendrect
    assert len(cb.dividers.get_segments()) > 0
    ticks = list(cb.get_ticks())
    assert ticks == pytest.approx([248, 256, 264, 272])     # internal boundaries only


def test_ferret_colour_key_is_boxed_vertical_labelled_at_every_level(gv):
    fig, _ = _draw(gv, {"figure": {"look": "ferret"}, "layers": [
        {"kind": "contourf", "var": "t",
         "options": {"levels": "240, 250, 260, 270, 280"}}]})
    cb = _colorbars(fig)[0]
    assert cb.orientation == "vertical" and cb.drawedges
    assert list(cb.get_ticks()) == pytest.approx([240, 250, 260, 270, 280])


def test_an_explicit_colorbar_side_wins_over_the_look(gv):
    fig, _ = _draw(gv, {"figure": {"look": "ferret", "colorbar": "bottom"},
                        "layers": [{"kind": "pcolormesh", "var": "t"}]})
    assert _colorbars(fig)[0].orientation == "horizontal"


@pytest.mark.parametrize("projection", ["Orthographic", "LambertConformal", "PlateCarree"])
def test_subtitle_and_footnotes_stay_inside_the_figure(gv, projection):
    fig, _ = _draw(gv, {"figure": {"projection": projection, "subtitle": "a subtitle",
                                   "footnote_left": "source: test",
                                   "footnote_right": "run: 1"},
                        "layers": [{"kind": "pcolormesh", "var": "t"}]})
    renderer = fig.canvas.get_renderer()
    texts = [fig._suptitle, fig._supxlabel] + [t for t in fig.artists
                                               if hasattr(t, "get_text")]
    assert "a subtitle" in fig._suptitle.get_text()
    assert {"source: test", "run: 1"} <= {t.get_text() for t in texts}
    for text in texts:
        box = text.get_window_extent(renderer)
        assert box.x0 >= -1 and box.y0 >= -1
        assert box.x1 <= fig.bbox.width + 1 and box.y1 <= fig.bbox.height + 1


def test_a_gif_freezes_band_ranges(gv):
    stack = gv.layer_stack({"layers": [{"kind": "pcolormesh", "var": "t",
                                        "options": {"bands": 6}}]}, "t")
    frozen = L.freeze_ranges(gv, stack, 0, (0, 58, 20, 60))
    opts = frozen["layers"][0]["options"]
    assert opts["bands"] == 6 and opts["vmin"] < opts["vmax"]
