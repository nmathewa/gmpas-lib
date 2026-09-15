"""Contour levels by interval, and every Nth line emphasised (#90, #91)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from gmpas import layers as L
from gmpas.generic import GenericViewer

LAT = np.arange(60, 19.5, -2.0)
LON = np.arange(0, 60, 2.0)

SPATIAL = ["mslp"]
COUNTS = {"mslp": 1}


@pytest.fixture
def gv(tmp_path):
    """Sea-level pressure from 979 to 1021 hPa: a field a forecaster would draw
    every 4 hPa from 1000."""
    import matplotlib
    matplotlib.use("Agg")

    la, lo = np.meshgrid(LAT, LON, indexing="ij")
    field = 1000.0 + 21.0 * np.cos(np.deg2rad(3 * lo)) * np.cos(np.deg2rad(2 * la))
    mslp = np.stack([field, field + 0.5])
    xr.Dataset({"mslp": (("time", "latitude", "longitude"), mslp, {"units": "hPa"})},
               coords={"time": pd.date_range("2024-01-01", periods=2),
                       "latitude": ("latitude", LAT, {"units": "degrees_north"}),
                       "longitude": ("longitude", LON, {"units": "degrees_east"})}
               ).to_netcdf(tmp_path / "mslp.nc")
    viewer = GenericViewer(tmp_path / "mslp.nc")
    yield viewer
    viewer.close()


@pytest.fixture(autouse=True)
def close_figures():
    """Each test draws at least one figure; twenty left open is a warning, and
    the suite turns warnings into errors for whichever test comes next."""
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


def _draw(gv, layer, extent=(0, 58, 20, 60)):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 6), dpi=60, layout="constrained")
    stack = gv.layer_stack({"layers": [layer]}, "mslp")
    ax = L.draw(gv, fig, stack, 0, 0, extent)
    fig.canvas.draw()
    return fig, ax


def _lines(ax):
    return next(c for c in ax.collections if getattr(c, "filled", True) is False)


def _clean(kind, options):
    return L.clean({"layers": [{"kind": kind, "var": "mslp", "options": options}]},
                   SPATIAL, COUNTS)


# ------------------------------------------------- levels from an interval


def test_an_interval_puts_levels_on_the_reference_value():
    levels = L.interval_levels(979.0, 1021.0, 4.0, 1000.0)
    assert levels[:3] == [980.0, 984.0, 988.0]
    assert 1000.0 in levels and levels[-1] == 1020.0


def test_a_reference_outside_the_data_still_sets_where_the_levels_fall():
    assert L.interval_levels(11.0, 29.0, 5.0, 1000.0) == [15.0, 20.0, 25.0]


def test_the_range_ends_are_levels_when_they_land_on_one():
    assert L.interval_levels(980.0, 1000.0, 10.0, 1000.0) == [980.0, 990.0, 1000.0]


def test_min_and_max_level_bound_the_set():
    levels = L.interval_levels(979.0, 1021.0, 4.0, 1000.0, min_level=1000.0,
                               max_level=1012.0)
    assert levels == [1000.0, 1004.0, 1008.0, 1012.0]


@pytest.mark.parametrize("args, message", [
    (dict(lo=0.0, hi=1000.0, interval=0.5, reference=0.0), "the limit is 256"),
    (dict(lo=1.0, hi=9.0, interval=10.0, reference=100.5), "no level between"),
    (dict(lo=0.0, hi=100.0, interval=5.0, reference=0.0, min_level=90.0,
          max_level=95.0), None),
])
def test_an_impossible_interval_says_what_to_change(args, message):
    if message is None:
        assert L.interval_levels(**args) == [90.0, 95.0]
        return
    with pytest.raises(ValueError, match=message):
        L.interval_levels(**args)


@pytest.mark.parametrize("kind", ["contour", "contourf"])
def test_the_drawn_levels_are_the_interval_levels(gv, kind):
    fig, ax = _draw(gv, {"kind": kind, "var": "mslp",
                         "options": {"interval": 4, "reference": 1000}})
    artist = next(c for c in ax.collections if hasattr(c, "levels"))
    levels = list(artist.levels)
    assert 1000.0 in levels
    assert all(abs((v - 1000.0) / 4.0 - round((v - 1000.0) / 4.0)) < 1e-9 for v in levels)


def test_interval_levels_hold_still_while_the_field_moves(gv):
    """The point of an interval: step 1 is half a hPa warmer, and a count would
    shift every line, but 1000 hPa stays 1000 hPa."""
    stack = gv.layer_stack({"layers": [{"kind": "contour", "var": "mslp", "options": {
        "interval": 4, "reference": 1000}}]}, "mslp")
    import matplotlib.pyplot as plt

    drawn = []
    for step in (0, 1):
        fig = plt.figure()
        ax = L.draw(gv, fig, stack, step, 0, (0, 58, 20, 60))
        drawn.append(list(_lines(ax).levels))
        plt.close(fig)
    assert drawn[0] == drawn[1]


def test_a_gif_freezes_the_interval_levels_too(gv):
    stack = gv.layer_stack({"layers": [{"kind": "contour", "var": "mslp", "options": {
        "interval": 4, "reference": 1000}}]}, "mslp")
    frozen = L.freeze_ranges(gv, stack, 0, (0, 58, 20, 60))
    opts = frozen["layers"][0]["options"]
    assert opts["interval"] == 4 and opts["vmin"] < opts["vmax"]


# -------------------------------------------------------------- validation


@pytest.mark.parametrize("options, message", [
    ({"interval": 4, "levels": "980, 1000"}, "interval and levels"),
    ({"interval": 0}, "must be above zero"),
    ({"interval": 4, "min_level": 1010, "max_level": 1000}, "must be above min_level"),
    ({"min_level": 1000}, "applies to interval levels"),
    ({"max_level": 1000}, "applies to interval levels"),
])
def test_contradictory_level_options_are_refused_by_name(options, message):
    with pytest.raises(ValueError, match=message):
        _clean("contour", options)


def test_too_many_levels_are_refused_when_the_layer_draws(gv):
    with pytest.raises(ValueError, match="the limit is 256"):
        _draw(gv, {"kind": "contour", "var": "mslp", "options": {"interval": 0.01}})


def test_highlighting_is_a_line_option_only():
    with pytest.raises(ValueError, match="unknown option"):
        _clean("contourf", {"highlight_every": 5})


# ------------------------------------------------------------ highlighting


def test_every_nth_line_is_drawn_heavier_counted_from_the_reference(gv):
    fig, ax = _draw(gv, {"kind": "contour", "var": "mslp", "options": {
        "interval": 4, "reference": 1000, "highlight_every": 5,
        "linewidths": 0.5, "highlight_linewidth": 2.5}})
    lines = _lines(ax)
    widths = dict(zip([float(v) for v in lines.levels],
                      np.atleast_1d(lines.get_linewidth()), strict=True))
    # every fifth 4 hPa step from 1000 is heavy wherever the field reaches
    assert widths[1000.0] == 2.5 and widths[1004.0] == 0.5 and widths[996.0] == 0.5
    for value, width in widths.items():
        every_fifth = round((value - 1000.0) / 4.0) % 5 == 0
        assert width == (2.5 if every_fifth else 0.5), value


def test_highlighted_lines_take_their_own_colour(gv):
    from matplotlib.colors import to_rgba

    fig, ax = _draw(gv, {"kind": "contour", "var": "mslp", "options": {
        "interval": 4, "reference": 1000, "highlight_every": 5,
        "colors": "gray", "highlight_color": "red"}})
    lines = _lines(ax)
    colours = dict(zip([float(v) for v in lines.levels],
                       [tuple(c) for c in lines.get_edgecolor()], strict=True))
    assert colours[1000.0] == to_rgba("red")
    assert colours[1004.0] == to_rgba("gray")


def test_without_an_interval_every_nth_line_counts_from_the_lowest(gv):
    fig, ax = _draw(gv, {"kind": "contour", "var": "mslp", "options": {
        "levels": "980, 990, 1000, 1010, 1020", "highlight_every": 2,
        "linewidths": 0.5, "highlight_linewidth": 3.0}})
    widths = list(np.atleast_1d(_lines(ax).get_linewidth()))
    assert widths == [3.0, 0.5, 3.0, 0.5, 3.0]


def test_highlight_off_leaves_one_width_for_every_line(gv):
    fig, ax = _draw(gv, {"kind": "contour", "var": "mslp",
                         "options": {"interval": 4, "linewidths": 0.8}})
    assert set(np.atleast_1d(_lines(ax).get_linewidth())) == {0.8}


# ----------------------------------------------------------------- labels


def _labels(ax):
    return sorted(float(t.get_text().replace("−", "-")) for t in ax.texts)


def test_only_every_nth_level_is_labelled(gv):
    fig, ax = _draw(gv, {"kind": "contour", "var": "mslp", "options": {
        "interval": 4, "reference": 1000, "label_every": 5, "label_fmt": "%g"}})
    drawn = [float(v) for v in _lines(ax).levels]
    labelled = sorted(set(_labels(ax)))
    assert labelled == drawn[::5]                  # every fifth line, from the lowest


def test_labels_land_on_the_highlighted_lines(gv):
    fig, ax = _draw(gv, {"kind": "contour", "var": "mslp", "options": {
        "interval": 4, "reference": 1000, "highlight_every": 5, "label_every": 5}})
    lines = _lines(ax)
    heavy = {float(v) for v, w in zip(lines.levels,
                                      np.atleast_1d(lines.get_linewidth()), strict=True)
             if w > 1.0}
    assert set(_labels(ax)) <= heavy


def test_labels_off_stays_off(gv):
    fig, ax = _draw(gv, {"kind": "contour", "var": "mslp", "options": {
        "interval": 4, "labels": False, "label_every": 2}})
    assert len(ax.texts) == 0
