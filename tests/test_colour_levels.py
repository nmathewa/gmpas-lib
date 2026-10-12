"""Colour levels: one Ferret-style expression for the colour steps.

`20` is twenty equal steps (what `bands` did), `(,,2.5)` puts them on
multiples of 2.5, `(-25,0,5)(0,25,1)` joins steps of different sizes, and
`(-inf)` / `(inf)` leave the ends open. It is parsed, never evaluated: the
page can be on the network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from gmpas import colour, palettes
from gmpas.generic import GenericViewer


def edges(text, lo=0.0, hi=1.0):
    e = palettes.level_edges(palettes.parse_levels(text), lo, hi)
    assert np.nextafter(e[-1], -np.inf) < e[-1]
    return [*e[:-1].tolist(), float(np.nextafter(e[-1], -np.inf))]


def test_a_count_is_equal_steps_like_bands():
    got = palettes.level_edges(palettes.parse_levels("6"), 0.0, 30.0)
    assert np.array_equal(got, palettes.band_edges(0.0, 30.0, 6))


def test_an_open_interval_sits_on_multiples_and_rounds_outwards():
    assert edges("(,,2.5)", 0.3, 28.9) == [2.5 * k for k in range(13)]
    assert edges("(,,1)", -3.2, 7.1)[0] == -4.0


def test_segments_join_and_count_a_shared_level_once():
    assert edges("(-25,0,5)(0,25,1)") == [-25, -20, -15, -10, -5, *range(0, 26)]


def test_single_levels_and_open_ends():
    assert edges("(-inf)(0)(1)(2)(5)(10)(inf)") == [0, 1, 2, 5, 10]
    assert palettes.parse_levels("(-inf)(0,30,2)(inf)")["open"] == (True, True)


def test_spaces_and_case_do_not_matter():
    assert edges(" ( -INF ) (0, 10, 2.5) ") == [0, 2.5, 5, 7.5, 10]


def test_a_segment_end_not_on_a_step_stops_below_it():
    assert edges("(0,10,3)") == [0, 3, 6, 9]


@pytest.mark.parametrize("text, why", [
    ("", "empty"),
    ("abc", "not understood"),
    ("__import__('os')", "not understood"),
    ("(1,2)", "not understood"),
    ("(0,10,0)", "above zero"),
    ("(10,0,1)", "hi above lo"),
    ("(0)(inf)(5)", "belongs at the end"),
    ("(5)(-inf)", "belongs at the start"),
    ("(-inf)(inf)", "levels between them"),
    ("0", "between 1 and 252"),
    ("300", "between 1 and 252"),
])
def test_bad_levels_are_refused_by_name(text, why):
    with pytest.raises(ValueError, match=why):
        palettes.parse_levels(text)


@pytest.mark.parametrize("text, why", [
    ("(5)(1)", "must increase"),
    ("(0)", "at least two"),
    ("(0,1000,0.1)", "at most 252"),
    ("(,,0.001)", "at most 252"),
])
def test_levels_that_make_no_key_say_why(text, why):
    with pytest.raises(ValueError, match=why):
        palettes.level_edges(palettes.parse_levels(text), 0.0, 30.0)


@pytest.mark.parametrize("opts, why", [
    ({"levels": "(,,2)", "bands": 8}, "cannot be combined"),
    ({"levels": "(,,2)", "norm": "log"}, "set the steps"),
    ({"levels": "nope"}, "colour: levels"),
])
def test_contradictions_are_refused(opts, why):
    with pytest.raises(ValueError, match=why):
        colour.clean(opts)


def test_levels_with_named_ends_are_the_range():
    assert colour.centred(3.0, 9.0, {"levels": "(-25,0,5)(0,25,1)"}, 1.0, 2.0) == (-25, 25)
    assert colour.centred(3.0, 9.0, {"levels": "(,,2)"}) == (3.0, 9.0)
    assert colour.centred(3.0, 9.0, {"levels": "(0,,2)"}) == (0.0, 9.0)


def test_open_ends_draw_the_key_triangles():
    assert palettes.extend_of({"levels": "(-inf)(0,1,0.5)"}) == "min"
    assert palettes.extend_of({"levels": "(0,1,0.5)(inf)", "extend": "min"}) == "both"
    assert palettes.extend_of({"extend": "max"}) == "max"


def test_bands_is_hidden_from_the_page_but_still_works():
    assert colour.OPTIONS["bands"]["hidden"] is True
    assert colour.clean({"bands": 6}) == {"bands": 6}


@pytest.fixture
def gv(tmp_path):
    lat, lon = np.linspace(-10, 10, 21), np.linspace(0, 20, 21)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    v = (-20 + 40 * (lo - lo.min()) / np.ptp(lo))[None].astype("f4")
    xr.Dataset({"sst": (("time", "lat", "lon"), v)},
               coords={"time": pd.date_range("2024-01-01", periods=1),
                       "lat": lat, "lon": lon}).to_netcdf(tmp_path / "s.nc")
    return GenericViewer(tmp_path / "s.nc", strict=True)


def test_uneven_levels_reach_the_frame_and_its_key(gv):
    meta = {}
    png, lo, hi = gv.frame("sst", 0, 0, gv.home, "viridis", None, None,
                           colour={"levels": "(-inf)(-10,0,5)(0,10,1)(inf)"}, meta=meta)[:3]
    bar = meta["colorbar"]
    assert bar["edges"] == [-10.0, -5.0, *map(float, range(0, 11))]
    assert (lo, hi) == (-10.0, 10.0)
    assert bar["extend"] == "both" and bar["under"] and bar["over"]
    assert len(bar["stops"]) == 12


def test_an_uneven_step_is_coloured_by_its_band(gv):
    img = np.array([[-7.0, -1.0, 0.5, 9.5, 15.0, -15.0]])
    from gmpas.palettes import encode
    idx, _, _ = encode.indices(img, {"levels": "(-10,0,5)(0,10,1)"}, -10.0, 10.0)
    assert idx[0].tolist() == [0, 1, 2, 11, encode.OVER, encode.UNDER]


def test_no_levels_draws_what_it_always_drew(gv):
    plain = gv.frame("sst", 0, 0, gv.home, "viridis", None, None)
    again = gv.frame("sst", 0, 0, gv.home, "viridis", None, None, colour={})
    assert plain[0] == again[0]


@pytest.mark.parametrize("kind", ["contourf", "map"])
def test_a_figure_with_levels_draws(gv, kind):
    png = gv.plot("sst", 0, 0, kind, gv.home, 400, 300,
                  colour={"levels": "(-inf)(-10,0,5)(0,10,1)(inf)"})
    assert png[:4] == b"\x89PNG"
