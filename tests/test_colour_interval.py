"""A colour step every N units: edges on multiples of an interval.

`bands` splits the range into equal fractions, so its edges land on numbers
like 271.38; an interval puts them on 0, 2.5, 5 ... (the GrADS/Ferret rule),
and an edge then means the same value in every frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from gmpas import colour, palettes
from gmpas.generic import GenericViewer


def test_edges_sit_on_multiples_and_round_outwards():
    e = palettes.interval_edges(0.3, 28.9, 2.5)
    assert e[:-1].tolist() == [2.5 * k for k in range(12)]
    assert e[-1] > 30.0 and np.nextafter(e[-1], -np.inf) == 30.0
    e = palettes.interval_edges(-3.2, 7.1, 1.0)
    assert e[0] == -4.0 and np.nextafter(e[-1], -np.inf) == 8.0


def test_an_exact_multiple_is_not_rounded_past():
    e = palettes.interval_edges(0.0, 30.0, 2.5)
    assert e[0] == 0.0 and np.nextafter(e[-1], -np.inf) == 30.0 and e.size == 13


def test_too_many_steps_say_so():
    with pytest.raises(ValueError, match="at most 252"):
        palettes.interval_edges(0.0, 1000.0, 0.1)


@pytest.mark.parametrize("opts, why", [
    ({"interval": 2.0, "bands": 8}, "cannot be combined"),
    ({"interval": 2.0, "norm": "log"}, "linear scale"),
    ({"interval": 0.0}, "positive"),
])
def test_contradictions_are_refused(opts, why):
    with pytest.raises(ValueError, match=why):
        colour.clean(opts)


@pytest.fixture
def gv(tmp_path):
    lat, lon = np.linspace(-10, 10, 21), np.linspace(0, 20, 21)
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    v = (0.3 + 28.6 * (lo - lo.min()) / np.ptp(lo))[None].astype("f4")
    xr.Dataset({"sst": (("time", "lat", "lon"), v)},
               coords={"time": pd.date_range("2024-01-01", periods=1),
                       "lat": lat, "lon": lon}).to_netcdf(tmp_path / "s.nc")
    return GenericViewer(tmp_path / "s.nc", strict=True)


def test_the_colour_key_shows_the_interval_edges(gv):
    meta = {}
    gv.frame("sst", 0, 0, gv.home, "viridis", None, None, colour={"interval": 2.5},
             meta=meta)
    assert meta["colorbar"]["edges"] == [2.5 * k for k in range(13)]


def test_bands_still_label_the_range_end(gv):
    meta = {}
    gv.frame("sst", 0, 0, gv.home, "viridis", 0.0, 30.0, colour={"bands": 6}, meta=meta)
    assert meta["colorbar"]["edges"] == [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]


def test_no_interval_draws_what_it_always_drew(gv):
    plain = gv.frame("sst", 0, 0, gv.home, "viridis", None, None)
    again = gv.frame("sst", 0, 0, gv.home, "viridis", None, None, colour={})
    assert plain[0] == again[0]


def test_a_figure_with_an_interval_draws(gv):
    png = gv.plot("sst", 0, 0, "contourf", gv.home, 400, 300, colour={"interval": 5.0})
    assert png[:4] == b"\x89PNG"
