"""Centring the fast map's colours on a value, and a heavier zero contour.

Anomalies, differences, w and u/v are read against 0: the colormap's middle
has to sit at 0, or the eye reads "warm" where the field is merely above the
frame's 2nd percentile. `center` is the layer option of the same name brought
to the fast map. Off by default, so nobody's frames change unasked.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from gmpas import colour as _colour
from gmpas import layers as L
from gmpas.generic import GenericViewer
from gmpas.viewer import Viewer

EXTENT = (-40, 60, 20, 70)


@pytest.fixture
def anomaly(tmp_path):
    """A field from -2 to +6: its own range is lopsided about 0."""
    lat, lon = np.linspace(30, 60, 16), np.linspace(-20, 40, 31)
    _, lo = np.meshgrid(lat, lon, indexing="ij")
    v = np.stack([-2 + 8 * (lo + 20) / 60, -2 + 8 * (lo + 20) / 60])
    xr.Dataset({"a": (("time", "lat", "lon"), v, {"units": "K"})},
               coords={"time": pd.date_range("2024", periods=2), "lat": lat, "lon": lon}
               ).to_netcdf(tmp_path / "a.nc")
    gv = GenericViewer(tmp_path / "a.nc")
    yield gv
    gv.close()


@pytest.fixture
def mpas(tmp_path):
    from conftest import write_mesh

    run = tmp_path / "run"
    run.mkdir()
    write_mesh(run / "history.2012-02-25_00.00.00.nc",
               [(0.0, 0.0), (10.0, 0.0), (5.0, 8.0), (-6.0, 4.0)])
    v = Viewer(run, nx=80, ny=60)
    yield v
    v.series.close()


def test_center_is_offered_with_the_other_colour_options(anomaly, mpas):
    assert "center" in anomaly.describe()["colour_options"]
    assert "center" in mpas.describe()["colour_options"]


@pytest.mark.parametrize("colour", [None, {}, {"reverse": False}])
def test_without_center_the_frame_is_unchanged_byte_for_byte(anomaly, colour):
    """An unset center must not move a byte: same frame, same range."""
    with_null = None if colour is None else {**colour, "center": None}
    a = anomaly.frame("a", 0, 0, EXTENT, "viridis", None, None, 100, 50,
                      colour=json.dumps(colour) if colour is not None else None)
    b = anomaly.frame("a", 0, 0, EXTENT, "viridis", None, None, 100, 50,
                      colour=json.dumps(with_null) if with_null is not None else None)
    assert a == b


def test_center_makes_the_generic_range_symmetric_and_says_so(anomaly):
    plain = anomaly.frame("a", 0, 0, EXTENT, "RdBu_r", None, None, 100, 50,
                          colour=json.dumps({"reverse": False}))
    meta = {}
    _, lo, hi = anomaly.frame("a", 0, 0, EXTENT, "RdBu_r", None, None, 100, 50,
                              colour=json.dumps({"center": 0}), meta=meta)
    assert lo == pytest.approx(-hi) and hi == pytest.approx(max(-plain[1], plain[2]))
    bar = meta["colorbar"]
    assert bar["lo"] == pytest.approx(lo) and bar["hi"] == pytest.approx(hi)


def test_center_makes_the_mpas_range_symmetric(mpas):
    field = mpas.values("areaCell", 0, 0)
    mid = float(np.median(field))
    _, lo, hi = mpas.frame("areaCell", 0, 0, (-30, 40, -20, 30), "RdBu_r", None, None,
                           80, 60, colour=json.dumps({"center": mid}))
    assert lo + hi == pytest.approx(2 * mid)


def test_a_range_typed_at_both_ends_is_left_alone(anomaly):
    _, lo, hi = anomaly.frame("a", 0, 0, EXTENT, "viridis", -1.0, 5.0, 100, 50,
                              colour=json.dumps({"center": 0}))
    assert (lo, hi) == (-1.0, 5.0)


def test_center_cannot_be_combined_with_a_log_scale(anomaly):
    with pytest.raises(ValueError, match="center and norm=log"):
        _colour.clean(json.dumps({"center": 0, "norm": "log"}))


def test_the_centring_rule_is_the_layers_rule():
    assert _colour.centred(-2.0, 6.0, {"center": 0}) == (-6.0, 6.0)
    assert _colour.centred(1.0, 3.0, {"center": 0}) == (-3.0, 3.0)
    assert _colour.centred(-2.0, 6.0, {"center": 1}) == (-4.0, 6.0)
    assert _colour.centred(-2.0, 6.0, {}) == (-2.0, 6.0)


# ---------------------------------------------------------------- contours


def _contour(field, **options):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        artist = ax.contour(field, levels=[-2, -1, 0, 1, 2], colors="k", linewidths=1.0)
        opts = {k: v["default"] for k, v in L.LAYER_KINDS["contour"]["options"].items()}
        opts.update(options)
        L._emphasise(artist, opts, False)
        return list(np.atleast_1d(artist.get_linewidth())), list(artist.levels)
    finally:
        plt.close(fig)


FIELD = np.add.outer(np.linspace(-3, 3, 20), np.zeros(20))


def test_the_zero_contour_is_drawn_heavy_when_asked():
    widths, levels = _contour(FIELD, zero_bold=True, highlight_linewidth=2.5)
    zero = levels.index(0)
    assert widths[zero] == 2.5
    assert all(w == 1.0 for i, w in enumerate(widths) if i != zero)


def test_the_zero_contour_is_ordinary_by_default():
    widths, _ = _contour(FIELD)
    assert all(w == 1.0 for w in widths)


def test_negative_contours_are_dashed_by_default():
    """Already so (negative_linestyles="dashed"); pinned here with the zero line,
    since together they are the Ferret/GrADS convention."""
    assert L.LAYER_KINDS["contour"]["options"]["negative_linestyles"]["default"] == "dashed"
