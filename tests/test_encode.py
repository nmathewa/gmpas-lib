"""The fast map's colour-option encoder, checked against matplotlib itself."""

from __future__ import annotations

import io

import numpy as np
import pytest

from gmpas import palettes
from gmpas.palettes import encode


def _decode(png: bytes):
    from PIL import Image

    im = Image.open(io.BytesIO(png))
    idx = np.asarray(im)[::-1]                       # back to row 0 = south
    pal = np.asarray(im.getpalette()[:768], dtype=np.uint8).reshape(256, 3)
    return im, idx, pal


def _matplotlib_rgb(values, opts, lo, hi):
    """What a matplotlib figure with the same options paints each value."""
    from matplotlib.cm import ScalarMappable

    cmap, norm, _ = palettes.scale(opts, lo, hi)
    return np.round(ScalarMappable(norm=norm, cmap=cmap).to_rgba(values)[..., :3] * 255)


FIELD = np.array([[-5.0, 0.0, 2.5, 5.0, 7.5, 10.0, 12.0, np.nan]])


def test_reserved_indices_for_under_over_missing_and_off_grid():
    outside = np.zeros(FIELD.shape, bool)
    outside[0, 6] = True                                     # 12.0, but off the grid
    opts = {"cmap": "viridis", "missing_color": "red"}
    png, spec = encode.png(FIELD, opts, 0.0, 10.0, outside=outside)
    im, idx, pal = _decode(png)
    assert im.mode == "P" and im.info["transparency"] == 255
    assert idx[0, 0] == encode.UNDER and idx[0, 7] == encode.MISSING
    assert idx[0, 6] == encode.CLEAR
    assert all(idx[0, i] < encode.DATA for i in range(1, 6))
    assert tuple(pal[encode.MISSING]) == (255, 0, 0)


def test_missing_cells_stay_transparent_without_a_colour():
    png, _ = encode.png(FIELD, {"cmap": "viridis"}, 0.0, 10.0)
    _, idx, _ = _decode(png)
    assert idx[0, 7] == encode.CLEAR


def test_over_the_range_uses_the_over_colour():
    png, spec = encode.png(np.array([[11.0]]), {"cmap": "viridis", "over_color": "white",
                                                "extend": "max"}, 0.0, 10.0)
    _, idx, pal = _decode(png)
    assert idx[0, 0] == encode.OVER and tuple(pal[encode.OVER]) == (255, 255, 255)
    assert spec["over"] == "#ffffff" and spec["under"] is None


@pytest.mark.parametrize("opts", [
    {"cmap": "viridis"},
    {"cmap": "ferret.rnb2", "reverse": True},
    {"cmap": "grads.rainbow", "norm": "power", "gamma": 0.4},
    {"cmap": "viridis", "norm": "log"},
])
def test_continuous_colours_match_matplotlib_to_a_quantisation_step(opts):
    lo, hi = (1.0, 100.0) if opts.get("norm") == "log" else (0.0, 10.0)
    values = np.linspace(lo, hi, 400).reshape(20, 20)
    png, _ = encode.png(values, opts, lo, hi)
    _, idx, pal = _decode(png)
    ours = pal[idx].astype(float)
    reference = _matplotlib_rgb(values, opts, lo, hi)
    # 252 palette steps against matplotlib's 256: neighbouring entries differ by a few
    assert np.abs(ours - reference).max() <= 12


@pytest.mark.parametrize("extend", ["neither", "both"])
def test_band_colours_match_matplotlib_exactly(extend):
    opts = {"cmap": "grads.rainbow", "bands": 13, "extend": extend}
    values = np.array([[0.0, 0.5, 6.4, 6.5, 12.99, 13.0]])
    png, spec = encode.png(values, opts, 0.0, 13.0)
    _, idx, pal = _decode(png)
    assert np.array_equal(pal[idx].astype(float), _matplotlib_rgb(values, opts, 0.0, 13.0))
    assert idx[0, -1] == 12                                  # the maximum: last band
    assert len(spec["stops"]) == 13 and spec["edges"][-1] == 13.0


def test_a_short_listed_palette_is_indexed_exactly_like_matplotlib():
    opts = {"cmap": "grads.rainbow"}
    values = np.linspace(0.0, 10.0, 1000).reshape(40, 25)
    png, _ = encode.png(values, opts, 0.0, 10.0)
    _, idx, pal = _decode(png)
    assert np.array_equal(pal[idx].astype(float), _matplotlib_rgb(values, opts, 0.0, 10.0))


def test_more_bands_than_palette_colours_repeat_colours():
    opts = {"cmap": "grads.rainbow", "bands": 26}
    png, spec = encode.png(np.linspace(0, 26, 52).reshape(2, 26), opts, 0.0, 26.0)
    _, idx, pal = _decode(png)
    colours = {tuple(c) for c in pal[:26]}
    assert len(spec["stops"]) == 26 and len(colours) == 13
