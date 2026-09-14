"""Palettes for --generic: the Ferret parser, GrADS colours, the registry."""

from __future__ import annotations

from importlib import resources

import numpy as np
import pytest

import gmpas.palettes as P
from gmpas.palettes import grads, spk


def _rgb255(cmap, x):
    return [round(v * 255) for v in cmap(x)[:3]]


# ------------------------------------------------------------------ .spk


def test_a_percent_palette_interpolates_between_positions():
    cmap = spk.parse("RGB_Mapping Percent\n0 0 0 0\n100 100 0 100\n", "t")
    assert _rgb255(cmap, 0.0) == [0, 0, 0]
    assert _rgb255(cmap, 1.0) == [255, 0, 255]
    assert _rgb255(cmap, 0.5) == pytest.approx([128, 0, 128], abs=1)


def test_no_header_means_percent_and_comments_are_ignored():
    cmap = spk.parse("! a comment\n 0.  100. 0. 0.   ! red\n100.  0. 0. 100.\n", "t")
    assert _rgb255(cmap, 0.0) == [255, 0, 0] and _rgb255(cmap, 1.0) == [0, 0, 255]


def test_a_repeated_position_is_a_hard_step():
    cmap = spk.parse("0 100 0 0\n50 100 0 0\n50 0 0 100\n100 0 0 100\n", "t")
    assert _rgb255(cmap, 0.49) == [255, 0, 0]
    assert _rgb255(cmap, 0.51) == [0, 0, 255]


def test_palettes_not_spanning_0_to_100_are_padded_with_their_end_colours():
    cmap = spk.parse("20 100 0 0\n80 0 0 100\n", "t")
    assert _rgb255(cmap, 0.0) == [255, 0, 0] and _rgb255(cmap, 1.0) == [0, 0, 255]


def test_a_by_level_palette_is_one_colour_per_level_never_blended():
    from matplotlib.colors import ListedColormap

    cmap = spk.parse("RGB_Mapping By_level\n\n2 0 100 0\n1 100 0 0\n3 0 0 100\n", "t")
    assert isinstance(cmap, ListedColormap) and cmap.N == 3
    assert [_rgb255(cmap, i) for i in range(3)] == [[255, 0, 0], [0, 255, 0], [0, 0, 255]]


def test_a_by_value_palette_is_refused():
    with pytest.raises(spk.SpkByValue):
        spk.parse("RGB_Mapping By_Value\n0 0 0 0\n30 100 0 0\n", "t")


@pytest.mark.parametrize("text, message", [
    ("0 1 2\n", "position red green blue"),
    ("RGB_Mapping Sideways\n0 0 0 0\n", "unknown RGB_Mapping"),
    ("! nothing\n", "no colours"),
])
def test_malformed_palettes_say_what_is_wrong(text, message):
    with pytest.raises(ValueError, match=message):
        spk.parse(text, "t")


# ------------------------------------------------------------------ GrADS


def test_the_grads_rainbow_is_its_thirteen_default_colours_exactly():
    """From GrADS src/gxdb.c pdcred/pdcgre/pdcblu and the documented sequence."""
    cmap = grads.rainbow()
    assert cmap.N == 13 and cmap.name == "grads.rainbow"
    expected = [(160, 0, 200), (130, 0, 220), (30, 60, 255), (0, 160, 255), (0, 200, 200),
                (0, 210, 140), (0, 220, 0), (160, 230, 50), (230, 220, 50), (230, 175, 45),
                (240, 130, 40), (250, 60, 60), (240, 0, 130)]
    assert [tuple(_rgb255(cmap, i)) for i in range(13)] == expected


def test_the_grads_default_colour_table():
    reds = [0, 255, 250, 0, 30, 0, 240, 230, 240, 160, 160, 0, 230, 0, 130, 170]
    greens = [0, 255, 60, 220, 60, 200, 0, 220, 130, 0, 230, 160, 175, 210, 0, 170]
    blues = [0, 255, 60, 0, 255, 200, 130, 50, 40, 200, 50, 255, 45, 140, 220, 170]
    assert list(grads.COLORS) == list(zip(reds, greens, blues, strict=True))
    assert grads.default16().N == 16


# --------------------------------------------------------------- registry


def test_every_offered_palette_resolves_in_matplotlib():
    from matplotlib import colormaps

    groups = P.groups()
    assert set(groups) == {"matplotlib", "cmocean", "ferret", "grads"}
    names = [n for group in groups.values() for n in group]
    assert all(n in colormaps for n in names)
    assert len(groups["ferret"]) == len(P.FERRET) >= 25
    assert "grads.rainbow" in groups["grads"]


def test_registering_twice_is_harmless():
    P.register()
    P.register()


def test_every_vendored_ferret_palette_parses_and_is_not_by_value():
    folder = resources.files("gmpas.palettes") / "ferret"
    for stem in P.FERRET:
        cmap = spk.parse((folder / f"{stem}.spk").read_text(errors="replace"), stem)
        assert cmap.N >= 2


def test_the_ferret_notice_ships_with_the_palettes():
    notice = (resources.files("gmpas.palettes") / "ferret" / "NOTICE.txt").read_text()
    assert "public domain" in notice and "NOAA-PMEL/Ferret" in notice
    for stem in P.FERRET:
        assert f"{stem}.spk" in notice


def test_cmocean_missing_leaves_its_group_empty(monkeypatch):
    monkeypatch.setattr(P, "_import_cmocean", lambda: None)
    assert P.groups()["cmocean"] == []


def test_cmocean_is_offered_when_installed():
    pytest.importorskip("cmocean")
    assert "cmo.thermal" in P.groups()["cmocean"]


def test_get_reverses_before_setting_extremes():
    cmap = P.get("ferret.rnb2", reverse=True, under="black", over="white", bad="0.5")
    assert tuple(cmap.get_under()) == (0, 0, 0, 1)
    assert tuple(cmap.get_over()) == (1, 1, 1, 1)
    forward, backward = P.get("ferret.rnb2"), P.get("ferret.rnb2", reverse=True)
    assert np.allclose(forward(0.0), backward(1.0))


def test_an_unknown_name_is_refused():
    with pytest.raises(ValueError, match="not a known colormap"):
        P.get("__import__")


# ------------------------------------------------------------------ scale


def test_bands_keep_the_field_maximum_inside_the_last_band():
    cmap, norm, edges = P.scale({"cmap": "grads.rainbow", "bands": 13}, 0.0, 13.0)
    assert edges.size == 14 and edges[0] == 0.0 and edges[-1] > 13.0
    assert norm(13.0) == 12 and norm(0.0) == 0
    assert norm(13.5) == cmap.N                       # over the range


@pytest.mark.parametrize("opts, kind", [
    ({}, "Normalize"),
    ({"norm": "log"}, "LogNorm"),
    ({"norm": "symlog", "linthresh": 2.0}, "SymLogNorm"),
    ({"norm": "power", "gamma": 0.5}, "PowerNorm"),
])
def test_scale_builds_the_asked_norm(opts, kind):
    _, norm, edges = P.scale({"cmap": "viridis", **opts}, 1.0, 100.0)
    assert type(norm).__name__ == kind and edges is None
