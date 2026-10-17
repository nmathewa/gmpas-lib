"""The chat model test's own cases and judging, checked without a model.

`chat_eval.py` asks a real model and is run by hand. Here a stand-in gives the
replies, so what is tested is that a right answer is called right and a wrong
one wrong -- a case that judged badly would blame the model for it.
"""

from __future__ import annotations

import pytest

import chat_eval as E
from gmpas import chat, script

CFG = {"url": "http://nowhere", "model": "stand-in", "timeout": 5.0, "log": "off"}


def _case(ask: str) -> dict:
    return next(c for c in E.CASES if c["ask"] == ask)


def _answer(ask: str, *replies: str) -> tuple[dict, list[str]]:
    left = list(replies)
    case = _case(ask)
    got = E.attempt(case, CFG, post=lambda cfg, messages: left.pop(0))
    return got, E.judge(case, got)


def test_every_case_starts_from_a_view_its_file_has():
    ids = [c["id"] for c in E.CASES]
    assert len(ids) == len(set(ids)) and len(ids) > 100
    for case in E.CASES:
        state = script.resolve(script.parse(case["view"]), E.META[case["ctx"]])
        assert state.get("var"), case["id"]
        assert all(set(turn) == {"request", "reply"} for turn in case["history"])
    assert {c["ctx"] for c in E.CASES} == set(E.META)
    core = E.chosen()
    assert 60 <= len(core) < len(E.chosen(everything=True)) == len(E.CASES)
    assert {c["group"] for c in E.chosen(only=["typo", "place-01"])} == {"typo", "place"}
    assert len(E.chosen(limit=7)) == 7


@pytest.mark.parametrize("ask, reply, right", [
    ("plot snow", "var = snow", True),
    ("plot snow", "shade(snowh.isel(time=0))", True),
    ("plot snow", "var = sst", False),
    ("plot snnow", "var = snow", True),
    ("show me ohio", "region = -85, -80.5, 38.4, 42", True),
    ("show me ohio", "view(region=(-84.9, -80.4, 38.3, 42.0))\nshade(h_oml)", True),
    ("show me ohio", "region = -180, 180, -90, 90", False),
    ("show me ohio", "region = -125, -115, 32, 42", False),        # California
    ("antarctica", "region = -180, 180, -90, -60", True),
    ("antarctica", "region = -180, 180, -90, 90", False),
    ("show me the whole world", "region = -180, 180, -90, 90", True),
    ("show me the whole world", "cmap = plasma", False),
    ("set gxout contour; d skintemp", "contour(skintemp)", True),
    ("set gxout contour; d skintemp", "var = skintemp", False),
    ("d sst-273.15", "shade(sst.isel(time=0) - 273.15)", True),
    ("d mag(hu_oml,hv_oml)", "var = hypot(hu_oml, hv_oml)", True),
    ("ds.skintemp.max()", "skintemp.max()", True),
    ("ds.skintemp.max()", "var = skintemp", False),
    ("use a red-blue colormap", "cmap = RdBu_r", True),
    ("use a red-blue colormap", "cmap = viridis", False),
    ("reset the colour range", "vmin =\nvmax =", True),
    ("reset the colour range", "vmin = 0", False),
    ("20 colour levels", "levels = 20", True),
    ("ferret style", "cmap = ferret.default", True),
    ("ferret style", 'figure(look="ferret")\nshade(sst)', True),
    ("show it as a globe over india", 'view(proj="Orthographic", lon0=80, lat0=22)', True),
    ("show it as a globe over india",
     'view(proj="Orthographic", lon0=-100, lat0=40)', False),
    ("back to the normal flat map", "proj = lonlat", True),
    ("sst with skintemp contours on top", "shade(sst)\ncontour(skintemp)", True),
    ("sst with skintemp contours on top", "shade(skintemp)\ncontour(sst)", False),
    ("remove the contours",
     "view(region=(95, 150, -10, 10))\npcolormesh(sst.isel(time=0))", True),
    ("remove the contours", "contour(skintemp)\npcolormesh(sst)", False),
    ("move the colorbar to the right",
     'pcolormesh(sst)\ncontour(skintemp)\nfigure(colorbar="right")', True),
    # changing one thing about a figure must not lose the rest of it
    ("move the colorbar to the right", 'var = sst\nfigure(colorbar="right")', False),
    ("change the contours to degrees celsius",
     "pcolormesh(sst)\ncontour(skintemp.isel(time=0) - 273.15, labels=True)", True),
    ("give it the title Snow test",
     'pcolormesh(sst)\ncontour(skintemp)\nfigure(title="Snow test")', True),
    ("where is the maximum snow", "snow.max()", True),
    ("where is the maximum snow", "shade(snow)", False),
    ("where is the maximum snow", "snow.max()\nvar = snow", False),      # changed the view
    ("plot the wind", "# this file has no wind", True),
    ("plot the wind", "var = hu_oml", False),
    ("now the depth instead", "var = snowh", True),
    ("and now india", "region = 68, 98, 6, 36", True),
    ("zoom out a bit", "region = -90, -75, 35, 45", True),
    ("zoom out a bit", "region = -84, -81, 39, 41", False),
    ("next time step", "time = 1", True),
    ("next time step", "time = 2", False),
    ("temperature with wind arrows on top", "shade(t)\nvectors(u, v)", True),
    ("histogram of temperature", "hist(t)", True),
    ("time series of temperature at 147E 2S", "series(t, lon=147, lat=-2)", True),
    ("where is the strongest wind", "hypot(u, v).max()", True),
    ("geopotential height in metres", "var = z / 9.80665", True),
    ("temperature at 850 hPa", "shade(t.sel(level=850))", True),
    ("temperature at 850 hPa", "level = 5", True),
    ("temperature at 850 hPa", "level = 3", False),
    ("reverse the colours", "cmap = viridis_r", True),
    ("back to the normal flat map", 'view(proj="PlateCarree")', True),
    # a still figure nobody asked for is not a right answer to a plain request
    ("plot snow", 'figure(title="Snow")\nshade(snow)', False),
    ("sst in celsius", "shade(sst - 273.15)", True),
    ("sst in celsius", 'figure(title="SST")\nshade(sst - 273.15)', False),
    # the part that can be done, and a line saying what was left out
    ("average sst over the indian ocean",
     "# an average is not possible yet; showing sst there\nvar = sst\n"
     "region = 20, 120, -40, 30", True),
    ("average sst over the indian ocean", "var = sst\nregion = 20, 120, -40, 30", False),
])
def test_a_right_answer_is_right_and_a_wrong_one_wrong(ask, reply, right):
    got, wrong = _answer(ask, reply, reply)
    assert (not wrong) is right, (got.get("script") or got.get("error"), wrong)


def test_saying_no_is_right_only_where_no_is_the_answer():
    assert _answer("plot snow", "# cannot")[1] == ["said: cannot"]
    # the checker's own refusal counts where the case names its reason
    got, wrong = _answer("time series of sst at miami",
                         "series(sst, lon=-80, lat=25.8)", "series(sst, lon=-80, lat=25.8)")
    assert got["status"] == "refused" and not wrong
    got, wrong = _answer("plot the wind", "var = wind", "var = wind")
    assert got["status"] == "refused" and wrong       # refused, but not for saying no


def test_a_run_counts_tries_and_stops_when_the_model_stops_answering():
    replies = iter(["var = snowz", "var = snow", "var = xice"])
    lines: list[str] = []
    rows = E.run([_case("plot snow"), _case("show sst")], CFG, say=lines.append,
                 post=lambda cfg, messages: next(replies))
    assert [(r["ok"], r["rounds"]) for r in rows] == [(True, 2), (False, 1)]
    assert lines[0].startswith("ok   plain-01") and lines[1].startswith("FAIL ")
    text = E.summary(rows)
    assert "3 requests to the model" in text and "var is 'xice'" in text

    def dead(cfg, messages):
        raise chat.ChatError("no answer from the model server", "server")
    rows = E.run(E.chosen(limit=9), CFG, post=dead, say=lines.append)
    assert len(rows) == 2 and "stopped" in lines[-1]


def test_a_known_gap_is_reported_apart():
    gap = {**_case("level 5"), "gap": "not built yet"}
    rows = E.run([gap], CFG, say=lambda line: None, post=lambda cfg, messages: "var = t")
    text = E.summary(rows)
    assert "known gap" in text and "still wrong" in text
    assert text.splitlines()[1].split()[:2] == ["all", "0"]      # not counted
