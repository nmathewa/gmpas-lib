"""The script layer: a view as lines of `key = value` (gmpas/script.py).

The parser is what stands between text someone typed -- or text something
else wrote -- and the map, so what it refuses matters as much as what it
reads. The page half is driven in a real browser at the end, when Playwright
is installed.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pytest
import xarray as xr

from gmpas import script

META = {
    "variables": [{"name": "theta", "levels": 1}, {"name": "qv", "levels": 4},
                  {"name": "u", "levels": 4}],
    "steps": 3,
    "labels": ["2012-01-01 00:00", "2012-01-01 06:00", "2012-01-01 12:00"],
    "cmaps": ["viridis", "RdBu_r"],
    "projections": ["auto", "Orthographic", "Mercator"],
    "colour_options": {"levels": {}},
}


def test_a_script_reads_as_plain_settings():
    got = script.parse("""
        # a comment
        ! and a Ferret one
        var    = qv
        time   = -1
        level  = 2
        cmap   = RdBu_r
        vmin   = -5
        vmax   = 5e0
        levels = (-inf)(-25,0,5)(0,25,1)(inf)
        reverse = true
        proj   = Mercator
        region = 90, 150, -15, 15
    """)
    assert got == {"var": "qv", "time": -1, "level": 2, "cmap": "RdBu_r",
                   "vmin": -5.0, "vmax": 5.0, "proj": "Mercator",
                   "region": [90.0, 150.0, -15.0, 15.0],
                   "colour": {"levels": "(-inf)(-25,0,5)(0,25,1)(inf)",
                              "reverse": True}}


def test_a_key_with_no_value_clears_it():
    got = script.parse("vmin =\nlevels =\nproj =\n")
    assert got == {"vmin": None, "proj": None, "colour": {"levels": None}}
    assert script.resolve(got, META)["proj"] == ""


@pytest.mark.parametrize("text, says", [
    ("var qv", "line 1: expected key = value"),
    ("var = qv\nspeed = 3", "line 2: unknown key 'speed'"),
    ("vmin = low", "line 1: vmin takes a number"),
    ("vmin = nan", "line 1: vmin must be finite"),
    ("level = 1.5", "line 1: level takes a whole number"),
    ("region = 1, 2, 3", "line 1: region takes four numbers"),
    ("region = 150, 90, -15, 15", "line 1: region: the first of each pair"),
    ("region = 0, 10, -95, 15", "line 1: region: latitudes"),
    ("cmap = a\ncmap = b", "line 2: cmap is already set on line 1"),
    ("var = qv\nlevels = (0,10", "line 2: "),
    ("levels = 10\nnorm = log", "levels and norm=log cannot be combined"),
    ("proj = lonlat\nplon = 30", "line 2: plon needs a projection"),
    ("proj = Mercator\nregion = 0,10,0,10\nbox = 0,1,0,1", "line 3: region and box"),
    ("__import__('os').system('x') = 1", "line 1: cannot assign"),
])
def test_what_does_not_parse_says_which_line(text, says):
    with pytest.raises(script.ScriptError, match=says.replace("(", r"\(")):
        script.parse(text)


def test_a_script_is_bounded():
    with pytest.raises(script.ScriptError, match="too long"):
        script.parse("#" * (script.MAX_TEXT + 1))
    with pytest.raises(script.ScriptError, match="at most"):
        script.parse("\n" * (script.MAX_LINES + 1))


def test_resolve_spells_things_as_the_file_does():
    got = script.resolve(script.parse("var = QV\ntime = -1\nproj = lonlat"), META)
    assert got["var"] == "qv" and got["time"] == 2 and got["proj"] == ""
    got = script.resolve(script.parse("time = 2012-01-01 06:00"), META)
    assert got["time"] == 1
    # a derived expression stays as typed: the server evaluates it per frame
    assert script.resolve(script.parse("var = u - qv"), META)["var"] == "u - qv"


@pytest.mark.parametrize("text, says", [
    ("var = nope", "var: "),
    ("var = u - nope", "var: 'nope' is not a variable"),
    ("var =", "var cannot be cleared"),
    ("time = 3", "time: step 3 is outside this run's 3 step"),
    ("time = noon", "time: no step is labelled"),
    ("var = qv\nlevel = 4", "level: 4 is outside qv's 4 level"),
    ("level = -1", "level: -1 is below 0"),
    ("cmap = jetty", "cmap: 'jetty' is not one of"),
    ("proj = Robinson", "proj: 'Robinson' is not one of lonlat, auto"),
])
def test_resolve_refuses_what_the_file_does_not_have(text, says):
    with pytest.raises(script.ScriptError, match=says):
        script.resolve(script.parse(text), META)


def test_format_reads_back_to_the_same_settings():
    state = script.parse("var = hypot(u, qv)\ntime = 2\nvmin = -2.5\nvmax = 40\n"
                         "proj = Orthographic\nplon = 120.25\nplat = -10\n"
                         "box = -1000000, 1000000, -500000.5, 500000\n"
                         "levels = (,,2.5)\nreverse = true\nextend = both\n")
    text = script.format(state)
    assert script.parse(text) == state
    assert "plon = 120.25" in text.replace("  ", " ").replace("  ", " ")


def test_every_key_has_help():
    keys = script.keys()
    assert set(script.KEYS) <= set(keys)
    assert "levels" in keys and "bands" not in keys      # bands is hidden
    assert all(keys.values())


# ------------------------------------------------------------- the routes

@pytest.fixture(scope="module")
def served(tmp_path_factory):
    from conftest import write_mesh
    from gmpas.viewer import PAGE, Viewer, _handler, bind

    folder = tmp_path_factory.mktemp("run")
    path = folder / "history.2012-01-01_00.00.00.nc"
    rng = np.random.default_rng(0)
    lon = rng.uniform(-180, 180, 600)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, 600)))
    write_mesh(path, list(zip(lon, lat, strict=True)))
    with xr.open_dataset(path) as ds:
        full = ds.load()
    full["theta"] = (("Time", "nCells"), (280 + np.sin(np.radians(lat)) * 20)[None, :])
    full["wind"] = (("Time", "nCells"), (np.cos(np.radians(lon)) * 10)[None, :])
    full.to_netcdf(path, mode="w")

    viewer = Viewer(folder, nx=600, ny=350)
    srv = bind(_handler(viewer, PAGE), 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/"
    srv.shutdown()
    viewer.close()


def _get(base, path, **q):
    url = base + path + "?" + urllib.parse.urlencode(q)
    try:
        with urllib.request.urlopen(url) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_the_route_answers_with_checked_settings(served):
    status, got = _get(served, "api/script",
                       text="var = THETA\ncmap = viridis\nproj = Mercator\n")
    assert status == 200
    assert got == {"var": "theta", "cmap": "viridis", "proj": "Mercator"}


def test_the_route_refuses_a_bad_script_as_the_users_mistake(served):
    status, got = _get(served, "api/script", text="var = theta\nvmin = low\n")
    assert status == 400
    assert got["error"].startswith("line 2: vmin takes a number")


def test_meta_offers_the_script_box(served):
    _, meta = _get(served, "api/meta")
    assert "region" in meta["script"]["keys"] and meta["script"]["startup"] == ""


def test_a_region_is_boxed_in_the_projection(served):
    status, got = _get(served, "api/projbox", proj="Mercator", region="90,150,-15,15")
    assert status == 200
    x0, x1, y0, y1 = got["extent"]
    assert x0 < x1 and y0 < 0 < y1 and abs(y0 + y1) < 1.0       # symmetric


def test_the_command_line_refuses_a_script_that_does_not_parse(tmp_path, capsys):
    from gmpas import cli

    bad = tmp_path / "view.txt"
    bad.write_text("var = theta\nspeed = 3\n")
    code = cli.main(["view", str(tmp_path), "--script", str(bad)])
    assert code != 0
    assert "unknown key 'speed'" in capsys.readouterr().err
    assert script.startup() == ""


# --------------------------------------------------------------- the page

@pytest.fixture(scope="module")
def page(served):
    sync_api = pytest.importorskip("playwright.sync_api",
                                   reason="browser tests need playwright")
    with sync_api.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:
            pytest.skip(f"no chromium: {exc}")
        pg = browser.new_page(viewport={"width": 1300, "height": 800})
        pg.goto(served, wait_until="networkidle")
        pg.wait_for_function(
            "() => { const d = document.querySelector('#data');"
            "        return d && d.naturalWidth > 0; }", timeout=30000)
        yield pg
        browser.close()


def _apply(pg, text):
    pg.evaluate("t => { document.querySelector('#scriptBox').value = t; }", text)
    ok = pg.evaluate("() => scriptRun(document.querySelector('#scriptBox').value)")
    pg.wait_for_load_state("networkidle")
    return ok


def test_the_page_applies_a_script(page):
    assert page.locator("#scriptsec").is_visible()
    assert _apply(page, "var = wind\ncmap = RdBu_r\nvmin = -8\nvmax = 8\n"
                        "levels = (-8,8,2)\nproj = lonlat\nregion = 60, 120, -15, 20\n")
    got = page.evaluate("""() => ({var: cur.name, cmap: $("#cmap").value,
        vmin: $("#vmin").value, vmax: $("#vmax").value, col: COL, P,
        box: boxOf(view), hint: $("#scripthint").textContent})""")
    assert got["var"] == "wind" and got["cmap"] == "RdBu_r"
    assert (got["vmin"], got["vmax"]) == ("-8", "8")
    assert got["col"] == {"levels": "(-8,8,2)"} and got["P"] is None
    # the region is contained, unstretched: it fits one way and pads the other
    w, e, s, n = got["box"]
    assert w <= 60 + 1e-6 and e >= 120 - 1e-6 and s <= -15 + 1e-6 and n >= 20 - 1e-6
    assert (w, e) == pytest.approx((60, 120)) or (s, n) == pytest.approx((-15, 20))
    assert got["hint"] == "applied"


def test_only_what_is_named_changes(page):
    _apply(page, "var = wind\ncmap = RdBu_r\nvmin = -8\nlevels = 10\n")
    assert _apply(page, "var = theta\nlevels =\n")
    got = page.evaluate("""() => ({var: cur.name, cmap: $("#cmap").value,
        vmin: $("#vmin").value, col: COL})""")
    assert got == {"var": "theta", "cmap": "RdBu_r", "vmin": "-8", "col": {}}


def test_a_bad_script_changes_nothing_and_says_why(page):
    _apply(page, "var = theta\ncmap = viridis\n")
    assert not _apply(page, "var = wind\ncmap = jetty\n")
    got = page.evaluate("""() => ({var: cur.name, cmap: $("#cmap").value,
        hint: $("#scripthint").textContent,
        bad: $("#scripthint").classList.contains("bad")})""")
    assert got["var"] == "theta" and got["cmap"] == "viridis"
    assert "jetty" in got["hint"] and got["bad"]


@pytest.mark.parametrize("setup", [
    "var = wind\nproj = lonlat\nregion = 60, 120, -15, 20\nvmin = -3\nextend = both\n",
    "var = theta - wind\nproj = Orthographic\nplon = 100\nplat = 20\n",
    "var = theta\nproj = Mercator\nregion = 60, 120, -15, 20\nlevels = (,,2.5)\n",
])
def test_a_view_written_as_a_script_comes_back_the_same(page, setup):
    """`from view` then `apply` is a round trip: the text the page writes is
    read by the Python parser and lands on the same view."""
    assert _apply(page, setup)
    snap = """() => ({var: cur.name, time: $("#time").value, level: $("#level").value,
        cmap: $("#cmap").value, vmin: $("#vmin").value, vmax: $("#vmax").value,
        col: COL, P, box: boxOf(view)})"""
    before = page.evaluate(snap)
    page.click("#scriptsec summary") if not page.evaluate(
        "() => $('#scriptsec').open") else None
    page.click("#scriptFrom")
    text = page.input_value("#scriptBox")
    assert script.parse(text)                       # Python reads what JS wrote
    # go somewhere else, then come back by the script alone
    assert _apply(page, "var = theta\nproj = lonlat\nregion =\nvmin =\nlevels =\n"
                        "extend =\n")
    assert _apply(page, text)
    after = page.evaluate(snap)
    assert after["box"] == pytest.approx(before["box"], rel=1e-6, abs=1e-6)
    for side in (before, after):
        side.pop("box")
        if side["P"]:
            side["P"] = {k: pytest.approx(v) if isinstance(v, float) else v
                         for k, v in side["P"].items()}
    assert after == before


def test_a_globe_keeps_its_turn(page):
    assert _apply(page, "proj = Orthographic\nplon = -60\nplat = 35\n")
    got = page.evaluate("() => P")
    assert got["proj"] == "Orthographic"
    assert (got["plon"], got["plat"]) == pytest.approx((-60, 35))


# ------------------------------------------------- comments, levels, mistakes

def test_a_comment_may_follow_a_value():
    from gmpas import script
    assert script.parse("cmap = viridis # dark\ntime = 3 #") == \
        {"cmap": "viridis", "time": 3}
    assert script.parse("under_color = #ff0000 # red")["colour"] == \
        {"under_color": "#ff0000"}


def test_a_level_may_be_said_in_pascals():
    from gmpas import script
    meta = {"steps": 1, "labels": ["x"], "variables": [
        {"name": "t", "levels": 3, "level_values": [100000, 85000, 70000],
         "level_units": "Pa"},
        {"name": "z", "levels": 3, "level_values": [1000, 850, 700], "level_units": "hPa"}]}
    for text in ("var = t\nlevel = 850 hPa", "var = t\nlevel = 85000 Pa",
                 "var = z\nlevel = 85000 Pa", "shade(t.sel(level=85000))"):
        assert script.resolve(script.parse(text), meta)["level"] == 1


def test_one_level_number_for_fields_with_different_axes_is_refused():
    from gmpas import script
    meta = {"steps": 1, "labels": ["x"], "variables": [
        {"name": "u", "levels": 3, "level_values": [1000, 850, 700], "level_units": "hPa"},
        {"name": "v", "levels": 2, "level_values": [850, 700], "level_units": "hPa"},
        {"name": "w", "levels": 3, "level_values": [1000, 850, 700], "level_units": "hPa"}]}
    with pytest.raises(script.ScriptError, match="level 1 of u but level 0 of v"):
        script.resolve(script.parse("shade(u.sel(level=850) + v.sel(level=850))"), meta)
    assert script.resolve(script.parse("shade(u.sel(level=850) + w.sel(level=850))"),
                          meta)["level"] == 1


def test_contour_levels_may_be_a_list():
    from gmpas import script
    got = script.parse("contourf(t, levels=[270, 280.5])")
    assert got["colour"]["levels"] == "(270)(280.5)"


def test_the_land_and_the_sea_lie_under_the_field_whatever_the_order():
    from gmpas import script
    got = script.parse("var = t2m\ngridlines()\nborders()\nland()\nocean()")
    assert [x["kind"] for x in got["layers"]["layers"]][:4] == \
        ["land", "ocean", "pcolormesh", "gridlines"]


@pytest.mark.parametrize("text, says", [
    ("shade(" + "+".join(["t"] * 1500) + ")", "nested too deeply"),
    ('contourf(t)\nfigure(look="nonexistent")', "look='nonexistent'"),
    ("spd = hypot(u, v)", "spd is set and never used: show it with var = spd"),
    ("cmapp = viridis", "unknown key 'cmapp'"),
])
def test_a_mistake_is_said_as_itself(text, says):
    from gmpas import script
    with pytest.raises(script.ScriptError, match=says):
        script.parse(text)


def test_an_empty_line_gives_way_to_a_pick_and_a_level_before_a_reduction_is_kept():
    from gmpas import script
    meta = {"steps": 2, "labels": ["2012-01-01 00:00", "2012-01-01 01:00"],
            "time_statistics": True, "variables": [
                {"name": "u", "levels": 3, "level_values": [1000, 850, 700],
                 "level_units": "hPa"},
                {"name": "v", "levels": 3, "level_values": [1000, 850, 700],
                 "level_units": "hPa"}]}
    # a view restated with its blanks, then a pick: the pick stands
    got = script.resolve(script.parse("var = u.sel(level=850)\ntime =\nlevel ="), meta)
    assert got["var"] == "u" and got["level"] == 1
    # the level picked before the time mean is the level of the mean
    for text in ('var = u.sel(level=850).mean("time")',
                 'shade(u.isel(level=1).max("time"))',
                 'shade(u.mean("time").sel(level=850))'):
        got = script.resolve(script.parse(text), meta)
        assert got["level"] == 1 and got["var"] in ("u.mean('time')", "u.max('time')")
    with pytest.raises(script.ScriptError, match="already set"):
        script.parse("var = u.sel(level=850)\nlevel = 2")
    one = {**meta, "variables": [{"name": "t2m", "levels": 1}]}
    with pytest.raises(script.ScriptError, match="lists no level values"):
        script.resolve(script.parse('shade(t2m.sel(level=3).mean("time"))'), one)


def test_a_level_past_a_derived_fields_levels_is_refused():
    from gmpas import script
    meta = {"steps": 1, "labels": ["x"], "variables": [
        {"name": "u", "levels": 3, "level_values": [1000, 850, 700], "level_units": "hPa"},
        {"name": "v", "levels": 3, "level_values": [1000, 850, 700], "level_units": "hPa"}]}
    with pytest.raises(script.ScriptError, match="write level = 850 hPa"):
        script.resolve(script.parse("var = hypot(u, v)\nlevel = 850"), meta)
    with pytest.raises(script.ScriptError, match="outside hypot"):
        script.resolve(script.parse("level = 5"), meta, "hypot(u, v)")
    assert script.resolve(script.parse("var = hypot(u, v)\nlevel = 2"), meta)["level"] == 2
