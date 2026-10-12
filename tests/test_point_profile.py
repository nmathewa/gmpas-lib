"""The profile in the point panel, in a real browser.

Click the map and the column under the click is one button away: value across,
level up. What matters is that the chart is about the point, the step and the
level the page is showing -- and stops being shown when any of them moves on
without it. Skipped when Playwright is not installed.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("playwright", reason="browser tests need playwright")

NLEV = 5


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    from conftest import write_mesh
    from gmpas.viewer import PAGE, Viewer, _handler, bind

    folder = tmp_path_factory.mktemp("run")
    rng = np.random.default_rng(0)
    lon = rng.uniform(-180, 180, 600)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, 600)))
    mesh = folder / "mesh.nc"
    write_mesh(mesh, list(zip(lon, lat, strict=True)))
    with xr.open_dataset(mesh) as ds:
        full = ds.load()
    mesh.unlink()
    n = full.sizes["nCells"]
    # value = 1000 * step + 100 * level + cell: a number names where it is from
    code = 100.0 * np.arange(NLEV)[None, :] + np.arange(n)[:, None]
    for step in range(2):
        out = full.copy()
        out["theta"] = (("Time", "nCells", "nVertLevels"), (1000 * step + code)[None])
        out["t2m"] = (("Time", "nCells"), np.arange(n, dtype=float)[None])
        out.to_netcdf(folder / f"history.2012-01-0{step + 1}_00.00.00.nc")

    viewer = Viewer(folder, nx=600, ny=350)
    srv = bind(_handler(viewer, PAGE), 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/"
    srv.shutdown()
    viewer.close()


@pytest.fixture(scope="module")
def page(served):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
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


def _pick(pg, name):
    pg.evaluate("n => { ptClose(); pick(n); }", name)
    pg.wait_for_load_state("networkidle")


def _click_map(pg, fx=0.5, fy=0.5):
    box = pg.locator("#wrap").bounding_box()
    before = pg.evaluate("() => PT ? PT.lon : null")
    pg.mouse.click(box["x"] + box["width"] * fx, box["y"] + box["height"] * fy)
    pg.wait_for_function("was => PT && PT.cell >= 0 && PT.lon !== was", arg=before)
    pg.wait_for_load_state("networkidle")


def _profile(pg):
    pg.click("#ptprof")
    pg.wait_for_function("() => PT && PT.profile && PT.show === 'profile'")
    return pg.evaluate("() => PT.profile")


def test_the_profile_is_the_clicked_cells_column(page):
    _pick(page, "theta")
    _click_map(page)
    assert page.locator("#ptprof").is_visible()
    cell = page.evaluate("() => PT.cell")
    got = _profile(page)
    assert got["cell"] == cell
    assert got["values"] == [100 * k + cell for k in range(NLEV)]
    assert page.locator("#ptchart path.ln").count() == 1
    assert f"{NLEV} levels" in page.inner_text("#pthint")


def test_the_profile_follows_time_and_a_new_click(page):
    _pick(page, "theta")
    _click_map(page)
    cell = _profile(page)["cell"]
    page.evaluate("() => nudge('#time', 0, 1)")
    page.wait_for_function(f"() => PT.profile && PT.profile.values[0] === {1000 + cell}")
    _click_map(page, 0.12, 0.15)
    other = page.evaluate("() => PT.cell")
    assert other != cell
    page.wait_for_function(f"() => PT.profile && PT.profile.cell === {other}")
    assert page.evaluate("() => PT.profile.values[1]") == 1000 + 100 + other
    page.evaluate("() => nudge('#time', 0, 0)")
    page.wait_for_load_state("networkidle")


def test_the_level_marker_moves_and_a_click_on_the_chart_sets_the_level(page):
    _pick(page, "theta")
    _click_map(page)
    got = _profile(page)
    page.evaluate("() => nudge('#level', 0, 3)")
    assert page.evaluate("() => !!PT.profile")          # still about this point
    assert f"{got['values'][3]:g}" in page.inner_text("#ptnow").replace(",", "")
    assert page.locator("#ptchart circle.lev").count() == 1
    # the top of the chart is the highest level; the bottom, the lowest
    box = page.locator("#ptchart .hit").bounding_box()
    page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] - 2)
    assert page.evaluate("() => +$('#level').value") == 0
    page.mouse.click(box["x"] + box["width"] / 2, box["y"] + 2)
    assert page.evaluate("() => +$('#level').value") == NLEV - 1
    page.evaluate("() => nudge('#level', 0, 0)")


def test_the_rows_are_the_profile(page):
    _pick(page, "theta")
    _click_map(page)
    got = _profile(page)
    out = page.evaluate("() => ptRows()")
    lines = out["text"].strip().split("\n")
    assert out["n"] == NLEV and out["what"] == "profile"
    assert lines[1] == "level,model level [index],value"
    assert lines[2] == f"0,0,{got['values'][0]:g}"


def test_a_flat_field_offers_no_profile_and_drops_the_old_one(page):
    _pick(page, "theta")
    _click_map(page)
    _profile(page)
    page.evaluate("() => pick('t2m')")              # the panel stays open
    page.wait_for_load_state("networkidle")
    assert not page.locator("#ptprof").is_visible()
    assert page.evaluate("() => PT.profile") is None
    assert not page.locator("#ptchart").is_visible()


def test_the_time_series_still_takes_the_chart_back(page):
    _pick(page, "theta")
    _click_map(page)
    _profile(page)
    page.click("#ptgo")
    page.wait_for_function("() => PT.series && PT.show === 'series'")
    assert page.evaluate("() => PT.series.values.length") == 2
    assert page.locator("#ptchart circle.lev").count() == 0
