"""Folding the right panel's sections, in a real browser.

A section folds to its heading on a click and stays folded on the next visit.
Folding only hides: the controls underneath keep their values. Skipped when
Playwright is not installed.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("playwright", reason="browser tests need playwright")


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    from conftest import write_mesh
    from gmpas.viewer import PAGE, Viewer, _handler, bind

    folder = tmp_path_factory.mktemp("run")
    rng = np.random.default_rng(0)
    lon = rng.uniform(-180, 180, 400)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, 400)))
    mesh = folder / "mesh.nc"
    write_mesh(mesh, list(zip(lon, lat, strict=True)))
    with xr.open_dataset(mesh) as ds:
        full = ds.load()
    mesh.unlink()
    full["t2m"] = (("Time", "nCells"), np.arange(full.sizes["nCells"], dtype=float)[None])
    for step in (1, 2):     # two steps, so there is something to play
        full.to_netcdf(folder / f"history.2012-01-0{step}_00.00.00.nc")

    viewer = Viewer(folder, nx=600, ny=350)
    srv = bind(_handler(viewer, PAGE), 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/"
    srv.shutdown()
    viewer.close()


def _ready(pg, url):
    pg.goto(url, wait_until="networkidle")
    pg.wait_for_function(
        "() => { const d = document.querySelector('#data');"
        "        return d && d.naturalWidth > 0; }", timeout=30000)


@pytest.fixture(scope="module")
def page(served):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:
            pytest.skip(f"no chromium: {exc}")
        pg = browser.new_page(viewport={"width": 1300, "height": 800})
        _ready(pg, served)
        yield pg
        browser.close()


def test_every_section_on_the_right_has_a_folding_heading(page):
    bare = page.evaluate(
        "() => [...document.querySelectorAll('#right > .sec')]"
        "  .filter(s => !s.querySelector(':scope > label.fold')).map(s => s.id)")
    assert bare == []


def test_a_click_folds_a_section_and_another_opens_it(page):
    assert page.is_visible("#vmin")
    page.click("#rangesec > label")
    assert not page.is_visible("#vmin")
    assert page.is_visible("#rangesec > label")
    assert page.get_attribute("#rangesec > label", "aria-expanded") == "false"
    assert page.is_visible("#cmap"), "only the clicked section folds"
    page.click("#rangesec > label")
    assert page.is_visible("#vmin")
    assert page.get_attribute("#rangesec > label", "aria-expanded") == "true"


def test_the_keyboard_folds_too(page):
    page.focus("#extentsec > label")
    page.keyboard.press("Enter")
    assert not page.is_visible("#elon0")
    page.keyboard.press(" ")
    assert page.is_visible("#elon0")


def test_space_on_a_heading_does_not_start_playback(page):
    page.focus("#extentsec > label")
    page.keyboard.press(" ")
    page.wait_for_timeout(300)
    playing = page.evaluate("playingKey")
    page.keyboard.press(" ")
    assert playing is None


def test_a_folded_section_keeps_what_it_holds(page):
    page.fill("#vmin", "12")
    page.dispatch_event("#vmin", "change")
    page.click("#rangesec > label")
    assert page.input_value("#vmin") == "12"
    page.click("#rangesec > label")
    assert page.input_value("#vmin") == "12"
    page.click("#reset")


def test_folded_sections_stay_folded_on_the_next_visit(page, served):
    page.click("#animsec > label")
    _ready(page, served)
    assert not page.is_visible("#fps")
    assert page.is_visible("#cmap")
    page.click("#animsec > label")
    _ready(page, served)
    assert page.is_visible("#fps")
