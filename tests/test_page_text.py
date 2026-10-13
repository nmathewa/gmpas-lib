"""What a file or a user wrote is text on the page, never markup -- and a
click on a missing value says so. Driven in a real browser; skipped when
Playwright is not installed.
"""

from __future__ import annotations

import threading

import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("playwright", reason="browser tests need playwright")

MARKUP = '<img src=x onerror="window.__ran=1">'


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    from playwright.sync_api import sync_playwright

    from gmpas.generic import GenericViewer
    from gmpas.viewer import PAGE, _handler, bind

    lat = np.arange(-40.0, 40.1, 5.0)
    lon = np.arange(0.0, 360.0, 5.0)
    field = np.add.outer(lat, lon)[None].repeat(2, 0).astype("f4")
    field[:, 8, :6] = np.nan                    # missing: lon 0-25 at the equator
    path = tmp_path_factory.mktemp("text") / "labels.nc"
    xr.Dataset({"sst": (("time", "lat", "lon"), field, {"long_name": f"SST {MARKUP}"})},
               coords={"time": pd.date_range("2020-01-01", periods=2, freq="D"),
                       "lat": lat, "lon": lon}).to_netcdf(path)
    viewer = GenericViewer(path)
    srv = bind(_handler(viewer, PAGE), 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:
            pytest.skip(f"no chromium: {exc}")
        pg = browser.new_page(viewport={"width": 1300, "height": 800})
        pg.errors = []
        pg.on("pageerror", lambda e: pg.errors.append(str(e)))
        pg.goto(f"http://127.0.0.1:{srv.server_address[1]}/", wait_until="networkidle")
        pg.wait_for_function("() => { const d = document.querySelector('#data');"
                             "        return d && d.naturalWidth > 0; }", timeout=30000)
        yield pg
        browser.close()
    srv.shutdown()
    viewer.close()


def _click(pg, lon, lat):
    xy = pg.evaluate("""([lon, lat]) => { const r = $('#wrap').getBoundingClientRect(),
        b = boxOf(view); return [r.left + (lon - b[0]) / (b[1] - b[0]) * r.width,
                                 r.top + (b[3] - lat) / (b[3] - b[2]) * r.height]; }""",
                     [lon, lat])
    pg.mouse.click(*xy)
    pg.wait_for_load_state("networkidle")


def test_a_label_from_the_file_is_shown_as_text(page):
    _click(page, 180, 20)
    page.wait_for_function("() => $('#point').style.display === 'block'")
    assert MARKUP in page.inner_text("#ptnow")              # the words, as written
    assert page.locator("#ptnow img").count() == 0          # and no element made of them
    assert page.evaluate("() => window.__ran") is None


def test_a_name_with_quotes_and_markup_is_text_in_the_animation_list(page):
    name = 't - t.sel(month=12).mean("time") ' + MARKUP
    made = page.evaluate(
        """name => { const row = animRow('k', {varName: name, loading: false, total: 1,
                                              ready: 1});
                     const span = row.querySelector('.animrow-name');
                     return [span.title, span.textContent,
                             row.querySelectorAll('img').length,
                             span.attributes.length]; }""", name)
    assert made == [name, name, 0, 2]
    assert page.evaluate("() => window.__ran") is None


def test_a_click_on_a_missing_value_says_no_data(page):
    _click(page, 10, 0)
    page.wait_for_function("() => /no data/.test($('#probe2').textContent)")
    assert "no data" in page.inner_text("#ptnow")
    assert page.errors == []


def test_the_view_as_a_script_never_passes_a_pole(page):
    got = page.evaluate("() => regionOf({clon: 180, clat: 80, w: 100})")
    assert got[3] == 90 and got[2] >= -90 and got[0] == 130
    # and the text the page writes for the view is a script the server takes
    ok = page.evaluate("""async () => { const r = await fetch('api/script?' +
        new URLSearchParams({text: scriptText()})); return r.ok; }""")
    assert ok


def test_marking_a_place_while_a_figure_is_shown_says_why_not(page):
    page.evaluate("() => { $('#kind').value = 'hist'; $('#kind').onchange(); }")
    page.wait_for_load_state("networkidle")
    page.evaluate("() => ptAt(180, 20, 'mark')")
    assert "go back to the map" in page.evaluate("() => $('#scripthint').textContent")
    page.evaluate("() => { $('#kind').value = 'map'; $('#kind').onchange(); }")
    page.wait_for_load_state("networkidle")
    # a longitude in the other convention is the same place
    page.evaluate("() => ptAt(-150, 20, 'mark')")
    page.wait_for_function("() => /210/.test($('#pttitle').textContent)")
    assert page.evaluate("() => $('#ptmark').style.display") == "block"
