"""The demo's tile mode: the real page, values from lon/lat value tiles.

The 3.75 km global demo cannot ship its 41.9M cells, so the data repository
samples them offline onto value tiles (nearest cell at each pixel centre, one
byte per pixel in 254 steps of the field's range) and `shim.js` reads those.
Here the same tiles are made from the small demo file, so the test needs no
network: the page must list the tiles' fields, answer a click with exactly the
byte the tile holds, and draw frames, projected ones included.

Skipped, like test_demo.py, without the demo dataset or a Playwright browser.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

DEMO = Path(__file__).resolve().parent.parent / "docs" / "demo"
DATA = DEMO / "data" / "demo.nc"
FIELDS = ("cape", "t2m")
TIMES = ("2019-09-01_00.00.00", "2019-09-01_01.00.00")
TILE, LEVELS, MAXZOOM = 256, 254, 2

pytestmark = pytest.mark.skipif(not DATA.exists(), reason="the demo dataset is not present")


def _tiles(viewer, out: Path) -> dict:
    """Value tiles and manifest.json as the data repository writes them; the
    second step is the first plus one, so a series has something to show."""
    from gmpas.viewer import ViewIndex

    sys.path.insert(0, str(DEMO))
    import bake

    home = (-180.0, 180.0, -90.0, 90.0)
    index = {z: ViewIndex(viewer.mesh, home, 2 * TILE * 2 ** z, TILE * 2 ** z)
             for z in range(MAXZOOM + 1)}
    fields = []
    for name in FIELDS:
        base = np.asarray(viewer.values(name, 0, 0), dtype=np.float32)
        steps = [base, base + 1]
        lo = float(min(np.nanmin(s) for s in steps))
        hi = float(max(np.nanmax(s) for s in steps))
        for t, values in zip(TIMES, steps):
            for z, vi in index.items():
                img = vi.frame(values)
                code = np.where(np.isfinite(img), 1 + np.clip(
                    np.rint((img - lo) * LEVELS / (hi - lo)), 0, LEVELS), 0).astype(np.uint8)
                from PIL import Image
                for x in range(code.shape[1] // TILE):
                    for y in range(code.shape[0] // TILE):
                        d = out / "tiles" / name / t / str(z) / str(x)
                        d.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(code[y * TILE:(y + 1) * TILE,
                                             x * TILE:(x + 1) * TILE], "L").save(d / f"{y}.png")
        fields.append({"name": name, "label": f"{name} [test]", "lo": lo, "hi": hi,
                       "show": [lo, hi], "maxzoom": MAXZOOM, "times": list(TIMES)})
    homes = {}
    for n in ("auto", "Orthographic"):
        from gmpas.projection import home_view
        proj, params, extent = home_view(n, None, home, 100.0)
        homes[n] = {"proj": proj, "extent": list(extent),
                    "plon": params.get("central_longitude"),
                    "plat": params.get("central_latitude"), "psp": None}
    manifest = {"run": {"title": "tile test", "mesh": "demo.nc",
                        "source": "Tester, A. (2026). Tile test data.", "doi": "10.0000/tiles"},
                "cells": viewer.mesh.n_cells, "home": list(home), "coverage": 100.0,
                "times": list(TIMES), "labels": ["00 UTC", "01 UTC"], "proj": homes,
                "tile": {"size": TILE, "format": "png", "levels": LEVELS},
                "fields": fields}
    (out / "manifest.json").write_text(json.dumps(manifest))
    return {"manifest": manifest, "index": index, "bake": bake}


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    from gmpas.viewer import Viewer

    root = tmp_path_factory.mktemp("tiles")
    viewer = Viewer(DATA, nx=400, ny=200)
    try:
        made = _tiles(viewer, root / "site" / "tiledata")
        cape = np.asarray(viewer.values("cape", 0, 0), dtype=np.float32)
    finally:
        viewer.close()
    # the bake wipes its target, so the tiles go in after it
    tmp = root / "baked"
    made["bake"].bake(DATA, tmp, tiles="tiledata/")
    for p in tmp.iterdir():
        dest = root / "site" / p.name
        p.rename(dest)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(root), **k)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield {"url": f"http://127.0.0.1:{srv.server_address[1]}/site/", "cape": cape, **made}
    srv.shutdown()


@pytest.fixture(scope="module")
def page(site):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.skip("playwright is not installed")
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:
            pytest.skip(f"no chromium: {exc}")
        pg = browser.new_page(viewport={"width": 1400, "height": 900})
        problems: list[str] = []
        pg.on("console", lambda m: problems.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
        pg.goto(site["url"], wait_until="networkidle")
        pg.wait_for_function("() => document.querySelectorAll('#vars div').length > 0",
                             timeout=30000)
        pg.wait_for_function("() => { const d = document.querySelector('#data');"
                             "        return d && d.naturalWidth > 0; }", timeout=30000)
        pg.problems = problems
        yield pg
        browser.close()


def _get(pg, path: str) -> dict:
    return pg.evaluate("async u => { const r = await fetch(u);"
                       " return {ok: r.ok, body: await r.json()}; }", path)


def test_the_page_lists_the_tiles_fields_and_steps(page):
    names = page.locator("#vars div").all_text_contents()
    assert names == list(FIELDS)
    meta = _get(page, "api/meta")["body"]
    assert meta["steps"] == len(TIMES) and meta["regional"] is False
    assert not page.problems, f"console errors: {page.problems}"


def test_a_click_answers_the_byte_the_tile_holds(page, site):
    """The value is the tile's own, decoded -- and so within one step of the
    cell's value -- at the finest zoom, at the pixel the click falls in."""
    m = site["manifest"]["fields"][0]
    step = (m["hi"] - m["lo"]) / LEVELS
    vi = site["index"][MAXZOOM]
    W, H = vi.nx, vi.ny
    cape = site["cape"]
    checked = 0
    for gy in range(0, H, 7):
        for gx in range(0, W, 11):
            p = gy * W + gx
            if vi.blank.ravel()[p]:
                continue
            lon = -180 + (gx + 0.5) * 360 / W
            lat = 90 - (gy + 0.5) * 180 / H
            got = _get(page, f"api/probe?lon={lon}&lat={lat}&var=cape&time=0&level=0")["body"]
            want = cape[vi.idx.ravel()[p]]
            byte = 1 + int(np.clip(np.rint((want - m["lo"]) / step), 0, LEVELS))
            assert got["value"] == pytest.approx(m["lo"] + (byte - 1) * step, abs=1e-3)
            assert abs(got["value"] - want) <= step / 2 + 1e-3
            checked += 1
            if checked >= 25:
                return
    assert checked, "no pixel of the demo mesh was on the tile grid"


def test_a_series_reads_every_step(page, site):
    vi = site["index"][MAXZOOM]
    p = int(np.flatnonzero(~vi.blank)[0])
    gx, gy = p % vi.nx, p // vi.nx
    lon, lat = -180 + (gx + 0.5) * 360 / vi.nx, 90 - (gy + 0.5) * 180 / vi.ny
    got = _get(page, f"api/series?lon={lon}&lat={lat}&var=cape&time=0&level=0")["body"]
    assert got["state"] == "done" and got["labels"] == ["00 UTC", "01 UTC"]
    a, b = got["values"]
    assert b - a == pytest.approx(1, abs=2 * (site["manifest"]["fields"][0]["hi"]
                                            - site["manifest"]["fields"][0]["lo"]) / LEVELS)


def test_frames_draw_on_the_lon_lat_map_and_projected(page):
    for extra in ("", "&proj=Orthographic&plon=147&plat=-2"):
        ext = "120,170,-25,20" if not extra else "-3000000,3000000,-3000000,3000000"
        n = page.evaluate("""async q => {
            const r = await fetch("api/frame?" + q);
            if (!r.ok) throw new Error((await r.json()).error);
            const bmp = await createImageBitmap(await r.blob());
            const g = new OffscreenCanvas(bmp.width, bmp.height).getContext("2d");
            g.drawImage(bmp, 0, 0);
            const d = g.getImageData(0, 0, bmp.width, bmp.height).data;
            let on = 0; for (let i = 3; i < d.length; i += 4) if (d[i]) on++;
            return on; }""", f"var=cape&time=1&level=0&extent={ext}&nx=300&ny=200"
                             f"&cmap=viridis{extra}")
        assert n > 1000, f"almost nothing drawn ({n} px) for {extra or 'lon/lat'}"
    assert not page.problems, f"console errors: {page.problems}"


def test_unreadable_tile_data_is_said_not_left_loading(page, site):
    """A manifest in the wrong shape -- an old copy, a broken deploy -- puts
    the reason in the page's title instead of leaving it at "loading"."""
    ctx = page.context.browser.new_context()   # the module's browser: one sync API at a time
    pg = ctx.new_page()
    try:
        pg.route("**/manifest.json", lambda r: r.fulfill(
            status=200, content_type="application/json", body='{"run": {}, "fields": []}'))
        pg.goto(site["url"])
        pg.wait_for_function(
            "() => document.querySelector('#title').textContent.includes('could not be read')",
            timeout=20000)
    finally:
        ctx.close()


def test_the_data_is_cited_under_the_fields(page):
    box = page.locator("#demodata")
    assert box.count() == 1
    assert "Tester, A. (2026). Tile test data." in box.inner_text()
    assert box.locator("a").get_attribute("href") == "https://doi.org/10.0000/tiles"
