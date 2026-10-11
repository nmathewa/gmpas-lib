"""The static demo draws the same picture the server would.

`docs/demo/shim.js` re-implements `ViewIndex` in JavaScript so the published
page can rasterize without Python behind it. A second implementation of a
render path is exactly the kind of thing that drifts from the first, silently,
and a demo that quietly draws a different map than the tool is worse than no
demo. So this renders the same frames both ways and compares the pixels.

Skipped unless the demo's dataset and a Playwright browser are both present,
which is the case in the demo workflow and on a developer's machine that has
run `bake.py`, and not in the ordinary test run.
"""

from __future__ import annotations

import base64
import io
import json
import shutil
import subprocess
import sys
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

DEMO = Path(__file__).resolve().parent.parent / "docs" / "demo"
DATA = DEMO / "data" / "demo.nc"

#: Python quantizes to 255 steps and looks the colour up in a 255-entry
#: palette; the browser interpolates a 256-stop ramp. The two sample the same
#: colormap a fraction of a step apart, so a handful of channel counts of
#: difference is the encoding, not a disagreement about what to draw. A wrong
#: *cell*, by contrast, puts an unrelated colour on the pixel.
CHANNEL_TOLERANCE = 12

pytestmark = pytest.mark.skipif(
    not DATA.exists(), reason="the demo dataset is not present")


def _playwright():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.skip("playwright is not installed")
    return sync_playwright


@lru_cache(maxsize=1)
def _site(tmp: str) -> Path:
    """Bake the demo once for the whole module."""
    sys.path.insert(0, str(DEMO))
    import bake

    out = Path(tmp) / "site"
    bake.bake(DATA, out)
    return out


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    """The baked demo, on a local server, under a subpath.

    Served from one level *above* the site so the page sits at `/site/`, the
    way GitHub Pages serves a project page. An absolute URL that crept into
    the page would reach the wrong place here and nowhere else.
    """
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    root = tmp_path_factory.mktemp("demo")
    site = _site(str(root))

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(root), **k)

        def log_message(self, *a):        # keep the test output readable
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/{site.name}/", site
    srv.shutdown()


@pytest.fixture(scope="module")
def page(served):
    url, _ = served
    sync_playwright = _playwright()
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:                       # no browser downloaded
            pytest.skip(f"no chromium: {exc}")
        pg = browser.new_page(viewport={"width": 1400, "height": 900})
        problems: list[str] = []
        pg.on("console", lambda m: problems.append(m.text)
              if m.type == "error" else None)
        pg.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
        pg.goto(url, wait_until="networkidle")
        # the page's `M` is a `let` inside its own script, not on window, so
        # wait on what it produces rather than on the variable
        pg.wait_for_function(
            "() => document.querySelectorAll('#vars div').length > 0",
            timeout=30000)
        pg.wait_for_function(
            "() => { const d = document.querySelector('#data');"
            "        return d && d.naturalWidth > 0; }", timeout=30000)
        pg.problems = problems
        yield pg
        browser.close()


def _browser_frame(pg, params: dict, headers: bool = False):
    """One frame, rendered by the page's own shim, as RGBA; with `headers`,
    also its X-Range and X-Colorbar."""
    from urllib.parse import urlencode

    from PIL import Image

    got = pg.evaluate(
        """async q => {
            const r = await fetch("api/frame?" + q);
            if (!r.ok) throw new Error((await r.json()).error);
            if (!r.headers.get("X-Range")) throw new Error("no X-Range header");
            const buf = new Uint8Array(await (await r.blob()).arrayBuffer());
            let s = ""; for (const b of buf) s += String.fromCharCode(b);
            return {png: btoa(s), range: r.headers.get("X-Range"),
                    bar: r.headers.get("X-Colorbar")};
        }""", urlencode(params))
    img = np.asarray(Image.open(io.BytesIO(base64.b64decode(got["png"]))).convert("RGBA"))
    if not headers:
        return img
    return img, [float(v) for v in got["range"].split(",")], \
        json.loads(got["bar"]) if got["bar"] else None


def _python_frame(params: dict, headers: bool = False):
    """The same frame from the Python renderer, read as the server would read
    the request: `colour` and the projection parameters included."""
    from PIL import Image

    from gmpas.viewer import Viewer, _proj_params

    viewer = Viewer(DATA, nx=1200, ny=700)
    try:
        meta: dict = {}
        q = {k: str(v) for k, v in params.items()}
        png, lo, hi = viewer.frame(
            params["var"], 0, int(params["level"]),
            [float(v) for v in str(params["extent"]).split(",")],
            params["cmap"],
            float(params["vmin"]) if "vmin" in params else None,
            float(params["vmax"]) if "vmax" in params else None,
            nx=int(params["nx"]), ny=int(params["ny"]),
            colour=params.get("colour"), meta=meta,
            proj=_proj_params(q, viewer) if "proj" in params else None)
    finally:
        viewer.close()
    img = np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))
    return (img, [lo, hi], meta.get("colorbar")) if headers else img


def _same_picture(mine: np.ndarray, theirs: np.ndarray, tolerance: int) -> None:
    """Both frames agree on which pixels are on the mesh and on their colours."""
    assert mine.shape == theirs.shape
    on_a = mine[..., 3] > 0
    on_b = theirs[..., 3] > 0
    # a frame that is blank both ways would agree about everything and prove
    # nothing, so insist there is a picture here to compare
    assert on_b.mean() > 0.2, "the reference frame is mostly empty"
    disagree = np.mean(on_a != on_b)
    assert disagree < 0.005, (
        f"{disagree:.2%} of pixels disagree about being on the mesh; "
        f"the JS nearest-cell search has drifted from ViewIndex")

    both = on_a & on_b
    gap = np.abs(mine[..., :3].astype(int) - theirs[..., :3].astype(int)).max(-1)
    bad = np.mean(gap[both] > tolerance)
    assert bad < 0.01, (
        f"{bad:.2%} of on-mesh pixels differ by more than {tolerance}/255 "
        f"(median {np.median(gap[both]):.0f}); that is a different cell, not "
        f"a different rounding")


HOME = "125.7,168.3,-24.3,20.23"
ZOOMED = "140.0,155.0,-10.0,5.0"


@pytest.mark.parametrize("var, level, extent, cmap", [
    ("cape", 0, HOME, "viridis"),
    ("cape", 0, ZOOMED, "plasma"),          # zoomed in, different colormap
    ("t_isobaric", 3, HOME, "viridis"),     # a level the slider has to reach
])
def test_the_browser_draws_what_the_server_would(page, var, level, extent, cmap):
    """Pixel for pixel, against the Python renderer.

    The mask is the strict half: whether a pixel is on the mesh at all is a
    yes/no answer both sides must agree on, and a disagreement there means
    the nearest-cell search or the radius test has drifted.
    """
    params = {"var": var, "level": level, "extent": extent, "cmap": cmap,
              "vmin": 0.0, "vmax": 100.0, "nx": 420, "ny": 260}
    if var == "t_isobaric":
        params |= {"vmin": 180.0, "vmax": 300.0}

    _same_picture(_browser_frame(page, params), _python_frame(params),
                  CHANNEL_TOLERANCE)


def _home(name: str) -> dict:
    from gmpas.viewer import Viewer

    viewer = Viewer(DATA, nx=1200, ny=700)
    try:
        return viewer.projected_home(name)
    finally:
        viewer.close()


def _proj_query(home: dict) -> dict:
    q = {"proj": home["proj"], "extent": ",".join(repr(v) for v in home["extent"])}
    if home["plon"] is not None:
        q["plon"] = home["plon"]
    if home["plat"] is not None:
        q["plat"] = home["plat"]
    if home["psp"]:
        q["psp"] = ",".join(repr(v) for v in home["psp"])
    return q


PROJECTIONS = ["auto", "Orthographic", "NorthPolarStereo", "SouthPolarStereo",
               "LambertConformal", "Mercator"]


@pytest.mark.parametrize("name", PROJECTIONS)
def test_a_projected_map_is_the_one_the_server_draws(page, name):
    """Each projection the demo offers, framed as the page frames it: the
    browser's own inverse puts the same cell under every pixel that cartopy's
    does, and /api/projview is the server's answer."""
    home = _home(name)
    got = page.evaluate(
        """async n => (await (await fetch("api/projview?proj=" + n)).json())""", name)
    assert got == home

    params = {"var": "cape", "level": 0, "cmap": "viridis", "vmin": 0.0,
              "vmax": 100.0, "nx": 420, "ny": 260, **_proj_query(home)}
    _same_picture(_browser_frame(page, params), _python_frame(params),
                  CHANNEL_TOLERANCE)


def test_a_turned_globe_is_the_one_the_server_draws(page):
    """The globe is dragged in the browser, so its centre is whatever the
    drag left: a view from well off the region, part of it past the limb."""
    params = {"var": "cape", "level": 0, "cmap": "viridis", "vmin": 0.0,
              "vmax": 100.0, "nx": 420, "ny": 260, "proj": "Orthographic",
              "plon": 95.0, "plat": 35.0,
              "extent": "0,6378137,-6378137,2000000"}
    mine, theirs = _browser_frame(page, params), _python_frame(params)
    assert np.mean((mine[..., 3] > 0) != (theirs[..., 3] > 0)) < 0.005
    assert (theirs[..., 3] > 0).mean() > 0.02
    both = (mine[..., 3] > 0) & (theirs[..., 3] > 0)
    gap = np.abs(mine[..., :3].astype(int) - theirs[..., :3].astype(int)).max(-1)
    assert np.mean(gap[both] > CHANNEL_TOLERANCE) < 0.01


@pytest.mark.parametrize("name", PROJECTIONS[1:])
def test_a_projected_click_lands_where_the_server_says(page, name):
    from gmpas.viewer import Viewer, _proj_params

    home = _home(name)
    q = _proj_query(home)
    x0, x1, y0, y1 = home["extent"]
    viewer = Viewer(DATA, nx=1200, ny=700)
    try:
        proj = _proj_params({k: str(v) for k, v in q.items()}, viewer)
        for fx, fy in [(0.5, 0.5), (0.2, 0.7), (0.9, 0.1)]:
            x, y = x0 + fx * (x1 - x0), y0 + fy * (y1 - y0)
            want = viewer.unproject(proj, x, y, 1000.0)
            got = page.evaluate(
                """async q => (await (await fetch("api/unproject?" + q)).json())""",
                "&".join(f"{k}={v}" for k, v in
                         {**q, "x": x, "y": y, "tol": 1000.0}.items() if k != "extent"))
            assert got["on"] == want["on"]
            if want["on"]:
                assert got["lon"] == pytest.approx(want["lon"], abs=1e-6)
                assert got["lat"] == pytest.approx(want["lat"], abs=1e-6)
    finally:
        viewer.close()


#: With colour options both sides paint from the same lookup table, so the
#: only slack left is the browser's PNG round trip.
COLOUR_TOLERANCE = 2


@pytest.mark.parametrize("cmap, colour", [
    ("viridis", {"levels": "12"}),
    ("viridis", {"levels": "(-inf)(0,500,100)(500,3000,500)(inf)"}),
    ("RdBu_r", {"levels": "(,,250)", "reverse": True}),
    ("viridis", {"levels": "(0)(10)(100)(1000)(2000)", "under_color": "#ff00ff",
                 "over_color": "#000000", "extend": "both"}),
    ("plasma", {"bands": 7}),
    ("viridis", {"reverse": True}),
    ("magma", {"norm": "power", "gamma": 0.5, "extend": "max"}),
    ("viridis", {"norm": "symlog", "linthresh": 10.0}),
    ("grads.rainbow", {"levels": "20"}),
    ("grads.rainbow", {"reverse": True}),
])
def test_colour_options_draw_what_the_server_would(page, cmap, colour):
    """The colour options, levels among them: the same colours on the same
    pixels, the same range, and the same bar beside the map."""
    params = {"var": "cape", "level": 0, "extent": HOME, "cmap": cmap,
              "vmin": 0.0, "vmax": 2500.0, "nx": 420, "ny": 260,
              "colour": json.dumps(colour)}
    mine, my_range, my_bar = _browser_frame(page, params, headers=True)
    theirs, their_range, their_bar = _python_frame(params, headers=True)
    _same_picture(mine, theirs, COLOUR_TOLERANCE)
    assert my_range == pytest.approx(their_range)
    assert my_bar["stops"] == their_bar["stops"]
    assert my_bar["extend"] == their_bar["extend"]
    assert my_bar["under"] == their_bar["under"] and my_bar["over"] == their_bar["over"]
    if their_bar["edges"] is None:
        assert my_bar["edges"] is None
    else:
        assert my_bar["edges"] == pytest.approx(their_bar["edges"])


def test_a_centre_makes_the_range_symmetric(page):
    """`center` with one end typed: the range is as far either side of it as
    the further end, as colour.centred has it."""
    _, (lo, hi), bar = _browser_frame(page, {
        "var": "cape", "level": 0, "extent": HOME, "cmap": "RdBu_r", "vmin": 100.0,
        "nx": 60, "ny": 40, "colour": json.dumps({"center": 500.0})}, headers=True)
    assert lo == pytest.approx(1000.0 - hi) and hi > 900.0
    assert bar["lo"] == lo and bar["hi"] == hi


@pytest.mark.parametrize("colour, message", [
    ({"levels": "(5)(1)"}, "levels must increase"),
    ({"levels": "abc"}, "is not understood"),
    ({"levels": "10", "bands": 5}, "cannot be combined"),
    ({"levels": "10", "norm": "log"}, "cannot be combined"),
])
def test_bad_levels_are_refused_in_the_servers_words(page, colour, message):
    got = page.evaluate(
        """async q => { const r = await fetch("api/frame?" + q);
                        return {ok: r.ok, body: r.ok ? null : await r.json()}; }""",
        "var=cape&level=0&nx=60&ny=40&cmap=viridis&extent=" + HOME
        + "&colour=" + json.dumps(colour))
    assert got["ok"] is False and message in got["body"]["error"]


def test_choosing_a_projection_on_the_page_draws_it(page):
    """Through the page's own select: the map redraws projected, the coast
    with it, and nothing lands in the console."""
    names = page.eval_on_selector_all("#proj option", "os => os.map(o => o.value)")
    assert names == [""] + PROJECTIONS
    for name in PROJECTIONS:
        page.select_option("#proj", name)
        page.wait_for_function(
            "() => /ms$/.test(document.querySelector('#msg').textContent)"
            "      && document.querySelector('#over').src.startsWith('data:')",
            timeout=30000)
        page.wait_for_timeout(300)
        assert page.locator("#data").evaluate("e => e.naturalWidth") > 0
    page.select_option("#proj", "")
    page.wait_for_timeout(500)
    assert not page.problems, f"console errors: {page.problems}"


def test_the_probe_agrees_with_the_server(page):
    from gmpas.viewer import Viewer

    points = [(140.0, 0.0), (150.0, -5.0), (130.0, 10.0)]
    viewer = Viewer(DATA, nx=1200, ny=700)
    try:
        for lon, lat in points:
            got = page.evaluate(
                """async q => (await (await fetch("api/probe?" + q)).json())""",
                f"lon={lon}&lat={lat}&var=cape&time=0&level=0")
            want = viewer.probe(lon, lat, "cape", 0, 0)
            assert got["cell"] == want["cell"], f"different cell at {lon},{lat}"
            assert got["value"] == pytest.approx(want["value"], rel=1e-6)
    finally:
        viewer.close()


def test_the_page_comes_up_clean_and_small(page, served):
    """No console errors, and a first paint a stranger on a slow link will
    actually wait for."""
    _, site = served
    assert page.locator("#vars div").count() == 96
    assert page.locator("#data").evaluate("e => e.naturalWidth") > 0
    assert not page.problems, f"console errors: {page.problems}"

    first = sum((site / p).stat().st_size for p in (
        "index.html", "shim.js", "data/meta.json", "data/mesh.bin",
        "data/fields.json", "data/coast.json", "data/ramps/viridis.json"))
    assert first < 500 * 1024, f"first paint is {first/1024:.0f} KiB"


def test_what_needs_a_server_says_so_rather_than_breaking(page):
    """Figures, GIFs and netCDF need matplotlib. The page shows the message
    it is given, so the shim has to answer in the shape it expects."""
    got = page.evaluate(
        """async () => {
            const r = await fetch("api/export/figure?var=cape&time=0&level=0"
                                  + "&extent=125,168,-24,20&cmap=viridis");
            return {ok: r.ok, body: await r.json()};
        }""")
    assert got["ok"] is False
    assert "static demo" in got["body"]["error"]


def test_the_bake_refuses_to_publish_a_page_it_could_not_hook(tmp_path):
    """The coastline hook is a string patch on viewer.PAGE, because the demo
    does not modify the package. If the page changes upstream the patch must
    fail loudly -- silently publishing a map with no coastlines is the bad
    outcome."""
    sys.path.insert(0, str(DEMO))
    import bake

    hooked = bake.patch_page(bake.OVERLAY_SRC + "\n")
    assert "if(window.GMPAS_OVERLAY)" in hooked
    assert hooked.count("api/overlay") == 1        # the original call survives
    with pytest.raises(SystemExit, match="could not find the overlay line"):
        bake.patch_page("<body>a page that moved on</body>")


def test_the_baked_page_is_the_real_one(tmp_path):
    """Not a copy that can rot: the demo's HTML is viewer.PAGE plus two
    lines."""
    sys.path.insert(0, str(DEMO))
    import bake

    from gmpas.viewer import PAGE

    out = bake.patch_page(PAGE)
    assert len(out) - len(PAGE) < 200
    assert "GMPAS_OVERLAY" in out and 'src="shim.js"' in out
