"""Dragging the map, in a real browser.

The pan is pointer-event code, and the way it broke could not be seen from
Python: a drag begun on the map became the *browser's* own image drag, which
fires `pointercancel` instead of `pointerup`. So the pan died two frames in,
and because only `pointerup` cleared the drag state, the map then followed
the cursor around the screen with no button held.

Nothing short of driving a browser catches that, so these tests do. Skipped
when Playwright is not installed, which is the ordinary test run.
"""

from __future__ import annotations

import re
import threading

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("playwright", reason="browser tests need playwright")


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    """A real viewer on a real port, serving the real page."""
    from conftest import write_mesh
    from gmpas.viewer import PAGE, Viewer, _handler, bind

    folder = tmp_path_factory.mktemp("run")
    path = folder / "history.2012-01-01_00.00.00.nc"
    rng = np.random.default_rng(0)
    lon = rng.uniform(-180, 180, 600)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, 600)))
    write_mesh(path, list(zip(lon, lat)))
    with xr.open_dataset(path) as ds:
        full = ds.load()
    full["theta"] = (("Time", "nCells"), (280 + np.sin(np.radians(lat)) * 20)[None, :])
    full.to_netcdf(path, mode="w")

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
        pg = browser.new_page(viewport={"width": 1100, "height": 800})
        pg.goto(served, wait_until="networkidle")
        pg.wait_for_function(
            "() => { const d = document.querySelector('#data');"
            "        return d && d.naturalWidth > 0; }", timeout=30000)
        yield pg
        browser.close()


def _zoom_in(pg, factor=4):
    """Pan is only possible zoomed in; at home the view is clamped to the
    mesh, and `clamp` legitimately pins the centre."""
    pg.evaluate(f"() => {{ view.w = home.w/{factor}; clamp(); schedule(0); }}")
    pg.wait_for_timeout(500)


def _drag(pg, dx, dy, steps=8):
    box = pg.locator("#wrap").bounding_box()
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    pg.mouse.move(cx, cy)
    pg.mouse.down()
    for i in range(1, steps + 1):
        pg.mouse.move(cx + dx * i / steps, cy + dy * i / steps)
        pg.wait_for_timeout(12)
    pg.mouse.up()
    pg.wait_for_timeout(250)
    return box


def test_dragging_moves_the_map_by_exactly_the_distance_dragged(page):
    """Not merely "it moved": the gesture and the map have to agree, or the
    map slides out from under the cursor."""
    _zoom_in(page)
    before = page.evaluate("() => ({...view, box: boxOf(view)})")
    dx, dy = -150, 60
    box = _drag(page, dx, dy)
    after = page.evaluate("() => ({...view})")

    span = before["box"][1] - before["box"][0]
    tall = before["box"][3] - before["box"][2]
    want_lon = before["clon"] - dx / box["width"] * span
    want_lat = before["clat"] + dy / box["height"] * tall
    assert after["clon"] == pytest.approx(want_lon, abs=0.05 * abs(span))
    assert after["clat"] == pytest.approx(want_lat, abs=0.05 * abs(tall))
    # and it really is most of the gesture, not the first two frames of it
    assert abs(after["clon"] - before["clon"]) > 0.5 * abs(dx / box["width"] * span)


def test_the_browser_never_takes_the_drag_for_an_image_drag(page):
    """`dragstart` here means the browser has started dragging the PNG, which
    cancels the pointer and strands the pan half-done."""
    _zoom_in(page)
    page.evaluate("""() => {
        window.__seen = [];
        for (const t of ["dragstart", "pointercancel"])
            document.addEventListener(t, e => window.__seen.push(t), true);
    }""")
    _drag(page, -140, 40)
    assert page.evaluate("() => window.__seen") == []


def test_the_map_stops_following_the_cursor_when_let_go(page):
    """The symptom that made panning feel broken: after a drag the map kept
    moving with the mouse, because only `pointerup` cleared the drag state."""
    _zoom_in(page)
    _drag(page, -120, 30)
    settled = page.evaluate("() => ({...view})")
    assert page.evaluate("() => drag === null")

    box = page.locator("#wrap").bounding_box()
    for i in range(1, 6):
        page.mouse.move(box["x"] + 40 + i * 40, box["y"] + 40 + i * 20)
        page.wait_for_timeout(15)
    page.wait_for_timeout(200)
    idle = page.evaluate("() => ({...view})")
    assert idle["clon"] == pytest.approx(settled["clon"], abs=1e-9)
    assert idle["clat"] == pytest.approx(settled["clat"], abs=1e-9)


def test_a_release_the_page_never_saw_still_ends_the_drag(page):
    """A pointerup outside the window, or a capture lost to something else,
    leaves no event to clear the drag. The next move with no button held has
    to do it, or the map pans for the rest of the session."""
    _zoom_in(page)
    box = page.locator("#wrap").bounding_box()
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    page.mouse.move(cx, cy)
    page.mouse.down()
    page.mouse.move(cx - 40, cy)
    page.wait_for_timeout(30)
    # the release itself never reaches the page
    page.evaluate("() => document.querySelector('#wrap').releasePointerCapture"
                  "&& 0")
    page.mouse.up()
    page.wait_for_timeout(50)
    page.evaluate("() => { if (typeof drag !== 'undefined' && drag) "
                  "         drag.__stale = true; }")
    page.mouse.move(cx + 200, cy + 100)     # no button held
    page.wait_for_timeout(120)
    assert page.evaluate("() => drag === null")


def test_the_right_button_does_not_start_a_pan(page):
    """A right-click opens the context menu, which swallows the pointerup --
    so treating it as a pan leaves the drag state set behind the menu."""
    _zoom_in(page)
    before = page.evaluate("() => ({...view})")
    box = page.locator("#wrap").bounding_box()
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    page.mouse.move(cx, cy)
    page.mouse.down(button="right")
    page.mouse.move(cx - 90, cy + 30)
    page.wait_for_timeout(40)
    assert page.evaluate("() => drag === null")
    page.mouse.up(button="right")
    after = page.evaluate("() => ({...view})")
    assert after["clon"] == pytest.approx(before["clon"], abs=1e-9)


def test_the_derive_box_draws_a_scalar_expression_and_finds_a_name_in_any_case(page):
    """`theta * 2` was read as two field names; `THETA` missed `theta`."""
    page.fill("#deriveExpr", "THETA")
    page.click("#deriveBtn")
    assert page.evaluate("() => cur.name") == "theta"          # the field itself
    with page.expect_response(lambda r: "api/frame" in r.url) as resp:
        page.fill("#deriveExpr", "theta * 2")
        page.click("#deriveBtn")
    assert resp.value.ok
    assert page.evaluate("() => cur.name") == "theta * 2"


def test_no_form_control_is_left_browser_white(page):
    """The page is dark, but number fields, the JSON box and inputs the page
    creates without a `type` attribute drew in the browser's default white."""
    page.evaluate("document.querySelectorAll('details').forEach(d => d.open = true)")
    light = page.evaluate("""() => [...document.querySelectorAll('input,select,textarea')]
      .filter(e => e.offsetParent !== null && !['range', 'checkbox'].includes(e.type))
      .map(e => [e.id || e.type, getComputedStyle(e).backgroundColor.match(/[\\d.]+/g).map(Number)])
      .filter(([, c]) => (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / 255 > 0.5)
      .map(([name]) => name)""")
    assert light == []


def test_coastlines_are_drawn_at_the_size_they_are_shown(page):
    """The overlay was sized by the data's own resolution (M.nx): on a 60x60
    --generic grid an 84-pixel image stretched over the map, coasts as smudges.
    It must follow the map box on screen instead."""
    page.evaluate("() => overlay()")          # the size of the map box as it is now
    nx = int(re.search(r"nx=(\d+)", page.get_attribute("#over", "src")).group(1))
    box = page.locator("#wrap").bounding_box()
    dpr = page.evaluate("() => window.devicePixelRatio || 1")
    assert nx == round(box["width"] * 1.4 * dpr)


# ------------------------------------------------------------------ keys


@pytest.fixture(scope="module")
def keyed(tmp_path_factory, page):
    """Three steps, three levels and three fields: enough for every key to move.

    A second page on the `page` fixture's browser: one sync Playwright session
    per process, so a second `sync_playwright()` here would refuse to start.
    """
    from conftest import write_mesh
    from gmpas.viewer import PAGE, Viewer, _handler, bind

    folder = tmp_path_factory.mktemp("keys")
    rng = np.random.default_rng(1)
    lon = rng.uniform(-180, 180, 300)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, 300)))
    base = folder / "mesh.nc"
    write_mesh(base, list(zip(lon, lat, strict=True)))
    with xr.open_dataset(base) as ds:
        mesh = ds.load()
    for step in range(3):
        ds = mesh.copy(deep=True)
        ds["theta"] = (("Time", "nCells", "nVertLevels"),
                       (280 + step + np.arange(3)[None, :] + 0 * lat[:, None])[None])
        ds["qv"] = (("Time", "nCells"), (0.01 * (step + 1) + 0 * lat)[None])
        ds["w"] = (("Time", "nCells"), (0.1 * step + 0 * lat)[None])
        ds.to_netcdf(folder / f"history.2012-01-0{step + 1}_00.00.00.nc")
    base.unlink()
    viewer = Viewer(folder, nx=400, ny=250)
    srv = bind(_handler(viewer, PAGE), 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    pg = page.context.browser.new_page(viewport={"width": 1100, "height": 800})
    pg.goto(f"http://127.0.0.1:{srv.server_address[1]}/", wait_until="networkidle")
    pg.wait_for_function("() => M && !M.scanning && M.steps === 3", timeout=30000)
    pg.click("#vars div:has-text('theta')")
    pg.wait_for_timeout(300)
    yield pg
    pg.close()
    srv.shutdown()
    viewer.close()


def _val(pg, sel):
    return int(pg.eval_on_selector(sel, "e => e.value"))


def test_arrows_and_home_end_move_through_time(keyed):
    keyed.locator("body").click(position={"x": 5, "y": 5})
    keyed.keyboard.press("Home")
    assert _val(keyed, "#time") == 0
    keyed.keyboard.press("ArrowRight")
    assert _val(keyed, "#time") == 1
    assert keyed.inner_text("#tlab") == keyed.evaluate("() => M.labels[1]")
    keyed.keyboard.press("End")
    assert _val(keyed, "#time") == 2
    keyed.keyboard.press("ArrowRight")                   # stops at the end
    assert _val(keyed, "#time") == 2
    keyed.keyboard.press("ArrowLeft")
    assert _val(keyed, "#time") == 1


def test_up_and_down_move_the_level(keyed):
    keyed.keyboard.press("ArrowUp")
    keyed.keyboard.press("ArrowUp")
    assert _val(keyed, "#level") == 2 and keyed.inner_text("#llab") == "2"
    keyed.keyboard.press("ArrowDown")
    assert _val(keyed, "#level") == 1


def test_brackets_step_through_the_variables(keyed):
    names = keyed.eval_on_selector_all(
        "#vars div", "d => d.filter(x => x.offsetParent).map(x => x.textContent)")
    at = names.index(keyed.evaluate("() => cur.name"))
    keyed.keyboard.press("]")
    assert keyed.evaluate("() => cur.name") == names[(at + 1) % len(names)]
    keyed.keyboard.press("[")
    assert keyed.evaluate("() => cur.name") == names[at]


def test_zoom_and_reset_keys(keyed):
    keyed.keyboard.press("+")
    assert _val(keyed, "#zoom") == 50
    keyed.keyboard.press("-")
    assert _val(keyed, "#zoom") == 0
    keyed.keyboard.press("+")
    keyed.keyboard.press("r")
    assert _val(keyed, "#zoom") == 0


def test_space_plays_and_pauses(keyed):
    keyed.keyboard.press("Space")
    keyed.wait_for_function("() => playingKey !== null", timeout=15000)
    keyed.keyboard.press("Space")
    assert keyed.evaluate("() => playingKey") is None


def test_typing_in_a_field_is_not_a_key_command(keyed):
    before = _val(keyed, "#time"), keyed.evaluate("() => cur.name")
    keyed.click("#deriveExpr")
    keyed.keyboard.type("]r+ ")
    keyed.keyboard.press("ArrowRight")
    assert (_val(keyed, "#time"), keyed.evaluate("() => cur.name")) == before
    assert keyed.input_value("#deriveExpr") == "]r+ "
    keyed.fill("#deriveExpr", "")


def test_question_mark_toggles_the_key_help(keyed):
    keyed.locator("body").click(position={"x": 5, "y": 5})
    keyed.keyboard.press("?")
    assert keyed.is_visible("#keyhelp")
    keyed.keyboard.press("Escape")
    assert not keyed.is_visible("#keyhelp")


def test_the_key_marks_the_data_and_flags_a_clipping_range(page):
    """Ferret's KEYMARK, plus automatic triangles: with a colour range inside
    the data the key says the data runs past it on both sides; with the auto
    range it marks where the data really ends."""
    def redraw(vmin, vmax):
        with page.expect_response(lambda r: "api/frame" in r.url):
            page.fill("#vmin", vmin)
            page.fill("#vmax", vmax)
            page.dispatch_event("#vmax", "change")
        page.wait_for_timeout(100)

    with page.expect_response(lambda r: "api/frame" in r.url):
        page.evaluate("() => pick('theta')")
    with page.expect_response(lambda r: "api/frame" in r.url):
        page.click("#home")                           # earlier tests zoomed in
    redraw("275", "285")                              # theta runs 260..300 globally
    assert page.is_visible("#cbunder") and page.is_visible("#cbover")
    assert "below the colour range" in page.get_attribute("#cbunder", "title")
    assert "above the colour range" in page.get_attribute("#cbover", "title")
    assert page.locator("#ramp .mark").count() == 0   # both ends are off the bar

    redraw("250", "310")                              # covers the data
    assert not page.is_visible("#cbunder") and not page.is_visible("#cbover")
    assert page.locator("#ramp .mark").count() == 2
    assert page.inner_text("#cbdata").startswith("data ")

    # the automatic range is the 2nd-98th percentile and clips by design: no
    # triangles for it, or they would show on every first view
    redraw("", "")
    assert not page.is_visible("#cbunder") and not page.is_visible("#cbover")
    assert page.inner_text("#cbdata").startswith("data ")


# ------------------------------------------------------------ projections


def _frames_during(page, action, wait=600):
    """The api/frame URLs requested while `action` runs, and shortly after."""
    from urllib.parse import parse_qs, urlparse

    seen = []
    on = lambda r: seen.append(r.url) if "api/frame" in r.url else None  # noqa: E731
    page.on("request", on)
    try:
        action()
        page.wait_for_timeout(wait)
    finally:
        page.remove_listener("request", on)
    return [{k: v[0] for k, v in parse_qs(urlparse(u).query).items()} for u in seen]


def _lonlat(page):
    page.select_option("#proj", "")
    page.wait_for_timeout(400)
    page.click("#home")
    page.wait_for_timeout(400)


def test_a_flat_projection_frames_the_mesh_in_metres_unstretched(page):
    _lonlat(page)
    reqs = _frames_during(page, lambda: page.select_option("#proj", "Robinson"))
    assert reqs, "no frame requested"
    q = reqs[-1]
    assert q["proj"] == "Robinson" and "plon" in q
    x0, x1, y0, y1 = map(float, q["extent"].split(","))
    assert abs(x1 - x0) > 1e6                     # metres, not degrees
    # the box keeps the image's aspect: a projected metre is square on screen
    assert (x1 - x0) / (y1 - y0) == pytest.approx(int(q["nx"]) / int(q["ny"]), rel=0.01)
    assert page.get_attribute("#over", "src").count("proj=Robinson") == 1
    assert page.is_disabled("#grid") and page.is_disabled("#expnc")
    _lonlat(page)


def test_dragging_in_a_projection_pans_the_projected_box(page):
    _lonlat(page)
    page.select_option("#proj", "Robinson")
    page.wait_for_timeout(500)
    _zoom_in(page)
    before = page.evaluate("() => ({...view})")
    reqs = _frames_during(page, lambda: _drag(page, 120, 0))
    after = page.evaluate("() => ({...view})")
    assert after["clon"] < before["clon"] - 1e5   # moved west, in metres
    assert reqs and all(r["proj"] == "Robinson" for r in reqs)
    _lonlat(page)


def test_dragging_the_globe_turns_it_and_asks_for_one_frame(page):
    _lonlat(page)
    page.select_option("#proj", "Orthographic")
    page.wait_for_timeout(500)
    p0 = page.evaluate("() => ({...P})")
    reqs = _frames_during(page, lambda: _drag(page, 150, 60), wait=900)
    p1 = page.evaluate("() => ({...P})")
    assert p1["plon"] != pytest.approx(p0["plon"])
    assert p1["plat"] != pytest.approx(p0["plat"])
    # nothing redrawn while dragging, one exact frame when let go
    assert len(reqs) == 1
    assert float(reqs[0]["plon"]) == pytest.approx(p1["plon"])
    assert float(reqs[0]["plat"]) == pytest.approx(p1["plat"])
    _lonlat(page)


def test_a_probe_in_a_projection_finds_the_cell_under_the_click(page):
    _lonlat(page)
    page.select_option("#proj", "Orthographic")
    page.wait_for_timeout(600)
    box = page.locator("#wrap").bounding_box()
    with page.expect_response(lambda r: "api/probe" in r.url):
        page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.wait_for_timeout(200)
    pt = page.evaluate("() => probePt")
    # the centre of the globe is its centre point, by definition
    p = page.evaluate("() => ({...P})")
    assert pt["lon"] == pytest.approx(p["plon"], abs=1.0)
    assert pt["lat"] == pytest.approx(p["plat"], abs=1.0)
    want = page.evaluate(f"""async () => (await (await fetch(
        "api/probe?lon={pt['lon']}&lat={pt['lat']}&var=theta&time=0&level=0")).json()).cell""")
    assert f"cell {want}" in page.inner_text("#probe2")
    _lonlat(page)


def test_back_to_lon_lat_asks_for_exactly_the_old_frame(page):
    _lonlat(page)
    first = _frames_during(page, lambda: page.click("#home"))
    page.select_option("#proj", "NorthPolarStereo")
    page.wait_for_timeout(500)
    back = _frames_during(page, lambda: page.select_option("#proj", ""))
    assert back and "proj" not in back[-1]
    if first:
        assert back[-1]["extent"] == first[-1]["extent"]
    assert not page.is_disabled("#grid")


def test_auto_on_a_global_mesh_is_the_lon_lat_map(page):
    """auto keeps a global mesh on PlateCarree, which is today's map: it must
    be drawn as that map (degrees), not as a projection whose box is degrees
    the page would read as metres."""
    _lonlat(page)
    first = _frames_during(page, lambda: page.click("#home"))
    reqs = _frames_during(page, lambda: page.select_option("#proj", "auto"))
    assert page.evaluate("() => P") is None
    assert "lon/lat" in page.inner_text("#projhint")
    assert reqs and "proj" not in reqs[-1]
    if first:
        assert reqs[-1]["extent"] == first[-1]["extent"]
    _lonlat(page)
