"""Several things to look at, on one port.

These go through a real socket rather than calling the handlers directly: the
whole point of the change is the routing, and prefix stripping, redirects and
relative URL resolution are exactly what a direct call would skip.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from http.client import HTTPConnection

import pytest

from conftest import write_mesh
from gmpas.dashboard import Source, build, index_page, router, with_nav
from gmpas.viewer import bind

HFUN = """
import numpy as np

hfun_min = 200.0

def get_hfun(lon, lat):
    return np.full(lon.size, hfun_min)
"""


@pytest.fixture
def mesh_file(tmp_path):
    return write_mesh(tmp_path / "dash.mesh.nc",
                      [(float(i) * 3.0, 0.0) for i in range(12)])


@pytest.fixture
def hfun_file(tmp_path):
    p = tmp_path / "hfun.py"
    p.write_text(HFUN)
    return p


@pytest.fixture
def server(mesh_file, hfun_file):
    """A live two-source dashboard: a mesh and an hfun."""
    sources, _banner = build(mesh_path=str(mesh_file), hfun_path=str(hfun_file),
                             nx=40, ny=30)
    srv = bind(router(sources), 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", sources
    srv.shutdown()
    srv.server_close()


def get(base, path):
    with urllib.request.urlopen(base + path) as r:
        return r.status, r.read()


# ------------------------------------------------------------------ routing


def test_the_index_lists_every_source(server):
    base, sources = server
    status, body = get(base, "/")
    html = body.decode()

    assert status == 200
    assert [s.slug for s in sources] == ["mesh", "hfun"]
    for s in sources:
        assert f'href="/{s.slug}/"' in html
        assert s.detail in html


def test_each_source_serves_its_own_page(server):
    base, _ = server
    for slug in ("mesh", "hfun"):
        status, body = get(base, f"/{slug}/")
        assert status == 200
        assert b"<!doctype html>" in body.lower()


def test_each_source_has_its_own_api(server):
    base, _ = server
    mesh = json.loads(get(base, "/mesh/api/meta")[1])
    hfun = json.loads(get(base, "/hfun/api/meta")[1])

    assert mesh["facts_label"] == "mesh"
    assert mesh["cells"] == 12
    assert hfun["facts_label"] == "distance function"
    assert "cells" not in hfun


def test_a_prefix_without_a_slash_redirects_to_one(server):
    """`/mesh` must become `/mesh/`, or every relative api/... in the page
    would resolve against `/` and reach the wrong viewer."""
    base, _ = server
    host, port = base.rsplit("/", 1)[-1].split(":")
    conn = HTTPConnection(host, int(port))
    conn.request("GET", "/mesh")
    resp = conn.getresponse()
    assert resp.status == 301
    assert resp.getheader("Location") == "/mesh/"
    conn.close()


def test_an_unknown_source_is_a_404_not_a_traceback(server):
    base, _ = server
    with pytest.raises(urllib.error.HTTPError) as exc:
        get(base, "/nope/api/meta")
    assert exc.value.code == 404


def test_a_frame_comes_back_from_the_right_source(server):
    base, _ = server
    extent = "-40,40,-20,20"
    _, mesh_png = get(base, f"/mesh/api/frame?extent={extent}&field=cell_width_km")
    _, hfun_png = get(base, f"/hfun/api/frame?extent={extent}&field=cell_width_km")

    assert mesh_png[:8] == b"\x89PNG\r\n\x1a\n"
    assert hfun_png[:8] == b"\x89PNG\r\n\x1a\n"
    assert mesh_png != hfun_png


# -------------------------------------------------------------- the switcher


def test_every_page_carries_a_switcher_naming_the_others(server):
    base, sources = server
    for s in sources:
        html = get(base, f"/{s.slug}/")[1].decode()
        assert 'id="gmpas-nav"' in html
        for other in sources:
            assert f'href="/{other.slug}/"' in html


def test_the_current_source_is_marked_in_its_own_switcher(server):
    base, _ = server
    html = get(base, "/hfun/")[1].decode()
    assert '<a href="/hfun/" class="on">' in html
    assert '<a href="/mesh/" class="">' in html


def test_a_page_without_a_body_tag_is_left_alone():
    """Splicing is by string, so it must no-op rather than corrupt."""
    assert with_nav("not html at all", [], "x") == "not html at all"


def test_the_index_is_singular_for_one_source():
    src = [Source("mesh", "mesh", "one.nc", None)]
    assert "1 source " in index_page(src)


# ------------------------------------------------------------------ assembly


def test_a_mesh_and_an_hfun_are_two_sources(mesh_file, hfun_file):
    sources, banner = build(mesh_path=str(mesh_file), hfun_path=str(hfun_file))
    assert [s.slug for s in sources] == ["mesh", "hfun"]
    assert "max cell size gradient" in banner       # the hfun report is printed


def test_a_mesh_alone_is_one_source(mesh_file):
    sources, banner = build(mesh_path=str(mesh_file))
    assert [s.slug for s in sources] == ["mesh"]
    assert banner == ""


def test_nothing_to_view_is_refused_before_binding():
    with pytest.raises(ValueError, match="nothing to view"):
        build()


def test_a_bad_hfun_fails_on_the_command_line_not_in_a_tab(mesh_file, tmp_path):
    """Sources are constructed before the server binds, so this is an error
    the user sees at the prompt rather than a 500 in a browser."""
    from gmpas.prep.hfun import HfunError

    bad = tmp_path / "hfun.py"
    bad.write_text("hfun_min = 12.0\n")
    with pytest.raises(HfunError):
        build(mesh_path=str(mesh_file), hfun_path=str(bad))


# ------------------------------------------------- one source, as it always was


@pytest.fixture
def solo(mesh_file):
    """A one-source server: `gmpas prep view mesh.nc` with nothing added."""
    sources, _ = build(mesh_path=str(mesh_file), nx=40, ny=30)
    srv = bind(router(sources), 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def test_one_source_still_serves_at_the_root(solo):
    """The paths `prep view` always used must keep working unchanged."""
    assert get(solo, "/")[0] == 200
    assert json.loads(get(solo, "/api/meta")[1])["cells"] == 12
    assert get(solo, "/api/frame?extent=-40,40,-20,20")[1][:8] == b"\x89PNG\r\n\x1a\n"


def test_one_source_gets_no_switcher(solo):
    """A bar offering one destination is noise, not navigation."""
    assert 'id="gmpas-nav"' not in get(solo, "/")[1].decode()


def test_one_source_is_also_reachable_under_its_slug(solo):
    assert json.loads(get(solo, "/mesh/api/meta")[1])["cells"] == 12
