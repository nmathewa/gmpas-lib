"""What another web site can make a browser send to a running viewer: a page
elsewhere can point an <img> at /api/chat, or rename itself to 127.0.0.1
(DNS rebinding) and read the answers. Both are refused; the page itself and
its own requests are not."""

from __future__ import annotations

import http.client
import threading
from http.server import ThreadingHTTPServer

import pytest

from gmpas import viewer


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), viewer._handler(None, "<p>page</p>"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


def _get(port, path="/", **headers):
    con = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    con.putrequest("GET", path, skip_host="Host" in headers)
    for k, v in headers.items():
        con.putheader(k.replace("_", "-"), v)
    con.endheaders()
    status = con.getresponse().status
    con.close()
    return status


def test_the_page_and_its_own_requests_are_answered(server):
    assert _get(server) == 200
    assert _get(server, Host=f"localhost:{server}") == 200
    assert _get(server, Host=f"127.0.0.1:{server}", Origin=f"http://127.0.0.1:{server}",
                Sec_Fetch_Site="same-origin") == 200
    assert _get(server, Host=f"[::1]:{server}") == 200


def test_a_host_that_is_not_this_machine_is_refused(server):
    assert _get(server, Host="evil.example") == 403
    assert _get(server, Host=f"rebind.attacker.com:{server}") == 403


def test_another_site_is_refused(server):
    assert _get(server, "/api/chat?text=x", Origin="http://evil.example") == 403
    assert _get(server, "/api/chat?text=x", Origin="null") == 403
    assert _get(server, "/api/chat?text=x", Sec_Fetch_Site="cross-site",
                Sec_Fetch_Mode="no-cors") == 403
    assert _get(server, "/api/dims", Sec_Fetch_Site="same-site") == 403


def test_a_link_from_another_site_still_opens_the_page(server):
    assert _get(server, "/", Sec_Fetch_Site="cross-site", Sec_Fetch_Mode="navigate") == 200
    # but not straight into an action
    assert _get(server, "/api/chat?text=x", Sec_Fetch_Site="cross-site",
                Sec_Fetch_Mode="navigate") == 403


def test_a_frame_size_is_bounded():
    assert viewer._px({"nx": "100000"}, "nx") == viewer.FRAME_MAX_PX
    assert viewer._px({"nx": "3"}, "nx") == 16
    assert viewer._px({}, "nx") is None
