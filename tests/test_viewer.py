"""The viewer's server plumbing. Rendering is covered by test_plot."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler

import pytest

from gmpas.viewer import PORT_ATTEMPTS, bind


class Quiet(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass


@pytest.fixture
def servers():
    opened = []
    yield opened
    for s in opened:
        s.server_close()


def test_a_free_port_is_used_as_asked(servers):
    s = bind(Quiet, 0)                 # 0 asks the OS for anything free
    servers.append(s)
    assert s.server_address[1] > 0


def test_a_busy_port_falls_forward(servers):
    """A viewer already running in another terminal must not block this one."""
    first = bind(Quiet, 8765)
    servers.append(first)
    port = first.server_address[1]

    second = bind(Quiet, port)
    servers.append(second)

    assert second.server_address[1] != port
    assert second.server_address[1] > port


def test_an_exhausted_range_falls_back_to_any_free_port(servers):
    """Rather than refusing to start, let the OS choose."""
    base = bind(Quiet, 0).server_address[1]
    held = []
    for offset in range(PORT_ATTEMPTS):
        try:
            held.append(bind(Quiet, base + offset, attempts=1))
        except OSError:
            pass
    servers.extend(held)
    taken = {s.server_address[1] for s in held}

    last = bind(Quiet, base, attempts=len(taken) or 1)
    servers.append(last)
    assert last.server_address[1] > 0


def test_an_explicit_port_fails_rather_than_moving(servers):
    """Somebody who named a port has a tunnel pointing at it.

    Listening somewhere else would leave them watching a browser that can
    never load, which is worse than a clear refusal.
    """
    held = bind(Quiet, 0)
    servers.append(held)
    port = held.server_address[1]

    with pytest.raises(OSError, match="cannot bind"):
        bind(Quiet, port, strict=True)


def test_binding_all_interfaces_is_possible(servers):
    """Loopback is unreachable from a login node; HPC needs 0.0.0.0."""
    s = bind(Quiet, 0, host="0.0.0.0")
    servers.append(s)
    assert s.server_address[0] == "0.0.0.0"


def test_a_bind_error_that_is_not_addr_in_use_is_raised(servers):
    """Don't paper over a real failure by wandering up the port range."""
    with pytest.raises(OSError):
        bind(Quiet, 80, host="203.0.113.1")     # unassignable address


# ------------------------------------------------------------- colour ramps


def test_ramp_returns_hex_stops_from_the_real_colormap():
    """The browser's bar must show the colours the image actually uses."""
    from gmpas.viewer import ramp

    stops = ramp("viridis", 5)
    assert len(stops) == 5
    assert all(s.startswith("#") and len(s) == 7 for s in stops)
    assert stops[0] == "#440154"          # viridis really does start here
    assert stops[-1] == "#fde725"


def test_every_offered_colormap_has_a_ramp():
    from gmpas.viewer import CMAPS, ramp

    for name in CMAPS:
        assert len(ramp(name, 8)) == 8


def test_a_diverging_ramp_is_light_in_the_middle():
    """Sanity that the sampling is ordered, not shuffled."""
    from gmpas.viewer import ramp

    stops = ramp("RdBu_r", 5)
    def lum(h):
        return sum(int(h[i:i + 2], 16) for i in (1, 3, 5))
    assert lum(stops[2]) > lum(stops[0])
    assert lum(stops[2]) > lum(stops[-1])
