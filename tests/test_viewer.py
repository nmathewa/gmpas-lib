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


def test_a_bind_error_that_is_not_addr_in_use_is_raised(servers):
    """Don't paper over a real failure by wandering up the port range."""
    with pytest.raises(OSError):
        bind(Quiet, 80, host="203.0.113.1")     # unassignable address
