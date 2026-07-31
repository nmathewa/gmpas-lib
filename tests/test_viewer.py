"""The viewer's server plumbing. Rendering is covered by test_plot."""

from __future__ import annotations

import io
from http.server import BaseHTTPRequestHandler

import numpy as np
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


# ------------------------------------------------- frames at arbitrary size


@pytest.fixture
def small_viewer(tmp_path):
    """A viewer over a tiny two-cell mesh, enough to exercise the plumbing."""
    from conftest import write_mesh
    from gmpas.viewer import Viewer

    run = tmp_path / "run"
    run.mkdir()
    write_mesh(run / "history.2012-02-25_00.00.00.nc", [(0.0, 0.0), (10.0, 0.0)])
    v = Viewer(run, nx=80, ny=50)
    yield v
    v.series.close()


def test_the_view_cache_keys_on_size_not_just_extent(small_viewer):
    """Panning renders a margin, so the same extent is requested at two sizes.

    Keying on extent alone handed back an index built for a different pixel
    grid, and the gather then reshaped into the wrong dimensions.
    """
    box = small_viewer.home

    a = small_viewer.view(box, 80, 50)
    b = small_viewer.view(box, 112, 70)

    assert a is not b
    assert a.idx.size == 80 * 50
    assert b.idx.size == 112 * 70
    assert small_viewer.view(box, 80, 50) is a       # still cached


def test_a_frame_can_be_asked_for_at_a_larger_size(small_viewer):
    """The margin is fetched at proportionally more pixels to stay sharp."""
    from PIL import Image

    png, _, _ = small_viewer.frame("areaCell", 0, 0, small_viewer.home,
                                   "viridis", None, None, nx=112, ny=70)
    with Image.open(io.BytesIO(png)) as im:
        assert im.size == (112, 70)


def test_frames_default_to_the_viewer_size(small_viewer):
    from PIL import Image

    png, _, _ = small_viewer.frame("areaCell", 0, 0, small_viewer.home,
                                   "viridis", None, None)
    with Image.open(io.BytesIO(png)) as im:
        assert im.size == (80, 50)


# ------------------------------------------------------------- frame encoding


def test_frames_are_palette_pngs_not_rgba(small_viewer):
    """A colormap has 256 entries, so 32-bit RGBA says nothing extra."""
    from PIL import Image

    png, _, _ = small_viewer.frame("areaCell", 0, 0, small_viewer.home,
                                   "viridis", 0.0, 1.0)
    with Image.open(io.BytesIO(png)) as im:
        assert im.mode == "P"
        assert im.info.get("transparency") == 255


def test_off_mesh_pixels_stay_transparent(small_viewer):
    """The map must show through where the mesh does not reach."""
    from PIL import Image

    png, _, _ = small_viewer.frame("areaCell", 0, 0,
                                   (-170.0, -160.0, -80.0, -70.0),
                                   "viridis", 0.0, 1.0)
    with Image.open(io.BytesIO(png)) as im:
        assert (np.asarray(im.convert("RGBA"))[..., 3] == 0).all()


def test_an_explicit_range_is_used_verbatim(small_viewer):
    """Animation fixes the range, and nothing may quietly re-derive it."""
    _, lo, hi = small_viewer.frame("areaCell", 0, 0, small_viewer.home,
                                   "viridis", -5.0, 12.5)
    assert (lo, hi) == (-5.0, 12.5)


def test_a_degenerate_range_is_widened(small_viewer):
    """A constant field would otherwise divide by zero."""
    _, lo, hi = small_viewer.frame("areaCell", 0, 0, small_viewer.home,
                                   "viridis", 3.0, 3.0)
    assert hi > lo


def test_compression_level_changes_size_not_content(small_viewer):
    from PIL import Image

    fast, _, _ = small_viewer.frame("areaCell", 0, 0, small_viewer.home,
                                    "viridis", 0.0, 1.0, compress=1)
    small, _, _ = small_viewer.frame("areaCell", 0, 0, small_viewer.home,
                                     "viridis", 0.0, 1.0, compress=9)
    a = np.asarray(Image.open(io.BytesIO(fast)).convert("RGBA"))
    b = np.asarray(Image.open(io.BytesIO(small)).convert("RGBA"))
    assert np.array_equal(a, b)          # lossless either way
