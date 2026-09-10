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


def test_a_second_stacking_axis_is_pinned_rather_than_dropped(tmp_path):
    """`select` refuses two free axes; a browser cannot show an exception.

    The slider takes the first axis, the rest are held at 0, and `describe`
    says so -- the alternative is hiding the field from the variable list.
    """
    import xarray as xr

    from conftest import write_mesh
    from gmpas.viewer import Viewer

    run = tmp_path / "run"
    run.mkdir()
    mesh = write_mesh(tmp_path / "m.nc", [(0.0, 0.0), (10.0, 0.0)])
    vals = np.arange(2 * 5 * 12, dtype="f4").reshape(2, 5, 12)
    xr.Dataset({
        "o3clim": (("nCells", "nOznLevels", "nMonths"), vals),
    }).to_netcdf(run / "history.2012-02-25_00.00.00.nc")

    v = Viewer(run, mesh_path=str(mesh), nx=20, ny=20)
    try:
        row = next(r for r in v.describe()["variables"] if r["name"] == "o3clim")
        assert row["levels"] == 5              # the slider drives nOznLevels
        assert row["dim"] == "nOznLevels"
        assert row["pinned"] == ["nMonths"]

        assert v.values("o3clim", 0, 3) == pytest.approx(vals[:, 3, 0])
    finally:
        v.series.close()


def test_the_values_cache_separates_two_pinnings_of_one_field(tmp_path):
    """(var, step, level) alone served January's slice for every month."""
    import xarray as xr

    from conftest import write_mesh
    from gmpas.series import Series

    run = tmp_path / "run"
    run.mkdir()
    mesh = write_mesh(tmp_path / "m.nc", [(0.0, 0.0), (10.0, 0.0)])
    vals = np.arange(2 * 5 * 12, dtype="f4").reshape(2, 5, 12)
    xr.Dataset({
        "o3clim": (("nCells", "nOznLevels", "nMonths"), vals),
    }).to_netcdf(run / "history.2012-02-25_00.00.00.nc")

    s = Series(run, mesh_path=str(mesh))
    try:
        jan = s.values("o3clim", step=0, level=3, sel={"nMonths": 0})
        jul = s.values("o3clim", step=0, level=3, sel={"nMonths": 6})
        assert jan == pytest.approx(vals[:, 3, 0])
        assert jul == pytest.approx(vals[:, 3, 6])
    finally:
        s.close()


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


def test_the_view_cache_does_not_grow_without_bound(small_viewer):
    """Nothing evicted this before: panning and zooming around over a long
    session -- exactly what setting up several animations for different
    variables looks like -- grew _views/_overlays forever, for the life of
    the server process.

    The bound is bytes, not entries. An entry is nx*ny*(8+1) bytes, so a count
    only bounds memory while the window size is fixed: twelve entries is
    ~178 MB at 1200x700 and ~1.1 GB at 3840x2160.
    """
    box = small_viewer.home
    budget = small_viewer._views.budget
    for i in range(40):
        small_viewer.view(box, 80 + i, 50)
        small_viewer.overlay(box, 80 + i, 50)

    assert small_viewer._views.nbytes <= budget
    assert small_viewer._overlays.nbytes <= budget
    # these entries are tiny, so nothing should have been evicted to hold 40
    assert len(small_viewer._views) == 40


def test_the_view_cache_evicts_once_the_budget_is_reached(small_viewer, monkeypatch):
    """The property that matters at scale, forced at fixture scale.

    A budget of a few kilobytes against 80x50 indices (~36 KB each) means the
    cache can hold one, which is exactly what a 4K browser window does to a
    256 MB budget.
    """
    from gmpas.cache import BuildCache

    small_viewer._views = BuildCache(budget=50_000)
    box = small_viewer.home
    for i in range(6):
        small_viewer.view(box, 80 + i, 50)

    assert small_viewer._views.nbytes <= 50_000
    assert len(small_viewer._views) < 6


def test_an_entry_larger_than_the_whole_budget_is_not_cached(small_viewer):
    """Caching it would evict everything else and still leave it as the sole
    occupant, to be evicted itself by the next distinct request. Same rule,
    and same reasoning, as Series._remember."""
    from gmpas.cache import BuildCache

    small_viewer._views = BuildCache(budget=100)
    view = small_viewer.view(small_viewer.home, 80, 50)

    assert view.nbytes > 100
    assert len(small_viewer._views) == 0
    # and it still returns a working index rather than failing
    assert view.idx.size == 80 * 50


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


# ------------------------------------------------------------------- export


def test_header_value_sanitizer_strips_crlf_and_quotes():
    """Content-Disposition is built from a filename derived from `var` and
    (when a file has no parseable timestamp) the file's own name -- and
    Linux filenames can contain raw CR/LF that send_header does not filter.

    Not reachable through /api/export/* itself: `var` must match a real
    dataset variable name before that code path is reached, and no valid
    netCDF variable name can carry a control character. This guards the
    filename-derived route (`label_of()`'s `path.stem` fallback) instead.
    """
    from gmpas.viewer import _safe_header_value

    assert _safe_header_value("evil\r\nX-Injected: yes\r\n") == "evilX-Injected: yes"
    assert "\r" not in _safe_header_value("a\rb\nc\x00d")
    assert "\n" not in _safe_header_value("a\rb\nc\x00d")
    assert _safe_header_value("ordinary_name.nc") == "ordinary_name.nc"


def test_figure_export_is_a_full_plot_not_the_bare_raster(small_viewer):
    """A screenshot of the viewer has no axes or colorbar; this does."""
    from PIL import Image

    png = small_viewer.figure("areaCell", 0, 0, small_viewer.home,
                              "viridis", None, None, style="notebook")
    with Image.open(io.BytesIO(png)) as im:
        assert im.size == (900, 500)          # notebook preset at 100 dpi


def test_netcdf_export_carries_the_grid_and_says_how_it_was_made(small_viewer,
                                                                 tmp_path):
    """Nearest-cell sampling is not a conservative remap, and must say so."""
    import xarray as xr

    out = tmp_path / "x.nc"
    out.write_bytes(small_viewer.netcdf("areaCell", 0, 0, small_viewer.home,
                                        40, 25))
    with xr.open_dataset(out) as ds:
        assert ds.sizes == {"lat": 25, "lon": 40}
        assert ds["areaCell"].dtype == np.float32     # source precision, not more
        assert "NOT area-conservative" in ds.attrs["method"]
        assert "antimeridian" in ds.attrs["longitude_convention"]
        assert ds.attrs["mesh_cells"] == small_viewer.mesh.n_cells


def test_gif_export_holds_every_step(tmp_path):
    """One frame per timestep, and the frames must actually differ."""
    import xarray as xr
    from PIL import Image

    from conftest import write_mesh
    from gmpas.viewer import Viewer

    run = tmp_path / "run"
    run.mkdir()
    for i in range(4):
        fp = run / f"history.2012-02-25_{i:02d}.00.00.nc"
        write_mesh(fp, [(0.0, 0.0), (5.0, 0.0), (0.0, 5.0)])
        with xr.open_dataset(fp) as d:
            ds = d.load()
        ds["fld"] = (("Time", "nCells"),
                     np.array([[i * 10.0, i * 20.0, i * 30.0]], dtype="f4"))
        ds.to_netcdf(fp, mode="w")

    v = Viewer(run, nx=60, ny=40)
    try:
        gif = v.gif("fld", 0, v.home, "viridis", 0.0, 100.0, 60, 40, fps=5)
        with Image.open(io.BytesIO(gif)) as im:
            assert im.n_frames == 4
    finally:
        v.series.close()


def test_concurrent_reads_do_not_corrupt_each_other(tmp_path, monkeypatch):
    """The bug this guards against: an animation's frame-by-frame loop and
    ordinary navigation (scrubbing, panning, a probe click) are never
    serialized by the UI, and both end up calling Series.values() on their
    own ThreadingHTTPServer request thread. Without a lock spanning the
    whole read -- cache lookup, LRU eviction, and the disk read itself --
    one thread's eviction can close a file another thread is mid-read on,
    which came back as NaN or a neighbour's value rather than an exception.

    Each file's field is a distinct, exactly-checkable constant so any
    cross-contamination or NaN is caught rather than merely suspected. A
    small LRU_SIZE against many more files forces the eviction this bug
    needs to happen on nearly every read, not rarely.
    """
    import threading

    import xarray as xr

    import gmpas.series as series_mod
    from conftest import write_mesh
    from gmpas.viewer import Viewer

    monkeypatch.setattr(series_mod, "LRU_SIZE", 2)

    run = tmp_path / "run"
    run.mkdir()
    n_files = 10
    for i in range(n_files):
        fp = run / f"history.2012-02-{i + 1:02d}_00.00.00.nc"
        write_mesh(fp, [(0.0, 0.0), (5.0, 0.0), (0.0, 5.0)])
        with xr.open_dataset(fp) as d:
            ds = d.load()
        ds["fld"] = (("Time", "nCells"), np.full((1, 3), i * 1000.0, dtype="f8"))
        ds.to_netcdf(fp, mode="w")

    v = Viewer(run, nx=20, ny=20)
    errors = []

    def hammer(step):
        try:
            for _ in range(30):
                got = v.series.values("fld", step=step, level=0)
                expected = step * 1000.0
                if not np.array_equal(got, np.full(3, expected)):
                    errors.append(
                        f"step {step}: expected all {expected}, got {got}")
        except Exception as exc:                # a race must not raise either
            errors.append(f"step {step}: {type(exc).__name__}: {exc}")

    try:
        threads = [threading.Thread(target=hammer, args=(i % n_files,))
                  for i in range(32)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
    finally:
        v.series.close()


def test_coastlines_stay_at_their_own_latitudes_past_a_pole():
    """Frames are rendered 1.4x wider than the window, so every global view
    asks for latitudes past +-90. `set_extent` used to clamp those to +-90 and
    then stretch the result to fill the figure, drawing the coastlines about
    1.4x too tall while the data raster spanned the box it was asked for. The
    two slid apart, and because the overshoot changes with zoom, they slid by a
    different amount at every zoom level.
    """
    import io

    import numpy as np
    from PIL import Image

    from gmpas.viewer import _overlay

    nx, ny = 400, 300
    lat_max = 126.0
    png = _overlay((-252.0, 252.0, -lat_max, lat_max), nx, ny)
    ink = np.array(Image.open(io.BytesIO(png)).convert("RGBA"))[..., 3] > 0

    rows = np.where(ink.any(axis=1))[0]
    top = lat_max - (rows.min() + 0.5) / ny * 2 * lat_max
    bottom = lat_max - (rows.max() + 0.5) / ny * 2 * lat_max

    # no land is drawn past a pole, so no ink may appear there either
    assert abs(top) <= 90.0
    assert abs(bottom) <= 90.0
    # and the band that is inked has to be real: northern Greenland reaches
    # into the 80s, so a coastline drawn only within +-60 would mean the
    # opposite mistake -- squashed rather than stretched
    assert top > 60.0
    assert bottom < -60.0


# ------------------------------------------------------ opening a browser


def test_a_headless_session_is_not_offered_a_browser(monkeypatch, capsys):
    """No DISPLAY and no $BROWSER: say so, rather than fail silently."""
    from gmpas.viewer import can_open_browser, open_in_browser

    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("BROWSER", raising=False)

    ok, why = can_open_browser()
    assert not ok
    assert "DISPLAY" in why

    open_in_browser("http://127.0.0.1:8765", delay=0)
    err = capsys.readouterr().err
    assert "not opening a browser" in err
    assert "http://127.0.0.1:8765" in err        # the URL stays actionable


def test_the_browser_env_var_is_enough_on_its_own(monkeypatch):
    """$BROWSER is how VS Code routes the URL back to a real browser."""
    from gmpas.viewer import can_open_browser

    monkeypatch.setattr("sys.platform", "linux")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("BROWSER", "/some/helper")

    ok, why = can_open_browser()
    assert ok and why == ""


def test_a_terminal_browser_is_refused_not_launched(monkeypatch, capsys):
    """elinks would seize the very terminal the server logs to."""
    import webbrowser

    from gmpas.viewer import open_in_browser

    class Console:
        name = "/usr/bin/elinks"
        def open(self, url, *a, **k):          # pragma: no cover - must not run
            raise AssertionError("a terminal browser must never be launched")

    monkeypatch.setenv("BROWSER", "/usr/bin/elinks")
    monkeypatch.setattr(webbrowser, "get", lambda *a, **k: Console())

    timer = open_in_browser("http://127.0.0.1:8765", delay=0)
    timer.join()
    err = capsys.readouterr().err
    assert "terminal browser" in err
    assert "http://127.0.0.1:8765" in err


def test_a_browser_that_will_not_start_is_reported(monkeypatch, capsys):
    """webbrowser.open()'s False used to be discarded inside a Timer."""
    import webbrowser

    from gmpas.viewer import open_in_browser

    class Dud:
        name = "firefox"
        def open(self, url, *a, **k):
            return False                       # could not launch

    monkeypatch.setenv("BROWSER", "firefox")
    monkeypatch.setattr(webbrowser, "get", lambda *a, **k: Dud())

    timer = open_in_browser("http://127.0.0.1:8765", delay=0)
    timer.join()
    assert "did not start" in capsys.readouterr().err


def test_a_working_browser_is_opened_and_stays_quiet(monkeypatch, capsys):
    import webbrowser

    from gmpas.viewer import open_in_browser

    opened = []

    class Good:
        name = "firefox"
        def open(self, url, *a, **k):
            opened.append(url)
            return True

    monkeypatch.setenv("BROWSER", "firefox")
    monkeypatch.setattr(webbrowser, "get", lambda *a, **k: Good())

    timer = open_in_browser("http://127.0.0.1:8765", delay=0)
    timer.join()
    assert opened == ["http://127.0.0.1:8765"]
    assert capsys.readouterr().err == ""        # success says nothing


# ------------------------------------------------- how to reach the server


def test_the_tunnel_command_is_filled_in_not_a_placeholder(monkeypatch):
    """A command still holding <host> is one the reader has to decode."""
    import getpass
    import socket

    from gmpas.viewer import reach_lines

    monkeypatch.setattr(socket, "gethostname", lambda: "dec1042")
    monkeypatch.setattr(getpass, "getuser", lambda: "someone")
    monkeypatch.setattr(socket, "getfqdn", lambda *a: "dec1042.hpc.example.edu")

    text = "\n".join(reach_lines("127.0.0.1", 8765))
    assert "ssh -N -L 8765:localhost:8765 someone@dec1042.hpc.example.edu" in text
    assert "http://localhost:8765" in text
    assert "<host>" not in text and "<user>" not in text


def test_binding_all_interfaces_outside_a_job_is_reached_directly(monkeypatch):
    """Not in a batch job -- a login node, say -- means no hop is needed.

    The old banner sent this reader through "<login-node>", which on a login
    node is the machine they are already on.
    """
    import getpass
    import socket

    from gmpas.viewer import reach_lines

    monkeypatch.setattr(socket, "gethostname", lambda: "dec1042")
    monkeypatch.setattr(getpass, "getuser", lambda: "someone")
    monkeypatch.setattr(socket, "getfqdn", lambda *a: "dec1042.hpc.example.edu")
    for var in ("PBS_JOBID", "SLURM_JOB_ID"):
        monkeypatch.delenv(var, raising=False)

    text = "\n".join(reach_lines("0.0.0.0", 8765))
    # fully qualified, so it can be typed straight into ssh
    assert "ssh -N -L 8765:localhost:8765 someone@dec1042.hpc.example.edu" in text
    assert "<login-node>" not in text          # nothing left to substitute


def test_a_bogus_fqdn_falls_back_to_the_short_name(monkeypatch):
    """getfqdn() answers from /etc/hosts and can be useless.

    Printing `someone@localhost` as the thing to SSH to would be worse than
    the bare hostname, which is at least honest about being incomplete.
    """
    import socket

    from gmpas.viewer import ssh_target

    monkeypatch.setattr(socket, "gethostname", lambda: "dec1042")

    monkeypatch.setattr(socket, "getfqdn", lambda *a: "localhost")
    assert ssh_target() == "dec1042"

    monkeypatch.setattr(socket, "getfqdn", lambda *a: "dec1042.hpc.example.edu")
    assert ssh_target() == "dec1042.hpc.example.edu"


def test_a_batch_job_tunnels_via_the_submitting_host(monkeypatch):
    """A compute node is not reachable from outside; the submit host is.

    PBS and Slurm both name that host in the environment, so the command can
    be complete without hardcoding any site's address into gmpas.
    """
    import getpass
    import socket

    from gmpas.viewer import reach_lines

    monkeypatch.setattr(socket, "gethostname", lambda: "dec0965")
    monkeypatch.setattr(getpass, "getuser", lambda: "someone")
    monkeypatch.setenv("PBS_JOBID", "1234567.desched1")
    monkeypatch.setenv("PBS_O_HOST", "login.cluster.example.org")

    text = "\n".join(reach_lines("0.0.0.0", 8787))
    assert ("ssh -N -L 8787:dec0965:8787 someone@login.cluster.example.org"
            in text)
    assert "<login-node>" not in text          # nothing left to substitute


def test_slurm_is_understood_as_well_as_pbs(monkeypatch):
    import getpass
    import socket

    from gmpas.viewer import reach_lines

    monkeypatch.setattr(socket, "gethostname", lambda: "nid001")
    monkeypatch.setattr(getpass, "getuser", lambda: "someone")
    monkeypatch.delenv("PBS_JOBID", raising=False)
    monkeypatch.delenv("PBS_O_HOST", raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "99")
    monkeypatch.setenv("SLURM_SUBMIT_HOST", "head.cluster.example.org")

    text = "\n".join(reach_lines("0.0.0.0", 8787))
    assert "ssh -N -L 8787:nid001:8787 someone@head.cluster.example.org" in text


def test_a_job_without_a_submit_host_still_says_something_useful(monkeypatch):
    """Some sites unset it; fall back to a placeholder rather than a wrong name."""
    import getpass
    import socket

    from gmpas.viewer import reach_lines

    monkeypatch.setattr(socket, "gethostname", lambda: "dec0965")
    monkeypatch.setattr(getpass, "getuser", lambda: "someone")
    monkeypatch.setenv("PBS_JOBID", "1234567")
    monkeypatch.delenv("PBS_O_HOST", raising=False)
    monkeypatch.delenv("SLURM_SUBMIT_HOST", raising=False)

    text = "\n".join(reach_lines("0.0.0.0", 8787))
    assert "ssh -N -L 8787:dec0965:8787 someone@<login-node>" in text


# ------------------------------------------- not opening a browser by default


def test_a_delegating_opener_counts_as_a_terminal_browser_when_headless():
    """`xdg-open` is the name gmpas sees; w3m is what it launches.

    CONSOLE_BROWSERS names every terminal browser, and caught none of this:
    the controller reports itself as the delegator. With nothing graphical
    running, whatever it hands the URL to renders into this terminal.
    """
    import os

    from gmpas.viewer import _console_browser

    class Ctl:
        name = "/usr/bin/xdg-open"

    seen = {k: os.environ.pop(k, None) for k in ("DISPLAY", "WAYLAND_DISPLAY")}
    try:
        assert _console_browser(Ctl()) is True
        os.environ["DISPLAY"] = ":0"
        assert _console_browser(Ctl()) is False    # a desktop: it opens a window
    finally:
        os.environ.pop("DISPLAY", None)
        for k, v in seen.items():
            if v is not None:
                os.environ[k] = v


def test_view_does_not_open_a_browser_unless_asked(monkeypatch):
    """The default is to print the URL, not to launch anything.

    On a login node the realistic outcome of launching was a text browser
    rendering the viewer's JavaScript shell over the log the user was reading.
    """
    from gmpas import cli

    opened = []
    monkeypatch.setattr("gmpas.dashboard.build",
                        lambda *a, **k: ([], "banner"))
    monkeypatch.setattr("gmpas.dashboard.serve",
                        lambda *a, **k: opened.append(k["open_browser"]))

    args = cli.build_parser().parse_args(["view", "run/"])
    assert args.browser is False
    cli._dashboard(args, data_path=args.path)
    assert opened == [False]

    args = cli.build_parser().parse_args(["view", "run/", "--browser"])
    cli._dashboard(args, data_path=args.path)
    assert opened == [False, True]


def test_the_old_no_browser_flag_still_parses():
    """It is in every job script already, and now asks for the default."""
    from gmpas import cli

    args = cli.build_parser().parse_args(["view", "run/", "--no-browser"])
    assert args.browser is False


def test_a_loopback_only_fqdn_is_not_offered_as_an_ssh_target(monkeypatch):
    """A stock /etc/hosts maps the machine's own FQDN to 127.0.1.1.

    That name looks perfectly plausible -- `box.example.dom` -- and is a
    loopback alias, so the ssh command built from it connects to nothing.
    """
    import socket

    from gmpas import viewer

    monkeypatch.setattr(socket, "gethostname", lambda: "box")
    monkeypatch.setattr(socket, "getfqdn", lambda *a: "box.example.dom")
    monkeypatch.setattr(viewer, "_loopback_only", lambda name, timeout=1.0: True)

    assert viewer.ssh_target() == "box"


def test_a_name_that_does_not_resolve_here_is_left_alone(monkeypatch):
    """Unresolvable is not disproved: login nodes resolve from outside.

    Dropping a correct hostname is the worse error, since the reader can
    sanity-check one that is shown and cannot recover one that is not.
    """
    import socket

    from gmpas import viewer

    monkeypatch.setattr(socket, "gethostname", lambda: "dec1042")
    monkeypatch.setattr(socket, "getfqdn", lambda *a: "dec1042.hpc.example.edu")
    monkeypatch.setattr(socket, "getaddrinfo",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("no")))

    assert viewer.ssh_target() == "dec1042.hpc.example.edu"


def test_a_job_bound_to_loopback_is_told_the_one_thing_that_helps(monkeypatch):
    """Inside a job there is no tunnel command that reaches loopback.

    The hop lands on the login node, where 127.0.0.1 is a different machine.
    Printing a tunnel there would be printing a command that cannot work.
    """
    import socket

    from gmpas.viewer import reach_lines

    monkeypatch.setattr(socket, "gethostname", lambda: "dec1042")
    monkeypatch.setenv("PBS_JOBID", "123456.desched1")
    monkeypatch.setenv("PBS_O_HOST", "derecho7")

    text = "\n".join(reach_lines("127.0.0.1", 8765))
    assert "--host 0.0.0.0" in text
    assert "derecho7" in text
    assert "ssh -N -L" not in text            # no command that cannot land


def test_nothing_launches_a_browser_behind_the_guards():
    """`webbrowser.open` walks its whole list, and that list ends in elinks.

    `gmpas prep view` and `gmpas prep hfun` each carried their own copy of
    serve() that called it directly, so neither the terminal-browser refusal
    nor the opt-in default applied to them: they opened a text browser over
    the running server's own log. The single guarded path in viewer.py, which
    resolves exactly one controller and vets it, is the only one left.
    """
    import pathlib

    import gmpas

    root = pathlib.Path(gmpas.__file__).parent
    users = sorted(p.relative_to(root).as_posix()
                   for p in root.rglob("*.py")
                   if "webbrowser" in p.read_text())
    assert users == ["viewer.py"]

    assert "webbrowser.open(" not in (root / "viewer.py").read_text()
