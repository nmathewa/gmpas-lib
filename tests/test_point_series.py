"""One point's time series: the per-cell read, and what it costs.

The MPAS side is the interesting one. `Series.values` materialises a whole
field per step, so the obvious implementation of a point series moves the
entire run through memory to collect one number per step. These tests pin the
two properties that make the real implementation usable: it agrees with the
slow way exactly, and it does not touch the values cache.
"""

from __future__ import annotations

import os
import threading
import tracemalloc

import numpy as np
import pytest
import xarray as xr

from gmpas.viewer import Viewer


@pytest.fixture
def run(tmp_path):
    """Twelve steps, one per file, with a 3-D field and a known pattern."""
    from conftest import write_mesh

    folder = tmp_path / "run"
    folder.mkdir()
    base = tmp_path / "mesh.nc"
    write_mesh(base, [(0.0, 0.0), (10.0, 0.0), (5.0, 8.0), (-6.0, 4.0)])
    mesh = xr.open_dataset(base)
    cells = mesh.sizes["nCells"]
    for step in range(12):
        ds = mesh.copy(deep=True)
        # value encodes (step, cell, level): 100*step + cell + level/10
        theta = (100 * step + np.arange(cells)[:, None]
                 + np.arange(4)[None, :] / 10.0)
        ds["theta"] = (("Time", "nCells", "nVertLevels"),
                       theta[None].astype("f8"), {"units": "K"})
        ds.to_netcdf(folder / f"history.2012-02-{step + 1:02d}_00.00.00.nc")
    mesh.close()
    v = Viewer(folder, nx=40, ny=30)
    yield v
    v.close()


def test_a_cell_series_is_what_reading_every_step_would_have_said(run):
    """The fast path and the slow one must agree exactly, not nearly."""
    slow = [float(run.values("theta", step, 2)[1]) for step in range(len(run.series))]
    fast = run.series.at_cell("theta", 1, 2)
    assert fast.tolist() == slow
    assert fast.tolist() == [100 * s + 1 + 0.2 for s in range(12)]


def test_the_level_and_the_cell_both_choose(run):
    assert run.series.at_cell("theta", 3, 0)[0] == pytest.approx(3.0)
    assert run.series.at_cell("theta", 3, 3)[0] == pytest.approx(3.3)


def test_a_field_with_no_level_axis_reads_too(run):
    values = run.series.at_cell("areaCell", 2)
    assert values.size == len(run.series) and np.isfinite(values).all()


def test_a_cell_outside_the_mesh_is_refused(run):
    with pytest.raises(IndexError, match="outside nCells"):
        run.series.at_cell("theta", 9999, 0)


def test_reading_a_series_leaves_the_values_cache_alone(run):
    """A full-field read per step would evict everything the map is using --
    the reason this path exists at all."""
    run.values("theta", 0, 0)                          # something worth keeping
    before = dict(run.series._values)
    run.series.at_cell("theta", 1, 0)
    assert dict(run.series._values).keys() == before.keys()
    assert run.series._values_bytes == sum(a.nbytes for a in before.values())


def test_the_whole_read_holds_only_the_answer(run):
    """Peak allocation is the series itself, not a field per step."""
    tracemalloc.start()
    try:
        run.series.at_cell("theta", 1, 0)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    # generous: the point is orders of magnitude, not a tight bound
    assert peak < 2 * 1024 * 1024, f"peak {peak / 1024:.0f} kB"


def test_progress_counts_files_and_cancel_stops_between_them(run):
    seen = []
    run.series.at_cell("theta", 1, 0, progress=lambda done, total: seen.append(done))
    assert seen == list(range(1, 13))

    from gmpas.jobs import Cancelled

    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        run.series.at_cell("theta", 1, 0, cancel=cancel)


# ------------------------------------------------------------- the viewer


def _wait(viewer, *args, tries=200, **kwargs):
    for _ in range(tries):
        state = viewer.series_at_point(*args, **kwargs)
        if state["state"] != "running":
            return state
        threading.Event().wait(0.01)
    raise AssertionError("series never finished")


def test_clicking_a_point_gives_the_series_of_the_cell_under_it(run):
    state = _wait(run, 10.0, 0.0, "theta", 1)
    assert state["cell"] == 1 and state["lon"] == 10.0 and state["lat"] == 0.0
    assert state["values"] == [100 * s + 1 + 0.1 for s in range(12)]
    assert state["labels"][0].startswith("2012-02-01")
    assert "K" in state["label"]


def test_the_first_call_does_not_block_and_the_page_polls(run):
    first = run.series_at_point(10.0, 0.0, "theta", 0)
    assert first["state"] == "running" and len(first["progress"]) == 2
    assert _wait(run, 10.0, 0.0, "theta", 0)["state"] == "done"


def test_an_export_can_ask_for_it_outright(run):
    state = run.series_at_point(10.0, 0.0, "theta", 0, blocking=True)
    assert state["state"] == "done" and len(state["values"]) == 12


def test_a_derived_expression_is_refused_by_name(run):
    with pytest.raises(ValueError, match="derived expression"):
        run.series_at_point(10.0, 0.0, "theta - theta", 0)


def test_a_run_longer_than_the_limit_is_refused_before_it_reads(run, monkeypatch):
    monkeypatch.setattr("gmpas.viewer.MAX_SERIES_STEPS", 3)
    with pytest.raises(ValueError, match="past the 3 a point series"):
        run.series_at_point(10.0, 0.0, "theta", 0)


def test_frames_are_still_served_while_a_series_is_read(run):
    """The read takes the lock per file and yields between them; a map request
    must not wait for the whole run."""
    done = threading.Event()

    def read():
        run.series_at_point(10.0, 0.0, "theta", 0, blocking=True)
        done.set()

    thread = threading.Thread(target=read)
    thread.start()
    png, _, _ = run.frame("theta", 0, 0, run.home, "viridis", None, None, 40, 30)
    assert png[:4] == b"\x89PNG"
    assert done.wait(30)
    thread.join()


# -------------------------------------------------------------- over HTTP


def _serve(viewer):
    import threading as th

    from gmpas.viewer import PAGE, _handler, bind

    srv = bind(_handler(viewer, PAGE), 0)
    th.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _poll(base, query, tries=200):
    import json
    import urllib.parse
    import urllib.request

    url = f"{base}/api/series?{urllib.parse.urlencode(query)}"
    saw_202 = False
    for _ in range(tries):
        with urllib.request.urlopen(url) as r:
            saw_202 |= r.status == 202
            body = json.loads(r.read())
        if body["state"] == "done":
            return body, saw_202
        threading.Event().wait(0.01)
    raise AssertionError("series never finished over HTTP")


def test_the_route_answers_202_while_reading_then_the_numbers(run):
    srv, base = _serve(run)
    try:
        body, saw_202 = _poll(base, {"lon": 10.0, "lat": 0.0, "var": "theta",
                                     "level": 1})
    finally:
        srv.shutdown()
    assert saw_202, "the first call should not have blocked"
    assert body["values"] == [100 * s + 1 + 0.1 for s in range(12)]
    assert body["cell"] == 1 and len(body["labels"]) == 12


def test_a_bad_request_is_an_error_not_a_hang(run):
    import json
    import urllib.error
    import urllib.parse
    import urllib.request

    srv, base = _serve(run)
    q = urllib.parse.urlencode({"lon": 10.0, "lat": 0.0, "var": "nope"})
    try:
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(f"{base}/api/series?{q}")
        assert "nope" in json.loads(err.value.read())["error"]
    finally:
        srv.shutdown()


# ------------------------------------------------------- the generic side


@pytest.fixture
def grid(tmp_path):
    """Three files of two steps each on a regular grid."""
    import pandas as pd

    from gmpas.generic import GenericViewer

    lat, lon = np.linspace(-10, 10, 5), np.linspace(0, 40, 9)
    for k, month in enumerate(("01", "02", "03")):
        t = 100 * k + np.arange(2)[:, None, None] + np.zeros((2, lat.size, lon.size))
        xr.Dataset({"t2m": (("time", "lat", "lon"), t, {"units": "K"})},
                   coords={"time": pd.date_range(f"2024-{month}-01", periods=2),
                           "lat": lat, "lon": lon}
                   ).to_netcdf(tmp_path / f"era5_{month}.nc")
    gv = GenericViewer(tmp_path)
    yield gv
    gv.close()


def test_both_viewers_answer_a_point_series_the_same_way(grid):
    state = _wait(grid, 20.0, 0.0, "t2m", 0)
    assert set(state) == {"state", "cell", "lon", "lat", "label", "labels", "values"}
    assert state["values"] == [0.0, 1.0, 100.0, 101.0, 200.0, 201.0]
    assert state["labels"][0].startswith("2024-01-01")


def test_a_point_off_the_grid_says_so(grid):
    with pytest.raises(ValueError, match="outside the grid"):
        grid.series_at_point(300.0, 80.0, "t2m", 0)


def test_the_generic_series_reports_progress_and_can_be_cancelled(grid):
    seen = []
    grid._series_at("t2m", 20.0, 0.0, 0, progress=lambda d, t: seen.append((d, t)))
    assert seen == [(1, 3), (2, 3), (3, 3)]

    from gmpas.jobs import Cancelled

    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        grid._series_at("t2m", 20.0, 0.0, 0, cancel=cancel)


def test_closing_a_series_waits_for_its_background_scan(tmp_path):
    """A scan left reading past its Series meets whatever writes netCDF next,
    and HDF5 answers a concurrent read-and-write with a segfault, not a stale
    value. Closing must leave no thread in the library."""
    import threading as th

    from conftest import write_mesh
    from gmpas.series import Series

    folder = tmp_path / "run"
    folder.mkdir()
    for step in range(6):
        write_mesh(folder / f"history.2012-03-{step + 1:02d}_00.00.00.nc",
                   [(0.0, 0.0), (10.0, 0.0)])
    series = Series(folder, background_scan=True)
    series.close()
    assert series.scanning is False
    assert not any(t.name == "gmpas-scan" and t.is_alive()
                   for t in th.enumerate())


# ------------------------------------------------------- reading in parallel


def test_the_parallel_read_agrees_with_the_serial_one_exactly(run):
    """A faster wrong answer is worse than a slow right one."""
    serial = run.series.at_cell("theta", 1, 2, workers=1)
    parallel = run.series.at_cell("theta", 1, 2, workers=4)
    assert parallel.tolist() == serial.tolist()


def test_a_short_run_does_not_pay_for_workers(run, monkeypatch):
    """Below the threshold the pool costs more than the read it parallelises."""
    def refuse(*a, **k):
        raise AssertionError("a 12-file run should not start a pool")

    monkeypatch.setattr("gmpas.series.series_pool", refuse)
    assert len(run.series.steps) < 32
    assert run.series.at_cell("theta", 1, 0, workers=8).size == 12


def test_the_worker_count_comes_from_the_environment(monkeypatch):
    from gmpas import series as S

    monkeypatch.setenv(S.SERIES_WORKERS_ENV, "3")
    assert S.series_workers() == 3
    monkeypatch.setenv(S.SERIES_WORKERS_ENV, "not a number")
    assert S.series_workers() == max(1, min(S.SERIES_WORKERS, os.cpu_count() or 1))
    monkeypatch.setenv(S.SERIES_WORKERS_ENV, "1")
    assert S.series_workers() == 1                  # one worker means no pool


def test_the_pool_never_forks_beside_an_open_hdf5(monkeypatch):
    """fork would hand a child a half-held HDF5 mutex; anything else is fine."""
    from gmpas.series import _pool_context

    assert _pool_context().get_start_method() in ("forkserver", "spawn")


def test_only_the_asked_for_steps_are_read(run):
    """The preview reads a stride and leaves the rest NaN, so the chart draws
    a line with gaps rather than pretending to values it has not read."""
    wanted = [0, 4, 8]
    out = run.series.at_cell("theta", 1, 0, steps=wanted, workers=1)
    assert [out[i] for i in wanted] == [100 * s + 1 for s in wanted]
    assert np.isnan([out[i] for i in range(12) if i not in wanted]).all()


def test_the_preview_is_the_same_numbers_as_the_full_read(run):
    from gmpas.viewer import _point_series

    seen = {}
    full = _point_series(run.series, "theta", 1, 0, {}, None, None,
                         lambda partial: seen.setdefault("preview", partial))
    assert full.tolist() == run.series.at_cell("theta", 1, 0, workers=1).tolist()
    if "preview" in seen:                    # only long runs take the two-phase path
        drawn = np.isfinite(seen["preview"])
        assert np.array_equal(seen["preview"][drawn], full[drawn])


def test_a_preview_is_never_cached_as_the_answer(run):
    """Polling must not leave a strided series behind as though it were whole:
    what is cached is what the work returned, never what it published."""
    _wait(run, 10.0, 0.0, "theta", 0)
    key = ("theta", 1, 0, ())
    cached = run._jobs.peek(key)
    assert cached is not None and np.isfinite(cached).all()
    again = run.series_at_point(10.0, 0.0, "theta", 0)
    assert again["state"] == "done" and again["values"].count(None) == 0


@pytest.fixture
def long_run(tmp_path):
    """Enough files to take the parallel, two-phase path for real."""
    from conftest import write_mesh

    folder = tmp_path / "long"
    folder.mkdir()
    base = tmp_path / "m.nc"
    write_mesh(base, [(0.0, 0.0), (10.0, 0.0)])
    mesh = xr.open_dataset(base)
    for step in range(40):
        ds = mesh.copy(deep=True)
        ds["theta"] = (("Time", "nCells"),
                       np.full((1, mesh.sizes["nCells"]), float(step)))
        ds.to_netcdf(folder / f"history.2012-03-{step + 1:02d}_00.00.00.nc")
    mesh.close()
    v = Viewer(folder, nx=20, ny=15)
    yield v
    v.close()


def test_a_long_run_is_read_in_parallel_and_previewed_first(long_run, monkeypatch):
    """The whole point, end to end: a strided picture arrives before the full
    series, and the full series is right."""
    import gmpas.viewer as V

    monkeypatch.setattr(V, "PREVIEW_STEPS", 8)
    seen = []
    full = V._point_series(long_run.series, "theta", 0, 0, {}, None, None,
                           seen.append)
    assert full.tolist() == [float(s) for s in range(40)]
    assert seen, "a 40-file run should have published a preview"
    preview = seen[0]
    drawn = np.isfinite(preview)
    assert 2 <= drawn.sum() <= 12                  # a stride, not the lot
    assert np.array_equal(preview[drawn], full[drawn])


def test_the_page_is_given_the_preview_to_draw(long_run, monkeypatch):
    import gmpas.viewer as V

    monkeypatch.setattr(V, "PREVIEW_STEPS", 8)
    for _ in range(400):
        state = long_run.series_at_point(10.0, 0.0, "theta", 0)
        if state["state"] == "done":
            break
        if state.get("preview"):
            assert any(v is not None for v in state["values"])
            assert any(v is None for v in state["values"])   # gaps, honestly
            break
        threading.Event().wait(0.01)
    assert long_run.series_at_point(10.0, 0.0, "theta", 0, blocking=True)["state"] \
        == "done"
