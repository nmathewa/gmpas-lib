"""One point's time series: the per-cell read, and what it costs.

The MPAS side is the interesting one. `Series.values` materialises a whole
field per step, so the obvious implementation of a point series moves the
entire run through memory to collect one number per step. These tests pin the
two properties that make the real implementation usable: it agrees with the
slow way exactly, and it does not touch the values cache.
"""

from __future__ import annotations

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
