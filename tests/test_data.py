"""Pairing output with a mesh, and reducing a field to one value per element."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from conftest import write_diag, write_mesh
from gmpas import field_label, open_data, open_mpas, plottable, select, spatial_dim


@pytest.fixture
def diag_beside_mesh(tmp_path, simple_mesh_file):
    """A diagnostics file with no mesh information, next to its mesh."""
    return write_diag(tmp_path / "diag.2019-09-01_00.00.00.nc",
                      n_cells=4, n_edges=24)


# -------------------------------------------------------------------- pairing


def test_mesh_is_found_beside_the_data_by_matching_cell_count(diag_beside_mesh):
    ds, mesh = open_data(diag_beside_mesh)
    assert mesh.n_cells == 4
    assert mesh.path.name == "simple.mesh.nc"
    ds.close()


def test_explicit_mesh_path_wins(diag_beside_mesh, simple_mesh_file):
    ds, mesh = open_data(diag_beside_mesh, simple_mesh_file)
    assert mesh.n_cells == 4
    ds.close()


def test_a_file_carrying_its_own_mesh_needs_no_second_file(simple_mesh_file):
    ds, mesh = open_data(simple_mesh_file)
    assert mesh.path == simple_mesh_file
    ds.close()


def test_unpairable_data_says_what_to_pass(tmp_path):
    """A mesh with the wrong cell count must not be silently accepted."""
    write_mesh(tmp_path / "other.mesh.nc", [(0.0, 0.0)])          # 1 cell
    diag = write_diag(tmp_path / "diag.nc", n_cells=4, n_edges=24)

    with pytest.raises(KeyError, match="mesh_path"):
        open_data(diag)


def test_missing_data_file_is_reported_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="No such data file"):
        open_data(tmp_path / "absent.nc")


# ------------------------------------------------------------------ selection


def test_time_and_level_are_indexed_away(diag_beside_mesh):
    ds, _ = open_data(diag_beside_mesh)

    assert select(ds.mslp, time=1).shape == (4,)
    assert select(ds.theta, time=2, level=3).shape == (4,)
    ds.close()


def test_selection_picks_the_right_slice(diag_beside_mesh):
    ds, _ = open_data(diag_beside_mesh)

    assert select(ds.mslp, time=2) == pytest.approx(ds.mslp.values[2])
    assert select(ds.theta, time=1, level=2) == pytest.approx(
        ds.theta.values[1, 2]
    )
    ds.close()


def test_negative_indices_work_like_numpy(diag_beside_mesh):
    ds, _ = open_data(diag_beside_mesh)
    assert select(ds.mslp, time=-1) == pytest.approx(ds.mslp.values[-1])
    ds.close()


def test_out_of_range_index_names_the_dimension_and_size(diag_beside_mesh):
    ds, _ = open_data(diag_beside_mesh)
    with pytest.raises(IndexError, match="Time=7"):
        select(ds.mslp, time=7)
    with pytest.raises(IndexError, match="nVertLevels=9"):
        select(ds.theta, level=9)
    ds.close()


# ------------------------------------------------------------------- grouping


def test_variables_are_grouped_by_the_element_they_live_on(diag_beside_mesh):
    ds, _ = open_data(diag_beside_mesh)
    groups = plottable(ds)

    assert groups["nCells"] == ["mslp", "theta"]
    assert groups["nEdges"] == ["u"]
    assert groups["nVertices"] == ["vorticity"]
    ds.close()


def test_spatial_dim_identifies_the_element(diag_beside_mesh):
    ds, _ = open_data(diag_beside_mesh)
    assert spatial_dim(ds.mslp) == "nCells"
    assert spatial_dim(ds.u) == "nEdges"
    ds.close()


def test_a_field_on_no_mesh_element_cannot_be_drawn():
    da = xr.DataArray(np.zeros(3), dims=("Time",), name="t_index")
    with pytest.raises(ValueError, match="cannot be drawn"):
        spatial_dim(da)


def test_label_uses_long_name_and_units_when_present(diag_beside_mesh):
    ds, _ = open_data(diag_beside_mesh)
    assert field_label(ds.mslp) == "mean sea level pressure [Pa]"
    assert field_label(ds.theta) == "theta [K]"
    ds.close()


# ------------------------------------------------------------------- accessor


def test_accessor_attaches_the_mesh_and_reports_it(diag_beside_mesh):
    ds = open_mpas(diag_beside_mesh)
    assert ds.mpas.mesh.n_cells == 4
    assert "4 cells" in repr(ds.mpas)
    ds.close()


def test_accessor_without_a_mesh_says_how_to_attach_one(tmp_path):
    path = write_diag(tmp_path / "lonely.nc", n_cells=4, n_edges=24)
    ds = open_mpas(path)

    with pytest.raises(ValueError, match="use_mesh"):
        _ = ds.mpas.mesh
    ds.close()


def test_use_mesh_attaches_one_after_the_fact(tmp_path, simple_mesh_file):
    sub = tmp_path / "sub"
    sub.mkdir()
    ds = open_mpas(write_diag(sub / "lonely.nc", n_cells=4, n_edges=24))
    ds.mpas.use_mesh(simple_mesh_file)

    assert ds.mpas.mesh.n_cells == 4
    ds.close()


def test_vertex_fields_are_refused_with_a_reason(diag_beside_mesh):
    ds = open_mpas(diag_beside_mesh)
    with pytest.raises(NotImplementedError, match="dual triangle mesh"):
        ds.mpas.plot("vorticity")
    ds.close()


def test_unknown_variable_lists_what_is_available(diag_beside_mesh):
    ds = open_mpas(diag_beside_mesh)
    with pytest.raises(KeyError, match="mslp"):
        ds.mpas.plot("no_such_field")
    ds.close()


# ------------------------------------------------------- multi-file series


def _run_dir(tmp_path, n_files, n_times=1):
    """A directory of MPAS-style history files, mesh in the first."""
    from conftest import write_mesh

    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    for i in range(n_files):
        path = run / f"history.2012-02-25_{i:02d}.00.00.nc"
        write_mesh(path, [(0.0, 0.0), (10.0, 0.0)])
        if n_times > 1:
            with xr.open_dataset(path) as ds:
                extra = ds.load()
            extra["fld"] = (("Time", "nCells"),
                            np.zeros((n_times, 2)) + i)
            extra.to_netcdf(path, mode="w")
    return run


def test_a_background_scan_serves_before_it_finishes(tmp_path):
    """The provisional axis is one step per file, available immediately."""
    from gmpas.series import Series

    run = _run_dir(tmp_path, n_files=5)
    s = Series(run, background_scan=True)
    try:
        assert len(s) == 5              # usable at once, no waiting
        assert [step[1] for step in s.steps] == [0] * 5
    finally:
        s.close()


def test_the_provisional_axis_is_a_subset_never_wrong(tmp_path):
    """Every provisional step maps to a real timestep, so no frame is bogus.

    Files here hold 3 steps each. Before the scan lands the axis shows one
    per file; afterwards it shows all of them. What it showed first was
    correct, just incomplete.
    """
    import time

    from gmpas.series import Series

    run = _run_dir(tmp_path, n_files=4, n_times=3)

    provisional = Series(run, background_scan=True)
    try:
        first_view = list(provisional.steps)
        for _ in range(200):
            if not provisional.scanning:
                break
            time.sleep(0.02)
        settled = list(provisional.steps)
    finally:
        provisional.close()

    # file 0 is already open for the mesh, so its 3 steps are known exactly;
    # the other three files are assumed to hold one each until scanned
    assert len(first_view) == 3 + 3
    assert len(settled) == 12            # 3 per file once counted
    for step in first_view:
        assert step in settled           # nothing shown was ever wrong


def test_an_eager_series_counts_everything_up_front(tmp_path):
    from gmpas.series import Series

    run = _run_dir(tmp_path, n_files=4, n_times=3)
    s = Series(run)                      # default: no background scan
    try:
        assert not s.scanning
        assert len(s) == 12
    finally:
        s.close()


def test_a_single_file_is_counted_immediately(tmp_path):
    """One file needs no scan -- it is already open for the mesh."""
    from gmpas.series import Series

    run = _run_dir(tmp_path, n_files=1, n_times=3)
    s = Series(run, background_scan=True)
    try:
        assert not s.scanning
        assert len(s) == 3
    finally:
        s.close()


# ------------------------------------------------- the axis from filenames


def test_timestamps_are_read_from_filenames(tmp_path):
    """MPAS puts the valid time in the name, so no file need be opened."""
    from gmpas.series import label_of, parse_time

    cases = {
        "history.2012-02-25_12.00.00.nc": "2012-02-25 12:00",
        "history.2012-02-25_12.00.nc": "2012-02-25 12:00",     # no seconds
        "history.2012-02-25T12:30:45.nc": "2012-02-25 12:30:45",
        "diag.2019-09-01_00.00.00.nc": "2019-09-01 00:00",
    }
    for name, expected in cases.items():
        assert parse_time(Path(name)) is not None, name
        assert label_of(Path(name)) == expected


def test_a_name_without_a_time_falls_back_to_the_stem(tmp_path):
    from gmpas.series import label_of, parse_time

    assert parse_time(Path("output.nc")) is None
    assert label_of(Path("output.nc")) == "output"


def test_something_that_only_looks_like_a_stamp_is_rejected():
    """Don't build a time axis out of a coincidence."""
    from gmpas.series import parse_time

    assert parse_time(Path("run.2012-02-30_12.00.00.nc")) is None   # no Feb 30
    assert parse_time(Path("run.2012-02-25_25.00.00.nc")) is None   # hour 25


def test_files_are_ordered_chronologically_not_alphabetically():
    """Text order is only chronological by luck of MPAS's zero padding."""
    from gmpas.series import order

    names = [Path("run.2012-03-09_00.00.00.nc"),
             Path("b.2012-03-01_06.00.00.nc"),
             Path("a.2012-03-01_00.00.00.nc")]
    assert [p.name for p in order(names)] == [
        "a.2012-03-01_00.00.00.nc",
        "b.2012-03-01_06.00.00.nc",
        "run.2012-03-09_00.00.00.nc",
    ]


def test_undated_files_fall_back_to_name_order():
    from gmpas.series import order

    names = [Path("z.nc"), Path("a.2012-03-01_00.00.00.nc"), Path("m.nc")]
    assert [p.name for p in order(names)] == ["a.2012-03-01_00.00.00.nc",
                                             "m.nc", "z.nc"]


def test_a_series_exposes_its_times_without_reading_files(tmp_path):
    from gmpas.series import Series

    run = _run_dir(tmp_path, n_files=3)
    s = Series(run, background_scan=True)
    try:
        assert s.dated
        assert [t.hour for t in s.times] == [0, 1, 2]
    finally:
        s.close()


# --------------------------------------------------- the values cache budget


def test_the_values_cache_is_bounded_by_bytes_not_entries(tmp_path):
    """The regression that killed a Derecho login node.

    The cache was capped at 64 *entries*, which is ~130 MB of small-mesh
    fields but ~20 GB of 41M-cell ones. Scrubbing a long series therefore
    grew without any bound that scaled with the mesh. The budget is in bytes
    now, so the same walk stays inside it whatever the field size.
    """
    from gmpas.series import Series

    run = _run_dir(tmp_path, n_files=6, n_times=3)
    s = Series(run)
    try:
        s._values_budget = 200            # bytes: room for a couple of fields
        for step in range(len(s)):
            s.values("fld", step=step)
        assert s._values_bytes <= s._values_budget
        assert s._values_bytes == sum(v.nbytes for v in s._values.values())
    finally:
        s.close()


def test_a_field_larger_than_the_whole_budget_is_not_cached(tmp_path):
    """Caching it would evict everything and still be evicted next read."""
    from gmpas.series import Series

    run = _run_dir(tmp_path, n_files=2, n_times=2)
    s = Series(run)
    try:
        s._values_budget = 1              # smaller than any real field
        got = s.values("fld", step=0)
        assert got is not None            # still returned to the caller
        assert not s._values                # but nothing retained
        assert s._values_bytes == 0
    finally:
        s.close()


def test_closing_a_series_releases_the_cached_fields(tmp_path):
    from gmpas.series import Series

    run = _run_dir(tmp_path, n_files=2, n_times=2)
    s = Series(run)
    s.values("fld", step=0)
    assert s._values_bytes > 0
    s.close()
    assert s._values_bytes == 0
    assert not s._values


def test_the_values_budget_honours_its_environment_override(monkeypatch):
    from gmpas.series import VALUES_CACHE_BYTES, VALUES_CACHE_ENV, values_budget

    monkeypatch.setenv(VALUES_CACHE_ENV, "64")
    assert values_budget() == 64 * 1024 * 1024

    # an unparseable value must not take the viewer down on startup
    monkeypatch.setenv(VALUES_CACHE_ENV, "not-a-number")
    assert values_budget() == VALUES_CACHE_BYTES

    monkeypatch.delenv(VALUES_CACHE_ENV)
    assert values_budget() == VALUES_CACHE_BYTES
