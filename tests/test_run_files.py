"""Which files of a run directory are its time axis, and files whose times
repeat each other."""

from __future__ import annotations

from pathlib import Path

from gmpas.generic import _overlap_note
from gmpas.series import _data_files


def _names(names):
    return [p.name for p in _data_files(Path(n) for n in names)]


def test_the_mesh_beside_the_output_is_not_a_time_step():
    assert _names(["history.2020-01-28_00.00.00.nc", "x1.41943042.grid.nc"]) == \
        ["history.2020-01-28_00.00.00.nc"]
    assert _names(["diag.2012-02-25_00.00.00.nc", "x1.40962.static.nc",
                   "x1.40962.init.nc", "init.nc", "mesh.nc"]) == \
        ["diag.2012-02-25_00.00.00.nc"]


def test_a_directory_of_mesh_files_alone_is_kept():
    assert _names(["x1.40962.init.nc"]) == ["x1.40962.init.nc"]


def test_files_without_times_in_their_names_are_all_kept():
    assert _names(["era5_t.nc", "era5_u.nc", "grid.nc"]) == \
        ["era5_t.nc", "era5_u.nc", "grid.nc"]


def test_files_that_repeat_each_others_times_are_named():
    a, b, c = Path("t.nc"), Path("u.nc"), Path("later.nc")
    days = ["2012-02-01 00:00", "2012-02-02 00:00"]
    note = _overlap_note([a, b], {a: days, b: days})
    assert "the same times in t.nc, u.nc" in note
    note = _overlap_note([a, c], {a: days, c: ["2012-02-01 12:00", "2012-02-03 00:00"]})
    assert "later.nc starts at 2012-02-01 12:00, before t.nc ends" in note
    assert _overlap_note([a, c], {a: days, c: ["2012-02-03 00:00"]}) == ""
