"""Applying weights, and the one command that runs the whole conversion."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from conftest import write_mesh
from gmpas.config import TargetDomain
from gmpas.remap import (RemapError, Weights, _esmf_supports_mpi,
                         _mpi_launch_prefix, ensure_weights, level_dim,
                         remappable, remap_file, valid_times)


def _weight_file(path, row, col, S, area_a, area_b, frac_a, frac_b):
    xr.Dataset({
        "row": ("n_s", np.asarray(row, dtype=np.int32)),
        "col": ("n_s", np.asarray(col, dtype=np.int32)),
        "S": ("n_s", np.asarray(S, dtype=float)),
        "area_a": ("n_a", np.asarray(area_a, dtype=float)),
        "area_b": ("n_b", np.asarray(area_b, dtype=float)),
        "frac_a": ("n_a", np.asarray(frac_a, dtype=float)),
        "frac_b": ("n_b", np.asarray(frac_b, dtype=float)),
    }).to_netcdf(path)
    return Weights.load(path)


@pytest.fixture
def weights(tmp_path):
    """Genuinely conservative weights, built by hand.

    Two source cells of area 4. Destination 0 (area 2) sits wholly inside
    source 0; destination 1 (area 6) takes the other 2 of source 0 plus all 4
    of source 1. With ESMF's dstarea normalisation S is overlap / area_b.
    """
    return _weight_file(
        tmp_path / "map.nc",
        row=[1, 2, 2], col=[1, 1, 2],
        S=[2 / 2, 2 / 6, 4 / 6],
        area_a=[4.0, 4.0], area_b=[2.0, 6.0],
        frac_a=[1.0, 1.0], frac_b=[1.0, 1.0],
    )


@pytest.fixture
def partial(tmp_path):
    """A destination cell only half covered by the source.

    Destination 0 (area 6) receives 3 units of overlap from source 0 (area 4),
    so frac_b is 0.5 and frac_a is 0.75. This is where multiplying by frac_b
    a second time goes wrong.
    """
    return _weight_file(
        tmp_path / "partial.nc",
        row=[1], col=[1], S=[3 / 6],
        area_a=[4.0], area_b=[6.0],
        frac_a=[0.75], frac_b=[0.5],
    )


def test_indices_are_converted_from_one_based(weights):
    """SCRIP counts from 1; forgetting that shifts every value by a cell."""
    assert weights.row.min() == 0
    assert weights.col.min() == 0
    assert (weights.n_a, weights.n_b) == (2, 2)


def test_applying_weights_is_the_sparse_product(weights):
    dst = weights.apply(np.array([4.0, 8.0]))
    assert dst == pytest.approx([4.0, 2 / 6 * 4.0 + 4 / 6 * 8.0])


def test_a_constant_field_stays_constant(weights):
    """Consistency: the weights into each destination cell sum to one."""
    assert weights.apply(np.ones(2)) == pytest.approx([1.0, 1.0])


def test_the_sparse_matrix_is_built_once_and_reused(weights):
    """apply() runs once per field/level/timestep slab over a real run --
    rebuilding the CSR matrix from row/col/S every call would throw away
    the whole point of caching it."""
    assert weights._matrix is None
    weights.apply(np.array([4.0, 8.0]))
    m = weights._matrix
    assert m is not None
    weights.apply(np.array([1.0, 2.0]))
    assert weights._matrix is m


def test_duplicate_row_col_pairs_accumulate_like_add_at_did(tmp_path):
    """csr_matrix construction from (row, col, S) triples sums duplicates
    automatically -- confirming that switching apply() from np.add.at to a
    precomputed sparse matvec is not an approximation of the old behaviour,
    it produces identical output. Two entries land on the same (row, col)."""
    w = _weight_file(
        tmp_path / "dup.nc",
        row=[1, 1], col=[1, 1], S=[0.5, 0.25],   # same destination and source cell
        area_a=[1.0], area_b=[1.0], frac_a=[1.0], frac_b=[1.0],
    )
    dst = w.apply(np.array([4.0]))
    assert dst == pytest.approx([0.5 * 4.0 + 0.25 * 4.0])


def test_the_area_integral_is_preserved(weights):
    for src in (np.ones(2), np.array([4.0, 8.0]), np.array([-1.5, 2.25])):
        dst = weights.apply(src)
        assert weights.conservation_error(src, dst) == pytest.approx(0.0, abs=1e-14)


def test_partial_coverage_conserves_without_double_counting(partial):
    """The frac_b trap, with numbers that actually expose it.

    A half-covered destination cell conserves when the integral uses area_b
    alone. Multiplying by frac_b as well halves it, which is how exact weights
    came to look 0.2% wrong.
    """
    src = np.array([2.0])
    dst = partial.apply(src)

    assert dst == pytest.approx([1.0])                  # 0.5 * 2, i.e. frac_b * value
    assert partial.conservation_error(src, dst) == pytest.approx(0.0, abs=1e-14)

    right = float((dst * partial.area_b).sum())
    wrong = float((dst * partial.area_b * partial.frac_b).sum())
    truth = float((src * partial.area_a * partial.frac_a).sum())
    assert right == pytest.approx(truth)
    assert wrong == pytest.approx(truth / 2)            # exactly the double count


# ------------------------------------------------------------ field triage


def test_only_cell_fields_can_use_cell_weights(tmp_path):
    """Velocity lives on edges and vorticity on vertices; both need their own."""
    from conftest import write_diag

    path = write_diag(tmp_path / "d.nc", n_cells=4, n_edges=24)
    with xr.open_dataset(path) as ds:
        keep, skip = remappable(ds, ["mslp", "theta", "u", "vorticity", "absent"])

    assert keep == ["mslp", "theta"]
    reasons = dict(skip)
    assert "nEdges" in reasons["u"]
    assert "nVertices" in reasons["vorticity"]
    assert "not in this file" in reasons["absent"]


def test_the_level_dimension_is_found(tmp_path):
    from conftest import write_diag

    path = write_diag(tmp_path / "d.nc", n_cells=4, n_edges=24)
    with xr.open_dataset(path) as ds:
        assert level_dim(ds.theta) == "nVertLevels"
        assert level_dim(ds.mslp) is None


# -------------------------------------------------------------- weight file


def test_existing_weights_are_reused(tmp_path):
    """Weights depend only on the grids, so a run must not rebuild them."""
    domain = TargetDomain(4, 8, -2.0, 2.0, 0.0, 8.0)
    (tmp_path / "map_conserve.nc").write_bytes(b"not really a weight file")

    path, built = ensure_weights(tmp_path / "unused.nc", domain, tmp_path,
                                 quiet=True)
    assert path.name == "map_conserve.nc"
    assert built is False                      # reused, and ESMF never ran


def test_a_missing_esmf_says_how_to_install_it(tmp_path, monkeypatch):
    import gmpas.remap as remap_mod

    monkeypatch.setattr(remap_mod.shutil, "which", lambda name: None)
    write_mesh(tmp_path / "m.nc", [(0.0, 0.0), (5.0, 0.0)])
    domain = TargetDomain(4, 8, -2.0, 2.0, 0.0, 8.0)

    with pytest.raises(RemapError, match="module load esmf"):
        ensure_weights(tmp_path / "m.nc", domain, tmp_path / "w", quiet=True)


def test_a_crashing_esmf_is_retried_then_reported(tmp_path, monkeypatch):
    """ESMF 8.9.1 segfaults intermittently, so a one-off retry is warranted."""
    import gmpas.remap as remap_mod

    calls = []

    class Result:
        returncode = -11
        stdout = "boom"
        stderr = ""

    monkeypatch.setattr(remap_mod.shutil, "which", lambda name: "/fake/esmf")
    monkeypatch.setattr(remap_mod.subprocess, "run",
                        lambda *a, **k: (calls.append(1), Result())[1])
    write_mesh(tmp_path / "m.nc", [(0.0, 0.0), (5.0, 0.0)])
    domain = TargetDomain(4, 8, -2.0, 2.0, 0.0, 8.0)

    with pytest.raises(RemapError, match="segfaulted"):
        ensure_weights(tmp_path / "m.nc", domain, tmp_path / "w", quiet=True)
    assert len(calls) == remap_mod.WEIGHT_ATTEMPTS


# -------------------------------------------------- ESMF's own MPI awareness


def _write_esmf_mk(path, comm=None, style="comment-colon"):
    """A minimal stand-in for ESMF's build-info makefile fragment.

    Real esmf.mk files bury `-DESMF_COMM=...` inside long *COMPILECPPFLAGS
    lines too -- `_esmf_supports_mpi` must not be fooled by those, so the
    fixture includes one, matching the real file this was written against
    (conda-forge esmf 8.9.1's mpiuni build).
    """
    lines = [
        "ESMF_F90COMPILECPPFLAGS=-DESMF_NO_INTEGER_1_BYTE "
        "-DESMF_COMM=not_the_real_value -DESMF_DIR=/wherever",
    ]
    if comm is not None:
        line = {"comment-colon": f"# ESMF_COMM: {comm}",
                "comment-equals": f"#  ESMF_COMM={comm}"}[style]
        lines.append(line)
    path.write_text("\n".join(lines) + "\n")


def test_a_mpiuni_build_is_not_mpi_capable(tmp_path, monkeypatch):
    monkeypatch.delenv("ESMFMKFILE", raising=False)
    tool = tmp_path / "bin" / "ESMF_RegridWeightGen"
    tool.parent.mkdir()
    (tmp_path / "lib").mkdir()
    _write_esmf_mk(tmp_path / "lib" / "esmf.mk", comm="mpiuni")

    assert _esmf_supports_mpi(str(tool)) is False


def test_a_real_mpi_build_is_detected(tmp_path, monkeypatch):
    monkeypatch.delenv("ESMFMKFILE", raising=False)
    tool = tmp_path / "bin" / "ESMF_RegridWeightGen"
    tool.parent.mkdir()
    (tmp_path / "lib").mkdir()
    _write_esmf_mk(tmp_path / "lib" / "esmf.mk", comm="mpich",
                   style="comment-equals")

    assert _esmf_supports_mpi(str(tool)) is True


def test_the_path_beside_the_resolved_tool_wins_over_esmfmkfile(tmp_path, monkeypatch):
    """The bug this guards against: `module load esmf` (real MPI) sets
    $ESMFMKFILE, then `conda activate` shadows PATH so the resolved tool is
    actually the conda mpiuni copy. $ESMFMKFILE is now stale -- it describes
    a build that isn't the one about to run. Trusting it over the file next
    to the *actual* resolved binary would call the mpiuni copy MPI-capable
    and reintroduce the corruption this function exists to prevent."""
    tool = tmp_path / "bin" / "ESMF_RegridWeightGen"
    tool.parent.mkdir()
    (tmp_path / "lib").mkdir()
    _write_esmf_mk(tmp_path / "lib" / "esmf.mk", comm="mpiuni")   # the real answer

    stale_module = tmp_path / "elsewhere.mk"
    _write_esmf_mk(stale_module, comm="openmpi")   # describes a *different* build
    monkeypatch.setenv("ESMFMKFILE", str(stale_module))

    assert _esmf_supports_mpi(str(tool)) is False


def test_esmfmkfile_is_used_only_when_the_local_path_has_nothing(tmp_path, monkeypatch):
    tool = tmp_path / "bin" / "ESMF_RegridWeightGen"
    tool.parent.mkdir()
    # no lib/esmf.mk beside the tool at all

    elsewhere = tmp_path / "elsewhere.mk"
    _write_esmf_mk(elsewhere, comm="openmpi")
    monkeypatch.setenv("ESMFMKFILE", str(elsewhere))

    assert _esmf_supports_mpi(str(tool)) is True


def test_unknown_when_no_esmf_mk_can_be_found(tmp_path, monkeypatch):
    monkeypatch.delenv("ESMFMKFILE", raising=False)
    tool = tmp_path / "bin" / "ESMF_RegridWeightGen"
    tool.parent.mkdir()   # no lib/esmf.mk beside it

    assert _esmf_supports_mpi(str(tool)) is None


def test_unknown_when_esmf_mk_never_states_esmf_comm(tmp_path, monkeypatch):
    monkeypatch.delenv("ESMFMKFILE", raising=False)
    tool = tmp_path / "bin" / "ESMF_RegridWeightGen"
    tool.parent.mkdir()
    (tmp_path / "lib").mkdir()
    _write_esmf_mk(tmp_path / "lib" / "esmf.mk", comm=None)   # only the flags line

    assert _esmf_supports_mpi(str(tool)) is None


# --------------------------------------------------------------- MPI launcher


def test_a_single_rank_needs_no_launcher():
    assert _mpi_launch_prefix(1, "/fake/esmf") == ([], None)
    assert _mpi_launch_prefix(0, "/fake/esmf") == ([], None)


def test_mpirun_is_used_outside_a_slurm_allocation(monkeypatch):
    import gmpas.remap as remap_mod

    monkeypatch.setattr(remap_mod, "_esmf_supports_mpi", lambda tool: True)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(remap_mod.shutil, "which",
                        lambda name: "/opt/mpirun" if name == "mpirun" else None)

    assert _mpi_launch_prefix(64, "/fake/esmf") == (["/opt/mpirun", "-np", "64"], None)


def test_mpiexec_is_used_when_mpirun_is_absent(monkeypatch):
    """mpiexec is the HPE Cray name for the same launcher."""
    import gmpas.remap as remap_mod

    monkeypatch.setattr(remap_mod, "_esmf_supports_mpi", lambda tool: True)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(remap_mod.shutil, "which",
                        lambda name: "/opt/cray/mpiexec" if name == "mpiexec" else None)

    assert _mpi_launch_prefix(64, "/fake/esmf") == (["/opt/cray/mpiexec", "-np", "64"], None)


def test_srun_is_preferred_inside_an_active_slurm_allocation(monkeypatch):
    import gmpas.remap as remap_mod

    monkeypatch.setattr(remap_mod, "_esmf_supports_mpi", lambda tool: True)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setattr(remap_mod.shutil, "which",
                        lambda name: f"/opt/{name}")   # srun and mpirun both "exist"

    assert _mpi_launch_prefix(64, "/fake/esmf") == (["/opt/srun", "-n", "64"], None)


def test_srun_is_not_used_outside_an_allocation_even_if_installed(monkeypatch):
    """A laptop can have Slurm client tools on PATH with no job running."""
    import gmpas.remap as remap_mod

    monkeypatch.setattr(remap_mod, "_esmf_supports_mpi", lambda tool: True)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(remap_mod.shutil, "which",
                        lambda name: f"/opt/{name}")

    assert _mpi_launch_prefix(64, "/fake/esmf") == (["/opt/mpirun", "-np", "64"], None)


def test_no_launcher_on_path_falls_back_to_a_bare_command(monkeypatch):
    import gmpas.remap as remap_mod

    monkeypatch.setattr(remap_mod, "_esmf_supports_mpi", lambda tool: True)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(remap_mod.shutil, "which", lambda name: None)

    prefix, note = _mpi_launch_prefix(64, "/fake/esmf")
    assert prefix == []
    assert "no srun/mpirun/mpiexec" in note


def test_a_mpiuni_build_declines_the_launcher_with_a_reason(monkeypatch):
    """The gate that matters: don't corrupt output on a build that can't
    coordinate ranks, and don't do it silently either."""
    import gmpas.remap as remap_mod

    monkeypatch.setattr(remap_mod, "_esmf_supports_mpi", lambda tool: False)
    monkeypatch.setattr(remap_mod.shutil, "which", lambda name: f"/opt/{name}")

    prefix, note = _mpi_launch_prefix(64, "/fake/esmf")
    assert prefix == []
    assert "mpiuni" in note


def test_an_undetectable_build_declines_the_launcher_too(monkeypatch):
    import gmpas.remap as remap_mod

    monkeypatch.setattr(remap_mod, "_esmf_supports_mpi", lambda tool: None)
    monkeypatch.setattr(remap_mod.shutil, "which", lambda name: f"/opt/{name}")

    prefix, note = _mpi_launch_prefix(64, "/fake/esmf")
    assert prefix == []
    assert "could not tell" in note


def test_ensure_weights_wraps_the_esmf_call_with_the_launcher(tmp_path, monkeypatch):
    import gmpas.remap as remap_mod

    calls = []

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_which(name):
        return {"ESMF_RegridWeightGen": "/fake/esmf",
                "mpirun": "/fake/mpirun"}.get(name)

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        (tmp_path / "w" / "map_conserve.nc").write_bytes(b"weights")
        return Result()

    monkeypatch.setattr(remap_mod, "_esmf_supports_mpi", lambda tool: True)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(remap_mod.shutil, "which", fake_which)
    monkeypatch.setattr(remap_mod.subprocess, "run", fake_run)
    write_mesh(tmp_path / "m.nc", [(0.0, 0.0), (5.0, 0.0)])
    domain = TargetDomain(4, 8, -2.0, 2.0, 0.0, 8.0)

    (tmp_path / "w").mkdir()
    ensure_weights(tmp_path / "m.nc", domain, tmp_path / "w",
                   ranks=8, quiet=True)

    assert len(calls) == 1
    assert calls[0][:3] == ["/fake/mpirun", "-np", "8"]
    assert "/fake/esmf" in calls[0]


def test_ensure_weights_declines_the_launcher_on_a_mpiuni_build(tmp_path, monkeypatch):
    import gmpas.remap as remap_mod

    calls = []

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_which(name):
        return {"ESMF_RegridWeightGen": "/fake/esmf",
                "mpirun": "/fake/mpirun"}.get(name)

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        (tmp_path / "w" / "map_conserve.nc").write_bytes(b"weights")
        return Result()

    monkeypatch.setattr(remap_mod, "_esmf_supports_mpi", lambda tool: False)
    monkeypatch.setattr(remap_mod.shutil, "which", fake_which)
    monkeypatch.setattr(remap_mod.subprocess, "run", fake_run)
    write_mesh(tmp_path / "m.nc", [(0.0, 0.0), (5.0, 0.0)])
    domain = TargetDomain(4, 8, -2.0, 2.0, 0.0, 8.0)

    (tmp_path / "w").mkdir()
    ensure_weights(tmp_path / "m.nc", domain, tmp_path / "w",
                   ranks=8, quiet=True)

    assert calls[0] == ["/fake/esmf", "-s", "src.scrip.nc", "-d", "dst.scrip.nc",
                        "-w", "map_conserve.nc", "-m", "conserve",
                        "--src_regional", "--dst_regional", "--ignore_unmapped",
                        "--no_log"]


# ------------------------------------------------------- output construction


def test_each_vertical_dimension_keeps_its_own_name(tmp_path):
    """MPAS has several, of different lengths, and they are not the same axis.

    nVertLevels (55), nVertLevelsP1 (56) and nSoilLevels (4) appear in one
    file. Calling them all "lev" makes xarray try to align them and refuse.
    """
    path = tmp_path / "h.nc"
    write_mesh(path, [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)])
    with xr.open_dataset(path) as d:
        ds = d.load()
    ds["flat"] = (("Time", "nCells"), np.zeros((1, 3), "f4"))
    ds["tall"] = (("Time", "nCells", "nVertLevels"), np.zeros((1, 3, 5), "f4"))
    ds["taller"] = (("Time", "nCells", "nVertLevelsP1"), np.zeros((1, 3, 6), "f4"))
    ds["soil"] = (("Time", "nCells", "nSoilLevels"), np.zeros((1, 3, 2), "f4"))
    ds.to_netcdf(path, mode="w")

    weights = _weight_file(
        tmp_path / "map.nc",
        row=[1, 2, 3], col=[1, 2, 3], S=[1.0, 1.0, 1.0],
        area_a=[1.0, 1.0, 1.0], area_b=[1.0, 1.0, 1.0],
        frac_a=[1.0, 1.0, 1.0], frac_b=[1.0, 1.0, 1.0],
    )
    domain = TargetDomain(nlat=1, nlon=3, startlat=-1.0, endlat=1.0,
                          startlon=-1.0, endlon=5.0)

    info = remap_file(path, weights, domain,
                      ["flat", "tall", "taller", "soil"],
                      tmp_path / "out.nc")
    assert info["fields"] == 4

    with xr.open_dataset(tmp_path / "out.nc") as out:
        assert out.flat.dims == ("Time", "lat", "lon")
        assert out.tall.dims == ("Time", "nVertLevels", "lat", "lon")
        assert out.taller.dims == ("Time", "nVertLevelsP1", "lat", "lon")
        assert out.soil.dims == ("Time", "nSoilLevels", "lat", "lon")
        assert out.sizes["nVertLevels"] == 5
        assert out.sizes["nVertLevelsP1"] == 6
        assert out.sizes["nSoilLevels"] == 2


# --------------------------------------------------------------- valid_times


def _xtime_ds(*stamps):
    """A bare Dataset carrying only what `valid_times` looks at."""
    raw = np.array([s.ljust(64).encode() for s in stamps], dtype="S64")
    return xr.Dataset({"xtime": ("Time", raw)})


def test_valid_times_reads_xtime():
    ds = _xtime_ds("2012-02-25_12:00:00", "2012-02-25_13:00:00")
    got = valid_times(ds, "irrelevant.nc", n_time=2)
    expected = np.array(["2012-02-25T12:00:00", "2012-02-25T13:00:00"],
                        dtype="datetime64[ns]")
    assert (got == expected).all()


def test_valid_times_prefers_xtime_over_the_filename():
    """xtime is MPAS's own record and can legitimately disagree with the
    filename -- e.g. a restart run stamped with its start time."""
    ds = _xtime_ds("2012-02-25_12:00:00")
    got = valid_times(ds, "history.2012-02-20_00.00.00.nc", n_time=1)
    assert got == np.array(["2012-02-25T12:00:00"], dtype="datetime64[ns]")


def test_valid_times_falls_back_to_the_filename_without_xtime():
    ds = xr.Dataset({"t2m": ("nCells", [1.0, 2.0])})
    got = valid_times(ds, "history.2012-02-25_12.00.00.nc", n_time=1)
    assert got == np.array(["2012-02-25T12:00:00"], dtype="datetime64[ns]")


def test_valid_times_cannot_recover_several_steps_from_one_filename():
    """One timestamp in a filename cannot stand in for several real steps."""
    ds = xr.Dataset({"t2m": ("Time", [1.0, 2.0])})
    assert valid_times(ds, "history.2012-02-25_12.00.00.nc", n_time=2) is None


def test_valid_times_is_none_with_neither_source():
    ds = xr.Dataset({"t2m": ("nCells", [1.0, 2.0])})
    assert valid_times(ds, "h.nc", n_time=1) is None


def test_valid_times_falls_back_when_xtime_is_not_a_gregorian_date():
    """An idealised run on a 360-day or no-leap calendar can write an xtime
    date plain Gregorian arithmetic cannot represent (day 30 of February).
    Falling back to the filename, when it can help, beats raising."""
    ds = _xtime_ds("0001-02-30_00:00:00")
    got = valid_times(ds, "history.2012-02-25_12.00.00.nc", n_time=1)
    assert got == np.array(["2012-02-25T12:00:00"], dtype="datetime64[ns]")


def test_valid_times_is_none_when_nothing_can_be_parsed():
    ds = _xtime_ds("0001-02-30_00:00:00")
    assert valid_times(ds, "h.nc", n_time=1) is None


# -------------------------------------------------- remap_file: Time output


def _write_time_varying(tmp_path, xtime=None, calendar=None, name="h.nc"):
    """A mesh file with one time-varying field, optionally carrying xtime
    and a calendar attribute, saved under `name`."""
    path = tmp_path / name
    write_mesh(path, [(0.0, 0.0), (2.0, 0.0), (0.0, 2.0)])
    with xr.open_dataset(path) as d:
        ds = d.load()
    ds["t2m"] = (("Time", "nCells"), np.array([[1.0, 2.0, 3.0]], dtype="f4"))
    if xtime is not None:
        ds["xtime"] = ("Time", np.array([xtime.ljust(64).encode()], dtype="S64"))
    if calendar is not None:
        ds.attrs["config_calendar_type"] = calendar
    ds.to_netcdf(path, mode="w")
    return path


def _domain_and_weights(tmp_path):
    weights = _weight_file(
        tmp_path / "map.nc",
        row=[1, 2, 3], col=[1, 2, 3], S=[1.0, 1.0, 1.0],
        area_a=[1.0, 1.0, 1.0], area_b=[1.0, 1.0, 1.0],
        frac_a=[1.0, 1.0, 1.0], frac_b=[1.0, 1.0, 1.0],
    )
    domain = TargetDomain(nlat=1, nlon=3, startlat=-1.0, endlat=1.0,
                          startlon=-1.0, endlon=5.0)
    return domain, weights


def test_remap_file_attaches_a_cf_time_coordinate_from_xtime(tmp_path):
    path = _write_time_varying(tmp_path, xtime="2012-02-25_12:00:00")
    domain, weights = _domain_and_weights(tmp_path)

    info = remap_file(path, weights, domain, ["t2m"], tmp_path / "out.nc")
    assert info["time_coord"] is True

    with xr.open_dataset(tmp_path / "out.nc") as out:
        assert out.Time.values == np.array(["2012-02-25T12:00:00"], dtype="datetime64[ns]")
        assert out.xtime.values[0].decode().strip() == "2012-02-25_12:00:00"


def test_remap_file_carries_the_source_calendar(tmp_path):
    path = _write_time_varying(tmp_path, xtime="2012-02-25_12:00:00",
                               calendar="360_day")
    domain, weights = _domain_and_weights(tmp_path)

    remap_file(path, weights, domain, ["t2m"], tmp_path / "out.nc")

    with xr.open_dataset(tmp_path / "out.nc", decode_times=False) as out:
        assert out.Time.attrs["calendar"] == "360_day"


def test_remap_file_defaults_to_gregorian_without_a_source_calendar(tmp_path):
    path = _write_time_varying(tmp_path, xtime="2012-02-25_12:00:00")
    domain, weights = _domain_and_weights(tmp_path)

    remap_file(path, weights, domain, ["t2m"], tmp_path / "out.nc")

    with xr.open_dataset(tmp_path / "out.nc", decode_times=False) as out:
        assert out.Time.attrs["calendar"] == "gregorian"


def test_remap_file_falls_back_to_the_filename_without_xtime(tmp_path):
    path = _write_time_varying(tmp_path, name="history.2012-02-25_12.00.00.nc")
    domain, weights = _domain_and_weights(tmp_path)

    info = remap_file(path, weights, domain, ["t2m"], tmp_path / "out.nc")
    assert info["time_coord"] is True

    with xr.open_dataset(tmp_path / "out.nc") as out:
        assert out.Time.values == np.array(["2012-02-25T12:00:00"], dtype="datetime64[ns]")


def test_remap_file_writes_without_a_time_coordinate_when_nothing_is_known(tmp_path):
    """No xtime, no timestamp in the filename -- writes exactly as it always
    did, just without the coordinate it has no way to determine."""
    path = _write_time_varying(tmp_path, name="h.nc")
    domain, weights = _domain_and_weights(tmp_path)

    info = remap_file(path, weights, domain, ["t2m"], tmp_path / "out.nc")
    assert info["time_coord"] is False
    assert info["fields"] == 1

    with xr.open_dataset(tmp_path / "out.nc") as out:
        assert "Time" not in out.coords
        assert "Time" in out.t2m.dims          # the dimension is still real
        assert out.t2m.isel(Time=0).values.tolist() == [[1.0, 2.0, 3.0]]


# ------------------------------------------------------------ core detection


def test_the_scheduler_allocation_beats_the_machine_size(monkeypatch):
    """os.cpu_count() is the node, not the job. Asking for 4 cores and then
    spawning 256 workers is a good way to be thrown off a shared system."""
    from gmpas.remap import detect_cores

    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "8")
    assert detect_cores() == (8, "SLURM_CPUS_PER_TASK")


def test_slurm_per_node_notation_is_parsed(monkeypatch):
    """SLURM writes counts like 4(x2)."""
    from gmpas.remap import detect_cores

    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "4(x2)")
    assert detect_cores() == (4, "SLURM_CPUS_ON_NODE")


def test_other_schedulers_are_recognised(monkeypatch):
    from gmpas.remap import CORE_VARS, detect_cores

    for var in ("PBS_NCPUS", "LSB_DJOB_NUMPROC", "NSLOTS"):
        for other in CORE_VARS:
            monkeypatch.delenv(other, raising=False)
        monkeypatch.setenv(var, "12")
        assert detect_cores() == (12, var)


def test_nothing_is_hardcoded_when_no_scheduler(monkeypatch):
    """Falls through to what the OS will actually allow."""
    from gmpas.remap import CORE_VARS, detect_cores

    for var in CORE_VARS:
        monkeypatch.delenv(var, raising=False)
    n, source = detect_cores()
    assert n >= 1
    assert source in {"affinity mask", "machine cores"}


def test_a_nonsense_value_is_ignored(monkeypatch):
    from gmpas.remap import CORE_VARS, detect_cores

    for var in CORE_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "not-a-number")
    _, source = detect_cores()
    assert source != "SLURM_CPUS_PER_TASK"


# -------------------------------------------------------------- parallel run


def test_a_failing_file_does_not_stop_the_run(tmp_path, weights):
    """One bad file in two hundred must not lose the other 199."""
    from gmpas.remap import remap_many

    domain = TargetDomain(nlat=1, nlon=2, startlat=-1.0, endlat=1.0,
                          startlon=0.0, endlon=2.0)
    jobs = [(tmp_path / "absent.nc", tmp_path / "a.nc", domain, ["x"])]

    results = list(remap_many(jobs, weights, weights.path, workers=1))
    assert len(results) == 1
    assert "error" in results[0]
    assert results[0]["source"] == "absent.nc"
