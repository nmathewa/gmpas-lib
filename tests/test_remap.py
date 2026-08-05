"""Applying weights, and the one command that runs the whole conversion."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from conftest import write_mesh
from gmpas.config import TargetDomain
from gmpas.remap import RemapError, Weights, ensure_weights, level_dim, remappable


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


# ------------------------------------------------------- output construction


def test_each_vertical_dimension_keeps_its_own_name(tmp_path):
    """MPAS has several, of different lengths, and they are not the same axis.

    nVertLevels (55), nVertLevelsP1 (56) and nSoilLevels (4) appear in one
    file. Calling them all "lev" makes xarray try to align them and refuse.
    """
    from conftest import write_mesh
    from gmpas.remap import remap_file

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
