"""Preparing JIGSAW's inputs, and running it.

JIGSAW itself is an external executable that most machines will not have, so
the tests that need it are skipped when it is absent — exactly as the remapping
tests treat ESMF. Everything up to the call is tested unconditionally, because
that is the part gmpas is actually responsible for.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

from gmpas.prep.generate import (
    GenerateError,
    find_jigsaw,
    generate,
    read_msh_counts,
    write_geom,
    write_hfun,
    write_jig,
)
from gmpas.prep.hfun import Hfun

COARSE = 200.0

GENTLE = """
import numpy as np

hfun_min = {h_min}
R = 6371.229

def get_hfun(lon, lat):
    lam, phi = np.radians(0.0), np.radians(0.0)
    c = np.array([np.cos(lam) * np.cos(phi),
                  np.sin(lam) * np.cos(phi), np.sin(phi)])
    p = np.column_stack([(np.cos(lon) * np.cos(lat)).ravel(),
                         (np.sin(lon) * np.cos(lat)).ravel(),
                         np.sin(lat).ravel()])
    r = R * np.arccos(np.clip(p @ c, -1.0, 1.0))
    return np.interp(r, [0, 2000, {t_end}],
                     [hfun_min, hfun_min, {h_max}]).reshape(np.shape(lon))
"""


def write_hfun_py(path, h_min=COARSE, h_max=400.0, t_end=20000.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(GENTLE.format(h_min=h_min, h_max=h_max, t_end=t_end))
    return path


jigsaw_available = pytest.mark.skipif(
    shutil.which("jigsaw") is None and not os.environ.get("JIGSAWDIR"),
    reason="jigsaw is not installed (set JIGSAWDIR or put it on PATH)",
)


# ------------------------------------------------------------- the executable


def test_a_missing_jigsaw_says_how_to_get_one(monkeypatch):
    monkeypatch.delenv("JIGSAWDIR", raising=False)
    monkeypatch.setattr(shutil, "which", lambda _n: None)

    with pytest.raises(GenerateError) as exc:
        find_jigsaw()
    msg = str(exc.value)
    assert "github.com/dengwirda/jigsaw" in msg      # where to get it
    assert "JIGSAWDIR" in msg                        # and how to point at it
    assert "--jigsaw" in msg


def test_jigsawdir_may_be_the_directory_or_the_binary(tmp_path, monkeypatch):
    """Both are things people reasonably put in a variable called *DIR."""
    exe = tmp_path / "jigsaw"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)

    monkeypatch.setenv("JIGSAWDIR", str(tmp_path))
    assert find_jigsaw() == exe.resolve()

    monkeypatch.setenv("JIGSAWDIR", str(exe))
    assert find_jigsaw() == exe.resolve()


def test_an_explicit_jigsaw_beats_the_environment(tmp_path, monkeypatch):
    good = tmp_path / "good" / "jigsaw"
    good.parent.mkdir()
    good.write_text("#!/bin/sh\n")
    good.chmod(0o755)
    monkeypatch.setenv("JIGSAWDIR", "/nowhere/at/all")

    assert find_jigsaw(good) == good.resolve()


def test_an_explicit_path_that_is_not_there_is_named(tmp_path):
    with pytest.raises(GenerateError, match="no jigsaw executable at"):
        find_jigsaw(tmp_path / "nope")


def test_a_build_directory_resolves_to_the_binary_inside_it(tmp_path):
    exe = tmp_path / "jigsaw"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    assert find_jigsaw(tmp_path) == exe.resolve()


def test_a_non_executable_file_is_refused(tmp_path):
    exe = tmp_path / "jigsaw"
    exe.write_text("not runnable")
    exe.chmod(0o644)
    with pytest.raises(GenerateError, match="not executable"):
        find_jigsaw(exe)


# ------------------------------------------------------------ JIGSAW's inputs


def test_geom_is_the_sphere_jigsaw_expects(tmp_path):
    text = write_geom(tmp_path / "GEOM.msh").read_text()
    assert text.startswith("MSHID=3;ellipsoid-mesh")
    assert "RADII=6371.229;6371.229;6371.229" in text


def test_the_jig_file_carries_the_tutorial_settings(tmp_path):
    text = write_jig(tmp_path / "MESH.jig", "GEOM.msh", "HFUN.msh",
                     "MESH.msh").read_text()
    for line in ("GEOM_FILE=GEOM.msh", "HFUN_FILE=HFUN.msh",
                 "MESH_FILE=MESH.msh", "HFUN_SCAL=absolute",
                 "HFUN_HMAX=inf", "HFUN_HMIN=0.0", "MESH_DIMS=2",
                 "OPTM_QLIM=0.9375"):
        assert line in text
    assert "INIT_FILE" not in text            # only when one is asked for


def test_an_init_file_is_added_only_when_given(tmp_path):
    text = write_jig(tmp_path / "MESH.jig", "GEOM.msh", "HFUN.msh", "MESH.msh",
                     init="ICOS.msh").read_text()
    assert "INIT_FILE=ICOS.msh" in text


def test_hfun_msh_matches_what_create_hfun_would_write(tmp_path):
    """The header, the grid and the value ordering, against the script itself.

    `create_hfun.py` builds its grid with `meshgrid(lats, lons)` and flattens
    the result, so the values run longitude-major. Getting that order wrong
    would transpose the whole distance function and JIGSAW would refine the
    wrong place, without failing.
    """
    hfun = Hfun.load(write_hfun_py(tmp_path / "hfun.py"))
    path, npts = write_hfun(hfun, tmp_path / "HFUN.msh", quiet=True)

    lines = path.read_text().splitlines()
    assert lines[0] == "MSHID=3;ellipsoid-grid"
    assert lines[1] == "NDIMS=2"

    nlon = int(lines[2].split(";")[1])
    nlat = int(lines[3 + nlon].split(";")[1])
    assert nlon == 2 * nlat                    # create_hfun.py's shape
    assert npts == nlon * nlat

    lons = np.array([float(v) for v in lines[3:3 + nlon]])
    lats = np.array([float(v) for v in lines[4 + nlon:4 + nlon + nlat]])
    assert lons[0] == pytest.approx(-np.pi)
    assert lons[-1] == pytest.approx(np.pi)
    assert lats[0] == pytest.approx(-0.5 * np.pi)
    assert lats[-1] == pytest.approx(0.5 * np.pi)

    head = lines[4 + nlon + nlat]
    assert head.startswith(f"VALUE={npts};")

    values = np.array([float(v) for v in lines[5 + nlon + nlat:]])
    assert values.size == npts

    # the same call the script makes, compared elementwise
    latgrid, longrid = np.meshgrid(lats, lons)
    assert values == pytest.approx(hfun.sample_radians(longrid, latgrid).ravel())


def test_msh_counts_are_read_without_the_body(tmp_path):
    p = tmp_path / "MESH.msh"
    p.write_text("# comment\nMSHID=3;EUCLIDEAN-MESH\nNDIMS=3\n"
                 "POINT=5\n1;2;3;0\n" * 1 + "TRIA3=6\n1;2;3;0\n")
    assert read_msh_counts(p) == (5, 6)


# ----------------------------------------------------------------- the guard


def test_a_steep_transition_stops_the_run_before_jigsaw(tmp_path):
    """Generation costs minutes; the check costs a second."""
    steep = write_hfun_py(tmp_path / "steep" / "hfun.py",
                          h_max=400.0, t_end=2400.0)
    with pytest.raises(GenerateError) as exc:
        generate(steep, out_dir=tmp_path / "out", jigsaw="/bin/echo")

    msg = str(exc.value)
    assert "gradient" in msg
    assert "--allow-steep" in msg
    assert not (tmp_path / "out" / "HFUN.msh").exists()   # nothing was spent


def test_the_executable_is_found_before_the_function_is_read(tmp_path, monkeypatch):
    """A missing jigsaw must not be reported only after a slow sampling pass.

    The hfun file here is invalid too, so whichever check runs first decides
    the error -- which is exactly what this pins down.
    """
    monkeypatch.delenv("JIGSAWDIR", raising=False)
    monkeypatch.setattr(shutil, "which", lambda _n: None)

    monkey = tmp_path / "hfun.py"
    monkey.write_text("hfun_min = 12.0\n")
    with pytest.raises(GenerateError, match="dengwirda"):
        generate(monkey, out_dir=tmp_path / "out", jigsaw=None)


# ------------------------------------------------------------------ real runs


@jigsaw_available
def test_jigsaw_produces_a_sphere_triangulation(tmp_path):
    """A closed triangulation of a sphere has exactly 2V - 4 triangles.

    That is Euler's formula, and it holds for any valid result whatever the
    resolution, so it checks the run really produced a mesh rather than a file.
    """
    hfun = write_hfun_py(tmp_path / "hfun.py", h_min=400.0, h_max=800.0)
    result = generate(hfun, out_dir=tmp_path / "out", quiet=True)

    assert result.points > 100
    assert result.triangles == 2 * result.points - 4
    assert result.mesh_msh.exists()
    assert not result.reused


@jigsaw_available
def test_a_second_run_reuses_the_mesh_unless_forced(tmp_path):
    hfun = write_hfun_py(tmp_path / "hfun.py", h_min=400.0, h_max=800.0)
    out = tmp_path / "out"

    first = generate(hfun, out_dir=out, quiet=True)
    again = generate(hfun, out_dir=out, quiet=True)
    assert again.reused
    assert again.points == first.points
    assert again.seconds == 0.0

    forced = generate(hfun, out_dir=out, quiet=True, force=True)
    assert not forced.reused


def test_a_failing_jigsaw_is_reported_in_its_own_words(tmp_path):
    """A stub, so this runs everywhere: what matters is that a non-zero exit
    becomes a GenerateError carrying the tool's last lines, not a silent
    success or a bare CalledProcessError."""
    stub = tmp_path / "jigsaw"
    stub.write_text("#!/bin/sh\necho '**parse error: no such file'\nexit 3\n")
    stub.chmod(0o755)

    hfun = write_hfun_py(tmp_path / "hfun.py", h_min=400.0, h_max=800.0)
    with pytest.raises(GenerateError) as exc:
        generate(hfun, out_dir=tmp_path / "out", jigsaw=stub, quiet=True)

    msg = str(exc.value)
    assert "exit 3" in msg
    assert "parse error" in msg               # jigsaw's own words, not ours


def test_a_jigsaw_that_exits_cleanly_without_a_mesh_still_fails(tmp_path):
    """Exit 0 is not proof: the file has to be there."""
    stub = tmp_path / "jigsaw"
    stub.write_text("#!/bin/sh\nexit 0\n")
    stub.chmod(0o755)

    hfun = write_hfun_py(tmp_path / "hfun.py", h_min=400.0, h_max=800.0)
    with pytest.raises(GenerateError, match="exit 0"):
        generate(hfun, out_dir=tmp_path / "out", jigsaw=stub, quiet=True)
