"""Reading a JIGSAW distance function, and what can be said about it.

The synthetic `hfun.py` files here follow the mini-tutorial's contract exactly,
including the awkward part of it: the tutorial's own `get_hfun` flattens its
inputs and hands back a 1-d array whatever shape it was given.
"""

from __future__ import annotations

import numpy as np
import pytest

from gmpas.prep.hfun import (
    DEG_TO_KM,
    GRADIENT_GUIDELINE,
    Hfun,
    HfunError,
    analysis_grid,
    diagnose,
)

#: coarse enough that the analysis grid is small and the tests are quick
COARSE = 200.0


def write_hfun(path, body: str, hfun_min: float = COARSE):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"import numpy as np\n\nhfun_min = {hfun_min}\n\n{body}\n")
    return path


UNIFORM = """
def get_hfun(lon, lat):
    return np.full(lon.size, hfun_min)
"""


def radial(h_max: float, t_begin: float, t_end: float,
           lon_c: float = 0.0, lat_c: float = 0.0) -> str:
    """The tutorial's own shape: flat, linear transition, flat."""
    return f"""
def get_hfun(lon, lat):
    r_earth = 6371.229
    lam, phi = np.radians({lon_c}), np.radians({lat_c})
    c = np.array([np.cos(lam) * np.cos(phi),
                  np.sin(lam) * np.cos(phi),
                  np.sin(phi)])
    p = np.column_stack([(np.cos(lon) * np.cos(lat)).flatten(),
                         (np.sin(lon) * np.cos(lat)).flatten(),
                         np.sin(lat).flatten()])
    r = r_earth * np.arccos(np.clip(p @ c, -1.0, 1.0))
    ret = np.full(r.shape, hfun_min)
    mid = np.logical_and(r >= {t_begin}, r < {t_end})
    ret[mid] = hfun_min + (r[mid] - {t_begin}) * ({h_max} - hfun_min) \\
                          / ({t_end} - {t_begin})
    ret[r >= {t_end}] = {h_max}
    return ret
"""


# ---------------------------------------------------------------- the contract


def test_a_file_defining_the_contract_loads(tmp_path):
    h = Hfun.load(write_hfun(tmp_path / "hfun.py", UNIFORM))
    assert h.hfun_min == COARSE
    assert h.sample_degrees(np.array([0.0, 90.0]),
                            np.array([0.0, 10.0])) == pytest.approx(COARSE)


def test_a_directory_is_read_as_its_hfun_py(tmp_path):
    write_hfun(tmp_path / "hfun.py", UNIFORM)
    assert Hfun.load(tmp_path).hfun_min == COARSE


def test_a_missing_get_hfun_says_what_the_contract_is(tmp_path):
    path = tmp_path / "hfun.py"
    path.write_text("hfun_min = 12.0\n")
    with pytest.raises(HfunError, match="get_hfun"):
        Hfun.load(path)


def test_a_missing_hfun_min_says_what_it_is_for(tmp_path):
    path = tmp_path / "hfun.py"
    path.write_text("def get_hfun(lon, lat):\n    return lon\n")
    with pytest.raises(HfunError, match="hfun_min"):
        Hfun.load(path)


def test_a_negative_hfun_min_is_refused(tmp_path):
    with pytest.raises(HfunError, match="positive"):
        Hfun.load(write_hfun(tmp_path / "hfun.py", UNIFORM, hfun_min=-5.0))


def test_a_get_hfun_that_raises_is_reported_against_its_file(tmp_path):
    body = "def get_hfun(lon, lat):\n    raise RuntimeError('no dataset')\n"
    with pytest.raises(HfunError, match="RuntimeError: no dataset"):
        Hfun.load(write_hfun(tmp_path / "hfun.py", body))


def test_a_get_hfun_returning_zero_is_refused(tmp_path):
    body = "def get_hfun(lon, lat):\n    return np.zeros(lon.size)\n"
    with pytest.raises(HfunError, match="<= 0"):
        Hfun.load(write_hfun(tmp_path / "hfun.py", body))


def test_a_flattened_return_is_reshaped_to_the_input(tmp_path):
    """The tutorial's get_hfun returns 1-d for a 2-d meshgrid; ours must not."""
    h = Hfun.load(write_hfun(tmp_path / "hfun.py", UNIFORM))
    lon, lat = np.meshgrid(np.linspace(-180, 180, 7), np.linspace(-80, 80, 5))
    assert h.sample_degrees(lon, lat).shape == (5, 7)


def test_a_wrong_length_return_is_caught(tmp_path):
    body = "def get_hfun(lon, lat):\n    return np.full(3, hfun_min)\n"
    h_path = write_hfun(tmp_path / "hfun.py", body)
    with pytest.raises(HfunError, match="3 values for 4 points"):
        Hfun.load(h_path)


def test_two_hfun_files_do_not_shadow_each_other(tmp_path):
    """Loading a second one in the same process must not reuse the first."""
    a = write_hfun(tmp_path / "a" / "hfun.py", UNIFORM, hfun_min=10.0)
    b = write_hfun(tmp_path / "b" / "hfun.py", UNIFORM, hfun_min=99.0)

    assert Hfun.load(a).hfun_min == 10.0
    assert Hfun.load(b).hfun_min == 99.0
    assert Hfun.load(a).hfun_min == 10.0


# ----------------------------------------------------------------- the grid


def test_the_analysis_grid_is_the_one_create_hfun_would_write(tmp_path):
    """Not an approximation of it -- the same formula, the same shape."""
    for hfun_min in (60.0, 120.0, 200.0):
        lons, lats = analysis_grid(hfun_min)
        assert lats.size == int(180.0 * DEG_TO_KM / hfun_min) + 1
        assert lons.size == 2 * lats.size
        assert lats[0] == pytest.approx(-0.5 * np.pi)
        assert lats[-1] == pytest.approx(0.5 * np.pi)
        assert lons[0] == pytest.approx(-np.pi)
        assert lons[-1] == pytest.approx(np.pi)


def test_a_very_fine_hfun_min_coarsens_rather_than_exhausting_memory():
    lons, lats = analysis_grid(0.5)              # would be ~3.2e9 points
    assert lons.size * lats.size <= 20_000_000


# ---------------------------------------------------------------- gradients


def test_a_uniform_function_has_no_gradient(tmp_path):
    d = diagnose(Hfun.load(write_hfun(tmp_path / "hfun.py", UNIFORM)))
    assert d.h_min == pytest.approx(COARSE)
    assert d.h_max == pytest.approx(COARSE)
    assert d.max_gradient == pytest.approx(0.0, abs=1e-12)
    assert d.within_guideline


def test_the_gradient_is_the_slope_of_the_transition(tmp_path):
    """The one number this is all for, against a case with a known answer.

    A linear transition from h_min to h_max over a radial distance w has slope
    exactly (h_max - h_min) / w everywhere inside it, so the measured maximum
    has an analytic value to be checked against.
    """
    h_max, t_begin, width = 800.0, 2000.0, 4000.0
    path = write_hfun(tmp_path / "hfun.py",
                      radial(h_max, t_begin, t_begin + width))
    d = diagnose(Hfun.load(path))

    expected = (h_max - COARSE) / width
    assert d.max_gradient == pytest.approx(expected, rel=0.02)
    assert d.h_min == pytest.approx(COARSE)
    assert d.h_max == pytest.approx(h_max)


def test_a_transition_that_is_too_steep_fails_the_guideline(tmp_path):
    steep = write_hfun(tmp_path / "steep" / "hfun.py",
                       radial(800.0, 2000.0, 2400.0))
    gentle = write_hfun(tmp_path / "gentle" / "hfun.py",
                        radial(800.0, 2000.0, 42000.0))

    assert diagnose(Hfun.load(steep)).max_gradient > GRADIENT_GUIDELINE
    assert not diagnose(Hfun.load(steep)).within_guideline
    assert diagnose(Hfun.load(gentle)).within_guideline


def test_the_steepest_point_is_inside_the_transition_ring(tmp_path):
    t_begin, t_end = 2000.0, 6000.0
    path = write_hfun(tmp_path / "hfun.py",
                      radial(800.0, t_begin, t_end, lon_c=0.0, lat_c=0.0))
    d = diagnose(Hfun.load(path))

    # great-circle distance from the refinement centre at (0, 0)
    r = 6371.229 * np.arccos(np.cos(np.radians(d.at_lat))
                             * np.cos(np.radians(d.at_lon)))
    assert t_begin <= r <= t_end


def test_the_poles_do_not_produce_an_infinite_gradient(tmp_path):
    """cos(lat) is zero there, so the zonal derivative is 0/0, not a gradient."""
    path = write_hfun(tmp_path / "hfun.py",
                      radial(800.0, 2000.0, 6000.0, lat_c=90.0))
    d = diagnose(Hfun.load(path))
    assert np.isfinite(d.max_gradient)
    assert d.max_gradient < 1.0


# ------------------------------------------------------------------- viewer


@pytest.fixture
def viewer(tmp_path):
    from gmpas.prep.hfunview import HfunViewer

    path = write_hfun(tmp_path / "hfun.py", radial(800.0, 2000.0, 6000.0))
    return HfunViewer(path, nx=60, ny=40)


def test_meta_is_json_serialisable_and_has_no_cell_counts(viewer):
    import json

    meta = json.loads(json.dumps(viewer.describe()))
    assert "cells" not in meta and "edges" not in meta
    assert meta["facts_label"] == "distance function"
    assert [name for name, _ in (r for r in meta["stats"])][0] == "hfun_min"


def test_mesh_density_is_one_where_the_mesh_is_finest(viewer):
    """rho(x_fine) = 1.0, as MPAS defines meshDensity."""
    vmin, vmax = viewer.limits("mesh_density")
    assert vmax == pytest.approx(1.0)
    assert vmin == pytest.approx((COARSE / 800.0) ** 4)


def test_the_scale_is_the_whole_sphere_not_the_view(viewer):
    """Panning must not recolour a transition band."""
    lo, hi = viewer.limits("cell_width_km")
    _, zoom_lo, zoom_hi = viewer.frame("cell_width_km", [-1.0, 1.0, -1.0, 1.0])
    assert (zoom_lo, zoom_hi) == (lo, hi)
    assert hi == pytest.approx(800.0)


def test_get_hfun_is_called_once_per_view_not_once_per_pixel(viewer):
    """The contract allows expensive setup, so this is a correctness matter."""
    calls = []
    inner = viewer.hfun._get_hfun
    viewer.hfun._get_hfun = lambda lon, lat: (calls.append(lon.size),
                                              inner(lon, lat))[1]

    viewer.frame("cell_width_km", [-180.0, 180.0, -90.0, 90.0])
    viewer.frame("mesh_density", [-180.0, 180.0, -90.0, 90.0])
    assert calls == [60 * 40]                 # both fields, one sampling


def test_an_unknown_field_says_what_it_offers(viewer):
    with pytest.raises(KeyError, match="cell_width_km"):
        viewer.limits("nope")


def test_prep_hfun_check_prints_the_report(tmp_path, capsys):
    from gmpas.cli import main

    path = write_hfun(tmp_path / "hfun.py", radial(800.0, 2000.0, 2400.0))
    assert main(["prep", "hfun", str(path), "--check"]) == 0

    out = capsys.readouterr().out
    assert "max cell size gradient" in out
    assert "ABOVE" in out                      # 600 km over 400 km is steep
    assert "create_hfun.py would write" in out


def test_a_bad_hfun_file_is_reported_not_traced(tmp_path, capsys):
    from gmpas.cli import main

    path = tmp_path / "hfun.py"
    path.write_text("hfun_min = 12.0\n")
    assert main(["prep", "hfun", str(path), "--check"]) == 1
    assert "get_hfun" in capsys.readouterr().err


def test_nothing_is_drawn_past_a_pole(viewer):
    """get_hfun answers for |lat| > 90 -- sin and cos are periodic, so it folds
    back and mirrors the other hemisphere. There is no sphere there and no
    coastline to sit under it, so the frame must be blank instead."""
    values = viewer.values("cell_width_km", [-180.0, 180.0, -126.0, 126.0],
                           40, 60)
    lat = np.linspace(-126.0, 126.0, 60, endpoint=False) + 126.0 / 60

    past = np.abs(lat) > 90.0
    assert past.any()                              # the test is testing something
    assert np.isnan(values[past]).all()
    assert np.isfinite(values[~past]).all()
