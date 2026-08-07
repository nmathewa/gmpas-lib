"""Rescaling a regional mesh around a stereographic tangent point."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from conftest import write_mesh
from gmpas.cli import main
from gmpas.prep.scale import (
    great_circle_distance,
    lonlat_to_xyz,
    scale_mesh,
    spherical_angle,
    spherical_triangle_area,
    stereo_inverse,
    stereo_project,
    xyz_to_lonlat,
)

# ------------------------------------------------------------- primitives

def test_lonlat_xyz_round_trip():
    lon = np.array([0.3, -1.2, 2.5, 0.0])
    lat = np.array([0.1, -0.4, 0.6, 0.0])
    lon2, lat2 = xyz_to_lonlat(lonlat_to_xyz(lon, lat))
    assert np.allclose(lon, lon2)
    assert np.allclose(lat, lat2)


def test_stereo_round_trip_at_scale_one():
    lon = np.array([0.3, -1.2, 2.5])
    lat = np.array([0.1, -0.4, 0.6])
    x, y = stereo_project(lon, lat, 0.5, 0.2)
    lon2, lat2 = stereo_inverse(x, y, 0.5, 0.2)
    assert np.allclose(lon, lon2)
    assert np.allclose(lat, lat2)


def test_stereo_inverse_at_the_tangent_point_itself():
    """rho=0 is a 0/0 in the raw formula -- must not come back nan."""
    lam, phi = stereo_inverse(np.array(0.0), np.array(0.0), 0.5, 0.2)
    assert lam == pytest.approx(0.5)
    assert phi == pytest.approx(0.2)


def test_octant_triangle_area_is_pi_over_2():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    c = np.array([0.0, 0.0, 1.0])
    assert spherical_triangle_area(a, b, c) == pytest.approx(np.pi / 2)


def test_great_circle_distance_quarter_sphere_apart():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert great_circle_distance(a, b) == pytest.approx(np.pi / 2)


def test_octant_angle_is_a_right_angle():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    c = np.array([0.0, 0.0, 1.0])
    assert abs(spherical_angle(a, b, c)) == pytest.approx(np.pi / 2)


def test_primitives_broadcast_over_a_whole_mesh_at_once():
    """scale_mesh calls these across every cell/edge/vertex in one shot --
    confirm they broadcast, not just work on single points."""
    a = np.tile([1.0, 0.0, 0.0], (5, 1))
    b = np.tile([0.0, 1.0, 0.0], (5, 1))
    c = np.tile([0.0, 0.0, 1.0], (5, 1))
    area = spherical_triangle_area(a, b, c)
    assert area.shape == (5,)
    assert np.allclose(area, np.pi / 2)


# --------------------------------------------------------------- areaCell

def test_area_cell_matches_a_brute_force_reference(tmp_path):
    """The ragged, vectorized areaCell sum against a plain Python loop over
    each cell's own vertex ring -- the one place a masking/indexing bug in
    the vectorization would most easily hide."""
    path = write_mesh(tmp_path / "m.nc",
                      [(100.0, 0.0), (110.0, 5.0), (120.0, -5.0)],
                      with_scale_vars=True, sphere_radius=1.0)
    # scale_factor=1 is a geometric identity (see test_scale_factor_one_is_
    # an_identity below), so the source file's own vertex positions are
    # exactly what areaCell gets recomputed from.
    out = scale_mesh(path, tmp_path / "scaled.nc", 1.0, 0.0, 100.0)

    with xr.open_dataset(path) as src, xr.open_dataset(out) as scaled:
        voc = src.verticesOnCell.values.astype(np.int64) - 1
        nedges = src.nEdgesOnCell.values.astype(np.int64)
        vtx_xyz = lonlat_to_xyz(src.lonVertex.values, src.latVertex.values)
        cell_xyz = lonlat_to_xyz(src.lonCell.values, src.latCell.values)

        reference = np.zeros(cell_xyz.shape[0])
        for i in range(cell_xyz.shape[0]):
            total = 0.0
            for j in range(nedges[i]):
                v1 = voc[i, j]
                v2 = voc[i, (j + 1) % nedges[i]]
                total += spherical_triangle_area(cell_xyz[i], vtx_xyz[v1], vtx_xyz[v2])
            reference[i] = total

        assert np.allclose(scaled.areaCell.values, reference, rtol=1e-6)


# -------------------------------------------------------------- scale_mesh

def test_scale_factor_one_is_an_identity(tmp_path):
    """Project, divide by 1, re-project -- coordinates shouldn't move."""
    path = write_mesh(tmp_path / "m.nc", [(100.0, 0.0), (110.0, 5.0)],
                      with_scale_vars=True, sphere_radius=1.0)
    out = scale_mesh(path, tmp_path / "scaled.nc", 1.0, 0.0, 105.0)

    with xr.open_dataset(path) as src, xr.open_dataset(out) as scaled:
        assert np.allclose(src.lonCell.values, scaled.lonCell.values, atol=1e-9)
        assert np.allclose(src.latCell.values, scaled.latCell.values, atol=1e-9)
        assert np.allclose(src.xCell.values, scaled.xCell.values, atol=1e-9)


def test_scale_factor_two_halves_nominal_min_dc(tmp_path):
    path = write_mesh(tmp_path / "m.nc", [(100.0, 0.0), (110.0, 5.0)],
                      with_scale_vars=True, sphere_radius=1.0)
    out = scale_mesh(path, tmp_path / "scaled.nc", 2.0, 0.0, 105.0)

    with xr.open_dataset(path) as src, xr.open_dataset(out) as scaled:
        assert float(scaled.nominalMinDc.values) == pytest.approx(
            float(src.nominalMinDc.values) / 2.0)


def test_boundary_quantities_take_the_scalefac_fallback(tmp_path):
    """Every edge/vertex in this fixture is a boundary one -- cells don't
    share topology -- so every quantity with a "no real neighbour" branch
    should take it rather than the geometric recompute."""
    path = write_mesh(tmp_path / "m.nc", [(100.0, 0.0), (110.0, 5.0)],
                      with_scale_vars=True, sphere_radius=1.0)
    out = scale_mesh(path, tmp_path / "scaled.nc", 2.0, 0.0, 105.0)

    with xr.open_dataset(path) as src, xr.open_dataset(out) as scaled:
        assert np.allclose(scaled.dcEdge.values, src.dcEdge.values / 2.0)
        assert np.allclose(scaled.areaTriangle.values, src.areaTriangle.values / 4.0)
        assert np.allclose(scaled.kiteAreasOnVertex.values,
                           src.kiteAreasOnVertex.values / 4.0)
        # no cell pair is ever valid here, so the weightsOnEdge loop never
        # touches a single entry
        assert np.allclose(scaled.weightsOnEdge.values, src.weightsOnEdge.values)


def test_output_carries_every_original_variable(tmp_path):
    path = write_mesh(tmp_path / "m.nc", [(100.0, 0.0), (110.0, 5.0)],
                      with_scale_vars=True, sphere_radius=1.0)
    out = scale_mesh(path, tmp_path / "scaled.nc", 2.0, 0.0, 105.0)

    with xr.open_dataset(path) as src, xr.open_dataset(out) as scaled:
        assert set(src.variables) <= set(scaled.variables)


def test_a_non_unit_sphere_is_refused(tmp_path):
    path = write_mesh(tmp_path / "m.nc", [(100.0, 0.0), (110.0, 5.0)],
                      with_scale_vars=True, sphere_radius=6_371_229.0)
    with pytest.raises(ValueError, match="sphere_radius"):
        scale_mesh(path, tmp_path / "scaled.nc", 2.0, 0.0, 105.0)


def test_a_missing_variable_names_what_is_missing(tmp_path):
    path = write_mesh(tmp_path / "m.nc", [(100.0, 0.0)])   # with_scale_vars defaults False
    with pytest.raises(KeyError, match="dcEdge"):
        scale_mesh(path, tmp_path / "scaled.nc", 2.0, 0.0, 100.0)


def test_a_missing_mesh_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="No such mesh file"):
        scale_mesh(tmp_path / "absent.nc", tmp_path / "scaled.nc", 2.0, 0.0, 100.0)


# -------------------------------------------------------------------- CLI

def test_cli_prep_scale(tmp_path, capsys):
    path = write_mesh(tmp_path / "m.nc", [(100.0, 0.0), (110.0, 5.0)],
                      with_scale_vars=True, sphere_radius=1.0)
    out = tmp_path / "scaled.nc"

    assert main(["prep", "scale", str(path), "-o", str(out),
                "--scale-factor", "2.0", "--tan-lat", "0", "--tan-lon", "105"]) == 0
    assert out.exists()
    assert "->" in capsys.readouterr().out


def test_cli_default_output_path_is_derived_from_the_input(tmp_path, monkeypatch):
    path = write_mesh(tmp_path / "m.nc", [(100.0, 0.0), (110.0, 5.0)],
                      with_scale_vars=True, sphere_radius=1.0)
    monkeypatch.chdir(tmp_path)

    assert main(["prep", "scale", str(path), "--scale-factor", "2.0",
                "--tan-lat", "0", "--tan-lon", "105"]) == 0
    assert (tmp_path / "m.scaled.nc").exists()
