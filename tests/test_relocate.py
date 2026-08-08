"""Moving a mesh's refined region to a new tangent point, without resizing."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from conftest import write_mesh
from gmpas.cli import main
from gmpas.prep.relocate import relocate_mesh

#: variables that must come out byte-for-byte identical to the source --
#: everything a rotation doesn't touch (see relocate.py's module docstring)
UNCHANGED_VARS = (
    "nEdgesOnCell", "verticesOnCell", "edgesOnCell", "cellsOnEdge",
    "cellsOnVertex", "edgesOnVertex", "dcEdge", "dvEdge", "areaCell",
    "areaTriangle", "kiteAreasOnVertex", "weightsOnEdge", "nominalMinDc",
)


def test_the_target_point_lands_exactly_on_the_tangent_point(tmp_path):
    path = write_mesh(tmp_path / "m.nc", [(100.0, 0.0), (110.0, 5.0), (120.0, -5.0)],
                      with_scale_vars=True, sphere_radius=1.0)
    out = relocate_mesh(path, tmp_path / "relocated.nc", 30.0, 200.0,
                        from_lat_deg=0.0, from_lon_deg=100.0)

    with xr.open_dataset(out) as ds:
        assert np.degrees(ds.latCell.values[0]) == pytest.approx(30.0)
        # 200E and -160E are the same point
        assert np.cos(np.radians(200.0) - ds.lonCell.values[0]) == pytest.approx(1.0)


def test_auto_detects_the_finest_cell_as_the_from_point(tmp_path):
    path = write_mesh(tmp_path / "m.nc", [(100.0, 0.0), (110.0, 5.0), (120.0, -5.0)],
                      areas=[5.0e10, 1.0e9, 8.0e10],   # cell 1 is finest
                      with_scale_vars=True, sphere_radius=1.0)
    out = relocate_mesh(path, tmp_path / "relocated.nc", 0.0, 0.0)

    with xr.open_dataset(out) as ds:
        assert np.degrees(ds.latCell.values[1]) == pytest.approx(0.0, abs=1e-8)
        assert np.cos(ds.lonCell.values[1]) == pytest.approx(1.0)


def test_every_area_distance_and_weight_is_byte_for_byte_unchanged(tmp_path):
    """The whole point: a rotation is an isometry, so nothing metric moves --
    only position and angleEdge (bearing relative to the fixed lat/lon grid,
    which does change) are touched."""
    path = write_mesh(tmp_path / "m.nc", [(100.0, 0.0), (110.0, 5.0), (120.0, -5.0)],
                      with_scale_vars=True, sphere_radius=1.0)
    out = relocate_mesh(path, tmp_path / "relocated.nc", 30.0, 200.0)

    with xr.open_dataset(path) as src, xr.open_dataset(out) as dst:
        for name in UNCHANGED_VARS:
            assert np.array_equal(src[name].values, dst[name].values), name


def test_works_at_any_sphere_radius_not_just_unit(tmp_path):
    """Unlike scale_mesh, a rotation doesn't care about sphere_radius --
    it's a linear transform of whatever coordinates are already there."""
    path = write_mesh(tmp_path / "m.nc", [(100.0, 0.0), (110.0, 5.0)],
                      with_scale_vars=True, sphere_radius=6_371_229.0)
    out = relocate_mesh(path, tmp_path / "relocated.nc", 30.0, 200.0,
                        from_lat_deg=0.0, from_lon_deg=100.0)
    assert out.exists()


def test_works_with_a_ragged_ie_differently_shaped_mesh(tmp_path):
    """Every cell has a different vertex count -- relocate never touches
    connectivity or any ragged/maxEdges-shaped array, so this should be a
    non-event, unlike scale_mesh's areaCell recompute."""
    path = write_mesh(tmp_path / "m.nc", [(100.0, 0.0), (110.0, 5.0), (120.0, -5.0)],
                      n_verts=[5, 4, 6], with_scale_vars=True, sphere_radius=1.0)
    out = relocate_mesh(path, tmp_path / "relocated.nc", 30.0, 200.0,
                        from_lat_deg=0.0, from_lon_deg=100.0)

    with xr.open_dataset(path) as src, xr.open_dataset(out) as dst:
        assert np.array_equal(src.nEdgesOnCell.values, dst.nEdgesOnCell.values)
        assert np.array_equal(src.verticesOnCell.values, dst.verticesOnCell.values)
        assert np.array_equal(src.areaCell.values, dst.areaCell.values)


def test_a_missing_variable_names_what_is_missing(tmp_path):
    path = write_mesh(tmp_path / "m.nc", [(100.0, 0.0)])   # with_scale_vars defaults False
    with pytest.raises(KeyError, match="xVertex"):
        relocate_mesh(path, tmp_path / "relocated.nc", 0.0, 0.0)


def test_a_missing_mesh_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="No such mesh file"):
        relocate_mesh(tmp_path / "absent.nc", tmp_path / "relocated.nc", 0.0, 0.0)


# -------------------------------------------------------------------- CLI

def test_cli_prep_relocate(tmp_path, capsys):
    path = write_mesh(tmp_path / "m.nc", [(100.0, 0.0), (110.0, 5.0)],
                      with_scale_vars=True, sphere_radius=1.0)
    out = tmp_path / "relocated.nc"

    assert main(["prep", "relocate", str(path), "-o", str(out),
                "--tan-lat", "30", "--tan-lon", "200"]) == 0
    assert out.exists()
    assert "->" in capsys.readouterr().out


def test_cli_default_output_path_is_derived_from_the_input(tmp_path, monkeypatch):
    path = write_mesh(tmp_path / "m.nc", [(100.0, 0.0), (110.0, 5.0)],
                      with_scale_vars=True, sphere_radius=1.0)
    monkeypatch.chdir(tmp_path)

    assert main(["prep", "relocate", str(path), "--tan-lat", "30",
                "--tan-lon", "200"]) == 0
    assert (tmp_path / "m.relocated.nc").exists()


def test_cli_from_lat_without_from_lon_is_rejected(tmp_path, capsys):
    path = write_mesh(tmp_path / "m.nc", [(100.0, 0.0), (110.0, 5.0)],
                      with_scale_vars=True, sphere_radius=1.0)

    assert main(["prep", "relocate", str(path), "--tan-lat", "30",
                "--tan-lon", "200", "--from-lat", "0"]) == 1
    assert "--from-lat and --from-lon" in capsys.readouterr().err
