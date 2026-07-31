"""Pairing output with a mesh, and reducing a field to one value per element."""

from __future__ import annotations

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
