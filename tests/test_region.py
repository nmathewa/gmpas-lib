"""Cropping a global mesh to a boundary-defined regional subset."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from conftest import write_grid_mesh
from gmpas.cli import main
from gmpas.prep.region import (
    N_BDY_LAYERS,
    _flood_fill,
    _relaxation_zones,
    _walk_boundary,
    create_region,
    write_graph_info,
)

#: a 20x20 patch, 2 degrees per cell, is comfortably bigger than N_BDY_LAYERS
#: (7) in every direction from a centred boundary, so a region drawn well
#: inside it exercises every relaxation ring without hitting the patch's own
#: (non-global, this is a regional test fixture) edge.
GRID = dict(nrows=20, ncols=20, lon0_deg=0.0, lat0_deg=0.0,
           dlon_deg=2.0, dlat_deg=2.0, sphere_radius=1.0)

#: a 10x10-cell box roughly in the middle of the 20x20 patch
BOX_LAT = [10.0, 10.0, 28.0, 28.0]
BOX_LON = [10.0, 28.0, 28.0, 10.0]


# ------------------------------------------------------------- graph algorithms

def test_flood_fill_stops_at_the_boundary_wall(tmp_path):
    path = write_grid_mesh(tmp_path / "m.nc", **GRID)
    with xr.open_dataset(path, decode_timedelta=False) as ds:
        coc = ds.cellsOnCell.values.astype(np.int64)
        nec = ds.nEdgesOnCell.values.astype(np.int64)
        cell_xyz = np.stack([ds.xCell.values, ds.yCell.values, ds.zCell.values], axis=-1)
        cell_xyz = cell_xyz / np.linalg.norm(cell_xyz, axis=-1, keepdims=True)

    # a small square loop, cell ids for grid corners (5,5) (5,8) (8,8) (8,5)
    # in a 20-wide row-major lattice (cell_id = row*20 + col)
    loop = np.array([5 * 20 + 5, 5 * 20 + 8, 8 * 20 + 8, 8 * 20 + 5])
    on_boundary = _walk_boundary(coc, nec, cell_xyz, loop)
    in_region = _flood_fill(coc, nec, 0, on_boundary)   # cell 0 is far outside the loop
    # the fill from outside the wall reaches every cell except the 2x2
    # pocket the wall encloses -- it cannot leak through the wall to get there
    enclosed = {6 * 20 + 6, 6 * 20 + 7, 7 * 20 + 6, 7 * 20 + 7}
    assert set(np.flatnonzero(~in_region).tolist()) == enclosed


def test_relaxation_zones_are_monotonic_rings_outward(tmp_path):
    path = write_grid_mesh(tmp_path / "m.nc", **GRID)
    with xr.open_dataset(path, decode_timedelta=False) as ds:
        coc = ds.cellsOnCell.values.astype(np.int64)
        nec = ds.nEdgesOnCell.values.astype(np.int64)

    in_region = np.zeros(coc.shape[0], dtype=bool)
    in_region[210] = True   # a single interior seed cell, row 10 col 10 of 20x20
    zone = _relaxation_zones(coc, nec, in_region, N_BDY_LAYERS)

    assert zone[210] == 0
    # every direct neighbour of the seed is layer 1, not further out
    for n in coc[210][coc[210] > 0]:
        assert zone[n - 1] == 1
    assert zone.max() == N_BDY_LAYERS
    assert (zone == -1).any()   # cells far enough away are dropped entirely


# --------------------------------------------------------------------- create_region

def test_the_interior_is_zone_zero_and_nothing_exceeds_n_bdy_layers(tmp_path):
    path = write_grid_mesh(tmp_path / "m.nc", **GRID)
    out = create_region(path, tmp_path / "region.nc", BOX_LAT, BOX_LON)

    with xr.open_dataset(out, decode_timedelta=False) as ds:
        assert ds.bdyMaskCell.values.min() == 0
        assert ds.bdyMaskCell.values.max() <= N_BDY_LAYERS
        assert (ds.bdyMaskCell.values == 0).sum() > 0


def test_the_subset_is_smaller_than_the_source_mesh(tmp_path):
    path = write_grid_mesh(tmp_path / "m.nc", **GRID)
    with xr.open_dataset(path, decode_timedelta=False) as src:
        n_cells_global = src.sizes["nCells"]

    out = create_region(path, tmp_path / "region.nc", BOX_LAT, BOX_LON)
    with xr.open_dataset(out, decode_timedelta=False) as ds:
        assert 0 < ds.sizes["nCells"] < n_cells_global


def test_connectivity_is_renumbered_with_no_dangling_or_out_of_range_ids(tmp_path):
    path = write_grid_mesh(tmp_path / "m.nc", **GRID)
    out = create_region(path, tmp_path / "region.nc", BOX_LAT, BOX_LON)

    with xr.open_dataset(out, decode_timedelta=False) as ds:
        n_cells = ds.sizes["nCells"]
        n_edges = ds.sizes["nEdges"]
        n_vertices = ds.sizes["nVertices"]
        assert ds.cellsOnCell.values.min() >= 0
        assert ds.cellsOnCell.values.max() <= n_cells
        assert ds.cellsOnEdge.values.min() >= 0
        assert ds.cellsOnEdge.values.max() <= n_cells
        assert ds.verticesOnCell.values.min() >= 0
        assert ds.verticesOnCell.values.max() <= n_vertices
        assert ds.edgesOnCell.values.min() >= 0
        assert ds.edgesOnCell.values.max() <= n_edges
        assert ds.cellsOnVertex.values.min() >= 0
        assert ds.cellsOnVertex.values.max() <= n_cells
        # every kept cell's own id, after renumbering, is exactly its new position
        assert np.array_equal(ds.indexToCellID.values, np.arange(1, n_cells + 1))


def test_an_edge_fully_surrounded_by_kept_cells_takes_the_more_interior_zone(tmp_path):
    """Covers only the all-neighbours-kept branch of `_neighbour_zone` -- an
    edge always has exactly 2 sides, so it can never exercise the
    some-neighbours-dropped branch on its own (see the vertex tests below,
    and `_neighbour_zone`'s docstring, for why that needs 3+ sides)."""
    path = write_grid_mesh(tmp_path / "m.nc", **GRID)
    out = create_region(path, tmp_path / "region.nc", BOX_LAT, BOX_LON)

    with xr.open_dataset(out, decode_timedelta=False) as ds:
        coe = ds.cellsOnEdge.values
        cell_zone = ds.bdyMaskCell.values
        edge_zone = ds.bdyMaskEdge.values
        both_kept = (coe[:, 0] > 0) & (coe[:, 1] > 0)
        expected = np.minimum(cell_zone[coe[both_kept, 0] - 1],
                              cell_zone[coe[both_kept, 1] - 1])
        assert np.array_equal(edge_zone[both_kept], expected)


def test_neighbour_zone_takes_the_min_when_every_side_is_kept():
    from gmpas.prep.region import _neighbour_zone

    zone = np.array([0, 3, 5, 2, -1])
    cells_on = np.array([[1, 2, 3, 4]])   # all 4 sides kept: zones 0, 3, 5, 2
    assert _neighbour_zone(zone, cells_on).tolist() == [0]


def test_neighbour_zone_takes_the_max_when_a_side_is_missing_or_dropped():
    """The case a 2-sided edge structurally cannot exercise: with at least
    one side missing (0-fill) or dropped (beyond N_BDY_LAYERS, zone -1),
    this element sits at the outer rim of whatever it does touch, so it
    takes the *more boundary-like* (max) of its kept sides, not the min.
    Kept sides here are zones {0, 5} (cell 1 and cell 3); a plain min
    -of-kept, which is what an earlier version of this function did, would
    wrongly give 0 instead of 5.
    """
    from gmpas.prep.region import _neighbour_zone

    zone = np.array([0, 3, 5, 2, -1])
    # 1-based cellsOnCell-style row: cell 1 (zone 0), cell 3 (zone 5),
    # a 0-fill "no neighbour" slot, and cell 5 (zone -1, dropped)
    cells_on = np.array([[1, 3, 0, 5]])
    assert _neighbour_zone(zone, cells_on).tolist() == [5]


def test_graph_info_header_matches_the_subset(tmp_path):
    path = write_grid_mesh(tmp_path / "m.nc", **GRID)
    out = create_region(path, tmp_path / "region.nc", BOX_LAT, BOX_LON)
    graph = write_graph_info(out, tmp_path / "graph.info")

    with xr.open_dataset(out, decode_timedelta=False) as ds:
        n_cells = ds.sizes["nCells"]
        coe = ds.cellsOnEdge.values
        expected_interior_edges = int(((coe[:, 0] > 0) & (coe[:, 1] > 0)).sum())

    lines = graph.read_text().splitlines()
    header_cells, header_edges = (int(x) for x in lines[0].split())
    assert header_cells == n_cells
    assert header_edges == expected_interior_edges
    assert len(lines) == n_cells + 1


def test_at_least_three_boundary_points_are_required(tmp_path):
    path = write_grid_mesh(tmp_path / "m.nc", **GRID)
    with pytest.raises(ValueError, match="at least 3"):
        create_region(path, tmp_path / "region.nc", [10.0, 12.0], [10.0, 12.0])


def test_a_degenerate_boundary_that_forms_no_wall_is_rejected(tmp_path):
    path = write_grid_mesh(tmp_path / "m.nc", **GRID)
    # three points close enough together to all snap to the same cell --
    # no real wall, so the flood fill leaks across the entire mesh
    with pytest.raises(ValueError, match="whole"):
        create_region(path, tmp_path / "region.nc",
                      [20.0, 20.001, 20.002], [20.0, 20.001, 20.002])


def test_a_missing_variable_names_what_is_missing(tmp_path):
    from conftest import write_mesh

    path = write_mesh(tmp_path / "m.nc", [(100.0, 0.0)])   # no cellsOnCell at all
    with pytest.raises(KeyError, match="cellsOnCell"):
        create_region(path, tmp_path / "region.nc", BOX_LAT, BOX_LON)


def test_a_missing_mesh_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="No such mesh file"):
        create_region(tmp_path / "absent.nc", tmp_path / "region.nc", BOX_LAT, BOX_LON)


# -------------------------------------------------------------------- CLI

def test_cli_prep_create_region(tmp_path):
    path = write_grid_mesh(tmp_path / "m.nc", **GRID)
    out = tmp_path / "region.nc"

    assert main(["prep", "create-region", str(path), "-o", str(out),
                "--polygon", "10,10", "10,28", "28,28", "28,10"]) == 0
    assert out.exists()
    assert (tmp_path / "region.graph.info").exists()


def test_cli_default_output_path_is_derived_from_the_input(tmp_path, monkeypatch):
    path = write_grid_mesh(tmp_path / "m.nc", **GRID)
    monkeypatch.chdir(tmp_path)

    assert main(["prep", "create-region", str(path),
                "--polygon", "10,10", "10,28", "28,28", "28,10"]) == 0
    assert (tmp_path / "m.region.nc").exists()


def test_cli_rejects_a_polygon_point_that_is_not_a_lat_lon_pair(tmp_path, capsys):
    path = write_grid_mesh(tmp_path / "m.nc", **GRID)

    assert main(["prep", "create-region", str(path),
                "--polygon", "10,10", "not-a-point", "28,28"]) == 1
    assert "'lat,lon' pairs" in capsys.readouterr().err


def test_cli_accepts_an_explicit_interior_point(tmp_path):
    path = write_grid_mesh(tmp_path / "m.nc", **GRID)
    out = tmp_path / "region.nc"

    assert main(["prep", "create-region", str(path), "-o", str(out),
                "--polygon", "10,10", "10,28", "28,28", "28,10",
                "--point", "19", "19"]) == 0
    assert out.exists()
