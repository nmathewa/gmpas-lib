"""Extents, presets, and where figures get written."""

from __future__ import annotations

import pytest

from gmpas import EXTENTS, Style, resolve_extent
from gmpas.paths import cache_dir, resolve_path


def test_named_extent_resolves_to_its_box(simple_mesh):
    assert resolve_extent(simple_mesh, "maritime_continent") == EXTENTS[
        "maritime_continent"
    ]


def test_no_extent_falls_back_to_the_mesh_itself(simple_mesh):
    assert resolve_extent(simple_mesh, None) == simple_mesh.extent
    assert resolve_extent(simple_mesh, "") == simple_mesh.extent


def test_an_explicit_box_passes_through(simple_mesh):
    assert resolve_extent(simple_mesh, (0, 10, -5, 5)) == (0.0, 10.0, -5.0, 5.0)


def test_unknown_extent_lists_the_known_ones(simple_mesh):
    with pytest.raises(KeyError, match="mjo_basin"):
        resolve_extent(simple_mesh, "atlantic")


def test_malformed_box_is_rejected(simple_mesh):
    with pytest.raises(ValueError, match="lon_min"):
        resolve_extent(simple_mesh, (0, 10, -5))


def test_mesh_preset_makes_cell_outlines_visible():
    """Every other preset hides them, or a big mesh turns into a black smear."""
    assert Style.preset("mesh").edge_lw > 0
    assert Style().edge_lw == 0
    assert Style.preset("poster").edge_lw == 0


def test_presets_scale_together():
    poster, notebook = Style.preset("poster"), Style.preset("notebook")
    assert poster.dpi > notebook.dpi
    assert poster.figsize > notebook.figsize
    assert poster.title_size > notebook.title_size


# ---------------------------------------------------------------------- paths


def test_cache_location_is_overridable(monkeypatch, tmp_path):
    """The MCP server points this back at its own project directory."""
    monkeypatch.setenv("GMPAS_CACHE_DIR", str(tmp_path / "elsewhere"))
    assert cache_dir() == tmp_path / "elsewhere"


def test_cache_defaults_outside_the_install_tree(monkeypatch):
    monkeypatch.delenv("GMPAS_CACHE_DIR", raising=False)
    assert cache_dir().parts[-2:] == ("gmpas", "mesh")


def test_absolute_paths_pass_through(tmp_path):
    assert resolve_path(tmp_path / "x.nc") == tmp_path / "x.nc"


def test_relative_paths_try_the_data_dir_first(monkeypatch, tmp_path):
    monkeypatch.setenv("GMPAS_DATA_DIR", str(tmp_path))
    (tmp_path / "diag.nc").touch()

    assert resolve_path("diag.nc") == tmp_path / "diag.nc"


def test_relative_paths_fall_back_to_the_working_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("GMPAS_DATA_DIR", str(tmp_path / "empty"))
    monkeypatch.chdir(tmp_path)

    assert resolve_path("absent.nc") == tmp_path / "absent.nc"
