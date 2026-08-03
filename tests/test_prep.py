"""The preprocessing section: a mesh viewer that never touches a data file."""

from __future__ import annotations

import json

import numpy as np
import pytest

from gmpas.cli import build_parser
from gmpas.prep import page
from gmpas.prep.meshview import FIELDS, MeshViewer


@pytest.fixture
def viewer(simple_mesh_file):
    return MeshViewer(simple_mesh_file, nx=64, ny=32)


# -- the point of the whole thing: no data file anywhere ---------------------


def test_a_mesh_alone_is_enough(viewer, simple_mesh):
    """`gmpas view` needs a Series; this needs only the mesh."""
    assert viewer.mesh.n_cells == simple_mesh.n_cells
    assert viewer.home == simple_mesh.extent


def test_meta_describes_the_mesh_and_offers_only_geometry_fields(viewer):
    meta = viewer.describe()
    assert meta["cells"] == 4
    assert meta["regional"] is True
    assert [f["name"] for f in meta["fields"]] == list(FIELDS)
    # a fixed ramp is shipped, because the page has no colormap picker
    assert len(meta["ramp"]) == 32
    assert meta["ramp"][0].startswith("#")


def test_meta_is_json_serialisable(viewer):
    """It is handed straight to json.dumps by the handler."""
    assert json.loads(json.dumps(viewer.describe()))["cells"] == 4


def test_cell_width_matches_the_mesh(viewer, simple_mesh):
    assert np.allclose(viewer.values("cell_width_km"), simple_mesh.cell_width_km)


def test_area_field_is_in_square_km(viewer, simple_mesh):
    assert np.allclose(viewer.values("cell_area_km2"),
                       np.asarray(simple_mesh.area_cell) / 1e6)


def test_an_unknown_field_says_what_it_offers(viewer):
    with pytest.raises(KeyError, match="cell_width_km"):
        viewer.values("theta")


def test_values_are_computed_once(viewer):
    assert viewer.values("cell_width_km") is viewer.values("cell_width_km")


# -- the fixed scale ---------------------------------------------------------


def test_the_scale_is_the_whole_mesh_not_the_view(viewer, simple_mesh):
    """No vmin/vmax controls, so panning must not restretch the colours."""
    width = simple_mesh.cell_width_km
    whole = viewer.frame("cell_width_km", simple_mesh.extent)[1:]
    # a corner of the domain, holding a subset of the cells
    lon0, lon1, lat0, lat1 = simple_mesh.extent
    corner = viewer.frame("cell_width_km",
                          (lon0, (lon0 + lon1) / 2, lat0, (lat0 + lat1) / 2))[1:]
    assert whole == corner == (pytest.approx(width.min()), pytest.approx(width.max()))


# -- rendering ---------------------------------------------------------------


def test_a_frame_is_a_png(viewer, simple_mesh):
    png, lo, hi = viewer.frame("cell_width_km", simple_mesh.extent)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert lo <= hi


def test_the_view_index_is_reused_across_frames(viewer, simple_mesh):
    """The pixel-to-cell map depends on the box, not the field."""
    viewer.frame("cell_width_km", simple_mesh.extent)
    viewer.frame("cell_area_km2", simple_mesh.extent)
    assert len(viewer._views) == 1


# -- the layout --------------------------------------------------------------


def test_the_page_drops_the_postprocessing_controls():
    html = page("gmpas prep view")
    for gone in ("#vmin", "#vmax", "id=\"cmap\"", "animate", "api/export/",
                 "api/probe", "id=\"time\"", "id=\"level\""):
        assert gone not in html
    # but keeps what a preprocessing step needs. The API paths are relative so
    # the page works mounted under a prefix as well as at the root
    for kept in ("id=\"elon0\"", "applyext", "copyext", "id=\"grat\"",
                 "id=\"scalebar\"", "api/frame", "api/overlay"):
        assert kept in html
    assert "/api/" not in html


def test_the_page_splices_in_a_step_panel():
    html = page("t", panel="<div id='mine'>hi</div>", script="say('ready');")
    assert "<div id='mine'>hi</div>" in html
    assert "say('ready');" in html
    assert "<title>t</title>" in html


# -- the CLI -----------------------------------------------------------------


def test_prep_view_parses(simple_mesh_file):
    args = build_parser().parse_args(["prep", "view", str(simple_mesh_file)])
    assert args.mesh_file == str(simple_mesh_file)
    assert args.func.__name__ == "_prep_view"


def test_bare_prep_prints_help_rather_than_failing(capsys):
    args = build_parser().parse_args(["prep"])
    assert args.func(args) == 0
    assert "view" in capsys.readouterr().out


def test_the_postprocessing_commands_are_untouched():
    """Adding a section must not move the existing verbs."""
    p = build_parser()
    for argv in (["info", "x.nc"], ["plot", "x.nc", "t2m"], ["view", "x.nc"]):
        assert p.parse_args(argv).func is not None
