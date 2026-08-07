"""Rendering. Skipped entirely when the optional `plot` extra is not installed."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import write_mesh
from gmpas import MpasMesh, cell_field, edge_field, mesh_structure, save_figure

pytest.importorskip("matplotlib", reason="needs the `plot` extra")
pytest.importorskip("cartopy", reason="needs the `plot` extra")

import matplotlib  # noqa: E402

matplotlib.use("Agg")


@pytest.fixture
def values(simple_mesh):
    return np.array([1.0, 2.0, 3.0, 4.0])


def data_collections(ax, kind=None):
    """The artists carrying mesh data.

    cartopy puts its own artists on the axes too -- the coastline feature and
    the gridlines -- so counting `ax.collections` directly counts decoration.
    """
    from matplotlib.collections import LineCollection, PolyCollection

    kind = kind or (PolyCollection, LineCollection)
    return [c for c in ax.collections if isinstance(c, kind)]


# ------------------------------------------------------------------ dispatch


def test_polygon_path_draws_one_collection_per_mesh(simple_mesh, values):
    fig, ax = cell_field(simple_mesh, values, method="poly")
    assert len(data_collections(ax)) == 1
    assert not ax.images
    matplotlib.pyplot.close(fig)


def test_raster_path_draws_an_image_instead(simple_mesh, values):
    fig, ax = cell_field(simple_mesh, values, method="raster", nx=64, ny=32)
    assert len(ax.images) == 1
    matplotlib.pyplot.close(fig)


def test_antimeridian_cells_are_drawn_twice(tmp_path):
    """A second copy 360 degrees west, so the seam is covered on both sides."""
    mesh = MpasMesh.load(write_mesh(tmp_path / "dateline.nc",
                                    [(180.0, 0.0), (100.0, 0.0)]))
    fig, ax = cell_field(mesh, np.array([1.0, 2.0]), method="poly")

    drawn = data_collections(ax)
    assert len(drawn) == 2
    main, seam = (c.get_paths() for c in drawn)
    assert len(seam) == 1                      # only the wrapped cell is copied
    assert len(main) == 2
    matplotlib.pyplot.close(fig)


def test_edge_fields_are_drawn_on_the_faces(simple_mesh):
    from matplotlib.collections import LineCollection

    fig, ax = edge_field(simple_mesh, np.arange(simple_mesh.n_edges, dtype=float))
    drawn = data_collections(ax, LineCollection)

    assert len(drawn) == 1
    assert len(drawn[0].get_paths()) == simple_mesh.n_edges
    matplotlib.pyplot.close(fig)


def test_mesh_structure_is_labelled_in_km(simple_mesh):
    fig, ax = mesh_structure(simple_mesh)
    assert "km" in fig.axes[-1].get_ylabel()          # the colorbar
    assert "4 cells" in ax.get_title()
    matplotlib.pyplot.close(fig)


def test_title_position_survives_notebook_style_tight_bbox(simple_mesh, values):
    """Regression: a cartopy GeoAxes title can get pinned at y=inf.

    `Gridliner.geo_labels` defaults True whenever `draw_labels=True` does,
    even with top/right labels turned off. `GeoAxes._update_title_position`
    then measures the (hidden) top label artists anyway; their null bbox's
    `.ymax` comes back `inf` rather than the intended `-inf`, and the title
    gets pinned there. A title at y=inf NaNs the axes' tight bbox, so
    `bbox_inches="tight"` -- what Jupyter's inline backend uses by default
    to auto-display a returned figure -- crops the map away entirely and
    leaves only the colorbar. `_basemap` clears `geo_labels` too to avoid
    this; guard both the position and the rendered size so a regression
    here is caught without a notebook.
    """
    import io

    fig, ax = cell_field(simple_mesh, values, method="poly")
    assert ax.title.get_position()[1] == pytest.approx(1.0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    matplotlib.pyplot.close(fig)

    full = io.BytesIO()
    fig2, _ = cell_field(simple_mesh, values, method="poly")
    fig2.savefig(full, format="png")
    matplotlib.pyplot.close(fig2)

    # a crop-to-colorbar comes back at a few KB; the real map is much larger
    assert buf.tell() > full.tell() * 0.5


# -------------------------------------------------------------------- errors


def test_a_field_of_the_wrong_length_names_both_counts(simple_mesh):
    with pytest.raises(ValueError, match="4 cells"):
        cell_field(simple_mesh, np.zeros(7))


def test_a_cell_field_passed_to_the_edge_plotter_is_caught(simple_mesh, values):
    with pytest.raises(ValueError, match="24 edges"):
        edge_field(simple_mesh, values)


# --------------------------------------------------------------------- saving


def test_save_figure_writes_where_told_and_creates_parents(simple_mesh, values,
                                                           tmp_path):
    fig, _ = cell_field(simple_mesh, values, method="poly")
    out = save_figure(fig, tmp_path / "deep" / "nested" / "mslp")

    assert out == tmp_path / "deep" / "nested" / "mslp.png"
    assert out.stat().st_size > 0


def test_an_explicit_suffix_is_kept(simple_mesh, values, tmp_path):
    fig, _ = cell_field(simple_mesh, values, method="poly")
    assert save_figure(fig, tmp_path / "mslp.pdf").suffix == ".pdf"
