"""The KD-tree Voronoi rasterizer and the poly/raster decision."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import write_mesh
from gmpas import RASTER_THRESHOLD, rasterize, should_raster, target_grid


# ------------------------------------------------------------------ the grid


def test_target_grid_returns_pixel_centres_not_edges():
    lon, lat = target_grid((0.0, 10.0, 0.0, 5.0), nx=10, ny=5)
    assert lon == pytest.approx(np.arange(10) + 0.5)
    assert lat == pytest.approx(np.arange(5) + 0.5)


# ------------------------------------------------------------------ sampling


def test_pixels_take_the_value_of_the_nearest_cell_centre(simple_mesh):
    """Sampled, never interpolated: a pixel gets the model's own cell value."""
    values = np.array([10.0, 20.0, 30.0, 40.0])
    img, lon, lat = rasterize(simple_mesh, values, nx=200, ny=120)

    for cell, value in enumerate(values):
        i = int(np.abs(lat - simple_mesh.lat_cell[cell]).argmin())
        j = int(np.abs(lon - simple_mesh.lon_cell[cell]).argmin())
        assert img[i, j] == value


def test_no_intermediate_values_are_invented(simple_mesh):
    """Sharp gradients survive: every finite pixel is one of the input values."""
    values = np.array([1.0, 2.0, 3.0, 4.0])
    img, _, _ = rasterize(simple_mesh, values, nx=150, ny=90)

    finite = img[np.isfinite(img)]
    assert set(np.unique(finite)).issubset(set(values))


def test_image_is_oriented_for_imshow_origin_lower(simple_mesh):
    img, lon, lat = rasterize(simple_mesh, np.arange(4.0), nx=64, ny=32)
    assert img.shape == (32, 64)
    assert lat[0] < lat[-1]


# ------------------------------------------------------------------- masking


def test_pixels_outside_a_regional_mesh_are_blanked(simple_mesh):
    """Without the mask, far pixels snap to a boundary cell and smear it."""
    img, _, _ = rasterize(simple_mesh, np.arange(4.0), nx=200, ny=120)

    # four ~100 km cells scattered over 30 degrees of longitude: most of the
    # frame is nowhere near a cell centre
    assert np.isnan(img).mean() > 0.9


def test_masking_uses_the_mesh_own_sphere_not_earth(tmp_path):
    """A reduced-radius run must not be blanked out of existence.

    These cells are ~460 m across on a planet 1/120 of Earth's radius, which is
    about half a degree of arc -- plainly visible. Divide those metres by
    Earth's radius instead and every cell shrinks to 0.004 degrees, far below
    one pixel here, so the entire field would come back NaN.
    """
    from gmpas import MpasMesh
    from gmpas.mesh import EARTH_RADIUS

    path = write_mesh(tmp_path / "small_planet.nc", [(0.0, 0.0), (2.0, 0.0)],
                      radius_deg=0.5, sphere_radius=EARTH_RADIUS / 120.0,
                      areas=[6.7e5, 6.7e5])
    mesh = MpasMesh.load(path)

    img, lon, lat = rasterize(mesh, np.array([1.0, 2.0]), nx=64, ny=24)

    for cell, value in enumerate([1.0, 2.0]):
        i = int(np.abs(lat - mesh.lat_cell[cell]).argmin())
        j = int(np.abs(lon - mesh.lon_cell[cell]).argmin())
        assert img[i, j] == value

    assert np.isfinite(img).any()


def _unbounded_reference(mesh, values, extent, nx, ny):
    """What rasterize did before the search bound: a plain nearest query."""
    lon, lat = target_grid(extent, nx, ny)
    lon2, lat2 = np.meshgrid(lon, lat)
    lon_r, lat_r = np.radians(lon2), np.radians(lat2)
    pts = np.stack([np.cos(lat_r) * np.cos(lon_r),
                    np.cos(lat_r) * np.sin(lon_r),
                    np.sin(lat_r)], axis=-1).reshape(-1, 3)
    dist, idx = mesh.tree().query(pts)
    img = np.asarray(values, dtype=np.float64)[idx].reshape(ny, nx)
    radius = np.sqrt(mesh.area_cell / np.pi) / mesh.sphere_radius
    return np.where(dist.reshape(ny, nx) > 2.0 * radius[idx].reshape(ny, nx),
                    np.nan, img)


def test_bounding_the_search_does_not_change_the_result(simple_mesh):
    """The bound only skips pixels the mask would have blanked anyway.

    Unbounded nearest-neighbour queries cannot prune when the point is far
    outside the mesh, so a regional mesh drawn on a wide frame spent nearly
    all its time on empty space. Capping the search at the largest cell's own
    cutoff is a pure speedup -- the output must stay identical.
    """
    values = np.array([10.0, 20.0, 30.0, 40.0])
    for extent in [simple_mesh.extent, (-180.0, 180.0, -90.0, 90.0)]:
        got, _, _ = rasterize(simple_mesh, values, extent, nx=120, ny=70)
        want = _unbounded_reference(simple_mesh, values, extent, 120, 70)
        assert np.array_equal(got, want, equal_nan=True)


def test_a_frame_far_from_the_mesh_is_entirely_blank(simple_mesh):
    """And must not raise: every pixel falls outside the search bound."""
    img, _, _ = rasterize(simple_mesh, np.arange(4.0),
                          (-60.0, -20.0, -60.0, -30.0), nx=40, ny=20)
    assert np.isnan(img).all()


def test_masking_can_be_turned_off(simple_mesh):
    img, _, _ = rasterize(simple_mesh, np.arange(4.0), nx=64, ny=32,
                          mask_outside=False)
    assert np.isfinite(img).all()


def test_explicit_extent_is_honoured(simple_mesh):
    box = (95.0, 105.0, -5.0, 5.0)
    img, lon, lat = rasterize(simple_mesh, np.arange(4.0), box, nx=50, ny=50)

    assert lon.min() > 95.0 and lon.max() < 105.0
    assert lat.min() > -5.0 and lat.max() < 5.0
    # only the cell at (100, 0) is inside this box
    assert set(np.unique(img[np.isfinite(img)])) == {0.0}


# ------------------------------------------------------------ path selection


def test_small_meshes_draw_polygons_and_large_ones_rasterize(simple_mesh):
    assert not should_raster(simple_mesh, "auto")

    simple_mesh.cell_verts = np.zeros((RASTER_THRESHOLD, 6, 2))
    assert should_raster(simple_mesh, "auto")


def test_explicit_method_overrides_the_threshold(simple_mesh):
    assert should_raster(simple_mesh, "raster")
    assert not should_raster(simple_mesh, "poly")


def test_unknown_method_is_rejected(simple_mesh):
    with pytest.raises(ValueError, match="auto"):
        should_raster(simple_mesh, "polygons")
