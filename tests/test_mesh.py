"""Geometry building, MPAS conventions, and the on-disk cache."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import EDGE_ANGLES, write_mesh
from gmpas import MpasMesh, reconstruct_cell_winds
from gmpas.mesh import EARTH_RADIUS, _wrap180, cache_path


# ------------------------------------------------------------------ conventions


def test_longitude_is_converted_from_radians_on_0_360():
    """MPAS stores lon on [0, 2pi); everything downstream wants [-180, 180)."""
    assert _wrap180(np.array([0.0, 90.0, 270.0, 359.0])) == pytest.approx(
        [0.0, 90.0, -90.0, -1.0]
    )


def test_cell_centres_round_trip_through_radians(simple_mesh):
    assert simple_mesh.lon_cell == pytest.approx([100.0, 110.0, 120.0, 130.0])
    assert simple_mesh.lat_cell == pytest.approx([0.0, 5.0, -5.0, 10.0])


def test_ragged_vertices_on_cell_are_padded_with_the_last_real_vertex(tmp_path):
    """A pentagon in a maxEdges=6 array: the unused slot repeats vertex 5.

    The repeat is degenerate, so matplotlib draws it identically to a properly
    closed five-sided polygon -- which is why the fill can be vectorized
    instead of looping over cells.
    """
    path = write_mesh(tmp_path / "ragged.nc", [(0.0, 0.0), (10.0, 0.0)],
                      n_verts=[5, 6])
    mesh = MpasMesh.load(path)

    pentagon = mesh.cell_verts[0]
    assert pentagon[5] == pytest.approx(pentagon[4])      # padded
    assert pentagon[4] != pytest.approx(pentagon[3])      # real vertices differ

    hexagon = mesh.cell_verts[1]
    assert hexagon[5] != pytest.approx(hexagon[4])        # nothing padded


def test_connectivity_is_read_as_one_based(simple_mesh, simple_mesh_file):
    """verticesOnCell is 1-based in the file; vertex 1 must map to index 0."""
    import xarray as xr

    with xr.open_dataset(simple_mesh_file) as ds:
        first = int(ds.verticesOnCell.values[0, 0])
        lon_v = np.degrees(float(ds.lonVertex.values[first - 1]))

    assert simple_mesh.cell_verts[0, 0, 0] == pytest.approx(_wrap180(lon_v))


# ---------------------------------------------------------------- antimeridian


def test_cell_straddling_the_antimeridian_is_flagged_and_unwrapped(tmp_path):
    """Vertices at both +179 and -179 would smear across the map if drawn raw.

    They are shifted onto a single branch just past +180; the renderer draws a
    second copy at -360 so the seam is covered on both sides.
    """
    path = write_mesh(tmp_path / "dateline.nc", [(180.0, 0.0), (100.0, 0.0)])
    mesh = MpasMesh.load(path)

    assert mesh.cell_wrapped[0]
    assert not mesh.cell_wrapped[1]

    lons = mesh.cell_verts[0, :, 0]
    assert lons.max() - lons.min() < 180.0        # contiguous, single branch
    assert lons.max() > 180.0                     # living just past the seam


def test_edges_get_the_same_antimeridian_treatment(tmp_path):
    path = write_mesh(tmp_path / "dateline.nc", [(180.0, 0.0)])
    mesh = MpasMesh.load(path)

    assert mesh.edge_wrapped.any()
    wrapped = mesh.edge_segs[mesh.edge_wrapped][..., 0]
    assert (wrapped.max(axis=1) - wrapped.min(axis=1) < 180.0).all()


# ------------------------------------------------------------------ mesh units


def test_jigsaw_meshes_are_redimensionalised_to_square_metres(tmp_path):
    """sphere_radius=1 means areaCell is non-dimensional until init_atmosphere.

    Both files describe the same 1e10 m^2 cells; area_cell must agree.
    """
    real = write_mesh(tmp_path / "real.nc", [(0.0, 0.0)], sphere_radius=EARTH_RADIUS)
    unit = write_mesh(tmp_path / "unit.nc", [(0.0, 0.0)], sphere_radius=1.0)

    assert MpasMesh.load(real).area_cell == pytest.approx(1.0e10)
    assert MpasMesh.load(unit).area_cell == pytest.approx(1.0e10, rel=1e-9)


def test_mesh_carries_the_sphere_it_is_actually_on(tmp_path):
    """Reduced-radius ("small planet") runs are a real MPAS configuration."""
    small = EARTH_RADIUS / 120.0
    path = write_mesh(tmp_path / "small_planet.nc", [(0.0, 0.0)],
                      sphere_radius=small, areas=[6.7e5])

    assert MpasMesh.load(path).sphere_radius == pytest.approx(small)


def test_a_non_dimensional_mesh_is_assumed_to_be_earth_sized(tmp_path):
    """Stated assumption, not a hidden one: nothing in a unit-sphere file says
    which planet it is for, and MPAS's own default is Earth."""
    path = write_mesh(tmp_path / "unit.nc", [(0.0, 0.0)], sphere_radius=1.0)

    assert MpasMesh.load(path).sphere_radius == pytest.approx(EARTH_RADIUS)


def test_sphere_radius_survives_the_cache(tmp_path):
    small = EARTH_RADIUS / 120.0
    path = write_mesh(tmp_path / "small_planet.nc", [(0.0, 0.0)],
                      sphere_radius=small, areas=[6.7e5])

    built = MpasMesh.load(path, use_cache=False)
    MpasMesh.load(path)                       # populates the cache
    cached = MpasMesh.load(path)

    assert cached.sphere_radius == pytest.approx(built.sphere_radius)
    assert isinstance(cached.sphere_radius, float)


def test_cell_width_is_hexagon_equivalent_in_km(simple_mesh):
    # a hexagon of area A has centre-to-face width 2*sqrt(A / (2*sqrt(3)))
    expected = 2.0 * np.sqrt(1.0e10 / (2.0 * np.sqrt(3.0))) / 1000.0
    assert simple_mesh.cell_width_km == pytest.approx(expected)


def test_kdtree_coordinates_are_normalised_regardless_of_sphere_radius(tmp_path):
    """xCell/yCell/zCell come on the sphere radius, so they cannot be trusted raw."""
    unit = MpasMesh.load(write_mesh(tmp_path / "u.nc", [(30.0, 10.0)], sphere_radius=1.0))
    real = MpasMesh.load(write_mesh(tmp_path / "r.nc", [(30.0, 10.0)]))

    assert np.linalg.norm(unit.xyz_cell, axis=-1) == pytest.approx(1.0)
    assert unit.xyz_cell == pytest.approx(real.xyz_cell)


# ---------------------------------------------------------------------- extent


def test_regional_mesh_reports_its_own_extent(simple_mesh):
    assert not simple_mesh.is_global
    lon_min, lon_max, lat_min, lat_max = simple_mesh.extent
    assert 98.0 < lon_min < 100.0
    assert 130.0 < lon_max < 132.0
    assert lat_min < -5.0 and lat_max > 10.0


def test_a_pole_to_pole_sliver_is_not_global(tmp_path):
    """Guards against reintroducing the old latitude-span heuristic.

    Two 100 km cells sitting at +-89 span the poles but cover 0.004% of the
    planet. Reaching the poles is not the same as covering the sphere.
    """
    path = write_mesh(tmp_path / "sliver.nc", [(0.0, 89.0), (0.0, -89.0)])
    mesh = MpasMesh.load(path)

    assert mesh.lat_cell.max() > 88.0 and mesh.lat_cell.min() < -88.0
    assert not mesh.is_global


def test_crossing_the_dateline_does_not_make_a_mesh_global(tmp_path):
    """Wrapping is not covering.

    A regional Pacific domain straddles the antimeridian while covering a
    sliver of the planet. Treating that as global handed it the whole-sphere
    extent, so it rendered as a speck on a world map.
    """
    path = write_mesh(tmp_path / "pacific.nc", [(179.0, 0.0), (-179.0, 0.0)])
    mesh = MpasMesh.load(path)

    assert mesh.cell_wrapped.any()          # it really does cross
    assert not mesh.is_global
    assert mesh.coverage < 0.01


def test_dateline_extent_is_returned_in_the_unwrapped_frame(tmp_path):
    """The domain is only contiguous past +180, so that is where it is framed.

    A plain min/max over wrapped longitudes would span -180..180 and lose the
    domain entirely.
    """
    path = write_mesh(tmp_path / "pacific.nc",
                      [(178.0, 0.0), (-178.0, 0.0)], radius_deg=0.5)
    lon_min, lon_max, _, _ = MpasMesh.load(path).extent

    assert lon_max > 180.0                  # runs past the seam
    assert lon_max - lon_min < 10.0         # tight around the domain
    assert 177.0 < lon_min < 179.0
    assert 181.0 < lon_max < 183.0


def test_a_mesh_covering_the_sphere_is_global(tmp_path):
    """Coverage is what decides, and it is computed from areaCell."""
    n = 60
    rng = np.random.default_rng(0)
    centres = np.stack([rng.uniform(-179, 179, n), rng.uniform(-89, 89, n)], -1)
    sphere = 4.0 * np.pi * EARTH_RADIUS**2
    path = write_mesh(tmp_path / "global.nc", centres,
                      areas=np.full(n, sphere / n))
    mesh = MpasMesh.load(path)

    assert mesh.coverage == pytest.approx(1.0, rel=1e-6)
    assert mesh.is_global
    assert mesh.extent == (-180.0, 180.0, -90.0, 90.0)


def test_ordinary_regional_extent_is_unchanged(simple_mesh):
    """The common case must not move: no wrap, so the wrapped frame wins."""
    lon_min, lon_max, lat_min, lat_max = simple_mesh.extent

    assert lon_max <= 180.0
    assert 98.0 < lon_min < 100.0
    assert 130.0 < lon_max < 132.0
    assert lat_min < -5.0 and lat_max > 10.0


# -------------------------------------------------------------- coordinate dtype


def test_float32_meshes_are_not_upcast(tmp_path):
    """Forcing float64 doubled the two largest arrays for no extra digits."""
    path = write_mesh(tmp_path / "f32.nc", [(0.0, 0.0), (10.0, 0.0)],
                      coord_dtype=np.float32)
    mesh = MpasMesh.load(path, use_cache=False)

    assert mesh.cell_verts.dtype == np.float32
    assert mesh.edge_segs.dtype == np.float32


def test_float64_meshes_keep_their_precision(tmp_path):
    path = write_mesh(tmp_path / "f64.nc", [(0.0, 0.0), (10.0, 0.0)],
                      coord_dtype=np.float64)
    mesh = MpasMesh.load(path, use_cache=False)

    assert mesh.cell_verts.dtype == np.float64
    assert mesh.edge_segs.dtype == np.float64


def test_dtype_survives_the_cache(tmp_path):
    path = write_mesh(tmp_path / "f32.nc", [(0.0, 0.0)], coord_dtype=np.float32)
    MpasMesh.load(path)
    assert MpasMesh.load(path).cell_verts.dtype == np.float32


# ----------------------------------------------------------------- nearest cell


def test_cell_of_returns_the_containing_cell(simple_mesh):
    """On a Voronoi mesh the nearest centre *is* the containing cell."""
    lon = np.array([100.1, 109.5, 120.2, 129.9])
    lat = np.array([0.05, 4.8, -5.1, 10.2])
    assert simple_mesh.cell_of(lon, lat).tolist() == [0, 1, 2, 3]


def test_cell_of_preserves_input_shape(simple_mesh):
    lon = np.full((3, 2), 100.0)
    assert simple_mesh.cell_of(lon, np.zeros((3, 2))).shape == (3, 2)


# ------------------------------------------------------------------ the cache


def test_chunked_build_matches_the_in_memory_build(tmp_path):
    """The cache is built a chunk at a time; it must equal the direct build.

    Two independent code paths produce the same geometry -- one holding whole
    arrays via xarray, one streaming through netCDF4 into mapped .npy. This is
    the test that keeps them honest.
    """
    rng = np.random.default_rng(1)
    centres = np.stack([rng.uniform(-179, 179, 400),
                        rng.uniform(-85, 85, 400)], axis=-1)
    path = write_mesh(tmp_path / "many.nc", centres,
                      n_verts=rng.integers(4, 7, 400))

    direct = MpasMesh.load(path, use_cache=False)
    from gmpas.mesh import BUILD_CHUNK, _build_to_dir, cache_path
    assert len(centres) < BUILD_CHUNK        # so force several chunks below

    cache = cache_path(path)
    _build_to_dir(path, cache, chunk=37)     # deliberately awkward chunk size
    chunked = MpasMesh._mapped(path, cache)

    for name in ("lon_cell", "lat_cell", "cell_verts", "cell_wrapped",
                 "edge_segs", "edge_wrapped", "lon_edge", "lat_edge",
                 "angle_edge", "area_cell", "xyz_cell"):
        a, b = getattr(direct, name), np.asarray(getattr(chunked, name))
        assert a.dtype == b.dtype, name
        assert np.array_equal(a, b, equal_nan=True), name

    assert chunked.sphere_radius == pytest.approx(direct.sphere_radius)
    assert chunked.extent == pytest.approx(direct.extent)
    assert chunked.coverage == pytest.approx(direct.coverage)


def test_a_chunk_boundary_does_not_split_a_polygon(tmp_path):
    """Every row is independent, so any chunk size must give the same answer."""
    from gmpas.mesh import _build_to_dir, cache_path

    centres = [(x * 3.0, 0.0) for x in range(40)]
    path = write_mesh(tmp_path / "row.nc", centres)

    results = []
    for chunk in (1, 7, 40, 1000):
        cache = cache_path(path)
        _build_to_dir(path, cache, chunk=chunk)
        results.append(np.asarray(MpasMesh._mapped(path, cache).cell_verts))

    for other in results[1:]:
        assert np.array_equal(results[0], other)


def test_cached_arrays_are_memory_mapped_not_read(simple_mesh_file):
    """The whole point: opening must not pull the arrays into memory."""
    mesh = MpasMesh.load(simple_mesh_file)

    assert isinstance(mesh.cell_verts, np.memmap)
    assert isinstance(mesh.xyz_cell, np.memmap)


def test_extent_comes_from_metadata_not_from_the_big_array(simple_mesh_file):
    """Reducing over cell_verts to learn four numbers would page in the lot."""
    mesh = MpasMesh.load(simple_mesh_file)

    assert "extent" in mesh._meta
    assert "coverage" in mesh._meta
    assert mesh.extent == pytest.approx(tuple(mesh._meta["extent"]))


def test_an_interrupted_build_leaves_no_usable_cache(tmp_path, monkeypatch):
    """A half-written cache that looks complete is worse than none."""
    from gmpas import mesh as mesh_mod

    path = write_mesh(tmp_path / "boom.nc", [(0.0, 0.0), (10.0, 0.0)])
    cache = mesh_mod.cache_path(path)

    def explode(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(mesh_mod, "_open_memmap", explode)
    with pytest.raises(RuntimeError):
        MpasMesh.load(path)

    assert not (cache / "meta.json").exists()

    monkeypatch.undo()
    assert MpasMesh.load(path).n_cells == 2      # recovers on the next try


def test_second_load_comes_from_cache_and_matches(simple_mesh_file):
    built = MpasMesh.load(simple_mesh_file, use_cache=False)
    assert not cache_path(simple_mesh_file).exists()

    first = MpasMesh.load(simple_mesh_file)
    assert cache_path(simple_mesh_file).exists()
    cached = MpasMesh.load(simple_mesh_file)

    for field in ("lon_cell", "lat_cell", "cell_verts", "cell_wrapped",
                  "edge_segs", "edge_wrapped", "lon_edge", "lat_edge",
                  "angle_edge", "area_cell", "xyz_cell"):
        assert getattr(cached, field) == pytest.approx(getattr(built, field))
        assert getattr(cached, field) == pytest.approx(getattr(first, field))


def test_cache_is_keyed_on_file_identity_not_just_name(tmp_path):
    """Same filename, different contents -- the second must not reuse the first.

    Regression test. Size and mtime alone collided here: netCDF4 padding makes
    both of these files 18436 bytes, and whole-second mtime cannot separate
    them, so the 3-cell mesh silently loaded the 1-cell geometry.
    """
    path = tmp_path / "mesh.nc"
    write_mesh(path, [(0.0, 0.0)])
    first = MpasMesh.load(path)

    path.unlink()
    write_mesh(path, [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)])
    second = MpasMesh.load(path)

    assert first.n_cells == 1
    assert second.n_cells == 3


def test_same_size_and_second_still_get_different_cache_entries(tmp_path):
    """The exact collision that made the cache return the wrong mesh."""
    a, b = tmp_path / "a.nc", tmp_path / "b.nc"
    write_mesh(a, [(0.0, 0.0)])
    write_mesh(b, [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0)])

    assert a.stat().st_size == b.stat().st_size          # netCDF padding
    assert cache_path(a).name != cache_path(b).name


def test_use_cache_false_neither_reads_nor_writes(simple_mesh_file):
    MpasMesh.load(simple_mesh_file, use_cache=False)
    assert not cache_path(simple_mesh_file).exists()


def test_missing_mesh_file_is_reported_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="No such mesh file"):
        MpasMesh.load(tmp_path / "absent.nc")


def test_file_without_mesh_variables_names_what_is_missing(tmp_path):
    import xarray as xr

    path = tmp_path / "diag.nc"
    xr.Dataset({"mslp": ("nCells", np.zeros(4))}).to_netcdf(path)

    with pytest.raises(KeyError, match="verticesOnCell"):
        MpasMesh.load(path)


# ------------------------------------------------------- wind reconstruction


def test_edge_normal_reconstruction_recovers_direction_but_halves_magnitude(tmp_path):
    """The documented approximation, pinned down.

    For uniform eastward flow U the edge-normal component is U*cos(angleEdge).
    Averaging u*(cos, sin) over six normals evenly spaced across [0, pi) gives
    U * mean(cos^2) = U/2 zonal and U * mean(cos*sin) = 0 meridional. The
    direction is right; the magnitude is not, because this is an unweighted
    average and not MPAS's RBF reconstruction.
    """
    path = write_mesh(tmp_path / "wind.nc", [(0.0, 0.0), (10.0, 0.0)])
    mesh = MpasMesh.load(path)

    speed = 20.0
    assert mesh.angle_edge == pytest.approx(np.tile(EDGE_ANGLES, mesh.n_cells))
    u_edge = speed * np.cos(mesh.angle_edge)
    zonal, meridional = reconstruct_cell_winds(mesh, u_edge)

    assert zonal == pytest.approx(speed / 2.0)
    assert meridional == pytest.approx(0.0, abs=1e-12)


def test_reconstruction_is_zero_for_zero_flow(simple_mesh):
    zonal, meridional = reconstruct_cell_winds(simple_mesh,
                                               np.zeros(simple_mesh.n_edges))
    assert zonal == pytest.approx(0.0)
    assert meridional == pytest.approx(0.0)
