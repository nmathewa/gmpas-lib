"""Writing a mesh as SCRIP, the handoff to a real conservative remapper."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from conftest import write_mesh
from gmpas.scrip import TWO_PI, coverage_of, write_scrip


def test_it_writes_what_scrip_requires(tmp_path):
    path = write_mesh(tmp_path / "m.nc", [(0.0, 0.0), (10.0, 0.0), (20.0, 5.0)])
    out, _ = write_scrip(path, tmp_path / "m.scrip.nc")

    with xr.open_dataset(out) as ds:
        assert set(ds.data_vars) >= {
            "grid_dims", "grid_center_lat", "grid_center_lon",
            "grid_corner_lat", "grid_corner_lon", "grid_area", "grid_imask",
        }
        assert ds.sizes["grid_size"] == 3
        assert ds.grid_center_lat.units == "radians"
        assert ds.grid_area.units == "radian^2"
        assert (ds.grid_imask.values == 1).all()


def test_corners_are_trimmed_to_the_widest_real_cell(tmp_path):
    """Not to the declared maxEdges.

    A mesh declaring maxEdges=6 but using at most 5 would otherwise carry a
    column of pure padding on every cell -- and ESMF 8.9.1 segfaults on
    `-m conserve` when a source is padded that way.
    """
    path = write_mesh(tmp_path / "m.nc", [(0.0, 0.0), (10.0, 0.0)],
                      n_verts=[5, 4])
    out, _ = write_scrip(path, tmp_path / "m.scrip.nc")

    with xr.open_dataset(path) as src:
        assert src.sizes["maxEdges"] == 6           # what the file declares
    with xr.open_dataset(out) as ds:
        assert ds.sizes["grid_corners"] == 5        # what is actually used


def test_a_short_cell_repeats_its_last_corner(tmp_path):
    """SCRIP's own convention for variable-sided cells."""
    path = write_mesh(tmp_path / "m.nc", [(0.0, 0.0), (10.0, 0.0)],
                      n_verts=[6, 4])
    out, _ = write_scrip(path, tmp_path / "m.scrip.nc")

    with xr.open_dataset(out) as ds:
        lon = ds.grid_corner_lon.values
        assert lon[1, 4] == lon[1, 3]               # the 4-sided cell pads
        assert lon[1, 5] == lon[1, 3]
        assert lon[0, 5] != lon[0, 4]               # the 6-sided one does not


def test_longitudes_are_normalised_and_the_count_reported(tmp_path):
    """Real MPAS files mix conventions, and silence would be the wrong answer.

    The mesh this was built against stores lonCell on [0, 2pi) but lonVertex
    on [-pi, pi), so cells near the dateline had their centre and their own
    corners on different branches.
    """
    import netCDF4

    # straddling the dateline, so some vertices land past pi
    path = write_mesh(tmp_path / "m.nc", [(179.0, 0.0), (-179.0, 0.0)])

    # the fixture writes tidy [0, 2pi) longitudes, so put the real file's
    # inconsistency in by hand: vertices on [-pi, pi), centres left alone
    with netCDF4.Dataset(path, "a") as nc:
        nc.set_auto_mask(False)
        lon = nc.variables["lonVertex"][:]
        nc.variables["lonVertex"][:] = np.where(lon > np.pi, lon - TWO_PI, lon)

    out, wrapped = write_scrip(path, tmp_path / "m.scrip.nc")

    assert wrapped > 0                              # this mesh needed it
    with xr.open_dataset(out) as ds:
        for name in ("grid_center_lon", "grid_corner_lon"):
            v = ds[name].values
            assert (v >= 0.0).all() and (v < TWO_PI).all(), name


def test_an_already_normalised_mesh_reports_no_wrapping(tmp_path):
    path = write_mesh(tmp_path / "m.nc", [(10.0, 0.0), (20.0, 0.0)])
    _, wrapped = write_scrip(path, tmp_path / "m.scrip.nc")
    assert wrapped == 0


def test_grid_area_is_a_solid_angle(tmp_path):
    """areaCell is m^2; SCRIP wants steradians, so divide by the radius."""
    from gmpas.mesh import EARTH_RADIUS

    sphere = 4.0 * np.pi * EARTH_RADIUS**2
    n = 40
    rng = np.random.default_rng(0)
    centres = np.stack([rng.uniform(-179, 179, n), rng.uniform(-80, 80, n)], -1)
    path = write_mesh(tmp_path / "m.nc", centres, areas=np.full(n, sphere / n))
    out, _ = write_scrip(path, tmp_path / "m.scrip.nc")

    assert coverage_of(out) == pytest.approx(1.0, rel=1e-6)


def test_a_unit_sphere_mesh_still_gives_the_right_solid_angle(tmp_path):
    """A JIGSAW mesh carries sphere_radius=1 and non-dimensional areas."""
    n = 20
    rng = np.random.default_rng(1)
    centres = np.stack([rng.uniform(-179, 179, n), rng.uniform(-80, 80, n)], -1)
    sphere = 4.0 * np.pi * 6_371_229.0**2
    path = write_mesh(tmp_path / "unit.nc", centres, sphere_radius=1.0,
                      areas=np.full(n, sphere / n))
    out, _ = write_scrip(path, tmp_path / "unit.scrip.nc")

    assert coverage_of(out) == pytest.approx(1.0, rel=1e-6)


def test_a_file_without_a_mesh_says_so(tmp_path):
    from conftest import write_diag

    path = write_diag(tmp_path / "diag.nc", n_cells=4, n_edges=24)
    with pytest.raises(KeyError, match="cannot be written as SCRIP"):
        write_scrip(path, tmp_path / "x.scrip.nc")


def test_a_missing_file_is_reported_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="No such mesh file"):
        write_scrip(tmp_path / "absent.nc", tmp_path / "x.scrip.nc")


def test_cells_at_the_antimeridian_stay_local(tmp_path):
    """The seam is where a remap goes from minutes to hours.

    MPAS stores lonVertex on [0, 2pi), so a cell straddling the antimeridian
    arrives with corners at ~359.9 and ~0.1 degrees. Written through, that is
    a polygon spanning nearly the globe -- ESMF finds candidate overlaps from
    bounding boxes, so such a cell is a candidate against very nearly every
    target cell, and a few thousand of them around the seam turn the search
    from roughly N log N into something closer to N x M.
    """
    from conftest import write_mesh
    from gmpas.scrip import write_scrip

    mesh = tmp_path / "seam.nc"
    write_mesh(mesh, [(147.0, -2.0), (359.5, 0.0), (0.5, 0.0), (180.0, 10.0)],
               radius_deg=1.0)
    out, _ = write_scrip(mesh, tmp_path / "s.nc")

    with xr.open_dataset(out) as scrip:
        lon = np.degrees(scrip.grid_corner_lon.values)

    spans = lon.max(axis=1) - lon.min(axis=1)
    # every cell is ~2 degrees across; none may look global
    assert spans.max() < 10.0, f"a cell spans {spans.max():.1f} degrees"


def test_unwrapping_moves_no_point_on_the_sphere(tmp_path):
    """Shifting a corner by a whole turn must be a relabelling, not a move."""
    from conftest import write_mesh
    from gmpas.scrip import write_scrip

    mesh = tmp_path / "seam.nc"
    write_mesh(mesh, [(359.5, 0.0), (0.5, 0.0)], radius_deg=1.0)
    out, _ = write_scrip(mesh, tmp_path / "s.nc")

    with xr.open_dataset(out) as scrip:
        lon = scrip.grid_corner_lon.values
        lat = scrip.grid_corner_lat.values

    # corners may sit outside [0, 2pi) now -- that is the point -- but each
    # must still be the same physical location it was
    assert (lon < 0).any() or (lon >= 2 * np.pi).any()
    xyz = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon),
                    np.sin(lat)], axis=-1)
    wrapped_lon = np.mod(lon, 2 * np.pi)
    xyz_wrapped = np.stack([np.cos(lat) * np.cos(wrapped_lon),
                            np.cos(lat) * np.sin(wrapped_lon),
                            np.sin(lat)], axis=-1)
    assert np.allclose(xyz, xyz_wrapped, atol=1e-12)
