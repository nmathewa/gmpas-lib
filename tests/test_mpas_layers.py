"""A figure of layers on an MPAS run: fields and winds over one another.

Each layer is the field sampled onto the view's pixels as the fast map samples
it -- the cell under each pixel -- and drawn by `gmpas.layers` on that grid.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import xarray as xr

from gmpas import script

pytest.importorskip("cartopy", reason="layers are drawn on a map")


@pytest.fixture(scope="module")
def viewer(tmp_path_factory):
    from conftest import EARTH_RADIUS, write_mesh
    from gmpas.viewer import Viewer

    folder = tmp_path_factory.mktemp("run")
    rng = np.random.default_rng(0)
    lon = rng.uniform(-180, 180, 500)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, 500)))
    mesh = folder / "mesh.nc"
    # cells wide enough to meet, so the map is drawn everywhere
    areas = np.full(500, 0.85 * 4 * np.pi * EARTH_RADIUS ** 2 / 500)
    write_mesh(mesh, list(zip(lon, lat, strict=True)), areas=areas)
    with xr.open_dataset(mesh) as ds:
        full = ds.load()
    mesh.unlink()
    n = full.sizes["nCells"]
    full["sst"] = (("Time", "nCells"), np.degrees(full["latCell"].values)[None])
    full["skintemp"] = (("Time", "nCells"), np.degrees(full["lonCell"].values)[None])
    full["u10"] = (("Time", "nCells"), np.ones((1, n)))
    full["v10"] = (("Time", "nCells"), np.zeros((1, n)))
    full.to_netcdf(folder / "history.2012-01-01_00.00.00.nc")
    v = Viewer(folder, nx=300, ny=150)
    yield v
    v.close()


def test_a_run_says_it_draws_layers(viewer):
    meta = viewer.describe()
    sst = next(v for v in meta["variables"] if v["name"] == "sst")
    assert "layers" in sst["kinds"] and sst["spatial"]
    assert "quiver" in meta["layer_schema"]["kinds"]
    assert meta["kind_caps"]["layers"]["colour"] is False


def test_a_layer_is_the_cell_under_each_pixel(viewer):
    from gmpas.viewer import _MeshOnGrid

    extent = (-60.0, 60.0, -30.0, 30.0)
    grid = _MeshOnGrid(viewer, extent, 400, 200)
    da = grid._cropped("sst", 0, 0, extent)
    assert da.dims == ("lat", "lon") and da.shape == (grid.ny, grid.nx)
    assert np.all(np.diff(da["lat"].values) > 0)
    # sst holds each cell's own latitude: the rows run south to north with it,
    # and every pixel is the value of the cell the mesh finds there
    lon, lat = np.meshgrid(da["lon"].values, da["lat"].values)
    cells = viewer.mesh.cell_of(lon.ravel(), lat.ravel())
    want = np.asarray(viewer.mesh.lat_cell)[cells].reshape(da.shape)
    shown = np.isfinite(da.values)
    assert shown.mean() > 0.5
    np.testing.assert_allclose(da.values[shown], want[shown])


def test_contours_and_winds_over_a_field_draw(viewer):
    from PIL import Image

    state = script.resolve(script.parse(
        "contourf(sst)\ncontour(skintemp, colors='black')\nvectors(u10, v10)"),
        viewer.describe())
    assert state["kind"] == "layers"
    png = viewer.plot("sst", 0, 0, "layers", (-60, 60, -30, 30), 640, 400,
                      layers=state["layers"])
    image = Image.open(io.BytesIO(png))
    assert image.size == (640, 400)
    assert len(image.convert("RGB").getcolors(1 << 20)) > 20      # not a blank page


def test_a_layer_the_run_does_not_have_is_refused(viewer):
    with pytest.raises(script.ScriptError, match="layer 2 \\(contour\\): var='slp'"):
        script.resolve(script.parse("contourf(sst)\ncontour(slp)"), viewer.describe())
    with pytest.raises(ValueError, match="not a map variable"):
        viewer.plot("sst", 0, 0, "layers", (-60, 60, -30, 30),
                    layers={"figure": {}, "layers": [{"kind": "contour", "var": "nope"}]})
