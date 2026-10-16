"""Faults a review of the viewer on real files turned up, each reproduced.

One file because they share nothing but their origin: a mesh in a
classic-format file, a click on a missing value, a click off a regional mesh,
a lat/lon region stored east of the dateline in 0-360, and a section that
leaves a mesh carrying its own heights.
"""

from __future__ import annotations

import json
import threading
import urllib.request

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from conftest import write_mesh
from gmpas import MpasMesh, native
from gmpas.generic import GenericViewer
from gmpas.series import Series
from gmpas.viewer import PAGE, Viewer, _handler, bind

# a regional mesh: a 9 x 9 patch of cells one degree apart
PATCH = [(100.0 + i, float(j)) for i in range(9) for j in range(-4, 5)]


def _strict(body: bytes):
    """Parsed as a browser parses it: NaN and Infinity are not JSON."""
    def refuse(name):
        raise ValueError(f"{name} is not JSON")
    return json.loads(body, parse_constant=refuse)


def _serve(viewer):
    srv = bind(_handler(viewer, PAGE), 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def _get(url):
    with urllib.request.urlopen(url) as r:
        return r.read()


# ------------------------------------------------ a classic-format mesh file

@pytest.mark.parametrize("fmt", ["NETCDF3_CLASSIC", "NETCDF3_64BIT_OFFSET",
                                 "NETCDF3_64BIT_DATA"])
def test_a_mesh_in_a_classic_format_file_builds(tmp_path, fmt):
    src = write_mesh(tmp_path / "hdf5.mesh.nc", PATCH)
    with xr.open_dataset(src) as ds:
        ds.load().to_netcdf(tmp_path / "classic.mesh.nc", format=fmt, engine="netcdf4")
    want = MpasMesh.load(src)
    got = MpasMesh.load(tmp_path / "classic.mesh.nc")
    assert got.n_cells == want.n_cells == len(PATCH)
    np.testing.assert_array_equal(got.lon_cell, want.lon_cell)
    np.testing.assert_array_equal(got.lat_cell, want.lat_cell)


# ------------------------------------------------- lat/lon: a missing value

@pytest.fixture
def masked(tmp_path):
    """A global grid whose western half is missing, as land is in an SST file."""
    lon, lat = np.arange(0.0, 360.0, 10.0), np.arange(-80.0, 81.0, 10.0)
    sst = np.broadcast_to(lon[None, :] + 0.0, (2, lat.size, lon.size)).copy()
    sst[:, :, lon < 180] = np.nan
    path = tmp_path / "sst.nc"
    xr.Dataset({"sst": (("time", "lat", "lon"), sst)},
               coords={"time": pd.date_range("2024-01-01", periods=2),
                       "lat": ("lat", lat, {"units": "degrees_north"}),
                       "lon": ("lon", lon, {"units": "degrees_east"})}).to_netcdf(path)
    gv = GenericViewer(path, strict=True)
    srv, url = _serve(gv)
    yield url
    srv.shutdown()
    gv.close()


def test_a_probe_of_a_missing_value_is_valid_json(masked):
    out = _strict(_get(masked + "/api/probe?lon=40&lat=0&var=sst&time=0"))
    assert out["value"] is None and out["cell"] >= 0
    sea = _strict(_get(masked + "/api/probe?lon=200&lat=0&var=sst&time=0"))
    assert sea["value"] == 200.0


# ------------------------------------------------------ lat/lon: off the grid

def _epac(tmp_path, lon):
    """A region of the east Pacific; value = the cell's own longitude."""
    lat = np.arange(-20.0, 21.0, 2.0)
    data = np.broadcast_to(lon[None, None, :], (1, lat.size, lon.size)).copy()
    path = tmp_path / "epac.nc"
    xr.Dataset({"v": (("time", "lat", "lon"), data)},
               coords={"time": pd.date_range("2024-01-01", periods=1),
                       "lat": ("lat", lat, {"units": "degrees_north"}),
                       "lon": ("lon", lon, {"units": "degrees_east"})}).to_netcdf(path)
    return GenericViewer(path, strict=True)


def test_a_probe_off_a_regional_grid_is_valid_json(tmp_path):
    gv = _epac(tmp_path, np.arange(200.0, 251.0, 2.0))
    srv, url = _serve(gv)
    try:
        out = _strict(_get(url + "/api/probe?lon=20&lat=0&var=v&time=0"))
        assert out["cell"] == -1 and out["value"] is None
    finally:
        srv.shutdown()
        gv.close()


@pytest.mark.parametrize("name", ["Mercator", "Orthographic", "Robinson",
                                  "LambertConformal"])
@pytest.mark.parametrize("lon0", [200.0, 160.0, -160.0])
def test_a_region_east_of_the_dateline_draws_in_a_projection(tmp_path, name, lon0):
    """The file says 200..250; a projection says -160..-110. Same place."""
    gv = _epac(tmp_path, np.arange(lon0, lon0 + 51.0, 2.0))
    try:
        home = gv.projected_home(name)
        params = {k: v for k, v in (("central_longitude", home["plon"]),
                                    ("central_latitude", home["plat"]),
                                    ("standard_parallels", home["psp"])) if v is not None}
        img, on = gv._raster_projected("v", 0, 0, home["extent"], 240, 160,
                                       (home["proj"], params))
        assert on.mean() > 0.2, "the region is on the map"
        want = np.mod(img[on] - lon0, 360.0)
        assert want.min() >= 0 and want.max() <= 50.0
        assert np.unique(img[on]).size >= 20, "every column of the region is drawn"
    finally:
        gv.close()


def test_a_point_given_either_way_round_finds_the_same_cell(tmp_path):
    gv = _epac(tmp_path, np.arange(200.0, 251.0, 2.0))
    try:
        assert gv.probe(-136.0, 0.0, "v", 0, 0) == gv.probe(224.0, 0.0, "v", 0, 0)
        assert gv.probe(224.0, 0.0, "v", 0, 0)["value"] == 224.0
    finally:
        gv.close()


# -------------------------------------------------- MPAS: off a regional mesh

@pytest.fixture
def patch_run(tmp_path):
    """The patch as a run, with heights of its own (zgrid) on model levels."""
    write_mesh(tmp_path / "mesh.nc", PATCH)
    with xr.open_dataset(tmp_path / "mesh.nc") as ds:
        full = ds.load()
    (tmp_path / "mesh.nc").unlink()
    n, nlev = full.sizes["nCells"], 4
    code = 100.0 * np.arange(nlev)[None, :] + np.arange(n)[:, None]
    full["theta"] = (("Time", "nCells", "nVertLevels"), code[None])
    full["t2m"] = (("Time", "nCells"), np.arange(n, dtype=float)[None])
    full["zgrid"] = (("nCells", "nVertLevelsP1"),
                     1000.0 * np.arange(nlev + 1)[None, :] + np.arange(n)[:, None],
                     {"units": "m"})
    folder = tmp_path / "run"
    folder.mkdir()
    full.to_netcdf(folder / "history.2012-02-01_00.00.00.nc")
    return folder


def test_a_click_off_a_regional_mesh_is_off_it(patch_run):
    v = Viewer(patch_run, nx=300, ny=200)
    srv, url = _serve(v)
    try:
        inside = _strict(_get(url + "/api/probe?lon=104&lat=0&var=t2m&time=0"))
        assert inside["cell"] >= 0 and inside["value"] is not None
        # 40 degrees from the patch's edge, where the map draws nothing
        out = _strict(_get(url + "/api/probe?lon=60&lat=0&var=t2m&time=0"))
        assert out["cell"] == -1 and out["value"] is None
        with pytest.raises(urllib.error.HTTPError) as err:
            _get(url + "/api/profile?lon=60&lat=0&var=theta&time=0")
        assert "outside the mesh" in err.value.read().decode()
    finally:
        srv.shutdown()
        v.close()


def test_a_section_leaving_a_mesh_with_its_own_heights_draws(patch_run):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = Series(patch_run)
    try:
        da = native.cross_section(s, "theta", 0, (90.0, 0.0), (120.0, 0.0))
        off = da["cell"].values < 0
        assert off.any() and (~off).any()
        fig, ax = plt.subplots()
        native.draw_section(ax, da)
        fig.canvas.draw()
        plt.close(fig)
    finally:
        s.close()
