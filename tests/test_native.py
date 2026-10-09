"""Hovmöller on the native, variable-resolution MPAS mesh.

The mesh here is a strip along the equator with a fine half and a coarse
half: 1-degree cells of small area west of 10E, 5-degree cells of large area
east of it. That is the shape that makes the method matter -- an unweighted
mean, or a bin narrower than the coarse cells, gives a different and wrong
answer, and these tests pin that the native Hovmöller gives the right one.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from gmpas import native
from gmpas.series import Series

FINE = [(lon + 0.5, lat) for lon in range(0, 10) for lat in (-0.5, 0.5)]   # 20 cells
COARSE = [(lon, lat) for lon in (12.5, 17.5, 22.5) for lat in (-2.0, 2.0)]  # 6 cells
# hexagon area for the cell spacing, as a real Voronoi mesh has it: the
# default bin width is derived from areaCell, so area and spacing must agree
FINE_AREA = np.sqrt(3) / 2 * (1.0 * 111.2e3) ** 2                          # m^2
COARSE_AREA = np.sqrt(3) / 2 * (5.0 * 111.2e3) ** 2


@pytest.fixture
def run(tmp_path):
    """Three steps, one file each; value = 100*step + (10 if coarse else 1)."""
    from conftest import write_mesh

    centres = FINE + COARSE
    areas = np.r_[np.full(len(FINE), FINE_AREA), np.full(len(COARSE), COARSE_AREA)]
    base = tmp_path / "mesh.nc"
    write_mesh(base, centres, areas=areas)
    mesh = xr.open_dataset(base)
    coarse = np.r_[np.zeros(len(FINE)), np.ones(len(COARSE))].astype(bool)
    folder = tmp_path / "run"
    folder.mkdir()
    for step in range(3):
        ds = mesh.copy(deep=True)
        v = 100.0 * step + np.where(coarse, 10.0, 1.0)
        ds["t2m"] = (("Time", "nCells"), v[None], {"units": "K"})
        ds.to_netcdf(folder / f"history.2012-02-{step + 1:02d}_00.00.00.nc")
    mesh.close()
    s = Series(folder)
    yield s
    s.close()


def test_the_default_bin_is_as_wide_as_the_widest_cell(run):
    """No bin may be narrower than the coarse cells, or it is empty there for
    lack of resolution, not for lack of data."""
    plan = native.hovmoller_plan(run.mesh, (-5, 5))
    # a cell's east-west extent in degrees of longitude grows with 1/cos(lat)
    lat = np.radians(np.asarray(run.mesh.lat_cell))
    extent = np.asarray(run.mesh.cell_width_km) / (native.KM_PER_DEG * np.cos(lat))
    assert plan.width == pytest.approx(extent.max(), rel=1e-9)
    assert plan.counts.min() >= 1


def test_each_bin_is_the_area_weighted_mean_of_the_cells_centred_in_it(run):
    """Brute force: loop over cells, put each in the bin its centre falls in,
    weight by its area. The sparse product must agree exactly."""
    da = native.hovmoller(run, "t2m", band=(-5, 5), lons=(0, 25), width=5.0)
    lon = np.asarray(run.mesh.lon_cell)
    area = np.asarray(run.mesh.area_cell)
    for step in range(3):
        field = run.values("t2m", step=step)
        for j, centre in enumerate(da.lon.values):
            inside = (lon >= centre - 2.5) & (lon < centre + 2.5)
            want = (field[inside] * area[inside]).sum() / area[inside].sum()
            assert da.values[step, j] == pytest.approx(want)


def test_a_count_mean_would_be_wrong(run):
    """The bin straddling 10E holds 10 fine cells and 2 coarse ones. Counting
    cells says the fine value dominates; by area the coarse cells do."""
    da = native.hovmoller(run, "t2m", band=(-5, 5), lons=(5, 15), width=10.0)
    got = float(da.values[0, 0])
    by_count = (10 * 1.0 + 2 * 10.0) / 12
    by_area = ((10 * FINE_AREA * 1.0 + 2 * COARSE_AREA * 10.0)
               / (10 * FINE_AREA + 2 * COARSE_AREA))
    assert got == pytest.approx(by_area)
    assert got != pytest.approx(by_count)


def test_a_bin_with_no_cell_is_nan_never_filled(run):
    """Bins finer than the coarse cells leave gaps there; they stay empty."""
    da = native.hovmoller(run, "t2m", band=(-5, 5), lons=(0, 25), width=1.0)
    assert np.isnan(da.values[0]).any() and da.attrs["empty_bins"] > 0
    assert np.all(np.isnan(da.values[0, da["cells"].values == 0]))
    assert not np.isnan(da.values[0, da["cells"].values > 0]).any()


def test_nan_cells_are_skipped_and_the_weights_renormalised(run):
    plan = native.hovmoller_plan(run.mesh, (-5, 5), lons=(5, 15), width=10.0)
    v = np.where(np.isin(plan.cells, np.arange(len(FINE))), 1.0, 10.0)
    v[np.asarray(run.mesh.lon_cell)[plan.cells] > 10] = np.nan      # coarse cells gone
    assert native.hovmoller_row(v, plan)[0] == pytest.approx(1.0)


def test_the_band_takes_cells_by_centre(run):
    """Fine cells sit at +-0.5, coarse ones at +-2: a band of +-1 is fine only."""
    plan = native.hovmoller_plan(run.mesh, (-1, 1))
    assert set(np.asarray(run.mesh.lon_cell)[plan.cells]) <= {x + 0.5 for x in range(10)}


def test_time_runs_down_the_rows_with_dates(run):
    da = native.hovmoller(run, "t2m", band=(-5, 5), lons=(0, 25), width=5.0)
    assert da.dims == ("time", "lon") and da.sizes["time"] == 3
    assert str(da.time.values[1])[:10] == "2012-02-02"
    assert da.values[2, 0] - da.values[0, 0] == pytest.approx(200.0)


def test_a_regional_mesh_across_the_antimeridian_is_one_range(tmp_path):
    from conftest import write_mesh
    from gmpas.mesh import MpasMesh

    write_mesh(tmp_path / "m.nc", [(170.0, 0.0), (175.0, 0.0), (-175.0, 0.0),
                                   (-170.0, 0.0)])
    plan = native.hovmoller_plan(MpasMesh.load(tmp_path / "m.nc"), (-1, 1))
    assert plan.lon_bounds[0] == pytest.approx(170.0)
    assert plan.lon_bounds[-1] < 200.0          # not the whole circle


@pytest.mark.parametrize("kwargs, why", [
    ({"band": (5, -5)}, "must not decrease"),
    ({"band": (40, 50)}, "no cell centre"),
    ({"band": (-5, 5), "lons": (10, 5)}, "must increase"),
    ({"band": (-5, 5), "width": 0}, "must be positive"),
])
def test_a_bad_request_is_refused_by_name(run, kwargs, why):
    with pytest.raises(ValueError, match=why):
        native.hovmoller_plan(run.mesh, **kwargs)


def test_cancelling_stops_the_read(run):
    import threading

    from gmpas.jobs import Cancelled

    stop = threading.Event()
    stop.set()
    with pytest.raises(Cancelled):
        native.hovmoller(run, "t2m", band=(-5, 5), cancel=stop)


# ------------------------------------------------------------------ the viewer

@pytest.fixture
def viewer(run):
    from gmpas.viewer import Viewer

    folder = run.files[0].parent
    v = Viewer(folder, nx=60, ny=40)
    deadline = __import__("time").time() + 10
    while v.series.scanning and __import__("time").time() < deadline:
        __import__("time").sleep(0.02)
    yield v
    v.close()


def test_a_time_varying_field_offers_the_hovmoller(viewer):
    rows = {r["name"]: r for r in viewer.describe()["variables"]}
    assert rows["t2m"]["kinds"] == ["map", "hovmoller"]
    assert "kinds" not in rows["areaCell"]                     # static: no time axis


def test_the_mpas_kind_table_matches_the_generic_one():
    """One page serves both viewers; a kind both offer must mean the same thing
    in each. The section is MPAS only, and a figure like the Hovmöller."""
    from gmpas import generic, viewer

    shared = set(viewer.KIND_CAPS) & set(generic.KIND_CAPS)
    assert shared == {"map", "hovmoller"}
    for kind in shared:
        assert viewer.KIND_CAPS[kind] == generic.KIND_CAPS[kind]
    assert viewer.KIND_CAPS["section"] == viewer.KIND_CAPS["hovmoller"]


def test_the_hovmoller_travels_over_http(viewer):
    """202 while it reads, then the PNG with every step's row for the marker."""
    import json
    import threading
    import time
    import urllib.request

    from gmpas.viewer import PAGE, _handler, bind

    srv = bind(_handler(viewer, PAGE), 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = (f"http://127.0.0.1:{srv.server_address[1]}/api/plot?var=t2m&kind=hovmoller"
           f"&time=0&level=0&extent=0,25,-5,5&hlat0=-5&hlat1=5&w=600&h=400")
    try:
        for _ in range(200):
            r = urllib.request.urlopen(url)
            if r.status == 200:
                break
            assert json.loads(r.read())["state"] == "running"
            time.sleep(0.05)
        assert r.headers["Content-Type"] == "image/png" and r.read()[:4] == b"\x89PNG"
        assert len(r.headers["X-Hov-Y"].split(",")) == 3
    finally:
        srv.shutdown()


def test_the_hovmoller_exports_as_netcdf_and_figure(viewer, tmp_path):
    out = tmp_path / "hov.nc"          # read back from disk, as test_hovmoller does
    out.write_bytes(viewer.netcdf("t2m", 0, 0, (0, 25, -5, 5), 60, 40,
                                  kind="hovmoller", hov={"band": (-5, 5)}))
    da = xr.open_dataarray(out)
    assert da.dims == ("time", "lon") and "area" in da.attrs["weighting"]
    png = viewer.figure("t2m", 0, 0, (0, 25, -5, 5), "viridis", None, None,
                        kind="hovmoller", hov={"band": (-5, 5)})
    assert png[:4] == b"\x89PNG"


def test_a_static_field_has_no_hovmoller(viewer):
    with pytest.raises(ValueError, match="no time axis"):
        viewer.hovmoller_progress("areaCell", 0, {"band": (-5, 5)}, viewer.home)


# --------------------------------------------------------- vertical section

KM = 111.2
SFINE = [(0.25 + 0.5 * i, -3.75 + 0.5 * j) for i in range(20) for j in range(16)]  # 0.5 deg
SCOARSE = [(11.0 + 2.0 * i, -3.0 + 2.0 * j) for i in range(10) for j in range(4)]   # 2 deg
PLEV = [50000.0, 70000.0, 85000.0]                                                 # Pa


@pytest.fixture
def sec_run(tmp_path):
    """A fine half and a coarse half, areas matching the spacing (a hexagon of
    width w has area sqrt(3)/2 w^2), with isobaric, model-level, flat and edge
    fields. Value = 100 * level + cell index, so a read names its cell."""
    from conftest import write_mesh

    centres = SFINE + SCOARSE
    w = np.r_[np.full(len(SFINE), 0.5 * KM), np.full(len(SCOARSE), 2.0 * KM)] * 1000
    write_mesh(tmp_path / "mesh.nc", centres, areas=np.sqrt(3) / 2 * w ** 2)
    ds = xr.open_dataset(tmp_path / "mesh.nc").load()
    n = ds.sizes["nCells"]
    code = 100.0 * np.arange(3)[None, :] + np.arange(n)[:, None]
    ds["t_isobaric"] = (("Time", "nCells", "nIsoLevelsT"), code[None], {"units": "K"})
    ds["t_iso_levels"] = (("nIsoLevelsT",), PLEV, {"units": "Pa"})
    ds["theta"] = (("Time", "nCells", "nVertLevels"), code[None])
    ds["t2m"] = (("Time", "nCells"), np.arange(n, dtype=float)[None])
    ds["u"] = (("Time", "nEdges"), np.zeros((1, ds.sizes["nEdges"])))
    folder = tmp_path / "run"
    folder.mkdir()
    ds.to_netcdf(folder / "history.2012-02-01_00.00.00.nc")
    s = Series(folder)
    yield s
    s.close()


def _dense_cells(mesh, p0, p1, n=2_000_000):
    """Brute force: the same path sampled at ~1-2 km, runs collapsed."""
    pts, _, _ = native._great_circle(p0, p1, n)
    ids = native._locate(mesh, pts)
    return ids[np.r_[True, ids[1:] != ids[:-1]]]


@pytest.mark.parametrize("p0, p1", [
    ((0.0, 0.0), (29.0, 0.5)),              # fine into coarse along the strip
    ((1.0, -3.5), (25.0, 3.0)),             # diagonal across both
    ((9.0, -3.0), (13.0, 2.5)),             # short, across the transition
])
def test_the_section_crosses_exactly_the_cells_a_dense_sampling_does(sec_run, p0, p1):
    sec = native.section_cells(sec_run.mesh, p0, p1)
    dense = _dense_cells(sec_run.mesh, p0, p1)
    # every cell dense sampling meets is there, in the same order...
    got = iter(sec.cells.tolist())
    assert all(c in got for c in dense.tolist())
    # ...and any it stepped over is a corner thinner than its own spacing
    spacing = sec.length / 2_000_000
    extra = ~np.isin(sec.cells, dense)
    assert np.all((sec.x1 - sec.x0)[extra] < 2 * spacing)
    on = sec.cells[sec.cells >= 0]
    assert on.size == np.unique(on).size                    # none repeated
    assert np.all(sec.cells[1:] != sec.cells[:-1])           # runs merged
    assert sec.x0[0] == 0 and sec.x1[-1] == pytest.approx(sec.length)
    np.testing.assert_allclose(sec.x0[1:], sec.x1[:-1])      # columns tile the path
    assert (sec.x1 - sec.x0).sum() == pytest.approx(sec.length)


def test_columns_are_as_wide_as_the_cells_they_cross(sec_run):
    """Variable resolution shows: narrow columns in the fine half, wide in the coarse."""
    sec = native.section_cells(sec_run.mesh, (0.0, 0.1), (29.0, 0.1))
    width = sec.x1 - sec.x0
    fine, coarse = width[sec.lon < 9.5], width[sec.lon > 12]
    assert np.median(fine) == pytest.approx(0.5 * KM, rel=0.15)
    assert np.median(coarse) == pytest.approx(2.0 * KM, rel=0.15)


def test_the_values_are_the_cells_own_on_the_pressure_axis_of_the_file(sec_run):
    da = native.cross_section(sec_run, "t_isobaric", 0, (0.0, 0.1), (29.0, 0.1))
    cells = da["cell"].values
    assert (cells >= 0).all()
    for k in range(3):
        np.testing.assert_array_equal(da.values[k],
                                      sec_run.values("t_isobaric", 0, level=k)[cells])
    assert da["level"].values.tolist() == [500.0, 700.0, 850.0]
    assert da["level"].attrs["units"] == "hPa" and da.attrs["pressure_down"] == 1


def test_a_level_axis_without_coordinates_is_labelled_as_an_index(sec_run):
    da = native.cross_section(sec_run, "theta", 0, (0.0, 0.1), (29.0, 0.1))
    assert da["level"].attrs["long_name"] == "model level"
    assert da["level"].values.tolist() == [0.0, 1.0, 2.0]


@pytest.mark.parametrize("var, why", [
    ("u", "cell fields only"),
    ("t2m", "no vertical axis"),
])
def test_a_field_without_a_section_is_refused_by_name(sec_run, var, why):
    with pytest.raises(ValueError, match=why):
        native.cross_section(sec_run, var, 0, (0.0, 0.0), (20.0, 0.0))


def test_a_path_leaving_a_regional_mesh_leaves_blank_columns(sec_run):
    da = native.cross_section(sec_run, "t_isobaric", 0, (-15.0, 0.1), (29.0, 0.1))
    off = da["cell"].values < 0
    assert off.any() and np.isnan(da.values[:, off]).all()
    assert not np.isnan(da.values[:, ~off]).any()


def test_the_section_draws_and_travels_over_http(sec_run, tmp_path):
    """describe offers it for multi-level fields; /api/plot draws it on the path
    asked for; netCDF export carries the native columns."""
    import threading
    import urllib.request

    from gmpas.viewer import PAGE, Viewer, _handler, bind

    v = Viewer(sec_run.files[0].parent, nx=60, ny=40)
    try:
        rows = {r["name"]: r for r in v.describe()["variables"]}
        assert "section" in rows["t_isobaric"]["kinds"]
        assert "section" not in rows["t2m"]["kinds"]
        srv = bind(_handler(v, PAGE), 0)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        url = (f"http://127.0.0.1:{srv.server_address[1]}/api/plot?var=t_isobaric"
               f"&kind=section&time=0&level=0&extent=0,30,-4,4&w=600&h=300"
               f"&s0lon=0&s0lat=0.1&s1lon=29&s1lat=0.1")
        try:
            assert urllib.request.urlopen(url).read()[:4] == b"\x89PNG"
        finally:
            srv.shutdown()
        out = tmp_path / "sec.nc"
        out.write_bytes(v.netcdf("t_isobaric", 0, 0, (0, 30, -4, 4), 60, 40,
                                 kind="section", sec=((0, 0.1), (29, 0.1))))
        back = xr.open_dataarray(out)
        assert back.dims == ("level", "distance") and "native" in back.attrs["method"]
    finally:
        v.close()
