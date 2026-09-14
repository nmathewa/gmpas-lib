"""Hovmöller diagrams (time x longitude) for --generic."""

from __future__ import annotations

import io
import json
import threading
import time

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import gmpas.generic as G
from gmpas.generic import GenericViewer, _hov_columns

LAT = np.arange(30, -30.5, -2.0)            # descending, as ERA5 ships it
LON = np.arange(0, 360, 5.0)
T0 = pd.Timestamp("2024-01-01")


def _write(path, times, field_of, var="t", level_dim=True, extra=None):
    lat2, lon2 = np.meshgrid(LAT, LON, indexing="ij")
    days = ((pd.DatetimeIndex(times) - T0) / pd.Timedelta(days=1)).values
    values = np.stack([field_of(lat2, lon2, d) for d in days]).astype("f8")
    dims = ("valid_time", "latitude", "longitude")
    coords = {"valid_time": times, "latitude": LAT, "longitude": LON}
    if level_dim:
        values = np.stack([values, values + 100.0], axis=1)
        dims = ("valid_time", "pressure_level", "latitude", "longitude")
        coords["pressure_level"] = ("pressure_level", [850.0, 500.0], {"units": "hPa"})
    ds = xr.Dataset({var: (dims, values, {"units": "K", "long_name": "Temperature"})},
                    coords=coords)
    if extra:
        ds = ds.assign(extra)
    ds.to_netcdf(path)


C = 5.0                                     # degrees east per day
K = 2 * 2 * np.pi / 360.0                   # wavenumber 2


def _wave(lat, lon, day):
    return 280 + 10 * np.cos(K * (lon - C * day)) * np.cos(np.deg2rad(lat))


@pytest.fixture
def waves(tmp_path):
    """Three files, 12 six-hourly steps each, of an eastward travelling wave."""
    import matplotlib
    matplotlib.use("Agg")

    for m in range(3):
        times = pd.date_range(T0 + pd.Timedelta(days=3 * m), periods=12, freq="6h")
        _write(tmp_path / f"wave_{m}.nc", times, _wave)
    gv = GenericViewer(tmp_path)
    yield gv
    # a job still reading would race the next test's unlocked file writes in HDF5
    gv.close()


# ------------------------------------------------------------- the columns


def test_a_range_across_the_seam_reads_two_slices_as_one_axis():
    pieces, x = _hov_columns(np.arange(0, 360, 10.0), True, -30, 30)
    assert pieces == [(slice(33, 36), False), (slice(0, 4), False)]
    assert x.tolist() == [-30, -20, -10, 0, 10, 20, 30]


def test_a_repeated_seam_column_is_dropped_for_a_full_turn():
    pieces, x = _hov_columns(np.arange(0, 361, 10.0), True, 0, 360)
    assert pieces == [(slice(0, 36), False)] and x.size == 36


def test_descending_longitudes_are_read_reversed():
    pieces, x = _hov_columns(np.arange(0, 360, 10.0)[::-1], True, 20, 50)
    assert pieces == [(slice(30, 34), True)] and x.tolist() == [20, 30, 40, 50]


def test_a_regional_grid_does_not_wrap():
    pieces, x = _hov_columns(np.arange(-20, 41, 5.0), False, 0, 20)
    assert pieces == [(slice(4, 9), False)] and x.tolist() == [0, 5, 10, 15, 20]
    with pytest.raises(ValueError, match="no grid column"):
        _hov_columns(np.arange(-20, 41, 5.0), False, 100, 120)


# ------------------------------------------------------------ the numbers


def test_the_band_mean_is_cos_latitude_weighted_and_skips_nans(tmp_path):
    rng = np.random.default_rng(3)
    values = rng.normal(size=(2, LAT.size, LON.size))
    values[0, 4:8, 10:20] = np.nan
    values[1, :, 40] = np.nan                            # a column with nothing valid
    xr.Dataset({"q": (("time", "latitude", "longitude"), values)},
               coords={"time": pd.date_range(T0, periods=2), "latitude": LAT,
                       "longitude": LON}).to_netcdf(tmp_path / "q.nc")
    gv = GenericViewer(tmp_path / "q.nc")

    with np.errstate(all="raise"):                       # and no warnings on the way
        h = gv.hovmoller("q", band=(-12, 21), lons=(0, 360))

    rows = (LAT >= -12) & (LAT <= 21)                     # inclusive edges
    sub = values[:, rows, :]
    w = np.cos(np.deg2rad(LAT[rows]))[None, :, None]
    with np.errstate(invalid="ignore"):
        expected = np.nansum(sub * w, axis=1) / np.sum(np.isfinite(sub) * w, axis=1)
    np.testing.assert_allclose(h.values, expected, rtol=0, atol=1e-12)
    assert np.isnan(h.values[1, 40])
    assert h.attrs["weighting"] == "cos(latitude), NaN skipped"
    assert h.attrs["band"] == "latitude -12 to 20 (17 rows)"


def test_a_travelling_wave_keeps_its_phase_speed_across_files(waves):
    h = waves.hovmoller("t", 0, band=(-10, 10), lons=(0, 360))
    assert h.dims == ("valid_time", "lon") and h.shape == (36, 72)
    days = ((pd.DatetimeIndex(h.valid_time.values) - T0) / pd.Timedelta(days=1)).values
    lag = 8                                                # two days, over a file boundary
    # search within half a wavelength (wavenumber 2: 180 degrees), or a shift by
    # one whole wavelength matches as well as the true one
    search = np.arange(-17, 18)
    shifts = []
    for i in range(h.shape[0] - lag):
        later = h.values[i + lag]
        corr = [np.dot(h.values[i], np.roll(later, -s)) for s in search]
        shifts.append(search[int(np.argmax(corr))] * 5.0)    # later[j + s] == now[j]
    speed = np.mean(shifts) / (days[lag] - days[0])
    assert speed == pytest.approx(C)
    assert np.ptp(shifts) == 0


def test_steps_levels_and_a_seam_crossing_range(waves):
    h = waves.hovmoller("t", 1, band=(-10, 10), lons=(300, 420), steps=(10, 25))
    assert h.shape == (16, 25)
    assert h.lon.values[0] == 300 and h.lon.values[-1] == 420
    assert np.all(np.diff(h.lon.values) == 5)
    level0 = waves.hovmoller("t", 0, band=(-10, 10), lons=(300, 420), steps=(10, 25))
    np.testing.assert_allclose(h.values - level0.values, 100.0)
    assert pd.Timestamp(h.valid_time.values[0]) == T0 + pd.Timedelta(hours=60)


def test_chunked_reads_match_one_read(waves, monkeypatch):
    whole = waves.hovmoller("t", 0, band=(-20, 20), lons=(40, 200))
    monkeypatch.setattr(G.GenericViewer, "_hov_chunk_steps", lambda self, *a: 1)
    one_at_a_time = waves.hovmoller("t", 0, band=(-20, 20), lons=(40, 200))
    np.testing.assert_array_equal(whole.values, one_at_a_time.values)


def test_without_a_time_coordinate_the_axis_is_step_numbers(tmp_path):
    for day in ("2024-01-01_00.00.00", "2024-01-02_00.00.00"):
        xr.Dataset({"v": (("lat", "lon"), np.ones((LAT.size, LON.size)))},
                   coords={"lat": LAT, "lon": LON}).to_netcdf(tmp_path / f"s.{day}.nc")
    h = GenericViewer(tmp_path).hovmoller("v", band=(-5, 5))
    assert h.dims == ("step", "lon") and h.step.values.tolist() == [0, 1]
    np.testing.assert_allclose(h.values, 1.0)


@pytest.mark.parametrize("kwargs, message", [
    ({"band": (50, 60)}, "no grid row"),
    ({"band": (10, -10)}, "must not decrease"),
    ({"band": (-10, 10), "steps": (5, 99)}, "outside"),
    ({"band": (-10, 10), "lons": (40, 20)}, "must increase"),
])
def test_bad_requests_are_refused_by_name(waves, kwargs, message):
    with pytest.raises(ValueError, match=message):
        waves.hovmoller("t", 0, **kwargs)


def test_refused_while_the_time_axis_is_being_counted(waves):
    waves.scanning = True
    with pytest.raises(ValueError, match="still being counted"):
        waves.hovmoller("t", 0, band=(-10, 10))


def test_a_result_too_big_to_keep_is_refused_up_front(waves):
    waves._hov_cache.budget = 1024
    with pytest.raises(ValueError, match="narrow the step or longitude range"):
        waves.hovmoller("t", 0, band=(-10, 10))


# ------------------------------------------------------------------ jobs


def _wait_done(gv, *args, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = gv.hovmoller_progress(*args)
        if state["state"] != "running":
            return state
        time.sleep(0.02)
    raise AssertionError("job never finished")


def test_progress_runs_then_is_done_and_bands_selecting_the_same_rows_share_it(waves):
    ext = waves.home
    first = waves.hovmoller_progress("t", 0, {"band": (-10, 10)}, ext)
    assert first["state"] in ("running", "done")
    assert _wait_done(waves, "t", 0, {"band": (-10, 10)}, ext)["state"] == "done"
    # -10.5 selects the same rows on a 2-degree grid: same key, no new job
    assert waves.hovmoller_progress("t", 0, {"band": (-10.5, 10)}, ext)["state"] == "done"


def test_a_failed_job_is_remembered_not_restarted(waves, monkeypatch):
    calls = []

    def broken(*a, **k):
        calls.append(1)
        raise OSError("disk went away")

    monkeypatch.setattr(waves, "hovmoller", broken)
    state = _wait_done(waves, "t", 0, {"band": (-10, 10)}, waves.home)
    assert state["state"] == "error" and "disk went away" in state["error"]
    for _ in range(3):
        state = waves.hovmoller_progress("t", 0, {"band": (-10, 10)}, waves.home)
        assert state["state"] == "error"
    assert len(calls) == 1


def test_a_new_request_cancels_the_running_one(waves, monkeypatch):
    real = G.GenericViewer.hovmoller
    cancelled = []

    def slow(self, *a, cancel=None, **k):
        for _ in range(200):
            if cancel is not None and cancel.is_set():
                cancelled.append(a[1] if len(a) > 1 else None)
                raise G.HovmollerCancelled()
            time.sleep(0.01)
        return real(self, *a, cancel=cancel, **k)

    monkeypatch.setattr(G.GenericViewer, "hovmoller", slow)
    waves.hovmoller_progress("t", 0, {"band": (-10, 10)}, waves.home)
    time.sleep(0.05)
    waves.hovmoller_progress("t", 0, {"band": (0, 20)}, waves.home)
    deadline = time.time() + 5
    while not cancelled and time.time() < deadline:
        time.sleep(0.02)
    assert cancelled


def test_stopping_jobs_leaves_nothing_reading(waves, monkeypatch):
    started = threading.Event()

    def slow(self, *a, cancel=None, **k):
        started.set()
        while not cancel.is_set():
            time.sleep(0.01)
        raise G.HovmollerCancelled()

    monkeypatch.setattr(G.GenericViewer, "hovmoller", slow)
    waves.hovmoller_progress("t", 0, {"band": (-10, 10)}, waves.home)
    assert started.wait(5)
    waves.stop_jobs(timeout=5)
    assert all(job["finished"].is_set() for job in waves._hov_jobs.values())


def test_an_export_waits_for_the_job(waves):
    waves.hovmoller_progress("t", 0, {"band": (-10, 10)}, waves.home)
    da = waves._hov_result("t", 0, {"band": (-10, 10)}, waves.home)
    assert da.shape == (36, 72)


# ------------------------------------------------------------- drawing


def test_it_is_offered_for_map_variables_with_a_time_axis(waves):
    assert "hovmoller" in waves.kinds("t")


@pytest.mark.parametrize("method", ["auto", "contourf", "pcolormesh", "contour"])
def test_every_method_draws_and_reports_where_steps_landed(waves, method):
    from PIL import Image

    meta = {}
    png = waves.plot("t", 0, 0, "hovmoller", waves.home, 600, 400,
                     hov={"band": (-10, 10), "method": method}, meta=meta)
    assert Image.open(io.BytesIO(png)).size == (600, 400)
    assert len(meta["y"]) == 36 and meta["y"][0] < meta["y"][-1]      # time runs down
    left, right, width, height = meta["box"]
    assert 0 < left < right <= width == 600 and height == 400


def test_time_can_run_up(waves):
    meta = {}
    waves.plot("t", 0, 0, "hovmoller", waves.home, 600, 400,
               hov={"band": (-10, 10), "ydir": "up"}, meta=meta)
    assert meta["y"][0] > meta["y"][-1]


def test_auto_draws_big_results_as_an_image_and_explicit_contours_are_capped(waves,
                                                                             monkeypatch):
    da = waves.hovmoller("t", 0, band=(-10, 10))
    monkeypatch.setattr(G, "HOV_CONTOUR_CELLS", 10)
    monkeypatch.setattr(G, "HOV_RENDER_CELLS", 100)
    assert waves._hov_method(da, "auto") == "imshow"
    with pytest.raises(ValueError, match="use auto"):
        waves._hov_method(da, "contourf")


def test_figure_netcdf_and_a_refused_gif(waves, tmp_path):
    from PIL import Image

    hov = {"band": (-10, 10), "lons": (0, 360)}
    fig = waves.figure("t", 0, 0, waves.home, "viridis", None, None, "notebook",
                       kind="hovmoller", hov=hov)
    assert Image.open(io.BytesIO(fig)).size == (900, 500)

    out = tmp_path / "hov.nc"
    out.write_bytes(waves.netcdf("t", 0, 0, waves.home, kind="hovmoller", hov=hov))
    with xr.open_dataset(out) as ds:
        assert dict(ds.sizes) == {"valid_time": 36, "lon": 72}
        assert ds.t.attrs["weighting"] == "cos(latitude), NaN skipped"

    with pytest.raises(ValueError, match="GIF steps through time"):
        waves.gif("t", 0, waves.home, None, None, None, kind="hovmoller", hov=hov)
    with pytest.raises(NotImplementedError):
        waves.netcdf("t", 0, 0, waves.home, kind="map")


# --------------------------------------------------------------- serving


@pytest.fixture
def served(waves):
    """Mounted the way `gmpas view --generic` mounts it: behind the dashboard's
    router, which calls the page's do_GET with its own `self`."""
    import urllib.request

    from gmpas.dashboard import Source, router
    from gmpas.viewer import PAGE, _handler, bind

    srv = bind(router([Source("run", "data", "test", _handler(waves, PAGE))]), 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}/run"
    yield base, urllib.request
    srv.shutdown()
    srv.server_close()


QUERY = ("var=t&time=3&level=0&kind=hovmoller&extent=0,360,-90,90"
         "&hlat0=-10&hlat1=10&hlon0=0&hlon1=360")


def test_the_page_polls_202_then_gets_the_image_and_its_rows(served):
    base, request = served
    statuses = []
    deadline = time.time() + 20
    while time.time() < deadline:
        with request.urlopen(f"{base}/api/plot?{QUERY}&w=500&h=400") as r:
            statuses.append(r.status)
            if r.status == 202:
                assert len(json.loads(r.read())["progress"]) == 2
                time.sleep(0.05)
                continue
            png, rows = r.read(), r.headers["X-Hov-Y"].split(",")
            box = r.headers["X-Hov-Box"].split(",")
            break
    assert statuses[-1] == 200 and png[:4] == b"\x89PNG"
    assert len(rows) == 36 and len(box) == 4


def test_bad_parameters_and_exports_over_http(served):
    import urllib.error

    base, request = served
    with pytest.raises(urllib.error.HTTPError) as err:
        request.urlopen(f"{base}/api/plot?{QUERY}&hmethod=exec")
    assert "not auto" in json.loads(err.value.read())["error"]

    nc = request.urlopen(f"{base}/api/export/netcdf?{QUERY}").read()
    assert nc[:4] == b"\x89HDF"
    with pytest.raises(urllib.error.HTTPError) as err:
        request.urlopen(f"{base}/api/export/gif?{QUERY}")
    assert "GIF steps through time" in json.loads(err.value.read())["error"]


def test_the_mpas_viewer_has_no_hovmoller():
    from gmpas.viewer import Viewer

    assert not hasattr(Viewer, "hovmoller_progress") and not hasattr(Viewer, "plot")
