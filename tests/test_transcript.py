"""The transcript (gmpas.transcript) is held to the numbers: each form it
writes is run as xarray on a small dataset and compared with what gmpas
computes for the same field, step and level (gmpas.derive)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from gmpas import derive, transcript

TIMES = pd.date_range("2011-11-16", periods=18, freq="MS") + pd.Timedelta(days=15)
LEVELS = [1000.0, 850.0, 500.0]


@pytest.fixture(scope="module")
def ds():
    rng = np.random.default_rng(3)
    shape = (len(TIMES), len(LEVELS), 4, 5)

    def field(base):
        data = base + rng.normal(0, 3, shape)
        data[rng.random(shape) < 0.05] = np.nan      # missing cells: skipna counts
        return (("valid_time", "pressure_level", "latitude", "longitude"), data)
    return xr.Dataset(
        {"t": field(280.0), "u": field(5.0), "v": field(-2.0)},
        coords={"valid_time": TIMES, "pressure_level": LEVELS,
                "latitude": [-6.0, -4.0, -2.0, 0.0], "longitude": [146.0, 147.0, 148.0,
                                                                  149.0, 150.0]})


META = {"file": "synthetic.nc", "time_dim": "valid_time",
        "labels": [t.strftime("%Y-%m-%d %H:%M") for t in TIMES],
        "lat_dim": "latitude", "lon_dim": "longitude",
        "variables": [{"name": n, "levels": 3, "dim": "pressure_level",
                       "level_values": LEVELS, "level_units": "hPa"} for n in "tuv"]}


def gmpas_value(ds, expr, time, level):
    """What gmpas computes: derive.evaluate, reading one step and level at a
    time, and reducing over time the way the viewers do."""
    plan = derive.parse(expr, ["t", "u", "v"])

    def read(name, step):
        return ds[name].isel(valid_time=step, pressure_level=level).values

    def reduce(term):
        _, name, how, picks, _ = term
        steps = derive.steps_for(picks, META["labels"])
        return derive.accumulate(how, (read(name, s) for s in steps))
    return derive.evaluate(plan, time, read, reduce)


@pytest.mark.parametrize("expr", [
    "t",
    "t - 273.15",
    "100 * u",
    "u / v",
    "hypot(u, v)",
    "diff(t)",
    "t.mean('time')",
    "t.sel(month=12).mean('time')",
    "t.sel(month=[6, 7, 8]).max('time')",
    "t.sel(season='DJF').mean('time')",
    "t.sel(year=2012).min('time')",
    "t.sel(time='2012').mean('time')",
    "t.sel(time=slice('2012-02', '2012-05')).mean('time')",
    "t.isel(time=slice(0, 6)).mean('time')",
    "t - t.sel(month=12).mean('time')",
])
@pytest.mark.parametrize("time, level", [(0, 0), (5, 1), (17, 2)])
def test_the_transcript_reads_the_same_numbers(ds, expr, time, level):
    code, _ = transcript.field(expr, META, time=time, level=level)
    got = eval(code, {"ds": ds, "np": np, "xr": xr})   # synthetic data, tests only
    np.testing.assert_allclose(np.asarray(got), gmpas_value(ds, expr, time, level),
                               equal_nan=True, rtol=1e-12)


def test_a_place_is_the_nearest_point(ds):
    code, _ = transcript.field("t", META, time=2, level=1, point=[147.4, -2.6])
    got = eval(code, {"ds": ds, "np": np, "xr": xr})
    assert float(got) == float(ds["t"].isel(valid_time=2, pressure_level=1)
                               .sel(latitude=-2.0, longitude=147.0)) or np.isnan(float(got))
    assert code.endswith('.sel(latitude=-2.6, longitude=147.4, method="nearest")')


def test_the_transcript_says_what_it_shows():
    text = transcript.spell({"var": "hypot(u, v)", "time": 5, "level": 1}, META)
    assert text == ('ds = xr.open_dataset("synthetic.nc")\n'
                    'np.hypot(ds["u"], ds["v"]).isel(valid_time=5)'
                    '.sel(pressure_level=850)   # 2012-05-16 00:00; 850 hPa\n')
    asked = transcript.spell({"ask": {"what": "max", "var": "t"}, "time": 0}, META)
    assert asked.splitlines()[1].startswith('ds["t"].isel(valid_time=0)'
                                            '.sel(pressure_level=1000).max()   #')
    # no lat/lon axes to pick by (an MPAS mesh): the place is said, not picked
    mpas = {**META, "lat_dim": "", "lon_dim": ""}
    assert "the nearest cell to 147.4E, 2.6S" in transcript.spell(
        {"var": "t", "point": [147.4, -2.6]}, mpas)
    assert transcript.spell({"var": "nope"}, META) == ""
    assert transcript.spell({"region": [0, 1, 0, 1]}, META) == ""
