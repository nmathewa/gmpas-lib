"""MetPy calculations in scripts and chat: MetPy computes them, with units;
gmpas reads the fields, checks each argument measures what it is for, and
writes the same call in the transcript."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("metpy")

from gmpas import derive, script, transcript  # noqa: E402

LEVELS = [1000.0, 850.0, 500.0]


@pytest.fixture(scope="module")
def grid(tmp_path_factory):
    """A small ERA5-like file: winds, temperature and dewpoint on pressure
    levels, each with units, as ERA5 writes them."""
    rng = np.random.default_rng(5)
    shape = (3, 3, 4, 5)
    lat, lon = [-6.0, -4.0, -2.0, 0.0], [146.0, 147.0, 148.0, 149.0, 150.0]

    def field(base, spread, units):
        return (("valid_time", "pressure_level", "latitude", "longitude"),
                base + rng.normal(0, spread, shape), {"units": units})
    ds = xr.Dataset(
        {"u": field(5.0, 3.0, "m s**-1"), "v": field(-2.0, 3.0, "m s**-1"),
         "t": field(285.0, 5.0, "K")},
        coords={"valid_time": np.array(["2012-02-13T00", "2012-02-13T01", "2012-02-13T02"],
                                       dtype="datetime64[ns]"),
                "pressure_level": ("pressure_level", LEVELS, {"units": "hPa"}),
                "latitude": lat, "longitude": lon})
    ds["td"] = ds["t"] - 4.0 - np.abs(rng.normal(0, 2, shape))
    ds["td"].attrs["units"] = "K"
    path = tmp_path_factory.mktemp("era5") / "levels.nc"
    ds.to_netcdf(path)
    return path


@pytest.fixture(scope="module")
def viewer(grid):
    from gmpas.generic import GenericViewer

    v = GenericViewer([str(grid)])
    yield v
    v.close()


@pytest.mark.parametrize("expr", [
    "mpcalc.wind_speed(u, v)",
    "mpcalc.wind_direction(u, v)",
    "mpcalc.potential_temperature(pressure_level, t)",
    "mpcalc.relative_humidity_from_dewpoint(t, td)",
    "mpcalc.equivalent_potential_temperature(pressure_level, t, td)",
    "mpcalc.saturation_vapor_pressure(t)",
])
@pytest.mark.parametrize("time, level", [(0, 0), (2, 1)])
def test_the_viewer_computes_what_metpy_computes(viewer, grid, expr, time, level):
    """The viewer's value at a cell, against MetPy run on the file through
    the transcript's own code."""
    import metpy.calc as mpcalc

    meta = viewer.describe()
    code, _ = transcript.field(expr, meta, time=time, level=level)
    with xr.open_dataset(grid) as ds:
        want = eval(code, {"ds": ds, "np": np, "xr": xr, "mpcalc": mpcalc})
        want = np.asarray(want.metpy.dequantify().values if hasattr(want, "metpy")
                          else want, dtype=float)
    got = viewer.probe(148.0, -4.0, expr, time, level)["value"]
    with xr.open_dataset(grid) as ds:
        j, i = list(ds.longitude.values).index(148.0), list(ds.latitude.values).index(-4.0)
    np.testing.assert_allclose(got, want[i, j], rtol=1e-9)


def test_an_argument_must_measure_what_it_is_for(viewer):
    meta = viewer.describe()
    with pytest.raises(script.ScriptError,
                       match="the relative humidity is td, in K, which is not a "
                             "fraction or a percentage"):
        script.read("mpcalc.dewpoint_from_relative_humidity(t, td)", meta)
    with pytest.raises(script.ScriptError, match="mpcalc.wind_speed\\(\\) takes 2"):
        script.read("mpcalc.wind_speed(u)", meta)
    with pytest.raises(script.ScriptError, match="not a calculation here"):
        script.read("mpcalc.vorticity(u, v)", meta)
    with pytest.raises(script.ScriptError, match="not a variable, nor a level axis"):
        script.read("mpcalc.potential_temperature(valid_time, t)", meta)


def test_xarray_and_metpy_spellings_are_read():
    meta = {"variables": [{"name": n, "levels": 1, "units": "m s-1"} for n in "uv"],
            "steps": 1, "labels": ["x"], "cmaps": ["viridis"], "projections": ["auto"],
            "colour_options": {"levels": {}}}
    for text in ("mpcalc.wind_speed(u, v)", "metpy.calc.wind_speed(ds['u'], ds.v)",
                 "wind_speed(u, v)"):
        assert script.read(text, meta)[1] == {"var": "mpcalc.wind_speed(u, v)"}
    assert script.read("np.hypot(ds.u, ds['v'])", meta)[1] == {"var": "hypot(u, v)"}


def test_units_from_the_tables_are_said_in_the_transcript():
    meta = {"file": "h.nc", "time_dim": "Time", "variables": [
        {"name": "theta", "levels": 1, "units": "K",
         "units_source": "MPAS core_atmosphere registry"},
        {"name": "pressure", "levels": 1, "units": "Pa", "units_source": ""}]}
    line = transcript.spell({"var": "mpcalc.potential_temperature(pressure, theta)"},
                            meta).splitlines()[1]
    assert line.startswith('mpcalc.potential_temperature(ds["pressure"], '
                           'ds["theta"].assign_attrs(units="K"))')
    assert "theta in K, from the MPAS core_atmosphere registry" in line and "in K" in line


def test_a_field_with_no_units_anywhere_is_refused():
    plan = derive.parse("mpcalc.wind_speed(u, v)", ["u", "v"])
    with pytest.raises(KeyError, match="u has no units"):
        derive.check_units(plan, lambda a: "")
