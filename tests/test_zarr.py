"""Zarr stores in the lat/lon viewer: a `.zarr` directory opens as one
dataset and reads the same numbers as the same data in a netCDF file."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

pytest.importorskip("zarr")

from gmpas import data, netcdf  # noqa: E402
from gmpas.series import expand  # noqa: E402


def _dataset():
    """ERA5-like: hourly q on pressure levels, north first, a missing hour."""
    rng = np.random.default_rng(4)
    time = pd.date_range("2001-08-27", periods=6, freq="h")
    q = rng.random((6, 3, 5, 8)).astype("f4") * 0.02
    q[2] = np.nan                                       # an hour without data
    return xr.Dataset(
        {"q": (("time", "level", "latitude", "longitude"), q,
               {"units": "kg kg**-1", "long_name": "Specific humidity"})},
        coords={"time": time, "level": ("level", [1000.0, 850.0, 500.0], {"units": "hPa"}),
                "latitude": np.linspace(10.0, -10.0, 5),
                "longitude": np.arange(140.0, 156.0, 2.0)})


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    folder = tmp_path_factory.mktemp("zarr")
    ds = _dataset()
    ds.to_zarr(folder / "q.zarr", zarr_format=3, consolidated=True,
               encoding={"q": {"chunks": (6, 1, 2, 2)}})
    ds.to_netcdf(folder / "q.nc")
    return folder / "q.zarr", folder / "q.nc"


def test_a_store_is_one_dataset_and_a_lat_lon_grid(pair):
    store, nc = pair
    assert netcdf.is_zarr(store) and not netcdf.is_zarr(nc)
    assert expand(store) == [store]
    kind, why = data.detect_kind(store)
    assert kind == "generic" and why.startswith("Zarr store · regular lat/lon grid · 5 x 8")


def test_a_store_reads_what_the_same_netcdf_reads(pair):
    from gmpas.generic import GenericViewer

    store, nc = pair
    z, n = GenericViewer(store), GenericViewer(nc)
    try:
        assert z.labels == n.labels and z.steps == 6
        assert [(v["name"], v["levels"], v["units"]) for v in z.describe()["variables"]] \
            == [(v["name"], v["levels"], v["units"]) for v in n.describe()["variables"]]
        for step, level in ((0, 0), (3, 2), (2, 1)):          # step 2 is all missing
            a = z._cropped("q", step, level, z.home)
            b = n._cropped("q", step, level, n.home)
            np.testing.assert_array_equal(np.asarray(a.values), np.asarray(b.values))
            assert z.probe(146.0, 0.0, "q", step, level)["value"] == pytest.approx(
                n.probe(146.0, 0.0, "q", step, level)["value"], nan_ok=True)
        frame_z = z.frame("q", 3, 1, z.home, "viridis", None, None)
        frame_n = n.frame("q", 3, 1, n.home, "viridis", None, None)
        assert frame_z[0] == frame_n[0]                       # the same picture
    finally:
        z.close()
        n.close()


def test_mpas_output_in_zarr_is_said_not_read(tmp_path):
    xr.Dataset({"t2m": (("Time", "nCells"), np.zeros((1, 4)))}).to_zarr(
        tmp_path / "mpas.zarr", zarr_format=3)
    with pytest.raises(ValueError, match="MPAS output from netCDF files, not Zarr"):
        data.detect_kind(tmp_path / "mpas.zarr")


def test_a_store_without_zarr_installed_names_the_extra(pair, monkeypatch):
    import builtins

    real = builtins.__import__

    def no_zarr(name, *args, **kwargs):
        if name == "zarr":
            raise ImportError("no zarr")
        return real(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_zarr)
    with pytest.raises(ModuleNotFoundError, match=r'pip install "gmpas\[zarr\]"'):
        netcdf.open_any(pair[0])
