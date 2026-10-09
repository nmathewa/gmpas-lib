"""`gmpas view` (and info/plot) decide MPAS output vs a lat/lon grid themselves."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from conftest import write_diag, write_mesh
from gmpas.data import detect_kind

LAT = np.linspace(-89.5, 89.5, 20)
LON = np.linspace(0.5, 359.5, 40)


def test_mpas_history_carrying_its_mesh(tmp_path):
    f = write_mesh(tmp_path / "history.2012-01-01_00.00.00.nc", [(0.0, 0.0), (10.0, 0.0)])
    kind, why = detect_kind(str(f))
    assert kind == "mpas" and why == "MPAS output · 2 cells"


def test_mpas_diag_with_its_mesh_beside_it(tmp_path):
    write_mesh(tmp_path / "x1.mesh.nc", [(0.0, 0.0), (10.0, 0.0), (5.0, 5.0)])
    diag = write_diag(tmp_path / "diag.2012-01-01_00.00.00.nc", 3, 9)
    kind, why = detect_kind(str(diag))
    assert kind == "mpas" and "mesh x1.mesh.nc (beside it)" in why


def test_mpas_diag_with_no_mesh_is_still_mpas(tmp_path):
    """Drawn as anything else it would be a wrong map; as MPAS it gets the
    reader's own "pass a mesh" message."""
    diag = write_diag(tmp_path / "diag.2012-01-01_00.00.00.nc", 3, 9)
    kind, why = detect_kind(str(diag))
    assert kind == "mpas" and "no mesh found" in why


def test_a_mesh_given_on_the_command_line_is_named(tmp_path):
    mesh = write_mesh(tmp_path / "m.nc", [(0.0, 0.0), (10.0, 0.0), (5.0, 5.0)])
    (tmp_path / "run").mkdir()                       # the mesh is not beside it
    diag = write_diag(tmp_path / "run" / "diag.nc", 3, 9)
    kind, why = detect_kind(str(diag), str(mesh))
    assert kind == "mpas" and why.endswith("mesh m.nc")


@pytest.mark.parametrize("names, tname", [
    (("latitude", "longitude"), "valid_time"),          # ERA5 / CDS
    (("lat", "lon"), "time"),
])
def test_a_regular_grid_is_generic(tmp_path, names, tname):
    lat, lon = names
    xr.Dataset({"v": ((tname, lat, lon), np.zeros((2, 20, 40)))},
               coords={tname: pd.date_range("2024-01-01", periods=2),
                       lat: LAT, lon: LON}).to_netcdf(tmp_path / "f.nc")
    kind, why = detect_kind(str(tmp_path / "f.nc"))
    assert kind == "generic" and why.startswith("regular lat/lon grid · 20 x 40")


def test_neither_asks(tmp_path):
    xr.Dataset({"v": (("a", "b"), np.zeros((4, 5)))},
               coords={"a": np.arange(4.0), "b": np.arange(5.0)}
               ).to_netcdf(tmp_path / "f.nc")
    assert detect_kind(str(tmp_path / "f.nc"))[0] == "ask"


def test_a_curvilinear_grid_asks(tmp_path):
    xr.Dataset({"v": (("y", "x"), np.zeros((20, 40))),
                "XLAT": (("y", "x"), np.repeat(LAT[:, None], 40, 1),
                         {"units": "degree_north"}),
                "XLONG": (("y", "x"), np.repeat(LON[None, :], 20, 0),
                          {"units": "degree_east"})}
               ).to_netcdf(tmp_path / "wrf.nc")
    kind, why = detect_kind(str(tmp_path / "wrf.nc"))
    assert kind == "ask" and "curvilinear" in why


def test_points_sharing_one_dimension_ask(tmp_path):
    xr.Dataset({"v": (("n",), np.zeros(10))},
               coords={"lat": ("n", np.linspace(-9, 9, 10)),
                       "lon": ("n", np.linspace(0, 9, 10))}).to_netcdf(tmp_path / "pts.nc")
    assert detect_kind(str(tmp_path / "pts.nc"))[0] == "ask"


# ------------------------------------------------------------------ the CLI


def _grid(tmp_path):
    xr.Dataset({"t2m": (("time", "lat", "lon"), np.full((2, 20, 40), 280.0),
                        {"units": "K"})},
               coords={"time": pd.date_range("2024-01-01", periods=2),
                       "lat": LAT, "lon": LON}).to_netcdf(tmp_path / "g.nc")
    return tmp_path / "g.nc"


def test_view_opens_a_grid_without_being_told(tmp_path, monkeypatch, capsys):
    from gmpas import cli

    seen = {}
    monkeypatch.setattr(cli, "_generic_view", lambda a: seen.setdefault("generic", 0))
    monkeypatch.setattr(cli, "_dashboard", lambda a, **k: seen.setdefault("mpas", 0))
    cli.main(["view", str(_grid(tmp_path))])
    assert "generic" in seen and "mpas" not in seen
    assert "gmpas: regular lat/lon grid" in capsys.readouterr().err


def test_view_opens_mpas_without_being_told(tmp_path, monkeypatch, capsys):
    from gmpas import cli

    f = write_mesh(tmp_path / "history.2012-01-01_00.00.00.nc", [(0.0, 0.0), (10.0, 0.0)])
    seen = {}
    monkeypatch.setattr(cli, "_generic_view", lambda a: seen.setdefault("generic", 0))
    monkeypatch.setattr(cli, "_dashboard", lambda a, **k: seen.setdefault("mpas", 0))
    cli.main(["view", str(f)])
    assert "mpas" in seen and "generic" not in seen
    assert "gmpas: MPAS output · 2 cells" in capsys.readouterr().err


def test_the_old_flag_still_works_and_says_it_is_not_needed(tmp_path, monkeypatch, capsys):
    from gmpas import cli

    seen = {}
    monkeypatch.setattr(cli, "_generic_view", lambda a: seen.setdefault("generic", 0))
    cli.main(["view", "--generic", str(_grid(tmp_path))])
    assert "generic" in seen
    assert "--generic is no longer needed" in capsys.readouterr().err


def test_info_summarises_a_grid(tmp_path, capsys):
    from gmpas import cli

    assert cli.main(["info", str(_grid(tmp_path))]) == 0
    out = capsys.readouterr().out
    assert "grid      : 20 x 40 (lat, lon), global" in out
    assert "steps     : 2 across 1 file" in out and "t2m ('time', 'lat', 'lon')  K" in out


def test_plot_draws_a_grid(tmp_path):
    from gmpas import cli

    out = tmp_path / "t2m.png"
    assert cli.main(["plot", str(_grid(tmp_path)), "t2m", "-o", str(out)]) == 0
    assert out.read_bytes()[:4] == b"\x89PNG"


def test_plot_refuses_mpas_only_flags_on_a_grid(tmp_path, capsys):
    from gmpas import cli

    assert cli.main(["plot", str(_grid(tmp_path)), "t2m", "--symmetric"]) == 1
    assert "--symmetric is for MPAS output" in capsys.readouterr().err
