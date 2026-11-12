"""The command line surface: help, dispatch, and error reporting."""

from __future__ import annotations

import pytest

from conftest import write_diag
from gmpas.cli import build_parser, main


def test_bare_gmpas_shows_the_commands(capsys):
    """Running the bare command is a question, not a mistake."""
    assert main([]) == 0

    out = capsys.readouterr().out
    for command in ("info", "plot", "view"):
        assert command in out
    assert "usage: gmpas" in out


def test_help_lists_examples_and_environment(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out

    assert "examples:" in out
    assert "--all-steps" in out          # the non-obvious one
    assert "GMPAS_CACHE_DIR" in out


def test_version_is_reported(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "gmpas" in capsys.readouterr().out


def test_an_unknown_command_still_fails(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["frobnicate"])
    assert exc.value.code != 0


def test_info_reports_the_mesh(simple_mesh_file, capsys):
    assert main(["info", str(simple_mesh_file)]) == 0

    out = capsys.readouterr().out
    assert "cells     : 4" in out
    assert "regional" in out


def test_a_missing_file_is_reported_not_traced(capsys, tmp_path):
    assert main(["info", str(tmp_path / "absent.nc")]) == 1
    assert "gmpas:" in capsys.readouterr().err


def test_missing_plot_dependency_is_reported_not_traced(
    capsys, tmp_path, simple_mesh_file, monkeypatch
):
    """matplotlib/cartopy are the optional `plot` extra -- point at it, don't trace."""
    diag = write_diag(tmp_path / "diag.nc", n_cells=4, n_edges=24)

    real_import = __import__

    def blocked(name, *args, **kwargs):
        if name == "matplotlib":
            raise ModuleNotFoundError("No module named 'matplotlib'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked)

    assert main(["plot", str(diag), "mslp", "--mesh", str(simple_mesh_file),
                 "-o", str(tmp_path / "out.png")]) == 1

    err = capsys.readouterr().err
    assert "gmpas:" in err
    assert "pip install --upgrade gmpas" in err


def test_all_steps_requires_a_step_placeholder(capsys, tmp_path, simple_mesh_file):
    """A format spec like {step:04d} must count -- it has no bare {step}."""
    diag = write_diag(tmp_path / "diag.nc", n_cells=4, n_edges=24)

    assert main(["plot", str(diag), "mslp", "--all-steps", "-o", "out.png"]) == 1
    assert "{step}" in capsys.readouterr().err

    parser = build_parser()
    args = parser.parse_args(["plot", str(diag), "mslp", "--all-steps",
                              "-o", "f_{step:04d}.png"])
    assert "{step" in args.out


def _seen_limits(monkeypatch):
    """Record the vmin/vmax each cell_field call is given, and draw nothing."""
    import matplotlib.pyplot as plt

    import gmpas.plot

    seen = []

    def fake(mesh, values, **kw):
        seen.append((kw.get("vmin"), kw.get("vmax")))
        return plt.figure(), None

    monkeypatch.setattr(gmpas.plot, "cell_field", fake)
    return seen


def test_plot_passes_a_fixed_colour_range(tmp_path, simple_mesh_file, monkeypatch):
    diag = write_diag(tmp_path / "diag.nc", n_cells=4, n_edges=24)
    seen = _seen_limits(monkeypatch)

    assert main(["plot", str(diag), "mslp", "--mesh", str(simple_mesh_file),
                 "--vmin", "990", "--vmax", "1030",
                 "-o", str(tmp_path / "out.png")]) == 0
    assert main(["plot", str(diag), "mslp", "--mesh", str(simple_mesh_file),
                 "-o", str(tmp_path / "auto.png")]) == 0

    assert seen == [(990.0, 1030.0), (None, None)]


def test_all_steps_share_the_fixed_colour_range(tmp_path, simple_mesh_file,
                                                monkeypatch):
    """The point of the flags: every frame of a sequence on one scale."""
    diag = write_diag(tmp_path / "diag.nc", n_cells=4, n_edges=24, n_times=3)
    seen = _seen_limits(monkeypatch)

    assert main(["plot", str(diag), "mslp", "--mesh", str(simple_mesh_file),
                 "--all-steps", "-j", "1", "--vmin", "-1", "--vmax", "1",
                 "-o", str(tmp_path / "f_{step:02d}.png")]) == 0

    assert seen == [(-1.0, 1.0)] * 3


def test_an_inverted_colour_range_is_refused(tmp_path, simple_mesh_file, capsys):
    diag = write_diag(tmp_path / "diag.nc", n_cells=4, n_edges=24)

    assert main(["plot", str(diag), "mslp", "--mesh", str(simple_mesh_file),
                 "--vmin", "5", "--vmax", "5", "-o", str(tmp_path / "x.png")]) == 1
    assert "--vmin" in capsys.readouterr().err
    assert not (tmp_path / "x.png").exists()


def test_a_lat_lon_plot_takes_the_colour_range(tmp_path, monkeypatch):
    import numpy as np
    import xarray as xr

    import gmpas.generic

    path = tmp_path / "grid.nc"
    xr.Dataset({"t2m": (("lat", "lon"), np.arange(12.0).reshape(3, 4))},
               coords={"lat": [-10.0, 0.0, 10.0], "lon": [0.0, 10.0, 20.0, 30.0]}
               ).to_netcdf(path)
    seen = []

    def fake(self, var, time, level, extent, cmap, vmin, vmax, **kw):
        seen.append((vmin, vmax))
        return b"png"

    monkeypatch.setattr(gmpas.generic.GenericViewer, "figure", fake)
    assert main(["plot", str(path), "t2m", "--vmin", "2", "--vmax", "8",
                 "-o", str(tmp_path / "g.png")]) == 0
    assert seen == [(2.0, 8.0)]


# ---------------------------------------------------------------- path forms


def test_a_directory_is_read_as_a_series(tmp_path, capsys, monkeypatch):
    """A directory means every .nc directly inside it, sorted by name."""
    from conftest import write_mesh

    run = tmp_path / "run"
    run.mkdir()
    for hour in ("00", "06", "12"):
        write_mesh(run / f"history.2012-02-25_{hour}.00.00.nc", [(0.0, 0.0)])

    assert main(["info", str(run), "--limit", "0"]) == 0
    assert "3 steps across 3 files" in capsys.readouterr().out


def test_several_paths_are_accepted(tmp_path, capsys):
    """An unquoted glob reaches us already expanded by the shell."""
    from conftest import write_mesh

    files = [str(write_mesh(tmp_path / f"history.2012-02-25_{h}.00.00.nc",
                            [(0.0, 0.0)])) for h in ("00", "06")]

    assert main(["info", *files, "--limit", "0"]) == 0
    assert "2 steps across 2 files" in capsys.readouterr().out


def test_paths_and_variable_do_not_collide(tmp_path):
    """`plot` takes both, so argparse must not swallow the variable name."""
    from conftest import write_mesh

    files = [str(write_mesh(tmp_path / f"m{i}.nc", [(0.0, 0.0)])) for i in (1, 2)]
    args = build_parser().parse_args(["plot", *files, "areaCell", "-o", "x.png"])

    assert args.path == files
    assert args.var == "areaCell"


def _numpy_mix_error():
    """The shape the failure takes in the wild: matplotlib built for NumPy 1.x."""
    try:
        try:
            raise AttributeError("_ARRAY_API not found")
        except AttributeError as inner:
            raise ImportError("numpy.core.multiarray failed to import") from inner
    except ImportError as exc:
        return exc


def test_a_numpy_1_and_2_mix_is_one_message_not_a_traceback(monkeypatch, capsys):
    """Issue #153: a pip --user copy beside system matplotlib died in a wall of
    tracebacks naming neither copy."""
    import gmpas.cli as cli

    def broken(args):
        raise _numpy_mix_error()

    monkeypatch.delenv("GMPAS_DEBUG", raising=False)
    monkeypatch.setattr(cli, "_info", broken)
    code = cli.main(["info", "x.nc"])
    err = capsys.readouterr().err
    assert code == 1
    assert "built for NumPy 1.x was loaded under NumPy 2.x" in err
    for name in ("python :", "gmpas  :", "numpy  :", "python -m pip install",
                 "PYTHONNOUSERSITE", "which -a gmpas", "GMPAS_DEBUG"):
        assert name in err
    assert "Traceback" not in err


def test_gmpas_debug_shows_the_traceback(monkeypatch):
    import gmpas.cli as cli

    def broken(args):
        raise _numpy_mix_error()

    monkeypatch.setenv("GMPAS_DEBUG", "1")
    monkeypatch.setattr(cli, "_info", broken)
    with pytest.raises(ImportError, match="multiarray"):
        cli.main(["info", "x.nc"])


def test_an_unrelated_import_error_is_not_dressed_up(monkeypatch):
    import gmpas.cli as cli

    def broken(args):
        raise ImportError("cannot import name 'thing' from 'somewhere'")

    monkeypatch.setattr(cli, "_info", broken)
    with pytest.raises(ImportError, match="thing"):
        cli.main(["info", "x.nc"])
