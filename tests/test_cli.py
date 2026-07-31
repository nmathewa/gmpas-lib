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


def test_all_steps_requires_a_step_placeholder(capsys, tmp_path, simple_mesh_file):
    """A format spec like {step:04d} must count -- it has no bare {step}."""
    diag = write_diag(tmp_path / "diag.nc", n_cells=4, n_edges=24)

    assert main(["plot", str(diag), "mslp", "--all-steps", "-o", "out.png"]) == 1
    assert "{step}" in capsys.readouterr().err

    parser = build_parser()
    args = parser.parse_args(["plot", str(diag), "mslp", "--all-steps",
                              "-o", "f_{step:04d}.png"])
    assert "{step" in args.out


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
