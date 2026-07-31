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
