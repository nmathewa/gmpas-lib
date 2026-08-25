"""The timing facility must cost nothing when nobody asked for it.

These assert on the *shape* of the instrumentation rather than on durations:
a duration test at fixture scale measures the clock, not the code. The label
set is asserted so that deleting an instrumentation point fails loudly --
these labels are how a run on a real mesh gets attributed, and the whole point
of adding them was that guessing which stage dominates has been wrong before.
"""

from __future__ import annotations

import pytest

from gmpas import timing


@pytest.fixture(autouse=True)
def _clean_timing(monkeypatch):
    """Leave the module exactly as it was found, however the test exits."""
    monkeypatch.delenv("GMPAS_TIMING", raising=False)
    monkeypatch.delenv("GMPAS_TIMING_FILE", raising=False)
    timing.refresh()
    timing.reset()
    yield
    monkeypatch.delenv("GMPAS_TIMING", raising=False)
    timing.refresh()
    timing.reset()


def test_disabled_timing_allocates_nothing(capsys):
    """The guarantee that lets this be called per animation frame: with
    timing off, `step` hands back one shared do-nothing object rather than
    building a context manager."""
    assert timing.LEVEL == 0
    assert timing.step("anything") is timing._NULL
    assert timing.step("a") is timing.step("b")

    with timing.step("mesh.tree_build", cells=41_902_592) as t:
        t.note(opened=3)

    assert capsys.readouterr().err == ""


def test_enabling_timing_reports_the_label_and_its_fields(capsys):
    import os

    os.environ["GMPAS_TIMING"] = "1"
    timing.refresh()

    with timing.step("mesh.discover", scanned=3000) as t:
        t.note(opened=1)

    err = capsys.readouterr().err
    assert "gmpas.timing" in err
    assert "mesh.discover" in err
    assert "scanned=3000" in err
    assert "opened=1" in err


def test_a_junk_level_disables_rather_than_crashing(monkeypatch):
    """GMPAS_TIMING=yes is a plausible thing to type, and instrumentation is
    the last thing that should take a run down."""
    monkeypatch.setenv("GMPAS_TIMING", "yes")
    timing.refresh()

    assert timing.LEVEL == 0
    assert timing.step("x") is timing._NULL


def test_the_roll_up_totals_repeated_labels(capsys):
    import os

    os.environ["GMPAS_TIMING"] = "1"
    timing.refresh()
    timing.reset()

    for _ in range(3):
        with timing.step("view.query", px=4000):
            pass
    capsys.readouterr()

    timing._report()
    err = capsys.readouterr().err
    assert "roll-up" in err
    assert "view.query" in err
    assert "n=3" in err


def test_an_exception_still_reports_the_step(capsys):
    """A stage that died is exactly the one worth knowing the duration of."""
    import os

    os.environ["GMPAS_TIMING"] = "1"
    timing.refresh()

    with pytest.raises(ValueError):
        with timing.step("mesh.build"):
            raise ValueError("no room")

    assert "mesh.build" in capsys.readouterr().err


def test_timing_can_be_sent_to_a_file(tmp_path, monkeypatch):
    """A -j 32 pool interleaves stderr into mush, so it has somewhere to go."""
    out = tmp_path / "timing.log"
    monkeypatch.setenv("GMPAS_TIMING", "1")
    monkeypatch.setenv("GMPAS_TIMING_FILE", str(out))
    timing.refresh()

    with timing.step("series.scan", files=3000):
        pass

    assert "series.scan" in out.read_text()


def test_the_load_path_reports_the_stages_it_was_added_for(tmp_path, capsys,
                                                           monkeypatch):
    """The label set is the deliverable: these are the stages that have to be
    attributable on a real run, so losing one should fail here."""
    from conftest import write_mesh
    from gmpas.mesh import MpasMesh

    path = tmp_path / "mesh.nc"
    write_mesh(path, [(0.0, 0.0), (10.0, 0.0)])

    monkeypatch.setenv("GMPAS_TIMING", "1")
    timing.refresh()

    mesh = MpasMesh.load(path)
    mesh.tree()

    err = capsys.readouterr().err
    for label in ("mesh.signature", "mesh.build", "mesh.cache_load",
                  "mesh.tree_build"):
        assert label in err, f"lost the {label} timing point"


def test_progress_is_quiet_about_nothing_and_loud_about_something(capsys):
    """Under a scheduler stdout is a log file, so the bar becomes lines."""
    bar = timing.Progress(4, unit="block")
    for _ in range(4):
        bar.advance()
    bar.close()

    out = capsys.readouterr().out
    assert "4/4" in out
    assert "s/block" in out


def test_clock_reads_as_time_not_as_a_float():
    assert timing.clock(45) == "45s"
    assert timing.clock(90) == "1m30s"
    assert timing.clock(3700) == "1h01m"
