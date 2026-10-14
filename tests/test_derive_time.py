"""Picking time steps and reducing over them (gmpas/derive.py).

The rules are xarray's where xarray has one: a date matches as far as it is
given, a slice of dates includes both ends, a season is its calendar months,
and a missing value is left out of its own cell's result.
"""

from __future__ import annotations

import numpy as np
import pytest

from gmpas import derive

NAMES = ["sst", "t2m"]
LABELS = [f"{y}-{m:02d}-01 00:00" for y in (2011, 2012, 2013) for m in range(1, 13)]


def _picked(text, labels=LABELS):
    return [labels[s] for s in derive.steps_for(derive.reduced(text, NAMES)[3], labels)]


def test_a_term_has_one_spelling():
    term = derive.reduced(
        'SST.sel(season="djf", year=[2012, 2013]).mean(dim="time")', NAMES)
    assert term[:3] == ("reduced", "sst", "mean")
    assert term[4] == "sst.sel(season='DJF').sel(year=[2012, 2013]).mean('time')"
    assert derive.reduced(term[4], NAMES) == term
    assert derive.parse("sst - " + term[4], NAMES) == ("binary", "-", "sst", term)
    assert derive.terms(derive.parse("sst - 273.15", NAMES)) == []


def test_a_date_matches_as_far_as_it_is_given():
    assert _picked("sst.sel(time='2012-12').mean('time')") == ["2012-12-01 00:00"]
    assert _picked("sst.sel(time='2012-1').mean('time')") == ["2012-01-01 00:00"]
    assert len(_picked("sst.sel(time='2012').mean('time')")) == 12
    assert _picked("sst.sel(time='2012-12-01 00').mean('time')") == ["2012-12-01 00:00"]
    hourly = [f"2012-01-0{d} {h:02d}:00" for d in (1, 2) for h in (0, 6, 12, 18)]
    assert _picked("sst.sel(time='2012-01-02').mean('time')", hourly) == hourly[4:]
    assert _picked("sst.sel(time='2012-01-01T06').mean('time')", hourly) == [hourly[1]]


def test_a_slice_of_dates_includes_both_ends_and_one_of_numbers_does_not():
    assert _picked("sst.sel(time=slice('2012-12', '2013-02')).mean('time')") == \
        ["2012-12-01 00:00", "2013-01-01 00:00", "2013-02-01 00:00"]
    assert len(_picked("sst.sel(time=slice('2012', '2013')).mean('time')")) == 24
    assert _picked("sst.sel(time=slice(None, '2011-02')).mean('time')") == LABELS[:2]
    assert _picked("sst.isel(time=slice(0, 3)).mean('time')") == LABELS[:3]
    assert _picked("sst.isel(time=slice(-2, None)).mean('time')") == LABELS[-2:]
    assert _picked("sst.isel(time=slice(0, 12, 3)).mean('time')") == LABELS[0:12:3]
    assert _picked("sst.isel(time=-1).max('time')") == LABELS[-1:]


def test_a_season_is_its_calendar_months():
    assert _picked("sst.sel(season='DJF', year=2012).mean('time')") == \
        ["2012-01-01 00:00", "2012-02-01 00:00", "2012-12-01 00:00"]
    assert len(_picked("sst.sel(season='JJA').mean('time')")) == 9
    assert _picked("sst.sel(month=[1, 7], year=2013).mean('time')") == \
        ["2013-01-01 00:00", "2013-07-01 00:00"]


@pytest.mark.parametrize("text, labels, says", [
    ("sst.sel(year=1999).mean('time')", LABELS,
     "no time step matches .sel\\(year=1999\\)"),
    ("sst.sel(time='2012-13').mean('time')", LABELS, "no time step matches"),
    ("sst.isel(time=slice(5, 5)).mean('time')", LABELS, "no time step matches"),
    ("sst.isel(time=99).mean('time')", LABELS, "step 99 is outside"),
    ("sst.sel(month=1).mean('time')", ["step 0", "step 1"], "carry no dates"),
    ("sst.sel(month=1).mean('time')", ["2012-01-01 00:00 +0", "2012-01-01 00:00 +1"],
     "carry no dates"),
    ("sst.sel(year=2011).mean('time')", LABELS + LABELS[:1], "both labelled"),
])
def test_picks_that_select_wrongly_are_refused(text, labels, says):
    with pytest.raises(KeyError, match=says):
        derive.steps_for(derive.reduced(text, NAMES)[3], labels)


def test_steps_with_no_dates_are_still_picked_by_number():
    term = derive.reduced("sst.isel(time=slice(0, 2)).mean('time')", NAMES)
    assert derive.steps_for(term[3], ["step 0", "step 1", "step 2"]) == [0, 1]


def test_a_cell_missing_in_a_step_is_left_out_of_its_own_result():
    a = np.array([[1.0, np.nan, 5.0], [np.nan, np.nan, 2.0]])
    b = np.array([[3.0, 4.0, np.nan], [np.nan, np.nan, 8.0]])
    with np.errstate(all="ignore"):
        stack = np.stack([a, b])
        np.testing.assert_array_equal(derive.accumulate("mean", [a, b]),
                                      np.nanmean(stack, axis=0))
        np.testing.assert_array_equal(derive.accumulate("max", [a, b]),
                                      np.nanmax(stack, axis=0))
        np.testing.assert_array_equal(derive.accumulate("min", [a, b]),
                                      np.nanmin(stack, axis=0))
    assert a[0, 0] == 1.0                         # the inputs are not written to


def test_a_viewer_that_cannot_reduce_says_so():
    plan = derive.parse("sst.mean('time')", NAMES)
    with pytest.raises(KeyError, match="for lat/lon files so far, not for an MPAS run"):
        derive.evaluate(plan, 0, lambda name, step: np.zeros(3))
