"""The derive box's grammar, shared by the MPAS and --generic viewers."""

from __future__ import annotations

import numpy as np
import pytest

from gmpas import derive

NAMES = ["q", "t2m", "u", "v", "2m_temp"]


@pytest.mark.parametrize("expr, plan", [
    ("q", ("field", "q")),
    ("Q", ("field", "q")),                              # ERA5 writes q; people type Q
    ("q + 1", ("binary", "+", "q", 1.0)),
    ("Q*1000", ("binary", "*", "q", 1000.0)),
    ("1000 * q", ("binary", "*", 1000.0, "q")),
    ("t2m - 273.15", ("binary", "-", "t2m", 273.15)),
    ("q-1e-3", ("binary", "-", "q", 1e-3)),             # one number, not 1e minus 3
    ("q + -1", ("binary", "+", "q", -1.0)),
    ("u - v", ("binary", "-", "u", "v")),
    ("2m_temp / 2", ("binary", "/", "2m_temp", 2.0)),   # a name may start with a digit
    ("hypot(U, v)", ("hypot", "u", "v")),
    ("diff(Q)", ("diff", "q")),
])
def test_what_each_form_reads(expr, plan):
    assert derive.parse(expr, NAMES) == plan


@pytest.mark.parametrize("expr, why", [
    ("1 + 2", "names no variable"),
    ("z + 1", "'z' is not a variable"),
    ("q ** 2", "not a recognised derived expression"),
    ("__import__('os')", "not a recognised derived expression"),
])
def test_what_is_refused(expr, why):
    with pytest.raises(KeyError, match=why):
        derive.parse(expr, NAMES)


def test_names_that_differ_only_in_case_need_the_exact_spelling():
    assert derive.parse("Q", ["q", "Q"]) == ("field", "Q")
    with pytest.raises(KeyError):
        derive.parse("QQ", ["qq", "Qq"])


def test_a_scalar_applies_to_every_value_and_keeps_nan():
    field = np.array([1.0, np.nan, 3.0], dtype="f4")
    out = derive.evaluate(derive.parse("q * 2", NAMES), 0, lambda name, step: field)
    np.testing.assert_array_equal(out, [2.0, np.nan, 6.0])
    assert out.dtype == field.dtype
