"""The *derive* box: a small, fixed expression grammar over real field names.

Shared by `viewer.Viewer` (MPAS) and `generic.GenericViewer` (`--generic`),
so an expression means the same thing in both.

Deliberately not eval()/exec(): the expression is untrusted input, reachable
over the network once --host 0.0.0.0 is in play for an HPC tunnel. Each form
below is matched whole and mapped to one specific numpy op; operands never
nest.

    a + b, a - b, a * b, a / b    either side may be a number: q * 1000, t2m - 273.15
    hypot(a, b)                   vector magnitude
    diff(a)                       this step minus the previous one
    a                             a field, matched case-insensitively (Q finds q)
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

import numpy as np

_NAME = r"\w+"
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
# a number is tried first so `1e-5` is one operand, not `1e` minus `5`; a name
# that merely starts with a digit (`2m_temp`) falls back to the second branch
_OPERAND = rf"(?:{_NUM}|{_NAME})"

_BARE = re.compile(rf"^\s*({_NAME})\s*$")
_DIFF = re.compile(rf"^\s*diff\(\s*({_NAME})\s*\)\s*$")
_HYPOT = re.compile(rf"^\s*hypot\(\s*({_NAME})\s*,\s*({_NAME})\s*\)\s*$")
_BINARY = re.compile(rf"^\s*({_OPERAND})\s*([+\-*/])\s*({_OPERAND})\s*$")
_OPS = {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide}

FORMS = ("a + b, a - b, a * b, a / b (either side may be a number), "
         "hypot(a, b), or diff(a)")


def resolve(name: str, names: Iterable[str]) -> str | None:
    """`name` as the file spells it: exact, else its one case-insensitive match.

    ERA5 names its fields `q`, `t`, `u`; people type `Q`. Two fields that
    differ only in case (`q` and `Q` in one file) make a loose match
    ambiguous, and then only the exact spelling is accepted.
    """
    names = list(names)
    if name in names:
        return name
    hits = [n for n in names if n.lower() == name.lower()]
    return hits[0] if len(hits) == 1 else None


def parse(expr: str, names: Iterable[str]) -> tuple:
    """The expression as a plan: ``(form, *operands)``, with each field
    operand spelled as the file spells it and each number a float.

    Raises KeyError, naming what was not understood.
    """
    names = list(names)

    def field(token: str) -> str:
        real = resolve(token, names)
        if real is None:
            raise KeyError(f"{token!r} is not a variable in this file")
        return real

    def operand(token: str):
        real = resolve(token, names)
        if real is not None:
            return real
        try:
            return float(token)
        except ValueError:
            raise KeyError(f"{token!r} is not a variable in this file") from None

    if m := _BARE.match(expr):
        return ("field", field(m.group(1)))
    if m := _DIFF.match(expr):
        return ("diff", field(m.group(1)))
    if m := _HYPOT.match(expr):
        return ("hypot", field(m.group(1)), field(m.group(2)))
    if m := _BINARY.match(expr):
        a, op, b = operand(m.group(1)), m.group(2), operand(m.group(3))
        if not (isinstance(a, str) or isinstance(b, str)):
            raise KeyError(f"{expr!r} names no variable")
        return ("binary", op, a, b)
    raise KeyError(f"{expr!r} is not a known variable, and not a recognised "
                   f"derived expression ({FORMS})")


def evaluate(plan: tuple, time: int, read: Callable[[str, int], np.ndarray]) -> np.ndarray:
    """Run a plan from `parse`. `read(name, step)` returns one real field."""
    form = plan[0]
    if form == "field":
        return read(plan[1], time)
    if form == "diff":
        cur = read(plan[1], time)
        if time == 0:
            return np.full_like(cur, np.nan)   # no previous step to diff against
        return cur - read(plan[1], time - 1)
    if form == "hypot":
        return np.hypot(read(plan[1], time), read(plan[2], time))
    _, op, a, b = plan
    a = read(a, time) if isinstance(a, str) else a
    b = read(b, time) if isinstance(b, str) else b
    return _OPS[op](a, b)
