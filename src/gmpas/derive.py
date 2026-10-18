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

A field may also be one reduced over time, written as xarray writes it:

    sst.sel(month=12).mean('time')          every December, averaged
    sst.sel(time='2012').max('time')        the largest value in 2012
    sst - sst.sel(season='DJF').mean('time')    an anomaly

Steps are picked by `.sel(time='2012-12')` (a date, as far as it is given),
`.sel(time=slice('2012-12', '2013-02'))` (both ends included),
`.isel(time=slice(0, 12))` (by number, the end left out) and by calendar field
-- `.sel(month=12)`, `.sel(year=2012)`, `.sel(season='DJF')` -- which are this
grammar's own shorthand, not xarray. A season is its calendar months, as in
xarray: DJF of 2012 is January, February and December 2012. The reduction is
`mean`, `max` or `min` over `'time'`: each step counts once, and a cell that
is missing in a step is left out of that cell's result (xarray's `skipna`).
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Iterable

import numpy as np

_NAME = r"\w+"
_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
# a number is tried first so `1e-5` is one operand, not `1e` minus `5`; a name
# that merely starts with a digit (`2m_temp`) falls back to the second branch
# a field reduced over time: a name, then .sel()/.isel() picks, then the
# reduction; one bracket may sit inside another (slice(...))
REDUCTIONS = ("mean", "max", "min")
_TERM = (rf"{_NAME}(?:\.(?:sel|isel|{'|'.join(REDUCTIONS)})"
         r"\((?:[^()]|\([^()]*\))*\))+")
_FIELD = rf"(?:{_TERM}|{_NAME})"
_OPERAND = rf"(?:{_NUM}|{_TERM}|{_NAME})"

_BARE = re.compile(rf"^\s*({_FIELD})\s*$")
_DIFF = re.compile(rf"^\s*diff\(\s*({_NAME})\s*\)\s*$")
_HYPOT = re.compile(rf"^\s*hypot\(\s*({_FIELD})\s*,\s*({_FIELD})\s*\)\s*$")
_BINARY = re.compile(rf"^\s*({_OPERAND})\s*([+\-*/])\s*({_OPERAND})\s*$")
#: The longest expression taken, in characters.
MAX_EXPR = 500
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
    # the patterns backtrack on a long run of digits, and the expression
    # comes from a request: past this it is refused before it is matched
    if len(expr) > MAX_EXPR:
        raise KeyError(f"an expression is at most {MAX_EXPR} characters")
    names = list(names)

    def field(token: str):
        if "(" in token:
            return reduced(token, names)
        real = resolve(token, names)
        if real is None:
            raise KeyError(f"{token!r} is not a variable in this file")
        return real

    def operand(token: str):
        if "(" in token:
            return reduced(token, names)
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
        if isinstance(a, float) and isinstance(b, float):
            raise KeyError(f"{expr!r} names no variable")
        return ("binary", op, a, b)
    raise KeyError(f"{expr!r} is not a known variable, and not a recognised "
                   f"derived expression ({FORMS})")


def evaluate(plan: tuple, time: int, read: Callable[[str, int], np.ndarray],
             reduce: Callable[[tuple], np.ndarray] | None = None) -> np.ndarray:
    """Run a plan from `parse`. `read(name, step)` returns one real field;
    `reduce(term)` returns a field reduced over time (see `reduced`), which
    does not depend on the step."""
    def get(what):
        if isinstance(what, str):
            return read(what, time)
        if isinstance(what, tuple):
            if reduce is None:
                raise KeyError(f"{what[4]}: a field reduced over time is for "
                               f"lat/lon files so far, not for an MPAS run")
            return reduce(what)
        return what

    form = plan[0]
    if form == "field":
        return get(plan[1])
    if form == "diff":
        cur = read(plan[1], time)
        if time == 0:
            return np.full_like(cur, np.nan)   # no previous step to diff against
        return cur - read(plan[1], time - 1)
    if form == "hypot":
        return np.hypot(get(plan[1]), get(plan[2]))
    _, op, a, b = plan
    return _OPS[op](get(a), get(b))


def terms(plan: tuple) -> list[tuple]:
    """The fields reduced over time that a plan uses."""
    return [part for part in plan[1:] if isinstance(part, tuple)]


# ------------------------------------------------------ a field over time

_SEASONS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}
_DATE = re.compile(r"^(\d{1,5})-(\d{1,2})-(\d{1,2})"
                   r"(?:[ T](\d{1,2})(?::(\d{1,2})(?::(\d{1,2}))?)?)?$")
_KEY = re.compile(r"^(\d{1,5})(?:-(\d{1,2})(?:-(\d{1,2})"
                  r"(?:[ T](\d{1,2})(?::(\d{1,2})(?::(\d{1,2}))?)?)?)?)?$")


def _value(node):
    """A pick's value as written: a number, text, a list of them, a slice."""
    if isinstance(node, ast.Constant) and (node.value is None
                                           or type(node.value) in (int, str)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) \
            and isinstance(node.operand, ast.Constant) and type(node.operand.value) is int:
        return -node.operand.value
    if isinstance(node, (ast.List, ast.Tuple)):
        return tuple(_value(e) for e in node.elts)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id == "slice" and not node.keywords and 1 <= len(node.args) <= 3:
        parts = [_value(a) for a in node.args]
        if len(parts) == 1:
            parts = [None, parts[0]]
        return ("slice", *parts)
    raise KeyError("a pick takes a number, a date in quotes, a list or a slice")


def _spell(value) -> str:
    if isinstance(value, tuple) and value[:1] == ("slice",):
        return f"slice({', '.join(repr(v) for v in value[1:])})"
    if isinstance(value, tuple):
        return f"[{', '.join(repr(v) for v in value)}]"
    return repr(value)


def reduced(text: str, names: Iterable[str]) -> tuple:
    """`sst.sel(month=12).mean('time')` as ("reduced", field, how, picks, text):
    the field as the file spells it, the picks in the order written, and the
    text in one spelling, so equal terms are equal. Raises KeyError."""
    try:
        node = ast.parse(text.strip(), mode="eval").body
    except SyntaxError:
        raise KeyError(f"{text!r} is not understood") from None
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr in REDUCTIONS):
        raise KeyError(f"{text!r}: several time steps are picked and not reduced; "
                       f"end with .mean('time'), .max('time') or .min('time')")
    how = node.func.attr
    over = [_value(a) for a in node.args] + [_value(k.value) for k in node.keywords
                                             if k.arg == "dim"]
    if over != ["time"] or any(k.arg != "dim" for k in node.keywords):
        raise KeyError(f".{how}() reduces over time and says so: .{how}('time')")
    picks, chain = [], node.func.value
    while isinstance(chain, ast.Call) and isinstance(chain.func, ast.Attribute):
        kind = chain.func.attr
        if kind in REDUCTIONS:
            raise KeyError(f"{text!r}: a field is reduced over time once")
        if kind not in ("sel", "isel") or chain.args:
            raise KeyError(f".{kind}() is not understood before .{how}('time')")
        here = []
        for kw in chain.keywords:
            value = _value(kw.value)
            _check(kind, kw.arg, value)
            if kw.arg == "season":
                value = value.upper()
            here.append((kind, kw.arg, value))
        picks = here + picks
        chain = chain.func.value
    if not isinstance(chain, ast.Name):
        raise KeyError(f"{text!r}: a field of the file is reduced over time, "
                       f"not an expression")
    name = resolve(chain.id, names)
    if name is None:
        raise KeyError(f"{chain.id!r} is not a variable in this file")
    seen = [(kind, key) for kind, key, _ in picks]
    if len(seen) != len(set(seen)):
        raise KeyError(f"{text!r}: one pick is written twice")
    spelt = name + "".join(f".{kind}({key}={_spell(value)})" for kind, key, value in picks)
    return ("reduced", name, how, tuple(picks), f"{spelt}.{how}('time')")


def _check(kind: str, key: str, value) -> None:
    """Refuse a pick that cannot mean anything, by name."""
    def whole(v):
        return type(v) is int

    def piece(v, ok):
        return v is None or ok(v)

    sliced = isinstance(value, tuple) and value[:1] == ("slice",)
    if (kind, key) == ("isel", "time"):
        if whole(value) or (sliced and all(piece(v, whole) for v in value[1:])):
            return
        raise KeyError(".isel(time=) takes a step number or slice(first, end)")
    if (kind, key) == ("sel", "time"):
        def date(v):
            return isinstance(v, str) and bool(_KEY.match(v.strip()))
        if date(value) or (sliced and len(value) == 3
                           and all(piece(v, date) for v in value[1:])):
            return
        raise KeyError(".sel(time=) takes a date such as '2012-12', or "
                       "slice('2012-12', '2013-02')")
    many = value if isinstance(value, tuple) and not sliced else (value,)
    if (kind, key) == ("sel", "month") and many and all(
            whole(v) and 1 <= v <= 12 for v in many):
        return
    if (kind, key) == ("sel", "year") and many and all(whole(v) for v in many):
        return
    if (kind, key) == ("sel", "season") and isinstance(value, str) \
            and value.upper() in _SEASONS:
        return
    if (kind, key) in (("sel", "month"), ("sel", "year"), ("sel", "season")):
        raise KeyError(f".sel({key}=) takes "
                       + {"month": "1 to 12, or a list of them",
                          "year": "a year, or a list of them",
                          "season": "DJF, MAM, JJA or SON"}[key])
    raise KeyError(f".{kind}({key}=) is not a way to pick time steps; time, month, "
                   f"year and season are")


def _dated(labels: list[str]) -> list[tuple]:
    """Each label as (year, month, day, hour, minute, second), or KeyError when
    the steps carry no dates of their own."""
    out = []
    for label in labels:
        hit = _DATE.match(label)
        if not hit:
            raise KeyError(f"this file's time steps carry no dates (one is "
                           f"labelled {label!r}); pick them by number: "
                           f".isel(time=slice(0, 12))")
        out.append(tuple(int(v) if v is not None else 0 for v in hit.groups()))
    return out


def _key(text: str) -> tuple:
    return tuple(int(v) for v in _KEY.match(text.strip()).groups() if v is not None)


def steps_for(picks, labels: list[str]) -> list[int]:
    """The steps a term's picks select, in step order. A date matches as far
    as it is given -- '2012-1' is January 2012 -- as xarray matches one.
    Raises KeyError when the steps have no dates to match, when nothing is
    picked, or when two picked steps carry one label."""
    steps = list(range(len(labels)))
    dates = None
    for kind, key, value in picks:
        sliced = isinstance(value, tuple) and value[:1] == ("slice",)
        if kind == "isel":
            if sliced:
                steps = steps[slice(*value[1:])]
            elif -len(steps) <= value < len(steps):
                steps = [steps[value]]
            else:
                raise KeyError(f"step {value} is outside this file's {len(steps)} step(s)")
            continue
        dates = dates if dates is not None else _dated(labels)
        if key == "time" and sliced:
            first, last = (None if v is None else _key(v) for v in value[1:])
            steps = [s for s in steps
                     if (first is None or dates[s][:len(first)] >= first)
                     and (last is None or dates[s][:len(last)] <= last)]
        elif key == "time":
            want = _key(value)
            steps = [s for s in steps if dates[s][:len(want)] == want]
        else:
            many = value if isinstance(value, tuple) else (value,)
            if key == "season":
                many, key = _SEASONS[value.upper()], "month"
            at = 0 if key == "year" else 1
            steps = [s for s in steps if dates[s][at] in many]
    if not steps:
        raise KeyError("no time step matches " + "".join(
            f".{kind}({key}={_spell(value)})" for kind, key, value in picks))
    names = [labels[s] for s in steps]
    if len(set(names)) != len(names):
        twice = next(n for n in names if names.count(n) > 1)
        raise KeyError(f"two of the picked steps are both labelled {twice!r}; "
                       f"counting one time twice would be wrong")
    return steps


def accumulate(how: str, fields) -> np.ndarray:
    """`fields`, one float64 array per step, reduced cell by cell in step
    order: the mean of the steps that have a value, or their largest or
    smallest. A cell with no value in any step is NaN."""
    total = count = best = None
    for field in fields:
        field = np.asarray(field, dtype=np.float64)
        if how == "mean":
            have = np.isfinite(field)
            if total is None:
                total, count = np.zeros(field.shape), np.zeros(field.shape, np.int64)
            total[have] += field[have]
            count += have
        else:
            pick = np.fmax if how == "max" else np.fmin
            best = field.copy() if best is None else pick(best, field)
    if how == "mean":
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(count > 0, total / count, np.nan)
    return best
