"""What a view reads, written as xarray would read it: the transcript.

gmpas reads its files natively -- its own reader, numpy -- and xarray is the
reference it follows, not the engine. `spell` writes the field a view shows
as the xarray code that gives the same numbers, with the file's own names:

    ds = xr.open_dataset("cmip.nc")
    ds["ua"].isel(time=3).sel(plev=85000)   # 2000-05-16 00:00; 85000 Pa

so a user can read what was done, check it, and run it themselves. The
tests evaluate every form written here on a small dataset and compare it
with what gmpas computes (test_transcript.py): the transcript is held to
the numbers, not only to the spelling.

Only the data is written. Framing, projection, palette and the kind of plot
are the page's, and are said in a comment.
"""

from __future__ import annotations

from . import derive as _derive


def _num(value) -> str:
    """A value as Python writes it, text in double quotes."""
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, str):
        return '"' + value.replace('"', '\\"') + '"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_num(v) for v in value) + "]"
    return repr(value)


def _place(lon: float, lat: float) -> str:
    return (f"{abs(lon):g}{'W' if lon < 0 else 'E'}, "
            f"{abs(lat):g}{'S' if lat < 0 else 'N'}")


def _field(name: str) -> str:
    return f'ds["{name}"]'


def _time_mask(time: str, key: str, value) -> str:
    """A calendar pick as xarray writes it: a mask on the time coordinate."""
    part = {"month": "month", "year": "year", "season": "season"}[key]
    if isinstance(value, tuple):
        return f'ds["{time}"].dt.{part}.isin({_num(list(value))})'
    return f'ds["{time}"].dt.{part} == {_num(value)}'


def _reduced(term: tuple, time: str) -> str:
    """("reduced", field, how, picks, text) as a selection over time and the
    reduction, in xarray."""
    _, name, how, picks, _ = term
    out = _field(name)
    for kind, key, value in picks:
        sliced = isinstance(value, tuple) and value[:1] == ("slice",)
        if sliced:
            ends = ", ".join(_num(v) for v in value[1:])
            out += f".{kind}({time}=slice({ends}))"
        elif kind == "isel" or key == "time":
            out += f".{kind}({time}={_num(value)})"
        else:
            out += f".sel({time}={_time_mask(time, key, value)})"
    return f'{out}.{how}("{time}")'


def _operand(part, time: str) -> str:
    if isinstance(part, str):
        return _field(part)
    if isinstance(part, tuple):
        return _reduced(part, time)
    return _num(part)


def _metpy(plan: tuple, variables: dict) -> tuple[str, list]:
    """A MetPy calculation as MetPy's docs write it, over `ds`: a field whose
    units come from the models' tables, not the file, is given them with
    .assign_attrs, and the comment says so. The result's units are MetPy's."""
    func, args = plan[1], plan[2]
    written, notes, units = [], [], {}
    for arg in args:
        if isinstance(arg, tuple):               # the level axis, as the pressure
            written.append(_field(arg[1]))
            owner = next((v for v in variables.values() if v.get("dim") == arg[1]), {})
            units[arg] = owner.get("level_units", "")
            continue
        var = variables.get(arg, {})
        units[arg] = var.get("units", "")
        if var.get("units_source"):
            written.append(f'{_field(arg)}.assign_attrs(units={_num(units[arg])})')
            notes.append(f"{arg} in {units[arg]}, from the {var['units_source']}")
        else:
            written.append(_field(arg))
    try:
        import warnings

        import numpy as np

        # only the result's units are wanted: one quiet value of each
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            _, out = _derive.calculate(
                func, args, lambda a: np.ones(1),
                lambda a: (1000.0, units[a]) if isinstance(a, tuple) else units[a])
        notes.append(f"in {out}" if out.strip() else "a fraction, no units")
    except KeyError:
        pass
    return f"mpcalc.{func}({', '.join(written)})", notes


def expression(plan: tuple, time: str, variables: dict | None = None) -> str:
    """A plan from `derive.parse`, as an xarray expression over `ds`."""
    form = plan[0]
    if form == "metpy":
        return _metpy(plan, variables or {})[0]
    if form == "field":
        return _operand(plan[1], time)
    if form == "hypot":
        return f"np.hypot({_operand(plan[1], time)}, {_operand(plan[2], time)})"
    if form == "diff":
        # this step minus the one before: the first step has none, and is
        # missing, as gmpas leaves it -- `.diff()` would drop it instead
        field = _operand(plan[1], time)
        return f"{field} - {field}.shift({time}=1)"
    _, op, a, b = plan
    return f"{_operand(a, time)} {op} {_operand(b, time)}"


_OPS = ("+", "-", "*", "/")


def _names(plan: tuple) -> list[str]:
    """The file's fields a plan reads."""
    if plan[0] == "metpy":
        return [a for a in plan[2] if isinstance(a, str)]
    out = []
    for part in plan[1:]:
        if isinstance(part, str) and part not in _OPS:
            out.append(part)
        elif isinstance(part, tuple):
            out.append(part[1])
    return out


def _stepped(plan: tuple) -> bool:
    """Whether a field of the plan still has a time axis to pick a step on: a
    field reduced over time has none."""
    if plan[0] == "metpy":
        return True
    return any(isinstance(part, str) and part not in _OPS for part in plan[1:])


def field(expr: str, meta: dict, time=None, level=None, point=None) -> tuple[str, list]:
    """One field -- `expr` at a step, a level and a place -- as an xarray
    expression, and the comments that go with it. Raises KeyError for an
    expression gmpas does not read."""
    variables = {v["name"]: v for v in meta.get("variables", [])}
    plan = _derive.parse(expr, variables)
    axis = meta.get("time_dim") or "time"
    body, picks, notes = expression(plan, axis, variables), [], []
    if plan[0] == "metpy":
        notes += _metpy(plan, variables)[1]
    if time is not None and _stepped(plan):
        picks.append(f".isel({axis}={time})")
        labels = meta.get("labels") or []
        if 0 <= time < len(labels):
            notes.append(labels[time])
    owner = next((variables[w] for w in _names(plan) if variables[w].get("levels", 1) > 1),
                 None)
    if owner is not None:
        lev = level or 0
        dim = owner.get("dim") or "level"
        values = owner.get("level_values") or []
        if 0 <= lev < len(values):
            picks.append(f".sel({dim}={values[lev]:g})")
            notes.append(f"{values[lev]:g} {owner.get('level_units') or ''}".strip())
        else:
            picks.append(f".isel({dim}={lev})")
    if point:
        lon, lat = point
        if meta.get("lat_dim") and meta.get("lon_dim"):
            picks.append(f'.sel({meta["lat_dim"]}={lat:g}, {meta["lon_dim"]}={lon:g}, '
                         f'method="nearest")')
        else:
            notes.append(f"the nearest cell to {_place(lon, lat)}")
    whole = f"({body})" if plan[0] in ("binary", "diff") and picks else body
    if plan[0] == "metpy" and picks:
        # MetPy keeps the units on the result; picks then work as on any field
        whole = body
    return whole + "".join(picks), notes


def spell(view: dict, meta: dict) -> str:
    """The data a view reads -- a resolved script state, the whole view -- as
    xarray code over the open file: the field shown, and the question asked
    of a field, if any. Empty when it reads no field gmpas can write."""
    lines = []
    if view.get("var"):
        try:
            code, notes = field(view["var"], meta, view.get("time"), view.get("level"),
                                view.get("point"))
        except KeyError:
            code, notes = "", []
        if code:
            lines.append(code + (f"   # {'; '.join(notes)}" if notes else ""))
    ask = view.get("ask")
    if ask:
        try:
            code, notes = field(ask["var"], meta, ask.get("time", view.get("time")),
                                ask.get("level", view.get("level")))
        except KeyError:
            code, notes = "", []
        if code:
            where = "largest" if ask["what"] == "max" else "smallest"
            lines.append(f"{code}.{ask['what']}()   # "
                         + "; ".join([*notes, f"the {where} value in the area shown, "
                                              f"and where it is"]))
    if not lines:
        return ""
    return f'ds = xr.open_dataset("{meta.get("file") or "the file"}")\n' \
        + "\n".join(lines) + "\n"
