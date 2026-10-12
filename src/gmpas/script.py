"""A view written down: the fast map's settings as a few lines of text.

    # 850 hPa wind speed over the Maritime Continent
    var    = hypot(u, v)
    time   = 12
    level  = 3
    cmap   = viridis
    levels = (0,20,2)(inf)
    proj   = Mercator
    region = 90, 150, -15, 15

One `key = value` per line. A line starting with `#` or `!` is a comment, and
a key with nothing after the `=` clears that setting. A script names only what
it wants to set; everything else on the page stays as it is.

The text is parsed, never evaluated. `parse` checks the grammar and the types,
and `resolve` checks the result against one open file -- the variable exists,
the time step is in range -- so the page (and anything else that writes a
script) gets an error naming the line rather than a wrong picture. The page
applies what `resolve` returns; this module draws nothing.
"""

from __future__ import annotations

import math

from . import colour as _colour
from . import derive as _derive

#: What "proj" takes for the plain lon/lat map.
LONLAT = "lonlat"

#: The settings that are not colour options, in the order `format` writes them.
KEYS = ("var", "time", "level", "cmap", "vmin", "vmax", "proj", "plon", "plat",
        "region", "box")

MAX_TEXT = 8000
MAX_LINES = 200

_HELP = {
    "var": "a variable, or a derived expression such as a - b",
    "time": "a step number (0 is the first, -1 the last) or the step's label",
    "level": "a level number, 0 first",
    "cmap": "a palette name",
    "vmin": "the bottom of the colour range",
    "vmax": "the top of the colour range",
    "proj": f"{LONLAT}, auto, or a projection name",
    "plon": "the projection's central longitude",
    "plat": "the projection's central latitude",
    "region": "west, east, south, north in degrees",
    "box": "the view box in the projection's own units: x0, x1, y0, y1",
}


class ScriptError(ValueError):
    """A script that cannot be applied; the message names the line."""


def keys() -> dict[str, str]:
    """Every key a script may set, with a line of help: the page's hint, and
    what a writer of scripts is told is on offer."""
    out = dict(_HELP)
    for name, spec in _colour.OPTIONS.items():
        if not spec.get("hidden"):
            out[name] = spec.get("help") or spec["type"]
    return out


def _number(key: str, text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"{key} takes a number, not {text!r}") from None
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value


def _four(key: str, text: str) -> list[float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError(f"{key} takes four numbers, e.g. {key} = 90, 150, -15, 15")
    a, b, c, d = (_number(key, p) for p in parts)
    if not (a < b and c < d):
        raise ValueError(f"{key}: the first of each pair must be the smaller")
    return [a, b, c, d]


def _value(key: str, text: str):
    if key in ("vmin", "vmax", "plon", "plat"):
        return _number(key, text)
    if key == "level":
        try:
            return int(text)
        except ValueError:
            raise ValueError(f"level takes a whole number, not {text!r}") from None
    if key == "time":
        try:
            return int(text)
        except ValueError:
            return text                      # a label, matched in `resolve`
    if key == "region":
        w, e, s, n = _four(key, text)
        if s < -90 or n > 90:
            raise ValueError("region: latitudes run from -90 to 90")
        return [w, e, s, n]
    if key == "box":
        return _four(key, text)
    return text


def parse(text: str) -> dict:
    """The script as {key: value}, with None for a key that is cleared.

    Colour options are gathered under "colour" and checked together, because
    some of them only make sense with or without others. Raises ScriptError.
    """
    if len(text) > MAX_TEXT:
        raise ScriptError("the script is too long")
    lines = text.splitlines()
    if len(lines) > MAX_LINES:
        raise ScriptError(f"a script has at most {MAX_LINES} lines")
    out: dict = {}
    colour: dict = {}
    where: dict[str, int] = {}
    for n, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line[0] in "#!":
            continue
        key, eq, value = line.partition("=")
        key, value = key.strip().lower(), value.strip()
        if not eq:
            raise ScriptError(f"line {n}: expected key = value, got {line!r}")
        if key not in KEYS and key not in _colour.OPTIONS:
            raise ScriptError(f"line {n}: unknown key {key!r}; one of "
                              f"{', '.join(keys())}")
        if key in where:
            raise ScriptError(f"line {n}: {key} is already set on line {where[key]}")
        where[key] = n
        if key in _colour.OPTIONS:
            colour[key] = value or None
            continue
        try:
            out[key] = _value(key, value) if value else None
        except ValueError as exc:
            raise ScriptError(f"line {n}: {exc}") from None
    if colour:
        given = {k: v for k, v in colour.items() if v is not None}
        try:
            cleaned = _colour.clean(given) if given else {}
        except ValueError as exc:
            message = str(exc).removeprefix("colour: ")
            n = min((where[k] for k in given if k in message), default=0)
            raise ScriptError(f"line {n}: {message}" if n else message) from None
        # a default spelled out (norm = linear) is cleaned away: it clears too
        out["colour"] = {k: cleaned.get(k) for k in colour}
    if out.get("proj") in (None, LONLAT) and "proj" in out:
        for key in ("plon", "plat", "box"):
            if out.get(key) is not None:
                raise ScriptError(f"line {where[key]}: {key} needs a projection, "
                                  f"and proj is {LONLAT}")
    if out.get("region") is not None and out.get("box") is not None:
        raise ScriptError(f"line {where['box']}: region and box both set the view; "
                          f"give one")
    return out


def resolve(state: dict, meta: dict) -> dict:
    """`state` checked against one open file, as the page applies it.

    `meta` is the viewer's `describe()`. The variable comes back spelled as
    the file spells it, the time step as a number, and "proj" as "" for the
    lon/lat map. Raises ScriptError.
    """
    out = dict(state)
    variables = {v["name"]: v for v in meta.get("variables", [])}
    levels = None
    if state.get("var") is not None:
        name = _derive.resolve(state["var"], variables)
        if name is None:
            try:
                _derive.parse(state["var"], variables)
            except KeyError as exc:
                raise ScriptError(f"var: {exc.args[0]}") from None
        else:
            out["var"] = name
            levels = variables[name].get("levels", 1)
    elif "var" in state:
        raise ScriptError("var cannot be cleared; name a variable")
    if state.get("time") is not None:
        steps, labels = meta.get("steps", 0), meta.get("labels", [])
        t = state["time"]
        if isinstance(t, str):
            if t not in labels:
                raise ScriptError(f"time: no step is labelled {t!r}")
            t = labels.index(t)
        elif not -steps <= t < steps:
            raise ScriptError(f"time: step {t} is outside this run's {steps} step(s)")
        out["time"] = t % steps
    if state.get("level") is not None:
        lev = state["level"]
        if lev < 0 or (levels is not None and lev >= levels):
            raise ScriptError(f"level: {lev} is outside {out['var']}'s "
                              f"{levels} level(s)" if levels is not None
                              else f"level: {lev} is below 0")
    if state.get("cmap") is not None and state["cmap"] not in meta.get("cmaps", []):
        raise ScriptError(f"cmap: {state['cmap']!r} is not one of this page's palettes")
    if "proj" in state:
        offered = meta.get("projections") or []
        name = state["proj"]
        if name in (None, LONLAT):
            out["proj"] = ""
        elif name not in offered:
            raise ScriptError(f"proj: {name!r} is not one of "
                              f"{', '.join([LONLAT, *offered])}")
    if "colour" in state and not meta.get("colour_options"):
        raise ScriptError("this page has no colour options")
    return out


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value) if value != int(value) or abs(value) >= 1e15 \
            else str(int(value))
    if isinstance(value, (list, tuple)):
        return ", ".join(_text(v) for v in value)
    return str(value)


def format(state: dict) -> str:
    """`state` as script text that `parse` reads back to the same thing."""
    rows = [(k, state[k]) for k in KEYS if k in state]
    rows += list((state.get("colour") or {}).items())
    width = max((len(k) for k, _ in rows), default=0)
    return "".join(f"{k.ljust(width)} = {_text(v)}".rstrip() + "\n" for k, v in rows)


_startup = ""


def set_startup(text: str) -> None:
    """The script `gmpas view --script` was given: the page applies it once,
    when it opens. One process serves one command, so one script is enough."""
    global _startup
    parse(text)
    _startup = text


def startup() -> str:
    return _startup
