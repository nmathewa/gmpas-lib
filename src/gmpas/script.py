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

The same view can be written the way it would be in Python with xarray, and
the two forms mix:

    spd = hypot(u, v)
    view(region=(90, 150, -15, 15), proj="Mercator")
    shade(spd.isel(time=12, level=3), cmap="viridis", levels="(0,20,2)(inf)")

A name that is not a key is a definition, `.isel(time=, level=)` picks a step
and a level by number (from 0, and -1 is the last step), `.sel(time="...")`
picks a step by its label and `.sel(level=850)` a level by its value,
`view()` frames the map and `shade()` draws. It
reads like Python and is not run as Python: the text goes through Python's
parser and only these forms are accepted.

What draws decides the kind of plot. `shade()` is the fast map; `contourf()`,
`contour()`, `pcolormesh()`, `imshow()`, `hist()`, `hovmoller()`, and
`series()` and `profile()` at a `lon=`, `lat=`, are the page's other plots.
Two or more of them, or winds, make a figure of layers, drawn in order:

    contourf(t.isel(time=0, level=3), cmap="RdBu_r")
    contour(z.isel(time=0, level=3), colors="black")
    vectors(u.isel(time=0, level=3), v.isel(time=0, level=3))
    rivers(color="blue")
    figure(title="925 hPa", look="ferret")

A question about a field is asked the way xarray would be: `snow.max()` and
`snow.min()` find the largest and smallest value in view and where it is. The
page says the value and the place; the picture is left as it is. `mark()`
round the question also marks the place on the map, and `mark(lon=, lat=)`
marks any place and shows the value there. In an expression `var` is the
field on screen.

The map under and over them is drawn the same way -- `coastlines()`,
`borders()`, `land()`, `ocean()`, `lakes()`, `rivers()`, `gridlines()` -- and
`figure()` sets the title, the projection and where the colour bars go.
Coastlines and gridlines are there unless a script draws its own.

The text is parsed, never evaluated. `parse` checks the grammar and the types,
and `resolve` checks the result against one open file -- the variable exists,
the time step is in range -- so the page (and anything else that writes a
script) gets an error naming the line rather than a wrong picture. The page
applies what `resolve` returns; this module draws nothing.
"""

from __future__ import annotations

import ast
import copy
import difflib
import math
import re

from . import colour as _colour
from . import derive as _derive
from . import layers as _layers

#: What "proj" takes for the plain lon/lat map.
LONLAT = "lonlat"

#: The settings that are not colour options, in the order `format` writes them.
KEYS = ("var", "time", "level", "kind", "point", "cmap", "vmin", "vmax", "proj",
        "plon", "plat", "region", "box")

MAX_TEXT = 8000
MAX_LINES = 200

_HELP = {
    "var": "a variable, or a derived expression such as a - b",
    "time": "a step number (0 is the first, -1 the last) or the step's label",
    "level": "a level number, 0 first",
    "kind": "the plot: map, contourf, contour, pcolormesh, hist, series, ...",
    "point": "lon, lat of the place a series or a profile is read at",
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


class _Unused(ScriptError):
    """A name was defined and nothing used it."""


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
    if key == "point":
        parts = [p.strip() for p in text.split(",")]
        if len(parts) != 2:
            raise ValueError("point takes a longitude and a latitude: point = 147.4, -2")
        lon, lat = (_number(key, p) for p in parts)
        if not -90 <= lat <= 90:
            raise ValueError("point: latitudes run from -90 to 90")
        return [lon, lat]
    return text


def parse(text: str, shown: str | None = None) -> dict:
    """The script as {key: value}, with None for a key that is cleared.

    Colour options are gathered under "colour" and checked together, because
    some of them only make sense with or without others. `shown` is the field
    on screen, which an expression may call `var`. Raises ScriptError.
    """
    if len(text) > MAX_TEXT:
        raise ScriptError("the script is too long")
    lines = text.splitlines()
    if len(lines) > MAX_LINES:
        raise ScriptError(f"a script has at most {MAX_LINES} lines")
    if _CALL_LINE.search(text):
        return _parse_python(text, shown)
    try:
        return _parse_lines(lines)
    except ScriptError:
        # `spd = hypot(u, v)` and then `var = spd`: a definition makes it the
        # Python form even with no call in it. A name set and never used is
        # not a definition, though -- it is a key spelled wrong.
        if _defines(text):
            try:
                return _parse_python(text, shown)
            except _Unused:
                pass
        raise


def _parse_lines(lines: list[str]) -> dict:
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
    return _finish(out, colour, where)


def _finish(out: dict, colour: dict, where: dict[str, int]) -> dict:
    """The checks that take the whole script: colour options against each
    other, and settings that only make sense together."""
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


# ------------------------------------------------------------ the Python form

#: A line that calls something, which only the Python form has.
_CALL_LINE = re.compile(r"^[ \t]*(?:[A-Za-z_][\w.]*[ \t]*)?\(", re.M)
#: What draws one field, and the plot kind it is.
_FIELD = {"shade": "map", "contourf": "contourf", "contour": "contour",
          "pcolormesh": "pcolormesh", "imshow": "imshow", "hist": "hist",
          "series": "series", "profile": "profile", "hovmoller": "hovmoller"}
#: What draws winds, and the layer it is.
_VECTOR = {"vectors": "quiver", "quiver": "quiver", "barbs": "barbs",
           "streamlines": "streamplot", "streamplot": "streamplot"}
#: A field call as a layer of a figure; the rest cannot be layered.
_AS_LAYER = {"shade": "pcolormesh", "contourf": "contourf", "contour": "contour",
             "pcolormesh": "pcolormesh", "imshow": "imshow", "gridvalues": "gridvalues"}
#: The map itself, as layers: each is a call with options and no field.
_FEATURE = ("coastlines", "borders", "land", "ocean", "lakes", "rivers", "gridlines")
_DRAW = (*_FIELD, *_VECTOR, *_FEATURE, "gridvalues", "figure")
_AT_POINT = ("series", "profile")
#: A question about a field, as xarray asks it, and what it finds.
_STATS = {"max": "max", "min": "min", "argmax": "max", "argmin": "min",
          "idxmax": "max", "idxmin": "min"}
_LATER = ("section", "save", "animate")
_VIEW_KEYS = {"region": "region", "box": "box", "proj": "proj",
              "lon0": "plon", "lat0": "plat", "plon": "plon", "plat": "plat"}
_FUNCS = ("hypot", "diff")


def _is_key(name: str) -> bool:
    return name in KEYS or name in _colour.OPTIONS


_SETTING = re.compile(r"^([A-Za-z_]\w*)\s*=(?!=)\s*(.*)$")
_QUOTED = re.compile(r"""^(['"])(.*)\1$""")


def _split(text: str) -> tuple[str, list[tuple[int, str, str | None]]]:
    """The script as (what Python's parser is given, the plain settings).

    A `key = value` line means in the Python form what it means on its own,
    whatever its value looks like -- `vmin =`, `cmap = cmo.thermal`,
    `levels = (0,20,2)(inf)` -- so those lines are read here and left blank
    for the parser. `var` is not one of them: it names a field, and may use a
    definition. A line inside an open bracket is the rest of a call.
    """
    kept, plain, depth = [], [], 0
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if line[:1] == "!":
            line = ""
        hit = _SETTING.match(line) if depth == 0 else None
        if hit and _is_key(hit[1].lower()) and (hit[1].lower() != "var"
                                                or not hit[2].strip()):
            key, value = hit[1].lower(), hit[2].strip()
            quoted = _QUOTED.match(value)
            if quoted:
                value = quoted[2].strip()
            elif value in ("None", ""):
                value = None
            elif value in ("True", "False"):
                value = value.lower()
            elif key in ("region", "box", "point") and value[:1] in "([" \
                    and value[-1:] in ")]":
                value = value[1:-1]
            plain.append((n, key, value or None))
            kept.append("")
            continue
        kept.append(line)
        bare = re.sub(r"""(['"]).*?\1""", "", line.split("#")[0])
        depth = max(0, depth + sum(bare.count(c) for c in "([")
                    - sum(bare.count(c) for c in ")]"))
    return "\n".join(kept), plain


def _defines(text: str) -> bool:
    text, _ = _split(text)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    return any(isinstance(s, ast.Assign) and len(s.targets) == 1
               and isinstance(s.targets[0], ast.Name) and not _is_key(s.targets[0].id)
               for s in tree.body)


def _near(name: str, choices) -> str:
    """` Did you mean x?` when one of `choices` is close to `name`, else ``."""
    hit = difflib.get_close_matches(name, list(choices), n=1, cutoff=0.6)
    return f"; did you mean {hit[0]}?" if hit else ""


def _fail(node, message: str):
    raise ScriptError(f"line {node.lineno}: {message}")


def _number_of(node):
    """The number a node writes, or None: 5, -5, 2.5e3."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        inner = _number_of(node.operand)
        return None if inner is None else (-inner if isinstance(node.op, ast.USub)
                                           else inner)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    return None


def _setting_text(key: str, node) -> str | None:
    """A setting's value as the line form writes it, or None to clear it."""
    if isinstance(node, ast.Constant) and node.value is None:
        return None
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return "true" if node.value else "false"
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value.strip() or None
    number = _number_of(node)
    if number is not None:
        return repr(number)
    if isinstance(node, ast.Name):                   # cmap = viridis, norm = log
        return node.id
    if isinstance(node, (ast.Tuple, ast.List)):
        parts = [_number_of(e) for e in node.elts]
        if None in parts:
            _fail(node, f"{key} takes numbers")
        return ", ".join(repr(v) for v in parts)
    _fail(node, f"{key} takes a number, a name or a quoted string")


class _Fields:
    """Expressions over the file's fields, with the names a script defined."""

    def __init__(self, shown: str | None = None):
        self.defs: dict[str, tuple[str, dict]] = {}
        self.used: set[str] = set()
        #: the field on screen: what `var` means where no `var = ...` line says
        self.shown = shown

    def read(self, node) -> tuple[str, dict]:
        """`node` as (the expression the derive box takes, {time, level})."""
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            how = node.func.attr
            if how not in ("isel", "sel"):
                _fail(node, f".{how}() is not understood; .isel() and .sel() are")
            text, picked = self.read(node.func.value)
            if node.args:
                _fail(node, f".{how}() takes names: .{how}(time=0)")
            for kw in node.keywords:
                if kw.arg not in ("time", "level"):
                    _fail(node, f".{how}() takes time and level, not {kw.arg}")
                if kw.arg in picked:
                    _fail(node, f"{kw.arg} is picked twice")
                picked[kw.arg] = self._pick(how, kw)
            return text, picked
        if isinstance(node, ast.Name) and node.id in self.defs:
            self.used.add(node.id)
            text, picked = self.defs[node.id]
            return text, dict(picked)
        if isinstance(node, ast.Name) and node.id == "var" and self.shown:
            return self.shown, {}
        picked: dict = {}
        return ast.unparse(self._plain(node, picked)), picked

    def _pick(self, how: str, kw):
        value = kw.value
        number = _number_of(value)
        if how == "isel":
            if type(number) is not int:
                _fail(value, f".isel({kw.arg}=) takes a whole number, counted from 0")
            return number
        if kw.arg == "level":
            if number is None:
                _fail(value, ".sel(level=) takes the level's own value: "
                             ".sel(level=850)")
            return {"value": float(number)}       # its number is the file's to say
        if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
            _fail(value, '.sel(time=) takes a step\'s label in quotes: '
                         '.sel(time="2012-03-01 12:00")')
        return value.value

    def _plain(self, node, picked: dict):
        """`node` with defined names written out and what its fields pick
        gathered into `picked`, refusing what is not arithmetic over fields.

        `t.isel(time=0) - 273.15` and `(t - 273.15).isel(time=0)` are the same
        field: the picks belong to the whole expression, so they must agree.
        """
        def gather(inner, found: dict):
            for key, value in found.items():
                if picked.setdefault(key, value) != value:
                    _fail(node, f"the fields of one expression are picked at one "
                                f"{key}, not {picked[key]!r} and {value!r}")
            return ast.parse(inner, mode="eval").body

        if isinstance(node, ast.Name):
            if node.id in self.defs:
                self.used.add(node.id)
                return gather(*self.defs[node.id])
            if node.id == "var" and self.shown:
                return gather(self.shown, {})
            return node
        if _number_of(node) is not None:
            return node
        if isinstance(node, ast.BinOp) and \
                isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            node.left = self._plain(node.left, picked)
            node.right = self._plain(node.right, picked)
            return node
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and not node.keywords:
            if node.func.id not in _FUNCS:
                _fail(node, f"{node.func.id}() is not a function here"
                            f"{_near(node.func.id, _FUNCS)}; "
                            f"{', '.join(f + '()' for f in _FUNCS)} are")
            node.args = [self._plain(a, picked) for a in node.args]
            return node
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            return gather(*self.read(node))
        _fail(node, "not understood; a field, a number, + - * /, hypot(a, b) "
                    "or diff(a)")


def _literal(node):
    """A layer option as written: a number, text, True/False/None, or a list
    of numbers and text."""
    if isinstance(node, ast.Constant) and \
            (node.value is None or isinstance(node.value, (bool, str))):
        return node.value
    number = _number_of(node)
    if number is not None:
        return number
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_literal(e) for e in node.elts]
    _fail(node, "an option takes a number, quoted text, True, False or a list")


def _option_of(kind: str, name: str, node):
    """A layer's option as `gmpas.layers` takes it: one of its own listed
    options takes a list as its form writes one -- 0, 5, 10 -- and an option
    passed on to matplotlib keeps the list."""
    value = _literal(node)
    listed = name in _layers.LAYER_KINDS.get(kind, {}).get("options", {})
    if isinstance(value, list) and name != "levels" and (listed or kind == "figure"):
        return ", ".join(str(v) for v in value)
    return value                # levels=[28] is a line at 28; levels=28 is a count


def _stack(draws: list, fields: _Fields) -> tuple[dict, dict]:
    """Several draw calls as a figure of layers: (the stack, {var, time})."""
    layers, times, first, figure = [], {}, None, {}
    for verb, call in draws:
        if verb == "figure" or verb in _FEATURE:
            if call.args:
                _fail(call, f"{verb}() takes options by name: {verb}(color=\"gray\")")
            options = {kw.arg: _option_of(verb, kw.arg, kw.value)
                       for kw in call.keywords}
            if verb == "figure":
                figure.update(options)
            else:
                layers.append({"kind": verb, "visible": True, "opacity": 1,
                               "level": "follow", "options": options})
            continue
        vector = verb in _VECTOR
        if not vector and verb not in _AS_LAYER:
            _fail(call, f"{verb}() is a plot of its own and cannot be layered")
        want = 2 if vector else 1
        if len(call.args) != want:
            _fail(call, f"{verb}() takes u and v: {verb}(u, v)" if vector
                  else f"{verb}() takes one field")
        layer = {"kind": _VECTOR[verb] if vector else _AS_LAYER[verb],
                 "visible": True, "opacity": 1, "level": "follow", "options": {}}
        levels = []
        for need, arg in zip(("u", "v") if vector else ("var",), call.args, strict=True):
            name, picked = fields.read(arg)
            layer[need] = name
            # the page's own variable is a field of the file: the first named
            words = [w for w in re.findall(r"[A-Za-z_]\w*", name) if w not in _FUNCS]
            first = first or (words[0] if words else None)
            if "time" in picked:
                times[picked["time"]] = call
            levels.append(picked.get("level"))
        if any(lev != levels[0] for lev in levels):
            _fail(call, "u and v are drawn at one level")
        if levels[0] is not None:
            layer["level"] = levels[0]
        for kw in call.keywords:
            layer["options"][kw.arg] = _option_of(layer["kind"], kw.arg, kw.value)
        layers.append(layer)
    if len(times) > 1:
        _fail(list(times.values())[-1], "a figure is of one time step; "
                                        "every layer must pick the same")
    for kind in ("coastlines", "gridlines"):         # there unless drawn already
        if not any(layer["kind"] == kind for layer in layers):
            layers.append({"kind": kind, "visible": True, "opacity": 1,
                           "level": "follow", "options": {}})
    extra = {"var": first} if first else {}     # none: the map's own layers only
    if times:
        extra["time"] = next(iter(times))
    return {"figure": figure, "layers": layers}, extra


def _parse_python(text: str, shown: str | None = None) -> dict:
    """The Python form: definitions, settings, view() and what draws."""
    # the script has no blocks, so how far a line is indented means nothing
    text, plain = _split(text)
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        shown = (exc.text or "").strip()
        raise ScriptError(f"line {exc.lineno}: {exc.msg}"
                          + (f": {shown}" if shown else "")) from None
    fields = _Fields(shown)
    out: dict = {}
    colour: dict = {}
    where: dict[str, int] = {}
    defined: dict[str, int] = {}
    draws: list = []

    lines: set[str] = set()          # keys set by a plain `key = value` line

    def put(key: str, node, line: int, written=None, call=True, clear=False,
            ready=None):
        # A call that draws says what it draws; a plain line saying the same
        # thing another way gives way to it, in whichever order they come.
        # Two lines, or two calls, that disagree are still a mistake.
        if key in where:
            if call and key in lines:
                lines.discard(key)
            elif not call and key not in lines:
                return
            else:
                raise ScriptError(f"line {line}: {key} is already set on line "
                                  f"{where[key]}")
        if not call:
            lines.add(key)
        where[key] = line
        if ready is not None:
            out[key] = ready
            return
        value = None if clear else \
            _setting_text(key, node) if written is None else written
        if key in _colour.OPTIONS:
            colour[key] = value
            return
        try:
            out[key] = _value(key, value) if value is not None else None
        except ValueError as exc:
            raise ScriptError(f"line {line}: {exc}") from None

    for line, key, value in plain:
        put(key, None, line, value, call=False, clear=value is None)

    def field(node, line: int, call=True):
        expr, picked = fields.read(node)
        put("var", node, line, expr, call)
        for key, value in picked.items():
            if isinstance(value, dict):
                put(key, node, line, call=call, ready=value)
            else:
                put(key, node, line, str(value), call)

    for stmt in tree.body:
        line = stmt.lineno
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
                and isinstance(stmt.targets[0], ast.Name):
            name = stmt.targets[0].id
            if name == "var":
                field(stmt.value, line, call=False)
                fields.defs["var"] = fields.read(stmt.value)   # `var` names it after
            elif _is_key(name):
                put(name, stmt.value, line, call=False)
            elif name in (*_DRAW, *_LATER, "view", "mark"):
                _fail(stmt, f"{name} is a word of the script and cannot be defined")
            else:
                fields.defs[name] = fields.read(stmt.value)
                defined[name] = line
            continue
        call = stmt.value if isinstance(stmt, ast.Expr) else None
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) \
                and call.func.attr in (*_STATS, "mean", "sum", "std", "median"):
            if call.func.attr not in _STATS:
                _fail(call, f".{call.func.attr}() is not in the script yet; "
                            f".max() and .min() are")
            if call.args or call.keywords:
                _fail(call, f".{call.func.attr}() with nothing in the brackets asks "
                            f"where a field is largest or smallest in view; over "
                            f"time or another axis is not in the script yet")
            if "ask" in out:
                _fail(call, "one question to a script")
            expr, picked = fields.read(call.func.value)
            # a question is about a field, not a change of what is shown
            out["ask"] = {"what": _STATS[call.func.attr], "var": expr, **picked}
            continue
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)):
            _fail(stmt, "expected name = value, view(...) or a call that draws, "
                        "such as shade(...)")
        verb = call.func.id
        if verb == "mark":
            # a question whose answer is also marked on the map, or a place
            arg = call.args[0] if len(call.args) == 1 and not call.keywords else None
            if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute) \
                    and arg.func.attr in _STATS and not arg.args and not arg.keywords:
                if "ask" in out:
                    _fail(call, "one question to a script")
                expr, picked = fields.read(arg.func.value)
                out["ask"] = {"what": _STATS[arg.func.attr], "var": expr, **picked,
                              "mark": True}
            elif not call.args and {kw.arg for kw in call.keywords} == {"lon", "lat"}:
                at = {kw.arg: _number_of(kw.value) for kw in call.keywords}
                if None in at.values():
                    _fail(call, "mark() takes numbers: mark(lon=147.4, lat=-2)")
                put("point", call, line, f"{at['lon']!r}, {at['lat']!r}")
                out["panel"] = "mark"
            else:
                _fail(call, "mark() takes a question or a place: mark(sst.max()) "
                            "or mark(lon=147.4, lat=-2)")
            continue
        if verb == "view":
            if call.args:
                _fail(call, "view() takes names: view(region=(90, 150, -15, 15))")
            for kw in call.keywords:
                if kw.arg not in _VIEW_KEYS:
                    _fail(call, f"view() has no {kw.arg}{_near(kw.arg or '', _VIEW_KEYS)}; "
                                f"it takes {', '.join(_VIEW_KEYS)}")
                put(_VIEW_KEYS[kw.arg], kw.value, line)
        elif verb in _DRAW:
            draws.append((verb, call))
        elif verb in _LATER:
            _fail(call, f"{verb}() is not in the script yet")
        else:
            words = ["view", "mark", *_DRAW, *_LATER]
            if verb in _FUNCS:
                _fail(call, f"{verb}() makes a field; draw it: shade({verb}(...))")
            _fail(call, f"{verb}() is not understood{_near(verb, words)}")
    # one call is the page's own plot of that kind -- unless it is given an
    # option only a layer has, and then it is a figure of that one layer
    own = ("cmap", "vmin", "vmax", "lon", "lat", *_colour.OPTIONS)
    alone = len(draws) == 1 and draws[0][0] in _FIELD and not (
        draws[0][0] in _AS_LAYER and draws[0][0] != "shade"
        and any(kw.arg not in own for kw in draws[0][1].keywords))
    # A plain `var = x` line names the shaded field. Beside calls that shade
    # nothing -- contour lines, winds -- it is what they are drawn over.
    under = out.get("var") if "var" in lines else None
    if alone and under and draws[0][0] == "contour" \
            and len(draws[0][1].args) == 1 \
            and fields.read(draws[0][1].args[0])[0] != under:
        alone = False
    if alone:
        verb, call = draws[0]
        line = call.lineno
        if len(call.args) != 1:
            _fail(call, f"{verb}() takes one field: {verb}(t2m)")
        field(call.args[0], line)
        put("kind", call, line, _FIELD[verb])
        point = {}
        for kw in call.keywords:
            if verb in _AT_POINT and kw.arg in ("lon", "lat"):
                point[kw.arg] = _number_of(kw.value)
                if point[kw.arg] is None:
                    _fail(call, f"{kw.arg} takes a number")
                continue
            if kw.arg not in ("cmap", "vmin", "vmax") and kw.arg not in _colour.OPTIONS:
                choices = ["cmap", "vmin", "vmax", *_colour.OPTIONS]
                _fail(call, f"{verb}() has no {kw.arg}{_near(kw.arg or '', choices)}")
            put(kw.arg, kw.value, line)
        if point:
            if len(point) != 2:
                _fail(call, f"{verb}() takes both lon and lat")
            put("point", call, line, f"{point['lon']!r}, {point['lat']!r}")
    elif draws:
        stack, extra = _stack(draws, fields)
        shaded = ("pcolormesh", "contourf", "imshow")
        if under and not any(x["kind"] in shaded for x in stack["layers"]):
            at = next(i for i, x in enumerate(stack["layers"])
                      if x["kind"] not in ("land", "ocean"))
            stack["layers"].insert(at, {"kind": "pcolormesh", "var": under,
                                        "visible": True, "opacity": 1,
                                        "level": "follow", "options": {}})
            extra["var"] = next(w for w in re.findall(r"[A-Za-z_]\w*", under)
                                if w not in _FUNCS)
        line = draws[0][1].lineno
        put("kind", draws[0][1], line, "layers")
        for key, value in extra.items():
            put(key, draws[0][1], line, str(value))
        out["layers"] = stack
    for name, line in defined.items():
        if name not in fields.used:
            raise _Unused(f"line {line}: {name} is set and never used"
                          f"{_near(name, keys())}")
    return _finish(out, colour, where)


_SHADED = ("pcolormesh", "contourf", "imshow")
_NO_FIELD = re.compile(r"^[ \t]*var[ \t]*=[ \t]*$", re.M)


def underlay(text: str, parsed: dict, shown: str | None) -> str | None:
    """`text` with the field on screen named as what it draws over, or None
    when it needs nothing added.

    A script changes only what it names, so contour lines, winds or rivers
    written alone are drawn over the field already shown: this says so in the
    script itself, with a `var = ...` line, so that what is applied and what
    is read are the same text. A script that shades a field of its own, or
    says `var =` -- nothing underneath -- is left as it is.
    """
    if not shown or parsed.get("kind") != "layers" or _NO_FIELD.search(text):
        return None
    layers = (parsed.get("layers") or {}).get("layers", [])
    if any(layer.get("kind") in _SHADED for layer in layers):
        return None
    return f"var = {shown}\n{text}"


def read(text: str, meta: dict, shown: str | None = None) -> tuple[str, dict]:
    """A script checked against one open file and the view on screen:
    (the text as applied, what `resolve` made of it)."""
    parsed = parse(text, shown)
    more = underlay(text, parsed, shown)
    if more is not None:
        try:
            return more, resolve(parse(more, shown), meta)
        except ScriptError:
            pass                  # what is shown cannot lie under a figure here
    return text, resolve(parsed, meta)


#: What the plain lon/lat map is also called.
_LONLAT_NAMES = {"lonlat", "latlon", "platecarree", "equirectangular", "flat", "plain"}


def _projection(name, offered: list[str]):
    """A projection as this page names it: its own spelling whatever the case,
    and the lon/lat map by any of its names."""
    if not isinstance(name, str):
        return name
    key = re.sub(r"[\s_/-]", "", name).lower()
    if key in _LONLAT_NAMES:
        return LONLAT
    return {p.lower(): p for p in offered}.get(key, name)


def _level_index(lev, expr, variables: dict):
    """A level given by its value as its number, where the file lists them."""
    if not isinstance(lev, dict):
        return lev
    words = [expr] if expr in variables else re.findall(r"[A-Za-z_]\w*", expr or "")
    owner = next((w for w in words if (variables.get(w) or {}).get("level_values")), None)
    if owner is None:
        raise ScriptError(f"level: {expr or 'this field'} lists no level values in "
                          f"this file; pick by number: .isel(level=0)")
    values, want = variables[owner]["level_values"], lev["value"]
    for i, value in enumerate(values):
        if abs(value - want) <= 1e-6 * max(1.0, abs(want)):
            return i
    units = variables[owner].get("level_units") or ""
    raise ScriptError(f"level: {want:g} is not one of {owner}'s levels: "
                      f"{', '.join(f'{v:g}' for v in values)} {units}".rstrip())


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
                words = re.findall(r"[A-Za-z_]\w*", state["var"])
                hint = next((h for h in (_near(w, variables) for w in words
                                         if w not in variables and w not in _FUNCS)
                             if h), "")
                raise ScriptError(f"var: {exc.args[0]}{hint}") from None
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
        lev = out["level"] = _level_index(state["level"], out.get("var"), variables)
        if lev < 0 or (levels is not None and lev >= levels):
            raise ScriptError(f"level: {lev} is outside {out['var']}'s "
                              f"{levels} level(s)" if levels is not None
                              else f"level: {lev} is below 0")
    if state.get("ask") is not None:
        ask = dict(state["ask"])
        name = _derive.resolve(ask["var"], variables)
        if name is None:
            try:
                _derive.parse(ask["var"], variables)
            except KeyError as exc:
                raise ScriptError(f"{ask['what']}: {exc.args[0]}"
                                  f"{_near(ask['var'], variables)}") from None
        else:
            ask["var"] = name
        steps, labels = meta.get("steps", 0), meta.get("labels", [])
        t = ask.get("time")
        if isinstance(t, str):
            if t not in labels:
                raise ScriptError(f"time: no step is labelled {t!r}")
            ask["time"] = labels.index(t)
        elif t is not None:
            if not -steps <= t < steps:
                raise ScriptError(f"time: step {t} is outside this run's "
                                  f"{steps} step(s)")
            ask["time"] = t % steps
        count = variables[name].get("levels", 1) if name in variables else None
        lev = ask.get("level")
        if lev is not None:
            lev = ask["level"] = _level_index(lev, ask["var"], variables)
        if lev is not None and (lev < 0 or (count is not None and lev >= count)):
            raise ScriptError(f"level: {lev} is outside {ask['var']}'s level(s)")
        out["ask"] = ask
    proj = _projection(state.get("proj"), meta.get("projections") or [])
    kind = state.get("kind")
    if kind is not None:
        named = out.get("var")
        offered = variables[named].get("kinds") if named in variables else None
        if offered is None:
            offered = list(meta.get("kind_labels") or ["map"])
            if named is not None and named not in variables:
                offered = ["map"]                # a derived field is the map only
        if kind == "series" and meta.get("steps", 0) <= 1:
            raise ScriptError("series: this file has one time step, so there is "
                              "nothing to plot against time")
        if kind == "profile" and named in variables \
                and variables[named].get("levels", 1) <= 1:
            raise ScriptError(f"profile: {named} has one level, so there is no "
                              f"column to plot")
        if kind in _AT_POINT and kind not in offered and meta.get("point_panel"):
            # this page reads a point's series and column in its point panel,
            # over the map, for any field
            if state.get("point") is None:
                raise ScriptError(f"{kind}: say where: "
                                  f"{kind}({named or 'field'}, lon=147.4, lat=-2.0)")
            out["panel"] = kind
            kind = out["kind"] = "map"
        if kind not in offered and kind in _AS_LAYER.values() and "layers" in offered:
            # a run with no contourf plot of its own draws one as a single layer
            keep = ("vmin", "vmax") if kind == "contour" else ("cmap", "vmin", "vmax")
            options = {k: state[k] for k in keep if state.get(k) is not None}
            lev = state.get("level")
            out["layers"] = {"figure": {}, "layers": [
                {"kind": kind, "var": named, "visible": True, "opacity": 1,
                 "level": "follow" if lev is None else lev, "options": options},
                *({"kind": k, "visible": True, "opacity": 1, "level": "follow",
                   "options": {}} for k in ("coastlines", "gridlines"))]}
            kind = out["kind"] = "layers"
        if kind not in offered:
            what = f"{named} has" if named else "this page has"
            raise ScriptError(f"kind: {what} no {kind!r} plot{_near(kind, offered)}; "
                              f"one of {', '.join(offered)}")
        if kind == "layers" and proj not in (None, LONLAT) \
                and (proj == "auto" or proj in _layers.PROJECTIONS):
            # a figure of layers has its own projection: the one named for
            # the map is the one meant for the figure, and the page's own map
            # stays the plain one underneath. "auto" is left to the figure.
            stack = out["layers"] = copy.deepcopy(out.get("layers") or state["layers"])
            if proj != "auto":
                stack["figure"].setdefault("projection", proj)
                for key, name in (("plon", "central_longitude"),
                                  ("plat", "central_latitude")):
                    if out.get(key) is not None:
                        stack["figure"].setdefault(name, out[key])
            for key in ("plon", "plat", "box"):
                out.pop(key, None)
            proj = LONLAT
        if kind != "map" and proj not in (None, LONLAT):
            raise ScriptError(f"kind: a {kind} plot is drawn on the {LONLAT} map; "
                              f"leave proj out or set proj = {LONLAT}")
    if out.get("layers") is not None:
        if not meta.get("layer_schema"):
            raise ScriptError("layers: this page draws one field at a time")
        spatial = [v["name"] for v in meta.get("variables", []) if v.get("spatial")]
        counts = {v["name"]: v.get("levels", 1) for v in meta.get("variables", [])}

        def derived(name: str) -> bool:
            try:
                _derive.parse(name, spatial)
            except KeyError:
                return False
            return True
        stack = out["layers"] = copy.deepcopy(out["layers"])
        if not any(_layers.LAYER_KINDS.get(layer.get("kind"), {}).get("needs")
                   for layer in stack["layers"]):
            raise ScriptError("layers: a figure draws a field, and this one names "
                              "none; say which lies underneath with a line: var = ...")
        for layer in stack["layers"]:
            if isinstance(layer.get("level"), dict):
                layer["level"] = _level_index(
                    layer["level"], layer.get("var") or layer.get("u"), variables)
        try:
            _layers.clean(stack, spatial, counts,
                          derived if meta.get("layer_expressions") else None)
        except ValueError as exc:
            raise ScriptError(f"layers: {exc}") from None
    if state.get("cmap") is not None and state["cmap"] not in meta.get("cmaps", []):
        # matplotlib's own way of saying "reversed": viridis_r is viridis with
        # reverse set, and RdBu is the page's RdBu_r the same way
        cmap, cmaps = state["cmap"], {c.lower(): c for c in meta.get("cmaps", [])}
        other = cmap[:-2] if cmap.endswith("_r") else cmap + "_r"
        if cmap.lower() in cmaps:
            out["cmap"] = cmaps[cmap.lower()]
        elif other.lower() in cmaps and meta.get("colour_options"):
            out["cmap"] = cmaps[other.lower()]
            colour = dict(out.get("colour") or {})
            # the other way round from whatever `reverse` already says
            colour["reverse"] = None if colour.get("reverse") else True
            out["colour"] = colour
        else:
            raise ScriptError(f"cmap: {cmap!r} is not one of this page's palettes"
                              f"{_near(cmap, meta.get('cmaps', []))}")
    if "proj" in state:
        offered = meta.get("projections") or []
        name = out["proj"] = proj
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
