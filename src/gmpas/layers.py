"""Composite maps for `--generic`: a stack of layers, bottom to top.

A layer is one thing drawn on the map -- a filled contour of one field, contour
lines of another at a different level, wind barbs, coastlines -- with its own
visibility, opacity and matplotlib options. The browser keeps the stack and
sends it with every request; this module checks it and draws it.

The options are matplotlib's own keyword arguments, but only the ones listed
in `LAYER_KINDS`, each coerced to its declared type. The stack arrives from the
network (`--host 0.0.0.0` on HPC), so nothing in it is ever evaluated, looked
up by attribute, or passed through unchecked: an unknown option is refused by
name, and a colormap, a colour or a label format is validated before
matplotlib sees it.

Opt-in by design: nothing here runs unless the "layers" plot is picked.
"""

from __future__ import annotations

import json
import math
import re

import numpy as np

from . import data as _data
from . import palettes

MAX_LAYERS = 24
MAX_LEVELS = 256
MAX_TEXT = 200

# ------------------------------------------------------------------ schema
#
# One entry per option: its type, its default (None: matplotlib's/xarray's
# own), and for choices the allowed values. The same table drives validation
# here and the option forms in the browser, so the two cannot drift apart.

_COLOUR_SCALE = {
    "cmap": {"type": "cmap", "default": "viridis",
             "help": "matplotlib, cmo.*, ferret.* or grads.*; empty follows the look"},
    "reverse": {"type": "bool", "default": False, "help": "run the palette backwards"},
    "vmin": {"type": "float", "default": None},
    "vmax": {"type": "float", "default": None},
    "robust": {"type": "bool", "default": False,
               "help": "colour range from the 2nd-98th percentile"},
    "center": {"type": "float", "default": None,
               "help": "diverging colormap centred on this value"},
    "norm": {"type": "choice", "default": "linear",
             "choices": ["linear", "log", "symlog", "power"]},
    "linthresh": {"type": "float", "default": 1.0, "help": "symlog linear range"},
    "gamma": {"type": "float", "default": 1.0, "min": 0.01, "max": 100.0,
              "help": "power norm: below 1 spreads low values, above 1 high ones "
                      "(ncview's Low/Hi)"},
    "extend": {"type": "choice", "default": "neither",
               "choices": ["neither", "both", "min", "max"],
               "help": "colorbar triangles for values outside the range"},
    "under_color": {"type": "color", "default": None,
                    "help": "values below the range; empty is the palette's end"},
    "over_color": {"type": "color", "default": None,
                   "help": "values above the range; empty is the palette's end"},
    "colorbar": {"type": "bool", "default": True},
    "colorbar_label": {"type": "str", "default": None},
    "colorbar_format": {"type": "fmt", "default": None, "help": "tick labels, e.g. %.0f"},
    "colorbar_ticks": {"type": "levels", "default": None,
                       "help": "a count (6) or the tick values"},
}

#: colour options only pixel layers have: contourf already bands by `levels`,
#: and leaves missing cells unfilled rather than colouring them
_RASTER_COLOUR = {
    "bands": {"type": "int", "default": None, "min": 2, "max": 256,
              "help": "N equal colour steps instead of a continuous scale"},
    "missing_color": {"type": "color", "default": None,
                      "help": "missing cells; empty leaves them transparent"},
}

_LINESTYLES = ["solid", "dashed", "dashdot", "dotted"]

# Levels at a fixed step from a reference value -- MSLP every 4 hPa from 1000,
# heights every 60 m -- which is how a weather map is specified and how Metview
# (contour_interval, contour_reference_level) and GrADS (set cint) say it. The
# alternative already here, a count, moves every line as soon as the field's
# range moves, so two frames of a GIF are not drawn at the same values.
_CONTOUR_INTERVAL = {
    "interval": {"type": "float", "default": None, "min": 0.0,
                 "help": "levels every this much, instead of a count"},
    "reference": {"type": "float", "default": 0.0,
                  "help": "interval counts from this value, e.g. 1000 hPa"},
    "min_level": {"type": "float", "default": None, "help": "lowest interval level"},
    "max_level": {"type": "float", "default": None, "help": "highest interval level"},
}

# Every Nth line drawn heavier, and labels thinned to match: dense contour sets
# are read by their emphasised lines (Metview contour_highlight*).
_CONTOUR_HIGHLIGHT = {
    "highlight_every": {"type": "int", "default": 0, "min": 0, "max": MAX_LEVELS,
                        "help": "emphasise every Nth level; 0 is off"},
    "highlight_color": {"type": "color", "default": None,
                        "help": "colour of the emphasised lines"},
    "highlight_linewidth": {"type": "float", "default": 2.0, "min": 0.0},
    "label_every": {"type": "int", "default": 1, "min": 1, "max": MAX_LEVELS,
                    "help": "label only every Nth level"},
}
_RESOLUTIONS = ["110m", "50m", "10m"]

LAYER_KINDS: dict[str, dict] = {
    # -- fields ------------------------------------------------------------
    "contourf": {
        "label": "filled contour", "group": "field", "needs": ["var"],
        "options": {**_COLOUR_SCALE,
                    "levels": {"type": "levels", "default": None,
                               "help": "a count (10) or the values (0, 5, 10)"},
                    **_CONTOUR_INTERVAL,
                    "hatches": {"type": "hatches", "default": None,
                                "help": "one pattern per band, from "
                                        "/ \\ | - + x o O . *"}},
    },
    "contour": {
        "label": "contour lines", "group": "field", "needs": ["var"],
        "options": {"levels": {"type": "levels", "default": None},
                    **_CONTOUR_INTERVAL,
                    "colors": {"type": "colors", "default": "black",
                               "help": "one colour, or one per level; ignored with a cmap"},
                    "cmap": {"type": "cmap", "default": None},
                    "vmin": {"type": "float", "default": None},
                    "vmax": {"type": "float", "default": None},
                    "linewidths": {"type": "float", "default": 1.0, "min": 0.0},
                    "linestyles": {"type": "choice", "default": "solid",
                                   "choices": _LINESTYLES},
                    "negative_linestyles": {"type": "choice", "default": "dashed",
                                            "choices": _LINESTYLES},
                    "labels": {"type": "bool", "default": True},
                    "label_fontsize": {"type": "float", "default": 8.0, "min": 1.0},
                    "label_fmt": {"type": "fmt", "default": "%g"},
                    **_CONTOUR_HIGHLIGHT,
                    "colorbar": {"type": "bool", "default": False}},
    },
    "pcolormesh": {
        "label": "pcolormesh", "group": "field", "needs": ["var"],
        "options": {**_COLOUR_SCALE, **_RASTER_COLOUR,
                    "shading": {"type": "choice", "default": "auto",
                                "choices": ["auto", "nearest", "gouraud"]}},
    },
    "imshow": {
        "label": "imshow", "group": "field", "needs": ["var"],
        "options": {**_COLOUR_SCALE, **_RASTER_COLOUR,
                    "interpolation": {"type": "choice", "default": "nearest",
                                      "choices": ["nearest", "bilinear", "bicubic"]}},
    },
    # -- vectors -----------------------------------------------------------
    "quiver": {
        "label": "arrows (quiver)", "group": "vector", "needs": ["u", "v"],
        "options": {"stride": {"type": "int", "default": 0, "min": 0,
                               "help": "every Nth point; 0 picks ~30 across"},
                    "color": {"type": "color", "default": "black"},
                    "colorize": {"type": "bool", "default": False,
                                 "help": "colour arrows by speed"},
                    "cmap": {"type": "cmap", "default": "viridis"},
                    "scale": {"type": "float", "default": None, "min": 0.0,
                              "help": "data units per arrow length; empty is automatic"},
                    "width": {"type": "float", "default": None, "min": 0.0},
                    "headwidth": {"type": "float", "default": 3.0, "min": 0.0},
                    "key": {"type": "bool", "default": True, "help": "reference arrow"},
                    "key_value": {"type": "float", "default": None,
                                  "help": "reference arrow length; empty picks one"},
                    "colorbar": {"type": "bool", "default": True}},
    },
    "barbs": {
        "label": "wind barbs", "group": "vector", "needs": ["u", "v"],
        "options": {"stride": {"type": "int", "default": 0, "min": 0},
                    "color": {"type": "color", "default": "black"},
                    "length": {"type": "float", "default": 5.0, "min": 1.0},
                    "linewidth": {"type": "float", "default": 0.6, "min": 0.0}},
    },
    "streamplot": {
        "label": "streamlines", "group": "vector", "needs": ["u", "v"],
        "options": {"density": {"type": "float", "default": 1.5, "min": 0.1, "max": 8.0},
                    "color": {"type": "color", "default": "black"},
                    "colorize": {"type": "bool", "default": False},
                    "cmap": {"type": "cmap", "default": "viridis"},
                    "linewidth": {"type": "float", "default": 0.8, "min": 0.0},
                    "arrowsize": {"type": "float", "default": 1.0, "min": 0.0},
                    "colorbar": {"type": "bool", "default": True}},
    },
    # -- map features --------------------------------------------------------
    "coastlines": {
        "label": "coastlines", "group": "feature", "needs": [],
        "options": {"color": {"type": "color", "default": "black"},
                    "linewidth": {"type": "float", "default": 0.7, "min": 0.0},
                    "resolution": {"type": "choice", "default": "110m",
                                   "choices": _RESOLUTIONS}},
    },
    "borders": {
        "label": "country borders", "group": "feature", "needs": [],
        "options": {"color": {"type": "color", "default": "#444444"},
                    "linewidth": {"type": "float", "default": 0.5, "min": 0.0},
                    "linestyle": {"type": "choice", "default": "solid",
                                  "choices": _LINESTYLES},
                    "resolution": {"type": "choice", "default": "110m",
                                   "choices": _RESOLUTIONS}},
    },
    "land": {
        "label": "land fill", "group": "feature", "needs": [],
        "options": {"facecolor": {"type": "color", "default": "#e8e2d0"},
                    "resolution": {"type": "choice", "default": "110m",
                                   "choices": _RESOLUTIONS}},
    },
    "ocean": {
        "label": "ocean fill", "group": "feature", "needs": [],
        "options": {"facecolor": {"type": "color", "default": "#cfe3f0"},
                    "resolution": {"type": "choice", "default": "110m",
                                   "choices": _RESOLUTIONS}},
    },
    "lakes": {
        "label": "lakes", "group": "feature", "needs": [],
        "options": {"facecolor": {"type": "color", "default": "#cfe3f0"},
                    "edgecolor": {"type": "color", "default": "#555555"},
                    "resolution": {"type": "choice", "default": "110m",
                                   "choices": _RESOLUTIONS}},
    },
    "rivers": {
        "label": "rivers", "group": "feature", "needs": [],
        "options": {"color": {"type": "color", "default": "#3a7bbf"},
                    "linewidth": {"type": "float", "default": 0.5, "min": 0.0},
                    "resolution": {"type": "choice", "default": "110m",
                                   "choices": _RESOLUTIONS}},
    },
    "gridlines": {
        "label": "gridlines", "group": "feature", "needs": [],
        "options": {"labels": {"type": "bool", "default": True},
                    "color": {"type": "color", "default": "gray"},
                    "linewidth": {"type": "float", "default": 0.4, "min": 0.0},
                    "linestyle": {"type": "choice", "default": "dashed",
                                  "choices": _LINESTYLES},
                    "dlon": {"type": "float", "default": None, "min": 0.0,
                             "help": "degrees between meridians; empty is automatic"},
                    "dlat": {"type": "float", "default": None, "min": 0.0}},
    },
}

#: (category, Natural Earth name) behind each feature layer
_NATURAL_EARTH = {
    "coastlines": ("physical", "coastline"),
    "borders": ("cultural", "admin_0_boundary_lines_land"),
    "land": ("physical", "land"),
    "ocean": ("physical", "ocean"),
    "lakes": ("physical", "lakes"),
    "rivers": ("physical", "rivers_lake_centerlines"),
}

#: projections that show the whole globe, or a whole hemisphere: they draw the
#: whole grid, not the view box, which would leave most of the globe empty
_WHOLE_GRID = {"Robinson", "Mollweide", "EqualEarth", "Orthographic",
               "NorthPolarStereo", "SouthPolarStereo"}

PROJECTIONS = ["PlateCarree", "Robinson", "Mollweide", "EqualEarth", "Orthographic",
               "NorthPolarStereo", "SouthPolarStereo", "LambertConformal", "Mercator"]

FIGURE_OPTIONS = {
    "projection": {"type": "choice", "default": "PlateCarree", "choices": PROJECTIONS},
    "central_longitude": {"type": "float", "default": None,
                          "help": "empty centres on the view"},
    "central_latitude": {"type": "float", "default": None,
                         "help": "Orthographic and LambertConformal"},
    "extent": {"type": "choice", "default": "view", "choices": ["view", "global"]},
    "title": {"type": "str", "default": None, "help": "empty writes one"},
    "subtitle": {"type": "str", "default": None},
    "footnote_left": {"type": "str", "default": None, "help": "e.g. the data source"},
    "footnote_right": {"type": "str", "default": None, "help": "e.g. the run or units"},
    "look": {"type": "choice", "default": "matplotlib",
             "choices": ["matplotlib", "grads", "ferret"],
             "help": "GrADS or Ferret: their default palette and colour key"},
    "colorbar_style": {"type": "choice", "default": None,
                       "choices": ["matplotlib", "grads", "ferret"],
                       "help": "empty follows the look"},
    "colorbar": {"type": "choice", "default": "bottom", "choices": ["bottom", "right"],
                 "help": "empty follows the colorbar style"},
}

#: what a look changes when the stack does not say otherwise
LOOKS = {
    "matplotlib": {"palette": None, "colorbar_style": "matplotlib"},
    # GrADS draws shaded fields and contour lines in its 13-colour rainbow,
    # with a horizontal bar of boxed colours under the plot (cbarn)
    "grads": {"palette": "grads.rainbow", "colorbar_style": "grads"},
    # Ferret's own default palette, with its key down the right-hand side
    "ferret": {"palette": "ferret.default", "colorbar_style": "ferret"},
}


class Stack(dict):
    """A stack that has been through `clean` -- so it is not cleaned twice.

    Cleaning is not idempotent on its own output (levels and hatches come back
    as lists, which the text parsers do not take), and a GIF redraws the same
    stack once per frame.
    """


def schema() -> dict:
    """What the browser needs to build its forms: kinds, options, defaults."""
    return {"kinds": LAYER_KINDS, "figure": FIGURE_OPTIONS}


def default_stack(var: str) -> dict:
    """The stack a first visit starts from: the field, coastlines, gridlines."""
    return {"figure": {},
            "layers": [{"kind": "contourf", "var": var},
                       {"kind": "coastlines"},
                       {"kind": "gridlines"}]}


# -------------------------------------------------------------- validation

_FMT = re.compile(r"^[^%]{0,24}%[-+ 0#]{0,3}\d{0,2}(\.\d{1,2})?[dfgeE][^%]{0,24}$")
_HATCH = re.compile(r"^[/\\|\-+xoO.*]{0,6}$")


def _coerce(name: str, spec: dict, value, where: str):
    kind = spec["type"]
    if value is None or value == "":
        return None
    try:
        if kind in ("float", "int"):
            if isinstance(value, bool):
                raise ValueError
            out = float(value) if kind == "float" else int(value)
            if not math.isfinite(out):
                raise ValueError
            if "min" in spec and out < spec["min"] or "max" in spec and out > spec["max"]:
                raise ValueError(f"{where}{name}={out} is outside "
                                 f"[{spec.get('min', '-inf')}, {spec.get('max', 'inf')}]")
            return out
        if kind == "bool":
            if isinstance(value, bool):
                return value
            if str(value).lower() in ("true", "1", "yes", "on"):
                return True
            if str(value).lower() in ("false", "0", "no", "off"):
                return False
            raise ValueError
        if kind == "choice":
            if value not in spec["choices"]:
                raise ValueError(f"{where}{name}={value!r} is not one of {spec['choices']}")
            return value
        text = str(value)
        if len(text) > MAX_TEXT:
            raise ValueError(f"{where}{name} is longer than {MAX_TEXT} characters")
        if kind == "str":
            return text
        if kind == "cmap":
            if not palettes.known(text):
                raise ValueError(f"{where}{name}={text!r} is not a known colormap "
                                 f"(matplotlib, cmo.*, ferret.* or grads.*)")
            return text
        if kind == "color":
            from matplotlib.colors import is_color_like
            if not is_color_like(text):
                raise ValueError(f"{where}{name}={text!r} is not a colour")
            return text
        if kind == "colors":
            from matplotlib.colors import is_color_like
            parts = [p.strip() for p in text.split(",") if p.strip()]
            if not parts or not all(is_color_like(p) for p in parts):
                raise ValueError(f"{where}{name}={text!r}: give colours separated "
                                 f"by commas")
            return parts[0] if len(parts) == 1 else parts
        if kind == "fmt":
            if not _FMT.match(text):
                raise ValueError(f"{where}{name}={text!r}: a printf number format such "
                                 f"as %g, %.0f or %.1f hPa")
            return text
        if kind == "hatches":
            parts = [p.strip() for p in text.split(",")][:64]
            if not all(_HATCH.match(p) for p in parts):
                raise ValueError(f"{where}{name}={text!r}: hatch patterns use / \\ | - + "
                                 f"x o O . * only")
            return [p or None for p in parts]
        if kind == "levels":
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values = [value]
            else:
                values = [v for v in re.split(r"[,\s]+", text.strip()) if v]
            numbers = [float(v) for v in values]
            if not numbers or len(numbers) > MAX_LEVELS \
                    or not all(math.isfinite(v) for v in numbers):
                raise ValueError
            if len(numbers) == 1:
                count = int(numbers[0])
                if count != numbers[0] or not 2 <= count <= MAX_LEVELS:
                    raise ValueError(f"{where}{name}: a single value is a count of "
                                     f"levels, 2 to {MAX_LEVELS}")
                return count
            if any(b <= a for a, b in zip(numbers, numbers[1:], strict=False)):
                raise ValueError(f"{where}{name}: level values must increase")
            return numbers
    except ValueError as exc:
        if exc.args:
            raise
        raise ValueError(f"{where}{name}={value!r} is not a valid {kind}") from None
    raise ValueError(f"{where}{name}: unknown option type {kind}")


def _clean_options(given, table: dict, where: str) -> dict:
    if given is None:
        return {}
    if not isinstance(given, dict):
        raise ValueError(f"{where}options must be an object")
    unknown = sorted(set(given) - set(table))
    if unknown:
        raise ValueError(f"{where}unknown option(s) {unknown}; allowed: {sorted(table)}")
    out = {}
    for name, value in given.items():
        coerced = _coerce(name, table[name], value, where)
        if coerced is not None:
            out[name] = coerced
    return out


def clean(stack, spatial_vars, level_counts: dict[str, int]) -> dict:
    """A validated copy of a stack, or ValueError naming what is wrong.

    `spatial_vars` are the variables a layer may draw; `level_counts` gives
    each one's level-axis length, so a pinned level is checked up front rather
    than failing deep inside a read.
    """
    if isinstance(stack, (str, bytes)):
        if len(stack) > 64_000:
            raise ValueError("layer stack is too large")
        try:
            stack = json.loads(stack)
        except json.JSONDecodeError as exc:
            raise ValueError(f"layer stack is not valid JSON: {exc}") from None
    if not isinstance(stack, dict) or not isinstance(stack.get("layers"), list):
        raise ValueError('a layer stack is {"figure": {...}, "layers": [...]}')
    if len(stack["layers"]) > MAX_LAYERS:
        raise ValueError(f"at most {MAX_LAYERS} layers")
    unknown = sorted(set(stack) - {"figure", "layers"})
    if unknown:
        raise ValueError(f"unknown stack key(s) {unknown}")

    figure = _clean_options(stack.get("figure") or {}, FIGURE_OPTIONS, "figure: ")
    out = Stack(figure=figure, layers=[])
    allowed_keys = {"kind", "visible", "opacity", "level", "var", "u", "v", "options",
                    "name", "id"}
    for i, layer in enumerate(stack["layers"]):
        where = f"layer {i + 1}: "
        if not isinstance(layer, dict):
            raise ValueError(f"{where}must be an object")
        extra = sorted(set(layer) - allowed_keys)
        if extra:
            raise ValueError(f"{where}unknown key(s) {extra}")
        kind = layer.get("kind")
        if kind not in LAYER_KINDS:
            raise ValueError(f"{where}kind {kind!r} is not one of {sorted(LAYER_KINDS)}")
        spec = LAYER_KINDS[kind]
        where = f"layer {i + 1} ({kind}): "
        clean_layer = {
            "kind": kind,
            "visible": _coerce("visible", {"type": "bool"}, layer.get("visible", True),
                               where) is not False,
            "opacity": _coerce("opacity", {"type": "float", "min": 0.0, "max": 1.0},
                               layer.get("opacity", 1.0), where),
            "options": _clean_options(layer.get("options"), spec["options"], where),
        }
        if clean_layer["opacity"] is None:
            clean_layer["opacity"] = 1.0
        _check_colour_options(clean_layer["options"], where)
        if kind in ("contour", "contourf"):
            _check_contour_options(clean_layer["options"], where)
        for need in spec["needs"]:
            name = layer.get(need)
            if name not in spatial_vars:
                raise ValueError(f"{where}{need}={name!r} is not a map variable of this "
                                 f"file; one of {sorted(spatial_vars)}")
            clean_layer[need] = name
        level = layer.get("level", "follow")
        if level in (None, "", "follow"):
            clean_layer["level"] = None
        else:
            idx = _coerce("level", {"type": "int", "min": 0}, level, where)
            for need in spec["needs"]:
                n = level_counts.get(clean_layer[need], 1)
                if idx >= n:
                    raise ValueError(f"{where}level {idx} is out of range for "
                                     f"{clean_layer[need]!r} ({n} level(s))")
            clean_layer["level"] = idx
        if isinstance(layer.get("name"), str):
            clean_layer["name"] = layer["name"][:MAX_TEXT]
        out["layers"].append(clean_layer)
    return out


def _check_contour_options(opts: dict, where: str) -> None:
    """Refuse contour levels that contradict each other, by name."""
    if opts.get("interval") is not None:
        if opts.get("levels") is not None:
            raise ValueError(f"{where}interval and levels cannot be combined: interval "
                             f"generates the levels, levels gives them outright")
        if opts["interval"] <= 0:
            raise ValueError(f"{where}interval={opts['interval']:g} must be above zero")
    low, high = opts.get("min_level"), opts.get("max_level")
    if low is not None and high is not None and high <= low:
        raise ValueError(f"{where}max_level={high:g} must be above min_level={low:g}")
    if opts.get("interval") is None:
        for name in ("min_level", "max_level"):
            if opts.get(name) is not None:
                raise ValueError(f"{where}{name} applies to interval levels; "
                                 f"set interval too")


def interval_levels(lo: float, hi: float, interval: float, reference: float = 0.0,
                    min_level=None, max_level=None, where: str = "") -> list[float]:
    """Levels at `reference + k*interval` covering `lo..hi`, inclusive of both.

    The reference is a level itself whenever it falls in range, so 1000 hPa
    every 4 gives ..., 996, 1000, 1004, ... wherever the field happens to sit;
    that is what makes the same line mean the same value in every frame.
    """
    if min_level is not None:
        lo = max(lo, float(min_level))
    if max_level is not None:
        hi = min(hi, float(max_level))
    if hi < lo:
        raise ValueError(f"{where}min_level..max_level leaves nothing to draw between "
                         f"{lo:g} and {hi:g}")
    first = math.ceil((lo - reference) / interval - 1e-9)
    last = math.floor((hi - reference) / interval + 1e-9)
    count = last - first + 1
    if count < 1:
        raise ValueError(f"{where}interval={interval:g} from reference={reference:g} "
                         f"puts no level between {lo:g} and {hi:g}")
    if count > MAX_LEVELS:
        raise ValueError(f"{where}interval={interval:g} would draw {count} levels "
                         f"between {lo:g} and {hi:g}; the limit is {MAX_LEVELS}. "
                         f"Use a larger interval, or min_level and max_level")
    return [reference + k * interval for k in range(first, last + 1)]


def _highlighted(levels, opts: dict) -> list[bool]:
    """Which levels are emphasised: every Nth, counted from the reference when
    an interval set them, so the emphasised lines keep their values as the
    field's range moves."""
    every = int(opts.get("highlight_every") or 0)
    if every < 1:
        return [False] * len(levels)
    interval, reference = opts.get("interval"), opts.get("reference") or 0.0
    if interval:
        steps = [round((float(v) - reference) / interval) for v in levels]
    else:
        steps = list(range(len(levels)))
    return [step % every == 0 for step in steps]


def _levels_of(values, opts: dict):
    """What to pass as `levels`: the interval's own values, or whatever the
    layer already asked for (a list, a count, or None for matplotlib's pick)."""
    if not opts.get("interval"):
        return opts.get("levels")
    lo, hi = _auto_range(values, opts)
    return interval_levels(lo, hi, opts["interval"], opts.get("reference") or 0.0,
                           opts.get("min_level"), opts.get("max_level"))


def _emphasise(artist, opts: dict, coloured: bool) -> list[bool]:
    """Draw every Nth contour heavier (and in its own colour). Returns which
    levels were emphasised, so the labels can follow the same lines."""
    heavy = _highlighted(list(artist.levels), opts)
    if not any(heavy):
        return heavy
    widths = [opts["highlight_linewidth"] if on else opts["linewidths"] for on in heavy]
    artist.set_linewidth(widths)
    if opts.get("highlight_color") and not coloured:
        base = artist.get_edgecolor()
        colours = [opts["highlight_color"] if on else base[i % len(base)]
                   for i, on in enumerate(heavy)]
        artist.set_edgecolor(colours)
    return heavy


def _labelled(artist, opts: dict, heavy: list[bool]) -> list[float]:
    """Which levels carry a label: every Nth, starting at an emphasised line
    when there is one, so a heavy line is the one that reads its value."""
    levels = list(artist.levels)
    every = int(opts.get("label_every") or 1)
    if every <= 1:
        return levels
    start = heavy.index(True) if any(heavy) else 0
    return levels[start::every]


def _check_colour_options(opts: dict, where: str) -> None:
    """Refuse colour options that contradict each other, by name."""
    norm = opts.get("norm", "linear")
    if opts.get("bands") and norm != "linear":
        raise ValueError(f"{where}bands and norm={norm} cannot be combined: bands are "
                         f"equal steps of the value, a {norm} scale is not")
    if opts.get("gamma", 1.0) != 1.0 and norm != "power":
        raise ValueError(f"{where}gamma applies to norm=power, not norm={norm}")
    if opts.get("center") is not None and norm in ("log", "power"):
        raise ValueError(f"{where}center and norm={norm} cannot be combined: the range "
                         f"is made symmetric about the centre, which a {norm} scale "
                         f"then distorts")


# --------------------------------------------------------------- drawing


def _option(layer: dict, name: str):
    given = layer["options"].get(name)
    if given is not None:
        return given
    return LAYER_KINDS[layer["kind"]]["options"][name]["default"]


def _level_of(layer: dict, level: int) -> int:
    return level if layer["level"] is None else layer["level"]


def layer_data(viewer, layer: dict, time: int, level: int, extent):
    """The data one layer draws, cropped to the view: a DataArray for a field,
    (u, v) for a vector layer, None for a map feature."""
    lvl = _level_of(layer, level)
    group = LAYER_KINDS[layer["kind"]]["group"]
    if group == "field":
        return viewer._cropped(layer["var"], time, lvl, extent)
    if group == "vector":
        u = viewer._cropped(layer["u"], time, lvl, extent)
        v = viewer._cropped(layer["v"], time, lvl, extent)
        if u.shape != v.shape:
            raise ValueError(f"{layer['u']!r} and {layer['v']!r} are not on the same axes")
        return u, v
    return None


def freeze_ranges(viewer, stack: dict, level: int, extent) -> dict:
    """Pin every open colour range to the first step's, for a GIF.

    A range left to matplotlib is recomputed per frame, so the same colour
    means a different value in each one; fixing it from step 0 is what makes
    frames comparable, the same choice the single-plot GIF makes.
    """
    frozen = Stack(json.loads(json.dumps(stack)))
    for layer in frozen["layers"]:
        spec = LAYER_KINDS[layer["kind"]]
        if spec["group"] != "field" or "vmin" not in spec["options"]:
            continue
        opts = layer["options"]
        if isinstance(opts.get("levels"), list):
            continue
        if opts.get("vmin") is not None and opts.get("vmax") is not None:
            continue
        values = np.asarray(layer_data(viewer, layer, 0, level, extent).values, float)
        if np.isfinite(values).any():
            lo, hi = _auto_range(values, opts)
            opts.setdefault("vmin", lo)
            opts.setdefault("vmax", hi)
            if opts["vmax"] <= opts["vmin"]:
                opts["vmax"] = opts["vmin"] + 1.0
    return frozen


def _auto_range(values, opts: dict) -> tuple[float, float]:
    """The colour range a layer draws with when vmin/vmax are not both given.

    Min/max, or the 2nd-98th percentile with `robust`; with `center`, widened
    to be symmetric about it -- the rule xarray applies, written out so bands
    (which need their edges before drawing) and GIF freezing agree with it.
    """
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return 0.0, 1.0
    robust = bool(opts.get("robust"))
    lo = float(np.percentile(finite, 2) if robust else finite.min())
    hi = float(np.percentile(finite, 98) if robust else finite.max())
    if opts.get("vmin") is not None:
        lo = float(opts["vmin"])
    if opts.get("vmax") is not None:
        hi = float(opts["vmax"])
    center = opts.get("center")
    if center is not None and opts.get("vmin") is None and opts.get("vmax") is None:
        half = max(abs(lo - center), abs(hi - center))
        lo, hi = center - half, center + half
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _projection(fig_opts: dict, box, ccrs):
    name = fig_opts.get("projection") or "PlateCarree"
    clon = fig_opts.get("central_longitude")
    clat = fig_opts.get("central_latitude")
    view_lon = 0.5 * (box[0] + box[1])
    view_lat = 0.5 * (box[2] + box[3])
    if name == "PlateCarree":
        from .plot import _frame
        central, src, framed = _frame(box, clon if clon is not None else 0.0)
        return ccrs.PlateCarree(central_longitude=central), src, framed
    lon0 = clon if clon is not None else (((view_lon + 180.0) % 360.0) - 180.0)
    if name == "Orthographic":
        lat0 = clat if clat is not None else view_lat
        proj = ccrs.Orthographic(central_longitude=lon0, central_latitude=lat0)
    elif name == "LambertConformal":
        lat0 = clat if clat is not None else view_lat
        # cartopy's default standard parallels are northern (33, 45) wherever
        # the map is centred, which leaves a southern map on a northern cone
        # -- singular at the very pole the map is looking at
        parallels = (-33.0, -45.0) if lat0 < 0 else (33.0, 45.0)
        proj = ccrs.LambertConformal(central_longitude=lon0, central_latitude=lat0,
                                     standard_parallels=parallels)
    elif name in ("NorthPolarStereo", "SouthPolarStereo"):
        proj = getattr(ccrs, name)(central_longitude=clon if clon is not None else 0.0)
    else:
        proj = getattr(ccrs, name)(central_longitude=lon0)
    return proj, ccrs.PlateCarree(), None


def _set_extent(ax, fig_opts, box, framed, src, global_view: bool, ccrs):
    name = fig_opts.get("projection") or "PlateCarree"
    want_global = fig_opts.get("extent") == "global" or global_view
    if name in ("NorthPolarStereo", "SouthPolarStereo"):
        north = name == "NorthPolarStereo"
        edge = box[2] if north else box[3]
        edge = max(edge, 0.0) if north else min(edge, 0.0)
        if want_global or (north and edge <= 0.0) or (not north and edge >= 0.0):
            edge = 30.0 if north else -30.0
        lats = (edge, 90.0) if north else (-90.0, edge)
        ax.set_extent([-180, 180, *lats], crs=ccrs.PlateCarree())
        return
    if name in ("Robinson", "Mollweide", "EqualEarth", "Orthographic"):
        if want_global or name == "Orthographic":
            ax.set_global()
            return
    if want_global and name == "PlateCarree":
        ax.set_global()
        return
    if framed is not None:
        ax.set_extent(framed, crs=src)
        return
    lat0, lat1 = max(box[2], -89.9), min(box[3], 89.9)
    if name == "Mercator":
        lat0, lat1 = max(lat0, -80.0), min(lat1, 80.0)
    elif name == "LambertConformal":
        # a conic projection runs to infinity at the pole opposite its cone:
        # a global view -- what a reload starts from -- made the extent's
        # corners inf, and matplotlib refused the axis limits
        if cone_lat(ax) >= 0:
            lat0 = max(lat0, -30.0)
        else:
            lat1 = min(lat1, 30.0)
    lon0, lon1 = box[0], box[1]
    if lon1 - lon0 >= 359.0:
        lon0, lon1 = -180.0, 180.0
    mid = 0.5 * (lon0 + lon1)
    ax.set_extent([lon0 - mid, lon1 - mid, lat0, lat1],
                  crs=ccrs.PlateCarree(central_longitude=mid))


def cone_lat(ax) -> float:
    """Which hemisphere a conic projection's cone opens toward: its first
    standard parallel. The pole on the other side is where it runs to inf."""
    return float(ax.projection.proj4_params.get("lat_1", 0.0))


def _norm(opts: dict, vmin, vmax):
    from matplotlib import colors
    kind = opts.get("norm") or "linear"
    if kind == "log":
        return colors.LogNorm(vmin=vmin, vmax=vmax)
    if kind == "symlog":
        return colors.SymLogNorm(linthresh=opts.get("linthresh") or 1.0,
                                 vmin=vmin, vmax=vmax)
    if kind == "power":
        return colors.PowerNorm(gamma=opts.get("gamma") or 1.0, vmin=vmin, vmax=vmax)
    return None


def _field_label(viewer, layer, da, lvl: int) -> str:
    label = layer["options"].get("colorbar_label")
    if label:
        return label
    text = _data.field_label(viewer.ds[layer["var"]])
    stack = viewer._stack_dims(viewer.ds[layer["var"]])
    if stack and stack[0] in viewer.ds.variables:
        coord = viewer.ds[stack[0]]
        value = coord.values[lvl]
        shown = f"{value:g}" if np.issubdtype(coord.dtype, np.number) else str(value)
        text += f" · {shown} {coord.attrs.get('units', '')}".rstrip()
    return text


def _stride(n: int, target: int, given: int) -> int:
    return given if given and given > 0 else max(1, math.ceil(n / target))


def _speed_units(viewer, name: str) -> str:
    return str(viewer.ds[name].attrs.get("units", ""))


def draw(viewer, fig, stack: dict, time: int, level: int, extent):
    """Draw a cleaned stack onto `fig`, bottom layer first. Returns the axes."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.ticker as mticker

    fig_opts = stack["figure"]
    layers = [layer for layer in stack["layers"] if layer["visible"]]
    box = (float(extent[0]), float(extent[1]),
           max(-90.0, float(extent[2])), min(90.0, float(extent[3])))
    proj, src, framed = _projection(fig_opts, box, ccrs)
    plate = isinstance(proj, ccrs.PlateCarree)
    # data in the map's own longitude frame when the map is PlateCarree: see
    # GenericViewer._draw -- it spares cartopy reprojecting every cell
    data_central = proj.proj4_params.get("lon_0", 0.0) if plate else 0.0
    data_crs = ccrs.PlateCarree(central_longitude=data_central)

    ax = fig.add_subplot(projection=proj)
    colorbars = []
    described = []
    look = LOOKS[fig_opts.get("look") or "matplotlib"]

    def palette(layer, opts, fallback="viridis"):
        """The layer's colormap: its own name, else the look's, else `fallback`,
        reversed and given its out-of-range and missing colours."""
        name = layer["options"].get("cmap") or look["palette"] or fallback
        return palettes.get(name, bool(opts.get("reverse")), opts.get("under_color"),
                            opts.get("over_color"), opts.get("missing_color"))
    whole = (fig_opts.get("projection") or "PlateCarree") in _WHOLE_GRID \
        or fig_opts.get("extent") == "global"
    lo = float(viewer.lon.min())
    data_extent = (lo, lo + 360.0, -90.0, 90.0) if whole and viewer.cyclic else (
        viewer.home if whole else extent)

    for index, layer in enumerate(layers):
        z = 10 + 10 * index
        alpha = layer["opacity"]
        kind, spec = layer["kind"], LAYER_KINDS[layer["kind"]]
        opts = {name: _option(layer, name) for name in spec["options"]}
        lvl = _level_of(layer, level)

        if spec["group"] == "field":
            da = layer_data(viewer, layer, time, level, data_extent)
            shifted = _in_frame(viewer, da, data_central)
            common = dict(ax=ax, x=viewer.lon_name, y=viewer.lat_name, transform=data_crs,
                          add_colorbar=False, add_labels=False, alpha=alpha, zorder=z)
            if kind == "contour":
                levels = _levels_of(shifted.values, opts)
                kw = dict(levels=levels, linewidths=opts["linewidths"],
                          linestyles=opts["linestyles"],
                          negative_linestyles=opts["negative_linestyles"])
                # vmin/vmax pick the levels even for single-colour lines, so a
                # GIF's frozen range keeps each frame's contours at the same values
                kw.update(vmin=opts["vmin"], vmax=opts["vmax"])
                given = layer["options"]
                rainbow = look["palette"] == "grads.rainbow" and not given.get("colors")
                if opts["cmap"] or rainbow:
                    kw["cmap"] = palettes.get(opts["cmap"] or look["palette"])
                else:
                    kw["colors"] = opts["colors"]
                artist = shifted.plot.contour(**common, **kw)
                # after drawing, not before: a count (or no levels at all)
                # leaves matplotlib to choose them, and which line is the Nth
                # can only be answered once they exist
                heavy = _emphasise(artist, opts, "cmap" in kw)
                if opts["labels"]:
                    ax.clabel(artist, levels=_labelled(artist, opts, heavy),
                              fmt=opts["label_fmt"],
                              fontsize=opts["label_fontsize"], inline=True)
                if opts["colorbar"] and "cmap" in kw:
                    colorbars.append({"artist": artist, "extend": "neither",
                                      "label": _field_label(viewer, layer, da, lvl)})
            else:
                cmap = palette(layer, opts)
                if opts.get("bands"):
                    # bands need their edges before drawing, so the range is
                    # settled here rather than left to xarray
                    lo, hi = _auto_range(shifted.values, opts)
                    cmap, norm, _ = palettes.scale(
                        {**opts, "cmap": layer["options"].get("cmap") or look["palette"]
                         or "viridis"}, lo, hi)
                    kw = dict(cmap=cmap, norm=norm, extend=opts["extend"])
                else:
                    norm = _norm(opts, opts["vmin"], opts["vmax"])
                    kw = dict(cmap=cmap, robust=opts["robust"], center=opts["center"],
                              extend=opts["extend"])
                    if norm is not None:
                        kw["norm"] = norm
                    else:
                        kw.update(vmin=opts["vmin"], vmax=opts["vmax"])
                    if opts["center"] is None:
                        kw.pop("center")
                if kind == "contourf":
                    kw["levels"] = _levels_of(shifted.values, opts)
                    if opts["hatches"]:
                        kw["hatches"] = opts["hatches"]
                elif kind == "pcolormesh":
                    kw["shading"] = opts["shading"]
                elif kind == "imshow":
                    kw["interpolation"] = opts["interpolation"]
                artist = getattr(shifted.plot, kind)(**common, **kw)
                if kind == "contourf":
                    # a discrete cmap built from `levels` would drop the extremes
                    artist.set_cmap(artist.get_cmap().with_extremes(
                        under=cmap.get_under(), over=cmap.get_over(), bad=cmap.get_bad()))
                if opts["colorbar"]:
                    colorbars.append({"artist": artist, "extend": opts["extend"],
                                      "label": _field_label(viewer, layer, da, lvl),
                                      "format": opts["colorbar_format"],
                                      "ticks": opts["colorbar_ticks"]})
            described.append(f"{spec['label']} {layer['var']}"
                             + _level_text(viewer, layer["var"], lvl))

        elif spec["group"] == "vector":
            u, v = layer_data(viewer, layer, time, level, data_extent)
            u, v = _in_frame(viewer, u, data_central), _in_frame(viewer, v, data_central)
            lon = np.asarray(u[viewer.lon_name].values, float)
            lat = np.asarray(u[viewer.lat_name].values, float)
            uu = np.asarray(u.transpose(viewer.lat_dim, viewer.lon_dim).values, float)
            vv = np.asarray(v.transpose(viewer.lat_dim, viewer.lon_dim).values, float)
            if kind == "streamplot":
                order = np.argsort(lat)
                lat, uu, vv = lat[order], uu[order], vv[order]
                speed = np.hypot(uu, vv)
                kw = dict(transform=data_crs, density=opts["density"],
                          linewidth=opts["linewidth"], arrowsize=opts["arrowsize"],
                          zorder=z)
                if opts["colorize"]:
                    kw.update(color=speed, cmap=palette(layer, opts))
                else:
                    kw["color"] = opts["color"]
                artist = ax.streamplot(lon, lat, uu, vv, **kw)
                artist.lines.set_alpha(alpha)
                artist.arrows.set_alpha(alpha)
                if opts["colorize"] and opts["colorbar"]:
                    units = _speed_units(viewer, layer["u"])
                    colorbars.append({"artist": artist.lines, "extend": "neither",
                                      "label": f"speed [{units}]"})
            else:
                target = 30 if kind == "quiver" else 22
                sy = _stride(lat.size, target, opts["stride"])
                sx = _stride(lon.size, target * 2, opts["stride"])
                X, Y = np.meshgrid(lon[::sx], lat[::sy])
                U, V = uu[::sy, ::sx], vv[::sy, ::sx]
                if kind == "quiver":
                    kw = dict(transform=data_crs, headwidth=opts["headwidth"], zorder=z,
                              alpha=alpha)
                    if opts["scale"]:
                        kw["scale"] = opts["scale"]
                    if opts["width"]:
                        kw["width"] = opts["width"]
                    if opts["colorize"]:
                        artist = ax.quiver(X, Y, U, V, np.hypot(U, V),
                                           cmap=palette(layer, opts), **kw)
                        if opts["colorbar"]:
                            units = _speed_units(viewer, layer["u"])
                            colorbars.append({"artist": artist, "extend": "neither",
                                              "label": f"speed [{units}]"})
                    else:
                        artist = ax.quiver(X, Y, U, V, color=opts["color"], **kw)
                    if opts["key"]:
                        speed = np.hypot(U, V)
                        finite = speed[np.isfinite(speed)]
                        value = opts["key_value"] or (
                            float(_nice(np.percentile(finite, 90))) if finite.size else 1.0)
                        units = _speed_units(viewer, layer["u"])
                        # inside the lower-right corner, on a white label: above
                        # the axes it ran into the title
                        qk = ax.quiverkey(artist, 0.97, 0.04, value,
                                          f"{value:g} {units}".strip(), labelpos="W",
                                          coordinates="axes", zorder=z + 1)
                        qk.text.set_bbox(dict(facecolor="white", edgecolor="none",
                                              alpha=0.8, pad=1.5))
                else:
                    ax.barbs(X, Y, U, V, transform=data_crs, color=opts["color"],
                             length=opts["length"], linewidth=opts["linewidth"],
                             alpha=alpha, zorder=z)
            described.append(f"{spec['label']} {layer['u']}/{layer['v']}"
                             + _level_text(viewer, layer["u"], lvl))

        elif kind == "gridlines":
            gl = ax.gridlines(draw_labels=opts["labels"], color=opts["color"],
                              linewidth=opts["linewidth"], linestyle=opts["linestyle"],
                              alpha=alpha, zorder=z)
            # Gridliner is its own artist and pins itself to zorder 2 on
            # creation, so the keyword above never reaches the draw order and
            # any filled layer covered the lines, wherever the layer sat
            gl.set_zorder(z)
            if opts["dlon"]:
                gl.xlocator = mticker.FixedLocator(np.arange(-180, 180.1, opts["dlon"]))
            if opts["dlat"]:
                gl.ylocator = mticker.FixedLocator(np.arange(-90, 90.1, opts["dlat"]))
            if opts["labels"]:
                gl.top_labels = gl.right_labels = False
                gl.geo_labels = False            # see plot._basemap: the y=inf title bug
        else:
            category, name = _NATURAL_EARTH[kind]
            _ensure_natural_earth(category, name, opts["resolution"])
            if kind == "coastlines":
                ax.coastlines(resolution=opts["resolution"], color=opts["color"],
                              linewidth=opts["linewidth"], alpha=alpha, zorder=z)
                continue
            feature = cfeature.NaturalEarthFeature(category, name, opts["resolution"])
            if kind in ("land", "ocean"):
                style = dict(facecolor=opts["facecolor"], edgecolor="none")
            elif kind == "lakes":
                style = dict(facecolor=opts["facecolor"], edgecolor=opts["edgecolor"],
                             linewidth=0.3)
            elif kind == "borders":
                style = dict(facecolor="none", edgecolor=opts["color"],
                             linewidth=opts["linewidth"], linestyle=opts["linestyle"])
            else:                                               # rivers
                style = dict(facecolor="none", edgecolor=opts["color"],
                             linewidth=opts["linewidth"])
            ax.add_feature(feature, alpha=alpha, zorder=z, **style)

    _set_extent(ax, fig_opts, box, framed, src,
                global_view=viewer.cyclic and box[1] - box[0] >= 359.0 - 2 * _cell(viewer),
                ccrs=ccrs)

    style = fig_opts.get("colorbar_style") or look["colorbar_style"]
    side = fig_opts.get("colorbar") or ("right" if style == "ferret" else "bottom")
    for entry in reversed(colorbars):                      # top layer nearest the map
        _colorbar(fig, ax, entry, style, side)

    title = fig_opts.get("title")
    if not title:
        title = viewer.labels[time] + (f"\n{' · '.join(reversed(described))}"
                                       if described else "")
    if fig_opts.get("subtitle"):
        title = f"{title}\n{fig_opts['subtitle']}"
    # a figure title, not an axes one: constrained layout reserves no room for
    # an axes title over a non-rectangular map (Lambert, Orthographic), and it
    # was placed above the top edge of the figure and cut off
    fig.suptitle(title, fontsize=10)
    _footnotes(fig, fig_opts.get("footnote_left"), fig_opts.get("footnote_right"))
    return ax


def _colorbar(fig, ax, entry: dict, style: str, side: str):
    """One colorbar, drawn the way the look's tool draws its colour key.

    - matplotlib: a plain continuous bar.
    - grads: GrADS' `cbarn` -- each colour its own outlined box, labelled at
      the boundaries between colours, with pointed ends when the range is
      extended.
    - ferret: Ferret's shade key -- boxed colours down the side, a label at
      every level boundary.

    The style changes only how the bar is drawn; a look never adds bands.
    """
    import matplotlib.ticker as mticker

    artist = entry["artist"]
    horizontal = side == "bottom"
    boundaries = getattr(getattr(artist, "norm", None), "boundaries", None)
    if boundaries is None and getattr(artist, "levels", None) is not None \
            and getattr(artist, "filled", False):
        boundaries = artist.levels
    boxed = style in ("grads", "ferret") and boundaries is not None
    kw = dict(ax=ax, orientation="horizontal" if horizontal else "vertical",
              label=entry["label"], extend=entry.get("extend", "neither"))
    if style == "matplotlib":
        kw.update(shrink=0.8 if horizontal else 0.9, aspect=45 if horizontal else 30,
                  pad=0.06 if horizontal else 0.03)
    else:
        kw.update(shrink=0.9 if horizontal else 0.95, aspect=40 if horizontal else 25,
                  pad=0.05 if horizontal else 0.03, drawedges=boxed,
                  extendrect=False, extendfrac="auto")
    if getattr(artist, "filled", True) is False:
        kw.pop("extend")                                    # contour lines: no ends
    cb = fig.colorbar(artist, **kw)
    if style != "matplotlib":
        cb.outline.set_linewidth(1.0)
        cb.outline.set_edgecolor("black")
    if boxed:
        cb.dividers.set_color("black")
        cb.dividers.set_linewidth(0.8)
        edges = [float(b) for b in boundaries]
        if style == "grads":
            ticks = edges[1:-1] or edges                    # between the boxes
        else:
            # every boundary, thinned once there are more than 20
            step = max(1, int(np.ceil(len(edges) / 20)))
            ticks = edges[::step]
        cb.set_ticks(ticks)
    ticks = entry.get("ticks")
    if isinstance(ticks, int):
        cb.locator = mticker.MaxNLocator(ticks)
    elif isinstance(ticks, list):
        cb.set_ticks(ticks)
    if entry.get("format"):
        cb.formatter = mticker.FormatStrFormatter(entry["format"])
    elif boxed:
        # band edges are exact and rarely round (247.6034...); four significant
        # figures reads like GrADS' and Ferret's own key labels
        cb.formatter = mticker.FormatStrFormatter("%.4g")
    cb.update_ticks()
    return cb


def _footnotes(fig, left, right) -> None:
    """Small text under the figure. Built on `supxlabel`, which constrained
    layout reserves room for, so a footnote never overlaps a colorbar or falls
    off the figure on a non-rectangular map. A right-hand footnote beside a
    left-hand one is placed on the left one's line, wherever layout puts it."""
    from matplotlib.text import Annotation

    if left:
        anchor = fig.supxlabel(left, x=0.01, ha="left", fontsize=7)
        if right:
            fig.add_artist(Annotation(right, xy=(0.99, 0.5),
                                      xycoords=("figure fraction", anchor),
                                      ha="right", va="center", fontsize=7))
    elif right:
        fig.supxlabel(right, x=0.99, ha="right", fontsize=7)


def _in_frame(viewer, da, central: float):
    """`da` with longitudes in the map's frame, closed round the globe.

    Two things a plain `lon - central` got wrong. A crop across the 0/360 seam
    comes back unwrapped (260..400); fills and arrows place each point and do
    not care, but streamplot regrids over the data's own x range, and 260..400
    misses a map framed at -100..40 entirely -- no streamlines at all. So the
    whole axis is moved by the multiple of 360 that centres it. And a global
    grid stops one cell short of 360, which leaves an empty wedge at the seam:
    at the map's edge on PlateCarree, but a visible radial gap on a polar or
    globe projection. The first column is repeated past the last to close it.
    """
    import xarray as xr

    lon = np.asarray(da[viewer.lon_name].values, dtype=float)
    if (viewer.cyclic and lon.size == viewer.lon.size and lon.size > 1
            and np.all(np.diff(lon) > 0)):
        first = da.isel({viewer.lon_dim: [0]})
        first = first.assign_coords({viewer.lon_name: (viewer.lon_dim, [lon[0] + 360.0])})
        da = xr.concat([da, first], dim=viewer.lon_dim)
        lon = np.append(lon, lon[0] + 360.0)
    x = lon - central
    x -= 360.0 * np.round(0.5 * (x.min() + x.max()) / 360.0)
    return da.assign_coords({viewer.lon_name: (viewer.lon_dim, x)})


def _cell(viewer) -> float:
    return abs(float(viewer.lon[1] - viewer.lon[0])) if viewer.lon.size > 1 else 1.0


def _level_text(viewer, var: str, lvl: int) -> str:
    stack = viewer._stack_dims(viewer.ds[var])
    if not stack:
        return ""
    dim = stack[0]
    if dim in viewer.ds.variables and viewer.ds[dim].ndim == 1:
        coord = viewer.ds[dim]
        value = coord.values[lvl]
        shown = f"{value:g}" if np.issubdtype(coord.dtype, np.number) else str(value)
        return f" {shown} {coord.attrs.get('units', '')}".rstrip()
    return f" {dim} {lvl}"


def _nice(x: float) -> float:
    """1, 2, 5, 10, 20, ... nearest below x: a reference arrow worth reading."""
    if not x or not math.isfinite(x) or x <= 0:
        return 1.0
    mag = 10 ** math.floor(math.log10(x))
    for step in (5, 2, 1):
        if step * mag <= x:
            return step * mag
    return mag


def _ensure_natural_earth(category: str, name: str, resolution: str) -> None:
    """Fetch a Natural Earth layer now, so a missing one is a message, not a
    traceback from inside savefig. cartopy downloads on first use; on an
    offline HPC node that download fails, and the fix is to pre-fetch it."""
    import cartopy
    from cartopy.io import shapereader

    try:
        shapereader.natural_earth(resolution=resolution, category=category, name=name)
    except Exception as exc:
        raise ValueError(
            f"the Natural Earth layer {name!r} at {resolution} is not available: {exc}. "
            f"cartopy downloads it on first use; offline, fetch it once on a machine "
            f"with internet into {cartopy.config['data_dir']}"
        ) from None
