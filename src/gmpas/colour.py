"""The one place a colour option becomes a colour, for every viewer.

Both fast maps -- the MPAS one in `viewer.py` and the `--generic` one in
`generic.py` -- offer the same palettes and the same colour options, and both
send the browser the same description of them. That is only true because they
call this module rather than each having its own copy: a second copy is how the
two pages drift apart, and how a palette added for one silently misses the
other.

What lives here is the *policy*; the mechanism stays where it was. `palettes`
turns options into a colormap and a norm, `palettes.encode` writes the indexed
PNG, and `viewer._png` writes the plain one. This module decides which of those
two encoders a request gets, and what the page is told is on offer.

Nothing here registers palettes at import: `describe()` and `clean()` call
`palettes.register()` themselves, so `import gmpas` stays as cheap as it was.
"""

from __future__ import annotations

import functools
import json

from . import layers as _layers
from . import palettes

#: The fast map's colour options: the layer colour-scale options that make
#: sense for a raster, checked by the same code as a layer's.
OPTIONS = {
    **{k: _layers._COLOUR_SCALE[k] for k in ("reverse", "norm", "gamma", "linthresh",
                                             "extend", "under_color", "over_color")},
    "bands": {**_layers._RASTER_COLOUR["bands"], "max": 252},   # 252 data palette entries
    "missing_color": _layers._RASTER_COLOUR["missing_color"],
}

#: A colour option set arrives as one query parameter. It is checked, never
#: evaluated: the page is reachable over the network once --host 0.0.0.0 is in
#: play for an HPC tunnel.
MAX_JSON = 4000


def clean(colour) -> dict:
    """A fast-map colour option set from the page, checked, or {} for none."""
    if colour in (None, "", {}):
        return {}
    if isinstance(colour, (str, bytes)):
        if len(colour) > MAX_JSON:
            raise ValueError("colour options are too large")
        try:
            colour = json.loads(colour)
        except json.JSONDecodeError as exc:
            raise ValueError(f"colour options are not valid JSON: {exc}") from None
    palettes.register()
    opts = _layers._clean_options(colour, OPTIONS, "colour: ")
    _layers._check_colour_options(opts, "colour: ")
    return opts


def groups() -> dict[str, list[str]]:
    """Every offered palette, by where it came from."""
    palettes.register()
    return palettes.groups()


@functools.lru_cache(maxsize=1)
def describe() -> dict:
    """What the page needs to build the picker, the bar and the colour form.

    Merged into each viewer's own `describe()`, so the MPAS page and the
    --generic page are offered exactly the same colours.

    Cached because it depends on nothing and costs ~100 ms: sixty-three
    colormaps, each sampled at thirty-two stops. Every /api/meta was paying
    that again, which on a small mesh is more than the first frame.
    """
    from .viewer import ramp

    by_group = groups()
    names = [n for group in by_group.values() for n in group]
    return {"cmaps": names,
            "ramps": {n: ramp(n) for n in names},
            "palettes": by_group,
            "colour_options": OPTIONS}


def description() -> dict:
    """A fresh copy of `describe`, for a caller that merges it into its own."""
    got = describe()
    return {k: (dict(v) if isinstance(v, dict) else list(v))
            for k, v in got.items()}


def frame_png(img, cmap: str, lo: float, hi: float, compress: int = 1,
              colour=None, outside=None, meta=None) -> bytes:
    """A fast-map frame, with the colour options if there are any.

    Without them this is `viewer._png` and nothing else -- the same bytes the
    viewer has always served, from the same 255-step palette. With them the
    indexed encoder runs instead, which reserves entries for out-of-range and
    missing values and so has 252 steps for the data.

    `outside` marks pixels that are off the mesh or off the grid. They stay
    transparent; a missing colour is for cells that exist but hold no value.
    `meta`, a dict, receives the colorbar description under "colorbar", which
    the handler returns as the X-Colorbar header so the bar on the page is
    drawn from the very colours the image was drawn with.
    """
    from .viewer import _png

    opts = clean(colour)
    if not opts:
        return _png(img, cmap, lo, hi, compress)

    from .palettes import encode

    png, spec = encode.png(img, {**opts, "cmap": cmap or "viridis"}, lo, hi, compress,
                           outside=outside)
    if meta is not None:
        meta["colorbar"] = spec
    return png


def figure_scale(cmap: str, lo: float, hi: float, colour=None):
    """(colormap, norm or None, extend) for a matplotlib export.

    The figure is drawn by matplotlib rather than by the indexed encoder, so it
    needs the colormap object and the norm the options describe. Without
    options this returns the name unchanged and no norm, which is what the
    export always passed.
    """
    opts = clean(colour)
    if not opts:
        return cmap or "viridis", None, "neither"
    cm, norm, _ = palettes.scale({**opts, "cmap": cmap or "viridis"}, lo, hi)
    return cm, norm, opts.get("extend") or "neither"
