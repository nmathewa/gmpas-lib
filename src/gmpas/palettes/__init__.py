"""Colour palettes for `--generic`: matplotlib, cmocean, Ferret and GrADS.

Everything here is registered into matplotlib's colormap registry under a
namespaced name -- `cmo.thermal`, `ferret.rnb2`, `grads.rainbow` -- so a name
means the same thing to a layer's validation, the fast map's palette and a
matplotlib figure.

Registration happens on first use from a `--generic` code path, never on
`import gmpas`: the MPAS viewer offers its own fixed list and does not change.
The registry is process-wide, so in a process that also serves MPAS those
names would resolve there too; the MPAS page never offers them.

Display only. Nothing here touches data values; `scale` decides which colour
a value is drawn in.
"""

from __future__ import annotations

import threading
from importlib import resources

import numpy as np

from . import grads, spk

#: the matplotlib colormaps --generic has always offered
MATPLOTLIB = ("viridis", "plasma", "magma", "cividis", "turbo", "RdBu_r", "coolwarm",
              "BrBG", "Blues", "Spectral_r")

#: the vendored Ferret palettes, in picker order (see ferret/NOTICE.txt)
FERRET = ("default", "rnb2", "rainbow", "light_rainbow", "medium_rainbow",
          "inverse_rainbow", "light_centered", "white_centered", "centered",
          "red_blue_centered", "no_green_centered", "blue_orange", "blue_darkred",
          "brown_blue", "bluescale", "redscale", "greenscale", "grayscale", "land_sea",
          "terrestrial", "dark_terrestrial", "rain_cmyk", "warm_cmyk",
          "blue_green_yellow", "light_bottom", "ten_by_levels", "fifteen_by_levels",
          "rainbow_by_levels", "categorical_12_step")

GRADS = ("grads.rainbow", "grads.default16")

_lock = threading.Lock()
_done = False


def register() -> None:
    """Put every palette into matplotlib's registry. Safe to call repeatedly."""
    global _done
    if _done:
        return
    with _lock:
        if _done:
            return
        from matplotlib import colormaps

        builtin = [grads.rainbow(), grads.default16()]
        folder = resources.files(__name__) / "ferret"
        for stem in FERRET:
            text = (folder / f"{stem}.spk").read_text(encoding="utf-8", errors="replace")
            builtin.append(spk.parse(text, f"ferret.{stem}"))
        for cmap in builtin:
            if cmap.name not in colormaps:
                colormaps.register(cmap, name=cmap.name)
        _import_cmocean()
        _done = True


def _import_cmocean():
    """cmocean registers its `cmo.*` maps when imported; None if it can't be.

    cmocean 4.0.3 builds its maps with `ListedColormap(..., N=)`, which
    matplotlib 3.11 deprecates and 3.13 removes (matplotlib/cmocean#121). The
    warning is silenced here, and any failure to import -- not only a missing
    package -- leaves the cmocean group empty rather than the viewer broken.
    """
    import warnings

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Passing 'N' to ListedColormap")
            import cmocean.cm
        return cmocean.cm
    except Exception:
        return None


def groups() -> dict[str, list[str]]:
    """Palette names by source, for the picker. cmocean is empty if absent."""
    register()
    from matplotlib import colormaps

    cm = _import_cmocean()
    cmo = [f"cmo.{n}" for n in cm.cmapnames if f"cmo.{n}" in colormaps] if cm else []
    return {"matplotlib": list(MATPLOTLIB), "cmocean": cmo,
            "ferret": [f"ferret.{s}" for s in FERRET], "grads": list(GRADS)}


def known(name: str) -> bool:
    register()
    from matplotlib import colormaps

    return name in colormaps


def get(name: str, reverse: bool = False, under=None, over=None, bad=None):
    """A colormap by name, reversed first if asked, then given its extremes.

    Reversing first matters: `reversed()` swaps a colormap's under and over
    colours, so extremes set before it would land on the wrong ends.
    """
    register()
    from matplotlib import colormaps

    if name not in colormaps:
        raise ValueError(f"{name!r} is not a known colormap")
    cmap = colormaps[name]
    if reverse:
        cmap = cmap.reversed()
    extremes = {k: v for k, v in (("under", under), ("over", over), ("bad", bad))
                if v is not None}
    return cmap.with_extremes(**extremes) if extremes else cmap


def band_edges(lo: float, hi: float, bands: int) -> np.ndarray:
    """`bands` equal colour steps over lo..hi.

    The top edge is nudged just past `hi` so the field's own maximum falls in
    the last band: matplotlib's BoundaryNorm treats a value equal to the top
    edge as over the range, which would paint the maximum in the over colour.
    """
    edges = np.linspace(float(lo), float(hi), int(bands) + 1)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    return edges


#: most colour steps one palette PNG can hold (4 of 256 entries are reserved)
MAX_STEPS = 252


def interval_edges(lo: float, hi: float, step: float) -> np.ndarray:
    """Colour edges every `step`, on its multiples, covering lo..hi.

    The GrADS/Ferret contour-interval rule applied to colours: 0, 2.5, 5, ...
    rather than equal fractions of the range, so an edge means the same value
    in every frame and every run. The ends round outwards to whole steps; the
    top edge is nudged past its value, as in `band_edges`.
    """
    step = float(step)
    if not step > 0:
        raise ValueError(f"a colour step of {step:g} must be above zero")
    i0 = int(np.floor(lo / step + 1e-9))
    i1 = int(np.ceil(hi / step - 1e-9))
    if i1 <= i0:
        i1 = i0 + 1
    if i1 - i0 > MAX_STEPS:
        raise ValueError(
            f"a colour step of {step:g} over {lo:g}..{hi:g} makes {i1 - i0} steps; "
            f"at most {MAX_STEPS} fit -- use a larger step or a narrower range")
    edges = np.arange(i0, i1 + 1, dtype=np.float64) * step
    edges[-1] = np.nextafter(edges[-1], np.inf)
    return edges


_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_INF = r"[-+]?inf"
_LEVELS_FORMS = ("N (N equal steps), (lo,hi,step) segments such as "
                 "(-25,0,5)(0,25,1), (,,step) over the colour range, single levels "
                 "such as (0)(1)(2)(5), and (-inf) / (inf) for open ends")


def parse_levels(text: str) -> dict:
    """A Ferret-style colour levels expression, checked, never evaluated.

    The subset of Ferret's /LEVELS that sets colour steps:

    - ``20``: twenty equal steps over the colour range (what `bands` does);
    - ``(lo,hi,step)``: lo, lo+step, ... up to hi; lo or hi may be left
      blank to take the colour range, ``(,,2.5)`` then putting the steps on
      multiples of 2.5 as `interval_edges` does;
    - ``(v)``: one level; segments join, ``(-25,0,5)(0,25,1)``, a level that
      two segments share counting once;
    - ``(-inf)`` first and ``(inf)`` last: open ends, values beyond them in
      the under/over colour with a triangle on the key.

    Returns {"count": N} or {"segments": [...], "open": (low, high)}, each
    segment ("value", v) or ("range", lo or None, hi or None, step).
    """
    import re

    s = re.sub(r"\s+", "", str(text)).lower()
    if not s:
        raise ValueError("levels is empty")
    if re.fullmatch(r"\d+", s):
        n = int(s)
        if not 1 <= n <= MAX_STEPS:
            raise ValueError(f"levels {n}: between 1 and {MAX_STEPS} steps fit")
        return {"count": n}
    groups = re.findall(r"\(([^()]*)\)", s)
    if not groups or "".join(f"({g})" for g in groups) != s:
        raise ValueError(f"levels {text!r} is not understood; use {_LEVELS_FORMS}")
    segments, low, high = [], False, False
    for i, g in enumerate(groups):
        parts = g.split(",")
        if len(parts) == 1 and re.fullmatch(_INF, parts[0]):
            if parts[0].startswith("-") and i == 0:
                low = True
            elif not parts[0].startswith("-") and i == len(groups) - 1:
                high = True
            else:
                raise ValueError(f"levels: ({g}) belongs at the "
                                 f"{'start' if parts[0].startswith('-') else 'end'}")
            continue
        if len(parts) == 1 and re.fullmatch(_NUMBER, parts[0]):
            segments.append(("value", float(parts[0])))
            continue
        if len(parts) == 3 and all(re.fullmatch(_NUMBER, p) for p in parts[2:]) \
                and all(p == "" or re.fullmatch(_NUMBER, p) for p in parts[:2]):
            lo, hi, step = (float(p) if p else None for p in parts)
            if not step > 0:
                raise ValueError(f"levels: ({g}) needs a step above zero")
            if lo is not None and hi is not None and not hi > lo:
                raise ValueError(f"levels: ({g}) needs hi above lo")
            segments.append(("range", lo, hi, step))
            continue
        raise ValueError(f"levels: ({g}) is not understood; use {_LEVELS_FORMS}")
    if not segments:
        raise ValueError("levels: open ends need levels between them")
    return {"segments": segments, "open": (low, high)}


def levels_ends(parsed: dict):
    """(lo, hi) the levels fix, None for an end that follows the colour range."""
    if "count" in parsed:
        return None, None
    first, last = parsed["segments"][0], parsed["segments"][-1]
    lo = first[1]
    hi = last[1] if last[0] == "value" else last[2]
    return lo, hi


def level_edges(parsed: dict, lo: float, hi: float) -> np.ndarray:
    """The colour edges a parsed levels expression makes over lo..hi.

    The top edge is nudged past its value, as in `band_edges`, so a field
    whose maximum sits on the last level draws it in the last step.
    """
    if "count" in parsed:
        return band_edges(lo, hi, parsed["count"])
    out: list[float] = []
    for seg in parsed["segments"]:
        if seg[0] == "value":
            part = [seg[1]]
        else:
            _, a, b, step = seg
            if a is None and b is None:
                part = list(interval_edges(lo, hi, step))
                part[-1] = float(np.nextafter(part[-1], -np.inf))   # undo its nudge
            else:
                a = lo if a is None else a
                b = hi if b is None else b
                if not b > a:
                    raise ValueError(f"levels: the segment runs from {a:g} to {b:g}, "
                                     f"which is empty")
                n = int(np.floor((b - a) / step + 1e-9))
                if n > MAX_STEPS:
                    raise ValueError(f"levels: {a:g} to {b:g} every {step:g} makes {n} "
                                     f"steps; at most {MAX_STEPS} fit")
                part = list(a + step * np.arange(n + 1, dtype=np.float64))
        for v in part:
            v = float(v)
            if out and abs(v - out[-1]) <= 1e-9 * max(1.0, abs(v)):
                continue                    # a level two segments share
            if out and v < out[-1]:
                raise ValueError(f"levels must increase; {v:g} comes after {out[-1]:g}")
            out.append(v)
    if len(out) < 2:
        raise ValueError("levels: at least two levels make a colour step")
    if len(out) - 1 > MAX_STEPS:
        raise ValueError(f"levels make {len(out) - 1} steps; at most {MAX_STEPS} fit")
    edges = np.asarray(out, dtype=np.float64)
    edges[-1] = np.nextafter(edges[-1], np.inf)
    return edges


def extend_of(opts: dict) -> str:
    """The key's triangles: the `extend` option joined with open-ended levels."""
    ext = opts.get("extend") or "neither"
    if not opts.get("levels"):
        return ext
    low, high = parse_levels(opts["levels"]).get("open", (False, False))
    low = low or ext in ("min", "both")
    high = high or ext in ("max", "both")
    return {(False, False): "neither", (True, False): "min",
            (False, True): "max", (True, True): "both"}[(low, high)]


def scale(opts: dict, lo: float, hi: float):
    """(colormap, norm, band edges or None) for a colour-scale option set.

    The one place that turns options into colours, used by layers, --generic
    figures and the fast map's encoder, so the three cannot disagree.
    """
    from matplotlib import colors

    cmap = get(opts.get("cmap") or "viridis", bool(opts.get("reverse")),
               opts.get("under_color"), opts.get("over_color"), opts.get("missing_color"))
    bands = opts.get("bands")
    if opts.get("levels"):
        # uneven steps each take an equal share of the colormap, as in Ferret
        edges = level_edges(parse_levels(opts["levels"]), lo, hi)
        bands = edges.size - 1
    elif bands:
        edges = band_edges(lo, hi, bands)
    if bands:
        if bands > cmap.N:
            # more bands than colours -- GrADS' 13-colour rainbow over 20
            # levels -- repeats colours, as GrADS does, instead of failing
            cmap = get(opts.get("cmap") or "viridis", bool(opts.get("reverse")),
                       opts.get("under_color"), opts.get("over_color"),
                       opts.get("missing_color")).resampled(int(bands))
        # No `extend` on the norm: values outside the range take the colormap's
        # own under/over colours (its end colours unless set), and `extend`
        # only draws the colorbar's triangles. Putting it on the norm makes the
        # norm spend two of the colormap's colours on the ends, which a
        # 13-colour GrADS rainbow over 13 bands does not have.
        return cmap, colors.BoundaryNorm(edges, cmap.N), edges
    kind = opts.get("norm") or "linear"
    if kind == "log":
        norm = colors.LogNorm(vmin=lo, vmax=hi)
    elif kind == "symlog":
        norm = colors.SymLogNorm(linthresh=opts.get("linthresh") or 1.0, vmin=lo, vmax=hi)
    elif kind == "power":
        norm = colors.PowerNorm(gamma=opts.get("gamma") or 1.0, vmin=lo, vmax=hi)
    else:
        norm = colors.Normalize(vmin=lo, vmax=hi)
    return cmap, norm, None
