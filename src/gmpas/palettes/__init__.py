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


def scale(opts: dict, lo: float, hi: float):
    """(colormap, norm, band edges or None) for a colour-scale option set.

    The one place that turns options into colours, used by layers, --generic
    figures and the fast map's encoder, so the three cannot disagree.
    """
    from matplotlib import colors

    cmap = get(opts.get("cmap") or "viridis", bool(opts.get("reverse")),
               opts.get("under_color"), opts.get("over_color"), opts.get("missing_color"))
    extend = opts.get("extend") or "neither"
    bands = opts.get("bands")
    if bands:
        edges = band_edges(lo, hi, bands)
        return cmap, colors.BoundaryNorm(edges, cmap.N, extend=extend), edges
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
