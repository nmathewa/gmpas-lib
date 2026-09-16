"""The fast map's palette PNG, with colour options.

`viewer._png` stays exactly as it is for everything that sets no colour
option -- the MPAS viewer and a plain --generic map. This encoder is used only
when a --generic map asks for bands, reversal, a power scale, or colours for
out-of-range and missing cells, which need palette entries `_png` does not
have.

Index layout of the 8-bit image:
    0..251  data: 252 steps of a continuous scale, or one entry per band
    252     under the range
    253     over the range
    254     missing (NaN) cells, when they are given a colour
    255     outside the grid, and missing cells left transparent

Only index 255 is transparent, so the frames still re-container as GIF, which
holds a single transparent index.

Colours come from `palettes.scale` -- the same colormap and norm a matplotlib
figure of the same options uses -- so the map and an exported figure agree.
"""

from __future__ import annotations

import io

import numpy as np

from . import scale

DATA = 252
UNDER, OVER, MISSING, CLEAR = 252, 253, 254, 255


def _hex(rgba) -> str:
    r, g, b = (int(round(float(c) * 255)) for c in rgba[:3])
    return f"#{r:02x}{g:02x}{b:02x}"


def indices(img: np.ndarray, opts: dict, lo: float, hi: float, outside=None):
    """(index image, palette as (256, 3) uint8, colorbar spec)."""
    cmap, norm, edges = scale(opts, lo, hi)
    values = np.asarray(img, dtype=np.float64)
    finite = np.isfinite(values)
    on_grid = np.ones(values.shape, bool) if outside is None else ~np.asarray(outside)

    idx = np.full(values.shape, CLEAR, dtype=np.uint8)
    palette = np.zeros((256, 3), dtype=np.uint8)

    if edges is not None:
        bands = edges.size - 1
        # band i holds edges[i] <= v < edges[i+1]; edges[-1] sits just past hi
        band = np.searchsorted(edges, np.where(finite, values, lo), side="right") - 1
        inside = finite & (band >= 0) & (band < bands)
        idx[inside] = band[inside].astype(np.uint8)
        centres = 0.5 * (edges[:-1] + edges[1:])
        colours = cmap(norm(centres))
        palette[:bands] = np.round(np.asarray(colours)[:, :3] * 255)
        under = finite & (values < edges[0])
        over = finite & (values >= edges[-1])
        stops = [_hex(c) for c in colours]
    else:
        inside = finite & (values >= lo) & (values <= hi)
        t = np.asarray(np.ma.filled(norm(np.where(inside, values, lo)), 0.0), float)
        # A colormap with no more colours than there are data entries (GrADS'
        # 13, a Ferret By_level list) is indexed exactly as matplotlib indexes
        # it, int(t * N): quantising it to 252 steps would put the pixels next
        # to each colour boundary in the neighbouring colour. Longer colormaps
        # are sampled at 252 steps.
        steps = cmap.N if cmap.N <= DATA else DATA
        k = np.clip(np.floor(t * steps), 0, steps - 1).astype(np.uint8)
        idx[inside] = k[inside]
        if cmap.N <= DATA:
            palette[:steps] = np.round(cmap(np.arange(steps))[:, :3] * 255)
        else:
            palette[:DATA] = np.round(cmap((np.arange(DATA) + 0.5) / DATA)[:, :3] * 255)
        under = finite & (values < lo)
        over = finite & (values > hi)
        # the HTML bar is linear in value: sample the colour at evenly spaced values
        probe = np.linspace(lo, hi, 32)
        stops = [_hex(c) for c in cmap(np.ma.filled(norm(probe), 0.0))]

    idx[under] = UNDER
    idx[over] = OVER
    palette[UNDER] = np.round(np.asarray(cmap.get_under())[:3] * 255)
    palette[OVER] = np.round(np.asarray(cmap.get_over())[:3] * 255)
    if opts.get("missing_color"):
        idx[on_grid & ~finite] = MISSING
        palette[MISSING] = np.round(np.asarray(cmap.get_bad())[:3] * 255)
    idx[~on_grid] = CLEAR

    extend = opts.get("extend") or "neither"
    spec = {"stops": stops,
            "edges": [float(e) for e in edges] if edges is not None else None,
            "under": _hex(cmap.get_under()) if extend in ("min", "both") else None,
            "over": _hex(cmap.get_over()) if extend in ("max", "both") else None,
            "extend": extend, "lo": float(lo), "hi": float(hi)}
    if edges is not None:
        spec["edges"][-1] = float(hi)                     # label the range, not the nudge
    return idx, palette, spec


def png(img: np.ndarray, opts: dict, lo: float, hi: float, compress: int = 1,
        outside=None) -> tuple[bytes, dict]:
    """(palette PNG bytes, colorbar spec). Row 0 of `img` is the south, as for
    `viewer._png`, so the image is flipped to put north at the top."""
    from PIL import Image

    idx, palette, spec = indices(img, opts, lo, hi, outside)
    im = Image.fromarray(idx[::-1], mode="P")
    im.putpalette(palette.ravel().tolist())
    buf = io.BytesIO()
    im.save(buf, format="PNG", transparency=CLEAR, compress_level=compress)
    return buf.getvalue(), spec
