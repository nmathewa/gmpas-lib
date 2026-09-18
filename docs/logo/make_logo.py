"""Draw the gmpas wordmark.

The letters are real outlines, not a font reference, so the mark renders
identically in the viewer, on GitHub and in a paper -- none of which can be
relied on to have any particular font installed. DejaVu Sans Bold is the
source (Bitstream Vera licence, which permits this); matplotlib is already a
dependency and can hand back glyph outlines.

The mark is a hexagon tiled with hexagons: MPAS's Voronoi mesh is the thing
that makes this tool different from every other netCDF viewer, and it is
legible down to a favicon.

    python docs/logo/make_logo.py        # rewrites the .svg files beside it
"""

from __future__ import annotations

import math
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: The brand blue, used where the background is light. MPAS's own blue.
BRAND = "#2b6cb0"
#: The same blue lifted for dark backgrounds. On #16181c the deep blue has a
#: contrast ratio of 3.28, which looks muddy; this is 5.32 and still reads as
#: the same colour. The viewer's --accent is this one.
BRAND_DARK_BG = "#4a90d9"

FG_ON_DARK = "#e6e8ec"
FG_ON_LIGHT = "#1e2127"


def glyphs(text: str, size: float = 100.0) -> tuple[str, float]:
    """`text` as one SVG path, with its advance width."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    path = TextPath((0, 0), text, size=size,
                    prop=FontProperties(family="DejaVu Sans", weight="bold"))
    out = []
    for v, code in path.iter_segments():
        # SVG's y grows downward; matplotlib's grows up
        if code == 1:
            out.append(f"M{v[0]:.1f},{-v[1]:.1f}")
        elif code == 2:
            out.append(f"L{v[0]:.1f},{-v[1]:.1f}")
        elif code == 3:
            out.append(f"Q{v[0]:.1f},{-v[1]:.1f} {v[2]:.1f},{-v[3]:.1f}")
        elif code == 4:
            out.append(f"C{v[0]:.1f},{-v[1]:.1f} {v[2]:.1f},{-v[3]:.1f} "
                       f"{v[4]:.1f},{-v[5]:.1f}")
        elif code == 79:
            out.append("Z")
    return " ".join(out), float(path.get_extents().x1)


def hexagon(cx: float, cy: float, r: float) -> str:
    """A flat-topped hexagon, the orientation MPAS meshes are drawn in."""
    pts = [(cx + r * math.cos(i * math.pi / 3), cy + r * math.sin(i * math.pi / 3))
           for i in range(6)]
    return "M" + " L".join(f"{x:.2f},{y:.2f}" for x, y in pts) + " Z"


def mark(cx: float, cy: float, R: float, colour: str, density: float = 2.3) -> str:
    """A hexagon filled with hexagons, clipped to its own silhouette.

    Cells fade with distance from the centre, which reads as a mesh refining
    -- the thing `prep` exists to do -- without needing any more detail than
    survives at 16 pixels.
    """
    r = R / density
    ident = f"m{int(cx)}{int(cy)}{int(R)}"
    cells = []
    step_x, step_y = r * 1.5, r * math.sqrt(3)
    reach = int(R / r) + 2
    for i in range(-reach, reach + 1):
        for j in range(-reach, reach + 1):
            ox = i * step_x
            oy = j * step_y + (step_y / 2 if i % 2 else 0)
            if abs(oy) > R * math.sqrt(3) / 2 or abs(ox) > R:
                continue
            fade = max(0.32, 1.0 - 0.62 * math.hypot(ox, oy) / R)
            cells.append(f'<path d="{hexagon(cx + ox, cy + oy, r * 0.9)}" '
                         f'fill="{colour}" opacity="{fade:.2f}"/>')
    return (f'<clipPath id="{ident}"><path d="{hexagon(cx, cy, R)}"/></clipPath>'
            f'<g clip-path="url(#{ident})">{"".join(cells)}</g>'
            f'<path d="{hexagon(cx, cy, R)}" fill="none" stroke="{colour}" '
            f'stroke-width="{R * 0.085:.1f}"/>')


def wordmark(colour: str, text_fill: str, size: float = 100.0) -> str:
    word, width = glyphs("gmpas", size)
    R = size * 0.52
    mx = 14 + R
    gap = mx + R + size * 0.26
    w, h = gap + width + 14, size * 1.8
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}" '
            f'width="{w:.0f}" height="{h:.0f}" role="img" aria-label="gmpas">'
            f'{mark(mx, h / 2, R, colour)}'
            f'<g transform="translate({gap:.1f},{h / 2 + size * 0.36:.1f})">'
            f'<path d="{word}" fill="{text_fill}"/></g></svg>')


def main() -> None:
    (HERE / "gmpas.svg").write_text(wordmark(BRAND, FG_ON_LIGHT))
    (HERE / "gmpas-dark.svg").write_text(wordmark(BRAND_DARK_BG, FG_ON_DARK))
    side = 112.0
    for name, colour in (("mark.svg", BRAND), ("mark-dark.svg", BRAND_DARK_BG)):
        (HERE / name).write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {side:.0f} '
            f'{side:.0f}" width="{side:.0f}" height="{side:.0f}" role="img" '
            f'aria-label="gmpas">{mark(side / 2, side / 2, side * 0.44, colour)}</svg>')
    for f in sorted(HERE.glob("*.svg")):
        print(f"  {f.name:18} {f.stat().st_size:5d} bytes")


if __name__ == "__main__":
    main()
