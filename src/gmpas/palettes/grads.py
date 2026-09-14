"""GrADS' default colours and its rainbow sequence, as colormaps.

The numbers are GrADS' published defaults: the `pdcred`, `pdcgre` and
`pdcblu` tables in `src/gxdb.c` of the GrADS source, the same values its
"GrADS Default Colors" page lists. (That page gives colour 14 as both
130,0,220 and 110,0,220; the source says 130.) Only the values are used --
no GrADS code, which is GPL -- so this file is data, not a derived work.

The rainbow is what `set gxout shaded` and `set ccolor rainbow` draw with:
colours 9 14 4 11 5 13 3 10 7 12 8 2 6, purple through red to magenta. It is
kept as a list of 13 distinct colours, not resampled into a gradient, because
GrADS never blends them.
"""

from __future__ import annotations

#: colour number -> (red, green, blue), 0..255
COLORS = (
    (0, 0, 0),         # 0  background (black by default)
    (255, 255, 255),   # 1  foreground (white by default)
    (250, 60, 60),     # 2  red
    (0, 220, 0),       # 3  green
    (30, 60, 255),     # 4  dark blue
    (0, 200, 200),     # 5  light blue
    (240, 0, 130),     # 6  magenta
    (230, 220, 50),    # 7  yellow
    (240, 130, 40),    # 8  orange
    (160, 0, 200),     # 9  purple
    (160, 230, 50),    # 10 yellow/green
    (0, 160, 255),     # 11 medium blue
    (230, 175, 45),    # 12 dark yellow
    (0, 210, 140),     # 13 aqua
    (130, 0, 220),     # 14 dark purple
    (170, 170, 170),   # 15 grey
)

#: the default rainbow sequence, as colour numbers
RAINBOW = (9, 14, 4, 11, 5, 13, 3, 10, 7, 12, 8, 2, 6)


def _rgb(number: int) -> tuple[float, float, float]:
    return tuple(v / 255.0 for v in COLORS[number])


def rainbow():
    from matplotlib.colors import ListedColormap

    return ListedColormap([_rgb(n) for n in RAINBOW], name="grads.rainbow")


def default16():
    from matplotlib.colors import ListedColormap

    return ListedColormap([_rgb(n) for n in range(16)], name="grads.default16")
