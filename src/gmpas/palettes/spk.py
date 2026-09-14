"""Ferret `.spk` colour palettes, read into matplotlib colormaps.

A palette file is an optional header line, `RGB_Mapping Percent`, `By_value`
or `By_level`, followed by one colour per line as four numbers: a position
and red, green, blue, each in percent. `!` starts a comment. Without a header
the mapping is Percent.

- Percent: positions run 0..100 along the colour scale and colours are
  interpolated between them; a position given twice is a hard step.
- By_level: one colour per contour level, in order, never interpolated.
- By_value: colours pinned to data values. That ties the palette to one
  field's units, which a general colour scale cannot honour, so it is
  refused rather than approximated.
"""

from __future__ import annotations

import re


class SpkByValue(ValueError):
    """A By_value palette: its colours belong to particular data values."""


_HEADER = re.compile(r"^\s*rgb_mapping\s+(\w+)", re.IGNORECASE)


def parse(text: str, name: str):
    """A matplotlib colormap from the text of a `.spk` file."""
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap

    mapping = "percent"
    points: list[tuple[float, tuple[float, float, float]]] = []
    for raw in text.splitlines():
        line = raw.split("!", 1)[0].split("#", 1)[0].strip()
        if not line:
            continue
        header = _HEADER.match(line)
        if header:
            mapping = header.group(1).lower()
            continue
        fields = line.split()
        if len(fields) < 4:
            raise ValueError(f"{name}: line {raw.strip()!r} is not "
                             f"'position red green blue'")
        position, *rgb = (float(v) for v in fields[:4])
        colour = tuple(min(max(c, 0.0), 100.0) / 100.0 for c in rgb)
        points.append((position, colour))

    if mapping == "by_value":
        raise SpkByValue(f"{name}: a By_value palette pins colours to data values")
    if mapping not in ("percent", "by_level"):
        raise ValueError(f"{name}: unknown RGB_Mapping {mapping!r}")
    if not points:
        raise ValueError(f"{name}: no colours")

    points.sort(key=lambda p: p[0])                    # stable: repeats keep file order
    if mapping == "by_level":
        return ListedColormap([c for _, c in points], name=name)

    stops = [(min(max(p, 0.0), 100.0) / 100.0, c) for p, c in points]
    if stops[0][0] > 0.0:
        stops.insert(0, (0.0, stops[0][1]))
    if stops[-1][0] < 1.0:
        stops.append((1.0, stops[-1][1]))
    if len(stops) == 1:
        stops.append((1.0, stops[0][1]))
    return LinearSegmentedColormap.from_list(name, stops, N=256)
