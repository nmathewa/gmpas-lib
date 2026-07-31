"""House style for MPAS figures: presets, colormaps, named extents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: named map extents, (lon_min, lon_max, lat_min, lat_max)
EXTENTS = {
    "global": (-180.0, 180.0, -90.0, 90.0),
    "maritime_continent": (90.0, 160.0, -20.0, 20.0),
    "mjo_basin": (40.0, 180.0, -30.0, 30.0),
    "indo_pacific": (60.0, 200.0, -40.0, 40.0),
    "conus": (-125.0, -66.0, 23.0, 50.0),
}

#: sensible default colormap per field kind
CMAPS = {
    "sequential": "viridis",
    "anomaly": "RdBu_r",
    "wind": "turbo",
    "terrain": "terrain",
    "precip": "Blues",
}


@dataclass
class Style:
    """Figure sizing and line weights, scaled per output target."""

    figsize: tuple[float, float] = (10.0, 6.0)
    dpi: int = 130
    coastline_lw: float = 0.6
    gridline_lw: float = 0.4
    edge_lw: float = 0.0          # cell outlines; 0 keeps big meshes readable
    title_size: float = 12.0
    label_size: float = 10.0

    @classmethod
    def preset(cls, name: str = "paper") -> "Style":
        if name == "poster":
            return cls(figsize=(16.0, 9.0), dpi=200, coastline_lw=1.0,
                       title_size=18.0, label_size=14.0)
        if name == "notebook":
            return cls(figsize=(9.0, 5.0), dpi=100)
        if name == "mesh":
            # for inspecting mesh structure, cell outlines must be visible
            return cls(figsize=(11.0, 7.0), dpi=150, edge_lw=0.15)
        return cls()


def resolve_extent(mesh, extent: str | tuple | None):
    """Turn a named extent, an explicit box, or None into a lon/lat box."""
    if extent is None or extent == "":
        return mesh.extent
    if isinstance(extent, str):
        if extent not in EXTENTS:
            raise KeyError(
                f"unknown extent {extent!r}; known: {sorted(EXTENTS)} "
                f"(or pass an explicit (lon_min, lon_max, lat_min, lat_max))"
            )
        return EXTENTS[extent]
    box = tuple(float(v) for v in extent)
    if len(box) != 4:
        raise ValueError("extent box must be (lon_min, lon_max, lat_min, lat_max)")
    return box


def save_figure(fig, path: str | Path, style: Style | None = None) -> Path:
    """Write a figure to `path`, creating parent directories, and return it.

    Unlike the MCP server this grew out of, nothing here writes to a fixed
    project directory -- a library saves where the caller says.

    Deliberately no bbox_inches="tight": these figures use constrained_layout,
    and the two together crop a cartopy GeoAxes down to nothing but its
    colorbar. constrained_layout already packs the figure.
    """
    import matplotlib.pyplot as plt

    style = style or Style()
    out = Path(path).expanduser()
    if out.suffix == "":
        out = out.with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=style.dpi)
    plt.close(fig)
    return out
