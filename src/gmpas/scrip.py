"""Write an MPAS mesh as a SCRIP grid file.

SCRIP is the lingua franca for remapping weights: give it to
`ESMF_RegridWeightGen -m conserve`, to TempestRemap, or to `ncremap`, and you
get back a weight file that is genuinely area-conservative. This module exists
so that step needs nothing beyond gmpas.

**gmpas deliberately does not compute conservative weights itself.** Doing it
properly means intersecting spherical polygons -- great-circle edges, degenerate
cells, poles, the antimeridian -- and ESMF and TempestRemap have both been
doing that correctly for years. Supersampling a target cell and counting which
source cells the samples land in, which is the obvious thing to reach for,
converges towards conservation as the sample count grows but never actually
guarantees it: the error falls off as 1/sqrt(N) and the cell integral is only
ever approximately preserved. That is not conservation, and calling it that
would be worse than not offering it.

What gmpas does own is the cheap half: writing the grid file, applying somebody
else's weights, and checking afterwards that the integral really was preserved.

See docs/REMAPPING.md for the whole terminal workflow.

The values are read from the mesh file rather than from the geometry cache.
The cache is built for drawing -- degrees, float32, polygons shifted across the
antimeridian so they render contiguously -- and none of that belongs in a file
that feeds a remapping. This reads the source arrays and converts once.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from .mesh import EARTH_RADIUS, MESH_VARS
from .paths import resolve_path

#: variables a SCRIP file is built from, beyond the mesh identity ones
SCRIP_SOURCE_VARS = ("latVertex", "lonVertex", "areaCell")

TWO_PI = 2.0 * np.pi


def write_scrip(mesh_path: str | Path, out_path: str | Path,
                title: str = "") -> tuple[Path, int]:
    """Write `mesh_path` as a SCRIP grid file at `out_path`.

    Returns the path written and how many longitudes had to be normalised onto
    [0, 2pi) -- nonzero means the source file mixed conventions, which is worth
    knowing about.

    `grid_corners` is the widest cell that actually exists, not the mesh's
    declared `maxEdges`. See the comment below: the difference decides whether
    ESMF will generate conservative weights or crash.

    Cells with fewer than `maxEdges` sides have their unused corners filled by
    repeating the cell's last real vertex. Those degenerate corners are how
    SCRIP has always represented variable-sided cells, and it is what
    `mpas_tools.scrip.from_mpas` writes too, so the file is interchangeable
    with the reference implementation's.
    """
    src = resolve_path(mesh_path)
    if not src.exists():
        raise FileNotFoundError(f"No such mesh file: {src}")

    with xr.open_dataset(src, decode_timedelta=False, engine="netcdf4") as ds:
        missing = [v for v in (*MESH_VARS, *SCRIP_SOURCE_VARS)
                   if v not in ds.variables]
        if missing:
            raise KeyError(
                f"{src.name} cannot be written as SCRIP: missing {missing}. "
                f"Pass the mesh/init/static file, or a history file that "
                f"carries its mesh."
            )

        lat_cell = np.asarray(ds.latCell.values, dtype=np.float64)
        lon_cell = np.asarray(ds.lonCell.values, dtype=np.float64)
        lat_vert = np.asarray(ds.latVertex.values, dtype=np.float64)
        lon_vert = np.asarray(ds.lonVertex.values, dtype=np.float64)
        voc = ds.verticesOnCell.values.astype(np.int64) - 1     # 1-based in file
        nedges = ds.nEdgesOnCell.values.astype(np.int64)
        area = np.asarray(ds.areaCell.values, dtype=np.float64)

        radius = float(ds.attrs.get("sphere_radius", EARTH_RADIUS) or EARTH_RADIUS)

    n_cells = voc.shape[0]

    # SCRIP wants longitude on [0, 2pi), and real MPAS files are not always
    # self-consistent about it: this one stores lonCell on [0, 2pi) reaching
    # 3.23 rad (185E) while storing lonVertex on [-pi, pi), so 12,774 vertices
    # west of the dateline come out negative. Written verbatim, a cell near the
    # seam would have its centre on one branch and its own corners on another,
    # and the polygon handed to a remapper would be nonsense. mpas_tools
    # refuses such a file outright; normalising is the same judgement, applied
    # rather than raised.
    wrapped = int(((lon_vert < 0.0) | (lon_vert >= TWO_PI)).sum()
                  + ((lon_cell < 0.0) | (lon_cell >= TWO_PI)).sum())
    lon_vert = np.mod(lon_vert, TWO_PI)
    lon_cell = np.mod(lon_cell, TWO_PI)

    # Trim to the widest cell that actually exists rather than to the maxEdges
    # dimension. MPAS declares maxEdges generously -- this mesh declares 10 and
    # uses at most 6 -- and every unused column becomes a degenerate corner
    # repeated on every cell. That padding is not free: ESMF 8.9.1 segfaults
    # outright on `-m conserve` with a source padded to 10, and completes on
    # the identical mesh trimmed to 6. mpas_tools writes the full maxEdges,
    # so its files hit the same wall.
    corners = int(nedges.max())
    voc = voc[:, :corners]

    # cells with fewer sides still repeat their last real vertex
    valid = np.arange(corners)[None, :] < nedges[:, None]
    last = voc[np.arange(n_cells), nedges - 1]
    voc = np.where(valid, voc, last[:, None])

    corner_lat = lat_vert[voc]
    corner_lon = lon_vert[voc]

    # grid_area is a solid angle, so it is areaCell on the unit sphere. A mesh
    # straight from JIGSAW already carries non-dimensional areas with
    # sphere_radius = 1, and dividing by that radius is right either way.
    grid_area = area / radius**2

    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    scrip = xr.Dataset(
        {
            "grid_dims": ("grid_rank", np.array([n_cells], dtype=np.int32)),
            "grid_center_lat": ("grid_size", lat_cell, {"units": "radians"}),
            "grid_center_lon": ("grid_size", lon_cell, {"units": "radians"}),
            "grid_corner_lat": (("grid_size", "grid_corners"), corner_lat,
                                {"units": "radians"}),
            "grid_corner_lon": (("grid_size", "grid_corners"), corner_lon,
                                {"units": "radians"}),
            "grid_area": ("grid_size", grid_area, {"units": "radian^2"}),
            "grid_imask": ("grid_size", np.ones(n_cells, dtype=np.int32),
                           {"units": "unitless"}),
        },
        attrs={
            "title": title or f"{src.name} as SCRIP",
            "source_mesh": src.name,
            "sphere_radius": radius,
            "history": "written by gmpas",
        },
    )
    scrip.to_netcdf(out)
    return out, wrapped


def coverage_of(scrip_path: str | Path) -> float:
    """Fraction of the sphere a SCRIP file's cells cover.

    A cheap sanity check on a written file: a global mesh should come out at
    1.0, and a regional one at its real fraction. Anything wildly off means the
    areas or the sphere radius went in wrong.
    """
    with xr.open_dataset(resolve_path(scrip_path), engine="netcdf4") as ds:
        return float(ds.grid_area.values.sum() / (4.0 * np.pi))
