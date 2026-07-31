"""Rasterize a native MPAS field by exact Voronoi lookup.

Polygon rendering costs O(nCells): a 2-million-cell global mesh means two
million matplotlib paths, and it crawls regardless of how the geometry was
cached. Rasterizing costs O(pixels) instead, so cost stops depending on mesh
size altogether -- a refined mesh renders as fast as a coarse one.

The lookup is exact rather than approximate. An MPAS mesh is a spherical
Voronoi tessellation of the cell centres, which *means* every point belongs to
the cell whose centre is nearest. So a single KD-tree nearest-neighbour query
gives the containing cell with no polygon clipping and no interpolation -- the
rendered value in each pixel is the model's own cell value, not a blend.

The same query is what a conservative remap needs: supersample a target cell,
look up which source cells the samples land in, and the sample counts are area
weights. That is the `convert` half of this project, sharing this machinery.
"""

from __future__ import annotations

import numpy as np

from .mesh import MpasMesh

#: above this many cells, polygon rendering stops being the faster option
RASTER_THRESHOLD = 150_000


def target_grid(extent: tuple[float, float, float, float], nx: int, ny: int):
    """Pixel-centre lon/lat arrays for a regular lat-lon target."""
    lon_min, lon_max, lat_min, lat_max = extent
    dx = (lon_max - lon_min) / nx
    dy = (lat_max - lat_min) / ny
    lon = lon_min + dx * (np.arange(nx) + 0.5)
    lat = lat_min + dy * (np.arange(ny) + 0.5)
    return lon, lat


def rasterize(mesh: MpasMesh, values: np.ndarray,
              extent: tuple[float, float, float, float] | None = None,
              nx: int = 1600, ny: int = 900, mask_outside: bool = True):
    """Sample a cell-centred field onto a regular lat-lon pixel grid.

    Returns (image, lon, lat). `image` is (ny, nx), oriented for imshow with
    origin="lower". Cells are sampled, never interpolated, so discrete fields
    and sharp gradients survive intact.

    For a regional mesh, pixels falling outside the mesh would otherwise snap to
    the nearest boundary cell and smear it across the whole frame; mask_outside
    blanks anything further from a cell centre than that cell's own radius.
    """
    extent = extent or mesh.extent
    lon, lat = target_grid(extent, nx, ny)
    lon2, lat2 = np.meshgrid(lon, lat)

    lon_r, lat_r = np.radians(lon2), np.radians(lat2)
    pts = np.stack(
        [np.cos(lat_r) * np.cos(lon_r),
         np.cos(lat_r) * np.sin(lon_r),
         np.sin(lat_r)],
        axis=-1,
    ).reshape(-1, 3)

    values = np.asarray(values, dtype=np.float64)

    if not mask_outside:
        _, idx = mesh.tree().query(pts, workers=-1)
        return values[idx].reshape(ny, nx), lon, lat

    # A cell's own radius, in metres, converted to an angle on the sphere this
    # mesh is actually on -- not on Earth. Reduced-radius ("small planet") runs
    # are a real MPAS configuration, and dividing their metres by Earth's
    # radius would understate every cell's angular size by the radius ratio,
    # blanking almost the whole domain.
    radius = np.sqrt(mesh.area_cell / np.pi) / mesh.sphere_radius

    # Bound the search at the largest cell's own cutoff. Every pixel beyond it
    # would be blanked anyway, so the result is unchanged -- but an unbounded
    # nearest-neighbour query on a point far outside the mesh cannot prune, and
    # backtracks across the whole tree. Plotting a regional mesh on a wider
    # frame spends nearly all its time on empty space without this: measured
    # 132.5 s against 0.56 s on a 414k-cell mesh drawn at global extent.
    dist, idx = mesh.tree().query(pts, workers=-1,
                                  distance_upper_bound=2.0 * float(radius.max()))

    # points with no neighbour inside the bound come back as idx == nCells
    missing = idx >= mesh.n_cells
    idx = np.where(missing, 0, idx)

    img = values[idx].reshape(ny, nx)
    too_far = missing | (dist > 2.0 * radius[idx])
    return np.where(too_far.reshape(ny, nx), np.nan, img), lon, lat


def should_raster(mesh: MpasMesh, method: str = "auto") -> bool:
    """Whether to take the raster path, given a `auto` | `poly` | `raster` choice."""
    if method == "raster":
        return True
    if method == "poly":
        return False
    if method != "auto":
        raise ValueError(f"method must be 'auto', 'poly' or 'raster', not {method!r}")
    return mesh.n_cells >= RASTER_THRESHOLD
