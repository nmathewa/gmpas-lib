"""Native-mesh rendering: cell fields, edge fields, vectors, mesh structure.

Everything here draws MPAS data on its own Voronoi mesh -- no regridding, so
variable resolution is preserved exactly as the model carries it. Geometry
comes pre-built from `mesh.MpasMesh`, so plotting is just a colour lookup.

matplotlib and cartopy are imported lazily, inside the functions that need
them, so that geometry, rasterizing and data access stay usable in a headless
install that never draws anything.
"""

from __future__ import annotations

import numpy as np

from .mesh import MpasMesh
from .raster import rasterize, should_raster
from .style import CMAPS, Style, resolve_extent


# ------------------------------------------------------------------ base map


def _frame(box, central_lon: float = 0.0):
    """Pick the projection frame a box should be drawn in.

    A box whose `lon_max` runs past +180 crosses the antimeridian, and is only
    contiguous in the unwrapped frame. Drawing it on a Greenwich-centred map
    would split it down both edges, so the map is centred on the domain
    instead and the box expressed relative to that centre.

    Returns (central_lon, source_crs, box_in_that_frame). Boxes that stay
    within +-180 are returned untouched on a Greenwich-centred frame, so
    nothing about the ordinary case changes.
    """
    import cartopy.crs as ccrs

    if box[1] > 180.0:
        if central_lon == 0.0:
            central_lon = 0.5 * (box[0] + box[1])
        src = ccrs.PlateCarree(central_longitude=central_lon)
        return central_lon, src, (box[0] - central_lon, box[1] - central_lon,
                                  box[2], box[3])
    return central_lon, ccrs.PlateCarree(), tuple(box)


def _basemap(mesh: MpasMesh, style: Style, extent, central_lon: float = 0.0,
             coastlines: bool = True):
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    import cartopy.crs as ccrs

    box = resolve_extent(mesh, extent)
    central_lon, src, framed = _frame(box, central_lon)

    proj = ccrs.PlateCarree(central_longitude=central_lon)
    fig, ax = plt.subplots(figsize=style.figsize, subplot_kw={"projection": proj},
                           constrained_layout=True)

    if coastlines:
        ax.add_feature(cfeature.COASTLINE, linewidth=style.coastline_lw)
    gl = ax.gridlines(draw_labels=True, linewidth=style.gridline_lw,
                      linestyle="--", alpha=0.5)
    gl.top_labels = gl.right_labels = False
    # geo_labels defaults True whenever draw_labels does. GeoAxes title
    # placement checks `top_labels or geo_labels` and, when true, measures
    # gl.top_label_artists even though top_labels is off here -- those Texts
    # are hidden, so matplotlib's null-bbox sentinel leaves .ymax == inf and
    # the title gets pinned there (cartopy bug). A title stuck at y=inf NaNs
    # the axes' tight bbox, so any inline display (Jupyter's default
    # bbox_inches="tight") renders nothing but the colorbar.
    gl.geo_labels = False

    if box[0] <= -179.9 and 179.9 <= box[1] <= 180.0:
        ax.set_global()
    else:
        ax.set_extent(framed, crs=src)
    return fig, ax


def _limits(values: np.ndarray, vmin, vmax, symmetric: bool, robust: bool):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo = vmin if vmin is not None else (np.percentile(finite, 2) if robust else finite.min())
    hi = vmax if vmax is not None else (np.percentile(finite, 98) if robust else finite.max())
    if symmetric:
        m = max(abs(lo), abs(hi))
        return -m, m
    return float(lo), float(hi)


def _check_len(values: np.ndarray, expected: int, kind: str, mesh: MpasMesh):
    if values.shape[-1] != expected:
        raise ValueError(
            f"field has {values.shape[-1]} values but the mesh has {expected} "
            f"{kind}. Mesh {mesh.path.name}: {mesh.n_cells} cells / "
            f"{mesh.n_edges} edges -- is this field on a different dimension, "
            f"or from a different mesh?"
        )


# --------------------------------------------------------------- cell fields


def cell_field(mesh: MpasMesh, values: np.ndarray, *, style: Style | None = None,
               cmap: str = "", vmin=None, vmax=None, symmetric: bool = False,
               robust: bool = True, extent=None, central_lon: float = 0.0,
               label: str = "", title: str = "", method: str = "auto",
               nx: int = 1600, ny: int = 900):
    """Fill each Voronoi cell with a cell-centred scalar.

    method: poly   -- one matplotlib polygon per cell, exact cell outlines
            raster -- KD-tree Voronoi lookup onto a pixel grid, O(pixels)
            auto   -- poly for small meshes, raster above ~150k cells
    """
    import cartopy.crs as ccrs
    from matplotlib.collections import PolyCollection

    style = style or Style()
    values = np.asarray(values).squeeze()
    _check_len(values, mesh.n_cells, "cells", mesh)

    lo, hi = _limits(values, vmin, vmax, symmetric, robust)
    cmap = cmap or (CMAPS["anomaly"] if symmetric else CMAPS["sequential"])

    if should_raster(mesh, method):
        return _cell_field_raster(mesh, values, style, cmap, lo, hi, extent,
                                  central_lon, label, title, nx, ny)

    fig, ax = _basemap(mesh, style, extent, central_lon)

    def _add(verts, vals):
        pc = PolyCollection(verts, array=vals, cmap=cmap, clim=(lo, hi),
                            transform=ccrs.PlateCarree(), edgecolors="face",
                            linewidths=style.edge_lw)
        ax.add_collection(pc, autolim=False)
        return pc

    pc = _add(mesh.cell_verts, values)
    # cells straddling the antimeridian were unwrapped past +180; draw a second
    # copy 360 degrees west so the seam is filled on both sides of the map
    if mesh.cell_wrapped.any():
        dup = mesh.cell_verts[mesh.cell_wrapped].copy()
        dup[..., 0] -= 360.0
        _add(dup, values[mesh.cell_wrapped])

    cb = fig.colorbar(pc, ax=ax, shrink=0.8, pad=0.02)
    if label:
        cb.set_label(label, size=style.label_size)
    if title:
        ax.set_title(title, size=style.title_size)
    return fig, ax


def _cell_field_raster(mesh, values, style, cmap, lo, hi, extent, central_lon,
                       label, title, nx, ny):
    """Raster path: cost scales with pixels, not with cell count."""
    box = resolve_extent(mesh, extent)
    # rasterize works in geographic degrees and is happy past +180, since the
    # pixel centres go straight through cos/sin onto the sphere
    img, _, _ = rasterize(mesh, values, box, nx=nx, ny=ny)

    fig, ax = _basemap(mesh, style, extent, central_lon)
    _, src, framed = _frame(box, central_lon)
    im = ax.imshow(img, origin="lower", extent=framed, cmap=cmap, vmin=lo, vmax=hi,
                   transform=src, interpolation="nearest")

    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    if label:
        cb.set_label(label, size=style.label_size)
    if title:
        ax.set_title(title, size=style.title_size)
    return fig, ax


# --------------------------------------------------------------- edge fields


def edge_field(mesh: MpasMesh, values: np.ndarray, *, style: Style | None = None,
               cmap: str = "", vmin=None, vmax=None, symmetric: bool = True,
               robust: bool = True, extent=None, central_lon: float = 0.0,
               linewidth: float = 1.2, label: str = "", title: str = ""):
    """Colour each Voronoi cell face by an edge-centred scalar.

    Edge quantities (normal velocity `u`, fluxes) live on the faces between
    cells, not at cell centres, so they are drawn as the faces themselves --
    the geometry the value actually belongs to. This is the case a cell-polygon
    GeoDataFrame cannot represent.
    """
    import cartopy.crs as ccrs
    from matplotlib.collections import LineCollection

    style = style or Style()
    values = np.asarray(values).squeeze()
    _check_len(values, mesh.n_edges, "edges", mesh)

    lo, hi = _limits(values, vmin, vmax, symmetric, robust)
    cmap = cmap or (CMAPS["anomaly"] if symmetric else CMAPS["sequential"])

    fig, ax = _basemap(mesh, style, extent, central_lon)

    def _add(segs, vals):
        lc = LineCollection(segs, array=vals, cmap=cmap, clim=(lo, hi),
                            transform=ccrs.PlateCarree(), linewidths=linewidth)
        ax.add_collection(lc, autolim=False)
        return lc

    lc = _add(mesh.edge_segs, values)
    if mesh.edge_wrapped.any():
        dup = mesh.edge_segs[mesh.edge_wrapped].copy()
        dup[..., 0] -= 360.0
        _add(dup, values[mesh.edge_wrapped])

    cb = fig.colorbar(lc, ax=ax, shrink=0.8, pad=0.02)
    if label:
        cb.set_label(label, size=style.label_size)
    if title:
        ax.set_title(title, size=style.title_size)
    return fig, ax


# ------------------------------------------------------------------- vectors


def vectors(mesh: MpasMesh, u: np.ndarray, v: np.ndarray, *,
            style: Style | None = None, extent=None, central_lon: float = 0.0,
            thin: int = 0, scale: float = 0.0,
            background: np.ndarray | None = None, cmap: str = "",
            label: str = "", title: str = ""):
    """Quiver cell-centred wind vectors, optionally over a filled field.

    `thin` keeps every Nth cell (0 picks a sane value from the mesh size);
    on a variable-resolution mesh this thins uniformly in cell index, which
    naturally samples the refined region more densely.
    """
    import cartopy.crs as ccrs

    style = style or Style()
    u = np.asarray(u).squeeze()
    v = np.asarray(v).squeeze()
    _check_len(u, mesh.n_cells, "cells", mesh)
    _check_len(v, mesh.n_cells, "cells", mesh)

    if background is not None:
        fig, ax = cell_field(mesh, background, style=style, cmap=cmap,
                             extent=extent, central_lon=central_lon,
                             label=label, title=title)
    else:
        fig, ax = _basemap(mesh, style, extent, central_lon)
        if title:
            ax.set_title(title, size=style.title_size)

    if thin <= 0:
        thin = max(1, mesh.n_cells // 2000)
    sl = slice(None, None, thin)

    # no regrid_shape: cartopy only skips regridding when it is absent/None,
    # and regridding scattered native-mesh points is exactly what we don't want
    q = ax.quiver(mesh.lon_cell[sl], mesh.lat_cell[sl], u[sl], v[sl],
                  transform=ccrs.PlateCarree(),
                  scale=scale or None, width=0.0018, color="black")
    ref = float(np.nanpercentile(np.hypot(u, v), 90))
    ax.quiverkey(q, 0.9, 1.02, ref, f"{ref:.1f}", labelpos="E", coordinates="axes")
    return fig, ax


# ------------------------------------------------------------ mesh structure


def mesh_structure(mesh: MpasMesh, *, style: Style | None = None,
                   extent=None, central_lon: float = 0.0, title: str = ""):
    """Show the mesh itself, cells coloured by their resolution in km.

    The point of MPAS is variable resolution, so this is the figure that shows
    where the refinement actually is.
    """
    style = style or Style.preset("mesh")
    width_km = mesh.cell_width_km
    return cell_field(
        mesh, width_km, style=style, cmap="Spectral_r", robust=False,
        extent=extent, central_lon=central_lon,
        label="cell width [km]",
        title=title or f"{mesh.path.name} — {mesh.n_cells} cells, "
                       f"{width_km.min():.1f}–{width_km.max():.1f} km",
    )
