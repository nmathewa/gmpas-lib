"""Rescale a regional mesh around a stereographic tangent point.

Ported from the MPAS-Tools `scale_regional_mesh.py` script: project every
cell/vertex/edge onto a stereographic plane centred on a tangent point,
divide by a scale factor, project back, then recompute everything derived
from those coordinates (`dcEdge`, `dvEdge`, `areaCell`, `areaTriangle`,
`kiteAreasOnVertex`, `weightsOnEdge`, `angleEdge`, `nominalMinDc`).

The original is a standalone script: plain Python loops over every
cell/edge/vertex, mutating a copy of the file in place with raw netCDF4.
This module instead vectorizes every independent per-element piece with
numpy -- coordinate scaling, `dcEdge`, `dvEdge`, `areaTriangle`,
`areaCell`, `kiteAreasOnVertex` -- using the same ragged-cell masking
idiom `scrip.py` already uses for MPAS's variable-sided cells. Only
`weightsOnEdge` stays a loop over edges: it walks each cell's edges in
rotated order and accumulates a running sum, a genuine sequential
dependency that isn't worth the risk of a subtle vectorization bug.
Writes a whole new file with xarray, following `write_scrip`'s pattern,
rather than mutating the input in place.

**Only for a unit-sphere mesh** -- straight off JIGSAW/mkgrid
(`gmpas prep generate`), before `init_atmosphere` redimensionalizes it.
The whole algorithm assumes `R = 1`; running it on an Earth-scaled mesh
would silently corrupt every coordinate, so `scale_mesh` checks
`sphere_radius` and refuses rather than guessing.

`plot_comparison` renders a before/after cell-width map as a quick visual
check that a scale did what was asked. matplotlib and cartopy are imported
lazily inside it only, same as everywhere else in gmpas that draws --
`scale_mesh` itself never needs them, and stays usable in a headless
install.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import xarray as xr

from ..mesh import MpasMesh
from ..paths import resolve_path
from ..style import CMAPS

#: variables scale_mesh reads and recomputes, beyond mesh identity
REQUIRED_VARS = (
    "latCell", "lonCell", "xCell", "yCell", "zCell",
    "latVertex", "lonVertex", "xVertex", "yVertex", "zVertex",
    "latEdge", "lonEdge", "xEdge", "yEdge", "zEdge",
    "nEdgesOnCell", "verticesOnCell", "edgesOnCell",
    "cellsOnEdge", "verticesOnEdge", "edgesOnEdge",
    "cellsOnVertex", "edgesOnVertex",
    "dcEdge", "dvEdge", "areaCell", "areaTriangle",
    "kiteAreasOnVertex", "weightsOnEdge", "angleEdge",
    "nominalMinDc",
)


# ------------------------------------------------------------- geometry

def lonlat_to_xyz(lon: np.ndarray, lat: np.ndarray,
                  radius: float = 1.0) -> np.ndarray:
    """Lon/lat (radians) to xyz on a sphere of the given radius.

    Returns an array one dimension larger than the input, with xyz stacked
    on the last axis -- broadcasts over any leading shape.
    """
    lon, lat = np.asarray(lon), np.asarray(lat)
    return np.stack([
        radius * np.cos(lon) * np.cos(lat),
        radius * np.sin(lon) * np.cos(lat),
        radius * np.sin(lat),
    ], axis=-1)


def xyz_to_lonlat(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """xyz (last axis length 3) to lon/lat, radians.

    Normalizes by the vector's own norm rather than trusting radius = 1
    exactly -- the same defensive move `mesh.py` makes before its own
    xCell/yCell/zCell go into a KD-tree, for the same reason: float error
    accumulates, and asin of something a hair over 1 is nan.
    """
    xyz = np.asarray(xyz)
    norm = np.linalg.norm(xyz, axis=-1)
    lon = np.arctan2(xyz[..., 1], xyz[..., 0])
    lat = np.arcsin(np.clip(xyz[..., 2] / norm, -1.0, 1.0))
    return lon, lat


def stereo_project(lam: np.ndarray, phi: np.ndarray,
                   lam_0: float, phi_1: float) -> tuple[np.ndarray, np.ndarray]:
    """Stereographic projection onto the plane tangent at (lam_0, phi_1)."""
    k = 2.0 / (1.0 + np.sin(phi_1) * np.sin(phi)
              + np.cos(phi_1) * np.cos(phi) * np.cos(lam - lam_0))
    x = k * np.cos(phi) * np.sin(lam - lam_0)
    y = k * (np.cos(phi_1) * np.sin(phi)
            - np.sin(phi_1) * np.cos(phi) * np.cos(lam - lam_0))
    return x, y


def stereo_inverse(x: np.ndarray, y: np.ndarray,
                   lam_0: float, phi_1: float) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of `stereo_project`.

    `rho == 0` is the tangent point itself, projected onto the plane's
    origin -- the formula's own y/rho term is 0/0 there, so it is handled
    explicitly rather than left to divide-by-zero warnings.
    """
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    rho = np.sqrt(x * x + y * y)
    c = 2.0 * np.arctan2(rho, 2.0)
    y_over_rho = np.divide(y, rho, out=np.zeros_like(rho), where=rho > 0)

    phi = np.arcsin(np.clip(
        np.cos(c) * np.sin(phi_1) + y_over_rho * np.sin(c) * np.cos(phi_1),
        -1.0, 1.0,
    ))
    lam = lam_0 + np.arctan2(
        x * np.sin(c),
        rho * np.cos(phi_1) * np.cos(c) - y * np.sin(phi_1) * np.sin(c),
    )
    return lam, phi


def chord_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Straight-line distance between two points, last axis = xyz."""
    d = np.asarray(b) - np.asarray(a)
    return np.sqrt(np.sum(d * d, axis=-1))


def great_circle_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Great-circle distance on a unit sphere between two xyz points."""
    return 2.0 * np.arcsin(np.clip(chord_distance(a, b) / 2.0, -1.0, 1.0))


def spherical_triangle_area(a: np.ndarray, b: np.ndarray,
                            c: np.ndarray) -> np.ndarray:
    """Area of the spherical triangle with unit-sphere xyz vertices a, b, c.

    L'Huilier's theorem, via the half-angle tangent form -- the same
    formula the original script's `triangle_area` uses, vectorized to
    broadcast over any leading shape instead of three scalars.
    """
    a_side = great_circle_distance(b, c)
    b_side = great_circle_distance(a, c)
    c_side = great_circle_distance(a, b)
    s = 0.5 * (a_side + b_side + c_side)
    tan_qe_sq = (np.tan(0.5 * s) * np.tan(0.5 * (s - a_side))
                * np.tan(0.5 * (s - b_side)) * np.tan(0.5 * (s - c_side)))
    tan_qe = np.sqrt(np.clip(tan_qe_sq, 0.0, None))
    return 4.0 * np.arctan(tan_qe)


def spherical_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle at vertex A between spherical arcs AB and AC.

    Eqn. (28) at mathworld.wolfram.com/SphericalTrigonometry.html, the same
    one the original script's `sphere_angle` cites. Vectorized: `np.cross`
    and an axis-aware dot product (`einsum`) replace the single triple
    product, broadcasting over any leading shape.
    """
    a_side = great_circle_distance(b, c)
    b_side = great_circle_distance(a, c)
    c_side = great_circle_distance(a, b)
    s = 0.5 * (a_side + b_side + c_side)
    ratio = (np.sin(s - b_side) * np.sin(s - c_side)) / (np.sin(b_side) * np.sin(c_side))
    sin_angle = np.sqrt(np.clip(ratio, 0.0, 1.0))

    ab = np.asarray(b) - np.asarray(a)
    ac = np.asarray(c) - np.asarray(a)
    d = np.cross(ab, ac)
    sign = np.where(np.einsum("...i,...i->...", d, a) >= 0.0, 1.0, -1.0)
    return sign * 2.0 * np.arcsin(np.clip(sin_angle, -1.0, 1.0))


def _scale_location(lon: np.ndarray, lat: np.ndarray, tan_lon: float,
                    tan_lat: float, scale_factor: float):
    """Project, scale, and re-project a set of points -- Cell, Vertex or
    Edge alike. Returns (lon, lat, x, y, z), unit-sphere xyz."""
    x, y = stereo_project(lon, lat, tan_lon, tan_lat)
    new_lon, new_lat = stereo_inverse(x / scale_factor, y / scale_factor,
                                      tan_lon, tan_lat)
    xyz = lonlat_to_xyz(new_lon, new_lat)
    return new_lon, new_lat, xyz[..., 0], xyz[..., 1], xyz[..., 2]


# ------------------------------------------------------------ weightsOnEdge

def _sign_for_edge(cell_id: int, edge_id: int, cells_on_edge: np.ndarray) -> float:
    return 1.0 if cell_id == cells_on_edge[edge_id, 0] else -1.0


def _rotate_kite_areas(kite_row: np.ndarray, edge_id: int,
                       edges_on_cell_row: np.ndarray, n: int) -> np.ndarray:
    """`kite_row`'s first `n` entries, rotated to start just after `edge_id`.

    `verticesOnCell`/`kiteAreasOnCell` lead `edgesOnCell` by one position,
    so the entry that "belongs" to `edge_id` is the next one round -- same
    convention the original script's loop-based rotation used, just as a
    single `np.roll` instead of two nested copy loops.
    """
    row = kite_row[:n]
    matches = np.flatnonzero(edges_on_cell_row[:n] == edge_id)
    istart = int((matches[0] + 1) % n) if matches.size else 0
    return np.roll(row, -istart)


def _weights_on_edge(kite_areas_on_cell, dc_edge, dv_edge, cells_on_edge,
                     edges_on_cell, edges_on_edge, n_edges_on_cell,
                     weights_on_edge_old):
    """The one piece with a genuine sequential dependency -- see the module
    docstring. `weights_on_edge_old` is copied and only the two cells'
    worth of columns per edge are overwritten, matching the original."""
    n_edges = cells_on_edge.shape[0]
    out = weights_on_edge_old.astype(np.float64).copy()

    for i in range(n_edges):
        cell1, cell2 = cells_on_edge[i]
        if cell1 < 0 or cell2 < 0:
            continue

        j = 0
        for cell, sign_flip in ((cell1, 1.0), (cell2, -1.0)):
            n = int(n_edges_on_cell[cell])
            rotated = _rotate_kite_areas(kite_areas_on_cell[cell], i,
                                         edges_on_cell[cell], n)
            sum_r = 0.0
            for ii in range(n - 1):
                sum_r += rotated[ii]
                edge_id = edges_on_edge[i, j]
                out[i, j] = (sign_flip * _sign_for_edge(cell, edge_id, cells_on_edge)
                            * (0.5 - sum_r) * dv_edge[edge_id] / dc_edge[i])
                j += 1

    return out


# ------------------------------------------------------------- regional check

#: past this angular distance from the tangent point, the stereographic
#: scale has visibly drifted from the requested factor (measured: at 60
#: degrees a factor of 2 already behaves like 1.63; past 120 degrees it
#: starts making cells *coarser* instead of finer, and near the antipode it
#: is close to the exact reciprocal of what was asked for)
REGIONAL_WARNING_DEG = 45.0


def max_angular_distance_deg(mesh_path: str | Path, tan_lat_deg: float,
                             tan_lon_deg: float) -> float:
    """How far the mesh's furthest cell sits from a tangent point, in
    degrees. Cheap -- reads only `lonCell`/`latCell`, not the whole mesh --
    so it's worth checking before `scale_mesh` does its much more expensive
    recompute. See `REGIONAL_WARNING_DEG` for why this matters: this whole
    module is a regional-mesh tool, correct only close to the tangent point.
    """
    with xr.open_dataset(resolve_path(mesh_path), decode_timedelta=False,
                         engine="netcdf4") as ds:
        lon = ds["lonCell"].values
        lat = ds["latCell"].values

    tan_xyz = lonlat_to_xyz(math.radians(tan_lon_deg), math.radians(tan_lat_deg))
    cell_xyz = lonlat_to_xyz(lon, lat)
    c = great_circle_distance(np.broadcast_to(tan_xyz, cell_xyz.shape), cell_xyz)
    return math.degrees(float(c.max()))


# ---------------------------------------------------------------- orchestrator

def scale_mesh(mesh_path: str | Path, out_path: str | Path,
               scale_factor: float, tan_lat_deg: float,
               tan_lon_deg: float) -> Path:
    """Rescale a regional mesh around (tan_lat_deg, tan_lon_deg) and write
    it to `out_path`. Values > 1 increase resolution (shrink cells).

    Reads the whole mesh into memory, recomputes every array derived from
    cell/vertex/edge position, and writes a complete new file -- everything
    else (dims, attrs, variables this doesn't touch) passes through
    unchanged. Never mutates `mesh_path`.
    """
    src = resolve_path(mesh_path)
    if not src.exists():
        raise FileNotFoundError(f"No such mesh file: {src}")

    with xr.open_dataset(src, decode_timedelta=False, engine="netcdf4") as ds:
        missing = [v for v in REQUIRED_VARS if v not in ds.variables]
        if missing:
            raise KeyError(
                f"{src.name} cannot be scaled: missing {missing}. "
                f"gmpas prep scale expects a full mesh straight out of "
                f"JIGSAW/mkgrid (gmpas prep generate), not a trimmed file "
                f"or one that has already been through init_atmosphere."
            )
        radius = float(ds.attrs.get("sphere_radius", 1.0) or 1.0)
        if not math.isclose(radius, 1.0, rel_tol=1e-6):
            raise ValueError(
                f"{src.name} has sphere_radius={radius}, not a unit sphere. "
                f"gmpas prep scale only supports a mesh straight off "
                f"JIGSAW/mkgrid, before init_atmosphere redimensionalizes "
                f"it -- scaling a metres-scale mesh with this formula would "
                f"silently corrupt its coordinates."
            )
        ds = ds.load()

    tan_lat = math.radians(tan_lat_deg)
    tan_lon = math.radians(tan_lon_deg)

    n_cells = ds.sizes["nCells"]

    # -- recompute coordinates, one call per location type -------------
    cell_lon, cell_lat, cell_x, cell_y, cell_z = _scale_location(
        ds["lonCell"].values, ds["latCell"].values, tan_lon, tan_lat, scale_factor)
    vtx_lon, vtx_lat, vtx_x, vtx_y, vtx_z = _scale_location(
        ds["lonVertex"].values, ds["latVertex"].values, tan_lon, tan_lat, scale_factor)
    edge_lon, edge_lat, edge_x, edge_y, edge_z = _scale_location(
        ds["lonEdge"].values, ds["latEdge"].values, tan_lon, tan_lat, scale_factor)

    cell_xyz = np.stack([cell_x, cell_y, cell_z], axis=-1)
    vtx_xyz = np.stack([vtx_x, vtx_y, vtx_z], axis=-1)

    # -- connectivity, 1-based/0-fill in the file -> 0-based/-1-fill here
    voc = ds["verticesOnCell"].values.astype(np.int64) - 1     # (nCells, maxEdges)
    eoc = ds["edgesOnCell"].values.astype(np.int64) - 1        # (nCells, maxEdges)
    n_edges_on_cell = ds["nEdgesOnCell"].values.astype(np.int64)
    max_edges = voc.shape[1]

    coe = ds["cellsOnEdge"].values.astype(np.int64) - 1        # (nEdges, 2)
    voe = ds["verticesOnEdge"].values.astype(np.int64) - 1     # (nEdges, 2)
    eoe = ds["edgesOnEdge"].values.astype(np.int64) - 1        # (nEdges, maxEdges2)

    cov = ds["cellsOnVertex"].values.astype(np.int64) - 1      # (nVertices, vertexDegree)
    eov = ds["edgesOnVertex"].values.astype(np.int64) - 1      # (nVertices, vertexDegree)
    vertex_degree = cov.shape[1]

    scale_sq = scale_factor ** 2

    # -- dcEdge: distance between the two cells on an edge --------------
    valid_dc = (coe[:, 0] >= 0) & (coe[:, 1] >= 0)
    coe_safe = np.where(coe >= 0, coe, 0)
    dc_computed = great_circle_distance(cell_xyz[coe_safe[:, 0]], cell_xyz[coe_safe[:, 1]])
    dc_edge = np.where(valid_dc, dc_computed, ds["dcEdge"].values / scale_factor)

    # -- dvEdge: distance between the two vertices on an edge ------------
    # every edge has exactly two vertices -- no boundary fallback needed
    dv_edge = great_circle_distance(vtx_xyz[voe[:, 0]], vtx_xyz[voe[:, 1]])

    # -- areaTriangle: one per vertex, from its 3 surrounding cells -------
    # MPAS's vertex degree is always 3 (a Voronoi mesh's dual is a Delaunay
    # triangulation); this assumes exactly that, as the original script does.
    valid_tri = (cov >= 0).all(axis=1)
    cov_safe = np.where(cov >= 0, cov, 0)
    area_triangle = np.where(
        valid_tri,
        spherical_triangle_area(cell_xyz[cov_safe[:, 0]], cell_xyz[cov_safe[:, 1]],
                                cell_xyz[cov_safe[:, 2]]),
        ds["areaTriangle"].values / scale_sq,
    )

    # -- areaCell: ragged sum of per-edge triangles ----------------------
    j = np.arange(max_edges)
    valid_j = j[None, :] < n_edges_on_cell[:, None]
    denom = np.maximum(n_edges_on_cell[:, None], 1)
    j_next = (j[None, :] + 1) % denom
    voc_safe = np.where(valid_j, voc, 0)
    voc_next = np.take_along_axis(voc_safe, j_next, axis=1)

    tri = spherical_triangle_area(
        np.broadcast_to(cell_xyz[:, None, :], (n_cells, max_edges, 3)),
        vtx_xyz[voc_safe], vtx_xyz[voc_next],
    )
    area_cell = np.where(valid_j, tri, 0.0).sum(axis=1)

    # -- kiteAreasOnVertex: two triangles per (vertex, local cell) --------
    k_next = (np.arange(vertex_degree) + 1) % vertex_degree
    eov1, eov2 = eov, eov[:, k_next]
    valid_kite = (cov >= 0) & (eov1 >= 0) & (eov2 >= 0)
    eov1_safe = np.where(eov1 >= 0, eov1, 0)
    eov2_safe = np.where(eov2 >= 0, eov2, 0)

    edge_xyz = np.stack([edge_x, edge_y, edge_z], axis=-1)
    vtx_b = np.broadcast_to(vtx_xyz[:, None, :], (vtx_xyz.shape[0], vertex_degree, 3))
    kite_computed = (
        spherical_triangle_area(vtx_b, edge_xyz[eov1_safe], cell_xyz[cov_safe])
        + spherical_triangle_area(vtx_b, cell_xyz[cov_safe], edge_xyz[eov2_safe])
    )
    kite_areas_on_vertex = np.where(
        valid_kite, kite_computed, ds["kiteAreasOnVertex"].values / scale_sq)

    # -- kiteAreasOnCell: kiteAreasOnVertex, gathered back onto cells -----
    cov_at_voc = cov[voc_safe]                    # (nCells, maxEdges, vertexDegree)
    match = cov_at_voc == np.arange(n_cells)[:, None, None]
    found = match.any(axis=-1)
    k_found = np.argmax(match, axis=-1)
    if np.any(valid_j & ~found):
        print("warning: some cells were not found in their vertices' "
             "cellsOnVertex list -- kiteAreasOnCell left at 0 there")
    kite_gather = kite_areas_on_vertex[voc_safe, k_found]
    kite_areas_on_cell = np.where(valid_j & found, kite_gather, 0.0) / area_cell[:, None]

    # -- weightsOnEdge: the one loop -------------------------------------
    weights_on_edge = _weights_on_edge(
        kite_areas_on_cell, dc_edge, dv_edge, coe, eoc, eoe, n_edges_on_cell,
        ds["weightsOnEdge"].values)

    # -- angleEdge --------------------------------------------------------
    a_pts = vtx_xyz[voe[:, 0]]
    tangent = np.stack([
        np.cos(edge_lon) * np.sin(edge_lat),
        np.sin(edge_lon) * np.sin(edge_lat),
        -np.cos(edge_lat),
    ], axis=-1)
    b_pts = a_pts - tangent
    b_pts = b_pts / np.linalg.norm(b_pts, axis=-1, keepdims=True)
    c_pts = vtx_xyz[voe[:, 1]]
    angle_edge = spherical_angle(a_pts, b_pts, c_pts)

    updates = {
        "lonCell": cell_lon, "latCell": cell_lat,
        "xCell": cell_x, "yCell": cell_y, "zCell": cell_z,
        "lonVertex": vtx_lon, "latVertex": vtx_lat,
        "xVertex": vtx_x, "yVertex": vtx_y, "zVertex": vtx_z,
        "lonEdge": edge_lon, "latEdge": edge_lat,
        "xEdge": edge_x, "yEdge": edge_y, "zEdge": edge_z,
        "nominalMinDc": ds["nominalMinDc"].values / scale_factor,
        "dcEdge": dc_edge, "dvEdge": dv_edge,
        "areaTriangle": area_triangle, "areaCell": area_cell,
        "kiteAreasOnVertex": kite_areas_on_vertex,
        "weightsOnEdge": weights_on_edge,
        "angleEdge": angle_edge,
    }
    for name, values in updates.items():
        da = ds[name]
        ds[name] = (da.dims, np.asarray(values).astype(da.dtype), da.attrs)

    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out)
    return out


# --------------------------------------------------------------- comparison

def plot_comparison(old_mesh_path: str | Path, new_mesh_path: str | Path,
                    out_png: str | Path) -> Path:
    """Before/after cell-width maps, side by side, on one shared colour
    scale -- a quick visual check that a scale actually did what was asked:
    resolution changed where expected, the domain pulled in toward the
    tangent point, and so on. Returns the PNG path.

    Deliberately not a full `plot.cell_field` reuse: this is a rough
    sanity-check image, not a publication figure, and a self-contained
    version here keeps this feature from touching the shared rendering path
    every other plot goes through.
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    before = MpasMesh.load(old_mesh_path)
    after = MpasMesh.load(new_mesh_path)

    lo = min(before.cell_width_km.min(), after.cell_width_km.min())
    hi = max(before.cell_width_km.max(), after.cell_width_km.max())
    cmap = CMAPS["sequential"]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.5),
                             subplot_kw={"projection": ccrs.PlateCarree()},
                             constrained_layout=True)

    pc = None
    for ax, mesh, label in ((axes[0], before, "before"), (axes[1], after, "after")):
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, linestyle="--", alpha=0.5)
        # geo_labels defaults True alongside draw_labels and, left alone,
        # pins the title at y=inf when top_labels is off, which NaNs the
        # axes' tight bbox and crops the figure to its colorbar under any
        # bbox_inches="tight" render (Jupyter's inline backend defaults to
        # exactly that). See plot._basemap for the full cartopy bug.
        gl.top_labels = gl.right_labels = gl.geo_labels = False

        if mesh.is_global:
            ax.set_global()
        else:
            ax.set_extent(mesh.extent, crs=ccrs.PlateCarree())

        pc = ax.add_collection(PolyCollection(
            mesh.cell_verts, array=mesh.cell_width_km, cmap=cmap,
            clim=(lo, hi), transform=ccrs.PlateCarree(), edgecolors="face"))
        if mesh.cell_wrapped.any():
            # a second copy 360 degrees west, so a cell straddling the
            # antimeridian doesn't smear across the map -- see plot.cell_field
            dup = mesh.cell_verts[mesh.cell_wrapped].copy()
            dup[..., 0] -= 360.0
            ax.add_collection(PolyCollection(
                dup, array=mesh.cell_width_km[mesh.cell_wrapped], cmap=cmap,
                clim=(lo, hi), transform=ccrs.PlateCarree(), edgecolors="face"))

        ax.set_title(f"{label}: {mesh.n_cells:,} cells, "
                    f"{mesh.cell_width_km.min():.1f}-{mesh.cell_width_km.max():.1f} km")

    fig.colorbar(pc, ax=list(axes), shrink=0.8, pad=0.02, label="cell width [km]")

    out = Path(out_png).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    # deliberately no bbox_inches="tight" -- see the geo_labels note above
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
