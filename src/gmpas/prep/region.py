"""Crop a global (or larger regional) mesh down to a boundary-defined subset.

Independent implementation -- not a port. MPAS-Dev/MPAS-Limited-Area does
the same job and is the tool referenced from `gmpas prep scale`'s docs, but
carries no LICENSE file (no explicit grant to redistribute or adapt its
source), unlike MPAS-Tools (used for `scale`/`relocate`), which is
permissively licensed. So this module is built from first principles --
graph traversal over `cellsOnCell` -- and from MPAS-Model's own public
Registry.xml/Fortran source for the one piece that is a real file-format
contract with the model, not an implementation detail: see `N_SPEC_ZONE`/
`N_RELAX_ZONE` below.

The shape:

1. Snap each boundary polygon vertex to its nearest cell (a cKDTree over
   cell centres, same technique `mesh.py`'s `MpasMesh.cell_of` already uses
   elsewhere in this package).
2. Walk the mesh graph between consecutive boundary cells, staying close to
   the great-circle arc between them -- `_walk_boundary`.
3. Flood-fill from an interior point out to that boundary -- `_flood_fill`.
   Together, 2 and 3 mark every cell inside the requested polygon as
   relaxation zone 0.
4. Grow `N_BDY_LAYERS` more rings of cells outward beyond the polygon --
   `_relaxation_zones` -- these are the cells `init_atmosphere`/the
   atmosphere core relax toward driving boundary data at runtime, not part
   of what the user asked for but required for the regional file to be
   usable.
5. Derive `bdyMaskEdge`/`bdyMaskVertex` from the cell zones -- `_edge_zone`/
   `_vertex_zone`.
6. Subset every variable to the kept cells/edges/vertices and renumber every
   connectivity array from the old (global) index space to the new one,
   preserving the 1-based/0-fill convention -- `_subset_and_remap`.
7. Write a `graph.info` METIS adjacency file for the regional subset --
   `write_graph_info` -- since there is no `mkgrid` run to produce one, as
   there is for a freshly generated global mesh.

`plot_region` renders the cropped mesh coloured by `bdyMaskCell`, as a quick
visual check that the kept cells and relaxation rings look right -- reusing
`plot.cell_field` rather than a custom rendering path, unlike `scale.py`'s
`plot_comparison`: that one needs two different meshes side by side on a
shared colour scale, which `cell_field` doesn't support, but this is a
single mesh and a single field, exactly `cell_field`'s own shape. matplotlib
and cartopy are imported lazily inside it only, same as everywhere else in
gmpas that draws -- `create_region` itself never needs them.

A concave region (or one spanning a pole or a large longitude range on a
`grid.nc`, before static interpolation) can produce incorrect terrain during
`init_atmosphere`'s static-field interpolation -- a property of that
downstream interpolation, not of this cropping step, but worth keeping in
mind when drawing `--polygon`. Prefer a convex boundary, or subset a
`static.nc` (post-interpolation) instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from ..paths import resolve_path
from .scale import great_circle_distance, lonlat_to_xyz, xyz_to_lonlat

#: MPAS-Model's own boundary-zone layout (mpas_atm_boundaries.F,
#: mpas_init_atm_cases.F): the outermost `N_SPEC_ZONE` layers are
#: prescribed directly from driving boundary data every step; the
#: `N_RELAX_ZONE` layers inward of those are blended toward it with
#: decreasing weight. `bdyMaskCell` is registered in Registry.xml with
#: `default_value="0"`, and `blend_bdy_terrain`'s own docstring states "for
#: global meshes, where bdyMaskCell == 0 everywhere, this routine will have
#: no impact" -- 0 is the deep interior, unrelated to the boundary, not a
#: "dropped" sentinel. This is a file-format contract with the model, not an
#: implementation choice: getting it wrong produces a file MPAS accepts but
#: relaxes incorrectly, silently.
N_SPEC_ZONE = 2
N_RELAX_ZONE = 5
N_BDY_LAYERS = N_SPEC_ZONE + N_RELAX_ZONE   # 7

#: value used to mean "no kept neighbour on this side" in the zone-of-edge/
#: vertex reductions below -- must sort after every real zone value (0..7)
_NO_NEIGHBOUR = N_BDY_LAYERS + 1

#: connectivity variables and the id-space their *values* belong to (as
#: opposed to the dimension they are indexed *by* -- e.g. `verticesOnCell`
#: is indexed by nCells but its entries are vertex ids). Only variables
#: present in a given mesh are touched.
CONNECTIVITY_VARS = {
    "cellsOnCell": "nCells", "cellsOnEdge": "nCells", "cellsOnVertex": "nCells",
    "indexToCellID": "nCells",
    "edgesOnCell": "nEdges", "edgesOnEdge": "nEdges", "edgesOnVertex": "nEdges",
    "indexToEdgeID": "nEdges",
    "verticesOnCell": "nVertices", "verticesOnEdge": "nVertices",
    "indexToVertexID": "nVertices",
}

#: variables `create_region` needs to read directly, beyond mesh identity
REQUIRED_VARS = (
    "latCell", "lonCell", "xCell", "yCell", "zCell",
    "nEdgesOnCell", "cellsOnCell", "verticesOnCell", "edgesOnCell",
    "cellsOnEdge", "verticesOnEdge", "cellsOnVertex", "edgesOnVertex",
)


# --------------------------------------------------------------- geometry

def _ragged_neighbours(cells_on_cell: np.ndarray, n_edges_on_cell: np.ndarray,
                       cell: int) -> np.ndarray:
    """0-based neighbour cell ids of `cell`, its ragged tail and any
    "no neighbour" (0) slots dropped."""
    row = cells_on_cell[cell, : n_edges_on_cell[cell]]
    return row[row > 0] - 1


def _nearest_cells(cell_xyz: np.ndarray, lon_deg: np.ndarray,
                   lat_deg: np.ndarray) -> np.ndarray:
    """0-based index of the cell nearest each (lon_deg, lat_deg) point."""
    from scipy.spatial import cKDTree

    pts = lonlat_to_xyz(np.radians(lon_deg), np.radians(lat_deg))
    return cKDTree(cell_xyz).query(np.atleast_2d(pts))[1]


def _walk_boundary(cells_on_cell: np.ndarray, n_edges_on_cell: np.ndarray,
                   cell_xyz: np.ndarray, boundary_cells: np.ndarray) -> np.ndarray:
    """Cells along the closed path connecting consecutive boundary cells,
    each segment staying close to the great-circle arc between its
    endpoints. Returns a boolean mask over all cells.

    Each step must strictly not increase the great-circle distance to the
    segment's target -- among neighbours that satisfy that, the one closest
    to the source/target great-circle plane is taken. The distance
    constraint guarantees termination on a finite mesh; ties are broken by
    lowest cell index, for determinism.
    """
    n_cells = cell_xyz.shape[0]
    on_boundary = np.zeros(n_cells, dtype=bool)
    on_boundary[boundary_cells] = True

    n = len(boundary_cells)
    for k in range(n):
        source, target = int(boundary_cells[k]), int(boundary_cells[(k + 1) % n])
        if source == target:
            continue

        normal = np.cross(cell_xyz[source], cell_xyz[target])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            raise ValueError(
                "gmpas prep create-region: two boundary points are "
                "antipodal (or nearly so), which leaves the great-circle "
                "arc between them undefined. Add an intermediate boundary "
                "point between them."
            )
        normal = normal / norm

        current = source
        target_xyz = cell_xyz[target]
        steps = 0
        while current != target:
            steps += 1
            if steps > n_cells:
                raise ValueError(
                    "gmpas prep create-region: the boundary walk did not "
                    "reach its target cell within the mesh size -- the "
                    "mesh may be disconnected between these two points."
                )
            dist_current = great_circle_distance(cell_xyz[current], target_xyz)
            candidates = _ragged_neighbours(cells_on_cell, n_edges_on_cell, current)
            if candidates.size == 0:
                raise ValueError(
                    f"gmpas prep create-region: cell {current} has no "
                    f"neighbours while walking the boundary."
                )
            dist_to_target = great_circle_distance(cell_xyz[candidates], target_xyz)
            progressing = candidates[dist_to_target <= dist_current + 1e-12]
            if progressing.size == 0:
                raise ValueError(
                    "gmpas prep create-region: the boundary walk got stuck "
                    "with no neighbour making progress toward the next "
                    "boundary point -- the mesh may be too coarse for "
                    "this boundary, or the points too close together."
                )
            deviation = np.abs(cell_xyz[progressing] @ normal)
            current = int(progressing[np.argmin(deviation)])
            on_boundary[current] = True

    return on_boundary


def _flood_fill(cells_on_cell: np.ndarray, n_edges_on_cell: np.ndarray,
                start_cell: int, on_boundary: np.ndarray) -> np.ndarray:
    """Every cell reachable from `start_cell` without crossing `on_boundary`
    -- the requested region's interior, plus the boundary path itself."""
    in_region = on_boundary.copy()
    in_region[start_cell] = True
    stack = [start_cell]
    while stack:
        cell = stack.pop()
        for neighbour in _ragged_neighbours(cells_on_cell, n_edges_on_cell, cell):
            if not in_region[neighbour]:
                in_region[neighbour] = True
                stack.append(int(neighbour))
    return in_region


def _relaxation_zones(cells_on_cell: np.ndarray, n_edges_on_cell: np.ndarray,
                      in_region: np.ndarray, n_layers: int) -> np.ndarray:
    """Zone index per cell: 0 for `in_region`, 1..n_layers outward from it
    by graph distance, -1 for cells never reached (dropped from the
    subset). Vectorized multi-source BFS -- one array pass per layer,
    rather than a per-cell Python loop, so this stays fast on a real
    (possibly multi-million-cell) global mesh.
    """
    n_cells = in_region.shape[0]
    max_edges = cells_on_cell.shape[1]
    zone = np.where(in_region, 0, -1)

    valid_col = np.arange(max_edges)[None, :] < n_edges_on_cell[:, None]
    frontier = in_region.copy()
    for layer in range(1, n_layers + 1):
        rows = np.flatnonzero(frontier)
        if rows.size == 0:
            break
        cand = cells_on_cell[rows]
        cand_valid = valid_col[rows] & (cand > 0)
        newly = np.unique(cand[cand_valid]) - 1
        newly = newly[zone[newly] == -1]
        if newly.size == 0:
            frontier = np.zeros(n_cells, dtype=bool)
            continue
        zone[newly] = layer
        frontier = np.zeros(n_cells, dtype=bool)
        frontier[newly] = True

    return zone


def _neighbour_zone(zone: np.ndarray, cells_on: np.ndarray) -> np.ndarray:
    """An edge's/vertex's own zone, derived from the zones of its
    surrounding cells.

    An element fully surrounded by kept cells takes the *more interior*
    (lower) of their zones -- it follows the most-restrictive neighbour
    inward. But an element with at least one side missing (0-fill, or a
    neighbour this crop dropped entirely) sits structurally at the outer
    rim of whatever it does touch, with nothing beyond to inform it -- it
    takes the *more boundary-like* (higher, i.e. max) of the sides it does
    have, not the lower. An edge only ever has 2 sides, so this distinction
    is invisible there (one reached side makes min and max the same value)
    -- it only shows up on a vertex with 3+ neighbours, some kept at
    different zones and at least one not. An element with no kept
    neighbours at all gets a sentinel above `N_BDY_LAYERS`, so the caller's
    keep-filter drops it.
    """
    zone_ext = np.append(zone, -1)                     # index n_cells == "none"
    idx = np.where(cells_on > 0, cells_on - 1, zone_ext.shape[0] - 1)
    side_zone = zone_ext[idx]
    reached = side_zone >= 0
    n_reached = reached.sum(axis=1)

    min_reached = np.where(reached, side_zone, N_BDY_LAYERS + 1).min(axis=1)
    max_reached = np.where(reached, side_zone, -1).max(axis=1)
    all_reached = n_reached == cells_on.shape[1]

    result = np.where(all_reached, min_reached, max_reached)
    result = np.where(n_reached == 0, _NO_NEIGHBOUR, result)
    return result


def _spherical_centroid(lon_deg: np.ndarray, lat_deg: np.ndarray) -> tuple[float, float]:
    """A reasonable default interior point for a polygon: the mean of its
    vertices' unit-sphere xyz, renormalized and projected back. Exact for a
    regular/convex polygon's centre; a caller with a concave or unusual
    shape should pass `--point` explicitly instead."""
    xyz = lonlat_to_xyz(np.radians(lon_deg), np.radians(lat_deg)).mean(axis=0)
    lon, lat = xyz_to_lonlat(xyz)
    return float(np.degrees(lon)), float(np.degrees(lat))


# ---------------------------------------------------------- subset/remap

def _build_remap(kept_0based: np.ndarray, n_total: int) -> np.ndarray:
    """`remap[v]` for a raw 1-based-with-0-fill value `v`: 0 stays 0 (no
    neighbour); a kept old index becomes its new 1-based position; a
    dropped old index becomes 0 (its former neighbours now have no
    neighbour on that side, same convention as a true mesh edge)."""
    new_id = np.zeros(n_total, dtype=np.int64)
    new_id[kept_0based] = np.arange(1, kept_0based.size + 1)
    return np.concatenate([[0], new_id])


def _subset_and_remap(ds: xr.Dataset, kept: dict[str, np.ndarray]) -> xr.Dataset:
    """Subset every variable to `kept` indices per dimension and renumber
    every connectivity array's values into the new index space."""
    isel = {dim: idx for dim, idx in kept.items() if dim in ds.dims}
    out = ds.isel(**isel)

    remaps = {dim: _build_remap(idx, ds.sizes[dim]) for dim, idx in isel.items()}
    for name, id_space in CONNECTIVITY_VARS.items():
        if name in out.variables and id_space in remaps:
            da = out[name]
            remapped = remaps[id_space][da.values]
            out[name] = (da.dims, remapped.astype(da.dtype), da.attrs)

    return out


# ---------------------------------------------------------------- orchestrator

def create_region(mesh_path: str | Path, out_path: str | Path,
                  boundary_lat_deg, boundary_lon_deg,
                  point_lat_deg: float | None = None,
                  point_lon_deg: float | None = None) -> Path:
    """Crop `mesh_path` to the region bounded by the polygon
    (`boundary_lat_deg`, `boundary_lon_deg`) -- at least 3 vertices, in
    order (winding direction does not matter; the boundary walk and flood
    fill do not depend on it) -- plus `N_BDY_LAYERS` rings of relaxation
    cells beyond it, and write the result to `out_path`.

    `point_lat_deg`/`point_lon_deg` name a point known to be inside the
    boundary; left as None, the polygon's spherical centroid is used, which
    is only reliable for a convex (or nearly so) boundary -- pass it
    explicitly for a concave or oddly-shaped region.

    Works on a global or an already-regional input mesh, at any
    `sphere_radius` -- this never touches position, only connectivity.
    Writes a whole new file; `mesh_path` is never touched. Call
    `write_graph_info` afterward for the accompanying `graph.info`
    partition file.
    """
    src = resolve_path(mesh_path)
    if not src.exists():
        raise FileNotFoundError(f"No such mesh file: {src}")

    boundary_lat_deg = np.atleast_1d(np.asarray(boundary_lat_deg, dtype=np.float64))
    boundary_lon_deg = np.atleast_1d(np.asarray(boundary_lon_deg, dtype=np.float64))
    if boundary_lat_deg.size < 3:
        raise ValueError(
            f"gmpas prep create-region needs at least 3 boundary points to "
            f"define a region, got {boundary_lat_deg.size}."
        )

    with xr.open_dataset(src, decode_timedelta=False, engine="netcdf4") as ds:
        missing = [v for v in REQUIRED_VARS if v not in ds.variables]
        if missing:
            raise KeyError(f"{src.name} cannot be cropped: missing {missing}.")
        ds = ds.load()

    cell_xyz = np.stack([ds["xCell"].values, ds["yCell"].values,
                         ds["zCell"].values], axis=-1)
    cell_xyz = cell_xyz / np.linalg.norm(cell_xyz, axis=-1, keepdims=True)

    if point_lat_deg is None or point_lon_deg is None:
        point_lon_deg, point_lat_deg = _spherical_centroid(
            boundary_lon_deg, boundary_lat_deg)

    cells_on_cell = ds["cellsOnCell"].values.astype(np.int64)
    n_edges_on_cell = ds["nEdgesOnCell"].values.astype(np.int64)

    boundary_cells = _nearest_cells(cell_xyz, boundary_lon_deg, boundary_lat_deg)
    on_boundary = _walk_boundary(cells_on_cell, n_edges_on_cell, cell_xyz, boundary_cells)

    start_cell = int(_nearest_cells(cell_xyz, np.array([point_lon_deg]),
                                    np.array([point_lat_deg]))[0])
    in_region = _flood_fill(cells_on_cell, n_edges_on_cell, start_cell, on_boundary)
    if in_region.all():
        raise ValueError(
            "gmpas prep create-region: the flood fill reached the whole "
            "mesh -- either --point is outside the --polygon boundary "
            "rather than inside it, or the polygon's vertices are too "
            "close together to form a real wall on this mesh (each "
            "consecutive pair should snap to a different cell)."
        )

    cell_zone = _relaxation_zones(cells_on_cell, n_edges_on_cell, in_region, N_BDY_LAYERS)
    edge_zone = _neighbour_zone(cell_zone, ds["cellsOnEdge"].values.astype(np.int64))
    vertex_zone = _neighbour_zone(cell_zone, ds["cellsOnVertex"].values.astype(np.int64))

    kept = {
        "nCells": np.flatnonzero(cell_zone >= 0),
        "nEdges": np.flatnonzero(edge_zone <= N_BDY_LAYERS),
        "nVertices": np.flatnonzero(vertex_zone <= N_BDY_LAYERS),
    }

    out = _subset_and_remap(ds, kept)
    out["bdyMaskCell"] = ("nCells", cell_zone[kept["nCells"]].astype(np.int32))
    out["bdyMaskEdge"] = ("nEdges", edge_zone[kept["nEdges"]].astype(np.int32))
    out["bdyMaskVertex"] = ("nVertices", vertex_zone[kept["nVertices"]].astype(np.int32))

    out_path = Path(out_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(out_path)
    return out_path


def write_graph_info(regional_mesh_path: str | Path, out_path: str | Path) -> Path:
    """Write a METIS `graph.info` adjacency file for a mesh produced by
    `create_region` -- there is no `mkgrid` run over a cropped subset to
    produce one, unlike a freshly generated global mesh.

    Format: a header line `nCells nInteriorEdges`, then one line per cell
    listing its neighbour cells' (1-based) ids -- the standard METIS graph
    format `gpmetis` reads directly.
    """
    with xr.open_dataset(resolve_path(regional_mesh_path), decode_timedelta=False,
                         engine="netcdf4") as ds:
        n_cells = ds.sizes["nCells"]
        n_edges_on_cell = ds["nEdgesOnCell"].values
        cells_on_cell = ds["cellsOnCell"].values
        cells_on_edge = ds["cellsOnEdge"].values

    n_interior_edges = int(((cells_on_edge[:, 0] > 0) & (cells_on_edge[:, 1] > 0)).sum())

    out_path = Path(out_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"{n_cells} {n_interior_edges}\n")
        for c in range(n_cells):
            row = cells_on_cell[c, : n_edges_on_cell[c]]
            f.write(" ".join(str(int(n)) for n in row if n > 0) + "\n")
    return out_path


# --------------------------------------------------------------- comparison

def plot_region(regional_mesh_path: str | Path, out_png: str | Path) -> Path:
    """Render the cropped mesh coloured by `bdyMaskCell` -- 0 for the
    untouched interior, up through `N_BDY_LAYERS` at the outer edge -- a
    quick visual check that `create_region` kept the right cells and that
    the relaxation rings form actual rings, not a smear. Returns the PNG
    path.
    """
    from ..mesh import MpasMesh
    from ..plot import cell_field
    from ..style import Style, save_figure

    path = resolve_path(regional_mesh_path)
    with xr.open_dataset(path, decode_timedelta=False, engine="netcdf4") as ds:
        zone = ds["bdyMaskCell"].values

    mesh = MpasMesh.load(path)
    fig, _ = cell_field(
        mesh, zone, style=Style.preset("mesh"), vmin=0, vmax=N_BDY_LAYERS,
        label=f"boundary zone (0 = interior, {N_BDY_LAYERS} = edge)",
        title=f"{mesh.n_cells:,} cells, {N_BDY_LAYERS} relaxation rings",
    )
    return save_figure(fig, out_png)
