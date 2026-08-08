"""Synthetic MPAS mesh files, so the tests need no real model output.

The fixture writes a netCDF carrying exactly the variables `MpasMesh._build`
reads, using MPAS's own conventions: radians on [0, 2pi), 1-based connectivity,
and a ragged `verticesOnCell` padded with 0.

It is not a true Voronoi tessellation -- each cell gets its own private ring of
vertices rather than sharing them with its neighbours. Everything under test
here is array indexing, unit handling and branch selection, none of which can
tell the difference; the one place topology would matter (nearest-centre
lookup) depends only on the cell centres, which are real.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

EARTH_RADIUS = 6_371_229.0

#: six edge-normal angles evenly spaced over [0, pi), so that for a uniform
#: eastward flow sum(cos^2) / n is exactly 1/2 and sum(cos*sin) is exactly 0
EDGE_ANGLES = np.arange(6) * np.pi / 6.0


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Never touch the user's real ~/.cache/gmpas during tests."""
    monkeypatch.setenv("GMPAS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("GMPAS_DATA_DIR", raising=False)


def _ring(lon_deg: float, lat_deg: float, radius_deg: float, n: int = 6):
    """A closed ring of n vertices around a centre, in degrees."""
    ang = np.arange(n) * 2.0 * np.pi / n
    # cos(lat) keeps the ring roughly circular on the sphere rather than
    # stretched in longitude near the poles
    lon = lon_deg + radius_deg * np.cos(ang) / max(np.cos(np.radians(lat_deg)), 0.1)
    lat = lat_deg + radius_deg * np.sin(ang)
    return lon, lat


def write_mesh(path, centres, *, n_verts=None, radius_deg=1.0,
               sphere_radius=EARTH_RADIUS, areas=None, coord_dtype=np.float64,
               with_scale_vars=False):
    """Write a synthetic MPAS mesh file.

    centres: sequence of (lon_deg, lat_deg), lon in -180..180 as a human would
             write it; stored in the file as MPAS does, radians on [0, 2pi).
    n_verts: per-cell vertex count (defaults to 6 for every cell). A value
             below 6 exercises the ragged `verticesOnCell` fill.
    sphere_radius: pass 1.0 to mimic a mesh straight out of JIGSAW, whose
             areas are non-dimensional.
    with_scale_vars: also write the connectivity/derived variables
             `scale_mesh` needs beyond what `MpasMesh._build` reads
             (`cellsOnEdge`, `cellsOnVertex`, `edgesOnVertex`, `dcEdge`,
             `dvEdge`, `areaTriangle`, `kiteAreasOnVertex`, `weightsOnEdge`,
             `nominalMinDc`, `x/y/zVertex`, `x/y/zEdge`). Off by default:
             `test_same_size_and_second_still_get_different_cache_entries`
             depends on two small meshes padding to the *same* netCDF file
             size, and this much extra per-cell/edge/vertex payload is
             enough to break that coincidence. `cellsOnEdge`/
             `cellsOnVertex`/`edgesOnVertex` are left at their 0-fill ("no
             neighbour") default -- consistent with cells not sharing
             topology here, and it exercises every boundary/fallback
             branch in `scale_mesh`.
    """
    centres = np.asarray(centres, dtype=np.float64)
    n_cells = len(centres)
    max_edges = 6
    n_verts = np.full(n_cells, max_edges) if n_verts is None else np.asarray(n_verts)

    lon_v, lat_v = [], []
    voc = np.zeros((n_cells, max_edges), dtype=np.int32)      # 1-based, 0 = fill
    for i, (lon_c, lat_c) in enumerate(centres):
        rl, rt = _ring(lon_c, lat_c, radius_deg, max_edges)
        base = i * max_edges
        lon_v.extend(rl)
        lat_v.extend(rt)
        voc[i, : n_verts[i]] = base + np.arange(n_verts[i]) + 1

    lon_v = np.asarray(lon_v)
    lat_v = np.asarray(lat_v)
    n_vertices = lon_v.size

    # one edge per vertex slot, joining consecutive vertices of that cell
    n_edges = n_cells * max_edges
    voe = np.zeros((n_edges, 2), dtype=np.int32)
    eoc = np.zeros((n_cells, max_edges), dtype=np.int32)
    for i in range(n_cells):
        base = i * max_edges
        for k in range(max_edges):
            e = base + k
            voe[e] = (base + k + 1, base + (k + 1) % max_edges + 1)
            eoc[i, k] = e + 1

    lon_e = lon_v[voe[:, 0] - 1]
    lat_e = lat_v[voe[:, 0] - 1]
    angle_edge = np.tile(EDGE_ANGLES, n_cells)

    if areas is None:
        areas = np.full(n_cells, 1.0e10)                       # ~100 km cells
    areas = np.asarray(areas, dtype=np.float64)
    if sphere_radius < 1.001:
        areas = areas / EARTH_RADIUS**2                        # non-dimensionalise

    lat_r = np.radians(centres[:, 1])
    lon_r = np.radians(centres[:, 0])
    xyz = np.stack([np.cos(lat_r) * np.cos(lon_r),
                    np.cos(lat_r) * np.sin(lon_r),
                    np.sin(lat_r)], axis=-1) * sphere_radius

    def rad360(deg):
        # MPAS stores longitude in radians on [0, 2pi)
        return np.radians(np.asarray(deg) % 360.0).astype(coord_dtype)

    def rad(deg):
        return np.radians(np.asarray(deg)).astype(coord_dtype)

    variables = {
        "latCell": ("nCells", rad(centres[:, 1])),
        "lonCell": ("nCells", rad360(centres[:, 0])),
        "xCell": ("nCells", xyz[:, 0].astype(coord_dtype)),
        "yCell": ("nCells", xyz[:, 1].astype(coord_dtype)),
        "zCell": ("nCells", xyz[:, 2].astype(coord_dtype)),
        "areaCell": ("nCells", areas.astype(coord_dtype)),
        "nEdgesOnCell": ("nCells", n_verts.astype(np.int32)),
        "verticesOnCell": (("nCells", "maxEdges"), voc),
        "edgesOnCell": (("nCells", "maxEdges"), eoc),
        "latVertex": ("nVertices", rad(lat_v)),
        "lonVertex": ("nVertices", rad360(lon_v)),
        "verticesOnEdge": (("nEdges", "TWO"), voe),
        "latEdge": ("nEdges", rad(lat_e)),
        "lonEdge": ("nEdges", rad360(lon_e)),
        "angleEdge": ("nEdges", angle_edge.astype(coord_dtype)),
    }

    if with_scale_vars:
        def xyz_of(lon_deg, lat_deg):
            lr, gr = np.radians(lat_deg), np.radians(lon_deg)
            return np.stack([np.cos(lr) * np.cos(gr), np.cos(lr) * np.sin(gr),
                             np.sin(lr)], axis=-1) * sphere_radius

        xyz_v = xyz_of(lon_v, lat_v)
        xyz_e = xyz_of(lon_e, lat_e)

        # Placeholder magnitudes for the derived quantities `scale_mesh`
        # only ever reads through its "/scalefac"-style fallback (every
        # edge/vertex here is a boundary one, since cells don't share
        # topology) -- the exact values don't matter, only that they exist
        # and scale correctly.
        vertex_degree = 3
        max_edges2 = 2 * max_edges
        edge_len = np.radians(radius_deg) * sphere_radius
        dc_edge = np.full(n_edges, edge_len, dtype=coord_dtype)
        dv_edge = np.full(n_edges, 0.5 * edge_len, dtype=coord_dtype)
        area_triangle = (np.repeat(areas, max_edges) / max_edges).astype(coord_dtype)
        kite_areas = np.repeat(area_triangle / vertex_degree, vertex_degree)
        kite_areas = kite_areas.reshape(n_vertices, vertex_degree)

        variables.update({
            "xVertex": ("nVertices", xyz_v[:, 0].astype(coord_dtype)),
            "yVertex": ("nVertices", xyz_v[:, 1].astype(coord_dtype)),
            "zVertex": ("nVertices", xyz_v[:, 2].astype(coord_dtype)),
            "xEdge": ("nEdges", xyz_e[:, 0].astype(coord_dtype)),
            "yEdge": ("nEdges", xyz_e[:, 1].astype(coord_dtype)),
            "zEdge": ("nEdges", xyz_e[:, 2].astype(coord_dtype)),
            "cellsOnEdge": (("nEdges", "TWO"),
                           np.zeros((n_edges, 2), dtype=np.int32)),
            "edgesOnEdge": (("nEdges", "maxEdges2"),
                           np.zeros((n_edges, max_edges2), dtype=np.int32)),
            "cellsOnVertex": (("nVertices", "vertexDegree"),
                             np.zeros((n_vertices, vertex_degree), dtype=np.int32)),
            "edgesOnVertex": (("nVertices", "vertexDegree"),
                             np.zeros((n_vertices, vertex_degree), dtype=np.int32)),
            "dcEdge": ("nEdges", dc_edge),
            "dvEdge": ("nEdges", dv_edge),
            "areaTriangle": ("nVertices", area_triangle),
            "kiteAreasOnVertex": (("nVertices", "vertexDegree"), kite_areas),
            "weightsOnEdge": (("nEdges", "maxEdges2"),
                             np.zeros((n_edges, max_edges2), dtype=coord_dtype)),
            "nominalMinDc": ((), np.float64(edge_len)),
        })

    ds = xr.Dataset(variables, attrs={"sphere_radius": float(sphere_radius)})
    assert ds.sizes["nVertices"] == n_vertices
    ds.to_netcdf(path)
    ds.close()
    return path


def write_grid_mesh(path, nrows, ncols, *, lon0_deg=0.0, lat0_deg=0.0,
                    dlon_deg=2.0, dlat_deg=2.0, sphere_radius=1.0):
    """Write a small structured lon/lat quad-lattice mesh, in MPAS's file
    format, with *genuine* shared connectivity -- unlike `write_mesh`, whose
    cells each own a private, unshared ring of vertices/edges and which has
    no `cellsOnCell` at all.

    `region.py`'s algorithm (flood fill, boundary walk, relaxation layers)
    is pure graph traversal over `cellsOnCell`; a fixture with fake adjacency
    would let a wrong traversal pass. This one is a real (if non-hexagonal)
    mesh graph: `nrows` x `ncols` cells, each cell's 4 corners are vertices
    shared with its diagonal/orthogonal neighbours, each edge shared by
    exactly the 2 cells on either side of it (0 on the side beyond the
    lattice's own edge -- there is no wraparound, so this is a regional
    patch, not a closed global mesh).

    Cell (i, j)'s corners/edges, CCW from the SW corner, in MPAS's own
    "verticesOnCell[k]/edgesOnCell[k] straddle the edge to cellsOnCell[k]"
    convention: south (row i-1), east (col j+1), north (row i+1), west
    (col j-1).

    Includes `indexToCellID`/`indexToEdgeID`/`indexToVertexID` and the
    `on_a_sphere`/`sphere_radius` global attributes -- not needed by
    `region.py` itself, but required by `MeshHandler._load_vars` in the
    external MPAS-Limited-Area comparison harness, so the same file drives
    both.
    """
    n_cells = nrows * ncols
    n_vertices = (nrows + 1) * (ncols + 1)
    n_hrow = (nrows + 1) * ncols          # horizontal edges (constant row)
    n_vcol = nrows * (ncols + 1)          # vertical edges (constant column)
    n_edges = n_hrow + n_vcol
    max_edges = 4
    vertex_degree = 4

    def cell_id(i, j):
        return i * ncols + j if 0 <= i < nrows and 0 <= j < ncols else -1

    def vtx_id(i, j):
        return i * (ncols + 1) + j

    def hrow_id(i, j):
        return i * ncols + j

    def vcol_id(i, j):
        return n_hrow + i * (ncols + 1) + j

    def to_xyz(lon_deg, lat_deg):
        lr, gr = np.radians(lat_deg), np.radians(lon_deg)
        return (np.cos(lr) * np.cos(gr) * sphere_radius,
               np.cos(lr) * np.sin(gr) * sphere_radius,
               np.sin(lr) * sphere_radius)

    # -- cells --------------------------------------------------------
    lon_c = np.zeros(n_cells)
    lat_c = np.zeros(n_cells)
    n_edges_on_cell = np.full(n_cells, max_edges, dtype=np.int32)
    coc = np.zeros((n_cells, max_edges), dtype=np.int32)   # cellsOnCell
    voc = np.zeros((n_cells, max_edges), dtype=np.int32)   # verticesOnCell
    eoc = np.zeros((n_cells, max_edges), dtype=np.int32)   # edgesOnCell

    for i in range(nrows):
        for j in range(ncols):
            c = cell_id(i, j)
            lon_c[c] = lon0_deg + j * dlon_deg
            lat_c[c] = lat0_deg + i * dlat_deg
            corners = [vtx_id(i, j), vtx_id(i, j + 1),
                      vtx_id(i + 1, j + 1), vtx_id(i + 1, j)]
            edges = [hrow_id(i, j), vcol_id(i, j + 1),
                    hrow_id(i + 1, j), vcol_id(i, j)]
            neighbours = [cell_id(i - 1, j), cell_id(i, j + 1),
                         cell_id(i + 1, j), cell_id(i, j - 1)]
            voc[c] = [v + 1 for v in corners]
            eoc[c] = [e + 1 for e in edges]
            coc[c] = [n + 1 if n >= 0 else 0 for n in neighbours]

    # -- vertices -------------------------------------------------------
    lon_v = np.zeros(n_vertices)
    lat_v = np.zeros(n_vertices)
    cov = np.zeros((n_vertices, vertex_degree), dtype=np.int32)  # cellsOnVertex
    eov = np.zeros((n_vertices, vertex_degree), dtype=np.int32)  # edgesOnVertex

    for i in range(nrows + 1):
        for j in range(ncols + 1):
            v = vtx_id(i, j)
            lon_v[v] = lon0_deg + (j - 0.5) * dlon_deg
            lat_v[v] = lat0_deg + (i - 0.5) * dlat_deg
            around_cells = [cell_id(i, j), cell_id(i, j - 1),
                            cell_id(i - 1, j - 1), cell_id(i - 1, j)]
            cov[v, :] = [c + 1 if c >= 0 else 0 for c in around_cells]
            around_edges = []
            around_edges.append(hrow_id(i, j - 1) if j - 1 >= 0 else -1)
            around_edges.append(hrow_id(i, j) if j < ncols else -1)
            around_edges.append(vcol_id(i - 1, j) if i - 1 >= 0 else -1)
            around_edges.append(vcol_id(i, j) if i < nrows else -1)
            eov[v, :] = [e + 1 if e >= 0 else 0 for e in around_edges]

    # -- edges ------------------------------------------------------------
    lon_e = np.zeros(n_edges)
    lat_e = np.zeros(n_edges)
    coe = np.zeros((n_edges, 2), dtype=np.int32)   # cellsOnEdge
    voe = np.zeros((n_edges, 2), dtype=np.int32)   # verticesOnEdge

    for i in range(nrows + 1):
        for j in range(ncols):
            e = hrow_id(i, j)
            lon_e[e] = lon0_deg + j * dlon_deg
            lat_e[e] = lat0_deg + (i - 0.5) * dlat_deg
            south, north = cell_id(i - 1, j), cell_id(i, j)
            coe[e] = [south + 1 if south >= 0 else 0,
                     north + 1 if north >= 0 else 0]
            voe[e] = [vtx_id(i, j) + 1, vtx_id(i, j + 1) + 1]

    for i in range(nrows):
        for j in range(ncols + 1):
            e = vcol_id(i, j)
            lon_e[e] = lon0_deg + (j - 0.5) * dlon_deg
            lat_e[e] = lat0_deg + i * dlat_deg
            west, east = cell_id(i, j - 1), cell_id(i, j)
            coe[e] = [west + 1 if west >= 0 else 0,
                     east + 1 if east >= 0 else 0]
            voe[e] = [vtx_id(i, j) + 1, vtx_id(i + 1, j) + 1]

    # edgesOnEdge -- every edge adjoining either of this edge's two cells,
    # other than itself. Order/exact membership isn't semantically load
    # -bearing anywhere in region.py; it only needs to be a valid, ragged,
    # 1-based-with-0-fill array of edge ids for the generic connectivity
    # -remap path to exercise.
    max_edges2 = 2 * max_edges
    eoe = np.zeros((n_edges, max_edges2), dtype=np.int32)
    for e in range(n_edges):
        neighbours = []
        for c0 in coe[e]:
            if c0 == 0:
                continue
            neighbours.extend(int(x) for x in eoc[c0 - 1] if x != 0 and x != e + 1)
        neighbours = neighbours[:max_edges2]
        eoe[e, :len(neighbours)] = neighbours

    xc, yc, zc = to_xyz(lon_c, lat_c)
    xv, yv, zv = to_xyz(lon_v, lat_v)
    xe, ye, ze = to_xyz(lon_e, lat_e)

    def rad360(deg):
        return np.radians(np.asarray(deg) % 360.0)

    def rad(deg):
        return np.radians(np.asarray(deg))

    ds = xr.Dataset(
        {
            "latCell": ("nCells", rad(lat_c)), "lonCell": ("nCells", rad360(lon_c)),
            "xCell": ("nCells", xc), "yCell": ("nCells", yc), "zCell": ("nCells", zc),
            # a uniform solid-angle-rectangle approximation is plenty for a
            # synthetic quad lattice -- nothing here checks areaCell's exact
            # value, only that it exists and is positive (MpasMesh._build
            # reads it unconditionally for cell_width_km)
            "areaCell": ("nCells", np.full(
                n_cells, np.radians(dlon_deg) * np.radians(dlat_deg) * sphere_radius ** 2)),
            "nEdgesOnCell": ("nCells", n_edges_on_cell),
            "cellsOnCell": (("nCells", "maxEdges"), coc),
            "verticesOnCell": (("nCells", "maxEdges"), voc),
            "edgesOnCell": (("nCells", "maxEdges"), eoc),
            "indexToCellID": ("nCells", np.arange(1, n_cells + 1, dtype=np.int32)),
            "latVertex": ("nVertices", rad(lat_v)),
            "lonVertex": ("nVertices", rad360(lon_v)),
            "xVertex": ("nVertices", xv), "yVertex": ("nVertices", yv),
            "zVertex": ("nVertices", zv),
            "cellsOnVertex": (("nVertices", "vertexDegree"), cov),
            "edgesOnVertex": (("nVertices", "vertexDegree"), eov),
            "indexToVertexID": ("nVertices", np.arange(1, n_vertices + 1, dtype=np.int32)),
            "latEdge": ("nEdges", rad(lat_e)), "lonEdge": ("nEdges", rad360(lon_e)),
            "xEdge": ("nEdges", xe), "yEdge": ("nEdges", ye), "zEdge": ("nEdges", ze),
            # not meaningful for a quad lattice's Hrow/Vcol edges (nothing
            # here checks its value) -- present only because MpasMesh._build
            # reads it unconditionally
            "angleEdge": ("nEdges", np.zeros(n_edges)),
            "cellsOnEdge": (("nEdges", "TWO"), coe),
            "verticesOnEdge": (("nEdges", "TWO"), voe),
            "edgesOnEdge": (("nEdges", "maxEdges2"), eoe),
            "indexToEdgeID": ("nEdges", np.arange(1, n_edges + 1, dtype=np.int32)),
        },
        attrs={"sphere_radius": float(sphere_radius), "on_a_sphere": "YES",
              "is_periodic": "NO"},
    )
    ds.to_netcdf(path)
    ds.close()
    return path


@pytest.fixture
def simple_mesh_file(tmp_path):
    """Four well-separated cells over the Maritime Continent, all hexagons."""
    return write_mesh(
        tmp_path / "simple.mesh.nc",
        [(100.0, 0.0), (110.0, 5.0), (120.0, -5.0), (130.0, 10.0)],
    )


@pytest.fixture
def simple_mesh(simple_mesh_file):
    from gmpas import MpasMesh

    return MpasMesh.load(simple_mesh_file)


def write_diag(path, n_cells, n_edges, *, n_times=3, n_levels=4):
    """An MPAS-style diagnostics file: no mesh information, just fields."""
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {
            "mslp": (("Time", "nCells"),
                     rng.normal(101325.0, 200.0, (n_times, n_cells)),
                     {"units": "Pa", "long_name": "mean sea level pressure"}),
            "theta": (("Time", "nVertLevels", "nCells"),
                      rng.normal(300.0, 5.0, (n_times, n_levels, n_cells)),
                      {"units": "K"}),
            "u": (("Time", "nVertLevels", "nEdges"),
                  rng.normal(0.0, 10.0, (n_times, n_levels, n_edges)),
                  {"units": "m s-1", "long_name": "edge normal velocity"}),
            "vorticity": (("Time", "nVertices"),
                          rng.normal(0.0, 1e-5, (n_times, n_cells * 6))),
        }
    )
    ds.to_netcdf(path)
    ds.close()
    return path
