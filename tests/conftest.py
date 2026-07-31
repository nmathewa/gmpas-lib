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
               sphere_radius=EARTH_RADIUS, areas=None, coord_dtype=np.float64):
    """Write a synthetic MPAS mesh file.

    centres: sequence of (lon_deg, lat_deg), lon in -180..180 as a human would
             write it; stored in the file as MPAS does, radians on [0, 2pi).
    n_verts: per-cell vertex count (defaults to 6 for every cell). A value
             below 6 exercises the ragged `verticesOnCell` fill.
    sphere_radius: pass 1.0 to mimic a mesh straight out of JIGSAW, whose
             areas are non-dimensional.
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

    ds = xr.Dataset(
        {
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
        },
        attrs={"sphere_radius": float(sphere_radius)},
    )
    assert ds.sizes["nVertices"] == n_vertices
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
