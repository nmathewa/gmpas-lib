"""MPAS native mesh geometry: build once, cache, reuse.

An MPAS mesh is static for an entire simulation, so every plot of every
variable at every timestep shares the same polygons. The slow part of native
plotting is rebuilding that geometry -- this module builds it vectorized and
caches it to .npz, so the second and later loads are effectively free.

Mesh conventions handled here:
  * lat/lon fields are stored in RADIANS, longitude on [0, 2pi)
  * connectivity arrays (verticesOnCell, verticesOnEdge, ...) are 1-BASED,
    with 0 used as the "no neighbour" fill
  * verticesOnCell is ragged: cell i uses only its first nEdgesOnCell[i] entries
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xarray as xr

from .paths import cache_dir, resolve_path

R2D = 180.0 / np.pi

#: MPAS's default sphere radius, used to redimensionalise unit-sphere meshes
EARTH_RADIUS = 6_371_229.0

#: variables that identify a file as carrying MPAS mesh information
MESH_VARS = ("latCell", "lonCell", "verticesOnCell", "nEdgesOnCell")

#: bump when the cached array layout or units change, to invalidate old caches
CACHE_VERSION = "v5"

#: cells (or edges) per chunk when building geometry straight to disk
BUILD_CHUNK = 250_000

#: arrays held in the cache directory, one .npy each
CACHED_ARRAYS = ("lon_cell", "lat_cell", "cell_verts", "cell_wrapped",
                 "edge_segs", "edge_wrapped", "lon_edge", "lat_edge",
                 "angle_edge", "area_cell", "xyz_cell")

#: fraction of its sphere a mesh must cover before it counts as global
GLOBAL_COVERAGE = 0.9

#: headroom demanded on top of the computed cache size, so the build does not
#: start against a filesystem it will fill exactly
SPACE_MARGIN = 64 * 1024 * 1024


class MeshCacheError(OSError):
    """The mesh cache could not be written -- almost always out of room."""


def has_mesh(ds: xr.Dataset) -> bool:
    """Whether a dataset carries enough information to build mesh geometry."""
    return all(v in ds.variables for v in MESH_VARS)


# ------------------------------------------------------------------- caching


def _header_dims(path: Path) -> tuple[int, int]:
    """(nCells, nEdges) straight from the netCDF header -- no data is read."""
    try:
        import netCDF4

        with netCDF4.Dataset(path) as nc:
            return (len(nc.dimensions.get("nCells", ())),
                    len(nc.dimensions.get("nEdges", ())))
    except Exception:
        return (-1, -1)


def _signature(path: Path) -> str:
    """Cheap identity for a mesh file: path, size, mtime, and element counts.

    Size and mtime alone are not enough. netCDF4's chunked, padded layout means
    two meshes with different cell counts can occupy exactly the same number of
    bytes, and `st_mtime` truncated to whole seconds cannot separate a file
    from the one that replaced it in the same second. Both together let a
    regenerated mesh silently reuse the previous mesh's cached geometry.

    nCells/nEdges come from the header, which netCDF reads without touching any
    data, so this stays cheap enough to run on every load.
    """
    st = path.stat()
    n_cells, n_edges = _header_dims(path)
    raw = (f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}"
           f"|{n_cells}|{n_edges}")
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def cache_path(path: Path) -> Path:
    """Directory holding this mesh file's cached geometry.

    A directory of `.npy`, not a single `.npz`: an `.npz` is a zip, and numpy
    cannot memory-map inside one, so opening it means reading every array into
    RAM. Separate `.npy` files can be mapped, which is what keeps a large mesh
    usable -- pages arrive only for the arrays actually touched.
    """
    return cache_dir() / f"{path.stem}.{_signature(path)}.{CACHE_VERSION}"


# --------------------------------------------------------------------- mesh


@dataclass
class MpasMesh:
    """Cached native-mesh geometry, ready to hand to matplotlib."""

    path: Path
    lon_cell: np.ndarray          # (nCells,) degrees, -180..180
    lat_cell: np.ndarray          # (nCells,)
    cell_verts: np.ndarray        # (nCells, maxEdges, 2) closed polygons, degrees
    cell_wrapped: np.ndarray      # (nCells,) bool: straddles the antimeridian
    edge_segs: np.ndarray         # (nEdges, 2, 2) vertex-to-vertex line segments
    edge_wrapped: np.ndarray      # (nEdges,) bool
    lon_edge: np.ndarray          # (nEdges,) degrees
    lat_edge: np.ndarray          # (nEdges,)
    angle_edge: np.ndarray        # (nEdges,) radians, edge normal vs local east
    area_cell: np.ndarray         # (nCells,) m^2
    xyz_cell: np.ndarray          # (nCells, 3) unit-sphere coords, for KD-tree
    sphere_radius: float          # metres: the sphere this mesh is actually on

    #: derived scalars, persisted alongside the arrays. Recomputing `extent`
    #: means reducing over cell_verts, which is the largest array there is --
    #: on a mapped cache that would page in gigabytes to learn four numbers.
    _meta: dict = field(default_factory=dict)

    _tree: object = None

    # -- properties ------------------------------------------------------

    @property
    def n_cells(self) -> int:
        return self.cell_verts.shape[0]

    @property
    def n_edges(self) -> int:
        return self.edge_segs.shape[0]

    @property
    def coverage(self) -> float:
        """Fraction of its sphere this mesh covers -- ~1.0 for a global mesh."""
        if "coverage" not in self._meta:
            self._meta["coverage"] = float(
                self.area_cell.sum() / (4.0 * np.pi * self.sphere_radius**2)
            )
        return self._meta["coverage"]

    @property
    def is_global(self) -> bool:
        """Whether the mesh actually covers the sphere.

        This asks about coverage, not about wrapping. Straddling the
        antimeridian is not the same thing: a regional Indo-Pacific domain
        crosses the dateline while covering a sixth of the planet, and calling
        it global gives it a whole-sphere extent, so it renders as a speck on
        a world map with most of the frame blank.
        """
        return self.coverage >= GLOBAL_COVERAGE

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """(lon_min, lon_max, lat_min, lat_max) covering the mesh, in degrees.

        For a mesh that crosses the antimeridian, `lon_max` runs past +180:
        such a domain is only contiguous in the unwrapped [0, 360) frame, and
        a plain min/max over wrapped longitudes would return the whole world.
        Whichever of the two frames gives the tighter span wins.
        """
        if "extent" not in self._meta:
            self._meta["extent"] = self._compute_extent()
        return tuple(self._meta["extent"])

    def _compute_extent(self) -> tuple[float, float, float, float]:
        if self.is_global:
            return (-180.0, 180.0, -90.0, 90.0)

        verts = np.asarray(self.cell_verts)      # reduces over the whole array
        lon = verts[..., 0]
        lat_min = float(verts[..., 1].min())
        lat_max = float(verts[..., 1].max())

        wrapped_lo, wrapped_hi = float(lon.min()), float(lon.max())
        lon360 = lon % 360.0
        unwrapped_lo, unwrapped_hi = float(lon360.min()), float(lon360.max())

        if (unwrapped_hi - unwrapped_lo) < (wrapped_hi - wrapped_lo):
            return (unwrapped_lo, unwrapped_hi, lat_min, lat_max)
        return (wrapped_lo, wrapped_hi, lat_min, lat_max)

    @property
    def cell_width_km(self) -> np.ndarray:
        """Cell width in km, from cell area assuming a regular hexagon.

        A hexagon of area A has centre-to-face width 2*sqrt(A / (2*sqrt(3))).
        This is the number that says where a variable-resolution mesh refines.
        """
        return 2.0 * np.sqrt(self.area_cell / (2.0 * np.sqrt(3.0))) / 1000.0

    def tree(self):
        """KD-tree over cell centres. For a Voronoi mesh, the nearest cell
        centre to a point *is* the cell containing it -- exact, no clipping."""
        if self._tree is None:
            from scipy.spatial import cKDTree

            self._tree = cKDTree(self.xyz_cell)
        return self._tree

    def cell_of(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Index of the cell containing each (lon, lat) point, in degrees."""
        lon = np.asarray(lon, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        lon_r, lat_r = np.radians(lon), np.radians(lat)
        pts = np.stack(
            [
                np.cos(lat_r) * np.cos(lon_r),
                np.cos(lat_r) * np.sin(lon_r),
                np.sin(lat_r),
            ],
            axis=-1,
        )
        return self.tree().query(pts.reshape(-1, 3))[1].reshape(lon.shape)

    # -- construction ----------------------------------------------------

    @classmethod
    def load(cls, mesh_path: str | Path, use_cache: bool = True) -> "MpasMesh":
        path = resolve_path(mesh_path)
        if not path.exists():
            raise FileNotFoundError(f"No such mesh file: {path}")

        if not use_cache:
            return cls._build(path)

        cache = cache_path(path)
        if not (cache / "meta.json").exists():
            _build_to_dir(path, cache)
        return cls._mapped(path, cache)

    @classmethod
    def _mapped(cls, path: Path, cache: Path) -> "MpasMesh":
        """Open a cached mesh without reading it.

        Every array is memory-mapped, so this costs one open per file and the
        pages arrive only for arrays something actually touches. Rasterizing a
        50M-cell mesh reads `xyz_cell` and `area_cell` and leaves the other
        several gigabytes on disk.
        """
        meta = json.loads((cache / "meta.json").read_text())
        arrays = {name: np.load(cache / f"{name}.npy", mmap_mode="r")
                  for name in CACHED_ARRAYS}
        return cls(path=path, sphere_radius=float(meta["sphere_radius"]),
                   _meta=meta, **arrays)

    @classmethod
    def _build(cls, path: Path) -> "MpasMesh":
        ds = xr.open_dataset(path, decode_timedelta=False, engine="netcdf4")
        if not has_mesh(ds):
            missing = [v for v in MESH_VARS if v not in ds.variables]
            ds.close()
            raise KeyError(
                f"{path.name} carries no MPAS mesh information (missing {missing}). "
                f"Pass the mesh/init/static file instead."
            )

        lon_v = _wrap180(ds.lonVertex.values * R2D)
        lat_v = ds.latVertex.values * R2D
        lon_c = _wrap180(ds.lonCell.values * R2D)
        lat_c = ds.latCell.values * R2D

        cell_verts, cell_wrapped = _cell_polygons(
            ds.verticesOnCell.values, ds.nEdgesOnCell.values, lon_v, lat_v
        )
        edge_segs, edge_wrapped = _edge_segments(ds.verticesOnEdge.values, lon_v, lat_v)

        # unit-sphere Cartesian for the KD-tree; xCell/yCell/zCell are on the
        # sphere radius, so normalise rather than trusting the units
        xyz = np.stack([ds.xCell.values, ds.yCell.values, ds.zCell.values], axis=-1)
        xyz = xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)

        # Meshes fresh out of JIGSAW/MPAS-Tools carry sphere_radius=1, so areas
        # and distances are non-dimensional; only after init_atmosphere are they
        # in metres. Redimensionalise so area_cell is always m^2.
        #
        # ASSUMPTION: a non-dimensional mesh is assumed to be Earth-sized, which
        # is MPAS's own default. A unit-sphere mesh actually intended for a
        # reduced-radius planet cannot be told apart from an Earth one by
        # anything in the file, and will be scaled to Earth here.
        #
        # A mesh that does declare its radius keeps it -- reduced-radius
        # ("small planet", X-factor) configurations are a real MPAS use case,
        # and everything downstream that converts between metres and angles
        # must divide by *this* radius rather than Earth's.
        radius = float(ds.attrs.get("sphere_radius", EARTH_RADIUS) or EARTH_RADIUS)
        area = ds.areaCell.values
        if radius < 1.001:
            radius = EARTH_RADIUS
            area = area * EARTH_RADIUS**2

        mesh = cls(
            path=path,
            lon_cell=lon_c, lat_cell=lat_c,
            cell_verts=cell_verts, cell_wrapped=cell_wrapped,
            edge_segs=edge_segs, edge_wrapped=edge_wrapped,
            lon_edge=_wrap180(ds.lonEdge.values * R2D),
            lat_edge=ds.latEdge.values * R2D,
            angle_edge=ds.angleEdge.values,
            area_cell=area,
            xyz_cell=xyz,
            sphere_radius=radius,
        )
        ds.close()
        return mesh


# ------------------------------------------------------------ chunked build


class _Bounds:
    """Running min/max of the two longitude frames, plus latitude.

    Accumulated per chunk so the extent is known without ever holding the
    whole vertex array, and without a second reduction pass over it later.
    """

    def __init__(self):
        self.lat = [np.inf, -np.inf]
        self.wrapped = [np.inf, -np.inf]      # lon on [-180, 180)
        self.unwrapped = [np.inf, -np.inf]    # lon on [0, 360)

    def update(self, block: np.ndarray) -> None:
        lon, lat = block[..., 0], block[..., 1]
        self.lat[0] = min(self.lat[0], float(lat.min()))
        self.lat[1] = max(self.lat[1], float(lat.max()))
        self.wrapped[0] = min(self.wrapped[0], float(lon.min()))
        self.wrapped[1] = max(self.wrapped[1], float(lon.max()))
        lon360 = lon % 360.0
        self.unwrapped[0] = min(self.unwrapped[0], float(lon360.min()))
        self.unwrapped[1] = max(self.unwrapped[1], float(lon360.max()))

    def extent(self) -> tuple[float, float, float, float]:
        if (self.unwrapped[1] - self.unwrapped[0]) < (self.wrapped[1] - self.wrapped[0]):
            lo, hi = self.unwrapped
        else:
            lo, hi = self.wrapped
        return (lo, hi, self.lat[0], self.lat[1])


class _NpyWriter:
    """Stream one `.npy` out chunk by chunk, through ordinary writes.

    Deliberately not `open_memmap(mode="w+")`. A writable mapping creates the
    file at its full size but *sparse*, so nothing is reserved: the assignment
    into the mapping always appears to succeed and the allocation failure lands
    later, in page writeback, where there is no caller left to return it to.

    Measured on a 120 MB volume, writing a 356 MB array through such a mapping:
    every write returned cleanly, `flush()` returned cleanly, and after
    unmounting and remounting two thirds of the array read back as zeros. On
    Linux the same overcommit arrives instead as SIGBUS, which kills the
    process outright with no traceback -- the symptom in issue #19.

    An ordinary buffered write has neither failure mode: it returns ENOSPC (or
    EDQUOT, over a quota) as an `OSError` the build can clean up after and the
    user can read. The chunking is unchanged, so peak memory is the same.
    """

    def __init__(self, path: Path, dtype, shape: tuple[int, ...]):
        self.path = Path(path)
        self.dtype = np.dtype(dtype)
        self.shape = tuple(int(n) for n in shape)
        self._rows = 0
        self._fp = open(self.path, "wb")
        np.lib.format.write_array_header_1_0(self._fp, {
            "descr": np.lib.format.dtype_to_descr(self.dtype),
            "fortran_order": False,
            "shape": self.shape,
        })

    def append(self, block: np.ndarray) -> None:
        block = np.ascontiguousarray(block, dtype=self.dtype)
        with _writing(self.path):
            block.tofile(self._fp)
        self._rows += block.shape[0]

    def close(self) -> None:
        if self._rows != self.shape[0]:
            raise MeshCacheError(
                f"{self.path.name}: wrote {self._rows} rows of {self.shape[0]}"
            )
        # fsync so a deferred allocation failure is raised here, as an error
        # about this file, rather than surfacing as a corrupt read much later
        with _writing(self.path):
            self._fp.flush()
            os.fsync(self._fp.fileno())
        self._fp.close()


@contextmanager
def _writing(path: Path):
    """Say which cached array failed, and that running out of room is why.

    numpy reports a short write as `1000000 requested and 0 written`, which
    names neither the file nor the cause. Out of room is overwhelmingly the
    reason a cache write fails, and the user can act on that sentence.
    """
    try:
        yield
    except OSError as exc:
        raise MeshCacheError(
            f"failed writing {path.name} to the mesh cache: {exc}\n"
            f"  Most likely {path.parent.parent} is full or over quota. "
            f"Free some space, or set GMPAS_CACHE_DIR to somewhere larger."
        ) from exc


def _save(path: Path, arr: np.ndarray) -> None:
    """`np.save` that has actually reached the disk when it returns."""
    with _writing(path), open(path, "wb") as fp:
        np.lib.format.write_array(fp, np.ascontiguousarray(arr))
        fp.flush()
        os.fsync(fp.fileno())


def _cache_bytes(nc) -> int:
    """Size of the finished cache, from the netCDF header alone -- no reads.

    Every cached array is a fixed multiple of nCells or nEdges, so this is
    exact rather than an estimate, and it costs nothing to compute before
    deciding whether the build can succeed at all.
    """
    n_cells = len(nc.dimensions["nCells"])
    n_edges = len(nc.dimensions["nEdges"])
    max_edges = len(nc.dimensions["maxEdges"])
    v = nc.variables

    def width(name: str) -> int:
        return v[name].dtype.itemsize

    return (
        n_cells * max_edges * 2 * width("lonVertex")     # cell_verts
        + n_edges * 2 * 2 * width("lonVertex")           # edge_segs
        + n_cells + n_edges                              # the two bool arrays
        + n_cells * 3 * width("xCell")                   # xyz_cell
        + n_cells * width("areaCell")                    # area_cell
        + n_cells * 2 * width("lonCell")                 # lon_cell, lat_cell
        + n_edges * 2 * width("lonEdge")                 # lon_edge, lat_edge
        + n_edges * width("angleEdge")                   # angle_edge
    )


def _size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def _check_space(where: Path, need: int, mesh: Path) -> None:
    """Refuse to start a build that cannot fit, and say so in bytes."""
    free = shutil.disk_usage(where).free
    if free >= need + SPACE_MARGIN:
        return

    raise MeshCacheError(
        f"not enough room to cache {mesh.name}: its geometry needs "
        f"{_size(need)} and {where} has {_size(free)} free.\n"
        f"  Free some space, or send the cache somewhere larger with "
        f"GMPAS_CACHE_DIR=/path/with/room\n"
        f"  (this reads free space, not your quota -- on a shared filesystem "
        f"a quota can bind well before the disk does)."
    )


def _build_to_dir(path: Path, cache: Path, chunk: int = BUILD_CHUNK) -> None:
    """Build geometry a chunk at a time, straight into mapped .npy files.

    The ragged fill and the antimeridian unwrap are both per-row, so the work
    decomposes cleanly over cells and no chunk needs to see another. Building
    the whole thing in memory first costs several times the size of the result
    in transient copies -- the int64 connectivity, the `np.where` fill, and the
    output all alive at once -- which is what makes a very large mesh
    impossible to cache rather than merely slow.

    Writes to a temporary directory and renames, so an interrupted build never
    leaves a half-written cache that looks complete -- and removes that
    directory if it fails, so a build that ran out of room does not leave its
    partial output behind to make the next attempt worse.
    """
    import netCDF4

    tmp = cache.with_name(cache.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    nc = netCDF4.Dataset(path)
    nc.set_auto_mask(False)
    try:
        missing = [v for v in MESH_VARS if v not in nc.variables]
        if missing:
            raise KeyError(
                f"{path.name} carries no MPAS mesh information (missing "
                f"{missing}). Pass the mesh/init/static file instead."
            )

        _check_space(cache.parent, _cache_bytes(nc), path)

        lon_v_var = nc.variables["lonVertex"]
        lat_v_var = nc.variables["latVertex"]
        dt = lon_v_var.dtype
        n_vertices = lon_v_var.shape[0]

        n_cells = len(nc.dimensions["nCells"])
        n_edges = len(nc.dimensions["nEdges"])
        max_edges = len(nc.dimensions["maxEdges"])

        # lon_v/lat_v must stay fully resident for the fancy-indexing below
        # (lon_v[voc], lat_v[voe] pick arbitrary, non-sequential vertices), so
        # unlike everything else in this function they can't become a genuine
        # streaming read -- but the read itself can still be chunked, which
        # avoids the transient 2-3x multiplier _wrap180's modulo chain costs
        # when applied in one shot to the whole array.
        lon_v = np.empty(n_vertices, dtype=dt)
        lat_v = np.empty(n_vertices, dtype=dt)
        for i in range(0, n_vertices, chunk):
            j = min(i + chunk, n_vertices)
            lon_v[i:j] = _wrap180(lon_v_var[i:j] * R2D)
            lat_v[i:j] = lat_v_var[i:j] * R2D

        bounds = _Bounds()

        verts = _NpyWriter(tmp / "cell_verts.npy", dt, (n_cells, max_edges, 2))
        cw = _NpyWriter(tmp / "cell_wrapped.npy", bool, (n_cells,))
        voc_var, ne_var = nc.variables["verticesOnCell"], nc.variables["nEdgesOnCell"]
        for i in range(0, n_cells, chunk):
            j = min(i + chunk, n_cells)
            voc = voc_var[i:j].astype(np.int64) - 1
            nedges = ne_var[i:j].astype(np.int64)
            valid = np.arange(max_edges)[None, :] < nedges[:, None]
            last = voc[np.arange(j - i), nedges - 1]
            voc = np.where(valid, voc, last[:, None])

            block = np.empty((j - i, max_edges, 2), dtype=dt)
            block[..., 0] = lon_v[voc]
            block[..., 1] = lat_v[voc]
            block, wrapped = _unwrap_polygons(block)

            verts.append(block)
            cw.append(wrapped)
            bounds.update(block)
        verts.close(); cw.close()

        segs = _NpyWriter(tmp / "edge_segs.npy", dt, (n_edges, 2, 2))
        ew = _NpyWriter(tmp / "edge_wrapped.npy", bool, (n_edges,))
        voe_var = nc.variables["verticesOnEdge"]
        for i in range(0, n_edges, chunk):
            j = min(i + chunk, n_edges)
            voe = voe_var[i:j].astype(np.int64) - 1
            block = np.empty((j - i, 2, 2), dtype=dt)
            block[..., 0] = lon_v[voe]
            block[..., 1] = lat_v[voe]
            block, wrapped = _unwrap_polygons(block)
            segs.append(block)
            ew.append(wrapped)
        segs.close(); ew.close()

        # nothing past this point indexes by vertex, so the full nVertices
        # arrays no longer need to stay resident alongside whatever comes next
        del lon_v, lat_v

        radius = float(getattr(nc, "sphere_radius", EARTH_RADIUS) or EARTH_RADIUS)
        rescale_area = radius < 1.001
        if rescale_area:
            radius = EARTH_RADIUS

        # xyz's normalize is a per-row reduction (each cell's own 3
        # components), so -- like the ragged fill and antimeridian unwrap
        # above -- it decomposes cleanly over cells with no chunk needing to
        # see another.
        xc_var, yc_var, zc_var = (nc.variables["xCell"], nc.variables["yCell"],
                                  nc.variables["zCell"])
        area_var = nc.variables["areaCell"]
        lonc_var, latc_var = nc.variables["lonCell"], nc.variables["latCell"]

        xyz_w = _NpyWriter(tmp / "xyz_cell.npy", xc_var.dtype, (n_cells, 3))
        area_w = _NpyWriter(tmp / "area_cell.npy", area_var.dtype, (n_cells,))
        lonc_w = _NpyWriter(tmp / "lon_cell.npy", lonc_var.dtype, (n_cells,))
        latc_w = _NpyWriter(tmp / "lat_cell.npy", latc_var.dtype, (n_cells,))

        # coverage needs sum(areaCell), so accumulate a running total instead
        # of summing the whole array at the end -- same idea as `_Bounds` for
        # extent. A chunked sum isn't bit-identical to a whole-array .sum()
        # (float addition isn't associative), but this only affects the
        # derived coverage scalar, not any cached array's own values.
        area_sum = 0.0
        for i in range(0, n_cells, chunk):
            j = min(i + chunk, n_cells)

            xyz_block = np.stack([xc_var[i:j], yc_var[i:j], zc_var[i:j]], axis=-1)
            xyz_block = xyz_block / np.linalg.norm(xyz_block, axis=-1, keepdims=True)
            xyz_w.append(xyz_block)

            area_block = area_var[i:j]
            if rescale_area:
                area_block = area_block * EARTH_RADIUS**2
            area_w.append(area_block)
            area_sum += float(area_block.sum())

            lonc_w.append(_wrap180(lonc_var[i:j] * R2D))
            latc_w.append(latc_var[i:j] * R2D)
        xyz_w.close(); area_w.close(); lonc_w.close(); latc_w.close()

        # pure elementwise scale/wrap, no reduction at all -- the simplest
        # case of the three chunked loops in this function
        lone_var, late_var, ang_var = (nc.variables["lonEdge"], nc.variables["latEdge"],
                                       nc.variables["angleEdge"])
        lone_w = _NpyWriter(tmp / "lon_edge.npy", lone_var.dtype, (n_edges,))
        late_w = _NpyWriter(tmp / "lat_edge.npy", late_var.dtype, (n_edges,))
        ang_w = _NpyWriter(tmp / "angle_edge.npy", ang_var.dtype, (n_edges,))
        for i in range(0, n_edges, chunk):
            j = min(i + chunk, n_edges)
            lone_w.append(_wrap180(lone_var[i:j] * R2D))
            late_w.append(late_var[i:j] * R2D)
            ang_w.append(ang_var[i:j])
        lone_w.close(); late_w.close(); ang_w.close()

        coverage = float(area_sum / (4.0 * np.pi * radius**2))
        extent = ((-180.0, 180.0, -90.0, 90.0) if coverage >= GLOBAL_COVERAGE
                  else bounds.extent())
        (tmp / "meta.json").write_text(json.dumps({
            "sphere_radius": radius,
            "coverage": coverage,
            "extent": list(extent),
            "n_cells": n_cells,
            "n_edges": n_edges,
            "cache_version": CACHE_VERSION,
        }, indent=2))
    except BaseException:
        # a build that died because the filesystem was full must not leave its
        # partial output sitting on that filesystem
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    finally:
        nc.close()

    if cache.exists():
        shutil.rmtree(cache)
    tmp.rename(cache)


# ----------------------------------------------------------------- geometry


def _wrap180(lon: np.ndarray) -> np.ndarray:
    """MPAS stores longitude on [0, 360); matplotlib wants [-180, 180)."""
    return (lon + 180.0) % 360.0 - 180.0


def _cell_polygons(voc: np.ndarray, nedges: np.ndarray,
                   lon_v: np.ndarray, lat_v: np.ndarray):
    """Vectorized Voronoi cell polygons -- no Python loop over cells.

    verticesOnCell is ragged, so the unused tail of each row is filled by
    repeating that cell's last real vertex. The repeat is degenerate, which
    matplotlib renders identically to a properly closed shorter polygon.
    """
    voc = voc.astype(np.int64) - 1                    # 1-based -> 0-based
    nedges = nedges.astype(np.int64)
    n_cells, max_edges = voc.shape

    valid = np.arange(max_edges)[None, :] < nedges[:, None]
    last = voc[np.arange(n_cells), nedges - 1]
    voc = np.where(valid, voc, last[:, None])

    # keep the file's own coordinate precision. Many MPAS meshes store lat/lon
    # as float32; forcing float64 here doubles the two largest cached arrays
    # without adding a single significant digit.
    verts = np.empty((n_cells, max_edges, 2), dtype=lon_v.dtype)
    verts[..., 0] = lon_v[voc]
    verts[..., 1] = lat_v[voc]

    return _unwrap_polygons(verts)


def _edge_segments(voe: np.ndarray, lon_v: np.ndarray, lat_v: np.ndarray):
    """Each MPAS edge is the Voronoi face between two cells: a vertex pair."""
    voe = voe.astype(np.int64) - 1
    segs = np.empty((voe.shape[0], 2, 2), dtype=lon_v.dtype)
    segs[..., 0] = lon_v[voe]
    segs[..., 1] = lat_v[voe]
    return _unwrap_polygons(segs)


def _unwrap_polygons(verts: np.ndarray):
    """Put every polygon on a single longitude branch.

    A cell straddling the antimeridian has vertices at both +179 and -179; drawn
    as-is it smears right across the map. Shifting the negative vertices by +360
    makes each such polygon contiguous (living just past +180), and the renderer
    draws a -360 copy so the seam is covered on both sides.
    """
    lon = verts[..., 0]
    wrapped = (lon.max(axis=1) - lon.min(axis=1)) > 180.0
    if wrapped.any():
        w = verts[wrapped, :, 0]
        verts[wrapped, :, 0] = np.where(w < 0, w + 360.0, w)
    return verts, wrapped


# --------------------------------------------------- edge-normal reconstruction


def reconstruct_cell_winds(mesh: MpasMesh,
                           u_edge: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Approximate cell-centre (zonal, meridional) wind from edge-normal velocity.

    MPAS carries the prognostic velocity `u` as the component NORMAL to each
    edge, which cannot be quivered directly. `angleEdge` gives the angle of that
    normal from local east, so each edge contributes u*(cos, sin) to its two
    neighbouring cells; averaging over a cell's edges recovers an approximate
    vector.

    This is a cheap area-agnostic average, adequate for looking at a flow field.
    It is NOT the RBF reconstruction MPAS itself uses -- when the diagnostics
    file provides `uReconstructZonal`/`uReconstructMeridional` (or
    `uzonal_*`/`umeridional_*`), prefer those.

    Reads `edgesOnCell` from the mesh file, which the cached geometry does not
    carry.
    """
    with xr.open_dataset(mesh.path, decode_timedelta=False, engine="netcdf4") as ds:
        eoc = ds.edgesOnCell.values.astype(np.int64) - 1
        nedges = ds.nEdgesOnCell.values.astype(np.int64)

    valid = np.arange(eoc.shape[1])[None, :] < nedges[:, None]
    idx = np.where(valid, eoc, 0)

    ang = mesh.angle_edge[idx]
    u = np.asarray(u_edge)[idx]
    zon = np.where(valid, u * np.cos(ang), 0.0).sum(axis=1) / nedges
    mer = np.where(valid, u * np.sin(ang), 0.0).sum(axis=1) / nedges
    return zon, mer
