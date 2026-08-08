"""Move a mesh's refined region to a new tangent point, without resizing it.

`scale_mesh` rescales resolution around a tangent point via a stereographic
projection, which is only close to uniform *near* that point -- far from it
(see the module docstring in `scale.py`), the same operation can end up
making cells coarser instead of finer. That makes it a regional-mesh tool:
correct once a mesh's whole extent is already close to the tangent point,
wrong applied to a global mesh where most of it isn't.

Repositioning *where* a refined patch sits, without also touching its
resolution, doesn't need a projection at all -- a rigid rotation of the
sphere does it exactly. A rotation is an isometry: it preserves every
distance, area and angle between points on the sphere, everywhere, with no
far-field distortion and no antipodal singularity. So `dcEdge`, `dvEdge`,
`areaCell`, `areaTriangle`, `kiteAreasOnVertex`, `weightsOnEdge` and
`nominalMinDc` all pass through completely unchanged here -- only the
coordinates themselves (lon/lat/xyz for Cell, Vertex, Edge) and `angleEdge`
(the edge-normal bearing relative to local east, which is measured against
the fixed lat/lon grid and so does change when a point moves to new
coordinates) need recomputing.

Also unlike `scale_mesh`, this has no unit-sphere precondition: a rotation
matrix is scale-independent, so it works on a mesh at any `sphere_radius`.

Works on a global mesh directly -- rotating first, then cropping to a
regional subset around the relocated patch, is the natural order (see
docs/preprocessing.md).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import xarray as xr

from ..paths import resolve_path
from .scale import lonlat_to_xyz, spherical_angle, xyz_to_lonlat

#: variables relocate_mesh reads and recomputes, beyond mesh identity
REQUIRED_VARS = (
    "latCell", "lonCell", "xCell", "yCell", "zCell", "areaCell",
    "latVertex", "lonVertex", "xVertex", "yVertex", "zVertex",
    "latEdge", "lonEdge", "xEdge", "yEdge", "zEdge",
    "verticesOnEdge", "angleEdge",
)


def _rotation_matrix(from_xyz: np.ndarray, to_xyz: np.ndarray) -> np.ndarray:
    """The minimal rotation mapping `from_xyz` exactly onto `to_xyz`.

    Rodrigues' formula around the axis perpendicular to both, by the angle
    between them -- the shortest-path rotation, with no free twist
    parameter to choose.
    """
    from_xyz = from_xyz / np.linalg.norm(from_xyz)
    to_xyz = to_xyz / np.linalg.norm(to_xyz)
    axis = np.cross(from_xyz, to_xyz)
    sin_angle = np.linalg.norm(axis)
    cos_angle = np.dot(from_xyz, to_xyz)

    if sin_angle < 1e-12:
        if cos_angle > 0:
            return np.eye(3)                    # already at the target
        # exactly antipodal: any axis perpendicular to from_xyz will do
        axis = np.cross(from_xyz, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-12:
            axis = np.cross(from_xyz, [0.0, 1.0, 0.0])
        axis = axis / np.linalg.norm(axis)
        sin_angle, cos_angle = 0.0, -1.0
    else:
        axis = axis / sin_angle

    k = np.array([[0.0, -axis[2], axis[1]],
                 [axis[2], 0.0, -axis[0]],
                 [-axis[1], axis[0], 0.0]])
    return np.eye(3) + k * sin_angle + (k @ k) * (1.0 - cos_angle)


def relocate_mesh(mesh_path: str | Path, out_path: str | Path,
                  tan_lat_deg: float, tan_lon_deg: float,
                  from_lat_deg: float | None = None,
                  from_lon_deg: float | None = None) -> Path:
    """Rotate a mesh so its refined region sits at (tan_lat_deg, tan_lon_deg).

    `from_lat_deg`/`from_lon_deg` name the region's current center; left as
    None, it defaults to the mesh's own finest cell (`min(areaCell)`) -- the
    conventional single point of maximum refinement in an MPAS
    variable-resolution mesh. Pass them explicitly for a mesh with more than
    one refined patch, where auto-detection would only find one of them.

    Works on a global or a regional mesh, at any `sphere_radius`. Writes a
    whole new file; `mesh_path` is never touched.
    """
    src = resolve_path(mesh_path)
    if not src.exists():
        raise FileNotFoundError(f"No such mesh file: {src}")

    with xr.open_dataset(src, decode_timedelta=False, engine="netcdf4") as ds:
        missing = [v for v in REQUIRED_VARS if v not in ds.variables]
        if missing:
            raise KeyError(
                f"{src.name} cannot be relocated: missing {missing}."
            )
        ds = ds.load()

    if from_lat_deg is None or from_lon_deg is None:
        finest = int(np.argmin(ds["areaCell"].values))
        from_lat_deg = math.degrees(float(ds["latCell"].values[finest]))
        from_lon_deg = math.degrees(float(ds["lonCell"].values[finest]))

    from_xyz = lonlat_to_xyz(math.radians(from_lon_deg), math.radians(from_lat_deg))
    to_xyz = lonlat_to_xyz(math.radians(tan_lon_deg), math.radians(tan_lat_deg))
    rotation = _rotation_matrix(from_xyz, to_xyz)

    updates = {}
    for loc in ("Cell", "Vertex", "Edge"):
        xyz = np.stack([ds[f"x{loc}"].values, ds[f"y{loc}"].values,
                        ds[f"z{loc}"].values], axis=-1)
        rotated = xyz @ rotation.T
        lon, lat = xyz_to_lonlat(rotated)
        updates[f"x{loc}"] = rotated[..., 0]
        updates[f"y{loc}"] = rotated[..., 1]
        updates[f"z{loc}"] = rotated[..., 2]
        updates[f"lon{loc}"] = lon
        updates[f"lat{loc}"] = lat

    # angleEdge is the edge-normal bearing relative to local east on the
    # fixed lat/lon grid -- unlike area/distance, that's not intrinsic to
    # the mesh's shape, so it does change when a point moves to new
    # coordinates. Recomputed exactly as scale_mesh's angleEdge step does,
    # just on the rotated positions instead of the rescaled ones.
    voe = ds["verticesOnEdge"].values.astype(np.int64) - 1
    vtx_xyz_new = np.stack([updates["xVertex"], updates["yVertex"],
                            updates["zVertex"]], axis=-1)
    a_pts = vtx_xyz_new[voe[:, 0]]
    tangent = np.stack([
        np.cos(updates["lonEdge"]) * np.sin(updates["latEdge"]),
        np.sin(updates["lonEdge"]) * np.sin(updates["latEdge"]),
        -np.cos(updates["latEdge"]),
    ], axis=-1)
    b_pts = a_pts - tangent
    b_pts = b_pts / np.linalg.norm(b_pts, axis=-1, keepdims=True)
    c_pts = vtx_xyz_new[voe[:, 1]]
    updates["angleEdge"] = spherical_angle(a_pts, b_pts, c_pts)

    for name, values in updates.items():
        da = ds[name]
        ds[name] = (da.dims, np.asarray(values).astype(da.dtype), da.attrs)

    out = Path(out_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out)
    return out
