"""Reduce the source run to one small, self-contained demo file.

The run this demo comes from is two files and 13.2 MB: a diagnostics file
with no mesh in it, and a 9.1 MB mesh file of which gmpas reads eleven
arrays. Publishing both would put 13 MB into git for good and into every
sdist, to show a page that needs about two.

So this writes a single file holding the mesh arrays `MpasMesh` actually
reads plus the diagnostic fields as float32 -- self-contained, so the demo
needs no sidecar mesh, and openable by `gmpas view` exactly like the
original. Run once; the result is what gets committed.

    python docs/demo/make_data.py SOURCE_DIAG SOURCE_MESH docs/demo/data/demo.nc
"""

from __future__ import annotations

import sys
from pathlib import Path

import netCDF4
import numpy as np

#: What `MpasMesh._build` reads, and nothing else. Everything the mesh file
#: carries beyond this -- edges, triangles, weights, the density function --
#: is unused by the viewer and is most of its 9.1 MB.
MESH_VARS = ("latCell", "lonCell", "xCell", "yCell", "zCell", "areaCell",
             "nEdgesOnCell", "verticesOnCell", "latVertex", "lonVertex",
             # the edge arrays too: the mesh cache holds edge segments and
             # wind angles, and `_build_to_dir` sizes itself from them before
             # it will start, so a file without them cannot be opened at all
             "lonEdge", "latEdge", "angleEdge", "verticesOnEdge")


def build(diag: Path, mesh: Path, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    src_d = netCDF4.Dataset(diag)
    src_m = netCDF4.Dataset(mesh)
    dst = netCDF4.Dataset(out, "w", format="NETCDF4")
    try:
        for name, dim in {**src_m.dimensions, **src_d.dimensions}.items():
            dst.createDimension(name, None if dim.isunlimited() else len(dim))
        for attr in ("on_a_sphere", "sphere_radius", "mesh_spec"):
            if hasattr(src_m, attr):
                dst.setncattr(attr, src_m.getncattr(attr))
        dst.setncattr("comment",
                      "Reduced extract for the gmpas web demo: the mesh arrays "
                      "the viewer reads, plus diagnostics as float32. Not the "
                      "complete model output.")

        for name in MESH_VARS:
            if name not in src_m.variables:
                raise SystemExit(f"{mesh.name} has no {name!r}")
            copy(src_m[name], dst, name)

        kept = 0
        for name, var in src_d.variables.items():
            if name in dst.variables or "nCells" not in var.dimensions:
                continue
            copy(var, dst, name, as_float32=True)
            kept += 1
        # xtime carries the timestamp the page puts on the time slider
        for name in ("xtime", "initial_time"):
            if name in src_d.variables and name not in dst.variables:
                copy(src_d[name], dst, name)
    finally:
        dst.close(); src_d.close(); src_m.close()

    print(f"wrote {out}  ({out.stat().st_size / 1e6:.2f} MB, "
          f"{kept} fields + {len(MESH_VARS)} mesh arrays)")


def copy(var, dst, name: str, as_float32: bool = False) -> None:
    dtype = "f4" if as_float32 and var.dtype.kind == "f" else var.dtype
    new = dst.createVariable(name, dtype, var.dimensions,
                             zlib=True, complevel=4)
    for attr in var.ncattrs():
        if attr not in ("_FillValue", "scale_factor", "add_offset"):
            new.setncattr(attr, var.getncattr(attr))
    new[:] = np.asarray(var[:])


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    build(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
