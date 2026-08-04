"""Run JIGSAW from an `hfun.py`, the way `remap` runs ESMF.

Same division of labour as the remapping side: gmpas does not generate the mesh
itself, it prepares every input JIGSAW needs, shells out, and reads the result
back. What it adds is everything around the call -- finding the executable,
checking the distance function before spending the time, reusing work that is
already done, and turning a non-zero exit into a sentence.

This covers the first half of the mini-tutorial's workflow:

    hfun.py --> HFUN.msh --+
             GEOM.msh -----+--> jigsaw --> MESH.msh
             MESH.jig -----+

The second half (`convert_jigsaw.py`, `create_density.py`, then `mkgrid` to get
`grid.nc`) is not here yet; `mkgrid` needs MPI and PnetCDF, which is a separate
problem from running JIGSAW.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .hfun import GRADIENT_GUIDELINE, R_EARTH_KM, Hfun, analysis_grid, diagnose

#: environment variable pointing at JIGSAW, for a build that is not on PATH --
#: which is the normal case, since JIGSAW installs wherever CMAKE_INSTALL_PREFIX
#: said. Either the executable itself or the directory holding it.
JIGSAW_ENV = "JIGSAWDIR"

#: what `MESH.jig` says. Straight from the tutorial; MESH_FILE and the two
#: input files are filled in per run.
JIG_TEMPLATE = """# written by gmpas prep generate
VERBOSITY=1
GEOM_FILE={geom}
HFUN_FILE={hfun}
HFUN_SCAL=absolute
HFUN_HMAX=inf
HFUN_HMIN=0.0
MESH_FILE={mesh}
MESH_DIMS=2
OPTM_QLIM={qlim}
"""

#: JIGSAW's mesh-quality limit, as the tutorial sets it
DEFAULT_QLIM = 0.9375


class GenerateError(RuntimeError):
    """Something the mesh generation cannot proceed without."""


@dataclass
class Generated:
    """What a run produced."""

    mesh_msh: Path
    points: int
    triangles: int
    seconds: float
    reused: bool


# ------------------------------------------------------------- the executable


def find_jigsaw(explicit: str | Path | None = None) -> Path:
    """Locate the JIGSAW executable: `--jigsaw`, then `$JIGSAWDIR`, then PATH.

    PATH last, not first. JIGSAW installs wherever `CMAKE_INSTALL_PREFIX`
    pointed and its build tree leaves the binary in `build/src/`, so on most
    machines it is not on PATH at all and `$JIGSAWDIR` is how you say where it
    went. Either form works — the directory holding it, or the binary itself —
    because both are things people reasonably put in that variable.
    """
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_dir():                       # a build directory, not the binary
            p = p / "jigsaw"
        if not p.exists():
            raise GenerateError(
                f"no jigsaw executable at {p}. Point --jigsaw or "
                f"${JIGSAW_ENV} at the binary, or at the directory holding it."
            )
        if not os.access(p, os.X_OK):
            raise GenerateError(f"{p} is not executable")
        return p.resolve()

    env = os.environ.get(JIGSAW_ENV)
    if env:
        return find_jigsaw(env)

    found = shutil.which("jigsaw")
    if found:
        return Path(found).resolve()

    raise GenerateError(
        f"jigsaw is not on your PATH and ${JIGSAW_ENV} is not set, so the "
        f"mesh cannot be generated. Build it with:\n"
        f"    git clone https://github.com/dengwirda/jigsaw.git\n"
        f"    cd jigsaw && mkdir build && cd build\n"
        f"    cmake .. -DCMAKE_BUILD_TYPE=Release "
        f"-DCMAKE_INSTALL_PREFIX=<where>\n"
        f"    make -j 4 install\n"
        f"then point gmpas at it:\n"
        f"    export {JIGSAW_ENV}=<where>/bin          # or .../build/src\n"
        f"or pass --jigsaw. gmpas deliberately does not generate meshes itself."
    )


# ------------------------------------------------------------- JIGSAW inputs


def write_geom(path: Path, radius_km: float = R_EARTH_KM) -> Path:
    """The sphere JIGSAW is meshing. Two lines, and no reason for more."""
    path.write_text(f"MSHID=3;ellipsoid-mesh\n"
                    f"RADII={radius_km};{radius_km};{radius_km}\n")
    return path


def write_hfun(hfun: Hfun, path: Path, quiet: bool = False) -> tuple[Path, int]:
    """Sample the distance function onto JIGSAW's lat-lon grid and write it.

    This is `create_hfun.py`, reproduced rather than approximated: the same
    grid, the same `meshgrid(lats, lons)` argument order, and therefore the
    same value ordering in the file. The one difference is that the values are
    formatted in bulk instead of one f-string per line, because at a 3 km
    hfun_min that loop is tens of millions of iterations.
    """
    lons, lats = analysis_grid(hfun.hfun_min)
    latgrid, longrid = np.meshgrid(lats, lons)        # (nlon, nlat)
    values = hfun.sample_radians(longrid, latgrid)

    npts = values.size
    if not quiet:
        print(f"  sampling hfun onto {lons.size} x {lats.size} "
              f"({npts / 1e6:.1f}M points)")

    with open(path, "w") as f:
        f.write("MSHID=3;ellipsoid-grid\n")
        f.write("NDIMS=2\n")
        f.write(f"COORD=1;{lons.size}\n")
        f.write("\n".join(map(str, lons.tolist())))
        f.write(f"\nCOORD=2;{lats.size}\n")
        f.write("\n".join(map(str, lats.tolist())))
        f.write(f"\nVALUE={npts}; 1\n")
        # chunked so the joined string never rivals the array itself in size
        flat = values.ravel()
        for i in range(0, npts, 1_000_000):
            f.write("\n".join(map(str, flat[i:i + 1_000_000].tolist())))
            f.write("\n")
    return path, npts


def write_jig(path: Path, geom: str, hfun: str, mesh: str,
              qlim: float = DEFAULT_QLIM, init: str | None = None) -> Path:
    """The config file. Names are relative, since JIGSAW runs in the out dir."""
    text = JIG_TEMPLATE.format(geom=geom, hfun=hfun, mesh=mesh, qlim=qlim)
    if init:
        # an initial point set imposes icosahedral structure, which a uniform
        # HFUN alone will not give -- it produces 7-sided cells instead
        text += f"INIT_FILE={init}\n"
    path.write_text(text)
    return path


# --------------------------------------------------------------- the mesh out


def read_msh_counts(path: Path) -> tuple[int, int]:
    """POINT= and TRIA3= from a JIGSAW mesh file, without reading the body."""
    points = triangles = 0
    with open(path) as f:
        for line in f:
            if line.startswith("POINT="):
                points = int(line.split("=", 1)[1])
            elif line.startswith("TRIA3="):
                triangles = int(line.split("=", 1)[1])
                break
    return points, triangles


# ------------------------------------------------------------------- the run


def generate(hfun_path, out_dir="mesh", jigsaw: str | Path | None = None,
             qlim: float = DEFAULT_QLIM, init: str | None = None,
             force: bool = False, quiet: bool = False,
             allow_steep: bool = False) -> Generated:
    """Take an `hfun.py` all the way to a JIGSAW `MESH.msh`.

    Everything is checked before anything is spent: the executable is located,
    the distance function is loaded and measured, and a transition steeper than
    the guideline stops the run rather than producing a mesh that has to be
    thrown away. Generation takes minutes and the check takes a second.
    """
    tool = find_jigsaw(jigsaw)
    hfun = Hfun.load(hfun_path)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mesh_msh = out / "MESH.msh"

    if mesh_msh.exists() and not force:
        points, triangles = read_msh_counts(mesh_msh)
        if not quiet:
            print(f"  reusing {mesh_msh} ({points:,} points)")
        return Generated(mesh_msh, points, triangles, 0.0, reused=True)

    d = diagnose(hfun)
    if not quiet:
        print(f"  {hfun.path.name}: {d.h_min:.4g} to {d.h_max:.4g} km, "
              f"max gradient {d.max_gradient:.4f}")
    if not d.within_guideline and not allow_steep:
        raise GenerateError(
            f"{hfun.path.name} has a maximum cell size gradient of "
            f"{d.max_gradient:.4f} at {d.at_lat:.2f}, {d.at_lon:.2f}, above "
            f"the {GRADIENT_GUIDELINE} guideline. A mesh built from it will "
            f"change cell size too quickly to be well behaved.\n"
            f"  Widen the transition region, or raise hfun_min.\n"
            f"  Pass --allow-steep to generate it anyway."
        )

    write_geom(out / "GEOM.msh")
    _, npts = write_hfun(hfun, out / "HFUN.msh", quiet=quiet)
    write_jig(out / "MESH.jig", "GEOM.msh", "HFUN.msh", "MESH.msh",
              qlim=qlim, init=init)
    if not quiet:
        size = (out / "HFUN.msh").stat().st_size / 1e6
        print(f"  wrote GEOM.msh, HFUN.msh ({size:.0f} MB) and MESH.jig")
        print(f"  running {tool.name} — this is the slow part")

    t0 = time.perf_counter()
    done = subprocess.run([str(tool), "MESH.jig"], cwd=out,
                          capture_output=True, text=True)
    seconds = time.perf_counter() - t0

    if done.returncode != 0 or not mesh_msh.exists():
        tail = (done.stdout or done.stderr or "").strip().splitlines()[-8:]
        raise GenerateError(
            f"jigsaw failed (exit {done.returncode}) after {seconds:.1f} s.\n"
            + "\n".join(f"    {line}" for line in tail)
        )

    points, triangles = read_msh_counts(mesh_msh)
    if not quiet:
        print(f"  {mesh_msh.name}: {points:,} generating points, "
              f"{triangles:,} triangles in {seconds:.1f} s")
    return Generated(mesh_msh, points, triangles, seconds, reused=False)


def next_steps(result: Generated, out_dir) -> str:
    """What still has to happen to get an MPAS `grid.nc`, and honestly why."""
    return (
        f"\n{result.mesh_msh} holds the generating points and their "
        f"triangulation.\nTurning it into an MPAS grid.nc still needs the "
        f"tutorial's remaining steps:\n"
        f"    convert_jigsaw.py  MESH.msh -> SaveVertices, SaveTriangles\n"
        f"    create_density.py  -> SaveDensity\n"
        f"    cp hfun.py SaveCode\n"
        f"    mkgrid <nominalMinDc_metres>  -> grid.nc, graph.info\n"
        f"gmpas does not run these yet: mkgrid needs MPI and PnetCDF."
    )
