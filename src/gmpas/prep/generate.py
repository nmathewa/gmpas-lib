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

#: environment variable pointing at mkgrid. mkgrid is not released anywhere on
#: its own -- it is mkgrid.c in the mini-tutorial repository, built against MPI
#: and PnetCDF -- so there is no package to depend on and no PATH convention.
MKGRID_ENV = "MKGRIDFILE"

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
    out_dir: Path = Path(".")
    hfun_min_km: float = 0.0
    grid_nc: Path | None = None      # set once mkgrid has run
    graph_info: Path | None = None
    cells: int = 0
    mkgrid_seconds: float = 0.0

    @property
    def nominal_min_dc(self) -> float:
        """`mkgrid`'s one argument: the finest grid distance, in METRES.

        hfun.py works in km throughout and mkgrid wants metres, which is the
        one unit seam in this workflow and an easy factor of a thousand to get
        wrong by hand.
        """
        return self.hfun_min_km * 1000.0


# ------------------------------------------------------------- the executable


def _resolve_tool(explicit, env_var: str, name: str, how: str) -> Path:
    """An external executable, from `--flag` or from the environment.

    Deliberately no PATH fallback. Both of these are built by hand into a
    location of the builder's choosing -- JIGSAW wherever CMAKE_INSTALL_PREFIX
    pointed, mkgrid wherever the tutorial repository was cloned -- so PATH is
    the exception rather than the rule, and silently picking up some other
    binary of the same name is a worse outcome than a sentence saying which
    variable to set. Naming the tool is a prerequisite, like a compiler.
    """
    if not explicit:
        explicit = os.environ.get(env_var)
    if not explicit:
        raise GenerateError(
            f"${env_var} is not set, so {name} cannot be run.\n{how}"
        )

    p = Path(explicit).expanduser()
    if p.is_dir():                    # the directory holding it, not the binary
        p = p / name
    if not p.exists():
        raise GenerateError(
            f"no {name} executable at {p}\n"
            f"  Point ${env_var} at the binary, or at the directory holding it."
        )
    if not os.access(p, os.X_OK):
        raise GenerateError(
            f"{p} is not executable\n"
            f"  chmod +x it, or point ${env_var} somewhere else."
        )
    return p.resolve()


def find_jigsaw(explicit: str | Path | None = None) -> Path:
    """The JIGSAW executable, from `--jigsaw` or `$JIGSAWDIR`."""
    return _resolve_tool(
        explicit, JIGSAW_ENV, "jigsaw",
        "  Build it:\n"
        "    git clone https://github.com/dengwirda/jigsaw.git\n"
        "    cd jigsaw && mkdir build && cd build\n"
        "    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<where>\n"
        "    make -j 4 install\n"
        "  then:\n"
        f"    export {JIGSAW_ENV}=<where>/bin        # or .../jigsaw/build/src\n"
        "  or pass --jigsaw. It is also on conda-forge: conda install -c "
        "conda-forge jigsaw",
    )


def find_mkgrid(explicit: str | Path | None = None) -> Path:
    """The mkgrid executable, from `--mkgrid` or `$MKGRIDFILE`.

    mkgrid is not released anywhere on its own -- it is `mkgrid.c` in the
    MPAS/WRF mini-tutorial repository, built against MPI and PnetCDF -- so
    there is nowhere for gmpas to fetch it from and nothing to fall back on.
    """
    return _resolve_tool(
        explicit, MKGRID_ENV, "mkgrid",
        "  It is mkgrid.c in the mini-tutorial repository, and has to be "
        "built:\n"
        "    git clone https://github.com/mgduda/mpas_jigsaw_tutorial.git\n"
        "    cd mpas_jigsaw_tutorial\n"
        "    export PNETCDF=$(brew --prefix pnetcdf)   # or your PnetCDF prefix\n"
        "    make                                      # needs mpicc\n"
        "  then:\n"
        f"    export {MKGRID_ENV}=<path>/mpas_jigsaw_tutorial/mkgrid\n"
        "  or pass --mkgrid.",
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


def convert_jigsaw(mesh_msh: Path, out_dir: Path,
                   quiet: bool = False) -> tuple[Path, Path, np.ndarray]:
    """MESH.msh -> SaveVertices, SaveTriangles, and the coordinates in memory.

    This is `convert_jigsaw.py`. It differs from that script in two ways, both
    deliberate.

    It dispatches on the section headers it knows rather than on a running
    counter. A real MESH.msh has a POWER block between POINT and TRIA3 -- the
    per-point weights for a power diagram -- which the script skips only as a
    side effect of `i >= n` staying true across it. Reading headers means an
    unfamiliar section is ignored because it is unfamiliar, not by luck.

    And it keeps the coordinates it parsed, so `create_density` does not have to
    read the file it just wrote back off disk with `np.loadtxt`. The written
    columns are still the file's own tokens rather than reformatted floats, so
    no precision is lost passing through.
    """
    vertices = out_dir / "SaveVertices"
    triangles = out_dir / "SaveTriangles"
    xyz: list[tuple[float, float, float]] = []
    n_tri = 0

    with open(mesh_msh) as msh:
        for line in msh:
            if line.startswith("POINT="):
                n = int(line.split("=", 1)[1])
                with open(vertices, "w") as out:
                    for _ in range(n):
                        parts = next(msh).split(";")
                        out.write(f"{parts[0]} {parts[1]} {parts[2]}\n")
                        xyz.append((float(parts[0]), float(parts[1]),
                                    float(parts[2])))
            elif line.startswith("TRIA3="):
                n_tri = int(line.split("=", 1)[1])
                with open(triangles, "w") as out:
                    for _ in range(n_tri):
                        parts = next(msh).split(";")
                        out.write(f"{parts[0]} {parts[1]} {parts[2]}\n")

    if not xyz:
        raise GenerateError(f"{mesh_msh} carries no POINT section")
    if not n_tri:
        raise GenerateError(f"{mesh_msh} carries no TRIA3 section")

    if not quiet:
        print(f"  SaveVertices ({len(xyz):,}) and SaveTriangles ({n_tri:,})")
    return vertices, triangles, np.asarray(xyz, dtype=np.float64)


def create_density(hfun: Hfun, xyz: np.ndarray, out_dir: Path,
                   radius_km: float = R_EARTH_KM,
                   quiet: bool = False) -> Path:
    """SaveDensity: the mesh density function at each generating point.

    This is `create_density.py`. MPAS's meshDensity is

        rho(x) = (h_fine / h(x)) ** 4

    which is 1 where the mesh is finest and falls off as the fourth power --
    the relation between cell size and density in a centroidal Voronoi
    tessellation. `mkgrid` combines it with nominalMinDc to recover a smooth
    nominal cell size anywhere on the mesh.
    """
    unit = xyz / radius_km
    lon = np.atan2(unit[:, 1], unit[:, 0])
    lat = np.asin(np.clip(unit[:, 2], -1.0, 1.0))

    dx = hfun.sample_radians(lon, lat)

    # Written the way create_density.py writes it, and not simplified to the
    # algebraically identical (hfun_min / dx) ** 4. The two differ in the last
    # bit or two, and matching the reference implementation exactly is worth
    # more than the tidier expression: it lets the output be checked against
    # the script byte for byte, which is a far stronger test than "close".
    density = (1.0 / (dx / hfun.hfun_min)) ** 4

    path = out_dir / "SaveDensity"
    with open(path, "w") as f:
        f.write("\n".join(map(str, density.tolist())))
        f.write("\n")
    if not quiet:
        print(f"  SaveDensity ({density.min():.4g} to {density.max():.4g})")
    return path


def save_code(hfun: Hfun, out_dir: Path) -> Path:
    """SaveCode: the hfun.py that produced all this, carried alongside it.

    `mkgrid` wants it there, and it doubles as the record of what the mesh was
    asked to be -- which is the one thing a grid.nc cannot tell you afterwards.
    """
    path = out_dir / "SaveCode"
    shutil.copyfile(hfun.path, path)
    return path


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


#: lines of a tool's output kept for the error message when it fails
TAIL_LINES = 8


def _run_streaming(cmd, cwd, quiet: bool = False,
                   prefix: str = "    ") -> tuple[int, list[str]]:
    """Run a tool, echoing its output live, and keep the tail for errors.

    `subprocess.run(capture_output=True)` swallows everything until the
    process exits, which for JIGSAW means one line of "this is the slow
    part" and then total silence for however many minutes it runs -- no way
    to tell progress from a stall. MESH.jig asks for VERBOSITY=1, so JIGSAW
    is emitting progress the whole time; this stops throwing it away.

    Returns (returncode, last TAIL_LINES lines), so a failure still reports
    the tail even though the output has already been shown.
    """
    from collections import deque

    tail: deque[str] = deque(maxlen=TAIL_LINES)
    proc = subprocess.Popen(
        [str(c) for c in cmd], cwd=cwd, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1,                       # line buffered: progress as it happens
    )
    with proc.stdout:
        for line in proc.stdout:
            line = line.rstrip("\n")
            tail.append(line)
            if not quiet:
                print(f"{prefix}{line}", flush=True)
    return proc.wait(), list(tail)


def generate(hfun_path, out_dir="mesh", jigsaw: str | Path | None = None,
             qlim: float = DEFAULT_QLIM, init: str | None = None,
             force: bool = False, quiet: bool = False,
             allow_steep: bool = False, mkgrid: str | Path | None = None,
             skip_mkgrid: bool = False) -> Generated:
    """Take an `hfun.py` all the way to an MPAS `grid.nc`.

    Everything is checked before anything is spent: BOTH executables are
    located, the distance function is loaded and measured, and a transition
    steeper than the guideline stops the run rather than producing a mesh that
    has to be thrown away. Generation takes minutes; the checks take a second,
    and finding out that mkgrid is missing after JIGSAW has run for five
    minutes helps nobody.
    """
    tool = find_jigsaw(jigsaw)
    mkgrid_tool = None if skip_mkgrid else find_mkgrid(mkgrid)
    hfun = Hfun.load(hfun_path)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mesh_msh = out / "MESH.msh"

    if mesh_msh.exists() and not force:
        points, triangles = read_msh_counts(mesh_msh)
        if not quiet:
            print(f"  reusing {mesh_msh} ({points:,} points)")
        result = Generated(mesh_msh, points, triangles, 0.0, reused=True,
                           out_dir=out, hfun_min_km=hfun.hfun_min)
        _mkgrid_inputs(hfun, result, out, quiet=quiet)
        run_mkgrid(mkgrid_tool, result, out, force=force, quiet=quiet)
        return result

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
    code, tail = _run_streaming([tool, "MESH.jig"], cwd=out, quiet=quiet)
    seconds = time.perf_counter() - t0

    if code != 0 or not mesh_msh.exists():
        raise GenerateError(
            f"jigsaw failed (exit {code}) after {seconds:.1f} s.\n"
            + "\n".join(f"    {line}" for line in tail)
        )

    points, triangles = read_msh_counts(mesh_msh)
    if not quiet:
        print(f"  {mesh_msh.name}: {points:,} generating points, "
              f"{triangles:,} triangles in {seconds:.1f} s")

    result = Generated(mesh_msh, points, triangles, seconds, reused=False,
                       out_dir=out, hfun_min_km=hfun.hfun_min)
    _mkgrid_inputs(hfun, result, out, quiet=quiet)
    run_mkgrid(mkgrid_tool, result, out, force=force, quiet=quiet)
    return result


def _mkgrid_inputs(hfun: Hfun, result: Generated, out: Path,
                   quiet: bool = False) -> None:
    """Everything `mkgrid` reads, beside the mesh JIGSAW just produced."""
    _, _, xyz = convert_jigsaw(result.mesh_msh, out, quiet=quiet)
    create_density(hfun, xyz, out, quiet=quiet)
    save_code(hfun, out)


def run_mkgrid(tool: Path | None, result: Generated, out: Path,
               force: bool = False, quiet: bool = False) -> None:
    """The last leg: generating points and their triangulation -> grid.nc.

    mkgrid takes one argument, `nominalMinDc` in METRES, while hfun.py works in
    km throughout. That factor of a thousand is the only unit seam in the
    workflow, so it is computed rather than typed.

    Like jigsaw, it insists on running where its inputs are: the Save* names it
    opens are relative to the working directory.
    """
    if tool is None:
        return

    grid = out / "grid.nc"
    graph = out / "graph.info"
    if grid.exists() and not force:
        if not quiet:
            print(f"  reusing {grid}")
    else:
        nominal = result.nominal_min_dc
        if not quiet:
            print(f"  running {tool.name} {nominal:g} "
                  f"(nominalMinDc in metres = hfun_min * 1000)")
        grid.unlink(missing_ok=True)

        t0 = time.perf_counter()
        code, tail = _run_streaming([tool, f"{nominal:g}"], cwd=out, quiet=quiet)
        result.mkgrid_seconds = time.perf_counter() - t0

        if code != 0 or not grid.exists():
            raise GenerateError(
                f"mkgrid failed (exit {code}) after "
                f"{result.mkgrid_seconds:.1f} s.\n"
                + "\n".join(f"    {line}" for line in tail)
            )

    result.grid_nc = grid
    result.graph_info = graph if graph.exists() else None
    result.cells = _cell_count(grid)
    if not quiet:
        size = grid.stat().st_size / 1e6
        print(f"  grid.nc: {result.cells:,} cells ({size:.0f} MB)"
              + (f" in {result.mkgrid_seconds:.1f} s"
                 if result.mkgrid_seconds else ""))


def _cell_count(grid: Path) -> int:
    """nCells straight from the netCDF header -- no data is read."""
    try:
        import netCDF4

        with netCDF4.Dataset(grid) as nc:
            return len(nc.dimensions.get("nCells", ()))
    except Exception:
        return 0


def next_steps(result: Generated, out_dir=None) -> str:
    """What was produced, and what to do with it."""
    out = Path(out_dir) if out_dir is not None else result.out_dir

    if result.grid_nc is None:
        # --skip-mkgrid: say exactly what is left, with the units already done
        return (
            f"\n{out}/ holds everything mkgrid reads:\n"
            f"    SaveVertices   {result.points:,} generating points\n"
            f"    SaveTriangles  {result.triangles:,} triangles\n"
            f"    SaveDensity    the mesh density at each point\n"
            f"    SaveCode       a copy of the hfun.py that produced them\n"
            f"\nmkgrid was skipped. To finish by hand:\n"
            f"    cd {out} && mkgrid {result.nominal_min_dc:g}\n"
            f"That argument is nominalMinDc in METRES; hfun.py works in km, "
            f"so it is hfun_min * 1000."
        )

    graph = (f"    {result.graph_info}   METIS input: "
             f"gpmetis graph.info <nparts>\n" if result.graph_info else "")
    return (
        f"\n{result.grid_nc}   {result.cells:,} cells\n"
        f"{graph}"
        f"\nLook at it:\n"
        f"    gmpas info      {result.grid_nc} --mesh-only\n"
        f"    gmpas prep view {result.grid_nc}"
    )
