"""The whole remap, as one command.

Read the configuration beside the run, build the weights once, then convert
every file. gmpas still does not compute the weights -- it shells out to
`ESMF_RegridWeightGen` for that -- but it prepares both grids, decides which
fields to carry, applies the result and checks the integral afterwards.

One output file per input file, deliberately. A run of a few hundred history
files concatenated into a single netCDF would be enormous and unwieldy, and
one-in-one-out keeps the valid time in the filename where it already was.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from .config import TargetDomain
from .mesh import GLOBAL_COVERAGE
from .scrip import coverage_of, write_scrip
from .series import parse_time

#: dimensions a field may be stacked along, beyond Time
LEVEL_PREFIXES = ("nVert", "nSoil", "nIso")

#: ESMF crashes intermittently on this platform; weights are a one-off, so retry
WEIGHT_ATTEMPTS = 4


class RemapError(RuntimeError):
    """Something the remap cannot proceed without."""


# ------------------------------------------------------------------ weights


@dataclass
class Weights:
    """A SCRIP weight file, loaded and ready to apply."""

    row: np.ndarray            # destination index, 0-based
    col: np.ndarray            # source index, 0-based
    S: np.ndarray
    area_a: np.ndarray
    area_b: np.ndarray
    frac_a: np.ndarray
    frac_b: np.ndarray
    path: Path
    #: CSR form, built once on first use rather than per apply() call --
    #: see _sparse(). Excluded from repr/equality, it's a cache, not data.
    _matrix: object = field(default=None, repr=False, compare=False)

    @classmethod
    def load(cls, path: str | Path) -> "Weights":
        p = Path(path)
        with xr.open_dataset(p, engine="netcdf4") as w:
            return cls(
                row=w.row.values - 1,        # SCRIP indices are 1-based
                col=w.col.values - 1,
                S=w.S.values,
                area_a=w.area_a.values, area_b=w.area_b.values,
                frac_a=w.frac_a.values, frac_b=w.frac_b.values,
                path=p,
            )

    @property
    def n_a(self) -> int:
        return self.area_a.size

    @property
    def n_b(self) -> int:
        return self.area_b.size

    def _sparse(self):
        """The weights as a CSR matrix, built once and reused.

        `apply()` runs once per (field, level, timestep) slab -- hundreds of
        times over a real run -- and `np.add.at` redid the same scatter-add
        setup from row/col/S every single call. Measured at ~99M nonzeros
        (a 5.7M-cell source, 4.4M-cell target): building the matrix once
        costs ~4s; every `apply()` after that is a single sparse @ dense
        matvec at roughly half of what `np.add.at` cost per call. `coo`-style
        construction from (row, col, S) sums duplicate entries automatically,
        same as `add.at` did -- this is not an approximation of the old
        behaviour, it produces identical output.
        """
        if self._matrix is None:
            from scipy.sparse import csr_matrix
            self._matrix = csr_matrix((self.S, (self.row, self.col)),
                                      shape=(self.n_b, self.n_a))
        return self._matrix

    def apply(self, src: np.ndarray) -> np.ndarray:
        """Sparse matrix multiply: one source field to one destination field."""
        return self._sparse() @ np.asarray(src, dtype=np.float64)

    def conservation_error(self, src: np.ndarray, dst: np.ndarray) -> float:
        """Relative difference between the two area integrals.

        Note the destination integral does **not** multiply by `frac_b`. With
        ESMF's default `norm_type=dstarea` the weights already carry the
        destination coverage fraction, so doing it again double counts and
        reports a fraction of a percent of error on weights that are exact.
        """
        finite = np.isfinite(src)
        i_src = float((np.where(finite, src, 0.0) * self.area_a * self.frac_a).sum())
        i_dst = float((np.nan_to_num(dst) * self.area_b).sum())
        return abs(i_dst - i_src) / abs(i_src) if i_src else 0.0


def _esmf_supports_mpi(tool: str) -> bool | None:
    """Whether this ESMF build can coordinate more than one rank.

    Only rules out the case actually observed: a `mpiuni` build -- ESMF's
    internal stub for "compiled with no real MPI library" -- identified from
    ESMF's own build-info makefile fragment, `esmf.mk`.

    Checked next to `tool` first (`<prefix>/lib/esmf.mk`), `$ESMFMKFILE`
    only as a fallback when that can't be read or has no ESMF_COMM line --
    deliberately the opposite of the order `module load esmf` usually
    implies. `$ESMFMKFILE` describes whichever module was last loaded, not
    necessarily the binary `tool` actually resolved to: this is the same
    shadowing issue 34 is about, one level down. `module load esmf` (real
    MPI) followed by `conda activate` (PATH now finds the conda mpiuni
    copy) leaves a stale `$ESMFMKFILE` pointing at a build that is not the
    one about to run. Trusting it first would call a mpiuni binary
    MPI-capable because a *different*, no-longer-relevant build says so --
    reintroducing the exact corruption this function exists to prevent.
    The file next to the resolved binary is the one description of that
    binary that can't be shadowed this way.

    Measured directly, not assumed: launching a `mpiuni` ESMF_RegridWeightGen
    (the conda-forge build) under `mpirun -np 2` did not parallelize it -- it
    ran two uncoordinated copies that both wrote the same output files in the
    same directory and corrupted each other, both failing with a nonsense
    NetCDF error. A build that doesn't declare itself `mpiuni` might still be
    unsafe to launch this way, for a different reason this check cannot see:
    if the launcher on PATH is a different MPI implementation than the one
    ESMF was linked against (e.g. a Homebrew/Open MPI `mpirun` in front of a
    conda/MPICH-linked ESMF), the ranks are just as uncoordinated and fail the
    same way. This function only answers "is it mpiuni," not "will this
    specific launcher work with it" -- see the HPC TODO on issue 34.

    Returns `None`, not `False`, when `esmf.mk` cannot be found or parsed:
    "unknown" and "known not to work" get different messages upstream.
    """
    import os

    candidates = [Path(tool).resolve().parent.parent / "lib" / "esmf.mk"]
    mkfile = os.environ.get("ESMFMKFILE")
    if mkfile:
        candidates.append(Path(mkfile))

    for mk in candidates:
        try:
            text = mk.read_text()
        except OSError:
            continue
        for line in text.splitlines():
            stripped = line.lstrip("#").strip()
            if not stripped.startswith("ESMF_COMM"):
                continue
            _, _, value = stripped.partition("=")
            if not value:
                _, _, value = stripped.partition(":")
            if value.strip():
                return value.strip() != "mpiuni"
    return None


def _mpi_launch_prefix(ranks: int, tool: str) -> tuple[list[str], str | None]:
    """How to run `ranks` copies of `tool`, or `[]` for one rank.

    ESMF_RegridWeightGen's search and polygon intersection is genuinely
    MPI-parallel, but gmpas invoking it bare only ever gave it one rank --
    see issue 34. srun and mpirun expect different flags and suit different
    schedulers: srun only works inside an active Slurm allocation, so it is
    tried first only when one is actually detected (`SLURM_JOB_ID`); mpirun
    (or mpiexec, the HPE Cray name for the same thing) is the fallback
    everywhere else, including PBS sites like Derecho.

    Returns `(prefix, note)`. `note` is set whenever `ranks > 1` was asked
    for but declined -- silently running single-rank when more was requested
    is exactly the confusion that started this. See `_esmf_supports_mpi` for
    why declining is sometimes the safe choice, not just the cautious one.
    """
    if ranks <= 1:
        return [], None

    supports_mpi = _esmf_supports_mpi(tool)
    if supports_mpi is False:
        return [], ("this ESMF build has no real MPI (ESMF_COMM=mpiuni) -- "
                    "running single-rank instead of corrupting output "
                    "across uncoordinated ranks")
    if supports_mpi is None:
        return [], ("could not tell whether this ESMF build supports MPI "
                    "(no esmf.mk found) -- running single-rank")

    import os

    srun = shutil.which("srun") if os.environ.get("SLURM_JOB_ID") else None
    if srun:
        return [srun, "-n", str(ranks)], None

    launcher = shutil.which("mpirun") or shutil.which("mpiexec")
    if launcher:
        return [launcher, "-np", str(ranks)], None

    return [], "no srun/mpirun/mpiexec on PATH -- running single-rank"


def ensure_weights(mesh_path, domain: TargetDomain, out_dir,
                   method: str = "conserve", force: bool = False,
                   ranks: int = 1, quiet: bool = False) -> tuple[Path, bool]:
    """Return a weight file for this mesh and target, building it if needed.

    Weights depend only on the two grids, never on the data, so this runs once
    for a whole run. Returns the path and whether it had to be generated.

    `ranks` is how many MPI ranks to ask for -- see `_mpi_launch_prefix`. It
    is the same count `gmpas remap` uses for its own worker pool (`-j`), not
    a separate knob: one number describes how much of the machine this run
    gets to use.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = out_dir / f"map_{method}.nc"

    if weights.exists() and not force:
        if not quiet:
            print(f"  weights: reusing {weights}")
        return weights, False

    tool = shutil.which("ESMF_RegridWeightGen")
    if tool is None:
        raise RemapError(
            "ESMF_RegridWeightGen is not on your PATH, so the weights cannot "
            "be generated. gmpas does not install ESMF itself -- load your "
            "system's build (e.g. `module load esmf` on an HPC site) before "
            "running this. A site-provided build is usually already tuned "
            "for the local MPI/interconnect; installing a second copy from "
            "conda-forge into this environment would only compete with it. "
            "gmpas deliberately does not compute conservative weights itself."
        )

    src_scrip = out_dir / "src.scrip.nc"
    dst_scrip = out_dir / "dst.scrip.nc"
    if not quiet:
        print(f"  writing {src_scrip.name} and {dst_scrip.name}")
    _, wrapped = write_scrip(mesh_path, src_scrip)
    if wrapped and not quiet:
        print(f"    normalised {wrapped:,} longitudes onto [0, 2pi)")
    domain.to_scrip(dst_scrip)

    # --src_regional tells ESMF the source does not cover the sphere, which
    # decides how it treats the poles and the seam. It was passed
    # unconditionally, so a global mesh was described to ESMF as regional --
    # and paired with --ignore_unmapped, cells it then failed to map came back
    # silently empty instead of as an error. gmpas already knows which this
    # is, so say so.
    src_coverage = coverage_of(src_scrip)
    src_is_global = src_coverage >= GLOBAL_COVERAGE
    if not quiet:
        kind = "global" if src_is_global else "regional"
        print(f"  source covers {src_coverage * 100:.1f}% of the sphere "
              f"— treating it as {kind}")

    launch, note = _mpi_launch_prefix(ranks, tool)
    cmd = launch + [tool, "-s", src_scrip.name, "-d", dst_scrip.name,
                    "-w", weights.name, "-m", method]
    if not src_is_global:
        cmd.append("--src_regional")
    cmd += ["--dst_regional", "--ignore_unmapped",
                    # ESMF's own default logs every message from every rank,
                    # and warns that this "may cause slowdown in performance"
                    # -- real cost under -np 64+ on a shared/parallel
                    # filesystem, not just noise. gmpas's own error handling
                    # only reads captured stdout/stderr (below), never these
                    # per-PET log files, so there is nothing here that relies
                    # on them existing.
                    "--no_log"]
    if not quiet:
        if note:
            print(f"  {note}")
        prefix = f"{' '.join(launch)} " if launch else ""
        print(f"  generating weights: {prefix}{Path(tool).name} ... -m {method}")
    # ESMF 8.9.1 on macOS segfaults intermittently -- measured at roughly one
    # run in five on byte-identical inputs that succeed the other four times.
    # Weights are generated once for a whole run, so retrying is cheap; it is
    # announced rather than hidden, because a crash that comes and goes is
    # worth knowing about.
    t0 = time.perf_counter()
    for attempt in range(1, WEIGHT_ATTEMPTS + 1):
        weights.unlink(missing_ok=True)
        done = subprocess.run(cmd, capture_output=True, text=True, cwd=out_dir)
        if done.returncode == 0 and weights.exists():
            break
        if attempt < WEIGHT_ATTEMPTS and not quiet:
            print(f"    attempt {attempt} failed (exit {done.returncode}), "
                  f"retrying")
    else:
        tail = (done.stdout or done.stderr or "").strip().splitlines()[-6:]
        crash = " — it segfaulted" if done.returncode < 0 else ""
        raise RemapError(
            f"ESMF_RegridWeightGen failed {WEIGHT_ATTEMPTS} times "
            f"(exit {done.returncode}){crash}.\n"
            + "\n".join(f"    {line}" for line in tail)
        )
    if not quiet:
        note = f" after {attempt} attempts" if attempt > 1 else ""
        print(f"  weights ready in {time.perf_counter() - t0:.1f} s "
              f"({weights.stat().st_size / 1e6:.0f} MB){note}")
    return weights, True


# ------------------------------------------------------------------- fields


def level_dim(da: xr.DataArray) -> str | None:
    for d in da.dims:
        if d.startswith(LEVEL_PREFIXES):
            return d
    return None


def remappable(ds: xr.Dataset, names) -> tuple[list[str], list[tuple[str, str]]]:
    """Split requested names into what these weights can carry, and what not.

    Cell weights only remap cell fields. MPAS carries velocity as `u` on edges
    and vorticity on vertices, and those need their own weight files -- so they
    are reported rather than silently dropped.
    """
    keep: list[str] = []
    skip: list[tuple[str, str]] = []
    for name in names:
        if name not in ds:
            skip.append((name, "not in this file"))
        elif "nCells" in ds[name].dims:
            # remap_file walks Time and one level axis and hands the rest to
            # the weights as a flat per-cell vector, so anything else left in
            # the shape has nowhere to go. Real MPAS history carries such
            # fields -- the ozone climatology is (nCells, nOznLevels,
            # nMonths) -- and reaching them with an unhandled axis fails deep
            # in the remap with numpy talking about dimensions, naming
            # neither the field nor the axis. Report them the way an edge
            # field is reported instead.
            spare = [d for d in ds[name].dims
                     if d != "nCells" and d != "Time"
                     and not d.startswith(LEVEL_PREFIXES)]
            levels = [d for d in ds[name].dims if d.startswith(LEVEL_PREFIXES)]
            if spare:
                skip.append((name, f"has {', '.join(spare)} as well as cells "
                                   f"— only Time and one level axis are handled"))
            elif len(levels) > 1:
                skip.append((name, f"has two level axes ({', '.join(levels)}) "
                                   f"— only one is handled"))
            else:
                keep.append(name)
        elif "nEdges" in ds[name].dims:
            skip.append((name, "on nEdges — needs edge weights"))
        elif "nVertices" in ds[name].dims:
            skip.append((name, "on nVertices — needs vertex weights"))
        else:
            skip.append((name, "not on a mesh element"))
    return keep, skip


# ----------------------------------------------------------------- parallel

#: scheduler variables that state how many cores a job was actually given
CORE_VARS = (
    "SLURM_CPUS_PER_TASK",      # SLURM, --cpus-per-task
    "SLURM_CPUS_ON_NODE",       # SLURM, whole-node allocation
    "PBS_NCPUS", "NCPUS",       # PBS / Torque
    "LSB_DJOB_NUMPROC",         # LSF
    "NSLOTS",                   # Grid Engine
)


def detect_cores() -> tuple[int, str]:
    """How many cores this process may actually use, and how we know.

    `os.cpu_count()` reports the machine, which on a shared HPC node is not
    what the job was given -- asking for 4 cores and then spawning 256 workers
    is a good way to be unpopular. So the scheduler's own statement comes
    first, then the process affinity mask (which respects cgroups and
    taskset), and only then the machine size.
    """
    import os

    for var in CORE_VARS:
        raw = os.environ.get(var, "").strip()
        if raw:
            head = raw.split(",")[0].split("(")[0]      # SLURM writes "4(x2)"
            if head.isdigit() and int(head) > 0:
                return int(head), var

    if hasattr(os, "sched_getaffinity"):                 # Linux: cgroup-aware
        try:
            n = len(os.sched_getaffinity(0))
            if n > 0:
                return n, "affinity mask"
        except OSError:
            pass

    return os.cpu_count() or 1, "machine cores"

#: loaded once per worker process; under fork it is inherited, not re-read
_WEIGHTS: "Weights | None" = None
_WEIGHTS_PATH: Path | None = None


def _init_worker(weights_path, preloaded=None):
    """Give a worker its weights.

    Under `fork` the parent's already-loaded copy is inherited and shared
    copy-on-write, which matters at high core counts: re-reading ~15 MB of
    index arrays in each of 256 workers is several gigabytes of duplication
    for data nobody writes to.
    """
    global _WEIGHTS, _WEIGHTS_PATH
    _WEIGHTS_PATH = Path(weights_path)
    _WEIGHTS = preloaded

    # one BLAS thread each, or the workers fight over the same cores
    import os
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def _remap_one(job):
    """Worker entry point. Top level so it survives being sent to a process."""
    global _WEIGHTS
    path, out_path, domain, fields = job
    if _WEIGHTS is None:                       # spawn: no inheritance
        _WEIGHTS = Weights.load(_WEIGHTS_PATH)
    try:
        info = remap_file(path, _WEIGHTS, domain, fields, out_path)
        info["source"] = Path(path).name
        return info
    except Exception as exc:                   # a bad file must not kill the run
        return {"source": Path(path).name, "error": f"{type(exc).__name__}: {exc}",
                "out": Path(out_path), "fields": 0, "slabs": 0,
                "skipped": [], "conservation": 0.0}


def remap_many(jobs, weights: "Weights", weights_path, workers: int = 1,
               on_done=None):
    """Remap many files, yielding each result as it lands.

    `jobs` is (source, output, domain, fields) tuples. With one worker this
    runs in-process; with more it forks where possible so the weights are
    shared rather than copied.

    Built here, before any forking, for the same reason: `apply()` builds
    and caches the sparse matrix lazily on first use, which is fine for one
    process, but under `-j N` with fork it would mean N workers each
    independently building and allocating their own ~nnz-sized copy at the
    same moment -- right as remapping starts, which is exactly the kind of
    synchronized burst that looks like the run is stuck before it even
    begins. Built here, forked workers inherit the same pages by
    copy-on-write: one build, one allocation, genuinely shared -- not N
    redundant ones. Spawn-context workers reload their own `Weights` from
    disk regardless (see `_remap_one`), so this only helps fork, but fork is
    what Linux -- every real HPC target -- actually uses.
    """
    jobs = list(jobs)
    weights._sparse()

    if workers <= 1:
        _init_worker(weights_path, preloaded=weights)
        for job in jobs:
            info = _remap_one(job)
            if on_done:
                on_done(info)
            yield info
        return

    import multiprocessing as mp

    try:
        ctx = mp.get_context("fork")
        preloaded = weights                    # inherited copy-on-write
    except ValueError:                         # no fork on this platform
        ctx = mp.get_context("spawn")
        preloaded = None

    with ctx.Pool(workers, initializer=_init_worker,
                  initargs=(weights_path, preloaded)) as pool:
        for info in pool.imap_unordered(_remap_one, jobs):
            if on_done:
                on_done(info)
            yield info


# --------------------------------------------------------------------- file


def _decode_xtime(value) -> str:
    """One `xtime` entry -- a fixed-width byte string -- as plain text."""
    if isinstance(value, bytes):
        return value.decode("utf-8").strip()
    return str(value).strip()


def valid_times(ds: xr.Dataset, path, n_time: int) -> np.ndarray | None:
    """Valid time for each of a file's Time steps, or None if none is known.

    `xtime` is MPAS's own record and is preferred when present: it can
    disagree with the filename (a restart run stamped with its start time,
    for instance), and unlike the filename it carries one value per step, not
    just one for the whole file. The filename is the fallback, and only
    usable at all for a single-step file -- there is no way to recover
    several steps' worth of times from one timestamp.

    Returns None rather than raising on a date `datetime` cannot represent
    (some idealised runs use a 360-day or no-leap calendar, where `xtime`
    can contain a date like day 30 of February) -- see the calendar note on
    `remap_file`.
    """
    if "xtime" in ds.variables:
        try:
            return np.array(
                [datetime.strptime(_decode_xtime(v), "%Y-%m-%d_%H:%M:%S")
                 for v in np.asarray(ds["xtime"].values)],
                dtype="datetime64[ns]",
            )
        except ValueError:
            pass   # not a calendar datetime can represent -- try the filename

    if n_time == 1:
        t = parse_time(Path(path))
        if t is not None:
            return np.array([t], dtype="datetime64[ns]")

    return None


def remap_file(path, weights: Weights, domain: TargetDomain, fields,
               out_path) -> dict:
    """Remap one file's selected fields and write one netCDF beside it.

    Attaches a real CF `Time` coordinate when the source's valid time can be
    determined (see `valid_times`), under the source's own declared calendar
    (`config_calendar_type`, MPAS's own namelist-derived attribute; "gregorian"
    -- MPAS's own default -- when the source doesn't carry one). An idealised
    run on a non-standard calendar (360-day, no-leap) can carry an `xtime`
    date that plain Gregorian arithmetic cannot represent at all; `valid_times`
    returns None rather than guessing in that case, and the output is written
    exactly as before this existed -- no Time coordinate, not a wrong one.
    """
    lat, lon = domain.lats(), domain.lons()

    # The source side is checked per field below, against nCells. The
    # destination side has to be checked here, because nothing downstream
    # would say what went wrong: the mismatch surfaces as `cannot reshape
    # array of size N into shape (nlat, nlon)` from numpy, once per file,
    # with nothing pointing at the weights that are actually stale.
    expected = domain.nlat * domain.nlon
    if weights.n_b != expected:
        raise ValueError(
            f"{weights.path.name} was built for a target grid of "
            f"{weights.n_b} points, but this target domain is "
            f"{domain.nlat} x {domain.nlon} = {expected}. The weights and the "
            f"domain disagree -- rebuild the weights for this domain with "
            f"`--force-weights`, or point at the target_domain the existing "
            f"weights were made for."
        )

    result: dict[str, xr.DataArray] = {}
    slabs = 0
    worst = 0.0

    with xr.open_dataset(path, decode_timedelta=False, engine="netcdf4") as ds:
        keep, skip = remappable(ds, fields)
        n_time = int(ds.sizes.get("Time", 1))
        has_source_time = "Time" in ds.dims
        times = valid_times(ds, path, n_time) if has_source_time else None
        xtime = (np.array(ds["xtime"].values)
                if has_source_time and "xtime" in ds.variables else None)
        calendar = ds.attrs.get("config_calendar_type") or "gregorian"

        for name in keep:
            da = ds[name]
            if da.sizes["nCells"] != weights.n_a:
                skip.append((name, f"{da.sizes['nCells']} cells, weights expect "
                                   f"{weights.n_a}"))
                continue

            lev = level_dim(da)
            n_lev = int(da.sizes[lev]) if lev else 1
            has_time = "Time" in da.dims

            # Keep the source dimension name. Calling every vertical axis
            # "lev" makes xarray try to align nVertLevels (55) against
            # nVertLevelsP1 (56) and nSoilLevels (4), and it refuses -- they
            # are genuinely different coordinates, not one dimension.
            dims: list[str] = []
            if has_time:
                dims.append("Time")
            if lev:
                dims.append(lev)
            dims += ["lat", "lon"]

            shape = ([n_time] if has_time else []) + ([n_lev] if lev else []) \
                + [domain.nlat, domain.nlon]
            block = np.empty(shape, dtype=np.float32)

            for t in range(n_time if has_time else 1):
                slice_t = da.isel(Time=t) if has_time else da
                for k in range(n_lev):
                    src = (slice_t.isel({lev: k}) if lev else slice_t).values
                    dst = weights.apply(src)
                    if slabs == 0:
                        worst = max(worst, weights.conservation_error(src, dst))
                    index = ((t,) if has_time else ()) + ((k,) if lev else ())
                    block[index] = dst.reshape(domain.nlat, domain.nlon)
                    slabs += 1

            result[name] = xr.DataArray(block, dims=dims, attrs=dict(da.attrs))

    coords = {"lat": ("lat", lat, {"units": "degrees_north"}),
             "lon": ("lon", lon, {"units": "degrees_east"})}
    encoding = {}
    if times is not None:
        coords["Time"] = ("Time", times)
        encoding["Time"] = {"units": "hours since 1970-01-01", "calendar": calendar}
    if xtime is not None:
        result["xtime"] = xr.DataArray(
            xtime, dims=("Time",),
            attrs={"note": "MPAS's own record of the valid time, carried "
                           "through unchanged"},
        )

    out = xr.Dataset(
        result,
        coords=coords,
        attrs={
            "title": "MPAS output remapped to a regular lat-lon grid",
            "source_file": Path(path).name,
            "weights": weights.path.name,
            "method": "first-order conservative (ESMF), area-preserving",
            "longitude_convention":
                "continuous across the antimeridian: values may exceed 180E",
            "history": "written by gmpas remap",
        },
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_netcdf(out_path, encoding=encoding or None)
    return {"out": out_path, "fields": len(result), "slabs": slabs,
            "skipped": skip, "conservation": worst, "time_coord": times is not None}
