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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from .config import TargetDomain
from .scrip import write_scrip

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

    def apply(self, src: np.ndarray) -> np.ndarray:
        """Sparse matrix multiply: one source field to one destination field."""
        dst = np.zeros(self.n_b)
        np.add.at(dst, self.row, self.S * np.asarray(src, dtype=np.float64)[self.col])
        return dst

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


def ensure_weights(mesh_path, domain: TargetDomain, out_dir,
                   method: str = "conserve", force: bool = False,
                   quiet: bool = False) -> tuple[Path, bool]:
    """Return a weight file for this mesh and target, building it if needed.

    Weights depend only on the two grids, never on the data, so this runs once
    for a whole run. Returns the path and whether it had to be generated.
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
            "be generated. Install it with:\n"
            "    conda install -c conda-forge esmf nco\n"
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

    cmd = [tool, "-s", src_scrip.name, "-d", dst_scrip.name, "-w", weights.name,
           "-m", method, "--src_regional", "--dst_regional", "--ignore_unmapped"]
    if not quiet:
        print(f"  generating weights: {' '.join(cmd[:1])} ... -m {method}")
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
    """
    jobs = list(jobs)
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


def remap_file(path, weights: Weights, domain: TargetDomain, fields,
               out_path) -> dict:
    """Remap one file's selected fields and write one netCDF beside it."""
    lat, lon = domain.lats(), domain.lons()
    result: dict[str, xr.DataArray] = {}
    slabs = 0
    worst = 0.0

    with xr.open_dataset(path, decode_timedelta=False, engine="netcdf4") as ds:
        keep, skip = remappable(ds, fields)
        n_time = int(ds.sizes.get("Time", 1))

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

    out = xr.Dataset(
        result,
        coords={"lat": ("lat", lat, {"units": "degrees_north"}),
                "lon": ("lon", lon, {"units": "degrees_east"})},
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
    out.to_netcdf(out_path)
    return {"out": out_path, "fields": len(result), "slabs": slabs,
            "skipped": skip, "conservation": worst}
