"""A time series spread across many MPAS output files.

MPAS writes one `history.YYYY-MM-DD_HH.MM.SS.nc` per output interval, so a run
is a directory of files rather than one file with a long Time dimension. A
viewer wants exactly one timestep at a time, which makes the obvious tool --
`open_mfdataset`, building a dask graph over every file -- the wrong shape: it
pays to describe the whole series when the answer needs a single slice.

Instead this keeps a list of (file, index-within-file) and opens the one file
needed, holding a few handles open in an LRU. Opening a file is milliseconds;
building the graph over hundreds is not.

For genuine multi-file *analysis* -- time means, composites, anomalies -- reach
for `xarray.open_mfdataset` with dask instead. That is what it is good at, and
this class deliberately does not try to replace it.
"""

from __future__ import annotations

import os
import re
import sys
import threading
import time as _time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from . import netcdf, timing
from .data import SPATIAL_DIMS, find_mesh_beside, plottable, select
from .mesh import MpasMesh, has_mesh
from .paths import resolve_path

#: MPAS names output files by valid time: history.2012-02-25_12.00.00.nc.
#: Separators vary between sites -- `_` or `T` between date and time, `.` or
#: `:` within it -- and the seconds are sometimes dropped.
STAMP = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})[_T](\d{2})(?:[.:](\d{2}))?(?:[.:](\d{2}))?"
)


def parse_time(path: Path) -> datetime | None:
    """The valid time in a filename, or None if it carries no timestamp.

    MPAS puts the valid time in the name, so the whole time axis can be built
    without opening a single file -- which matters on a parallel filesystem
    where opening several hundred files is the slowest thing startup does.
    """
    m = STAMP.search(path.name)
    if m is None:
        return None
    year, month, day, hour, minute, second = m.groups()
    try:
        return datetime(int(year), int(month), int(day), int(hour),
                        int(minute or 0), int(second or 0))
    except ValueError:            # e.g. hour 25 in something that only looked like a stamp
        return None

#: open file handles to keep around while scrubbing through time
LRU_SIZE = 4

#: how much memory materialised (var, step, level) reads may hold, in bytes.
#:
#: A budget in BYTES, deliberately not a count of entries. One field is ~2 MB
#: on a small regional mesh and ~320 MB on a 41M-cell global one, so any fixed
#: entry count is either useless at one end or an out-of-memory kill at the
#: other: 64 entries was ~130 MB on the mesh it was tuned against and ~20 GB
#: on a 3.75 km global mesh, which is exactly how it got an HPC login node
#: killed. Sizing by bytes scales itself -- dozens of small fields, or one
#: large one, for the same footprint either way.
VALUES_CACHE_BYTES = 512 * 1024 * 1024

#: overrides the values-cache budget, in MB. Worth setting on HPC, where a
#: login node's cgroup cap and a compute node's memory differ by orders of
#: magnitude and the same install serves both.
VALUES_CACHE_ENV = "GMPAS_VALUES_CACHE_MB"


def values_budget() -> int:
    """The values-cache budget in bytes, honouring the environment override."""
    raw = os.environ.get(VALUES_CACHE_ENV)
    if not raw:
        return VALUES_CACHE_BYTES
    try:
        return max(0, int(float(raw) * 1024 * 1024))
    except ValueError:                      # unparseable: keep the default
        return VALUES_CACHE_BYTES


#: Below this many files a pool costs more than it saves: the read is a
#: fraction of a second either way, and starting workers is not.
PARALLEL_MIN_FILES = 32

#: How many files to have open at once when reading a point's series. The
#: default is deliberately modest: this often runs on a login node shared with
#: everyone else on the cluster, and the gain is in overlapping I/O latency,
#: not in using every core.
SERIES_WORKERS = 8
SERIES_WORKERS_ENV = "GMPAS_SERIES_WORKERS"


def series_workers() -> int:
    """How many worker processes a point series may use."""
    raw = os.environ.get(SERIES_WORKERS_ENV)
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:                  # unparseable: keep the default
            pass
    return max(1, min(SERIES_WORKERS, os.cpu_count() or 1))


def _pool_context():
    """A start method that is safe to use beside an open HDF5 library.

    Never `fork`: this process has netCDF files open and may be inside the
    library on another thread, and a child that inherits a locked internal
    mutex deadlocks the first time it reads. `forkserver` forks from a clean
    process started before any of that, which costs about 100 ms once --
    nothing against a read whose whole point is that it takes seconds.
    """
    import multiprocessing as mp

    for method in ("forkserver", "spawn"):
        if method in mp.get_all_start_methods():
            return mp.get_context(method)
    return mp.get_context()


def series_pool(workers: int):
    """A pool of reader processes, for one series job to use and shut down."""
    from concurrent.futures import ProcessPoolExecutor

    return ProcessPoolExecutor(max_workers=max(1, workers),
                               mp_context=_pool_context())


def _point_in(nc, name: str, var: str, cell: int, level: int, pins: dict,
              wanted) -> list:
    """(position, value) for each wanted step of one already-open file."""
    if var not in nc.variables:
        raise KeyError(f"{var!r} not in {name}")
    v = nc.variables[var]
    picks = Series._cell_index(v, cell, level, pins)
    out = []
    for index, local in wanted:
        take = tuple(local if d == "Time" else picks[d] for d in v.dimensions)
        out.append((index, float(np.asarray(v[take]).reshape(-1)[0])))
    return out


def _read_point(job) -> list:
    """One file's worth of a point series, in a worker process.

    Deliberately a module-level function taking plain data: it is pickled to
    the worker, so it must not close over a Series, a mesh or anything else
    that would drag the package -- and the point of the worker is to do
    nothing but open, read one hyperslab, and answer.
    """
    import netCDF4

    path, var, cell, level, pins, wanted = job
    with netCDF4.Dataset(path) as nc:
        return _point_in(nc, path, var, cell, level, pins, wanted)


def is_sidecar(path: Path) -> bool:
    """Whether this is a metadata shadow of a real file rather than data.

    macOS writes an AppleDouble sidecar next to every file it copies onto a
    filesystem that cannot hold a resource fork -- `._history.....nc`, four
    kilobytes carrying the same `.nc` suffix as the file it shadows. Any run
    directory that has been through a Mac, a USB stick, or an rsync from one
    holds a shadow copy of itself, and this is common on HPC precisely because
    that is how data arrives there.

    They matter more than their size suggests: `order()` sorts on the
    timestamp in the name, which a sidecar copies verbatim, and ties break on
    the name, where `._x.nc` sorts before `x.nc`. So the sidecar became
    `files[0]` -- the one file `Series.__init__` opens eagerly -- and the
    viewer died with `NetCDF: Unknown file format` before serving a frame.

    Only ever applied to names *found by globbing*: a path given explicitly is
    always honoured, since guessing that a user did not mean the file they
    named is worse than any error it produces.
    """
    return path.name.startswith("._")


def expand(paths) -> list[Path]:
    """Turn a path, glob, directory or list into a sorted list of files."""
    if isinstance(paths, (str, Path)):
        paths = [paths]

    out: list[Path] = []
    for item in paths:
        p = resolve_path(item)
        if p.is_dir():
            out.extend(f for f in p.glob("*.nc") if not is_sidecar(f))
        elif any(ch in str(item) for ch in "*?["):
            base = p.parent
            out.extend(sorted(f for f in base.glob(Path(str(item)).name)
                              if not is_sidecar(f)))
        else:
            out.append(p)

    seen, unique = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    if not unique:
        raise FileNotFoundError(f"no files matched {paths!r}")
    with timing.step("series.expand", files=len(unique)):
        return order(unique)


def label_of(path: Path) -> str:
    """A human-readable valid time, or the filename if it carries none."""
    when = parse_time(path)
    if when is None:
        return path.stem
    return when.strftime("%Y-%m-%d %H:%M" if when.second == 0
                         else "%Y-%m-%d %H:%M:%S")


def order(paths: list[Path]) -> list[Path]:
    """Chronological when every name carries a time, alphabetical otherwise.

    Sorting the names as text happens to be chronological for MPAS's own
    format, but only because it is zero-padded and big-endian. Anything that
    mixes prefixes, or numbers steps rather than stamping them, would come out
    shuffled -- and a shuffled time axis is the kind of wrong that looks like
    a physics problem. So parse, and fall back to names only if some file has
    no timestamp at all.
    """
    times = {p: parse_time(p) for p in paths}
    if all(t is not None for t in times.values()):
        return sorted(paths, key=lambda p: (times[p], p.name))
    return sorted(paths, key=lambda p: p.name)


class Series:
    """Many MPAS files presented as one time axis, opened on demand."""

    def __init__(self, paths, mesh_path: str = "",
                 background_scan: bool = False):
        self.files = expand(paths)
        self._open: OrderedDict[Path, xr.Dataset] = OrderedDict()
        self._values: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._values_bytes = 0
        self._values_budget = values_budget()

        # netCDF4/HDF5 is not safe for concurrent access from multiple
        # threads, and the viewer serves every HTTP request on its own
        # thread (ThreadingHTTPServer) -- a request scrubbing the time slider
        # can run at the same moment as an animation's frame-by-frame loop,
        # which nothing in the UI serializes. Every read through this Series
        # -- cache lookup, LRU eviction, and the disk read itself -- happens
        # under this one lock, held for the read's full duration rather than
        # just around the dict bookkeeping: releasing it as soon as a cached
        # handle is returned would still let one thread's eviction close a
        # dataset another thread is mid-read on, corrupting or NaN-ing that
        # frame rather than raising. `_scan`'s own handles are included, even
        # though they never touch this cache, because the race is in the
        # underlying C library, not just this dict.
        #
        # And for that same reason it is the *process-wide* lock rather than
        # one per Series: a per-instance lock cannot exclude the other Series
        # a dashboard holds, nor the `MpasMesh.load` on its mesh page, and
        # those enter HDF5 concurrently with this one's background scan. See
        # netcdf.LOCK.
        self._lock = netcdf.LOCK

        with self._lock:
            first = self._open_first()

        if mesh_path:
            self.mesh = MpasMesh.load(resolve_path(mesh_path))
        elif has_mesh(first):
            self.mesh = MpasMesh.load(self.files[0])
        else:
            found = find_mesh_beside(self.files[0],
                                     int(first.sizes.get("nCells", -1)))
            if found is None:
                raise KeyError(
                    f"{self.files[0].name} carries no mesh information and no "
                    f"mesh file was found beside it. Pass mesh_path explicitly."
                )
            self.mesh = MpasMesh.load(found)

        self.groups = plottable(first)

        # Counting timesteps means opening every file, which is 86% of startup
        # and grows with the run length -- painful on a parallel filesystem
        # where each open is a network round trip. So start from the assumption
        # every file holds one step, which is what MPAS history output almost
        # always is, and correct it in the background.
        #
        # The provisional axis is a strict *subset* of the real one: step i
        # maps to (file i, 0), which is a genuine timestep whatever the true
        # count turns out to be. Scrubbing during the scan shows real data,
        # never the wrong frame -- only fewer frames than there will be.
        self._counts = {self.files[0]: int(first.sizes.get("Time", 1))}
        self.steps, self.labels = self._axis()
        self.scanning = False

        # A scan left running past its Series keeps reading HDF5 while
        # whatever comes next may be writing a file without the lock, which is
        # a segfault rather than a stale read. So it is stoppable and `close`
        # waits for it.
        self._stop_scan = threading.Event()
        self._scan_thread: threading.Thread | None = None
        if background_scan and len(self.files) > 1:
            self.scanning = True
            self._scan_thread = threading.Thread(target=self._scan, daemon=True,
                                                 name="gmpas-scan")
            self._scan_thread.start()
        elif len(self.files) > 1:
            self._scan()

    # -- the time axis ---------------------------------------------------

    def _axis(self) -> tuple[list[tuple[Path, int]], list[str]]:
        """Build (steps, labels) from whatever counts are known so far."""
        steps: list[tuple[Path, int]] = []
        labels: list[str] = []
        for path in self.files:
            n = self._counts.get(path, 1)
            base = label_of(path)
            for i in range(n):
                steps.append((path, i))
                labels.append(base if n == 1 else f"{base} +{i}")
        return steps, labels

    def _open_first(self) -> xr.Dataset:
        """The first file that actually opens, dropping any that do not.

        Caller must hold `self._lock`.

        One unreadable file used to end the run: `__init__` opened `files[0]`
        eagerly and let the netCDF error out of the constructor, so `gmpas
        view` on a directory exited with a traceback instead of serving the
        other thousand files. On HPC that is a normal state, not a corrupt
        one -- a model still running leaves its newest history file half
        written, and a transfer still in flight leaves a truncated one.

        A file that cannot be opened is dropped from the axis rather than
        held with a placeholder: its timestep count is unknown, so any
        placeholder would put a step on the slider that renders nothing. It
        is reported by name on stderr, never skipped in silence -- if the
        file was supposed to be readable, that is the only clue the user
        gets, and the alternative reads as gmpas quietly losing data.
        """
        problems: list[str] = []
        failures: list[Exception] = []
        while self.files:
            try:
                ds = self._dataset(self.files[0])
            except Exception as exc:
                bad = self.files.pop(0)
                problems.append(f"  {bad.name}: {exc}")
                failures.append(exc)
                continue
            if problems:
                print(f"gmpas: skipped {len(problems)} unreadable file(s):\n"
                      + "\n".join(problems), file=sys.stderr)
            return ds

        # A path that simply is not there stays a FileNotFoundError. It is the
        # overwhelmingly common way to get here -- a typo in an argument -- and
        # the CLI turns that one into a plain message rather than a traceback.
        # Widening it to OSError would have made `gmpas info typo.nc` traceback.
        kind = (FileNotFoundError
                if failures and all(isinstance(e, FileNotFoundError)
                                    for e in failures)
                else OSError)
        raise kind(
            "no readable file among those matched — every candidate failed "
            "to open:\n" + "\n".join(problems)
        )

    def _scan(self) -> None:
        """Count timesteps in every file, then swap the axis in.

        Uses netCDF4 rather than xarray -- 1.6 ms per file against 7.3 ms,
        because reading one dimension does not need xarray's decoding. Keeps
        its own handles so it never touches the LRU another thread is using
        -- but still takes `self._lock` per file, held only for that one
        open+read: the cache dict is not the only thing at risk here, the
        underlying netCDF4/HDF5 library itself is not safe under concurrent
        access from another thread, whether or not the two sides share a
        handle. Locked per file rather than for the whole scan so a frame
        request only ever waits as long as one file's dimension read.
        """
        import netCDF4

        counts = dict(self._counts)
        opened = 0
        with timing.step("series.scan", files=len(self.files)) as t:
            for path in self.files:
                if self._stop_scan.is_set():
                    return                       # the Series is going away
                if path in counts:
                    continue
                try:
                    opened += 1
                    with self._lock, netCDF4.Dataset(path) as nc:
                        dim = nc.dimensions.get("Time")
                        counts[path] = len(dim) if dim is not None else 1
                except Exception:
                    counts[path] = 1      # unreadable: leave it as one step
            t.note(opened=opened)
        self._counts = counts
        # plain assignment, so a reader mid-request keeps a consistent list
        self.steps, self.labels = self._axis()
        self.scanning = False

    # -- files -----------------------------------------------------------

    def _dataset(self, path: Path) -> xr.Dataset:
        """Caller must hold `self._lock` -- see the note in `__init__`."""
        if path in self._open:
            self._open.move_to_end(path)
            return self._open[path]

        ds = xr.open_dataset(path, decode_timedelta=False, engine="netcdf4")
        self._open[path] = ds
        while len(self._open) > LRU_SIZE:
            _, old = self._open.popitem(last=False)
            old.close()
        return ds

    def close(self) -> None:
        self.stop_scan()
        with self._lock:
            for ds in self._open.values():
                ds.close()
            self._open.clear()
            # the materialised fields are the bulk of what this holds; closing
            # the handles but keeping them resident would free almost nothing
            self._values.clear()
            self._values_bytes = 0

    def stop_scan(self, timeout: float = 30.0) -> None:
        """Stop the background step count and wait for it to let go.

        Called by `close`, and worth calling directly before anything writes
        netCDF beside a live Series -- a test fixture building the next file,
        say. The scan checks between files, so this returns as soon as the one
        in flight is done.
        """
        self._stop_scan.set()
        thread, self._scan_thread = self._scan_thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout)
        self.scanning = False

    # -- access ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def times(self) -> list[datetime | None]:
        """Valid time per step, read from the filenames -- no file opened."""
        return [parse_time(path) for path, _ in self.steps]

    @property
    def dated(self) -> bool:
        """Whether every file carries a parseable timestamp."""
        return all(t is not None for t in self.times)

    @property
    def first(self) -> xr.Dataset:
        """The dataset backing step 0 -- what to introspect for variables."""
        with self._lock:
            return self._dataset(self.files[0])

    @property
    def n_files(self) -> int:
        return len(self.files)

    def variables(self, dim: str = "nCells") -> list[str]:
        return self.groups.get(dim, [])

    def dataarray(self, var: str, step: int = 0) -> xr.DataArray:
        """A lazy reference -- `.attrs`/`.dims`/`.sizes` are safe to read
        afterward, but not `.values`: the file behind it can be evicted and
        closed by another thread before you get to it. Use `values()` to
        actually read data."""
        path, _ = self.steps[step]
        with self._lock:
            ds = self._dataset(path)
            if var not in ds:
                raise KeyError(f"{var!r} not in {path.name}")
            return ds[var]

    def values(self, var: str, step: int = 0, level: int = 0,
               sel: dict[str, int] | None = None) -> np.ndarray:
        """One field at one step in the series, as a flat per-element array.

        Re-rendering the same (var, step, level, sel) -- a colormap or range
        change, scrubbing back to a step already visited -- is common and
        was paying a full disk read every time, serialized behind every
        other read in the process (see the lock's own note above). Cache
        the materialised result so only the first request for a given key
        pays that cost; later ones return the same detached array straight
        from the cache lookup, which is why they can share the lock with
        the disk read below rather than needing one of their own.

        The cache is bounded by total BYTES, not entry count -- see
        `VALUES_CACHE_BYTES`. What it holds therefore depends on the mesh:
        many fields of a small one, or a single field of a 41M-cell global
        one, for the same footprint either way.

        The actual disk read (`select` materialises `.values`) happens
        inside the lock along with the cache lookup, not after it -- the
        returned array is a plain, fully detached numpy array, so nothing
        else needs to hold the lock once this returns.
        """
        # sel joins the key: the same (var, step, level) at two different
        # months is two different fields, and a key that ignored it would
        # serve the first one for both.
        key = (var, step, level, tuple(sorted((sel or {}).items())))
        with self._lock:
            cached = self._values.get(key)
            if cached is not None:
                self._values.move_to_end(key)
                return cached
            path, local = self.steps[step]
            ds = self._dataset(path)
            if var not in ds:
                raise KeyError(f"{var!r} not in {path.name}")
            arr = select(ds[var], time=local, level=level, sel=sel)
            self._remember(key, arr)
            return arr

    def at_cell(self, var: str, cell: int, level: int = 0,
                sel: dict[str, int] | None = None, steps=None,
                progress=None, cancel=None, workers=None, pool=None) -> np.ndarray:
        """One mesh element's value at every step: the series behind a probe.

        The only read here that does not materialise a whole field, and it
        exists because the obvious way round is ruinous. `values()` reads the
        entire nCells vector for a step -- 328 MB on a 41M-cell mesh -- so
        asking it for one cell at three thousand steps moves a terabyte to
        collect three thousand numbers, and evicts the whole values cache
        doing it. netCDF reads a hyperslab instead: `v[step, cell]` touches
        one chunk, costs the same whatever the mesh, and the array that grows
        here is the answer itself, 8 bytes a step.

        netCDF4 directly rather than xarray, for the reason `_scan` gives: the
        decoding xarray does is most of the per-file cost and none of it is
        needed for one number. Nothing read here enters the values cache --
        there is nothing worth keeping, and it would only evict what the map
        is using.

        One open per FILE, not per step, so a file holding many steps is read
        once -- and measured, that open is 81% of the whole cost, the value
        itself 0.3 ms. Which is why a long run is read by a pool of PROCESSES:
        threads cannot, since the HDF5 we ship against is not thread-safe, and
        processes have the better property anyway -- the parent never enters
        the library, so `netcdf.LOCK` stays free and the map keeps redrawing
        at full speed while the series reads.

        `steps` limits which of the series' steps are read, as indices into
        `self.steps`; the preview pass uses it to draw a strided series before
        the full one exists. Unread positions come back NaN.

        `progress(done, total)` is called per file and `cancel` is checked
        there, which is where nothing is held.
        """
        wanted = range(len(self.steps)) if steps is None else steps
        plan: OrderedDict[Path, list] = OrderedDict()
        for index in wanted:
            path, local = self.steps[index]
            plan.setdefault(path, []).append((int(index), int(local)))

        out = np.full(len(self.steps), np.nan, dtype=np.float64)
        pins = dict(sel or {})
        if workers is None:
            workers = series_workers()
        if pool is not None or (workers > 1 and len(plan) >= PARALLEL_MIN_FILES):
            return self._at_cell_parallel(var, cell, level, pins, plan, out,
                                          progress, cancel, workers, pool)
        return self._at_cell_serial(var, cell, level, pins, plan, out,
                                    progress, cancel)

    def _at_cell_serial(self, var, cell, level, pins, plan, out,
                        progress, cancel) -> np.ndarray:
        """One file at a time, in this process, under the lock."""
        import netCDF4

        from .jobs import Cancelled

        for done, (path, wanted) in enumerate(plan.items()):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            with self._lock, netCDF4.Dataset(path) as nc:
                for index, value in _point_in(nc, str(path), var, cell, level,
                                              pins, wanted):
                    out[index] = value
            if progress is not None:
                progress(done + 1, len(plan))
            _time.sleep(0)                       # let a frame request in
        return out

    def _at_cell_parallel(self, var, cell, level, pins, plan, out,
                          progress, cancel, workers, pool=None) -> np.ndarray:
        """Many files at once, in worker processes.

        The opens are what this costs, and they overlap: on a parallel
        filesystem each one is a round trip to a metadata server, so the
        wall time falls with the number of them in flight rather than with
        any CPU. Nothing HDF5 happens in this process, so a frame request
        arriving mid-read does not queue behind the lock.
        """
        from concurrent.futures import as_completed

        from .jobs import Cancelled

        jobs = [(str(path), var, cell, level, pins, wanted)
                for path, wanted in plan.items()]
        done = 0
        # A pool handed in belongs to the caller and outlives this pass: the
        # preview and the full read share one, so the ~100 ms of starting
        # workers is paid once per click rather than twice.
        own = pool is None
        if own:
            pool = series_pool(min(workers, len(jobs)))
        try:
            futures = [pool.submit(_read_point, job) for job in jobs]
            for future in as_completed(futures):
                if cancel is not None and cancel.is_set():
                    for f in futures:
                        f.cancel()
                    raise Cancelled()
                for index, value in future.result():
                    out[index] = value
                done += 1
                if progress is not None:
                    progress(done, len(jobs))
        finally:
            if own:
                pool.shutdown(wait=False, cancel_futures=True)
        return out

    @staticmethod
    def _cell_index(v, cell: int, level: int, pins: dict) -> dict:
        """Which index each of a variable's dimensions takes for one cell.

        The mesh dimension takes `cell`; a stacking axis takes `level`, or
        whatever `sel` pins it to. Named by dimension rather than by position
        because a diagnostic writes its levels wherever it likes -- the
        `nIsoLevels` convention this package already follows elsewhere.
        """
        picks = {}
        stack = [d for d in v.dimensions if d != "Time" and d not in SPATIAL_DIMS]
        for dim in v.dimensions:
            if dim == "Time":
                continue
            if dim in SPATIAL_DIMS:
                size = v.shape[v.dimensions.index(dim)]
                if not 0 <= cell < size:
                    raise IndexError(f"cell {cell} is outside {dim}={size}")
                picks[dim] = cell
            elif dim in pins:
                picks[dim] = pins[dim]
            else:
                # the slider drives the first stacking axis; the rest sit at 0,
                # which is what `Viewer._pins` shows the page
                picks[dim] = level if dim == stack[0] else 0
        return picks

    def _remember(self, key: tuple, arr: np.ndarray) -> None:
        """Cache `arr`, evicting oldest entries to stay inside the budget.

        Caller must hold `self._lock`. Note this bounds only what the *cache*
        keeps: a caller combining two fields (a derived variable, say) holds
        its own references to both regardless, which is inherent to the
        operation rather than something a cache can bound away.
        """
        nbytes = int(arr.nbytes)
        # An array larger than the entire budget is never worth keeping: it
        # would evict everything else and still sit there as the sole entry,
        # so the next distinct read evicts it again. Better to not cache it
        # and leave the budget serving reads it can actually satisfy twice.
        if nbytes > self._values_budget:
            return

        self._values[key] = arr
        self._values_bytes += nbytes
        while self._values_bytes > self._values_budget and len(self._values) > 1:
            _, old = self._values.popitem(last=False)
            self._values_bytes -= int(old.nbytes)

    def __repr__(self) -> str:
        return (f"<Series {len(self.steps)} steps across {self.n_files} files"
                f" — {self.mesh.n_cells:,} cells>")
