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
import threading
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from .data import find_mesh_beside, plottable, select
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
#: on a 3.75 km global mesh, which is exactly how it got a Derecho login node
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


def expand(paths) -> list[Path]:
    """Turn a path, glob, directory or list into a sorted list of files."""
    if isinstance(paths, (str, Path)):
        paths = [paths]

    out: list[Path] = []
    for item in paths:
        p = resolve_path(item)
        if p.is_dir():
            out.extend(p.glob("*.nc"))
        elif any(ch in str(item) for ch in "*?["):
            base = p.parent
            out.extend(sorted(base.glob(Path(str(item)).name)))
        else:
            out.append(p)

    seen, unique = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    if not unique:
        raise FileNotFoundError(f"no files matched {paths!r}")
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
        self._lock = threading.Lock()

        with self._lock:
            first = self._dataset(self.files[0])

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

        if background_scan and len(self.files) > 1:
            self.scanning = True
            threading.Thread(target=self._scan, daemon=True).start()
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
        for path in self.files:
            if path in counts:
                continue
            try:
                with self._lock, netCDF4.Dataset(path) as nc:
                    dim = nc.dimensions.get("Time")
                    counts[path] = len(dim) if dim is not None else 1
            except Exception:
                counts[path] = 1          # unreadable: leave it as one step
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
        with self._lock:
            for ds in self._open.values():
                ds.close()
            self._open.clear()
            # the materialised fields are the bulk of what this holds; closing
            # the handles but keeping them resident would free almost nothing
            self._values.clear()
            self._values_bytes = 0

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

    def values(self, var: str, step: int = 0, level: int = 0) -> np.ndarray:
        """One field at one step in the series, as a flat per-element array.

        Re-rendering the same (var, step, level) -- a colormap or range
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
        key = (var, step, level)
        with self._lock:
            cached = self._values.get(key)
            if cached is not None:
                self._values.move_to_end(key)
                return cached
            path, local = self.steps[step]
            ds = self._dataset(path)
            if var not in ds:
                raise KeyError(f"{var!r} not in {path.name}")
            arr = select(ds[var], time=local, level=level)
            self._remember(key, arr)
            return arr

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
