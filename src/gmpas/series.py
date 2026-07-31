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

import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import xarray as xr

from .data import find_mesh_beside, plottable, select
from .mesh import MpasMesh, has_mesh
from .paths import resolve_path

#: MPAS names output files by valid time, which sorts chronologically as text
STAMP = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2})[.:](\d{2})[.:](\d{2})")

#: open file handles to keep around while scrubbing through time
LRU_SIZE = 4


def expand(paths) -> list[Path]:
    """Turn a path, glob, directory or list into a sorted list of files."""
    if isinstance(paths, (str, Path)):
        paths = [paths]

    out: list[Path] = []
    for item in paths:
        p = resolve_path(item)
        if p.is_dir():
            out.extend(sorted(p.glob("*.nc")))
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
    return unique


def label_of(path: Path) -> str:
    """The valid time in a filename, or the filename if it carries none."""
    m = STAMP.search(path.name)
    return f"{m.group(1)} {m.group(2)}:{m.group(3)}" if m else path.stem


class Series:
    """Many MPAS files presented as one time axis, opened on demand."""

    def __init__(self, paths, mesh_path: str = ""):
        self.files = expand(paths)
        self._open: OrderedDict[Path, xr.Dataset] = OrderedDict()

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

        # (file, index within that file) for every step in the series
        self.steps: list[tuple[Path, int]] = []
        self.labels: list[str] = []
        for path in self.files:
            n = int(self._dataset(path).sizes.get("Time", 1))
            base = label_of(path)
            for i in range(n):
                self.steps.append((path, i))
                self.labels.append(base if n == 1 else f"{base} +{i}")

        self.groups = plottable(first)

    # -- files -----------------------------------------------------------

    def _dataset(self, path: Path) -> xr.Dataset:
        if path in self._open:
            self._open.move_to_end(path)
            return self._open[path]

        ds = xr.open_dataset(path, decode_timedelta=False)
        self._open[path] = ds
        while len(self._open) > LRU_SIZE:
            _, old = self._open.popitem(last=False)
            old.close()
        return ds

    def close(self) -> None:
        for ds in self._open.values():
            ds.close()
        self._open.clear()

    # -- access ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def first(self) -> xr.Dataset:
        """The dataset backing step 0 -- what to introspect for variables."""
        return self._dataset(self.files[0])

    @property
    def n_files(self) -> int:
        return len(self.files)

    def variables(self, dim: str = "nCells") -> list[str]:
        return self.groups.get(dim, [])

    def dataarray(self, var: str, step: int = 0) -> xr.DataArray:
        path, _ = self.steps[step]
        ds = self._dataset(path)
        if var not in ds:
            raise KeyError(f"{var!r} not in {path.name}")
        return ds[var]

    def values(self, var: str, step: int = 0, level: int = 0) -> np.ndarray:
        """One field at one step in the series, as a flat per-element array."""
        path, local = self.steps[step]
        ds = self._dataset(path)
        if var not in ds:
            raise KeyError(f"{var!r} not in {path.name}")
        return select(ds[var], time=local, level=level)

    def __repr__(self) -> str:
        return (f"<Series {len(self.steps)} steps across {self.n_files} files"
                f" — {self.mesh.n_cells:,} cells>")
