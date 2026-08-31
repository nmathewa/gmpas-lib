"""One lock for every netCDF read in this process.

netCDF4 is a wrapper around HDF5, and unless HDF5 was built `--enable-threadsafe`
-- which the conda-forge and wheel builds are not -- its global state is not
safe under concurrent access. Not "may return stale data": the C library walks
its own structures while another thread is rearranging them, and the process
dies with SIGSEGV and no Python traceback.

That is easy to hit here, because two things run at once by design:

* `Series` counts timesteps on a background thread, so the viewer can serve a
  provisional time axis rather than blocking on a directory of thousands of
  files.
* the viewer serves every HTTP request on its own thread, and a dashboard
  holds several sources -- a run, the mesh under it, an hfun -- each reading
  its own files.

`Series` already had a lock, but one *per instance*, which is the wrong scope:
two `Series` objects, or a `Series` and an `MpasMesh.load` on the mesh page,
take different locks and enter HDF5 together anyway. The hazard belongs to the
library, not to any object, so the lock has to be process-wide too.

Reentrant because the read paths nest: `Series.values` holds it across a read
that goes through `_dataset`, which documents needing it held.

The cost is that netCDF reads in one process serialise. That is what safety
costs here, and it is not the bottleneck: reads are dominated by filesystem
round trips a lock does not add to, and every parallel path that matters --
`plot --all-steps`, `remap -j` -- is multi-*process*, where this lock does not
apply at all.
"""

from __future__ import annotations

import threading

#: held for the whole duration of any netCDF open or read, not just around
#: bookkeeping: releasing early would let one thread close a dataset another
#: is mid-read on.
LOCK = threading.RLock()
