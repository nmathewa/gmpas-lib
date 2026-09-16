"""Background reads the page can poll, for work too slow to answer inline.

A Hovmöller over a year of hourly files, or one point's time series across three
thousand history files, is minutes of I/O. Neither can be answered inside an
HTTP request without the browser giving up, and neither may hold the netCDF lock
while it runs or the map stops redrawing. So the request starts a job and
returns what it knows -- HTTP 202 and a count -- and the page polls.

The rules this encodes, learned from the Hovmöller it was lifted from:

* **Never block the caller.** `progress()` starts the work and returns
  immediately; only `result()`, which exports need, waits.
* **One job at a time.** A request for something else cancels the running job,
  unless an export is waiting on it -- typing a new band otherwise leaves the
  old read holding the lock with nobody wanting its answer.
* **Errors are remembered, not retried.** A failing read polled every 700 ms
  would otherwise fail forever, once per poll.
* **Results are cached by bytes**, in the same `BuildCache` the rest of the
  viewer budgets with, so a finished read is instant to revisit and a big one
  cannot grow the process without bound.
* **Everything stops on close.** A job outliving its viewer keeps entering HDF5
  while whatever runs next may be writing a file without the lock.

The work itself is a callable taking `progress`, `cancel` and `publish`; it
decides how often to report and where it is safe to give up. It must check
`cancel` at points where it holds nothing, and raise `Cancelled` there.
`publish(partial)` offers something worth drawing before the whole answer
exists -- a strided preview of a long read -- which polling picks up. A
partial is never cached: only what the work returns is, so nothing can later
be served a preview as though it were the finished thing.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from .cache import BuildCache, view_budget

#: How many failed keys to remember. Bounded because the key comes from the
#: page and a user dragging a control can produce a great many of them.
MAX_ERRORS = 16


class Cancelled(Exception):
    """A job was superseded by a request for a different one."""


class Jobs:
    """The background jobs of one viewer, and their finished results."""

    def __init__(self, budget: int | None = None, name: str = "gmpas-job"):
        self.cache = BuildCache(budget=view_budget() if budget is None else budget)
        self._jobs: dict = {}
        self._errors: dict = {}
        self._lock = threading.Lock()
        self._name = name

    # -- asking ----------------------------------------------------------

    def peek(self, key):
        """The finished result for `key`, or None. Never builds, never waits."""
        return self.cache.peek(key)

    def progress(self, key, total: int, work: Callable) -> dict:
        """Where `key` stands, starting it if nothing has: done, running or
        error. Never blocks; the page polls this through HTTP 202."""
        with self._lock:
            if self.cache.peek(key) is not None:
                return {"state": "done", "progress": [total, total]}
            if key in self._errors:
                return {"state": "error", "error": self._errors[key]}
            job = self._start(key, total, work)
            state = {"state": "running", "progress": [job["done"], job["total"]]}
            if job["partial"] is not None:
                state["partial"] = job["partial"]
            return state

    def result(self, key, total: int, work: Callable):
        """The finished thing, waiting for its job. For figures and exports,
        which cannot return a progress count to a user saving a file."""
        while True:
            with self._lock:
                cached = self.cache.peek(key)
                if cached is not None:
                    return cached
                if key in self._errors:
                    raise ValueError(self._errors[key])
                job = self._start(key, total, work)
                job["waiters"] += 1
            job["finished"].wait()
            with self._lock:
                job["waiters"] -= 1
            if job["result"] is not None:
                return job["result"]
            if not job["cancel"].is_set():
                with self._lock:
                    if key in self._errors:
                        raise ValueError(self._errors[key])

    # -- running ---------------------------------------------------------

    def _start(self, key, total: int, work: Callable) -> dict:
        """The job for `key`, started if needed. Caller holds `self._lock`."""
        job = self._jobs.get(key)
        if job is not None and not job["finished"].is_set():
            return job
        for other in self._jobs.values():
            if not other["finished"].is_set() and other["waiters"] == 0:
                other["cancel"].set()

        job = {"done": 0, "total": total, "waiters": 0, "result": None,
               "partial": None,
               "cancel": threading.Event(), "finished": threading.Event()}

        def progress(done, total=None):
            job["done"] = done
            if total is not None:
                job["total"] = total

        def publish(partial):
            job["partial"] = partial

        def run():
            try:
                result = work(progress, job["cancel"], publish)
                job["result"] = result
                self.cache.get(key, lambda: result)
            except Cancelled:
                pass
            except Exception as exc:                        # remembered, not retried
                with self._lock:
                    self._errors[key] = f"{type(exc).__name__}: {exc}"
                    while len(self._errors) > MAX_ERRORS:
                        self._errors.pop(next(iter(self._errors)))
            finally:
                job["finished"].set()
                with self._lock:
                    for k in [k for k, j in self._jobs.items()
                              if j["finished"].is_set() and j is not job]:
                        self._jobs.pop(k)

        self._jobs[key] = job
        threading.Thread(target=run, daemon=True, name=self._name).start()
        return job

    def running(self) -> int:
        """How many jobs are still reading. For shutdown checks and tests."""
        with self._lock:
            return sum(1 for job in self._jobs.values()
                       if not job["finished"].is_set())

    # -- stopping --------------------------------------------------------

    def stop(self, timeout: float = 30.0) -> None:
        """Cancel every job and wait for it to let go of its files."""
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            job["cancel"].set()
        for job in jobs:
            job["finished"].wait(timeout)
