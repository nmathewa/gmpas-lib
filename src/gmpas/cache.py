"""A bounded, concurrent get-or-build cache for expensive per-view objects.

Four viewers (`viewer`, `generic`, `prep.meshview`, `prep.hfunview`) each kept
their own pixel-to-cell indices and coastline overlays, and only one of them
had solved the two problems that come with it:

  * **Bound it by bytes, not by entries.** A count is a proxy for memory that
    holds only while entry size is fixed. `VIEW_LRU_SIZE = 12` is ~178 MB of
    view indices at the 1200x700 default and ~1.1 GB at 3840x2160, and two of
    the four viewers used plain dicts that never evicted at all. The same
    reasoning already cost this project an OOM on a memory-capped login node
    once, in the values cache -- see `series.values_budget`.

  * **Never hold the lock across the build.** Building a view index queries a
    KD-tree and rendering an overlay runs cartopy; both take long enough that
    holding a shared lock over them serialises every unrelated request in the
    process. But dropping the lock naively lets N concurrent requests for the
    same missing key each do the whole build, so each key gets a one-shot
    Event that latecomers wait on instead.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict

#: bytes of cached view indices and overlays to keep, per viewer
VIEW_CACHE_BYTES = 256 * 1024 * 1024

VIEW_CACHE_ENV = "GMPAS_VIEW_CACHE_MB"


def view_budget() -> int:
    """Byte budget for one viewer's caches, overridable by environment."""
    raw = os.environ.get(VIEW_CACHE_ENV)
    if not raw:
        return VIEW_CACHE_BYTES
    try:
        return max(0, int(float(raw) * 1024 * 1024))
    except ValueError:
        return VIEW_CACHE_BYTES


def sizeof(value) -> int:
    """Bytes `value` occupies, for the things these caches actually hold.

    Overlays are encoded PNG bytes and view indices expose `nbytes`; anything
    else is charged a nominal amount rather than guessed at, since a wrong
    guess silently breaks the budget rather than failing loudly.
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    n = getattr(value, "nbytes", None)
    return int(n) if n is not None else 1


class BuildCache:
    """Get-or-build, bounded by bytes, safe to share across request threads."""

    def __init__(self, budget: int | None = None, measure=sizeof):
        self.budget = view_budget() if budget is None else budget
        self._measure = measure
        self._items: OrderedDict = OrderedDict()
        self._bytes = 0
        self._pending: dict = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._items)

    @property
    def nbytes(self) -> int:
        return self._bytes

    def get(self, key, build):
        """Return `cache[key]`, calling `build()` at most once per key.

        `build()` runs outside the lock, so a request for a different key never
        waits on it. A build that raises leaves nothing cached and wakes its
        waiters, who retry rather than silently reusing a failure.
        """
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
            ev = self._pending.get(key)
            if ev is None:
                self._pending[key] = ev = threading.Event()
                mine = True
            else:
                mine = False

        if not mine:
            ev.wait()
            with self._lock:
                if key in self._items:
                    return self._items[key]
            return self.get(key, build)          # builder failed: retry

        try:
            value = build()
        except BaseException:
            with self._lock:
                self._pending.pop(key, None)
            ev.set()
            raise

        with self._lock:
            self._remember(key, value)
            self._pending.pop(key, None)
        ev.set()
        return value

    def _remember(self, key, value) -> None:
        """Store `value`, evicting oldest first to stay inside the budget.

        Caller holds the lock. An entry bigger than the whole budget is not
        cached at all: keeping it would evict everything else and still leave
        it as the sole occupant, to be evicted itself by the next distinct
        request. Same rule, and same reasoning, as `Series._remember`.
        """
        n = int(self._measure(value))
        if n > self.budget:
            return

        self._items[key] = value
        self._bytes += n
        while self._bytes > self.budget and len(self._items) > 1:
            _, old = self._items.popitem(last=False)
            self._bytes -= int(self._measure(old))

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._bytes = 0
