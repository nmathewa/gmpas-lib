"""The shared view cache: bounded by bytes, and never locked across a build.

Both properties only show themselves at production scale -- a 41M-cell mesh
and a 4K browser window -- so they are forced here with tiny budgets and
deliberately slow builds instead.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from gmpas.cache import VIEW_CACHE_BYTES, BuildCache, sizeof, view_budget


def test_the_budget_is_bytes_and_overridable(monkeypatch):
    monkeypatch.delenv("GMPAS_VIEW_CACHE_MB", raising=False)
    assert view_budget() == VIEW_CACHE_BYTES

    monkeypatch.setenv("GMPAS_VIEW_CACHE_MB", "8")
    assert view_budget() == 8 * 1024 * 1024

    # a knob nobody can read the units of is worse than no knob
    monkeypatch.setenv("GMPAS_VIEW_CACHE_MB", "0.5")
    assert view_budget() == 512 * 1024


def test_an_unparseable_budget_falls_back_rather_than_crashing(monkeypatch):
    monkeypatch.setenv("GMPAS_VIEW_CACHE_MB", "lots")
    assert view_budget() == VIEW_CACHE_BYTES


def test_sizeof_measures_what_these_caches_actually_hold():
    """Overlays are encoded PNG bytes; view indices expose nbytes."""
    assert sizeof(b"1234") == 4
    assert sizeof(np.zeros(10, dtype=np.int64)) == 80

    class Index:
        nbytes = 4096

    assert sizeof(Index()) == 4096


def test_a_value_is_built_once_per_key():
    calls = []
    cache = BuildCache(budget=1_000_000)

    for _ in range(5):
        cache.get("k", lambda: calls.append(1) or "value")

    assert calls == [1]


def test_eviction_is_oldest_first_and_stops_at_the_budget():
    cache = BuildCache(budget=300)
    for i in range(10):
        cache.get(i, lambda: b"x" * 100)

    assert cache.nbytes <= 300
    assert len(cache) == 3
    assert 0 not in cache._items          # the oldest went first
    assert 9 in cache._items


def test_an_entry_larger_than_the_budget_is_returned_but_not_kept():
    """Keeping it would evict everything else and still leave it as the sole
    occupant, to be evicted itself by the next request. Same rule as
    Series._remember."""
    cache = BuildCache(budget=50)
    value = cache.get("big", lambda: b"x" * 500)

    assert value == b"x" * 500
    assert len(cache) == 0
    assert cache.nbytes == 0


def test_a_failed_build_caches_nothing_and_is_retried():
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("cartopy had a bad day")
        return "second time lucky"

    cache = BuildCache(budget=1_000_000)
    with pytest.raises(RuntimeError):
        cache.get("k", flaky)
    assert len(cache) == 0

    assert cache.get("k", flaky) == "second time lucky"
    assert len(attempts) == 2


def test_the_lock_is_not_held_across_a_build():
    """The bug this class exists to prevent: a KD-tree query or a cartopy
    render takes long enough that holding a shared lock over it serialises
    every unrelated request in the process."""
    cache = BuildCache(budget=1_000_000)
    started = threading.Event()

    def slow():
        started.set()
        time.sleep(0.5)
        return b"slow"

    t = threading.Thread(target=lambda: cache.get("slow", slow))
    t.start()
    assert started.wait(2.0)

    # a different key must not wait behind the slow build
    began = time.perf_counter()
    cache.get("other", lambda: b"fast")
    assert time.perf_counter() - began < 0.25

    t.join()


def test_concurrent_requests_for_one_key_share_a_single_build():
    """Dropping the lock naively lets N threads each do the whole build."""
    builds = []
    cache = BuildCache(budget=1_000_000)

    def slow():
        builds.append(1)
        time.sleep(0.2)
        return b"once"

    threads = [threading.Thread(target=lambda: cache.get("k", slow))
               for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert builds == [1]
