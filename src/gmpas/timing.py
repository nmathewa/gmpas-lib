"""Where the time went, when someone asks.

Nothing in this package was instrumented, which made "it takes one to two
minutes to open a 41M-cell run" an unattributable number -- and it is not even
one number, because `info` never builds a KD-tree, `view` builds it inside the
first frame request, and `plot` builds it inside `rasterize`. So the answer
depends on the subcommand, and guessing which stage dominates has been wrong
before.

Off by default and free when off: `step()` is rebound at import to a function
returning a stateless singleton, so a disabled call allocates nothing. That
matters because some of the call sites run per animation frame, where a
`@contextmanager` checking a flag would still build a generator every time.

    GMPAS_TIMING=1      startup-scale stages
    GMPAS_TIMING=2      adds per-frame stages and peak-RSS deltas
    GMPAS_TIMING_FILE   write here instead of stderr

Import this as a module (`from . import timing`) and call `timing.step(...)`,
not `from .timing import step` -- the name is rebound when the environment
changes, which is what lets the tests turn it on.
"""

from __future__ import annotations

import atexit
import os
import sys
import threading
import time

#: prefix every line carries, so a Derecho batch log greps cleanly
PREFIX = "gmpas.timing"

#: set when GMPAS_TIMING parses as an integer > 0
LEVEL = 0

_MAIN_PID = os.getpid()
_lock = threading.Lock()

#: label -> [count, total seconds, slowest single occurrence]
_totals: dict[str, list] = {}


class _Null:
    """What `step()` returns when timing is off: no allocation, no state.

    A single shared instance is safe to enter from several threads at once
    precisely because it holds nothing -- the viewer serves every request on
    its own thread, and per-frame call sites must not pay for a lock.
    """

    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def note(self, **fields):
        pass


_NULL = _Null()


def _stream():
    path = os.environ.get("GMPAS_TIMING_FILE")
    if not path:
        return sys.stderr
    try:
        # line-buffered and append-mode: a -j 32 pool has 32 writers, and
        # short line-sized appends to one file interleave without tearing
        # where a buffered write would
        return open(path, "a", buffering=1)
    except OSError:
        return sys.stderr


def _rss_kb() -> int:
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:            # not POSIX, or resource unavailable
        return 0


def _fmt(n: float) -> str:
    """Bytes as a short human string. Local to keep this module dependency-free."""
    for unit in ("KB", "MB", "GB"):
        if abs(n) < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


class _Step:
    """One timed stage. Fields set with `note()` land on the same line."""

    __slots__ = ("label", "fields", "_t0", "_rss0")

    def __init__(self, label: str, fields: dict):
        self.label = label
        self.fields = fields
        self._t0 = time.perf_counter()
        self._rss0 = _rss_kb() if LEVEL >= 2 else 0

    def note(self, **fields):
        self.fields.update(fields)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        elapsed = time.perf_counter() - self._t0

        with _lock:
            slot = _totals.get(self.label)
            if slot is None:
                _totals[self.label] = [1, elapsed, elapsed]
            else:
                slot[0] += 1
                slot[1] += elapsed
                slot[2] = max(slot[2], elapsed)

        parts = [f"{k}={v}" for k, v in self.fields.items() if v is not None]
        if LEVEL >= 2:
            grew = _rss_kb() - self._rss0
            if grew > 0:
                parts.append(f"rss+={_fmt(grew * 1024)}")
        if os.getpid() != _MAIN_PID:
            parts.append(f"pid={os.getpid()}")

        print(f"{PREFIX}  {self.label:<20} {elapsed:8.3f}s  {' '.join(parts)}".rstrip(),
              file=_stream())
        return False


def _step_null(label: str, **fields):
    return _NULL


def _step_real(label: str, **fields):
    return _Step(label, fields)


#: rebound by `refresh()`; see the module docstring for why call sites must
#: reach it through the module rather than importing the name
step = _step_null


def enabled(level: int = 1) -> bool:
    """Whether timing is on at or above `level`. For guarding costly fields."""
    return LEVEL >= level


def _report() -> None:
    if not _totals:
        return
    out = _stream()
    print(f"{PREFIX}  ---- roll-up " + "-" * 44, file=out)
    for label, (n, total, worst) in sorted(
            _totals.items(), key=lambda kv: -kv[1][1]):
        print(f"{PREFIX}  {label:<20} {total:8.3f}s  n={n} max={worst:.3f}s",
              file=out)


def refresh() -> None:
    """Re-read the environment and rebind `step`.

    Called once at import. Tests call it again after monkeypatching the
    environment, since the level is resolved at bind time rather than on every
    call -- that is the whole point of the disabled path being free.
    """
    global LEVEL, step
    try:
        LEVEL = int(os.environ.get("GMPAS_TIMING") or 0)
    except ValueError:
        LEVEL = 0
    step = _step_real if LEVEL > 0 else _step_null


def reset() -> None:
    """Drop accumulated totals. For tests that assert on a single run."""
    with _lock:
        _totals.clear()


# ------------------------------------------------------- progress reporting
#
# Unlike everything above, this is always on: it exists so a long operation
# says it is still working. It lives here rather than in `cli` so that library
# modules can reach it -- `cli` imports the package, so the package importing
# `cli` back would be a cycle.


class Progress:
    """A bar when someone is watching, periodic lines when nobody is.

    Under a scheduler stdout is a log file, and a carriage-returning bar just
    fills it with thousands of partial lines. So the same information is
    emitted either way, in whichever shape suits the destination.
    """

    def __init__(self, total: int, width: int = 32, every: int = 10,
                 unit: str = "file"):
        self.total = total
        self.width = width
        self.every = every            # percent between lines when not a tty
        self.unit = unit
        self.done = 0
        self.t0 = time.perf_counter()
        self.tty = sys.stdout.isatty()
        self._last = -1

    def advance(self, label: str = "") -> None:
        self.done += 1
        elapsed = time.perf_counter() - self.t0
        frac = self.done / self.total if self.total else 1.0
        eta = (elapsed / self.done) * (self.total - self.done) if self.done else 0

        if self.tty:
            filled = int(self.width * frac)
            bar = "#" * filled + "-" * (self.width - filled)
            sys.stdout.write(
                f"\r  [{bar}] {self.done}/{self.total} {frac * 100:3.0f}%  "
                f"{elapsed / self.done:.1f}s/{self.unit}  eta {clock(eta)}   "
            )
            sys.stdout.flush()
        else:
            pct = int(frac * 100)
            if pct // self.every > self._last // self.every or self.done == self.total:
                self._last = pct
                print(f"  {self.done}/{self.total} ({pct}%)  "
                      f"{elapsed / self.done:.1f}s/{self.unit}  eta {clock(eta)}")

    def close(self) -> None:
        if self.tty:
            sys.stdout.write("\r" + " " * (self.width + 60) + "\r")
            sys.stdout.flush()


def clock(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, sec = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{sec:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


refresh()
atexit.register(_report)
