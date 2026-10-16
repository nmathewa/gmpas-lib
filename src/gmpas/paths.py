"""Where the package reads data from and writes its cache to.

The MCP server this package grew out of kept both inside the repo
(`data/`, `research/mesh_cache/`). A library cannot do that: it is installed
somewhere the user never looks and must not write into its own install tree.
So the cache goes to the usual per-user cache location, overridable by
environment variable, and relative paths resolve against the working
directory.
"""

from __future__ import annotations

import os
from pathlib import Path

#: overrides `~/.cache/gmpas/mesh` for cached mesh geometry
CACHE_ENV = "GMPAS_CACHE_DIR"

def cache_dir() -> Path:
    """Directory holding cached mesh geometry (one `.npy` array per file).

    Safe to delete at any time — it rebuilds on next use.
    """
    override = os.environ.get(CACHE_ENV)
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base).expanduser() / "gmpas" / "mesh"


def resolve_path(path: str | Path) -> Path:
    """Absolute paths pass through; relative ones resolve against the cwd."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return Path.cwd() / p
