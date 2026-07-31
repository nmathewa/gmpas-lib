"""Where the package reads data from and writes its cache to.

The MCP server this package grew out of kept both inside the repo
(`data/`, `research/mesh_cache/`). A library cannot do that: it is installed
somewhere the user never looks and must not write into its own install tree.
So the cache goes to the usual per-user cache location, and relative paths
resolve against the working directory unless the caller says otherwise.

Both are overridable by environment variable, which is what makes the MCP
server (or any pipeline with its own layout) able to point them back at
project-local directories.
"""

from __future__ import annotations

import os
from pathlib import Path

#: overrides `~/.cache/gmpas/mesh` for cached mesh geometry
CACHE_ENV = "GMPAS_CACHE_DIR"

#: when set, relative paths are tried here before the working directory
DATA_ENV = "GMPAS_DATA_DIR"


def cache_dir() -> Path:
    """Directory holding cached `.npz` mesh geometry.

    Safe to delete at any time — it rebuilds on next use.
    """
    override = os.environ.get(CACHE_ENV)
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base).expanduser() / "gmpas" / "mesh"


def data_dir() -> Path | None:
    """Directory to try first for relative paths, if the user set one."""
    override = os.environ.get(DATA_ENV)
    return Path(override).expanduser() if override else None


def resolve_path(path: str | Path) -> Path:
    """Absolute paths pass through; relative ones try GMPAS_DATA_DIR, then cwd."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    root = data_dir()
    if root is not None:
        candidate = root / p
        if candidate.exists():
            return candidate
    return Path.cwd() / p
