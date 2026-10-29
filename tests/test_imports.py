"""What importing gmpas costs, and what it still provides (#129).

`gmpas/__init__` loads its public names on first use, so a process that only
needs part of the package -- above all a reader worker, which needs netCDF4
and nothing else -- does not pay for xarray and pandas. Each check runs in a
fresh interpreter: this one has long since imported everything.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def _run(code: str) -> str:
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True)
    return out.stdout.strip()


HEAVY = "print(sorted(m for m in ('xarray', 'pandas', 'matplotlib') if m in sys.modules))"


@pytest.mark.parametrize("module", ["gmpas", "gmpas._reader", "gmpas.cli"])
def test_light_imports_load_no_xarray_or_pandas(module):
    """`gmpas` for scripts, `_reader` for the workers, `cli` because the
    workers re-import the `gmpas` command's main module."""
    assert _run(f"import sys, {module}; {HEAVY}") == "[]"


@pytest.mark.parametrize("first", ["gmpas", "xarray"])
def test_the_accessor_is_registered_in_either_import_order(first):
    """`import gmpas` before `import xarray` is what isort writes."""
    second = "xarray" if first == "gmpas" else "gmpas"
    got = _run(f"import {first}, {second}, xarray as xr; "
               "print(type(xr.Dataset().mpas).__name__)")
    assert got == "MpasAccessor"


def test_every_public_name_resolves_to_its_module():
    import gmpas

    for name in gmpas.__all__:
        value = getattr(gmpas, name)
        if name != "__version__":
            home = sys.modules[f"gmpas.{gmpas._HOME[name]}"]
            assert value is getattr(home, name), name
    assert set(gmpas.__all__) <= set(dir(gmpas))
    assert _run("from gmpas import *; print(callable(open_mpas), RASTER_THRESHOLD > 0)") \
        == "True True"


def test_submodules_and_unknown_names_after_a_bare_import():
    assert _run("import gmpas; print(gmpas.series.Series.__name__)") == "Series"
    with pytest.raises(AttributeError, match="no attribute 'nope'"):
        import gmpas
        gmpas.nope  # noqa: B018


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
def test_a_reader_worker_does_not_load_pandas():
    """The pool's real worker, started the pool's own way. pandas has
    compiled extensions, so its being loaded shows in the process's maps."""
    import os

    from gmpas.series import _warm_worker, series_pool

    pool = series_pool(1)
    try:
        assert pool.submit(_warm_worker, 0).result(timeout=60)
        pid = pool.submit(os.getpid).result(timeout=60)
        with open(f"/proc/{pid}/maps") as f:
            maps = f.read()
    finally:
        pool.shutdown()
    assert "netCDF4" in maps
    assert "pandas/_libs" not in maps
