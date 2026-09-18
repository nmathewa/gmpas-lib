"""What actually ends up in the things people install.

The demo's dataset is 3.7 MB and exists to be served from GitHub Pages, not
to be downloaded by everyone who pip-installs from source. `pyproject.toml`
excludes it -- but an exclude pattern that matches nothing fails silently,
and the first sign is a release already on PyPI. A bare `docs/demo/data`
was exactly that: inert, while looking right.
"""

from __future__ import annotations

import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

#: The sdist is a few hundred KB of source. Anything approaching a megabyte
#: means data has crept back in.
SDIST_BUDGET = 1_500_000


def _build(kind: str, out: Path) -> Path:
    try:
        subprocess.run([sys.executable, "-m", "build", f"--{kind}", "-o", str(out),
                        str(ROOT)], check=True, capture_output=True, timeout=600)
    except FileNotFoundError:                       # pragma: no cover
        pytest.skip("no python -m build")
    except subprocess.CalledProcessError as exc:    # pragma: no cover
        if b"No module named build" in exc.stderr:
            pytest.skip("the build module is not installed")
        raise AssertionError(exc.stderr.decode()[-2000:]) from None
    made = list(out.iterdir())
    assert len(made) == 1, made
    return made[0]


@pytest.mark.slow
def test_the_sdist_leaves_the_demo_data_behind(tmp_path):
    sdist = _build("sdist", tmp_path)
    with tarfile.open(sdist) as tar:
        names = [m.name.split("/", 1)[1] for m in tar.getmembers()]

    heavy = [n for n in names if n.startswith("docs/demo/data")
             or n.startswith("docs/demo/site")]
    assert not heavy, f"the demo's data is in the sdist: {heavy}"
    assert sdist.stat().st_size < SDIST_BUDGET, (
        f"sdist is {sdist.stat().st_size / 1e6:.2f} MB")

    # the scripts stay, so the demo is still readable from a release tarball
    assert "docs/demo/bake.py" in names and "docs/demo/shim.js" in names
    # and the things that must never go missing
    assert sum(1 for n in names if n.startswith("src/gmpas/")) > 40
    assert sum(1 for n in names if n.startswith("tests/")) > 20


@pytest.mark.slow
def test_the_wheel_carries_its_data_files(tmp_path):
    """The vendored Ferret palettes and py.typed are not Python, so they only
    ship if the packaging says so."""
    wheel = _build("wheel", tmp_path)
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()

    assert sum(1 for n in names if n.endswith(".spk")) > 20
    assert any(n.endswith("palettes/ferret/NOTICE.txt") for n in names)
    assert "gmpas/py.typed" in names
    # the demo has no business in the wheel at all
    assert not [n for n in names if "demo" in n]
