"""The version is written in two places; they must not drift apart.

`gmpas --version` reads `gmpas.__version__`, while the wheel and the sdist take
theirs from `pyproject.toml`. Nothing links the two, so a release can ship a
package whose own CLI reports the previous version -- which is only ever
noticed afterwards.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import gmpas

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _declared_version() -> str:
    try:
        import tomllib
    except ModuleNotFoundError:                      # py3.10
        pytest.skip("tomllib needs Python 3.11+")
    return tomllib.loads(PYPROJECT.read_text())["project"]["version"]


@pytest.mark.skipif(not PYPROJECT.exists(),
                    reason="installed without the source tree")
def test_the_two_version_strings_agree():
    assert gmpas.__version__ == _declared_version()


def test_the_version_is_a_release_number():
    assert re.fullmatch(r"\d+\.\d+\.\d+", gmpas.__version__), gmpas.__version__
