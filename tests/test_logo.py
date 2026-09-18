"""The wordmark, and the two copies of it that must not drift apart.

`docs/logo/make_logo.py` writes the SVGs; the dark one is also inlined into
`viewer.PAGE`, so the page stays a single self-contained file with no extra
request -- which is what lets the static demo serve it too. Two copies means
they can disagree, and the way that shows up is a stale logo on the website
long after the file in the repo was changed.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LOGO = ROOT / "docs" / "logo"


def test_the_generator_reproduces_exactly_what_is_committed(tmp_path):
    """Rerunning it must be a no-op, or the committed art and the script that
    claims to produce it have parted company."""
    before = {f.name: f.read_text() for f in sorted(LOGO.glob("*.svg"))}
    assert before, "no logo files"

    subprocess.run([sys.executable, str(LOGO / "make_logo.py")],
                   check=True, capture_output=True, timeout=120)
    after = {f.name: f.read_text() for f in sorted(LOGO.glob("*.svg"))}

    changed = [n for n in before if before[n] != after.get(n)]
    if changed:                       # put them back before failing
        for name, text in before.items():
            (LOGO / name).write_text(text)
    assert not changed, (
        f"{changed} differ from what make_logo.py produces; rerun it and "
        f"commit the result")


def test_the_page_carries_the_same_wordmark_as_the_repo(tmp_path):
    """The inline copy in PAGE against the file on disk."""
    from gmpas.viewer import PAGE

    committed = (LOGO / "gmpas-dark.svg").read_text().strip()
    # the inline copy is the same SVG with a class added and the fixed
    # width/height dropped so CSS can size it
    body = re.search(r"<svg[^>]*>(.*)</svg>", committed, re.S).group(1)
    assert body in PAGE, (
        "the wordmark inlined in viewer.PAGE is not the one in docs/logo; "
        "rerun docs/logo/make_logo.py and re-inline it")
    assert 'class="logo"' in PAGE


def test_the_page_has_a_favicon_that_needs_no_extra_request(tmp_path):
    """A route for it would have to be reproduced by the demo's shim; a data
    URI just works, in the app and on Pages alike."""
    from gmpas.viewer import PAGE

    assert re.search(r'<link rel="icon" href="data:image/svg\+xml,', PAGE)


def test_text_on_the_accent_stays_legible():
    """The brand blue is much darker than the mint it replaced, and the
    accent is used as a *background* with text on it. Checked here because
    the failure is a contrast regression nobody notices in review."""
    from gmpas.viewer import PAGE

    def luminance(hex_colour: str) -> float:
        parts = [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        parts = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
                 for c in parts]
        return 0.2126 * parts[0] + 0.7152 * parts[1] + 0.0722 * parts[2]

    def ratio(a: str, b: str) -> float:
        hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
        return (hi + 0.05) / (lo + 0.05)

    found = dict(re.findall(r"--(bg|accent|on-accent|brand):(#[0-9a-f]{6})", PAGE))
    assert set(found) >= {"bg", "accent", "on-accent", "brand"}, found

    # text sitting on the accent: WCAG AA for normal text
    assert ratio(found["on-accent"], found["accent"]) >= 4.5
    # the accent itself against the page, as a UI element
    assert ratio(found["accent"], found["bg"]) >= 3.0


@pytest.mark.parametrize("name", ["gmpas.svg", "gmpas-dark.svg",
                                  "mark.svg", "mark-dark.svg"])
def test_every_logo_file_is_self_contained(name):
    """No font references and no external URLs: the letters are outlines, so
    the mark renders the same on a machine with no fonts installed."""
    text = (LOGO / name).read_text()
    assert text.startswith("<svg") and text.rstrip().endswith("</svg>")
    assert "font" not in text.lower()
    assert "<text" not in text and "<image" not in text
    # the xmlns is the SVG namespace, not a fetch; what would actually reach
    # the network is a reference
    outside = re.findall(r'(?:href|src)\s*=\s*"[^"]*//', text)
    outside += re.findall(r"url\(\s*['\"]?https?:", text)
    assert not outside, outside
