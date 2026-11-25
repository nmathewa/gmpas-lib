"""Places by name, as boxes: "United States", "Bay of Bengal", "Nino 3.4".

The gazetteer, `places.json`, is built from Natural Earth (public domain) by
a script kept outside the package; its header says how. Each name holds one
or more places, each a box [west, east, south, north] in degrees and a label
saying what it is. A box across 180 degrees runs past 180 (Fiji is 176.6 to
183); `fit` moves a box into a file's own longitudes.

A name with more than one place -- Georgia the country and the US state -- is
not guessed between: `lookup` returns them all and the caller asks, or is
told to write one of the labels, which are names too.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

#: Names too common as words to be taken for places inside a sentence; they
#: still work written as a region.
_WORDS = {"us", "chad", "turkey", "jordan", "niger", "guinea", "georgia", "victoria",
          "washington", "plain", "coast", "delta", "basin", "valley", "range"}


@lru_cache(maxsize=1)
def _places() -> dict[str, list]:
    doc = json.loads(Path(__file__).with_name("places.json").read_text())
    places = doc["places"]
    # a label is a name too: "georgia (us state)" picks the one meant
    labels: dict[str, list] = {}
    for found in places.values():
        for place in found:
            labels.setdefault(place[4].lower(), [place])
    return {**labels, **places}


def _key(name: str) -> str:
    key = re.sub(r"\s+", " ", name).strip().lower()
    return re.sub(r"^the ", "", key)


def lookup(name: str) -> list[tuple[list[float], str]]:
    """The places called `name`, as ([west, east, south, north], label),
    the most likely first; empty when there is none."""
    return [(p[:4], p[4]) for p in _places().get(_key(name), [])]


def find(text: str, at_most: int = 3) -> list[tuple[str, list]]:
    """The places named in a sentence, in the order they are named:
    [(name, lookup)]. Longer names are matched first, so "New Britain" is not
    taken for Britain; a name that is also a common word counts only written
    as a name (Georgia, US)."""
    words = re.findall(r"[\w'‘’.+-]+", text)
    lower = [w.lower() for w in words]
    hits: list[tuple[str, list]] = []
    taken: set[int] = set()
    for size in range(6, 0, -1):
        for i in range(len(words) - size + 1):
            if taken & set(range(i, i + size)) or lower[i] == "the":
                continue                     # "the" is not part of a name
            name = " ".join(lower[i:i + size]).strip(".")
            first = words[i]
            # a name that is also a word counts when written as a name: Georgia,
            # Turkey; the short ones only in capitals: US, UK
            if size == 1 and (name in _WORDS or len(name) < 3) and not (
                    first.isupper() if len(name) < 3 or name == "us"
                    else first[:1].isupper()):
                continue
            found = lookup(name)
            if found:
                hits.append((name, found))
                taken |= set(range(i, i + size))
    order = {name: text.lower().find(name) for name, _ in hits}
    return sorted(hits, key=lambda h: order[h[0]])[:at_most]


def fit(box, home) -> list[float] | None:
    """`box` moved by whole turns of 360 degrees into the longitudes of a
    file covering `home`, or None when no turn makes them overlap."""
    w, e, s, n = box
    hw, he, hs, hn = home
    if n < hs or s > hn:
        return None
    best = None
    for turn in (0.0, 360.0, -360.0, 720.0, -720.0):
        lo, hi = w + turn, e + turn
        shared = min(hi, he) - max(lo, hw)
        if shared > 0 and (best is None or shared > best[0]):
            best = (shared, [lo, hi, s, n])
    return best[1] if best else None
