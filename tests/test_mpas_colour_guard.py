"""The MPAS viewer's default colours, pinned.

The MPAS viewer now offers the same palettes and colour options as `--generic`,
but a frame nobody has set an option on must still be the frame it always was.
These hashes were taken before any of that work began; a change here is a
change to what every existing MPAS user sees and needs to be deliberate.

What the page is *offered* did change, by design: the picker lists every
palette and the colour form appears. Those assertions moved to
`test_mpas_colour.py`, which pins the new behaviour.

Decoded pixels and palettes are hashed, not the PNG bytes: compressed bytes
differ between zlib/Pillow builds, which would fail CI without any change to
what anyone sees.
"""

from __future__ import annotations

import hashlib
import io

import numpy as np
import pytest

from gmpas.viewer import CMAPS, _palette, _png, ramp


def _decoded(png: bytes) -> str:
    from PIL import Image

    im = Image.open(io.BytesIO(png))
    h = hashlib.sha256()
    h.update(im.mode.encode())
    h.update(np.asarray(im).tobytes())
    h.update(bytes(im.getpalette()[:768]))
    h.update(str(im.info.get("transparency")).encode())
    return h.hexdigest()


def _field():
    img = np.linspace(-0.5, 1.5, 40 * 30).reshape(30, 40)
    img[3, 5] = np.nan
    return img


def test_the_matplotlib_colormaps_are_unchanged():
    """The picker offers more now, but these ten still mean what they meant:
    a saved figure or a colleague's screenshot must still be reproducible."""
    assert CMAPS == ["viridis", "plasma", "magma", "cividis", "turbo", "RdBu_r",
                     "coolwarm", "BrBG", "Blues", "Spectral_r"]


def test_the_palette_lookup_table_is_unchanged():
    assert hashlib.sha256(bytes(_palette("viridis"))).hexdigest() == \
        "ac9aa2d5507749ec59799cacb5975a9dc5fd0a11fad50ef73e4345c32207559d"


def test_the_colour_ramps_sent_to_the_page_are_unchanged():
    joined = "|".join(",".join(ramp(name)) for name in CMAPS)
    assert hashlib.sha256(joined.encode()).hexdigest() == \
        "beb9a95aab563128fb0be9f80f518a4572d6e6447bfff40f34b958732c8a16d1"


@pytest.mark.parametrize("cmap, lo, hi, compress", [
    ("viridis", 0.0, 1.0, 1),
    ("RdBu_r", -1.0, 2.0, 6),
])
def test_frames_decode_to_the_same_pixels_and_palette(cmap, lo, hi, compress):
    assert _decoded(_png(_field(), cmap, lo, hi, compress)) == GOLDEN[(cmap, compress)]


GOLDEN = {
    ("viridis", 1): "050374c9c4f068e73555905cd08814579a8971cdbcda318cc2bad7feeedbb151",
    ("RdBu_r", 6): "e3e3f5be0e901a7bdd215e90ca3b96db4a21649dcd57e38f8fbe6183601e115e",
}


def test_the_mpas_viewer_still_describes_its_own_shape(tmp_path):
    """The colour keys grew (see test_mpas_colour.py); nothing else did, and
    no plot kinds or layer schema leak onto the MPAS page."""
    from conftest import write_mesh
    from gmpas.viewer import Viewer

    run = tmp_path / "run"
    run.mkdir()
    write_mesh(run / "history.2012-02-25_00.00.00.nc", [(0.0, 0.0), (10.0, 0.0)])
    v = Viewer(run, nx=40, ny=30)
    try:
        meta = v.describe()
    finally:
        v.series.close()
    assert set(meta) == {"file", "mesh", "cells", "regional", "coverage", "files", "steps",
                         "labels", "scanning", "home", "nx", "ny", "cmaps", "ramps",
                         "palettes", "colour_options", "variables"}
    assert "kind_labels" not in meta and "layer_schema" not in meta
    assert meta["cmaps"][:len(CMAPS)] == CMAPS          # matplotlib first, in order
    assert set(meta["variables"][0]) == {"name", "label", "static", "levels", "dim",
                                         "pinned"}
