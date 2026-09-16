"""Palettes and colour options on the MPAS viewer.

The same colours `--generic` has had since #113, on the native-mesh path. What
is different here is the raster underneath: pixels with no cell under them are
off the mesh and must stay transparent, whatever colour missing data is given.
"""

from __future__ import annotations

import io
import json
import threading
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pytest

from gmpas.palettes import encode
from gmpas.viewer import CMAPS, PAGE, Viewer, _handler, _png, bind


@pytest.fixture
def viewer(tmp_path):
    """A four-cell regional mesh: any wide view has pixels off the mesh."""
    from conftest import write_mesh

    run = tmp_path / "run"
    run.mkdir()
    write_mesh(run / "history.2012-02-25_00.00.00.nc",
               [(0.0, 0.0), (10.0, 0.0), (5.0, 8.0), (-6.0, 4.0)])
    v = Viewer(run, nx=80, ny=60)
    yield v
    v.series.close()


def _decode(png):
    from PIL import Image

    im = Image.open(io.BytesIO(png))
    return im, np.asarray(im)[::-1]


def _serve(v):
    srv = bind(_handler(v, PAGE), 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


WIDE = (-30.0, 40.0, -20.0, 30.0)


# ------------------------------------------------------------ what is offered


def test_the_mpas_page_is_offered_every_palette_and_the_colour_options(viewer):
    meta = viewer.describe()
    assert {"viridis", "ferret.rnb2", "grads.rainbow"} <= set(meta["cmaps"])
    assert set(meta["ramps"]) == set(meta["cmaps"])
    assert set(meta["palettes"]) == {"matplotlib", "cmocean", "ferret", "grads"}
    assert meta["palettes"]["matplotlib"] == list(CMAPS)
    assert {"bands", "reverse", "under_color", "over_color", "missing_color"} <= \
        set(meta["colour_options"])


def test_the_generic_and_mpas_pages_are_offered_the_same_colours(viewer, tmp_path):
    """One table, two viewers: a palette added for one cannot miss the other."""
    import pandas as pd
    import xarray as xr

    from gmpas.generic import GenericViewer

    lat, lon = np.linspace(0, 10, 5), np.linspace(0, 10, 5)
    xr.Dataset({"t": (("time", "lat", "lon"), np.zeros((1, 5, 5)))},
               coords={"time": pd.date_range("2024", periods=1), "lat": lat, "lon": lon}
               ).to_netcdf(tmp_path / "g.nc")
    gv = GenericViewer(tmp_path / "g.nc")
    try:
        generic, mpas = gv.describe(), viewer.describe()
    finally:
        gv.close()
    for key in ("cmaps", "ramps", "palettes", "colour_options"):
        assert generic[key] == mpas[key], key


# -------------------------------------------------------------- the fast map


def test_without_colour_options_a_frame_is_the_old_encoder_byte_for_byte(viewer):
    png, lo, hi = viewer.frame("areaCell", 0, 0, WIDE, "viridis", None, None, 80, 60)
    img = viewer.view(WIDE, 80, 60).frame(viewer.values("areaCell", 0, 0))
    assert png == _png(img, "viridis", lo, hi, 1)


def test_a_palette_from_another_tool_draws(viewer):
    png, _, _ = viewer.frame("latCell", 0, 0, WIDE, "ferret.rnb2", None, None, 80, 60)
    assert png[:4] == b"\x89PNG"


def test_bands_use_the_reserved_indices_and_describe_their_bar(viewer):
    meta = {}
    png, _, _ = viewer.frame("latCell", 0, 0, WIDE, "grads.rainbow", 0.0, 8.0, 80, 60,
                             colour={"bands": 4, "extend": "both",
                                     "under_color": "black", "over_color": "white"},
                             meta=meta)
    im, idx = _decode(png)
    assert im.info["transparency"] == encode.CLEAR
    assert set(np.unique(idx)) <= set(range(4)) | {encode.UNDER, encode.OVER,
                                                   encode.MISSING, encode.CLEAR}
    spec = meta["colorbar"]
    assert len(spec["stops"]) == 4
    assert spec["under"] == "#000000" and spec["over"] == "#ffffff"


def test_off_mesh_pixels_stay_transparent_even_with_a_missing_colour(viewer):
    """The case the mesh path has and the regular grid does not: most of a wide
    view has no cell under it at all, and a colour for missing data is not a
    licence to paint the ocean around the mesh."""
    png, _, _ = viewer.frame("areaCell", 0, 0, WIDE, "viridis", None, None, 80, 60,
                             colour={"missing_color": "red"})
    _, idx = _decode(png)
    assert idx[0, 0] == encode.CLEAR                     # a corner, far off the mesh
    assert (idx == encode.CLEAR).any() and (idx < encode.DATA).any()
    assert not (idx == encode.MISSING).any()             # no NaN cells in this field


def test_a_nan_cell_takes_the_missing_colour_while_off_mesh_does_not(viewer,
                                                                     monkeypatch):
    values = viewer.values("areaCell", 0, 0).astype(float).copy()
    values[0] = np.nan
    monkeypatch.setattr(viewer, "values", lambda *a, **k: values)
    png, _, _ = viewer.frame("areaCell", 0, 0, WIDE, "viridis", None, None, 80, 60,
                             colour={"missing_color": "red"})
    im, idx = _decode(png)
    palette = np.asarray(im.getpalette()[:768], np.uint8).reshape(256, 3)
    assert (idx == encode.MISSING).any()
    assert tuple(palette[encode.MISSING]) == (255, 0, 0)
    assert idx[0, 0] == encode.CLEAR


def test_bad_colour_options_are_refused_by_name(viewer):
    with pytest.raises(ValueError, match="bands and norm=log"):
        viewer.frame("areaCell", 0, 0, WIDE, "viridis", None, None, 80, 60,
                     colour='{"bands": 4, "norm": "log"}')


# ------------------------------------------------------------------ the wire


def test_the_frame_route_sends_the_colorbar_only_with_colour_options(viewer):
    srv, base = _serve(viewer)
    q = {"var": "latCell", "extent": ",".join(str(v) for v in WIDE),
         "cmap": "grads.rainbow", "nx": 60, "ny": 40}
    try:
        with urllib.request.urlopen(f"{base}/api/frame?{urllib.parse.urlencode(q)}") as r:
            plain = r.read()
            assert r.headers.get("X-Colorbar") is None
        colour = json.dumps({"bands": 5, "extend": "max", "over_color": "white"})
        url = f"{base}/api/frame?{urllib.parse.urlencode({**q, 'colour': colour})}"
        with urllib.request.urlopen(url) as r:
            spec = json.loads(r.headers["X-Colorbar"])
            assert r.read() != plain
        assert len(spec["stops"]) == 5 and spec["over"] == "#ffffff"
        bad = urllib.parse.urlencode({**q, "colour": '{"gamma": 3}'})
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(f"{base}/api/frame?{bad}")
        assert "gamma applies to norm=power" in json.loads(err.value.read())["error"]
    finally:
        srv.shutdown()


def test_an_export_carries_the_colour_options_over_the_wire(viewer):
    """_plot_extras forwards `colour` as soon as the page offers the options, so
    every export method has to take it -- a TypeError here is a 500 in the
    browser."""
    srv, base = _serve(viewer)
    q = {"var": "latCell", "extent": ",".join(str(v) for v in WIDE),
         "cmap": "grads.rainbow", "nx": 40, "ny": 30,
         "colour": json.dumps({"bands": 4})}
    try:
        for kind, magic in (("figure", b"\x89PNG"), ("gif", b"GIF"),
                            ("netcdf", None)):
            url = f"{base}/api/export/{kind}?{urllib.parse.urlencode(q)}"
            with urllib.request.urlopen(url) as r:
                body = r.read()
            assert body and (magic is None or body.startswith(magic)), kind
    finally:
        srv.shutdown()


# ---------------------------------------------------------------- the exports


def test_a_figure_export_draws_the_bands_it_was_asked_for(viewer):
    import matplotlib
    matplotlib.use("Agg")

    png = viewer.figure("latCell", 0, 0, WIDE, "grads.rainbow", None, None, "notebook",
                        colour={"bands": 5, "extend": "both"})
    assert png[:4] == b"\x89PNG"


def test_a_figure_export_without_options_is_what_it_always_was(viewer):
    import matplotlib
    matplotlib.use("Agg")

    assert viewer.figure("latCell", 0, 0, WIDE, "viridis", None, None, "notebook") \
        == viewer.figure("latCell", 0, 0, WIDE, "viridis", None, None, "notebook",
                         colour=None)


def test_a_gif_export_carries_the_colour_options(viewer):
    """Off-mesh pixels must still be transparent in the GIF. Pillow renumbers
    the palette of a single-frame GIF, so this asks the pixels, not the index."""
    from PIL import Image

    gif = viewer.gif("latCell", 0, WIDE, "grads.rainbow", 0.0, 8.0, 60, 40,
                     colour={"bands": 4})
    im = Image.open(io.BytesIO(gif))
    assert im.n_frames == len(viewer.series)
    assert im.convert("RGBA").load()[0, 0][3] == 0        # a corner, off the mesh


# ------------------------------------------------------------- the prep pages


def test_the_prep_pages_still_draw_their_own_single_colormap(tmp_path):
    """meshview and hfunview share _png with a fixed colormap and no picker."""
    from conftest import write_mesh
    from gmpas.prep.meshview import MeshViewer

    mesh = tmp_path / "mesh.nc"
    write_mesh(mesh, [(0.0, 0.0), (10.0, 0.0), (5.0, 8.0), (-6.0, 4.0)])
    mv = MeshViewer(mesh, nx=40, ny=30)
    meta = mv.describe()
    assert meta["cmap"] == "viridis" and len(meta["ramp"]) == 32
    assert "palettes" not in meta and "colour_options" not in meta
    png, lo, hi = mv.frame(meta["fields"][0]["name"], mv.home, 40, 30, 1)
    assert png[:4] == b"\x89PNG" and hi >= lo
