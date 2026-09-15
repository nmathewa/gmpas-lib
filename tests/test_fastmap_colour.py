"""The --generic fast map's palettes and colour options, end to end."""

from __future__ import annotations

import io
import json
import threading
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from gmpas.generic import GenericViewer
from gmpas.palettes import encode
from gmpas.viewer import PAGE, _handler, _png, bind


@pytest.fixture
def regional(tmp_path):
    """A regional grid (so part of any wide view is off the grid) with a NaN."""
    lat, lon = np.linspace(30, 60, 16), np.linspace(-20, 40, 31)
    la, _ = np.meshgrid(lat, lon, indexing="ij")
    t = np.stack([240 + (60 - la), 245 + (60 - la)])
    t[:, 8, 15] = np.nan
    xr.Dataset({"t": (("time", "lat", "lon"), t, {"units": "K"})},
               coords={"time": pd.date_range("2024", periods=2), "lat": lat, "lon": lon}
               ).to_netcdf(tmp_path / "r.nc")
    gv = GenericViewer(tmp_path / "r.nc")
    yield gv
    gv.close()


def _decode(png):
    from PIL import Image

    im = Image.open(io.BytesIO(png))
    return im, np.asarray(im)[::-1]


def test_the_page_is_offered_every_palette_with_a_ramp_and_the_colour_options(regional):
    meta = regional.describe()
    names = [n for group in meta["palettes"].values() for n in group]
    assert {"viridis", "ferret.rnb2", "grads.rainbow"} <= set(names)
    assert set(meta["ramps"]) == set(names) == set(meta["cmaps"])
    assert {"bands", "reverse", "under_color", "over_color", "missing_color"} <= \
        set(meta["colour_options"])


def test_without_colour_options_a_frame_is_the_old_encoder_byte_for_byte(regional):
    extent = (-40, 60, 20, 70)
    png, lo, hi = regional.frame("t", 0, 0, extent, "ferret.rnb2", None, None, 100, 50)
    img = regional._raster("t", 0, 0, extent, 100, 50)
    assert png == _png(img, "ferret.rnb2", lo, hi, 1)


def test_off_grid_stays_clear_while_missing_cells_take_their_colour(regional):
    meta = {}
    png, _, _ = regional.frame("t", 0, 0, (-40, 60, 20, 70), "viridis", 245.0, 265.0,
                               100, 50, colour={"missing_color": "red", "bands": 4},
                               meta=meta)
    im, idx = _decode(png)
    assert im.info["transparency"] == encode.CLEAR
    assert idx[0, 0] == encode.CLEAR                        # south-west: beyond the grid
    assert (idx == encode.MISSING).any()                    # the NaN cell
    assert set(np.unique(idx)) <= set(range(4)) | {encode.UNDER, encode.OVER,
                                                   encode.MISSING, encode.CLEAR}
    assert len(meta["colorbar"]["stops"]) == 4


def test_bad_colour_options_are_refused_by_name(regional):
    with pytest.raises(ValueError, match="bands and norm=log"):
        regional.frame("t", 0, 0, (-20, 40, 30, 60), "viridis", None, None, 40, 20,
                       colour='{"bands": 4, "norm": "log"}')


def test_figures_and_gifs_take_the_colour_options(regional):
    from PIL import Image

    colour = {"bands": 5, "under_color": "black", "extend": "min"}
    png = regional.figure("t", 0, 0, (-20, 40, 30, 60), "grads.rainbow", 245.0, 280.0,
                          "notebook", kind="map", colour=colour)
    assert png[:4] == b"\x89PNG"
    gif = regional.gif("t", 0, (-40, 60, 20, 70), "grads.rainbow", 245.0, 280.0, 60, 30,
                       colour=colour)
    assert Image.open(io.BytesIO(gif)).n_frames == 2


def test_a_figure_gif_keeps_its_colour_options_frame_by_frame(regional, monkeypatch):
    """The figure path carries hov, layers and colour; none may take another's
    place, and a dropped colour would silently draw the plain viridis scale."""
    from gmpas import palettes

    seen = []
    scale = palettes.scale
    monkeypatch.setattr(palettes, "scale", lambda opts, lo, hi: (
        seen.append(opts), scale(opts, lo, hi))[1])
    regional.gif("t", 0, (-20, 40, 30, 60), "grads.rainbow", 245.0, 280.0, kind="contourf",
                 colour={"bands": 5, "reverse": True})
    assert len(seen) == 2                                    # one per step
    assert all(o["cmap"] == "grads.rainbow" and o["reverse"] for o in seen)


def _serve(viewer):
    srv = bind(_handler(viewer, PAGE), 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def test_the_frame_route_sends_the_colorbar_only_with_colour_options(regional):
    srv, base = _serve(regional)
    q = {"var": "t", "extent": "-40,60,20,70", "cmap": "viridis",
         "nx": 60, "ny": 30}
    try:
        with urllib.request.urlopen(f"{base}/api/frame?{urllib.parse.urlencode(q)}") as r:
            assert r.headers.get("X-Colorbar") is None
        colour = json.dumps({"bands": 6, "extend": "both", "over_color": "white"})
        url = f"{base}/api/frame?{urllib.parse.urlencode({**q, 'colour': colour})}"
        with urllib.request.urlopen(url) as r:
            spec = json.loads(r.headers["X-Colorbar"])
        assert len(spec["stops"]) == 6 and spec["over"] == "#ffffff"
        bad = urllib.parse.urlencode({**q, "colour": '{"gamma": 3}'})
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(f"{base}/api/frame?{bad}")
        assert "gamma applies to norm=power" in json.loads(err.value.read())["error"]
    finally:
        srv.shutdown()


def test_the_mpas_frame_route_ignores_colour_options(tmp_path):
    from conftest import write_mesh
    from gmpas.viewer import Viewer

    run = tmp_path / "run"
    run.mkdir()
    write_mesh(run / "history.2012-02-25_00.00.00.nc", [(0.0, 0.0), (10.0, 0.0)])
    viewer = Viewer(run, nx=40, ny=30)
    srv, base = _serve(viewer)
    q = {"var": "areaCell", "extent": "-5,15,-5,5", "cmap": "viridis", "nx": 40, "ny": 20}
    try:
        with urllib.request.urlopen(f"{base}/api/frame?{urllib.parse.urlencode(q)}") as r:
            plain = r.read()
        coloured = urllib.parse.urlencode({**q, "colour": '{"bands": 4}'})
        with urllib.request.urlopen(f"{base}/api/frame?{coloured}") as r:
            assert r.headers.get("X-Colorbar") is None
            assert r.read() == plain
    finally:
        srv.shutdown()
        viewer.series.close()
