"""A local browser viewer for MPAS output -- ncview for an unstructured mesh.

The trick that makes this interactive: **the pixel to cell mapping depends only
on the mesh and the view box**, not on the variable, the timestep or the level.
So the KD-tree is queried once per view (~200 ms on a 400k-cell mesh) and every
frame after that is a gather, `values[idx]`, at a few milliseconds. Changing
variable or scrubbing time is instant; only panning and zooming pay again.

Coastlines are drawn once per view into a transparent overlay and stacked in
the browser, so geographic context costs nothing per frame either.

Served over stdlib http.server on localhost, which means it also works through
an SSH tunnel -- the case where ncview's X11 forwarding hurts most.
"""

from __future__ import annotations

import errno
import io
import json
import threading
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from . import data as _data
from .mesh import MpasMesh
from .raster import target_grid
from .series import Series

#: colormaps offered in the picker, chosen to cover the usual field kinds
CMAPS = ["viridis", "plasma", "magma", "cividis", "turbo",
         "RdBu_r", "coolwarm", "BrBG", "Blues", "Spectral_r"]

#: distinct view boxes to keep a ViewIndex/overlay cached for. Each entry is
#: an nx*ny array pair (~7.5 MB at the 1200x700 default), and nothing evicted
#: this before -- panning and zooming around over a long session, which is
#: exactly what building up several animations for different variables looks
#: like, grew both dicts without bound for the life of the server process.
VIEW_LRU_SIZE = 12


def ramp(name: str, n: int = 32) -> list[str]:
    """Hex stops for a colormap, so the browser's bar matches the image."""
    from matplotlib import colormaps

    cm = colormaps[name]
    return ["#%02x%02x%02x" % tuple(int(round(255 * c)) for c in cm(i / (n - 1))[:3])
            for i in range(n)]


# --------------------------------------------------------------- view index


class ViewIndex:
    """Pixel to cell lookup for one view box, reused across every frame."""

    def __init__(self, mesh: MpasMesh, extent, nx: int, ny: int):
        self.extent, self.nx, self.ny = tuple(extent), nx, ny

        lon, lat = target_grid(self.extent, nx, ny)
        lon2, lat2 = np.meshgrid(lon, lat)
        lon_r, lat_r = np.radians(lon2), np.radians(lat2)
        pts = np.stack([np.cos(lat_r) * np.cos(lon_r),
                        np.cos(lat_r) * np.sin(lon_r),
                        np.sin(lat_r)], axis=-1).reshape(-1, 3)

        radius = np.sqrt(np.asarray(mesh.area_cell) / np.pi) / mesh.sphere_radius
        dist, idx = mesh.tree().query(
            pts, workers=-1, distance_upper_bound=2.0 * float(radius.max())
        )
        missing = idx >= mesh.n_cells
        idx = np.where(missing, 0, idx)

        self.idx = idx
        self.blank = (missing | (dist > 2.0 * radius[idx])).reshape(ny, nx)

    def frame(self, values: np.ndarray) -> np.ndarray:
        """One field, sampled onto this view. A gather, nothing more."""
        img = np.asarray(values, dtype=np.float64)[self.idx].reshape(self.ny, self.nx)
        return np.where(self.blank, np.nan, img)


def _png(img: np.ndarray, cmap: str, vmin: float, vmax: float,
         compress: int = 1) -> bytes:
    """Colour-map an array straight to an 8-bit palette PNG.

    A colormap is a 256-entry lookup table, so the coloured image never holds
    more than 256 distinct colours -- encoding it as 32-bit RGBA means
    compressing four bytes per pixel to say what one byte already says.
    Writing a palette PNG instead is 6.5x faster to encode and ~25% smaller,
    and differs from the RGBA version by at most 3/255 in any channel, which
    comes from quantising to 255 levels so index 255 can mean transparent.

    Going through a matplotlib Figure would cost tens of milliseconds more and
    undo the point of reusing the view index.
    """
    from matplotlib import colormaps
    from PIL import Image

    lut = (np.asarray(colormaps[cmap](np.linspace(0, 1, 255)))[:, :3] * 255)
    lut = lut.round().astype(np.uint8)

    span = (vmax - vmin) or 1.0
    idx = np.clip(np.rint((img - vmin) / span * 254.0), 0, 254)
    idx = np.where(np.isfinite(img), idx, 255).astype(np.uint8)

    im = Image.fromarray(idx[::-1], mode="P")          # imshow origin=lower
    im.putpalette(np.vstack([lut, [[0, 0, 0]]]).ravel().tolist())

    buf = io.BytesIO()
    im.save(buf, format="PNG", transparency=255, compress_level=compress)
    return buf.getvalue()


def _overlay(extent, nx: int, ny: int) -> bytes:
    """Transparent coastline layer for one view box.

    Only coastlines: the graticule is drawn in the browser from the same
    extent, so it toggles without a round trip and its labels stay sharp.
    """
    import matplotlib
    matplotlib.use("Agg")
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt

    lon_min, lon_max, lat_min, lat_max = extent
    central = 0.5 * (lon_min + lon_max) if lon_max > 180.0 else 0.0
    src = ccrs.PlateCarree(central_longitude=central)

    fig = plt.figure(figsize=(nx / 100, ny / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1], projection=src)

    # Limits set directly, NOT through set_extent. set_extent clamps latitude
    # to the projection's valid domain, so a box reaching past a pole -- which
    # every global view does, because frames are rendered 1.4x wider than the
    # window -- came back covering only +-90 but still stretched to fill the
    # figure. The data raster spans the box it was asked for, so the two
    # disagreed by the amount of overshoot, and since that amount changes with
    # zoom the coastlines appeared to slide over the field as you zoomed.
    ax.set_xlim(lon_min - central, lon_max - central)
    ax.set_ylim(lat_min, lat_max)

    # GeoAxes defaults to aspect="equal", which letterboxes the extent inside
    # the figure box and leaves margins. The data raster spans the extent
    # exactly, edge to edge, so any letterboxing shifts the two layers apart
    # and the coastlines sit off the land. Stretch to fill instead, matching
    # the raster's linear lon/lat mapping precisely.
    ax.set_aspect("auto")

    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="#111")
    ax.patch.set_alpha(0)
    ax.spines["geo"].set_visible(False)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, dpi=100)
    plt.close(fig)
    return buf.getvalue()


# ------------------------------------------------------------------- state


class Viewer:
    """Everything the request handlers need, built once at startup."""

    def __init__(self, data_path, mesh_path="", nx=1200, ny=700):
        # serve as soon as the mesh is ready; counting timesteps across
        # every file happens behind the first request
        self.series = Series(data_path, mesh_path, background_scan=True)
        self.mesh = self.series.mesh
        self.nx, self.ny = nx, ny
        self.home = self.mesh.extent
        self._views: OrderedDict[tuple, ViewIndex] = OrderedDict()
        self._overlays: OrderedDict[tuple, bytes] = OrderedDict()
        self._lock = threading.Lock()

    # -- variables -------------------------------------------------------

    def plottable_cell_vars(self) -> list[str]:
        return self.series.variables("nCells")

    def describe(self) -> dict:
        out = []
        for name in self.series.variables("nCells"):
            da = self.series.dataarray(name, 0)
            out.append({
                "name": name,
                "label": _data.field_label(da),
                # A history file carries the mesh alongside the fields, so
                # latCell, edgesOnCell, meshDensity and friends all live on
                # nCells too. Carrying a Time dimension is what separates a
                # model field from mesh furniture.
                "static": "Time" not in da.dims,
                "levels": max((int(da.sizes[d]) for d in da.dims
                               if d.startswith("nVert") or d.startswith("nIso")
                               or d.startswith("nSoil")), default=1),
            })
        out.sort(key=lambda v: (v["static"], v["name"]))
        first = self.series.files[0].name
        title = (first if self.series.n_files == 1
                 else f"{first}  +{self.series.n_files - 1} more")
        return {
            "file": title,
            "mesh": self.mesh.path.name,
            "cells": int(self.mesh.n_cells),
            "regional": not self.mesh.is_global,
            "coverage": round(self.mesh.coverage * 100, 1),
            "files": self.series.n_files,
            "steps": len(self.series),
            "labels": self.series.labels,
            "scanning": self.series.scanning,
            "home": list(self.home),
            "nx": self.nx,
            "ny": self.ny,
            "cmaps": CMAPS,
            "ramps": {name: ramp(name) for name in CMAPS},
            "variables": out,
        }

    # -- frames ----------------------------------------------------------

    def view(self, extent, nx=None, ny=None) -> ViewIndex:
        nx, ny = nx or self.nx, ny or self.ny
        key = (*(round(float(v), 6) for v in extent), nx, ny)
        with self._lock:
            if key in self._views:
                self._views.move_to_end(key)
                return self._views[key]
            index = ViewIndex(self.mesh, extent, nx, ny)
            self._views[key] = index
            if len(self._views) > VIEW_LRU_SIZE:
                self._views.popitem(last=False)
            return index

    def overlay(self, extent, nx=None, ny=None) -> bytes:
        nx, ny = nx or self.nx, ny or self.ny
        key = (*(round(float(v), 6) for v in extent), nx, ny)
        with self._lock:
            if key in self._overlays:
                self._overlays.move_to_end(key)
                return self._overlays[key]
            png = _overlay(extent, nx, ny)
            self._overlays[key] = png
            if len(self._overlays) > VIEW_LRU_SIZE:
                self._overlays.popitem(last=False)
            return png

    def values(self, var: str, time: int, level: int) -> np.ndarray:
        """`time` indexes the whole series, across files, not one file."""
        return self.series.values(var, step=time, level=level)

    def frame(self, var, time, level, extent, cmap, vmin, vmax,
              nx=None, ny=None, compress=1):
        img = self.view(extent, nx, ny).frame(self.values(var, time, level))

        if vmin is not None and vmax is not None:
            lo, hi = vmin, vmax        # animation fixes the range: measure nothing
        else:
            # Estimate the percentiles from a subsample. Scanning every pixel of
            # a 1.5M-pixel frame costs ~8 ms to place two percentiles, and a
            # ninth of the pixels puts them in the same place to well within a
            # colour step.
            sample = img[::3, ::3]
            finite = sample[np.isfinite(sample)]
            if finite.size < 1000:                     # sparse view: be exact
                finite = img[np.isfinite(img)]
            lo = vmin if vmin is not None else (
                float(np.percentile(finite, 2)) if finite.size else 0.0)
            hi = vmax if vmax is not None else (
                float(np.percentile(finite, 98)) if finite.size else 1.0)
        if hi <= lo:
            hi = lo + 1.0
        return _png(img, cmap, lo, hi, compress), lo, hi

    # -- export ----------------------------------------------------------

    def figure(self, var, time, level, extent, cmap, vmin, vmax, style="paper"):
        """A publication-shaped figure, not the bare raster the browser shows.

        Goes through the ordinary plotting path so it gets cartopy axes,
        coastlines, a labelled colorbar and a title -- the things a screenshot
        of the viewer does not give you.
        """
        import matplotlib
        matplotlib.use("Agg")

        from .plot import cell_field
        from .style import Style

        da = self.series.dataarray(var, time)
        values = self.values(var, time, level)
        label = _data.field_label(da)
        title = f"{var} — {self.series.labels[time]}"
        if int(da.sizes.get("nVertLevels", 1)) > 1:
            title += f"  (level {level})"

        fig, _ = cell_field(self.mesh, values, style=Style.preset(style),
                            cmap=cmap or "viridis", vmin=vmin, vmax=vmax,
                            extent=tuple(extent), label=label, title=title)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=Style.preset(style).dpi)
        import matplotlib.pyplot as plt
        plt.close(fig)
        return buf.getvalue()

    def gif(self, var, level, extent, cmap, vmin, vmax, nx, ny, fps=8):
        """Every timestep as one animated GIF.

        Frames are already palette images, which is exactly what GIF wants, so
        this is a re-container rather than a re-encode.
        """
        from PIL import Image

        frames = []
        for step in range(len(self.series)):
            png, _, _ = self.frame(var, step, level, extent, cmap,
                                   vmin, vmax, nx, ny, compress=1)
            frames.append(Image.open(io.BytesIO(png)).convert("P"))

        buf = io.BytesIO()
        frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:],
                       duration=max(20, int(1000 / max(fps, 1))), loop=0,
                       disposal=2, transparency=255)
        return buf.getvalue()

    def netcdf(self, var, time, level, extent, nx, ny):
        """The current view sampled onto a regular lat-lon grid, as netCDF.

        NEAREST-CELL SAMPLING, not a conservative remap: every point takes the
        value of the cell containing it. Cell integrals are NOT preserved, so
        this is for inspection and downstream plotting, not for budgets.
        """
        import xarray as xr

        from .raster import target_grid

        view = self.view(extent, nx, ny)
        img = view.frame(self.values(var, time, level)).astype(np.float32)
        lon, lat = target_grid(tuple(extent), view.nx, view.ny)
        da = self.series.dataarray(var, time)

        ds = xr.Dataset(
            {var: (("lat", "lon"), img, dict(da.attrs))},
            coords={"lat": ("lat", lat, {"units": "degrees_north"}),
                    "lon": ("lon", lon, {"units": "degrees_east"})},
            attrs={
                "title": f"{var} sampled from an MPAS native mesh",
                "source_mesh": self.mesh.path.name,
                "mesh_cells": int(self.mesh.n_cells),
                "valid_time": self.series.labels[time],
                "method": "nearest cell centre (Voronoi containment); "
                          "NOT area-conservative",
                "longitude_convention":
                    "continuous across the antimeridian: values may exceed "
                    "180 degrees east so a dateline-crossing domain stays "
                    "contiguous",
                "history": "written by gmpas",
            },
        )
        buf = io.BytesIO()
        buf.write(ds.to_netcdf())
        return buf.getvalue()

    def probe(self, lon, lat, var, time, level):
        cell = int(self.mesh.cell_of(np.array([lon]), np.array([lat]))[0])
        value = float(self.values(var, time, level)[cell])
        return {"cell": cell, "value": value,
                "lon": round(float(self.mesh.lon_cell[cell]), 4),
                "lat": round(float(self.mesh.lat_cell[cell]), 4)}


# ----------------------------------------------------------------- serving


class PageHandler(BaseHTTPRequestHandler):
    """What every gmpas page handler has in common.

    Shared rather than copied because the dashboard mounts these handlers
    behind a prefix by calling one's `do_GET` with another's `self`. That only
    works if `_send` is part of the contract between them, so it lives here
    instead of being repeated identically in each.
    """

    def log_message(self, *a):              # keep the console quiet
        pass

    def _send(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def _handler(viewer: Viewer, html: str = ""):
    """Routes for one run. `html` overrides the page, which is how the
    dashboard splices its source switcher in without forking `PAGE`."""
    html = html or PAGE

    class Handler(PageHandler):
        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == "/":
                    return self._send(html.encode(), "text/html; charset=utf-8")
                if url.path == "/api/status":
                    # cheap: how the time axis stands while the scan runs
                    return self._send(json.dumps({
                        "scanning": viewer.series.scanning,
                        "steps": len(viewer.series),
                        "labels": viewer.series.labels,
                    }).encode(), "application/json")
                if url.path == "/api/meta":
                    return self._send(json.dumps(viewer.describe()).encode(),
                                      "application/json")
                if url.path == "/api/frame":
                    extent = [float(v) for v in q["extent"].split(",")]
                    png, lo, hi = viewer.frame(
                        q["var"], int(q.get("time", 0)), int(q.get("level", 0)),
                        extent, q.get("cmap", "viridis"),
                        float(q["vmin"]) if q.get("vmin") else None,
                        float(q["vmax"]) if q.get("vmax") else None,
                        int(q["nx"]) if q.get("nx") else None,
                        int(q["ny"]) if q.get("ny") else None,
                        int(q.get("compress", 1)),
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("X-Range", f"{lo},{hi}")
                    self.send_header("Content-Length", str(len(png)))
                    self.end_headers()
                    return self.wfile.write(png)
                if url.path == "/api/overlay":
                    extent = [float(v) for v in q["extent"].split(",")]
                    return self._send(viewer.overlay(
                        extent,
                        int(q["nx"]) if q.get("nx") else None,
                        int(q["ny"]) if q.get("ny") else None), "image/png")
                if url.path.startswith("/api/export/"):
                    kind = url.path.rsplit("/", 1)[-1]
                    extent = [float(v) for v in q["extent"].split(",")]
                    var, lvl = q["var"], int(q.get("level", 0))
                    step = int(q.get("time", 0))
                    vmin = float(q["vmin"]) if q.get("vmin") else None
                    vmax = float(q["vmax"]) if q.get("vmax") else None
                    stem = f"{var}_{viewer.series.labels[step]}".replace(" ", "_") \
                                                                .replace(":", "")
                    if kind == "figure":
                        body, ctype, name = (
                            viewer.figure(var, step, lvl, extent,
                                          q.get("cmap", "viridis"), vmin, vmax,
                                          q.get("style", "paper")),
                            "image/png", f"{stem}.png")
                    elif kind == "gif":
                        body, ctype, name = (
                            viewer.gif(var, lvl, extent, q.get("cmap", "viridis"),
                                       vmin, vmax,
                                       int(q["nx"]) if q.get("nx") else None,
                                       int(q["ny"]) if q.get("ny") else None,
                                       int(q.get("fps", 8))),
                            "image/gif", f"{var}_animation.gif")
                    elif kind == "netcdf":
                        body, ctype, name = (
                            viewer.netcdf(var, step, lvl, extent,
                                          int(q["nx"]) if q.get("nx") else None,
                                          int(q["ny"]) if q.get("ny") else None),
                            "application/x-netcdf", f"{stem}.nc")
                    else:
                        return self.send_error(404)
                    self.send_response(200)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Content-Disposition",
                                     f'attachment; filename="{name}"')
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    return self.wfile.write(body)
                if url.path == "/api/probe":
                    out = viewer.probe(float(q["lon"]), float(q["lat"]), q["var"],
                                       int(q.get("time", 0)), int(q.get("level", 0)))
                    return self._send(json.dumps(out).encode(), "application/json")
            except Exception as exc:                      # surface, don't hang
                body = json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                return self.wfile.write(body)
            self.send_error(404)

    return Handler


#: consecutive ports to try before asking the OS to pick one
PORT_ATTEMPTS = 20


def bind(handler, port: int, host: str = "127.0.0.1",
         attempts: int = PORT_ATTEMPTS,
         strict: bool = False) -> ThreadingHTTPServer:
    """Bind to `port`, or -- unless `strict` -- the next free one after it.

    `strict` matters for tunnelling. Somebody who passed an explicit port did
    so because an SSH tunnel is already pointing at it, and quietly listening
    somewhere else leaves them staring at a browser that will never load. So
    an explicit port fails loudly, and only the default wanders.
    """
    last = None
    for candidate in range(port, port + (1 if strict else attempts)):
        try:
            return ThreadingHTTPServer((host, candidate), handler)
        except OSError as exc:
            if exc.errno not in (errno.EADDRINUSE, errno.EACCES):
                raise
            last = exc
    if strict:
        raise OSError(
            last.errno,
            f"cannot bind {host}:{port} — {last.strerror}. "
            f"Pick a free port, or drop --port to let one be chosen."
        )
    return ThreadingHTTPServer((host, 0), handler)      # 0 = any free port


def serve(data_path, mesh_path="", port=8765, nx=1200, ny=700, open_browser=True,
          host="127.0.0.1", strict_port=False):
    """Start the viewer and block until interrupted.

    `host` defaults to loopback, which is right on a laptop and wrong on a
    compute node: an SSH tunnel from your machine lands on the *login* node,
    so a viewer listening only on the compute node's loopback is unreachable.
    Pass host="0.0.0.0" there, exactly as one does for Jupyter.
    """
    import socket

    viewer = Viewer(data_path, mesh_path, nx=nx, ny=ny)
    server = bind(_handler(viewer), port, host=host, strict=strict_port)
    port = server.server_address[1]

    cells = viewer.mesh.n_cells
    kind = "regional" if not viewer.mesh.is_global else "global"
    print(f"{cells:,} cells, {kind} — {len(viewer.series)} steps across "
          f"{viewer.series.n_files} file(s), "
          f"{len(viewer.plottable_cell_vars())} plottable fields")

    node = socket.gethostname()
    if host in ("127.0.0.1", "localhost"):
        print(f"listening on 127.0.0.1:{port} — this machine only")
        print(f"  open  http://127.0.0.1:{port}")
        print(f"  if {node} is a remote node, this is NOT reachable through a "
              f"tunnel to a login node; restart with --host 0.0.0.0")
    else:
        print(f"listening on {host}:{port} — reachable as {node}:{port}")
        print(f"  from your machine:  ssh -N -L {port}:{node}:{port} <login-node>")
        print(f"  then open           http://localhost:{port}")
    print("ctrl-c to stop")

    if open_browser:
        threading.Timer(0.5, webbrowser.open,
                        args=(f"http://127.0.0.1:{port}",)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
        viewer.series.close()


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>gmpas view</title>
<style>
:root{--bg:#16181c;--panel:#1e2127;--line:#2c313a;--fg:#e6e8ec;--dim:#9aa3b0;--accent:#5dcaa5}
*{box-sizing:border-box}
body{margin:0;font:13px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
     background:var(--bg);color:var(--fg);display:flex;height:100vh;overflow:hidden}
#side{width:250px;flex:none;background:var(--panel);border-right:1px solid var(--line);
      display:flex;flex-direction:column;overflow:hidden}
#side h1{font-size:13px;font-weight:500;margin:0;padding:12px 14px;border-bottom:1px solid var(--line)}
#side h1 small{display:block;color:var(--dim);font-weight:400;margin-top:2px}
.sec{padding:10px 14px;border-bottom:1px solid var(--line);flex:none}
.sec label{display:block;color:var(--dim);font-size:11px;letter-spacing:.04em;
           text-transform:uppercase;margin-bottom:6px}
select,input[type=text]{width:100%;background:#14161a;color:var(--fg);
  border:1px solid var(--line);border-radius:4px;padding:5px 6px;font:inherit}
input[type=range]{width:100%;accent-color:var(--accent)}
#vars{flex:1;overflow-y:auto;padding:6px 0;min-height:60px}
#vars div{padding:4px 14px;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#vars div:hover{background:#252932}
#vars div.on{background:var(--accent);color:#08201a}
#vars div.static{color:var(--dim);font-style:italic}
#main{flex:1;display:flex;flex-direction:column;min-width:0}
#right{width:220px;flex:none;background:var(--panel);border-left:1px solid var(--line);
       overflow-y:auto}
#right h1{font-size:13px;font-weight:500;margin:0;padding:12px 14px;
          border-bottom:1px solid var(--line)}
#right .row{display:flex;gap:6px}
#right .kv{display:flex;justify-content:space-between;color:var(--dim);font-size:11px;
           margin-bottom:4px}
#right .kv b{color:var(--fg);font-weight:500;font-variant-numeric:tabular-nums}
.hint{color:var(--dim);font-size:11px;margin-top:6px;line-height:1.45;
      font-variant-numeric:tabular-nums;word-break:break-word}
#top{padding:8px 14px;border-bottom:1px solid var(--line);display:flex;gap:14px;
     align-items:center;color:var(--dim);flex-wrap:wrap}
#top b{color:var(--fg);font-weight:500;white-space:nowrap}
#tlab{font-variant-numeric:tabular-nums}
#stage{flex:1;display:flex;align-items:center;justify-content:center;position:relative;
       padding:12px;min-height:0;overflow:hidden}
#frame{display:grid;grid-template-columns:auto auto;grid-template-rows:auto auto}
#latax{position:relative;width:52px}
#lonax{position:relative;height:18px}
#corner{width:52px;height:18px}
#latax span,#lonax span{position:absolute;color:var(--dim);font-size:11px;
  font-variant-numeric:tabular-nums;white-space:nowrap}
#latax span{right:6px;transform:translateY(-50%)}
#lonax span{top:2px;transform:translateX(-50%)}
#grat{position:absolute;inset:0;pointer-events:none}
#grat i{position:absolute;background:#fff;opacity:.28}
#grat i.v{top:0;bottom:0;width:1px}
#grat i.h{left:0;right:0;height:1px}
#wrap{position:relative;line-height:0;box-shadow:0 0 0 1px var(--line);overflow:hidden;
      cursor:grab}
#wrap.drag{cursor:grabbing}
#wrap img{display:block;width:100%;height:100%;transform-origin:0 0;will-change:transform}
#over{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
#scale{position:absolute;left:14px;bottom:14px;color:#fff;font-size:11px;
       text-shadow:0 0 3px #000,0 0 6px #000;pointer-events:none}
#scalebar{height:5px;border:1px solid #fff;border-top:none;box-shadow:0 0 3px #000;
          margin-top:2px}
#scaletext{display:block;font-variant-numeric:tabular-nums}
#msg{position:absolute;top:14px;left:50%;transform:translateX(-50%);background:#000a;
     padding:4px 10px;border-radius:4px;color:var(--dim);opacity:0;transition:opacity .2s;
     pointer-events:none}
#bar{padding:8px 14px;border-top:1px solid var(--line)}
#ramp{height:14px;border-radius:2px;border:1px solid var(--line)}
#ticks{display:flex;justify-content:space-between;margin-top:3px;color:var(--dim);
       font-size:11px;font-variant-numeric:tabular-nums}
#cblabel{color:var(--dim);font-size:11px;margin-bottom:4px}
button{background:#252932;color:var(--fg);border:1px solid var(--line);border-radius:4px;
       padding:5px 9px;font:inherit;cursor:pointer}
button:hover{border-color:var(--accent)}
button:disabled{opacity:.5;cursor:default;border-color:var(--line)}
button.on{background:var(--accent);color:#08201a;border-color:var(--accent)}
#animstate{font-variant-numeric:tabular-nums}
#fps{width:80px}
</style></head><body>
<div id="side">
  <h1><span id="title">loading…</span><small id="sub"></small></h1>
  <div class="sec"><label style="margin:0"><input type="checkbox" id="showstatic"
    style="width:auto;vertical-align:-1px"> show mesh &amp; static arrays</label></div>
  <div id="vars"></div>
</div>
<div id="main">
  <div id="top">
    <span>time <b id="tlab">–</b></span><input type="range" id="time" min="0" max="0" style="width:150px">
    <span>level <b id="llab">0</b></span><input type="range" id="level" min="0" max="0" style="width:110px">
    <span>zoom</span><input type="range" id="zoom" min="0" max="800" value="0" style="width:110px">
    <label style="white-space:nowrap"><input type="checkbox" id="grid" checked
      style="vertical-align:-1px"> grid</label>
    <button id="home">reset view</button>
    <button id="anim">▶ play</button>
    <span id="animstate"></span>
  </div>
  <div id="stage">
    <div id="frame">
      <div id="latax"></div>
      <div id="wrap">
        <img id="data"><img id="over">
        <div id="grat"></div>
        <div id="scale"><span id="scaletext"></span><div id="scalebar"></div></div>
      </div>
      <div id="corner"></div>
      <div id="lonax"></div>
    </div>
    <div id="msg"></div>
  </div>
  <div id="bar">
    <div id="cblabel"></div>
    <div id="ramp"></div>
    <div id="ticks"></div>
  </div>
</div>
<div id="right">
  <h1>options</h1>

  <div class="sec"><label>colormap</label><select id="cmap"></select></div>

  <div class="sec"><label>colour range</label>
    <div class="row">
      <input type="text" id="vmin" placeholder="auto"><input type="text" id="vmax" placeholder="auto">
    </div>
    <div class="row" style="margin-top:6px">
      <button id="reset">auto</button><button id="lockrange">lock to view</button>
    </div>
    <div class="hint" id="rangehint"></div>
  </div>

  <div class="sec"><label>extent</label>
    <div class="row"><input type="text" id="elon0" placeholder="lon min"><input type="text" id="elon1" placeholder="lon max"></div>
    <div class="row" style="margin-top:6px"><input type="text" id="elat0" placeholder="lat min"><input type="text" id="elat1" placeholder="lat max"></div>
    <div class="row" style="margin-top:6px"><button id="applyext">apply</button><button id="copyext">copy</button></div>
    <div class="hint" id="exthint"></div>
  </div>

  <div class="sec"><label>animation</label>
    <div class="kv"><span>frames / second</span><b id="fpslab">8</b></div>
    <input type="range" id="fps" min="1" max="24" value="8">
    <div class="kv" style="margin-top:8px"><span>quality</span><b id="qlab">fast</b></div>
    <input type="range" id="quality" min="1" max="9" value="1"
           title="PNG compression: lower is faster to build, higher is smaller to transfer">
    <div class="row" style="margin-top:8px"><button id="clearanim">clear cached frames</button></div>
    <div class="hint" id="animhint"></div>
  </div>

  <div class="sec"><label>export</label>
    <select id="figstyle" title="figure size and dpi">
      <option value="paper">paper &middot; 10x6 @130</option>
      <option value="notebook">notebook &middot; 9x5 @100</option>
      <option value="poster">poster &middot; 16x9 @200</option>
    </select>
    <div class="row" style="margin-top:6px">
      <button id="expfig">figure</button><button id="expnc">data</button>
    </div>
    <div class="row" style="margin-top:6px">
      <button id="expgif" style="flex:1">animation (GIF)</button>
    </div>
    <div class="hint" id="exphint">figure is a full plot with axes and
      colorbar; data is this view on a lat-lon grid, sampled nearest-cell
      and <b>not</b> area-conservative</div>
  </div>

  <div class="sec"><label>probe</label>
    <div class="hint" id="probe2">click the map</div>
  </div>
</div>
<script>
const $=s=>document.querySelector(s);
let M=null, cur=null, busy=false, pend=false;
let view=null, home=null, rendered=null;      // geographic state
const ZMAX=800;                               // slider units, 100 per doubling

function say(t){const m=$("#msg");m.textContent=t;m.style.opacity=t?1:0;}
function aspect(){ return M.nx/M.ny; }

// a view is a centre plus a longitude span; latitude span follows the image
// aspect so nothing is ever stretched
function boxOf(v){
  const h=v.w/aspect();
  return [v.clon-v.w/2, v.clon+v.w/2, v.clat-h/2, v.clat+h/2];
}
function fit(box){
  const [a,b,c,d]=box, clon=(a+b)/2, clat=(c+d)/2;
  return {clon, clat, w: Math.max(b-a, (d-c)*aspect())};
}
function clamp(){
  view.w = Math.min(view.w, home.w);                 // never wider than the mesh
  view.w = Math.max(view.w, home.w/Math.pow(2,ZMAX/100));
  const h=view.w/aspect(), hh=home.w/aspect();
  const dx=Math.max(0,(home.w-view.w)/2), dy=Math.max(0,(hh-h)/2);
  view.clon=Math.min(home.clon+dx, Math.max(home.clon-dx, view.clon));
  view.clat=Math.min(home.clat+dy, Math.max(home.clat-dy, view.clat));
  $("#zoom").value = Math.round(Math.log2(home.w/view.w)*100);
}

async function boot(){
  M = await (await fetch("api/meta")).json();
  $("#title").textContent = M.file;
  M.cmaps.forEach(c=>{const o=document.createElement("option");o.textContent=c;$("#cmap").append(o)});
  home = fit(M.home); view = {...home}; rendered = null;
  layout();
  subtitle(); fillVars();
  if(M.scanning) setTimeout(pollScan, 400);
  animUI(); panel();
  pick(M.variables.find(v=>!v.static)?.name ?? M.variables[0].name);
}
function subtitle(){
  $("#sub").textContent = `${M.cells.toLocaleString()} cells \u00b7 `+
    `${M.regional?"regional":"global"} \u00b7 ${M.steps} step${M.steps>1?"s":""}`+
    ` in ${M.files} file${M.files>1?"s":""}${M.scanning?" \u00b7 scanning\u2026":""}`;
}
function fillVars(){
  const show=$("#showstatic").checked;
  $("#vars").innerHTML="";
  M.variables.filter(v=>show||!v.static).forEach(v=>{
    const d=document.createElement("div");d.textContent=v.name;d.title=v.label;
    if(v.static) d.classList.add("static");
    d.onclick=()=>pick(v.name);$("#vars").append(d);
  });
  if(cur) [...$("#vars").children].forEach(d=>d.classList.toggle("on",d.textContent===cur.name));
}
$("#showstatic").addEventListener("change", fillVars);

function pick(name){
  animStop();
  cur = M.variables.find(v=>v.name===name);
  [...$("#vars").children].forEach(d=>d.classList.toggle("on",d.textContent===name));
  $("#time").max=M.steps-1; $("#tlab").textContent=M.labels[$("#time").value|0];
  $("#level").max=cur.levels-1; $("#level").value=0; $("#llab").textContent=0;
  overlay(); draw();
}
async function pollScan(){
  if(!M||!M.scanning) return;
  const s=await (await fetch("api/status")).json();
  const keep=$("#time").value;
  M.steps=s.steps; M.labels=s.labels; M.scanning=s.scanning;
  $("#time").max=M.steps-1; $("#time").value=keep;
  subtitle();
  if(M.scanning) setTimeout(pollScan, 700);
}

async function overlay(){
  const b=outset(boxOf(view));
  $("#over").src=`api/overlay?extent=${b.join(",")}`+
                 `&nx=${Math.round(M.nx*OUTSET)}&ny=${Math.round(M.ny*OUTSET)}`;
}

// show the frame we already have, transformed into place, until the real one
// lands. Without this a zoom looks like nothing happens and then it jumps.
function preview(){
  if(!rendered){ return; }
  const b=boxOf(view), s=(rendered[1]-rendered[0])/(b[1]-b[0]);
  const fx=(b[0]-rendered[0])/(rendered[1]-rendered[0]);
  const fy=(rendered[3]-b[3])/(rendered[3]-rendered[2]);
  const t=`translate(${-fx*s*100}%, ${-fy*s*100}%) scale(${s})`;
  $("#data").style.transform=t; $("#over").style.transform=t;
}
function scalebar(){
  const b=boxOf(view), wkm=(b[1]-b[0])*111.320*Math.cos(view.clat*Math.PI/180);
  const px=$("#wrap").clientWidth||600;
  let target=wkm*0.25, mag=Math.pow(10,Math.floor(Math.log10(target)));
  const nice=[1,2,5,10].map(n=>n*mag).filter(n=>n<=wkm*0.45);
  const len=nice.length?nice[nice.length-1]:target;
  $("#scalebar").style.width=(len/wkm*px)+"px";
  $("#scaletext").textContent = len>=1 ? `${len.toLocaleString()} km`
                                       : `${Math.round(len*1000).toLocaleString()} m`;
}
// Fit the image into whatever the stage has left, preserving the raster's
// aspect. Letting CSS decide meant an unconstrained height, so a wide view
// spilled over the colorbar; and the graticule needs exact pixel dimensions
// anyway.
const GUTTER_X=52, GUTTER_Y=18;
// Frames are rendered wider than the window. Without a margin, dragging slid
// the image off its own edge and exposed blank background until the redraw
// landed -- which read as the pan being broken. 1.4x costs ~2x the pixels and
// buys a screen-width of slack in every direction.
const OUTSET=1.4;
function outset(b, f=OUTSET){
  const cx=(b[0]+b[1])/2, cy=(b[2]+b[3])/2, w=(b[1]-b[0])*f/2, h=(b[3]-b[2])*f/2;
  return [cx-w, cx+w, cy-h, cy+h];
}
function covers(outer, inner){
  return outer && outer[0]<=inner[0]+1e-9 && outer[1]>=inner[1]-1e-9
                && outer[2]<=inner[2]+1e-9 && outer[3]>=inner[3]-1e-9;
}
function layout(){
  if(!M) return;
  const st=$("#stage"), cs=getComputedStyle(st), pad=parseFloat(cs.padding)||0;
  const availW=st.clientWidth-2*pad-GUTTER_X, availH=st.clientHeight-2*pad-GUTTER_Y;
  const a=aspect();
  let w=availW, h=w/a;
  if(h>availH){ h=availH; w=h*a; }
  const wrap=$("#wrap");
  wrap.style.width=Math.max(80,Math.floor(w))+"px";
  wrap.style.height=Math.max(60,Math.floor(h))+"px";
}

const STEPS=[0.05,0.1,0.2,0.25,0.5,1,2,2.5,5,10,15,20,30,45,60];
function niceStep(span, want){
  for(const s of STEPS) if(span/s <= want) return s;
  return STEPS[STEPS.length-1];
}
function fmtLon(v){
  let x=((v+180)%360+360)%360-180;                 // a view can run past +180
  const r=Math.round(x*100)/100;
  if(r===0) return "0\u00b0";
  if(Math.abs(r)===180) return "180\u00b0";        // the antimeridian is neither
  return `${Math.abs(r)}\u00b0${r>0?"E":"W"}`;
}
function fmtLat(v){
  const r=Math.round(v*100)/100;
  return r===0?"0\u00b0":`${Math.abs(r)}\u00b0${r>0?"N":"S"}`;
}
function graticule(){
  const g=$("#grat"), la=$("#latax"), lo=$("#lonax");
  g.innerHTML=""; la.innerHTML=""; lo.innerHTML="";
  if(!$("#grid").checked) return;
  const b=boxOf(view);
  const w=$("#wrap").clientWidth, h=$("#wrap").clientHeight;
  la.style.height=h+"px"; lo.style.width=w+"px";

  const sx=niceStep(b[1]-b[0], 7), sy=niceStep(b[3]-b[2], 6);
  for(let v=Math.ceil(b[0]/sx)*sx; v<=b[1]+1e-9; v+=sx){
    const f=(v-b[0])/(b[1]-b[0]);
    g.insertAdjacentHTML("beforeend",`<i class="v" style="left:${f*100}%"></i>`);
    lo.insertAdjacentHTML("beforeend",`<span style="left:${f*w}px">${fmtLon(v)}</span>`);
  }
  for(let v=Math.ceil(b[2]/sy)*sy; v<=b[3]+1e-9; v+=sy){
    const f=(b[3]-v)/(b[3]-b[2]);
    g.insertAdjacentHTML("beforeend",`<i class="h" style="top:${f*100}%"></i>`);
    la.insertAdjacentHTML("beforeend",`<span style="top:${f*h}px">${fmtLat(v)}</span>`);
  }
}

function colorbar(lo,hi){
  const stops=M.ramps[$("#cmap").value];
  $("#ramp").style.background=`linear-gradient(90deg,${stops.join(",")})`;
  const n=5, out=[];
  for(let i=0;i<n;i++){
    const v=lo+(hi-lo)*i/(n-1);
    out.push(`<span>${Math.abs(v)>=1e4||(v!==0&&Math.abs(v)<1e-3)?v.toExponential(2):v.toPrecision(4)}</span>`);
  }
  $("#ticks").innerHTML=out.join("");
  $("#cblabel").textContent=cur?cur.label:"";
}

async function draw(){
  if(!cur) return;
  if(busy){ pend=true; return; }
  busy=true;
  // everything below runs under try/finally: a fetch that rejects -- a
  // superseded request, a reload mid-flight, a server hiccup -- used to
  // escape with busy still set, and the viewer then never drew again. It
  // looked like zoom and pan had broken, because the preview transform kept
  // updating over a frame that could no longer be replaced.
  try{
  const b=outset(boxOf(view));
  const p=new URLSearchParams({var:cur.name,time:$("#time").value,
    level:$("#level").value,extent:b.join(","),cmap:$("#cmap").value,
    nx:Math.round(M.nx*OUTSET), ny:Math.round(M.ny*OUTSET)});
  if($("#vmin").value) p.set("vmin",$("#vmin").value);
  if($("#vmax").value) p.set("vmax",$("#vmax").value);
  const t0=performance.now();
  const r=await fetch("api/frame?"+p);
  if(!r.ok){ say((await r.json()).error); return; }
  const [lo,hi]=r.headers.get("X-Range").split(",").map(Number);
  const url=URL.createObjectURL(await r.blob());
  const img=$("#data"), old=img.src;
  img.onload=()=>{ if(old.startsWith("blob:")) URL.revokeObjectURL(old); };
  img.src=url;
  rendered=b;
  preview();          // the frame is larger than the window: crop to the view
  lastRange=[lo,hi];
  colorbar(lo,hi); scalebar(); graticule(); animUI(); panel();
  say(`${cur.label} \u00b7 ${Math.round(performance.now()-t0)} ms`);
  }catch(e){
    say("render failed: "+e);
  }finally{
    busy=false;
    if(pend){ pend=false; draw(); }      // always drains, however we left
  }
}
// Every frame of a run, held for the session. Keyed by the whole render
// configuration, so panning or changing variable starts a new set while an
// earlier one stays instantly replayable.
const animCache=new Map();
const anim={loading:false, playing:false, timer:null, cancel:false, urls:null,
            ready:0, total:0, firstReady:null};
let lastRange=null;

function animKey(){
  const b=boxOf(view).map(v=>+v.toFixed(4));
  return JSON.stringify([cur&&cur.name, $("#level").value, b,
                         $("#cmap").value, $("#vmin").value, $("#vmax").value]);
}
function animUI(){
  if($("#animhint")) { const f=[...animCache.values()].reduce((n,u)=>n+u.length,0);
    $("#animhint").textContent=f?`${f} frames held in ${animCache.size} set${animCache.size>1?"s":""}`:""; }
  const k=animKey(), have=animCache.has(k);
  const btn=$("#anim");
  // disabled only in the brief gap before frame 0 has rendered -- once
  // anything is ready, play can start (and keep extending as more streams
  // in), and pausing/resuming a still-loading run must stay clickable
  btn.disabled = anim.loading && !anim.playing && anim.ready===0;
  btn.textContent = anim.playing ? "\u23f8 pause"
                  : anim.loading ? "loading\u2026"
                  : "\u25b6 play";
  btn.classList.toggle("on", anim.playing);
  if(!anim.loading && !anim.playing)
    $("#animstate").textContent = have ? `${animCache.get(k).length} frames ready` : "";
}

async function animLoad(){
  const k=animKey(), n=M.steps;
  let resolveFirst;
  anim.firstReady=new Promise(res=>{resolveFirst=res});
  anim.ready=0; anim.total=n;

  if(n<2){ say("only one timestep"); resolveFirst(); return null; }

  // Lock the colour range across the run: autoscaling every frame separately
  // makes the sequence flicker and stops frames being comparable.
  let lo=lastRange&&lastRange[0], hi=lastRange&&lastRange[1];
  if(lo===undefined||lo===null){ await draw(); [lo,hi]=lastRange||[0,1]; }

  anim.loading=true; anim.cancel=false; animUI();
  const urls=new Array(n), b=outset(boxOf(view));
  anim.urls=urls;
  const nx=Math.round(M.nx*OUTSET), ny=Math.round(M.ny*OUTSET);
  const t0=performance.now();
  try{
    for(let i=0;i<n;i++){
      if(anim.cancel){ urls.slice(0,i).forEach(u=>u&&URL.revokeObjectURL(u)); resolveFirst(); return null; }
      const p=new URLSearchParams({var:cur.name, time:i, level:$("#level").value,
        extent:b.join(","), cmap:$("#cmap").value, nx, ny, vmin:lo, vmax:hi,
        compress:$("#quality").value});
      const r=await fetch("api/frame?"+p);
      if(!r.ok) throw new Error((await r.json()).error);
      urls[i]=URL.createObjectURL(await r.blob());
      await new Promise(res=>{ const im=new Image(); im.onload=im.onerror=res; im.src=urls[i]; });
      // frame 0 is the earliest point playback can usefully start -- resolve
      // as soon as it lands rather than waiting for the whole run
      anim.ready=i+1;
      if(i===0) resolveFirst();
      if(!anim.playing) $("#animstate").textContent=`loading ${anim.ready} / ${n}`;
    }
  }catch(e){
    say("animation failed: "+e);
    urls.filter(Boolean).forEach(u=>u&&URL.revokeObjectURL(u));
    anim.loading=false; anim.ready=0; resolveFirst(); animUI(); return null;
  }
  animCache.set(k, urls);
  anim.loading=false;
  if(!anim.playing)
    $("#animstate").textContent=`${n} frames in ${((performance.now()-t0)/1000).toFixed(1)}s`;
  animUI();
  return urls;
}

function animStop(){
  anim.playing=false; clearInterval(anim.timer); anim.timer=null; animUI();
}
function animPlay(urls){
  anim.urls=urls; anim.playing=true; animUI();
  const tick=()=>{
    if(!anim.playing) return;
    // urls is pre-sized to the full run and filled in as frames arrive, so
    // its own .length is not how many are actually usable yet -- loop over
    // what anim.ready says is loaded so far while a stream is still filling,
    // and only then over the whole thing
    const ready = anim.loading ? anim.ready : urls.length;
    if(ready<1) return;
    let i=(+$("#time").value+1)%ready;
    $("#time").value=i; $("#tlab").textContent=M.labels[i];
    $("#data").src=urls[i];
    $("#animstate").textContent = anim.loading
      ? `playing \u00b7 ${i+1}/${ready} rendered (of ${anim.total})`
      : `${i+1} / ${urls.length}`;
  };
  clearInterval(anim.timer);
  anim.timer=setInterval(tick, 1000/(+$("#fps").value||8));
}

function panel(){
  const b=boxOf(view);
  $("#elon0").placeholder=b[0].toFixed(2); $("#elon1").placeholder=b[1].toFixed(2);
  $("#elat0").placeholder=b[2].toFixed(2); $("#elat1").placeholder=b[3].toFixed(2);
  $("#exthint").textContent=`${b[0].toFixed(2)}, ${b[1].toFixed(2)}, `+
                            `${b[2].toFixed(2)}, ${b[3].toFixed(2)}`;
  $("#rangehint").textContent = lastRange
    ? ($("#vmin").value||$("#vmax").value ? "manual" : `auto ${lastRange[0].toPrecision(4)} .. ${lastRange[1].toPrecision(4)}`)
    : "";
  const frames=[...animCache.values()].reduce((n,u)=>n+u.length,0);
  $("#animhint").textContent = frames
    ? `${frames} frames held in ${animCache.size} set${animCache.size>1?"s":""}` : "";
  $("#fpslab").textContent=$("#fps").value;
  const q=+$("#quality").value;
  $("#qlab").textContent = q<=2?"fast" : q<=5?"balanced" : "small";
}
$("#fps").addEventListener("input", panel);
$("#quality").addEventListener("input", panel);
$("#applyext").onclick = ()=>{
  const g=(id,d)=>{ const v=parseFloat($(id).value); return isFinite(v)?v:d; };
  const b=boxOf(view);
  const box=[g("#elon0",b[0]), g("#elon1",b[1]), g("#elat0",b[2]), g("#elat1",b[3])];
  if(box[1]<=box[0]||box[3]<=box[2]){ say("extent must be min then max"); return; }
  animStop(); view=fit(box); clamp(); overlay(); draw(); panel();
};
$("#copyext").onclick = ()=>{
  const b=boxOf(view).map(v=>v.toFixed(4));
  navigator.clipboard?.writeText(b.join(", "));
  say("extent copied");
};
$("#lockrange").onclick = ()=>{
  if(!lastRange) return;
  $("#vmin").value=lastRange[0].toPrecision(6);
  $("#vmax").value=lastRange[1].toPrecision(6);
  animStop(); draw(); panel();
};
$("#clearanim").onclick = ()=>{
  animStop();
  for(const urls of animCache.values()) urls.forEach(URL.revokeObjectURL);
  animCache.clear(); animUI(); panel(); say("cached frames released");
};

async function exportAs(kind, label){
  const btns=[...document.querySelectorAll("#expfig,#expnc,#expgif")];
  btns.forEach(b=>b.disabled=true);
  $("#exphint").textContent=`building ${label}\u2026`;
  const b=kind==="figure" ? boxOf(view) : outset(boxOf(view));
  const p=new URLSearchParams({var:cur.name, time:$("#time").value,
    level:$("#level").value, extent:b.join(","), cmap:$("#cmap").value,
    style:$("#figstyle").value, fps:$("#fps").value,
    nx:Math.round(M.nx*OUTSET), ny:Math.round(M.ny*OUTSET)});
  if($("#vmin").value) p.set("vmin",$("#vmin").value);
  if($("#vmax").value) p.set("vmax",$("#vmax").value);
  // GIF must have one range for the whole run, or every frame rescales
  if(kind==="gif" && !$("#vmin").value && lastRange){
    p.set("vmin",lastRange[0]); p.set("vmax",lastRange[1]);
  }
  const t0=performance.now();
  try{
    const r=await fetch(`api/export/${kind}?`+p);
    if(!r.ok) throw new Error((await r.json()).error);
    const blob=await r.blob();
    const name=(r.headers.get("Content-Disposition")||"").match(/filename="(.+?)"/);
    const a=document.createElement("a");
    a.href=URL.createObjectURL(blob);
    a.download=name?name[1]:`gmpas.${kind}`;
    a.click(); setTimeout(()=>URL.revokeObjectURL(a.href), 10000);
    $("#exphint").textContent=`${a.download} \u00b7 ${(blob.size/1048576).toFixed(1)} MB `+
                              `\u00b7 ${((performance.now()-t0)/1000).toFixed(1)}s`;
  }catch(e){ $("#exphint").textContent="export failed: "+e; }
  finally{ btns.forEach(b=>b.disabled=false); }
}
$("#expfig").onclick = ()=>exportAs("figure","figure");
$("#expnc").onclick  = ()=>exportAs("netcdf","netCDF");
$("#expgif").onclick = ()=>{ animStop(); exportAs("gif","animation"); };

$("#anim").onclick = async ()=>{
  if(anim.playing){ animStop(); return; }
  const k=animKey();
  const cached=animCache.get(k);
  if(cached){ animPlay(cached); return; }
  if(anim.loading){                    // a stream is already filling in
    if(anim.ready>0) animPlay(anim.urls);
    return;
  }
  animLoad();                          // don't await -- let it stream in the background
  await anim.firstReady;               // resolves once frame 0 has rendered (or on failure)
  if(anim.ready>0 && !anim.playing) animPlay(anim.urls);
};
$("#fps").oninput = ()=>{ if(anim.playing) animPlay(anim.urls); };

let redrawTimer=null;
function schedule(ms){ animStop(); preview(); scalebar(); graticule(); clearTimeout(redrawTimer);
  redrawTimer=setTimeout(()=>{ overlay(); draw(); }, ms); }

$("#time").oninput = e=>{ $("#tlab").textContent=M.labels[e.target.value];
  if(anim.playing) return;                       // scrubbing during playback
  const urls=animCache.get(animKey());
  if(urls){ $("#data").src=urls[e.target.value]; return; }   // instant if loaded
  draw(); };
$("#level").oninput = e=>{ $("#llab").textContent=e.target.value; animStop(); draw(); };
$("#cmap").onchange = ()=>{ animStop(); draw(); };
$("#vmin").onchange = draw; $("#vmax").onchange = draw;
$("#reset").onclick = ()=>{ $("#vmin").value=""; $("#vmax").value=""; draw(); };
$("#home").onclick = ()=>{ view={...home}; clamp(); schedule(0); };
$("#zoom").oninput = e=>{ view.w = home.w/Math.pow(2, e.target.value/100);
                          clamp(); schedule(180); };

$("#wrap").onwheel = ev=>{
  ev.preventDefault();
  const r=$("#wrap").getBoundingClientRect();   // untransformed reference
  const fx=(ev.clientX-r.left)/r.width, fy=(ev.clientY-r.top)/r.height;
  const b=boxOf(view);
  const lon=b[0]+fx*(b[1]-b[0]), lat=b[3]-fy*(b[3]-b[2]);
  const k=Math.exp(ev.deltaY*0.0015);        // continuous, not stepped
  const nw=Math.min(home.w, Math.max(home.w/Math.pow(2,ZMAX/100), view.w*k));
  const s=nw/view.w;
  view.clon = lon + (view.clon-lon)*s;       // keep the cursor point fixed
  view.clat = lat + (view.clat-lat)*s;
  view.w = nw;
  clamp(); schedule(160);
};

let drag=null;
$("#wrap").onpointerdown = ev=>{
  drag={x:ev.clientX, y:ev.clientY, clon:view.clon, clat:view.clat, moved:false};
  try{ $("#wrap").setPointerCapture(ev.pointerId); }catch(e){}   // may refuse
  $("#wrap").classList.add("drag");
};
$("#wrap").onpointermove = ev=>{
  if(!drag) return;
  const r=$("#wrap").getBoundingClientRect(), b=boxOf(view);
  const dx=(ev.clientX-drag.x)/r.width*(b[1]-b[0]);
  const dy=(ev.clientY-drag.y)/r.height*(b[3]-b[2]);
  if(Math.abs(ev.clientX-drag.x)+Math.abs(ev.clientY-drag.y)>3) drag.moved=true;
  view.clon=drag.clon-dx; view.clat=drag.clat+dy;
  clamp(); preview(); scalebar(); graticule();
  if(!covers(rendered, boxOf(view))) schedule(90);   // ran past the margin
};
$("#wrap").onpointerup = async ev=>{
  const moved=drag&&drag.moved; drag=null;
  $("#wrap").classList.remove("drag");
  if(moved){ schedule(0); return; }
  if(!cur) return;
  const r=$("#wrap").getBoundingClientRect(), b=boxOf(view);
  const lon=b[0]+((ev.clientX-r.left)/r.width)*(b[1]-b[0]);
  const lat=b[3]-((ev.clientY-r.top)/r.height)*(b[3]-b[2]);
  const q=new URLSearchParams({lon,lat,var:cur.name,
    time:$("#time").value,level:$("#level").value});
  const d=await (await fetch("api/probe?"+q)).json();
  $("#probe2").innerHTML=`cell ${d.cell}<br>${d.lat}\u00b0, ${d.lon}\u00b0<br>`+
                         `<b>${d.value.toPrecision(6)}</b>`;
};
$("#grid").onchange = graticule;
addEventListener("resize", ()=>{ layout(); scalebar(); graticule(); });
boot();
</script></body></html>
"""
