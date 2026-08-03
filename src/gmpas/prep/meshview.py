"""Browse a mesh file on its own, with no model output anywhere near it.

`gmpas view` is built on a `Series`: it opens a run, finds the mesh alongside
it, and every route it serves is about a variable at a timestep. That is the
wrong shape for a mesh you have just generated and want to look at, which has
no run and no timesteps at all.

So this is the mesh-only half, and it starts from `MpasMesh.load()` -- the only
thing that has to succeed. The fields it draws are derived from geometry that
the mesh cache already holds, so nothing is computed twice:

    cell_width_km   where the mesh refines (mesh.cell_width_km)
    cell_area_km2   the same information as an area, for sizing work

The scale for each is fixed once from the whole mesh rather than autoscaled per
view, so a refinement band keeps its colour while you pan.

`gmpas view` is untouched: the rendering plumbing here is imported from it --
`ViewIndex`, the palette PNG encoder, the coastline overlay and `bind` -- and
none of it is modified.
"""

from __future__ import annotations

import json
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

import numpy as np

from ..mesh import MpasMesh
from ..viewer import ViewIndex, _overlay, _png, bind, ramp
from .layout import page

#: the one colormap this section uses. Sequential and perceptually uniform,
#: which is what a width field wants; there is nothing here to diverge about.
CMAP = "viridis"

#: fields derivable from mesh geometry alone -- label, and how to get it. Both
#: come off arrays the mesh cache already holds, so neither costs a rebuild.
FIELDS = {
    "cell_width_km": ("cell width (km)", lambda m: m.cell_width_km),
    "cell_area_km2": ("cell area (km²)", lambda m: np.asarray(m.area_cell) / 1.0e6),
}


class MeshViewer:
    """Everything the mesh-only routes need, built once at startup."""

    def __init__(self, mesh_path, nx: int = 1200, ny: int = 700):
        self.mesh = MpasMesh.load(mesh_path)
        self.nx, self.ny = nx, ny
        self.home = self.mesh.extent
        self._views: dict[tuple, ViewIndex] = {}
        self._overlays: dict[tuple, bytes] = {}
        self._values: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    # -- fields ----------------------------------------------------------

    def values(self, field: str) -> np.ndarray:
        """The cell array for one field, computed once and kept."""
        if field not in FIELDS:
            raise KeyError(f"unknown mesh field: {field}, "
                           f"expected one of {', '.join(FIELDS)}")
        if field not in self._values:
            self._values[field] = np.asarray(FIELDS[field][1](self.mesh))
        return self._values[field]

    def describe(self) -> dict:
        fields = []
        for name, (label, _) in FIELDS.items():
            v = self.values(name)
            fields.append({
                "name": name,
                "label": label,
                # fixed, whole-mesh limits: the front end has no vmin/vmax box
                # because there is nothing view-dependent to tune
                "vmin": float(np.nanmin(v)),
                "vmax": float(np.nanmax(v)),
            })
        cells, edges = int(self.mesh.n_cells), int(self.mesh.n_edges)
        coverage = round(self.mesh.coverage * 100, 1)
        regional = not self.mesh.is_global
        width = next(f for f in fields if f["name"] == "cell_width_km")
        return {
            "file": self.mesh.path.name,
            "cells": cells,
            "edges": edges,
            "regional": regional,
            "coverage": coverage,
            # the sidebar is data-driven now, so a step with no cells or edges
            # can reuse the same shell rather than fork it
            "subtitle": f"{cells:,} cells · {'regional' if regional else 'global'}"
                        f" · {coverage}% of its sphere",
            "facts_label": "mesh",
            "stats": [
                ["cells", f"{cells:,}"],
                ["edges", f"{edges:,}"],
                ["coverage", f"{coverage}%"],
                ["width min", f"{width['vmin']:.4g} km"],
                ["width max", f"{width['vmax']:.4g} km"],
            ],
            "home": list(self.home),
            "nx": self.nx,
            "ny": self.ny,
            "cmap": CMAP,
            "ramp": ramp(CMAP),
            "fields": fields,
        }

    # -- frames ----------------------------------------------------------

    def view(self, extent, nx=None, ny=None) -> ViewIndex:
        nx, ny = nx or self.nx, ny or self.ny
        key = (*(round(float(v), 6) for v in extent), nx, ny)
        with self._lock:
            if key not in self._views:
                self._views[key] = ViewIndex(self.mesh, extent, nx, ny)
            return self._views[key]

    def overlay(self, extent, nx=None, ny=None) -> bytes:
        nx, ny = nx or self.nx, ny or self.ny
        key = (*(round(float(v), 6) for v in extent), nx, ny)
        with self._lock:
            if key not in self._overlays:
                self._overlays[key] = _overlay(extent, nx, ny)
            return self._overlays[key]

    def frame(self, field: str, extent, nx=None, ny=None,
              compress: int = 1) -> tuple[bytes, float, float]:
        v = self.values(field)
        img = self.view(extent, nx, ny).frame(v)
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        return _png(img, CMAP, lo, hi, compress), lo, hi


# ------------------------------------------------------------------ serving


def _handler(viewer: MeshViewer, html: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):          # keep the console quiet
            pass

        def _send(self, body: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path == "/":
                    return self._send(html.encode(), "text/html; charset=utf-8")
                if url.path == "/api/meta":
                    return self._send(json.dumps(viewer.describe()).encode(),
                                      "application/json")
                if url.path == "/api/frame":
                    extent = [float(v) for v in q["extent"].split(",")]
                    png, lo, hi = viewer.frame(
                        q.get("field", "cell_width_km"), extent,
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
            except Exception as exc:                      # surface, don't hang
                body = json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                return self.wfile.write(body)
            self.send_error(404)

    return Handler


def serve(mesh_path, port: int = 8765, nx: int = 1200, ny: int = 700,
          open_browser: bool = True, host: str = "127.0.0.1",
          strict_port: bool = False):
    """Start the mesh viewer and block until interrupted.

    Same tunnelling rules as `gmpas view`: loopback is right on a laptop and
    wrong on a compute node, where an SSH tunnel lands on the login node.
    """
    viewer = MeshViewer(mesh_path, nx=nx, ny=ny)
    server = bind(_handler(viewer, page(f"gmpas prep view · {viewer.mesh.path.name}")),
                  port, host=host, strict=strict_port)
    port = server.server_address[1]

    width = viewer.values("cell_width_km")
    kind = "regional" if not viewer.mesh.is_global else "global"
    print(f"{viewer.mesh.n_cells:,} cells, {viewer.mesh.n_edges:,} edges, {kind} — "
          f"cell width {width.min():.3g} to {width.max():.3g} km")

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
