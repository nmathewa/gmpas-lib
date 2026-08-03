"""Browse a JIGSAW distance function, with no mesh anywhere near it.

`gmpas prep view` needs a mesh file; this needs only the `hfun.py` that would
produce one. It answers the question you actually have while writing that file
-- where does it refine, how fast does it get there, and is the transition
gentle enough -- without spending the generation time first.

The fields are the two quantities the workflow itself derives from `hfun.py`,
neither of them invented here:

    cell_width_km   get_hfun, straight out                     (create_hfun.py)
    mesh_density    (hfun_min / h)**4, MPAS's meshDensity      (create_density.py)

Rendering is the mesh viewer's, imported and not modified. There is no mesh, so
there is no KD-tree and no `ViewIndex`: the distance function is defined at
every point, and a frame is one call to `get_hfun` on the pixel centres. That
call may be expensive by design -- the contract says it is made once with whole
arrays -- so frames are computed per view box and kept.
"""

from __future__ import annotations

import json
import socket
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

import numpy as np

from ..raster import target_grid
from ..viewer import _overlay, _png, bind, ramp
from .hfun import GRADIENT_GUIDELINE, Hfun, diagnose
from .layout import page

#: sequential and perceptually uniform, and the same way round as
#: `prep view`'s cell_width_km: small grid distance dark, large light. A
#: refinement region then reads as the dark blob it is on the tutorial's own
#: HFUN plots, and the two viewers agree on what "fine" looks like.
CMAP = "viridis"

FIELDS = {
    "cell_width_km": "grid distance (km)",
    "mesh_density": "meshDensity",
}


class HfunViewer:
    """Everything the distance-function routes need, built once at startup."""

    def __init__(self, hfun_path, nx: int = 1200, ny: int = 700):
        self.hfun = Hfun.load(hfun_path)
        self.nx, self.ny = nx, ny
        self.diagnosis = diagnose(self.hfun)

        # a distance function is defined over the whole sphere, so there is no
        # domain to fit to -- the home view is the world
        self.home = (-180.0, 180.0, -90.0, 90.0)
        self._frames: dict[tuple, np.ndarray] = {}
        self._overlays: dict[tuple, bytes] = {}
        self._lock = threading.Lock()

    # -- fields ----------------------------------------------------------

    def limits(self, field: str) -> tuple[float, float]:
        """Fixed whole-sphere limits, so a band keeps its colour while you pan.

        Taken from the diagnosis, which sampled the same grid `create_hfun.py`
        will, rather than from whatever happens to be in the current view.
        """
        d = self.diagnosis
        if field == "cell_width_km":
            return d.h_min, d.h_max
        if field == "mesh_density":
            return (self.hfun.hfun_min / d.h_max) ** 4, (
                self.hfun.hfun_min / d.h_min
            ) ** 4
        raise KeyError(f"unknown hfun field: {field}, "
                       f"expected one of {', '.join(FIELDS)}")

    def describe(self) -> dict:
        d = self.diagnosis
        fields = []
        for name, label in FIELDS.items():
            vmin, vmax = self.limits(name)
            fields.append({"name": name, "label": label,
                           "vmin": float(vmin), "vmax": float(vmax)})

        verdict = "within" if d.within_guideline else "ABOVE"
        return {
            "file": self.hfun.path.name,
            "subtitle": f"distance function · {d.h_min:.4g} to "
                        f"{d.h_max:.4g} km · max gradient "
                        f"{d.max_gradient:.4f}",
            "facts_label": "distance function",
            "stats": [
                ["hfun_min", f"{self.hfun.hfun_min:.4g} km"],
                ["h min", f"{d.h_min:.4g} km"],
                ["h max", f"{d.h_max:.4g} km"],
                ["max gradient", f"{d.max_gradient:.4f}"],
                ["guideline", f"{verdict} {GRADIENT_GUIDELINE}"],
                ["steepest at", f"{d.at_lat:.1f}, {d.at_lon:.1f}"],
                ["HFUN grid", f"{d.nlon} x {d.nlat}"],
            ],
            "home": list(self.home),
            "nx": self.nx,
            "ny": self.ny,
            "cmap": CMAP,
            "ramp": ramp(CMAP),
            "fields": fields,
        }

    # -- frames ----------------------------------------------------------

    def values(self, field: str, extent, nx: int, ny: int) -> np.ndarray:
        """The field on this view's pixel centres, from one `get_hfun` call."""
        key = (*(round(float(v), 6) for v in extent), nx, ny)
        with self._lock:
            if key not in self._frames:
                lon, lat = target_grid(tuple(extent), nx, ny)
                lon2, lat2 = np.meshgrid(lon, lat)
                self._frames[key] = self.hfun.sample_degrees(lon2, lat2)
            h = self._frames[key]

        if field == "cell_width_km":
            return h
        if field == "mesh_density":
            return (self.hfun.hfun_min / h) ** 4
        raise KeyError(f"unknown hfun field: {field}, "
                       f"expected one of {', '.join(FIELDS)}")

    def overlay(self, extent, nx=None, ny=None) -> bytes:
        nx, ny = nx or self.nx, ny or self.ny
        key = (*(round(float(v), 6) for v in extent), nx, ny)
        with self._lock:
            if key not in self._overlays:
                self._overlays[key] = _overlay(extent, nx, ny)
            return self._overlays[key]

    def frame(self, field: str, extent, nx=None, ny=None,
              compress: int = 1) -> tuple[bytes, float, float]:
        nx, ny = nx or self.nx, ny or self.ny
        img = self.values(field, extent, nx, ny)
        lo, hi = self.limits(field)
        return _png(img, CMAP, lo, hi, compress), float(lo), float(hi)


# ------------------------------------------------------------------ serving


def _handler(viewer: HfunViewer, html: str):
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


def report(viewer: HfunViewer) -> str:
    """The startup summary, also what `gmpas prep hfun --check` prints."""
    d = viewer.diagnosis
    lines = [
        f"{viewer.hfun.path.name}: hfun_min {viewer.hfun.hfun_min:.4g} km, "
        f"grid distance {d.h_min:.4g} to {d.h_max:.4g} km",
        f"max cell size gradient {d.max_gradient:.4f} at "
        f"{d.at_lat:.2f}, {d.at_lon:.2f} "
        f"({'within' if d.within_guideline else 'ABOVE'} the "
        f"{GRADIENT_GUIDELINE} guideline)",
        f"measured on the {d.nlon} x {d.nlat} lat-lon grid create_hfun.py "
        f"would write ({d.spacing_km:.3g} km spacing)",
    ]
    if d.coarsened:
        lines.append(
            "  note: that grid was coarsened to fit in memory, so the "
            "gradient above is a lower bound on the real one"
        )
    if not d.within_guideline:
        lines.append(
            "  a gradient above a few percent changes cell size too quickly; "
            "widen the transition region or raise hfun_min"
        )
    return "\n".join(lines)


def serve(hfun_path, port: int = 8765, nx: int = 1200, ny: int = 700,
          open_browser: bool = True, host: str = "127.0.0.1",
          strict_port: bool = False):
    """Start the distance-function viewer and block until interrupted.

    Same tunnelling rules as `gmpas view` and `gmpas prep view`.
    """
    viewer = HfunViewer(hfun_path, nx=nx, ny=ny)
    server = bind(_handler(viewer, page(f"gmpas prep hfun · "
                                        f"{viewer.hfun.path.name}")),
                  port, host=host, strict=strict_port)
    port = server.server_address[1]

    print(report(viewer))

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
