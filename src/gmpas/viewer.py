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


def _png(img: np.ndarray, cmap: str, vmin: float, vmax: float) -> bytes:
    """Colour-map an array straight to PNG bytes, skipping matplotlib figures.

    Going through a Figure costs tens of milliseconds and would undo the point
    of the index reuse. NaN becomes fully transparent so the map shows through.
    """
    from matplotlib import colormaps
    from matplotlib.colors import Normalize
    from PIL import Image

    cm = colormaps[cmap].with_extremes(bad=(0, 0, 0, 0))
    norm = Normalize(vmin=vmin, vmax=vmax)
    rgba = cm(norm(np.ma.masked_invalid(img)), bytes=True)

    buf = io.BytesIO()
    Image.fromarray(rgba[::-1]).save(buf, format="PNG")   # imshow origin=lower
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
    ax.set_extent((lon_min - central, lon_max - central, lat_min, lat_max), crs=src)

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
        self._views: dict[tuple, ViewIndex] = {}
        self._overlays: dict[tuple, bytes] = {}
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

    def view(self, extent) -> ViewIndex:
        key = tuple(round(float(v), 6) for v in extent)
        with self._lock:
            if key not in self._views:
                self._views[key] = ViewIndex(self.mesh, extent, self.nx, self.ny)
            return self._views[key]

    def overlay(self, extent) -> bytes:
        key = tuple(round(float(v), 6) for v in extent)
        with self._lock:
            if key not in self._overlays:
                self._overlays[key] = _overlay(extent, self.nx, self.ny)
            return self._overlays[key]

    def values(self, var: str, time: int, level: int) -> np.ndarray:
        """`time` indexes the whole series, across files, not one file."""
        return self.series.values(var, step=time, level=level)

    def frame(self, var, time, level, extent, cmap, vmin, vmax):
        img = self.view(extent).frame(self.values(var, time, level))
        finite = img[np.isfinite(img)]
        if vmin is None or vmax is None:
            lo = float(np.percentile(finite, 2)) if finite.size else 0.0
            hi = float(np.percentile(finite, 98)) if finite.size else 1.0
        else:
            lo, hi = vmin, vmax
        if hi <= lo:
            hi = lo + 1.0
        return _png(img, cmap, lo, hi), lo, hi

    def probe(self, lon, lat, var, time, level):
        cell = int(self.mesh.cell_of(np.array([lon]), np.array([lat]))[0])
        value = float(self.values(var, time, level)[cell])
        return {"cell": cell, "value": value,
                "lon": round(float(self.mesh.lon_cell[cell]), 4),
                "lat": round(float(self.mesh.lat_cell[cell]), 4)}


# ----------------------------------------------------------------- serving


def _handler(viewer: Viewer):
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
                    return self._send(PAGE.encode(), "text/html; charset=utf-8")
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
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("X-Range", f"{lo},{hi}")
                    self.send_header("Content-Length", str(len(png)))
                    self.end_headers()
                    return self.wfile.write(png)
                if url.path == "/api/overlay":
                    extent = [float(v) for v in q["extent"].split(",")]
                    return self._send(viewer.overlay(extent), "image/png")
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
#top{padding:8px 14px;border-bottom:1px solid var(--line);display:flex;gap:14px;
     align-items:center;color:var(--dim);flex-wrap:wrap}
#top b{color:var(--fg);font-weight:500;white-space:nowrap}
#tlab{font-variant-numeric:tabular-nums}
#stage{flex:1;display:flex;align-items:center;justify-content:center;position:relative;padding:12px;min-height:0}
#frame{display:grid;grid-template-columns:auto auto;grid-template-rows:auto auto;
       max-width:100%;max-height:100%}
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
      cursor:grab;max-width:100%;max-height:100%}
#wrap.drag{cursor:grabbing}
#wrap img{display:block;max-width:100%;height:auto;transform-origin:0 0;will-change:transform}
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
</style></head><body>
<div id="side">
  <h1><span id="title">loading…</span><small id="sub"></small></h1>
  <div class="sec"><label style="margin:0"><input type="checkbox" id="showstatic"
    style="width:auto;vertical-align:-1px"> show mesh &amp; static arrays</label></div>
  <div id="vars"></div>
  <div class="sec"><label>colormap</label><select id="cmap"></select></div>
  <div class="sec"><label>range</label>
    <div style="display:flex;gap:6px">
      <input type="text" id="vmin" placeholder="auto"><input type="text" id="vmax" placeholder="auto">
    </div>
    <div style="margin-top:6px"><button id="reset">reset range</button></div>
  </div>
</div>
<div id="main">
  <div id="top">
    <span>time <b id="tlab">–</b></span><input type="range" id="time" min="0" max="0" style="width:150px">
    <span>level <b id="llab">0</b></span><input type="range" id="level" min="0" max="0" style="width:110px">
    <span>zoom</span><input type="range" id="zoom" min="0" max="800" value="0" style="width:110px">
    <label style="white-space:nowrap"><input type="checkbox" id="grid" checked
      style="vertical-align:-1px"> grid</label>
    <button id="home">reset view</button>
    <span id="probe"></span>
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
  M = await (await fetch("/api/meta")).json();
  $("#title").textContent = M.file;
  M.cmaps.forEach(c=>{const o=document.createElement("option");o.textContent=c;$("#cmap").append(o)});
  home = fit(M.home); view = {...home}; rendered = null;
  subtitle(); fillVars();
  if(M.scanning) setTimeout(pollScan, 400);
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
  cur = M.variables.find(v=>v.name===name);
  [...$("#vars").children].forEach(d=>d.classList.toggle("on",d.textContent===name));
  $("#time").max=M.steps-1; $("#tlab").textContent=M.labels[$("#time").value|0];
  $("#level").max=cur.levels-1; $("#level").value=0; $("#llab").textContent=0;
  overlay(); draw();
}
async function pollScan(){
  if(!M||!M.scanning) return;
  const s=await (await fetch("/api/status")).json();
  const keep=$("#time").value;
  M.steps=s.steps; M.labels=s.labels; M.scanning=s.scanning;
  $("#time").max=M.steps-1; $("#time").value=keep;
  subtitle();
  if(M.scanning) setTimeout(pollScan, 700);
}

async function overlay(){ $("#over").src="/api/overlay?extent="+boxOf(view).join(","); }

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
  const b=boxOf(view);
  const p=new URLSearchParams({var:cur.name,time:$("#time").value,
    level:$("#level").value,extent:b.join(","),cmap:$("#cmap").value});
  if($("#vmin").value) p.set("vmin",$("#vmin").value);
  if($("#vmax").value) p.set("vmax",$("#vmax").value);
  const t0=performance.now();
  const r=await fetch("/api/frame?"+p);
  if(!r.ok){ say((await r.json()).error); busy=false; return; }
  const [lo,hi]=r.headers.get("X-Range").split(",").map(Number);
  const url=URL.createObjectURL(await r.blob());
  const img=$("#data"), old=img.src;
  img.onload=()=>{ if(old.startsWith("blob:")) URL.revokeObjectURL(old); };
  img.src=url;
  rendered=b;
  img.style.transform=""; $("#over").style.transform="";
  colorbar(lo,hi); scalebar(); graticule();
  say(`${cur.label} \u00b7 ${Math.round(performance.now()-t0)} ms`);
  busy=false;
  if(pend){ pend=false; draw(); }
}
let redrawTimer=null;
function schedule(ms){ preview(); scalebar(); graticule(); clearTimeout(redrawTimer);
  redrawTimer=setTimeout(()=>{ overlay(); draw(); }, ms); }

$("#time").oninput = e=>{ $("#tlab").textContent=M.labels[e.target.value]; draw(); };
$("#level").oninput = e=>{ $("#llab").textContent=e.target.value; draw(); };
$("#cmap").onchange = draw;
$("#vmin").onchange = draw; $("#vmax").onchange = draw;
$("#reset").onclick = ()=>{ $("#vmin").value=""; $("#vmax").value=""; draw(); };
$("#home").onclick = ()=>{ view={...home}; clamp(); schedule(0); };
$("#zoom").oninput = e=>{ view.w = home.w/Math.pow(2, e.target.value/100);
                          clamp(); schedule(180); };

$("#wrap").onwheel = ev=>{
  ev.preventDefault();
  const r=$("#data").getBoundingClientRect();
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
  $("#wrap").setPointerCapture(ev.pointerId); $("#wrap").classList.add("drag");
};
$("#wrap").onpointermove = ev=>{
  if(!drag) return;
  const r=$("#data").getBoundingClientRect(), b=boxOf(view);
  const dx=(ev.clientX-drag.x)/r.width*(b[1]-b[0]);
  const dy=(ev.clientY-drag.y)/r.height*(b[3]-b[2]);
  if(Math.abs(ev.clientX-drag.x)+Math.abs(ev.clientY-drag.y)>3) drag.moved=true;
  view.clon=drag.clon-dx; view.clat=drag.clat+dy;
  clamp(); preview(); scalebar(); graticule();
};
$("#wrap").onpointerup = async ev=>{
  const moved=drag&&drag.moved; drag=null;
  $("#wrap").classList.remove("drag");
  if(moved){ schedule(0); return; }
  if(!cur) return;
  const r=$("#data").getBoundingClientRect(), b=boxOf(view);
  const lon=b[0]+((ev.clientX-r.left)/r.width)*(b[1]-b[0]);
  const lat=b[3]-((ev.clientY-r.top)/r.height)*(b[3]-b[2]);
  const q=new URLSearchParams({lon,lat,var:cur.name,
    time:$("#time").value,level:$("#level").value});
  const d=await (await fetch("/api/probe?"+q)).json();
  $("#probe").textContent=`cell ${d.cell} @ ${d.lat}\u00b0, ${d.lon}\u00b0 = ${d.value.toPrecision(6)}`;
};
$("#grid").onchange = graticule;
addEventListener("resize", ()=>{ scalebar(); graticule(); });
boot();
</script></body></html>
"""
