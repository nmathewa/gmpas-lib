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
    """Transparent coastline and graticule layer for one view box."""
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
    ax.gridlines(linewidth=0.4, linestyle="--", alpha=0.45, color="#111")
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
        self.series = Series(data_path, mesh_path)
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
            "home": list(self.home),
            "cmaps": CMAPS,
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


def serve(data_path, mesh_path="", port=8765, nx=1200, ny=700, open_browser=True):
    """Start the viewer and block until interrupted."""
    viewer = Viewer(data_path, mesh_path, nx=nx, ny=ny)
    server = ThreadingHTTPServer(("127.0.0.1", port), _handler(viewer))

    cells = viewer.mesh.n_cells
    kind = "regional" if not viewer.mesh.is_global else "global"
    print(f"{cells:,} cells, {kind} — {len(viewer.series)} steps across "
          f"{viewer.series.n_files} file(s), "
          f"{len(viewer.plottable_cell_vars())} plottable fields")
    print(f"http://127.0.0.1:{port}   (ctrl-c to stop)")
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
#side{width:260px;flex:none;background:var(--panel);border-right:1px solid var(--line);
      display:flex;flex-direction:column;overflow-y:auto}
#side h1{font-size:13px;font-weight:500;margin:0;padding:12px 14px;border-bottom:1px solid var(--line)}
#side h1 small{display:block;color:var(--dim);font-weight:400;margin-top:2px}
.sec{padding:10px 14px;border-bottom:1px solid var(--line)}
.sec label{display:block;color:var(--dim);font-size:11px;letter-spacing:.04em;
           text-transform:uppercase;margin-bottom:6px}
select,input[type=text]{width:100%;background:#14161a;color:var(--fg);
  border:1px solid var(--line);border-radius:4px;padding:5px 6px;font:inherit}
input[type=range]{width:100%;accent-color:var(--accent)}
#vars{flex:1;overflow-y:auto;padding:6px 0}
#vars div{padding:4px 14px;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#vars div:hover{background:#252932}
#vars div.on{background:var(--accent);color:#08201a}
#vars div.static{color:var(--dim);font-style:italic}
#main{flex:1;display:flex;flex-direction:column;min-width:0}
#top{padding:8px 14px;border-bottom:1px solid var(--line);display:flex;gap:14px;
     align-items:center;color:var(--dim);flex-wrap:wrap}
#top b{white-space:nowrap}
#tlab{font-variant-numeric:tabular-nums}
#top b{color:var(--fg);font-weight:500}
#stage{flex:1;display:flex;align-items:center;justify-content:center;position:relative;padding:12px}
#wrap{position:relative;line-height:0;box-shadow:0 0 0 1px var(--line)}
#wrap img{display:block;max-width:100%;height:auto}
#over{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
#bar{display:flex;gap:10px;align-items:center;padding:8px 14px;border-top:1px solid var(--line)}
#ramp{flex:1;height:12px;border-radius:2px;border:1px solid var(--line)}
.tick{color:var(--dim);font-variant-numeric:tabular-nums}
button{background:#252932;color:var(--fg);border:1px solid var(--line);border-radius:4px;
       padding:5px 9px;font:inherit;cursor:pointer}
button:hover{border-color:var(--accent)}
#msg{position:absolute;top:14px;left:50%;transform:translateX(-50%);background:#000a;
     padding:4px 10px;border-radius:4px;color:var(--dim);opacity:0;transition:opacity .2s}
</style></head><body>
<div id="side">
  <h1><span id="title">loading…</span><small id="sub"></small></h1>
  <div class="sec" style="padding-top:8px;padding-bottom:8px">
    <label style="margin:0"><input type="checkbox" id="showstatic" style="width:auto;vertical-align:-1px">
    show mesh &amp; static arrays</label>
  </div>
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
    <span>time <b id="tlab">0</b></span><input type="range" id="time" min="0" max="0" style="width:160px">
    <span>level <b id="llab">0</b></span><input type="range" id="level" min="0" max="0" style="width:160px">
    <button id="home">reset view</button>
    <span id="probe"></span>
  </div>
  <div id="stage">
    <div id="wrap"><img id="data"><img id="over"></div>
    <div id="msg"></div>
  </div>
  <div id="bar"><span class="tick" id="lo"></span><div id="ramp"></div><span class="tick" id="hi"></span></div>
</div>
<script>
const $=s=>document.querySelector(s);
let M=null, cur=null, extent=null, busy=false, pend=false;

function say(t){const m=$("#msg");m.textContent=t;m.style.opacity=t?1:0;}

async function boot(){
  M = await (await fetch("/api/meta")).json();
  $("#title").textContent = M.file;
  $("#sub").textContent = `${M.cells.toLocaleString()} cells · ${M.regional?"regional":"global"} · ${M.steps} step${M.steps>1?"s":""} in ${M.files} file${M.files>1?"s":""}`;
  M.cmaps.forEach(c=>{const o=document.createElement("option");o.textContent=c;$("#cmap").append(o)});
  fillVars();
  extent = M.home.slice();
  pick(M.variables.find(v=>!v.static)?.name ?? M.variables[0].name);
}
function fillVars(){
  const show = $("#showstatic").checked;
  $("#vars").innerHTML = "";
  M.variables.filter(v=>show||!v.static).forEach(v=>{
    const d=document.createElement("div");d.textContent=v.name;d.title=v.label;
    if(v.static) d.classList.add("static");
    d.onclick=()=>pick(v.name);$("#vars").append(d);
  });
  if(cur) [...$("#vars").children].forEach(d=>d.classList.toggle("on",d.textContent===cur.name));
}
$("#showstatic")?.addEventListener("change", fillVars);

function pick(name){
  cur = M.variables.find(v=>v.name===name);
  [...$("#vars").children].forEach(d=>d.classList.toggle("on",d.textContent===name));
  $("#time").max = M.steps-1; $("#tlab").textContent = M.labels[$("#time").value|0];
  $("#level").max = cur.levels-1; $("#level").value = 0; $("#llab").textContent = 0;
  overlay(); draw();
}
async function overlay(){ $("#over").src = "/api/overlay?extent="+extent.join(","); }

async function draw(){
  if(!cur) return;
  if(busy){ pend=true; return; }
  busy=true;
  const p = new URLSearchParams({var:cur.name, time:$("#time").value,
    level:$("#level").value, extent:extent.join(","), cmap:$("#cmap").value});
  if($("#vmin").value) p.set("vmin",$("#vmin").value);
  if($("#vmax").value) p.set("vmax",$("#vmax").value);
  const t0=performance.now();
  const r = await fetch("/api/frame?"+p);
  if(!r.ok){ say((await r.json()).error); busy=false; return; }
  const [lo,hi] = r.headers.get("X-Range").split(",").map(Number);
  const blob = await r.blob();
  $("#data").src = URL.createObjectURL(blob);
  $("#lo").textContent = lo.toPrecision(4); $("#hi").textContent = hi.toPrecision(4);
  ramp($("#cmap").value);
  say(`${cur.label} · ${Math.round(performance.now()-t0)} ms`);
  busy=false;
  if(pend){ pend=false; draw(); }
}
function ramp(c){
  const grad={viridis:["#440154","#21918c","#fde725"],plasma:["#0d0887","#cc4778","#f0f921"],
    magma:["#000004","#b73779","#fcfdbf"],cividis:["#00224e","#7c7b78","#fee838"],
    turbo:["#30123b","#a2fc3c","#7a0403"],RdBu_r:["#053061","#f7f7f7","#67001f"],
    coolwarm:["#3b4cc0","#dddddd","#b40426"],BrBG:["#543005","#f5f5f5","#003c30"],
    Blues:["#f7fbff","#6baed6","#08306b"],Spectral_r:["#5e4fa2","#ffffbf","#9e0142"]}[c];
  $("#ramp").style.background = `linear-gradient(90deg,${grad.join(",")})`;
}
$("#time").oninput = e=>{ $("#tlab").textContent=M.labels[e.target.value]; draw(); };
$("#level").oninput = e=>{ $("#llab").textContent=e.target.value; draw(); };
$("#cmap").onchange = draw;
$("#vmin").onchange = draw; $("#vmax").onchange = draw;
$("#reset").onclick = ()=>{ $("#vmin").value=""; $("#vmax").value=""; draw(); };
$("#home").onclick = ()=>{ extent = M.home.slice(); overlay(); draw(); };

$("#wrap").onclick = async ev=>{
  const r = $("#data").getBoundingClientRect();
  const fx = (ev.clientX-r.left)/r.width, fy = (ev.clientY-r.top)/r.height;
  const lon = extent[0] + fx*(extent[1]-extent[0]);
  const lat = extent[3] - fy*(extent[3]-extent[2]);
  const q = new URLSearchParams({lon, lat, var:cur.name,
    time:$("#time").value, level:$("#level").value});
  const d = await (await fetch("/api/probe?"+q)).json();
  $("#probe").textContent = `cell ${d.cell} @ ${d.lat}°, ${d.lon}° = ${d.value.toPrecision(6)}`;
};
$("#wrap").onwheel = ev=>{
  ev.preventDefault();
  const r=$("#data").getBoundingClientRect();
  const fx=(ev.clientX-r.left)/r.width, fy=(ev.clientY-r.top)/r.height;
  const lon = extent[0]+fx*(extent[1]-extent[0]), lat = extent[3]-fy*(extent[3]-extent[2]);
  const k = ev.deltaY>0 ? 1.25 : 0.8;
  extent = [lon-(lon-extent[0])*k, lon+(extent[1]-lon)*k,
            lat-(lat-extent[2])*k, lat+(extent[3]-lat)*k];
  clearTimeout(window._z);
  window._z = setTimeout(()=>{ overlay(); draw(); }, 150);
};
boot();
</script></body></html>
"""
