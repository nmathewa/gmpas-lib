"""One server, one port, several things to look at.

`gmpas view`, `gmpas prep view` and `gmpas prep hfun` each serve a complete
page, and each used to be its own process on its own port. On a laptop that is
merely untidy. On a compute node it is three SSH tunnels to look at one
experiment -- and the run, the mesh it is on, and the distance function that
produced that mesh are exactly the three things one wants side by side.

So this mounts them together. Each page keeps its own handler, its own routes
and its own logic, unchanged; this adds a prefix in front of each, an index at
`/` listing what is available, and a switcher across the top of every page.

The one thing the pages had to give up is absolute API URLs. A page mounted at
`/mesh/` that fetched `/api/meta` would reach the wrong viewer, so they now
fetch `api/meta` relative to wherever they are served. Mounted at the root --
which is what a single source still gets -- that resolves to exactly the paths
they used before.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .viewer import PageHandler

#: inserted right after <body>, so neither page template has to know about it
_NAV = """
<div id="gmpas-nav">
  <a href="/" title="all sources">gmpas</a>
  __LINKS__
</div>
<style>
#gmpas-nav{position:fixed;top:0;right:0;z-index:99;display:flex;gap:2px;
  padding:4px 6px;background:#1e2127;border:0 solid #2c313a;border-width:0 0 1px 1px;
  border-bottom-left-radius:6px;
  font:11px/1 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
#gmpas-nav a{color:#9aa3b0;text-decoration:none;padding:4px 8px;border-radius:4px}
#gmpas-nav a:hover{color:#e6e8ec;background:#252932}
#gmpas-nav a.on{color:#16181c;background:#5dcaa5}
</style>
"""

_INDEX = """<!doctype html>
<html><head><meta charset="utf-8"><title>gmpas</title>
<style>
body{margin:0;font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
     background:#16181c;color:#e6e8ec;display:flex;align-items:center;
     justify-content:center;min-height:100vh}
main{width:min(560px,90vw)}
h1{font-size:15px;font-weight:500;margin:0 0 2px}
p.sub{color:#9aa3b0;margin:0 0 20px;font-size:12px}
a.card{display:block;text-decoration:none;color:inherit;background:#1e2127;
  border:1px solid #2c313a;border-radius:8px;padding:14px 16px;margin-bottom:8px}
a.card:hover{border-color:#5dcaa5}
a.card b{display:block;font-weight:500;margin-bottom:2px}
a.card span{color:#9aa3b0;font-size:12px}
</style></head><body><main>
<h1>gmpas</h1>
<p class="sub">__COUNT__ on this server &middot; one port, one tunnel</p>
__CARDS__
</main></body></html>
"""


@dataclass
class Source:
    """One mounted page: what it is called, and what serves it."""

    slug: str            # "run", "mesh", "hfun" -- also the URL prefix
    label: str           # what the switcher says
    detail: str          # one line on the index card
    handler: object      # a BaseHTTPRequestHandler subclass serving "/" + "api/*"
    warm: object = None  # a Warmup while it is prepared in the background


def nav(sources: list[Source], current: str) -> str:
    links = "".join(
        f'<a href="/{s.slug}/" class="{"on" if s.slug == current else ""}">'
        f'{s.label}</a>'
        for s in sources
    )
    return _NAV.replace("__LINKS__", links)


def index_page(sources: list[Source]) -> str:
    cards = "".join(
        f'<a class="card" href="/{s.slug}/"><b>{s.label}</b>'
        f'<span>{s.detail}</span></a>'
        for s in sources
    )
    n = len(sources)
    return (_INDEX.replace("__CARDS__", cards)
                  .replace("__COUNT__", f"{n} source{'' if n == 1 else 's'}"))


def with_nav(html: str, sources: list[Source], current: str) -> str:
    """Splice the switcher into a page without either page knowing about it."""
    marker = "<body>"
    at = html.find(marker)
    if at < 0:                       # not our page; leave it exactly as it is
        return html
    at += len(marker)
    return html[:at] + nav(sources, current) + html[at:]


def router(sources: list[Source]):
    """Dispatch by prefix to each source's own handler.

    The delegation is deliberate: each mounted handler is called with `self`
    and a rewritten `self.path`, so it runs its own `do_GET` against its own
    viewer and never learns it is not at the root.
    """
    by_slug = {s.slug: s for s in sources}

    # One source needs no index and no switcher: it is mounted at the root as
    # well as under its slug, so `gmpas prep view mesh.nc` serves exactly the
    # paths it always did and costs nobody an extra click.
    only = sources[0] if len(sources) == 1 else None

    class Router(PageHandler):
        def do_GET(self):
            url = urlparse(self.path)
            parts = url.path.lstrip("/").split("/", 1)
            slug = parts[0]

            if only is not None and slug != only.slug:
                return only.handler.do_GET(self)

            if url.path in ("", "/"):
                # built per request: a source still preparing changes its card
                return self._send(index_page(sources).encode(),
                                  "text/html; charset=utf-8")

            source = by_slug.get(slug)
            if source is None:
                return self.send_error(404)

            # /mesh must become /mesh/ before the page loads, or every relative
            # api/... in it would resolve against / and reach the wrong viewer
            if len(parts) == 1 and not url.path.endswith("/"):
                self.send_response(301)
                self.send_header("Location", f"/{slug}/")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            rest = "/" + (parts[1] if len(parts) > 1 else "")
            # answered here, not by the page's handler: a preparing page polls
            # it, and the handler it polls is swapped for the real one when
            # the build finishes -- which would answer 404 to the last poll
            if source.warm is not None and urlparse(rest).path == "/api/warm":
                return self._send(json.dumps(source.warm.state()).encode(),
                                  "application/json")
            self.path = rest + (f"?{url.query}" if url.query else "")
            return source.handler.do_GET(self)

    return Router


def serve(sources: list[Source], port: int = 8765, host: str = "127.0.0.1",
          open_browser: bool = False, strict_port: bool = False,
          banner: str = ""):
    """Start one server carrying every source, and block until interrupted.

    `open_browser` is off by default: see `viewer.open_in_browser` for why
    launching one unasked is the wrong default where gmpas actually runs.
    """
    import sys

    from .viewer import bind, open_in_browser, reach_lines

    server = bind(router(sources), port, host=host, strict=strict_port)
    port = server.server_address[1]

    if banner:
        print(banner)
    print(f"{len(sources)} sources on one port:")
    for s in sources:
        print(f"  /{s.slug:<5} {s.label} — {s.detail}")

    for line in reach_lines(host, port):
        print(line)
    print("ctrl-c to stop")
    # stdout is block-buffered whenever it is not a terminal, so the usual
    # `gmpas view ... > viewer.log &` on a compute node left the log empty --
    # no URL, no tunnel command, nothing to act on until the process ended
    sys.stdout.flush()

    if open_browser:
        open_in_browser(f"http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()


# -------------------------------------------------------- preparing in the background

_PREPARING = """<!doctype html>
<html><head><meta charset="utf-8"><title>gmpas · preparing</title>
<style>
:root{--bg:#16181c;--panel:#1e2127;--line:#2c313a;--fg:#e6e8ec;--dim:#9aa3b0;
      --accent:#4a90d9;color-scheme:dark}
body{margin:0;font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
     background:var(--bg);color:var(--fg)}
main{max-width:560px;margin:14vh auto;padding:0 16px}
h1{font-size:16px;font-weight:500;margin:0 0 4px}
.sub{color:var(--dim);margin:0 0 22px;word-break:break-all}
.bar{height:8px;border-radius:4px;background:var(--panel);border:1px solid var(--line);
     overflow:hidden}
.bar i{display:block;height:100%;width:0;background:var(--accent);transition:width .4s}
#phase{margin-top:10px}#eta{color:var(--dim);font-size:12px}
#err{color:#e5736b;white-space:pre-wrap;margin-top:14px}
</style></head><body><main>
<h1>gmpas is preparing this run</h1>
<p class="sub">__WHAT__</p>
<div class="bar"><i id="fill"></i></div>
<div id="phase">starting</div><div id="eta"></div><div id="err"></div>
<script>
async function poll(){
  try{
    const s = await (await fetch("api/warm", {cache:"no-store"})).json();
    if(s.ready){ location.reload(); return; }
    document.getElementById("phase").textContent = s.phase;
    document.getElementById("fill").style.width =
      (s.total ? 100 * s.done / s.total : 4) + "%";
    document.getElementById("eta").textContent = s.detail;
    if(s.error){ document.getElementById("err").textContent = s.error; return; }
  }catch(e){}
  setTimeout(poll, 1000);
}
poll();
</script></main></body></html>
"""


class Warmup:
    """How a background start stands: read by the preparing page, written by
    the thread building the viewers. Plain attribute writes, read whole."""

    def __init__(self, what: str):
        self.what = what
        self.phase = "opening the files"
        self.done = self.total = 0
        self.unit = ""
        self.eta = 0.0
        self.error: str | None = None
        self.ready = threading.Event()
        self.t0 = time.perf_counter()

    def progress(self, done: int, total: int, unit: str, eta: float) -> None:
        self.phase = "building the mesh cache (first open of this mesh only)"
        self.done, self.total, self.unit, self.eta = done, total, unit, eta

    def state(self) -> dict:
        from .timing import clock

        elapsed = time.perf_counter() - self.t0
        detail = f"{clock(elapsed)} elapsed"
        if self.total and not self.ready.is_set():
            detail = (f"{self.done:,}/{self.total:,} {self.unit}s · eta "
                      f"{clock(self.eta)} · " + detail)
        return {"ready": self.ready.is_set(), "phase": self.phase, "done": self.done,
                "total": self.total, "detail": detail, "error": self.error}


def _pending(warm: Warmup, html: str):
    """A page that shows `warm` and answers every API call with 503 until the
    real viewer replaces it -- nothing hangs on a mesh still being built."""
    from .viewer import PageHandler

    page = html.encode()

    class Pending(PageHandler):
        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("", "/"):
                return self._send(page, "text/html; charset=utf-8")
            if path == "/api/warm":
                return self._send(json.dumps(warm.state()).encode(), "application/json")
            body = json.dumps({"error": warm.error or f"still preparing: {warm.phase}"})
            return self._send(body.encode(), "application/json", 503)

    return Pending


def _check_paths(data_path, mesh_path: str) -> None:
    """What a background start cannot leave to the browser: a path that is not
    there is still an error on the command line, before anything binds."""
    from .series import expand

    if data_path is not None:
        expand(data_path)                  # raises, naming what matched nothing
    if mesh_path and not Path(mesh_path).exists():
        raise FileNotFoundError(f"mesh file {mesh_path} does not exist")


# ----------------------------------------------------------------- assembly


def build(data_path=None, mesh_path: str = "", hfun_path: str = "",
          nx: int = 1200, ny: int = 700,
          background: bool = False) -> tuple[list[Source], str]:
    """Assemble the sources the user actually asked for, and a banner.

    A run always brings its mesh with it, so opening one gives both the data
    page and the mesh page without asking for either; `--hfun` adds the third.
    Every source is constructed before the server binds, so a bad path is an
    error on the command line rather than a 500 in a browser tab.
    """
    from .prep.hfunview import HfunViewer, report
    from .prep.hfunview import _handler as hfun_handler
    from .prep.layout import page as prep_page
    from .prep.meshview import MeshViewer
    from .prep.meshview import _handler as mesh_handler
    from .viewer import PAGE, Viewer
    from .viewer import _handler as run_handler

    if background and (data_path is not None or mesh_path):
        return _build_in_background(data_path, mesh_path, hfun_path, nx, ny)

    built, lines = [], []

    if data_path is not None:
        run = Viewer(data_path, mesh_path, nx=nx, ny=ny)
        n_vars = len(run.series.variables("nCells"))
        built.append(("run", "data",
                      f"{len(run.series)} steps · {n_vars} cell variables", run))
        built.append(("mesh", "mesh",
                      f"{run.mesh.path.name} · {run.mesh.n_cells:,} cells",
                      MeshViewer(run.mesh, nx=nx, ny=ny)))
    elif mesh_path:
        mv = MeshViewer(mesh_path, nx=nx, ny=ny)
        built.append(("mesh", "mesh",
                      f"{mv.mesh.path.name} · {mv.mesh.n_cells:,} cells", mv))

    if hfun_path:
        hv = HfunViewer(hfun_path, nx=nx, ny=ny)
        lines.append(report(hv))
        built.append(("hfun", "hfun",
                      f"{hv.hfun.path.name} · {hv.diagnosis.h_min:.4g} to "
                      f"{hv.diagnosis.h_max:.4g} km · gradient "
                      f"{hv.diagnosis.max_gradient:.4f}", hv))

    if not built:
        raise ValueError("nothing to view: give a run, a mesh, or an hfun file")

    # the switcher has to name every source, so the list is completed first and
    # the pages -- which embed it -- are built in a second pass. With only one
    # source there is nothing to switch to, so the bar is left off entirely
    sources = [Source(slug, label, detail, None)
               for slug, label, detail, _viewer in built]

    def dressed(html: str, slug: str) -> str:
        return html if len(sources) == 1 else with_nav(html, sources, slug)

    for source, (slug, _label, _detail, viewer) in zip(sources, built):
        if slug == "run":
            source.handler = run_handler(viewer, dressed(PAGE, slug))
        elif slug == "mesh":
            source.handler = mesh_handler(
                viewer, dressed(prep_page(f"gmpas · {viewer.mesh.path.name}"),
                                slug))
        else:
            source.handler = hfun_handler(
                viewer, dressed(prep_page(f"gmpas · {viewer.hfun.path.name}"),
                                slug))

    return sources, "\n".join(lines)


def _install(source: Source, viewer, sources: list[Source]) -> None:
    """Give `source` the real handler for `viewer`."""
    from .prep.hfunview import _handler as hfun_handler
    from .prep.layout import page as prep_page
    from .prep.meshview import _handler as mesh_handler
    from .viewer import PAGE
    from .viewer import _handler as run_handler

    def dressed(html: str) -> str:
        return html if len(sources) == 1 else with_nav(html, sources, source.slug)

    if source.slug == "run":
        source.handler = run_handler(viewer, dressed(PAGE))
    elif source.slug == "mesh":
        source.handler = mesh_handler(
            viewer, dressed(prep_page(f"gmpas · {viewer.mesh.path.name}")))
    else:
        source.handler = hfun_handler(
            viewer, dressed(prep_page(f"gmpas · {viewer.hfun.path.name}")))


def _build_in_background(data_path, mesh_path, hfun_path, nx, ny):
    """Serve at once; open the run and its mesh behind a progress page.

    The first open of a large mesh builds its geometry cache -- minutes at
    42M cells -- and a terminal that prints nothing for that long looks
    hung. So the server binds immediately, /run and /mesh show the build's
    progress (the same counter the terminal bar draws), every API call on
    them answers 503 until they are ready, and the real pages replace the
    progress page in place.
    """
    from .prep.hfunview import HfunViewer, report
    from .prep.meshview import MeshViewer
    from .timing import Progress
    from .viewer import Viewer

    _check_paths(data_path, mesh_path)
    lines: list[str] = []
    hfun = None
    if hfun_path:                          # small, and a bad one should fail here
        hfun = HfunViewer(hfun_path, nx=nx, ny=ny)
        lines.append(report(hfun))

    slugs = [("run", "data"), ("mesh", "mesh")] if data_path is not None \
        else [("mesh", "mesh")]
    sources = [Source(slug, label, "preparing…", None) for slug, label in slugs]
    if hfun is not None:
        d = hfun.diagnosis
        sources.append(Source("hfun", "hfun",
                              f"{hfun.hfun.path.name} · {d.h_min:.4g} to {d.h_max:.4g} km"
                              f" · gradient {d.max_gradient:.4f}", None))
        _install(sources[-1], hfun, sources)

    what = str(data_path if data_path is not None else mesh_path)
    warm = Warmup(what)
    page = _PREPARING.replace("__WHAT__", what.replace("&", "&amp;").replace("<", "&lt;"))
    for source in sources[:len(slugs)]:
        html = page if len(sources) == 1 else with_nav(page, sources, source.slug)
        source.handler = _pending(warm, html)
        source.warm = warm

    def work():
        Progress.listener = warm.progress
        try:
            if data_path is not None:
                run = Viewer(data_path, mesh_path, nx=nx, ny=ny)
                mesh = MeshViewer(run.mesh, nx=nx, ny=ny)
                n_vars = len(run.series.variables("nCells"))
                sources[0].detail = f"{len(run.series)} steps · {n_vars} cell variables"
                sources[1].detail = f"{run.mesh.path.name} · {run.mesh.n_cells:,} cells"
                _install(sources[0], run, sources)
                _install(sources[1], mesh, sources)
            else:
                mesh = MeshViewer(mesh_path, nx=nx, ny=ny)
                sources[0].detail = f"{mesh.mesh.path.name} · {mesh.mesh.n_cells:,} cells"
                _install(sources[0], mesh, sources)
            warm.phase = "ready"
            warm.ready.set()
            print(f"gmpas: ready -- {sources[0].detail}", flush=True)
        except Exception as exc:                      # shown on the page, and here
            warm.error = f"{type(exc).__name__}: {exc}"
            warm.phase = "failed"
            print(f"gmpas: could not open {what}: {warm.error}", file=sys.stderr,
                  flush=True)
        finally:
            Progress.listener = None

    threading.Thread(target=work, name="gmpas-warmup", daemon=True).start()
    lines.append("preparing in the background -- the page shows progress and opens "
                 "itself when ready")
    return sources, "\n".join(lines)
