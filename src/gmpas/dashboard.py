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

from dataclasses import dataclass
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
    index = index_page(sources).encode()
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
                return self._send(index, "text/html; charset=utf-8")

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
            self.path = rest + (f"?{url.query}" if url.query else "")
            return source.handler.do_GET(self)

    return Router


def serve(sources: list[Source], port: int = 8765, host: str = "127.0.0.1",
          open_browser: bool = True, strict_port: bool = False,
          banner: str = ""):
    """Start one server carrying every source, and block until interrupted."""
    import socket
    import threading
    import webbrowser

    from .viewer import bind

    server = bind(router(sources), port, host=host, strict=strict_port)
    port = server.server_address[1]

    if banner:
        print(banner)
    print(f"{len(sources)} sources on one port:")
    for s in sources:
        print(f"  /{s.slug:<5} {s.label} — {s.detail}")

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


# ----------------------------------------------------------------- assembly


def build(data_path=None, mesh_path: str = "", hfun_path: str = "",
          nx: int = 1200, ny: int = 700) -> tuple[list[Source], str]:
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

    built, lines = [], []

    if data_path is not None:
        run = Viewer(data_path, mesh_path, nx=nx, ny=ny)
        n_vars = len(run.series.variables("nCells"))
        built.append(("run", "data",
                      f"{len(run.series)} steps · {n_vars} cell variables", run))
        built.append(("mesh", "mesh",
                      f"{run.mesh.path.name} · {run.mesh.n_cells:,} cells",
                      MeshViewer(run.mesh.path, nx=nx, ny=ny)))
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
