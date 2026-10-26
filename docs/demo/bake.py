"""Write the static demo: the real viewer page, with its data beside it.

`gmpas view` is a Python server that rasterizes an unstructured mesh on
demand. GitHub Pages serves files. This bridges the two by shipping the mesh
and the per-cell values and letting `shim.js` do the rasterizing in the
browser -- so the published page is the *actual* interface, not a mock-up or
a screenshot, and it stays in step with the viewer because it is the same
HTML and the same JavaScript.

Shipping cells rather than pre-rendered images is what makes the demo worth
using: 8,228 cells is 33 KB a field, against ~240 KB for one baked PNG of one
view, and it keeps pan, zoom, the colour range and exact probe values all
working client-side instead of freezing them at bake time.

Run it directly to rebuild into `docs/demo/site/`:

    python docs/demo/bake.py

With `--tiles URL` it bakes the same page for a run too big to ship as cells
(the 3.75 km global one, 41.9M cells): the page's own files come from the
small demo file as usual, and the values come from the value tiles and
manifest.json published at URL (see `tiles` in shim.js). No mesh or field
files are written then.

    python docs/demo/bake.py docs/demo/data/demo.nc docs/demo/site \
        --tiles https://nmathewa.github.io/gmpas-demo-data/
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "data" / "demo.nc"

#: How the demo names itself. The viewer page keys its localStorage on this
#: (`gmpas.colour.<file>`), and every GitHub Pages project on an account
#: shares one origin -- so a generic name here would collide with any other
#: gmpas demo the same user publishes.
TITLE = "gmpas demo - MPAS maritime continent, 2019-09-01"

#: Colormap stops written per colormap. The page ships 32 for its colorbar;
#: the image itself is drawn from these, and 32 linearly interpolated stops
#: are visibly not viridis. 256 matches what matplotlib would have produced.
RAMP_STOPS = 256


def mesh_arrays(mesh) -> dict[str, np.ndarray]:
    """What the browser needs to decide which cell each pixel falls in.

    `ViewIndex` (viewer.py) finds the nearest cell centre on the unit sphere
    and then blanks the pixel when it is further away than twice *that
    cell's* own equivalent-disc radius. Both halves have to be reproduced or
    a regional mesh bleeds across its own edge, so the radius ships too --
    it cannot be derived from the centres.

    Only lon/lat go over the wire; the browser rebuilds the unit vectors from
    them. Shipping `xyz_cell` outright would cost 197 KB to avoid a rounding
    difference of 8.6e-08, against a smallest cell radius of 3.5e-03 -- four
    and a half orders of magnitude smaller than the thing it would decide.
    """
    radius = np.sqrt(np.asarray(mesh.area_cell) / np.pi) / mesh.sphere_radius
    return {
        "lon": np.asarray(mesh.lon_cell, dtype=np.float32),
        "lat": np.asarray(mesh.lat_cell, dtype=np.float32),
        "radius": np.asarray(radius, dtype=np.float32),
    }


#: Coastlines, kept beside the dataset. Cartopy fetches Natural Earth from
#: the network the first time it is asked, which is a download this build
#: does not need and cannot rely on -- a demo that publishes without
#: coastlines because a third-party host was down is a bad trade for 10 KB.
COAST_CACHE = HERE / "data" / "coast.json"


def coastlines(extent, margin: float = 5.0) -> list[list[list[float]]]:
    """Coastline polylines for the region, as plain [lon, lat] pairs.

    The server draws these into a PNG per view; the browser can draw them
    itself far more cheaply, because a plate carree map is a linear mapping
    and the same polyline reprojects on every zoom for free. Clipped to the
    region so the demo does not carry the whole world's coastline.

    Regenerated from cartopy when it can be, and otherwise read from the
    copy committed beside the data. Delete that file to force a refresh.
    """
    if COAST_CACHE.exists():
        return json.loads(COAST_CACHE.read_text())
    try:
        import cartopy.feature as cfeature
        from shapely.geometry import box
    except ImportError as exc:                 # cartopy is an optional extra
        raise SystemExit(f"bake: no {COAST_CACHE.name} and no cartopy ({exc})")

    lon0, lon1, lat0, lat1 = extent
    clip = box(lon0 - margin, lat0 - margin, lon1 + margin, lat1 + margin)
    out: list[list[list[float]]] = []
    for geom in cfeature.COASTLINE.geometries():
        piece = geom.intersection(clip)
        if piece.is_empty:
            continue
        parts = getattr(piece, "geoms", [piece])
        for part in parts:
            coords = np.asarray(part.coords, dtype=np.float32)
            if len(coords) > 1:
                out.append([[round(float(x), 4), round(float(y), 4)]
                            for x, y in coords])
    COAST_CACHE.write_text(json.dumps(out))
    print(f"  wrote {COAST_CACHE} ({len(out)} polylines) -- commit it")
    return out


#: The world's coastlines (Natural Earth 110m), for a projected map: a globe
#: turned away from the region, or a polar map of it, shows far more of the
#: world than `coast.json` holds. Committed beside the data for the same
#: reason, and fetched by the page only after first paint.
WORLD_CACHE = HERE / "data" / "coast_world.json"


def world_coastlines() -> list[list[list[float]]]:
    if WORLD_CACHE.exists():
        return json.loads(WORLD_CACHE.read_text())
    try:
        import cartopy.feature as cfeature
    except ImportError as exc:
        raise SystemExit(f"bake: no {WORLD_CACHE.name} and no cartopy ({exc})") from None
    out = []
    for geom in cfeature.COASTLINE.with_scale("110m").geometries():
        for part in getattr(geom, "geoms", [geom]):
            if len(part.coords) > 1:
                out.append([[round(float(x), 2), round(float(y), 2)]
                            for x, y in part.coords])
    WORLD_CACHE.write_text(json.dumps(out, separators=(",", ":")))
    print(f"  wrote {WORLD_CACHE} ({len(out)} polylines) -- commit it")
    return out


#: The projections the demo offers: the ones `shim.js` has the formulas for.
#: Robinson and Equal Earth are world maps with long tabulated inverses, of
#: little use on a regional mesh, and stay with the server.
PROJECTIONS = ("auto", "Orthographic", "NorthPolarStereo", "SouthPolarStereo",
               "LambertConformal", "Mercator")


def projected_homes(viewer) -> dict[str, dict]:
    """What /api/projview answers for each offered projection: the mesh framed
    in it, worked out here so the browser need not."""
    homes = {name: viewer.projected_home(name) for name in PROJECTIONS}
    known = set(PROJECTIONS) | {"PlateCarree"}
    for name, home in homes.items():
        if home["proj"] not in known:
            raise SystemExit(f"bake: {name} resolves to {home['proj']}, which "
                             f"shim.js cannot draw")
    return homes


def cmap_sizes(names) -> dict[str, int]:
    """Colours in each colormap that is not the usual 256: the browser needs
    the count to index a short one (GrADS' 13) as matplotlib does."""
    from matplotlib import colormaps

    return {n: int(colormaps[n].N) for n in names if colormaps[n].N != RAMP_STOPS}


def ramps(names, stops: int = RAMP_STOPS) -> dict[str, list[str]]:
    from gmpas.viewer import ramp

    return {n: ramp(n, stops) for n in names}


#: The page sets the coastline layer as an <img src>, not a fetch, so it is
#: the one call a fetch shim cannot intercept. Rather than change the
#: package for the demo's benefit, the bake rewrites that one function --
#: and asserts it found it, so an upstream edit breaks the build loudly
#: instead of quietly publishing a map with no coastlines.
OVERLAY_SRC = '  $("#over").src=`api/overlay?extent=${b.join(",")}&nx=${nx}&ny=${ny}`+'
OVERLAY_HOOK = ('  if(window.GMPAS_OVERLAY) return window.GMPAS_OVERLAY(b, P);\n'
                + OVERLAY_SRC)


#: The demo is the only page a visitor may reach without having installed
#: gmpas, so it is the one that says where gmpas lives. The local viewer is
#: left as it is.
DOCS_URL = "https://gmpas.readthedocs.io/en/latest/"
SOURCE_URL = "https://github.com/nmathewa/gmpas-lib"
PYPI_URL = "https://pypi.org/project/gmpas/"
_LINK = 'style="color:var(--accent)" target="_blank" rel="noopener"'
LINKS_AFTER = '<small id="sub"></small></h1>'
LINKS = (LINKS_AFTER + '\n  <div class="sec hint" id="demolinks" style="margin-top:0">'
         f'a demo of <b>gmpas</b> &middot; <a href="{DOCS_URL}" {_LINK}>documentation</a>'
         f' &middot; <a href="{SOURCE_URL}" {_LINK}>source</a>'
         f' &middot; <a href="{PYPI_URL}" {_LINK}>PyPI</a></div>')

#: The page's tab title, and the chat model's API key field: the demo has no
#: chat server, so a key field there is only a browser warning about a
#: password input (the local viewer keeps it).
PAGE_EDITS = (
    ("<title>gmpas view</title>", "<title>gmpas demo</title>"),
    ('      <label for="cfgKey">API key</label>\n'
     '      <input id="cfgKey" type="password" autocomplete="off" placeholder="none">\n', ""),
    (LINKS_AFTER, LINKS),
)


def patch_page(page: str) -> str:
    """The viewer's own page, with the shim loaded, coastlines hooked, links
    back to the project and no API key field."""
    for old, new in ((OVERLAY_SRC, OVERLAY_HOOK), *PAGE_EDITS):
        if page.count(old) != 1:
            raise SystemExit(
                "bake: could not find, exactly once, a line in viewer.PAGE to patch.\n"
                "  The page changed upstream; update bake.py to match.\n"
                f"  Looked for: {old!r}")
        page = page.replace(old, new, 1)
    tag = '<script src="shim.js"></script>'
    if "</head>" in page:
        return page.replace("</head>", tag + "</head>", 1)
    return page.replace("<body>", "<body>" + tag, 1)


def bake(source: Path, out: Path, tiles: str | None = None) -> None:
    from gmpas.viewer import PAGE, Viewer

    if out.exists():
        shutil.rmtree(out)
    (out / "data" / "f").mkdir(parents=True)

    viewer = Viewer(source, nx=1200, ny=700)
    try:
        meta = viewer.describe()
        meta["file"] = TITLE
        meta["scanning"] = False          # or the page polls api/status for ever
        meta.pop("setup", None)           # no dimension chooser without a server
        meta.pop("chat", None)            # nor a model to ask
        if tiles:
            meta["tiles"] = tiles.rstrip("/") + "/"
        # only the projections shim.js can draw; their home views are baked
        homes = projected_homes(viewer)
        meta["projections"] = [n for n in meta.get("projections", [])
                               if n in PROJECTIONS]
        for row in meta["variables"]:
            # `kinds` is what puts the page into matplotlib plot mode, which
            # needs a server. Without it the page is the fast map and the
            # Hovmoller, contour and layer routes are never called.
            row.pop("kinds", None)

        n = viewer.mesh.n_cells
        if not tiles:
            arrays = mesh_arrays(viewer.mesh)
            blob = b"".join(arrays[k].tobytes() for k in ("lon", "lat", "radius"))
            (out / "data" / "mesh.bin").write_bytes(blob)

        fields = {}
        for row in ([] if tiles else meta["variables"]):
            name, levels = row["name"], max(1, int(row["levels"]))
            stack = np.concatenate([
                np.asarray(viewer.values(name, 0, k), dtype=np.float32)
                for k in range(levels)
            ])
            (out / "data" / "f" / f"{name}.bin").write_bytes(stack.tobytes())
            finite = stack[np.isfinite(stack)]
            fields[name] = {
                "levels": levels,
                # the page's own 2nd/98th percentile default, precomputed so
                # the browser need not sort 100k values to draw one frame
                "lo": float(np.percentile(finite, 2)) if finite.size else 0.0,
                "hi": float(np.percentile(finite, 98)) if finite.size else 1.0,
            }

        (out / "data" / "meta.json").write_text(json.dumps(meta))
        (out / "data" / "fields.json").write_text(json.dumps(fields))
        # a global run needs the whole world's coastline from the start
        (out / "data" / "coast.json").write_text(json.dumps(
            world_coastlines() if tiles else coastlines(meta["home"])))
        (out / "data" / "mesh.json").write_text(json.dumps(
            {"cells": n, "proj": homes, "cmap_n": cmap_sizes(meta["cmaps"])}))
        (out / "data" / "coast_world.json").write_text(
            json.dumps(world_coastlines(), separators=(",", ":")))

        # One file per colormap, fetched when it is chosen. All of them at
        # 256 stops is 174 KB, which is a quarter of first paint spent on 62
        # colormaps nobody has picked yet.
        (out / "data" / "ramps").mkdir()
        for name, stops in ramps(meta["cmaps"]).items():
            safe = name.replace("/", "_")
            (out / "data" / "ramps" / f"{safe}.json").write_text(json.dumps(stops))

        # Serve the *undressed* page: dashboard.with_nav splices in absolute
        # links to "/" and "/<slug>/", which on Pages point off the demo.
        (out / "index.html").write_text(patch_page(PAGE))
        shutil.copy(HERE / "shim.js", out / "shim.js")
        shutil.copy(HERE / "README.md", out / "about.md")
    finally:
        viewer.close()

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    first = sum((out / p).stat().st_size for p in
                ("index.html", "shim.js", "data/meta.json", "data/mesh.bin",
                 "data/fields.json", "data/coast.json",
                 "data/ramps/viridis.json") if (out / p).exists())
    print(f"baked {out}")
    print(f"  {len(meta['variables'])} variables, {n:,} cells")
    print(f"  first paint {first/1024:7.1f} KiB   (page, mesh, meta, one field)")
    print(f"  whole site  {total/1024:7.1f} KiB")


if __name__ == "__main__":
    args = sys.argv[1:]
    tiles = None
    if "--tiles" in args:
        i = args.index("--tiles")
        tiles = args[i + 1]
        del args[i:i + 2]
    src = Path(args[0]) if args else DEFAULT_DATA
    dest = Path(args[1]) if len(args) > 1 else HERE / "site"
    if not src.exists():
        raise SystemExit(f"no dataset at {src}")
    bake(src, dest, tiles)
