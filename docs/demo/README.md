# The static demo

A live gmpas viewer at
<https://nmathewa.github.io/gmpas-lib/>, with no server behind it.

## What it is

The published page is `viewer.PAGE` verbatim — the same HTML and JavaScript
`gmpas view` serves. What differs is underneath: `shim.js` intercepts the
page's `fetch("api/…")` calls and answers them in the browser, from the mesh
and the per-cell values that `bake.py` writes beside the page.

So it is the real interface, not a screenshot of one. Pan, zoom, every
colormap, the colour range, the level sliders and the probe all work, at full
resolution, because the browser is doing the same gather the server would.

The colour options work too, `levels` included — `20`, `(,,250)`,
`(-inf)(0,500,100)(500,3000,500)(inf)` — and so do the map projections with a
short closed form: orthographic (drag to turn the globe), polar
stereographic, Lambert conformal and Mercator. `shim.js` carries its own copy
of that arithmetic, and `tests/test_demo.py` compares its frames with the
Python renderer's.

## What it cannot do

- **No time slider or animation** — the demo run is a single timestep.
- **No contour, layers or Hovmöller, and no exports** — those are drawn by
  matplotlib, which needs Python. The page reports this instead of failing
  silently.
- **No Robinson or Equal Earth** — world maps with long tabulated inverses,
  of little use on a regional mesh; `gmpas view` offers them.

## The data

`data/demo.nc` — a real regional MPAS run over the
maritime continent: 8,228 cells, 96 diagnostic fields, 2019-09-01 00:00 UTC.
Published with permission of its author.

## Rebuilding it

```bash
python docs/demo/bake.py                  # -> docs/demo/site/
python -m http.server -d docs/demo        # then open /site/
```

`.github/workflows/demo.yml` does the same on every push to `main` that
touches `src/gmpas/` or this directory, so the demo cannot drift from the
viewer it is demonstrating.

## The one thing that can drift

`shim.js` re-implements `ViewIndex` (`src/gmpas/viewer.py`) in JavaScript:
nearest cell centre on the unit sphere, blanked where the pixel lies further
than twice that cell's own equivalent-disc radius. A second implementation of
a render path can disagree with the first, so `tests/test_demo.py` renders the
same frames through both and compares the pixels rather than assuming.
