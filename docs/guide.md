# The gmpas guide

Everything gmpas does, explained from scratch. No prior knowledge of MPAS
internals assumed. If you want a terse flag reference instead, that's
[command-line.md](command-line.md); this is the "what is this and why"
version.

- [The problem gmpas solves](#the-problem-gmpas-solves)
- [The one idea behind everything](#the-one-idea-behind-everything)
- [Installing it](#installing-it)
- [Looking at a run](#looking-at-a-run) — `info`, `plot`, `view`
- [The interactive viewer, in detail](#the-interactive-viewer-in-detail)
- [Getting data onto a lat-lon grid](#getting-data-onto-a-lat-lon-grid) — `remap`, `scrip`, `target`
- [Designing and building a mesh](#designing-and-building-a-mesh) — the `prep` family
- [Using it from Python](#using-it-from-python)
- [Settings, caches and files it writes](#settings-caches-and-files-it-writes)
- [Running it on a cluster](#running-it-on-a-cluster)
- [When something goes wrong](#when-something-goes-wrong)

---

## The problem gmpas solves

Most weather and climate data comes on a **regular grid**: rows and columns,
like a spreadsheet or an image. Every cell is a rectangle in latitude and
longitude. Software for this is everywhere, because a grid is just a 2D array.

**MPAS does not work that way.** Its grid is an unstructured mesh of
hexagons and pentagons — a *spherical centroidal Voronoi tessellation*, if
you want the full name. Picture a football stitched from patches, except the
patches can be different sizes: small and dense over the region you care
about, large and coarse elsewhere. That variable resolution is the entire
point of MPAS. You get fine detail where it matters without paying for fine
detail over the whole planet.

The cost is that none of the usual tooling applies. There are no rows and
columns. The data for a variable is one long flat list — cell 0, cell 1, cell
2, up to 41 million — and the shape of each cell lives in a separate set of
arrays.

So people generally do one of two things, and both lose something:

1. **Convert to a regular grid first, then use normal tools.** You've now
   thrown away the variable resolution you paid for, and depending on the
   conversion you may have quietly broken conservation (see
   [remapping](#getting-data-onto-a-lat-lon-grid)).
2. **Draw every cell as a polygon.** Correct, but it does not scale — see the
   measurements in [why.md](why.md), where drawing polygons on a 413k-cell
   mesh takes ~50 seconds and gmpas's own default takes under two.

gmpas exists to plot MPAS output **on its own mesh**, fast, without
converting anything first — and to handle the other end too, designing and
building the mesh in the first place.

## The one idea behind everything

Two observations do most of the work.

**One: the mesh never changes.** A simulation runs for days of model time and
writes hundreds of files, but the mesh is fixed the whole way through. So the
expensive geometry — where every cell's corners are, which cells touch which
— only needs to be worked out **once**, then saved to disk and reused
forever. That's the mesh cache, and it's why the first `gmpas` command on a
new mesh is slow and every one after it is fast.

**Two: an MPAS cell is exactly "the area closest to its centre".** That's
what a Voronoi tessellation *means*. So to find which cell a point falls in,
you don't need to test polygons at all — you just find the nearest cell
centre. That's a nearest-neighbour lookup, which is a solved problem
(gmpas uses a k-d tree from SciPy).

Together those give the trick that makes the viewer feel instant:

> To draw a picture, ask "which cell is nearest?" **once per screen pixel**,
> not once per mesh cell.

A screen is about a million pixels whether your mesh has 8,000 cells or 41
million. So the cost of drawing stops depending on mesh size. And because
that pixel→cell mapping depends only on the mesh and where you're looking —
not on *which variable* or *which timestep* — you compute it once and then
every subsequent frame is a lookup. Changing variable or scrubbing through
time is essentially free. Only panning and zooming pay again.

For small meshes gmpas draws real polygons instead, because below roughly
150,000 cells that's both fast enough and slightly sharper. It switches
automatically; `--method` lets you force either.

---

## Installing it

```bash
pip install gmpas            # the core
pip install "gmpas[plot]"    # + matplotlib and cartopy, needed to draw anything
```

The core needs numpy, scipy, xarray and netCDF4. Drawing needs matplotlib and
cartopy, imported only when you actually draw, so the core stays light.

Two things gmpas **deliberately does not install**, because they're big
native programs best supplied by your system:

| Program | Needed for | How to get it |
|---|---|---|
| `ESMF_RegridWeightGen` | `gmpas remap` | `module load esmf` on HPC, or conda-forge |
| `jigsaw` + `mkgrid` | `gmpas prep generate` | build from source; point `$JIGSAWDIR`/`$MKGRIDFILE` at them |

Full detail in [installation.md](installation.md).

---

## Looking at a run

### `gmpas info` — what is in this file?

The first thing to run on anything unfamiliar.

```bash
gmpas info run/history.2012-02-25_12.00.00.nc
gmpas info run/ --limit 10
```

It prints the mesh (how many cells, how many edges, global or regional, the
lat/lon box it covers, the smallest and largest cell in km, and the ratio
between them) and then lists the variables grouped by what they sit on —
cells, edges, or vertices.

That grouping matters more than it sounds. **MPAS puts different variables in
different places**, and it changes what you can do with them:

- **Cell variables** (`nCells`) — temperature, pressure, most things. One
  value per hexagon.
- **Edge variables** (`nEdges`) — the wind component *perpendicular to each
  cell wall*. This is how MPAS actually stores wind.
- **Vertex variables** (`nVertices`) — vorticity and friends, at the corners.

`--mesh-only` skips the variable listing when you just want to inspect a mesh
file. `--limit` caps how many variables are listed per group.

### `gmpas plot` — one picture, saved to a file

```bash
gmpas plot run/history.nc precipw -o pw.png
gmpas plot run/ theta -l 20 --cmap turbo -o theta20.png
gmpas plot 'run/*.nc' precipw --all-steps -o 'frames/pw_{step:04d}.png' -j 8
```

A publication-shaped figure: coastlines, gridlines, a labelled colourbar, a
title. Useful flags:

| Flag | What it does |
|---|---|
| `-t` / `--time` | which timestep (counts across *all* files, not within one) |
| `-l` / `--level` | which vertical level |
| `--cmap` | any matplotlib colormap name |
| `--extent` | a named region (`global`, `maritime_continent`, `mjo_basin`, `indo_pacific`, `conus`) or leave blank to fit the mesh |
| `--style` | `paper` (default), `poster`, `notebook`, or `mesh` (thin cell outlines, for inspecting structure) |
| `--symmetric` | diverging colours centred on zero — for anomalies and differences, where the zero point should be visually neutral |
| `--method` | `auto`, `poly`, or `raster` — see [the one idea](#the-one-idea-behind-everything) |
| `--all-steps` | render every timestep; `--out` must contain `{step}` |
| `-j` | parallel workers for `--all-steps` |

`--all-steps` is how you make an animation: render the frames, then stitch
them with `ffmpeg` or similar.

### `gmpas view` — the interactive browser viewer

```bash
gmpas view run/
```

Starts a small web server and opens your browser. This is the one you'll use
most, and it gets its own section below.

---

## The interactive viewer, in detail

Think "ncview, but for an unstructured mesh, in a browser". It runs a local
server; nothing leaves your machine.

**Why a browser rather than a desktop window?** Because it works through an
SSH tunnel. Your data is usually on a cluster, and a browser page survives
that hop far better than X11 forwarding does.

If you pass a run, you get **two pages on one port**: the data, and the mesh
it sits on. Add `--hfun` for a third (the distance function that generated
the mesh). One server, one tunnel, a switcher across the top.

### What you can do in it

**Pick a variable** from the sidebar list. Mesh arrays and static fields are
hidden by default — tick "show mesh & static arrays" to see things like
`latCell` and `meshDensity`. Switching variable is near-instant, because the
pixel→cell mapping is already built.

**Scrub through time** with the slider. Same reason: instant.

**Choose a vertical level** for 3D fields.

**Zoom and pan** — mouse wheel zooms toward the cursor, drag pans. While you
move, the existing frame is transformed to fit so you see something
immediately; a sharper one is fetched once you settle.

**Colour controls** — pick a colormap, set explicit min/max, or lock the
range so it stops rescaling as you move through time (important when
comparing timesteps: an auto-scaling colourbar makes everything look the
same).

**Click the map to probe** — gives you the exact cell number, its lat/lon,
and the value there.

**A graticule and scale bar**, both toggleable. The graticule is drawn in the
browser, so it toggles without a round trip.

**Derived variables.** Type an expression into the "derive" box to combine
fields:

| Expression | Meaning |
|---|---|
| `a + b`, `a - b`, `a * b`, `a / b` | elementwise arithmetic on two fields |
| `hypot(a, b)` | √(a²+b²) — vector magnitude, e.g. wind speed |
| `diff(a)` | this timestep minus the previous one — a tendency |

For wind speed you want the **cell-centred** components, not the raw edge
wind: `hypot(uReconstructZonal, uReconstructMeridional)`.

This is a fixed grammar, not a general expression evaluator — deliberately,
since the viewer can be bound to a network interface on a cluster.

**Animations.** Build a named animation over the time axis; it loads frames
in the background and plays them when ready. Several can load independently.

**Export**, from the sidebar:

- **figure** — a proper publication figure (axes, coastlines, colourbar), at
  `paper`/`notebook`/`poster` size
- **data** — the current view as a regular lat-lon netCDF. Note this is
  **nearest-cell sampling, not conservative remapping** — good for
  inspection, wrong for budgets. Use `gmpas remap` when the numbers must add
  up.
- **animation (GIF)**

### `--generic`: plain netCDF files

```bash
gmpas view reanalysis.nc --generic
```

For files that are *already* on a regular lat/lon grid — reanalysis,
satellite products, anything CF-conventional. It skips the mesh and k-d tree
entirely (a regular grid is already the picture, so a view is just an index
range) and serves slices at the file's native resolution.

Limits: it needs real 1D `lat`/`lon` coordinates (2D curvilinear coordinates
won't work), and export isn't implemented for this mode yet. Variables with
no spatial dimensions are shown as a simple line plot instead of a map.

---

## Getting data onto a lat-lon grid

Sometimes you genuinely need a regular grid — to compare against
observations, feed another model, or compute an area budget. That's
`gmpas remap`.

### What "conservative" means, and why you should care

Naively, you could set each output grid point to the value of whichever MPAS
cell it lands in. Fast, and **wrong** for anything that has to add up. If a
coarse cell happens to land on several output points it gets counted several
times; if a fine cell falls between points it vanishes. Total rainfall before
and after would differ.

**Conservative remapping** instead computes, for every pair of overlapping
cells, the exact area of overlap, and weights by that. The area integral is
preserved: total rainfall in equals total rainfall out. That's what you want
for precipitation, fluxes, energy — anything you'll later sum or average over
an area.

gmpas **does not compute those weights itself**, on purpose. Doing it
properly means intersecting polygons on a sphere — great-circle edges, poles,
the dateline, degenerate cells — and ESMF has been doing that correctly for
years. gmpas prepares the inputs, calls ESMF, applies the result, and then
**checks the integral afterwards** and reports the error.

### Running it

`gmpas remap` reads small config files from the directory you point it at:

| File | Required | What it holds |
|---|---|---|
| `target_domain` | **yes** | the output grid: `nlat`, `nlon`, `startlat`, `endlat`, `startlon`, `endlon` |
| `include_fields` | no | which variables to convert |
| `exclude_fields` | no | which to leave out |
| `mesh_file` | no | where the mesh is, if not beside the data |

```bash
gmpas target                      # show the config it found, sanity-check it
gmpas remap run/*.nc -o out/ -j 8 # do it
```

It runs in three stages, which it prints as `[1/3] [2/3] [3/3]`:

1. **Configuration** — reads the config, opens the run, picks fields.
2. **Weights** — writes both grids as SCRIP files and calls
   `ESMF_RegridWeightGen`. This is the slow part, but it happens **once** for
   a whole run: weights depend only on the two grids, never on the data.
   Cached as `map_<method>.nc` and reused.
3. **Remapping** — applies the weights to every file, one output per input,
   in parallel.

Useful flags: `--method conserve` (first-order, default) or `conserve2nd`
(second-order, smoother); `--force-weights` to rebuild the weights;
`--overwrite` to redo existing outputs; `-j` for parallelism, which also sets
the MPI ranks ESMF gets.

**Fields it will skip, and tell you about:** edge and vertex variables (they
need their own weight files, since cell weights only map cells), and fields
with an extra axis beyond time and one vertical level.

### `gmpas scrip` and `gmpas target`

Lower-level pieces, for when you want to drive the remapping yourself with
ESMF, TempestRemap or `ncremap`:

- `gmpas scrip mesh.nc -o mesh.scrip.nc` — write the MPAS mesh as a SCRIP
  grid file, the standard interchange format for remapping tools.
- `gmpas target -o dst.scrip.nc` — write the *target* grid from your config
  as SCRIP. With a data file argument it also lists which fields would be
  remapped.

---

## Designing and building a mesh

Everything above assumes a mesh already exists. The `prep` family is the
other end: making one. This is where MPAS's variable resolution actually gets
decided.

### The idea: a distance function

You don't place cells by hand. You write a small Python file — by convention
`hfun.py` — that answers one question:

> at this longitude and latitude, how big should a cell be, in km?

That's the **distance function** (JIGSAW calls it `hfun`). It defines two
things:

- `hfun_min` — the smallest cell size anywhere, in km
- `get_hfun(lon, lat)` — the desired cell size at each point

A typical one returns 3 km over your region of interest, 60 km far away, and
something smoothly in between. Ready-to-edit templates are in
[`examples/`](../examples/).

### `gmpas prep hfun` — check the design before building

```bash
gmpas prep hfun hfun.py --check     # numbers only, no server
gmpas prep hfun hfun.py             # interactive viewer
```

Building a mesh is slow, so check the design first. It reports:

- the cell size range your function actually produces (which often surprises
  people — the function you wrote and the sizes you imagined can differ)
- the **maximum cell-size gradient**, and where it occurs

That gradient is the number to watch. It measures how fast cell size changes
from one cell to its neighbour. **Change size too abruptly and the mesh
becomes numerically unstable** — waves reflect off the transition instead of
passing through. gmpas warns when it exceeds the usual guideline and tells
you the fix: widen the transition zone, or raise `hfun_min`.

`--mesh` also serves a mesh built from that hfun, so you can compare what you
asked for against what you got.

### `gmpas prep generate` — actually build it

```bash
gmpas prep generate hfun.py -o mesh/
```

Runs JIGSAW to generate the mesh, then `mkgrid` to convert it into MPAS's
own format. Needs `$JIGSAWDIR` and `$MKGRIDFILE` (or `--jigsaw`/`--mkgrid`).

Useful flags: `--skip-mkgrid` stops after JIGSAW's files (then `mkgrid` isn't
needed); `--init` supplies starting points for a quasi-uniform icosahedral
mesh; `--qlim` sets JIGSAW's mesh-quality limit; `--force` regenerates;
`--allow-steep` proceeds despite a gradient warning.

It prints what it produced and what to do next, including the `graph.info`
file METIS needs to partition the mesh across MPI ranks.

### `gmpas prep view` — look at a mesh

```bash
gmpas prep view mesh.nc
```

The same browser viewer, but for a mesh on its own — no data needed. Cells
are coloured by their width in km, so you can see directly where the
refinement is and how smoothly it transitions.

### Reshaping an existing mesh

Three tools that transform a mesh rather than build one:

- **`prep scale`** — make a regional mesh finer or coarser overall, around a
  tangent point. `--scale-factor 2` roughly halves cell size. Only sensible
  for regional meshes: the stereographic scaling drifts far from the tangent
  point, and past ~120° it starts making cells *coarser* instead of finer.
  gmpas warns if your mesh reaches that far. Takes a mesh straight from
  `prep generate`, not one already through `init_atmosphere`.

- **`prep relocate`** — move a mesh's refined region somewhere else. Built a
  nice 3 km patch over the Maritime Continent and now want the same mesh over
  the Caribbean? This rotates it. It auto-detects where the refinement
  currently is (from the finest cell) unless you say.

- **`prep create-region`** — cut a regional mesh out of a global one, given a
  boundary polygon:

  ```bash
  gmpas prep create-region global.nc --polygon 40,-100 50,-100 50,-70 40,-70
  ```

  Writes a PNG of the boundary zone so you can check the cut before using it.

---

## Using it from Python

For notebooks and scripts. The main entry point attaches to xarray:

```python
import gmpas

ds = gmpas.open_mpas("diag.nc", mesh="maritime.region.nc")

ds.mpas.plot("mslp")          # cell field, filled Voronoi cells
ds.mpas.plot("u")             # edge field, drawn on the cell walls
ds.mpas.plot_mesh()           # the mesh itself, coloured by cell width
ds.mpas.quiver("uReconstructZonal", "uReconstructMeridional")
ds.mpas.variables             # what's available, grouped by element
```

`plot()` takes `time=`, `level=`, and everything `cell_field` does
(`cmap`, `extent`, `vmin`/`vmax`, `symmetric`, `style`, `method`).

The pieces are usable directly too:

```python
from gmpas import MpasMesh, rasterize, cell_field, save_figure

mesh = MpasMesh.load("mesh.nc")     # cached geometry
img, lon, lat = rasterize(mesh, values, extent=(90, 160, -20, 20))
fig, ax = cell_field(mesh, values, label="mm", title="precipitable water")
save_figure(fig, "out.png")
```

Wind deserves a note. MPAS stores wind as the component **normal to each cell
edge**, which isn't directly plottable as a vector. `reconstruct_cell_winds`
turns edge winds into cell-centred zonal/meridional components — but if your
file already has `uReconstructZonal`/`uReconstructMeridional`, prefer those:
they're what the model itself computed, and the naive reconstruction has a
known factor-of-two bias that gmpas pins with a test.

More in [notebook.md](notebook.md).

---

## Settings, caches and files it writes

### Environment variables

| Variable | What it does |
|---|---|
| `GMPAS_CACHE_DIR` | where mesh geometry is cached (default `~/.cache/gmpas/mesh`) |
| `GMPAS_DATA_DIR` | tried first when resolving relative paths |
| `GMPAS_VALUES_CACHE_MB` | memory budget for cached field reads (default 512 MB) |
| `JIGSAWDIR`, `MKGRIDFILE` | the mesh-generation programs |

`--cache-dir` overrides the cache location per run.

### The mesh cache

The first `gmpas` command on a new mesh builds its geometry and writes it to
the cache directory; everything afterwards memory-maps it. It's keyed on the
mesh file's identity (path, size, modification time, cell and edge counts),
so **regenerating a mesh gets a fresh cache** rather than silently reusing
the old one.

Safe to delete at any time — it rebuilds. Worth knowing it isn't small: a
41-million-cell mesh caches to roughly 15 GB. gmpas checks there's room
before starting and refuses with a clear message if there isn't.

`GMPAS_VALUES_CACHE_MB` is separate: it caps how much memory is spent
remembering recently-read fields, so re-rendering the same timestep doesn't
re-read from disk. It's a **byte budget, not a count**, because one field is
about 2 MB on a small mesh and 320 MB on a 41-million-cell one.

---

## Running it on a cluster

Your data is on the cluster; your browser is on your laptop. The server has
to be reachable across that gap.

**If gmpas runs on a login node** (or any machine you can SSH to directly):

```bash
# on the cluster
gmpas view /scratch/run/ --no-browser

# on your laptop, leave running
ssh -N -L 8765:localhost:8765 you@cluster
# then open http://localhost:8765
```

**If gmpas runs on a compute node**, your tunnel lands on the login node
instead, so the viewer has to listen on all interfaces and you hop through:

```bash
# on the compute node
gmpas view /scratch/run/ --host 0.0.0.0 --no-browser

# on your laptop
ssh -N -L 8765:<compute-node>:8765 you@login-node
```

gmpas prints the exact command with your hostname and username filled in — on
a scheduler it reads the submitting host from `PBS_O_HOST`/`SLURM_SUBMIT_HOST`
so the command is complete rather than a template.

To avoid retyping the tunnel, put it in your laptop's `~/.ssh/config`:

```ssh-config
Host cluster
  LocalForward 8765 localhost:8765
```

Then plain `ssh cluster` forwards it every time. (This is what editors like
VS Code do for you automatically — they detect the listening port and forward
it silently, which is why it "just works" there and needs a step here.)

Two more cluster notes:

- **`--no-browser` is usually right remotely.** A browser opening *on the
  compute node* isn't useful. Without a graphical session gmpas will say so
  rather than launching a terminal browser into your session.
- **`-j` sizes both** the MPI ranks ESMF gets and the worker pool that
  converts files. gmpas reads the scheduler's own statement of what your job
  was given (`SLURM_CPUS_PER_TASK`, `PBS_NCPUS`, and friends), then the
  affinity mask, then the machine — so it doesn't spawn 256 workers on a
  4-core allocation.

More in [cluster.md](cluster.md).

---

## When something goes wrong

**"no mesh information and no mesh file was found beside it"** — the data
file doesn't carry the mesh and gmpas couldn't find one nearby. Pass
`-m /path/to/mesh.nc`. History files often carry their own mesh; diagnostic
files usually don't.

**The first command on a big mesh takes minutes** — that's the geometry cache
being built. Once. Everything after is a memory-map. If it happens *every*
time, the cache directory probably isn't writable or keeps changing.

**`gmpas view` prints a URL but the page won't load** — you're almost
certainly running remotely without a tunnel. See
[running it on a cluster](#running-it-on-a-cluster). The server itself is
fine; nothing on your laptop is listening on that port.

**`weights map onto N points but this target domain is ...`** — the cached
weights were built for a different output grid. Weights are cached per output
directory as `map_<method>.nc` and aren't keyed on the grids, so editing
`target_domain` and reusing the directory reuses the old weights. Rerun with
`--force-weights`.

**A field was skipped during remapping** — gmpas says why. Edge and vertex
fields need their own weights; fields with an extra axis beyond time and one
level aren't handled.

**Weight generation seems to hang** — gmpas now streams ESMF's output as it
arrives, so read what it's actually saying. On Slurm, running `srun` from
inside an interactive `srun --pty` shell creates a *nested* job step, and
Slurm blocks it while the outer step holds the resources — the message says
so. Use `salloc` for interactive allocations instead.

**Conservation error is reported** — gmpas checks the area integral after
remapping. A tiny number is normal floating-point noise. A large one means
something is genuinely off (often a mismatch between the mesh and the data).
