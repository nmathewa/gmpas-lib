# Preprocessing

The rest of gmpas opens a run. `gmpas prep` covers what happens before one
exists — building a mesh and looking at it on its own.

```bash
gmpas prep view mesh.nc
```

A browser view of a mesh file with no output data anywhere near it: cells
coloured by width, so where the mesh refines is visible at a glance.

```
8,228 cells, 24,987 edges, regional — cell width 43 to 68.1 km
listening on 127.0.0.1:8765 — this machine only
```

Same flags as `view` for tunnelling and size: `-p/--port`, `--host`,
`--width`, `--height`, `--no-browser`. On a compute node use `--host 0.0.0.0`
and tunnel, exactly as for `view`.

Two fields are offered, both derived from geometry the mesh cache already
holds, so neither costs a rebuild: `cell_width_km` (the default) and
`cell_area_km2`.

**The colour scale is fixed**, taken once from the whole mesh rather than
autoscaled to the current view. There is no vmin/vmax control to put it back,
and that is the point: a scale that restretched as you panned would make the
same refinement band change colour while you moved across it, which misreads a
variable-resolution mesh badly.

The layout is the `view` page reduced to what a preprocessing step uses. The
timestep and level sliders, the colormap picker, the colour-range boxes, the
animation panel, the exports and the probe all belong to a model run, so they
are absent rather than disabled. Pan, zoom, graticule, scale bar and the
extent box stay — the last because reading a domain off a mesh is how one
picks a region.

`prep.layout.page(title, panel=..., script=...)` is that layout as a reusable
shell, with slots for a step's own controls. The sidebar's facts block is
data-driven — a step sends `subtitle` and `stats`, so one with no cells or
edges reuses the shell rather than forking it. `prep hfun` is the second user;
mesh generation is meant to be the third
([issue 15](https://github.com/nmathewa/gmpas-lib/issues/15)); like the
remapping tools, JIGSAW would be an external executable gmpas shells out to,
not a Python dependency.

### One server, one port

A run, the mesh it is on, and the distance function behind that mesh are the
three things one wants side by side, and they used to be three processes on
three ports — three SSH tunnels from a compute node. They are now one:

```bash
gmpas view /path/to/run/ --hfun hfun.py     # data + mesh + hfun
gmpas prep view mesh.nc --hfun hfun.py      # mesh + hfun
gmpas prep hfun hfun.py --mesh mesh.nc      # hfun + mesh
```

```
2 sources on one port:
  /mesh  mesh — maritime.region.nc · 8,228 cells
  /hfun  hfun — hfun.py · 12 to 60 km · gradient 0.0300
```

`/` lists what is available and each page carries a switcher, so one tunnel
reaches all of them. Opening a run gives the mesh page for free — a run always
brings its mesh with it. Every source is constructed before the server binds,
so a bad path is an error at the prompt rather than a 500 in a browser tab.

A single source is unchanged: it is served at the root, with no index and no
switcher, on exactly the paths it always used.

### Generating a mesh with JIGSAW

```bash
export JIGSAWDIR=/path/to/jigsaw/build/src     # or wherever it installed
gmpas prep generate hfun.py -o mesh/
```

Same division of labour as the remapping side: gmpas does not generate the mesh
itself, it prepares every input JIGSAW needs, shells out, and reads the result
back. What it adds is everything around the call.

```
  hfun.py: 12 to 60 km, max gradient 0.0300
  sampling hfun onto 3336 x 1668 (5.6M points)
  wrote GEOM.msh, HFUN.msh (32 MB) and MESH.jig
  running jigsaw — this is the slow part
  MESH.msh: 9,734 generating points, 19,464 triangles in 8.7 s
```

`HFUN.msh` is written the way `create_hfun.py` writes it — the same grid, the
same `meshgrid(lats, lons)` argument order, and therefore the same
longitude-major value ordering. Getting that order wrong would transpose the
distance function and refine the wrong part of the planet *without failing*, so
there is a test comparing the file back against the function elementwise.

**The distance function is checked before any time is spent.** Generation takes
minutes; measuring the gradient takes a second. A transition steeper than the
0.03 guideline stops the run rather than producing a mesh you would throw away:

```
gmpas: hfun.py has a maximum cell size gradient of 0.0501 at 41.2, -95.0,
above the 0.03 guideline. A mesh built from it will change cell size too
quickly to be well behaved.
  Widen the transition region, or raise hfun_min.
  Pass --allow-steep to generate it anyway.
```

JIGSAW is found by `--jigsaw`, then `$JIGSAWDIR`, then `PATH` — in that order.
PATH is last because JIGSAW installs wherever `CMAKE_INSTALL_PREFIX` pointed
and its build tree leaves the binary in `build/src/`, so on most machines it is
not on PATH at all. `$JIGSAWDIR` accepts either the binary or the directory
holding it.

`MESH.msh` is reused if it is already there, unless `--force`. `--init` passes
an initial point set through to `INIT_FILE`, which is what gives a
quasi-uniform mesh icosahedral structure — a constant `HFUN` alone produces
7-sided cells.

After JIGSAW, the run continues into the two conversion steps, so the output
directory ends up holding everything `mkgrid` reads:

```
  SaveVertices   9,734 generating points
  SaveTriangles  19,464 triangles
  SaveDensity    the mesh density at each point
  SaveCode       a copy of the hfun.py that produced them
```

`SaveVertices` and `SaveTriangles` carry the file's own tokens rather than
floats reformatted by us, so JIGSAW's precision survives the trip, and the
triangle indices are left exactly as JIGSAW wrote them — 0-based, which is what
`mkgrid` expects. `SaveDensity` is MPAS's `meshDensity`:

```
rho(x) = (h_fine / h(x)) ** 4
```

one value per generating point, 1.0 where the mesh is finest.

Two details are worth knowing. A real `MESH.msh` has a `POWER` block between
`POINT` and `TRIA3` — the per-point weights for a power diagram — which
`convert_jigsaw.py` skips only as a side effect of its counter staying at the
limit across those lines; gmpas dispatches on section headers instead, so an
unfamiliar block is ignored because it is unfamiliar rather than by luck. And
`create_density` reuses the coordinates already parsed out of `MESH.msh`
instead of reading back the `SaveVertices` it just wrote with `np.loadtxt`.

All three files were checked **byte for byte** against the tutorial's own
scripts run on the same mesh with the same interpreter.

**The one step gmpas does not take is `mkgrid`**, which needs MPI and PnetCDF:

```bash
cd mesh/ && mkgrid 12000
```

That argument is `nominalMinDc` in **metres** while `hfun.py` works in km
throughout — the one unit seam in this workflow, and an easy factor of a
thousand to get wrong — so the command computes it for you and prints the line
to run.

### Looking at a mesh before it exists

```bash
gmpas prep hfun hfun.py            # browse it
gmpas prep hfun hfun.py --check    # just the numbers, no server
```

In the JIGSAW workflow for MPAS-Atmosphere, a variable-resolution mesh is
defined entirely by a small Python file. `create_hfun.py` samples it onto a
lat-lon grid to write `HFUN.msh`, JIGSAW turns that into generating points, and
`mkgrid` turns those into `grid.nc`. Every design decision lives in that one
file, so this reads it on its own — no JIGSAW, no mesh, no generation time:

```
hfun.py: hfun_min 12 km, grid distance 12 to 60 km
max cell size gradient 0.0300 at 75.64, -95.05 (within the 0.03 guideline)
measured on the 3336 x 1668 lat-lon grid create_hfun.py would write (12 km spacing)
```

The contract is the mini-tutorial's: a module-level `hfun_min` in km, and
`get_hfun(lon, lat)` taking **radians** and returning **km**. That function is
called once with whole arrays and may do expensive setup — a real one might
interpolate a raster — so it is called once per view here, never per pixel.

**The gradient is the number worth having.** The tutorial's guidance is to keep
cell size changing by no more than a few percent per km of distance, with 0.03
generally safe, and `mesh_quality.py` reports exactly that from a finished
`grid.nc`:

```python
nominalDx = r_earth * nominalMinDc * (1.0 / meshDensity) ** 0.25
gradient  = abs(nominalDx[c0] - nominalDx[c1]) / dcEdge / r_earth
```

Since `meshDensity` is `(hfun_min / h) ** 4` and `nominalMinDc` is `hfun_min`,
that is the change in grid distance between neighbouring cells over the
distance between them. Its continuous limit is `|grad h|` with respect to arc
length, which is what is computed here — on the same lat-lon grid
`create_hfun.py` would write, so the number is comparable both to the 0.03
guideline and to what `mesh_quality.py` will say once the mesh exists. The two
polar rows are excluded: every longitude at a pole is the same point, so the
zonal derivative there is 0/0 rather than a gradient.

Two fields are offered, neither invented here: `cell_width_km` is `get_hfun`
straight out, and `mesh_density` is `(hfun_min / h) ** 4` — MPAS's own
`meshDensity`, 1.0 where the mesh is finest.

Nothing in `prep/` modifies the postprocessing path. It imports the viewer's
rasterizing, PNG encoding, coastline overlay and port binding, and changes
none of them.
