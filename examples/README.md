# Examples

Ready-to-edit `hfun.py` templates. An `hfun.py` is the whole definition of a
variable-resolution MPAS mesh — everything after it is mechanical — so this is
where the design decisions live.

| file | what it builds |
|---|---|
| [hfun_concentric.py](hfun_concentric.py) | any number of nested refinement rings about one centre |
| [hfun_uniform.py](hfun_uniform.py) | a quasi-uniform mesh at one resolution |

## Using one

```bash
cp examples/hfun_concentric.py my_mesh/hfun.py
$EDITOR my_mesh/hfun.py                       # edit CENTER and RINGS

gmpas prep hfun     my_mesh/hfun.py --check   # the numbers, before spending time
gmpas prep hfun     my_mesh/hfun.py           # look at it in a browser
gmpas prep generate my_mesh/hfun.py -o mesh/  # build it with JIGSAW
```

`python hfun_concentric.py` on its own prints the resolution profile it
describes, which is the quickest way to see the effect of an edit:

```
hfun_min = 3 km, centre 38.0, -95.0
        0 -     600 km     3.0 km
      600 -     900 km     3.0 ->  12.0 km   slope 0.0300
      900 -    1400 km    12.0 km
     1400 -    2000 km    12.0 ->  30.0 km   slope 0.0300
     2000 -    3200 km    30.0 km
     3200 -    4200 km    30.0 ->  60.0 km   slope 0.0300
```

## The contract

Both files satisfy the mini-tutorial's contract, and so will anything you write
from scratch:

- `hfun_min` — a module-level float, the minimum grid distance in **km**
- `get_hfun(lon, lat)` — takes **radians**, returns **km**

`get_hfun` is called once with whole arrays, so it may do expensive setup —
loading a raster and building an interpolator, for instance. Nothing in gmpas
calls it per pixel.

## What to watch

The one number that decides whether a mesh is well behaved is the **cell size
gradient**: km of cell size per km of distance. The guidance is a few percent
at most, with 0.03 generally safe. `gmpas prep hfun --check` measures it from
the distance function alone, before any mesh exists, and `gmpas prep generate`
refuses to run above it unless you pass `--allow-steep`.

`hfun_concentric.py` derives its transition widths from a single `SLOPE`
constant, so every transition is equally gentle by construction and a larger
jump in resolution automatically gets the distance it needs.

Two other things it checks, both failures the tutorial names: transition
regions that overlap because the rings are too close (this raises), and
refinement regions that are only a few tens of cells across when they want to
be several hundred (this warns).
