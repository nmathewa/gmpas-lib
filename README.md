<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/logo/gmpas-dark.svg">
  <img src="docs/logo/gmpas.svg" alt="gmpas" height="64">
</picture>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22003177.svg)](https://doi.org/10.5281/zenodo.22003177)
[![tests](https://github.com/nmathewa/gmpas-lib/actions/workflows/tests.yml/badge.svg)](https://github.com/nmathewa/gmpas-lib/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/gmpas)](https://pypi.org/project/gmpas/)
[![docs](https://readthedocs.org/projects/gmpas/badge/?version=latest)](https://gmpas.readthedocs.io)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/gmpas/)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**[Try the viewer in your browser](https://nmathewa.github.io/gmpas-lib/)** —
the real interface on a real MPAS run, with nothing to install.


```python
import gmpas

ds = gmpas.open_mpas("diag.2019-09-01_00.00.00.nc", mesh="maritime.region.nc")

ds.mpas.plot("mslp")        # cell field, filled Voronoi polygons
ds.mpas.plot("u")           # edge field, drawn on the cell faces themselves
ds.mpas.plot_mesh()         # where the mesh actually refines, in km
```

`mesh=` may be omitted when the file carries its own mesh information, or when
a mesh file with a matching cell count sits beside it.

## Installation

```bash
pip install gmpas            # core: geometry, caching, remap weights
pip install "gmpas[plot]"    # + matplotlib and cartopy, for plotting
```


```bash
conda env create -f environment.yml && conda activate gmpas && pip install -e . --no-deps
```

Mesh generation requires [JIGSAW](https://github.com/dengwirda/jigsaw), MPI and PnetCDF.

```bash
gmpas --version
```

From a source install, `pytest -q` runs the test suite.

Full detail, including the extras and what each one pulls in:
[gmpas.readthedocs.io](https://gmpas.readthedocs.io).

## Usage

```bash
gmpas info          history.2012-02-25_12.00.00.nc
gmpas plot          history.2012-02-25_12.00.00.nc precipw -o pw.png
gmpas view          /path/to/run/
gmpas remap         history.*.nc -o out/
gmpas prep view     mesh.nc
gmpas prep hfun     hfun.py --check
gmpas prep generate hfun.py -o mesh/     # needs $JIGSAWDIR and $MKGRIDFILE
```


## Documentation

**[gmpas.readthedocs.io](https://gmpas.readthedocs.io)**

Ready-to-edit `hfun.py` templates are in [examples/](examples/).

> This document is generated using LLM (Claude)
