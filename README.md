# gmpas
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22003177.svg)](https://doi.org/10.5281/zenodo.22003177)
[![tests](https://github.com/nmathewa/gmpas-lib/actions/workflows/tests.yml/badge.svg)](https://github.com/nmathewa/gmpas-lib/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/gmpas)](https://pypi.org/project/gmpas/)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](docs/installation.md)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)


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

mesh generation requires [JIGSAW](https://github.com/dengwirda/jigsaw) and, MPI and PnetCDF

```bash
gmpas --version
```

From a source install, `pytest -q` runs the test suite.

Full detail, including the extras and what each one pulls in:
[docs/installation.md](docs/installation.md).

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

| | |
|---|---|
| [**The guide**](docs/guide.md) | **start here** — every feature explained from scratch, no MPAS background assumed |
| [Why gmpas exists](docs/why.md) | the problem with lat-lon tooling, and why this is fast |
| [Installation](docs/installation.md) | conda, pip, extras, and the external programs |
| [Command line](docs/command-line.md) | every command and its flags |
| [In a notebook](docs/notebook.md) | the accessor, and using the pieces directly |
| [Preprocessing](docs/preprocessing.md) | `prep view`, `prep hfun`, `prep generate` — mesh design and JIGSAW |
| [Conservative remapping](docs/REMAPPING.md) | the whole terminal workflow, and two MPAS traps |
| [On a cluster](docs/cluster.md) | port forwarding, and the two variables that matter |
| [Configuration](docs/configuration.md) | `GMPAS_CACHE_DIR` and `GMPAS_DATA_DIR` |
| [Examples](examples/) | ready-to-edit `hfun.py` templates |
| [Tests](docs/testing.md) | what the suite covers |
| [Layout](docs/layout.md) | what lives in which module |
| [Differences from the MCP server](docs/mcp-server.md) | what changed on the way to a package |
| [Status](docs/status.md) | what is implemented and what is not |
