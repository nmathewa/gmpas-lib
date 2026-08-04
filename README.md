# gmpas

[![tests](https://github.com/nmathewa/gmpas-lib/actions/workflows/tests.yml/badge.svg)](https://github.com/nmathewa/gmpas-lib/actions/workflows/tests.yml)
[![version](https://img.shields.io/badge/version-0.4.0-blue)](docs/status.md)
[![python](https://img.shields.io/badge/python-3.10%2B-blue)](docs/installation.md)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Fast plotting of MPAS output **on its own native mesh** — no regridding, so
variable resolution is preserved exactly as the model carries it. Plus the
other end of the pipeline: designing a mesh, building it with JIGSAW, and
looking at either before or after it exists.

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
conda env create -f environment.yml && conda activate gmpas && pip install -e . --no-deps
```

Pure pip works too where wheels exist:

```bash
pip install -e ".[dev]"
```

Two workflows need external programs, which pip cannot provide. Conservative
remapping needs ESMF (`conda install -c conda-forge esmf nco`); mesh generation
needs [JIGSAW](https://github.com/dengwirda/jigsaw) and, for the final step,
MPI and PnetCDF. Skip both if you only plot and view.

Check it landed:

```bash
gmpas --version && pytest -q
```

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
gmpas prep generate hfun.py -o mesh/
```

Any path may be a file, a directory, or a glob; a directory or glob is read as
one time series across files, which is how MPAS writes output. Running `gmpas`
with no arguments prints the whole list with examples.

Everything except `prep` is **postprocessing** — it opens a run and renders,
remaps or exports it. `prep` is the other end, for work that happens before
there is any output.

## Documentation

| | |
|---|---|
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
