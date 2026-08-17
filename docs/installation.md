# Installation

Nothing here needs an MCP server, Claude Desktop, or any of the surrounding
tooling — it is an ordinary Python package.

## From PyPI

```bash
pip install gmpas            # core: geometry, caching, remap weights
pip install "gmpas[plot]"    # + matplotlib and cartopy, for plotting
```

`plot` is the only extra relevant here — it pulls in matplotlib and cartopy.
The core install has no plotting dependency at all: geometry, caching and
rasterizing are usable headless, and `plot.py` imports matplotlib lazily so
`import gmpas` never pays for it. This is the whole install for *using*
gmpas; the rest of this page is for developing gmpas itself.

Conservative remapping additionally needs `ESMF_RegridWeightGen`, and
optionally NCO for `ncremap`. **gmpas does not install either itself** — on an
HPC site, load whatever build the site provides (`module load esmf`, `module
load nco`) rather than pulling a copy from conda-forge into this environment:
a site build is tuned for the local MPI/interconnect, and a second copy in
this environment would only compete with it on `PATH`/`LD_LIBRARY_PATH`,
whether gmpas shells out to it (ESMF) or you run it yourself (`ncremap`) —
see docs/REMAPPING.md and [issue 34](https://github.com/nmathewa/gmpas-lib/issues/34).
On a laptop with no site module to load, `conda install -c conda-forge esmf
nco` into a *separate* environment works fine. These are programs rather than
Python packages, so pip cannot provide them and they are not in
`pyproject.toml`. Skip them if you only plot and view.

Check it landed:

```bash
gmpas --version
```

## From source

Copy the directory (or clone it) onto the target machine and install it.

Everything in one go:

```bash
conda env create -f environment.yml && conda activate gmpas && pip install -e . --no-deps
```

Or by hand — the scientific stack is much easier from conda-forge than from
pip:

```bash
conda create -n gmpas -c conda-forge python=3.12 numpy scipy xarray netcdf4 matplotlib cartopy pillow -y
conda activate gmpas
pip install -e ".[dev]" --no-deps
```

Pure pip works too, if wheels are available for your platform:

```bash
pip install -e ".[dev]"
```

Extras: `plot` pulls in matplotlib and cartopy, `test` pulls in pytest, `dev`
is both plus ruff.

Check it landed:

```bash
gmpas --version && pytest -q
```
