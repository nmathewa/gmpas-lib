# Installation

Nothing here needs an MCP server, Claude Desktop, or any of the surrounding
tooling — it is an ordinary Python package. Copy the directory (or clone it)
onto the target machine and install it.

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

Conservative remapping additionally needs `ESMF_RegridWeightGen`, and
optionally NCO for `ncremap`. **gmpas does not install ESMF itself** — on an
HPC site, load whatever build the site provides (`module load esmf`) rather
than pulling one from conda-forge into this environment: a site build is
tuned for the local MPI/interconnect, and a second copy in this environment
would only compete with it on `PATH`/`LD_LIBRARY_PATH` (see
docs/REMAPPING.md and [issue 34](https://github.com/nmathewa/gmpas-lib/issues/34)).
On a laptop with no site module to load, `conda install -c conda-forge esmf`
into a *separate* environment (not this one) works fine. NCO is lower-risk
and can go straight into this environment if you want it:

```bash
conda install -c conda-forge nco
```

These are programs rather than Python packages, so pip cannot provide them and
they are not in `pyproject.toml`. Skip them if you only plot and view.

Pure pip works too, if wheels are available for your platform:

```bash
pip install -e ".[dev]"
```

Extras: `plot` pulls in matplotlib and cartopy, `test` pulls in pytest, `dev`
is both plus ruff. The core install has no plotting dependency at all —
geometry, caching and rasterizing are usable headless, and `plot.py` imports
matplotlib lazily so `import gmpas` never pays for it.

Check it landed:

```bash
gmpas --version && pytest -q
```
