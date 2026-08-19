"""Reproduces the numbers in docs/why.md: gmpas vs uxarray vs MPAS-Viewer.

Not part of gmpas's own test suite or dependencies -- this needs two extra
packages gmpas deliberately doesn't depend on:

    pip install uxarray holoviews
    pip install git+https://github.com/jhbravo/mpasviewer.git --no-deps
    # mpasviewer's own pyproject.toml pins `earthcmap`, a GitHub-only
    # package unrelated to the code path this script exercises (it's
    # imported nowhere in mpasviewer's source) -- --no-deps skips it.

Run once per tool, in a *fresh interpreter* each time, and pass --tool. A
driver that imports all three in one process would let whichever imports
matplotlib/cartopy first pay that shared cost, making the other two look
artificially fast -- exactly the kind of thing that makes a benchmark
untrustworthy. Two calls are timed per run: the first (cold: nothing this
process has done yet is reusable) and a second, immediately after, against
a different variable (warm: whatever each tool caches in-process, if
anything, is now warm). A second *process* against the same mesh shows
whether anything survives across runs at all -- only gmpas persists
anything to disk.

    python compare_gmpas_uxarray_mpasviewer.py --tool gmpas      --mesh M --data D --var V
    python compare_gmpas_uxarray_mpasviewer.py --tool mpasviewer --mesh M --data D --var V
    python compare_gmpas_uxarray_mpasviewer.py --tool uxarray    --mesh M --data D --var V

Prints one line of JSON: n_cells, t_import, t_mesh_load, t_frame1, t_frame2.
"""

from __future__ import annotations

import argparse
import json
import time


def bench_gmpas(mesh_path, data_path, var, out_dir, method="auto"):
    t0 = time.perf_counter()
    import matplotlib
    matplotlib.use("Agg")
    from gmpas.mesh import MpasMesh
    from gmpas.data import select
    from gmpas import plot as gplot
    from gmpas.style import Style
    import xarray as xr
    t_import = time.perf_counter() - t0

    t0 = time.perf_counter()
    mesh = MpasMesh.load(mesh_path)
    t_mesh = time.perf_counter() - t0

    ds = xr.open_dataset(data_path, decode_timedelta=False)
    values = select(ds[var], time=0, level=0)

    t0 = time.perf_counter()
    fig, _ = gplot.cell_field(mesh, values, style=Style.preset("paper"),
                              label=var, title=var, method=method)
    fig.savefig(f"{out_dir}/gmpas_1.png", dpi=Style.preset("paper").dpi)
    t_frame1 = time.perf_counter() - t0

    others = [v for v in ds.data_vars if v != var and ds[v].dims == ds[var].dims]
    var2 = others[0] if others else var
    values2 = select(ds[var2], time=0, level=0)
    t0 = time.perf_counter()
    fig2, _ = gplot.cell_field(mesh, values2, style=Style.preset("paper"),
                               label=var2, title=var2, method=method)
    fig2.savefig(f"{out_dir}/gmpas_2.png", dpi=Style.preset("paper").dpi)
    t_frame2 = time.perf_counter() - t0

    return dict(n_cells=int(mesh.n_cells), t_import=t_import, t_mesh_load=t_mesh,
               t_frame1=t_frame1, t_frame2=t_frame2, var1=var, var2=var2,
               method=("raster" if mesh.n_cells >= 150_000 or method == "raster"
                       else "poly") if method == "auto" else method)


def bench_mpasviewer(mesh_path, data_path, var, out_dir):
    t0 = time.perf_counter()
    import matplotlib
    matplotlib.use("Agg")
    from mpasviewer import scvtmesh
    t_import = time.perf_counter() - t0

    t0 = time.perf_counter()
    mpasd = scvtmesh(grid_file=mesh_path, diag_list=data_path)
    # Restricted to one variable: mpasviewer's own variable-type detection
    # (grouping by pressure level etc.) errors on some real MPAS history
    # files -- e.g. a (nCells, nOznLevels, nMonths) ozone climatology field
    # with no Time dimension trips its "has a Time dim" assumption. Passing
    # load_variables explicitly skips that heuristic for the field actually
    # being timed here.
    mpasd.dataset(load_variables=[var])
    dta = mpasd.load()
    t_mesh = time.perf_counter() - t0

    import matplotlib.pyplot as plt
    t0 = time.perf_counter()
    mpasd.show(dta, var_name=var, time_index=0)
    plt.savefig(f"{out_dir}/mpasviewer_1.png", dpi=100)
    plt.close("all")
    t_frame1 = time.perf_counter() - t0

    others = [v for v in dta.data_vars if v != var and dta[v].dims == dta[var].dims]
    var2 = others[0] if others else var
    t0 = time.perf_counter()
    mpasd.show(dta, var_name=var2, time_index=0)
    plt.savefig(f"{out_dir}/mpasviewer_2.png", dpi=100)
    plt.close("all")
    t_frame2 = time.perf_counter() - t0

    return dict(n_cells=int(dta.sizes.get("face") or 0), t_import=t_import,
               t_mesh_load=t_mesh, t_frame1=t_frame1, t_frame2=t_frame2,
               var1=var, var2=var2)


def bench_uxarray(mesh_path, data_path, var, out_dir):
    t0 = time.perf_counter()
    import matplotlib
    matplotlib.use("Agg")
    import uxarray as ux
    import holoviews as hv
    t_import = time.perf_counter() - t0

    t0 = time.perf_counter()
    uxds = ux.open_dataset(mesh_path, data_path)
    t_mesh = time.perf_counter() - t0

    t0 = time.perf_counter()
    plot = uxds[var].isel(Time=0).plot.polygons(backend="matplotlib")
    hv.save(plot, f"{out_dir}/uxarray_1.png", fmt="png")
    t_frame1 = time.perf_counter() - t0

    others = [v for v in uxds.data_vars if v != var and uxds[v].dims == uxds[var].dims]
    var2 = others[0] if others else var
    t0 = time.perf_counter()
    plot2 = uxds[var2].isel(Time=0).plot.polygons(backend="matplotlib")
    hv.save(plot2, f"{out_dir}/uxarray_2.png", fmt="png")
    t_frame2 = time.perf_counter() - t0

    return dict(n_cells=int(uxds.uxgrid.n_face), t_import=t_import, t_mesh_load=t_mesh,
               t_frame1=t_frame1, t_frame2=t_frame2, var1=var, var2=var2)


BENCHERS = {"gmpas": bench_gmpas, "mpasviewer": bench_mpasviewer, "uxarray": bench_uxarray}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tool", required=True, choices=sorted(BENCHERS))
    p.add_argument("--mesh", required=True, help="MPAS grid/mesh netCDF file")
    p.add_argument("--data", required=True,
                   help="MPAS diagnostic/history netCDF file (can be the same as --mesh)")
    p.add_argument("--var", required=True, help="a (Time, nCells) variable in --data")
    p.add_argument("--out-dir", default="/tmp", help="where to write the rendered PNGs")
    p.add_argument("--method", default="auto", choices=["auto", "poly", "raster"],
                   help="gmpas only: force a method instead of the size-based default")
    args = p.parse_args()

    fn = BENCHERS[args.tool]
    kwargs = {"method": args.method} if args.tool == "gmpas" else {}
    result = fn(args.mesh, args.data, args.var, args.out_dir, **kwargs)
    print(json.dumps({"tool": args.tool, **result}))


if __name__ == "__main__":
    main()
