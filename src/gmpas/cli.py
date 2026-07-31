"""Command line entry points: inspect, plot, and view MPAS output."""

from __future__ import annotations

import argparse
import sys

from . import __version__


def _info(args) -> int:
    from .data import plottable
    from .mesh import MpasMesh
    from .series import Series

    series = None
    if args.mesh_only:
        mesh, ds = MpasMesh.load(args.path), None
    else:
        series = Series(args.path, args.mesh or "")
        mesh, ds = series.mesh, series.first

    width = mesh.cell_width_km
    print(f"# {mesh.path.name}")
    print(f"cells     : {mesh.n_cells:,}")
    print(f"edges     : {mesh.n_edges:,}")
    print(f"kind      : {'global' if mesh.is_global else 'regional'} "
          f"({mesh.coverage * 100:.1f}% of its sphere)")
    lo, hi, la, lb = mesh.extent
    print(f"extent    : lon {lo:.2f} .. {hi:.2f}, lat {la:.2f} .. {lb:.2f}")
    print(f"cell size : {width.min():.1f} .. {width.max():.1f} km "
          f"(ratio {width.max() / width.min():.1f}x)")
    print(f"radius    : {mesh.sphere_radius:,.0f} m")
    if series is not None and series.n_files > 1:
        print(f"series    : {len(series)} steps across {series.n_files} files "
              f"({series.labels[0]} .. {series.labels[-1]})")

    if ds is not None:
        print()
        for dim, names in plottable(ds).items():
            if not names:
                continue
            print(f"## on {dim} ({len(names)})")
            for n in names[:args.limit]:
                print(f"  {n} {tuple(ds[n].dims)}  {ds[n].attrs.get('units', '')}".rstrip())
            if len(names) > args.limit:
                print(f"  ... and {len(names) - args.limit} more")
    if series is not None:
        series.close()
    return 0


def _render_one(job):
    """Render a single step. Top level so it survives being sent to a worker."""
    import matplotlib
    matplotlib.use("Agg")

    from .data import field_label
    from .plot import cell_field, edge_field
    from .series import Series
    from .style import Style, save_figure

    paths, mesh_path, var, step, opts = job
    series = Series(paths, mesh_path)
    try:
        da = series.dataarray(var, step)
        values = series.values(var, step=step, level=opts["level"])
        style = Style.preset(opts["style"])
        label = field_label(da)
        title = opts["title"] or f"{var} — {series.labels[step]}"
        common = dict(label=label, title=title, style=style, cmap=opts["cmap"],
                      extent=opts["extent"] or None)

        if "nCells" in da.dims:
            fig, _ = cell_field(series.mesh, values, method=opts["method"],
                                symmetric=opts["symmetric"], **common)
        else:
            fig, _ = edge_field(series.mesh, values,
                                symmetric=opts["symmetric"], **common)
        out = save_figure(fig, opts["pattern"].format(step=step, var=var),
                          style=style)
    finally:
        series.close()
    return str(out)


def _plot_series(args) -> int:
    """Render every step, in parallel.

    Each worker builds its own KD-tree and view geometry, which sounds
    wasteful but is not: the mesh cache is memory-mapped, so every process
    shares one copy of the geometry through the page cache rather than each
    reading its own.
    """
    import os
    from multiprocessing import Pool

    from .series import Series

    series = Series(args.path, args.mesh or "")
    n = len(series)
    series.close()

    jobs = args.jobs or os.cpu_count() or 1
    jobs = max(1, min(jobs, n))
    opts = {
        "level": args.level, "style": args.style, "cmap": args.cmap,
        "extent": args.extent, "symmetric": args.symmetric,
        "method": args.method, "title": args.title, "pattern": args.out,
    }
    work = [(args.path, args.mesh or "", args.var, s, opts) for s in range(n)]

    print(f"rendering {n} steps with {jobs} worker(s)", file=sys.stderr)
    if jobs == 1:
        for job in work:
            print(_render_one(job))
        return 0

    # one BLAS/OpenMP thread per worker, or the processes fight for the cores
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    with Pool(jobs) as pool:
        for out in pool.imap_unordered(_render_one, work):
            print(out)
    return 0


def _plot(args) -> int:
    if args.all_steps:
        # "{step" not "{step}" -- a format spec like {step:04d} is the normal
        # case and does not contain the bare placeholder
        if "{step" not in args.out:
            print("gmpas: --all-steps needs {step} in --out, "
                  "e.g. -o 'frames/mslp_{step:04d}.png'", file=sys.stderr)
            return 1
        return _plot_series(args)
    return _plot_single(args)


def _plot_single(args) -> int:
    from .data import field_label, plottable, spatial_dim
    from .plot import cell_field, edge_field
    from .series import Series
    from .style import Style, save_figure

    series = Series(args.path, args.mesh or "")
    mesh = series.mesh
    if args.var not in series.first:
        print(f"{args.var!r} not in {args.path}. Available on cells: "
              f"{plottable(series.first)['nCells'][:20]}", file=sys.stderr)
        series.close()
        return 1

    da = series.dataarray(args.var, args.time)
    values = series.values(args.var, step=args.time, level=args.level)
    dim = spatial_dim(da)
    kwargs = dict(label=field_label(da), title=args.title or args.var,
                  style=Style.preset(args.style), cmap=args.cmap,
                  extent=args.extent or None, symmetric=args.symmetric)
    series.close()

    if dim == "nCells":
        fig, _ = cell_field(mesh, values, method=args.method, **kwargs)
    elif dim == "nEdges":
        kwargs.pop("symmetric")
        fig, _ = edge_field(mesh, values, symmetric=args.symmetric, **kwargs)
    else:
        print(f"{args.var!r} lives on {dim}; only cell and edge fields plot.",
              file=sys.stderr)
        return 1

    print(save_figure(fig, args.out, style=Style.preset(args.style)))
    return 0


def _view(args) -> int:
    from .viewer import serve

    serve(args.path, args.mesh or "", port=args.port,
          nx=args.width, ny=args.height, open_browser=not args.no_browser)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gmpas",
        description="Plot MPAS output on its own native mesh, without regridding.",
    )
    p.add_argument("--version", action="version", version=f"gmpas {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("path", help="MPAS output or mesh file")
        sp.add_argument("-m", "--mesh", help="mesh file, if not alongside the data")

    i = sub.add_parser("info", help="summarise a mesh and its variables")
    common(i)
    i.add_argument("--mesh-only", action="store_true",
                   help="treat the file as a mesh and skip the variable listing")
    i.add_argument("--limit", type=int, default=40, help="variables to list per group")
    i.set_defaults(func=_info)

    o = sub.add_parser("plot", help="render one field to a PNG")
    common(o)
    o.add_argument("var", help="variable to plot")
    o.add_argument("-o", "--out", default="plot.png", help="output path")
    o.add_argument("-t", "--time", type=int, default=0)
    o.add_argument("-l", "--level", type=int, default=0)
    o.add_argument("--method", default="auto", choices=["auto", "poly", "raster"])
    o.add_argument("--cmap", default="")
    o.add_argument("--extent", default="", help="named region, or blank to fit the mesh")
    o.add_argument("--style", default="paper",
                   choices=["paper", "poster", "notebook", "mesh"])
    o.add_argument("--symmetric", action="store_true",
                   help="diverging colormap centred on zero, for anomalies")
    o.add_argument("--title", default="")
    o.add_argument("--all-steps", action="store_true",
                   help="render every step in the series; --out needs {step}")
    o.add_argument("-j", "--jobs", type=int, default=0,
                   help="parallel workers for --all-steps (default: all cores)")
    o.set_defaults(func=_plot)

    v = sub.add_parser("view", help="browse a file interactively in a browser")
    common(v)
    v.add_argument("-p", "--port", type=int, default=8765)
    v.add_argument("--width", type=int, default=1200, help="raster width in pixels")
    v.add_argument("--height", type=int, default=700)
    v.add_argument("--no-browser", action="store_true",
                   help="do not open a browser (useful over an SSH tunnel)")
    v.set_defaults(func=_view)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"gmpas: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
