"""Command line entry points: inspect, plot, and view MPAS output."""

from __future__ import annotations

import argparse
import faulthandler
import io
import sys
import time
from pathlib import Path

from . import __version__

#: only the default port is allowed to wander when busy
DEFAULT_PORT = 8765

BANNER = r"""
    __    __    __
   /  \__/  \__/  \     g m p a s   {version}
   \__/  \__/  \__/
   /  \__/  \__/  \     MPAS output on its own mesh
   \__/  \__/  \__/
"""


def banner() -> str:
    return BANNER.format(version=__version__)


def human(n: float) -> str:
    """Bytes, at a glance."""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


class Progress:
    """A bar when someone is watching, periodic lines when nobody is.

    Under a scheduler stdout is a log file, and a carriage-returning bar just
    fills it with thousands of partial lines. So the same information is
    emitted either way, in whichever shape suits the destination.
    """

    def __init__(self, total: int, width: int = 32, every: int = 10):
        self.total = total
        self.width = width
        self.every = every            # percent between lines when not a tty
        self.done = 0
        self.t0 = time.perf_counter()
        self.tty = sys.stdout.isatty()
        self._last = -1

    def advance(self, label: str = "") -> None:
        self.done += 1
        elapsed = time.perf_counter() - self.t0
        frac = self.done / self.total if self.total else 1.0
        eta = (elapsed / self.done) * (self.total - self.done) if self.done else 0

        if self.tty:
            filled = int(self.width * frac)
            bar = "#" * filled + "-" * (self.width - filled)
            sys.stdout.write(
                f"\r  [{bar}] {self.done}/{self.total} {frac * 100:3.0f}%  "
                f"{elapsed / self.done:.1f}s/file  eta {clock(eta)}   "
            )
            sys.stdout.flush()
        else:
            pct = int(frac * 100)
            if pct // self.every > self._last // self.every or self.done == self.total:
                self._last = pct
                print(f"  {self.done}/{self.total} ({pct}%)  "
                      f"{elapsed / self.done:.1f}s/file  eta {clock(eta)}")

    def close(self) -> None:
        if self.tty:
            sys.stdout.write("\r" + " " * (self.width + 60) + "\r")
            sys.stdout.flush()


def clock(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, sec = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{sec:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


EXAMPLES = """
examples:
  gmpas info  run/history.2012-02-25_12.00.00.nc
  gmpas info  run/ --limit 10

  gmpas plot  run/history.2012-02-25_12.00.00.nc precipw -o pw.png
  gmpas plot  run/ theta -l 20 --cmap turbo -o theta20.png
  gmpas plot  'run/history.*.nc' precipw --all-steps -o 'frames/pw_{step:04d}.png' -j 8

  gmpas scrip run/init.nc -o mesh.scrip.nc  for conservative remapping weights
  gmpas remap  run/history.*.nc -o out/     the whole conversion, one command
  gmpas target -o dst.scrip.nc              reads target_domain from this directory
  gmpas target run/history.nc               and which fields would be remapped

  gmpas view  run/                          browse interactively in a browser
  gmpas view  run/ --host 0.0.0.0 --no-browser   on an HPC compute node

  gmpas prep hfun     hfun.py --check     the mesh you are about to build
  gmpas prep generate hfun.py -o mesh/    run JIGSAW and build it

on a cluster the job runs on a compute node but your tunnel lands on the login
node, so bind all interfaces and tunnel to the node by name:

  compute node:  gmpas view /scratch/run/ --host 0.0.0.0 --no-browser
  your machine:  ssh -N -L 8765:<compute-node>:8765 <login-node>
                 then open http://localhost:8765

Any path may be a file, a directory, or a glob. A directory or glob is read as
one time series across files, which is how MPAS writes output.

environment:
  GMPAS_CACHE_DIR   where cached mesh geometry goes (default ~/.cache/gmpas/mesh)
  GMPAS_DATA_DIR    tried first when resolving relative paths
  JIGSAWDIR         the jigsaw executable, or the directory holding it
  MKGRIDFILE        the mkgrid executable, or the directory holding it
                    both required by `gmpas prep generate`
"""


def _shown(paths) -> str:
    return paths[0] if len(paths) == 1 else f"{paths[0]} (+{len(paths) - 1} more)"


def _info(args) -> int:
    from .data import plottable
    from .mesh import MpasMesh
    from .series import Series

    series = None
    if args.mesh_only:
        mesh, ds = MpasMesh.load(args.path[0]), None
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
        print(f"{args.var!r} not in {_shown(args.path)}. Available on cells: "
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


def _scrip(args) -> int:
    from .scrip import coverage_of, write_scrip

    out, wrapped = write_scrip(args.path[0], args.out)
    frac = coverage_of(out)
    print(out)
    print(f"covers {frac * 100:.1f}% of the sphere")
    if wrapped:
        print(f"normalised {wrapped:,} longitudes onto [0, 2pi) — the source "
              f"file mixed conventions")
    print("\nnext, generate conservative weights with one of:")
    print(f"  ESMF_RegridWeightGen -s {out.name} -d dst.scrip.nc "
          f"-w map.nc -m conserve")
    print(f"  ncremap -s {out.name} -g dst.nc -m map.nc -a aave")
    print("then apply them; gmpas does not compute weights itself.")
    return 0


def _target(args) -> int:
    """Report the discovered configuration, and write the destination grid."""
    from .config import CONFIG_NAMES, discover

    cfg = discover(args.dir, warn=False)
    where = cfg.directory
    print(f"# configuration in {where}")
    for role, name in CONFIG_NAMES.items():
        mark = "found  " if role in cfg.found else "absent "
        print(f"  {mark} {name}")

    domain = cfg.require_domain()          # raises with guidance when absent
    lat, lon = domain.lats(), domain.lons()
    print(f"\ntarget: {domain}")
    print(f"  {domain.size:,} cells")
    print(f"  covers  lon {lon[0] - domain.dlon / 2:.4f} .. "
          f"{lon[-1] + domain.dlon / 2:.4f}, "
          f"lat {lat[0] - domain.dlat / 2:.4f} .. "
          f"{lat[-1] + domain.dlat / 2:.4f}   (as requested)")
    print(f"  centres lon {lon[0]:.4f} .. {lon[-1]:.4f}, "
          f"lat {lat[0]:.4f} .. {lat[-1]:.4f}   "
          f"(inset half a cell, {domain.dlat / 2:.4f} deg)")

    if cfg.include or cfg.exclude:
        print()
        if cfg.include:
            print(f"include: {len(cfg.include)} field(s)")
        if cfg.exclude:
            print(f"exclude: {len(cfg.exclude)} field(s)")

    if args.path:
        from .series import Series
        series = Series(args.path, args.mesh or "")
        try:
            available = list(series.first.variables)
            selected, notes = cfg.select(available, warn=False)
        finally:
            series.close()
        sys.stdout.flush()
        for note in notes:
            print(f"  warning: {note}", file=sys.stderr)
        sys.stderr.flush()
        print(f"\nwould remap {len(selected)} field(s):")
        for name in selected:
            print(f"  {name}")

    if args.out:
        import shutil

        out = domain.to_scrip(args.out)
        print(f"\nwrote {out}")
        print("  pair it with the source grid:")
        print("    gmpas scrip <mesh>.nc -o src.scrip.nc")
        print(f"    ESMF_RegridWeightGen -s src.scrip.nc -d {out.name} "
              f"-w map.nc -m conserve --src_regional --dst_regional "
              f"--ignore_unmapped")
        if shutil.which("ESMF_RegridWeightGen") is None:
            sys.stdout.flush()
            print("\n  ESMF_RegridWeightGen is not on your PATH — install it with:",
                  file=sys.stderr)
            print("    conda install -c conda-forge esmf nco", file=sys.stderr)
            sys.stderr.flush()
    return 0


def _resolve_mesh(args, cfg, series):
    """Explicit flag, then mesh_file, then a self-describing file."""
    from .mesh import has_mesh

    if args.mesh:
        return Path(args.mesh), "--mesh"
    if cfg.mesh is not None:
        if not cfg.mesh.exists():
            raise SystemExit(f"gmpas: mesh_file names {cfg.mesh}, which does "
                             f"not exist")
        return cfg.mesh, "mesh_file"
    if has_mesh(series.first):
        return series.files[0], "the data file itself"
    return series.mesh.path, "found beside the data"


def _remap(args) -> int:
    """Read the config, build weights once, convert every file."""
    from .config import CONFIG_NAMES, discover
    from .remap import (RemapError, Weights, detect_cores,
                        ensure_weights, remap_many)
    from .series import Series

    print(banner())
    cfg = discover(args.dir)
    domain = cfg.require_domain()
    print(f"[1/3] configuration from {cfg.directory}")
    for role, name in CONFIG_NAMES.items():
        mark = "found " if role in cfg.found else "absent"
        print(f"        {mark}  {name}")
    print(f"  target: {domain}")
    print("  opening the run — building mesh geometry if it is not cached")

    series = Series(args.path, args.mesh or (str(cfg.mesh) if cfg.mesh else ""))
    try:
        mesh_path, how = _resolve_mesh(args, cfg, series)
        available = list(series.first.variables)
        fields, notes = cfg.select(available, warn=False)

        print(f"  input : {len(series.files)} file(s), "
              f"{series.mesh.n_cells:,} cells")
        print(f"  mesh  : {Path(mesh_path).name}  (from {how})")
        print(f"  fields: {len(fields)} selected of {len(available)} available")
        for note in notes:
            print(f"  warning: {note}")
        if not fields:
            raise SystemExit("gmpas: no fields selected — check include_fields")

        work = Path(args.out)
        print(f"\n[2/3] weights")
        weights_path, built = ensure_weights(
            mesh_path, domain, work, method=args.method, force=args.force_weights
        )
        weights = Weights.load(weights_path)
        if weights.n_a != series.mesh.n_cells:
            raise SystemExit(
                f"gmpas: weights are for {weights.n_a:,} cells but the mesh has "
                f"{series.mesh.n_cells:,}. Delete {weights_path.name} to rebuild."
            )
    finally:
        series.close()

    n = len(series.files)
    todo = [(src, work / f"{src.stem}.remap.nc") for src in series.files]
    if not args.overwrite:
        existing = [(a, b) for a, b in todo if b.exists()]
        todo = [(a, b) for a, b in todo if not b.exists()]
    else:
        existing = []

    detected, source = detect_cores()
    workers = args.jobs if args.jobs > 0 else detected
    workers = max(1, min(workers, len(todo) or 1))

    print(f"\n[3/3] remapping {len(todo)} file(s) -> {work}/")
    if existing:
        print(f"  {len(existing)} already done — pass --overwrite to redo")
    if args.jobs > 0:
        print(f"  {workers} worker(s)  (-j {args.jobs}; "
              f"{detected} detected from {source})")
    else:
        print(f"  {workers} worker(s)  (detected from {source})")
        if workers == 1 and len(todo) > 1:
            import os
            machine = os.cpu_count() or 1
            print(f"        {source} says 1. If this job has more cores, set "
                  f"them with -j")
            print(f"        (this machine reports {machine}; "
                  f"-j {machine} would use them all)")

    t0 = time.perf_counter()
    total_slabs = 0
    written: list[Path] = []
    failures: list[tuple[str, str]] = []
    first_reported = False
    bar = Progress(len(todo)) if todo else None

    jobs = [(src, out, domain, fields) for src, out in todo]
    for info in remap_many(jobs, weights, weights_path, workers=workers):
        if bar:
            bar.advance()
        if info.get("error"):
            failures.append((info["source"], info["error"]))
            continue
        written.append(info["out"])
        total_slabs += info["slabs"]
        if not first_reported:
            first_reported = True
            if bar:
                bar.close()
            for name, why in info["skipped"]:
                print(f"  not remapped — {name}: {why}")
            print(f"  conservation error {info['conservation']:.1e} (0 is exact)")
    if bar:
        bar.close()

    for name, why in failures:
        print(f"  failed — {name}: {why}", file=sys.stderr)

    dt = time.perf_counter() - t0
    size = sum(p.stat().st_size for p in written if p.exists())
    print(f"\n{'-' * 60}")
    print("generated:")
    print(f"  {weights_path}  ({human(weights_path.stat().st_size)})"
          f"{'  [new]' if built else '  [reused]'}")
    for extra in ("src.scrip.nc", "dst.scrip.nc"):
        q = work / extra
        if q.exists():
            print(f"  {q}  ({human(q.stat().st_size)})")
    print(f"  {len(written)} remapped file(s) in {work}/  ({human(size)} total)")
    if failures:
        print(f"  {len(failures)} file(s) failed")
    print(f"\n{total_slabs:,} slabs in {clock(dt)}"
          + (f"  ({dt / len(written):.2f}s per file, "
             f"{len(written) / dt:.1f} files/s)" if written and dt else ""))
    return 1 if failures else 0


def _dashboard(args, data_path=None, mesh_path="", hfun_path="") -> int:
    """Serve whatever sources were asked for, on one port.

    Every command that opens a browser goes through here, so a run, the mesh it
    is on and the distance function behind that mesh are one server and one
    tunnel rather than three. An explicit --port is usually a tunnel already
    pointing at it, so listening elsewhere would be worse than failing.
    """
    from .dashboard import build, serve

    sources, banner = build(data_path, mesh_path, hfun_path,
                            nx=args.width, ny=args.height)
    serve(sources, port=args.port, host=args.host,
          open_browser=not args.no_browser,
          strict_port=args.port != DEFAULT_PORT, banner=banner)
    return 0


def _view(args) -> int:
    return _dashboard(args, data_path=args.path, mesh_path=args.mesh or "",
                      hfun_path=args.hfun or "")


def _prep_view(args) -> int:
    return _dashboard(args, mesh_path=args.mesh_file,
                      hfun_path=args.hfun or "")


def _prep_hfun(args) -> int:
    from .prep.hfunview import HfunViewer, report

    # --check is the whole point on a login node or in a script: the numbers
    # without a server, so a bad transition is caught before JIGSAW runs
    if args.check:
        print(report(HfunViewer(args.hfun_file)))
        return 0

    return _dashboard(args, mesh_path=args.mesh or "",
                      hfun_path=args.hfun_file)


def _prep_generate(args) -> int:
    from .prep.generate import generate, next_steps

    print(banner())
    print(f"generating a mesh from {args.hfun_file}")
    result = generate(args.hfun_file, out_dir=args.out, jigsaw=args.jigsaw,
                      qlim=args.qlim, init=args.init, force=args.force,
                      allow_steep=args.allow_steep, mkgrid=args.mkgrid,
                      skip_mkgrid=args.skip_mkgrid)
    print(next_steps(result, args.out))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gmpas",
        description="Plot MPAS output on its own native mesh, without regridding.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"gmpas {__version__}")
    # not required: a bare `gmpas` prints help rather than an argparse error
    sub = p.add_subparsers(dest="cmd", metavar="{info,plot,view}")

    def common(sp):
        # nargs="+" so an unquoted glob works too: the shell expands it into
        # many arguments, and Series.expand already accepts a list
        sp.add_argument("path", nargs="+", metavar="PATH",
                        help="file, directory, or glob (quoted or not)")
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

    c = sub.add_parser("scrip", help="write the mesh as a SCRIP grid file")
    common(c)
    c.add_argument("-o", "--out", default="mesh.scrip.nc", help="output path")
    c.set_defaults(func=_scrip)

    t = sub.add_parser("target", help="show the discovered config and write "
                                     "the destination grid")
    t.add_argument("path", nargs="*", help="optional data file, to list the "
                                           "fields that would be remapped")
    t.add_argument("-m", "--mesh", help="mesh file, if not alongside the data")
    t.add_argument("-d", "--dir", help="where to look for the config files "
                                       "(default: the working directory)")
    t.add_argument("-o", "--out", help="write the target grid as SCRIP here")
    t.set_defaults(func=_target)

    r = sub.add_parser("remap", help="conservatively remap a run to a lat-lon grid")
    common(r)
    r.add_argument("-o", "--out", default="remapped",
                   help="directory for the weights and the output files")
    r.add_argument("-d", "--dir", help="where to look for the config files "
                                       "(default: the working directory)")
    r.add_argument("--method", default="conserve",
                   choices=["conserve", "conserve2nd"],
                   help="ESMF regrid method (default: first-order conservative)")
    r.add_argument("--force-weights", action="store_true",
                   help="rebuild map.nc even if it already exists")
    r.add_argument("--overwrite", action="store_true",
                   help="rewrite output files that already exist")
    r.add_argument("-j", "--jobs", type=int, default=0,
                   help="parallel workers (default: the cores this job was "
                        "given, from the scheduler or the affinity mask)")
    r.set_defaults(func=_remap)

    v = sub.add_parser("view", help="browse a file interactively in a browser")
    common(v)
    v.add_argument("--hfun", help="also serve the JIGSAW hfun.py behind this "
                                  "mesh, as a third page on the same port")
    v.add_argument("-p", "--port", type=int, default=DEFAULT_PORT,
                   help=f"port (default {DEFAULT_PORT}; the default wanders "
                        f"if busy, an explicit one fails instead)")
    v.add_argument("--host", default="127.0.0.1",
                   help="interface to bind. Use 0.0.0.0 on an HPC compute "
                        "node so a tunnel from the login node can reach it")
    v.add_argument("--width", type=int, default=1200, help="raster width in pixels")
    v.add_argument("--height", type=int, default=700)
    v.add_argument("--no-browser", action="store_true",
                   help="do not open a browser (useful over an SSH tunnel)")
    v.set_defaults(func=_view)

    # -- preprocessing ---------------------------------------------------
    # Everything above is postprocessing: it opens a run and renders, remaps or
    # exports it. `prep` is the other end of the pipeline -- building a mesh and
    # looking at it before any model output exists -- so it gets its own
    # namespace rather than more top-level verbs. Mesh generation (#15) belongs
    # here next, as `gmpas prep generate`.
    pre = sub.add_parser("prep", help="preprocessing: inspect and build meshes",
                         description="Preprocessing steps, which run before "
                                     "there is any model output to plot.")
    presub = pre.add_subparsers(dest="prep_cmd", metavar="{view,hfun,generate}")
    # a bare `gmpas prep` should list its steps, not error
    pre.set_defaults(func=lambda _a, _p=pre: (_p.print_help(), 0)[1])

    pv = presub.add_parser("view", help="browse a mesh file on its own, "
                                        "with no output data")
    pv.add_argument("mesh_file", metavar="MESH", help="MPAS mesh file")
    pv.add_argument("--hfun", help="also serve the JIGSAW hfun.py behind this "
                                   "mesh, so intent and result are one click "
                                   "apart")
    pv.add_argument("-p", "--port", type=int, default=DEFAULT_PORT,
                    help=f"port (default {DEFAULT_PORT}; the default wanders "
                         f"if busy, an explicit one fails instead)")
    pv.add_argument("--host", default="127.0.0.1",
                    help="interface to bind. Use 0.0.0.0 on an HPC compute "
                         "node so a tunnel from the login node can reach it")
    pv.add_argument("--width", type=int, default=1200,
                    help="raster width in pixels")
    pv.add_argument("--height", type=int, default=700)
    pv.add_argument("--no-browser", action="store_true",
                    help="do not open a browser (useful over an SSH tunnel)")
    pv.set_defaults(func=_prep_view)

    ph = presub.add_parser("hfun", help="browse a JIGSAW distance function "
                                        "before any mesh exists")
    ph.add_argument("hfun_file", metavar="HFUN",
                    help="a JIGSAW hfun.py defining hfun_min and "
                         "get_hfun(lon, lat); a directory is read as its "
                         "hfun.py")
    ph.add_argument("--check", action="store_true",
                    help="print the resolution report and exit, without "
                         "serving anything")
    ph.add_argument("--mesh", help="also serve a mesh built from this hfun, "
                                   "to compare what was asked for against "
                                   "what came out")
    ph.add_argument("-p", "--port", type=int, default=DEFAULT_PORT,
                    help=f"port (default {DEFAULT_PORT}; the default wanders "
                         f"if busy, an explicit one fails instead)")
    ph.add_argument("--host", default="127.0.0.1",
                    help="interface to bind. Use 0.0.0.0 on an HPC compute "
                         "node so a tunnel from the login node can reach it")
    ph.add_argument("--width", type=int, default=1200,
                    help="raster width in pixels")
    ph.add_argument("--height", type=int, default=700)
    ph.add_argument("--no-browser", action="store_true",
                    help="do not open a browser (useful over an SSH tunnel)")
    ph.set_defaults(func=_prep_hfun)

    pg = presub.add_parser("generate", help="run JIGSAW to build a mesh from "
                                            "an hfun.py")
    pg.add_argument("hfun_file", metavar="HFUN",
                    help="a JIGSAW hfun.py defining hfun_min and "
                         "get_hfun(lon, lat)")
    pg.add_argument("-o", "--out", default="mesh",
                   help="directory for the JIGSAW files (default: mesh/)")
    pg.add_argument("--jigsaw", help="the jigsaw executable, or the directory "
                                     "holding it. Overrides $JIGSAWDIR, which "
                                     "is otherwise required")
    pg.add_argument("--mkgrid", help="the mkgrid executable, or the directory "
                                     "holding it. Overrides $MKGRIDFILE, which "
                                     "is otherwise required")
    pg.add_argument("--skip-mkgrid", action="store_true",
                    help="stop after the Save* files, without building "
                         "grid.nc (then $MKGRIDFILE is not needed)")
    pg.add_argument("--init", help="a JIGSAW mesh file of initial points, for "
                                   "a quasi-uniform mesh with icosahedral "
                                   "structure (INIT_FILE)")
    pg.add_argument("--qlim", type=float, default=0.9375,
                    help="JIGSAW's OPTM_QLIM mesh-quality limit "
                         "(default 0.9375)")
    pg.add_argument("--force", action="store_true",
                    help="regenerate even if MESH.msh already exists")
    pg.add_argument("--allow-steep", action="store_true",
                    help="generate even if the cell size gradient is above "
                         "the guideline")
    pg.set_defaults(func=_prep_generate)
    return p


def main(argv=None) -> int:
    # A fatal signal is not an exception and cannot be caught below: SIGBUS
    # from a memory-mapped page the filesystem could not supply kills the
    # process where it stands, and all the user sees is `Bus error`. This
    # installs a C-level handler that prints the Python traceback first, so
    # such a death at least says which line and which array (issue #19).
    # It writes to a real file descriptor, which a captured or wrapped stderr
    # does not have -- and a missing crash dump is not worth failing over.
    try:
        faulthandler.enable()
    except (AttributeError, ValueError, io.UnsupportedOperation):
        pass

    # Python block-buffers stdout when it is not a terminal, so under a job
    # scheduler or through a pipe none of the progress appears until the run
    # ends -- which looks exactly like a command producing no output at all.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):        # not a regular stream
        pass

    parser = build_parser()
    args = parser.parse_args(argv)

    # `gmpas` on its own is a request to see what it can do, not a mistake
    if getattr(args, "func", None) is None:
        parser.print_help()
        return 0

    from .mesh import MeshCacheError
    from .prep.generate import GenerateError
    from .remap import RemapError

    try:
        return args.func(args)
    except (RemapError, MeshCacheError, GenerateError) as exc:
        print(f"\ngmpas: {exc}", file=sys.stderr)
        return 1
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"gmpas: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
