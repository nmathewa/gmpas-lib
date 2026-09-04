# On a cluster

`gmpas view` never opens a browser. It prints a URL and, when that URL is on
another machine, the exact `ssh` command that reaches it — copy both. Nothing
here needs a display, and forwarding a port beats exporting one: this is the
case where ncview's X11 forwarding hurts most.

## On a login node

```bash
gmpas view /scratch/run/ --port 8765
```

It prints the tunnel command with your username and this node's name already
filled in. Run that on your own machine:

```bash
ssh -N -L 8765:localhost:8765 you@login.cluster.edu
```

and open `http://localhost:8765` in your own browser. `-N` means "no remote
command" — it just forwards, and holds the terminal until you stop it.

## Inside a batch job

A compute node is not reachable from outside, and your tunnel lands on the
login node — where `127.0.0.1` is a *different machine's* loopback. So the
default bind is the one thing that cannot work there:

```bash
gmpas view /scratch/run/ --host 0.0.0.0 --port 8765
```

gmpas notices it is inside a PBS or Slurm allocation, and prints a tunnel that
hops through the node you submitted from, naming the compute node explicitly:

```bash
ssh -N -L 8765:dec1042:8765 you@derecho7
```

Bound to `0.0.0.0` the viewer is reachable by anything that can route to the
node, which on most clusters is everyone else logged in
([issue 1](https://github.com/nmathewa/gmpas-lib/issues/1)). It serves your
run read-only and holds no credentials, but pick a nonstandard `--port` and
stop it when you are done.

If you run it inside a job *without* `--host 0.0.0.0`, gmpas says so rather
than printing a tunnel command that cannot land.

## A note on ports

An explicit `--port` is strict: if it is busy, gmpas fails instead of quietly
listening somewhere else. That is deliberate — your tunnel is pinned to that
number, and a viewer that wandered to 8766 looks exactly like a dead server
from the browser. Without `--port` it starts at 8765 and moves on if busy.

## If you want a browser opened anyway

`--browser` asks for it. On a login node it usually finds no graphical browser
and hands the URL to whatever the system default is, which is often a terminal
browser that takes over the terminal the server is logging to — so it is off
by default, and gmpas refuses to launch one that would do that.

Environment variables that matter on a shared machine:

- `GMPAS_CACHE_DIR` — where cached mesh geometry goes. Defaults to
  `~/.cache/gmpas/mesh`; point it at scratch if your home quota is small.
  The cache is memory-mapped, so several processes reading the same mesh share
  one copy through the page cache.
- `GMPAS_DATA_DIR` — tried first when resolving relative paths.
- `GMPAS_VALUES_CACHE_MB` (default 512) and `GMPAS_VIEW_CACHE_MB` (default 256)
  — memory budgets, in megabytes. Lower them on a login node with a small
  cgroup limit; see [configuration.md](configuration.md).
- `GMPAS_TIMING=1` with `GMPAS_TIMING_FILE=timing.log` — when something is
  slow, this says which stage. On a run directory holding thousands of files,
  expect `mesh.discover` and `series.scan` to dominate a cold start: both open
  every file in the directory, and every open is a round trip to the metadata
  server.

Batch rendering scales with `-j`, but two things limit it: worker startup
(cheap under `fork` on Linux, expensive under `spawn` on macOS), and the
KD-tree query inside each worker still requesting every core, which
oversubscribes when many workers run at once. Measured on 12 steps: 10.2 s at
`-j 1`, 5.7 s at `-j 4`, 6.2 s at `-j 10`. Until the query's worker count is
plumbed through, `-j` around half your cores is the sweet spot.
