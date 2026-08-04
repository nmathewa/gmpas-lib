# On a cluster

The viewer serves on `127.0.0.1`, so forward the port rather than exporting a
display — this is the case where ncview's X11 forwarding hurts most:

```bash
ssh -L 8765:localhost:8765 user@cluster
```

then on the node, inside your interactive job:

```bash
gmpas view /scratch/run/ --no-browser --port 8765
```

and open `http://localhost:8765` in your own browser.

Two environment variables matter on a shared machine:

- `GMPAS_CACHE_DIR` — where cached mesh geometry goes. Defaults to
  `~/.cache/gmpas/mesh`; point it at scratch if your home quota is small.
  The cache is memory-mapped, so several processes reading the same mesh share
  one copy through the page cache.
- `GMPAS_DATA_DIR` — tried first when resolving relative paths.

Batch rendering scales with `-j`, but two things limit it: worker startup
(cheap under `fork` on Linux, expensive under `spawn` on macOS), and the
KD-tree query inside each worker still requesting every core, which
oversubscribes when many workers run at once. Measured on 12 steps: 10.2 s at
`-j 1`, 5.7 s at `-j 4`, 6.2 s at `-j 10`. Until the query's worker count is
plumbed through, `-j` around half your cores is the sweet spot.
