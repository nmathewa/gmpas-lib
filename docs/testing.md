# Tests


## Trying a chat model

```bash
python tests/chat_eval.py --list     # the cases; nothing is called
python tests/chat_eval.py            # the 100 core cases, one request each
python tests/chat_eval.py --all      # every case
```

## Timing

```bash
GMPAS_TIMING=1 gmpas info run/      # stage times
GMPAS_TIMING=2 gmpas view run/      # + per-frame stages and memory
```

```
gmpas.timing  mesh.discover      0.013s  scanned=13 opened=12
gmpas.timing  series.scan        0.012s  files=13 opened=12
gmpas.timing  mesh.tree_build    0.006s  cells=300
```

- `GMPAS_TIMING_FILE=<file>` writes the lines to a file instead of stderr.
