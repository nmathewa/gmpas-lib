# Configuration

Two environment variables, both optional:

- `GMPAS_CACHE_DIR` — where cached `.npz` mesh geometry goes.
  Defaults to `~/.cache/gmpas/mesh`. Safe to delete; it rebuilds.
- `GMPAS_DATA_DIR` — tried first when resolving a relative path, before the
  working directory.
