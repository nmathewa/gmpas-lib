# Tests

```bash
pytest
```

Tests build synthetic MPAS mesh files in `tmp_path`, so no model output is
needed and the real cache directory is never touched. They pin down the parts
a reader cannot eyeball: the ragged 1-based `verticesOnCell` fill, the
antimeridian unwrap, the `sphere_radius = 1` redimensionalisation, cache
identity, and the factor-of-two bias in the edge-normal wind reconstruction.
Rendering tests skip themselves when the `plot` extra is absent.
