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

## Trying a chat model

`tests/chat_eval.py` asks the configured chat model (see
[configuration](configuration.md)) about 150 requests once each -- plain,
misspelt, chatty, written as GrADS, Ferret or xarray would be told, about
places, about things the file does not hold -- and judges each answer by what
it would leave on screen. It needs a model and is run by hand:

```bash
python tests/chat_eval.py --list     # the cases; nothing is called
python tests/chat_eval.py            # the 100 core cases, one request each
python tests/chat_eval.py --all      # every case
```

It opens no data and keeps its requests out of the chat log. The cases and
the judging are themselves tested, without a model, in `test_chat_eval.py`.
