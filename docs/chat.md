# Scripting and chat

## Scripting

A view written as text, in the viewer's script box or from a file:

```bash
gmpas view /path/to/run/ --script view.txt
```

```
# 850 hPa wind speed over the Maritime Continent
var    = hypot(u, v)
time   = 12
level  = 850 hPa
cmap   = viridis
levels = (0,20,2)(inf)
proj   = Mercator
region = 90, 150, -15, 15
```

- One `key = value` per line; `#` starts a comment; `key =` clears a setting.
- Only the keys named change; everything else on the page stays.

| key | value |
|---|---|
| `var` | a variable, or an expression such as `a - b`, `hypot(u, v)` |
| `time` | step number (0 first, -1 last) or its label |
| `level` | level number (0 first), or a pressure such as `850 hPa` |
| `kind` | `map`, `contourf`, `contour`, `pcolormesh`, `hist`, `series`, ... |
| `point` | `lon, lat` for a series or profile |
| `cmap`, `vmin`, `vmax` | palette and colour range |
| `proj`, `plon`, `plat` | projection and its centre |
| `region` | `west, east, south, north` in degrees |

### xarray style

The same view, the way it is written with xarray. The two forms mix.

```python
spd = hypot(u, v)
view(region=(90, 150, -15, 15), proj="Mercator")
shade(spd.isel(time=12).sel(level=850), cmap="viridis", levels="(0,20,2)(inf)")
```

```python
contourf(t.isel(time=0, level=3), cmap="RdBu_r")
contour(z.isel(time=0, level=3), colors="black")
vectors(u.isel(time=0, level=3), v.isel(time=0, level=3))
rivers(color="blue")
figure(title="925 hPa", look="ferret")
```

| form | does |
|---|---|
| `.isel(time=, level=)` | pick by number |
| `.sel(time="...", level=850)` | pick by label or value |
| `shade()` | the fast map |
| `contourf()`, `contour()`, `pcolormesh()`, `imshow()`, `hist()`, `hovmoller()` | other plots; two or more make a figure of layers |
| `series()`, `profile()` | at `lon=`, `lat=` |
| `vectors()` | winds |
| `coastlines()`, `borders()`, `land()`, `ocean()`, `lakes()`, `rivers()`, `gridlines()` | map features |
| `view()`, `figure()` | frame, projection, title |
| `snow.max()`, `snow.min()` | largest / smallest value in view, and where |
| `mark()`, `mark(lon=, lat=)` | mark a place on the map |

- Parsed, never run as Python. Errors name the line.

## Chat

```bash
pip install "gmpas[chat]"
gmpas chat-model        # optional: fetch the built-in model now (about 160 MB)
```

- A request in words ("wind speed over Africa, in Mercator") becomes script lines.
- Without `gmpas[chat]` there is no chat panel.
- With it, requests are answered by gmpas-small, the built-in model, on this machine.
- The reply is checked like a typed script; nothing is run as code.
- `GMPAS_CHAT_MODEL_DIR` sets where the built-in model lives (copy it there on an offline node).

### A larger model

Name it in `~/.config/gmpas/chat.conf` (or `$GMPAS_CHAT_CONFIG`):

```
provider = ollama
url      = http://127.0.0.1:11434
model    = qwen2.5-coder:7b
```

| key | value |
|---|---|
| `provider` | `ollama` (default), `openai`, `anthropic`, `builtin` |
| `url` | the server; for `openai` the API base, e.g. `http://127.0.0.1:8000/v1` |
| `model` | a model the server has |
| `key` | optional; else `$GMPAS_CHAT_KEY` |
| `timeout` | seconds, 120 |
| `agent` | `on` (default) or `off` |
| `log` | a file, or `off`; default `~/.local/share/gmpas/chat.jsonl` |
