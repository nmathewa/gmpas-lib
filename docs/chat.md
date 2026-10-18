# Scripting and chat

## Scripting

<video width="100%" autoplay loop muted playsinline controls>
  <source src="_static/gmpas-scripting.webm" type="video/webm">
  <source src="_static/gmpas-scripting.mp4" type="video/mp4">
</video>

*A script, an xarray-style layered figure and `cape.max()` on the MPAS file.*

Or directly from the CLI:

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
| `time` | step number (0 first, -1 last) or the label (e.g. `2012-02-05 12:00`) |
| `level` | level number (0 first), or a pressure such as `850 hPa` |
| `kind` | `map`, `contourf`, `contour`, `pcolormesh`, `hist`, `series`, ... |
| `point` | `lon, lat` for a series or profile |
| `cmap`, `vmin`, `vmax` | palette and colour range |
| `proj`, `plon`, `plat` | projection and its centre |
| `region` | `west, east, south, north` in degrees |

### xarray style


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

| command |  |
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

> Note that this is not a Python script; it is parsed directly by gmpas.

## Chat


You can also use the optional LLM-based chat tool.

To install:
```bash
pip install "gmpas[chat]"
```

<video width="100%" autoplay loop muted playsinline controls>
  <source src="_static/gmpas-chat.webm" type="video/webm">
  <source src="_static/gmpas-chat.mp4" type="video/mp4">
</video>

*Requests answered by gpt-oss:120b, named in `chat.conf`, on the demo MPAS file.*

- A request in words (e.g. "wind speed over Africa, in Mercator").
- The model is one you name in `~/.config/gmpas/chat.conf` (or `$GMPAS_CHAT_CONFIG`).
- You can configure the API key, chat engine (e.g. ollama, anthropic or openai) and model selection in `chat.conf`.

**Example**
```
provider = ollama
url      = http://127.0.0.1:11434
model    = qwen2.5-coder:7b
```

| key | value |
|---|---|
| `provider` | `ollama` (default), `openai`, `anthropic` |
| `url` | the server; for `openai` the API base, e.g. `http://127.0.0.1:8000/v1` |
| `model` | a model the server has |
| `key` | optional; else `$GMPAS_CHAT_KEY` |
| `timeout` | seconds, 120 |
| `agent` | `on` (default) or `off` |
| `log` | a file, or `off`; default `~/.local/share/gmpas/chat.jsonl` |

> The chat runs on the gmpas scripting tool (no Python calls are involved).
