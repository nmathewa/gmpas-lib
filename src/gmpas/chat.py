"""A view asked for in plain words, written as a script by a local model.

    "wind speed at the third level over the Maritime Continent, in Mercator"

becomes

    var    = hypot(u, v)
    level  = 2
    proj   = Mercator
    region = 90, 150, -15, 15

The model is one the user runs: an Ollama server, named in a small file (see
`config_path`). gmpas ships no model and calls no hosted service. Chat is an
optional install -- `pip install "gmpas[chat]"` brings the Ollama client --
and without it the page is as it always was.

The model is told what the open file holds and which keys a script may set,
and is asked for script lines and nothing else. Its reply is text, and it is
treated as text: `gmpas.script` parses it and checks it against the file, so
the worst a wrong reply can do is name a setting that is refused. One refusal
is sent back for a second try. Nothing the model writes is ever evaluated.

The configuration is `key = value` lines, as a script is:

    url   = http://127.0.0.1:11434
    model = qwen2.5-coder:7b

`key` (sent as a bearer token), `timeout` (seconds) and `log` are optional.

Every request is written down, with each reply the model gave and what became
of it, as one JSON line in `log_path()` -- on this machine only. That file is
what a smaller model can later be taught from. `log = off` stops it, and
`log = <file>` moves it.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import time
from pathlib import Path

from . import script as _script

#: The longest request taken, in characters.
MAX_TEXT = 500
DEFAULT_TIMEOUT = 120.0
_KEYS = ("url", "model", "key", "timeout", "log")


_CALLS = re.compile(r"[A-Za-z_]\w*\s*\(")


class ChatError(ValueError):
    """A request that could not be turned into a script; the message says why."""


def installed() -> bool:
    """Whether the chat extra is installed: `pip install "gmpas[chat]"`."""
    return importlib.util.find_spec("ollama") is not None


def log_path(cfg: dict | None = None) -> Path | None:
    """Where requests and replies are written, or None when that is off."""
    named = (cfg or {}).get("log") or os.environ.get("GMPAS_CHAT_LOG") or ""
    if named.lower() in ("off", "no", "false", "0"):
        return None
    if named:
        return Path(named).expanduser()
    base = os.environ.get("XDG_DATA_HOME") or "~/.local/share"
    return Path(base).expanduser() / "gmpas" / "chat.jsonl"


def _log(cfg: dict, record: dict) -> None:
    """One request, as one line. A log that cannot be written is not a reason
    to refuse the request."""
    path = log_path(cfg)
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def config_path() -> Path:
    """Where the configuration is read from: `$GMPAS_CHAT_CONFIG`, or
    `chat.conf` in gmpas's configuration directory."""
    named = os.environ.get("GMPAS_CHAT_CONFIG")
    if named:
        return Path(named).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME") or "~/.config"
    return Path(base).expanduser() / "gmpas" / "chat.conf"


def load_config(path=None) -> dict | None:
    """The configuration, or None when there is no file. Raises ChatError for
    a file that is there and cannot be used."""
    path = Path(path) if path else config_path()
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ChatError(f"{path}: {exc.strerror}") from None
    out: dict = {}
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line[0] in "#!":
            continue
        name, eq, value = line.partition("=")
        name, value = name.strip().lower(), value.strip().strip("\"'")
        if not eq or name not in _KEYS:
            raise ChatError(f"{path}, line {n}: expected one of "
                            f"{', '.join(_KEYS)} = value")
        out[name] = value
    for name in ("url", "model"):
        if not out.get(name):
            raise ChatError(f"{path}: {name} is not set")
    try:
        out["timeout"] = float(out.get("timeout") or DEFAULT_TIMEOUT)
    except ValueError:
        raise ChatError(f"{path}: timeout takes a number of seconds") from None
    out["url"] = out["url"].rstrip("/")
    return out


_LANGUAGE = """\
A script is a few lines that read like Python with xarray. It is not run as
Python: only the forms below exist. Counting is from 0; time=-1 is the last step.

Pick from a field:
  t.isel(time=0, level=3)          a step and a level, by number
  t.sel(time="2012-03-01 12:00")   a step, by its label
Make a field (then draw it with shade):
  spd = hypot(u, v)                also a - b, a * 1000, a / b, diff(a)
Frame the map:
  view(region=(west, east, south, north))
  view(proj="Mercator")            a projection; shade() only
  view(proj="Orthographic", lon0=120, lat0=10)
Draw one field -- the call chooses the kind of plot:
  shade(x, cmap="viridis", vmin=0, vmax=30, levels="(0,30,2)")   the fast map
  contourf(x)   contour(x)   pcolormesh(x)   imshow(x)   hist(x)   hovmoller(x)
  series(x, lon=147.4, lat=-2.0)   profile(x, lon=147.4, lat=-2.0)
Draw several things as one figure, in order, the first underneath:
  contourf(t.isel(time=0, level=3), cmap="RdBu_r", levels=[270, 280, 290])
  contour(z.isel(time=0, level=3), colors="black", levels=10)
  vectors(u.isel(time=0, level=3), v.isel(time=0, level=3), stride=4)
  also barbs(u, v) and streamlines(u, v); a layer draws a field of the file,
  not an expression, and every layer picks the same time. Such a figure is on
  the plain lon/lat map: give no proj with it. To overlay on what is shown
  now, draw the variable the view names first, then the overlay.
"""


def system_prompt(meta: dict) -> str:
    """What the model is told: the task, the language, and the open file."""
    def line(v):
        out = f"  {v['name']}: {v.get('label') or v['name']}"
        if v.get("levels", 1) > 1:
            out += f" ({v['levels']} levels, 0 to {v['levels'] - 1})"
        if v.get("kinds"):
            out += f"; plots: {', '.join(v['kinds'])}"
        return out

    variables = "\n".join(line(v) for v in meta.get("variables", []))
    labels = meta.get("labels") or []
    steps = meta.get("steps", len(labels))
    times = f"{steps} time step(s), 0 to {max(steps - 1, 0)}"
    if labels:
        times += f"; the first is {labels[0]}, the last {labels[-1]}"
    home = meta.get("home")
    where = (f"The data covers longitude {home[0]:g} to {home[1]:g} and latitude "
             f"{home[2]:g} to {home[3]:g}; give regions in that longitude range.\n"
             if home else "")
    layers = "" if meta.get("layer_schema") else \
        "This file is drawn one field at a time: no layers, no winds.\n"
    return (
        "You turn a request about a map of weather or climate data into a gmpas "
        "script. Reply with the script only: no explanation, no code fence. Write "
        "only what the request changes; when it asks for a different plot or an "
        "overlay, write every draw call the figure needs. If the request cannot "
        "be met, reply with one line starting `# ` that says why.\n\n"
        f"{_LANGUAGE}\n"
        "The view now is given as `key = value` lines; those keys may also be "
        "set that way.\n\n"
        f"Variables in this file:\n{variables}\n\n"
        f"{times}.\n{where}{layers}"
        f"Projections: {', '.join(meta.get('projections') or []) or 'none'}.\n"
        f"Palettes: {', '.join(meta.get('cmaps') or [])}.\n")


def _post(cfg: dict, messages: list[dict]) -> str:
    """One round with the model: the reply's text."""
    try:
        import ollama
    except ImportError:
        raise ChatError('chat is not installed: pip install "gmpas[chat]"') from None
    headers = {"Authorization": f"Bearer {cfg['key']}"} if cfg.get("key") else {}
    try:
        client = ollama.Client(host=cfg["url"], headers=headers, timeout=cfg["timeout"])
        got = client.chat(model=cfg["model"], messages=messages, stream=False,
                          options={"temperature": 0})
        return got["message"]["content"] or ""
    except ollama.ResponseError as exc:
        raise ChatError(f"the model server answered {exc.status_code}: "
                        f"{exc.error}") from None
    except (KeyError, TypeError):
        raise ChatError("the model server's answer had no message in it") from None
    except Exception as exc:                 # no route, refused, timed out
        raise ChatError(f"no answer from the model server at {cfg['url']}: "
                        f"{exc}") from None


def script_of(reply: str) -> str:
    """The script in a model's reply: what is inside a code fence if it used
    one, and otherwise the reply without the lines that are plainly prose."""
    lines = reply.strip().splitlines()
    fences = [i for i, line in enumerate(lines) if line.strip().startswith("```")]
    if len(fences) >= 2:
        lines = lines[fences[0] + 1:fences[1]]
    kept, depth = [], 0
    for line in lines:
        text = line.strip()
        if not text:
            continue
        # inside an open bracket a line is the rest of a call, whatever it holds
        if depth > 0 or text[0] in "#!" or "=" in text or _CALLS.match(text):
            kept.append(text)
            depth += text.count("(") + text.count("[") - text.count(")") - text.count("]")
    return "\n".join(kept) + ("\n" if kept else "")


def ask(text: str, meta: dict, current: str = "", cfg: dict | None = None,
        post=_post) -> dict:
    """The script for `text`, checked against the file `meta` describes.

    `current` is the view on screen, as a script, so a request such as
    "the same, but zoomed on Africa" has something to mean. Returns
    {"script": the text, "state": what `gmpas.script.resolve` made of it}.
    Raises ChatError when there is no usable script after one retry.
    """
    text = " ".join(text.split())
    if not text:
        raise ChatError("ask for a view in a sentence")
    if len(text) > MAX_TEXT:
        raise ChatError(f"a request is at most {MAX_TEXT} characters")
    if cfg is None:
        cfg = load_config()
    if cfg is None:
        raise ChatError(f"no model is configured; write {config_path()}")
    asked = text if not current.strip() else \
        f"The view now:\n{current.strip()}\n\nRequest: {text}"
    messages = [{"role": "system", "content": system_prompt(meta)},
                {"role": "user", "content": asked}]
    record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "model": cfg["model"],
              "file": meta.get("file"), "request": text, "view": current.strip(),
              "system": messages[0]["content"], "rounds": []}
    try:
        return _rounds(cfg, messages, meta, post, record)
    except ChatError as exc:
        record["error"] = str(exc)
        raise
    finally:
        _log(cfg, record)


def _rounds(cfg, messages, meta, post, record) -> dict:
    problem = ""
    for _ in range(2):
        reply = post(cfg, messages)
        turn = {"reply": reply}
        record["rounds"].append(turn)
        written = script_of(reply)
        settings = [line for line in written.splitlines() if line[0] not in "#!"]
        if not settings:
            said = written.strip().lstrip("#! ").strip() or reply.strip()[:200]
            raise ChatError(said or "the model wrote no script")
        try:
            state = _script.resolve(_script.parse(written), meta)
        except _script.ScriptError as exc:
            problem = turn["refused"] = str(exc)
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content":
                          f"That script was refused: {problem}. Write it again, "
                          f"corrected, as script lines only."}]
            continue
        record["script"] = written
        return {"script": written, "state": state}
    raise ChatError(f"the model's script was refused twice: {problem}")
