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


_CALLS = re.compile(r"[A-Za-z_(][\w.() +*/-]*\(")


class ChatError(ValueError):
    """A request that could not be turned into a script; the message says why.

    `kind` says which way it failed: "said" when the model answered in words
    that it cannot, "refused" when its script was refused twice, "server"
    when the model could not be reached, and "request" for the rest.
    """

    def __init__(self, message: str, kind: str = "request"):
        super().__init__(message)
        self.kind = kind


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
  t.sel(level=850)                 a level, by its value, where the file lists them
Make a field (then draw it with shade):
  spd = hypot(u, v)                also a - b, a * 1000, a / b, diff(a)
Ask where a field is largest or smallest in view. The page answers with the
value and the place and leaves the picture alone, so write this line only:
  snow.max()                       also .min(); .isel() picks a step or level
  mark(snow.max())                 the same, and the place is marked on the map
  mark(lon=147.4, lat=-2.0)        marks a place and shows the value there
Frame the map:
  view(region=(west, east, south, north))
  view(proj="Mercator")            a projection; shade() only
  view(proj="Orthographic", lon0=120, lat0=10)
  view(proj="lonlat")              back to the plain map
Draw one field -- the call chooses the kind of plot:
  shade(x, cmap="viridis", vmin=0, vmax=30, levels="(0,30,2)")   the fast map
    more options: levels="20" (that many colours), reverse=True,
    norm="log" (also "symlog", "power"), center=0, extend="both"
  contourf(x)   contour(x)   pcolormesh(x)   imshow(x)   hist(x)   hovmoller(x)
  series(x, lon=147.4, lat=-2.0)   profile(x, lon=147.4, lat=-2.0)
Draw several things as one figure, in order, the first underneath:
  contourf(t.isel(time=0, level=3), cmap="RdBu_r", levels=[270, 280, 290])
  contour(z.isel(time=0, level=3), colors="black", levels=10)
  vectors(u.isel(time=0, level=3), v.isel(time=0, level=3), stride=4)
  also barbs(u, v) and streamlines(u, v); every layer picks the same time.
  contour options: levels=10 (a count) or [0, 5, 10] (values; [28] is the
    one line at 28), colors="black", linewidths=1,
    labels=True, label_fontsize=8, label_every=2, cmap="viridis"
  contourf options: levels, cmap, vmin, vmax
  vectors options: stride=4 (every 4th point), color="black", scale, colorize=True
  gridvalues(t) writes the numbers on the map.
  The map itself, each a call with options and no field:
    coastlines(color="black", linewidth=0.8, resolution="10m")
    borders()  rivers(color="blue")  lakes()  land(facecolor="lightgray")
    ocean(facecolor="lightblue")  gridlines(labels=True, dlon=10, dlat=10)
    Draw land() and ocean() before the fields they sit under.
  figure(title="...", subtitle="...", projection="Robinson", look="ferret",
         colorbar="right")   projection: PlateCarree, Robinson, Mollweide,
         EqualEarth, Orthographic, NorthPolarStereo, SouthPolarStereo,
         LambertConformal, Mercator; look: matplotlib, grads, ferret
{EXPRESSIONS}  Such a figure takes its
  projection from figure(projection=), not from view(proj=).
"""


def system_prompt(meta: dict) -> str:
    """What the model is told: the task, the language, and the open file."""
    def line(v):
        out = f"  {v['name']}: {v.get('label') or v['name']}"
        if v.get("levels", 1) > 1:
            out += f" ({v['levels']} levels, 0 to {v['levels'] - 1}"
            if v.get("level_values"):
                out += ": " + ", ".join(f"{x:g}" for x in v["level_values"]) \
                    + f" {v.get('level_units') or ''}".rstrip()
            out += ")"
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
    expressions = ("  A layer may be arithmetic on a field: "
                   "contour(skintemp - 273.15).\n") if meta.get("layer_expressions") \
        else "  A layer draws a field of the file, not an expression.\n"
    projections = [p for p in meta.get("projections") or [] if p != "auto"]
    layers = "" if meta.get("layer_schema") else \
        "This file is drawn one field at a time: no layers, no winds.\n"
    return (
        "You turn a request about a map of weather or climate data into a gmpas "
        "script. Reply with the script only: no explanation, no code fence.\n"
        "- Write only the lines that change. Do not repeat the view: what a "
        "script does not name stays as it is.\n"
        "- One field is `shade(x)`, or a `var = x` line: that is the fast map the "
        "user pans and zooms. Use figure(), contours, winds and the map's own "
        "layers only when the request asks for an overlay, a map feature, a "
        "title or a look; they make a still figure.\n"
        "- Contour lines, winds and the map's own layers written alone are drawn "
        "over the field on screen. To put another field underneath, begin with "
        "`var = x` or shade(x); for nothing underneath, begin with `var =`. When "
        "the view already lists draw calls, write all of them again with the "
        "change.\n"
        "- `level = 3` is a level by its number. A level by its value is "
        "`x.sel(level=850)`.\n"
        "- To clear a `key = value` setting such as vmin or levels, write it with "
        "nothing after the equals sign: `vmin =`. A layer is not a setting: to "
        "take one out of a figure, write the figure's other calls again without "
        "it.\n"
        "- Not possible yet: an average, a sum or another statistic over space "
        "or time; saving or animating (the page's export buttons do those); a "
        "cross-section. Say so in one line starting `# `, and do not draw "
        "something else in its place.\n"
        "- Asked why something was not done, or what you can do, answer in one "
        "line starting `# `. Small talk gets one friendly `# ` line that says "
        "what can be asked about the map.\n"
        "- A field is what its label says and nothing else. If the file holds "
        "no wind, do not draw ocean transport or anything else in its place: "
        "say what the file does hold.\n"
        "- If the file does not hold what is asked for, or the request is not "
        "about the map, reply with one line starting `# ` that says why. If only "
        "part can be met, write that part and add a `# ` line saying what was "
        "left out.\n\n"
        f"{_LANGUAGE.replace('{EXPRESSIONS}', expressions)}\n"
        "The view now is given as a script: `key = value` lines for one field, "
        "draw calls for a figure of layers. To change a figure of layers, write "
        "all of its calls again with the change.\n\n"
        f"Variables in this file:\n{variables}\n\n"
        f"{times}.\n{where}{layers}"
        # "auto" is left out: the model cannot know what it would pick, and on
        # a global file it is the plain map already on screen
        f"Projections: {', '.join(projections) or 'none'}.\n"
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
                        f"{exc.error}", "server") from None
    except (KeyError, TypeError):
        raise ChatError("the model server's answer had no message in it",
                        "server") from None
    except Exception as exc:                 # no route, refused, timed out
        raise ChatError(f"no answer from the model server at {cfg['url']}: "
                        f"{exc}", "server") from None


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


#: How many earlier requests go with a new one.
MAX_HISTORY = 6


def ask(text: str, meta: dict, current: str = "", cfg: dict | None = None,
        post=_post, history=None) -> dict:
    """The script for `text`, checked against the file `meta` describes.

    `current` is the view on screen, as a script, so a request such as
    "the same, but zoomed on Africa" has something to mean; `history` is the
    earlier requests of this conversation and what was answered, as
    [{"request": ..., "reply": ...}], so that "no, the other one" does. Returns
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
    messages = [{"role": "system", "content": system_prompt(meta)}]
    said = [h for h in (history or []) if isinstance(h, dict)
            and isinstance(h.get("request"), str) and isinstance(h.get("reply"), str)]
    for turn in said[-MAX_HISTORY:]:
        messages += [{"role": "user", "content": turn["request"][:MAX_TEXT]},
                     {"role": "assistant", "content": turn["reply"][:2000]}]
    messages.append({"role": "user", "content": asked})
    record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "model": cfg["model"],
              "file": meta.get("file"), "request": text, "view": current.strip(),
              "system": messages[0]["content"],
              "history": said[-MAX_HISTORY:], "rounds": []}
    try:
        return _rounds(cfg, messages, meta, post, record, _shown(current, meta))
    except ChatError as exc:
        record["error"], record["outcome"] = str(exc), exc.kind
        raise
    finally:
        _log(cfg, record)


def _shown(current: str, meta: dict):
    """The field the view on screen shows, when its script says so."""
    try:
        return _script.resolve(_script.parse(current), meta).get("var")
    except _script.ScriptError:
        return None


def _fits(state: dict, shown, meta: dict) -> None:
    """What a reply leaves to the view on screen must fit it too: a level set
    with no field named is a level of the field shown."""
    lev = state.get("level")
    fields = {v["name"]: v for v in meta.get("variables", [])}
    if lev is None or "var" in state or shown not in fields:
        return
    count = fields[shown].get("levels", 1)
    if lev >= count:
        raise _script.ScriptError(
            f"level: {lev} is outside {shown}'s {count} level(s), numbered from 0; "
            f"a level by its value is {shown}.sel(level={lev})")


def _rounds(cfg, messages, meta, post, record, shown=None) -> dict:
    problem = ""
    for _ in range(2):
        reply = post(cfg, messages)
        turn = {"reply": reply}
        record["rounds"].append(turn)
        written = script_of(reply)
        settings = [line for line in written.splitlines() if line[0] not in "#!"]
        if not settings:
            said = written.strip().lstrip("#! ").strip() or reply.strip()[:200]
            raise ChatError(said or "the model wrote no script", "said")
        try:
            written, state = _script.read(written, meta, shown)
            _fits(state, shown, meta)
        except _script.ScriptError as exc:
            problem = turn["refused"] = str(exc)
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content":
                          f"That script was refused: {problem}. Write it again, "
                          f"corrected, as script lines only."}]
            continue
        record["script"] = written
        return {"script": written, "state": state}
    raise ChatError(f"the model's script was refused twice: {problem}", "refused")
