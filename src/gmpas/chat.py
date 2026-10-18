"""A view asked for in plain words, written as a script by a local model.

    "wind speed at the third level over the Maritime Continent, in Mercator"

becomes

    var    = hypot(u, v)
    level  = 2
    proj   = Mercator
    region = 90, 150, -15, 15

The model is the one named in a small file (see `config_path`): an Ollama
server by default, or any server that speaks the OpenAI chat API (vLLM,
llama.cpp, LM Studio, a hosted one), or Anthropic's API. gmpas ships no model
and talks to no server but that one. Chat is an optional install --
`pip install "gmpas[chat]"` brings the Ollama client -- and without it the
page is as it always was.

The model is told what the open file holds and which keys a script may set,
and is asked for script lines and nothing else. Its reply is text, and it is
treated as text: `gmpas.script` parses it and checks it against the file, so
the worst a wrong reply can do is name a setting that is refused. One refusal
is sent back for a second try. Nothing the model writes is ever evaluated.

The configuration is `key = value` lines, as a script is:

    url   = http://127.0.0.1:11434
    model = qwen2.5-coder:7b

`provider` is `ollama` (the default), `openai` -- `url` is then the API's base,
such as http://127.0.0.1:8000/v1 -- or `anthropic`, whose `url` may be left
out. `key` (sent as the server expects it), `timeout` (seconds) and `log` are
optional; with no `key`, `$GMPAS_CHAT_KEY` is used. `think = off` asks an
Ollama model not to reason before it answers, and caps the reply's length: a
small model can otherwise reason for thousands of tokens over a two-line script.

Every request is written down, with each reply the model gave and what became
of it, as one JSON line in `log_path()` -- on this machine only. That file is
what a smaller model can later be taught from. `log = off` stops it, and
`log = <file>` moves it.

Agent mode. With `agent = on` (the default, whatever the provider) the model
may take a few steps before it answers, by calling tools the same three ways
over: Ollama's `tools` and `tool_calls`, the OpenAI API's `tools` and `tool`
replies, Anthropic's `tool_use` and `tool_result` blocks. The tools are
`describe_file` (what the file holds), `check_script` (parse and check a
script against the file and the view, nothing drawn), and `apply` (the final
script, with a note for the user if there is one). A plain-text reply ends the
run too, as in the one-shot path. There is no tool that computes a value:
"where is it largest" is a question the page answers once the script is
applied. At most `steps` rounds of tool calls are taken (default 6, each
request bounded by `timeout`); a model that is still calling tools after that
is asked once more the one-shot way. The script it ends with goes through the
same checker and the same single retry as ever, and nothing it says is
evaluated. `agent = off` keeps the one-shot path exactly, which is also the
only path for a model that cannot call tools. Every tool call and its result
is kept in the log.
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
_KEYS = ("provider", "url", "model", "key", "timeout", "log", "think", "agent", "steps")
PROVIDERS = ("ollama", "openai", "anthropic")
_ANTHROPIC_URL = "https://api.anthropic.com"
_ANTHROPIC_VERSION = "2023-06-01"
#: The longest reply asked of a server that needs a limit (Anthropic's).
MAX_REPLY_TOKENS = 2048
#: How many rounds of tool calls an agent run may take.
DEFAULT_STEPS = 6
MAX_STEPS = 20
_ON = ("on", "yes", "true", "1")
_OFF = ("off", "no", "false", "0")


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
    """Whether chat is installed: `pip install "gmpas[chat]"`. Without it
    there is no chat at all -- no panel, no configuration read, no server
    asked -- and the page is as it always was."""
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
    provider = out.get("provider", "ollama").lower()
    if provider not in PROVIDERS:
        raise ChatError(f"{path}: provider is one of {', '.join(PROVIDERS)}")
    if "provider" in out:
        out["provider"] = provider
    if provider == "anthropic" and not out.get("url"):
        out["url"] = _ANTHROPIC_URL
    for name in ("url", "model"):
        if not out.get(name):
            raise ChatError(f"{path}: {name} is not set")
    try:
        out["timeout"] = float(out.get("timeout") or DEFAULT_TIMEOUT)
    except ValueError:
        raise ChatError(f"{path}: timeout takes a number of seconds") from None
    out["url"] = out["url"].rstrip("/")
    named = out.get("agent", "on").lower()
    if named not in _ON + _OFF:
        raise ChatError(f"{path}: agent is on or off")
    out["agent"] = named in _ON
    if "steps" in out:
        try:
            out["steps"] = int(out["steps"])
        except ValueError:
            out["steps"] = 0
        if not 1 <= out["steps"] <= MAX_STEPS:
            raise ChatError(f"{path}: steps is a whole number from 1 to {MAX_STEPS}")
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
{TIME}Ask where a field is largest or smallest in view. The page answers with the
value and the place and leaves the picture alone, so write this line only:
  snow.max()                       also .min(); .isel() picks a step or level
  mark(snow.max())                 the same, and the place is marked on the map
  mark(lon=147.4, lat=-2.0)        marks a place and shows the value there
Frame the map:
  view(region=(west, east, south, north))
  view()                           the whole area again
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


_OVER_TIME = """\
A field reduced over time is a field too (draw it, combine it, ask about it):
  sst.sel(month=12).mean("time")           every December, averaged
  sst.sel(season="DJF").mean("time")       a season: DJF, MAM, JJA, SON
  sst.sel(year=2012).max("time")           also .min("time"); month and year
                                           take a list: month=[6, 7, 8]
  sst.sel(time="2012").mean("time")        a date, as far as it is given
  sst.sel(time=slice("2012-12", "2013-02")).mean("time")   both ends included
  sst.isel(time=slice(0, 12)).mean("time")                 by number
  sst.mean("time")                         every step
  An anomaly is a field minus its mean: sst - sst.sel(month=12).mean("time").
  One step by its date needs no reduction: sst.sel(time="2012-12").
  Reduce a field of the file, then combine: not (a - b).mean("time").
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
    over_time = _OVER_TIME if meta.get("time_statistics") else \
        "  (No average over time on this file: say so if one is asked for.)\n"
    language = _LANGUAGE.replace("{EXPRESSIONS}", expressions) \
                        .replace("{TIME}", over_time)
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
        "- Not possible yet: an average or a sum over space; a trend; a sum or "
        "a standard deviation over time; saving or animating (the page's export "
        "buttons do those); a cross-section. Say so in one line starting `# `, "
        "and do not draw something else in its place.\n"
        "- A request that leaves a choice open -- any projection, another "
        "palette -- is met by picking one, not by asking which.\n"
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
        f"{language}\n"
        "The view now is given as a script: `key = value` lines for one field, "
        "draw calls for a figure of layers. To change a figure of layers, write "
        "all of its calls again with the change.\n\n"
        f"Variables in this file:\n{variables}\n\n"
        f"{times}.\n{where}{layers}"
        # "auto" is left out: the model cannot know what it would pick, and on
        # a global file it is the plain map already on screen
        f"Projections: {', '.join(projections) or 'none'}.\n"
        f"Palettes: {', '.join(meta.get('cmaps') or [])}.\n")


def _key(cfg: dict) -> str:
    return cfg.get("key") or os.environ.get("GMPAS_CHAT_KEY") or ""


def _post(cfg: dict, messages: list[dict]) -> str:
    """One round with the model: the reply's text."""
    provider = cfg.get("provider", "ollama")
    if provider == "openai":
        return _post_openai(cfg, messages)
    if provider == "anthropic":
        return _post_anthropic(cfg, messages)
    try:
        import ollama
    except ImportError:
        raise ChatError('chat is not installed: pip install "gmpas[chat]"') from None
    key = _key(cfg)
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    try:
        client = ollama.Client(host=cfg["url"], headers=headers, timeout=cfg["timeout"])
        options = {"temperature": 0}
        extra = {}
        think = cfg.get("think", "").lower()
        if think in ("off", "no", "false", "0"):
            extra["think"] = False
            options["num_predict"] = MAX_REPLY_TOKENS
        elif think:
            extra["think"] = True if think in ("on", "yes", "true", "1") else think
        got = client.chat(model=cfg["model"], messages=messages, stream=False,
                          options=options, **extra)
        return got["message"]["content"] or ""
    except ollama.ResponseError as exc:
        raise ChatError(f"the model server answered {exc.status_code}: "
                        f"{_shown_url(str(exc.error))}", "server") from None
    except (KeyError, TypeError):
        raise ChatError("the model server's answer had no message in it",
                        "server") from None
    except Exception as exc:                 # no route, refused, timed out
        raise ChatError(f"no answer from the model server at {_shown_url(cfg['url'])}: "
                        f"{_shown_url(str(exc))}", "server") from None


def _post_json(cfg: dict, url: str, headers: dict, body: dict) -> dict:
    """POST `body` as JSON and read the JSON answer; any failure is a ChatError."""
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=cfg["timeout"]) as got:
            return json.loads(got.read())
    except urllib.error.HTTPError as exc:
        said = exc.read().decode("utf-8", "replace")[:300]
        try:
            err = json.loads(said).get("error", said)
            said = err.get("message", said) if isinstance(err, dict) else str(err)
        except (ValueError, AttributeError):
            pass
        raise ChatError(f"the model server answered {exc.code}: {said}", "server") from None
    except ValueError:
        raise ChatError("the model server's answer was not JSON", "server") from None
    except Exception as exc:                 # no route, refused, timed out
        raise ChatError(f"no answer from the model server at {_shown_url(cfg['url'])}: "
                        f"{_shown_url(str(exc))}", "server") from None


def _shown_url(text: str) -> str:
    """`text` with any user:password@ taken out of the addresses in it."""
    return re.sub(r"(\w+://)[^/@\s]*@", r"\1", text)


def _post_openai(cfg: dict, messages: list[dict]) -> str:
    """The OpenAI chat API, which most servers speak: vLLM, llama.cpp, LM Studio."""
    key = _key(cfg)
    got = _post_json(cfg, cfg["url"] + "/chat/completions",
                     {"Authorization": f"Bearer {key}"} if key else {},
                     {"model": cfg["model"], "messages": messages, "temperature": 0,
                      "stream": False})
    try:
        return got["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        raise ChatError("the model server's answer had no message in it",
                        "server") from None


def _post_anthropic(cfg: dict, messages: list[dict]) -> str:
    """Anthropic's Messages API: the system prompt goes beside the turns."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    turns = [m for m in messages if m["role"] != "system"]
    got = _post_json(cfg, cfg["url"] + "/v1/messages",
                     {"x-api-key": _key(cfg), "anthropic-version": _ANTHROPIC_VERSION},
                     {"model": cfg["model"], "system": system, "messages": turns,
                      "max_tokens": MAX_REPLY_TOKENS, "temperature": 0})
    try:
        return "".join(b.get("text", "") for b in got["content"] if b.get("type") == "text")
    except (KeyError, TypeError, AttributeError):
        raise ChatError("the model server's answer had no message in it",
                        "server") from None


# ------------------------------------------------------------------- agent

#: What the model may call, in JSON Schema; every provider is given the same.
TOOLS = [
    {"name": "describe_file",
     "description": "What the open file holds: its fields with their levels and "
                    "level values, its time steps and their labels, the area it "
                    "covers, the projections and palettes on offer.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "check_script",
     "description": "Check a script against the file and the view on screen "
                    "without drawing anything. Returns what the script would "
                    "set, or why it is refused.",
     "parameters": {"type": "object", "required": ["script"], "properties": {
         "script": {"type": "string", "description": "the script lines"}}}},
    {"name": "apply",
     "description": "Give the final script, which ends the run. It is checked "
                    "once more and then applied to the page.",
     "parameters": {"type": "object", "required": ["script"], "properties": {
         "script": {"type": "string", "description": "the script lines"},
         "message": {"type": "string",
                     "description": "a line for the user, if there is one to say"}}}},
]

_AGENT_PROMPT = (
    "\n\nYou may call tools before you answer. describe_file lists what the file "
    "holds; check_script says whether a script is accepted and what it would "
    "set, without drawing it. Check a script you are not sure of, and correct it "
    "if it is refused. Finish by calling apply with the final script (and a "
    "message if something could not be done), or by replying with the script "
    "lines as text. Nothing is drawn until you finish.\n")


def _args(value) -> dict:
    """A tool call's arguments as a dict; some servers send them as JSON text."""
    if isinstance(value, str):
        try:
            value = json.loads(value or "{}")
        except ValueError:
            raise ValueError("the arguments were not JSON") from None
    if not isinstance(value, dict):
        raise ValueError("the arguments were not an object")
    return value


def _describe_file(meta: dict) -> str:
    """The file `meta` describes, in words: what describe_file answers."""
    out = ["Fields:"]
    for v in meta.get("variables", []):
        line = f"  {v['name']}: {v.get('label') or v['name']}"
        if v.get("levels", 1) > 1:
            line += f"; {v['levels']} levels, numbered 0 to {v['levels'] - 1}"
            if v.get("level_values"):
                line += " = " + ", ".join(f"{x:g}" for x in v["level_values"]) \
                    + f" {v.get('level_units') or ''}".rstrip()
        if v.get("kinds"):
            line += f"; plots: {', '.join(v['kinds'])}"
        out.append(line)
    labels = meta.get("labels") or []
    steps = meta.get("steps", len(labels))
    times = f"{steps} time step(s), numbered 0 to {max(steps - 1, 0)}"
    if labels and len(labels) <= 40:
        times += ": " + ", ".join(labels)
    elif labels:
        times += f"; the first are {', '.join(labels[:3])} and the last " \
                 f"{', '.join(labels[-3:])}"
    out.append(times)
    home = meta.get("home")
    if home:
        out.append(f"Area: longitude {home[0]:g} to {home[1]:g}, "
                   f"latitude {home[2]:g} to {home[3]:g}")
    projections = [p for p in meta.get("projections") or [] if p != "auto"]
    out.append(f"Projections: {', '.join(projections) or 'none'}")
    out.append(f"Palettes: {', '.join(meta.get('cmaps') or [])}")
    return "\n".join(out)


def _summary(state: dict, meta: dict) -> str:
    """What a checked script would set, one thing to a line."""
    fields = {v["name"]: v for v in meta.get("variables", [])}
    labels = meta.get("labels") or []
    out = []
    for key in ("var", "time", "level", "kind", "proj", "region", "cmap", "vmin", "vmax"):
        value = state.get(key)
        if value is None:
            continue
        note = ""
        if key == "time" and 0 <= value < len(labels):
            note = f" ({labels[value]})"
        owner = fields.get(state.get("var")) or {}
        values = owner.get("level_values") or []
        if key == "level" and 0 <= value < len(values):
            units = owner.get("level_units") or ""
            note = f" ({values[value]:g} {units})".replace(" )", ")")
        out.append(f"{key} = {value or 'lonlat' if key == 'proj' else value}{note}")
    ask = state.get("ask")
    if ask:
        big = "largest" if ask["what"] == "max" else "smallest"
        out.append(f"question: where {ask['var']} is {big} "
                   f"(the page answers it; nothing is drawn)")
    layers = (state.get("layers") or {}).get("layers")
    if layers:
        out.append("layers: " + ", ".join(str(x.get("kind")) for x in layers))
    return "\n".join(out) or "nothing is changed"


def _check_script(text, meta: dict, shown) -> str:
    """check_script: the script read the way the page reads it."""
    if not isinstance(text, str):
        raise ValueError("script is the script's text")
    written = script_of(text)
    if not written.strip():
        return "refused: there is no script line in that"
    try:
        written, state = _script.read(written, meta, shown)
        _fits(state, shown, meta)
    except _script.ScriptError as exc:
        return f"refused: {exc}"
    return "accepted. It would set:\n" + _summary(state, meta)


def _run_tool(call: dict, meta: dict, shown) -> str:
    """What a tool answers. A fault in the call is an answer too -- the model
    reads it and tries again -- never an exception."""
    try:
        args = _args(call["args"])
        if call["name"] == "describe_file":
            return _describe_file(meta)
        if call["name"] == "check_script":
            return _check_script(args.get("script"), meta, shown)
    except ValueError as exc:
        return f"error: {exc}"
    return f"error: there is no tool {call['name']!r}; the tools are " \
           f"{', '.join(t['name'] for t in TOOLS)}"


def _tool_list(provider: str) -> list[dict]:
    if provider == "anthropic":
        return [{"name": t["name"], "description": t["description"],
                 "input_schema": t["parameters"]} for t in TOOLS]
    return [{"type": "function", "function": t} for t in TOOLS]


def _as_wire(provider: str, messages: list[dict]) -> list[dict]:
    """The turns so far -- {"role", "content"}, an assistant's "calls" and a
    tool's "id" and "name" -- in the shape one provider's API takes."""
    out: list[dict] = []
    for m in messages:
        if m["role"] == "assistant" and m.get("calls"):
            calls = m["calls"]
            if provider == "anthropic":
                blocks = [{"type": "text", "text": m["content"]}] if m["content"] else []
                blocks += [{"type": "tool_use", "id": c["id"], "name": c["name"],
                            "input": c["args"]} for c in calls]
                out.append({"role": "assistant", "content": blocks})
            elif provider == "openai":
                out.append({"role": "assistant", "content": m["content"] or None,
                            "tool_calls": [
                                {"id": c["id"], "type": "function",
                                 "function": {"name": c["name"],
                                              "arguments": json.dumps(c["args"])}}
                                for c in calls]})
            else:
                out.append({"role": "assistant", "content": m["content"],
                            "tool_calls": [{"function": {"name": c["name"],
                                                         "arguments": c["args"]}}
                                           for c in calls]})
        elif m["role"] == "tool":
            if provider == "anthropic":
                block = {"type": "tool_result", "tool_use_id": m["id"],
                         "content": m["content"]}
                last = out[-1]["content"] if out and out[-1]["role"] == "user" else None
                if isinstance(last, list):
                    last.append(block)                   # results go in one turn
                else:
                    out.append({"role": "user", "content": [block]})
            elif provider == "openai":
                out.append({"role": "tool", "tool_call_id": m["id"],
                            "content": m["content"]})
            else:
                out.append({"role": "tool", "tool_name": m["name"],
                            "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def _talk(cfg: dict, messages: list[dict]) -> dict:
    """One round with the model, tools on offer: {"text": ..., "calls":
    [{"id", "name", "args"}]}. Arguments that were not JSON come back as the
    text, for `_run_tool` to refuse."""
    provider, key = cfg.get("provider", "ollama"), _key(cfg)
    wire = _as_wire(provider, messages)
    if provider == "anthropic":
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        got = _post_json(cfg, cfg["url"] + "/v1/messages",
                         {"x-api-key": key, "anthropic-version": _ANTHROPIC_VERSION},
                         {"model": cfg["model"], "system": system,
                          "messages": [m for m in wire if m["role"] != "system"],
                          "tools": _tool_list(provider), "max_tokens": MAX_REPLY_TOKENS,
                          "temperature": 0})
        try:
            blocks = got["content"]
            return {"text": "".join(b.get("text", "") for b in blocks
                                    if b.get("type") == "text"),
                    "calls": [{"id": b["id"], "name": b["name"], "args": b.get("input", {})}
                              for b in blocks if b.get("type") == "tool_use"]}
        except (KeyError, TypeError, AttributeError):
            raise ChatError("the model server's answer had no message in it",
                            "server") from None
    auth = {"Authorization": f"Bearer {key}"} if key else {}
    if provider == "openai":
        got = _post_json(cfg, cfg["url"] + "/chat/completions", auth,
                         {"model": cfg["model"], "messages": wire,
                          "tools": _tool_list(provider), "temperature": 0, "stream": False})
        try:
            message = got["choices"][0]["message"]
            calls = [{"id": c.get("id") or f"call_{i}", "name": c["function"]["name"],
                      "args": c["function"].get("arguments") or {}}
                     for i, c in enumerate(message.get("tool_calls") or [])]
        except (KeyError, IndexError, TypeError, AttributeError):
            raise ChatError("the model server's answer had no message in it",
                            "server") from None
    else:
        options, extra = {"temperature": 0}, {}
        think = cfg.get("think", "").lower()
        if think in _OFF:
            extra["think"] = False
            options["num_predict"] = MAX_REPLY_TOKENS
        elif think:
            extra["think"] = True if think in _ON else think
        got = _post_json(cfg, cfg["url"] + "/api/chat", auth,
                         {"model": cfg["model"], "messages": wire,
                          "tools": _tool_list(provider), "stream": False,
                          "options": options, **extra})
        try:
            message = got["message"]
            calls = [{"id": f"call_{i}", "name": c["function"]["name"],
                      "args": c["function"].get("arguments") or {}}
                     for i, c in enumerate(message.get("tool_calls") or [])]
        except (KeyError, TypeError, AttributeError):
            raise ChatError("the model server's answer had no message in it",
                            "server") from None
    return {"text": message.get("content") or "", "calls": calls}


def _given(args) -> str:
    """The final script from apply's arguments, with its message as a `# ` line.
    Raises ValueError for arguments that are not a script."""
    args = _args(args)
    text, note = args.get("script"), args.get("message")
    if not isinstance(text, str) or not isinstance(note, (str, type(None))):
        raise ValueError("apply takes script and message as text")
    return text.rstrip() + ("\n# " + " ".join(note.split()) if note else "")


def _agent(cfg: dict, messages: list[dict], meta: dict, record: dict, shown,
           talk=_talk) -> str | None:
    """Let the model call tools until it ends the run: its final reply, as
    text, or None when it was still calling tools after `steps` rounds."""
    work = [dict(m) for m in messages]
    work[0]["content"] += _AGENT_PROMPT
    log = record["tools"] = []
    for turn in range(1, cfg.get("steps", DEFAULT_STEPS) + 1):
        got = talk(cfg, work)
        if not got["calls"]:
            return got["text"]
        for call in got["calls"]:
            try:
                call["args"] = _args(call["args"])
            except ValueError:
                pass                          # left as sent; the tool refuses it
        # what is sent back as the model's own turn has its arguments as an object
        work.append({"role": "assistant", "content": got["text"], "calls": [
            {**c, "args": c["args"] if isinstance(c["args"], dict) else {}}
            for c in got["calls"]]})
        for call in got["calls"]:
            if call["name"] == "apply":
                try:
                    reply = _given(call["args"])
                except ValueError as exc:
                    result = f"error: {exc}"
                else:
                    log.append({"turn": turn, "tool": "apply", "args": call["args"]})
                    return reply
            else:
                result = _run_tool(call, meta, shown)
            log.append({"turn": turn, "tool": call["name"], "args": call["args"],
                        "result": result})
            work.append({"role": "tool", "id": call["id"], "name": call["name"],
                         "content": result})
    return None

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
    said = [h for h in (history if isinstance(history, list) else [])
            if isinstance(h, dict)
            and isinstance(h.get("request"), str) and isinstance(h.get("reply"), str)]
    for turn in said[-MAX_HISTORY:]:
        messages += [{"role": "user", "content": turn["request"][:MAX_TEXT]},
                     {"role": "assistant", "content": turn["reply"][:2000]}]
    messages.append({"role": "user", "content": asked})
    record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "model": cfg["model"],
              "provider": cfg.get("provider", "ollama"),
              "file": meta.get("file"), "request": text, "view": current.strip(),
              "system": messages[0]["content"],
              "history": said[-MAX_HISTORY:], "rounds": []}
    shown = _shown(current, meta)
    try:
        if cfg.get("agent") and post is _post:
            record["agent"] = True
            try:
                reply = _agent(cfg, messages, meta, record, shown)
            except ChatError as exc:
                # a model that takes no tools is asked the one-shot way
                if exc.kind != "server" or "tool" not in str(exc).lower():
                    raise
                record["no_tools"], reply = str(exc), None
            if reply is not None:
                # the run's last word is the first reply; a retry is one-shot
                first = [reply]
                return _rounds(cfg, messages, meta,
                               lambda c, m: first.pop(0) if first else _post(c, m),
                               record, shown)
            if "no_tools" not in record:
                record["exhausted"] = True   # out of steps: ask the one-shot way
        return _rounds(cfg, messages, meta, post, record, shown)
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
            if not said:             # nothing at all: a fault, not an answer
                raise ChatError("the model gave no answer", "server")
            raise ChatError(said, "said")
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
