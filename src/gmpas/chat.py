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
import time
from pathlib import Path

from . import script as _script

#: The longest request taken, in characters.
MAX_TEXT = 500
DEFAULT_TIMEOUT = 120.0
_KEYS = ("url", "model", "key", "timeout", "log")


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


def system_prompt(meta: dict) -> str:
    """What the model is told: the task, the keys, and the open file."""
    keys = "\n".join(f"  {k}: {v}" for k, v in _script.keys().items())
    variables = "\n".join(
        f"  {v['name']}: {v.get('label') or v['name']}"
        + (f" ({v['levels']} levels, numbered from 0)" if v.get("levels", 1) > 1 else "")
        for v in meta.get("variables", []))
    labels = meta.get("labels") or []
    steps = meta.get("steps", len(labels))
    times = f"{steps} time step(s), numbered from 0"
    if labels:
        times += f"; the first is {labels[0]}, the last {labels[-1]}"
    home = meta.get("home")
    where = (f"The data covers longitude {home[0]:g} to {home[1]:g} and latitude "
             f"{home[2]:g} to {home[3]:g}; give regions in that longitude range.\n"
             if home else "")
    projections = ", ".join([_script.LONLAT, *(meta.get("projections") or [])])
    return (
        "You turn a request about a map of weather or climate data into a gmpas "
        "script. Reply with script lines only: one `key = value` per line, no "
        "explanation, no code fence. Write only the keys the request changes. "
        "If the request cannot be met with these keys, reply with one line "
        "starting `# ` that says why.\n\n"
        f"Keys:\n{keys}\n\n"
        "`var` is a variable name from the list, or an expression of them: "
        "a - b, a * 1000, hypot(a, b).\n"
        "`region` is west, east, south, north in degrees.\n\n"
        f"Variables in this file:\n{variables}\n\n"
        f"{times}.\n{where}"
        f"Projections: {projections}.\n"
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
    """The script lines in a model's reply: what is inside a code fence if it
    used one, and otherwise every line that is a comment or sets a key."""
    lines = reply.strip().splitlines()
    fences = [i for i, line in enumerate(lines) if line.strip().startswith("```")]
    if len(fences) >= 2:
        lines = lines[fences[0] + 1:fences[1]]
    kept = [line.strip() for line in lines
            if line.strip() and (line.strip()[0] in "#!" or "=" in line)]
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
