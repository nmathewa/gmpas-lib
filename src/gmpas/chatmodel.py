"""The model that ships with gmpas: what chat answers with when no other
model is named.

It is a small encoder-decoder (flan-t5-small, 77M parameters, Apache-2.0)
fine-tuned to turn a request, with the open file's fields and the view on
screen, into script lines. It runs on the CPU with onnxruntime, in about a
tenth of a second.

Its three files (about 160 MB) are release assets of the gmpas version that
first shipped it. They are downloaded once, on the first request or by
`gmpas chat-model`, into `model_dir()`, and each is checked against the hash
below before it is kept. A compute node with no route out uses a copy fetched
on a login node: into the default folder when the home directory is shared,
or into any folder named by `$GMPAS_CHAT_MODEL_DIR` on both.

It knows the forms it was taught and nothing else: a request it was not
taught is answered wrongly or refused, and a larger model named in the chat
configuration is the way past that.

Its input is one block of text, in the layout it was trained on (`encode`):
the fields, the number of steps, the view, places found in the request in the
gazetteer `places.json` (a box per name), the request before this one with
its answer, and the request. Its output is script lines separated by " ; ".
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import tempfile
import threading
from functools import lru_cache
from pathlib import Path

NAME = "gmpas-small"
VERSION = "1"
#: Where the files are published. `$GMPAS_CHAT_MODEL_URL` replaces it.
URL = "https://github.com/nmathewa/gmpas-lib/releases/download/v0.9.0"
#: file in the model folder -> (published asset, sha256)
FILES = {
    "encoder.onnx": (f"{NAME}-{VERSION}-encoder.onnx",
                     "7d34123783f29defdcbe18ca9ceaf247f373ef6f6bd03b40f1c84d6d21ba3222"),
    "decoder.onnx": (f"{NAME}-{VERSION}-decoder.onnx",
                     "998b8d88d5fa96fe9f378ca605d4155e50204ac7f05222da0295a455804d77e4"),
    "tokenizer.json": (f"{NAME}-{VERSION}-tokenizer.json",
                       "95feaacb5ba3b9fd9bb27eec5763244563b9f5d8cf62fc270471b39900d72697"),
}
SIZE_MB = 160
#: The longest answer, in tokens.
MAX_TOKENS = 96
_START, _END = 0, 1
_THREADS = 4


class ModelError(RuntimeError):
    """The model could not be fetched or run; the message says why."""


def available() -> bool:
    """Whether what runs the model is installed: `pip install "gmpas[chat]"`."""
    return all(importlib.util.find_spec(m) is not None
               for m in ("onnxruntime", "tokenizers"))


def model_dir() -> Path:
    """`$GMPAS_CHAT_MODEL_DIR`, or the model's folder in gmpas's data directory."""
    named = os.environ.get("GMPAS_CHAT_MODEL_DIR")
    if named:
        return Path(named).expanduser()
    base = os.environ.get("XDG_DATA_HOME") or "~/.local/share"
    return Path(base).expanduser() / "gmpas" / "models" / f"{NAME}-{VERSION}"


def ready() -> bool:
    """Whether the model's files are on this machine."""
    where = model_dir()
    return all((where / name).is_file() for name in FILES)


_fetching = threading.Lock()


def fetch(say=None) -> Path:
    """Download whatever of the model is missing, check it, and return its
    folder. `say`, if given, is called with a line per file."""
    import urllib.request

    where = model_dir()
    base = (os.environ.get("GMPAS_CHAT_MODEL_URL") or URL).rstrip("/")
    with _fetching:
        for name, (asset, digest) in FILES.items():
            path = where / name
            if path.is_file():
                continue
            if say:
                say(f"{base}/{asset} -> {path}")
            try:
                where.mkdir(parents=True, exist_ok=True)
                # a name of this process's own: another one may be fetching too
                fd, named = tempfile.mkstemp(prefix=name + ".", suffix=".part",
                                             dir=where)
            except OSError as exc:
                raise ModelError(f"{NAME} cannot be kept in {where}: {exc}") from None
            part = Path(named)
            sha = hashlib.sha256()
            try:
                with urllib.request.urlopen(f"{base}/{asset}", timeout=60) as got, \
                        os.fdopen(fd, "wb") as out:
                    while chunk := got.read(1 << 20):
                        sha.update(chunk)
                        out.write(chunk)
            except Exception as exc:          # no route, 404, disk full
                part.unlink(missing_ok=True)
                raise ModelError(
                    f"{NAME} could not be downloaded from {base} ({exc}); on a "
                    f"machine with no route out, run `gmpas chat-model` where there "
                    f"is one and set GMPAS_CHAT_MODEL_DIR to that folder") from None
            if sha.hexdigest() != digest:
                part.unlink(missing_ok=True)
                raise ModelError(f"{asset} from {base} is not the published file "
                                 f"(its sha256 differs); nothing was kept")
            try:
                part.replace(path)
            except OSError as exc:
                part.unlink(missing_ok=True)
                raise ModelError(f"{NAME} cannot be kept in {where}: {exc}") from None
    return where


@lru_cache(maxsize=1)
def places() -> dict:
    """The gazetteer: a name -> [west, east, south, north]."""
    return json.loads((Path(__file__).with_name("places.json")).read_text())


def _words(text: str) -> str:
    """Lower case, each run of punctuation and spaces one space, padded: so
    "over borneo." and "(india)" name places, and "St. Kitts" is "st kitts"."""
    return " " + " ".join(re.findall(r"[\w-]+", text.lower())) + " "


@lru_cache(maxsize=1)
def _by_words() -> dict:
    return {_words(k): k for k in places()}


def find_places(text: str) -> list:
    """The places named in `text`, longest name first, at most two."""
    t = _words(text)
    known = places()
    hits = sorted((k for w, k in _by_words().items() if w in t), key=len, reverse=True)
    out, used = [], ""
    for k in hits:
        if k not in used:
            out.append((k, known[k]))
            used += " " + k
    return out[:2]


_ASSIGN = re.compile(r"^\s*\w+\s*=")


def encode(meta: dict, view: str, ask: str, found: list, before=None) -> str:
    """The model's input, in the layout it was trained on."""
    vs = []
    for v in meta.get("variables", []):
        s = f"{v['name']} ({v.get('label', v['name'])})"
        if v.get("levels", 1) > 1 and v.get("level_values"):
            s += " levels " + " ".join(f"{x:g}" for x in v["level_values"]) \
                + f" {v.get('level_units', '')}"
        vs.append(s)
    shown, calls = {}, []
    for line in view.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if _ASSIGN.match(line):
            k, _, val = line.partition("=")
            shown[k.strip()] = val.strip()
        else:
            calls.append(line.strip())
    pl = "; ".join(f"{k} = {w:g}, {e:g}, {s:g}, {n:g}" for k, (w, e, s, n) in found)
    prev = f"\nbefore: {before[0]} => {before[1].replace(chr(10), ' ; ')}" if before else ""
    lay = f"\nlayers: {' ; '.join(calls)}" if calls else ""
    return (f"fields: {' | '.join(vs)}\nsteps: {meta.get('steps', 1)}\n"
            f"view: var={shown.get('var')} time={shown.get('time')} "
            f"level={shown.get('level')} cmap={shown.get('cmap')} "
            f"proj={shown.get('proj')}{lay}\n"
            f"places: {pl or 'none'}{prev}\nask: {ask}")


#: The longest input the model reads, in tokens; what is past it is cut.
MAX_INPUT = 512
_loading = threading.Lock()
_loaded: dict = {}


def _sessions(where: str):
    """The model's two sessions and its tokenizer, loaded once: requests that
    come together while it loads wait for the one load."""
    with _loading:
        if where not in _loaded:
            if not available():
                raise ModelError(f'{NAME} needs onnxruntime: pip install "gmpas[chat]"')
            try:
                import onnxruntime as ort
                from tokenizers import Tokenizer

                opts = ort.SessionOptions()
                opts.intra_op_num_threads = _THREADS
                _loaded.clear()
                _loaded[where] = (ort.InferenceSession(f"{where}/encoder.onnx", opts),
                                  ort.InferenceSession(f"{where}/decoder.onnx", opts),
                                  Tokenizer.from_file(f"{where}/tokenizer.json"))
            except Exception as exc:
                raise ModelError(f"{NAME} in {where} could not be loaded: {exc}") from None
        return _loaded[where]


def _model():
    if not available():
        raise ModelError(f'{NAME} needs onnxruntime: pip install "gmpas[chat]"')
    return _sessions(str(model_dir() if ready() else fetch()))


def _named(v: dict, words: set) -> int:
    """How many of the request's words a field's name and label hold."""
    own = set(re.findall(r"\w+", f"{v['name']} {v.get('label', '')}".lower()))
    return len(own & words)


def prompt(meta: dict, view: str, ask: str, before=None) -> str:
    """The model's input for a request, fitted to the MAX_INPUT tokens it reads.

    A file with many fields does not fit, and the request comes last, so the
    field list is cut instead: the fields the view shows, then those whose
    name or label share a word with the request (or the one before), then
    the rest in the file's order, as many as fit."""
    found = find_places(ask)
    src = encode(meta, view, ask, found, before)
    tok = _model()[2]
    if len(tok.encode(src).ids) < MAX_INPUT:
        return src
    fields = meta.get("variables", [])
    shown = set(re.findall(r"\w+", view))
    words = set(re.findall(r"\w+", (ask + " " + (before[0] if before else "")).lower()))
    order = sorted(range(len(fields)), key=lambda i: (
        fields[i]["name"] not in shown, -_named(fields[i], words), i))
    kept: list = []
    for i in order:
        trial = sorted(kept + [i])
        if len(tok.encode(encode({**meta, "variables": [fields[j] for j in trial]},
                                 view, ask, found, before)).ids) >= MAX_INPUT:
            break
        kept = trial
    if not kept:
        raise ModelError(f"this request and view are too long for {NAME}; "
                         f"shorten it, or name a larger model in the chat configuration")
    return encode({**meta, "variables": [fields[j] for j in kept]},
                  view, ask, found, before)


_running = threading.Lock()


def generate(src: str) -> str:
    """The model's answer to one input, greedily decoded."""
    import numpy as np

    enc, dec, tok = _model()
    ids = np.array([tok.encode(src).ids], dtype=np.int64)
    mask = np.ones_like(ids)
    with _running:
        hidden = enc.run(None, {"ids": ids, "mask": mask})[0]
        out = [_START]
        for _ in range(MAX_TOKENS):
            logits = dec.run(None, {"dec_ids": np.array([out], dtype=np.int64),
                                    "hidden": hidden, "mask": mask})[0]
            nxt = int(logits[0].argmax())
            if nxt == _END:
                break
            out.append(nxt)
    return tok.decode(out[1:], skip_special_tokens=True)
