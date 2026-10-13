"""A view asked for in words: the model's reply is text, and is checked as text.

The model here is a stand-in -- a function, or a small server speaking
Ollama's protocol -- so the tests say what gmpas does with a reply, not how
good any model is.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from gmpas import chat

META = {
    "variables": [{"name": "t2m", "label": "2 m temperature [K]", "levels": 1},
                  {"name": "u", "label": "zonal wind [m s-1]", "levels": 5},
                  {"name": "v", "label": "meridional wind [m s-1]", "levels": 5}],
    "steps": 4, "labels": ["2012-03-01 00:00", "2012-03-01 06:00",
                           "2012-03-01 12:00", "2012-03-01 18:00"],
    "home": [0.0, 359.0, -90.0, 90.0],
    "cmaps": ["viridis", "RdBu_r"], "projections": ["auto", "Mercator"],
    "colour_options": {"levels": {}},
}
CFG = {"url": "http://nowhere", "model": "stand-in", "timeout": 5.0, "log": "off"}


def _says(*replies):
    """A model that gives these replies in turn, and keeps what it was sent."""
    left, seen = list(replies), []

    def post(cfg, messages):
        seen.append([dict(m) for m in messages])
        return left.pop(0)
    post.seen = seen
    return post


def test_a_reply_becomes_a_checked_script():
    got = chat.ask("wind speed at level 2 in Mercator", META, cfg=CFG,
                   post=_says("var = hypot(u, v)\nlevel = 2\nproj = Mercator\n"))
    assert got["state"]["var"] == "hypot(u, v)"
    assert got["state"]["level"] == 2 and got["state"]["proj"] == "Mercator"
    assert got["script"].splitlines()[0] == "var = hypot(u, v)"


def test_the_model_is_told_the_file_and_the_view():
    post = _says("time = -1\n")
    chat.ask("the last step", META, current="var = t2m\ntime = 0\n", cfg=CFG, post=post)
    system, user = post.seen[0]
    assert "t2m: 2 m temperature [K]" in system["content"]
    assert "u: zonal wind [m s-1] (5 levels, 0 to 4)" in system["content"]
    assert "4 time step(s), 0 to 3" in system["content"]
    assert "Mercator" in system["content"] and "vectors(u" in system["content"]
    assert "var = t2m" in user["content"] and "the last step" in user["content"]


def test_prose_and_a_code_fence_around_the_script_are_dropped():
    reply = "Sure! Here is the script:\n```\nvar = t2m\ncmap = RdBu_r\n```\nEnjoy."
    got = chat.ask("temperature in red and blue", META, cfg=CFG, post=_says(reply))
    assert got["script"] == "var = t2m\ncmap = RdBu_r\n"


def test_a_reply_in_the_python_form_with_an_overlay_is_a_figure_of_layers():
    meta = {**META, "layer_schema": {"kinds": {}}, "kind_labels": {"layers": "layers"},
            "variables": [{**v, "spatial": True, "kinds": ["map", "layers"]}
                          for v in META["variables"]]}
    reply = ("Here you go:\n"
             "contourf(t2m.isel(time=1), cmap=\"RdBu_r\",\n"
             "         levels=[270, 280, 290])\n"
             "vectors(u.isel(time=1, level=2), v.isel(time=1, level=2))\n")
    got = chat.ask("temperature with wind arrows", meta, cfg=CFG, post=_says(reply))
    assert got["state"]["kind"] == "layers" and got["state"]["time"] == 1
    assert [x["kind"] for x in got["state"]["layers"]["layers"]][:2] == \
        ["contourf", "quiver"]
    assert got["script"].startswith("contourf(") and "Here" not in got["script"]


def test_earlier_requests_go_with_a_new_one():
    post = _says("var = t2m\n")
    said = [{"request": f"request {i}", "reply": f"var = u\n# {i}"} for i in range(9)]
    said.append({"request": 5, "reply": None})                  # not a turn: dropped
    chat.ask("no, the temperature", META, cfg=CFG, post=post, history=said)
    sent = post.seen[0]
    assert [m["role"] for m in sent] == ["system"] + ["user", "assistant"] * 6 + ["user"]
    assert sent[1]["content"] == "request 3" and sent[-2]["content"] == "var = u\n# 8"
    assert "no, the temperature" in sent[-1]["content"]


def test_a_question_comes_back_as_a_question():
    got = chat.ask("where is the warmest place", META, cfg=CFG,
                   post=_says("Sure:\nt2m.max()\n"))
    assert got["script"] == "t2m.max()\n"
    assert got["state"] == {"ask": {"what": "max", "var": "t2m"}}
    got = chat.ask("strongest wind", META, cfg=CFG, post=_says("hypot(u, v).max()"))
    assert got["state"]["ask"]["var"] == "hypot(u, v)"
    got = chat.ask("coldest", META, cfg=CFG, post=_says("(t2m - 273.15).min()"))
    assert got["state"]["ask"] == {"what": "min", "var": "t2m - 273.15"}


def test_a_refused_script_is_sent_back_once():
    post = _says("var = sst\n", "var = t2m\n")
    got = chat.ask("sea surface temperature", META, cfg=CFG, post=post)
    assert got["state"]["var"] == "t2m"
    assert "refused" in post.seen[1][-1]["content"]
    assert "sst" in post.seen[1][-1]["content"]


def test_a_script_refused_twice_is_an_error_naming_the_fault():
    with pytest.raises(chat.ChatError, match="refused twice.*level"):
        chat.ask("level nine", META, cfg=CFG,
                 post=_says("var = u\nlevel = 9\n", "var = u\nlevel = 9\n"))


def test_what_the_model_says_instead_of_a_script_is_passed_on():
    with pytest.raises(chat.ChatError, match="no rainfall in this file"):
        chat.ask("rainfall", META, cfg=CFG, post=_says("# no rainfall in this file\n"))


@pytest.mark.parametrize("reply", ["import os\nos.system('x')", "__import__('os')"])
def test_nothing_in_a_reply_is_run(reply):
    with pytest.raises(chat.ChatError):
        chat.ask("anything", META, cfg=CFG, post=_says(reply, reply))


def test_an_empty_or_overlong_request_never_reaches_the_model():
    post = _says()
    with pytest.raises(chat.ChatError, match="sentence"):
        chat.ask("   ", META, cfg=CFG, post=post)
    with pytest.raises(chat.ChatError, match="at most"):
        chat.ask("x" * (chat.MAX_TEXT + 1), META, cfg=CFG, post=post)
    assert post.seen == []


# ------------------------------------------------------------------- config

def test_no_file_means_no_chat(tmp_path, monkeypatch):
    monkeypatch.setenv("GMPAS_CHAT_CONFIG", str(tmp_path / "absent.conf"))
    assert chat.load_config() is None
    with pytest.raises(chat.ChatError, match="no model is configured"):
        chat.ask("anything", META)


def test_the_file_names_the_server_and_the_model(tmp_path):
    path = tmp_path / "chat.conf"
    path.write_text("# my model\nurl = http://127.0.0.1:11434/\n"
                    "model = \"qwen2.5-coder:7b\"\ntimeout = 30\n")
    assert chat.load_config(path) == {"url": "http://127.0.0.1:11434",
                                      "model": "qwen2.5-coder:7b", "timeout": 30.0}


@pytest.mark.parametrize("text, said", [
    ("url = http://x\n", "model is not set"),
    ("model = m\n", "url is not set"),
    ("url = http://x\nmodel = m\nspeed = fast\n", "line 3"),
    ("url = http://x\nmodel = m\ntimeout = soon\n", "timeout"),
])
def test_a_file_that_cannot_be_used_says_why(tmp_path, text, said):
    path = tmp_path / "chat.conf"
    path.write_text(text)
    with pytest.raises(chat.ChatError, match=said):
        chat.load_config(path)


# ---------------------------------------------------------------------- log

def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_a_request_and_every_reply_are_kept_on_this_machine(tmp_path):
    log = tmp_path / "kept" / "chat.jsonl"
    cfg = {**CFG, "log": str(log)}
    chat.ask("sea surface temperature", META, current="var = u\n", cfg=cfg,
             post=_says("var = sst\n", "var = t2m\n"))
    (kept,) = _lines(log)
    assert kept["request"] == "sea surface temperature" and kept["view"] == "var = u"
    assert kept["model"] == "stand-in" and "t2m: 2 m temperature" in kept["system"]
    assert [r["reply"] for r in kept["rounds"]] == ["var = sst\n", "var = t2m\n"]
    assert "sst" in kept["rounds"][0]["refused"] and "refused" not in kept["rounds"][1]
    assert kept["script"] == "var = t2m\n" and "error" not in kept


def test_a_request_that_fails_is_kept_with_why(tmp_path):
    log = tmp_path / "chat.jsonl"
    with pytest.raises(chat.ChatError):
        chat.ask("rainfall", META, cfg={**CFG, "log": str(log)},
                 post=_says("# no rainfall in this file\n"))
    chat.ask("temperature", META, cfg={**CFG, "log": str(log)}, post=_says("var = t2m\n"))
    first, second = _lines(log)
    assert first["error"] == "no rainfall in this file" and "script" not in first
    assert second["script"] == "var = t2m\n"


def test_log_off_keeps_nothing_and_a_log_that_cannot_be_written_is_not_fatal(tmp_path):
    assert chat.log_path({"log": "off"}) is None
    blocked = tmp_path / "file"
    blocked.write_text("")
    cfg = {**CFG, "log": str(blocked / "under-a-file" / "chat.jsonl")}
    assert chat.ask("temperature", META, cfg=cfg, post=_says("var = t2m\n"))["script"]


def test_the_log_goes_to_the_data_directory_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("GMPAS_CHAT_LOG", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert chat.log_path({}) == tmp_path / "gmpas" / "chat.jsonl"


# ------------------------------------------------------------ over the wire

@pytest.fixture
def model_server():
    """A server speaking Ollama's /api/chat."""
    pytest.importorskip("ollama", reason='the wire tests need "gmpas[chat]"')
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, body, self.headers.get("Authorization")))
            if body["model"] == "missing":
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'{"error":"model \\"missing\\" not found"}')
                return
            message = {"role": "assistant", "content": "var = t2m\n"}
            out = {"model": body["model"], "message": message, "done": True}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(out).encode())

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", seen
    srv.shutdown()


def test_ollama_is_asked_without_streaming(model_server):
    url, seen = model_server
    got = chat.ask("temperature", META,
                   cfg={"url": url, "model": "m", "timeout": 5.0, "log": "off"})
    assert got["state"]["var"] == "t2m"
    path, body, auth = seen[0]
    assert path == "/api/chat" and body["stream"] is False and auth is None
    assert body["options"] == {"temperature": 0}


def test_the_key_is_sent_as_a_bearer_token(model_server):
    url, seen = model_server
    cfg = {"url": url, "model": "m", "key": "secret", "timeout": 5.0, "log": "off"}
    assert chat.ask("temperature", META, cfg=cfg)["state"]["var"] == "t2m"
    assert seen[0][2] == "Bearer secret"


def test_a_server_error_is_reported_with_what_it_said(model_server):
    url, _ = model_server
    with pytest.raises(chat.ChatError, match="404.*not found"):
        chat.ask("temperature", META, cfg={"url": url, "model": "missing",
                                           "timeout": 5.0, "log": "off"})


def test_no_server_is_reported_as_no_answer():
    pytest.importorskip("ollama", reason='the wire tests need "gmpas[chat]"')
    cfg = {"url": "http://127.0.0.1:9", "model": "m", "timeout": 2.0, "log": "off"}
    with pytest.raises(chat.ChatError, match="no answer from the model server"):
        chat.ask("temperature", META, cfg=cfg)
