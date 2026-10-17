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

from gmpas import chat, chatmodel

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
    listed = {**META, "variables": [{"name": "t", "label": "T", "levels": 3,
                                     "level_values": [1000.0, 850.0, 500.0],
                                     "level_units": "hPa"}]}
    assert "t: T (3 levels, 0 to 2: 1000, 850, 500 hPa)" in chat.system_prompt(listed)
    # every option the prompt offers for the fast map is one the page has
    from gmpas import colour
    assert {"levels", "reverse", "norm", "center", "extend"} <= set(colour.OPTIONS)
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


def test_a_level_with_no_field_named_is_a_level_of_the_field_shown():
    post = _says("level = 850\n", "shade(u.isel(level=3))\n")
    got = chat.ask("u at 850", META, current="var = u\nlevel = 0\n", cfg=CFG, post=post)
    assert got["state"]["level"] == 3
    assert "850 is outside u's 5 level(s)" in post.seen[1][-1]["content"]
    assert "u.sel(level=850)" in post.seen[1][-1]["content"]
    # in range, or with nothing known about what is shown, it is left alone
    assert chat.ask("level 4", META, current="var = u\n", cfg=CFG,
                    post=_says("level = 4\n"))["state"] == {"level": 4}
    assert chat.ask("level 9", META, cfg=CFG, post=_says("level = 9\n"))["state"]


def test_an_overlay_alone_goes_over_what_is_shown_and_the_script_says_so():
    meta = {**META, "layer_schema": {"kinds": {}}, "kind_labels": {"layers": "layers"},
            "variables": [{**v, "spatial": True, "kinds": ["map", "layers"]}
                          for v in META["variables"]]}
    got = chat.ask("add wind arrows", meta, current="var = t2m\ntime = 0\n", cfg=CFG,
                   post=_says("vectors(u.isel(level=1), v.isel(level=1))\n"))
    assert got["script"].splitlines()[0] == "var = t2m"
    assert [x["kind"] for x in got["state"]["layers"]["layers"][:2]] == \
        ["pcolormesh", "quiver"]


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

def test_no_file_means_the_built_in_model(tmp_path, monkeypatch):
    monkeypatch.setenv("GMPAS_CHAT_CONFIG", str(tmp_path / "absent.conf"))
    monkeypatch.setenv("GMPAS_CHAT_LOG", "off")
    assert chat.load_config() is None
    seen = []

    def post(cfg, messages):
        seen.append((cfg, messages))
        return "var = t2m\n"

    got = chat.ask("temperature over peru", META, post=post)
    assert got["state"]["var"] == "t2m"
    cfg, messages = seen[0]
    assert cfg["provider"] == "builtin" and cfg["model"] == chatmodel.NAME
    # one block in the layout it was taught, with the place it found
    assert len(messages) == 1 and messages[0]["content"].startswith("fields: ")
    assert "places: peru = " in messages[0]["content"]
    assert messages[0]["content"].endswith("ask: temperature over peru")


def test_the_built_in_model_is_not_asked_twice():
    post = _says("var = nothing\n", "var = t2m\n")
    with pytest.raises(chat.ChatError, match="refused: ") as err:
        chat.ask("temperature", META, cfg={**chat.builtin_config(), "log": "off"},
                 post=post)
    assert err.value.kind == "refused" and len(post.seen) == 1


def test_the_built_in_model_can_be_named(tmp_path):
    path = tmp_path / "chat.conf"
    path.write_text("provider = builtin\nlog = off\n")
    cfg = chat.load_config(path)
    assert cfg["provider"] == "builtin" and cfg["model"] == chatmodel.NAME
    assert cfg["agent"] is False


def test_a_server_needs_nothing_installed(tmp_path, monkeypatch):
    monkeypatch.setattr(chatmodel, "available", lambda: False)
    monkeypatch.setenv("GMPAS_CHAT_CONFIG", str(tmp_path / "chat.conf"))
    assert not chat.installed()
    (tmp_path / "chat.conf").write_text("provider = builtin\n")
    assert not chat.installed()
    (tmp_path / "chat.conf").write_text("url = http://127.0.0.1:11434\nmodel = m\n")
    assert chat.installed()


def test_the_file_names_the_server_and_the_model(tmp_path):
    path = tmp_path / "chat.conf"
    path.write_text("# my model\nurl = http://127.0.0.1:11434/\n"
                    "model = \"qwen2.5-coder:7b\"\ntimeout = 30\n")
    assert chat.load_config(path) == {"url": "http://127.0.0.1:11434",
                                      "model": "qwen2.5-coder:7b", "timeout": 30.0,
                                      "agent": True}


@pytest.mark.parametrize("text, said", [
    ("url = http://x\n", "model is not set"),
    ("model = m\n", "url is not set"),
    ("url = http://x\nmodel = m\nspeed = fast\n", "line 3"),
    ("url = http://x\nmodel = m\ntimeout = soon\n", "timeout"),
    ("url = http://x\nmodel = m\nagent = maybe\n", "agent is on or off"),
    ("url = http://x\nmodel = m\nsteps = many\n", "steps is a whole number"),
    ("url = http://x\nmodel = m\nsteps = 0\n", "steps is a whole number"),
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
    cfg = {"url": "http://127.0.0.1:9", "model": "m", "timeout": 2.0, "log": "off"}
    with pytest.raises(chat.ChatError, match="no answer from the model server"):
        chat.ask("temperature", META, cfg=cfg)


@pytest.fixture
def api_server():
    """A server speaking both the OpenAI chat API and Anthropic's Messages API."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, body, {k.lower(): v for k, v in self.headers.items()}))
            if body["model"] == "missing":
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'{"error": {"message": "no such model"}}')
                return
            if self.path.endswith("/chat/completions"):
                out = {"choices": [{"message": {"role": "assistant",
                                                "content": "var = t2m\n"}}]}
            else:
                out = {"content": [{"type": "text", "text": "var = t2m\n"}]}
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


def test_an_openai_compatible_server_is_asked_at_chat_completions(api_server):
    url, seen = api_server
    cfg = {"provider": "openai", "url": url + "/v1", "model": "m", "key": "secret",
           "timeout": 5.0, "log": "off"}
    assert chat.ask("temperature", META, cfg=cfg)["state"]["var"] == "t2m"
    path, body, headers = seen[0]
    assert path == "/v1/chat/completions" and headers["authorization"] == "Bearer secret"
    assert body["temperature"] == 0 and body["stream"] is False
    assert body["messages"][0]["role"] == "system" and "tools" not in body


def test_anthropic_gets_the_system_prompt_beside_the_turns(api_server, monkeypatch):
    url, seen = api_server
    monkeypatch.setenv("GMPAS_CHAT_KEY", "from-env")
    cfg = {"provider": "anthropic", "url": url, "model": "m", "timeout": 5.0, "log": "off"}
    assert chat.ask("temperature", META, cfg=cfg)["state"]["var"] == "t2m"
    path, body, headers = seen[0]
    assert path == "/v1/messages" and headers["x-api-key"] == "from-env"
    assert headers["anthropic-version"] and body["max_tokens"] > 0
    assert "t2m: 2 m temperature" in body["system"] and "tools" not in body
    assert [m["role"] for m in body["messages"]] == ["user"]


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_an_api_error_is_reported_with_what_it_said(api_server, provider):
    url, _ = api_server
    cfg = {"provider": provider, "url": url, "model": "missing", "timeout": 5.0,
           "log": "off"}
    with pytest.raises(chat.ChatError, match="404: no such model"):
        chat.ask("temperature", META, cfg=cfg)


def test_the_provider_is_named_in_the_file(tmp_path):
    path = tmp_path / "chat.conf"
    path.write_text("provider = Anthropic\nmodel = claude-haiku-5-5\n")
    assert chat.load_config(path) == {"provider": "anthropic", "model": "claude-haiku-5-5",
                                      "url": "https://api.anthropic.com", "timeout": 120.0,
                                      "agent": True}
    path.write_text("provider = openai\nmodel = m\n")
    with pytest.raises(chat.ChatError, match="url is not set"):
        chat.load_config(path)
    path.write_text("provider = gemini\nurl = http://x\nmodel = m\n")
    with pytest.raises(chat.ChatError, match="provider is one of"):
        chat.load_config(path)


def test_think_off_asks_ollama_not_to_reason_and_caps_the_reply(model_server):
    url, seen = model_server
    cfg = {"url": url, "model": "m", "timeout": 5.0, "log": "off", "think": "off"}
    assert chat.ask("temperature", META, cfg=cfg)["state"]["var"] == "t2m"
    body = seen[0][1]
    assert body["think"] is False
    assert body["options"]["num_predict"] == chat.MAX_REPLY_TOKENS


# ------------------------------------------------------------------- agent

PROVIDERS = ["ollama", "openai", "anthropic"]


def _reply(provider, text="", *calls):
    """What a server of this provider answers with: `text`, and the tool calls
    (name, arguments) it asks for."""
    if provider == "openai":
        message = {"role": "assistant", "content": text or None}
        if calls:
            message["tool_calls"] = [
                {"id": f"c{i}", "type": "function",
                 "function": {"name": name, "arguments": args if isinstance(args, str)
                              else json.dumps(args)}}
                for i, (name, args) in enumerate(calls)]
        return {"choices": [{"message": message}]}
    if provider == "anthropic":
        return {"content": ([{"type": "text", "text": text}] if text else [])
                + [{"type": "tool_use", "id": f"tu{i}", "name": name, "input": args}
                   for i, (name, args) in enumerate(calls)]}
    message = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = [{"function": {"name": name, "arguments": args}}
                                 for name, args in calls]
    return {"model": "m", "message": message, "done": True}


@pytest.fixture
def agent_server():
    """One server for all three wire formats. It answers each request with the
    next of the replies it was given, and keeps what it was sent."""
    seen, replies = [], []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, body))
            out = json.dumps(replies.pop(0)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield base, seen, replies
    srv.shutdown()


def _cfg(provider, base, **more):
    url = base + ("/v1" if provider == "openai" else "")
    return {"provider": provider, "url": url, "model": "m", "timeout": 5.0,
            "log": "off", "agent": True, **more}


def _tool_results(provider, body):
    """The tool results the last request carried: [(id or name, text)]."""
    messages = body["messages"]
    if provider == "anthropic":
        return [(b["tool_use_id"], b["content"]) for b in messages[-1]["content"]
                if b["type"] == "tool_result"]
    out = []
    for m in reversed(messages):             # one message to each result
        if m["role"] != "tool":
            break
        out.insert(0, (m.get("tool_call_id") or m.get("tool_name"), m["content"]))
    return out


@pytest.mark.parametrize("provider", PROVIDERS)
def test_an_agent_looks_checks_corrects_and_applies(agent_server, provider, tmp_path):
    base, seen, replies = agent_server
    replies += [
        _reply(provider, "", ("describe_file", {})),
        _reply(provider, "", ("check_script", {"script": "var = sst\n"})),
        _reply(provider, "", ("check_script", {"script": "var = t2m\ntime = -1\n"})),
        _reply(provider, "", ("apply", {"script": "var = t2m\ntime = -1\n",
                                        "message": "the last step"}))]
    log = tmp_path / "chat.jsonl"
    got = chat.ask("temperature at the end", META, cfg=_cfg(provider, base, log=str(log)))
    assert got["state"]["var"] == "t2m" and got["state"]["time"] == 3
    assert got["script"] == "var = t2m\ntime = -1\n# the last step\n"
    # four requests, all to the one server, all offering the same three tools
    assert len(seen) == 4
    assert {path for path, _ in seen} == {"/v1/chat/completions" if provider == "openai"
                                          else "/v1/messages" if provider == "anthropic"
                                          else "/api/chat"}
    for _, body in seen:
        names = [t.get("name") or t["function"]["name"] for t in body["tools"]]
        assert names == ["describe_file", "check_script", "apply"]
        shaped = body["tools"][0].get("input_schema") is not None
        assert shaped == (provider == "anthropic")
    # what each tool said went back in the provider's own shape
    (_, file_said), = _tool_results(provider, seen[1][1])
    assert "u: zonal wind [m s-1]; 5 levels" in file_said
    assert "2012-03-01 18:00" in file_said
    (key, refused), = _tool_results(provider, seen[2][1])
    assert refused.startswith("refused:") and "sst" in refused
    (_, accepted), = _tool_results(provider, seen[3][1])
    assert accepted.startswith("accepted") and "time = 3 (2012-03-01 18:00)" in accepted
    if provider == "openai":
        assert key == "c0"
        assert seen[1][1]["messages"][-2]["tool_calls"][0]["function"]["arguments"] == "{}"
    if provider == "anthropic":
        assert key == "tu0"
        assert seen[1][1]["messages"][-2]["content"][0]["type"] == "tool_use"
        assert "tools" in seen[0][1] and "system" in seen[0][1]
    if provider == "ollama":
        assert key == "check_script"
        assert seen[2][1]["messages"][-2]["tool_calls"][0]["function"]["arguments"] \
            == {"script": "var = sst\n"}
    # the log has each call and what it said
    (kept,) = _lines(log)
    assert [t["tool"] for t in kept["tools"]] == \
        ["describe_file", "check_script", "check_script", "apply"]
    assert kept["tools"][1]["args"] == {"script": "var = sst\n"}
    assert kept["tools"][1]["result"].startswith("refused:")
    assert kept["agent"] is True and kept["script"] == got["script"]


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_plain_reply_ends_an_agent_run(agent_server, provider):
    base, seen, replies = agent_server
    replies.append(_reply(provider, "Here:\nvar = t2m\n"))
    got = chat.ask("temperature", META, cfg=_cfg(provider, base))
    assert got["state"]["var"] == "t2m" and len(seen) == 1 and "tools" in seen[0][1]


@pytest.mark.parametrize("provider", PROVIDERS)
def test_an_agent_that_runs_out_of_steps_is_asked_the_one_shot_way(
        agent_server, provider, tmp_path):
    base, seen, replies = agent_server
    replies += [_reply(provider, "", ("describe_file", {}))] * 2 \
        + [_reply(provider, "var = t2m\n")]
    log = tmp_path / "chat.jsonl"
    got = chat.ask("temperature", META, cfg=_cfg(provider, base, steps=2, log=str(log)))
    assert got["state"]["var"] == "t2m" and len(seen) == 3
    assert "tools" in seen[0][1] and "tools" in seen[1][1] and not seen[2][1].get("tools")
    (kept,) = _lines(log)
    assert kept["exhausted"] is True and len(kept["tools"]) == 2


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_fault_in_a_tool_call_is_an_answer_the_model_reads(agent_server, provider):
    base, seen, replies = agent_server
    bad = "{not json" if provider == "openai" else {"script": 5}
    replies += [
        _reply(provider, "", ("check_script", bad), ("lookup", {})),
        _reply(provider, "", ("apply", {"message": "no script"})),
        _reply(provider, "", ("apply", {"script": "var = t2m"}))]
    got = chat.ask("temperature", META, cfg=_cfg(provider, base))
    assert got["script"] == "var = t2m\n" and len(seen) == 3
    said = _tool_results(provider, seen[1][1])
    texts = " ".join(text for _, text in said)
    assert "error:" in texts
    assert len(said) == 2 and "no tool 'lookup'" in texts
    assert _tool_results(provider, seen[2][1])[0][1].startswith("error:")


@pytest.mark.parametrize("provider", PROVIDERS)
def test_an_applied_script_is_checked_and_the_retry_is_one_shot(agent_server, provider):
    base, seen, replies = agent_server
    replies += [_reply(provider, "", ("apply", {"script": "var = sst"})),
                _reply(provider, "var = t2m\n")]
    got = chat.ask("temperature", META, cfg=_cfg(provider, base))
    assert got["script"] == "var = t2m\n" and len(seen) == 2
    assert not seen[1][1].get("tools")
    last = seen[1][1]["messages"][-1]["content"]
    assert "was refused" in last and "sst" in last
    # a second refusal is an error, as ever
    replies += [_reply(provider, "", ("apply", {"script": "var = sst"})),
                _reply(provider, "var = rain\n")]
    with pytest.raises(chat.ChatError, match="refused twice") as err:
        chat.ask("temperature", META, cfg=_cfg(provider, base))
    assert err.value.kind == "refused"


@pytest.mark.parametrize("provider", PROVIDERS)
def test_an_agent_that_says_it_cannot_is_passed_on(agent_server, provider):
    base, seen, replies = agent_server
    replies.append(_reply(provider, "", ("apply", {"script": "",
                                                   "message": "no rainfall here"})))
    with pytest.raises(chat.ChatError, match="no rainfall here") as err:
        chat.ask("rainfall", META, cfg=_cfg(provider, base))
    assert err.value.kind == "said"


@pytest.mark.parametrize("provider", PROVIDERS)
def test_agent_off_is_the_one_shot_path(agent_server, provider):
    base, seen, replies = agent_server
    replies.append(_reply(provider, "var = t2m\n"))
    cfg = _cfg(provider, base, agent=False)
    assert chat.ask("temperature", META, cfg=cfg)["state"]["var"] == "t2m"
    assert len(seen) == 1 and not seen[0][1].get("tools")


def test_a_stand_in_for_the_model_is_never_an_agent(agent_server):
    base, seen, _ = agent_server
    got = chat.ask("temperature", META, cfg=_cfg("openai", base), post=_says("var = t2m\n"))
    assert got["state"]["var"] == "t2m" and not seen


@pytest.mark.parametrize("provider", PROVIDERS)
def test_an_agent_talks_to_the_configured_url_and_no_other(
        agent_server, provider, monkeypatch):
    import urllib.request
    base, seen, replies = agent_server
    replies += [_reply(provider, "", ("describe_file", {})),
                _reply(provider, "", ("apply", {"script": "var = t2m"}))]
    asked = []
    real = urllib.request.urlopen

    def urlopen(req, *a, **kw):
        asked.append(req.full_url)
        return real(req, *a, **kw)
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    chat.ask("temperature", META, cfg=_cfg(provider, base))
    assert asked and all(u.startswith(base) for u in asked)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_an_agent_reports_a_server_that_fails(agent_server, provider):
    base, _, replies = agent_server
    replies.append({"error": "overloaded"})          # no message in the answer
    with pytest.raises(chat.ChatError, match="no message") as err:
        chat.ask("temperature", META, cfg=_cfg(provider, base))
    assert err.value.kind == "server"


def test_the_agent_settings_are_read_from_the_file(tmp_path):
    path = tmp_path / "chat.conf"
    path.write_text("url = http://x\nmodel = m\n")
    assert chat.load_config(path)["agent"] is True and "steps" not in chat.load_config(path)
    path.write_text("url = http://x\nmodel = m\nagent = OFF\nsteps = 3\n")
    assert chat.load_config(path)["agent"] is False and chat.load_config(path)["steps"] == 3
    path.write_text("provider = anthropic\nmodel = m\nagent = yes\n")
    assert chat.load_config(path)["agent"] is True


def test_the_tools_describe_and_check_without_drawing():
    meta = {**META, "variables": [{"name": "t", "levels": 3, "label": "T",
                                   "level_values": [1000.0, 850.0, 500.0],
                                   "level_units": "hPa"}]}
    said = chat._describe_file(meta)
    assert "t: T; 3 levels, numbered 0 to 2 = 1000, 850, 500 hPa" in said
    assert "4 time step(s), numbered 0 to 3: 2012-03-01 00:00" in said
    assert "longitude 0 to 359" in said and "Mercator" in said and "RdBu_r" in said
    checked = chat._check_script("var = t.sel(level=850)", meta, None)
    assert checked.startswith("accepted") and "level = 1 (850 hPa)" in checked
    assert chat._check_script("```\nvar = t.sel(level=800)\n```", meta, None) \
        .startswith("refused: level: 800 is not one of t's levels")
    assert chat._check_script("", meta, None).startswith("refused:")
    # the view on screen counts, as in a real request
    assert "outside t's 3 level" in chat._check_script("level = 7", meta, "t")
    asked = chat._check_script("t2m.max()", META, None)
    assert "question: where t2m is largest" in asked
