"""The chat panel's model form: what the page is told, and what it may write.

gmpas ships no model. The form names one in chat.conf, which may hold a key,
so the file is this user's alone, the key never goes back to the page, and
only the page itself may save it.
"""

from __future__ import annotations

import http.client
import json
import os
import stat
import threading
from http.server import ThreadingHTTPServer

import pytest

from gmpas import chat, viewer


@pytest.fixture
def conf(tmp_path, monkeypatch):
    path = tmp_path / "chat.conf"
    monkeypatch.setenv("GMPAS_CHAT_CONFIG", str(path))
    return path


def test_with_no_file_the_page_is_told_to_name_a_model(conf):
    got = chat.settings()
    assert got["configured"] is False and got["path"] == str(conf)
    assert got["providers"] == ["ollama", "openai", "anthropic"]


def test_the_form_writes_the_file_for_this_user_alone(conf):
    got = chat.save_settings({"provider": "ollama", "url": "http://127.0.0.1:11434",
                              "model": "qwen3:8b", "key": "SECRET"})
    assert got["configured"] and got["model"] == "qwen3:8b" and got["has_key"]
    assert "SECRET" not in json.dumps(got)
    assert stat.S_IMODE(os.stat(conf).st_mode) == 0o600
    assert chat.load_config()["key"] == "SECRET"


def test_the_rest_of_the_file_is_kept(conf):
    conf.write_text("# mine\nurl = http://a:1\nmodel = m\ntimeout = 30\nlog = off\n")
    chat.save_settings({"url": "http://a:1", "model": "n"})
    text = conf.read_text()
    assert "# mine" in text and "timeout = 30" in text and "log = off" in text
    assert chat.load_config()["model"] == "n"


def test_a_saved_key_stays_with_its_server(conf):
    chat.save_settings({"url": "http://a:1", "model": "m", "key": "K"})
    chat.save_settings({"url": "http://a:1/", "model": "m2"})
    assert chat.load_config()["key"] == "K"
    # a new address does not inherit it: the form cannot send it elsewhere
    chat.save_settings({"url": "http://b:2", "model": "m2"})
    assert "key" not in chat.load_config()
    chat.save_settings({"url": "http://b:2", "model": "m2", "key": "K2"})
    chat.save_settings({"url": "http://b:2", "model": "m2", "clear_key": True})
    assert "key" not in chat.load_config()


@pytest.mark.parametrize("given, said", [
    ({"url": "ftp://x", "model": "m"}, "url"),
    ({"url": "http://x", "model": ""}, "model"),
    ({"url": "http://x", "model": "m", "provider": "other"}, "provider"),
    ({"url": "http://x\nmodel = evil", "model": "m"}, "one line"),
    ({"url": "http://x", "model": "m", "key": 'a"b'}, "one line"),
])
def test_a_wrong_form_writes_nothing(conf, given, said):
    with pytest.raises(chat.ChatError, match=said):
        chat.save_settings(given)
    assert not conf.exists()


def test_the_log_is_this_users_alone(tmp_path):
    log = tmp_path / "chat.jsonl"
    chat._log({"log": str(log)}, {"request": "x"})
    assert stat.S_IMODE(os.stat(log).st_mode) == 0o600


# ------------------------------------------------------------ saving over HTTP

@pytest.fixture
def server(conf):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), viewer._handler(None, "<p>page</p>"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


def _post(port, body, **headers):
    con = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    data = json.dumps(body).encode()
    con.request("POST", "/api/chat/config", body=data,
                headers={"Content-Length": str(len(data)), **headers})
    status = con.getresponse().status
    con.close()
    return status


FORM = {"url": "http://127.0.0.1:11434", "model": "m"}


def test_the_page_itself_may_save(server, conf):
    assert _post(server, FORM, Origin=f"http://127.0.0.1:{server}",
                 **{"Content-Type": "application/json"}) == 200
    assert conf.exists()


@pytest.mark.parametrize("origin, ctype", [
    (None, "application/json"),                    # no Origin: not a page's form
    ("http://evil.example", "application/json"),
    ("http://127.0.0.1:1", "application/json"),     # another page on this machine
    ("null", "application/json"),
    ("SELF", "text/plain"),                        # a form post needs no preflight
])
def test_anything_else_may_not(server, conf, origin, ctype):
    headers = {"Content-Type": ctype}
    if origin:
        headers["Origin"] = f"http://127.0.0.1:{server}" if origin == "SELF" else origin
    assert _post(server, FORM, **headers) == 403
    assert not conf.exists()
