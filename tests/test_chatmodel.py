"""The built-in chat model: where it lives, how it is fetched and checked, and
the input it is given. Its answers are not tested here -- the weights are not
in the repository -- but its layout is, since it was trained on that layout."""

from __future__ import annotations

import hashlib
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import pytest

from gmpas import chatmodel


@pytest.fixture
def release(tmp_path, monkeypatch):
    """Three small files served as the release assets, with their hashes."""
    served = tmp_path / "served"
    served.mkdir()
    files = {}
    for name in chatmodel.FILES:
        body = f"not really {name}".encode()
        asset = f"x-{name}"
        (served / asset).write_bytes(body)
        files[name] = (asset, hashlib.sha256(body).hexdigest())
    monkeypatch.setattr(chatmodel, "FILES", files)
    handler = partial(SimpleHTTPRequestHandler, directory=str(served))
    handler.log_message = lambda *a: None
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("GMPAS_CHAT_MODEL_URL", f"http://127.0.0.1:{srv.server_address[1]}")
    monkeypatch.setenv("GMPAS_CHAT_MODEL_DIR", str(tmp_path / "model"))
    yield served
    srv.shutdown()


def test_the_model_folder_is_in_the_data_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("GMPAS_CHAT_MODEL_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert chatmodel.model_dir() == \
        tmp_path / "gmpas" / "models" / f"{chatmodel.NAME}-{chatmodel.VERSION}"
    monkeypatch.setenv("GMPAS_CHAT_MODEL_DIR", str(tmp_path / "shared"))
    assert chatmodel.model_dir() == tmp_path / "shared"


def test_the_files_are_fetched_once(release, tmp_path):
    assert not chatmodel.ready()
    said = []
    where = chatmodel.fetch(say=said.append)
    assert where == tmp_path / "model" and chatmodel.ready() and len(said) == 3
    assert not list(where.glob("*.part"))
    chatmodel.fetch(say=said.append)
    assert len(said) == 3


def test_a_file_that_is_not_the_published_one_is_not_kept(release, tmp_path):
    name = next(iter(chatmodel.FILES))
    (release / chatmodel.FILES[name][0]).write_bytes(b"something else")
    with pytest.raises(chatmodel.ModelError, match="sha256 differs"):
        chatmodel.fetch()
    assert not (tmp_path / "model" / name).exists()
    assert not list((tmp_path / "model").glob("*.part"))


def test_no_route_out_says_how_to_bring_the_model(monkeypatch, tmp_path):
    monkeypatch.setenv("GMPAS_CHAT_MODEL_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("GMPAS_CHAT_MODEL_DIR", str(tmp_path / "model"))
    with pytest.raises(chatmodel.ModelError, match="GMPAS_CHAT_MODEL_DIR"):
        chatmodel.fetch()


def test_places_are_found_longest_name_first():
    got = chatmodel.find_places("rain over the tibetan plateau and peru")
    assert [name for name, _ in got] == ["tibetan plateau", "peru"]
    assert chatmodel.find_places("nothing here") == []


def test_the_input_is_the_layout_it_was_taught():
    meta = {"steps": 4, "variables": [
        {"name": "t", "label": "Temperature [K]", "levels": 2,
         "level_values": [850.0, 500.0], "level_units": "hPa"},
        {"name": "sst", "label": "sea surface temperature [K]"}]}
    view = "var = t\ntime = 0\nlevel = 1\ncmap = viridis\nproj = lonlat\n" \
           "contour(sst)\n"
    src = chatmodel.encode(meta, view, "now in red", [("peru", [-81.4, -68.7, -18.4, 0.0])],
                           ("show t", "var = t\nlevel = 1"))
    assert src == (
        "fields: t (Temperature [K]) levels 850 500 hPa"
        " | sst (sea surface temperature [K])\n"
        "steps: 4\n"
        "view: var=t time=0 level=1 cmap=viridis proj=lonlat\n"
        "layers: contour(sst)\n"
        "places: peru = -81.4, -68.7, -18.4, 0\n"
        "before: show t => var = t ; level = 1\n"
        "ask: now in red")
