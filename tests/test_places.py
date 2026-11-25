"""Places by name (gmpas.places): the gazetteer, finding names in a request,
and a box moved into a file's own longitudes."""

from __future__ import annotations

import pytest

from gmpas import chat, places, script


def test_a_country_is_its_main_land_and_its_parts_are_places_too():
    (box, label), = places.lookup("United States")
    assert label == "United States of America (country)"
    assert box[0] < -120 and box[1] > -70 and box[2] > 24 and box[3] < 50   # the 48
    assert places.lookup("usa") == places.lookup("the united states")
    (alaska, _), = places.lookup("Alaska")
    assert alaska[0] > 170 and alaska[1] > 180            # across the date line


def test_a_name_with_two_places_keeps_both_and_a_label_picks_one():
    found = places.lookup("georgia")
    assert [label for _, label in found] == ["Georgia (country)", "Georgia (US state)"]
    (box, label), = places.lookup("Georgia (US state)")
    assert label == "Georgia (US state)" and box[0] < -80
    assert places.lookup("atlantis") == []


@pytest.mark.parametrize("text, names", [
    ("show sst over the Bay of Bengal", ["bay of bengal"]),
    ("wind over the United States and Canada", ["united states", "canada"]),
    ("zoom to new britain", ["new britain"]),          # not Britain
    ("rain over Turkey", ["turkey"]),
    ("a turkey sandwich", []),                           # a word, not a place
    ("show us the rain in the US", ["us"]),             # "us" only in capitals
    ("sst in nino 3.4", ["nino 3.4"]),
    ("plot snow", []),
])
def test_places_are_found_in_a_request(text, names):
    assert [name for name, _ in places.find(text)] == names


def test_a_box_moves_into_the_files_longitudes():
    us = [-124.7, -67.0, 25.1, 49.4]
    assert places.fit(us, [0.0, 360.0, -90.0, 90.0]) == [235.3, 293.0, 25.1, 49.4]
    assert places.fit(us, [-180.0, 180.0, -90.0, 90.0]) == us
    assert places.fit(us, [140.0, 155.0, -10.0, 5.0]) is None        # elsewhere
    assert places.fit([172.5, 230.0, 51.2, 71.4], [-180.0, 180.0, -90.0, 90.0]) \
        == [-187.5, -130.0, 51.2, 71.4]


META = {"variables": [{"name": "tas", "levels": 1}], "steps": 1, "labels": ["x"],
        "cmaps": ["viridis"], "projections": ["auto"], "colour_options": {"levels": {}},
        "home": [1.25, 358.75, -88.75, 88.75]}


def test_a_script_frames_a_place_by_its_name():
    for text in ('view(region="United States")', "region = United States"):
        got = script.read(text, META)[1]
        assert got["region"] == [235.3, 293.0, 25.1, 49.4]
        assert got["notes"] == ["United States of America (country): "
                                "region = 235.3, 293, 25.1, 49.4"]
    with pytest.raises(script.ScriptError, match=r"Georgia \(country\).*or Georgia "
                                                 r"\(US state\).*name one"):
        script.read('view(region="georgia")', META)
    with pytest.raises(script.ScriptError, match="not a place gmpas knows"):
        script.read('view(region="atlantis")', META)
    manus = {**META, "home": [140.0, 155.0, -10.0, 5.0]}
    with pytest.raises(script.ScriptError, match="outside the data"):
        script.read('view(region="United States")', manus)


def test_a_box_in_numbers_is_moved_too():
    assert script.read("region = -100, -80, 20, 30", META)[1]["region"] == \
        [260.0, 280.0, 20.0, 30.0]


def test_the_model_is_told_where_the_places_named_are():
    said = chat._named_places("PBL height over New Britain")
    assert '"new britain" is New Britain (island group): view(region="new britain")' \
        in said
    told = chat._named_places("temperature over Georgia")
    assert "Georgia (country)" in told and "Georgia (US state)" in told and "ask" in told
    assert chat._named_places("plot snow") == ""
