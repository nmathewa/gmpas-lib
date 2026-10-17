"""The script written as Python with xarray would write it (gmpas/script.py).

It reads like Python and is never run as Python. Both forms come out as the
same settings, so the page applies either without knowing which it was.
"""

from __future__ import annotations

import pytest

from gmpas import layers as L
from gmpas import script

META = {
    "variables": [{"name": "t2m", "levels": 1}, {"name": "u", "levels": 4},
                  {"name": "v", "levels": 4}],
    "steps": 3,
    "labels": ["2012-01-01 00:00", "2012-01-01 06:00", "2012-01-01 12:00"],
    "cmaps": ["viridis", "RdBu_r"],
    "projections": ["auto", "Orthographic", "Mercator"],
    "colour_options": {"levels": {}},
}


def test_the_python_form_is_the_same_settings_as_the_line_form():
    python = script.parse("""
        # 850 hPa wind speed over the Maritime Continent
        spd = hypot(u, v)
        view(region=(90, 150, -15, 15), proj="Mercator")
        shade(spd.isel(time=2, level=3), cmap="viridis", vmin=0, vmax=20.5,
              levels="(0,20,2)(inf)", reverse=True)
    """.replace("\n        ", "\n"))
    lines = script.parse("""
        var = hypot(u, v)
        time = 2
        level = 3
        cmap = viridis
        vmin = 0
        vmax = 20.5
        levels = (0,20,2)(inf)
        reverse = true
        proj = Mercator
        region = 90, 150, -15, 15
    """)
    assert python == {**lines, "kind": "map"}      # shade() is the fast map
    assert script.resolve(python, META)["var"] == "hypot(u, v)"


def test_a_step_is_picked_by_number_from_zero_or_by_its_label():
    assert script.parse("shade(t2m.isel(time=0))") == \
        {"var": "t2m", "time": 0, "kind": "map"}
    assert script.parse("shade(t2m.isel(time=-1))")["time"] == -1
    got = script.parse('shade(t2m.sel(time="2012-01-01 06:00"))')
    assert script.resolve(got, META)["time"] == 1
    got = script.parse("shade(u.isel(time=1).isel(level=2))")
    assert (got["time"], got["level"]) == (1, 2)


def test_arithmetic_is_written_out_for_the_derive_box():
    assert script.parse("shade(t2m - 273.15)")["var"] == "t2m - 273.15"
    assert script.parse("anom = u - v\nshade(anom.isel(time=1))") == \
        {"var": "u - v", "time": 1, "kind": "map"}
    assert script.parse("shade((u - v).isel(level=1))")["level"] == 1


def test_the_two_forms_mix():
    got = script.parse('spd = hypot(u, v)\nvar = spd\ntime = 1\ncmap = "RdBu_r"\n')
    assert got == {"var": "hypot(u, v)", "time": 1, "cmap": "RdBu_r"}
    got = script.parse('shade(t2m)\nlevels = "(,,2.5)"\nvmin = None\n')
    assert got == {"var": "t2m", "kind": "map", "vmin": None,
                   "colour": {"levels": "(,,2.5)"}}


def test_view_frames_the_map():
    got = script.parse('view(proj="Orthographic", lon0=120, lat0=-10)')
    assert got == {"proj": "Orthographic", "plon": 120.0, "plat": -10.0}
    assert script.parse("view(region=[90, 150, -15, 15])")["region"] == \
        [90.0, 150.0, -15.0, 15.0]


@pytest.mark.parametrize("text, says", [
    ("shade(t2m.isel(time=1.5))", "line 1: .isel\\(time=\\) takes a whole number"),
    ("shade(t2m.isel(step=1))", "takes time and level, not step"),
    ("shade(t2m.median('time'))", ".median\\(\\) is not understood"),
    ("shade(t2m.isel(time=1).isel(time=2))", "time is picked twice"),
    ("shade(t2m, colormap='viridis')", "shade\\(\\) has no colormap"),
    ("shade(t2m, cmpa='viridis')", "has no cmpa; did you mean cmap"),
    ("veiw(proj='Mercator')", "veiw\\(\\) is not understood; did you mean view"),
    ("view(center=3)", "view\\(\\) has no center"),
    ("section(t2m)", "section\\(\\) is not in the script yet"),
    ("hypot(u, v)", "makes a field; draw it"),
    ("series(t2m, lon=147)", "takes both lon and lat"),
    ("contourf(t2m)\nhist(u)", "hist\\(\\) is a plot of its own"),
    ("vectors(u)", "vectors\\(\\) takes u and v"),
    ("vectors(u.isel(level=1), v.isel(level=2))", "u and v are drawn at one level"),
    ("contourf(t2m.isel(time=0))\ncontour(u.isel(time=1))", "one time step"),
    ("contourf(t2m, cmap=open)\ncontour(u)", "an option takes"),
    ("shade(sqrt(t2m))", "sqrt\\(\\) is not a function here"),
    ("shade(u.isel(time=1) - v.isel(time=2))", "picked at one time, not 1 and 2"),
    ("spd = hypot(u, v)\nshade(t2m)", "line 1: spd is set and never used"),
    ("shade(t2m", "line 1: "),
    ("shade(t2m)\nx = = 3", "line 2: invalid syntax: x = = 3"),
    ("shade(t2m)\nvmin = abc", "line 2: vmin takes a number"),
    ("var = t2m\nvar = u\nshade(t2m)", "line 2: var is already set on line 1"),
    ("view(region=(1, 2, 3, 4))\nview(region=(1, 2, 3, 5))\nshade(u)",
     "region is already set"),
    ("view = 3\nshade(t2m)", "view is a word of the script"),
])
def test_what_is_not_understood_says_which_line_and_why(text, says):
    with pytest.raises(script.ScriptError, match=says):
        script.parse(text)


@pytest.mark.parametrize("text", [
    "import os",
    "shade(__import__('os').system('true'))",
    "shade(t2m.__class__)",
    "x = open('/etc/passwd').read()\nshade(x)",
    "shade(t2m if u else v)",
    "shade([t2m for t2m in u])",
    "shade(lambda: 1)",
    "shade(t2m); exec('1')",
    "shade(u ** 2)",
])
def test_python_that_is_not_the_script_is_refused_not_run(text):
    with pytest.raises(script.ScriptError):
        script.parse(text)


@pytest.mark.parametrize("text, says", [
    ("shade(tm2)", "'tm2' is not a variable.*did you mean t2m"),
    ("shade(t2m, cmap='viridsi')", "did you mean viridis"),
    ("shade(u.isel(level=9))", "level: 9 is outside u's 4 level"),
    ("shade(t2m.sel(time='noon'))", "no step is labelled"),
    ("view(proj='Robinson')", "proj: 'Robinson' is not one of"),
])
def test_the_file_is_checked_as_for_the_line_form(text, says):
    with pytest.raises(script.ScriptError, match=says):
        script.resolve(script.parse(text), META)


def test_a_misspelt_key_is_still_called_an_unknown_key():
    with pytest.raises(script.ScriptError, match="line 2: unknown key 'speed'"):
        script.parse("var = t2m\nspeed = 3")


# ------------------------------------------------- what draws picks the plot

KINDS = {**META,
         "kind_labels": {k: k for k in ("map", "contourf", "contour", "hist", "series",
                                        "layers")},
         "layer_schema": {"kinds": {}},
         "variables": [{**v, "spatial": True,
                        "kinds": ["map", "contourf", "contour", "hist", "series",
                                  "layers"]}
                       for v in META["variables"]]}


def test_the_call_that_draws_names_the_plot():
    assert script.parse("contourf(t2m)")["kind"] == "contourf"
    assert script.parse("hist(u.isel(level=1))") == {"var": "u", "level": 1,
                                                     "kind": "hist"}
    got = script.parse("series(t2m, lon=147.4, lat=-2.06)")
    assert got == {"var": "t2m", "kind": "series", "point": [147.4, -2.06]}
    assert script.parse("var = t2m\nkind = contour\npoint = 10, 5\n") == \
        {"var": "t2m", "kind": "contour", "point": [10.0, 5.0]}
    assert script.resolve(script.parse("contourf(t2m, cmap='viridis')"), KINDS)["kind"] \
        == "contourf"


def test_winds_over_a_field_are_a_figure_of_layers():
    got = script.parse("""
contourf(t2m.isel(time=1), cmap="RdBu_r", levels=[270, 280, 290])
contour(u.isel(time=1, level=2), colors="black")
vectors(u.isel(time=1, level=3), v.isel(time=1, level=3), stride=4, color="white")
""")
    assert (got["kind"], got["var"], got["time"]) == ("layers", "t2m", 1)
    layers = got["layers"]["layers"]
    assert [x["kind"] for x in layers] == ["contourf", "contour", "quiver",
                                           "coastlines", "gridlines"]
    assert layers[0] == {"kind": "contourf", "var": "t2m", "level": "follow",
                         "visible": True, "opacity": 1,
                         "options": {"cmap": "RdBu_r", "levels": [270, 280, 290]}}
    assert (layers[1]["var"], layers[1]["level"]) == ("u", 2)
    assert (layers[2]["u"], layers[2]["v"], layers[2]["level"]) == ("u", "v", 3)
    assert layers[2]["options"] == {"stride": 4, "color": "white"}
    assert script.resolve(got, KINDS)["kind"] == "layers"


def test_winds_alone_are_layers_too_and_a_definition_may_name_a_field():
    got = script.parse("uu = u\nbarbs(uu, v)")
    assert got["kind"] == "layers" and got["var"] == "u"
    assert got["layers"]["layers"][0]["kind"] == "barbs"
    assert script.parse("streamlines(u, v)")["layers"]["layers"][0]["kind"] == "streamplot"


@pytest.mark.parametrize("text, says", [
    ("contourf(tm2)", "did you mean t2m"),
    ("profile(u, lon=1, lat=2)", "kind: u has no 'profile' plot"),
    ("kind = contuor\nvar = t2m", "did you mean contour"),
    ("contourf(hypot(u, v))", "kind: hypot\\(u, v\\) has no 'contourf' plot"),
    ("contourf(t2m)\nview(proj='Mercator')", "is drawn on the lonlat map"),
    ("contourf(t2m)\nvectors(u, w)", "layers: layer 2 \\(quiver\\): v='w' is not"),
    ("contourf(t2m, cmpa='jet')\nvectors(u, v)", "layer 1 \\(contourf\\): unknown option"),
    ("contourf(t2m)\ncontour(u.isel(level=7))", "layers: .*level 7 is out of range"),
    ("contourf(u - v)\ncontour(u)", "var='u - v' is not a map variable"),
])
def test_a_plot_the_file_cannot_draw_is_refused(text, says):
    with pytest.raises(script.ScriptError, match=says):
        script.resolve(script.parse(text), KINDS)


def test_a_page_without_layers_says_so():
    with pytest.raises(script.ScriptError, match="one field at a time"):
        script.resolve(script.parse("contourf(t2m)\nvectors(u, v)"),
                       {**KINDS, "layer_schema": None})


def test_a_call_that_draws_wins_over_a_line_saying_the_same_another_way():
    """A model restates the view as lines and then draws: that is one script."""
    got = script.parse('var = t2m\ntime = 0\ncmap = viridis\nregion = 110, 155, -45, -10\n'
                       'shade(u.isel(time=2), cmap="RdBu_r")\n')
    assert (got["var"], got["time"], got["cmap"]) == ("u", 2, "RdBu_r")
    assert got["region"] == [110.0, 155.0, -45.0, -10.0]
    # in either order
    assert script.parse("shade(u.isel(time=2))\nvar = t2m\ntime = 0\n")["var"] == "u"


def test_a_layer_may_be_arithmetic_where_the_page_can_draw_it():
    text = "contourf(t2m)\ncontour(t2m - 273.15, labels=True)"
    got = script.parse(text)
    assert got["layers"]["layers"][1]["var"] == "t2m - 273.15" and got["var"] == "t2m"
    assert script.resolve(got, {**KINDS, "layer_expressions": True})["kind"] == "layers"
    with pytest.raises(script.ScriptError, match="var='t2m - nope' is not a map"):
        script.resolve(script.parse("contourf(t2m)\ncontour(t2m - nope)"),
                       {**KINDS, "layer_expressions": True})


def test_a_lone_contour_is_a_one_layer_figure_where_that_is_the_only_way():
    meta = {**KINDS, "variables": [{**v, "kinds": ["map", "layers"]}
                                   for v in KINDS["variables"]]}
    got = script.resolve(script.parse('contourf(u.isel(level=2), cmap="RdBu_r")'), meta)
    assert got["kind"] == "layers"
    first = got["layers"]["layers"][0]
    assert (first["kind"], first["var"], first["level"]) == ("contourf", "u", 2)
    assert first["options"] == {"cmap": "RdBu_r"}
    assert [x["kind"] for x in got["layers"]["layers"][1:]] == ["coastlines", "gridlines"]


def test_a_series_needs_more_than_one_step_and_a_profile_more_than_one_level():
    one = {**KINDS, "steps": 1, "labels": KINDS["labels"][:1]}
    with pytest.raises(script.ScriptError, match="series: this file has one time step"):
        script.resolve(script.parse("series(t2m, lon=147, lat=-2)"), one)
    with pytest.raises(script.ScriptError, match="profile: t2m has one level"):
        script.resolve(script.parse("profile(t2m, lon=147, lat=-2)"), KINDS)


def test_where_a_point_is_read_in_the_panel_any_field_has_a_series():
    panel = {**KINDS, "point_panel": True,
             "variables": [{**v, "kinds": ["map", "layers"]} for v in KINDS["variables"]]}
    got = script.resolve(script.parse("series(t2m, lon=147.4, lat=-2)"), panel)
    assert (got["kind"], got["panel"], got["point"]) == ("map", "series", [147.4, -2.0])
    got = script.resolve(script.parse("profile(u.isel(time=1), lon=10, lat=5)"), panel)
    assert (got["kind"], got["panel"], got["time"]) == ("map", "profile", 1)
    with pytest.raises(script.ScriptError, match="series: say where"):
        script.resolve(script.parse("series(t2m)"), panel)


def test_picking_before_or_after_arithmetic_is_the_same_field():
    after = script.parse("shade((t2m - 273.15).isel(time=1))")
    assert script.parse("shade(t2m.isel(time=1) - 273.15)") == after
    assert script.parse("a = t2m.isel(time=1)\nshade(a - 273.15)") == after
    got = script.parse("shade(hypot(u.isel(time=0, level=2), v.isel(level=2)))")
    assert (got["var"], got["time"], got["level"]) == ("hypot(u, v)", 0, 2)
    layer = script.parse("contourf(t2m)\ncontour(t2m.isel(time=0) - 273.15)")
    assert layer["layers"]["layers"][1]["var"] == "t2m - 273.15" and layer["time"] == 0


def test_a_plain_line_means_the_same_beside_a_call():
    """A model restates the view -- empty settings and all -- and then draws."""
    got = script.parse(
        'var = t2m\ntime = 0\nlevel = 0\ncmap = viridis\nvmin =\nvmax =\nproj = lonlat\n'
        'region = 94.3, 154.3, -45, -10\n'
        'shade(t2m.isel(time=0), cmap="viridis")\n'
        'contour(u.isel(time=0), colors="black", levels=10)\n')
    assert got["kind"] == "layers" and got["vmin"] is None and got["proj"] == "lonlat"
    assert [x["kind"] for x in got["layers"]["layers"][:2]] == ["pcolormesh", "contour"]
    got = script.parse('shade(t2m,\n      cmap = "RdBu_r",\n      vmin = 1)\n'
                       "levels = (0,20,2)(inf)\ntime = 2012-01-01 06:00\n"
                       "region = (1, 2, 3, 4)\nreverse = True\n! a Ferret comment\n")
    assert got == {"var": "t2m", "kind": "map", "cmap": "RdBu_r", "vmin": 1.0,
                   "time": "2012-01-01 06:00", "region": [1.0, 2.0, 3.0, 4.0],
                   "colour": {"levels": "(0,20,2)(inf)", "reverse": True}}
    assert script.parse("cmap = cmo.thermal\nshade(t2m)")["cmap"] == "cmo.thermal"


def test_the_map_itself_is_drawn_by_calls_too():
    got = script.parse('ocean(facecolor="lightblue")\ncontourf(t2m)\nrivers(color="blue")\n'
                       'coastlines(linewidth=1.2)\n'
                       'figure(title="2 m temperature", look="ferret")')
    stack = got["layers"]
    assert [x["kind"] for x in stack["layers"]] == \
        ["ocean", "contourf", "rivers", "coastlines", "gridlines"]   # one coastlines
    assert stack["layers"][2]["options"] == {"color": "blue"}
    assert stack["figure"] == {"title": "2 m temperature", "look": "ferret"}
    assert script.resolve(got, KINDS)["kind"] == "layers"
    assert script.parse("contourf(t2m)\ngridvalues(u)")["layers"]["layers"][1]["kind"] \
        == "gridvalues"


@pytest.mark.parametrize("text, says", [
    ("rivers()", "a figure draws a field, and this one names none"),
    ("contourf(t2m)\nrivers('blue')", "rivers\\(\\) takes options by name"),
    ("contourf(t2m)\nrivers(colour='blue')", "layer 2 \\(rivers\\): unknown option"),
    ("contourf(t2m)\nfigure(projection='Peters')", "figure: projection='Peters'"),
])
def test_a_feature_or_a_figure_option_that_does_not_exist_is_refused(text, says):
    with pytest.raises(script.ScriptError, match=says):
        script.resolve(script.parse(text), KINDS)


def test_a_question_about_a_field_changes_nothing_shown():
    assert script.parse("t2m.max()") == {"ask": {"what": "max", "var": "t2m"}}
    got = script.parse("u.isel(time=1, level=2).argmin()")
    assert got == {"ask": {"what": "min", "var": "u", "time": 1, "level": 2}}
    assert "var" not in got and "kind" not in got
    assert script.resolve(script.parse("T2M.sel(time='2012-01-01 06:00').max()"),
                          META)["ask"] == {"what": "max", "var": "t2m", "time": 1}
    assert script.parse("hypot(u, v).max()")["ask"]["var"] == "hypot(u, v)"


@pytest.mark.parametrize("text, says", [
    ("t2m.mean()", ".mean\\(\\) is not in the script yet"),
    ("t2m.max()\nu.min()", "line 2: one question to a script"),
])
def test_a_question_the_script_cannot_answer_is_refused(text, says):
    with pytest.raises(script.ScriptError, match=says):
        script.parse(text)


@pytest.mark.parametrize("text, says", [
    ("tm2.max()", "max: 'tm2' is not a variable.*did you mean t2m"),
    ("u.isel(level=9).max()", "level: 9 is outside u's level"),
    ("t2m.isel(time=7).max()", "time: step 7 is outside"),
])
def test_a_question_is_checked_against_the_file(text, says):
    with pytest.raises(script.ScriptError, match=says):
        script.resolve(script.parse(text), META)


def test_an_option_matplotlib_takes_is_passed_on_as_a_plain_value():
    got = script.resolve(script.parse(
        'contourf(t2m, antialiased=True)\n'
        'vectors(u, v, stride=4, pivot="mid", headlength=7)\n'
        'contour(t2m, levels=[270, 280], corner_mask=[1, 2])'), KINDS)
    fill, arrows, lines = got["layers"]["layers"][:3]
    assert fill["options"] == {"antialiased": True}
    assert arrows["options"] == {"stride": 4, "pivot": "mid", "headlength": 7}
    # its own option takes the list as its form writes it; matplotlib's keeps it
    assert lines["options"] == {"levels": [270, 280], "corner_mask": [1, 2]}
    # one call with a layer's option is a figure of that one layer
    alone = script.resolve(script.parse("contourf(t2m, antialiased=True)"), KINDS)
    assert alone["kind"] == "layers"
    assert script.resolve(script.parse("contourf(t2m)"), KINDS)["kind"] == "contourf"


LEVELS = {**META, "variables": [
    {"name": "t", "levels": 3, "level_values": [1000.0, 850.0, 500.0],
     "level_units": "hPa", "spatial": True, "kinds": ["map", "layers"]},
    {"name": "u", "levels": 3, "level_values": [1000.0, 850.0, 500.0],
     "spatial": True, "kinds": ["map", "layers"]},
    {"name": "t2m", "levels": 1, "spatial": True, "kinds": ["map", "layers"]}],
    "layer_schema": {"kinds": {}}, "kind_labels": {"map": "map", "layers": "layers"}}


def test_a_level_is_picked_by_its_value_where_the_file_lists_them():
    assert script.parse("shade(t.sel(level=850))")["level"] == {"value": 850.0}
    assert script.resolve(script.parse("shade(t.sel(level=850))"), LEVELS)["level"] == 1
    got = script.resolve(
        script.parse("shade((t - u).sel(level=500).isel(time=1))"), LEVELS)
    assert (got["var"], got["level"], got["time"]) == ("t - u", 2, 1)
    got = script.resolve(script.parse(
        "contourf(t.sel(level=850))\ncontour(u.sel(level=500))"), LEVELS)
    assert [x["level"] for x in got["layers"]["layers"][:2]] == [1, 2]
    asked = script.resolve(script.parse("t.sel(level=1000).max()"), LEVELS)["ask"]
    assert asked["level"] == 0


@pytest.mark.parametrize("text, says", [
    ("shade(t.sel(level=860))",
     "level: 860 is not one of t's levels: 1000, 850, 500 hPa"),
    ("shade(t2m.sel(level=850))", "level: t2m lists no level values"),
    ("shade(t.sel(level='low'))", "takes the level's own value"),
    ("vectors(u.sel(level=850), u.sel(level=500))\nshade(t)",
     "u and v are drawn at one level"),
])
def test_a_level_value_the_file_does_not_have_is_refused(text, says):
    with pytest.raises(script.ScriptError, match=says):
        script.resolve(script.parse(text), LEVELS)


def test_the_plain_map_and_a_reversed_palette_go_by_their_usual_names():
    for name in ("PlateCarree", "lat/lon", "latlon", "lonlat"):
        assert script.resolve(script.parse(f'view(proj="{name}")'), META)["proj"] == ""
    assert script.resolve(script.parse("proj = mercator"), META)["proj"] == "Mercator"
    # a figure of layers may name the plain map; it may not name a projection
    plain = 'view(proj="PlateCarree")\ncontourf(t2m)\ncontour(u)'
    assert script.resolve(script.parse(plain), KINDS)["kind"] == "layers"
    got = script.resolve(script.parse("cmap = viridis_r"), META)
    assert (got["cmap"], got["colour"]) == ("viridis", {"reverse": True})
    got = script.resolve(script.parse("cmap = RdBu\nreverse = false"), META)
    assert got["cmap"] == "RdBu_r" and got["colour"]["reverse"] is True
    assert script.resolve(script.parse("cmap = rdbu_r"), META)["cmap"] == "RdBu_r"
    with pytest.raises(script.ScriptError, match="'jet_r' is not one of"):
        script.resolve(script.parse("cmap = jet_r"), META)


def test_a_plain_var_line_is_the_field_that_overlays_are_drawn_over():
    def layers(text):
        return [(x["kind"], x.get("var") or x.get("u"))
                for x in script.parse(text)["layers"]["layers"]]

    assert layers("var = t2m\ncontour(u, colors='black')")[:2] == \
        [("pcolormesh", "t2m"), ("contour", "u")]
    assert layers("var = t2m\nbarbs(u, v)")[:2] == [("pcolormesh", "t2m"), ("barbs", "u")]
    # under the lines and the winds, over the land and the sea
    assert layers("ocean()\nvar = t2m\nrivers()\ncontour(u)")[:4] == \
        [("ocean", None), ("pcolormesh", "t2m"), ("rivers", None), ("contour", "u")]
    assert script.parse("var = t2m\nvectors(u, v)")["var"] == "t2m"
    # a call that shades has its own field, and the line gives way to it
    assert layers("var = t2m\ncontourf(u)\ncontour(v)")[:2] == \
        [("contourf", "u"), ("contour", "v")]
    assert script.parse("var = t2m\ncontour(t2m)") == {"var": "t2m", "kind": "contour"}
    # an empty var line: nothing underneath
    assert layers("var =\nvectors(u, v)")[0] == ("quiver", "u")


def test_a_definition_may_share_a_functions_name():
    assert script.parse("diff = u - v\nvar = diff") == {"var": "u - v"}
    assert script.parse("diff = u - v\nshade(diff.isel(time=1))")["var"] == "u - v"
    assert script.parse("shade(diff(t2m))")["var"] == "diff(t2m)"


def test_an_overlay_written_alone_is_drawn_over_the_field_on_screen():
    text, state = script.read("vectors(u, v)", KINDS, shown="t2m")
    assert text == "var = t2m\nvectors(u, v)" and state["var"] == "t2m"
    assert [x["kind"] for x in state["layers"]["layers"][:2]] == ["pcolormesh", "quiver"]
    text, state = script.read('rivers(color="blue")\nborders()', KINDS, shown="u")
    assert text.startswith("var = u\n")
    assert [x["kind"] for x in state["layers"]["layers"][:3]] == \
        ["pcolormesh", "rivers", "borders"]
    # nothing is added to a script that shades its own field, says `var =`,
    # is one plot of one field, or is not a figure at all
    for same in ("shade(t2m)\ncontour(u)", "var =\nvectors(u, v)", "contour(u)",
                 "cmap = viridis", "t2m.max()"):
        assert script.read(same, KINDS, shown="t2m")[0] == same
    # with nothing on screen to lie underneath, the map's own layers are not a figure
    with pytest.raises(script.ScriptError, match="names none"):
        script.read("rivers()", KINDS)
    # where the field shown cannot be a layer, the overlay is drawn alone
    text, state = script.read("vectors(u, v)", KINDS, shown="t2m - 273.15")
    assert text == "vectors(u, v)" and state["layers"]["layers"][0]["kind"] == "quiver"


def test_var_in_an_expression_is_the_field_on_screen():
    assert script.parse("var.max()", shown="t2m")["ask"] == {"what": "max", "var": "t2m"}
    assert script.parse("shade(var - 273.15)", shown="t2m")["var"] == "t2m - 273.15"
    # a var line in the script itself is what `var` means after it
    got = script.parse("var = u\nvar.isel(level=2).min()", shown="t2m")
    assert got["ask"] == {"what": "min", "var": "u", "level": 2}
    assert script.read("var.max()", META, shown="T2M")[1]["ask"]["var"] == "t2m"
    # with nothing on screen to mean, it is a name the file does not have
    with pytest.raises(script.ScriptError, match="'var' is not a variable"):
        script.resolve(script.parse("var.max()"), META)


def test_mark_puts_the_answer_or_a_place_on_the_map():
    got = script.parse("mark(t2m.isel(time=1).max())")
    assert got == {"ask": {"what": "max", "var": "t2m", "time": 1, "mark": True}}
    assert script.resolve(got, META)["ask"]["mark"] is True
    assert "mark" not in script.parse("t2m.max()")["ask"]      # asking marks nothing
    got = script.resolve(script.parse("mark(lon=147.4, lat=-2)"), META)
    assert (got["point"], got["panel"]) == ([147.4, -2.0], "mark")


@pytest.mark.parametrize("text, says", [
    ("mark(t2m)", "mark\\(\\) takes a question or a place"),
    ("mark(lon=147)", "mark\\(\\) takes a question or a place"),
    ("mark(lon='a', lat=1)", "mark\\(\\) takes numbers"),
    ("mark(lon=1, lat=95)", "latitudes run from -90 to 90"),
    ("mark(t2m.max())\nu.min()", "one question to a script"),
    ("mark = 3\nshade(t2m)", "mark is a word of the script"),
])
def test_a_mark_that_names_nothing_is_refused(text, says):
    with pytest.raises(script.ScriptError, match=says):
        script.parse(text)


def test_one_level_in_a_list_is_a_line_at_that_value_not_a_count():
    def levels(text):
        stack = script.resolve(script.parse(f"shade(t2m)\ncontour(u, {text})"), KINDS)
        return L.clean(stack["layers"], ["t2m", "u"], {})["layers"][1]["options"]["levels"]

    assert levels("levels=[28]") == [28.0]
    assert levels("levels=[270, 280]") == [270.0, 280.0]
    assert levels("levels=10") == 10
    assert levels('colors=["red", "blue"], levels=[1, 2]') == [1.0, 2.0]
    with pytest.raises(script.ScriptError, match="a single value is a count"):
        levels("levels=1")


def test_a_projection_named_beside_a_figure_is_the_figures_own():
    got = script.resolve(script.parse(
        'view(proj="Mercator", lon0=120)\nshade(t2m)\ncontour(u)'), KINDS)
    assert got["proj"] == "" and "plon" not in got
    assert got["layers"]["figure"] == {"projection": "Mercator", "central_longitude": 120.0}
    # figure(projection=) said outright is kept; "auto" is left to the figure
    got = script.resolve(script.parse(
        'proj = Mercator\nshade(t2m)\ncontour(u)\nfigure(projection="Robinson")'), KINDS)
    assert got["layers"]["figure"]["projection"] == "Robinson"
    got = script.resolve(script.parse('view(proj="auto")\nshade(t2m)\nvectors(u, v)'),
                         {**KINDS, "projections": ["auto", "Mercator"]})
    assert got["proj"] == "" and got["layers"]["figure"] == {}
    # a single plot of its own kind is still drawn on the plain map only
    with pytest.raises(script.ScriptError, match="is drawn on the lonlat map"):
        script.resolve(script.parse('hist(t2m)\nview(proj="Mercator")'), KINDS)


def test_a_question_takes_no_argument():
    for text in ('t2m.max("time")', "t2m.min(dim='time')", 'mark(t2m.max("time"))',
                 't2m.mean("time")', "t2m.max(3)"):
        with pytest.raises(script.ScriptError,
                           match="is a field, not a question|takes a question|in view"):
            script.parse(text)


def test_an_overlay_goes_over_a_derived_field_too():
    meta = {**KINDS, "layer_expressions": True}
    text, got = script.read("vectors(u, v)", meta, shown="hypot(u, v)")
    assert text.startswith("var = hypot(u, v)") and got["var"] == "u"
    assert got["layers"]["layers"][0]["var"] == "hypot(u, v)"


# ------------------------------------------------------ a field over time

MONTHLY = {**META, "time_statistics": True, "steps": 36,
           "labels": [f"{y}-{m:02d}-01 00:00" for y in (2011, 2012, 2013)
                      for m in range(1, 13)]}


def test_a_field_reduced_over_time_is_a_field():
    got = script.resolve(script.parse('shade(t2m.sel(season="djf").mean("time"))'),
                         MONTHLY)
    assert got == {"var": "t2m.sel(season='DJF').mean('time')", "kind": "map"}
    for text, var in [
            ('shade(t2m.sel(month=[6, 7, 8]).max(dim="time"))',
             "t2m.sel(month=[6, 7, 8]).max('time')"),
            ('shade(t2m.sel(time=slice("2012-12", "2013-02")).min("time"))',
             "t2m.sel(time=slice('2012-12', '2013-02')).min('time')"),
            ('shade(t2m.isel(time=slice(0, 12)).mean("time"))',
             "t2m.isel(time=slice(0, 12)).mean('time')"),
            ('clim = t2m.sel(month=12).mean("time")\nshade(t2m - clim)',
             "t2m - t2m.sel(month=12).mean('time')"),
            ('x = t2m\nshade(x.mean("time") - 273.15)', "t2m.mean('time') - 273.15")]:
        assert script.resolve(script.parse(text), MONTHLY)["var"] == var
    # the step of the plain field is still picked; the mean has none to pick
    got = script.resolve(script.parse(
        'shade(t2m.sel(time="2012-12") - t2m.sel(month=12).mean("time"))'), MONTHLY)
    assert (got["var"], got["time"]) == ("t2m - t2m.sel(month=12).mean('time')", 23)
    asked = script.resolve(script.parse('t2m.sel(year=2012).mean("time").max()'), MONTHLY)
    assert asked["ask"] == {"what": "max", "var": "t2m.sel(year=2012).mean('time')"}
    marked = script.read('mark(var.mean("time").min())', MONTHLY, shown="t2m")[1]
    assert marked["ask"]["var"] == "t2m.mean('time')" and marked["ask"]["mark"]


def test_a_date_given_in_part_is_a_time_when_it_is_one_step():
    one = script.resolve(script.parse('shade(t2m.sel(time="2012-12"))'), MONTHLY)
    assert one["time"] == 23
    assert script.resolve(script.parse("time = 2012-1"), MONTHLY)["time"] == 12
    assert script.resolve(script.parse('shade(t2m.sel(time="2012-12-01 00:00"))'),
                          MONTHLY)["time"] == 23
    with pytest.raises(script.ScriptError, match="'2012' is 12 steps, 2012-01-01 00:00 "
                                                 "to 2012-12-01 00:00; reduce them"):
        script.resolve(script.parse('shade(t2m.sel(time="2012"))'), MONTHLY)
    with pytest.raises(script.ScriptError, match="no step is labelled '2015-01'"):
        script.resolve(script.parse('shade(t2m.sel(time="2015-01"))'), MONTHLY)
    # a bare number in the line form is still a step number
    assert script.resolve(script.parse("time = 20"), MONTHLY)["time"] == 20


@pytest.mark.parametrize("text, says", [
    ('shade(t2m.sel(month=12))', "picks several time steps.*reduce them"),
    ('shade(t2m.isel(time=slice(0, 3)))', "picks several time steps"),
    ('shade(t2m.mean())', "reduces over time and says so"),
    ('shade(t2m.mean("lat"))', "reduces over time and says so"),
    ('shade((t2m - u).mean("time"))', "a field of the file is reduced over time"),
    ('a = t2m - 1\nshade(a.mean("time"))', "a is not a field of the file as it stands"),
    ('shade(t2m.sel(month=13).mean("time"))', "takes 1 to 12"),
    ('shade(t2m.sel(season="winter").mean("time"))', "DJF, MAM, JJA or SON"),
    ('shade(t2m.sel(level=3).mean("time"))', "not a way to pick time steps"),
    ('shade(t2m.mean("time").mean("time"))', "reduced over time once"),
    ('shade(t2m.sel(time="December").mean("time"))', "takes a date such as"),
])
def test_a_reduction_that_does_not_parse_says_why(text, says):
    with pytest.raises(script.ScriptError, match=says):
        script.parse(text)


@pytest.mark.parametrize("meta, text, says", [
    (MONTHLY, 'shade(t2m.sel(year=1999).mean("time"))', "no time step matches"),
    (MONTHLY, 'shade(nope.mean("time"))', "'nope' is not a variable"),
    ({**MONTHLY, "scanning": True}, 'shade(t2m.mean("time"))', "still being read"),
    ({**MONTHLY, "time_statistics": False}, 'shade(t2m.mean("time"))',
     "for lat/lon files so far, not for an MPAS run"),
    ({**MONTHLY, "labels": [f"step {i}" for i in range(36)]},
     'shade(t2m.sel(month=1).mean("time"))', "carry no dates.*pick them by number"),
    ({**MONTHLY, "labels": MONTHLY["labels"][:12] * 3},
     'shade(t2m.sel(month=1).mean("time"))', "both labelled"),
])
def test_a_reduction_the_file_cannot_make_is_refused(meta, text, says):
    with pytest.raises(script.ScriptError, match=says):
        script.resolve(script.parse(text), meta)


def test_an_expression_has_one_spelling_however_it_was_typed():
    typed = script.resolve(script.parse('var = t2m.sel(month=12).mean("time")-273.15'),
                           MONTHLY)
    drawn = script.resolve(script.parse(
        "shade(t2m.sel(month=12).mean('time') - 273.15)"), MONTHLY)
    assert typed["var"] == drawn["var"] == "t2m.sel(month=12).mean('time') - 273.15"
    assert script.resolve(script.parse("var = hypot( u,v )"), META)["var"] == "hypot(u, v)"


def test_kind_shade_is_the_map():
    got = script.resolve(script.parse('var = t2m.mean("time")\nkind = shade'), MONTHLY)
    assert got == {"var": "t2m.mean('time')", "kind": "map"}


@pytest.mark.parametrize("text", ["view()", "view(region=)", "view(region=None)"])
def test_view_back_to_the_whole_area(text):
    assert script.parse(text) == {"region": None}
    assert script.parse('view(region=, proj="Mercator")') == \
        {"region": None, "proj": "Mercator"}
