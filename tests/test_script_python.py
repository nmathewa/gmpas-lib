"""The script written as Python with xarray would write it (gmpas/script.py).

It reads like Python and is never run as Python. Both forms come out as the
same settings, so the page applies either without knowing which it was.
"""

from __future__ import annotations

import pytest

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
    ("shade(u.sel(level=850))", "a level is picked by number for now"),
    ("shade(t2m.mean('time'))", ".mean\\(\\) is not understood"),
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
    ("shade(t2m)\nlevels = (,,2.5)", "line 2: .*text goes in quotes"),
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
                         "options": {"cmap": "RdBu_r", "levels": "270, 280, 290"}}
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
    ("contourf(t2m, stride=3)\nvectors(u, v)", "layers: layer 1 \\(contourf\\)"),
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
