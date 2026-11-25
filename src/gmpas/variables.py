"""What a field is and its units, from the models' own tables, for a file
that does not say.

`variables.json` is built from MPAS's registry (the names MPAS writes), WRF's
Registry.EM_COMMON and WPS's ungrib Vtables (GRIB codes, which ERA5 netCDF
files carry as GRIB_paramId), by a script kept outside the package; its
header says how. Nothing in it is written by hand.

A name means different things to different models -- MPAS's `u` is the
normal velocity on cell edges, a lat/lon file's `u` is the zonal wind -- so a
name is looked up only in the table of the model that wrote the file, and a
lat/lon file is matched by its GRIB code or not at all. A file's own
attributes always come first; this fills only what is missing.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

#: GRIB typeOfLevel, as cfgrib writes it, to the GRIB1 level type of a Vtable.
_LEVEL = {"isobaricInhPa": "100", "isobaricInPa": "100", "surface": "1",
          "heightAboveGround": "1", "meanSea": "1", "hybrid": "109"}


@lru_cache(maxsize=1)
def _table() -> dict:
    return json.loads(Path(__file__).with_name("variables.json").read_text())


def lookup(name: str, attrs: dict | None = None, model: str = "") -> dict | None:
    """{"units", "description", "source"} for a field, or None.

    `model` is "mpas" for MPAS output and "" for anything else; `attrs` are
    the field's attributes, searched for a GRIB code."""
    table = _table()
    attrs = attrs or {}
    code = attrs.get("GRIB_paramId")
    if code is not None:
        level = _LEVEL.get(str(attrs.get("GRIB_typeOfLevel", "")), "")
        prefix = f"ecmwf:{int(code)}:"
        keys = [f"{prefix}{level}", f"{prefix}100", f"{prefix}1"] + sorted(
            k for k in table["grib"] if k.startswith(prefix))   # any level: same units
        for key in keys:
            entry = table["grib"].get(key)
            if entry:
                return {"units": entry[0], "description": entry[1] or entry[3],
                        "source": f"{entry[2]}, GRIB {int(code)}"}
    if model == "mpas":
        entry = table["names"].get(name)
        if entry and entry[2].startswith("MPAS"):
            return {"units": entry[0], "description": entry[1],
                    "source": f"{entry[2]} registry"}
    return None


def filled(name: str, attrs: dict, model: str = "") -> dict:
    """`attrs` with a missing long_name and units taken from the tables, and
    "units_source" naming where they came from when they did."""
    if attrs.get("long_name") and attrs.get("units") not in (None, ""):
        return attrs
    known = lookup(name, attrs, model)
    if not known:
        return attrs
    out = dict(attrs)
    if not out.get("long_name") and known["description"]:
        out["long_name"] = known["description"]
    if out.get("units") in (None, "") and known["units"]:
        out["units"] = known["units"]
        out["units_source"] = known["source"]
    return out
