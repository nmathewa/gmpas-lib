"""What a field is and its units, from the models' own tables (gmpas.variables),
for a file that does not say."""

from __future__ import annotations

import numpy as np
import xarray as xr

from gmpas import data, variables


def test_mpas_names_come_from_the_mpas_registry():
    got = variables.lookup("qv", model="mpas")
    assert got == {"units": "kg kg-1", "description": "Water vapor mixing ratio",
                   "source": "MPAS core_atmosphere registry"}
    assert variables.lookup("theta", model="mpas")["units"] == "K"


def test_a_name_is_not_looked_up_outside_its_model():
    """MPAS's `u` is the normal velocity on edges; a lat/lon file's `u` is not."""
    assert variables.lookup("u", model="") is None
    assert variables.lookup("qv") is None


def test_an_era5_field_is_matched_by_its_grib_code():
    got = variables.lookup("t", {"GRIB_paramId": 130, "GRIB_typeOfLevel": "isobaricInhPa"})
    assert got["units"] == "K" and "GRIB 130" in got["source"]
    q = variables.lookup("q", {"GRIB_paramId": 133, "GRIB_typeOfLevel": "isobaricInhPa"})
    assert q["units"] == "kg kg-1"


def test_the_files_own_attributes_come_first():
    own = {"long_name": "my temperature", "units": "degC"}
    assert variables.filled("theta", own, "mpas") is own
    half = variables.filled("theta", {"long_name": "theta, mine"}, "mpas")
    assert half == {"long_name": "theta, mine", "units": "K",
                    "units_source": "MPAS core_atmosphere registry"}


def test_a_label_says_what_the_table_knows():
    da = xr.DataArray(np.zeros(3), dims="nCells", name="precipw")
    assert data.field_label(da) == "precipw"
    label = data.field_label(da, variables.filled("precipw", {}, "mpas"))
    assert label.startswith("precipitable water [") and "kg" in label


def test_every_unit_in_the_table_is_plain_text():
    """MPAS writes kg kg^{-1}: the table holds the UDUNITS form pint reads."""
    table = variables._table()
    assert not [u for u, *_ in table["names"].values() if "^{" in u or "{-" in u]
