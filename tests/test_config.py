"""The plain-text configuration files that sit beside a run."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from gmpas.config import (
    ConfigError,
    discover,
    read_domain,
    read_field_list,
    select_fields,
)

DOMAIN = """\
nlat     = 267
nlon     = 534
startlat = -20.0
endlat   =  20.0
startlon =  80.0
endlon   = 160.0"""          # deliberately no trailing newline, as written


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return p


# ------------------------------------------------------------ target domain


def test_a_domain_reads_with_ragged_spacing_and_no_final_newline(tmp_path):
    d = read_domain(write(tmp_path, "target_domain", DOMAIN))

    assert (d.nlat, d.nlon) == (267, 534)
    assert (d.startlat, d.endlat) == (-20.0, 20.0)
    assert (d.startlon, d.endlon) == (80.0, 160.0)


def test_bounds_are_edges_so_the_spacing_comes_out_isotropic(tmp_path):
    """The reading the numbers argue for.

    267 x 534 cells across 40 x 80 degrees gives exactly the same spacing in
    both directions when the bounds are edges. Read as first and last centre
    they would differ (40/266 against 80/533), which for a grid somebody chose
    to be square is the less likely intent.
    """
    d = read_domain(write(tmp_path, "target_domain", DOMAIN))

    assert d.dlat == pytest.approx(40.0 / 267.0)
    assert d.dlon == pytest.approx(80.0 / 534.0)
    assert d.dlat == pytest.approx(d.dlon, abs=1e-12)

    # centres sit half a cell inside each edge
    assert d.lats()[0] == pytest.approx(-20.0 + d.dlat / 2)
    assert d.lats()[-1] == pytest.approx(20.0 - d.dlat / 2)
    assert len(d.lats()) == 267 and len(d.lons()) == 534


def test_comments_and_blank_lines_are_ignored(tmp_path):
    text = "# the target\n\nnlat = 10\nnlon = 20   # cells\n" \
           "startlat = 0\nendlat = 5\nstartlon = 0\nendlon = 10\n"
    d = read_domain(write(tmp_path, "target_domain", text))
    assert (d.nlat, d.nlon) == (10, 20)


@pytest.mark.parametrize("text, message", [
    ("nlat = 10\n", "missing"),
    ("nlat = ten\nnlon = 2\nstartlat=0\nendlat=1\nstartlon=0\nendlon=1", "number"),
    ("nlat=10\nnlon=20\nstartlat=5\nendlat=-5\nstartlon=0\nendlon=1", "endlat"),
    ("nlat=10\nnlon=20\nstartlat=0\nendlat=5\nstartlon=9\nendlon=1", "endlon"),
    ("nlat=10\nnlon=20\nstartlat=-95\nendlat=5\nstartlon=0\nendlon=1", "-90"),
    ("nlat 10\n", "key = value"),
])
def test_a_broken_domain_says_what_is_wrong(tmp_path, text, message):
    with pytest.raises((ConfigError, FileNotFoundError), match=message):
        read_domain(write(tmp_path, "target_domain", text))


def test_the_domain_writes_a_usable_scrip_target(tmp_path):
    d = read_domain(write(tmp_path, "target_domain", DOMAIN))
    out = d.to_scrip(tmp_path / "dst.scrip.nc")

    with xr.open_dataset(out) as ds:
        assert ds.sizes["grid_size"] == 267 * 534
        assert ds.sizes["grid_corners"] == 4
        assert list(ds.grid_dims.values) == [534, 267]
        # areas are exact solid angles, and must sum to the domain's own
        band = np.radians(80.0) * (np.sin(np.radians(20.0))
                                   - np.sin(np.radians(-20.0)))
        assert float(ds.grid_area.values.sum()) == pytest.approx(band, rel=1e-10)


# -------------------------------------------------------------- field lists


def test_field_lists_tolerate_hand_editing(tmp_path):
    """Trailing spaces, blank lines, comments, no final newline."""
    text = "u10\nv10\n\nrainc  \n# skip this\nv \ntheta"
    assert read_field_list(write(tmp_path, "include_fields", text)) == [
        "u10", "v10", "rainc", "v", "theta"
    ]


def test_duplicates_collapse_but_order_is_kept(tmp_path):
    text = "theta\nrho\ntheta\nq2\n"
    assert read_field_list(write(tmp_path, "include_fields", text)) == [
        "theta", "rho", "q2"
    ]


# ----------------------------------------------------------------- selection


AVAILABLE = ["theta", "precipw", "t2m", "rho", "qv", "u", "v"]


def test_include_alone_defines_the_whole_set():
    sel, _ = select_fields(AVAILABLE, include=["theta", "t2m"], warn=False)
    assert sel == ["theta", "t2m"]


def test_exclude_alone_removes_from_everything():
    sel, _ = select_fields(AVAILABLE, exclude=["qv", "u", "v"], warn=False)
    assert sel == ["theta", "precipw", "t2m", "rho"]


def test_a_field_in_both_lists_is_kept_and_reported():
    """Include wins, loudly.

    Silently dropping something explicitly asked for is the worse failure:
    the output is simply missing a field, with nothing to explain why.
    """
    sel, notes = select_fields(AVAILABLE, include=["theta", "rho", "t2m"],
                               exclude=["rho", "t2m", "qv"], warn=False)

    assert sel == ["theta", "rho", "t2m"]            # both conflicts survive
    assert any("include wins" in n for n in notes)
    assert any("rho" in n and "t2m" in n for n in notes)


def test_the_conflict_also_raises_a_warning():
    with pytest.warns(UserWarning, match="include wins"):
        select_fields(AVAILABLE, include=["rho"], exclude=["rho"])


def test_requested_fields_missing_from_the_data_are_reported():
    sel, notes = select_fields(AVAILABLE, include=["theta", "nosuchfield"],
                               warn=False)
    assert sel == ["theta"]
    assert any("not in the data" in n and "nosuchfield" in n for n in notes)


def test_no_lists_means_everything():
    sel, _ = select_fields(AVAILABLE, warn=False)
    assert sel == AVAILABLE


# ------------------------------------------------------------------ discovery


def test_the_files_are_found_in_the_working_directory(tmp_path, monkeypatch):
    write(tmp_path, "target_domain", DOMAIN)
    write(tmp_path, "include_fields", "theta\nrho\n")
    monkeypatch.chdir(tmp_path)

    cfg = discover()
    assert set(cfg.found) == {"domain", "include"}
    assert cfg.domain.nlat == 267
    assert cfg.include == ["theta", "rho"]
    assert cfg.exclude is None


def test_a_directory_can_be_given_instead(tmp_path):
    write(tmp_path, "target_domain", DOMAIN)
    assert discover(tmp_path).domain.nlon == 534


def test_a_missing_domain_explains_what_to_write(tmp_path):
    cfg = discover(tmp_path)
    with pytest.raises(ConfigError, match="target_domain"):
        cfg.require_domain()


def test_discovery_warns_when_the_lists_contradict(tmp_path):
    write(tmp_path, "target_domain", DOMAIN)
    write(tmp_path, "include_fields", "theta\nrho\n")
    write(tmp_path, "exclude_fields", "rho\nqv\n")

    with pytest.warns(UserWarning, match="include takes precedence"):
        discover(tmp_path)


def test_discovery_is_quiet_when_the_lists_agree(tmp_path):
    write(tmp_path, "target_domain", DOMAIN)
    write(tmp_path, "include_fields", "theta\n")
    write(tmp_path, "exclude_fields", "qv\n")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        discover(tmp_path)
