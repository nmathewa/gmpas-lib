"""Plain-text configuration: a target domain, and which fields to remap.

Three files, all optional to *find* but meaningful when present:

    target_domain     nlat/nlon and the domain bounds, as `key = value`
    include_fields    one variable name per line: remap only these
    exclude_fields    one variable name per line: remap everything but these

They are deliberately dumb formats -- no YAML, no JSON -- because they get
edited by hand next to a run and pasted into job scripts.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

from .paths import resolve_path

#: keys a target domain must define
DOMAIN_KEYS = ("nlat", "nlon", "startlat", "endlat", "startlon", "endlon")

#: the filenames looked for, by role
CONFIG_NAMES = {
    "domain": "target_domain",
    "include": "include_fields",
    "exclude": "exclude_fields",
}


class ConfigError(ValueError):
    """A configuration file that cannot be used as written."""


# --------------------------------------------------------------- discovery


@dataclass
class Config:
    """Whatever configuration was found, and where it came from."""

    directory: Path
    domain: "TargetDomain | None" = None
    include: list[str] | None = None
    exclude: list[str] | None = None
    found: dict[str, Path] = None            # role -> path, for reporting

    def require_domain(self) -> "TargetDomain":
        if self.domain is None:
            raise ConfigError(
                f"no {CONFIG_NAMES['domain']!r} in {self.directory}. A target "
                f"domain is required; write one with nlat, nlon, startlat, "
                f"endlat, startlon and endlon."
            )
        return self.domain

    def select(self, available, warn=True):
        """Apply the discovered include/exclude lists to what a file holds."""
        return select_fields(available, self.include, self.exclude, warn=warn)


def discover(directory=None, warn: bool = True) -> Config:
    """Look for the configuration files beside you.

    Defaults to the working directory, because these files are meant to sit
    next to a run and be edited by hand -- naming them on every invocation
    would defeat the point.
    """
    where = Path(directory).expanduser() if directory else Path.cwd()
    found: dict[str, Path] = {}
    for role, name in CONFIG_NAMES.items():
        candidate = where / name
        if candidate.is_file():
            found[role] = candidate

    cfg = Config(directory=where, found=found)
    if "domain" in found:
        cfg.domain = read_domain(found["domain"])
    if "include" in found:
        cfg.include = read_field_list(found["include"])
    if "exclude" in found:
        cfg.exclude = read_field_list(found["exclude"])

    if warn and cfg.include and cfg.exclude:
        overlap = [f for f in cfg.include if f in set(cfg.exclude)]
        if overlap:
            warnings.warn(
                f"{len(overlap)} field(s) listed in both "
                f"{CONFIG_NAMES['include']} and {CONFIG_NAMES['exclude']}; "
                f"include takes precedence: {', '.join(overlap)}",
                stacklevel=2,
            )
    return cfg


# ------------------------------------------------------------ target domain


@dataclass(frozen=True)
class TargetDomain:
    """A regular lat-lon grid, given as bounds and a cell count.

    `startlat`/`endlat` bound the domain; `nlat` counts cells across it. So
    the spacing is `(endlat - startlat) / nlat` and centres sit half a cell
    inside each edge.

    That is the reading the numbers themselves argue for: a 267 x 534 grid
    spanning 40 by 80 degrees gives exactly 0.1498127 degrees in both
    directions as bounds, and two slightly different spacings if the bounds
    are read as first and last centre instead. It also means the grid covers
    exactly the requested box -- the alternative reading would overhang it by
    half a cell on every side.

    The extent is not treated as sacred beyond that. The target need only be
    roughly the region asked for: cells reaching outside the source mesh come
    back unmapped, and a target narrower than the mesh simply crops. Both are
    normal. What would not be acceptable is a systematic shift of whole
    degrees, and reading the bounds as edges avoids any shift at all.
    """

    nlat: int
    nlon: int
    startlat: float
    endlat: float
    startlon: float
    endlon: float

    @property
    def dlat(self) -> float:
        return (self.endlat - self.startlat) / self.nlat

    @property
    def dlon(self) -> float:
        return (self.endlon - self.startlon) / self.nlon

    def lats(self) -> np.ndarray:
        """Cell-centre latitudes."""
        return self.startlat + self.dlat * (np.arange(self.nlat) + 0.5)

    def lons(self) -> np.ndarray:
        """Cell-centre longitudes."""
        return self.startlon + self.dlon * (np.arange(self.nlon) + 0.5)

    @property
    def size(self) -> int:
        return self.nlat * self.nlon

    def __str__(self) -> str:
        return (f"{self.nlon} x {self.nlat} cells, "
                f"lon {self.startlon} .. {self.endlon}, "
                f"lat {self.startlat} .. {self.endlat}, "
                f"{self.dlon:.6g} x {self.dlat:.6g} deg")

    # -- output ----------------------------------------------------------

    def to_scrip(self, out_path: str | Path) -> Path:
        """Write this grid as a SCRIP file, ready to be a remap destination.

        Cell areas are exact solid angles, `dlon * (sin(north) - sin(south))`,
        rather than the `dlon * dlat * cos(lat)` approximation -- a remapper
        compares them against its own computed areas and the difference shows
        up as a conservation error.
        """
        lon2, lat2 = np.meshgrid(self.lons(), self.lats())
        cx = np.radians(lon2.ravel())
        cy = np.radians(lat2.ravel())
        hx, hy = np.radians(self.dlon) / 2.0, np.radians(self.dlat) / 2.0

        corner_lon = np.stack([cx - hx, cx + hx, cx + hx, cx - hx], axis=-1)
        corner_lat = np.stack([cy - hy, cy - hy, cy + hy, cy + hy], axis=-1)
        area = (np.sin(cy + hy) - np.sin(cy - hy)) * 2.0 * hx

        out = Path(out_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        xr.Dataset(
            {
                "grid_dims": ("grid_rank",
                              np.array([self.nlon, self.nlat], dtype=np.int32)),
                "grid_center_lat": ("grid_size", cy, {"units": "radians"}),
                "grid_center_lon": ("grid_size", cx, {"units": "radians"}),
                "grid_corner_lat": (("grid_size", "grid_corners"), corner_lat,
                                    {"units": "radians"}),
                "grid_corner_lon": (("grid_size", "grid_corners"), corner_lon,
                                    {"units": "radians"}),
                "grid_area": ("grid_size", area, {"units": "radian^2"}),
                "grid_imask": ("grid_size", np.ones(self.size, dtype=np.int32),
                               {"units": "unitless"}),
            },
            attrs={"title": "regular lat-lon target", "history": "written by gmpas"},
        ).to_netcdf(out)
        return out


def read_domain(path: str | Path) -> TargetDomain:
    """Read a `key = value` target domain file."""
    src = resolve_path(path)
    if not src.exists():
        raise FileNotFoundError(f"No such target domain file: {src}")

    values: dict[str, str] = {}
    for lineno, raw in enumerate(src.read_text().splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise ConfigError(
                f"{src.name} line {lineno}: expected 'key = value', got {raw!r}"
            )
        key, _, val = line.partition("=")
        values[key.strip().lower()] = val.strip()

    missing = [k for k in DOMAIN_KEYS if k not in values]
    if missing:
        raise ConfigError(f"{src.name} is missing {missing}; needs {list(DOMAIN_KEYS)}")

    try:
        domain = TargetDomain(
            nlat=int(values["nlat"]), nlon=int(values["nlon"]),
            startlat=float(values["startlat"]), endlat=float(values["endlat"]),
            startlon=float(values["startlon"]), endlon=float(values["endlon"]),
        )
    except ValueError as exc:
        raise ConfigError(f"{src.name}: could not read a number ({exc})") from exc

    if domain.nlat < 1 or domain.nlon < 1:
        raise ConfigError(f"{src.name}: nlat and nlon must be at least 1")
    if domain.endlat <= domain.startlat:
        raise ConfigError(
            f"{src.name}: endlat ({domain.endlat}) must exceed "
            f"startlat ({domain.startlat})"
        )
    if domain.endlon <= domain.startlon:
        raise ConfigError(
            f"{src.name}: endlon ({domain.endlon}) must exceed "
            f"startlon ({domain.startlon})"
        )
    if not (-90.0 <= domain.startlat and domain.endlat <= 90.0):
        raise ConfigError(
            f"{src.name}: latitudes must lie within [-90, 90], got "
            f"{domain.startlat} .. {domain.endlat}"
        )
    return domain


# ------------------------------------------------------------ field lists


def read_field_list(path: str | Path) -> list[str]:
    """Read one variable name per line, in file order, duplicates removed.

    Blank lines and `#` comments are skipped, and names are stripped -- the
    lists get hand-edited, so a stray trailing space is expected rather than
    an error.
    """
    src = resolve_path(path)
    if not src.exists():
        raise FileNotFoundError(f"No such field list: {src}")

    seen: set[str] = set()
    fields: list[str] = []
    for raw in src.read_text().splitlines():
        name = raw.split("#", 1)[0].strip()
        if name and name not in seen:
            seen.add(name)
            fields.append(name)
    return fields


def select_fields(available, include=None, exclude=None,
                  warn=True) -> tuple[list[str], list[str]]:
    """Decide which variables to act on.

    `include` names the whole set when given; `exclude` removes from it, or
    from everything available when there is no include list.

    A name appearing in **both** lists is contradictory. Include wins, and it
    is reported: silently dropping a field somebody explicitly asked for is
    the worse failure, since the output would simply be missing without
    explanation.

    Returns the selected names, in the order they appear in the file, and the
    list of notes worth showing the user.
    """
    available = list(available)
    have = set(available)
    include = list(include) if include else []
    exclude = list(exclude) if exclude else []
    notes: list[str] = []

    conflict = [f for f in include if f in set(exclude)]
    if conflict:
        notes.append(
            f"{len(conflict)} field(s) in both include and exclude — "
            f"include wins: {', '.join(conflict)}"
        )

    if include:
        chosen = [f for f in include if f not in set(exclude) or f in set(conflict)]
    else:
        chosen = [f for f in available if f not in set(exclude)]

    missing = [f for f in chosen if f not in have]
    if missing:
        notes.append(
            f"{len(missing)} requested field(s) not in the data: "
            f"{', '.join(missing)}"
        )

    selected = [f for f in chosen if f in have]
    if not selected:
        notes.append("no fields selected")

    if warn:
        for note in notes:
            warnings.warn(note, stacklevel=2)
    return selected, notes
