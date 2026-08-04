"""Concentric refinement rings for JIGSAW — a template to copy and edit.

Any number of rings about one centre. To change the mesh, edit CENTER and RINGS
below and nothing else; the transition widths, hfun_min, and the checks all
follow from them.

    gmpas prep hfun     hfun_concentric.py --check     # the numbers, no server
    gmpas prep hfun     hfun_concentric.py             # look at it
    gmpas prep generate hfun_concentric.py -o mesh/    # build it

The contract this satisfies is the mini-tutorial's (Duda, MPAS/WRF Users
Workshop 2026): a module-level `hfun_min` in km, and `get_hfun(lon, lat)`
taking RADIANS and returning KM.

Rings about *different* centres are a different problem — see the note at the
bottom of this file.
"""

import numpy as np

# ============================================================== EDIT FROM HERE

#: centre of every ring, in degrees
CENTER_LON, CENTER_LAT = -95.0, 38.0

#: The rings, finest first. Each is (radius_km, spacing_km): hold `spacing_km`
#: out to `radius_km`, then transition to the next entry. The last entry is the
#: background everywhere beyond, and has no radius.
#:
#: Add a ring by adding a line. Change a resolution or a radius by editing one.
#: You do NOT set the transition widths -- they follow from SLOPE below, so a
#: bigger jump in resolution automatically gets the distance it needs.
RINGS = [
    (600.0,  3.0),      # 3 km out to 600 km from the centre
    (1400.0, 12.0),     # 12 km out to 1400 km
    (3200.0, 30.0),     # 30 km out to 3200 km
    (None,   60.0),     # 60 km everywhere beyond
]

#: km of cell size per km of distance, in every transition. The tutorial's
#: guidance is a few percent at most, with 0.03 generally safe. Giving every
#: transition the same slope is deliberate: it is the steepest one that limits
#: mesh quality, so there is nothing to gain by making the others gentler.
SLOPE = 0.03

# ================================================================ EDIT TO HERE

#: MPAS-Atmosphere's Earth radius (km)
R_EARTH = 6371.229

#: the contract's minimum grid distance: whatever the finest ring asks for
hfun_min = RINGS[0][1]


def _knots():
    """Turn RINGS into the breakpoints of a piecewise-linear h(r).

    A flat band, then a ramp, repeated. Between them the radii strictly
    increase, which is all `np.interp` needs, and beyond the last one it clamps
    -- which is exactly "h_max everywhere outside".
    """
    r, h = [0.0], [RINGS[0][1]]
    for (radius, spacing), (_, next_spacing) in zip(RINGS, RINGS[1:]):
        width = abs(next_spacing - spacing) / SLOPE      # implied by the slope
        r += [radius, radius + width]
        h += [spacing, next_spacing]

    _check(r)
    return np.array(r), np.array(h)


def _check(r):
    """Refuse the two mistakes the tutorial calls out by name.

    The cell-count figure is a rough reading of "ideally several hundred cells
    in diameter" as 2*radius/spacing, not a rule from the tutorial. Treat it as
    a prompt to think rather than a verdict.
    """
    # The breakpoints must march outwards. Anywhere they go backwards, a
    # transition has overrun the ring meant to follow it -- the "transition
    # regions begin to overlap" failure. Checked as plain monotonicity rather
    # than by indexing the pairs, because getting that indexing subtly wrong is
    # how a check silently passes a mesh it should have refused.
    for i in range(1, len(r)):
        if r[i] < r[i - 1]:
            raise ValueError(
                f"ring {(i - 1) // 2} begins at {r[i]:.0f} km, but the "
                f"transition before it does not finish until {r[i - 1]:.0f} km "
                f"-- they overlap. Move the rings apart, or make the jump in "
                f"resolution between them smaller."
            )

    for i, (radius, spacing) in enumerate(RINGS[:-1]):
        cells = 2.0 * radius / spacing
        if cells < 200:
            print(f"note: ring {i} is ~{cells:.0f} cells across at "
                  f"{spacing:g} km; the guidance is several hundred")


#: built once at import. `get_hfun` may be handed millions of points, and the
#: checks above should speak once rather than on every frame.
R_KNOTS, H_KNOTS = _knots()


def get_hfun(lon, lat):
    """Grid distance (km) at each (lon, lat), both in RADIANS.

    Called once with whole arrays, so it is allowed to be expensive.
    """
    r = _radius_km(lon, lat)

    # a piecewise-linear h(r) is exactly an interpolation over the breakpoints,
    # so any number of rings costs one call and no extra code
    return np.interp(r, R_KNOTS, H_KNOTS).reshape(np.shape(lon))


def _radius_km(lon, lat):
    """Great-circle distance (km) from the centre to each point."""
    lam, phi = np.radians(CENTER_LON), np.radians(CENTER_LAT)
    centre = np.array([np.cos(lam) * np.cos(phi),
                       np.sin(lam) * np.cos(phi),
                       np.sin(phi)])

    lon, lat = np.asarray(lon), np.asarray(lat)
    p = np.column_stack([(np.cos(lon) * np.cos(lat)).ravel(),
                         (np.sin(lon) * np.cos(lat)).ravel(),
                         np.sin(lat).ravel()])

    # clip because round-off can push the dot product just past +-1, where
    # arccos returns nan rather than 0 or pi
    return R_EARTH * np.arccos(np.clip(p @ centre, -1.0, 1.0))


if __name__ == "__main__":
    # `python hfun_concentric.py` prints the profile it describes
    print(f"hfun_min = {hfun_min:g} km, centre {CENTER_LAT}, {CENTER_LON}")
    for i in range(len(R_KNOTS) - 1):
        if H_KNOTS[i + 1] != H_KNOTS[i]:
            span = R_KNOTS[i + 1] - R_KNOTS[i]
            slope = (H_KNOTS[i + 1] - H_KNOTS[i]) / span
            print(f"  {R_KNOTS[i]:7.0f} - {R_KNOTS[i + 1]:7.0f} km   "
                  f"{H_KNOTS[i]:5.1f} -> {H_KNOTS[i + 1]:5.1f} km   "
                  f"slope {slope:.4f}")
        else:
            print(f"  {R_KNOTS[i]:7.0f} - {R_KNOTS[i + 1]:7.0f} km   "
                  f"{H_KNOTS[i]:5.1f} km")


# --------------------------------------------------------------------- notes
#
# Rings about DIFFERENT centres cannot be written as one h(r). Compute a full
# field for each and take the pointwise minimum:
#
#     def get_hfun(lon, lat):
#         return np.minimum(_ring_field(lon, lat, centre_a, ...),
#                           _ring_field(lon, lat, centre_b, ...))
#
# For a nested pair the child must return np.inf beyond its own transition, so
# the minimum falls back to the parent; the child and its transition must sit
# entirely inside the parent, and the two must agree on the resolution where
# they meet.
