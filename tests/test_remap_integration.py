"""End-to-end check against a real ESMF_RegridWeightGen.

Every other remap test mocks `shutil.which`/`subprocess.run`, on purpose --
that is what keeps CI green without installing ESMF (see docs/REMAPPING.md
and issue 34). This file is the other half: it exercises the real binary, so
a change to the ESMF command line, the SCRIP files gmpas writes, or the
weight-file schema ESMF returns gets caught somewhere.

Skipped unless ESMF_RegridWeightGen is on PATH -- exactly how test_generate.py
treats JIGSAW. gmpas does not install ESMF itself (issue 34), so getting a
copy on PATH for a local run means pointing at a separate environment, never
the one gmpas itself runs in, e.g.:

    PATH=/path/to/some/other/conda/env/bin:$PATH pytest tests/test_remap_integration.py
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from conftest import write_mesh
from gmpas.config import TargetDomain
from gmpas.remap import Weights, ensure_weights

pytestmark = pytest.mark.skipif(
    shutil.which("ESMF_RegridWeightGen") is None,
    reason="ESMF_RegridWeightGen not on PATH -- install it in a separate "
           "environment from the one gmpas runs in, then put that "
           "environment's bin/ on PATH for this run. See docs/REMAPPING.md.",
)


def test_esmf_generates_weights_gmpas_can_load_and_apply(tmp_path):
    centres = [(lon, lat) for lon in (1.0, 3.0, 5.0, 7.0)
                          for lat in (-1.5, 0.0, 1.5)]
    mesh_path = tmp_path / "mesh.nc"
    write_mesh(mesh_path, centres, radius_deg=1.1)

    domain = TargetDomain(nlat=6, nlon=12,
                          startlat=-2.0, endlat=2.0,
                          startlon=0.0, endlon=8.0)

    weights_path, built = ensure_weights(mesh_path, domain, tmp_path,
                                         method="conserve")
    assert built is True
    assert weights_path.exists()

    weights = Weights.load(weights_path)
    assert weights.n_a == len(centres)
    assert weights.n_b == domain.nlat * domain.nlon
    assert weights.row.size > 0
    assert weights.row.max() < weights.n_b
    assert weights.col.max() < weights.n_a

    # A constant field is the sharpest conservation check: any coefficient or
    # frac_b handling mistake shows up directly as error here. Compare via
    # conservation_error(), not raw dst values -- partially covered
    # destination cells are legitimately not equal to the source constant in
    # dst itself (see the frac_b note on Weights.conservation_error);
    # dst / frac_b would be, but the raw integral is what conserves.
    constant = np.full(weights.n_a, 3.0)
    dst = weights.apply(constant)
    assert (dst != 0.0).any()
    assert weights.conservation_error(constant, dst) < 1e-6
