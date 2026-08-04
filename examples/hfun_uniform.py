"""A quasi-uniform mesh — the simplest hfun.py there is.

    gmpas prep hfun     hfun_uniform.py --check
    gmpas prep generate hfun_uniform.py -o mesh/

A constant distance function is not quite enough on its own. JIGSAW will
produce a mesh at the right resolution, but with 7-sided cells and less
desirable shapes, because nothing imposes icosahedral structure on it. To get
that structure, hand JIGSAW an initial point set:

    sphtri_subdiv ICOS.msh 120KM.msh 63        # from islas/mesh_tools
    gmpas prep generate hfun_uniform.py --init 120KM.msh -o mesh/

`ICOS.msh` ships with the mini-tutorial repository. The divisor sets the
resolution: with R = 6371.229 km the base icosahedron has dx = 7053.898 km, so
a divisor of n gives roughly dx/n.
"""

import numpy as np

#: the grid distance everywhere, in km
hfun_min = 120.0


def get_hfun(lon, lat):
    """Grid distance (km) at each (lon, lat), both in RADIANS."""
    return np.full(np.shape(lon), hfun_min)
