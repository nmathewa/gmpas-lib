"""Preprocessing: everything that happens *before* there is model output.

The rest of gmpas is postprocessing -- it opens a run and renders, remaps or
exports it. This subpackage is the other end: building a mesh and looking at it
on its own, with no history file in sight.

Nothing here modifies the postprocessing path. `gmpas view`, `plot`, `remap`
and friends behave exactly as before; the preprocessing commands live under
`gmpas prep` and reuse the viewer's plumbing (`ViewIndex`, the palette PNG
encoder, the coastline overlay, `bind`) by importing it, never by changing it.

`layout.page()` is the shared browser layout for this section: the same map,
graticule, scale bar and extent box, with a slot for whatever controls a given
step needs. Mesh viewing is the first user; mesh generation is meant to be the
second.
"""

from __future__ import annotations

from .layout import page
from .meshview import MeshViewer, serve

__all__ = ["MeshViewer", "page", "serve"]
