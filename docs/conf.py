# Sphinx configuration for the Read the Docs site. The pages are the same
# Markdown files the README links to, read through MyST.
import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

from gmpas import __version__  # noqa: E402

project = "gmpas"
author = "Nirmal Mathew Alex"
copyright = "Nirmal Mathew Alex and contributors"
release = __version__
version = __version__

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
root_doc = "index"
exclude_patterns = [
    "_build",
    "demo/**",
    "benchmarks/**",
    "mcp-server.md",  # the MCP server project was removed
]

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 4
# GitHub draws ```mermaid fences; on the site they become diagrams too.
myst_fence_as_directive = ["mermaid"]
# GitHub-only fence languages (ssh-config) have no Pygments lexer; the block
# is still shown, unhighlighted.
suppress_warnings = ["misc.highlighting_failure"]

# The library documents itself without the drawing stack installed: the site
# build imports gmpas, not matplotlib or cartopy.
autodoc_mock_imports = ["matplotlib", "cartopy", "PIL", "cmocean"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"

html_theme = "furo"
html_title = f"gmpas {version}"
html_logo = "logo/gmpas.svg"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/nmathewa/gmpas-lib",
    "source_branch": "main",
    "source_directory": "docs/",
}
