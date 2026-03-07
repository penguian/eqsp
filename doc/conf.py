import os
import sys

sys.path.insert(0, os.path.abspath(".."))

from importlib.metadata import version as _pkg_version

project = "eqsp"
copyright = "2026, Paul Leopardi"
author = "Paul Leopardi"

release = _pkg_version("eqsp")
# The short X.Y version.
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_rtd_theme",
]

autodoc_mock_imports = ["mayavi", "mayavi.mlab", "PyQt5"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
