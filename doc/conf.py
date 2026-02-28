import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "eqsp"
copyright = "2026, Paul Leopardi"
author = "Paul Leopardi"
version = "0.98.0"
release = "0.98.0"

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
