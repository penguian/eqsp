import os
import sys

sys.path.insert(0, os.path.abspath(".."))

from unittest.mock import MagicMock

# Mock optional dependencies for headless doctest environments.
# We catch all exceptions because some libraries (like mayavi/vtk) may be
# installed but fail to initialize in headless CI runners.
try:
    import mayavi  # noqa: F401
    import PyQt5  # noqa: F401
except Exception:
    mock_mayavi = MagicMock()
    sys.modules["mayavi"] = mock_mayavi
    sys.modules["mayavi.mlab"] = mock_mayavi
    sys.modules["PyQt5"] = MagicMock()
    sys.modules["PyQt5.QtWidgets"] = MagicMock()
    sys.modules["PyQt5.QtCore"] = MagicMock()

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

# The project name used for branding and titles.
project = "PyEQSP: Python Equal Area Sphere Partitioning Library"

# The package name used for version lookup.
distribution_name = "pyeqsp"
copyright = "2026, Paul Leopardi"
author = "Paul Leopardi"

try:
    release = _pkg_version(distribution_name)
except PackageNotFoundError:
    release = "unknown"

# The short X.Y version.
version = ".".join(release.split(".")[:2]) if release != "unknown" else "unknown"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_rtd_theme",
    "sphinxcontrib.mermaid",
]

# Nitpicky mode ensures all internal and external links are valid.
nitpicky = True

# Intersphinx mapping for external documentation links.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

# Nitpick ignore patterns for common scientific docstring labels
# that aren't real classes.
nitpick_ignore = [
    ("py:class", "M"),
    ("py:class", "N"),
    ("py:class", "array_like"),
    ("py:class", "array-like"),
    ("py:class", "ndarray"),
    ("py:class", "np.ndarray"),
    ("py:class", "optional"),
    ("py:class", "shape"),
    ("py:class", "{'stereo'"),
    ("py:class", "'eqarea'}"),
    ("py:class", "{'long'"),
    ("py:class", "'short'"),
    ("py:class", "'none'}"),
]


doctest_global_setup = """
import numpy as np
from math import pi
from eqsp.utilities import *
from eqsp.partitions import *
from eqsp.point_set_props import *
from eqsp.region_props import *
from eqsp.histograms import *
"""

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "substitution",
    "colon_fence",
]
myst_heading_anchors = 3

autodoc_mock_imports = ["mayavi", "mayavi.mlab", "PyQt5"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
