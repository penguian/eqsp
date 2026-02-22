"""
EQSP Tests: Private Region Props doctests bridge

Copyright Paul Leopardi 2026
"""

import doctest

from eqsp._private import _region_props
from eqsp.partitions import eq_regions
import numpy as np


def test_doctests():
    """Test function test_doctests."""
    results = doctest.testmod(
        _region_props,
        extraglobs={"eq_regions": eq_regions, "np": np},
    )
    assert results.failed == 0
