"""
EQSP Tests: Private Region Props doctests bridge

Copyright Paul Leopardi 2026
"""

import doctest
from eqsp._private import _region_props

def test_doctests():
    """Test function test_doctests."""
    results = doctest.testmod(_region_props)
    assert results.failed == 0
