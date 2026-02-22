"""
EQSP Tests: Private Partitions doctests bridge

Copyright Paul Leopardi 2026
"""

import doctest
from eqsp._private import _partitions

def test_doctests():
    """Test function test_doctests."""
    results = doctest.testmod(_partitions)
    assert results.failed == 0
