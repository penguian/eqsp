"""
PyEQSP Tests: High-N Histogram Stress Tests

Copyright Paul Leopardi 2026
"""

from math import pi

import numpy as np
import pytest

from eqsp._private._histograms import lookup_s2_region


@pytest.mark.slow
def test_high_n_determinism():
    """
    Stress test for Option C: Index Rotation.
    Uses N=10,000 with a jump at index N/2.
    """
    N = 10_000
    shift = pi  # 180 degree rotation

    # Generate longitudes: [pi, ... 2*pi, 0, ... pi]
    longs = (np.linspace(0, 2 * pi, N + 1) + shift) % (2 * pi)

    s_regions = np.zeros((2, 2, N))
    s_regions[0, 0, :] = longs[:-1]  # starts
    s_regions[0, 1, :] = longs[1:]  # ends
    s_regions[1, :, :] = pi / 4
    c_regions = np.array([0, N, N])
    s_cap = np.array([0, pi / 2, pi])

    # 1. Find the wrapping region
    jumps = np.where(longs[:-1] > longs[1:])[0]
    jump_idx = jumps[0]

    # 2. Test Point AT THE CUSP (Very high end)
    test_cusp = np.array([[2 * pi - 1e-12], [pi / 4]])
    r_idx_cusp = lookup_s2_region(test_cusp, s_regions, s_cap, c_regions)

    # Expectation: Must fall exactly in the wrapped jump region.
    expected_cusp_idx = jump_idx + 1
    assert r_idx_cusp[0] == expected_cusp_idx, (
        f"Expected {expected_cusp_idx}, got {r_idx_cusp[0]}"
    )

    # 3. Test Point AFTER THE JUMP (Very low end)
    test_low = np.array([[1e-12], [pi / 4]])
    r_idx_low = lookup_s2_region(test_low, s_regions, s_cap, c_regions)

    # Expectation: Should fall in the next region after the jump.
    expected_low_idx = jump_idx + 2
    assert r_idx_low[0] == expected_low_idx, (
        f"Expected {expected_low_idx}, got {r_idx_low[0]}"
    )
