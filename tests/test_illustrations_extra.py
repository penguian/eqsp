import re
import numpy as np
import matplotlib.pyplot as plt
import pytest
from eqsp import illustrations as ill
from eqsp.utilities import ideal_collar_angle
from eqsp._private._partitions import (
    polar_colat,
    num_collars,
    ideal_region_list,
    round_to_naturals,
)

def test_project_point_set_stereo():
    plt.switch_backend('Agg')
    points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).T
    # Line 154: proj='stereo'
    # Line 163-166: ax=None
    # Line 175: color='red'
    ax = ill.project_point_set(points, ax=None, proj='stereo', color='red', show=False)
    assert len(ax.collections) == 1
    plt.close()

def test_project_point_set_dim3_error():
    # Line 179-180: dim=3 raises NotImplementedError
    points = np.zeros((4, 5))
    with pytest.raises(NotImplementedError):
        ill.project_point_set(points)

def test_project_s2_partition_stereo():
    plt.switch_backend('Agg')
    # Line 241: proj='stereo'
    # Line 251-254: ax=None
    # Line 271: N=10 often hits near-zero/boundary logic
    ax = ill.project_s2_partition(10, proj='stereo', ax=None, show=False)
    assert len(ax.lines) > 0
    plt.close()

def test_illustrate_eq_algorithm_dim3():
    plt.switch_backend('Agg')
    # Line 409-421: dim=3 exercises sub-collar visualization
    # Line 399: show_title=False -> title='none'
    ill.illustrate_eq_algorithm(3, 10, show_title=False, show=False)
    plt.close()

def test_illustration_steps_high_n():
    plt.switch_backend('Agg')
    # N=10 ensures n_collars > 1, hitting Line 528 and 585
    fig = plt.figure()
    ill.illustrate_steps_3_5(2, 10, show_title=True)
    ill.illustrate_steps_6_7(2, 10, show_title=True)
    plt.close()

def test_project_s2_partition_boundary_case():
    plt.switch_backend('Agg')
    # Use a small N that might trigger line 271 (boundary normalization)
    ax = ill.project_s2_partition(3, proj='eqarea', show=False)
    plt.close()

def test_indexing_steps_3_5():
    """Verify that y_1 corresponds to the first collar."""
    plt.switch_backend('Agg')
    dim = 3
    N = 99
    
    # Calculate expected value
    c_polar = polar_colat(dim, N)
    a_ideal = ideal_collar_angle(dim, N)
    n_collars = num_collars(N, c_polar, a_ideal)
    r_regions = ideal_region_list(dim, N, c_polar, n_collars)
    
    expected_y1 = r_regions[1]
    
    # Run illustration
    plt.figure()
    ill.illustrate_steps_3_5(dim, N, show_title=False)
    
    # Inspect texts
    ax = plt.gca()
    texts = [t.get_text() for t in ax.texts]
    
    # Find y_1 label
    y1_text = next((t for t in texts if "y_{1}" in t), None)
    assert y1_text is not None, "Could not find label for y_1"
    
    # Extract value
    m = re.search(r"=\s*([0-9]+\.[0-9]+)", y1_text)
    assert m is not None, "Could not parse value from y_1 label"
    
    val = float(m.group(1))
    assert abs(val - expected_y1) < 0.2, f"y_1 value {val} mismatch expected {expected_y1}"
    plt.close()

def test_indexing_steps_6_7():
    """Verify that m_1 corresponds to the first collar."""
    plt.switch_backend('Agg')
    dim = 3
    N = 99
    
    # Calculate expected value
    c_polar = polar_colat(dim, N)
    a_ideal = ideal_collar_angle(dim, N)
    n_collars = num_collars(N, c_polar, a_ideal)
    r_regions = ideal_region_list(dim, N, c_polar, n_collars)
    n_regions = round_to_naturals(N, r_regions)
    
    expected_m1 = n_regions[1]
    
    # Run illustration
    plt.figure()
    ill.illustrate_steps_6_7(dim, N, show_title=False)
    
    # Inspect texts
    ax = plt.gca()
    texts = [t.get_text() for t in ax.texts]
    
    m1_text = next((t for t in texts if "m_{1}" in t), None)
    assert m1_text is not None, "Could not find label for m_1"
    
    m = re.search(r"=\s*([0-9]+)", m1_text)
    assert m is not None, "Could not parse value from m_1 label"
    
    val = int(m.group(1))
    assert val == expected_m1, f"m_1 value {val} mismatch expected {expected_m1}"
    plt.close()
