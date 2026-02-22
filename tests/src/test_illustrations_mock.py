"""
EQSP Tests: Illustrations Mock features

Copyright Paul Leopardi 2026
"""

# pylint: disable=import-outside-toplevel

import doctest
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


def test_doctests():
    """Test function test_doctests."""
    from eqsp import illustrations

    with patch("eqsp.illustrations.plt"):
        results = doctest.testmod(illustrations)
        assert results.failed == 0


# Ensure eqsp is importable


class TestIllustrations(unittest.TestCase):
    """Test suite for the illustrations module using mocks for matplotlib."""
    def setUp(self):
        # Patch matplotlib.pyplot in eqsp.illustrations
        self.plt_patcher = patch("eqsp.illustrations.plt")
        self.mock_plt = self.plt_patcher.start()

        # Configure mock_plt
        self.mock_fig = MagicMock()
        self.mock_ax = MagicMock()
        self.mock_plt.figure.return_value = self.mock_fig
        self.mock_fig.add_subplot.return_value = self.mock_ax
        self.mock_plt.gca.return_value = self.mock_ax
        self.mock_plt.get_backend.return_value = "Agg"  # Non-interactive

    def tearDown(self):
        self.plt_patcher.stop()

    def test_show_s2_partition_not_implemented(self):
        """Verify NotImplementedError for 3D partition visualization."""
        from eqsp import illustrations

        with self.assertRaises(NotImplementedError):
            illustrations.show_s2_partition(4)

    def test_project_s3_partition_not_implemented(self):
        """Verify NotImplementedError for S3 partition projection."""
        from eqsp import illustrations

        with self.assertRaises(NotImplementedError):
            illustrations.project_s3_partition(4)

    def test_show_s2_sphere_not_implemented(self):
        """Verify NotImplementedError for showing the sphere."""
        from eqsp import illustrations

        with self.assertRaises(NotImplementedError):
            illustrations.show_s2_sphere()

    def test_show_r3_point_set_not_implemented(self):
        """Verify NotImplementedError for R3 point set visualization."""
        from eqsp import illustrations

        with self.assertRaises(NotImplementedError):
            illustrations.show_r3_point_set(None)

    def test_show_s2_region_not_implemented(self):
        """Verify NotImplementedError for S2 region visualization."""
        from eqsp import illustrations

        with self.assertRaises(NotImplementedError):
            illustrations.show_s2_region(None)

    def test_project_point_set_invalid_dim(self):
        """Verify ValueError for invalid point set dimensions."""
        from eqsp import illustrations

        points = np.array([[1, 0], [0, 1]]).T
        with self.assertRaises(ValueError):
            illustrations.project_point_set(points)

    def test_project_point_set_invalid_proj(self):
        """Verify ValueError for invalid projection types."""
        from eqsp import illustrations

        points = np.array([[1, 0, 0], [0, 1, 0]]).T
        with self.assertRaises(ValueError):
            illustrations.project_point_set(points, proj="invalid")

    def test_project_s2_partition_invalid_proj(self):
        """Verify ValueError for invalid partition projection types."""
        from eqsp import illustrations

        with self.assertRaises(ValueError):
            illustrations.project_s2_partition(4, proj="invalid")

    def test_illustrate_eq_algorithm(self):
        """Verify basic behavior of the EQ algorithm illustration."""
        from eqsp import illustrations

        illustrations.illustrate_eq_algorithm(2, 4)
        # Verify subplots were created
        self.assertTrue(self.mock_plt.subplot.called)
        self.assertTrue(self.mock_plt.plot.called)
