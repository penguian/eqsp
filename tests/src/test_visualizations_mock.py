"""
PyEQSP Tests: Visualizations (Mock features)

Copyright Paul Leopardi 2026
"""

# pylint: disable=import-outside-toplevel

import doctest
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


def test_doctests():
    """Test function test_doctests."""
    with patch.dict(sys.modules, {"pyvista": MagicMock()}):
        try:
            from eqsp import visualizations

            with patch("eqsp.visualizations.pv"):
                results = doctest.testmod(visualizations)
                assert results.failed == 0
        finally:
            sys.modules.pop("eqsp.visualizations", None)


class TestVisualizationsSetup(unittest.TestCase):
    """Shared setUp/tearDown for all visualizations tests."""

    def setUp(self):
        self.mock_pv = MagicMock()
        self.mock_pv.OFF_SCREEN = True
        self.modules_patcher = patch.dict(
            sys.modules,
            {"pyvista": self.mock_pv},
        )
        self.modules_patcher.start()
        sys.modules.pop("eqsp.visualizations", None)

    def tearDown(self):
        self.modules_patcher.stop()
        sys.modules.pop("eqsp.visualizations", None)

    def _import_vis(self):
        import eqsp.visualizations as vis

        vis.pv = self.mock_pv
        return vis


# ---------------------------------------------------------------------------
# show_s2_sphere
# ---------------------------------------------------------------------------


class TestShowS2Sphere(TestVisualizationsSetup):
    """Test function TestShowS2Sphere."""

    def test_calls_add_mesh(self):
        """Test function test_calls_add_mesh."""
        vis = self._import_vis()
        pl = vis.show_s2_sphere()
        pl.add_mesh.assert_called_once()

    def test_accepts_custom_color_and_opacity(self):
        """Test function test_accepts_custom_color_and_opacity."""
        vis = self._import_vis()
        pl = vis.show_s2_sphere(opacity=0.5, color=(1, 0, 0))
        _, kwargs = pl.add_mesh.call_args
        self.assertEqual(kwargs["opacity"], 0.5)
        self.assertEqual(kwargs["color"], (1, 0, 0))


# ---------------------------------------------------------------------------
# show_r3_point_set
# ---------------------------------------------------------------------------


class TestShowR3PointSet(TestVisualizationsSetup):
    """Test function TestShowR3PointSet."""

    def _points(self):
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float).T

    def test_calls_add_mesh(self):
        """Test function test_calls_add_mesh."""
        vis = self._import_vis()
        pl = vis.show_r3_point_set(self._points())
        pl.add_mesh.assert_called_once()

    def test_show_sphere_calls_sphere(self):
        """Test function test_show_sphere_calls_sphere."""
        vis = self._import_vis()
        vis.show_r3_point_set(self._points(), show_sphere=True)
        self.assertTrue(vis.pv.Sphere.called)

    def test_save_file_calls_screenshot(self):
        """Test function test_save_file_calls_screenshot."""
        vis = self._import_vis()
        pl = vis.show_r3_point_set(self._points(), save_file="out.png")
        pl.screenshot.assert_called_once_with("out.png")


# ---------------------------------------------------------------------------
# show_s2_region
# ---------------------------------------------------------------------------


class TestShowS2Region(TestVisualizationsSetup):
    """Test function TestShowS2Region."""

    def _region(self):
        return np.array([[0.2, 0.8], [0.0, 1.0]])

    def test_calls_add_mesh(self):
        """Test function test_calls_add_mesh."""
        vis = self._import_vis()
        pl = vis.show_s2_region(self._region(), N=10)
        self.assertTrue(pl.add_mesh.called)


# ---------------------------------------------------------------------------
# show_s2_partition
# ---------------------------------------------------------------------------


class TestShowS2Partition(TestVisualizationsSetup):
    """Test function TestShowS2Partition."""

    def test_calls_add_mesh(self):
        """Test function test_calls_add_mesh."""
        vis = self._import_vis()
        pl = vis.show_s2_partition(4, show_points=True, show_sphere=True, show=False)
        self.assertTrue(pl.add_mesh.called)

    def test_title_long_calls_add_text(self):
        """Test function test_title_long_calls_add_text."""
        vis = self._import_vis()
        pl = vis.show_s2_partition(4, title="long", show=False)
        pl.add_text.assert_called_once()

    def test_title_custom_calls_add_text(self):
        """Test function test_title_custom_calls_add_text."""
        vis = self._import_vis()
        pl = vis.show_s2_partition(4, title="My Title", show=False)
        pl.add_text.assert_called_once()

    def test_title_none_no_add_text(self):
        """Test function test_title_none_no_add_text."""
        vis = self._import_vis()
        pl = vis.show_s2_partition(4, title="none", show=False)
        pl.add_text.assert_not_called()

    def test_save_file_calls_screenshot(self):
        """Test function test_save_file_calls_screenshot."""
        vis = self._import_vis()
        pl = vis.show_s2_partition(4, show=False, save_file="snap.png")
        pl.screenshot.assert_called_once_with("snap.png")


# ---------------------------------------------------------------------------
# project_point_set
# ---------------------------------------------------------------------------


class TestProjectPointSet(TestVisualizationsSetup):
    """Test function TestProjectPointSet."""

    def test_s2_points_stereo_calls_add_mesh(self):
        """Test function test_s2_points_stereo_calls_add_mesh."""
        vis = self._import_vis()
        points = np.eye(3)
        pl = vis.project_point_set(points, proj="stereo", show=False)
        pl.add_mesh.assert_called_once()

    def test_s2_points_eqarea_calls_add_mesh(self):
        """Test function test_s2_points_eqarea_calls_add_mesh."""
        vis = self._import_vis()
        points = np.eye(3)
        pl = vis.project_point_set(points, proj="eqarea", show=False)
        pl.add_mesh.assert_called_once()

    def test_s3_points_calls_add_mesh(self):
        """Test function test_s3_points_calls_add_mesh."""
        vis = self._import_vis()
        points = np.eye(4)
        pl = vis.project_point_set(points, proj="stereo", show=False)
        pl.add_mesh.assert_called_once()

    def test_invalid_dim_raises_value_error(self):
        """Test function test_invalid_dim_raises_value_error."""
        vis = self._import_vis()
        points = np.array([[1, 0], [0, 1]])
        with self.assertRaises(ValueError):
            vis.project_point_set(points)

    def test_invalid_proj_raises_value_error(self):
        """Test function test_invalid_proj_raises_value_error."""
        vis = self._import_vis()
        with self.assertRaises(ValueError):
            vis.project_point_set(np.eye(3), proj="invalid")


# ---------------------------------------------------------------------------
# project_s3_partition
# ---------------------------------------------------------------------------


class TestProjectS3Partition(TestVisualizationsSetup):
    """Test function TestProjectS3Partition."""

    def test_default_calls_add_mesh(self):
        """Test function test_default_calls_add_mesh."""
        vis = self._import_vis()
        pl = vis.project_s3_partition(
            4, show_points=True, show_surfaces=True, show=False
        )
        self.assertTrue(pl.add_mesh.called)

    def test_invalid_proj_raises_value_error(self):
        """Test function test_invalid_proj_raises_value_error."""
        vis = self._import_vis()
        with self.assertRaises(ValueError):
            vis.project_s3_partition(4, proj="invalid")

    def test_save_file_calls_screenshot(self):
        """Test function test_save_file_calls_screenshot."""
        vis = self._import_vis()
        pl = vis.project_s3_partition(4, show=False, save_file="s3.png")
        pl.screenshot.assert_called_once_with("s3.png")


if __name__ == "__main__":
    unittest.main()
