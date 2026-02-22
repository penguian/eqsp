"""
EQSP Tests: Visualizations Mock features

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
    # visualizations patches mlab internally on import if missing,
    # so we must mock mayavi in sys.modules BEFORE the import.
    with patch.dict(sys.modules, {"mayavi": MagicMock(), "mayavi.mlab": MagicMock()}):
        try:
            from eqsp import visualizations

            with patch("eqsp.visualizations.mlab"):
                results = doctest.testmod(visualizations)
                assert results.failed == 0
        finally:
            # Ensure the module cache is cleaned up so we don't leak mocks
            sys.modules.pop("eqsp.visualizations", None)


def _make_mock_mlab():
    """Return a MagicMock that mimics the mlab engine scene query."""
    mock_mlab = MagicMock()
    # get_engine().current_scene is checked before creating a figure;
    # start as None so the first call creates a figure.
    mock_engine = mock_mlab.get_engine.return_value
    mock_engine.current_scene = None

    def set_scene(*_args, **_kwargs):
        mock_engine.current_scene = MagicMock()

    mock_mlab.figure.side_effect = set_scene
    return mock_mlab


class TestVisualizationsSetup(unittest.TestCase):
    """Shared setUp/tearDown for all visualizations tests."""

    def setUp(self):
        # Patch mayavi in sys.modules BEFORE importing eqsp.visualizations,
        # and remove any cached import so the mock takes effect.
        self.modules_patcher = patch.dict(
            sys.modules,
            {"mayavi": MagicMock(), "mayavi.mlab": MagicMock()},
        )
        self.modules_patcher.start()
        sys.modules.pop("eqsp.visualizations", None)

    def tearDown(self):
        self.modules_patcher.stop()
        sys.modules.pop("eqsp.visualizations", None)

    def _import_vis(self):
        import eqsp.visualizations as vis

        vis.mlab = _make_mock_mlab()
        return vis


# ---------------------------------------------------------------------------
# show_s2_sphere
# ---------------------------------------------------------------------------


class TestShowS2Sphere(TestVisualizationsSetup):
    """Test function TestShowS2Sphere."""
    def test_calls_mlab_mesh(self):
        """Test function test_calls_mlab_mesh."""
        vis = self._import_vis()
        vis.show_s2_sphere()
        vis.mlab.mesh.assert_called_once()

    def test_accepts_custom_color_and_opacity(self):
        """Test function test_accepts_custom_color_and_opacity."""
        vis = self._import_vis()
        vis.show_s2_sphere(opacity=0.5, color=(1, 0, 0))
        _, kwargs = vis.mlab.mesh.call_args
        self.assertEqual(kwargs["opacity"], 0.5)
        self.assertEqual(kwargs["color"], (1, 0, 0))


# ---------------------------------------------------------------------------
# show_r3_point_set
# ---------------------------------------------------------------------------


class TestShowR3PointSet(TestVisualizationsSetup):
    """Test function TestShowR3PointSet."""
    def _points(self):
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float).T  # (3,3)

    def test_calls_points3d(self):
        """Test function test_calls_points3d."""
        vis = self._import_vis()
        vis.show_r3_point_set(self._points())
        vis.mlab.points3d.assert_called_once()

    def test_show_sphere_calls_mesh(self):
        """Test function test_show_sphere_calls_mesh."""
        vis = self._import_vis()
        vis.show_r3_point_set(self._points(), show_sphere=True)
        vis.mlab.mesh.assert_called_once()  # from show_s2_sphere

    def test_no_sphere_no_mesh(self):
        """Test function test_no_sphere_no_mesh."""
        vis = self._import_vis()
        vis.show_r3_point_set(self._points(), show_sphere=False)
        vis.mlab.mesh.assert_not_called()

    def test_save_file_calls_savefig(self):
        """Test function test_save_file_calls_savefig."""
        vis = self._import_vis()
        vis.show_r3_point_set(self._points(), save_file="out.png")
        vis.mlab.savefig.assert_called_once_with("out.png")

    def test_no_save_file_no_savefig(self):
        """Test function test_no_save_file_no_savefig."""
        vis = self._import_vis()
        vis.show_r3_point_set(self._points())
        vis.mlab.savefig.assert_not_called()


# ---------------------------------------------------------------------------
# show_s2_region
# ---------------------------------------------------------------------------


class TestShowS2Region(TestVisualizationsSetup):
    """Test function TestShowS2Region."""
    def _region(self):
        # A simple non-polar region: theta in [0.2, 0.8], phi in [0.0, 1.0]
        return np.array([[0.2, 0.8], [0.0, 1.0]])

    def test_calls_plot3d(self):
        """Test function test_calls_plot3d."""
        vis = self._import_vis()
        vis.show_s2_region(self._region(), N=10)
        self.assertTrue(vis.mlab.plot3d.called)


# ---------------------------------------------------------------------------
# show_s2_partition
# ---------------------------------------------------------------------------


class TestShowS2Partition(TestVisualizationsSetup):
    """Test function TestShowS2Partition."""
    def test_calls_figure_mesh_show(self):
        """Test function test_calls_figure_mesh_show."""
        vis = self._import_vis()
        vis.show_s2_partition(4, show_points=True, show_sphere=True)
        vis.mlab.figure.assert_called_once()
        self.assertTrue(vis.mlab.mesh.called)
        self.assertTrue(vis.mlab.points3d.called)
        vis.mlab.show.assert_called_once()

    def test_show_false_does_not_call_mlab_show(self):
        """Test function test_show_false_does_not_call_mlab_show."""
        vis = self._import_vis()
        vis.show_s2_partition(4, show=False)
        vis.mlab.show.assert_not_called()

    def test_title_long_calls_mlab_text(self):
        """Test function test_title_long_calls_mlab_text."""
        vis = self._import_vis()
        vis.show_s2_partition(4, title="long", show=False)
        vis.mlab.text.assert_called_once()

    def test_title_short_calls_mlab_text(self):
        """Test function test_title_short_calls_mlab_text."""
        vis = self._import_vis()
        vis.show_s2_partition(4, title="short", show=False)
        vis.mlab.text.assert_called_once()

    def test_title_none_no_mlab_text(self):
        """Test function test_title_none_no_mlab_text."""
        vis = self._import_vis()
        vis.show_s2_partition(4, title="none", show=False)
        vis.mlab.text.assert_not_called()

    def test_save_file_calls_savefig(self):
        """Test function test_save_file_calls_savefig."""
        vis = self._import_vis()
        vis.show_s2_partition(4, show=False, save_file="snap.png")
        vis.mlab.savefig.assert_called_once_with("snap.png")

    def test_no_save_file_no_savefig(self):
        """Test function test_no_save_file_no_savefig."""
        vis = self._import_vis()
        vis.show_s2_partition(4, show=False)
        vis.mlab.savefig.assert_not_called()

    def test_no_sphere_skips_sphere_mesh(self):
        """Test function test_no_sphere_skips_sphere_mesh."""
        vis = self._import_vis()
        vis.show_s2_partition(4, show_sphere=False, show_points=False, show=False)
        # show_sphere=False: the sphere mesh should not be rendered.
        # Region boundaries use plot3d, not mesh, so mesh should not be called.
        vis.mlab.mesh.assert_not_called()

    def test_no_points_skips_points3d(self):
        """Test function test_no_points_skips_points3d."""
        vis = self._import_vis()
        vis.show_s2_partition(4, show_points=False, show_sphere=False, show=False)
        vis.mlab.points3d.assert_not_called()


# ---------------------------------------------------------------------------
# project_point_set
# ---------------------------------------------------------------------------


class TestProjectPointSet(TestVisualizationsSetup):
    """Test function TestProjectPointSet."""
    def test_s2_points_stereo_calls_points3d(self):
        """Test function test_s2_points_stereo_calls_points3d."""
        vis = self._import_vis()
        points = np.eye(3)  # shape (3,3), dim=2 (S^2 -> R^2)
        vis.project_point_set(points, proj="stereo", show=False)
        vis.mlab.points3d.assert_called_once()

    def test_s2_points_eqarea_calls_points3d(self):
        """Test function test_s2_points_eqarea_calls_points3d."""
        vis = self._import_vis()
        points = np.eye(3)
        vis.project_point_set(points, proj="eqarea", show=False)
        vis.mlab.points3d.assert_called_once()

    def test_s3_points_calls_points3d(self):
        """Test function test_s3_points_calls_points3d."""
        vis = self._import_vis()
        # shape (4, 4) → dim=3, S^3 -> R^3
        points = np.eye(4)
        vis.project_point_set(points, proj="stereo", show=False)
        vis.mlab.points3d.assert_called_once()

    def test_show_true_calls_mlab_show(self):
        """Test function test_show_true_calls_mlab_show."""
        vis = self._import_vis()
        vis.project_point_set(np.eye(3), proj="stereo", show=True)
        vis.mlab.show.assert_called_once()

    def test_show_false_no_mlab_show(self):
        """Test function test_show_false_no_mlab_show."""
        vis = self._import_vis()
        vis.project_point_set(np.eye(3), proj="stereo", show=False)
        vis.mlab.show.assert_not_called()

    def test_save_file_calls_savefig(self):
        """Test function test_save_file_calls_savefig."""
        vis = self._import_vis()
        vis.project_point_set(np.eye(3), proj="stereo", show=False, save_file="pts.png")
        vis.mlab.savefig.assert_called_once_with("pts.png")

    def test_invalid_dim_raises_value_error(self):
        """Test function test_invalid_dim_raises_value_error."""
        vis = self._import_vis()
        points = np.array([[1, 0], [0, 1]])  # shape (2,2) → dim=1
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
    def test_default_calls_figure_mesh_points3d_show(self):
        """Test function test_default_calls_figure_mesh_points3d_show."""
        vis = self._import_vis()
        vis.project_s3_partition(4, show_points=True, show_surfaces=True)
        vis.mlab.figure.assert_called_once()
        self.assertTrue(vis.mlab.mesh.called)
        self.assertTrue(vis.mlab.points3d.called)
        vis.mlab.show.assert_called_once()

    def test_show_false_no_mlab_show(self):
        """Test function test_show_false_no_mlab_show."""
        vis = self._import_vis()
        vis.project_s3_partition(4, show=False)
        vis.mlab.show.assert_not_called()

    def test_no_surfaces_no_mesh(self):
        """Test function test_no_surfaces_no_mesh."""
        vis = self._import_vis()
        vis.project_s3_partition(4, show_surfaces=False, show_points=False, show=False)
        vis.mlab.mesh.assert_not_called()

    def test_no_points_no_points3d(self):
        """Test function test_no_points_no_points3d."""
        vis = self._import_vis()
        vis.project_s3_partition(4, show_surfaces=False, show_points=False, show=False)
        vis.mlab.points3d.assert_not_called()

    def test_title_long_calls_mlab_text(self):
        """Test function test_title_long_calls_mlab_text."""
        vis = self._import_vis()
        vis.project_s3_partition(4, title="long", show=False)
        vis.mlab.text.assert_called_once()

    def test_title_none_no_mlab_text(self):
        """Test function test_title_none_no_mlab_text."""
        vis = self._import_vis()
        vis.project_s3_partition(4, title="none", show=False)
        vis.mlab.text.assert_not_called()

    def test_save_file_calls_savefig(self):
        """Test function test_save_file_calls_savefig."""
        vis = self._import_vis()
        vis.project_s3_partition(4, show=False, save_file="s3.png")
        vis.mlab.savefig.assert_called_once_with("s3.png")

    def test_invalid_proj_raises_value_error(self):
        """Test function test_invalid_proj_raises_value_error."""
        vis = self._import_vis()
        with self.assertRaises(ValueError):
            vis.project_s3_partition(4, proj="invalid")

    def test_eqarea_proj_runs(self):
        """Test function test_eqarea_proj_runs."""
        vis = self._import_vis()
        vis.project_s3_partition(4, proj="eqarea", show=False)
        vis.mlab.figure.assert_called_once()


if __name__ == "__main__":
    unittest.main()
