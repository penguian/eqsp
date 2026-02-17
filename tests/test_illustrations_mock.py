
import sys
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

# Ensure eqsp is importable
from eqsp import partitions

class TestIllustrations(unittest.TestCase):
    def setUp(self):
        # Patch matplotlib.pyplot in eqsp.illustrations
        self.plt_patcher = patch('eqsp.illustrations.plt')
        self.mock_plt = self.plt_patcher.start()
        
        # Also patch Axes3D to avoid issues
        self.axes3d_patcher = patch('eqsp.illustrations.Axes3D')
        self.mock_axes3d = self.axes3d_patcher.start()
        
        # Configure mock_plt
        self.mock_fig = MagicMock()
        self.mock_ax = MagicMock()
        self.mock_plt.figure.return_value = self.mock_fig
        self.mock_fig.add_subplot.return_value = self.mock_ax
        self.mock_plt.gca.return_value = self.mock_ax
        self.mock_plt.get_backend.return_value = 'Agg' # Non-interactive

    def tearDown(self):
        self.plt_patcher.stop()
        self.axes3d_patcher.stop()

    def test_show_s2_partition(self):
        from eqsp import illustrations
        illustrations.show_s2_partition(4, show_points=True, show_sphere=True)
        
        # Verify calls
        self.mock_plt.figure.assert_called()
        self.mock_fig.add_subplot.assert_called()
        # Verify plotting calls on ax
        self.assertTrue(self.mock_ax.plot_surface.called)
        self.assertTrue(self.mock_ax.scatter.called)

    def test_project_s2_partition(self):
        from eqsp import illustrations
        illustrations.project_s2_partition(4, proj='stereo', show_points=True)
        self.assertTrue(self.mock_ax.plot.called)
        self.assertTrue(self.mock_ax.scatter.called)

    def test_project_s3_partition(self):
        from eqsp import illustrations
        illustrations.project_s3_partition(4, proj='stereo', show_points=True, show_surfaces=True)
        self.assertTrue(self.mock_ax.plot_surface.called)

    def test_illustrate_eq_algorithm(self):
        from eqsp import illustrations
        illustrations.illustrate_eq_algorithm(2, 4)
        # Verify subplots were created
        self.assertTrue(self.mock_plt.subplot.called)
        self.assertTrue(self.mock_plt.plot.called)


class TestIllustrationsMayavi(unittest.TestCase):
    def setUp(self):
        # We need to mock mayavi BEFORE importing eqsp.illustrations_mayavi
        # if it's not already imported.
        
        self.modules_patcher = patch.dict(sys.modules, {'mayavi': MagicMock(), 'mayavi.mlab': MagicMock()})
        self.modules_patcher.start()
        
        # Now we can safely import it, even if mayavi is missing
        if 'eqsp.illustrations_mayavi' in sys.modules:
            del sys.modules['eqsp.illustrations_mayavi']
            
    def tearDown(self):
        self.modules_patcher.stop()

    def test_show_s2_partition(self):
        import eqsp.illustrations_mayavi as im
        # Mock mlab
        im.mlab = MagicMock()
        
        im.show_s2_partition(4, show_points=True, show_sphere=True)
        
        self.assertTrue(im.mlab.figure.called)
        self.assertTrue(im.mlab.mesh.called) # sphere and regions
        self.assertTrue(im.mlab.points3d.called) # points
        self.assertTrue(im.mlab.show.called)

    def test_project_point_set(self):
        import eqsp.illustrations_mayavi as im
        im.mlab = MagicMock()
        
        points = np.array([[1,0,0], [0,1,0]]).T
        im.project_point_set(points, proj='stereo')
        
        self.assertTrue(im.mlab.points3d.called)

    def test_project_s3_partition(self):
        import eqsp.illustrations_mayavi as im
        im.mlab = MagicMock()
        
        im.project_s3_partition(4, show_points=True, show_surfaces=True)
        
        self.assertTrue(im.mlab.mesh.called) # surfaces
        self.assertTrue(im.mlab.points3d.called) # items

