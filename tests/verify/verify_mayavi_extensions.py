from unittest.mock import patch
import sys
import numpy as np

# Mock mlab.show to avoid blocking
with patch("mayavi.mlab.show") as mock_show:
    from eqsp.illustrations_mayavi import project_point_set, project_s3_partition
    from mayavi import mlab

    print("Testing project_point_set (dim 3 -> R^3)...")
    try:
        # 3D points (S^3 -> R^4, but projected to R^3 by x2stereo/eqarea)
        # Wait, points input to project_point_set are Euclidean points in R^{dim+1}.
        # For S^3, points are in R^4.
        points_s3 = np.random.randn(4, 10)
        points_s3 = points_s3 / np.linalg.norm(points_s3, axis=0)  # Normalize to sphere

        mlab.options.offscreen = True
        project_point_set(points_s3, proj="stereo")
        print("Success: project_point_set(S^3) ran without error.")
    except Exception as e:
        print(f"Failed: project_point_set(S^3) raised exception: {e}")
        # sys.exit(1) # Don't exit yet, try next test

    print("Testing project_s3_partition(10)...")
    try:
        mlab.options.offscreen = True
        # Small N to run faster
        project_s3_partition(4, show_title=True)
        print("Success: project_s3_partition ran without error.")
    except Exception as e:
        print(f"Failed: project_s3_partition raised exception: {e}")
        sys.exit(1)
