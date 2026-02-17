from unittest.mock import patch
import sys
import numpy as np
import matplotlib.pyplot as plt

# Mock mlab.show to avoid blocking
with patch("mayavi.mlab.show") as mock_show:
    from eqsp.illustrations_mayavi import project_s3_partition
    from mayavi import mlab

    print("Testing project_s3_partition(10, proj='stereo')...")
    try:
        mlab.options.offscreen = True
        # Small N to run faster
        project_s3_partition(4, show_title=True)
        print("Success: project_s3_partition ran without error.")
    except Exception as e:
        print(f"Failed: project_s3_partition raised exception: {e}")
        sys.exit(1)
