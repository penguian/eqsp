from unittest.mock import patch
import sys

# Mock mlab.show to avoid blocking
with patch("mayavi.mlab.show") as mock_show:
    from eqsp.illustrations_mayavi import show_s2_partition
    from mayavi import mlab

    print("Testing show_s2_partition(10, show_title=True)...")
    try:
        # Force offscreen to avoid X11 issues if possible
        mlab.options.offscreen = True

        show_s2_partition(10)
        print("Success: show_s2_partition ran without error.")
    except Exception as e:
        print(f"Failed: show_s2_partition raised exception: {e}")
        sys.exit(1)
