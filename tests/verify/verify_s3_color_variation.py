import matplotlib.pyplot as plt
import numpy as np
import sys
from eqsp.illustrations import project_s3_partition

print("Testing project_s3_partition(10, proj='stereo')...")
try:
    # Use non-interactive backend
    plt.switch_backend("Agg")
    ax = project_s3_partition(10, proj="stereo")
    print("Success: project_s3_partition ran without error.")
except Exception as e:
    print(f"Failed: project_s3_partition raised exception: {e}")
    sys.exit(1)
