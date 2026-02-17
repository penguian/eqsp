import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from eqsp.illustrations import illustrate_eq_algorithm
import numpy as np

print("Running algorithm illustration verification...")

if __name__ == "__main__":
    try:
        print("Test 1: Full Algorithm Illustration (Default)")
        # This function creates subplots. We just need to ensure it runs.
        plt.figure(figsize=(10, 10))
        illustrate_eq_algorithm(3, 10)
        plt.close("all")
        print("PASS")

        print("Test 2: Full Algorithm with Extra Offset & EqArea")
        plt.figure(figsize=(10, 10))
        illustrate_eq_algorithm(3, 10, extra_offset=True, proj="eqarea")
        plt.close("all")
        print("PASS")

    except Exception as e:
        print(f"FAIL: {e}")
        import traceback

        traceback.print_exc()
        exit(1)

print("All algorithm illustration tests passed.")
