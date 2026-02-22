import argparse
import time

import numpy as np

from eqsp.point_set_props import eq_energy_dist


def run(n_max=3000, dim=2, s=None):
    """Run the benchmark.

    Args:
        n_max (int): Total number of regions to evaluate up to.
        dim (int): Sphere dimension.
        s (float): Exponent parameter.
    """
    print(f"{'N range':<15} | {'Time (s)':>10}")
    print("-" * 28)

    # Split n_max into 5 intervals
    chunk_size = max(1, n_max // 5)
    ranges = []
    for i in range(0, n_max, chunk_size):
        ranges.append((i + 1, min(i + chunk_size, n_max)))

    for start, end in ranges:
        if start > end:
            continue
        N_array = np.arange(start, end + 1)
        t0 = time.perf_counter()
        eq_energy_dist(dim, N_array, s=s)
        t1 = time.perf_counter()
        print(f"{f'{start}-{end}':<15} | {t1 - t0:>10.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark for energy and distance.")
    parser.add_argument(
        "--n-max",
        type=int,
        default=2400,
        help="Total number of points to evaluate up to (default: 2400).",
    )
    parser.add_argument(
        "--dim", type=int, default=2, help="Sphere dimension (default: 2)."
    )
    parser.add_argument(
        "--s", type=float, default=None, help="Exponent parameter (default: dim-1)."
    )
    args = parser.parse_args()
    run(n_max=args.n_max, dim=args.dim, s=args.s)
