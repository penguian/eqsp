import argparse
import time

import numpy as np

from eqsp.utilities import area_of_sphere, sradius_of_cap


def run(n_max=100000, dim=3):
    """Run the benchmark.

    Args:
        n_max (int): Total number of areas to evaluate up to.
        dim (int): Sphere dimension.
    """
    print(f"{'Size':<10} | {'Time (s)':>10}")
    print("-" * 23)

    # Generate sizes dynamically
    sizes = []
    chunk = max(1, n_max // 5)
    for i in range(chunk, n_max + 1, chunk):
        sizes.append(i)

    max_area = area_of_sphere(dim)

    for size in sizes:
        areas = np.linspace(0.1, max_area - 0.1, size)
        t0 = time.perf_counter()
        sradius_of_cap(dim, areas)
        t1 = time.perf_counter()
        print(f"{size:<10} | {t1 - t0:>10.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark for sradius_of_cap.")
    parser.add_argument(
        "--n-max",
        type=int,
        default=40000000,
        help="Total number of areas to evaluate up to (default: 40,000,000).",
    )
    parser.add_argument(
        "--dim", type=int, default=3, help="Sphere dimension (default: 3)."
    )
    args = parser.parse_args()
    run(n_max=args.n_max, dim=args.dim)
