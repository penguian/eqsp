"""Main script to run all performance benchmarks."""

import argparse
import time

# Import all benchmarks
import benchmark_area_error
import benchmark_energy_dist
import benchmark_eq_regions
import benchmark_histograms
import benchmark_sradius


def main():
    """Run the benchmark."""
    parser = argparse.ArgumentParser(description="EQSP Performance Benchmarks.")
    parser.add_argument(
        "--n-max",
        type=int,
        help="General n_max to override defaults for all benchmarks.",
    )
    parser.add_argument("--dim", type=int, default=2, help="Sphere dimension (default: 2).")
    parser.add_argument(
        "--s", type=float, help="Exponent for energy distance benchmark."
    )
    parser.add_argument(
        "--regions",
        type=int,
        default=1000,
        help="Partition size for histogram benchmark (default: 1000).",
    )

    args = parser.parse_args()

    # Define benchmarks and their specific run arguments if provided
    benchmarks = [
        (
            "eq_area_error (Redundant area calculation)",
            benchmark_area_error.run,
            {"n_max": args.n_max or 15000, "dim": args.dim},
        ),
        (
            "point_set_energy_dist (O(N^2) memory & broadcasting)",
            benchmark_energy_dist.run,
            {"n_max": args.n_max or 2400, "dim": args.dim, "s": args.s},
        ),
        (
            "eq_regions (Python loop overhead)",
            benchmark_eq_regions.run,
            {"n_max": args.n_max or 16000, "dim": args.dim},
        ),
        (
            "eq_find_s2_region (Vectorized histogram lookup)",
            benchmark_histograms.run,
            {"n_max": args.n_max or 80000000, "N": args.regions},
        ),
        (
            "sradius_of_cap (Root finding loop bottleneck)",
            benchmark_sradius.run,
            {"n_max": args.n_max or 40000000, "dim": args.dim + 1},  # Usually dim+1 in usage
        ),
    ]

    print("=======================================")
    print("      EQSP Performance Benchmarks      ")
    print("=======================================\n")

    t_start = time.perf_counter()

    for name, run_func, kwargs in benchmarks:
        print(f"Running benchmark: {name}")
        # Only pass arguments that aren't None
        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        run_func(**filtered_kwargs)
        print()

    t_end = time.perf_counter()
    print("=======================================")
    print(f"Total benchmark time: {t_end - t_start:.2f} seconds")
    print("=======================================")


if __name__ == "__main__":
    main()
