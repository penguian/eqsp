# Performance Benchmarks

The `eqsp` package includes a suite of performance benchmarks to measure the efficiency of core mathematical operations and identify potential bottlenecks.

## Running Benchmarks

### All-in-One Runner
To run the entire benchmark suite with default settings:

```bash
python3 benchmarks/run_benchmarks.py
```

### Symmetric (Even-Collar) Benchmarks
To run the benchmark suite specifically for symmetric partitions (forcing an even number of collars):

```bash
python3 benchmarks/run_benchmarks.py --even-collars
```

### Logarithmic Sampling & Scaling
The Python benchmark suite now matches the MATLAB EQSP Toolbox by using **1-2-5 logarithmic sampling** (e.g., 10, 20, 50, 100, 200, 500...). This allows for more precise verification of asymptotic behavior across many orders of magnitude.

Additionally, each script now performs a **Scaling Analysis** by calculating the best-fitting power $x$ in $O(N^x)$ using a log-log regression. This automatically verifies that the implementation follows its theoretical complexity (e.g., $O(N \log N)$ for spatial lookups).

### Configurable Runs

The top-level runner (`run_benchmarks.py`) supports a subset of common configuration flags:

| Flag | Description | Default |
| :--- | :--- | :--- |
| `--results-dir` | Directory to save log files. | `benchmarks/results` |
| `--n-max` | Scales the problem size for all benchmarks. | Varied (see below) |
| `--regions` | Number of regions in the partition for histogram lookups. | 1000 |
| `--s` | Exponent for the Riesz energy calculation. | `dim - 1` |

**Note:** To run benchmarks for specific dimensions or other advanced parameters, use the individual scripts in `benchmarks/src/` directly.

```bash
# Example: High-dimensional benchmark for recursive partitioning
python3 benchmarks/src/benchmark_eq_regions.py --dim 4 --n-max 1000000

# Example: Symmetric min-distance benchmark on S^2
python3 benchmarks/src/benchmark_mindist.py --dim 2 --even-collars
```

### Default Scales and Dimensions (`--n-max` overrides)
The benchmark suite uses different default dimensions and problem sizes depending on the mathematical property being tested:

- **`eq_area_error`**: `n-max=100,000,000`, **dim=2** ($S^2$)
- **`point_set_energy_dist`**: `n-max=50,000`, **dim=2** ($S^2$)
- **`sradius_of_cap`**: `n-max=10,000,000`, **dim=3** ($S^3$) *(standard runner only)*
- **`eq_regions`**: `n-max=100,000,000`, **dim=2** ($S^2$)
- **`eq_min_dist`**: `n-max=10,000,000`, **dim=2** ($S^2$)
- **`eq_find_s2_region`**: `n-max=10,000,000`, **dim=2** ($S^2$)

## Benchmark Categories

1.  **`eq_area_error`**: Measures the time to calculate area errors for a range of partition sizes. This captures $O(N)$ recurrence overhead.
2.  **`point_set_energy_dist`**: Measures energy and distance calculations. This captures the optimized block-based summation performance.
3.  **`sradius_of_cap`** *(standard runner only)*: Benchmarks the root-finding logic used for spherical cap calculations. Defaults to **$S^3$** to exercise higher-dimensional root finding.
4.  **`eq_regions`**: Measures the overhead of the Python loop used in recursive partitioning.
5.  **`eq_min_dist`**: Measures the performance of the structure-aware min-distance calculation.
6.  **`eq_find_s2_region`**: Measures the performance of the vectorized histogram-based region lookup on $S^2$.

## Thesis Scaling Benchmark (Section 3.10.2)

For formal verification of the partitioning algorithm's scaling, use the dedicated thesis benchmark:

```bash
# Run the formal d=[1..8], N=2^k sweep
python3 benchmarks/src/benchmark_eq_regions.py --show-progress

# Compare with even collars
python3 benchmarks/src/benchmark_eq_regions.py --show-progress --even-collars
```

This script generates high-fidelity timing data to verify the $O(\mathcal{N}^{0.6})$ scaling theory described in Chapter 3 of the thesis. The symmetric partition method (`even_collars=True`) generally follows the same scaling but may be slightly faster for certain $N$ due to the forced even number of collars simplifying the recurrence.

## Interpreting Results

Benchmarks are printed to the console and also saved as individual log files in `benchmarks/results/`.

### Log Naming Convention
- **Standard Runs** (`run_benchmarks.py` without flags): Individual logs are saved as `benchmark_[name].log`; the summary log is `run_benchmarks.log`.
- **Symmetric Runs** (`run_benchmarks.py --even-collars`): Individual logs are saved with an `_even` suffix (e.g., `benchmark_eq_regions_even.log`); the summary log is `run_benchmarks_even.log`.

### Warm-up Phases
To minimize variability from JIT/caching effects, each benchmark task includes an initial un-timed **warm-up call**. This ensures that subsequent measurements represent steady-state performance.

This allows for side-by-side performance comparison of the different partitioning strategies.
- **$O(N^2)$ scaling**: Common in distance matrix calculations.
- **Python Loop Overhead**: Evident in the recursive partitioning logic.
- **Cache Misses**: May occur when $N$ exceeds specific size thresholds.

For a detailed analysis of known bottlenecks and optimization opportunities, see the performance highlights in [Algorithmic Implementation & Optimizations](algorithmic_optimizations.md).

:::{note} Beta Feedback Wanted
We are specifically looking for performance data and algorithm scaling results across different Operating Systems and CPU architectures (e.g., Apple Silicon, Windows 11, different Linux distros). If you run these benchmarks, please consider sharing your results in our [GitHub Discussions](https://github.com/penguian/pyeqsp/discussions/categories/beta-testing).
:::

## Future Optimization Opportunities

### Persistent Caching
The current implementation of `eq_regions` use a **local, ephemeral cache** to exploit hemispherical symmetry within a single function call.

An opportunity exists to add a **persistent caching layer** (e.g., using `functools.lru_cache` or a disk-backed cache) for common partition sizes ($N$). This would provide significant speedups for analyses that repeatedly iterate through ranges of $N$, common in property convergence studies.

### Energy Calculation Optimization
The calculation of the full Riesz energy sum ($s \neq 0$) is inherently $O(N^2)$, as it requires visiting every pair of points.

To address the $O(N^2)$ memory bottleneck, `eqsp` uses a **block-based processing (tiling) approach** that limits peak memory usage to $O(N \times \text{block\_size})$. Additionally, the implementation **exploits symmetry** ($d_{ij} = d_{ji}$) to reduce the computational work by half.

### Accelerated Loops
The recursive structure of the partitioning algorithm and the coordinate search logic are prime candidates for acceleration using **Numba** or **Cython** to remove Python interpreter overhead in hot loops.

---

For a comprehensive view of proposed enhancements currently under review, see the [Release Roadmap](release_roadmap.md).
