# Performance Benchmarks

The `eqsp` package includes a suite of performance benchmarks to measure the efficiency of core mathematical operations and identify potential bottlenecks.

## Running Benchmarks

### All-in-One Runner
To run the entire benchmark suite with default settings:

```bash
python3 benchmarks/run_benchmarks.py
```

### Configurable Runs
The runner supports many flags to configure the benchmark run:

```bash
# Quick sanity check with small problem size
python3 benchmarks/run_benchmarks.py --n-max 500

# High-dimensional benchmark
python3 benchmarks/run_benchmarks.py --dim 4

# Specific partition size for histogram lookups
python3 benchmarks/run_benchmarks.py --regions 100000

# Benchmark symmetric partitions (forces even number of collars)
python3 benchmarks/run_benchmarks.py --even-collars
```

| Flag | Description | Default |
| :--- | :--- | :--- |
| `--results-dir` | Directory to save log files. | `benchmarks/results` |
| `--n-max` | Scales the problem size for all benchmarks. | Varied (see below) |
| `--dim` | Sphere dimension for partitioning and mathematics. | 2 |
| `--regions` | Number of regions in the partition for histogram lookups. | 1000 |
| `--s` | Exponent for the Riesz energy calculation. | `dim - 1` |
| `--even-collars` | Force an even number of collars (symmetric partitions). | `False` |

### Default Scales (`--n-max` overrides)
- **`eq_area_error`**: `n-max=15000`
- **`point_set_energy_dist`**: `n-max=2400`
- **`sradius_of_cap`**: `n-max=100,000,000`
- **`eq_regions`**: `n-max=16000`
- **`eq_min_dist`**: `n-max=6400`
- **`eq_find_s2_region`**: `n-max=200,000,000`

## Benchmark Categories

1.  **`eq_area_error`**: Measures the time to calculate area errors for a range of partition sizes. This captures $O(N)$ recurrence overhead.
2.  **`point_set_energy_dist`**: Measures energy and distance calculations. This captures the optimized block-based summation performance.
3.  **`sradius_of_cap`**: Benchmarks the root-finding logic used for spherical cap calculations.
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
- **Standard Runs**: Logs are saved as `benchmark_[name].log`.
- **Symmetric Runs (`--even-collars`)**: Logs are saved with an `_even` suffix (e.g., `benchmark_eq_regions_even.log`).

This allows for side-by-side performance comparison of the different partitioning strategies.
- **$O(N^2)$ scaling**: Common in distance matrix calculations.
- **Python Loop Overhead**: Evident in the recursive partitioning logic.
- **Cache Misses**: May occur when $N$ exceeds specific size thresholds.

For a detailed analysis of known bottlenecks and optimization opportunities, see the performance highlights in [Algorithmic Implementation & Optimizations](internal/algorithmic_optimizations.md).

## Future Optimization Opportunities

### Persistent Caching
The current implementation of `eq_regions` use a **local, ephemeral cache** to exploit hemispherical symmetry within a single function call. 

An opportunity exists to add a **persistent caching layer** (e.g., using `functools.lru_cache` or a disk-backed cache) for common partition sizes ($N$). This would provide significant speedups for analyses that repeatedly iterate through ranges of $N$, common in property convergence studies.

### Energy Calculation Optimization
The calculation of the full Riesz energy sum ($s \neq 0$) is inherently $O(N^2)$, as it requires visiting every pair of points.

To address the $O(N^2)$ memory bottleneck, `eqsp` uses a **block-based processing (tiling) approach** that limits peak memory usage to $O(N \times \text{block\_size})$. Additionally, the implementation **exploits symmetry** ($d_{ij} = d_{ji}$) to reduce the computational work by half.

### Accelerated Loops
The recursive structure of the partitioning algorithm and the coordinate search logic are prime candidates for acceleration using **Numba** or **Cython** to remove Python interpreter overhead in hot loops.
