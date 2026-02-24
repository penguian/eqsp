# Performance Benchmarks

The `eqsp` package includes a suite of performance benchmarks to measure the efficiency of core mathematical operations and identify potential bottlenecks.

## 1. Running Benchmarks

### 1.1 All-in-One Runner
To run the entire benchmark suite with default settings:

```bash
python3 benchmarks/src/run_benchmarks.py
```

### 1.2 Configurable Runs
The runner supports several flags to control the problem size and duration:

```bash
# Quick sanity check with small problem size
python3 benchmarks/src/run_benchmarks.py --n-max 500

# High-dimensional benchmark
python3 benchmarks/src/run_benchmarks.py --dim 4

# Specific partition size for histogram lookups
python3 benchmarks/src/run_benchmarks.py --regions 100000
```

| Flag | Description | Default |
| :--- | :--- | :--- |
| `--n-max` | Scales the problem size for all benchmarks. | Varied (see below) |
| `--dim` | Sphere dimension for partitioning and math. | 2 |
| `--regions` | Number of regions in the partition for histogram lookups. | 1000 |
| `--s` | Exponent for the Riesz energy calculation. | `dim - 1` |

### Default Scales (`--n-max` overrides)
- **`eq_area_error`**: `n-max=15000`
- **`point_set_energy_dist`**: `n-max=2400`
- **`sradius_of_cap`**: `n-max=100,000,000`
- **`eq_regions`**: `n-max=16000`
- **`eq_min_dist`**: `n-max=6400`
- **`eq_find_s2_region`**: `n-max=200,000,000`

## 2. Benchmark Categories

1.  **`eq_area_error`**: Measures the time to calculate area errors for a range of partition sizes. This captures $O(N)$ recurrence overhead.
2.  **`point_set_energy_dist`**: Measures energy and distance calculations. This captures the $O(N^2)$ memory and distance matrix bottleneck.
3.  **`sradius_of_cap`**: Benchmarks the root-finding logic used for spherical cap calculations.
4.  **`eq_regions`**: Measures the overhead of the Python loop used in recursive partitioning.
5.  **`eq_min_dist`**: Measures the performance of the structure-aware minimum distance calculation.
6.  **`eq_find_s2_region`**: Measures the performance of the vectorized histogram-based region lookup on $S^2$.

## 3. Interpreting Results

Benchmarks are printed as a table of ranges vs. execution time. Significant jumps in time per range may indicate:
- **$O(N^2)$ scaling**: Common in distance matrix calculations.
- **Python Loop Overhead**: Evident in the recursive partitioning logic.
- **Cache Misses**: May occur when $N$ exceeds specific size thresholds.

For a detailed analysis of known bottlenecks and optimization opportunities, see the `efficiency_report.md` in the `workspace-artifacts` directory.

## 4. Future Optimization Opportunities

### 4.1 Persistent Caching
The current implementation of `eq_regions` use a **local, ephemeral cache** to exploit hemispherical symmetry within a single function call. 

An opportunity exists to implement a **persistent caching layer** (e.g., using `functools.lru_cache` or a disk-backed cache) for common partition sizes ($N$). This would provide significant speedups for analyses that repeatedly iterate through ranges of $N$, common in property convergence studies.

### 4.2 Energy Calculation Optimization
The $O(N^2)$ energy calculations in `point_set_energy_dist` currently use SciPy's `cdist`. While minimum distance calculations have been optimized using KDTrees and structure-aware localized searches, the full Riesz energy sum still requires visiting all pairs.

For points on a unit sphere, the squared Euclidean distance can be calculated more efficiently using matrix multiplication ($||x-y||^2 = 2 - 2x \cdot y$), which would reduce both compute time and peak memory usage for large $N$.

### 4.3 Accelerated Loops
The recursive structure of the partitioning algorithm and the coordinate search logic are prime candidates for acceleration using **Numba** or **Cython** to eliminate Python interpreter overhead in hot loops.
