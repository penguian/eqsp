# Testing Guide: EQSP

This guide outlines the testing strategy, dependencies, and instructions for verifying the `eqsp` package.

## 1. Testing Strategy

The `eqsp` package uses a **Hybrid Testing Approach** that integrates standard unit tests with `doctest` examples to ensure documentation and code remain in sync.

### 1.1 Test Categories

1.  **Unit Tests (`tests/src/test_*.py`)**:
    *   Verify core mathematical logic using static assertions.
    *   Compare results against known-good values from the original Matlab implementation.
2.  **Mock Tests (`tests/src/*_mock.py`)**:
    *   Verify library interaction (e.g., Mayavi/Matplotlib) without needing a display.
    *   Check if the correct arguments are passed to the plotting engines.
3.  **Extra Tests (`tests/src/*_extra.py`)**:
    *   Introspective integration tests.
    *   Use non-interactive backends (like Matplotlib's `Agg`) to verify that the rendered plot objects contain the expected mathematical labels and data properties.
4.  **Doctests**:
    *   Live-tested examples inside module docstrings.
    *   Ensures that every `>>>` example in the documentation is verified during test runs.

## 2. Dependencies

Install development dependencies via pip:

```bash
pip install pytest coverage
```

## 3. Running Tests

### 3.1 Project-Wide Run
To run the entire suite (recommended):
```bash
pytest
```

### 3.2 Granular Control
You can run tests at three levels of granularity:

1.  **By Module (Bridge)**:
    Runs both manual unit tests and bridged doctests for that module.
    ```bash
    pytest tests/src/test_point_set_props.py
    ```
2.  **By File (Direct)**:
    Runs **only** the doctests localized in that source file.
    ```bash
    python3 -m eqsp.point_set_props -v
    ```
3.  **Interactive Inspection**:
    To visually verify 2D or 3D output on your local machine:
    ```bash
    python3 tests/src/inspect_illustrations.py
    python3 tests/src/inspect_visualizations.py
    ```

### 3.3 Visual Verification (Thesis Examples)

The `examples/phd-thesis/` directory contains high-fidelity scripts that reproduce figures from the canonical PhD thesis. Use these to verify that the library's results match the originally published data:

```bash
cd examples/phd-thesis
# Run a numerical plot (Agg backend, saves PNG)
python3 fig_4_2_min_dist_s2.py --upper-bound 5000

# Run a 3D visualization (Mayavi, requires venv_sys)
python3 fig_3_1_partition_s2_33.py
```

> **Note:** The `venv_sys` configuration used for automated and manual testing was specific to **Kubuntu Linux 25.10**. Different Linux distributions may require adjustments to environment variables.

See [PhD Thesis Example Reproductions](phd-thesis-examples.md) for a full mapping of scripts to thesis figures.

## 4. Code Coverage

To generate a full coverage report, use the provided helper script:

```bash
python3 tests/run_coverage.py
```

This will run all tests (including doctests) and produce a summary in the terminal. The project maintains a strict benchmark of **100% coverage**. A comprehensive pragma audit was conducted in 0.99 Beta to ensure that all testable code paths are exercised, and that existing pragmas are only applied to truly unreachable or environment-dependent code.

Detailed results are saved in `tests/results/run_coverage.log`.

### 4.1 Private Implementation Tests

By default, the coverage script strictly excludes **private implementation tests** and **internal doctests** to maintain a clear boundary between the Public API and internal performance optimizations.

To include these high-fidelity tests in the coverage report, use the `--include-private` flag (benchmark: **100% coverage**):

```bash
python3 tests/run_coverage.py --include-private
```

Detailed results are saved in `tests/results/run_coverage_include_private.log`.

The private testing suite includes:
- **`tests/src/test_private_histograms.py`**: Verifies vectorized region lookup logic on S².
- **`tests/src/test_private_partitions.py`**: Bridge test for `eqsp._private._partitions` doctests.
- **`tests/src/test_private_region_props.py`**: Bridge test for `eqsp._private._region_props` doctests.

These tests ensure that internal math optimizations (such as vectorized colatitude lookups) match the reference Matlab logic with high precision.

## 5. Performance Benchmarking

The `benchmarks/` directory contains scripts to verify the algorithmic complexity and execution speed of core functions.

### 5.1 Running the Suite
To run all system benchmarks and generate a summary report:
```bash
python3 benchmarks/run_benchmarks.py

# Benchmark symmetric partitions
python3 benchmarks/run_benchmarks.py --even-collars
```

### 5.2 Results and Logging
The runner saves individual results for each benchmark in a standardized format:
- **Main Summary**: `benchmarks/results/run_benchmarks.log`
- **Individual Logs**: `benchmarks/results/benchmark_*.log` (e.g., `benchmark_eq_regions.log`)

### 5.3 Thesis Benchmark (Section 3.10.2)
The script `benchmarks/src/benchmark_eq_regions.py` specifically replicates the "Running time" benchmark from Section 3.10.2 of the thesis. It verifies the **$O(N^{0.6})$** scaling behavior.

To run it independently with progress tracking:
```bash
python3 benchmarks/src/benchmark_eq_regions.py --show-progress
```

## 6. Code Quality

The project uses `ruff` and `pylint` to maintain high code quality standards.

### 6.1 Ruff (Style and Formatting)
Ruff handles fast linting and automatic formatting:
```bash
ruff check .
ruff format .
```

### 6.2 Pylint (Deep Static Analysis)
Pylint is used for deep semantic analysis. The configuration is refined to allow standard mathematical notation (including variable names like `N_values`, `Ns`, `Phi`) while enforcing strict code quality across the entire repository. The project baseline is a **10.00/10** rating:
```bash
pylint eqsp benchmarks examples tests  # Project-wide scan
```
