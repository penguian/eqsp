# Testing Guide: EQSP

This guide outlines the testing strategy, dependencies, and instructions for verifying the `eqsp` package.

## 1. Testing Strategy

The `eqsp` package uses a **Hybrid Testing Approach** that integrates standard unit tests with `doctest` examples to ensure documentation and code remain in sync.

### 1.1 Test Categories

1.  **Unit Tests (`tests/test_*.py`)**:
    *   Verify core mathematical logic using static assertions.
    *   Compare results against known-good values from the original Matlab implementation.
2.  **Mock Tests (`tests/*_mock.py`)**:
    *   Verify library interaction (e.g., Mayavi/Matplotlib) without needing a display.
    *   Check if the correct arguments are passed to the plotting engines.
3.  **Extra Tests (`tests/*_extra.py`)**:
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
    pytest tests/test_point_set_props.py
    ```
2.  **By File (Direct)**:
    Runs **only** the doctests localized in that source file.
    ```bash
    python3 -m eqsp.point_set_props -v
    ```
3.  **Interactive Inspection**:
    To visually verify 2D or 3D output on your local machine:
    ```bash
    python3 tests/inspect_illustrations.py
    python3 tests/inspect_visualizations.py
    ```

## 4. Code Coverage

To generate a full coverage report, use the provided helper script:

```bash
./run_coverage.sh
```

This will run all tests (including doctests) and produce a summary in the terminal. The project maintains a benchmark of **94% coverage**.
