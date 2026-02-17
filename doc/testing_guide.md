# Testing Guide: EQSP

This guide outlines the testing strategy, dependencies, and instructions for verifying the `eqsp` package.

## 1. Testing Strategy

The `eqsp` package employs a multi-layered testing strategy:

1.  **Unit Tests (`tests/`)**:
    *   Verify core logic of partitions, properties, and utilities.
    *   Compare results against known-good values (often from the original Matlab implementation).
    *   Verify edge cases and invalid inputs.
2.  **Doctests**:
    *   Ensure examples in documentation strings are correct and runnable.
    *   Covered by standard `pytest` runs.
3.  **Mock Tests (`tests/test_illustrations_mock.py`)**:
    *   Verify visualization logic without requiring a GUI or heavy dependencies (Matplotlib/Mayavi).
    *   Simulate library calls to ensure data is passed correctly to plotting functions.
4.  **Verification Scripts (`tests/verify/`)**:
    *   Integration tests that exercise full workflows (plotting, illustration).
    *   Example: `tests/verify/verify_algorithm_illustration.py`.

## 2. Dependencies

To run the full test suite, you need the following development dependencies:

*   **`pytest`**: The test runner.
*   **`coverage`**: For code coverage analysis.
*   **`mock`** (standard `unittest.mock` in Python 3): For mocking graphics libraries.

Install them via pip:
```bash
pip install pytest coverage
```

## 3. Running Tests

### 3.1 Basic Test Run
To run all unit tests and doctests (fast):
```bash
pytest
```

### 3.2 Running with Coverage
To measure code coverage:

**Option A: Using the Helper Script**
We provide a script that sets up the environment and runs the full suite:
```bash
./run_coverage.sh
```

**Option B: Manual Command**
```bash
# 1. Run tests with coverage
python3 -m coverage run --source=eqsp -m pytest

# 2. View report
python3 -m coverage report
```

### 3.3 Running Specific Tests
To run only the histogram tests, for example:
```bash
pytest tests/test_histograms.py
```

To run only the mock illustration tests:
```bash
pytest tests/test_illustrations_mock.py
```

### 3.4 Verification Scripts
The integration scripts are located in `tests/verify/`. They can be run directly:
```bash
python3 tests/verify/verify_algorithm_illustration.py
```
Most are configured to run headlessly (non-interactive backend), making them suitable for automated checks.
