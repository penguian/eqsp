"""
Unified verification script.
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def run_step(command, name):
    """Run a single verification step and exit on failure."""
    print("========================================")
    print(f"Running {name}...")
    print("========================================")
    result = subprocess.run(command, check=False, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f"\n[FAILED] {name}\n")
        sys.exit(result.returncode)
    print(f"[PASSED] {name}\n")


def main():
    """Execute all verification steps."""
    py = sys.executable
    steps = [
        (
            [
                py,
                "-m",
                "ruff",
                "check",
                "eqsp",
                "tests",
                "examples/phd-thesis",
                "benchmarks",
                "verify_all.py",
            ],
            "Ruff Linter",
        ),
        (
            [
                py,
                "-m",
                "pylint",
                "eqsp",
                "tests",
                "examples/phd-thesis",
                "benchmarks",
                "verify_all.py",
            ],
            "Pylint",
        ),
        ([py, "tests/run_coverage.py", "--include-private"], "Test Suite & Coverage"),
    ]

    for cmd, name in steps:
        run_step(cmd, name)

    print("All verification steps passed successfully!")

if __name__ == "__main__":
    main()
