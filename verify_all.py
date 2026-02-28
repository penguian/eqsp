"""
Unified verification script.
"""
import subprocess
import sys


def run_step(command, name):
    """Run a single verification step and exit on failure."""
    print("========================================")
    print(f"Running {name}...")
    print("========================================")
    result = subprocess.run(command, shell=True, check=False)
    if result.returncode != 0:
        print(f"\n[FAILED] {name}\n")
        sys.exit(result.returncode)
    print(f"[PASSED] {name}\n")

def main():
    """Execute all verification steps."""
    steps = [
        (
            "python3 -m ruff check eqsp tests examples/phd-thesis "
            "benchmarks verify_all.py",
            "Ruff Linter",
        ),
        (
            "python3 -m pylint eqsp tests examples/phd-thesis "
            "benchmarks verify_all.py",
            "Pylint",
        ),
        ("python3 tests/run_coverage.py --include-private", "Test Suite & Coverage"),
    ]

    for cmd, name in steps:
        run_step(cmd, name)

    print("All verification steps passed successfully!")

if __name__ == "__main__":
    main()
