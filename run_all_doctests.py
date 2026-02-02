import doctest
import os
import sys

# Ensure current directory is in path
sys.path.insert(0, os.getcwd())

import eqsp.partitions
import eqsp.utilities
import eqsp.point_set_props
import eqsp.point_set_props
import eqsp.region_props
import eqsp.histograms

def run_tests():
    modules = [
        eqsp.utilities,
        eqsp.partitions,
        eqsp.point_set_props,
        eqsp.region_props,
        eqsp.histograms
    ]
    
    total_failures = 0
    total_tests = 0
    
    for mod in modules:
        print(f"Testing {mod.__name__}...")
        failures, tests = doctest.testmod(mod, verbose=True)
        total_failures += failures
        total_tests += tests
        print(f"Finished {mod.__name__}: {failures} failures in {tests} tests.\n")

    if total_failures == 0:
        print(f"All {total_tests} tests passed!")
        sys.exit(0)
    else:
        print(f"{total_failures} failures out of {total_tests} tests.")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
