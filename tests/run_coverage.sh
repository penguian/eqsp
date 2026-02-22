#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.

INCLUDE_PRIVATE=false
for arg in "$@"; do
    if [ "$arg" == "--include-private" ]; then
        INCLUDE_PRIVATE=true
    fi
done

# Standard options: doctests ignoring _private
PYTEST_OPTS="--doctest-modules --ignore=eqsp/_private"

if [ "$INCLUDE_PRIVATE" = true ]; then
    echo "Running coverage including private tests..."
    # Include everything in tests/src
    python3 -m coverage run --source=eqsp -m pytest eqsp tests/src $PYTEST_OPTS
else
    echo "Running coverage excluding private tests..."
    # Ignore the specific private test files
    python3 -m coverage run --source=eqsp -m pytest eqsp tests/src $PYTEST_OPTS \
        --ignore=tests/src/test_private_histograms.py \
        --ignore=tests/src/test_private_partitions.py \
        --ignore=tests/src/test_private_region_props.py
fi

python3 -m coverage report
