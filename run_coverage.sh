#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.
python3 -m coverage run --source=eqsp -m pytest eqsp tests/src --doctest-modules --ignore=eqsp/_private
python3 -m coverage report
