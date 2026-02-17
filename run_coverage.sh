#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.
python3 -m coverage run --source=eqsp -m pytest
python3 -m coverage report
