#!/usr/bin/env python3
"""Shared core for PyEQSP performance benchmarks."""

import time

import numpy as np

from eqsp.histograms import eq_find_s2_region
from eqsp.partitions import eq_point_set_polar, eq_regions
from eqsp.point_set_props import eq_energy_dist, eq_min_dist
from eqsp.region_props import eq_area_error
from eqsp.utilities import area_of_sphere, sradius_of_cap


def generate_125_sequence(n_max):
    """Generate 1-2-5 logarithmic sequence: 10, 20, 50, 100..."""
    if n_max < 10:
        return np.array([max(1, int(n_max))])
    v = np.array([1, 2, 5])
    p = np.arange(0, int(np.ceil(np.log10(n_max))) + 1)
    vals = (v[:, None] * (10**p)).flatten()
    vals = np.sort(vals)
    return vals[(vals >= 10) & (vals <= n_max)]


def perform_benchmark(label, timing_func, n_values, description="N"):
    """Standardized performance benchmark loop with scaling analysis."""
    if len(n_values) == 0:
        print(f"{label:<15} | No values to benchmark")
        return
    print(f"{label:<15} | {'Time (s)':>10}")
    print("-" * 28)

    results = []
    for N in n_values:
        t0 = time.perf_counter()
        timing_func(N)
        t_elapsed = time.perf_counter() - t0
        results.append([N, t_elapsed])
        if N >= 100:
            print(f"{N:<15} | {t_elapsed:>10.4f}")

    results = np.array(results)
    # Scaling Analysis (from N=100)
    mask = (results[:, 0] >= 100) & (results[:, 1] > 0.0001)
    if np.sum(mask) >= 2:
        log_n = np.log(results[mask, 0])
        log_t = np.log(results[mask, 1])
        p = np.polyfit(log_n, log_t, 1)
        print("-" * 28)
        print(f"Best fitting order: O({description}^{p[0]:.2f})")


def run_area_error(n_max, dim=2, even_collars=False):
    """Runner for eq_area_error benchmark."""
    n_values = generate_125_sequence(n_max)
    # Warm-up
    for n_warm in [10, 20, 50]:
        eq_area_error(dim, [n_warm], even_collars=even_collars)

    perform_benchmark(
        "N", lambda N: eq_area_error(dim, [N], even_collars=even_collars), n_values
    )


def run_eq_regions(n_max, dim=2, even_collars=False):
    """Runner for eq_regions benchmark."""
    n_values = generate_125_sequence(n_max)
    # Warm-up
    for n_warm in [10, 20, 50]:
        eq_regions(dim, int(n_warm), even_collars=even_collars)

    perform_benchmark(
        "N", lambda N: eq_regions(dim, int(N), even_collars=even_collars), n_values
    )


def run_histograms(n_max, n_regions=1000, even_collars=False):
    """Runner for eq_find_s2_region benchmark."""
    n_values = generate_125_sequence(n_max)
    # Warm-up
    for n_warm in [10, 20, 50]:
        test_points = eq_point_set_polar(2, int(n_warm))
        eq_find_s2_region(test_points, n_regions, even_collars=even_collars)

    perform_benchmark(
        "Points",
        lambda N: eq_find_s2_region(
            eq_point_set_polar(2, int(N)), n_regions, even_collars=even_collars
        ),
        n_values,
    )


def run_sradius(n_max, dim=3, _even_collars=False):
    """Runner for sradius_of_cap benchmark (ignores even_collars)."""
    n_values = generate_125_sequence(n_max)
    max_area = area_of_sphere(dim)
    # Warm-up
    for n_warm in [10, 20, 50]:
        sradius_of_cap(dim, np.linspace(0.1, max_area - 0.1, int(n_warm)))

    perform_benchmark(
        "Size",
        lambda N: sradius_of_cap(dim, np.linspace(0.1, max_area - 0.1, int(N))),
        n_values,
    )


def run_mindist(n_max, dim=2, even_collars=False):
    """Runner for eq_min_dist benchmark."""
    n_values = generate_125_sequence(n_max)
    # Warm-up
    for n_warm in [10, 20, 50]:
        eq_min_dist(dim, int(n_warm), even_collars=even_collars)

    perform_benchmark(
        "N", lambda N: eq_min_dist(dim, int(N), even_collars=even_collars), n_values
    )


def run_energy_dist(n_max, dim=2, s=None, even_collars=False):
    """Runner for point_set_energy_dist benchmark."""
    n_values = generate_125_sequence(n_max)
    # Warm-up
    for n_warm in [10, 20, 50]:
        eq_energy_dist(dim, int(n_warm), s=s, even_collars=even_collars)

    perform_benchmark(
        "N",
        lambda N: eq_energy_dist(dim, int(N), s=s, even_collars=even_collars),
        n_values,
    )
