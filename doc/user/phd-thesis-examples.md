# Appendix E: Thesis Research Reproduction & Setup

Each script uses the **PyEQSP** Python library (this repository) to reproduce results originally computed in MATLAB using the EQ Sphere Partitioning Toolbox.

> Paul Leopardi, *"Distributing points on the sphere: Partitions, separation, quadrature and energy"*, UNSW, 2007.

:::{note} Beta Feedback Wanted
If you are running these thesis reproduction scripts on your local system, we would love to know if the results are consistent with your hardware. Please share a screenshot or report any numerical variances in our [Feedback Hub](https://github.com/penguian/pyeqsp/issues/26).
:::

## Script Inventory

The following table maps the scripts in `examples/phd-thesis/src/` to the original thesis figures.

| Script | Thesis Figure | Backend | Description |
|:-------|:-------------|:--------|:------------|
| `fig_3_1_partition_s2_33.py` | Fig 3.1 | **Mayavi** | 3D partition EQ(2, 33) on S² |
| `fig_3_3_algorithm_s3_99.py` | Fig 3.3 | Matplotlib (Agg) | Algorithm illustration for EQ(3, 99) |
| `fig_3_4_max_diam_s2.py` | Fig 3.4 | Matplotlib (Agg) | Max diameter coefficients of EQ(2, N), N=1..100,000 |
| `fig_3_5_max_diam_s3.py` | Fig 3.5 | Matplotlib (Agg) | Max diameter coefficients of EQ(3, N), N=1..100,000 |
| `fig_3_6_max_diam_s4.py` | Fig 3.6 | Matplotlib (Agg) | Max diameter coefficients of EQ(4, N), N=1..100,000 |
| `fig_3_7_max_diam_multi_dim.py` | Fig 3.7 | Matplotlib (Agg) | Max diameters of EQ(d, 2^k), d=2..8, k=1..20 (parallelized) |
| `fig_4_1_eqp_s2_33.py` | Fig 4.1 | **Mayavi** | 3D EQ code EQP(2, 33) with partition EQ(2, 33) |
| `fig_4_2_min_dist_s2.py` | Fig 4.2 | Matplotlib (Agg) | Min distance coefficient of EQP(2, N), N=1..20,000 |
| `fig_4_3_min_dist_s3.py` | Fig 4.3 | Matplotlib (Agg) | Min distance coefficient of EQP(3, N), N=1..20,000 |
| `fig_4_4_min_dist_s4.py` | Fig 4.4 | Matplotlib (Agg) | Min distance coefficient of EQP(4, N), N=1..20,000 |
| `fig_4_5_packing_s2.py` | Fig 4.5 | Matplotlib (Agg) | Packing density of EQP(2, N), N=1..20,000 |
| `fig_4_6_packing_s3.py` | Fig 4.6 | Matplotlib (Agg) | Packing density of EQP(3, N), N=1..20,000 |
| `fig_4_7_packing_s4.py` | Fig 4.7 | Matplotlib (Agg) | Packing density of EQP(4, N), N=1..20,000 |
| `fig_4_8_wyner_s2_s3_s4.py` | Fig 4.8 | Matplotlib (Agg) | Wyner ratios for EQP(2), EQP(3), EQP(4), N=2..20,000 |
| `fig_4_9_wyner_s5_s6_s7.py` | Fig 4.9 | Matplotlib (Agg) | Wyner ratios for EQP(5), EQP(6), EQP(7), N=2..20,000 |
| `fig_4_10_eqp_voronoi_s2_33.py` | Fig 4.10 | **Mayavi** | EQP(2, 33) Voronoi edges on S² in 3D |
| `fig_5_1_normalized_energy.py` | Fig 5.1 | Matplotlib (Agg) | Normalized energy of EQP(d, N), d=2,3,4 |
| `fig_5_2_diff_normalized_energy.py` | Fig 5.2 | Matplotlib (Agg) | Convergence of normalized energy |
| `fig_5_3_energy_coeff_s2.py` | Fig 5.3 | Matplotlib (Agg) | Energy coefficient of EQP(2, N), N=2..20,000 |
| `fig_5_4_energy_coeff_s3.py` | Fig 5.4 | Matplotlib (Agg) | Energy coefficient of EQP(3, N), N=2..20,000 |
| `fig_5_5_energy_coeff_s4.py` | Fig 5.5 | Matplotlib (Agg) | Energy coefficient of EQP(4, N), N=2..20,000 |

All scripts save a PNG output file to the current directory and print a confirmation message on completion.

## Reproduction Notes

- **3D Figures**: Visualizations using `Mayavi` are not always bitwise identical across hardware due to non-deterministic GPU rasterization, even when the underlying mesh is identical.
- **Min-distance**: Unified search using KDTrees ensures $O(N \log N)$ scaling, allowing $N=20,000$ to complete in seconds.
- **Riesz Energy**: Exact summations for $s > 0$ use memory-efficient block-tiling to ensure $O(N)$ peak RAM.

## Running Time Benchmark (Section 3.10.2)

Beyond figure reproduction, the project includes a formal replication of the performance benchmark described in **Section 3.10.2: Running time**. It verifies that the recursive zonal partitioning algorithm scales as $O(N^{0.6})$.

To execute:
```bash
python3 benchmarks/src/benchmark_eq_regions.py --max-d 11 --max-k 22
```

## Technical Setup & Troubleshooting
<a id="thesis-setup"></a>

Reproduction scripts for 3D figures require the **Mayavi** engine. We strongly recommend using the **`venv_sys`** environment for this work. See the {ref}`venv-sys-setup` section in **Appendix B** for more.

### Backend Configuration

Rendering 3D Great Circle arcs and Voronoi cells on the sphere requires specific VTK/Mayavi backends. On Linux systems, calibrate your environment as follows:

```bash
export QT_API="pyqt5"
export QT_QPA_PLATFORM="xcb"
```

> [!WARNING]
> Without these exports, Mayavi may fail to initialize a window or crash with a `Segmentation Fault` when attempting to rasterize GREAT CIRCLE edges.

### Bitwise Reproducibility

PyEQSP aims for identical numerical results to the original thesis. But users should be aware of two potential sources of variance:

1.  **Hardware Rasterization**: 3D plots generated via Mayavi/VTK use GPU hardware. Minor variances (e.g., ~2 pixel differences) can occur between different GPUs or drivers due to non-deterministic anti-aliasing.
2.  **Floating-Point Drift**: While we use `numpy.longdouble` for critical recursions, extreme depths in high dimensions ($d > 8$) may show sub-microscopic differences across different CPU architectures.

### Headless Execution

To run reproduction scripts on a server or in CI without a display:

```bash
export HEADLESS=1
python3 examples/phd-thesis/src/fig_3_4_max_diam_s2.py
```

Numerical scripts (Matplotlib `Agg` backend) will save PNGs directly. 3D scripts (Mayavi) will attempt to use an offscreen buffer if `xvfb` is available.
