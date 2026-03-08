# PhD Thesis Examples

This directory contains Python scripts that reproduce the computationally
generated figures from the PhD thesis:

> Paul Leopardi, *"Distributing points on the sphere: Partitions, separation,
> quadrature and energy"*, UNSW, 2007.

Each script uses the `eqsp` Python library (this repository) to reproduce
results originally computed in Matlab using the EQ Sphere Partitioning Toolbox.

## Scripts

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
| `fig_4_10_eqp_voronoi_s2_33.py` | Fig 4.10 | Matplotlib (Agg) | EQP(2, 33) Voronoi cells in 2D stereographic projection (headless) |
| `fig_4_10_eqp_voronoi_s2_33_3d.py` | Fig 4.10 | **Mayavi** | EQP(2, 33) Voronoi edges projected back to S² in 3D |
| `fig_5_1_normalized_energy.py` | Fig 5.1 | Matplotlib (Agg) | Normalized energy of EQP(d, N), d=2,3,4 |
| `fig_5_2_diff_normalized_energy.py` | Fig 5.2 | Matplotlib (Agg) | Convergence of normalized energy |
| `fig_5_3_energy_coeff_s2.py` | Fig 5.3 | Matplotlib (Agg) | Energy coefficient of EQP(2, N), N=2..20,000 |
| `fig_5_4_energy_coeff_s3.py` | Fig 5.4 | Matplotlib (Agg) | Energy coefficient of EQP(3, N), N=2..20,000 |
| `fig_5_5_energy_coeff_s4.py` | Fig 5.5 | Matplotlib (Agg) | Energy coefficient of EQP(4, N), N=2..20,000 |

All scripts save a PNG output file to the current directory and print a
confirmation message on completion.

## Usage

### Matplotlib (Agg) scripts — no display required

These scripts run headlessly and require only the standard `venv`:
```bash
cd examples/phd-thesis
source ../../.venvs/.venv/bin/activate
python fig_3_4_max_diam_s2.py
```

### Mayavi scripts — require `.venvs/.venv_sys`

Scripts labelled **Mayavi** use `eqsp.visualizations` and require
system-installed Mayavi via `.venvs/.venv_sys`:

```bash
cd examples/phd-thesis
export QT_API="pyqt5"
export QT_QPA_PLATFORM="xcb"
```

> **Important:** The environment variables shown above (`QT_API`, `QT_QPA_PLATFORM`) were configured specifically for **Kubuntu Linux 25.10**. Your specific environment may require different values or a different setup entirely.

```bash
# Display help and available arguments
python fig_3_4_max_diam_s2.py --help

# Run with custom parameters
python fig_3_4_max_diam_s2.py --upper-bound 1000
python fig_3_4_max_diam_s2.py --max-points 500
python fig_3_1_partition_s2_33.py
```

All 21 scripts follow the `if __name__ == "__main__":` pattern and can be imported as modules without side effects.

For full setup instructions for `.venvs/.venv_sys`, see
[doc/python_environments.md](../doc/python_environments.md).

### Visual inspection of eqsp outputs

To interactively inspect 2D illustrations and 3D visualizations from the
main library (not thesis-specific), see the inspection scripts:
```bash
python ../../tests/src/inspect_illustrations.py
python ../../tests/src/inspect_visualizations.py
```

For the full testing strategy, see [doc/testing_guide.md](../doc/testing_guide.md).

## Notes on differences from the original figures

  `eqsp.visualizations.show_s2_partition` (Mayavi), which produces a
  comparable 3D rendering.
  
  > [!NOTE]
  > 3D plots generated via Mayavi/VTK may not always be exactly bitwise reproducible. Minor variances (e.g., 1-byte file size differences or ~2 pixel Absolute Error) can occur due to non-deterministic GPU hardware rasterization and anti-aliasing, even when the underlying data is identical.
- **Fig 4.10 (3D Voronoi)**: Computed using `scipy.spatial.SphericalVoronoi`.
  Edges are rendered as true great circle arcs on the sphere using SLERP
  (Spherical Linear Interpolation). This ensures geometric accuracy that
  matches the blue region boundaries. A 2D stereographic projection
  fallback (`fig_4_10_eqp_voronoi_s2_33.py`) is also provided.
- **Figs 5.1–5.5 (Energy/Distance)**: 
  - Minimum distance calculations are optimized ($O(N \log N)$) and finish in seconds even for $N=20,000$.
  - Exact Riesz energy calculations ($s > 0$) use a memory-efficient block-based summation ($O(N)$ peak memory). For $N=20,000$, these typically complete in 5–10 minutes.
  - **Sampling Strategy**: Most Chapter 4 and 5 plots use a hybrid sampling strategy (N=1..100 linear, then 900 logarithmically spaced points up to 20,000) for high resolution across both small and large N.
  - **Energy Coefficient mapping**: Figures 5.3–5.5 plot the specific "energy coefficient" $ec_d(\mathcal{N})$ defined in Section 5.7 of the thesis, which relates to the library output $C$ as $ec_d(\mathcal{N}) = -2 \times C$.

## Not Reproduced

The following figures are excluded as they are theoretical diagrams, not
computational outputs:
- **Figs 3.8–3.12**: Steps of the Feige-Schechtman construction (proof diagrams).
- **Fig 3.2**: Pseudocode description of the partition algorithm (text figure).
- **Tables 3.1–3.3**: Constants from theorem proofs (analytically derived).
- **Table 4.1**: Hamkins-Zeger vs EQP comparison (partially tabulated data).

## Running Time Benchmark (Section 3.10.2)

In addition to the figure reproduction scripts, the project includes a formal replication of the performance benchmark described in **Section 3.10.2: Running time**.

- **Purpose**: Verifies that the recursive zonal partitioning algorithm scales as $O(N^{0.6})$.
- **Script**: `benchmarks/src/benchmark_eq_regions.py`
- **Verification**: The Python implementation achieves empirical scaling coefficients identifying with the original thesis baseline (~0.60) across dimensions $d=1 \dots 11$ and partitions up to $N=2^{22}$ (4.2 million regions).

To execute this benchmark:
```bash
python3 benchmarks/src/benchmark_eq_regions.py --max-d 11 --max-k 22
```
Detailed results are saved in `benchmarks/results/benchmark_eq_regions.log`.
