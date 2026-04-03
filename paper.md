---
title: 'PyEQSP: A Python Library for Recursive Zonal Equal Area Sphere Partitioning'
tags:
  - Python
  - sphere partitioning
  - numerical analysis
  - discrete geometry
  - manifolds
authors:
  - name: Paul Leopardi
    orcid: 0000-0003-2891-5969
    affiliation: "1, 2"
affiliations:
  - name: ACCESS-NRI
    index: 1
  - name: The Australian National University, Australia
    index: 2
date: 01 April 2026
bibliography: paper.bib
---

# Summary

PyEQSP is a Python library implementing the Recursive Zonal Equal Area (EQ) partitioning algorithm for the unit sphere $S^d$ in $\mathbb{R}^{d+1}$ for any dimension $d \ge 1$. The algorithm divides the sphere into $N$ regions of equal area and small diameter, providing a constructive solution to the problem of distributing points or partitioning regions on high-dimensional spheres. This implementation facilitates coordinate transformations, analysis of region properties, and the generation of spherical codes for use in numerical integration, modeling, and visualization.

![A 3-dimensional EQ partition $EQ(2,100)$ of the sphere $S^2$ into 100 regions, showing the polar caps and the symmetric collar structure.](doc/_static/images/s2_partition_3d.png)

# Statement of Need

The problem of partitioning a sphere into regions of equal area and small diameter is a recurrent requirement in many fields of science and engineering. While the problem is easily solved for the circle $S^1$, it becomes significantly more difficult to pose and solve for $d > 1$ [@Saf97]. The EQ algorithm provides a pragmatic construction with proven diameter bounds for $S^d$ [@Leo06; @Leo09], making it a standard choice for applications requiring well-distributed points or balanced partitions on spherical manifolds.

The applicability of the EQ algorithm has been demonstrated in a wide range of disciplines, including climate science [@Wer18; @Fau08], medical imaging [@Laz21], robotics and orientation estimation [@Pfa20], and mathematical physics [@Ben21]. In particular, PyEQSP serves as the functional successor to the original **Recursive Zonal Equal Area Sphere Partitioning Toolbox** for MATLAB [@Leo05; @Leo07; @Leo24], which has been used in academic research for nearly two decades.

By transitioning from a proprietary MATLAB environment to an open-source Python implementation, the library removes significant barriers to entry and integrates directly with the modern scientific Python ecosystem (NumPy, SciPy, Matplotlib). This transition ensures that researchers can continue to use and extend the algorithm in a reproducible, high-performance environment.

# The EQ Algorithm and Implementation

The core logic of PyEQSP is based on a recursive zonal partitioning scheme originally described in [@Leo06] and detailed in [@Leo07]. For a given dimension $d$ and number of regions $N$, the sphere is divided into a North polar cap, a sequence of collars arranged symmetrically around the polar axis, and a South polar cap. Each collar is itself a product of intervals in spherical coordinates and is partitioned recursively using the algorithm for dimension $d-1$.

PyEQSP introduces several technical refinements over its predecessors:
* **Vectorization**: Leveraging NumPy for efficient coordinate transformations and property analysis, maintaining $O(N \log N)$ or $O(N)$ performance for large-scale simulations.
* **Hemisphere Partitioning**: Explicit support for the `even_collars` parameter to ensure partitions align with the equatorial hyperplane, which is essential for applications such as $S^3$ hemisphere to $\text{SO}(3)$ quaternion sampling.
* **Maintenance Infrastructure**: A modern CI/CD suite with comprehensive test coverage and strict linting (Ruff, Pylint) to ensure reliable research reproducibility.

The library enables the generation of region boundaries (`eq_regions`), centre points in both polar (`eq_point_set_polar`) and Cartesian (`eq_point_set`) coordinates, and the calculation of spherical metrics such as the minimum distance and Riesz energy of point sets.

# Acknowledgements

The original MATLAB implementation of the EQ toolbox was supported by the University of New South Wales and the Australian National University. AI-assisted porting was performed with the support of GitHub Copilot and Google Antigravity.

# References
