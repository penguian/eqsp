# PyEQSP: Python Equal Area Sphere Partitioning Library

**Release 1.0b1** (2026-04-16): Copyright 2026 Paul Leopardi

# Authors and Acknowledgements

For licensing, see `LICENSE`.
For revision history, see `CHANGELOG.md`.

## Ancestry

The **PyEQSP** Python library is a new implementation based on the original **Recursive Zonal Equal Area (EQ) Sphere Partitioning Toolbox** for Matlab.

### Matlab Toolbox History
- **Matlab Release 1.12.1** (2025-08-10)
- **Matlab Release 1.12** (2024-10-20)
- **Matlab Release 1.10** (2005-06-26)

## Origin

The prototype Maple code and the Matlab toolbox are based on work by:
- **Ed Saff** [SafSP], [Saf03]
- **Ian Sloan** [Slo03]

## References

- **[Dah78]** Dahlberg, B. E. J. (1978). *"On the distribution of Fekete points"*, Duke Math. J. 45 (1978), no. 3, pp. 537--542.
- **[Kui98]** Kuijlaars, A. B. J. and Saff, E. B. (1998). *"Asymptotics for minimal discrete energy on the sphere"*, Transactions of the American Mathematical Society, v. 350 no. 2, pp. 523--538.
- **[Kui07]** Kuijlaars, A. B. J., Saff, E. B., and Sun, X. (2007). *"On separation of minimal Riesz energy points on spheres in Euclidean spaces"*, Journal of Computational and Applied Mathematics, 199(1), 172-180.
- **[LeGS01]** T. Le Gia, I. H. Sloan, *"The uniform norm of hyperinterpolation on the unit sphere in an arbitrary number of dimensions"*, Constructive Approximation (2001) 17: p249-265.
- **[Leo06]** Leopardi, P. (2006). *"A partition of the unit sphere into regions of equal area and small diameter"*, Electronic Transactions on Numerical Analysis, Volume 25, pp. 309-327.
- **[Leo07]** Leopardi, P. (2007). *"Distributing points on the sphere: Partitions, separation, quadrature and energy"*, PhD thesis, UNSW.
- **[Leo09]** Leopardi, P. (2009). *"Diameter bounds for equal area partitions of the unit sphere"*, Electronic Transactions on Numerical Analysis, Volume 35, pp. 1-16.
- **[Leo24-JAS]** Leopardi, P. (2024). *"The applicability of equal area partitions of the unit sphere"*, Journal of Approximation Software, 1(2). DOI: 10.13135/jas.10248.
- **[Mue98]** C. Mueller, *"Analysis of spherical symmetries in Euclidean spaces"*, Springer, 1998.
- **[Rak94]** Rakhmanov, E. A., Saff, E. B., and Zhou, Y. M. (1994). *"Minimal discrete energy on the sphere"*, Mathematics Research Letters, 1, pp. 647--662.
- **[Rak95]** E. A. Rakhmanov, E. B. Saff, Y. M. Zhou, *"Electrons on the sphere"*, Computational methods and function theory 1994 (Penang), pp. 293--309, Ser. Approx. Decompos., 5, World Sci. Publishing, River Edge, NJ, 1995.
- **[Saf97]** Saff, E. B. and Kuijlaars, A. B. J. (1997). *"Distributing many points on a sphere"*, Mathematical Intelligencer, v19 no1, pp. 5--11.
- **[SafSP]** E. B. Saff, *"Sphere Points"*, http://www.math.vanderbilt.edu/~esaff/sphere_points.html
- **[Saf03]** Ed Saff, *"Equal-area partitions of sphere"*, Presentation at UNSW, 2003-07-28.
- **[Slo03]** Ian Sloan, *"Equal area partition of S³"*, Notes, 2003-07-29.
- **[WeiMW]** E. Weisstein, *"Spherical Coordinates"*, MathWorld -- A Wolfram web resource, http://mathworld.wolfram.com/SphericalCoordinates.html
- **[Zho95]** Y. M. Zhou, *"Arrangement of points on the sphere"*, PhD thesis, University of South Florida, 1995.
- **[Zho98]** Y. M. Zhou, *"Equidistribution and extremal energy of N points on the sphere"*, Modelling and computation for applications in mathematics, science, and engineering (Evanston, IL, 1996), pp. 39--57, Numer. Math. Sci. Comput., Oxford Univ. Press, New York, 1998.

## Research Context

The **PyEQSP** repository is the software implementation of research into Recursive Zonal Equal Area (EQ) sphere partitioning. The mathematical foundation and original context for this software are provided by the following works:

- **PhD Thesis (2007)**: *"Distributing points on the sphere: Partitions, separation, quadrature and energy"*. This thesis describes the partition of the unit sphere into regions of equal area and bounded diameter, establishing the theoretical bounds that justify the partitioning algorithm.
- **Publications**: The core algorithms and diameter bounds are described and proven in **Leopardi (2006)** and **Leopardi (2009)**. A more recent case study, **Leopardi (2024)**, examines the applicability and impact of these constructions.

The **PyEQSP** library is the Python implementation of the research presented in these documents, evolving from the original Matlab implementation to provide an open-source tool for the scientific community.

## Installation and Utilities (Matlab Original)

- **Toolbox Installer 2.2**, 2003-07-22 by Rasmus Anthin.
- **Matlab Central File Exchange**: https://au.mathworks.com/matlabcentral/fileexchange/3726-toolbox-installer-2-2

Files modified and relicensed with permission of Rasmus Anthin in the original Matlab toolbox:
- `private/install.m`
- `private/uninstall.m`

## AI Support

GitHub Copilot assisted with the original port from Matlab to Python.

Google Antigravity, powered by Gemini 3 Pro, completed the port, including verification, testing, illustration porting, and documentation.
