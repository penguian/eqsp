# Authors and Acknowledgements

**Recursive Zonal Equal Area (EQ) Sphere Partitioning Toolbox**

- **Release 1.12.1** (2025-08-10)
- **Release 1.12** (2024-10-20): Copyright 2024-2025 Paul Leopardi
- **Release 1.10** (2005-06-26): Copyright 2004-2005 Paul Leopardi for the University of New South Wales.

For licensing, see `COPYING`.
For revision history, see `CHANGELOG`.

## Origin

Maple and Matlab code is based on work by:
- **Ed Saff** [SafSP], [Saf03]
- **Ian Sloan** [Slo03]

## References

- **[Dahl78]** B. E. J. Dahlberg, *"On the distribution of Fekete points"*, Duke Math. J. 45 (1978), no. 3, pp. 537--542.
- **[KuiS98]** A. B. J. Kuijlaars, E. B. Saff, *"Asymptotics for minimal discrete energy on the sphere"*, Transactions of the American Mathematical Society, v. 350 no. 2 (Feb 1998) pp. 523--538.
- **[KuiSS04]** A. B. J. Kuijlaars, E. B. Saff, X. Sun, *"On separation of minimal Riesz energy points on spheres in Euclidean spaces"*, Journal of computational and applied mathematics 199.1 (2007): 172-180.
- **[LeGS01]** T. Le Gia, I. H. Sloan, *"The uniform norm of hyperinterpolation on the unit sphere in an arbitrary number of dimensions"*, Constructive Approximation (2001) 17: p249-265.
- **[Leo06]** P. Leopardi, *"A partition of the unit sphere into regions of equal area and small diameter"*, Electronic Transactions on Numerical Analysis, Volume 25, 2006, pp. 309-327.
- **[Leo07]** P. Leopardi, *"Distributing points on the sphere: Partitions, separation, quadrature and energy"*, PhD thesis, UNSW, 2007.
- **[Leo09]** P. Leopardi, *"Diameter bounds for equal area partitions of the unit sphere"*, Electronic Transactions on Numerical Analysis, Volume 35, 2009, pp. 1-16.
- **[Leo24]** P. Leopardi, *"The applicability of equal area partitions of the unit sphere"*, Journal of Approximation Software, 1(2), 2024.
- **[Mue98]** C. Mueller, *"Analysis of spherical symmetries in Euclidean spaces"*, Springer, 1998.
- **[RakSZ94]** E. A. Rakhmanov, E. B. Saff, Y. M. Zhou, *"Minimal discrete energy on the sphere"*, Mathematics Research Letters, 1 (1994), pp. 647--662.
- **[RakSZ95]** E. A. Rakhmanov, E. B. Saff, Y. M. Zhou, *"Electrons on the sphere"*, Computational methods and function theory 1994 (Penang), pp. 293--309, Ser. Approx. Decompos., 5, World Sci. Publishing, River Edge, NJ, 1995.
- **[SafK97]** E. B. Saff, A. B. J. Kuijlaars, *"Distributing many points on a sphere"*, Mathematical Intelligencer, v19 no1 (1997), pp. 5--11.
- **[SafSP]** E. B. Saff, *"Sphere Points"*, http://www.math.vanderbilt.edu/~esaff/sphere_points.html
- **[Saf03]** Ed Saff, *"Equal-area partitions of sphere"*, Presentation at UNSW, 2003-07-28.
- **[Slo03]** Ian Sloan, *"Equal area partition of S^3"*, Notes, 2003-07-29.
- **[WeiMW]** E. W. Weisstein, *"Hypersphere"*, From MathWorld--A Wolfram Web Resource. http://mathworld.wolfram.com/Hypersphere.html
- **[Zho95]** Y. M. Zhou, *"Arrangement of points on the sphere"*, PhD thesis, University of South Florida, 1995.
- **[Zho98]** Y. M. Zhou, *"Equidistribution and extremal energy of N points on the sphere"*, Modelling and computation for applications in mathematics, science, and engineering (Evanston, IL, 1996), pp. 39--57, Numer. Math. Sci. Comput., Oxford Univ. Press, New York, 1998.

## Research Context

The `eqsp` repository is the software implementation of research into Recursive Zonal Equal Area (EQ) sphere partitioning. The mathematical foundation and original context for this software are provided by the following works:

- **PhD Thesis (2007)**: *"Distributing points on the sphere: Partitions, separation, quadrature and energy"*. This thesis describes the partition of the unit sphere into regions of equal area and bounded diameter, establishing the theoretical bounds that justify the partitioning algorithm.
- **Publications**: The core algorithms and diameter bounds are described and proven in **Leopardi (2006)** and **Leopardi (2009)**. A more recent case study, **Leopardi (2024)**, examines the applicability and impact of these constructions.

The `eqsp` toolbox is the modern Python-native incarnation of the research presented in these documents, evolving from the original Matlab implementation to provide an open-source tool for the scientific community.

## Installation and Utilities (Matlab Original)

- **Toolbox Installer 2.2**, 2003-07-22 by Rasmus Anthin.
- **Matlab Central File Exchange**: https://au.mathworks.com/matlabcentral/fileexchange/3726-toolbox-installer-2-2

Files modified and relicensed with permission of Rasmus Anthin in the original Matlab toolbox:
- `private/install.m`
- `private/uninstall.m`

## AI Assistance

The original port from Matlab to Python was assisted by **GitHub Copilot**.

The completion of the port, including verification, testing, illustration porting, and documentation, was performed with the assistance of **Google Antigravity** powered by **Gemini 3 Pro**.
