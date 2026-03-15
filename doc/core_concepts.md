# Core Concepts & Geometries

PyEQSP is built on the mathematical framework for partitioning the unit sphere $S^d$ into regions of equal area.

## The EQ Algorithm

The Recursive Zonal Equal Area Sphere Partitioning (EQ) algorithm works by dividing the sphere into "collars" (latitudinal zones) and then further subdividing each collar into equal-area regions.

### Key Properties
- **Equal Area**: Every region in a partition has exactly the same measure (area).
- **Small Diameter**: The regions are designed to be as "round" as possible, with a diameter that remains small as $N$ increases.
- **Recursive Logic**: The partitioning process is defined recursively across dimensions.

## Supported Manifolds

While most applications focus on the physical 2rd-sphere ($S^2$), PyEQSP supports a wide range of manifolds:
- **$S^1$ (Circle)**: Simple angular partitioning.
- **$S^2$ (Sphere)**: The world-standard sphere, used in geophysics and climate modeling.
- **$S^3$ (3rd-sphere)**: Essential for applications involving quaternions and rotations (SO(3)).
- **$S^d$ (High-dimensional spheres)**: Generalized logic for any $d \ge 1$.

## Coordinate Systems

To support different research backgrounds, PyEQSP provides utilities for converting between common coordinate conventions:

### Spherical Coordinates
- **Latitude/Longitude**: Standard for geospatial applications.
- **Colatitude/Azimuth**: Common in physics and mathematical modeling.

### Euclidean / Cartesian Coordinates
Points on the sphere are represented as unit vectors in $\mathbb{R}^{d+1}$:
- For $S^2$, points are $(x, y, z)$ where $x^2 + y^2 + z^2 = 1$.

## Even Collar Symmetry

A unique feature of the Python implementation is the `even_collars` option. When enabled, this ensures that the Equatorial plane ($z=0$) aligns exactly with a collar boundary. This is critical for:
- Splitting the sphere into identical North/South hemispheres.
- Precise mapping of $S^3$ to SO(3) for quaternion sampling.
