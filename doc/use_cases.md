# The Applicability of PyEQSP: Use Cases

Based on: *"The applicability of equal area partitions of the unit sphere"*, {ref}`Leopardi (2024) <leo24jas>`, Journal of Approximation Software, 1(2).

## History: From Toolbox to Library

The journey of Equal Area Sphere Partitioning (EQ) code began with the 2005 **Recursive Zonal Equal Area Sphere Partitioning Toolbox** for MATLAB. Over nearly two decades, the toolbox supported numerous research projects. **PyEQSP** is the modern Python-based successor, designed for high performance and integration with the scientific Python ecosystem (NumPy, SciPy, Matplotlib).

## Cross-Disciplinary Applications

The EQ algorithm provides a robust way to partition the unit sphere $\mathbb{S}^d$ into regions of equal area and small diameter. This property is critical in several scientific domains.

### Biology and Bioinformatics
*   **Sampling Rotations ({ref}`Chu et al., 2009 <chu09>`)**: EQSP partitions were used to sample rotations of protein domains ($S^3$). This allowed for an efficient and unbiased exploration of the conformational space to simulate electrostatic interactions and protein flexibility.
*   **Species Diversity Visualization ({ref}`Arrigo et al., 2012 <arr12>`)**: The R2G2 package incorporates EQSP partitions to provide global grids for 3D histograms in Google Earth, enabling quantitative visualization of species richness across equal-area regions.

### Medicine and MRI
*   **SPARKLING MRI Trajectories ({ref}`Lazarus et al., 2021 <laz21>`)**: EQSP partitioning was used to sample k-space trajectories on a sphere, ensuring trajectories are distributed according to a target density while maintaining globally uniform sampling.
*   **Automated Brain Parcellation ({ref}`Das and Maharatna, 2020 <das20>`)**: EQ partitions were used to create a non-anatomical, equal-area "igloo" grid of the brain, facilitating the analysis of structural and functional connectivity.

### Climate and Geoscience
*   **Arctic Temperature Gridding ({ref}`Werner et al., 2018 <wer18>`)**: EQSP partitions were used as a global equal-area grid to construct millennium-length summer temperature reconstructions, avoiding high-latitude distortions.
*   **Empirical Mode Decomposition ({ref}`Fauchereau et al., 2008 <fau08>`)**: EQSP provided the mandatory equal-area grid for applying 2D EMD to geophysical fields, separating spatial scales in climate variations.

### Planetary Science and Geophysics
*   **Lunar Tectonic Patterns ({ref}`Matsuyama et al., 2021 <mat21>`)**: Digitized fault segments were sampled into 400 equal-area regions to distinguish between isotropic contraction and other stress-generating mechanisms on the moon.
*   **Geomagnetic Virtual Observatories ({ref}`Hammer et al., 2021 <ham21>`)**: EQSP partitions were used to define a global grid of GVOs for studying sub-decadal variations in the Earth's core field.

### Engineering and Materials Science
*   **Composite Fiber Orientations ({ref}`Sabiston et al., 2021 <sab21>`)**: EQSP partitioning (specifically with 1200 partitions) was used to define representative fiber orientations for micromechanics modelling of injection-molded composites.

### Numerical Weather Prediction
*   **Parallel Load Balancing ({ref}`Mozdzynski et al., 2015 <moz15>`)**: EQ-regions were used to decompose reduced Gaussian grids into equal-area regions for efficient parallel processing in ECMWF's IFS model.

### Mathematical Physics and Estimation
*   **Orientation Estimation ({ref}`Pfaff et al., 2020 <pfa20>`)**: Developed hyperhemispherical grid filters based on the EQSP algorithm for robust orientation estimation in robotics.
*   **Fermi Gas Correlation Energy ({ref}`Benedikter et al., 2021 <ben21>`)**: The EQSP construction was used to partition the Fermi surface in momentum space into patches, a key step in calculating correlation energy.

## Evaluating Performance

In comparative studies, PyEQSP's recursive zonal partitioning has shown distinct advantages over other methods like $k$-means clustering, spiral points, and icosahedral partitions, particularly in terms of area consistency and bounded-diameter regions for large $N$.
