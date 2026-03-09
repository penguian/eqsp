# Symmetric EQ Partitions

This document describes the `even_collars` parameter of `eq_caps` and
justifies the approach taken.

## Motivation

Some applications require an EQ partition that is symmetric about the
equatorial hyperplane of $S^d$.  For example, a hyperhemispherical grid
filter (Pfaff, Li & Hanebeck, 2020) constructs the northern half of the
partition and mirrors it — via plane reflection or antipodal mapping — to
produce the full sphere.  This requires the equator ($\theta = \pi/2$) to
fall exactly on a cap boundary so that the partition splits cleanly into
two halves of $N/2$ regions each. This inherently requires $N$ to be an
even number.

Furthermore, applications sampling the rotation groups O(3) or SO(3) 
(such as RNA folding, or orientation estimation in robotics) map 4D
quaternions onto $S^3$. Due to the double-covering of $SO(3)$, they only
need to uniformly sample the upper "hyperhemisphere" of $S^3$. This again
requires the equatorial hyperplane ($\theta_3 = \pi/2$) to perfectly fall on
an EQ cap boundary.

## The Modification

The standard EQ algorithm chooses the number of collars by rounding a
continuous ratio to the nearest integer:

$$
n_{\text{collars}} = \max\!\bigl(1,\;\operatorname{round}\bigl((\pi - 2\,c_{\text{polar}}) \,/\, \alpha_{\text{ideal}}\bigr)\bigr)
$$

where $c_{\text{polar}}$ is the polar cap colatitude and
$\alpha_{\text{ideal}} = (\lvert S^d \rvert / N)^{1/d}$ is the ideal
collar angle.

When `even_collars=True`, `eq_caps` instead forces an **even** collar count
(provided $N$ is even; if $N$ is odd, it raises a `ValueError`):

$$
n_{\text{half}} = \max\!\bigl(1,\;\operatorname{round}\bigl((\pi - 2\,c_{\text{polar}}) \,/\, (2\,\alpha_{\text{ideal}})\bigr)\bigr),
\qquad
n_{\text{collars}} = 2\,n_{\text{half}}.
$$

This differs from the standard count by at most $\pm 1$.

## Why Even Collars Place the Equator on a Boundary

The fitting angle is $\alpha_{\text{fit}} = (\pi - 2\,c_{\text{polar}}) /
n_{\text{collars}}$.  The $k$-th collar boundary sits at colatitude
$c_{\text{polar}} + k\,\alpha_{\text{fit}}$.  When $n_{\text{collars}}$ is
even, the boundary at $k = n_{\text{collars}}/2$ falls at

$$
c_{\text{polar}} + \tfrac{n_{\text{collars}}}{2}\,\alpha_{\text{fit}}
\;=\; c_{\text{polar}} + \tfrac{\pi - 2\,c_{\text{polar}}}{2}
\;=\; \frac{\pi}{2}.
$$

## Properties That Hold Regardless of Collar Parity

Two structural properties of the EQ partition are consequences of the
symmetry $\sin(\theta) = \sin(\pi - \theta)$ of the sphere's area element,
and hold for **any** collar count (odd or even):

1. **Collar boundary symmetry.**  The fitting boundaries
   $c_{\text{polar}} + k\,\alpha_{\text{fit}}$ are symmetric about
   $\pi/2$, since
   $n_{\text{collars}} \cdot \alpha_{\text{fit}} = \pi - 2\,c_{\text{polar}}$.

2. **Ideal region count symmetry.**  The ideal (real-valued) region count
   for collar $k$ equals that of collar $(n_{\text{collars}} + 1 - k)$,
   because the two collars are reflections of each other about the equator
   and the area element $\sin^{d-1}(\theta)$ is symmetric about $\pi/2$.

The property that **does** require even collar count is:

3. **Equator on a cap boundary.**  When $N$ is even, an even number of
   collars guarantees a cap boundary falls exactly at $\theta = \pi/2$.
   (If $N$ is odd, it is mathematically impossible for the equator to lie on
   a boundary while maintaining equal area regions, as the hemispheres would
   need to contain a fractional number of regions. For this reason,
   `even_collars=True` requires $N$ to be even).

## Equivalence to Hemisphere Partitioning

The even-collars modification produces the same ideal collar structure as
directly partitioning the hemisphere into $N/2$ equal-area regions.

Consider applying the EQ algorithm to the hemisphere
$[0, \pi/2]$ on $S^d$ with $N/2$ regions.  The ideal region area is
$|S^d|/(N/2) \cdot (1/2) = |S^d|/N$ — the same as for the full sphere
with $N$ regions.  Therefore:

- $c_{\text{polar}}$ is the same (it depends only on $|S^d|/N$).
- $\alpha_{\text{ideal}}$ is the same.
- The number of collars fitting in $[c_{\text{polar}}, \pi/2]$ is
  $\operatorname{round}\bigl((\pi/2 - c_{\text{polar}}) / \alpha_{\text{ideal}}\bigr)$,
  which equals $n_{\text{half}}$ from the even-collars formula.
- The fitting angle $(\pi/2 - c_{\text{polar}}) / n_{\text{half}}$ is the
  same.
- The collar areas and hence the ideal region counts per collar are
  identical.

The only difference is in rounding: the hemisphere approach rounds
$n_{\text{half}} + 2$ ideal counts to sum to $N/2$, while the even-collars
approach rounds $2\,n_{\text{half}} + 2$ counts to sum to $N$.  Since
`round_to_naturals` processes greedily north-to-south, the northern
collar counts agree in both cases.

The forced-symmetry approach is therefore preferred: it reuses the
existing `eq_caps` code path with a one-line change to the collar count.

## Effect on Region Quality

When the even-collars modification changes the collar count by $\pm 1$:

- **Equal area** is preserved exactly — each region still has area
  $|S^d|/N$.
- **Diameter bound** $O(N^{-1/d})$ is preserved (Leopardi, 2009) — only
  the constant factor changes slightly, since the fitting angle deviates
  from the ideal by at most one collar's worth.
- The fitting angle ratio $\alpha_{\text{fit}} / \alpha_{\text{ideal}}$
  stays within approximately 0.8–1.3 for $N \le 200$ on $S^2$, and the
  deviation shrinks as $N$ grows.

## Performance and High-Dimension Implications

The forced symmetry provided by `even_collars=True` introduces two major
architectural benefits, particularly for higher-dimensional spaces like
$S^3$ ($dim=3$), which is heavily used for sampling SO(3) rotations via
quaternions.

### Symmetric Region Distribution

Because $n_{\text{collars}}$ is forced to be even, the continuous sequence
of ideal region counts generated by `ideal_region_list` is perfectly
palindromic (e.g., the 2nd collar and the 2nd-to-last collar have the exact
same ideal area). When $N$ is also even, the strict, greedy tracking inside
`round_to_naturals` perfectly preserves this symmetry. 

Thus, for any $N$ on $S^3$, the northern hemisphere collars contain the exact
same integer region counts as their southern hemisphere equivalents. It is
this precise numeric symmetry that allows a perfectly un-distorted
hyperhemispherical partition.

### The 100% Cache Optimization

The recursive functions `eq_regions` and `eq_point_set_polar` utilize a
calculation cache that re-uses lower-dimensional $S^{d-1}$ sub-partitions
for the southern hemisphere if a collar with the exact same region count was
already generated in the northern hemisphere.

With standard asymmetric partitions, the discrepancy rounding often assigns
slightly different region counts to mirrored collars (e.g. 7 vs 8), resulting
in a high rate of cache misses. By forcing symmetric region distribution,
`even_collars=True` guarantees a **100% cache hit rate** for the entire
southern hemisphere, substantially accelerating partition generation for large
$N$ in $S^3$ (e.g., Chu et al.'s 8,000 quaternion points).

### Performance Benchmarking

To verify these performance gains, the project's benchmark suite includes specific support for symmetric partitions. You can run the comparative benchmarks using:

```bash
# Run all benchmarks in symmetric mode
python3 benchmarks/run_benchmarks.py --even-collars
```

See [doc/benchmarks.md](benchmarks.md) for more details on running and interpreting these performance tests.

## The "Interface Symmetry" Rationale

To maintain high code quality and follow Python best practices, the `eqsp` library utilizes **Interface Symmetry** for its public functions and property helpers. This refers to the practice of maintaining identical function signatures across all related operations, even when a specific parameter (like `even_collars`) has no mathematical effect on a specific function's result (like total area error).

This design provides three key benefits:

1.  **Polymorphic Compatibility**: The generic helper `eq_regions_property(fhandle, ...)` passes a standard set of parameters to `fhandle`. For property functions to be "pluggable," they must all accept the same arguments to prevent `TypeError` exceptions during execution.
2.  **User API Consistency**: Consistency reduces cognitive load. Users can predictably use `func(dim, N, even_collars=True)` for any property function in the `region_props` or `point_set_props` modules without needing to remember which specific properties are invariant to the equatorial split.
3.  **Call-Site Stability**: Maintaining a consistent signature ensures that the user's "contract" with the library remains stable. If the underlying logic or research requirements change in the future, the user's code will not require breaking changes.

## References

1. Leopardi, P. (2006). A partition of the unit sphere into regions of
   equal area and small diameter. *ETNA*, 25, 309–327.

2. Leopardi, P. (2007). *Distributing points on the sphere: Partitions,
   separation, quadrature and energy.* PhD thesis, UNSW.

3. Leopardi, P. (2009). Diameter bounds for equal area partitions of the
   unit sphere. *ETNA*, 35, 1–16.

4. Pfaff, F., Li, K. & Hanebeck, U.D. (2020). A Hyperhemispherical Grid
   Filter for Orientation Estimation. *Proc. 23rd Int. Conf. Information
   Fusion.* DOI: 10.23919/FUSION45008.2020.9190611.
