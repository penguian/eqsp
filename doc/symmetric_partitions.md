# Symmetric EQ Partitions

This document describes the `symmetric` parameter of `eq_caps` and
justifies the approach taken.

## 1. Motivation

Some applications require an EQ partition that is symmetric about the
equatorial hyperplane of $S^d$.  For example, a hyperhemispherical grid
filter (Pfaff, Li & Hanebeck, 2020) constructs the northern half of the
partition and mirrors it — via plane reflection or antipodal mapping — to
produce the full sphere.  This requires the equator ($\theta = \pi/2$) to
fall exactly on a cap boundary so that the partition splits cleanly into
two halves of $N/2$ regions each.

## 2. The Modification

The standard EQ algorithm chooses the number of collars by rounding a
continuous ratio to the nearest integer:

$$
n_{\text{collars}} = \max\!\bigl(1,\;\operatorname{round}\bigl((\pi - 2\,c_{\text{polar}}) \,/\, \alpha_{\text{ideal}}\bigr)\bigr)
$$

where $c_{\text{polar}}$ is the polar cap colatitude and
$\alpha_{\text{ideal}} = (\lvert S^d \rvert / N)^{1/d}$ is the ideal
collar angle.

When `symmetric=True`, `eq_caps` instead forces an **even** collar count:

$$
n_{\text{half}} = \max\!\bigl(1,\;\operatorname{round}\bigl((\pi - 2\,c_{\text{polar}}) \,/\, (2\,\alpha_{\text{ideal}})\bigr)\bigr),
\qquad
n_{\text{collars}} = 2\,n_{\text{half}}.
$$

This differs from the standard count by at most $\pm 1$.

## 3. Why Even Collars Place the Equator on a Boundary

The fitting angle is $\alpha_{\text{fit}} = (\pi - 2\,c_{\text{polar}}) /
n_{\text{collars}}$.  The $k$-th collar boundary sits at colatitude
$c_{\text{polar}} + k\,\alpha_{\text{fit}}$.  When $n_{\text{collars}}$ is
even, the boundary at $k = n_{\text{collars}}/2$ falls at

$$
c_{\text{polar}} + \tfrac{n_{\text{collars}}}{2}\,\alpha_{\text{fit}}
\;=\; c_{\text{polar}} + \tfrac{\pi - 2\,c_{\text{polar}}}{2}
\;=\; \frac{\pi}{2}.
$$

## 4. Properties That Hold Regardless of Collar Parity

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

3. **Equator on a cap boundary.**  Only with an even number of collars
   does a cap boundary fall at $\theta = \pi/2$.  With an odd count the
   middle collar straddles the equator and the partition cannot be split
   cleanly into hemispheres.

## 5. Equivalence to Hemisphere Partitioning

The symmetric modification produces the same ideal collar structure as
directly partitioning the hemisphere into $N/2$ equal-area regions.

Consider applying the EQ algorithm to the hemisphere
$[0, \pi/2]$ on $S^d$ with $N/2$ regions.  The ideal region area is
$|S^d|/(N/2) \cdot (1/2) = |S^d|/N$ — the same as for the full sphere
with $N$ regions.  Therefore:

- $c_{\text{polar}}$ is the same (it depends only on $|S^d|/N$).
- $\alpha_{\text{ideal}}$ is the same.
- The number of collars fitting in $[c_{\text{polar}}, \pi/2]$ is
  $\operatorname{round}\bigl((\pi/2 - c_{\text{polar}}) / \alpha_{\text{ideal}}\bigr)$,
  which equals $n_{\text{half}}$ from the symmetric formula.
- The fitting angle $(\pi/2 - c_{\text{polar}}) / n_{\text{half}}$ is the
  same.
- The collar areas and hence the ideal region counts per collar are
  identical.

The only difference is in rounding: the hemisphere approach rounds
$n_{\text{half}} + 2$ ideal counts to sum to $N/2$, while the symmetric
approach rounds $2\,n_{\text{half}} + 2$ counts to sum to $N$.  Since
`round_to_naturals` processes greedily north-to-south, the northern
collar counts agree in both cases.

The forced-symmetry approach is therefore preferred: it reuses the
existing `eq_caps` code path with a one-line change to the collar count.

## 6. Effect on Region Quality

When the symmetric modification changes the collar count by $\pm 1$:

- **Equal area** is preserved exactly — each region still has area
  $|S^d|/N$.
- **Diameter bound** $O(N^{-1/d})$ is preserved (Leopardi, 2009) — only
  the constant factor changes slightly, since the fitting angle deviates
  from the ideal by at most one collar's worth.
- The fitting angle ratio $\alpha_{\text{fit}} / \alpha_{\text{ideal}}$
  stays within approximately 0.8–1.3 for $N \le 200$ on $S^2$, and the
  deviation shrinks as $N$ grows.

## 7. References

1. Leopardi, P. (2006). A partition of the unit sphere into regions of
   equal area and small diameter. *ETNA*, 25, 309–327.

2. Leopardi, P. (2007). *Distributing points on the sphere: Partitions,
   separation, quadrature and energy.* PhD thesis, UNSW.

3. Leopardi, P. (2009). Diameter bounds for equal area partitions of the
   unit sphere. *ETNA*, 35, 1–16.

4. Pfaff, F., Li, K. & Hanebeck, U.D. (2020). A Hyperhemispherical Grid
   Filter for Orientation Estimation. *Proc. 23rd Int. Conf. Information
   Fusion.* DOI: 10.23919/FUSION45008.2020.9190611.
