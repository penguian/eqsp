# Symmetry & Parity Implementation

This document provides the mathematical justification and implementation details for the `even_collars` parameter in `eq_caps`.

## Mathematical Derivations

The standard EQ algorithm determines the collar count through rounding:

$$
n_{\text{collars}} = \max\!\bigl(1,\;\operatorname{round}\bigl((\pi - 2\,c_{\text{polar}}) \,/\, \alpha_{\text{ideal}}\bigr)\bigr)
$$

### Forced Parity

When `even_collars=True`, we constrain $n_{\text{collars}}$ to be even by rounding the half-count:

$$
n_{\text{half}} = \max\!\bigl(1,\;\operatorname{round}\bigl((\pi - 2\,c_{\text{polar}}) \,/\, (2\,\alpha_{\text{ideal}})\bigr)\bigr),
\qquad
n_{\text{collars}} = 2\,n_{\text{half}}.
$$

This ensures that the boundary at $k = n_{\text{collars}}/2$ sits exactly at:

$$
c_{\text{polar}} + \tfrac{n_{\text{collars}}}{2}\,\alpha_{\text{fit}}
\;=\; c_{\text{polar}} + \tfrac{\pi - 2\,c_{\text{polar}}}{2}
\;=\; \frac{\pi}{2}.
$$

## Architectural Optimizations

### Cache Reuse Strategy

PyEQSP's recursive partitioning uses a `_private` cache for $S^{d-1}$ partitions. **Hemispherical Symmetry**: Partitions generated with `even_collars=True` are perfectly symmetric about the equator. This allows the Southern hemisphere regions to be calculated once and then mirrored, providing a 100% cache hit rate for the second half of the partition generation for $S^3$.

### Interface Symmetry

To maintain polymorphic compatibility, all property functions must accept `even_collars`, even if the property (like total area error) is parity-invariant. This ensures call-site stability for users while allowing internal mathematics to exploit symmetry when available.
For foundational citations, see the Volume 2 [References](references_vol2.md).

## Audit Summary (PyEQSP 0.99.9 Beta)

- **Derivation Alignment**: The `even_collars` algebraic derivation structurally mirrors the runtime calculation in `eq_caps()` (`eqsp/partitions.py`).
- **Cache Integrity**: The $n_{\text{collars}}/2$ cache boundary correctly mirrors collar $k$ to $n_{\text{collars}} - k + 1$ ensuring 100% cache utilization for $S^3$ southern hemisphere regions.
- **Status**: VERIFIED
