# Design & Architecture

This document tracks the technical philosophy and internal structure of the PyEQSP library.

## Module Breakdown

The library is organized to separate performance-critical mathematical logic from user-facing utilities.

### Public API (`eqsp.*`)
- **`partitions.py`**: The core of the EQ algorithm. Contains `eq_regions` and `eq_point_set`.
- **`point_set_props.py`**: Modules for measuring distance, energy, and density.
- **`visualizations.py`**: High-level wrappers for Matplotlib and Mayavi.
- **`conversions.py`**: Coordinate system transformations.

```{mermaid}
graph TD
    subgraph PublicAPI ["Public API (eqsp.*)"]
        A[partitions.py] --> C[point_set_props.py]
        A --> V[visualizations.py]
        A --> CONV[conversions.py]
        C --> UTILS[utilities.py]
    end

    subgraph Private ["Internal (_private.*)"]
        A -.-> P_PART[_partitions.py]
        C -.-> P_RPROP[_region_props.py]
    end

    style PublicAPI fill:#f9f,stroke:#333,stroke-width:2px
    style Private fill:#eee,stroke:#999,stroke-dasharray: 5 5
```

### Internal Logic (`eqsp._private`)
Modules prefixed with an underscore are intended for internal use and may change without notice. These include:
- Helper functions for recursive collar calculations.
- Optimization constants.

## The Composite Strategy

PyEQSP uses a "composite" visualization strategy:
1. **Matplotlib** is used for 2D projections and publishing-quality PDF/EPS output.
2. **Mayavi/VTK** is used for real-time 3D interaction.
The `visualizations` module acts as a bridge, choosing the best backend for the requested manifold ($S^2$ vs $S^3$).

```{mermaid}
graph LR
    User([User Call]) --> V[visualizations.py]
    
    V -->|Manifold = S^2, 2D| MAT[Matplotlib]
    V -->|Manifold = S^2, 3D| MAY[Mayavi/VTK]
    V -->|Manifold = S^3| MAY
    
    subgraph Backends ["Plotting Engines"]
        MAT
        MAY
    end

    style User fill:#d4edda,stroke:#28a745
    style Backends fill:#fff3cd,stroke:#ffc107
```

## Extensibility

### Adding New Manifolds
The library is designed to be extensible. To add support for a new manifold:
1. Implement the measure (area) calculation.
2. Define the recursive partitioning logic in `partitions.py`.
3. Update the visualization suite to handle the new projection requirements.

### Property Metrics
New statistical properties can be added by implementing vectorized functions in `point_set_props.py`.

## Numerical Stability

Precision is maintained by:
- Using `numpy.longdouble` where recursion depth might lead to floating-point drift.
- Implementing robust root-finding for cap radius calculations.
