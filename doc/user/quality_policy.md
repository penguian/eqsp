# Quality Policy & Research Integrity

The **PyEQSP** project maintains a strict quality policy to ensure that researchers can rely on its mathematical outputs and visualizations for high-fidelity scientific work.

## Research Integrity Standards

We adhere to the following benchmarks for every release:

*   **100% Test Coverage**: Every mathematical logic path, coordinate transform, and property metric is exercised by our test suite. This ensures that no "silent" numerical drifts are introduced during updates.
*   **Pylint 10.00/10 Compliance**: We use deep static analysis to enforce strict code quality. While we allow standard mathematical notation (like `N_values` or `Ns`), we enforce consistent variable naming and structural integrity across the entire codebase.
*   **Reproducibility**: The library includes a full "Reproduction Kit" to verify results against the canonical PhD thesis baseline ({ref}`Leopardi, 2007 <leo07>`).

 ## High-Fidelity Verification

 We utilize a multi-layered verification system ("Defense in Depth") to maintain numerical and structural integrity. This strategy deploys automated local hooks for every commit, comprehensive project-wide verification suites, and a robust CI pipeline monitoring all changes across multiple platforms. Beyond standard Python lints, we enforce project-specific guardrails for documentation quality and research terminology to maintain professional scientific standards.

 ## Verification Baselines

Our hybrid testing approach combines:
1.  **Unit Tests**: Comparing results against known-good values from the original MATLAB implementation.
2.  **Doctests**: Ensuring that every example shown in the documentation is live-tested and correct.
3.  **Property Assertions**: Verifying that partitions always meet the "Equal Area" and "Small Diameter" requirements.

For technical details on running the test suite, coverage audits, and performance benchmarks, see the [Technical Testing & Verification](../maintainer/testing_details.md) guide in the Maintenance Guide.
