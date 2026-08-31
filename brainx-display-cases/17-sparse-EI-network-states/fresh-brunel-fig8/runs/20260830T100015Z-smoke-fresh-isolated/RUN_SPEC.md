# Smoke run specification

- Purpose: mechanical gate for the fresh isolated implementation before paper-scale execution.
- Model: the locked Brunel Model A equations and update order, reduced to `NE=800`, `NI=200`, `CE=80`, `CI=20`, and `Cext=80`.
- Protocol: `dt=0.1 ms`, `100 ms` burn-in, `1,000 ms` analysis, all four fixed Fig. 8 parameter points, seed `17729`.
- Expected limitation: finite-size dynamics and rates are not expected to match the paper-scale acceptance predicates. Those verdicts are diagnostic only in this smoke run.
- Gate: require successful completion, four parseable raw panel files, finite metrics, fixed graph/probe hashes, provenance, result assessment, and manifests.
- Rendering: forbidden before review passes; this run produces no figure.
