# Test results

## Focused suite

- Command: `MPLCONFIGDIR=/tmp/brainx-mpl-cache pytest -q test_sparse_ei_network.py`
- Result: `9 passed in 26.87s`
- Backend: CPU
- Covered invariants: canonical units and external-rate conversion; exact unique fixed indegree and autapse exclusion; 15-step delay impulse; LIF threshold/reset/refractory behavior; independent deterministic replay; eager/compiled parity with separate transform-owned topology objects; finite active external-drive control with recurrence disabled; prospective classifier predicates; exact locked production-config round trip.

## End-to-end smoke

- Command: `MPLCONFIGDIR=/tmp/brainx-mpl-cache python sparse_ei_network.py --smoke --output-dir /tmp/brainx-sparse-ei-smoke-v1`
- Result: exit code 0 in 25.90 s.
- Artifacts: `/tmp/brainx-sparse-ei-smoke-v1/config.json`, `metrics.json`, `metrics.csv`, `robustness.json`, `graph-hashes.json`, `provenance.json`, and four compressed raw condition files.
- Scope: one 80-E/20-I repeat, 20 ms burn-in, 80 ms analysis. This run validates mechanics and artifact production only; its regime labels are not scientific evidence.
