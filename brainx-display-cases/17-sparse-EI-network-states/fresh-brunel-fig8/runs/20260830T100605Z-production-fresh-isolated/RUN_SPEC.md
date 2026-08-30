# Production run specification

- Purpose: paper-scale forward simulation of Brunel (2000), Fig. 8 under the locked prospective specification.
- Model: Model A with `NE=10000`, `NI=2500`, `CE=1000`, `CI=250`, `Cext=1000`, `tau_m=20 ms`, `tau_ref=2 ms`, `J=0.1 mV`, threshold `20 mV`, reset `10 mV`, and delay `1.5 ms`.
- Protocol: `dt=0.1 ms`, `500 ms` burn-in, `2,000 ms` analysis, shared fixed graph/probes, independent reset/reseed per condition, seed `17729`.
- Conditions: A `(g,eta)=(3,2)`, B `(6,4)`, C `(5,2)`, D `(4.5,0.9)`.
- Acceptance: apply only the predicates locked in `NeuroSpecification.md`; report every failure without tuning.
- Required outputs: four raw NPZ panel records, JSON/CSV metrics, graph/probe hashes, runtime provenance, deterministic assessment, and exact manifests.
- Rendering: forbidden before an independent read-only Codex review passes; this run produces no figure.
