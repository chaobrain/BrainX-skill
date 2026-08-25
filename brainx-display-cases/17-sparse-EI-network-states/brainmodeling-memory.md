# Brain modeling memory

## Checkpoint
- Iteration: 1
- Step: 0

### Artifacts
- `prompt.md`: researcher request, verified present.
- `NeuroSpecification.md`: prospective draft awaiting researcher approval.

### Important milestones
- Fresh-new entry selected because the case contained only the prompt and no prior loop artifacts.
- Canonical Brunel model-A parameters and the four Figure 8 parameter points were verified against the source paper.
- Step 0 is blocked pending approval of the proposed 500 ms burn-in, 2,000 ms analysis, five fixed-seed repeats, and prospective regime predicates.

## Checkpoint
- Iteration: 1
- Step: 0

### Artifacts
- `NeuroSpecification.md`: locked prospective specification.

### Important milestones
- Researcher approved the proposed numerical protocol and regime predicates on 2026-08-24 without changes; this checkpoint supersedes the earlier blocked step-0 milestone.
- Step 0 is complete; continue to step 1.

## Checkpoint
- Iteration: 1
- Step: 1

### Artifacts
- `brainx-study-record.md`: selected BrainX abstractions, source-model contract, update order, randomness, fixed metrics, checks, and implementation design.

### Important milestones
- Selected scale: point-neuron spiking network, owned by BrainPy-State; BrainEvent, BrainUnit, and BrainState provide fixed-fan-in communication, units, delay/random State, and compiled execution.
- Active optional coverage: none (forward simulation).
- All task-relevant routed references and canonical scripts were traced before implementation.
- Step 1 is complete; continue to step 2.

## Checkpoint
- Iteration: 1
- Step: 2

### Artifacts
- `sparse_ei_network.py`: BrainPy-State LIF model, BrainEvent fixed-fan-in communication, BrainState delays/randomness/compiled rollout, analysis, classification, and artifact writer.
- `test_sparse_ei_network.py`: eight focused implementation and lifecycle checks.
- `test-results.md`: passing focused suite and reduced end-to-end smoke record.
- `/tmp/brainx-sparse-ei-smoke-v1`: mechanically complete reduced smoke artifacts; excluded from scientific claims.

### Important milestones
- Focused suite passed: 8 tests in 23.12 s on CPU.
- Reduced four-condition smoke completed with exit code 0 and emitted configuration, topology hashes, per-repeat metrics, robustness summary, provenance, raster probes, rates, and spectra.
- BrainEvent fixed-degree topology must be attached to the Module before tracing; each repeat compiles one graph-specific runner reused across all four dynamic `(g, eta)` conditions.
- Step 2 is complete; continue to step 3.

## Checkpoint
- Iteration: 1
- Step: 3

### Artifacts
- `acceleration-and-parity.md`: hot-path inventory, unchanged acceleration decision, deterministic parity, cold/warm timing, memory accounting, and remaining risks.

### Important milestones
- The accepted acceleration path is one graph-specific `brainstate.transform.jit(for_loop)` per repeat with population-wide LIF State and BrainEvent fixed-degree communication.
- A 1,000-neuron, 100 ms benchmark took 12.3306 s cold and 0.9256 s warm on CPU; replay produced bit-identical spikes and `0.0 mV` final-voltage difference.
- Batching conditions/repeats and generated-probability connectivity were rejected because they increase full-history memory or violate exact-indegree semantics.
- Step 3 is complete; continue to step 4.

## Checkpoint
- Iteration: 1
- Step: 2

### Artifacts
- `sparse_ei_network.py`: adds an exact immutable production-config loader without changing model or analysis semantics.
- `test_sparse_ei_network.py`: adds a locked-config round-trip check.
- `test-results.md`: superseding focused result, 9 tests passed in 26.87 s.

### Important milestones
- This execution-plumbing correction supersedes the earlier eight-test count; model, approved parameters, metrics, thresholds, and acceleration design are unchanged.
