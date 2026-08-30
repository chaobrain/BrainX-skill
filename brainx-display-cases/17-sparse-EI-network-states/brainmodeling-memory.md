# Brain modeling memory

## Checkpoint
- Iteration: 1
- Step: 0

### Artifacts
- `NeuroSpecification.md`: Locked fresh-project scientific model, Fig. 8 protocol, observables, controls, and prospective acceptance boundary.

### Important milestones
- Selected entry case `fresh-new` under the researcher's explicit instruction not to read prior memory.
- Did not read or reuse any prior case-17 or case-19 memory, code, results, or figures.
- Fixed the paper-derived parameters and acceptance tests before observing any new simulation result.
- The loop advances to step 1.

## Checkpoint
- Iteration: 1
- Step: 1

### Artifacts
- `brainx-study-record.md`: Complete paper-to-BrainX translation, package/API selection, update order, lifecycle, output reduction, tests, and implementation boundary.

### Important milestones
- Selected BrainPy-State as the sole biological-scale owner with BrainEvent, BrainState, and BrainUnit support.
- Chose explicit exact fixed fan-in, one 15-step recurrent delay, independent per-neuron aggregate external Poisson input, and one compiled State-aware time loop.
- Optional training/fitting coverage is none.
- Finished study before implementation; the loop advances to step 2.

## Checkpoint
- Iteration: 1
- Step: 2

### Artifacts
- `sparse_ei_network.py`: Fresh BrainX-native Brunel Model A network, exact fixed-fan-in graph, compiled experiment runner, host analyses, compact raw artifacts, deterministic assessment, provenance, manifest, and review-gated Fig. 8 renderer.
- `test_sparse_ei_network.py`: Focused scientific and execution contract tests.
- `test-results.md`: Nine passing CPU implementation checks.

### Important milestones
- Implemented the four paper parameter points without reading or reusing prior project code or results.
- Kept the complete timestep sequence in one State-aware compiled loop and reduced results only at the explicit host boundary.
- Verified exact replay and eager/JIT parity after complete State and RNG reset.
- Step 2 passes and the loop advances to acceleration step 3.

## Checkpoint
- Iteration: 1
- Step: 3

### Artifacts
- `acceleration-and-parity.md`: Hot-path inventory, unchanged acceleration decision, benchmark, exact parity, memory rationale, and remaining risks.
- `benchmark_parity.py`: Reproducible cold/warm reset-parity benchmark.

### Important milestones
- Preserved the already-fused population/event path and one complete compiled time loop.
- Measured 25.5102 s cold and 2.5236 s warm on CPU for the representative workload.
- Proved bit-identical spike histories and `0.0 mV` final-voltage difference after complete reset/reseed.
- Rejected panel mapping because it multiplies independent writable State and 312.5 MB histories.
- Step 3 passes and the loop advances to experiment execution step 4.

## Checkpoint
- Iteration: 1
- Step: 4

### Artifacts
- `runs/20260830T162520+0800-smoke-brunel-fresh/`: Mechanically complete reduced CPU smoke with immutable contract, four finite rows, four raw files, in-process provenance, and verified manifest.
- `runs/20260830T163045+0800-production-brunel-fresh/`: Mechanically complete full paper-scale production run with immutable contract, exact graph/probe hashes, four raw panels, metrics, provenance, deterministic assessment, and verified manifest.

### Important milestones
- Production completed on CPU in about 67.6 minutes with exit code zero and all declared artifacts.
- Panel A reproduced synchronous regular activity; panel B reproduced fast synchronous irregular activity with `60.550 Hz` firing and `173.340 Hz` global frequency.
- Panel C matched the paper's mean firing rate (`37.898 Hz`) but failed the locked AI irregularity and stationarity predicates.
- Panel D matched the paper's low firing rate and slow frequency (`5.838 Hz`, `19.531 Hz`) but failed the locked `ISI CV >= 0.7` predicate.
- Preserved all unfavorable findings without parameter or threshold tuning.
- Step 4 is complete and advances to mandatory Codex review step 5.

## Checkpoint
- Iteration: 1
- Step: 5

### Artifacts
- `reviews/iteration-1-review.md`: Preserved raw refusal summary and thread ID `01a0520b-3931-78c0-9027-d746bfb868a7`.

### Important milestones
- Reviewer outcome is `REFUSE`; scientific outcome is `INVALID` under the locked no-reuse condition.
- Critical `FRESH-001` requires an isolated clean addition-only workspace; major `BRAINPY-001` requires the BrainPy-State neuron owner.
- Minor `ASSESS-001` removes undeclared A/C rate-CV verdict thresholds; minor `REPORT-001` removes the premature figure-provenance claim.
- The corrective run will be a genuinely isolated fresh project under `fresh-brunel-fig8/`, still inside the researcher-selected case-17 directory.
