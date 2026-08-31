# Brain modeling memory

## Checkpoint
- Iteration: 1
- Step: 0

### Artifacts
- `request.md`: Literal researcher request establishing the fresh-project boundary.
- `NeuroSpecification.md`: Locked paper model, protocol, observables, controls, and prospective acceptance criteria.

### Important milestones
- Selected entry case `fresh-new` inside an isolated addition-only subproject.
- The subproject began with only the request and locked specification; no prior memory, implementation, result, or figure was read or copied into it.
- Fixed all paper parameters, duration, seed policy, and acceptance predicates before implementation or simulation.
- The loop advances to step 1.

## Checkpoint
- Iteration: 1
- Step: 1

### Artifacts
- `brainx-study-record.md`: Fresh package/API study and paper-to-BrainX design.

### Important milestones
- Selected BrainPy-State as the sole biological-scale owner; the custom exact jump neuron will subclass `brainpy.state.Neuron`.
- Selected BrainEvent fixed fan-in, BrainState State/delay/RNG/transforms, and BrainUnit quantities as supporting infrastructure.
- Optional training/fitting coverage is none.
- The loop advances to step 2.

## Checkpoint
- Iteration: 1
- Step: 2

### Artifacts
- `brunel_fig8.py`: Addition-only BrainPy-State model, BrainEvent network, compiled runner, fixed analyses, result/run manifests, deterministic assessment, and review-gated renderer.
- `test_brunel_fig8.py`: Ten focused scientific and execution checks.
- `test-results.md`: Ten passing CPU checks.

### Important milestones
- Implemented the custom exact jump neuron as `brainpy.state.Neuron`.
- Acceptance uses only predicates explicitly locked in `NeuroSpecification.md`; 1 ms rate CV is descriptive.
- Pre-review assessment states that no image exists and makes no figure-hash claim.
- Run manifests bind immutable source/contract/command/environment files to result hashes.
- Step 2 passes and the loop advances to acceleration step 3.

## Checkpoint
- Iteration: 1
- Step: 3

### Artifacts
- `benchmark_parity.py`: Independent eager/JIT benchmark using the complete network update path.
- `acceleration-and-parity.md`: Frozen CPU timing and exact parity evidence.

### Important milestones
- Kept the complete State-driven time loop under one `brainstate.transform.jit` and `brainstate.transform.for_loop` boundary.
- Confirmed bit-identical spike histories and zero final-voltage difference between eager and compiled execution.
- Measured 13.8570 s cold and 1.1607 s warm execution for 1,000 neurons over 1,000 steps.
- No model or protocol parameter changed during acceleration; the loop advances to smoke execution step 4.

## Checkpoint
- Iteration: 1
- Step: 4

### Artifacts
- `runs/20260830T100015Z-smoke-fresh-isolated/`: Immutable reduced run snapshot, raw data, metrics, assessment, provenance, and manifests.

### Important milestones
- Completed all four reduced conditions with exit code 0 and no image generation.
- Parsed four raw NPZ files and confirmed all required metrics are finite.
- Recomputed every hash in both manifests; 17 outer files and 10 result files match exactly.
- Treated reduced-network scientific verdicts as diagnostic only, as declared prospectively in `RUN_SPEC.md`.
- The mechanical gate passes and the loop advances to paper-scale production execution.

## Checkpoint
- Iteration: 1
- Step: 4

### Artifacts
- `runs/20260830T100605Z-production-fresh-isolated/`: Sealed paper-scale run contract, raw records, metrics, provenance, assessment, and manifests.

### Important milestones
- Completed all four full-scale conditions with exit code 0 and no image generation.
- Panel A passed its locked regularity predicate; panel B passed every locked rate, irregularity, and frequency predicate.
- Panel C matched the paper rate but failed only the locked ISI-CV predicate; panel D matched the rate and rhythm tolerances but failed only the locked ISI-CV predicate.
- Parsed all raw files, confirmed finite metrics and expected shapes, and recomputed all 7 source hashes, 17 outer-manifest hashes, and 10 result-manifest hashes exactly.
- The production run is immutable and advances to independent read-only review step 5.

## Checkpoint
- Iteration: 1
- Step: 5

### Artifacts
- `review-request-iteration-1.md`: Explicit read-only gate packet naming the fresh-boundary, scientific, lifecycle, and artifact-integrity checks.
- `reviews/iteration-1-review.md`: Independent Codex `PASS` with scientific outcome `PARTIALLY_SUPPORTED`.

### Important milestones
- The reviewer independently inspected source, tests, raw outputs, metrics, assessments, and manifests.
- It confirmed that A/B pass and C/D fail only their prospective ISI-CV predicates, with those negative results represented honestly.
- It authorized visualization from the accepted production raw data.
- One minor documentation issue remains: the timing benchmark itself is cold/warm JIT replay, while true eager/JIT parity is established by the focused test. This does not invalidate the run and the source-bound evidence remains unchanged.
- The loop advances to post-review visualization step 6.

## Checkpoint
- Iteration: 1
- Step: 6

### Artifacts
- `figures/brunel_fig8_reproduction.png`: Post-review four-condition raster and instantaneous-global-rate reproduction.
- `figures/figure-manifest.json`: Review gate, rendering command, immutable raw-input hashes, output hash, dimensions, and pixel checks.

### Important milestones
- Rendered only after the independent review returned `PASS` and `ADVANCE_TO_VISUALIZATION`.
- Rendered directly from the accepted production NPZ files without rerunning or modifying the simulation.
- Visually inspected all four raster/activity pairs at original resolution.
- Verified a 3,300 x 2,400 nonblank image, 9.61% nonwhite pixels, and SHA-256 `3285b03609cb8e1db213711ce344a1b26b748816f9c1980f2b13201d43fb89ad`.
- The fresh modeling loop is complete.
