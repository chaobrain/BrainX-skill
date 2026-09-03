# Brain modeling memory

## Checkpoint
- Iteration: 1
- Step: 0

### Artifacts
- `NeuroSpecification.md`: Locked model, numerical protocol, seed policy, observables, and prospective regime predicates.

### Important milestones
- Selected entry case `fresh-new`; no artifacts existed in case 19.
- Recovered the previously approved `dt`, duration, seed set, and acceptance thresholds from committed case-17 artifacts without restoring deleted worktree files.
- Specification is locked and the loop advances to step 1.

## Checkpoint
- Iteration: 1
- Step: 1

### Artifacts
- `brainx-study-record.md`: Complete point-neuron, event communication, units, State lifecycle, delay, randomness, analysis, validation, and acceleration design.

### Important milestones
- Selected BrainPy-State as the sole biological-scale owner, with BrainEvent, BrainUnit, and BrainState as supporting infrastructure.
- Fixed exact fan-in, update order, delay convention, matched seed policy, output reduction, and prospective analysis before implementation.
- Optional training/fitting coverage is none; the loop advances to step 2.

## Checkpoint
- Iteration: 1
- Step: 2

### Artifacts
- `brunel_lif_regimes.py`: BrainX-native model, compiled experiment runner, host-side analyses, compact raw outputs, provenance, manifest, and deterministic result assessment.
- `test_brunel_lif_regimes.py`: Focused model, State, RNG, delay, connectivity, metric, and contract checks.
- `test-results.md`: Nine passing focused checks on CPU.

### Important milestones
- Implemented one population-shaped point-neuron Module with exact fixed-fan-in BrainEvent communication and no dense recurrent matrix.
- Kept the complete 25,000-step rollout inside one BrainState transform and all host analysis outside it.
- Preserved the fixed 50-neuron sample, full-population ISI analysis, per-run spectra, partial progress evidence, and final artifact hashes.
- Step 2 passes and the loop advances to acceleration step 3.

## Checkpoint
- Iteration: 1
- Step: 3

### Artifacts
- `acceleration-and-parity.md`: Hot-path inventory, unchanged acceleration decision, memory rationale, benchmark, parity, and residual risks.
- `benchmark_parity.py`: Reproducible representative cold/warm benchmark.

### Important milestones
- Preserved the already-fused population/event path and complete compiled time loop.
- Measured 9.5112 s cold and 0.8086 s warm on CPU for the representative workload.
- Proved bit-identical spikes and `0.0 mV` final-voltage difference after full reset/reseed.
- Rejected condition/seed mapping because it multiplies independent writable State and full spike histories on the available 15 GiB host.
- Step 3 passes and the loop advances to experiment execution step 4.

## Checkpoint
- Iteration: 1
- Step: 4

### Artifacts
- `runs/20260829T190025+0800-smoke-brunel/`: Preserved failed smoke snapshot, complete outputs, log, mutable failed status, and process exit code.

### Important milestones
- CPU preflight passed and the smoke process exited zero with all declared files.
- Mechanical validation failed because the 80 ms analysis window produced no eligible slow-condition neurons for ISI CV, yielding non-finite metrics.
- Classified the failure as deterministic experiment-readiness failure; no unchanged retry is allowed.
- Returned to iteration 2, step 2 without changing the locked production model, seeds, duration, analysis, or predicates.

## Checkpoint
- Iteration: 2
- Step: 2

### Artifacts
- `brunel_lif_regimes.py`: Smoke-only analysis duration changed from 80 ms to 2,000 ms with a 100 ms burn-in.

### Important milestones
- Corrected only the reduced mechanical witness so low-rate neurons can satisfy the predeclared four-spike ISI eligibility rule.
- Production configuration and all scientific semantics remain unchanged; focused validation must pass again before acceleration and launch.

## Checkpoint
- Iteration: 2
- Step: 3

### Artifacts
- `test-results.md`: Iteration-2 focused suite, `9 passed in 30.76s`.
- `acceleration-and-parity.md`: Iteration-2 unchanged hot-path decision and repeated exact parity evidence.

### Important milestones
- Revalidated deterministic replay, State, units, delays, connectivity, and classifiers after the smoke-only correction.
- Repeated the representative benchmark with bit-identical spikes and `0.0 mV` final-voltage difference.
- Iteration 2 advances to experiment execution step 4 with no production acceleration change.

## Checkpoint
- Iteration: 2
- Step: 4

### Artifacts
- `runs/20260829T190025+0800-smoke-brunel/`: Preserved deterministic failed smoke with non-finite low-rate ISI CV.
- `runs/20260829T190412+0800-smoke-brunel-v2/`: Mechanically complete corrected CPU smoke with four finite rows, four raw files, and a hashed manifest.
- `runs/20260829T190714+0800-production-brunel-seeds5/`: Stopped production snapshot with exact log, exit code 130, 13 parseable raw artifacts, and 13 finite partial metric rows.

### Important milestones
- Production ran from 19:08 to the declared 23:08 wall-time boundary and was interrupted gracefully through its exact session handle.
- Completed all four conditions for seeds 1729, 2718, and 3141 plus synchronous regular for seed 5772.
- Preserved the stop as a planned resource-budget outcome, not scientific acceptance or deterministic model failure.
- Step 4 remains incomplete; iteration 3 returns to step 2 to add condition-boundary continuation without changing model dynamics, analysis, seeds, or predicates.

## Checkpoint
- Iteration: 3
- Step: 2

### Artifacts
- `brunel_lif_regimes.py`: Added validated condition-boundary continuation through `--resume-from`.
- `test_brunel_lif_regimes.py`: Added completed-row/raw-artifact resume validation and copy coverage.
- `test-results.md`: Iteration-3 suite, `10 passed in 46.27s`, plus the actual 13-artifact parent witness.

### Important milestones
- Continuation accepts only finite, unique rows matching the locked repeat seed and condition names.
- Every referenced raw artifact must parse, expose required arrays, and have a 20,000 by 50 raster before copying.
- Completed pairs are skipped; missing pairs rebuild deterministic exact-indegree graphs and execute the unchanged full simulation.

## Checkpoint
- Iteration: 3
- Step: 3

### Artifacts
- `acceleration-and-parity.md`: Continuation-only unchanged hot-path audit and repeated exact parity.
- `benchmark_parity.py`: Repeated representative benchmark with bit-identical spikes and `0.0 mV` final-voltage difference.

### Important milestones
- Preserved the original simulation, State, RNG, analysis, and memory behavior for all seven remaining conditions.
- Rejected shortening, mapping, or changing scientific parameters to recover time.
- Iteration 3 advances to a linked continuation run in step 4.

## Checkpoint
- Iteration: 3
- Step: 4

### Artifacts
- `runs/20260829T231314+0800-continuation-brunel-missing7/`: Completed linked production run with the exact config, environment, log, exit code, 20 raw artifacts, 20 finite metrics, five graph hashes, robustness assessment, result assessment, provenance, and 28-file manifest.
- `runs/20260829T190714+0800-production-brunel-seeds5/`: Preserved stopped parent supplying 13 byte-identical inherited condition artifacts.

### Important milestones
- Mechanical validation passed: 20 unique repeat-condition rows, all finite; 20 parseable raw files; five graph identities; one fixed probe sample; 13 inherited files byte-identical; 28 manifest entries.
- Robust aggregate outcomes: synchronous regular passed 5/5; fast synchronous irregular passed 5/5; requested AI failed 0/5 and classified synchronous regular; requested slow-SI failed 0/5 and remained inconclusive.
- Aggregate median dominant frequencies are 625.000, 173.340, 129.395, and 21.973 Hz in requested-condition order.
- Step 4 is complete and advances to mandatory Codex review step 5.

## Checkpoint
- Iteration: 3
- Step: 5

### Artifacts
- `review-request.txt`: Complete, path-verified review packet ready for a fresh BrainX Codex MCP call.

### Important milestones
- The current tool registry reports `NO_CODEX_MCP_TOOL`.
- Mandatory external review is unavailable; no self-review or substitute invocation was used.
- Step 5 is blocked. Visualization step 6 cannot begin until a fresh configured Codex MCP review returns `PASS`.

## Checkpoint
- Iteration: 3
- Step: 5

### Artifacts
- `reviews/iteration-3-review.md`: Preserved raw Codex review and thread ID `01a0514c-8ef3-7070-9fcd-1ffe666695c5`.

### Important milestones
- This checkpoint supersedes the earlier iteration-3 step-5 blocked record because the configured Codex MCP reviewer became available.
- Reviewer outcome is `REFUSE`; scientific outcome is `PARTIALLY_SUPPORTED`.
- Major finding `PROV-001` requires process-captured interpreter and dependency provenance; minor finding `RESUME-001` requires full continuation-contract and source-manifest validation.
- Iteration 4 returns to step 2 with the locked specification unchanged.

## Checkpoint
- Iteration: 4
- Step: 2

### Artifacts
- `brunel_lif_regimes.py`: Hardened source-manifest, condition, metric, classification, probe, raw-array, and in-process provenance validation.
- `test_brunel_lif_regimes.py`: Updated continuation fixture and rejected incorrect-condition coverage.
- `test-results.md`: Iteration-4 suite, `10 passed in 23.47s`, plus a successful 20-row/20-raw actual-source witness.
- `provenance-reconciliation.md`: Reconciles the incorrect prelaunch Python field with the absolute interpreter, its pre-run timestamp, in-process Python identity, and matching BrainX dependency tuple.

### Important milestones
- Addressed `RESUME-001`: inherited evidence is accepted only after complete locked-schema and source-manifest validation.
- Addressed `PROV-001` in code: every new result records `sys.executable`, Python implementation/version, BrainX dependency versions, backend/devices, and source-manifest SHA-256 inside the executing process.
- Model dynamics, seeds, analysis, prospective predicates, and existing raw evidence remain unchanged.
- Iteration 4 advances to acceleration and parity step 3.

## Checkpoint
- Iteration: 4
- Step: 3

### Artifacts
- `acceleration-and-parity.md`: Iteration-4 unchanged hot-path audit and exact parity evidence.
- `benchmark_parity.py`: Repeated representative benchmark with 8,625 bit-identical spikes and `0.0 mV` final-voltage difference.

### Important milestones
- Measured 10.0524 s cold and 0.9344 s warm on CPU after the host-only validation/provenance patch.
- Preserved the compiled population/event path, State and RNG lifecycle, fixed graphs, full-history analysis, and all scientific outputs.
- Iteration 4 advances to experiment execution step 4.

## Checkpoint
- Iteration: 4
- Step: 4

### Artifacts
- `runs/20260830T142816+0800-validation-continuation-brunel/`: Completed linked validation continuation with frozen contract, correct process-captured environment, exit 0, 20 rows/raw artifacts, rebuilt aggregates, and a 28-entry verified manifest.
- `provenance-reconciliation.md`: Evidence that the named Python 3.11.15 interpreter predates the source runs and its process-captured BrainX dependency tuple matches the recorded package tuple.

### Important milestones
- Hardened validation accepted all 20 source rows and raw files, each against the source manifest and locked scientific schema.
- New raw NPZ files are byte-identical to the completed source; graph hashes and robustness output are identical.
- In-process provenance records the exact interpreter, Python 3.11.15, CPython, the complete BrainX dependency tuple, CPU backend/device, and source-manifest SHA-256.
- Mechanical status is `done`; iteration 4 advances to mandatory fresh Codex review step 5.

## Checkpoint
- Iteration: 4
- Step: 5

### Artifacts
- `reviews/iteration-4-review.md`: Preserved raw Codex review and thread ID `01a0519a-9c93-7e71-969a-9410b7faf1f0`.

### Important milestones
- Reviewer outcome is `PASS` with no findings; scientific outcome is `PARTIALLY_SUPPORTED`.
- Accepted evidence robustly supports synchronous regular and fast synchronous irregular, refutes the requested asynchronous-irregular label at `(g, eta)=(5,2)`, and leaves `(4.5,0.9)` slow-SI inconclusive under frozen predicates.
- Iteration 4 advances to visualization step 6.

## Checkpoint
- Iteration: 4
- Step: 6

### Artifacts
- `skills/brainx-modeling-loop/references/visualization-workflow.md`: Required route is absent.
- `skills/brainx-visualization/`: Planned visualization skill is absent.
- `runs/20260830T142816+0800-validation-continuation-brunel/results/`: Review-passed evidence preserved and ready for visualization.

### Important milestones
- Step 6 is blocked by the modeling-loop instruction: do not embed or invent the missing visualization workflow.
- No PNG was written. The accepted run remains unchanged until the required visualization instruction exists.

## Checkpoint
- Iteration: 4
- Step: 6

### Artifacts
- `FIGURE_CONTRACT.md`: Frozen final-evidence display contract for the accepted iteration-4 run.
- `visualize_brunel_results.py`: Source-manifest-verifying renderer with source-value and PNG integrity checks.
- `figures/iteration-4-final/four-condition-rate-raster.png`: Four-condition 0.1 ms global-rate and fixed 50-neuron raster figure.
- `figures/iteration-4-final/ei-rates.png`: Five-seed paired E/I firing-rate summary.
- `figures/iteration-4-final/isi-cv.png`: Five-seed paired E/I ISI-CV summary with frozen decision boundaries.
- `figures/iteration-4-final/global-rate-spectrum.png`: Five-seed global-rate Welch spectra and aggregate dominant frequencies.
- `figures/iteration-4-final/FIGURE_MANIFEST.md`: Per-figure provenance, transformations, hashes, dimensions, and render checks.
- `figures/iteration-4-final/render-checks.json`: Machine-readable nonblank-image checks and output hashes.

### Important milestones
- This checkpoint supersedes the earlier iteration-4 step-6 blocked record because `skills/brainx-visualization/SKILL.md` and its required references are now present.
- Verified every accepted result artifact against its run manifest before rendering and cross-checked plotted rates and spectral peaks against the accepted metrics.
- Rendered and visually inspected all four requested PNGs without changing the accepted simulation, scientific metrics, or regime predicates.
- Completed the modeling loop. Final evidence supports synchronous regular and fast synchronous irregular, refutes the requested asynchronous-irregular assignment, and leaves the requested slow synchronous irregular assignment inconclusive.
