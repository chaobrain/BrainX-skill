# Brain modeling memory

## Checkpoint
- Iteration: 1
- Step: 0

### Artifacts
- `NeuroSpecification.md`: Fresh specification draft created only from the researcher request; no prior cases, papers, or open-source implementations were consulted.

### Important milestones
- Selected entry case `fresh-new` in a separate run directory.
- Step 0 is blocked pending researcher decisions on duration, transient, and fixed seeds; the specification remains draft.

## Checkpoint
- Iteration: 1
- Step: 0

### Artifacts
- `NeuroSpecification.md`: Locked specification with approved duration, transient, fixed seeds, shared random structures, initialization, timestep, and qualitative decision boundaries.

### Important milestones
- Researcher approved the proposed experiment contract on 2026-09-03.
- Step 0 completed; proceed to step 1 with forward-simulation coverage and no training or fitting workflow.

## Checkpoint
- Iteration: 1
- Step: 1

### Artifacts
- `BrainXStudy.md`: Completed point-neuron implementation design, lifecycle, API choices, randomness contract, online metrics, and frozen regime criteria.
- `/home/yixinliu/anaconda3/envs/braincell-released`: Verified coherent BrainX 2026.7.9 environment on CPU.

### Important milestones
- Selected BrainPy-State with BrainEvent, BrainUnit, and BrainState support; no other modeling scale is represented.
- Honored the researcher constraint by not opening prior case artifacts, papers, external source code, or bundled example scripts.
- Runtime API probes verified unit-bearing `FixedNumPerPost` products, `DeltaProj` integration order, and deterministic BrainState Poisson replay.
- Step 1 completed; proceed to step 2 implementation.

## Checkpoint
- Iteration: 1
- Step: 2

### Artifacts
- `lif_network.py`: BrainX-native current LIF model, exact fixed-fan-in construction, compiled rollout, online observables, spectral analysis, classification, and artifact writer.
- `test_lif_network.py`: Focused unit, connectivity, delay, refractory, reset, replay, shape, and contract checks.
- `test-results-step2.xml`: Passing report for 7 focused tests.
- `config.json`: Frozen approved production configuration and analysis thresholds.
- `runs/smoke-step2/`: Completed reduced implementation smoke artifacts.

### Important milestones
- Selected explicit hard reset after the refractory test exposed the native soft-reset default as incompatible with the specified `V_reset = 10 mV` rule.
- Avoided a full time-by-neuron spike allocation by accumulating each neuron's ISI moments online.
- Reduced only the smoke Welch window to fit its short diagnostic trace; production retains the frozen spectrum contract.
- Step 2 completed; proceed to step 3 acceleration and parity.

## Checkpoint
- Iteration: 1
- Step: 3

### Artifacts
- `acceleration_audit.md`: Hot-path inventory, rewrite decision, validation plan, measured outcome, and remaining risks.
- `benchmark_acceleration.py`: Representative cold/warm/replay benchmark.
- `acceleration_parity.json`: Exact output and final-voltage replay with cold and warm timings.
- `benchmark_connectivity.py`: Fixed-fan-in versus equivalent CSR event-product benchmark.
- `connectivity_benchmark.json`: Exact count parity and density-dependent timing evidence.
- `benchmark_full_scale.py`: Full production-shape short benchmark.
- `full_scale_benchmark.json`: Full-shape cold/warm timing and production projection.

### Important milestones
- Retained the existing `jit(for_loop)` hot path, structured fixed-fan-in matrices, online ISI moments, and sampled spike history after the audit found no safe higher-impact rewrite.
- Rejected CSR conversion because it preserved results but gave no consistent material speedup.
- Exact reset/reseed replay preserved all outputs and final voltage.
- Selected three concurrent seed processes on disjoint CPU core sets for execution-level wall-time reduction without changing any scientific result.
- Step 3 completed; proceed to step 4 experiment execution.

## Checkpoint
- Iteration: 1
- Step: 4

### Artifacts
- `runs/smoke-cpu-20260903/`: Immutable CPU smoke snapshot, log, exit code, finite raw output, and metrics; status `done`.
- `runs/production-seed-11-cpu-20260903/`: Immutable seed-11 production snapshot with four completed conditions; status `done`, exit code 0.
- `runs/production-seed-29-cpu-20260903/`: Immutable seed-29 production snapshot with four completed conditions; status `done`, exit code 0.
- `runs/production-seed-47-cpu-20260903/`: Immutable seed-47 production snapshot with four completed conditions; status `done`, exit code 0.
- `runs/production-combined-20260903/mechanical_validation.json`: Passing shape, parse, finiteness, and hash checks for all twelve raw runs.
- `runs/production-combined-20260903/artifact_manifest.json`: File sizes and SHA-256 hashes for immutable run evidence.
- `runs/production-combined-20260903/condition_assessment.json`: Frozen three-seed classifications and underlying metrics.
- `runs/production-combined-20260903/RESULT_ASSESSMENT.md`: Deterministic assessment and claim-evidence matrix.

### Important milestones
- All twelve production rollouts completed on CPU with exit code 0; no retry or parameter change occurred.
- `(3, 2)` verified synchronous regular at a median dominant frequency of 333 Hz and pooled median ISI CV 0.000.
- `(6, 4)` measured synchronous with a fast 184 Hz peak but regularity was indeterminate at pooled median ISI CV 0.798, so fast synchronous irregular was not verified.
- `(5, 2)` measured synchronous regular at 121 Hz and pooled median ISI CV 0.410, so asynchronous irregular was not verified.
- `(4.5, 0.9)` measured synchronous with a slow 22 Hz peak but regularity was indeterminate at pooled median ISI CV 0.679, so slow synchronous irregular was not verified.
- Step 4 completed mechanically; proceed to step 5 Codex review without visualizing yet.

## Checkpoint
- Iteration: 1
- Step: 5

### Artifacts
- `reviews/iteration-1.md`: Verbatim Codex reviewer response.
- `reviews/iteration-1-thread.txt`: Reviewer thread ID `01a065af-042d-7d42-af99-240b0ad3bc66`.

### Important milestones
- Codex outcome `REFUSE`, scientific outcome `PARTIALLY_SUPPORTED`, next action `RETURN_TO_STUDY`.
- Required corrections: BrainTools API-gap evidence; native BrainState Delay replacement with exact parity; Poisson/signed-delta/reference-transition controls; production final-membrane State finiteness evidence.
- Increment to iteration 2 and return to study of the affected API routes before implementation.

## Checkpoint
- Iteration: 2
- Step: 1

### Artifacts
- `BrainXStudy-iteration2.md`: Finding-by-finding restudy and corrected implementation/evidence design.
- `BrainToolsAPIGap.md`: Exact checked APIs, missing capabilities, minimum external boundaries, and required parity evidence.

### Important milestones
- Selected native `brainstate.nn.Delay` with a documented previous-spike insertion convention and step-14 retrieval that delivers the event exactly 15 simulation steps after emission.
- Retained NumPy topology sampling and SciPy Welch only as documented minimum host boundaries under the locked exact topology and spectrum contracts.
- Required full iteration-2 production re-execution because final membrane State evidence cannot be recovered from iteration 1.
- Iteration 2 study completed; proceed to step 2 corrections.

## Checkpoint
- Iteration: 2
- Step: 2

### Artifacts
- `lif_network.py`: Replaced manual ring State with `brainstate.nn.Delay`; added per-run complete final-voltage vector, hash, range, and finiteness evidence.
- `test_lif_network.py`: Expanded to 10 passing controls covering Poisson moments, signed delta summation, hand-computed leak, native delay timing, threshold/reset lifecycle, refractory exclusion, replay, units, and topology.
- `test-results-iteration2.xml`: Passing iteration-2 test report.
- `runs/smoke-iteration2-delay/`: Corrected reduced smoke output with final-voltage evidence.
- `iteration2_control_evidence.json`: Exact old/new smoke parity and fixed Poisson snapshot statistics/hashes.

### Important milestones
- Native BrainState Delay produced exact equality with every preserved iteration-1 smoke observable and analysis array.
- Confirmed `LIFRef` exposes the threshold-crossing voltage on the spike update and commits hard reset on the following refractory update; the control verifies reset before subsequent integration.
- Complete final membrane State is now a required raw and metric artifact and causes immediate failure if non-finite.
- Iteration 2 step 2 completed; proceed to fresh acceleration/parity evidence.

## Checkpoint
- Iteration: 2
- Step: 3

### Artifacts
- `acceleration_audit-iteration2.md`: Corrected hot-path audit, validation, unchanged decision, and remaining risks.
- `acceleration_parity-iteration2.json`: Representative cold/warm timings with exact output and final-voltage replay.
- `full_scale_benchmark-iteration2.json`: Native-delay full-scale timing and production projection.
- `iteration2_control_evidence.json`: Exact migration parity supporting the acceleration decision.

### Important milestones
- Native Delay preserved exact smoke outputs and representative reset/reseed outputs plus final voltage.
- Full-scale warm projection is 1,415.2 seconds per 5-second condition on the unrestricted CPU benchmark.
- Retained concurrent seed-level execution on disjoint CPU core sets; no model or analysis change was made in acceleration.
- Iteration 2 step 3 completed; proceed to new immutable production execution.

## Checkpoint
- Iteration: 2
- Step: 4

### Artifacts
- `runs/production-v2-seed-11-cpu-20260903/`: Immutable seed-11 iteration-2 production snapshot with four conditions; status `done`, exit code 0.
- `runs/production-v2-seed-29-cpu-20260903/`: Immutable seed-29 iteration-2 production snapshot with four conditions; status `done`, exit code 0.
- `runs/production-v2-seed-47-cpu-20260903/`: Immutable seed-47 iteration-2 production snapshot with four conditions; status `done`, exit code 0.
- `runs/production-v2-combined-20260903/mechanical_validation.json`: Passing shape, finiteness, raw hash, and complete final-voltage hash checks for all twelve runs.
- `runs/production-v2-combined-20260903/braintools_psd_parity.json`: BrainTools/SciPy frequency-grid, dominant-peak, significance, and final-classification parity evidence.
- `runs/production-v2-combined-20260903/condition_assessment.json`: Frozen iteration-2 three-seed classifications and metrics.
- `runs/production-v2-combined-20260903/RESULT_ASSESSMENT.md`: Deterministic iteration-2 claim-evidence matrix.

### Important milestones
- All twelve corrected production rollouts completed on CPU with exit code 0; no retry, parameter change, or shortened protocol occurred.
- Every complete 12,500-neuron final membrane vector is finite and matches the SHA-256 stored in its metric record.
- BrainTools and frozen SciPy spectra have exact frequency-grid, dominant-frequency, significance-predicate, and final-classification parity across all twelve traces; numerical PSD differences remain recorded.
- Iteration-2 results reproduce the iteration-1 scientific metrics: `(3, 2)` verifies synchronous regular at 333 Hz; the 184 Hz, 121 Hz, and 22 Hz conditions do not verify their requested full labels under the frozen ISI-CV and synchrony rules.
- Iteration 2 step 4 completed; proceed to a fresh Codex review without visualizing yet.

## Checkpoint
- Iteration: 2
- Step: 5

### Artifacts
- `reviews/iteration-2.md`: Verbatim fresh Codex reviewer response.
- `reviews/iteration-2-thread.txt`: Reviewer thread ID `01a06648-dc4f-7903-ae71-16473eb2b5c9`.
- `BrainXStudy-iteration2.md`: Corrected the reviewer-noted minor Poisson control sample-size provenance mismatch from one million to the executed 250,000 samples.

### Important milestones
- Codex outcome `PASS`, scientific outcome `PARTIALLY_SUPPORTED`, next action `ADVANCE_TO_VISUALIZATION`.
- All iteration-1 critical and major findings are closed; no critical or major iteration-2 defect remains.
- The sole minor documentation mismatch was corrected without changing code, production results, or frozen analysis criteria.
- Iteration 2 step 5 completed; proceed to visualization of the accepted measured results.

## Checkpoint
- Iteration: 2
- Step: 6

### Artifacts
- `FIGURE_CONTRACT.md`: Frozen final-figure questions, sources, transformations, sampling units, encodings, and comparison settings.
- `visualize_results.py`: Reproducible BrainTools/Matplotlib renderer and validation writer.
- `figures/four_condition_rate_raster.png`: Four accepted conditions with unsmoothed 0.1 ms global rate above the fixed seed-11 50-neuron raster.
- `figures/ei_rates.png`: E/I post-transient mean rates with all three seed values and cross-seed means.
- `figures/isi_cv.png`: Pooled valid per-neuron ISI-CV distributions, three seed medians, and frozen regularity boundaries.
- `figures/global_rate_spectrum.png`: Individual and cross-seed mean Welch spectra with accepted median dominant frequencies.
- `FIGURE_MANIFEST.md`: Source hashes, meanings, transformations, sample sizes, output dimensions, and PNG hashes.
- `figure_validation.json`: Exact plotted source-value checks and finite/nonblank pixel evidence.

### Important milestones
- Used accepted iteration-2 evidence only and preserved measured labels where three requested regime names were not verified.
- Visual inspection found every PNG nonblank, legible, unclipped, overlap-free, and consistent with accepted metrics.
- The final verification reran all 10 model/control tests successfully and compiled the simulation, collector, and visualization scripts.
- Iteration 2 step 6 completed; the modeling, review, and visualization workflow is complete.
