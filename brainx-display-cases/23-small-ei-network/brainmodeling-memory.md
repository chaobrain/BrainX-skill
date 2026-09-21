# Brain modeling memory

## Checkpoint
- Iteration: 1
- Step: 0

### Artifacts
- `NeuroSpecification.md`: locked compact specification for a 20-neuron point-neuron E/I teaching network.
- `brainx-study-record.md`: BrainX API and lifecycle decisions used for implementation.

### Important milestones
- The request did not specify a split, neuron family, synapse model, drive, or duration. I selected a reversible teaching-model default: 16 E + 4 I, LIFRef, exponential COBA synapses, 20 mA drive, 0.1 ms step, 100 ms run.
- Literature gate skipped: no biological claim or mechanism comparison is requested; the model is explicitly labelled teaching-only.

## Checkpoint
- Iteration: 1
- Step: 1

### Artifacts
- `brainx-study-record.md`: completed study of BrainPy-State, BrainState, BrainUnit, and the canonical E/I scripts.

### Important milestones
- Use `LIFRef`, `AlignPostProj`, `Expon.desc`, `COBA.desc`, `brainstate.nn.init_all_states`, and `brainstate.transform.for_loop`.
- Feed the previous step's spikes to both projections, then integrate the postsynaptic population in the same update call.

## Checkpoint
- Iteration: 1
- Step: 2

### Artifacts
- `small_ei_network.py`: BrainX-native implementation.
- `test_small_ei_network.py`: structural and short-rollout checks.
- `test-results.md`: syntax check passed; runtime tests skipped because BrainPy is absent.

### Important milestones
- The implementation preserves units, registered Module State, projection-before-neuron update order, and a transformed time loop.

## Checkpoint
- Iteration: 1
- Step: 3

### Artifacts
- `acceleration-and-parity.md`: acceleration audit.

### Important milestones
- No additional acceleration was needed; the canonical vectorized BrainPy-State path is already used.

## Checkpoint
- Iteration: 1
- Step: 4

### Artifacts
- `result-assessment.md`: runtime result status.

### Important milestones
- Runtime execution is pending installation of the matched BrainX packages. The code and structural validation are complete, but no numerical result is reported.

## Checkpoint
- Iteration: 1
- Step: 5

### Artifacts
- `reviews/iteration-1.md`: verbatim Codex review report.

### Important milestones
- Review outcome: `REFUSE`, scientific outcome `INCONCLUSIVE`.
- Finding `RUNTIME-001`: install matched BrainX packages and run the full simulation/tests before claiming a completed build-and-simulate result.
- Reviewer thread: `01a0c33a-cbe1-7a53-85a6-2dc8c2c63626`.

## Checkpoint
- Iteration: 1
- Step: 5

### Artifacts
- `test-results.md`: appended runtime rerun with `2 passed` and full-rollout checks.
- `result-assessment.md`: appended runtime assessment.
- `outputs/small_ei_network_run.npz`: full raw time-major spike array.
- `outputs/small_ei_network_metadata.json`: protocol, shape, event-count, and provenance summary.
- `outputs/small_ei_network_raster.png`: diagnostic raster rendered from the raw events.
- `FIGURE_MANIFEST.md`: diagnostic figure contract and source-value checks.
- `report.md`: provisional status report; it is not final because a fresh Codex review is pending.

### Important milestones
- This checkpoint supersedes the earlier step-5 refusal record for the runtime gap: the matched environment now runs the 100 ms protocol and the saved output passes the declared shape and event checks.
- Diagnostic closure completed before the next review: report and figure manifest exist, and the raster is explicitly labeled unaccepted.
- Next action: send the complete runtime and diagnostic artifact packet to a fresh step-5 Codex review; do not mark the study final before `PASS` and step 6.
