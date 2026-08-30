# Iteration 1 Codex review

- Thread ID: `01a0520b-3931-78c0-9027-d746bfb868a7`

```text
OUTCOME: REFUSE
SCIENTIFIC_OUTCOME: INVALID
LOSS_CLOSURE: NOT_APPLICABLE
OPTIMIZATION_ADEQUACY: NOT_APPLICABLE
GOOD_ENOUGH_REASON: Raw metrics recompute exactly and support A/B while rejecting C/D, but the iteration violates the locked fresh-project condition and bypasses the required BrainPy-State modeling owner.
FINDINGS:
- ID: FRESH-001
  SEVERITY: critical
  PROBLEM: The tracked case-17 diff cannot prove that the project was created without prior reuse.
  MINIMUM_FIX: Rerun from an isolated clean workspace containing only the request and approved specification.
- ID: BRAINPY-001
  SEVERITY: major
  PROBLEM: The custom neuron subclasses brainstate.nn.Module instead of the selected BrainPy-State owner.
  MINIMUM_FIX: Express the exact custom jump dynamics through brainpy.state.Neuron or a validated native composition.
- ID: ASSESS-001
  SEVERITY: minor
  PROBLEM: A/C verdict logic uses 1 ms rate-CV thresholds absent from the locked acceptance boundary.
  MINIMUM_FIX: Remove those thresholds or obtain prospective approval.
- ID: REPORT-001
  SEVERITY: minor
  PROBLEM: The pre-figure assessment claims figure hashes before a figure exists.
  MINIMUM_FIX: Remove the figure claim until rendering and hashing occur.
UNVERIFIED_ASSUMPTIONS:
- Results are not cryptographically bound to the outer run contract/code diff/command.
- Author RNG, initialization, and exact duration are unavailable.
NEXT_ACTION: RETURN_TO_IMPLEMENTATION
```
