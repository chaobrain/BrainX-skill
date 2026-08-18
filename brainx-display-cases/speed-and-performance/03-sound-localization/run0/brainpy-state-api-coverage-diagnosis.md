# BrainX diagnosis: sound localization from timing

## Evidence studied

Generated artifacts:

- `sound_localization.py`, `test_sound_localization.py`, and `README.md`.
- `agent-final.md`, `codex-events.jsonl`, `codex-stderr.log`, and
  `harness-metadata.txt`.
- The entry point and all six tests, rerun without modifying the archived files
  under `/Users/nijiachen/Downloads/Brainx testing/.venv-brainx`.
- A reviewer-only off-grid probe at ITDs `[-0.33, -0.15, -0.05, 0.05,
  0.15, 0.33] ms`.

Owning skills and references:

- `skills/brainx-general-guard/SKILL.md`.
- `skills/brainpy-state/SKILL.md`, `references/component-selection.md`,
  `references/projection-patterns.md`, and
  `references/brain-dynamics-delay-protocol.md`.
- `skills/brainevent/SKILL.md` and
  `references/scripts/coba_ei_teaching.py`.
- `skills/brainstate/SKILL.md`,
  `references/brainstate/transformation-vmap-expansion.md`,
  `references/collective_model_operations.md`,
  `references/state_collections_and_utilities.md`, and
  `references/brainstate/brainstate-control-flow-patterns.md`.
- `skills/brainunit/SKILL.md`.

Closest executable examples:

- `skills/brainpy-state/references/scripts/103_COBA_2005.py` for `LIFRef`,
  projection ordering, initialization, and `for_loop`.
- `skills/brainpy-state/references/scripts/109_fast_global_oscillation.py` for
  `brainstate.nn.Delay`, integer-step retrieval, `DeltaProj`, and the time loop.
- `skills/brainevent/references/scripts/coba_ei_teaching.py` for
  `BinaryArray @ FixedNumPerPre`, unit-aware event weights, initialization,
  JIT, and monitoring.
- `skills/brainevent/references/scripts/204_joglekar_2018_propagation.py` for
  the transferable delayed-event and mapped-connectivity pattern; its legacy
  BrainPy mechanics were not used as the current API standard.

Authoritative API and tutorial pages:

- `brainpy.state.LIFRef` generated API.
- BrainPy-State synaptic-delay how-to and `AlignPostProj` generated API.
- BrainState Delay Protocol tutorial and generated `brainstate.nn.Delay` API.
- Generated `brainstate.transform.vmap2`,
  `brainstate.nn.vmap_init_all_states`, and
  `brainstate.transform.for_loop` APIs.
- Generated `brainstate.util.OfType` and `brainstate.util.Any` APIs.
- Generated `brainevent.BinaryArray` and `brainevent.FixedNumPerPre` APIs.
- Generated `brainunit.math.asarray` and `brainunit.math.arange` APIs.
- Generated `braintools.init.Constant` API.

## Executive diagnosis

Run 0 is a compact, executable Jeffress-style teaching circuit. The archived
entry point prints the expected `RIGHT ... CENTER ... LEFT` sweep, every
preferred ITD activates its matching detector, all six tests pass, units remain
attached through the simulation, and both neural communication stages use
BrainEvent correctly.

The largest scientific gap is hidden by circular validation: the tested ITDs
are exactly the detector preferences. Every reviewer-only off-grid ITD produced
zero detector and readout spikes and was decoded as `CENTER`, including clearly
leftward and rightward inputs. The network therefore demonstrates a lookup grid
rather than a classifier over its advertised ITD interval.

The most consequential skill gap is the composition of mapped dynamical State,
time control flow, and delay history. The run attempted several invalid State
axis policies, a shared mutable delay pointer, a native `Delay` inside a full
rollout mapping, and a full `for_loop` nested under `vmap2` before finding the
stable composition: initialize per-lane State, map the complete per-step
transition, then call that mapped step from one `for_loop`. The final result is
correct for the grid, but its `state.value.ndim > 0` filter encodes shape rather
than State ownership and would map any future vector parameter accidentally.

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P1 | `sound_localization.py:24-25`, `test_sound_localization.py:18-21` | The evaluated ITDs are the same 13 values used as detector preferences, so the one-hot detector test is circular. | Passing tests do not show that the circuit localizes an unseen timing difference. | Give the delay bank one tap per simulation step across the supported range and test ITDs that are not detector-construction values, including positive, negative, and zero cases. |
| P1 | `sound_localization.py:182-188` | Zero readout is always decoded as `CENTER`, even when no coincidence detector fired. | Unsupported or missed nonzero ITDs become confident center classifications. The reviewer probe classified all six nonzero off-grid ITDs as `CENTER`. | Distinguish a true zero-ITD center response from no evidence, or ensure every in-range quantized ITD reaches a detector and assert detector activity before decoding. |
| P2 | `sound_localization.py:54-71` | The custom delay line has no isolated impulse-response test. | An indexing or update-order error could preserve direction labels while shifting the represented latency. | Assert that step 0 is current and step `d` returns an impulse exactly `d` completed updates later. |
| P2 | `sound_localization.py:118-145` | Auditory, detector, and readout neurons all advance in one acyclic step, so both projection stages have zero additional simulated latency. | The circuit is a valid compact classifier but should not be presented as a biophysical auditory pathway with explicit synaptic transmission delays. | Keep the teaching-model boundary explicit; add per-stage delay only if the claim requires biological transmission timing. |

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Physical time, voltage, resistance, and current | BrainUnit quantities on all physical constants and inputs | BrainUnit `Quantity` and predefined units | Correct | Keep quantities intact. |
| ITD normalization and dimension checking | `u.math.asarray(itds, unit=u.ms)` | `brainunit.math.asarray` | Correct | Keep; its documented `UnitMismatchError` is tested. |
| Time-axis construction | `jnp.arange(NUM_STEPS) * DT` | `brainunit.math.arange` | Safe but bypasses the canonical unit-aware constructor | Use `u.math.arange(0 * u.ms, DURATION, DT)`. |
| Auditory relays, coincidence detectors, and readouts | Three `brainpy.state.LIFRef` populations with `Constant` voltage initialization | BrainPy-State `LIFRef`; BrainTools `Constant` | Correct and appropriately minimal | Preserve explicit initialization and unit-aware parameters. |
| Independent condition State initialization | `brainstate.nn.vmap_init_all_states` | BrainState collective lifecycle API | Correct | Keep one lane per ITD and verify every writable State has an axis policy. |
| State-axis selection | Callback returns `state.value.ndim > 0` | `brainstate.util.Any` and `brainstate.util.OfType` with `vmap2` State filters | Fragile misuse | Select `HiddenState` and `ShortTermState` semantically; do not infer ownership from rank or a coincidental leading size. |
| Stateful ITD mapping | `vmap2(net.update, in_axes=(None, 0), ...)` | `brainstate.transform.vmap2` | Correct final transform boundary | Teach this mapped-step-inside-`for_loop` composition directly. |
| Time evolution | One `brainstate.transform.for_loop` | BrainState control flow | Correct | Keep one State-aware time loop; optionally wrap the reusable complete run in `brainstate.transform.jit`. |
| Ear-event history | Pointer-free `HiddenState` shifted with `jnp.concatenate` | Prefer `brainstate.nn.Delay`; custom State is a verified fallback for a small fixed integer bank when a mapped Delay's mutable bookkeeping cannot satisfy the lane policy | Legitimate but undocumented fallback | Route native scalar/vector delays first, then document the narrow fallback and require an impulse test. |
| Heterogeneous delay taps | Direct indexing with integer step arrays | `Delay.register_entry`, `Delay.at`, or `retrieve_at_step` for native delay buffers | Direct indexing is correct for the custom history | Add the official vector-delay decision to the delay reference. |
| Binary delayed-event communication | `BinaryArray(delayed_events) @ FixedNumPerPre` | BrainEvent event array and fixed fan-out connectivity | Correct | Keep and verify the pre dimension and postsynaptic result shape. |
| Detector-to-readout communication | Second `BinaryArray @ FixedNumPerPre` | BrainEvent | Correct | Keep; zero-weight center edges are explicit. |
| Spike counting | `jnp.sum` over dimensionless boolean outputs | JAX/host numerical boundary | Legitimate | No BrainX replacement is needed. |
| Direction labels and table formatting | NumPy arrays and Python strings/printing | Host analysis and presentation boundary | Legitimate | Reject no-evidence rows instead of silently labeling them center. |
| Validation | Unit test suite over the construction grid | Host test boundary | Incomplete | Add unseen in-range ITDs, no-evidence handling, impulse timing, symmetry, and repeated-run checks. |

## Missing, bypassed, or misused BrainX APIs

### Semantic State filters

Use `brainstate.util.OfType` to select each dynamical State role and combine the
roles with `brainstate.util.Any`. Replace the rank predicate in
`_mapped_dynamics`. The official `vmap2` contract treats `state_in_axes` and
`state_out_axes` as ownership declarations; array rank is not an ownership
signal and can accidentally map parameters or shared statistics.

### Native delay APIs

Prefer `brainstate.nn.Delay` for a reusable history buffer. Use
`register_entry(name, vector_delays, indices)` plus `at(name)` for named
heterogeneous taps, or `retrieve_at_step(step, *indices)` when the delay is
already an exact integer number of simulation steps. For binary spikes on the
time grid, integer-step retrieval preserves event values; continuous linear
interpolation can return numeric mixtures that are no longer binary.

The final Run 0 custom `HiddenState` history should not be replaced blindly.
The event log demonstrates that mapping the installed ring-buffer `Delay`
created an incompatible combination of per-lane history and scalar mutable
write-pointer State. The refined guidance must make this a decision boundary:
use native Delay when its complete mutable State can remain on one stable axis;
use a small pointer-free per-lane history only after that condition fails, and
verify it with an impulse test.

### Canonical unit-aware range construction

Use `u.math.arange` for the time axis. The existing BrainUnit skill already
states this rule, so no BrainUnit skill edit is justified by this run.

### Whole-run compilation

The final `for_loop` is a transformed scan, but `simulate_itds()` rebuilds the
network and mapping object on each call and does not expose a reusable compiled
run. This is acceptable for the one-off demonstration. If repeated sweeps are
benchmarked, construct one stable `brainstate.transform.jit` callable outside
the timed repetitions; do not add compilation solely to satisfy an API list.

## Performance and code simplicity

- The final code has one time loop and one actual stateful lane mapping. It does
  not fake batching through input construction or host-side scoring.
- `FixedNumPerPre` is the right representation: each event source has exactly
  one target, and the sparse fixed-degree structure avoids a dense routing
  matrix.
- The pointer-free history shifts a tiny fixed buffer and is simple at this
  scale, but its update cost grows with the maximum delay. Native ring-buffer
  Delay remains preferable when its State-axis contract is compatible.
- The transform-discovery failures dominate the implementation trace. A direct
  mapped-step pattern and semantic State filter would remove most of that
  search without adding abstraction.
- NumPy decoding and text presentation are appropriate host boundaries. No
  BrainX API should be invented for label strings or console tables.

## Skill improvements

1. Refine `brainx-general-guard` so a stateful `vmap` requirement can be
   satisfied by mapping the complete per-step transition inside one
   `for_loop`; distinguish this from mapping only input construction or host
   scoring. Require semantic State-role filters rather than shape heuristics.
2. Add a compact mapped-dynamics section to
   `brainstate/references/brainstate/transformation-vmap-expansion.md` showing
   `vmap_init_all_states`, `Any(OfType(HiddenState),
   OfType(ShortTermState))`, `vmap2`, and the mapped-step-inside-`for_loop`
   composition. State that shared mutable State is invalid even when every
   lane would write the same value.
3. Extend
   `brainpy-state/references/brain-dynamics-delay-protocol.md` with the official
   heterogeneous `register_entry`/`at` path, the integer-step rule for binary
   events, and the mapped-delay State boundary exposed by this run.
4. Add one validated BrainPy-State application script for delay-line
   coincidence localization. It must use semantic State filters, one detector
   preference per `dt`, unseen evaluation ITDs, BrainEvent fixed fan-out
   products, BrainUnit ranges, and an impulse-delay assertion.
5. Route that script precisely from the BrainPy-State root and projection/delay
   references. Do not expand the root with sound-localization theory.
6. Do not change `brainevent` or `brainunit`; Run 0 used their material APIs
   correctly and their current guidance already covers the small bypasses.

## Checks for the next run

- The entry point exits zero in the required BrainX virtualenv.
- Positive nonzero ITDs decode left, negative nonzero ITDs decode right, and
  zero ITD is handled explicitly.
- At least four tested ITDs are not detector-construction values or default
  sweep values copied into the delay bank.
- Every in-range tested ITD produces detector evidence; no nonzero case reaches
  `CENTER` merely because both readout counts are zero.
- A unit impulse proves the delay convention for step 0 and at least one
  nonzero integer delay.
- `vmap2` maps writable dynamical State by semantic role, not by `.ndim`, path
  coincidence, or parameter shape.
- The actual neural transition is mapped; input-only or scoring-only `vmap`
  does not count.
- The simulation contains one BrainState time loop and no Python timestep loop.
- Binary communication uses `BinaryArray` with a dimensionally correct
  connectivity representation and unit-aware weights.
- The same prompt bytes, model, reasoning effort, virtualenv, CLI, and isolated
  harness conditions match Run 0.
