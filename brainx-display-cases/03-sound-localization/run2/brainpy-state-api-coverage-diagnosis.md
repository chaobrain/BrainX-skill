# BrainX diagnosis: sound localization from timing

## Evidence studied

Generated artifacts:

- `sound_localization.py`, `test_sound_localization.py`, and `README.md`.
- `agent-final.md`, `codex-events.jsonl`, `codex-stderr.log`, and
  `harness-metadata.txt`.
- The archived entry point and all three archived tests, rerun without
  modifying generated files under
  `/Users/nijiachen/Downloads/Brainx testing/.venv-brainx`.
- Reviewer-only probes over the exact `+/-0.6 ms` boundaries, unseen ITDs,
  exact half-step ITDs, sub-resolution ITDs, the complete 25-detector grid,
  repeated runs, and a synthetic no-evidence row.

Owning skills and references:

- `skills/brainx-general-guard/SKILL.md`.
- `skills/brainpy-state/SKILL.md`, `references/projection-patterns.md`,
  `references/brain-dynamics-delay-protocol.md`, and
  `references/scripts/sound_localization.py`.
- `skills/brainstate/SKILL.md` and
  `references/brainstate/transformation-vmap-expansion.md`.
- `skills/brainevent/SKILL.md` and `skills/brainunit/SKILL.md`.

Review standard:

- `skills/brainpy-state/references/scripts/103_COBA_2005.py` and
  `109_fast_global_oscillation.py`.
- `skills/brainevent/references/scripts/coba_ei_teaching.py`.
- Generated contracts for `brainpy.state.LIFRef`,
  `brainstate.transform.vmap2`, `brainstate.nn.vmap_init_all_states`,
  `brainstate.transform.for_loop`, `brainstate.nn.Delay`,
  `brainstate.util.Any`, `brainstate.util.OfType`,
  `brainevent.BinaryArray`, `brainevent.FixedNumPerPre`, and the BrainUnit
  array constructors.
- The official BrainState Delay Protocol tutorial.

## Executive diagnosis

Run 2 is scientifically coherent for its stated discrete-time teaching model
and uses the material BrainX APIs correctly. It validates physical ITD units
and bounds before rounding once to integer `dt` steps, generates auditory
events through exact step equality, maps independent neural State by semantic
role, evolves all conditions in one BrainState time loop, and refuses to decode
rows without detector evidence.

The entry point, all three archived tests, and every reviewer probe pass.
Negative resolvable ITDs decode `RIGHT`, positive resolvable ITDs decode
`LEFT`, zero and sub-resolution offsets reach the center detector, exact
half-step examples are symmetric, and every supported detector-grid value
activates its matching cell. No further skill refinement is justified by this
run.

## Scientific problems

No unresolved P0-P2 scientific problem remains.

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P3 boundary | `README.md:33-34`, `sound_localization.py:150-152` | The circuit resolves ITD only on a `0.05 ms` grid and uses nearest-step rounding. | Offsets smaller than half a step are intentionally perceived as center; the model is not a continuous-time estimator. | Already documented. Reduce `dt` or use a continuous interpolation model only when the scientific question requires finer timing. |
| P3 boundary | `sound_localization.py:116-138` | Auditory, detector, and readout populations advance in one acyclic transition without explicit per-stage transmission delays. | The circuit demonstrates Jeffress-style coincidence classification, not a biophysical auditory pathway with measured stage latencies. | Keep this compact boundary; add stage delays only when the requested claim requires them. |

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Physical parameters | BrainUnit quantities for time, voltage, current, and resistance | BrainUnit `Quantity` and predefined units | Correct | Keep quantities intact. |
| ITD normalization | `u.math.asarray(itds, unit=u.ms)` | `brainunit.math.asarray` | Correct | Keep at the public boundary. |
| Supported physical interval | BrainUnit comparison against `MAX_INTERNAL_DELAY_STEPS * DT` | BrainUnit comparison, followed by a host validation branch | Correct for this eager public wrapper | Keep validation before rounding. |
| Discrete timing boundary | One `jnp.rint(itds / DT)` after unit conversion | Explicit dimensionless JAX index boundary | Correct | Keep one documented conversion before the rollout. |
| Time and step axes | `u.math.arange` for time and integer `jnp.arange` for event indices | BrainUnit physical axis plus legitimate dimensionless JAX index axis | Correct | Keep both axes synchronized by construction. |
| Point-neuron dynamics | Three `brainpy.state.LIFRef` populations with `Constant` initialization | BrainPy-State `LIFRef`; BrainTools `Constant` | Correct | Keep. |
| Independent condition State | `vmap_init_all_states` | BrainState collective lifecycle API | Correct | Keep. |
| State ownership | `Any(OfType(HiddenState), OfType(ShortTermState))` | BrainState semantic filters | Correct | Keep; no shape heuristic remains. |
| Stateful batching | Complete `net.update` mapped by `vmap2` | `brainstate.transform.vmap2` | Correct | Keep explicit input axes, output axis, and fail-closed write policy. |
| Time evolution | Mapped step called inside one `for_loop` | `brainstate.transform.for_loop` | Correct | Keep. |
| Ear-event history | Small pointer-free `HiddenState` shift register | Verified custom fallback when mapped ring-buffer bookkeeping cannot follow one axis policy | Correct at this fixed scale | Keep the impulse test; use native `Delay` when its complete State is compatible or the history is longer. |
| Integer heterogeneous taps | Direct indexing by detector delay steps | Custom fixed-bank indexing; native `Delay.register_entry`/`at` is the general path | Correct for the fallback | Keep binary integer taps. |
| Detector communication | `BinaryArray @ FixedNumPerPre` | BrainEvent binary event and fixed fan-out APIs | Correct | Keep. |
| Readout communication | Second `BinaryArray @ FixedNumPerPre` | BrainEvent | Correct | Keep. |
| Spike aggregation | `jnp.sum` over boolean monitors | JAX numerical boundary | Legitimate | No BrainX replacement is needed. |
| Evidence-aware direction labels | NumPy and Python strings | Host analysis boundary | Correct | Keep detector-evidence and center-tie checks. |
| Presentation and instructions | Python printing and Markdown | Host presentation boundary | Correct | Keep. |
| Tests | Standard-library `unittest` over delay, direction, unit, and range behavior | Host validation boundary | Correct and self-contained | Reviewer probes add coverage; no extra framework dependency is needed. |

## Missing, bypassed, or misused BrainX APIs

No material BrainX API is missing, bypassed, or misused.

### Native delay API boundary

Do not replace the fixed 13-row pointer-free history mechanically. The buffer
is binary, small, mapped per condition, and verified by an impulse test. Prefer
`brainstate.nn.Delay` for a general or longer history only when all of its
mutable State can follow the mapped axis policy.

### Compilation boundary

The script is a one-off demonstration and `for_loop` already lowers the
stateful time sequence. A reusable `brainstate.transform.jit` wrapper is not
required without a repeated-sweep benchmark or API requirement.

### Host boundaries

The eager range check, spike-count reductions, string labels, table output,
and `unittest` assertions are legitimate host responsibilities. Do not invent
BrainX APIs for them.

## Performance and code simplicity

- One mapped complete neural transition owns independent conditions; one
  `for_loop` owns time. There is no Python timestep loop or input-only mapping.
- ITDs are quantized once outside the rollout, removing the Run 1 floating
  arrival comparison from every step.
- Semantic State filters prevent accidental parameter mapping and the
  unexpected-write policy fails closed.
- Both connections use fixed fan-out storage that matches their exact graph
  structure.
- The pointer-free history shifts 13 rows per step. This is simple and small
  here, but should not be scaled into a general long-delay implementation.
- NumPy conversion occurs only after the transformed simulation, at analysis
  and presentation boundaries.
- The README and three standard-library tests add useful execution guidance
  and validation without a project framework.

## Skill improvements

No further skill edit is justified. Run 2 follows the latest general guard,
BrainPy-State delay/application guidance, BrainState State-axis composition,
BrainEvent communication, and BrainUnit boundaries correctly.

## Checks for the next run

No Run 3 is required. Final repository validation should retain these checks:

- The reference script exits zero in the required BrainX virtualenv.
- Negative, zero, and positive default ITDs retain one ear spike per channel,
  detector evidence, and the expected label.
- Exact `+/-0.6 ms` inputs succeed; values just outside fail before rounding.
- Exact half-step values are symmetric and sign-correct; sub-resolution values
  follow documented nearest-step behavior.
- The complete 25-value detector grid selects one matching detector per input.
- Synthetic no-evidence rows are rejected.
- The delay impulse, semantic State filters, mapped-step-inside-loop
  composition, BrainEvent products, units, and repeated-run determinism remain
  intact.

## Comparison with Runs 0 and 1

Run 2 preserves every Run 1 improvement over Run 0: off-grid evaluation,
detector-evidence decoding, semantic State filters, one stateful mapped step,
unit-aware time construction, and an isolated delay test. It also removes Run
1's floating half-step asymmetry by quantizing ITD once and generating auditory
events at exact integer steps. The final implementation is both more explicit
and cheaper per timestep than either earlier checkpoint.
