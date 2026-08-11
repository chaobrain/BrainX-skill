# BrainX diagnosis: sound localization from timing

## Evidence studied

Generated artifacts:

- `sound_localization.py`, `agent-final.md`, `codex-events.jsonl`,
  `codex-stderr.log`, and `harness-metadata.txt`.
- The archived entry point, rerun without modifying the generated source under
  `/Users/nijiachen/Downloads/Brainx testing/.venv-brainx`.
- A reviewer-only unseen-ITD probe at `[-0.58, -0.41, -0.28, -0.12,
  -0.02, 0.0, 0.02, 0.12, 0.28, 0.41, 0.58] ms`.
- Reviewer-only half-step probes at `[-0.075, -0.025, -0.0249, 0.0249,
  0.025, 0.075] ms`.

Owning skills and references:

- `skills/brainx-general-guard/SKILL.md`.
- `skills/brainpy-state/SKILL.md`, `references/projection-patterns.md`,
  `references/brain-dynamics-delay-protocol.md`, and
  `references/scripts/sound_localization.py`.
- `skills/brainstate/SKILL.md` and
  `references/brainstate/transformation-vmap-expansion.md`.
- `skills/brainevent/SKILL.md` and `skills/brainunit/SKILL.md`.

Closest executable examples and authoritative contracts remain the Run 0
standard: `103_COBA_2005.py`, `109_fast_global_oscillation.py`,
`coba_ei_teaching.py`, the generated `LIFRef`, `vmap2`,
`vmap_init_all_states`, `for_loop`, `Delay`, `Any`, `OfType`, `BinaryArray`,
`FixedNumPerPre`, and BrainUnit array-construction APIs, plus the BrainState
Delay Protocol tutorial.

## Executive diagnosis

Run 1 fixes the consequential Run 0 failures. Its default evaluation uses six
non-grid ITDs plus zero and two exact-grid controls, every default condition
produces detector evidence, nonzero decoded directions have the correct
polarity, and center requires evidence from the zero-delay detector. It uses
semantic State-role filters, initializes independent State lanes, maps the
complete per-step neural transition, and calls that mapping from one
`for_loop`. An isolated impulse proves the custom delay convention.

The remaining scientific defect is confined to stimulus discretization.
Continuous arrival times are sampled with a strict half-`dt` window. At exact
half-step values the result depends on floating representation: the reviewer
probe produced two ear spikes and a valid rightward response at `-0.075 ms`,
but only one ear spike, zero detector evidence, and a decoding error at
`+0.075 ms`. The model should quantize the unit-bearing ITD to one integer step
once before the rollout, then generate both auditory events on that explicit
grid. This also makes the expected `dt` resolution and sub-resolution center
behavior deliberate rather than incidental.

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P1 | `sound_localization.py:118-122` | A strict half-`dt` sampling window converts a continuous arrival time into an event implicitly and is floating-point sensitive at exact half steps. | `+0.075 ms` loses the right-ear spike and has no detector evidence while `-0.075 ms` localizes correctly, creating an artificial directional asymmetry. | Validate the quantity, quantize ITD to an integer number of `dt` steps once before `vmap`, construct the grid-aligned arrival from that integer, and use a narrow grid-equality tolerance or an integer arrival index. |
| P3 | `sound_localization.py:20`, `sound_localization.py:118-121` | The model's temporal resolution is encoded only by the sampling expression. | Values such as `+/-0.02 ms` reach the center detector, but the artifact does not state that offsets below half a timestep are intentionally unresolved. | Make timestep quantization explicit in code and describe detector output as the quantized ITD estimate. |
| P3 | `sound_localization.py:116-138` | Auditory, detector, and readout neurons advance in one acyclic transition with no additional synaptic transmission step. | This remains a compact classifier rather than a biophysical auditory pathway with explicit stage latencies. | Keep the teaching-model boundary; add stage delays only if the claimed mechanism requires biological transmission timing. |

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Physical time, voltage, resistance, and current | BrainUnit quantities | BrainUnit `Quantity` and predefined units | Correct | Preserve quantities. |
| ITD validation | `u.math.asarray(itds, unit=u.ms)` | `brainunit.math.asarray` | Correct | Keep before discretization. |
| Time-axis construction | `u.math.arange(0 * u.ms, DURATION, DT)` | `brainunit.math.arange` | Correct | Keep. |
| Arrival discretization | Strict `abs(t - arrival) < dt / 2` | Dimensionless step-index boundary after BrainUnit validation | Floating-sensitive misuse | Quantize once to integer steps, then map the grid-aligned result. |
| Auditory, detector, and readout populations | Three `brainpy.state.LIFRef` populations with `Constant` initialization | BrainPy-State `LIFRef`; BrainTools `Constant` | Correct | Keep. |
| Per-condition State allocation | `vmap_init_all_states` | BrainState collective lifecycle API | Correct | Keep. |
| State-axis selection | `Any(OfType(HiddenState), OfType(ShortTermState))` | BrainState semantic filters | Correct | Keep; this fixes the Run 0 rank predicate. |
| Stateful condition batching | `vmap2(net.update, ...)` | `brainstate.transform.vmap2` | Correct | Keep the complete mapped step and explicit output policy. |
| Time evolution | Mapped step inside one `for_loop` | `brainstate.transform.for_loop` | Correct | Keep. |
| Ear-event history | Small pointer-free `HiddenState` shift register | Verified fallback when mapped `Delay` bookkeeping cannot follow one axis policy | Correct at this fixed 13-step scale | Keep the impulse assertion; prefer native `Delay` for compatible or longer buffers. |
| Integer heterogeneous taps | Direct indexing by signed-preference-derived step arrays | Native `Delay.register_entry`/`at` or custom fixed-bank indexing | Correct for the verified fallback | Keep binary values and integer taps. |
| Event communication | Two `BinaryArray @ FixedNumPerPre` products | BrainEvent event array and fixed fan-out connectivity | Correct | Keep. |
| Spike counting | `jnp.sum` over dimensionless boolean monitors | JAX numerical boundary | Legitimate | No replacement needed. |
| Evidence-aware decoding and text output | NumPy arrays and Python labels/printing | Host analysis and presentation boundary | Correct | Keep no-evidence rejection and center-detector tie validation. |
| Validation | Default off-grid checks, full-grid event-log check, and impulse assertion | Host validation boundary | Substantially improved | Add an exact half-step symmetry check after explicit quantization. |

## Missing, bypassed, or misused BrainX APIs

### Explicit discrete-time stimulus construction

Keep `u.math.asarray(..., unit=u.ms)` as the dimensional boundary, then convert
the ratio `itds / DT` to integer step offsets once. The rollout should consume
those offsets or their grid-aligned BrainUnit quantities. Do not repeatedly
sample a nominally continuous arrival with a half-open floating window when the
delay bank itself supports only integer steps.

This is not a reason to add a generic input generator. The two single-spike
auditory drives are model-specific and the direct comparison is simpler once
their step convention is explicit.

### Native delay APIs

No direct replacement is justified. The custom buffer is small, fixed, binary,
and now has an impulse test. The Run 0 mapped ring-buffer boundary still
applies: use `brainstate.nn.Delay` only when its complete mutable State can
follow the per-condition axis policy.

### Whole-run compilation

The one-off script uses transformed control flow and rebuilds the model for
each call. A separate stable `brainstate.transform.jit` wrapper is unnecessary
without a repeated-sweep performance requirement.

## Performance and code simplicity

- The implementation has one actual stateful condition mapping and one time
  loop; no Python timestep loop or host-side simulation batch remains.
- The semantic State filter prevents accidental parameter mapping and the
  explicit unexpected-write policy fails closed.
- `FixedNumPerPre` matches the two exact fan-out structures without dense
  matrices.
- The short shift register is acceptable for 13 history rows; its cost grows
  with maximum delay and should not become the general delay implementation.
- Quantizing ITDs once before the rollout removes a floating comparison from
  every time step and makes the supported resolution explicit with less
  runtime work.
- One runnable source file is sufficient for the prompt. The event-log-only
  full-grid checks are useful review evidence but need not become project
  scaffolding.

## Skill improvements

1. Refine only `skills/brainpy-state/references/scripts/sound_localization.py`:
   quantize validated ITDs to integer steps once, drive grid-aligned auditory
   events, retain the off-grid default sweep, and assert exact half-step
   polarity/evidence.
2. Add one concise rule to
   `skills/brainpy-state/references/brain-dynamics-delay-protocol.md`: when a
   physical event feeds an integer delay bank, quantize it once to `dt` before
   the rollout rather than using a half-open floating sampling window.
3. Do not change `brainx-general-guard`, BrainState, BrainEvent, or BrainUnit;
   Run 1 follows their refined workflows correctly.

## Checks for the next run

- The entry point exits zero in the required BrainX virtualenv.
- Default non-grid ITDs all produce one spike per ear, detector evidence, and
  correct polarity; zero reaches the center detector.
- Exact positive and negative half-step inputs produce symmetric ear counts,
  detector evidence, and sign-correct labels under one documented rounding
  rule.
- Sub-resolution offsets follow the documented quantization result rather than
  floating representation.
- No-evidence rows are still rejected.
- `vmap2` maps writable State with semantic State-role filters and the mapped
  neural step remains inside one `for_loop`.
- The delay impulse assertion, BrainEvent products, units, repeated-run
  determinism, and invalid-unit checks still pass.
- Prompt bytes, model, effort, virtualenv, CLI, and isolated harness conditions
  remain identical to Runs 0 and 1.

## Comparison with Run 0

Run 1 eliminates Run 0's circular default validation, false-center
no-evidence decoding, rank-based State filter, missing impulse test, generic
time range, and transform-discovery failures. It preserves the same compact
BrainPy-State and BrainEvent circuit while extending reliable classification
from the detector-construction grid to ordinary unseen ITDs. Only the exact
half-step discretization boundary requires another refinement.
