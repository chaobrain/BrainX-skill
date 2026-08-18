# BrainX diagnosis: neural compass Run 1

## Evidence studied

- Generated artifacts: `README.md`, `neural_compass.py`,
  `neural_compass_results.png`, `neural_compass_results.npz`, `agent-final.md`,
  `codex-events.jsonl`, and `codex-stderr.log`.
- Execution: an untouched temporary copy completed under the frozen BrainX
  virtualenv. The PNG and NPZ reproduced byte-for-byte with SHA-256 values
  `9e87bf5b2f3746f202f7188202e2269c046663871f573c494ef5785af17a6d05`
  and `854f2f6b7841d3cf09e4fdacf0f022ea420bbefa06ca330560d63f80452a254b`.
- Artifact inspection: all 21 saved arrays were finite and equal across runs;
  the figure was opened at its native `2040 x 1530` resolution.
- Independent scientific checks: all 48 matched controls finished within
  `0.25 deg`; the four recovered headings departed by `34.79-41.70 deg`, then
  returned to `15.02-15.23 deg` final error with `98.5-100%` of the final
  window inside the success limit. Their measured departure intervals ended
  before their final return evidence.
- Owning skills: `skills/brainx-general-guard/SKILL.md`,
  `skills/brainpy-state/SKILL.md`, `skills/brainevent/SKILL.md`,
  `skills/brainstate/SKILL.md`, and `skills/brainunit/SKILL.md`.
- Routed references: `brainpy-state/references/component-selection.md`,
  `brainpy-state/references/projection-patterns.md`,
  `brainpy-state/references/brain-dynamics-delay-protocol.md`,
  `brainstate/references/brainstate/transformation-vmap-expansion.md`, and
  `brainstate/references/collective_model_operations.md`.
- Closest executable examples:
  `brainpy-state/references/scripts/sound_localization.py` for independent
  mapped State and a verified pointer-free integer delay, and
  `brainevent/references/scripts/coba_ei_teaching.py` for binary event
  communication into BrainPy-State `LIFRef` and `Expon` dynamics.
- API standard: the Run 0 review had already verified the generated official
  contracts for `LIFRef`, `Expon`, `BinaryArray`, `for_loop`, `vmap2`, `jit`,
  and `vmap_init_all_states`. The repository's official-source-indexed
  references were rechecked for Run 1. Live BrainX pages could not be reopened
  because command-line DNS resolution failed and no in-app browser was
  available; no contract changed in the generated implementation.

## Executive diagnosis

Run 1 corrects Run 0's central labeling defect. It distinguishes `spared` from
`recovered`, retains the continuous scalar reductions behind every label, and
verifies each emitted label against its stated boolean predicate. The four
reported recoveries also satisfy the intended time order under independent
analysis. BrainX API selection, units, mapped State ownership, compiled time
loop, dense event communication, and fixed-delay fallback are correct.

Two cross-package failures remain. First, the program requires all three
outcome categories to appear, turning a requested display vocabulary into a
calibration target. Its boolean predicate also combines a departure anywhere
after intervention with a final-window return fraction without explicitly
requiring the departure to precede the return window. Second, the figure shows
only final angular error and the categorical map. It omits coherence, matched
mass, maximum departure, minimum post-intervention mass, and sustained-return
fraction even though those quantities determine the labels. The plotting code
also uses `plt.figure`, `GridSpec`, repeated `add_subplot`, and a polar
projection despite the guard's explicit requirement for the shortest direct
`pyplot` composition.

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P1 | `neural_compass.py:402` | Validation fails unless `spared`, `recovered`, and `failed` all appear. | The agent can tune the intervention to manufacture requested regimes instead of reporting the model's behavior. | Remove category-presence assertions; accept an empty regime when the measured predicates produce one. |
| P1 | `neural_compass.py:313` | `departed` is reduced over the complete post-intervention interval while `sustained` is reduced independently over the final window. | The predicate does not itself prove that departure precedes the claimed sustained return, even though the four current labels happen to have the correct order. | Record the departure time and require a later, non-overlapping sustained-return window. |
| P1 | `neural_compass.py:446` | The figure visualizes final error and labels but not every continuous boundary observable. | Readers cannot reconstruct why a point is spared, recovered, or failed from the figure. | Show control validity, departure, final error, coherence, matched mass, and sustained return with their thresholds using simple aligned Cartesian panels. |
| P2 | `neural_compass.py:412` | Plot composition uses `plt.figure`, `add_gridspec`, four `add_subplot` calls, and a polar projection. | The forward test bypasses the new absolute-simplicity rule and spends code on layout rather than evidence. | Create each figure with one `plt.subplots(...)` call and basic high-level methods; simplify the categorical view instead of introducing mixed projection machinery. |
| P2 | `neural_compass_results.npz` | Only reduced lesion predicates are saved, not the matched error and mass-ratio trajectories from which departure order is derived. | A reviewer can reconstruct the boolean label but cannot independently recompute the claimed temporal order from the archive. | Save the minimal aligned continuous trajectories or explicit ordered event times needed to audit departure followed by return. |

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Point-neuron dynamics | `brainpy.state.LIFRef` | BrainPy-State `LIFRef` | Correct. Refractory timing, voltage parameters, resistance, and initializer are explicit and unit-aware. | None. |
| Recurrent and readout filtering | Two `brainpy.state.Expon` modules | BrainPy-State `Expon` | Correct. Recurrent State carries current units; readout State is dimensionless. | None. |
| Binary recurrent communication | `BinaryArray(delayed_spikes) @ dense_weights` | BrainEvent `BinaryArray` | Correct. Only boolean spikes cross the event boundary. | None. |
| Ring and velocity kernels | Small dense signed matrices | Custom model behavior | Correct. The `48 x 48` cosine and derivative kernels are genuinely dense and time-varying velocity gain makes direct composition appropriate. | None. |
| Fixed synaptic delay | Pointer-free `HiddenState` shift register plus impulse check | BrainState delay fallback routed by the delay protocol | Correct for a two-step binary delay whose mapped native buffer failed in this environment. | Preserve the impulse check. |
| Physical quantities | BrainUnit for time, voltage, current, resistance, angle, and angular velocity | BrainUnit quantities and `brainunit.math` | Correct. `u.math.where` preserves current units and `to_decimal` is used at host boundaries. | None. |
| Independent conditions | `vmap_init_all_states` and semantic Hidden/ShortTerm filters | BrainState lifecycle and `vmap2` | Correct. All 96 control/intervention lanes own independent writable State. | None. |
| Condition mapping | One `vmap2(model.update, ...)` | BrainState `vmap2` | Correct. It maps the complete step and raises on undeclared State writes. | None. |
| Time rollout | One `for_loop` inside `brainstate.transform.jit` | BrainState `for_loop` and `jit` | Correct. There is no Python timestep loop. | None. |
| Cue, turn, and lesion schedules | Time-major arrays built before rollout | BrainUnit plus host protocol construction | Correct for deterministic dimensionless spatial masks and explicitly unit-bearing current/time boundaries. | None. |
| Circular decoding and labels | Host NumPy reductions | Host scientific-analysis boundary | Appropriate boundary, but the temporal predicate and category-presence assertion are scientifically unsafe. | Encode ordered departure/return and allow empty categories. |
| Serialization | `np.savez_compressed` | Host serialization boundary | Correct format, but temporal recovery evidence is incomplete. | Add only the minimal aligned evidence needed for auditability. |
| Figure | Custom `Figure`/`GridSpec`/polar layout | High-level Matplotlib host boundary | Scientifically readable but contrary to the requested composition rule and incomplete as label evidence. | Use one `plt.subplots` composition and plot every boundary observable. |

## Missing, bypassed, or misused BrainX APIs

No material BrainX API is missing or misused.

- Do not force the time-varying signed recurrent kernel through a static stock
  projection. The current direct BrainEvent-to-`Expon` composition is the
  smaller faithful model.
- Do not replace the small dense ring matrices with sparse or generated
  connectivity; their signed cosine and derivative structure is dense.
- Do not replace the fixed mapped delay fallback without first proving that all
  writable delay State follows the condition axis in this environment.
- Do not invent a BrainX circular decoder, category engine, serializer, or
  plotting API. These remain legitimate host boundaries.

## Performance and code simplicity

- One mapped complete step inside one compiled time loop is the correct
  execution structure. Controls and interventions share that same path.
- The model has one State-owning class and no unnecessary simulation layer.
- Host NumPy decoding and serialization occur after State-aware execution and
  are appropriate.
- Plotting is the main avoidable complexity. Mixed Cartesian/polar layout adds
  no scientific evidence and can become a direct `plt.subplots` grid.
- The full post-intervention activity need not be retained if explicit ordered
  departure and return times plus every scalar boundary observable are saved;
  choose whichever produces the smaller auditable artifact.

## Skill improvements

Make two small cross-package edits in `brainx-general-guard` and mirror them in
`plan.md`:

1. Tighten recovery guidance so a measured departure must precede a later
   sustained-return window, and explicitly forbid requiring requested outcome
   categories to appear.
2. Make Matplotlib composition concrete: one `plt.subplots(...)` call per
   figure and only basic high-level plotting methods by default. Permit
   `Figure`, `GridSpec`, `add_subplot`, projection-specific axes, custom
   artists, or manual layout only when the user explicitly requests a result
   that `subplots()` cannot express.

No BrainPy-State, BrainState, BrainEvent, or BrainUnit edit is justified.

## Checks for the next run

- The exact prompt and frozen execution conditions remain unchanged.
- Every represented heading has an independent matched control and
  intervention lane in one `vmap2` inside one `for_loop`.
- Controls validate before intervention labels are interpreted.
- `spared`, `recovered`, and `failed` may each be empty; no check requires a
  regime to appear.
- A recovered condition has an explicit departure time before a later
  sustained-return window; spared conditions never cross the departure
  boundary.
- Every continuous boundary observable is saved, tested, and visible with its
  threshold; the categorical map is not a substitute.
- Each Matplotlib figure starts with one `plt.subplots(...)` call and uses only
  basic high-level plotting methods unless the prompt explicitly demands a
  layout or projection that this cannot express.
- The default run completes and all saved outputs reproduce under the frozen
  environment.
