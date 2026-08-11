# BrainX diagnosis: neural compass

## Evidence studied

- Generated artifacts: `README.md`, `head_direction_compass.py`,
  `test_head_direction_compass.py`, `results/head_direction_compass.png`,
  `results/lesion_sweep.csv`, `results/summary.json`, `agent-final.md`,
  `codex-events.jsonl`, and `codex-stderr.log`.
- Execution: an untouched Run 0 copy completed the default experiment and all
  five unit tests under the frozen BrainX virtualenv. The PNG, CSV, and JSON
  reproduced byte-for-byte with SHA-256 values `152793500cb5a622e8ca69ab8b95dee7541a5c159e5477bce51126b3fe33c84c`,
  `f933d3549e1f6270a40c49475b21250be8aba02195735672b0e52558f0aa17c2`,
  and `a676b07bf931c9b11969c0933c90ba743d9fa708c5a7cb33b66e5e6fc12d9b3f`.
- Independent scientific checks: all intact conditions held a localized bump,
  tracked the turn, and remained localized afterward; the maximum pre-turn
  control error was below `0.00005 deg`, maximum post-turn error was `4.39 deg`,
  and minimum control resultant after cue offset was `0.68`. Returned lesion
  events contained zero wedge spikes after lesion onset.
- Owning skills: `skills/brainx-general-guard/SKILL.md`,
  `skills/brainpy-state/SKILL.md`, `skills/brainevent/SKILL.md`,
  `skills/brainstate/SKILL.md`, and `skills/brainunit/SKILL.md`.
- Routed references: `brainpy-state/references/component-selection.md`,
  `brainpy-state/references/projection-patterns.md`,
  `brainpy-state/references/braintools/connectivity.md`,
  `brainstate/references/brainstate/transformation-vmap-expansion.md`,
  `brainstate/references/brainstate/brainstate-control-flow-patterns.md`,
  `brainevent/references/sparse-formats.md`, and
  `brainunit/references/typing-and-runtime-validation.md`.
- Closest executable examples:
  `brainpy-state/references/scripts/sound_localization.py` for independent
  mapped dynamical State inside one transformed time loop, and
  `brainevent/references/scripts/coba_ei_teaching.py` for `BinaryArray`
  communication into BrainPy-State `Expon` and `CUBA` dynamics.
- Official examples and contracts: BrainPy-State COBA/CUBA selection how-to;
  BrainEvent quickstart and unit-aware computation how-to; generated APIs for
  `LIFRef`, `Expon`, `CUBA`, `BinaryArray`, `brainstate.transform.for_loop`,
  `brainstate.transform.vmap2`, `brainstate.transform.jit`,
  `brainstate.nn.vmap_init_all_states`, `braintools.input.Constant`,
  `braintools.conn.Ring`, and `braintools.conn.ConnectionResult`; central APIs
  for BrainTools connectivity, metrics, and visualization.

## Executive diagnosis

Run 0 is reproducible and uses the requested BrainX packages correctly. One
State-aware `vmap2` owns all 72 independent control/lesion conditions, one
`for_loop` owns time, `BinaryArray` is used only for binary spikes, the event
weights and neuron dynamics remain unit-aware, and the dense ring matrices are
justified by the small genuinely dense signed kernel. The four-panel figure is
made with direct high-level `matplotlib.pyplot` calls and has no custom artist,
manual-axes, or style-system machinery.

The central scientific conclusion is nevertheless mislabeled. The reported
`recovered` predicate checks only final-window error, activity, and resultant.
It does not require the lesion trajectory to depart from its matched control
before returning. Independent time-resolved comparison showed that six of the
seven green headings never exceeded the stated `30 deg` perturbation boundary;
they were spared, not recovered. The seventh exceeded it by only `0.01 deg`.
The summary, README, figure title, and regression test therefore overstate
evidence for recovery even though the underlying simulation is valid.

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P1 | `head_direction_compass.py:297` | `recovered` is an endpoint predicate and contains no prior-departure or sustained-return condition. | Six of seven reported recoveries never left the success band, so the main recovery claim is false. | Compute matched time-resolved lesion error. Label `spared` when it never crosses the perturbation boundary; label `recovered` only after a crossing followed by a sustained return window; keep failed and invalid-control conditions separate. |
| P1 | `head_direction_compass.py:469`, `README.md`, `results/summary.json`, and the figure title | Outputs collapse spared and recovered conditions into one `recovered` count. | Readers cannot distinguish robustness to an irrelevant wedge from restoration after impairment. | Report separate spared, recovered, failed, and invalid-control headings with explicit predicates. |
| P2 | `head_direction_compass.py:304` and `results/lesion_sweep.csv` | The sweep retains only final lesion error, activity, and resultant; it omits maximum post-lesion deviation, return duration, and recovery time. | The categorical label cannot be independently reconstructed or audited from saved evidence. | Retain every time-resolved boundary reduction used by the label and write it per heading. |
| P2 | `head_direction_compass.py:416` | The categorical map visualizes final lesion predicates but not the departure/return evidence or all continuous control gates. | The figure cannot show that a green point actually recovered or that every matched control passed its full predicate. | Plot the smallest set of control and lesion boundary observables needed to reconstruct every label; keep the direct `pyplot` implementation. |
| P2 | `test_head_direction_compass.py:60` | The integration test requires only a nonempty, noncomplete set of endpoint `recovered` labels. | The same spared-as-recovered error is protected as desired behavior. | Assert the complete spared/recovered/failed predicates from continuous trajectories, plus wedge silence and valid matched controls. |

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Point-neuron dynamics | `brainpy.state.LIFRef` | BrainPy-State `LIFRef` | Correct. Explicit resistance, voltage, time constants, refractory period, and initializer match the generated contract. | None. |
| Recurrent temporal filtering | `brainpy.state.Expon` | BrainPy-State `Expon` | Correct. Event products carry `mS`, matching the synaptic State unit. | None. |
| Signed current conversion | `brainpy.state.CUBA(scale=nA/mS)` | BrainPy-State `CUBA` | Correct. The product has current units and is independent of postsynaptic voltage as intended. | None. |
| Recurrent projection composition | Explicit `BinaryArray @ weights`, `Expon`, `CUBA`, and registered current | BrainPy-State projection components plus custom model behavior | Correct custom boundary. Time-varying signed velocity gating is the scientific rule; forcing it through a stock static projection would add machinery without preserving a simpler contract. | Keep the direct Module composition. |
| Ring hold and derivative kernels | Dense rotation-equivariant matrices | Custom model behavior; BrainTools `Ring` cannot express the dense signed cosine and derivative weights | Correct host/model boundary. Dense storage is appropriate for `36 x 36` genuinely dense kernels. | None. |
| Binary spike communication | `brainevent.BinaryArray(previous_spikes) @ dense_weights` | BrainEvent `BinaryArray` | Correct. Only boolean spikes cross the event boundary; analog activity is not wrapped. | None. |
| Physical quantities | BrainUnit time, angular velocity, voltage, current, resistance, conductance, and boundary conversion | BrainUnit quantities and `brainunit.math` | Correct. Raw values appear only at trigonometric dimensionless coordinates or host reporting/analysis boundaries. | None. |
| Cue, velocity, and lesion schedules | `braintools.input.Constant` | BrainTools input API | Correct. Protocols are created once under `dt` and passed time-major into the loop. | None. |
| Independent trial State | `vmap_init_all_states` plus semantic Hidden/ShortTerm filters | BrainState initialization and `vmap2` | Correct. Every heading/control condition owns separate writable dynamical State while model parameters and topology remain shared. | None. |
| Condition mapping | One `vmap2(model.update, ...)` | BrainState `vmap2` | Correct. It maps the complete stateful step, declares State input/output axes together, and raises on undeclared writes. | None. |
| Time rollout | One `for_loop` inside `brainstate.transform.jit` | BrainState `for_loop` and `jit` | Correct. There is no Python timestep loop and the stable logical rollout is compiled once. | None. |
| Recurrent delay | Previous completed spike plus an explicit `DT == RECURRENT_DELAY` check | BrainPy-State step lifecycle; general delay APIs unnecessary for exactly one step | Correct for the stated one-step delay. | None. |
| Per-neuron moving rates | Host NumPy trailing-window reduction | Host analysis boundary | Correct. BrainTools `firing_rate` is a population statistic and does not own this per-condition, per-neuron decoder input. | None. |
| Circular population decoder and errors | Host NumPy trigonometry | Host scientific-analysis boundary | Correct. No official BrainX circular head-direction decoder owns this operation. | Add departure and sustained-return reductions to support the claimed categories. |
| CSV and JSON | Python standard library | Host serialization boundary | Correct. | Preserve only fields needed to reconstruct claims. |
| Figure | Direct `plt.subplots`, `imshow`, `plot`, `scatter`, labels, legend, and save | High-level Matplotlib host presentation boundary | Correct and suitably simple. BrainTools helpers do not express this paired circular trajectory and predicate figure more directly. | Change scientific content, not plotting architecture. |
| Regression checks | `unittest` over helpers and full sweep | Host test boundary | Mechanically sound but protects the wrong recovery definition. | Test complete temporal predicates. |

## Missing, bypassed, or misused BrainX APIs

No material BrainX API is missing or misused.

- Do not replace the dense signed kernel with `braintools.conn.Ring`: the
  official `Ring` contract creates uniform neighbor edges, excludes self
  connections, and cannot represent the dense cosine hold kernel or its signed
  derivative pathway.
- Do not replace the custom recurrent composition with a stock projection
  merely to name another API. The velocity-dependent signed communication rule
  is custom model behavior, while BrainPy-State already owns its neuron,
  synapse, output, and current-input lifecycle.
- Do not replace the per-neuron moving-rate tensor with
  `braintools.metric.firing_rate`; that API owns population firing-rate
  smoothing, not the trial-by-neuron activity required for circular decoding.
- Do not replace the direct `pyplot` figure with BrainTools figure scaffolding.
  The generated code already uses the shortest high-level plotting primitives
  that preserve the required comparison and boundary evidence.

## Performance and code simplicity

- The main performance structure is correct: one compiled logical rollout,
  mapped independent State, and event-driven products only at binary spike
  boundaries.
- The full spike tensor is appropriate because time-resolved circular decoding,
  wedge-silence verification, and lesion trajectories all depend on it.
- Host NumPy analysis, CSV/JSON writing, and PNG creation occur after the
  transformed simulation and are legitimate host boundaries.
- The implementation has one model class and direct data flow. The CLI, README,
  CSV, JSON, and tests add artifact volume, but they make the claims auditable;
  no new abstraction is justified.
- The Matplotlib composition is high-level and clear. Retain `plt.subplots` and
  ordinary `Axes` calls; do not add custom `Figure`, `Artist`, style, or layout
  infrastructure when adding the corrected observables.

## Skill improvements

Make one cross-package refinement in `brainx-general-guard` and mirror it in
`plan.md`: a recovery label requires a measured post-intervention departure
followed by sustained return; a condition that never departs must be labeled
spared. Fold this into the existing categorical-map bullet so the scientific
validation section remains compact.

No BrainPy-State, BrainState, BrainEvent, or BrainUnit edit is justified. Run 0
followed their API and execution guidance correctly; the failure is the
cross-package scientific interpretation of intervention outcomes.

## Checks for the next run

- The intact control for every represented start holds a localized bump in
  darkness, follows the full unwrapped turn without extra laps, and remains
  localized afterward.
- All starts and matched control/lesion conditions run with independent mapped
  dynamical State inside one transformed time loop.
- The wedge emits no observable spikes after lesion onset.
- Saved time-resolved matched error distinguishes:
  - `spared`: never crosses the declared perturbation boundary and satisfies
    the endpoint predicate;
  - `recovered`: crosses the perturbation boundary, then satisfies the return
    predicate for the declared sustained window;
  - `failed`: does not satisfy the sustained return/activity/coherence predicate;
  - `invalid_control`: the matched control fails its own full predicate.
- Every continuous value used by these predicates is retained, tested, and
  visible in the figure or an equally direct saved plot.
- The figure remains a short high-level `matplotlib.pyplot` composition with
  readable physical axes, comparison styles, legend, unclipped layout, and
  sufficient resolution.
- The default run and tests complete, and all saved outputs reproduce under the
  frozen environment.
