# BrainX diagnosis: 07 Cortical Wave

## Evidence studied

- Generated artifacts: `cortical_wave_obstacle.py`, `README.md`,
  `requirements.txt`, `outputs/phase_metrics.csv`,
  `outputs/cortical_wave_obstacle.png`, `agent-final.md`,
  `harness-metadata.txt`, `codex-events.jsonl`, and `codex-stderr.log`.
- Execution: the archived entry point was copied to a separate temporary
  directory and run with the required BrainX virtualenv. The reproduced PNG
  and CSV were byte-identical to the archived outputs. Independent checks
  confirmed matched-control propagation, E/I lesion silence, category-rule
  consistency, and zero E or I spikes when the spark amplitude was set to
  zero.
- Owning guidance: `skills/brainx-general-guard/SKILL.md`,
  `skills/brainpy-state/SKILL.md`, the BrainPy-State connectivity,
  input-current, component-selection, and projection references, the
  BrainState mapped-State reference, and the BrainEvent sparse-format
  reference.
- Closest executable examples:
  `skills/brainpy-state/references/scripts/sound_localization.py`,
  `skills/brainevent/references/scripts/coba_ei_teaching.py`, and
  `skills/brainpy-state/references/scripts/103_COBA_2005.py`.
- Authoritative API pages: BrainTools `Grid2d`, `ConnectionResult`, and input
  `Constant`; BrainEvent `BinaryArray`, `coo2csr`, and `CSR`; BrainPy-State
  `LIFRef`, `Expon`, and `COBA`; BrainState `vmap_init_all_states`, `vmap2`, and
  `for_loop`.

## Executive diagnosis

Run 2 is reproducible and its BrainX implementation path is strong. It uses a
named BrainTools grid topology, preserves `coo2csr()`'s weight permutation,
generates the spark with BrainTools, maps independent dynamical State with
`vmap2`, and advances all conditions through one transformed time loop. Every
matched control reaches the far edge in causal order, the silent patches emit
no spikes, the no-spark baseline is quiescent, and the displayed wave front is
spatially coherent.

The remaining defect is scientific presentation rather than package API use.
The activity snapshots show only one lesion condition, so they do not show the
same-inhibition intact wave that establishes the intervention effect. The
categorical map is defined by transmission, flank balance, and wake delay, but
the figure visualizes only transmission. The CSV retains the raw flank and wake
observables, yet a reader cannot inspect the bend/split boundary without
reconstructing two derived measures. One latent predicate also permits a
`splits` label after only one finite flank arrival, although all current split
records happen to have both arrivals.

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| P1 | `cortical_wave_obstacle.py:385`, PNG top row | The wave snapshots show only the radius-0.28 mm lesion at inhibition 1.00; no radius-zero rollout at inhibition 1.00 is shown at matched times. | The figure does not visually establish how the intervention changed an otherwise propagating wave, despite the prompt asking to show the wave crossing the sheet and then place a silent patch in its path. | Show intact and lesion activity at the same inhibition and aligned physical times or event landmarks, with an explicit condition label on every panel. |
| P1 | `cortical_wave_obstacle.py:328`, `cortical_wave_obstacle.py:464`, PNG lower row | `dies` depends on transmission, while `bends` versus `splits` depends on flank balance and wake delay; only transmission is visualized. | The categorical map is not visually supported at the bend/split boundary. Raw CSV columns permit reconstruction but do not satisfy the figure-level evidence contract. | Retain explicit `flank_balance` and `wake_delay_ms` fields and visualize every boundary-defining continuous observable, or visualize a documented continuous boundary margin that preserves all terms. |
| P2 | `cortical_wave_obstacle.py:332` | `route_order` requires at least one finite flank arrival, while the documented split rule says both flanks must propagate. `balance >= 0.55` does not explicitly require two finite arrivals. | A future calibrated condition could be labeled `splits` without observed arrival through both flanks. The current saved split records are not affected because both arrivals are finite. | Require two finite flank arrivals before assigning `splits`, and verify the complete label predicate for every record. |

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Unit-aware model and sheet parameters | BrainUnit quantities for time, voltage, current, conductance, resistance, and distance | BrainUnit | Correct | Preserve. |
| Two-dimensional coordinates and lesion distances | `u.math.arange`, `meshgrid`, `sqrt`, and unit-aware comparisons | BrainUnit math | Correct | Preserve. |
| Local cortical topology | `braintools.conn.Grid2d(connectivity="moore", periodic=False)` | BrainTools connectivity | Correct and semantically clearer than manual neighbor indexing | Preserve. |
| Topology-to-event handoff | `coo2csr(pre_indices, post_indices)` followed by `result.weights[order]` | BrainEvent `coo2csr` and `CSR` | Correct; the required edge permutation is preserved | Preserve. |
| Sparse spike communication | `BinaryArray(spikes) @ connectivity` | BrainEvent | Correct event-driven path | Preserve. |
| E/I point-neuron dynamics | Two `brainpy.state.LIFRef` populations with explicit initialization | BrainPy-State | Correct for the represented scale | Preserve. |
| Conductance kinetics and current law | `Expon` plus `COBA`, registered through `add_current_input` | BrainPy-State | Correct and consistent with the E/I teaching example | Preserve. |
| Silent-patch intervention | Mask both E and I emitted spikes while retaining the surrounding topology | Model-owned mechanism | Correct host/model boundary; no package API owns this scientific intervention | Add a focused assertion to the final verification path if useful. |
| Spark protocol | One baseline-pulse-baseline `braintools.input.Constant` generated under `dt` | BrainTools input | Correct; named sections are not rebuilt with time predicates | Preserve. |
| Independent sweep State | `vmap_init_all_states` plus semantic Hidden/ShortTerm State filters in `vmap2` | BrainState | Correct; each condition owns dynamical State and the current sample is shared with `in_axes=None` | Preserve. |
| Time evolution and compilation | One `for_loop(step, times, spark)` inside one `brainstate.transform.jit` callable | BrainState | Correct and efficient | Preserve. |
| Monitoring | Return E/I spikes from the mapped step and let `for_loop` stack them | BrainState transform output | Correct; no separate monitor API is required | Preserve. |
| Arrival, peak, transmission, and category analysis | NumPy after the BrainX rollout | Legitimate host-side analysis boundary | Correct boundary, but the split predicate and explicit derived fields need tightening | Require both flanks and store the derived continuous measures. |
| CSV persistence | `csv.DictWriter` | Host serialization boundary | Correct | Preserve. |
| Figure generation | High-level Matplotlib image, text, patches, and colorbars | Host presentation boundary | Appropriate API level, but comparison and continuous-evidence coverage are incomplete | Add matched condition panels and visualize all category boundaries without unnecessary low-level plotting machinery. |
| Command-line output directory | `argparse` and `pathlib` | Host interface boundary | Acceptable for a runnable artifact | Preserve. |

## Missing, bypassed, or misused BrainX APIs

No material BrainX modeling API is missing, bypassed, or misused in the final
artifact. In particular, Run 2 uses the exact BrainTools current and topology
families, the correct BrainEvent weight permutation, independent mapped
BrainState, and one transformed time loop. NumPy analysis, CSV output, CLI
handling, and the custom multi-panel figure are legitimate host boundaries.

The remaining failures belong to scientific evidence construction. They do not
justify replacing the successful connectivity, input, neuron, synapse, State,
or transform APIs.

## Performance and code simplicity

- One CSR topology is built once and reused across E-to-E, E-to-I, and I-to-E
  paths.
- All 25 independent radius/inhibition conditions run in one state-aware map;
  there is no Python simulation loop over conditions.
- One transformed `for_loop` owns all 500 timesteps, and one stable JIT boundary
  owns the complete rollout.
- The per-condition Python loop is confined to post-rollout host analysis.
- Monitoring two boolean spike arrays is proportionate to the requested spatial
  snapshots and causal region metrics.
- High-level Matplotlib is appropriate for the custom phase-map composition.
  The needed evidence panels can be added without introducing another plotting
  abstraction.

## Skill improvements

Make only a cross-cutting refinement in `skills/brainx-general-guard/SKILL.md`:

1. Clarify that a multi-category phase map must retain and visualize every
   continuous observable used by every boundary. A single metric does not
   support labels defined by additional thresholds. Require each categorical
   predicate to encode its full stated conjunction.
2. Require a figure claiming an intervention effect to show the matched control
   and intervention under the same nuisance settings and aligned times or event
   landmarks, not only a normalized summary statistic.

Mirror this refinement in the `brainx-general-guard` section of `plan.md`.
Do not change the BrainPy-State connectivity or input-current guidance: Run 2
demonstrates that both now route and execute correctly.

## Checks for the next run

- Use the byte-identical 836-byte prompt under the frozen model, effort,
  virtualenv, CLI, config, isolation, and installed-skill conditions.
- Retain `Grid2d`, `result.weights[order]`, `braintools.input.Constant`,
  independent mapped dynamical State, shared per-step current, and one
  transformed time loop.
- Reproduce matched no-lesion controls at every inhibition value, a quiescent
  no-spark baseline, E/I lesion silence, and causal source-to-route-to-target
  order.
- Verify every saved label against its complete predicate; `splits` must require
  two finite flank arrivals.
- Retain explicit transmission, flank-balance, and wake-delay values, and make
  every categorical boundary inspectable in the figure.
- Show matched intact and lesion activity under the same inhibition and aligned
  times or event landmarks.
- Confirm that the figure still shows a coherent propagating front and that the
  category map agrees with the continuous evidence.
