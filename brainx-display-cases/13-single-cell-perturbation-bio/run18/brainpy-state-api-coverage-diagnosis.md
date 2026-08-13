# BrainX diagnosis: V1 single-cell perturbation

## Evidence studied

- Generated artifacts: `../run17/v1_influence.py`, `../run17/parameters.json`,
  `../run17/results.json`, `../run17/pair_influences.csv`, `../run17/REPORT.md`,
  and this run's `audit.json`.
- Owning guidance: `skills/brainx-general-guard/SKILL.md`,
  `skills/brainpy-state/SKILL.md`, `skills/brainevent/SKILL.md`,
  `skills/brainstate/SKILL.md`, and `skills/brainunit/SKILL.md`.
- Closest executable examples: `skills/brainpy-state/references/scripts/103_COBA_2005.py`
  and `skills/brainevent/references/scripts/coba_ei_teaching.py`.
- Relevant contracts indexed by `source_html_references/`: BrainPy-State E/I
  balanced-network, neuron, projection, and input APIs; BrainEvent event-array,
  sparse-data, and operation APIs; BrainState State, transform, environment,
  and collective-operation APIs.

## Executive diagnosis

Run 17 is an execution-complete negative result, but it is not an informative
test of the requested influence signatures. The excitatory population is nearly
silent, stimulus tuning is degenerate, signal-correlation bins are almost empty,
and the delivered target dose is less than one tenth of the cited mean. The
paired causal execution is valid, yet the scientific observables have too little
dynamic range to identify the requested effects. The next skill revision must
make these checks preconditions of the formal experiment instead of statistics
discovered only after 512 matched pairs have run.

## Scientific problems

| Severity | Artifact location | Problem | Consequence | Correction |
|---|---|---|---|---|
| Critical | `../run17/results.json` baseline rates | Tuning-trial E activity is 0.113 Hz, with median 0 Hz and 60.6% silent neurons, while I activity is 28.87 Hz. | Non-target spike-count differences are almost always zero, so suppression, distance dependence, and feature dependence are not identifiable. | Reject the pilot before formal execution unless the declared response statistic has sufficient nonzero observations and the E/I regime is scientifically defensible. |
| Critical | `audit.json` signal correlation | Six quantile bins contain `[606, 0, 0, 0, 0, 4448]` observations and only 7.97% of neuron-direction means are nonzero. | The correlation regression and plot are dominated by tied, degenerate tuning vectors rather than graded feature similarity. | Check unique tuning profiles, finite correlations, bin occupancy, and target coverage during a small unperturbed pilot. |
| Critical | `../run17/results.json` photostimulation | The measured target increment is 0.527 spikes in 250 ms versus the cited 6.38 +/- 1.01 benchmark. | The perturbation is too weak to test the published intervention even though its waveform timing is correct. | Calibrate the intervention against the target-neuron dose using pilot trials before freezing the formal protocol; report the searched candidates and acceptance rule without inspecting downstream influence signs. |
| High | `audit.json` preference counts | 68.4% of estimated preferences collapse to 0 degrees. | Target-matched versus orthogonal comparisons mostly reflect arbitrary tie breaking. | Require preference coverage and tuning reliability before selecting targets or defining match bins. |
| Medium | `../run17/v1_influence.py` | The entry point imports the full implementation from `../run11`, so the archived run is not standalone. | A later reader cannot execute the run from its own immutable directory without earlier history. | Future agents must archive a self-contained runnable script in the new run directory. |

## Complete BrainX API coverage map

| Generated code or responsibility | Current approach | Owning BrainX API or host boundary | Assessment | Improvement |
|---|---|---|---|---|
| Conductance-based point neurons | BrainPy-State LIF populations | `brainpy.state.LIFRef` | Appropriate. | Keep explicit initialization and units. |
| Recurrent E/I synapses | BrainPy-State projections with event communication | BrainPy-State projection APIs plus BrainEvent connectivity | Appropriate in principle. | Preserve the event-driven path and validate the realized operating regime. |
| Physical parameters | BrainUnit quantities at simulation boundaries | BrainUnit | Appropriate. | Keep raw values only at analysis and serialization boundaries. |
| Stateful rollout | `brainstate.transform.for_loop` with preallocated batch State | BrainState transforms and collective lifecycle APIs | Correct after the run17 lifecycle fix. | Retain fixed-shape State and exact reset/restore checks. |
| Matched causal trials | Identical external arrays and baseline/perturbation lanes | BrainState paired State lifecycle plus host-side input generation | Correct. | Prefer one shared prefix and exact onset snapshot when feasible. |
| Pilot validity checks | No blocking checks before the formal run | Host-side scientific validation around BrainX outputs | Missing. | Add explicit activity, tuning, bin-support, target-dose, and artifact gates. |
| Statistics and bootstrap | NumPy/Pandas-style host analysis | Legitimate host boundary | Appropriate. | Define effective sampling units and minimum support before formal execution. |
| Figures and reports | Matplotlib and JSON/CSV/Markdown | Legitimate host boundary | Appropriate. | Mark empty or tied bins and preserve all gate results. |

## Missing, bypassed, or misused BrainX APIs

No missing BrainX numerical API explains the scientific failure. The important
gap is workflow guidance: BrainPy-State produces the relevant spike and State
observables, but the skill did not require the agent to demonstrate that those
observables can identify the requested hypotheses before scaling the run.
BrainState's existing progressive gates likewise verified execution without
requiring scientific dynamic range. Ordinary tuning statistics, calibration,
bin support, bootstrap analysis, and reporting remain valid host boundaries.

## Performance and code simplicity

The fixed 32-lane transformed rollout completed and all 512 causal pairs passed,
so batching is no longer the limiting defect. The formal run was nevertheless
wasteful because a small pilot could have detected silent E neurons, degenerate
tuning, empty correlation bins, and inadequate target dose. Future runs should
return aggregate spike counts for gates and stop immediately at the first failed
scientific-validity condition.

## Skill improvements

1. Add one BrainPy-State reference for intervention-experiment validity gates:
   establish the unperturbed operating regime, tuning reliability and support,
   and target dose before freezing and running downstream causal tests.
2. Route perturbation and matched-trial tasks to that reference from the
   BrainPy-State root skill.
3. Tighten BrainState's paired-execution reference so a small experiment must
   pass declared scientific observability and protocol checks before a formal
   run; distinguish dose calibration from outcome-driven tuning.

## Checks for the next run

- The agent creates a self-contained script and runs progressive construction,
  one-trial, one-pair, and small-pilot gates before the formal experiment.
- It records acceptance rules and pilot summaries for E/I firing, silent
  fractions, tuning reliability or dynamic range, preference and bin support,
  and delivered target spike increment.
- It calibrates only intervention delivery and baseline observability before
  freezing the formal protocol; it does not inspect or optimize influence signs.
- It stops and reports an invalid or inconclusive protocol if a gate fails.
- A formal run, if reached, retains exact within-pair inputs and State, reports
  effective target and bin counts, and preserves negative scientific outcomes.
