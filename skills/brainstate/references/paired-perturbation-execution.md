# Reliable paired perturbation execution

Open this reference when a stateful simulation compares matched baseline and perturbation trials, especially when recurrent dynamics, stochastic input, JIT compilation, batching, or long rollouts make causal matching and resource use easy to get wrong. It settles how to branch from one shared pre-perturbation trajectory, scale the run in fixed-shape chunks, and distinguish execution failure from scientific non-reproduction.

## Preserve the causal comparison

Treat each matched pair as one trajectory until the perturbation begins. Run the common prefix once, snapshot every model State at the intervention boundary, then run the baseline suffix, restore the boundary snapshot, and run the perturbed suffix with the same external inputs.

| API | Description |
|---|---|
| `brainstate.nn.init_all_states(model, **kwargs)` | Use before the shared prefix to initialize the complete Module graph; verify the resulting State shapes before compiling the full experiment. |
| `model.states()` | Use at the intervention boundary to collect every registered State by absolute path; retain parameters, dynamics, synapses, delays, and model-owned random State needed for exact replay. |
| `brainstate.nn.assign_state_values(model, snapshot)` | Use before the second suffix to restore the intervention-boundary snapshot; reject both unexpected and missing paths before continuing. |
| `brainstate.transform.for_loop(step, *xs)` | Use to execute each time-major prefix or suffix while model State carries the recurrent dynamics; return only observations required by the analysis. |

```python
import brainstate


def restore_exact(model, snapshot):
    unexpected, missing = brainstate.nn.assign_state_values(model, snapshot)
    if unexpected or missing:
        raise ValueError(
            f"state restore mismatch: unexpected={unexpected}, missing={missing}"
        )


brainstate.nn.init_all_states(model, batch_size=batch_size)

# Both conditions share this execution and therefore cannot diverge before onset.
prefix_observations = run_prefix(shared_inputs[:onset])
onset_snapshot = {
    path: state.value
    for path, state in model.states().items()
}

baseline = run_suffix(shared_inputs[onset:], perturbation=None)
restore_exact(model, onset_snapshot)
perturbed = run_suffix(shared_inputs[onset:], perturbation=protocol)
```

Generate each trial's stimulus and external noise once, then pass the same arrays to both suffixes. Do not regenerate noise by reseeding inside each branch. Include a model-owned random generator in the State snapshot when the transition itself samples randomness.

Measure the perturbation from the executed trajectory, not only from the requested waveform. For a stimulated neuron, report the intended waveform and the paired change in its spikes or other target response over the declared dose window.

Open `references/collective_model_operations.md` instead when the task only needs general initialization, reset, vmapped lifecycle, or checkpoint restoration without a matched causal intervention.

## Keep transformed execution shape-stable

Use one fixed trial-axis shape for a compiled paired runner. Split a larger experiment into chunks of that shape; pad only the final chunk, mask padded lanes from every statistic, and verify that model operations do not couple trial lanes.

| Mechanism | Action |
|---|---|
| Fixed chunk size | Choose the largest batch that passes the resource preflight, then keep it unchanged throughout the formal run. |
| Final partial chunk | Pad inputs to the fixed shape and carry an explicit validity mask into output collection and analysis. |
| Independent trials | Keep recurrence within each trial lane; do not reduce, normalize, or communicate across the trial axis. |
| Different logical workloads | Use the same runner shape or separate model/runner instances; do not mutate an initialized State graph between incompatible batch layouts. |

Do not increase batch size merely to finish sooner. In a time-unrolled recurrent model, simultaneous trial lanes multiply dynamical State and monitored output storage. Prefer smaller chunks when a full run exits without a Python traceback or produces no artifacts.

## Pass progressive execution gates

Freeze scientific parameters before inspecting perturbation outcomes, then advance through these gates in order. Stop at the first failure and preserve its run as an execution failure.

| Gate | Required check |
|---|---|
| Construction | Import packages, construct the complete model, initialize State, and record State paths, shapes, dtypes, units, and device. |
| One-step | Advance one step and verify output shapes, finite values, State write-back, and expected units. |
| One unperturbed trial | Run one complete trial and verify onset indices, baseline activity, monitoring windows, and artifact writing. |
| One matched pair | Branch at perturbation onset, restore every State without mismatches, verify identical suffix inputs, and measure the target dose. |
| Small experiment | Run at least two perturbation targets and enough conditions to exercise every analysis path; require the owning modeling skill's declared baseline-observability, intervention-dose, finite-value, bin-support, and effective-sample checks to pass without inspecting the requested causal outcome. |
| Scale probe | Run the intended model with one candidate chunk, record elapsed time and peak resource use, then choose and freeze the formal chunk size. |
| Formal run | Create a new immutable run directory, execute all frozen trials, and write parameters, logs, raw summaries, statistics, figures, and run notes. |

Unit tests for analysis helpers do not replace these gates. Add an end-to-end test that constructs the actual model, executes one matched pair, restores State, writes artifacts, and reads them back.

Do not enter the formal-run gate when the small experiment has silent or
saturated observables, degenerate feature estimates, unsupported analysis bins,
or an out-of-tolerance delivered intervention. Classify and preserve that pilot
as an invalid or inconclusive protocol. Calibrating the target dose against a
predeclared delivery rule is protocol development; changing model or stimulus
parameters after inspecting downstream causal signs is outcome-driven tuning.

## Limit monitoring before reducing model size

Estimate the largest retained arrays before execution from `time steps * trial lanes * monitored elements * bytes per element`. Count compiler intermediates and recurrent State separately; the output estimate is a lower bound, not a memory guarantee.

Return aggregates from transformed code when the analysis only needs spike counts, rates, sums, or mismatch counts. Return full time series only for signals that the scientific analysis or a named validation gate requires. Use a one-pair diagnostic run for exact full-trajectory inspection instead of transferring every neuron's trajectory for every formal trial.

If a process disappears without a Python traceback, record the command, exit status or signal when available, last completed gate, artifact inventory, device, batch shape, and available system evidence. Label resource exhaustion as suspected unless the runtime or operating system confirms it.

## Classify the outcome correctly

Assign exactly one primary status before interpreting the science:

| Status | Use when |
|---|---|
| Execution failure | An exception, failed invariant, incomplete artifact set, invalid causal pairing, or unexplained process termination prevents analysis. |
| Invalid protocol | The run completes but the delivered perturbation, timing, baseline regime, sample definition, or analysis differs materially from the declared experiment. |
| Inconclusive | The protocol is valid but uncertainty, target count, effective pair count, firing-rate regime, or bin occupancy cannot support the requested inference. |
| Scientific non-reproduction | The complete frozen protocol is valid and adequately powered, yet one or more preregistered signatures fail. |
| Reproduction | The complete frozen protocol is valid and the preregistered effect-size and uncertainty criteria pass without outcome-driven tuning. |

Do not tune scientific parameters in response to execution failures. Fix only the execution defect in the next run and state exactly what changed. After a valid scientific run, retain negative or inconclusive results and identify the failed mechanistic or statistical checks before proposing a new parameter hypothesis.

## Boundaries and common failures

- Running baseline and perturbation as unrelated trials and assuming equal seeds imply equal recurrent trajectories.
- Reinitializing only membrane voltage while leaving synaptic, refractory, delay, adaptation, or random State unmatched.
- Changing scientific parameters while repairing a JIT, batching, State, or resource failure.
- Treating passing helper tests as evidence that the full transformed workflow runs.
- Treating an empty result directory or silent process exit as a biological null result.
- Reporting a sign in the expected direction as reproduction when its uncertainty includes the null or its effective sample count is too small.
