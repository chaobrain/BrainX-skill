# Train through the NEST Simulator

Use this version-specific path to optimize a continuous membrane-voltage objective directly through `Simulator.simulate()` in BrainSpikes 0.2.0. The forward model remains a NEST-created network; no native-neuron substitute or external learned surrogate replaces it.

## Keep the objective boundary pure

Construct the same deterministic topology inside each objective evaluation, supply the candidate physical weight/current, call `simulate()` for a fresh rollout, and reduce the recorded voltage after an explicit unit conversion. No simulator or transient State escapes the objective.

| API | Use |
|---|---|
| `Simulator.create()` / `connect(..., weight=theta*scale, comm=...)` | Construct the fixed NEST topology with a differentiable numeric candidate and explicit physical scale. Keep connection rules, spike times, and delay fixed. |
| `Simulator.simulate(duration)` | Execute the complete independent NEST rollout for each loss evaluation. |
| `SimulationResult.trace(meter, "V_m")` | Supply the continuous observable used in the verified loss; preserve units until the declared comparison boundary. |
| `jax.value_and_grad(objective)` | Differentiate a pure candidate-to-loss function whose temporary State is entirely local. This is not permission to wrap persistent mutable BrainState graphs with raw JAX. |
| `brainstate.ParamState` and `braintools.optim.Adam` | Store the candidate and apply matching gradients between complete Simulator evaluations. |

Open `scripts/train_nest_simulator.py` for the runnable objective, finite-difference comparison, and optimizer. From this skill root run:

```bash
python scripts/train_nest_simulator.py --comm dense
python scripts/train_nest_simulator.py --comm sparse
```

The canonical fixture uses `spike_generator → iaf_psc_alpha`, a fixed three-spike source, `0.1 ms` timestep/delay, a multimeter, and a `5 ms` rollout. A dimensionless candidate scales a `100 pA` connection weight. The target is a synthetic voltage trace at weight `1.7`; Adam starts at `0.4` and updates the actual Simulator input weight for 100 independent rollouts.

Open `references/braintools/metric.md` for observation units and loss reduction, and `references/braintools/optimizer.md` for candidate registration and update ownership.

## Interpret the measured result

In the pinned Docker image, dense and sparse both recover `1.6985590956`, while mean voltage-trace loss falls from `1.4608960822` to `1.7947458e-6`. At the initial candidate, autodiff gives `-2.2475324342` and centered finite differences give `-2.2475324340`. A separate external-current gradient also agrees with finite differences. Repeated independent rollouts are identical.

This establishes differentiable NEST Simulator weight fitting for the tested continuous objective. It does not establish every model, plasticity rule, recurrent spike path, or scientific parameter identifiability.

## Check whether multiple weights are distinguishable

Choose training spike times that give separate weights different voltage responses. Fit from several preset starts, check each normalized loss gradient against centered differences away from convergence, and check a changed spike schedule only after fitting. Verify subthreshold voltage and zero spikes if the comparison assumes no threshold resets.

Use `jax.jacrev` on the fresh candidate-to-voltage function to form the sample-by-weight Jacobian in physical weight coordinates. Inspect the singular values and the ratio of smallest to largest. A near-zero singular value can permit low voltage error with poor weight recovery. If the Jacobian is well conditioned at the probe and recovery still fails, check optimizer convergence and scaling; local sensitivity does not prove global identifiability.

Open `scripts/train_nest_simulator_two_weights.py` for the tested dense, subthreshold two-input fixture. The same-timing control also matches synapse model and delay. In this additive model, equal total weights give identical voltage despite different splits, and the Jacobian is numerically rank one. Only the sum is identifiable under that control.

## Respect the observed boundaries

- Keep host-side graph construction outside an outer JIT. The tested multi-spike `spike_generator` performs host time-order validation that raises `TracerBoolConversionError` if the entire builder is outer-jitted. The canonical optimizer therefore runs on the host while Simulator owns its rollout execution.
- Do not infer spike-loss learning from voltage-loss learning. The recorded hard spike-count control emits three spikes yet has zero gradient with respect to external current in the tested configuration. Use `references/native-training.md` for the verified surrogate-gradient recurrent SNN path.
- Rebuild the same topology for each independent candidate here. This does not establish differentiable `SynapseCollection.set()` on an existing compiled Simulator, or a reusable fully compiled training step. The separate `window_voltage` fixture in `scripts/nest_boundary_workflows.py` does preserve the tested gradient through two local `cont(5 ms)` calls: final voltage and weight gradient match a single `simulate(10 ms)` run. All source spikes occur in the first window, so the nonzero final gradient crosses that boundary.
- Keep sparse connectivity structure fixed. The experiment differentiates weights, not discrete connection choices or spike times.

## Fit a hard spike count without gradients

The hard-count control still has zero autodiff gradient. `fit_hard_count()` in `scripts/nest_boundary_workflows.py` instead uses `braintools.optim.ScipyOptimizer(..., method="Powell")` with a bounded scalar external-current candidate. It finds five recorded spikes and zero count error in the tested 20 ms fixture. The candidate scales `100 pA` and lies in `[5,20]`.

This is derivative-free fitting, not surrogate-gradient learning. A count objective has plateaus and does not identify a unique physical current; a successful scalar fixture does not establish robust high-dimensional spike-time fitting. Use native surrogate training when temporal gradient credit is required.

## Evidence and public contracts

The repository-maintained Docker receipt and full log record the immutable image, versions, code hashes, positive controls, and negative controls. Reproduce the installed skill's fitting path through its bundled script.

- [Simulator lifecycle and recording](https://brainx.chaobrain.com/brainspike/apis/generated/brainspike.network.Simulator.html)
- [iaf_psc_alpha parameters and update contract](https://brainx.chaobrain.com/brainspike/apis/generated/brainspike.iaf_psc_alpha.html)
- [JAX reverse-mode Jacobian](https://docs.jax.dev/en/latest/_autosummary/jax.jacrev.html)
