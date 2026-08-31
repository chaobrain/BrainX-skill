# BrainX study record

## Selected route

- Represented scale: one isopotential conductance-based muscle cell with membrane and channel-gate dynamics.
- Scale owner: `braincell.SingleCompartment`; no point-neuron network or neural-mass package is represented.
- Supporting packages: BrainUnit for mV, pA, nS, pF, and ms quantities; BrainState for channel State, parameter State, environment time, reset lifecycle, JIT, and transformed time loops; BrainTools for bounded SciPy optimization.
- Active optional coverage: parameter fitting.

## API and lifecycle decisions

- Each named current is a `braincell.channel._base.HH` channel attached directly to the `HHTypedNeuron` root. This is appropriate for an isopotential custom current with fixed reversal potential and independent first-order gates.
- `Gate` declarations own gate powers; `f_<gate>_inf/tau` or `alpha/beta` own kinetics; `current()` returns `g * gates * (E - V)` so positive current is inward under BrainCell's convention.
- Conductances are `brainstate.ParamState` quantities read through `.value`; voltage and gates are BrainCell-managed dynamical State. Independent candidate/protocol boundaries call `reset_state()` without resetting parameter State.
- `SingleCompartment` uses total capacitance, total conductances, and total injected current consistently. This avoids an unknown-area conversion because the recordings provide total pA but no cell geometry.
- A complete rollout uses `brainstate.environ.context(dt=...)` and `brainstate.transform.for_loop(step, times, currents)`. The step sets `t`, advances `cell.update(I_ext)`, and records voltage plus named currents.
- `ind_exp_euler` is selected for repeated long fitting rollouts. Final evidence includes dt versus dt/2 parity; no biological conclusion is taken from an unconverged solver trace.

## Channel mapping

| Requested label | BrainCell implementation | Fixed qualitative role |
|---|---|---|
| Na | custom fast activation/inactivation HH channel | rapid inward depolarization |
| Kr | custom delayed-rectifier HH channel | spike repolarization |
| SHK-1 | custom transient activation/inactivation HH channel | early outward current |
| EGL-19 | custom L-type-like activation/inactivation HH channel | slower inward calcium-like current |
| SLO-2 | custom slow activation HH channel | delayed outward adaptation surrogate |
| Leak | custom ohmic BrainCell channel | passive resting current |

The installed package has literature-derived mammalian channel families but no Du et al. (2025) C. elegans mechanisms. The custom kinetics therefore preserve the requested HH roles and labels but are a phenomenological surrogate, not an exact paper-equation port.

## Fitting design

- Explicit parameter order: `g_na`, `g_kr`, `g_shk1`, `g_egl19`, `g_slo2`, `g_leak`, all in nS with finite positive bounds.
- Model parameters are applied, runtime State is reset, the exact 25 pA protocol is run, and voltage MSE is returned for every objective call.
- `braintools.optim.ScipyOptimizer` is used because Nevergrad is not installed. A derivative-free bounded Nelder-Mead search is used after the full 10,000-step reverse-mode L-BFGS-B smoke test exhausted the process; bounds are numeric at the optimizer boundary and reconstructed as nS quantities in the objective.
- The initial candidate is a fixed translated/slowed HH baseline. Candidate selection uses training MSE only. Held-out traces are evaluated once after fitting.
- Mechanics checks: parameter order/round trip, current-window construction, unit dimensions, State-reset deterministic replay, finite nominal/boundary rollouts, and objective improvement.
- Recovery: use the same 25 pA protocol and fitting path on a synthetic trace generated from a perturbed subset. Failure to recover blocks mechanistic interpretation but not a narrowly predictive result.

## Validation and evidence

- Waveform metrics: RMSE, MAE, Pearson correlation, and baseline/stimulus/recovery RMSE in mV.
- Event metrics: upward -10 mV crossings with 5 ms refractory, spike count, first-spike latency, and peak voltage.
- Held-out consistency requires finite traces and nondecreasing spike count with increasing current. All conditions are preserved regardless of outcome.
- Zero-current control must remain below the event threshold. Fitted and initial traces are both retained.

## Sources studied

- BrainX modeling loop and parameter-fitting workflow.
- BrainX general guard.
- BrainCell root skill, custom channel authoring, channel library, mixed-ion adaptation pattern, area-scaled HH pattern, solver effects, and official HH fitting composition.
- BrainUnit root skill.
- BrainState root skill and Braintools optimizer selection reference.
