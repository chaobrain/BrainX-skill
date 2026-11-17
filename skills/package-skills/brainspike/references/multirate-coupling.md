# Exchange point and regional signals on two clocks

Use a declared integer clock ratio when point neurons require fine updates and a regional model receives windowed population input. The exchange rule is part of the model: a held signal and a window-averaged rate are not automatically equivalent to continuously coupled dynamics.

## Preserve availability and counts

| API | Use |
|---|---|
| `MultirateCoupling` in `scripts/advanced_native_patterns.py` | Run five `0.1 ms` point-neuron updates inside one `0.5 ms` regional window in the canonical fixture. |
| `brainstate.transform.for_loop` | Advance the fixed number of fine steps with State tracking; snapshot the previous regional signal before the window. |
| `brainstate.environ.context(dt=..., t=...)` | Scope fine and coarse clocks explicitly around their owning model updates. |
| `brainmass.Simulator(wrapper, dt=coarse_dt).run(...)` | Let the standard runner own independent initialization and coarse-window execution. |

Run `python scripts/advanced_native_patterns.py` from this skill root. The wrapper holds the previous completed regional signal for all fine steps, sums their point spikes, divides by the full window duration, and converts the resulting physical rate into the regional model's declared dimensionless drive. The region advances only after that window has completed. Its new signal becomes available to point neurons in the following window.

The runner's `ts` labels window starts in this fixture. Use the explicitly monitored `end_time` when interpreting the newly completed regional State; do not assign its value to a time before the source window was available.

Docker checks verify exact spike-count/rate conservation, the one-window hold relation, no feedback from a future regional update, independent replay, and the zero-coupling limit against an independently driven point neuron. The fixture includes point and regional models; it does not add a separate slow-clock detailed-cell solver.

## Separate correct exchange from numerical equivalence

Changing only the regional exchange interval from `0.5` to `0.2` to `0.1 ms` produced final regional values approximately `0.05951`, `0.05745`, and `0.05886`. They are finite but do not show monotonic refinement in this threshold-driven fixture. Do not claim timestep independence, convergence, or equivalence to an earlier continuously coupled model from the conservation/causality checks.

Non-integer clock ratios, asynchronous interpolation, arbitrary multirate cellular integration, and other exchange kernels require their own implementation and scientific validation. Re-run refinement and matched controls under the intended exchange rule before interpreting results.

Open `references/multiscale-composition.md` for the separately verified common-clock point/HH-cell/region graph. Open `brainx-modeling-loop` → `references/multiscale-coupling.md` for scientific units, observables, windows, and delay decisions.

## Sources

- [BrainState control flow](https://brainx.chaobrain.com/brainstate/tutorials/transformations/05_control_flow.html)
- [BrainMass regional inputs](https://brainx.chaobrain.com/brainmass/reference/generated/brainmass.XY_Oscillator.html)
