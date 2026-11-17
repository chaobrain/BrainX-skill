# Compose point, cellular, and regional dynamics

Use one native Module graph when BrainSpike point neurons, BrainCell cellular mechanisms, and BrainMass regional dynamics exchange observables during a run. Each package still owns its equations; the wrapper owns only the exchange contract.

Open `scripts/multiscale_workflow.py` for `CoupledScales`: a custom BrainSpike LIF, a BrainCell HH cell with sodium/potassium/leak currents, and a BrainMass Hopf region. The fixture uses the same `0.1 ms` clock and reads all source observables before updating targets.

| API | Use |
|---|---|
| `brainstate.nn.Module` | Own the component graph and read previous source values before any target advances. |
| `brainmass.Simulator(wrapper, dt=...).run(..., monitors=...)` | Run a compatible native wrapper and collect post-step State without rebuilding the time loop. |
| `point.get_spike() / dt` | Convert previous-step spike counts to an instantaneous rate with physical inverse-time units; choose an explicit rate scale before supplying a dimensionless regional drive. |
| `cell.update(current_density)` | Advance the cellular component with current density when its channel currents are densities; do not add bare total current without a defined area conversion. |
| `mass.update(x_inp=..., y_inp=...)` | Supply the Hopf component's declared input channels; do not overwrite its hidden State to inject a drive. |

For the bundled fixture, use `brainmass.Simulator(CoupledScales(gain=1), dt=0.1*u.ms).run(...)`. Observe `point.V`, `cell.V`, and `mass.x` through named callable monitors. The gains are explicit mechanics examples and require scientific calibration in a real study.

Set `gain=0` and compare against separately driven components under the same seed and initialization. Injected point spikes may influence the region only on the next step; verify the first allowed arrival. Use a declared window/filter for scientific population-rate estimates, and test finer clocks when discretization affects conclusions.

BrainCell `0.2.0` in the tested image imports BrainPy transitively. The application code uses public BrainCell/BrainSpike APIs, but this three-scale fixture does not establish a BrainPy-free runtime. Native point-neuron and training tests block BrainPy imports independently.

Open `brainx-modeling-loop` → `references/multiscale-coupling.md` for exchange-window, resampling, multiple-clock, and scientific-delay decisions. The common-clock fixture does not validate arbitrary multirate coupling or insertion of custom components into the NEST Simulator.

Open `references/multirate-coupling.md` for the separately verified integer-clock point/regional exchange, count conservation, availability timestamps, and lack of demonstrated monotonic refinement.

## Official contracts

- [BrainSpike Neuron](https://brainx.chaobrain.com/brainspike/apis/generated/brainspike.Neuron.html)
- [BrainMass XY oscillator input channels](https://brainx.chaobrain.com/brainmass/reference/generated/brainmass.XY_Oscillator.html)
