# Migration boundary

Use `brainspike` as the sole point-neuron skill. Choose the NEST-compatible Simulator or public Neuron/Synapse composition; do not rename `brainpy.state` symbols mechanically.

## Select the supported workflow

- Use the root skill for point-neuron simulation, synapse specifications, devices, spatial connectivity, and PyNEST porting.
- Open `brainstate` for generic State, Module, control-flow, gradient, and optimizer lifecycle. For the verified native recurrent graph, open `references/native-training.md`; open `references/simulator-training.md` for the separately verified NEST continuous-voltage objective.
- Open `braintrace` for the verified native pp-prop path in `references/native-training.md`.
- Open `braincell` for detailed cellular mechanisms and `brainmass` for aggregate dynamics. For coupling, define exchanged observables, units, clocks, and causal order, then use `references/multiscale-composition.md` for the verified common-clock native wrapper.

Open the local `references/braintools/` guides through the root routing table for BrainSpike-specific inputs, initialization, losses, optimization, surrogates, encoding, and topology. BrainState owns State and transforms rather than these package integration decisions.

## Select the verified replacement

| Retired workflow | Replacement decision |
|---|---|
| LIF and exponential/current/conductance synapse networks | Select the appropriate documented NEST model and synapse; reconstruct parameters and validate startup, units, delays, and recorded outputs. Do not assume identical equations or discretization. |
| AlignPre/AlignPost, descriptors, manual projection stepping | Express supported connectivity through `Simulator.connect`; use `references/native-projections.md` for complete target-aligned native E/I filtering and current conversion. Open `references/nonlinear-state-sharing.md` for verified nonlinear source sharing and counterexamples to nonlinear target aggregation/heterogeneous reuse. Keep edge-specific State when sharing conditions fail. |
| Custom Dynamics or neuron/synapse definitions | Use public `brainspike.Neuron` and `Synapse`; `scripts/native_workflows.py` verifies an Euler LIF and target-aligned exponential conductance. A zero-argument autonomous custom LIF also runs under Simulator scheduling; arbitrary incoming-synapse adapters are not established. Open `references/nest-compatible/network-building.md` for that boundary. |
| Surrogate-gradient BPTT and trainable readouts | Use `references/native-training.md` and `scripts/train_native_snn.py`; the recurrent spike graph and readout both receive nonzero gradients and update under BPTT. |
| BrainTrace pp-prop integration | Use ETP-aware native projections and pp-prop; Docker checks compiler participation, recurrent/readout gradients, reduced BPTT gradient direction, and decreasing fixture loss. |
| Heterogeneous delay banks, gap junctions, or multiscale coupling | Use NEST connectivity references for model-specific delay and electrical synapse semantics. Separate NEST connections verify two distinct target delays; native symmetric gap currents verify sign/conservation behavior. Native point/cell/region exchange passes zero-coupling and one-step-causality checks. Named native delay taps and the complete conductance/voltage path are verified in `references/native-delays.md`. Open `references/nest-delay-boundaries.md` for the tested delay vector, grouped matrix fallback, fractional PSC behavior, and spec failures. Open `references/multirate-coupling.md` for verified causal/conservative integer-clock exchange and its numerical-equivalence limits. |
| Legacy BrainPy analysis and maintenance | Outside the published skill set. Treat migration as a separate behavior-preservation task; never silently restore the retired API. |

The unified kernel remains private, but public Neuron/Synapse bases already support the verified native workflows. Do not conflate kernel availability with base-class composition. NEST Simulator continuous-voltage weight fitting is separately verified in `references/simulator-training.md`; hard recorded spike counts still have zero autodiff gradients, but bounded derivative-free count fitting succeeds. One/two-window continuous objective gradients agree in the local-construction fixture. Fully compiled persistent training, arbitrary delay forms/kinetics, and timestep-independent multirate equivalence remain unproven.

## Separate skill removal from runtime removal

BrainSpikes `0.2.0` declares no direct BrainPy or BrainPy-State dependency. BrainX `2026.10.25` still bundles BrainPy `2.9.0`. BrainCell `0.2.0` also imports BrainPy transitively in the tested image. Removing the skill does not change those runtime dependencies. Install `brainspikes`, import `brainspike`, and open `brainx-install` → `references/compatibility-and-release-matching.md` before changing runtime packages.

Validate migrated startup State, units, timestep, connection direction, delay/weight semantics, recordings, and stochastic tolerances. Open `references/nest-compatible/divergence-and-parity.md` before asserting NEST equivalence.

## Official evidence

- [BrainSpikes 0.2.0 release metadata and description](https://pypi.org/pypi/brainspikes/0.2.0/json)
- [BrainSpike public documentation](https://brainx.chaobrain.com/brainspike/index.html)
- [BrainX release change log](https://brainmodeling.readthedocs.io/CHANGELOG.html)
