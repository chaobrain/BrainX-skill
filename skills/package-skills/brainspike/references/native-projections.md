# Build a native conductance network

A chemical connection maps completed source events into a temporal synaptic state, converts that state into physical target current, and deposits the current before the target integrates. Preserve these four roles even when no projection wrapper class is available.

## Compose the four roles

| API | Use |
|---|---|
| `brainevent.BinaryArray(events) @ weights` | Map fixed binary source events to target conductance increments; the example uses small dense wiring. |
| `ExponentialConductance` in `scripts/native_workflows.py` | Decay a target-aligned linear trace and add the current increment; allocate one state value per target. |
| `g * (E - V_post)` | Convert a nonnegative conductance into signed current using the old target voltage and channel reversal. |
| `LIF(current)` in `scripts/native_workflows.py` | Advance the target exactly once after all incoming currents are assembled. |

Run the complete canonical example from this skill root:

```bash
python scripts/native_ei_network.py
```

`EINetwork` registers separate excitatory/inhibitory point populations and two postsynaptic conductance filters. It snapshots both populations' previous events and voltages, communicates all source events, filters both receptor contributions, computes their currents, then advances both populations. The small dense Bernoulli wiring allows autapses; remove the corresponding diagonal entries explicitly when the scientific design forbids them. Filters own `N_exc+N_inh` values each rather than one value per edge. The returned monitors expose conductance, signed current, voltage, and binary events.

The fixture uses positive excitatory and inhibitory conductance magnitudes. Excitation has reversal `0 mV`; inhibition has reversal `-80 mV`. Inhibitory current is negative only while target voltage is above that reversal. Do not encode inhibition by making a conductance negative. For a current-based network, use a current-valued filter/weight contract; do not change only the unit of this conductance filter. The NEST COBA/CUBA alternative is `references/nest-compatible/scripts/brette_et_al_2007.py`.

## Preserve the alignment conditions

Target-aligned reduction is exact for a linear exponential filter when all merged inputs to that filter share its time constant and output contract. Use separate filters for distinct reversal potentials or kinetics. Do not merge nonlinear receptors or heterogeneous edge histories merely to reduce memory.

Source-aligned sharing requires identical outgoing kinetics for each shared source trace. Connection-specific nonlinear dynamics may require per-edge State. Treat old AlignPre/AlignPost descriptors as retired APIs; derive State placement from the equations and verify equivalence before implementing an advanced composition.

Open `references/nonlinear-state-sharing.md` before reusing nonlinear source traces; it contains the homogeneous edge-reference comparison and explicit counterexamples to invalid aggregation.

## Verify before scaling

The Docker checks inject known previous-step events and verify conductance increments, current signs, and target voltage on the same update, then check exponential decay and independent reset. The short E/I rollout checks finite traces and nonzero activity. It is a mechanics fixture using a custom Euler LIF without an absolute refractory period; it does not reproduce the old COBA benchmark or claim a calibrated network regime.

When porting an existing COBA model, replay the same edge, weight, initial voltage, and drive arrays in both implementations. Compare post-update conductance and voltage on the same clock step under separate subthreshold E and I impulses. Then drive an isolated neuron through consecutive spikes: a subthreshold match cannot check the reset phase or the refractory boundary. In the BrainPy 2.9.0 `LIFRef` soft-reset path, the spike is visible on the threshold-crossing step and reset appears on the next update; probe the exact refractory endpoint on the port's time grid. When matching exponential Euler, include the COBA conductance in the voltage slope if the baseline integrates its voltage dependence within the step. Recheck these behaviors in the pinned runtime before comparing recurrent population statistics.

Open `references/native-delays.md` before adding transmission history. Open `references/model-and-readout-selection.md` before choosing physiology or temporal readout. Open `brainevent` → `references/sparse-formats.md` when scaling beyond the small dense connectivity fixture.

## Sources

- [BrainSpike neuron contract](https://brainx.chaobrain.com/brainspike/apis/generated/brainspike.Neuron.html)
- [BrainSpike synapse contract](https://brainx.chaobrain.com/brainspike/apis/generated/brainspike.Synapse.html)
