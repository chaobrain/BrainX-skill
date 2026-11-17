# Delay native point-neuron inputs

Delay the source signal before communication and synaptic filtering. Prove arrival at the actual conductance and target-voltage boundary; a buffer-only check or postsynaptic spike time alone does not establish the transmission delay.

## Select the delay owner

| API | Use |
|---|---|
| `brainspike.Simulator.connect(..., delay=...)` | Use for a NEST graph; preserve its own device and transmission timestamp conventions. |
| `brainstate.nn.Delay` | Use for explicit native signal history; initialize it with the rest of the Module graph. |
| `DelayBank` in `scripts/delay_workflow.py` | Use the verified three-source named-tap example with delays `[0,2,3]*dt`. |
| `DelayedProjection` in `scripts/delay_workflow.py` | Follow the complete external-event → buffer → communication → conductance → target update path. |

Run the named-tap example from this skill root:

```bash
python scripts/delay_workflow.py
```

For `DelayedProjection`, initialize the graph and drive a boolean `(time,3)` input through `brainstate.transform.for_loop`. Its source event on channel 1 at input step zero first changes conductance and target voltage at step 2. An independent quiet rollout after whole-graph initialization has zero conductance. The target is subthreshold in the impulse check, so refractory/threshold effects do not obscure arrival.

Open `brainstate` → `references/brainstate/delay-buffers.md` for the authoritative write-before-read convention, named entries, fixed timestep, reset, and mapped-pointer constraints. Keep those buffer rules in BrainState; this reference owns their point-neuron integration.

The native fixture consumes an externally available event in its current update. In a recurrent network that first reads a previous completed source event, include that availability offset when interpreting latency. Record source emission, buffer input, conductance onset, and voltage response separately. Do not shift a tap to hide an unexplained one-step discrepancy.

Use integer grid taps for binary events. Off-grid interpolation, arbitrary per-edge delays, and mapped heterogeneous pointer State require separate verification. Separate delayed NEST connections are already checked, but that result does not validate this native buffer path or restore old delayed-prefetch/projection APIs.

Open `references/native-projections.md` for conductance and current order, and `references/nest-compatible/network-building.md` for NEST delay/lifecycle control.

Open `references/nest-delay-boundaries.md` for the separately tested NEST vector delay, matrix grouping, fractional current transmission, and explicit-spec failures.

## Sources

- [BrainState delay protocol](https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/02_synaptic_delays.html)
