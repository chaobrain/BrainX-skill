# Generate point-neuron input currents

Use `braintools.input` to construct a physical stimulation protocol once under the model timestep. The generated array is time-major data; the consuming neuron or device determines how it enters the simulation.

## Choose the input boundary

| API | Use |
|---|---|
| `braintools.input.Constant([(current, duration), ...])` | Use for baseline, pulse, and recovery sections; calling the protocol returns one physical current sample per step. |
| `braintools.input.Ramp(start, end, duration)` | Use for a continuous current ramp; preserve the same current unit at both endpoints. |
| `brainstate.transform.for_loop(step, times, current)` | Use for a native rollout; slice time and drive together and advance the neuron once per sample. |
| `sim.create(brainspike.step_current_generator, ...)` | Use the NEST device path for piecewise current injection; supply the documented change times and amplitudes rather than a native input array. |

## Run a generated protocol through a native neuron

This example calls a public BrainSpike neuron directly inside a native State-aware loop. Normalize generated current to `pA` before the transformed loop: this model stores buffered current in `pA`, and changing its unit metadata during a State-aware transform violates the fixed State tree. Its NEST-compatible `update(x=...)` buffers current for the next step; do not assume same-step injection when porting a timing-sensitive protocol.

```python
import brainspike
import brainstate
import braintools
import brainunit as u

with brainstate.environ.context(dt=0.1*u.ms):
    protocol = braintools.input.Constant([
        (0.0*u.nA, 2.0*u.ms),
        (0.8*u.nA, 4.0*u.ms),
        (0.0*u.nA, 2.0*u.ms),
    ])
    current = protocol().in_unit(u.pA)
    neuron = brainspike.iaf_psc_alpha(
        1, V_initializer=braintools.init.Constant(-65.0*u.mV),
    )
    brainstate.nn.init_all_states(neuron)
    times = u.math.arange(current.shape[0])*brainstate.environ.get_dt()

    def step(t, drive):
        with brainstate.environ.context(t=t):
            neuron.update(x=drive)
            return neuron.V.value

    voltage = brainstate.transform.for_loop(step, times, current)

assert voltage.shape == (80, 1)
```

For the custom `LIF` in `scripts/native_workflows.py`, pass each current sample to the neuron call; its update consumes that sample immediately. Keep this distinction in the scientific exchange contract.

For a NEST Simulator graph, open `references/nest-compatible/devices.md` for current-generator parameters and connection direction. `simulate()` has no native time-major-input argument; do not pass a generated array as its duration or replace the graph's loop with manual neuron stepping.

Open `references/perturbation-experiment-validity.md` when current delivery is an intervention whose actual spike dose must be calibrated. Generate noisy protocols outside the update, declare whether trials reuse or resample them, and regenerate cached inputs when timestep or stochastic realization changes.

## Source

[BrainTools input API](https://brainx.chaobrain.com/braintools/apis/input.html). Use BrainTools 0.4.0 in the declared BrainX release; do not infer newer live-documentation APIs.
