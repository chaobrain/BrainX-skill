# Preserve signal history with delay buffers

Use `brainstate.nn.Delay` when a native Module needs explicit signal history. The buffer owns both stored samples and mutable bookkeeping; initialize and advance them as one registered Module.

## Write before retrieving

| API | Use |
|---|---|
| `brainstate.nn.Delay(target_info, max_time)` | Allocate history from a shape/dtype specification and maximum physical delay under the initialization timestep. |
| `delay.update(value)` | Write one current sample per step before retrieval in this convention. |
| `delay.retrieve_at_step(step)` | Read an integer offset; step zero is the just-written sample and step `d` is `d` updates earlier. |
| `delay.register_entry(name, delays, indices)` / `delay.at(name)` | Register a named heterogeneous tap before initialization, then retrieve the selected per-source histories. |
| `brainstate.nn.init_all_states(model)` | Initialize the whole registered graph, including the buffer and write position, for an independent rollout. |

```python
import brainstate
import brainunit as u
import jax
import jax.numpy as jnp

with brainstate.environ.context(dt=0.1*u.ms):
    history = brainstate.nn.Delay(jax.ShapeDtypeStruct((3,),jnp.bool_),0.3*u.ms)
    history.register_entry("taps",jnp.asarray([0,2,3])*0.1*u.ms,jnp.arange(3))
    brainstate.nn.init_all_states(history)
    impulses = jnp.zeros((6,3),dtype=jnp.bool_).at[0].set(True)
    def step(events):
        history.update(events)
        return history.at("taps")
    delayed = brainstate.transform.for_loop(step,impulses)
assert jnp.array_equal(jnp.argmax(delayed,axis=0),jnp.asarray([0,2,3]))
```

Initialize under the intended `dt` and keep it fixed for the rollout. Reinitialize all history and bookkeeping at independent boundaries; clearing a neural voltage or sending zeros alone does not reset a delay. Keep signal shape, dtype, and physical unit metadata stable.

Reading before `update()` changes the indexing convention by one step. Choose one order and prove it with an impulse; do not insert an unexplained `d-1` correction. A delay measured from an already completed source event also includes that event's availability convention.

Use grid-aligned integer taps for binary events. Off-grid interpolation can produce fractional values that are not binary spikes; choose a documented continuous-signal interpolation rule only when the model requires it. Quantize physical event times before the rollout and state the rounding policy.

For mapped conditions, map the history and mutable write position compatibly. Do not vmap only the stored samples while sharing a writable scalar pointer. The canonical example verifies unbatched named taps; it does not establish every mapped or interpolated configuration.

Open `brainspike` → `references/native-delays.md` for a complete delayed conductance-to-neuron path and its source, conductance-onset, voltage-onset, and reset checks.

## Source

[BrainState delay protocol](https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/02_synaptic_delays.html). The example is executed on BrainState 0.6.0 in the declared BrainX release.
