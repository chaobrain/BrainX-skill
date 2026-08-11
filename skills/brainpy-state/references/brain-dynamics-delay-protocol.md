# BrainPy projection delay protocol

Use this reference after choosing a native `brainpy.state` projection when presynaptic spikes or continuous values must reach the target after an axonal or synaptic transmission delay. This reference owns BrainPy projection integration; use the general BrainState delay protocol for arbitrary history buffers, named delay taps, delayed feedback State, or manual retrieval.

## Choose the delay integration point

Delay the signal before communication. Choose the API from how the projection receives that signal:

| Signal path | Use when | Call contract |
|---|---|---|
| `AlignPostProj(..., delay=delay)` | The step pushes a spike/event array directly into an AlignPost projection. | Call `proj(signal)`; the projection delays the supplied presynaptic input before `comm`. |
| `CurrentProj(..., delay=delay)` | The step pushes a continuous presynaptic value through communication and output conversion without synaptic dynamics. | Call `proj(signal)`; the projection delays the supplied value before `comm`. |
| `pre.prefetch("V").delay.at(delay)` | The projection pulls from a built-in neuron, which stores membrane potential rather than a separate historical spike State. | Build the projection once with the delayed prefetch and a spike decoder, then call `proj()` with no explicit signal. |
| General `brainstate.nn.Delay` APIs | The task needs reusable buffers, multiple named taps, manual updates, step-based retrieval, round interpolation, or custom delayed State. | Leave the BrainPy projection path and use the general BrainState protocol. |

Do not combine a delayed prefetch with an explicit signal at update time. The projection either pulls its configured delayed source or delays the value passed to `proj(signal)`.

## Push a delayed signal

Use `delay=` for a direct-input `AlignPostProj` or `CurrentProj`. A scalar time applies one homogeneous delay; an `(N_pre,)` quantity applies one delay per presynaptic neuron.

| API | Behavior |
|---|---|
| `AlignPostProj(..., delay=None)` | Keep the zero-overhead undelayed path. |
| `AlignPostProj(..., delay=scalar_time)` | Delay every presynaptic element by the same duration. |
| `AlignPostProj(..., delay=per_pre_time)` | Apply heterogeneous axonal delays; the delay array must have shape `(N_pre,)`. |
| `CurrentProj(..., delay=...)` | Apply the same scalar or per-presynaptic delay contract to continuous direct input. |

```python
proj = brainpy.state.AlignPostProj(
    comm=brainstate.nn.EventFixedProb(
        n_pre,
        n_post,
        conn_num=0.1,
        conn_weight=0.5 * u.mS,
    ),
    syn=brainpy.state.Expon.desc(n_post, tau=5.0 * u.ms),
    out=brainpy.state.COBA.desc(E=0.0 * u.mV),
    post=post,
    delay=5.0 * u.ms,
)

# Inside the step, before post(...) integrates:
proj(pre.get_spike() != 0.0)
```

Sub-`dt` delays are linearly interpolated. `init_state()` sizes the buffer from `ceil(max(delay) / dt)`, so establish `dt` before State initialization and do not change it during that initialized run.

## Pull a delayed neuron signal

Use delayed prefetch when the projection owns presynaptic extraction. Built-in neurons expose historical `V`, not a separate stored spike history, so delay `V` and re-derive the spike from the delayed voltage.

```python
proj = brainpy.state.AlignPostProj(
    pre.prefetch("V").delay.at(5.0 * u.ms),
    lambda voltage: pre.get_spike(voltage) != 0.0,
    comm=brainstate.nn.EventFixedProb(
        n_pre,
        n_post,
        conn_num=0.1,
        conn_weight=0.5 * u.mS,
    ),
    syn=brainpy.state.Expon.desc(n_post, tau=5.0 * u.ms),
    out=brainpy.state.COBA.desc(E=0.0 * u.mV),
    post=post,
)

# The projection pulls the delayed voltage and decodes its spike.
proj()
pre(drive)
post(0.0 * u.mA)
```

Keep the projection-before-post order. The delayed signal must be communicated before `post(...)` integrates the current step.

## Initialize, reset, and verify

1. Enter `brainstate.environ.context(dt=...)`.
2. Construct the network and delay-bearing projection.
3. Call `brainstate.nn.init_all_states(net)` so neural State and delay buffers are initialized together.
4. Run time through `brainstate.transform.for_loop`.
5. For every independent rollout, call `init_all_states()` again.
6. Compare otherwise identical undelayed, short-delay, and long-delay runs; monitor the presynaptic spike and projection conductance/current to verify the expected arrival-time shift before interpreting postsynaptic spikes.

Do not validate a delay only from postsynaptic spike time. Membrane integration, thresholding, refractory State, and other inputs can shift or suppress the postsynaptic spike even when the transmission delay is correct.

## Use a manual BrainState delay

Insert the current sample before retrieval so step 0 always means current and step `d` means exactly `d` completed updates earlier.

| API | Description |
|---|---|
| `brainstate.nn.Delay(target_info, max_time)` | Use when model logic needs an explicit reusable history buffer rather than a projection-owned delay; it allocates enough history for `max_time` under the initialization `dt`. |
| `delay.update(value)` | Insert the current sample once per step before any retrieval that should include the updated history. |
| `delay.retrieve_at_step(step)` | Retrieve an integer-offset sample; after the current update, step 0 is `value` and step `d` is the value from `d` updates earlier. |
| `delay.retrieve_at_time(time)` | Retrieve by a unit-bearing time when the model is defined in physical time; select the documented interpolation behavior rather than manually rounding an off-grid delay. |

```python
import brainstate
import brainunit as u
import jax
import jax.numpy as jnp

with brainstate.environ.context(dt=1.0 * u.ms):
    delay = brainstate.nn.Delay(
        jax.ShapeDtypeStruct((), jnp.float32),
        3.0 * u.ms,
    )
    brainstate.nn.init_all_states(delay)
    impulse = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0])

    def step(value):
        delay.update(value)
        return delay.retrieve_at_step(jnp.asarray(3, dtype=jnp.int32))

    delayed = brainstate.transform.for_loop(step, impulse)

assert jnp.array_equal(delayed, jnp.array([0.0, 0.0, 0.0, 1.0, 0.0]))
```

If retrieval occurs before `update(value)`, the buffer still represents the previous completed step and the index for the same observed latency changes by one. Do not compensate with an unexplained `d - 1`; choose one call order, document it beside the step, and lock the convention with an impulse assertion before interpreting neural output.

Reinitialize the delay for every independent rollout. A silent input interval advances the buffer but does not establish an independent State lifecycle.

## General BrainState boundary

Open the official BrainState [Delay Protocol](https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/02_synaptic_delays.html) when implementing `brainstate.nn.DelayAccess`, `brainstate.nn.StateWithDelay`, rotation versus concatenation buffers, named entries, multiple taps, or linear-versus-round interpolation. Those mechanisms are more general than BrainPy projection delays and should not be reconstructed inside a projection unless the projection APIs cannot express the required history.

## Official sources

- `https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/sim-delays.html`
- `https://brainx.chaobrain.com/brainpy-state/apis/generated/brainpy.state.AlignPostProj.html`
- `https://brainx.chaobrain.com/brainpy-state/apis/generated/brainpy.state.CurrentProj.html`
- `https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/02_synaptic_delays.html`
