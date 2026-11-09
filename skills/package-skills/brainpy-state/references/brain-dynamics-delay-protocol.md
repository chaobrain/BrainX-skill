# BrainPy projection delay protocol

Use this reference after choosing a native `brainpy.state` projection when presynaptic spikes or continuous values must reach the target after an axonal or synaptic transmission delay. This reference owns BrainPy projection integration; use the general BrainState delay protocol for arbitrary history buffers, named delay taps, delayed feedback State, or manual retrieval.

## Choose the delay integration point

Delay the signal before communication. Choose the API from how the projection receives that signal:

| Signal path | Use when | Call contract |
|---|---|---|
| `AlignPostProj(..., delay=delay)` | The step pushes a spike/event array directly into an AlignPost projection. | Call `proj(signal)`; the projection delays the supplied presynaptic input before `comm`. |
| `CurrentProj(...)` | The step pushes a continuous presynaptic value through communication and output conversion without synaptic dynamics. | Verify delay support and input shape in the target release's `CurrentProj` contract before using it for delayed input. |
| `pre.prefetch("V").delay.at(delay)` | The projection pulls from a built-in neuron, which stores membrane potential rather than a separate historical spike State. | Build the projection once with the delayed prefetch and a spike decoder, then call `proj()` with no explicit signal. |
| General `brainstate.nn.Delay` APIs | The task needs reusable buffers, multiple named taps, manual updates, step-based retrieval, round interpolation, or custom delayed State. | Leave the BrainPy projection path and use the general BrainState protocol. |

Do not combine a delayed prefetch with an explicit signal at update time. The projection either pulls its configured delayed source or delays the value passed to `proj(signal)`.

## Push a delayed signal

Use `delay=` for a direct-input `AlignPostProj`. The [official `delay` parameter reference](https://brainpy.readthedocs.io/apis/generated/brainpy.state.AlignPostProj.html) documents scalar physical time and per-source delays. Use the BrainPy 2.9 documentation for the parameter contract and the release-scoped checks below for measured arrival timing.

| API | Behavior |
|---|---|
| `AlignPostProj(..., delay=None)` | Keep the zero-overhead undelayed path. |
| `AlignPostProj(..., delay=scalar_time)` | Delay every presynaptic element by the same duration. |
| `AlignPostProj(..., delay=per_pre_time)` | Apply heterogeneous axonal delays; the delay array must have shape `(N_pre,)`. |

For continuous input, the [BrainPy 2.9 `CurrentProj` API](https://brainpy.readthedocs.io/apis/generated/brainpy.state.CurrentProj.html) documents `delay=None`, a scalar physical time, or a `(N_pre,)` delay array. The BrainPy 2.9.0 delay suite covers the homogeneous delayed `CurrentProj` path; verify the input shape against that contract before use.

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

The same official `AlignPostProj` reference documents linear interpolation below `dt` and buffer sizing from `ceil(max(delay) / dt)` during initialization. Establish `dt` before initializing State and keep it fixed for that rollout. Verify off-grid behavior separately before sending interpolated values to binary-event communication.

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
6. Inject one event through the production delay path and compare it with an undelayed control. Record the input step, first conductance/current change, and first target-voltage response. For voltage timing, keep the target out of its refractory interval and compare against the quiet baseline with a tolerance. Test off-grid delays separately when the model uses them.

**Release-scoped checks:** In BrainX `2026.8.18` / BrainPy `2.8.2` (`brainx-eval:2026-11-02-login-fix`), a one-edge `AlignPostProj` with instantiated `Expon` and `COBA` received a boolean impulse at step 1 with `dt=0.1 ms`. Conductance and target voltage first changed at step 1 for `delay=None`, and at step 3 for `delay=0.2 ms`.

In BrainX `2026.10.25` / BrainPy `2.9.0` (`brainx-eval:2026.10.25-py312`), the same probe reproduced the conductance onsets at steps 1 and 3, and the installed `brainpy/state/delay_test.py` suite passed all 11 cases. Those cases cover `AlignPostProj` with `delay=None`, a scalar delay, and per-element delays, plus homogeneous delayed `CurrentProj`.

Both checks cover the scalar grid-aligned native path and projection-before-target order. Verify off-grid delays and other projection forms separately; a manual BrainState buffer test does not validate the native projection path.

Do not validate a delay only from postsynaptic spike time. Membrane integration, thresholding, refractory State, and other inputs can shift or suppress the postsynaptic spike even when the transmission delay is correct.

## Use a manual BrainState delay

Insert the current sample before retrieval so step 0 always means current and step `d` means exactly `d` completed updates earlier.

| API | Description |
|---|---|
| `brainstate.nn.Delay(target_info, max_time)` | Use when model logic needs an explicit reusable history buffer rather than a projection-owned delay; it allocates enough history for `max_time` under the initialization `dt`. |
| `delay.update(value)` | Insert the current sample once per step before any retrieval that should include the updated history. |
| `delay.retrieve_at_step(step)` | Retrieve an integer-offset sample; after the current update, step 0 is `value` and step `d` is the value from `d` updates earlier. |
| `delay.retrieve_at_time(time)` | Retrieve by a unit-bearing time when the model is defined in physical time; select the documented interpolation behavior rather than manually rounding an off-grid delay. |
| `delay.register_entry(name, vector_delays, indices)` and `delay.at(name)` | Use for one named heterogeneous tap per selected source element; register before State initialization, then retrieve the selected delayed values after each update. |

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

For grid-aligned binary spikes, retrieve exact integer steps or use named entries whose delays are integer multiples of `dt`. Linear time interpolation can produce fractional numeric values, which no longer satisfy binary-event semantics. Use continuous interpolation only when the modeled signal and off-grid timing require it.

When a unit-bearing event time feeds an integer delay bank, convert `time / dt` to one documented integer step before the rollout and generate the event on that grid. Do not rediscover its step inside the time loop with a half-open `abs(t - arrival) < dt / 2` window; exact half-step results can depend on floating representation.

## Map delay State across independent conditions

Keep every writable part of a delay on one compatible mapped axis. A ring buffer whose history is per lane but whose mutable write pointer remains scalar cannot be shared across a mapped step, even when every lane advances in lockstep.

Prefer `brainstate.nn.Delay` when its complete mutable State can be initialized and mapped per condition. If its bookkeeping cannot satisfy the State-axis policy, use a small pointer-free `HiddenState` shift register only for a fixed integer delay bank; keep step 0 as the current event and prove at least one nonzero tap with an impulse assertion. Do not generalize this fallback to long histories because shifting the full buffer costs work proportional to the maximum delay.

Open `references/scripts/sound_localization.py` when building a mapped multi-tap coincidence detector, for the verified pointer-free fallback, semantic State filters, integer event taps, and impulse check.

## General BrainState boundary

Open the official BrainState [Delay Protocol](https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/02_synaptic_delays.html) when implementing `brainstate.nn.DelayAccess`, `brainstate.nn.StateWithDelay`, rotation versus concatenation buffers, named entries, multiple taps, or linear-versus-round interpolation. Those mechanisms are more general than BrainPy projection delays and should not be reconstructed inside a projection unless the projection APIs cannot express the required history.

## Official sources

- [BrainPy synaptic delays guide](https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/sim-delays.html)
- [BrainPy AlignPostProj API](https://brainpy.readthedocs.io/apis/generated/brainpy.state.AlignPostProj.html)
- [BrainPy CurrentProj API](https://brainpy.readthedocs.io/apis/generated/brainpy.state.CurrentProj.html)
- [BrainState synaptic delays tutorial](https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/02_synaptic_delays.html)
