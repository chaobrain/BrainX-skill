# More about legacy BrainPy simulation

Use this reference after a legacy `brainpy.DynamicalSystem` is constructed and
needs `DSRunner` monitor/input configuration, repeated rollouts, or parallel
parameter exploration. Keep model construction in the legacy workflow and use
the APIs here only to execute and observe it.

## Run a `DynamicalSystem`

`DSRunner` lowers repeated model updates through a structural loop while owning
time, input application, monitoring, optional JIT compilation, and result
conversion.

| API | Description |
|---|---|
| `bp.DSRunner(target, inputs=(), monitors=None, numpy_mon_after_run=True, jit=True, dyn_vars=None, memory_efficient=False, dt=None, t0=0.0, progress_bar=True, data_first_axis=None, fun_inputs=None, fun_monitors=None)` | Use to configure execution around one legacy `DynamicalSystem`; it records monitor histories in `runner.mon`. |
| `runner.run(duration, inputs=None, reset_state=False, shared_args=None, progress_bar=None, eval_time=False)` | Run for a physical duration and optionally override run inputs, reset State, provide shared arguments, or evaluate runtime. |
| `runner.predict(duration=None, inputs=None, reset_state=False, eval_time=False, shared_args=None, inputs_are_batching=None)` | Use the prediction interface when inputs may be unbatched `(time, ...)` or batched `(sample, time, ...)`. |
| `runner.reset_state()` | Reset the target through the runner before an independent rollout. |

```python
import brainpy as bp

neuron = bp.neurons.LIF(100)
runner = bp.DSRunner(
    target=neuron,
    monitors=['spike', 'V'],
    inputs=[('input', 20.0)],
    dt=0.1,
    jit=True,
)

runner.run(100.0)

assert runner.mon.ts.shape[0] == runner.mon['V'].shape[0]
assert runner.mon['spike'].shape[1] == 100
```

The target must be a legacy `brainpy.DynamicalSystem`. Treat two calls as one
continuous rollout unless `reset_state=True` or `reset_state()` is used.

## Select monitor forms

Monitoring every value in a large network can dominate host memory. Record only
the paths, indices, or derived values needed for the scientific check.

| Monitor form | Use and result |
|---|---|
| `'E.spike'` | Resolve and record the named `bm.Variable`; retrieve it as `runner.mon['E.spike']`. |
| `('E.spike', [1, 2, 3])` | Record only the selected indices of a named variable. |
| `{'spike': net.E.spike}` | Record an explicitly supplied variable under a concise result key. |
| `{'spike': (net.E.spike, [1, 2, 3])}` | Record selected indices from an explicit target. |
| `{'E-I.spike': lambda: bm.concatenate((net.E.spike, net.I.spike))}` | Record a derived value returned by a callable. |

`runner.mon.ts` is the recorded time axis. With `numpy_mon_after_run=True`,
monitor arrays are converted after the run; keep them as device arrays when a
later compiled computation must consume them without a host transfer.

## Select input forms

An input specification identifies the target variable, value source, source
type, and update operation.

| Input form | Use and behavior |
|---|---|
| `('E.input', 20.0)` | Add one fixed scalar at every step; `'fix'` and `'+'` are the defaults. |
| `('E.input', current, 'iter')` | Consume axis 0 of `current` once per simulation step. |
| `('E.input', current, 'iter', '=')` | Assign rather than add each iterable sample. |
| `[spec1, spec2, ...]` | Apply several input operations to one or more target variables. |
| `inputs=input_function` | Execute a function that directly sets or updates target variables each step. |

```python
current, duration = bp.inputs.section_input(
    values=[0.0, 20.0, 0.0],
    durations=[100.0, 1000.0, 100.0],
    dt=0.1,
    return_length=True,
)

runner = bp.DSRunner(
    target=network,
    monitors=['E.spike'],
    inputs=[
        ('E.input', current, 'iter'),
        ('I.input', current, 'iter'),
    ],
    dt=0.1,
)
runner.run(duration)
```

For `predict(inputs_are_batching=False)`, each input leaf is time-major
`(num_time, ...)`. For `inputs_are_batching=True`, each leaf is
`(num_sample, num_time, ...)`. Do not infer batching from an ambiguous array;
set the flag when using the prediction interface.

## Choose a parameter-exploration executor

Each executor maps the same simulation function over parameter arrays, but its
parallelism and memory boundary differ.

| API | Use when | Key constraint |
|---|---|---|
| `bp.running.cpu_ordered_parallel(func, arguments, num_process=None, num_task=None, **tqdm_kwargs)` | Independent runs should use CPU worker processes and preserve input order. | Guard calls from Python files with `if __name__ == '__main__':`. |
| `bp.running.cpu_unordered_parallel(func, arguments, num_process=None, num_task=None, **tqdm_kwargs)` | Worker completion order is irrelevant and non-blocking collection is preferred. | Results do not preserve parameter order. |
| `bp.running.jax_vectorize_map(func, arguments, num_parallel, clear_buffer=False)` | Same-device vectorized execution is desired but the full parameter set is too large for one `jax.vmap`. | Tune `num_parallel` to device memory. |
| `bp.running.jax_parallelize_map(func, arguments, num_parallel, clear_buffer=False)` | Independent runs should be mapped over multiple JAX devices in bounded groups. | Device availability bounds meaningful parallelism. |
| `jax.vmap(func)(*arguments)` | The full parameter batch fits on one device and the function is vectorizable. | The whole mapped batch is compiled and resident together. |
| `jax.pmap(func)(*arguments)` | The leading mapped axis matches available devices. | Each mapped shard runs the same program on a separate device. |

```python
import brainpy as bp
import brainpy.math as bm


def spike_count(background_current):
    neuron = bp.neurons.HH(1)
    runner = bp.DSRunner(
        neuron,
        monitors=['spike'],
        inputs=[('input', background_current)],
        numpy_mon_after_run=False,
        progress_bar=False,
    )
    runner.run(1000.0)
    return runner.mon['spike'].sum()


currents = bm.linspace(1.0, 10.0, 20)
counts = bp.running.jax_vectorize_map(
    spike_count,
    [currents],
    num_parallel=4,
    clear_buffer=True,
)

assert counts.shape == currents.shape
```

Use `num_parallel` as the explicit memory-control knob. Start with a small value
and increase it only after measuring peak device memory and compilation cost.

## Manage process and device memory

- In a CPU worker function, call `bm.clear_buffer_memory()` only after converting
  required results to NumPy. The operation clears JAX device buffers, so prefer
  NumPy worker inputs and outputs when using it.
- Do not call `clear_buffer_memory()` inside a function mapped by
  `jax_vectorize_map`; use that wrapper's `clear_buffer=` argument.
- Set virtual CPU device count before importing BrainPy or otherwise
  initializing JAX. Changing it after backend initialization has no effect.
- Disable a runner progress bar inside `pmap`-style execution when its ordered
  callback cannot be carried by the mapped computation.
- Use array sharding APIs for distributing one array or model computation. Open
  `Multi-device array sharding.md` rather than treating an independent
  parameter sweep as array partitioning.

## Official sources

- `https://brainpy.readthedocs.io/tutorial_simulation/simulation_dsrunner.html`
- `https://brainpy.readthedocs.io/tutorial_simulation/parallel_for_parameter_exploration.html`
- Generated `DSRunner` and `brainpy.running` pages linked by the official
  documentation.
