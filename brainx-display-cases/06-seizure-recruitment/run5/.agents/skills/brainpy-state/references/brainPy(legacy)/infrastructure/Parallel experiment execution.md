# Parallel experiment execution

Use this reference to map independent legacy BrainPy experiments across a device batch or CPU processes and to control monitor memory inside each long simulation. Keep model dynamics in ordinary `brainpy` objects; this reference owns experiment-level execution, not `brainpy.state` transformations.

## Selection map

| Need | Use | Key constraint |
|---|---|---|
| Vectorize independent array tasks on one CPU or GPU backend | `jax_vectorize_map(...)` | Every mapped argument leaf must have the same leading length. |
| Map independent tasks across multiple JAX devices | `jax_parallelize_map(...)` | The mapped batch must fit the available device count for each `pmap` call. |
| Run Python-level model tasks in separate native processes | `process_pool(...)` | Call it under an `if __name__ == '__main__':` guard. |
| Serialize a shared side effect across processes | `process_pool_lock(...)` | The worker must accept the supplied `lock` argument. |
| Run CPU tasks with a progress bar and preserve input order | `cpu_ordered_parallel(...)` | Ordered collection can wait for an earlier slow task. |
| Return CPU results as tasks finish | `cpu_unordered_parallel(...)` | Do not infer parameter order from the returned list. |
| Reduce monitor memory at a fixed sampling interval | Chunked `brainpy.math.for_loop(...)` | Reshape time and inputs so each outer item contains the steps between samples. |
| Monitor different variables at different intervals | A Python loop around `model.jit_step_run(...)` | Keep the compiled step but perform monitor selection explicitly. |

## Map experiments with JAX

Use the JAX mapping helpers when the experiment function is compatible with JAX array mapping and all tasks share one result structure.

| API | Description |
|---|---|
| `jax_vectorize_map(func, arguments, num_parallel, clear_buffer=False)` | Batch tasks through `jax.vmap`; use on CPU or GPU, especially for single-device GPU vectorization. It concatenates batch results and optionally clears cached buffer memory after each batch. |
| `jax_parallelize_map(func, arguments, num_parallel, clear_buffer=False)` | Batch tasks through `jax.pmap` across CPU or GPU devices. On a single CPU, configure the host device count with `brainpy.math.set_host_device_count(n)` before using this route. |

Pass `arguments` as a sequence of mapped positional-argument arrays or as a dictionary of mapped keyword-argument arrays. All leaves must have the same leading task count; otherwise the helpers raise an error rather than silently truncating the experiment set.

`num_parallel` is the batch size processed per map call. Set `clear_buffer=True` only when lower retained memory is worth recreating the mapped function and converting batch results between batches.

## Run experiments in CPU processes

Use process workers when each experiment constructs and runs a Python-level BrainPy model independently.

| API | Description |
|---|---|
| `process_pool(func, all_params, num_process)` | Run one task for each tuple/list of positional parameters or dictionary of keyword parameters and return a result list. |
| `process_pool_lock(func, all_params, num_process)` | Run the same task pattern while adding a shared process lock to each worker for synchronized file or resource access. |
| `cpu_ordered_parallel(func, arguments, num_process=None, num_task=None, **tqdm_kwargs)` | Apply `func` to iterable positional or keyword arguments with a progress bar and return results in input order. `num_process` may be a worker count or a fraction of available CPUs. |
| `cpu_unordered_parallel(func, arguments, num_process=None, num_task=None, **tqdm_kwargs)` | Apply the same CPU mapping interface but collect results in completion order. |

**Invariant:** place every multiprocessing call under the main-module guard. Without it, spawned workers can import the module and recursively launch more workers.

### Canonical CPU sweep

Construct the model inside the worker so each process owns independent model state:

```python
import brainpy as bp
import brainpy.math as bm
import numpy as np


def simulate(input_current):
    current = bm.as_jax(input_current)
    model = bp.dyn.HH(1)
    runner = bp.DSRunner(
        model,
        monitors=["spike"],
        progress_bar=False,
    )
    steps = int(100.0 / bm.get_dt())
    inputs = bm.ones(steps) * current
    runner.run(inputs=inputs)
    spike_count = runner.mon.spike.sum()
    bm.clear_buffer_memory()
    return spike_count


if __name__ == "__main__":
    currents = np.linspace(1.0, 10.0, 8)
    spike_counts = bp.running.cpu_ordered_parallel(
        simulate,
        [currents],
        num_process=4,
    )
    assert len(spike_counts) == len(currents)
```

Use the unordered variant only when completion-order results are acceptable or when each worker returns its parameter identity with the result.

## Reduce monitor memory within an experiment

Long simulations can exhaust memory when every variable is recorded at every integration step. Sample only at the temporal precision required by the analysis.

### Use one interval for all monitored values

Reshape the step indices and inputs to `[n_samples, steps_per_sample]`. The function passed to `brainpy.math.for_loop(...)` runs the inner steps and returns only the final value to record for each outer sample.

```python
import brainpy as bp
import brainpy.math as bm
import numpy as np


class Accumulator(bp.DynamicalSystem):
    def __init__(self):
        super().__init__()
        self.value = bm.Variable(bm.zeros(1))

    def update(self, input_value):
        self.value.value += input_value

    def run(self, ids, values):
        for step, input_value in zip(ids, values):
            bp.share.save(i=step, t=bm.get_dt() * step)
            self.update(input_value)
        return self.value.value


steps_per_sample = 10
indices = np.arange(10_000).reshape(-1, steps_per_sample)
inputs = np.full(indices.shape, 20.0)
model = Accumulator()

sampled_values = bm.for_loop(
    model.run,
    (indices, inputs),
    progress_bar=True,
)

assert sampled_values.shape == (indices.shape[0], 1)
```

The model's chunk method must still update shared time on every inner step, for example with `bp.share.save(i=step, t=bm.get_dt() * step)`, before calling its ordinary update method.

### Use different intervals for different values

Call `model.jit_step_run(step, input_value)` for each step, append fast monitors every step, and append slower monitors only when `step % interval == 0`. Convert the collected lists with `bm.as_numpy(...)` after the loop.

Choose this flexible Python monitor loop only when variables genuinely require different intervals. Use the chunked `for_loop` route when one interval is sufficient and whole-loop execution matters more than per-variable sampling flexibility.

## Sources

- https://brainpy.readthedocs.io/tutorial_simulation/monitor_per_multiple_steps.html
- https://brainpy.readthedocs.io/apis/running.html
