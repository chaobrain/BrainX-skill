# Legacy BrainPy multi-device array sharding

Use this reference when a legacy `brainpy.math` array or PyTree must be placed
across a named JAX device mesh and the computation must preserve that placement.
This is array sharding, not independent parameter-sweep execution and not a
`brainpy.state` workflow.

## Build a named device mesh

`device_mesh` installs a scoped default `jax.sharding.Mesh`; array axis names
are resolved against that mesh to build a `NamedSharding` and `PartitionSpec`.

| API | Description |
|---|---|
| `bm.sharding.device_mesh(devices, axis_names)` | Use as a context manager to install a default mesh and restore the previous mesh on exit; the device-array rank must equal the number of mesh-axis names. |
| `bm.sharding.get_sharding(axis_names=None, mesh=None)` | Convert array axis names into `NamedSharding`; unresolved names become replicated `None` entries. |
| `bm.sharding.partition_by_axname(x, axis_names=None, mesh=None)` | Partition every supported array leaf by named array axes; the number of names must equal each leaf's rank. |

```python
import jax
import numpy as np

import brainpy.math as bm
from brainpy.math import sharding

devices = np.asarray(jax.devices())

with sharding.device_mesh(
    devices,
    axis_names=(sharding.BATCH_AXIS,),
) as mesh:
    values = bm.arange(devices.size * 8).reshape((devices.size, 8))
    values = sharding.partition_by_axname(
        values,
        axis_names=(sharding.BATCH_AXIS, None),
        mesh=mesh,
    )

    assert values.shape == (devices.size, 8)
```

`None` leaves an array axis replicated. If no mesh is supplied and no default
mesh is active, `partition_by_axname()` returns the input unchanged. If every
requested string is absent from the mesh, `get_sharding()` warns and produces a
fully replicated specification; treat that warning as a misspelled sharding
plan, not as successful partitioning.

## Choose a placement interface

Use named axes for maintainable model code, an explicit `Sharding` when another
JAX component already owns the placement, and `partition()` only at generic
boundaries accepting several placement forms.

| API | Description |
|---|---|
| `bm.sharding.partition_by_axname(x, axis_names=None, mesh=None)` | Use for a BrainPy-defined named-axis plan and validate one axis name per array dimension. |
| `bm.sharding.partition_by_sharding(x, sharding=None)` | Use with an existing `jax.sharding.Sharding`; it raises `TypeError` for another strategy type. |
| `bm.sharding.partition(x, sharding=None)` | Use at a generic placement boundary; accept a JAX device, a `Sharding`, or a sequence containing named axes. |
| `bm.sharding.keep_constraint(x)` | Use inside computation to reapply each JAX array leaf's current sharding with `jax.lax.with_sharding_constraint`; it returns a matching PyTree. |

These placement functions traverse PyTrees. BrainPy arrays keep their wrapper
while their underlying value is device-placed; JAX or NumPy arrays are returned
through BrainPy's sharded-array wrapper.

## Use canonical axis names

The constants standardize intent across legacy BrainPy models; only names also
present in the active mesh cause partitioning.

| Constant | Value and use |
|---|---|
| `bm.sharding.NEU_AXIS` | `'neuron'`; shard a neuron axis. |
| `bm.sharding.PRE_AXIS` | `'pre'`; shard a presynaptic axis. |
| `bm.sharding.POST_AXIS` | `'post'`; shard a postsynaptic axis. |
| `bm.sharding.SYN_AXIS` | `'synapse'`; shard a synapse axis. |
| `bm.sharding.TIME_AXIS` | `'time'`; shard a time axis only when the algorithm supports distributed temporal data. |
| `bm.sharding.BATCH_AXIS` | `'batch'`; shard independent batch elements. |

The constant names do not create mesh axes. Include the selected value in
`device_mesh(..., axis_names=...)`, then use the same value in the array's axis
specification.

## Distinguish sharding from mapped simulations

| Need | Use |
|---|---|
| Partition one array or PyTree across a mesh | `bm.sharding.partition_by_axname()` or explicit `Sharding` placement. |
| Preserve placement through a compiled operation | `bm.sharding.keep_constraint()`. |
| Run one independent model per device | `jax.pmap()` or `bp.running.jax_parallelize_map()`. |
| Bound same-device vectorized sweep memory | `bp.running.jax_vectorize_map(..., num_parallel=...)`. |

Do not add a leading parameter axis and call it model sharding unless each
device is intentionally running an independent replica. Open `More about
simulation.md` for mapped parameter sweeps and their memory-management rules.

## Source-backed failures

- Make `np.asarray(devices).ndim == len(mesh_axis_names)` before entering
  `device_mesh()`.
- Give `partition_by_axname()` one array-axis entry per leaf dimension; a
  mismatch raises `ValueError`.
- Do not mix PyTree leaves of different ranks under one shared axis-name tuple;
  partition those leaves with distinct specifications.
- Do not assume placement alone parallelizes an arbitrary legacy model. Verify
  the compiled operation's input and output shardings on the actual hardware.
- Configure virtual CPU devices before BrainPy or JAX initializes its backend.

## Official sources

- `https://brainpy.readthedocs.io/apis/brainpy.math.sharding.html`
- `https://brainpy.readthedocs.io/tutorial_simulation/parallel_for_parameter_exploration.html`
- Generated sharding-function pages and the mirrored official module source
  linked from the API index.
