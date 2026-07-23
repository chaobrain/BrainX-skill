# BrainState Extension Mechanisms

Use this reference when a BrainState task needs reusable class behavior, reusable constructor presets, runtime-mode objects, or callbacks around State operations. It deliberately does not restate ordinary State or Module construction and update patterns.

## Mixin System

Source: https://brainx.chaobrain.com/brainstate/how_to/custom_states_and_mixins.html

### Reuse focused behavior with `Mixin`

A mixin is a lightweight class that contributes methods or attributes without forcing a rigid inheritance hierarchy. In BrainState, every mixin inherits from `brainstate.mixin.Mixin`: it provides optional behavior, should not define its own `__init__`, and is usually paired with a core component such as `brainstate.nn.Module`.

```python
import datetime

import brainstate
from brainstate import mixin


class LoggingMixin(mixin.Mixin):
    """Attach timestamped logging without touching the host constructor."""

    def log(self, message: str) -> None:
        stamp = datetime.datetime.now().strftime('%H:%M:%S')
        print(f'[LOG {stamp}] {self.__class__.__name__}: {message}')


class Accumulator(brainstate.nn.Module, LoggingMixin):
    def __init__(self):
        super().__init__()
        self.total = 0.0

    def add(self, value):
        self.total += float(value)
        self.log(f'updated running total to {self.total:.2f}')
        return self.total
```

Keep mixins focused, avoid new required constructor arguments, and document any host attributes that a mixin reads or writes. Several small mixins compose better than one opinionated base class.

### Capture constructor presets with `ParamDesc`

`ParamDesc.desc(...)` stores constructor arguments in a `ParamDescriber`. Calling the descriptor creates a new object, and call-time arguments may override stored arguments. The descriptor is hashable; `descriptor.identifier` is safe to use as a dictionary key.

```python
from brainstate import mixin


class DenseBlock(mixin.ParamDesc):
    def __init__(self, in_features, out_features, *, activation='relu'):
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation


encoder_block = DenseBlock.desc(256, 128, activation='gelu')

gelu_block = encoder_block()
relu_block = encoder_block(activation='relu')  # override at call time
cache_key = encoder_block.identifier
```

For a class that does not inherit from `ParamDesc`, construct `ParamDescriber` directly:

```python
from dataclasses import dataclass

from brainstate import mixin


@dataclass
class OptimConfig:
    lr: float
    beta1: float = 0.9
    beta2: float = 0.999


adam_template = mixin.ParamDescriber(
    OptimConfig,
    lr=1e-3,
    beta1=0.95,
)
opt_a = adam_template()
opt_b = adam_template(lr=5e-4)
```

### Express runtime type expectations

`JointTypes[A, B, ...]` behaves like an intersection: an instance must satisfy all listed types. `OneOfTypes[A, B, ...]` behaves like a union: an instance may satisfy any listed type.

```python
from brainstate import mixin


class Persistable:
    pass


class Visualisable:
    pass


class Report(Persistable, Visualisable):
    pass


FullFeatureType = mixin.JointTypes[Persistable, Visualisable]
OptionalNumber = mixin.OneOfTypes[int, float, type(None)]

assert isinstance(Report(), FullFeatureType)
assert isinstance(3.14, OptionalNumber)
assert isinstance(None, OptionalNumber)
```

These helpers make interface expectations explicit; they are used with `isinstance` in the official example.

### Centralize runtime behavior with modes

Mode objects capture the context in which computation happens. The lightweight `Mode` base class and the built-ins `Training`, `Batching`, and `JointMode` cover common runtime switches. `mode.has(SomeModeType)` tests membership. A `JointMode` exposes member attributes, so `mode.batch_size` works when one member is `Batching(batch_size=...)`.

```python
import jax.numpy as jnp
from brainstate import mixin


class ToyPipeline:
    def __init__(self):
        self.mode: mixin.Mode = mixin.Mode()

    def set_mode(self, *modes: mixin.Mode):
        if not modes:
            self.mode = mixin.Mode()
        elif len(modes) == 1:
            self.mode = modes[0]
        else:
            self.mode = mixin.JointMode(*modes)

    def forward(self, values):
        x = jnp.asarray(values, dtype=jnp.float32)
        if self.mode.has(mixin.Training):
            x = x + 0.1
        if self.mode.has(mixin.Batching):
            x = x.reshape((self.mode.batch_size, -1)).mean(axis=1)
        return x


pipeline = ToyPipeline()
pipeline.set_mode(
    mixin.Training(),
    mixin.Batching(batch_size=2),
)
result = pipeline.forward(jnp.arange(4.0))  # [0.6, 2.6]
```

Use mode objects to centralize runtime semantics instead of scattering ad hoc boolean flags.

## Observe and Intercept State Access with Hooks

Source: https://brainx.chaobrain.com/brainstate/how_to/state_hooks.html

State hooks run a callback when a State is read, written, restored, or created without editing the model that owns it. They are intended for cross-cutting concerns such as logging changes, validating writes, enforcing invariants, and tracing access patterns while debugging.

| Operation | Fires when | Can modify or cancel? |
|---|---|---|
| `read` | `state.value` is read | No; inspect only |
| `write_before` | Just before `state.value = ...` | Yes; transform or cancel |
| `write_after` | Just after a write completes | No; inspect only |
| `restore` | `state.restore_value(...)` is called | No |
| `init` | A State is constructed | No |

`brainstate.register_state_hook(...)` registers a global hook that fires for every State. `state.register_hook(...)` scopes the hook to one instance.

### Observe completed writes

A `write_after` callback receives a context containing the State, its `old_value`, and the new `value`. The context also exposes `state_name`, `operation`, `timestamp`, and a `metadata` dictionary for passing information between hooks. `ctx.state` is a weak-reference target and returns `None` after the State has been garbage-collected, so long-lived callbacks must guard it.

```python
import jax.numpy as jnp
import brainstate

brainstate.clear_state_hooks()
history = []


def record(ctx):
    history.append(
        (ctx.state_name, float(ctx.old_value), float(ctx.value))
    )


handle = brainstate.register_state_hook('write_after', record)

weight = brainstate.State(jnp.array(1.0), name='weight')
weight.value = jnp.array(2.0)
weight.value = jnp.array(2.5)
# history == [('weight', 1.0, 2.0), ('weight', 2.0, 2.5)]
```

### Transform or reject a pending write

A `write_before` hook runs before storage and may set `ctx.transformed_value`. Multiple `write_before` hooks chain in priority order, and each hook sees the previous hook's output.

```python
brainstate.clear_state_hooks()


def clip_to_unit(ctx):
    current = (
        ctx.transformed_value
        if ctx.transformed_value is not None
        else ctx.value
    )
    ctx.transformed_value = jnp.clip(current, -1.0, 1.0)


clip_handle = brainstate.register_state_hook(
    'write_before',
    clip_to_unit,
)

gate = brainstate.State(jnp.array(0.0))
gate.value = jnp.array(5.0)  # stored as 1.0
```

To reject rather than transform, set `ctx.cancel = True`. Assignment then raises `brainstate.HookCancellationError` and preserves the previous value; `ctx.cancel_reason` supplies the reason.

```python
brainstate.clear_state_hooks()


def reject_negative(ctx):
    value = (
        ctx.transformed_value
        if ctx.transformed_value is not None
        else ctx.value
    )
    if jnp.any(value < 0):
        ctx.cancel = True
        ctx.cancel_reason = 'value must be non-negative'


reject_handle = brainstate.register_state_hook(
    'write_before',
    reject_negative,
)
```

### Scope and manage hook lifetime

Use `state.register_hook(...)` when only one State needs observation or interception; other States are unaffected.

```python
brainstate.clear_state_hooks()

watched = brainstate.State(jnp.array(0.0), name='watched')
other = brainstate.State(jnp.array(0.0), name='other')
seen = []

handle = watched.register_hook(
    'write_after',
    lambda ctx: seen.append(float(ctx.value)),
)
watched.value = jnp.array(1.0)
other.value = jnp.array(99.0)  # not observed
```

Every registration returns a `HookHandle`. Use `disable()` to silence it temporarily, `enable()` to reactivate it, `remove()` to remove it permanently, and `is_removed()` to inspect removal status. Use `brainstate.list_state_hooks()` to list registered hooks, optionally filtered by type; `brainstate.has_state_hooks()` to report whether any are active; and `brainstate.clear_state_hooks()` to remove all global hooks.

Bound temporary hook lifetimes explicitly:

```python
handle.disable()
handle.enable()
handle.remove()
assert handle.is_removed()
```

### Keep hooks safe under compiled execution

Hooks are ordinary Python callbacks. They fire on every concrete `.value` access; inside a `brainstate.transform.jit` step, they fire once per call at runtime and once during the initial trace, when `ctx.value` is an abstract tracer rather than a concrete array.

Do not branch in Python on a hooked value's contents, such as `if float(...)`. For NaN guards or bounds assertions that must execute inside compiled code, use the transformation-aware error tools `brainstate.transform.checkify`, `brainstate.transform.check`, or `brainstate.transform.debug_nan`.
