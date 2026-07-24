# Brainstate randomness and reproducibility

Open this reference after `skills/brainstate/SKILL.md` when a task needs independent streams, repeated stochastic trials, random calls inside a transformed operation, dropout or noise, direct key control, parallel key preparation, exact replay, or RNG state stored with a checkpoint.

The skill owns the global `DEFAULT` generator and ordinary seeding path; do not repeat those basics here. The supplemental rule is that `brainstate.random` wraps JAX random generation in a stateful interface with automatic key splitting, while its random functions remain JIT-compatible.

Source: https://brainx.chaobrain.com/brainstate/apis/random.html

## Independent streams and repeated trials

"For advanced use cases, you can create custom `RandomState` instances with independent random streams." Use one `RandomState` per concern whose sequence must advance independently, such as data augmentation and model initialization. Consecutive calls on one instance are successive stochastic trials from that stream.

```python
import brainstate
import jax.numpy as jnp

rng1 = brainstate.random.RandomState(42)
rng2 = brainstate.random.RandomState(123)

samples1 = rng1.randn(5)
samples2 = rng2.randn(5)

assert not jnp.allclose(samples1, samples2)
```

The API also supports the explicit keyword form `brainstate.random.RandomState(seed=123)` and instance methods such as `rng.normal(0, 1, size=(10, 10))`. Use the advanced section below when the task requires direct key manipulation.

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html

## Stochastic calls, transforms, dropout, and noise

BrainState random functions are JIT-compatible. Apply the whole-operation `brainstate.transform` rule from `skills/brainstate/SKILL.md` to stochastic code just as you do to other stateful code. The advanced section below covers mapped randomness under `vmap`, parallel key assignment, and explicit key control.

The tutorial's dropout pattern bundles fit-mode gating, mask generation, and inverse-keep scaling in the stochastic call:

```python
class Dropout(brainstate.nn.Module):
    def __init__(self, drop_rate=0.5):
        super().__init__()
        self.drop_rate = drop_rate

    def __call__(self, x):
        fit = brainstate.environ.get('fit', False)
        if not fit:
            return x

        keep_prob = 1.0 - self.drop_rate
        mask = brainstate.random.bernoulli(keep_prob, x.shape)
        return x * mask / keep_prob
```

When `fit` is false, the module returns `x` unchanged. When fitting, `bernoulli` draws the mask with mean `keep_prob`, and division by `keep_prob` applies the tutorial's scaling.

The tutorial's noisy-layer pattern keeps trainable values in `ParamState` and draws fresh Gaussian weight noise inside every call. Two consecutive calls are two stochastic trials because automatic key management advances the random state.

```python
class NoisyLayer(brainstate.nn.Module):
    def __init__(self, d_in, d_out, noise_std=0.1):
        super().__init__()
        self.noise_std = noise_std
        self.w = brainstate.ParamState(
            brainstate.random.randn(d_in, d_out) * 0.1
        )
        self.b = brainstate.ParamState(jnp.zeros(d_out))

    def __call__(self, x):
        w_noisy = self.w.value + brainstate.random.normal(
            0,
            self.noise_std,
            self.w.value.shape,
        )
        return x @ w_noisy + self.b.value


layer = NoisyLayer(5, 3, noise_std=0.01)
x = jnp.ones(5)
y1 = layer(x)
y2 = layer(x)
```

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html

## Checkpoint the rng with the model

When exact stochastic continuation matters, the tutorial saves the model and current RNG key in the same checkpoint, then restores the key before continuing:

```python
checkpoint = {
    'model': model.state_dict(),
    'rng_key': brainstate.random.get_key()
}

brainstate.random.set_key(checkpoint['rng_key'])
```

Keep this compact checkpoint requirement here because it determines what must be saved. Read the advanced section below before adapting key restoration, restoring multiple streams, or defining more elaborate checkpoint behavior.

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html

## Advanced brainstate randomness

Use this section when a task needs direct key control, parallel key preparation, exact key replay, custom-generator state inspection, or checkpoint behavior beyond the ordinary stochastic path above.

### Snapshot a global key and replay from it

The tutorial's advanced boundary is: "For advanced use cases, you can directly access and manipulate keys." Bundle inspection and restoration with the operation whose randomness must be replayed:

```python
import brainstate
import jax.numpy as jnp

saved_key = brainstate.random.get_key()
v1 = brainstate.random.randn(3)
_ = brainstate.random.randn(3)

brainstate.random.set_key(saved_key)
v1_replayed = brainstate.random.randn(3)

assert jnp.allclose(v1, v1_replayed)
```

The snapshot is the sequence position immediately before `v1`; restoring it makes the next draw reproduce `v1`. Use `get_key()` / `set_key(...)` for this explicit snapshot-and-replay contract. Keep `restore_key()` distinct: the API describes it only as restoring "the default random key to its previous state," so do not silently substitute it for a named checkpoint snapshot.

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html

API definitions: https://brainx.chaobrain.com/brainstate/apis/random.html

### Prepare keys for parallel or mapped work

The API groups these as "Functions for creating independent random keys for parallel computation." The tutorial's documented parallel preparation is:

```python
keys = brainstate.random.split_key(n=4)

for i, key in enumerate(keys):
    print(i, key)
```

Select the narrowest documented operation:

| API | Documented role |
|---|---|
| `split_key` | Create new random key(s) from the current seed. The tutorial uses `split_key(n=4)` for parallel operations. |
| `split_keys` | Create multiple independent random keys from the current seed. |
| `self_assign_multi_keys` | Assign multiple keys to the global random state for parallel access. |

The routed pages do not provide a `vmap` call, mapped key-axis configuration, or per-example key-consumption rule. They only identify `vmap` as a next step and document the parallel key utilities above. Do not invent mapped semantics from these pages; combine these RNG operations with a separately documented BrainState mapping workflow when concrete `vmap` mechanics are required.

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html

API definitions: https://brainx.chaobrain.com/brainstate/apis/random.html

### Inspect custom-stream state and choose advanced generators

After selecting a custom `RandomState`, the API documents its current key through `.value`:

```python
rng = brainstate.random.RandomState(seed=123)
_ = rng.normal(0, 1, size=(10, 10))
current_stream_key = rng.value
```

This is the documented custom-stream state inspection surface. The routed pages demonstrate reseeding a custom generator with `rng.seed(999)`, but they do not demonstrate assigning a saved value back to a custom generator or restoring several custom streams. Do not apply the global `set_key(...)` API to `rng`: its documented target is the global random state.

For generator selection without restating the independent-stream example above:

| API | Documented role |
|---|---|
| `default_rng` | Get the default random state or create a new one with a specified seed. |
| `clone_rng` | Create a clone of the random state or a new random state. |
| `seed_context` | Temporarily change the random seed with automatic restoration. |

The same API page's example spells the temporary context as `local_seed(123)`, while its API table lists `seed_context`. Because the routed source is internally inconsistent, verify the installed BrainState version before choosing the callable name.

Source: https://brainx.chaobrain.com/brainstate/apis/random.html

Custom reseeding example: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html

### Checkpoint at the intended sequence boundary

The tutorial's checkpoint pattern stores the current global key beside the model and restores that exact key before stochastic work resumes:

```python
checkpoint = {
    'model': model.state_dict(),
    'rng_key': brainstate.random.get_key(),
}

# Restore model state through the model's checkpoint path first.
brainstate.random.set_key(checkpoint['rng_key'])
```

Capture `rng_key` after every random operation that belongs to the completed portion of the run. Restoring an earlier snapshot intentionally replays draws after that snapshot, as the replay script above demonstrates. A checkpoint with multiple custom streams additionally needs each stream's current key, but these routed pages document only reading a custom key through `rng.value`; they do not supply a multi-stream restoration script. Keep such restoration out of generated code until a source for that specific mechanism is selected.

Source: https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html

### Key api distinctions

- `get_key()` returns the current global random key.
- `get_key_data()` returns the current global random key as raw `uint32[2]` data; use it only when raw key data is the required interchange form.
- `set_key(...)` sets a new key on the global random state.
- `restore_key()` restores the default key to its previous state.
- `RandomState.value` is the API page's documented way to inspect the current key of a custom generator.

Source: https://brainx.chaobrain.com/brainstate/apis/random.html

## Official sources

- https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html
- https://brainx.chaobrain.com/brainstate/apis/random.html
