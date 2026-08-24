# Brainstate randomness and reproducibility

Use this reference after `skills/package-skills/brainstate/SKILL.md` for advanced randomness. It covers independent streams, stochastic transforms, key control, parallel keys, replay, and checkpointing.

The root skill owns `brainstate.random.DEFAULT` and ordinary seeding. This reference owns advanced stream and key control.

## Independent streams

Use one `RandomState` per independent concern. Calls consume successive keys from that stream.

| API | Description |
|---|---|
| `RandomState(seed=...)` | Create an independent stateful random stream. |
| `rng.seed(seed)` | Restart a custom stream from a seed. |
| `rng.value` | Inspect the custom stream's current key. |

```python
import brainstate
import jax.numpy as jnp

init_rng = brainstate.random.RandomState(seed=42)
augment_rng = brainstate.random.RandomState(seed=123)

initial_weights = init_rng.randn(5)
augmented_values = augment_rng.randn(5)

assert not jnp.allclose(initial_weights, augmented_values)
```

Do not pass a custom stream to the global `set_key(...)`. The sources document `rng.value` inspection and `rng.seed(...)` reseeding. They do not document restoring a saved custom-stream key.

## Stochastic calls under transforms

BrainState random calls advance state and work under JIT. Transform the complete stateful operation with `brainstate.transform`.

| API | Description |
|---|---|
| `brainstate.random.bernoulli(p, size)` | Draw binary events or masks with probability `p`. |
| `brainstate.random.normal(loc, scale, size)` | Draw Gaussian initialization or noise. |
| `brainstate.transform.jit(fn)` | Compile the complete stochastic operation while preserving State effects. |

```python
import brainstate


class Dropout(brainstate.nn.Module):
    def __init__(self, drop_rate=0.5):
        super().__init__()
        self.drop_rate = drop_rate

    def __call__(self, x):
        if not brainstate.environ.get("fit", False):
            return x

        keep_prob = 1.0 - self.drop_rate
        mask = brainstate.random.bernoulli(keep_prob, x.shape)
        return x * mask / keep_prob
```

With `fit=False`, dropout returns `x` without drawing. With `fit=True`, each call draws a new mask and applies inverse-keep scaling. Noise drawn inside a Module call behaves the same way.

Open `references/brainstate/transformation-vmap-expansion.md` for mapped State axes, per-example randomness, and concrete `vmap` workflows.

## Exact replay and checkpoints

Snapshot the global key immediately before the sequence to replay. Restoring that snapshot reproduces the next draw.

| API | Description |
|---|---|
| `get_key()` | Return the current global key. |
| `set_key(key)` | Set the global key to an explicit snapshot. |
| `restore_key()` | Restore the default key to its previous state; do not substitute it for `set_key(saved_key)`. |
| `get_key_data()` | Return the global key as raw `uint32[2]` data for interchange. |

```python
import brainstate
import jax.numpy as jnp

saved_key = brainstate.random.get_key()
sample = brainstate.random.randn(3)

brainstate.random.set_key(saved_key)
replayed = brainstate.random.randn(3)

assert jnp.allclose(sample, replayed)
```

For exact continuation, save the model and RNG key together:

```python
checkpoint = {
    "model": model.state_dict(),
    "rng_key": brainstate.random.get_key(),
}

# Restore model state first, then restore the RNG before the next random call.
brainstate.random.set_key(checkpoint["rng_key"])
```

Capture the key after every random operation included in the completed run. An earlier key intentionally replays later draws.

## Parallel key preparation

Prepare independent keys before parallel work. Do not infer mapped key-consumption semantics from these utilities alone.

| API | Description |
|---|---|
| `split_key(n=...)` | Create new key or keys from the current seed; use `n` for parallel preparation. |
| `split_keys(...)` | Create multiple independent keys from the current seed. |
| `self_assign_multi_keys(...)` | Assign multiple keys to the global random state for parallel access. |

```python
keys = brainstate.random.split_key(n=4)
```

Open `references/brainstate/transformation-vmap-expansion.md` for mapped axes, mapped State, and concrete `vmap` mechanics.

## Generator helpers

| API | Description |
|---|---|
| `default_rng(...)` | Get the default random state or create one from a seed. |
| `clone_rng(...)` | Clone a random state or create a new one. |
| `seed_context(...)` | Temporarily change the seed and restore it on exit. |

The API table names `seed_context`; its example uses `local_seed`. Check the installed BrainState version before calling either name.

## Official sources

- [Random number generation](https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html)
- [`brainstate.random` API](https://brainx.chaobrain.com/brainstate/apis/random.html)
