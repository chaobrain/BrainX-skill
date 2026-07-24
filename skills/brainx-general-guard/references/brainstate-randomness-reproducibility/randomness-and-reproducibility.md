# BrainState Randomness and Reproducibility Parent Reference

Use this reference when BrainState code needs random numbers, reproducible examples/tests, stochastic masks/noise, seed control, random trials, stochastic state, dropout/noise, RNG under transforms, or a decision between the default BrainState RNG path and advanced key/stream APIs.

Usually reached from primary skills only when the task actually involves randomness. Do not load this for ordinary deterministic state, module, training, or BrainCell modeling tasks.

## Parent-reference boundary

This is this skill's only first-layer randomness reference. `skills/brainx-general-guard/SKILL.md` routes here. `skills/brainx-general-guard/references/brainstate-randomness-reproducibility/advanced-randomness.md` is its nested child and has no other selection route.

Use `brainstate.random` for BrainState RNG. Official anchor phrase: it "wraps JAX’s random number generation capabilities with a stateful interface that simplifies usage while maintaining reproducibility and performance."

## Core rules

- Default path: use `brainstate.random`; BrainState uses automatic key management, so do not manually create/split `jax.random.PRNGKey` unless explicit key control is requested.
- Reproducibility: call `brainstate.random.seed(seed)` once at the beginning of scripts, notebooks, examples, tests, or reproducible experiments.
- Global state: all random functions in `brainstate.random` use a global `DEFAULT` instance of `RandomState`.
- Common functions only: `random`, `rand`, `randn`, `normal`, `uniform`, `randint`, `bernoulli`, `choice`, `permutation`, `shuffle`.
- Stochastic modules: generate masks/noise inside the stochastic call using `brainstate.random`.
- Train-only randomness: gate stochastic behavior with training/fit mode; for dropout-style code, use `brainstate.environ.get('fit', False)` and return the input unchanged when not fitting.

## Canonical official workflow

```python
import brainstate

brainstate.random.seed(0)
x = brainstate.random.normal(0, 1, (100,))  # Keys handled automatically
```

## Stochastic-module pattern

```python
fit = brainstate.environ.get('fit', False)
if not fit:
    return x

keep_prob = 1.0 - drop_rate
mask = brainstate.random.bernoulli(keep_prob, x.shape)
return x * mask / keep_prob
```

For noisy stochastic behavior, generate noise at call time with `brainstate.random.normal(..., shape)`.

## Open the nested child `skills/brainx-general-guard/references/brainstate-randomness-reproducibility/advanced-randomness.md` for

- manual keys: `get_key`, `set_key`, `get_key_data`, `restore_key`, `split_key`, `split_keys`, `self_assign_multi_keys`
- custom `RandomState` instances or independent streams
- checkpoint RNG save/restore
- temporary seed contexts, `default_rng`, or `clone_rng`
- full probability distribution catalog
- PyTorch-compatible `rand_like`, `randint_like`, `randn_like`
- distribution visualization, mini-batch sampling, full dropout/noisy-layer examples
- RNG combined with transformations, parallelism, or brain-dynamics workflows

## Mirror Source URLs

- https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html
- https://brainx.chaobrain.com/brainstate/apis/random.html
- https://brainx.chaobrain.com/brainstate/tutorials/transformations/index.html
- https://brainx.chaobrain.com/brainstate/tutorials/core/index.html
- https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/index.html
