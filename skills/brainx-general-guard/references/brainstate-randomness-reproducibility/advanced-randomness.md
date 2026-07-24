# Advanced BrainState RNG reference

This is a nested child with exactly one selector: `skills/brainx-general-guard/references/brainstate-randomness-reproducibility/randomness-and-reproducibility.md`. Do not route here directly from a skill, workspace router, index, transform reference, or dynamics reference. Keep the default path compact: seed once, call `brainstate.random`, and let BrainState manage keys automatically.

Official anchor phrase: `brainstate.random` provides "a comprehensive set of random number generation functions and utilities for neural network simulations and scientific computing." It includes standard random distributions, random state management with automatic key splitting, seed management utilities, and a NumPy-compatible API.

## 1. Manual key management

Official boundary: "For advanced use cases, you can directly access and manipulate keys."

Use manual keys only when the task explicitly needs current-key inspection, exact RNG restoration, or parallel key splitting.

```python
# Get current key
current_key = brainstate.random.get_key()
print(f"Current key: {current_key}")

# Split key for parallel operations
keys = brainstate.random.split_key(n=4)
print(f"\n Split into {len(keys)} keys:")
for i, key in enumerate(keys):
    print(f"  Key {i}: {key}")
```

```python
# Save and restore state
saved_key = brainstate.random.get_key()
v1 = brainstate.random.randn(3)
# Generate more numbers
v2 = brainstate.random.randn(3)

# Restore state
brainstate.random.set_key(saved_key)
v3 = brainstate.random.randn(3)

print(f"v1: {v1}")
print(f"v2: {v2}")
print(f"v3 (restored): {v3}")
print(f"\n v1 == v3? {jnp.allclose(v1, v3)}")
```

Related API utilities:

- `get_key`: get the current global random key.
- `set_key`: set a new random key for the global random state.
- `get_key_data`: get the current global random key as raw `uint32[2]` data.
- `restore_key`: restore the default random key to its previous state.
- `split_key`: create new random key(s) from the current seed.
- `split_keys`: create multiple independent random keys from the current seed.
- `self_assign_multi_keys`: assign multiple keys to the global random state for parallel access.

## 2. Custom `RandomState` instances / independent streams

Official boundary: "For advanced use cases, you can create custom `RandomState` instances with independent random streams."

```python
# Create custom random generators
rng1 = brainstate.random.RandomState(42)
rng2 = brainstate.random.RandomState(123)

print("RNG 1:")
print(rng1)

# Generate independent random sequences
samples1 = rng1.randn(5)
samples2 = rng2.randn(5)
print(f"Samples from RNG1: {samples1}")
print(f"Samples from RNG2: {samples2}")
print(f"\n Are they different? {not jnp.allclose(samples1, samples2)}")

# Re-seed custom RNG
rng1.seed(999)
samples3 = rng1.randn(5)
print(f"\n After re-seeding RNG1 to 999: {samples3}")
```

Best-practice examples from the RNG tutorial:

```python
# For data augmentation
aug_rng = brainstate.random.RandomState(seed=123)

# For model initialization
init_rng = brainstate.random.RandomState(seed=456)
```

## 3. Checkpoint RNG save/restore

Use this only when exact continuation of a run matters.

```python
# Save state
checkpoint = {
    'model': model.state_dict(),
    'rng_key': brainstate.random.get_key()
}

# Restore state
brainstate.random.set_key(checkpoint['rng_key'])
```

## 4. Temporary seed contexts and RNG cloning

Official API phrases:

- `seed_context`: "Context manager for temporary random seed changes with automatic restoration."
- `default_rng`: "Get the default random state or create a new one with specified seed."
- `clone_rng`: "Create a clone of the random state or a new random state."

The API page’s seed-management example shows a temporary seed context as:

```python
>>> import brainstate.random as bsr
>>>
>>> # Set a global seed
>>> bsr.seed(42)
>>>
>>> # Get current seed/key
>>> key = bsr.get_key()
>>>
>>> # Split the key for parallel operations
>>> keys = bsr.split_key(n=4)
>>>
>>> # Use context manager for temporary seed
>>> with bsr.local_seed(123):
...     x = bsr.normal(0, 1, (5,))  # Uses seed 123
>>> y = bsr.normal(0, 1, (5,))  # Uses original seed
```

## 5. Full probability distribution catalog

The compact skill lists only everyday functions. Use this catalog when the user asks for specific distributions or API coverage.

### Basic random sampling

- `rand`: random values in a given shape.
- `randn`: sample(s) from the standard normal distribution.
- `random`, `random_sample`, `ranf`, `sample`: random floats in `[0.0, 1.0)`.
- `randint`: random integers from low inclusive to high exclusive.
- `random_integers`: random integers between low and high inclusive.

### Array-like generation: PyTorch compatibility

- `rand_like`: similar to `rand_like` in torch.
- `randint_like`: similar to `randint_like` in torch.
- `randn_like`: similar to `randn_like` in torch.

### Array manipulation

- `choice`: generates a random sample from a given 1-D array.
- `permutation`: randomly permute a sequence, or return a permuted range.
- `shuffle`: return a randomly shuffled copy of an array along an axis.

### Continuous distributions

- `beta`: draw samples from a Beta distribution.
- `exponential`, `standard_exponential`: draw samples from an exponential distribution.
- `gamma`, `standard_gamma`: draw samples from a Gamma distribution.
- `gumbel`: draw samples from a Gumbel distribution.
- `laplace`: draw samples from the Laplace or double exponential distribution.
- `logistic`: draw samples from a logistic distribution.
- `normal`: draw random samples from a normal/Gaussian distribution.
- `pareto`: draw samples from a Pareto II or Lomax distribution.
- `standard_cauchy`: draw samples from a standard Cauchy distribution.
- `standard_normal`: draw samples from a standard Normal distribution.
- `standard_t`: draw samples from a standard Student’s t distribution.
- `uniform`: draw samples from a uniform distribution.
- `truncated_normal`: sample truncated standard normal random values.
- `lognormal`, `power`, `rayleigh`, `triangular`, `vonmises`, `wald`, `weibull`, `weibull_min`, `maxwell`, `t`, `loggamma`.

### Discrete distributions

- `bernoulli`: sample Bernoulli random values with given shape and mean.
- `binomial`: draw samples from a binomial distribution.
- `categorical`: sample random values from categorical distributions.
- `geometric`, `hypergeometric`, `logseries`, `multinomial`, `negative_binomial`, `poisson`, `zipf`.

### Special distributions

- `chisquare`: draw samples from a chi-square distribution.
- `dirichlet`: draw samples from the Dirichlet distribution.
- `f`: draw samples from an F distribution.
- `multivariate_normal`: draw random samples from a multivariate normal distribution.
- `noncentral_chisquare`, `noncentral_f`, `orthogonal`.

## 6. Distribution visualization

Official boundary: visualization belongs in reference/tutorial material. The RNG tutorial introduces it with: "Let’s visualize several distributions."

Use the official tutorial section for histogram examples rather than placing plotting code in the compact skill.

## 7. Full practical examples from the RNG tutorial

### Mini-batch sampling

```python
def create_mini_batches(X, y, batch_size=32, shuffle=True):
    """Create mini-batches for training."""
    n_samples = len(X)

    # Shuffle indices
    if shuffle:
        indices = brainstate.random.permutation(n_samples)
    else:
        indices = jnp.arange(n_samples)

    # Create batches
    batches = []
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        batches.append((X[batch_indices], y[batch_indices]))

    return batches

# Generate dummy dataset
brainstate.random.seed(0)
X = brainstate.random.randn(100, 10)  # 100 samples, 10 features
y = brainstate.random.randint(0, 2, 100)  # Binary labels
# Create mini-batches
batches = create_mini_batches(X, y, batch_size=32)
print(f"Created {len(batches)} mini-batches")
print(f"First batch shape: X={batches[0][0].shape}, y={batches[0][1].shape}")
```

### Dropout layer

```python
class Dropout(brainstate.nn.Module):
    """Dropout layer with random masking."""

    def __init__(self, drop_rate=0.5):
        super().__init__()
        self.drop_rate = drop_rate

    def __call__(self, x):
        fit = brainstate.environ.get('fit', False)
        if not fit:
            return x
        # Generate random mask
        keep_prob = 1.0 - self.drop_rate
        mask = brainstate.random.bernoulli(keep_prob, x.shape)

        # Apply dropout and scale
        return x * mask / keep_prob

# Test dropout
brainstate.random.seed(42)
dropout = Dropout(drop_rate=0.3)

x = jnp.ones(10)
y = dropout(x)

print(f"Input:  {x}")
print(f"Output: {y}")
print(f"Zeros:  {jnp.sum(y == 0)}/10")
```

### Noisy neural network layer

```python
class NoisyLayer(brainstate.nn.Module):
    """Linear layer with Gaussian noise."""

    def __init__(self, d_in, d_out, noise_std=0.1):
        super().__init__()
        self.noise_std = noise_std

        # Parameters with noise
        self.w = brainstate.ParamState(brainstate.random.randn(d_in, d_out) * 0.1)
        self.b = brainstate.ParamState(jnp.zeros(d_out))
    def __call__(self, x):
        # Add weight noise
        w_noisy = self.w.value + brainstate.random.normal(0, self.noise_std, self.w.value.shape)

        # Forward pass
        return x @ w_noisy + self.b.value

# Create and test
brainstate.random.seed(0)
layer = NoisyLayer(5, 3, noise_std=0.01)

x = jnp.ones(5)
y1 = layer(x)
y2 = layer(x)  # Different due to noise
print(f"Output 1: {y1}")
print(f"Output 2: {y2}")
print(f"Difference: {y2 - y1}")
```

## 8. Reference routing for broader BrainState workflows

- Core tutorials: "define states, compose modules and layers, apply the essential transformations, and train a model."
- Transformations tutorials: "state-aware program transformations, from the essential `jit` / `grad` / `vmap` through advanced batching, error handling, debugging, and intermediate-representation tooling."
- Brain Dynamics tutorials: "neuron dynamics and numerical integration, synaptic delays, event-driven operators, and spiking neural networks."

## Official BrainX sources

- https://brainx.chaobrain.com/brainstate/tutorials/core/08_randomness.html
- https://brainx.chaobrain.com/brainstate/apis/random.html
- https://brainx.chaobrain.com/brainstate/tutorials/transformations/index.html
- https://brainx.chaobrain.com/brainstate/tutorials/core/index.html
- https://brainx.chaobrain.com/brainstate/tutorials/brain_dynamics/index.html
