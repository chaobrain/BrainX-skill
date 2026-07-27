# Training variations

Use this reference after the forward simulation produces the intended dynamics and tensor shapes. Change the simulation graph only when the loss requires a surrogate derivative, trainable decoder, different State layout, explicit carry, or lower-memory reverse pass.

## Selection map

| Decision | Default | Change when |
|---|---|---|
| Spike derivative | `braintools.surrogate.ReluGrad()` | Gradient support, smoothness, or a reproduced method requires another surrogate. |
| Loss-facing output | Time-reduced logits from the existing graph | The task requires a trainable dynamical readout, final-State decision, or timing-sensitive loss. |
| Batch representation | `init_all_states(net, batch_size=B)` with time-major inputs | Single-example computation must be mapped explicitly or State axes need controlled sharing. |
| Rollout | `brainstate.transform.for_loop` | The loss needs explicit non-Module carry. |
| Reverse-pass memory | Plain differentiable loop | Long-rollout BPTT exhausts memory. |
| Optimizer | `braintools.optim.Adam` | A validated optimization requirement calls for another interface or schedule. |

Keep model time inside a BrainState transform. Keep the outer epoch and data-loader loop in Python, where each iteration calls the compiled train step.

## Surrogate gradients

A surrogate preserves the hard forward spike and replaces only its unusable threshold derivative during backpropagation. Put one on every spiking layer crossed by the loss gradient.

Use a class instance for a configured, reusable neuron `spk_fun=`. Use the lowercase functional form when applying a surrogate directly to an array; it takes `x` as its first argument.

### Compact and smooth baselines

Use this family for the initial training baseline; choose compact support for efficiency or a smooth bounded derivative when transition smoothness matters.

| API | Use when |
|---|---|
| `braintools.surrogate.ReluGrad(alpha=0.3, width=1.0)` | Use as the canonical configured neuron surrogate; `width` controls where its triangular gradient is nonzero. |
| `braintools.surrogate.relu_grad(x, alpha=0.3, width=1.0)` | Use to apply the same triangular surrogate directly to an array. |
| `braintools.surrogate.Sigmoid(alpha=4.0)` | Use as a configured neuron surrogate when a smooth bounded derivative is preferred. |
| `braintools.surrogate.sigmoid(x, alpha=4.0)` | Use to apply the same sigmoid-shaped surrogate directly to an array. |

### Piecewise surrogates

Use this family when the training method specifies a compact piecewise derivative rather than the baseline shape.

| API | Use when |
|---|---|
| `braintools.surrogate.PiecewiseQuadratic(alpha=1.0)` | Use as a configured neuron surrogate with a piecewise-quadratic derivative. |
| `braintools.surrogate.piecewise_quadratic(x, alpha=1.0)` | Use to apply the piecewise-quadratic surrogate directly to an array. |
| `braintools.surrogate.PiecewiseExp(alpha=1.0)` | Use as a configured neuron surrogate with a piecewise-exponential derivative. |
| `braintools.surrogate.piecewise_exp(x, alpha=1.0)` | Use to apply the piecewise-exponential surrogate directly to an array. |

### Localized gradient surrogates

Use this family when localized or multiple gradient peaks are part of the intended temporal-credit design.

| API | Use when |
|---|---|
| `braintools.surrogate.GaussianGrad(sigma=0.5, alpha=0.5)` | Use as a configured neuron surrogate with one Gaussian-shaped gradient peak. |
| `braintools.surrogate.gaussian_grad(x, sigma=0.5, alpha=0.5)` | Use to apply the single-Gaussian surrogate directly to an array. |
| `braintools.surrogate.MultiGaussianGrad(h=0.15, s=6.0, sigma=0.5, scale=0.5)` | Use as a configured neuron surrogate when multiple Gaussian components are required. |
| `braintools.surrogate.multi_gaussian_grad(x, h=0.15, s=6.0, sigma=0.5, scale=0.5)` | Use to apply the multi-Gaussian surrogate directly to an array. |

### Specialized surrogate formulations

Use this family only when reproducing or deliberately testing the named surrogate formulation.

| API | Use when |
|---|---|
| `braintools.surrogate.S2NN(alpha=4.0, beta=1.0, epsilon=1e-8)` | Use as a configured neuron surrogate for the S2NN formulation. |
| `braintools.surrogate.s2nn(x, alpha=4.0, beta=1.0, epsilon=1e-8)` | Use to apply the S2NN formulation directly to an array. |
| `braintools.surrogate.QPseudoSpike(alpha=2.0)` | Use as a configured neuron surrogate for the q-pseudo-spike formulation. |
| `braintools.surrogate.q_pseudo_spike(x, alpha=2.0)` | Use to apply the q-pseudo-spike formulation directly to an array. |
| `braintools.surrogate.SlayerGrad(alpha=1.0)` | Use as a configured neuron surrogate for the SLAYER gradient formulation. |
| `braintools.surrogate.slayer_grad(x, alpha=1.0)` | Use to apply the SLAYER gradient formulation directly to an array. |
| `braintools.surrogate.InvSquareGrad(alpha=100.0)` | Use as a configured neuron surrogate for an inverse-square gradient profile. |
| `braintools.surrogate.inv_square_grad(x, alpha=100.0)` | Use to apply the inverse-square surrogate directly to an array. |

```python
hidden = brainpy.state.LIF(
    n_hidden,
    tau=20.0 * u.ms,
    V_rest=0.0 * u.mV,
    V_reset=0.0 * u.mV,
    V_th=1.0 * u.mV,
    spk_fun=braintools.surrogate.ReluGrad(
        alpha=0.3,
        width=1.0,
    ),
)
```

Start with one surrogate and confirm finite, nonzero representative gradients. Change one surrogate family or parameter at a time; do not diagnose the surrogate while also changing the optimizer and temporal loss.

Use the official `braintools.surrogate` API page for the complete class/function catalog and exact parameters. Do not infer class names from older BrainPy surrogate APIs.

## Readout and temporal objective

The readout determines which trajectory information reaches the loss.

### Stateful trainable readouts

Use this family when the model must learn a continuous decoder, with either integrated or independently replaceable temporal filtering.

| API | Use when |
|---|---|
| `brainpy.state.LeakyRateReadout(in_size, out_size, tau=..., w_init=..., name=None)` | Use for one trainable mapping with built-in leaky low-pass dynamics; reset its State with the rest of the network. |
| `brainstate.nn.Linear(in_size, out_size, w_init=..., b_init=..., ...)` | Use as the mapping stage when decoder weights and temporal filtering must remain separate. |
| `brainpy.state.Expon(in_size, name=None, tau=..., g_initializer=...)` | Use after a separate mapping when its filtered State and lifecycle must remain independently replaceable. |

### Explicit temporal objectives

Use this family when the loss should consume a trajectory statistic or a selected time step rather than a dynamical readout State.

| API or expression | Use when |
|---|---|
| `u.math.sum(outputs, axis=0)` | Use for spike-count or accumulated-evidence targets that are invariant to event timing within the window. |
| `u.math.mean(outputs, axis=0)` | Use for average-rate or time-averaged-logit targets that are invariant to event timing within the window. |
| `outputs[-1]` | Use for a sequence-end decision after handling padding and variable-length masks. |

For latency or precise-timing targets, keep the returned trajectory unreduced and apply the task-specific windowed or per-step loss without averaging away event time.

Run the stateful rollout once per loss evaluation. Derive loss and auxiliary metrics from that same returned trajectory; a second unguarded model call advances State again.

## Batch State and mapped execution

Use built-in batch-aware State allocation for the canonical SNN path:

```python
# inputs: [time, batch, ...features]
brainstate.nn.init_all_states(net, batch_size=batch_size)
outputs = brainstate.transform.for_loop(net.update, inputs)

assert outputs.shape[:2] == inputs.shape[:2]
```

The leading input axis is time; `for_loop` slices one `[batch, ...features]` value per step. Built-in dynamics allocate State as `[batch, ...state_shape]`.

Use `brainstate.transform.vmap` instead only when mapping a single-example function or explicitly choosing which State axes are mapped, shared, or returned. Open `skills/brainstate/references/brainstate/transformation-vmap-expansion.md` for `state_in_axes`, `state_out_axes`, mapped randomness, and ensembles.

Reset all dynamical State before each independent batch. Do not reset between time steps of the same sequence.

## Rollout and memory

Choose the plain State-aware loop first, then switch to the matching checkpointed family only when reverse-mode activation storage becomes the binding constraint.

### Plain differentiable rollouts

Use this family while the complete reverse pass fits in memory; choose by whether recurrent values live entirely in Module State or require explicit carry.

| API | Use when |
|---|---|
| `brainstate.transform.for_loop(f, *xs, length=None, reverse=False, unroll=1, pbar=None)` | Use when Module State carries recurrent variables; it slices leading input axes and stacks per-step outputs. |
| `brainstate.transform.scan(f, init, xs, length=None, reverse=False, unroll=1, pbar=None)` | Use when the body must thread explicit carry through `f(carry, x) -> (carry, y)`; it returns final carry and stacked outputs. |

### Checkpointed differentiable rollouts

Use this family only when a long differentiable rollout exhausts memory and recomputation is acceptable.

| API | Use when |
|---|---|
| `brainstate.transform.checkpointed_for_loop(f, *xs, length=None, base=16, pbar=None)` | Use as the State-carried checkpointed form; it preserves outputs while rematerializing activations during backward. |
| `brainstate.transform.checkpointed_scan(f, init, xs, length=None, base=16, pbar=None)` | Use as the explicit-carry checkpointed form; it returns final carry and stacked outputs while rematerializing activations. |

Use a checkpointed loop as a drop-in replacement only inside the differentiable rollout:

```python
outputs = brainstate.transform.checkpointed_for_loop(
    net.update,
    inputs,
    base=16,
)
```

Plain BPTT stores every step's activations. Checkpointed loops trade recomputation for lower peak memory; for rollout length `T`, the documented rough memory scale is `base + T / base`. Use plain loops first and tune `base` only after memory is the binding constraint.

Open `skills/brainstate/references/brainstate/brainstate-control-flow-patterns.md` when the loop also needs branches, early stopping, multiple explicit carries, or more detailed checkpoint selection.

## Diagnose training failures

Check the training contract in this order:

1. Run an untrained forward rollout and verify time, batch, feature, and class axes before any reduction.
2. Confirm `params = net.states(brainstate.ParamState)` is nonempty and excludes dynamical State.
3. Reset State, evaluate one loss, and confirm it is scalar and finite.
4. Differentiate the same loss with respect to `params`; inspect representative leaves for matching shapes and finite values.
5. Confirm at least one intended gradient leaf is nonzero under a nontrivial batch.
6. Register exactly `params` with the optimizer and apply one update.
7. Reset and re-evaluate the same batch; confirm the parameter and loss changes are plausible.

Preserve units in physical trainable weights and biases with unit-aware initializers. Do not strip units inside the model merely to satisfy a loss; convert only at a dimensionless or external-library boundary.

## Routing and official sources

Use `references/brainstate-dynamics/scripts/training-snn.py` for the compact runnable surrogate-gradient SNN pattern. Use `references/scripts/201_surrogate_grad_lif_fashion_mnist.py` for the complete real-data Fashion-MNIST workflow. Open `references/braintools-optimizer.md` only after loss, reset, gradient selection, and rollout are correct.

Official sources:

- `https://brainx.chaobrain.com/brainpy-state/concepts/differentiability.html`
- `https://brainx.chaobrain.com/brainpy-state/brainpy-style/tutorials/04-train-an-snn.html`
- `https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/train-surrogate-gradients.html`
- `https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/train-readouts.html`
- `https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/train-long-rollouts-checkpoint.html`
- `https://brainx.chaobrain.com/braintools/apis/surrogate.html`
