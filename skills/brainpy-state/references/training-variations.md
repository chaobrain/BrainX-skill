# Training variations

Use this reference after the forward simulation produces the intended shapes and dynamics. Keep the simulation graph unchanged unless the training objective requires a surrogate, trainable parameter, readout, or memory policy.

## Canonical training lifecycle

1. Put a surrogate `spk_fun` on every spiking layer crossed by the loss gradient.
2. Construct the network and select `params = net.states(brainstate.ParamState)`.
3. Register exactly `params` with the optimizer.
4. Define a loss that unrolls model time with `for_loop` or `scan` and reduces time according to the target.
5. In a `brainstate.transform.jit` train step, reset dynamical State with the correct `batch_size`, differentiate the loss with respect to `params`, and apply the gradients.
6. Keep the outer epoch/data loop in Python; keep neural time inside the transformed loop.

## Surrogate APIs

Pass either an object-style surrogate instance or a lowercase functional surrogate through `spk_fun=`. Both keep the forward spike discrete and replace only the backward derivative.

| Family | Class API | Functional API | Use |
|---|---|---|---|
| Sigmoid | `Sigmoid` | `sigmoid` | Use a smooth bounded derivative. |
| Piecewise | `PiecewiseQuadratic`, `PiecewiseExp`, `PiecewiseLeakyRelu` | `piecewise_quadratic`, `piecewise_exp`, `piecewise_leaky_relu` | Use simple compact or piecewise gradient shapes. |
| Smooth tails | `SoftSign`, `Arctan`, `NonzeroSignLog`, `ERF` | `soft_sign`, `arctan`, `nonzero_sign_log`, `erf` | Use when tail behavior or smoothness changes optimization stability. |
| ReLU-based | `LeakyRelu`, `LogTailedRelu`, `ReluGrad` | `leaky_relu`, `log_tailed_relu`, `relu_grad` | Use `ReluGrad()` as the canonical fast default; select the others for different tail/leak behavior. |
| Gaussian | `GaussianGrad`, `MultiGaussianGrad` | `gaussian_grad`, `multi_gaussian_grad` | Use localized or multi-peak backward gradients for spike-timing-sensitive behavior. |
| Other published forms | `SquarewaveFourierSeries`, `S2NN`, `QPseudoSpike`, `InvSquareGrad`, `SlayerGrad` | `squarewave_fourier_series`, `s2nn`, `q_pseudo_spike`, `inv_square_grad`, `slayer_grad` | Use only when the task or reproduced method requires the corresponding gradient form. |

Do not change several surrogate families and optimizer settings simultaneously when diagnosing training. Start with `ReluGrad()`, validate nonzero finite gradients, then vary one gradient shape or parameter at a time.

## Readout choices

| Need | Use | Loss input |
|---|---|---|
| Explicit trainable low-pass decoder | `brainpy.state.LeakyRateReadout` | Its per-step continuous output, reduced over time as the task specifies |
| Separate trainable mapping and filter | `brainstate.nn.Linear` followed by `brainpy.state.Expon` | The filtered per-step logits |
| Spike-count/rate classification without a dynamical head | Reduce spikes or logits with `mean` or `sum` | The time-reduced tensor |
| Decision at the end of a sequence | Select the final valid step | Final-step logits, with padding/masking handled explicitly |

Do not default to time averaging when the label depends on precise timing or the final state.

## Batching

Use `brainstate.nn.init_all_states(net, batch_size=B)` when built-in dynamics should allocate State shaped `[B, ...feature_shape]` and inputs are time-major `[T, B, ...]`. `for_loop` slices the leading time axis and passes one `[B, ...]` input per step.

Use `brainstate.transform.vmap` instead when the single-example computation must be mapped explicitly or model State must be mapped/shared with controlled axes. Open the BrainState `skills/brainstate/references/brainstate/transformation-vmap-expansion.md` reference for `state_in_axes`, `state_out_axes`, ensembles, or mapped randomness.

## Rollout and memory choices

| Condition | API | Consequence |
|---|---|---|
| Model State carries all recurrent variables | `brainstate.transform.for_loop` | Simplest BPTT path; returned monitors are stacked over time. |
| The loss needs explicit carry outside Module State | `brainstate.transform.scan` | Threads `carry` and returns `(carry, ys)`. |
| Long `for_loop` BPTT exceeds memory | `checkpointed_for_loop(..., base=...)` | Recomputes intervening activations during backward; lower memory, more compute. |
| Long `scan` BPTT exceeds memory | `checkpointed_scan(...)` | Applies the same rematerialization policy to explicit-carry scans. |

Use plain loops first. Tune `base` only after confirming memory is the binding constraint; larger checkpoint spacing reduces stored activations and increases recomputation.

## Gradient checks

- Confirm the loss consumes outputs from the same rollout whose parameters are differentiated.
- Confirm `params` is nonempty and contains only intended `ParamState` leaves.
- Reset dynamical State before each independent batch; otherwise gradients depend on State leaked from a previous batch.
- Check loss and representative gradient leaves for finite values before long training.
- Verify output shape before applying a temporal reduction or classification loss.
- Preserve units in trainable physical weights and biases by using unit-aware initializers.

Use `references/brainstate-dynamics/scripts/training-snn.py` for the full runnable pattern. Open `references/braintools-optimizer.md` only after the BrainPy loss, State reset, gradient selection, and transformed rollout are correct.

## Official sources

- `https://brainx.chaobrain.com/brainpy-state/concepts/differentiability.html`
- `https://brainx.chaobrain.com/brainpy-state/brainpy-style/tutorials/04-train-an-snn.html`
- `https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/train-surrogate-gradients.html`
- `https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/train-readouts.html`
- `https://brainx.chaobrain.com/brainpy-state/brainpy-style/howto/train-long-rollouts-checkpoint.html`
- `https://brainx.chaobrain.com/braintools/apis/surrogate.html`
