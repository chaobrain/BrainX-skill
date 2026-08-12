# HORN task training

Use this reference when training a HORN network over minibatched sequence-task examples and evaluating a held-out metric. Do not use `Fitter`: `Fitter` targets one fixed prediction and target, while task training owns data batches, epochs, hidden-State reset, and validation.

## Choose the HORN abstraction

| API | Description |
|---|---|
| `brainmass.HORNStep(...)` | Use when composing one explicit harmonic-oscillator recurrent update. |
| `brainmass.HORNSeqLayer(...)` | Use for one HORN layer that processes sequential input. |
| `brainmass.HORNSeqNetwork(n_input, n_hidden, n_output, alpha=..., omega=...)` | Use for the documented task-training path; it maps a time-major sequence to final output logits. |
| `brainmass.datasets.delayed_match_task(...)` | Use for the bundled delayed match-to-sample benchmark with `(sample, time, symbol)` inputs and binary targets. |

Use the constructor names from the installed release. Older HORN API material may show `in_size`, `hidden_sizes`, and `out_size`; the current official task workflow uses `n_input`, `n_hidden`, and `n_output`.

## Prepare task data

```python
import brainmass
import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

brainstate.environ.set(dt=1.0 * u.ms)
brainstate.random.seed(0)

inputs_np, targets_np = brainmass.datasets.delayed_match_task(
    n_samples=320,
    seq_len=8,
    n_symbols=2,
    seed=0,
)
inputs = jnp.asarray(inputs_np, dtype=jnp.float32)
targets = jnp.asarray(targets_np, dtype=jnp.int32)

n_train = 256
x_train, y_train = inputs[:n_train], targets[:n_train]
x_test, y_test = inputs[n_train:], targets[n_train:]

assert inputs.shape == (320, 8, 2)
assert targets.shape == (320,)
```

Create train and held-out splits before optimization. Do not report the training metric as generalization.

Open `braintools/cogtask.md` when a pre-built or custom phase-structured
cognitive paradigm should generate the trials. Open
`braintools/data-preprocessing.md` only when custom inputs must be encoded before
they enter HORN; the ordinary feature sequences above need no spike encoder.

## Build the classifier and reset hidden State

The network consumes `(time, batch, feature)`. Reset every HORN layer to the current batch shape before each forward pass so one batch cannot leak memory into the next.

```python
network = brainmass.HORNSeqNetwork(
    n_input=inputs.shape[2],
    n_hidden=64,
    n_output=2,
    alpha=0.2,
    omega=2.0 * np.pi / 28.0,
)
brainstate.nn.init_all_states(network)

def reset_hidden(batch_size):
    for layer in network.layers:
        shape = (batch_size,) + tuple(layer.horn.in_size)
        layer.horn.x.value = jnp.zeros(shape)
        layer.horn.y.value = jnp.zeros(shape)

def logits(batch_inputs):
    reset_hidden(batch_inputs.shape[0])
    time_major = jnp.transpose(batch_inputs, (1, 0, 2))
    return network(time_major)

weights = network.states(brainstate.ParamState)
```

**Invariant:** reset hidden State for training and evaluation. A held-out evaluation that begins from training State is invalid.

Open `braintools/parameter-initializer.md` when HORN weights or model parameters
need a reusable variance-scaling, orthogonal, bounded, or unit-aware
initialization policy.

## Define and compile the training step

Register the same `ParamState` collection used as the gradient target. The current official HORN workflow applies gradients with `optimizer.step(grads)`.

```python
optimizer = braintools.optim.Adam(lr=3e-2)
optimizer.register_trainable_weights(weights)

def loss_and_accuracy(batch_inputs, batch_targets):
    output = logits(batch_inputs)
    log_probability = jax.nn.log_softmax(output, axis=-1)
    loss = -jnp.mean(
        log_probability[
            jnp.arange(batch_targets.shape[0]),
            batch_targets,
        ]
    )
    accuracy = jnp.mean(
        jnp.argmax(output, axis=-1) == batch_targets
    )
    return loss, accuracy

@brainstate.transform.jit
def train_step(batch_inputs, batch_targets):
    gradient = brainstate.transform.grad(
        lambda: loss_and_accuracy(batch_inputs, batch_targets),
        weights,
        has_aux=True,
        return_value=True,
    )
    gradients, loss, accuracy = gradient()
    optimizer.step(gradients)
    return loss, accuracy

@brainstate.transform.jit
def evaluate(batch_inputs, batch_targets):
    return loss_and_accuracy(batch_inputs, batch_targets)
```

If the installed Braintools release exposes `optimizer.update(grads)` instead
of `step(grads)`, use the installed optimizer contract consistently; do not mix
methods in one loop. Open `braintools/optimizer.md` for optimizer and scheduler
selection, and `braintools/metric.md` for supervised loss, reduction, and
held-out metric selection.

## Run epochs and evaluate held-out data

Use a reproducible permutation policy and keep incomplete-batch handling explicit.

```python
n_epoch = 20
batch_size = 32
history = []

for epoch in range(n_epoch):
    permutation = np.random.RandomState(epoch).permutation(n_train)
    train_losses = []
    train_accuracies = []

    for start in range(0, n_train, batch_size):
        index = permutation[start:start + batch_size]
        if len(index) < batch_size:
            continue
        loss, accuracy = train_step(
            x_train[index],
            y_train[index],
        )
        train_losses.append(float(loss))
        train_accuracies.append(float(accuracy))

    _, test_accuracy = evaluate(x_test, y_test)
    history.append({
        "epoch": epoch,
        "train_loss": float(np.mean(train_losses)),
        "train_accuracy": float(np.mean(train_accuracies)),
        "test_accuracy": float(test_accuracy),
    })
```

Validate that chance-level accuracy, label balance, loss reduction, and held-out improvement are all consistent with the task. A final accuracy alone does not reveal leakage or class imbalance.

Open `scripts/horn-cognitive-task-training.py` when the task needs the complete delayed-match dataset, HORN network, compiled training loop, held-out evaluation, and learning curves.

## Choose task-training variations

- Use another `brainmass.datasets` generator or a validated external loader for a different task.
- Keep sequence inputs time-major inside the network even when storage is sample-major.
- Add a learning-rate scheduler only after the fixed-rate baseline trains correctly.
- Clip gradients only after confirming large or non-finite gradients.
- Tune oscillator `alpha`, `omega`, damping, and hidden size one decision at a time.
- Use direct Braintools metrics when the held-out criterion is not accuracy.
- Open `brainstate/parameter-constraints-regularization.md` when a directly
  trained oscillator parameter needs a valid domain, penalty, or prior.
- Open `braintools/surrogate.md` only when a custom path introduces a hard
  threshold or spike; canonical HORN training does not need a surrogate.

## Common failures

- Sending task training through `Fitter`.
- Omitting hidden-State reset between batches or between training and evaluation.
- Feeding `(batch, time, feature)` directly to a time-major HORN network.
- Collecting all `State` instead of only `ParamState` as trainable.
- Re-registering optimizer weights inside every compiled step.
- Using raw `jax.grad` or `jax.jit` around State-aware code.
- Reporting only training accuracy.
- Changing optimizer, loss, HORN dynamics, and reset policy simultaneously.

## Official sources

- `https://brainx.chaobrain.com/brainmass/reference/horn.html`
- `https://brainx.chaobrain.com/brainmass/tutorials/08_training_on_tasks.html`
- `https://brainx.chaobrain.com/brainmass/gallery/case_studies/horn_cognitive_task.html`
